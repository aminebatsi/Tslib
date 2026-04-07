import torch
from torch import nn
from mamba_ssm import Mamba
from layers.Embed import PatchEmbedding


# ----------------------------
# Utils
# ----------------------------
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


# ----------------------------
# Volatility-aware SE (finance)
# ----------------------------
class VolatilityAwareScaleGatingSE(nn.Module):
    """
    Finance/crypto volatility-aware SE gating.

    Input:  z  [B, V, D, P]
    Gate:   g  [B, V, D, 1]  (channel-wise)
    Output: z * g

    Signals (computed along patch axis P):
      - mean(z)
      - RV   (realized volatility proxy)
      - JUMP (jump proxy using bipower variation idea)
      - TAIL (extreme-move proxy)
    """
    def __init__(self, d_model, reduction=4, gate_dropout=0.1, eps=1e-6):
        super().__init__()
        self.eps = eps
        hidden = max(8, d_model // reduction)

        self.fc1 = nn.Linear(4 * d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(gate_dropout)
        self.fc2 = nn.Linear(hidden, d_model)  # channel-wise gate

    def forward(self, z):
        # z: [B, V, D, P]
        mean_z = z.mean(dim=-1)  # [B,V,D]

        # returns proxy along patches
        r = z[..., 1:] - z[..., :-1]  # [B,V,D,P-1]
        abs_r = r.abs()
        r2 = r * r

        # realized vol
        rv = torch.sqrt(r2.mean(dim=-1) + self.eps)  # [B,V,D]

        # bipower variation proxy
        if r.shape[-1] >= 2:
            bv = (abs_r[..., 1:] * abs_r[..., :-1]).mean(dim=-1)  # [B,V,D]
        else:
            bv = torch.zeros_like(rv)

        # jump proxy
        jump = torch.clamp((rv * rv) - bv, min=0.0)  # [B,V,D]

        # tail proxy
        tail = abs_r.amax(dim=-1) / (abs_r.mean(dim=-1) + self.eps)  # [B,V,D]

        s = torch.cat([mean_z, rv, jump, tail], dim=-1)  # [B,V,4D]

        g = self.fc2(self.drop(self.act(self.fc1(s))))  # [B,V,D]
        g = torch.sigmoid(g).unsqueeze(-1)              # [B,V,D,1]
        return z * g


# ----------------------------
# Mamba blocks
# ----------------------------
class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaEncoder(nn.Module):
    def __init__(self, d_model, n_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model, dropout) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------------------
# Multi-Patch Mamba Model
# ----------------------------
class Model(nn.Module):
    """
    Multi-Patch PatchEmbedding + shared Mamba encoder
    + Volatility-aware ScaleGatingSE (per-scale)

    stride = padding = patch_len // 2 (per-scale)
    """

    def __init__(
        self,
        configs,
        patch_lens=(8, 16, 32),
        use_scale_gating=True,
        se_reduction=4,
        se_dropout=0.1,
    ):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.se_seduction = configs.se_reduction
        self.se_dropout = configs.se_dropout

        # per-scale patch settings
        self.patch_lens = list(configs.patch_lens)
        self.strides = [max(1, p // 2) for p in self.patch_lens]
        self.paddings = self.strides[:]  # padding = stride

        # --- multi-patch embedding ---
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                d_model=configs.d_model,
                patch_len=p,
                stride=s,
                padding=pad,
                dropout=configs.dropout,
            )
            for p, s, pad in zip(self.patch_lens, self.strides, self.paddings)
        ])

        # --- shared Mamba encoder ---
        self.encoder = MambaEncoder(
            d_model=configs.d_model,
            n_layers=configs.e_layers,
            dropout=configs.dropout,
        )

        # --- finance-aware scale gating (per-scale) ---
        self.use_scale_gating = configs.use_scale_gating
        if self.use_scale_gating:
            self.scale_gates = nn.ModuleList([
                VolatilityAwareScaleGatingSE(
                    d_model=configs.d_model,
                    reduction=self.se_seduction,
                    gate_dropout=self.se_dropout,
                )
                for _ in self.patch_lens
            ])

        # --- compute total patches (per-scale stride) ---
        self.patch_nums = [
            int((configs.seq_len - p) / s + 2)  # keep your "+2" logic
            for p, s in zip(self.patch_lens, self.strides)
        ]
        self.total_patches = sum(self.patch_nums)

        # --- head nf ---
        self.head_nf = configs.d_model * self.total_patches

        # --- heads ---
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.pred_len,
                head_dropout=configs.dropout,
            )

        elif self.task_name in ["imputation", "anomaly_detection"]:
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.seq_len,
                head_dropout=configs.dropout,
            )

        elif self.task_name == "classification":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in,
                configs.num_class,
            )

    # ----------------------------
    # Core encode (multi-patch)
    # ----------------------------
    def _encode(self, x_enc):
        """
        x_enc: [B, L, V]
        return: [B, V, D, sum(P)]
        """
        x_enc = x_enc.permute(0, 2, 1)  # [B, V, L]

        enc_outs = []
        for idx, patch_embed in enumerate(self.patch_embeddings):
            # PatchEmbedding expects [B, V, L] and returns z: [B*V, P, D]
            z, n_vars = patch_embed(x_enc)  # [B*V, P, D]

            # Shared Mamba encoder
            z = self.encoder(z)             # [B*V, P, D]

            # reshape -> [B, V, P, D] then -> [B, V, D, P]
            z = z.view(-1, n_vars, z.shape[-2], z.shape[-1])  # [B, V, P, D]
            z = z.permute(0, 1, 3, 2)                          # [B, V, D, P]

            # Volatility-aware gating per scale
            if self.use_scale_gating:
                z = self.scale_gates[idx](z)

            enc_outs.append(z)

        # concat over patch dimension
        return torch.cat(enc_outs, dim=-1)  # [B, V, D, sum(P)]

    # ----------------------------
    # Forecast task
    # ----------------------------
    def forecast(self, x_enc):
        # x_enc: [B, L, V]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means

        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self._encode(x_enc)                   # [B, V, D, sum(P)]
        dec_out = self.head(enc_out).permute(0, 2, 1)   # [B, pred_len, V]

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out

    def forward(self, x_enc, *args, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            out = self.forecast(x_enc)
            return out[:, -self.pred_len:, :]
        return None