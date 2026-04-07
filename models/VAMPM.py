import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Helpers
# =========================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class CausalConv1d(nn.Module):
    """
    1D causal convolution over temporal dimension.
    Input:  [B, L, C]
    Output: [B, L, C_out]
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C, L]
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        return x.transpose(1, 2)  # [B, L, C_out]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RevIN(nn.Module):
    """
    Simple RevIN for [B, L, C]
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        self._mean = None
        self._std = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x

        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x

        raise ValueError(f"Unknown RevIN mode: {mode}")


# =========================================================
# Volatility descriptors
# =========================================================
class VolatilityDescriptor(nn.Module):
    """
    Quant-finance inspired summary statistics from the target channel.
    Input: target series [B, L]
    Output: descriptors [B, 8]
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        dx = x[:, 1:] - x[:, :-1]
        abs_dx = dx.abs()

        mean_level = x.mean(dim=1)
        std_level = x.std(dim=1, unbiased=False)

        realized_vol = torch.sqrt((dx.pow(2).mean(dim=1)).clamp_min(self.eps))
        bipower_var = (abs_dx[:, 1:] * abs_dx[:, :-1]).mean(dim=1).clamp_min(self.eps)

        downside_semivar = torch.sqrt(
            (torch.clamp(dx, max=0.0).pow(2).mean(dim=1)).clamp_min(self.eps)
        )
        upside_semivar = torch.sqrt(
            (torch.clamp(dx, min=0.0).pow(2).mean(dim=1)).clamp_min(self.eps)
        )

        tail_ratio = abs_dx.max(dim=1).values / (abs_dx.mean(dim=1) + self.eps)
        jump_proxy = abs_dx.max(dim=1).values / (realized_vol + self.eps)

        out = torch.stack(
            [
                mean_level,
                std_level,
                realized_vol,
                bipower_var,
                downside_semivar,
                upside_semivar,
                tail_ratio,
                jump_proxy,
            ],
            dim=-1,
        )
        return out


# =========================================================
# Local shock branch
# =========================================================
class CausalShockBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_model * 4, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.conv1(h))
        h = self.drop(self.conv2(h))
        x = x + h
        x = x + self.ffn(self.norm(x))
        return x


class LocalShockBranch(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                CausalShockBlock(
                    d_model=d_model,
                    kernel_size=3,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# =========================================================
# Patch branch
# =========================================================
class PatchMixerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mlp1 = FeedForward(d_model, d_ff, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp2 = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp1(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        return x


class MultiScalePatchBranch(nn.Module):
    """
    Converts sequence to multi-scale patch tokens, processes each scale,
    then upsamples back to original length.
    """
    def __init__(
        self,
        d_model: int,
        patch_lens=(8, 16, 32),
        stride_ratio: float = 0.5,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_lens = list(patch_lens)
        self.scale_mixers = nn.ModuleList(
            [PatchMixerBlock(d_model, d_ff, dropout) for _ in self.patch_lens]
        )
        self.scale_proj = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in self.patch_lens]
        )
        self.out_proj = nn.Linear(len(self.patch_lens) * d_model, d_model)
        self.stride_ratio = stride_ratio

    def _make_patches(self, x: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, int]:
        # x: [B, L, D]
        B, L, D = x.shape
        stride = max(1, int(patch_len * self.stride_ratio))

        if L < patch_len:
            pad_len = patch_len - L
        else:
            rem = (L - patch_len) % stride
            pad_len = 0 if rem == 0 else (stride - rem)

        x_pad = F.pad(x, (0, 0, 0, pad_len))  # pad on temporal dimension
        patches = x_pad.unfold(dimension=1, size=patch_len, step=stride)  # [B, N, D, P] or [B,N,P,D] depending version
        if patches.dim() != 4:
            raise RuntimeError("Unexpected patch tensor shape")

        # Normalize layout to [B, N, P, D]
        if patches.shape[-1] == patch_len:
            patches = patches.permute(0, 1, 3, 2)

        N = patches.shape[1]
        return patches, x_pad.shape[1]

    def _patch_encode(self, x: torch.Tensor, patch_len: int, mixer: nn.Module, proj: nn.Module) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        patches, L_pad = self._make_patches(x, patch_len)  # [B, N, P, D]
        patch_tokens = patches.mean(dim=2)                 # [B, N, D]
        patch_tokens = proj(patch_tokens)
        patch_tokens = mixer(patch_tokens)                 # [B, N, D]

        # Upsample to original length
        patch_tokens = patch_tokens.transpose(1, 2)        # [B, D, N]
        up = F.interpolate(patch_tokens, size=L, mode="linear", align_corners=False)
        up = up.transpose(1, 2)                            # [B, L, D]
        return up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for p, mixer, proj in zip(self.patch_lens, self.scale_mixers, self.scale_proj):
            outs.append(self._patch_encode(x, p, mixer, proj))
        out = torch.cat(outs, dim=-1)
        out = self.out_proj(out)
        return out


# =========================================================
# Mamba-like long memory branch
# =========================================================
class MambaLikeBlock(nn.Module):
    """
    Pure PyTorch lightweight selective recurrent block.
    Not the official Mamba implementation, but a Mamba-inspired
    gated scan block that stays self-contained and TSLib-friendly.
    """
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 3 * d_state)
        self.x_proj = nn.Linear(d_state, d_state)
        self.dt_proj = nn.Linear(d_state, d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        residual = x
        x = self.norm(x)
        u, z, g = self.in_proj(x).chunk(3, dim=-1)  # [B, L, S]

        h = torch.zeros_like(u[:, 0, :])  # [B, S]
        ys = []

        for t in range(u.size(1)):
            u_t = torch.tanh(self.x_proj(u[:, t, :]))
            dt_t = torch.sigmoid(self.dt_proj(z[:, t, :]))
            h = (1.0 - dt_t) * h + dt_t * u_t
            y_t = h * torch.sigmoid(g[:, t, :])
            ys.append(y_t.unsqueeze(1))

        y = torch.cat(ys, dim=1)  # [B, L, S]
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class LongMemoryBranch(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MambaLikeBlock(d_model=d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)]
        )
        self.ffns = nn.ModuleList(
            [FeedForward(d_model, d_model * 4, dropout) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk, ffn, norm in zip(self.blocks, self.ffns, self.norms):
            x = blk(x)
            x = x + ffn(norm(x))
        return x


# =========================================================
# Cross-branch fusion
# =========================================================
class VolatilityAwareFusion(nn.Module):
    def __init__(self, d_model: int, se_reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = max(8, d_model // se_reduction)

        self.branch_gate = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4 * d_model),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x_base: torch.Tensor,
        x_local: torch.Tensor,
        x_patch: torch.Tensor,
        x_long: torch.Tensor,
        vol_embed_seq: torch.Tensor
    ) -> torch.Tensor:
        # each: [B, L, D]
        x_cat = torch.cat([x_local, x_patch, x_long, vol_embed_seq], dim=-1)
        gate = self.branch_gate(x_cat)
        x_cat = x_cat * gate
        out = x_base + self.fuse(x_cat)
        out = self.norm(out)
        return out


# =========================================================
# Regime detector + experts
# =========================================================
class RegimeDetector(nn.Module):
    def __init__(self, d_model: int, n_regimes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 8, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_regimes)
        )

    def forward(self, context: torch.Tensor, vol_desc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, vol_desc], dim=-1)
        return self.net(x)


class RegimeExpert(nn.Module):
    """
    One expert predicts:
    - mean returns
    - vol scale
    - direction logits
    - jump logits
    - quantiles
    """
    def __init__(self, d_model: int, pred_len: int, n_quantiles: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_quantiles = n_quantiles

        hidden = d_model * 2
        self.backbone = nn.Sequential(
            nn.Linear(d_model + 8, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mean_head = nn.Linear(hidden, pred_len)
        self.vol_head = nn.Linear(hidden, pred_len)
        self.dir_head = nn.Linear(hidden, pred_len)
        self.jump_head = nn.Linear(hidden, pred_len)
        self.quantile_head = nn.Linear(hidden, pred_len * n_quantiles)

    def forward(self, context: torch.Tensor, vol_desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(torch.cat([context, vol_desc], dim=-1))
        mean = self.mean_head(h)                                      # [B, H]
        vol = F.softplus(self.vol_head(h)) + 1e-6                     # [B, H]
        direction = self.dir_head(h)                                  # [B, H]
        jump = self.jump_head(h)                                      # [B, H]
        quantiles = self.quantile_head(h).view(h.size(0), self.pred_len, self.n_quantiles)
        return {
            "mean": mean,
            "vol": vol,
            "direction": direction,
            "jump": jump,
            "quantiles": quantiles
        }


# =========================================================
# Core block
# =========================================================
class ReturnSpecializedBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        d_model = configs.d_model
        d_ff = getattr(configs, "d_ff", d_model * 4)
        dropout = getattr(configs, "dropout", 0.1)
        patch_lens = getattr(configs, "patch_lens", [8, 16, 32])
        se_reduction = getattr(configs, "se_reduction", 4)
        d_state = getattr(configs, "d_state", 64)

        self.local_branch = LocalShockBranch(d_model, n_layers=2, dropout=dropout)
        self.patch_branch = MultiScalePatchBranch(
            d_model=d_model,
            patch_lens=patch_lens,
            stride_ratio=getattr(configs, "patch_stride_ratio", 0.5),
            d_ff=d_ff,
            dropout=dropout
        )
        self.long_branch = LongMemoryBranch(
            d_model=d_model,
            n_layers=2,
            d_state=d_state,
            dropout=dropout
        )
        self.fusion = VolatilityAwareFusion(
            d_model=d_model,
            se_reduction=se_reduction,
            dropout=dropout
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, vol_embed_seq: torch.Tensor) -> torch.Tensor:
        x_local = self.local_branch(x)
        x_patch = self.patch_branch(x)
        x_long = self.long_branch(x)
        x = self.fusion(x, x_local, x_patch, x_long, vol_embed_seq)
        x = x + self.ffn(self.norm(x))
        return x


# =========================================================
# Main TSLib-style model
# =========================================================
class Model(nn.Module):
    """
    TSLib-style forecasting model specialized for returns.

    Expected config fields:
    -----------------------
    task_name             = 'long_term_forecast' or 'short_term_forecast'
    seq_len               = input length
    pred_len              = output horizon
    enc_in                = number of encoder input channels
    c_out                 = should be 1 for return forecasting
    d_model               = latent dimension
    d_ff                  = ff hidden
    e_layers              = number of core blocks
    dropout               = dropout
    patch_lens            = [8, 16, 32]
    n_regimes             = number of regimes
    n_quantiles           = e.g. 3
    target_index          = index of primary return signal in x_enc
    use_revin             = True/False
    use_time_marks        = True/False
    time_dim              = number of x_mark features if concatenated
    return_aux_outputs    = True/False
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, "c_out", 1)
        self.d_model = configs.d_model
        self.e_layers = getattr(configs, "e_layers", 2)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.n_regimes = getattr(configs, "n_regimes", 4)
        self.n_quantiles = getattr(configs, "n_quantiles", 3)
        self.target_index = getattr(configs, "target_index", 0)
        self.use_revin = getattr(configs, "use_revin", False)
        self.use_time_marks = getattr(configs, "use_time_marks", False)
        self.time_dim = getattr(configs, "time_dim", 0)
        self.return_aux_outputs = getattr(configs, "return_aux_outputs", False)

        total_in = self.enc_in + (self.time_dim if self.use_time_marks else 0)

        if self.use_revin:
            self.revin = RevIN(self.enc_in)

        self.input_proj = nn.Sequential(
            nn.Linear(total_in, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model)
        )

        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, self.d_model) * 0.02)

        self.vol_desc = VolatilityDescriptor()
        self.vol_proj = nn.Sequential(
            nn.Linear(8, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.blocks = nn.ModuleList([ReturnSpecializedBlock(configs) for _ in range(self.e_layers)])
        self.final_norm = RMSNorm(self.d_model)

        self.context_pool = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh()
        )

        self.regime_detector = RegimeDetector(
            d_model=self.d_model,
            n_regimes=self.n_regimes,
            dropout=self.dropout
        )

        self.experts = nn.ModuleList(
            [
                RegimeExpert(
                    d_model=self.d_model,
                    pred_len=self.pred_len,
                    n_quantiles=self.n_quantiles,
                    dropout=self.dropout
                )
                for _ in range(self.n_regimes)
            ]
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(self.d_model + 8, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len),
            nn.Sigmoid()
        )

    # -----------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------
    def _prepare_inputs(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_enc: [B, L, C]
        x_target = x_enc[..., self.target_index]  # [B, L]

        if self.use_revin:
            x_enc = self.revin(x_enc, "norm")

        if self.use_time_marks and (x_mark_enc is not None):
            x = torch.cat([x_enc, x_mark_enc], dim=-1)
        else:
            x = x_enc

        x = self.input_proj(x)

        # positional embedding
        if x.size(1) <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :x.size(1), :]
        else:
            extra = x.size(1) - self.pos_emb.size(1)
            pe = F.interpolate(
                self.pos_emb.transpose(1, 2),
                size=x.size(1),
                mode="linear",
                align_corners=False
            ).transpose(1, 2)
            x = x + pe

        vol_desc = self.vol_desc(x_target)        # [B, 8]
        vol_embed = self.vol_proj(vol_desc)       # [B, D]
        vol_embed_seq = vol_embed.unsqueeze(1).expand(-1, x.size(1), -1)

        return x, vol_desc, vol_embed_seq

    def _encode(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vol_desc, vol_embed_seq = self._prepare_inputs(x_enc, x_mark_enc)

        for blk in self.blocks:
            x = blk(x, vol_embed_seq)

        x = self.final_norm(x)

        # attention-like learned temporal pooling
        score = self.context_pool(x).mean(dim=-1, keepdim=True)  # [B, L, 1]
        attn = torch.softmax(score, dim=1)
        context = (x * attn).sum(dim=1)                          # [B, D]

        return context, vol_desc

    def _mixture_forward(self, context: torch.Tensor, vol_desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        regime_logits = self.regime_detector(context, vol_desc)        # [B, K]
        regime_probs = torch.softmax(regime_logits, dim=-1)            # [B, K]

        means = []
        vols = []
        dirs = []
        jumps = []
        quants = []

        for expert in self.experts:
            out = expert(context, vol_desc)
            means.append(out["mean"].unsqueeze(1))         # [B,1,H]
            vols.append(out["vol"].unsqueeze(1))           # [B,1,H]
            dirs.append(out["direction"].unsqueeze(1))     # [B,1,H]
            jumps.append(out["jump"].unsqueeze(1))         # [B,1,H]
            quants.append(out["quantiles"].unsqueeze(1))   # [B,1,H,Q]

        means = torch.cat(means, dim=1)    # [B,K,H]
        vols = torch.cat(vols, dim=1)      # [B,K,H]
        dirs = torch.cat(dirs, dim=1)      # [B,K,H]
        jumps = torch.cat(jumps, dim=1)    # [B,K,H]
        quants = torch.cat(quants, dim=1)  # [B,K,H,Q]

        w = regime_probs.unsqueeze(-1)     # [B,K,1]
        mean = (means * w).sum(dim=1)      # [B,H]
        vol = (vols * w).sum(dim=1)        # [B,H]
        direction = (dirs * w).sum(dim=1)  # [B,H]
        jump = (jumps * w).sum(dim=1)      # [B,H]

        wq = regime_probs.unsqueeze(-1).unsqueeze(-1)  # [B,K,1,1]
        quantiles = (quants * wq).sum(dim=1)           # [B,H,Q]

        confidence = self.confidence_head(torch.cat([context, vol_desc], dim=-1))  # [B,H]
        mean = mean * confidence

        return {
            "pred": mean.unsqueeze(-1),                 # [B,H,1]
            "mean": mean,                               # [B,H]
            "vol": vol,                                 # [B,H]
            "direction_logits": direction,              # [B,H]
            "jump_logits": jump,                        # [B,H]
            "quantiles": quantiles,                     # [B,H,Q]
            "regime_logits": regime_logits,             # [B,K]
            "regime_probs": regime_probs,               # [B,K]
            "confidence": confidence                    # [B,H]
        }

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None
    ):
        context, vol_desc = self._encode(x_enc, x_mark_enc)
        out = self._mixture_forward(context, vol_desc)

        pred = out["pred"]  # [B,H,1]

        if self.c_out > 1:
            pred = pred.repeat(1, 1, self.c_out)

        if self.return_aux_outputs:
            out["pred"] = pred
            return out

        return pred

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: Optional[torch.Tensor],
        x_mark_dec: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise NotImplementedError(f"Task {self.task_name} not supported.")


# =========================================================
# Optional finance-aware loss for this architecture
# =========================================================
class ReturnQuantFinanceLoss(nn.Module):
    """
    Loss for training the model with:
    - robust return loss
    - direction loss
    - volatility calibration
    - quantile loss
    - regime entropy regularization
    - jump event loss (optional)
    """
    def __init__(
        self,
        quantiles=(0.1, 0.5, 0.9),
        lambda_return=1.0,
        lambda_direction=0.3,
        lambda_vol=0.2,
        lambda_quantile=0.3,
        lambda_regime=0.01,
        lambda_jump=0.1
    ):
        super().__init__()
        self.quantiles = quantiles
        self.lambda_return = lambda_return
        self.lambda_direction = lambda_direction
        self.lambda_vol = lambda_vol
        self.lambda_quantile = lambda_quantile
        self.lambda_regime = lambda_regime
        self.lambda_jump = lambda_jump

    @staticmethod
    def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
        err = target - pred_q
        return torch.mean(torch.maximum(q * err, (q - 1.0) * err))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_returns: torch.Tensor,
        target_vol: Optional[torch.Tensor] = None,
        target_jump: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        target_returns: [B, H] or [B, H, 1]
        target_vol:     [B, H] optional
        target_jump:    [B, H] optional binary
        """
        if target_returns.dim() == 3:
            target_returns = target_returns.squeeze(-1)

        mean = outputs["mean"]
        vol = outputs["vol"]
        direction_logits = outputs["direction_logits"]
        jump_logits = outputs["jump_logits"]
        quantiles = outputs["quantiles"]
        regime_probs = outputs["regime_probs"]

        # 1) robust return loss
        return_loss = F.huber_loss(mean, target_returns, delta=0.01)

        # 2) direction loss
        target_dir = (target_returns > 0).float()
        direction_loss = F.binary_cross_entropy_with_logits(direction_logits, target_dir)

        # 3) volatility loss
        if target_vol is None:
            target_vol = target_returns.abs().detach()
        if target_vol.dim() == 3:
            target_vol = target_vol.squeeze(-1)

        vol_loss = F.huber_loss(vol, target_vol, delta=0.01)

        # 4) quantile loss
        qloss = 0.0
        for i, q in enumerate(self.quantiles):
            qloss = qloss + self.pinball_loss(quantiles[..., i], target_returns, q)
        qloss = qloss / len(self.quantiles)

        # 5) regime entropy reg (encourage decisive but not collapsed probabilities)
        entropy = -(regime_probs * torch.log(regime_probs.clamp_min(1e-8))).sum(dim=-1).mean()

        # 6) jump loss
        if target_jump is None:
            jump_threshold = 2.0 * target_returns.std(dim=1, keepdim=True).clamp_min(1e-6)
            target_jump = (target_returns.abs() > jump_threshold).float()

        jump_loss = F.binary_cross_entropy_with_logits(jump_logits, target_jump)

        total = (
            self.lambda_return * return_loss
            + self.lambda_direction * direction_loss
            + self.lambda_vol * vol_loss
            + self.lambda_quantile * qloss
            + self.lambda_regime * entropy
            + self.lambda_jump * jump_loss
        )

        return {
            "loss": total,
            "return_loss": return_loss.detach(),
            "direction_loss": direction_loss.detach(),
            "vol_loss": vol_loss.detach(),
            "quantile_loss": qloss.detach(),
            "regime_entropy": entropy.detach(),
            "jump_loss": jump_loss.detach()
        }