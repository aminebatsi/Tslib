from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
warnings.filterwarnings('ignore')


class SharpeReward(nn.Module):
    """
    Differentiable Sharpe-ratio reward for use inside a loss function.
 
    Simulates a long/short strategy from predictions and computes the
    annualised Sharpe ratio of the resulting P&L. Used as a reward
    term (subtracted from the loss) so the model is trained to generate
    profitable signals, not just accurate predictions.
 
    Soft position:
        pos_t = tanh(r̂_t · κ)   ∈ (-1, +1)
 
    This approximates sign(r̂_t) but is fully differentiable, allowing
    gradients to flow from the trading outcome back to the prediction.
 
    Strategy return:
        pnl_t = pos_t · r_t
 
    Sharpe reward (annualised, assuming weekly steps):
        R_sharpe = (mean(pnl) / (std(pnl) + ε)) · √52
 
    Parameters
    ----------
    kappa : float
        Sharpness of the soft position. Higher κ → harder sign boundary.
        κ=5  is a good default for weekly log-returns (~0.05 magnitude).
        Default: 5.0
    annualise : bool
        If True, multiply by √52 (weekly) to annualise Sharpe.
        Default: True
    eps : float
        Numerical stability for std denominator.
        Default: 1e-6
    """
 
    def __init__(
        self,
        kappa: float = 5.0,
        annualise: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.kappa     = kappa
        self.annualise = annualise
        self.eps       = eps
 
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Returns the Sharpe ratio of the strategy implied by pred.
        Positive = good (profitable), negative = bad.
 
        Parameters
        ----------
        pred   : (B,) predicted log returns
        target : (B,) actual log returns
        """
        # soft long/short position ∈ (-1, +1)
        position = torch.tanh(pred * self.kappa)
 
        # per-step P&L
        pnl = position * target                                # (B,)
 
        sharpe = pnl.mean() / (pnl.std() + self.eps)
 
        if self.annualise:
            sharpe = sharpe * (52 ** 0.5)
 
        return sharpe
 
    def extra_repr(self) -> str:
        return f"kappa={self.kappa}, annualise={self.annualise}"
 
 
# ======================================================================
# 1. Lag-Aware MSE + RL
# ======================================================================
 
class LagAwareMSELossRL(nn.Module):
    """
    Lag-Aware MSE Loss with RL-inspired Sharpe reward.
 
    Full form:
        L = (1/T) Σ (r_t - r̂_t)²                       [MSE]
          + β  · (1/T-1) Σ (Δr̂_t - Δr_t)²              [anti-lag]
          - λ  · Sharpe(tanh(κ·r̂) · r)                  [RL reward]
 
    The three terms optimise for:
        MSE    → prediction accuracy
        β term → temporal alignment  (removes phase shift)
        λ term → risk-adjusted profit (aligns with trading objective)
 
    Parameters
    ----------
    beta   : float  -- anti-lag weight.      β=0 removes lag penalty.
    lam    : float  -- RL reward weight.     λ=0 removes Sharpe term.
    kappa  : float  -- soft-sign sharpness.
    reduction : str -- 'mean' | 'sum' | 'none'
 
    Example
    -------
    >>> criterion = LagAwareMSELossRL(beta=0.2, lam=0.1)
    >>> loss = criterion(pred, target)
    >>> loss.backward()
    """
 
    def __init__(
        self,
        beta: float = 0.2,
        lam: float = 0.1,
        kappa: float = 5.0,
        reduction: str = "mean",
    ):
        super().__init__()
        assert beta >= 0.0 and lam >= 0.0
        self.beta      = beta
        self.lam       = lam
        self.reduction = reduction
        self.sharpe    = SharpeReward(kappa=kappa)
 
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred   = pred.view(-1)
        target = target.view(-1)
 
        # --- MSE ---
        mse = (target - pred) ** 2
        if self.reduction == "mean":
            mse_loss = mse.mean()
        elif self.reduction == "sum":
            mse_loss = mse.sum()
        else:
            mse_loss = mse
 
        total = mse_loss
 
        # --- anti-lag ---
        if self.beta > 0.0 and pred.shape[0] > 1:
            delta_pred   = pred[1:]   - pred[:-1]
            delta_target = target[1:] - target[:-1]
            total = total + self.beta * ((delta_pred - delta_target) ** 2).mean()
 
        # --- RL reward (subtract to maximise) ---
        if self.lam > 0.0:
            total = total - self.lam * self.sharpe(pred, target)
 
        return total
 
    def extra_repr(self) -> str:
        return f"beta={self.beta}, lam={self.lam}, reduction='{self.reduction}'"
 
 
# ======================================================================
# 2. Full ARC-Loss + Anti-Lag + RL
# ======================================================================
 
class ARCLossRL(nn.Module):
    """
    Asymmetric Regime-Conditioned Loss v3 — with Anti-Lag + RL reward.
 
    Full form:
        L = (1/T) Σ λ(z_t) · φ(r_t,r̂_t) · (r_t-r̂_t)² · (1+γ|r_t|)   [ARC]
          + β · (1/T-1) Σ (Δr̂_t - Δr_t)²                               [anti-lag]
          - λ · Sharpe(tanh(κ·r̂) · r)                                   [RL reward]
 
    The four components ablate cleanly:
        without_rl()              → ARC v2  (no Sharpe term)
        without_anti_lag()        → ARC + RL (no lag penalty)
        without_direction()       → removes φ asymmetry
        without_magnitude()       → removes ψ scaling
        without_regime()          → removes λ(z_t)
 
    Parameters
    ----------
    alpha_plus  : float  -- penalty for missing upside  (default 2.0)
    alpha_minus : float  -- penalty for false upside    (default 3.0)
    gamma       : float  -- magnitude scaling           (default 1.0)
    beta        : float  -- anti-lag weight             (default 0.2)
    lam         : float  -- RL Sharpe reward weight     (default 0.1)
    kappa       : float  -- soft-sign sharpness         (default 5.0)
    regime_weights : Tensor (K,) or None
    learnable_weights : bool
    num_regimes : int
    reduction   : str
 
    Example
    -------
    >>> vols  = torch.tensor([0.02, 0.05, 0.08])
    >>> probs = torch.tensor([0.50, 0.30, 0.20])
    >>> criterion = ARCLossRL.from_volatility(vols, probs, lam=0.1, beta=0.2)
    >>> loss = criterion(pred, target, regimes)
    >>> loss.backward()
    """
 
    def __init__(
        self,
        alpha_plus: float = 2.0,
        alpha_minus: float = 3.0,
        gamma: float = 1.0,
        beta: float = 0.2,
        lam: float = 0.1,
        kappa: float = 5.0,
        regime_weights: Optional[Tensor] = None,
        learnable_weights: bool = False,
        num_regimes: int = 3,
        reduction: str = "mean",
    ):
        super().__init__()
        assert alpha_plus  >= 1.0
        assert alpha_minus >= 1.0
        assert gamma >= 0.0
        assert beta  >= 0.0
        assert lam   >= 0.0
        assert reduction in ("mean", "sum", "none")
 
        self.alpha_plus  = alpha_plus
        self.alpha_minus = alpha_minus
        self.gamma       = gamma
        self.beta        = beta
        self.lam         = lam
        self.reduction   = reduction
        self.sharpe      = SharpeReward(kappa=kappa)
        self.learnable_weights = learnable_weights
 
        if learnable_weights:
            self.eta = nn.Parameter(torch.zeros(num_regimes))
        else:
            if regime_weights is not None:
                self.register_buffer("regime_weights", regime_weights.float())
            else:
                self.regime_weights = None
 
    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_volatility(
        cls,
        regime_vols: Tensor,
        regime_probs: Tensor,
        **kwargs,
    ) -> "ARCLossRL":
        var = regime_vols ** 2
        weights = var / (regime_probs * var).sum()
        return cls(regime_weights=weights, **kwargs)
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_regime_weights(self, device) -> Optional[Tensor]:
        if self.learnable_weights:
            return torch.softmax(self.eta, dim=0).to(device)
        if hasattr(self, "regime_weights") and self.regime_weights is not None:
            return self.regime_weights.to(device)
        return None
 
    def _direction_penalty(self, target: Tensor, pred: Tensor) -> Tensor:
        phi = torch.ones_like(target)
        phi = torch.where((target > 0) & (pred <= 0),
                          torch.full_like(phi, self.alpha_plus),  phi)
        phi = torch.where((target < 0) & (pred >= 0),
                          torch.full_like(phi, self.alpha_minus), phi)
        return phi
 
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        regimes: Optional[Tensor] = None,
    ) -> Tensor:
        pred   = pred.view(-1)
        target = target.view(-1)
 
        # --- ARC core ---
        sq_err = (target - pred) ** 2
        phi    = self._direction_penalty(target, pred)
        psi    = 1.0 + self.gamma * target.abs()
        loss   = phi * psi * sq_err
 
        weights = self._get_regime_weights(pred.device)
        if weights is not None and regimes is not None:
            loss = weights[regimes] * loss
 
        if self.reduction == "mean":
            arc_loss = loss.mean()
        elif self.reduction == "sum":
            arc_loss = loss.sum()
        else:
            arc_loss = loss
 
        total = arc_loss
 
        # --- anti-lag ---
        if self.beta > 0.0 and pred.shape[0] > 1:
            dp = pred[1:]   - pred[:-1]
            dt = target[1:] - target[:-1]
            total = total + self.beta * ((dp - dt) ** 2).mean()
 
        # --- RL reward ---
        if self.lam > 0.0:
            total = total - self.lam * self.sharpe(pred, target)
 
        return total
 
    # ------------------------------------------------------------------
    # Ablation helpers
    # ------------------------------------------------------------------
    def without_rl(self) -> "ARCLossRL":
        cfg = self._config(); cfg["lam"] = 0.0
        return ARCLossRL(**cfg)
 
    def without_anti_lag(self) -> "ARCLossRL":
        cfg = self._config(); cfg["beta"] = 0.0
        return ARCLossRL(**cfg)
 
    def without_direction_penalty(self) -> "ARCLossRL":
        cfg = self._config(); cfg["alpha_plus"] = 1.0; cfg["alpha_minus"] = 1.0
        return ARCLossRL(**cfg)
 
    def without_magnitude_scaling(self) -> "ARCLossRL":
        cfg = self._config(); cfg["gamma"] = 0.0
        return ARCLossRL(**cfg)
 
    def without_regime_weighting(self) -> "ARCLossRL":
        cfg = self._config(); cfg["regime_weights"] = None; cfg["learnable_weights"] = False
        return ARCLossRL(**cfg)
 
    def _config(self) -> dict:
        base = dict(
            alpha_plus=self.alpha_plus,
            alpha_minus=self.alpha_minus,
            gamma=self.gamma,
            beta=self.beta,
            lam=self.lam,
            reduction=self.reduction,
            learnable_weights=self.learnable_weights,
        )
        if not self.learnable_weights and hasattr(self, "regime_weights"):
            base["regime_weights"] = (
                self.regime_weights.clone()
                if self.regime_weights is not None else None
            )
        return base
 
    def extra_repr(self) -> str:
        return (
            f"alpha_plus={self.alpha_plus}, alpha_minus={self.alpha_minus}, "
            f"gamma={self.gamma}, beta={self.beta}, lam={self.lam}, "
            f"reduction='{self.reduction}'"
        )
 
 
# ======================================================================
# Convenience: crypto defaults with all components
# ======================================================================
class CryptoARCLossRL(ARCLossRL):
    """
    Full ARC-Loss v3 with crypto-tuned defaults.
    3 regimes: 0=sideways, 1=bull, 2=bear.
    All four components active: ARC + anti-lag + RL reward.
    """
    def __init__(
        self,
        beta: float = 0.2,
        lam: float = 0.1,
        kappa: float = 5.0,
        reduction: str = "mean",
        learnable_weights: bool = False,
    ):
        vols    = torch.tensor([0.02, 0.05, 0.08])
        probs   = torch.tensor([0.50, 0.30, 0.20])
        var     = vols ** 2
        weights = var / (probs * var).sum()
        super().__init__(
            alpha_plus=2.0,
            alpha_minus=3.0,
            gamma=1.0,
            beta=beta,
            lam=lam,
            kappa=kappa,
            regime_weights=weights,
            learnable_weights=learnable_weights,
            num_regimes=3,
            reduction=reduction,
        )
 
    # ------------------------------------------------------------------
    # Diagnostic: estimate phase shift in steps
    # ------------------------------------------------------------------
    @staticmethod
    def measure_lag(pred: Tensor, target: Tensor, max_lag: int = 10) -> int:
        """
        Find the lag k* that maximises cross-correlation between pred and target.
        k*=0 means no lag; k*>0 means prediction is behind target by k* steps.
 
        Example
        -------
        >>> lag = LagAwareMSELoss.measure_lag(pred, target)
        >>> print(f"Phase shift: {lag} steps")
        """
        pred   = pred.view(-1).detach().float()
        target = target.view(-1).detach().float()
        pred   = (pred   - pred.mean())   / (pred.std()   + 1e-8)
        target = (target - target.mean()) / (target.std() + 1e-8)
 
        correlations = []
        for k in range(max_lag + 1):
            if k == 0:
                corr = (pred * target).mean()
            else:
                corr = (pred[k:] * target[:-k]).mean()
            correlations.append(corr.item())
 
        return int(torch.tensor(correlations).argmax().item())
 
    def extra_repr(self) -> str:
        return f"beta={self.beta}, reduction='{self.reduction}'"

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = CryptoARCLossRL(beta=0.2, lam=0.1)
        #criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return