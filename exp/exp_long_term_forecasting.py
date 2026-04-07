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

warnings.filterwarnings('ignore')


import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable Pearson correlation over the last dimension."""
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    num = (x_c * y_c).sum(dim=-1)
    denom = x_c.norm(dim=-1) * y_c.norm(dim=-1) + eps
    return num / denom                      # shape: (...,) in [-1, 1]
 
 
def _rolling_realized_vol(
    x: torch.Tensor, window: int, eps: float = 1e-8
) -> torch.Tensor:
    """
    Rolling realised volatility: RV_t = sqrt(mean(r_{t-w:t}^2)).
    Returns a tensor of same length as input (first window-1 steps use
    expanding window for stability).
    x shape: (T,) or (B, T).
    """
    flat = x.shape[-1] == x.numel()         # single series
    if x.dim() == 1:
        x = x.unsqueeze(0)                  # (1, T)
 
    B, T = x.shape
    rv = torch.zeros_like(x)
    for t in range(T):
        start = max(0, t - window + 1)
        rv[:, t] = x[:, start:t + 1].pow(2).mean(dim=-1).sqrt()
 
    return rv.squeeze(0) if flat else rv    # restore original shape
 
 
def _standardized_moment(x: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    """k-th standardized moment (skewness k=3, kurtosis k=4)."""
    x_c = x - x.mean(dim=-1, keepdim=True)
    std = x_c.std(dim=-1, keepdim=True) + eps
    return ((x_c / std) ** k).mean(dim=-1)
 
 
def _cumulative_drawdown_weights(
    target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes a weight per timestep proportional to how deep the
    cumulative loss run is at that point.
 
    A 'loss run' is a streak of consecutive negative returns.  The weight
    at step t = (length of current negative streak)^0.5, normalised so
    mean = 1.  This up-weights errors that occur during sustained sell-offs
    without completely ignoring quiet or positive-return periods.
 
    Returns weights with same shape as target, mean ≈ 1.
    """
    neg = (target < 0).float()              # 1 where negative return
    T   = target.shape[-1]
    streak = torch.zeros_like(target)
    for t in range(T):
        if t == 0:
            streak[..., t] = neg[..., t]
        else:
            streak[..., t] = (streak[..., t - 1] + 1) * neg[..., t]
 
    weights = (streak + 1.0).sqrt()         # +1 so non-drawdown steps have weight 1
    weights = weights / (weights.mean(dim=-1, keepdim=True) + eps)
    return weights
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main loss class
# ─────────────────────────────────────────────────────────────────────────────
 
class QuantNativeLoss(nn.Module):
    """
    Ultimate quant-native loss for log-return prediction.
 
    Args:
        w_ic         : weight for IC (correlation) loss.           Default 0.40
        w_vol        : weight for log-vol matching loss.           Default 0.30
        w_norm       : weight for normalised-residual loss.        Default 0.20
        w_moment     : weight for moment-matching loss.            Default 0.07
        w_drawdown   : weight for drawdown-weighted residual.      Default 0.03
        rv_window    : rolling window size for realised vol.       Default 20
        norm_window  : rolling window for local vol normalisation. Default 10
        moment_alpha : relative weight of skewness vs variance
                       inside the moment term (0=var only,1=skew only).
                                                                   Default 0.5
        eps          : numerical stability constant.
    """
 
    def __init__(
        self,
        w_ic: float        = 0.40,
        w_vol: float       = 0.30,
        w_norm: float      = 0.20,
        w_moment: float    = 0.07,
        w_drawdown: float  = 0.03,
        rv_window: int     = 20,
        norm_window: int   = 10,
        moment_alpha: float = 0.5,
        eps: float         = 1e-8,
    ):
        super().__init__()
        total = w_ic + w_vol + w_norm + w_moment + w_drawdown
        self.w_ic       = w_ic       / total
        self.w_vol      = w_vol      / total
        self.w_norm     = w_norm     / total
        self.w_moment   = w_moment   / total
        self.w_drawdown = w_drawdown / total
 
        self.rv_window    = rv_window
        self.norm_window  = norm_window
        self.moment_alpha = moment_alpha
        self.eps          = eps
 
    # ── Component 1 ── IC Loss ──────────────────────────────────────────────
    def _ic_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        L_IC = 1 - Pearson_corr(pred, target)
 
        Range: [0, 2].  0 = perfect positive correlation.
        Gradient: directly increases the linear relationship between
        pred and target.  This is the differentiable analogue of the
        Information Coefficient used in systematic fund performance
        attribution.
 
        Key insight: MSE minimises E[(p-t)²], which is minimised by
        predicting the conditional mean (≈ 0 for zero-mean returns).
        Correlation maximisation is minimised by pred = 0 *only* if
        target = 0, which is never true — so this term always pushes
        the model toward tracking the actual return path.
        """
        corr = _pearson_corr(pred, target, self.eps)
        return (1.0 - corr).mean()
 
    # ── Component 2 ── Log-Vol Matching ─────────────────────────────────────
    def _log_vol_matching(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        L_vol = MSE( log(RV_pred + eps), log(RV_true + eps) )
 
        RV_t = sqrt( mean( r_{t-w:t}^2 ) )   — rolling realised vol
 
        Log-space comparison is critical:
          • RV typically spans 0.001 to 0.05 for daily crypto returns.
          • In linear space, MSE is dominated by large-vol periods.
          • In log space, relative errors are penalised equally at all
            volatility levels — exactly what we want.
 
        This is the PRIMARY fix for the flat orange line.  If pred has
        near-zero amplitude (RV ≈ 0.001) and target has RV ≈ 0.02,
        log(0.001) vs log(0.02) is a large penalty that MSE would barely
        notice.
        """
        rv_pred   = _rolling_realized_vol(pred,   self.rv_window, self.eps)
        rv_target = _rolling_realized_vol(target, self.rv_window, self.eps)
 
        log_rv_pred   = torch.log(rv_pred   + self.eps)
        log_rv_target = torch.log(rv_target + self.eps)
 
        return F.mse_loss(log_rv_pred, log_rv_target)
 
    # ── Component 3 ── Normalised Residual Loss ──────────────────────────────
    def _normalised_residual_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        L_norm = mean( (pred - target)² / sigma_t² )
 
        sigma_t = rolling_std(target, window)   — local volatility estimate
 
        This is the log-likelihood of a locally-Gaussian model:
          r_t ~ N(pred_t, sigma_t²)
          -log p(r_t) ∝ (r_t - pred_t)² / sigma_t²
 
        Effect: a 2% error on a 0.5%-vol day is penalised 16× more than
        the same error on a 2%-vol day.  The model cannot reduce loss by
        predicting the mean — it must track actual returns proportionally
        to local vol.
 
        Unlike plain MSE, this enforces *equal risk contribution* from
        every timestep, matching how quants think about diversified signal.
        """
        T = target.shape[-1]
        sigma = torch.zeros_like(target)
        for t in range(T):
            start = max(0, t - self.norm_window + 1)
            window_data = target[..., start:t + 1]
            sigma[..., t] = window_data.std(dim=-1) + self.eps
 
        normalised_sq_err = ((pred - target) / sigma) ** 2
        return normalised_sq_err.mean()
 
    # ── Component 4 ── Moment Matching ──────────────────────────────────────
    def _moment_matching_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        L_moment = (1-alpha) * L_var + alpha * L_skew
 
        L_var  = ( (std_pred - std_target) / std_target )²
        L_skew = ( skew_pred - skew_target )²
 
        BTC log-returns are:
          • Leptokurtic (fat tails): excess kurtosis ≈ 5–15
          • Negatively skewed:       skewness ≈ -0.5 to -1.5
 
        MSE will produce predictions with near-Gaussian, near-zero-mean
        characteristics.  This term penalises the prediction distribution
        directly — the model must not only track the path but reproduce
        the statistical texture of BTC returns.
 
        Variance term is normalised by target variance so it is scale-free.
        """
        std_pred   = pred.std(dim=-1)   + self.eps
        std_target = target.std(dim=-1) + self.eps
 
        # Relative variance mismatch
        l_var = ((std_pred - std_target) / std_target).pow(2).mean()
 
        # Skewness mismatch
        skew_pred   = _standardized_moment(pred,   3, self.eps)
        skew_target = _standardized_moment(target, 3, self.eps)
        l_skew = (skew_pred - skew_target).pow(2).mean()
 
        return (1.0 - self.moment_alpha) * l_var + self.moment_alpha * l_skew
 
    # ── Component 5 ── Drawdown-Weighted Residual ───────────────────────────
    def _drawdown_weighted_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        L_dd = mean( w_t * (pred - target)² )
 
        w_t = sqrt(consecutive_negative_streak_length_at_t)  normalised.
 
        Intuition from risk management: sustained drawdown periods are
        when prediction quality matters most for portfolio protection.
        A model that is accurate during rallies but badly wrong during
        a 5-day sell-off is dangerous.  This term ensures the gradient
        specifically cares about those windows.
        """
        weights = _cumulative_drawdown_weights(target, self.eps)
        return (weights * (pred - target).pow(2)).mean()
 
    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred   : predicted log-returns  shape (batch, seq_len) or (seq_len,)
            target : true log-returns       same shape
 
        Returns:
            Scalar loss.
        """
        assert pred.shape == target.shape, (
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )
        # pred, target: (batch, seq_len, num_vars) or (batch, seq_len)
        if pred.dim() == 3:
            B, T, V = pred.shape
            # Treat each variable independently: flatten batch & var
            pred   = pred.permute(0,2,1).reshape(B*V, T)
            target = target.permute(0,2,1).reshape(B*V, T)
        elif pred.dim() == 2:
            pass  # already (batch, seq_len)
        elif pred.dim() == 1:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported shape: {pred.shape}")

 
        l_ic   = self._ic_loss(pred, target)
        l_vol  = self._log_vol_matching(pred, target)
        l_norm = self._normalised_residual_loss(pred, target)
        l_mom  = self._moment_matching_loss(pred, target)
        l_dd   = self._drawdown_weighted_loss(pred, target)
 
        return (
            self.w_ic   * l_ic
            + self.w_vol  * l_vol
            + self.w_norm * l_norm
            + self.w_mom  * l_mom
            + self.w_drawdown * l_dd
        )
 
    @property
    def w_mom(self):
        return self.w_moment
 
    def component_losses(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict:
        """
        Returns all raw component losses for W&B / TensorBoard logging.
 
        Recommended usage:
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            # log every N steps:
            comps = criterion.component_losses(pred.detach(), target)
            wandb.log(comps)
        """
        if pred.dim() == 1:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)
 
        return {
            "loss/ic":       self._ic_loss(pred, target).item(),
            "loss/log_vol":  self._log_vol_matching(pred, target).item(),
            "loss/norm_res": self._normalised_residual_loss(pred, target).item(),
            "loss/moment":   self._moment_matching_loss(pred, target).item(),
            "loss/drawdown": self._drawdown_weighted_loss(pred, target).item(),
        }
 
    def ic_score(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float:
        """
        Returns the raw IC (Pearson corr) as a diagnostic metric.
        IC > 0.05 is considered meaningful in systematic trading.
        IC > 0.10 is considered good.
        """
        if pred.dim() == 1:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        return _pearson_corr(pred, target, self.eps).mean().item()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Training recipe — important scheduler tip
# ─────────────────────────────────────────────────────────────────────────────
 
class WarmStartQuantLoss(nn.Module):
    """
    Two-phase loss scheduler.
 
    Phase 1 (epochs 0 → warmup_epochs):
        Pure MSELoss — lets the model learn the rough scale and
        mean of returns before introducing the harder objectives.
 
    Phase 2 (epochs > warmup_epochs):
        QuantNativeLoss — full quant-native objectives take over.
 
    Why this matters:
        The IC loss gradient is noisy when the model output is random
        (early training).  Starting with MSE warm-up stabilises the
        hidden states before switching to the correlation objective.
    """
 
    def __init__(self, warmup_epochs: int = 5, **quant_kwargs):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.mse           = nn.MSELoss()
        self.quant         = QuantNativeLoss(**quant_kwargs)
        self._epoch        = 0
 
    def step_epoch(self):
        """Call once per epoch: criterion.step_epoch()"""
        self._epoch += 1
 
    @property
    def phase(self) -> str:
        return "warmup" if self._epoch < self.warmup_epochs else "quant"
 
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self._epoch < self.warmup_epochs:
            return self.mse(pred, target)
        return self.quant(pred, target)
    
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
        return QuantNativeLoss()

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