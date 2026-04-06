# exp/exp_long_term_forecasting.py
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
import io
import csv
from datetime import datetime

from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

# trading utils
from utils.trading import simulate_trade, max_drawdown

warnings.filterwarnings("ignore")


# =========================================================
#   Complexity + Training time helpers
# =========================================================
def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb_state_dict(model: nn.Module):
    """
    Reviewer-safe model size: serialized state_dict size (MB).
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getbuffer()) / (1024**2)


def profile_macs_flops(model, loader, args, device):
    """
    Estimate MACs/FLOPs using THOP on a real batch (B=1).
    FLOPs is approximated as 2 * MACs.

    IMPORTANT: profile on a deepcopy so THOP doesn't pollute the real model.
    """
    try:
        from thop import profile
    except Exception:
        return None, None

    import copy

    m0 = model.module if hasattr(model, "module") else model
    m = copy.deepcopy(m0).to(device)
    m.eval()

    try:
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
        batch_x = batch_x[:1].float().to(device)
        batch_x_mark = batch_x_mark[:1].float().to(device)
        batch_y_mark = batch_y_mark[:1].float().to(device)

        dec_inp = torch.zeros_like(batch_y[:1, -args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:1, : args.label_len, :], dec_inp], dim=1).float().to(device)

        macs, _ = profile(m, inputs=(batch_x, batch_x_mark, dec_inp, batch_y_mark), verbose=False)
        flops = 2 * macs
        return macs, flops
    except Exception:
        return None, None


def measure_train_iter_time_ms(model, loader, args, device, criterion, optimizer, warmup=20, runs=100):
    """
    Training time per iteration (ms): forward + loss + backward + optimizer step.
    Compute-only (no dataloader time), uses a real batch.
    """
    m = model.module if hasattr(model, "module") else model
    m.train()

    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        outputs = m(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len :, f_dim:]
        y_cut = batch_y[:, -args.pred_len :, f_dim:]
        loss = criterion(outputs, y_cut)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        optimizer.zero_grad(set_to_none=True)
        outputs = m(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len :, f_dim:]
        y_cut = batch_y[:, -args.pred_len :, f_dim:]
        loss = criterion(outputs, y_cut)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) * 1000.0 / runs


def _strip_thop_keys_from_state_dict(state_dict: dict) -> dict:
    """
    Removes THOP-injected keys like *.total_ops / *.total_params so checkpoints remain loadable.
    """
    bad = [
        k
        for k in state_dict.keys()
        if k.endswith("total_ops") or k.endswith("total_params") or ".total_ops" in k or ".total_params" in k
    ]
    if not bad:
        return state_dict
    for k in bad:
        state_dict.pop(k, None)
    return state_dict


def _append_row_csv(csv_file: str, fieldnames: list, row: dict):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._complexity_cache = {}  # will be filled in train()

    # =========================================================
    # setting: Model_Dataset_seq_pred_len_e_layers_d_model_d_ff
    # =========================================================
    def _make_setting(self) -> str:
        model = str(getattr(self.args, "model", "Model"))
        dataset = str(getattr(self.args, "data", getattr(self.args, "dataset", "Dataset")))
        seq_len = int(getattr(self.args, "seq_len", 0))
        pred_len = int(getattr(self.args, "pred_len", 0))
        e_layers = int(getattr(self.args, "e_layers", 0))
        d_model = int(getattr(self.args, "d_model", 0))
        d_ff = int(getattr(self.args, "d_ff", 0))
        return f"{model}_{dataset}_{seq_len}_{pred_len}_{e_layers}_{d_model}_{d_ff}"

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        setting = self._make_setting() if not setting else setting

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # =================================================
        # Complexity + Train iter time (ONCE)
        # =================================================
        m = self.model.module if hasattr(self.model, "module") else self.model
        total_params, trainable_params = count_params(m)
        model_mb = model_size_mb_state_dict(m)
        macs, flops = profile_macs_flops(self.model, train_loader, self.args, self.device)
        train_iter_ms = measure_train_iter_time_ms(self.model, train_loader, self.args, self.device, criterion, model_optim)

        print(f"[Complexity] Params: {total_params:,} (trainable={trainable_params:,})")
        print(f"[Complexity] Model size (state_dict): {model_mb:.2f} MB")
        print(f"[Complexity] MACs: {macs} | FLOPs: {flops}")
        print(f"[Complexity] Training time per iteration: {train_iter_ms:.3f} ms")

        # cache for test() main CSV
        self._complexity_cache = {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "model_mb_state_dict": float(model_mb),
            "macs": float(macs) if macs is not None else None,
            "flops": float(flops) if flops is not None else None,
            "train_iter_ms": float(train_iter_ms),
        }

        # keep your existing benchmark_complexity.csv (optional, but harmless)
        csv_path = os.path.join(self.args.checkpoints, "benchmark_complexity.csv")
        header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if header:
                f.write(
                    "setting,model,seq_len,label_len,pred_len,features,enc_in,dec_in,c_out,"
                    "total_params,trainable_params,model_mb,macs,flops,train_iter_ms\n"
                )
            f.write(
                f"{setting},{self.args.model},{self.args.seq_len},{self.args.label_len},{self.args.pred_len},"
                f"{self.args.features},{self.args.enc_in},{self.args.dec_in},{self.args.c_out},"
                f"{total_params},{trainable_params},{model_mb:.6f},"
                f"{macs if macs is not None else 'NA'},{flops if flops is not None else 'NA'},"
                f"{train_iter_ms:.6f}\n"
            )

        # =================================================
        # ORIGINAL TRAINING LOOP (unchanged)
        # =================================================
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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y_cut)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y_cut)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
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
            train_loss_v = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss_v, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting=None, test=0):
        setting = self._make_setting() if not setting else setting

        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            ckpt = torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"), map_location=self.device)
            ckpt = _strip_thop_keys_from_state_dict(ckpt)
            self.model.load_state_dict(ckpt, strict=False)

        preds, trues, todays = [], [], []

        # per-batch timestamps + forward timing
        batch_timestamps = []
        batch_infer_ms = []
        test_time_start = time.time()

        vis_path = "./test_results/" + setting + "/"
        os.makedirs(vis_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_timestamps.append(datetime.utcnow().isoformat())

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                batch_infer_ms.append((time.time() - t0) * 1000.0)

                # keep original inverse logic but also inverse today
                f_dim = -1 if self.args.features == "MS" else 0

                today_t = batch_x[:, -1:, :].detach().cpu().numpy()  # (B,1,C)

                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y_cut = batch_y[:, -self.args.pred_len :, :]

                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y_cut.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y_np.shape
                    if outputs_np.shape[-1] != batch_y_np.shape[-1]:
                        outputs_np = np.tile(outputs_np, [1, 1, int(batch_y_np.shape[-1] / outputs_np.shape[-1])])
                    outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    tshape = today_t.shape
                    today_2d = today_t.reshape(tshape[0] * tshape[1], -1)
                    today_t = test_data.inverse_transform(today_2d).reshape(tshape)

                outputs_np = outputs_np[:, :, f_dim:]
                batch_y_np = batch_y_np[:, :, f_dim:]

                today_t = today_t[:, :, f_dim:]  # (B,1,C)
                today_1d = today_t[:, 0, 0]  # (B,)

                pred = outputs_np
                true = batch_y_np

                preds.append(pred)
                trues.append(true)
                todays.append(today_1d)

                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        ishape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.reshape(ishape[0] * ishape[1], -1)).reshape(ishape)
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(vis_path, str(i) + ".pdf"))

        test_time_total_s = time.time() - test_time_start
        batch_count = len(batch_infer_ms)
        avg_batch_infer_ms = float(np.mean(batch_infer_ms)) if batch_count > 0 else float("nan")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        todays = np.concatenate(todays, axis=0)

        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        res_path = "./results/" + setting + "/"
        os.makedirs(res_path, exist_ok=True)

        # DTW
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
            dtw_v = np.array(dtw_list).mean()
        else:
            dtw_v = "Not calculated"

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw_v))

        with open("result_long_term_forecast.txt", "a", encoding="utf-8") as f:
            f.write(setting + "  \n")
            f.write("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw_v))
            f.write("\n\n")

        np.save(os.path.join(res_path, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(res_path, "pred.npy"), preds)
        np.save(os.path.join(res_path, "true.npy"), trues)

        # =========================================================
        # MAIN GLOBAL CSV: EVERYTHING in one line per run
        # =========================================================
        global_metrics_csv = os.path.join("./results", "all_metrics.csv")

        metrics_fields = [
            # run/meta
            "timestamp_run_utc",
            "setting",
            "model",
            "dataset",
            # config
            "seq_len",
            "label_len",
            "pred_len",
            "features",
            "enc_in",
            "dec_in",
            "c_out",
            # complexity
            "total_params",
            "trainable_params",
            "model_mb_state_dict",
            "macs",
            "flops",
            "train_iter_ms",
            # test timing
            "test_batches",
            "test_time_total_s",
            "avg_batch_infer_ms",
            # per-batch timing/timestamps (pipe-separated)
            "batch_timestamps_utc",
            "batch_infer_ms",
            # metrics
            "use_dtw",
            "dtw",
            "mae",
            "mse",
            "rmse",
            "mape",
            "mspe",
        ]

        cplx = getattr(self, "_complexity_cache", {})

        row = {
            "timestamp_run_utc": datetime.utcnow().isoformat(),
            "setting": setting,
            "model": str(getattr(self.args, "model", "")),
            "dataset": str(getattr(self.args, "data_path", getattr(self.args, "dataset", ""))),
            "seq_len": int(getattr(self.args, "seq_len", 0)),
            "label_len": int(getattr(self.args, "label_len", 0)),
            "pred_len": int(getattr(self.args, "pred_len", 0)),
            "features": str(getattr(self.args, "features", "")),
            "enc_in": int(getattr(self.args, "enc_in", 0)),
            "dec_in": int(getattr(self.args, "dec_in", 0)),
            "c_out": int(getattr(self.args, "c_out", 0)),
            "total_params": cplx.get("total_params"),
            "trainable_params": cplx.get("trainable_params"),
            "model_mb_state_dict": cplx.get("model_mb_state_dict"),
            "macs": cplx.get("macs"),
            "flops": cplx.get("flops"),
            "train_iter_ms": cplx.get("train_iter_ms"),
            "test_batches": int(batch_count),
            "test_time_total_s": float(test_time_total_s),
            "avg_batch_infer_ms": float(avg_batch_infer_ms),
            "batch_timestamps_utc": "|".join(batch_timestamps),
            "batch_infer_ms": "|".join(f"{x:.6f}" for x in batch_infer_ms),
            "use_dtw": int(bool(getattr(self.args, "use_dtw", False))),
            "dtw": float(dtw_v) if isinstance(dtw_v, (int, float, np.floating)) else str(dtw_v),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
        }

        _append_row_csv(global_metrics_csv, metrics_fields, row)

        # =========================================================
        # Trading simulation (only if activated)
        # =========================================================
        use_trading = bool(getattr(self.args, "use_trading", False))

        if use_trading:
            try:
                pred_next = preds[:, 0, 0]
                today_prices = todays

                init_balance = float(getattr(self.args, "trade_balance", 100.0))
                mode = str(getattr(self.args, "trade_mode", "smart"))
                risk = float(getattr(self.args, "trade_risk", 5.0))
                tr = float(getattr(self.args, "trade_tr", 0.01))
                fee_bps = float(getattr(self.args, "trade_fee_bps", 0.0))
                max_short = float(getattr(self.args, "trade_max_short", 0.002))

                final_balance, equity_curve, n_trades = simulate_trade(
                    today_prices=today_prices,
                    pred_next_prices=pred_next,
                    init_balance=init_balance,
                    mode=mode,
                    risk=risk,
                    tr=tr,
                    fee_bps=fee_bps,
                    max_short=max_short,
                )

                mdd = max_drawdown(equity_curve)
                total_return = (final_balance / max(init_balance, 1e-12)) - 1.0

                equity_curve = np.asarray(equity_curve, dtype=float)
                trade_curve_csv = os.path.join(res_path, f"trading_equity_{mode}.csv")
                np.savetxt(
                    trade_curve_csv,
                    np.column_stack([np.arange(len(equity_curve)), equity_curve]),
                    delimiter=",",
                    header="step,equity",
                    comments="",
                )

                # text summary
                with open(os.path.join(res_path, "trading_summary.txt"), "w", encoding="utf-8") as fsum:
                    fsum.write(f"setting={setting}\n")
                    fsum.write(f"trade_mode={mode}\n")
                    fsum.write(f"init_balance={init_balance}\n")
                    fsum.write(f"final_balance={final_balance:.6f}\n")
                    fsum.write(f"total_return_pct={total_return*100:.4f}\n")
                    fsum.write(f"max_drawdown_pct={mdd*100:.4f}\n")
                    fsum.write(f"trades={n_trades}\n")
                    fsum.write(f"risk={risk}\n")
                    fsum.write(f"tr={tr}\n")
                    fsum.write(f"fee_bps={fee_bps}\n")
                    fsum.write(f"max_short={max_short}\n")

                # separate trading summary CSV
                trading_csv = os.path.join("./results", "trading_summary.csv")
                trading_fields = [
                    "timestamp",
                    "setting",
                    "model",
                    "dataset",
                    "seq_len",
                    "pred_len",
                    "trade_mode",
                    "init_balance",
                    "final_balance",
                    "total_return_pct",
                    "max_drawdown_pct",
                    "trades",
                    "risk",
                    "tr",
                    "fee_bps",
                    "max_short",
                ]
                trow = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "setting": setting,
                    "model": str(getattr(self.args, "model", "")),
                    "dataset": str(getattr(self.args, "data", getattr(self.args, "dataset", ""))),
                    "seq_len": int(getattr(self.args, "seq_len", 0)),
                    "pred_len": int(getattr(self.args, "pred_len", 0)),
                    "trade_mode": str(mode),
                    "init_balance": float(init_balance),
                    "final_balance": float(final_balance),
                    "total_return_pct": float(total_return * 100.0),
                    "max_drawdown_pct": float(mdd * 100.0),
                    "trades": int(n_trades),
                    "risk": float(risk),
                    "tr": float(tr),
                    "fee_bps": float(fee_bps),
                    "max_short": float(max_short),
                }
                _append_row_csv(trading_csv, trading_fields, trow)

                print(
                    f"[TRADING] mode={mode} final={final_balance:.2f} "
                    f"ret={total_return*100:.2f}% mdd={mdd*100:.2f}% trades={n_trades}"
                )
            except Exception as e:
                print(f"[TRADING] skipped due to error: {repr(e)}")

        return