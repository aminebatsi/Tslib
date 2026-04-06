# utils/trading.py
import numpy as np


def buy_sell_smart(today, pred, balance, shares, risk=5.0):
    diff = pred * risk / 100.0
    if diff <= 0:
        return balance, shares

    if today > pred + diff:
        balance += shares * today
        shares = 0.0
    elif today > pred:
        factor = (today - pred) / diff
        factor = float(np.clip(factor, 0.0, 1.0))
        balance += shares * factor * today
        shares *= (1.0 - factor)
    elif today > pred - diff:
        factor = (pred - today) / diff
        factor = float(np.clip(factor, 0.0, 1.0))
        shares += balance * factor / today
        balance *= (1.0 - factor)
    else:
        shares += balance / today
        balance = 0.0

    return balance, shares


def buy_sell_smart_w_short(today, pred, balance, shares, risk=5.0, max_short=0.002):
    diff = pred * risk / 100.0
    if diff <= 0:
        return balance, shares

    if today < pred - diff:
        shares += balance / today
        balance = 0.0
    elif today < pred:
        factor = (pred - today) / diff
        factor = float(np.clip(factor, 0.0, 1.0))
        shares += balance * factor / today
        balance *= (1.0 - factor)
    elif today < pred + diff:
        if shares > 0:
            factor = (today - pred) / diff
            factor = float(np.clip(factor, 0.0, 1.0))
            balance += shares * factor * today
            shares *= (1.0 - factor)
    else:
        balance += (shares + max_short) * today
        shares = -max_short

    return balance, shares


def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    tmp = abs((pred - today) / max(today, 1e-12))
    if tmp < tr:
        return balance, shares

    if pred > today:
        shares += balance / today
        balance = 0.0
    else:
        balance += shares * today
        shares = 0.0
    return balance, shares


def max_drawdown(equity_curve):
    x = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(x)
    dd = (x - peak) / np.where(peak == 0, 1.0, peak)
    return float(-dd.min())


def simulate_trade(
    today_prices,
    pred_next_prices,
    init_balance=100.0,
    mode="smart",
    risk=5.0,
    tr=0.01,
    fee_bps=0.0,
    max_short=0.002,
):
    balance = float(init_balance)
    shares = 0.0
    equity = [balance]
    trades = 0

    today_prices = np.asarray(today_prices, dtype=float)
    pred_next_prices = np.asarray(pred_next_prices, dtype=float)

    for today, pred in zip(today_prices, pred_next_prices):
        if today <= 0:
            equity.append(balance + shares * max(today, 0.0))
            continue

        bal_before = balance
        sh_before = shares

        if mode == "smart":
            balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
        elif mode == "smart_w_short":
            balance, shares = buy_sell_smart_w_short(today, pred, balance, shares, risk=risk, max_short=max_short)
        elif mode == "vanilla":
            balance, shares = buy_sell_vanilla(today, pred, balance, shares, tr=tr)
        elif mode == "no_strategy":
            shares += balance / today
            balance = 0.0
        else:
            raise ValueError(f"Unknown trade mode: {mode}")

        if fee_bps > 0 and ((balance != bal_before) or (shares != sh_before)):
            trades += 1
            notional = abs((shares - sh_before) * today)
            balance -= notional * (fee_bps / 10000.0)

        equity.append(balance + shares * today)

    if len(today_prices) > 0:
        last_price = float(today_prices[-1])
        balance = balance + shares * last_price

    return balance, equity, trades