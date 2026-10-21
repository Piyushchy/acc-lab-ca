"""Feature engineering. Every feature at row t uses only data up to time t."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12, 36, 72, 144, 288)  # 5 min ... 24 h
ROLL = (6, 12, 36)  # 30 min, 1 h, 3 h
DAY = 288


@dataclass
class FeatureSet:
    X: pd.DataFrame          # features at time t
    y_delta: pd.Series       # memory(t+h) - memory(t)   <- what the model learns
    mem: pd.Series           # memory(t)
    y_abs: pd.Series         # memory(t+h)
    horizon: int
    X_live: pd.DataFrame     # newest feature row (target not yet known) for inference


def build_features(df: pd.DataFrame, horizon: int) -> FeatureSet:
    if not 1 <= horizon < DAY:
        raise ValueError("horizon must be in [1, 287] steps")
    m = df["memory_mb"]
    f: dict[str, pd.Series] = {"mem": m}

    for l in LAGS:
        f[f"mem_d{l}"] = m - m.shift(l)
    for w in ROLL:
        r = m.rolling(w)
        f[f"mem_dev_mean{w}"] = m - r.mean()
        f[f"mem_std{w}"] = r.std()
    f["mem_dev_max36"] = m - m.rolling(36).max()
    # what happened over the same window yesterday (seasonal signal)
    f["seas_delta"] = m.shift(DAY - horizon) - m.shift(DAY)

    if "requests" in df:
        q = df["requests"]
        f["req"] = q
        for l in (1, 3, 6, 12):
            f[f"req_d{l}"] = q - q.shift(l)
        f["req_ema"] = q.ewm(alpha=0.2, adjust=False).mean()
        f["req_mean36"] = q.rolling(36).mean()
    if "cpu_pct" in df:
        c = df["cpu_pct"]
        f["cpu"] = c
        f["cpu_d3"] = c - c.shift(3)

    # calendar features of the *target* time (known in advance)
    tt = df.index + pd.Timedelta(minutes=5 * horizon)
    hr = tt.hour + tt.minute / 60.0
    f["hour_sin"] = pd.Series(np.sin(2 * np.pi * hr / 24), index=df.index)
    f["hour_cos"] = pd.Series(np.cos(2 * np.pi * hr / 24), index=df.index)
    f["dow_sin"] = pd.Series(np.sin(2 * np.pi * tt.dayofweek / 7), index=df.index)
    f["dow_cos"] = pd.Series(np.cos(2 * np.pi * tt.dayofweek / 7), index=df.index)
    f["is_weekend"] = pd.Series((tt.dayofweek >= 5).astype(float), index=df.index)

    frame = pd.DataFrame(f)
    y_abs = m.shift(-horizon)
    ok_x = frame.notna().all(axis=1)
    ok = ok_x & y_abs.notna()
    return FeatureSet(
        X=frame[ok],
        y_delta=(y_abs - m)[ok],
        mem=m[ok],
        y_abs=y_abs[ok],
        horizon=horizon,
        X_live=frame[ok_x].iloc[[-1]],
    )


def chrono_split(n: int, train: float = 0.6, val: float = 0.2) -> tuple[slice, slice, slice]:
    """Chronological split (no shuffling: avoids look-ahead leakage)."""
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)
