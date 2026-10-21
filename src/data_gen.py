"""Synthetic cloud workload generator (5-minute resolution).

The generator produces a realistic-looking aggregate workload for one service:

* requests/s : diurnal + weekly seasonality, slow growth trend, AR(1) noise and
               random traffic bursts
* cpu_pct    : roughly proportional to the request rate
* memory_mb  : OS/cache baseline + a *lagged* (EMA) response to the request rate
               (caches/connection state fill up gradually) + a slow memory leak
               that resets on deployment/restart + noise

IMPORTANT: this is synthetic data. It embeds the assumption that the request
rate *leads* memory usage. To use a real trace instead (Google cluster data,
Alibaba, Azure, or metrics collected by cloud/collect_metrics.py), pass
``--data your.csv`` to main.py (see README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP_MIN = 5
STEPS_PER_DAY = 24 * 60 // STEP_MIN  # 288


def generate_workload(days: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * STEPS_PER_DAY
    ts = pd.date_range("2026-01-05", periods=n, freq=f"{STEP_MIN}min")  # a Monday
    hour = ts.hour.values + ts.minute.values / 60.0
    dow = ts.dayofweek.values

    # --- request rate -------------------------------------------------------
    diurnal = 0.55 * np.sin(2 * np.pi * (hour - 8) / 24) + 0.20 * np.sin(4 * np.pi * (hour - 9) / 24)
    weekly = np.where(dow >= 5, 0.75, 1.0)
    trend = 1.0 + 0.12 * np.arange(n) / n  # 12 % growth over the period
    base = 1000.0 * weekly * trend * (1.0 + diurnal)

    noise = np.zeros(n)
    for i in range(1, n):  # AR(1) multiplicative noise
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.03)

    bursts = np.zeros(n)
    n_events = int(days * 1.0)  # ~1 burst per day
    for s in rng.integers(0, n - 40, n_events):
        mag = rng.uniform(150, 450)
        tau = rng.uniform(6, 18)
        k = np.arange(0, 40)
        bursts[s : s + 40] += mag * np.exp(-k / tau)

    requests = np.clip(base * (1.0 + noise) + bursts, 50, None)

    # --- cpu ----------------------------------------------------------------
    cpu = np.clip(8 + 0.028 * requests + rng.normal(0, 2.0, n), 1, 100)

    # --- memory -------------------------------------------------------------
    ema = pd.Series(requests).ewm(alpha=0.2, adjust=False).mean().values
    leak = np.zeros(n)
    t0 = 0
    while t0 < n:  # leak resets on each "deployment"
        period = int(rng.integers(400, 700))
        k = np.arange(min(period, n - t0))
        leak[t0 : t0 + len(k)] = 2.5 * k
        t0 += period
    mem_noise = np.zeros(n)
    for i in range(1, n):
        mem_noise[i] = 0.8 * mem_noise[i - 1] + rng.normal(0, 60)
    memory = 9000.0 + 14.0 * ema + leak + mem_noise
    memory = np.clip(memory, 1000, None)

    return pd.DataFrame(
        {"requests": requests, "cpu_pct": cpu, "memory_mb": memory},
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def inject_memory_surges(df: pd.DataFrame, surges: list[tuple[int, int, float]]) -> pd.DataFrame:
    """Return a copy with *unforeseeable* memory surges (cache stampede, leak, bad deploy).

    surges: list of (start_row, duration_steps, fraction_of_baseline). The surge
    ramps up in 2 steps and drops off after ``duration``. Requests are NOT changed,
    so a model that relies on request rate cannot anticipate it.
    """
    out = df.copy()
    base = df["memory_mb"].values
    extra = np.zeros(len(df))
    for start, dur, frac in surges:
        for k in range(dur):
            if start + k < len(df):
                extra[start + k] += frac * base[start + k] * min(1.0, (k + 1) / 2)
    out["memory_mb"] = base + extra
    return out


def load_csv(path: str) -> pd.DataFrame:
    """Load a real trace. Needs a 'timestamp' and 'memory_mb' column; 'requests'
    and 'cpu_pct' are optional. Data is resampled to 5-minute means."""
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if "memory_mb" not in df:
        raise ValueError("CSV must contain a 'memory_mb' column")
    keep = [c for c in ("requests", "cpu_pct", "memory_mb") if c in df]
    df = df[keep].resample(f"{STEP_MIN}min").mean().interpolate(limit=6).dropna()
    return df
