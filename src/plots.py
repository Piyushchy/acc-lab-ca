"""Matplotlib figures used in the report (saved as PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "Static (peak-sized)": "#7f7f7f",
    "Reactive (target tracking)": "#d62728",
    "Predictive (ML mean)": "#ff7f0e",
    "Predictive (ML q95)": "#9467bd",
    "Hybrid (ML q95 + guard + fallback)": "#2ca02c",
    "Oracle (perfect forecast)": "#1f77b4",
}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_workload(df, t_val, t_test, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(df.index, df["memory_mb"] / 1024, lw=0.5, color="#1f77b4")
    ax[0].set_ylabel("Memory demand (GB)")
    ax[0].set_title("Workload: memory demand and request rate (train | validation | test)")
    if "requests" in df:
        ax[1].plot(df.index, df["requests"], lw=0.5, color="#ff7f0e")
    ax[1].set_ylabel("Requests / s")
    for a in ax:
        a.axvline(t_val, color="k", ls="--", lw=0.8)
        a.axvline(t_test, color="k", ls="--", lw=0.8)
    _save(fig, path)


def plot_model_comparison(tbl, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    t = tbl.sort_values("MAE_MB", ascending=False)
    ax[0].barh(t.index, t["MAE_MB"], color="#1f77b4")
    ax[0].set_xlabel("MAE (MB)  - lower is better")
    ax[0].set_title("Test MAE by model")
    ax[1].barh(t.index, t["MAPE_%"], color="#ff7f0e")
    ax[1].set_xlabel("MAPE (%)")
    ax[1].set_title("Test MAPE by model")
    _save(fig, path)


def plot_prediction(ts, actual, pred, upper, path, n=576):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(ts[:n], actual[:n] / 1024, label="actual memory (t+h)", color="k", lw=1)
    ax.plot(ts[:n], pred[:n] / 1024, label="forecast (mean)", color="#ff7f0e", lw=1)
    ax.plot(ts[:n], upper[:n] / 1024, label="forecast (q95 upper bound)", color="#9467bd", lw=0.9, ls="--")
    ax.set_ylabel("GB")
    ax.set_title("Forecast vs actual (first 2 days of the test set)")
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_importance(imp, path, top=12):
    imp = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values, color="#2ca02c")
    ax.set_xlabel("Permutation importance (increase in MAE, MB)")
    ax.set_title("Top features")
    _save(fig, path)


def plot_policies(tbl, path):
    metrics = [("cost_usd", "Cost (USD)"), ("avg_util_%", "Avg memory utilisation (%)"),
               ("violation_steps_%", "Steps with memory exhausted (%)"),
               ("latency_p99_ms", "p99 latency proxy (ms)"), ("energy_kWh", "Energy (kWh)")]
    fig, ax = plt.subplots(1, len(metrics), figsize=(16, 3.8))
    for a, (col, title) in zip(ax, metrics):
        names = list(tbl["policy"])
        a.bar(range(len(names)), tbl[col], color=[COLORS.get(n, "#333") for n in names])
        a.set_xticks(range(len(names)))
        a.set_xticklabels([n.split(" (")[0] for n in names], rotation=60, ha="right", fontsize=7)
        a.set_title(title, fontsize=9)
    _save(fig, path)


def plot_timeline(ts, demand, runs, path, start=0, n=576, cfg=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sl = slice(start, start + n)
    ax[0].plot(ts[sl], demand[sl] / 1024, color="k", lw=1.2, label="memory demand")
    for r in runs:
        nm = r["policy"].name
        ax[0].step(ts[sl], r["ready"][sl] * cfg.instance_mb / 1024, where="post", lw=1.1,
                   color=COLORS.get(nm), label=f"capacity: {nm}")
        ax[1].plot(ts[sl], np.minimum(r["util"][sl], 1.3) * 100, lw=0.9, color=COLORS.get(nm), label=nm)
    ax[0].set_ylabel("GB")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("Demand vs provisioned capacity")
    ax[1].axhline(100, color="r", ls=":", lw=1)
    ax[1].set_ylabel("Utilisation (%)")
    ax[1].set_title("Memory utilisation (>100 % = memory exhausted)")
    _save(fig, path)


def plot_wrong(bias_df, noise_df, path):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for pol, g in bias_df.groupby("policy"):
        c = COLORS.get(pol)
        ax[0, 0].plot(g["x"], g["violation_steps_%"], marker="o", color=c, label=pol)
        ax[0, 1].plot(g["x"], g["cost_usd"], marker="o", color=c, label=pol)
    for pol, g in noise_df.groupby("policy"):
        c = COLORS.get(pol)
        ax[1, 0].plot(g["x"], g["violation_steps_%"], marker="o", color=c, label=pol)
        ax[1, 1].plot(g["x"], g["cost_usd"], marker="o", color=c, label=pol)
    ax[0, 0].set_title("Systematic bias in forecast: SLA violations")
    ax[0, 1].set_title("Systematic bias in forecast: cost")
    ax[1, 0].set_title("Random forecast noise: SLA violations")
    ax[1, 1].set_title("Random forecast noise: cost")
    for a in ax[0]:
        a.set_xlabel("forecast multiplier (1.0 = unbiased)")
    for a in ax[1]:
        a.set_xlabel("noise std (fraction of forecast)")
    ax[0, 0].set_ylabel("% steps memory exhausted")
    ax[1, 0].set_ylabel("% steps memory exhausted")
    ax[0, 1].set_ylabel("USD")
    ax[1, 1].set_ylabel("USD")
    ax[0, 0].legend(fontsize=7)
    _save(fig, path)


def plot_shock(ts, demand, runs, path, start, n, cfg):
    plot_timeline(ts, demand, runs, path, start=start, n=n, cfg=cfg)


def plot_horizon(h_df, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    x = h_df["horizon_min"]
    ax[0].plot(x, h_df["MAE_MB"], marker="o", label="GB model")
    ax[0].plot(x, h_df["persistence_MAE_MB"], marker="s", ls="--", color="gray", label="persistence")
    ax[0].set_title("Forecast error vs horizon")
    ax[0].set_xlabel("horizon (min)")
    ax[0].set_ylabel("MAE (MB)")
    ax[0].legend()
    ax[1].plot(x, h_df["violation_steps_%"], marker="o", color="#d62728")
    ax[1].set_title("SLA violations (hybrid policy)")
    ax[1].set_xlabel("horizon (min)")
    ax[1].set_ylabel("% steps memory exhausted")
    ax[2].plot(x, h_df["cost_usd"], marker="o", color="#2ca02c")
    ax[2].set_title("Cost (hybrid policy)")
    ax[2].set_xlabel("horizon (min)")
    ax[2].set_ylabel("USD")
    _save(fig, path)


def plot_util_sweep(normal, surge, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    for row, (lab, d) in enumerate((("normal test period", normal), ("with unforeseen memory surges", surge))):
        for pol, g in d.groupby("policy", sort=False):
            c = COLORS.get(pol)
            ax[row, 0].plot(g["target_util"] * 100, g["violation_steps_%"], marker="o", color=c, label=pol)
            ax[row, 1].plot(g["target_util"] * 100, g["cost_usd"], marker="o", color=c, label=pol)
        ax[row, 0].set_title(f"SLA violations vs target utilisation ({lab})", fontsize=8.5)
        ax[row, 1].set_title(f"Cost vs target utilisation ({lab})", fontsize=8.5)
        ax[row, 0].set_ylabel("% steps memory exhausted")
        ax[row, 1].set_ylabel("USD")
    for a in ax[1]:
        a.set_xlabel("target utilisation used for sizing (%)")
    ax[0, 0].legend(fontsize=7)
    _save(fig, path)


def plot_boot_sweep(d, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for pol, g in d.groupby("policy", sort=False):
        c = COLORS.get(pol)
        ax[0].plot(g["boot_min"], g["violation_steps_%"], marker="o", color=c, label=pol)
        ax[1].plot(g["boot_min"], g["cost_usd"], marker="o", color=c, label=pol)
    ax[0].set_title("Surge scenario: SLA violations vs launch delay", fontsize=9)
    ax[1].set_title("Cost vs launch delay", fontsize=9)
    ax[0].set_ylabel("% steps memory exhausted")
    ax[1].set_ylabel("USD")
    for a in ax:
        a.set_xlabel("instance launch + warm-up delay (min)")
    ax[0].legend(fontsize=7)
    _save(fig, path)
