"""End-to-end pipeline:
   workload -> features -> forecasting models -> auto-scaling simulation -> evaluation
   -> 'what if the prediction is wrong?' experiments -> figures + result tables.

Usage:
    python main.py                       # synthetic 60-day workload, horizon = 3 steps (15 min)
    python main.py --data my_trace.csv   # real trace: columns timestamp, memory_mb[, requests, cpu_pct]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src import plots
from src.autoscaler import ClusterConfig, Hybrid, Oracle, Predictive, Reactive, Static, simulate
from src.data_gen import generate_workload, inject_memory_surges, load_csv
from src.features import build_features, chrono_split
from src.metrics import regression_metrics
from src.models import LEARNED, get_models, quantile_model


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_policies(cfg, demand, pm, pu, future, horizon, static_n, which=None):
    pols = [Static(static_n), Reactive(), Predictive("mean"), Predictive("upper"), Hybrid(), Oracle()]
    if which:
        pols = [p for p in pols if p.name in which]
    return [simulate(p, demand, cfg, pm=pm, pu=pu, future=future, horizon=horizon) for p in pols]


def summary_df(runs) -> pd.DataFrame:
    return pd.DataFrame([r["summary"] for r in runs])


def fit_mean_and_upper(fs, tr_va, seed, model_name="Gradient Boosting"):
    mean = clone(get_models(seed)[model_name]).fit(fs.X.iloc[tr_va], fs.y_delta.iloc[tr_va])
    upper = quantile_model(0.95, seed).fit(fs.X.iloc[tr_va], fs.y_delta.iloc[tr_va])
    return mean, upper


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV with timestamp, memory_mb[, requests, cpu_pct]")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=3, help="prediction horizon in 5-min steps (3 = 15 min)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--instance-mb", type=float, default=4096.0)
    ap.add_argument("--price", type=float, default=0.0416, help="USD per instance-hour")
    ap.add_argument("--boot-steps", type=int, default=3, help="launch+warm-up delay in 5-min steps")
    ap.add_argument("--target-util", type=float, default=0.85)
    args = ap.parse_args()

    t_start = time.time()
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    H = args.horizon
    cfg = ClusterConfig(instance_mb=args.instance_mb, price_per_hour=args.price,
                        boot_steps=args.boot_steps, target_util=args.target_util)

    # 1. data ---------------------------------------------------------------------------
    if args.data:
        df = load_csv(args.data)
        print(f"Loaded {len(df)} rows from {args.data}")
    else:
        df = generate_workload(days=args.days, seed=args.seed)
        df.to_csv("data/workload_synthetic.csv")
        print(f"Generated synthetic workload: {len(df)} rows ({args.days} days @ 5 min)")

    # 2. features / split ------------------------------------------------------------------
    fs = build_features(df, H)
    n = len(fs.X)
    s_tr, s_va, s_te = chrono_split(n)
    s_trva = slice(0, s_va.stop)
    ts_te = fs.X.index[s_te]
    plots.plot_workload(df, fs.X.index[s_va.start], fs.X.index[s_te.start], f"{fig_dir}/01_workload.png")
    print(f"Features: {fs.X.shape[1]} | train {s_tr.stop} / val {s_va.stop - s_va.start} / test {n - s_te.start} rows")

    # 3. model comparison -----------------------------------------------------------------
    models = get_models(args.seed)
    val_rows, test_rows, final = {}, {}, {}
    for name, m in models.items():                       # stage 1: fit on train, score on validation
        m_ = clone(m).fit(fs.X.iloc[s_tr], fs.y_delta.iloc[s_tr])
        pred = fs.mem.iloc[s_va].to_numpy() + m_.predict(fs.X.iloc[s_va])
        val_rows[name] = regression_metrics(fs.y_abs.iloc[s_va], pred)
    val_tbl = pd.DataFrame(val_rows).T
    best = val_tbl.loc[list(LEARNED), "MAE_MB"].idxmin()
    print(f"Best learned model on validation MAE: {best}")

    for name, m in models.items():                       # stage 2: refit on train+val, score on test
        m_ = clone(m).fit(fs.X.iloc[s_trva], fs.y_delta.iloc[s_trva])
        final[name] = m_
        pred = fs.mem.iloc[s_te].to_numpy() + m_.predict(fs.X.iloc[s_te])
        test_rows[name] = regression_metrics(fs.y_abs.iloc[s_te], pred)
    test_tbl = pd.DataFrame(test_rows).T
    test_tbl["skill_vs_persistence_%"] = 100 * (1 - test_tbl["MAE_MB"] / test_tbl.loc["Persistence", "MAE_MB"])
    val_tbl.index.name = test_tbl.index.name = "model"

    mean_model = final[best]
    q_model = quantile_model(0.95, args.seed).fit(fs.X.iloc[s_trva], fs.y_delta.iloc[s_trva])
    mem_te, y_te = fs.mem.iloc[s_te].to_numpy(), fs.y_abs.iloc[s_te].to_numpy()
    pm = mem_te + mean_model.predict(fs.X.iloc[s_te])
    pu = np.maximum(mem_te + q_model.predict(fs.X.iloc[s_te]), pm)
    coverage = float(100 * np.mean(y_te <= pu))
    print(f"q95 upper bound coverage on test: {coverage:.1f}% (target 95%)")
    plots.plot_model_comparison(test_tbl, f"{fig_dir}/02_model_comparison.png")
    plots.plot_prediction(ts_te, y_te, pm, pu, f"{fig_dir}/03_forecast_vs_actual.png")

    rs = np.random.default_rng(args.seed)
    sub = rs.choice(len(fs.X.iloc[s_te]), size=min(1500, len(fs.X.iloc[s_te])), replace=False)
    pi = permutation_importance(mean_model, fs.X.iloc[s_te].iloc[sub], fs.y_delta.iloc[s_te].iloc[sub],
                                scoring="neg_mean_absolute_error", n_repeats=3, random_state=args.seed)
    imp = pd.Series(pi.importances_mean, index=fs.X.columns).sort_values(ascending=False)
    plots.plot_importance(imp, f"{fig_dir}/04_feature_importance.png")

    # 4. auto-scaling simulation on the test period ---------------------------------------------
    static_n = math.ceil(1.2 * fs.mem.iloc[s_tr].max() / cfg.instance_mb)
    runs = run_policies(cfg, mem_te, pm, pu, y_te, H, static_n)
    pol_tbl = summary_df(runs)
    plots.plot_policies(pol_tbl, f"{fig_dir}/05_policy_comparison.png")
    pick = ["Reactive (target tracking)", "Predictive (ML q95)", "Hybrid (ML q95 + guard + fallback)"]
    plots.plot_timeline(ts_te, mem_te, [r for r in runs if r["policy"].name in pick],
                        f"{fig_dir}/06_capacity_timeline.png", start=0, n=576, cfg=cfg)
    hyb = [r for r in runs if isinstance(r["policy"], Hybrid)][0]["policy"]
    diag = {"hybrid_fallback_steps_%": 100 * float(np.mean(hyb.fallback_log)),
            "hybrid_guard_steps_%": 100 * float(np.mean(hyb.guard_log))}

    # 5. what happens when the prediction is wrong? ------------------------------------------------
    wrong = ["Predictive (ML mean)", "Hybrid (ML q95 + guard + fallback)"]
    bias_rows, noise_rows = [], []
    for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4):
        for r in run_policies(cfg, mem_te, pm * f, pu * f, y_te, H, static_n, wrong):
            bias_rows.append({"x": f, "policy": r["summary"]["policy"], **r["summary"]})
    rn = np.random.default_rng(args.seed + 1)
    for s in (0.0, 0.05, 0.10, 0.20, 0.30):
        eps = rn.normal(1.0, s, size=len(pm))
        for r in run_policies(cfg, mem_te, pm * eps, pu * eps, y_te, H, static_n, wrong):
            noise_rows.append({"x": s, "policy": r["summary"]["policy"], **r["summary"]})
    bias_df, noise_df = pd.DataFrame(bias_rows), pd.DataFrame(noise_rows)
    plots.plot_wrong(bias_df, noise_df, f"{fig_dir}/07_wrong_predictions.png")

    # unforeseeable memory surges injected into the test period (model never saw this)
    t0 = df.index.get_loc(ts_te[0])
    surges = [(t0 + int(0.30 * len(ts_te)), 18, 0.40), (t0 + int(0.65 * len(ts_te)), 12, 0.60)]
    df_s = inject_memory_surges(df, surges)
    fs_s = build_features(df_s, H)
    Xs = fs_s.X.loc[ts_te]
    mem_s, y_s = fs_s.mem.loc[ts_te].to_numpy(), fs_s.y_abs.loc[ts_te].to_numpy()
    pm_s = mem_s + mean_model.predict(Xs)
    pu_s = np.maximum(mem_s + q_model.predict(Xs), pm_s)
    shock_runs = run_policies(cfg, mem_s, pm_s, pu_s, y_s, H, static_n)
    shock_tbl = summary_df(shock_runs)
    shyb = [r for r in shock_runs if isinstance(r["policy"], Hybrid)][0]["policy"]
    diag["shock_hybrid_fallback_steps_%"] = 100 * float(np.mean(shyb.fallback_log))
    diag["shock_hybrid_guard_steps_%"] = 100 * float(np.mean(shyb.guard_log))
    diag["shock_forecast_MAPE_%"] = regression_metrics(y_s, pm_s)["MAPE_%"]
    plots.plot_shock(ts_te, mem_s, [r for r in shock_runs if r["policy"].name in pick],
                     f"{fig_dir}/08_shock_scenario.png", start=surges[0][0] - t0 - 36, n=144, cfg=cfg)

    # 5b. target-utilisation sweep: how tightly can each policy pack the cluster? ---------------------------
    sweep_names = ["Reactive (target tracking)", "Predictive (ML mean)", "Predictive (ML q95)",
                   "Hybrid (ML q95 + guard + fallback)", "Oracle (perfect forecast)"]
    sw_rows, sws_rows = [], []
    for tu in (0.70, 0.80, 0.85, 0.90, 0.95):
        c2 = replace(cfg, target_util=tu)
        for r in run_policies(c2, mem_te, pm, pu, y_te, H, static_n, sweep_names):
            sw_rows.append({"target_util": tu, **r["summary"]})
        for r in run_policies(c2, mem_s, pm_s, pu_s, y_s, H, static_n, sweep_names):
            sws_rows.append({"target_util": tu, **r["summary"]})
    sw_df, sws_df = pd.DataFrame(sw_rows), pd.DataFrame(sws_rows)
    plots.plot_util_sweep(sw_df, sws_df, f"{fig_dir}/10_target_util_sweep.png")

    # 5c. does a faster scale-out help when the forecast is wrong? (surge scenario, vary launch delay)
    boot_rows = []
    for b in (1, 2, 3, 6):
        c2 = replace(cfg, boot_steps=b)
        for r in run_policies(c2, mem_s, pm_s, pu_s, y_s, H, static_n, sweep_names):
            boot_rows.append({"boot_min": 5 * b, **r["summary"]})
    boot_df = pd.DataFrame(boot_rows)
    plots.plot_boot_sweep(boot_df, f"{fig_dir}/11_launch_delay_sweep.png")

    # 6. prediction-horizon sweep --------------------------------------------------------------------
    h_rows = []
    for h in (1, 2, 3, 6, 12):
        fh = build_features(df, h)
        a, b, c = chrono_split(len(fh.X))
        mh, qh = fit_mean_and_upper(fh, slice(0, b.stop), args.seed)
        mem_h, y_h = fh.mem.iloc[c].to_numpy(), fh.y_abs.iloc[c].to_numpy()
        pm_h = mem_h + mh.predict(fh.X.iloc[c])
        pu_h = np.maximum(mem_h + qh.predict(fh.X.iloc[c]), pm_h)
        met = regression_metrics(y_h, pm_h)
        r = simulate(Hybrid(), mem_h, cfg, pm=pm_h, pu=pu_h, future=y_h, horizon=h)["summary"]
        h_rows.append({"horizon_steps": h, "horizon_min": 5 * h, "MAE_MB": met["MAE_MB"],
                       "persistence_MAE_MB": float(np.mean(np.abs(y_h - mem_h))), "MAPE_%": met["MAPE_%"],
                       "violation_steps_%": r["violation_steps_%"], "cost_usd": r["cost_usd"],
                       "avg_util_%": r["avg_util_%"]})
    h_df = pd.DataFrame(h_rows)
    plots.plot_horizon(h_df, f"{fig_dir}/09_horizon_sweep.png")

    # 7. latest recommendation (consumed by cloud/aws_apply.py) ---------------------------------------
    live = fs.X_live
    mem_now = float(df["memory_mb"].iloc[-1])
    p_mean = mem_now + float(mean_model.predict(live)[0])
    p_up = max(mem_now + float(q_model.predict(live)[0]), p_mean)
    rec = {"timestamp": str(live.index[0]), "horizon_min": 5 * H, "current_memory_mb": round(mem_now, 1),
           "predicted_memory_mb": round(p_mean, 1), "predicted_upper_mb": round(p_up, 1),
           "instance_mb": cfg.instance_mb, "target_util": cfg.target_util,
           "recommended_instances": int(np.clip(math.ceil(p_up / (cfg.instance_mb * cfg.target_util)),
                                                cfg.min_instances, cfg.max_instances))}

    # 8. save everything -------------------------------------------------------------------------------
    val_tbl.to_csv(f"{args.out}/model_comparison_validation.csv")
    test_tbl.to_csv(f"{args.out}/model_comparison_test.csv")
    pol_tbl.to_csv(f"{args.out}/policy_comparison.csv", index=False)
    shock_tbl.to_csv(f"{args.out}/shock_scenario.csv", index=False)
    bias_df.to_csv(f"{args.out}/wrong_prediction_bias.csv", index=False)
    noise_df.to_csv(f"{args.out}/wrong_prediction_noise.csv", index=False)
    h_df.to_csv(f"{args.out}/horizon_sweep.csv", index=False)
    boot_df.to_csv(f"{args.out}/launch_delay_sweep_surge.csv", index=False)
    sw_df.to_csv(f"{args.out}/target_util_sweep_normal.csv", index=False)
    sws_df.to_csv(f"{args.out}/target_util_sweep_surge.csv", index=False)
    imp.to_csv(f"{args.out}/feature_importance.csv", header=["permutation_importance_MB"])
    with open(f"{args.out}/latest_recommendation.json", "w") as fh_:
        json.dump(rec, fh_, indent=2)
    summ = {"best_model": best, "horizon_min": 5 * H, "q95_coverage_%": coverage, "static_instances": static_n,
            "test_rows": int(len(ts_te)), "diagnostics": diag, "config": cfg.__dict__}
    with open(f"{args.out}/summary.json", "w") as fh_:
        json.dump(summ, fh_, indent=2, default=float)

    pcols = ["policy", "avg_util_%", "violation_steps_%", "oom_episodes", "latency_mean_ms",
             "latency_p99_ms", "cost_usd", "energy_kWh", "avg_instances", "scaling_actions"]
    with open(f"{args.out}/RESULTS.md", "w") as f:
        f.write("# Auto-generated results (python main.py)\n\n")
        f.write(f"Best model: **{best}**, horizon {5 * H} min, q95 coverage {coverage:.1f}%\n\n")
        f.write("## Model comparison (test)\n\n" + md_table(test_tbl) + "\n\n")
        f.write("## Policy comparison (test period)\n\n" + md_table(pol_tbl[pcols].set_index("policy")) + "\n\n")
        f.write("## Unforeseen memory surges injected into test period\n\n"
                + md_table(shock_tbl[pcols].set_index("policy")) + "\n\n")
        scols = ["target_util", "policy", "avg_util_%", "violation_steps_%", "oom_episodes", "latency_p99_ms",
                 "cost_usd", "energy_kWh"]
        f.write("## Target-utilisation sweep, normal test period\n\n"
                + md_table(sw_df[scols].set_index("target_util")) + "\n\n")
        f.write("## Target-utilisation sweep, with unforeseen surges\n\n"
                + md_table(sws_df[scols].set_index("target_util")) + "\n\n")
        f.write("## Launch-delay sweep, with unforeseen surges (target util 85 %)\n\n"
                + md_table(boot_df[["boot_min", "policy", "violation_steps_%", "oom_episodes", "cost_usd"]]
                           .set_index("boot_min")) + "\n\n")
        f.write("## Horizon sweep\n\n" + md_table(h_df.set_index("horizon_steps")) + "\n\n")
        f.write("## Latest recommendation\n\n```json\n" + json.dumps(rec, indent=2) + "\n```\n")

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== Model comparison (test) ===\n", test_tbl.round(2).to_string())
    print("\n=== Policy comparison (test) ===\n", pol_tbl[pcols].round(2).to_string(index=False))
    print("\n=== With unforeseen memory surges ===\n", shock_tbl[pcols].round(2).to_string(index=False))
    print("\n=== Target-util sweep (normal) ===\n", sw_df[["target_util", "policy", "avg_util_%", "violation_steps_%", "cost_usd", "energy_kWh"]].round(2).to_string(index=False))
    print("\n=== Target-util sweep (surges) ===\n", sws_df[["target_util", "policy", "avg_util_%", "violation_steps_%", "cost_usd", "energy_kWh"]].round(2).to_string(index=False))
    print("\n=== Launch-delay sweep (surges) ===\n", boot_df[["boot_min", "policy", "violation_steps_%", "oom_episodes", "cost_usd"]].round(2).to_string(index=False))
    print("\n=== Horizon sweep ===\n", h_df.round(2).to_string(index=False))
    print("\nDiagnostics:", {k: round(v, 2) for k, v in diag.items()})
    print("Recommendation:", rec)
    print(f"\nDone in {time.time() - t_start:.0f}s. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
