import math

import numpy as np

from src.autoscaler import ClusterConfig, Hybrid, Reactive, Static, simulate
from src.data_gen import generate_workload, inject_memory_surges
from src.features import build_features, chrono_split


def small_df():
    return generate_workload(days=12, seed=1)


def test_target_alignment():
    df = small_df()
    fs = build_features(df, 3)
    ts = fs.X.index[100]
    i = df.index.get_loc(ts)
    assert fs.mem.loc[ts] == df["memory_mb"].iloc[i]
    assert fs.y_abs.loc[ts] == df["memory_mb"].iloc[i + 3]
    assert math.isclose(fs.y_delta.loc[ts], df["memory_mb"].iloc[i + 3] - df["memory_mb"].iloc[i])


def test_no_future_leakage_in_features():
    df = small_df()
    fs = build_features(df, 3)
    ts = fs.X.index[1500]
    i = df.index.get_loc(ts)
    df2 = df.copy()
    df2.iloc[i + 1:, :] = df2.iloc[i + 1:, :] * 3 + 100  # corrupt everything after t
    fs2 = build_features(df2, 3)
    assert np.allclose(fs.X.loc[ts].values, fs2.X.loc[ts].values)


def test_chrono_split_is_ordered_and_disjoint():
    a, b, c = chrono_split(1000)
    assert a.stop == b.start and b.stop == c.start and c.stop == 1000


def test_static_policy_constant_capacity_and_cost():
    cfg = ClusterConfig()
    demand = np.full(288, 10000.0)
    r = simulate(Static(5), demand, cfg)
    assert (r["ready"] == 5).all()
    assert math.isclose(r["summary"]["cost_usd"], 5 * cfg.price_per_hour * 24, rel_tol=1e-9)


def test_launch_delay_causes_temporary_shortfall_for_reactive():
    cfg = ClusterConfig(target_util=0.8, boot_steps=3, min_instances=2)
    demand = np.r_[np.full(30, 6000.0), np.full(30, 20000.0)]  # sudden 3x jump at t=30
    r = simulate(Reactive(), demand, cfg)
    assert r["util"][30] > 1.0                    # capacity has not arrived yet
    assert r["util"][30 + cfg.boot_steps + 1] < 1.0  # ...but it does after the launch delay
    assert (r["ready"] >= cfg.min_instances).all()


def test_hybrid_falls_back_when_forecast_is_badly_biased():
    cfg = ClusterConfig()
    t = np.arange(400)
    demand = 12000 + 3000 * np.sin(t / 30)
    biased = demand * 0.6                          # forecast 40 % too low
    h = Hybrid()
    r = simulate(h, demand, cfg, pm=biased, pu=biased, future=demand, horizon=3)
    assert any(h.fallback_log)
    assert r["summary"]["violation_steps_%"] < 5.0


def test_surge_injection_changes_only_memory():
    df = small_df()
    s = inject_memory_surges(df, [(1000, 10, 0.5)])
    assert (s["requests"] == df["requests"]).all()
    assert s["memory_mb"].iloc[1005] > df["memory_mb"].iloc[1005] * 1.4
    assert s["memory_mb"].iloc[1015] == df["memory_mb"].iloc[1015]
