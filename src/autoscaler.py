"""Resource allocation / auto-scaling policies and a discrete-time cluster simulator.

Time step = 5 minutes. At every step t the simulator:
  1. brings instances online whose launch finished (boot + warm-up delay),
  2. scores the step: utilisation, latency proxy, cost, energy,
  3. asks the policy how many instances it wants and launches / terminates.

Scale-out is immediate (but capacity only arrives ``boot_steps`` later);
scale-in is stabilised: the cluster only shrinks to the highest request seen in
the last ``scale_in_window`` steps (anti-flapping, like an autoscaling cooldown).
The same stabilisation is applied to every dynamic policy for a fair comparison.

Cost / power numbers are ASSUMPTIONS (documented in the README), roughly an
on-demand 4 GB general purpose VM; change them in ClusterConfig.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass
class ClusterConfig:
    instance_mb: float = 4096.0        # memory per instance
    price_per_hour: float = 0.0416     # USD / instance-hour (assumed, ~t3.medium on-demand)
    boot_steps: int = 3                # 15 min: launch + app warm-up + health check
    step_minutes: int = 5
    min_instances: int = 2
    max_instances: int = 80
    target_util: float = 0.85          # sizing target for all dynamic policies (swept 0.70-0.95 in main.py)
    scale_in_window: int = 6           # 30 min
    p_idle_w: float = 45.0             # per-instance power model (assumed)
    p_max_w: float = 110.0
    base_latency_ms: float = 40.0
    timeout_latency_ms: float = 2000.0  # latency when memory is exhausted (swap / OOM kill)


# ----------------------------------------------------------------------------- policies
class Static:
    dynamic = False

    def __init__(self, n: int, name: str = "Static (peak-sized)"):
        self.n, self.name = n, name

    def desired(self, t, S):
        return self.n


class Reactive:
    """Threshold / target-tracking on *current* usage (what AWS target tracking does)."""
    dynamic = True
    name = "Reactive (target tracking)"

    def desired(self, t, S):
        return math.ceil(S.demand[t] / (S.cfg.instance_mb * S.cfg.target_util))


class Predictive:
    """Size the cluster for the forecast demand at t+h (mean or upper-quantile)."""
    dynamic = True

    def __init__(self, mode: str = "mean"):
        self.mode = mode
        self.name = {"mean": "Predictive (ML mean)", "upper": "Predictive (ML q95)"}[mode]

    def desired(self, t, S):
        pred = S.pm[t] if self.mode == "mean" else S.pu[t]
        return math.ceil(pred / (S.cfg.instance_mb * S.cfg.target_util))


class Hybrid:
    """ML upper-bound forecast + safety nets for when the forecast is wrong:

    * emergency guard  : if current utilisation > target_util + ``emergency_margin`` (cap 0.97)
                         scale to current demand + surge
    * drift fallback   : if the recent realised forecast error (MAPE over ``drift_win`` steps)
                         exceeds ``drift_thr`` the model is not trusted -> reactive sizing with extra headroom
    """
    dynamic = True
    name = "Hybrid (ML q95 + guard + fallback)"

    def __init__(self, emergency_margin=0.08, surge=0.25, drift_thr=0.12, drift_win=12, fallback_headroom=0.30):
        self.emergency_margin, self.surge = emergency_margin, surge
        self.drift_thr, self.drift_win, self.fb_headroom = drift_thr, drift_win, fallback_headroom
        self.fallback_log: list[bool] = []
        self.guard_log: list[bool] = []

    def desired(self, t, S):
        cfg, h = S.cfg, S.horizon
        per = cfg.instance_mb * cfg.target_util
        D = S.demand[t]

        # realised error of the forecasts made h steps ago
        idx = np.arange(max(t - self.drift_win + 1, h), t + 1)
        mape = float(np.mean(np.abs(S.pm[idx - h] - S.demand[idx]) / S.demand[idx])) if len(idx) >= 3 else 0.0
        fallback = mape > self.drift_thr
        need = max(S.pu[t], D * (1 + self.fb_headroom)) if fallback else S.pu[t]

        guard = D / (S.ready * cfg.instance_mb) > min(0.97, cfg.target_util + self.emergency_margin)
        if guard:
            need = max(need, D * (1 + self.surge))
        self.fallback_log.append(fallback)
        self.guard_log.append(guard)
        return math.ceil(need / per)


class Oracle:
    """Perfect forecast of demand at t+h (upper bound on what prediction can achieve)."""
    dynamic = True
    name = "Oracle (perfect forecast)"

    def desired(self, t, S):
        return math.ceil(S.future[t] / (S.cfg.instance_mb * S.cfg.target_util))


# ----------------------------------------------------------------------------- simulator
def latency_ms(u: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """Queueing-style latency proxy: L0 / (1 - u), capped at 20*L0 below saturation;
    when memory is exhausted (u >= 1) requests time out / hit swap."""
    lat = cfg.base_latency_ms / (1.0 - np.minimum(u, 0.95))
    return np.where(u >= 1.0, cfg.timeout_latency_ms, lat)


def simulate(policy, demand, cfg: ClusterConfig, pm=None, pu=None, future=None, horizon: int = 3,
             initial: int | None = None) -> dict:
    """Run one policy over a demand series (MB). Returns per-step arrays + summary."""
    demand = np.asarray(demand, float)
    n = len(demand)
    pm = demand if pm is None else np.asarray(pm, float)
    pu = pm if pu is None else np.asarray(pu, float)
    future = demand if future is None else np.asarray(future, float)

    if policy.dynamic:
        start = math.ceil(demand[0] / (cfg.instance_mb * cfg.target_util)) if initial is None else initial
    else:
        start = policy.n
    start = int(np.clip(start, cfg.min_instances, cfg.max_instances))

    S = SimpleNamespace(demand=demand, pm=pm, pu=pu, future=future, cfg=cfg, horizon=horizon,
                        ready=start, total=start)
    pending: list[int] = []
    raw_hist: list[int] = []
    ready_arr = np.zeros(n, int)
    billed_arr = np.zeros(n, int)
    actions = 0

    for t in range(n):
        while pending and pending[0] <= t:
            pending.pop(0)
            S.ready += 1
        ready_arr[t] = S.ready
        billed_arr[t] = S.ready + len(pending)

        if policy.dynamic:
            S.total = S.ready + len(pending)
            raw = int(np.clip(policy.desired(t, S), cfg.min_instances, cfg.max_instances))
            raw_hist.append(raw)
            if raw > S.total:                                   # scale out (arrives after boot delay)
                pending.extend([t + cfg.boot_steps] * (raw - S.total))
                actions += 1
            else:                                               # stabilised scale in
                target = max(raw_hist[-cfg.scale_in_window:])
                remove = S.total - target
                if remove > 0:
                    actions += 1
                    while remove > 0 and pending:               # cancel not-yet-ready launches first
                        pending.pop()
                        remove -= 1
                    S.ready = max(cfg.min_instances, S.ready - remove)

    cap = ready_arr * cfg.instance_mb
    u = demand / cap
    lat = latency_ms(u, cfg)
    step_h = cfg.step_minutes / 60.0
    power_w = ready_arr * (cfg.p_idle_w + (cfg.p_max_w - cfg.p_idle_w) * np.minimum(u, 1.0)) \
        + (billed_arr - ready_arr) * cfg.p_idle_w
    viol = u > 1.0
    episodes = int(np.sum(viol[1:] & ~viol[:-1]) + (1 if viol[0] else 0))
    summary = {
        "policy": policy.name,
        "avg_util_%": float(100 * np.mean(np.minimum(u, 1.0))),
        "violation_steps_%": float(100 * np.mean(viol)),
        "oom_episodes": episodes,
        "near_saturation_%": float(100 * np.mean(u > 0.9)),
        "shortfall_GB_h": float(np.sum(np.maximum(demand - cap, 0)) / 1024 * step_h),
        "latency_mean_ms": float(np.mean(lat)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_p99_ms": float(np.percentile(lat, 99)),
        "cost_usd": float(np.sum(billed_arr) * cfg.price_per_hour * step_h),
        "energy_kWh": float(np.sum(power_w) * step_h / 1000.0),
        "avg_instances": float(np.mean(ready_arr)),
        "scaling_actions": actions,
    }
    return {"summary": summary, "ready": ready_arr, "billed": billed_arr, "util": u, "latency": lat,
            "policy": policy}
