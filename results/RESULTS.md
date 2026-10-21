# Auto-generated results (python main.py)

Best model: **MLP (64-32)**, horizon 15 min, q95 coverage 90.8%

## Model comparison (test)

| model | MAE_MB | RMSE_MB | MAPE_% | R2 | under_predicted_% | skill_vs_persistence_% |
|---|---|---|---|---|---|---|
| Persistence | 497.00 | 648.12 | 2.01 | 0.99 | 44.97 | 0.00 |
| Moving average (1h) | 1249.48 | 1575.57 | 5.05 | 0.94 | 42.67 | -151.40 |
| Seasonal naive (24h) | 436.76 | 597.49 | 1.75 | 0.99 | 50.06 | 12.12 |
| Ridge regression | 256.02 | 367.40 | 1.02 | 1.00 | 56.15 | 48.49 |
| Random Forest | 259.50 | 372.03 | 1.03 | 1.00 | 52.03 | 47.79 |
| Gradient Boosting | 260.53 | 371.84 | 1.03 | 1.00 | 53.47 | 47.58 |
| MLP (64-32) | 252.63 | 361.91 | 1.01 | 1.00 | 54.03 | 49.17 |

## Policy comparison (test period)

| policy | avg_util_% | violation_steps_% | oom_episodes | latency_mean_ms | latency_p99_ms | cost_usd | energy_kWh | avg_instances | scaling_actions |
|---|---|---|---|---|---|---|---|---|---|
| Static (peak-sized) | 50.30 | 0.00 | 0.00 | 88.65 | 194.11 | 141.36 | 264.00 | 12.00 | 0.00 |
| Reactive (target tracking) | 78.56 | 0.03 | 1.00 | 195.61 | 332.56 | 90.95 | 209.47 | 7.67 | 157.00 |
| Predictive (ML mean) | 78.53 | 0.00 | 0.00 | 192.32 | 278.76 | 90.95 | 209.48 | 7.67 | 161.00 |
| Predictive (ML q95) | 77.00 | 0.00 | 0.00 | 178.67 | 255.35 | 92.71 | 211.38 | 7.82 | 159.00 |
| Hybrid (ML q95 + guard + fallback) | 76.99 | 0.00 | 0.00 | 178.59 | 254.22 | 92.72 | 211.39 | 7.82 | 158.00 |
| Oracle (perfect forecast) | 78.49 | 0.00 | 0.00 | 191.17 | 262.71 | 90.98 | 209.50 | 7.68 | 157.00 |

## Unforeseen memory surges injected into test period

| policy | avg_util_% | violation_steps_% | oom_episodes | latency_mean_ms | latency_p99_ms | cost_usd | energy_kWh | avg_instances | scaling_actions |
|---|---|---|---|---|---|---|---|---|---|
| Static (peak-sized) | 50.50 | 0.00 | 0.00 | 89.33 | 195.19 | 141.36 | 264.46 | 12.00 | 0.00 |
| Reactive (target tracking) | 78.52 | 0.18 | 3.00 | 198.37 | 340.60 | 91.42 | 210.40 | 7.71 | 164.00 |
| Predictive (ML mean) | 78.52 | 0.15 | 2.00 | 195.33 | 282.63 | 91.38 | 210.36 | 7.71 | 166.00 |
| Predictive (ML q95) | 76.97 | 0.15 | 2.00 | 181.39 | 260.44 | 93.18 | 212.31 | 7.86 | 165.00 |
| Hybrid (ML q95 + guard + fallback) | 76.83 | 0.15 | 2.00 | 180.63 | 258.79 | 93.49 | 212.64 | 7.88 | 168.00 |
| Oracle (perfect forecast) | 78.46 | 0.00 | 0.00 | 191.13 | 262.71 | 91.44 | 210.46 | 7.71 | 164.00 |

## Target-utilisation sweep, normal test period

| target_util | policy | avg_util_% | violation_steps_% | oom_episodes | latency_p99_ms | cost_usd | energy_kWh |
|---|---|---|---|---|---|---|---|
| 0.7 | Reactive (target tracking) | 65.44 | 0.00 | 0 | 146.44 | 109.14 | 229.16 |
| 0.7 | Predictive (ML mean) | 65.49 | 0.00 | 0 | 137.01 | 109.05 | 229.06 |
| 0.7 | Predictive (ML q95) | 64.11 | 0.00 | 0 | 130.88 | 111.34 | 231.53 |
| 0.7 | Hybrid (ML q95 + guard + fallback) | 64.11 | 0.00 | 0 | 130.88 | 111.34 | 231.53 |
| 0.7 | Oracle (perfect forecast) | 65.38 | 0.00 | 0 | 132.57 | 109.18 | 229.19 |
| 0.8 | Reactive (target tracking) | 74.18 | 0.00 | 0 | 233.43 | 96.21 | 215.16 |
| 0.8 | Predictive (ML mean) | 74.09 | 0.00 | 0 | 206.43 | 96.30 | 215.27 |
| 0.8 | Predictive (ML q95) | 72.72 | 0.00 | 0 | 194.59 | 98.12 | 217.23 |
| 0.8 | Hybrid (ML q95 + guard + fallback) | 72.72 | 0.00 | 0 | 194.59 | 98.12 | 217.23 |
| 0.8 | Oracle (perfect forecast) | 74.13 | 0.00 | 0 | 197.67 | 96.23 | 215.19 |
| 0.85 | Reactive (target tracking) | 78.56 | 0.03 | 1 | 332.56 | 90.95 | 209.47 |
| 0.85 | Predictive (ML mean) | 78.53 | 0.00 | 0 | 278.76 | 90.95 | 209.48 |
| 0.85 | Predictive (ML q95) | 77.00 | 0.00 | 0 | 255.35 | 92.71 | 211.38 |
| 0.85 | Hybrid (ML q95 + guard + fallback) | 76.99 | 0.00 | 0 | 254.22 | 92.72 | 211.39 |
| 0.85 | Oracle (perfect forecast) | 78.49 | 0.00 | 0 | 262.71 | 90.98 | 209.50 |
| 0.9 | Reactive (target tracking) | 82.92 | 0.06 | 1 | 563.35 | 86.04 | 204.16 |
| 0.9 | Predictive (ML mean) | 82.85 | 0.00 | 0 | 424.52 | 86.12 | 204.25 |
| 0.9 | Predictive (ML q95) | 81.27 | 0.00 | 0 | 366.65 | 87.78 | 206.04 |
| 0.9 | Hybrid (ML q95 + guard + fallback) | 81.27 | 0.00 | 0 | 366.65 | 87.78 | 206.04 |
| 0.9 | Oracle (perfect forecast) | 82.85 | 0.00 | 0 | 388.66 | 86.07 | 204.20 |
| 0.95 | Reactive (target tracking) | 86.85 | 0.24 | 7 | 800.00 | 82.14 | 199.95 |
| 0.95 | Predictive (ML mean) | 86.73 | 0.03 | 1 | 800.00 | 82.26 | 200.08 |
| 0.95 | Predictive (ML q95) | 85.42 | 0.03 | 1 | 660.27 | 83.50 | 201.42 |
| 0.95 | Hybrid (ML q95 + guard + fallback) | 85.37 | 0.03 | 1 | 654.33 | 83.58 | 201.51 |
| 0.95 | Oracle (perfect forecast) | 86.79 | 0.00 | 0 | 746.66 | 82.16 | 199.97 |

## Target-utilisation sweep, with unforeseen surges

| target_util | policy | avg_util_% | violation_steps_% | oom_episodes | latency_p99_ms | cost_usd | energy_kWh |
|---|---|---|---|---|---|---|---|
| 0.7 | Reactive (target tracking) | 65.44 | 0.09 | 1 | 148.07 | 109.67 | 230.17 |
| 0.7 | Predictive (ML mean) | 65.46 | 0.09 | 1 | 137.67 | 109.62 | 230.12 |
| 0.7 | Predictive (ML q95) | 64.09 | 0.09 | 1 | 131.51 | 111.92 | 232.60 |
| 0.7 | Hybrid (ML q95 + guard + fallback) | 63.99 | 0.09 | 1 | 131.43 | 112.26 | 232.97 |
| 0.7 | Oracle (perfect forecast) | 65.37 | 0.00 | 0 | 132.58 | 109.70 | 230.22 |
| 0.8 | Reactive (target tracking) | 74.14 | 0.09 | 1 | 236.81 | 96.73 | 216.16 |
| 0.8 | Predictive (ML mean) | 74.06 | 0.15 | 2 | 210.14 | 96.78 | 216.21 |
| 0.8 | Predictive (ML q95) | 72.68 | 0.09 | 1 | 196.02 | 98.64 | 218.23 |
| 0.8 | Hybrid (ML q95 + guard + fallback) | 72.56 | 0.09 | 1 | 195.85 | 98.96 | 218.58 |
| 0.8 | Oracle (perfect forecast) | 74.08 | 0.00 | 0 | 197.66 | 96.75 | 216.21 |
| 0.85 | Reactive (target tracking) | 78.52 | 0.18 | 3 | 340.60 | 91.42 | 210.40 |
| 0.85 | Predictive (ML mean) | 78.52 | 0.15 | 2 | 282.63 | 91.38 | 210.36 |
| 0.85 | Predictive (ML q95) | 76.97 | 0.15 | 2 | 260.44 | 93.18 | 212.31 |
| 0.85 | Hybrid (ML q95 + guard + fallback) | 76.83 | 0.15 | 2 | 258.79 | 93.49 | 212.64 |
| 0.85 | Oracle (perfect forecast) | 78.46 | 0.00 | 0 | 262.71 | 91.44 | 210.46 |
| 0.9 | Reactive (target tracking) | 82.90 | 0.24 | 3 | 585.38 | 86.45 | 205.01 |
| 0.9 | Predictive (ML mean) | 82.83 | 0.24 | 2 | 440.49 | 86.52 | 205.09 |
| 0.9 | Predictive (ML q95) | 81.23 | 0.15 | 2 | 373.98 | 88.22 | 206.94 |
| 0.9 | Hybrid (ML q95 + guard + fallback) | 81.07 | 0.15 | 2 | 373.98 | 88.52 | 207.27 |
| 0.9 | Oracle (perfect forecast) | 82.84 | 0.00 | 0 | 388.97 | 86.48 | 205.10 |
| 0.95 | Reactive (target tracking) | 86.77 | 0.47 | 9 | 800.00 | 82.59 | 200.82 |
| 0.95 | Predictive (ML mean) | 86.69 | 0.29 | 3 | 800.00 | 82.65 | 200.89 |
| 0.95 | Predictive (ML q95) | 85.36 | 0.24 | 3 | 685.00 | 83.92 | 202.26 |
| 0.95 | Hybrid (ML q95 + guard + fallback) | 85.17 | 0.21 | 3 | 677.46 | 84.27 | 202.65 |
| 0.95 | Oracle (perfect forecast) | 86.73 | 0.00 | 0 | 746.66 | 82.61 | 200.91 |

## Launch-delay sweep, with unforeseen surges (target util 85 %)

| boot_min | policy | violation_steps_% | oom_episodes | cost_usd |
|---|---|---|---|---|
| 5 | Reactive (target tracking) | 0.03 | 1 | 91.42 |
| 5 | Predictive (ML mean) | 0.03 | 1 | 91.38 |
| 5 | Predictive (ML q95) | 0.03 | 1 | 93.18 |
| 5 | Hybrid (ML q95 + guard + fallback) | 0.03 | 1 | 93.40 |
| 5 | Oracle (perfect forecast) | 0.00 | 0 | 91.44 |
| 10 | Reactive (target tracking) | 0.09 | 2 | 91.42 |
| 10 | Predictive (ML mean) | 0.09 | 2 | 91.38 |
| 10 | Predictive (ML q95) | 0.09 | 2 | 93.18 |
| 10 | Hybrid (ML q95 + guard + fallback) | 0.09 | 2 | 93.46 |
| 10 | Oracle (perfect forecast) | 0.00 | 0 | 91.44 |
| 15 | Reactive (target tracking) | 0.18 | 3 | 91.42 |
| 15 | Predictive (ML mean) | 0.15 | 2 | 91.38 |
| 15 | Predictive (ML q95) | 0.15 | 2 | 93.18 |
| 15 | Hybrid (ML q95 + guard + fallback) | 0.15 | 2 | 93.49 |
| 15 | Oracle (perfect forecast) | 0.00 | 0 | 91.44 |
| 30 | Reactive (target tracking) | 0.44 | 3 | 91.42 |
| 30 | Predictive (ML mean) | 0.41 | 3 | 91.38 |
| 30 | Predictive (ML q95) | 0.41 | 3 | 93.18 |
| 30 | Hybrid (ML q95 + guard + fallback) | 0.41 | 3 | 93.54 |
| 30 | Oracle (perfect forecast) | 0.18 | 3 | 91.44 |

## Horizon sweep

| horizon_steps | horizon_min | MAE_MB | persistence_MAE_MB | MAPE_% | violation_steps_% | cost_usd | avg_util_% |
|---|---|---|---|---|---|---|---|
| 1 | 5.00 | 97.21 | 179.18 | 0.39 | 0.00 | 91.73 | 77.87 |
| 2 | 10.00 | 176.25 | 340.30 | 0.70 | 0.00 | 92.23 | 77.39 |
| 3 | 15.00 | 260.53 | 497.00 | 1.03 | 0.00 | 92.72 | 76.99 |
| 6 | 30.00 | 489.17 | 941.59 | 1.92 | 0.00 | 93.50 | 76.29 |
| 12 | 60.00 | 810.49 | 1756.24 | 3.15 | 0.00 | 94.71 | 75.31 |

## Latest recommendation

```json
{
  "timestamp": "2026-03-05 23:55:00",
  "horizon_min": 15,
  "current_memory_mb": 21125.5,
  "predicted_memory_mb": 20824.2,
  "predicted_upper_mb": 21258.3,
  "instance_mb": 4096.0,
  "target_util": 0.85,
  "recommended_instances": 7
}
```
