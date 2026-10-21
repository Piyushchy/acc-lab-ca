# ML-Based Memory Usage Prediction and Intelligent Cloud Resource Allocation

ACC lab CA — *Cloud Resource Management* project.
Author: Piyush

An ML-based cloud resource manager that **forecasts memory demand 15 minutes ahead** and uses the forecast to
**auto-scale** a pool of VM instances. It is compared against static provisioning and a reactive (target-tracking)
autoscaler on accuracy, utilisation, latency, cost and energy, and it is stress-tested for the question
*"what happens when the prediction is wrong?"*

> **Read this first (honest scope).** The default workload is **synthetic** (generated in `src/data_gen.py`), and the
> cluster is a **discrete-time simulator** with *assumed* prices/power numbers (see `ClusterConfig`). Nothing in the
> headline tables was measured on a live cloud account. The repo also contains scripts to collect real metrics on a
> cloud VM and to push the recommendation to an AWS Auto Scaling Group (`cloud/`), and the pipeline accepts any real
> trace via `--data`. The AWS script has not been run against a live account.

## Headline results (synthetic 60-day trace, last ~12 days held out, 15-min horizon)

| | Static (peak-sized) | Reactive | ML predictive | Hybrid (ML + safety nets) |
|---|---|---|---|---|
| Avg memory utilisation | 50.3 % | 78.6 % | 78.5 % | 77.0 % |
| Steps with memory exhausted | 0 % | 0.03 % | 0 % | 0 % |
| p99 latency proxy | 194 ms | 333 ms | 279 ms | 254 ms |
| Cost (USD, assumed pricing) | 141.4 | 91.0 | 91.0 | 92.7 |
| Energy (kWh, assumed power model) | 264 | 209 | 209 | 211 |

* Forecast MAE **253 MB (1.0 % MAPE)** vs 497 MB (2.0 %) for "memory stays the same". Ridge, Random Forest,
  Gradient Boosting and an MLP are all within ~3 % of each other on this data (the synthetic data is close to linear).
* Dynamic scaling saves ~36 % cost and ~21 % energy versus static peak sizing.
* At the default 85 % sizing target all dynamic policies are nearly violation-free, so the interesting differences
  appear when packing tighter or when the forecast is wrong — see `results/RESULTS.md` and `REPORT.md`
  (target-utilisation sweep, bias/noise injection, surge scenario, launch-delay sweep, horizon sweep).

## Repository layout

```
main.py                  # whole pipeline: data -> features -> models -> simulation -> experiments -> figures
src/data_gen.py          # synthetic workload + surge injection + CSV loader for real traces
src/features.py          # leakage-free lag / rolling / calendar / request-rate features
src/models.py            # baselines + Ridge, Random Forest, Gradient Boosting, MLP, quantile (q95) model
src/autoscaler.py        # policies (static, reactive, predictive, hybrid, oracle) + cluster simulator
src/metrics.py, plots.py
cloud/collect_metrics.py # record REAL memory/CPU on a cloud VM -> CSV for main.py --data
cloud/load_generator.py  # create a real, repeating memory workload on a VM for the demo
cloud/aws_apply.py       # apply recommended capacity to an AWS Auto Scaling Group (dry run by default)
tests/                   # pytest: no-leakage, alignment, simulator invariants, fallback behaviour
results/                 # generated tables (CSV + RESULTS.md) and figures/
data/workload_synthetic.csv
REPORT.md                # the written report
docs/DEMO_SCRIPT.md      # outline for the screencast
Dockerfile
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                  # ~1.5 min; writes results/ and results/figures/
python -m pytest                # 7 tests
```

Useful options: `--horizon 6` (steps of 5 min), `--target-util 0.8`, `--boot-steps 2`, `--instance-mb 8192`,
`--price 0.083`, `--days 90`, `--seed 7`.

## How this maps to the assignment

| Assignment item | Where |
|---|---|
| Input data and features | `src/data_gen.py`, `src/features.py` (32 features: memory lags/diffs, rolling stats, request-rate and CPU signals, calendar of the target time, yesterday's change) |
| ML algorithm and prediction horizon | `src/models.py` (model comparison, q95 upper-bound model); horizon = 15 min, swept 5-60 min (`results/horizon_sweep.csv`) |
| Allocation / auto-scaling mechanism | `src/autoscaler.py` (sizing rule, launch delay, stabilised scale-in, emergency guard, drift fallback) |
| Evaluate accuracy, utilisation, latency, cost, energy | `results/model_comparison_test.csv`, `results/policy_comparison.csv`, `results/RESULTS.md` |
| What if the prediction is wrong? | bias / noise injection, unforeseen surges, drift fallback, launch-delay sweep — `REPORT.md` section 7 |
| Cloud demonstration + screencast + report | `cloud/`, `docs/DEMO_SCRIPT.md`, `REPORT.md` |

## Using real data

Provide a CSV with `timestamp` and `memory_mb` (optional `requests`, `cpu_pct`); it is resampled to 5-minute means:

```bash
python main.py --data my_trace.csv --instance-mb 8192
```

Public traces (Google cluster-data, Alibaba cluster-trace, Azure public dataset) can be converted into this format.
Or collect your own on a cloud VM:

```bash
pip install -r requirements-cloud.txt
python cloud/load_generator.py --min-mb 300 --max-mb 1500 --period-s 1440 --duration-s 7200 &
python cloud/collect_metrics.py --out data/collected_vm.csv --interval 5 --virtual-step-min 5   # compressed time
python main.py --data data/collected_vm.csv --instance-mb 1024
python cloud/aws_apply.py --asg my-asg --region <region>            # dry run; add --apply to act
```

Note: the 24-hour lag features need roughly 5 days of (virtual) history to train sensibly (with the commands above: 2 hours of real time).

## Assumptions and limitations

* Prices (`$0.0416`/instance-hour, roughly a 4 GB on-demand VM), the linear power model (45–110 W per instance)
  and the latency proxy (queueing-style `L0/(1-u)`, 2 s on memory exhaustion) are **assumptions**, not measurements.
* Synthetic data embeds the assumption that request rate leads memory; results on real traces will differ.
* Memory is treated as the only scaling dimension; CPU, network and disk are not modelled.
* The `q95` upper-bound model covered 90.8 % of test points (target 95 %), i.e. it is somewhat over-confident.
* No claim is made that the ML policy beats a well-tuned reactive policy by a large margin on this data; the gains
  are small in absolute terms and show up mainly at high target utilisation and under biased forecasts.
