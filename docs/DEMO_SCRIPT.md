# Screencast outline (~8-10 min)

Record on the cloud VM (EC2 / Compute Engine / Azure VM) so the cloud platform is visible in the recording
(console page with the instance + terminal/SSH session).

1. **Intro (30 s)** — problem statement from the assignment; one sentence on the approach
   (forecast memory 15 min ahead -> size the instance pool -> compare with static and reactive).
2. **Cloud setup (1 min)** — show the VM in the cloud console, SSH in, `git clone`, `pip install -r requirements-cloud.txt`.
3. **Data and features (1 min)** — open `results/figures/01_workload.png`; list the feature groups from `src/features.py`.
4. **Run the pipeline (1 min)** — `python main.py` (about 1.5 min; speed up the video if needed), `python -m pytest`.
5. **Prediction accuracy (1.5 min)** — `results/figures/02_model_comparison.png`, `03_forecast_vs_actual.png`; the
   model table (MAE 253 MB / 1.0 % MAPE vs persistence 497 MB / 2.0 %); mention q95 coverage 90.8 %.
6. **Auto-scaling (1.5 min)** — `06_capacity_timeline.png`; explain sizing rule, 15-min launch delay, stabilised scale-in;
   show `results/RESULTS.md` policy table (utilisation, latency, cost, energy).
7. **When the prediction is wrong (2 min)** — `07_wrong_predictions.png` (bias/noise), `08_shock_scenario.png`,
   `11_launch_delay_sweep.png`; say plainly that under-prediction is the dangerous direction, that the hybrid fallback
   protects against biased forecasts, and that launch delay limits what any policy can do against sudden surges.
8. **Real cloud workload (1.5 min, live)** — in two terminals:
   `python cloud/load_generator.py --min-mb 300 --max-mb 1500 --period-s 1440 --duration-s 7200` and
   `python cloud/collect_metrics.py --out data/collected_vm.csv --interval 5 --virtual-step-min 5`.
   Show memory rising (e.g. `htop`/cloud monitoring). The pipeline needs roughly 5 virtual days of history
   (7200 s = 2 hours real time at 5 s/sample), so start the collector well before recording, then train: `python main.py --data data/collected_vm.csv --instance-mb 1024`.
9. **Optional: act on the forecast (1 min)** — `python cloud/aws_apply.py --asg <name> --region <r>` (dry run) and,
   if an Auto Scaling Group exists, `--apply --publish-metric`; show the desired capacity change / CloudWatch metric.
10. **Conclusion and limitations (30 s)** — synthetic data, assumed prices, small absolute gains; what real data would change.
