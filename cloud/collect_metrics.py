"""Collect REAL memory/CPU metrics on a cloud VM and write a CSV that main.py can load.

    pip install psutil
    python cloud/collect_metrics.py --out data/collected_vm.csv --interval 300          # real 5-min sampling
    python cloud/collect_metrics.py --out data/collected_vm.csv --interval 5 --virtual-step-min 5
        ^ "compressed time" for demos: samples every 5 s but stamps each row 5 virtual minutes apart,
          so a 24-minute run looks like ~1 day to the pipeline (pair with cloud/load_generator.py).

Then:  python main.py --data data/collected_vm.csv --instance-mb <memory per VM in MB>
"""
import argparse
import csv
import os
import time
from datetime import datetime, timedelta

import psutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/collected_vm.csv")
    ap.add_argument("--interval", type=float, default=300, help="seconds between real samples")
    ap.add_argument("--virtual-step-min", type=float, default=None,
                    help="if set, timestamps advance by this many minutes per sample (compressed time)")
    ap.add_argument("--samples", type=int, default=0, help="stop after N samples (0 = run until Ctrl-C)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    t0 = datetime.now().replace(second=0, microsecond=0)
    i = 0
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "memory_mb", "cpu_pct"])
        psutil.cpu_percent(None)
        try:
            while args.samples == 0 or i < args.samples:
                time.sleep(args.interval)
                ts = t0 + timedelta(minutes=args.virtual_step_min * i) if args.virtual_step_min else datetime.now()
                mem_mb = psutil.virtual_memory().used / 2**20
                w.writerow([ts.isoformat(timespec="seconds"), round(mem_mb, 1), psutil.cpu_percent(None)])
                f.flush()
                i += 1
        except KeyboardInterrupt:
            pass
    print(f"wrote {i} samples to {args.out}")


if __name__ == "__main__":
    main()
