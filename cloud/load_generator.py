"""Generate a REAL, repeating memory workload on a cloud VM so there is something to forecast.

Allocates/frees memory following a "day-like" sine wave plus random bursts. Run it next to
cloud/collect_metrics.py (compressed time, see that file).

    python cloud/load_generator.py --min-mb 300 --max-mb 1500 --period-s 1440 --duration-s 7200
"""
import argparse
import math
import random
import time

CHUNK = 16 * 2**20  # allocate in 16 MB chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mb", type=float, default=300)
    ap.add_argument("--max-mb", type=float, default=1500)
    ap.add_argument("--period-s", type=float, default=1440, help="length of one 'day' in seconds")
    ap.add_argument("--duration-s", type=float, default=7200)
    ap.add_argument("--step-s", type=float, default=5)
    args = ap.parse_args()

    chunks: list[bytearray] = []
    burst, t0 = 0.0, time.time()
    while time.time() - t0 < args.duration_s:
        t = time.time() - t0
        wave = 0.5 * (1 + math.sin(2 * math.pi * (t / args.period_s) - math.pi / 2))
        if random.random() < 0.01:
            burst = random.uniform(0.1, 0.3) * (args.max_mb - args.min_mb)
        burst *= 0.85
        target_mb = args.min_mb + wave * (args.max_mb - args.min_mb) + burst
        want = int(target_mb * 2**20 / CHUNK)
        while len(chunks) < want:
            chunks.append(bytearray(b"\x01") * CHUNK)  # touch pages so they are really resident
        while len(chunks) > want:
            chunks.pop()
        time.sleep(args.step_s)


if __name__ == "__main__":
    main()
