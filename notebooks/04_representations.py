"""RQ1: representation variants over the same 48 Tuning series.

Motifs are discovered once per (series, backend) and cached in results/masks/,
then scored under each representation in REPS, each with its random-motif
floor (the same cached motifs, positions shuffled). Everything else — window
rule, segments, scorer, metric — is as in 01_tuning_set.py, so "shape-min"
reproduces its numbers.

REPS is an ablation ladder: each step changes one thing from the one above.
The two fixes (centre, excl) come from diagnosing where MatrixProfile beats
the motif pipeline; see `representation.distance_to_nearest_motif`.

    .venv/bin/python notebooks/04_representations.py [--backends LoCoMotif] [--reps shape-min] [--files a.csv]
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from data import DATA_DIR, load_series  # noqa: E402
from representation import uncovered_fraction  # noqa: E402
from pipeline import discover_cached, score_segments, shuffle_mask  # noqa: E402
from scoring import evaluate, minmax  # noqa: E402

tuning = importlib.import_module("01_tuning_set")

OUT_CSV = ROOT / "results" / "representation_metrics.csv"
MASK_DIR = ROOT / "results" / "masks"
REPS = {
    "shape-min": dict(norm="shape", point="min"),  # 01_tuning_set.py
    "shape-centre": dict(norm="shape", point="centre"),
    "shape-centre-excl": dict(norm="shape", point="centre", exclude_self=True),
    "centred-centre-excl": dict(norm="centred", point="centre", exclude_self=True),
    "centred-max-excl": dict(norm="centred", point="max", exclude_self=True),  # Louis: max-aggregation
    "raw-centre-excl": dict(norm="raw", point="centre", exclude_self=True),
    "uncovered": dict(represent=uncovered_fraction),  # occurrence-based: where, not what
}
# Equal-weight mean of min-max scores: the motif-specific "where" signal plus
# the best "what" signal. No weights to tune.
COMBOS = {"centred-centre-excl+uncovered": ["centred-centre-excl", "uncovered"]}


def rep_score(values: np.ndarray, segments: list, rep: str, shuffle: bool) -> np.ndarray:
    if rep in COMBOS:
        return np.mean([rep_score(values, segments, r, shuffle) for r in COMBOS[rep]], axis=0)
    rng = np.random.default_rng(2024)
    hook = (lambda m: shuffle_mask(m, rng)) if shuffle else None
    return minmax(score_segments(values, segments, hook, **REPS[rep]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="LoCoMotif,k-Motiflets")
    parser.add_argument("--reps", default=",".join([*REPS, *COMBOS]))
    parser.add_argument("--files", default=None, help="comma-separated subset")
    args = parser.parse_args()

    from TSB_AD.utils.slidingWindows import find_length_rank

    factories = tuning.backend_factories()
    backends = args.backends.split(",")
    reps = args.reps.split(",")
    files = args.files.split(",") if args.files else pd.read_csv(tuning.FILE_LIST)["file_name"].tolist()
    done = set()
    if OUT_CSV.exists():
        frame = pd.read_csv(OUT_CSV)
        done = set(zip(frame["file"], frame["backend"], frame["representation"]))

    for i, filename in enumerate(files, 1):
        series = load_series(filename)
        period = max(4, int(find_length_rank(series.values.reshape(-1, 1), rank=1)))
        n = len(series)
        print(f"\n[{i}/{len(files)}] {filename}  n={n}  period={period}", flush=True)

        for backend in backends:
            for rep in reps:
                for variant, shuffle in ((backend, False), (f"{backend}+random", True)):
                    if (filename, variant, rep) in done:
                        continue
                    row = {"file": filename, "backend": variant, "representation": rep, "n": n, "period": period,
                           "anomaly_ratio": float(series.labels.mean())}
                    start = time.perf_counter()
                    try:
                        segments = discover_cached(MASK_DIR / backend / f"{Path(filename).stem}.pkl",
                                                   series.values, factories[backend], period)
                        score = rep_score(series.values, segments, rep, shuffle)
                        metrics = evaluate(score, series.labels, period)
                        row["status"] = "ok"
                        row.update({k: metrics.get(k, float("nan")) for k in tuning.KEEP})
                        print(f"    {variant:<20} {rep:<20} VUS-PR {row['VUS-PR']:.4f}", end="", flush=True)
                    except Exception as exc:  # keep going; record the failure
                        row["status"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                        row.update({k: float("nan") for k in tuning.KEEP})
                        print(f"    {variant:<20} {rep:<20} FAILED {row['status']}", end="", flush=True)
                    row["seconds"] = round(time.perf_counter() - start, 1)
                    print(f"  ({row['seconds']}s)", flush=True)
                    tuning.append_row(OUT_CSV, row)


if __name__ == "__main__":
    main()
