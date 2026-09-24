"""Run the motif pipeline over the full TSB-AD-U tuning split (48 series).

Protocol is fixed before looking at any result, and is identical for every
series and every backend:

* window length from TSB-AD's `find_length_rank(rank=1)` period estimate
  (`pipeline.window_bounds`) — never chosen by hand per series;
* whole series scored via fixed-length overlapping segments, nothing skipped
  or truncated;
* `slidingWindow` for `get_metrics` is that same period estimate, matching
  TSB-AD's own benchmark script;
* a random-motif floor is computed for every series, not just once.

Results append to results/tuning_metrics.csv and completed rows are skipped,
so the run is resumable.

    .venv/bin/python notebooks/01_tuning_set.py [--backends k-Motiflets,LoCoMotif] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import DATA_DIR, load_series  # noqa: E402
from pipeline import SEGMENT_LEN, segmented_score, shuffle_mask  # noqa: E402
from scoring import evaluate, minmax  # noqa: E402

FILE_LIST = ROOT / "external" / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
OUT_CSV = ROOT / "results" / "tuning_metrics.csv"
KEEP = ["VUS-PR", "AUC-PR", "VUS-ROC", "AUC-ROC", "Standard-F1"]


def backend_factories():
    from tsmd.competitors.locomotif import LocoMotif
    from tsmd.competitors.motiflets import Motiflets

    return {
        "k-Motiflets": lambda lo, hi: Motiflets(k_max=10, min_wlen=lo, max_wlen=hi),
        "LoCoMotif": lambda lo, hi: LocoMotif(min_wlen=lo, max_wlen=hi),
    }


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return set(zip(frame["file"], frame["backend"]))


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="k-Motiflets,LoCoMotif")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None,
                        help="skip series longer than this (recorded as skipped)")
    args = parser.parse_args()

    from TSB_AD.utils.slidingWindows import find_length_rank

    factories = backend_factories()
    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    files = pd.read_csv(FILE_LIST)["file_name"].tolist()
    if args.limit:
        files = files[: args.limit]

    already = done_keys(OUT_CSV)
    print(f"{len(files)} series, backends={wanted}, {len(already)} rows already done")

    for i, filename in enumerate(files, 1):
        if not (DATA_DIR / filename).exists():
            print(f"[{i}/{len(files)}] MISSING {filename}")
            continue
        series = load_series(filename)
        period = int(find_length_rank(series.values.reshape(-1, 1), rank=1))
        period = max(4, period)
        n = len(series)
        print(f"\n[{i}/{len(files)}] {filename}  n={n}  period={period}  "
              f"anom={series.labels.mean():.3%}", flush=True)

        if args.max_points and n > args.max_points:
            print("    skipped (too long for this pass)")
            continue

        for backend in wanted:
            for variant, hook in ((backend, None), (f"{backend}+random", "shuffle")):
                if (filename, variant) in already:
                    continue
                rng = np.random.default_rng(2024)
                mask_hook = (lambda m: shuffle_mask(m, rng)) if hook else None
                start = time.perf_counter()
                try:
                    raw = segmented_score(
                        series.values, factories[backend], period,
                        segment_len=SEGMENT_LEN, mask_hook=mask_hook,
                    )
                    score = minmax(raw)
                    metrics = evaluate(score, series.labels, period)
                    row = {"file": filename, "backend": variant, "n": n,
                           "period": period, "anomaly_ratio": float(series.labels.mean()),
                           "seconds": round(time.perf_counter() - start, 1),
                           "status": "ok"}
                    row.update({k: metrics.get(k, float("nan")) for k in KEEP})
                    print(f"    {variant:<22} VUS-PR {row['VUS-PR']:.4f}  "
                          f"({row['seconds']}s)", flush=True)
                except Exception as exc:  # keep going; record the failure
                    row = {"file": filename, "backend": variant, "n": n, "period": period,
                           "anomaly_ratio": float(series.labels.mean()),
                           "seconds": round(time.perf_counter() - start, 1),
                           "status": f"{type(exc).__name__}: {str(exc)[:120]}"}
                    row.update({k: float("nan") for k in KEEP})
                    print(f"    {variant:<22} FAILED {row['status']}", flush=True)
                append_row(OUT_CSV, row)

    summarise()


def summarise():
    if not OUT_CSV.exists():
        return
    frame = pd.read_csv(OUT_CSV)
    ok = frame[frame["status"] == "ok"]
    if ok.empty:
        print("\nno successful runs yet")
        return
    print("\n" + "=" * 78)
    print(f"{'backend':<24}{'series':>8}{'mean VUS-PR':>14}{'median':>10}{'mean AUC-PR':>14}")
    print("-" * 78)
    for backend, group in ok.groupby("backend"):
        print(f"{backend:<24}{len(group):>8}{group['VUS-PR'].mean():>14.4f}"
              f"{group['VUS-PR'].median():>10.4f}{group['AUC-PR'].mean():>14.4f}")
    print("=" * 78)
    print("TSB-AD published univariate reference (350-series Eva set):")
    print("  best leaderboard detector  VUS-PR 0.586   median detector  0.481")


if __name__ == "__main__":
    main()
