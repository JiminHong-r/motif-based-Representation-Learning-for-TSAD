"""Run TSB-AD baseline detectors over the same 48 Tuning series.

Hyperparameters are TSB-AD's own `Optimal_Uni_algo_HP_dict` — no tuning on
our side. Scores go through the same `minmax` + `evaluate` as the motif
pipeline, with the same period as `slidingWindow`, so rows are directly
comparable with results/tuning_metrics.csv.

    .venv/bin/python notebooks/02_baselines.py [--detectors MatrixProfile,SAND] [--files a.csv,b.csv]
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
from scoring import evaluate, minmax  # noqa: E402

tuning = importlib.import_module("01_tuning_set")

OUT_CSV = ROOT / "results" / "baseline_metrics.csv"
# Ablation (no motif selection), normal-pattern-library competitors, ceiling.
DETECTORS = ["MatrixProfile", "KShapeAD", "Series2Graph", "KMeansAD_U", "SAND", "NORMA", "Sub_PCA"]


def run_detector(name: str, filename: str, values: np.ndarray) -> np.ndarray:
    from TSB_AD import model_wrapper as mw
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

    data = values.reshape(-1, 1)
    hp = Optimal_Uni_algo_HP_dict[name]
    if name in mw.Semisupervise_AD_Pool:
        train_end = int(Path(filename).stem.split("_")[-3])  # ..._tr_<end>_1st_<idx>
        out = getattr(mw, f"run_{name}")(data[:train_end], data, **hp)
    else:
        out = getattr(mw, f"run_{name}")(data, **hp)
    return np.asarray(out, dtype=np.float64).ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", default=",".join(DETECTORS))
    parser.add_argument("--files", default=None, help="comma-separated subset")
    args = parser.parse_args()

    from TSB_AD.utils.slidingWindows import find_length_rank

    wanted = [d.strip() for d in args.detectors.split(",") if d.strip()]
    files = args.files.split(",") if args.files else pd.read_csv(tuning.FILE_LIST)["file_name"].tolist()
    already = tuning.done_keys(OUT_CSV)

    for i, filename in enumerate(files, 1):
        if not (DATA_DIR / filename).exists():
            print(f"[{i}/{len(files)}] MISSING {filename}")
            continue
        series = load_series(filename)
        period = max(4, int(find_length_rank(series.values.reshape(-1, 1), rank=1)))
        n = len(series)
        print(f"\n[{i}/{len(files)}] {filename}  n={n}  period={period}", flush=True)

        for name in wanted:
            if (filename, name) in already:
                continue
            row = {"file": filename, "backend": name, "n": n, "period": period,
                   "anomaly_ratio": float(series.labels.mean())}
            start = time.perf_counter()
            try:
                score = run_detector(name, filename, series.values)
                if len(score) != n:
                    raise ValueError(f"score length {len(score)} != {n}")
                metrics = evaluate(minmax(score), series.labels, period)
                row["status"] = "ok"
                row.update({k: metrics.get(k, float("nan")) for k in tuning.KEEP})
                print(f"    {name:<16} VUS-PR {row['VUS-PR']:.4f}", end="", flush=True)
            except Exception as exc:  # keep going; record the failure
                row["status"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                row.update({k: float("nan") for k in tuning.KEEP})
                print(f"    {name:<16} FAILED {row['status']}", end="", flush=True)
            row["seconds"] = round(time.perf_counter() - start, 1)
            print(f"  ({row['seconds']}s)", flush=True)
            tuning.append_row(OUT_CSV, row)


if __name__ == "__main__":
    main()
