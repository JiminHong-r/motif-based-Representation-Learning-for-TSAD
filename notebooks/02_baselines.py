"""Run TSB-AD baseline detectors over the same 48 Tuning series.

Hyperparameters are TSB-AD's own `Optimal_Uni_algo_HP_dict` — no tuning on
our side. Scores go through the same `minmax` + `evaluate` as the motif
pipeline, with the same period as `slidingWindow`, so rows are directly
comparable with results/tuning_metrics.csv.

    .venv/bin/python notebooks/02_baselines.py [--detectors MatrixProfile,SAND] [--files a.csv,b.csv] [--split eval]

`--split eval` runs the 350 Eval series into results/baseline_eval_metrics.csv,
used to check our pipeline against TSB-AD's published per-series VUS-PR.
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


def run_pbad(values: np.ndarray, period: int) -> np.ndarray:
    """PBAD (Feremans et al. 2019), authors' code in external/PBAD (+ patches/pbad-numpy.patch), README defaults.

    Unsupervised on the whole series. Each window spans one period, binned
    (PBAD's moving-average `bin_size`) to ~12 bins as in the README, stride 1 bin:
    unbinned 158-point windows make SPMF's MaxSP run for hours, and a stride of
    half a window leaves so few windows that minsup 1% is an absolute support of 1.
    Each point gets the max score of the windows covering it (PBAD labels a
    window anomalous if any of its points is).
    """
    import contextlib, io, os
    os.environ["PATH"] = "/opt/homebrew/opt/openjdk/bin:" + os.environ["PATH"]  # SPMF runs `java -jar`
    sys.path.insert(0, str(ROOT / "external" / "PBAD" / "src"))
    from methods.PBAD import PBAD
    from methods.PreProcessor import PreProcessor

    data = values.reshape(-1, 1) if values.ndim == 1 else values
    n = len(data)
    bin_size = -(-period // 12)
    bins = -(-period // bin_size)  # window length in bins
    with contextlib.redirect_stdout(io.StringIO()):  # PBAD prints debug lines per mining call
        w_disc, w, _, _ = PreProcessor(window_size=bins, window_incr=1, bin_size=bin_size, alphabet_size=30,
                                       verbose=False).preprocess(
            continuous_series={d: data[:, d].astype(np.float64) for d in range(data.shape[1])},
            labels=np.zeros(n), return_undiscretized=True)
        try:
            win = PBAD(relative_minsup=0.01, jaccard_threshold=0.9, pattern_type="all",
                       pattern_pruning="maximal", verbose=False).fit_predict(w, w_disc)
        except ValueError:  # no usable features; the authors' fix (PBAD.fit comment): closed patterns
            win = PBAD(relative_minsup=0.01, jaccard_threshold=0.9, pattern_type="all",
                       pattern_pruning="closed", verbose=False).fit_predict(w, w_disc)
    score = np.zeros(n)
    for i, s in enumerate(win):
        seg = score[i * bin_size:(i + bins) * bin_size]
        np.maximum(seg, s, out=seg)
    return score


def run_detector(name: str, filename: str, values: np.ndarray, period: int = 0) -> np.ndarray:
    if name == "PBAD":
        return run_pbad(values, period)
    from TSB_AD import model_wrapper as mw
    from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict

    data = values.reshape(-1, 1) if values.ndim == 1 else values
    hp = (Optimal_Uni_algo_HP_dict if values.ndim == 1 else Optimal_Multi_algo_HP_dict)[name]
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
    parser.add_argument("--split", default="tuning", choices=["tuning", "eval", "dev", "dev_small"])
    parser.add_argument("--dataset", default="U", choices=["U", "M"])
    args = parser.parse_args()
    lists = tuning.FILE_LIST.parent
    file_list = {"tuning": lists / f"TSB-AD-{args.dataset}-Tuning.csv", "eval": lists / f"TSB-AD-{args.dataset}-Eva.csv",
                 "dev": ROOT / "splits" / "dev.csv", "dev_small": ROOT / "splits" / "dev_small.csv"}[args.split]
    out_csv = OUT_CSV.with_name({("U", "tuning"): "baseline_metrics.csv", ("U", "eval"): "baseline_eval_metrics.csv"}.get(
        (args.dataset, args.split), f"baseline_{args.dataset}_{args.split}_metrics.csv"))

    from TSB_AD.utils.slidingWindows import find_length_rank

    wanted = [d.strip() for d in args.detectors.split(",") if d.strip()]
    files = args.files.split(",") if args.files else pd.read_csv(file_list)["file_name"].tolist()
    already = tuning.done_keys(out_csv)

    for i, filename in enumerate(files, 1):
        if not (DATA_DIR / filename).exists() and not (DATA_DIR.with_name("TSB-AD-M") / filename).exists():
            print(f"[{i}/{len(files)}] MISSING {filename}")
            continue
        series = load_series(filename)
        first = series.values if series.values.ndim == 1 else series.values[:, 0]  # TSB-AD: first dimension
        period = max(4, int(find_length_rank(first.reshape(-1, 1), rank=1)))
        n = len(series)
        print(f"\n[{i}/{len(files)}] {filename}  n={n}  period={period}", flush=True)

        for name in wanted:
            if (filename, name) in already:
                continue
            row = {"file": filename, "backend": name, "n": n, "period": period,
                   "anomaly_ratio": float(series.labels.mean())}
            start = time.perf_counter()
            try:
                score = run_detector(name, filename, series.values, period)
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
            tuning.append_row(out_csv, row)


if __name__ == "__main__":
    main()
