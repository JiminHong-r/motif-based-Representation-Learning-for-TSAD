"""Semi-supervised variant: motifs from the training part, per-motif features.

Supervisor meeting 2026-09-29: TSB-AD's `tr_<end>` prefix may be used as
training data. Protocol, fixed for every series:

* training part = first max(tr_end, 4 x min motif length) points — the floor
  only matters for 2 Tuning series, and there it still ends before the first
  anomaly;
* motifs are discovered on the training part only (segmented as elsewhere);
* every window of the whole series gets one distance per motif
  (`representation.motif_features`), each column divided by its maximum over
  the training windows (0-1 scale);
* scorers: "nearest" = smallest per-motif distance; "iforest" = sklearn
  IsolationForest with default settings fitted on the training windows;
  "weighted" = NormA-style weighted average of the per-motif distances, each
  motif weighted by its share of all occurrences (Louis, 2026-09-30);
  window scores go to points via the centred window, or with --point max via
  the maximum over covering windows (labelled "-max");
* the whole series is scored and evaluated, as TSB-AD does for its
  semi-supervised detectors; the random floor shuffles the training motifs.

Motif-discovery hyperparameters (supervisor: changing those of existing methods
is fair game) go in --params; each setting is its own backend label, e.g.
"k-Motiflets[k_max=20]", with its own motif cache.

    .venv/bin/python notebooks/05_trained.py [--backends LoCoMotif] [--params rho=0.6] [--files a.csv]

`--split eval` runs the 350 Eval series into results/eval_metrics.csv with the
frozen scorer only (PROJECT.md §4) — run once, never to choose between variants:

    .venv/bin/python notebooks/05_trained.py --split eval --backends LoCoMotif --params rho=0.6 --point max
"""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from data import load_series  # noqa: E402
from pipeline import discover_cached, pooled_occurrences, shuffle_mask, window_bounds  # noqa: E402
from representation import motif_features, window_to_points  # noqa: E402
from scoring import evaluate, minmax  # noqa: E402

tuning = importlib.import_module("01_tuning_set")

OUT_CSV = ROOT / "results" / "trained_metrics.csv"
MASK_DIR = ROOT / "results" / "masks_train"
SCORERS = ["nearest", "iforest", "weighted"]


def train_end(filename: str, period: int) -> int:
    tr = int(Path(filename).stem.split("_")[-3])  # ..._tr_<end>_1st_<idx>
    return max(tr, 4 * window_bounds(period)[0])


def scores(values: np.ndarray, occurrences: list, n_train: int, point: str = "centre") -> dict[str, np.ndarray]:
    from sklearn.ensemble import IsolationForest

    features, wlen = motif_features(values, occurrences)
    train_rows = features[: max(1, n_train - wlen + 1)]
    features = features / np.maximum(train_rows.max(axis=0), 1e-12)
    train_rows = features[: len(train_rows)]
    ids, counts = np.unique([m for m, _, _ in occurrences], return_counts=True)  # same order as columns
    window = {
        "nearest": features.min(axis=1),
        "weighted": features @ (counts / counts.sum()),
        "iforest": -IsolationForest(random_state=0).fit(train_rows).score_samples(features),
    }
    return {k: minmax(window_to_points(v, len(values), wlen, point)) for k, v in window.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="LoCoMotif,k-Motiflets")
    parser.add_argument("--files", default=None, help="comma-separated subset")
    parser.add_argument("--params", default="", help="discovery hyperparameters, e.g. k_max=20")
    parser.add_argument("--point", default="centre", help="window->point aggregation: centre or max")
    parser.add_argument("--split", default="tuning", choices=["tuning", "eval"])
    args = parser.parse_args()
    file_list = tuning.FILE_LIST if args.split == "tuning" else tuning.FILE_LIST.with_name("TSB-AD-U-Eva.csv")
    out_csv = OUT_CSV if args.split == "tuning" else OUT_CSV.with_name("eval_metrics.csv")
    scorers = SCORERS if args.split == "tuning" else ["nearest"]  # the frozen scorer only
    params = {k: float(v) if "." in v else int(v) for k, v in (kv.split("=") for kv in args.params.split(",") if kv)}
    tag = f"[{args.params}]" if params else ""
    suffix = "" if args.point == "centre" else f"-{args.point}"

    from TSB_AD.utils.slidingWindows import find_length_rank

    factories = tuning.backend_factories()
    files = args.files.split(",") if args.files else pd.read_csv(file_list)["file_name"].tolist()
    done = set()
    if out_csv.exists():
        frame = pd.read_csv(out_csv)
        done = set(zip(frame["file"], frame["backend"], frame["representation"]))

    for i, filename in enumerate(files, 1):
        series = load_series(filename)
        period = max(4, int(find_length_rank(series.values.reshape(-1, 1), rank=1)))
        n, n_train = len(series), train_end(filename, period)
        print(f"\n[{i}/{len(files)}] {filename}  n={n}  period={period}  train={n_train}", flush=True)

        for backend in args.backends.split(","):
            label = backend + tag
            for variant, shuffle in ((label, False), (f"{label}+random", True)):
                todo = [sc for sc in scorers if (filename, variant, f"train-{sc}{suffix}") not in done]
                if not todo:
                    continue
                base = {"file": filename, "backend": variant, "n": n, "period": period, "train": n_train,
                        "anomaly_ratio": float(series.labels.mean())}
                start = time.perf_counter()
                try:
                    segments = discover_cached(MASK_DIR / label / f"{Path(filename).stem}.pkl",
                                               series.values[:n_train],
                                               functools.partial(factories[backend], **params), period)
                    rng = np.random.default_rng(2024)
                    occ = pooled_occurrences(segments, (lambda m: shuffle_mask(m, rng)) if shuffle else None)
                    rows = []
                    for scorer, score in scores(series.values, occ, n_train, args.point).items():
                        if scorer not in todo:
                            continue
                        metrics = evaluate(score, series.labels, period)
                        rows.append({**base, "representation": f"train-{scorer}{suffix}", "status": "ok",
                                     **{k: metrics.get(k, float("nan")) for k in tuning.KEEP}})
                        print(f"    {variant:<20} train-{scorer + suffix:<10} VUS-PR {metrics['VUS-PR']:.4f}", flush=True)
                except Exception as exc:  # keep going; record the failure
                    status = f"{type(exc).__name__}: {str(exc)[:120]}"
                    rows = [{**base, "representation": f"train-{s}{suffix}", "status": status,
                             **{k: float("nan") for k in tuning.KEEP}} for s in todo]
                    print(f"    {variant:<20} FAILED {status}", flush=True)
                seconds = round(time.perf_counter() - start, 1)
                for row in rows:
                    tuning.append_row(out_csv, {**row, "seconds": seconds})


if __name__ == "__main__":
    main()
