"""Run motif-bank anomaly detection for one or more configs (src/config.py) on a split.

For every series: estimate the period → build the bank with motif discovery on
the training part (cached) → per-motif distances of every window → score →
window-to-point aggregation → min-max → TSB-AD VUS-PR. Each config also runs
its random-bank controls. Rows go to results/runs/<name>/<split>.csv; the
config is copied next to them. Resumable: finished (file, variant) rows are skipped.

    .venv/bin/python notebooks/run.py configs/U-LoCoMotif.toml [more.toml] --split dev_small [--files a.csv]

Splits: U — tuning (48), dev_small (150), dev (520), eval (350); M — tuning (20), eval (180).
Eval is for frozen configurations only (PROJECT.md §5).
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backends import factories, multivariate_factories  # noqa: E402
from config import Config  # noqa: E402
from data import load_series  # noqa: E402
from pipeline import discover_cached, pooled_occurrences, shuffle_mask, window_bounds  # noqa: E402
from representation import add_shifts, motif_features, split_long, window_to_points  # noqa: E402
from scoring import evaluate, minmax  # noqa: E402

LISTS = ROOT / "external" / "TSB-AD" / "Datasets" / "File_List"
SPLITS = {
    ("U", "tuning"): LISTS / "TSB-AD-U-Tuning.csv",
    ("U", "eval"): LISTS / "TSB-AD-U-Eva.csv",
    ("U", "dev"): ROOT / "splits" / "dev.csv",
    ("U", "dev_small"): ROOT / "splits" / "dev_small.csv",
    ("M", "tuning"): LISTS / "TSB-AD-M-Tuning.csv",
    ("M", "eval"): LISTS / "TSB-AD-M-Eva.csv",
}
KEEP = ["VUS-PR", "AUC-PR", "VUS-ROC", "AUC-ROC", "Standard-F1"]
COLUMNS = ["file", "n", "dims", "period", "train", "anomaly_ratio", "variant", "status", "motifs", "occurrences",
           *KEEP, "seconds"]  # fixed, so failure rows and scored rows always line up


def period_of(values: np.ndarray) -> int:
    """TSB-AD's window rule: find_length_rank on the first dimension."""
    from TSB_AD.utils.slidingWindows import find_length_rank

    first = values if values.ndim == 1 else values[:, 0]
    return max(4, int(find_length_rank(first.reshape(-1, 1), rank=1)))


def bank_scores(features: np.ndarray, occurrences: list, n_train_windows: int, scorer: str) -> np.ndarray:
    """Per-motif distances -> one score per window."""
    train = features[: max(1, n_train_windows)]
    features = features / np.maximum(train.max(axis=0), 1e-12)  # 0-1 scale on the training windows
    if scorer == "nearest":
        return features.min(axis=1)
    if scorer == "weighted":  # NormA-style: weight = share of occurrences
        _, counts = np.unique([o[0] for o in occurrences], return_counts=True)
        return features @ (counts / counts.sum())
    if scorer == "iforest":
        from sklearn.ensemble import IsolationForest

        return -IsolationForest(random_state=0).fit(features[: len(train)]).score_samples(features)
    raise ValueError(f"unknown scorer {scorer!r}")


def variant_score(cfg: Config, values: np.ndarray, segments: list, n_train: int, bounds, fraction=None):
    """Bank (motifs, or a random bank when `fraction` is set) -> per-point score in [0, 1]."""
    rng = np.random.default_rng(cfg.seed)
    occ = pooled_occurrences(segments, None if fraction is None else lambda m: shuffle_mask(m, rng, fraction))
    if cfg.split_long:
        occ = split_long(occ, bounds[1])
    if cfg.shift:
        occ = add_shifts(occ, cfg.shift, len(values))
    features, wlen = motif_features(values, occ, cfg.norm)
    window = bank_scores(features, occ, n_train - wlen + 1, cfg.scorer)
    return minmax(window_to_points(window, len(values), wlen, cfg.point)), occ


def run_series(cfg: Config, filename: str, make_model) -> list[dict]:
    series = load_series(filename)
    values = series.values
    period = period_of(values)
    bounds = window_bounds(period, cfg.wlen_floor, cfg.wlen_factor)
    tr = int(Path(filename).stem.split("_")[-3])  # ..._tr_<end>_1st_<idx>
    n_train = len(values) if cfg.source == "whole" else max(tr, 4 * bounds[0])
    base = {"file": filename, "n": len(values), "dims": 1 if values.ndim == 1 else values.shape[1],
            "period": period, "train": n_train, "anomaly_ratio": float(series.labels.mean())}
    cache = ROOT / "results" / f"masks_{cfg.source}" / cfg.bank_label() / f"{Path(filename).stem}.pkl"
    if cfg.source == "whole":  # cycle-1 cache layout
        cache = ROOT / "results" / "masks" / cfg.bank_label() / f"{Path(filename).stem}.pkl"
    segments = discover_cached(cache, values[:n_train], make_model, period, cfg.segment_len, bounds)

    rows = []
    for variant, fraction in [("motif", None)] + [(f"random{f:g}", f) for f in cfg.random_fractions]:
        start = time.perf_counter()
        try:
            score, occ = variant_score(cfg, values, segments, n_train, bounds, fraction)
            metrics = evaluate(score, series.labels, period)
            row = {**base, "variant": variant, "status": "ok", "motifs": len({o[0] for o in occ}),
                   "occurrences": len(occ), **{k: metrics.get(k, np.nan) for k in KEEP}}
        except Exception as exc:  # recorded, counted as failure_value in summaries
            row = {**base, "variant": variant, "status": f"{type(exc).__name__}: {str(exc)[:120]}",
                   **{k: np.nan for k in KEEP}}
        rows.append({**row, "seconds": round(time.perf_counter() - start, 1)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--split", default="dev_small")
    parser.add_argument("--files", default=None, help="comma-separated subset")
    args = parser.parse_args()

    for path in args.configs:
        cfg = Config.load(path)
        files = args.files.split(",") if args.files else pd.read_csv(SPLITS[(cfg.dataset, args.split)])["file_name"].tolist()
        out_dir = ROOT / "results" / "runs" / cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, out_dir / "config.toml")
        (out_dir / "config.resolved.json").write_text(json.dumps(cfg.as_dict(), indent=2))
        out_csv = out_dir / f"{args.split}.csv"
        done = set(pd.read_csv(out_csv)["file"]) if out_csv.exists() else set()
        backends = factories() if cfg.dataset == "U" else multivariate_factories()
        make_model = functools.partial(backends[cfg.backend], **cfg.params)
        print(f"== {cfg.name} on {args.split}: {len(files)} series", flush=True)
        for i, filename in enumerate(files, 1):
            if filename in done:
                continue
            try:
                rows = run_series(cfg, filename, make_model)
            except Exception as exc:  # discovery itself failed
                rows = [{"file": filename, "variant": v, "status": f"{type(exc).__name__}: {str(exc)[:120]}"}
                        for v in ["motif"] + [f"random{f:g}" for f in cfg.random_fractions]]
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            shown = " ".join(f"{r['variant']}={r.get('VUS-PR', float('nan')):.3f}" for r in rows)
            print(f"[{i}/{len(files)}] {filename[:45]:45} {shown}", flush=True)


if __name__ == "__main__":
    main()
