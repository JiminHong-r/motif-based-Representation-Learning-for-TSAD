"""Phase 1, step 3: run two tsmd backends on one ECG slice and eyeball the motifs.

No representation and no scoring yet — the only question here is whether the
discovered motifs line up with the visible heartbeat pattern.

    .venv/bin/python notebooks/01_toy_slice.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_series, slice_around_first_anomaly  # noqa: E402

SERIES = "170_MITDB_id_1_Medical_tr_17675_1st_17775.csv"
WIDTH = 6000
# MITDB is sampled at 360 Hz, so one heartbeat is roughly 250-350 samples.
MIN_WLEN, MAX_WLEN = 200, 400
OUT_DIR = ROOT / "results"


def run_backend(name, build, signal):
    """Fit one backend, returning its per-pattern prediction mask and timing."""
    print(f"\n--- {name} ---", flush=True)
    model = build()
    start = time.perf_counter()
    model.fit(signal)
    elapsed = time.perf_counter() - start
    mask = np.asarray(model.prediction_mask_)
    if mask.ndim == 1:  # normalise to (n_patterns, n_samples)
        mask = mask[None, :]
    print(f"fitted in {elapsed:.1f}s | mask {mask.shape} | {mask.shape[0]} motif(s)")
    for i, row in enumerate(mask):
        covered = int(row.sum())
        print(f"  motif {i}: {covered:5d} pts covered ({100 * covered / len(signal):5.1f}%)")
    return mask, elapsed


def plot(sl, results, path):
    fig, axes = plt.subplots(len(results) + 1, 1, figsize=(15, 3 + 2.2 * len(results)), sharex=True)
    t = np.arange(len(sl))

    axes[0].plot(t, sl.values, lw=0.6, color="black")
    axes[0].set_title(f"{sl.name}  (offset {sl.offset} in the full recording)")
    axes[0].set_ylabel("ECG")

    for ax in axes:
        for start, end in sl.anomaly_runs:
            ax.axvspan(start, end, color="red", alpha=0.18, lw=0)

    for ax, (name, (mask, elapsed)) in zip(axes[1:], results.items()):
        ax.plot(t, sl.values, lw=0.5, color="lightgray")
        colours = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, row in enumerate(mask):
            occ = sl.values.astype(float).copy()
            occ[row == 0] = np.nan
            ax.plot(t, occ, lw=1.1, color=colours[i % 10], label=f"motif {i}")
        ax.set_title(f"{name}  ({mask.shape[0]} motifs, {elapsed:.1f}s)  — red = labelled anomaly")
        ax.legend(loc="upper right", fontsize=7, ncol=4)

    axes[-1].set_xlabel("sample index within slice")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


def main():
    series = load_series(SERIES)
    sl = slice_around_first_anomaly(series, width=WIDTH)
    anomalous = int(sl.labels.sum())
    print(f"full series : {len(series)} pts, {int(series.labels.sum())} anomalous")
    print(f"slice       : {len(sl)} pts at offset {sl.offset}, "
          f"{anomalous} anomalous ({100 * anomalous / len(sl):.1f}%)")
    print(f"anomaly runs (local): {sl.anomaly_runs}")

    from tsmd.competitors.locomotif import LocoMotif
    from tsmd.competitors.motiflets import Motiflets

    results = {}
    results["k-Motiflets"] = run_backend(
        "k-Motiflets",
        lambda: Motiflets(k_max=10, min_wlen=MIN_WLEN, max_wlen=MAX_WLEN),
        sl.values,
    )
    results["LoCoMotif"] = run_backend(
        "LoCoMotif",
        lambda: LocoMotif(min_wlen=MIN_WLEN, max_wlen=MAX_WLEN),
        sl.values,
    )

    plot(sl, results, OUT_DIR / "01_toy_slice.png")


if __name__ == "__main__":
    main()
