"""Phase 1, steps 4-6: motif mask -> representation -> score -> VUS-PR.

Closes the thin vertical slice. The question this answers is narrow: does the
whole chain run end to end and produce a *non-degenerate* VUS-PR (neither 0
nor trivially 1)? It is not yet evidence that the approach works.

    .venv/bin/python notebooks/02_slice_to_vuspr.py
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
from representation import distance_to_nearest_motif, occurrences_from_mask  # noqa: E402
from scoring import evaluate, minmax, threshold_predictions  # noqa: E402

SERIES = "170_MITDB_id_1_Medical_tr_17675_1st_17775.csv"
WIDTH = 6000
MIN_WLEN, MAX_WLEN = 200, 400
OUT_DIR = ROOT / "results"
CACHE = OUT_DIR / "masks_01.npz"
HEADLINE = ["VUS-PR", "AUC-PR", "VUS-ROC", "AUC-ROC", "Standard-F1", "Affiliation-F"]


def fit_masks(signal):
    """Fit both backends (cached — k-Motiflets takes ~1 min)."""
    if CACHE.exists():
        cached = np.load(CACHE)
        print(f"loaded cached masks from {CACHE.name}")
        return {k: cached[k] for k in cached.files}

    from tsmd.competitors.locomotif import LocoMotif
    from tsmd.competitors.motiflets import Motiflets

    backends = {
        "k-Motiflets": Motiflets(k_max=10, min_wlen=MIN_WLEN, max_wlen=MAX_WLEN),
        "LoCoMotif": LocoMotif(min_wlen=MIN_WLEN, max_wlen=MAX_WLEN),
    }
    masks = {}
    for name, model in backends.items():
        start = time.perf_counter()
        model.fit(signal)
        masks[name] = np.atleast_2d(np.asarray(model.prediction_mask_))
        print(f"fitted {name} in {time.perf_counter() - start:.1f}s -> {masks[name].shape}")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **masks)
    return masks


def random_baseline(reference_mask, seed=0):
    """RQ3's floor: motifs of the same size and count, placed at random."""
    rng = np.random.default_rng(seed)
    n = reference_mask.shape[1]
    mask = np.zeros_like(reference_mask)
    for pattern, start, end in occurrences_from_mask(reference_mask):
        length = end - start + 1
        new_start = int(rng.integers(0, max(1, n - length)))
        mask[pattern, new_start : new_start + length] = 1
    return mask


def main():
    series = load_series(SERIES)
    sl = slice_around_first_anomaly(series, width=WIDTH)
    print(f"slice: {len(sl)} pts at offset {sl.offset}, "
          f"{int(sl.labels.sum())} anomalous, runs {sl.anomaly_runs}")

    from TSB_AD.utils.slidingWindows import find_length_rank

    sliding_window = find_length_rank(sl.values.reshape(-1, 1), rank=1)
    print(f"TSB-AD estimated period / slidingWindow = {sliding_window}")

    masks = fit_masks(sl.values)
    masks["random-motif (floor)"] = random_baseline(masks["LoCoMotif"])

    rows, scores = {}, {}
    for name, mask in masks.items():
        distance = distance_to_nearest_motif(sl.values, mask)
        score = minmax(distance)
        pred = threshold_predictions(score, k=3.0)
        metrics = evaluate(score, sl.labels, sliding_window, pred=pred)
        rows[name] = metrics
        scores[name] = score
        print(f"\n{name}: {mask.shape[0]} motifs, "
              f"{len(occurrences_from_mask(mask))} occurrences, "
              f"{int(pred.sum())} points flagged")
        for key in HEADLINE:
            if key in metrics:
                print(f"    {key:<16} {metrics[key]:.4f}")

    print("\n" + "=" * 72)
    print(f"{'backend':<24}" + "".join(f"{k:>14}" for k in HEADLINE[:4]))
    print("-" * 72)
    for name, metrics in rows.items():
        print(f"{name:<24}" + "".join(f"{metrics.get(k, float('nan')):>14.4f}" for k in HEADLINE[:4]))
    print("=" * 72)

    plot(sl, scores, rows, OUT_DIR / "02_scores.png")


def plot(sl, scores, rows, path):
    fig, axes = plt.subplots(len(scores) + 1, 1, figsize=(15, 2.4 * (len(scores) + 1)), sharex=True)
    t = np.arange(len(sl))
    axes[0].plot(t, sl.values, lw=0.6, color="black")
    axes[0].set_ylabel("ECG")
    axes[0].set_title(f"{sl.name} — red = labelled anomaly")

    for ax, (name, score) in zip(axes[1:], scores.items()):
        ax.plot(t, score, lw=0.8, color="tab:blue")
        ax.set_ylabel("score")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name}  —  VUS-PR {rows[name].get('VUS-PR', float('nan')):.3f}")

    for ax in axes:
        for start, end in sl.anomaly_runs:
            ax.axvspan(start, end, color="red", alpha=0.18, lw=0)

    axes[-1].set_xlabel("sample index within slice")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
