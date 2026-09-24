"""RQ2, minimum viable version: turn a representation into an anomaly score.

The distance-to-nearest-motif representation is already monotone in
"anomalousness", so the minimal scorer is the identity plus a normalisation.
Anything smarter (kNN / IsolationForest over the representation) belongs to
Phase 3.
"""

from __future__ import annotations

import numpy as np


def minmax(score: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant score collapses to all zeros."""
    score = np.asarray(score, dtype=np.float64)
    low, high = np.nanmin(score), np.nanmax(score)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(score)
    return (score - low) / (high - low)


def evaluate(score: np.ndarray, labels: np.ndarray, sliding_window: int) -> dict:
    """TSB-AD's own metric suite, including VUS-PR."""
    from TSB_AD.evaluation.metrics import get_metrics

    return get_metrics(
        np.asarray(score, dtype=np.float64),
        np.asarray(labels).astype(int),
        slidingWindow=int(sliding_window),
    )
