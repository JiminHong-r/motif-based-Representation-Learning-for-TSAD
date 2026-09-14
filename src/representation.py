"""RQ1, minimum viable version: distance-to-nearest-motif representation.

`tsmd` backends expose `prediction_mask_`, a binary array of shape
(n_patterns, n_samples) marking which samples belong to occurrences of each
discovered motif. This module turns that mask plus the raw signal into a
per-point representation.

"Distance to the nearest motif occurrence" is read here as a *shape* distance,
not a temporal one: each sliding window is compared against the set of
discovered motif occurrences under z-normalised Euclidean distance. A window
that looks like some occurrence scores low; one that looks like none scores
high. (The purely temporal reading — how many samples away the nearest covered
point is — carries no shape information and is degenerate inside motifs, so it
is only useful as a sanity baseline. `temporal_distance` provides it.)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def occurrences_from_mask(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous occurrence segments as (pattern_index, start, end) inclusive."""
    mask = np.atleast_2d(np.asarray(mask))
    segments = []
    for pattern, row in enumerate(mask):
        padded = np.concatenate(([0], (row > 0).astype(np.int8), [0]))
        edges = np.diff(padded)
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
            segments.append((pattern, int(start), int(end)))
    return segments


def _znorm(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-normalise each row independently; flat rows become all-zero."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return (windows - mean) / np.maximum(std, eps)


def _sliding_windows(signal: np.ndarray, wlen: int) -> np.ndarray:
    """All length-`wlen` windows, shape (n - wlen + 1, wlen)."""
    return np.lib.stride_tricks.sliding_window_view(signal, wlen)


def _resample(segment: np.ndarray, wlen: int) -> np.ndarray:
    """Linearly resample a segment to exactly `wlen` points.

    Occurrences vary in length (LoCoMotif warps them especially), so they are
    put on a common length before being compared.
    """
    if len(segment) == wlen:
        return segment
    src = np.linspace(0.0, 1.0, num=len(segment))
    dst = np.linspace(0.0, 1.0, num=wlen)
    return np.interp(dst, src, segment)


def motif_references(
    signal: np.ndarray, mask: np.ndarray, wlen: int | None = None
) -> tuple[np.ndarray, int]:
    """Z-normalised, length-aligned motif occurrences to compare windows against.

    Returns (references of shape (n_occurrences, wlen), wlen).
    """
    segments = occurrences_from_mask(mask)
    if not segments:
        raise ValueError("mask contains no motif occurrences")
    if wlen is None:
        wlen = int(np.median([end - start + 1 for _, start, end in segments]))
    wlen = max(4, min(wlen, len(signal)))
    stacked = np.vstack([_resample(signal[start : end + 1], wlen) for _, start, end in segments])
    return _znorm(stacked), wlen


def distance_to_nearest_motif(
    signal: np.ndarray,
    mask: np.ndarray,
    wlen: int | None = None,
    reduce: str = "min",
    fix_edge_bias: bool = True,
) -> np.ndarray:
    """Per-point distance to the nearest discovered motif occurrence.

    Each length-`wlen` window is z-normalised and matched against every motif
    occurrence; the window keeps its smallest distance. Window distances are
    then folded back onto points: every point takes the `reduce` (default
    "min") over the windows covering it, so a point is only called far-from-
    motif when *no* window containing it matches anything.

    `fix_edge_bias` corrects a systematic artefact at the two boundaries. An
    interior point is covered by `wlen` windows, but point i < wlen-1 is
    covered by only i+1 of them, so its minimum is taken over a smaller
    candidate set and is biased upward — and a slice almost always starts and
    ends mid-pattern, which inflates those few windows further. Left alone the
    boundary dominates the score range, flattening the real anomaly under
    min-max scaling and capturing any global threshold. The under-covered
    points are therefore not treated as evaluable: they are filled with the
    median of the fully covered interior.
    """
    signal = np.asarray(signal, dtype=np.float64)
    references, wlen = motif_references(signal, mask, wlen)

    windows = _znorm(_sliding_windows(signal, wlen))
    window_distance = cdist(windows, references, metric="euclidean").min(axis=1)

    point_distance = _fold_to_points(window_distance, len(signal), wlen, reduce)
    if fix_edge_bias:
        point_distance = _fill_edges(point_distance, wlen)
    return point_distance


def _fill_edges(distance: np.ndarray, wlen: int) -> np.ndarray:
    """Replace the under-covered boundary points with the interior median."""
    margin = wlen - 1
    if margin <= 0 or 2 * margin >= len(distance):
        return distance
    out = distance.copy()
    interior = out[margin : len(out) - margin]
    out[:margin] = np.median(interior)
    out[len(out) - margin :] = np.median(interior)
    return out


def _fold_to_points(
    window_distance: np.ndarray, n: int, wlen: int, reduce: str
) -> np.ndarray:
    """Spread per-window values over the points each window covers."""
    if reduce not in {"min", "max", "mean"}:
        raise ValueError(f"unknown reduce: {reduce}")

    if reduce == "mean":
        total = np.zeros(n)
        count = np.zeros(n)
        for offset in range(wlen):
            total[offset : offset + len(window_distance)] += window_distance
            count[offset : offset + len(window_distance)] += 1
        return total / np.maximum(count, 1)

    fill = np.inf if reduce == "min" else -np.inf
    out = np.full(n, fill)
    op = np.minimum if reduce == "min" else np.maximum
    for offset in range(wlen):
        chunk = slice(offset, offset + len(window_distance))
        out[chunk] = op(out[chunk], window_distance)
    # Points covered by no full window (only possible if wlen > n) fall back.
    uncovered = ~np.isfinite(out)
    if uncovered.any():
        out[uncovered] = window_distance.max()
    return out


def temporal_distance(mask: np.ndarray) -> np.ndarray:
    """Sanity baseline: samples to the nearest motif-covered point."""
    covered = np.atleast_2d(np.asarray(mask)).any(axis=0)
    if not covered.any():
        return np.full(covered.shape, float(len(covered)))
    index = np.arange(len(covered), dtype=np.float64)
    covered_index = index[covered]
    nearest = np.searchsorted(covered_index, index).clip(0, len(covered_index) - 1)
    previous = (nearest - 1).clip(0, len(covered_index) - 1)
    return np.minimum(
        np.abs(index - covered_index[nearest]), np.abs(index - covered_index[previous])
    )
