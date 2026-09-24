"""RQ1, minimum viable version: distance-to-nearest-motif representation.

`tsmd` backends expose `prediction_mask_`, a binary array of shape
(n_patterns, n_samples) marking which samples belong to occurrences of each
discovered motif. This module turns that mask plus the raw signal into a
per-point representation.

The occurrences of all motifs together form a library of "normal" shapes.
Each sliding window is compared against that library under z-normalised
Euclidean distance: a window that looks like some occurrence scores low, one
that looks like none scores high. Z-normalisation makes the comparison
shape-only, so amplitude or level anomalies with an ordinary shape are not
caught by this representation.
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


def motif_references(signal: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The library: z-normalised, length-aligned motif occurrences.

    The common length is the median occurrence length. Returns
    (references of shape (n_occurrences, wlen), wlen).
    """
    segments = occurrences_from_mask(mask)
    if not segments:
        raise ValueError("mask contains no motif occurrences")
    wlen = int(np.median([end - start + 1 for _, start, end in segments]))
    wlen = max(4, min(wlen, len(signal)))
    stacked = np.vstack([_resample(signal[start : end + 1], wlen) for _, start, end in segments])
    return _znorm(stacked), wlen


def distance_to_nearest_motif(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-point distance to the nearest discovered motif occurrence.

    Each length-`wlen` window is z-normalised and matched against every
    occurrence in the library; the window keeps its smallest distance. Each
    point then takes the minimum over the windows covering it, so a point is
    only called far-from-motif when *no* window containing it matches anything.

    Boundary correction: an interior point is covered by `wlen` windows, but
    point i < wlen-1 is covered by only i+1 of them, so its minimum is taken
    over fewer candidates and is biased upward — and a series or segment almost
    always starts and ends mid-pattern, which inflates those windows further.
    Uncorrected, the boundary dominates the score range and captures any global
    threshold, so those under-covered points are filled with the median of the
    fully covered interior.
    """
    signal = np.asarray(signal, dtype=np.float64)
    references, wlen = motif_references(signal, mask)

    windows = _znorm(np.lib.stride_tricks.sliding_window_view(signal, wlen))
    window_distance = cdist(windows, references, metric="euclidean").min(axis=1)

    return _fill_edges(_min_over_covering_windows(window_distance, len(signal), wlen), wlen)


def _min_over_covering_windows(window_distance: np.ndarray, n: int, wlen: int) -> np.ndarray:
    """Point i takes the minimum distance over every window that contains it."""
    out = np.full(n, np.inf)
    for offset in range(wlen):
        chunk = slice(offset, offset + len(window_distance))
        out[chunk] = np.minimum(out[chunk], window_distance)
    return out


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
