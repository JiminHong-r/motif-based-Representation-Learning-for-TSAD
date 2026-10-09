"""RQ1, minimum viable version: distance-to-nearest-motif representation.

`tsmd` backends expose `prediction_mask_`, a binary array of shape
(n_patterns, n_samples) marking which samples belong to occurrences of each
discovered motif. This module turns that mask plus the raw signal into a
per-point representation.

The occurrences of all motifs together form a library of "normal" shapes.
Each sliding window is compared against that library under z-normalised
Euclidean distance: a window that looks like some occurrence scores low, one
that looks like none scores high.

`norm` controls what "looks like" means, one knob from shape-only to raw:

* ``"shape"``   z-normalised windows: shape only. Amplitude or level anomalies
  with an ordinary shape are invisible (see 004_NAB in PROJECT.md).
* ``"centred"`` mean removed, scale kept: shape + amplitude, tolerant to a
  drifting baseline.
* ``"raw"``     no normalisation: shape + amplitude + level.
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


def _normalise(windows: np.ndarray, norm: str) -> np.ndarray:
    if norm == "shape":
        return _znorm(windows)
    if norm == "centred":
        return windows - windows.mean(axis=1, keepdims=True)
    if norm == "raw":
        return windows
    raise ValueError(f"unknown norm {norm!r}")


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


def motif_references(signal: np.ndarray, mask: np.ndarray, norm: str = "shape") -> tuple[np.ndarray, int]:
    """The library: normalised, length-aligned motif occurrences.

    The common length is the median occurrence length. Returns
    (references of shape (n_occurrences, wlen), wlen).
    """
    segments = occurrences_from_mask(mask)
    if not segments:
        raise ValueError("mask contains no motif occurrences")
    wlen = int(np.median([end - start + 1 for _, start, end in segments]))
    wlen = max(4, min(wlen, len(signal)))
    stacked = np.vstack([_resample(signal[start : end + 1], wlen) for _, start, end in segments])
    return _normalise(stacked, norm), wlen


def distance_to_nearest_motif(
    signal: np.ndarray, mask: np.ndarray, norm: str = "shape",
    point: str = "min", exclude_self: bool = False,
) -> np.ndarray:
    """Per-point distance to the nearest discovered motif occurrence.

    Each length-`wlen` window is normalised (see module docstring) and matched
    against every occurrence in the library; the window keeps its smallest
    distance. Window distances then become per-point scores via `point`:

    * ``"min"``    each point takes the minimum over the windows covering it
      (a point is far only when *no* window containing it matches). Smears:
      a short anomaly always shares some mostly-normal window, so it is hidden.
    * ``"centre"`` each point takes the distance of the window centred on it,
      as TSB-AD's MatrixProfile does.
    * ``"max"``    each point takes the maximum over the windows covering it
      (a point is far when *any* window containing it is far).

    `exclude_self` ignores library entries that overlap the window being
    scored. Without it, a window lying on an occurrence matches itself at
    distance ~0, so the score mostly says "not inside an occurrence" — true of
    random occurrences too. MatrixProfile's exclusion zone does the same job.

    Boundary correction: points near the ends are covered by fewer windows (or
    have no centred window), and a series or segment almost always starts and
    ends mid-pattern. Uncorrected, the boundary dominates the score range, so
    those points are filled with the median of the interior.
    """
    signal = np.asarray(signal, dtype=np.float64)
    references, wlen = motif_references(signal, mask, norm)

    windows = _normalise(np.lib.stride_tricks.sliding_window_view(signal, wlen), norm)
    distance = cdist(windows, references, metric="euclidean")
    if exclude_self:
        occ = np.array([(s, e) for _, s, e in occurrences_from_mask(mask)])
        t = np.arange(len(windows))[:, None]
        distance[(occ[:, 0] <= t + wlen - 1) & (occ[:, 1] >= t)] = np.inf
        # a window overlapping every occurrence has nothing left to match
        distance[np.isinf(distance).all(axis=1)] = np.max(distance[np.isfinite(distance)], initial=0.0)
    window_distance = distance.min(axis=1)

    n = len(signal)
    return window_to_points(window_distance, n, wlen, point)


def _over_covering_windows(window_distance: np.ndarray, n: int, wlen: int, reduce) -> np.ndarray:
    """Point i takes `reduce` (np.minimum / np.maximum) over every window that contains it."""
    out = np.full(n, np.inf if reduce is np.minimum else -np.inf)
    for offset in range(wlen):
        chunk = slice(offset, offset + len(window_distance))
        out[chunk] = reduce(out[chunk], window_distance)
    return out


def _fill_edges(distance: np.ndarray, margin: int) -> np.ndarray:
    """Replace the first and last `margin` points with the interior median."""
    if margin <= 0 or 2 * margin >= len(distance):
        return distance
    out = distance.copy()
    interior = np.median(out[margin : len(out) - margin])
    out[:margin] = interior
    out[len(out) - margin :] = interior
    return out


def uncovered_fraction(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RQ1 occurrence-based variant: how little of each point's neighbourhood motifs cover.

    A motif is by definition something that repeats and an anomaly by
    definition does not, so anomalies should fall outside motif occurrences.
    Point i scores the fraction of the `wlen`-point window centred on it that no
    occurrence covers (`wlen` as in `motif_references`). Uses only *where* the
    motifs are, so random occurrences carry no information here.
    """
    occurrences = occurrences_from_mask(mask)
    if not occurrences:
        raise ValueError("mask contains no motif occurrences")
    wlen = int(np.median([end - start + 1 for _, start, end in occurrences]))
    covered = (np.atleast_2d(mask) > 0).any(axis=0).astype(np.float64)
    return 1.0 - np.convolve(covered, np.ones(wlen) / wlen, mode="same")


def motif_features(signal: np.ndarray, occurrences: list, norm: str = "centred", budget: float = 2e7):
    """Per-motif distances for every window; univariate or multivariate.

    `occurrences` are (motif_id, start, end) or (motif_id, start, end, dims),
    in `signal` coordinates; `dims` is the tuple of columns the motif lives in
    (None = all). A window is compared with an occurrence only on that
    occurrence's dimensions: each dimension normalised separately, the
    per-dimension windows concatenated, and the Euclidean distance divided by
    sqrt(len(dims)) so motifs on different numbers of dimensions are comparable.
    For a univariate series this is plain distance on one dimension.

    Column k is the window's distance to its nearest occurrence of motif k
    (like KMeansAD's distance to each centroid); occurrences overlapping the
    window are excluded (no self-matches). Returns (features, wlen).
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    occ = sorted((o[0], o[1], o[2], tuple(range(x.shape[1])) if len(o) < 4 or o[3] is None else tuple(o[3]))
                 for o in occurrences)
    if not occ:
        raise ValueError("no motif occurrences")
    ids = np.array([o[0] for o in occ])
    motif_ids = np.unique(ids)
    wlen = max(4, min(int(np.median([e - s + 1 for _, s, e, _ in occ])), len(x)))
    n_windows = len(x) - wlen + 1
    out = np.full((n_windows, len(motif_ids)), np.inf)

    def windows(dims, rows):  # (len(rows), len(dims) * wlen), each dimension normalised separately
        w = np.lib.stride_tricks.sliding_window_view(x[:, dims], wlen, axis=0)[rows]  # (k, d, wlen)
        return _normalise(w.reshape(-1, wlen), norm).reshape(len(w), -1) / np.sqrt(len(dims))

    for dims in sorted({o[3] for o in occ}):
        group = [i for i, o in enumerate(occ) if o[3] == dims]
        d = list(dims)
        refs = np.vstack([
            _normalise(np.stack([_resample(x[s:e + 1, j], wlen) for j in d]), norm).reshape(-1) / np.sqrt(len(d))
            for _, s, e, _ in (occ[i] for i in group)
        ])
        starts = np.array([occ[i][1] for i in group]); ends = np.array([occ[i][2] for i in group])
        col = np.searchsorted(motif_ids, ids[group])
        chunk = max(256, int(budget // (wlen * len(d) + len(group))))
        for c in range(0, n_windows, chunk):
            rows = np.arange(c, min(c + chunk, n_windows))
            distance = cdist(windows(d, rows), refs)
            distance[(starts <= rows[:, None] + wlen - 1) & (ends >= rows[:, None])] = np.inf
            for k in np.unique(col):
                out[rows, k] = np.minimum(out[rows, k], distance[:, col == k].min(axis=1))
    # a window overlapping every occurrence of a motif has nothing left to match
    finite = np.where(np.isfinite(out), out, -np.inf).max(axis=0)
    return np.where(np.isfinite(out), out, np.maximum(finite, 0.0)), wlen


def window_to_points(window_score: np.ndarray, n: int, wlen: int, point: str = "centre") -> np.ndarray:
    """Window scores -> point scores; `point` as in `distance_to_nearest_motif`.

    Under-covered edge points are filled with the interior median.
    """
    if point in ("min", "max"):
        reduce = np.minimum if point == "min" else np.maximum
        return _fill_edges(_over_covering_windows(window_score, n, wlen, reduce), wlen - 1)
    if point == "centre":
        half = wlen // 2
        out = np.full(n, np.median(window_score))
        out[half : half + len(window_score)] = window_score
        return _fill_edges(out, half)
    raise ValueError(f"unknown point {point!r}")
