"""Whole-series motif -> representation -> score pipeline.

Two problems have to be solved to move off the toy slice:

1. **Scale.** The motif backends cannot be run on a 650k-point series, so the
   series is processed in fixed-length segments and the per-point scores are
   stitched back together. Every point is still scored — nothing is truncated
   or skipped — but motifs are discovered locally within each segment.

2. **Window length.** The toy slice hardcoded 200-400, which only made sense
   for 360 Hz ECG. Across 48 heterogeneous series the window is instead
   derived per series from TSB-AD's own period estimator (`find_length_rank`,
   the same one that sets `slidingWindow` for the metric), so no per-series
   choice is made by hand.

Segments overlap by a margin and each point takes its value from the segment
in which it is interior, which keeps the boundary bias documented in
`representation.distance_to_nearest_motif` out of the internal seams.
"""

from __future__ import annotations

import numpy as np

from representation import distance_to_nearest_motif

# Fixed a priori for every series and every backend. These are a compute
# budget, not tuned values: SEGMENT_LEN is what the slowest backend can fit in
# reasonable time, and the window rule follows the estimated period.
SEGMENT_LEN = 8000
MIN_WLEN_FLOOR = 16


def window_bounds(period: int) -> tuple[int, int]:
    """Motif length range from the estimated period. One rule for all series."""
    low = max(MIN_WLEN_FLOOR, int(period))
    return low, 2 * low


def segment_spans(n: int, segment_len: int, margin: int):
    """Overlapping (start, end, write_start, write_end) spans covering [0, n).

    `write_*` is the sub-range this segment owns, i.e. the part where its
    points are interior rather than boundary-biased.
    """
    if n <= segment_len:
        return [(0, n, 0, n)]
    stride = max(1, segment_len - 2 * margin)
    spans = []
    start = 0
    while start < n:
        end = min(start + segment_len, n)
        if end - start < 2 * margin + 2:  # tail too short to stand alone
            if spans:
                prev = spans[-1]
                spans[-1] = (prev[0], n, prev[2], n)
                break
        write_start = start if start == 0 else start + margin
        write_end = end if end == n else end - margin
        spans.append((start, end, write_start, write_end))
        if end == n:
            break
        start += stride
    return spans


def fit_mask(make_model, signal: np.ndarray) -> np.ndarray:
    """Fit one tsmd backend and return its (n_patterns, n_samples) mask."""
    model = make_model()
    model.fit(signal)
    return np.atleast_2d(np.asarray(model.prediction_mask_))


def segmented_score(
    signal: np.ndarray,
    make_model,
    period: int,
    segment_len: int = SEGMENT_LEN,
    mask_hook=None,
) -> np.ndarray:
    """Per-point distance-to-nearest-motif over a whole series.

    `mask_hook(mask)` may rewrite each segment's mask before scoring; the
    random-motif floor uses it to shuffle occurrence positions while keeping
    their number and lengths.
    """
    signal = np.asarray(signal, dtype=np.float64)
    min_wlen, max_wlen = window_bounds(period)
    margin = max_wlen
    out = np.zeros(len(signal))

    for start, end, write_start, write_end in segment_spans(len(signal), segment_len, margin):
        chunk = signal[start:end]
        if len(chunk) < 4 * min_wlen:  # too short for motifs to be meaningful
            out[write_start:write_end] = 0.0
            continue
        mask = fit_mask(lambda: make_model(min_wlen, max_wlen), chunk)
        if mask_hook is not None:
            mask = mask_hook(mask)
        if mask.sum() == 0:  # backend found nothing here
            out[write_start:write_end] = 0.0
            continue
        local = distance_to_nearest_motif(chunk, mask)
        out[write_start:write_end] = local[write_start - start : write_end - start]

    return out


def shuffle_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """RQ3 floor: same number of occurrences, same lengths, random positions."""
    from representation import occurrences_from_mask

    n = mask.shape[1]
    out = np.zeros_like(mask)
    for pattern, start, end in occurrences_from_mask(mask):
        length = end - start + 1
        new_start = int(rng.integers(0, max(1, n - length)))
        out[pattern, new_start : new_start + length] = 1
    return out
