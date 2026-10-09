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


def window_bounds(period: int, floor: int = MIN_WLEN_FLOOR, factor: float = 2.0) -> tuple[int, int]:
    """Motif length range from the estimated period. One rule for all series."""
    low = max(floor, int(period))
    return low, int(factor * low)


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


def fit_mask(make_model, signal: np.ndarray):
    """Fit one backend; return its (n_patterns, n_samples) mask and, for
    multivariate motifs, the dimensions of each pattern (None = not reported)."""
    model = make_model()
    model.fit(signal)
    return np.atleast_2d(np.asarray(model.prediction_mask_)), getattr(model, "prediction_dims_", None)


def discover(signal: np.ndarray, make_model, period: int, segment_len: int = SEGMENT_LEN,
             bounds: tuple[int, int] | None = None) -> list:
    """Run the backend on every segment. The slow step, so its output is cached.

    Returns one (start, end, write_start, write_end, occurrences) per segment,
    `occurrences` being (pattern, start, end) triples relative to the segment
    — (pattern, start, end, dims) when the backend reports dimensions —
    or None where the segment is too short to fit.
    """
    from representation import occurrences_from_mask

    signal = np.asarray(signal, dtype=np.float64)
    min_wlen, max_wlen = bounds or window_bounds(period)
    out = []
    for start, end, write_start, write_end in segment_spans(len(signal), segment_len, max_wlen):
        occ = None
        if end - start >= 4 * min_wlen:  # too short for motifs to be meaningful
            mask, dims = fit_mask(lambda: make_model(min_wlen, max_wlen), signal[start:end])
            occ = occurrences_from_mask(mask)
            if dims is not None:
                occ = [(p, s, e, tuple(dims[p])) for p, s, e in occ]
        out.append((start, end, write_start, write_end, occ))
    return out


def discover_cached(path, signal, make_model, period, segment_len: int = SEGMENT_LEN, bounds=None) -> list:
    """`discover`, pickled to `path`; motif backends here are deterministic."""
    import pickle
    from pathlib import Path

    path = Path(path)
    if path.exists():
        return pickle.loads(path.read_bytes())
    segments = discover(signal, make_model, period, segment_len, bounds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(segments))
    return segments


def score_segments(signal: np.ndarray, segments: list, mask_hook=None,
                   represent=distance_to_nearest_motif, **rep) -> np.ndarray:
    """Per-point distance-to-nearest-motif over a whole series, from `discover`.

    `mask_hook(mask)` may rewrite each segment's mask before scoring; the
    random-motif floor uses it to shuffle occurrence positions while keeping
    their number and lengths. `rep` goes to `represent`.
    """
    signal = np.asarray(signal, dtype=np.float64)
    out = np.zeros(len(signal))
    for start, end, write_start, write_end, occ in segments:
        if not occ:  # segment too short, or backend found nothing here
            continue
        mask = _segment_mask(occ, end - start)
        if mask_hook is not None:
            mask = mask_hook(mask)
        local = represent(signal[start:end], mask, **rep)
        out[write_start:write_end] = local[write_start - start : write_end - start]
    return out


def _segment_mask(occ: list, length: int) -> np.ndarray:
    mask = np.zeros((max(o[0] for o in occ) + 1, length), dtype=np.int8)
    for pattern, s, e, *_ in occ:
        mask[pattern, s : e + 1] = 1
    return mask


def pooled_occurrences(segments: list, mask_hook=None) -> list:
    """All segments' occurrences in series coordinates, one motif id per (segment, pattern).

    Occurrences keep their motif's dimensions (4th element) when discovery recorded them.
    """
    from representation import occurrences_from_mask

    out, next_id = [], 0
    for start, end, _, _, occ in segments:
        if not occ:
            continue
        dims = {o[0]: o[3] for o in occ if len(o) > 3}
        mask = _segment_mask(occ, end - start)
        if mask_hook is not None:
            mask = mask_hook(mask)
        out += [(next_id + p, start + s, start + e) + ((dims[p],) if dims else ())
                for p, s, e in occurrences_from_mask(mask)]
        next_id += mask.shape[0]
    return out


def segmented_score(signal, make_model, period, segment_len: int = SEGMENT_LEN, mask_hook=None, **rep):
    """`discover` + `score_segments` in one call, uncached."""
    return score_segments(signal, discover(signal, make_model, period, segment_len), mask_hook, **rep)


def shuffle_mask(mask: np.ndarray, rng: np.random.Generator, fraction: float = 1.0) -> np.ndarray:
    """RQ3 floor: same number of occurrences, same lengths, random positions.

    With `fraction` < 1 only that share of the occurrences (chosen at random) is
    moved and the rest stay where the motif method put them — the
    bank-degradation curve. `fraction=1` moves all of them, drawing exactly the
    same random numbers as before, so earlier random-floor results reproduce.
    """
    from representation import occurrences_from_mask

    n = mask.shape[1]
    occurrences = occurrences_from_mask(mask)
    move = np.ones(len(occurrences), bool)
    if fraction < 1:
        move[:] = False
        move[rng.permutation(len(occurrences))[: int(round(fraction * len(occurrences)))]] = True
    out = np.zeros_like(mask)
    for (pattern, start, end), moved in zip(occurrences, move):
        length = end - start + 1
        if moved:
            start = int(rng.integers(0, max(1, n - length)))
        out[pattern, start : start + length] = 1
    return out
