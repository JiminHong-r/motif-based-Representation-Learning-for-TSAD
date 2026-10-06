"""Motif-discovery backends from `tsmd`, with one settings rule for all of them.

The TSMD benchmark (Guerrini et al.) sets the number of motifs and the radius
from ground-truth motif labels, which anomaly data does not have. Here every
backend gets the same a-priori rule instead (PROJECT.md §5, WP1):

* motif length: `pipeline.window_bounds` (period to 2 x period); backends
  taking a single length use the lower bound;
* number of motifs: 10 — the size of KMeansAD's bank (10 centroids);
* radius: matrix-profile backends keep their own rule (3 x the closest motif
  pair, `radius_ratio=3`); SetFinder and LatentMotif, which need an absolute
  radius, get that same quantity computed on the series they are fitted on;
* distance: `UnitEuclidean` (z-normalised Euclidean / sqrt(length)), the tsmd
  default; everything else at the package defaults.

Each factory is `make(min_wlen, max_wlen, **overrides)` returning an object
with `.fit(signal)` and `.prediction_mask_`, as `pipeline.discover` expects.
"""

from __future__ import annotations

import numpy as np

N_PATTERNS = 10


def closest_pair_radius(signal: np.ndarray, wlen: int, ratio: float = 3.0) -> float:
    """`ratio` x the smallest z-normalised distance between two non-trivial windows, in UnitEuclidean units."""
    import stumpy

    profile = stumpy.stump(np.asarray(signal, dtype=np.float64), m=wlen)[:, 0].astype(float)
    profile = profile[np.isfinite(profile)]
    return ratio * float(profile.min()) / np.sqrt(wlen) if len(profile) else 0.0


class _AbsoluteRadius:
    """Fit-time radius for backends that take an absolute one (SetFinder, LatentMotif)."""

    def __init__(self, cls, wlen: int, **kwargs):
        self.cls, self.wlen, self.kwargs = cls, wlen, kwargs

    def fit(self, signal):
        radius = self.kwargs.pop("radius", None) or closest_pair_radius(signal, self.wlen)
        self.model_ = self.cls(radius=radius, wlen=self.wlen, **self.kwargs)
        self.model_.fit(signal)
        return self

    @property
    def prediction_mask_(self):
        return self.model_.prediction_mask_


def _lazy(module: str, name: str):
    """Import a tsmd class only when its backend is used (some need extra packages)."""
    def cls(*args, **kwargs):
        import importlib

        return getattr(importlib.import_module(f"tsmd.competitors.{module}"), name)(*args, **kwargs)
    return cls


def factories() -> dict:
    LocoMotif = _lazy("locomotif", "LocoMotif")
    Motiflets_numba = _lazy("motiflets_numba", "Motiflets_numba")
    MatrixProfile = _lazy("matrixprofile", "MatrixProfile")
    PanMatrixProfile = _lazy("panmatrixprofile", "PanMatrixProfile")
    Valmod = _lazy("valmod", "Valmod")
    Baseline = _lazy("setfinder", "Baseline")
    LatentMotif = _lazy("latentmotifs", "LatentMotif")
    BasePersistentPattern = _lazy("persistence", "BasePersistentPattern")
    AdaptativeBasePersistentPattern = _lazy("adaptative_persistence", "AdaptativeBasePersistentPattern")
    MDL = _lazy("mdl", "MDL")

    k, dist = N_PATTERNS, "UnitEuclidean"
    return {
        "LoCoMotif": lambda lo, hi, **kw: LocoMotif(min_wlen=lo, max_wlen=hi, **kw),
        "k-Motiflets": lambda lo, hi, **kw: Motiflets_numba(**{"k_max": 10, **kw}, min_wlen=lo, max_wlen=hi),
        "STOMP": lambda lo, hi, **kw: MatrixProfile(**{"n_patterns": k, "distance_name": dist, **kw}, wlen=lo),
        "PanMP": lambda lo, hi, **kw: PanMatrixProfile(**{"n_patterns": k, "distance_name": dist, **kw}, min_wlen=lo, max_wlen=hi),
        "VALMOD": lambda lo, hi, **kw: Valmod(**{"n_patterns": k, "distance_name": dist, **kw}, min_wlen=lo, max_wlen=hi),
        "SetFinder": lambda lo, hi, **kw: _AbsoluteRadius(Baseline, lo, **{"n_patterns": k, "distance_name": dist, **kw}),
        "LatentMotif": lambda lo, hi, **kw: _AbsoluteRadius(LatentMotif, lo, **{"n_patterns": k, **kw}),
        # PEPA: crashes on this data (fixed n_patterns: IndexError; automatic: NaN in its threshold) — A-PEPA represents the family
        "PEPA": lambda lo, hi, **kw: BasePersistentPattern(wlen_for_persistence=lo, **kw),
        "A-PEPA": lambda lo, hi, **kw: AdaptativeBasePersistentPattern(wlen_for_persistence=lo, **kw),
        "MDL-Clust": lambda lo, hi, **kw: MDL(min_wlen=lo, max_wlen=hi, **kw),
    }
