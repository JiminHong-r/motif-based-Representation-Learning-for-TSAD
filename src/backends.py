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

Multivariate (TSB-AD-M), three ways to get a motif bank — `MULTIVARIATE`:

* per dimension: a univariate method on each dimension separately
  (`PerDimension`), every motif living in one dimension;
* all dimensions: one motif spans every dimension (LoCoMotif's native
  multivariate mode; mSTAMP with all dimensions);
* subset of dimensions: each motif lives in the dimensions the method selects
  (mSTAMP k-dimensional motifs, Yeh et al. 2017, subspace chosen by MDL;
  SubTSMD, Carpentier et al. 2026: LoCoMotif per dimension, then motifs combined
  bottom-up into subspace motifs, allowing small time shifts between dimensions).

Multivariate backends also set `.prediction_dims_`: one tuple of column
indices per mask row. Constant columns are dropped before discovery.
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


def _varying(X: np.ndarray) -> np.ndarray:
    """Columns that are not constant (z-normalised distances are undefined on constant ones)."""
    return np.flatnonzero(np.nanstd(X, axis=0) > 0)


class PerDimension:
    """A univariate backend run on every dimension separately."""

    def __init__(self, make_univariate):
        self.make = make_univariate

    def fit(self, X):
        rows, dims = [], []
        for j in _varying(X):
            try:
                model = self.make(); model.fit(X[:, j])
                mask = np.atleast_2d(np.asarray(model.prediction_mask_))
            except Exception:  # a dimension where the method fails or finds nothing
                continue
            rows.append(mask); dims += [(int(j),)] * len(mask)
        self.prediction_mask_ = np.vstack(rows) if rows else np.zeros((1, len(X)))
        self.prediction_dims_ = dims or [(0,)]
        return self


class MSTAMP:
    """mSTAMP motifs (stumpy.mstump + stumpy.mmotifs, package defaults otherwise).

    `subspace=True`: each motif's dimensions chosen by MDL (subset of attributes);
    `subspace=False`: every motif uses all varying dimensions.
    """

    def __init__(self, wlen: int, n_patterns: int = N_PATTERNS, max_matches: int = 10, subspace: bool = True):
        self.wlen, self.n_patterns, self.max_matches, self.subspace = wlen, n_patterns, max_matches, subspace

    def fit(self, X):
        import stumpy

        cols = _varying(X)
        T = np.ascontiguousarray(X[:, cols].T)
        P, I = stumpy.mstump(T, m=self.wlen)
        kw = {} if self.subspace else {"k": len(cols) - 1}
        _, indices, subspaces, _ = stumpy.mmotifs(T, P, I, max_motifs=self.n_patterns, max_matches=self.max_matches, **kw)
        mask, dims = [], []
        for starts, sub in zip(indices, subspaces):
            row = np.zeros(len(X))
            for st in np.asarray(starts):
                if st >= 0:
                    row[st : st + self.wlen] = 1
            mask.append(row); dims.append(tuple(int(cols[k]) for k in np.asarray(sub)))
        self.prediction_mask_ = np.vstack(mask) if mask else np.zeros((1, len(X)))
        self.prediction_dims_ = dims or [tuple(int(c) for c in cols)]
        return self


class AllDimensions:
    """A natively multivariate backend whose motifs span every (varying) dimension."""

    def __init__(self, model):
        self.model = model

    def fit(self, X):
        cols = _varying(X)
        self.model.fit(X[:, cols])
        self.prediction_mask_ = np.atleast_2d(np.asarray(self.model.prediction_mask_))
        self.prediction_dims_ = [tuple(int(c) for c in cols)] * len(self.prediction_mask_)
        return self


class SubTSMDBank:
    """SubTSMD (external/SubTSMD, package defaults) as a bank.

    SubTSMD lets a motif's dimensions be slightly shifted in time; the bank
    stores one span per occurrence, so each occurrence keeps the span covering
    all its dimensions.
    """

    def __init__(self, l_min: int, l_max: int, **kwargs):
        self.l_min, self.l_max, self.kwargs = l_min, l_max, kwargs

    def fit(self, X):
        import numba

        if not hasattr(numba, "bool"):  # SubTSMD uses the alias numba.bool, removed in newer numba
            numba.bool = numba.boolean
        from sub_tsmd import SubTSMD

        cols = _varying(X)
        motif_sets = SubTSMD(l_min=self.l_min, l_max=self.l_max, **self.kwargs).apply(X[:, cols])
        mask, dims = [], []
        for motif_set in motif_sets:
            row = np.zeros(len(X))
            for motif in motif_set:
                row[int(motif.start()) : int(motif.end()) + 1] = 1
            mask.append(row); dims.append(tuple(int(cols[k]) for k in np.flatnonzero(motif_set.mask)))
        self.prediction_mask_ = np.vstack(mask) if mask else np.zeros((1, len(X)))
        self.prediction_dims_ = dims or [tuple(int(c) for c in cols)]
        return self


def multivariate_factories() -> dict:
    uni = factories()
    out = {f"{name}/per-dim": (lambda name: lambda lo, hi, **kw: PerDimension(lambda: uni[name](lo, hi, **kw)))(name)
           for name in ["VALMOD", "PEPA", "A-PEPA", "STOMP", "LoCoMotif"]}
    out["LoCoMotif/all-dims"] = lambda lo, hi, **kw: AllDimensions(uni["LoCoMotif"](lo, hi, **kw))
    out["mSTAMP/all-dims"] = lambda lo, hi, **kw: MSTAMP(lo, subspace=False, **kw)
    out["mSTAMP/subset"] = lambda lo, hi, **kw: MSTAMP(lo, subspace=True, **kw)
    out["SubTSMD/subset"] = lambda lo, hi, **kw: SubTSMDBank(lo, hi, **kw)
    return out
