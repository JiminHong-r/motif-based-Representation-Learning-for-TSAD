"""Loading and slicing of TSB-AD-U univariate series.

TSB-AD-U files are flat CSVs with two columns, `Data` and `Label`, one row per
timestamp. Filenames encode metadata, e.g.

    170_MITDB_id_1_Medical_tr_17675_1st_17775.csv
                                  ^train end  ^first anomaly index
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "TSB-AD-U"


@dataclass
class Series:
    """A (sliced) univariate series with its anomaly labels."""

    name: str
    values: np.ndarray  # float64, shape (n,)
    labels: np.ndarray  # int, shape (n,), 1 = anomalous
    offset: int = 0  # index of values[0] in the original recording

    def __len__(self) -> int:
        return len(self.values)

    @property
    def anomaly_runs(self) -> list[tuple[int, int]]:
        """Contiguous anomalous index ranges, inclusive, local to this slice."""
        return _runs(self.labels)


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1s as (start, end) inclusive index pairs."""
    padded = np.concatenate(([0], (labels > 0).astype(np.int8), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def load_series(filename: str, data_dir: Path | None = None) -> Series:
    """Load one full TSB-AD-U recording."""
    path = (data_dir or DATA_DIR) / filename
    frame = pd.read_csv(path)
    return Series(
        name=path.stem,
        values=frame["Data"].to_numpy(dtype=np.float64),
        labels=frame["Label"].to_numpy(dtype=np.int64),
    )


def slice_around(series: Series, centre: int, width: int) -> Series:
    """Take a `width`-point window centred on `centre`, clipped to the series."""
    half = width // 2
    start = max(0, centre - half)
    end = min(len(series), start + width)
    start = max(0, end - width)  # keep full width when clipped at the right edge
    return Series(
        name=f"{series.name}[{start}:{end}]",
        values=series.values[start:end],
        labels=series.labels[start:end],
        offset=start,
    )


def slice_around_first_anomaly(series: Series, width: int = 6000) -> Series:
    """Window centred on the midpoint of the first labelled anomaly run."""
    runs = series.anomaly_runs
    if not runs:
        raise ValueError(f"{series.name} has no labelled anomalies")
    start, end = runs[0]
    return slice_around(series, centre=(start + end) // 2, width=width)
