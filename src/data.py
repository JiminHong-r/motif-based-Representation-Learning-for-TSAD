"""Loading of TSB-AD-U univariate series.

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
    """A univariate series with its anomaly labels."""

    name: str
    values: np.ndarray  # float64, shape (n,)
    labels: np.ndarray  # int, shape (n,), 1 = anomalous

    def __len__(self) -> int:
        return len(self.values)


def load_series(filename: str, data_dir: Path | None = None) -> Series:
    """Load one full TSB-AD-U recording."""
    path = (data_dir or DATA_DIR) / filename
    frame = pd.read_csv(path)
    return Series(
        name=path.stem,
        values=frame["Data"].to_numpy(dtype=np.float64),
        labels=frame["Label"].to_numpy(dtype=np.int64),
    )
