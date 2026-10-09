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
DATA_DIR_M = DATA_DIR.with_name("TSB-AD-M")


@dataclass
class Series:
    """A series with its anomaly labels. `values` is (n,) for TSB-AD-U, (n, d) for TSB-AD-M."""

    name: str
    values: np.ndarray  # float64
    labels: np.ndarray  # int, shape (n,), 1 = anomalous

    def __len__(self) -> int:
        return len(self.values)


def load_series(filename: str, data_dir: Path | None = None) -> Series:
    """Load one full TSB-AD-U or TSB-AD-M recording (TSB-AD-M if not found in TSB-AD-U).

    Rows with missing values are dropped, as TSB-AD's own benchmark script does.
    """
    path = (data_dir or DATA_DIR) / filename
    if data_dir is None and not path.exists():
        path = DATA_DIR_M / filename
    frame = pd.read_csv(path).dropna()
    values = frame.drop(columns="Label").to_numpy(dtype=np.float64)
    return Series(
        name=path.stem,
        values=values[:, 0] if values.shape[1] == 1 else values,
        labels=frame["Label"].to_numpy(dtype=np.int64),
    )
