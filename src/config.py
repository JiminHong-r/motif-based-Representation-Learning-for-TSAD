"""One experiment = one TOML file in configs/. Every setting of the pipeline is here.

    name = "U-LoCoMotif"            # results go to results/runs/<name>/
    [data]
    dataset = "U"                   # "U" (TSB-AD-U) or "M" (TSB-AD-M)
    [bank]                          # step 1: build the bank of normal subsequences
    backend = "LoCoMotif"           # src/backends.py: factories() for U, multivariate_factories() for M
    params = { rho = 0.6 }          # backend hyperparameters (overrides of its defaults)
    source = "train"                # "train": the tr_ prefix (at least 4 x min length) | "whole": entire series
    random_fractions = [1.0]        # random-bank controls: share of occurrences moved (1.0 = random bank)
    segment_len = 8000              # discovery runs on segments of this length
    wlen_floor = 16                 # motif length = max(floor, period) .. factor x that
    wlen_factor = 2.0
    split_long = false              # cut merged runs longer than the max motif length into motif-length pieces
    shift = 0.0                     # add each occurrence shifted by up to ±shift x its length (all phases in the bank)
    [distance]                      # step 2: compare windows with the bank
    norm = "centred"                # "shape" (z-norm) | "centred" (mean removed) | "raw"
    [score]                         # step 3: distances -> anomaly score
    scorer = "nearest"              # "nearest" | "weighted" (NormA-style) | "iforest"
    point = "max"                   # window -> point: "max" | "mean" | "centre" | "min"
    [eval]
    failure_value = 0.0             # VUS-PR of a series that cannot be scored (TSB-AD convention)

Fields left out take the defaults below (the cycle-1 frozen configuration).
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    name: str
    dataset: str = "U"
    backend: str = "LoCoMotif"
    params: dict = field(default_factory=dict)
    source: str = "train"
    random_fractions: list = field(default_factory=lambda: [1.0])
    segment_len: int = 8000
    wlen_floor: int = 16
    wlen_factor: float = 2.0
    split_long: bool = False
    shift: float = 0.0
    norm: str = "centred"
    scorer: str = "nearest"
    point: str = "max"
    failure_value: float = 0.0
    seed: int = 2024

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = tomllib.loads(Path(path).read_text())
        flat = {k: v for k, v in raw.items() if not isinstance(v, dict)}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
        unknown = set(flat) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
        return cls(**flat)

    def bank_label(self) -> str:
        """Backend + every discovery setting that changes the bank: names its motif cache."""
        extra = dict(self.params)
        if self.segment_len != 8000:
            extra["segment_len"] = self.segment_len
        if (self.wlen_floor, self.wlen_factor) != (16, 2.0):
            extra["wlen"] = f"{self.wlen_floor}x{self.wlen_factor:g}"
        return self.backend + (f"[{','.join(f'{k}={v}' for k, v in extra.items())}]" if extra else "")

    def as_dict(self) -> dict:
        return asdict(self)
