"""Supervisor tables: motif backends vs. random floor vs. TSB-AD baselines.

Reads results/tuning_metrics.csv and results/baseline_metrics.csv, prints
tables. Means are over series where *every* method succeeded, so
rows are comparable; per-method failure counts are shown alongside.

    .venv/bin/python notebooks/03_compare.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
frames = [pd.read_csv(ROOT / "results" / f) for f in ("tuning_metrics.csv", "baseline_metrics.csv")]
runs = pd.concat(frames, ignore_index=True)
ok = runs[runs["status"] == "ok"]

vus = ok.pivot_table(index="file", columns="backend", values="VUS-PR")
common = vus.dropna()
motif = [b for b in vus.columns if "+random" not in b and f"{b}+random" in vus.columns]

print(f"## Overall ({len(common)} series where every method succeeded)\n")
table = pd.DataFrame({
    "ok / total": runs.groupby("backend")["status"].apply(lambda s: f"{(s == 'ok').sum()}/{len(s)}"),
    "mean VUS-PR": common.mean(),
    "median VUS-PR": common.median(),
    "mean AUC-ROC": ok[ok["file"].isin(common.index)].groupby("backend")["AUC-ROC"].mean(),
    "mean seconds": ok.groupby("backend")["seconds"].mean(),
}).sort_values("mean VUS-PR", ascending=False)
print(table.round(3).to_string(), "\n")

print("## Paired wins (VUS-PR, same series)\n")
rows = []
for b in motif:
    for ref in [f"{b}+random", "MatrixProfile"]:
        if ref in common:
            diff = common[b] - common[ref]
            rows.append({"method": b, "vs": ref, "wins": f"{(diff > 0).sum()}/{len(diff)}",
                         "mean diff": diff.mean(), "median diff": diff.median()})
print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

print("## Mean VUS-PR per data source\n")
by_src = common.groupby(common.index.str.split("_").str[1])
print(pd.concat([by_src.size().rename("n"), by_src.mean()], axis=1).round(3).to_string())
