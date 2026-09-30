"""Leaderboard: mean VUS-PR of every published TSB-AD-U detector, ours highlighted.

Published per-series VUS-PR exists for the Eval split only (32 detectors from
the TSB-AD paper plus newer leaderboard entries), so our methods are placed on
it only from an Eval run. Pass that run's CSV(s) with --ours; rows need
file, backend, VUS-PR and optionally representation. Methods missing any Eval
series are listed with their coverage rather than silently averaged.

    .venv/bin/python notebooks/06_leaderboard.py [--ours results/eval_metrics.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "external" / "TSB-AD" / "benchmark_exp"
EVAL_LIST = ROOT / "external" / "TSB-AD" / "Datasets" / "File_List" / "TSB-AD-U-Eva.csv"
OUT_PNG = ROOT / "results" / "leaderboard.png"
OURS, OTHERS = "#2a78d6", "#c3c2bd"


def published() -> pd.DataFrame:
    """file x method VUS-PR over the Eval split, from TSB-AD's published tables."""
    table = pd.read_csv(BENCH / "benchmark_eval_results" / "uni_mergedTable_VUS-PR.csv").set_index("file")
    meta = ["ts_len", "anomaly_len", "num_anomaly", "avg_anomaly_len", "anomaly_ratio", "point_anomaly", "seq_anomaly"]
    table = table.drop(columns=meta)
    for path in sorted((BENCH / "leaderboard_results").glob("Uni_*.csv")):
        name = path.stem.removeprefix("Uni_")
        frame = pd.read_csv(path)
        key = "file" if "file" in frame else "filename"
        # one column per method: named after the file, "VUS-PR", or variants like "TSPulse (ZS)"
        columns = {name: name} if name in frame else {name: "VUS-PR"} if "VUS-PR" in frame else {
            c: c for c in frame.columns if c.startswith(name)}
        for method, column in columns.items():
            if method not in table:
                table[method] = frame.groupby(key)[column].mean()
    return table


def ours(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[frame.get("status", "ok") == "ok"]
        rep = frame["representation"] if "representation" in frame else ""
        frame = frame.assign(method=frame["backend"] + (" [" + rep + "]" if "representation" in frame else ""))
        frames.append(frame.pivot_table(index="file", columns="method", values="VUS-PR"))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", nargs="*", default=[])
    args = parser.parse_args()

    files = pd.read_csv(EVAL_LIST)["file_name"]
    table = published().reindex(files)
    mine = ours(args.ours).reindex(files)
    full = pd.concat([table, mine], axis=1)
    summary = pd.DataFrame({"mean VUS-PR": full.mean(), "series": full.count()}).sort_values("mean VUS-PR")
    summary["ours"] = summary.index.isin(mine.columns)
    summary["rank"] = summary["mean VUS-PR"].rank(ascending=False, method="min").astype(int)
    print(summary.iloc[::-1].round(3).to_string())
    partial = summary[summary["series"] < len(files)]
    if len(partial):
        print(f"\nnot on all {len(files)} Eval series: {', '.join(partial.index)}")

    fig, ax = plt.subplots(figsize=(7, 0.22 * len(summary) + 1))
    colors = [OURS if o else OTHERS for o in summary["ours"]]
    ax.barh(summary.index, summary["mean VUS-PR"], color=colors, height=0.7)
    for label in ax.get_yticklabels():
        if summary.loc[label.get_text(), "ours"]:
            label.set_fontweight("bold")
    ax.set_xlabel(f"mean VUS-PR over TSB-AD-U Eval ({len(files)} series)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
