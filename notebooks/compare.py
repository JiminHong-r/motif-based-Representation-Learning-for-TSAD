"""Compare every motif-bank config with the baselines on one dataset + split; table + plot.

Collects results/runs/*/<split>.csv for configs of the chosen dataset (motif
bank and its random bank), our baseline runs on the same split, and — on Eval —
TSB-AD's published per-series VUS-PR for every detector. Means are over all
series of the split; a series a method could not score counts as its
config's failure_value (0, TSB-AD's convention). The plot has one bar per
method, VUS-PR on the y-axis; motif banks in blue, their random banks light
blue, baselines grey.

    .venv/bin/python notebooks/compare.py --dataset U --split dev_small
    .venv/bin/python notebooks/compare.py --dataset M --split tuning
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src"))
run = importlib.import_module("run")
leaderboard = importlib.import_module("06_leaderboard")
MOTIF, RANDOM, OTHER = "#2a78d6", "#9cc3ee", "#c3c2bd"


def motif_runs(dataset: str, split: str, files: pd.Index) -> tuple[pd.DataFrame, dict]:
    """file x method VUS-PR for every config of `dataset` that has a run on `split`."""
    table, kind = {}, {}
    for cfg_path in sorted((ROOT / "results" / "runs").glob("*/config.resolved.json")):
        cfg = json.loads(cfg_path.read_text())
        csv = cfg_path.parent / f"{split}.csv"
        if cfg["dataset"] != dataset or not csv.exists():
            continue
        rows = pd.read_csv(csv)
        rows["VUS-PR"] = rows["VUS-PR"].where(rows["status"] == "ok", cfg["failure_value"])
        for variant, group in rows.groupby("variant"):
            share = None if variant == "motif" else float(variant.removeprefix("random"))
            name = cfg["name"] if share is None else f"{cfg['name']} ({'random bank' if share == 1 else f'{share:.0%} random'})"
            table[name] = group.set_index("file")["VUS-PR"]
            kind[name] = "motif" if variant == "motif" else "random"
    return pd.DataFrame(table).reindex(files), kind


def baseline_runs(dataset: str, split: str, files: pd.Index) -> pd.DataFrame:
    name = {("U", "tuning"): "baseline_metrics.csv", ("U", "eval"): "baseline_eval_metrics.csv"}.get(
        (dataset, split), f"baseline_{dataset}_{split}_metrics.csv")
    path = ROOT / "results" / name
    if not path.exists():
        return pd.DataFrame(index=files)
    rows = pd.read_csv(path)
    rows["VUS-PR"] = rows["VUS-PR"].where(rows["status"] == "ok", 0.0)
    return rows.pivot_table(index="file", columns="backend", values="VUS-PR").reindex(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="U", choices=["U", "M"])
    parser.add_argument("--split", default="dev_small")
    args = parser.parse_args()

    files = pd.Index(pd.read_csv(run.SPLITS[(args.dataset, args.split)])["file_name"])
    motifs, kind = motif_runs(args.dataset, args.split, files)
    others = leaderboard.published(args.dataset).reindex(files) if args.split == "eval" else baseline_runs(args.dataset, args.split, files)
    others = others[[c for c in others.columns if c not in motifs.columns]]
    table = pd.concat([motifs, others], axis=1)

    summary = pd.DataFrame({"mean VUS-PR": table.mean(), "median": table.median(), "series": table.count()})
    summary["kind"] = [kind.get(m, "baseline") for m in summary.index]
    for m in motifs.columns:
        if kind[m] == "motif" and f"{m} (random bank)" in motifs:
            diff = (motifs[m] - motifs[f"{m} (random bank)"]).dropna()
            summary.loc[m, "beats random bank"] = f"{(diff > 0).sum()}/{len(diff)}"
    summary = summary.sort_values("mean VUS-PR", ascending=False)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    incomplete = summary.index[summary["series"] < len(files)]
    print(f"{args.dataset} / {args.split}: {len(files)} series\n")
    print(summary.round(3).fillna("").to_string())
    if len(incomplete):
        print(f"\nnot (yet) on all {len(files)} series: {', '.join(incomplete)}")

    fig, ax = plt.subplots(figsize=(max(6, 0.38 * len(summary) + 2), 5))
    colors = summary["kind"].map({"motif": MOTIF, "random": RANDOM, "baseline": OTHER})
    ax.bar(range(len(summary)), summary["mean VUS-PR"], color=colors, width=0.75)
    ax.set_xticks(range(len(summary)), summary.index, rotation=70, ha="right", fontsize=8)
    for tick, k in zip(ax.get_xticklabels(), summary["kind"]):
        tick.set_fontweight("bold" if k == "motif" else "normal")
    ax.set_ylabel("mean VUS-PR")
    ax.set_title(f"TSB-AD-{args.dataset}, {args.split} ({len(files)} series)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MOTIF, RANDOM, OTHER)]
    ax.legend(handles, ["motif bank", "random bank (control)", "baseline / published"], fontsize=8, frameon=False)
    fig.tight_layout()
    out = ROOT / "results" / f"compare_{args.dataset}_{args.split}.png"
    fig.savefig(out, dpi=150)
    summary.to_csv(out.with_suffix(".csv"))
    print(f"\nsaved {out.relative_to(ROOT)} and .csv")


if __name__ == "__main__":
    main()
