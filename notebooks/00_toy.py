"""Toy dataset: the whole pipeline on one small synthetic series, for slides and sanity checks.

A periodic signal (two harmonics + noise, period 50) with a clean training part
and five planted anomalies of different types. Runs the same steps as run.py —
period → motif bank on the training part → per-motif distances → score →
VUS-PR, plus the random bank — and plots signal, bank and scores.

    .venv/bin/python notebooks/00_toy.py [config.toml]   # default: the frozen U-STOMP-mean → results/toy.png

Runs in seconds on a laptop; no data download needed.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "notebooks")]

from backends import factories  # noqa: E402
from config import Config  # noqa: E402
from pipeline import discover, window_bounds  # noqa: E402
from run import period_of, variant_score  # noqa: E402
from scoring import evaluate  # noqa: E402

N, PERIOD, TRAIN = 4000, 50, 1500


def toy_series(seed: int = 0):
    """Signal, labels, and (name, start, end) of each planted anomaly."""
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    x = np.sin(2 * np.pi * t / PERIOD) + 0.4 * np.sin(4 * np.pi * t / PERIOD) + 0.05 * rng.standard_normal(N)
    anomalies = [("point spike", 1800, 1800), ("amplitude x2.5", 2100, 2199), ("shape (flat)", 2500, 2549),
                 ("frequency x2", 2900, 2999), ("level shift", 3400, 3449)]
    x[1800] += 3.0
    x[2100:2200] *= 2.5
    x[2500:2550] = x[2500]
    x[2900:3000] = np.sin(4 * np.pi * t[2900:3000] / PERIOD) + 0.4 * np.sin(8 * np.pi * t[2900:3000] / PERIOD)
    x[3400:3450] += 1.5
    labels = np.zeros(N, int)
    for _, s, e in anomalies:
        labels[s:e + 1] = 1
    return x, labels, anomalies


def main():
    cfg = Config.load(sys.argv[1]) if len(sys.argv) > 1 else Config.load(ROOT / "configs" / "U-STOMP-mean.toml")
    x, labels, anomalies = toy_series()
    period = period_of(x)
    bounds = window_bounds(period, cfg.wlen_floor, cfg.wlen_factor)
    n_train = N if cfg.source == "whole" else TRAIN
    make_model = functools.partial(factories()[cfg.backend], **cfg.params)
    segments = discover(x[:n_train], make_model, period, cfg.segment_len, bounds)

    score, occ = variant_score(cfg, x, segments, n_train, bounds)
    rand, rocc = variant_score(cfg, x, segments, n_train, bounds, fraction=1.0)
    vus = {k: evaluate(s, labels, period)["VUS-PR"] for k, s in [("motif", score), ("random", rand)]}
    print(f"period {period}, {len({o[0] for o in occ})} motifs, {len(occ)} occurrences; "
          f"VUS-PR motif {vus['motif']:.3f}, random bank {vus['random']:.3f}")
    for name, s, e in anomalies:  # how far above normal each anomaly scores (normal = training part)
        print(f"  {name:15s} max score {score[s:e + 1].max():.2f}  (training-part 99th pct {np.percentile(score[:TRAIN], 99):.2f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(x, lw=0.6, color="k")
    ax[0].axvspan(0, TRAIN, color="tab:green", alpha=0.08, label="training part (bank learned here)")
    for name, s, e in anomalies:
        ax[0].axvspan(s - 3, e + 3, color="tab:red", alpha=0.25)
        ax[0].text((s + e) / 2, x.max() + 0.2, name, ha="center", fontsize=8)
    colours = plt.cm.tab10(np.arange(10))
    for m, s, e, *_ in occ:
        ax[0].plot([s, e], [x.min() - 0.3] * 2, color=colours[m % 10], lw=3)
    ax[0].set_title(f"Toy series: motif occurrences (coloured bars, one colour per motif) and planted anomalies (red)")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].plot(score, color="tab:blue", lw=0.8)
    ax[1].set_title(f"Anomaly score, motif bank ({cfg.backend}) — VUS-PR {vus['motif']:.3f}")
    ax[2].plot(rand, color="tab:gray", lw=0.8)
    ax[2].set_title(f"Anomaly score, random bank (same occurrences, random positions) — VUS-PR {vus['random']:.3f}")
    for a in ax[1:]:
        for _, s, e in anomalies:
            a.axvspan(s - 3, e + 3, color="tab:red", alpha=0.15)
    fig.tight_layout()
    out = ROOT / "results" / "toy.png"
    fig.savefig(out, dpi=130)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
