# Motif-Based Representation Learning for Time Series Anomaly Detection

**Status as of 2026-09-24:** Phase 1 done. LoCoMotif + random floor finished on all 48 Tuning series: **no better than random** (VUS-PR 0.260 vs. 0.258, 24/48 wins). k-Motiflets run and baseline detectors on the same 48 series in progress (§5).

## 1. Goal

Build a **decoupled representation layer** that turns the output of any motif-discovery (TSMD) method into an anomaly-detection-ready representation, and use it as an instrument to answer one central empirical question:

> **Does motif discovery quality actually matter for anomaly detection performance — or is it decoupled from it?**

This is a **benchmark-style contribution**, not a new detector or a new motif-discovery algorithm. The representation layer and scoring strategies (RQ1/RQ2) exist to make RQ3 measurable across many backends under identical conditions.

A negative result is a valid answer: if randomly placed motifs perform as well as discovered ones, that is evidence for "decoupled".

## 2. Research questions

- **RQ1** — How should a time series be encoded into a motif-based representation? (distance-to-nearest-motif, occurrence-based, reconstruction-error-based)
- **RQ2** — How should that representation be turned into an anomaly score? (rule-based threshold vs. a lightweight detector like kNN/IsolationForest on top of the representation)
- **RQ3** — How does motif discovery *quality* relate to downstream anomaly detection *performance*, across backends of differing sophistication?

Advisor-confirmed: these RQs are a guide, not a fixed contract. Pivoting within "TSAD + TSMD" is fine, and RQs get refined together throughout.

## 3. Method stack

| Role | Tool | Notes |
|---|---|---|
| Motif discovery backends (primary) | `tsmd` (git submodule `external/TSMD`, installed editable) | 11 methods behind one common interface (`.fit(signal)` → `.prediction_mask_`, shape `(n_motifs, n_samples)`): SetFinder, LatentMotif, STOMP, VALMOD, PanMP, k-Motiflets, PEPA, A-PEPA, GrammarViz, MDL-Clust, LoCoMotif. Interface verified on every backend. **GrammarViz does not work out of the box** — it needs Java plus a jar from a nested repo upstream never registered, so effectively 10 of 11. `Motiflets_numba` is the same algorithm ~1.7× faster. |
| Motif discovery backend (extra) | `leitmotif` (LAMA), from PyPI | Distribution name `leitmotif`, **import name `leitmotifs`**. Not in `tsmd`; needs its own small adapter. |
| Dataset + anomaly ground truth | TSB-AD (git submodule `external/TSB-AD`, installed editable; data in `data/TSB-AD-U/`, from thedatum.org) | Submodule rather than PyPI because `Datasets/File_List/` (the Tuning/Eval split) is not in the wheel. **Do not use `tsmd`'s own bundled datasets** — they have motif ground truth but no anomaly labels. |
| Window length | TSB-AD's `find_length_rank(rank=1)` | Period estimate per series. Sets both the motif window and the metric's `slidingWindow`, as in TSB-AD's own benchmark script. |
| Downstream evaluation metric | TSB-AD's `get_metrics()` | VUS-PR is the headline metric. |
| Long-lead / pending | SubTSMD | No public repo as of now. Email sent to KU Leuven authors — optional bonus backend, not a blocker. |

**Environment:** Python **3.12** is required — TSB-AD pins `numpy<2.0`, and numpy 1.26.4 has no Python 3.13 wheels. `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

**Decision:** PROM / TSMD-Bench is dropped entirely — not used anywhere in this project, including as a validation step. Motif quality is measured only indirectly, through its effect on downstream TSB-AD performance.

## 4. Evaluation protocol (fixed — do not change after seeing results)

Supervisor guidance: realistic VUS-PR is around 0.4–0.5; **don't overfit, don't over-optimise.**

- **Tuning split (48 series, `TSB-AD-U-Tuning.csv`)** — all design decisions are made here.
- **Eval split (350 series, `TSB-AD-U-Eva.csv`)** — run **once** at the end with everything frozen. Never revise a design after seeing Eval results.
- **Whole series are scored**, as in TSB-AD's benchmark script. No hand-picked slices — they are not comparable to published numbers.
- Hyperparameters are **fixed a priori and identical for every series and backend**: motif length = period to 2×period (period from `find_length_rank`, floored at 16); segments of 8000 points with overlap, so long series are fully scored.
- A **random-motif floor** (same number and lengths of occurrences, random positions) is computed on **every** series. VUS-PR's chance level is not 0 and varies per series, so results are only read relative to the floor on the same series.
- Reference numbers (TSB-AD paper, 32 detectors, Eva split): best **Sub-PCA 0.423**, MatrixProfile 0.350, IForest 0.298. Newer leaderboard entries reach 0.586 (TimeRCD_MAFT). Published **per-series** VUS-PR for all 32 detectors exists for Eva only (`benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv`), so the final Eval comparison does not require rerunning them — but the Tuning split does.

## 5. What to do now — immediate next actions (in order)

1. ~~**Finish and read the Tuning run.**~~ Done 2026-09-22 — see §6.
2. **Run baseline detectors on the same 48 series**, with TSB-AD's own hyperparameters (`HP_list.py`) — no tuning on our side:
   - *Ablation:* **MatrixProfile** — the same distance-to-nearest-subsequence idea without motif selection. If motifs don't beat it, motif selection adds nothing.
   - *Direct competitors* — detectors that also build a library of normal patterns, differing only in how: **KShapeAD** (#2 on Eva), **Series2Graph** (#4), **KMeansAD**, **SAND**, **NORMA** (closest prior work; not in the published table).
   - *Ceiling:* **Sub-PCA** (#1 of the 32 on Eva).
   - Deep / foundation models are out of scope — cite their published numbers.
   - Measure the runtime of Series2Graph and NORMA on the 650K-point series before launching all 48.
3. **Queue k-Motiflets** (`Motiflets_numba`) over the Tuning split — ~7 h, run overnight.
4. **Supervisor meeting** with the tables from 1–2 and the decisions in §7.

## 6. Findings so far

- **Phase 1 slice (one 6000-point MITDB ECG window centred on an anomaly):** VUS-PR 0.896 (k-Motiflets), 0.700 (LoCoMotif), 0.406 (random floor). **Not a real result** — the window was hand-picked around a labelled anomaly in highly periodic data; a random floor of 0.406 is close to the best published detector on the real benchmark. The slice code has been removed.
- **Edge-bias bug (fixed):** points near series/segment boundaries are covered by fewer sliding windows, so their min-distance is biased upward. Uncorrected it made the edges the global maximum (Standard-F1 = 0 at AUC-ROC 0.87). Boundary points are now filled with the interior median.
- **Z-normalisation makes the representation shape-only.** On `004_NAB`, the anomaly has 19× the normal signal std, yet scores *lower* than normal data — amplitude/level anomalies with an ordinary shape are invisible. Not a code bug; a property of the method. Predicts which TSB-AD series the approach can work on.
- **Period estimate fails on non-periodic series** (e.g. period 5 on `007_NAB`, floored to a 16-point window), where almost everything matches some motif.
- **Tuning run, LoCoMotif, complete (48/48, 0 failures):** mean VUS-PR **0.260 vs. random floor 0.258** (medians 0.096 vs. 0.093); beats random on exactly **24/48**, median paired difference 0.000. Mean AUC-ROC 0.641 vs. 0.571 — the only metric with a gap. Per source: clear wins on periodic ECG/synthetic (TODS 0.76 vs. 0.44, SVDB 0.33 vs. 0.19, LTDB 0.39 vs. 0.27); clear losses on UCR (0.17 vs. 0.30), OPPORTUNITY (0.14 vs. 0.24), NAB (0.15 vs. 0.17) — consistent with the z-normalisation and period-estimate findings above. First evidence for "decoupled", one backend only; ~115 s per series.
- **Scaling:** k-Motiflets 17.8 s → 37.8 s → 91.9 s → 364 s at n = 2K/4K/8K/16K (superlinear); LoCoMotif 2.7 s → 64 s over the same range.

## 7. Phase plan

| Phase | Focus | Status / notes | Est. duration |
|---|---|---|---|
| 0 — Framing | Lock in benchmark framing, univariate scope, toy dataset | Done | 1 week |
| 1 — Thin vertical slice | k-Motiflets + LoCoMotif via `tsmd`, one representation, one scorer, VUS-PR sanity check | **Done.** Extended to the full Tuning split. | 2–3 weeks |
| 2 — Representation layer (RQ1) | 3 representation variants (distance/occurrence/reconstruction-error) | Next after baselines. Start with an **amplitude-aware** variant (motivated by the z-normalisation finding), and consider scoring per motif rather than pooling all occurrences into one library. `tsmd` supplies the backend interface; LAMA needs one adapter. | 2–3 weeks |
| 3 — Scoring strategies (RQ2) | Rule-based vs. lightweight detector (kNN/IForest on the representation) | Run on the 48-series Tuning split (was: 5–10 series). | 3–4 weeks |
| 4 — Propagation study (RQ3) | Quality vs. performance across backends | Use the `tsmd` methods' spread of sophistication (cheap first: SetFinder, STOMP, PanMP) plus the random-motif floor on every series, and compare against MatrixProfile (no motif selection) and the normal-pattern-library detectors. Quality is inferred entirely from downstream VUS-PR. | 4 weeks |
| 5 — Final evaluation + write-up | Eval split once, frozen; full draft | Compare against published per-series Eval numbers. | 3–4 weeks |

## 8. Open decisions to raise with the supervisor

- **RQ3 design (settled)**: motif discovery quality is treated as a latent variable, operationalized only through its effect on downstream VUS-PR — no PROM/TSMD-Bench validation step. Quality varies via (a) the natural spread across `tsmd`'s methods, and (b) a random-motif baseline as the floor.
- **Framing consequence**: since quality is never measured directly, the thesis's claims about "quality" are strictly claims about *downstream-performance-inferred* quality. Worth stating explicitly in the write-up as a scoping choice, not an oversight.
- **Scope to periodic series?** If motifs only help on periodic data, that is a legitimate, stated boundary.
- **Unsupervised vs. semi-supervised**: learn motifs from the anomaly-free `tr_` prefix only (TSB-AD's semi-supervised category)? Cleaner conceptually and much cheaper.
- **Positioning against NORMA** and the other normal-pattern-library detectors (KShapeAD, KMeansAD, Series2Graph, SAND): the thesis's angle is "motif discovery as a way of building the normal library".
- **Whether to include LAMA and/or SubTSMD** as extra backends beyond `tsmd`, given the adapter work each requires.

## 9. Code layout

```
notebooks/01_tuning_set.py   run the pipeline over the 48 Tuning series → results/tuning_metrics.csv
src/data.py                  load a TSB-AD-U CSV (values + labels)
src/pipeline.py              window rule, overlapping segments, per-segment motif fit, random-motif floor
src/representation.py        mask → library of z-normalised occurrences → per-point distance to nearest
src/scoring.py               min-max normalisation, TSB-AD get_metrics wrapper
external/TSMD, external/TSB-AD   pinned submodules (not our code)
data/                        TSB-AD-U (gitignored)
```

## 10. Key references

- TSAD foundations: Schmidl et al. (VLDB 2022); Liu & Paparrizos, *TSB-AD* (NeurIPS 2024); Boniol et al., decade review (2024)
- TSMD backends: Motiflets (Schäfer & Leser, VLDB 2022); LoCoMotif (Van Wesenbeeck et al., DMKD 2024); LAMA/leitmotifs (Schäfer & Leser, VLDB 2025); Guerrini et al., *TSMD: A Comprehensive Evaluation* (VLDB 2025) — source of the `tsmd` package
- Normal-pattern-library detectors (baselines): NORMA, SAND, Series2Graph, KShapeAD — see TSB-AD's model list for the original papers
- Pending: SubTSMD (Carpentier et al., DMKD 2026) — no public repo yet, KU Leuven contact in progress.
