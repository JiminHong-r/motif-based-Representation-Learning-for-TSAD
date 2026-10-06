# Motif-Based Representation Learning for Time Series Anomaly Detection

**Status as of 2026-10-05:** Eval done (frozen configuration, 350 series). VUS-PR **0.333** (failures as 0; 0.350 over the 333 series with motifs) — rank **22 of 42** on the TSB-AD-U leaderboard, between CNN (0.343) and LSTMAD (0.325). Random bank 0.321; motifs beat random on only 164/333 series (median difference 0.000). The Tuning result (0.378, 30/48 vs. random) did not carry over. Next: write-up around the decoupling answer, RQ3 with more backends, NormA/PBAD (§5). Cycle 2 plan (Oct–Feb) in §5.

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

**Frozen configuration (2026-10-01, chosen on Tuning by the §5 Step 2 rule — do not change after Eval):**
LoCoMotif with `rho=0.6` (other settings default), motifs discovered on the training part (`tr_` prefix, at least 4 × min motif length), 8000-point segments, motif length period–2×period; per-motif distance with mean-centring and self-match exclusion; score = distance to the nearest motif; window → point by max over covering windows; min-max. Random-motif floor with seed 2024. `notebooks/05_trained.py --backends LoCoMotif --params rho=0.6 --point max`, representation `train-nearest-max`. Tuning: 0.378 mean (median 0.315), random 0.312, 30/48 wins.

**Eval run started 2026-10-03** at commit `6baf029` (decision: training part, not whole series — it is the setting where motif discovery beats a random bank). Output `results/eval_metrics.csv`, log `results/eval_full.log`. Then `06_leaderboard.py --ours results/eval_metrics.csv`, on all 350 series and on the 326 in `results/eval_period_check.csv`.

**Failure rule (fixed 2026-10-03 before any Eval score was read):** at 263/350, LoCoMotif found no motifs in the training part of 17 series (SMAP 4, MSL 3, SMD 3, WSD 2, IOPS 2, Stock 1, Daphnet 1, LTDB 1). Headline number counts every failure as VUS-PR 0 — TSB-AD's own benchmark script does the same (`list_w = [0]*9` on exception). Also reported: mean over successful series with the failure count. The configuration is not changed because of Eval failures.

## 5. Research plan, October 2026 – February 2027 (cycle 2)

Cycle 1 (Sept) built the pipeline, froze one configuration and ran Eval once: 0.333, rank 22/42, motif bank = random bank. Cycle 2 uses the remaining ~5 months to answer the RQs properly and push "how far can a motif bank go" — with a larger development set so Eval stays clean.

**Framing (Louis, 2026-09-30):** build a representative **bank of normal subsequences** with motif discovery → distances from test subsequences to the bank → aggregate (e.g. max) into an anomaly score. Compare with other ways of building the bank: clustering centroids (KShapeAD, KMeansAD), NormA's weighted hierarchical-clustering model, PBAD's frequent patterns in a discretised series.

**RQs restated in that framing**
- **RQ1 — building the bank:** which motif method and settings; what goes in the bank (all occurrences, one representative per motif, all phases); weights; continuous vs. discretised patterns; learned on the training part vs. whole series.
- **RQ2 — from distances to a score:** normalisation (shape / amplitude / both), aggregation (max / centre / multi-scale), per-motif distance vector + an existing detector (kNN, IForest, OCSVM).
- **RQ3 — does bank quality matter:** random bank on every series, a bank-degradation curve, a spread of motif methods, and the occurrence-based signal.
- **Final goal:** rank of the best motif bank among the 41 published TSB-AD-U detectors (good: top 10, > 0.401).

**Protocol for cycle 2**
- **Development set = Tuning (48) + the 472 TSB-AD-U series outside Tuning and Eval** (YAHOO 224, UCR 152, WSD 86, Stock 10). Skewed towards periodic/shape data — exactly the known weakness — so results are always reported per source as well. A fixed, stratified subsample of ~150 series for fast iteration; full 520 to confirm.
- Every design decision on the dev set by a rule written down before the run. Eval is touched once more, at the end, with the frozen cycle-2 configuration. The cycle-1 Eval result (0.333) is reported as is — the thesis shows both, transparently.

**Work packages**

| weeks | WP | what | answers |
|---|---|---|---|
| 1 | 0 — setup | dev split + subsample; compute (university machine?); meeting with Louis; NormA password email | — |
| 1–3 | 1 — RQ3 on cycle 1 | bank-degradation curve (0/25/50/75/100% random occurrences) on Eval and dev; all working `tsmd` methods (k-Motiflets, SetFinder, STOMP, PanMP, VALMOD, LatentMotif, A-PEPA, MDL-Clust, LoCoMotif) with random banks on dev; occurrence signal on Eval | RQ3 for the cycle-1 design |
| 3–8 | 2 — bank construction | best methods from WP1 × bank content (occurrences / one representative per motif / phase-complete / fixed size of 10 like KMeansAD) × weights (NormA-style) × discretised bank (SAX + frequent patterns, PBAD-style) | RQ1 |
| 8–12 | 3 — scoring | shape + amplitude distances as one per-motif vector; multi-scale windows for point anomalies; aggregation; existing detectors on the vector. Targets the measured weaknesses: periodic shape anomalies (UCR, YAHOO), point anomalies | RQ2 |
| 1–12 | 4 — baselines | NormA (password), PBAD (authors' code or a faithful re-implementation), published KShapeAD/KMeansAD/MatrixProfile | comparison |
| 13–14 | 5 — final | freeze cycle-2 configuration on dev; Eval once; leaderboard; random bank + degradation curve on Eval | final goal, RQ3 |
| 14–20 | 6 — writing | method/setup chapters start in week 2; results chapters after each WP; buffer | thesis |

**Decision points with Louis:** after WP1 (which motif methods are worth building banks from), after WP2 (which bank goes into WP3), before WP5 (freeze).

## 6. Findings so far

- **Phase 1 slice (one 6000-point MITDB ECG window centred on an anomaly):** VUS-PR 0.896 (k-Motiflets), 0.700 (LoCoMotif), 0.406 (random floor). **Not a real result** — the window was hand-picked around a labelled anomaly in highly periodic data; a random floor of 0.406 is close to the best published detector on the real benchmark. The slice code has been removed.
- **Edge-bias bug (fixed):** points near series/segment boundaries are covered by fewer sliding windows, so their min-distance is biased upward. Uncorrected it made the edges the global maximum (Standard-F1 = 0 at AUC-ROC 0.87). Boundary points are now filled with the interior median.
- **Z-normalisation makes the representation shape-only.** On `004_NAB`, the anomaly has 19× the normal signal std, yet scores *lower* than normal data — amplitude/level anomalies with an ordinary shape are invisible. Not a code bug; a property of the method. Predicts which TSB-AD series the approach can work on.
- **Period estimate fails on non-periodic series** (e.g. period 5 on `007_NAB`, floored to a 16-point window), where almost everything matches some motif.
- **Tuning run, LoCoMotif, complete (48/48, 0 failures):** mean VUS-PR **0.260 vs. random floor 0.258** (medians 0.096 vs. 0.093); beats random on exactly **24/48**, median paired difference 0.000. Mean AUC-ROC 0.641 vs. 0.571 — the only metric with a gap. Per source: clear wins on periodic ECG/synthetic (TODS 0.76 vs. 0.44, SVDB 0.33 vs. 0.19, LTDB 0.39 vs. 0.27); clear losses on UCR (0.17 vs. 0.30), OPPORTUNITY (0.14 vs. 0.24), NAB (0.15 vs. 0.17) — consistent with the z-normalisation and period-estimate findings above. First evidence for "decoupled", one backend only; ~115 s per series.
- **Scaling:** k-Motiflets 17.8 s → 37.8 s → 91.9 s → 364 s at n = 2K/4K/8K/16K (superlinear); LoCoMotif 2.7 s → 64 s over the same range.

- **Baselines on the 48 Tuning series** (TSB-AD hyperparameters, our `evaluate`): KShapeAD 0.369, MatrixProfile 0.328, KMeansAD 0.327, Sub-PCA 0.326, SAND 0.152 (crashes on 24/48). NormA and Series2Graph are patent-protected and need a password from the authors.
- **Three design flaws in the first representation** (found by diagnosing where MatrixProfile wins): (1) self-matches — windows on an occurrence matched themselves at distance ~0, so the score mostly said "not inside an occurrence", which random occurrences do too; (2) min over covering windows hid short anomalies; (3) z-normalisation hid amplitude anomalies. Fixes: exclude overlapping library entries, centre/max aggregation, mean-centring instead of z-norm.
- **Ablation, whole series, LoCoMotif:** shape/min 0.260 → shape/centre 0.111 → +excl 0.105 → centred 0.355 → centred/max 0.387. The random floor follows every step (0.399 at the end): in the whole-series setting the gain comes from how windows are compared, not from the motifs.
- **Training part + per-motif features:** nearest 0.323 vs. random 0.295; with max 0.381 vs. 0.323. IsolationForest on the per-motif features is worse (0.29). LoCoMotif finds no motifs in 3 short training parts; `rho=0.6` fixes that (0.378, 48/48, 30/48 wins vs. random).
- **Hyperparameters:** k-Motiflets `k_max=20` worse (0.281 vs. 0.299). k-Motiflets returns a single motif per series, and on some series near-duplicate occurrences (starts 3420…3428 on SMD_16) — its library can be 2–3 distinct windows.
- **Max aggregation** (Louis) helps in every setting: +0.03 to +0.06.
- **Occurrence-based "uncovered":** weak alone (0.14) but beats random on 39/48 — anomalies fall outside motif occurrences, which random positions cannot reproduce. Fused with the distance score it beats random 31/48 but lowers the mean (0.30).
- **Per anomaly type (Tuning):** non-periodic series (period < 16, mostly server/web metrics with spikes and level shifts: SMD, WSD, MSL) — motifs 0.58–0.61 vs. KShapeAD 0.10, MatrixProfile 0.18, Sub-PCA 0.45. Periodic series (ECG, synthetic: UCR, YAHOO, LTDB, TODS) — motifs 0.30–0.32 vs. KShapeAD 0.46. Point anomalies (length ≤ 3): KShapeAD 0.51 vs. motifs 0.35–0.39.

- **Last round (§5 Step 1), both negative, as decided in advance:**
  - *Periodic weakness:* not flat windows (they explain the top scores on 2/29 periodic series). Cause: a small bank holds each motif at one phase, so normal windows at other phases look far and the score oscillates with the period; resampling occurrences of different lengths (13–79 points on YAHOO_93) to one length also distorts their frequency. Min-aggregation hid this, which is why shape/min worked on periodic series. A phase-complete, unresampled bank did not help the frozen configuration (0.418 vs. 0.430 on the 12 short periodic series). Recorded as a limitation.
  - *Weighted bank (NormA-style weighted average, weights = occurrence share):* 0.372 vs. 0.378 for nearest, better on only 12/48.
- **Our VUS-PR vs. TSB-AD's published numbers (10 Eval series):** MatrixProfile identical on 8/10, Sub-PCA within 0.003. The other 2 (151_Stock_3, 277_NEK_1) reproduce exactly when the window is 125: in TSB-AD's run `find_length_rank` fell back to 125 there, in our environment it returns 5 and 4 (likely a statsmodels version difference; TSB-AD does not pin it). MatrixProfile on all 350 Eval series: identical on **326/350** (93%), mean 0.345 vs. published 0.350. The 24 mismatches are period-estimate disagreements (our period 4–5 on 20, the 125 fallback on 2). List in `results/eval_period_check.csv`; the leaderboard is reported on all 350 and, as a robustness check, on the 326 consistent series.

- **Eval (350 series, frozen configuration, run once):**
  - Mean VUS-PR 0.333 with the 17 no-motif series as 0 (0.350 over the 333 others); rank 22/42 — below MatrixProfile 0.350, KMeansAD 0.366, KShapeAD 0.401, Sub-PCA 0.423. Same picture on the 326 period-consistent series (0.321, rank 22). Plots: `results/leaderboard.png`, `results/leaderboard_consistent.png`.
  - Random bank 0.321 (0.338 over 333). Motifs beat random on 164/333, median paired difference 0.000. **On Eval, the quality of the motif bank does not matter — the decoupled answer to the central question.** The Tuning gap (0.378 vs. 0.312) was optimistic, as expected after ~30 variants.
  - Per series vs. published: wins against KShapeAD 157/350, KMeansAD 168, MatrixProfile 165, Sub-PCA 161 — roughly half each: complementary, not dominated.
  - Non-periodic series (68): 0.547 vs. KShapeAD 0.24, KMeansAD 0.26, MatrixProfile 0.33 (Sub-PCA 0.64). Strong on server/web metrics: SMD 0.58 (KShapeAD 0.13), WSD 0.36 (0.10), IOPS 0.25 (0.09), Exathlon 0.80 (0.33). Random equally strong there.
  - Periodic series (282): 0.282 vs. KShapeAD 0.44. Weak on UCR (70 series, 0.11 vs. 0.38), YAHOO (0.14 vs. 0.55), OPPORTUNITY, MGAB, SED — the phase limitation found on Tuning.
  - Point anomalies (28): 0.097 vs. KShapeAD 0.51 — max aggregation over a motif-length window spreads a single-point spike over the whole window.
  - LoCoMotif found no motifs in the training part of 17/350 series (SMAP 4, MSL 3, SMD 3, WSD 2, IOPS 2, Stock, Daphnet, LTDB).

## 7. Phase plan

| Phase | Focus | Status |
|---|---|---|
| 0 — Framing | Benchmark framing, univariate scope | Done |
| 1 — Thin vertical slice | k-Motiflets + LoCoMotif, one representation, VUS-PR | Done, on the full Tuning split |
| 2 — Representation layer (RQ1) | distance (shape/centred/raw), per-motif features, occurrence-based | Done on Tuning |
| 3 — Scoring strategies (RQ2) | nearest vs. IsolationForest; min/centre/max aggregation | Done on Tuning |
| 3b — Last round | periodic weakness, weighted bank | Done, both negative; configuration frozen |
| 4 — Eval + baselines | frozen config on 350 Eval series; NormA, PBAD | Eval done (0.333, rank 22/42); NormA, PBAD pending |
| 4b — Propagation study (RQ3) | more `tsmd` backends under the frozen config | After Eval |
| 5 — Write-up | method + setup now, results after Eval | Started |
| Cycle 2 | dev set of 520 series; RQ1–RQ3 work packages; final Eval | §5, Oct 2026 – Feb 2027 |

## 8. Open decisions to raise with the supervisor

- **Periodic vs. non-periodic.** Motifs win clearly on non-periodic series and lose on periodic ones (§6). Accept as a stated boundary, or worth investigating further (§5 Step 1)?
- **Training part vs. whole series.** Whole series scores slightly higher (0.387) but random banks match it; training part (0.378) is where motifs beat random. Proposal: training part, because it is the setting where the motif step actually contributes.
- **Compute for Eval.** 350 series of motif discovery takes days on a laptop — is a university machine available?
- **PBAD code** — does Louis know of a maintained implementation?
- Settled: RQ3 quality is inferred from downstream VUS-PR only; changing hyperparameters of existing methods is allowed; the `tr_` prefix may be used as training data.

## 9. Code layout

```
notebooks/01_tuning_set.py      first pipeline, whole series → results/tuning_metrics.csv
notebooks/02_baselines.py       TSB-AD baselines → results/baseline_metrics.csv
notebooks/03_compare.py         tables for 01 + 02
notebooks/04_representations.py whole-series ablation over cached motifs → results/representation_metrics.csv
notebooks/05_trained.py         motifs from the training part, per-motif features, --params, --point → results/trained_metrics.csv
notebooks/06_leaderboard.py     published TSB-AD-U Eval VUS-PR, ours highlighted → results/leaderboard.png
src/data.py                     load a TSB-AD-U CSV (values + labels)
src/pipeline.py                 window rule, segments, cached motif discovery, random-motif floor
src/representation.py           reference bank, distances (norm / aggregation / self-exclusion), per-motif features, occurrence score
src/scoring.py                  min-max normalisation, TSB-AD get_metrics wrapper
external/TSMD, external/TSB-AD  pinned submodules (not our code)
data/, results/                 gitignored
```

## 10. Key references

- TSAD foundations: Schmidl et al. (VLDB 2022); Liu & Paparrizos, *TSB-AD* (NeurIPS 2024); Boniol et al., decade review (2024)
- TSMD backends: Motiflets (Schäfer & Leser, VLDB 2022); LoCoMotif (Van Wesenbeeck et al., DMKD 2024); LAMA/leitmotifs (Schäfer & Leser, VLDB 2025); Guerrini et al., *TSMD: A Comprehensive Evaluation* (VLDB 2025) — source of the `tsmd` package
- Normal-pattern-library detectors (baselines): NORMA, SAND, Series2Graph, KShapeAD — see TSB-AD's model list for the original papers
- Reference-bank baselines from Louis (2026-09-30): NormA (Boniol et al., VLDB J 2021, doi:10.1007/s00778-021-00655-8); PBAD (Feremans et al., ECML PKDD 2019, doi:10.1007/978-3-030-46150-8_15)
- Pending: SubTSMD (Carpentier et al., DMKD 2026) — no public repo yet, KU Leuven contact in progress.
