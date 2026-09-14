# Motif-Based Representation Learning for Time Series Anomaly Detection

**Status as of:** thin vertical slice not yet started — this plan reflects the `tsmd` package discovery, which compresses Phase 2.

## 1. Goal

Build a **decoupled representation layer** that turns the output of any motif-discovery (TSMD) method into an anomaly-detection-ready representation, and use it as an instrument to answer one central empirical question:

> **Does motif discovery quality actually matter for anomaly detection performance — or is it decoupled from it?**

This is a **benchmark-style contribution**, not a new detector or a new motif-discovery algorithm. The representation layer and scoring strategies (RQ1/RQ2) exist to make RQ3 measurable across many backends under identical conditions.

## 2. Research questions

- **RQ1** — How should a time series be encoded into a motif-based representation? (distance-to-nearest-motif, occurrence-based, reconstruction-error-based)
- **RQ2** — How should that representation be turned into an anomaly score? (rule-based threshold vs. a lightweight detector like kNN/IsolationForest on top of the representation)
- **RQ3** — How does motif discovery *quality* relate to downstream anomaly detection *performance*, across backends of differing sophistication?

Advisor-confirmed: these RQs are a guide, not a fixed contract. Pivoting within "TSAD + TSMD" is fine, and RQs get refined together throughout.

## 3. Method stack (updated)

| Role | Tool | Notes |
|---|---|---|
| Motif discovery backends (primary) | `pip install tsmd` | 11 methods behind one common interface (`.fit(signal)` → `.prediction_mask_`): SetFinder, LatentMotif, STOMP, VALMOD, PanMP, k-Motiflets, PEPA, A-PEPA, GrammarViz, MDL-Clust, LoCoMotif. **This replaces the hand-built adapters originally planned for Phase 2.** |
| Motif discovery backend (extra) | `pip install leitmotif` (LAMA) | Not included in `tsmd`; needs its own small adapter if you want it in the comparison. |
| Dataset + anomaly ground truth | `TSB-AD` (`pip install TSB-AD`, data from thedatum.org) | Source of the actual time series and anomaly labels. **Do not use `tsmd`'s own bundled datasets (arm-CODA, mitdb, REFIT, etc.) for this** — they have motif ground truth but no anomaly labels. |
| Downstream evaluation metric | TSB-AD's `get_metrics()` | Gives VUS-PR / F1 etc. |
| Long-lead / pending | SubTSMD | No public repo as of now. Email sent to KU Leuven authors — treat as optional bonus backend, not a blocker. |

**Decision:** PROM / TSMD-Bench is dropped entirely — not used anywhere in this project, including as a validation step. Motif quality is measured only indirectly, through its effect on downstream TSB-AD performance.

## 4. What to do now — immediate next actions (in order)

1. **Set up the environment.**
   ```bash
   pip install tsmd TSB-AD leitmotif
   ```
2. **Get the toy dataset.** Download `TSB-AD-U.zip` from thedatum.org, unzip, and pick one ECG series from the MITDB folder. Slice a few thousand points around one labeled anomaly (don't use the full ~630K-point recording yet — speed matters more than completeness at this stage).
3. **Run two `tsmd` backends on the slice.** Start with k-Motiflets and LoCoMotif (both already in `tsmd`) — call `.fit(signal)`, inspect `.prediction_mask_`. No representation or scoring yet, just confirm the motifs look right against the visible heartbeat pattern.
4. **Build the smallest possible representation.** Distance from each point to the nearest detected motif occurrence — this is RQ1's minimum viable version.
5. **Build the smallest possible scorer.** Threshold the distance from step 4 — this is RQ2's minimum viable version.
6. **Evaluate with TSB-AD's own metric.** Feed your anomaly score array into `get_metrics()` alongside the label column, check VUS-PR is non-degenerate (not 0, not trivially 1).
7. **Write the one-page framing memo** (the "Goal" + "Method stack" sections above are most of it) and bring it, plus this plan, to your next supervisor meeting.

None of steps 1–6 require supervisor sign-off — they're the thin vertical slice from Phase 1.

## 5. Phase plan (updated)

| Phase | Focus | What changed | Est. duration |
|---|---|---|---|
| 0 — Framing | Lock in benchmark framing, univariate scope, toy dataset | No change | 1 week |
| 1 — Thin vertical slice | k-Motiflets + LoCoMotif via `tsmd`, one representation, one scorer, VUS-PR sanity check | No change in goal, but `tsmd`'s common interface makes this faster to reach | 2–3 weeks |
| 2 — Representation layer (RQ1) | Representation logic across backends; 3 representation variants (distance/occurrence/reconstruction-error) | **Shrinks significantly** — `tsmd` already supplies the common backend interface for 11 methods. Remaining work is mostly the representation logic itself, plus one manual adapter for LAMA. Consider reallocating saved time to Phase 4. | 2–3 weeks (was 4) |
| 3 — Scoring strategies (RQ2) | Rule-based vs. lightweight detector, run across a TSB-AD subset (5–10 series) | No change | 3–4 weeks |
| 4 — Propagation study (RQ3) | Quality vs. performance across backends | Use the 11 `tsmd` methods' natural spread of sophistication as the quality gradient (crude/fast SetFinder → sophisticated/slow LoCoMotif/PEPA), instead of manually degrading one method. Add a random-motif baseline as the floor. Quality is inferred entirely from downstream VUS-PR — no separate motif-quality metric is computed. | 4 weeks |
| 5 — Write-up | Full draft | No change | 3–4 weeks |

Net effect: Phase 2 shrinks by roughly a week or two; reallocate that time to Phase 4, where the actual contribution lives.

## 6. Open decisions to raise with the supervisor

- **RQ3 design (settled)**: motif discovery quality is treated as a latent variable, operationalized only through its effect on downstream VUS-PR — no PROM/TSMD-Bench validation step. Quality varies via (a) the natural spread across `tsmd`'s 11 methods, and (b) a random-motif baseline as the floor.
- **Framing consequence**: since quality is never measured directly, the thesis's claims about "quality" are strictly claims about *downstream-performance-inferred* quality. Worth stating explicitly in the write-up as a scoping choice, not an oversight.
- **Whether to include LAMA and/or SubTSMD** as extra backends beyond the 11 in `tsmd`, given the adapter work each requires.

## 7. Key references

- TSAD foundations: Schmidl et al. (VLDB 2022); Liu & Paparrizos, *TSB-AD* (NeurIPS 2024); Boniol et al., decade review (2024)
- TSMD backends: Motiflets (Schäfer & Leser, VLDB 2022); LoCoMotif (Van Wesenbeeck et al., DMKD 2024); LAMA/leitmotifs (Schäfer & Leser, VLDB 2025); Guerrini et al., *TSMD: A Comprehensive Evaluation* (VLDB 2025) — source of the `tsmd` package
- Pending: SubTSMD (Carpentier et al., DMKD 2026) — no public repo yet, KU Leuven contact in progress