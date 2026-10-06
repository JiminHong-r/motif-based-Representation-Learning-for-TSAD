Development splits for cycle 2 (PROJECT.md §5). Eval (`TSB-AD-U-Eva.csv`) is never part of these.

- `dev.csv` — the 48 Tuning series + the 472 TSB-AD-U series in neither Tuning nor Eval (520).
- `dev_small.csv` — all 48 Tuning series + 102 extra series sampled proportionally to source with `random_state=2026` (150), for fast iteration.
