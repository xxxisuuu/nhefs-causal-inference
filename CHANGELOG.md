# Changelog

## v1.1.0 (2026-10-27)

### Methods
- G-formula and AIPW now use the same inverse probability of censoring weights (IPCW) as IP weighting, so all three target the full baseline population instead of the 1,566 uncensored. G-formula 3.52 -> 3.50 kg, AIPW 3.46 -> 3.47 kg.
- `results.csv` gains `target` (ATE / ATT / crude) and `censoring` (IPCW / complete case) columns.
- PS matching caliper is now an explicit `caliper_sd` argument (default 0.2 SD of the logit PS, same as before). The run prints how many quitters went unmatched (3 of 403).
- Love plot (fig 2) adds post-matching balance.

### New
- E-values for unmeasured confounding (`models.evalue`), added to `results.csv` and the README.

### Fixes
- One PS truncation rule (`PS_TRIM = 0.01`) for IPW and AIPW; before, only AIPW was clipped.
- The bootstrap no longer swallows every exception. Only separation / singular-matrix failures are tolerated, and their count per estimator is printed.

### Engineering
- `requirements.txt` pins tested versions; README notes that kaleido 1.x needs Chrome.
- The data URL is pinned to an Rdatasets commit instead of `master`.
- pytest regression tests (`tests/`) and a GitHub Actions workflow.
