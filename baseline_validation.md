# Baseline Validation Report (Gate G2)

## What was validated

The NAIVE update strategy (retrain-from-scratch on the full, updated
training pool, no compatibility awareness) is the real-world default
practice critiqued by the backward-compatibility literature, and therefore
serves as this study's primary baseline. Before any mitigation method (PCT,
BCWI) was implemented or run, I validated that:

1. **The environment is correctly configured and the dataset loads
   offline**, with no silent leakage between the fixed test set and any
   training pool. Evidence: `results/logs/00_env_check.log` (full console
   output of `scripts/00_env_check.py`, including an explicit disjointness
   assertion between TEST and TRAIN-POOL indices, and a determinism check
   confirming the same seed reproduces the identical test split).
2. **The from-scratch NumPy softmax classifier actually trains and
   converges** — cross-entropy loss decreases monotonically for both OLD
   and NAIVE across 300 epochs, and the trained classifier clears a sanity
   floor (>50% test accuracy on a 10-class task; actual values below).
   Evidence: `results/logs/01_train_baseline.log`.
3. **The negative-flip-rate (NFR) metric is wired correctly and is
   non-degenerate**: NAIVE's NFR relative to OLD is strictly greater than
   zero at the reference seed (0.83%, i.e., 3 of 360 held-out test
   instances that OLD got right were newly misclassified by NAIVE), which
   confirms the metric is detecting real instance-level regressions rather
   than trivially returning zero.
4. **OLD and NAIVE share an identical initial weight vector**, which is a
   precondition this study relies on for BCWI's weight interpolation
   (Section 8 of `research_decisions.md`) to be a meaningful operation
   later; this was asserted in code (`np.array_equal`), not just assumed.

## Reference-seed numbers (seed = 42, no corruption, OLD = 25% of pool, NAIVE = 100% of pool)

| Model | Test accuracy | NFR vs. OLD | PFR vs. OLD |
|---|---|---|---|
| OLD   | 0.9278 | — | — |
| NAIVE | 0.9611 | 0.0083 (3/360) | 0.0417 (15/360) |

VERIFIED: NAIVE (trained on ~4x more data than OLD) outperforms OLD in
aggregate accuracy, as expected, while still introducing a small but
nonzero rate of instance-level regressions — reproducing, in miniature, the
"aggregate improvement coexists with instance-level regression" pattern
that Bansal et al. (2019) and Echterhoff et al. (2024) report for
production classifiers and LLMs, respectively.

## Discrepancies / things checked and found acceptable

- The reference-seed NFR (0.83%) is small in absolute terms because
  `load_digits` is a comparatively easy, near-linearly-separable 10-class
  task. This was investigated via the protocol-design pilot documented in
  `research_decisions.md` Section 9, which led to the corruption-magnitude
  (σ) axis used in the main sweep — explicitly reported as a design
  decision, not hidden.
- No numerical-stability issues (NaN/Inf losses, exploding gradients) were
  observed across any of the 300-epoch runs logged in
  `results/logs/02_ablation.log` or `results/logs/03_magnitude_sweep.log`.

## Decision

Baseline accepted. The frozen protocol (`research_decisions.md`, Sections
6–9) was then applied unchanged across the ablation sweep (`scripts/
02_ablation.py`) and the magnitude-sensitivity sweep (`scripts/
03_magnitude_sweep.py`).

---
**Update (late-session correction).** `scripts/01_train_baseline.py` is a single-seed (42) diagnostic
run that predates the validation split and prints test-set numbers; it checks that the code works
and is *not* a reproduction of any published result (none exists for this testbed) and not a
reported result. All reported results come from `03_magnitude_sweep.py` on the test split after
selection on validation. Items still NEEDS CORRECTION / UNKNOWN: no unit tests beyond the sanity
checks; PCT not verified against the original paper; trainer lr/epochs not validated.
