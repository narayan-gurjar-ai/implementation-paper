# Negative-flip-aware model updates under controlled drift (AITR Lab 02)

Author: Narayan Gurjar (MSE2026002). Pathway B (controlled comparative study).

**Question.** In a controlled testbed, how do a simplified PCT-style consistency penalty and
backward-compatible weight interpolation (BCWI) compare with naive retraining in the
accuracy-vs-negative-flip-rate trade-off, as the "update magnitude" grows?
Derived from my literature review on LLM performance drift (Paper 1); see `research_decisions.md`.

**Scope warning.** One small dataset (sklearn digits), one linear model, synthetic noise as the
drift proxy. Nothing here is evidence about commercial LLMs. PCT is a simplified variant, not a
verified reproduction of Yan et al. (2021).

## Reproduce (CPU only, ~80 s, executed successfully in the development sandbox)
```bash
pip install -r requirements.txt
./run_all.sh            # runs scripts/00..06 and builds paper/main.pdf (needs a LaTeX install with acmart fonts)
```
Without LaTeX: run the `python3 scripts/<name>.py` lines from `run_all.sh` individually.
Every script writes a log to `results/logs/`, CSVs to `results/csv/`, figures to `results/figures/`
and `paper/figures/`, tables to `paper/tables/`. Runs are deterministic (re-run gave identical CSVs).

| Script | Purpose |
|---|---|
| `00_env_check.py` | versions, offline data load, split determinism/disjointness, tiny training run |
| `00b_pilot_design.py` | choice of update-magnitude axis (validation split) |
| `01_train_baseline.py` | baseline validation sanity checks (see `baseline_validation.md`) |
| `02_ablation.py` | PCT lambda / BCWI alpha sweep, **validation split only**; writes selected alpha |
| `03_magnitude_sweep.py` | frozen comparison over sigma in {0,...,10}, 5 seeds, **test split** |
| `04_failure_analysis.py` | per-class negative flips, NAIVE vs BCWI at sigma=8 |
| `05_make_figures.py`, `06_make_tables.py` | figures/tables generated from the CSVs |

`notebooks/walkthrough.ipynb` loads the saved results and re-runs one condition.

## Data
`sklearn.datasets.load_digits` (loaded offline from scikit-learn). See `data/README.md`.

## Protocol (frozen in `configs/experiment_config.yaml`)
Per seed: stratified 20% test set (never used for selection); 20% of the rest = validation;
old model = 25% of the training pool (clean); new model = 100% with Gaussian noise of std sigma.
Seeds 0-4. Note: the success criterion was fixed after exploratory pilots (see paper, Limitations).

## Known limitations / honesty notes
See paper Section "Limitations" and `ai_assistance_log.md` (including a test-set-tuning mistake
that was found and fixed by re-running everything on a validation split).
Pre-fix numbers are not retained. `baseline_validation.md` / `01_train_baseline.py` use a
sanity-check run at seed 42 (pre-validation-split, a diagnostic only, not a reported result).

## Credits
scikit-learn digits dataset (Alpaydin & Kaynak, UCI). ACM `acmart` class (LPPL). Code written
with AI assistance (Claude); see `ai_assistance_log.md`. License: MIT for this repository's code
[STUDENT TO CONFIRM].
