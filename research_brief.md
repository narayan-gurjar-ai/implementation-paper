# One-page research brief (Gate G1 template, brief §11)

| Field | Content |
|---|---|
| Review topic and key papers | Performance drift across LLM version updates (Paper 1). Papers used: Bansal et al. 2019 (negative flips); Yan et al. 2021 (positive-congruent training); Schumann et al. 2024 (BCWI); Chen et al. 2024 and Echterhoff et al. 2024 (LLM drift evidence); Gama et al. 2014. **[STUDENT TO VERIFY each against the original paper; the .bib entries were written by the AI and only partly checked against sources.]** |
| Research gap | Paper 1 notes mitigation methods are each evaluated against their own baselines, with heterogeneous metrics. [INFERRED reading of Paper 1; novelty not established by a separate literature search - UNKNOWN] |
| Research question | How do a PCT-style consistency penalty and BCWI compare to naive retraining in accuracy-vs-NFR as update magnitude grows? SQ1-SQ3 in the paper. |
| Study pathway | B - controlled comparative study |
| Dataset / measurement | sklearn digits (offline, BSD-licensed package); stratified test 20% never used for selection, validation 20% of remainder; 5 seeds |
| Baseline / comparators | NAIVE retraining; PCT-style (simplified, unverified vs original); BCWI; frozen OLD reference. All implemented from scratch in NumPy (no official code used) |
| Primary metric and control | NFR (primary), accuracy, PFR. Held constant: data splits, init weights, trainer settings, seeds |
| Computing and risks | CPU only, ~80 s. Risks: easy dataset, synthetic drift proxy, no LLM relevance, AI-written code |
| First executable milestone | Done: `scripts/00_env_check.py`, log in `results/logs/` |
