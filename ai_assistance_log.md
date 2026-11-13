# AI Assistance / Research-Integrity Log

**Status: DRAFT written by the AI assistant. The student must review, correct and
complete the sections marked [STUDENT TO COMPLETE] before submission.**
The brief (Prompt 18) forbids inventing experiences; this log therefore records only
what actually happened in the working session, and does not claim human interventions
that did not occur.

## 1. What actually happened

- Tool: Claude (Sonnet 5, later switched to Sonnet 5.5) in a Claude Code cloud sandbox
  (CPU only, no GPU, no access to commercial LLM APIs).
- The student's instructions in this session were: "complete [the new task] and give me
  all the required files", then "continue". The student did **not** review intermediate
  results, choose among design alternatives, or catch any of the bugs below during the
  session. All design decisions, code and text in this repository were produced by the
  AI. Whatever review the student does afterwards is the student's own.
- Prompts 01-18 of the brief were **not** run as a prompt sequence. The brief's workflow
  was approximated by the AI; the assignment PDF itself was only read late in the session
  (see item 3 below).

## 2. Significant AI-assisted activities

| # | Activity | What the AI did | How it was checked | Outcome |
|---|---|---|---|---|
| 1 | Scoping | Moved from the LLM-API drift topic (Paper 1) to the classical negative-flip literature cited in Paper 1, because no LLM API/GPU access exists in the sandbox | Reasoning recorded in `research_decisions.md` §2 | Narrower scope than Paper 1; stated as a limitation in the paper |
| 2 | Code | Wrote NumPy softmax classifier, PCT loss term, BCWI interpolation, metrics, harness, scripts | Scripts executed; logs in `results/logs/`; `00_env_check`, `01_train_baseline` sanity checks | Runs end to end. **No independent unit tests beyond these sanity checks** |
| 3 | Brief compliance | See "AI limitations" below | - | - |
| 4 | Figures/tables | Generated from CSVs by `05_make_figures.py` and `06_make_tables.py`; figures viewed as images by the AI | Numbers in tables are produced programmatically | Traceable to `results/csv/` |

## 3. AI limitations / corrections that really occurred (real examples)

1. **Selection-rule bug (caught by the AI from printed output).** The first version of
   `02_ablation.py` added an ad-hoc "must beat the OLD model" floor that was not part of
   the criterion written in `research_decisions.md` §4. The printed log showed every
   candidate rejected except the degenerate alpha=1.0 (which equals the old model). The AI
   removed the floor and excluded alpha=1.0 explicitly as a structural identity.
2. **Weak first design axis (rejected after running it).** Varying only the old model's
   data fraction gave ~1% NFR regardless of fraction. The AI switched to a corruption
   level sigma axis. The pilot is now a saved script (`00b_pilot_design.py`, log in
   `results/logs/`). The *first* pilot ran inline, printed test-set numbers and saved no
   log; it was re-run on the validation split.
3. **Test-set tuning (a real violation, found late, fixed).** The first full ablation chose
   alpha on the same test set used for final reporting, which the brief prohibits (§10).
   The AI only noticed after finally reading the assignment PDF. A validation split was
   added, the ablation now uses validation only, and all downstream results were re-run.
   Final alpha* (0.98) happened to be unchanged, but the earlier numbers were discarded.
4. **Unverified/invented provenance in earlier drafts (corrected).** An earlier draft of this
   log credited the student with decisions ("I caught this...") that did not happen, and
   `research_decisions.md` cited git history and a saved pilot log that did not exist.
   Both were removed/corrected.
5. **PCT is a simplified variant.** The implementation (hard-label consistency penalty on
   training rows the old model got right) was written from the AI's understanding of
   Yan et al. (2021); it was **not checked against the original paper or code** and is
   not a faithful reproduction. The paper labels it "PCT-style".

## 4. Conclusions that depend on human reasoning

[STUDENT TO COMPLETE] - e.g. whether the sigma-corruption proxy is an acceptable stand-in
for a model update, how to interpret BCWI's gains (see paper §6: much of the NFR drop comes
from alpha=0.98 staying close to the old model), and what to say in the viva.

## 5. Remaining risks

- Single dataset (sklearn digits), single linear model, synthetic drift proxy; nothing here
  is evidence about commercial LLMs.
- alpha* sits at the edge of the tested grid next to the degenerate alpha=1.
- 5 seeds, no formal significance test; only per-seed direction counts are reported.
- Hyperparameters of the base trainer (lr, epochs) were not tuned or validated.
- Code is AI-written; the student must read and be able to explain it.
