"""
Ablation sweep (Gate G3 evidence, part 1): PCT's lambda and BCWI's alpha,
swept independently at a single fixed, pre-declared update magnitude
(sigma = 8.0, "strong update" — see research_decisions.md Section 9) across
5 seeds. ALL METRICS HERE ARE ON THE VALIDATION SPLIT (never the test set). Selects the default (lambda*, alpha*) carried into the magnitude
sweep (script 03), per the frozen Section-4 success criterion: >=20% mean
NFR reduction vs. NAIVE at <=2-point mean accuracy cost.
"""
import sys
import csv
import numpy as np

sys.path.insert(0, ".")
from src.harness import build_condition, train_naive, train_pct, build_bcwi, evaluate

SEEDS = [0, 1, 2, 3, 4]
SIGMA_FIXED = 8.0
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.0]

rows = []

print("=" * 70)
print(f"ABLATION SWEEP  (sigma fixed = {SIGMA_FIXED}, seeds = {SEEDS})")
print("=" * 70)

for seed in SEEDS:
    print(f"\n--- seed {seed} ---")
    cond = build_condition(seed=seed, sigma=SIGMA_FIXED)

    naive_model = train_naive(cond)
    r = evaluate(cond, naive_model, "NAIVE", split="val")
    rows.append(r)
    print(f"NAIVE          acc={r['accuracy']:.4f}  NFR={r['nfr']:.4f}  PFR={r['pfr']:.4f}")

    for lam in LAMBDAS:
        if lam == 0.0:
            m = naive_model  # lambda=0 reduces exactly to NAIVE; reuse, don't retrain
        else:
            m = train_pct(cond, lam=lam)
        r = evaluate(cond, m, f"PCT_lambda={lam}", split="val")
        rows.append(r)
        print(f"PCT lam={lam:<5.2f}  acc={r['accuracy']:.4f}  NFR={r['nfr']:.4f}  PFR={r['pfr']:.4f}")

    for alpha in ALPHAS:
        if alpha == 0.0:
            m = naive_model  # alpha=0 reduces exactly to NAIVE
        else:
            m = build_bcwi(cond, naive_model, alpha=alpha)
        r = evaluate(cond, m, f"BCWI_alpha={alpha}", split="val")
        rows.append(r)
        print(f"BCWI a={alpha:<5.2f}   acc={r['accuracy']:.4f}  NFR={r['nfr']:.4f}  PFR={r['pfr']:.4f}")

out_path = "results/csv/ablation_results.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "seed", "sigma", "accuracy", "nfr", "pfr", "acc_old", "split"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {out_path}")

# ---- Select (lambda*, alpha*) per the frozen success criterion ----
import pandas as pd
df = pd.DataFrame(rows)

naive_mean_acc = df[df.label == "NAIVE"]["accuracy"].mean()
naive_mean_nfr = df[df.label == "NAIVE"]["nfr"].mean()
old_mean_acc = df[df.label == "NAIVE"]["acc_old"].mean()
print(f"\nNAIVE mean over seeds: acc={naive_mean_acc:.4f}  NFR={naive_mean_nfr:.4f}")
print(f"OLD   mean over seeds: acc={old_mean_acc:.4f}  (reference floor)")

# NOTE: the criterion frozen in research_decisions.md Section 4 is relative
# to NAIVE (NFR reduction >=20%, accuracy cost <=2 points vs. NAIVE) and was
# never "must beat OLD" — that extra floor was a mistake introduced ad hoc
# in an earlier pass of this script (visible in git history / the first
# version of this file) and has been removed. The only exclusion that is
# legitimately needed is alpha=1.0 itself: that point is a STRUCTURAL
# identity (W_interp = W_old exactly), i.e. "don't apply the update at
# all," not an empirical compatibility-training result, so it is excluded
# from candidate SELECTION even though it is still plotted as a reference
# endpoint in the trade-off curves (Figure: ablation curves).
DEGENERATE_ALPHA = 1.0

print("\n--- PCT candidates (mean over seeds) ---")
best_lambda, best_lambda_nfr = None, None
for lam in LAMBDAS:
    sub = df[df.label == f"PCT_lambda={lam}"]
    macc, mnfr = sub["accuracy"].mean(), sub["nfr"].mean()
    nfr_reduction = 1 - (mnfr / naive_mean_nfr) if naive_mean_nfr > 0 else 0.0
    acc_cost = naive_mean_acc - macc
    meets = (nfr_reduction >= 0.20) and (acc_cost <= 0.02)
    flag = "  <= MEETS CRITERION" if meets else ""
    print(f"lambda={lam:<5.2f} acc={macc:.4f} NFR={mnfr:.4f} NFR_reduction={nfr_reduction:+.1%} acc_cost={acc_cost:+.4f}{flag}")
    if meets and (best_lambda_nfr is None or mnfr < best_lambda_nfr):
        best_lambda, best_lambda_nfr = lam, mnfr

print("\n--- BCWI candidates (mean over seeds) ---")
best_alpha, best_alpha_nfr = None, None
for alpha in ALPHAS:
    sub = df[df.label == f"BCWI_alpha={alpha}"]
    macc, mnfr = sub["accuracy"].mean(), sub["nfr"].mean()
    nfr_reduction = 1 - (mnfr / naive_mean_nfr) if naive_mean_nfr > 0 else 0.0
    acc_cost = naive_mean_acc - macc
    is_degenerate = abs(alpha - DEGENERATE_ALPHA) < 1e-9
    meets = (nfr_reduction >= 0.20) and (acc_cost <= 0.02) and not is_degenerate
    flag = "  <= MEETS CRITERION" if meets else ("  [excluded: structural identity, alpha=1.0]" if is_degenerate else "")
    print(f"alpha={alpha:<5.2f}  acc={macc:.4f} NFR={mnfr:.4f} NFR_reduction={nfr_reduction:+.1%} acc_cost={acc_cost:+.4f}{flag}")
    if meets and (best_alpha_nfr is None or mnfr < best_alpha_nfr):
        best_alpha, best_alpha_nfr = alpha, mnfr

print("\n" + "=" * 70)
print(f"SELECTED DEFAULTS: lambda* = {best_lambda}   alpha* = {best_alpha}")
print("(carried unchanged into scripts/03_magnitude_sweep.py)")
print("=" * 70)

with open("results/csv/selected_hyperparams.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["param", "value"])
    writer.writerow(["lambda_star", best_lambda])
    writer.writerow(["alpha_star", best_alpha])
