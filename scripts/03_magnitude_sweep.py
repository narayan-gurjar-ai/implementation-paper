"""
Magnitude-sensitivity sweep (Gate G3 evidence, part 2 — answers SQ2).

Uses the (lambda, alpha) selected by scripts/02_ablation.py (read from
results/csv/selected_hyperparams.csv) and holds them FIXED across every
update magnitude sigma in {0, 2, 4, 6, 8, 10}, for 5 seeds, comparing
OLD-frozen, NAIVE, PCT, and BCWI.

PCT note: no lambda met the Section-4 success criterion in the ablation
(a genuine negative result — see results/logs/02_ablation.log). We still
carry PCT forward at its mildest tested setting, lambda=0.25, specifically
to check whether that negative result is consistent across magnitudes or
only an artifact of the sigma=8 ablation point (an honest robustness check
on the negative finding itself, not a cherry-picked omission).
"""
import sys
import csv

sys.path.insert(0, ".")
from src.harness import build_condition, train_naive, train_pct, build_bcwi, evaluate

SEEDS = [0, 1, 2, 3, 4]
SIGMAS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
PCT_LAMBDA_CARRIED = 0.25  # mildest tested setting (see module docstring)

# Read the BCWI alpha selected by the ablation sweep
alpha_star = None
with open("results/csv/selected_hyperparams.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["param"] == "alpha_star":
            alpha_star = float(row["value"])
assert alpha_star is not None, "run scripts/02_ablation.py first"
print(f"Using BCWI alpha* = {alpha_star} (selected by ablation sweep)")
print(f"Using PCT lambda (carried, did not meet criterion) = {PCT_LAMBDA_CARRIED}")

rows = []
print("=" * 70)
print(f"MAGNITUDE SWEEP  (seeds = {SEEDS}, sigmas = {SIGMAS})")
print("=" * 70)

for seed in SEEDS:
    print(f"\n--- seed {seed} ---")
    for sigma in SIGMAS:
        cond = build_condition(seed=seed, sigma=sigma)

        r_old = evaluate(cond, cond["old_model"], "OLD_frozen")
        rows.append(r_old)

        naive_model = train_naive(cond)
        r_naive = evaluate(cond, naive_model, "NAIVE")
        rows.append(r_naive)

        pct_model = train_pct(cond, lam=PCT_LAMBDA_CARRIED)
        r_pct = evaluate(cond, pct_model, "PCT")
        rows.append(r_pct)

        bcwi_model = build_bcwi(cond, naive_model, alpha=alpha_star)
        r_bcwi = evaluate(cond, bcwi_model, "BCWI")
        rows.append(r_bcwi)

        print(
            f"sigma={sigma:4.1f}  "
            f"OLD acc={r_old['accuracy']:.3f}  "
            f"NAIVE acc={r_naive['accuracy']:.3f} NFR={r_naive['nfr']:.3f}  "
            f"PCT acc={r_pct['accuracy']:.3f} NFR={r_pct['nfr']:.3f}  "
            f"BCWI acc={r_bcwi['accuracy']:.3f} NFR={r_bcwi['nfr']:.3f}"
        )

out_path = "results/csv/magnitude_sweep_results.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "seed", "sigma", "accuracy", "nfr", "pfr", "acc_old", "split"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {out_path}")

# --- summary table (mean +/- std across seeds, per method per sigma) ---
import pandas as pd
df = pd.DataFrame(rows)
summary = df.groupby(["label", "sigma"]).agg(
    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    nfr_mean=("nfr", "mean"), nfr_std=("nfr", "std"),
    pfr_mean=("pfr", "mean"), pfr_std=("pfr", "std"),
).reset_index()
summary.to_csv("results/csv/magnitude_sweep_summary.csv", index=False)
print("\n--- Summary (mean +/- std over 5 seeds) ---")
with pd.option_context("display.width", 140, "display.max_rows", 50):
    print(summary.round(4).to_string(index=False))
