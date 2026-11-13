"""
Failure/error analysis (Gate G3 evidence, part 3 — answers SQ3; satisfies
the "at least one deeper analysis" requirement).

At a fixed, strong update magnitude (sigma=8), aggregated over the 5 seeds,
breaks down WHICH true digit classes drive NAIVE's negative flips, and
checks whether BCWI's residual negative flips concentrate on the same
classes or different ones.
"""
import sys
import csv
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from src.harness import build_condition, train_naive, build_bcwi
from src.metrics import negative_flip_indices

SEEDS = [0, 1, 2, 3, 4]
SIGMA = 8.0
import csv as _csv
ALPHA_STAR = None
with open("results/csv/selected_hyperparams.csv") as _f:
    for _r in _csv.DictReader(_f):
        if _r["param"] == "alpha_star":
            ALPHA_STAR = float(_r["value"])
assert ALPHA_STAR is not None

print("=" * 70)
print(f"FAILURE ANALYSIS  (sigma={SIGMA}, seeds={SEEDS}, alpha*={ALPHA_STAR})")
print("=" * 70)

naive_flip_classes = Counter()
bcwi_flip_classes = Counter()
naive_total_flips = 0
bcwi_total_flips = 0
example_rows = []  # (seed, true_class, old_pred, naive_pred, bcwi_pred) for a few examples

for seed in SEEDS:
    cond = build_condition(seed=seed, sigma=SIGMA)
    y_test = cond["y_test"]

    naive_model = train_naive(cond)
    bcwi_model = build_bcwi(cond, naive_model, alpha=ALPHA_STAR)

    pred_old = cond["old_model"].predict(cond["X_test_s"])
    pred_naive = naive_model.predict(cond["X_test_s"])
    pred_bcwi = bcwi_model.predict(cond["X_test_s"])

    naive_flips = negative_flip_indices(y_test, pred_old, pred_naive)
    bcwi_flips = negative_flip_indices(y_test, pred_old, pred_bcwi)

    naive_total_flips += len(naive_flips)
    bcwi_total_flips += len(bcwi_flips)

    for i in naive_flips:
        naive_flip_classes[int(y_test[i])] += 1
        example_rows.append(dict(
            seed=seed, method="NAIVE", true_class=int(y_test[i]),
            old_pred=int(pred_old[i]), new_pred=int(pred_naive[i]),
        ))
    for i in bcwi_flips:
        bcwi_flip_classes[int(y_test[i])] += 1
        example_rows.append(dict(
            seed=seed, method="BCWI", true_class=int(y_test[i]),
            old_pred=int(pred_old[i]), new_pred=int(pred_bcwi[i]),
        ))

    print(f"seed {seed}: NAIVE flips = {len(naive_flips):3d} / 360   BCWI flips = {len(bcwi_flips):3d} / 360")

print(f"\nTOTAL across 5 seeds (out of {5*360} test evaluations):")
print(f"  NAIVE negative flips: {naive_total_flips}  ({naive_total_flips/(5*360):.3%} of all test evals)")
print(f"  BCWI  negative flips: {bcwi_total_flips}  ({bcwi_total_flips/(5*360):.3%} of all test evals)")
print(f"  Relative reduction: {1 - bcwi_total_flips/naive_total_flips:+.1%}")

print("\n--- Negative flips by TRUE digit class (NAIVE) ---")
for cls in range(10):
    print(f"  digit {cls}: {naive_flip_classes.get(cls, 0):3d}")

print("\n--- Negative flips by TRUE digit class (BCWI) ---")
for cls in range(10):
    print(f"  digit {cls}: {bcwi_flip_classes.get(cls, 0):3d}")

most_naive = naive_flip_classes.most_common(3)
most_bcwi = bcwi_flip_classes.most_common(3)
print(f"\nTop-3 most-flipped true classes, NAIVE: {most_naive}")
print(f"Top-3 most-flipped true classes, BCWI : {most_bcwi}")

out_path = "results/csv/failure_analysis_flips.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed", "method", "true_class", "old_pred", "new_pred"])
    writer.writeheader()
    writer.writerows(example_rows)
print(f"\nSaved {len(example_rows)} individual flip records to {out_path}")

by_class_path = "results/csv/failure_analysis_by_class.csv"
with open(by_class_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true_class", "naive_flip_count", "bcwi_flip_count"])
    for cls in range(10):
        writer.writerow([cls, naive_flip_classes.get(cls, 0), bcwi_flip_classes.get(cls, 0)])
print(f"Saved per-class summary to {by_class_path}")
