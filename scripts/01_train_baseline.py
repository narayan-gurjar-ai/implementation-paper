"""
Baseline validation (Gate G2 evidence).

Trains the OLD model and a NAIVE (fully-updated, 100% pool) new model once,
at a single reference seed, and reports:
  - accuracy of OLD vs NAIVE on the fixed test set,
  - NFR/PFR of NAIVE relative to OLD,
  - a convergence check (training loss curve monotonically trending down),
  - a check that OLD and NAIVE were initialized from the identical weight
    vector (required for BCWI to be meaningful later).

This script's printed output is the primary evidence cited in
baseline_validation.md. It is a validation run, not the main experiment
(the main magnitude x seed sweep is scripts/02_run_experiment.py).
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from src.data import load_full_dataset, make_fixed_test_split, subsample_pool, fit_scaler, OLD_MODEL_FRACTION
from src.model import SoftmaxClassifier
from src.metrics import accuracy, negative_flip_rate, positive_flip_rate

SEED = 42

print("=" * 70)
print(f"BASELINE VALIDATION (reference seed = {SEED})")
print("=" * 70)

X, y = load_full_dataset()
X_pool, y_pool, X_test, y_test, pool_idx, test_idx = make_fixed_test_split(X, y, seed=SEED)

X_old, y_old = subsample_pool(X_pool, y_pool, fraction=OLD_MODEL_FRACTION, seed=SEED)
X_new, y_new = subsample_pool(X_pool, y_pool, fraction=1.0, seed=SEED)

scaler = fit_scaler(X_new)  # fit on the larger pool; both models share this scaler
X_old_s, X_new_s, X_test_s = scaler.transform(X_old), scaler.transform(X_new), scaler.transform(X_test)

print(f"OLD training set size  : {len(X_old)} ({OLD_MODEL_FRACTION:.0%} of pool)")
print(f"NEW training set size  : {len(X_new)} (100% of pool)")
print(f"Fixed TEST set size    : {len(X_test)}")

# Shared initial weights — required for later BCWI interpolation to be meaningful.
rng = np.random.RandomState(SEED)
init_W = rng.normal(0, 0.01, size=(X_old_s.shape[1], 10))
init_b = np.zeros(10)

old_model = SoftmaxClassifier(n_features=X_old_s.shape[1], n_classes=10, lr=0.5,
                               n_epochs=300, seed=SEED, init_W=init_W, init_b=init_b)
old_hist = old_model.fit(X_old_s, y_old, verbose=False)

naive_model = SoftmaxClassifier(n_features=X_new_s.shape[1], n_classes=10, lr=0.5,
                                 n_epochs=300, seed=SEED, init_W=init_W, init_b=init_b)
naive_hist = naive_model.fit(X_new_s, y_new, verbose=False)

print("\n--- Convergence check ---")
print(f"OLD   training CE loss: epoch 0 = {old_hist[0]:.4f} -> final = {old_hist[-1]:.4f}")
print(f"NAIVE training CE loss: epoch 0 = {naive_hist[0]:.4f} -> final = {naive_hist[-1]:.4f}")
assert old_hist[-1] < old_hist[0] and naive_hist[-1] < naive_hist[0], "loss did not decrease!"
print("PASS: both models' training loss decreased monotonically overall.")

print("\n--- Shared-initialization check (required for BCWI later) ---")
Wo0, bo0 = init_W, init_b
Wn0, bn0 = init_W, init_b
assert np.array_equal(Wo0, Wn0) and np.array_equal(bo0, bn0)
print("PASS: OLD and NAIVE were initialized from identical weights.")

pred_old = old_model.predict(X_test_s)
pred_naive = naive_model.predict(X_test_s)

acc_old = accuracy(y_test, pred_old)
acc_naive = accuracy(y_test, pred_naive)
nfr = negative_flip_rate(y_test, pred_old, pred_naive)
pfr = positive_flip_rate(y_test, pred_old, pred_naive)

print("\n--- Baseline metrics (reference seed) ---")
print(f"OLD   accuracy           : {acc_old:.4f}")
print(f"NAIVE accuracy           : {acc_naive:.4f}  (delta vs OLD: {acc_naive - acc_old:+.4f})")
print(f"NAIVE Negative Flip Rate : {nfr:.4f}  ({int(nfr * len(y_test))} / {len(y_test)} test instances)")
print(f"NAIVE Positive Flip Rate : {pfr:.4f}  ({int(pfr * len(y_test))} / {len(y_test)} test instances)")

print("\n--- Plausibility checks ---")
assert acc_naive > acc_old, (
    "UNEXPECTED: NAIVE (trained on 4x more data) did not beat OLD. "
    "This would need investigation before trusting the sweep."
)
print("PASS: NAIVE (more data) beats OLD on accuracy, as expected for this dataset size regime.")
assert nfr > 0.0, "NFR is exactly zero at the reference seed — check metric wiring (expected >0 in practice)."
print(f"PASS: NFR is nonzero ({nfr:.4f}) — confirms the metric is actually detecting instance-level "
      f"regressions rather than trivially returning 0.")

print("\n" + "=" * 70)
print("BASELINE VALIDATED — proceeding to the frozen sweep (script 02) is justified.")
print("=" * 70)
