"""
Environment and data sanity check (Gate G1/G2 evidence).

Verifies: library versions, offline dataset load, fixed-test-set
determinism across repeated calls, and one tiny end-to-end training step
with real, printed console output. Exit code 0 => all checks passed.
"""
import sys
import time
import numpy as np
import sklearn

sys.path.insert(0, ".")
from src.data import load_full_dataset, make_fixed_test_split, subsample_pool, fit_scaler
from src.model import SoftmaxClassifier
from src.metrics import accuracy

print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)
print(f"numpy version      : {np.__version__}")
print(f"sklearn version     : {sklearn.__version__}")
print(f"python version      : {sys.version.split()[0]}")
print(f"timestamp (UTC-ish) : {time.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n--- Dataset load ---")
X, y = load_full_dataset()
print(f"X shape = {X.shape}, y shape = {y.shape}, classes = {sorted(set(y.tolist()))}")
assert X.shape == (1797, 64), "unexpected digits dataset shape"
assert len(set(y.tolist())) == 10, "expected 10 classes"
print("PASS: dataset shape and class count as expected.")

print("\n--- Fixed test-set determinism check ---")
X_pool1, y_pool1, X_test1, y_test1, pool_idx1, test_idx1 = make_fixed_test_split(X, y, seed=42)
X_pool2, y_pool2, X_test2, y_test2, pool_idx2, test_idx2 = make_fixed_test_split(X, y, seed=42)
assert np.array_equal(test_idx1, test_idx2), "test split is not deterministic for a fixed seed!"
print(f"TEST set size = {len(test_idx1)} ({len(test_idx1)/len(y):.1%} of data)")
print(f"TRAIN pool size = {len(pool_idx1)} ({len(pool_idx1)/len(y):.1%} of data)")
print("PASS: identical seed reproduces an identical test set (no leakage drift across runs).")

# confirm test set is disjoint from pool
overlap = set(test_idx1.tolist()) & set(pool_idx1.tolist())
assert len(overlap) == 0, "TEST and POOL overlap!"
print("PASS: TEST and TRAIN-POOL are disjoint (no leakage).")

print("\n--- Tiny end-to-end training step (real execution, not a mock) ---")
X_old, y_old = subsample_pool(X_pool1, y_pool1, fraction=0.25, seed=0)
scaler = fit_scaler(X_old)
X_old_s = scaler.transform(X_old)
X_test_s = scaler.transform(X_test1)

clf = SoftmaxClassifier(n_features=X_old_s.shape[1], n_classes=10, lr=0.5, n_epochs=60, seed=0)
t0 = time.time()
history = clf.fit(X_old_s, y_old, verbose=True)
elapsed = time.time() - t0
pred = clf.predict(X_test_s)
acc = accuracy(y_test1, pred)
print(f"\nTraining wall time: {elapsed:.2f}s for {len(X_old)} samples, 60 epochs")
print(f"Final training CE loss: {history[-1]:.4f}")
print(f"Test accuracy (sanity run, NOT the main experiment): {acc:.4f}")
assert acc > 0.5, "sanity-check accuracy implausibly low; investigate before proceeding"
print("PASS: sanity-check model trains and clears a plausibility floor (>50% acc on a 10-class task).")

print("\n" + "=" * 70)
print("ALL ENVIRONMENT CHECKS PASSED")
print("=" * 70)
