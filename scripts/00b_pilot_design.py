"""
Design pilot (run on the VALIDATION split only, never the test set) used to
choose the 'update magnitude' axis. Reproduces, with a saved log, the two
pilots described in research_decisions.md Section 9.
  (a) vary only the OLD model's data fraction (rejected: too little signal)
  (b) vary sigma = Gaussian corruption of NEW model's training data
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from src.data import load_full_dataset, make_fixed_test_split, make_val_split, subsample_pool, fit_scaler
from src.model import SoftmaxClassifier
from src.metrics import accuracy, negative_flip_rate
from src.harness import build_condition, train_naive, evaluate

SEEDS = [0, 1, 2, 3, 4]
print("=== Pilot (a): vary OLD-model data fraction only (validation split, mean over seeds) ===")
X, y = load_full_dataset()
for old_frac in [0.08, 0.10, 0.15, 0.20, 0.25]:
    nfrs, accs = [], []
    for seed in SEEDS:
        Xp, yp, _, _, _, _ = make_fixed_test_split(X, y, seed=seed)
        Xt, yt, Xv, yv = make_val_split(Xp, yp, seed=seed)
        Xo, yo = subsample_pool(Xt, yt, old_frac, seed)
        Xn, yn = subsample_pool(Xt, yt, 1.0, seed)
        sc = fit_scaler(Xn)
        rng = np.random.RandomState(seed); W = rng.normal(0, .01, (64, 10)); b = np.zeros(10)
        mo = SoftmaxClassifier(64, 10, lr=.5, n_epochs=300, seed=seed, init_W=W, init_b=b); mo.fit(sc.transform(Xo), yo)
        mn = SoftmaxClassifier(64, 10, lr=.5, n_epochs=300, seed=seed, init_W=W, init_b=b); mn.fit(sc.transform(Xn), yn)
        po, pn = mo.predict(sc.transform(Xv)), mn.predict(sc.transform(Xv))
        nfrs.append(negative_flip_rate(yv, po, pn)); accs.append(accuracy(yv, pn))
    print(f"old_frac={old_frac:.2f}  NAIVE val acc={np.mean(accs):.4f}  val NFR={np.mean(nfrs):.4f}")

print("\n=== Pilot (b): vary sigma (corruption of NEW training data), validation split, mean over seeds ===")
for sigma in [0, 2, 4, 6, 8, 10]:
    nfrs, accs = [], []
    for seed in SEEDS:
        c = build_condition(seed, float(sigma)); r = evaluate(c, train_naive(c), "NAIVE", split="val")
        nfrs.append(r["nfr"]); accs.append(r["accuracy"])
    print(f"sigma={sigma:>2}  NAIVE val acc={np.mean(accs):.4f}  val NFR={np.mean(nfrs):.4f}")
