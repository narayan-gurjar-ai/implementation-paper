"""
Shared experiment harness: builds OLD / NAIVE / PCT / BCWI models for one
(seed, sigma) condition and returns their test-set predictions and metrics.
Used by both scripts/02_ablation.py and scripts/03_magnitude_sweep.py so the
two scripts cannot silently drift apart in how a "condition" is constructed.
"""
from __future__ import annotations

import numpy as np

from src.data import (
    load_full_dataset, make_fixed_test_split, make_val_split, subsample_pool, fit_scaler,
    corrupt_images, OLD_MODEL_FRACTION,
)
from src.model import SoftmaxClassifier, interpolate_weights
from src.metrics import accuracy, negative_flip_rate, positive_flip_rate

N_EPOCHS = 300
LR = 0.5


def build_condition(seed: int, sigma: float):
    """
    Returns a dict with the fixed test set, the trained OLD model, the
    corrupted NEW training data, the shared scaler, and the shared initial
    weights for this (seed, sigma) condition. sigma only affects the NEW
    model's training data, never TEST, never OLD.
    """
    X, y = load_full_dataset()
    X_pool_full, y_pool_full, X_test, y_test, _, _ = make_fixed_test_split(X, y, seed=seed)
    X_pool, y_pool, X_val, y_val = make_val_split(X_pool_full, y_pool_full, seed=seed)

    X_old, y_old = subsample_pool(X_pool, y_pool, fraction=OLD_MODEL_FRACTION, seed=seed)
    X_new_clean, y_new = subsample_pool(X_pool, y_pool, fraction=1.0, seed=seed)
    X_new = corrupt_images(X_new_clean, sigma=sigma, seed=seed)

    scaler = fit_scaler(X_new_clean)  # scaler fit on clean new-pool features only
    X_old_s = scaler.transform(X_old)
    X_new_s = scaler.transform(X_new)
    X_test_s = scaler.transform(X_test)
    X_val_s = scaler.transform(X_val)

    rng = np.random.RandomState(seed)
    init_W = rng.normal(0, 0.01, size=(X_old_s.shape[1], 10))
    init_b = np.zeros(10)

    old_model = SoftmaxClassifier(
        n_features=X_old_s.shape[1], n_classes=10, lr=LR, n_epochs=N_EPOCHS,
        seed=seed, init_W=init_W, init_b=init_b,
    )
    old_model.fit(X_old_s, y_old)

    return dict(
        X_new_s=X_new_s, y_new=y_new, X_test_s=X_test_s, y_test=y_test, X_val_s=X_val_s, y_val=y_val,
        old_model=old_model, init_W=init_W, init_b=init_b, seed=seed, sigma=sigma,
    )


def train_naive(cond):
    m = SoftmaxClassifier(
        n_features=cond["X_new_s"].shape[1], n_classes=10, lr=LR, n_epochs=N_EPOCHS,
        seed=cond["seed"], init_W=cond["init_W"], init_b=cond["init_b"],
    )
    m.fit(cond["X_new_s"], cond["y_new"])
    return m


def train_pct(cond, lam: float):
    old_model = cond["old_model"]
    old_pred_on_new = old_model.predict(cond["X_new_s"])
    old_correct_mask = old_pred_on_new == cond["y_new"]

    m = SoftmaxClassifier(
        n_features=cond["X_new_s"].shape[1], n_classes=10, lr=LR, n_epochs=N_EPOCHS,
        seed=cond["seed"], init_W=cond["init_W"], init_b=cond["init_b"],
    )
    m.fit(
        cond["X_new_s"], cond["y_new"],
        old_model_correct_mask=old_correct_mask,
        old_model_pred=old_pred_on_new,
        pct_lambda=lam,
    )
    return m


def build_bcwi(cond, naive_model, alpha: float):
    return interpolate_weights(cond["old_model"], naive_model, alpha=alpha)


def evaluate(cond, model, label: str, split: str = "test"):
    """split='val' for hyperparameter selection; split='test' for reported results."""
    assert split in ("val", "test")
    X_test_s, y_test = cond[f"X_{split}_s"], cond[f"y_{split}"]
    pred_old = cond["old_model"].predict(X_test_s)
    pred_new = model.predict(X_test_s)
    return dict(
        label=label,
        seed=cond["seed"],
        sigma=cond["sigma"],
        accuracy=accuracy(y_test, pred_new),
        nfr=negative_flip_rate(y_test, pred_old, pred_new),
        pfr=positive_flip_rate(y_test, pred_old, pred_new),
        acc_old=accuracy(y_test, pred_old),
        split=split,
    )
