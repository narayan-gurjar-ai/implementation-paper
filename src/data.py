"""
Data loading and splitting utilities.

Dataset: sklearn.datasets.load_digits (Optical Recognition of Handwritten
Digits, 1797 samples, 8x8 grayscale, 10 classes). Bundled with scikit-learn,
loads fully offline.

Design (frozen in research_decisions.md, Section 6):
  - A single, stratified, seeded 80/20 split produces TRAIN_POOL and a fixed
    TEST set. TEST is never touched by any training step in this project.
  - The "old" model is trained on a fixed 25% stratified subsample of
    TRAIN_POOL (per seed).
  - The "new" model(s) are trained on a larger stratified subsample of
    TRAIN_POOL, whose size (the "update fraction") is the independent
    variable in the magnitude sweep.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

TEST_FRACTION = 0.20
OLD_MODEL_FRACTION = 0.25  # fraction of TRAIN_POOL used for the old model


def load_full_dataset():
    """Load the full digits dataset. Returns (X, y)."""
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)
    return X, y


def make_fixed_test_split(X, y, seed: int):
    """
    Stratified 80/20 split. Returns (X_pool, y_pool, X_test, y_test,
    test_idx) where test_idx are the original row indices assigned to TEST,
    so other scripts can assert the same test set is reused.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=seed
    )
    idx = np.arange(len(y))
    pool_idx, test_idx = next(splitter.split(X, y))
    return (
        X[pool_idx], y[pool_idx],
        X[test_idx], y[test_idx],
        pool_idx, test_idx,
    )


VAL_FRACTION = 0.20  # fraction of the (non-test) pool held out for hyperparameter selection


def make_val_split(X_pool, y_pool, seed: int):
    """Stratified split of the non-test pool into (train_pool, validation).
    Validation is used ONLY for selecting PCT lambda / BCWI alpha; the
    final test set is never used for any selection."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=seed + 7)
    tr, va = next(splitter.split(X_pool, y_pool))
    return X_pool[tr], y_pool[tr], X_pool[va], y_pool[va]


def subsample_pool(X_pool, y_pool, fraction: float, seed: int):
    """
    Stratified subsample of the training pool at the given fraction.
    fraction=1.0 returns the full pool (order-shuffled, deterministic).
    """
    if fraction >= 1.0:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(y_pool))
        return X_pool[order], y_pool[order]
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=fraction, random_state=seed
    )
    idx = np.arange(len(y_pool))
    sub_idx, _ = next(splitter.split(X_pool, y_pool))
    return X_pool[sub_idx], y_pool[sub_idx]


def fit_scaler(X_train):
    """Fit a StandardScaler on TRAIN data only (never on TEST) to avoid leakage."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def one_hot(y, n_classes: int = 10):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


PIXEL_MAX = 16.0  # sklearn digits pixel intensity range is [0, 16]


def corrupt_images(X, sigma: float, seed: int, fraction: float = 1.0):
    """
    "Update magnitude" proxy: additive Gaussian pixel noise applied to a
    stratified-random `fraction` of the rows of X, clipped back to the
    valid pixel range. sigma=0 returns X unchanged. Used ONLY on the new
    model's training data; the fixed TEST set is never corrupted. See
    research_decisions.md, Section 9, for the justification and the pilot
    that led to this design.
    """
    if sigma <= 0:
        return X.copy()
    rng = np.random.RandomState(seed)
    Xc = X.copy()
    n = len(X)
    n_corrupt = int(round(fraction * n))
    idx = rng.choice(n, size=n_corrupt, replace=False)
    noise = rng.normal(0, sigma, size=(n_corrupt, X.shape[1]))
    Xc[idx] = np.clip(Xc[idx] + noise, 0, PIXEL_MAX)
    return Xc
