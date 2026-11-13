"""
A from-scratch NumPy multinomial logistic regression (softmax classifier),
trained by mini-batch gradient descent.

Implemented from scratch (rather than sklearn.linear_model.LogisticRegression)
for two reasons that are load-bearing for this study, not just style:
  1. Positive-congruent training (Yan et al., 2021) needs a custom loss term
     added to the standard cross-entropy objective; sklearn's solver does not
     expose a hook for this.
  2. Backward-compatible weight interpolation (Schumann et al., 2024) needs
     direct access to, and control over, the raw weight matrix/bias vector,
     and both the "old" and "new" models must be initialized from the SAME
     initial weights for interpolation to be a meaningful operation (this is
     a known simplifying condition in the weight-interpolation literature;
     see research_decisions.md, Section 8).
"""
from __future__ import annotations

import numpy as np


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """
    Linear softmax classifier: logits = X @ W + b.

    Supports an optional positive-congruent training (PCT) penalty:
        L = CE(y, softmax(X W + b))
            + lam * CE_consistency(old_model_pred, softmax(X W + b))
    where the consistency term is applied only to the subset of training
    instances the OLD model classified correctly (Yan et al. 2021's
    "focal distillation" idea, restricted to the old model's own
    successes rather than its full soft-label distribution, which is the
    simplest faithful reading of "preserve the correct predictions of its
    predecessor").
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 10,
        lr: float = 0.5,
        n_epochs: int = 300,
        batch_size: int = 64,
        l2: float = 1e-4,
        seed: int = 0,
        init_W: np.ndarray | None = None,
        init_b: np.ndarray | None = None,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed

        rng = np.random.RandomState(seed)
        if init_W is not None:
            self.W = init_W.copy()
            self.b = init_b.copy()
        else:
            self.W = rng.normal(0, 0.01, size=(n_features, n_classes))
            self.b = np.zeros(n_classes)

    # -- core API -----------------------------------------------------
    def get_weights(self):
        return self.W.copy(), self.b.copy()

    def set_weights(self, W, b):
        self.W = W.copy()
        self.b = b.copy()

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # -- training -------------------------------------------------------
    def fit(
        self,
        X,
        y,
        old_model_correct_mask: np.ndarray | None = None,
        old_model_pred: np.ndarray | None = None,
        pct_lambda: float = 0.0,
        verbose: bool = False,
    ):
        """
        Train by mini-batch gradient descent.

        If pct_lambda > 0, `old_model_correct_mask` (bool array over the
        TRAIN rows, True where the old model predicted correctly) and
        `old_model_pred` (the old model's predicted class per TRAIN row)
        must both be supplied; the PCT penalty is added only on rows where
        the mask is True, pushing the new model's softmax output toward the
        old model's (correct) predicted class on those rows.
        """
        n = len(y)
        rng = np.random.RandomState(self.seed + 1000)
        Y_onehot = np.zeros((n, self.n_classes))
        Y_onehot[np.arange(n), y] = 1.0

        if pct_lambda > 0:
            assert old_model_correct_mask is not None and old_model_pred is not None
            pct_target = np.zeros((n, self.n_classes))
            pct_target[np.arange(n), old_model_pred] = 1.0
            pct_mask = old_model_correct_mask.astype(np.float64).reshape(-1, 1)
        else:
            pct_target = None
            pct_mask = None

        n_batches = max(1, n // self.batch_size)
        history = []
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for bidx in np.array_split(order, n_batches):
                Xb, Yb = X[bidx], Y_onehot[bidx]
                logits = Xb @ self.W + self.b
                P = softmax(logits)

                # standard cross-entropy gradient
                grad_logits = (P - Yb) / len(bidx)

                if pct_lambda > 0:
                    Tb = pct_target[bidx]
                    Mb = pct_mask[bidx]
                    grad_logits_pct = Mb * (P - Tb) / max(1.0, Mb.sum())
                    grad_logits = grad_logits + pct_lambda * grad_logits_pct

                grad_W = Xb.T @ grad_logits + self.l2 * self.W
                grad_b = grad_logits.sum(axis=0)

                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

                eps = 1e-12
                ce = -(Yb * np.log(P + eps)).sum(axis=1).mean()
                epoch_loss += ce
            history.append(epoch_loss / n_batches)
            if verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"  epoch {epoch:4d}  mean CE loss = {history[-1]:.4f}")
        return history


def interpolate_weights(model_old: SoftmaxClassifier, model_new: SoftmaxClassifier, alpha: float):
    """
    Backward-compatible weight interpolation (BCWI), Schumann et al. (2024)-
    style: W_interp = alpha * W_old + (1 - alpha) * W_new (and likewise for
    the bias). alpha=0 recovers the new model; alpha=1 recovers the old
    model. Requires both models to share the same shape (enforced by
    construction: both are initialized from the same init_W/init_b in this
    project's scripts).
    """
    Wo, bo = model_old.get_weights()
    Wn, bn = model_new.get_weights()
    assert Wo.shape == Wn.shape, "BCWI requires matching parameterizations"
    W_interp = alpha * Wo + (1 - alpha) * Wn
    b_interp = alpha * bo + (1 - alpha) * bn
    merged = SoftmaxClassifier(
        n_features=model_new.n_features,
        n_classes=model_new.n_classes,
        init_W=W_interp,
        init_b=b_interp,
        seed=model_new.seed,
    )
    return merged
