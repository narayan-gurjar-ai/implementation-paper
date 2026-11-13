"""
Evaluation metrics. NFR/PFR definitions follow Bansal et al. (2019), as
formalized in Paper 1, Equation 2.
"""
from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean())


def negative_flip_rate(y_true, old_pred, new_pred) -> float:
    """
    NFR = |{x : old correct AND new incorrect}| / |D|
    """
    old_correct = old_pred == y_true
    new_incorrect = new_pred != y_true
    flips = old_correct & new_incorrect
    return float(flips.sum() / len(y_true))


def positive_flip_rate(y_true, old_pred, new_pred) -> float:
    """
    PFR = |{x : old incorrect AND new correct}| / |D|
    (reported for context; not a target of optimization in this study.)
    """
    old_incorrect = old_pred != y_true
    new_correct = new_pred == y_true
    flips = old_incorrect & new_correct
    return float(flips.sum() / len(y_true))


def negative_flip_indices(y_true, old_pred, new_pred):
    old_correct = old_pred == y_true
    new_incorrect = new_pred != y_true
    return np.where(old_correct & new_incorrect)[0]
