"""
Official evaluation metrics for the AmbiStory plausibility prediction task.

Two metrics are computed:

1. **Spearman Correlation**
   Measures the rank-order correlation between predicted scores and average
   human ratings.  A higher value indicates better agreement with annotator
   consensus.

2. **Accuracy Within Standard Deviation (Acc±σ)**
   A prediction is considered correct if it falls within one standard
   deviation of the average human rating.  The floor of the tolerance is 1.0
   so that samples with full annotator consensus still have a non-zero
   acceptance window.
"""

import numpy as np
from scipy.stats import spearmanr


def spearman_correlation(
    predictions: list[float],
    targets:     list[float],
) -> float:
    """
    Compute Spearman rank correlation between predictions and targets.

    Parameters
    ----------
    predictions : list[float]
        Model-predicted plausibility scores.
    targets : list[float]
        Average human plausibility ratings.

    Returns
    -------
    float
        Spearman ρ in [-1, 1].  Returns 0.0 if correlation is undefined
        (e.g. all predictions are identical).
    """
    correlation, _ = spearmanr(predictions, targets)
    # spearmanr returns NaN when variance is zero; treat that as 0
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def accuracy_within_stdev(
    predictions: list[float],
    targets:     list[float],
    stdevs:      list[float],
) -> float:
    """
    Compute the proportion of predictions that fall within one standard
    deviation of the average human rating.

    The tolerance for sample *i* is ``max(stdev_i, 1.0)`` so that even
    samples with full annotator agreement (stdev = 0) have a tolerance of 1.

    Parameters
    ----------
    predictions : list[float]
        Model-predicted plausibility scores.
    targets : list[float]
        Average human plausibility ratings.
    stdevs : list[float]
        Per-sample standard deviations of human ratings.

    Returns
    -------
    float
        Proportion of correct predictions in [0, 1].
    """
    predictions = np.array(predictions, dtype=float)
    targets     = np.array(targets,     dtype=float)
    stdevs      = np.array(stdevs,      dtype=float)

    # Enforce minimum tolerance of 1.0 (as specified in the task description)
    tolerances = np.maximum(stdevs, 1.0)

    correct = np.abs(predictions - targets) <= tolerances
    return float(correct.mean())


def evaluate(
    predictions: list[float],
    targets:     list[float],
    stdevs:      list[float],
) -> dict[str, float]:
    """
    Compute both official metrics at once and return them in a dict.

    Parameters
    ----------
    predictions : list[float]
        Model-predicted plausibility scores.
    targets : list[float]
        Average human plausibility ratings.
    stdevs : list[float]
        Per-sample standard deviations of human ratings.

    Returns
    -------
    dict with keys ``"spearman"`` and ``"accuracy_within_stdev"``.
    """
    return {
        "spearman":            spearman_correlation(predictions, targets),
        "accuracy_within_stdev": accuracy_within_stdev(predictions, targets, stdevs),
    }
