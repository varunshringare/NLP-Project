
import numpy as np
from scipy.stats import spearmanr


def spearman_correlation(
    predictions: list[float],
    targets:     list[float],
) -> float:
    
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
    
    return {
        "spearman":            spearman_correlation(predictions, targets),
        "accuracy_within_stdev": accuracy_within_stdev(predictions, targets, stdevs),
    }
