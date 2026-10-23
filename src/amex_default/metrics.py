"""AMEX ranking metric and sampled-population probability diagnostics."""

import numpy as np


def amex_components(y_true, y_score):
    y = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(y_score, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or not len(y):
        raise ValueError("Use two nonempty one-dimensional arrays of equal length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("AMEX needs binary labels and finite scores")
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AMEX needs both classes")
    ranked = y[np.argsort(-score, kind="stable")]
    weights = np.where(ranked == 0, 20.0, 1.0)
    cumulative = np.cumsum(weights)
    cutoff = int(0.04 * weights.sum())
    capture = ranked[cumulative <= cutoff].sum() / positives
    weighted_position = cumulative / weights.sum()
    lorenz = np.cumsum(ranked) / positives
    gini = np.sum((lorenz - weighted_position) * weights)
    perfect_gini = 10 * negatives * (1 - 19 / (positives + 20 * negatives))
    normalized_gini = gini / perfect_gini
    return {"amex": float((normalized_gini + capture) / 2),
            "normalized_gini": float(normalized_gini), "capture_at_4pct": float(capture)}


def amex_metric(y_true, y_score):
    return amex_components(y_true, y_score)["amex"]


def lightgbm_amex(y_true, y_score):
    return "amex", amex_metric(y_true, y_score), True


def sigmoid(raw):
    raw = np.asarray(raw, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -raw))


def evaluate(y_true, probability):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    result = amex_components(y_true, probability)
    result.update(roc_auc=float(roc_auc_score(y_true, probability)),
                  average_precision=float(average_precision_score(y_true, probability)),
                  log_loss=float(log_loss(y_true, probability, labels=[0, 1])),
                  brier=float(brier_score_loss(y_true, probability)),
                  mean_prediction=float(np.mean(probability)),
                  observed_default_rate=float(np.mean(y_true)))
    return result
