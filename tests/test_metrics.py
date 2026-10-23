import numpy as np
import pytest

from amex_default.metrics import amex_components, amex_metric, sigmoid


def reference_metric(y, scores):
    """Independent weighted Lorenz calculation, normalized by perfect ordering."""
    def gini(labels):
        weights = np.where(labels == 0, 20.0, 1.0)
        fraction = weights.cumsum() / weights.sum()
        return np.sum((labels.cumsum() / labels.sum() - fraction) * weights)
    ranked = y[np.argsort(-scores, kind="stable")]
    weights = np.where(ranked == 0, 20, 1)
    captured = ranked[weights.cumsum() <= int(weights.sum() * 0.04)].sum() / y.sum()
    return (gini(ranked) / gini(np.sort(y)[::-1]) + captured) / 2


def test_metric_matches_reference_and_perfect_order():
    rng = np.random.default_rng(42)
    for n in [50, 501, 2000]:
        y = rng.binomial(1, 0.26, size=n)
        scores = rng.normal(size=n)
        assert amex_metric(y, scores) == pytest.approx(reference_metric(y, scores), abs=1e-12)
        assert amex_metric(y, sigmoid(scores)) == pytest.approx(amex_metric(y, scores))
    y = np.array([1] * 50 + [0] * 450)
    assert amex_metric(y, y.astype(float)) == pytest.approx(1)


def test_weighted_cutoff_is_not_four_percent_of_rows():
    y = np.array([1] * 10 + [0] * 90)
    result = amex_components(y, np.arange(100, 0, -1))
    assert result["capture_at_4pct"] == 1.0


@pytest.mark.parametrize("y,p", [([1, 1], [0.2, 0.3]), ([0, 1], [0.2, np.nan]),
                                  ([0, 2], [0.1, 0.2]), ([], [])])
def test_metric_validation(y, p):
    with pytest.raises(ValueError):
        amex_metric(y, p)
