"""
Evaluation metrics: R2, PICP, PINAW, ECE, CRPS, AUSE.

Wencan Guan <wencan.guan@uni-weimar.de>
"""

import numpy as np
from typing import Optional


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def compute_picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability."""
    return ((y_true >= lower) & (y_true <= upper)).astype(float).mean()


def compute_pinaw(
    lower: np.ndarray, upper: np.ndarray,
    y_range: Optional[float] = None, y_true: Optional[np.ndarray] = None
) -> float:
    """Prediction Interval Normalized Average Width."""
    if y_range is None:
        if y_true is None:
            raise ValueError("need y_range or y_true")
        y_range = y_true.max() - y_true.min()
    return (upper - lower).mean() / (y_range + 1e-12)


def compute_ece(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 10
) -> float:
    """Confidence-binned Expected Calibration Error."""
    confidence = 1.0 / (sigma + 1e-8)
    errors = np.abs(y_true - mu)

    bin_edges = np.linspace(confidence.min(), confidence.max() + 1e-8, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        n_b = mask.sum()
        if n_b == 0:
            continue
        avg_conf = confidence[mask].mean()
        avg_acc = 1.0 / (errors[mask].mean() + 1e-8)
        ece += (n_b / N) * abs(avg_acc - avg_conf)
    return ece


def compute_ece_quantile(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 10
) -> float:
    """Quantile-based ECE: check if actual coverage matches the nominal level."""
    from scipy.stats import norm
    quantiles = np.linspace(0.05, 0.95, n_bins)
    ece = 0.0
    for q in quantiles:
        z = norm.ppf(0.5 + q / 2.0)
        actual = ((y_true >= mu - z * sigma) & (y_true <= mu + z * sigma)).mean()
        ece += abs(actual - q)
    return ece / n_bins


def compute_crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian predictive distribution."""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma
    return (sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))).mean()


def compute_ause(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 20
) -> float:
    """Area Under the Sparsification Error curve."""
    errors = (y_true - mu) ** 2
    order = np.argsort(-sigma)

    fractions = np.linspace(0, 1.0, n_bins + 1)
    N = len(y_true)

    def _build_curve(idx_order):
        curve = []
        for f in fractions:
            n_rm = int(f * N)
            if n_rm >= N:
                curve.append(0.0)
            else:
                curve.append(np.delete(errors, idx_order[:n_rm]).mean())
        return np.array(curve)

    se = _build_curve(order) - _build_curve(np.argsort(-errors))

    try:
        return np.trapezoid(se, fractions)
    except AttributeError:
        return np.trapz(se, fractions)
