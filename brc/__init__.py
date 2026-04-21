"""
BRC - Bayesian Residual Cascade

Wencan Guan <wencan.guan@uni-weimar.de>
"""

from .model import BRC
from .layers import (
    BayesianLinear,
    BayesianMultiHeadAttention,
    BayesianFeedForward,
    BayesianBlock,
    ProbabilisticResidualConnection,
    CascadeStage,
)
from .losses import BRCLoss
from .metrics import compute_r2, compute_picp, compute_pinaw, compute_ece, compute_crps

__version__ = "1.0.0"
__all__ = [
    "BRC",
    "BRCLoss",
    "BayesianLinear",
    "BayesianMultiHeadAttention",
    "BayesianFeedForward",
    "BayesianBlock",
    "ProbabilisticResidualConnection",
    "CascadeStage",
    "compute_r2",
    "compute_picp",
    "compute_pinaw",
    "compute_ece",
    "compute_crps",
]
