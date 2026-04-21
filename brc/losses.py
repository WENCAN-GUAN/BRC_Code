"""
BRC loss function.

Wencan Guan <wencan.guan@uni-weimar.de>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BRCLoss(nn.Module):
    """
    Standalone loss module for use with an external training loop.
    Combines heteroscedastic NLL, KL regularization, and cascade variance penalty.
    """

    def __init__(self, lambda_kl: float = 1e-3, lambda_cas: float = 1.0):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_cas = lambda_cas

    def forward(self, mu, sigma2, y, kl, stage_vars, thresholds, N):
        sigma2 = sigma2.clamp(min=1e-6)

        L_pred = 0.5 * (torch.log(sigma2) + (y - mu).pow(2) / sigma2).mean()
        L_kl = kl / N

        excess = F.relu(stage_vars - thresholds.unsqueeze(0))
        L_cas = excess.sum(dim=1).mean()

        loss = L_pred + self.lambda_kl * L_kl + self.lambda_cas * L_cas

        info = {
            "loss": loss.item(),
            "L_pred": L_pred.item(),
            "L_kl": L_kl.item(),
            "L_cas": L_cas.item(),
        }
        return loss, info
