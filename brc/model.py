"""
BRC main model.

Wencan Guan <wencan.guan@uni-weimar.de>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .layers import BayesianLinear, CascadeStage


class BRC(nn.Module):
    """
    Bayesian Residual Cascade.

    Pipeline: input projection -> K cascade stages -> cascade aggregation -> (mu, sigma^2)

    Each stage runs a shared-weight Bayesian block up to T_max times,
    stopping early when the running variance falls below a stage-specific threshold.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        K: int = 4,
        d_ff: int = 1024,
        T_max: int = 3,
        tau_base: float = 0.1,
        r: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.K = K
        self.T_max = T_max

        self.proj = nn.Linear(input_dim, d_model)

        self.stages = nn.ModuleList([
            CascadeStage(d_model, n_heads, d_ff, T_max, dropout)
            for _ in range(K)
        ])

        # threshold schedule: tau_k = tau_base * r^k  (stricter for deeper stages)
        thresholds = [tau_base * (r ** k) for k in range(K)]
        self.register_buffer("thresholds", torch.tensor(thresholds))

        self.w_out = nn.Linear(d_model, 1)   # point prediction
        self.w_ale = nn.Linear(d_model, 1)   # log aleatoric variance

    def forward(self, x: torch.Tensor, return_details: bool = False) -> Dict[str, torch.Tensor]:
        squeeze = False
        if x.dim() == 2:
            x_raw = x.unsqueeze(1)
            squeeze = True
        else:
            x_raw = x

        B, N, _ = x_raw.shape

        H = self.proj(x_raw)
        H_var = torch.zeros_like(H)

        stage_vars = []
        stage_iters = []

        for k, stage in enumerate(self.stages):
            H, H_var, t_k = stage(H, H_var, self.thresholds[k], x_raw)
            stage_vars.append(H_var.mean(dim=(1, 2)))
            stage_iters.append(t_k)

        # pool over sequence dim
        h_final = H.mean(dim=1) if N > 1 else H.squeeze(1)

        mu = self.w_out(h_final).squeeze(-1)

        log_sigma2_ale = self.w_ale(h_final).squeeze(-1)
        sigma2_ale = torch.exp(log_sigma2_ale)

        # epistemic part: softmax-weighted per-stage variances
        stage_var_stack = torch.stack(stage_vars, dim=1)
        weights = F.softmax(-stage_var_stack / self.thresholds.unsqueeze(0), dim=1)
        sigma2_epi = (weights * stage_var_stack).sum(dim=1)

        sigma2_total = sigma2_epi + sigma2_ale

        out = {
            "mu": mu,
            "sigma2_total": sigma2_total,
            "sigma2_epistemic": sigma2_epi,
            "sigma2_aleatoric": sigma2_ale,
        }
        if return_details:
            out["stage_vars"] = stage_var_stack
            out["stage_iters"] = stage_iters
            out["weights"] = weights

        return out

    def kl_divergence(self) -> torch.Tensor:
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for stage in self.stages:
            kl = kl + stage.kl_divergence()
        return kl

    def compute_loss(self, x, y, lambda_kl=1e-3, lambda_cas=1.0):
        """
        Composite ELBO: heteroscedastic NLL + KL + cascade variance penalty.
        Returns (loss, info_dict).
        """
        out = self.forward(x, return_details=True)
        mu = out["mu"]
        sigma2 = out["sigma2_total"].clamp(min=1e-6)

        # NLL
        L_pred = 0.5 * (torch.log(sigma2) + (y - mu).pow(2) / sigma2).mean()

        # KL
        N = x.shape[0]
        L_kl = self.kl_divergence() / N

        # cascade constraint: penalize stages that didn't meet their threshold
        stage_vars = out["stage_vars"]
        excess = F.relu(stage_vars - self.thresholds.unsqueeze(0))
        L_cas = excess.sum(dim=1).mean()

        loss = L_pred + lambda_kl * L_kl + lambda_cas * L_cas

        info = {
            "loss": loss.item(),
            "L_pred": L_pred.item(),
            "L_kl": L_kl.item(),
            "L_cas": L_cas.item(),
        }
        return loss, info

    def predict(self, x: torch.Tensor, confidence: float = 0.95) -> Dict[str, torch.Tensor]:
        """Run inference and return mu, sigma, and prediction interval bounds."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        mu = out["mu"]
        sigma = out["sigma2_total"].clamp(min=1e-8).sqrt()

        from scipy.stats import norm as _norm
        z = _norm.ppf(0.5 + confidence / 2.0)

        return {
            "mu": mu,
            "sigma": sigma,
            "lower": mu - z * sigma,
            "upper": mu + z * sigma,
        }
