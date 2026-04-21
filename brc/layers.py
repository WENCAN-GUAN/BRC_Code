"""
BRC layers.

Wencan Guan <wencan.guan@uni-weimar.de>
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ---- Bayesian Linear (local reparameterization, Kingma 2015) ----

class BayesianLinear(nn.Module):
    """
    Variational linear layer: keeps mu/sigma for weights,
    outputs closed-form mean and variance (no MC sampling needed).
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # variational posterior q(w) = N(mu, sigma^2)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.register_buffer("prior_sigma", torch.tensor(prior_sigma))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (output_mean, output_variance) without weight sampling."""
        weight_sigma = self._softplus(self.weight_rho)
        bias_sigma = self._softplus(self.bias_rho)

        out_mean = F.linear(x, self.weight_mu, self.bias_mu)
        # Var[y_j] = sum_i sigma_Wji^2 * x_i^2 + sigma_bj^2
        out_var = F.linear(x.pow(2), weight_sigma.pow(2), bias_sigma.pow(2))
        return out_mean, out_var

    def kl_divergence(self) -> torch.Tensor:
        """KL(q || prior) for diagonal Gaussians."""
        weight_sigma = self._softplus(self.weight_rho)
        bias_sigma = self._softplus(self.bias_rho)
        prior_var = self.prior_sigma.pow(2)

        kl_w = 0.5 * (
            weight_sigma.pow(2) / prior_var
            + self.weight_mu.pow(2) / prior_var
            - 1.0
            - 2.0 * torch.log(weight_sigma / self.prior_sigma)
        ).sum()

        kl_b = 0.5 * (
            bias_sigma.pow(2) / prior_var
            + self.bias_mu.pow(2) / prior_var
            - 1.0
            - 2.0 * torch.log(bias_sigma / self.prior_sigma)
        ).sum()

        return kl_w + kl_b


# ---- Bayesian Multi-Head Attention (dot-product + RBF kernel) ----

class BayesianMultiHeadAttention(nn.Module):
    """
    Standard multi-head attention but with Bayesian projections.
    Scores = scaled dot-product + learnable RBF kernel on raw inputs.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = BayesianLinear(d_model, d_model)
        self.W_k = BayesianLinear(d_model, d_model)
        self.W_v = BayesianLinear(d_model, d_model)
        self.W_o = BayesianLinear(d_model, d_model)

        # per-head RBF hyperparams (log-space)
        self.log_sigma_f = nn.Parameter(torch.zeros(n_heads))
        self.log_length = nn.Parameter(torch.zeros(n_heads))

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self, H: torch.Tensor, x_raw: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = H.shape

        Q_mean, Q_var = self.W_q(H)
        K_mean, K_var = self.W_k(H)
        V_mean, V_var = self.W_v(H)

        def reshape(t):
            return t.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        Q = reshape(Q_mean)
        K = reshape(K_mean)
        V = reshape(V_mean)
        V_v = reshape(V_var)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # add RBF kernel term computed on original inputs
        if x_raw is not None:
            sigma_f2 = torch.exp(self.log_sigma_f * 2)
            length2 = torch.exp(self.log_length * 2)
            diff = x_raw.unsqueeze(2) - x_raw.unsqueeze(1)
            dist2 = diff.pow(2).sum(-1)
            for h in range(self.n_heads):
                scores[:, h] = scores[:, h] + sigma_f2[h] * torch.exp(-dist2 / (2.0 * length2[h]))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)

        # propagate V-variance through the attention weights
        var_through_attn = torch.matmul(attn.pow(2), V_v)
        var_through_attn = var_through_attn.transpose(1, 2).contiguous().view(B, N, self.d_model)

        out_mean, out_var_proj = self.W_o(context)
        out_var = out_var_proj + var_through_attn
        return out_mean, out_var

    def kl_divergence(self) -> torch.Tensor:
        return (self.W_q.kl_divergence() + self.W_k.kl_divergence()
                + self.W_v.kl_divergence() + self.W_o.kl_divergence())


# ---- Bayesian FFN (two layers, GELU, delta-method variance) ----

class BayesianFeedForward(nn.Module):
    """Two Bayesian linear layers with GELU; variance via first-order delta method."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = BayesianLinear(d_model, d_ff)
        self.linear2 = BayesianLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gelu_derivative(x: torch.Tensor) -> torch.Tensor:
        """Phi(z) + z * phi(z)"""
        phi = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * x.pow(2))
        return phi + x * pdf

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_mean, h_var = self.linear1(x)

        act_mean = F.gelu(h_mean)
        act_var = self._gelu_derivative(h_mean).pow(2) * h_var

        out_mean, out_var_direct = self.linear2(act_mean)

        # cross term: input variance * (mu_W^2 + sigma_W^2)
        W2_mu_sq = self.linear2.weight_mu.pow(2)
        W2_var = BayesianLinear._softplus(self.linear2.weight_rho).pow(2)
        var_cross = F.linear(act_var, W2_mu_sq + W2_var)
        total_var = out_var_direct + var_cross

        return self.dropout(out_mean), total_var

    def kl_divergence(self) -> torch.Tensor:
        return self.linear1.kl_divergence() + self.linear2.kl_divergence()


# ---- Bayesian Block (attention + FFN with pre-norm) ----

class BayesianBlock(nn.Module):
    """Attention sub-layer then FFN sub-layer, both with pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = BayesianMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = BayesianFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, H: torch.Tensor, x_raw: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(H)
        attn_out, attn_var = self.attention(normed, x_raw)
        H = H + attn_out

        normed = self.norm2(H)
        ffn_out, ffn_var = self.ffn(normed)
        H = H + ffn_out

        return H, attn_var + ffn_var

    def kl_divergence(self) -> torch.Tensor:
        return self.attention.kl_divergence() + self.ffn.kl_divergence()


# ---- Probabilistic Residual (Beta-distributed weights, Kumaraswamy sampling) ----

class KumaraswamySampler:
    """Differentiable approximation to Beta sampling."""

    @staticmethod
    def sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(a).clamp(1e-6, 1.0 - 1e-6)
        return 1.0 - (1.0 - u.pow(1.0 / b)).pow(1.0 / a)

    @staticmethod
    def mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a / (a + b)


class ProbabilisticResidualConnection(nn.Module):
    """
    H_out = alpha * H_in + beta * F(H_in)
    where alpha ~ Beta(g1, g2), beta ~ Beta(g3, g4).
    During training we sample via Kumaraswamy; at inference we use the posterior mean.
    """

    def __init__(self):
        super().__init__()
        self.log_gamma1 = nn.Parameter(torch.tensor(1.0))
        self.log_gamma2 = nn.Parameter(torch.tensor(1.0))
        self.log_gamma3 = nn.Parameter(torch.tensor(1.0))
        self.log_gamma4 = nn.Parameter(torch.tensor(1.0))

    def _gamma(self):
        return (F.softplus(self.log_gamma1), F.softplus(self.log_gamma2),
                F.softplus(self.log_gamma3), F.softplus(self.log_gamma4))

    @staticmethod
    def _beta_moments(a, b):
        mean = a / (a + b)
        var = a * b / ((a + b).pow(2) * (a + b + 1.0))
        return mean, var

    def forward(self, H_in, F_out, H_in_var, F_out_var):
        g1, g2, g3, g4 = self._gamma()

        if self.training:
            alpha = KumaraswamySampler.sample(g1, g2)
            beta = KumaraswamySampler.sample(g3, g4)
        else:
            alpha = g1 / (g1 + g2)
            beta = g3 / (g3 + g4)

        out_mean = alpha * H_in + beta * F_out

        # variance with all cross terms (Goodman 1960)
        E_a, V_a = self._beta_moments(g1, g2)
        E_b, V_b = self._beta_moments(g3, g4)

        out_var = (
            E_a.pow(2) * H_in_var
            + E_b.pow(2) * F_out_var
            + V_a * H_in.pow(2)
            + V_b * F_out.pow(2)
            + V_a * H_in_var
            + V_b * F_out_var
        )
        return out_mean, out_var

    def kl_divergence(self) -> torch.Tensor:
        """KL against uniform Beta(1,1) prior for both alpha and beta."""
        g1, g2, g3, g4 = self._gamma()
        return self._beta_kl(g1, g2) + self._beta_kl(g3, g4)

    @staticmethod
    def _beta_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """KL[ Beta(a,b) || Beta(1,1) ]"""
        return (
            -torch.lgamma(a) - torch.lgamma(b) + torch.lgamma(a + b)
            + (a - 1.0) * torch.digamma(a)
            + (b - 1.0) * torch.digamma(b)
            - (a + b - 2.0) * torch.digamma(a + b)
        )


# ---- Cascade Stage (iterative refinement until variance < threshold) ----

class CascadeStage(nn.Module):
    """
    Wraps a BayesianBlock + probabilistic residual.
    Runs the block up to T_max times; stops early (at inference)
    when the mean variance drops below the stage threshold.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 T_max: int = 3, dropout: float = 0.1):
        super().__init__()
        self.block = BayesianBlock(d_model, n_heads, d_ff, dropout)
        self.residual = ProbabilisticResidualConnection()
        self.T_max = T_max

    def forward(self, H, H_var, tau_k, x_raw=None):
        actual_t = self.T_max
        for t in range(1, self.T_max + 1):
            F_out, F_var = self.block(H, x_raw)
            H, H_var = self.residual(H, F_out, H_var, F_var)

            if not self.training and H_var.mean() < tau_k:
                actual_t = t
                break

        return H, H_var, actual_t

    def kl_divergence(self) -> torch.Tensor:
        return self.block.kl_divergence() + self.residual.kl_divergence()
