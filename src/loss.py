"""Loss functions for quantile regression."""

from typing import List

import torch
from torch import nn


class SmoothedPinballLoss(nn.Module):
    """
    Differentiable (smoothed) pinball / quantile loss.

    Instead of the non-differentiable absolute value, uses a soft-plus
    relaxation controlled by `alpha`.
    """

    def __init__(self, taus: List[float], alpha: float = 1e-2):
        super().__init__()
        self.register_buffer("taus", torch.tensor(taus, dtype=torch.float32))
        self.alpha = float(alpha)

    def forward(self, pred, target):
        taus = self.taus.to(pred.dtype)
        target = target.unsqueeze(1)
        under = torch.clamp((target - pred) / self.alpha, -50, 50)
        over = torch.clamp((pred - target) / self.alpha, -50, 50)
        smooth_under = self.alpha * torch.log1p(torch.exp(under))
        smooth_over = self.alpha * torch.log1p(torch.exp(over))
        return (taus * smooth_under + (1 - taus) * smooth_over).mean()
