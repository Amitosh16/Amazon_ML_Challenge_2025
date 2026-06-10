"""Model definitions: QuantileHead and QuantileModel wrapper."""

import math

import torch
from torch import nn


class QuantileHead(nn.Module):
    """
    Custom head that predicts k monotonically increasing quantiles.

    Architecture:
        Linear(hidden_size -> 512) -> LayerNorm -> GELU -> Dropout -> Linear(512 -> k)

    The raw outputs are transformed so that:
        q[0] = raw[0]
        delta[i] = softplus(raw[i])   for i > 0
        Q = cumsum([q[0], delta[1], ..., delta[k-1]])
    """

    def __init__(self, hidden_size: int, k: int = 200, target_mean: float = 4.0):
        super().__init__()
        self.k = k
        self.target_mean = target_mean

        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, k),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    if m == self.net[-1]:
                        bias_init = torch.zeros(k)
                        bias_init[0] = 0.0
                        # Auto-scale spread ~3× target_mean, clamped to [4, 10]
                        spread_target = min(max(target_mean * 3.0, 4.0), 10.0)
                        delta_target = spread_target / (k - 1)
                        bias_init[1:] = math.log(delta_target + 1e-6)
                        m.bias.data = bias_init
                    else:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        raw = self.net(x)
        raw = torch.clamp(raw, -10, 10)
        first_q = raw[:, 0:1]
        deltas = torch.nn.functional.softplus(raw[:, 1:])
        deltas = torch.clamp(deltas, max=5.0)
        z = torch.cat([first_q, deltas], dim=1)
        return torch.cumsum(z, dim=1)


class QuantileModel(nn.Module):
    """Wrapper that combines a base causal LM with the QuantileHead."""

    def __init__(self, base_model, head):
        super().__init__()
        self.base = base_model
        self.head = head

    def forward(self, ids, mask):
        out = self.base(ids, attention_mask=mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        # Pool using the last real token (not padding)
        seq_lens = mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(ids.size(0), device=ids.device), seq_lens]
        return self.head(pooled)
