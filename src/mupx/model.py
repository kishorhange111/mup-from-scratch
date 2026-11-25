"""A byte-level GPT in two parametrisations: standard (SP) and maximal update (μP).

Width is ``d_model = m · base_width``. Head dim stays at 64, so heads scale with width.
μP for Adam (Yang et al. 2022, Tensor Programs V, Table 3, "output multiplier" variant):

| tensor | init std | Adam lr | forward |
|---|---|---|---|
| embeddings (input-like) | σ | η | — |
| hidden matrices (q, k, v, out, MLP) | σ / √m | **η / m** | — |
| readout | **0** | η | logits **× 1/m** |
| attention scores | — | — | ``q·k / d_head`` (instead of ``/√d_head``) |
| LayerNorm gains | 1 | η | — |

The point: under SP, the per-step change of each hidden activation grows with width (Adam
updates are O(1) per entry and sum over fan-in), so the best lr shrinks as the model
grows. μP rescales so every layer's update is Θ(1) at every width. The best lr found on
a small model then transfers to a large one.

SP here is GPT-2's recipe: std 0.02 everywhere, one lr for everything, no logit multiplier,
``1/√d_head`` attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    width: int = 256
    base_width: int = 128
    depth: int = 4
    head_dim: int = 64
    vocab: int = 256
    ctx: int = 128
    mup: bool = True
    sigma: float = 0.02

    @property
    def m(self) -> float:
        return self.width / self.base_width


class Block(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        d = c.width
        self.h, self.hd = d // c.head_dim, c.head_dim
        self.scale = 1.0 / c.head_dim if c.mup else 1.0 / math.sqrt(c.head_dim)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, l, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        x = x + self.proj(y.transpose(1, 2).reshape(b, l, d))
        return x + self.out(F.gelu(self.fc(self.n2(x))))


class GPT(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.width)
        self.pos = nn.Embedding(c.ctx, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.depth))
        self.norm = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width, c.vocab, bias=False)
        self.reset_parameters()

    def hidden_matrices(self):
        return [
            p for b in self.blocks for p in (b.qkv.weight, b.proj.weight, b.fc.weight, b.out.weight)
        ]

    @torch.no_grad()
    def reset_parameters(self):
        c = self.c
        nn.init.normal_(self.emb.weight, std=c.sigma)
        nn.init.normal_(self.pos.weight, std=c.sigma)
        for b in self.blocks:
            for lin in (b.qkv, b.proj, b.fc, b.out):
                # μP: std ∝ 1/√fan_in relative to the base model; fc→out has 4× fan-in
                std = c.sigma / math.sqrt(c.m) if c.mup else c.sigma
                nn.init.normal_(lin.weight, std=std)
        if c.mup:
            nn.init.zeros_(self.head.weight)
        else:
            nn.init.normal_(self.head.weight, std=c.sigma)

    def forward(self, idx, return_acts: bool = False):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        acts = [x]
        for b in self.blocks:
            x = b(x)
            acts.append(x)
        logits = self.head(self.norm(x))
        if self.c.mup:
            logits = logits / self.c.m
        acts.append(logits)
        return (logits, acts) if return_acts else logits


def param_groups(model: GPT, lr: float, weight_decay: float = 0.0):
    """Adam parameter groups. Under μP, hidden matrices get lr / m; everything else gets lr."""
    hidden = model.hidden_matrices()
    ids = {id(p) for p in hidden}
    rest = [p for p in model.parameters() if id(p) not in ids]
    h_lr = lr / model.c.m if model.c.mup else lr
    return [
        {"params": hidden, "lr": h_lr, "weight_decay": weight_decay},
        {"params": rest, "lr": lr, "weight_decay": 0.0},
    ]
