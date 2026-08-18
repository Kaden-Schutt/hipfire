# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Minimal FWHT matching model.py rotate_activation (scale = n^-0.5)."""
from __future__ import annotations
import torch

def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """FWHT over the last dimension (must be power of two)."""
    n = x.size(-1)
    assert n > 0 and (n & (n - 1)) == 0, n
    y = x.reshape(-1, n).to(torch.float32).clone()
    h = 1
    while h < n:
        out = y.clone()
        i = 0
        while i < n:
            for j in range(i, i + h):
                u = y[:, j]
                v = y[:, j + h]
                out[:, j] = u + v
                out[:, j + h] = u - v
            i += 2 * h
        y = out
        h *= 2
    y = y.reshape(x.shape) * float(scale)
    return y.to(x.dtype)
