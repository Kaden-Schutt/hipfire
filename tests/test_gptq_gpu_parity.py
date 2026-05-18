#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def unpack_mq4_blocks(payload: bytes) -> np.ndarray:
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 136)
    scale = blocks[:, 0:4].copy().view("<f4").reshape(-1)
    zero = blocks[:, 4:8].copy().view("<f4").reshape(-1)
    packed = blocks[:, 8:]
    q = np.empty((blocks.shape[0], 256), dtype=np.uint8)
    q[:, 0::2] = packed & np.uint8(0x0F)
    q[:, 1::2] = (packed >> np.uint8(4)) & np.uint8(0x0F)
    return q.astype(np.float32) * scale[:, None] + zero[:, None]


def main() -> int:
    try:
        import torch
    except ImportError:
        print("skip: torch not available")
        return 0

    from mq4_masked_calib import quantize_mq4_gptq, quantize_mq4_gptq_torch

    rng = np.random.default_rng(42)
    shape = [64, 256]
    damp = 0.01
    w = rng.standard_normal(shape, dtype=np.float32)
    x = rng.standard_normal((512, 256), dtype=np.float32)
    h = (x.T @ x).astype(np.float32)
    h += np.eye(256, dtype=np.float32) * np.float32(damp)
    hessian = h.reshape(1, 256, 256)

    packed_cpu, _ = quantize_mq4_gptq(
        w,
        hessian=hessian,
        shape=shape,
        damp=damp,
        refit_iters=2,
    )
    packed_torch, _ = quantize_mq4_gptq_torch(
        w,
        hessian=hessian,
        shape=shape,
        damp=damp,
        refit_iters=2,
        device=torch.device("cpu"),
    )

    q_cpu = unpack_mq4_blocks(packed_cpu)
    q_torch = unpack_mq4_blocks(packed_torch)
    rel = float(np.linalg.norm(q_cpu - q_torch) / max(float(np.linalg.norm(q_cpu)), 1.0e-12))
    print(f"relative_l2={rel:.8g}")
    assert rel < 1.0e-4, f"relative L2 mismatch too high: {rel:.8g}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
