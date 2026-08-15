# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# Independent parsers for hipfire Hessian on-disk formats, written from the
# format contracts (docs/plans/gptq-hessian-format.md and the E8H1 header
# documented in parent/hessian.rs + collect_e8_hessian), NOT from the Rust
# reader source.
#
# HFHS v1 (full KxK per tensor):
#   Header 24 B LE:
#     magic[4]=b"HFHS" | version:u32=1 | n_tensors:u64 | reserved:u64=0
#   Record:
#     name_len:u32 | name:utf8 | expert_idx:u32 | K:u32 | dtype_flag:u32
#     payload: K*K floats row-major (flag 1=f32, 2=f64)
#
# E8H1 .hblk (block-diagonal 256):
#   Header 12 B LE:
#     magic:u32=0x45384831 ("E8H1") | n_blocks:u32 | K:u32
#   Payload: n_blocks * 256*256 f32 LE, row-major per block
#   K MUST equal n_blocks*256. Values are raw sum_t x_b x_b^T (NOT normalized).

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Optional

import numpy as np

HFHS_MAGIC = b"HFHS"
HFHS_VERSION = 1
HFHS_HEADER_SIZE = 24
HFHS_DTYPE_F32 = 1
HFHS_DTYPE_F64 = 2

E8H1_MAGIC = 0x45384831
E8H1_HEADER_SIZE = 12
HESSIAN_BLOCK = 256
HBLK_BLOCK_BYTES = HESSIAN_BLOCK * HESSIAN_BLOCK * 4


@dataclass
class HfhsTensor:
    name: str
    expert_idx: int
    k: int
    dtype_flag: int
    h: np.ndarray  # [K, K], float64 view


@dataclass
class HfhsFile:
    version: int
    reserved: int
    tensors: list[HfhsTensor]

    def get(self, name: str, expert_idx: int = 0) -> Optional[HfhsTensor]:
        for t in self.tensors:
            if t.name == name and t.expert_idx == expert_idx:
                return t
        return None


def write_hfhs(path: Path | str, tensors: Iterable[tuple[str, int, np.ndarray]], *,
               dtype: str = "f32") -> None:
    """Write HFHS v1. tensors: iterable of (name, expert_idx, H[K,K])."""
    path = Path(path)
    items = list(tensors)
    dtype_flag = HFHS_DTYPE_F32 if dtype == "f32" else HFHS_DTYPE_F64
    np_dtype = np.float32 if dtype == "f32" else np.float64
    with path.open("wb") as f:
        f.write(HFHS_MAGIC)
        f.write(struct.pack("<I", HFHS_VERSION))
        f.write(struct.pack("<Q", len(items)))
        f.write(struct.pack("<Q", 0))
        for name, expert_idx, h in items:
            h = np.asarray(h, dtype=np_dtype)
            if h.ndim != 2 or h.shape[0] != h.shape[1]:
                raise ValueError(f"{name}: H must be square, got {h.shape}")
            k = h.shape[0]
            name_b = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<I", int(expert_idx)))
            f.write(struct.pack("<I", int(k)))
            f.write(struct.pack("<I", dtype_flag))
            f.write(np.ascontiguousarray(h, dtype=np_dtype).tobytes(order="C"))


def read_hfhs(path: Path | str) -> HfhsFile:
    """Independent HFHS v1 reader (from format contract)."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < HFHS_HEADER_SIZE:
        raise ValueError(f"HFHS truncated header: {len(data)} < {HFHS_HEADER_SIZE}")
    magic = data[0:4]
    if magic != HFHS_MAGIC:
        raise ValueError(f"bad HFHS magic {magic!r}")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != HFHS_VERSION:
        raise ValueError(f"unsupported HFHS version {version}")
    n_tensors = struct.unpack_from("<Q", data, 8)[0]
    reserved = struct.unpack_from("<Q", data, 16)[0]
    pos = HFHS_HEADER_SIZE
    tensors: list[HfhsTensor] = []
    for _ in range(n_tensors):
        if pos + 4 > len(data):
            raise ValueError("HFHS truncated at name_len")
        name_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + name_len + 12 > len(data):
            raise ValueError("HFHS truncated at name/meta")
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        expert_idx, k, dtype_flag = struct.unpack_from("<III", data, pos)
        pos += 12
        if dtype_flag == HFHS_DTYPE_F32:
            nbytes = k * k * 4
            np_dtype = np.float32
        elif dtype_flag == HFHS_DTYPE_F64:
            nbytes = k * k * 8
            np_dtype = np.float64
        else:
            raise ValueError(f"unknown dtype_flag {dtype_flag}")
        if pos + nbytes > len(data):
            raise ValueError(f"HFHS truncated payload for {name}")
        payload = np.frombuffer(data[pos : pos + nbytes], dtype=np_dtype).reshape(k, k).astype(np.float64)
        pos += nbytes
        tensors.append(HfhsTensor(name=name, expert_idx=expert_idx, k=k, dtype_flag=dtype_flag, h=payload))
    return HfhsFile(version=version, reserved=reserved, tensors=tensors)


@dataclass
class E8H1File:
    n_blocks: int
    k: int
    blocks: np.ndarray  # [n_blocks, 256, 256] float64


def hessian_key(tensor_name: str) -> str:
    """Filesystem key: replace / \\ and .. — contract shared by all writers."""
    return tensor_name.replace("/", "_").replace("\\", "_").replace("..", "_")


def write_hblk(path: Path | str, k: int, blocks: np.ndarray) -> None:
    """Write one E8H1 .hblk. blocks: [n_blocks, 256, 256] or flat n_blocks*256*256."""
    path = Path(path)
    if k <= 0 or k % HESSIAN_BLOCK != 0:
        raise ValueError(f"K={k} must be positive multiple of {HESSIAN_BLOCK}")
    n_blocks = k // HESSIAN_BLOCK
    arr = np.asarray(blocks, dtype=np.float64)
    if arr.size != n_blocks * HESSIAN_BLOCK * HESSIAN_BLOCK:
        raise ValueError(f"blocks size {arr.size} != {n_blocks}*256*256")
    flat_f32 = arr.astype(np.float32).reshape(-1)
    with path.open("wb") as f:
        f.write(struct.pack("<I", E8H1_MAGIC))
        f.write(struct.pack("<I", n_blocks))
        f.write(struct.pack("<I", k))
        f.write(flat_f32.tobytes(order="C"))


def read_hblk(path: Path | str) -> E8H1File:
    """Independent E8H1 reader (from format contract)."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < E8H1_HEADER_SIZE:
        raise ValueError(f"E8H1 truncated header: {len(data)}")
    magic, n_blocks, k = struct.unpack_from("<III", data, 0)
    if magic != E8H1_MAGIC:
        raise ValueError(f"bad E8H1 magic 0x{magic:08x}")
    if k != n_blocks * HESSIAN_BLOCK:
        raise ValueError(f"K={k} disagrees with n_blocks={n_blocks}")
    want = E8H1_HEADER_SIZE + n_blocks * HBLK_BLOCK_BYTES
    if len(data) < want:
        raise ValueError(f"E8H1 truncated: {len(data)} < {want}")
    payload = np.frombuffer(data[E8H1_HEADER_SIZE:want], dtype=np.float32)
    blocks = payload.reshape(n_blocks, HESSIAN_BLOCK, HESSIAN_BLOCK).astype(np.float64)
    return E8H1File(n_blocks=n_blocks, k=k, blocks=blocks)


def accumulate_block_diagonal_xxT(activations: np.ndarray) -> np.ndarray:
    """E8H1 producer contract: per-256-block raw sum x_b x_b^T.

    activations: [N, K] with K % 256 == 0. Returns [n_blocks, 256, 256] float64.
    """
    x = np.asarray(activations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("activations must be [N, K]")
    n, k = x.shape
    if k % HESSIAN_BLOCK != 0:
        raise ValueError(f"K={k} not multiple of 256")
    n_blocks = k // HESSIAN_BLOCK
    out = np.zeros((n_blocks, HESSIAN_BLOCK, HESSIAN_BLOCK), dtype=np.float64)
    for b in range(n_blocks):
        xb = x[:, b * HESSIAN_BLOCK : (b + 1) * HESSIAN_BLOCK]  # [N, 256]
        out[b] = xb.T @ xb
    return out


__all__ = [
    "HFHS_MAGIC",
    "E8H1_MAGIC",
    "HESSIAN_BLOCK",
    "HfhsTensor",
    "HfhsFile",
    "E8H1File",
    "write_hfhs",
    "read_hfhs",
    "write_hblk",
    "read_hblk",
    "hessian_key",
    "accumulate_block_diagonal_xxT",
]
