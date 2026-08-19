#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Extract deterministic G256 weight-block corpora from safetensors fixtures.

Reads a corpus.json config, samples distinct 256-wide blocks from each named
rank-2 weight tensor, and writes little-endian f32 files plus train/heldout
concatenations and a manifest.tsv.

Usage:
  python3 extract_corpus.py path/to/corpus.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

BLOCK = 256
MANIFEST_HEADER = "name\tsplit\tmodel\ttensor\tshape\tblocks\tpath\tsha256\n"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def fixture_seed(global_seed: int, name: str) -> int:
    """First 8 LE bytes of SHA-256(f\"{global_seed}:{name}\") as uint64."""
    digest = hashlib.sha256(f"{global_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def resolve_shard(snapshot: Path, tensor: str) -> Path:
    """Locate the safetensors file holding `tensor` inside a HF snapshot dir."""
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map") or {}
        shard_name = weight_map.get(tensor)
        if shard_name is None:
            die(f"tensor {tensor!r} missing from {index_path}")
        shard = snapshot / shard_name
        if not shard.is_file():
            die(f"shard {shard} for tensor {tensor!r} does not exist")
        return shard

    candidates = sorted(snapshot.glob("model*.safetensors"))
    if len(candidates) == 0:
        die(f"no model*.safetensors in {snapshot}")
    if len(candidates) != 1:
        die(
            f"no model.safetensors.index.json and {len(candidates)} "
            f"model*.safetensors under {snapshot}; need exactly one"
        )
    return candidates[0]


def load_tensor_cpu(shard: Path, tensor: str) -> torch.Tensor:
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if tensor not in keys:
            die(f"tensor {tensor!r} not found in {shard}")
        return f.get_tensor(tensor)


def sample_block_indices(total_blocks: int, n_blocks: int, seed: int) -> list[int]:
    if n_blocks < 0:
        die(f"blocks must be non-negative, got {n_blocks}")
    if n_blocks > total_blocks:
        die(f"requested {n_blocks} blocks but tensor only has {total_blocks}")
    rng = random.Random(seed)
    indices = rng.sample(range(total_blocks), n_blocks)
    if len(set(indices)) != len(indices):
        die("duplicate sampled block indices")
    return indices


def gather_blocks_f32(weight: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Gather selected G256 blocks without converting the full tensor to f32.

    Weight is rank-2 [M, K] with K % 256 == 0. Block index is row-major:
      block_id = row * (K // 256) + group
    """
    if weight.ndim != 2:
        die(f"expected rank-2 weight, got shape {tuple(weight.shape)}")
    m, k = int(weight.shape[0]), int(weight.shape[1])
    if k % BLOCK != 0:
        die(f"K={k} is not divisible by {BLOCK}")
    groups = k // BLOCK
    total_blocks = m * groups
    if any(i < 0 or i >= total_blocks for i in indices):
        die("block index out of range")

    out = torch.empty((len(indices), BLOCK), dtype=torch.float32)
    for dst, block_id in enumerate(indices):
        row = block_id // groups
        group = block_id % groups
        col0 = group * BLOCK
        # Slice stays in source dtype; only the 256 values become f32.
        block = weight[row, col0 : col0 + BLOCK].to(dtype=torch.float32)
        if block.numel() != BLOCK:
            die(f"short block gather at index {block_id}")
        out[dst].copy_(block)

    if not torch.isfinite(out).all():
        die("non-finite values in gathered blocks")
    return out


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as tmp:
            n = tmp.write(data)
            if n != len(data):
                die(f"short write to {tmp_name}: wrote {n} of {len(data)} bytes")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_f32(path: Path, tensor: torch.Tensor) -> bytes:
    """Serialize contiguous little-endian f32 rows; return raw bytes."""
    if tensor.dtype != torch.float32:
        die(f"write_f32 expected float32, got {tensor.dtype}")
    contig = tensor.detach().cpu().contiguous()
    raw = contig.numpy().astype("<f4", copy=False).tobytes(order="C")
    expected = contig.numel() * 4
    if len(raw) != expected:
        die(f"short f32 serialize for {path}: {len(raw)} != {expected}")
    atomic_write_bytes(path, raw)
    return raw


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def concat_split(
    out_dir: Path,
    name: str,
    parts: list[tuple[str, bytes]],
) -> tuple[Path, bytes, str]:
    """Concatenate fixture payloads in config order; atomic write."""
    blob = b"".join(payload for _, payload in parts)
    path = out_dir / name
    atomic_write_bytes(path, blob)
    return path, blob, sha256_hex(blob)


def extract_fixture(
    global_seed: int,
    out_dir: Path,
    fixture: dict[str, Any],
) -> tuple[dict[str, str], bytes]:
    name = fixture["name"]
    split = fixture["split"]
    model = fixture["model"]
    snapshot = Path(fixture["snapshot"])
    tensor_name = fixture["tensor"]
    n_blocks = int(fixture["blocks"])

    if not snapshot.is_dir():
        die(f"snapshot dir missing for {name}: {snapshot}")

    shard = resolve_shard(snapshot, tensor_name)
    weight = load_tensor_cpu(shard, tensor_name)

    if weight.ndim != 2:
        die(f"{name}: expected rank-2, got {tuple(weight.shape)}")
    m, k = int(weight.shape[0]), int(weight.shape[1])
    if k % BLOCK != 0:
        die(f"{name}: K={k} not divisible by {BLOCK}")

    total_blocks = m * (k // BLOCK)
    seed = fixture_seed(global_seed, name)
    indices = sample_block_indices(total_blocks, n_blocks, seed)
    blocks = gather_blocks_f32(weight, indices)

    out_path = out_dir / f"{name}.f32"
    raw = write_f32(out_path, blocks)
    digest = sha256_hex(raw)

    row = {
        "name": name,
        "split": split,
        "model": model,
        "tensor": tensor_name,
        "shape": f"{m}x{k}",
        "blocks": str(n_blocks),
        "path": str(out_path),
        "sha256": digest,
    }
    return row, raw


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} path/to/corpus.json", file=sys.stderr)
        return 2

    config_path = Path(argv[1])
    if not config_path.is_file():
        die(f"config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    global_seed = int(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    fixtures = cfg["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        die("corpus.json fixtures must be a non-empty list")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    train_parts: list[tuple[str, bytes]] = []
    heldout_parts: list[tuple[str, bytes]] = []

    for fixture in fixtures:
        row, raw = extract_fixture(global_seed, out_dir, fixture)
        manifest_rows.append(row)
        split = row["split"]
        if split == "train":
            train_parts.append((row["name"], raw))
        elif split == "heldout":
            heldout_parts.append((row["name"], raw))
        else:
            die(f"unknown split {split!r} for fixture {row['name']}")
        print(
            f"[ok] {row['name']}: {row['blocks']} blocks "
            f"shape={row['shape']} sha256={row['sha256'][:16]}…",
            flush=True,
        )

    # Equal-fixture concats: every fixture in a split is appended in config order
    # (each fixture already carries the same configured block count within a split).
    if train_parts:
        t_path, t_blob, t_hash = concat_split(out_dir, "train_concat.f32", train_parts)
        print(
            f"[ok] train_concat.f32: {len(train_parts)} fixtures "
            f"{len(t_blob)} bytes sha256={t_hash[:16]}…",
            flush=True,
        )
        _ = t_path
    else:
        die("no train fixtures")

    if heldout_parts:
        h_path, h_blob, h_hash = concat_split(
            out_dir, "heldout_concat.f32", heldout_parts
        )
        print(
            f"[ok] heldout_concat.f32: {len(heldout_parts)} fixtures "
            f"{len(h_blob)} bytes sha256={h_hash[:16]}…",
            flush=True,
        )
        _ = h_path
    else:
        die("no heldout fixtures")

    # manifest covers per-fixture files only (concats are side products)
    lines = [MANIFEST_HEADER]
    for row in manifest_rows:
        lines.append(
            f"{row['name']}\t{row['split']}\t{row['model']}\t{row['tensor']}\t"
            f"{row['shape']}\t{row['blocks']}\t{row['path']}\t{row['sha256']}\n"
        )
    atomic_write_bytes(out_dir / "manifest.tsv", "".join(lines).encode("utf-8"))
    print(f"[ok] manifest.tsv: {len(manifest_rows)} rows → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
