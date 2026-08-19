#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Extract paired weight+importance G256 corpora from activation_corpus.json.

Reads activation_corpus.json, samples distinct 256-wide row-major blocks from
each named rank-2 weight, pairs each block with the matching imatrix
input-channel diagonal slice, and writes little-endian f32 files plus
train/heldout raw+importance concatenations and a manifest.tsv.

Imatrix `.in_sum2` parsing ports mq_kld_proxy.rs with strict bounds/type checks.
Block sampling uses the same SHA-256 seed mixing contract as extract_corpus.py.

Usage:
  python3 extract_activation_corpus.py path/to/activation_corpus.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

BLOCK = 256
# GGUF scalar value sizes (bytes) keyed by type id.
_GGUF_SCALAR_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
MANIFEST_HEADER = (
    "name\tsplit\tmodel\ttensor\tshape\tblocks\t"
    "raw_path\traw_sha256\timportance_path\timportance_sha256\n"
)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def fixture_seed(global_seed: int, name: str) -> int:
    """First 8 LE bytes of SHA-256(f\"{global_seed}:{name}\") as uint64."""
    digest = hashlib.sha256(f"{global_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)

def imatrix_key(tensor: str) -> str:
    """Map Hugging Face Qwen3.8 names to llama.cpp imatrix tensor names."""
    prefix = "model.language_model.layers."
    if not tensor.startswith(prefix):
        die(f"unsupported tensor name for imatrix mapping: {tensor!r}")
    rest = tensor[len(prefix) :]
    layer, sep, suffix = rest.partition(".")
    if not sep or not layer.isdigit():
        die(f"malformed layer tensor name: {tensor!r}")
    roles = {
        "linear_attn.out_proj.weight": "ssm_out.weight",
        "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
    }
    role = roles.get(suffix)
    if role is None:
        die(f"unsupported Qwen3.8 imatrix role: {tensor!r}")
    return f"blk.{layer}.{role}"



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
        block = weight[row, col0 : col0 + BLOCK].to(dtype=torch.float32)
        if block.numel() != BLOCK:
            die(f"short block gather at index {block_id}")
        out[dst].copy_(block)

    if not torch.isfinite(out).all():
        die("non-finite values in gathered weight blocks")
    return out


def gather_importance_f32(
    imatrix: torch.Tensor, k: int, indices: list[int]
) -> torch.Tensor:
    """Pair each sampled weight block with imatrix[g*256:(g+1)*256].

    g is the K-group of the row-major block id (same grouping as weights).
    """
    if imatrix.ndim != 1:
        die(f"imatrix must be 1-D length K, got shape {tuple(imatrix.shape)}")
    if int(imatrix.numel()) != k:
        die(f"imatrix length {int(imatrix.numel())} != K={k}")
    if k % BLOCK != 0:
        die(f"K={k} is not divisible by {BLOCK}")
    groups = k // BLOCK
    total_blocks_along_k = groups  # only need group from block_id

    out = torch.empty((len(indices), BLOCK), dtype=torch.float32)
    for dst, block_id in enumerate(indices):
        if block_id < 0:
            die(f"negative block index {block_id}")
        group = block_id % groups
        if group < 0 or group >= total_blocks_along_k:
            die(f"group {group} out of range for K={k}")
        col0 = group * BLOCK
        slice_ = imatrix[col0 : col0 + BLOCK]
        if slice_.numel() != BLOCK:
            die(f"short imatrix slice at group {group}")
        out[dst].copy_(slice_)

    if not torch.isfinite(out).all():
        die("non-finite values in gathered importance")
    if (out < 0).any():
        die("negative values in gathered importance")
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


class _Buf:
    """Bounds-checked little-endian reader over a bytes/bytearray/mmap."""

    __slots__ = ("data", "pos", "n")

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        self.data = memoryview(data)
        self.pos = 0
        self.n = len(self.data)

    def remaining(self) -> int:
        return self.n - self.pos

    def need(self, nbytes: int) -> None:
        if nbytes < 0 or self.pos + nbytes > self.n:
            die(
                f"gguf truncated: need {nbytes} bytes at pos={self.pos} "
                f"(file size {self.n})"
            )

    def skip(self, nbytes: int) -> None:
        self.need(nbytes)
        self.pos += nbytes

    def read(self, nbytes: int) -> bytes:
        self.need(nbytes)
        out = self.data[self.pos : self.pos + nbytes].tobytes()
        self.pos += nbytes
        return out

    def u32(self) -> int:
        self.need(4)
        v = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def u64(self) -> int:
        self.need(8)
        v = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return v

    def string(self) -> str:
        # GGUF string: u64 length + bytes (not NUL-terminated)
        nlen = self.u64()
        if nlen > self.remaining():
            die(f"gguf string length {nlen} exceeds remaining {self.remaining()}")
        raw = self.read(nlen)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as e:
            die(f"gguf string not utf-8 at pos={self.pos - nlen}: {e}")


def _skip_gguf_value(buf: _Buf, vtype: int) -> None:
    """Skip one GGUF metadata value; strict on unknown types and bounds."""
    if vtype == 8:  # STRING
        slen = buf.u64()
        if slen > buf.remaining():
            die(f"gguf STRING length {slen} exceeds remaining {buf.remaining()}")
        buf.skip(slen)
        return
    if vtype == 9:  # ARRAY
        at = buf.u32()
        alen = buf.u64()
        if alen > 10_000_000_000:
            die(f"gguf ARRAY length {alen} implausibly large")
        if at == 8:  # array of strings
            for _ in range(alen):
                slen = buf.u64()
                if slen > buf.remaining():
                    die(
                        f"gguf ARRAY[STRING] length {slen} exceeds "
                        f"remaining {buf.remaining()}"
                    )
                buf.skip(slen)
            return
        if at == 9:
            # Nested arrays — recurse element-wise.
            for _ in range(alen):
                _skip_gguf_value(buf, 9)
            return
        elem = _GGUF_SCALAR_SIZES.get(at)
        if elem is None:
            die(f"unsupported gguf array element type {at}")
        total = alen * elem
        if total > buf.remaining():
            die(
                f"gguf ARRAY type={at} len={alen} needs {total} bytes, "
                f"remaining {buf.remaining()}"
            )
        buf.skip(total)
        return
    size = _GGUF_SCALAR_SIZES.get(vtype)
    if size is None:
        die(f"unsupported gguf kv vtype {vtype}")
    buf.skip(size)


def load_imatrix_gguf(path: Path) -> dict[str, torch.Tensor]:
    """Port of mq_kld_proxy.rs load_imatrix_gguf with strict bounds/type checks.

    Returns map: logical tensor name (suffix `.in_sum2` stripped) → 1-D f32
    vector of length K. Only F32 (dtype==0) rank-1 `.in_sum2` tensors are kept.
    """
    if not path.is_file():
        die(f"imatrix not found: {path}")
    try:
        data = path.read_bytes()
    except OSError as e:
        die(f"imatrix read {path}: {e}")

    if len(data) < 24:
        die(f"imatrix too small ({len(data)} bytes): {path}")

    buf = _Buf(data)
    magic = buf.read(4)
    if magic != b"GGUF":
        die(f"imatrix magic {magic!r} != b'GGUF': {path}")

    ver = buf.u32()
    if ver not in (1, 2, 3):
        die(f"unsupported gguf version {ver} in {path}")
    n_tensors = buf.u64()
    n_kv = buf.u64()
    if n_tensors > 10_000_000 or n_kv > 10_000_000:
        die(f"gguf header counts implausible: n_tensors={n_tensors} n_kv={n_kv}")

    for _ in range(n_kv):
        _key = buf.string()
        vtype = buf.u32()
        _skip_gguf_value(buf, vtype)

    entries: list[tuple[str, list[int], int, int]] = []
    for _ in range(n_tensors):
        name = buf.string()
        ndims = buf.u32()
        if ndims > 4:
            die(f"gguf tensor {name!r} ndims={ndims} > 4")
        shape: list[int] = []
        for _d in range(ndims):
            shape.append(buf.u64())
        dtype = buf.u32()
        off = buf.u64()
        entries.append((name, shape, dtype, off))

    # Tensor data region is aligned to 32 bytes after the info block.
    base = (buf.pos + 31) // 32 * 32
    if base > len(data):
        die(f"gguf data base {base} past EOF {len(data)}")

    out: dict[str, torch.Tensor] = {}
    for name, shape, dtype, off in entries:
        if not name.endswith(".in_sum2"):
            continue
        # dtype 0 == F32 in GGUF; only rank-1 accepted (matches mq_kld_proxy).
        if dtype != 0:
            die(
                f"imatrix tensor {name!r} dtype={dtype} (want F32=0); "
                f"refusing non-f32 .in_sum2"
            )
        if len(shape) != 1:
            # Skip multi-matrix MoE layouts silently (same filter as mq_kld_proxy).
            continue
        k = shape[0]
        if k == 0:
            die(f"imatrix tensor {name!r} has K=0")
        start = base + off
        nbytes = k * 4
        if off > len(data) or start > len(data) or start + nbytes > len(data):
            die(
                f"imatrix tensor {name!r} out of bounds: base={base} off={off} "
                f"k={k} file_size={len(data)}"
            )
        # Validate alignment: f32 payload should be 4-byte aligned.
        if start % 4 != 0:
            die(f"imatrix tensor {name!r} misaligned start={start}")

        raw = data[start : start + nbytes]
        if len(raw) != nbytes:
            die(f"imatrix tensor {name!r} short read")
        vals = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        if vals.numel() != k:
            die(f"imatrix tensor {name!r}: got {vals.numel()} != k={k}")
        if not torch.isfinite(vals).all():
            die(f"imatrix tensor {name!r} contains non-finite values")
        if (vals < 0).any():
            die(f"imatrix tensor {name!r} contains negative values")

        key = name[: -len(".in_sum2")]
        if key in out:
            die(f"duplicate imatrix key {key!r}")
        out[key] = vals

    if not out:
        die(f"no F32 rank-1 .in_sum2 tensors found in {path}")
    return out


def extract_fixture(
    global_seed: int,
    model_dir: Path,
    model_label: str,
    imatrix_map: dict[str, torch.Tensor],
    out_dir: Path,
    fixture: dict[str, Any],
) -> tuple[dict[str, str], bytes, bytes]:
    name = fixture["name"]
    split = fixture["split"]
    tensor_name = fixture["tensor"]
    n_blocks = int(fixture["blocks"])

    if not model_dir.is_dir():
        die(f"model_dir missing for {name}: {model_dir}")

    shard = resolve_shard(model_dir, tensor_name)
    weight = load_tensor_cpu(shard, tensor_name)

    if weight.ndim != 2:
        die(f"{name}: expected rank-2, got {tuple(weight.shape)}")
    m, k = int(weight.shape[0]), int(weight.shape[1])
    if k % BLOCK != 0:
        die(f"{name}: K={k} not divisible by {BLOCK}")

    im_key = imatrix_key(tensor_name)
    im = imatrix_map.get(im_key)
    if im is None:
        die(
            f"{name}: no imatrix .in_sum2 entry {im_key!r} for tensor "
            f"{tensor_name!r} ({len(imatrix_map)} keys loaded)"
        )
    if int(im.numel()) != k:
        die(
            f"{name}: imatrix length {int(im.numel())} != K={k} "
            f"for tensor {tensor_name!r}"
        )

    total_blocks = m * (k // BLOCK)
    seed = fixture_seed(global_seed, name)
    indices = sample_block_indices(total_blocks, n_blocks, seed)
    raw_blocks = gather_blocks_f32(weight, indices)
    imp_blocks = gather_importance_f32(im, k, indices)

    if raw_blocks.shape != (n_blocks, BLOCK) or imp_blocks.shape != (n_blocks, BLOCK):
        die(
            f"{name}: shape mismatch raw={tuple(raw_blocks.shape)} "
            f"imp={tuple(imp_blocks.shape)} expected ({n_blocks},{BLOCK})"
        )

    raw_path = out_dir / f"{name}.raw.f32"
    imp_path = out_dir / f"{name}.importance.f32"
    raw_bytes = write_f32(raw_path, raw_blocks)
    imp_bytes = write_f32(imp_path, imp_blocks)

    # Paired files must be byte-aligned (same nblocks * 256 * 4).
    if len(raw_bytes) != len(imp_bytes):
        die(
            f"{name}: raw/importance byte length mismatch "
            f"{len(raw_bytes)} vs {len(imp_bytes)}"
        )
    expected_bytes = n_blocks * BLOCK * 4
    if len(raw_bytes) != expected_bytes:
        die(
            f"{name}: expected {expected_bytes} bytes per file, "
            f"got {len(raw_bytes)}"
        )

    row = {
        "name": name,
        "split": split,
        "model": model_label,
        "tensor": tensor_name,
        "shape": f"{m}x{k}",
        "blocks": str(n_blocks),
        "raw_path": str(raw_path),
        "raw_sha256": sha256_hex(raw_bytes),
        "importance_path": str(imp_path),
        "importance_sha256": sha256_hex(imp_bytes),
    }
    return row, raw_bytes, imp_bytes


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            f"usage: {argv[0]} path/to/activation_corpus.json",
            file=sys.stderr,
        )
        return 2

    config_path = Path(argv[1])
    if not config_path.is_file():
        die(f"config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    global_seed = int(cfg["seed"])
    model_dir = Path(cfg["model_dir"])
    imatrix_path = Path(cfg["imatrix"])
    out_dir = Path(cfg["output_dir"])
    fixtures = cfg["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        die("activation_corpus.json fixtures must be a non-empty list")

    model_label = model_dir.name or str(model_dir)

    print(f"loading imatrix: {imatrix_path}", flush=True)
    imatrix_map = load_imatrix_gguf(imatrix_path)
    print(f"imatrix: {len(imatrix_map)} .in_sum2 entries", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    train_raw: list[tuple[str, bytes]] = []
    train_imp: list[tuple[str, bytes]] = []
    heldout_raw: list[tuple[str, bytes]] = []
    heldout_imp: list[tuple[str, bytes]] = []

    for fixture in fixtures:
        row, raw_b, imp_b = extract_fixture(
            global_seed, model_dir, model_label, imatrix_map, out_dir, fixture
        )
        manifest_rows.append(row)
        split = row["split"]
        if split == "train":
            train_raw.append((row["name"], raw_b))
            train_imp.append((row["name"], imp_b))
        elif split == "heldout":
            heldout_raw.append((row["name"], raw_b))
            heldout_imp.append((row["name"], imp_b))
        else:
            die(f"unknown split {split!r} for fixture {row['name']}")
        print(
            f"[ok] {row['name']}: {row['blocks']} blocks shape={row['shape']} "
            f"raw={row['raw_sha256'][:16]}… imp={row['importance_sha256'][:16]}…",
            flush=True,
        )

    if not train_raw:
        die("no train fixtures")
    if not heldout_raw:
        die("no heldout fixtures")

    # Concatenate in config order; raw and importance stay paired by fixture order.
    for label, raw_parts, imp_parts in (
        ("train", train_raw, train_imp),
        ("heldout", heldout_raw, heldout_imp),
    ):
        r_path, r_blob, r_hash = concat_split(
            out_dir, f"{label}_concat.raw.f32", raw_parts
        )
        i_path, i_blob, i_hash = concat_split(
            out_dir, f"{label}_concat.importance.f32", imp_parts
        )
        if len(r_blob) != len(i_blob):
            die(
                f"{label} concat raw/importance length mismatch "
                f"{len(r_blob)} vs {len(i_blob)}"
            )
        print(
            f"[ok] {r_path.name}: {len(raw_parts)} fixtures "
            f"{len(r_blob)} bytes sha256={r_hash[:16]}…",
            flush=True,
        )
        print(
            f"[ok] {i_path.name}: {len(imp_parts)} fixtures "
            f"{len(i_blob)} bytes sha256={i_hash[:16]}…",
            flush=True,
        )

    lines = [MANIFEST_HEADER]
    for row in manifest_rows:
        lines.append(
            f"{row['name']}\t{row['split']}\t{row['model']}\t{row['tensor']}\t"
            f"{row['shape']}\t{row['blocks']}\t"
            f"{row['raw_path']}\t{row['raw_sha256']}\t"
            f"{row['importance_path']}\t{row['importance_sha256']}\n"
        )
    atomic_write_bytes(out_dir / "manifest.tsv", "".join(lines).encode("utf-8"))
    print(f"[ok] manifest.tsv: {len(manifest_rows)} rows → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
