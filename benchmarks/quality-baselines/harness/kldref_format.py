"""Hipfire-internal KLD reference format — Python reader.

Format spec (β, see docs/plans/issue-113-quant-quality-eval.md
§"Hipfire-derived top-K format"):

    Header (32 bytes):
      bytes  0-7   magic "HFKLDR\\0\\0"  (8 ASCII chars, null-padded)
      bytes  8-11  version              (uint32, currently 1)
      bytes 12-15  n_ctx                (uint32)
      bytes 16-19  n_vocab              (uint32)
      bytes 20-23  n_chunk              (uint32)
      bytes 24-25  top_k                (uint16)
      bytes 26-27  flags                (uint16, currently 0)
      bytes 28-31  reserved             (uint32, zero)

    Tokens:
      n_ctx × n_chunk × uint32 token IDs

    Per-chunk × per-scored-token (n_ctx − 1 − n_ctx/2 tokens per chunk):
      uint32 top_indices[top_k]    (vocab IDs, descending log-prob)
      fp32   top_log_probs[top_k]  (log P(i), descending)
      fp32   sum_p_residual        (sum P(i) for i NOT in top-K, in [0, 1])
      fp32   reserved_pad          (zero, for 8-byte alignment)

Per-token block size: 8 + 8 * top_k bytes. At top_k=256: 2056 B/token.

Reconstruction at consumer:
  log P(i) = top_log_probs[j] if i == top_indices[j] for some j
  Σ_{i not in top-K} P(i) = sum_p_residual

Why log-probs (β) rather than raw logits + max_logit + log_sum_exp (α)?
KLD math operates on log-probs directly. llama.cpp's
--kl-divergence-base format encodes log-probs already (the per-token
'min_log_prob' + 'scale' encoding reconstructs to log-probs, not raw
logits — see tools/perplexity/perplexity.cpp:222-225 on commit
9dcf83552). Storing log-probs in the hipfire format avoids unnecessary
machinery to back out max_logit / log_sum_exp from llama.cpp's
encoding.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MAGIC = b"HFKLDR\x00\x00"
VERSION = 1


@dataclass
class KldRefHeader:
    version: int
    n_ctx: int
    n_vocab: int
    n_chunk: int
    top_k: int
    flags: int

    @property
    def per_token_bytes(self) -> int:
        # top_k*4 (indices) + top_k*4 (log_probs) + 4 (sum_p_residual) + 4 (pad)
        return 8 + self.top_k * 8

    @property
    def scored_per_chunk(self) -> int:
        return self.n_ctx - 1 - self.n_ctx // 2

    @property
    def total_scored_tokens(self) -> int:
        return self.scored_per_chunk * self.n_chunk


@dataclass
class TokenBlock:
    """Per-token reference distribution (top-K log-probs + residual prob mass)."""
    top_indices: list[int]      # len == top_k
    top_log_probs: list[float]  # len == top_k, descending log P(i)
    sum_p_residual: float       # Σ P(i) for i NOT in top-K, in [0, 1]


def read_header(f) -> KldRefHeader:
    raw = f.read(32)
    if len(raw) != 32:
        raise ValueError(f"short read on header: got {len(raw)}, want 32")
    magic = raw[:8]
    if magic != MAGIC:
        raise ValueError(f"bad magic: got {magic!r}, want {MAGIC!r}")
    version, n_ctx, n_vocab, n_chunk = struct.unpack("<IIII", raw[8:24])
    top_k, flags = struct.unpack("<HH", raw[24:28])
    if version != VERSION:
        raise ValueError(f"unsupported version {version}, this reader supports {VERSION}")
    return KldRefHeader(
        version=version, n_ctx=n_ctx, n_vocab=n_vocab,
        n_chunk=n_chunk, top_k=top_k, flags=flags,
    )


def write_header(f, header: KldRefHeader) -> None:
    f.write(MAGIC)
    f.write(struct.pack("<IIII", header.version, header.n_ctx, header.n_vocab, header.n_chunk))
    f.write(struct.pack("<HHI", header.top_k, header.flags, 0))


def read_tokens(f, header: KldRefHeader) -> list[int]:
    n = header.n_ctx * header.n_chunk
    raw = f.read(n * 4)
    if len(raw) != n * 4:
        raise ValueError(f"short read on tokens: got {len(raw)}, want {n*4}")
    return list(struct.unpack(f"<{n}I", raw))


def read_block(f, header: KldRefHeader) -> TokenBlock:
    raw = f.read(header.per_token_bytes)
    if len(raw) != header.per_token_bytes:
        raise ValueError(f"short read on block: got {len(raw)}, want {header.per_token_bytes}")
    off = 0
    top_indices = list(struct.unpack_from(f"<{header.top_k}I", raw, off))
    off += header.top_k * 4
    top_log_probs = list(struct.unpack_from(f"<{header.top_k}f", raw, off))
    off += header.top_k * 4
    (sum_p_residual,) = struct.unpack_from("<f", raw, off); off += 4
    # last 4 bytes are reserved_pad — ignored
    return TokenBlock(
        top_indices=top_indices, top_log_probs=top_log_probs,
        sum_p_residual=sum_p_residual,
    )


def write_block(f, header: KldRefHeader, block: TokenBlock) -> None:
    if len(block.top_indices) != header.top_k:
        raise ValueError(f"top_indices length {len(block.top_indices)} != top_k {header.top_k}")
    if len(block.top_log_probs) != header.top_k:
        raise ValueError(f"top_log_probs length {len(block.top_log_probs)} != top_k {header.top_k}")
    f.write(struct.pack(f"<{header.top_k}I", *block.top_indices))
    f.write(struct.pack(f"<{header.top_k}f", *block.top_log_probs))
    f.write(struct.pack("<ff", block.sum_p_residual, 0.0))


def iter_blocks(f, header: KldRefHeader) -> Iterator[TokenBlock]:
    for _ in range(header.total_scored_tokens):
        yield read_block(f, header)


def open_ref(path: str | Path) -> tuple[KldRefHeader, list[int], Iterator[TokenBlock]]:
    """Open a hipfire KLD reference file. Returns (header, tokens, block_iter).

    The block_iter is consumed in order (one pass) — keep the file open while
    iterating; the function does not buffer all blocks in memory.
    """
    f = open(path, "rb")
    header = read_header(f)
    tokens = read_tokens(f, header)
    return header, tokens, iter_blocks(f, header)


# ---------- Per-sequence-KLD result format (small sidecar) ----------

# After eval_hipfire.rs / eval_gguf.rs run a candidate against a ref, they
# emit a small "per-sequence-KLD" file that kld_reduce.py aggregates.
#
# Layout (v1, deprecated but still readable):
#   bytes  0-7   magic "HFKSEQ\0\0"
#   bytes  8-11  version (uint32, 1)
#   bytes 12-15  n_chunk (uint32)
#   bytes 16-19  reserved (uint32, zero)
#   bytes 20-?   n_chunk × {fp64 mean_kld_seq, fp64 mean_p99_seq}
#                  (16 B per sequence)
#
# Layout (v2, current):
#   bytes  0-7   magic "HFKSEQ\0\0"
#   bytes  8-11  version (uint32, 2)
#   bytes 12-15  n_chunk (uint32)
#   bytes 16-19  reserved (uint32, zero)
#   bytes 20-?   n_chunk × {fp64 mean_kld_seq, fp64 mean_p99_seq, fp64 mean_nll_seq}
#                  (24 B per sequence)
#
# v2 adds mean_nll per sequence, enabling the PPL column in the result table
# (PPL = exp(mean_nll)). Reading v1 returns mean_nll = NaN per chunk, which
# the reducer renders as `—` in the PPL column.

SEQKLD_MAGIC = b"HFKSEQ\x00\x00"
SEQKLD_VERSION = 2


def write_per_seq_kld(
    path: str | Path,
    mean_kld_per_seq: list[float],
    p99_kld_per_seq: list[float],
    mean_nll_per_seq: list[float] | None = None,
) -> None:
    """Write HFKSEQ (v2 if mean_nll_per_seq provided, else v1)."""
    n_chunk = len(mean_kld_per_seq)
    if len(p99_kld_per_seq) != n_chunk:
        raise ValueError("mean and p99 sequences must have same length")
    if mean_nll_per_seq is not None and len(mean_nll_per_seq) != n_chunk:
        raise ValueError("mean and nll sequences must have same length")

    version = 2 if mean_nll_per_seq is not None else 1
    with open(path, "wb") as f:
        f.write(SEQKLD_MAGIC)
        f.write(struct.pack("<III", version, n_chunk, 0))
        if version == 2:
            for m, p, n in zip(mean_kld_per_seq, p99_kld_per_seq, mean_nll_per_seq):
                f.write(struct.pack("<ddd", m, p, n))
        else:
            for m, p in zip(mean_kld_per_seq, p99_kld_per_seq):
                f.write(struct.pack("<dd", m, p))


def read_per_seq_kld(
    path: str | Path,
) -> tuple[list[float], list[float], list[float]]:
    """Read HFKSEQ (v1 or v2). Returns (mean_klds, p99_klds, mean_nlls).
    For v1 inputs, mean_nlls is filled with NaN."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != SEQKLD_MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        version, n_chunk, _reserved = struct.unpack("<III", f.read(12))
        if version not in (1, 2):
            raise ValueError(f"unsupported HFKSEQ version {version}")
        means, p99s, nlls = [], [], []
        if version == 2:
            for _ in range(n_chunk):
                m, p, n = struct.unpack("<ddd", f.read(24))
                means.append(m)
                p99s.append(p)
                nlls.append(n)
        else:  # v1
            import math
            for _ in range(n_chunk):
                m, p = struct.unpack("<dd", f.read(16))
                means.append(m)
                p99s.append(p)
                nlls.append(math.nan)
        return means, p99s, nlls
