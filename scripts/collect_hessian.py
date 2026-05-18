#!/usr/bin/env python3
"""collect_hessian.py — calibration-time Hessian collector for Stage B GPTQ.

Phase A Stage B Phase 1.1 per `docs/plans/gptq.md` v2.

`crates/hipfire-runtime/examples/imatrix_collect.rs` is a llama.cpp subprocess
wrapper (no native forward pass), so we use HF transformers + ROCm PyTorch here
for the calibration forward pass. The output Hessian binary is consumed by
`crates/hipfire-quantize/src/gptq.rs` (Phase 2) at quantize time.

Loads a BF16 model, runs a forward pass on calibration tokens, registers
`nn.Linear` forward hooks that accumulate per-tensor outer products
`H_t = (1/N) * sum_t x_t · x_t^T`, writes to a sidecar binary file.

Calibration subset matches the GPTQ paper's scale: 128 sequences × 2048
tokens = 262144 calibration tokens. The Hessian is a smoothed expectation
that converges well at this scale; tokenizer disagreement with hipfire's
tokenizer is averaged out (per `docs/plans/gptq_plan_rev_synthesis.md`
Topic 6).

Tensor coverage: all `nn.Linear` modules whose name matches the GPTQ
whitelist (mirrors the quantizer's `awq_eligible` plus the non-AWQ MQ4
projections we want to GPTQ-quantize: o_proj/out_proj/down_proj).

Output file format (HFHS — Hipfire Hessian Sidecar, v1):

  Header (24 bytes):
    [0..4)    magic     u8[4]  = b"HFHS"
    [4..8)    version   u32_le = 1
    [8..16)   n_tensors u64_le
    [16..24)  reserved  u64_le = 0

  Per-tensor record (variable length):
    [0..4)              name_len    u32_le
    [4..4+name_len)     name        utf8 bytes
    [4+name_len..+4)    expert_idx  u32_le   (default 0; reserved for MoE)
    [+4..+4)            K           u32_le
    [+4..+4)            dtype_flag  u32_le   (1=F32, 2=F64)
    [+4..+(K*K*sz))     payload     K*K * sizeof(dtype) bytes (row-major,
                                                                native endian)

Hessian values are float32 by default. The quantizer promotes to FP64 for
Cholesky.

Usage:
    .venv/bin/python3 scripts/collect_hessian.py \\
        --model    <hf-model-id-or-path> \\
        --output   <path-to-out.hessian.bin> \\
        [--n-sequences 128] \\
        [--ctx-len    2048] \\
        [--corpus     <hf-dataset-id-or-text-file>] \\
        [--device     cuda] \\
        [--dtype      bfloat16]

Examples:
    # Qwen3.5-9B, 128 sequences × 2048 ctx from wikitext-2 train
    .venv/bin/python3 scripts/collect_hessian.py \\
        --model /data/cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202.../ \\
        --output ~/.hipfire/refs/qwen3.5-9b-bf16.hessian.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Tensor name patterns that should get GPTQ Hessians. Matches hipfire's
# AWQ whitelist plus the non-AWQ MQ4 projections (o_proj/out_proj/down_proj).
# Pattern check is done on the FULL module name (e.g.
# "model.layers.0.self_attn.q_proj"); we match against the last 2-3 components.
GPTQ_TARGET_SUFFIXES = (
    # Attention input projections
    "q_proj", "k_proj", "v_proj", "qkv_proj",
    # Attention output (no AWQ but yes GPTQ)
    "o_proj",
    # MLP
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    # Linear-attention (Gated DeltaNet)
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
    "out_proj",
    # MoE router
    "gate",
)


def is_gptq_target(name: str) -> bool:
    """Returns True if this module name should have a Hessian collected."""
    last = name.rsplit(".", 1)[-1]
    if last in GPTQ_TARGET_SUFFIXES:
        return True
    # `mlp.gate.weight` (MoE router) — matched by trailing ".gate" — but we
    # need to disambiguate from `gate_proj`. The check above on the literal
    # last segment handles both.
    return False


def _safetensors_keys(model_path: str) -> set[str]:
    """Returns the set of all tensor keys stored in the safetensors files,
    stripped of `.weight` suffix to match against module names.

    Used to translate HF's flattened in-memory module names (e.g.
    `model.layers.0.q_proj` for Qwen3.5 ForCausalLM) to the canonical
    safetensors / .hfq naming (`model.language_model.layers.0.q_proj`).
    """
    import json
    p = Path(model_path)
    keys: set[str] = set()
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        with idx.open() as f:
            data = json.load(f)
        keys = set(data["weight_map"].keys())
    else:
        # Single-file safetensors path
        from safetensors import safe_open  # type: ignore[import-not-found]
        for st_file in p.glob("*.safetensors"):
            with safe_open(st_file, framework="pt") as f:
                keys.update(f.keys())
    # Strip `.weight` so the keys align with the module-name basis (modules
    # without `.weight` such as norms/biases are not GPTQ targets anyway).
    return {k.removesuffix(".weight") for k in keys if k.endswith(".weight")}


def _translate_to_stored_name(mod_name: str, stored_keys: set[str]) -> str | None:
    """Find the canonical safetensors-stored name for a hooked module.

    Strategy: longest-suffix match. If `model.layers.0.q_proj` is the
    in-memory name, look for any stored key ending in
    `.layers.0.q_proj` — and pick the longest match (typically
    `model.language_model.layers.0.q_proj` for Qwen3.5 ConditionalGen).

    Returns the stored name (without `.weight`) on success, or None if no
    suffix-matching key exists in the safetensors (which would mean the
    module is in-memory only and not stored — unusual; skip with a warning).
    """
    # Strip the leading "model." (common prefix in both naming conventions).
    # We match on the trailing portion after the FIRST "." — typically
    # ".layers.N.<...>" — to be robust against the language_model. insertion.
    if mod_name in stored_keys:
        return mod_name
    # Build a candidate suffix by stripping a single leading component
    # ("model.") — most common HF wrapper pattern.
    trail = mod_name.split(".", 1)[1] if "." in mod_name else mod_name
    # Find any stored key whose suffix matches.
    matches = [k for k in stored_keys if k.endswith(trail)]
    if not matches:
        return None
    # If multiple, pick the longest (most fully-qualified — e.g. prefers
    # `model.language_model.layers.0.q_proj` over a hypothetical bare
    # `layers.0.q_proj`).
    matches.sort(key=len, reverse=True)
    return matches[0]


class HessianAccumulator:
    """One per nn.Linear we hook. Accumulates H = sum_t x_t · x_t^T (FP32, K×K).

    We accumulate in FP32 on the CPU side to keep GPU memory free for the
    forward pass. The transfer per token is K*4 bytes — for K=4096 that's
    16 KB per linear per token, ~5 MB per layer per token — fits in PCIe
    bandwidth.

    Promoted to FP64 only at the quantizer for Cholesky (per plan v2 §2).
    """

    def __init__(self, name: str, K: int) -> None:
        self.name = name
        self.K = K
        # FP32 accumulator. K=12288 → 576 MB. For 27B's largest tensor we'd
        # need ~600 MB host RAM per Hessian; manageable.
        self.H = np.zeros((K, K), dtype=np.float32)
        self.n_tokens = 0

    def update(self, x_2d: np.ndarray) -> None:
        """x_2d: shape [num_tokens, K] in FP32 on CPU.

        We use BLAS gemm via numpy: H += x.T @ x. Single-precision; the
        rounding error per accumulation is ~1 ULP per entry per token —
        across 262k tokens, accumulated error stays well below the
        damping λ that GPTQ adds anyway.
        """
        assert x_2d.ndim == 2 and x_2d.shape[1] == self.K, \
            f"{self.name}: shape mismatch {x_2d.shape} vs K={self.K}"
        # numpy's gemm path: x.T @ x is the standard outer-product
        # accumulation. For x shape [N, K], x.T @ x gives [K, K].
        self.H += x_2d.T @ x_2d
        self.n_tokens += x_2d.shape[0]

    def finalize(self) -> np.ndarray:
        """Returns H / n_tokens (the actual expectation E[x x^T])."""
        if self.n_tokens == 0:
            return self.H  # uninitialized — caller must skip
        return self.H / self.n_tokens


def build_hook(acc: HessianAccumulator):
    """Returns a forward-pre hook that captures the input activations.

    Hooks are registered on `nn.Linear`; the input is `x` (the matmul
    input, shape [..., K]). We flatten leading dims and move to CPU FP32
    before accumulating.
    """

    def hook(module, inputs):  # noqa: ARG001 — module is unused but required by API
        x = inputs[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        # Cast to FP32 on the GPU then transfer — avoids a host-side
        # cast and minimizes PCIe bytes (BF16→FP32 is 2x more bytes,
        # but the .cpu() transfer is the bottleneck either way).
        x_cpu = x.to(dtype=torch.float32).detach().cpu().numpy()
        acc.update(x_cpu)

    return hook


def write_hessian_file(out_path: Path, accs: dict[str, HessianAccumulator]) -> None:
    """Serialize all accumulated Hessians to HFHS v1 binary format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        # Header
        f.write(b"HFHS")
        f.write(struct.pack("<I", 1))                  # version
        f.write(struct.pack("<Q", len(accs)))          # n_tensors
        f.write(struct.pack("<Q", 0))                  # reserved

        for name, acc in accs.items():
            if acc.n_tokens == 0:
                print(f"  WARN: {name} accumulated 0 tokens — skipping", file=sys.stderr)
                continue
            H_final = acc.finalize().astype(np.float32, copy=False)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))   # name_len
            f.write(name_bytes)                           # name
            f.write(struct.pack("<I", 0))                 # expert_idx (default 0)
            f.write(struct.pack("<I", acc.K))             # K
            f.write(struct.pack("<I", 1))                 # dtype_flag (1 = F32)
            f.write(H_final.tobytes(order="C"))           # K*K*4 bytes


def load_calibration_text(corpus: str, n_sequences: int, ctx_len: int,
                          tokenizer) -> list[torch.Tensor]:
    """Returns a list of n_sequences token tensors, each of length ctx_len.

    `corpus` is either a HF datasets identifier (e.g. "wikitext") or a
    path to a plain-text file. For HF datasets we use the "train" split
    by default (avoids data leakage against wikitext-2-test which is the
    eval slice).
    """
    if Path(corpus).is_file():
        text = Path(corpus).read_text()
        all_tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    else:
        # HF datasets path. Default: wikitext-2-raw-v1 train split.
        if "/" in corpus or corpus == "wikitext":
            ds_name = corpus if "/" in corpus else "wikitext"
            cfg = "wikitext-2-raw-v1" if ds_name == "wikitext" else None
            ds = load_dataset(ds_name, cfg, split="train", trust_remote_code=False)
        else:
            ds = load_dataset(corpus, split="train", trust_remote_code=False)
        # Concatenate text rows; the dataset's "text" field is conventional.
        text_field = "text" if "text" in ds.column_names else ds.column_names[0]
        text = "\n\n".join(row[text_field] for row in ds if row[text_field].strip())
        all_tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    n_total_tokens = all_tokens.shape[0]
    needed_tokens = n_sequences * ctx_len
    if n_total_tokens < needed_tokens:
        raise SystemExit(
            f"corpus has only {n_total_tokens} tokens, need {needed_tokens} "
            f"({n_sequences} seqs × {ctx_len} ctx)"
        )

    seqs = [all_tokens[i * ctx_len: (i + 1) * ctx_len] for i in range(n_sequences)]
    return seqs


def main():
    ap = argparse.ArgumentParser(description="Collect per-tensor Hessians for Stage B GPTQ.")
    ap.add_argument("--model", required=True,
                    help="HF model identifier or local path to BF16 model dir.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output Hessian sidecar (.hessian.bin).")
    ap.add_argument("--n-sequences", type=int, default=128,
                    help="Calibration sequences (default 128, matches GPTQ paper).")
    ap.add_argument("--ctx-len", type=int, default=2048,
                    help="Tokens per calibration sequence (default 2048).")
    ap.add_argument("--corpus", default="wikitext",
                    help="HF dataset ID or path to plain-text file "
                         "(default 'wikitext' = wikitext-2-raw-v1 train).")
    ap.add_argument("--device", default="cuda",
                    help="Device for the forward pass (default 'cuda').")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model dtype (default bfloat16).")
    args = ap.parse_args()

    print(f"=== Hessian collector — Stage B Phase 1.1 ===")
    print(f"  model:        {args.model}")
    print(f"  output:       {args.output}")
    print(f"  corpus:       {args.corpus}")
    print(f"  sequences:    {args.n_sequences} × {args.ctx_len} ctx "
          f"= {args.n_sequences * args.ctx_len} calibration tokens")
    print(f"  device:       {args.device}")
    print(f"  dtype:        {args.dtype}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"\n[1/4] Loading tokenizer + model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    # device_map="auto" → HF/accelerate uses meta tensors, skipping the
    # random-init step that would otherwise allocate the full model in
    # GPU memory before safetensors weights replace them. Critical for
    # 9B on 20 GB VRAM where the init buffer alone exceeds capacity.
    # Offloads layers that don't fit to CPU automatically.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()
    print(f"      loaded in {time.time() - t0:.1f}s")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"      VRAM in use: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f}/"
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\n[2/4] Registering Hessian hooks on GPTQ-target Linear modules...")
    # HF's AutoModelForCausalLM flattens multimodal submodules (e.g. Qwen3.5's
    # `language_model.` is dropped from in-memory module names when loaded as
    # a CausalLM). Hipfire's .hfq, however, mirrors the safetensors naming
    # which keeps the full prefix. We translate in-memory names → stored
    # names via safetensors-index matching so the Rust quantizer can look up
    # Hessians by the same key as the .hfq tensors.
    stored_names = _safetensors_keys(args.model)
    accs: dict[str, HessianAccumulator] = {}
    handles = []
    name_remap_count = 0
    for mod_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and is_gptq_target(mod_name):
            K = module.in_features
            stored = _translate_to_stored_name(mod_name, stored_names)
            if stored is None:
                print(f"  WARN: no safetensors key matches {mod_name!r} — skipping",
                      file=sys.stderr)
                continue
            if stored != mod_name:
                name_remap_count += 1
            accs[stored] = HessianAccumulator(stored, K)
            handles.append(module.register_forward_pre_hook(build_hook(accs[stored])))
    print(f"      {name_remap_count} of {len(accs)} names remapped from in-memory "
          f"→ stored (multimodal-flatten translation)")
    print(f"      registered {len(accs)} hooks "
          f"({len(set(K_ for K_ in [acc.K for acc in accs.values()]))} distinct K dims)")
    print(f"      K range: {min(acc.K for acc in accs.values())}..{max(acc.K for acc in accs.values())}")

    print(f"\n[3/4] Loading calibration corpus + running forward pass...")
    t0 = time.time()
    seqs = load_calibration_text(args.corpus, args.n_sequences, args.ctx_len, tokenizer)
    print(f"      loaded {len(seqs)} calibration sequences in {time.time() - t0:.1f}s")
    print(f"      starting forward pass (no_grad, eval mode)...")

    t0 = time.time()
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            seq = seq.unsqueeze(0).to(args.device)
            model(seq)
            if (i + 1) % 8 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(seqs) - i - 1) / rate
                print(f"      seq {i+1}/{len(seqs)} "
                      f"({elapsed:.1f}s elapsed, {rate:.2f} seq/s, ETA {eta:.0f}s)")

    print(f"      forward pass complete: {time.time() - t0:.1f}s")
    print(f"      total tokens accumulated per tensor (sample): "
          f"{next(iter(accs.values())).n_tokens}")

    # Detach hooks before writing — they'll keep accumulating otherwise.
    for h in handles:
        h.remove()

    print(f"\n[4/4] Writing Hessian sidecar to {args.output}...")
    t0 = time.time()
    write_hessian_file(args.output, accs)
    print(f"      wrote {args.output.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.1f}s")
    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
