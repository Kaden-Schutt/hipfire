# Qwen3.5 Engine-Drift Hidden-State Diagnostics

This directory preserves the reusable part of the 2026-05-12 engine-drift
investigation tooling from PR #244. The old PR also carried several
diagnostic-only runtime env gates and FP64 probe kernels. This salvage branch
intentionally keeps the first landed surface narrower:

- a hipfire hidden-state dump example,
- an HF Transformers hidden-state oracle,
- offline hidden-state comparators,
- a cross-engine KLD sanity script,
- the investigation notes needed to reproduce the workflow.

Normal runtime behavior is unchanged.

## Background

The quality pipeline measures KLD between hipfire logits and a BF16 reference
stored in hipfire beta `.kldref` format. On Qwen3.5 models, near-lossless Q8
weights still showed a large engine-vs-reference floor, which meant quant
candidate comparisons needed an engine-control row before their absolute KLD
numbers could be interpreted.

The investigation localized a major source of this floor to the Qwen3.5 RoPE
pairing convention. HF uses half-split rotary pairs, while hipfire had been
using the interleaved pairing for Qwen3.5 without a matching Q/K weight
permutation. That root cause is already fixed in current master; the legacy
path is retained behind `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`.

The hidden-state dump tools are still useful for future drift work because they
let us compare hipfire's post-layer residual stream against an HF oracle on the
same `.kldref` chunk.

## Shipped Tooling

### Hipfire Dump

`crates/hipfire-runtime/examples/dump_qwen35_hidden_states.rs`

This example reads chunk `N` from a `.kldref`, runs Qwen3.5 token-by-token
through `qwen35::forward_scratch_with_hidden`, and writes an `HFHS` binary:

```text
magic       8 bytes: HFHS\0\0\0\0
n_layers    u32
n_pos       u32
hidden_dim  u32
reserved    u32
body        n_layers * [n_pos, hidden_dim] f32 row-major
```

It reuses the existing `HiddenStateRingBuffer` from spec-decode infrastructure
and overrides `extract_layers` to capture every layer.

### HF Oracle

`scripts/dump_hf_hidden_states.py`

This script runs the same `.kldref` chunk through HF Transformers and emits the
same `HFHS` binary format. It uses a pre-hook on the final model norm so the
last captured state matches hipfire's pre-final-norm capture point.

### Comparators

`scripts/compare_hidden_states.py`

Computes per-layer relative L2 and cosine between two `HFHS` dumps.

`scripts/compare_layer_positions.py`

Buckets one layer's drift by position, which helps distinguish steady input
bias from recurrent or long-context accumulation.

`scripts/cross_engine_check.py`

Checks whether the BF16 `.kldref` distribution and an HF Transformers run agree
on the same token chunk. Use this before blaming hipfire if the external
reference itself may have drifted.

## Historical Findings

These findings are preserved from the original investigation and have already
been folded into `docs/plans/qwen35-mq4-quality-gap.md`.

- The pre-fix Qwen3.5 RoPE path used interleaved rotary pairs against
  half-split HF weight layout. This produced a large Qwen3.5-specific
  engine-drift floor.
- The current default half-split path removed the major floor component.
- KV-cache quantization was not the primary layer-local divergence source.
- The old PR used additional diagnostic-only probes to rule out lm-head Q8 vs
  F16 precision, Qwen3.5 QK-norm accumulation precision, DeltaNet recurrent
  state precision, and DeltaNet loader convention mismatches. Those probes are
  not part of this first salvage PR.

## Reproducer

Build the hipfire dump example:

```bash
cargo build --release -p hipfire-runtime \
  --example dump_qwen35_hidden_states \
  --features deltanet
```

Dump hipfire hidden states from a local model and reference:

```bash
HIPFIRE_KV_MODE=q8 ./target/release/examples/dump_qwen35_hidden_states \
  --model <path-to-qwen35-0.8b-q8f16.hfq> \
  --ref <path-to-qwen3.5-0.8b-bf16.kldref.bin> \
  --chunk 0 \
  --kv-mode q8 \
  --out /tmp/q3.5-0.8b-hipfire-hidden-chunk0.bin
```

Dump the HF oracle:

```bash
python3 scripts/dump_hf_hidden_states.py \
  --model <path-to-Qwen3.5-0.8B-safetensors-snapshot> \
  --ref <path-to-qwen3.5-0.8b-bf16.kldref.bin> \
  --chunk 0 \
  --device auto \
  --out /tmp/q3.5-0.8b-hf-hidden-chunk0.bin
```

Compare:

```bash
python3 scripts/compare_hidden_states.py \
  --hf /tmp/q3.5-0.8b-hf-hidden-chunk0.bin \
  --hipfire /tmp/q3.5-0.8b-hipfire-hidden-chunk0.bin
```

For position-bucketed analysis:

```bash
python3 scripts/compare_layer_positions.py \
  --hf /tmp/q3.5-0.8b-hf-hidden-chunk0.bin \
  --hipfire /tmp/q3.5-0.8b-hipfire-hidden-chunk0.bin \
  --layer 4
```
