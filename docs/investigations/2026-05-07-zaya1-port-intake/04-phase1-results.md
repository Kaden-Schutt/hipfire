# 04 - Phase 1 Results + Empirical Findings

**Date:** 2026-05-07
**Branch:** `feat/zaya1-port-intake`
**Hardware:** hiptrx GPU 2 (R9700, gfx1201, 32 GB VRAM)
**Lane:** `HIP_VISIBLE_DEVICES=2` (GPUs 0, 1 reserved for the concurrent
gemma-eseries agent per overnight contract).

## What landed

1. **Arch crate scaffold** - `crates/hipfire-arch-zaya/` compiles, 5 unit
   tests pass. All 5 trait methods either return typed defaults or
   `Err` with a pointer to the intake docs. arch_id = 7. EOS filter
   overrides Gemma turn markers.

2. **Intake harness** - `scripts/arch-intake/dump_zaya_reference.py`
   plus `scripts/arch-intake/prompts/zaya_canonical.txt`. Hooks
   `model.layers[i].input_norm`, `.res_scale`, `.layer_out`,
   per-layer `.self_attn` (ATT layers) or `.zaya_block + .router +
   .experts` (MLP layers), plus `model.final_norm` and
   `model.lm_head`. Skips None mods at registration time. Captures
   tensor outputs as 1D fp32 safetensors with a manifest.

3. **First reference dump captured.** 80 sub-layers + final layer,
   23-token canonical prompt, bf16 model, fp32 dump. 402 hooks
   registered, 321 tensor outputs captured (the 81 misses are hooks
   on modules that returned non-tensor outputs, e.g. CCA's `(q, k, v)`
   tuple where output[0] only captures q; refinement is Phase 2).
   Total dump size 75 MB. Lives at `/tmp/zaya-port/refs/refs-canonical/`
   on the dev box (not committed; per file policy).

## Empirical findings (corrections to earlier docs)

### Layer structure: 40 ATT + 40 MLP, alternating

`model.layers` (`ZayaModel.layers`) is 80 items long, alternating:

```
layer 0: ZayaDecoderATTLayer  (.self_attn, .input_norm, .res_scale)
layer 1: ZayaDecoderMLPLayer  (.zaya_block, .input_norm, .res_scale)
layer 2: ZayaDecoderATTLayer
layer 3: ZayaDecoderMLPLayer
...
layer 79: ZayaDecoderMLPLayer
```

So `num_hidden_layers=80` in the config means **40 ATT sub-layers + 40 MLP
sub-layers**, NOT 80 attention blocks. Implications:

- **CCA recurrent state count = 40, not 80.** Per-sequence at fp16:
  ~370 KB instead of the ~720 KB the original disambiguation doc
  estimated. `cca_state_bytes_per_seq()` now divides by 2.
- **KV cache layer count = 40.** Halves vs my earlier estimate.
- **MoE layer count = 40.** Each MLP layer carries its own router +
  16 experts (or 17 with MoD's skip slot).
- **Block layout:** every consecutive pair = (ATT, MLP) is one
  "logical decoder block" in Llama-equivalence terms; ZAYA1's
  per-block parameter count is half of what 80-layer Llama would be.

### Final norm attribute

`model.model.final_norm` (NOT `.norm` or `.final_layernorm`). Hook list updated.

### Tokenization

Canonical prompt "The recurrent state in CCA carries two buffers per
layer per sequence: conv_states and prev_hs." → 23 tokens. First few:
`[2, 818, 58944, 1883, 528, 127887, 23076, 1156, 68378, 810, 6352,
 810, 7501, 236787, 5492, 236779, 26582, 532, 8820, 236779]`. Token 2
is BOS (matches `bos_token_id=2` in config). Token 106
(`<end_of_turn>`) does not appear in this prompt; expected to appear
at decode time.

Prompt md5 (committed for repeatability): `5d3130fb415613133c4f8f7889770188`.

### Side names actually captured

- `experts` (MLP layers, output of SequentialMLP)
- `final_norm`
- `input_norm` (every layer)
- `lm_head`
- `res_scale` (every layer)
- `router` (MLP layers)
- `self_attn` (ATT layers)
- `zaya_block` (MLP layers)

Missing on first capture: `cca` (CCA submodule's tuple output).
Phase 2 work to add a multi-output hook that captures (q, k, v)
separately.

## What did NOT land tonight

- **HFQ writer for ZAYA1.** Per Phase 0 design notes, this is a
  prerequisite for any hipfire-side forward execution.
  `hipfire-quantize` does not know the Zaya tensor naming yet. ETA
  ~3-5 days after Phase 6.A approval.
- **CCA scalar reference + kernel.** Phase 3 is gated on Phase 6
  Option A/B decision. Per contract, no autonomous attempt.
- **Phase 2 free-component validation.** Each free component (RMSNorm
  reuse, SwiGLU reuse, GQA reuse, partial-RoPE, scale_residual_merge,
  MLP router, top-1 routing) needs hipfire-side forward to be running.
  Forward path is currently fully stubbed (returns Err). Without a
  ZAYA1 HFQ representation, NRMSE comparison can't run. Phase 2
  becomes feasible once the HFQ writer lands.

A workaround for partial Phase 2 validation: a pure-Rust CPU forward
that consumes the bf16 safetensors directly (skipping HFQ entirely)
and runs each component as the reference goes. ~4-8 hours of work to
build that scaffold; deliberately not attempted tonight to keep this
PR scoped to intake.

## Decision points open for Kaden (recap)

See MANUAL_REVIEW.md for the full escalation. Three coupled decisions
gate Phase 6.A:

1. State location: per-arch State (Option A, recommended) vs
   first-class recurrent-cache primitive (Option B).
2. First-ship spec-decode policy: AR-only (recommended) vs
   recurrent-spec-decode from day one.
3. Paging policy: HBM-pinned (recommended) vs paged.

## Branch state at end of intake

| Commit | Subject |
|---|---|
| `9a4ba59` | docs(zaya1-intake): Phase 0 CCA disambiguation, VERDICT: RECURRENT |
| `36c296a` | feat(zaya1-intake): Phase 1 scaffold for hipfire-arch-zaya |
| `bfce6e7` | docs(zaya1-intake): Phase 4 MoD + Phase 5 EDA + Phase 6 recurrent-state |
| (this commit) | docs(zaya1-intake): Phase 1 results + 40+40 layer correction |

Push branch + open draft PR is the final overnight action.
