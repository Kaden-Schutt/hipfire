<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# DFlash drafter spike — dense LFM2.5-350M drafter for Qwen3.6-27B (hipfire-native)

**Status: PLAN — awaiting e2e approval.** Authored 2026-06-08. Branch
`feat/dflash-lfm2-drafter` off `origin/integration/dispatch-unification` (64708bc9).
The far end is **launching a hipfire-Rust-native DFlash training run on MI300X** — no
PyTorch / SpecForge / vLLM in the loop. This doc is the design to approve before code.

## Goal & success criteria

Prove the hipfire-native DFlash trainer end-to-end by producing a **dense
LFM2.5-350M** DFlash drafter for the **Qwen3.6-27B** target, on MI300X, in **hours of
compute** (not days). Spike success =:
- Drafter forward + injection runs block-parallel (B positions) and is gradient-checked.
- A training run converges (loss down, per-position accept up) in hours on a small corpus.
- `dflash_convert` -> hfq -> `spec_step_dflash` runs; **tau measurably > 1** and the output
  is **byte-identical to AR-greedy** (`scripts/coherence-gate-dflash.sh`).

Not in scope for the spike: peak tau, full corpus, multi-GPU, conv-layer injection v2.

## Recipe (grounded — DFlash paper arXiv:2602.06036 + speculators v0.5.0 + kimi)

- Drafter conditions on target hiddens from **5 layers uniformly between layer-2 and
  the 3rd-to-last**, concatenated -> an `fc` projector -> a **target context feature**.
- **KV injection (not input fusion):** the projected feature is injected into the
  **K/V of every draft layer** and the draft block attends over it (paper 4.1, Table 9:
  KV-inject > input-fusion; injecting at every layer > input-only).
- **Block diffusion:** block_size **16 train / 8 infer** (16 generalizes down, not up),
  `mask_token_id`, multi-anchor masked blocks, sparse/Flex attention (bidirectional
  within block + to injected context, no cross-block).
- **Position-weighted CE:** `w_k = exp(-(k-1)/gamma)`.
- **Frozen shared target embed + lm_head;** only the draft layers + adapters train.
- Data = **target-regenerated** responses (run the target over prompts).
- Hyperparams (speculators canonical): `num-layers 5`, `block-size 8/16`, `epochs 5`,
  `lr 1e-4`, AdamW, `max-anchors 3072`, `target-layer-ids "2 18 33"`-style.

## The drafter: dense LFM2.5-350M body + DFlash conditioning

LFM2.5-350M = `lfm2moe` arch_id 11 with `num_experts == 0` -> all layers dense:
16 layers = **10 LIV double-gated conv + 6 GQA**, dense SwiGLU FFN (validated path).

**I/O (vocab align — keep the warm-start):** the draft body runs at LFM2's native
hidden dim `d_draft`; the target's embed/lm_head (Qwen3.6-27B, vocab 248320, hidden
`d_tgt`) are **frozen and shared**:
```
token -> target embed (frozen, d_tgt) -> in_proj (d_tgt->d_draft) -> LFM2.5 body (d_draft)
      -> out_proj (d_draft->d_tgt) -> target lm_head (frozen) -> logits over 248320
```
LFM2.5's native 65536-vocab embed/lm_head are discarded; the 28T-token body is the
warm-start, bridged to target vocab by `in_proj`/`out_proj`.

## PINNED: per-layer target-context injection (the architectural decision)

The target context feature `ctx = fc(concat(target_hiddens))` (one row per context
position, in `d_draft`) is injected **at every draft layer**, with the mechanism chosen
by the layer's mixer — KV-injection where there's attention, gate-injection where
there's conv:

- **GQA layers (x6) -> KV-injection** (exact DFlash, mirrors `dflash.rs`):
  the layer's existing `k_proj`/`v_proj` project `ctx` into context K/V; the draft
  block's queries attend over `[ctx_K/V ; block_K/V]` (concat along the key axis).
  **No new per-layer weights** — reuses `k_proj`/`v_proj`; only the attention-context
  concat is added. This is the strong conditioning the paper relies on.

- **Conv layers (x10) -> gate-injection** (the conv analog of KV-injection):
  LFM2's conv is already double-gated — `conv_y = C_gate (.) conv(B_gate (.) x)`.
  Inject the target context into the **output gate**:
  `C_gate <- C_gate + W_c[l] . ctx`, with a per-conv-layer learned `W_c[l]:
  d_draft -> hidden`. This conditions each conv layer's contribution to the residual on
  the target — reusing the existing gating machinery, the principled conv counterpart of
  KV-injection (vs plain input-fusion). Spike default = additive bias into `C_gate`;
  v2 option = sigmoid-multiplicative gate, or inject into `B_gate`.

  **Spike-minimal fallback:** if conv gate-injection is fiddly first pass, ship
  **GQA-only injection** initially (6 spread injection points; the residual carries
  context through the intervening conv layers since conv/attn interleave) and add
  conv gate-injection as the refinement. Decide by tau.

**New trainable params** (on top of the warm-started body): `fc` (shared),
`in_proj`/`out_proj` (vocab bridge), `W_c[l]` x10 (conv gate-injection). The body
(conv/GQA/SwiGLU) is finetuned; target embed/lm_head frozen.

## Pipeline (3 phases)

1. **Data-gen (mostly HAVE):** hipfire daemon serves quantized Qwen3.6-27B;
   `dump_qwen35_hidden_states.rs` extracts hiddens at `target_layer_ids ~ [2,16,31,46,61]`
   (5 of 64, layer-2 -> 3rd-to-last) + the target's tokens, over a **target-regenerated**
   corpus. **Spike: offline** — ~50-100K samples x short ctx x 5 layers fits on the 192 GB
   box; cache once, train 1-2 epochs. (Online streaming = the scale-up.)
2. **Train (NEW — the hipfire-native Rust trainer):** see below.
3. **Convert + validate (HAVE):** `dflash_convert` -> hfq -> `spec_step_dflash` -> tau +
   `scripts/coherence-gate-dflash.sh` (q8 KV, max=256, md5-pinned prompt).

## HAVE vs NEW (on this branch)

| Piece | Status |
|---|---|
| Dense LFM2.5-350M body **decode** inference (arch_id 11, conv+GQA+SwiGLU) | **HAVE** (validated) |
| Target hidden extraction (`dump_qwen35_hidden_states`) | **HAVE** |
| DFlash convert + spec-step + coherence gate | **HAVE** |
| Banked trainer plan | **HAVE** |
| **Batched/block-parallel gated-conv forward** (B positions) + backward | **NEW** |
| **Block-parallel LFM2 draft forward** (over `[ctx ; block]`) + injection | **NEW** |
| **DFlash conditioning** (`fc`, GQA KV-inject, conv gate-inject, in/out_proj) | **NEW** |
| **Rust trainer subsystem** (backward + Adam + driver + block-diffusion loss) | **NEW** |
| Graft from other branches | **NONE** (integration is complete; stale-branch HFIM is AWQ-calib, irrelevant) |

## The trainer subsystem (greenfield, bounded fixed-arch — no autograd)

- **Forward:** block-parallel LFM2 draft forward (reuses arch_id-11 primitives in a
  batched layout) + `fc` + injection.
- **Backward (hand-coded):** transposed-GEMM (dW/dX); rmsnorm; dense-SwiGLU; **LFM2
  gated-conv backward** (the one genuinely novel kernel — derive from the gated-conv
  forward; **finite-difference gradient-check on a 2-layer toy before any full run**);
  GQA backward via **gradient checkpointing** (recompute the flash *forward* per layer,
  transient softmax+matmul backward — no fused flash-backward kernel); `fc` /
  `in_proj` / `out_proj` / `W_c[l]` backward; **block-diffusion position-weighted-CE**
  backward. Target embed/lm_head **frozen** (no grad).
- **Optimizer:** Adam, fp32 m/v (~5 GB for 350M — trivial on 192 GB).
- **Driver:** streaming loader over cached `(target_hiddens, target_tokens)`;
  multi-anchor block builder + sparse mask; micro-batch + grad-accum; bf16 compute /
  fp32 master; `lr 1e-4` + warmup + cosine; ckpt save/resume; metrics (loss,
  per-position accept). **Single MI300X** for the spike.

## Compute (why "hours" is real)

Kimi's 3-6 days = 1T target + 1.2B drafter + 1.16M samples + 6 epochs + 8 GPUs. The
spike scales every axis down: **27B target** (~37x smaller, quantized-fast), **350M
drafter** (conv-cheap backward), **~50-100K samples, 1-2 epochs, single MI300X**.
Data-gen (the dominant phase) is cheap with the fast quantized 27B. A spike-quality
drafter in **hours**; a polished one (full corpus, 5-6 epochs) in ~a day.

## Risks (ranked)

1. **Batched gated-conv forward + backward** — new kernels; the conv backward is the
   one to gradient-check first. Biggest build risk.
2. **Conv gate-injection efficacy** — spike-minimal GQA-only fallback de-risks.
3. **Block-diffusion loss fidelity** — reimplement the paper's mask schedule + `w_k`.
4. **Vocab/hidden bridge** (`in_proj`/`out_proj`) — costs some warm-start at the edges.

## Build sequence (vertical slice -> the launch)

1. Quantize LFM2.5-350M-Base -> hfq (dense, arch_id 11); confirm decode coherent.
2. `dump_qwen35_hidden_states` on Qwen3.6-27B -> cache a small offline dataset.
3. Batched gated-conv forward (+ block-parallel LFM2 draft forward) + GQA KV-inject.
4. Hand-coded backward + **finite-diff gradient check on a 2-layer toy** (the gate).
5. Block-diffusion multi-anchor loss + Adam + driver.
6. **LAUNCH: hipfire-Rust-native train on MI300X** — hours; watch loss + per-pos accept.
7. `dflash_convert` -> `spec_step_dflash` -> tau + `coherence-gate-dflash.sh`.
8. (refine) conv gate-injection v2 + scale corpus.

The launch in step 6 is the deliverable the user approves toward.
