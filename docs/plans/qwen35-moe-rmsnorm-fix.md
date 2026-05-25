# Qwen3.5/3.6 MoE final-norm under-scaling

**Date:** 2026-05-09 (rev 3 — corrected after glm-5 review)
**Status:** Diagnosis settled. The bug is in the MoE path, not the
dense path. Fix is ~3 lines and the opposite direction of rev 2.
**Trigger:** vLLM, HF transformers, and llama.cpp all compute Qwen3.5+
RMSNorm as `x * (1 + w) * rsqrt(mean(x²) + eps)` (`GemmaRMSNorm`). The
`+1` is applied at inference time; safetensors store raw `w`.

This audit went through three drafts. Both prior drafts misdiagnosed
the bug. The current draft is what survives a concrete arithmetic
trace against vLLM. See `qwen35-rmsnorm-convince-glm5.md` for the
reasoning chain that led to this version, including the failure modes
of the prior drafts.

## TL;DR

1. **Storage convention is uniform.** Every Qwen3.5+ checkpoint stores
   `nn.Parameter(torch.zeros(dim))`-init values for every GemmaRMSNorm
   tensor — per-layer or final, dense or MoE. `state_dict()` saves the
   raw parameter `w`, not the runtime `(1 + w)`.
2. **Hipfire's dense path is correct.** `load_norm_weight()` reads `w`
   from disk and bakes `+= 1.0` at load → kernel multiplies by
   `(1 + w_raw)`. Matches vLLM's runtime `weight + 1.0` and llama.cpp's
   conversion-time bake. Effective scale agrees with all references.
3. **Hipfire's MoE final-norm path is wrong.** `load_norm_weight_raw()`
   skips the `+= 1.0` and uploads raw `w` → kernel multiplies by `w`,
   missing the `+1`. Effective final-norm scale is **~38% too small**
   on Qwen3.6-A3B (1.63 vs the correct 2.63).
4. **Commit `1e01c0b` was a mis-fix.** It traded the correct scale for
   a smaller scale to silence a `<think>` spiral on 3.6-A3B reasoning
   prompts. The spiral is likely a separate precision issue in the MoE
   expert routing or quantization that became symptomatic only at the
   correct final-norm magnitude. Reducing the scale masked the
   underlying bug rather than fixing it.
5. **Fix:** remove the `if config.num_experts > 0` fork at
   `qwen35.rs:1253-1257` and `qwen35.rs:1483-1485` so both paths use
   `load_norm_weight`. Be ready for the spiral to return; that's a
   separate audit if it does.

## Empirical evidence

`/tmp/inspect_norms.py` (Python) and
`crates/hipfire-runtime/examples/dump_norms.rs` (Rust) walk the HFQ
format and dump per-tensor mean/std/min/max. The Rust port uses
`HfqFile::tensor_data_vec()` which is byte-identical mmap slicing — no
arithmetic is applied between disk and the dumped values. The numbers
below are the **on-disk values of `w`**.

### Per-layer norms — `w` near init (zeros)

| Tensor | Model | mean | std | range |
|---|---|---|---|---|
| `layers.0.input_layernorm.weight` | 3.5-0.8B | +0.24 | 0.12 | -0.12 to +1.00 |
| `layers.0.input_layernorm.weight` | 3.5-9B | +0.03 | 0.04 | -0.06 to +0.30 |
| `layers.0.input_layernorm.weight` | 3.6-27B | -0.02 | 0.04 | -0.13 to +0.20 |
| `layers.0.input_layernorm.weight` | 3.6-A3B-MoE | +0.03 | 0.05 | -0.08 to +0.33 |
| `layers.11.q_norm.weight` | 3.5-0.8B | +0.63 | 0.21 | -1.02 to +1.13 |

These are deviations-from-zero. The optimizer barely moved them. Effective
scale `(1 + w)` is dominated by the `+1` from the formula. Hipfire's
`load_norm_weight` bakes the `+1` into the f32 buffer before upload,
producing identical effective scale to vLLM.

### Final `norm.weight` — `w` trained to large magnitudes

| Model | On-disk `w` mean | What `load_norm_weight` produces (correct) | What current MoE branch produces (wrong) |
|---|---|---|---|
| Qwen3.5-0.8B (dense) | +3.31 | `(1 + 3.31) = 4.31` ✓ | n/a (dense path is correct) |
| Qwen3.5-9B (dense)   | +1.14 | `(1 + 1.14) = 2.14` ✓ | n/a |
| Qwen3.6-27B (dense)  | +0.96 | `(1 + 0.96) = 1.96` ✓ | n/a |
| Qwen3.6-A3B (MoE)    | +1.63 | should be `(1 + 1.63) = 2.63` | `1.63` (under-scaled by 38%) |

The "trained to large magnitudes" part is just optimizer behavior. The
final norm sits one matmul before the lm_head and gets the most
gradient pressure, so its `w` drifts further from zero than per-layer
`w` does. **The convention is uniform**; only the trained values
differ. Both vLLM and llama.cpp apply `+1` to all of these uniformly,
which is why all three engines produce the correct effective scale.

### `linear_attn.norm.weight` — RMSNormGated, no `+1` (correct)

| Model | mean | std |
|---|---|---|
| 3.5-0.8B layer 0 | +0.88 | 0.07 |
| 3.6-27B layer 60 | +1.02 | 0.07 |

These are `RMSNormGated` weights, initialized to ones (not zeros), no
`+1` in the formula. Hipfire's `load_any_as_f32()` correctly leaves
them raw. ✓

## Tracing through three engines

For `w = +1.14` (Qwen3.5-9B final norm):

| Engine | Operation | Effective scale |
|---|---|---|
| HF transformers | `output * (1.0 + self.weight)` at forward | `2.14` |
| vLLM | `weight = self.weight.data.float() + 1.0`; `rms_norm(x, weight, eps)` (`layernorm.py:380-394`) | `2.14` |
| llama.cpp | `data_torch + 1` at GGUF conversion (`convert_hf_to_gguf.py:4865`); runtime plain `x * w_baked * rms` | `2.14` |
| **Hipfire dense** | `*v += 1.0` in `load_norm_weight` (`qwen35.rs:697`); kernel plain `x * w_shifted * rms` | **`2.14`** ✓ |
| **Hipfire MoE** | `load_norm_weight_raw` (no `+= 1.0`); kernel plain `x * w * rms` | **`1.14`** ✗ |

Four engines agree at 2.14. One outlier at 1.14. Hipfire's MoE branch
is the outlier.

## What hipfire does today

### Kernels (correct as-is, no change)

| File | Line | Formula |
|---|---|---|
| `kernels/src/rmsnorm.hip` | 24 | `out[idx] = x[idx] * weight[i] * rms` |
| `kernels/src/fused_rmsnorm_mq_rotate.hip` | 79 | `x_shared[i] = x_shared[i] * weight[i] * rms` |
| `kernels/src/gated_norm.hip` | 29 | RMSNormGated, no `+1` per vLLM |
| `kernels/src/l2_norm.hip` | n/a | no weight |
| `kernels/src/layernorm.hip` | n/a | full LayerNorm, ViT path |

These are correct. The `+1` lives in `load_norm_weight`, not the kernel.

### Load paths

- `qwen35.rs:685-699 load_norm_weight()` — bakes `*v += 1.0` into the
  f32 buffer before upload. Used for per-layer norms (28 sites). Used
  for **dense** final `norm.weight` (2 sites). **Correct.**
- `qwen35.rs:704-715 load_norm_weight_raw()` — uploads raw, no `+1`.
  Used for **MoE** final `norm.weight` (2 sites). **The bug.**
- `qwen35.rs:810+ load_any_as_f32()` — raw upload. Used for
  `linear_attn.norm.weight` (DeltaNet RMSNormGated). **Correct.**

### The bug, exactly

```rust
// qwen35.rs:1253-1257  (load path #1)
let output_norm = if config.num_experts > 0 {
    load_norm_weight_raw(hfq, gpu, "norm.weight", &[config.dim])?  // ❌ skips +1
} else {
    load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?      // ✓ correct
};

// qwen35.rs:1483-1485  (load path #2, same pattern)
```

The fix is to delete the `if/else` and always use `load_norm_weight`:

```rust
// qwen35.rs:1253 and qwen35.rs:1483 (both occurrences)
let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?;
```

## How prior drafts went wrong

**Rev 1 (kernel-flag refactor, ~60-site change).** Asserted the kernels
should compute `(1 + w)` directly via a `gemma_offset` flag, removing
the `+= 1.0` bake and threading the flag through every dispatch site.
Mathematically equivalent to the current bake-at-load approach but a
much larger blast radius for no benefit. Reviewed by Gemini and
glm-5 (v1) and endorsed; no reviewer noticed that hipfire's existing
load-time bake already produces the correct scale.

**Rev 2 (the wrong direction).** After empirical norm-dump showed
final-norm means of +0.96 to +3.31, I claimed the on-disk values were
already "raw `(1 + w)` scales" and dense `+= 1.0` was double-applying
the offset. Wrong inference. PyTorch `state_dict()` saves
`self.weight.data` (raw `w`); the framework can store any trained
value; large `w` after training is just trained behavior, not a
storage-convention change. Glm-5 (v2) caught this by tracing the
arithmetic concretely: vLLM applies `+1` to whatever's on disk, so
on-disk `+1.14` becomes vLLM-effective `2.14`. Hipfire's dense path
also reaches `2.14`. The proposed "fix" would have under-scaled to
`1.14` — a ~50% regression on every dense Qwen3.5/3.6 inference.

**Rev 3 (current).** Trace says dense matches; MoE doesn't. Fix MoE.

## Risks of the fix

### Risk 1: Qwen3.6-A3B `<think>` spiral returns

Commit `1e01c0b` introduced the under-scaling specifically to silence a
`<think>` infinite spiral on 3.6-A3B reasoning prompts. With the fix
applied, the final-norm scale doubles back to the correct `2.63` and
the spiral may return.

**This is the expected outcome.** The spiral is a real bug, but it is
not a norm-convention bug. Most likely candidates for the actual root
cause:

1. **MoE expert routing precision.** Top-k softmax across 256 experts
   at low precision (MQ4-G256 router) can produce near-tied scores;
   small perturbations from upstream tip the routing into a degenerate
   self-reinforcing pattern. The corrected final-norm scale doubles the
   logit magnitude, which exposes whatever instability was previously
   hidden.
2. **Quantization interaction.** Specific tensors (e.g. the lm_head or
   one of the routed experts on a hot expert) may be at a quant level
   that loses precision when the input has full magnitude. The reduced
   scale was effectively a noise-floor mask.
3. **Attention precision cascade.** Per
   `feedback_attention_precision.md`, "5% attention error cascades into
   attractor within ~10 tokens." If the final-norm under-scaling was
   keeping outputs within a regime where attention error stayed below
   that threshold, restoring the correct scale could push it over.

**Mitigation:** the original PR shipped an opt-out env var
`HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1` (default off post-fix) so we could
A/B back to the broken-but-stable behavior on a single command without
redeploying.

**Update 2026-05-17:** the underlying root cause was identified and
fixed in commit `9b4ab74a` — the daemon's `repeat_penalty` default of
1.3 over a 128-token window was penalizing legitimately repeated
chain-of-thought formatting tokens, dropping the trajectory into a
self-doubt / number-hallucination attractor. The 1.3 → 1.0 default
flip dissolves the spiral on Qwen3.6-A3B reasoning prompts without
needing the under-scaled final norm. A/B verified on /local/hipfire/
qwen3.6-35b-a3b.mq4 (post-merge `f57e07df`): correct GemmaRMSNorm +
`repeat_penalty=1.0` produces the same clean step-by-step reasoning
as the `HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1` workaround. The env-var
fallback was therefore removed.

### Risk 2: Dense quality-gate baselines may shift

The current dense path is correct, so dense outputs *should not change*
post-fix. If the quality gate detects diffs on dense models, something
else is going on — probably an unrelated change in the same PR.

### Risk 3: I'm wrong again

Two prior drafts were wrong. Future agent should not accept this
revision on the strength of the writeup alone. The verification steps
in `qwen35-rmsnorm-convince-glm5.md` are designed to be run
independently. If any step fails, the diagnosis needs another revision.

## Implementation surface (small)

### Code change as shipped

The fix removes the `if config.num_experts > 0` fork (1e01c0b) so
MoE and dense both go through `load_norm_weight` (with the `+= 1.0`
GemmaRMSNorm bake) unconditionally. Two sites: `load_weights` (main
path) and `load_output_into` (multi-GPU output reload).

```rust
let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?;
```

The `load_weights` site carries the rationale block citing 1e01c0b
(the under-scaling workaround that this PR removes) and 9b4ab74a (the
`repeat_penalty` default that actually fixed the spiral).

### What does NOT change

- Kernels (`rmsnorm.hip`, `fused_rmsnorm_mq_rotate.hip`,
  `gated_norm.hip`, `layernorm.hip`, `l2_norm.hip`).
- Dispatch (`rdna-compute/src/dispatch.rs`).
- Per-layer norm loading (`load_norm_weight` stays).
- DeltaNet `linear_attn.norm.weight` (`load_any_as_f32` stays).
- llama.rs / TinyLlama / Qwen3 / Qwen2 (already correct, plain RMSNorm).
- ViT / qwen35-vl (full LayerNorm path).
- Any `.hfq` / `.mq*` model files.

## Verification plan

1. **Capture before.** Run the canonical PEP-8 prompt on:
   - dense 3.5-9B, 3.5-27B, 3.6-27B (expected: unchanged post-fix).
   - 3.6-A3B reasoning prompts (expected: spiral may return).
2. **Apply fix.**
3. **Compare.** Dense outputs should be byte-equivalent (or
   numerically very close — ULP differences from f16 quant interaction
   are acceptable). MoE may regress on reasoning; this is expected and
   not a fix-rollback signal.
4. **Numerical ground truth.** Use `dump_logits_qwen35.rs` to dump
   logits for one prompt; compare against vLLM logits on the same
   prompt at the same position. NRMSE should drop on MoE post-fix
   (final norm now matches vLLM). NRMSE on dense should be unchanged.
5. **MoE spiral path.** Was load-bearing in earlier revisions of this
   plan; superseded after commit `9b4ab74a` (repeat_penalty 1.3 → 1.0
   default) dissolved the spiral. See the updated mitigation note
   above and the script `scripts/test_pr228_spiral_check.sh` which
   reproduces the A/B (rp=1.0 coherent / rp=1.3 spiral / workaround
   coherent) for any future agent who needs to re-confirm.

## Files referenced

- `kernels/src/rmsnorm.hip:24` — kernel (no change).
- `kernels/src/fused_rmsnorm_mq_rotate.hip:79` — fused kernel (no change).
- `crates/hipfire-arch-qwen35/src/qwen35.rs:685-715` — load helpers.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1253-1257`,
  `qwen35.rs:1483-1485` — the two-site fix.
- `crates/hipfire-runtime/examples/dump_norms.rs` — Rust empirical tool.
- `/tmp/inspect_norms.py` — Python equivalent (in active use, do not move).
- `/local/git/vllm-gfx906-mobydick/vllm/model_executor/layers/layernorm.py:356-394` — `GemmaRMSNorm`.
- `/local/git/vllm-gfx906-mobydick/vllm/model_executor/models/qwen3_5.py:40` — explicit `GemmaRMSNorm` import.
- `/home/kread/git/llama.cpp/convert_hf_to_gguf.py:4865` — conversion-time `+1` bake.
- HF transformers `modeling_qwen3_next.py` — `output * (1.0 + self.weight.float())`.
- Prior commits: `2fd1d9f` (introduced `load_norm_weight` with `+= 1.0`), `1e01c0b` (introduced MoE under-scaling).
- Reviews: `qwen35-rmsnorm-bug-plan-rev-gemini.md` (rev 1 endorsement),
  `qwen35-rmsnorm-bug-plan-rev-glm5.md` (rev 2 refutation, correct).
- Concession: `qwen35-rmsnorm-convince-glm5.md`.

## Reproducible empirical commands

```bash
# Build the dump tool (no GPU required).
cargo build --release --example dump_norms

# Confirm the on-disk values for the four reference cases.
./target/release/examples/dump_norms /local/hipfire/qwen3.5-0.8b.mq4 \
    "language_model.norm.weight"
# Expected: model.language_model.norm.weight ... +3.3092 ... -0.777 +5.312

./target/release/examples/dump_norms /local/hipfire/qwen3.5-9b.mq4 \
    "language_model.norm.weight"
# Expected mean ~+1.14

./target/release/examples/dump_norms /local/hipfire/qwen3.6-27b.mq4 \
    "language_model.norm.weight"
# Expected: ... +0.9619 0.1364 -0.271 +1.758

./target/release/examples/dump_norms /local/hipfire/qwen3.6-35b-a3b.mq4 \
    "language_model.norm.weight"
# Expected: ... +1.6279 0.2278 -0.233 +2.484
```

These are `w`. After the fix, every effective scale becomes `(1 + w)`,
matching vLLM and llama.cpp.
