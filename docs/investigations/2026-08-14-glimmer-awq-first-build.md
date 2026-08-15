# Muse Glimmer — first AWQ build (dense 30B)

**Date:** 2026-08-14 · **Status:** artifacts built and coherent; **quality unmeasured
and currently unmeasurable** · **Hardware:** hiptrx gfx1201 (R9700)

## Summary

Three MQ4 Glimmer artifacts now exist with an identical recipe (MQ4 body + MQ4
attention + Q8 lm_head, arch-14 default `HIPFIRE_Q8_CLASSES=lm_head,embed`),
differing only in AWQ scope:

| artifact | AWQ scope | sidecars | bytes | sha256 (first 32) |
|---|---|---|---|---|
| `muse-glimmer-30b.mq4r` | none | 0 | 16,261,223,424 | `47ccfccddbef5b8e14040bae567c8712` |
| `muse-glimmer-30b-awq-f1.mq4r` | **F1** (q/k/v, attn-gate, mlp gate/up) | 312 | 16,265,401,344 | `c67d3066604103c38d27ebed3fd9bbfa` |
| `muse-glimmer-30b-awq.mq4r` | F2 (F1 + o_proj + down_proj) | 416 | 16,267,912,192 | `cf2f012ee133d87b1314e18faf40f5f0` |

Sidecar counts are exactly `slots × 52 layers` (6×52=312, 8×52=416). AWQ costs
**~4–7 MB** of F16 sidecars and **nothing at runtime**.

## hfim was obtained, not generated

`bartowski/Muse-Glimmer-30B-GGUF` publishes `Muse-Glimmer-30B-imatrix.gguf`
(13.4 MB, md5 `9864f369af8189ecb53cd556030bbd2e`). Verified structurally against
Glimmer before use:

- 832 tensors = 416 `.in_sum2` (F32) + 416 `.counts`
- 8 slots × 52 layers, matching Glimmer's depth
- `K ∈ {6656, 4096, 19968}` = hidden / q_dim / intermediate, all 256-divisible
- includes `blk.N.attn_gate.weight.in_sum2`

Evidence for preferring it over unsloth: `b7507ef51` (2026-06-07) measured
bartowski-HFIM at **+11pt HumanEval** vs unsloth on the code axis, and
`0dfb81dc7` measured a wired unsloth-imat build *worse* (0.075857). Note that
`unsloth/Muse-Glimmer-30B-GGUF` ships **no** imatrix file at all (22 files).

So no native collector was needed. That matters because the native collectors are
narrower than they look: `Gpu::hessian_capture` (`dispatch.rs:544`) is consumed at
exactly one site, `hipfire-dispatch/src/pipeline/mod.rs:1319-1398`, inside
`run_moe_decode_cpu_fallback` — the **MoE expert** path. Glimmer is dense, so it
never reaches it. Generating `hfhs` for Glimmer would still require a new tap at
the projection inputs (`glimmer_layer_decode` / the prefill `proj_gemm_batched*`
calls); `BlockHessianAcc`, the `.hblk` E8H1 format, and its `--hessian-dir`
consumer all already exist and are byte-matched to each other, and
`load_hessian_blocks` is **not** MoE-gated, so GPTQ-E8 does apply to dense.

## Quantizer fix required (committed `425c5031b`)

`safetensors_to_ggml_name` had no arm for Glimmer's attention output gate, so all
52 `self_attn.gate_proj` tensors returned `None` and silently missed AWQ — despite
`awq_eligible` matching `gate_proj.weight` and the imatrix carrying the entry.
Added `"self_attn.gate_proj" => "attn_gate"`. Glimmer-specific: Qwen's gate is
`linear_attn.in_proj_z`, which already mapped. Confirmed live — `attn_gate.awq_scale`
appears in both builds.

## Recipe, with the evidence behind each choice

| choice | value | evidence |
|---|---|---|
| α | **0.55** | `9ed82fce2` (06-28) made α per-arch and kept 0.55 for transformers; only `nemotron_h` wants 0.1. The May α=0.5 optimum was for AWQ **+ aware-GPTQ**, a different stack. |
| scope | **F1** (`HIPFIRE_AWQ_F1_ONLY=1`) | `41b3efc07` (06-19) shipped `v3-awq-f1` as the production dense recipe. May measured F1 0.1257 vs F2 0.1386 *with* GPTQ. |
| formula | paper `s[j]=in_sum2[j]^(α/2)`, geo-mean normalized, clamp [1e-2,1e2] | AutoAWQ weight-magnitude form measured KLD **1.8257** |
| lm_head | leave Q8, no AWQ | output-side AWQ without a runtime `x/s` was a measured bug class |

The runtime **does** apply `.awq_scale` for Glimmer. This is worth stating because
`3533b8983` (06-10) records a gemma4 bug where it did not, producing a
"plain-rotate = KLD~11 lobotomy". Our artifacts emit correct `merge_sort` code at
normal τ, which is only possible if the sidecar is applied — pre-scaled weights
that are never un-scaled would produce garbage.

## Measured: nothing distinguishable, and why

merge_sort fixture (md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`), greedy so each arm is
deterministic, demo md5 `a202053efb80a037fc4cff15f4e77d36`, commit `dccb0f33c`:

| artifact | τ | decode tok/s | output md5 |
|---|---|---|---|
| trunk | 13.900 | 230.16 | `329b4372b83bbe5d77ca9b525f62ec9c` |
| AWQ F1 | 14.000 | 227.81 | **`329b4372b83bbe5d77ca9b525f62ec9c`** |
| AWQ F2 | 14.200 | 229.36 | `1a7c7b7b9c2fa569165275ef395af9b4` |

**AWQ F1's greedy output is byte-identical to the non-AWQ trunk.** Every q/k/v/
gate/up weight changed and not one argmax flipped across 256 tokens. The τ spread
is 1–3 committed tokens over 10 cycles — noise-scale.

**Conclusion: this fixture cannot measure AWQ quality.** Do not cite these τ
numbers as evidence for or against AWQ.

## The blocker

Glimmer has no quality-evaluation path:

- `eval_hipfire` is qwen35+deltanet only. Siblings `eval_hipfire_llama` and
  `eval_hipfire_gemma4` establish the per-arch pattern; there is no arch-14 one.
- No Glimmer BF16/F32 `.kldref` exists (`benchmarks/quality-baselines/harness/manifest.json`
  covers Qwen 0.8b/9b/27b only).
- No Glimmer entry in `docs/perf-checkpoints/` or `benchmarks/quality-baselines/`
  carries KLD or PPL — the 2026-08-11 bring-up checkpoints are decode tok/s only.

To close it: `eval_hipfire_glimmer` (template: `eval_hipfire_gemma4`) plus a
Glimmer F32 reference via `build_kld_ref_native`. Then the project's standard bar
applies — `--scoring-mode prefill --kv-mode q8`, mean KLD below the non-AWQ twin
on identical chunks, reduced with `benchmarks/quality-baselines/harness/kld_reduce.py`,
recorded under `benchmarks/quality-baselines/results/<date>-…/`.

**Until then neither AWQ artifact should become the default**, because there is no
way to demonstrate it helps.

## Do not re-run (measured dead ends)

- MoE per-expert AWQ — wash-to-harmful (`a0c172662`, wt2 0.03338 → 0.03453)
- GuidedQuant g=1 Fisher weighting — 1–2% worse held-out (`caf772a14`)
- OBS saliency tail protection — worse than random (`fdeb905ba`)
- Folding AWQ out before Q4K encode — measured worse (`041ce82b1`)
- AutoAWQ weight-magnitude α formula — KLD 1.8257
- Global α retune past 0.5 for the AWQ+GPTQ stack — exhausted
- Naive GPTQ on an AWQ base without source `W·s` — KLD ~1.75
