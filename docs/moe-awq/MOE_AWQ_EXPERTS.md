# MoE AWQ for 3D Experts — Design & Implementation Guide

**Status (2026-06-11):** quantizer side shipped + validated; runtime kernels designed, not built.
**Branch:** `feat/moe-awq-experts` — sits **directly on master tip** (`e2d21ae2`),
1 commit ahead (`6198851e`), 0 behind. Not stale. On mi300 at `/root/dflash`.

---

## 0. TL;DR

hipfire's AWQ (activation-aware weight quant) was **dense/attention-only**. MoE
routed experts — the bulk of an A3B-class model's params — got *no* AWQ; their
only imatrix lever was MQ2/MQ3-Lloyd. This work adds per-expert AWQ to 3D MoE
experts.

- **Quantizer (DONE):** per-expert `W·s` pre-scaling + a per-expert
  `.awq_scale.weight` F16 sidecar, fed by per-expert `in_sum2` from a GGUF
  imatrix. Validated on **Nex-N2-mini** (Qwen3.5-VL-MoE, 256 experts × 40
  layers): emits 20,480 sidecars, logs `MQ4G256+AWQ`.
- **Runtime (TODO):** the sidecar is **inert + currently WRONG to run** — the
  weights are `W·s` but nothing applies the compensating `x/s`. The kernel work
  is the divide-by-scale, which has a non-obvious FWHT-ordering constraint
  (§3).

---

## 1. The math (model-agnostic)

For weight `W[m,k]` and per-input-channel scale `s[k]` (from activation stats):

```
W' = W · diag(s)          (pre-scaled, baked at quant time — ZERO runtime cost)
y  = W·x = W'·(x/s)        (runtime must divide x by s before the matmul)
```

`s[j] = (in_sum2[j])^(α/2)` normalized to geomean 1, clamped [1e-2, 1e2]
(`compute_awq_scales`). MagnumQuant (MQ4) additionally FWHT-rotates W at quant
time and the runtime FWHT-rotates x. **Order matters:** `x/s` must precede the
FWHT — a diagonal and the Hadamard do not commute. The orthogonal FWHT
preserves the dot product, so `FWHT(x/s)·FWHT(W'_row) = (x/s)·W'_row = x·W_row`. ✓

---

## 2. Quantizer side — DONE (`crates/hipfire-quantize/src/main.rs`)

All in the **3D-stacked expert loop** (entry ~line 6005:
`name.contains("mlp.experts.") && (gate_up_proj||down_proj) && shape.len()==3`,
under `is_moe = arch_id == 6`). Commit `6198851e`.

| piece | location | notes |
|---|---|---|
| `imatrix_in_sum2_for_parent` | new fn after `imatrix_col_weights_for_parent` (~2005) | returns **raw** per-expert in_sum2 (NOT `sqrt(in_sum2/count)` — `compute_awq_scales` applies `^(α/2)` internally; rms would halve effective α) |
| AWQ gate + table | before rayon loop (~6108) | `expert_awq_active = AWQ_ALPHA.is_some() && imatrix_gguf.is_some() && supports_g256 && !any_lloyd_or_mq6_branch` |
| AWQ override branch | top of rayon `.map` (~6119) | `compute_awq_scales(raw_in_sum2_e, α)` → `awq_pre_scale_weights(slice, m, k, s)` → `quantize_mq4g256`; returns `(weight, Some(sidecar))` |
| sidecar emit | paired return + flatten (~6188) | `{parent}{X}.{gate_up_proj,down_proj}.awq_scale.weight`, `QuantType::F16`, shape `[K]` |
| imatrix name map | `safetensors_to_imatrix_key` (~1928) | **Qwen-hardcoded**: `mlp.experts.gate_up_proj → blk.N.ffn_gate_exps`, `down_proj → ffn_down_exps` |

Reusable AWQ primitives (shared with dense, ~3890–4002): `compute_awq_scales`,
`awq_pre_scale_weights`, `awq_scales_to_f16_bytes`, `awq_eligible`.

**Recipe:** `--format mq4 --awq --awq-alpha 0.5 --imatrix <gguf> --no-kmap`
(`--no-kmap` keeps experts MQ4 so AWQ fires on all; with kmap, promoted experts
go MQ6 and skip AWQ by design).

**gate_up fusion note:** safetensors stores `gate_up_proj` fused
`[2*moe_inter, hidden]` (K=hidden). GGUF splits it (`ffn_gate_exps` +
`ffn_up_exps`), but gate and up share the *same input* (post-norm hidden), so
their `in_sum2` are identical → using `ffn_gate_exps.in_sum2 [hidden, n_exp]` is
the correct fused-K scale.

---

## 3. Kernel side — DESIGN (TODO)

The runtime must apply `x/s` per-expert before the FWHT. Two projections,
asymmetric difficulty:

### 3a. down_proj — EASY (≈free) — KERNEL WRITTEN
**DONE:** `kernels/src/fused_silu_mul_mq_rotate_awq_indexed.hip` (registered
`FUSED_SILU_MUL_MQ_ROTATE_AWQ_INDEXED_SRC`). Near-exact copy of the dense
`fused_silu_mul_mq_rotate_awq` + per-expert scale select
`expert_down_awq_ptrs[topk_indices[krank]]`. Compiles for gfx942.
REMAINING PLUMBING: (1) loader — extend `hfq.rs::load_awq_scale` to load the
per-expert `down_proj.awq_scale.weight` sidecars + build an `expert_down_awq_ptrs`
[n_exp] table (mirror `expert_down_ptrs`, qwen35.rs ~579/2105); (2) ExpertWeights
field; (3) dispatch in the MoE decode forward (`moe_ffn_decode_impl`): when down
has awq_scale, call the `_awq_indexed` silu-rotate (pass ptr table + topk_indices)
instead of the plain `fused_silu_mul_mq_rotate`. Gate behind HIPFIRE_MOE_AWQ.

### 3a-NAV. Dispatch wiring runs THROUGH the unified MoE dispatch family (KEY)

Navigation finding (2026-06-11): the routed-expert silu-rotate is NOT a direct
gpu.fused_silu_mul_rotate(...) call in qwen35.rs. Path:

    moe_ffn_decode_impl (qwen35.rs:4665)
      -> builds MoeParams { expert_down_ptrs, down_expanded, routed_down_*, ... }
      -> moe_family().run(&ctx, gpu, &moe_params)
        -> MoeFamily::run (hipfire-dispatch/src/families/moe.rs; MoeParams @148-185)
          -> deeper kernel sequence (gate_up GEMV -> silu-rotate -> down GEMV);
             the silu-rotate is dispatched inside the run impl (not in
             families/moe.rs directly).

So AWQ-down dispatch is a MULTI-CRATE change, NOT a kernel swap:
1. MoeParams (+ MoePrefillParams, hipfire-dispatch) gains
   expert_down_awq_ptrs: Option<&GpuTensor> (None on non-AWQ files).
2. MoeFamily::run silu-rotate step: when expert_down_awq_ptrs.is_some(), call
   fused_silu_mul_mq_rotate_awq_indexed (pass ptr table + topk_indices) instead
   of fused_silu_mul_rotate_mq_batched.
3. Loader (qwen35.rs load_moe_ffn @4331, NOT paro_load_moe_ffn): per-expert
   down.awq_scale = load_awq_scale(hfq, gpu, "...experts.{x}.down_proj.weight", mi)
   (hfq.rs:589; load_weight_tensor does NOT auto-load it). Build
   expert_down_awq_ptrs from e.down.awq_scale (mirror expert_down_ptrs @~4402).
   Add the field to MoeFfnWeights (@566) + ALL constructors (paro_load_moe_ffn
   @1959 builds an empty/None table).
4. rdna-compute launcher for the indexed kernel (or call from the run impl).

Gate on down.awq_scale.is_some() (auto-detect) + HIPFIRE_MOE_AWQ kill-switch.
Prefill (MoePrefillParams) mirrors decode. Test file ready:
/workspace/nex-n2-mq4awq-down.hfq (down-only AWQ, gate_up plain).

### 3a-orig. down_proj — design
Input = per-expert SwiGLU output. The existing per-expert kernel
**`fused_silu_mul_mq_rotate_awq.hip`** already does silu·mul + AWQ-divide +
FWHT for the *dense* down path. Make an **indexed/MoE variant** that:
- reads the per-expert `down.awq_scale` (new `expert_down_awq_ptrs` table),
- multiplies the SwiGLU output by `1/s_e[j]` before the FWHT it already does.
No new launch (folds into the per-expert silu-mul-rotate that already runs).

### 3b. gate_up — the costly one
Input = shared hidden `[K=hidden]`. Baseline rotates it **once** (shared, via
`fused_rmsnorm_rotate_mq_batched`), then the indexed gate_up GEMV
(`gemv_hfq4g256_moe_gate_up_*_indexed.hip`, **no in-kernel FWHT**) consumes it.
AWQ needs `FWHT(hidden / s_e)` *per routed expert*. Design:
- New kernel **`moe_awq_divide_rotate`**: input `hidden[K]` + `topk_indices` +
  `expert_gate_up_awq_ptrs`; output `x_e[K_TOP, K] = FWHT(hidden / s_e)`.
  Compute `x_e` **once per (krank, group)** — do NOT fold into the gate_up GEMV
  (it has 1024 output rows/expert → would recompute the rotation 1024×).
- Patch the gate_up GEMV to index `x` by `krank` (`x + krank*K`) instead of the
  shared `x`. One-line change to `gemv_hfq4g256_moe_gate_up_*_indexed.hip`.

### 3c. Runtime plumbing (`crates/hipfire-arch-qwen35/src/qwen35.rs`)
- `ExpertWeights` (~552): the per-expert `expert_gate_up_ptrs`/`expert_down_ptrs`
  pointer-table pattern (~579, built ~2105) already exists — mirror it as
  `expert_gate_up_awq_ptrs` / `expert_down_awq_ptrs` from the loaded
  `.awq_scale` sidecars.
- Load the sidecars at model open (pair `<parent>.{X}.{proj}.awq_scale.weight`
  with the expert weight). Store the inverse `1/s` (or divide in-kernel).

### 3d. Dispatch (`crates/rdna-compute/src/moe.rs`)
Detect AWQ (sidecars present) → route to the `_awq` MoE GEMV path + pass the
scale tables. Gate behind a flag (`HIPFIRE_MOE_AWQ`) until validated.

---

## 4. Cost on RDNA & cheaper options

Weight side = free (baked). x side, per token decode:

- **down: ≈free** — divide folds into the silu-mul-rotate that already runs.
- **gate_up:** arithmetic ~0.8% of the expert GEMV, bandwidth ~0.7%; the real
  cost is **one extra launch/layer** → **~2–3% decode** on RDNA (launch-bound).
  Prefill near-free (fold into the grouped-GEMM scatter).

**Cheaper:**
1. down-only AWQ → free, and the post-SwiGLU activations are where the outliers
   AWQ targets actually live.
2. **per-LAYER-shared gate_up scale** → shared rotate, zero extra launch, FREE
   (lose per-expert granularity on the input side, which is the post-layernorm
   hidden = already well-conditioned). Quantizer tweak: average per-expert
   `in_sum2` for gate_up, emit one sidecar/layer. **Measure free-vs-(2–3%)
   before committing to per-expert gate_up.**
3. Recommended first cut: **per-expert down (free) + per-layer-shared (or
   skipped) gate_up**, A/B the gate_up quality delta.

---

## 5. Other arches — what each needs

The AWQ math + the runtime kernel (keyed on the `hfq4g256_moe` kernel family,
not the arch) are **shared**. What's per-arch is (a) the expert-layout branch
the AWQ override lives in, and (b) the imatrix name mapping. Per-arch:

- **Qwen3.5-MoE (arch 6) — DONE.** 3D-stacked fused `mlp.experts.gate_up_proj`/
  `down_proj`. A3B, 35B-A3B, VL-MoE (Nex-N2). This is the shipped path.
- **DeepSeek-V4 (arch 9).** Separate per-expert 2D `layers.L.ffn.experts.E.{w1,
  w2,w3}.weight` (own branch ~5919, currently MQ2-Lloyd unit-weight). To add
  AWQ: (1) add an AWQ branch to *that* 2D path (w1=gate, w3=up share input → one
  scale; w2=down own scale); (2) name-map `ffn.experts.E.wN` → ds4 imatrix keys;
  (3) ds4 experts are currently **MQ2-Lloyd**, not g256 — to use the AWQ kernel
  they'd need to move to MQ4G256 first (or build an MQ2-Lloyd-AWQ path). Biggest
  lift of the three.
- **Gemma-MoE (if/when).** Check its expert layout: if it uses the HF 3D
  `mlp.experts.gate_up_proj` convention it nearly drops into the arch-6 path
  (verify the `language_model.` prefix + GGUF `ffn_*_exps` naming, extend
  `safetensors_to_imatrix_key`); if separate 2D, treat like ds4. Gemma also has
  per-layer norm quirks (sandwich norms) but those don't touch expert quant.
- **MiniMax-MoE.** Verify layout (likely separate 2D experts). Same recipe as
  ds4: layout branch + name map. Its experts may already be a specific format
  (check whether g256 or Lloyd) — AWQ kernel only applies on the g256 path.
- **GPT-OSS / other HF-3D-MoE.** If 3D-stacked fused gate_up, generalize the
  arch-6 branch guard (it's currently name-pattern + `is_moe==arch6`; relaxing
  to any 3D `mlp.experts.*` with a per-arch name map would cover them).

**General rule:** to add an arch you write (1) a ~20-line AWQ branch in its
expert-quant path and (2) a `safetensors_to_imatrix_key`-style name map. The
kernel is written once.

---

## 6. Validation plan

1. **Quantizer (done):** sidecar count = `n_layers × n_experts × 2`, F16, len K.
2. **Kernel correctness:** CPU-ref a single expert GEMV: `W·s` dequant ·
   `(x/s)` (rotated) == `W·x` within MQ4 quant error. Compare AWQ-off vs AWQ-on
   dequant error on a synthetic outlier-channel tensor (AWQ should reduce it on
   the salient channels).
3. **End-to-end:** load the 19.7GB Nex-N2 file, coherence-gate, and KLD A/B:
   MQ4 (no AWQ) vs MQ4+AWQ-experts vs MQ2-Lloyd-imatrix (route C). Use an
   **agentic** calib/eval set (Nex-N2 is an agent model — generic wikitext
   undercalibrates its domain).
4. **Perf:** decode tok/s AWQ-on vs off (expect ~2–3% if per-expert gate_up,
   ~0% if down-only/shared-gate_up).

---

## 7. File index

**Quantizer:** `crates/hipfire-quantize/src/main.rs` — `imatrix_in_sum2_for_parent`,
`safetensors_to_imatrix_key` (~1928), 3D expert loop (~6005), AWQ primitives
(~3890–4002), dense sidecar emit pattern (~7164).
**Kernels (templates):** `kernels/src/fused_silu_mul_mq_rotate_awq.hip`,
`rotate_x_mq_awq.hip` (dense AWQ-rotate), `gemv_hfq4g256_moe_gate_up_*_indexed.hip`,
`gemv_hfq4g256_moe_down_*_indexed*.hip`, `fused_rmsnorm_rotate_mq_batched` (shared rotate).
**Runtime:** `crates/hipfire-arch-qwen35/src/qwen35.rs` — `ExpertWeights` (~552),
expert ptr tables (~579, built ~2105); `crates/rdna-compute/src/moe.rs` (dispatch).
**Test model:** `/workspace/nex-n2-mq4awq.hfq` (19.7GB, on mi300); imatrix
`/workspace/nex-imatrix/imatrix_unsloth.gguf_file`; bf16 src `/workspace/nex-n2-mini`.
**Memory:** `project_moe_awq_experts_2026_06_11`,
`project_hfim_native_imatrix_2026_06_07` (dense HFIM), `ragged-k-g128-localized`
(separate G128 stride bug, NOT this).
