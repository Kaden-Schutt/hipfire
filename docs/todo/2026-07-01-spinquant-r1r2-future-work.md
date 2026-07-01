# TODO — SpinQuant R1/R2 learned-rotation W4A4: future work

Companion to `docs/todo/2026-07-01-spinquant-learned-rotation-w4a4.md` (the main
plan, phase-by-phase status). That doc records what **landed** (Phases 0–3 merge,
Phase 2 learned R1, Phase 5 bake math). This doc details the **remaining** work,
each item self-contained enough to pick up cold.

Landed foundation (all on `chaingun`, `hipfire-train`):
- `src/rotation.rs` — `Rotation` (identity/random/hadamard/block_fwht), `apply_r1`
  (hidden-dim, fold+rotate, unties head), `apply_r2` (head-wise value/o_proj),
  `bake_for_oq4_recipe(M) = Fᵀ M`, `rotate_rows`, `compose`/`transpose`.
- `src/learn_rotation.rs` — Cayley-SGD Stiefel optimizer (`cayley_step`) + kurtosis
  objective (`learn_rotation_kurtosis`, `learn_rotation_joint`).
- `src/a4_quant.rs` — int4 per-256-group activation sim (`a4_simquant`, `snr_db`).
- `src/model.rs` — untied-head backward (all `model_*_backward`).
- kernel copy `kernels/src/gemm_iu4_i32_wmma_r1.hip` + `Gpu::gemm_iu4_i32_wmma_r1`.
- Probes: `rotation_invariance_probe`, `rotation_a4_snr_probe`, `w4a4_r1_probe`,
  `learned_r1_probe`, `learned_r1_w4a4_probe`, `gradcheck_model_untied`; parity
  `parity_gemm_iu4_i32_wmma_r1`.

Key results to beat / not regress: full-W4A4 q_proj SQNR on real Supra-50M —
naive 13.1 dB → per-group FWHT (deployed) 20.1 → fixed Hadamard R1 20.1 → **learned
R1 21.85** (+1.73). Act-only A4 output SNR: learned 27.2 vs fixed-Hadamard 22.7.

---

## 1. Learn R2 (head-wise) — **DONE** (act-only + joint); joint {R1,R2} still open

**Landed.** `examples/learned_r2_w4a4_probe.rs` — mirror of the R1 probe on the
head_dim axis. Captures per-layer value `v` (n_kv heads) and o_proj input `ctx`
(n_heads), learns `R2 [head_dim,head_dim]` three ways (value-only, ctx-only,
joint ctx+o_proj-weight) via the *unchanged* `learn_rotation_kurtosis`/`_joint`,
expands each to a block-diagonal `[q_dim,q_dim]` rotation (R2 per head), and
scores full-W4A4 `ctx·Woᵀ` SQNR through the real `iu4·iu4` kernel. No new lib
mechanism — reuses `rotate_rows`, the kurtosis learners, and the r1 kernel copy.

**Result (real Supra-50M, GQA n_heads=8/n_kv=4/head_dim=64, mean over layers):**
naive 15.24 dB → fixed per-head Hadamard 15.90 → **learned R2**: value +0.44,
**ctx +1.35 (17.24)**, joint +1.09. The learned per-head R2 beats the fixed
per-head Hadamard, and learned-on-ctx (17.24) even beats the codec's cross-head
per-256-group FWHT (16.93) — a *head-local, `apply_r2`-mergeable* rotation wins.
Orthonormality ≤4.8e-7.

**Insight — learn on `ctx`, not `v`.** The doc originally nominated the value
activations (the SpinQuant target / KV4 relevance). But on the **o_proj int4
path** the tensor actually quantized is `ctx = P·V` (R2 flows through attention:
rotating V by R2 rotates ctx by R2 per head). Learning directly on ctx beats
learning on v by ~0.9 dB here — v is one GEMM upstream of the grid. Keep the
value-learned R2 for the future KV4/R3 path; use the **ctx-learned** R2 for the
o_proj weight/activation W4A4.

**Still open:** truly *joint* {R1,R2} (independent-axis composition is expected to
just add; measure the whole attention block under both, `learned_r1_w4a4_probe` ∘
this). GQA is handled (value set uses n_kv, o_proj/ctx use n_heads, shared R2).

<details><summary>Original plan (for the joint-{R1,R2} follow-up)</summary>

**Why.** `apply_r2` (the merge) is done and fp-invariant, but R2 is still only
ever the identity/random in tests — nothing *learns* it. R2 is the pair that
closes the W4A8 gap on the **value / o_proj** int4 path, which R1 doesn't touch.

**Approach (mirror the R1 learning).**
- Capture per-head value activations: run a fold-only (`apply_r1(identity)`)
  forward, and for each layer take the post-`v_proj` value `v [seq, kv_dim]`.
  Reshape to per-head blocks `[·, head_dim]` and stack across heads+layers into
  `X_v [rows, head_dim]`.
- Learn `R2 = learn_rotation_kurtosis(X_v, rows, head_dim, Rotation::identity(head_dim), …)`.
  `head_dim` is small (64 for Supra), so `hadamard(head_dim)` needs a power-of-two
  head_dim; use `identity` warm start if not. Cayley-SGD is cheap here (`O(hd³)`).
- Also capture the o_proj **input** columns per head from `ctx` and/or fold the
  o_proj weight rows into a joint objective (analog of `learn_rotation_joint`, but
  the per-head axis).
- Joint {R1,R2}: learn independently first (orthogonal axes — R1 on hidden, R2 on
  head_dim; they commute in `apply_r1`/`apply_r2`), then measure. A truly *joint*
  objective is only needed if independent learning underperforms.

**Measure.** A per-head analog of `learned_r1_w4a4_probe`: full-W4A4 SQNR of the
o_proj (`ctx → attn`) path under identity vs fixed vs learned R2, and the whole
attention block under {R1,R2}. New probe `learned_r2_w4a4_probe` (reuse the
`w4a4`/`quant_int4`/`pack_group` helpers; the iu4 kernel copy).

**Gotcha.** GQA: `n_kv ≠ n_heads`. R2 is shared across heads (one `[head_dim,
head_dim]`). The value has `n_kv` heads; the context/o_proj has `n_heads`. The
learning set for R2 should be the **value** activations (n_kv heads); the merge
already handles the query-head o_proj columns.

**Files.** `learn_rotation.rs` (reuse as-is), new `examples/learned_r2_*`. No new
lib mechanism needed beyond the value-activation capture.

</details>

---

## 2. `.hfq` emission — deploy the learned rotation as a real artifact

**Why.** The bake *math* is proven (`bake_for_oq4_recipe`, unit-tested exact), but
nothing writes a servable `.hfq` yet. This is the tangible payoff.

**Constraint.** Export/quantize tooling lives in `hipfire-quantize` (AGENTS
invariant — not the inference path, not `hipfire-train`). `hipfire-quantize`'s
`main.rs` is ~13k lines; the codec `quantize_oq4g256` is `pub(crate)` in
`codecs.rs`.

**Two clean paths (pick one):**
- **(a) `--rotate <M.bin>` in `hipfire-quantize`.** Load a learned rotation `M`
  (dump it from a `hipfire-train` probe as raw `f32 [h*h]`), compute `R1 = Fᵀ M`
  in-quantizer (re-derive `F = block_fwht` there, or ship `R1` directly), and apply
  the fold+reader/writer rotation to each residual-reading/writing weight **before**
  the existing Oq4G256 quantize. Also apply R2 per-head to v_proj/o_proj. This
  re-implements `apply_r1`/`apply_r2`'s host math inside the quantizer (they can't
  depend on `hipfire-train`). ~1 new module in `hipfire-quantize`, wired into the
  per-tensor loop (`quantize_hfq_source_tensor` / `run_hfq_source_pipeline`).
- **(b) Rotated-safetensors round-trip.** A `hipfire-train` tool loads fp32,
  `apply_r1(bake_for_oq4_recipe(M))` + `apply_r2(...)`, writes the rotated weights
  back to a safetensors dir (+ copy config/tokenizer), then the **existing**
  `hipfire-quantize` CLI produces the `.hfq` unchanged. No quantizer edits; heavier
  I/O (full-model round-trip). Good for a first end-to-end artifact; (a) is the
  productionizable form.

**Validate.** W4A4 KLD / zero-shot of the learned-R1 `.hfq` vs a fixed-rotation
Oq4 `.hfq` (baseline), on the tiny-quant battery. Confirm the +1.7 dB SQNR
translates to a KLD/perplexity gain. Then **measure prefill/batched `iu4·iu4` vs
`iu4·iu8` throughput** on gfx1151 (the compute-bound ~2× claim; decode stays
bandwidth-bound — frame the benchmark accordingly).

**Naming.** Per the artifact convention, the deployed file is a standard
`…oq4…hfq` — R1/R2 are merged and invisible to the loader. If a tag is wanted to
mark the learned-rotation provenance, encode it as a dot-group before the quant
token (do **not** invent a new quant token).

---

## 3. Quantized-CE forward (now unblocked; do only if it beats the surrogate)

**Status.** The untied-head backward is landed, so gradient work on rotated models
is possible. What's missing is a **differentiable quantized forward** that threads
R1 through every linear with a straight-through estimator (STE) and optimizes the
quantized network's CE.

**Known risk — measure before building.** A plain STE on `Q(X Rᵀ)·Q(W Rᵀ)ᵀ` has a
~zero gradient w.r.t. `R`: the clean `X Rᵀ·R Wᵀ = X Wᵀ` term is rotation-invariant,
and STE zeroes the derivative of the quant-noise term. SpinQuant's public code
trains R with STE and it works — because the loss is end-to-end CE over a **deep**
network (noise compounds), and/or the per-group **scale** is kept differentiable
(scale = f(amax(rotated group)), which *does* depend on R). Any implementation must:
- keep the per-group scale a differentiable function of the rotated activation
  (don't freeze it), so the gradient is nonzero; and
- verify on a tiny model that the quantized-CE gradient w.r.t. R is nonzero and
  that learning it **beats the kurtosis surrogate** (21.85 dB) before investing.

**If it clears that bar:** thread R1 into `block_forward` (rotate readers/writers +
A4 sim on the linear inputs via `a4_quant`, weight side via `oqplus_quant`), add R1
as a Cayley-SGD parameter, freeze base weights, optimize CE over ~800 WikiText
seqs, ~100–200 iters. This is a multi-file autograd feature; scope it as its own
plan. **Default assumption: the kurtosis surrogate is sufficient** — only pursue
this if a real gap to the surrogate is demonstrated.

---

## 4. Phase 4 — R3/R4 online Hadamard (verify + KV4)

- **R4** = the existing down_proj input FWHT. Verify it is positioned exactly as
  SpinQuant's R4 (online per-group Hadamard on the down_proj input); it likely
  already is (the Oq4 codec's per-group FWHT on the MLP-down contraction dim).
  Mostly a confirmation + doc task.
- **R3** = KV-cache rotation (online Hadamard on Q·K), only relevant once 4-bit KV
  (KV4) is on the table. Defer until the KV4 path exists.

---

## 5. Scale-out / robustness

- **Larger models.** All numbers are Supra-50M (h=512). The learned-over-fixed gap
  and the joint-objective gain are expected to grow where outlier channels bind
  harder (larger hidden, the wide down_proj). Re-run `learned_r1_w4a4_probe` on a
  bigger dense llama once available; `hidden` must be power-of-two & %256 for the
  probes as written (generalize the Hadamard/`block_fwht` for other dims if needed).
- **Offline learn cost.** `cayley_step` is `O(h³)`/iter and the weight-kurtosis
  gradient is `O(rows·h²)`. For h≫512, subsample weight rows (the probe caps at
  ~4096) and/or cap iters; consider a GPU `gemm_f32_train`-backed matmul for the
  `[h,h]` products if h≥4096 makes the host loop too slow.
- **MoE / non-llama.** hipfire-train models are LLaMA-dense only; MoE
  (minimax/qwen35-moe) needs trainable experts — a separate lift, out of scope here.

---

## Quick-start pointers

- Reproduce the headline result:
  `cargo run -p hipfire-train --release --example learned_r1_w4a4_probe`
  (needs the JIT toolchain for the `_r1` kernel:
  `export ROCM_PATH=$HOME/.venv/lib/python3.14/site-packages/_rocm_sdk_core`).
- All rotation/learning math is CPU-unit-tested: `cargo test -p hipfire-train --lib
  rotation learn_rotation a4_quant`.
- The bake identity (`apply_r1(Fᵀ M)` then codec FWHT = `M`) is
  `bake_composes_to_learned_through_codec_fwht` in `rotation.rs`.
