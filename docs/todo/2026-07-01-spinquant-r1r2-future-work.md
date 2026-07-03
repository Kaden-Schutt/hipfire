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

## 1. Learn R2 (head-wise) + joint {R1,R2} — **DONE**

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

**Joint {R1,R2} — DONE.** `examples/learned_joint_r1r2_probe.rs`. Part 1 re-runs
the two int4 W4A4 measurements side by side (q_proj@R1 21.85 dB, o_proj@R2 17.24
dB — both reproduced exactly). Part 2 is the composition proof *through the
model*: bake `apply_r1(R1) ∘ apply_r2(R2)` into a fresh model, re-forward, and
check the joint activations against the analytic single-rotation predictions —
logits vs fold-only 5.8e-5, xn1 vs `R1·xn1_fold` 1.3e-5 (R2 leaves the q_proj
input untouched), ctx vs `blockdiag(R2)·ctx_fold` 1.6e-5 (R1 leaves the o_proj
input untouched). **R1 and R2 commute and compose with zero interference** — the
joint attention block keeps both gains simultaneously. This settles the doc's
open question: a *truly joint objective is unnecessary* (it was only warranted "if
independent learning underperforms"; it doesn't — the axes are orthogonal and the
gains simply add). GQA handled throughout (value set n_kv, o_proj/ctx n_heads,
shared R2).

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

## 2. `.hfq` emission — R1 deploy **DONE** (path (a), serves coherently); KLD number pending

**Landed (path (a), R1-only).**
- `hipfire-train` `examples/learn_r1_dump.rs` — learns R1 (kurtosis on residual
  xn1+xn2) on a real model and writes the raw learned `M` as a `.r1` file
  (`b"HFR1"` + `h:u32` + `h*h` f32 LE). Validates before writing: bakes `FᵀM`,
  reloads a fresh model, `apply_r1(FᵀM)`, asserts fp-invariance vs fold-only
  (Supra-50M max|Δlogit| 5.6e-5).
- `hipfire-quantize` `src/rotate.rs` + `--rotate <M.r1>` — the in-quantizer
  re-implementation of `apply_r1`'s host math with `R1 = FᵀM`. A pre-pass folds
  each RMSNorm into its readers and rotates all residual readers/writers/embedding;
  the rotated f32 is installed as a global override that the universal extractor
  (`tensor_to_f32_with_optional_fp8_scale`) returns, so **every** codec branch
  quantizes rotated weights with no per-branch edits. Tied models are untied: a
  rotated `lm_head` (final_norm folded) is synthesized and emitted F16 — the
  runtime loader auto-prefers an explicit `lm_head.weight` (`hfq.rs:1823`), so **no
  runtime change**. Unit test `bake_cancels_codec_fwht_leaving_m` proves
  rotate-by-`FᵀM` then codec FWHT `==` rotate-by-`M`. Dense llama (arch 0/1).
- Real Supra-50M → 73 MB oq4 `.hfq` (R1 orthonormality 1.1e-6, 110 tensors
  rotated). Tiny-quant gate unchanged (`--rotate` inert when unused).

**SERVING — DONE (coherent end-to-end).** Two blockers found and fixed:
1. *Oq4 not served on llama.* The arch-0/1 llama loader supported qt≤31 but not
   qt=34 (Oq4G256) — a *baseline* oq4 llama panicked at the first weight, pre-
   existing/orthogonal. Fixed by wiring the qt=34/37 loader arm (repack →
   `DType::Oq4G256`; the shared `weight_gemv`/`weight_gemm` Oq4 arms already exist).
   The repack `oq4_pack_arch_combined` was converged into `hipfire_runtime::hfq`
   (qwen35 re-exports, nemotron drops its private copy).
2. *`--rotate` norm double-apply.* The quantizer's non-quantized fallback (norms,
   which `should_quantize` skips) stored RAW source bytes, bypassing the rotation
   override — so a folded RMSNorm kept its real γ while the readers also absorbed
   it (γ² → garbage). Fixed: the fallback honors the override. **This was the whole
   "fold breaks serving" mystery** — found by isolating with M=block_fwht (net=I)
   and lossless q8, which stayed garbage until the norm fix.

Result: `learn_r1_dump → hipfire-quantize --rotate → serve` produces a coherent
oq4 llama ("The capital of France is Paris"; "The sky is blue and white") on
real Supra-50M. Weights are correct by construction (== fp-invariance-proven
`apply_r1`) + unit-tested (`bake_cancels_codec_fwht_leaving_m`).

**Measured — KLD (decode).** `hipfire eval --battery quality`, Supra-50M oq4 vs
bf16: baseline 0.082, learned-R1 **0.100 (+22%, regressed)**. The eval KLD scores
the **decode** path = W4A16 (weight int4, activation f16), but the learned M was
optimized on **activation** kurtosis for W4A4 (both-int4). At decode only weight
int4 matters, so the act-optimized rotation doesn't help — and the tiny model's
baseline is already clean. The +1.7 dB is a W4A4 prefill/batch metric, not W4A16
decode. (The eval path itself had a bug — it VMFaulted on Supra because `kld_eval`
chunked at `n_ctx=2048` past the model's `max_position_embeddings=1024`; fixed to
clamp to `m.max_seq`.)

**Measured — ISA compute rate (gfx1151):** int4 gives a genuine **2× compute
advantage over int8** at the instruction level, on both paths:
- **WMMA** `v_wmma_i32_16x16x16_iu4` = **2.0×** `..._iu8` (AMD matrix calculator:
  16 vs 32 exec cycles / 2048 vs 1024 ops-per-WGP-per-cycle; **measured 1.995×** at
  the latency-hidden sweet spot INDEP=8, cross-validated hipEvent + rocprofv3). At
  INDEP=4 it reads 1.87× — insufficient latency hiding, not a hardware limit.
- **DOT** `v_dot8_i32_iu4` (8 int4 MACs/instr) = **2.0×** the MAC throughput of
  `v_dot4_i32_iu8` (4 int8 MACs; same issue rate). Note `v_dot8_i32_iu8` does not
  exist — the int8 dot is 4-wide.

So the SpinQuant **W4A4 compute premise holds on gfx1151** (contrary to my first
`w4a4_throughput_bench.rs`, which read ~1.0× — that was a *kernel-level* artifact:
it compared W4A4 against **W8A8** weight bytes and was bandwidth/tiling-bound, not
the matrix core). **Caveat:** the *hardware* does 2×; realizing it in the Oq4 path
still needs a compute-bound shape + a well-tiled `iu4` GEMM (the production
`gemm_iu4_i32_wmma` did not hit it at balanced shapes — a kernel-optimization gap).
Tool: `scratchpad/isa_bench.hip` (matrix calc → disasm-verified microbench →
rocprofv3). See memory `reference_gfx1151_int4_isa_rate`.

**Measured — tuned iu4 GEMM + the deployment W4A4/W4A8/W8A8 comparison**
(scratchpad prototypes; memory `reference_gfx1151_iu4_gemm_tuning`). A tuned iu4
GEMM (wave64 + double-buffered LDS + N-heavy 2×8 tiling + BK64 + `ds_load_b128`)
reaches **~50k GOP/s (~14× the production single-chain, ~50% of peak)** — the
"kernel-optimization gap" above is closable. Levers: LDS staging > wave64 (4 vs 8
acc GPR) > larger BK. Dead ends (ISA+measurement): `__shfl` = `ds_bpermute` (the LDS
unit, 6–8× slower), LDS bank-padding (read at bandwidth floor), register-blocking
alone. **Strategic result:** with equally-tuned kernels, **W4A4 ≈ true W4A8
(~1.04–1.06×)** on gfx1151 — W4A8 already captures the int4-*weight* bandwidth, and
the kernels aren't pure-compute-bound so W4A4's 2× *core* edge is masked (W4A4 vs
W8A8 = 1.2–1.9×, but that gap is weight bandwidth W4A8 also gets). ⇒ **on this APU,
W4A8 (int8 activations, no rotation) is ~as fast as W4A4 with far less risk than the
SpinQuant learned-rotation path** — the W4A4 throughput payoff over W4A8 is marginal
here. Revisit on CDNA/gfx12 (core-bound) and productionize the tuned kernel.

**Still to measure:** (a) **DONE above** — W4A4/W4A8/W8A8 tuned-kernel comparison
(W4A4≈W4A8); remaining sub-item: `#2`/`#4` kernel tweaks on small-K/decode shapes;
(b) a **W4A4-prefill** KLD (score via the batched both-int4 path, not decode); (c) a
**joint act+weight** learned M (item ii) for the W4A16 decode path; (d) a larger
dense-llama oq4 with real outliers (Supra-50M is too clean).

**R2 deploy** stays out of this item: the codec's 256-wide FWHT crosses head
boundaries, so a per-head `apply_r2` can't be un-baked (net would be
`F₂₅₆ ∘ blockdiag(R2)`, not `R2`) — needs a per-head codec variant, separate work.

<details><summary>Original two-path plan (path (a) was chosen)</summary>

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

</details>

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
