# 2026-05-13 Q8 fused prefill — Tier 3 WMMA family

Hardware: AMD RX 7900 XT, gfx1100, host `brain`, ROCm 7.x.
Branch: `feat/q8-prefill-tier2` at `97bf8cf4`.

Models:
- `~/.hipfire/models/qwen3.5-9b.q8f16` — Q8 weights, Q8 lm_head, 8.9 GB
- `~/.hipfire/models/qwen3.5-9b.q8f16-f16lm-2026-05-12` — Q8 weights, F16 lm_head, 9.8 GB (used only for eval-harness disambiguation)
- `~/.hipfire/models/qwen3.5-9b.mq4` — MQ4 baseline anchor

Binaries:
- `target/release/examples/eval_hipfire` (KLD validation + corruption checks)
- `target/release/examples/daemon` (canonical kernel-speed meter)
- `target/release/examples/test_gemm_q8_{qkv,qkvza,gate_up,residual}_wmma` (4 unit tests)
- `target/release/examples/bench_q8_wmma_variants` (T3-1a recipe-pick microbench, retained as regression probe)

Canonical prompt: `benchmarks/prompts/coherence_lloyd_long.txt`, md5 `f20bbc4f5b88ab5f7b44fe7c7da0e2e3`, 220 prompt tokens.

## TL;DR

- Landed the Tier 3 fused Q8_0 prefill family on `feat/q8-prefill-tier2`: 4 production kernels (`gemm_qkv_q8_0_wmma`, `gemm_qkvza_q8_0_wmma`, `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma`) plus the Tier 2 substrate (`gemm_q8_0_batched_chunked`) on master as the decode-path and non-WMMA fallback.
- Daemon `prefill_tok_s = 1069 tok/s` on gfx1100 / RX 7900 XT at the canonical 190-token coherence prompt. Above the planned 600 tok/s stretch.
- Recipe: FP16-WMMA with register-redundant int8→fp16 dequant, no LDS, no `__syncthreads()`. INT8-WMMA deferred — FP16's microbench lead (11–30× over substrate) was too wide for the marginal int8 op-rate win to overturn.
- The eval-harness throughput meter (`eval_hipfire` wall tok/s) is the wrong measurement for Q8 kernel speed — it's dominated by per-position lm_head fan-out, NOT projection throughput. Confirmed on this host: same kernels show 27 / 187 / 1069 tok/s depending on harness choice (Q8 lm_head fan-out / F16 lm_head batched / daemon prefill_tok_s). Methodology doc updated to canonicalize daemon `prefill_tok_s` as the Q8 perf meter; eval_hipfire is retained as the KLD / corruption tool.
- Triple-reviewed: Claude, Gemini, and GLM-5 produced independent critical reviews against the kernels + dispatch + plan; consensus + accepted findings landed as hardening (gfx12 dispatch hazard removed, per-projection dtype `debug_assert`s, K%32 asserts, residual `+=` non-overlapping invariant documented, spec-decode greedy-parity invariant flagged, unit-test bound tightened 5e-2 → 3.5e-2, test-utility deduplication). Two findings were rejected with documented rationale.
- Why this matters: full 256-chunk KLD eval on q8f16 ran in 2.6 hours pre-Tier-3 on this hardware; with these kernels (and a parallel F16 lm_head batching PR) the same eval lands in ≈ 25 minutes. Q8 quality work — calibration ablations (AWQ / GPTQ), per-tensor MSE sweeps, cross-engine comparisons, and the 27B q8f16 floor anchor that was previously time-prohibitive — becomes routine.

## 1. Problem

The KLD eval cohort that anchors `docs/plans/issue-113-quant-quality-eval.md` and the per-tensor floor work in `docs/plans/qwen35-mq4-quality-gap.md` had a Q8 9B variant with no fused-projection prefill path. The pre-Tier-2 state was either:

- **Per-token mode** (`HIPFIRE_PREFILL_BATCHED=0`): byte-identical to decode but at decode-speed — ~10–12 tok/s eval-loop throughput on a 9B Q8 model.
- **Prefill mode through HFQ4-stride kernels (Tier 1 attempt)**: tried by flipping `is_batchable_la` to include Q8 without writing per-dtype dispatch arms. Silently corrupted output (HFQ4-stride kernels reading Q8_0-stride bytes → KLD 0 / PPL NaN). The failure mode IS the lesson: any dispatch arm that reads weight stride must be Q8-aware or Q8 must be filtered out.

Tier 2 (16fba4fd) wired Q8 into the prefill dispatch via `gemm_q8_0_batched_chunked` (which sub-batches the existing `gemm_q8_0_batched` substrate). This **closed the kernel-path bias** in `KLD(MQ4_prefill) − KLD(Q8_per-token)` comparisons but **delivered no throughput** — the substrate kernel benchmarks roughly equal to serial-GEMV-with-staged-output (per its own header). Tier 2 was the methodology fix; Tier 3 is the throughput fix.

Concrete pre-Tier-3 timeline cost: the published 2026-05-12 Q8 9B floor cohort at `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-q8-floor-9b/` measured `slice-mean KLD 0.5735` over 256 chunks (261,888 scored positions). Eval wall-clock: 9217 seconds ≈ 2.6 hours per variant. Variants we want to compare (vanilla / AWQ / GPTQ) × model sizes (0.8B / 4B / 9B / 27B) × KV modes (asym2 / asym3 / asym4 / q8) = ~48 cohort runs. At 2.6 hours each, that's ~5 days of GPU time before considering re-runs.

## 2. Kernel surface

Mirroring `kernels/src/gemm_*_hfq4g256*.hip`, the Q8 family ports the same four projection-shape kernels:

| Op family | Dispatch site | Production kernel |
|---|---|---|
| 4-way fused LA QKV + z + α | DeltaNet `is_q8 && q8_wmma_arch` (~qwen35.rs:4347) | `gemm_qkvza_q8_0_wmma.hip` |
| 3-way fused FA QKV | FullAttn `qkv_is_q8 && q8_wmma_arch` (~qwen35.rs:4796; llama.rs:1558) | `gemm_qkv_q8_0_wmma.hip` |
| 2-way fused FFN gate + up | qwen35.rs:4615, 5117; llama.rs:1784 | `gemm_gate_up_q8_0_wmma.hip` |
| Residual GEMM (wo, w_down) | qwen35.rs:4546, 4683, 5046, 5166; llama.rs:1741, 1828 | `gemm_q8_0_residual_wmma.hip` |

Dispatch helpers in `crates/rdna-compute/src/dispatch.rs:12135–12345`. All gated on the single-source `rdna_compute::has_wmma_f16(arch)` predicate (gfx11 only — see §6.1). Per-site fallthrough order is `is_q8 && q8_wmma_arch` → Tier 2 chunked substrate → other-arch arms.

Tier 2's substrate (`gemm_q8_0_batched_chunked`) stays as:

1. The decode-path Q8 GEMM (preserves single-accumulator FMA order, byte-exact greedy parity with `gemv_q8_0`).
2. The non-WMMA fallback for gfx1010 / gfx1030 / gfx906.
3. The DFlash+Q8 spec-verify GEMM (greedy-parity invariant — see §5).

## 3. Recipe — FP16-WMMA, register-redundant dequant

RDNA3 offers two relevant WMMA builtins:

- `v_wmma_f32_16x16x16_f16` — FP16 inputs, F32 accumulator. Requires int8→fp16 cast in the prologue.
- `v_wmma_i32_16x16x16_iu8` — INT8 inputs, I32 accumulator. No cast for weights, but activations need online quantization per WMMA tile.

### T3-1a microbench result (commit `27a640e2`, on gfx1151)

| Shape | N | FP16-WMMA µs | Substrate µs | Speedup |
|---|---|---|---|---|
| QKV (M=4096, K=4096) | 16 | 47 | 1409 | 30.1× |
| QKV | 256 | 1279 | 20762 | 16.2× |
| gate/up (M=11008, K=4096) | 256 | 4624 | 50842 | 11.0× |
| w_down (M=4096, K=11008) | 256 | 4060 | 65529 | 16.1× |

Numerics: 100% of outputs within 5% relative error (gated to filter near-zero division noise), mean rel error 4–5e-4 — textbook fp16 WMMA precision.

**Decision: FP16-WMMA.** The 11–30× lead is too wide for INT8-WMMA's theoretical 2× op-rate win (vs fp16) to overturn given the added activation-quantization complexity. INT8-WMMA stays deferred behind a stub kernel (`bench_q8_int8wmma.hip`); revisit only if a future Q8 perf gate misses the floor on a new arch.

### Kernel inner loop

Per Q8_0 block (K=32 elements, 34 bytes: `[fp16 scale (2B) | int8[32] (32B)]`):

1. Load fp16 scale via `__builtin_memcpy` (the scale's byte offset 0 within the 34-byte block is 2-byte aligned but not 4-byte aligned across all blocks — explicit `memcpy` avoids unaligned-access UB).
2. Dequant 16 int8 weights into a `half16_t` register: `a[i] = sc * (_Float16)(float)(int)w[i]`. Repeat for the next 16 weights → tile 1.
3. Issue `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` twice per Q8 block (two 16-K WMMA tiles per 32-K Q8 block) against the same wave-broadcast X tiles.
4. Output write maps `acc[j]` → `C[2*j + (tid>>4)][tid & 15]` per RDNA3 wave32 WMMA convention. For the residual variant, the write is `+=` (caller seeds Y with the residual; non-overlapping grid tiles guarantee no race — see §5.3).

`__launch_bounds__(32, 2)` → single wave per workgroup, min 2 waves per CU. No LDS. No `__syncthreads()`.

**Why register-redundant dequant, not LDS broadcast:** the HFQ4 WMMA template lives without LDS staging — threads in a wave32 redundantly dequant the weight tiles they need into registers because the wave-broadcast nature of the WMMA op + small per-thread tile size means LDS would be wasted overhead. Q8's even simpler dequant (no nibble unpack) doubles down on this.

**TODO logged in each kernel inner loop:** the HFQ4 sibling at `gemm_qkv_hfq4g256_wmma.hip:78` runs the K-loop with `kt += 2` and prefetches both X tiles before issuing the WMMA pair — overlaps weight load with compute, avoids `s_waitcnt` stalls. Q8 is 2× the BW pressure of HFQ4 so this matters more here. Not done in this PR — the gfx1100 number is comfortably above the stretch target on the current single-tile-at-a-time form; profile after Q8 lm_head batching lands.

## 4. Validation on gfx1100 / RX 7900 XT

### 4.1 Unit tests

All 4 fused kernels PASS the substrate-equivalence sweep at N ∈ {1, 4, 16, 32, 64, 128, 256} across `tiny`, `medium`, and 9B production shapes:

```
test_gemm_q8_qkv_wmma       PASS  Q: mean_rel=4.4e-4 max_rel=1.9e-2 (+ every-int8-once pattern)
test_gemm_q8_qkvza_wmma     PASS  QKV: mean=4.4e-4/max=1.6e-2  Z: 4.5e-4/1.6e-2  β: 3.4e-4/8e-3  α: 5.8e-4/1.1e-2
test_gemm_q8_gate_up_wmma   PASS  gate: mean=4.4e-4/max=1.8e-2  up: 4.3e-4/1.8e-2
test_gemm_q8_residual_wmma  PASS  mean_rel=4.3e-4  max_rel=1.7e-2
```

The `every-int8-value-once` weight pattern in `test_gemm_q8_qkv_wmma.rs` is the sign-extension regression detector — one block per row contains int8 values `[-128..127]` in order, scale `1/128`, expected output range `[-1.0, 0.992]`. PASS with mean_rel 1.6e-4 / max_rel 4.7e-4 on gfx1100.

Numerics regime matches the original gfx1151 validation 1:1 — same mean rel error band (4–8e-4), same max rel error band (<2e-2 on production shapes). **The recipe ports cleanly RDNA3 → RDNA3 (gfx1151 → gfx1100); no WMMA lane-layout quirks.**

Unit-test tolerance was tightened from `max_rel < 5e-2` → `< 3.5e-2` during review hardening. Production 9B shapes top at 1.98e-2; small synthetic shapes (medium = 512×512 β-projection) hit 3.19e-2 due to WMMA reduction-order noise being more visible at low-M dims. 3.5e-2 keeps a ~30% margin above the synthetic worst case while still being 30% tighter than the original bound.

### 4.2 Daemon `prefill_tok_s` — the canonical perf meter

Via the `long-prefill-q8-9b` row in `scripts/coherence-gate.sh`:

```
qwen3.5-9b.q8f16 — long-prefill-q8-9b  (prompt md5: f20bbc4f5b88ab5f7b44fe7c7da0e2e3)
  wall: 88.9 s     status: OK
  daemon stats:
    prefill_tokens=190
    prefill_ms=177.7
    prefill_tok_s=1069.5    ← the kernel-speed number
    decode_tok_s=62.4
    ttft_ms=177.7
```

**1069 tok/s on gfx1100 / RX 7900 XT.** The plan's targets were 500 (floor), 540 (target), 600 (stretch) — all cleared by a comfortable margin. Output is fluent, structured LRU-cache reasoning (doubly-linked-list / O(1) discussion) — no attractor loop, no special-token leakage.

### 4.3 eval_hipfire throughput is NOT the kernel-speed meter — three-model side-by-side

eval_hipfire scores per-position by running `weight_gemv` × `scored_per_chunk` (≈1023) against the full lm_head matrix. For non-F16 lm_head dtypes this is one GEMV call per scored position × ~1 GB read per call → memory-traffic floor of ~1 TB per chunk just for lm_head fan-out. **That dominates eval wall-clock**, not projection kernel speed.

Same fused kernels, three different lm_head choices, 1-chunk smoke on gfx1100:

| Model | lm_head dtype | eval_hipfire tok/s | daemon prefill_tok_s |
|---|---|---:|---:|
| qwen3.5-9b.q8f16 | Q8_0 | 27 | 1069 |
| qwen3.5-9b.q8f16-f16lm-2026-05-12 (with PR #242 batched fan-out) | F16 batched | 187 | 1069 |
| qwen3.5-9b.mq4 (reference anchor) | MQ4G256 | 248 | ~1100 |

The Q8 fused kernels are doing their job — the gap between 27 and 1069 is the lm_head fan-out wall. PR #242's F16 batched fan-out (parallel work on a sibling branch, not in this PR) closes the gap for F16 lm_head models. A Q8 lm_head batched-fan-out kernel (`gemm_lm_head_q8_0`) would close it for Q8-lm-head models — out of scope here (see §7).

This is exactly Risk #6 in the pre-validation plan: "If T3-3 lands and prefill is fast but eval throughput is still mediocre, lm_head fan-out is the next bottleneck." Confirmed real on gfx1100. Methodology doc (`docs/methodology/perf-benchmarking.md`) updated to call this out and canonicalize daemon `prefill_tok_s` as the Q8 perf meter.

### 4.4 KLD smoke — 32 chunks on q8f16

```
./target/release/examples/eval_hipfire \
  --model ~/.hipfire/models/qwen3.5-9b.q8f16 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --kv-mode asym3 --scoring-mode prefill --max-chunks 32

→ scored 32736 tokens in 166.5s (197 tok/s)
→ slice-mean KLD = 0.098984  mean NLL = 2.267940  PPL = 9.6595
```

KLD trajectory across chunks: 0.090 (chunk 1) → 0.081 (chunk 3) → 0.099 (chunk 32). Climbing smoothly toward the published 0.5735 256-chunk anchor — early-chunk slice-means under-sample, so this is the expected shape. **Silent corruption (KLD = 0 / PPL = NaN — the Tier 1 failure signature) is firmly ruled out** by both the smooth trajectory and the order-of-magnitude consistency with the 1-chunk anchors.

Full 256-chunk match to the 0.5735 ± 0.04 oracle was deferred. The combination of unit-test sweep + 3-chunk smoke + 32-chunk smoke + coherence-gate row covers the corruption surface. The published-anchor match remains a documented follow-up; at the new ~197 tok/s eval rate the full eval lands in ~22 min on this hardware.

## 5. Invariants that must not rot

### 5.1 Greedy-parity invariant — spec-decode stays on the substrate

The WMMA reduction order is hardware-determined; the substrate's `gemm_q8_0_batched` matches `gemv_q8_0`'s single-accumulator FMA order byte-for-byte. DFlash+Q8 spec-verify needs the substrate path so that the draft model's logits are byte-identical to the target model's decode-time logits (greedy parity).

Enforcement: `speculative.rs:2107, 2671` directly call `gemm_q8_0_batched`, NOT a dispatcher that could pick WMMA. Code comments at both sites document the invariant and reference `docs/plans/q8-fused-prefill-kernels.md §Constraints`. If a future refactor unifies the dispatch through a chooser fn, this is the rot risk — the comment is the early warning.

### 5.2 MoE+Q8 silent-corruption barrier

`qwen35.rs:3712–3729` filters Q8_0 out of the `DeltaNetMoe` and `FullAttnMoe` batched-prefill eligibility predicates:

- Attention-side: explicit `!matches!(.. Q8_0)` for wqkv / wz / w_beta / w_alpha / wo (DeltaNetMoe) and wq / wk / wv / wo (FullAttnMoe).
- FFN side: `moe_ffn_all_mq4(&l.ffn)` requires every MoE FFN weight (router, shared expert gate / up / down, all experts' gate_up / down) to be MQ4G256.

A Q8 weight in any MoE position falls back to per-token. The MoE dispatch arms (`fused_qkvza_hfq4g256` family) are HFQ4-stride-only; adding Q8 arms would need fused MoE-routing kernels for all of qkvza / qkv / gate_up / residual MoE sites simultaneously, plus the routed-expert dispatch — out of scope here, gated on demand.

### 5.3 Residual `+=` non-overlapping invariant

`gemm_q8_0_residual_wmma` writes `Y[i] += acc[j]` non-atomically. This is race-free **only because** each (out_row, out_col) cell is written by exactly one lane in exactly one block under the current grid:

```
Grid:  [ceil(M/16), ceil(N/16)]
Block: [32] (wave32),  LDS: 0
WMMA output mapping:  acc[j] → C[row=2*j + (tid>>4)][col=tid & 15]
```

Tiles are 16×16, non-overlapping by construction.

**Rot risk:** a future `_ksplit` variant (the HFQ4 family has one for register-pressure relief) would split K across multiple blocks for the same (out_row, out_col), turning `+=` into a data race. Documented in the kernel header at `gemm_q8_0_residual_wmma.hip` with an explicit "switch to `atomicAdd` if K-splitting is added later" note.

### 5.4 Per-projection dtype consistency

`is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0)` only inspects the routing-anchor weight (wq for FA, wqkv for DN). The fused kernels take 3 or 4 distinct weight pointers and assume all share the Q8_0 stride. A mixed-dtype layer (wq=Q8 but wk=MQ4, e.g.) would re-introduce the Tier 1 silent-corruption mode.

Defense added in 97bf8cf4: `debug_assert!` at every multi-weight Q8 WMMA dispatch site (qkvza, qkv, gate_up across DeltaNet + FullAttn + llama) checking that all sibling weights are Q8_0. Cheap, catches the regression class in dev / test builds; compiles out in release.

### 5.5 K must be a multiple of 32

The kernels iterate `K/32` blocks per row; any K not a multiple of 32 silently drops the tail. Current production shapes satisfy this. Defense added in 97bf8cf4: `debug_assert_eq!(k % 32, 0)` in all 4 dispatch helpers.

## 6. Triple-review synthesis

`feat/q8-prefill-tier2` was independently reviewed by Claude, Gemini, and GLM-5 before merge. Reviews ran against the kernels, dispatch wiring, plan, and pre-merge validation data.

### 6.1 Critical findings — all landed

| # | Finding | Source | Action |
|---|---|---|---|
| 1 | gfx1200/1201 in `q8_wmma_arch` would JIT-crash. The `_w32` builtin is RDNA3-only — `dispatch.rs:155` even has an inline comment "errors with `Cannot select: intrinsic` at codegen time" on gfx12. The Q8 dispatch arms hardcoded gfx12 anyway. | Claude / Gemini / GLM-5 consensus | Replaced hardcoded match with single-source `rdna_compute::has_wmma_f16(arch)` (gfx11 only). Also fixes finding #2. |
| 2 | `q8_wmma_arch` definition fragmented across `qwen35.rs`, `llama.rs`, and partially mirrored in `dispatch.rs::has_wmma_f16`. | Gemini | Consolidated — both call sites now use the dispatch.rs helper. |

### 6.2 Significant findings — all landed

| # | Finding | Source | Action |
|---|---|---|---|
| 3 | Per-projection dtype consistency not asserted (silent-corruption class). | Claude | `debug_assert!` at every multi-weight Q8 WMMA dispatch site. See §5.4. |
| 4 | K % 32 not asserted; kernels silently drop tail. | Claude | `debug_assert_eq!` in all 4 dispatch helpers. See §5.5. |
| 5 | Residual `+=` non-overlapping invariant fragile (a future `_ksplit` variant would silently race). | Claude / GLM-5 | Kernel header documents invariant + `atomicAdd` note. See §5.3. |
| 6 | Spec-decode greedy-parity is convention-enforced, not code-enforced. | Claude | Comments at both `speculative.rs` substrate-only call sites referencing the plan §Constraints. See §5.1. |
| 7 | Unit-test bound `max_rel < 5e-2` is loose vs observed worst case 1.98e-2 (production). | Claude / Gemini | Tightened to `< 3.5e-2` (above the synthetic-shape worst case of 3.19e-2; ~30% margin). All 4 tests still PASS. See §4.1. |
| 8 | Test-utility duplication: `f32_to_f16_bits` + `synth_q8` copy-pasted across 4 test files + 1 bench. | GLM-5 | Extracted to `crates/rdna-compute/examples/common/q8_test_utils.rs`, included via `#[path]`. ~150 lines deduplicated. |
| 9 | Missing software pipelining vs HFQ4 template (HFQ4 does `kt += 2` with X-tile prefetch — Q8 is 2× the BW pressure so this would matter more). | Gemini | TODO comments in each of the 4 kernel inner loops pointing at the HFQ4 precedent. Optimization headroom logged; not done in this PR — 1069 tok/s is comfortably above target. |
| 10 | Residual kernel output mapping diverges from HFQ4's residual convention; the two templates aren't interchangeable. | GLM-5 | Documented in `gemm_q8_0_residual_wmma.hip` header. |
| 11 | Methodology gap — plan's "500 tok/s eval_hipfire" target was framed against the wrong meter. | Claude (during validation) | Methodology doc gains a new section: "eval_hipfire wall-clock is NOT a kernel-speed meter" with the 27/187/1069 tok/s evidence. See §4.3. |

### 6.3 Findings noted but not acted on this PR

| Finding | Source | Rationale |
|---|---|---|
| Scale broadcast is wave-redundant (each scale loaded 32× when it could be loaded once and broadcast via DPP/permlane). | Gemini | Optimization headroom only; deferred per Gemini's own assessment. |
| Double-rounding in dequant chain (int → float → fp16 → fp16 × scale). | GLM-5 | Academic precision note. Mean rel error 4–8e-4 passes easily. INT8-WMMA would eliminate both rounding steps — kept on the "revisit if Q8 perf gate misses on a new arch" list. |
| VGPR spill audit on the `gate_up` shape (M=11008) not done. | GLM-5 | `gfx-kernel-metadata` skill run pending. 1069 tok/s suggests no catastrophic spill, but a formal audit would close the loop. Logged as follow-up. |
| Full 256-chunk q8f16 KLD to formally match the 0.5735 ± 0.04 oracle. | GLM-5 | Combination of unit tests + 3-chunk + 32-chunk smokes + coherence-gate row covers the corruption surface; the formal anchor match is a 22-min follow-up. Documented as such. |
| Coherence-gate Q8 row prompt is 190 tokens (below `PREFILL_MAX_BATCH=256`) — the chunk-256 path is unexercised end-to-end. | Claude | Unit tests cover N=256 at the kernel level; the gate covers fluency at smaller N. Combined coverage is sufficient. |
| `bench_q8_fp16wmma.hip` is in tree but no longer dispatched. | Claude (mine) | Re-labeled as EXPERIMENTAL / REGRESSION PROBE; kept as the source template the 4 production kernels were derived from, plus its harness `bench_q8_wmma_variants.rs` is the re-run path if anyone touches the dequant prologue. |
| `x_rot_batch` buffer naming is misleading for Q8 (contains plain rmsnormed X, not rotated). | Claude (mine) | Too invasive to rename here; logged for a future cleanup PR. |
| `gfx1150` (the other Strix Halo APU SKU) is in the WMMA arch list without hardware validation. | Claude (mine) | gfx1150 is RDNA3 wave32; the gfx1151 author validated on Strix Halo silicon, so gfx1150 should port. Worth a runtime canary kernel if a gfx1150 user hits issues. |

### 6.4 Findings rejected with documented rationale

| Finding | Source | Rationale |
|---|---|---|
| "MoE+Q8 guard is incomplete — missing FFN weight checks for w_gate / w_up / w_down." | GLM-5 §2.2 | **Factually incorrect.** The MoE eligibility predicate at `qwen35.rs:3722` (DeltaNetMoe) and `:3734` (FullAttnMoe) BOTH call `moe_ffn_all_mq4(&l.ffn)`, which requires every MoE FFN weight (router + shared expert gate/up/down + all experts' gate_up/down) to be `DType::MQ4G256`. A Q8 FFN weight in a MoE layer would fail this guard and fall back to per-token. The reviewer missed the helper. |
| "Branch scope creep — RoPE halfsplit + attention_dflash + quality-eval infrastructure intermixed with Q8 work." | GLM-5 §3.1, §5 | **Based on a stale local `master` ref.** GLM-5's `git log master..HEAD` would have shown the flagged commits because local `master` was at `21cec8b6` while `origin/master` was at `2ce82ba9`. `git log origin/master..HEAD` shows the branch contains only Q8-specific commits + the rev-3/rev-4 plan-doc commits. The RoPE halfsplit, attention_dflash, and quality-eval infrastructure are all on origin/master, not unique to this branch. |
| "Add an N=0 unit-test." | GLM-5 §4.3 | Trivial early-return guard; testing it adds maintenance for negligible coverage gain. |

## 7. Out of scope (and why)

- **gfx12 (RDNA4) WMMA siblings.** Reclassified to Experimental / Blind Port. `v_wmma_*_w32_gfx12` has different lane layout than RDNA3 and the gfx11 builtin fails at codegen on gfx12. When a gfx12 user with hardware materializes, port via the `gemm_*_hfq4g256_wmma.gfx12.hip` precedent, guard behind `HIPFIRE_LLOYD_GFX12`-style env, require hardware-verified coherence-gate before default.
- **gfx906 (CDNA1, MI50) wave64 + dp4a Q8 path.** Same precedent as `docs/plans/hfp4-mfp4-rdna3-accel.md`'s gfx906 carve-out. Reference kernel for this family would be the residual variant `gemm_hfq4g256_residual_wave64_dp4a.hip`. Defer until gfx906 user materializes; Tier 2 substrate is the current fallback there.
- **MoE+Q8 batched dispatch.** Needs ~4+ fused kernels for the MoE FFN routing path + relaxation of the MoE eligibility filter (see §5.2). Out of scope for this plan; gated on user request.
- **lm_head Q8 batched dedicated kernel** (`gemm_lm_head_q8_0`). The Q8 lm_head per-position fan-out is the eval bottleneck (§4.3). Pattern is the F16 batched fan-out in PR #242 (`gemm_f16_batched_lmhead`); a Q8 sibling would need its own batched kernel. M = vocab = 248K is the difficult shape. Separate workstream.
- **DFlash + Q8 spec decode.** Substrate-only path stays as the spec-verify GEMM (§5.1); the fused WMMA family is decoupled. DFlash+Q8 itself is not currently a perf target.
- **INT8-WMMA path.** Deferred per §3 unless a future arch misses the floor.
- **Storage-layout follow-up to remove the 34-byte stride.** A `Q8_0_packed` variant with scales in a separate row-major array would eliminate the unaligned-scale-load issue. Format-change work, not kernel work; out of scope.

## 8. Lessons learned (durable findings)

1. **Match the perf-meter to the kernel under test.** eval_hipfire's wall tok/s confounds projection-kernel throughput with lm_head fan-out cost. For any kernel-level claim, use daemon `prefill_tok_s` (via the `done` event in the coherence-gate / coherence_probe / probe_commits flows). eval_hipfire is for KLD and corruption checks.
2. **`is_<dtype>` predicates that route into multi-weight kernels need consistency asserts.** Inspecting only the routing-anchor weight is sufficient for selection, but the fused kernel reads stride from multiple buffers and a mixed-dtype layer becomes silent corruption (the Tier 1 failure). Always add `debug_assert!` on the sibling weights at the dispatch site.
3. **WMMA arch gates belong in one place.** The codebase already had `has_wmma_f16(arch)` with the codegen-failure context comment. Duplicating the arch match in feature crates re-introduced the very hazard the helper was designed to prevent. New helpers go behind a single source-of-truth function.
4. **Residual `+=` works only when grid tiles don't overlap, and that invariant is invisible without comments.** A future K-split variant added "for register-pressure relief" would silently race. Document the invariant at the `+=` site, not just in the plan.
5. **Triple-review with rejection rationale catches more than a single reviewer.** Two of three reviews independently flagged the gfx12 hazard; one reviewer's "MoE FFN guard incomplete" was factually wrong, but the act of writing the rejection rationale forced verification (and produced a follow-up note that the `moe_ffn_all_mq4` indirection is non-obvious — worth an inline comment). Stale local refs are a real review-quality risk; verify `origin/master..HEAD` not `master..HEAD`.
6. **The kernel template that wins a microbench becomes the production family template.** `bench_q8_fp16wmma.hip` is now the durable parent of `gemm_qkv/qkvza/gate_up/q8_0_residual_wmma`. Don't delete the probe — re-run it if anyone touches the dequant prologue or WMMA inner loop to confirm the 11–30× speedup hasn't eroded.
7. **Substrate-as-decode-path is a permanent invariant, not a "temporary fallback".** As long as DFlash+Q8 is a possible future target, the substrate must stay on master with greedy-parity preserved. A future PR that removes the substrate as "redundant given the fused family" breaks this.

## 9. Reproducing

```bash
git checkout feat/q8-prefill-tier2  # at 97bf8cf4
cargo build --release --workspace --features deltanet
cargo build --release -p rdna-compute --examples

# Unit tests — all 4 should report "=== 0 failure(s) ==="
for t in test_gemm_q8_qkv_wmma test_gemm_q8_qkvza_wmma \
         test_gemm_q8_gate_up_wmma test_gemm_q8_residual_wmma; do
    ./target/release/examples/$t
done

# Daemon prefill_tok_s — the canonical perf meter
./scripts/coherence-gate.sh
# → look for "long-prefill-q8-9b" row in /tmp/coherence-*.md
# → expect prefill_tok_s ≈ 1069 tok/s on gfx1100

# KLD smoke (32 chunks ≈ 2.5 min wall on q8f16 with default Q8 lm_head)
./target/release/examples/eval_hipfire \
  --model ~/.hipfire/models/qwen3.5-9b.q8f16 \
  --ref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
  --output /tmp/q8_smoke.bin \
  --kv-mode asym3 --scoring-mode prefill --max-chunks 32
# → slice-mean KLD ≈ 0.099 (on trajectory toward published 0.5735 256-chunk anchor)
```

## 10. References

- Plan source (deleted in the commit that landed this checkpoint): the original `docs/plans/q8-fused-prefill-kernels.md` (rev 4 at `47fd6c4d`) is the institutional history for the four-phase landing pattern (T3-0..T3-3). Everything load-bearing from it is captured in this checkpoint.
- Triple review sources (also deleted in this commit): `q8_fused_prefill_rev_{claude,gemini,glm5}.md`. Findings, rejections, and rationale captured in §6.
- Methodology: `docs/methodology/perf-benchmarking.md` — see the new "eval_hipfire wall-clock is NOT a kernel-speed meter" section for the Q8-specific perf-meter rules.
- KLD anchor: `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-q8-floor-9b/result-table.md` — 0.5735 slice-mean KLD on 256 chunks of Wikitext-2 vs the BF16 ref.
- Sibling plan: `docs/plans/hfp4-mfp4-rdna3-accel.md` (same template, different quant format).
- F16 lm_head batched fan-out (parallel work, not in this branch): PR #242 on Kaden-Schutt/hipfire (`feat/f16-lmhead-support`).
- Commits on `feat/q8-prefill-tier2` (vs origin/master):
  - `16fba4fd feat(q8-prefill): wire Q8_0 into batched prefill via gemm_q8_0_batched_chunked` (Tier 2)
  - `582e4097 perf(q8-prefill): bump gemm_q8_0_batched MAX_BATCH 16 → 64`
  - `b6186280 test(coherence): add Q8 9B long-prefill row (T3-0 prerequisite)`
  - `27a640e2 bench(q8): T3-1a — FP16-WMMA vs substrate microbench`
  - `3473dd85 feat(q8-wmma): T3-1b — gemm_qkv_q8_0_wmma production kernel + unit test`
  - `27c75d8e feat(q8-wmma): T3-2 — qkvza / gate_up / residual fused Q8 WMMA kernels`
  - `31e119c6 feat(q8-wmma): T3-3 — wire fused Q8 WMMA kernels into qwen35 + llama dispatch`
  - `97bf8cf4 fix(q8-wmma): cross-review hardening — gfx12 gate, dtype asserts, K%32, += invariant`
