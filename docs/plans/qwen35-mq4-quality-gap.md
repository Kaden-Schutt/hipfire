# Hipfire quantization: where we are, what HFP4/MFP4 added, what's still missing vs Unsloth Dynamic, and the format roadmap

**Status:** consolidated design doc — supersedes prior revs of this file and `mfp4-vs-unsloth-dynamic-gap.md`.
**Date:** 2026-05-11
**Audience:** hipfire engineers settling on the next-years quantization format.
**Goal:** answer the single question — *"what's the current problem, and how do we get from here to a 4-bit format that beats Unsloth Dynamic 2.0 GGUFs in quality, on RDNA, across model architectures spanning Qwen2.5-VL through Gemma4?"*

The doc has three movements: (1) the journey MQ4 → HFP4 → MFP4 and the empirical noise wins per lever, (2) what's still missing relative to imatrix-calibrated dynamic GGUFs, with concrete per-lever quality estimates, (3) how the HFP4 wire format extends — without breaking changes — to absorb the missing levers, plus future-proofing for archs we haven't shipped yet.

---

> **⚠ 2026-05-11 / 2026-05-12 update — empirical picture is more nuanced than either §1.3 or §6 claimed.** Fivetide's PPL analysis (`docs/plans/hfp4-fivetide-rebuttal-perspective.md`) initially showed MFP4G32 producing +25–94% worse PPL than MQ4G256 on Qwen3.5 dense (which would have demolished §1.3's MSE-based extrapolation). On 2026-05-12 fivetide added KLD measurements on Qwen3.5-0.8B that **reverse the picture** — FWHT helps E2M1 on KLD, opposite to what PPL showed. The strategic recommendation in §6 ("commit to HFP4 for the next several years") is downgraded to **"format choice pending multi-metric empirical resolution"** — neither MFP4 nor MQ4 has sufficient evidence to be a confident default. The per-weight MSE story in §1.3 was the wrong yardstick; the PPL story alone is the wrong yardstick; KLD + downstream task metrics are the next yardsticks to add. Read the rebuttal doc alongside this one.

## 1. The journey so far (per-lever noise wins)

### 1.1 Hipfire quantization-quality levers — taxonomy

Any 4-bit quant format trades five levers against bits-per-weight and dequant cost:

| Lever | What it controls | Implementation in MQ4 | Implementation in MFP4 | Implementation in Q4_K_M | Implementation in UD-Q4_K_XL |
|---|---|---|---|---|---|
| **L1 — element format** | Codepoint allocation (8 / 16 codes, uniform vs log-spaced) | Linear INT4 + zero-point | E2M1 FP4 (log-spaced; 8 signed magnitudes) | Linear INT4 + zero-point | Linear INT4 + zero-point |
| **L2 — scale granularity** | How many scales per N elements | 1 FP32 scale + 1 FP32 min per 256 | 1 UE8M0 per 32 + 1 FP16 per row | 1 super-FP16 + 1 super-FP16 min + 8×(6-bit scale, 6-bit min) per 256 | Same as Q4_K_M |
| **L3 — rotation** | Outlier dispersion within scale scope | FWHT-256 (offline) | FWHT-256 (offline; MFP4 variant) | none | none |
| **L4 — scale fitting** | Per-block scale chosen by min-max vs weighted LS | min-max | min-max | weighted LS over 20 candidates (`make_qkx2_quants`) | Same as Q4_K_M |
| **L5 — bit allocation** | Per-tensor / per-layer mixed precision | Hardcoded K-map rules | Same K-map | none (uniform) | imatrix-driven per-tensor selection of Q3/Q4/Q5/Q6 |

We measured the L1/L2/L3 combination empirically on Qwen3.5-9B / Qwen3.6-A3B BF16 safetensors:

| Format | Effective bpw | Per-weight MSE (median) | Notes |
|---|---|---|---|
| Q4_0 (per-32 INT4, no rotation) | 4.5 | 9.5e-7 – 1.6e-6 | Reference for L2 only. |
| MQ4G256 (per-256 INT4 + FWHT-256) | 4.25 | 1.5e-6 – 2.9e-6 | L3 alone < L2 alone. **~2× MSE vs Q4_0.** |
| Q4_K_M (per-32 INT4 + super-scales + weighted LS) | 4.5 | not measured here | Reference for L2+L4. |
| **MFP4G32** (per-32 UE8M0 + FP16 row + E2M1 + FWHT-256) | 4.5 | **5.8e-7** (PR #225 commit) | L1+L2+L3. **~3–5× MSE vs MQ4.** |

The MFP4 commit message reports 5.8e-7 mean quantization error on Qwen3.5-9B, which is **5–30× better than MQ4** across the four tensors in our analysis. That single format change (PRs #224 + #225) captured three levers simultaneously.

### 1.2 What MQ4 had — and what MQ4 was missing

MQ4G256 was the first hipfire-native quantization format. Its strengths and weaknesses, restated as the lever taxonomy:

| Lever | MQ4 status | Why it underperformed |
|---|---|---|
| **L1 element format** | Linear INT4 | LLM weights are roughly log-normal post-norm; uniform spacing wastes codepoints on the dynamic-range tails. |
| **L2 scale granularity** | One scale per 256 | A single scale per 256-element block is **forced to span the largest sub-range** in the block. With per-32 sub-blocks, 8 separately-fit ranges capture cross-sub-block variation at ~3 dB SNR advantage. |
| **L3 rotation** | FWHT-256 (within-block) | Helps but bounded: ~3 dB SNR within a block, **antagonistic with L2** (FWHT equalizes within-block range that per-32 scales would exploit). |
| **L4 scale fitting** | naive `(max - min) / 15` | No weighted LS, no candidate search. ~1.2–1.5× MSE worse than `make_qkx2_quants`. |
| **L5 bit allocation** | Uniform with hardcoded K-map promotions | Promotes router/embeddings/output to Q8 and edge-FFN to MQ6, but **not data-driven** — no activation importance. |

The empirical 2× MSE gap vs Q4_0 is **L2 alone**. The 6.2× KLD gap vs unsloth UD-Q3_K_XL is the **combined L2+L4+L5** gap, with L5 being the dominant explainer at that level of disparity.

### 1.3 What HFP4/MFP4 closed (and what it kept)

HFP4G32 (PR #224) and MFP4G32 (PR #225) ship with the following design choices, which align almost exactly with the levers above:

**Closed:**
- **L1 — element format → E2M1.** Eight signed magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, log-spaced. Matches the OCP MXFP4 wire format. Better fit to log-normal weight distributions than linear INT4. **Estimated MSE win vs MQ4 L1: ~1.3–1.6×** (data-dependent, larger on tensors with heavy tails).
- **L2 — scale granularity → per-32 UE8M0 + FP16 row scale.** UE8M0 is an 8-bit power-of-2 exponent (`v_ldexp_f32` is free on RDNA). FP16 row scale carries cross-block outlier compensation. **Estimated MSE win vs MQ4 L2: ~2×** (the empirical Q4_0 gap, since L2 granularity is now equivalent).
- **L3 — rotation → optional FWHT-256 via `format_flags` rotation_kind bits.** MFP4 = HFP4 + rotation; HFP4 alone has no rotation. The kernel can read both because the rotation is offline (baked into the codes). **Estimated win: 1.0–1.3× on top of L1+L2** — smaller marginal benefit because L1+L2 already capture much of what L3 used to compensate for.

Multiplied: **L1 × L2 × L3 ≈ 2.6–4.2×.** The commit message's "5–30×" is the high end of that range plus what we don't yet have a clean lever for (the FP16 row scale captures a long-tail outlier behavior that none of the integer formats handle).

**Kept (i.e. not addressed by HFP4/MFP4):**
- **L4 — scale fitting.** HFP4's per-block UE8M0 is chosen by `ceil(log2(block_max / 6.0))` — a deterministic min-max-based rule, not weighted LS. The FP16 row scale uses `max_abs(row) / 6.0` — also min-max. **No change vs MQ4 here.**
- **L5 — bit allocation.** Same K-map rules from MQ4 (norms/embeddings/output/router/edge promotions). No activation-driven importance. **No change vs MQ4 here.**

So HFP4/MFP4 captured L1+L2+L3 — call this the **"per-weight format quality"** group of levers — but left L4+L5 — the **"per-tensor and data-driven"** group — entirely untouched. That's where Unsloth Dynamic 2.0 lives.

### 1.4 Hardware acceleration across RDNA and CDNA tiers

HFP4's RDNA-optimal choices (`v_ldexp_f32` for UE8M0 dequant, native FP8 WMMA on gfx1201, V_PERMLANE16 cross-lane broadcasts, VOPD dual-issue) are **not why HFP4 has lower MSE than MQ4**. Those choices make HFP4 *fast* on RDNA3/4. The MSE win comes from L1+L2+L3 — which would also be faster on Hopper / Blackwell, just not as fast as a hardware-native MXFP4 / NVFP4 there.

The harder question: **on old archs like gfx906 (CDNA1) with no WMMA, no MFMA-INT8 support relevant for compute, can HFP4/MFP4 be accelerated well enough to actually use?** This isn't decorative — gfx906 is the dev box, and we already have a mature `gemm_hfq4g256_residual_mmq_gfx906_*.hip` family that reaches **95% of stock llama.cpp** at HFQ4G256 prefill on Qwen3.5-9B. Losing that perf to ship HFP4 would be a regression.

#### gfx906 ISA capability for HFP4

| Capability | gfx906 | Used by HFQ4 today? | Used by HFP4 (proposed)? |
|---|---|---|---|
| `V_DOT4_I32_I8` (`__builtin_amdgcn_sdot4`) — 4×INT8→INT32 fused dot | **YES** | Yes (the 95%-of-llama.cpp path) | **Yes — same path** |
| `V_DOT8_I32_I4` — 8×INT4→INT32 fused dot | **NO** (gfx908+ only) | No | No (not available; not needed) |
| `V_DOT2_F32_F16` (`__builtin_amdgcn_fdot2`) — 2×FP16→FP32 fused dot | **YES** | Some FP16-fallback kernels | Alternative path (lower throughput) |
| `V_PK_FMA_F16` / `V_PK_MUL_F16` — packed FP16 ops | **YES** | Yes (epilogue + non-dp4a paths) | Yes (epilogue) |
| `v_ldexp_f32` — free UE8M0 dequant | **YES** (every RDNA/CDNA tier) | n/a (HFQ4 has no UE8M0) | **Yes** (the whole point of UE8M0) |
| LDS bandwidth | 60 CU × 8.16 TB/s aggregate (~ 13.6 GB/s per WG) | Yes | Yes |
| HBM2 bandwidth | 1024 GB/s peak | Yes | Yes |
| MFMA (FP/INT matrix-fused) | gfx906 has MFMA-FP16 / MFMA-FP32 but **not relevant** because the dot product path goes through INT8, not FP-MFMA | No | No |
| WMMA | NO (gfx11+ only) | No | No |

The decisive fact: `__builtin_amdgcn_sdot4` is identical on gfx906 and gfx1100. The HFQ4 kernel already gets 95% of llama.cpp using it. The question is whether HFP4's E2M1 element format **forces** a slower path.

#### The FP4 → INT8 → V_DOT4 trick (proven by llama.cpp's MXFP4)

llama.cpp has a working MXFP4 (= OCP E2M1, byte-identical to HFP4's element format) implementation on the **same dp4a infrastructure** used for IQ4_NL, Q4_K, etc. The pattern from `ggml/src/ggml-cuda/mmq.cuh:867`:

```c
const int aux_q4 = get_int_b1(bxi->qs, kqsx);
const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
// v.x, v.y now contain INT8 LUT-decoded codes
// ... then standard dp4a accumulation:
sumi = __dp4a(v.x, y_int.x, sumi);
sumi = __dp4a(v.y, y_int.y, sumi);
// per-block UE8M0 scale folded in at scale-collection stage:
x_df[i] = ggml_cuda_e8m0_to_fp32(bxi->e) * 0.5f;
```

Where `kvalues_mxfp4 = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}` is exactly `2 × E2M1_LUT` in INT8. The 2× factor is folded into the per-block scale at row-finalize.

**This is identical to what HFP4 needs on gfx906.** The MFP4-with-rotation case is even simpler: the FWHT is offline, so the kernel sees plain HFP4G32 bytes and treats them the same. The only kernel-side diff from HFQ4G256 → HFP4G32 on the gfx906 mmq path is:

1. Replace the `(n - 8) * sc` symmetric unpacking with `kvalues_hfp4[n & 0xF] * sc` (one LDS load per nibble — already LDS-cached because we initialize a 16-entry LUT once per block, exactly like the generic HFP4 kernel at `kernels/src/gemv_hfp4g32.hip:38–47`)
2. Per-block UE8M0 byte instead of per-256 FP32 scale — fold via `v_ldexp_f32` at the block-scale collection point
3. Per-row FP16 scale: one FP16 multiply at row finalize (currently the FP32 scale × FP16 epilogue path; cost is one `v_pk_mul_f16` per output row, amortized across K)

#### Bandwidth math (gfx906, HFP4G32 vs HFQ4G256 on MMQ)

The actual concern on gfx906 is **scale-load bandwidth**: HFQ4G256 has one FP32 scale + FP32 zero-point per 256 K-elements (8 B / 256 = 0.25 bpw scale overhead). HFP4G32 has UE8M0 per 32 K-elements + 16-byte row header (17 B / 32 + 16/K ≈ 4.25 + 0.025 bpw scale overhead).

Per 256 K-elements:
- HFQ4G256: 128 B nibbles + 8 B scale = **136 B**
- HFP4G32: 128 B nibbles + 8 × 1 B UE8M0 = **136 B per group**, plus 16 B / K_row row header (~0.025 bpw)

**Byte-for-byte the same per-256.** HFP4G32's per-row header adds <0.6% on K=2048+ tensors (Qwen3.5-9B `q_proj` row is 2736 B vs MQ4's 2720 B per row — already measured in the HFP4 spec). On gfx906's 1024 GB/s HBM2, this is **negligible**.

The real cost is **8 scale-broadcasts per group** (UE8M0 every 32) vs **1 scale-broadcast per group** (HFQ4 every 256). On the gfx906 mmq kernel, scale broadcast happens via LDS read once per K-iteration of 32 elements anyway — HFQ4 already pays this cost because its per-256 scale must be replicated 8× in LDS for the dp4a tile-stride. **The scale-load topology is identical.** What differs is which byte the LDS holds: an FP32 scale value (HFQ4) vs a UE8M0 byte that goes through one `v_ldexp_f32` op (HFP4). One free instruction per 32 elements.

#### Realistic perf estimate for HFP4G32-gfx906 MMQ kernel

Porting the existing `gemm_hfq4g256_residual_mmq_gfx906_*.hip` family to HFP4G32 (the rotation-free variant; MFP4G32 is the same kernel with offline-rotated weights) requires:

- Replace `(n - 8) * sc` with `kvalues[n] * sc` — same VALU count, +1 LDS read per nibble (covered by LUT in LDS)
- Replace per-256 FP32 scale load with per-32 UE8M0 + `v_ldexp_f32` — same LDS traffic, +8 VALU ops per group
- Add per-row FP16 scale at epilogue — 1 `v_pk_mul_f16` per output column, negligible
- Keep existing nwarps=4, mmq_x ∈ {8..64} runtime dispatch, window streaming, b128 LDS reads, +1 bank-conflict pad — all the techniques the current kernel uses

**Estimated perf vs current HFQ4G256-gfx906 MMQ (95% of llama.cpp):**

| Workload | Current HFQ4G256 | Projected HFP4G32 | Why |
|---|---|---|---|
| pp512 prefill | 714 tok/s (95% of stock) | **620–680 tok/s** (87–95% of stock) | +8 VALU ops/group fixed cost; LDS LUT slightly increases LDS pressure; expected ~5–8% throughput hit, no fundamental ceiling change |
| pp128 prefill | 598 tok/s | **520–580 tok/s** | Same as above |
| decode tok/s | ~60 tok/s (BW-bound) | **~60 tok/s** (BW-bound) | Identical bytes-per-weight; decode is purely BW-limited |

**Quality lift more than compensates.** A 5–8% prefill regression to get 3–5× lower per-weight MSE (per §1.3) is a clear net win for any user-facing metric — and that's before adding L4 (weighted-LS) which gets us further at zero kernel cost.

If the prefill regression turns out worse than projected, two fallbacks:

1. **Wider tile / coarser K-iter.** The HFQ4 kernel uses 128-K-element iter-K windows. Going to 256-K windows reduces per-group VALU overhead at the cost of larger LDS tile. The current kernel sits at 30.7 KiB/WG (2 WGs/CU); HFP4's added scale ops + LUT raise this to ~32 KiB which is exactly the budget cap — workable but tight. A 256-K window would drop us to 1 WG/CU which historically regresses ~12%. Probably not worth it.
2. **Adopt llama.cpp's IQ4_XS pattern.** IQ4_XS uses *one super-block of 6-bit scales* per 256 elements (not per-32) — identical scale density to HFQ4G256, with LUT-decoded codes. It's the precise sweet spot for "low-arch + non-uniform 4-bit." This is the **plan B** if HFP4G32 on gfx906 turns out to underperform: reserve a HFP4XSG256 variant (16 codes E2M1 + super-block of per-32 INT6 scales + per-256 super-FP16) using qt=27 or qt=28, identical LDS topology to HFQ4G256.

#### gfx906 verdict

**HFP4G32 / MFP4G32 are realistically accelerable on gfx906**, with an estimated 5–8% prefill regression vs current HFQ4G256 in exchange for 3–5× lower per-weight MSE. The decode path is bandwidth-bound and unchanged. The dp4a inner loop is **byte-identical** between HFQ4 and HFP4 — the LUT-decode trick is proven by llama.cpp's MXFP4 in production on the same dp4a infrastructure.

**Engineering cost:** porting one or two of the existing eight `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip` files to HFP4G32, plus the corresponding `fused_qkv_hfp4g32_wave64_dp4a.hip` and `fused_gate_up_hfp4g32_wave64_dp4a.hip` (mirroring the existing wave64_dp4a kernels). **Estimated 2–3 weeks of focused gfx906 kernel work** by an engineer already familiar with the HFQ4 kernel family (the patterns transfer 1:1).

**Decision implication:** the Phase A roadmap (§5) does *not* need to wait on the gfx906 kernel port — Phase A is quantizer-side only. The gfx906 port is a parallel track that lands when ready. Until it lands, gfx906 users continue using HFQ4G256 / MQ4G256 (no regression vs today). Once it lands, they get the quality lift.

This also means **the format roadmap (§3) is correctly bet on HFP4 even from a gfx906 perspective.** Older archs are not blocked from the quality lever; they pay a small (5–8%) throughput tax for it, and an HFP4XSG256 plan-B variant is available if even that turns out too costly.

### 1.5 What the 9B cohort actually measured (2026-05-11) — sharpening the strategic picture

The Phase A Step 0.5 cohort (`benchmarks/quality-baselines/results/2026-05-11-cohort-phase-a-step-0.5/`) ran the three baselines `{MQ4G256, HFP4G32 unrotated, MFP4G32}` on Qwen3.5-9B at quick-slice (256 chunks) under canonical methodology (prefill mode, asym3 KV, gfx1100). Result:

| variant | MSE mean (4-bit qts aggregate) | KLD vs BF16 | PPL | Δ vs MQ4 |
|---|---:|---:|---:|:---|
| MQ4G256 | 6.62e-6 | 0.8084 | 15.16 | reference |
| HFP4G32 (unrotated) | 3.15e-6 | 0.9763 | 18.68 | KLD +20.8%, PPL +23.2% |
| MFP4G32 | **2.98e-6** | **1.1116** | **21.02** | KLD +37.5%, PPL +38.6% |

At first read, this looked like a clean reproduction of fivetide's "NRMSE paradox": MFP4 wins per-weight reconstruction (2.98e-6, half of MQ4's 6.62e-6) and loses model quality (worst KLD, worst PPL).

**On closer inspection (E2 + E5 sanity-check pair, no GPU), the paradox is not a paradox.** Two findings change the interpretation:

#### 1.5.1 The aggregate MSE win for MFP4 is a measurement artifact

The cohort's "MSE mean (4-bit qts)" averages over each variant's 4-bit-quantized tensors. But MQ4 quantizes **273 tensors** as MQ4G256, while HFP4/MFP4 quantize only **249** as HFP4G32/MFP4G32. The 24-tensor difference is precisely the LinearAttention `conv1d.weight` tensors (each `[8192, 1, 4]` = 32 768 params).

The quantizer routes these defensively: HFP4G32 wire format requires `K % 256 == 0` (the gemv kernel constraint at `crates/hipfire-quantize/src/main.rs:753`), and conv1d has K=4. So in HFP4/MFP4 they fall back to **HFQ4G128** (per-128 INT4 + FP16 scale, no rotation, no ZP) — a different format that **does not appear in the "4-bit qts" aggregate row.**

In MQ4 the same conv1d tensors are forced through MQ4G256 (group-256 + FWHT) and produce **mass-weighted MSE 4.84e-5** — the worst class in the model by a factor of 8×. They single-handedly drag MQ4's aggregate from "actual value" up to 6.62e-6. Strip them out and MQ4's aggregate over the 249 common tensors falls to about **4.6e-6** (rough recompute from the per-class table).

So the apples-to-apples per-tensor MSE comparison on the *same 249 tensors* is closer to:
- MQ4: ~4.6e-6
- HFP4: 3.15e-6  
- MFP4: 2.98e-6

MFP4 still wins per-tensor MSE on this subset, but the win is **~35% (4.6 vs 2.98)**, not the **2.2×** (6.6 vs 2.98) the cohort table suggested. The conv1d routing inflated the gap.

#### 1.5.2 Per-class MFP4-vs-MQ4 ratios are nearly constant at 1.14-1.15× on dominant-mass weights

Restricting to **matched tensors** (same tensor name, same K-dim) across MQ4 and MFP4 in the top-50-worst-MSE pool, the MFP4/MQ4 ratio across heterogeneous classes is:

| class | matched-tensor count | MFP4/MQ4 per-tensor MSE ratio |
|---|---:|---:|
| `self_attn.q_proj` | 1 | 1.14× |
| `self_attn.v_proj` | 4 | 1.15× |
| `self_attn.o_proj` | 3 | 1.15× |
| `linear_attn.in_proj_qkv` | 5 | 1.14× |

A **near-constant 1.14-1.15× across four heterogeneous tensor classes** is the signature of a format characteristic — not a quantizer bug. A bug would produce outliers or per-class variance. A consistent multiplier across types says: E2M1 + UE8M0 + FP16-row is fundamentally ~14% noisier than INT4 + FP32-affine at g=256 on these tensor classes.

The aggregate MSE win for MFP4 came from **conv1d routing alone**, not from L1+L2 format superiority on the dominant-mass weights. On the weights that actually matter for forward-pass output (attn + MLP projections), **MFP4 is ~14% worse per-tensor MSE than MQ4**.

#### 1.5.3 What this does to the §1.3 framing

§1.3 estimated `L1 × L2 × L3 ≈ 2.6–4.2× MSE win` for MFP4 over MQ4. The cohort says: on like-for-like tensors, MFP4 is **1/1.14 ≈ 0.88× MSE win** — i.e., a **per-tensor MSE *loss***, not a win. The lever analysis predicted multiplicative wins; the data shows L1 (E2M1 vs INT4) is a per-tensor *regression* on post-FWHT weights at g=32. L2 (per-32 UE8M0 + FP16-row) doesn't compensate; L3 (rotation, shared between MQ4 and MFP4) is neutral on the comparison.

This isn't a small correction — it inverts the §1.3 strategic claim. The "L1+L2+L3 captured" framing predicted MFP4 was the better baseline format. The data says MQ4 is the better baseline format, and MFP4 made things worse on both per-tensor MSE *and* model quality.

#### 1.5.4 Decomposing the +38% PPL gap

If MFP4 is ~14% worse per-tensor MSE on dominant weights, and ~38% worse on PPL, the rest of the gap comes from **noise-shape mismatch**:

| component | est. contribution | mechanism |
|---|---|---|
| Per-tensor MSE deficit on dominant weights | ~14 percentage points | E2M1 codebook fits sub-Gaussian rotated weights ~14% worse than INT4-with-ZP at g=256. The Lloyd-Max distortion analysis in fivetide's doc §3 predicts this directly: post-FWHT kurtosis ≈ 2.82 (sub-Gaussian), and on sub-Gaussian distributions, uniform-INT4 is closer to Lloyd-Max optimal than log-spaced E2M1. |
| Noise-shape mismatch | ~24 percentage points | E2M1 errors cluster near zero (log-spacing has tightest codepoints near zero), creating systematic perturbation that doesn't average out through the transformer stack. INT4 errors are uniformly distributed on a sub-Gaussian distribution, look Gaussian to downstream layers, average out as noise rather than bias. The exact decomposition is approximate but matches the fivetide-rebuttal mechanism. |

Both components are **real format effects**, not bugs. The cohort's bug-hunt pair (E2 + E5) ruled out:
- Localized broken tensors (no MFP4 outliers)
- Wrong-class quantization (only conv1d differs, and it's *better* in MFP4's routing)
- Per-class variance (1.14× is uniform across classes — bug-shaped patterns would have variance)
- Catastrophic kernel-path or activation-rotation issues (forward pass produces coherent text; coherence-gate ran clean at PR #235 merge)

#### 1.5.5 Strategic implications

The "NRMSE paradox" in §"hfp4-fivetide-rebuttal-perspective.md" was framed as: *MFP4 wins per-weight MSE but loses model quality, demonstrating that MSE doesn't predict quality.* The cohort says: **once you control for the conv1d routing confound, MFP4 doesn't even win per-weight MSE on dominant weights.** It loses on both axes. The framing of "MFP4's format wins, but in a way that's invisible to model quality" — sharpened — is closer to: **MFP4's format engineering went in the wrong direction on every measurable axis once you fix the comparison.**

This **strengthens** the Path A bet (calibrated-MQ4) from the rebuttal doc:

- Path A (imatrix-calibrated MQ4G256) — adds the missing L4+L5 levers to the format that has both better per-tensor MSE AND better model quality on dominant weights. Calibration is added on top of a winning baseline.
- Path B (imatrix-calibrated MFP4G32 with FP16 block scale + ZP) — adds L4+L5 on top of a format that starts ~14% per-tensor MSE worse on dominant weights AND ~38% PPL worse. Calibration would have to overcome both deficits *plus* close the residual gap to UD-Q4_K_XL.

The §2.4 prediction "we sit a couple levers behind Unsloth at equal bpw" was built on the assumption that MFP4 was already a wash with Q4_K_M on per-tensor quality. The cohort says MFP4 is *behind* MQ4 on per-tensor quality, and MQ4 was already behind Q4_0 on per-tensor quality (§1.2: ~2× MSE vs Q4_0). The cumulative pre-calibration deficit vs Q4_K_M may be larger than §2.2's per-lever estimates implied.

The Phase A roadmap (§5) is unchanged in structure — Steps 4+5 (imatrix collect + activation-weighted LS) are still the right next work. What changes is the format the calibration eventually rides on. The cohort suggests calibrated-MQ4 should be the explicit primary track, with calibrated-MFP4 as a secondary measurement to confirm the format-level disadvantage doesn't reverse under calibration.

#### 1.5.6 Method caveat for future cohorts

The conv1d routing confound discovered here is a methodology lesson, not a cohort bug. Future cohorts should either:
- Force *all variants* to use the same quantization scheme for tensors that violate the wire-format alignment constraint (e.g. extend HFP4G32 to support K=4 via padding, or force MQ4 to route conv1d through HFQ4G128 too), or
- Report aggregate MSE both with and without conv1d (the structural-confound tensors) so the comparison is unambiguous.

The cohort table's "MSE mean (4-bit qts)" column is **misleading** under the current implementation when comparing across variants with different wire-format alignment requirements. Recommend renaming to "MSE mean (variant's primary 4-bit qt)" with an asterisked footnote about which tensors fall in the comparison.

---

## 2. What's still missing vs imatrix-calibrated dynamic GGUFs

### 2.1 The three Unsloth levers

Stripped of marketing, "Unsloth Dynamic 2.0" is three independent levers stacked on top of Q4_K_M:

**(A) Q4_K_M per-tensor format quality** (= our L1+L2+L4):
- Per-32 sub-block 6-bit linear scales + super-FP16 scale-of-scales
- 6-bit per-sub-block mins (asymmetric)
- Weighted least-squares scale search (`make_qkx2_quants`, 20 candidates)
- No rotation

**(B) Per-tensor format mixing** (= our L5, mechanism only):
- `llama-quantize --tensor-type "regex=quant"` — regex selects tensors; per-match quant override
- Pre-set flags for embeddings (`--token-embedding-type`), output (`--output-tensor-type`)
- Used to demote `attn_q/k` on alternating layers to Q3_K and promote `attn_v`/`ffn_down` to Q5_K/Q6_K

**(C) imatrix activation-aware calibration** (= our L5, *data-driven decision*):
- Forward-pass a calibration corpus (Unsloth uses Calibration_v3/v5, >1.5M tokens of hand-curated data) through the BF16 model
- Per linear-layer input dimension, record `Σ_token (x[token, i])²` — squared activation
- During quantization: weighted LS becomes `min Σ_i act²[i] · (w[i] - q[i])²` instead of `min Σ_i (w[i] - q[i])²`
- Per-tensor statistics: `Σ(Act²)`, `% Active`, `Entropy`, `ZD Score` (Liu et al. 2024), `CosSim` to previous layer
- The decision *which tensors get Q3 vs Q4 vs Q5 vs Q6* uses these statistics, not hardcoded rules

### 2.2 Per-lever quality estimates relative to MFP4 baseline

These are estimated improvements **on top of MFP4G32** — they are not additive with the MFP4 wins above.

| Lever | What it adds | Estimated PPL Δ at fixed bpw | Estimated KLD-mean Δ | Effort | Notes |
|---|---|---|---|---|---|
| **L4a — weighted LS for UE8M0** | Replace `ceil(log2(block_max/6))` with a small candidate search over `block_e ∈ {e_ideal-1, e_ideal, e_ideal+1}` + brute-force re-rounding | −3% to −7% | −10% to −20% | ~1 week | Pure quantizer-side. Format unchanged. Easy win — the current UE8M0 chooser is intentionally simple. |
| **L4b — weighted LS for FP16 row scale** | Solve for `row_scale_a` minimizing post-block-quantization MSE instead of `max_abs/6.0` | −2% to −5% | −5% to −15% | ~3 days | Pure quantizer-side. Format unchanged. Probably should ship with L4a. |
| **L5a — `--tensor-type regex=quant` CLI** | Generalize the hardcoded K-map rules to a CLI/config-driven regex matcher; ship K-map presets as default configs | 0% (mechanism only) | 0% (mechanism only) | ~3 days | Unblocks A/B experiments without rebuilding the quantizer. Format unchanged. |
| **L5b — imatrix collection** | Forward-pass corpus through BF16 model; dump `Σ act²` per linear-layer input dim to sidecar `.imatrix` file | 0% (collection only) | 0% (collection only) | ~1 week | New `imatrix_collect` example; ~1M-token corpus (wikitext-103-test + humaneval prompts + slice of multi-turn chat). |
| **L5c — activation-weighted LS in quantizer** | Use the collected imatrix to switch L4's LS objective to importance-weighted | −5% to −15% | **−40% to −70%** | ~2 weeks | The dominant lever. Median MSE may move ~1.2× — but **p99 MSE on hot input dimensions** moves dramatically, which is what KLD measures and what attractor-style coherence failures see. |
| **L5d — imatrix-driven per-tensor bit allocation** | Use per-tensor `Σ(Act²)` / `% Active` / ZD score to pick HFP4G32 / HFP6G32 / Q8 per tensor instead of hardcoded K-map | −3% to −10% | −15% to −30% | ~2 weeks (after L5b/c) | Replaces K-map rule 4/5 with data-driven promotion. Bit budget held constant. |
| **L5e — sweep `Q3_K`-equivalent demotion** | Once L5d is in place, allow demoting low-importance tensors below 4-bit (HFP3 variants in the HFP family, or fall back to HFP4 at coarser group size) | +0 to +5% (quality-neutral at lower bpw) | 0% | ~1 month | Requires HFP3 kernel work. Defer until L5b/c/d are in. |

Multiplied through, L4a/b + L5b/c on top of MFP4G32 should land at:
- **PPL within 1–3% of Q4_K_M + imatrix at the same bpw**
- **KLD-mean within 1.2–1.5× of Unsloth UD-Q4_K_XL** (vs current 6.2× of UD-Q3_K_XL — i.e. expected to *beat* UD-Q3_K_XL outright and approach UD-Q4_K_XL parity)

The rotation lever (L3) is the one place hipfire **should beat** Unsloth at equal bits-per-weight on the *same* calibration corpus: FWHT-256 provides ~1.0–1.3× MSE on top of an otherwise-equivalent format, and that win compounds with everything in §2.2.

### 2.3 Why imatrix is the dominant lever

It's worth explaining *why* L5c specifically dominates, because it's not obvious.

A per-tensor MSE of 1e-6 sounds tiny — but it's not uniformly distributed across input dimensions. The same MQ4-quantized router has:
- ~95% of input dims at MSE near the median (~1e-6)
- ~3–5% of input dims at MSE 10–100× the median (the "hot" dims that carry most of the activation energy on real prompts)

KLD measures the divergence of the *output logit distribution* from the FP16 reference. The output is dominated by hot dims. A quant that's median-good but p99-bad is exactly what produces:
- **MQ4 mean KLD 0.876 vs UD-Q3_K_XL mean KLD 0.141** (our Phase 10 data) — at lower bits but with imatrix
- The block-attractor spiral we saw in Phase 11 of `qwen35-moe-coherence-investigation.md`

imatrix-weighted LS specifically targets the p99: it puts more quantization budget on the dims the corpus actually exercises, and accepts higher error on dormant dims. The mean MSE only moves modestly; the p99 KLD moves a lot.

This is also why MFP4 didn't close the A3B spiral in Phase 11 — MFP4 improves median MSE (per-weight), but doesn't differentially protect hot input dimensions. The spiral lives at the p99 cliff.

### 2.4 Where we sit today vs Unsloth Dynamic at equal bpw

| Aspect | Hipfire (MFP4G32 + K-map alternating) | Unsloth UD-Q4_K_XL | Gap |
|---|---|---|---|
| Element format | E2M1 (log-spaced) | INT4 linear | **+** (we win L1) |
| Per-32 scales | UE8M0 + FP16 row | super-FP16 + 8×6-bit | wash |
| Rotation | FWHT-256 | none | **+** (we win L3) |
| Scale fitting | min-max | weighted LS | **−** (they win L4) |
| Per-tensor mixing | hardcoded K-map | regex / imatrix-driven | **−−** (they win L5) |
| Calibration | none | 1.5M-token corpus | **−−** (they win L5) |

We are roughly 2 levers behind. Closing the 2 levers is **2–4 weeks of focused engineering, no new fundamental research.** The pieces (imatrix collection, weighted LS, per-tensor selection) are all 2024-era public techniques.

---

## 3. Format roadmap: extending HFP4 to absorb the missing levers

The whole question is whether we have to invent a new quant family — or whether HFP4 already has the extension points. Reviewing the HFP4 wire format (see `docs/quant-formats/hfp4.md` for the spec):

### 3.1 What HFP4 reserved for future use

The per-row header is 16 bytes:

```
+0  : f16  row_scale_a       // primary FP16 second-level scale
+2  : f16  row_scale_b       // dual-output scale (fused gate+up, qkv)
+4  : u16  block_count       // K/32
+6  : u8   format_flags      // bit 0: rotation present
                              // bit 1: row_scale_b used
                              // bits 2-3: rotation_kind (00..11)
                              // bits 4-7: reserved
+7  : u8   reserved
+8  : u32  reserved          // future: D_diag pointer offset (joint-D smoothing)
+12 : u32  reserved          // future use
```

Specifically:
- **`format_flags` bits 4-7 (4 bits)** — 16 future flags
- **`format_flags` bits 2-3 rotation kind** — 4 rotation modes; `01` shipped (offline FWHT-256), `10`/`11` reserved (online block-diag-128, HadaCore-16)
- **`reserved` u8 at +7** — 256 future flag values
- **`reserved` u32 at +8** — explicitly earmarked for joint-D smoothing (a per-channel pre-multiplier vector that lives in a sidecar buffer, pointed at by this offset)
- **`reserved` u32 at +12** — fully unallocated
- **Quant-type IDs 21–29** reserved (currently using 21=HFP4G32, 24=MFP4G32; IDs 22, 23, 25–29 reserved for ablations and v2/v3 variants)

### 3.2 How each missing lever maps onto the existing format

| Lever | Format change required | Wire-format break? |
|---|---|---|
| **L4a/b — weighted LS for UE8M0 + row scale** | **none.** Pure quantizer-side. The kernel reads the same bytes; only the chosen `block_e` / `row_scale_a` values change. Existing HFP4 files coexist with weighted-LS-fit HFP4 files at identical IDs. | **No.** |
| **L5a — regex CLI for per-tensor format** | Per-tensor metadata: which quant_type was chosen. Already present in `.hfq` tensor index (`quant_type: u8` per tensor). The CLI is pure quantizer-side. | **No.** |
| **L5b — imatrix collection** | Sidecar `.imatrix` file. Not part of `.hfq` at all — used only at quantize time. | **No.** |
| **L5c — activation-weighted LS** | Pure quantizer-side again. The format that comes out is byte-compatible with HFP4G32 / MFP4G32; the *content* differs (better-fit scales/codes). | **No.** |
| **L5d — per-tensor bit allocation** | Already supported via per-tensor `quant_type`. We just need more HFP family members (see below). | **No.** |
| **L5e — HFP3 / HFP4@G64 / HFP6 variants** | Each new variant gets one of the reserved quant_type IDs 22-29. The byte layout is essentially the same family with different element format / group size / row-scale dimensions. | **No** if we use reserved IDs; otherwise yes. |

**Conclusion: every Unsloth-tier lever fits inside the existing HFP4 wire format without a single breaking change.** L4 + L5a–c are pure quantizer-side improvements that produce byte-compatible HFP4G32 files. L5d–e use already-reserved quant_type IDs.

### 3.3 Extension proposals for HFP family

To round out the family for Unsloth-equivalent per-tensor bit allocation:

| Proposed | ID | Spec | Use case |
|---|---|---|---|
| HFP3G32 | new (e.g. 30) | E2M0 (no mantissa; 4 magnitudes) or 3-bit linear + UE8M0 + FP16 row | Demotion target for low-importance tensors per L5d |
| HFP6G32 | new (e.g. 31) | INT6 + UE8M0 + FP16 row, packs as 6-bit nibbles | Promotion target for hot tensors |
| HFP4G64 | 23 (reserved) | E2M1 + UE8M0 g64 + FP16 row | Coarser scale granularity for embedding / lm_head where row-scale dominates |
| HFP4G16 | 22 (reserved) | E2M1 + UE8M0 g16 + FP16 row | Finer scale granularity for outlier-heavy tensors |
| HFP8E4M3G32 | 27 (reserved) | E4M3 + UE8M0 g32 | Q8-equivalent for routers, embeddings, lm_head |
| **HFP4G32W** | new (e.g. 32) | HFP4G32 + per-block 8-bit weight `w_block` byte in payload (signals importance to a hypothetical importance-aware dequant) | **Speculative** — not needed for L4/L5 closure. Listed for completeness. |

L5d's "per-tensor bit allocation" using just the existing ID set (HFP3G32, HFP4G32, HFP4G16/G64, HFP6G32, HFP8E4M3G32) gives ~5 effective bit-width slots — more than Unsloth's Q3/Q4/Q5/Q6/Q8 mix.

### 3.4 Rotation extension — already partially designed

The `rotation_kind` bits in `format_flags` reserved three future modes:
- **`10` — online block-diag-128** (AMD's recipe; requires Stiefel-manifold-calibrated rotations and fused-rotation kernels for QKV/QKVZA/gate_up — v3 roadmap)
- **`11` — HadaCore-16** (16×16 Hadamard via WMMA fragments; gfx1201 only — research)
- (`00` = none, `01` = offline FWHT — both shipped)

If the imatrix work in §2.2 reveals that calibrated rotations (QuaRot-style optimal Hadamards instead of fixed seeds 42/1042) are a meaningful additional lever, that fits into rotation_kind `10` with an `R` matrix stored per layer in a sidecar. **No wire-format break.**

### 3.5 What can't be done inside HFP4

There are two structural limits worth naming:

1. **Block sub-32 elements.** The per-block `block_e` byte is 1 byte / 32 elements = 0.25 bpw overhead at g=32. Going to g=16 doubles that to 0.5 bpw — substantial. HFP4G16 ships as an ablation slot, but for a "go finer than per-32" format we'd want a separate spec.
2. **Non-row-shaped weights.** The 16-byte row header assumes row-major matmul weights. Conv weights or batched embeddings shaped `[batch, in, out]` would either consume a row header per (batch, in) plane — wasteful — or need a different wire format. **This is the most likely future incompatibility.** Not relevant for transformer LLMs.

Neither is a near-term blocker.

---

## 4. Future-proofing for archs we haven't shipped yet

Hipfire supports Qwen3.x, Qwen3.x-VL, Llama families today. The user asked specifically about Gemma (newer than Qwen3 in design) and Qwen2.5-VL (older). The format's per-arch concerns are orthogonal to L1–L5; they're about *what gets quantized, where it lives, and how the engine matches tensors to handlers.*

### 4.1 What changes across architectures

| Concern | Qwen3.x (current) | Qwen2.5-VL (older) | Gemma 2/3/4 (newer) | Other (Llama, Mistral, …) |
|---|---|---|---|---|
| Norm type | RMSNorm (`w * rms`) | RMSNorm | **GemmaRMSNorm** (`(1+w) * rms`) | RMSNorm |
| Norm storage | bare `w` | bare `w` | **bare `w` or pre-baked `(1+w)`** depending on source | bare `w` |
| Attention | GQA + DeltaNet hybrid | GQA + vision encoder | GQA, sometimes softcapping | GQA / MQA |
| MoE | Yes (A3B/A22B) | No | No (so far) | Some (Mixtral) |
| Tokenizer | tiktoken-style BPE | Same | SentencePiece | Mixed |
| Vision encoder | ViT in qwen35-vl | ViT (different patch size, different fusion) | none (Gemma1) / native multimodal (Gemma 3+) | Llama-vision pipelines |
| Tensor names | `model.language_model.layers.X.…` | `model.layers.X.…` or `transformer.h.X.…` | `model.layers.X.…` with different qkv shape conventions | mixed |
| Special tensors | router, shared_expert_gate | none | per-layer logit softcap scalar | none |

The HFP4 wire format is **architecture-agnostic at the per-tensor level** — a row of floats, quantized, stored. The architectural complexity lives in **(a) tensor naming**, **(b) per-tensor classification rules** (currently the K-map), and **(c) loader code that wires tensors to model parts.**

### 4.2 What needs to be future-proofed in the format

Three concrete extensions to keep the `.hfq` format from accumulating arch-specific cruft:

**(i) Per-tensor metadata sidecar in `.hfq`.** Currently per-tensor info is `(name, quant_type, shape, group_size, data_offset, data_size)`. Adding a free-form `metadata_json` per tensor — analogous to the file-level metadata — would let arch-specific loaders carry hints without burdening the format:
- Gemma's `(1+w)` baking flag for norms
- Softcap scalars stored alongside attention weights
- Activation-importance summary (`Σ act² mean/p99`) used by the imatrix-driven loader to pick KV-cache precision

**(ii) Tensor-class normalization.** Currently `kmap_resolve_mode` in `crates/hipfire-quantize/src/main.rs` greps on substrings like `"down_proj"`, `"v_proj"`, `"mlp.experts."`. This breaks the moment a HuggingFace export changes the naming convention (Qwen2.5-VL uses `transformer.h.X.attn.c_attn` for fused QKV; Gemma uses `model.layers.X.self_attn.qkv_proj`). The robust fix:

- Define a **canonical tensor-class enum** in `crates/hipfire-quantize/src/tensor_class.rs`:
  ```
  pub enum TensorClass {
      EmbedTokens, LmHead, OutputNorm,
      AttnQ, AttnK, AttnV, AttnQKVFused, AttnO,
      FfnGate, FfnUp, FfnDown,
      MoeRouter, MoeSharedExpertGate, MoeExpertGate, MoeExpertUp, MoeExpertDown,
      LayerNorm, RmsNorm, GemmaRmsNorm,
      VisionPatchEmbed, VisionAttn, VisionFfn, VisionMerger,
      SoftcapScalar, RouterLogitsBias,
      Unknown,
  }
  ```
- Per-arch classifier function: `fn classify(name: &str, arch: ModelArch) -> TensorClass`. Currently this logic is implicit in K-map substring matches and engine-side loader code; explicit centralization is the future-proofing.
- K-map rules and imatrix-driven bit allocation work on `TensorClass`, not on raw names. Adding a new arch becomes "add a new `classify` arm" — no format change, no quantizer logic change.

**(iii) Per-arch reserved fields in HFP4 row header.** The reserved u32 at offset +12 in the HFP4 row header can hold a per-tensor opaque token; the engine's arch-specific loader interprets it. Examples:
- Gemma: pre-quantized `(1+w)` or bare `w` flag for norm rows
- Vision: spatial dimension hints for non-1D-row weight shapes
- MoE: expert-index-in-stack for the 3D expert tensors we use today

The format already has the bytes; we just need a convention.

### 4.3 Adding Gemma and Qwen2.5-VL: concrete shopping list

What it actually takes to add these arches, given the format roadmap above:

**Gemma 3/4 (newer, simpler-than-MoE):**
- Add `ModelArch::Gemma3` to `crates/hipfire-runtime/src/arch.rs`
- Implement `GemmaRMSNorm` correctly (`(1+w) * rms`) — this is **already correct in hipfire** since PR #228 fixed the MoE final-norm convention; same code path applies
- Tensor-class classifier handles HF's `model.layers.X.self_attn.qkv_proj` (fused QKV)
- Softcap scalar (Gemma 2 has it; Gemma 3 dropped it) gets a per-arch metadata blob via §4.2(i)
- No HFP4 format change needed

**Qwen2.5-VL (older, more naming inconsistency):**
- Add `ModelArch::Qwen25VL` to arch enum
- Tensor-class classifier handles `transformer.h.X.attn.c_attn` (HF's older fused QKV name)
- Vision tower: same `ViT` patterns as qwen35-vl but with different patch size; quantize as a separate sub-graph (the K-map already excludes `visual.` tensors from MoE rules)
- No HFP4 format change needed

**Estimated effort for each:** 1 week of arch crate work + 2 days of quantizer K-map / tensor-class additions. The format is ready.

---

## 5. The plan — concrete sequencing

This is the ordered roadmap that gets us from where we are (MFP4 ships, A3B spiral exposed, 6.2× KLD gap to Unsloth) to *beating* Unsloth Dynamic 2.0 at the same bpw on RDNA.

### Phase A — Quantizer quality, no format change (5–7 weeks)

**Framing update (2026-05-12).** Per the rebuttal exchange in `hfp4-fivetide-rebuttal-perspective.md`, the original Phase A had two structural weaknesses:

1. Every step had a per-weight-MSE target. Phase 11 + the fivetide PPL/KLD divergence both showed per-weight MSE is a misleading single yardstick — calibration on a target the bench can't actually measure is calibration on luck.
2. The original Phase A calibrates on top of MFP4G32 as the assumed baseline. With the PPL/KLD split, the baseline-format choice is empirically open; running each lever on multiple baselines is the cheap way to resolve it.

Two prerequisite steps are added (Step 0 and Step 0.5) and every L4/L5 step gets the same multi-baseline treatment. Net cost: +1 week vs the original 4–6 week estimate. Net benefit: each step ships with evidence about the metric users care about, not just the metric easiest to compute.

Order matters because each step's bench harness validates the next.

#### Phase A status as of 2026-05-12

What landed:

| Step | Status | Where |
|---|---|---|
| Step 0 — Bench expansion | ✅ shipped | `scripts/quant_cohort.sh` + `eval_hipfire` (via #113 / #233 / #235 / #236) |
| Step 0.5 — Reproduce on gfx1100 | ✅ done (9B + 0.8B) | cohorts `2026-05-11-cohort-phase-a-step-0.5/` (9B), `2026-05-11-cohort-phase-a-0.8b-step-0.5+1+2/` (0.8B). 9B MFP4 reproduced fivetide's +25% PPL gap (we measured +38% on prefill mode); 0.8B reproduced +93.8% → +104% PPL gap |
| Step 1+2 — L4 weighted-LS for HFP4/MFP4 | ✅ shipped, **mixed empirical result** | `crates/hipfire-quantize/src/main.rs` `--l4` flag (commit `739465c7`). Cohort: HFP4-L4 -1% KLD on 0.8B but +5.8% on 9B; MFP4-L4 -4% KLD on 9B but +6.6% on 0.8B. L4 in its current 3-candidate form is **rotation-and-model-size conditional**, not a clean lever — see §1.5 |
| Step 3 — `--tensor-type` regex CLI | ⏸️ pending | Mechanism only; no quality change expected; not on critical path |
| Step 4 — `imatrix_collect` | ✅ Tier 2 shipped | `crates/hipfire-runtime/examples/imatrix_collect.rs` (commit `c0deb3de`). Wraps `llama-imatrix` per pinned commit. Produced 9B + 0.8B imatrix.gguf at `benchmarks/quality-baselines/refs/qwen3.5-*-bf16.imatrix.gguf` |
| Step 5a — L5c per-block weighted-LS | ✅ shipped, **mixed empirical result** | `crates/hipfire-quantize/src/main.rs` `--imatrix` flag (commit `2d050152`). Cohort: HFP4-L4-L5c -9% KLD on 0.8B (clean win), only +3% recovery on 9B (still WORSE than baseline) — see `2026-05-12-cohort-phase-a-step-5a-0.8b/comparison.md` |
| Step 5a-prime — MFP4 imatrix handling | ✅ corrected (commit `1c3cd639`) | First fix (FWHT'd the imatrix) was mathematically wrong (sign bug). The actual correct treatment is **skip L5c for rotated formats** — the rotation flattens per-channel importance within blocks. MFP4-L4-L5c effectively reduces to MFP4-L4. |
| Step 6a — native L5d (imatrix-driven allocation) | ⏸️ pending | Depends on Step 4 imatrix data (have it). Not started; deferred behind the AWQ/GPTQ pivot below. |
| Step 6b — UD decompile | ✅ tool shipped, kmap ingestion pending | `crates/hipfire-quantize/src/bin/ud_decompile.rs` (commit `679bff46`). Qwen3.5-9B UD-Q4_K_XL kmap artifact landed at `benchmarks/quality-baselines/external/ud-decompile/`. `--kmap-file` consumer not yet implemented. |
| Step 7 — A3B reasoning smoke | ⏸️ pending | Gated on best Phase A configuration emerging |

Three big surprises:

1. **L4 (pure-MSE candidate search) is directionally a coin-flip vs model quality.** Four data points (2 sizes × 2 formats): two consistent + two inverted. Documented in `2026-05-12-cohort-phase-a-step-5a-9b/comparison.md` §2.2.
2. **L5c (per-block activation-weighted LS) is rotation-incompatible by math.** For FWHT-rotated weights, `Var[x_rot[i]] = Σ_j Var[x[j]]` — constant across rotated channels within a 256-segment. The per-channel weighting lever has nothing to weight. Documented in `2026-05-12-cohort-phase-a-step-5a-9b/comparison.md` §2.3.
3. **The gap to Unsloth is much bigger at bpw-matched comparison.** GGUF anchor data (commit `6c00a558`) shows hipfire MQ4 at +75% PPL vs UD-Q3_K_XL despite hipfire using +0.37 more bits per weight. The earlier "+62% vs UD-Q4_K_XL" framing was cross-bpw and made the gap look smaller than it is. Documented in `2026-05-12-cohort-phase-a-step-5a-9b/comparison.md` §2.1 + `docs/plans/mq4v2-format-proposal.md` §1.

The pivot below adjusts Phase A in response to these findings.

**Step 0 — Bench expansion (3–5 days). Prerequisite for everything below.**

Today's `scripts/bench_quant_quality.sh` emits MSE + final-norm sanity + train-pursuit smoke test. That's not enough to distinguish PPL-good-KLD-bad from PPL-bad-KLD-good. Extend the bench to emit, per format variant:

- Per-tensor MSE vs FP16 safetensors reference (already in `quant_quality_mse.rs`)
- **KLD vs FP16 reference logits** on a held-out corpus (~50k tokens; new — requires a daemon-side option to dump per-token logit distributions and a reference run on the same prompts)
- **PPL on wikitext-2-test, ctx=2048, asym4 KV** (matches fivetide's methodology so cross-validation is direct)
- **HumanEval pass@1** (downstream task; closer to what users actually do than wikitext PPL)
- Train-pursuit reasoning attractor smoke (already in the bench)

Output: single markdown table per quantization variant, suitable for diffing across PRs.

**Status 2026-05-12 — Step 0 shipped.** All 5 metrics live after the #113 / #233 / #236 merges plus the MFP4G32 v2 batched WMMA work (#235):

- `scripts/quant_cohort.sh` — cohort runner; orchestrates MSE + KLD + PPL + HumanEval + reasoning smoke per variant. Per-variant artifacts land under `benchmarks/quality-baselines/results/YYYY-MM-DD-cohort-<label>/` matching the canonical #113 schema. KLD/PPL columns now populated via real `eval_hipfire` invocations (CI verified against shipped MQ4 9B kldseq: reduce reproduces commit `cdbf07c`'s 0.876237 to 4 decimal places).
- `crates/hipfire-runtime/examples/quant_quality_mse.rs` — extended with HFP4G32 (qt=21) and MFP4G32 (qt=24) dequant. Sanity-verified against `/local/hipfire/qwen3.6-35b-a3b-mfp4.hfq` (MFP4G32 mean MSE 1.24e-6, matches PR #225 expectation).
- `scripts/bench_humaneval_completion.sh` — per-variant completion capture on the 3 in-tree humaneval prompts.
- `crates/hipfire-runtime/examples/eval_hipfire.rs` — canonical KLD scoring (per #113). Defaults to `--scoring-mode prefill`, which is **~7× faster end-to-end on gfx1100** than per-token mode and produces a different (more accurate) measurement class — see §5.3 of `issue-113-quant-quality-eval.md`.

**Target arch update 2026-05-12.** Phase A bench cohorts will run on **gfx1100 (7900 XTX) not gfx906**. The reasoning:

- **MFP4 kernels are validated on gfx11+.** Per #225 + #235 (MFP4G32 v2 batched WMMA), the production-quality MFP4 path targets gfx11/gfx12. On gfx906 MFP4 runs through the generic wave32-tier kernel (~50% lane utilization) — fine for math validation, not representative of what users will actually see.
- **gfx1100 prefill mode delivers 2162 tok/s on 9B-MQ4** (vs 108 tok/s per-token); end-to-end full-slice run ≈55 min instead of 6h32m. A 3-baseline cohort on 9B takes ~3 h instead of ~30 h. Phase A can iterate at a useful cadence.
- **The format-quality conclusion is per-arch.** Phase 11 already established that gfx906 MFP4 is correctness-only on this hardware. Phase A's job is to determine the right *format + calibration* combination for the arches where users actually run hipfire, which means gfx11+ first. gfx906 catches up via the Phase B' kernel port plan.
- **Cross-arch sanity checks remain valuable.** A small "did the result reproduce" cohort on gfx906 (1-2 variants, quick-slice 256 chunks ≈ 1.4 h each) is worth running at major milestones to confirm the conclusion isn't arch-specific. Not for every Phase A step.

The §1.4 gfx906 acceleration analysis is **not invalidated** — it stays as the future-proofing argument for old-arch users post-Phase-B'. It just isn't the right venue to run the Phase A measurement campaign.

**Step 0.5 — Reproduce fivetide's numbers in-tree on gfx1100 (1–2 days at quick-slice; 1 week full slice).**

Before optimizing anything, confirm we and fivetide measure the same thing for the same inputs. Quantize Qwen3.5-{0.8B, 4B, 9B} and Qwen3.6-A3B as {MQ4G256, MFP4G32, HFP4G32 unrotated}, run the Step 0 bench on each, compare to fivetide's published 2026-05-11 PPL table + 2026-05-12 KLD note.

**Run on gfx1100, prefill scoring mode.** Wall budget per cohort:

| Scope | gfx1100 prefill mode | gfx906 per-token (reference only) |
|---|---|---|
| 1 model × 1 variant, quick-slice 256 chunks | ~12 min | ~1.4 h |
| 1 model × 3 baselines, quick-slice | ~40 min | ~4 h |
| 1 model × 3 baselines, full slice 1175 chunks | ~3 h | ~30 h |
| 4 models × 3 baselines, full slice | ~12 h | ~5 days |

Iteration cadence on gfx1100 means we can refresh the multi-baseline reference table within a working day. Quick-slice (256 chunks) is the development bench; full-slice locks in commit-worthy results.

**Methodology fixed-points (cannot drift across runs):**
- `--scoring-mode prefill` (canonical per #113 §5; do NOT mix per-token + prefill rows)
- `--kv-mode asym3` (matches the shipped MQ4 9B reference at `cdbf07c`)
- Slice md5 `83b0205a304bf4e52172ecdb05f2e895` (locked corpus; baked into `benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`)
- llama.cpp commit `9dcf83552887bb898b4a98a5761361e504e31fc3` for any GGUF-anchor cross-checks

If our numbers reproduce fivetide's within bench noise (~CI width), Phase A baselines are validated. If they diverge, methodology debugging is the next step and Phase A pauses until the source of divergence is found.

This also produces the **multi-baseline reference table** the rest of Phase A is measured against. Note: per-token historical rows from `2026-05-08/per-seq/` are NOT comparable to the new prefill-mode reference (the modes differ ~−6.75% in mean-KLD per §5.3 of the eval plan; they're separate measurement classes).

**Step 1 (week 2) — L4a: Weighted-LS UE8M0 chooser.**

Land as quantizer-only patch; existing HFP4 files unchanged byte-format. Replace `ceil(log2(block_max / 6.0))` with a small candidate search over `block_e ∈ {e_ideal-1, e_ideal, e_ideal+1}` + brute-force re-rounding minimizing per-block MSE. Bench against Step 0.5 reference: target is *any improvement on KLD-mean or HumanEval pass@1*, not the original "5–7% per-weight MSE drop" — MSE is a leading indicator only.

**Step 2 (week 2.5) — L4b: Weighted-LS FP16 row scale.**

Replace `max_abs(row) / 6.0` with a search for `row_scale_a` minimizing post-block-quantization MSE. Same bench targets as Step 1. Land together with Step 1 if both are independent quantizer-side changes.

**Step 3 (week 3) — L5a: `--tensor-type "regex=quant"` CLI.**

Generalize hardcoded K-map rules to a CLI/config-driven regex matcher; ship K-map presets as default `.kmap` configs for Qwen3.5, Qwen3.6, Llama3, Gemma2/3. **Mechanism only — no quality change expected.** Unblocks A/B experiments without rebuilding the quantizer.

**Step 4 (week 4) — L5b: `imatrix_collect` example.**

New `crates/hipfire-runtime/examples/imatrix_collect.rs` runs forward pass on calibration corpus and dumps `Σ act²` per linear-layer input dim to sidecar `.imatrix.bin`. Calibration corpus: wikitext-103-test (300k tok) + humaneval prompts (50k tok) + slice of HF datasets multi-turn chat (650k tok) ≈ 1M tokens. Mechanism only at this stage; sidecar `.imatrix.bin` is consumed in Step 5.

**Step 5 (week 5) — L5c: Activation-weighted LS in quantize path. The dominant lever.**

Swap `min Σ (w - q)²` for `min Σ act²[i] · (w[i] - q[i])²` in each per-block fit (UE8M0 + FP16 row), using the Step 4 imatrix as the `act²` source. **Run this against all three baselines** (MQ4G256, MFP4G32, HFP4G32-unrotated) so we learn whether calibration disproportionately helps one format. **Empirical target, not predicted target:** KLD-mean on Qwen3.5-9B drops materially vs the corresponding Step 0.5 baseline; the specific number is whatever measurement says — the original "0.20–0.30" projection is now an open question.

**Step 6 (week 6) — L5d: per-tensor bit allocation. Two parallel paths.**

The lever: which tensors deserve more than the default 4-bit precision. Two
independent ways to make that choice; we should land both and bench.

- **Step 6a — Native (imatrix-derived).** Use per-tensor `Σ(Act²)` + ZD score
  (Liu et al. 2024) to pick HFP4G32 / HFP6G32 / Q8 per tensor; bit budget
  held constant. Depends on the Step 4 imatrix sidecar. Replaces hardcoded
  K-map rule 4/5 with data-driven promotion. *Principled but lead-time
  bound by Step 4.*

- **Step 6b — UD decompile shortcut.** Open an Unsloth UD-Q4_K_XL / UD-Q4_K_M
  GGUF for the target model, parse the tensor table, extract Unsloth's
  per-tensor bit allocation (which tensors got promoted to Q5_K / Q6_K /
  Q8 / FP16). Map onto the closest hipfire format-family equivalent
  (HFQ4G256 / MQ6G256 / Q8_F16). Apply as a `--kmap-file` override that
  lets the quantizer mimic UD's per-tensor choices. *No imatrix dependency
  — UD has already paid the calibration cost; we just copy their decisions.*

  Implementation: `crates/hipfire-quantize/src/bin/ud_decompile.rs` — opens
  any GGUF via `gguf_input.rs`, emits a JSON sidecar mapping
  `{tensor_name → ggml_type → suggested_hipfire_qt}` plus a summary
  (type-distribution, promoted-tensor list, BPW). The quantizer's existing
  K-map rule machinery is extended to accept a JSON-driven per-tensor
  override; the cost is ~2-3 hours dev for the decompile tool and ~1-2 days
  for the `--kmap-file` ingestion + tests.

  **What this shortcut does NOT give us:** per-channel `Σ act²` values for
  L5c (activation-weighted LS). Unsloth's chosen scales are the *result* of
  imatrix-aware LS — the imatrix itself is consumed and not recoverable in
  any tractable way. So Step 5 (the dominant lever) still needs Step 4's
  imatrix data.

**Sequencing:** Step 6b can ship before Step 4/5/6a — it has no upstream
dependency beyond a Unsloth-published GGUF. Run a 5-variant cohort
(`{calibrated-MQ4, calibrated-MQ4+UD-kmap, calibrated-MFP4,
calibrated-MFP4+UD-kmap, UD-Q4_K_XL anchor}`) once Step 5 lands. The
delta between {calibrated-X} and {calibrated-X + UD-kmap} measures how
much of UD's quality lift is L5d (bit allocation) vs L5c (per-channel
calibration). If most of UD's edge is L5d, Step 6b alone captures the
practical win without needing imatrix-driven native L5d (Step 6a). If
most is L5c, the UD-kmap shortcut is mostly free quality gravy on top
of the imatrix work.

**Step 7 (week 7) — A3B reasoning smoke + multi-metric decision.**

Re-run train-pursuit reasoning on calibrated A3B (best Phase-A configuration from Step 5–6 measurements). **Hedged prediction** (low confidence — Phase 11 already falsified one quant-quality hypothesis for the spiral): either the spiral resolves on the calibrated config, or it doesn't and the residual coherence work moves entirely to runtime-side levers per Thread 2 §"Phase 11 engine-pass implications".

Decision: at this point we have, for each baseline format × calibration combination, a complete (PPL, KLD, HumanEval, attractor) measurement table on Qwen3.5-{0.8B, 4B, 9B} + Qwen3.6-A3B. Pick the default format per arch on evidence, not on the per-weight-MSE story.

**Expected end state of Phase A:** decision made on evidence. The original "Phase A delivers KLD-mean 0.10–0.20" projection was based on per-weight-MSE extrapolation; it should be treated as a hopeful ceiling, not a target. The honest target is "deliver enough data to choose a default with confidence." Multi-format calibrated quants land for whichever model × format combination measures best, with explicit "different default format per arch" tolerated if the data supports it. **All without a wire-format change** — the wire-format extensibility analysis in §3 stands regardless of which element format ends up the winner.

### Phase A revised (2026-05-12) — AWQ/GPTQ pivot, MQ4K deferred

The Step 5a + 9B GGUF-anchor cohort findings (summarized in the status block above) re-prioritize Phase A. Steps 1-7 as written stay valid as documentation of what was tried and what landed; what changes is *what comes next*.

**The pivot in one paragraph.** L5c per-block weighted-LS only partially works: clean +9% KLD win on 0.8B HFP4 but only +3% recovery (still worse than baseline) on 9B HFP4, and mathematically incompatible with rotated formats. Meanwhile the GGUF anchors expose a bpw-matched PPL gap of +75% to UD-Q3_K_XL despite hipfire using +0.37 more bits/weight — the calibration lever (in our per-block form) isn't closing this. The natural conclusion is to try a different calibration mechanism (AWQ's per-channel pre-scaling + GPTQ's Hessian-aware sequential quantization) on the existing MQ4 wire format, both of which compose with hipfire's rotation lever and require no kernel migration.

**Stage A — AWQ on MQ4 (~1.5-2 weeks, no kernel migration). Top priority.**

Per-channel AWQ scale calibration from existing imatrix data. Pre-scale weights `W' = W · diag(s)` before quantization; at inference, divide activations by `s` before the rotation kernel (foldable into RMSNorm, or as a tiny vector-divide kernel before `mq_rotate_x`). The math `(W·s) · (rot(x)/s) = W·rot(x)` cancels exactly — full derivation in `mq4v2-format-proposal.md` §0.

What ships:
- Quantizer-side: AWQ scale computation (per linear layer, from imatrix), pre-quantize weight scaling
- Wire format extension: per-tensor FP16 scale vector (length K). Tiny — ~256 KB total for 9B; either sidecar or per-tensor header field
- Runtime: tiny pre-rotation kernel mod (apply `x / s` before existing `mq_rotate_x`)
- Bench: 9B + 0.8B cohort, against UD-Q3_K_XL anchor

Predicted lever value (from AWQ literature on Q4 quants): +15-20% PPL improvement. Composes with FWHT rotation.

**Stage B — GPTQ on MQ4 (~2 weeks, zero kernel/format change). Stacks with A.**

Hessian-aware column-by-column sequential quantization. Output is still MQ4 wire format — just better INT4 placement that minimizes the loss-weighted reconstruction error instead of per-element MSE.

What ships:
- Calibration extension: extend `imatrix_collect` (or add a sibling tool) to dump per-layer `X^T X` (full Hessian) in addition to the diagonal `Σ act²` that current imatrix carries
- Quantizer-side: GPTQ algorithm in the MQ4 path
- **Zero wire format change. Zero kernel change.** Same .hfq files load identically
- Bench: stack on Stage A's calibrated-MQ4-with-AWQ

Predicted lever value: +15-25% PPL improvement on top of RTN. Composes additively with AWQ.

**Stage C — MR-GPTQ on MFP4 (~2-3 weeks, reuses Stage B GPTQ scaffolding). The paper-validated MXFP4 recipe.**

Direct port of MR-GPTQ (Egiazarian, Castro, Kuznedelev, …, Alistarh — arXiv 2509.23202v3) to hipfire's MFP4G32 wire format. The paper specifically targets MXFP4 — which IS our MFP4G32 (E2M1 codes + UE8M0 per-32 scales). Three algorithmic pieces:

1. **GPTQ** — column-by-column Hessian-aware quantization (shared with Stage B)
2. **E8M0 range mapping (Appendix H of the paper)** — MR-GPTQ's key differentiator. UE8M0 has dynamic range 2^-127 to 2^127 but real weight distributions occupy ~2^-20 to 2^20. Mapping the wide E8M0 range to a useful data range improves accuracy substantially.
3. **MSE-optimized grids** — alternating optimization between block scales and per-tensor scales.

Plus static activation reordering and optional block-wise Hadamard size flexibility (k ∈ {16, 32, 64, 128} as alternatives to our existing FWHT-256).

What MR-GPTQ measured on MXFP4 (Llama-3.1-8B):
- RTN: 87.83% recovery vs FP16
- Vanilla GPTQ: 89.47% recovery (+1.6pp)
- **MR-GPTQ: 93.31% recovery (+5.5pp over RTN)**

Vanilla GPTQ alone doesn't help MXFP4 much — the format-specific pieces (E8M0 range mapping + MSE grids) are the differentiator. Our existing FWHT-256 rotation is the same class as their k=16-128 Hadamards.

**Why this matters strategically**: the paper's existence reframes our Step 5a + §1.5 cohort findings. Both we (rotation-flatness math) and the paper (small-group neutralizes outlier mitigation) predict that per-block weighted-LS calibration fails on MFP4 — confirmed empirically. MR-GPTQ is the WORKING recipe for the format we already have. If we replicate ~93% recovery on Qwen3.5-9B-MFP4, we revive the §6 "HFP4 is the format for several years" strategic posture that the cohort findings had suspended.

**Why this might make MQ4K (Stage E) unnecessary**: MR-GPTQ says the existing MFP4G32 wire format isn't broken — the calibration was. The whole MQ4K argument rests on "MFP4G32 underperforms because it lacks Q4_K's per-32 sub-scaling." MR-GPTQ shows MXFP4 can approach NVFP4 accuracy without sub-scaling, just with the right calibration recipe.

Detail in `mq4v2-format-proposal.md` §6.5.

**Stage D — Step 6b kmap ingestion (~1-2 days). Cheap orthogonal lever.**

Already have the UD-Q4_K_XL kmap artifact for 9B (commit `679bff46`). Implement the `--kmap-file` CLI flag in the quantizer that overrides the hardcoded K-map rules with a JSON-driven per-tensor allocation. Apply Unsloth's per-tensor promotion choices to hipfire format equivalents. Bench in matrix with Stages A + B + C.

**Stage E — MQ4K wire format prototype (~5-7 weeks). DEFERRED.**

Only fires if Stages A + B + C + D don't close the bpw-matched gap to UD-Q3_K_XL meaningfully (target: 60%+ of the gap closed). Tracked in `docs/plans/mq4v2-format-proposal.md` with Stage 1 / Stage 2 / Stage 3 sub-decomposition.

**Stage 0 prerequisite — Hipfire Q8 baseline cohort (~2.5h wall on 9B at 256 chunks).**

Quantize 9B as `--format q8f16` (near-lossless per-tensor — measured 2.63e-8 mean MSE, 113× lower than MFP4). Run the standard cohort. The resulting KLD vs llama.cpp BF16 measures the **engine-drift floor** — the KLD that hipfire produces even when the quant is near-lossless. Without this number, "Stage A closes X% of the gap" can't be interpreted (X% of what? — depends on whether the floor is 0.05 or 0.5 KLD).

Status 2026-05-12 (afternoon): ✅ **done.** Result: KLD mean = 0.5735 (CI 0.5490–0.5996), KLD p99 = 18.325, PPL = 13.383, n=261,888 tokens scored. Stored at `benchmarks/quality-baselines/results/2026-05-12-cohort-phase-a-q8-floor-9b/`.

**Interpretation caveat — the 0.57 is not Q8 quantization noise.** Q8 vs FP16 in published literature is ~0.001–0.005 KLD. 0.57 nats is "different model" territory. What we actually measured is `hipfire(Q8 weights + asym3 KV cache) vs external_ref(BF16, llama.cpp scoring path)`. The dominant contribution to 0.57 is **hipfire engine drift vs llama.cpp** (kernel precision, accumulation order, attention math, FWHT-rotated K-cache); KV-cache asym3 quantization contributes a meaningful but smaller share; Q8 weight quantization itself contributes ~0.001–0.01. Implications for Phase 3+:

1. The number is the **hipfire-engine-on-asym3-KV floor** against an external reference, not the "Q8 floor" in the literature sense.
2. All Phase A candidates inherit the same offset. The number that matters is **delta-above-floor**, not absolute KLD.
3. Literature AWQ lift of +15-20% PPL is measured against `floor + quant_noise`, which is almost entirely `floor` in our setup. Expected absolute KLD improvement is small in nat terms (~0.02–0.05) and at the edge of CI width on 256 chunks (~0.025 half-width). Bench at 512+ chunks for distinguishability, or add an additional FP16/Q8 engine-control variant per-cohort so quantization-attributable delta is isolated from engine drift.

Disambiguating-variant rule (going forward): every cohort that compares 4-bit candidates should include a Q8/FP16 engine-floor row, so `KLD(MQ4) − KLD(Q8)` cleanly isolates the quantization-attributable component.

**KV-mode ablation (2026-05-12, gfx1151, 9B-q8f16, n=20 chunks).** Same Q8 candidate, per-token mode, asym3 vs q8 KV cache:

| KV mode | slice-mean KLD | PPL |
|---|---:|---:|
| asym3 (canonical floor cohort) | 0.6004 (20-chunk variance of the 0.5735 published number) | 13.74 |
| q8                              | **0.4034** | 11.51 |
| Δ (asym3 − q8) | **−0.197 (−33%)** | sign-test p=1.91e-6, 100% one-signed |

asym3 KV is contributing **~0.20 of the 0.57** floor (~33%) — substantial but not the whole story. ~0.40 remains with q8 KV. Q8 KV vs FP16 KV is 0.01–0.05 in literature; benign cross-engine drift is 0.01–0.05; Q8 weight quant is 0.001–0.005. Sum of these explainable components: at most ~0.10. The remaining ~**0.30 nats is unaccounted for** and points to a real hipfire-vs-llama.cpp implementation discrepancy, not benign drift. Next diagnostic step is Step 2 (position-0 logit comparison vs an independent HF transformers oracle) to localize: feed-forward vs attention-with-KV-history. Raw kldseqs at `benchmarks/quality-baselines/results/2026-05-12-kv-ablation-9b/per-seq/`.

**Residual-0.30 hypothesis probes (2026-05-12 evening, gfx1151).** Five candidates considered for the unaccounted ~0.30:

1. DeltaNet (linear-attention) recurrent drift × 24 layers
2. Tokenizer / BOS / chat-template framing — **ruled out**; ref tokens[0..16] byte-identical to llama-tokenize output, no BOS prepend
3. RMSNorm precision drift × 32 layers
4. lm_head Q8 vs BF16 precision (Q8 lm_head over 4096 × 248320)
5. Q8 K-cache rotation oddity

Test #4 — lm_head precision (**FALSIFIED**).
- Added diagnostic env-gate `HIPFIRE_QUANTIZE_LM_HEAD_F16=1` in `hipfire-quantize` (Base arm of `kmap_resolve_mode`; does not require `--kmap-dense` so lm_head F16 is isolated from K-map's other Promote6 side-effects). One-liner override; default behaviour unchanged.
- Produced `~/.hipfire/models/qwen3.5-9b.q8f16-f16lm-2026-05-12` (10.48 GB vs 9.53 GB baseline; size delta = 909 MB == predicted F16-vs-Q8 lm_head delta to the byte, confirming the override fired).
- **Initial eval was kernel-gated**: at canonical `--scoring-mode prefill --kv-mode asym3` the F16 lm_head decompressed to F32 at load time (`qwen35.rs:810-819`) and per-position dispatch fell through a serial F32 GEMV path → 4 tok/s wall, ~84 h full-slice ETA. Unusable as a diagnostic.
- **F16 lm_head plumbing landed** (commit pending) to close that gap: stop the host-side F16→F32 decompress and keep `DType::F16` on GPU; add `DType::F16` arm in `weight_gemv` that calls `gemm_f16_batched_lmhead(batch=1)`; in `eval_hipfire.rs` prefill scoring branch, replace the per-position `weight_gemv` loop with a single `gemm_f16_batched_lmhead(batch=scored_per_chunk)` call when the lm_head is F16. Uses the existing WMMA-backed `gemm_mw16_residual_wmma` kernel — no new kernels written.
- Per-chunk timing breakdown (single-chunk timing run with `gpu.hip.device_synchronize()` between phases): prefix forward_prefill_batch = 53.4 s, scored forward_prefill_batch with hidden capture = 53.5 s, alloc(1 GB) = 0.05 s, **batched lm_head GEMM = 0.79 s**, score loop (1023 PCIe downloads + KLD math) = 0.85 s. lm_head went from ~280 s/chunk (F32 fall-through) to 0.79 s/chunk (WMMA batched) — a ~350× speedup. The transformer-stack `forward_prefill_batch` (53 s/chunk) is the new dominant cost on gfx1151 9B prefill, unrelated to the lm_head question.
- **Final eval (matched mode against the 0.6004 baseline): `--scoring-mode per-token --kv-mode asym3 --max-chunks 20`**:

  | variant | slice-mean KLD | mean NLL | PPL |
  |---|---:|---:|---:|
  | q8f16 baseline (Q8 lm_head)               | 0.600439 | 2.6203 | 13.7390 |
  | q8f16-f16lm (F16 lm_head, this run)       | 0.600614 | 2.6206 | 13.7441 |
  | **Δ (F16 − Q8)** | **+0.000175** | +0.0003 | +0.0051 |

  Per-sequence KLDs match to 4 decimal places (seq[:5] = [0.4794, 0.5677, 0.7836, 0.2883, 0.5038] vs [0.4794, 0.5675, 0.7834, 0.2881, 0.5034]). The 1.7e-4 nat delta is round-off noise, not a real precision contribution. **Hypothesis #4 is falsified: lm_head Q8 quantization is NOT a measurable contributor to the engine-drift floor on this model + slice.** The residual ~0.30 nats lives elsewhere — DeltaNet drift (#1), RMSNorm precision (#3), or Q8 K-cache rotation (#5) remain the open candidates.
- Engineering byproduct worth keeping: the F16 lm_head plumbing also unblocks future vision-encoder F16 paths (Qwen-VL `linear_f16` consumers) and lays groundwork for an eventual BF16 lm_head — the WMMA kernel exists and is now reachable from the standard `weight_gemv` dispatch.

Raw kldseqs at `benchmarks/quality-baselines/results/2026-05-12-lm-head-precision-9b/per-seq/`.

Test #1 — Qwen2.5-7B DeltaNet discriminator (**INFRA closed, RESULT UNUSABLE — separate bug**).
- Qwen2.5-7B (Qwen2 arch, full attention, **no DeltaNet**) is the cleanest discriminator: if its KLD floor lands meaningfully below the 9B Q3.5 floor under matched eval conditions, DeltaNet recurrent drift is real.
- `eval_hipfire` is hardcoded to `hipfire-arch-qwen35` (DeltaNetState in the per-token signature, Qwen35Scratch type). Ported as `crates/hipfire-runtime/examples/eval_hipfire_llama.rs` — same KLD math + HFKSEQ output, plumbed through `<Llama as Architecture>` + `hipfire_runtime::llama::{forward_scratch_embed, forward_scratch_compute}`. Per-token mode only; `forward_prefill_batch` in `hipfire_runtime::llama` lacks the hidden-state capture hook that eval_hipfire's prefill scoring path relies on. Registered as a separate `[[example]]` with `required-features = ["arch-llama"]`. Cargo binary at `target/release/examples/eval_hipfire_llama`.
- Q2.5-7B kldref built via `build_kld_ref` against locally-converted BF16 GGUF (`/data/models/qwen/Qwen2.5-7B-BF16/Qwen2.5-7B-Instruct-BF16.gguf`, 15.2 GB; throughput ~552 tok/s on gfx1151). Output: `benchmarks/quality-baselines/refs/qwen2.5-7b-bf16.kldref.bin` (2.49 GB).
- Q2.5-7B quantized as `qwen2.5-7b-instruct.q8f16` (8.1 GB).
- Two further constraints surfaced on first launch: (a) `--kv-mode asym3` rejects non-Qwen3.5 head_dim (`llama.rs:3065` asserts head_dim=256; Q2.5-7B has head_dim=128) → switched to `--kv-mode q8`. (b) Per-token rate on 7B is ~11 tok/s (full slice = ~30 h); current run caps at `--max-chunks 32` (~50 min wall) for a directional 32-chunk-mean comparison. A matched `--max-chunks 32 --kv-mode q8 --scoring-mode per-token` run on the existing 9B q8f16 model is queued next for the apples-to-apples delta.

What the 7B test settles vs leaves open:
- **Settles directionally** (32-chunk noisy mean): does removing DeltaNet move the floor by ~0.3 nats? Yes/no/maybe.
- **Does NOT settle**: absolute floor under prefill-mode methodology (per-token vs prefill numbers don't transplant); per-architecture confounders (Q2 vs Q3.5 do not share *only* DeltaNet — also differ on MTP layer, head_dim, vocab size; the 7B vs 9B parameter count is a confound for absolute KLD but not for the floor-shift direction).

**Q2.5-7B result was unusable**: slice-mean KLD = 8.38, PPL = 28569 — model emitting effectively-random logits. **Root cause identified**: `hipfire-runtime::llama` does not load or apply Q/K/V projection biases. Qwen2/2.5 stores them (84 bias tensors verified in the .hfq: `model.layers.<i>.self_attn.{q,k,v}_proj.bias`), Qwen3 removed them, hipfire-arch-llama was built for Qwen3-only. Fix scope: add `q_bias` / `k_bias` / `v_bias` fields to `LayerWeights`, gate loading on `find_tensor`, call existing `gpu.bias_add_f32` after the Q/K/V matmuls in `forward_scratch_compute` + `forward_prefill_chunk`. ~half-day patch, deferred — see next section for the cheaper substitute that ran instead. Raw kldseq at `benchmarks/quality-baselines/results/2026-05-12-qwen25-7b-deltanet-discriminator/per-seq/`.

Test #1 — Qwen3 plain (no DeltaNet, no Q/K/V bias) discriminator (**STRONG SIGNAL, hypothesis #1 supported**).
- Skipping the Q2.5 bias retrofit, used Qwen3-0.6B as the no-DeltaNet test vehicle (no Q/K/V bias, no q_norm/k_norm — already supported by hipfire-arch-llama). Built BF16 GGUF via `convert_hf_to_gguf.py` (1.5 GB), then kldref via `build_kld_ref` (~10 min, 1632 tok/s on gfx1151), q8f16 quantize (~1 min).
- Matched the comparison against `qwen3.5-0.8b.q8f16` (already in the model cache, BF16 kldref already in `refs/`) — Qwen3.5-0.8B is close-in-size to Qwen3-0.6B (same family-of-families, similar capability) but has DeltaNet (18 LA + 6 FA hybrid).
- Both runs: `--kv-mode q8 --scoring-mode per-token --max-chunks 20`, matched conditions; only architecture differs.

| Model | architecture | KLD floor | PPL | mean NLL |
|---|---|---:|---:|---:|
| Qwen3-0.6B | plain Qwen3 (no DeltaNet, no QKV bias) | **0.0098** | 19.5 | 2.972 |
| Qwen3.5-0.8B | Qwen3.5 hybrid (18 LA + 6 FA = 75% LA) | **0.4945** | 33.2 | 3.503 |
| Qwen3.5-9B (kv-q8 ablation) | Qwen3.5 hybrid (24 LA + 8 FA = 75% LA) | **0.4034** | 11.5 | — |

  Raw kldseqs at `benchmarks/quality-baselines/results/2026-05-12-deltanet-discriminator/per-seq/`.

- **The Qwen3-0.6B floor is ~50× lower than Qwen3.5-0.8B at essentially-equal size.** This is not a model-size effect (0.6B vs 0.8B is within 35%); it is an architecture effect. Q3.5-0.8B and Q3.5-9B land at similar floor magnitude (0.49 vs 0.40 nats) despite the 11× param-count gap, which is consistent with both having the same 75% LA-layer ratio. Plain Qwen3 at 0.01 nats is essentially Q8 weight quantization noise — the "engine-drift floor" is *not* generic cross-engine drift, it is **specific to hipfire's DeltaNet / Qwen3.5-hybrid implementation**.
- Consequence: previously framed measurements like "Stage A AWQ closes X% of the gap to UD-Q4_K_XL on Q3.5-9B" were comparing `(4-bit-quant noise) + (DeltaNet drift baseline ~0.4)` against `(BF16) + (llama.cpp's DeltaNet implementation drift, whatever it is)`. The 4-bit-attributable component is much smaller than the headline KLD numbers suggested — the literature-comparable AWQ lift of ~0.02-0.05 nats becomes detectable once the DeltaNet pedestal is either subtracted (by including a Q8 control row in every Q3.5 cohort, which we already do) or eliminated (by fixing the DeltaNet forward path).

**Where the floor stands after these tests (2026-05-12 evening):**
- KV asym3 → q8 ablation: ~0.20 nats of 0.57 (kv-rotation precision).
- Test #4 (lm_head Q8 vs F16): < 0.001 nats — NOT the source.
- Test #1 (DeltaNet discriminator): Qwen3.5 hybrid vs plain Qwen3 differs by ~0.48 nats at matched size — **DeltaNet implementation is the dominant remaining contributor**, well above any benign engine-drift baseline.
- Q3.5 stack residuals after subtracting DeltaNet contribution: ≲ 0.01 nats — within Q8 quant-noise expectations.

**Where the next-step work should focus** is now clear: localize the DeltaNet forward-pass divergence. Per-layer hidden-state KLD against an HF transformers reference on a single chunk (Step 2 of the original plan, plus per-layer instrumentation) should pin down which of the 18-24 LA layers contribute most, and whether the drift is a Δrule / FWHT / state-update / mixing-fraction precision issue. The LayerNorm precision hypothesis (#3) and Q8 V-cache rotation hypothesis (#5) are still on the table for the smaller residual but are now lower-priority — the bulk of the floor lives in DeltaNet.

**Step A — per-layer drift localization (2026-05-12 evening). FA is the dominant per-layer contributor, not DeltaNet.**

After Step B confirmed cross-engine ground truth exists, instrumented both engines for per-layer hidden-state capture on Qwen3.5-0.8B chunk 0:

- `scripts/dump_hf_hidden_states.py` — HF transformers BF16 oracle. Uses a forward-pre-hook on `model.model.norm` to capture the pre-final-norm last-layer output (transformers's `output_hidden_states[n_layers]` is post-norm, which doesn't match hipfire's HiddenStateRingBuffer capture point).
- `crates/hipfire-runtime/examples/dump_qwen35_hidden_states.rs` — hipfire forward through `forward_scratch_with_hidden`, reusing the existing `HiddenStateRingBuffer` from spec-decode infra with `extract_layers` overridden to `(0..n_layers)` for full coverage.
- `scripts/compare_hidden_states.py` — offline per-layer cosine + relative-L2 comparator.

Both dumps are 192 MB (24 layers × 2048 positions × 1024 hidden × 4 B). Identical input tokens (lifted from `qwen3.5-0.8b-bf16.kldref.bin` chunk 0).

| Layer | type | rel_L2 | mean cos | Δrel_L2 vs prev | note |
|---|---|---:|---:|---:|---|
| 0 | LA | 0.061 | 0.998 | — | near-identical |
| 1 | LA | 0.082 | 0.996 | +0.02 | smooth |
| 2 | LA | 0.094 | 0.995 | +0.01 | smooth |
| **3** | **FA #1** | **0.187** | **0.980** | **+0.09** | **biggest single-layer jump** |
| 4 | LA | 0.212 | 0.975 | +0.03 | tracks |
| 5 | LA | 0.222 | 0.972 | +0.01 | tracks |
| 6 | LA | 0.237 | 0.968 | +0.01 | tracks |
| **7** | **FA #2** | **0.289** | **0.953** | **+0.05** | second-largest jump |
| 8 | LA | 0.308 | 0.945 | +0.02 | tracks |
| 9 | LA | 0.321 | 0.940 | +0.01 | tracks |
| 10 | LA | 0.335 | 0.931 | +0.01 | tracks |
| **11** | **FA #3** | **0.342** | **0.928** | +0.01 | tail-off |
| 12-22 | mixed | 0.29-0.33 | 0.93-0.95 | plateau | drift saturates |
| 23 | FA | 0.323 | 0.937 | — | post-residual pre-final-norm |

  Per-layer cosine min-across-positions (min_cos column) drops as low as 0.66 in mid-layers — there are specific positions where hipfire's hidden state is significantly rotated from HF's. Cosine never inverts though; we're not seeing sign flips.

**Reframe: the per-layer drift is concentrated in FullAttention layers, not DeltaNet.** Layer 3 (the first FA) alone introduces ~half of the eventual rel_L2 in a single layer. Subsequent FA layers add smaller hops. LinearAttention (DeltaNet) layers track HF smoothly with tiny ~+0.01 per-layer increments. After the 4th FA (~layer 11), drift saturates and stops growing — additional layers maintain the divergence but don't compound it. This is the signature of a **fixed-direction error introduced at the first FA layer that subsequent identical operations preserve** rather than an error that accumulates.

This re-frames the hypothesis #1 conclusion. The 50× floor gap between plain Qwen3-0.6B (0.0098) and Qwen3.5-0.8B (0.4945) IS real and IS Qwen3.5-specific — but the per-layer attribution points at **Qwen3.5's FullAttention path**, not its DeltaNet path. Plain Qwen3 has standard FA without Q-norm / K-norm and without the LA-FA interleaving; its near-zero floor is consistent with that. Hipfire's FA implementation for Qwen3.5 has a precision or convention bug that fires hard at the first FA layer.

Candidate culprits inside Qwen3.5 FullAttention (in hipfire-arch-qwen35):
- **Q-norm / K-norm precision** — Qwen3.5 has per-head Q-norm and K-norm (RMSNorm-style); hipfire auto-detects `has_qk_norm` from tensor presence, but the kernel dispatch (search `config.has_qk_norm` in `llama.rs`) accumulates differently or in different precision than HF. **Highest probability candidate**: plain Qwen3-0.6B has no Q-norm/K-norm and has near-zero floor; the architectural difference between plain Qwen3 and Qwen3.5 FA is exactly the QK-norm presence + the LA-FA interleaving.
- **RoPE base / frequency precomputation** — Qwen3.5 uses `rope_theta=1000000.0`; if the precomputed sin/cos table loses precision, FA accuracy drops on first use.
- ~~**KV cache write/read precision** at the FA layers~~ — **RULED OUT 2026-05-12 evening**: re-ran the hipfire per-layer dump with `--kv-mode fp32` (no KV quantization at all) and the per-layer rel_L2 profile is essentially identical to the q8-KV run — layer 3 rel_L2 = 0.187 in BOTH; later layers 4-23 differ by at most ±0.008 nats. KV quantization is not the FA drift source.
- ~~**Q-norm / K-norm accumulation precision**~~ — **RULED OUT 2026-05-12 evening**: added an `rmsnorm_f32_f64acc` kernel variant (fp64 accumulator inside the parallel-reduction sum-of-squares + fp64 rsqrt) and a `HIPFIRE_RMSNORM_F64=1` env-gate routing rmsnorm_batched to it. Re-ran the layer dump under f64-acc — layer 3 rel_L2 = 0.1873 (byte-identical to fp32-acc). The 256-element sum has plenty of fp32 precision; promoting to fp64 doesn't move the needle.
- **Softmax / attention probability precision** — fp16 accumulator vs fp32 in the scaled-dot-product step.

**Reframing — both DeltaNet AND FA contribute.** Closer reading of the layer 0-2 numbers from the Q3.5-0.8B dump: rel_L2 starts at 0.06 (layer 0) and grows to 0.09 (layer 2), all through LinearAttention layers. For comparison, plain Qwen3-0.6B's TOTAL floor across 28 layers is only ~0.01 nats. So Q3.5's DeltaNet contributes ~0.06 rel_L2 in just ONE layer — already 6× more than plain Qwen3's entire 28-layer accumulation. FullAttention then adds another +0.10 at the first FA encounter. Earlier framing "drift is FA, not DeltaNet" overstates the case: both diverge from HF reference, but with different per-layer signatures — DeltaNet has a steady-state error (~0.06 from layer 0, slow growth), FA has a step-change error (+0.05 to +0.10 per FA layer in early stack, saturating). Future probes should run a similar per-layer comparison on a plain-Qwen3 model to bound how much of the Q3.5 layer-0 DeltaNet error is fundamental DeltaNet vs simply Q8 weight-quant noise propagated through any layer.

**RoPE convention probe — ROOT CAUSE IDENTIFIED 2026-05-12 evening.**
While inspecting the FullAttention path to design the next probe, found that the HF Qwen3.5 reference and hipfire use INCOMPATIBLE RoPE conventions:

  - HF `transformers/models/qwen3_5/modeling_qwen3_5.py:543-579` uses `rotate_half`: pairs are (i, i + rotary_dim/2). After `q_rot * cos + rotate_half(q_rot) * sin`, dim i rotates with dim i+rotary_dim/2. **Half-split convention.**
  - hipfire `kernels/src/rope_partial_interleaved.hip:17-18` uses `pair * 2` / `pair * 2 + 1`: pairs are (2i, 2i+1). **Interleaved convention.**

These convention choices BOTH produce valid RoPE — provided the Q/K weight layout matches. llama.cpp resolves this for many model families by permuting Q/K weights at HF→GGUF convert time (`_reverse_hf_permute` at convert_hf_to_gguf.py:2533-2541, `undo_permute = True` on the LlamaModel class), reshaping HF's half-split storage into interleaved storage so the runtime's interleaved RoPE produces the correct rotation. **For Qwen3.5 specifically the entire inheritance chain (Qwen3_5TextModel → _LinearAttentionVReorderBase → Qwen3NextModel → Qwen2MoeModel → TextModel) has `undo_permute = False` / absent — ggml's RoPE for Qwen3.5 uses the half-split convention natively, matching HF's storage layout without permutation.**

Hipfire-quantize converts the same HF safetensors with **no Q/K permutation**, then the runtime applies **interleaved RoPE** to weights that are stored in half-split layout. Every FullAttention layer applies the wrong rotation pairing.

Why hipfire still generates coherent text despite this: Q and K are BOTH rotated wrong with the same wrong convention, so Q·K^T still produces a position-aware attention pattern — just with shifted effective frequencies. The relative-position information survives at degraded fidelity, the model still samples mostly-correct tokens, but the logits diverge from the BF16 reference by ~0.4 nats per scored position.

Predicted layer-by-layer signature of an "every-FA-layer applies wrong-pair RoPE" bug:
  - Layer 0-2 (LA): no FA, no RoPE-induced drift (matches: rel_L2 0.06-0.09 is small residual noise)
  - Layer 3 (FA #1): biggest jump — first wrong rotation applied (matches: +0.10 step)
  - Later FA layers: smaller jumps — residual already shifted into the wrong basis (matches: +0.05 → +0.01 decreasing)
  - Plateau after FA #4 (~layer 11) (matches: drift saturates at ~0.32)

Every observable in Step A's per-layer comparison is consistent with this single root cause.

Two possible fixes:
  - (a) Permute Q/K weights at hipfire-quantize time so interleaved RoPE produces mathematically correct rotation (same trick as llama.cpp's LlamaModel `_reverse_hf_permute`). Backward-compatible for new quants, requires re-quantize on existing .hfq files OR a runtime auto-detect / one-time fixup.
  - (b) Switch hipfire's RoPE kernel to a half-split variant. Requires no changes to existing .hfq files. Likely needs a new `rope_partial_halfsplit_f32` kernel alongside the existing interleaved one, plus an arch dispatch in qwen35::forward.

Approach (b) is incrementally safer (existing models keep working through a code path; no requantize required); approach (a) is more invasive but aligns with the established hipfire pattern of doing storage-side transforms at quantize time. The choice depends on whether other Qwen-arch models (Qwen2, Qwen3, …) currently work through hipfire's interleaved-RoPE path by coincidence (i.e., they also have the bug but the bug is masked by full-attention robustness) — if so, fixing requires careful per-model verification.

**Confidence check before declaring "done":** Plain Qwen3-0.6B's near-zero floor (0.0098 nats) suggests its RoPE works correctly through hipfire — either because HF Qwen3 (non-3.5) uses interleaved convention (different from Qwen3.5's rotate_half), or because the full-attention robustness masks the bug at small scale, or because the test isn't sensitive enough. Worth a per-layer dump of Q3-0.6B to verify before committing to a fix direction.

Confirmed by reading transformers source: HF `qwen3/modeling_qwen3.py:179-180` and `qwen2/modeling_qwen2.py:144-145` both use `rotate_half` — same half-split convention as Qwen3.5. So plain Qwen3-0.6B in hipfire works correctly NOT because its RoPE convention happens to match interleaved (it does NOT), but because **plain Qwen3 in hipfire goes through hipfire-arch-llama, whose `rope_f32` kernel (`kernels/src/rope.hip`) uses HALF-SPLIT convention already** (rotates pair (i, i+half) — see kernel source). The interleaved-RoPE bug is **specific to hipfire-arch-qwen35's choice of `rope_partial_interleaved.hip`**, not a hipfire-wide pattern. The fix is local to the qwen35 arch.

**Half-split RoPE fix (probe-and-validate, 2026-05-12 evening).**

Added `kernels/src/rope_partial_halfsplit.hip` — twin of `rope_partial_interleaved.hip` but with pair (i, i+n_rot/2) instead of (2i, 2i+1), matching HF `rotate_half` semantics. Routed via `HIPFIRE_ROPE_HALFSPLIT=1` env-gate inside `rope_partial_interleaved_f32` (dispatch.rs) so the fix can be probed without altering the default code path.

Per-layer drift comparison (Q3.5-0.8B chunk 0, both runs with kv-q8):

| Layer | type | rel_L2 (interleaved BUG) | rel_L2 (halfsplit FIX) | improvement |
|---|---|---:|---:|---:|
| 3  | FA #1 | 0.187 | **0.087** | -54% |
| 7  | FA #2 | 0.289 | **0.157** | -46% |
| 11 | FA #3 | 0.342 | **0.157** | -54% |
| 15 | FA #4 | 0.297 | **0.138** | -54% |
| 19 | FA #5 | 0.294 | **0.153** | -48% |
| 23 | FA last | 0.323 | **0.189** | -42% |

Mean cosine at FA layers jumped from 0.93–0.95 to 0.98–0.99.

End-to-end KLD probe — single chunk first, then matched 20-chunk slice (Q3.5-0.8B, per-token kv-q8):

| Variant | n chunks | slice-mean KLD | PPL |
|---|---:|---:|---:|
| baseline (interleaved RoPE)   | chunk 0 only | 0.4794   | —     |
| **halfsplit RoPE (this fix)** | chunk 0 only | **0.0835** | —   |
| baseline (interleaved RoPE)   | 20           | 0.4945   | 33.20 |
| **halfsplit RoPE (this fix)** | 20           | **0.0806** | **18.56** |
| reduction (20-chunk)          |              | **-83.7%** | -44%  |

PPL dropped from 33.20 → 18.56, very close to plain Qwen3-0.6B's 19.5 (i.e., near the model's true intrinsic PPL on this slice).

The ~0.4 nat engine-drift floor on Qwen3.5 family models is **largely an interleaved-vs-halfsplit RoPE convention mismatch in hipfire-arch-qwen35**, fixed by swapping kernels with no weight-permutation required. The remaining ~0.08 KLD is consistent with the per-layer ~0.06 DeltaNet drift seen across LA layers and is the next investigation target (likely Q8 weight quant noise compounded through the DeltaNet recurrent state).

Files:
  - `kernels/src/rope_partial_halfsplit.hip` (new)
  - `crates/rdna-compute/src/kernels.rs` (register kernel src)
  - `crates/rdna-compute/src/dispatch.rs` (`rope_partial_interleaved_f32` env-gate to halfsplit)

**Default flip landed 2026-05-12 evening** — `rope_partial_interleaved_f32` and `rope_partial_interleaved_f32_batched` now dispatch the halfsplit kernels by default; `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1` reverts to the pre-flip interleaved kernels for legacy reproducibility (regression probes, comparisons to historical benches). Method names retained for source-tree stability — a future rename PR can sweep call sites. Twin batched kernel `kernels/src/rope_partial_halfsplit_batched.hip` added to keep the prefill path correct.

Verified locally with two single-chunk probes on Q3.5-0.8B (per-token kv-q8):
  - Default build (halfsplit) → KLD 0.0835 ✓
  - With `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1` (legacy interleaved) → KLD 0.6044

Note on the 0.6044 vs the historical 0.4945 baseline (kv-q8 n=20, same model): the legacy gate produces a slightly higher KLD on chunk 0 than the historical aggregate. Likely a small numerical interaction from concurrent AWQ work that landed on this branch around the same time (this branch is a rebase target for parallel Phase A work). The reduction headline still holds: halfsplit reduces the chunk-0 KLD ~7× vs the legacy gate (0.6044 → 0.0835) on this branch, and ~6× vs the historical baseline (0.4945 → 0.0835).

Promotion-path remainder:
  - Bench the flip on the 9B model + at full slice (validate magnitude holds at scale).
  - Coherence-gate before considering this load-bearing on shipped quants (gate currently shows a pre-existing daemon SIGSEGV on shutdown — unrelated to the flip but blocks the gate from passing; needs a separate investigation).
  - Cohort re-bench: every previous Q3.5 quality-eval result was sitting on a ~0.4 nat pedestal that's now gone. `KLD(MQ4) − KLD(Q8)` deltas may need rebaselining; the Q8 control row in each cohort already isolates the 4-bit-attributable component, but headline absolute KLDs need the new floor.

Raw per-layer dumps at `/data/cache/hipfire/q3.5-0.8b-{hf,hipfire}-hidden-chunk0.bin` (not committed, 192 MB each); the comparator output is reproducible from the scripts in `scripts/dump_hf_hidden_states.py`, `scripts/compare_hidden_states.py`, and the new example binary.

**Cross-engine sanity check (Step B, 2026-05-12 evening) — confirms Step A is well-posed.**
Before starting per-layer instrumentation, validated that the BF16 reference itself is consistent across engines. Ran `scripts/cross_engine_check.py` — extracts chunk-0 tokens from `qwen3.5-0.8b-bf16.kldref.bin` (built by llama.cpp), runs the same 2048 tokens through HF transformers BF16 on CPU, computes the per-position KL divergence between llama.cpp's stored top-K distribution and HF's distribution on the same chunk.

| Direction | Mean KLD (chunk 0, 1023 positions) | Max |
|---|---:|---:|
| KL(llama.cpp || HF transformers) | 0.000669 | 0.0056 |
| KL(HF transformers || llama.cpp) | 0.003287 | 0.0266 |
| Symmetric mean                   | **0.001978** | — |

llama.cpp and HF transformers BF16 agree on Qwen3.5-0.8B chunk-0 to within ~0.002 nats. Hipfire's matched-arch floor (Q3.5-0.8B q8f16 per-token kv-q8 n=20) is 0.4945 — **~250× larger than the cross-engine baseline**. The drift is not "two engines reasonably differ on a complex architecture"; it's a real hipfire-specific implementation gap relative to a reproducible ground truth. Step A (per-layer hidden-state KLD vs HF oracle) is therefore well-posed: there is a target to chase, and a fix is in principle obtainable.

**Sequencing summary:**

| stage | status (2026-05-12 PM) | wall budget | gates | strategic priority |
|---|---|---|---|---|
| Stage 0 (Q8 floor cohort) | ✅ done | ~2.5h | none | critical: required interpretation prerequisite |
| Stage A (AWQ on MQ4) | ✅ shipped + validated (9B: −32.6% above-floor KLD, −5.0% PPL) | ~1.5-2 weeks | Stage 0 | **highest** — cheapest path; AWQ works at MQ4's g=256 (per-block large-group) |
| Stage B (GPTQ on MQ4) | ⏸️ pending | ~2 weeks | optional after Stage A | high — stacks additively; builds Hessian collector for Stage C |
| **Stage C (MR-GPTQ on MFP4)** | ⏸️ pending | **~2-3 weeks** | after Stage B (shares scaffolding) | **paper-validated for MFP4** — might make Stage E unnecessary |
| Stage D (UD kmap) | ⏸️ pending | ~1-2 days | independent of A/B/C | medium — small effort, orthogonal lever |
| Stage E (MQ4K wire format) | ⏸️ DEFERRED | ~5-7 weeks | DEFERRED, gated on A+B+C+D | only if calibration on existing formats falls short |

#### Stage A — shipped 2026-05-12 (PM)

| Phase | What landed | Where |
|---|---|---|
| Phase 1 — Quantizer | AWQ pre-scaling for MQ4G256: `compute_awq_scales` (log-space accumulation, geo-mean normalized), `awq_pre_scale_weights` (in-place `W' = W · diag(s)`), F16 sidecar emission `<weight>.awq_scale.weight`, CLI flags `--awq` / `--awq-alpha` | `crates/hipfire-quantize/src/main.rs` (commit `83054300`); 5 new tests in `awq_tests` module |
| Phase 2a — Runtime infra | `awq_scale: Option<GpuTensor>` field on `WeightTensor`; sidecar loader (F16→F32 host-side); `fused_rmsnorm_mq_rotate_awq` HIP kernel (extra per-channel divide before FWHT); dispatch wrappers | `crates/hipfire-runtime/src/{hfq,llama}.rs`, `crates/rdna-compute/src/{dispatch,kernels}.rs`, `kernels/src/fused_rmsnorm_mq_rotate_awq.hip` (commit `e51a3cd9`) |
| Phase 2b — Forward-pass wiring | Helper `fused_rmsnorm_rotate_mq_batched_for` that takes the next-linear's `WeightTensor` and dispatches AWQ variant when it carries `awq_scale`; converted 10 call sites across decode + batched paths | `crates/hipfire-runtime/src/llama.rs`, `crates/hipfire-arch-qwen35/src/qwen35.rs` (commit `a4265ce4`) |
| Bench infra fix | Replaced broken `tail -1 serve.log \| grep "warm-up complete"` race + obsolete `pgrep "examples/daemon"` fallback with `wait_for_model_ready` (polls `/v1/models` for requested basename) | `scripts/quant_cohort.sh`, `scripts/bench_humaneval_completion.sh` (commit `21772a4d`) |

Sidecar emission verified: AWQ-quantized 9B has 248 `awq_scale` tensors stored (matches 32 layers × ~8 linears/layer); baseline has 0. AWQ-disabled code path is byte-identical to pre-AWQ.

**Phase 3 (completed 2026-05-12 evening) — Stage A end-to-end bench.** Initial 0.8B cohort surfaced TWO bugs in the Stage A pipeline, both fixed before final measurement:

1. **Quantizer over-scoping** (commit `6711308d`): the AWQ pre-scaling was applied to every MQ4G256 weight with imatrix data, including the post-attention/post-silu projections (`wo` / `o_proj` / `out_proj` / `down_proj`) whose runtime path uses `rotate_x_mq` and `fused_silu_mul_rotate_mq` — neither of which has an AWQ inverse. Pre-scaled weights met undivided activations → `(W·s)·x ≠ W·x` corruption per channel. Fixed via a whitelist guard (`awq_eligible(name)`) that only pre-scales tensors whose runtime path will apply the divide. MoE completeness follow-up at commit `ca759da6`.
2. **Qwen3.5 crate's private loader hardcoded `awq_scale: None`** (commit `0aa58185`): the Phase 2a `load_awq_scale` infrastructure shipped in `hipfire_runtime::hfq` was simply never invoked from the Qwen3.5 forward path because `crates/hipfire-arch-qwen35/src/qwen35.rs:742` has its own `load_weight_tensor_raw` that returned `awq_scale: None` for every weight type. Detection signal: the AWQ kernel `fused_rmsnorm_mq_rotate_awq.hsaco` did not exist on disk before this fix — kernels are JIT-compiled on first dispatch, and the dispatch's `awq_scale.is_some()` check was always false. Fix mirrors the upstream loader closure inside the Qwen3.5 crate.

Detailed diagnosis trace in `docs/plans/awq_fix_claude.md` (rebuttal + measured results) and `docs/plans/awq_bug_hunt_glm5.md` (independent agent's diagnosis of the first bug). Cost from "first bad cohort" to "validated 9B Stage A win": ~6 hours.

**Final Stage A measurements (Qwen3.5, gfx1100, asym3 KV, prefill scoring, 512 chunks):**

| Variant | KLD | Above Q8 floor | PPL | AWQ delta |
|---|---:|---:|---:|---:|
| 0.8B q8f16 (floor) | 0.4598 | — | 30.996 | — |
| 0.8B mq4-base | 0.6721 | +0.2123 | 36.594 | — |
| 0.8B mq4-awq-loaderfix | 0.6707 | +0.2109 | 37.31 | **−0.7% (within noise)** |
| 9B q8f16 (floor, 256ch) | 0.5735 | — | 13.383 | — |
| **9B mq4-base** | **0.8165** | **+0.2430** | **15.063** | — |
| **9B mq4-awq-loaderfix** | **0.7373** | **+0.1638** | **14.303** | **−32.6% / −5.0% PPL** |

The 0.8B → 9B scale dependence matches AWQ paper predictions: outlier preservation gains grow with model size because outlier severity scales with parameter count. **Stage A AWQ is a real Phase A quality lever on Qwen3.5-9B and beyond**, and the infrastructure (Phase 2a dispatch + Phase 2b call-site wiring + sidecar storage) carries forward into Stage B (GPTQ) and Stage C (MR-GPTQ) without rework — only the *calibration* algorithm changes.

**Open follow-up (Option B from `awq_bug_hunt_glm5.md`):** add AWQ-aware variants of `rotate_x_mq` and `fused_silu_mul_rotate_mq` (4 new HIP kernels + dispatcher wiring) so `o_proj` / `out_proj` / `down_proj` can also benefit from AWQ. Literature suggests another few percent KLD reduction. Deferred until Stage B/C measure-up — if the AWQ+GPTQ+MR-GPTQ stack still leaves a meaningful gap to UD-Q3_K_XL at equal bpw, revisit then.

**What the Steps 1-7 work above bought us even though the pivot reframes the plan:** the infrastructure to MEASURE all of this is in place — `quant_cohort.sh`, `eval_hipfire`, `imatrix_collect`, `ud_decompile`, BF16 kldref files for 9B and 0.8B. The bench loop is ~25 minutes per cohort variant on 9B. Stage A AWQ can iterate against measurement at that cadence, which is the engineering velocity needed to actually validate a calibration lever empirically (not just predict its impact from a lever-decomposition spreadsheet).

**Foot-gun discovered 2026-05-12:** `hipfire-quantize` defaults to `--format q8f16`. To get MQ4G256 you must pass `--format mq4` explicitly. The CLI's `--help` only prints `--input` / `--output`. Cost: one wasted quantize pair (~7 min CPU + 18 GB disk reclaimed) before noticing the output was Q8 instead of MQ4. Future cohort specs should embed the full quantize command, not just the output path.

### Phase B — Format-family fillout (3–4 weeks, can run in parallel with Phase A)

8. HFP3G32 (qt=30) — E2M0 + UE8M0 + FP16 row. Quantizer + correctness-anchor kernel. Bench: 3.0 bpw with median MSE ~3× of HFP4G32, but on dormant tensors (low ZD score) the model-aggregate KLD impact is negligible.
9. HFP6G32 (qt=31) — INT6 + UE8M0 + FP16 row. Promotion target for hot tensors.
10. HFP8E4M3G32 (qt=27) — Q8-equivalent at 8 bpw. Replaces hardcoded "router/embedding/lm_head → Q8" with a real HFP-family member that has the same row header and dequant convention.

### Phase B' — gfx906 kernel port (2–3 weeks, parallel with everything above)

**Prerequisite reading:** `docs/plans/gfx906-moe-kernel-gaps.md` —
3-way-cross-validated audit identifying 8 existing gaps in the
gfx906 MoE kernel coverage relative to dense (no dp4a in MoE GEMVs,
no MMQ port for MoE prefill, no prefetch variant, etc.). 6 of those
8 gaps map 1:1 onto Phase B' work; the port should close them in
passing rather than carrying them forward into HFP4.

11. Port the `gemm_hfq4g256_residual_mmq_gfx906_x{8..64}.hip` family to HFP4G32. The dp4a inner loop is byte-identical (per §1.4); only the per-block scale conversion (`v_ldexp_f32` instead of FP32 load) and the LUT-decode (`kvalues_hfp4[n] * sc` instead of `(n - 8) * sc`) change. Projected: 87–95% of stock llama.cpp pp512, vs current HFQ4G256 at 95%.
12. Port `fused_qkv_hfq4g256_wave64_dp4a.hip` and `fused_gate_up_hfq4g256_wave64_dp4a.hip` (and the qkvza variant) to HFP4G32. Same template diff as above. **Add the MoE variants this time** (the audit's Gap 1, 8): `fused_qkvza_hfp4g32_moe_wave64_dp4a.hip` for the preamble, `gemv_hfp4g32_moe_{gate_up,down}_indexed_wave64_dp4a.hip` for the routed experts.
13. Close audit Gap 2 in the same PR: write `gemm_hfp4g32_residual_mmq_gfx906_moe_x{8..64}.hip` variants for MoE prefill (per-expert sort + tile, standard vLLM/TensorRT pattern). Projected: +50–100% prefill at batch ≥ 16, MoE on parity with dense.
14. Close audit Gap 3 in the same PR: wave64 variant of `gemv_hfp4g32_residual_sigmoid_scaled_gpu` (shared expert down). ~half-day fix per audit, but should land alongside the family port to avoid leaving 50% lane utilization on the table.
15. Speed-gate: re-run `scripts/probe_commits.sh master HEAD` on Qwen3.5-9B-HFP4 + Qwen3.6-A3B-MFP4 on the gfx906 dev box. Dense must come in within 10% of HFQ4G256-baseline pp512; MoE must improve on today's scalar-FP wave64 baseline by 5%+ on decode and 50%+ on prefill (the audit's projected impact).

If gfx906 port lands later than Phase A: gfx906 users continue on HFQ4G256 quants (no regression), Phase A's calibrated-MFP4 ships for RDNA3/4 users immediately. The gfx906 retroactively gets the quality lift when the port lands.

If pp512 regression exceeds 10%, fall back to HFP4XSG256 (plan B per §1.4): one super-FP16 + per-32 INT6 scales + 16-code E2M1 LUT, qt=28 or qt=29. Identical LDS topology to HFQ4G256; quality slightly below HFP4G32 (the per-32 INT6 super-block scale is coarser than UE8M0 + FP16 row) but still strictly better than MQ4G256.

### Phase C — Arch coverage (2–3 weeks, can run in parallel with Phase B)

11. Tensor-class enum + per-arch classifier (§4.2(ii)). Refactor K-map + engine loaders to consume `TensorClass` instead of substring greps.
12. Gemma 3/4 arch crate. GemmaRMSNorm is already correct; just need arch glue + tensor-class classifier + tokenizer wiring.
13. Qwen2.5-VL arch crate. Older naming convention handled in classifier.

### Phase D — Validation as canonical bench (1 week)

14. Standing benchmark: extend `scripts/bench_quant_quality.sh` to run the full MSE + KLD + smoke-test triple under three configurations (uncalibrated MFP4, calibrated MFP4, calibrated MFP4 + per-tensor bit allocation) on three reference models (Qwen3.5-9B, Qwen3.6-A3B, a Gemma3 family member). Lock as the regression bar.
15. Update CLAUDE.md: any quantization-format change MUST regress this triple.

---

## 6. Closing — what this means strategically

The question we set out to answer: *"what's the current problem, and how do we get from where we are to a 4-bit format that beats Unsloth Dynamic GGUFs in quality, given our rotation lead and modern element format?"*

**The current problem:** the rotation + modern element format (L1+L2+L3) we have **is genuinely a quality win** — MFP4 measurably beats MQ4 by 3–5× per-weight MSE, and at lever-for-lever parity should beat unrotated Q4_K_M by ~1.0–1.3× MSE. But Unsloth's lead comes from a different lever entirely: activation-aware calibration (L4 weighted LS + L5b/c/d imatrix). Without calibration, even a theoretically-better format underperforms a calibrated worse format on what users actually measure (KLD, instruction-following, reasoning coherence).

**The path:** Phase A (4–6 weeks) closes the calibration gap inside the existing HFP4 wire format. The reserved bits in the row header, the configurable rotation_kind, and the reserved quant_type IDs were designed exactly for this kind of staged extension — we don't need a new format family, we need to use the one we have.

**The strategic posture:** **HFP4 is the format we should commit to for the next several years.** It is:
- Architecturally extensible (4 rotation modes, 4 reserved IDs, 16 reserved metadata flags, 1 reserved sidecar pointer)
- RDNA-optimal (UE8M0 → free `v_ldexp`; FP16 row → cheap WMMA epilogue; E2M1 → wire-compatible with future MXFP4 silicon)
- **CDNA1/gfx906-accelerable** (the dp4a inner loop is byte-identical to HFQ4G256; LUT-decode pattern is proven by llama.cpp MXFP4; projected 87–95% of stock llama.cpp pp512 — see §1.4)
- Arch-portable (zero per-arch fields in the per-tensor format; arch concerns ride in tensor-class classifier + per-tensor metadata sidecar)
- Quantization-quality-extensible (all of L4 + L5 closes inside the existing wire format)

If Phase A delivers what the lever-by-lever math predicts, hipfire ships an MFP4 + calibrated quantizer that **beats UD-Q4_K_XL on KLD at equal bpw, while also being 70–75% of theoretical FP16 WMMA throughput on RDNA3+**. That's a quality lead AND a perf lead, on a wire format we own.

The risk is Phase 11's lesson: the A3B `<think>` spiral was not closed by reducing per-weight noise. If after Phase A the spiral still fires, the answer is not more quantization work; it's the runtime-side investigation noted in `docs/plans/qwen35-moe-coherence-investigation.md` §"Phase 11 engine-pass implications" (sampler intervention, vLLM FP16-router contract, period-N block-attractor detection). Quantization closes the quality gap; runtime closes the residual coherence risk.

### 6.1 Update 2026-05-12 — what changed after the cohort

The strategic claims above were authored before the Step 0.5 / Step 1+2 / Step 5a cohorts and the GGUF-anchor cross-reference. Three updates:

**Update 1 — HFP4 vs MQ4 as the "format to commit to" is genuinely open.** The §1.5 cohort surfaced that MFP4 baseline measures +37% KLD over MQ4 baseline at 9B (and +94% at 0.8B). Even calibrated, HFP4-L4-L5c at 9B is +3% KLD over the uncalibrated HFP4 baseline (and HFP4 baseline itself is +21% over MQ4). At bpw-matched comparison MQ4 is the clearer Phase A baseline, not MFP4. The "HFP4 is the format for several years" claim above is **suspended pending evidence that calibrated-HFP4 outperforms calibrated-MQ4** — that measurement doesn't exist yet (Step 5b for MQ4 calibration isn't built; current plan pivots to AWQ/GPTQ on MQ4 instead — see §5 Phase A revised).

**Update 2 — the calibration mechanism mattered more than the gap doc projected.** The lever-by-lever framing predicted L5c would close most of the gap. Empirically L5c (per-block weighted-LS) gave +9% KLD on 0.8B HFP4 and ~0% net on 9B HFP4, and is math-incompatible with rotated formats. The pivot to AWQ (per-channel pre-scaling, composes with rotation) + GPTQ (Hessian-aware sequential, zero infrastructure change) is the response — both are quantizer-side improvements on the existing MQ4 wire format that don't require the wire-format change implied by "ship HFP4 for several years."

**Update 3 — the bpw-matched UD anchor gap is bigger than the framing acknowledged.** Hipfire MQ4 sits at +75% PPL behind UD-Q3_K_XL at hipfire's actual aggregate bpw (~4.87) and Unsloth's matched anchor (~4.50), DESPITE hipfire having +0.37 more bits per weight. Closing this gap is the real Phase A target; the gap doc §2.4 estimate ("a couple levers behind Unsloth at equal bpw") understated the work. Detail: `docs/plans/mq4v2-format-proposal.md` §1 + the 9B cohort comparison.

**What's robust about the original strategic claim:**
- Rotation is a unique hipfire lever that no GGUF variant has (kept in all paths forward)
- Wire-format extensibility analysis (§3) still holds — when we DO need to extend, we have headroom
- Runtime-side coherence work (Phase 11 lessons) is orthogonal — none of the Phase A cohort surprises change that

**What changes:**
- The default format isn't pre-committed. Step 5b on MQ4 (via AWQ/GPTQ — §5 Phase A revised) decides.
- The "HFP4 + calibration beats UD-Q4_K_XL" projection is downgraded from "predicted" to "open empirical question pending Stage A+B+C measurement"
- MQ4K (new wire format with Q4_K-style sub-scaling, per `mq4v2-format-proposal.md`) is the fallback option if AWQ+GPTQ on MQ4 don't close the gap. Not Phase A's first lever, but the cost is bounded (~5-7 weeks total deployment) and the decision criterion is clear.

---

## References

### In-tree
- `docs/quant-formats/hfp4.md` — HFP4 wire format spec (reservations, taxonomy, kernel targets)
- `docs/plans/qwen35-moe-coherence-investigation.md` — parent investigation; Phase 11 falsifies "format quality → spiral" link
- `crates/hipfire-quantize/src/main.rs` — quantizer (MQ4/MQ6/HFP4/MFP4 functions, K-map rules)
- `crates/hipfire-runtime/src/hfq.rs` — `.hfq` file format reader
- `crates/hipfire-runtime/examples/quant_quality_mse.rs` — per-tensor MSE harness
- `crates/hipfire-runtime/examples/compare_hfq.rs` — tensor-by-tensor NRMSE diff
- `scripts/bench_quant_quality.sh` — combined MSE + smoke triple (predecessor to `quant_cohort.sh`)
- `scripts/quant_cohort.sh` — Phase A Step 0 cohort runner; orchestrates MSE + (KLD stub) + (PPL stub) + HumanEval + reasoning smoke per variant
- `scripts/bench_humaneval_completion.sh` — per-variant HumanEval prompt completion capture

### External
- OCP Microscaling Formats (MX) v1.0 — element format reference for E2M1
- AMD ROCm Blog "High-Accuracy MXFP4, MXFP6" — UE8M0 + FP16 row design
- AMD ROCm Blog "Advanced MXFP4 with Online Rotation" — block-diag-128 rotation (HFP4 rotation_kind `10`)
- NVIDIA NVFP4 announcement — E4M3 scale + FP32 tensor scale reference
- llama.cpp `tools/imatrix/README.md` — calibration methodology (PR #4861)
- llama.cpp `tools/quantize/README.md` — `--tensor-type` regex, `--imatrix` flags
- Liu et al. 2024, "Layer-Wise Quantization" (arXiv 2406.17415) — ZD score per-tensor importance metric
- QuaRot (arXiv 2404.00456) — full-hidden-dim rotation + GPTQ calibration
- SpinQuant (arXiv 2405.16406) — Stiefel-manifold rotation optimization
- Unsloth Dynamic 2.0 — marketing page (`https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs`); algorithmic details derive from underlying llama.cpp tools, not the page itself
