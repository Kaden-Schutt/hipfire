# Accelerate Q8 long-context flash attention

Follow-up to `fix/q8-batched-masked-no-lds-cap` (committed `abd95245`).
The new batched Q8 kernel is correct (NIAH PASS @ 21551 tok) but scalar:
**10 tok/s** at 21.5k ctx. This plan accelerates it.

## Root cause — MEASURED (2026-05-26), hypothesis REVISED

Original guess was DRAM-bound (GQA 4× K/V reload). **Refuted by
microbench BW measurement:** the kernel achieves **<1% of gfx906's
~954 GiB/s peak** (0.9–7 GiB/s on unique bytes; ≤~14 GiB/s even with the
×4 GQA reload and ÷2 causal factored in). The K/V cache is 16–33 MiB —
fits in L2. **DRAM is not the bottleneck. GQA-reuse is the WRONG lever.**

The kernel is **compute/issue-rate-bound** (consistent with
`feedback_gfx906_mmq_y128_sweet_spot`: gfx906 is issue-starved). The
Phase A loop in `attention_flash_q8_0_tile_batched` (and its asym3
parent) processes **one token at a time**:

```
for t_local in 0..tile_len {        // 128 SERIAL iterations
    all 32 lanes compute partial for the SAME token t
    __shfl_xor reduce → one score   // 31/32 lanes' work discarded
    if tid==0 { scores[t_local] = ... }
}
```

So the wave does ~32× redundant compute and serializes over 128 tokens.
Phase D (V accumulation) has the same per-token serial shape. This is
why BW is ~1% — it's not moving bytes, it's spinning on redundant FMAs.

rocprofv3 cores out on this ROCm 6.4 / gfx906 combo (dumped core on
`--pmc`); BW-from-wall-time is the reliable signal here.

## The real lever: parallelize over tokens — TRIED, FALSIFIED (2026-05-27)

Built `attention_flash_q8_0_tokpar` on that hypothesis (each thread owns
whole tokens, full head_dim dot, online softmax, writes out directly).
**VERIFIED correct** (1.7e-6 vs tile_batched) but **0.63–0.70× — SLOWER.**

Kernel-metadata diff (gfx906, via gfx-kernel-metadata skill):

| kernel | VGPR | spill | wave | waves/SIMD |
|---|---:|---:|---:|---:|
| tile_batched (current) | 31 | 0 | 64 | **8** |
| tokpar (new) | 42 | 0 | 64 | 5 |

tile_batched is ALREADY at 8 waves/SIMD, 0 spills — good occupancy. The
whole-head_dim serial dot in tokpar raised VGPR to 42 (occupancy 8→5)
AND removed the head-dim ILP that tile_batched's 32-lane split provides.
Both hurt. <1% peak BW rules out memory-bound. High occupancy + slow ⇒
**instruction-issue / VALU-bound** on the scalar Q8 dequant-dot.

## Real acceleration: matrix/dp4a, per arch (deferred)

Scalar tuning is exhausted: tile_batched is near-optimal for a scalar
kernel (8 waves, 0 spills, lane-split dot). The remaining lever is to
issue fewer, wider dot instructions:

- **gfx906 (Vega 20 / MI50, GCN5): dp4a, NOT MFMA.** gfx906 has no
  matrix cores — MFMA starts at gfx908 (MI100, CDNA1). gfx906's lever
  is `__builtin_amdgcn_sdot4` (`v_dot4_i32_i8`): 4 int8 MACs/instr.
  The Q8 K codes are already i8; quantize Q to i8 on load, dp4a the
  Q·K dot (×8 fewer issue slots for head_dim=256 → 64 dot4 ops vs 256
  FMAs). The project already gates dp4a on gfx906 for GEMVs
  (`gemv_dp4a_enabled`). CAVEAT: `feedback_dp4a_prefetch_no_op` — dp4a
  is a no-op if the body is already latency-hidden; but this body is
  ISSUE-bound (the opposite), so dp4a is plausibly a real win here.
  Risk: i8-quantizing Q loses precision — must NIAH-gate.
- **gfx908/90a/942 (CDNA): MFMA** (`v_mfma_*`) — if the serve ever runs
  on a CDNA matrix arch. Not our local hardware.
- **gfx1100+: WMMA** — the commit-2 path below (off-machine).

All are substantial (dp4a ~moderate, MFMA/WMMA ~300 LOC each). They are
the genuine path past 10 tok/s, but out of scope for the cliff-fix PR.
The fix PR ships the correct scalar tile_batched (NIAH-PASS, 1.5–2.2×
over the per-position fallback). tokpar stays behind HIPFIRE_Q8_TOKPAR=1
(default OFF — measured regression) as a scaffold for the dp4a rewrite
(its per-thread whole-token dot is the natural place to drop in sdot4).

## dp4a kernel — BUILT, CORRECT, MODEST WIN (2026-05-27)

`attention_flash_q8_0_dp4a.gfx906.hip`: same tiled-partials shape (reuses
asym reduce), Q quantized to i8 per-32-block once/head, Q·K via 2×
`__builtin_amdgcn_sdot4` (ILP2). Per the skyne98 gfx906 studies (dot4/dot8,
quant-dequant ISA, latency-hiding).

- **VERIFY:** 3.7e-3 vs tile_batched (Q i8-quant error — acceptable).
- **NIAH 32k (21551 tok): PASS** — needle recovered, correct at scale.
- **Microbench: 1.24–1.27× over tile_batched, 2.6–2.8× over fallback.**
  31 VGPR, 0 spills, 8 waves/SIMD (no occupancy cost), 2 v_dot4 emitted.
- Gate: `HIPFIRE_Q8_DP4A=1`; `dp4a_default=false` until the e2e picture
  justifies default-on. To enable: flip `dp4a_default` to `self.arch == "gfx906"`.

**E2E reality check (important):** end-to-end NIAH prefill stayed ~10
tok/s (2.22M ms) — UNCHANGED from tile_batched, despite the 1.25× kernel
win. Attention is NOT the e2e prefill bottleneck at this scale; the
projection GEMMs + per-chunk structure dominate wall time. So dp4a is a
real win on the attention-kernel axis but does not move e2e prefill. The
e2e lever is elsewhere (GEMM prefill path / PR #335-class gate work).

## Remaining dp4a headroom (next, if pursued)
- **Phase D (V accumulation) is still scalar f32** and is now the
  dominant cost inside the attention kernel. dp4a doesn't apply (scores
  are f32). Options: (a) quantize scores to i8 + sdot4 the V dot (accuracy
  risk — NIAH-gate); (b) `v_dot2_f32_f16` if V staged as f16; (c) leave it.
- ILP4 on Q·K is capped at ILP2 here (8 dims/lane = 2 dot4); a wider
  per-lane tile (16 dims/lane, 16 lanes) would unlock ILP4 (study: 2×
  on the dot) — but the dot is already not the sole cost.

## Commit 1 — optimized scalar kernel (gfx906 + gfx1031, testable here)

**Basis: token-parallel structure from `attention_flash_gqa.hip`
(decode kernel, commit 3b3b0b64).** Its `for t = tid; t < chunk_len;
t += nthreads` loop gives each thread distinct tokens — no per-token
wave reduction. That is the exact fix for our compute-bound Phase A.

New `attention_flash_q8_0_tokpar.hip` (token-parallel batched Q8):

1. **Token-parallel Phase A.** Grid `[n_heads, sub_batch]` (drop the
   tile grid dim). Block = larger (128/256 threads). Each thread loops
   `for t = tid; t < seq_len; t += blockDim` computing the FULL Q·K dot
   for ITS tokens (all head_dim, not a lane-split), writing `scores[t]`
   to LDS directly — NO `__shfl_xor` per token. One block now owns the
   whole context for one (head, query-row).
   - LDS: `scores[seq_len]` — but seq_len can be 120k → back to the LDS
     cliff! So keep the **online-softmax tiling**: outer
     `for tile in 0..n_tiles { token-parallel over the tile's tokens;
     tile max/sum; accumulate O with running correction }`. LDS stays
     `tile_size` (e.g. 256). This merges the gqa kernel's token-parallel
     inner loop with the current kernel's tiled-partials outer loop.
2. **Whole-head_dim dot per thread.** Each thread computes a complete
   256-dim Q8 dot for its token (vs today's 8-dim lane-split + reduce).
   More registers, but no reduction — and the dot is a tight FMA chain
   the issue-starved gfx906 can actually pipeline.
3. **Q8 dequant inline** — same fp16-scale × i8 body, full head_dim.
4. **Causal + tree mask** at the score write (per-token, in-loop).
5. **Partials layout** `[sub_batch × n_heads × n_tiles × stride]` →
   reuse `attention_flash_asym_reduce_batched` unchanged.
6. **Block dim** arch-tuned (gfx906 wave64-friendly 256; gfx1031 128).
7. **Dispatch gate:** select `_tokpar` on gfx906/gfx1031; keep the
   current `attention_flash_q8_0_tile_batched` as universal fallback and
   for tree-verify (small blocks, not the bottleneck).

**Note:** GQA K/V-reuse is dropped from the design — DRAM isn't the
bottleneck (measured <1% peak). If a later rocprof on a working setup
shows DRAM pressure at very long ctx, revisit; for now, token-level
parallelism is the lever.

**Verify (here, gfx906 + gfx1031):**
- Parity: new vs current kernel, max abs diff < 1e-3 on random Q/KV
  (extend `q8_batched_attn_microbench` with a `--verify` mode).
- NIAH 9B Q8 niah_32k (21551 tok) PASS — same needle.
- Microbench speedup vs current scalar (correct dims nh=16/nkv=4/hd=256).
  Target: ≥ gqa_ratio×0.6 ≈ 2.4× if DRAM-bound as predicted.
- coherence-gate (kernel change → mandatory).

## Commit 2 — WMMA fast path (gfx1100+, tested on the remote box)

WMMA is NOT on gfx906/gfx1031 (CDNA1 / RDNA2). Build + test on the
separate gfx1100 machine.

New `attention_flash_q8_0_wmma.hip`, adapted from
`attention_dflash_wmma_m64_n128_f16kv_v3_causal` (commit 3b3b0b64):
- **Q8 dequant on load:** the v3 kernel reads K/V already-f16 from DRAM.
  Q8 path dequants i8×fp16-scale → f16 register fragments at load time
  (Phase A K-load and Phase D V-stage), then feeds the SAME
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` MAC. Adds a per-fragment
  dequant but keeps the WMMA core.
- **head_dim=256:** v3 hard-returns if head_dim != 128. Loop 2 halves
  of 128 (d_chunks 8→16, or two passes) — the scalar kernels already
  show the n_halves pattern.
- **Masking:** v3_causal applies a causal mask. Our prefill uses
  `positions[]`-based causal + optional tree_bias. Port the per-row
  position cutoff into the S-write mask (same place v3 masks future
  keys).
- **Reduce/finalize:** v3 is single-pass O-resident with running m/l —
  no separate reduce kernel. Keeps the partials buffer out entirely.

**Verify (remote gfx1100):**
- Parity vs scalar kernel (the existing `parity_causal_wmma.rs` is a
  template).
- NIAH 9B Q8 niah_32k PASS on gfx1100.
- Microbench / real prefill tok/s vs scalar. Target ≫ 10 tok/s.
- coherence-gate on gfx1100.

## Dispatch arch gate (both commits)

```
attention_flash_q8_0_batched_masked:
  gfx1100|1101|1102|1200  -> wmma kernel        (commit 2)
  gfx906|gfx1031          -> scalar gqa kernel  (commit 1)
  else                    -> current portable kernel (fallback)
```

## Non-goals
- asym3/asym4 acceleration (separate kernels, same scalar problem — do
  later if the serve switches modes; serve is Q8 by quality requirement).
- decode-path attention (single-query GEMV, already a different kernel).
- Changing the reduce kernel for the scalar path (reused as-is).

## Interaction with PR #335
PR #335 (gfx1031 MMQ prefill gate removal) speeds the GEMM half of
prefill on RDNA2 — orthogonal to attention. When the 27B-awqg serve
runs prefill on gfx1031, #335 + the scalar-gqa attention kernel stack:
GEMMs faster (#335) AND attention faster (commit 1). Both needed for a
usable long-ctx serve prefill on the RDNA2 card.

## Risk / open questions
- LDS budget for staging V at tile=128 hd=256 (64 KB just for f16 V).
  May force tile=64 or K-only-register caching. Decide at rocprof step.
- dp4a on gfx906 may be a no-op (see memory) — gate + measure, don't
  assume.
- WMMA Q8-dequant-on-load may bottleneck on the dequant, not the MAC —
  measure on the remote box; if so, pre-dequant K/V to a f16 scratch
  once per prefill (trades VRAM for speed).
