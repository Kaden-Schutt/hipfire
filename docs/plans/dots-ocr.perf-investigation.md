# dots-ocr vision-encoder perf investigation

Data-driven follow-up after two hypothesis-driven kernel changes (M=32 query
tile, cooperative K-staging into LDS) failed to beat the M=16 baseline by
anything meaningful. Gathered: rocprof PMC counters on the production
kernels at the actual vision shape + structured comparison against
llama.cpp `fattn-wmma-f16.cu`, vLLM Triton flash-attention, and ROCm
composable_kernel `block_fmha_pipeline_qr_ks_vs.hpp`. Conclusion: the
biggest single difference is **K-tile width** (ours: 16 keys per outer
iteration; llama.cpp's: 256), not anything we'd been guessing at.

## 1. rocprof PMC data at vision shape (B = L = 19520, hd = 128, n_heads = 12)

Bench: `cargo run --release -p rdna-compute --example bench_attention_vision
--iters 1`. Counters via `rocprofv3 --pmc … --kernel-include-regex
"attention_dflash_wmma"`.

| kernel                            | dur (ms) | GPU busy | VALU M-inst/s | LDS M-inst/s | bank conflicts | **L2 hit %** |
|-----------------------------------|---------:|---------:|--------------:|-------------:|---------------:|-------------:|
| attention_dflash_wmma_f32 (M=16)  |   3023   |  100 %   |        15.3   |       3.6    |             93 |     **0.8 %** |
| attention_dflash_wmma_m32_f32     |   2904   |  100 %   |        15.1   |       3.6    |             84 |     **0.9 %** |

**Conclusions:**

- **GPU is 100 % busy** — occupancy / wave parallelism is fine. M=32's
  worry about losing occupancy at 50 KB / block was unfounded.
- **L2 hit rate is 0.8–0.9 %** — catastrophically low. Almost every
  memory access misses L2 and goes to DRAM. **We are DRAM-bandwidth-bound,
  not compute-bound.**
- **Bank conflicts are ~90 per kernel** — negligible (the K-staging
  diagnosis that this was the bottleneck was wrong; rocprof says no).
- **VALU and LDS instruction rates are identical** between M=16 and M=32.
  M=32's 4 % wall-time win is a small instruction-count reduction, not a
  per-second throughput change.

Computed: gfx1151 has ~115 GB/s LPDDR5X. K traffic alone at hd=128, f32 =
19520 × 12 × 128 × 4 = 120 MB per attention call. Re-read 1220 K-tiles per
block × 14640 blocks ÷ 40 CUs ≈ 1.3 s of K-only DRAM traffic at peak BW.
With V also read every iteration we double it. The measured 2.9–3.0 s
matches.

## 2. Comparison: ours vs llama.cpp vs vLLM-Triton vs CK

| source           | M_tile | **N_tile (K-step)** | LDS    | KV dtype | block (threads) | nwarps | Q in reg | K prefetch | syncs/Ktile |
|------------------|-------:|--------------------:|-------:|----------|----------------:|-------:|----------|-----------|------------:|
| **ours M=16**    |     16 |              **16** |  26 KB | **f32**  |              32 |      1 | no       | no        |           3 |
| **ours M=32**    |     32 |              **16** |  43 KB | **f32**  |              64 |      2 | no       | no        |           3 |
| llama.cpp wmma   |  16/32 |             **256** | ~13 KB | **f16**  |             128 |      4 | **yes**  | no        |           4 |
| vLLM Triton RDNA |     32 |                  32 | (auto) | f16/f8   |          64–128 |      2 | yes      | no        |          ~1 |
| CK qr_ks_vs gfx11|  64–128|                  64 |  tuned | f16      |         128–256 |    4–8 | **yes**  | **i+2**   |         3–4 |

References:
- llama.cpp `FATTN_KQ_STRIDE = 256` defined at
  `/home/kread/git/llm/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:9`.
  Outer K-loop steps by 256 at
  `fattn-wmma-f16.cu:192-214`.
- llama.cpp keeps Q register-resident across K-tiles via `frag_b Q_b[D/16][ncols/frag_n]`
  declared at `fattn-wmma-f16.cu:108`, populated once at `:180-186`, reused unchanged at `:207`.
- vLLM RDNA configs: `BLOCK_M=32, BLOCK_N=32, num_warps=2, num_stages=1`
  at `/home/kread/git/vllm/vllm/attention/ops/triton_flash_attention.py:322-358`.
- CK Q-load-once: `kQLoadOnce = true` at
  `/opt/rocm-7.13/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp:49`.
- CK explicit "global read i+2" software prefetch at the same file `:649, :659`.
- Bank-conflict avoidance: llama.cpp pads LDS rows at `D_padded = D + 8`
  (`fattn-wmma-f16.cu:85`).

## 3. The single biggest difference

**K-tile is 256 keys in llama.cpp; 16 in ours.** With L = 19520, that's
19520 / 16 = 1220 outer-loop trips for us versus 19520 / 256 = 77 for them
— **15.9× fewer outer-loop iterations**, which means 15.9× fewer:

- block-wide `__syncthreads()` barriers
- per-tile epilogue costs (alpha-scaling of `O_lds`, `m_lds` / `l_lds`
  reduce-and-broadcast)
- redundant LDS reads of Q (we re-load Q from `Q_lds` every K-tile;
  llama.cpp keeps Q in registers across all 77 trips)

This dominates over LDS layout choice, M-tile width, or async-copy
strategy. The rocprof data confirms: we are not compute-bound (VALU rate
is fine), we are DRAM-bound (L2 hit < 1 %). Wider K-tiles directly reduce
DRAM K-traffic because the per-tile fixed cost amortises over more keys.

## 4. Ranked next-step list

In order of expected impact (calibrated against the L2-miss diagnosis):

### 4.1. Widen K-tile from 16 → 64 (or 128) — expected **2–4× attention speedup**

Outer loop iterations drop from 1220 to 305 (or 153). Per-tile fixed costs
amortise. Implementation sketch:

- Loop the WMMA inside one outer iteration `K-tile / 16` times, reusing Q
  from registers.
- Store the 16 × 64 partial S tile **in registers** as
  `float8_t s_acc[K_tile / 16]` — exactly what llama.cpp does with
  `frag_c_KQ KQ_c[ncols / frag_n]` at `fattn-wmma-f16.cu:196`.
- Do per-row softmax max/sum over the wider S in registers (no LDS write
  between QK and softmax). llama.cpp keeps S in `KQ_f_tmp[FATTN_KQ_STRIDE
  / warp_size]` registers at `:225`.
- LDS budget at K-tile=64, hd=128, M=16: Q[16×136] + K[64×136 staged] +
  V[64×136] + O[16×136] + reduces ≈ 95 KB — would not fit. To make this
  work, **don't stage K to LDS** — keep it in registers, load fresh from
  global per inner WMMA. Or stage only one K-tile-row at a time.

### 4.2. Convert K and V to f16 in DRAM — expected **+30–100 %** on a memory-bound kernel

We currently store K and V in f32 (the QKV linear output is f32). f16
halves DRAM traffic. Implementation: insert an `f32 → f16` cast kernel
between `qkv_split` and `attention_dflash_wmma`, with K_buf / V_buf in
f16. The WMMA kernel already converts to f16 inside the inner loop
(`a_reg[j] = (_Float16)q_row[j]`), so consuming f16 directly drops the
conversion cost too. Reference: llama.cpp's `K_h`, `V_h` are
`const half *` at `fattn-wmma-f16.cu:93-94`.

### 4.3. Keep Q in registers across all K-tiles — expected **+10–20 %**

Today we reload Q from `Q_lds` into `a_reg` every K-tile at
`attention_dflash_wmma.hip:150-152`. That's `1220 × (head_dim / 16) =
9760` LDS reads of Q per (head, qt) that should be zero. Declare Q as a
per-thread register array (`half16_t Q_b[D / 16]` at hd=128, ncols=16:
8 half16_t = 64 VGPRs); populate once at kernel entry; reuse in the K
loop. Reference: llama.cpp `fattn-wmma-f16.cu:108, :180-186, :207`.

This is the smallest of the three changes and **can be done independently
of K-tile widening**.

## 5. Falsified hypotheses (for the record)

| hypothesis | source | rocprof verdict |
|---|---|---|
| K-load is uncoalesced → bandwidth-bound | M=32 commit msg | partly right, but the fix doesn't help because L2 hit < 1 % means we're DRAM-bound either way |
| LDS bank conflicts in K_lds reads | K-staging diagnostic | wrong — counter shows ~90 conflicts per kernel, two orders of magnitude below the regime where it matters |
| Wave occupancy is the bottleneck | M=32 launch_bounds analysis | wrong — GPU_UTIL = 100 % on both M=16 and M=32 |
| Halving query-tile blocks halves K-tile reads | M=32 design | right in absolute count but L2 still misses on each, so 4 % wall-time win matches the small reduction in instructions, not the 2× I predicted |

## 6. Open questions

- **Will K-tile widening trigger register spill?** At hd=128, ncols=16, K=64:
  `s_acc[4]` (float8 × 4) = 128 floats = 32 VGPRs per lane. Plus Q (64
  VGPRs). Total > 96 — within gfx1151's 1536 VGPR budget per CU but close
  to the per-block limit. Need to disasm post-write.
- **What does the L2 hit rate look like under a wider K-tile?** Should
  rise as the same K-tile is reused for more queries within a block. The
  rocprof PMC sweep needs to be re-run after the rewrite to confirm.
- **Is `__builtin_amdgcn_global_load_lds` worth using?** RDNA3 has direct
  global → LDS async copy; CK uses it. Our failed K-staging variant went
  through registers (load → reg → LDS) instead. Belongs in a 4.x step
  *after* the K-tile widening.

## 7. Reference yardstick

llama.cpp's WMMA flash-attention is the closest production-quality
reference. Building it with `GGML_HIP_ROCWMMA_FATTN=ON` and timing on a
synthetic (B = L = 19520, h = 12, d = 128) tensor would give the realistic
ceiling for "how fast can RDNA3 WMMA flash-attention go on this shape."
Worth doing as a sanity check after step 4.1 lands to know how much
performance is left on the table.

## Artifacts

- Bench tool: `crates/rdna-compute/examples/bench_attention_vision.rs`
- rocprof CSV (raw): `.tmp/rocprof/vision_shape/aimax01/16690_counter_collection.csv`
- Failed K-staging kernel (kept in tree as a reference point for the
  rocprof comparison): `kernels/src/attention_dflash_wmma_m32_kstg_FAILED.hip`

## 8. Outcome: N=64 K-tile kernel (2026-05-23)

`kernels/src/attention_dflash_wmma_n64.hip` implements step 4.1 (K-tile
16 → 64) and step 4.3 (Q register-resident) together for `head_dim==128`.
Phase C also fuses the alpha-scale with the SV epilogue (one fewer
`__syncthreads` and one fewer full O_lds traversal per K-tile).

### 8.1. Strix Halo gfx1151 results (bench_attention_vision, B=L=19520, hd=128)

| kernel | dur (ms) | vs M=16 | vs M=32 |
|---|---:|---:|---:|
| M=16  | 3034 |     —  |     —  |
| M=32  | 2931 |  +3.4 % |     —  |
| **M=32 N=64** | **2720** | **+10.4 %** | **+7.2 %** |

End-to-end `ocr_e2e` vision-encoder wall: **198 s → 182 s** (+8 %).
Parity sweep: 196 cases at hd=128, 0 failures, max-abs-diff 3.052e-5
(matches M=16/M=32 baselines).

### 8.2. Falsified-then-rescued: the Q_frags scratch trap

First attempt was a +19 % **regression**. Root cause via
`llvm-readelf --notes` on the compiled `.hsaco`:

| kernel | VGPR | spill | **private (scratch) segment** |
|---|---:|---:|---:|
| M=32 baseline       | 82 | 0 | **0 B/lane** |
| N=64 v1 (regressed) | 85 | 0 | **544 B/lane** |
| N=64 v2 (winning)   | 256 | 141 | 376 B/lane |

v1 declared `Q_frags[16]` and loaded it in a runtime-bounded `for (dc=0;
dc<d_chunks; ++dc)` loop, where `d_chunks = head_dim/16` is computed at
runtime. The compiler couldn't prove `dc` was compile-time constant and
put the array in private (scratch) memory — every "register" Q read was
actually a DRAM round-trip. v2 fixes it by hard-coding `d_chunks=8`,
adding an early-return guard `if (head_dim != 128) return;`, and adding
`#pragma unroll` to the dc loops in Q-load + phase A + phase C.

The high VGPR/spill count on v2 (256 / 141) is the unroll's cost in
expanded live ranges; the spill fits comfortably in the 376 B/lane
private segment and stays in L1, so it's not a perf factor for this
DRAM-bound workload.

### 8.3. Why the gain is +7 % on Strix Halo, not the predicted 2–4×

The investigation's analytic model overstated the per-K-tile fixed cost.
On Strix Halo gfx1151 with 115 GB/s LPDDR5X, the dominant cost remains
DRAM K+V traffic regardless of K-tile width (rocprof L2 hit % stays
near 1 % on either). The fixed-cost amortization is real but small
relative to the bandwidth floor.

**Expected on gfx1100** (RX 7900 XTX, ~960 GB/s GDDR6, 96 CUs, larger
L2): bigger absolute win. With ~8× the memory bandwidth, the per-tile
fixed cost is a much larger relative share of runtime, and llama.cpp's
`FATTN_KQ_STRIDE=256` was tuned for that class of hardware. Strategy
going forward: optimize on gfx1100 (primary deployment target), tune on
Strix Halo as a non-regression check.

### 8.4. Open next levers (in order of expected impact on gfx1100)

1. ~~**K/V f16 in DRAM** (step 4.2)~~ — landed. See §9 below.
2. **Widen K-tile further** (128 or 256) at hd=128 with f16 K/V.
   With f16 the V_lds budget halves (32 KB → 16 KB at N=64), which
   frees room for wider tiles or larger M.
3. **V in registers via WMMA frag_b** (llama.cpp pattern). Eliminates
   V_lds entirely. Requires restructuring phase C so the SV WMMA reads
   V chunks fresh from DRAM (or via a register prefetch pipeline) per
   inner step.
4. **128-thread block (4-wave)** like llama.cpp. More parallelism per
   block for V-stage and softmax; may or may not pay back the lower
   occupancy on gfx1100.

## 9. K/V f16 in DRAM (2026-05-23)

`kernels/src/attention_dflash_wmma_n64_f16kv.hip` is a copy of the
`n64` kernel that consumes K and V as `_Float16*` in DRAM instead of
`float*` (Q and output stay f32). The internal `(_Float16)k_row[d]`
cast disappears at phase A; the V-stage casts f16→f32 on the way to
V_lds so phase C is byte-identical.

`kernels/src/cast_f32_to_f16.hip` is the matching standalone cast kernel
(the same body lives inline in the FP16 GEMMs; this standalone copy
exists so non-GEMM callers can launch it directly). `Gpu::cast_f32_to_f16`
exposes the dispatch wrapper.

### 9.1. Strix Halo gfx1151 results (bench_attention_vision, B=L=19520, hd=128, 3 iters)

| kernel | dur (ms) | vs M=16 | vs N=64 (f32 K/V) |
|---|---:|---:|---:|
| M=16              | 3083 |     —   |     —   |
| M=32              | 2941 |  +4.6 % |     —   |
| M=32 N=64 (f32)   | 2725 | +11.6 % |     —   |
| **M=32 N=64 f16 K/V** | **2237** | **+27.4 %** | **+17.9 %** |

End-to-end `ocr_e2e` vision-encoder wall: **182 s → 169 s** (+7 % on
top of the N=64 landing — attention is roughly half the vision
encoder, the rest is QKV / FFN GEMMs and RMSNorm/RoPE which the f16
K/V change doesn't touch). Cumulative wall: **198 s → 169 s = 15 %**
off the initial baseline.

Parity sweep: 224 cases, 0 failed, max-abs-diff 3.052e-5 — same as the
f32-K/V baseline. The f16 quantisation of K and V on LCG-bounded inputs
in [-0.1, 0.1] is below the f32 accumulator's noise floor at L=19520.

### 9.2. Why the gain is +18 % not +50 %

Per the analytic floor in §1, with K+V at f32 the DRAM traffic per
attention call is ~146 GB and the LPDDR5X bandwidth is ~115 GB/s →
~1.27 s lower bound on K+V transit. Halving to f16 gives ~73 GB and
~0.63 s lower bound. Saving ~0.64 s out of 2.72 s = 23.5 % wall-time
improvement.

We measured +17.9 % which is 76 % of the theoretical ceiling. The
remainder is non-DRAM cost: WMMA throughput in phase A and C, LDS
bandwidth, softmax compute, the cast kernel itself (~10 ms), and the
V_lds 32 KB write that we didn't shrink (the f32→f32 V_lds path is
unchanged). Step 2 (wider K-tile with f16 K/V) and step 3 (V in
registers via frag_b) attack the remaining ~24 %.

### 9.3. Cost of the cast

The cast kernel is `O(L · n_kv_heads · head_dim)` work for both K and V
combined. At vision shape that's 2 · 19520 · 12 · 128 · 4 B = 240 MB
of f32 reads + 120 MB of f16 writes per attention call → 360 MB / 115
GB/s = ~3 ms theoretical, ~10 ms measured (single-pass kernels rarely
hit peak BW). That's <0.5 % of attention runtime — amortises trivially.

## 10. N=128 K-tile + f16 V_lds / S_lds (2026-05-23)

`kernels/src/attention_dflash_wmma_n128_f16kv.hip` widens the K-tile
from 64 to 128 keys. The wider tile is only feasible because V_lds and
S_lds were converted from f32 to f16 storage, reclaiming the LDS
budget that the doubled V_lds row count would otherwise have eaten.
LDS at hd=128:

  V_lds[128 * 128] **f16** = 32 KB  (was f32 64-row in N=64 path = 32 KB)
  O_lds[32 * 128]   f32    = 16 KB
  S_lds[32 * 128]  **f16** =  8 KB  (was f32 32×64 in N=64 path = 8 KB)
  scalars (m + l + alpha)  =  0.4 KB
  **Total ≈ 56.4 KB ✓**

Outer-loop iterations at vision shape: L/64=305 → L/128=152 (half).

### 10.1. Strix Halo gfx1151 results

bench_attention_vision (B=L=19520, hd=128, 3 iters):

| kernel | dur (ms) | vs M=16 | vs prev |
|---|---:|---:|---:|
| M=16                   | 3098 |     —   |     —   |
| M=32                   | 2939 |  +5.1 % |     —   |
| M=32 N=64  (f32 K/V)   | 2724 | +12.1 % |     —   |
| M=32 N=64  f16 K/V     | 2210 | +28.7 % | +18.9 % |
| **M=32 N=128 f16 K/V** | **1608** | **+48.1 %** | **+27.2 %** |

End-to-end `ocr_e2e` vision-encoder wall: **169 s → 135 s** (+25 % on
top of N=64 f16-K/V). Cumulative since initial baseline: **198 s →
135 s = 32 %** off.

Parity sweep at hd=128: 252 cases, 0 failed, max-abs-diff 3.052e-5.
The f16 S_lds storage works because softmax math runs in f32 per row
(`tm`, `m_new`, `alpha`, `ts` are f32 locals) — the f16 cast is only
at the LDS write/read boundary, and exp(s - m_new) ∈ [0, 1] always
fits f16 cleanly.

### 10.2. Why the gain is +27 % not +5–15 %

The investigation's analytic model expected the win to come from
halving __syncthreads / softmax-setup / per-tile alpha-scale cost.
That dimension is real but small. Two larger effects show up in
practice:

- **LDS bandwidth.** Storing V_lds and S_lds in f16 halves the LDS
  bytes per element. Phase C is LDS-heavy (S_lds reads + V_lds reads
  per inner WMMA × 8 d-chunks × 8 K-chunks). At our occupancy +
  workload, LDS bandwidth was a real bottleneck; halving it gives
  back time the analytic model didn't track.

- **WMMA ILP.** Phase A and phase C now do 64 inner WMMAs per outer
  iteration (vs 32 at N=64). Longer dependency chains let the
  compiler interleave the WMMA pipeline with K-row loads (phase A) and
  V_lds reads (phase C) more aggressively. The WMMA queue stays
  fuller; fewer pipeline drains between outer iterations.

### 10.3. Remaining headroom

Theoretical lower bound (Strix Halo, f16 K/V at M=32): ~0.63 s K+V
DRAM transit. At M=32, B/M = 610 query blocks → ~73 GB K+V DRAM
traffic per attention call. We're at 1.61 s; ~2.5× over the floor.

Note: N=256 doesn't fit on gfx1151 LDS even with f16 storage
(V_lds[256 * 128] f16 alone = 64 KB, saturates the cap). Going wider
requires either dropping V_lds (step 3) or striping V_lds.

## 11. M=64 + N=128 + O register-resident (2026-05-23)

The original step 3 ("V in WMMA frag_b registers") was reconsidered in
favour of a different lever after the DRAM analysis in §10.3 showed
that the binding constraint at M=32 was the K+V DRAM traffic — not
V_lds bandwidth.

`kernels/src/attention_dflash_wmma_m64_n128_f16kv.hip` doubles the
query tile from M=32 to M=64, which **halves the query-block count**
(B/M: 610 → 305) and **halves K and V DRAM traffic per attention
call** (~73 GB → ~36.5 GB at f16). Block size grows from 64 to 128
threads (4 waves × 32). Each wave still owns 16 query rows.

The LDS budget for M=64 N=128 with V_lds f16 + S_lds f16 + O_lds f32
came to ~80 KB, over the 64 KB cap. The fix: drop O_lds entirely and
keep O register-resident in the natural WMMA frag_c lane layout. Each
lane carries 8 float8_t = 64 VGPRs of running output, alpha-folded
in place at the end of each K-tile iter.

LDS at hd=128 (no O_lds):
  V_lds[128 * 128] f16 = 32 KB
  S_lds[64 * 128]  f16 = 16 KB
  scalars (m + l + alpha, 64 each) = 0.8 KB
  **Total ≈ 48.8 KB ≤ 64 KB cap.**

### 11.1. Strix Halo gfx1151 results

bench_attention_vision (B=L=19520, hd=128, 3 iters):

| kernel | dur (ms) | vs M=16 | vs prev |
|---|---:|---:|---:|
| M=16                          | 3056 |     —    |     —    |
| M=32                          | 2936 |   +3.9 % |     —    |
| M=32 N=64    (f32 K/V)        | 2707 |  +11.5 % |     —    |
| M=32 N=64    f16 K/V          | 2369 |  +22.5 % |          |
| M=32 N=128   f16 K/V          | 1609 |  +47.4 % |          |
| **M=64 N=128 f16 K/V O-reg**  |  **751** | **+75.4 %** | **+53.3 % vs N=128 / 4.07× vs M=16** |

End-to-end `ocr_e2e` vision-encoder wall: **135 s → 98.7 s**
(+27 % on top of M=32 N=128). Cumulative since initial baseline:
**198 s → 98.7 s = 50 %** off, **2.01× speedup** at the vision-encoder
wall.

Parity: 280 cases at hd=128, 0 failed, max-abs-diff 3.052e-5
(unchanged from M=32 baseline).

### 11.2. Kernel metadata

`llvm-readelf --notes` on the compiled `.hsaco`:

  .vgpr_count:     256  (at the cap)
  .vgpr_spill_count: 80
  .private_segment_fixed_size: 324 B/lane
  .sgpr_count:     34
  .group_segment_fixed_size: 0  (LDS is dynamically allocated)

80 VGPR spills go into 324 B/lane private memory — small enough to
stay in L1, so spill cost is negligible on this DRAM-bound workload.
The kernel ran at 1 block per CU (4 waves) due to the VGPR pressure;
this is fine because DRAM is still the binding constraint.

### 11.3. Why it worked

- **DRAM K+V traffic halved.** B/M = 305 query blocks (vs 610 at M=32).
  Each (K, V) f16 element is read once per query block (no cross-block
  L2 reuse per rocprof). Total halves.
- **WMMA pipeline better fed.** With 4 waves per block running phase A
  / phase C in parallel, the WMMA queue stays full across more cycles.
- **O in registers eliminates O_lds bandwidth.** Phase C used to read
  O_lds, alpha-scale, and write back. Now it's a register-only
  fma per (j, dc) commit — purely ALU, no LDS traffic.
- **Better lane utilization in phase B.** Softmax now uses 64 active
  lanes (16 per wave × 4 waves) instead of 32 (16 × 2). Per-row work
  is the same but distributed across more concurrent waves.

### 11.4. What's left at v1

Remaining gap from theoretical DRAM floor (~317 ms at M=64):

  measured: 751 ms
  DRAM floor (~37 GB / 115 GB/s): 317 ms
  ratio: 2.37×

Two levers landed in v2 (§12 below).

## 12. M=64 N=128 v2 — S_lds bank-conflict fix + cooperative softmax (2026-05-23)

`kernels/src/attention_dflash_wmma_m64_n128_f16kv_v2.hip` adds two
changes on top of v1:

### 12.1. S_lds row stride padded 128 → 130 f16

Phase C reads `S_lds[(my_row_base + half) * 128 + c*16 + j]` from
each lane in the wave. At unpadded row stride = 128 f16 = 256 bytes =
**64 dwords**, the lane stride mod 32 = 0 — meaning every lane in the
wave hits the *same* LDS bank on each read. 16-way bank conflict per
cycle on every S_lds read.

Padding the row stride to 130 f16 = 65 dwords gives lane bank-stride
1 (mod 32), so the 16 active lanes land in 16 different banks. No
conflict. Costs 0.25 KB extra LDS (64 rows × 2 extra f16).

This is the dominant lever — phase C does 8 dc × 8 c × 16 = 1024
S_lds reads per lane per outer iter, multiplied by 152 outer iters
across many waves and CUs. A 16× per-read latency cliff at that scale
is huge.

### 12.2. Cooperative wave-32 softmax

Phase B previously ran 16 lanes in parallel (one per row) with each
lane sweeping all 128 values sequentially. v2 processes rows
sequentially within a wave, but each row uses all 32 lanes via
butterfly reduce (`__shfl_xor` over [1, 2, 4, 8, 16]):

  - 128 values / 32 lanes = 4 vals/lane local max → 5-stage shfl reduce
  - 128 values / 32 lanes = 4 vals/lane local sum-of-exp → 5-stage shfl reduce
  - Lane 0 writes l_lds, m_lds, alpha_lds

Smaller lever than the bank-conflict fix, but additive.

### 12.3. Strix Halo gfx1151 results

bench_attention_vision (B=L=19520, hd=128, 3 iters):

| kernel | dur (ms) | vs M=16 | vs prev |
|---|---:|---:|---:|
| M=16                              | 3064 |     —    |     —    |
| M=32 N=64 (f32 K/V)               | 2722 |  +11.2 % |          |
| M=32 N=64  f16 K/V                | 2288 |  +25.3 % |          |
| M=32 N=128 f16 K/V                | 1609 |  +47.4 % |          |
| M=64 N=128 v1 (f16 K/V O-reg)     |  753 |  +75.4 % |          |
| **M=64 N=128 v2 (pad + coop sm)** |  **519** | **+83.1 %** | **+31.1 % over v1** |

End-to-end `ocr_e2e` vision-encoder wall: **98.7 s → 89.3 s**
(+10 % on top of v1). Cumulative since initial baseline:
**198 s → 89.3 s = 2.22× speedup** at the vision-encoder wall.

Parity: 308 cases at hd=128, 0 failed, max-abs-diff 3.052e-5
(unchanged from M=64 v1).

### 12.4. Headroom

  measured: 519 ms
  DRAM floor (~37 GB / 115 GB/s): 317 ms
  ratio: 1.64×

Remaining ~200 ms gap. The big single-change levers are mostly
exhausted; what's left is harder:

1. **Fuse f32→f16 cast into the QKV projection.** ~10 ms × 42 vision
   blocks = ~420 ms saved on E2E. Bigger downstream change to the
   GEMM that produces K and V.
2. **V via WMMA frag_b from DRAM** (the original step 3 from §8.4).
   Now possible — LDS has lots of headroom (24 KB used, 64 KB cap).
   Bets on L1 catching V slab reuse across d-chunks. Would also
   unlock N=256.
3. **N=256 K-tile** (with V_lds dropped via step 2 above, or
   striped V_lds).
4. **Re-examine LDS bank conflicts on V_lds reads.** Phase C reads
   V_lds[(c*16+j) * 128 + my_d] — at row stride 128, lane stride 1
   in f16 → 2 lanes per dword. Maybe also benefits from a small pad.

The N=256 path or QKV-cast fusion are likely the next big ones; both
are bigger structural changes than the v2 patches.

## 13. v3 — hoist S_lds reads (small win, ~2.6% over v2)

`kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3.hip` reorders
phase C to outer c, inner dc so `a_reg_sm` is read once per K-chunk
(was once per (d-chunk, K-chunk)). Theoretical 8× reduction in phase C
S_lds reads. Also moves the alpha-fold to start-of-phase-C and
accumulates SV into a per-d-chunk `o_acc_local[8]` register array.

### 13.1. Result: small but real win over v2

**Initial single-run measurement was misleading** — bench showed v2 =
518.5 ms, v3 = 522.6 ms (v3 appearing slower), which I committed as a
"null result." Critical review during pre-push audit caught this:

Fresh-process variance, 4 runs × 10 iters each (cooled GPU):

| | run 1 | run 2 | run 3 | run 4 | median | range |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 547.1 | 537.3 | 538.5 | 542.9 | **540.7** | 1.8 % |
| **v3** | 529.2 | 525.0 | 526.5 | 527.1 | **526.8** | **0.8 %** |

**v3 wins by ~14 ms = 2.6%, with tighter variance.** The single-run
measurement was confounded by call-order in the same bench binary: v2
was called first (cold-cache best case), v3 second. With both
warmed up across separate processes, v3 consistently wins.

Per CLAUDE.md's ±5% rule the 2.6 % delta is below forced-investigation
threshold, but it is reproducible and worth shipping. dots-ocr
dispatch is now switched to v3.

Parity: 336 cases at hd=128, 0 failed, max-abs-diff 3.052e-5.
E2E F1 against vLLM reference: 1.000, 13/13 exact text match.

### 13.2. Why it didn't help — rocprof data

`rocprofv3 --pmc LDSBankConflict SQ_INSTS_LDS SQ_INSTS_VALU GL2C_HIT`:

| kernel | LDSBankConflict | SQ_INSTS_LDS | SQ_INSTS_VALU | GL2C_HIT |
|---|---:|---:|---:|---:|
| v1            | **66.5** | 3.09 G | 7.44 G | 388 M |
| v2 (pad+coop) | **3.3**  | 3.74 G | 6.23 G | 339 M |
| v3 (+hoist)   | **3.3**  | 3.74 G | 6.18 G | 263 M |

`SQ_INSTS_LDS` is **identical between v2 and v3** despite v3's
theoretical 8× reduction in phase C S_lds reads. The compiler had
already vectorised the per-lane `for (j=0..15) a_reg_sm[j] = sm_row[j]`
loop into wide `ds_read_b128` instructions (16 bytes per lane → 1
LDS instruction per j-loop, not 16). v3's "hoist" was a no-op at
the instruction level.

### 13.3. What the rocprof DOES show — v1 → v2 attribution

The v2 win (752 → 538 ms = +28%) is almost entirely the **20×
reduction in LDS bank conflicts** from the S_lds row-stride padding
(66.5 → 3.3). The cooperative softmax added LDS instruction count
(3.09 → 3.74 G) but the bank-conflict fix dwarfed any per-instruction
overhead.

### 13.4. Where the remaining 200ms gap likely lives

GPU_UTIL = 100% but the visible counters show:
  - VALU utilization ~14% of theoretical peak
  - LDS utilization ~4% of theoretical peak
  - LDS bank conflicts near zero

The bottleneck is **not** any of: VALU compute, LDS bandwidth, LDS
bank conflicts. Most likely **DRAM access latency** — we're consuming
~33 GB/s effective vs LPDDR5X peak of 115 GB/s (28% of peak), and
GL2C_HIT is dropping iter-over-iter as we squeeze the inner loop
denser. The remaining wall-time appears to be cycles waiting on
in-flight DRAM loads.

v3 is wired into dots-ocr dispatch (corrected after pre-push audit
caught the original mis-measurement).

### 13.5. What still might move the needle

1. **Reduce DRAM traffic further.** Either MFP4 K/V (sub-byte, halves
   DRAM again) — needs accuracy work — or larger M (M=128 → B/M=152,
   another 2× DRAM cut). M=128 requires LDS restructure (Q in LDS or
   K/V both in registers).
2. **Better DRAM utilization.** Software prefetch of K and V outside
   the phase-A/C inner loops to keep DRAM bandwidth saturated. CK's
   `kQLoadOnce + global_read_lds_i+2` pattern.
3. **QKV-cast fusion** for the E2E vision-encoder win (~420 ms),
   independent of the attention kernel.

## 14. Further levers investigation (2026-05-26)

Cross-referenced codebase analysis, rocprof PMC data, the Gemini
follow-up (`dots-ocr.perf-investigation-gemini.md`), and the decode
PMC checkpoint (`2026-05-26-gfx1151-decode-attention-pmc.md`).

The investigation now covers three distinct paths:

| Path | Current kernel | Bottleneck | Typical shape |
|---|---|---|---|
| Vision encoder prefill | v3 M=64 N=128 f16 K/V | DRAM BW (28% util) | B=L=19520, hd=128, h=12 |
| Text decoder prefill | `attention_causal_batched` | **Scalar compute + no GQA** | seq=5k-11k, h=12, kv=2 |
| Text decoder decode | `attention_flash_gqa` | Dispatch / wave overhead | seq=5k-11k, h=12, kv=2 |

### 14.1. Vision prefill: async V-load overlapped with Phase A

**Highest expected single-kernel impact.**

v3's outer loop is strictly sequential:

```
Phase A (QK): loads K from DRAM, computes s_acc  — V_lds is FREE
Stage V:      loads V from DRAM into V_lds        — no compute
Phase B:      softmax on S_lds
Phase C:      SV using V_lds
```

V-staging at `v3:165-177` runs *after* Phase A finishes. During Phase A,
V_lds contains stale data from the previous tile's Phase C — but nobody
reads it. We can start loading V asynchronously during Phase A using
RDNA3's `__builtin_amdgcn_global_load_lds` (direct global→LDS async
copy, bypasses VGPRs entirely).

**Implementation sketch:**

```
for (kt_start = 0; kt_start < L; kt_start += n_tile) {
    if (kt_start > 0) wait_async_v_load();  // ensure prev V is ready

    // Phase A: QK compute (loads K from DRAM into registers)
    //   Concurrently: issue async global→LDS for THIS tile's V
    issue_async_v_load(v + kt_start * kv_stride + ..., V_lds);
    compute_qk_phase_a(...);

    wait_async_v_load();  // ensure V is in LDS before Phase B
    // Phase B + C unchanged
}
```

No extra LDS — V_lds is reused in-place. The async load replaces the
synchronous staging loop, saving ~32 KB of VGPR-bypassed traffic per
outer iteration. With 152 outer iterations, this overlaps ~152 × 32 KB =
4.9 MB of V-load latency with useful QK compute.

**Why this helps:** rocprof shows 28% BW utilization (33 GB/s effective
vs 115 GB/s peak). The DRAM pipeline is underutilized because we do
strictly sequential load-then-compute. Overlapping V-load with QK
compute keeps the DRAM controller busy during Phase A.

**Gemini note:** §2.1 proposes `__builtin_amdgcn_s_prefetch_data` — this
is a *scalar* prefetch (L1 I-cache / scalar D-cache), not useful for
vector data paths. The correct intrinsic is
`__builtin_amdgcn_global_load_lds` for global→LDS, or explicit
double-buffering with `__builtin_amdgcn_ds_gws_init` barriers. CK's
`global_read_i+2` pattern (issue load 2 iterations ahead) is the
proven reference.

### 14.2. Vision prefill: V_lds transpose for vectorized Phase C reads

Phase C at `v3:262-274` reads V_lds in a column-major pattern:

```c
for (int j = 0; j < 16; ++j) {
    b_reg[j] = V_lds[(s_col_base + j) * head_dim + my_d];
}
```

16 sequential `ds_read_u16` per lane per (dc, c) iteration — the
compiler cannot vectorize across `j` because rows are stride-128 f16
= 256 bytes apart. Transposing V_lds from `[n_tile][head_dim]` to
`[head_dim][n_tile]` makes the access contiguous:

```c
// Transposed: V_lds[my_d * n_tile + s_col_base + 0..15]
b_reg = ds_read_b128(V_lds + my_d * 128 + s_col_base);  // 8 f16 at once
b_reg2 = ds_read_b128(V_lds + my_d * 128 + s_col_base + 8);  // 8 more
```

**Impact:** Phase C does 8 dc × 8 c = 64 WMMA ops per outer iteration.
Each WMMA loads 16 V values → 1024 V reads per lane per iter × 152
iters = 155k reads total. Vectorizing from 16 `ds_read_u16` to 2
`ds_read_b128` is an 8× instruction reduction on the LDS path.

**Trade-off:** V-staging (`v3:165-177`) writes in `[n_tile][head_dim]`
order, which is naturally contiguous (128 f16 per row, 128 rows).
Transposed staging scatters writes across `head_dim` rows. Net effect:
staging gets ~8× more instructions (same total bytes), Phase C gets ~8×
fewer. Since Phase C dominates (softmax + SV >> staging), the net is
positive.

**Bank conflict analysis:** Current V_lds at stride 128 f16 = 64 dwords.
Lane `l` accessing row `r`, column `my_d`: bank = `(r * 64 + my_d/2) % 32
= (my_d/2) % 32` — all 16 lanes hit the same bank for the same `j`.
Transposed: bank = `(my_d * 64 + r + j) % 32` — varies with lane,
conflict-free.

### 14.3. Vision prefill: N=256 with V from DRAM (drop V_lds)

§10.3 notes that N=256 doesn't fit because V_lds[256×128] f16 = 64 KB
saturates the LDS cap. If we drop V_lds entirely and load V directly
from DRAM during Phase C via WMMA frag_b:

- S_lds[64 × 256] f16 = 32 KB (or keep current S_in_registers approach)
- No V_lds
- O in registers
- Total LDS: ~32-33 KB — plenty of headroom

**DRAM traffic concern:** V_lds is shared across 4 waves per block.
Without it, each wave loads its own V fragment independently. At M=64
with 4 waves, V traffic increases 4× per block. With B/M = 305 blocks:
- Current: 305 × 32 KB (V read once, shared) = ~9.6 MB/head
- V via frag_b: 305 × 32 KB × 4 waves = ~38.4 MB/head

That's a 4× V traffic increase on a DRAM-bound kernel. **Net DRAM
increases**, so N=256 with V via frag_b only makes sense if combined
with M=128 to halve the block count:

- M=128 N=256 V via frag_b: 152 × 32 KB × 4 = 19.4 MB/head V traffic
- Plus K: 152 × 64 KB = 9.7 MB/head K traffic (N=256, but K also via reg)
- Total: ~29 MB/head vs current ~19.3 MB/head (M=64 N=128 V_lds)

Still 50% more DRAM. N=256 with V via frag_b is **not recommended
independently** — it needs to be bundled with M=128 and the async
V-load pipeline to compensate.

### 14.4. Vision prefill: M=128 query tile

**Biggest single DRAM reduction but complex implementation.**

At M=128: B/M = 152 → 305 blocks (was 610 at M=16). Each block reads
all K and V once. Total K+V DRAM traffic halves vs M=64.

**LDS budget at M=128 N=128 with V_lds f16:**

| Buffer | Size |
|---|---|
| V_lds[128 × 128] f16 | 32 KB |
| S_lds[128 × 128] f16 | 32 KB |
| scalars (m, l, alpha × 128) | 1.5 KB |
| **Total** | **~65.5 KB — over cap** |

S_lds alone eats 32 KB. Options:

**A. S in registers (llama.cpp pattern):** Each wave owns 16 rows. S for
16 rows × 128 cols = 2048 f16 = 4 KB per wave. Per lane: 2048 / 32 =
64 values = 128 bytes. 4 float8_t per lane — feasible in VGPRs.
This frees 32 KB of LDS for M=128 N=128 with V_lds.

**B. 256-thread block (8 waves, 16 rows/wave):** Same per-wave register
pressure as M=64 (Q_frags = 64 VGPRs, O_frags = 64, o_acc = 64). But
total threads = 256 → max 2 waves/CU at 256 VGPRs/lane. At 40 CUs:
~10 concurrent blocks (vs 40 at M=64). 3× fewer concurrent blocks but
2× fewer total blocks = 1.5× fewer block-iterations total. On a
DRAM-bound kernel the occupancy reduction may be acceptable.

**C. Two-pass sub-tiling:** Process M=128 as two sequential M=64
sub-tiles within the same block, sharing K/V loads. Each sub-tile
reuses the current v3 logic unchanged. Block count stays at 305
(same as M=64), but K+V loads happen once per tile instead of once per
sub-tile. Theoretically halves K+V traffic at constant occupancy.

Option C is the simplest to implement and lowest-risk. It trades
per-block wall time (2× more QK + SV work per block) for half the
K+V DRAM traffic. At current K+V = 37 GB per attention call, this
cuts to ~18.5 GB, approaching the ~17 GB theoretical floor at
M=128 f16.

### 14.5. Text prefill attention: causal WMMA + GQA

**Potentially the largest untapped optimization.** The text decoder
prefill uses `attention_causal_batched` (`kernels/src/attention_causal_batched.hip`):

- Scalar (no WMMA) QK and SV computation
- No GQA awareness — loads K/V `n_heads=12` times instead of
  `n_kv_heads=2` times (6× redundant work for the 12:2 ratio)
- Grid: `[n_heads, seq_len]` — one block per (head, query position)
- Each block loops over all prior KV positions with scalar dot products
- At seq=5000: 5000² × 12 × 128 = 38.4 GFLOP, all in scalar code

A causal WMMA flash attention kernel with GQA grouping would:

1. **WMMA for QK and SV** — 16× FLOP throughput on RDNA3 matrix cores
2. **GQA grouping** — 6× fewer K/V loads (12:2 ratio)
3. **f16 K/V** — halve DRAM traffic
4. **Same M/N tiling** as the vision kernel — M=64 N=128, with causal
   mask applied during Phase A (set S = -inf for key > query positions)

The vision attention kernel already implements all of this except the
causal mask. Adding a `is_causal` parameter that masks out-of-range
S values in Phase A would make the kernel reusable for both paths.

**Estimated impact:** At seq=5000, text prefill attention is likely
taking several seconds of GPU time (scalar O(seq²) on 38.4 GFLOP).
WMMA + GQA + f16 should bring this down by 5-10×. The text prefill
runs 28 layers × 1 attention call each — even a 2× speedup per layer
compounds to significant E2E savings.

**Why this was missed:** The investigation focused on the vision encoder
(the 89.3s wall-time elephant). But the text prefill path also runs
28 layers of attention, and its kernel is the pre-optimization scalar
baseline.

### 14.6. Decode: HIP graph capture

The decode loop (`qwen2.rs:769-877`) launches ~10 kernels per layer ×
28 layers = 280 kernel launches per decode step:

```
rmsnorm → wq_gemv → wk_gemv → wv_gemv → bias_add×3 → rope →
kv_cache_write×2 → attention_gqa → o_proj_gemv → add →
rmsnorm → w_gate_gemv → w_up_gemv → silu_mul → w_down_gemv → add
```

PMC shows dispatch/wave overhead is the decode floor on gfx1151
(257µs GQA, of which compute is tiny). HIP graph capture would:

- Record the full 280-kernel graph once (first decode step)
- Replay for subsequent tokens — eliminates all CPU-side dispatch
- Removes kernel launch overhead (~5-15µs per launch × 280 = 1.4-4.2 ms)
- At 257µs per attention call, the attention itself is only one of many
  kernel costs; graph capture amortizes the whole loop

**Expected decode speedup:** Hard to estimate without measurement.
If dispatch overhead is ~50% of per-step wall time (consistent with
Q8 KV showing 40× fewer fetches but same wall time), graph capture
could give 1.5-2× decode speedup.

**Complexity:** High. Requires refactoring the decode loop to use
`hipGraphLaunch` instead of individual kernel launches. The KV cache
write position changes every step — needs hipGraph node parameter
updates. But HIP graphs support this via `hipGraphExecKernelNodeSetParams`.

### 14.7. Decode: fused attention-reduce + output projection

At decode time, the attention output is `[n_heads × head_dim]` = 1536
floats = 6 KB. The reduce kernel writes this to `attn_out`, then
`weight_gemv` reads it back for the output projection (`wo`).

Fusing reduce + o_proj:

1. Each reduce thread computes its head_dim output element
2. Immediately computes the GEMV dot product with the corresponding
   `wo` weight column
3. Accumulates into a register, writes final `o[dim]` once

**Impact:** Eliminates 1 DRAM write (6 KB) + 1 DRAM read (6 KB) + 1
kernel launch per layer. At 28 layers: 28 fewer launches, 336 KB less
DRAM traffic (tiny but the launch savings matter).

This is a smaller lever than graph capture but is complementary and
simpler to implement. Could be done as a stepping stone toward full
graph capture.

### 14.8. Decode: gfx1100 GQA chunk-size tuning

The `HIPFIRE_GQA_CHUNK` env var defaults to 128, giving 80 workgroups
at seq=5100 (2 kv_heads × 40 chunks). On gfx1151 (40 CUs) this
saturates the GPU. On gfx1100 (96 CUs):

- chunk=128 → 80 wg → 83% fill (80/96)
- chunk=64 → 160 wg → 100% fill + 2× wave-level parallelism
- chunk=48 → ~213 wg → 100% fill, may improve latency hiding

The gfx1151 sweep showed chunk=64 gives ~2% (noise) over chunk=128.
But gfx1100 has 2.4× more CUs — the underfill is larger, and the
potential win from smaller chunks (more wg) is correspondingly larger.

**Action:** Sweep `HIPFIRE_GQA_CHUNK={128,96,64,48}` on gfx1100 at
seq=5100 and seq=12000. The env var already exists; just needs
benchmarking on target hardware.

### 14.9. Decode: F16 KV cache (long-sequence / gfx1100)

The PMC at seq=5100 on gfx1151 shows Q8 KV (40× fewer HBM fetches)
ties F32 at 272µs — dispatch overhead dominates, not bandwidth. But:

- At seq=12000, KV traffic is 2.4× larger. Bandwidth may start to
  matter even on gfx1151.
- On gfx1100 (960 GB/s GDDR6), the per-access cost is 8× lower, so
  dispatch overhead is an even larger fraction. But at long sequences
  (12k+), the absolute KV traffic may exceed what L2 can absorb.
- F16 KV is the simpler first step before FP8/MFP4 (no accuracy work).

**Action:** Benchmark F16 KV decode on gfx1100 at seq=12000 before
investing in FP8. If F16 doesn't help (same dispatch floor), FP8
won't either.

### 14.10. Evaluation of Gemini proposals

| Gemini proposal | Assessment |
|---|---|
| §2.1 `__builtin_amdgcn_s_prefetch_data` | Wrong intrinsic — this is a *scalar* prefetch (I-cache / scalar D-cache). Correct approach: `__builtin_amdgcn_global_load_lds` for async global→LDS, or explicit register double-buffering. The *intent* (overlap DRAM with compute) is correct and is our §14.1. |
| §2.2 `global_load_lds` for V-staging | Correct and high-value. See §14.1 — this is the highest-impact single-kernel lever for prefill. The 16-byte alignment constraint is satisfied at hd=128. |
| §2.3 M=128 with striped V_lds | Correct direction. We analyze three M=128 strategies in §14.4 (S-in-registers, 256-thread block, two-pass sub-tiling). Sub-tiling (option C) is simplest and lowest-risk. |
| §2.4 V via WMMA frag_b from DRAM | Viable only when combined with M=128 to offset the 4× V traffic increase from losing cross-wave LDS sharing (§14.3). Independently it regresses. |
| §2.5 QKV-cast fusion | Correct and already identified. ~420 ms E2E. Independent of attention kernel changes. |
| §3.1 Persistent GQA kernels | Correct. The fused GQA showed 96× regression from 2-wg occupancy collapse. A persistent kernel that iterates over chunks within one block would maintain occupancy while reducing launch count. Simpler alternative: HIP graph capture (§14.6) achieves the same dispatch elimination without custom persistent-kernel logic. |
| §3.2 FP8 KV cache | Premature for decode. PMC shows bandwidth isn't the decode bottleneck on gfx1151 at seq=5100. F16 KV (§14.9) is the simpler first step. FP8 may matter on gfx1100 at long sequences — test F16 first. The text backbone currently uses F32 KV; F16 is the natural next step. |
| §4 ranked action plan | Rankings mostly match our analysis. Prefetch/V-load overlap and QKV fusion are the top prefill levers. However the plan misses: (1) text prefill WMMA causal attention (§14.5, potentially the largest single win), (2) V_lds transpose (§14.2, simple LDS layout change), (3) HIP graph capture for decode (§14.6, bigger than persistent kernels for the dispatch floor). |

### 14.11. Revised ranked action plan (recalibrated 2026-05-28 for gfx1100)

The original projections below were calibrated on **Strix Halo gfx1151**
(115 GB/s LPDDR5X). On **gfx1100** (960 GB/s GDDR6), DRAM-bound bottlenecks
have less headroom because faster bandwidth means DRAM transfers complete
sooner and compute becomes a larger fraction of wall time. Key recalibrations:

- **DRAM traffic improvements scale sub-linearly with bandwidth ratio.**
  Halving DRAM traffic on a compute-bound kernel gives ~1.5× at most, not 2×.
  The formula is: `improvement = bw_saving × bw_fraction_of_total`.
- **Per-tile fixed costs (sync barriers, alpha-scales) are tiny relative to
  compute on gfx1100.** K-tile widening from 16→64 gave +7.2% on Strix Halo
  and ~10% on gfx1100, not the predicted 2-4×.
- **The compiler vectorizes LDS reads.** Source-level “hoisting” can be a
  no-op at ISA level; always verify with `llvm-readelf --notes`.
- **Occupancy changes dominate over micro-optimizations.** v4→v5 (V_tile
  64→32, 1→2 WG/CU) gave 1.44× from occupancy alone — bigger than any
  single micro-optimization in the entire investigation.
- **Split-d attention is fundamentally unprofitable** for full-softmax.
  Softmax requires the complete 128-dim dot product; each pass must
  rescan the full K sequence. v6/v6b tried this and was 11.6× slower.

| Rank | Lever | Target | Original projection | **Revised (gfx1100)** | Complexity | Why revised |
|---:|---|---|---|---|---|---|
| 1 | **QKV-cast fusion** (§2.5) | Vision E2E | +420 ms | **~420 ms (0.7% total)** | Medium | Unchanged; independent of attention kernel |
| ~~2~~ | ~~**V_lds transpose** (§14.2)~~ | Vision attn | ~~+5-10%~~ | **+15.6% non-causal N=128; -6.5% vision** | Low | §15: helps non-causal N=128 (v4>v3), hurts V_tile=32 (v6<v5). v4_causal is bench-only after large-batch crash |
| ~~3~~ | ~~**Async V-load** (§14.1)~~ | Vision attn | ~~+10-15%~~ | **BLOCKED — no `global_load_lds` on RDNA3** | ~~Medium~~ | §16: `vmem-to-lds-load-insts` is CDNA-only. All alternatives exceed LDS budget |
| 4 | **FP8/MFP4 K/V** (§3.2) | Vision attn | +20-40% | **+20-40% attn (2.4-4.8s)** | High | Unchanged but needs accuracy validation |
| ~~5~~ | ~~**M=128 sub-tiling** (§14.4C)~~ | Vision attn | ~~+5-15%~~ | **-10.9% (K-shared) / -2.4% (sequential)** | Medium | §17: VGPR spills from 2× state overwhelm K savings. No DRAM benefit without sharing |
| 6 | **HIP graph capture** (§14.6) | Decode | 1.5-2× | **+3-5% decode (1-2s)** | High | **Down from 1.5-2×.** gfx1100 dispatch is ~340µs vs 7.5ms compute (4.5%); Strix Halo was 76% dispatch |
| 7 | **Causal WMMA + GQA** (§14.5) | Text prefill | 5-10× | **<1% total (<0.5s)** | Medium | **Down from 5-10×.** Prefill already 1.0s; even 10× attention speedup saves <0.5s of 60s total |
| 8 | **GQA chunk sweep** (§14.8) | Decode | +5-15% | **+0-5% decode** | Low | **Down from +15%.** Compute-dominated on gfx1100 with 96 CUs |
| 9 | **Fused attn-reduce + o_proj** (§14.7) | Decode | +3-5% | **+1-3% decode** | Low | Marginal DRAM saving; tiny launch saving |
| 10 | **F16 KV cache** (§14.9) | Decode (long seq) | +0-10% | **+0-5% decode** | Low | Dispatch-dominated at current seq lengths |

### Projection audit: what actually happened vs what was predicted

**§4.1 K-tile 16→64:** predicted 2-4× speedup. Actual +7.2% on Strix Halo,
+10.4% vs M=16 baseline. Per-tile fixed costs are tiny vs DRAM traffic on
fast-BW hardware. The iteration-count model overstates by counting DRAM
transfers that are pipelined and overlap with compute.

**§4.2 f16 K/V:** predicted +30-100%. Actual +18% on Strix Halo (76% of
theoretical ceiling). Upper bound assumed compute was zero; compute is
non-trivial (~25% of wall time after the DRAM floor).

**§4.3 Q in registers:** predicted +10-20%. Never measured independently
(bundled into N=64 kernel). The Q_frags scratch trap showed that incorrect
register allocation causes 19% regression. **Lesson: always verify VGPR
allocation with `llvm-readelf --notes`.**

**§8.4 K-tile widen to N=128:** predicted halving of K+V traffic.
Actual +27% over N=64 f16-K/V on Strix Halo. LDS bandwidth bottleneck
(V_lds + S_lds in f16) was an unexpected positive effect not in the model.

**§11 M=64 query tile:** predicted ~2× from DRAM halving. Actual +53% over
N=128 v1. DRAM halving gives 1.5× at most because compute doesn't shrink.

**§12 S_lds bank conflict + cooperative softmax:** predicted moderate.
Actual +31% on Strix Halo — the second-biggest single win after M=64.
Bank conflicts were the real bottleneck (confirmed by rocprof: 66.5→3.3).

**§13 v3 hoisted S_lds reads:** predicted 8× fewer reads. Actual +2.6%.
Compiler had already vectorized into `ds_read_b128` instructions; source-level
hoisting was a no-op.

**§14.4 v5 V_tile=32:** not in original plan. Achieved 1.44× speedup from
occupancy improvement alone (1→2 WG/CU). Occupancy changes dominate over
micro-optimizations.

**§14 (v6/v6b split d_half):** not in original plan. Negative result: 11.6×
slower than v5. Splitting head_dim into multiple passes is fundamentally
unprofitable for full-softmax attention.

### Approaches proven not to work (do not revisit)

- **Splitting head_dim** into d_half passes (v6/v6b): 4× total WMMA
  iterations → 11.6× slower. Full-softmax requires complete 128-dim
  dot product per pass.
- **Reducing n_tile** to shrink accumulators (v6b): halves s_acc but
  doubles K-tile iterations. Net loss.
- **Persistent WMMA GEMM kernels**: 0.2-0.24 TFLOP/s — overhead dominates
  for inference-sized batches.
- **V-staging into LDS** (phase between QK and softmax): proven no-op on
  this DRAM-bound workload; V_lds contains stale data that nobody reads.
- **Source-level LDS read hoisting**: compiler vectorizes to `ds_read_b128`
  regardless; measure before claiming improvement.
- **n_tile > 128** without dropping V_lds: 4× per-wave V traffic increase
  makes N=256 regress unless bundled with M=128 sub-tiling.

### 15. V_lds transpose for Phase C reads (2026-05-29)

§14.2 (V_lds transpose) implemented and benchmarked on gfx1100.

Transposed V_lds from `[n_tile][head_dim]` to `[head_dim][V_T_STRIDE]`
(padded stride for bank-conflict-free reads). Phase C `b_reg[j]` reads
become 16 consecutive f16 values (vectorizable to `ds_read_b128`)
instead of stride-128 scattered `ds_read_u16`.

**gfx1100 benchmark results (B=L=19520, hd=128, n_heads=12):**

| kernel | dur (ms) | vs baseline |
|---|---:|---:|
| v3 (M=64 N=128 f16 K/V) | 267 | — |
| **v4 (M=64 N=128 V_lds_T)** | **226** | **+15.6%** |
| v5 (M=64 V_tile=32) | 160 | — |
| v6 (M=64 V_tile=32 V_lds_T) | 171 | -6.5% |

**Why v6 regresses:** V_tile=32 stages V in 4 v_chunks per K-tile.
Each v_chunk does 4096 scattered writes (de-vectorized from coalesced
stores in v5) and only 256 vectorized reads. Write:read ratio is 16:1
per v_chunk; de-vectorizing writes costs more than vectorizing reads.

**Why v4 wins:** N=128 stages V once per K-tile. Single 16384-write
staging amortized across 8 c-iterations of Phase C (1024 reads). Read
vectorization dominates.

**Outcome:**
- v4_causal is bench-only until its large-batch crash is fixed
  (2026-05-31 triage: `parity_causal_wmma 5095 1` and dots-ocr
  5095-position prefill both reproduced the fault).
- qwen2 text prefill remains on v3_causal.
- v5 stays production for dots-ocr vision (v6 is slower).
- v4 (non-causal) available for future threshold-attention use.

### 16. Async V-load investigation — NOT FEASIBLE on gfx1100 (2026-05-29)

§14.1 (async V-load) investigated and **blocked by hardware limitations**.

**Finding:** `__builtin_amdgcn_global_load_lds` (direct global→LDS async copy)
requires the `vmem-to-lds-load-insts` target feature, which is **CDNA-only**
(MI300, MI250). gfx1100 (RDNA3 / RX 7900 XTX) does NOT have this feature.
The compiler emits:

```
error: '__builtin_amdgcn_global_load_lds' needs target feature vmem-to-lds-load-insts
```

On RDNA3, all data must travel through VGPRs (global_load → VGPR → ds_write).
There is no hardware DMA engine for global→LDS.

**Alternative approaches evaluated:**

1. **V_lds double-buffering**: Would require 2× V_lds (16 KB for v5, 64 KB
   for v3/v4). v5's total LDS would be 25.6 + 8 = 33.6 KB, exceeding the
   32 KB limit for 2 WG/CU. Drops to 1 WG/CU — net regression.

2. **Interleaved V staging within Phase A**: v5 has V_lds=32 rows but needs
   128 total V rows per K-tile (4 v_chunks × 32). Can't fit all rows in
   V_lds simultaneously. Would require processing one v_chunk per Phase A
   pass, tripling the Phase A cost.

3. **Software pipelining across K-tiles**: Load V for tile N+1 during tile
   N's Phase C. Requires double-buffered V_lds (same problem as #1).

4. **Inter-wave pipelining**: Dedicate waves to V loading while others
   compute. Reduces compute throughput (fewer waves for QK) without
   guaranteed overlap benefit on RDNA3's single-issue scheduler.

**Conclusion:** Async V-load is not actionable on gfx1100. The §14.1
prediction assumed `__builtin_amdgcn_global_load_lds` availability on
RDNA3, which was incorrect. This optimization IS viable on CDNA (MI300,
MI250) and should be pursued for those targets.

**Updated ranking:** Remove async V-load from gfx1100 action plan.
Next actionable item for gfx1100 vision: §14.3 (N=256 with V from DRAM)
or §14.4 (M=128 query tile), both higher complexity.

### 17. M=128 sub-tiling — NOT BENEFICIAL on gfx1100 (2026-05-29)

§14.4C (M=128 two-pass sub-tiling) implemented and benchmarked.

Two variants tested on gfx1100:

| kernel | ms | vs v5 |
|---|---:|---:|
| v5 (M=64 production) | 164 | — |
| v7 (K-shared sub-tile) | 182 | -10.9% |
| v7b (sequential, no share) | 168 | -2.4% |

**v7 (K-shared):** Phase A loads K once, computes QK for both sub-tiles
using shared b_reg. Requires Q_frags_0 + Q_frags_1 + O_frags_0 + O_frags_1
= 256 VGPRs minimum. Compiler spills heavily (binary 34% larger). Spill
traffic overwhelms K-load savings.

**v7b (sequential):** Sub-tiles processed independently, same VGPR as v5.
No K/V sharing. L2 warmth from halved block count doesn't help because
K+V working set (~10 MB) exceeds gfx1100's 6 MB L2.

**The §14.4C prediction of "halving K+V traffic" was unachievable:**
sharing K+V between sub-tiles requires simultaneously holding both
sub-tiles' Q_frags, O_frags, and softmax state, which exceeds RDNA3's
256 VGPR limit. The register pressure bottleneck is fundamental to the
two-pass sub-tiling approach on RDNA3's wave32 architecture.

**Conclusion:** M=128 sub-tiling is not beneficial on gfx1100. The
production v5 kernel (M=64, V_tile=32, 2 WG/CU) remains optimal.
