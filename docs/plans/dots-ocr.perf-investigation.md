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

## 13. v3 — hoist S_lds reads (null result, archived 2026-05-23)

`kernels/src/attention_dflash_wmma_m64_n128_f16kv_v3.hip` reorders
phase C to outer c, inner dc so `a_reg_sm` is read once per K-chunk
(was once per (d-chunk, K-chunk)). Theoretical 8× reduction in phase C
S_lds reads. Also moves the alpha-fold to start-of-phase-C and
accumulates SV into a per-d-chunk `o_acc_local[8]` register array.

### 13.1. Result: tied with v2 (within noise)

bench (B=L=19520, hd=128):
  v2: 518.5 ms
  v3: 522.6 ms

Parity: 336 cases, 0 failed.

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

v3 is committed but **not wired into dots-ocr dispatch** — kept as a
documented ablation point. dots-ocr stays on v2.

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
