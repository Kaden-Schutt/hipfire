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
