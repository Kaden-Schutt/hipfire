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
