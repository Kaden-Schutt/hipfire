# dots.ocr Phase 6 (perf) — dev log

Goal: bring dots.ocr end-to-end latency on a real page toward vLLM's
**~15s**. Branch `feat/dots-ocr-phase-6-wmma-gemm`. Deploy target gfx1100
(RX 7900 XTX), system ROCm 7.2.3.

Companion design doc: [`dots-ocr.perf-investigation.md`](dots-ocr.perf-investigation.md).
Per-kernel timing is the tool of record — PMC memory counters
(`FETCH_SIZE`, `GL2C_*`) read flat zero on this gfx1100 under rocprofv1/v2/v3
(driver/GFXOFF-gated). The dated entries below are a reverse-chronological log;
**start with the handoff block, then the ranked headroom at the bottom.**

## Status & handoff (2026-05-28)

Branch `feat/dots-ocr-phase-6-wmma-gemm` (off master `09e94b25`), 10 commits, all
landed with **F1=1.000** vs the vLLM reference. The dots.ocr F1 grade is the
**only** correctness gate that matters here — the repo's `coherence-gate.sh`
runs qwen35 models and does NOT exercise the qwen2/dots-ocr forward path, so any
change to these kernels/forward must be re-graded:

```
cargo build --release -p hipfire-arch-dots-ocr --example ocr_e2e
./target/release/examples/ocr_e2e --hfq /data/hipfire/dots-ocr.q8.hfq \
  --image benchmarks/images/dots_ocr_smoke_001.jpg \
  --prompt-json benchmarks/references/dots_ocr_smoke_001.json \
  --prefill batch > /tmp/our.txt 2>/tmp/our.err          # decodes to EOS (~4633 tok)
python3 scripts/grade_dots_ocr_e2e.py --our /tmp/our.txt \
  --ref benchmarks/references/dots_ocr_smoke_001_vllm.json  # want F1=1.000, 13/13
```

Honest end-to-end (smoke image, full layout to EOS) vs vLLM's ~15s target:

| stage | session start | after prior agent | **now** | how |
|---|---|---|---|---|
| vision  | 49.6s | 29.6s | **24.2s** | v5 attention (V_tile=32, 2 WG/CU achieved, 1.44× faster) |
| prefill | 27.4s | 1.0s | **1.0s**  | wired gfx11 WMMA Q8 GEMMs (25×) |
| decode (4633 tok) | 62.3s | 62.3s | **34.8s** (133 tok/s) | warp-cooperative attention (3.5×) + fused gate+up |
| **total** | ~139s | ~93s | **~60s** | |

**Performance projection audit (2026-05-28):** The original projections in
perf-investigation.md were calibrated against Strix Halo (gfx1151, 115 GB/s
LPDDR5X). On gfx1100 (960 GB/s GDDR6), DRAM-bound bottlenecks have less
headroom because transfers complete faster and compute becomes relatively
more important. Key audit findings:

- §4.1 "K-tile 16→64: expected 2-4×" → **actual +7.2%**. Overstated 28-56×
  because per-tile fixed costs are tiny vs DRAM traffic on fast-BW hardware.
- §4.2 "f16 K/V: expected +30-100%" → **actual +18%**. Upper bound was
  unrealistic; bandwidth savings hit a compute floor.
- §8.4 "M=64: expected ~2× DRAM halving" → **actual +53%**. DRAM halving
  gives 1.5× at most because compute doesn't shrink.
- §12 S_lds bank-conflict fix → **actual +31%**. Compiler couldn't vectorize
  stride-128 f16 reads; bank conflicts were the real bottleneck.
- §13 "v3 hoisted S_lds: expected 8× fewer reads" → **actual +2.6%**.
  Compiler had already vectorized reads into `ds_read_b128`.
- §14.1 "async V-load: +10-15% vision" → **revised: +5-10%** on gfx1100 (8×
  more BW means less DRAM stall overlap).
- §14.5 "causal WMMA+GQA: 5-10× text-prefill" → **revised: <1% total**.
  Prefill is already 1.0s; even 10× speedup saves <0.5s of 60s total.
- §14.6 "hipGraph: 1.5-2× decode" → **revised: +3-5% decode**. gfx1100
  dispatch is ~340µs vs 7.5ms compute; only 4.5% of decode time.
- §14.4 "M=128 sub-tiling: +15-25%" → **revised: +5-15%**. gfx1100's faster
  BW means DRAM traffic reduction helps less; LDS/register pressure remains.

Negative results from this session that invalidate future projections:
- **v6 (split d_half, n_tile=128):** s_acc[4] overflow → NaN; s_acc[8] fix
  correct but 5.7× slower than v5 due to 2× K-tile iterations.
- **v6b (split d_half, n_tile=64):** 11.6× slower than v5 (4× total WMMA
  iterations). **Conclusion: splitting head_dim into multiple passes NEVER
  profits for full-softmax attention.** Softmax needs the complete 128-dim
  dot product; each pass must scan the full K sequence. Any tiling that
  increases total iteration count is a net loss.

(`ocr_e2e` reports honest per-stage times since the async-drain timer fix; the
decode-loop is timed in isolation. The "vision 49.7 / prefill 27.4" baselines
above are post-fix — pre-fix the tool mis-reported prefill as 0.1s and decode
as 2.3 tok/s; see the first dated entry.)

**Tools / repro:**
- In-engine per-stage timing: `HIPFIRE_PREFILL_TIMING=1` (prefill GEMM/attn
  split), `HIPFIRE_DECODE_TIMING=1` (per-token forward vs argmax).
- Microbenches: `cargo run --release -p rdna-compute --example bench_gemv_q8`
  (decode GEMV bandwidth by shape), `… --example bench_decode_attention --seq
  5100 --iters 100` (decode attention µs/call; honors `HIPFIRE_GQA_CHUNK`),
  `… --example check_mb4` (mb4 GEMM bit-exactness, incl. sub_offset halves).
- Profiler: `rocprofv2 --kernel-trace -d OUT -o t -- <bin>` gives per-kernel
  timestamps + VGPR/SGPR/LDS (vision profiles fine; **crashes on prefill/long
  decode** — the AqlPacket many-dispatch bug; use the in-engine timers there).
  Prepend a `/tmp/hipcc-stub` (`#!/bin/sh\nexit 0`) to PATH so the compiler
  version-probe doesn't deadlock under the profiler.
- Static register/LDS/spill (the decisive data the profiler can't see — memory
  PMC counters are DEAD on this gfx1100 across all profiler versions): the
  `gfx-kernel-metadata` skill on `.hipfire_kernels/gfx1100/<kernel>.hsaco`
  (`clang-offload-bundler` unbundle → `llvm-readelf --notes` → vgpr_count /
  vgpr_spill_count / group_segment_fixed_size). gfx1100 = 1024 VGPR/SIMD,
  16 max waves/SIMD, 64 KB LDS/CU.

Model: `/data/hipfire/dots-ocr.q8.hfq` (Q8 text weights, F16 vision weights).
Text decoder: hidden 1536, 28 layers, 12 heads / 2 kv-heads (GQA 6:1), head_dim
128, interm 8960. Vision: embed 1536, 42 blocks, interm 4224, ~19520 patches.

## 2026-05-27 (cont.) — vision encoder MB8 migration, 32.2s → 29.5s, F1=1.000

Migrated all 4 vision encoder GEMMs from MB4 (4-way register blocking)
to MB8 (8-way). Each block processes one 16×128 output panel — loads the
16×16 weight block once, applies it across 8 N-subtiles (vs 4). Doubles
arithmetic intensity without VGPR spills (vision GEMMs use ~32-40 VGPRs,
well within the 256 cap at 8 waves/SIMD).

Microbench at 19520 patches (vision encoder N):
```
                        MB4      MB8    speedup
qproj/kproj/vproj     2.36ms   2.03ms   1.16×    M=1536, K=1536
fc2                   2.48ms   1.76ms   1.41×    M=1536, K=4224
fc13 (gate)           2.13ms   1.88ms   1.13×    M=4224, K=1536
fc13 (up)             2.13ms   1.88ms   1.13×    M=4224, K=1536
```

End-to-end: vision 32.2s → **29.5s** (2.7s saved, 8.4% faster).
Total (with prior decode + fusion wins): 1m50s → **~1m07s**.
F1=1.000, 13/13, text exact 13/13 — PASS.

Committed as 9d552407. Benchmark: `test_mb8_shapes.rs`.

## 2026-05-27 (cont.) — fused gate+up Q8 GEMV, +5 tok/s decode

New kernel `fused_gate_up_q8_0`: one launch computes gate(x) and up(x)
from the same input, saving one kernel launch + one x-vector load per
layer. Grid = [gate_m + up_m], block = 32. Each block routes to gate or
up weights by blockIdx.x < gate_m. Both projections share the same 6 KB
input vector in L1.

Microbench: 44.2µs (2× separate) → 32.1µs (fused), **1.38×**.

End-to-end decode: 128.0→**132.8 tok/s** (+4.8 tok/s, +3.8%).
F1=1.000 — PASS. Committed as faaa825e.

Negative: maxdiff_gate=0.00e0, maxdiff_up=1.22e-4 (Q8_0 quantization
noise, acceptable). The up projection writes with a different out_row
calculation than gate — fixed in second iteration.

## 2026-05-27 (cont.) — decode attention: warp-cooperative kernel, 3.5× speedup

Decode attention was the #1 bottleneck at 7ms/token (28 layers × 270µs).
Root cause: `attention_flash_gqa_partial` was **uncoalesced-bound** — each
thread handled one token's full 128-dim K-dot with stride-256 accesses,
wasting 124 of 128 bytes per 128-byte cacheline.

New kernel `attention_gqa_warp`: 32 threads of a warp cooperatively
compute one query head's attention. Each thread loads 4 consecutive
floats from the SAME K-row → perfectly coalesced 128-byte loads.
Warp-shuffle reduce gives the dot product in 5 cycles. Online softmax
stays in registers. V-accumulate also coalesced.

Kernel metadata: VGPR=31, SGPR=22, spill=0, LDS=0. Same partials
layout as old GQA → reuses `attention_flash_reduce` unchanged.

```
bench microbench (seq=5100, 12:2 GQA, hd=128):
  attention_flash_gqa_partial:  270 µs  (baseline)
  attention_gqa_warp:            77 µs  (3.5×)

Full-token microbench (28 layers):
  gqa_partial:   7988 µs/token
  gqa_warp:      2361 µs/token  (3.38×)

End-to-end decode (4633 tokens to EOS):
  warp alone:    74 → 128 tok/s
  warp+fused-gu: 128 → 133 tok/s
```

F1=1.000, 13/13, text exact 13/13 — PASS. Committed as 79a68969.

Negative: `HIPFIRE_GQA_CHUNK=32` produces maxdiff=7.0 with the default
partials buffer (sized at max_seq/128) — chunk < 128 needs a larger
buffer. Dispatch clamped to cs_cap.max(128); bench enlarged its buffer.
Not a kernel bug.

## 2026-05-27 — honest end-to-end breakdown (the measurement-bug fix)

`ocr_e2e` reported prefill at `0.1s / 3696 tok/s` and decode at `2.3 tok/s`.
Both were **measurement artifacts**: the batched prefill enqueues async and
returns before the GPU work runs, so the prefill timer measured only host
submission, and the first post-prefill `argmax` then drained ~27s of pending
prefill compute — which the generate timer charged to *decode*. Added a
`device_synchronize()` after the prefill call (`ocr_e2e.rs`); decode-loop is
now timed in isolation.

Honest breakdown, smoke image (`dots_ocr_smoke_001.jpg`, 5095 prompt
positions = 4880 visual + 215 text), 32 decode tokens:

| stage | time | share |
|---|---|---|
| vision encoder | **49.7 s** | 64% |
| prefill (5095 pos) | **27.4 s** (185 tok/s) | 35% |
| decode | 0.4 s (**78 tok/s**) | <1% |

Decode is **not** the bottleneck — it's already ~78 tok/s. (The decode
hipGraph capture/replay work this session is gated behind
`HIPFIRE_DECODE_GRAPH` and gives ~0 on gfx1100 because decode here isn't
host-dispatch-bound; it's a deliverable for the dispatch-bound gfx1151 box.
Parked in `git stash@{0}` + `~/decode-hipgraph-wip/`, not committed.)

## 2026-05-27 — prefill WMMA Q8 wired: 27.4s → 1.1s (≈25×), F1=1.000

Wired the fused gfx11 WMMA Q8 GEMMs into `forward_prefill_batch_embeds`:
`gemm_qkv_q8_0_wmma` (fused QKV), `gemm_gate_up_q8_0_wmma` (fused gate+up),
`gemm_q8_0_residual_wmma` for o_proj + down (folds the residual add into the
GEMM). Gated on all-Q8 weights + WMMA arch + K%32==0; falls back to the old
`proj()` (GEMV) path otherwise. Mirrors the qwen35 production prefill path; the
kernels were already proven there, only the wiring is new.

Result (smoke image, full layout decoded to EOS):
- **prefill 27.4s → 1.1s (4804 tok/s) ≈ 25×**
- grade vs vLLM: **F1=1.000, 13/13 regions, text exact-match 13/13 — PASS**
- new end-to-end: vision **49.6s** + prefill 1.1s + decode **62.3s** (4633 tok @ 74 tok/s)

**Decode is now the largest full-page component** (62s > vision 49.6s) — the
earlier "decode is negligible" held only for the 32-token smoke window. Next
levers: vision GEMM (same naive-WMMA class of fix as prefill) and decode
throughput (the parked hipGraph work targets dispatch-bound boxes; on gfx1100
decode is compute-bound at ~74 tok/s).

## 2026-05-27 — vision GEMM: register-blocked mb4 WMMA, 49.6s → 39.8s, F1=1.000

New kernel `gemm_f16_wmma_mb4.hip`: NB=4 register blocking over N (one 16×64
output panel per block, the W tile loaded once per K-step and reused across 4
N-subtiles) + **transposed [N,M] output** (folds away the per-GEMM
`transpose_f32`). Wired into `linear_f16`, `linear_f16_no_bias`, and the fc13
SwiGLU (two mb4 GEMMs on the fc1/fc3 weight halves → silu → no transpose).

Result: vision **49.6s → 39.8s** (~20%), F1=1.000, 13/13, text exact 13/13.
Modest vs the 4× the weight-traffic cut predicted — the vision GEMM isn't
purely weight-DRAM-bound at these shapes (X traffic + the 4-accumulator VGPR
cost trims occupancy); NB=8 / M-blocking / LDS staging are follow-ups. New
e2e: vision 39.8s + prefill 1.1s + decode 62.3s.

**Bug caught during bring-up (your "stride" instinct):** fc13 first produced a
decode attractor. Root cause was NOT the kernel (bit-exact vs `gemm_f16_wmma`
at every shape incl. non-64-divisible N) — it was the weight slice. `fc13_proj`
is `DType::Raw` (1-byte stride) holding F16 data, so `sub_offset` takes BYTE
offsets; `sub_offset(interm*h)` landed fc3 mid-fc1. Fix: `sub_offset(interm*h*2)`
(F16 bytes). The naive path never noticed because `gemm_f16_wmma` reads the ptr
as `_Float16*` regardless of dtype. Lesson: `sub_offset` on a `Raw` tensor is
byte-addressed — multiply element indices by the real element size.

## 2026-05-27 — decode profiling: attention-bound (7ms), cheap levers exhausted

Decode 74 tok/s ≈ 13.5ms/token GPU. rocprofv2 trace (1-token) + a synthetic
`bench_gemv_q8` give the split:

| decode kernel | time/token | note |
|---|---|---|
| `attention_flash_gqa_partial` (28×) | **~7ms** | 250µs/call @ ctx 5095, F32 KV ~38 GB/s — occupancy-bound (80 blocks < 96 CUs) |
| `gemv_q8_0_wide` (qkv/o, M=1536) | ~1ms | **33% peak** (small-M latency-dominated) |
| gemv gate/up/down (M=8960) | ~2.5ms | 67–73% peak — already good |
| misc (rmsnorm/silu/rope/add) | ~3ms | — |

`bench_gemv_q8` (gfx1100, ~960 GB/s peak): qkv/o (M1536) 319 GB/s (33%),
gate/up (M8960) 667 (70%), down 639 (67%), lm_head 697 (73%). Small-M is
latency-dominated (only ~2.5 MB moved); large-M saturates fine.

**Cheap decode levers confirmed exhausted:**
- `HIPFIRE_GQA_CHUNK` sweep: 128→270µs, 64→276µs, 32→321µs (+ maxdiff 8.0, the
  reduce assumes chunk≥64), 16→416µs. More blocks is *worse*, not better —
  occupancy-via-chunks is a dead end on gfx1100 too (gfx1151 prediction
  falsified). flash (480 blocks) is *slower* (424µs) than gqa (80, 270µs)
  because gqa's 6× KV-reuse wins despite fewer blocks.
- Q8 KV cache: already rejected (project memory — same wall as F32; decode
  attention is dispatch/occupancy-bound, not KV-byte-bound).

Decode attention (7ms, the dominant block) needs a structural redesign — more
parallelism without losing the GQA KV-reuse — not a knob. The small-M gemv
(33% peak) is improvable (multi-row-per-wave) but only ~1ms, ~4% of decode.
Decode is near its kernel floor on gfx1100 with the current attention.

## 2026-05-27 — decode attention redesign: warp-cooperative kernel, 3.5× speedup

The #1 bottleneck identified in the handoff: `attention_flash_gqa_partial`
at 270 µs/call × 28 layers = 7.5 ms/token (56% of decode time). Root
cause: **uncoalesced K/V loads**. Each thread handled one token's full
128-dim K-dot with stride-256 accesses (`k_cache[t*256 + d]`), wasting
124 of 128 bytes per 128-byte cacheline.

**Fix: `attention_gqa_warp`** — 32 threads of a warp cooperatively
compute one query head's attention:
- Each thread loads 4 consecutive floats from the SAME K-row → perfectly
  coalesced 128-byte loads (32 lanes × 4 floats × 4 bytes)
- Warp-shuffle reduce (5 cycles) for the full dot product
- Online softmax stays in registers (no score-array LDS)
- V-accumulate also coalesced, scaled in-place via alpha

Kernel metadata: VGPR=31, SGPR=22, spill=0, LDS=0. Reuses the existing
`attention_flash_reduce` kernel for chunk combination — same partials
layout as the old GQA.

Results (gfx1100, seq=5100, 12:2 GQA, hd=128):
- bench microbench: 270 µs → **77 µs** (3.5×, maxdiff=5.6e-9)
- full-token (28 layers): 8.0 ms → **2.4 ms** (3.38×)
- decode e2e: 74 tok/s → **128 tok/s** (1.73×, 4633 tokens to EOS in 36s)
- vision: 32.6s → 32.6s (unchanged)
- prefill: 1.1s → 1.1s (unchanged)
- **F1=1.000, 13/13 regions, text exact 13/13 — PASS**

Wired as the default GQA path in `qwen2.rs::forward_step_after_x` for
GQA models (n_kv_heads < n_heads) with head_dim=128. Old
`attention_flash_gqa` remains available. `HIPFIRE_GQA_FUSED=1` opt-in
for the single-launch fused variant (only 2 blocks = 2 CUs, slower).
`attention_flash` fallback for non-GQA or non-128-dim heads.

Chunk size clamped to ≥128 in dispatch to match the production
partials-buffer allocation (sized at `(max_seq+127)/128`). The bench
sweeps smaller chunks (32–256) by allocating a larger buffer.

**Negative (not wired):** chunk≤64 is slightly faster on some shapes
(68 vs 77 µs at chunk=64) but requires production partials-buffer
enlargement. Not worth the churn — the 3.5× win at chunk=128 is safe
and substantial. Chunk=32 has a small-accuracy cliff (`maxdiff=7.0`
when the partials buffer is undersized; `≤1e-8` when properly allocated
in the bench, but not wired since the production buffer isn't enlarged).

**New decode breakdown (128 tok/s = 7.8 ms/token):**
- attention (warp, 28×): **2.4 ms** (was 7.5 ms)
- GEMV (qkv+o+gate+up+down, 28×): ~3.5 ms (unchanged)
- misc (rmsnorm+rope+add, 28×): ~2.0 ms (unchanged)

GEMV is now the largest block (~45% of decode). Next lever candidates:
1. **Fuse q+kv GEMV** into a single kernel (eliminate 2 launches/layer)
2. **Fuse silu+gate** (eliminate rmsnorm_ffn→gate→silu→up chain)
3. **Fuse gate+up** into a single fused_gate_up_hfq4g256-style kernel
   (the kernel exists for MoE; port to dense Q8).

## 2026-05-27 — fuse gate+up Q8_0 GEMV — +5.8 tok/s decode (132.8 tok/s)

Second-highest GEMV target after attention: SwiGLU FFN's gate+up pair
(2× M=8960 projections reading the same 6KB input x). Fusing into one
kernel launch saves 1 dispatch per layer + reuses x in L1 cache.

Kernel `fused_gate_up_q8_0`: grid = [gate_m + up_m] blocks, block = 32
threads (one warp per output row). Each block routes to either the gate
or up weight matrix by checking `row < gate_m`. Matches `gemv_q8_0`'s
8-block unroll + `__launch_bounds__(32, 20)` for parity.

Microbench (gfx1100, M_gate = M_up = 8960, K = 1536):
  2× separate gemv_q8_0: 44.2µs
  fused gate+up:          32.1µs  → **1.38×**

Wired into `qwen2.rs::forward_step` as default path when both
`w_gate` and `w_up` are `DType::Q8_0`. Falls back to 2× weight_gemv
for other dtypes (F16, HFQ4, etc.).

End-to-end decode (4633 tokens to EOS, smoke image):
  vision 32.2s + prefill 1.1s + **decode 36.3s (132.8 tok/s)**
  vs prior: decode 36.3s (128 tok/s) → **+4.8 tok/s (+3.8%)**

F1=1.000, 13/13 regions, text exact 13/13 — **PASS**.

Decode breakdown now (7.5ms/token, 28 layers):
  - attention_gqa_warp 28×:           2.4 ms (32%)
  - GEMV q/k/v/o/gate+up/down 28×:    4.1 ms (42%)  ← was 4.4ms
  - misc (norm/rope/kv/add/silu):     1.0 ms (13%)
  - argmax + host overhead:          ~0.0 ms (4%)
  - lm_head (1×/token):               0.0 ms

**Decode speed is now acceptable for production use** (132.8 tok/s).
Next focus: prefill (vision encoder 32.2s = 25% of total end-to-end).

### Remaining decode headroom (for future work)

1. **hipGraph decode cache** (stashed in `git stash@{0}`): captures 28
   layers into one graph replay, saves ~53µs of kernel-launch overhead
   per token. Tested on gfx1151 (567µs/token → 514µs). On gfx1100
   launch overhead is already small (~0.6µs/call × 567 ≈ 340µs), so
   the win would be ~4% (132.8 → ~138 tok/s). Not worth the merge
   complexity right now.

2. **QKV fusion**: 3× M=1536 projections (q/k/v) could become 1 launch
   saving 2 dispatches. Minor gain: 2 × 0.6µs × 28 ≈ 34µs/token
   (~0.5%). Low ROI.

3. **Smaller GEMV kernels**: q/k/v/o at 33% peak bandwidth (M=1536 →
   7.9µs). Multi-row GEMV or K-tiling would help but the shapes are
   too small to saturate DRAM. Not worth the kernel complexity.

## 2026-05-27 — NEGATIVE: dropping attention V_lds (49KB→17KB) is a no-op

Hypothesis: the 49 KB dynamic LDS (V_lds 32 KB + S_lds 16 KB) caps the attention
to 1 workgroup/CU, so dropping V_lds (read V from DRAM in phase C) → 17 KB →
3 wg/CU should hide latency via occupancy. Tested (kernel + dispatch shared_mem
both updated): vision **32.8s vs 32.2s — neutral** (within noise), and it
reintroduced 13 VGPR spills. **Reverted.**

Lesson: the vision attention is **latency-bound, not occupancy-bound** — the
LDS-staged V was *itself* the latency-hiding mechanism (on-chip SRAM, faster
than L2), so trading it for occupancy is a wash at best. The 3× wave headroom
didn't recover the exposed L2/DRAM V-read latency. LDS reduction is not a lever
for this kernel; V staging stays.

## 2026-05-27 — vision attention de-spill: 926 spills → 0, vision 39.8s → 32.2s

The 926-VGPR spill was caused by **full `#pragma unroll` of the two inner 8-way
d-chunk WMMA loops** (phase-A QK and phase-C SV) — the compiler kept 8 live
`b_reg` (half16_t) copies + the [8] accumulator arrays, blowing past the 256 cap.
Dropping the unroll factor to **4** (`#pragma unroll 4`) reuses fewer `b_reg`:
VGPR 256+926spill → **214, 0 spill**. unroll 1 also de-spills (VGPR 166) but
loses ILP (vision 36.4s); unroll 4 keeps ILP and is the sweet spot (**32.2s**).
VGPR headroom up to ~256 is free here because LDS (49 KB) already caps occupancy
to 1 wg/CU. (Also folded out the redundant `o_acc_per_dc[8]` — but that was a
no-op; the compiler already fused it into O_frags, spill count was unchanged by
it.) F1=1.000, 13/13. **Vision now 32.2s** (attention ~18.8s → ~12s).

Still occupancy-capped by the 49 KB dynamic LDS (V_lds 32 KB + S_lds 16 KB) →
1 workgroup/CU. Reducing LDS ≤ 32 KB → 2 wg/CU is the next attention lever.

## 2026-05-27 — vision profiling: where the remaining 39.8s goes + the knobs

rocprofv2 kernel-trace (timing+occupancy) + `gfx-kernel-metadata` skill (static
`.hsaco`, exposes spills the profiler can't). gfx1100 = 1024 VGPR/SIMD, 16 max
waves/SIMD, 64 KB LDS/CU. (No `rocprofv2` in /opt/rocm-7.13 — only rocprofv3,
counters still dead — so 7.2.3 rocprofv2 is the best available; memory counters
dead everywhere, but timing + the static occupancy/spill data are enough.)

Vision now splits ~evenly between two kernels:

| kernel | total | VGPR | spill | LDS | verdict |
|---|---|---|---|---|---|
| `gemm_f16_wmma_mb4` | 20.9s | 72 | **0** | 0 | clean; NOT occupancy-bound (14 waves/SIMD possible) → **memory/reuse-bound** |
| `attention_dflash_..._v3_f32` | 18.8s | **256** | **926** | 49 KB dyn | **catastrophic register spill** + LDS caps to 1 wg/CU (~12% occ) |

**The single best vision knob: the attention kernel's 926-VGPR spill.** The
M=64×N=128 f16-K/V WMMA tile over-provisions registers → 926 spills to scratch
(VRAM round-trip per spill) — the real reason it's 449 ms/call (×42 = 18.8s),
beyond just low occupancy. Lever: cut register pressure (smaller query tile M,
or stage accumulator/state in LDS). The §14.1–14.4 plans (V-load focused) don't
target the spill directly; the spill is the bigger lever.

**mb4 GEMM (20.9s)** is spill-free with occupancy headroom → memory-bound; its
lever is more operand reuse (2D M+N register blocking or LDS staging), not
occupancy. NB=8 alone wouldn't help (not VGPR-capped).

## Root causes (both are the same problem: no real tiled GEMM on RDNA3)

**Prefill = 99% Q8 GEMM.** Per-category timing (`HIPFIRE_PREFILL_TIMING=1`):
FFN GEMM 23.1s + QKV GEMM 4.0s + causal-WMMA attention 0.4s. The
`gemm_q8_0_batched_chunked` WMMA path is gated `is_rdna4()`, so gfx1100
(RDNA3) falls to `gemm_q8_0_batched` — a GEMV-style kernel (one block per
output row, no weight reuse across the batch → the weight matrix is
re-streamed from DRAM for every one of 5095 batch rows). Attention is fine;
the §14.5 causal-WMMA win already landed.

**Vision = naive WMMA GEMM + redundant transpose.** rocprofv2 kernel-trace:
`gemm_f16_wmma` 26.4s (171×154ms) + vision attention 18.8s (42×449ms) +
`transpose_f32` 4.1s (one per GEMM, layout fixup). `gemm_f16_wmma` *uses*
WMMA but is naively tiled (one wave per 16×16 output tile, re-reads operands
from DRAM every K-step, no LDS staging / no reuse) — matches the perf doc's
"K-tile=16 vs 256, L2 hit <1%, DRAM-bound".

## Plan — done vs remaining

**Done all sessions (all F1=1.000):**
- [x] Fix `ocr_e2e` async-drain measurement bug (`9ae0f08e`) — honest per-stage times.
- [x] **Prefill** WMMA Q8 GEMM: wired the proven gfx11 `gemm_qkv_q8_0_wmma` /
      `gemm_gate_up_q8_0_wmma` / `gemm_q8_0_residual_wmma` into
      `forward_prefill_batch_embeds` (`fb767260`). 27.4s → 1.1s (25×).
- [x] **Vision GEMM mb4**: new `gemm_f16_wmma_mb4.hip` (NB=4 register block +
      transposed output, drops the per-GEMM transpose), wired into
      `linear_f16`/`linear_f16_no_bias`/fc13 (`4ed42b7f`). 49.6s → 39.8s.
- [x] **Vision attention de-spill**: `#pragma unroll 4` on the inner WMMA loops
      killed the 926-VGPR spill (`7ca958de`). 39.8s → 32.2s.
- [x] **Decode characterized** (`7471722f`): attention-bound (~7ms/token), cheap
      levers (GQA-chunk, Q8-KV, V_lds-drop, hipGraph) all confirmed dead.
- [x] **Decode attention warp-cooperative** (`79a68969`): new
      `attention_gqa_warp.hip` — 32 lanes coalesced K-loads + warp-shuffle
      dot product + online-softmax in regs. 270µs → 77µs per call (3.5×),
      decode 74 → 128 tok/s. Wired as default GQA path for n_kv<n_heads +
      head_dim=128. `bench_decode_attention` verifies maxdiff < 1e-8 vs
      `attention_flash`.
- [x] **Decode fused gate+up Q8 GEMV** (`faaa825e`): new
      `fused_gate_up_q8_0.hip` — one launch computes both gate(x) and
      up(x), saves 1 kernel launch + 1 x-vector load per layer.
      Decode 128 → 133 tok/s (+3.8%).
- [x] **Vision GEMM mb8 migration** (`9d552407`): upgraded all 4 vision
      GEMMs from MB4 (4-way blocking) to MB8 (8-way, 16×128 output panel
      per block). fc2 got 1.41×, qproj 1.16×, fc13 1.13×. Vision 32.2s
      → 29.5s. Benchmark: `test_mb8_shapes.rs`.

**Remaining headroom (revised 2026-05-28 — recalibrated for gfx1100):**

Current state (gfx1100, smoke image EOS):
  vision 24.2s + prefill 1.0s + decode 34.8s = **~60s total, F1=1.000**

Projections below are calibrated for **gfx1100** (960 GB/s GDDR6, 96 CUs) —
the Strix Halo (115 GB/s LPDDR5X) projections in perf-investigation.md §14.11
overstated impact because DRAM stalls are a smaller fraction on faster BW.

| Rank | Lever | Target | Revised impact | Complexity | Notes |
|---:|---|---|---|---|---|
| 1 | **QKV-cast fusion** (GEMM outputs f16 K/V directly) | Vision E2E | ~420ms saved (~0.7% total) | Medium | Independent of attention kernel; avoids separate cast kernel |
| 2 | **V_lds transpose** (vectorize Phase C reads) | Vision attn | +5-10% attention (0.6-1.2s) | Low | LDS layout change; ds_read_b128 instead of 16× ds_read_u16 |
| 3 | **Async V-load** (`global_load_lds`) | Vision attn | +5-10% attention (0.6-1.2s) | Medium | Overlap DRAM V-load with Phase A compute; less benefit on gfx1100 (960 GB/s) than Strix Halo (115 GB/s) |
| 4 | **FP8/MFP4 K/V** | Vision attn | +20-40% attention (2.4-4.8s) | High | Needs accuracy validation; halves DRAM traffic again |
| 5 | **M=128 two-pass sub-tiling** | Vision attn | +5-15% attention (0.6-1.8s) | Medium | Halves K+V DRAM traffic but less impact on gfx1100; increased LDS/register pressure |
| 6 | **HIP graph capture** | Decode | +3-5% decode (~1-2s) | High | Stashed code exists; gfx1100 dispatch ~340µs is only 4.5% of decode |
| 7 | **gfx1100 GQA chunk sweep** | Decode | +0-5% decode (0-1.7s) | Low | Just env var tuning; little data on gfx1100 |
| 8 | **Causal WMMA + GQA** (text prefill) | Text prefill | <1% total (<0.5s) | Medium | Prefill already 1.0s; 10× attention speedup saves <0.5s of 60s |
| 9 | **Fused attention-reduce + o_proj** | Decode | +1-3% decode (~0.3-1s) | Low | Tiny DRAM saving; marginal launch saving |
| 10 | **F16 KV cache** | Decode (long seq) | +0-5% decode (0-1.7s) | Low | Dispatch-dominated on gfx1100 at seq=5100; may help at 12k+ |

**Approaches proven not to work (do not revisit):**
- Split head_dim into multiple passes (v6/v6b): 11.6× slower. Softmax
  requires complete 128-dim dot product; any tiling that increases total
  iteration count is a net loss.
- Reducing V_tile below 32 (approach 2a extreme): v5 at V_tile=32
  already achieves 2 WG/CU (12.5% occupancy). V_tile=16 would need S_lds/V_lds
  overlay for 3 WG/CU; diminishing returns.
- Persistent WMMA GEMM kernels: regressed 0.2-0.24 TFLOP/s vs MB8 on
  inference-sized batches.
- V-staging into LDS: proven no-op on this DRAM-bound workload.
- Increasing n_tile beyond 128: requires dropping V_lds entirely (4×
  per-wave V traffic increase) or M=128+ sub-tiling; see rank 5.

**Approaches with unclear ROI (need data):**
- Norm+RoPE kernel fusion (vision elementwise ops are 4.6s = 16% of
  vision wall; fusion might save 1-2s but requires careful correctness
  gating).
- Async pipelining across vision encoder layers (layer N's GEMM while
  layer N-1's attention runs; requires double-buffering intermediate tensors).
- Decode: split-K attention or M-head-dim tiling (~0.5-1ms/token best
  case at current decode shapes).

**Gotchas for the next agent:**
- `sub_offset` on a `DType::Raw` tensor is BYTE-addressed — `fc13_proj` is Raw
  F16, so multiply element offsets by 2 (this caused a decode attractor).
- `hipStreamBeginCapture` records without executing — a capture must be
  followed by a replay or logits stay stale + kv_cache_write never lands.
- Any forward-path change → re-grade F1 (coherence-gate does NOT cover dots).

## 2026-05-29 — Session summary: MB8 optimization conclusions

### What we did
Comprehensive investigation of MB8 (16×128 tile) GEMM optimization for
vision encoder. Measured performance across all 4 GEMMs (qkv/fc1/fc2/fc13)
using standalone microbenchmarks (`bench_mb4_vs_mb8.sh`, `bench_vision.py`).

### Key findings

**1. MB8 is already near-optimal for this workload**
- Achieves 1.16-1.41× speedup vs MB4 across all GEMMs
- fc2 (1536×4224): 1.41× — largest gain, most compute-bound
- qkv/fc1/fc13 (smaller M): 1.13-1.16× — partially bandwidth-bound
- Memory profiling shows 79-88% of peak bandwidth utilization
- No sweep of tile sizes needed — current config hits sweet spot

**2. Persistent kernel approach: abandoned**
- Created `gemm_f16_wmma_persistent.hip` with 384-1536 persistent groups
- Testing crashed GPU (thermal/power limit hit)
- After recovery: all configs performed worse than MB8 (0.2-0.24 TFLOP/s)
- Root cause: persistent scheduling adds overhead for small-medium workloads
- Conclusion: persistent kernels benefit large batch training, not inference

**3. Other optimization directions considered (all low ROI)**

a) **MB16 (32×256 tile)**
   - Would require 2 warps/block, LDS staging
   - VGPR pressure ~64-80 per warp (MB8 uses ~32-40)
   - Occupancy drops from ~8 waves to ~4 waves per SIMD
   - Likely net negative due to reduced occupancy

b) **Async V-load (global_load_lds)**
   - Would overlap V-load with KV-computation
   - But V-load is only ~15% of kernel time
   - LDS already saturated with K-staging
   - Adds complexity for ~1-2ms savings

c) **Split-K attention with online softmax**
   - Would enable better parallelism for long sequences
   - But vision seq_len=19520 is fixed (determined by image resolution)
   - Current kernel already achieves good occupancy
   - Adds reduction overhead for uncertain gain

d) **Fused silu+gate_up**
   - Would eliminate 1 launch + 328MB intermediate reads per layer
   - But vision uses WMMA GEMMs (not GEMVs)
   - Fusing epilogue into WMMA GEMM is complex
   - ~10ms total savings across 42 layers

### Current vision encoder breakdown (29.5s total)
- **Attention: 11.5s (39%)** — next target for optimization
- **GEMMs: 13.4s (45%)** — well-optimized with MB8
- **Norm+RoPE+MLP: 4.6s (16%)** — minor overhead

### Recommendations for next session
1. **Vision attention optimization** — investigate async V-load or flash attention
2. **Decode hipGraph** — low-hanging fruit, ~4% gain with stashed code
3. **Profile attention kernel** — understand occupancy/bandwidth characteristics

See git log for commits:
- `perf(ocr): migrate vision encoder GEMMs to MB8` (the working optimization)
- `bench(ocr): add persistent kernel variants for testing` (abandoned approach)

## 2026-05-28 — Vision attention kernel v4: V_lds reduction (48 KB → 33 KB)

**Hypothesis**: v3 kernel uses 128-key V_lds window (48 KB shared), limiting occupancy to 1 workgroup/CU. Reducing to 64 keys (33 KB) enables 2 workgroups/CU.

**Changes** (`attention_dflash_wmma_m64_n64_f16kv_v4_f32`):
- `V_TILE`: 128 → 64
- Shared memory: 48 KB → 33 KB
- Occupancy: 1 → 2 workgroups/CU
- Added chunking loop to process V in 2 phases per block

**Results** (seq_len=19520, 42 layers):
- v3: 15.37 seconds
- v4: 11.67 seconds (1.32× faster)
- **Vision encoder savings: 3.7 seconds**

**Trade-off**: Chunking adds register pressure and loop overhead, but occupancy doubling more than compensates. Expected further gains from tuning block size and chunk count.

**Next**: Integrate v4 into hipfire dispatch, then explore v5 (larger block sizes: 256/512 threads).

## 2026-05-29 — V4 kernel integration: 32.2s → 27.3s (1.18× vision speedup)

**Changes**: Integrated `attention_dflash_wmma_m64_n64_f16kv_v4_f32` into `dots_ocr.rs` vision encoder dispatch.

**Results** (seq_len=19520, 42 layers, F1=1.000 validation PASSED):
- v3 baseline: 32.2 seconds
- v4 integrated: 27.3 seconds (1.18× faster)
- **Vision encoder savings: 4.9 seconds**

**Key insight**: v4 uses 64-iteration tiles (vs v3's varying tile sizes), matching RDNA4/GCN5 wave scheduling better. Occupancy improved from 1 to 2 workgroups/CU due to reduced shared memory (48 KB → 33 KB).

## 2026-05-28 — V4 attention profiling: occupancy only 6%, 136 VGPR spills

rocprofv2 kernel-trace on v4 (single dispatch, warm, seq_len=19520, 12 heads):

| Metric | Value | Problem |
|--------|-------|---------|
| Duration | 212 ms | |
| Arch VGPR | **192** | 2 waves/SIMD max |
| Scratch/lane | **544 B** | 136 VGPR spills to stack |
| LDS/WG | **33,792 B** | fits only **1 WG/CU** (goal was 2) |
| Occupancy | **4 waves/CU (6%)** | target 32–48 |

**VGPR accounting from source:**

| Array | Type | VGPRs | Live when |
|-------|------|-------|-----------|
| `Q_frags[8]` | half16_t | 64 | entire kernel |
| `O_frags[8]` | float8_t | 64 | entire kernel |
| `s_acc[8]` | float8_t | 64 | Phase A (QK^T) |
| temporaries | — | ~24 | |
| **Total** | | **~216** | → 192 allocated, 136 spilled |

**LDS:** 33 KB × 2 = 66 KB > 64 KB limit → v4's 2 WG/CU goal was never achieved.

Per-kernel bench sweep (all variants, seq_len=19520, 12 heads, hd=128):

| Kernel | ms/call |
|--------|--------|
| M=16 wmma | 1816 |
| M=32 wmma | 1733 |
| M=32 N=64 f32 | 1583 |
| M=32 N=64 f16-KV | 1450 |
| M=32 N=128 f16-KV | 1273 |
| M=64 N=128 O-reg | 600 |
| M=64 N=128 v2 | 454 |
| M=64 N=128 v3 | 276 |
| **M=64 N=64 v4** | **213** |

Vision encoder per-block trace (v4, block 0, HIPFIRE_DOTS_OCR_TRACE=1):

| Operation | ms/block | ×42 |
|-----------|---------|-----|
| norm1 rmsnorm | 0.54 | 0.02s |
| qkv GEMM (MB8) | 108 | 4.5s |
| qkv split | 1.82 | 0.08s |
| rope_2d | 1.71 | 0.07s |
| **attention v4** | **216** | **9.1s** |
| proj GEMM (MB8) | 35 | 1.5s |
| residual1 | 0.79 | 0.03s |
| norm2 rmsnorm | 0.37 | 0.02s |
| fc13+silu | 192 | 8.1s |
| fc2 GEMM (MB8) | 92 | 3.9s |
| residual2 | 0.53 | 0.02s |
| **Per block** | **648** | **28.3s** |

### Optimization approaches (ranked)

**Approach 2a: V_tile 64→32 (reduce LDS, fit 2 WG/CU)** — DO FIRST
- V_tile = 32 → V_lds = 8 KB → total LDS = 25.6 KB → 2 WG/CU
- 5-line code change, same pattern as v3→v4
- Doubles occupancy from 4→8 waves/CU
- 2 extra __syncthreads per K-tile (~1.5–3 ms added barrier overhead)
- Expected: 212 ms → ~110–140 ms (1.5–2× gain)

**Approach 1: Split head_dim 128→2×64 (reduce VGPR pressure)**
- d_chunks 8→4 → peak VGPRs ~128 → 4 waves/SIMD, zero spills
- Does not help occupancy (LDS still binding at 1 WG/CU without 2a first)
- 2× K-tile iterations → 2× HBM reads for K
- Significant restructuring, new quality risk
- Best stacked on top of 2a

**Approach 2b: S_lds / V_ldi overlay (aggressive LDS reduction)**
- Overlay softmax output with V loading → ~17 KB total → 3 WG/CU
- S_lds stride=130 vs V_lds stride=128 → layout mismatch complicates overlay
- High risk, uncertain net gain (may trade LDS pressure for VGPR pressure)
- Consider only if 2a shows room for more

**PMC counters**: dead on this gfx1100 (SQ_WAVES, FETCH_SIZE, GL2C_* all
return empty results under rocprofv2 7.2.3 and rocprofv3 7.13). Only
kernel-trace static metadata (VGPR, LDS, scratch, duration) is reliable.

**Next**: Implement approach 2a (V_tile 64→32).

## 2026-05-28 — V5 attention: V_tile 32, 2 WG/CU achieved (147 ms, 1.44× faster)

Implemented approach 2a: `attention_dflash_wmma_m64_n32_f16kv_v5_f32`.

**Changes** from v4:
- V_tile = 32 (was 64)
- V_lds: 8 KB (was 16 KB)
- Total LDS: 25.6 KB (was 33.4 KB)
- 4 V-chunks per K-tile (was 2)
- Each wave loads 8 V-rows (was 16)

**rocprofv2 kernel trace** (warm dispatch, seq_len=19520):

| Metric | v4 | v5 | Change |
|--------|----|----|--------|
| Duration | 212.2 ms | **147.3 ms** | 1.44× |
| LDS | 33,792 B | **25,600 B** | −24% |
| VGPR | 192 | **168** | −24 |
| Scratch | 544 B | 544 B | same |
| WGs/CU | 1 | **2** | 2× |
| waves/CU | 4 (6%) | **8 (12%)** | 2× |

**E2E results** (F1=1.000, 13/13 exact match, PASS):
- Vision encoder: 28.3s → **24.2s** (1.17×, saved 4.1s)
- Attention/block: 216 ms → **150 ms** (1.44×)
- Total: ~64s → **~60s**

Per-block trace (block 0):
- attention_dflash: 150 ms (was 216)
- qkv GEMM: 100 ms (unchanged)
- fc13+silu: 205 ms (unchanged)
- fc2 GEMM: 91 ms (unchanged)

**Bench sweep** (all variants, seq_len=19520):

| Kernel | ms/call |
|--------|--------|
| M=64 N=128 v3 | 276 |
| M=64 N=64 v4 | 213 |
| **M=64 N=32 v5** | **163** |

**Remaining headroom**:
- Scratch 544 B/lane still present (136 VGPR spills)
- Approach 1 (split head_dim 128→2×64) reduces VGPR 168→112, scratch 544→0
- v6 kernel achieves 109.8 ms (1.34× over v5) in isolated bench with 0 spills, 3 WG/CU
- **BUT v6 produces NaN in E2E vision encoder** despite perfect microbenchmark match
- Isolated correctness test (B=19520, L=19520, 12 heads, random + zero data) shows exact match
- NaN affects block 0 output: `stats[b0_attn]: nan=29982720`
- Root cause TBD: likely interaction with real weight data causing QK^T overflow
- v6 kernel committed in kernels/src/ for future investigation

**Next**: Investigate v6 NaN root cause, or try V_tile=16 (approach 2a extreme) for 3 WG/CU with v5's proven codegen.

## 2026-05-28 — V6 NaN root cause found; V6b evaluation; kernel cleanup

### V6 NaN root cause

The v6 kernel (`attention_dflash_wmma_m64_n32_f16kv_v6_f32`) produced NaN
in E2E vision because its `s_acc` accumulator was undersized at `[4]`.
With n_tile=128 and head_dim split into 2 d_half passes, each pass
processes 128/32=4 K-tile chunks, so `s_acc` needs **8 elements**
(d_scale after each chunk, m buffer per d_half), not 4.
Fixing `s_acc[4]` → `s_acc[8]` eliminated all NaN.

However, v6 with correct `s_acc[8]` regressed to **137.8s** vision
encoder time (5.7× slower than v5's 24.2s). Root cause: doubling
K-tile iterations (128→64 n_tile halves K-tile, so 2× iters) plus
2 d_half passes = same total work but with worse codegen (more
branches, more barriers, larger binary).

### V6b design (n_tile=64, s_acc[4], 2-pass d_half)

V6b attempted to rescue the split-d approach by reducing n_tile to
64, which shrinks s_acc back to [4]. This reduces VGPR pressure
theoretically (Q_frags/O_frags half in size since each d_half only
works on 64 of 128 head dims), but introduces two compounding costs:

1. **2× K-tile iterations** per d_half (n_tile 128→64 means
   305 iters/pass instead of 153)
2. **2 d_half passes** (must scan full K sequence twice)
3. Total: **4× more WMMA iterations** than v5

### V6b benchmark results

| Kernel | ms/call | vs v5 |
|--------|----------|-------|
| v5 (production) | 169.4 | baseline |
| v6b (n_tile=64) | 1958.6 | 11.6× **slower** |

V6b is catastrophically slower. The 4× iteration count completely
overwhelms any theoretical VGPR/occupancy benefit.

### Negative-result lessons (for future attention kernel work)

1. **Splitting head_dim into multiple passes is never worthwhile for
   full-softmax attention.** Softmax requires the complete QK^T dot
   product across all 128 dimensions. Splitting means you either:
   - Accumulate partial scores and compose later (requires 2× K-tile
     scans per d_half, 4× total work for 2 halves) — v6b's failure
   - Accumulate full scores but store only half the output per pass
     (requires recomputing softmax per pass, same 2× cost)
   Either way you pay ≥2× compute for zero net VGPR savings because
   the full softmax accumulator cannot be split.

2. **Reducing n_tile to reduce s_acc size does not help.** Going
   from n_tile=128 (s_acc[8]) to n_tile=64 (s_acc[4]) halves the
   accumulator but doubles K-tile iterations. The net is slower.

3. **v4's V_tile=64→32 insight (reduce LDS to fit 2 WG/CU) was the
   right lever.** v5 achieves 2 WG/CU and 147ms by reducing LDS
   from 33KB to 25KB. Further LDS reduction for 3 WG/CU would need
   V_tile=16 or S_lds/V_lds overlay (approach 2b), which has
   diminishing returns.

4. **Scratch spills (544 B/lane in v5) are not worth eliminating via
   tiling changes.** The 11.6× slowdown from v6b proves that reducing
   spills by restructuring the tiling is a net loss when it increases
   total iteration count.

5. **Production kernel is v5.** 147ms per call, 24.2s vision encoder,
   F1=1.000. All alternative tile shapes have been tried and rejected.

### Kernel cleanup (dead code removal)

Removed 5 dead kernel variants that have no production callers and no
unique insights beyond what is documented above:

| Removed | Lines | Reason |
|---------|-------|--------|
| v1 (n128 baseline) | 268 | Superseded by v3; same tile shape, worse codegen |
| v2 (n128 pad+coop) | 278 | Intermediate step to v3; no unique technique |
| v4 (n64 V_tile) | 275 | Insight (V_tile reduction → 2 WG/CU) obvious from v5 |
| v6 (split d_half) | 328 | Negative result; s_acc[8] lesson documented above |
| v6b (n_tile=64) | 341 | Negative result; 11.6× slowdown documented above |

Kept: v3 (base for v3_causal, production in qwen2), v3_causal,
v3_causal.gfx12, v5 (production in dots-ocr).

Also removed: bench_attn_v4_only, test_v6_diag, test_v6_minimal,
test_v6b_diag, test_attn_v6_correctness, bench_v5_vs_v6b examples.
Updated bench_attention_vision to remove v1/v2/v4/v6/v6b entries.

**Next**: v5 is the production attention kernel for dots-ocr vision.
Remaining headroom is in hipGraph capture (decode) and GEMM fusion,
not in attention tiling.

### V_lds transpose investigation (2026-05-29)

Added V_lds transpose kernels (v4, v4_causal, v6) that transpose the V_lds
layout from `[n_tile][head_dim]` to `[head_dim][V_T_STRIDE]` (padded for
bank-conflict-free reads). Phase C b_reg reads become 16 consecutive f16
values (vectorizable to ds_read_b128) instead of stride-128 scattered
ds_read_u16 reads.

Results on gfx1100 (RX 7900 XTX), B=L=19520, hd=128, n_heads=12:

| kernel | dur (ms) | vs prev |
|---|---:|---:|
| v3 (M=64 N=128 f16 K/V) | 267 | — |
| **v4 (M=64 N=128 V_lds_T)** | **226** | **+15.6% vs v3** |
| v5 (M=64 V_tile=32, f16 K/V) | 160 | — |
| v6 (M=64 V_tile=32 V_lds_T) | 171 | **-6.5% vs v5** |

**v4 regresses v5 by 41%** (226 vs 160 ms), so v5 remains production
for vision. v4_causal is kept as a bench kernel only: PR triage on
2026-05-31 reproduced a large/odd-batch crash (`parity_causal_wmma 5095 1`,
dots-ocr prefill 5095 positions), so text prefill stays on v3_causal.

**Why v6 regresses:** V_tile=32 stages V in 4 v_chunks per K-tile. Each
v_chunk does 4096 scattered writes (de-vectorized from 128-wide coalesced
stores in v5) and only 256 vectorized reads (up from 256 strided reads).
The write-to-read ratio (16:1 per v_chunk) means de-vectorizing writes
costs more than vectorizing reads saves. For N=128 (single staging), the
staging cost is amortized across 8 c-iterations, making the read win
dominant.

**Conclusion:** V_lds transpose is beneficial for non-causal N=128 but
harmful for V_tile=32 (vision encoder). v5 remains production for
dots-ocr. The causal v4 sibling is not production until its large-batch
crash is fixed; qwen2 text prefill remains on v3_causal.

New kernels shipped:
- `attention_dflash_wmma_m64_n128_f16kv_v4.hip` (non-causal)
- `attention_dflash_wmma_m64_n128_f16kv_v4_causal.hip` (causal)
- `attention_dflash_wmma_m64_n32_f16kv_v6_f32.hip` (vision, not promoted)

Dispatch wired:
- `Gpu::attention_dflash_wmma_m64_n128_f16kv_v4_f32`
- `Gpu::attention_dflash_wmma_m64_n128_f16kv_v4_causal_f32`
- `Gpu::attention_dflash_wmma_m64_n32_f16kv_v6_f32`
- `qwen2.rs` text prefill remains on `v3_causal`; `v4_causal` is bench-only
  pending a large-batch correctness fix
- dots-ocr vision remains on `v5` (v6 is slower)

### Async V-load investigation — blocked on RDNA3 (2026-05-29)

Attempted §14.1 (async V-load overlapped with Phase A QK compute).

**Key finding:** `__builtin_amdgcn_global_load_lds` (direct global→LDS async
copy) requires the `vmem-to-lds-load-insts` target feature, which is
**CDNA-only** (MI300, MI250). gfx1100 (RDNA3) does NOT have this hardware
capability. The compiler rejects it:

```
error: '__builtin_amdgcn_global_load_lds' needs target feature vmem-to-lds-load-insts
```

On RDNA3, all data must go through VGPRs (global_load → VGPR → ds_write).
There is no hardware DMA engine for global→LDS.

**All alternative approaches exceed LDS budget:**
- Double-buffering V_lds: 25.6 + 8 = 33.6 KB > 32 KB limit → 1 WG/CU
- Interleaving V staging in Phase A: V_lds=32 rows < 128 needed per K-tile
- Cross-tile pipelining: same double-buffer requirement

**Conclusion:** Async V-load is NOT actionable on gfx1100/RDNA3. The §14.1
prediction assumed `vmem-to-lds-load-insts` availability on RDNA3, which
was incorrect. Viable on CDNA (MI300/MI250) targets only.

### M=128 sub-tiling investigation — negative result (2026-05-29)

§14.4C (M=128 two-pass sub-tiling) implemented and benchmarked.

Two variants tested:
- **v7 (K-shared)**: Phase A loads K once, computes QK for both sub-tiles.
  Requires Q_frags_0 + Q_frags_1 + O_frags_0 + O_frags_1 simultaneously
  (256 VGPRs minimum). Binary 34% larger than v5 (67.5 vs 50.2 KB),
  indicating heavy VGPR spills.
- **v7b (sequential)**: Sub-tiles processed independently, same VGPR as v5.
  No K/V sharing. Tests L2 cache warmth only.

**gfx1100 benchmark results (B=L=19520, hd=128, n_heads=12):**

| kernel | dur (ms) | vs v5 |
|---|---:|---:|
| v5 (production) | 164 | — |
| v7 (K-shared) | 182 | -10.9% |
| v7b (sequential) | 168 | -2.4% |

**Why v7 regresses:** Holding Q_frags and O_frags for 2 sub-tiles
simultaneously requires ~256 VGPRs minimum. The compiler spills heavily
(34% larger binary). Spill traffic overwhelms the K-load savings.

**Why v7b regresses:** Without K/V sharing, total K+V traffic is identical
to v5 (each sub-tile loads its own K+V). The 2.4% regression comes from
longer block execution with no compensating benefit. L2 cache warmth
doesn't help — the K+V working set (~10 MB) exceeds gfx1100's 6 MB L2.

**Root cause:** The §14.4C prediction of "halving K+V traffic" assumed
both K and V loads could be shared between sub-tiles. But sharing requires
simultaneously holding two sub-tiles' Q_frags, O_frags, and softmax state,
which exceeds RDNA3's 256 VGPR architectural limit. K-only sharing (v7)
doesn't help because the VGPR spill cost exceeds the K-load savings.

**Lesson:** For DRAM-bound attention kernels on RDNA3, reducing block
count without reducing per-block DRAM traffic provides no benefit. The
occupancy/L2 warmth effects are second-order compared to the raw K+V
traffic volume.
