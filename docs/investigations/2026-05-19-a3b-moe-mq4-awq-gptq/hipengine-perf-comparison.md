# hipEngine perf comparison — what hipfire is missing on Qwen3.6-35B-A3B-PARO / gfx1151

Read-only audit, 2026-05-20. Cross-references:

- hipEngine survey: `/tmp/hipengine-survey/`
- hipfire branch: `paroquant-a3b` worktree

## Baseline (this branch, `bench_qwen35_mq4`, prefill 256 / gen 100 / kv-mode q8 / gfx1151)

| Metric  | hipfire (today) |
| ---     | --:             |
| Prefill | 31.4 tok/s      |
| Decode  | 30.8 tok/s      |
| Load    | 26.3 s          |

hipEngine does NOT publish a directly comparable gfx1151 row for Qwen3.5-35B-A3B-PARO MoE: the
retained gfx1151 row in `docs/ROOFLINE-gfx1151.md:325-327` is the 0.8B PARO model (prefill
2451 tok/s, decode 145 tok/s). hipEngine's W7900/gfx1100 35B-A3B row is from
`benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json` and the
audit in `docs/OPTIMIZE.md:256-273` (~115 tok/s decode, ~877 dispatches/token). Direct gfx1151
A3B numbers therefore can't be cited from this snapshot — but the architectural and
methodology gaps are visible regardless.

---

## Roofline-gap verdict — bandwidth-bound, with kernel-bound tail

From `docs/ROOFLINE-gfx1151.md:182-217`: gfx1151 has 256 GB/s LPDDR5X theoretical, ~221 GB/s
measured read, 59.4 TFLOP/s FP16/BF16/INT8 WMMA. Active-weight bandwidth ceiling for
A3B-PARO (3B active params × 0.5 byte/W4 ≈ 1.5 GB/token) is **~147 tok/s** at the 221 GB/s
practical ceiling (`ROOFLINE-gfx1151.md:248-251`). KV reads + attention add ~10-20% at 256
prefill context, so the headroom for **decode is roughly 110-130 tok/s** before bandwidth
saturation. We are at 30.8 tok/s → **~25% of bandwidth roof**. The gap is dominated by
*kernel inefficiency / dispatch overhead* (decode is launch-floored), not memory bandwidth.

**Prefill** (31.4 tok/s) is far below kernel-bound ceiling. hipEngine measures their own 0.8B
prefill at 22% of W7900 (`ROOFLINE-gfx1151.md:602-605`) and explicitly diagnoses it as path
selection / dequant / attention / occupancy, not a hardware wall. Prefill is **kernel-bound**
with room for 3-10× before hitting the FP16/BF16 WMMA roof.

---

## Top-5 lever differences (ranked by suspected wall-clock impact)

### 1. Decode hipGraph capture is OFF for MoE in hipfire — DISABLED due to drift

- **Description**: hipEngine captures the full decode step into a hipGraph and replays it
  with the device-side token auto-advancing
  (`hipengine/runtime/qwen35_paro_runner.py:1469-1559`, `capture_decode_graph`). On gfx1151
  the M.4 audit (OPTIMIZE.md:256-273) shows 877 dispatches/token; without graph replay each
  dispatch pays ~10-20 µs HIP launch overhead → 9-17 ms/token of pure launch cost.

  Hipfire has the infrastructure (`crates/hipfire-arch-qwen35/src/qwen35.rs:3842-3879`,
  `begin_graph_capture` / `graph_launch`), but it is **hard-disabled for MoE configs**
  (`qwen35.rs:3765-3777`): "state diverges from the direct-dispatch path after ~30-50 decoded
  tokens." `qwen35.rs:3811-3813` says MoE always takes direct path unless
  `HIPFIRE_GRAPH_MOE=1` is set. A3B is MoE, therefore this is OFF for the bench.

- **Suspected impact**: **+20-40% decode** at minimum (matches the 877-dispatch/token x
  ~15 µs/dispatch arithmetic and the 877 → ~50 dispatch ratio under replay). hipEngine's
  W7900 comparison-tables show graph_replay vs eager are 32.96 vs 32.01 tok/s at 512/128
  (OPTIMIZE-DENSE.md:81-86) — only ~3% on dGPU because launch overhead is smaller — but on
  gfx1151's slower scheduler, every avoided submit matters proportionally more.
- **Prefill impact**: none directly (prefill is GEMM-shaped, batched).
- **Effort**: **L**. The drift bug needs root-causing first — `qwen35.rs:3787` suspects
  atomic/wavefront ordering inside `gated_delta_net_q8` reductions. Fixing it is a
  correctness investigation, then enabling is one-line.
- **Citations (hipEngine)**: `runtime/qwen35_paro_runner.py:1469-1559`,
  `runtime/qwen35_paro_runner.py:3282-3343` (Qwen35ParoDecodeGraph),
  `docs/OPTIMIZE.md:255-273` (877 dispatches/token measurement).
- **Citations (hipfire)**: `crates/hipfire-arch-qwen35/src/qwen35.rs:3765-3825`,
  `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:128-156` (DPM warmup + prefill warmup).
- **Portability**: portable to all RDNA arches; CDNA already has HIP graph support.

### 2. GDN prefill is strictly sequential per token in both engines — same wall, but hipEngine has the open ticket

- **Description**: GDN linear-attention prefill runs `for (token = 0; token < tokens; ++token)`
  with `__syncthreads` per step in both engines. hipfire:
  `kernels/src/gated_delta_net_q8.hip:56` (`for (int t = 0; t < n_tokens; t++)`); hipEngine:
  `kernels/hip_gfx1100/linear_attn/gdn.hip:304-375`
  (`qwen35_gdn_prefill_recurrent_k2_kernel`). hipEngine's OPTIMIZE-DENSE.md:140-145 calls this
  "the single largest structural prefill miss" — `48 × 4096 ≈ 196k` strictly sequential
  token-steps for the 27B dense model. Same applies to 35B A3B.

  hipEngine **has not yet shipped** chunkwise GDN prefill — they have it as P1 in
  OPTIMIZE-DENSE.md:146-159 ("highest expected leverage", references atlas/FLA/vllm
  chunkwise ports). This is a structural lever, not a tuning knob.

- **Suspected impact**: **+30-50% prefill** if chunkwise is landed (estimate from the GDN
  share of prefill kernel time — 20.5%/21.1%/15.7% at 512/4K/32K per OPTIMIZE.md:178 — and
  the multi-token-per-block reuse a chunkwise rewrite enables).
- **Decode impact**: none (decode is per-token by definition).
- **Effort**: **L**. Real algorithm port (WY-chunkwise from Gated Linear Attention papers,
  see `~/amd-gpu-tuning/reference/atlas/kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu`
  per hipEngine OPTIMIZE-DENSE.md:46-49).
- **Citations (hipEngine)**: `kernels/hip_gfx1100/linear_attn/gdn.hip:304-375`,
  `docs/OPTIMIZE-DENSE.md:140-145`, `docs/OPTIMIZE-DENSE.md:244-260` (P1 candidate).
- **Citations (hipfire)**: `kernels/src/gated_delta_net_q8.hip:56-86`,
  `crates/hipfire-arch-qwen35/src/qwen35.rs:4481` ("inner gated_delta_net_q8_batch_seq loop
  is still sequential per token").
- **Portability**: chunkwise GDN is architecture-agnostic and would work on all RDNA/CDNA.

### 3. Multi-rotation fusion: `paro_rotate2_kernel` and `paro_rotate3_kernel`

- **Description**: hipEngine has kernels that take **one input x** and emit it rotated under
  **N different rotation tables** in a single launch, using grid.z to select which rotation:
  `kernels/hip_gfx1100/rotary/paro_rotate.hip:163-219` (rotate2 — two outputs, two pairs/theta
  tables) and `:222-330` (rotate3 — three outputs). The shared input x is read **once** and
  the LDS rotation buffer is reused per grid.z plane.

  Hipfire has fused-rmsnorm-rotate and fused-silu-rotate, but each rotation is a separate
  launch when multiple downstream consumers need different rotated bases. From
  `qwen35.rs:5325-5350` and `crates/rdna-compute/src/kernels.rs:335-353`, hipfire's path is:
  one `fused_silu_mul_rotate_mq_batched_for` per consumer.

- **Suspected impact**: **+3-7% decode** on MoE because the routed-experts down-projection and
  shared-expert down-projection use different rotation tables but read the **same SwiGLU
  intermediate** — that's exactly what `silu_mul_pair_rotate_out_kernel`
  (`kernels/hip_gfx1100/fused/paro_silu.hip:144-197`) optimizes. Audit decode rotation cost:
  M.4 (OPTIMIZE.md:262) measures rotation at 9.4-9.6% of decode kernel time at 160 calls/token.
- **Prefill impact**: **+1-3%** — prefill rotation is 6.7-7.7% per OPTIMIZE.md:182, batched.
- **Effort**: **M**. New HIP kernel + dispatch wiring; needs AWQ-aware variant. Hipfire's
  rotation is FWHT-based, not Givens; the algorithmic structure is different, so this is a
  port-and-adapt, not a copy.
- **Citations (hipEngine)**: `kernels/hip_gfx1100/rotary/paro_rotate.hip:163-219`,
  `kernels/hip_gfx1100/rotary/paro_rotate.hip:222-330`,
  `kernels/hip_gfx1100/fused/paro_silu.hip:87-197`.
- **Citations (hipfire)**: `crates/hipfire-arch-qwen35/src/qwen35.rs:2860, 2896, 2929, 2959,
  5052, 5175`, `kernels/src/fused_silu_mul_mq_rotate.hip`.
- **Portability**: portable.

### 4. AWQ pack8 dot-product GEMV layout for non-expert decode (Marlin-K vec8)

- **Description**: hipEngine's `gemv_awq_dual_pack8_kernel`
  (`kernels/hip_gfx1100/quant/paro_awq_gemv.hip:612-700`) is the W4 decode workhorse: 128
  threads, `__launch_bounds__(128, 4)`, 8-way packed AWQ dequant, dual-projection fusion
  (q+k or k+v same-input). hipEngine measured **+5.6% decode** at 512/128 and 4K/128 from
  porting the Marlin-K vec8 layout + qweight-neutral repack (`docs/OPTIMIZE.md:295`
  `D2.1 accepted`).

  Hipfire uses HFQ4G256 group-256 layout (different storage), and the equivalent
  is `gpu.fused_qkvza_hfq4g256` (`qwen35.rs:2791`). The decode dispatch for non-MoE GEMV
  uses `weight_gemv` (`qwen35.rs:2803-2806` fallback). Hipfire does have pack8 dispatch
  surfaces (`crates/rdna-compute/src/dispatch.rs:10966` `gemv_hfq4g256_moe_gate_up_k8`),
  but layout/packing differs from AWQ.

- **Suspected impact**: **+5-8% decode**. Direct port of the AWQ pack8 layout for non-expert
  linears (lm_head, attention projections) — these are the W8A16-equivalent ~16% of decode
  kernel time per M.4 (OPTIMIZE.md:257). Hipfire's lm_head is already W8A16
  (`qwen35.rs:NANOVLLM_PARO_LM_HEAD_W8A16=1` analog in ROOFLINE-gfx1151.md:495).
- **Prefill impact**: neutral.
- **Effort**: **M-L**. Requires repacking weights at load time (hipfire's MQ4 format would
  need an alternate pack8 storage), and a new dual-projection GEMV kernel.
- **Citations (hipEngine)**: `kernels/hip_gfx1100/quant/paro_awq_gemv.hip:497-700` (single +
  dual pack8 GEMV), `docs/OPTIMIZE.md:285-295` (D2.1 Marlin-K port, accepted +5.6%).
- **Citations (hipfire)**: `crates/hipfire-arch-qwen35/src/qwen35.rs:2791` (fused_qkvza),
  `crates/rdna-compute/src/dispatch.rs:62, 573-603` (pack8 GEMV surface).
- **Portability**: layout decision is RDNA-wide; gfx1100/gfx1151/gfx12 all have wave32 + dot
  variants suitable.

### 5. AOTriton long-context attention and shape-bucketed graph dispatch

- **Description**: hipEngine ships AOTriton V3 BF16 head-dim-256 forward images and uses them
  for full-attention prefill above 512 tokens
  (`docs/PREFILL.md:936-1105`, `kernels/hip_gfx1100/attention/aotriton_runtime/0.11.2b/`).
  Threshold `--attn-aotriton-min-tokens 512` is the deployment default
  (OPTIMIZE.md:213-214: native attention @ 4K = 662 tok/s; AOTriton = 2346 tok/s, a **+254%
  prefill gain** at 4K). hipfire uses a custom flash-attention HIP kernel for all token
  counts.

  Also: hipEngine uses **shape-bucketed graph buckets** for c=1 decode at 512/4K/32K/128K
  contexts (decode graph captures different `num_splits` and split_K configurations per
  bucket — see `runtime/qwen35_paro_runner.py:1500` `num_splits = max(1, (position +
  replay_span + decode_chunk_size - 1) // decode_chunk_size`).

- **Suspected impact**: **+0% to +200% prefill at long context**, ~0% at the current bench
  shape (prefill=256 < 512 threshold so AOTriton wouldn't fire). At the bench's prefill=256
  / gen=100 / kv-mode q8 row, attention is a small share. But the user's reported "long
  context" implication (KV INT8 at 256K) does benefit from grouped-GQA split-K
  (`OPTIMIZE.md:303` D3.1 accepted **+42-105% decode at 32K-128K** from hipEngine's
  grouped-GQA paged producer port).
- **Effort (AOTriton)**: **L** — vendor distribution, dynamic loader. **Effort (split-K
  grouped-GQA producer)**: **M**.
- **Citations (hipEngine)**: `docs/PREFILL.md:936-1105` (AOTriton distribution),
  `docs/OPTIMIZE.md:303-307` (D3.1-D3.3 long-context wins).
- **Citations (hipfire)**: `kernels/src/attention.hip`, `kernels/src/attention_flash.hip`,
  `kernels/src/attention_flash_q8_0_tile.hip`.
- **Portability**: AOTriton is gfx1100/gfx1151 only currently; not portable to gfx9/gfx10.

---

## Top-3 quick wins (≤1 day, ≥10% perf each)

### Quick win #1 — Enable decode hipGraph for MoE (after fixing the drift)

**Sketch**: The drift is suspected in `gated_delta_net_q8` atomic / wavefront-ordering
behaviour under capture-replay (`qwen35.rs:3787`). Try:

1. Replace any `atomicAdd` in `kernels/src/gated_delta_net_q8.hip` with a deterministic
   2-stage warp-shuffle reduction (the kernel already does `__shfl_down` in lines 72-74; check
   if the S-state update path has any atomic dependency).
2. If clean: flip `HIPFIRE_GRAPH_MOE=1` to default-on for MoE only when
   `allow_moe || config.num_experts == 0` passes the drift gate. Probe with the existing
   `Count from one to twenty` reproducer (`qwen35.rs:3788-3791`).
3. Coherence-gate test then bench.

**Expected**: +20-40% decode (30.8 → 37-43 tok/s).
**Files**: `crates/hipfire-arch-qwen35/src/qwen35.rs:3796-3825`,
`kernels/src/gated_delta_net_q8.hip`.

### Quick win #2 — Sweep `PREFILL_MAX_BATCH` for chunk-size tuning

**Sketch**: hipEngine's prefill grouping uses `--prefill-moe-chunk-size 1024` and `--prefill-
linear-chunk-size 1024` defaults (`docs/OPTIMIZE.md:55-58`). Hipfire's `PREFILL_MAX_BATCH`
comment (`qwen35.rs:4474-4485`) says current default is 256 (chunk count for pp2048 = 8).
For pp=256 bench this is one chunk, but raising it to 512 may help once the MoE prefill
kernels saturate.

For the immediate bench (prefill=256), the better lever is **dropping the chunk size to
match the MoE WMMA tile multiple** (16). hipEngine's `MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS=2`
(`OPTIMIZE.md:204` P1.4) — hipfire already takes Path 2 grouped-WMMA for arch_supported
(`qwen35.rs:5101`), but with N=256 and k=8, m_total ≈ 2048 + n_exp×15 (≈ ~2128 for 8 experts).
Worth profiling whether the WMMA path is grid-saturated on gfx1151's 40 CUs — at tile (16x16)
the grid is 2*mi/16 × m_total/16 ≈ 224 × 128 = 28672 tiles, ample.

**Expected**: +5-15% prefill on long prefills; mostly a no-op at pp=256 — but **double-check
that the Path 2 grouped-WMMA actually fired** (env `HIPFIRE_MOE_GROUPED_GEMM=1`). If we are
hitting the per-token GEMV fallback at decode-shape, that's the real miss.
**Files**: `crates/hipfire-arch-qwen35/src/qwen35.rs:5091-5169` (verify Path 2 fired).

### Quick win #3 — Profile current dispatch counts and decode time/token

This is the cheapest win because we don't yet know whether hipfire's current bench is
dispatch-bound or kernel-bound on gfx1151. hipEngine's audit drove every accepted/rejected
candidate in OPTIMIZE.md from rocprofv3 `--kernel-trace --selected-regions` (`docs/OPTIMIZE.md:151-167`).

**Sketch**: Run `bench_qwen35_mq4 --emit-atlas` (the bench already supports JSONL emit —
`crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:37-42`) and add a rocprofv3
kernel-trace pass. Rank by total kernel time; expect to find:

- If GDN kernels dominate (>20%) → win #2 (chunkwise GDN) is the lever.
- If launch gaps dominate (sum of kernel time << wall time) → win #1 (graph) is the lever.
- If `gemv_hfq4g256_*` and `gemv_hfq4g256_moe_*` dominate → memory bandwidth is the wall;
  port AWQ pack8 dual layout (lever #4).

**Expected**: not a perf win itself, but **redirects the next day's work** with hipEngine-grade
evidence rather than guessing.
**Files**: bench is ready; needs a `rocprofv3` wrapper script.

---

## Single biggest quick-win

**Quick win #1 — fix MoE decode hipGraph drift and enable replay.** The infrastructure is
already there, the disable is a known bug with a reproducer, and at 877 dispatches/token
(hipEngine's measured MoE A3B value, scales to a similar order on hipfire), eliminating
launch overhead is the highest-leverage decode change for the least new code. Expected
+20-40% decode (30.8 → 37-43 tok/s) in ≤1 day if the drift root-causes cleanly. The
caveat: it requires solving a real correctness bug first, so the time risk is bimodal.

If that bug doesn't yield in a day, **fall back to win #3 (profile)** to avoid speculation
and pick the next lever from data.

---

## Methodology lessons hipEngine learned that hipfire should adopt

From `docs/LESSONS-LEARNED.md` and `docs/BENCHMARK.md`:

1. **"Audit first, optimize second"** (`LESSONS-LEARNED.md:510`) — every candidate row in
   their OPTIMIZE.md cites a rocprof bucket share before claiming a win. Hipfire's current
   bench harness emits aggregate tok/s but no per-kernel attribution.
2. **"Fast rows are invalid until output sanity proves they are real"**
   (`LESSONS-LEARNED.md:549`) — hipfire's coherence gate already enforces this; hipEngine's
   `BENCHMARK.md:244-289` adds "Post-run quality gates" as a hard requirement on every
   accepted row. Worth importing the "+1.7% τ regression is acceptable for +10% prefill if
   coherence passes" tradeoff language into hipfire's perf workflow.
3. **"Output buffers alone are rarely enough under graph replay"**
   (`LESSONS-LEARNED.md:983`) — fusion that only reduces dispatch count (no arithmetic /
   data-reuse improvement) tends to wash out at <1% under graph replay. Multiple D1.x
   fusions in OPTIMIZE.md were rejected for this reason — useful negative-result library.

---

## Files of interest (hipEngine snapshot)

- `/tmp/hipengine-survey/hipengine/runtime/qwen35_paro.py` — 6570-line orchestrator
- `/tmp/hipengine-survey/hipengine/runtime/qwen35_paro_runner.py` — generator + graph capture
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.hip` — compact WMMA MoE prefill
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip` — recurrent GDN kernels
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` — W4 pack8 GEMV
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/rotary/paro_rotate.hip` — rotate1/2/3 multi-fused
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/fused/paro_silu.hip` — silu+rotate fusion
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1151/__init__.py` — gfx1151 reuses gfx1100 sources verbatim
- `/tmp/hipengine-survey/docs/ROOFLINE-gfx1151.md:182-292` — bandwidth/compute roof model
- `/tmp/hipengine-survey/docs/OPTIMIZE.md:245-285` — MoE decode Amdahl table (D1.x rejected
  fusions; D2.1 Marlin-K accepted)
- `/tmp/hipengine-survey/docs/OPTIMIZE-DENSE.md:126-179` — "why MoE wins and dense lags" +
  GDN chunkwise lane

## Files of interest (hipfire, this branch)

- `/home/bjoern/hipfire/.worktrees/paroquant-a3b/crates/hipfire-arch-qwen35/src/qwen35.rs:3765-3825` — MoE graph disable
- `/home/bjoern/hipfire/.worktrees/paroquant-a3b/crates/hipfire-arch-qwen35/src/qwen35.rs:2685-2974` — MoE decode hot path
- `/home/bjoern/hipfire/.worktrees/paroquant-a3b/crates/hipfire-arch-qwen35/src/qwen35.rs:4426-5243` — MoE prefill (Path 2 grouped-WMMA)
- `/home/bjoern/hipfire/.worktrees/paroquant-a3b/kernels/src/gated_delta_net_q8.hip:56-86` — GDN serial-per-token loop
- `/home/bjoern/hipfire/.worktrees/paroquant-a3b/kernels/src/gemm_hfq4g256_moe_grouped_wmma_k2.hip:43-100` — grouped-MoE WMMA
- `/home/bjoern/hipfire/.worktrees/paroquant-a3b/crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:1-200` — bench config
