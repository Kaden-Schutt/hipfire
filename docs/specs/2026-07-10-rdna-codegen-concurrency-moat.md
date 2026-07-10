# RDNA Codegen & Concurrency Moat — Design Spec

> Status: **DRAFT / research-backed strategy** · Date: 2026-07-10 · Arch focus: gfx1201 (RDNA4, R9700) · Owner: Kaden Schutt
>
> This spec captures an empirical investigation into *why* Vulkan/RADV inference
> backends outperform ROCm/HIP on RDNA, quantifies the two independent gaps, and
> proposes the durable moat: close them in our own codegen + dispatch layer so
> ACO-grade performance is the default, not the product of per-kernel hand-labor.

## 1. Thesis

Across the ecosystem (llama.cpp, mlc, etc.) **Vulkan/RADV backends beat ROCm/HIP
on RDNA**, and it is under-investigated *why*. We measured it directly on
gfx1201 and found **two independent, real, non-hand-waveable gaps**:

- **(A) Default codegen.** On an *identical* mq4 dequant-GEMV, RADV's **ACO**
  compiler is **~43% faster than hipcc/LLVM by default** in the latency-bound
  regime that real decode runs in. ACO gets it for free; hipfire only matches it
  via a 482-attempt-per-kernel autoresearch hand-tuning loop. That asymmetry —
  free vs. enormously expensive — is the leverage gap.
- **(B) Launch-set concurrency.** A Vulkan **command buffer overlaps independent
  dispatches** (fills the GPU); ROCm **hipGraph serializes them** and adds
  per-node overhead. RADV is ~3.6× faster on an independent-kernel launch set.

Both are **greenfield** on the HIP side — the CUDA-edge equivalents (ptxas
aggressive scheduling + CUTLASS/cuDNN templated kernels + graph/stream overlap)
have no built-out counterpart in the AMD ML stack. Building them is a defensible
moat: a competitor (ZINC, rocmfp4, stock RADV engines) cannot match hipfire
without reproducing our codegen and dispatch layers.

**Non-goal / hard constraint:** we do **not** ship a Vulkan/RADV backend
(issue #44, closed). We *study* ACO's codegen to lift the *mechanism* into HIP.
AR-only decode; no spec-decode; mqN quant only (no ggml/Q4_K import).

## 2. Motivation / observed gap

- ZINC does **166 tok/s AR** on gfx1201 via RADV/Vulkan with **Q4_K_M XL UD
  (~4.5–4.8 bpw)** — *more* bytes/token than our mq4r (~4.25 bpw) — while hipfire
  is at **162 tok/s**. Normalized for weight traffic, RADV's compute path is
  materially more efficient than the raw 162-vs-166 implies. The whole-model
  number *dilutes* the per-kernel gap because our kernels are already hand-tuned.
- The gap is **regime-dependent**, largest exactly where decode lives (latency-
  bound, small grids, mem_busy 40–55%), and it vanishes when memory-BW-saturated.

## 3. Empirical findings (evidence base)

All measured on hiptrx gfx1201 (R9700), free card (dev1) to avoid the sol-swarm
contention on dev0. Benchmark sources: `/tmp/acostudy` (k9lin) + `/tmp/*.{comp,c,hip}`
(hiptrx). Method: identical mq4 dequant-GEMV compiled via RADV/ACO (GLSL →
glslangValidator → Vulkan compute pipeline) and via hipcc/LLVM (HIP), GPU-timed
(Vulkan timestamp queries / hipEvent), min-over-N.

### 3.1 Default codegen: ACO +43% (latency-bound)

| variant (naive source, free to each compiler) | latency-bound M=6144 | DRAM-saturated M=65536 |
|---|---|---|
| naive ACO / RADV | **6.6 µs / 1015 GB/s** | ~765 GB/s |
| naive LLVM / hipcc | **9.4 µs / 713 GB/s** | ~753 GB/s |
| gap | **ACO +43%** | ~0 (BW-bound hides it) |

- The **+43% is the fair, un-confounded comparison** (naive-vs-naive; each compiler
  free to optimize; for ACO its naive *is* its optimum because it self-schedules).
- In the BW-saturated regime the gap disappears — all variants hit the memory roof.
  **Real decode is latency-bound** (small grids, single dispatch, mem_busy ~41%),
  i.e. the +43% regime, not the ~0% one.
- **Not flag-fixable:** `-mllvm -amdgpu-sched-strategy=max-ilp`, `-misched-postra`,
  `-O3 -ffast-math` all leave LLVM at ~9.4 µs.
- **Mechanism:** ACO emits RDNA4 **dual-issue** (`v_dual_*`) + pervasive
  **`s_delay_alu`** dependency-distance scheduling (13× in the dumped GEMV) and a
  deep loads-ahead window (pre-sched VGPR 14 → post-sched 96). LLVM's default
  scheduler does not reach this. Same ISA — LLVM *can* emit it, it *doesn't*.

### 3.2 The "tuned ACO" confound (corrected)

An earlier reading ("per-kernel is a wash / tuned-LLVM ≥ ACO") was **wrong**.
Imposing our explicit hoist+preload structure on the GLSL *handicapped* ACO —
it fought ACO's auto-scheduler and regressed it **below** naive ACO
(tuned-ACO 744 < naive-ACO 765 GB/s DRAM; 662 vs 600 hand-tuned-LLVM-beats-only-
that-handicap). ACO auto-tunes; the only fair "which compiler is better" is
naive-vs-naive → **ACO wins**. hipfire's hand-tuned LLVM can *edge* ACO's naive,
but only after ~500 hand-tuning iterations — which is the cost, not a win.

### 3.3 The 482-attempt cost (source campaign plateau)

The gfx1201 autoresearch ledger over the whole campaign: **482 logged attempts,
75 WINs (many are NOISE-tier phantoms), 310 DEAD.** Best individual deltas have
decayed to **+1–3.78%** (early rollovers were +4–15% composed). The high-wall
kernels are saturated: `fused_qkvza` 121 attempts → 1 durable win (+3.78%),
`moe_gate_up` 103 attempts → 1 win. The DEAD-verdict learnings are uniform:
"TLP already hides latency / not MLP-limited / occupancy structurally
catastrophic / register pressure is not the limiter." **The source-level lever
space is tapped** — consistent with the codegen gap being sub-source-level.

### 3.4 Launch-set concurrency: RADV overlaps, hipGraph doesn't

N independent tuned GEMVs, both **pre-recorded** (Vulkan command buffer vs HIP
hipGraph), GPU-execution-time (per-launch host overhead amortized), dev1:

| N=8 | SET µs | overlap factor | effBW |
|---|---|---|---|
| RADV command buffer | **43.8** | **1.29×** (per-dispatch drops in-set) | 1222 GB/s |
| HIP hipGraph | **157.7** | **0.77×** (worse than serial + per-node overhead) | 339 GB/s |

- **RADV overlaps independent dispatches** to fill the GPU. **ROCm hipGraph
  serializes independent nodes** and adds per-node overhead — a concrete,
  nameable ROCm limitation. RADV is ~3.6× on the set.
- Caveat: the microbench is 8 *independent* GEMVs; hipfire's real decode is mostly
  *dependency-chained* (gate_up→silu→down) with experts already fused into
  `gate_up_k8`, so there is limited independent parallelism to exploit *today*.

### 3.5 What is NOT the gap (ruled out)

- **Dispatch/megakernel fusion.** A fused cooperative megakernel to collapse the
  122-dispatch/token gap is a **dead end** — bit-exact but a perf *trap*: hipGraph
  makes a cooperative kernel ~2× *slower* (per-launch full-device drain).
  See `project_coop_kernel_graph_capture_gfx1201_2026_07_10`.
- **Occupancy.** ACO uses *more* VGPR than LLVM; our kernels are already
  higher-occupancy. Every occupancy-raising lever measured DEAD. Not the limiter.
- **Per-dispatch host submission (redline's original framing).** Paid once per
  token under a graph → negligible. Redline's real value is §5.2 concurrency.

## 4. Problem statement

hipfire's decode kernels are **memory-latency-bound**, and the two things that
would raise sustained memory throughput in that regime — (A) ACO-grade
instruction scheduling within a wave, and (B) overlapping independent dispatches
across the pipeline — are **both absent by default** in the ROCm/HIP path. We
currently substitute (A) with an expensive per-kernel hand-tuning loop and get
none of (B). The moat is to make both the *default* of hipfire's own codegen +
dispatch layer.

## 5. Proposed design

Two independent workstreams (A and B), each phased cheapest-first.

### 5.1 Moat A — ACO-grade default codegen

**A0 — Validate on the real path (prerequisite).** Confirm the +43% microbench
result translates to a real decode kernel: build a hand-scheduled variant of one
hot GEMV (`fused_qkvza` or `gate_up`) and certify it on the real serve path. If
the real-kernel delta is materially below 43%, re-scope; if it holds, proceed.

**A1 — Inline-asm / intrinsic templates (near-term, CUTLASS/cuDNN move).**
Author *one* hand-scheduled hot dequant-GEMV inner loop — RDNA4 dual-issue via
`__builtin`/inline-asm ordering, hand-placed `s_delay_alu`, deep loads-ahead
(prefetch depth 2–3, double/triple-buffered weights) — as a **reusable template
macro** parameterized by group stride / quant format. Apply it across the whole
`gemv_hfq*` family from the single template. This replaces the 482-attempt-per-
kernel loop with one-template-many-kernels: ACO-grade codegen without brute force.
- Deliverable: a `kernels/src/templates/` header of scheduled GEMV primitives.
- Gate: each derived kernel is bit-exact (parity) and certified perf-positive on
  the real serve path; gfx12-gated so gfx11/gfx10 machine code is byte-unchanged.

**A2 — Forked / patched LLVM AMDGPU scheduler pass (deeper, true "make it
default").** A codegen pass that emits ACO-grade dual-issue + `s_delay_alu` +
loads-ahead scheduling for *all* gfx1201 kernels by default, so no per-kernel
hand-work is needed. This is the durable ceiling and the hard-to-copy moat.
- Investigate first whether it lives as an out-of-tree LLVM pass loaded via
  `-fpass-plugin` / `-mllvm` vs. a compiler.rs JIT-flag surface vs. a genuine
  hipcc fork. Prefer the least-invasive that reaches parity with A1's templates.
- This is the "redline compiler" idea, scoped concretely.

### 5.2 Moat B — Launch-set concurrency (redline dispatch)

**B0 — Expose independent parallelism.** Real decode is dependency-chained, so
first identify genuinely-independent kernel groups per layer. The obvious first
candidate: the **qkv projections vs the DeltaNet/LA-path projections** are
independent within a layer. (Also: experts as separate overlappable dispatches
instead of the fused `gate_up_k8`, if the fusion isn't already a net win.)

**B1 — Overlap them with a mechanism that works.** hipGraph provably serializes
independent nodes (0.77×), so use **multi-stream HIP** or a **redline direct
command buffer** to actually run them concurrently. Smallest real test: race the
two independent projection groups on two HIP streams vs single-stream/graph in a
real layer; measure whether layer mem_busy rises and time drops.

**B2 — Redline concurrency layer.** If B1 validates, build the direct-KMD /
command-buffer dispatch path (redline) that schedules the whole decode pipeline
with RADV-grade overlap — the concurrency ROCm's graph can't deliver.

## 6. Roadmap / sequencing

1. **A0** (validate +43% on a real kernel) — cheapest, gates everything in A.
2. **A1** (GEMV template) — highest near-term ROI; replaces the sol swarm's labor.
3. **B0 + B1** (expose + two-stream overlap test) — cheap, independent of A.
4. **A2** (scheduler pass) and **B2** (redline concurrency) — the durable moats,
   pursued once A1/B1 prove the ceilings are real on the serve path.

The autoresearch loop (sol) continues mining the source tail in parallel, but it
is explicitly the *interim/expensive* path; A1 is its scalable replacement.

## 7. Measurement discipline (mandatory)

- Compiler comparisons: **naive-vs-naive** for "which compiler is better";
  hand-tuned only to establish *our* ceiling. Never compare tuned-LLVM to a
  handicapped-ACO.
- **Latency-bound regime (small grid, M≈6144, weights cache-resident) is the
  decode-relevant one.** BW-saturated microbenches (huge grids) hide the gap.
- GPU-timestamped, min-over-N, on a **free card** (never share a card with the
  sol swarm); ±1% tight or it's contention.
- Real-path claims certified through **serve_harness** on the daemon, parity
  (bit-exact token-ids, FP32+`HIPFIRE_DETERMINISTIC=1`) + interleaved warm A/B
  (absolute baseline drifts 153–162/day on DPM/clock).

## 8. Risks / open questions

- **A2 feasibility:** can LLVM be made to emit ACO-grade scheduling via a pass, or
  does it need a deeper backend fork? Unknown until A2 is scoped.
- **B real-path yield:** decode's dependency-chaining may cap the independent
  parallelism available; the overlap win could be smaller than the 3.6× microbench.
- **Regime transfer:** the +43% is cache-resident-latency-bound; real decode reads
  fresh DRAM weights per token — A0 must confirm the delta survives that.
- **Maintenance:** a forked scheduler / inline-asm templates are a maintenance
  surface across ROCm versions; A1 templates are more portable than A2 forks.

## 9. Appendix — references

- `project_aco_vs_llvm_codegen_2026_07_10` (memory) — the +43% default gap + confound.
- `project_launchset_hipgraph_vs_radv_2026_07_10` (memory) — the concurrency gap.
- `project_roofline_mlp_campaign_2026_07_10` (memory) — the 482-attempt plateau + Lever 1.
- `project_coop_kernel_graph_capture_gfx1201_2026_07_10` (memory) — megakernel dead-end.
- `feedback_competitive_mining_hygiene_2026_07_09` (memory) — study codegen, don't ship Vulkan; lift mechanism, drop format.
- Benchmarks: `/tmp/acostudy` (k9lin), `/tmp/{bench_vk,bench_hip,bench_hip_tuned,bench_vk_set,bench_hip_graph}*` (hiptrx).
