# R5 — cascade / K-resident-C systolic W4A8 (the SOTA-beating bet)

## Why this is the prize

Three independent measurements agree that real NPU W4A8 prefill is stuck at
**~5 TOPS effective**, far below the array's ~40-TOPS compute capacity:

- My R4b tiny-tile real-GEMM (INNER=0): **5.25 TOPS**.
- The IRON/whole_array + DynamicDispatch reference dataflows: 15.7 / 7 TOPS (`../findings.md`).
- **FastFlowLM (current SOTA) production numbers** back-solve to ~5 TOPS:
  LFM2-1.2B prefill 2518 tok/s ÷ ~1.0 G MAC/tok = ~5.0 TOPS; LFM2-2.6B 1206 tok/s ÷
  ~2.4 G = ~5.8 TOPS. Decode is bandwidth-bound (~34–35 GB/s effective ≈ 64% of the
  measured 55 GB/s feed).

Everyone is on the **DMA-through-memtile** dataflow, which the fixed `mmul<4,16,16>`
int8×int4 shape (the only one aie2p provides) shoehorns into a tiny 16×16 output tile
with a **per-tile C load/store + objectfifo sync** every step — the overhead that
pins throughput to ~5–15.7 TOPS regardless of columns/depth (`../findings.md`).

The AIE hardware has two under-used features that break this:
1. **Per-tile scalar RISC** — runs address-gen/control in parallel with the 512-bit
   fixed-point vector core, so per-tile control overhead can overlap MACs.
2. **Cascade stream / inter-tile bus** — a direct 512-bit core→core accumulator path.

**The bet:** split K spatially down a column of cores; partial sums flow core→core
over the cascade stream; **C stays in the flowing accumulator and is stored only once
by the tail core** — no per-tile C reload. If it lands even at the 15.7 reference that
is ~3× SOTA prefill (~7.5k tok/s for a 1.2B); near the ~40 capacity it is ~8× SOTA and
beats the gfx1151 GPU, making concurrent prefill offload a real win. No one has claimed
this — the cascade stream sits unused in every shipped kernel.

## The cascade API (reverse-engineered — no examples exist in this install)

- Graph: `aie.cascade_flow(src_tile, dst_tile)` wires a directed core→core cascade;
  `aie.configure_cascade(tile, inDir, outDir)` sets the West/East ports. (Python:
  `aie/dialects/_aie_ops_gen.py`.)
- Kernel: cores take `input_cascade<acc>* ` / `output_cascade<acc>* ` and use
  `readincr` / `writeincr` on an `aie::accum<acc32, N>` (cascade width **512 bits**;
  `aie_api/adf/stream.hpp::cascade_stream_helper`). For int8×int4 the mmul accumulator
  is acc32.

## The blocker (next iteration's job)

**IRON's `@jit` / `ExternalFunction` cannot pass cascade stream args** (only ObjectFifo
ndarray types) and exposes no `cascade_flow`. So R5 must use the lower-level mlir-aie
flow (explicit `aie.tile`/`aie.core`/`aie.objectfifo` + `aie.cascade_flow`), which I
have **not** yet driven end-to-end to an xclbin+insts. The critical de-risking step —
exactly like the amdxdna dispatch work — is to first get a *minimal* explicit (non-IRON)
design built to xclbin+insts and dispatched through `NpuKernel`, before layering on the
cascade kernel. Only then is the cascade GEMM buildable/testable.

## Plan (incremental, correctness-gated)

1. **Establish the low-level build**: minimal explicit `aie.dialects` design (1 core,
   trivial kernel) → xclbin+insts → dispatch via NpuKernel, all-ones validated.
   *Build path confirmed*: construct the module with `aie.dialects.aie`/`aiex`, then
   compile with `aie.utils.compile.compile(module, …, xclbin_path=…)` (the same helper
   IRON's `@jit` calls — it shells out to `aiecc` with `--aie-generate-xclbin`; insts
   come out alongside). So the explicit flow is fully buildable without IRON; the open
   work is writing the module by hand (tiles, cores, objectfifos, `cascade_flow`).
2. **2-core K-cascade**: split K across 2 cores, cascade-accumulate, validate all-ones
   equals the single-core result (the cascade sum is correct).
3. **4-core column** (full K-split down a column), then **8 columns** = 32 cores.
4. Measure real (INNER=0) TOPS through NpuKernel vs the 5-TOPS SOTA / 15.7 reference.

`r5_cascade.cc` holds the compute-core skeleton (head/middle/tail variants).

## Progress — both hard unknowns cleared (2026-07-05)

**1. Low-level build path PROVEN.** Compiled a known-good `aie.mlir` (from a cache
dir) *directly* via `aiecc` — bypassing IRON entirely — to xclbin+insts, and
dispatched it through `NpuKernel`: `C[0]=1040`, clean 2× reuse. So I control the MLIR
end-to-end. Recipe:
```
aiecc <dir>/aie.mlir --no-compile-host --no-xchesscc --no-xbridge --peano=$PEANO \
  --aie-generate-npu-insts --npu-insts-name=<dir>/insts.bin \
  --aie-generate-xclbin --xclbin-name=<dir>/final.xclbin --tmpdir=<dir>
```
Kernel `.o`: `$PEANO/bin/clang++ src.cc -c -o out.o -I$MLIR_AIE_INC -std=c++20 -O2 \
  -DNDEBUG --target=aie2p-none-unknown-elf [-DROLE=… -DKSLICE=…]` (drop into `<dir>`
so the MLIR's `link_with` resolves).

**2. Cascade API validated — kernel compiles.** The real API (not the ADF
`adf/stream.hpp` I first guessed) is the raw aie2p intrinsics from
`aie_kernels/aie2/cascade_mm.cc`: `put_mcd(v16acc32)` / `get_scd_v16acc32()`, one
512-bit beat = 16 acc32, so the `mmul<4,16,16>` 64-acc32 partial = 4 beats. `r5_cascade.cc`
(head/middle/tail) now **compiles for all three roles** to `.o` on the aie2p target,
using `aie::accum::extract<16>(i).to_native()` → `put_mcd`, and `get_scd_v16acc32()`
→ `aie::accum::insert`.

**3. 2-core K-cascade WORKS on hardware.** `r5_2core.mlir` places two adjacent cores
in column 0 (head=(0,3) north of tail=(0,2) — cascade source must be North/West of
dest), `aie.cascade_flow(head, tail)`, broadcasts all-ones A/W to both, and drains C
from the tail. Built via `r5_build.sh` and dispatched through NpuKernel:
**`C[0]=512`** — each core's KSLICE=16 partial is 256, and the cascade summed them
core-to-core (256 would mean the cascade dropped the head). dispatch2 (A=2) = 1024,
confirming linearity. **First working cascade GEMM through hipfire — the K-resident-C
systolic dataflow no shipped kernel uses is validated end-to-end.**

`r5_build.sh <mlir> <workdir> [KSLICE]` builds any R5 cascade design (compiles the
head/mid/tail objects, runs aiecc) → `final.xclbin` + `insts.bin`, reproducibly.

**4. 4-core column + FULL 32-core array validated on hardware.** `r5_4core.mlir`
(head + 2 `mid` + tail down column 0) → `C[0]=1024` (4×256, summed through the two
middle get+add+put cores). `r5_gen.py COLS ROWS` emits an arbitrary cascade array;
the generated **8×4 = 32-core** design builds and dispatches with **all 8 column C
blocks = 1024** — the cascade dataflow works at full array scale. So the mechanism is
proven end-to-end from 2 → 4 → 32 cores.

**5. First compute-rate measurement: ~4 TOPS — a per-core stall to diagnose.**
Added an `INNER` reuse knob to `r5_cascade.cc` (four register-resident K-tiles
reused INNER times — the II=1 recipe, pure register macs). The 32-core array with
INNER=16384 is bit-exact (C[0]=4,194,560) but sustains only **~4 TOPS** — ~12× below
an r4b core. Key clue: **1 accumulator (4.44) ≈ 4 accumulators (4.03)**, so it is
*not* mac-pipeline/II or accumulator spill; each cascade core is individually slow
regardless. **Diagnostic run (ROLE=3 standalone, cascade op removed, single core) isolates TWO
separable slowdowns:**

| kernel | TOPS/core | vs r2a |
|---|---|---|
| r2a_mac (r4b reference) | ~1.6 | 1× |
| r5 standalone, NO cascade | **0.40** | **~4× slow** |
| r5 in 32-core cascade array | **0.126** | ~13× slow |

So (a) the **base kslice_partial is ~4× slower than r2a even without any cascade**,
and (b) the **cascade adds a further ~3×** on top. Both are tunable:
- Kernel (4×): r2a keeps 4 A-tiles + **one shared** W-tile in registers; r5 loads 4
  A **and 4 W** tiles + does an accum-sum at the end → register pressure/spill. Fix:
  share the weight tile across the 4 accumulators (or reduce live tiles), drop the
  end-sum onto fewer accumulators.
- Cascade (3×): the 4 cores in a column aren't overlapping — likely the cascade
  FIFO depth / the get-then-compute ordering serializing them. Fix: deepen the
  cascade path, and ensure compute is issued *before* the blocking `get_scd`.

The cascade *mechanism* is fully validated (2/4/32 cores, exact sums); reaching a
SOTA-beating number is now a two-part **perf-tuning** problem on a working dataflow,
not a correctness one.

**6. KERNEL FIX → 42 TOPS: the cascade reaches full compute capacity.** The ~4× base
slowdown was register pressure — the kernel now uses R2a's exact recipe: 4 M-block
accumulators sharing **one** resident weight tile (`compute()` in `r5_cascade.cc`),
and the cascade carries all four accumulators (16 beats) core→core. Re-measured, all
bit-exact:

| | TOPS | vs before |
|---|---|---|
| 32-core cascade, INNER=16384 | **25.7** | (was 4.03) |
| 32-core cascade, INNER=65536 | **42.2** | overhead amortized |

The kernel fix alone was a **10×** jump, and it revealed that the earlier "cascade
adds 3×" was an *artifact of the slow kernel* — with a fast kernel the cascade is
nearly free (42 TOPS ≈ R4b's 40-TOPS fake-reuse ceiling). So the cascade dataflow
**accesses the array's full compute capacity while keeping C resident** — exactly
the property the memtile dataflow lacks. vs SOTA FastFlowLM's ~5 TOPS and the 15.7
reference, this is the compute path that was missing.

**7. Real streaming (INNER=0) measured — and the honest verdict: the cascade does
NOT help real prefill.** Built the streaming-cascade (A resident, W streamed over an
`NB` inner loop, one cascade round per output tile; `r5_gen.py COLS ROWS NB`). At
NB=4096, M=16 (4 accumulators × MR=4), bit-exact:

| dataflow, real INNER=0 (M=16) | TOPS |
|---|---|
| **R5 streaming cascade** | **3.22** (feed-bound) |
| R4b independent cores | 5.25 |
| SOTA FastFlowLM (real) | ~5 |

The cascade is *worse* than the simple independent-core dataflow here (3.22 < 5.25).
**Why:** real prefill GEMM at feasible M is **feed / arithmetic-intensity-bound**, not
C-store-overhead-bound. AI = M·1024 / weight-bytes; at M=16, AI≈32 MACs/B → the 16 MB
weight stream, not compute, sets the rate. The cascade's win (C resident, no per-tile
reload) only matters in the *overhead* regime (tiny tiles) — it does nothing for the
feed, and its extra 16-beat transfer + reduced output parallelism (8 columns vs 32
independent cores) make it slightly *worse* in the feed-bound regime.

**Conclusion — the cascade was the wrong lever for real prefill:**
- The cascade genuinely unlocks the **compute ceiling** (42 TOPS, fake reuse) —
  proving the silicon *can* do it, and that the ~5-TOPS "wall" is a dataflow limit.
- But **real streaming prefill is feed/AI-bound**, and there the lever is **M
  (weight reuse), not K-depth**. The cascade adds K-depth, which isn't the
  bottleneck. R4b's plain independent-core dataflow already ≈ **matches SOTA** (5.25
  vs ~5) at M=16 with *zero* cascade complexity.
- **Beating SOTA needs higher M** — more weight reuse per streamed byte (more
  accumulators / bigger MR / A-resident across more output rows), pushing AI past
  ~58 MACs/B toward the compute ceiling. That, not the cascade, is the real R6.

A clean negative result (per AGENTS.md, it narrows the search): the cascade is a
validated, working systolic dataflow that reaches full compute capacity, but it does
not help the actual real-prefill bottleneck. The next lever is M/AI, not K.
