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
