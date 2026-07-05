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

## Streaming payoff measurement — DONE, and it's a NO-GO on XDNA1 (2026-07-05)

Retargeted the whole R5 line from aie2p/Strix Halo to **aie2 / XDNA1 (gfx1103, the
nix2 Phoenix NPU)** — build + measure now run locally, no halo round-trip. Arch is
selected by `R5_ARCH={aie2,aie2p}` across `r5_cascade.cc` (kernel), `r5_build.sh`,
`r5_gen.py`, and `r5_stream_gen.py`; aie2p stays the default so the Halo repro is
untouched. XDNA1 deltas: int8×int4 mmul is only `<4,16,8>` (size_C=32, 2 cascade
beats) vs aie2p's `<4,16,16>`; the cascade read is `get_scd_v16int32` (no
`get_scd_v16acc32` builtin) so we sum in the int32 domain; device is `npu1`
(4 cols × 4 compute rows = **16-core max array**). Build needs XRT's `xclbinutil` +
a user-space boost-1.83 on `LD_LIBRARY_PATH` (both auto-added by `r5_build.sh`).

Re-validated the mechanism on XDNA1 hardware: 2-core `C[0]=512`, full **4×4=16-core**
`C[0]=1024`, both linear under A=2 — the cascade K-split works on Phoenix too.

`r5_stream_gen.py` streams `N_BTILES` output tiles per dispatch (persistent cores,
one tile per objectfifo iteration; A/W fed as real bytes so it's a genuine feed-bound
measurement; C never reloads — the cascade carries it and the tail stores once).
`r5_stream.sh` sweeps N_BTILES, building one xclbin each and timing
`npu_gemm_bench`. Sweep on the 4×4 array (all-ones, `c0=1024` throughout):

| N_BTILES | 64 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| per-dispatch | 163 µs | 255 | 382 | 636 | 1114 µs |
| **TOPS** | 0.10 | 0.26 | 0.35 | 0.42 | **0.48** |

The fixed per-dispatch overhead (~140 µs ERT/hwctx/DMA floor) amortizes as designed,
but TOPS asymptotes at **~0.56 TOPS** (steady-state slope 0.466 µs/tile over
NBT 1024→2048) — **~10× below SOTA and ~20× below the array's compute peak.**

**Why (two discriminator sweeps, steady-state µs/tile):**

| config | cores | cascade | µs/tile | steady TOPS |
|---|---|---|---|---|
| 1×4 | 4 | 4-deep | 0.492 | 0.13 |
| 4×2 | 8 | 2-deep | 0.346 | 0.38 |
| 4×4 | 16 | 4-deep | 0.466 | 0.56 |

- **1×4 ≈ 4×4 per-tile** despite 4× the columns → columns parallelize for free; the
  per-tile cost is *per column*, not global shim feed. Column-scaling is ~linear
  (0.13→0.56 TOPS for 1→4 columns), but XDNA1 caps at 4 columns.
- **4×2 < 4×4 per-tile** → deeper cascade adds ~0.07 µs/row of chain-sync latency;
  cascade depth scales *sub-linearly*.

So the ceiling is set by **per-tile objectfifo-sync + cascade-beat transfer**, not
compute: the `<4,16,8>` tile is so small each core does ~8k MACs (~26 ns) wrapped in
~470 ns of sync. The cascade eliminates the C round-trip, but that never mattered
here — per-tile sync swamps the compute regardless. **Conclusion: the K-cascade axis
is the wrong axis on XDNA1 — it multiplies per-tile sync without growing the effective
output tile.** This extends the aie2p "feeding, not compute" thesis (`../findings.md`)
to Phoenix. The only levers that could help are the ones that raise compute-per-sync:
a within-kernel N-loop (many independent output columns per acquired W) or a
weight-stationary dataflow (W resident, stream only A) — *not* spatial K-splitting.
