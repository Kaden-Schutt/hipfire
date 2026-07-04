# Oq4 → NPU: int4-to-register staging (buildable design)

> **PIVOT (2026-07-04): use AMD IRON, not hand-wired whole_array.** IRON
> (`third_party/IRON`) has the exact building blocks: an `operators/dequant`
> that consumes `int4-packed + per-group scale` (= Oq4G256's layout, `group_size`
> configurable to 256), an `operators/gemm`, operator **fusion**
> (`FusedMLIROperator`), and Python host-buffer declaration (`AIERuntimeArgSpec`)
> — which dissolves the `test.cpp` buffer-size friction below. **Both the gemm
> and the int4 dequant operators RUN on this NPU** (confirmed). The hand-kernel
> `oq4_dequant_mm.cc` + whole_array route below is now REFERENCE for the in-core
> mechanics; the buildable path is IRON.
>
> **Working IRON setup on halo (hard-won — versions matter):**
> - venv is **Python 3.14** (system `pyxrt` is cp314-only; py3.12 can't load it).
> - install IRON's **pinned** `mlir_aie==0.0.1.2026033104+e4f35d6` (has
>   `aie.iron.placers`; the newer 886d932 REMOVED it → `No module named
>   'aie.iron.placers'`) + `llvm-aie==...2026062201` via `--find-links` (the
>   GitHub `expanded_assets` pages are flat wheel lists, NOT pip indexes — using
>   `--extra-index-url` gives "wrong package metadata" errors) + CPU torch + `-e .`.
> - run env: activate venv, `PEANO_INSTALL_DIR=<venv>/.../llvm-aie`,
>   `source /opt/xilinx/xrt/setup.sh`, **`PYTHONPATH=/opt/xilinx/xrt/python`**
>   (system pyxrt). Then `pytest iron/operators/gemm/test.py -s`.
> - The py3.14 mlir_aie native-binding segfault I hit earlier (conv `test.py`)
>   does NOT recur with the March wheel + IRON's runtime — it runs clean.
>
> **(b) IN PROGRESS — fused `dequant(gs=256,int4→bf16)`→`gemm` (drivers in `iron/`):**
> - `iron/oq4_dequant_run.py`: IRON dequant at **group_size=256 (Oq4)** RUNS +
>   VERIFIES on NPU (size 262144, 8×2 cores, 159.5µs, errors=False). Oq4
>   weight-decode confirmed on hardware.
> - `iron/oq4_fused_gemm.py`: fused runlist `[(dequant,"Wp","Wdeq"),(gemm,"A","Wdeq","C")]`
>   via `FusedMLIROperator`. Fusion machinery WORKS (compiles, reaches runtime
>   buffer validation). Two IRON gaps found:
>   1. **FIXED (local patch to IRON):** dequant `design.py my_dequant_kernel`
>      lacked the `func_prefix` kwarg that fusion passes to namespace child-op
>      functions (gemm has it). Added `func_prefix=""` + prefixed the Kernel
>      name/object. → gets past fusion into buffer validation. (Submodule edit;
>      upstream-worthy.)
>   2. **BLOCKER:** `FusedMLIROperator` assumes a UNIFORM input dtype, so it types
>      the int4/uint8 `Wp` buffer as 2-byte (bf16) → size check halves it
>      (expects 133120 B, sees my 133120-elem uint8 as 66560). Oq4 mixes
>      uint8-packed weights + bf16 activations; fusion needs per-input dtype.
>      NEXT: teach the fused op the Wp input is uint8 (buffer_sizes/dtype decl or
>      a small FusedMLIROperator patch), then the fused Oq4 gemm runs → first
>      feed-win number vs 15.7.
> FWHT stays an activation-side op. Template: llama_3.2_1b/llama_npu.py.
> Run env: the py3.14 iron venv recipe above.


Concrete design for feeding Oq4G256 weights to the Strix Halo NPU (aie2p) as
**int4 all the way to the register file**, dequantizing in-core on the surplus
compute. Grafts onto the mlir-aie `whole_array` matmul + `aie_kernels/aie2p/mm.cc`
so it's measurable through `tune.sh`.

Rationale: the NPU is feed-bound (npuclk 1267 < hclk 1800; TOPS = compute-clock ×
cols, ignores feed). Moving int4 weights instead of int8/bf16 cuts the scarce
off-chip (shim, DDR→memtile) traffic ~2–4×. See `findings.md` + docs/npu/NPU-RESULTS.md.

## Oq4G256 on-disk (from hipfire-quant-format)
Per 256-weight K-group: `[f16 scale : 2 B][128 B nibbles = 256 signed-int4]` =
130 B/group (0.508 B/wt). Symmetric signed int4 (`sext4`). Weights **pre-FWHT'd
offline** (256-pt Hadamard along K); runtime FWHT-rotates the activation `x`
(`mq_rotate_x`) so the rotations cancel.

## Deltas vs stock whole_array (i8/i8)
| Piece | Stock | Oq4 design |
|---|---|---|
| B (weight) objectFIFO element type | `int8` tile | **`int8` packed 2-per-byte** (int4), half the bytes |
| Scale stream | none | **new objectFIFO**: 1 `f16`/256-group, memtile→core |
| Memtile role | double-buffer passthrough | **weight-stationary**: hold column's int4 slab, reuse across all M |
| Microkernel | `matmul_vectorized_8x8x8_i8_i8` | **`oq4_dequant_mm`**: unpack int4→int8, ×scale, then the same 2×2 mmul |
| Activation | int8 direct | **FWHT-256 per K-group** then quantize (phase 2) |

## Phasing (each independently measurable via tune.sh)

**P1 — feed win, no correctness (measure the hypothesis first).**
B fed as int4 (half bytes across shim+mem DMA), unpacked to int8 in-core, stock
int8 mmul, **no scale, no FWHT**. Run with `--verify false`; compare TOPS to the
stock i8/i8 best (15.7). If int4-weight-feed lifts throughput on the feed-bound
engine, the thesis holds.

STATUS: **kernel done + verified** — `kernels/oq4_dequant_mm.cc` compiles clean to
aie2p via Peano (int4 nibble-unpack via `downshift`/`interleave_zip`/`concat` →
stock 2×2 `aie::mmul<8,8,8,int8>`; 23 vector insns). Remaining = wire an int4-B
`whole_array` variant. The 4 edits that must move together (B is now K·N/2 bytes):
1. `whole_array.py` B objectFIFO: `B_l2_ty`/`B_l1_ty` = `(k*n//2,) int8`; keep the
   8×8 re-tile dims but on half the bytes.
2. `whole_array.py` `external_func` matmul → `oq4_dequant_mm_i4_i8` (sig
   `a:int8, b:int8[packed], c:int8`), `link_with` the oq4 `.o`.
3. Runtime B `npu_dma_memcpy_nd` sizes/strides → half (int4 slab).
4. **Host `test.cpp` B buffer** (framework-generated, assumes K·N int8) → allocate
   K·N/2 and skip B verify. This is the real friction: the makefile-common host
   is shared; needs a `test.cpp` override or a `b_bytes` knob. `--verify false`
   sidesteps *correctness* but not the *allocation* size.
Then build via a new design `.py` + kernel `.o` rule and run at `m128 k256 n128`,
`--verify false`, TOPS vs 15.7.

**P2 — correctness (W4A16-on-NPU).** Add the per-group f16 scale multiply and the
activation 256-pt FWHT butterfly (once per K-group, shared across N). `--verify`
against a dequant+FWHT reference. This is a usable Oq4 decode kernel.

**P3 — W4A4 / native iu4.** Quantize the FWHT'd activation to int4 in-core and use
a native `iu4×iu4` mmul if aie2p exposes it (Peano intrinsic; not in high-level
aie_api — verify). Doubles compute *and* keeps feed minimal.

## objectFIFO wiring (whole_array.py changes)
```python
# B is now int4-packed: half the L2/L1 element bytes. n*k int4 = n*k/2 bytes.
B_l2_ty  = np.ndarray[(k * n // 2,), np.int8]        # packed nibbles
B_l1_ty  = np.ndarray[(k * n // 2,), np.int8]
# scales: one f16 per 256-group; K-groups = k/256 per tile (k>=256)
S_l2l1   = object_fifo("S_L2L1_%d"%col, mem, cores, fifo_depth,
                       np.ndarray[(k//256 * n,), np.dtype[bf16]])
# B_l3l2 (DDR->memtile): CONTIGUOUS int4 slab (pre-tiled offline) -> no dims
# B_l2l1 (memtile->L1):  same 8x8 re-tile dims as stock but on HALF the bytes
# Memtile residency: raise the B objectFIFO producer depth so the column's
#   int4 slab stays resident and is re-fed to L1 across the M loop (weights read
#   from DDR once). This is the biggest prefill lever.
```
Route unchanged: N split across 8 cols, weight slab broadcast down the 4 rows;
A broadcast across cols, M split across rows. Scales follow B's column split.

## In-core kernel: `oq4_dequant_mm.cc` (sketch in this dir)
Wraps the real `matmul_vectorized_2x2_mmul` from `aie2p/mm.cc`; the only change is
the B load path: load 32 packed bytes → unpack to 64 `int8` (sext4) → (P2: ×scale)
→ feed the existing `MMUL::mac`. A/C paths untouched.

## Build / measure
Extend `tune.sh` with a `KERNEL=oq4_dequant_mm` + int4 B objectFIFO variant of
the design (new `.py`), keep the same `M K N m k n cols` sweep. Key configs:
- k must be a multiple of 256 (one Oq4 group); start `m128 k256 n128` (fits L1 in
  int4) — note int4 weights let the **tile grow** vs int8 (feed + L1 both win).
- P1 metric: TOPS vs 15.7 at matched shape, `--verify false`.
- Watch: decode (unpack+scale+FWHT) must stay ahead of the mmul — if TOPS drops,
  decode is the new bottleneck (unlikely at 27% feed-idle, but measure).

## Open verifications
1. Native `iu4` mmul on aie2p (Peano intrinsic)? If yes → P3 W4A4 = 2× compute.
2. Memtile capacity for a useful resident int4 weight slab (512 KB/col).
3. FWHT-256 butterfly cost in-core vs its once-per-group amortization.
