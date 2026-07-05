#!/usr/bin/env python3
# Generate a STREAMING R5 cascade-array MLIR — the payoff-measurement design.
#
# Same spatial K-cascade as r5_gen.py (COLS independent columns, each a ROWS-deep
# K-cascade, head north / tail south, cascade flowing north->south), but each
# dispatch now streams N_BTILES output tiles through the array instead of one.
# The persistent (INF) cores process one (A,W[,C]) tile per loop iteration exactly
# as before, so the kernel r5_cascade.cc is UNCHANGED; only the runtime_sequence
# feeds N_BTILES tiles of real bytes (A/W in, C out) per dispatch. That amortizes
# the fixed per-dispatch overhead (ERT command submit, hwctx sync, DMA task
# setup/await — the ~180us floor in ../../../docs/npu/NPU-RESULTS.md) across
# N_BTILES tiles, so the steady-state throughput reveals the true cascade-dataflow
# TOPS. C never round-trips memory — the cascade carries it core->core and the tail
# stores it once per tile (no per-tile C reload); that property is what a memtile
# dataflow cannot do and is the whole bet.
#
# Every core still sees all-ones A/W (broadcast), so the MAC count is real but the
# data is trivial — a throughput measurement, not a distinct-K GEMM. Each column's
# C tile = ROWS * KSLICE * 16 (each core's KSLICE-mmul partial is KSLICE*16, summed
# over ROWS by the cascade), identical for every one of the N_BTILES tiles.
#
# Usage: r5_stream_gen.py COLS ROWS N_BTILES > r5_stream.mlir  (build: r5_build.sh)
#
# Feed the built dir to npu_gemm_bench with:
#   asz  = N_BTILES * 1024          (A: KSLICE*size_A bytes/tile)
#   wsz  = N_BTILES * 2048          (W: KSLICE*(size_B/2) bytes/tile)
#   csz  = COLS * N_BTILES * 256    (C: size_C*4 bytes/tile, all columns)
#   macs = COLS * ROWS * KSLICE * 1024 * N_BTILES
#   expect_c0 = ROWS * KSLICE * 16
# (r5_stream.sh computes these and sweeps N_BTILES for you.)
import os, sys

# R5_ARCH: aie2 = XDNA1/Phoenix/npu1 (gfx1103; 4x4, int8xint4 <4,16,8>);
# aie2p = Strix Halo/npu2 (default; 8x4, int8xint4 <4,16,16>). Build with the same R5_ARCH.
ARCH = os.environ.get("R5_ARCH", "aie2p")
if ARCH == "aie2":
    DEV, MAXCOL, AW, WW, CW = "npu1", 4, 1024, 1024, 32   # <4,16,8>: size_A=64, size_B/2=64B, size_C=32
elif ARCH == "aie2p":
    DEV, MAXCOL, AW, WW, CW = "npu2", 8, 1024, 2048, 64   # <4,16,16>: size_A=64, size_B/2=128B, size_C=64
else:
    sys.exit(f"unknown R5_ARCH={ARCH} (want aie2 or aie2p)")

COLS = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, MAXCOL)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 4   # cascade depth per column
NBT  = int(sys.argv[3]) if len(sys.argv) > 3 else 32  # output tiles streamed per dispatch
if COLS > MAXCOL:
    sys.exit(f"COLS={COLS} exceeds {ARCH} column count {MAXCOL}")
if ROWS > 4:
    sys.exit(f"ROWS={ROWS} exceeds 4 compute rows")

rows = list(range(2, 2 + ROWS))          # bottom..top tile rows
top = rows[-1]                            # head (northernmost)
INF = 9223372036854775807

def core_body(cid, name, tile, kern, has_c):  # cid = unique core SSA id; name = column (fifo) id
    c_acq = f'''
        %c = aie.objectfifo.acquire @fc{name}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>''' if has_c else ""
    call = (f"func.call @{kern}(%av, %wv, %cv) : (memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()"
            if has_c else
            f"func.call @{kern}(%av, %wv) : (memref<{AW}xi8>, memref<{WW}xi8>) -> ()")
    c_rel = f"\n        aie.objectfifo.release @fc{name}(Produce, 1)" if has_c else ""
    # Persistent core: INF loop consumes one streamed tile per iteration. A dispatch
    # that feeds N_BTILES tiles into the fifos drives exactly N_BTILES iterations.
    return f'''    %core_{cid} = aie.core({tile}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {{
        %a = aie.objectfifo.acquire @fa{name}(Consume, 1) : !aie.objectfifosubview<memref<{AW}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AW}xi8>> -> memref<{AW}xi8>
        %w = aie.objectfifo.acquire @fw{name}(Consume, 1) : !aie.objectfifosubview<memref<{WW}xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>{c_acq}
        {call}
        aie.objectfifo.release @fa{name}(Consume, 1)
        aie.objectfifo.release @fw{name}(Consume, 1){c_rel}
      }}
      aie.end
    }}'''

out = ["module {", f"  aie.device({DEV}) {{"]
# tiles
for c in range(COLS):
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    for r in rows:
        out.append(f"    %t{c}_{r} = aie.tile({c}, {r})")
# cascade flows: north(r+1) -> south(r) within each column
for c in range(COLS):
    for r in rows[:-1]:
        out.append(f"    aie.cascade_flow(%t{c}_{r+1}, %t{c}_{r})")
# objectfifos: broadcast A/W to all rows in a column; C from the tail (row 2).
# Depth 2 double-buffers so the streamed DMA overlaps compute (the pipelining that
# makes N_BTILES>1 amortize overhead rather than just serialize it).
cons = lambda c: "{" + ", ".join(f"%t{c}_{r}" for r in rows) + "}"
for c in range(COLS):
    out.append(f"    aie.objectfifo @fa{c}(%shim{c}, {cons(c)}, 2 : i32) : !aie.objectfifo<memref<{AW}xi8>>")
    out.append(f"    aie.objectfifo @fw{c}(%shim{c}, {cons(c)}, 2 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
    out.append(f"    aie.objectfifo @fc{c}(%t{c}_2, {{%shim{c}}}, 2 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
# kernel decls (unchanged r5_cascade.cc roles)
out.append(f'    func.func private @r5_cascade_head(memref<{AW}xi8>, memref<{WW}xi8>) attributes {{link_with = "r5_head.o"}}')
out.append(f'    func.func private @r5_cascade_mid(memref<{AW}xi8>, memref<{WW}xi8>) attributes {{link_with = "r5_mid.o"}}')
out.append(f'    func.func private @r5_cascade_tail(memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) attributes {{link_with = "r5_tail.o"}}')
# cores
for c in range(COLS):
    for r in rows:
        if r == top:
            out.append(core_body(f"{c}_{r}", f"{c}", f"%t{c}_{r}", "r5_cascade_head", False))
        elif r == 2:
            out.append(core_body(f"{c}_{r}", f"{c}", f"%t{c}_{r}", "r5_cascade_tail", True))
        else:
            out.append(core_body(f"{c}_{r}", f"{c}", f"%t{c}_{r}", "r5_cascade_mid", False))
# runtime sequence: stream N_BTILES tiles per dispatch. A is N_BTILES*AW, W is
# N_BTILES*WW (real bytes moved — this is a FEED-bound measurement, so no stride-0
# replay tricks), C is COLS*N_BTILES*CW. Each DMA's 3rd wrap dim repeats the per-tile
# block N_BTILES times (<size=N_BTILES, stride=tile_elems>).
AT, WT, CT = NBT * AW, NBT * WW, NBT * CW           # per-column A/W/C element counts
args = ", ".join([f"%A: memref<{AT}xi8>", f"%W: memref<{WT}xi8>", f"%C: memref<{COLS*CT}xi32>"])
out.append(f"    aie.runtime_sequence({args}) {{")
for c in range(COLS):
    out.append(f'''      %ta{c} = aiex.dma_configure_task_for @fa{c} {{
        aie.dma_bd(%A : memref<{AT}xi8>, 0, {AT}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBT}, stride = {AW}>, <size = {AW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{c})
      %tw{c} = aiex.dma_configure_task_for @fw{c} {{
        aie.dma_bd(%W : memref<{WT}xi8>, 0, {WT}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBT}, stride = {WW}>, <size = {WW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw{c})
      %tc{c} = aiex.dma_configure_task_for @fc{c} {{
        aie.dma_bd(%C : memref<{COLS*CT}xi32>, {c*CT}, {CT}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBT}, stride = {CW}>, <size = {CW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{c})''')
for c in range(COLS):
    out.append(f"      aiex.dma_await_task(%tc{c})")
for c in range(COLS):
    out.append(f"      aiex.dma_free_task(%ta{c})")
    out.append(f"      aiex.dma_free_task(%tw{c})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
