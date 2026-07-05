#!/usr/bin/env python3
# Generate an R5 cascade-array MLIR: COLS independent columns, each a ROWS-deep
# K-cascade (head at the top row, tail at row 2, cascade flowing north->south).
# All-ones A/W are broadcast to every core in a column, so each column's C block =
# ROWS * KSLICE * 16 (each core's KSLICE-mmul partial is KSLICE*16, summed over ROWS
# by the cascade). Columns are independent -> COLS C blocks of 64 i32 each.
#
# Usage: r5_gen.py COLS ROWS > r5_array.mlir   (then build with r5_build.sh)
import sys

COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # cascade depth per column (tile rows 2..2+ROWS-1)
AW = 256    # 4 M-tiles * size_A (4*64)
WW = 128    # one shared int4 weight tile
CW = 256    # 4 M-tiles * size_C

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

out = ["module {", "  aie.device(npu2) {"]
# tiles
for c in range(COLS):
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    for r in rows:
        out.append(f"    %t{c}_{r} = aie.tile({c}, {r})")
# cascade flows: north(r+1) -> south(r) within each column
for c in range(COLS):
    for r in rows[:-1]:
        out.append(f"    aie.cascade_flow(%t{c}_{r+1}, %t{c}_{r})")
# objectfifos: broadcast A/W to all rows in a column; C from the tail (row 2)
cons = lambda c: "{" + ", ".join(f"%t{c}_{r}" for r in rows) + "}"
for c in range(COLS):
    out.append(f"    aie.objectfifo @fa{c}(%shim{c}, {cons(c)}, 2 : i32) : !aie.objectfifo<memref<{AW}xi8>>")
    out.append(f"    aie.objectfifo @fw{c}(%shim{c}, {cons(c)}, 2 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
    out.append(f"    aie.objectfifo @fc{c}(%t{c}_2, {{%shim{c}}}, 1 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
# kernel decls
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
# runtime sequence: A (shared broadcast), W (shared), C (COLS*64 i32)
args = ", ".join([f"%A: memref<{AW}xi8>", f"%W: memref<{WW}xi8>", f"%C: memref<{COLS*CW}xi32>"])
out.append(f"    aie.runtime_sequence({args}) {{")
tid = 0
for c in range(COLS):
    out.append(f'''      %ta{c} = aiex.dma_configure_task_for @fa{c} {{
        aie.dma_bd(%A : memref<{AW}xi8>, 0, {AW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {AW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{c})
      %tw{c} = aiex.dma_configure_task_for @fw{c} {{
        aie.dma_bd(%W : memref<{WW}xi8>, 0, {WW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {WW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw{c})
      %tc{c} = aiex.dma_configure_task_for @fc{c} {{
        aie.dma_bd(%C : memref<{COLS*CW}xi32>, {c*CW}, {CW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {CW}, stride = 1>]) {{burst_length = 0 : i32}}
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
