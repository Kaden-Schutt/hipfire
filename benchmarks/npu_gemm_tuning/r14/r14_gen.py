#!/usr/bin/env python3
# R14 — the full 4x4 whole_array broadcast GEMM (npu1/aie2). Output C is a 4x4 grid of
# LMxLN blocks; block (i,j) is computed by core (col=j, row=2+i). Two orthogonal reuses,
# each memtile-routed (mechanism proven in r13):
#   - W (in-column broadcast): column j's shim feeds W-stripe_j -> memtile_j -> the 4 cores
#     of column j (all block-rows i share the same W cols).
#   - A (CROSS-column broadcast): row i's A-stripe enters shim_i -> memtile_i -> broadcast to
#     the 4 cores at physical row 2+i across ALL columns (all block-cols j share the same A
#     rows). This is the horizontal broadcast a single column can't do (r13's limit).
#   - C (in-column join): each column's 4 cores' C joined via its memtile -> its shim.
# Uses all 4 column shims (4 DDR paths) and feeds only 4 A + 4 W stripes for 16 cores' macs
# -> DMA intensity 32*LM*LN/(LM+LN) (3-4x single-core), the reuse the DDR-bound array needs.
#
# Usage: r14_gen.py LM LN KT N_BLK > r14.mlir   (r11.o built with the same LM/LN/KT)
import sys
LM  = int(sys.argv[1]) if len(sys.argv) > 1 else 6
LN  = int(sys.argv[2]) if len(sys.argv) > 2 else 12
KT  = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NBLK = int(sys.argv[4]) if len(sys.argv) > 4 else 256
AB = LM * KT * 64        # A-stripe bytes (shared by a physical row)
WB = LN * KT * 64        # W-stripe bytes (shared by a physical column)
CB = LM * LN * 32        # per-core C i32
CJ = 4 * CB              # joined C per column
INF = 9223372036854775807
G = range(4)             # 4 cols, 4 block-rows

out = ["module {", "  aie.device(npu1) {"]
for c in G:
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %mt{c} = aie.tile({c}, 1)")
    for i in G:
        out.append(f"    %c{c}_{i} = aie.tile({c}, {2+i})")

# W: per column j, shim -> memtile -> broadcast to the 4 cores of column j
for j in G:
    cores = ", ".join(f"%c{j}_{i}" for i in G)
    out.append(f"    aie.objectfifo @wsh{j}(%shim{j}, {{%mt{j}}}, 2 : i32) : !aie.objectfifo<memref<{WB}xi8>>")
    out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{cores}}}, 2 : i32) : !aie.objectfifo<memref<{WB}xi8>>")
    out.append(f"    aie.objectfifo.link [@wsh{j}] -> [@wbc{j}] ([] [0])")
# A: per block-row i, shim_i -> memtile_i -> broadcast to core (c, 2+i) across all columns
for i in G:
    cores = ", ".join(f"%c{c}_{i}" for c in G)
    out.append(f"    aie.objectfifo @ash{i}(%shim{i}, {{%mt{i}}}, 2 : i32) : !aie.objectfifo<memref<{AB}xi8>>")
    out.append(f"    aie.objectfifo @abc{i}(%mt{i}, {{{cores}}}, 2 : i32) : !aie.objectfifo<memref<{AB}xi8>>")
    out.append(f"    aie.objectfifo.link [@ash{i}] -> [@abc{i}] ([] [0])")
# C: per column j, 4 cores -> memtile -> shim (join)
for j in G:
    ins = ", ".join(f"@cc{j}_{i}" for i in G)
    offs = ", ".join(str(i * CB) for i in G)
    for i in G:
        out.append(f"    aie.objectfifo @cc{j}_{i}(%c{j}_{i}, {{%mt{j}}}, 2 : i32) : !aie.objectfifo<memref<{CB}xi32>>")
    out.append(f"    aie.objectfifo @csh{j}(%mt{j}, {{%shim{j}}}, 2 : i32) : !aie.objectfifo<memref<{CJ}xi32>>")
    out.append(f"    aie.objectfifo.link [{ins}] -> [@csh{j}] ([{offs}] [])")

out.append(f'    func.func private @r11_gemm(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r11.o"}}')
for c in G:
    for i in G:
        out.append(f'''    %core{c}_{i} = aie.core(%c{c}_{i}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %k = %z to %m step %o {{
        %a = aie.objectfifo.acquire @abc{i}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>
        %w = aie.objectfifo.acquire @wbc{c}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>
        %cc = aie.objectfifo.acquire @cc{c}_{i}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>
        %cv = aie.objectfifo.subview.access %cc[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()
        aie.objectfifo.release @abc{i}(Consume, 1)
        aie.objectfifo.release @wbc{c}(Consume, 1)
        aie.objectfifo.release @cc{c}_{i}(Produce, 1)
      }}
      aie.end
    }}''')

# runtime sequence: 4 A-stripe streams (shim_i), 4 W-stripe streams (shim_j), 4 C joins out.
At, Wt, Ct = NBLK * AB, NBLK * WB, NBLK * CJ
args = ", ".join([f"%A: memref<{4*At}xi8>", f"%W: memref<{4*Wt}xi8>", f"%C: memref<{4*Ct}xi32>"])
out.append(f"    aie.runtime_sequence({args}) {{")
for i in G:
    out.append(f'''      %ta{i} = aiex.dma_configure_task_for @ash{i} {{
        aie.dma_bd(%A : memref<{4*At}xi8>, {i*At}, {At}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {AB}>, <size = {AB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{i})''')
for j in G:
    out.append(f'''      %tw{j} = aiex.dma_configure_task_for @wsh{j} {{
        aie.dma_bd(%W : memref<{4*Wt}xi8>, {j*Wt}, {Wt}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {WB}>, <size = {WB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw{j})''')
for j in G:
    out.append(f'''      %tc{j} = aiex.dma_configure_task_for @csh{j} {{
        aie.dma_bd(%C : memref<{4*Ct}xi32>, {j*Ct}, {Ct}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {CJ}>, <size = {CJ}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{j})''')
for j in G:
    out.append(f"      aiex.dma_await_task(%tc{j})")
for i in G:
    out.append(f"      aiex.dma_free_task(%ta{i})")
for j in G:
    out.append(f"      aiex.dma_free_task(%tw{j})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
