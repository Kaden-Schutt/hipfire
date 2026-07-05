#!/usr/bin/env python3
# Emit a multi-core R12 array streaming-GEMM MLIR (npu1/aie2). Places COLS x ROWS cores
# (cols 0..COLS-1, rows 2..2+ROWS-1), each running the R11 L2-tiled block kernel (r11_gemm)
# on its OWN distinct A/W stream from DDR (no cross-core reuse — the pessimistic, feed-heavy
# case). Sweeping COLS/ROWS shows where the shared shim DMA saturates: R6 found columns are
# independent (own shim), so 1-per-column should scale ~linearly; extra rows per column
# contend for that column's shim. Aggregate GMAC/s = total macs / dispatch time.
#
# Per core: A = LM*KT tiles*64 B, W = LN*KT tiles*64 B, C = LM*LN tiles*32 i32.
# Usage: r12_gen.py COLS ROWS LM LN KT N_BLK > r12.mlir
import sys
COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LM   = int(sys.argv[3]) if len(sys.argv) > 3 else 6
LN   = int(sys.argv[4]) if len(sys.argv) > 4 else 12
KT   = int(sys.argv[5]) if len(sys.argv) > 5 else 16
NBLK = int(sys.argv[6]) if len(sys.argv) > 6 else 256
SA, SBb, SC = 64, 64, 32
AB = LM * KT * SA
WB = LN * KT * SBb
CB = LM * LN * SC
NC = COLS * ROWS
INF = 9223372036854775807
rows = list(range(2, 2 + ROWS))

out = ["module {", "  aie.device(npu1) {"]
for c in range(COLS):
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    for r in rows:
        out.append(f"    %t{c}_{r} = aie.tile({c}, {r})")
# per-core objectfifos (A/W in from the column shim, C out to it)
for c in range(COLS):
    for r in rows:
        out.append(f"    aie.objectfifo @fa{c}_{r}(%shim{c}, {{%t{c}_{r}}}, 2 : i32) : !aie.objectfifo<memref<{AB}xi8>>")
        out.append(f"    aie.objectfifo @fw{c}_{r}(%shim{c}, {{%t{c}_{r}}}, 2 : i32) : !aie.objectfifo<memref<{WB}xi8>>")
        out.append(f"    aie.objectfifo @fc{c}_{r}(%t{c}_{r}, {{%shim{c}}}, 2 : i32) : !aie.objectfifo<memref<{CB}xi32>>")
out.append(f'    func.func private @r11_gemm(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r11.o"}}')
for c in range(COLS):
    for r in rows:
        out.append(f'''    %core{c}_{r} = aie.core(%t{c}_{r}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {{
        %a = aie.objectfifo.acquire @fa{c}_{r}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>
        %w = aie.objectfifo.acquire @fw{c}_{r}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>
        %c0 = aie.objectfifo.acquire @fc{c}_{r}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>
        %cv = aie.objectfifo.subview.access %c0[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()
        aie.objectfifo.release @fa{c}_{r}(Consume, 1)
        aie.objectfifo.release @fw{c}_{r}(Consume, 1)
        aie.objectfifo.release @fc{c}_{r}(Produce, 1)
      }}
      aie.end
    }}''')
# runtime sequence: each core streams NBLK blocks from its own slice of A/W; C to its slice.
A_i = "%A: memref<" + str(NC * NBLK * AB) + "xi8>"
W_i = "%W: memref<" + str(NC * NBLK * WB) + "xi8>"
C_i = "%C: memref<" + str(NC * NBLK * CB) + "xi32>"
out.append(f"    aie.runtime_sequence({A_i}, {W_i}, {C_i}) {{")
idx = 0
tasks = []
for c in range(COLS):
    for r in rows:
        aoff, woff, coff = idx * NBLK * AB, idx * NBLK * WB, idx * NBLK * CB
        out.append(f'''      %ta{c}_{r} = aiex.dma_configure_task_for @fa{c}_{r} {{
        aie.dma_bd(%A : memref<{NC*NBLK*AB}xi8>, {aoff}, {NBLK*AB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {AB}>, <size = {AB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{c}_{r})
      %tw{c}_{r} = aiex.dma_configure_task_for @fw{c}_{r} {{
        aie.dma_bd(%W : memref<{NC*NBLK*WB}xi8>, {woff}, {NBLK*WB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {WB}>, <size = {WB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw{c}_{r})
      %tc{c}_{r} = aiex.dma_configure_task_for @fc{c}_{r} {{
        aie.dma_bd(%C : memref<{NC*NBLK*CB}xi32>, {coff}, {NBLK*CB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {CB}>, <size = {CB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{c}_{r})''')
        tasks.append(f"{c}_{r}")
        idx += 1
for t in tasks:
    out.append(f"      aiex.dma_await_task(%tc{t})")
for t in tasks:
    out.append(f"      aiex.dma_free_task(%ta{t})")
    out.append(f"      aiex.dma_free_task(%tw{t})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
