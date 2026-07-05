#!/usr/bin/env python3
# Generate an R6 array: COLS INDEPENDENT 2D-tiled GEMM cores (one per column, tile
# row 2 — no cascade, the array IS the M-parallelism). Each core keeps A resident and
# streams NB weight super-tiles from a SHARED DDR region (all shims hit the same
# region = realistic aggregate-feed contention), computing its own C block. This
# measures the real feed-bound aggregate: the fabric caps at ~55 GB/s, so 8 cores
# saturate it just as 32 would.
#
# Sizes assume the r6_mac build-time MT/NT/KCHUNK. Defaults: MT=8 NT=4 KCHUNK=16 ->
# A=MT*KCHUNK*64=8192, W super=NT*KCHUNK*128=8192, C=MT*NT*64=2048 i32.
# Usage: r6_gen.py COLS NB [AW WW CW] > r6_array.mlir
import sys

COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 256
AW = int(sys.argv[3]) if len(sys.argv) > 3 else 8192   # A bytes (resident)
WW = int(sys.argv[4]) if len(sys.argv) > 4 else 8192   # W super-tile bytes (streamed)
CW = int(sys.argv[5]) if len(sys.argv) > 5 else 2048   # C i32 elements per core
# ROUNDS = M-blocks streamed per core in ONE dispatch (whole-GEMM). The cores loop forever,
# so feeding ROUNDS A-blocks (+ ROUNDS*NB W slabs, C blocks) computes ROUNDS M-blocks
# continuously — no inter-dispatch host stall. ROUNDS=1 is the per-dispatch form.
ROUNDS = int(sys.argv[6]) if len(sys.argv) > 6 else 1
INF = 9223372036854775807

out = ["module {", "  aie.device(npu2) {"]
for c in range(COLS):
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %t{c} = aie.tile({c}, 2)")
for c in range(COLS):
    out.append(f"    aie.objectfifo @fa{c}(%shim{c}, {{%t{c}}}, 1 : i32) : !aie.objectfifo<memref<{AW}xi8>>")  # A resident (single-buffer)
    out.append(f"    aie.objectfifo @fw{c}(%shim{c}, {{%t{c}}}, 2 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
    out.append(f"    aie.objectfifo @fc{c}(%t{c}, {{%shim{c}}}, 1 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
out.append(f'    func.func private @r6_mac(memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) attributes {{link_with = "r6_mac.o"}}')
for c in range(COLS):
    out.append(f'''    %core{c} = aie.core(%t{c}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      %nb = arith.constant {NB} : index
      scf.for %i = %z to %m step %o {{
        %a = aie.objectfifo.acquire @fa{c}(Consume, 1) : !aie.objectfifosubview<memref<{AW}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AW}xi8>> -> memref<{AW}xi8>
        scf.for %j = %z to %nb step %o {{
          %w = aie.objectfifo.acquire @fw{c}(Consume, 1) : !aie.objectfifosubview<memref<{WW}xi8>>
          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>
          %cc = aie.objectfifo.acquire @fc{c}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>
          %cv = aie.objectfifo.subview.access %cc[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>
          func.call @r6_mac(%av, %wv, %cv) : (memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()
          aie.objectfifo.release @fw{c}(Consume, 1)
          aie.objectfifo.release @fc{c}(Produce, 1)
        }}
        aie.objectfifo.release @fa{c}(Consume, 1)
      }}
      aie.end
    }}''')
# ROUNDS whole-GEMM streaming (pure-linear DMAs; a repeat BD dim bypasses the objectfifo
# semaphore). A is BROADCAST: one ROUNDS*AW copy, every core's shim streams all of it (all
# cores share the same ROUNDS M-blocks). W: each core its OWN ROUNDS*NB*WW region (weights
# re-streamed per M-block; N-parallel has no W broadcast). C: each core ROUNDS*NB*CW. One
# dispatch computes ROUNDS M-blocks x full N continuously. ROUNDS=1 is the per-dispatch form.
ATOT = ROUNDS * AW
WTOT = COLS * ROUNDS * NB * WW
CTOT = COLS * ROUNDS * NB * CW
args = ", ".join([f"%A: memref<{ATOT}xi8>", f"%W: memref<{WTOT}xi8>", f"%C: memref<{CTOT}xi32>"])
out.append(f"    aie.runtime_sequence({args}) {{")
for c in range(COLS):
    out.append(f'''      %ta{c} = aiex.dma_configure_task_for @fa{c} {{
        aie.dma_bd(%A : memref<{ATOT}xi8>, 0, {ATOT}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {ATOT}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{c})
      %tw{c} = aiex.dma_configure_task_for @fw{c} {{
        aie.dma_bd(%W : memref<{WTOT}xi8>, {c*ROUNDS*NB*WW}, {ROUNDS*NB*WW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {ROUNDS*NB*WW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw{c})
      %tc{c} = aiex.dma_configure_task_for @fc{c} {{
        aie.dma_bd(%C : memref<{CTOT}xi32>, {c*ROUNDS*NB*CW}, {ROUNDS*NB*CW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {ROUNDS*NB*CW}, stride = 1>]) {{burst_length = 0 : i32}}
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
