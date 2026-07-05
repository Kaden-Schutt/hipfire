#!/usr/bin/env python3
# M-PARALLEL, W-BROADCAST R6 array (mirror of r6_gen.py). Each of COLS cores computes a
# DISTINCT M-block over full N (streaming NB N-slabs), all sharing ONE broadcast W stream:
# shim0 -> memtile -> {all cores}. Because every M-block reuses the same weights, the W is
# read from DRAM once and fanned out, instead of re-fed per M-block (r6_gen.py N-parallel
# re-reads W for every M-block). This cuts the W DRAM re-feed COLS-fold and lets one
# dispatch cover COLS M-blocks, collapsing the dispatch count.
#
# Uses the r6_gemm_ts.cc kernel (row-major A/C, linear DMA). Sizes come from the build-time
# MT/NT/KCHUNK. Usage: r6_gen_mp.py COLS NB AW WW CW > aie_mp.mlir
import sys

COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2
AW = int(sys.argv[3]) if len(sys.argv) > 3 else 8192   # A bytes per M-block (resident)
WW = int(sys.argv[4]) if len(sys.argv) > 4 else 8192   # W bytes per N-slab (broadcast)
CW = int(sys.argv[5]) if len(sys.argv) > 5 else 2048   # C i32 elements per (M-block,N-slab)
INF = 9223372036854775807

out = ["module {", "  aie.device(npu2) {"]
for c in range(COLS):
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
out.append("    %mt = aie.tile(0, 1)")  # memtile: W broadcast distribution point
for c in range(COLS):
    out.append(f"    %t{c} = aie.tile({c}, 2)")

# A: per-core distinct M-block, resident (single-buffer).
for c in range(COLS):
    out.append(f"    aie.objectfifo @fa{c}(%shim{c}, {{%t{c}}}, 1 : i32) : !aie.objectfifo<memref<{AW}xi8>>")
# W: broadcast. shim0 -> memtile (fw_in), memtile -> all cores (fw), linked.
cores = ", ".join(f"%t{c}" for c in range(COLS))
out.append(f"    aie.objectfifo @fw_in(%shim0, {{%mt}}, 2 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
out.append(f"    aie.objectfifo @fw(%mt, {{{cores}}}, 2 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
out.append("    aie.objectfifo.link [@fw_in] -> [@fw]([] [])")
# C: per-core, NB blocks.
for c in range(COLS):
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
          %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<{WW}xi8>>
          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>
          %cc = aie.objectfifo.acquire @fc{c}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>
          %cv = aie.objectfifo.subview.access %cc[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>
          func.call @r6_mac(%av, %wv, %cv) : (memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()
          aie.objectfifo.release @fw(Consume, 1)
          aie.objectfifo.release @fc{c}(Produce, 1)
        }}
        aie.objectfifo.release @fa{c}(Consume, 1)
      }}
      aie.end
    }}''')

# runtime: A = COLS distinct M-blocks; W = ONE region (NB slabs) fed once to the broadcast;
# C = COLS*NB blocks (each core's M-block × N-slabs).
ATOT = COLS * AW
WTOT = NB * WW
CTOT = COLS * NB * CW
args = ", ".join([f"%A: memref<{ATOT}xi8>", f"%W: memref<{WTOT}xi8>", f"%C: memref<{CTOT}xi32>"])
out.append(f"    aie.runtime_sequence({args}) {{")
for c in range(COLS):
    out.append(f'''      %ta{c} = aiex.dma_configure_task_for @fa{c} {{
        aie.dma_bd(%A : memref<{ATOT}xi8>, {c*AW}, {AW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {AW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{c})''')
out.append(f'''      %tw = aiex.dma_configure_task_for @fw_in {{
        aie.dma_bd(%W : memref<{WTOT}xi8>, 0, {WTOT}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {WTOT}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw)''')
for c in range(COLS):
    out.append(f'''      %tc{c} = aiex.dma_configure_task_for @fc{c} {{
        aie.dma_bd(%C : memref<{CTOT}xi32>, {c*NB*CW}, {NB*CW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NB*CW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{c})''')
for c in range(COLS):
    out.append(f"      aiex.dma_await_task(%tc{c})")
for c in range(COLS):
    out.append(f"      aiex.dma_free_task(%ta{c})")
out.append("      aiex.dma_free_task(%tw)")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
