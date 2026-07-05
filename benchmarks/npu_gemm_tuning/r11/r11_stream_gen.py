#!/usr/bin/env python3
# Emit the R11 single-core L2-tiled streaming-GEMM MLIR (npu1/aie2). One core streams
# N_BLK LMxLN output blocks per dispatch; each iteration acquires the block's A/W stripes
# (DMA'd once from DDR, double-buffered) and produces its C block via r11_gemm, which
# sweeps the 3x3 register tile over the LMxLN block reusing the resident stripes. DMA
# intensity = 8*LM*LN/(LM+LN) mac/byte (vs R10's NT*4).
#
# Per block: A = LM*KT tiles * 64 B, W = LN*KT tiles * 64 B, C = LM*LN tiles * 32 i32.
# Usage: r11_stream_gen.py LM LN KT N_BLK > r11.mlir
import sys
LM  = int(sys.argv[1]) if len(sys.argv) > 1 else 6
LN  = int(sys.argv[2]) if len(sys.argv) > 2 else 6
KT  = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NBLK = int(sys.argv[4]) if len(sys.argv) > 4 else 512
SA, SBb, SC = 64, 64, 32
AB = LM * KT * SA
WB = LN * KT * SBb
CB = LM * LN * SC
INF = 9223372036854775807

print(f'''module {{
  aie.device(npu1) {{
    %shim = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    aie.objectfifo @fa(%shim, {{%t}}, 2 : i32) : !aie.objectfifo<memref<{AB}xi8>>
    aie.objectfifo @fw(%shim, {{%t}}, 2 : i32) : !aie.objectfifo<memref<{WB}xi8>>
    aie.objectfifo @fc(%t, {{%shim}}, 2 : i32) : !aie.objectfifo<memref<{CB}xi32>>
    func.func private @r11_gemm(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r11.o"}}
    %core = aie.core(%t) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {{
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>
        %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>
        %c = aie.objectfifo.acquire @fc(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fw(Consume, 1)
        aie.objectfifo.release @fc(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%A: memref<{NBLK*AB}xi8>, %W: memref<{NBLK*WB}xi8>, %C: memref<{NBLK*CB}xi32>) {{
      %ta = aiex.dma_configure_task_for @fa {{
        aie.dma_bd(%A : memref<{NBLK*AB}xi8>, 0, {NBLK*AB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {AB}>, <size = {AB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %tw = aiex.dma_configure_task_for @fw {{
        aie.dma_bd(%W : memref<{NBLK*WB}xi8>, 0, {NBLK*WB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {WB}>, <size = {WB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw)
      %tc = aiex.dma_configure_task_for @fc {{
        aie.dma_bd(%C : memref<{NBLK*CB}xi32>, 0, {NBLK*CB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBLK}, stride = {CB}>, <size = {CB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc)
      aiex.dma_await_task(%tc)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tw)
    }}
  }}
}}''')
