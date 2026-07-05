#!/usr/bin/env python3
# Minimal single-core passthrough for the tensor-buffer-stream reshuffle test: shim DMAs
# row-major A in (linear) and the tiled output back (linear); the core's ts_a kernel does
# the reshuffle in-core via a tensor descriptor. Both DMAs are plain linear dma_bd — the
# whole point is that NO strided DMA / memtile hop is needed; the AGUs do the tiling.
# Usage: ts_gen.py N > aie.mlir   (N = bytes in = bytes out)
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
INF = 9223372036854775807
print(f'''module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    aie.objectfifo @fa(%shim, {{%t}}, 2 : i32) : !aie.objectfifo<memref<{N}xi8>>
    aie.objectfifo @fo(%t, {{%shim}}, 2 : i32) : !aie.objectfifo<memref<{N}xi8>>
    func.func private @ts_a(memref<{N}xi8>, memref<{N}xi8>) attributes {{link_with = "ts_a.o"}}
    %core = aie.core(%t) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {{
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<{N}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{N}xi8>> -> memref<{N}xi8>
        %oo = aie.objectfifo.acquire @fo(Produce, 1) : !aie.objectfifosubview<memref<{N}xi8>>
        %ov = aie.objectfifo.subview.access %oo[0] : !aie.objectfifosubview<memref<{N}xi8>> -> memref<{N}xi8>
        func.call @ts_a(%av, %ov) : (memref<{N}xi8>, memref<{N}xi8>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fo(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%A: memref<{N}xi8>, %O: memref<{N}xi8>) {{
      %ta = aiex.dma_configure_task_for @fa {{
        aie.dma_bd(%A : memref<{N}xi8>, 0, {N}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {N}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %to = aiex.dma_configure_task_for @fo {{
        aie.dma_bd(%O : memref<{N}xi8>, 0, {N}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {N}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ta)
    }}
  }}
}}''')
