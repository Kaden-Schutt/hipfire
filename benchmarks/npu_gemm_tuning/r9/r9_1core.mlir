// R9 single-core load-reuse microbench harness (npu1/aie2). One core acquires the tiled
// A/W working set (up to 256 tiles each) + C block, and runs r9_ubench's REPEAT*KT hot
// loop. Buffers cover any MTxNTxKT with MT*KT<=256, NT*KT<=256, MT*NT<=16.
module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    aie.objectfifo @fa(%shim, {%t}, 1 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @fw(%shim, {%t}, 1 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @fc(%t, {%shim}, 1 : i32) : !aie.objectfifo<memref<512xi32>>
    func.func private @r9_ubench(memref<16384xi8>, memref<16384xi8>, memref<512xi32>) attributes {link_with = "r9.o"}
    %core = aie.core(%t) {
      %z = arith.constant 0 : index
      %m = arith.constant 9223372036854775807 : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %c = aie.objectfifo.acquire @fc(Produce, 1) : !aie.objectfifosubview<memref<512xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<512xi32>> -> memref<512xi32>
        func.call @r9_ubench(%av, %wv, %cv) : (memref<16384xi8>, memref<16384xi8>, memref<512xi32>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fw(Consume, 1)
        aie.objectfifo.release @fc(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%A: memref<16384xi8>, %W: memref<16384xi8>, %C: memref<512xi32>) {
      %ta = aiex.dma_configure_task_for @fa {
        aie.dma_bd(%A : memref<16384xi8>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta)
      %tw = aiex.dma_configure_task_for @fw {
        aie.dma_bd(%W : memref<16384xi8>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tw)
      %tc = aiex.dma_configure_task_for @fc {
        aie.dma_bd(%C : memref<512xi32>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tc)
      aiex.dma_await_task(%tc)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tw)
    }
  }
}
