// R8 single-core microbench harness (npu1/aie2). One core acquires a single resident
// A/W tile and stores C; all the compute is the REPEAT loop inside r8_ubench, so one
// dispatch = NACC*REPEAT mmuls. Buffers are sized to cover every SHAPE (size_A<=64 i8,
// size_B<=128 i8, size_C<=64 i32); the kernel loads/stores only its shape's footprint.
module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    aie.objectfifo @fa(%shim, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @fw(%shim, {%t}, 2 : i32) : !aie.objectfifo<memref<128xi8>>
    aie.objectfifo @fc(%t, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    func.func private @r8_ubench(memref<64xi8>, memref<128xi8>, memref<64xi32>) attributes {link_with = "r8.o"}
    %core = aie.core(%t) {
      %z = arith.constant 0 : index
      %m = arith.constant 9223372036854775807 : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<64xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<64xi8>> -> memref<64xi8>
        %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<128xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<128xi8>> -> memref<128xi8>
        %c = aie.objectfifo.acquire @fc(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        func.call @r8_ubench(%av, %wv, %cv) : (memref<64xi8>, memref<128xi8>, memref<64xi32>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fw(Consume, 1)
        aie.objectfifo.release @fc(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%A: memref<64xi8>, %W: memref<128xi8>, %C: memref<64xi32>) {
      %ta = aiex.dma_configure_task_for @fa {
        aie.dma_bd(%A : memref<64xi8>, 0, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta)
      %tw = aiex.dma_configure_task_for @fw {
        aie.dma_bd(%W : memref<128xi8>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tw)
      %tc = aiex.dma_configure_task_for @fc {
        aie.dma_bd(%C : memref<64xi32>, 0, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tc)
      aiex.dma_await_task(%tc)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tw)
    }
  }
}
