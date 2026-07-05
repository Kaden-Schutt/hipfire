// R5 4-core K-cascade: a full column (rows 5,4,3,2) splits the K contraction.
// Cascade flows north->south head(0,5) -> mid(0,4) -> mid(0,3) -> tail(0,2), each
// core adding its KSLICE partial to the running accumulator; the tail stores C.
// All-ones A/W broadcast to all 4 cores => each partial = KSLICE*16 = 256, so the
// cascade sum C[0] must be 4*256 = 1024.
//
// Build: r5_build.sh r5_4core.mlir <workdir> 16
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %head = aie.tile(0, 5)   // cascade source must be North/West of dest
    %m1   = aie.tile(0, 4)
    %m2   = aie.tile(0, 3)
    %tail = aie.tile(0, 2)
    aie.cascade_flow(%head, %m1)
    aie.cascade_flow(%m1, %m2)
    aie.cascade_flow(%m2, %tail)

    aie.objectfifo @fa(%shim, {%head, %m1, %m2, %tail}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>
    aie.objectfifo @fw(%shim, {%head, %m1, %m2, %tail}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @fc(%tail, {%shim}, 1 : i32) : !aie.objectfifo<memref<64xi32>>

    func.func private @r5_cascade_head(memref<1024xi8>, memref<2048xi8>) attributes {link_with = "r5_head.o"}
    func.func private @r5_cascade_mid(memref<1024xi8>, memref<2048xi8>) attributes {link_with = "r5_mid.o"}
    func.func private @r5_cascade_tail(memref<1024xi8>, memref<2048xi8>, memref<64xi32>) attributes {link_with = "r5_tail.o"}

    %ch = aie.core(%head) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
        %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
        func.call @r5_cascade_head(%av, %wv) : (memref<1024xi8>, memref<2048xi8>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fw(Consume, 1)
      }
      aie.end
    }

    %cm1 = aie.core(%m1) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
        %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
        func.call @r5_cascade_mid(%av, %wv) : (memref<1024xi8>, memref<2048xi8>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fw(Consume, 1)
      }
      aie.end
    }

    %cm2 = aie.core(%m2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
        %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
        func.call @r5_cascade_mid(%av, %wv) : (memref<1024xi8>, memref<2048xi8>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fw(Consume, 1)
      }
      aie.end
    }

    %ct = aie.core(%tail) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @fa(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
        %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
        %c = aie.objectfifo.acquire @fc(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        func.call @r5_cascade_tail(%av, %wv, %cv) : (memref<1024xi8>, memref<2048xi8>, memref<64xi32>) -> ()
        aie.objectfifo.release @fa(Consume, 1)
        aie.objectfifo.release @fw(Consume, 1)
        aie.objectfifo.release @fc(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%A: memref<1024xi8>, %W: memref<2048xi8>, %C: memref<64xi32>) {
      %t1 = aiex.dma_configure_task_for @fa {
        aie.dma_bd(%A : memref<1024xi8>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @fw {
        aie.dma_bd(%W : memref<2048xi8>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @fc {
        aie.dma_bd(%C : memref<64xi32>, 0, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
    }
  }
}
