// R5 2-core K-cascade validation. Two adjacent cores in column 0 (rows 2,3) split
// the K contraction: the head (0,2) computes its KSLICE partial and put_mcd's it
// onto the cascade; the tail (0,3) get_scd's it, adds its own KSLICE partial, and
// stores C. A/W are broadcast all-ones to both cores, so each partial = KSLICE*16 =
// 256 and the cascade sum C[0] must be 512 (proves the cascade adds across cores;
// 256 would mean the cascade dropped the head).
//
// Build (in a workdir holding r5_head.o + r5_tail.o):
//   aiecc r5_2core.mlir --no-compile-host --no-xchesscc --no-xbridge --peano=$PEANO \
//     --aie-generate-npu-insts --npu-insts-name=insts.bin \
//     --aie-generate-xclbin --xclbin-name=final.xclbin --tmpdir=.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    // Cascade source must be North/West of dest, so head is the NORTHERN tile (row 3)
    // and the cascade flows south to the tail (row 2), which stores C.
    %head = aie.tile(0, 3)
    %tail = aie.tile(0, 2)
    aie.cascade_flow(%head, %tail)

    // A (KSLICE*size_A = 16*64 = 1024 i8) and W (KSLICE*128 = 2048 i8) broadcast to
    // both cores; C (size_C = 64 i32) drained from the tail.
    aie.objectfifo @fa(%shim, {%head, %tail}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>
    aie.objectfifo @fw(%shim, {%head, %tail}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @fc(%tail, {%shim}, 1 : i32) : !aie.objectfifo<memref<64xi32>>

    func.func private @r5_cascade_head(memref<1024xi8>, memref<2048xi8>) attributes {link_with = "r5_head.o"}
    func.func private @r5_cascade_tail(memref<1024xi8>, memref<2048xi8>, memref<64xi32>) attributes {link_with = "r5_tail.o"}

    %h = aie.core(%head) {
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

    %t = aie.core(%tail) {
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
