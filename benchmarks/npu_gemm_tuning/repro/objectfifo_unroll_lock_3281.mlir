module {
  aie.device(npu2) @op1_GEMM {
    %tile_0_5 = aie.tile(0, 5)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @A_L2L1_3(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    %rtp3_0 = aie.buffer(%tile_0_5) {sym_name = "rtp3_0"} : memref<2xi32> = dense<0>
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp3_0[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp3_0[%c1_1] : memref<2xi32>
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          }
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<65536xbf16>) {
      %0 = aiex.dma_configure_task_for @C_L2L3_0 {
      } {repeat_count = 3 : i32}
    }
  }
}
