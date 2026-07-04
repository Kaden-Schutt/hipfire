module {
  aie.device(npu2) @op1_GEMM {
    %tile_0_5 = aie.tile(0, 5)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @A_L2L1_3(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c8 step %c1 {
            %7 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          }
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
  }
}
