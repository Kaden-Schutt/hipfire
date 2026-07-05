// R13 — one memtile-routed column: the whole_array broadcast/distribute mechanism.
// shim(0,0) -> memtile(0,1) -> 4 cores (0,2..5). A super-block (4 core-slices) is
// DISTRIBUTED (one 6x16-tile A-block per core); the W block is BROADCAST (same K x 12
// to all 4 cores). Each core runs r11_gemm on its A-slice x the shared W -> a distinct
// output block-row of a [24 x 12]-tile column stripe. C is JOINED back through the memtile.
// So DMA per column-block = 4*A-block + 1*W-block for 4 cores' macs: W is fed once and
// reused 4x, raising DMA intensity to 32*LM*LN/(4*LM+LN) = 64 mac/byte (2x the R12 per-core
// 32), the reuse the DDR-bound array needs. Block LM=6 LN=12 KT=16 (r11.o built to match).
module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    // A: shim -> memtile (24576 B = 4 x 6144), distributed one 6144 B slice per core.
    aie.objectfifo @a_sh(%shim, {%mt}, 2 : i32) : !aie.objectfifo<memref<24576xi8>>
    aie.objectfifo @a_0(%mt, {%c2}, 2 : i32) : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo @a_1(%mt, {%c3}, 2 : i32) : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo @a_2(%mt, {%c4}, 2 : i32) : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo @a_3(%mt, {%c5}, 2 : i32) : !aie.objectfifo<memref<6144xi8>>
    aie.objectfifo.link [@a_sh] -> [@a_0, @a_1, @a_2, @a_3] ([] [0, 6144, 12288, 18432])

    // W: shim -> memtile (12288 B), broadcast identically to all 4 cores.
    aie.objectfifo @w_sh(%shim, {%mt}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
    aie.objectfifo @w_bc(%mt, {%c2, %c3, %c4, %c5}, 2 : i32) : !aie.objectfifo<memref<12288xi8>>
    aie.objectfifo.link [@w_sh] -> [@w_bc] ([] [0])

    // C: 4 cores -> memtile (joined into 9216 i32) -> shim.
    aie.objectfifo @c_0(%c2, {%mt}, 2 : i32) : !aie.objectfifo<memref<2304xi32>>
    aie.objectfifo @c_1(%c3, {%mt}, 2 : i32) : !aie.objectfifo<memref<2304xi32>>
    aie.objectfifo @c_2(%c4, {%mt}, 2 : i32) : !aie.objectfifo<memref<2304xi32>>
    aie.objectfifo @c_3(%c5, {%mt}, 2 : i32) : !aie.objectfifo<memref<2304xi32>>
    aie.objectfifo @c_sh(%mt, {%shim}, 2 : i32) : !aie.objectfifo<memref<9216xi32>>
    aie.objectfifo.link [@c_0, @c_1, @c_2, @c_3] -> [@c_sh] ([0, 2304, 4608, 6912] [])

    func.func private @r11_gemm(memref<6144xi8>, memref<12288xi8>, memref<2304xi32>) attributes {link_with = "r11.o"}

    %core2 = aie.core(%c2) {
      %z = arith.constant 0 : index
      %m = arith.constant 9223372036854775807 : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {
        %a = aie.objectfifo.acquire @a_0(Consume, 1) : !aie.objectfifosubview<memref<6144xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<6144xi8>> -> memref<6144xi8>
        %w = aie.objectfifo.acquire @w_bc(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
        %c = aie.objectfifo.acquire @c_0(Produce, 1) : !aie.objectfifosubview<memref<2304xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<2304xi32>> -> memref<2304xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<6144xi8>, memref<12288xi8>, memref<2304xi32>) -> ()
        aie.objectfifo.release @a_0(Consume, 1)
        aie.objectfifo.release @w_bc(Consume, 1)
        aie.objectfifo.release @c_0(Produce, 1)
      }
      aie.end
    }
    %core3 = aie.core(%c3) {
      %z = arith.constant 0 : index
      %m = arith.constant 9223372036854775807 : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {
        %a = aie.objectfifo.acquire @a_1(Consume, 1) : !aie.objectfifosubview<memref<6144xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<6144xi8>> -> memref<6144xi8>
        %w = aie.objectfifo.acquire @w_bc(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
        %c = aie.objectfifo.acquire @c_1(Produce, 1) : !aie.objectfifosubview<memref<2304xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<2304xi32>> -> memref<2304xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<6144xi8>, memref<12288xi8>, memref<2304xi32>) -> ()
        aie.objectfifo.release @a_1(Consume, 1)
        aie.objectfifo.release @w_bc(Consume, 1)
        aie.objectfifo.release @c_1(Produce, 1)
      }
      aie.end
    }
    %core4 = aie.core(%c4) {
      %z = arith.constant 0 : index
      %m = arith.constant 9223372036854775807 : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {
        %a = aie.objectfifo.acquire @a_2(Consume, 1) : !aie.objectfifosubview<memref<6144xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<6144xi8>> -> memref<6144xi8>
        %w = aie.objectfifo.acquire @w_bc(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
        %c = aie.objectfifo.acquire @c_2(Produce, 1) : !aie.objectfifosubview<memref<2304xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<2304xi32>> -> memref<2304xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<6144xi8>, memref<12288xi8>, memref<2304xi32>) -> ()
        aie.objectfifo.release @a_2(Consume, 1)
        aie.objectfifo.release @w_bc(Consume, 1)
        aie.objectfifo.release @c_2(Produce, 1)
      }
      aie.end
    }
    %core5 = aie.core(%c5) {
      %z = arith.constant 0 : index
      %m = arith.constant 9223372036854775807 : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {
        %a = aie.objectfifo.acquire @a_3(Consume, 1) : !aie.objectfifosubview<memref<6144xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<6144xi8>> -> memref<6144xi8>
        %w = aie.objectfifo.acquire @w_bc(Consume, 1) : !aie.objectfifosubview<memref<12288xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<12288xi8>> -> memref<12288xi8>
        %c = aie.objectfifo.acquire @c_3(Produce, 1) : !aie.objectfifosubview<memref<2304xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<2304xi32>> -> memref<2304xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<6144xi8>, memref<12288xi8>, memref<2304xi32>) -> ()
        aie.objectfifo.release @a_3(Consume, 1)
        aie.objectfifo.release @w_bc(Consume, 1)
        aie.objectfifo.release @c_3(Produce, 1)
      }
      aie.end
    }

    // Stream N_BLK column-blocks: A super (NBLK x 24576), W (NBLK x 12288), C (NBLK x 9216 i32).
    aie.runtime_sequence(%A: memref<12582912xi8>, %W: memref<6291456xi8>, %C: memref<4718592xi32>) {
      %ta = aiex.dma_configure_task_for @a_sh {
        aie.dma_bd(%A : memref<12582912xi8>, 0, 12582912, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 24576>, <size = 24576, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta)
      %tw = aiex.dma_configure_task_for @w_sh {
        aie.dma_bd(%W : memref<6291456xi8>, 0, 6291456, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 12288>, <size = 12288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tw)
      %tc = aiex.dma_configure_task_for @c_sh {
        aie.dma_bd(%C : memref<4718592xi32>, 0, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 9216>, <size = 9216, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tc)
      aiex.dma_await_task(%tc)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tw)
    }
  }
}
