// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// 4-way fused HFQ4-G256 GEMV for the Qwen3.5 DeltaNet linear-attention
// layer preamble. One kernel launch performs all four projections from a
// single shared x_rot input:
//
//   y_qkv   = W_qkv   · x_rot      (M=qkv_m,   e.g. 6144 on 0.8B)
//   y_z     = W_z     · x_rot      (M=z_m,     e.g. 2048)
//   y_beta  = W_beta  · x_rot      (M=beta_m,  e.g. 16)
//   y_alpha = W_alpha · x_rot      (M=alpha_m, e.g. 16)
//
// Grid = qkv_m + z_m + beta_m + alpha_m workgroups × 32 threads. Each
// workgroup picks its target matrix/row via blockIdx.x range check, then
// runs the SAME 4x-unroll inner loop as gemv_hfq4g256.hip — including
// the 5302926 tail-group accumulator fix (acc[g%4], not all-into-acc0).
//
// Arch coverage: works on every RDNA generation (gfx1010 / gfx1013 /
// gfx1030 / gfx1100+) because gemv_hfq4g256.hip and its arch variants
// were unified onto the same 4-accumulator inner loop after the 5302926
// fix. The launch_bounds(32, 16) target matches the gfx1100 register
// budget; on gfx1010 this is the same single-wave-per-SIMD-half
// occupancy used by the standalone gemv_hfq4g256.hip — both kernels
// share the same VGPR pressure profile.
//
// Why fuse:
//   1. Four launches → one → saves ~3 × (host submit + dispatch floor)
//      per LA layer. On gfx1100 that's ~36 µs/layer ≈ ~640 µs per
//      forward. On gfx1010 the per-launch dispatch floor is even higher,
//      so the relative win is at least as large.
//   2. The 16-wave w_beta and w_alpha launches are catastrophically
//      under-subscribed on their own (16 / N_max_waves slots ≪ 1%).
//      Co-scheduling them with the 6144-wave wqkv body lets the wave
//      scheduler keep real work in flight during the tails.
//
// Bit-exact with four sequential gemv_hfq4g256 calls — the inner loop
// is a verbatim copy.

#ifndef HIPFIRE_QKVZA_WAVES_PER_BLOCK
#define HIPFIRE_QKVZA_WAVES_PER_BLOCK 1
#endif

#ifndef HIPFIRE_QKVZA_BLOCK_SIZE
#define HIPFIRE_QKVZA_BLOCK_SIZE 32
#endif

#ifndef HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU
#define HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU 16
#endif

#ifndef HIPFIRE_QKVZA_KERNEL_NAME
#define HIPFIRE_QKVZA_KERNEL_NAME fused_qkvza_hfq4g256
#endif

__launch_bounds__(HIPFIRE_QKVZA_BLOCK_SIZE, HIPFIRE_QKVZA_MIN_BLOCKS_PER_CU)
extern "C" __global__ void HIPFIRE_QKVZA_KERNEL_NAME(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ x,
    float*       __restrict__ y_qkv,
    float*       __restrict__ y_z,
    float*       __restrict__ y_beta,
    float*       __restrict__ y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K
) {
    const int block_tid = threadIdx.x;
#if defined(HIPFIRE_RDNA3_QKVZA_LDSX8)
    // Eight row waves reuse one K=2048 activation staged as
    // [group][element-in-lane][lane]. The transposition makes every LDS
    // consumer read contiguous across its 32 lanes instead of producing an
    // eight-way bank conflict from the natural lane*8 layout.
    __shared__ float x_lds[2048];
    #pragma unroll
    for (int d = block_tid; d < 2048; d += HIPFIRE_QKVZA_BLOCK_SIZE) {
        const int group = d >> 8;
        const int j = (d >> 5) & 7;
        const int lane = d & 31;
        x_lds[d] = x[group * 256 + lane * 8 + j];
    }
    __syncthreads();
#endif
#if HIPFIRE_QKVZA_WAVES_PER_BLOCK > 1
    // Pack independent row waves into one workgroup. This does not split a
    // row or introduce a cross-wave reduction: each wave retains the base
    // kernel's lane mapping, accumulator order, and shuffle tree.
    const int wave = threadIdx.x >> 5;
    const int tid = threadIdx.x & 31;
    const int gid = blockIdx.x * HIPFIRE_QKVZA_WAVES_PER_BLOCK + wave;
#else
    const int tid = threadIdx.x;
    const int gid = blockIdx.x;
#endif
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    // Route this workgroup to its matrix + output row.
    const char* A;
    float* y;
    int row;
    if (gid < qkv_m) {
        A = A_qkv;   y = y_qkv;   row = gid;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     y = y_z;     row = gid - qkv_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  y = y_beta;  row = gid - (qkv_m + z_m);
    } else {
        A = A_alpha; y = y_alpha; row = gid - (qkv_m + z_m + beta_m);
    }

#if defined(HIPFIRE_RDNA3_QKVZA_K2048)
    // Qwen3.5/3.6 A3B decode always reaches this path with K=2048. Exposing
    // that contract to LLVM removes the runtime divide, quad-loop trip count,
    // and all three dead tail bodies while retaining the exact base-kernel
    // load and FMA order. The host dispatch enforces K==2048.
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
#if defined(HIPFIRE_RDNA3_QKVZA_REDUCE_CHAIN)
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_stride;
#else
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
#endif
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif

    // Four accumulators for the 4x group unroll. Tail groups go back into
    // acc[gi] to preserve the 4-way interleaving the combine step assumes
    // (this is the 5302926 bug fix — DO NOT collapse into acc0).
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;
#if defined(__gfx1200__) || defined(__gfx1201__)
        // R4c0_qkvza_spref272: L2-resident BOD (occ~35% mem_busy~41%);
        // soft-prefetch next quad weights (+272B = 2 groups) while this quad's DOG runs.
        // gfx12-ONLY: s_prefetch_data needs target feature gfx12-insts. The ENTIRE block is
        // gated so gfx11/gfx10 (RDNA1-3) compile BYTE-IDENTICALLY to the pre-lever kernel —
        // zero effect on any other arch's codegen. Hint only; identical load/FMA stream.
        if (q + 1 < quads) {
            __builtin_amdgcn_s_prefetch_data((const void*)(gp0 + 272), 0);
        }
#endif

        float sc0 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 136));
        float zp0 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 136 + 4));
        float sc1 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 136));
        float zp1 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 136 + 4));
        float sc2 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 136));
        float zp2 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 136 + 4));
        float sc3 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 136));
        float zp3 = __builtin_bit_cast(float, HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 136 + 4));

        unsigned int pk0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 8 + boff, (g + 0) * 136 + 8 + boff);
        unsigned int pk1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 8 + boff, (g + 1) * 136 + 8 + boff);
        unsigned int pk2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 8 + boff, (g + 2) * 136 + 8 + boff);
        unsigned int pk3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 8 + boff, (g + 3) * 136 + 8 + boff);

        int base = g * 256 + tid * 8;

        // R-hoist_x32: vectorize x as float4 pairs BEFORE the DOG FMAs,
        // matching the winning gate_up R20c2_hoist_x32. Bit-exact — a float4 load of
        // contiguous x is the same 8 values as scalar x[base..base+7], just one
        // 128-bit load instead of 8 scalar loads (more in-flight bytes → higher
        // mem_busy on the L2-resident-but-mem-busy-41% qkvza). Default gfx12;
        // Radiowave can opt gfx1100 into the same schedule after checking emitted
        // register pressure and model-level throughput. Other gfx11/gfx10 builds
        // retain the original scalar path byte-for-byte.
#if defined(HIPFIRE_RDNA3_QKVZA_LDSX8)
        #define DOG_LDS(pk, sc, zp, group, a) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * x_lds[(group) * 256 + 0 * 32 + tid] \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_lds[(group) * 256 + 1 * 32 + tid] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_lds[(group) * 256 + 2 * 32 + tid] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_lds[(group) * 256 + 3 * 32 + tid] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_lds[(group) * 256 + 4 * 32 + tid] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_lds[(group) * 256 + 5 * 32 + tid] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_lds[(group) * 256 + 6 * 32 + tid] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_lds[(group) * 256 + 7 * 32 + tid]
        DOG_LDS(pk0, sc0, zp0, g + 0, acc0);
        DOG_LDS(pk1, sc1, zp1, g + 1, acc1);
        DOG_LDS(pk2, sc2, zp2, g + 2, acc2);
        DOG_LDS(pk3, sc3, zp3, g + 3, acc3);
        #undef DOG_LDS
#elif defined(__gfx1200__) || defined(__gfx1201__) || defined(HIPFIRE_RDNA3_QKVZA_HOIST_X32)
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base + 772);
        #define DOG_V(pk, sc, zp, a, x0, x1, x2, x3, x4, x5, x6, x7) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * (x0) \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (x1) \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (x2) \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (x3) \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (x4) \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (x5) \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (x6) \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (x7)
        DOG_V(pk0, sc0, zp0, acc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
        DOG_V(pk1, sc1, zp1, acc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
        DOG_V(pk2, sc2, zp2, acc2, xv2a.x,xv2a.y,xv2a.z,xv2a.w,xv2b.x,xv2b.y,xv2b.z,xv2b.w);
        DOG_V(pk3, sc3, zp3, acc3, xv3a.x,xv3a.y,xv3a.z,xv3a.w,xv3b.x,xv3b.y,xv3b.z,xv3b.w);
        #undef DOG_V
#else
        // ORIGINAL scalar path — verbatim, so gfx11/gfx10 compile byte-identically.
        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]
        DOG(pk0, sc0, zp0, base,       acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
#endif
    }

    // Tail handling: preserve acc[gi] interleaving. See 5302926 bug report.
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >>  4) & 0xFu) + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >>  8) & 0xFu) + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
#if defined(HIPFIRE_RDNA3_QKVZA_LDSX8)
        #define TAIL_DOG_LDS(pk, sc, zp, group, a) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * x_lds[(group) * 256 + 0 * 32 + tid] \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_lds[(group) * 256 + 1 * 32 + tid] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_lds[(group) * 256 + 2 * 32 + tid] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_lds[(group) * 256 + 3 * 32 + tid] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_lds[(group) * 256 + 4 * 32 + tid] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_lds[(group) * 256 + 5 * 32 + tid] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_lds[(group) * 256 + 6 * 32 + tid] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_lds[(group) * 256 + 7 * 32 + tid]
        TAIL_DOG_LDS(pk, sc, zp, g, acc0);
#else
        TAIL_DOG(pk, sc, zp, base, acc0);
#endif
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
#if defined(HIPFIRE_RDNA3_QKVZA_LDSX8)
        TAIL_DOG_LDS(pk, sc, zp, g, acc1);
#else
        TAIL_DOG(pk, sc, zp, base, acc1);
#endif
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
#if defined(HIPFIRE_RDNA3_QKVZA_LDSX8)
        TAIL_DOG_LDS(pk, sc, zp, g, acc2);
#else
        TAIL_DOG(pk, sc, zp, base, acc2);
#endif
    }
    #undef TAIL_DOG
#if defined(HIPFIRE_RDNA3_QKVZA_LDSX8)
    #undef TAIL_DOG_LDS
#endif

    float acc = (acc0 + acc1) + (acc2 + acc3);
#if defined(HIPFIRE_RDNA3_QKVZA_REDUCE_CHAIN)
    acc += __shfl_down(acc, 16);
    acc += __shfl_down(acc, 8);
    acc += __shfl_down(acc, 4);
    acc += __shfl_down(acc, 2);
    acc += __shfl_down(acc, 1);
#else
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
#endif

    if (tid == 0) y[row] = acc;
}
