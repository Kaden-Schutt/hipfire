// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE gate_up GEMV for MQ3-Lloyd (3-bit + per-block 8-entry
// fp16 codebook, 112 B/group). Sibling of
// gemv_mq2g256_lloyd_moe_gate_up_indexed — same grid + expert-ptrs dispatch,
// but the per-group decode is an 8-entry codebook lookup over 3-bit indices
// instead of MQ2's 4-entry/2-bit decode.
//
// X must be FWHT-pre-rotated by the caller (MQ-family kernels rotate X
// once per token, then re-use the rotated vector across all top-k experts).
//
// Block layout (112 B / 256 weights):
//   [0..16)   : 8 × fp16 codebook entries (sorted ascending)
//   [16..112) : 96 bytes packed 3-bit indices (32 chunks × 3 bytes,
//               8 weights/chunk; per-thread 3-byte little-endian load,
//               (pk >> (3*i)) & 7u for i in 0..8). Same cross-byte layout
//               as uniform HFQ3-G256 / gemv_mq3g256_lloyd.
//
// Grid: [M, k_top, 1], block [32]. Output split: row<M/2 → y_gate, ≥M/2 → y_up.
//
// LDS codebook is 32 slots (4 groups × 8 entries). The K4 cooperative load
// uses all 32 threads (group = tid>>3, entry = tid&7); the tail load uses
// tid<8 for the single leftover group.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x_rot,                     // [K] (FWHT-rotated)
    float* __restrict__ y_gate,                          // [k_top × mi]
    float* __restrict__ y_up,                            // [k_top × mi]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 112;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 3;  // 3 bytes of indices per group per thread

    // 32-slot LDS codebook: 4 groups × 8 entries. Group g's codebook
    // occupies cb_lds[g*8 .. g*8+8].
    __shared__ float cb_lds[32];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // ----- main K4 loop -----
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 112;
        const char* gp1 = gp0 + 112;
        const char* gp2 = gp1 + 112;
        const char* gp3 = gp2 + 112;

        // Cooperative load: 32 threads × 1 fp16 → 32 fp32 entries in LDS.
        // thread tid → group (tid >> 3), entry (tid & 7).
        {
            const __half* cb_h =
                (const __half*)(gp0 + (long long)(tid >> 3) * 112);
            cb_lds[tid] = __half2float(cb_h[tid & 7]);
        }
        __syncthreads();

        // Load 3 packed bytes (one per group, this thread's 3 bytes).
        const unsigned char* d0 = (const unsigned char*)(gp0 + 16) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 16) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 16) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 16) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8) | ((unsigned int)d0[2] << 16);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8) | ((unsigned int)d1[2] << 16);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8) | ((unsigned int)d2[2] << 16);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8) | ((unsigned int)d3[2] << 16);

        int base = g * 256 + tid * 8;

        #define DOG3_LDS(pk, lds, b, a)                                   \
            (a) += cb_lds[(lds) + ( (pk)        & 7u)] * x_rot[(b)]       \
                 + cb_lds[(lds) + (((pk) >>  3) & 7u)] * x_rot[(b) + 1]   \
                 + cb_lds[(lds) + (((pk) >>  6) & 7u)] * x_rot[(b) + 2]   \
                 + cb_lds[(lds) + (((pk) >>  9) & 7u)] * x_rot[(b) + 3]   \
                 + cb_lds[(lds) + (((pk) >> 12) & 7u)] * x_rot[(b) + 4]   \
                 + cb_lds[(lds) + (((pk) >> 15) & 7u)] * x_rot[(b) + 5]   \
                 + cb_lds[(lds) + (((pk) >> 18) & 7u)] * x_rot[(b) + 6]   \
                 + cb_lds[(lds) + (((pk) >> 21) & 7u)] * x_rot[(b) + 7]

        DOG3_LDS(pk0,  0, base,        acc0);
        DOG3_LDS(pk1,  8, base + 256,  acc1);
        DOG3_LDS(pk2, 16, base + 512,  acc2);
        DOG3_LDS(pk3, 24, base + 768,  acc3);
        #undef DOG3_LDS
    }

    // ----- tail loop (1, 2, or 3 leftover groups) -----
    #define TAIL_LOAD_AND_DOT(g_expr, acc) do {                                 \
        const int g = (g_expr);                                                  \
        const char* gp = row_ptr + g * 112;                                      \
        if (tid < 8) {                                                           \
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);                \
        }                                                                        \
        __syncthreads();                                                         \
        const unsigned char* d = (const unsigned char*)(gp + 16) + boff;         \
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8)         \
                        | ((unsigned int)d[2] << 16);                            \
        int base = g * 256 + tid * 8;                                            \
        (acc) += cb_lds[ (pk)        & 7u] * x_rot[base]                         \
               + cb_lds[((pk) >>  3) & 7u] * x_rot[base + 1]                     \
               + cb_lds[((pk) >>  6) & 7u] * x_rot[base + 2]                     \
               + cb_lds[((pk) >>  9) & 7u] * x_rot[base + 3]                     \
               + cb_lds[((pk) >> 12) & 7u] * x_rot[base + 4]                     \
               + cb_lds[((pk) >> 15) & 7u] * x_rot[base + 5]                     \
               + cb_lds[((pk) >> 18) & 7u] * x_rot[base + 6]                     \
               + cb_lds[((pk) >> 21) & 7u] * x_rot[base + 7];                    \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT( quads << 2,       acc0);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1,  acc1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2,  acc2);
    #undef TAIL_LOAD_AND_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
