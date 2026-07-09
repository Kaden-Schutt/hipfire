// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Index-aware fused MoE down GEMV with scaled residual for MQ3-Lloyd
// (3-bit + per-block 8-entry fp16 codebook, 112 B/group). Sibling of
// gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed — same expert-ptrs
// dispatch + atomicAdd residual, but the per-group decode is the 8-entry
// codebook lookup over 3-bit indices.
//
// `rot_batch[krank * K .. krank * K + K]` must be FWHT-pre-rotated for the
// down projection (each top-k expert has its own post-gate-up activation;
// each needs an independent rotation before this kernel sees it).
//
// atomicAdd into x_residual accumulates topk_weights[krank] * (W_down · rot_batch[krank])
// across all top-k experts and rows. One launch replaces k_top per-expert
// scaled-gemv-residual launches.
//
// Block layout (112 B / 256 weights): see
// gemv_mq3g256_lloyd_moe_gate_up_indexed.hip for design notes.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,              // [k_top]
    const float* __restrict__ topk_weights,              // [k_top]
    const float* __restrict__ rot_batch,                 // [k_top × K] FWHT-rotated
    float*       __restrict__ x_residual,                // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 112;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 3;

    // 32-slot LDS codebook: 4 groups × 8 entries.
    __shared__ float cb_lds[32];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

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

        const unsigned char* d0 = (const unsigned char*)(gp0 + 16) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 16) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 16) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 16) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8) | ((unsigned int)d0[2] << 16);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8) | ((unsigned int)d1[2] << 16);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8) | ((unsigned int)d2[2] << 16);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8) | ((unsigned int)d3[2] << 16);

        int base = g * 256 + tid * 8;

        #define DOG3_LDS(pk, lds, b, a)                               \
            (a) += cb_lds[(lds) + ( (pk)        & 7u)] * x[(b)]       \
                 + cb_lds[(lds) + (((pk) >>  3) & 7u)] * x[(b) + 1]   \
                 + cb_lds[(lds) + (((pk) >>  6) & 7u)] * x[(b) + 2]   \
                 + cb_lds[(lds) + (((pk) >>  9) & 7u)] * x[(b) + 3]   \
                 + cb_lds[(lds) + (((pk) >> 12) & 7u)] * x[(b) + 4]   \
                 + cb_lds[(lds) + (((pk) >> 15) & 7u)] * x[(b) + 5]   \
                 + cb_lds[(lds) + (((pk) >> 18) & 7u)] * x[(b) + 6]   \
                 + cb_lds[(lds) + (((pk) >> 21) & 7u)] * x[(b) + 7]

        DOG3_LDS(pk0,  0, base,        acc0);
        DOG3_LDS(pk1,  8, base + 256,  acc1);
        DOG3_LDS(pk2, 16, base + 512,  acc2);
        DOG3_LDS(pk3, 24, base + 768,  acc3);
        #undef DOG3_LDS
    }

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
        (acc) += cb_lds[ (pk)        & 7u] * x[base]                             \
               + cb_lds[((pk) >>  3) & 7u] * x[base + 1]                         \
               + cb_lds[((pk) >>  6) & 7u] * x[base + 2]                         \
               + cb_lds[((pk) >>  9) & 7u] * x[base + 3]                         \
               + cb_lds[((pk) >> 12) & 7u] * x[base + 4]                         \
               + cb_lds[((pk) >> 15) & 7u] * x[base + 5]                         \
               + cb_lds[((pk) >> 18) & 7u] * x[base + 6]                         \
               + cb_lds[((pk) >> 21) & 7u] * x[base + 7];                        \
    } while (0)

    if (tail >= 1) TAIL_LOAD_AND_DOT( quads << 2,       acc0);
    if (tail >= 2) TAIL_LOAD_AND_DOT((quads << 2) + 1,  acc1);
    if (tail >= 3) TAIL_LOAD_AND_DOT((quads << 2) + 2,  acc2);
    #undef TAIL_LOAD_AND_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const float scale = topk_weights[krank];
        atomicAdd(&x_residual[row], scale * acc);
    }
}

// ---------------------------------------------------------------------------
// Reproducible int64-accumulating variant
//
// Same codebook decode as above, but each 256-group's FP warp-reduced partial
// is converted to i64 (S=2^24 fixed-point) before cross-group accumulation.
// The topk_w is folded at GROUP level (not expert level) for partition
// invariance: `round(topk_w * partial_g * S)` is accumulated per group, then
// written via integer atomicAdd.  This makes the result partition-invariant
// under any K-split where groups are 256-aligned:
//   full_i64[row] == half_a_i64[row] + half_b_i64[row]   (bit-exact)
// because (a) each group is wholly on one rank (K 256-aligned), (b) the
// within-group warp reduce is FP (same as before, independent per group),
// (c) topk_w is folded per group (so each group contributes exactly the same
//     bits regardless of which rank holds it), and
// (d) atomicAdd on unsigned long long is integer (order-independent).
//
// NOTE: folding topk_w AFTER summing all groups (i.e. round(topk_w * acc_i64))
// is NOT partition-invariant because round(w*(a+b)) != round(w*a)+round(w*b).
// The per-group fold avoids this: each rank accumulates only its own groups and
// the integer sum across ranks equals the single-rank result.
//
// Call `moe_i64_residual_to_f32` after all expert launches to convert back.
//
// residual_i64  must be zero-initialised by the caller before the launch.
// ---------------------------------------------------------------------------

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq3g256_lloyd_moe_down_residual_i64_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int*   __restrict__ topk_indices,              // [k_top]
    const float* __restrict__ topk_weights,              // [k_top]
    const float* __restrict__ rot_batch,                 // [k_top × K] FWHT-rotated
    unsigned long long* __restrict__ residual_i64,       // [M] int64, S-scaled, init=0
    int M, int K
) {
    const int row   = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid   = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 112;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 3;

    __shared__ float cb_lds[8]; // single-group codebook in LDS

    long long acc_i64 = 0LL;

    // Scale: S = 2^24.  topk_w read once per thread block (same for all groups).
    const float S = (float)(1 << 24);
    const float topk_w = topk_weights[krank];

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + g * 112;

        // Cooperative load: 8 threads load the 8 fp16 codebook entries.
        if (tid < 8) {
            cb_lds[tid] = __half2float(((const __half*)gp)[tid]);
        }
        __syncthreads();

        const unsigned char* d = (const unsigned char*)(gp + 16) + boff;
        unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8)
                        | ((unsigned int)d[2] << 16);
        int base = g * 256 + tid * 8;

        float partial =
              cb_lds[ (pk)        & 7u] * x[base]
            + cb_lds[((pk) >>  3) & 7u] * x[base + 1]
            + cb_lds[((pk) >>  6) & 7u] * x[base + 2]
            + cb_lds[((pk) >>  9) & 7u] * x[base + 3]
            + cb_lds[((pk) >> 12) & 7u] * x[base + 4]
            + cb_lds[((pk) >> 15) & 7u] * x[base + 5]
            + cb_lds[((pk) >> 18) & 7u] * x[base + 6]
            + cb_lds[((pk) >> 21) & 7u] * x[base + 7];

        // FP warp reduce across 32 threads → group_partial on lane 0
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down(partial, offset);

        // Lane 0: fold topk_w at GROUP level and accumulate in int64.
        // round(topk_w * partial * S) per group — partition-invariant because
        // each group is wholly on one TP rank; no cross-rank rounding error.
        if (tid == 0) {
            acc_i64 += (long long)llroundf(topk_w * partial * S);
        }
        __syncthreads(); // guard LDS reload for next iteration
    }

    // Write accumulated i64 (topk_w already folded per group) via atomicAdd.
    // unsigned long long atomicAdd reinterprets signed i64 bits (two's-complement
    // addition is identical for both types).
    if (tid == 0) {
        atomicAdd(&residual_i64[row], (unsigned long long)acc_i64);
    }
}

// Convert int64 S-scaled residual buffer to f32 output.
// Launch with a 1-D grid: gridDim.x × blockDim.x >= n elements.
extern "C" __global__ void moe_i64_residual_to_f32(
    const unsigned long long* __restrict__ residual_i64, // [n] int64 as ull
    float*                    __restrict__ out_f32,       // [n]
    int n,
    float inv_S                                           // = 1.0f / (1<<24)
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Reinterpret as signed i64
    long long v;
    memcpy(&v, &residual_i64[i], sizeof(v));
    out_f32[i] = (float)((double)v * (double)inv_S);
}
