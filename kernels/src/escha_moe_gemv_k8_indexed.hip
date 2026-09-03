// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Escha-W2 routed-expert GEMVs for the GPU-top-K (indexed) decode path.
//
// # Why these exist when `gemv_q8_0_moe_{gate_up,down}_*_k8_indexed` already do
//
// The two pre-existing indexed Q8_0 MoE kernels are shaped for the ORDINARY
// routed path and neither fits Escha-W2:
//
//   * `gemv_q8_0_moe_gate_up_k8_indexed` broadcasts ONE `x` vector to all k
//     ranks and splits its result into separate `y_gate` / `y_up` buffers.
//     Escha needs the opposite on both counts: every rank has its OWN input
//     (the gate_up input transform folds a PER-EXPERT `rin_eff` row into x, so
//     the k slots are k different vectors), and the output must stay a
//     contiguous `[k, 2*mi]` block because the gate_up OUTPUT transform reads
//     a per-expert `rout_eff` row of width `2*mi` spanning both halves.
//   * `gemv_q8_0_moe_down_residual_scaled_k8_indexed` folds the weighted
//     combine into the GEMV via atomicAdd. Escha cannot: the down result is
//     still in the ROTATED domain at that point and must pass through the
//     down output transform BEFORE it is combined.
//
// So both escha phases want the same thing — per-slot x in, per-slot y out,
// no combine — and that is exactly one kernel used twice.
//
// # Bit-exactness contract (load-bearing)
//
// These are transcriptions of `gemv_q8_0.hip` / `gemv_q8_0_wide.hip` with two
// changes and no others: the weight base comes from
// `expert_ptrs[topk_indices[krank]]` instead of a kernarg, and `x` / `y` are
// offset by `krank`. The accumulate order, the unroll structure, the
// accumulator count and the final reduction are copied verbatim, because the
// escha routed path must produce the SAME numbers on the indexed route as it
// does on the per-expert `GemvFamily::run_auto` route it replaces — the G4
// block gate's tolerances are calibrated against those numbers and a changed
// summation order would move them.
//
// The narrow/wide split mirrors `Gpu::gemv_q8_0`'s own `k <= 1536` rule for
// the same reason: `gemv_q8_0_wide` sums into FOUR interleaved accumulators
// and `gemv_q8_0` into one, so picking the other variant for a given
// projection is a numerical change, not just a performance one. The host
// wrapper re-uses that threshold rather than restating it.
//
// Q8_0 layout: 34 bytes per block of 32 elements (2 B f16 scale + 32 B int8).
// Expert ptrs: [n_exp] weight-base pointers packed as `unsigned long long`.
// No kernarg depends on the routing result, so both replay under hipGraph.

// Narrow variant. Grid: (M, K_TOP, 1). Block: 32.
// x: [K_TOP, K], y: [K_TOP, M].
__launch_bounds__(32, 20)
extern "C" __global__ void escha_gemv_q8_0_moe_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,               // [K_TOP]
    const float* __restrict__ x_batch,                  // [K_TOP, K]
    float* __restrict__ y_batch,                        // [K_TOP, M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const unsigned char* __restrict__ A_q8 =
        reinterpret_cast<const unsigned char*>(expert_ptrs[topk_indices[krank]]);
    const float* __restrict__ x = x_batch + (size_t)krank * K;
    float* __restrict__ y = y_batch + (size_t)krank * M;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    // Process 8 Q8_0 blocks (256 elements) per outer iteration
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;

        // Unrolled: 8 blocks, each: load scale (f16→f32), load byte, FMA
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }

    // Handle remaining blocks (if K is not multiple of 256)
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (tid == 0) y[row] = sum;
}

// Wide variant: 2 rows per block, one row per warp. Grid: (ceil(M/2), K_TOP, 1).
// Block: 64. x: [K_TOP, K], y: [K_TOP, M].
//
// Four interleaved accumulators with the `(acc0 + acc1) + (acc2 + acc3)` fold,
// copied from `gemv_q8_0_wide.hip` — see that file for why the ordering is
// fixed rather than incidental.
extern "C" __global__ void escha_gemv_q8_0_wide_moe_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,               // [K_TOP]
    const float* __restrict__ x_batch,                  // [K_TOP, K]
    float* __restrict__ y_batch,                        // [K_TOP, M]
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;
    const int krank = blockIdx.y;

    const unsigned char* __restrict__ A_q8 =
        reinterpret_cast<const unsigned char*>(expert_ptrs[topk_indices[krank]]);
    const float* __restrict__ x = x_batch + (size_t)krank * K;
    float* __restrict__ y = y_batch + (size_t)krank * M;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = blocks_per_row >> 2;
    const int tail = blocks_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int bi = q << 2;
        const unsigned char* bp0 = row_data + (bi + 0) * 34;
        const unsigned char* bp1 = row_data + (bi + 1) * 34;
        const unsigned char* bp2 = row_data + (bi + 2) * 34;
        const unsigned char* bp3 = row_data + (bi + 3) * 34;

        float d0 = (float)*((const _Float16*)bp0);
        float d1 = (float)*((const _Float16*)bp1);
        float d2 = (float)*((const _Float16*)bp2);
        float d3 = (float)*((const _Float16*)bp3);

        signed char q0 = (signed char)bp0[2 + lane];
        signed char q1 = (signed char)bp1[2 + lane];
        signed char q2 = (signed char)bp2[2 + lane];
        signed char q3 = (signed char)bp3[2 + lane];

        acc0 += d0 * (float)q0 * x[(bi + 0) * 32 + lane];
        acc1 += d1 * (float)q1 * x[(bi + 1) * 32 + lane];
        acc2 += d2 * (float)q2 * x[(bi + 2) * 32 + lane];
        acc3 += d3 * (float)q3 * x[(bi + 3) * 32 + lane];
    }

    // Tail blocks accumulate into acc[bi % 4].
    for (int t = 0; t < tail; t++) {
        const int bi = (quads << 2) + t;
        const unsigned char* bp = row_data + bi * 34;
        float d = (float)*((const _Float16*)bp);
        signed char qv = (signed char)bp[2 + lane];
        float contrib = d * (float)qv * x[bi * 32 + lane];
        if (t == 0) acc0 += contrib;
        else if (t == 1) acc1 += contrib;
        else if (t == 2) acc2 += contrib;
    }

    float sum = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0) y[row] = sum;
}

// Out-of-place F32 -> f16 -> F32 round-trip of the k combine weights.
//
// The escha combine multiplies each expert's contribution by `f16(score)` —
// one of the three load-bearing rounding points of the format (EschaLabs'
// runtime stores the routing weights as f16). On the CPU-top-K route that
// rounding is applied host-side, on the already-downloaded weights. The
// indexed route never downloads them, so it must be done on device.
//
// It is OUT-OF-PLACE on purpose: `topk_weights` is the shared routing buffer
// and other consumers (expert-stats capture, any future reader) must keep
// seeing the unrounded F32 values. `dst` is the escha layer's own [k] scratch.
//
// The exact-zero guard is the same `f2h_rne` workaround the other escha
// kernels carry: on this ROCm 7.2.2/gfx1151 build `__float2half` loses the
// sign of a runtime-produced exact zero. Softmax weights are non-negative so
// the sign cannot matter here, but the guard is free and keeps one rounding
// idiom across the escha kernels rather than two.
extern "C" __global__ void escha_round_weights_f16_rne(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    if (!isfinite(v)) { dst[i] = v; return; }
    __half h;
    if (v == 0.0f) {
        unsigned int bits = __float_as_uint(v);
        h = __ushort_as_half((unsigned short)(bits >> 16));
    } else {
        h = __float2half(v);
    }
    dst[i] = __half2float(h);
}
