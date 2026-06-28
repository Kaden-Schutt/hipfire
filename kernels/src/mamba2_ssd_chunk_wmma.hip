// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Mamba-2 SSD chunked prefill — bf16-WMMA, f32 accumulate. RDNA3+ (gfx11xx/12xx)
// wave32. The matrix-core path that replaces the per-channel scalar dot products
// of mamba2_ssd_seq.hip with 16x16x16 WMMA matmuls. Correctness floor + GPU
// oracle: mamba2_ssd_chunk_f32.hip. See docs/plans/
// 2026-06-28-mamba2-ssd-wmma-prefill.md.
//
// MULTI-WAVE per head (WAVES wave32 per block, one block per head): the SSD chunk
// is memory/latency-bound, so the per-head stage-2/3/4 tile loops are split across
// the block's waves (by warp_id) to raise occupancy and hide latency, and h_in is
// staged into LDS once per chunk (coalesced) so stage 3 stops doing scattered
// per-element global loads inside the WMMA loop. Chunks are serial (state carry).
//
// Stages (per head, per chunk): S_t = sum_{r<=t} dt_r*A, A=-exp(A_log):
//   1. CB[t][s] = sum_n C[t][n]*B[s][n]                      (Gram, WMMA; warp 0)
//   2. M[t][s]  = (s<=t) ? exp(S_t-S_s)*dt_s*CB[t][s] : 0    (LDS, VALU)
//      y[t][p]  = sum_s M[t][s]*x[s][p] + D*x[t][p]          (intra, WMMA)
//   3. y[t][p] += exp(S_t) * sum_n C[t][n]*h_in[p][n]        (inter, WMMA; LDS h_in)
//   4. h_out[p][n] = exp(S_{L-1})*h_in[p][n]
//                  + sum_s (exp(S_{L-1}-S_s)*dt_s*x[s][p])*B[s][n]  (carry, WMMA)
//
// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,c) = A @ B^T + C for 16x16 bf16
// A,B and 16x16 f32 C/D. Lane (t&15) holds row t of A and B; acc[j] at lane t maps
// to D[2*j + (t>>4)][t&15] (see gemm_bf16_x_bf16_wmma.hip). Dims that contract over
// the chunk axis use LDS transposes (x^T, B^T).

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef short __attribute__((ext_vector_type(16))) b16x16_t;
typedef float __attribute__((ext_vector_type(8)))  float8_t;

#define WT    16 // WMMA tile edge
#define WAVES 16 // wave32s per head-block

__device__ __forceinline__ unsigned short f2bf16(float f) {
    unsigned int x = __float_as_uint(f);
    unsigned int r = x + 0x00007fffu + ((x >> 16) & 1u); // round to nearest even
    return (unsigned short)(r >> 16);
}

__device__ __forceinline__ float softplus_stable_wmma(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__launch_bounds__(32 * WAVES)
extern "C" __global__ void mamba2_ssd_chunk_wmma(
    float* __restrict__ y,             // [seq * num_heads * head_dim]
    float* __restrict__ state,         // [num_heads * head_dim * state_size] (in/out)
    const float* __restrict__ x,       // [seq * num_heads * head_dim]
    const float* __restrict__ B,       // [seq * n_groups * state_size]
    const float* __restrict__ C,       // [seq * n_groups * state_size]
    const float* __restrict__ dt_raw,  // [seq * num_heads]
    const float* __restrict__ A_log,   // [num_heads]
    const float* __restrict__ D,       // [num_heads]
    const float* __restrict__ dt_bias, // [num_heads]
    int seq,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    float dt_min,
    float dt_max,
    int chunk_size
) {
    const int head = blockIdx.x;
    if (head >= num_heads) return;
    const int tot = threadIdx.x;        // 0..32*WAVES-1
    const int warp = tot >> 5;          // 0..WAVES-1
    const int lane16 = tot & 15;
    const int hi = (tot & 31) >> 4;     // 0/1 within the wave
    const int nthreads = blockDim.x;

    const int N = state_size;
    const int P = head_dim;
    const int ND = ((N + 15) / 16) * 16;       // N padded to a WMMA tile
    const int nptile = (P + 15) / 16;          // head_dim tiles
    const int nntile = ND / 16;                // state tiles
    int step = chunk_size < WT ? chunk_size : WT;
    if (step < 1) step = 1;

    const int grp = head / (num_heads / n_groups);
    const float a = -expf(A_log[head]);
    const float Dh = D[head];
    const float dtb = dt_bias[head];
    const int xd = num_heads * head_dim;
    const int bd = n_groups * state_size;

    // LDS tiles (N,P <= 128).
    __shared__ unsigned short Cf[WT * 128];    // C[t][n], stride ND
    __shared__ unsigned short Bf[WT * 128];    // B[s][n], stride ND
    __shared__ unsigned short XT[128 * WT];    // x^T[p][s], stride 16
    __shared__ unsigned short BT[128 * WT];    // B^T[n][s], stride 16
    __shared__ unsigned short coefT[128 * WT]; // coef^T[p][s] = w[s]*x[s][p]
    __shared__ unsigned short Mbf[WT * WT];    // M[t][s], stride 16
    __shared__ float CB[WT * WT];              // gram, stride 16
    __shared__ float scum[WT], dtl[WT], wl[WT];

    // SSM state, RESIDENT in LDS across the whole sequence (f32, [P][ND]): read
    // global state once here, carry it in LDS through every chunk, write once at
    // the end. This collapses the per-chunk full-state global read+write (the
    // memory bottleneck) into a single round-trip, while keeping the f32 carry
    // accumulation identical to the global path. Dynamic shared mem, sized
    // P*ND*4 by the launch.
    extern __shared__ float St[]; // h state, [p*ND + n]
    for (int i = tot; i < P * ND; i += nthreads) {
        int p = i / ND, n = i % ND;
        St[i] = (n < N) ? state[((size_t)head * P + p) * N + n] : 0.0f;
    }
    __syncthreads();

    for (int t0 = 0; t0 < seq; t0 += step) {
        int L = step;
        if (L > seq - t0) L = seq - t0;

        // cumulative decay (lane 0)
        if (tot == 0) {
            float run = 0.0f;
            for (int t = 0; t < L; ++t) {
                float v = softplus_stable_wmma(dt_raw[(size_t)(t0 + t) * num_heads + head] + dtb);
                v = fminf(fmaxf(v, dt_min), dt_max);
                dtl[t] = v;
                run += v * a;
                scum[t] = run;
            }
            float s_end = scum[L - 1];
            for (int s = 0; s < L; ++s) wl[s] = expf(s_end - scum[s]) * dtl[s];
        }

        // stage bf16 tiles (all waves cooperate). h_in is the LDS-resident St
        // (read in stage 3), so no per-chunk h_in staging.
        for (int i = tot; i < WT * ND; i += nthreads) {
            int t = i / ND, n = i % ND;
            Cf[i] = (t < L && n < N) ? f2bf16(C[(size_t)(t0 + t) * bd + grp * N + n]) : 0;
            Bf[i] = (t < L && n < N) ? f2bf16(B[(size_t)(t0 + t) * bd + grp * N + n]) : 0;
        }
        __syncthreads(); // wl[] ready for coefT; tiles for matmuls
        for (int i = tot; i < 128 * WT; i += nthreads) {
            int p = i / WT, s = i % WT;
            float xv = (p < P && s < L) ? x[(size_t)(t0 + s) * xd + head * P + p] : 0.0f;
            XT[i] = f2bf16(xv);
            int n = p; // B^T[n][s]
            BT[i] = (n < N && s < L) ? f2bf16(B[(size_t)(t0 + s) * bd + grp * N + n]) : 0;
            coefT[i] = (p < P && s < L) ? f2bf16(wl[s] * xv) : 0;
        }
        __syncthreads();

        // stage 1: CB gram = C @ B^T (warp 0 only — one 16x16 output tile).
        if (warp == 0) {
            float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int n0 = 0; n0 < ND; n0 += WT) {
                b16x16_t af = *(const b16x16_t*)(Cf + lane16 * ND + n0);
                b16x16_t bf = *(const b16x16_t*)(Bf + lane16 * ND + n0);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc);
            }
            for (int j = 0; j < 8; ++j) CB[(2 * j + hi) * WT + lane16] = acc[j];
        }
        __syncthreads();

        // stage 2 build M (lower-tri exp/dt weighted), cooperative.
        for (int i = tot; i < WT * WT; i += nthreads) {
            int t = i / WT, s = i % WT;
            float v = 0.0f;
            if (t < L && s < L && s <= t) v = expf(scum[t] - scum[s]) * dtl[s] * CB[t * WT + s];
            Mbf[i] = f2bf16(v);
        }
        __syncthreads();

        // stage 2 matmul + stage 3 inter-chunk: p-tiles split across waves.
        // Both use the same p-tile→warp mapping, so y[t][p] is touched by one warp.
        for (int pt = warp; pt < nptile; pt += WAVES) {
            const int p0 = pt * WT;
            // stage 2: y = M @ x + D*x
            float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
            {
                b16x16_t af = *(const b16x16_t*)(Mbf + lane16 * WT);
                b16x16_t bf = *(const b16x16_t*)(XT + (p0 + lane16) * WT);
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc2);
            }
            // stage 3: + exp(S_t) * (C @ h_in)  (contract over n; h_in = resident
            // St, converted f32->bf16 lane-side for the WMMA operand)
            float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
            const int p_s3 = p0 + lane16;
            for (int n0 = 0; n0 < ND; n0 += WT) {
                b16x16_t af = *(const b16x16_t*)(Cf + lane16 * ND + n0);
                b16x16_t bf;
                #pragma unroll
                for (int k = 0; k < WT; ++k) {
                    int n = n0 + k;
                    bf[k] = (p_s3 < P && n < N) ? (short)f2bf16(St[p_s3 * ND + n]) : (short)0;
                }
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc3);
            }
            for (int j = 0; j < 8; ++j) {
                int t = 2 * j + hi;
                int p = p0 + lane16;
                if (t < L && p < P) {
                    float xt = x[(size_t)(t0 + t) * xd + head * P + p];
                    y[(size_t)(t0 + t) * xd + head * P + p] =
                        acc2[j] + Dh * xt + expf(scum[t]) * acc3[j];
                }
            }
        }
        __syncthreads(); // h_in (St) fully read before stage 4 updates it

        // stage 4: h_out[p][n] = exp(S_end)*h_in + coef^T @ B^T, in place in the
        // resident LDS state St (f32 carry — no global state I/O per chunk).
        // (p,n) tiles split across waves; each St elem owned by one warp.
        float decay_end = expf(scum[L - 1]);
        const int ntiles = nptile * nntile;
        for (int tile = warp; tile < ntiles; tile += WAVES) {
            const int p0 = (tile / nntile) * WT;
            const int n0 = (tile % nntile) * WT;
            float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
            b16x16_t af = *(const b16x16_t*)(coefT + (p0 + lane16) * WT);
            b16x16_t bf = *(const b16x16_t*)(BT + (n0 + lane16) * WT);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc);
            for (int j = 0; j < 8; ++j) {
                int p = p0 + 2 * j + hi;
                int n = n0 + lane16;
                if (p < P && n < N) {
                    int sidx = p * ND + n;
                    St[sidx] = decay_end * St[sidx] + acc[j];
                }
            }
        }
        __syncthreads(); // St updates visible + LDS tiles free before next chunk
    }

    // write the resident state back to global once, post-sequence (decode hand-off)
    for (int i = tot; i < P * ND; i += nthreads) {
        int p = i / ND, n = i % ND;
        if (n < N) state[((size_t)head * P + p) * N + n] = St[i];
    }
}
