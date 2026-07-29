// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Query-tiled Q8_0 flash prefill attention.
//
// Grid:  [ceil(batch_size / BR), n_heads]
// Block: NTHREADS (256) threads.
//
// One workgroup owns BR query rows for one head and streams the K/V cache in
// BC-key tiles staged in LDS in native Q8_0 block form. K/V is therefore read
// once per BR queries rather than once per query, which is the whole point:
// the legacy attention_q8_0_kv_batched launches one workgroup per individual
// query and re-streams the entire prefix for each one.
//
// Because the LDS footprint depends only on BR/BC and never on context length,
// this kernel serves every sequence length — there is no LDS-capacity crossover
// and no occupancy decay as the context grows.
//
// TASK 1 SCOPE: exactly one K/V tile. The tile loop and online softmax arrive
// in Task 2.
//
// LDS layout (all float regions first so the byte regions stay 4-aligned and
// each 34-byte block start stays 2-aligned for the fp16 scale load):
//   s[BR*BC] | m[BR] | l[BR] | corr[BR] | q[BR*head_dim] | kt[BC*bph*34] | vt[...]

#include <hip/hip_runtime.h>

#ifndef BR
#define BR 16
#endif
#ifndef BC
#define BC 32
#endif
#ifndef NTHREADS
#define NTHREADS 256
#endif

extern "C" __global__ __launch_bounds__(NTHREADS) void attention_q8_0_flash_prefill(
    const float* __restrict__ q,               // [batch_size × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache, // [max_seq × n_kv_heads × bph × 34]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                   // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,         // [batch_size]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn
) {
    const int q_tile = blockIdx.x;
    const int h      = blockIdx.y;
    const int tid    = threadIdx.x;
    if (h >= n_heads) return;

    const int q_base = q_tile * BR;
    if (q_base >= batch_size) return;

    const int bph          = head_dim / 32;          // Q8_0 blocks per head
    const int total_blocks = n_kv_heads * bph;
    const int row_stride   = total_blocks * 34;      // bytes per cache position
    const int kv_group     = n_heads / n_kv_heads;
    const int kv_h         = h / kv_group;
    const int kv_blk       = kv_h * bph;
    const int q_stride     = n_heads * head_dim;

    extern __shared__ float smem[];
    float* s_lds    = smem;                       // [BR*BC]
    float* m_lds    = s_lds + BR * BC;            // [BR]
    float* l_lds    = m_lds + BR;                 // [BR]
    float* corr_lds = l_lds + BR;                 // [BR]
    float* q_lds    = corr_lds + BR;              // [BR*head_dim]
    unsigned char* kt = (unsigned char*)(q_lds + BR * head_dim);
    unsigned char* vt = kt + BC * bph * 34;

    // Rows present in this tile (last tile may be ragged).
    const int rows = min(BR, batch_size - q_base);

    // ── Stage Q tile ────────────────────────────────────────────────────
    for (int i = tid; i < rows * head_dim; i += NTHREADS) {
        const int r = i / head_dim;
        const int d = i % head_dim;
        q_lds[r * head_dim + d] = q[(long long)(q_base + r) * q_stride + h * head_dim + d];
    }
    // Per-row running state.
    if (tid < BR) { m_lds[tid] = -1e30f; l_lds[tid] = 0.0f; corr_lds[tid] = 1.0f; }
    __syncthreads();

    // Output accumulator: thread owns row r = tid / lanes_per_row, dims
    // [d0, d0+dpt) where d0 = (tid % lanes_per_row) * dpt.
    const int lanes_per_row = NTHREADS / BR;
    const int acc_r  = tid / lanes_per_row;
    const int acc_c  = tid % lanes_per_row;
    const int dpt    = head_dim / lanes_per_row;
    const int d0     = acc_c * dpt;
    float acc[32];                                   // host asserts dpt <= 32
    for (int i = 0; i < dpt; i++) acc[i] = 0.0f;

    // Longest causal window in this tile decides how many keys to visit.
    int seq_len = 0;
    for (int r = 0; r < rows; r++) seq_len = max(seq_len, positions[q_base + r] + 1);

    // ── Single K/V tile (Task 1 scope) ──────────────────────────────────
    const int t_len = min(BC, seq_len);

    for (int i = tid; i < t_len * bph * 34; i += NTHREADS) {
        const int j  = i / (bph * 34);
        const int bo = i % (bph * 34);
        kt[i] = k_cache[(size_t)j * row_stride + kv_blk * 34 + bo];
        vt[i] = v_cache[(size_t)j * row_stride + kv_blk * 34 + bo];
    }
    __syncthreads();

    // S = Q · K^T with causal mask. BR*BC entries over NTHREADS threads.
    for (int e = tid; e < BR * BC; e += NTHREADS) {
        const int r = e / BC;
        const int j = e % BC;
        float sv = -INFINITY;
        if (r < rows && j < t_len && j <= positions[q_base + r]) {
            float dot = 0.0f;
            const unsigned char* kb = kt + (size_t)j * bph * 34;
            for (int bi = 0; bi < bph; bi++) {
                const float ks = (float)*((const _Float16*)(kb + bi * 34));
                const float* qb = q_lds + r * head_dim + bi * 32;
                for (int c = 0; c < 32; c++)
                    dot += qb[c] * (ks * (float)((signed char)kb[bi * 34 + 2 + c]));
            }
            sv = dot * scale_attn;
        }
        s_lds[e] = sv;
    }
    __syncthreads();

    // Row softmax (one thread per row; Task 2 keeps this shape).
    if (tid < rows) {
        float mx = -1e30f;
        for (int j = 0; j < BC; j++) mx = fmaxf(mx, s_lds[tid * BC + j]);
        float sum = 0.0f;
        for (int j = 0; j < BC; j++) {
            const float e = (s_lds[tid * BC + j] == -INFINITY) ? 0.0f : __expf(s_lds[tid * BC + j] - mx);
            s_lds[tid * BC + j] = e;
            sum += e;
        }
        m_lds[tid] = mx;
        l_lds[tid] = sum;
    }
    __syncthreads();

    // O += P · V
    if (acc_r < rows) {
        for (int j = 0; j < t_len; j++) {
            const float p = s_lds[acc_r * BC + j];
            if (p == 0.0f) continue;
            const unsigned char* vb = vt + (size_t)j * bph * 34;
            for (int i = 0; i < dpt; i++) {
                const int d  = d0 + i;
                const int bi = d >> 5;
                const int bj = d & 31;
                const float vs = (float)*((const _Float16*)(vb + bi * 34));
                acc[i] += p * (vs * (float)((signed char)vb[bi * 34 + 2 + bj]));
            }
        }
        float* o = out + (long long)(q_base + acc_r) * q_stride + h * head_dim;
        const float inv = 1.0f / l_lds[acc_r];
        for (int i = 0; i < dpt; i++) o[d0 + i] = acc[i] * inv;
    }
}
