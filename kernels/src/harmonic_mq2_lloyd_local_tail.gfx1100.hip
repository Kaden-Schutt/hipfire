// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Exact-gfx1100 DS4 harmonic local-expert tail. Every dispatch has fixed
// top-k=6 geometry so it can live behind the route checkpoint in a retained
// batch. `local_count` is produced by the preceding owner-local stage kernel;
// inactive workgroups return before touching weight or activation storage.

__launch_bounds__(32, 20)
extern "C" __global__ void harmonic_mq2_lloyd_gate_up_gfx1100(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned int* __restrict__ local_expert_ids,
    const float* __restrict__ x_rot,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    const unsigned int* __restrict__ local_count,
    int M,
    int K
) {
    const int row = blockIdx.x;
    const int krank = blockIdx.y;
    if (row >= M || krank >= *local_count) return;
    const int tid = threadIdx.x;

    const unsigned int expert_id = local_expert_ids[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;
    __shared__ float cb_lds[16];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define HARMONIC_GATE_DOT(pk, lds, base, acc) do {                    \
        (acc) += cb_lds[(lds) + ((pk) & 3u)] * x_rot[(base)]             \
              + cb_lds[(lds) + (((pk) >> 2) & 3u)] * x_rot[(base) + 1]  \
              + cb_lds[(lds) + (((pk) >> 4) & 3u)] * x_rot[(base) + 2]  \
              + cb_lds[(lds) + (((pk) >> 6) & 3u)] * x_rot[(base) + 3]  \
              + cb_lds[(lds) + (((pk) >> 8) & 3u)] * x_rot[(base) + 4]  \
              + cb_lds[(lds) + (((pk) >> 10) & 3u)] * x_rot[(base) + 5] \
              + cb_lds[(lds) + (((pk) >> 12) & 3u)] * x_rot[(base) + 6] \
              + cb_lds[(lds) + (((pk) >> 14) & 3u)] * x_rot[(base) + 7];\
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        if (tid < 16) {
            const __half* cb_h =
                reinterpret_cast<const __half*>(gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(cb_h[tid & 3]);
        }
        __syncthreads();

        const unsigned char* d0 = reinterpret_cast<const unsigned char*>(gp0 + 8) + boff;
        const unsigned char* d1 = reinterpret_cast<const unsigned char*>(gp1 + 8) + boff;
        const unsigned char* d2 = reinterpret_cast<const unsigned char*>(gp2 + 8) + boff;
        const unsigned char* d3 = reinterpret_cast<const unsigned char*>(gp3 + 8) + boff;
        const unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8);
        const unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8);
        const unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8);
        const unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8);
        const int base = g * 256 + tid * 8;
        HARMONIC_GATE_DOT(pk0, 0, base, acc0);
        HARMONIC_GATE_DOT(pk1, 4, base + 256, acc1);
        HARMONIC_GATE_DOT(pk2, 8, base + 512, acc2);
        HARMONIC_GATE_DOT(pk3, 12, base + 768, acc3);
    }

    const int tail_start = quads << 2;
    for (int g = tail_start; g < tail_start + tail; g++) {
        const char* gp = row_ptr + g * 72;
        if (tid < 4) {
            cb_lds[tid] = __half2float(reinterpret_cast<const __half*>(gp)[tid]);
        }
        __syncthreads();
        const unsigned char* d = reinterpret_cast<const unsigned char*>(gp + 8) + boff;
        const unsigned int pk = (unsigned int)d[0] | ((unsigned int)d[1] << 8);
        const int base = g * 256 + tid * 8;
        HARMONIC_GATE_DOT(pk, 0, base, acc0);
    }
    #undef HARMONIC_GATE_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[(long long)krank * mi + row] = acc;
        } else {
            y_up[(long long)krank * mi + row - mi] = acc;
        }
    }
}

extern "C" __global__ void harmonic_silu_mul_clamp_gfx1100(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const unsigned int* __restrict__ local_count,
    int n,
    float swiglu_limit
) {
    const int batch = blockIdx.y;
    if (batch >= *local_count) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long offset = (long long)batch * n + i;
    float g = gate[offset];
    float u = up[offset];
    if (g > swiglu_limit) g = swiglu_limit;
    if (u > swiglu_limit) u = swiglu_limit;
    if (u < -swiglu_limit) u = -swiglu_limit;
    const float silu_g = g / (1.0f + __expf(-g));
    out[offset] = silu_g * u;
}

extern "C" __launch_bounds__(32, 16)
__global__ void harmonic_mq_rotate_x_gfx1100(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    const unsigned int* __restrict__ local_count,
    int K
) {
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    const int linear_group = blockIdx.x;
    const int batch = linear_group / groups_total;
    if (batch >= *local_count) return;
    const int group = linear_group - batch * groups_total;
    const long long batch_off = (long long)batch * K;
    const float* x_in_b = x_in + batch_off;
    float* x_out_b = x_out + batch_off;
    const int d0 = tid * 8;
    const int base = group * 256 + d0;

    float v0 = x_in_b[base] * signs1[d0];
    float v1 = x_in_b[base + 1] * signs1[d0 + 1];
    float v2 = x_in_b[base + 2] * signs1[d0 + 2];
    float v3 = x_in_b[base + 3] * signs1[d0 + 3];
    float v4 = x_in_b[base + 4] * signs1[d0 + 4];
    float v5 = x_in_b[base + 5] * signs1[d0 + 5];
    float v6 = x_in_b[base + 6] * signs1[d0 + 6];
    float v7 = x_in_b[base + 7] * signs1[d0 + 7];

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    #define HARMONIC_HBFLY(v, pat, stride) do {                              \
        float peer = __int_as_float(                                         \
            __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat)));          \
        if (tid & (stride)) { (v) = peer - (v); } else { (v) = (v) + peer; }\
    } while (0)
    #define HARMONIC_HBFLY8(pat, stride) \
        HARMONIC_HBFLY(v0,pat,stride); HARMONIC_HBFLY(v1,pat,stride); \
        HARMONIC_HBFLY(v2,pat,stride); HARMONIC_HBFLY(v3,pat,stride); \
        HARMONIC_HBFLY(v4,pat,stride); HARMONIC_HBFLY(v5,pat,stride); \
        HARMONIC_HBFLY(v6,pat,stride); HARMONIC_HBFLY(v7,pat,stride)
    HARMONIC_HBFLY8(0x041F, 1);
    HARMONIC_HBFLY8(0x081F, 2);
    HARMONIC_HBFLY8(0x101F, 4);
    HARMONIC_HBFLY8(0x201F, 8);
    HARMONIC_HBFLY8(0x401F, 16);
    #undef HARMONIC_HBFLY8
    #undef HARMONIC_HBFLY

    const float scale = 0.0625f;
    x_out_b[base] = v0 * scale * signs2[d0];
    x_out_b[base + 1] = v1 * scale * signs2[d0 + 1];
    x_out_b[base + 2] = v2 * scale * signs2[d0 + 2];
    x_out_b[base + 3] = v3 * scale * signs2[d0 + 3];
    x_out_b[base + 4] = v4 * scale * signs2[d0 + 4];
    x_out_b[base + 5] = v5 * scale * signs2[d0 + 5];
    x_out_b[base + 6] = v6 * scale * signs2[d0 + 6];
    x_out_b[base + 7] = v7 * scale * signs2[d0 + 7];
}

__launch_bounds__(32, 20)
extern "C" __global__ void harmonic_mq2_lloyd_down_gfx1100(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned int* __restrict__ local_expert_ids,
    const float* __restrict__ rot_batch,
    float* __restrict__ expert_outputs,
    const unsigned int* __restrict__ local_count,
    int M,
    int K
) {
    const int row = blockIdx.x;
    const int krank = blockIdx.y;
    if (row >= M || krank >= *local_count) return;
    const int tid = threadIdx.x;

    const unsigned int expert_id = local_expert_ids[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = rot_batch + (long long)krank * K;
    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = tid * 2;
    const int xoff = tid * 8;
    __shared__ float cb_lds[16];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    #define HARMONIC_DOWN_DOT(gptr, lds, base, acc) do {                 \
        const unsigned char* data =                                     \
            reinterpret_cast<const unsigned char*>((gptr) + 8);         \
        const unsigned char b0 = data[boff];                             \
        const unsigned char b1 = data[boff + 1];                         \
        (acc) += cb_lds[(lds) + (b0 & 3)] * x[(base)]                    \
              + cb_lds[(lds) + ((b0 >> 2) & 3)] * x[(base) + 1]         \
              + cb_lds[(lds) + ((b0 >> 4) & 3)] * x[(base) + 2]         \
              + cb_lds[(lds) + (b0 >> 6)] * x[(base) + 3]               \
              + cb_lds[(lds) + (b1 & 3)] * x[(base) + 4]                \
              + cb_lds[(lds) + ((b1 >> 2) & 3)] * x[(base) + 5]         \
              + cb_lds[(lds) + ((b1 >> 4) & 3)] * x[(base) + 6]         \
              + cb_lds[(lds) + (b1 >> 6)] * x[(base) + 7];              \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 72;
        const char* gp1 = gp0 + 72;
        const char* gp2 = gp1 + 72;
        const char* gp3 = gp2 + 72;
        if (tid < 16) {
            const __half* codebook = reinterpret_cast<const __half*>(
                gp0 + (long long)(tid >> 2) * 72);
            cb_lds[tid] = __half2float(codebook[tid & 3]);
        }
        __syncthreads();
        const int base = g * 256 + xoff;
        HARMONIC_DOWN_DOT(gp0, 0, base, acc0);
        HARMONIC_DOWN_DOT(gp1, 4, base + 256, acc1);
        HARMONIC_DOWN_DOT(gp2, 8, base + 512, acc2);
        HARMONIC_DOWN_DOT(gp3, 12, base + 768, acc3);
    }
    const int tail_start = quads << 2;
    for (int g = tail_start; g < tail_start + tail; g++) {
        const char* gp = row_ptr + g * 72;
        if (tid < 4) {
            cb_lds[tid] = __half2float(reinterpret_cast<const __half*>(gp)[tid]);
        }
        __syncthreads();
        const int base = g * 256 + xoff;
        HARMONIC_DOWN_DOT(gp, 0, base, acc0);
    }
    #undef HARMONIC_DOWN_DOT

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }
    if (tid == 0) {
        expert_outputs[(long long)krank * M + row] = acc;
    }
}
