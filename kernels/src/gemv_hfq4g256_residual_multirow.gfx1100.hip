#include <hip/hip_runtime.h>

// gfx1100 (RDNA3) HFQ4-G256 multi-row GEMV with fused residual add:
// y[row_base + r] += A[row_base + r] · x  for r in [0, R).
//
// Same inner loop and tail handling as gemv_hfq4g256_multirow.gfx1100.hip —
// only the final write differs (+= instead of =). Used for wo/w_down in
// the Qwen3.5 forward path to elide the add_inplace_f32 follow-up launch.
//
// STATUS: opt-in experimental (HIPFIRE_GEMV_ROWS={2,4,8}). Empirically
// slower than single-row on all tested matrix sizes. See
// gemv_hfq4g256_multirow.gfx1100.hip for the full measurement table and
// explanation. Tail-handling bug background: commit 5302926.

template<int R>
__device__ __forceinline__ void gemv_hfq4g256_residual_multirow_body(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const int boff = tid * 4;

    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
    }

    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base = g * 256 + tid * 8;

        const float x00 = x[base + 0],   x01 = x[base + 1],
                    x02 = x[base + 2],   x03 = x[base + 3],
                    x04 = x[base + 4],   x05 = x[base + 5],
                    x06 = x[base + 6],   x07 = x[base + 7];
        const float x10 = x[base + 256 + 0], x11 = x[base + 256 + 1],
                    x12 = x[base + 256 + 2], x13 = x[base + 256 + 3],
                    x14 = x[base + 256 + 4], x15 = x[base + 256 + 5],
                    x16 = x[base + 256 + 6], x17 = x[base + 256 + 7];
        const float x20 = x[base + 512 + 0], x21 = x[base + 512 + 1],
                    x22 = x[base + 512 + 2], x23 = x[base + 512 + 3],
                    x24 = x[base + 512 + 4], x25 = x[base + 512 + 5],
                    x26 = x[base + 512 + 6], x27 = x[base + 512 + 7];
        const float x30 = x[base + 768 + 0], x31 = x[base + 768 + 1],
                    x32 = x[base + 768 + 2], x33 = x[base + 768 + 3],
                    x34 = x[base + 768 + 4], x35 = x[base + 768 + 5],
                    x36 = x[base + 768 + 6], x37 = x[base + 768 + 7];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp0 = row_ptrs[r] + g * 136;
            const char* gp1 = gp0 + 136;
            const char* gp2 = gp1 + 136;
            const char* gp3 = gp2 + 136;

            float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
            float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
            float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
            float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
            float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
            float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
            float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
            float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));

            unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

            acc[r][0] += (sc0 * (float)((pk0 >>  0) & 0xFu) + zp0) * x00
                       + (sc0 * (float)((pk0 >>  4) & 0xFu) + zp0) * x01
                       + (sc0 * (float)((pk0 >>  8) & 0xFu) + zp0) * x02
                       + (sc0 * (float)((pk0 >> 12) & 0xFu) + zp0) * x03
                       + (sc0 * (float)((pk0 >> 16) & 0xFu) + zp0) * x04
                       + (sc0 * (float)((pk0 >> 20) & 0xFu) + zp0) * x05
                       + (sc0 * (float)((pk0 >> 24) & 0xFu) + zp0) * x06
                       + (sc0 * (float)((pk0 >> 28) & 0xFu) + zp0) * x07;
            acc[r][1] += (sc1 * (float)((pk1 >>  0) & 0xFu) + zp1) * x10
                       + (sc1 * (float)((pk1 >>  4) & 0xFu) + zp1) * x11
                       + (sc1 * (float)((pk1 >>  8) & 0xFu) + zp1) * x12
                       + (sc1 * (float)((pk1 >> 12) & 0xFu) + zp1) * x13
                       + (sc1 * (float)((pk1 >> 16) & 0xFu) + zp1) * x14
                       + (sc1 * (float)((pk1 >> 20) & 0xFu) + zp1) * x15
                       + (sc1 * (float)((pk1 >> 24) & 0xFu) + zp1) * x16
                       + (sc1 * (float)((pk1 >> 28) & 0xFu) + zp1) * x17;
            acc[r][2] += (sc2 * (float)((pk2 >>  0) & 0xFu) + zp2) * x20
                       + (sc2 * (float)((pk2 >>  4) & 0xFu) + zp2) * x21
                       + (sc2 * (float)((pk2 >>  8) & 0xFu) + zp2) * x22
                       + (sc2 * (float)((pk2 >> 12) & 0xFu) + zp2) * x23
                       + (sc2 * (float)((pk2 >> 16) & 0xFu) + zp2) * x24
                       + (sc2 * (float)((pk2 >> 20) & 0xFu) + zp2) * x25
                       + (sc2 * (float)((pk2 >> 24) & 0xFu) + zp2) * x26
                       + (sc2 * (float)((pk2 >> 28) & 0xFu) + zp2) * x27;
            acc[r][3] += (sc3 * (float)((pk3 >>  0) & 0xFu) + zp3) * x30
                       + (sc3 * (float)((pk3 >>  4) & 0xFu) + zp3) * x31
                       + (sc3 * (float)((pk3 >>  8) & 0xFu) + zp3) * x32
                       + (sc3 * (float)((pk3 >> 12) & 0xFu) + zp3) * x33
                       + (sc3 * (float)((pk3 >> 16) & 0xFu) + zp3) * x34
                       + (sc3 * (float)((pk3 >> 20) & 0xFu) + zp3) * x35
                       + (sc3 * (float)((pk3 >> 24) & 0xFu) + zp3) * x36
                       + (sc3 * (float)((pk3 >> 28) & 0xFu) + zp3) * x37;
        }
    }

    #define TAIL_LOAD_X(b) \
        const float txv0 = x[(b) + 0], txv1 = x[(b) + 1], \
                    txv2 = x[(b) + 2], txv3 = x[(b) + 3], \
                    txv4 = x[(b) + 4], txv5 = x[(b) + 5], \
                    txv6 = x[(b) + 6], txv7 = x[(b) + 7]
    #define TAIL_DOG(pk, sc, zp, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * txv0  \
             + (sc * (float)(((pk) >>  4) & 0xFu) + zp) * txv1 \
             + (sc * (float)(((pk) >>  8) & 0xFu) + zp) * txv2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * txv3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * txv4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * txv5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * txv6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * txv7

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            TAIL_DOG(pk, sc, zp, acc[r][0]);
        }
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            TAIL_DOG(pk, sc, zp, acc[r][1]);
        }
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        TAIL_LOAD_X(base);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp = row_ptrs[r] + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            TAIL_DOG(pk, sc, zp, acc[r][2]);
        }
    }
    #undef TAIL_LOAD_X
    #undef TAIL_DOG

    // Per-row warp reduction with fused residual add.
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] += s;   // fused residual
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_multirow_r2(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq4g256_residual_multirow_body<2>(A, x, y, M, K);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_residual_multirow_r4(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq4g256_residual_multirow_body<4>(A, x, y, M, K);
}

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_hfq4g256_residual_multirow_r8(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq4g256_residual_multirow_body<8>(A, x, y, M, K);
}
