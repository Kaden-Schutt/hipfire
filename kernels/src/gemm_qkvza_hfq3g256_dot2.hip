#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// v_dot2_f32_f16-accelerated batched 4-way fused HFQ3-G256 GEMM for RDNA2.
// HFQ3 sibling of gemm_qkvza_hfq4g256_dot2.hip — same dispatch shape and
// dot2 inner loop; only the weight unpack differs (104 B/group, tid*3
// byte offset, uint24 byte-combine for 8 3-bit trits per group per thread).
//
// Overwrite semantics (Y = acc) — same as scalar/fp16/wmma fused variants.

#define BATCH_TILE 8

__launch_bounds__(32, 6)
extern "C" __global__ void gemm_qkvza_hfq3g256_dot2(
    const char*    __restrict__ A_qkv,
    const char*    __restrict__ A_z,
    const char*    __restrict__ A_beta,
    const char*    __restrict__ A_alpha,
    const _Float16* __restrict__ X,        // [N × K], FP16 pre-converted
    float*         __restrict__ Y_qkv,     // [N × qkv_m], FP32
    float*         __restrict__ Y_z,       // [N × z_m], FP32
    float*         __restrict__ Y_beta,    // [N × beta_m], FP32
    float*         __restrict__ Y_alpha,   // [N × alpha_m], FP32
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = N - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < qkv_m) {
        A = A_qkv;   Y = Y_qkv;   row = gid;                                  out_stride = qkv_m;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     Y = Y_z;     row = gid - qkv_m;                           out_stride = z_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  Y = Y_beta;  row = gid - (qkv_m + z_m);                   out_stride = beta_m;
    } else {
        A = A_alpha; Y = Y_alpha; row = gid - (qkv_m + z_m + beta_m);          out_stride = alpha_m;
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 104;
    const int boff = tid * 3;

    float acc0[BATCH_TILE], acc1[BATCH_TILE], acc2[BATCH_TILE], acc3[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    #define DEQUANT_H2_HFQ3(pk, sc_h, zp_h, h01, h23, h45, h67)         \
        do {                                                              \
            __half w0 = __float2half((float)((pk      ) & 7u));           \
            __half w1 = __float2half((float)((pk >>  3) & 7u));           \
            __half w2 = __float2half((float)((pk >>  6) & 7u));           \
            __half w3 = __float2half((float)((pk >>  9) & 7u));           \
            __half w4 = __float2half((float)((pk >> 12) & 7u));           \
            __half w5 = __float2half((float)((pk >> 15) & 7u));           \
            __half w6 = __float2half((float)((pk >> 18) & 7u));           \
            __half w7 = __float2half((float)((pk >> 21) & 7u));           \
            __half2 sc2 = __half2half2(sc_h);                             \
            __half2 zp2 = __half2half2(zp_h);                             \
            h01 = __hfma2(__halves2half2(w0, w1), sc2, zp2);              \
            h23 = __hfma2(__halves2half2(w2, w3), sc2, zp2);              \
            h45 = __hfma2(__halves2half2(w4, w5), sc2, zp2);              \
            h67 = __hfma2(__halves2half2(w6, w7), sc2, zp2);              \
        } while(0)

    #define LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67)                  \
        do {                                                              \
            const __half2* xp = (const __half2*)(xb + (base));            \
            xh01 = xp[0]; xh23 = xp[1]; xh45 = xp[2]; xh67 = xp[3];      \
        } while(0)

    #define DOT2_ACC(wh01, wh23, wh45, wh67, xh01, xh23, xh45, xh67, acc)   \
        do {                                                                  \
            (acc) = amd_mixed_dot((wh01), (xh01), (acc), false);              \
            (acc) = amd_mixed_dot((wh23), (xh23), (acc), false);              \
            (acc) = amd_mixed_dot((wh45), (xh45), (acc), false);              \
            (acc) = amd_mixed_dot((wh67), (xh67), (acc), false);              \
        } while(0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 104;
        const char* gp1 = gp0 + 104;
        const char* gp2 = gp1 + 104;
        const char* gp3 = gp2 + 104;

        __half sc0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0)));
        __half zp0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4)));
        __half sc1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1)));
        __half zp1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4)));
        __half sc2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2)));
        __half zp2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4)));
        __half sc3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3)));
        __half zp3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4)));

        const unsigned char* d0 = (const unsigned char*)(gp0 + 8) + boff;
        const unsigned char* d1 = (const unsigned char*)(gp1 + 8) + boff;
        const unsigned char* d2 = (const unsigned char*)(gp2 + 8) + boff;
        const unsigned char* d3 = (const unsigned char*)(gp3 + 8) + boff;

        unsigned int pk0 = (unsigned int)d0[0] | ((unsigned int)d0[1] << 8) | ((unsigned int)d0[2] << 16);
        unsigned int pk1 = (unsigned int)d1[0] | ((unsigned int)d1[1] << 8) | ((unsigned int)d1[2] << 16);
        unsigned int pk2 = (unsigned int)d2[0] | ((unsigned int)d2[1] << 8) | ((unsigned int)d2[2] << 16);
        unsigned int pk3 = (unsigned int)d3[0] | ((unsigned int)d3[1] << 8) | ((unsigned int)d3[2] << 16);

        __half2 w0_01, w0_23, w0_45, w0_67;
        __half2 w1_01, w1_23, w1_45, w1_67;
        __half2 w2_01, w2_23, w2_45, w2_67;
        __half2 w3_01, w3_23, w3_45, w3_67;
        DEQUANT_H2_HFQ3(pk0, sc0_h, zp0_h, w0_01, w0_23, w0_45, w0_67);
        DEQUANT_H2_HFQ3(pk1, sc1_h, zp1_h, w1_01, w1_23, w1_45, w1_67);
        DEQUANT_H2_HFQ3(pk2, sc2_h, zp2_h, w2_01, w2_23, w2_45, w2_67);
        DEQUANT_H2_HFQ3(pk3, sc3_h, zp3_h, w3_01, w3_23, w3_45, w3_67);

        const int base = g * 256 + tid * 8;

        #pragma unroll
        for (int b = 0; b < BATCH_TILE; b++) {
            if (b >= local_bs) break;
            const _Float16* xb = X + (long long)(batch_start + b) * K;

            __half2 xh01, xh23, xh45, xh67;

            LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67);
            DOT2_ACC(w0_01, w0_23, w0_45, w0_67, xh01, xh23, xh45, xh67, acc0[b]);

            LOAD_X_H2(xb, base + 256, xh01, xh23, xh45, xh67);
            DOT2_ACC(w1_01, w1_23, w1_45, w1_67, xh01, xh23, xh45, xh67, acc1[b]);

            LOAD_X_H2(xb, base + 512, xh01, xh23, xh45, xh67);
            DOT2_ACC(w2_01, w2_23, w2_45, w2_67, xh01, xh23, xh45, xh67, acc2[b]);

            LOAD_X_H2(xb, base + 768, xh01, xh23, xh45, xh67);
            DOT2_ACC(w3_01, w3_23, w3_45, w3_67, xh01, xh23, xh45, xh67, acc3[b]);
        }
    }

    #define TAIL_DOT2_HFQ3(acc_arr, g_idx)                                  \
        {                                                                    \
            const int g = g_idx;                                             \
            const char* gp = row_ptr + g * 104;                              \
            __half sc_h = __float2half(__builtin_bit_cast(float,             \
                           *(const unsigned int*)(gp)));                     \
            __half zp_h = __float2half(__builtin_bit_cast(float,             \
                           *(const unsigned int*)(gp + 4)));                 \
            const unsigned char* d = (const unsigned char*)(gp + 8) + boff;  \
            unsigned int pk = (unsigned int)d[0]                             \
                            | ((unsigned int)d[1] << 8)                      \
                            | ((unsigned int)d[2] << 16);                    \
            __half2 wh01, wh23, wh45, wh67;                                  \
            DEQUANT_H2_HFQ3(pk, sc_h, zp_h, wh01, wh23, wh45, wh67);         \
            const int base = g * 256 + tid * 8;                              \
            _Pragma("unroll")                                                \
            for (int b = 0; b < BATCH_TILE; b++) {                           \
                if (b >= local_bs) break;                                    \
                const _Float16* xb = X + (long long)(batch_start + b) * K;   \
                __half2 xh01, xh23, xh45, xh67;                              \
                LOAD_X_H2(xb, base, xh01, xh23, xh45, xh67);                \
                DOT2_ACC(wh01, wh23, wh45, wh67,                            \
                          xh01, xh23, xh45, xh67, (acc_arr)[b]);             \
            }                                                                \
        }

    if (tail >= 1) { TAIL_DOT2_HFQ3(acc0, quads << 2); }
    if (tail >= 2) { TAIL_DOT2_HFQ3(acc1, (quads << 2) + 1); }
    if (tail >= 3) { TAIL_DOT2_HFQ3(acc2, (quads << 2) + 2); }

    #undef DEQUANT_H2_HFQ3
    #undef LOAD_X_H2
    #undef DOT2_ACC
    #undef TAIL_DOT2_HFQ3

    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) {
        if (b >= local_bs) break;
        float val = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) Y[(long long)(batch_start + b) * out_stride + row] = val;
    }
}
