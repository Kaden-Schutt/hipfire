#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Wave32+dp4a batched 2-way fused gate+up GEMM for HFQ3-G256 on RDNA2.
//
// HFQ3 sibling of gemm_qkv_hfq3g256_dp4a; same lane layout and inner
// loop, only the row-routing prologue differs (2-way gate/up vs 3-way
// q/k/v).
//
// Phase 2 of docs/plans/gfx10_mq3_prefill.md.

#define QK8_1 32
#define BATCH_TILE 16

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void gemm_gate_up_hfq3g256_dp4a(
    const char*  __restrict__ A_gate,
    const char*  __restrict__ A_up,
    const block_q8_1_mmq* __restrict__ Xq,
    float*       __restrict__ Y_gate,
    float*       __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int batch_size
) {
    const int lane = threadIdx.x;
    const int gid = blockIdx.x;

    const int total_m = gate_m + up_m;
    if (gid >= total_m) return;

    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < gate_m) {
        A = A_gate; Y = Y_gate; row = gid;             out_stride = gate_m;
    } else {
        A = A_up;   Y = Y_up;   row = gid - gate_m;    out_stride = up_m;
    }

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 104;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = lane * 3;

    const int sub_in_block       = (lane / 4) % 4;
    const int block_within_group = lane / 16;
    const int qs_byte_off        = 8 * (lane % 16);

    float acc[BATCH_TILE];
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 104;

        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const float zp_eff = zp + 4.0f * sc;

        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;
        const unsigned int pk = (unsigned int)d[0]
                              | ((unsigned int)d[1] << 8)
                              | ((unsigned int)d[2] << 16);

        const unsigned int n0 = (pk      ) & 7u;
        const unsigned int n1 = (pk >>  3) & 7u;
        const unsigned int n2 = (pk >>  6) & 7u;
        const unsigned int n3 = (pk >>  9) & 7u;
        const unsigned int n4 = (pk >> 12) & 7u;
        const unsigned int n5 = (pk >> 15) & 7u;
        const unsigned int n6 = (pk >> 18) & 7u;
        const unsigned int n7 = (pk >> 21) & 7u;
        const int int_a = (int)(((n0 - 4) & 0xFF)
                              | (((n1 - 4) & 0xFF) << 8)
                              | (((n2 - 4) & 0xFF) << 16)
                              | (((n3 - 4) & 0xFF) << 24));
        const int int_b = (int)(((n4 - 4) & 0xFF)
                              | (((n5 - 4) & 0xFF) << 8)
                              | (((n6 - 4) & 0xFF) << 16)
                              | (((n7 - 4) & 0xFF) << 24));

        const int kblock = 2 * g + block_within_group;

        for (int b = 0; b < local_bs; b++) {
            const int token = batch_start + b;
            const block_q8_1_mmq* xb = Xq + (long long)kblock * batch_size + token;

            const int* xqs = (const int*)(xb->qs + qs_byte_off);
            const int xq_a = xqs[0];
            const int xq_b = xqs[1];

            int sumi = 0;
            sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
            sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

            const half2 ds = xb->ds4[sub_in_block];
            const float2 dsf = __half22float2(ds);
            const float d_x   = dsf.x;
            const float sum_x = dsf.y;

            acc[b] += sc * d_x * (float)sumi + zp_eff * sum_x * 0.25f;
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (lane == 0) Y[(long long)(batch_start + b) * out_stride + row] = val;
    }
}
