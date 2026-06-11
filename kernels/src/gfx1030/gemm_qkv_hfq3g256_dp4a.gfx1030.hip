#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Wave32+dp4a batched 3-way fused qkv GEMM for HFQ3-G256 on RDNA2.
//
// Port of gemm_qkv_hfq4g256_wave64_dp4a.hip (gfx906) to wave32 (gfx1030+)
// with the HFQ3 3-bit unpack. Phase 2 of docs/plans/gfx10_mq3_prefill.md
// — an experiment to see whether the sdot4 (v_dot4_i32_i8) lift that
// helped gfx906 transfers to RDNA2's wave32 organization.
//
// ISA notes vs the gfx906 wave64 reference:
//   - wave32 native: 32 threads/block, 1 row/block (the wave64 ref uses
//     64 threads / 2 rows via warp_id split; we drop that split)
//   - __builtin_amdgcn_sdot4 lowers to v_dot4_i32_i8 on both archs
//   - __shfl_down offset ≤ 16 valid on wave32 (same as the gfx906 ref
//     since it only used offsets up to 16 within each logical warp)
//
// HFQ3 unpack vs HFQ4:
//   - 104 B group stride (vs 136 B for HFQ4)
//   - tid*3 byte offset (vs tid*4 for HFQ4)
//   - uint24 byte-combine for OOB safety
//   - 8 trits per thread per group, 3 bits each
//   - Signed mapping: trit_signed = trit_unsigned - 4 → [-4, 3]
//   - zp_eff = zp + 4 * sc (vs HFQ4's zp + 8 * sc)
//
// Risk flagged in the plan: 3-bit unpack overhead may eat the dp4a 4×
// ALU uplift over scalar (and may not beat the dot2 path which is
// already at ~234 tok/s on gfx1031). This kernel is a measurement,
// not a guaranteed win.
//
// Q8_1_mmq layout (kblock-major, [K/128, batch_size] of 144 B blocks):
//   - 4 × half2 ds  (d_x scale + sum_x per 32-element sub-block)
//   - 4 × 32 INT8 quantized X values
//
// Output semantics: y[b][row] = acc (overwrite). Matches gemm_qkv_hfq3g256
// at N=1 within FP-non-associativity and INT8 quant-noise bounds.

#define QK8_1 32
#define BATCH_TILE 16

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void gemm_qkv_hfq3g256_dp4a(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const block_q8_1_mmq* __restrict__ Xq,    // kblock-major: [K/128 * batch_size]
    float*       __restrict__ Y_q,             // [batch_size, q_m]
    float*       __restrict__ Y_k,
    float*       __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int batch_size
) {
    const int lane = threadIdx.x;  // 0..31
    const int gid = blockIdx.x;    // 1 row per block

    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < q_m) {
        A = A_q; Y = Y_q; row = gid;                              out_stride = q_m;
    } else if (gid < q_m + k_m) {
        A = A_k; Y = Y_k; row = gid - q_m;                         out_stride = k_m;
    } else {
        A = A_v; Y = Y_v; row = gid - q_m - k_m;                    out_stride = v_m;
    }

    const int batch_start = blockIdx.y * BATCH_TILE;
    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 104;
    const char* row_ptr = A + (long long)row * row_bytes;
    const int boff = lane * 3;

    // Lane layout mirrors the gfx906 wave64 reference exactly within a
    // single 32-lane warp:
    //   - lane / 16:        0 → first kblock (K[0..128)) of the 256-group
    //                       1 → second kblock (K[128..256))
    //   - (lane / 4) % 4:   sub_in_block index (0..3) — 4 lanes share a
    //                       32-element sub-block's ds metadata
    //   - 8 * (lane % 16):  byte offset within the kblock (0..120)
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
        // Signed mapping: trit_signed = trit_unsigned - 4. So
        // w_unsigned = sc * trit_unsigned + zp
        //            = sc * (trit_signed + 4) + zp
        //            = sc * trit_signed + (zp + 4 * sc)
        const float zp_eff = zp + 4.0f * sc;

        // uint24 byte-combine for OOB safety (matches scalar HFQ3 family).
        const unsigned char* d = (const unsigned char*)(gp + 8) + boff;
        const unsigned int pk = (unsigned int)d[0]
                              | ((unsigned int)d[1] << 8)
                              | ((unsigned int)d[2] << 16);

        // Unpack 8 trits → 8 signed bytes packed into two int32s.
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

            // sc * d_x * sumi : the signed-trit × INT8 dot product
            //                   scaled by both quant factors.
            // zp_eff * sum_x * 0.25 : 4 lanes share ds4[sub_in_block] but
            //                          each lane covers only 8/32 of the
            //                          sub-block — × 0.25 ensures the warp
            //                          reduction sums to zp_eff * sum_x once.
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
