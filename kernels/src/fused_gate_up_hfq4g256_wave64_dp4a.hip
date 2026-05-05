#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// dp4a-port of fused_gate_up_hfq4g256_wave64. Activations are
// pre-quantized to Q8_1 (block_q8_1_mmq layout, see
// kernels/src/gemm_hfq4g256_residual_mmq.hip:31). Weights stay
// HFQ4-G256 (4-bit nibbles + per-256-group fp32 scale + zp).
//
// Math identity:
//   sum_k (sc * nibble_k + zp) * x_k
// = sum_k (sc * (nibble_k - 8) + (zp + 8*sc)) * x_k    [shift by 8]
// = sc * sum_k((nibble_k - 8) * x_k) + (zp+8*sc) * sum_k(x_k)
// =       ^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^   ^^^^^^^^^^^
//         dp4a-able with int8 lanes      zp_eff    block_sum_x
//
// And with x quantized as x_k ≈ d_x * x_int8_k (per 32-K sub-block):
//   = sc * d_x * sum_k(int8_pack_w · int8_pack_x) + zp_eff * d_x * sum_k(x_int8_k)
//
// But Q8_1 stores both d_x and sum_x = sum of original (f32) x in the
// sub-block, so the second term is just zp_eff * sum_x (no d_x mult).
//
// Per-sub-block contribution:
//   acc += sc * d_x * sumi_dp4a + zp_eff * sum_x_subblock
//
// Lane mapping inside one HFQ4-G256 group (256 K-elements):
//   lane l owns K-elements [8l, 8l+7]
//   sub-block index s(l) = l/4   (8 sub-blocks total: 4 in block A, 4 in B)
//   block_index   b(l) = l/16    (lanes 0..15 → block A, lanes 16..31 → block B)
//   in-block sub  i(l) = (l/4) % 4
//
// So for lane l:
//   Read 8 nibbles (1 uint) from the HFQ4 weight at byte boff = lane*4.
//   Read 8 int8 from xq at qs offset 8*((l/4)*4) within the lane's block.
//   Read d, sum from ds4[i(l)] of the lane's block.
//
// Rationale for porting only fused_gate_up / qkv / qkvza (not
// gemv_residual): per-kernel PMC pass at 2026-05-05 showed those three
// sit at 38-41 % VALUBusy + 3.8-4.7 % MemUnitStalled — memory-bound,
// hence dp4a's 75 % x-traffic reduction lands on the actual
// bottleneck. gemv_residual was instruction-issue-serialized and got
// its win from the prefetch/ILP variant instead.

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];           // (d, sum_x) per 32-K sub-block, 4 sub-blocks/block
    int8_t qs[4 * QK8_1];   // 128 int8 K-elements per block
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

extern "C" __global__ void fused_gate_up_hfq4g256_wave64_dp4a(
    const char* __restrict__ A_gate,
    const char* __restrict__ A_up,
    const block_q8_1_mmq* __restrict__ Xq,   // K/128 blocks
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int gate_m, int up_m, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int gid = blockIdx.x * 2 + warp_id;
    const int total_m = gate_m + up_m;
    if (gid >= total_m) return;

    const char* A;
    float* y;
    int local_row;
    if (gid < gate_m) {
        A = A_gate; y = y_gate; local_row = gid;
    } else {
        A = A_up; y = y_up; local_row = gid - gate_m;
    }

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)local_row * groups_per_row * 136;
    const int boff = lane * 4;

    // Per-lane mapping inside one HFQ4-G256 group (256 K):
    //   K range  : [8*lane, 8*lane+7]
    //   sub-block: l/4   (0..7), maps to Q8_1 sub-block within group
    //   block    : l/16  (0 = block A, 1 = block B; HFQ4 group spans 2 blocks)
    //   ds4 index: (l/4) % 4  (sub-block index within the lane's Q8_1 block)
    //   qs offset: (8*lane) % 128 = 8 * (lane % 16)  (byte offset into qs[])
    const int sub_in_block = (lane / 4) % 4;     // ds4[i] index, 0..3
    const int block_within_group = lane / 16;    // 0 (A) or 1 (B)
    const int qs_byte_off = 8 * (lane % 16);     // 0,8,16,...,120 within a block

    float acc = 0.0f;

    // One HFQ4 group per outer iter.
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_ptr + (long long)g * 136;

        // Wave-uniform-per-warp (all 32 lanes see same gp): scale + zp.
        // Zero-point is shifted by 8*sc because nibbles are stored as
        // (n - 8) for the dp4a path (signed int8).
        const float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        const float zp_eff = zp + 8.0f * sc;

        // Load this lane's 8 nibbles (1 uint = 4 bytes).
        const unsigned int pk = *(const unsigned int*)(gp + 8 + boff);

        // Decode into 2× int8_pack (each = 4 signed int8, lane stride 4).
        // Nibble layout: pk = [n7|n6|n5|n4|n3|n2|n1|n0] (low nibble first)
        // int_a packs [n0,n1,n2,n3] - 8 each (low half)
        // int_b packs [n4,n5,n6,n7] - 8 each (high half)
        const unsigned int n0 = (pk >>  0) & 0xFu;
        const unsigned int n1 = (pk >>  4) & 0xFu;
        const unsigned int n2 = (pk >>  8) & 0xFu;
        const unsigned int n3 = (pk >> 12) & 0xFu;
        const unsigned int n4 = (pk >> 16) & 0xFu;
        const unsigned int n5 = (pk >> 20) & 0xFu;
        const unsigned int n6 = (pk >> 24) & 0xFu;
        const unsigned int n7 = (pk >> 28) & 0xFu;
        const int int_a = (int)(((n0 - 8) & 0xFF)
                              | (((n1 - 8) & 0xFF) << 8)
                              | (((n2 - 8) & 0xFF) << 16)
                              | (((n3 - 8) & 0xFF) << 24));
        const int int_b = (int)(((n4 - 8) & 0xFF)
                              | (((n5 - 8) & 0xFF) << 8)
                              | (((n6 - 8) & 0xFF) << 16)
                              | (((n7 - 8) & 0xFF) << 24));

        // Pick the right Q8_1 block (A or B) for this lane.
        // HFQ4 group g spans Q8_1 blocks (2g, 2g+1).
        const int kblock = 2 * g + block_within_group;
        const block_q8_1_mmq* xb = Xq + kblock;

        // Load 2 ints (8 int8) from xq.qs at the right offset.
        const int* xqs = (const int*)(xb->qs + qs_byte_off);
        const int xq_a = xqs[0];
        const int xq_b = xqs[1];

        // dp4a: signed int8 lanes. v_dot4_i32_i8 = sum of 4 signed
        // products, accumulated.
        int sumi = 0;
        sumi = __builtin_amdgcn_sdot4(int_a, xq_a, sumi, false);
        sumi = __builtin_amdgcn_sdot4(int_b, xq_b, sumi, false);

        // Per-sub-block (d, sum_x). Read the half2 for this lane's
        // sub-block and convert.
        const half2 ds = xb->ds4[sub_in_block];
        const float2 dsf = __half22float2(ds);
        const float d_x = dsf.x;
        const float sum_x = dsf.y;

        // Contribution: this lane's 8-K-element dot-product partial.
        // Note: sum_x is the *sub-block* sum (32 K-elements), but each
        // sub-block is shared across 4 lanes (8 K each). Divide by 4
        // so the 32-lane reduction recovers exactly zp_eff * sum_x per
        // sub-block.
        acc += sc * d_x * (float)sumi + zp_eff * sum_x * 0.25f;
    }

    // Reduce across the 32-lane warp. Same pyramid as the FP kernel.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (lane == 0) y[local_row] = acc;
}
