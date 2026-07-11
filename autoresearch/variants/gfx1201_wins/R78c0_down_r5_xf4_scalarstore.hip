#include <hip/hip_runtime.h>

// Atomic-free variant of `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched`.
//
// Same per-(token, krank) GEMV compute. Instead of `atomicAdd(scale * acc)`
// into a shared `x_residual[N × M]` row (K_TOP-way contention per output
// cell, observed ~387 GiB/s on R9700/gfx1201 vs the sister gate_up at
// 954 GiB/s with the same MQ4 layout but no atomic), this kernel writes
// each (token, krank) result to its own row in an expanded buffer
// `expert_outputs[N × K_TOP × M]`. A separate `moe_down_combine_k8_batched`
// kernel folds K_TOP outputs back into the residual stream with topk_weights
// applied — see kernels/src/moe_down_combine_k8_batched.hip.
//
// Shapes (vs the residual_scaled variant):
//   expert_ptrs    : [n_exp]                — unchanged
//   topk_indices   : [N × K_TOP]            — unchanged
//   rot_batch      : [N × K_TOP × K]        — unchanged
//   expert_outputs : [N × K_TOP × M]        — replaces x_residual; non-atomic
// Note: `topk_weights` is NOT consumed here; the combine kernel applies the
// scale during reduction over K_TOP. Dropping the weights load also saves
// a small amount of bandwidth.
//
// Grid: (M, K_TOP, N). Block = 32 threads. No atomic ops — each output
// address is written by exactly one workgroup, so the writes are pure
// stores.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
    int M, int K, int K_TOP
) {
    // R78c0_down_r5_xf4_scalarstore: keep the confirmed r5 traffic cut and
    // float4 x loads, but remove the vector-output branch tree. This checks
    // whether the scalar store path is cheaper once row reuse has already cut
    // most of this kernel's memory traffic.
    // The launch grid is still M blocks in x; blocks with row0 >= M exit.
    // This preserves the ABI while reusing the same x[base..base+7] values
    // across five rows and testing whether the R2/R3/R4/R5 row-reuse ladder has
    // any remaining headroom before VGPR pressure overtakes the traffic win.
    const int row0 = blockIdx.x * 5;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const int row4 = row0 + 4;
    const bool have_row1 = row1 < M;
    const bool have_row2 = row2 < M;
    const bool have_row3 = row3 < M;
    const bool have_row4 = row4 < M;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = K / 256;
    const char* row_ptr0 = A + (long long)row0 * groups_per_row * 136;
    const char* row_ptr1 = A + (long long)row1 * groups_per_row * 136;
    const char* row_ptr2 = A + (long long)row2 * groups_per_row * 136;
    const char* row_ptr3 = A + (long long)row3 * groups_per_row * 136;
    const char* row_ptr4 = A + (long long)row4 * groups_per_row * 136;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    float ccc0 = 0.0f, ccc1 = 0.0f, ccc2 = 0.0f, ccc3 = 0.0f;
    float dcc0 = 0.0f, dcc1 = 0.0f, dcc2 = 0.0f, dcc3 = 0.0f;
    float ecc0 = 0.0f, ecc1 = 0.0f, ecc2 = 0.0f, ecc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8(b) \
        const float4 xv0 = *reinterpret_cast<const float4*>(x + (b)); \
        const float4 xv1 = *reinterpret_cast<const float4*>(x + (b) + 4); \
        const float x0 = xv0.x; \
        const float x1 = xv0.y; \
        const float x2 = xv0.z; \
        const float x3 = xv0.w; \
        const float x4 = xv1.x; \
        const float x5 = xv1.y; \
        const float x6 = xv1.z; \
        const float x7 = xv1.w

    #define DOG_X8(gp, a) do { \
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        {
            LOAD_X8(base0);
            const char* gp0 = row_ptr0 + g * 136;
            DOG_X8(gp0, acc0);
            if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc0);
            if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc0);
            if (have_row3) DOG_X8(row_ptr3 + g * 136, dcc0);
            if (have_row4) DOG_X8(row_ptr4 + g * 136, ecc0);
        }
        {
            LOAD_X8(base0 + 256);
            const char* gp0 = row_ptr0 + (g + 1) * 136;
            DOG_X8(gp0, acc1);
            if (have_row1) DOG_X8(row_ptr1 + (g + 1) * 136, bcc1);
            if (have_row2) DOG_X8(row_ptr2 + (g + 1) * 136, ccc1);
            if (have_row3) DOG_X8(row_ptr3 + (g + 1) * 136, dcc1);
            if (have_row4) DOG_X8(row_ptr4 + (g + 1) * 136, ecc1);
        }
        {
            LOAD_X8(base0 + 512);
            const char* gp0 = row_ptr0 + (g + 2) * 136;
            DOG_X8(gp0, acc2);
            if (have_row1) DOG_X8(row_ptr1 + (g + 2) * 136, bcc2);
            if (have_row2) DOG_X8(row_ptr2 + (g + 2) * 136, ccc2);
            if (have_row3) DOG_X8(row_ptr3 + (g + 2) * 136, dcc2);
            if (have_row4) DOG_X8(row_ptr4 + (g + 2) * 136, ecc2);
        }
        {
            LOAD_X8(base0 + 768);
            const char* gp0 = row_ptr0 + (g + 3) * 136;
            DOG_X8(gp0, acc3);
            if (have_row1) DOG_X8(row_ptr1 + (g + 3) * 136, bcc3);
            if (have_row2) DOG_X8(row_ptr2 + (g + 3) * 136, ccc3);
            if (have_row3) DOG_X8(row_ptr3 + (g + 3) * 136, dcc3);
            if (have_row4) DOG_X8(row_ptr4 + (g + 3) * 136, ecc3);
        }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc0);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc0);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc0);
        if (have_row3) DOG_X8(row_ptr3 + g * 136, dcc0);
        if (have_row4) DOG_X8(row_ptr4 + g * 136, ecc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc1);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc1);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc1);
        if (have_row3) DOG_X8(row_ptr3 + g * 136, dcc1);
        if (have_row4) DOG_X8(row_ptr4 + g * 136, ecc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8(row_ptr0 + g * 136, acc2);
        if (have_row1) DOG_X8(row_ptr1 + g * 136, bcc2);
        if (have_row2) DOG_X8(row_ptr2 + g * 136, ccc2);
        if (have_row3) DOG_X8(row_ptr3 + g * 136, dcc2);
        if (have_row4) DOG_X8(row_ptr4 + g * 136, ecc2);
    }
    #undef DOG_X8
    #undef LOAD_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    float ccc = (ccc0 + ccc1) + (ccc2 + ccc3);
    float dcc = (dcc0 + dcc1) + (dcc2 + dcc3);
    float ecc = (ecc0 + ecc1) + (ecc2 + ecc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        ccc += __shfl_down(ccc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        dcc += __shfl_down(dcc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        ecc += __shfl_down(ecc, offset);
    if (tid == 0) {
        // Direct store to per-(token, krank) row — no atomic contention.
        // Layout: expert_outputs[bid][krank][row].
        const long long out_base =
            ((long long)bid * K_TOP + krank) * (long long)M;
        expert_outputs[out_base + row0] = acc;
        if (have_row1) expert_outputs[out_base + row1] = bcc;
        if (have_row2) expert_outputs[out_base + row2] = ccc;
        if (have_row3) expert_outputs[out_base + row3] = dcc;
        if (have_row4) expert_outputs[out_base + row4] = ecc;
    }
}
