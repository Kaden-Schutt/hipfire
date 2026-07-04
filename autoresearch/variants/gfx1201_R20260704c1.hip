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
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
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
    const int row_bytes = groups_per_row * 136;
    const char* row0_ptr = A + (long long)row0 * row_bytes;
    const char* row1_ptr = row1 < M ? row0_ptr + row_bytes : row0_ptr;
    const int boff = tid * 4;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* g00 = row0_ptr + g * 136;
        const char* g01 = g00 + 136;
        const char* g02 = g01 + 136;
        const char* g03 = g02 + 136;
        const char* g10 = row1_ptr + g * 136;
        const char* g11 = g10 + 136;
        const char* g12 = g11 + 136;
        const char* g13 = g12 + 136;

        float sc00 = __builtin_bit_cast(float, *(const unsigned int*)(g00));
        float zp00 = __builtin_bit_cast(float, *(const unsigned int*)(g00 + 4));
        float sc01 = __builtin_bit_cast(float, *(const unsigned int*)(g01));
        float zp01 = __builtin_bit_cast(float, *(const unsigned int*)(g01 + 4));
        float sc02 = __builtin_bit_cast(float, *(const unsigned int*)(g02));
        float zp02 = __builtin_bit_cast(float, *(const unsigned int*)(g02 + 4));
        float sc03 = __builtin_bit_cast(float, *(const unsigned int*)(g03));
        float zp03 = __builtin_bit_cast(float, *(const unsigned int*)(g03 + 4));
        float sc10 = __builtin_bit_cast(float, *(const unsigned int*)(g10));
        float zp10 = __builtin_bit_cast(float, *(const unsigned int*)(g10 + 4));
        float sc11 = __builtin_bit_cast(float, *(const unsigned int*)(g11));
        float zp11 = __builtin_bit_cast(float, *(const unsigned int*)(g11 + 4));
        float sc12 = __builtin_bit_cast(float, *(const unsigned int*)(g12));
        float zp12 = __builtin_bit_cast(float, *(const unsigned int*)(g12 + 4));
        float sc13 = __builtin_bit_cast(float, *(const unsigned int*)(g13));
        float zp13 = __builtin_bit_cast(float, *(const unsigned int*)(g13 + 4));

        unsigned int pk00 = *(const unsigned int*)(g00 + 8 + boff);
        unsigned int pk01 = *(const unsigned int*)(g01 + 8 + boff);
        unsigned int pk02 = *(const unsigned int*)(g02 + 8 + boff);
        unsigned int pk03 = *(const unsigned int*)(g03 + 8 + boff);
        unsigned int pk10 = *(const unsigned int*)(g10 + 8 + boff);
        unsigned int pk11 = *(const unsigned int*)(g11 + 8 + boff);
        unsigned int pk12 = *(const unsigned int*)(g12 + 8 + boff);
        unsigned int pk13 = *(const unsigned int*)(g13 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG2(pk0, sc0, zp0, pk1, sc1, zp1, b, a0, a1) do { \
            float xv = x[(b)]; \
            (a0) += ((sc0) * (float)((pk0) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)((pk1) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 1]; \
            (a0) += ((sc0) * (float)(((pk0) >> 4) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 4) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 2]; \
            (a0) += ((sc0) * (float)(((pk0) >> 8) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 8) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 3]; \
            (a0) += ((sc0) * (float)(((pk0) >> 12) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 12) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 4]; \
            (a0) += ((sc0) * (float)(((pk0) >> 16) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 16) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 5]; \
            (a0) += ((sc0) * (float)(((pk0) >> 20) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 20) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 6]; \
            (a0) += ((sc0) * (float)(((pk0) >> 24) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 24) & 0xFu) + (zp1)) * xv; \
            xv = x[(b) + 7]; \
            (a0) += ((sc0) * (float)(((pk0) >> 28) & 0xFu) + (zp0)) * xv; \
            (a1) += ((sc1) * (float)(((pk1) >> 28) & 0xFu) + (zp1)) * xv; \
        } while (0)

        DOG2(pk00, sc00, zp00, pk10, sc10, zp10, base,       acc00, acc10);
        DOG2(pk01, sc01, zp01, pk11, sc11, zp11, base + 256, acc01, acc11);
        DOG2(pk02, sc02, zp02, pk12, sc12, zp12, base + 512, acc02, acc12);
        DOG2(pk03, sc03, zp03, pk13, sc13, zp13, base + 768, acc03, acc13);
        #undef DOG2
    }

    #define TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, b, a0, a1) do { \
        float xv = x[(b)]; \
        (a0) += ((sc0) * (float)((pk0) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)((pk1) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 1]; \
        (a0) += ((sc0) * (float)(((pk0) >> 4) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 4) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 2]; \
        (a0) += ((sc0) * (float)(((pk0) >> 8) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 8) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 3]; \
        (a0) += ((sc0) * (float)(((pk0) >> 12) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 12) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 4]; \
        (a0) += ((sc0) * (float)(((pk0) >> 16) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 16) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 5]; \
        (a0) += ((sc0) * (float)(((pk0) >> 20) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 20) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 6]; \
        (a0) += ((sc0) * (float)(((pk0) >> 24) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 24) & 0xFu) + (zp1)) * xv; \
        xv = x[(b) + 7]; \
        (a0) += ((sc0) * (float)(((pk0) >> 28) & 0xFu) + (zp0)) * xv; \
        (a1) += ((sc1) * (float)(((pk1) >> 28) & 0xFu) + (zp1)) * xv; \
    } while (0)

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp0 = row0_ptr + g * 136;
        const char* gp1 = row1_ptr + g * 136;
        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, base, acc00, acc10);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp0 = row0_ptr + g * 136;
        const char* gp1 = row1_ptr + g * 136;
        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, base, acc01, acc11);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp0 = row0_ptr + g * 136;
        const char* gp1 = row1_ptr + g * 136;
        float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
        float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG2(pk0, sc0, zp0, pk1, sc1, zp1, base, acc02, acc12);
    }
    #undef TAIL_DOG2

    float acc0 = (acc00 + acc01) + (acc02 + acc03);
    float acc1 = (acc10 + acc11) + (acc12 + acc13);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down(acc0, offset);
        acc1 += __shfl_down(acc1, offset);
    }

    if (tid == 0) {
        // Direct store to per-(token, krank) row — no atomic contention.
        // Layout: expert_outputs[bid][krank][row].
        const long long out_base =
            ((long long)bid * K_TOP + krank) * (long long)M;
        expert_outputs[out_base + row0] = acc0;
        if (row1 < M) expert_outputs[out_base + row1] = acc1;
    }
}
