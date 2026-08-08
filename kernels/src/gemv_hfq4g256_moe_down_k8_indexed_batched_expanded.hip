#include <hip/hip_runtime.h>

#ifndef HIPFIRE_MOE_DOWN_LAST_COMBINE
#define HIPFIRE_MOE_DOWN_LAST_COMBINE 0
#endif

#ifndef HIPFIRE_MOE_DOWN_KERNEL
#define HIPFIRE_MOE_DOWN_KERNEL gemv_hfq4g256_moe_down_k8_indexed_batched_expanded
#endif

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
//
// VARIANT R17c1 down_g2_hoist_x16:
// BOD: L2-resident/mixed occ~43% mem_busy~44% (post g2_branchless_fulltile WIN).
// Current body: per-group LOAD_X8 then 4 DOGs → x-load serializes with weight
// DOG for each of the 2 groups. residual/multirow R13/R15 WIN was hoisting
// full x for the group-quad BEFORE dual-row DOG so weight MAC dual-issues
// against resident x under L2-resident weights. Apply that here: float4×4
// hoist of both groups' 16 x lanes first, then all 8 DOGs (4 rows × 2
// groups) with x resident. Value-preserving: same sc/zp/pk, same FMA order
// per DOG, same pairwise slot fold + shfl reduce + masked store.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_MOE_DOWN_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
#if HIPFIRE_MOE_DOWN_LAST_COMBINE
    const float* __restrict__ topk_weights,               // [N × K_TOP]
    float*       __restrict__ x_residual,                 // [N × M], in-place
#endif
    int M, int K, int K_TOP
) {
    const int row0 = blockIdx.x * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const bool have_row1 = row1 < M;
    const bool have_row2 = row2 < M;
    const bool have_row3 = row3 < M;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    // K=512 → 2 groups; row stride matches baseline (groups*136).
    const int groups_per_row = K / 256;
    // The R17c1 fast path below is specialized to EXACTLY two 256-groups
    // (K=512, the qwen35-A3B routed-down shape). Any other K must take the
    // generic quads+tail loop in the else branch — silently dropping groups
    // 2+ produced wrong down projections for K=1792 (lfm2moe) etc.
    if (groups_per_row == 2) {
    const long long row_stride = (long long)groups_per_row * 136;
    // Clamp OOB rows to row0 for weight reads (safe in-bounds); store still masked.
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const char* row_ptr2 = A + (long long)(have_row2 ? row2 : row0) * row_stride;
    const char* row_ptr3 = A + (long long)(have_row3 ? row3 : row0) * row_stride;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc0 = HIPFIRE_WEIGHT_RSRC(row_ptr0);
    const auto row_rsrc1 = HIPFIRE_WEIGHT_RSRC(row_ptr1);
    const auto row_rsrc2 = HIPFIRE_WEIGHT_RSRC(row_ptr2);
    const auto row_rsrc3 = HIPFIRE_WEIGHT_RSRC(row_ptr3);
#endif

    // gfx12: initiate the four first-demand group-0 cacheline fetches while
    // the existing x float4 hoist provides independent VMEM work.  Group 1
    // is adjacent and already covered by the group-0 compute window.
#if defined(__gfx1200__) || defined(__gfx1201__)
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr0, 0);
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr1, 0);
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr2, 0);
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr3, 0);
#endif

    // 2 live pairwise slots/row (acc2/acc3 would stay 0 on groups==2).
    float acc0 = 0.0f, acc1 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f;
    float ccc0 = 0.0f, ccc1 = 0.0f;
    float dcc0 = 0.0f, dcc1 = 0.0f;

    #if HIPFIRE_GFX1151_MOE_DOWN_HYBRID_BUFFER
    // gfx1151 hybrid cache lowering: scale/zero-point are wave-uniform, so
    // preserve their scalar/global path. Only the lane-divergent packed nibbles
    // use temporal buffer VMEM. This avoids turning uniform headers into 64
    // redundant vector loads while retaining the lean packed-weight address.
    #define HIPFIRE_MOE_DOWN_LOAD_SC(rsrc, gp, weight_offset) (*(const unsigned int*)(gp))
    #define HIPFIRE_MOE_DOWN_LOAD_ZP(rsrc, gp, weight_offset) (*(const unsigned int*)((gp) + 4))
    #else
    #define HIPFIRE_MOE_DOWN_LOAD_SC(rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, gp, weight_offset)
    #define HIPFIRE_MOE_DOWN_LOAD_ZP(rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, (gp) + 4, (weight_offset) + 4)
    #endif

    #define DOG_X8(gp, rsrc, weight_offset, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        float sc = __builtin_bit_cast(float, HIPFIRE_MOE_DOWN_LOAD_SC(rsrc, gp, weight_offset)); \
        float zp = __builtin_bit_cast(float, HIPFIRE_MOE_DOWN_LOAD_ZP(rsrc, gp, weight_offset)); \
        unsigned int pk = HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, (gp) + 8 + boff, (weight_offset) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * (x0) \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (x1) \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (x2) \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (x3) \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (x4) \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (x5) \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (x6) \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (x7); \
    } while (0)

    // Hoist both groups' 16 x elements (float4×4) before any weight DOG.
    const int base0 = tid * 8;
    const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
    const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
    const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
    const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);

    // Group 0 → pairwise slot 0 (matches baseline tail g=0 → acc0).
    DOG_X8(row_ptr0 + 0 * 136, row_rsrc0, 0 * 136, acc0,
           xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
    DOG_X8(row_ptr1 + 0 * 136, row_rsrc1, 0 * 136, bcc0,
           xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
    DOG_X8(row_ptr2 + 0 * 136, row_rsrc2, 0 * 136, ccc0,
           xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
    DOG_X8(row_ptr3 + 0 * 136, row_rsrc3, 0 * 136, dcc0,
           xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);

    // Group 1 → pairwise slot 1 (matches baseline tail g=1 → acc1).
    DOG_X8(row_ptr0 + 1 * 136, row_rsrc0, 1 * 136, acc1,
           xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
    DOG_X8(row_ptr1 + 1 * 136, row_rsrc1, 1 * 136, bcc1,
           xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
    DOG_X8(row_ptr2 + 1 * 136, row_rsrc2, 1 * 136, ccc1,
           xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
    DOG_X8(row_ptr3 + 1 * 136, row_rsrc3, 1 * 136, dcc1,
           xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
    #undef DOG_X8
    #undef HIPFIRE_MOE_DOWN_LOAD_SC
    #undef HIPFIRE_MOE_DOWN_LOAD_ZP

    // groups==2: only slots 0/1 live; (a0+a1) ≡ baseline (a0+a1)+(a2+a3) with a2=a3=0.
    float acc = acc0 + acc1;
    float bcc = bcc0 + bcc1;
    float ccc = ccc0 + ccc1;
    float dcc = dcc0 + dcc1;
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        ccc += __shfl_down(ccc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        dcc += __shfl_down(dcc, offset);

    if (tid == 0) {
        // Direct store to per-(token, krank) row — no atomic contention.
        // Layout: expert_outputs[bid][krank][row].
        const long long out_base =
            ((long long)bid * K_TOP + krank) * (long long)M;
        expert_outputs[out_base + row0] = acc;
        if (have_row1) expert_outputs[out_base + row1] = bcc;
        if (have_row2) expert_outputs[out_base + row2] = ccc;
        if (have_row3) expert_outputs[out_base + row3] = dcc;
    }
    } else {
        // ── Generic group count (groups_per_row != 2) ─────────────────────
        // Restored from the pre-July-2026 baseline (before the R17c1 K=512
        // specialization): quads+tail loop over ALL 256-groups. Four rows
        // per block share the activation x; each row accumulates its own
        // four pairwise slots. Correct for any K % 256 == 0 (e.g. lfm2moe
        // down K=1792 → 7 groups) — the fast path above would silently
        // drop groups 2+.
        const long long row_stride = (long long)groups_per_row * 136;
        const char* row_ptr0 = A + (long long)row0 * row_stride;
        const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
        const char* row_ptr2 = A + (long long)(have_row2 ? row2 : row0) * row_stride;
        const char* row_ptr3 = A + (long long)(have_row3 ? row3 : row0) * row_stride;
        const int boff = tid * 4;
        const int quads = groups_per_row >> 2;
        const int tail = groups_per_row & 3;

        float racc[4][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                            {0.0f, 0.0f, 0.0f, 0.0f},
                            {0.0f, 0.0f, 0.0f, 0.0f},
                            {0.0f, 0.0f, 0.0f, 0.0f}};
        const char* row_ptrs[4] = {row_ptr0, row_ptr1, row_ptr2, row_ptr3};

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        for (int r = 0; r < 4; r++) {
            const char* rp = row_ptrs[r];
            for (int q = 0; q < quads; q++) {
                const int g = q << 2;
                const char* gp0 = rp + g * 136;
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

                int base = g * 256 + tid * 8;

                DOG(pk0, sc0, zp0, base, racc[r][0]);
                DOG(pk1, sc1, zp1, base + 256, racc[r][1]);
                DOG(pk2, sc2, zp2, base + 512, racc[r][2]);
                DOG(pk3, sc3, zp3, base + 768, racc[r][3]);
            }
            const int tg = quads << 2;
            #define TAIL_DOG(pk, sc, zp, b, a) \
                (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                     + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                     + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                     + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                     + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                     + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                     + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                     + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]
            if (tail >= 1) {
                const char* gp = rp + tg * 136;
                float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                TAIL_DOG(pk, sc, zp, tg * 256 + tid * 8, racc[r][0]);
            }
            if (tail >= 2) {
                const char* gp = rp + (tg + 1) * 136;
                float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                TAIL_DOG(pk, sc, zp, (tg + 1) * 256 + tid * 8, racc[r][1]);
            }
            if (tail >= 3) {
                const char* gp = rp + (tg + 2) * 136;
                float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
                float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
                TAIL_DOG(pk, sc, zp, (tg + 2) * 256 + tid * 8, racc[r][2]);
            }
            #undef TAIL_DOG
        }
        #undef DOG

        float acc = (racc[0][0] + racc[0][1]) + (racc[0][2] + racc[0][3]);
        float bcc = (racc[1][0] + racc[1][1]) + (racc[1][2] + racc[1][3]);
        float ccc = (racc[2][0] + racc[2][1]) + (racc[2][2] + racc[2][3]);
        float dcc = (racc[3][0] + racc[3][1]) + (racc[3][2] + racc[3][3]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        for (int offset = 16; offset > 0; offset >>= 1)
            bcc += __shfl_down(bcc, offset);
        for (int offset = 16; offset > 0; offset >>= 1)
            ccc += __shfl_down(ccc, offset);
        for (int offset = 16; offset > 0; offset >>= 1)
            dcc += __shfl_down(dcc, offset);

        if (tid == 0) {
            const long long out_base =
                ((long long)bid * K_TOP + krank) * (long long)M;
            expert_outputs[out_base + row0] = acc;
            if (have_row1) expert_outputs[out_base + row1] = bcc;
            if (have_row2) expert_outputs[out_base + row2] = ccc;
            if (have_row3) expert_outputs[out_base + row3] = dcc;
        }
    }

#if HIPFIRE_MOE_DOWN_LAST_COMBINE
    // Preserve one independently schedulable expert wave per workgroup. Each
    // row tile publishes its four expanded outputs, then increments a reusable
    // counter in the scratch tail. Only the last expert wave performs the
    // fixed k=0..7 weighted fold, replacing the separate combine dispatch.
    // The scratch tail is [N × ceil(M/4)] u32 immediately after the expanded
    // [N × K_TOP × M] payload and is zeroed once at scratch allocation.
    int prior = -1;
    const int row_tiles = (M + 3) / 4;
    int* counters = reinterpret_cast<int*>(
        expert_outputs + (long long)gridDim.z * K_TOP * M);
    const int counter_index = bid * row_tiles + blockIdx.x;
    if (tid == 0) {
        // Publish this block's output stores before advertising arrival.
        __threadfence();
        prior = atomicAdd(counters + counter_index, 1);
    }
    prior = __shfl(prior, 0);
    if (prior == K_TOP - 1) {
        if (tid < 4) {
            const int row = row0 + tid;
            if (row < M) {
                const long long token_base = (long long)bid * K_TOP * M;
                const int weights_base = bid * K_TOP;
                float combined = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (k < K_TOP) {
                        combined += topk_weights[weights_base + k]
                                  * expert_outputs[token_base + (long long)k * M + row];
                    }
                }
                x_residual[(long long)bid * M + row] += combined;
            }
        }
        __syncwarp();
        if (tid == 0)
            atomicExch(counters + counter_index, 0);
    }
#endif
}
