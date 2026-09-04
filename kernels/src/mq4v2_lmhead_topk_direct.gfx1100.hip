// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1100 fused MQ4G256V2 (qt=44) LM-head GEMM + greedy DDTree top-K /
// log-sum-exp selection, single launch, resident single-wave workgroups.
//
// GEMM tile-body arithmetic / dual-half header contract / interleaved-C
// mapping authority:
//   gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds.hip
// The baseline DDTree draft lm_head (N<=16, gfx1100, default flags) runs the
// KS{2,4,8} split-K tier: KW waves accumulate DISJOINT K ranges of the same
// tile and wave 0 sums the KW fp32 accs in fixed order. This kernel
// reproduces that association EXACTLY inside one wave: KW private float8
// accs per lane, acc w accumulating groups [w*G/KW, (w+1)*G/KW) in
// ascending-g order, summed w=0..KW-1 at tile end. Tile results are therefore
// BIT-IDENTICAL to the baseline ksplit_lds GEMM for the same KW, so selected
// ids and their order (including intra-partition ties) match the old path
// exactly. (Non-default baseline tiers — ldsstage under
// HIPFIRE_RESIDUAL_LDSSTAGE=1 — reassociate differently; the launcher leaves
// that env alone and the slice kill switch covers it.)
//
// Selection / log_z numerics authority:
//   topk_logsumexp_batched.hip  (top-K replacement policy, serial merge)
//
// Route contract (launcher-enforced): N <= 16 (one batch tile, so
// col_tile == 0 always and lane ml of every workgroup owns column ml across
// every tile it processes — a register-resident running top-K per (wg, lane)
// is legal and logits are never materialized), K % 256 == 0, KW in {2,4,8}
// with (K/256) % KW == 0, KTOP in 1..=8, M arbitrary (clamped tail rows are
// discarded).
//
// Grid:  [WG_COUNT, 1, 1] — host-computed as min(m_tiles, 1344) and
//        statically proven resident: __launch_bounds__(32, 14) is met by the
//        compiler (amdgpu-waves-per-eu check; 78 VGPRs, phase-2 LDS promoted
//        to registers — zero group segment); 96 CU * 14 = 1344 >= WG_COUNT.
//        The phase-2 generation barrier would hang on a non-resident grid;
//        the launcher never emits one.
// Block: [32, 1, 1] — one wave32 per workgroup, the oracle's tile shape.
//
// Phase 1: static strided task map row_tile = wg + epoch * WG_COUNT (pure
//          function of (wg, epoch); deterministic, no routing atomics). Per
//          tile: the oracle's exact per-group loop (dual fp16 headers,
//          nibble dequant, 16 WMMA per group) into the KW accs, fixed-order
//          split-K sum, then lane ml < N folds its 8 column-ml cells into
//          (a) a running per-lane top-K list (oracle replacement policy:
//          replace current min when v > min) and (b) an online (max, sumexp)
//          pair (fp32 expf rescale).
//
// Barrier: after staging partials + __threadfence(), one lane per workgroup
//          bumps ctl[0]; the LAST arrival re-zeros ctl[0] for the NEXT launch
//          and flips ctl[1] (release); the rest spin on ctl[1] (acquire).
//          HIP stream serialization guarantees a quiescent gap between
//          launches, so the counters are clean at every launch start (the
//          launcher zeroes them once at scratch allocation). The barrier
//          gates visibility, never values — results are deterministic.
//
// Phase 2: workgroup c (c < N) reduces column c. Thread 0 serially merges
//          the staged per-wg top-K lists in ascending wg order (t-major,
//          j-minor, strict-> insertion — the oracle's merge order over
//          per-wg lists instead of per-thread lists; strict ties that cross
//          the old 256-way thread partition are NOT bitwise-reproduced, and
//          the parity harness documents this), folds the online (max, sumexp)
//          pairs in the same order, and writes top_idx[c*K + j] and
//          top_logp[c*K + j] = val - (max + log(se)).
//
// Scratch layout (one Raw alloc, element offsets):
//   partials[(wg * 32 + lane) * KMAX * 2]     : per-(wg,lane) top-K (f32,i32)
//   partials[WGM*32*KMAX*2 + (wg*32+lane)*2]  : per-(wg,lane) online max/se
//   ctl[0] = arrival counter, ctl[1] = generation (separate 8-byte alloc)
// (A column's 16 tile rows split across lanes ml and ml+16 — even/odd row
//  parity — so partials are per-LANE, 32 per workgroup, and phase 2 merges
//  both halves of each column.)
//
// Numerics note: logits are bit-identical to the baseline ks-tier GEMM for
// the same KW. sumexp partials use an fp32 online rescale tree, not the
// oracle's f64 accumulator; log-probs are gated against an f64 floor by
// test_mq4v2_topk_direct_gfx1100 (IDs are required exact, and are).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define TDK_MAX_K 8
#define TDK_WG_MAX 1344  // 96 CU x 14 resident 1-wave blocks (proven by amdgpu-waves-per-eu)
#define TDK_N_MAX 16

extern "C" __launch_bounds__(32, 14) __global__ void
mq4v2_lmhead_topk_direct_gfx1100(
    const char* __restrict__ A,        // [M, K] MQ4G256V2 packed (136 B / 256-K group)
    const _Float16* __restrict__ X,    // [N, K] fp16 rotated hidden
    float* __restrict__ partials,      // [TDK_WG_MAX*16*TDK_MAX_K*2 + TDK_WG_MAX*16*2]
    int* __restrict__ ctl,             // [2]: arrival counter, generation
    int* __restrict__ top_idx,         // [N * K]
    float* __restrict__ top_logp,      // [N * K]
    int M,                             // vocab rows
    int K,                             // hidden (multiple of 256)
    int N,                             // batch columns (1..TDK_N_MAX)
    int KTOP,                          // top-K (1..TDK_MAX_K)
    int KW                             // split-K width (2/4/8, G % KW == 0)
) {
    const int lane = threadIdx.x;
    const int wg = blockIdx.x;
    const int nwg = gridDim.x;
    const int ml = lane & 15;
    const bool col_live = ml < N;      // this lane's permanent column

    const int m_tiles = (M + 15) >> 4;
    const int groups_per_row = K >> 8; // K / 256
    const int gpw = groups_per_row / KW;

    float run_val[TDK_MAX_K];
    int run_idx[TDK_MAX_K];
    #pragma unroll
    for (int j = 0; j < TDK_MAX_K; j++) { run_val[j] = -INFINITY; run_idx[j] = 0; }
    float run_max = -INFINITY;
    float run_se = 0.0f;

    // Clamped X row for dead lanes (ml >= N): the WMMA is warp-collective so
    // every lane must issue its loads; their tile cells are discarded via
    // col_live. Mirrors the oracle's safe_batch clamp.
    const int safe_col = col_live ? ml : 0;
    const _Float16* x_base = X + (long long)safe_col * K;

    for (int tile = wg; tile < m_tiles; tile += nwg) {
        const int row0 = tile << 4;
        const int safe_row = (row0 + ml < M) ? (row0 + ml) : (M - 1);
        const char* row_base = A + (long long)safe_row * groups_per_row * 136;

        // Split-K replication WITHOUT a dynamically-indexed acc array (which
        // would spill to scratch): `acc` is the current wave's private chain
        // over groups [w*gpw, (w+1)*gpw), folded into `total` after each w.
        // total[j] = ((0 + acc_0[j]) + acc_1[j]) + ... — bit-identical to the
        // oracle's wave-0 fixed-order LDS sum.
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float8_t total = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int w_done = 0; // waves folded into total so far

        for (int g = 0; g < groups_per_row; g++) {
            const int wsel = g / gpw;
            if (wsel > w_done) {
                // Fold the finished wave's chain in ascending-w order and
                // start the next wave's chain at zero.
                _Pragma("unroll")
                for (int j = 0; j < 8; j++) total[j] += acc[j];
                acc = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                w_done = wsel;
            }
            const char* gp = row_base + g * 136;
            const unsigned int hA = *(const unsigned int*)(gp);
            const unsigned int hB = *(const unsigned int*)(gp + 4);
            const _Float16 sc0 =
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
            const _Float16 zp0 =
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
            const _Float16 sc1 =
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
            const _Float16 zp1 =
                (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
            const _Float16* xg = x_base + g * 256;

            _Pragma("unroll")
            for (int kt = 0; kt < 16; kt++) {
                const int k_off = kt * 16;
                const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
                const _Float16 zp_h = (kt < 8) ? zp0 : zp1;

                unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
                unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);

                half16_t a_reg;
                a_reg[0]  = sc_h * (_Float16)(float)((pk0 >> 0) & 0xFu) + zp_h;
                a_reg[1]  = sc_h * (_Float16)(float)((pk0 >> 4) & 0xFu) + zp_h;
                a_reg[2]  = sc_h * (_Float16)(float)((pk0 >> 8) & 0xFu) + zp_h;
                a_reg[3]  = sc_h * (_Float16)(float)((pk0 >> 12) & 0xFu) + zp_h;
                a_reg[4]  = sc_h * (_Float16)(float)((pk0 >> 16) & 0xFu) + zp_h;
                a_reg[5]  = sc_h * (_Float16)(float)((pk0 >> 20) & 0xFu) + zp_h;
                a_reg[6]  = sc_h * (_Float16)(float)((pk0 >> 24) & 0xFu) + zp_h;
                a_reg[7]  = sc_h * (_Float16)(float)((pk0 >> 28) & 0xFu) + zp_h;
                a_reg[8]  = sc_h * (_Float16)(float)((pk1 >> 0) & 0xFu) + zp_h;
                a_reg[9]  = sc_h * (_Float16)(float)((pk1 >> 4) & 0xFu) + zp_h;
                a_reg[10] = sc_h * (_Float16)(float)((pk1 >> 8) & 0xFu) + zp_h;
                a_reg[11] = sc_h * (_Float16)(float)((pk1 >> 12) & 0xFu) + zp_h;
                a_reg[12] = sc_h * (_Float16)(float)((pk1 >> 16) & 0xFu) + zp_h;
                a_reg[13] = sc_h * (_Float16)(float)((pk1 >> 20) & 0xFu) + zp_h;
                a_reg[14] = sc_h * (_Float16)(float)((pk1 >> 24) & 0xFu) + zp_h;
                a_reg[15] = sc_h * (_Float16)(float)((pk1 >> 28) & 0xFu) + zp_h;

                half16_t b_reg = *(const half16_t*)(xg + k_off);

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
            }
        }
        // Fold the final wave's chain.
        _Pragma("unroll")
        for (int j = 0; j < 8; j++) total[j] += acc[j];
        const float8_t tile_c = total;

        // tile_c[j] = C[2*j + (lane>>4)][ml]: column ml, rows row0 + 2j + (lane>>4).
        if (col_live) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int row = row0 + 2 * j + (lane >> 4);
                if (row >= M) continue; // clamped duplicate tail row: discard
                const float v = tile_c[j];
                // Online (max, sumexp) for this column (fp32 rescale).
                if (v > run_max) {
                    run_se = run_se * expf(run_max - v) + 1.0f;
                    run_max = v;
                } else {
                    run_se += expf(v - run_max);
                }
                // Running top-K: oracle per-thread replacement policy.
                int min_j = 0;
                float min_v = run_val[0];
                #pragma unroll
                for (int q = 1; q < TDK_MAX_K; q++) {
                    if (q < KTOP && run_val[q] < min_v) {
                        min_v = run_val[q];
                        min_j = q;
                    }
                }
                if (v > min_v) {
                    run_val[min_j] = v;
                    run_idx[min_j] = row;
                }
            }
        }
    }

    // ---- stage per-(wg, lane) partials -------------------------------------
    if (col_live) {
        float* dst = partials + ((size_t)wg * 32 + (size_t)lane) * TDK_MAX_K * 2;
        #pragma unroll
        for (int j = 0; j < TDK_MAX_K; j++) {
            dst[j * 2 + 0] = run_val[j];
            *reinterpret_cast<int*>(&dst[j * 2 + 1]) = run_idx[j];
        }
        float* onl = partials + (size_t)TDK_WG_MAX * 32 * TDK_MAX_K * 2
                    + ((size_t)wg * 32 + (size_t)lane) * 2;
        onl[0] = run_max;
        onl[1] = run_se;
    }
    __threadfence();

    // ---- grid-wide generation barrier (resident grid only) -----------------
    if (lane == 0) {
        const int gen_seen = atomicAdd(&ctl[1], 0); // acquire-read generation
        const int arrived = atomicAdd(&ctl[0], 1) + 1;
        if (arrived == nwg) {
            ctl[0] = 0;                    // reset for the NEXT launch (stream-serial)
            __threadfence();
            atomicAdd(&ctl[1], 1);         // release: flip generation
        } else {
            int g;
            do {
                g = atomicAdd(&ctl[1], 0); // acquire spin
            } while (g == gen_seen);
        }
    }
    __syncwarp();

    // ---- phase 2: workgroup c reduces column c ------------------------------
    const int col = wg;
    if (col >= N) return;

    // Lane l merges the per-(wg, lane) lists of column `col` for wgs
    // w = l, l+32, ... in ascending w order, half-parity 0 then 1 (lane col
    // holds even rows, lane col+16 odd rows). Private lists are built with
    // the same strict-> sorted insertion as the oracle's serial merge.
    float pv[TDK_MAX_K];
    int pi[TDK_MAX_K];
    #pragma unroll
    for (int j = 0; j < TDK_MAX_K; j++) { pv[j] = -INFINITY; pi[j] = 0; }
    float p_max = -INFINITY;
    float p_se = 0.0f;
    const float* onl_base = partials + (size_t)TDK_WG_MAX * 32 * TDK_MAX_K * 2;
    for (int w = lane; w < nwg; w += 32) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const float* src =
                partials + ((size_t)w * 32 + (size_t)(col + 16 * half)) * TDK_MAX_K * 2;
            #pragma unroll
            for (int j = 0; j < TDK_MAX_K; j++) {
                if (j >= KTOP) break;
                const float v = src[j * 2 + 0];
                const int i = *reinterpret_cast<const int*>(&src[j * 2 + 1]);
                int ins = KTOP;
                #pragma unroll
                for (int q = TDK_MAX_K - 1; q >= 0; q--) {
                    if (q < KTOP && v > pv[q]) ins = q;
                }
                if (ins < KTOP) {
                    for (int q = KTOP - 1; q > ins; q--) {
                        pv[q] = pv[q - 1];
                        pi[q] = pi[q - 1];
                    }
                    pv[ins] = v;
                    pi[ins] = i;
                }
            }
            const float wm = onl_base[((size_t)w * 32 + (size_t)(col + 16 * half)) * 2 + 0];
            const float ws = onl_base[((size_t)w * 32 + (size_t)(col + 16 * half)) * 2 + 1];
            if (wm > p_max) {
                p_se = p_se * expf(p_max - wm) + ws;
                p_max = wm;
            } else if (wm > -INFINITY) {
                // Skip empty (-inf, 0) partials: expf(-inf - -inf) is NaN.
                p_se += ws * expf(wm - p_max);
            }
        }
    }

    __shared__ float s_red[32][TDK_MAX_K * 2];
    __shared__ float s_onl[32][2];
    #pragma unroll
    for (int j = 0; j < TDK_MAX_K; j++) {
        s_red[lane][j * 2 + 0] = pv[j];
        *reinterpret_cast<int*>(&s_red[lane][j * 2 + 1]) = pi[j];
    }
    s_onl[lane][0] = p_max;
    s_onl[lane][1] = p_se;
    __syncwarp();

    // Lane 0: final serial merge over the 32 lane lists in ascending lane
    // order (deterministic; mirrors the harness simulation), then log_z.
    if (lane == 0) {
        float f_val[TDK_MAX_K];
        int f_idx[TDK_MAX_K];
        #pragma unroll
        for (int j = 0; j < TDK_MAX_K; j++) { f_val[j] = -INFINITY; f_idx[j] = 0; }
        for (int l = 0; l < 32; l++) {
            #pragma unroll
            for (int j = 0; j < TDK_MAX_K; j++) {
                if (j >= KTOP) break;
                const float v = s_red[l][j * 2 + 0];
                const int i = *reinterpret_cast<const int*>(&s_red[l][j * 2 + 1]);
                int ins = KTOP;
                #pragma unroll
                for (int q = TDK_MAX_K - 1; q >= 0; q--) {
                    if (q < KTOP && v > f_val[q]) ins = q;
                }
                if (ins < KTOP) {
                    for (int q = KTOP - 1; q > ins; q--) {
                        f_val[q] = f_val[q - 1];
                        f_idx[q] = f_idx[q - 1];
                    }
                    f_val[ins] = v;
                    f_idx[ins] = i;
                }
            }
        }
        float row_max = -INFINITY;
        float row_se = 0.0f;
        for (int l = 0; l < 32; l++) {
            const float wm = s_onl[l][0];
            const float ws = s_onl[l][1];
            if (wm > row_max) {
                row_se = row_se * expf(row_max - wm) + ws;
                row_max = wm;
            } else if (wm > -INFINITY) {
                row_se += ws * expf(wm - row_max);
            }
        }
        const float log_z = row_max + __logf(row_se);
        for (int j = 0; j < KTOP; j++) {
            top_idx[(size_t)col * KTOP + j] = f_idx[j];
            top_logp[(size_t)col * KTOP + j] = f_val[j] - log_z;
        }
    }
}
