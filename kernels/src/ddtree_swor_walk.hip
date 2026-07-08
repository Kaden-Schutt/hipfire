// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fully on-device, single-workgroup fused SWOR (sampling-without-replacement)
// speculative tree-verify walk. Replaces the host-side O(vocab·k)/slot residual
// loop with one kernel launch that consumes the target logits (already device-
// resident in verify scratch) and the draft logits (kept from the draft gemm),
// loops the sequential tree descent internally, and parallelizes every per-slot
// vocab sweep — target softmax, draft softmax, the recursive residual
// `p_{j+1}=normalize(relu(p_j − q_j))`, and the final categorical draw — across
// the block's threads with LDS reductions. Distribution-EXACT: mirrors the
// reference `ddtree::swor_step` verbatim.
//
// One block, blockDim.x ≤ 256. Scratch p_res/q_pos are [vocab] device buffers.
//
// out layout (int32): out[0]=accept_len; out[1]=bonus_token;
//                     out[2+i]=accepted child node index (i in 0..accept_len).

#include <hip/hip_runtime.h>

#define SWOR_NTH 256
#define SWOR_MAXK 8

extern "C" __global__ void ddtree_swor_walk_f32(
    const float* __restrict__ target_logits, // [n_slots * vocab]
    const float* __restrict__ draft_logits,  // [num_pos * vocab]
    const int* __restrict__ pos_cands,       // [num_pos * k] draw-ordered tokens
    const int* __restrict__ slot_depth,      // [n_slots]
    const int* __restrict__ child_of_cand,   // [n_slots * k] child node idx or -1
    float* __restrict__ p_res,               // scratch [vocab]
    float* __restrict__ q_pos,               // scratch [vocab]
    int* __restrict__ out,                   // [2 + num_pos]
    float temp, int k, int vocab, int n_slots, int num_pos, unsigned int seed)
{
    if (blockIdx.x != 0) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const float inv_t = 1.0f / fmaxf(temp, 1e-4f);

    __shared__ float red[SWOR_NTH];
    __shared__ int s_accept_len;
    __shared__ int s_cur_slot;
    __shared__ int s_done;
    __shared__ int s_loaded_depth;
    __shared__ float s_drawn_qsum;
    __shared__ int s_chosen_rank;
    __shared__ int s_drawn[SWOR_MAXK];
    __shared__ unsigned int s_rng;

    if (tid == 0) {
        s_accept_len = 0;
        s_cur_slot = 0;
        s_done = 0;
        s_loaded_depth = -1;
        s_rng = seed | 1u;
    }
    __syncthreads();

    for (int step = 0; step <= num_pos; ++step) {
        if (s_done) break;
        const int slot = s_cur_slot;
        const int depth = slot_depth[slot];

        // ---- target softmax → p_res (full vocab) ----
        const float* trow = target_logits + (size_t)slot * vocab;
        float m = -INFINITY;
        for (int t = tid; t < vocab; t += nth) m = fmaxf(m, trow[t] * inv_t);
        red[tid] = m; __syncthreads();
        for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
        const float tmax = red[0]; __syncthreads();
        float lsum = 0.f;
        for (int t = tid; t < vocab; t += nth) { float e = __expf(trow[t] * inv_t - tmax); p_res[t] = e; lsum += e; }
        red[tid] = lsum; __syncthreads();
        for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
        const float tinv = 1.f / red[0]; __syncthreads();
        for (int t = tid; t < vocab; t += nth) p_res[t] *= tinv;
        __syncthreads();

        // ---- leaf (no draft proposals): sample bonus from p_res, done ----
        if (depth >= num_pos) {
            if (tid == 0) {
                s_rng = s_rng * 1664525u + 1013904223u;
                float u = (float)(s_rng >> 8) * (1.f / 16777216.f);
                float acc = 0.f; int pick = vocab - 1;
                for (int t = 0; t < vocab; ++t) { acc += p_res[t]; if (u < acc) { pick = t; break; } }
                out[1] = pick;
                out[0] = s_accept_len;
                s_done = 1;
            }
            __syncthreads();
            break;
        }

        // ---- draft softmax → q_pos (recompute only when depth changes) ----
        if (s_loaded_depth != depth) {
            const float* drow = draft_logits + (size_t)depth * vocab;
            float dm = -INFINITY;
            for (int t = tid; t < vocab; t += nth) dm = fmaxf(dm, drow[t] * inv_t);
            red[tid] = dm; __syncthreads();
            for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
            const float dmax = red[0]; __syncthreads();
            float dsum = 0.f;
            for (int t = tid; t < vocab; t += nth) { float e = __expf(drow[t] * inv_t - dmax); q_pos[t] = e; dsum += e; }
            red[tid] = dsum; __syncthreads();
            for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
            const float dinv = 1.f / red[0]; __syncthreads();
            for (int t = tid; t < vocab; t += nth) q_pos[t] *= dinv;
            if (tid == 0) s_loaded_depth = depth;
            __syncthreads();
        }

        // ---- recursive SWOR over the k draw-ordered candidates ----
        if (tid == 0) { s_drawn_qsum = 0.f; s_chosen_rank = -1; }
        __syncthreads();
        const int* cands = pos_cands + (size_t)depth * k;
        for (int j = 0; j < k; ++j) {
            const int c = cands[j];
            if (tid == 0) {
                float Z = fmaxf(1.f - s_drawn_qsum, 1e-9f);
                float qjc = fminf(fmaxf(q_pos[c] / Z, 0.f), 1.f);
                float pc = p_res[c];
                float ratio = (qjc > 0.f) ? fminf(pc / qjc, 1.f) : 0.f;
                s_rng = s_rng * 1664525u + 1013904223u;
                float u = (float)(s_rng >> 8) * (1.f / 16777216.f);
                s_chosen_rank = (pc > 0.f && u < ratio) ? j : -1;
            }
            __syncthreads();
            if (s_chosen_rank >= 0) break;

            // Reject: p_res = normalize(relu(p_res − q_j)), q_j(t)=q_pos[t]/Z for
            // t ∉ {c_0..c_{j-1}} (the j already-drawn tokens in s_drawn).
            const float Z = fmaxf(1.f - s_drawn_qsum, 1e-9f);
            for (int t = tid; t < vocab; t += nth) {
                bool drawn = false;
                for (int i = 0; i < j; ++i) if (s_drawn[i] == t) { drawn = true; break; }
                float qj = drawn ? 0.f : q_pos[t] / Z;
                p_res[t] = fmaxf(p_res[t] - qj, 0.f);
            }
            __syncthreads();
            float rs = 0.f;
            for (int t = tid; t < vocab; t += nth) rs += p_res[t];
            red[tid] = rs; __syncthreads();
            for (int s = nth / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
            const float rinv = (red[0] > 0.f) ? 1.f / red[0] : 0.f; __syncthreads();
            for (int t = tid; t < vocab; t += nth) p_res[t] *= rinv;
            if (tid == 0) { s_drawn[j] = c; s_drawn_qsum += q_pos[c]; }
            __syncthreads();
        }

        // ---- act on the result (thread 0 mutates walk state) ----
        if (tid == 0) {
            if (s_chosen_rank >= 0) {
                int child = child_of_cand[slot * k + s_chosen_rank];
                if (child >= 0) {
                    out[2 + s_accept_len] = child;
                    s_accept_len += 1;
                    s_cur_slot = child + 1;
                } else {
                    // Accepted a candidate the budget pruned — emit its token.
                    out[1] = cands[s_chosen_rank];
                    out[0] = s_accept_len;
                    s_done = 1;
                }
            } else {
                // All k rejected: draw the bonus from the residual target.
                s_rng = s_rng * 1664525u + 1013904223u;
                float u = (float)(s_rng >> 8) * (1.f / 16777216.f);
                float acc = 0.f; int pick = vocab - 1;
                for (int t = 0; t < vocab; ++t) { acc += p_res[t]; if (u < acc) { pick = t; break; } }
                out[1] = pick;
                out[0] = s_accept_len;
                s_done = 1;
            }
        }
        __syncthreads();
    }

    // Safety: if we exhausted the depth loop without finishing, commit state.
    if (tid == 0 && !s_done) out[0] = s_accept_len;
}
