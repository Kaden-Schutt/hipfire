#include <hip/hip_runtime.h>

// Batched repeat-penalty + argmax for DFlash greedy verify path.
//
// For each row i (0..B):
//   1. Build history = prev_committed ++ block[0..i-1]  (rows before i)
//   2. Apply frequency-weighted + recency-decayed repeat penalty
//      (same Phase 0 logic as sample_top_p kernel)
//   3. Take argmax
//
// One block per batch row (grid.x = batch_size). Each block reduces `n`
// logits via shared-memory tournament.
//
// Unlike the CPU host_path which downloads B×vocab to host and applies
// penalties per-row → argmax, this kernel keeps everything on GPU:
//   - Penalty computation is parallel across threads
//   - No D2H except the final B×4 bytes of argmax indices
//
// When repeat_penalty <= 1.0, the penalty phase is a no-op (fast path).
// When ngram_block is active, also applies n-gram continuation bans.

extern "C" __global__ void repeat_penalty_argmax_batched(
    float* __restrict__ logits,                   // [batch, n] row-major, MODIFIED IN PLACE
    int* __restrict__ result,                      // [batch] argmax per row
    const unsigned int* __restrict__ prev_committed, // committed tokens before this batch
    const unsigned int* __restrict__ block_tokens,   // [B] draft+seed tokens for current cycle
    int n,                                          // vocab_size
    int batch_size,                                 // B
    int prev_committed_len,                         // len(prev_committed)
    int repeat_window,                              // max tokens to scan for penalty
    float repeat_penalty,
    int ngram_block_active,                         // 0 or 1
    int ban_token_id                                // token to NEG_INFINITY in every row, -1 = none
) {
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);
    const int batch_idx = blockIdx.x;

    // ── Phase 0: Apply repeat penalty ──────────────────────────────
    if (repeat_penalty > 1.0f && repeat_window > 0) {
        float window_len = (float)repeat_window;
        // Build the effective history for this row:
        //   prev_committed[0..prev_committed_len] ++ block_tokens[0..batch_idx]
        // Total history len = prev_committed_len + batch_idx + 1
        int total_history = prev_committed_len + batch_idx + 1;
        int window_start = total_history > repeat_window ? total_history - repeat_window : 0;
        int window_effective = total_history - window_start;

        // Each thread handles one position in the effective window
        for (int idx = threadIdx.x; idx < window_effective; idx += blockDim.x) {
            int global_idx = window_start + idx;
            unsigned int tok;
            if (global_idx < prev_committed_len) {
                tok = prev_committed[global_idx];
            } else {
                tok = block_tokens[global_idx - prev_committed_len];
            }
            if ((int)tok >= n) continue;

            // Check for later occurrences within the window
            bool has_later = false;
            int count = 1;
            for (int j = 0; j < window_effective; j++) {
                int gj = window_start + j;
                unsigned int jtok;
                if (gj < prev_committed_len) {
                    jtok = prev_committed[gj];
                } else {
                    jtok = block_tokens[gj - prev_committed_len];
                }
                if (jtok == tok) {
                    if (j > idx) { has_later = true; break; }
                    count++;
                }
            }
            if (has_later) continue;

            float recency = ((float)idx + 1.0f) / window_len;
            float effective = powf(repeat_penalty, (float)count * recency);
            if (effective > 1.5f) effective = 1.5f;
            float v = logits[(size_t)batch_idx * n + tok];
            logits[(size_t)batch_idx * n + tok] = (v > 0.0f) ? (v / effective) : (v * effective);
        }
        __syncthreads();
    }

    // ── Phase 0b: N-gram blocking ──────────────────────────────────
    // When ngram_block_active is set, scan history for 3/4/5/6-gram
    // suffix matches and NEG_INFINITY the continuation token.
    // Simplified: only check the most recent 3/4/5/6-gram ending at
    // the boundary before this row against earlier occurrences.
    if (ngram_block_active) {
        int total_history = prev_committed_len + batch_idx + 1;
        // Build a local copy of history for this row for ngram scanning.
        // We scan from the beginning to find suffix matches.
        __shared__ unsigned int local_hist[256]; // max window for ngram scan
        int window_n = total_history > 256 ? 256 : total_history;
        int hist_start = total_history > 256 ? total_history - 256 : 0;

        // Copy history to shared memory cooperatively
        for (int idx = threadIdx.x; idx < window_n; idx += blockDim.x) {
            int gi = hist_start + idx;
            if (gi < prev_committed_len) {
                local_hist[idx] = prev_committed[gi];
            } else {
                local_hist[idx] = block_tokens[gi - prev_committed_len];
            }
        }
        __syncthreads();

        // Each thread handles one ngram_size from [3,4,5,6]
        // Only thread 0..3 do actual work
        if (threadIdx.x < 4) {
            int ngram_size = 3 + threadIdx.x; // 3, 4, 5, 6
            if (window_n > ngram_size) {
                unsigned int suffix[6]; // max 6
                for (int k = 0; k < ngram_size; k++) {
                    suffix[k] = local_hist[window_n - ngram_size + k];
                }
                int search_end = window_n - ngram_size;
                for (int i = 0; i < search_end; i++) {
                    bool match = true;
                    for (int k = 0; k < ngram_size; k++) {
                        if (local_hist[i + k] != suffix[k]) { match = false; break; }
                    }
                    if (match) {
                        int next_tok = (int)local_hist[i + ngram_size];
                        if (next_tok < n) {
                            logits[(size_t)batch_idx * n + next_tok] = -1e30f; // NEG_INFINITY
                        }
                        break; // only need the first match
                    }
                }
            }
        }
        __syncthreads();
    }

    // ── Phase 0c: Hard ban on a specific token ─────────────────────
    // Used for structural control: ban reopening <think> after answer
    // has started, ban nested tool calls, etc.
    if (ban_token_id >= 0 && threadIdx.x == 0) {
        for (int row = 0; row < batch_size; row++) {
            logits[(size_t)row * n + ban_token_id] = -1e30f; // NEG_INFINITY
        }
    }
    __syncthreads();

    // ── Phase 1: Shared-memory tournament argmax ──────────────────
    float lmax = -1e30f;
    int lidx = 0;
    const float* row = logits + (size_t)batch_idx * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > lmax) { lmax = v; lidx = i; }
    }
    s[threadIdx.x] = lmax;
    si[threadIdx.x] = lidx;
    __syncthreads();

    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz && s[threadIdx.x + sz] > s[threadIdx.x]) {
            s[threadIdx.x] = s[threadIdx.x + sz];
            si[threadIdx.x] = si[threadIdx.x + sz];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) result[batch_idx] = si[0];
}
