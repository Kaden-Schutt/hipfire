// ALU-only Phase 1.0 spike: WMMA-FA on PRE-DEQUANTIZED fp16 K/V.
//
// Pinned tile params (must match grid setup on host):
//   BLOCK_M = 16, BLOCK_N = 64, WMMA_N = 16, WMMA_K = 16, TILE_SIZE = 128,
//   HEAD_DIM_SPIKE = 128, ND_FRAGS = 8.
//
// Per-lane VGPR budget (target ≤ 128 for 2-wave occupancy; check via
// gfx-kernel-metadata after compile):
//   acc_o[ND_FRAGS]    8 × float8_t = 64 VGPRs
//   acc_qk             8 VGPRs
//   m_state, l_state   8 + 8 = 16 VGPRs
//   active a/b/v/p     4 × 8 = 32 VGPRs (not simultaneously live)
//   misc loop/ptr      ~16 VGPRs
//   ─────
//   ~120-136 VGPRs estimated; compiler may need 1-wave occupancy.
//
// LDS layout (per workgroup):
//   lds_v_tile[BLOCK_N_SUB=16][HD=128]     4 KB (fp16) — V B-layout transpose staging
//   lds_p_tile[BLOCK_M=16][BLOCK_N_SUB=16] 512 B (fp32) — P round-trip C→A layout
//   lds_q_pos[BLOCK_M]                     64 B (i32) — per-row positions for causal mask
//   lds_seq_len[BLOCK_M]                   64 B (i32)
//   ─────
//   ~4.7 KB per workgroup (well under 64 KB LDS budget).
//
// Grid: [n_heads, ceil(batch/BLOCK_M), max_tiles].  Block: [32, 1, 1].
// Partials layout: identical to scalar baseline → both kernels feed the
// same attention_flash_asym_reduce_batched output.
//
// Correctness anchors:
//  - WMMA wave32 layout: lane (tid & 15) provides one M-row of A and one
//    N-row of B; lanes 16..31 are redundant copies of rows 0..15.
//  - Output C/D-layout: acc[j] = C[2*j + (tid >> 4), tid & 15].
//  - Row-reductions via __shfl_xor with off ∈ {1,2,4,8} stay within the
//    half-wave; NEVER off=16 (crosses half-wave, contaminates other rows).
//  - V load: natural fetch is A-layout (one KV pos × 16 V-dims per lane,
//    coalesced). Production needs B-layout (one V-dim × 16 KV pos per lane,
//    strided). Solution: LDS staging — coalesced load, transposed read.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define BLOCK_M        16
#define BLOCK_N        64
#define BLOCK_N_SUB    16
#define WMMA_N         16
#define WMMA_K         16
#define TILE_SIZE      128
#define HEAD_DIM_SPIKE 128
#define ND_FRAGS       8   // HEAD_DIM_SPIKE / WMMA_N

extern "C" __launch_bounds__(32, 2)
__global__ void fa_wmma_fp16(
    const _Float16* __restrict__ q,               // [batch × n_heads × head_dim] fp16
    const _Float16* __restrict__ k_cache,         // [seq × n_kv_heads × head_dim] fp16
    const _Float16* __restrict__ v_cache,         // [seq × n_kv_heads × head_dim] fp16
    float* __restrict__ partials,                 // [batch × n_heads × max_tiles × (2+head_dim)]
    const int* __restrict__ positions,            // [batch]
    int n_heads,
    int n_kv_heads,
    int head_dim,                                 // assumed = 128 for spike
    int max_seq,
    float scale_attn,
    int tile_size,                                // assumed = 128
    int max_tiles,
    int batch_size
) {
    const int h = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int kv_tile = blockIdx.z;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int lane_row = tid & 15;                // 0..15 within half-wave
    const int half = tid >> 4;                    // 0 (lanes 0..15) or 1 (lanes 16..31)

    const int m_start = m_tile * BLOCK_M;
    const int kv_tile_start = kv_tile * TILE_SIZE;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    // ── LDS allocation (statically sized for HEAD_DIM_SPIKE=128) ─────────
    __shared__ _Float16 lds_v_tile[BLOCK_N_SUB * HEAD_DIM_SPIKE];   // 4 KB
    __shared__ float    lds_p_tile[BLOCK_M * BLOCK_N_SUB];          // 1 KB (fp32 to dodge fp16 bank conflicts)
    __shared__ int      lds_q_pos[BLOCK_M];
    __shared__ int      lds_seq_len[BLOCK_M];

    // ── Preload per-row positions ───────────────────────────────────────
    if (tid < BLOCK_M) {
        int bid = m_start + tid;
        if (bid < batch_size) {
            int p = positions[bid];
            lds_q_pos[tid] = p;
            lds_seq_len[tid] = p + 1;
        } else {
            lds_q_pos[tid] = -1;
            lds_seq_len[tid] = 0;
        }
    }
    __syncthreads();

    // OOB tile guard: skip workgroup if every row's seq_len ≤ kv_tile_start.
    bool any_work = false;
    #pragma unroll
    for (int j = 0; j < BLOCK_M; j++) {
        if (lds_seq_len[j] > kv_tile_start) { any_work = true; break; }
    }
    if (!any_work) return;

    // ── Per-lane FA-2 state (live across all sub-blocks in this tile) ───
    float8_t acc_o[ND_FRAGS];
    float m_state[8], l_state[8];
    #pragma unroll
    for (int nd = 0; nd < ND_FRAGS; nd++) {
        acc_o[nd] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        m_state[j] = -1e30f;
        l_state[j] = 0.0f;
    }

    // ════════════════════════════════════════════════════════════════════
    // Walk this kv_tile in BLOCK_N=64 chunks; each chunk has 4× WMMA_N=16
    // sub-blocks. Total: 2 × 4 = 8 sub-blocks per workgroup (one full
    // TILE_SIZE=128 tile partial).
    // ════════════════════════════════════════════════════════════════════
    for (int kvb = 0; kvb < TILE_SIZE; kvb += BLOCK_N) {
        const int kv_block_start = kv_tile_start + kvb;
        if (kv_block_start >= max_seq) break;

        for (int sub = 0; sub < BLOCK_N / WMMA_N; sub++) {
            const int kv_sub_start = kv_block_start + sub * WMMA_N;
            if (kv_sub_start >= max_seq) break;

            // ──────────────────────────────────────────────────────────
            // QK^T WMMA accumulator (one 16×16 fragment, fp32)
            // ──────────────────────────────────────────────────────────
            float8_t acc_qk = {0, 0, 0, 0, 0, 0, 0, 0};

            // Iterate K-dim in WMMA_K=16 chunks across head_dim=128.
            #pragma unroll
            for (int k0 = 0; k0 < HEAD_DIM_SPIKE; k0 += WMMA_K) {
                half16_t a_reg, b_reg;

                // A-fragment: lane (lane_row) holds Q-row (m_start + lane_row)'s
                // 16 head_dim values at columns k0..k0+15. Coalesced load.
                int my_bid = m_start + lane_row;
                if (my_bid < batch_size) {
                    const _Float16* qsrc = q + (size_t)my_bid * q_dim + h * head_dim + k0;
                    a_reg = *((const half16_t*)qsrc);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j++) a_reg[j] = (_Float16)0.0f;
                }

                // B-fragment: lane (lane_row) holds KV-position (kv_sub_start + lane_row)'s
                // 16 head_dim values at columns k0..k0+15. Coalesced load.
                int my_kv = kv_sub_start + lane_row;
                if (my_kv < max_seq) {
                    const _Float16* ksrc = k_cache + (size_t)my_kv * kv_dim + kv_h * head_dim + k0;
                    b_reg = *((const half16_t*)ksrc);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j++) b_reg[j] = (_Float16)0.0f;
                }

                acc_qk = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc_qk);
            }

            // ──────────────────────────────────────────────────────────
            // Scale + per-cell causal mask
            // Each lane's 8 cells map to (cell_row, cell_col) = (2*j + half, lane_row).
            // Mask: kv_pos (= kv_sub_start + lane_row) > q_pos[cell_row] → -inf.
            // Also: cell_row's bid OOB → -inf.
            // ──────────────────────────────────────────────────────────
            const int my_col_pos = kv_sub_start + lane_row;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int cell_row = 2 * j + half;
                int q_pos_j = lds_q_pos[cell_row];
                float val = acc_qk[j] * scale_attn;
                bool ok_row  = (q_pos_j >= 0);              // bid in range
                bool ok_kv   = (my_col_pos < max_seq);      // kv position in range
                bool ok_mask = (my_col_pos <= q_pos_j);     // causal
                acc_qk[j] = (ok_row && ok_kv && ok_mask) ? val : -1e30f;
            }

            // ──────────────────────────────────────────────────────────
            // FA-2 online softmax update — per row, in C/D-layout
            // Each lane owns 8 cells across 8 row-slots. Reductions across
            // the 16 cols of each row stay within the half-wave (off ∈ {1,2,4,8}).
            // ──────────────────────────────────────────────────────────
            float m_new[8], l_new[8], alpha[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float row_max = acc_qk[j];
                row_max = fmaxf(row_max, __shfl_xor(row_max, 1));
                row_max = fmaxf(row_max, __shfl_xor(row_max, 2));
                row_max = fmaxf(row_max, __shfl_xor(row_max, 4));
                row_max = fmaxf(row_max, __shfl_xor(row_max, 8));

                m_new[j] = fmaxf(m_state[j], row_max);
                alpha[j] = expf(m_state[j] - m_new[j]);
                float p_cell = expf(acc_qk[j] - m_new[j]);
                acc_qk[j] = p_cell;

                float row_sum = p_cell;
                row_sum += __shfl_xor(row_sum, 1);
                row_sum += __shfl_xor(row_sum, 2);
                row_sum += __shfl_xor(row_sum, 4);
                row_sum += __shfl_xor(row_sum, 8);
                l_new[j] = alpha[j] * l_state[j] + row_sum;
            }

            // ──────────────────────────────────────────────────────────
            // LDS round-trip: C-layout (acc_qk) → A-layout (p_reg)
            // Write cell (2j+half, lane_row) ← acc_qk[j]. Each cell written
            // exactly once. Lanes 0..15 + lanes 16..31 partition rows even/odd.
            // ──────────────────────────────────────────────────────────
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int r = 2 * j + half;
                lds_p_tile[r * BLOCK_N_SUB + lane_row] = acc_qk[j];
            }

            // ──────────────────────────────────────────────────────────
            // V LDS staging — load V_sub into LDS in natural (coalesced)
            // order so the WMMA can read it transposed (B-layout).
            //
            // Each lane loads 4 chunks of 16 fp16 = 64 fp16 from global:
            //   tid covers KV row (tid % 16) and head_dim slice (tid / 16) * 64.
            // (lanes 16..31 cover head_dim [64..128); lanes 0..15 cover [0..64).)
            // Read pattern is contiguous within each lane → coalesced when
            // adjacent lanes share the same KV row (lanes 0,16 / 1,17 / etc.).
            // ──────────────────────────────────────────────────────────
            {
                int v_row = tid & 15;
                int v_chunk = tid >> 4;            // 0 or 1
                int v_kv = kv_sub_start + v_row;
                int v_dim_lds_base = v_chunk * 64;
                _Float16* v_dst = &lds_v_tile[v_row * HEAD_DIM_SPIKE + v_dim_lds_base];

                if (v_kv < max_seq) {
                    const _Float16* vsrc = v_cache
                        + (size_t)v_kv * kv_dim
                        + kv_h * head_dim
                        + v_dim_lds_base;
                    // 4× half16_t = 64 fp16 = 128 B per lane.
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        *((half16_t*)(v_dst + i * 16)) = *((const half16_t*)(vsrc + i * 16));
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 64; i++) v_dst[i] = (_Float16)0.0f;
                }
            }
            __syncthreads();

            // ──────────────────────────────────────────────────────────
            // Load p_reg in A-layout from LDS (16 P-values for lane_row's row).
            // ──────────────────────────────────────────────────────────
            half16_t p_reg;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                p_reg[j] = (_Float16) lds_p_tile[lane_row * BLOCK_N_SUB + j];
            }

            // ──────────────────────────────────────────────────────────
            // P @ V — iterate over V-dim chunks (8 chunks for head_dim=128).
            // Each chunk produces one 16×16 fragment that updates acc_o[nd].
            //
            // B-fragment: lane (lane_row) holds V-dim (v_dim_start + lane_row)'s
            // 16 KV-position values. Reads from LDS at stride HEAD_DIM_SPIKE.
            // ──────────────────────────────────────────────────────────
            #pragma unroll
            for (int nd = 0; nd < ND_FRAGS; nd++) {
                int v_dim_start = nd * WMMA_N;
                half16_t v_reg;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    v_reg[i] = lds_v_tile[i * HEAD_DIM_SPIKE + v_dim_start + lane_row];
                }
                float8_t pv = {0, 0, 0, 0, 0, 0, 0, 0};
                pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_reg, v_reg, pv);

                // Rescale prior O slice by alpha (per cell j → per row).
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc_o[nd][j] = alpha[j] * acc_o[nd][j] + pv[j];
                }
            }

            // Commit softmax state.
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                m_state[j] = m_new[j];
                l_state[j] = l_new[j];
            }

            __syncthreads();   // before reusing lds_p_tile / lds_v_tile next sub-block
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Write partials [batch × n_heads × max_tiles × (2 + head_dim)]
    //
    // Each lane owns 8 cells of acc_o per V-dim slice (nd in [0, ND_FRAGS)).
    // Lane (lane_row, half) writes cells at (row, col):
    //   row = 2 * j + half        (j in [0, 8))
    //   col = nd * WMMA_N + lane_row
    //
    // m_state[j] / l_state[j] are broadcast across the half-wave by the
    // softmax reductions — lane (lane_row == 0) writes the (row) header.
    // ════════════════════════════════════════════════════════════════════
    const int stride = 2 + head_dim;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int row = 2 * j + half;
        int bid = m_start + row;
        if (bid >= batch_size) continue;
        int sl = lds_seq_len[row];
        if (sl <= kv_tile_start) continue;

        float* p = partials
                 + ((long long)bid * n_heads + h) * max_tiles * stride
                 + kv_tile * stride;

        if (lane_row == 0) {
            // m_state[j] and l_state[j] are broadcast across the half-wave's
            // 16 col-lanes after __shfl_xor reductions. Lane (tid in {0, 16})
            // writes the header for row = 2*j + (tid >> 4).
            p[0] = m_state[j];
            p[1] = l_state[j];
        }

        // Scatter: lane lane_row writes one V-dim slot of each acc_o[nd] frag.
        #pragma unroll
        for (int nd = 0; nd < ND_FRAGS; nd++) {
            int v_dim = nd * WMMA_N + lane_row;
            if (v_dim < head_dim) {
                p[2 + v_dim] = acc_o[nd][j];
            }
        }
    }
}
