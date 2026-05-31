// Phase 1.1 WMMA flash attention — asym4 batched, hd=128/256, gfx12 port.
//
// RDNA4 WMMA uses half8 operands and splits the K tile across the two
// half-waves. Accumulator layout is therefore:
//   acc[j] = C[8 * (tid >> 4) + j][tid & 15]
//
// This is the gfx12 sister of attention_flash_asym4_wmma_tile_batched.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "turbo_common.h"
#include "givens_common.h"

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define BLOCK_M    16
#define WMMA_N     16
#define WMMA_K     16
#define TILE_SIZE  128

extern "C"
__global__ void attention_flash_asym4_wmma_tile_batched_gfx12(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    const float* __restrict__ tree_bias,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_offset,
    int block_start,
    int block_cols
) {
    const int h = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int tile_id = blockIdx.z;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int lane_row = tid & 15;
    const int k_grp = tid >> 4;

    const int m_start = m_tile * BLOCK_M;
    const bool tree_mode = (tree_bias != nullptr);
    const int tile_start = tile_id * TILE_SIZE;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int HD = head_dim;
    const int MAX_HD = 256;

    const int k_bytes_per_head = 4 + HD / 2;
    const int k_bytes_per_pos = n_kv_heads * k_bytes_per_head;
    const int k_head_off = kv_h * k_bytes_per_head;

    const int v_blocks_per_head = HD / 32;
    const int v_total_blocks = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk = kv_h * v_blocks_per_head;
    const int v_row_stride = v_total_blocks * 34;

    __shared__ float scores[BLOCK_M][TILE_SIZE];
    __shared__ int   lds_q_pos[BLOCK_M];
    __shared__ int   lds_seq_len[BLOCK_M];
    __shared__ float lds_q_rot[BLOCK_M][MAX_HD];

    if (tid < BLOCK_M) {
        int bid = batch_offset + m_start + tid;
        int p = positions[bid];
        lds_q_pos[tid] = p;
        lds_seq_len[tid] = tree_mode ? (block_start + block_cols) : (p + 1);
    }
    __syncthreads();

    bool any_work = false;
    #pragma unroll
    for (int j = 0; j < BLOCK_M; j++) {
        if (lds_seq_len[j] > tile_start) { any_work = true; break; }
    }
    if (!any_work) return;

    // Load Q + apply Givens once per row. Lanes 16..31 are idle here.
    if (k_grp == 0) {
        const int my_bid_local = m_start + lane_row;
        const float* q_head = q + (size_t)my_bid_local * q_dim + h * head_dim;
        for (int k0 = 0; k0 < HD; k0 += WMMA_K) {
            float buf[WMMA_K];
            #pragma unroll
            for (int j = 0; j < WMMA_K; j++) buf[j] = q_head[k0 + j];

            #pragma unroll
            for (int qg = 0; qg < WMMA_K / 4; qg++) {
                int qs = qg * 4;
                int dim = k0 + qs;
                int half_idx = dim / 128;
                int block_tid = half_idx * 32 + (dim % 128) / 4;
                givens_forward(buf[qs], buf[qs + 1], buf[qs + 2], buf[qs + 3],
                               cos_theta, sin_theta, block_tid);
            }
            #pragma unroll
            for (int j = 0; j < WMMA_K; j++) {
                lds_q_rot[lane_row][k0 + j] = buf[j];
            }
        }
    }
    __syncthreads();

    const int n_kv_sub = TILE_SIZE / WMMA_N;

    for (int sub = 0; sub < n_kv_sub; sub++) {
        const int kv_sub_start = tile_start + sub * WMMA_N;
        if (kv_sub_start >= max_seq) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int row = 8 * k_grp + j;
                scores[row][sub * WMMA_N + lane_row] = -1e30f;
            }
            continue;
        }

        float8_t acc_qk = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};

        #pragma unroll
        for (int k0 = 0; k0 < HD; k0 += WMMA_K) {
            half8_t a_reg;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                a_reg[j] = (_Float16)lds_q_rot[lane_row][k0 + k_grp * 8 + j];
            }

            int my_kv = kv_sub_start + lane_row;
            half8_t b_reg;
            if (my_kv < max_seq) {
                const unsigned char* kb = k_cache + (size_t)my_kv * k_bytes_per_pos + k_head_off;
                float cnorm = *((const float*)kb);
                unsigned int pk = *((const unsigned int*)(kb + 4 + (k0 + k_grp * 8) / 2));
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    b_reg[j] = (_Float16)(cnorm * TURBO_C4[(pk >> (4 * j)) & 0xF]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) b_reg[j] = (_Float16)0.0f;
            }

            acc_qk = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, acc_qk);
        }

        const int my_col_pos = kv_sub_start + lane_row;
        const int col_in_tile = sub * WMMA_N + lane_row;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int cell_row = 8 * k_grp + j;
            int q_pos_j = lds_q_pos[cell_row];
            int sl_j = lds_seq_len[cell_row];

            float val = acc_qk[j] * scale_attn;
            bool ok_kv = (my_col_pos < max_seq) && (my_col_pos < sl_j);
            bool ok_mask = tree_mode ? true : (my_col_pos <= q_pos_j);
            float masked = (ok_kv && ok_mask) ? val : -1e30f;

            if (tree_mode && ok_kv && my_col_pos >= block_start) {
                int bias_col = my_col_pos - block_start;
                int gbid = batch_offset + m_start + cell_row;
                masked += tree_bias[gbid * block_cols + bias_col];
            }

            scores[cell_row][col_in_tile] = masked;
        }
    }
    __syncthreads();

    __shared__ float lds_row_max[BLOCK_M];
    __shared__ float lds_row_sum[BLOCK_M];
    __shared__ int   lds_row_tile_len[BLOCK_M];

    int my_row = lane_row;
    if (k_grp == 0) {
        int my_seq = lds_seq_len[my_row];
        int my_tile_len = my_seq - tile_start;
        if (my_tile_len < 0) my_tile_len = 0;
        if (my_tile_len > TILE_SIZE) my_tile_len = TILE_SIZE;

        float my_max = -1e30f;
        for (int i = 0; i < my_tile_len; i++) {
            my_max = fmaxf(my_max, scores[my_row][i]);
        }

        float my_sum = 0.0f;
        for (int i = 0; i < my_tile_len; i++) {
            float e = expf(scores[my_row][i] - my_max);
            scores[my_row][i] = e;
            my_sum += e;
        }

        for (int i = my_tile_len; i < TILE_SIZE; i++) {
            scores[my_row][i] = 0.0f;
        }

        lds_row_max[my_row] = my_max;
        lds_row_sum[my_row] = my_sum;
        lds_row_tile_len[my_row] = my_tile_len;
    }
    __syncthreads();

    int my_seq = lds_seq_len[my_row];
    int my_tile_len = lds_row_tile_len[my_row];
    float my_max = lds_row_max[my_row];
    float my_sum = lds_row_sum[my_row];

    const int my_local_bid = m_start + my_row;
    const int stride = 2 + HD;
    const bool will_write = (my_seq > tile_start);
    float* my_partials_row = partials
        + ((long long)my_local_bid * n_heads + h) * max_tiles * stride
        + tile_id * stride;
    if (will_write && k_grp == 0) {
        my_partials_row[0] = my_max;
        my_partials_row[1] = my_sum;
    }

    constexpr int OUT_CHUNK = 32;
    const int v_dim_per_lane = HD / 2;
    const int v_chunks_per_lane = v_dim_per_lane / OUT_CHUNK;
    const int v_dim_start = k_grp * v_dim_per_lane;

    for (int c = 0; c < v_chunks_per_lane; c++) {
        float out_chunk[OUT_CHUNK];
        #pragma unroll
        for (int d = 0; d < OUT_CHUNK; d++) out_chunk[d] = 0.0f;

        int d_base = v_dim_start + c * OUT_CHUNK;
        int v_bi = d_base / 32;
        int v_blk_off = (v_kv_head_blk + v_bi) * 34;

        if (my_tile_len > 0) {
            for (int k = 0; k < my_tile_len; k++) {
                float w = scores[my_row][k];
                int t = tile_start + k;
                const unsigned char* vbh = v_cache + (size_t)t * v_row_stride + v_blk_off;
                float vs = (float)*((const _Float16*)vbh);
                #pragma unroll
                for (int d = 0; d < 32; d++) {
                    out_chunk[d] += w * (vs * (float)((signed char)vbh[2 + d]));
                }
            }
        }

        if (will_write) {
            #pragma unroll
            for (int d = 0; d < OUT_CHUNK; d++) {
                my_partials_row[2 + d_base + d] = out_chunk[d];
            }
        }
    }
}
