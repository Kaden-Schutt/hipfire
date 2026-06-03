// ALU-only Phase 1.0 spike: WMMA-FA on PRE-DEQUANTIZED fp16 K/V.
//
// gfx12/RDNA4 variant. The WMMA builtin takes half8 operands and splits the
// 16-wide K tile across the two half-waves:
//   k_grp = tid >> 4, acc[j] = C[8 * k_grp + j][tid & 15].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

#define BLOCK_M        16
#define BLOCK_N        64
#define BLOCK_N_SUB    16
#define WMMA_N         16
#define WMMA_K         16
#define TILE_SIZE      128
#define HEAD_DIM_SPIKE 128
#define ND_FRAGS       8

extern "C" __launch_bounds__(32, 2)
__global__ void fa_wmma_fp16(
    const _Float16* __restrict__ q,
    const _Float16* __restrict__ k_cache,
    const _Float16* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,
    int batch_size
) {
    const int h = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int kv_tile = blockIdx.z;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int lane_row = tid & 15;
    const int k_grp = tid >> 4;

    const int m_start = m_tile * BLOCK_M;
    const int kv_tile_start = kv_tile * TILE_SIZE;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    __shared__ _Float16 lds_v_tile[BLOCK_N_SUB * HEAD_DIM_SPIKE];
    __shared__ float    lds_p_tile[BLOCK_M * BLOCK_N_SUB];
    __shared__ int      lds_q_pos[BLOCK_M];
    __shared__ int      lds_seq_len[BLOCK_M];

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

    bool any_work = false;
    #pragma unroll
    for (int j = 0; j < BLOCK_M; j++) {
        if (lds_seq_len[j] > kv_tile_start) { any_work = true; break; }
    }
    if (!any_work) return;

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

    for (int kvb = 0; kvb < TILE_SIZE; kvb += BLOCK_N) {
        const int kv_block_start = kv_tile_start + kvb;
        if (kv_block_start >= max_seq) break;

        for (int sub = 0; sub < BLOCK_N / WMMA_N; sub++) {
            const int kv_sub_start = kv_block_start + sub * WMMA_N;
            if (kv_sub_start >= max_seq) break;

            float8_t acc_qk = {0, 0, 0, 0, 0, 0, 0, 0};

            #pragma unroll
            for (int k0 = 0; k0 < HEAD_DIM_SPIKE; k0 += WMMA_K) {
                half8_t a_reg, b_reg;

                int my_bid = m_start + lane_row;
                if (my_bid < batch_size) {
                    const _Float16* qsrc = q + (size_t)my_bid * q_dim + h * head_dim + k0 + k_grp * 8;
                    a_reg = *((const half8_t*)qsrc);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) a_reg[j] = (_Float16)0.0f;
                }

                int my_kv = kv_sub_start + lane_row;
                if (my_kv < max_seq) {
                    const _Float16* ksrc = k_cache + (size_t)my_kv * kv_dim + kv_h * head_dim + k0 + k_grp * 8;
                    b_reg = *((const half8_t*)ksrc);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) b_reg[j] = (_Float16)0.0f;
                }

                acc_qk = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_reg, acc_qk);
            }

            const int my_col_pos = kv_sub_start + lane_row;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int cell_row = 8 * k_grp + j;
                int q_pos_j = lds_q_pos[cell_row];
                float val = acc_qk[j] * scale_attn;
                bool ok_row = (q_pos_j >= 0);
                bool ok_kv = (my_col_pos < max_seq);
                bool ok_mask = (my_col_pos <= q_pos_j);
                acc_qk[j] = (ok_row && ok_kv && ok_mask) ? val : -1e30f;
            }

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

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int r = 8 * k_grp + j;
                lds_p_tile[r * BLOCK_N_SUB + lane_row] = acc_qk[j];
            }

            {
                int v_row = tid & 15;
                int v_chunk = tid >> 4;
                int v_kv = kv_sub_start + v_row;
                int v_dim_lds_base = v_chunk * 64;
                _Float16* v_dst = &lds_v_tile[v_row * HEAD_DIM_SPIKE + v_dim_lds_base];

                if (v_kv < max_seq) {
                    const _Float16* vsrc = v_cache
                        + (size_t)v_kv * kv_dim
                        + kv_h * head_dim
                        + v_dim_lds_base;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        *((half8_t*)(v_dst + i * 8)) = *((const half8_t*)(vsrc + i * 8));
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 64; i++) v_dst[i] = (_Float16)0.0f;
                }
            }
            __syncthreads();

            half8_t p_reg;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                p_reg[j] = (_Float16)lds_p_tile[lane_row * BLOCK_N_SUB + k_grp * 8 + j];
            }

            #pragma unroll
            for (int nd = 0; nd < ND_FRAGS; nd++) {
                int v_dim_start = nd * WMMA_N;
                half8_t v_reg;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    v_reg[i] = lds_v_tile[(k_grp * 8 + i) * HEAD_DIM_SPIKE + v_dim_start + lane_row];
                }
                float8_t pv = {0, 0, 0, 0, 0, 0, 0, 0};
                pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(p_reg, v_reg, pv);

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc_o[nd][j] = alpha[j] * acc_o[nd][j] + pv[j];
                }
            }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                m_state[j] = m_new[j];
                l_state[j] = l_new[j];
            }

            __syncthreads();
        }
    }

    const int stride = 2 + head_dim;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int row = 8 * k_grp + j;
        int bid = m_start + row;
        if (bid >= batch_size) continue;
        int sl = lds_seq_len[row];
        if (sl <= kv_tile_start) continue;

        float* p = partials
                 + ((long long)bid * n_heads + h) * max_tiles * stride
                 + kv_tile * stride;

        if (lane_row == 0) {
            p[0] = m_state[j];
            p[1] = l_state[j];
        }

        #pragma unroll
        for (int nd = 0; nd < ND_FRAGS; nd++) {
            int v_dim = nd * WMMA_N + lane_row;
            if (v_dim < head_dim) {
                p[2 + v_dim] = acc_o[nd][j];
            }
        }
    }
}
