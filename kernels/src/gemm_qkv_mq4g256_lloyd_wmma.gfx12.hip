#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx12 (RDNA4) variant of MQ4-Lloyd-G256 fused QKV WMMA (FullAttention preamble).
// Sister of gemm_qkv_mq4g256_lloyd_wmma.hip. See that file's header for the
// LDS-codebook + group-stride rationale; this file is the gfx12 lane-split
// (half8_t / K4-unroll) port.
//
// **gfx12 variant: code-complete, runtime-unvalidated locally.**

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
// C symbol kept unsuffixed (see residual gfx12 sibling for rationale).
extern "C" __global__ void gemm_qkv_mq4g256_lloyd_wmma(
    const char*    __restrict__ A_q,
    const char*    __restrict__ A_k,
    const char*    __restrict__ A_v,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_q,
    float*         __restrict__ Y_k,
    float*         __restrict__ Y_v,
    int q_m, int k_m, int v_m,
    int K, int N
) {
    const int total_m = q_m + k_m + v_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int my_row = row_start + m_lane;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < q_m) {
        A = A_q;  local_row = safe_row;
    } else if (safe_row < q_m + k_m) {
        A = A_k;  local_row = safe_row - q_m;
    } else {
        A = A_v;  local_row = safe_row - q_m - k_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + m_lane < N) ? (batch_start + m_lane) : 0;
    const _Float16* x_base = X + (long long)safe_batch * K;

    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 8;
    const int load_my_row = row_start + load_tile_row;
    const int load_safe_row = (load_my_row < total_m) ? load_my_row : (total_m - 1);
    const char* load_A;
    int load_local_row;
    if (load_safe_row < q_m) {
        load_A = A_q;  load_local_row = load_safe_row;
    } else if (load_safe_row < q_m + k_m) {
        load_A = A_k;  load_local_row = load_safe_row - q_m;
    } else {
        load_A = A_v;  load_local_row = load_safe_row - q_m - k_m;
    }
    const char* load_row_base = load_A + (long long)load_local_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    #define UNPACK_CHUNK_DQ(pk)                                                   \
        do {                                                                      \
            unsigned int _p = (pk);                                               \
            a_reg[0] = cb_lds[cb_base + ((_p >>  0) & 0xFu)];                     \
            a_reg[1] = cb_lds[cb_base + ((_p >>  4) & 0xFu)];                     \
            a_reg[2] = cb_lds[cb_base + ((_p >>  8) & 0xFu)];                     \
            a_reg[3] = cb_lds[cb_base + ((_p >> 12) & 0xFu)];                     \
            a_reg[4] = cb_lds[cb_base + ((_p >> 16) & 0xFu)];                     \
            a_reg[5] = cb_lds[cb_base + ((_p >> 20) & 0xFu)];                     \
            a_reg[6] = cb_lds[cb_base + ((_p >> 24) & 0xFu)];                     \
            a_reg[7] = cb_lds[cb_base + ((_p >> 28) & 0xFu)];                     \
        } while (0)

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const _Float16* xg = x_base + g * 256;
        const char* dp = gp + 32;
        const int cb_base = m_lane * 16;

        for (int kt = 0; kt < 16; kt += 4) {
            unsigned int pka = *(const unsigned int*)(dp + (kt + 0) * 8 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(dp + (kt + 1) * 8 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(dp + (kt + 2) * 8 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(dp + (kt + 3) * 8 + k_grp * 4);

            half8_t b_a = *(const half8_t*)(xg + (kt + 0) * 16 + k_grp * 8);
            half8_t b_b = *(const half8_t*)(xg + (kt + 1) * 16 + k_grp * 8);
            half8_t b_c = *(const half8_t*)(xg + (kt + 2) * 16 + k_grp * 8);
            half8_t b_d = *(const half8_t*)(xg + (kt + 3) * 16 + k_grp * 8);

            half8_t a_reg;
            UNPACK_CHUNK_DQ(pka);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            UNPACK_CHUNK_DQ(pkb);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            UNPACK_CHUNK_DQ(pkc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            UNPACK_CHUNK_DQ(pkd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        __syncthreads();
    }
    #undef UNPACK_CHUNK_DQ

    const int out_col = batch_start + m_lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 8 * k_grp + j;
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < q_m) {
                    Y = Y_q;  proj_row = out_row;               out_stride = q_m;
                } else if (out_row < q_m + k_m) {
                    Y = Y_k;  proj_row = out_row - q_m;          out_stride = k_m;
                } else {
                    Y = Y_v;  proj_row = out_row - q_m - k_m;    out_stride = v_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
