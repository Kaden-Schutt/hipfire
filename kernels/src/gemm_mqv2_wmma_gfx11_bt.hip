// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Generic batch-tiled gfx11 WMMA prefill for MQ{2,3,5,6}G256V2.
// Modeled on accepted MQ4 gfx11 BT kernels:
//   gemm_qkvza_mq4g256v2_wmma_gfx11_bt.hip  (overwrite BT4)
//   gemm_qkv_mq4g256v2_wmma_gfx11_bt.hip    (overwrite BT4)
//   gemm_gate_up_mq4g256v2_wmma_gfx11_bt.hip (overwrite BT12)
//   gemm_mq4g256v2_residual_wmma_gfx11_bt.hip (residual += BT4)
//
// V2 group: 8 B dual-FP16 header + contiguous bit-packed payload.
// GROUP_BYTES = 8 + 32*BITS. Half header selected by kt < 8.
// Weights decode once per K-tile and reuse across B independent N-tile accs.
// Output: acc[j] = C[2*j + (tid>>4)][tid & 15].
// Exact gfx1100 / gfx1151 only; launch_bounds(32, 2).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1100__) || defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Decode one 16-weight K-tile from contiguous bit-packed payload after the
// 8-byte header. Reads only the current payload byte and, when the code
// straddles a byte boundary ((bit_in_byte + BITS) > 8), the next byte.
// Never reads outside the tile's bit range within the group payload.
template <int BITS>
__device__ __forceinline__ half16_t decode_tile(const unsigned char* __restrict__ payload,
                                                int kt,
                                                _Float16 sc_h,
                                                _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int bit_pos = (kt * 16 + i) * BITS;
        const int byte_i = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned int v = (unsigned int)payload[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= ((unsigned int)payload[byte_i + 1]) << 8;
        const unsigned int q = (v >> bit_in_byte) & qmask;
        out[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return out;
}

// ---------------------------------------------------------------------------
// QKVZA overwrite BT4 — four projection regions (QKV/Z/beta/alpha)
// Grid: [ceil((qkv_m+z_m+beta_m+alpha_m)/16), ceil(N/(16*BV))]. Block: [32].
// ---------------------------------------------------------------------------
#define GEN_QKVZA_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_qkvza_mq##BITS##g256v2_wmma_gfx11_bt##BV( \
    const char* __restrict__ A_qkv, \
    const char* __restrict__ A_z, \
    const char* __restrict__ A_beta, \
    const char* __restrict__ A_alpha, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_qkv, \
    float* __restrict__ Y_z, \
    float* __restrict__ Y_beta, \
    float* __restrict__ Y_alpha, \
    int qkv_m, int z_m, int beta_m, int alpha_m, \
    int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = qkv_m + z_m + beta_m + alpha_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; \
    int local_row; \
    if (safe_row < qkv_m) { \
        A = A_qkv; \
        local_row = safe_row; \
    } else if (safe_row < qkv_m + z_m) { \
        A = A_z; \
        local_row = safe_row - qkv_m; \
    } else if (safe_row < qkv_m + z_m + beta_m) { \
        A = A_beta; \
        local_row = safe_row - (qkv_m + z_m); \
    } else { \
        A = A_alpha; \
        local_row = safe_row - (qkv_m + z_m + beta_m); \
    } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xb[BV]; \
        for (int b = 0; b < (BV); b++) \
            xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); \
            } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < total_m) { \
                    float* Y; \
                    int proj_row; \
                    int out_stride; \
                    if (out_row < qkv_m) { \
                        Y = Y_qkv; \
                        proj_row = out_row; \
                        out_stride = qkv_m; \
                    } else if (out_row < qkv_m + z_m) { \
                        Y = Y_z; \
                        proj_row = out_row - qkv_m; \
                        out_stride = z_m; \
                    } else if (out_row < qkv_m + z_m + beta_m) { \
                        Y = Y_beta; \
                        proj_row = out_row - (qkv_m + z_m); \
                        out_stride = beta_m; \
                    } else { \
                        Y = Y_alpha; \
                        proj_row = out_row - (qkv_m + z_m + beta_m); \
                        out_stride = alpha_m; \
                    } \
                    Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; \
                } \
            } \
        } \
    } \
}

// ---------------------------------------------------------------------------
// QKV overwrite BT4 — three projection regions (Q/K/V)
// Grid: [ceil((q_m+k_m+v_m)/16), ceil(N/(16*BV))]. Block: [32].
// ---------------------------------------------------------------------------
#define GEN_QKV_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_qkv_mq##BITS##g256v2_wmma_gfx11_bt##BV( \
    const char* __restrict__ A_q, \
    const char* __restrict__ A_k, \
    const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, \
    float* __restrict__ Y_k, \
    float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, \
    int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = q_m + k_m + v_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; \
    int local_row; \
    if (safe_row < q_m) { \
        A = A_q; \
        local_row = safe_row; \
    } else if (safe_row < q_m + k_m) { \
        A = A_k; \
        local_row = safe_row - q_m; \
    } else { \
        A = A_v; \
        local_row = safe_row - q_m - k_m; \
    } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xb[BV]; \
        for (int b = 0; b < (BV); b++) \
            xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); \
            } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < total_m) { \
                    float* Y; \
                    int proj_row; \
                    int out_stride; \
                    if (out_row < q_m) { \
                        Y = Y_q; \
                        proj_row = out_row; \
                        out_stride = q_m; \
                    } else if (out_row < q_m + k_m) { \
                        Y = Y_k; \
                        proj_row = out_row - q_m; \
                        out_stride = k_m; \
                    } else { \
                        Y = Y_v; \
                        proj_row = out_row - q_m - k_m; \
                        out_stride = v_m; \
                    } \
                    Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; \
                } \
            } \
        } \
    } \
}

// ---------------------------------------------------------------------------
// Gate/Up overwrite BT12 — two projection regions (gate/up)
// Grid: [ceil((gate_m+up_m)/16), ceil(N/(16*BV))]. Block: [32].
// ---------------------------------------------------------------------------
#define GEN_GATE_UP_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_gate_up_mq##BITS##g256v2_wmma_gfx11_bt##BV( \
    const char* __restrict__ A_gate, \
    const char* __restrict__ A_up, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_gate, \
    float* __restrict__ Y_up, \
    int gate_m, int up_m, \
    int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = gate_m + up_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; \
    int local_row; \
    if (safe_row < gate_m) { \
        A = A_gate; \
        local_row = safe_row; \
    } else { \
        A = A_up; \
        local_row = safe_row - gate_m; \
    } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xb[BV]; \
        for (int b = 0; b < (BV); b++) \
            xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); \
            } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < total_m) { \
                    float* Y; \
                    int proj_row; \
                    int out_stride; \
                    if (out_row < gate_m) { \
                        Y = Y_gate; \
                        proj_row = out_row; \
                        out_stride = gate_m; \
                    } else { \
                        Y = Y_up; \
                        proj_row = out_row - gate_m; \
                        out_stride = up_m; \
                    } \
                    Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; \
                } \
            } \
        } \
    } \
}

// ---------------------------------------------------------------------------
// Residual += BT4 — single projection, Y += W@X
// Grid: [ceil(M/16), ceil(N/(16*BV))]. Block: [32].
// Ascending WMMA order (kt = 0..15), matching MQ4 residual base.
// ---------------------------------------------------------------------------
#define GEN_RESID_BT(BITS, BV) \
extern "C" __launch_bounds__(32, 2) __global__ void \
gemm_mq##BITS##g256v2_residual_wmma_gfx11_bt##BV( \
    const char* __restrict__ A, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y, \
    int M, int K, int N \
) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * (BV)); \
    const int tid = threadIdx.x; \
    if (row_start >= M) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1); \
    const int groups_per_row = K / 256; \
    const char* row_base = A + (long long)safe_row * groups_per_row * GROUP_BYTES; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { \
        int t = batch_start + b * 16 + ml; \
        sb[b] = (t < N) ? t : 0; \
    } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) \
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        const char* gp = row_base + g * GROUP_BYTES; \
        const unsigned int hA = *(const unsigned int*)(gp); \
        const unsigned int hB = *(const unsigned int*)(gp + 4); \
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu))); \
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16))); \
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu))); \
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16))); \
        const unsigned char* payload = (const unsigned char*)(gp + 8); \
        const _Float16* xg[BV]; \
        for (int b = 0; b < (BV); b++) \
            xg[b] = X + (long long)sb[b] * K + g * 256; \
        _Pragma("unroll") \
        for (int kt = 0; kt < 16; kt++) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off = kt * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") \
            for (int b = 0; b < (BV); b++) { \
                half16_t b_reg = *(const half16_t*)(xg[b] + k_off); \
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]); \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int b = 0; b < (BV); b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { \
            _Pragma("unroll") \
            for (int j = 0; j < 8; j++) { \
                int out_row = row_start + 2 * j + (tid >> 4); \
                if (out_row < M) { \
                    Y[(long long)out_col * M + out_row] += acc[b][j]; \
                } \
            } \
        } \
    } \
}

// Instantiations: BITS ∈ {2,3,5,6}; QKVZA/QKV/residual BT4; gate_up BT12.
// gfx1100 policy extras (bits 3/5/6 only — no MQ2 production variants):
//   QKVZA/QKV BT12, gate_up BT6, residual BT6/BT8.
GEN_QKVZA_BT(2, 4)
GEN_QKVZA_BT(3, 4)
GEN_QKVZA_BT(5, 4)
GEN_QKVZA_BT(6, 4)

GEN_QKV_BT(2, 4)
GEN_QKV_BT(3, 4)
GEN_QKV_BT(5, 4)
GEN_QKV_BT(6, 4)

GEN_GATE_UP_BT(2, 12)
GEN_GATE_UP_BT(3, 12)
GEN_GATE_UP_BT(5, 12)
GEN_GATE_UP_BT(6, 12)

GEN_RESID_BT(2, 4)
GEN_RESID_BT(3, 4)
GEN_RESID_BT(5, 4)
GEN_RESID_BT(6, 4)

// gfx1100 policy tiles for MQ{3,5,6}V2 (mirror accepted MQ4 BT set).
GEN_QKVZA_BT(3, 12)
GEN_QKVZA_BT(5, 12)
GEN_QKVZA_BT(6, 12)

GEN_QKV_BT(3, 12)
GEN_QKV_BT(5, 12)
GEN_QKV_BT(6, 12)

GEN_GATE_UP_BT(3, 6)
GEN_GATE_UP_BT(5, 6)
GEN_GATE_UP_BT(6, 6)

GEN_RESID_BT(3, 6)
GEN_RESID_BT(5, 6)
GEN_RESID_BT(6, 6)

GEN_RESID_BT(3, 8)
GEN_RESID_BT(5, 8)
GEN_RESID_BT(6, 8)

#endif // __gfx1100__ || __gfx1151__
