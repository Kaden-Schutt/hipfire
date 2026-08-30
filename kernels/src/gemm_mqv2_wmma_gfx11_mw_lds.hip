// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Generic multi-wave same-row LDS prefill for MQ{3,4,5,6}G256V2 on gfx11.
// Same-row LDS / barrier / tail structure authority:
//   gemm_gate_up_mq4g256v2_wmma_gfx1100_mw_lds.hip
//   gemm_mq4g256v2_residual_wmma_gfx1100_mw_lds.hip
// Compile-time bit decoder authority:
//   gemm_mqv2_wmma_gfx11_bt.hip  (decode_tile<BITS>)
// Dword-window decoder + per-BITS selection + pad-18 LDS layout measured in
// tools/kernels/{gfx1100,gfx1151} microbench screen, 2026-08-30 (bit-exact,
// >=3 fresh-process runs; see .codeinsight+research/kernels/SYNTHESIS.md).
//
// One workgroup owns a single 16-row global tile. NW waves own distinct
// adjacent 16-column N tiles. For each 256-weight group the block cooperatively
// stages the dequantized 16x256 half tile into static LDS (tile-major
// [kt][row][ki], innermost dim padded 16->18 halfs: 36 B row stride,
// gcd(9,32)=1 bank-conflict-free; 9216 B / 4608 fp16), then every wave
// reuses that A fragment against its private X columns.
//
// V2 group: 8 B dual-FP16 header + contiguous LSB-first payload.
// GROUP_BYTES = 8 + 32*BITS. Half header selected by segment < 4 (kt < 8).
// Raw decode: sc_h * (_Float16)(float)q + zp_h via decode_stage (byte-load
// decode_tile for BITS<=4, dword-window decode_tile_window for BITS>=5).
//
// Staging: 128 logical jobs in a strided loop (job=tid; job<128; job+=32*NW).
// job row=job/8, segment=job%8. Each job owns two K-tiles (kt0=2*segment,
// kt1=kt0+1) and writes each LDS slot once per group. All 32*NW threads
// execute both __syncthreads per group; no early returns around barriers.
// M tails duplicate the last valid row into invalid LDS rows (discarded by
// out_row bounds). Partial N lanes and whole tail waves read X batch 0 and
// discard via out_col<N.
//
// Grid:  [ceil(M_or_total_m/16), ceil(N/(16*NW)), 1]
// Block: [32*NW, 1, 1]. Static LDS 9216 B; dynamic shared_mem=0.
// Instantiated: BITS ∈ {3,4,5,6} × MW4/MW8 for gate_up (overwrite),
// residual (+=), qkv (overwrite), qkvza (overwrite).
// Exact gfx1100 / gfx1151 only.
// Symbols:
//   gemm_gate_up_mq{BITS}g256v2_wmma_gfx11_mw{4|8}_lds
//   gemm_mq{BITS}g256v2_residual_wmma_gfx11_mw{4|8}_lds
//   gemm_qkv_mq{BITS}g256v2_wmma_gfx11_mw{4|8}_lds
//   gemm_qkvza_mq{BITS}g256v2_wmma_gfx11_mw{4|8}_lds

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

// Dword-window variant: the 16-code K-tile spans exactly 2*BITS bytes at byte
// offset kt*2*BITS from the 8-byte-aligned payload. Load the covering aligned
// u64s (2nd/3rd conditional — the last tile of a group ends exactly at the
// payload end, so unconditional trailing loads would read past the last group
// of the last row) and extract codes with register shifts. Bit-equal to
// decode_tile; kt may be a runtime value.
template <int BITS>
__device__ __forceinline__ half16_t decode_tile_window(const unsigned char* __restrict__ payload,
                                                       int kt,
                                                       _Float16 sc_h,
                                                       _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    const int off = kt * 2 * BITS;
    const int idx = off >> 3;
    const int bit_in = (off & 7) * 8;
    const unsigned long long* __restrict__ pu64 = (const unsigned long long* __restrict__)payload;
    unsigned long long w0 = pu64[idx];
    unsigned long long w1 = 0, w2 = 0;
    if ((bit_in + 16 * BITS) > 64) w1 = pu64[idx + 1];
    if ((bit_in + 16 * BITS) > 128) w2 = pu64[idx + 2];
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int b = bit_in + i * BITS;
        unsigned int q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned int)((w0 >> b) & qmask);
            else q = (unsigned int)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else if (b < 128) {
            int b2 = b - 64;
            if (b2 + BITS <= 64) q = (unsigned int)((w1 >> b2) & qmask);
            else q = (unsigned int)(((w1 >> b2) | (w2 << (64 - b2))) & qmask);
        } else {
            q = (unsigned int)((w2 >> (b - 128)) & qmask);
        }
        out[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return out;
}

// Staging decoder selection, measured per-BITS (tools/kernels screen,
// 2026-08-30): the dword window wins for the wider payloads (BITS >= 5:
// +4-15% in MW staging on gfx1151, +18% in BT context on gfx1100) and loses
// for BITS <= 4 (bits3 MW staging: -13% vs byte loads). Both are bit-equal.
template <int BITS>
__device__ __forceinline__ half16_t decode_stage(const unsigned char* __restrict__ payload,
                                                 int kt,
                                                 _Float16 sc_h,
                                                 _Float16 zp_h) {
    if constexpr (BITS >= 5)
        return decode_tile_window<BITS>(payload, kt, sc_h, zp_h);
    else
        return decode_tile<BITS>(payload, kt, sc_h, zp_h);
}

// ---------------------------------------------------------------------------
// Gate/Up overwrite MW — two projection regions (gate/up)
// Grid: [ceil((gate_m+up_m)/16), ceil(N/(16*NW))]. Block: [32*NW].
// ---------------------------------------------------------------------------
#define GEN_GATE_UP_MW_LDS(BITS, NW)                                                           \
    extern "C" __launch_bounds__(32 * (NW), 1) __global__ void                                 \
        gemm_gate_up_mq##BITS##g256v2_wmma_gfx11_mw##NW##_lds(                                 \
            const char* __restrict__ A_gate,                                                   \
            const char* __restrict__ A_up,                                                     \
            const _Float16* __restrict__ X,                                                    \
            float* __restrict__ Y_gate,                                                        \
            float* __restrict__ Y_up,                                                          \
            int gate_m,                                                                        \
            int up_m,                                                                          \
            int K,                                                                             \
            int N) {                                                                           \
        constexpr int GROUP_BYTES = 8 + 32 * (BITS);                                           \
        const int tid = threadIdx.x;                                                           \
        const int lane = tid & 31;                                                             \
        const int wave_id = tid >> 5;                                                          \
        const int ml = lane & 15;                                                              \
        const int total_m = gate_m + up_m;                                                     \
        const int row_start = blockIdx.x * 16;                                                 \
        const int n_block_start = blockIdx.y * (16 * (NW));                                    \
        const int out_col = n_block_start + wave_id * 16 + ml;                                 \
        const int safe_batch = (out_col < N) ? out_col : 0;                                    \
        const int groups_per_row = K / 256;                                                    \
                                                                                               \
        /* Tile-major [kt][row][ki], ki padded to 18: 16*16*18 half = 9216 B. */               \
        __shared__ __align__(32) _Float16 w_lds[16][16][18];                                    \
                                                                                               \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                        \
                                                                                               \
        for (int g = 0; g < groups_per_row; g++) {                                             \
            /* 128 logical staging jobs: 16 rows × 8 segments. */                              \
            for (int job = tid; job < 128; job += 32 * (NW)) {                                 \
                const int row = job / 8;                                                       \
                const int segment = job % 8;                                                   \
                const int safe_row =                                                           \
                    (row_start + row < total_m) ? (row_start + row) : (total_m - 1);            \
                const char* A;                                                                 \
                int local_row;                                                                 \
                if (safe_row < gate_m) {                                                       \
                    A = A_gate;                                                                \
                    local_row = safe_row;                                                      \
                } else {                                                                       \
                    A = A_up;                                                                  \
                    local_row = safe_row - gate_m;                                             \
                }                                                                              \
                const char* gp = A + (long long)local_row * groups_per_row * GROUP_BYTES +      \
                                 (long long)g * GROUP_BYTES;                                   \
                /* One relevant dual-half V2 header for this segment's K half. */              \
                const unsigned int h =                                                         \
                    *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4));                      \
                const _Float16 sc_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));    \
                const _Float16 zp_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));        \
                const unsigned char* payload = (const unsigned char*)(gp + 8);                 \
                const int kt0 = 2 * segment;                                                   \
                const int kt1 = kt0 + 1;                                                       \
                half16_t frag = decode_stage<BITS>(payload, kt0, sc_h, zp_h);                  \
                *(half16_t*)(&w_lds[kt0][row][0]) = frag;                                      \
                frag = decode_stage<BITS>(payload, kt1, sc_h, zp_h);                           \
                *(half16_t*)(&w_lds[kt1][row][0]) = frag;                                      \
            }                                                                                  \
            /* All threads reach both barriers every group — no early returns. */              \
            __syncthreads();                                                                   \
                                                                                               \
            /* K2 ascending: kt pairs 0,1; 2,3; ... 14,15. One float8 acc / wave. */           \
            const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256;           \
            _Pragma("unroll")                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                               \
                half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]);                        \
                half16_t b_reg = *(const half16_t*)(xg + kt * 16);                             \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
                a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]);                             \
                b_reg = *(const half16_t*)(xg + (kt + 1) * 16);                                \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
            }                                                                                  \
                                                                                               \
            __syncthreads();                                                                   \
        }                                                                                      \
                                                                                               \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. Overwrite (=). */        \
        if (out_col < N) {                                                                     \
            _Pragma("unroll")                                                                  \
            for (int j = 0; j < 8; j++) {                                                      \
                const int out_row = row_start + 2 * j + (lane >> 4);                            \
                if (out_row < total_m) {                                                       \
                    if (out_row < gate_m) {                                                    \
                        Y_gate[(long long)out_col * gate_m + out_row] = acc[j];                \
                    } else {                                                                   \
                        Y_up[(long long)out_col * up_m + (out_row - gate_m)] = acc[j];         \
                    }                                                                          \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
    }

// ---------------------------------------------------------------------------
// Residual += MW — single projection, Y += W@X
// Grid: [ceil(M/16), ceil(N/(16*NW))]. Block: [32*NW].
// ---------------------------------------------------------------------------
#define GEN_RESID_MW_LDS(BITS, NW)                                                             \
    extern "C" __launch_bounds__(32 * (NW), 1) __global__ void                                 \
        gemm_mq##BITS##g256v2_residual_wmma_gfx11_mw##NW##_lds(                                \
            const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
            int M, int K, int N) {                                                             \
        constexpr int GROUP_BYTES = 8 + 32 * (BITS);                                           \
        const int tid = threadIdx.x;                                                           \
        const int lane = tid & 31;                                                             \
        const int wave_id = tid >> 5;                                                          \
        const int ml = lane & 15;                                                              \
        const int row_start = blockIdx.x * 16;                                                 \
        const int n_block_start = blockIdx.y * (16 * (NW));                                    \
        const int out_col = n_block_start + wave_id * 16 + ml;                                 \
        const int safe_batch = (out_col < N) ? out_col : 0;                                    \
        const int groups_per_row = K / 256;                                                    \
                                                                                               \
        /* Tile-major [kt][row][ki], ki padded to 18: 16*16*18 half = 9216 B. */               \
        __shared__ __align__(32) _Float16 w_lds[16][16][18];                                    \
                                                                                               \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                        \
                                                                                               \
        for (int g = 0; g < groups_per_row; g++) {                                             \
            /* 128 logical staging jobs: 16 rows × 8 segments. */                              \
            for (int job = tid; job < 128; job += 32 * (NW)) {                                 \
                const int row = job / 8;                                                       \
                const int segment = job % 8;                                                   \
                const int safe_row = (row_start + row < M) ? (row_start + row) : (M - 1);       \
                const char* gp = A + (long long)safe_row * groups_per_row * GROUP_BYTES +      \
                                 (long long)g * GROUP_BYTES;                                   \
                /* One relevant dual-half V2 header for this segment's K half. */              \
                const unsigned int h =                                                         \
                    *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4));                      \
                const _Float16 sc_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));    \
                const _Float16 zp_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));        \
                const unsigned char* payload = (const unsigned char*)(gp + 8);                 \
                const int kt0 = 2 * segment;                                                   \
                const int kt1 = kt0 + 1;                                                       \
                half16_t frag = decode_stage<BITS>(payload, kt0, sc_h, zp_h);                  \
                *(half16_t*)(&w_lds[kt0][row][0]) = frag;                                      \
                frag = decode_stage<BITS>(payload, kt1, sc_h, zp_h);                           \
                *(half16_t*)(&w_lds[kt1][row][0]) = frag;                                      \
            }                                                                                  \
            /* All threads reach both barriers every group — no early returns. */              \
            __syncthreads();                                                                   \
                                                                                               \
            /* K2 ascending: kt pairs 0,1; 2,3; ... 14,15. One float8 acc / wave. */           \
            const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256;           \
            _Pragma("unroll")                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                               \
                half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]);                        \
                half16_t b_reg = *(const half16_t*)(xg + kt * 16);                             \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
                a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]);                             \
                b_reg = *(const half16_t*)(xg + (kt + 1) * 16);                                \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
            }                                                                                  \
                                                                                               \
            __syncthreads();                                                                   \
        }                                                                                      \
                                                                                               \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. Unique Y+= owner. */     \
        if (out_col < N) {                                                                     \
            _Pragma("unroll")                                                                  \
            for (int j = 0; j < 8; j++) {                                                      \
                const int out_row = row_start + 2 * j + (lane >> 4);                            \
                if (out_row < M) {                                                             \
                    Y[(long long)out_col * M + out_row] += acc[j];                             \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
    }

// ---------------------------------------------------------------------------
// QKV overwrite MW — three projection regions (Q/K/V)
// Grid: [ceil((q_m+k_m+v_m)/16), ceil(N/(16*NW))]. Block: [32*NW].
// Region routing mirrors gemm_qkv_*_wmma_gfx11_bt; measured 1.42x vs BT12 on
// gfx1100 (N=512) and 3.1-3.4x vs BT4 on gfx1151, bit-exact.
// ---------------------------------------------------------------------------
#define GEN_QKV_MW_LDS(BITS, NW)                                                               \
    extern "C" __launch_bounds__(32 * (NW), 1) __global__ void                                 \
        gemm_qkv_mq##BITS##g256v2_wmma_gfx11_mw##NW##_lds(                                     \
            const char* __restrict__ A_q,                                                      \
            const char* __restrict__ A_k,                                                      \
            const char* __restrict__ A_v,                                                      \
            const _Float16* __restrict__ X,                                                    \
            float* __restrict__ Y_q,                                                           \
            float* __restrict__ Y_k,                                                           \
            float* __restrict__ Y_v,                                                           \
            int q_m,                                                                           \
            int k_m,                                                                           \
            int v_m,                                                                           \
            int K,                                                                             \
            int N) {                                                                           \
        constexpr int GROUP_BYTES = 8 + 32 * (BITS);                                           \
        const int tid = threadIdx.x;                                                           \
        const int lane = tid & 31;                                                             \
        const int wave_id = tid >> 5;                                                          \
        const int ml = lane & 15;                                                              \
        const int total_m = q_m + k_m + v_m;                                                   \
        const int row_start = blockIdx.x * 16;                                                 \
        const int n_block_start = blockIdx.y * (16 * (NW));                                    \
        const int out_col = n_block_start + wave_id * 16 + ml;                                 \
        const int safe_batch = (out_col < N) ? out_col : 0;                                    \
        const int groups_per_row = K / 256;                                                    \
                                                                                               \
        /* Tile-major [kt][row][ki], ki padded to 18: 16*16*18 half = 9216 B. */               \
        __shared__ __align__(32) _Float16 w_lds[16][16][18];                                    \
                                                                                               \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                        \
                                                                                               \
        for (int g = 0; g < groups_per_row; g++) {                                             \
            /* 128 logical staging jobs: 16 rows × 8 segments. */                              \
            for (int job = tid; job < 128; job += 32 * (NW)) {                                 \
                const int row = job / 8;                                                       \
                const int segment = job % 8;                                                   \
                const int safe_row =                                                           \
                    (row_start + row < total_m) ? (row_start + row) : (total_m - 1);            \
                const char* A;                                                                 \
                int local_row;                                                                 \
                if (safe_row < q_m) {                                                          \
                    A = A_q;                                                                   \
                    local_row = safe_row;                                                      \
                } else if (safe_row < q_m + k_m) {                                             \
                    A = A_k;                                                                   \
                    local_row = safe_row - q_m;                                                \
                } else {                                                                       \
                    A = A_v;                                                                   \
                    local_row = safe_row - q_m - k_m;                                          \
                }                                                                              \
                const char* gp = A + (long long)local_row * groups_per_row * GROUP_BYTES +      \
                                 (long long)g * GROUP_BYTES;                                   \
                /* One relevant dual-half V2 header for this segment's K half. */              \
                const unsigned int h =                                                         \
                    *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4));                      \
                const _Float16 sc_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));    \
                const _Float16 zp_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));        \
                const unsigned char* payload = (const unsigned char*)(gp + 8);                 \
                const int kt0 = 2 * segment;                                                   \
                const int kt1 = kt0 + 1;                                                       \
                half16_t frag = decode_stage<BITS>(payload, kt0, sc_h, zp_h);                  \
                *(half16_t*)(&w_lds[kt0][row][0]) = frag;                                      \
                frag = decode_stage<BITS>(payload, kt1, sc_h, zp_h);                           \
                *(half16_t*)(&w_lds[kt1][row][0]) = frag;                                      \
            }                                                                                  \
            /* All threads reach both barriers every group — no early returns. */              \
            __syncthreads();                                                                   \
                                                                                               \
            /* K2 ascending: kt pairs 0,1; 2,3; ... 14,15. One float8 acc / wave. */           \
            const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256;           \
            _Pragma("unroll")                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                               \
                half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]);                        \
                half16_t b_reg = *(const half16_t*)(xg + kt * 16);                             \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
                a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]);                             \
                b_reg = *(const half16_t*)(xg + (kt + 1) * 16);                                \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
            }                                                                                  \
                                                                                               \
            __syncthreads();                                                                   \
        }                                                                                      \
                                                                                               \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. Overwrite (=). */        \
        if (out_col < N) {                                                                     \
            _Pragma("unroll")                                                                  \
            for (int j = 0; j < 8; j++) {                                                      \
                const int out_row = row_start + 2 * j + (lane >> 4);                            \
                if (out_row < total_m) {                                                       \
                    if (out_row < q_m) {                                                       \
                        Y_q[(long long)out_col * q_m + out_row] = acc[j];                      \
                    } else if (out_row < q_m + k_m) {                                          \
                        Y_k[(long long)out_col * k_m + (out_row - q_m)] = acc[j];              \
                    } else {                                                                   \
                        Y_v[(long long)out_col * v_m + (out_row - q_m - k_m)] = acc[j];        \
                    }                                                                          \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
    }

// ---------------------------------------------------------------------------
// QKVZA overwrite MW — four projection regions (QKV/Z/beta/alpha)
// Grid: [ceil((qkv_m+z_m+beta_m+alpha_m)/16), ceil(N/(16*NW))]. Block: [32*NW].
// Region routing mirrors gemm_qkvza_*_wmma_gfx11_bt.
// ---------------------------------------------------------------------------
#define GEN_QKVZA_MW_LDS(BITS, NW)                                                             \
    extern "C" __launch_bounds__(32 * (NW), 1) __global__ void                                 \
        gemm_qkvza_mq##BITS##g256v2_wmma_gfx11_mw##NW##_lds(                                   \
            const char* __restrict__ A_qkv,                                                    \
            const char* __restrict__ A_z,                                                      \
            const char* __restrict__ A_beta,                                                   \
            const char* __restrict__ A_alpha,                                                  \
            const _Float16* __restrict__ X,                                                    \
            float* __restrict__ Y_qkv,                                                         \
            float* __restrict__ Y_z,                                                           \
            float* __restrict__ Y_beta,                                                        \
            float* __restrict__ Y_alpha,                                                       \
            int qkv_m,                                                                         \
            int z_m,                                                                           \
            int beta_m,                                                                        \
            int alpha_m,                                                                       \
            int K,                                                                             \
            int N) {                                                                           \
        constexpr int GROUP_BYTES = 8 + 32 * (BITS);                                           \
        const int tid = threadIdx.x;                                                           \
        const int lane = tid & 31;                                                             \
        const int wave_id = tid >> 5;                                                          \
        const int ml = lane & 15;                                                              \
        const int total_m = qkv_m + z_m + beta_m + alpha_m;                                    \
        const int row_start = blockIdx.x * 16;                                                 \
        const int n_block_start = blockIdx.y * (16 * (NW));                                    \
        const int out_col = n_block_start + wave_id * 16 + ml;                                 \
        const int safe_batch = (out_col < N) ? out_col : 0;                                    \
        const int groups_per_row = K / 256;                                                    \
                                                                                               \
        /* Tile-major [kt][row][ki], ki padded to 18: 16*16*18 half = 9216 B. */               \
        __shared__ __align__(32) _Float16 w_lds[16][16][18];                                    \
                                                                                               \
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                        \
                                                                                               \
        for (int g = 0; g < groups_per_row; g++) {                                             \
            /* 128 logical staging jobs: 16 rows × 8 segments. */                              \
            for (int job = tid; job < 128; job += 32 * (NW)) {                                 \
                const int row = job / 8;                                                       \
                const int segment = job % 8;                                                   \
                const int safe_row =                                                           \
                    (row_start + row < total_m) ? (row_start + row) : (total_m - 1);            \
                const char* A;                                                                 \
                int local_row;                                                                 \
                if (safe_row < qkv_m) {                                                        \
                    A = A_qkv;                                                                 \
                    local_row = safe_row;                                                      \
                } else if (safe_row < qkv_m + z_m) {                                           \
                    A = A_z;                                                                   \
                    local_row = safe_row - qkv_m;                                              \
                } else if (safe_row < qkv_m + z_m + beta_m) {                                  \
                    A = A_beta;                                                                \
                    local_row = safe_row - (qkv_m + z_m);                                      \
                } else {                                                                       \
                    A = A_alpha;                                                               \
                    local_row = safe_row - (qkv_m + z_m + beta_m);                             \
                }                                                                              \
                const char* gp = A + (long long)local_row * groups_per_row * GROUP_BYTES +      \
                                 (long long)g * GROUP_BYTES;                                   \
                /* One relevant dual-half V2 header for this segment's K half. */              \
                const unsigned int h =                                                         \
                    *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4));                      \
                const _Float16 sc_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));    \
                const _Float16 zp_h =                                                          \
                    (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16)));        \
                const unsigned char* payload = (const unsigned char*)(gp + 8);                 \
                const int kt0 = 2 * segment;                                                   \
                const int kt1 = kt0 + 1;                                                       \
                half16_t frag = decode_stage<BITS>(payload, kt0, sc_h, zp_h);                  \
                *(half16_t*)(&w_lds[kt0][row][0]) = frag;                                      \
                frag = decode_stage<BITS>(payload, kt1, sc_h, zp_h);                           \
                *(half16_t*)(&w_lds[kt1][row][0]) = frag;                                      \
            }                                                                                  \
            /* All threads reach both barriers every group — no early returns. */              \
            __syncthreads();                                                                   \
                                                                                               \
            /* K2 ascending: kt pairs 0,1; 2,3; ... 14,15. One float8 acc / wave. */           \
            const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256;           \
            _Pragma("unroll")                                                                  \
            for (int kt = 0; kt < 16; kt += 2) {                                               \
                half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]);                        \
                half16_t b_reg = *(const half16_t*)(xg + kt * 16);                             \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
                a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]);                             \
                b_reg = *(const half16_t*)(xg + (kt + 1) * 16);                                \
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);           \
            }                                                                                  \
                                                                                               \
            __syncthreads();                                                                   \
        }                                                                                      \
                                                                                               \
        /* RDNA3 wave32 WMMA: acc[j] = C[2*j + (lane>>4)][lane & 15]. Overwrite (=). */        \
        if (out_col < N) {                                                                     \
            _Pragma("unroll")                                                                  \
            for (int j = 0; j < 8; j++) {                                                      \
                const int out_row = row_start + 2 * j + (lane >> 4);                            \
                if (out_row < total_m) {                                                       \
                    if (out_row < qkv_m) {                                                     \
                        Y_qkv[(long long)out_col * qkv_m + out_row] = acc[j];                  \
                    } else if (out_row < qkv_m + z_m) {                                        \
                        Y_z[(long long)out_col * z_m + (out_row - qkv_m)] = acc[j];            \
                    } else if (out_row < qkv_m + z_m + beta_m) {                               \
                        Y_beta[(long long)out_col * beta_m + (out_row - (qkv_m + z_m))] =      \
                            acc[j];                                                            \
                    } else {                                                                   \
                        Y_alpha[(long long)out_col * alpha_m +                                 \
                                (out_row - (qkv_m + z_m + beta_m))] = acc[j];                  \
                    }                                                                          \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
    }

// Instantiations: BITS ∈ {3,4,5,6} × NW ∈ {4,8} × 4 families — 32 extern-C symbols.
GEN_GATE_UP_MW_LDS(3, 4)
GEN_GATE_UP_MW_LDS(3, 8)
GEN_GATE_UP_MW_LDS(4, 4)
GEN_GATE_UP_MW_LDS(4, 8)
GEN_GATE_UP_MW_LDS(5, 4)
GEN_GATE_UP_MW_LDS(5, 8)
GEN_GATE_UP_MW_LDS(6, 4)
GEN_GATE_UP_MW_LDS(6, 8)

GEN_RESID_MW_LDS(3, 4)
GEN_RESID_MW_LDS(3, 8)
GEN_RESID_MW_LDS(4, 4)
GEN_RESID_MW_LDS(4, 8)
GEN_RESID_MW_LDS(5, 4)
GEN_RESID_MW_LDS(5, 8)
GEN_RESID_MW_LDS(6, 4)
GEN_RESID_MW_LDS(6, 8)

GEN_QKV_MW_LDS(3, 4)
GEN_QKV_MW_LDS(3, 8)
GEN_QKV_MW_LDS(4, 4)
GEN_QKV_MW_LDS(4, 8)
GEN_QKV_MW_LDS(5, 4)
GEN_QKV_MW_LDS(5, 8)
GEN_QKV_MW_LDS(6, 4)
GEN_QKV_MW_LDS(6, 8)

GEN_QKVZA_MW_LDS(3, 4)
GEN_QKVZA_MW_LDS(3, 8)
GEN_QKVZA_MW_LDS(4, 4)
GEN_QKVZA_MW_LDS(4, 8)
GEN_QKVZA_MW_LDS(5, 4)
GEN_QKVZA_MW_LDS(5, 8)
GEN_QKVZA_MW_LDS(6, 4)
GEN_QKVZA_MW_LDS(6, 8)

#endif // __gfx1100__ || __gfx1151__
