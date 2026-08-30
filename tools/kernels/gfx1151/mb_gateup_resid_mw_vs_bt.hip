// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mb-gateup-resid-mw-vs-bt-gfx1151-v1-20260830
//
// QUESTION: does MW-LDS (pad-18 + decode_stage) beat the production BT arm for
// gate_up (BT12) and residual (BT4) on gfx1151?
//
// Production constructs interrogated (arm 0, verbatim):
//   gate_up: GEN_GATE_UP_BT(BITS, 12)  kernels/src/gemm_mqv2_wmma_gfx11_bt.hip
//   resid:   GEN_RESID_BT(BITS, 4)     kernels/src/gemm_mqv2_wmma_gfx11_bt.hip
// Candidate arms (verbatim GEN_*_MW_LDS + pad18 + decode_stage from
//   kernels/src/gemm_mqv2_wmma_gfx11_mw_lds.hip): MW4 / MW8.
//
// Semantics:
//   gate_up: overwrite Y_gate/Y_up. Fixture gate_m=up_m=12288, K=5120.
//   resid:   Y += W@X. Fixture Y prefilled nonzero; BITEQ on final Y.
//            M=5120, K=12288.
//
// Correctness: MW arms f32 to_bits-equal to BT (BITEQ); BT vs sampled host
// fp16-fma double reference rel_l2 <= 1e-4 (HOSTREF sanity).
//
// Build (fat: ROCm 7.15 rejects single-arch gfx1151 images):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 --offload-arch=gfx1151 -O3 \
//       tools/kernels/gfx1151/mb_gateup_resid_mw_vs_bt.hip -o mb_gateup_resid
// Run:
//   ./mb_gateup_resid --arch gfx1151 --op gateup|resid [--bits 3|5|6] [--n N]
//   Defaults: bits=3 samples=50 warmup=20
//     gateup: n=512 gate_m=up_m=12288 K=5120
//     resid:  n=512 M=5120 K=12288
//
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr const char* kBenchId = "mb-gateup-resid-mw-vs-bt-gfx1151-v1-20260830";
static constexpr int kDefWarmup = 20;
static constexpr int kDefSamples = 50;
static constexpr int kBatchLaunches = 8; // independent A slabs per timed sample

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ std::fprintf(stderr,"HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} }while(0)

static inline int v2_group_bytes(int bits){ return 8 + 32*bits; }

// ---------------------------------------------------------------------------
// Device code — gfx11 only (host pass and other arches take the stub branch)
// ---------------------------------------------------------------------------
#if defined(__gfx1100__) || defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Verbatim production decoder (kernels/src/gemm_mqv2_wmma_gfx11_bt.hip).
template <int BITS>
__device__ __forceinline__ half16_t decode_tile(const unsigned char* __restrict__ payload,
                                                int kt, _Float16 sc_h, _Float16 zp_h) {
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

// Dword-window decoder (verbatim gemm_mqv2_wmma_gfx11_mw_lds.hip).
template <int BITS>
__device__ __forceinline__ half16_t decode_tile_window(const unsigned char* __restrict__ payload,
                                                       int kt, _Float16 sc_h, _Float16 zp_h) {
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

// Staging decoder selection (verbatim production MW): window for BITS>=5.
template <int BITS>
__device__ __forceinline__ half16_t decode_stage(const unsigned char* __restrict__ payload,
                                                 int kt, _Float16 sc_h, _Float16 zp_h) {
    if constexpr (BITS >= 5)
        return decode_tile_window<BITS>(payload, kt, sc_h, zp_h);
    else
        return decode_tile<BITS>(payload, kt, sc_h, zp_h);
}

// Arm 0 gate_up — verbatim GEN_GATE_UP_BT (BV=12) from gemm_mqv2_wmma_gfx11_bt.hip.
#define GEN_GATE_UP_BT12(BITS) \
extern "C" __launch_bounds__(32, 2) __global__ void \
mb_gate_up_mq##BITS##_bt12( \
    const char* __restrict__ A_gate, const char* __restrict__ A_up, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_gate, float* __restrict__ Y_up, \
    int gate_m, int up_m, int K, int N \
) { \
    constexpr int BV = 12; \
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
    const char* A; int local_row; \
    if (safe_row < gate_m) { A = A_gate; local_row = safe_row; } \
    else { A = A_up; local_row = safe_row - gate_m; } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < BV; b++) { int t = batch_start + b * 16 + ml; sb[b] = (t < N) ? t : 0; } \
    float8_t acc[BV]; \
    for (int b = 0; b < BV; b++) acc[b] = float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}; \
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
        for (int b = 0; b < BV; b++) xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") for (int b = 0; b < BV; b++) { half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") for (int b = 0; b < BV; b++) { half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); } \
        } \
    } \
    _Pragma("unroll") for (int b = 0; b < BV; b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { int out_row = row_start + 2 * j + (tid >> 4); if (out_row < total_m) { float* Y; int proj_row; int out_stride; if (out_row < gate_m) { Y = Y_gate; proj_row = out_row; out_stride = gate_m; } else { Y = Y_up; proj_row = out_row - gate_m; out_stride = up_m; } Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; } } } \
    } \
}

// Candidate gate_up — verbatim GEN_GATE_UP_MW_LDS (pad18 + decode_stage).
#define GEN_GATE_UP_MW(BITS, NW) \
extern "C" __launch_bounds__(32 * (NW), 1) __global__ void \
mb_gate_up_mq##BITS##_mw##NW( \
    const char* __restrict__ A_gate, const char* __restrict__ A_up, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_gate, float* __restrict__ Y_up, \
    int gate_m, int up_m, int K, int N) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int tid = threadIdx.x; \
    const int lane = tid & 31; \
    const int wave_id = tid >> 5; \
    const int ml = lane & 15; \
    const int total_m = gate_m + up_m; \
    const int row_start = blockIdx.x * 16; \
    const int n_block_start = blockIdx.y * (16 * (NW)); \
    const int out_col = n_block_start + wave_id * 16 + ml; \
    const int safe_batch = (out_col < N) ? out_col : 0; \
    const int groups_per_row = K / 256; \
    __shared__ __align__(32) _Float16 w_lds[16][16][18]; \
    float8_t acc = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        for (int job = tid; job < 128; job += 32 * (NW)) { \
            const int row = job / 8; \
            const int segment = job % 8; \
            const int safe_row = (row_start + row < total_m) ? (row_start + row) : (total_m - 1); \
            const char* A; int local_row; \
            if (safe_row < gate_m) { A = A_gate; local_row = safe_row; } \
            else { A = A_up; local_row = safe_row - gate_m; } \
            const char* gp = A + (long long)local_row * groups_per_row * GROUP_BYTES + (long long)g * GROUP_BYTES; \
            const unsigned int h = *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4)); \
            const _Float16 sc_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
            const _Float16 zp_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16))); \
            const unsigned char* payload = (const unsigned char*)(gp + 8); \
            const int kt0 = 2 * segment; const int kt1 = kt0 + 1; \
            half16_t frag = decode_stage<BITS>(payload, kt0, sc_h, zp_h); *(half16_t*)(&w_lds[kt0][row][0]) = frag; \
            frag = decode_stage<BITS>(payload, kt1, sc_h, zp_h); *(half16_t*)(&w_lds[kt1][row][0]) = frag; \
        } \
        __syncthreads(); \
        const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256; \
        _Pragma("unroll") for (int kt = 0; kt < 16; kt += 2) { \
            half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]); half16_t b_reg = *(const half16_t*)(xg + kt * 16); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc); \
            a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]); b_reg = *(const half16_t*)(xg + (kt + 1) * 16); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc); \
        } \
        __syncthreads(); \
    } \
    if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { const int out_row = row_start + 2 * j + (lane >> 4); if (out_row < total_m) { if (out_row < gate_m) { Y_gate[(long long)out_col * gate_m + out_row] = acc[j]; } else { Y_up[(long long)out_col * up_m + (out_row - gate_m)] = acc[j]; } } } } \
}

// Arm 0 residual — verbatim GEN_RESID_BT (BV=4).
#define GEN_RESID_BT4(BITS) \
extern "C" __launch_bounds__(32, 2) __global__ void \
mb_resid_mq##BITS##_bt4( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int N \
) { \
    constexpr int BV = 4; \
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
    for (int b = 0; b < BV; b++) { int t = batch_start + b * 16 + ml; sb[b] = (t < N) ? t : 0; } \
    float8_t acc[BV]; \
    for (int b = 0; b < BV; b++) acc[b] = float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}; \
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
        for (int b = 0; b < BV; b++) xg[b] = X + (long long)sb[b] * K + g * 256; \
        _Pragma("unroll") for (int kt = 0; kt < 16; kt++) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off = kt * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") for (int b = 0; b < BV; b++) { half16_t b_reg = *(const half16_t*)(xg[b] + k_off); acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]); } \
        } \
    } \
    _Pragma("unroll") for (int b = 0; b < BV; b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { int out_row = row_start + 2 * j + (tid >> 4); if (out_row < M) { Y[(long long)out_col * M + out_row] += acc[b][j]; } } } \
    } \
}

// Candidate residual — verbatim GEN_RESID_MW_LDS (pad18 + decode_stage).
#define GEN_RESID_MW(BITS, NW) \
extern "C" __launch_bounds__(32 * (NW), 1) __global__ void \
mb_resid_mq##BITS##_mw##NW( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int N) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int tid = threadIdx.x; \
    const int lane = tid & 31; \
    const int wave_id = tid >> 5; \
    const int ml = lane & 15; \
    const int row_start = blockIdx.x * 16; \
    const int n_block_start = blockIdx.y * (16 * (NW)); \
    const int out_col = n_block_start + wave_id * 16 + ml; \
    const int safe_batch = (out_col < N) ? out_col : 0; \
    const int groups_per_row = K / 256; \
    __shared__ __align__(32) _Float16 w_lds[16][16][18]; \
    float8_t acc = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        for (int job = tid; job < 128; job += 32 * (NW)) { \
            const int row = job / 8; \
            const int segment = job % 8; \
            const int safe_row = (row_start + row < M) ? (row_start + row) : (M - 1); \
            const char* gp = A + (long long)safe_row * groups_per_row * GROUP_BYTES + (long long)g * GROUP_BYTES; \
            const unsigned int h = *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4)); \
            const _Float16 sc_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
            const _Float16 zp_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16))); \
            const unsigned char* payload = (const unsigned char*)(gp + 8); \
            const int kt0 = 2 * segment; const int kt1 = kt0 + 1; \
            half16_t frag = decode_stage<BITS>(payload, kt0, sc_h, zp_h); *(half16_t*)(&w_lds[kt0][row][0]) = frag; \
            frag = decode_stage<BITS>(payload, kt1, sc_h, zp_h); *(half16_t*)(&w_lds[kt1][row][0]) = frag; \
        } \
        __syncthreads(); \
        const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256; \
        _Pragma("unroll") for (int kt = 0; kt < 16; kt += 2) { \
            half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]); half16_t b_reg = *(const half16_t*)(xg + kt * 16); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc); \
            a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]); b_reg = *(const half16_t*)(xg + (kt + 1) * 16); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc); \
        } \
        __syncthreads(); \
    } \
    if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { const int out_row = row_start + 2 * j + (lane >> 4); if (out_row < M) { Y[(long long)out_col * M + out_row] += acc[j]; } } } \
}

GEN_GATE_UP_BT12(3)
GEN_GATE_UP_BT12(5)
GEN_GATE_UP_BT12(6)
GEN_GATE_UP_MW(3, 4)
GEN_GATE_UP_MW(3, 8)
GEN_GATE_UP_MW(5, 4)
GEN_GATE_UP_MW(5, 8)
GEN_GATE_UP_MW(6, 4)
GEN_GATE_UP_MW(6, 8)

GEN_RESID_BT4(3)
GEN_RESID_BT4(5)
GEN_RESID_BT4(6)
GEN_RESID_MW(3, 4)
GEN_RESID_MW(3, 8)
GEN_RESID_MW(5, 4)
GEN_RESID_MW(5, 8)
GEN_RESID_MW(6, 4)
GEN_RESID_MW(6, 8)

#else // host pass / other arches — matching stubs
#define GEN_STUB_GU(NAME) extern "C" __global__ void NAME(const char*,const char*,const _Float16*,float*,float*,int,int,int,int){}
#define GEN_STUB_R(NAME)  extern "C" __global__ void NAME(const char*,const _Float16*,float*,int,int,int){}
GEN_STUB_GU(mb_gate_up_mq3_bt12) GEN_STUB_GU(mb_gate_up_mq5_bt12) GEN_STUB_GU(mb_gate_up_mq6_bt12)
GEN_STUB_GU(mb_gate_up_mq3_mw4) GEN_STUB_GU(mb_gate_up_mq3_mw8)
GEN_STUB_GU(mb_gate_up_mq5_mw4) GEN_STUB_GU(mb_gate_up_mq5_mw8)
GEN_STUB_GU(mb_gate_up_mq6_mw4) GEN_STUB_GU(mb_gate_up_mq6_mw8)
GEN_STUB_R(mb_resid_mq3_bt4) GEN_STUB_R(mb_resid_mq5_bt4) GEN_STUB_R(mb_resid_mq6_bt4)
GEN_STUB_R(mb_resid_mq3_mw4) GEN_STUB_R(mb_resid_mq3_mw8)
GEN_STUB_R(mb_resid_mq5_mw4) GEN_STUB_R(mb_resid_mq5_mw8)
GEN_STUB_R(mb_resid_mq6_mw4) GEN_STUB_R(mb_resid_mq6_mw8)
#undef GEN_STUB_GU
#undef GEN_STUB_R
#endif

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
static inline uint64_t lcg_u64(uint64_t* s){ *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s; }
static inline float lcg_unit(uint64_t* s){ return (float)(lcg_u64(s)>>40)*(1.f/(float)(1u<<24)); }
static inline float lcg_finite(uint64_t* s,float lo,float hi){ return lo + (hi-lo)*lcg_unit(s); }

static inline uint16_t f32_to_f16_bits(float v){
    uint32_t x; std::memcpy(&x,&v,4);
    const uint32_t sign=(x>>16)&0x8000u;
    int32_t exp=(int32_t)((x>>23)&0xFFu)-127+15;
    uint32_t mant=x&0x7FFFFFu;
    if(exp<=0){
        if(exp<-10) return (uint16_t)sign;
        mant|=0x800000u;
        const uint32_t shift=(uint32_t)(1-exp);
        uint32_t half=mant>>(shift+13);
        const uint32_t rem=mant&((1u<<(shift+13))-1u);
        if(rem > (1u<<(shift+12)) || (rem==(1u<<(shift+12)) && (half&1u))) half++;
        return (uint16_t)(sign|half);
    }
    if(exp>=31) return (uint16_t)(sign|0x7C00u);
    uint32_t half=(uint32_t)exp<<10 | (mant>>13);
    const uint32_t rem=mant&0x1FFFu;
    if(rem>0x1000u || (rem==0x1000u && (half&1u))) half++;
    return (uint16_t)(sign|half);
}
static inline float f16_bits_to_f32(uint16_t b){
    const uint32_t sign=(uint32_t)(b&0x8000u)<<16;
    uint32_t exp=(b>>10)&0x1Fu, mant=b&0x3FFu, out;
    if(exp==0){
        if(mant==0) out=sign;
        else { int e=-1; do{ e++; mant<<=1; }while(!(mant&0x400u)); mant&=0x3FFu; out=sign|((uint32_t)(127-15-e)<<23)|(mant<<13); }
    }
    else if(exp==31) out=sign|0x7F800000u|(mant<<13);
    else out=sign|((exp+(127-15))<<23)|(mant<<13);
    float f; std::memcpy(&f,&out,4); return f;
}

static void pack_matrix(std::vector<uint8_t>& out, int bits, int M, int K, uint64_t seed){
    const int gpr=K/256, gb=v2_group_bytes(bits), nlev=1<<bits;
    out.assign((size_t)M*gpr*gb,0);
    uint64_t rng=seed;
    for(int r=0;r<M;r++) for(int g=0;g<gpr;g++){
        uint8_t* gp=out.data()+((size_t)r*gpr+g)*gb;
        const uint16_t h[4]={ f32_to_f16_bits(lcg_finite(&rng,0.01f,0.03f)), f32_to_f16_bits(lcg_finite(&rng,-0.4f,0.4f)),
                              f32_to_f16_bits(lcg_finite(&rng,0.01f,0.03f)), f32_to_f16_bits(lcg_finite(&rng,-0.4f,0.4f)) };
        for(int i=0;i<4;i++){ gp[2*i]=(uint8_t)(h[i]&0xFF); gp[2*i+1]=(uint8_t)(h[i]>>8); }
        uint8_t* payload=gp+8;
        for(int c=0;c<256;c++){
            const unsigned q=(unsigned)(lcg_u64(&rng)%(unsigned)nlev);
            const int bit_pos=c*bits, byte_i=bit_pos>>3, bib=bit_pos&7;
            payload[byte_i] |= (uint8_t)((q<<bib)&0xFF);
            if(bib+bits>8) payload[byte_i+1] |= (uint8_t)((q>>(8-bib))&0xFF);
        }
    }
}

// Sampled host double reference (fp16-fma dequant model). overwrite: Y = W@X;
// residual: Y_ref = Y_init + W@X (pass Yinit non-null).
static double sampled_ref_rel_l2(const std::vector<uint8_t>& A, const std::vector<uint16_t>& Xh,
                                 const std::vector<float>& Ydev, const float* Yinit,
                                 int M, int K, int N, int bits){
    const int gpr=K/256, gb=v2_group_bytes(bits);
    const int ncols=std::min(N,16), nrows=std::min(M,24);
    double num=0, den=0;
    for(int ri=0;ri<nrows;ri++){
        const int r=(int)(((long long)ri*(M-1))/std::max(1,nrows-1));
        for(int n=0;n<ncols;n++){
            double acc = Yinit ? (double)Yinit[(size_t)n*M+r] : 0.0;
            for(int g=0;g<gpr;g++){
                const uint8_t* gp=A.data()+((size_t)r*gpr+g)*gb;
                const float sc0=f16_bits_to_f32((uint16_t)(gp[0]|(gp[1]<<8))), zp0=f16_bits_to_f32((uint16_t)(gp[2]|(gp[3]<<8)));
                const float sc1=f16_bits_to_f32((uint16_t)(gp[4]|(gp[5]<<8))), zp1=f16_bits_to_f32((uint16_t)(gp[6]|(gp[7]<<8)));
                const uint8_t* payload=gp+8;
                for(int c=0;c<256;c++){
                    const int bit_pos=c*bits, byte_i=bit_pos>>3, bib=bit_pos&7;
                    unsigned v=(unsigned)payload[byte_i];
                    if(bib+bits>8) v|=(unsigned)payload[byte_i+1]<<8;
                    const unsigned q=(v>>bib)&((1u<<bits)-1u);
                    const float sc=(c<128)?sc0:sc1, zp=(c<128)?zp0:zp1;
                    /* Mirror device codegen: dequant contracts to v_fma_f16 —
                       exact sc*q+zp (product exact in f32: 11-bit sc × 6-bit q),
                       ONE rounding to f16. */
                    const float w=f16_bits_to_f32(f32_to_f16_bits(sc*(float)q + zp));
                    acc += (double)w*(double)f16_bits_to_f32(Xh[(size_t)n*K+g*256+c]);
                }
            }
            const double d=(double)Ydev[(size_t)n*M+r]-acc;
            num+=d*d; den+=acc*acc;
        }
    }
    return den==0 ? std::sqrt(num) : std::sqrt(num/den);
}

static bool biteq_vec(const std::vector<float>& a, const std::vector<float>& b){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();i++){ uint32_t x,y; std::memcpy(&x,&a[i],4); std::memcpy(&y,&b[i],4); if(x!=y) return false; }
    return true;
}

static int run_gateup(const std::string& arch, int bits, int N, int samples, int warmup,
                      int gate_m, int up_m, int K){
    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1; hipDeviceProp_t prop{};
    for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        if(std::strncmp(p.gcnArchName,arch.c_str(),arch.size())==0){ dev=d; prop=p; break; } }
    if(dev<0){ std::fprintf(stderr,"no device with gcnArchName %s*:\n",arch.c_str());
        for(int d=0;d<ndev;d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d)); std::fprintf(stderr,"  device %d: %s (%s)\n",d,p.name,p.gcnArchName); }
        return 2; }
    CK(hipSetDevice(dev));

    const int total_m=gate_m+up_m, gpr=K/256, gb=v2_group_bytes(bits);
    const size_t a_bytes=(size_t)total_m*gpr*gb;
    std::printf("%s op=gateup\narch=%s dev=%d (%s)\nbits=%d N=%d gate/up_m=%d/%d K=%d  A=%zu bytes  samples=%d warmup=%d slabs=%d\n",
                kBenchId, arch.c_str(), dev, prop.name, bits, N, gate_m, up_m, K, a_bytes, samples, warmup, kBatchLaunches);

    std::vector<uint8_t> Ag,Au; std::vector<uint16_t> Xh((size_t)N*K);
    pack_matrix(Ag,bits,gate_m,K,42); pack_matrix(Au,bits,up_m,K,1042);
    { uint64_t rng=7; for(auto& x:Xh) x=f32_to_f16_bits(lcg_finite(&rng,-1.f,1.f)); }

    struct Slab{ void *g,*u; };
    std::vector<Slab> slabs(kBatchLaunches);
    for(auto& s:slabs){
        CK(hipMalloc(&s.g,Ag.size())); CK(hipMemcpy(s.g,Ag.data(),Ag.size(),hipMemcpyHostToDevice));
        CK(hipMalloc(&s.u,Au.size())); CK(hipMemcpy(s.u,Au.data(),Au.size(),hipMemcpyHostToDevice));
    }
    void* dX; CK(hipMalloc(&dX,Xh.size()*2)); CK(hipMemcpy(dX,Xh.data(),Xh.size()*2,hipMemcpyHostToDevice));
    void *dYg,*dYu;
    CK(hipMalloc(&dYg,(size_t)N*gate_m*4)); CK(hipMalloc(&dYu,(size_t)N*up_m*4));

    using KFn = void(*)(const char*,const char*,const _Float16*,float*,float*,int,int,int,int);
    KFn bt  = (bits==3)?mb_gate_up_mq3_bt12:(bits==5)?mb_gate_up_mq5_bt12:mb_gate_up_mq6_bt12;
    KFn mw4 = (bits==3)?mb_gate_up_mq3_mw4 :(bits==5)?mb_gate_up_mq5_mw4 :mb_gate_up_mq6_mw4;
    KFn mw8 = (bits==3)?mb_gate_up_mq3_mw8 :(bits==5)?mb_gate_up_mq5_mw8 :mb_gate_up_mq6_mw8;
    struct Arm{ const char* name; KFn fn; int nw; int bv; };
    const Arm arms[3]={ {"BT12",bt,0,12}, {"MW4",mw4,4,0}, {"MW8",mw8,8,0} };

    auto launch=[&](const Arm& a, const Slab& s){
        if(a.nw==0){
            dim3 grid((total_m+15)/16,(N+16*a.bv-1)/(16*a.bv)), block(32);
            hipLaunchKernelGGL(a.fn,grid,block,0,0,(const char*)s.g,(const char*)s.u,(const _Float16*)dX,(float*)dYg,(float*)dYu,gate_m,up_m,K,N);
        } else {
            dim3 grid((total_m+15)/16,(N+16*a.nw-1)/(16*a.nw)), block(32*a.nw);
            hipLaunchKernelGGL(a.fn,grid,block,0,0,(const char*)s.g,(const char*)s.u,(const _Float16*)dX,(float*)dYg,(float*)dYu,gate_m,up_m,K,N);
        }
    };
    auto fetch=[&](std::vector<float>& yg,std::vector<float>& yu){
        yg.resize((size_t)N*gate_m); yu.resize((size_t)N*up_m);
        CK(hipMemcpy(yg.data(),dYg,yg.size()*4,hipMemcpyDeviceToHost));
        CK(hipMemcpy(yu.data(),dYu,yu.size()*4,hipMemcpyDeviceToHost));
    };

    bool pass=true;
    std::vector<float> bg,bu;
    CK(hipMemset(dYg,0,(size_t)N*gate_m*4)); CK(hipMemset(dYu,0,(size_t)N*up_m*4));
    launch(arms[0],slabs[0]); CK(hipDeviceSynchronize()); fetch(bg,bu);
    const double rg=sampled_ref_rel_l2(Ag,Xh,bg,nullptr,gate_m,K,N,bits);
    const double ru=sampled_ref_rel_l2(Au,Xh,bu,nullptr,up_m,K,N,bits);
    const bool refok = rg<=1e-4 && ru<=1e-4;
    std::printf("HOSTREF BT12 rel_l2 gate/up = %.2e/%.2e : %s\n", rg,ru, refok?"PASS":"FAIL");
    pass &= refok;
    for(int ai=1; ai<3; ai++){
        std::vector<float> yg,yu;
        CK(hipMemset(dYg,0,(size_t)N*gate_m*4)); CK(hipMemset(dYu,0,(size_t)N*up_m*4));
        launch(arms[ai],slabs[0]); CK(hipDeviceSynchronize()); fetch(yg,yu);
        const bool ok = biteq_vec(yg,bg)&&biteq_vec(yu,bu);
        std::printf("BITEQ %s vs BT12 : %s\n", arms[ai].name, ok?"PASS":"FAIL");
        pass &= ok;
    }

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    double med_us[3]={0,0,0};
    for(int ai=0; ai<3; ai++){
        for(int w=0;w<warmup;w++) launch(arms[ai],slabs[w%kBatchLaunches]);
        CK(hipDeviceSynchronize());
        std::vector<float> us; us.reserve(samples);
        for(int s=0;s<samples;s++){
            CK(hipEventRecord(ev0,0));
            for(int r=0;r<kBatchLaunches;r++) launch(arms[ai],slabs[r]);
            CK(hipEventRecord(ev1,0));
            CK(hipEventSynchronize(ev1));
            float ms=0; CK(hipEventElapsedTime(&ms,ev0,ev1));
            us.push_back(ms*1000.f/(float)kBatchLaunches);
        }
        std::sort(us.begin(),us.end());
        med_us[ai]=us[us.size()/2];
        const double gibs=(double)a_bytes/(med_us[ai]*1e-6)/(1024.0*1024.0*1024.0);
        std::printf("%-4s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s=%7.2f  vsBT12=%5.2fx\n",
                    arms[ai].name, med_us[ai], us[(size_t)(us.size()*0.10)], us[(size_t)(us.size()*0.90)],
                    gibs, med_us[0]/med_us[ai]);
    }
    std::printf("%s\n", pass?"OVERALL PASS":"OVERALL FAIL");
    return pass?0:1;
}

static int run_resid(const std::string& arch, int bits, int N, int samples, int warmup,
                     int M, int K){
    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1; hipDeviceProp_t prop{};
    for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        if(std::strncmp(p.gcnArchName,arch.c_str(),arch.size())==0){ dev=d; prop=p; break; } }
    if(dev<0){ std::fprintf(stderr,"no device with gcnArchName %s*:\n",arch.c_str());
        for(int d=0;d<ndev;d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d)); std::fprintf(stderr,"  device %d: %s (%s)\n",d,p.name,p.gcnArchName); }
        return 2; }
    CK(hipSetDevice(dev));

    const int gpr=K/256, gb=v2_group_bytes(bits);
    const size_t a_bytes=(size_t)M*gpr*gb;
    std::printf("%s op=resid\narch=%s dev=%d (%s)\nbits=%d N=%d M=%d K=%d  A=%zu bytes  samples=%d warmup=%d slabs=%d\n",
                kBenchId, arch.c_str(), dev, prop.name, bits, N, M, K, a_bytes, samples, warmup, kBatchLaunches);

    std::vector<uint8_t> A; std::vector<uint16_t> Xh((size_t)N*K);
    std::vector<float> Yh((size_t)N*M);
    pack_matrix(A,bits,M,K,42);
    { uint64_t rng=7; for(auto& x:Xh) x=f32_to_f16_bits(lcg_finite(&rng,-1.f,1.f)); }
    { uint64_t rng=99; for(auto& y:Yh) y=lcg_finite(&rng,-0.5f,0.5f); }

    struct Slab{ void *a; };
    std::vector<Slab> slabs(kBatchLaunches);
    for(auto& s:slabs){
        CK(hipMalloc(&s.a,A.size())); CK(hipMemcpy(s.a,A.data(),A.size(),hipMemcpyHostToDevice));
    }
    void* dX; CK(hipMalloc(&dX,Xh.size()*2)); CK(hipMemcpy(dX,Xh.data(),Xh.size()*2,hipMemcpyHostToDevice));
    void* dY; CK(hipMalloc(&dY,(size_t)N*M*4));

    using KFn = void(*)(const char*,const _Float16*,float*,int,int,int);
    KFn bt  = (bits==3)?mb_resid_mq3_bt4:(bits==5)?mb_resid_mq5_bt4:mb_resid_mq6_bt4;
    KFn mw4 = (bits==3)?mb_resid_mq3_mw4:(bits==5)?mb_resid_mq5_mw4:mb_resid_mq6_mw4;
    KFn mw8 = (bits==3)?mb_resid_mq3_mw8:(bits==5)?mb_resid_mq5_mw8:mb_resid_mq6_mw8;
    struct Arm{ const char* name; KFn fn; int nw; int bv; };
    const Arm arms[3]={ {"BT4",bt,0,4}, {"MW4",mw4,4,0}, {"MW8",mw8,8,0} };

    auto restore_y=[&]{ CK(hipMemcpy(dY,Yh.data(),Yh.size()*4,hipMemcpyHostToDevice)); };
    auto launch=[&](const Arm& a, const Slab& s){
        if(a.nw==0){
            dim3 grid((M+15)/16,(N+16*a.bv-1)/(16*a.bv)), block(32);
            hipLaunchKernelGGL(a.fn,grid,block,0,0,(const char*)s.a,(const _Float16*)dX,(float*)dY,M,K,N);
        } else {
            dim3 grid((M+15)/16,(N+16*a.nw-1)/(16*a.nw)), block(32*a.nw);
            hipLaunchKernelGGL(a.fn,grid,block,0,0,(const char*)s.a,(const _Float16*)dX,(float*)dY,M,K,N);
        }
    };
    auto fetch=[&](std::vector<float>& y){
        y.resize((size_t)N*M);
        CK(hipMemcpy(y.data(),dY,y.size()*4,hipMemcpyDeviceToHost));
    };

    bool pass=true;
    std::vector<float> by;
    restore_y();
    launch(arms[0],slabs[0]); CK(hipDeviceSynchronize()); fetch(by);
    const double rr=sampled_ref_rel_l2(A,Xh,by,Yh.data(),M,K,N,bits);
    const bool refok = rr<=1e-4;
    std::printf("HOSTREF BT4 rel_l2 = %.2e : %s\n", rr, refok?"PASS":"FAIL");
    pass &= refok;
    for(int ai=1; ai<3; ai++){
        std::vector<float> y;
        restore_y();
        launch(arms[ai],slabs[0]); CK(hipDeviceSynchronize()); fetch(y);
        const bool ok = biteq_vec(y,by);
        std::printf("BITEQ %s vs BT4 : %s\n", arms[ai].name, ok?"PASS":"FAIL");
        pass &= ok;
    }

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    double med_us[3]={0,0,0};
    for(int ai=0; ai<3; ai++){
        // Warmup: restore Y each launch so += doesn't accumulate runaway values
        // (timing path only needs consistent launch cost; correctness already done).
        for(int w=0;w<warmup;w++){ restore_y(); launch(arms[ai],slabs[w%kBatchLaunches]); }
        CK(hipDeviceSynchronize());
        std::vector<float> us; us.reserve(samples);
        for(int s=0;s<samples;s++){
            // One restore before the 8-slab timed batch (production residual also
            // starts from a live Y; we keep Y live across the 8 rotates so the
            // timed path matches real += traffic without H2D inside the event).
            restore_y();
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(ev0,0));
            for(int r=0;r<kBatchLaunches;r++) launch(arms[ai],slabs[r]);
            CK(hipEventRecord(ev1,0));
            CK(hipEventSynchronize(ev1));
            float ms=0; CK(hipEventElapsedTime(&ms,ev0,ev1));
            us.push_back(ms*1000.f/(float)kBatchLaunches);
        }
        std::sort(us.begin(),us.end());
        med_us[ai]=us[us.size()/2];
        const double gibs=(double)a_bytes/(med_us[ai]*1e-6)/(1024.0*1024.0*1024.0);
        std::printf("%-4s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s=%7.2f  vsBT4=%5.2fx\n",
                    arms[ai].name, med_us[ai], us[(size_t)(us.size()*0.10)], us[(size_t)(us.size()*0.90)],
                    gibs, med_us[0]/med_us[ai]);
    }
    std::printf("%s\n", pass?"OVERALL PASS":"OVERALL FAIL");
    return pass?0:1;
}

int main(int argc,char** argv){
    std::string arch, op="gateup";
    int bits=3, N=-1, samples=kDefSamples, warmup=kDefWarmup;
    int gate_m=12288, up_m=12288, K_gate=5120, M_res=5120, K_res=12288;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        auto next=[&]{ if(i+1>=argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} return std::atoi(argv[++i]); };
        if(a=="--arch"&&i+1<argc) arch=argv[++i];
        else if(a=="--op"&&i+1<argc) op=argv[++i];
        else if(a=="--bits") bits=next();
        else if(a=="--n") N=next();
        else if(a=="--gate_m") gate_m=next();
        else if(a=="--up_m") up_m=next();
        else if(a=="--M") M_res=next();
        else if(a=="--K") { K_gate=next(); K_res=K_gate; }
        else if(a=="--samples") samples=next();
        else if(a=="--warmup") warmup=next();
        else {
            std::printf("Usage: %s --arch gfxNNNN --op gateup|resid [--bits 3|5|6] [--n N] [--samples S] [--warmup W]\nIdentity: %s\n",
                        argv[0], kBenchId);
            return a=="--help"||a=="-h" ? 0 : 2;
        }
    }
    if(arch.empty()){ std::fprintf(stderr,"--arch required\n"); return 2; }
    if(arch!="gfx1100"&&arch!="gfx1151"){ std::printf("SKIP: %s is gfx11-only (requested %s)\n", kBenchId, arch.c_str()); return 0; }
    if(bits!=3&&bits!=5&&bits!=6){ std::fprintf(stderr,"--bits must be 3, 5 or 6\n"); return 2; }
    if(op!="gateup"&&op!="resid"){ std::fprintf(stderr,"--op must be gateup or resid\n"); return 2; }

    if(op=="gateup"){
        if(N<0) N=512;
        if(K_gate%256){ std::fprintf(stderr,"K%%256!=0\n"); return 2; }
        return run_gateup(arch,bits,N,samples,warmup,gate_m,up_m,K_gate);
    } else {
        if(N<0) N=512;
        if(K_res%256){ std::fprintf(stderr,"K%%256!=0\n"); return 2; }
        return run_resid(arch,bits,N,samples,warmup,M_res,K_res);
    }
}
