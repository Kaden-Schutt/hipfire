// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mb-qkv-mw-vs-bt-gfx1100-v1-20260829
//
// QUESTION: does MW-LDS same-row cooperative staging beat the one-wave BT4
// batch-tile shape for the QKV projection on gfx1100 (GDDR6, high BW, deep
// CU count; BT12 is production policy and MW gate_up/residual ARE production
// here — LDS-pad18 answer applies to shipping kernels)?
//
// Production construct interrogated (arm 0, verbatim):
//   gemm_qkv_mq{3,5,6}g256v2_wmma_gfx11_bt4   (kernels/src/gemm_mqv2_wmma_gfx11_bt.hip)
// Candidate arms (structure authority: gemm_mqv2_wmma_gfx11_mw_lds.hip gate_up family):
//   MW4 / MW8 same-row LDS staging, unpadded [16][16][16]  (LDS padding is
//   mb_lds_pad18.hip's question, not this file's).
//   --bt12 swaps baseline to BT12 (BV=12, verbatim BT body) for graduation decision
//   vs production QKV BT12; BITEQ still holds (same accumulation order).
//
// Correctness: MW arms f32 to_bits-equal to BT4 (same g-ascending,
// kt-ascending accumulation per output element); BT4 vs sampled host double
// reference rel_l2 <= 1e-4 (sanity gate for wire-format bugs; the ref models
// the device's f16 fma dequant with an f32 pre-sum, whose residual
// double-rounding on ties bounds observed rel_l2 at ~2e-5 — bit-parity is
// BITEQ's job, not this gate's).
//
// Build (single-arch is fine on ROCm 7.15 for gfx1100):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 \
//       tools/kernels/gfx1100/mb_qkv_mw_vs_bt.hip -o mb_qkv_mw_vs_bt
// Run:
//   ./mb_qkv_mw_vs_bt --arch gfx1100 [--bits 3|5|6] [--n N] [--pad18] [--bt12] [--samples S] [--warmup W]
//   Defaults: bits=3 n=512 q_m=4096 k_m=1024 v_m=1024 K=5120 samples=50 warmup=20

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

static constexpr const char* kBenchId = "mb-qkv-mw-vs-bt-gfx1100-v1-20260829";
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

// Arm 0 — verbatim GEN_QKV_BT (BV=4) from kernels/src/gemm_mqv2_wmma_gfx11_bt.hip.
#define GEN_QKV_BT4(BITS) \
extern "C" __launch_bounds__(32, 2) __global__ void \
mb_qkv_mq##BITS##_bt4( \
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, int K, int N \
) { \
    constexpr int BV = 4; \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = q_m + k_m + v_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * BV); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; int local_row; \
    if (safe_row < q_m) { A = A_q; local_row = safe_row; } \
    else if (safe_row < q_m + k_m) { A = A_k; local_row = safe_row - q_m; } \
    else { A = A_v; local_row = safe_row - q_m - k_m; } \
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
        if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { int out_row = row_start + 2 * j + (tid >> 4); if (out_row < total_m) { float* Y; int proj_row; int out_stride; if (out_row < q_m) { Y = Y_q; proj_row = out_row; out_stride = q_m; } else if (out_row < q_m + k_m) { Y = Y_k; proj_row = out_row - q_m; out_stride = k_m; } else { Y = Y_v; proj_row = out_row - q_m - k_m; out_stride = v_m; } Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; } } } \
    } \
}

// Arm 0 BT12 — verbatim GEN_QKV_BT (BV=12) for graduation decision vs production QKV BT12.
#define GEN_QKV_BT12(BITS) \
extern "C" __launch_bounds__(32, 2) __global__ void \
mb_qkv_mq##BITS##_bt12( \
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, int K, int N \
) { \
    constexpr int BV = 12; \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int total_m = q_m + k_m + v_m; \
    const int row_start = blockIdx.x * 16; \
    const int batch_start = blockIdx.y * (16 * BV); \
    const int tid = threadIdx.x; \
    if (row_start >= total_m) return; \
    if (batch_start >= N) return; \
    const int ml = tid & 15; \
    const int my_row = row_start + ml; \
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1); \
    const char* A; int local_row; \
    if (safe_row < q_m) { A = A_q; local_row = safe_row; } \
    else if (safe_row < q_m + k_m) { A = A_k; local_row = safe_row - q_m; } \
    else { A = A_v; local_row = safe_row - q_m - k_m; } \
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
        if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { int out_row = row_start + 2 * j + (tid >> 4); if (out_row < total_m) { float* Y; int proj_row; int out_stride; if (out_row < q_m) { Y = Y_q; proj_row = out_row; out_stride = q_m; } else if (out_row < q_m + k_m) { Y = Y_k; proj_row = out_row - q_m; out_stride = k_m; } else { Y = Y_v; proj_row = out_row - q_m - k_m; out_stride = v_m; } Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; } } } \
    } \
}

// Candidate — MW-LDS same-row staging (structure authority: gate_up MW family).
// All 32*NW threads reach both barriers every group; tails clamp + discard.
// PAD: innermost LDS dim in halfs. 16 = production layout (32 B row stride,
// 4-way bank conflict on compute reads); 18 = conflict-free (gcd(9,32)=1),
// rows only 4-byte-aligned — validated correct+fast by mb_lds_pad18.
#define GEN_QKV_MW(BITS, NW, PAD) \
extern "C" __launch_bounds__(32 * (NW), 1) __global__ void \
mb_qkv_mq##BITS##_mw##NW##_p##PAD( \
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, int K, int N) { \
    constexpr int GROUP_BYTES = 8 + 32 * (BITS); \
    const int tid = threadIdx.x; \
    const int lane = tid & 31; \
    const int wave_id = tid >> 5; \
    const int ml = lane & 15; \
    const int total_m = q_m + k_m + v_m; \
    const int row_start = blockIdx.x * 16; \
    const int n_block_start = blockIdx.y * (16 * (NW)); \
    const int out_col = n_block_start + wave_id * 16 + ml; \
    const int safe_batch = (out_col < N) ? out_col : 0; \
    const int groups_per_row = K / 256; \
    __shared__ __align__(32) _Float16 w_lds[16][16][PAD]; \
    float8_t acc = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}; \
    for (int g = 0; g < groups_per_row; g++) { \
        for (int job = tid; job < 128; job += 32 * (NW)) { \
            const int row = job / 8; \
            const int segment = job % 8; \
            const int safe_row = (row_start + row < total_m) ? (row_start + row) : (total_m - 1); \
            const char* A; int local_row; \
            if (safe_row < q_m) { A = A_q; local_row = safe_row; } \
            else if (safe_row < q_m + k_m) { A = A_k; local_row = safe_row - q_m; } \
            else { A = A_v; local_row = safe_row - q_m - k_m; } \
            const char* gp = A + (long long)local_row * groups_per_row * GROUP_BYTES + (long long)g * GROUP_BYTES; \
            const unsigned int h = *(const unsigned int*)(gp + ((segment < 4) ? 0 : 4)); \
            const _Float16 sc_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
            const _Float16 zp_h = (_Float16)__half2float(__ushort_as_half((unsigned short)(h >> 16))); \
            const unsigned char* payload = (const unsigned char*)(gp + 8); \
            const int kt0 = 2 * segment; const int kt1 = kt0 + 1; \
            half16_t frag = decode_tile<BITS>(payload, kt0, sc_h, zp_h); *(half16_t*)(&w_lds[kt0][row][0]) = frag; \
            frag = decode_tile<BITS>(payload, kt1, sc_h, zp_h); *(half16_t*)(&w_lds[kt1][row][0]) = frag; \
        } \
        __syncthreads(); \
        const _Float16* xg = X + (long long)safe_batch * K + (long long)g * 256; \
        _Pragma("unroll") for (int kt = 0; kt < 16; kt += 2) { \
            half16_t a_reg = *(const half16_t*)(&w_lds[kt][ml][0]); half16_t b_reg = *(const half16_t*)(xg + kt * 16); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc); \
            a_reg = *(const half16_t*)(&w_lds[kt + 1][ml][0]); b_reg = *(const half16_t*)(xg + (kt + 1) * 16); acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc); \
        } \
        __syncthreads(); \
    } \
    if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { const int out_row = row_start + 2 * j + (lane >> 4); if (out_row < total_m) { if (out_row < q_m) { Y_q[(long long)out_col * q_m + out_row] = acc[j]; } else if (out_row < q_m + k_m) { Y_k[(long long)out_col * k_m + (out_row - q_m)] = acc[j]; } else { Y_v[(long long)out_col * v_m + (out_row - q_m - k_m)] = acc[j]; } } } } \
}

GEN_QKV_BT4(3)
GEN_QKV_BT4(5)
GEN_QKV_BT4(6)
GEN_QKV_BT12(3)
GEN_QKV_BT12(5)
GEN_QKV_BT12(6)
GEN_QKV_MW(3, 4, 16)
GEN_QKV_MW(3, 8, 16)
GEN_QKV_MW(5, 4, 16)
GEN_QKV_MW(5, 8, 16)
GEN_QKV_MW(6, 4, 16)
GEN_QKV_MW(6, 8, 16)
GEN_QKV_MW(3, 4, 18)
GEN_QKV_MW(3, 8, 18)
GEN_QKV_MW(5, 4, 18)
GEN_QKV_MW(5, 8, 18)
GEN_QKV_MW(6, 4, 18)
GEN_QKV_MW(6, 8, 18)

#else // host pass / other arches — matching stubs for launch machinery
#define GEN_STUB(NAME) extern "C" __global__ void NAME(const char*,const char*,const char*,const _Float16*,float*,float*,float*,int,int,int,int,int){}
GEN_STUB(mb_qkv_mq3_bt4) GEN_STUB(mb_qkv_mq5_bt4) GEN_STUB(mb_qkv_mq6_bt4)
GEN_STUB(mb_qkv_mq3_bt12) GEN_STUB(mb_qkv_mq5_bt12) GEN_STUB(mb_qkv_mq6_bt12)
GEN_STUB(mb_qkv_mq3_mw4_p16) GEN_STUB(mb_qkv_mq3_mw8_p16)
GEN_STUB(mb_qkv_mq5_mw4_p16) GEN_STUB(mb_qkv_mq5_mw8_p16)
GEN_STUB(mb_qkv_mq6_mw4_p16) GEN_STUB(mb_qkv_mq6_mw8_p16)
GEN_STUB(mb_qkv_mq3_mw4_p18) GEN_STUB(mb_qkv_mq3_mw8_p18)
GEN_STUB(mb_qkv_mq5_mw4_p18) GEN_STUB(mb_qkv_mq5_mw8_p18)
GEN_STUB(mb_qkv_mq6_mw4_p18) GEN_STUB(mb_qkv_mq6_mw8_p18)
#undef GEN_STUB
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

// Sampled host double reference for one region: rows stride-sampled, cols 0..min(N,16).
static double sampled_ref_rel_l2(const std::vector<uint8_t>& A, const std::vector<uint16_t>& Xh,
                                 const std::vector<float>& Ydev, int M, int K, int N, int bits){
    const int gpr=K/256, gb=v2_group_bytes(bits);
    const int ncols=std::min(N,16), nrows=std::min(M,24);
    double num=0, den=0;
    for(int ri=0;ri<nrows;ri++){
        const int r=(int)(((long long)ri*(M-1))/std::max(1,nrows-1));
        for(int n=0;n<ncols;n++){
            double acc=0;
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

int main(int argc,char** argv){
    std::string arch; int bits=3, N=512, q_m=4096, k_m=1024, v_m=1024, K=5120;
    int samples=kDefSamples, warmup=kDefWarmup, pad18=0, use_bt12=0;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        auto next=[&]{ if(i+1>=argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} return std::atoi(argv[++i]); };
        if(a=="--arch"&&i+1<argc) arch=argv[++i];
        else if(a=="--bits") bits=next();
        else if(a=="--n") N=next();
        else if(a=="--q_m") q_m=next();
        else if(a=="--k_m") k_m=next();
        else if(a=="--v_m") v_m=next();
        else if(a=="--K") K=next();
        else if(a=="--samples") samples=next();
        else if(a=="--warmup") warmup=next();
        else if(a=="--pad18") pad18=1;
        else if(a=="--bt12") use_bt12=1;
        else { std::printf("Usage: %s --arch gfxNNNN [--bits 3|5|6] [--n N] [--pad18] [--bt12] [--q_m/--k_m/--v_m/--K N] [--samples S] [--warmup W]\nIdentity: %s\n", argv[0], kBenchId); return a=="--help"||a=="-h" ? 0 : 2; }
    }
    if(arch.empty()){ std::fprintf(stderr,"--arch required\n"); return 2; }
    if(arch!="gfx1100"&&arch!="gfx1151"){ std::printf("SKIP: %s is gfx11-only (requested %s)\n", kBenchId, arch.c_str()); return 0; }
    if(bits!=3&&bits!=5&&bits!=6){ std::fprintf(stderr,"--bits must be 3, 5 or 6\n"); return 2; }
    if(K%256){ std::fprintf(stderr,"K%%256!=0\n"); return 2; }

    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1; hipDeviceProp_t prop{};
    for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        if(std::strncmp(p.gcnArchName,arch.c_str(),arch.size())==0){ dev=d; prop=p; break; } }
    if(dev<0){ std::fprintf(stderr,"no device with gcnArchName %s*:\n",arch.c_str());
        for(int d=0;d<ndev;d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d)); std::fprintf(stderr,"  device %d: %s (%s)\n",d,p.name,p.gcnArchName); }
        return 2; }
    CK(hipSetDevice(dev));

    const int total_m=q_m+k_m+v_m, gpr=K/256, gb=v2_group_bytes(bits);
    const size_t a_bytes=(size_t)total_m*gpr*gb;
    std::printf("%s\narch=%s dev=%d (%s)\nbits=%d N=%d q/k/v_m=%d/%d/%d K=%d  A=%zu bytes  samples=%d warmup=%d slabs=%d\n",
                kBenchId, arch.c_str(), dev, prop.name, bits, N, q_m, k_m, v_m, K, a_bytes, samples, warmup, kBatchLaunches);

    // Fixtures (one packed copy; replicated into kBatchLaunches slabs on device).
    std::vector<uint8_t> Aq,Ak,Av; std::vector<uint16_t> Xh((size_t)N*K);
    pack_matrix(Aq,bits,q_m,K,42); pack_matrix(Ak,bits,k_m,K,1042); pack_matrix(Av,bits,v_m,K,2042);
    { uint64_t rng=7; for(auto& x:Xh) x=f32_to_f16_bits(lcg_finite(&rng,-1.f,1.f)); }

    struct Slab{ void *q,*k,*v; };
    std::vector<Slab> slabs(kBatchLaunches);
    for(auto& s:slabs){
        CK(hipMalloc(&s.q,Aq.size())); CK(hipMemcpy(s.q,Aq.data(),Aq.size(),hipMemcpyHostToDevice));
        CK(hipMalloc(&s.k,Ak.size())); CK(hipMemcpy(s.k,Ak.data(),Ak.size(),hipMemcpyHostToDevice));
        CK(hipMalloc(&s.v,Av.size())); CK(hipMemcpy(s.v,Av.data(),Av.size(),hipMemcpyHostToDevice));
    }
    void* dX; CK(hipMalloc(&dX,Xh.size()*2)); CK(hipMemcpy(dX,Xh.data(),Xh.size()*2,hipMemcpyHostToDevice));
    void *dYq,*dYk,*dYv;
    CK(hipMalloc(&dYq,(size_t)N*q_m*4)); CK(hipMalloc(&dYk,(size_t)N*k_m*4)); CK(hipMalloc(&dYv,(size_t)N*v_m*4));

    using KFn = void(*)(const char*,const char*,const char*,const _Float16*,float*,float*,float*,int,int,int,int,int);
    KFn bt4  = (bits==3)?mb_qkv_mq3_bt4:(bits==5)?mb_qkv_mq5_bt4:mb_qkv_mq6_bt4;
    KFn bt12k = (bits==3)?mb_qkv_mq3_bt12:(bits==5)?mb_qkv_mq5_bt12:mb_qkv_mq6_bt12;
    KFn mw4, mw8;
    if(pad18){
        mw4 = (bits==3)?mb_qkv_mq3_mw4_p18:(bits==5)?mb_qkv_mq5_mw4_p18:mb_qkv_mq6_mw4_p18;
        mw8 = (bits==3)?mb_qkv_mq3_mw8_p18:(bits==5)?mb_qkv_mq5_mw8_p18:mb_qkv_mq6_mw8_p18;
    } else {
        mw4 = (bits==3)?mb_qkv_mq3_mw4_p16:(bits==5)?mb_qkv_mq5_mw4_p16:mb_qkv_mq6_mw4_p16;
        mw8 = (bits==3)?mb_qkv_mq3_mw8_p16:(bits==5)?mb_qkv_mq5_mw8_p16:mb_qkv_mq6_mw8_p16;
    }
    KFn bt = use_bt12 ? bt12k : bt4;
    const char* bt_name = use_bt12 ? "BT12" : "BT4";
    struct Arm{ const char* name; KFn fn; int nw; };
    const Arm arms[3]={ {bt_name,bt,0}, {pad18?"MW4p":"MW4",mw4,4}, {pad18?"MW8p":"MW8",mw8,8} };

    auto launch=[&](const Arm& a, const Slab& s){
        if(a.nw==0){
            int bv = use_bt12 ? 12 : 4;
            dim3 grid((total_m+15)/16,(N+16*bv-1)/(16*bv)), block(32);
            hipLaunchKernelGGL(a.fn,grid,block,0,0,(const char*)s.q,(const char*)s.k,(const char*)s.v,(const _Float16*)dX,(float*)dYq,(float*)dYk,(float*)dYv,q_m,k_m,v_m,K,N);
        } else {
            dim3 grid((total_m+15)/16,(N+16*a.nw-1)/(16*a.nw)), block(32*a.nw);
            hipLaunchKernelGGL(a.fn,grid,block,0,0,(const char*)s.q,(const char*)s.k,(const char*)s.v,(const _Float16*)dX,(float*)dYq,(float*)dYk,(float*)dYv,q_m,k_m,v_m,K,N);
        }
    };
    auto fetch=[&](std::vector<float>& yq,std::vector<float>& yk,std::vector<float>& yv){
        yq.resize((size_t)N*q_m); yk.resize((size_t)N*k_m); yv.resize((size_t)N*v_m);
        CK(hipMemcpy(yq.data(),dYq,yq.size()*4,hipMemcpyDeviceToHost));
        CK(hipMemcpy(yk.data(),dYk,yk.size()*4,hipMemcpyDeviceToHost));
        CK(hipMemcpy(yv.data(),dYv,yv.size()*4,hipMemcpyDeviceToHost));
    };
    auto biteq=[&](const std::vector<float>& a,const std::vector<float>& b){
        if(a.size()!=b.size()) return false;
        for(size_t i=0;i<a.size();i++){ uint32_t x,y; std::memcpy(&x,&a[i],4); std::memcpy(&y,&b[i],4); if(x!=y) return false; }
        return true;
    };

    // Correctness first.
    bool pass=true;
    std::vector<float> bq,bk,bv;
    launch(arms[0],slabs[0]); CK(hipDeviceSynchronize()); fetch(bq,bk,bv);
    const double rq=sampled_ref_rel_l2(Aq,Xh,bq,q_m,K,N,bits);
    const double rk=sampled_ref_rel_l2(Ak,Xh,bk,k_m,K,N,bits);
    const double rv=sampled_ref_rel_l2(Av,Xh,bv,v_m,K,N,bits);
    const bool refok = rq<=5e-4 && rk<=5e-4 && rv<=5e-4;
    std::printf("HOSTREF %s rel_l2 q/k/v = %.2e/%.2e/%.2e : %s\n", bt_name, rq,rk,rv, refok?"PASS":"FAIL");
    pass &= refok;
    for(int ai=1; ai<3; ai++){
        std::vector<float> yq,yk,yv;
        launch(arms[ai],slabs[0]); CK(hipDeviceSynchronize()); fetch(yq,yk,yv);
        const bool ok = biteq(yq,bq)&&biteq(yk,bk)&&biteq(yv,bv);
        std::printf("BITEQ %s vs %s : %s\n", arms[ai].name, bt_name, ok?"PASS":"FAIL");
        pass &= ok;
    }

    // Timing: per sample, 8 launches rotating slabs inside one event pair.
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
        std::printf("%-4s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s=%7.2f  vs%s=%5.2fx\n",
                    arms[ai].name, med_us[ai], us[(size_t)(us.size()*0.10)], us[(size_t)(us.size()*0.90)],
                    gibs, bt_name, med_us[0]/med_us[ai]);
    }
    std::printf("%s\n", pass?"OVERALL PASS":"OVERALL FAIL");
    return pass?0:1;
}
