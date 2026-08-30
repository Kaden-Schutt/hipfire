// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mb-qkv-fp8-iu8-wmma-gfx1201-v1-20260830
//
// QUESTION: at zero/near-zero decode cost, do RDNA4 FP8 or IU8 WMMA rates beat
// the production f16 decode+WMMA path on the MQ4V2 prefill shape?
// Expectations LOW (prior int-dot GEMV work was canned as no-better-than-fp16);
// this screen exists to put a number on the ceiling.
//
// Production construct interrogated (arm 0, verbatim bits=4):
//   kernels/src/gemm_qkv_mqv2_wmma_gfx1201_bt.hip  decode8<4> + BT8 K4 body
//   (BV=8, K4 quartet, half8 decode, gfx12 f16 WMMA). Authority lineage also
//   tools/kernels/gfx1201/mb_decode8_window.hip A0 body.
//
// Arms (bits=4 ONLY, MQ4V2 wire 136 B/group):
//   A0 baseline — verbatim decode8<4> + f16 WMMA, packed MQ4V2 wire
//   A1 FP8 dense-shadow ceiling — pre-quant E4M3FN W/X planes, fp8 WMMA
//   A2 IU8 affine wire-preserving — nibble codes + Q8 X, iu8 WMMA + f32 fold
//
// Shapes fixed: M=4096 K=5120 N=512 BV=8 K4 samples=50 warmup=20 slabs=8
//
// Build (local gfx1201):
//   /opt/rocm/core/bin/hipcc --offload-arch=gfx1201 -O3 \
//       tools/kernels/gfx1201/mb_qkv_fp8_iu8_wmma.hip -o mb_qkv_fp8_iu8_wmma
// Run:
//   ./mb_qkv_fp8_iu8_wmma --arch gfx1201 [--samples S] [--warmup W]
//
// Correctness gates (per arm, own quantized operands):
//   A0 sampled host-ref rel_l2 <= 1e-5
//   A1 fp8-table ref rel_l2 <= 1e-4
//   A2 int64-idot + f32-fold ref rel_l2 <= 1e-3
// Quality (not a gate): full-buffer rel_l2 of A1/A2 vs A0 f16 output.

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

static constexpr const char* kBenchId = "mb-qkv-fp8-iu8-wmma-gfx1201-v1-20260830";
static constexpr int kDefWarmup = 20;
static constexpr int kDefSamples = 50;
static constexpr int kBatchLaunches = 8;
static constexpr int kBits = 4;
static constexpr int kGroupBytes = 8 + 32 * kBits; // 136
static constexpr int kM = 4096;
static constexpr int kK = 5120;
static constexpr int kN = 512;

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ std::fprintf(stderr,"HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} }while(0)

// ---------------------------------------------------------------------------
// Device code — exact gfx1201 only (host pass and other arches take stub)
// ---------------------------------------------------------------------------
#if defined(__gfx1201__)

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;
typedef int      __attribute__((ext_vector_type(2))) int2_t;
typedef int      __attribute__((ext_vector_type(8))) int8v_t;

// Verbatim production decoder (kernels/src/gemm_qkv_mqv2_wmma_gfx1201_bt.hip)
template <int BITS>
__device__ __forceinline__ half8_t decode8(const unsigned char* __restrict__ codes,
                                           int code0,
                                           _Float16 sc_h,
                                           _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int bit_pos     = (code0 + i) * BITS;
        const int byte_i      = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned v = (unsigned)codes[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= (unsigned)codes[byte_i + 1] << 8;
        const unsigned q = (v >> bit_in_byte) & qmask;
        a[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return a;
}

// A0: production BT8 K4 f16 body, bits=4 fixed
extern "C" __launch_bounds__(32, 2) __global__
void mb_qkv_fp8_iu8_a0(const char* __restrict__ A,
                       const _Float16* __restrict__ X,
                       float* __restrict__ Y,
                       int M, int K, int N) {
    constexpr int GROUP_BYTES = kGroupBytes;
    constexpr int BV = 8;
    const int total_m = M;
    const int rs = (int)blockIdx.x * 16;
    const int bs = (int)blockIdx.y * (16 * BV);
    const int tid = (int)threadIdx.x;
    if (rs >= total_m || bs >= N) return;
    const int ml = tid & 15;
    const int kg = tid >> 4;
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1);
    const int gpr = K / 256;
    const char* rb = A + (long long)sr * (long long)gpr * (long long)GROUP_BYTES;
    int sb[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }
    float8_t acc[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) acc[b] = float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for (int g = 0; g < gpr; ++g) {
        const char* gp = rb + (long long)g * (long long)GROUP_BYTES;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const unsigned char* codes = (const unsigned char*)(gp + 8);
        const _Float16* xg[BV];
#pragma unroll
        for (int b = 0; b < BV; ++b) xg[b] = X + (long long)sb[b] * (long long)K + (long long)g * 256;
        for (int kt = 0; kt < 16; kt += 4) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                const int code0 = (kt + ki) * 16 + kg * 8;
                half8_t a = decode8<4>(codes, code0, sc_h, zp_h);
#pragma unroll
                for (int b = 0; b < BV; ++b) {
                    const half8_t bb = *(const half8_t*)(xg[b] + (kt + ki) * 16 + kg * 8);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]);
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int orow = rs + 8 * kg + j;
                if (orow < total_m) {
                    Y[(long long)oc * (long long)total_m + orow] = acc[b][j];
                }
            }
        }
    }
}

// A1: FP8 dense-shadow — pre-built E4M3FN planes, zero decode
extern "C" __launch_bounds__(32, 2) __global__
void mb_qkv_fp8_iu8_a1(const unsigned char* __restrict__ Wf8,
                       const unsigned char* __restrict__ Xf8,
                       float* __restrict__ Y,
                       int M, int K, int N) {
    constexpr int BV = 8;
    const int total_m = M;
    const int rs = (int)blockIdx.x * 16;
    const int bs = (int)blockIdx.y * (16 * BV);
    const int tid = (int)threadIdx.x;
    if (rs >= total_m || bs >= N) return;
    const int ml = tid & 15;
    const int kg = tid >> 4;
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1);
    int sb[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }
    float8_t acc[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) acc[b] = float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    const unsigned char* wr = Wf8 + (long long)sr * (long long)K;
    const unsigned char* xg[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) xg[b] = Xf8 + (long long)sb[b] * (long long)K;
    for (int k0 = 0; k0 < K; k0 += 256) {
        for (int kt = 0; kt < 16; kt += 4) {
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                const int koff = k0 + (kt + ki) * 16 + kg * 8;
                const int2_t a = *(const int2_t*)(wr + koff);
#pragma unroll
                for (int b = 0; b < BV; ++b) {
                    const int2_t bb = *(const int2_t*)(xg[b] + koff);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, bb, acc[b]);
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int orow = rs + 8 * kg + j;
                if (orow < total_m) {
                    Y[(long long)oc * (long long)total_m + orow] = acc[b][j];
                }
            }
        }
    }
}

// Pack 8 LSB-first 4-bit codes starting at code0 (code0%8==0) into v2i bytes.
__device__ __forceinline__ int2_t pack_nibble8_v2i(const unsigned char* __restrict__ codes,
                                                   int code0) {
    const unsigned int pk = *(const unsigned int*)(codes + (code0 >> 1));
    unsigned int lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lo |= ((pk >> (i * 4)) & 0xFu) << (i * 8);
        hi |= ((pk >> ((i + 4) * 4)) & 0xFu) << (i * 8);
    }
    return int2_t{(int)lo, (int)hi};
}

// A2: IU8 affine, wire-preserving — nibble codes + Q8 X, i32 WMMA per half-group
extern "C" __launch_bounds__(32, 2) __global__
void mb_qkv_fp8_iu8_a2(const char* __restrict__ A,
                       const int8_t* __restrict__ Xq8,
                       const float* __restrict__ scW,   // [M * nh], nh=2*gpr
                       const float* __restrict__ zpW,
                       const float* __restrict__ sx,    // [N * nh]
                       const float* __restrict__ sumx,
                       float* __restrict__ Y,
                       int M, int K, int N) {
    constexpr int GROUP_BYTES = kGroupBytes;
    constexpr int BV = 8;
    const int total_m = M;
    const int rs = (int)blockIdx.x * 16;
    const int bs = (int)blockIdx.y * (16 * BV);
    const int tid = (int)threadIdx.x;
    if (rs >= total_m || bs >= N) return;
    const int ml = tid & 15;
    const int kg = tid >> 4;
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1);
    const int gpr = K / 256;
    const int nh = 2 * gpr;
    const char* rb = A + (long long)sr * (long long)gpr * (long long)GROUP_BYTES;
    int sb[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }
    float8_t facc[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) facc[b] = float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    for (int g = 0; g < gpr; ++g) {
        const char* gp = rb + (long long)g * (long long)GROUP_BYTES;
        const unsigned char* codes = (const unsigned char*)(gp + 8);
        const int8_t* xg[BV];
#pragma unroll
        for (int b = 0; b < BV; ++b)
            xg[b] = Xq8 + (long long)sb[b] * (long long)K + (long long)g * 256;

        for (int half = 0; half < 2; ++half) {
            int8v_t iacc[BV];
#pragma unroll
            for (int b = 0; b < BV; ++b)
                iacc[b] = int8v_t{0,0,0,0,0,0,0,0};
            const int kt0 = half * 8;
            for (int kt = kt0; kt < kt0 + 8; kt += 4) {
#pragma unroll
                for (int ki = 0; ki < 4; ++ki) {
                    const int code0 = (kt + ki) * 16 + kg * 8;
                    const int2_t wcodes = pack_nibble8_v2i(codes, code0);
#pragma unroll
                    for (int b = 0; b < BV; ++b) {
                        const int2_t xq = *(const int2_t*)(xg[b] + (kt + ki) * 16 + kg * 8);
                        iacc[b] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                            false, wcodes, true, xq, iacc[b], false);
                    }
                }
            }
            const int h = 2 * g + half;
#pragma unroll
            for (int b = 0; b < BV; ++b) {
                const float sx_h = sx[(long long)sb[b] * nh + h];
                const float sumx_h = sumx[(long long)sb[b] * nh + h];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int orow = rs + 8 * kg + j;
                    const float sc_h = scW[(long long)orow * nh + h];
                    const float zp_h = zpW[(long long)orow * nh + h];
                    facc[b][j] += sc_h * sx_h * (float)iacc[b][j] + zp_h * sumx_h;
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int orow = rs + 8 * kg + j;
                if (orow < total_m) {
                    Y[(long long)oc * (long long)total_m + orow] = facc[b][j];
                }
            }
        }
    }
}

#else // host pass / other arches — matching stubs
extern "C" __global__ void mb_qkv_fp8_iu8_a0(const char*, const _Float16*, float*, int, int, int) {}
extern "C" __global__ void mb_qkv_fp8_iu8_a1(const unsigned char*, const unsigned char*, float*, int, int, int) {}
extern "C" __global__ void mb_qkv_fp8_iu8_a2(const char*, const int8_t*, const float*, const float*,
                                              const float*, const float*, float*, int, int, int) {}
#endif

// ---------------------------------------------------------------------------
// Host helpers
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

static void pack_matrix_mq4(std::vector<uint8_t>& out, int M, int K, uint64_t seed){
    const int gpr=K/256, gb=kGroupBytes, nlev=1<<kBits;
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
            const int bit_pos=c*kBits, byte_i=bit_pos>>3, bib=bit_pos&7;
            payload[byte_i] |= (uint8_t)((q<<bib)&0xFF);
            if(bib+kBits>8) payload[byte_i+1] |= (uint8_t)((q>>(8-bib))&0xFF);
        }
    }
}

// E4M3FN table + encode (exclude NaN bytes 0x7F/0xFF; ties toward even byte)
static float g_e4m3_tbl[256];
static void build_e4m3_table(){
    for(int b=0;b<256;b++){
        const int sign=b>>7, exp=(b>>3)&0xF, mant=b&7;
        float v;
        if(exp==0xF && mant==7){
            v = NAN;
        } else if(exp==0){
            v = std::ldexp((float)mant/8.f, -6);
        } else {
            v = std::ldexp(1.f+(float)mant/8.f, exp-7);
        }
        g_e4m3_tbl[b] = sign ? -v : v;
    }
}
static uint8_t encode_e4m3fn(float x){
    if(!(x==x)) return 0;
    if(x > 448.f) return 0x7E;
    if(x < -448.f) return 0xFE;
    float best_d = 1e30f;
    uint8_t best = 0;
    for(int b=0;b<256;b++){
        if(b==0x7F || b==0xFF) continue;
        const float d = std::fabs(g_e4m3_tbl[b] - x);
        if(d < best_d || (d == best_d && (b & 1) == 0)){
            best_d = d;
            best = (uint8_t)b;
        }
    }
    return best;
}

static void build_fp8_planes(const std::vector<uint8_t>& A, const std::vector<uint16_t>& Xh,
                             std::vector<uint8_t>& Wf8, std::vector<uint8_t>& Xf8,
                             int M, int K, int N){
    const int gpr=K/256, gb=kGroupBytes;
    Wf8.resize((size_t)M*K);
    Xf8.resize((size_t)N*K);
    for(int r=0;r<M;r++){
        for(int g=0;g<gpr;g++){
            const uint8_t* gp=A.data()+((size_t)r*gpr+g)*gb;
            const _Float16 sc0=(_Float16)f16_bits_to_f32((uint16_t)(gp[0]|(gp[1]<<8)));
            const _Float16 zp0=(_Float16)f16_bits_to_f32((uint16_t)(gp[2]|(gp[3]<<8)));
            const _Float16 sc1=(_Float16)f16_bits_to_f32((uint16_t)(gp[4]|(gp[5]<<8)));
            const _Float16 zp1=(_Float16)f16_bits_to_f32((uint16_t)(gp[6]|(gp[7]<<8)));
            const uint8_t* payload=gp+8;
            for(int c=0;c<256;c++){
                const int bit_pos=c*kBits, byte_i=bit_pos>>3, bib=bit_pos&7;
                unsigned v=(unsigned)payload[byte_i];
                if(bib+kBits>8) v|=(unsigned)payload[byte_i+1]<<8;
                const unsigned q=(v>>bib)&((1u<<kBits)-1u);
                const _Float16 sc_h=(c<128)?sc0:sc1;
                const _Float16 zp_h=(c<128)?zp0:zp1;
                const _Float16 w_h = sc_h * (_Float16)(float)q + zp_h;
                Wf8[(size_t)r*K + g*256 + c] = encode_e4m3fn((float)w_h);
            }
        }
    }
    for(int n=0;n<N;n++) for(int k=0;k<K;k++)
        Xf8[(size_t)n*K+k] = encode_e4m3fn(f16_bits_to_f32(Xh[(size_t)n*K+k]));
}

static void build_iu8_aux(const std::vector<uint8_t>& A, const std::vector<uint16_t>& Xh,
                          std::vector<int8_t>& Xq8,
                          std::vector<float>& scW, std::vector<float>& zpW,
                          std::vector<float>& sx, std::vector<float>& sumx,
                          int M, int K, int N){
    const int gpr=K/256, nh=2*gpr, gb=kGroupBytes;
    Xq8.resize((size_t)N*K);
    scW.assign((size_t)M*nh, 0.f);
    zpW.assign((size_t)M*nh, 0.f);
    sx.assign((size_t)N*nh, 0.f);
    sumx.assign((size_t)N*nh, 0.f);

    for(int r=0;r<M;r++){
        for(int g=0;g<gpr;g++){
            const uint8_t* gp=A.data()+((size_t)r*gpr+g)*gb;
            scW[(size_t)r*nh + 2*g+0] = f16_bits_to_f32((uint16_t)(gp[0]|(gp[1]<<8)));
            zpW[(size_t)r*nh + 2*g+0] = f16_bits_to_f32((uint16_t)(gp[2]|(gp[3]<<8)));
            scW[(size_t)r*nh + 2*g+1] = f16_bits_to_f32((uint16_t)(gp[4]|(gp[5]<<8)));
            zpW[(size_t)r*nh + 2*g+1] = f16_bits_to_f32((uint16_t)(gp[6]|(gp[7]<<8)));
        }
    }
    for(int n=0;n<N;n++){
        for(int g=0;g<gpr;g++){
            for(int half=0;half<2;half++){
                const int h=2*g+half;
                const int k0=g*256+half*128;
                float maxabs=0.f, s=0.f;
                for(int c=0;c<128;c++){
                    const float x=f16_bits_to_f32(Xh[(size_t)n*K+k0+c]);
                    maxabs=std::max(maxabs, std::fabs(x));
                    s+=x;
                }
                const float scale = (maxabs>0.f) ? (maxabs/127.f) : 1.f;
                sx[(size_t)n*nh+h]=scale;
                sumx[(size_t)n*nh+h]=s;
                for(int c=0;c<128;c++){
                    const float x=f16_bits_to_f32(Xh[(size_t)n*K+k0+c]);
                    int q=(int)std::lrintf(x/scale);
                    if(q>127) q=127; if(q<-127) q=-127;
                    Xq8[(size_t)n*K+k0+c]=(int8_t)q;
                }
            }
        }
    }
}

static double sampled_ref_a0(const std::vector<uint8_t>& A, const std::vector<uint16_t>& Xh,
                             const std::vector<float>& Ydev, int M, int K, int N){
    const int gpr=K/256, gb=kGroupBytes;
    const int ncols=std::min(N,16), nrows=std::min(M,24);
    double num=0, den=0;
    for(int ri=0;ri<nrows;ri++){
        const int r=(int)(((long long)ri*(M-1))/std::max(1,nrows-1));
        for(int n=0;n<ncols;n++){
            double acc=0;
            for(int g=0;g<gpr;g++){
                const uint8_t* gp=A.data()+((size_t)r*gpr+g)*gb;
                const _Float16 sc0_h=(_Float16)f16_bits_to_f32((uint16_t)(gp[0]|(gp[1]<<8)));
                const _Float16 zp0_h=(_Float16)f16_bits_to_f32((uint16_t)(gp[2]|(gp[3]<<8)));
                const _Float16 sc1_h=(_Float16)f16_bits_to_f32((uint16_t)(gp[4]|(gp[5]<<8)));
                const _Float16 zp1_h=(_Float16)f16_bits_to_f32((uint16_t)(gp[6]|(gp[7]<<8)));
                const uint8_t* payload=gp+8;
                for(int c=0;c<256;c++){
                    const int bit_pos=c*kBits, byte_i=bit_pos>>3, bib=bit_pos&7;
                    unsigned v=(unsigned)payload[byte_i];
                    if(bib+kBits>8) v|=(unsigned)payload[byte_i+1]<<8;
                    const unsigned q=(v>>bib)&((1u<<kBits)-1u);
                    const _Float16 sc_h=(c<128)?sc0_h:sc1_h;
                    const _Float16 zp_h=(c<128)?zp0_h:zp1_h;
                    const _Float16 w_h=sc_h*(_Float16)(float)q+zp_h;
                    acc += (double)(float)w_h * (double)f16_bits_to_f32(Xh[(size_t)n*K+g*256+c]);
                }
            }
            const double d=(double)Ydev[(size_t)n*M+r]-acc;
            num+=d*d; den+=acc*acc;
        }
    }
    return den==0 ? std::sqrt(num) : std::sqrt(num/den);
}

static double sampled_ref_a1(const std::vector<uint8_t>& Wf8, const std::vector<uint8_t>& Xf8,
                             const std::vector<float>& Ydev, int M, int K, int N){
    const int ncols=std::min(N,16), nrows=std::min(M,24);
    double num=0, den=0;
    for(int ri=0;ri<nrows;ri++){
        const int r=(int)(((long long)ri*(M-1))/std::max(1,nrows-1));
        for(int n=0;n<ncols;n++){
            double acc=0;
            for(int k=0;k<K;k++){
                acc += (double)g_e4m3_tbl[Wf8[(size_t)r*K+k]]
                     * (double)g_e4m3_tbl[Xf8[(size_t)n*K+k]];
            }
            const double d=(double)Ydev[(size_t)n*M+r]-acc;
            num+=d*d; den+=acc*acc;
        }
    }
    return den==0 ? std::sqrt(num) : std::sqrt(num/den);
}

static double sampled_ref_a2(const std::vector<uint8_t>& A, const std::vector<int8_t>& Xq8,
                             const std::vector<float>& scW, const std::vector<float>& zpW,
                             const std::vector<float>& sx, const std::vector<float>& sumx,
                             const std::vector<float>& Ydev, int M, int K, int N){
    const int gpr=K/256, nh=2*gpr, gb=kGroupBytes;
    const int ncols=std::min(N,16), nrows=std::min(M,24);
    double num=0, den=0;
    for(int ri=0;ri<nrows;ri++){
        const int r=(int)(((long long)ri*(M-1))/std::max(1,nrows-1));
        for(int n=0;n<ncols;n++){
            double acc=0;
            for(int g=0;g<gpr;g++){
                const uint8_t* gp=A.data()+((size_t)r*gpr+g)*gb;
                const uint8_t* payload=gp+8;
                for(int half=0;half<2;half++){
                    const int h=2*g+half;
                    int64_t idot=0;
                    for(int c=0;c<128;c++){
                        const int code=half*128+c;
                        const int bit_pos=code*kBits, byte_i=bit_pos>>3, bib=bit_pos&7;
                        unsigned v=(unsigned)payload[byte_i];
                        if(bib+kBits>8) v|=(unsigned)payload[byte_i+1]<<8;
                        const unsigned q=(v>>bib)&((1u<<kBits)-1u);
                        const int8_t xq=Xq8[(size_t)n*K+g*256+code];
                        idot += (int64_t)q * (int64_t)xq;
                    }
                    acc += (double)scW[(size_t)r*nh+h] * (double)sx[(size_t)n*nh+h] * (double)idot
                         + (double)zpW[(size_t)r*nh+h] * (double)sumx[(size_t)n*nh+h];
                }
            }
            const double d=(double)Ydev[(size_t)n*M+r]-acc;
            num+=d*d; den+=acc*acc;
        }
    }
    return den==0 ? std::sqrt(num) : std::sqrt(num/den);
}

static double full_rel_l2(const std::vector<float>& a, const std::vector<float>& b){
    double num=0, den=0;
    const size_t n=a.size();
    for(size_t i=0;i<n;i++){
        const double d=(double)a[i]-(double)b[i];
        num+=d*d; den+=(double)b[i]*(double)b[i];
    }
    return den==0 ? std::sqrt(num) : std::sqrt(num/den);
}

struct Timing {
    double med=0, p10=0, p90=0;
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc,char** argv){
    std::string arch; int samples=kDefSamples, warmup=kDefWarmup;
    const int M=kM, K=kK, N=kN;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        auto next=[&]{ if(i+1>=argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} return std::atoi(argv[++i]); };
        if(a=="--arch"&&i+1<argc) arch=argv[++i];
        else if(a=="--samples") samples=next();
        else if(a=="--warmup") warmup=next();
        else { std::printf("Usage: %s --arch gfx1201 [--samples S] [--warmup W]\nIdentity: %s\n", argv[0], kBenchId); return a=="--help"||a=="-h" ? 0 : 2; }
    }
    if(arch.empty()){ std::fprintf(stderr,"--arch required\n"); return 2; }
    if(arch!="gfx1201"){ std::printf("SKIP: %s is gfx1201-only (requested %s)\n", kBenchId, arch.c_str()); return 0; }
    if(K%256){ std::fprintf(stderr,"K%%256!=0\n"); return 2; }

    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1; hipDeviceProp_t prop{};
    for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        if(std::strncmp(p.gcnArchName,arch.c_str(),arch.size())==0){ dev=d; prop=p; break; } }
    if(dev<0){ std::fprintf(stderr,"no device with gcnArchName %s*\n",arch.c_str()); return 2; }
    CK(hipSetDevice(dev));

    const int gpr=K/256;
    const size_t a_bytes=(size_t)M*gpr*kGroupBytes;
    const size_t w_plane_bytes=(size_t)M*K;
    std::printf("%s\narch=%s dev=%d (%s)\nbits=%d N=%d M=%d K=%d BV=8 K4  wireA=%zu planeW=%zu  samples=%d warmup=%d slabs=%d\n",
                kBenchId, arch.c_str(), dev, prop.name, kBits, N, M, K, a_bytes, w_plane_bytes, samples, warmup, kBatchLaunches);

    build_e4m3_table();

    std::vector<uint8_t> A; std::vector<uint16_t> Xh((size_t)N*K);
    pack_matrix_mq4(A,M,K,42);
    { uint64_t rng=7; for(auto& x:Xh) x=f32_to_f16_bits(lcg_finite(&rng,-1.f,1.f)); }

    std::vector<uint8_t> Wf8, Xf8;
    build_fp8_planes(A, Xh, Wf8, Xf8, M, K, N);
    std::vector<int8_t> Xq8;
    std::vector<float> scW, zpW, sx, sumx;
    build_iu8_aux(A, Xh, Xq8, scW, zpW, sx, sumx, M, K, N);

    std::vector<char*> a_slabs(kBatchLaunches);
    std::vector<unsigned char*> w_slabs(kBatchLaunches);
    for(int s=0;s<kBatchLaunches;s++){
        CK(hipMalloc((void**)&a_slabs[s], A.size()));
        CK(hipMemcpy(a_slabs[s], A.data(), A.size(), hipMemcpyHostToDevice));
        CK(hipMalloc((void**)&w_slabs[s], Wf8.size()));
        CK(hipMemcpy(w_slabs[s], Wf8.data(), Wf8.size(), hipMemcpyHostToDevice));
    }
    _Float16* dX; CK(hipMalloc((void**)&dX, Xh.size()*2)); CK(hipMemcpy(dX, Xh.data(), Xh.size()*2, hipMemcpyHostToDevice));
    unsigned char* dXf8; CK(hipMalloc((void**)&dXf8, Xf8.size())); CK(hipMemcpy(dXf8, Xf8.data(), Xf8.size(), hipMemcpyHostToDevice));
    int8_t* dXq8; CK(hipMalloc((void**)&dXq8, Xq8.size())); CK(hipMemcpy(dXq8, Xq8.data(), Xq8.size(), hipMemcpyHostToDevice));
    float *dScW, *dZpW, *dSx, *dSumx;
    CK(hipMalloc((void**)&dScW, scW.size()*4)); CK(hipMemcpy(dScW, scW.data(), scW.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc((void**)&dZpW, zpW.size()*4)); CK(hipMemcpy(dZpW, zpW.data(), zpW.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc((void**)&dSx, sx.size()*4)); CK(hipMemcpy(dSx, sx.data(), sx.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc((void**)&dSumx, sumx.size()*4)); CK(hipMemcpy(dSumx, sumx.data(), sumx.size()*4, hipMemcpyHostToDevice));
    float* dY; CK(hipMalloc((void**)&dY, (size_t)N*M*4));

    dim3 grid((M+15)/16, (N+128-1)/128), block(32);
    auto fetch=[&](std::vector<float>& out){
        out.resize((size_t)N*M);
        CK(hipMemcpy(out.data(), dY, out.size()*4, hipMemcpyDeviceToHost));
    };

    bool pass_a0=false, pass_a1=false, pass_a2=false;
    double rel_a0=0, rel_a1=0, rel_a2=0;
    double q_a1=0, q_a2=0;
    std::vector<float> y0, y1, y2;

    {
        hipLaunchKernelGGL(mb_qkv_fp8_iu8_a0, grid, block, 0, 0,
                           (const char*)a_slabs[0], (const _Float16*)dX, dY, M, K, N);
        CK(hipDeviceSynchronize());
        fetch(y0);
        rel_a0 = sampled_ref_a0(A, Xh, y0, M, K, N);
        pass_a0 = rel_a0 <= 1e-5;
        std::printf("HOSTREF A0 rel_l2=%.2e : %s\n", rel_a0, pass_a0?"PASS":"FAIL");
    }
    {
        hipLaunchKernelGGL(mb_qkv_fp8_iu8_a1, grid, block, 0, 0,
                           (const unsigned char*)w_slabs[0], (const unsigned char*)dXf8, dY, M, K, N);
        CK(hipDeviceSynchronize());
        fetch(y1);
        rel_a1 = sampled_ref_a1(Wf8, Xf8, y1, M, K, N);
        pass_a1 = rel_a1 <= 1e-4;
        q_a1 = full_rel_l2(y1, y0);
        std::printf("HOSTREF A1 rel_l2=%.2e : %s  quality_vs_A0=%.2e\n", rel_a1, pass_a1?"PASS":"FAIL", q_a1);
    }
    {
        hipLaunchKernelGGL(mb_qkv_fp8_iu8_a2, grid, block, 0, 0,
                           (const char*)a_slabs[0], (const int8_t*)dXq8,
                           dScW, dZpW, dSx, dSumx, dY, M, K, N);
        CK(hipDeviceSynchronize());
        fetch(y2);
        rel_a2 = sampled_ref_a2(A, Xq8, scW, zpW, sx, sumx, y2, M, K, N);
        pass_a2 = rel_a2 <= 1e-3;
        q_a2 = full_rel_l2(y2, y0);
        std::printf("HOSTREF A2 rel_l2=%.2e : %s  quality_vs_A0=%.2e\n", rel_a2, pass_a2?"PASS":"FAIL", q_a2);
    }

    auto launch_a0=[&](int slab){
        hipLaunchKernelGGL(mb_qkv_fp8_iu8_a0, grid, block, 0, 0,
                           (const char*)a_slabs[slab], (const _Float16*)dX, dY, M, K, N);
    };
    auto launch_a1=[&](int slab){
        hipLaunchKernelGGL(mb_qkv_fp8_iu8_a1, grid, block, 0, 0,
                           (const unsigned char*)w_slabs[slab], (const unsigned char*)dXf8, dY, M, K, N);
    };
    auto launch_a2=[&](int slab){
        hipLaunchKernelGGL(mb_qkv_fp8_iu8_a2, grid, block, 0, 0,
                           (const char*)a_slabs[slab], (const int8_t*)dXq8,
                           dScW, dZpW, dSx, dSumx, dY, M, K, N);
    };

    auto time_direct=[&](auto launch_fn)->Timing{
        for(int w=0;w<warmup;w++) launch_fn(w%kBatchLaunches);
        CK(hipDeviceSynchronize());
        hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
        std::vector<float> us; us.reserve(samples);
        for(int s=0;s<samples;s++){
            CK(hipEventRecord(ev0,0));
            for(int r=0;r<kBatchLaunches;r++) launch_fn(r);
            CK(hipEventRecord(ev1,0));
            CK(hipEventSynchronize(ev1));
            float ms=0; CK(hipEventElapsedTime(&ms,ev0,ev1));
            us.push_back(ms*1000.f/(float)kBatchLaunches);
        }
        CK(hipEventDestroy(ev0)); CK(hipEventDestroy(ev1));
        std::sort(us.begin(),us.end());
        Timing t;
        t.med=us[us.size()/2];
        t.p10=us[(size_t)(us.size()*0.10)];
        t.p90=us[(size_t)(us.size()*0.90)];
        return t;
    };

    Timing t0 = time_direct(launch_a0);
    Timing t1 = time_direct(launch_a1);
    Timing t2 = time_direct(launch_a2);

    auto gibs=[&](size_t bytes, double med_us){
        return (double)bytes/(med_us*1e-6)/(1024.0*1024.0*1024.0);
    };

    const double sp1 = t0.med/t1.med;
    const double sp2 = t0.med/t2.med;

    std::printf("%-14s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s(wire)=%7.2f  vsA0=%5.3fx  own_ref=%s\n",
                "A0_f16_base", t0.med, t0.p10, t0.p90, gibs(a_bytes,t0.med), 1.0, pass_a0?"PASS":"FAIL");
    std::printf("%-14s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s(wire)=%7.2f wGiB/s(plane)=%7.2f  vsA0=%5.3fx  own_ref=%s\n",
                "A1_fp8_dense", t1.med, t1.p10, t1.p90, gibs(a_bytes,t1.med), gibs(w_plane_bytes,t1.med),
                sp1, pass_a1?"PASS":"FAIL");
    std::printf("%-14s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s(wire)=%7.2f  vsA0=%5.3fx  own_ref=%s\n",
                "A2_iu8_affine", t2.med, t2.p10, t2.p90, gibs(a_bytes,t2.med), sp2, pass_a2?"PASS":"FAIL");

    auto verdict=[&](double sp, bool ok)->const char*{
        if(!ok) return "no-claim (ref FAIL)";
        if(sp > 1.02) return "win";
        if(sp < 0.98) return "loss";
        return "neutral";
    };
    std::printf("VERDICT A1 vs A0 = %.3fx %s  (quality_vs_A0=%.2e)\n", sp1, verdict(sp1,pass_a1), q_a1);
    std::printf("VERDICT A2 vs A0 = %.3fx %s  (quality_vs_A0=%.2e)\n", sp2, verdict(sp2,pass_a2), q_a2);

    std::printf("SUMMARY a0_med=%.4f a0_p10=%.4f a0_p90=%.4f a0_rel=%.3e "
                "a1_med=%.4f a1_p10=%.4f a1_p90=%.4f a1_rel=%.3e a1_q=%.3e a1_sp=%.4f a1_ok=%d "
                "a2_med=%.4f a2_p10=%.4f a2_p90=%.4f a2_rel=%.3e a2_q=%.3e a2_sp=%.4f a2_ok=%d "
                "wire=%zu plane=%zu\n",
                t0.med,t0.p10,t0.p90,rel_a0,
                t1.med,t1.p10,t1.p90,rel_a1,q_a1,sp1,(int)pass_a1,
                t2.med,t2.p10,t2.p90,rel_a2,q_a2,sp2,(int)pass_a2,
                a_bytes, w_plane_bytes);

    const bool pass = pass_a0 && pass_a1 && pass_a2;
    std::printf("%s\n", pass?"OVERALL PASS":"OVERALL FAIL");
    return pass?0:1;
}
