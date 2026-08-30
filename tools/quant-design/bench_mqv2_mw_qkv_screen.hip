// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mqv2-mw-qkv-screen-v1-20260829
//
// gfx11 HIP microbench: QKV BT vs MW-LDS same-row staging.
// Fixture: 3-region QKV with q_m=4096, k_m=1024, v_m=1024, K=5120.
// N swept over {128,256,512,1024,2048}; argv overrides plus required --arch.
// Per BITS in {3,5,6}.
// Arms (all use SAME packed A_q/A_k/A_v and X per BITS):
//   B0: QKV BT4  (BV=4)  — verbatim GEN_QKV_BT, current gfx1151 policy shape
//   B1: QKV BT12 (BV=12) — gfx1100 policy shape
//   B2: QKV MW-LDS NW=4  [16][16][16]
//   B3: QKV MW-LDS NW=8  [16][16][16]
//   B4: QKV MW-LDS NW=4  [16][16][18] (padded)
//   B5: QKV MW-LDS NW=8  [16][16][18] (padded)
//
// Timing: kBatchLaunches=8 independent A slabs per timed sample so weight
// working set stays DRAM-resident, HIP events, warmup, cyclic rotation.
// Correctness: B1..B5 f32 to_bits-equal to B0; B0 vs host double rel_l2<=1e-5.
// Build (single-arch, exact gfx11):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 \
//       tools/quant-design/bench_mqv2_mw_qkv_screen.hip \
//       -o .codeinsight+research/quant-design/mw-qkv-screen/bench_mqv2_mw_qkv_screen_gfx1151
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 \
//       tools/quant-design/bench_mqv2_mw_qkv_screen.hip \
//       -o .codeinsight+research/quant-design/mw-qkv-screen/bench_mqv2_mw_qkv_screen_gfx1100
// Run (arch selection, never rely on HIP_VISIBLE_DEVICES ordering):
//   ./bench --arch gfx1151  [--samples N] [--warmup N] [--Ns 128,256,...] [--q_m N --k_m N --v_m N --K N]
//
// Guard: device code is gfx11 only; other arches print SKIP and exit 0.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static constexpr const char* kBenchId = "mqv2-mw-qkv-screen-v1-20260829";
static constexpr int kWarmup = 30;
static constexpr int kSamples = 200;
static constexpr int kBatchLaunches = 8;

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ std::fprintf(stderr,"HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} }while(0)

static inline int v2_group_bytes(int bits){ return 8 + 32*bits; }

// ---------------------------------------------------------------------------
// Device code — gfx11 only (guard device definitions, host sees declarations)
// ---------------------------------------------------------------------------
// Device kernels — real for gfx11 device, stubs otherwise
#if defined(__HIP_DEVICE_COMPILE__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Verbatim decode_tile<BITS> from kernels/src/gemm_mqv2_wmma_gfx11_bt.hip:31-49
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
    if (safe_row < q_m) { A = A_q; local_row = safe_row; } \
    else if (safe_row < q_m + k_m) { A = A_k; local_row = safe_row - q_m; } \
    else { A = A_v; local_row = safe_row - q_m - k_m; } \
    const int groups_per_row = K / 256; \
    const long long row_stride = (long long)groups_per_row * GROUP_BYTES; \
    const char* row_base = A + local_row * row_stride; \
    int sb[BV]; \
    for (int b = 0; b < (BV); b++) { int t = batch_start + b * 16 + ml; sb[b] = (t < N) ? t : 0; } \
    float8_t acc[BV]; \
    for (int b = 0; b < (BV); b++) acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
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
        for (int b = 0; b < (BV); b++) xb[b] = X + (long long)sb[b] * K + g * 256; \
        for (int kt = 0; kt < 16; kt += 2) { \
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1; \
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1; \
            const int k_off_a = kt * 16; \
            const int k_off_b = (kt + 1) * 16; \
            half16_t a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h); \
            _Pragma("unroll") for (int b = 0; b < (BV); b++) { half16_t b_a = *(const half16_t*)(xb[b] + k_off_a); acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_a, acc[b]); } \
            a_reg = decode_tile<BITS>(payload, kt + 1, sc_h, zp_h); \
            _Pragma("unroll") for (int b = 0; b < (BV); b++) { half16_t b_b = *(const half16_t*)(xb[b] + k_off_b); acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_b, acc[b]); } \
        } \
    } \
    _Pragma("unroll") for (int b = 0; b < (BV); b++) { \
        const int out_col = batch_start + b * 16 + ml; \
        if (out_col < N) { _Pragma("unroll") for (int j = 0; j < 8; j++) { int out_row = row_start + 2 * j + (tid >> 4); if (out_row < total_m) { float* Y; int proj_row; int out_stride; if (out_row < q_m) { Y = Y_q; proj_row = out_row; out_stride = q_m; } else if (out_row < q_m + k_m) { Y = Y_k; proj_row = out_row - q_m; out_stride = k_m; } else { Y = Y_v; proj_row = out_row - q_m - k_m; out_stride = v_m; } Y[(long long)out_col * out_stride + proj_row] = acc[b][j]; } } } \
    } \
}

GEN_QKV_BT(3, 4)
GEN_QKV_BT(3, 12)
GEN_QKV_BT(5, 4)
GEN_QKV_BT(5, 12)
GEN_QKV_BT(6, 4)
GEN_QKV_BT(6, 12)

#define GEN_QKV_MW_LDS(BITS, NW, PAD) \
extern "C" __launch_bounds__(32 * (NW), 1) __global__ void \
gemm_qkv_mq##BITS##g256v2_wmma_gfx11_mw##NW##_lds##PAD( \
    const char* __restrict__ A_q, \
    const char* __restrict__ A_k, \
    const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, \
    float* __restrict__ Y_k, \
    float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, \
    int K, int N) { \
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
    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; \
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

GEN_QKV_MW_LDS(3, 4, 16)
GEN_QKV_MW_LDS(3, 8, 16)
GEN_QKV_MW_LDS(5, 4, 16)
GEN_QKV_MW_LDS(5, 8, 16)
GEN_QKV_MW_LDS(6, 4, 16)
GEN_QKV_MW_LDS(6, 8, 16)
GEN_QKV_MW_LDS(3, 4, 18)
GEN_QKV_MW_LDS(3, 8, 18)
GEN_QKV_MW_LDS(5, 4, 18)
GEN_QKV_MW_LDS(5, 8, 18)
GEN_QKV_MW_LDS(6, 4, 18)
GEN_QKV_MW_LDS(6, 8, 18)
#else // host — stubs (empty bodies, just for launch syntax)
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
template<int BITS> __device__ __forceinline__ half16_t decode_tile(const unsigned char*,int,_Float16,_Float16){half16_t o; for(int i=0;i<16;i++) o[i]=0; return o;}
#define GEN_QKV_BT_H(BITS,BV) extern "C" __global__ void gemm_qkv_mq##BITS##g256v2_wmma_gfx11_bt##BV(const char*,const char*,const char*,const _Float16*,float*,float*,float*,int,int,int,int,int){}
GEN_QKV_BT_H(3,4)
GEN_QKV_BT_H(3,12)
GEN_QKV_BT_H(5,4)
GEN_QKV_BT_H(5,12)
GEN_QKV_BT_H(6,4)
GEN_QKV_BT_H(6,12)
#define GEN_QKV_MW_H(BITS,NW,PAD) extern "C" __global__ void gemm_qkv_mq##BITS##g256v2_wmma_gfx11_mw##NW##_lds##PAD(const char*,const char*,const char*,const _Float16*,float*,float*,float*,int,int,int,int,int){}
GEN_QKV_MW_H(3,4,16)
GEN_QKV_MW_H(3,8,16)
GEN_QKV_MW_H(5,4,16)
GEN_QKV_MW_H(5,8,16)
GEN_QKV_MW_H(6,4,16)
GEN_QKV_MW_H(6,8,16)
GEN_QKV_MW_H(3,4,18)
GEN_QKV_MW_H(3,8,18)
GEN_QKV_MW_H(5,4,18)
GEN_QKV_MW_H(5,8,18)
GEN_QKV_MW_H(6,4,18)
GEN_QKV_MW_H(6,8,18)
#endif // __HIP_DEVICE_COMPILE__



// ===========================================================================
// Host side
// ===========================================================================
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
        const uint32_t rem=mant & ((1u<<(shift+13))-1u);
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
    uint32_t exp=(b>>10)&0x1Fu;
    uint32_t mant=b&0x3FFu;
    uint32_t out;
    if(exp==0){
        if(mant==0) out=sign;
        else{
            exp=127-15+1;
            while((mant&0x400u)==0){ mant<<=1; exp--; }
            mant&=0x3FFu;
            out=sign | (exp<<23) | (mant<<13);
        }
    } else if(exp==0x1Fu) out=sign|0x7F800000u|(mant<<13);
    else out=sign|((exp+(127-15))<<23)|(mant<<13);
    float f; std::memcpy(&f,&out,4); return f;
}

static float percentile_sorted(std::vector<float>& v, double p){
    if(v.empty()) return 0.f;
    const double idx=p*(double)(v.size()-1);
    const size_t lo=(size_t)idx;
    const size_t hi=std::min(lo+1, v.size()-1);
    const float t=(float)(idx-(double)lo);
    return v[lo]*(1.f-t)+v[hi]*t;
}

static void pack_group(uint8_t* gp, int bits, uint16_t sc0b,uint16_t zp0b,uint16_t sc1b,uint16_t zp1b, const uint8_t* qs){
    gp[0]= sc0b & 0xFF; gp[1]= sc0b>>8; gp[2]= zp0b &0xFF; gp[3]= zp0b>>8;
    gp[4]= sc1b & 0xFF; gp[5]= sc1b>>8; gp[6]= zp1b &0xFF; gp[7]= zp1b>>8;
    uint8_t* payload=gp+8;
    int P=32*bits;
    std::memset(payload,0,P);
    for(int c=0;c<256;c++){
        int bit_pos=c*bits;
        int byte_i=bit_pos>>3;
        int bit_in_byte=bit_pos&7;
        unsigned q=qs[c] & ((1u<<bits)-1u);
        payload[byte_i] |= (uint8_t)((q << bit_in_byte) & 0xFF);
        if(bit_in_byte+bits>8){
            payload[byte_i+1] |= (uint8_t)((q >> (8-bit_in_byte)) & 0xFF);
            if(bit_in_byte+bits>16){
                payload[byte_i+2] |= (uint8_t)((q >> (16-bit_in_byte)) & 0xFF);
            }
        }
    }
}

static void pack_matrix(std::vector<uint8_t>& out, int bits, int M, int K, uint64_t seed){
    int gpr=K/256;
    int gb=v2_group_bytes(bits);
    out.assign((size_t)M*gpr*gb,0);
    uint64_t rng=seed;
    int nlev=1<<bits;
    std::vector<uint8_t> qs(256);
    for(int r=0;r<M;r++){
        for(int g=0;g<gpr;g++){
            uint8_t* gp=out.data()+((size_t)r*gpr+g)*gb;
            float sc0=lcg_finite(&rng,0.01f,0.03f);
            float zp0=lcg_finite(&rng,-0.4f,0.4f);
            float sc1=lcg_finite(&rng,0.01f,0.03f);
            float zp1=lcg_finite(&rng,-0.4f,0.4f);
            uint16_t sc0b=f32_to_f16_bits(sc0);
            uint16_t zp0b=f32_to_f16_bits(zp0);
            uint16_t sc1b=f32_to_f16_bits(sc1);
            uint16_t zp1b=f32_to_f16_bits(zp1);
            for(int c=0;c<256;c++) qs[c]=(uint8_t)(lcg_u64(&rng)%(unsigned)nlev);
            pack_group(gp,bits,sc0b,zp0b,sc1b,zp1b,qs.data());
        }
    }
}

static void gen_X_half(std::vector<uint16_t>& out, int K, int N, uint64_t seed){
    out.resize((size_t)K*N);
    uint64_t rng=seed;
    for(size_t i=0;i<out.size();i++){
        float v=lcg_finite(&rng,-1.f,1.f);
        out[i]=f32_to_f16_bits(v);
    }
}

static void host_ref_qkv(const std::vector<uint8_t>& Aq, const std::vector<uint8_t>& Ak, const std::vector<uint8_t>& Av,
                         const std::vector<uint16_t>& Xh, int q_m,int k_m,int v_m,int K,int N,
                         std::vector<float>& Yq, std::vector<float>& Yk, std::vector<float>& Yv,
                         int bits){
    int gpr=K/256;
    int gb=v2_group_bytes(bits);
    Yq.assign((size_t)N*q_m,0.f);
    Yk.assign((size_t)N*k_m,0.f);
    Yv.assign((size_t)N*v_m,0.f);
    auto ref_one=[&](const std::vector<uint8_t>& A, std::vector<float>& Y, int M){
        for(int r=0;r<M;r++){
            for(int n=0;n<N;n++){
                double acc=0;
                for(int g=0; g<gpr; g++){
                    const uint8_t* gp=A.data()+((size_t)r*gpr+g)*gb;
                    float sc0=f16_bits_to_f32(*(uint16_t*)(gp+0));
                    float zp0=f16_bits_to_f32(*(uint16_t*)(gp+2));
                    float sc1=f16_bits_to_f32(*(uint16_t*)(gp+4));
                    float zp1=f16_bits_to_f32(*(uint16_t*)(gp+6));
                    const uint8_t* payload=gp+8;
                    for(int kt=0; kt<16; kt++){
                        float sc=(kt<8)?sc0:sc1;
                        float zp=(kt<8)?zp0:zp1;
                        for(int i=0;i<16;i++){
                            int c=kt*16+i;
                            int bit_pos=c*bits;
                            int byte_i=bit_pos>>3;
                            int bit_in_byte=bit_pos&7;
                            unsigned v=(unsigned)payload[byte_i];
                            if(bit_in_byte+bits>8) v|= (unsigned)payload[byte_i+1]<<8;
                            if(bit_in_byte+bits>16) v|= (unsigned)payload[byte_i+2]<<16;
                            unsigned q=(v>>bit_in_byte)&((1u<<bits)-1u);
                            float w=sc*(float)q+zp;
                            uint16_t xb=Xh[(size_t)n*K + g*256 + kt*16 + i];
                            float xf=f16_bits_to_f32(xb);
                            acc += (double)w * (double)xf;
                        }
                    }
                }
                Y[(size_t)n*M + r]=(float)acc;
            }
        }
    };
    ref_one(Aq,Yq,q_m);
    ref_one(Ak,Yk,k_m);
    ref_one(Av,Yv,v_m);
}

static double rel_l2(const std::vector<float>& a, const std::vector<float>& ref){
    double num=0, den=0;
    for(size_t i=0;i<a.size();i++){
        double d=(double)a[i]-(double)ref[i];
        num+=d*d;
        den+=(double)ref[i]*(double)ref[i];
    }
    if(den==0) return std::sqrt(num);
    return std::sqrt(num/den);
}

static void print_help(const char* a0){
    std::printf("Usage: %s --arch <gfxNNNN> [options]\n"
                "Identity: %s\n"
                "Options: --q_m N --k_m N --v_m N --K N --Ns 128,256,... --samples N --warmup N --help\n"
                "Defaults: q_m=4096 k_m=1024 v_m=1024 K=5120 Ns=128,256,512,1024,2048 samples=%d warmup=%d\n",
                a0, kBenchId, kSamples, kWarmup);
}

static bool is_gfx11_arch(const std::string& s){ return s=="gfx1100" || s=="gfx1151"; }

int main(int argc,char** argv){
    for(int i=1;i<argc;i++) if(!std::strcmp(argv[i],"--help")||!std::strcmp(argv[i],"-h")){ print_help(argv[0]); return 0; }
    std::string arch_req_early;
    for(int i=1;i<argc;i++) if(!std::strcmp(argv[i],"--arch") && i+1<argc) arch_req_early=argv[i+1];
    if(!arch_req_early.empty() && !is_gfx11_arch(arch_req_early)){
        std::printf("SKIP: %s requires gfx1100 or gfx1151 (requested %s)\n", kBenchId, arch_req_early.c_str());
        return 0;
    }
    // Full parsing
    std::string arch_parsed;
    int q_m=4096, k_m=1024, v_m=1024, K=5120;
    std::vector<int> Ns={128,256,512,1024,2048};
    int samples=kSamples;
    int warmup=kWarmup;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        auto need=[&](int j){ if(j>=argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} };
        if(a=="--arch"){ need(i+1); arch_parsed=argv[++i]; }
        else if(a=="--q_m"){ need(i+1); q_m=std::atoi(argv[++i]); }
        else if(a=="--k_m"){ need(i+1); k_m=std::atoi(argv[++i]); }
        else if(a=="--v_m"){ need(i+1); v_m=std::atoi(argv[++i]); }
        else if(a=="--K"){ need(i+1); K=std::atoi(argv[++i]); }
        else if(a=="--Ns"){ need(i+1); std::string s=argv[++i]; Ns.clear(); size_t p=0; while(p<s.size()){ size_t q=s.find(',',p); std::string t=s.substr(p, q==std::string::npos?std::string::npos:q-p); if(!t.empty()) Ns.push_back(std::atoi(t.c_str())); if(q==std::string::npos) break; p=q+1; } }
        else if(a=="--samples"){ need(i+1); samples=std::atoi(argv[++i]); }
        else if(a=="--warmup"){ need(i+1); warmup=std::atoi(argv[++i]); }
        else{ std::fprintf(stderr,"unknown arg %s\n",a.c_str()); print_help(argv[0]); return 2; }
    }
    std::string arch_req = arch_parsed;
    if(arch_req.empty()){
        std::fprintf(stderr,"ERROR: --arch <gfxNNNN> required\n");
        print_help(argv[0]); return 2;
    }
    if(K%256!=0){ std::fprintf(stderr,"K must be multiple of 256\n"); return 2; }
    int total_m=q_m+k_m+v_m;

    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1;
    hipDeviceProp_t chosen{};
    for(int d=0; d<ndev; d++){
        hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        std::string arch=p.gcnArchName;
        size_t colon=arch.find(':');
        std::string prefix = (colon==std::string::npos)?arch:arch.substr(0,colon);
        if(prefix==arch_req){ dev=d; chosen=p; break; }
        if(arch.rfind(arch_req,0)==0){ dev=d; chosen=p; break; }
    }
    if(dev<0){
        std::fprintf(stderr,"ERROR: --arch %s not found among %d devices:\n", arch_req.c_str(), ndev);
        for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d)); std::fprintf(stderr,"  device %d: %s  gcnArchName=%s\n", d, p.name, p.gcnArchName); }
        return 2;
    }
    CK(hipSetDevice(dev));
    hipDeviceProp_t prop=chosen;
    std::printf("=== %s ===\n", kBenchId);
    std::printf("arch_req=%s  device=%d  name=%s  gcnArchName=%s\n", arch_req.c_str(), dev, prop.name, prop.gcnArchName);
    std::printf("q_m=%d k_m=%d v_m=%d K=%d total_m=%d Ns=", q_m,k_m,v_m,K,total_m);
    for(size_t i=0;i<Ns.size();i++) std::printf("%s%d", i?",":"", Ns[i]); std::printf("  samples=%d warmup=%d batch=%d\n", samples, warmup, kBatchLaunches);
    // hipcc version for log — query via __VERSION__? We'll print ROCm version env
    std::printf("hipcc: %s\n", __VERSION__);

    using QkvFN = void(*)(const char*,const char*,const char*,const _Float16*,float*,float*,float*,int,int,int,int,int);
    auto pick_fn=[&](int bits,int arm)->QkvFN{
        if(arm==0){
            if(bits==3) return gemm_qkv_mq3g256v2_wmma_gfx11_bt4;
            if(bits==5) return gemm_qkv_mq5g256v2_wmma_gfx11_bt4;
            return gemm_qkv_mq6g256v2_wmma_gfx11_bt4;
        } else if(arm==1){
            if(bits==3) return gemm_qkv_mq3g256v2_wmma_gfx11_bt12;
            if(bits==5) return gemm_qkv_mq5g256v2_wmma_gfx11_bt12;
            return gemm_qkv_mq6g256v2_wmma_gfx11_bt12;
        } else if(arm==2){
            if(bits==3) return gemm_qkv_mq3g256v2_wmma_gfx11_mw4_lds16;
            if(bits==5) return gemm_qkv_mq5g256v2_wmma_gfx11_mw4_lds16;
            return gemm_qkv_mq6g256v2_wmma_gfx11_mw4_lds16;
        } else if(arm==3){
            if(bits==3) return gemm_qkv_mq3g256v2_wmma_gfx11_mw8_lds16;
            if(bits==5) return gemm_qkv_mq5g256v2_wmma_gfx11_mw8_lds16;
            return gemm_qkv_mq6g256v2_wmma_gfx11_mw8_lds16;
        } else if(arm==4){
            if(bits==3) return gemm_qkv_mq3g256v2_wmma_gfx11_mw4_lds18;
            if(bits==5) return gemm_qkv_mq5g256v2_wmma_gfx11_mw4_lds18;
            return gemm_qkv_mq6g256v2_wmma_gfx11_mw4_lds18;
        } else {
            if(bits==3) return gemm_qkv_mq3g256v2_wmma_gfx11_mw8_lds18;
            if(bits==5) return gemm_qkv_mq5g256v2_wmma_gfx11_mw8_lds18;
            return gemm_qkv_mq6g256v2_wmma_gfx11_mw8_lds18;
        }
    };
    const char* arm_names[6]={"B0_BT4","B1_BT12","B2_MW4","B3_MW8","B4_MW4_pad18","B5_MW8_pad18"};
    const int bits_list[3]={3,5,6};
    bool overall_pass=true;
    std::printf("bits N arm per_launch_us p10_us p90_us weight_GiB/s speedup_vs_B0 bit_equal\n");
    for(int bits: bits_list){
        int gb=v2_group_bytes(bits);
        size_t bytes_per_mat = (size_t)total_m * (K/256) * gb;
        size_t bq=(size_t)q_m*(K/256)*gb, bk=(size_t)k_m*(K/256)*gb, bv=(size_t)v_m*(K/256)*gb;
        std::vector<uint8_t> hAq,hAk,hAv;
        pack_matrix(hAq,bits,q_m,K, 0xBEEF0001ULL ^ (uint64_t)bits);
        pack_matrix(hAk,bits,k_m,K, 0xBEEF0002ULL ^ (uint64_t)bits);
        pack_matrix(hAv,bits,v_m,K, 0xBEEF0003ULL ^ (uint64_t)bits);
        int Nmax=*std::max_element(Ns.begin(), Ns.end());
        std::vector<uint16_t> hXh;
        gen_X_half(hXh, K, Nmax, 0xA11CE5EEDULL ^ (uint64_t)bits);
        for(int N: Ns){
            size_t x_elems=(size_t)K*N;
            std::vector<uint16_t> hX_slice(x_elems);
            for(int n=0;n<N;n++) std::memcpy(hX_slice.data()+(size_t)n*K, hXh.data()+(size_t)n*K, K*sizeof(uint16_t));
            std::vector<float> ref_q, ref_k, ref_v;
            host_ref_qkv(hAq,hAk,hAv,hX_slice,q_m,k_m,v_m,K,N,ref_q,ref_k,ref_v,bits);
            char *dAq=nullptr,*dAk=nullptr,*dAv=nullptr;
            CK(hipMalloc(&dAq, bq*kBatchLaunches));
            CK(hipMalloc(&dAk, bk*kBatchLaunches));
            CK(hipMalloc(&dAv, bv*kBatchLaunches));
            for(int r=0;r<kBatchLaunches;r++){
                CK(hipMemcpy(dAq+(size_t)r*bq, hAq.data(), bq, hipMemcpyHostToDevice));
                CK(hipMemcpy(dAk+(size_t)r*bk, hAk.data(), bk, hipMemcpyHostToDevice));
                CK(hipMemcpy(dAv+(size_t)r*bv, hAv.data(), bv, hipMemcpyHostToDevice));
            }
            _Float16* dX=nullptr;
            CK(hipMalloc(&dX, x_elems*sizeof(_Float16)));
            if(x_elems) CK(hipMemcpy(dX, hX_slice.data(), x_elems*sizeof(uint16_t), hipMemcpyHostToDevice));
            struct YTrip{ float *q,*k,*v; };
            YTrip dY[6];
            for(int a=0;a<6;a++){
                CK(hipMalloc(&dY[a].q, (size_t)q_m*N*sizeof(float)));
                CK(hipMalloc(&dY[a].k, (size_t)k_m*N*sizeof(float)));
                CK(hipMalloc(&dY[a].v, (size_t)v_m*N*sizeof(float)));
                CK(hipMemset(dY[a].q,0,(size_t)q_m*N*sizeof(float)));
                CK(hipMemset(dY[a].k,0,(size_t)k_m*N*sizeof(float)));
                CK(hipMemset(dY[a].v,0,(size_t)v_m*N*sizeof(float)));
            }
            std::vector<float> base_q,base_k,base_v, dev_q,dev_k,dev_v;
            bool correct[6]={};
            {
                QkvFN fn=pick_fn(bits,0);
                int bv0=4;
                dim3 grid((total_m+15)/16, (N+16*bv0-1)/(16*bv0));
                fn<<<grid,32>>>(dAq,dAk,dAv,dX,dY[0].q,dY[0].k,dY[0].v,q_m,k_m,v_m,K,N);
                CK(hipGetLastError()); CK(hipDeviceSynchronize());
                base_q.resize((size_t)N*q_m); base_k.resize((size_t)N*k_m); base_v.resize((size_t)N*v_m);
                if(N*q_m) CK(hipMemcpy(base_q.data(), dY[0].q, base_q.size()*4, hipMemcpyDeviceToHost));
                if(N*k_m) CK(hipMemcpy(base_k.data(), dY[0].k, base_k.size()*4, hipMemcpyDeviceToHost));
                if(N*v_m) CK(hipMemcpy(base_v.data(), dY[0].v, base_v.size()*4, hipMemcpyDeviceToHost));
                double rl2q=rel_l2(base_q, ref_q);
                double rl2k=rel_l2(base_k, ref_k);
                double rl2v=rel_l2(base_v, ref_v);
                double rl2 = std::max({rl2q,rl2k,rl2v});
                if(rl2 > 1e-3){
                    std::printf("FAIL BITS=%d N=%d B0 vs host rel_l2 %.3e (q %.3e k %.3e v %.3e) >1e-3\n", bits,N,rl2,rl2q,rl2k,rl2v);
                    // debug first few
                    for(int i=0;i<std::min((size_t)4, base_q.size()); i++) std::printf("  host_q[%d]=%f dev_q[%d]=%f\n", i, ref_q[i], i, base_q[i]);
                    for(int i=0;i<std::min((size_t)4, base_k.size()); i++) std::printf("  host_k[%d]=%f dev_k[%d]=%f\n", i, ref_k[i], i, base_k[i]);
                    overall_pass=false;
                }
            }
            correct[0]=true;
            for(int a=1;a<6;a++){
                QkvFN fn=pick_fn(bits,a);
                int block; dim3 grid;
                if(a==1){ int bvv=12; grid=dim3((total_m+15)/16, (N+16*bvv-1)/(16*bvv)); block=32; }
                else { int nw=(a==2||a==4?4:8); grid=dim3((total_m+15)/16, (N+16*nw-1)/(16*nw)); block=32*nw; }
                CK(hipMemset(dY[a].q,0,(size_t)q_m*N*sizeof(float)));
                CK(hipMemset(dY[a].k,0,(size_t)k_m*N*sizeof(float)));
                CK(hipMemset(dY[a].v,0,(size_t)v_m*N*sizeof(float)));
                fn<<<grid,block>>>(dAq,dAk,dAv,dX,dY[a].q,dY[a].k,dY[a].v,q_m,k_m,v_m,K,N);
                CK(hipGetLastError()); CK(hipDeviceSynchronize());
                dev_q.resize((size_t)N*q_m); dev_k.resize((size_t)N*k_m); dev_v.resize((size_t)N*v_m);
                if(N*q_m) CK(hipMemcpy(dev_q.data(), dY[a].q, dev_q.size()*4, hipMemcpyDeviceToHost));
                if(N*k_m) CK(hipMemcpy(dev_k.data(), dY[a].k, dev_k.size()*4, hipMemcpyDeviceToHost));
                if(N*v_m) CK(hipMemcpy(dev_v.data(), dY[a].v, dev_v.size()*4, hipMemcpyDeviceToHost));
                auto bits_equal=[&](const std::vector<float>& p, const std::vector<float>& q)->bool{
                    if(p.size()!=q.size()) return false;
                    for(size_t i=0;i<p.size();i++){ uint32_t a,b; std::memcpy(&a,&p[i],4); std::memcpy(&b,&q[i],4); if(a!=b) return false; }
                    return true;
                };
                bool eq_q=bits_equal(dev_q, base_q);
                bool eq_k=bits_equal(dev_k, base_k);
                bool eq_v=bits_equal(dev_v, base_v);
                if(!eq_q||!eq_k||!eq_v){
                    std::string which; if(!eq_q) which+="Q"; if(!eq_k) which+="K"; if(!eq_v) which+="V";
                    std::printf("FAIL BITS=%d N=%d %s vs B0 bit-mismatch regions=%s\n", bits,N,arm_names[a],which.c_str());
                    overall_pass=false;
                    correct[a]=false;
                } else correct[a]=true;
            }
            for(int w=0;w<warmup;w++){
                for(int a=0;a<6;a++){
                    QkvFN fn=pick_fn(bits,a);
                    if(a<=1){ int bvv=(a==0?4:12); dim3 grid((total_m+15)/16, (N+16*bvv-1)/(16*bvv)); for(int r=0;r<kBatchLaunches;r++){ char* aq=dAq+r*bq; char* ak=dAk+r*bk; char* av=dAv+r*bv; fn<<<grid,32>>>(aq,ak,av,dX,dY[a].q,dY[a].k,dY[a].v,q_m,k_m,v_m,K,N); } }
                    else { int nw=(a==2||a==4?4:8); dim3 grid((total_m+15)/16, (N+16*nw-1)/(16*nw)); for(int r=0;r<kBatchLaunches;r++){ char* aq=dAq+r*bq; char* ak=dAk+r*bk; char* av=dAv+r*bv; fn<<<grid,32*nw>>>(aq,ak,av,dX,dY[a].q,dY[a].k,dY[a].v,q_m,k_m,v_m,K,N); } }
                }
            }
            CK(hipDeviceSynchronize());
            std::vector<std::vector<float>> samples_us(6);
            hipEvent_t ev0,ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
            for(int s=0;s<samples;s++){
                int order[6]={0,1,2,3,4,5};
                int rot=s%6; std::rotate(order, order+rot, order+6);
                if(s&1) std::reverse(order, order+6);
                for(int oi=0;oi<6;oi++){
                    int a=order[oi];
                    QkvFN fn=pick_fn(bits,a);
                    CK(hipEventRecord(ev0,nullptr));
                    if(a<=1){ int bvv=(a==0?4:12); dim3 grid((total_m+15)/16, (N+16*bvv-1)/(16*bvv)); for(int r=0;r<kBatchLaunches;r++){ char* aq=dAq+r*bq; char* ak=dAk+r*bk; char* av=dAv+r*bv; fn<<<grid,32>>>(aq,ak,av,dX,dY[a].q,dY[a].k,dY[a].v,q_m,k_m,v_m,K,N); } }
                    else { int nw=(a==2||a==4?4:8); dim3 grid((total_m+15)/16, (N+16*nw-1)/(16*nw)); for(int r=0;r<kBatchLaunches;r++){ char* aq=dAq+r*bq; char* ak=dAk+r*bk; char* av=dAv+r*bv; fn<<<grid,32*nw>>>(aq,ak,av,dX,dY[a].q,dY[a].k,dY[a].v,q_m,k_m,v_m,K,N); } }
                    CK(hipEventRecord(ev1,nullptr));
                    CK(hipEventSynchronize(ev1));
                    float ms=0; CK(hipEventElapsedTime(&ms,ev0,ev1));
                    float per_launch_us = (ms*1000.f)/(float)kBatchLaunches;
                    samples_us[a].push_back(per_launch_us);
                }
            }
            CK(hipEventDestroy(ev0)); CK(hipEventDestroy(ev1));
            std::vector<float> med(6),p10(6),p90(6),gibs(6);
            for(int a=0;a<6;a++){
                auto &v=samples_us[a];
                std::sort(v.begin(), v.end());
                med[a]=percentile_sorted(v,0.50);
                p10[a]=percentile_sorted(v,0.10);
                p90[a]=percentile_sorted(v,0.90);
                double sec=(double)med[a]*1e-6;
                gibs[a]= sec>0 ? (float)((double)bytes_per_mat/sec/(1024.*1024.*1024.)) : 0.f;
            }
            float b0med=med[0];
            for(int a=0;a<6;a++){
                float speedup = (med[a]>0 && b0med>0) ? b0med/med[a] : 0.f;
                const char* bit_eq = (a==0)?"PASS":(correct[a]?"PASS":"FAIL");
                std::printf("%d %d %s %.2f %.2f %.2f %.2f %.3f %s\n",
                    bits, N, arm_names[a], med[a], p10[a], p90[a], gibs[a], speedup, bit_eq);
            }
            CK(hipFree(dAq)); CK(hipFree(dAk)); CK(hipFree(dAv));
            CK(hipFree(dX));
            for(int a=0;a<6;a++){ CK(hipFree(dY[a].q)); CK(hipFree(dY[a].k)); CK(hipFree(dY[a].v)); }
        }
    }
    if(!overall_pass) std::printf("CORRECTNESS: FAIL\n");
    else std::printf("CORRECTNESS: PASS\n");
    std::printf("identity: %s\n", kBenchId);
    return overall_pass?0:1;
}
