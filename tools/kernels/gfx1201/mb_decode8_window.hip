// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mb-decode8-window-gfx1201-v1-20260830
//
// QUESTION: does the dword-window decode8 beat the verbatim baseline byte-load
// decode8<BITS> in the BT8 K4 single-matrix body on gfx1201 (Navi 48, RDNA4)?
// The window collapses 8 per-code byte loads + branches to 1-2 aligned u64
// loads + funnel shifts per 8-wide fragment; does the reduced load/branch
// pressure outweigh the extra integer ops on this arch's WMMA path?
//
// Production construct interrogated (arm 0, verbatim):
//   kernels/src/gemm_qkv_mqv2_wmma_gfx1201_bt.hip  decode8<BITS> + BT8 K4 body
//   (BV=8, K4 quartet kt in {0,4,8,12}, half8 decode, gfx12 WMMA).
// Candidate arm (extracted verbatim from the monolith's gfx1201 section):
//   tools/quant-design/bench_mqv2_decode_screen.hip  decode8_window<BITS>
//   (A1 dword-window, float convert; monolith § decode8_window).
//
// BANKED VERDICT (durable home note):
//   The verdict for this question on gfx1201 is already banked from the
//   sweep monolith .codeinsight+research/quant-design/decode-screen/report.md:
//   dword-window is a LOSS on production shapes (M=4096 K=5120 N=512, all
//   BITS 3/5/6 and BITS2 N512) and a win only for the narrow BITS2 N128 case
//   (2.2x, not a production shape). For the production-relevant
//   BITS 3/5/6 at both N and BITS2 N512 it is neutral-to-loss (0.52-0.95x).
//   Monolith was 2026-08-29 on local gfx1201 (RX 9070, ROCm 7.14) and hipx
//   gfx1100/gfx1151. This file is the durable per-question home that
//   preserves the interrogation as a <30 s per-run microbench; it does not
//   re-litigate the sweep.
//
// Arms:
//   A0 baseline — verbatim decode8<BITS> (production)
//   A1 window   — verbatim decode8_window<BITS> from the monolith
//   Both preserve per-output-element g-ascending, kt-ascending accumulation
//   order, so bit-equality is exact.
//
// Build (local gfx1201):
//   /opt/rocm/core/bin/hipcc --offload-arch=gfx1201 -O3 \
//       tools/kernels/gfx1201/mb_decode8_window.hip -o mb_decode8_window
// Run:
//   ./mb_decode8_window --arch gfx1201 [--bits 2|3|5|6] [--n N] [--samples S] [--warmup W]
//   Defaults: bits=3 n=512 samples=50 warmup=20  M=4096 K=5120 BV=8 K4 fixed
//
// Correctness: A0 vs sampled host double reference rel_l2 <= 1e-5; A1 f32
// to_bits-equal to A0. Timing: warmup, HIP events, 8 independent A slabs
// per sample rotated inside one event pair (DRAM-resident), per-launch µs +
// weight GiB/s, median/p10/p90 + speedup vs A0.

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

static constexpr const char* kBenchId = "mb-decode8-window-gfx1201-v1-20260830";
static constexpr int kDefWarmup = 20;
static constexpr int kDefSamples = 50;
static constexpr int kBatchLaunches = 8;

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ std::fprintf(stderr,"HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} }while(0)

static inline int v2_group_bytes(int bits){ return 8 + 32*bits; }

// ---------------------------------------------------------------------------
// Device code — exact gfx1201 only (host pass and other arches take stub)
// ---------------------------------------------------------------------------
#if defined(__gfx1201__)

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// Verbatim production decoder (kernels/src/gemm_qkv_mqv2_wmma_gfx1201_bt.hip:38-56)
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

// Verbatim dword-window decoder (tools/quant-design/bench_mqv2_decode_screen.hip:584-611 gfx1201 section)
template <int BITS>
__device__ __forceinline__ half8_t decode8_window(const unsigned char* __restrict__ codes,
                                                  int code0,
                                                  _Float16 sc_h,
                                                  _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    const int off = (code0 >> 3) * BITS;
    const int bit_in = (off & 7) * 8;
    const int idx = off >> 3;
    const uint64_t* __restrict__ pu64 = (const uint64_t* __restrict__)codes;
    uint64_t w0 = pu64[idx];
    uint64_t w1 = 0;
    if ((bit_in + 8 * BITS) > 64) w1 = pu64[idx + 1];
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int b = bit_in + i * BITS;
        unsigned q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned)((w0 >> b) & qmask);
            else q = (unsigned)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else {
            q = (unsigned)((w1 >> (b - 64)) & qmask);
        }
        a[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return a;
}

template <int BITS, int ARM>
__device__ __forceinline__ void body_gfx1201_window(const char* __restrict__ A,
                                                    const _Float16* __restrict__ X,
                                                    float* __restrict__ Y,
                                                    int M, int K, int N) {
    constexpr int GROUP_BYTES = 8 + 32 * (BITS);
    constexpr int BV = 8;
    const int total_m = M;
    const int rs = (int)blockIdx.x * 16;
    const int bs = (int)blockIdx.y * (16 * BV);
    const int tid = (int)threadIdx.x;
    if (rs >= total_m || bs >= N) return;
    const int ml = tid & 15;
    const int kg = tid >> 4;
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1);
    const char* A_ptr = A;
    int lrow = sr;
    const int gpr = K / 256;
    const char* rb = A_ptr + (long long)lrow * (long long)gpr * (long long)GROUP_BYTES;
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
                half8_t a;
                if constexpr (ARM == 0) a = decode8<BITS>(codes, code0, sc_h, zp_h);
                else a = decode8_window<BITS>(codes, code0, sc_h, zp_h);
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

#define GEN_WINDOW(BITS, ARM, NAME) \
extern "C" __launch_bounds__(32, 2) __global__ void NAME(const char* A, const _Float16* X, float* Y, int M, int K, int N){ body_gfx1201_window<BITS, ARM>(A,X,Y,M,K,N); }

GEN_WINDOW(2, 0, mb_decode8_window_mq2_a0)
GEN_WINDOW(2, 1, mb_decode8_window_mq2_a1)
GEN_WINDOW(3, 0, mb_decode8_window_mq3_a0)
GEN_WINDOW(3, 1, mb_decode8_window_mq3_a1)
GEN_WINDOW(5, 0, mb_decode8_window_mq5_a0)
GEN_WINDOW(5, 1, mb_decode8_window_mq5_a1)
GEN_WINDOW(6, 0, mb_decode8_window_mq6_a0)
GEN_WINDOW(6, 1, mb_decode8_window_mq6_a1)

#undef GEN_WINDOW

#else // host pass / other arches — matching stubs
#define GEN_STUB(NAME) extern "C" __global__ void NAME(const char*,const _Float16*,float*,int,int,int){}
GEN_STUB(mb_decode8_window_mq2_a0) GEN_STUB(mb_decode8_window_mq2_a1)
GEN_STUB(mb_decode8_window_mq3_a0) GEN_STUB(mb_decode8_window_mq3_a1)
GEN_STUB(mb_decode8_window_mq5_a0) GEN_STUB(mb_decode8_window_mq5_a1)
GEN_STUB(mb_decode8_window_mq6_a0) GEN_STUB(mb_decode8_window_mq6_a1)
#undef GEN_STUB
#endif

// ---------------------------------------------------------------------------
// Host helpers — LCG, f16, packing, sampled reference (copied from authority)
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
                const _Float16 sc0_h = (_Float16)f16_bits_to_f32((uint16_t)(gp[0]|(gp[1]<<8)));
                const _Float16 zp0_h = (_Float16)f16_bits_to_f32((uint16_t)(gp[2]|(gp[3]<<8)));
                const _Float16 sc1_h = (_Float16)f16_bits_to_f32((uint16_t)(gp[4]|(gp[5]<<8)));
                const _Float16 zp1_h = (_Float16)f16_bits_to_f32((uint16_t)(gp[6]|(gp[7]<<8)));
                const uint8_t* payload=gp+8;
                for(int c=0;c<256;c++){
                    const int bit_pos=c*bits, byte_i=bit_pos>>3, bib=bit_pos&7;
                    unsigned v=(unsigned)payload[byte_i];
                    if(bib+bits>8) v|=(unsigned)payload[byte_i+1]<<8;
                    const unsigned q=(v>>bib)&((1u<<bits)-1u);
                    const _Float16 sc_h = (c<128)?sc0_h:sc1_h;
                    const _Float16 zp_h = (c<128)?zp0_h:zp1_h;
                    const _Float16 q_h = (_Float16)(float)q;
                    const _Float16 w_h = sc_h * q_h + zp_h;
                    const float w = (float)w_h;
                    acc += (double)w*(double)f16_bits_to_f32(Xh[(size_t)n*K+g*256+c]);
                }
            }
            const double d=(double)Ydev[(size_t)n*M+r]-acc;
            num+=d*d; den+=acc*acc;
        }
    }
    return den==0 ? std::sqrt(num) : std::sqrt(num/den);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc,char** argv){
    std::string arch; int bits=3, N=512, samples=kDefSamples, warmup=kDefWarmup;
    const int M=4096, K=5120;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        auto next=[&]{ if(i+1>=argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} return std::atoi(argv[++i]); };
        if(a=="--arch"&&i+1<argc) arch=argv[++i];
        else if(a=="--bits") bits=next();
        else if(a=="--n") N=next();
        else if(a=="--samples") samples=next();
        else if(a=="--warmup") warmup=next();
        else { std::printf("Usage: %s --arch gfx1201 [--bits 2|3|5|6] [--n N] [--samples S] [--warmup W]\nIdentity: %s\n", argv[0], kBenchId); return a=="--help"||a=="-h" ? 0 : 2; }
    }
    if(arch.empty()){ std::fprintf(stderr,"--arch required\n"); return 2; }
    if(arch!="gfx1201"){ std::printf("SKIP: %s is gfx1201-only (requested %s)\n", kBenchId, arch.c_str()); return 0; }
    if(bits!=2&&bits!=3&&bits!=5&&bits!=6){ std::fprintf(stderr,"--bits must be 2, 3, 5 or 6\n"); return 2; }
    if(K%256){ std::fprintf(stderr,"K%%256!=0\n"); return 2; }

    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1; hipDeviceProp_t prop{};
    for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        if(std::strncmp(p.gcnArchName,arch.c_str(),arch.size())==0){ dev=d; prop=p; break; } }
    if(dev<0){ std::fprintf(stderr,"no device with gcnArchName %s*\n",arch.c_str()); return 2; }
    CK(hipSetDevice(dev));

    const int total_m=M, gpr=K/256, gb=v2_group_bytes(bits);
    const size_t a_bytes=(size_t)total_m*gpr*gb;
    std::printf("%s\narch=%s dev=%d (%s)\nbits=%d N=%d M=%d K=%d BV=8 K4  A=%zu bytes  samples=%d warmup=%d slabs=%d\n",
                kBenchId, arch.c_str(), dev, prop.name, bits, N, M, K, a_bytes, samples, warmup, kBatchLaunches);

    // Fixtures
    std::vector<uint8_t> A; std::vector<uint16_t> Xh((size_t)N*K);
    pack_matrix(A,bits,M,K,42);
    { uint64_t rng=7; for(auto& x:Xh) x=f32_to_f16_bits(lcg_finite(&rng,-1.f,1.f)); }

    std::vector<char*> slabs(kBatchLaunches);
    for(int s=0;s<kBatchLaunches;s++){
        CK(hipMalloc((void**)&slabs[s], A.size()));
        CK(hipMemcpy(slabs[s], A.data(), A.size(), hipMemcpyHostToDevice));
    }
    _Float16* dX; CK(hipMalloc((void**)&dX, Xh.size()*2)); CK(hipMemcpy(dX, Xh.data(), Xh.size()*2, hipMemcpyHostToDevice));
    float* dY; CK(hipMalloc((void**)&dY, (size_t)N*M*4));

    using KFn = void(*)(const char*,const _Float16*,float*,int,int,int);
    KFn a0=nullptr, a1=nullptr;
    if(bits==2){ a0=mb_decode8_window_mq2_a0; a1=mb_decode8_window_mq2_a1; }
    else if(bits==3){ a0=mb_decode8_window_mq3_a0; a1=mb_decode8_window_mq3_a1; }
    else if(bits==5){ a0=mb_decode8_window_mq5_a0; a1=mb_decode8_window_mq5_a1; }
    else { a0=mb_decode8_window_mq6_a0; a1=mb_decode8_window_mq6_a1; }
    struct Arm{ const char* name; KFn fn; };
    const Arm arms[2]={ {"A0_baseline", a0}, {"A1_window", a1} };

    auto launch=[&](const Arm& ar, char* slab){
        dim3 grid((M+15)/16, (N+128-1)/128), block(32);
        hipLaunchKernelGGL(ar.fn, grid, block, 0, 0, (const char*)slab, (const _Float16*)dX, (float*)dY, M, K, N);
    };
    auto fetch=[&](std::vector<float>& out){
        out.resize((size_t)N*M);
        CK(hipMemcpy(out.data(), dY, out.size()*4, hipMemcpyDeviceToHost));
    };
    auto biteq=[&](const std::vector<float>& a,const std::vector<float>& b){
        if(a.size()!=b.size()) return false;
        for(size_t i=0;i<a.size();i++){ uint32_t x,y; std::memcpy(&x,&a[i],4); std::memcpy(&y,&b[i],4); if(x!=y) return false; }
        return true;
    };

    // Correctness
    bool pass=true;
    std::vector<float> y0;
    launch(arms[0], slabs[0]); CK(hipDeviceSynchronize()); fetch(y0);
    const double rel = sampled_ref_rel_l2(A, Xh, y0, M, K, N, bits);
    const bool refok = rel <= 1e-5;
    std::printf("HOSTREF A0 rel_l2=%.2e : %s\n", rel, refok?"PASS":"FAIL");
    pass &= refok;
    {
        std::vector<float> y1;
        launch(arms[1], slabs[0]); CK(hipDeviceSynchronize()); fetch(y1);
        const bool ok = biteq(y1, y0);
        std::printf("BITEQ A1 vs A0 : %s\n", ok?"PASS":"FAIL");
        if(!ok){
            size_t mis=0; for(size_t i=0;i<y0.size();i++){ uint32_t x,y; std::memcpy(&x,&y0[i],4); std::memcpy(&y,&y1[i],4); if(x!=y){mis=i; break;}}
            uint32_t xa,ya; std::memcpy(&xa,&y0[mis],4); std::memcpy(&ya,&y1[mis],4);
            std::printf("  first mismatch %zu: A0=0x%08x (%.6f) A1=0x%08x (%.6f)\n", mis, xa, y0[mis], ya, y1[mis]);
        }
        pass &= ok;
    }

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    double med_us[2]={0,0};
    for(int ai=0; ai<2; ai++){
        for(int w=0;w<warmup;w++) launch(arms[ai], slabs[w%kBatchLaunches]);
        CK(hipDeviceSynchronize());
        std::vector<float> us; us.reserve(samples);
        for(int s=0;s<samples;s++){
            CK(hipEventRecord(ev0,0));
            for(int r=0;r<kBatchLaunches;r++) launch(arms[ai], slabs[r]);
            CK(hipEventRecord(ev1,0));
            CK(hipEventSynchronize(ev1));
            float ms=0; CK(hipEventElapsedTime(&ms,ev0,ev1));
            us.push_back(ms*1000.f/(float)kBatchLaunches);
        }
        std::sort(us.begin(),us.end());
        med_us[ai]=us[us.size()/2];
        const double gibs=(double)a_bytes/(med_us[ai]*1e-6)/(1024.0*1024.0*1024.0);
        std::printf("%-12s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s=%7.2f  vsA0=%5.3fx  %s\n",
                    arms[ai].name, med_us[ai], us[(size_t)(us.size()*0.10)], us[(size_t)(us.size()*0.90)],
                    gibs, med_us[0]/med_us[ai], (ai==0||pass)?"":"FAIL");
    }
    std::printf("SPEEDUP A1 vs A0 = %.3fx  (%s)\n", med_us[0]/med_us[1],
                (med_us[0]/med_us[1] > 1.02 ? "win" : (med_us[0]/med_us[1] < 0.98 ? "loss" : "neutral")));
    std::printf("%s\n", pass?"OVERALL PASS":"OVERALL FAIL");
    return pass?0:1;
}
