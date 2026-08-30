// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mb-payload-nt-gfx1100-v1-20260829
//
// QUESTION: does __builtin_nontemporal_load on payload byte reads help the
// verbatim decode_tile path on gfx1100?
//
// Production construct interrogated (arm 0, verbatim):
//   decode_tile<BITS> in kernels/src/gemm_mqv2_wmma_gfx11_bt.hip:31-49
//   GEN_RESID_BT-style single-matrix overwrite body (M rows, X [N x K] fp16,
//   Y [N x M] f32 overwrite, kt 0..15 ascending), extracted verbatim from
//   tools/quant-design/bench_mqv2_decode_screen.hip body_gfx11.
// Candidate:
//   A0 baseline decode_tile (byte loads, float(q) convert)
//   A1 same decode but payload bytes via __builtin_nontemporal_load
//      (NT wrapping extracted from bench_mqv2_decode_screen.hip A3 window_nt
//      variant, but applied to the BASELINE byte-load decode — not the window
//      decode — so the question is the hint alone).
// Correctness: A1 f32 to_bits-equal to A0; A0 vs sampled host double ref rel_l2 <= 1e-4
// (host ref mirrors device fp16 rounding).
//
// Build gfx1100 (single-arch is fine on ROCm 7.15 for gfx1100):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 \
//       tools/kernels/gfx1100/mb_payload_nt.hip -o mb_payload_nt
// Run:
//   ./mb_payload_nt --arch gfx1100 [--bits 2|3|5|6] [--n N] [--bv 4|12] [--samples S] [--warmup W]
//   Defaults: bits=3 n=512 bv=12 M=4096 K=5120 samples=50 warmup=20

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

static constexpr const char* kBenchId = "mb-payload-nt-gfx1100-v1-20260829";
static constexpr int kDefWarmup = 20;
static constexpr int kDefSamples = 50;
static constexpr int kBatchLaunches = 8;
static constexpr int kDefM = 4096;
static constexpr int kDefK = 5120;

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ std::fprintf(stderr,"HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} }while(0)

static inline int v2_group_bytes(int bits){ return 8 + 32*bits; }

// ---------------------------------------------------------------------------
// Device code — gfx11 only
// ---------------------------------------------------------------------------
#if defined(__gfx1100__) || defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

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

template <int BITS>
__device__ __forceinline__ half16_t decode_tile_nt(const unsigned char* __restrict__ payload,
                                                   int kt, _Float16 sc_h, _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int bit_pos = (kt * 16 + i) * BITS;
        const int byte_i = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned int v = (unsigned int)__builtin_nontemporal_load(payload + byte_i);
        if ((bit_in_byte + BITS) > 8)
            v |= ((unsigned int)__builtin_nontemporal_load(payload + byte_i + 1)) << 8;
        const unsigned int q = (v >> bit_in_byte) & qmask;
        out[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return out;
}

template <int BITS, int BV, int ARM>
__device__ __forceinline__ void body_gfx11(const char* __restrict__ A,
                                           const _Float16* __restrict__ X,
                                           float* __restrict__ Y,
                                           int M, int K, int N) {
    constexpr int GROUP_BYTES = 8 + 32 * (BITS);
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * (16 * (BV));
    const int tid = threadIdx.x;
    if (row_start >= M) return;
    if (batch_start >= N) return;
    const int ml = tid & 15;
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * GROUP_BYTES;
    int sb[BV];
    for (int b = 0; b < (BV); b++) {
        int t = batch_start + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }
    float8_t acc[BV];
    for (int b = 0; b < (BV); b++)
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * GROUP_BYTES;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const unsigned char* payload = (const unsigned char*)(gp + 8);
        const _Float16* xg[BV];
        for (int b = 0; b < (BV); b++)
            xg[b] = X + (long long)sb[b] * K + g * 256;
        _Pragma("unroll")
        for (int kt = 0; kt < 16; kt++) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;
            const int k_off = kt * 16;
            half16_t a_reg;
            if constexpr (ARM == 0) a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h);
            else a_reg = decode_tile_nt<BITS>(payload, kt, sc_h, zp_h);
            _Pragma("unroll")
            for (int b = 0; b < (BV); b++) {
                half16_t b_reg = *(const half16_t*)(xg[b] + k_off);
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]);
            }
        }
    }
    _Pragma("unroll")
    for (int b = 0; b < (BV); b++) {
        const int out_col = batch_start + b * 16 + ml;
        if (out_col < N) {
            _Pragma("unroll")
            for (int j = 0; j < 8; j++) {
                int out_row = row_start + 2 * j + (tid >> 4);
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[b][j];
                }
            }
        }
    }
}

#define GEN_KERNEL(BITS,BV,ARM) extern "C" __launch_bounds__(32, 2) __global__ void mb_payload_nt_mq##BITS##_bv##BV##_a##ARM(const char* A, const _Float16* X, float* Y, int M,int K,int N){     body_gfx11<BITS,BV,ARM>(A,X,Y,M,K,N); }

GEN_KERNEL(2,4,0)
GEN_KERNEL(2,4,1)
GEN_KERNEL(2,12,0)
GEN_KERNEL(2,12,1)
GEN_KERNEL(3,4,0)
GEN_KERNEL(3,4,1)
GEN_KERNEL(3,12,0)
GEN_KERNEL(3,12,1)
GEN_KERNEL(5,4,0)
GEN_KERNEL(5,4,1)
GEN_KERNEL(5,12,0)
GEN_KERNEL(5,12,1)
GEN_KERNEL(6,4,0)
GEN_KERNEL(6,4,1)
GEN_KERNEL(6,12,0)
GEN_KERNEL(6,12,1)

#else
#define GEN_STUB(BITS,BV,ARM) extern "C" __global__ void mb_payload_nt_mq##BITS##_bv##BV##_a##ARM(const char*,const _Float16*,float*,int,int,int){}
GEN_STUB(2,4,0) GEN_STUB(2,4,1) GEN_STUB(2,12,0) GEN_STUB(2,12,1)
GEN_STUB(3,4,0) GEN_STUB(3,4,1) GEN_STUB(3,12,0) GEN_STUB(3,12,1)
GEN_STUB(5,4,0) GEN_STUB(5,4,1) GEN_STUB(5,12,0) GEN_STUB(5,12,1)
GEN_STUB(6,4,0) GEN_STUB(6,4,1) GEN_STUB(6,12,0) GEN_STUB(6,12,1)
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
                    const float prod=f16_bits_to_f32(f32_to_f16_bits(sc*(float)q));
                    const float w=f16_bits_to_f32(f32_to_f16_bits(prod+zp));
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
    std::string arch; int bits=3, N=512, bv=12, M=kDefM, K=kDefK;
    int samples=kDefSamples, warmup=kDefWarmup;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        auto next=[&]{ if(i+1>=argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} return std::atoi(argv[++i]); };
        if(a=="--arch"&&i+1<argc) arch=argv[++i];
        else if(a=="--bits") bits=next();
        else if(a=="--n") N=next();
        else if(a=="--bv") bv=next();
        else if(a=="--M") M=next();
        else if(a=="--K") K=next();
        else if(a=="--samples") samples=next();
        else if(a=="--warmup") warmup=next();
        else { std::printf("Usage: %s --arch gfxNNNN [--bits 2|3|5|6] [--n N] [--bv 4|12] [--M M] [--K K] [--samples S] [--warmup W]\nIdentity: %s\n", argv[0], kBenchId); return a=="--help"||a=="-h" ? 0 : 2; }
    }
    if(arch.empty()){ std::fprintf(stderr,"--arch required\n"); return 2; }
    if(arch!="gfx1100"&&arch!="gfx1151"){ std::printf("SKIP: %s is gfx11-only (requested %s)\n", kBenchId, arch.c_str()); return 0; }
    if(bits!=2&&bits!=3&&bits!=5&&bits!=6){ std::fprintf(stderr,"--bits must be 2,3,5 or 6\n"); return 2; }
    if(bv!=4&&bv!=12){ std::fprintf(stderr,"--bv must be 4 or 12\n"); return 2; }
    if(K%256){ std::fprintf(stderr,"K%%256!=0\n"); return 2; }

    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1; hipDeviceProp_t prop{};
    for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        if(std::strncmp(p.gcnArchName,arch.c_str(),arch.size())==0){ dev=d; prop=p; break; } }
    if(dev<0){ std::fprintf(stderr,"no device with gcnArchName %s*\n",arch.c_str()); return 2; }
    CK(hipSetDevice(dev));

    const int gpr=K/256, gb=v2_group_bytes(bits);
    const size_t a_bytes=(size_t)M*gpr*gb;
    std::printf("%s\narch=%s dev=%d (%s)\nbits=%d N=%d bv=%d M=%d K=%d  A=%zu bytes  samples=%d warmup=%d slabs=%d\n",
                kBenchId, arch.c_str(), dev, prop.name, bits, N, bv, M, K, a_bytes, samples, warmup, kBatchLaunches);

    std::vector<uint8_t> A; std::vector<uint16_t> Xh((size_t)N*K);
    pack_matrix(A,bits,M,K,42);
    { uint64_t rng=7; for(auto& x:Xh) x=f32_to_f16_bits(lcg_finite(&rng,-1.f,1.f)); }

    std::vector<void*> slabs(kBatchLaunches);
    for(int i=0;i<kBatchLaunches;i++){ CK(hipMalloc(&slabs[i],A.size())); CK(hipMemcpy(slabs[i],A.data(),A.size(),hipMemcpyHostToDevice)); }
    void* dX; CK(hipMalloc(&dX,Xh.size()*2)); CK(hipMemcpy(dX,Xh.data(),Xh.size()*2,hipMemcpyHostToDevice));
    void* dY; CK(hipMalloc(&dY,(size_t)N*M*4));

    using KFn = void(*)(const char*,const _Float16*,float*,int,int,int);
    auto pick = [&](int arm)->KFn{
        if(bits==2){
            if(bv==4) return arm==0?mb_payload_nt_mq2_bv4_a0:mb_payload_nt_mq2_bv4_a1;
            else return arm==0?mb_payload_nt_mq2_bv12_a0:mb_payload_nt_mq2_bv12_a1;
        } else if(bits==3){
            if(bv==4) return arm==0?mb_payload_nt_mq3_bv4_a0:mb_payload_nt_mq3_bv4_a1;
            else return arm==0?mb_payload_nt_mq3_bv12_a0:mb_payload_nt_mq3_bv12_a1;
        } else if(bits==5){
            if(bv==4) return arm==0?mb_payload_nt_mq5_bv4_a0:mb_payload_nt_mq5_bv4_a1;
            else return arm==0?mb_payload_nt_mq5_bv12_a0:mb_payload_nt_mq5_bv12_a1;
        } else {
            if(bv==4) return arm==0?mb_payload_nt_mq6_bv4_a0:mb_payload_nt_mq6_bv4_a1;
            else return arm==0?mb_payload_nt_mq6_bv12_a0:mb_payload_nt_mq6_bv12_a1;
        }
    };
    KFn k0=pick(0), k1=pick(1);
    struct Arm{ const char* name; KFn fn; };
    const Arm arms[2]={ {"A0_baseline",k0}, {"A1_nt", k1} };

    auto launch=[&](const Arm& a, void* s){
        dim3 grid((M+15)/16,(N+16*bv-1)/(16*bv)), block(32);
        hipLaunchKernelGGL(a.fn,grid,block,0,0,(const char*)s,(const _Float16*)dX,(float*)dY,M,K,N);
    };
    auto fetch=[&](std::vector<float>& out){
        out.resize((size_t)N*M);
        CK(hipMemcpy(out.data(),dY,out.size()*4,hipMemcpyDeviceToHost));
    };
    auto biteq=[&](const std::vector<float>& a,const std::vector<float>& b){
        if(a.size()!=b.size()) return false;
        for(size_t i=0;i<a.size();i++){ uint32_t x,y; std::memcpy(&x,&a[i],4); std::memcpy(&y,&b[i],4); if(x!=y) return false; }
        return true;
    };

    bool pass=true;
    std::vector<float> base, cand;
    launch(arms[0],slabs[0]); CK(hipDeviceSynchronize()); fetch(base);
    const double rr=sampled_ref_rel_l2(A,Xh,base,M,K,N,bits);
    const bool refok = rr<=5e-4;
    std::printf("HOSTREF A0 rel_l2=%.2e : %s\n", rr, refok?"PASS":"FAIL");
    pass &= refok;
    launch(arms[1],slabs[0]); CK(hipDeviceSynchronize()); fetch(cand);
    { const bool ok=biteq(cand,base); std::printf("BITEQ A1_nt vs A0 : %s\n", ok?"PASS":"FAIL"); pass &= ok; }

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    double med_us[2]={0,0};
    for(int ai=0; ai<2; ai++){
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
        std::printf("%-12s med_us=%9.2f  p10=%9.2f p90=%9.2f  wGiB/s=%7.2f  vsA0=%5.2fx\n",
                    arms[ai].name, med_us[ai], us[(size_t)(us.size()*0.10)], us[(size_t)(us.size()*0.90)],
                    gibs, med_us[0]/med_us[ai]);
    }
    std::printf("%s\n", pass?"OVERALL PASS":"OVERALL FAIL");
    return pass?0:1;
}
