// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mb-lds-pad18-gfx1151-v1-20260830
//
// QUESTION: the MW-LDS family stages weights as [16][16][16] fp16
// (row stride 32 B = 8 dwords), so compute-side row reads collide 4-way on
// LDS banks (ml 0/4/8/12 -> same bank). Does padding the innermost dim to 18
// halfs (36 B = 9 dwords, gcd(9,32)=1, conflict-free) actually help — and
// does a wide 16-byte access on a 36 B-strided (4-byte-aligned) row even
// function correctly, or must a padded layout use dword access?
//
// Production construct interrogated (LDS layout + access shape only):
//   gemm_{gate_up,*_residual}_mq{3,4,5,6}g256v2_wmma_gfx11_mw{4,8}_lds
//   (kernels/src/gemm_mqv2_wmma_gfx11_mw_lds.hip)
//
// This is a pure LDS exerciser — no global memory traffic in the timed loop,
// no WMMA — modelling exactly the MW staging-write + compute-read pattern in
// dword units (LDS banks are dword-granular; fp16 payload adds nothing).
// Arms:
//   P0: stride 8 dwords, wide b128 access (production layout)
//   P1: stride 9 dwords, wide b128 access (the review's literal ask; rows
//       only 4-byte-aligned — alignment-suspect)
//   P2: stride 9 dwords, dword access (what a padded production kernel
//       would have to ship)
// Values staged are a deterministic f(iter,kt,row,ki), XOR-accumulated by
// every thread and written out; all arms MUST produce identical checksums.
//
// Build (fat: ROCm 7.15 rejects single-arch gfx1151 images):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 --offload-arch=gfx1151 -O3 \
//       tools/kernels/gfx1151/mb_lds_pad18.hip -o mb_lds_pad18
// Run:
//   ./mb_lds_pad18 --arch gfx1151 [--nw 4|8] [--iters I] [--wgs G] [--samples S]
//   Defaults: nw=4 iters=2048 wgs=256 samples=30 warmup=10

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr const char* kBenchId = "mb-lds-pad18-gfx1151-v1-20260830";

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ std::fprintf(stderr,"HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} }while(0)

#if defined(__gfx1100__) || defined(__gfx1151__)

typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ unsigned int lds_val(unsigned int it, int kt, int row, int ki){
    return (it * 2654435761u) ^ (unsigned int)(kt * 40503 + row * 65599 + ki * 2971);
}

// PADU: dwords per LDS row (8 = production, 9 = padded). WIDE: b128 vs b32.
template <int PADU, bool WIDE, int NW>
__global__ __launch_bounds__(32 * NW, 1) void mb_lds_probe(unsigned int* __restrict__ out, int iters){
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int ml = lane & 15;
    __shared__ __align__(32) unsigned int w[16][16][PADU];
    unsigned int acc = 0;
    for (int it = 0; it < iters; ++it) {
        // Staging: same 128-job pattern as MW (job -> row, segment -> kt pair).
        for (int job = tid; job < 128; job += 32 * NW) {
            const int row = job / 8;
            const int segment = job % 8;
            const int kt0 = 2 * segment, kt1 = kt0 + 1;
            if (WIDE) {
                uint4_t a, b;
                #pragma unroll
                for (int i = 0; i < 4; ++i) { a[i] = lds_val((unsigned)it, kt0, row, i); b[i] = lds_val((unsigned)it, kt0, row, 4 + i); }
                *(uint4_t*)(&w[kt0][row][0]) = a;
                *(uint4_t*)(&w[kt0][row][4]) = b;
                #pragma unroll
                for (int i = 0; i < 4; ++i) { a[i] = lds_val((unsigned)it, kt1, row, i); b[i] = lds_val((unsigned)it, kt1, row, 4 + i); }
                *(uint4_t*)(&w[kt1][row][0]) = a;
                *(uint4_t*)(&w[kt1][row][4]) = b;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) w[kt0][row][i] = lds_val((unsigned)it, kt0, row, i);
                #pragma unroll
                for (int i = 0; i < 8; ++i) w[kt1][row][i] = lds_val((unsigned)it, kt1, row, i);
            }
        }
        __syncthreads();
        // Compute-side reads: every lane reads row [kt][ml] for all 16 kt —
        // the exact MW a_reg load pattern, in dword units.
        #pragma unroll
        for (int kt = 0; kt < 16; ++kt) {
            if (WIDE) {
                const uint4_t a = *(const uint4_t*)(&w[kt][ml][0]);
                const uint4_t b = *(const uint4_t*)(&w[kt][ml][4]);
                acc ^= a[0] ^ a[1] ^ a[2] ^ a[3] ^ b[0] ^ b[1] ^ b[2] ^ b[3];
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc ^= w[kt][ml][i];
            }
        }
        __syncthreads();
    }
    out[(size_t)blockIdx.x * blockDim.x + tid] = acc;
}

template __global__ void mb_lds_probe<8, true, 4>(unsigned int*, int);
template __global__ void mb_lds_probe<9, true, 4>(unsigned int*, int);
template __global__ void mb_lds_probe<9, false, 4>(unsigned int*, int);
template __global__ void mb_lds_probe<8, true, 8>(unsigned int*, int);
template __global__ void mb_lds_probe<9, true, 8>(unsigned int*, int);
template __global__ void mb_lds_probe<9, false, 8>(unsigned int*, int);

#else // host pass / other arches
template <int PADU, bool WIDE, int NW>
__global__ void mb_lds_probe(unsigned int*, int) {}
template __global__ void mb_lds_probe<8, true, 4>(unsigned int*, int);
template __global__ void mb_lds_probe<9, true, 4>(unsigned int*, int);
template __global__ void mb_lds_probe<9, false, 4>(unsigned int*, int);
template __global__ void mb_lds_probe<8, true, 8>(unsigned int*, int);
template __global__ void mb_lds_probe<9, true, 8>(unsigned int*, int);
template __global__ void mb_lds_probe<9, false, 8>(unsigned int*, int);
#endif

int main(int argc, char** argv){
    std::string arch; int nw=4, iters=2048, wgs=256, samples=30, warmup=10;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        auto next=[&]{ if(i+1>=argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} return std::atoi(argv[++i]); };
        if(a=="--arch"&&i+1<argc) arch=argv[++i];
        else if(a=="--nw") nw=next();
        else if(a=="--iters") iters=next();
        else if(a=="--wgs") wgs=next();
        else if(a=="--samples") samples=next();
        else if(a=="--warmup") warmup=next();
        else { std::printf("Usage: %s --arch gfxNNNN [--nw 4|8] [--iters I] [--wgs G] [--samples S] [--warmup W]\nIdentity: %s\n", argv[0], kBenchId); return a=="--help"||a=="-h" ? 0 : 2; }
    }
    if(arch.empty()){ std::fprintf(stderr,"--arch required\n"); return 2; }
    if(arch!="gfx1100"&&arch!="gfx1151"){ std::printf("SKIP: %s is gfx11-only (requested %s)\n", kBenchId, arch.c_str()); return 0; }
    if(nw!=4&&nw!=8){ std::fprintf(stderr,"--nw must be 4 or 8\n"); return 2; }

    int ndev=0; CK(hipGetDeviceCount(&ndev));
    int dev=-1; hipDeviceProp_t prop{};
    for(int d=0; d<ndev; d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        if(std::strncmp(p.gcnArchName,arch.c_str(),arch.size())==0){ dev=d; prop=p; break; } }
    if(dev<0){ std::fprintf(stderr,"no device with gcnArchName %s*\n",arch.c_str()); return 2; }
    CK(hipSetDevice(dev));

    const int block=32*nw;
    const size_t nout=(size_t)wgs*block;
    std::printf("%s\narch=%s dev=%d (%s)  nw=%d iters=%d wgs=%d samples=%d\n", kBenchId, arch.c_str(), dev, prop.name, nw, iters, wgs, samples);

    unsigned int* dout; CK(hipMalloc(&dout, nout*4));
    struct Arm{ const char* name; void(*fn)(unsigned int*,int); };
    const Arm arms4[3]={ {"P0_pad16_wide", mb_lds_probe<8,true,4>}, {"P1_pad18_wide", mb_lds_probe<9,true,4>}, {"P2_pad18_dword", mb_lds_probe<9,false,4>} };
    const Arm arms8[3]={ {"P0_pad16_wide", mb_lds_probe<8,true,8>}, {"P1_pad18_wide", mb_lds_probe<9,true,8>}, {"P2_pad18_dword", mb_lds_probe<9,false,8>} };
    const Arm* arms = (nw==4)?arms4:arms8;

    // Correctness: checksums identical across arms.
    std::vector<unsigned int> base(nout), got(nout);
    bool pass=true;
    for(int ai=0; ai<3; ai++){
        CK(hipMemset(dout,0,nout*4));
        hipLaunchKernelGGL(arms[ai].fn, dim3(wgs), dim3(block), 0, 0, dout, iters);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(ai? got.data():base.data(), dout, nout*4, hipMemcpyDeviceToHost));
        if(ai){
            const bool ok = std::memcmp(got.data(), base.data(), nout*4)==0;
            std::printf("CHECKSUM %s vs P0 : %s\n", arms[ai].name, ok?"PASS":"FAIL");
            pass &= ok;
        }
    }

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    double med[3]={0,0,0};
    for(int ai=0; ai<3; ai++){
        for(int w=0;w<warmup;w++) hipLaunchKernelGGL(arms[ai].fn, dim3(wgs), dim3(block), 0, 0, dout, iters);
        CK(hipDeviceSynchronize());
        std::vector<float> us; us.reserve(samples);
        for(int s=0;s<samples;s++){
            CK(hipEventRecord(ev0,0));
            hipLaunchKernelGGL(arms[ai].fn, dim3(wgs), dim3(block), 0, 0, dout, iters);
            CK(hipEventRecord(ev1,0));
            CK(hipEventSynchronize(ev1));
            float ms=0; CK(hipEventElapsedTime(&ms,ev0,ev1));
            us.push_back(ms*1000.f);
        }
        std::sort(us.begin(),us.end());
        med[ai]=us[us.size()/2];
        std::printf("%-15s med_us=%9.2f  p10=%9.2f p90=%9.2f  ns/iter/wg=%7.2f  vsP0=%5.2fx\n",
                    arms[ai].name, med[ai], us[(size_t)(us.size()*0.10)], us[(size_t)(us.size()*0.90)],
                    med[ai]*1000.0/(double)iters, med[0]/med[ai]);
    }
    std::printf("%s\n", pass?"OVERALL PASS":"OVERALL FAIL");
    return pass?0:1;
}
