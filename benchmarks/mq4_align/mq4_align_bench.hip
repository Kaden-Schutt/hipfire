// SPDX-License-Identifier: Apache-2.0
// MQ4 (HFQ4-G256) weight-layout cacheline-alignment microbench.
//
// Layout A (CURRENT): 136B interleaved groups.
//   group @ (row*gpr+g)*136 : [0:4]=scale f32, [4:8]=zp f32, [8:136]=128B nibbles(256)
//   The warp's 32x4B nibble load spans [group_base+8, group_base+136) -> a 128B
//   span at a NON-128-aligned offset -> straddles two 128B lines.
// Layout B (ALIGNED): nibble blocks in their own array @ 128B stride (aligned),
//   scales+zp in separate arrays. The warp's nibble load is a fully aligned 128B
//   span -> one transaction.
//
// Same dequant math, same values (B derived from A). Measures wall time + a
// checksum for A/B parity. rocprofv3 supplies the L1/L2/DEP_WAIT counters.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
    printf("HIP err %s @ %d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} } while(0)

#define DOG8(sc,zp,pk,x0,x1,x2,x3,x4,x5,x6,x7) ( \
      (sc*(float)((pk)&0xFu)         + zp)*x0 \
    + (sc*(float)(((pk)>>4)&0xFu)    + zp)*x1 \
    + (sc*(float)(((pk)>>8)&0xFu)    + zp)*x2 \
    + (sc*(float)(((pk)>>12)&0xFu)   + zp)*x3 \
    + (sc*(float)(((pk)>>16)&0xFu)   + zp)*x4 \
    + (sc*(float)(((pk)>>20)&0xFu)   + zp)*x5 \
    + (sc*(float)(((pk)>>24)&0xFu)   + zp)*x6 \
    + (sc*(float)(((pk)>>28)&0xFu)   + zp)*x7)

// Layout A: interleaved 136B groups (current hipfire MQ4).
__global__ void gemvA(const char* __restrict__ W, const float* __restrict__ x,
                      float* __restrict__ y, int gpr, int M, int rpb) {
    const int tid = threadIdx.x;
    const int boff = tid * 4;
    const int row0 = blockIdx.x * rpb;
    for (int r = 0; r < rpb; r++) {
        const int row = row0 + r;
        if (row >= M) break;
        const char* base = W + (long long)row * gpr * 136;
        float acc = 0.0f;
        for (int g = 0; g < gpr; g++) {
            const char* gp = base + (long long)g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            const int b = g * 256 + tid * 8;
            float x0=x[b],x1=x[b+1],x2=x[b+2],x3=x[b+3],x4=x[b+4],x5=x[b+5],x6=x[b+6],x7=x[b+7];
            acc += DOG8(sc,zp,pk,x0,x1,x2,x3,x4,x5,x6,x7);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
        if (tid == 0) y[row] = acc;
    }
}

// Layout B: 128B-aligned nibble blocks + separate scale/zp arrays.
__global__ void gemvB(const char* __restrict__ nib, const float* __restrict__ sc_arr,
                      const float* __restrict__ zp_arr, const float* __restrict__ x,
                      float* __restrict__ y, int gpr, int M, int rpb) {
    const int tid = threadIdx.x;
    const int boff = tid * 4;
    const int row0 = blockIdx.x * rpb;
    for (int r = 0; r < rpb; r++) {
        const int row = row0 + r;
        if (row >= M) break;
        float acc = 0.0f;
        for (int g = 0; g < gpr; g++) {
            const int gi = row * gpr + g;
            float sc = sc_arr[gi];
            float zp = zp_arr[gi];
            const char* nb = nib + (long long)gi * 128;   // aligned
            unsigned int pk = *(const unsigned int*)(nb + boff);
            const int b = g * 256 + tid * 8;
            float x0=x[b],x1=x[b+1],x2=x[b+2],x3=x[b+3],x4=x[b+4],x5=x[b+5],x6=x[b+6],x7=x[b+7];
            acc += DOG8(sc,zp,pk,x0,x1,x2,x3,x4,x5,x6,x7);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
        if (tid == 0) y[row] = acc;
    }
}

int main(int argc, char** argv) {
    int M = 16384, K = 4096, rpb = 1, iters = 200, warm = 30, runs = 5;
    int which = 2; // 0=A only, 1=B only, 2=both
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-M")) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-K")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-rpb")) rpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-warm")) warm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-runs")) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-which")) which = atoi(argv[++i]);
    }
    const int gpr = K / 256;
    const long long ngroups = (long long)M * gpr;
    const long long bytesA = ngroups * 136;
    const long long bytesB = ngroups * 128 + ngroups * 8; // nibbles + (sc+zp)
    printf("M=%d K=%d gpr=%d rpb=%d iters=%d runs=%d | A=%.2fMB B=%.2fMB (L2=8MB)\n",
           M, K, gpr, rpb, iters, runs, bytesA/1048576.0, bytesB/1048576.0);

    // Host build of layout A.
    std::vector<char> hA(bytesA);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<unsigned> nd(0, 0xFFFFFFFFu);
    std::uniform_real_distribution<float> sd(0.001f, 0.05f), zd(-0.3f, 0.3f);
    std::vector<float> hsc(ngroups), hzp(ngroups);
    std::vector<char> hnib((size_t)ngroups * 128);
    for (long long gi = 0; gi < ngroups; gi++) {
        char* gp = hA.data() + gi * 136;
        float sc = sd(rng), zp = zd(rng);
        memcpy(gp, &sc, 4); memcpy(gp + 4, &zp, 4);
        hsc[gi] = sc; hzp[gi] = zp;
        for (int b = 0; b < 32; b++) {
            unsigned pk = nd(rng);
            memcpy(gp + 8 + b * 4, &pk, 4);
            memcpy(hnib.data() + gi * 128 + b * 4, &pk, 4);
        }
    }
    std::vector<float> hx(K);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    for (int i = 0; i < K; i++) hx[i] = xd(rng);

    char *dA, *dnib; float *dsc, *dzp, *dx, *dyA, *dyB;
    HIP_CHECK(hipMalloc(&dA, bytesA));
    HIP_CHECK(hipMalloc(&dnib, (size_t)ngroups * 128));
    HIP_CHECK(hipMalloc(&dsc, ngroups * sizeof(float)));
    HIP_CHECK(hipMalloc(&dzp, ngroups * sizeof(float)));
    HIP_CHECK(hipMalloc(&dx, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&dyA, M * sizeof(float)));
    HIP_CHECK(hipMalloc(&dyB, M * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dnib, hnib.data(), (size_t)ngroups * 128, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsc, hsc.data(), ngroups * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dzp, hzp.data(), ngroups * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), K * sizeof(float), hipMemcpyHostToDevice));

    const int nblk = (M + rpb - 1) / rpb;
    dim3 grid(nblk), block(32);

    // Alignment sanity: report the dA base and offsets mod 128.
    printf("dA base %% 128 = %ld ; layout-A nibble off(group0) = (base+8) %% 128 = %ld\n",
           (long)((uintptr_t)dA % 128), (long)(((uintptr_t)dA + 8) % 128));
    printf("dnib base %% 128 = %ld (layout-B nibble blocks must be 0)\n",
           (long)((uintptr_t)dnib % 128));

    auto runK = [&](int mode)->double {
        hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
        for (int i = 0; i < warm; i++) {
            if (mode==0) gemvA<<<grid,block>>>(dA,dx,dyA,gpr,M,rpb);
            else         gemvB<<<grid,block>>>(dnib,dsc,dzp,dx,dyB,gpr,M,rpb);
        }
        HIP_CHECK(hipDeviceSynchronize());
        double best = 1e30;
        for (int run = 0; run < runs; run++) {
            HIP_CHECK(hipEventRecord(t0));
            for (int i = 0; i < iters; i++) {
                if (mode==0) gemvA<<<grid,block>>>(dA,dx,dyA,gpr,M,rpb);
                else         gemvB<<<grid,block>>>(dnib,dsc,dzp,dx,dyB,gpr,M,rpb);
            }
            HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
            float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,t0,t1));
            double us = (double)ms*1000.0/iters;
            if (us < best) best = us;
        }
        return best; // us/iter, best-of-runs
    };

    double tA=0, tB=0;
    if (which==0 || which==2) tA = runK(0);
    if (which==1 || which==2) tB = runK(1);

    // Parity check (only if both ran).
    if (which==2) {
        std::vector<float> yA(M), yB(M);
        HIP_CHECK(hipMemcpy(yA.data(), dyA, M*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(yB.data(), dyB, M*sizeof(float), hipMemcpyDeviceToHost));
        double maxabs=0; int nbad=0;
        for (int i=0;i<M;i++){ double d=fabs((double)yA[i]-(double)yB[i]); if(d>maxabs)maxabs=d; if(d>1e-3)nbad++; }
        printf("PARITY: max|A-B|=%.3e  nbad(>1e-3)=%d  y[0]=%.5f\n", maxabs, nbad, yA[0]);
    }

    double effGB_A = (double)bytesA * iters / (tA*1e-6*iters) / 1e9; // = bytesA/tA_s
    if (which==0||which==2) printf("A (136B interleaved): %.3f us/iter  %.1f GB/s\n", tA, bytesA/(tA*1e-6)/1e9);
    if (which==1||which==2) printf("B (128B aligned)    : %.3f us/iter  %.1f GB/s\n", tB, bytesB/(tB*1e-6)/1e9);
    if (which==2) printf("DELTA: B vs A = %+.2f%% (%s)\n", (tA-tB)/tA*100.0, tB<tA?"B faster":"A faster");
    (void)effGB_A;
    return 0;
}
