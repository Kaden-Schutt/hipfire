// SPDX-License-Identifier: Apache-2.0
// Faithful MoE gate_up microbench: E scattered expert weight regions, grid=(mi,E),
// each block reads expert[blockIdx.y]'s base — mirrors gemv_hfq4g256_moe_gate_up_indexed.
// Layout A = 136B interleaved (current); Layout B = 128B-aligned nibbles + separate scales.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
    printf("HIP err %s @ %d: %s\n",#x,__LINE__,hipGetErrorString(e)); exit(1);} } while(0)

#define DOG8(sc,zp,pk,x0,x1,x2,x3,x4,x5,x6,x7) ( \
      (sc*(float)((pk)&0xFu)+zp)*x0 + (sc*(float)(((pk)>>4)&0xFu)+zp)*x1 \
    + (sc*(float)(((pk)>>8)&0xFu)+zp)*x2 + (sc*(float)(((pk)>>12)&0xFu)+zp)*x3 \
    + (sc*(float)(((pk)>>16)&0xFu)+zp)*x4 + (sc*(float)(((pk)>>20)&0xFu)+zp)*x5 \
    + (sc*(float)(((pk)>>24)&0xFu)+zp)*x6 + (sc*(float)(((pk)>>28)&0xFu)+zp)*x7)

// A: expert bases point to 136B-interleaved matrices.
__global__ void moeA(const unsigned long long* __restrict__ bases, const float* __restrict__ x,
                     float* __restrict__ y, int gpr, int M) {
    const int row = blockIdx.x; const int e = blockIdx.y; const int tid = threadIdx.x;
    const char* base = reinterpret_cast<const char*>(bases[e]);
    const char* rp = base + (long long)row * gpr * 136;
    const int boff = tid * 4; float acc = 0.0f;
    for (int g = 0; g < gpr; g++) {
        const char* gp = rp + (long long)g * 136;
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
        float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        const int b = g * 256 + tid * 8;
        float x0=x[b],x1=x[b+1],x2=x[b+2],x3=x[b+3],x4=x[b+4],x5=x[b+5],x6=x[b+6],x7=x[b+7];
        acc += DOG8(sc,zp,pk,x0,x1,x2,x3,x4,x5,x6,x7);
    }
    for (int o=16;o>0;o>>=1) acc += __shfl_down(acc,o);
    if (tid==0) y[e*M + row] = acc;
}
// B: expert bases point to 128B-aligned nibble matrices; separate per-expert scale/zp arrays.
__global__ void moeB(const unsigned long long* __restrict__ nbases, const unsigned long long* __restrict__ sbases,
                     const float* __restrict__ x, float* __restrict__ y, int gpr, int M) {
    const int row = blockIdx.x; const int e = blockIdx.y; const int tid = threadIdx.x;
    const char* nb0 = reinterpret_cast<const char*>(nbases[e]);
    const float* sc0 = reinterpret_cast<const float*>(sbases[e]); // [rows*gpr*2] : sc,zp interleaved per group
    const int boff = tid * 4; float acc = 0.0f;
    for (int g = 0; g < gpr; g++) {
        const int gi = row * gpr + g;
        float sc = sc0[gi*2]; float zp = sc0[gi*2+1];
        const char* nb = nb0 + (long long)gi * 128;
        unsigned int pk = *(const unsigned int*)(nb + boff);
        const int b = g * 256 + tid * 8;
        float x0=x[b],x1=x[b+1],x2=x[b+2],x3=x[b+3],x4=x[b+4],x5=x[b+5],x6=x[b+6],x7=x[b+7];
        acc += DOG8(sc,zp,pk,x0,x1,x2,x3,x4,x5,x6,x7);
    }
    for (int o=16;o>0;o>>=1) acc += __shfl_down(acc,o);
    if (tid==0) y[e*M + row] = acc;
}

int main(int argc, char** argv) {
    int M = 1536, K = 2048, E = 8, iters = 300, warm = 30, runs = 5;
    for (int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-M"))M=atoi(argv[++i]); else if(!strcmp(argv[i],"-K"))K=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-E"))E=atoi(argv[++i]); else if(!strcmp(argv[i],"-iters"))iters=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-runs"))runs=atoi(argv[++i]);
    }
    const int gpr = K/256; const long long ngr = (long long)M*gpr;
    const long long bytesA = ngr*136, bytesB = ngr*128, bytesS = ngr*2*sizeof(float);
    printf("MoE: E=%d M=%d K=%d gpr=%d | per-expert A=%.2fMB total=%.2fMB (L2=8MB)\n",
           E, M, K, gpr, bytesA/1048576.0, E*bytesA/1048576.0);
    std::mt19937 rng(7); std::uniform_int_distribution<unsigned> nd(0,0xFFFFFFFFu);
    std::uniform_real_distribution<float> sd(0.001f,0.05f), zd(-0.3f,0.3f), xd(-1.f,1.f);

    std::vector<char*> dA(E), dnib(E); std::vector<float*> dsc(E);
    std::vector<unsigned long long> hbA(E), hbN(E), hbS(E);
    std::vector<char> hA(bytesA), hN(bytesB); std::vector<float> hS(ngr*2);
    for (int e=0;e<E;e++){
        for(long long gi=0;gi<ngr;gi++){
            char* gp=hA.data()+gi*136; float sc=sd(rng),zp=zd(rng);
            memcpy(gp,&sc,4);memcpy(gp+4,&zp,4); hS[gi*2]=sc;hS[gi*2+1]=zp;
            for(int b=0;b<32;b++){unsigned pk=nd(rng);memcpy(gp+8+b*4,&pk,4);memcpy(hN.data()+gi*128+b*4,&pk,4);}
        }
        HIP_CHECK(hipMalloc(&dA[e],bytesA)); HIP_CHECK(hipMalloc(&dnib[e],bytesB)); HIP_CHECK(hipMalloc(&dsc[e],bytesS));
        HIP_CHECK(hipMemcpy(dA[e],hA.data(),bytesA,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dnib[e],hN.data(),bytesB,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dsc[e],hS.data(),bytesS,hipMemcpyHostToDevice));
        hbA[e]=(unsigned long long)dA[e]; hbN[e]=(unsigned long long)dnib[e]; hbS[e]=(unsigned long long)dsc[e];
    }
    unsigned long long *dbA,*dbN,*dbS; HIP_CHECK(hipMalloc(&dbA,E*8));HIP_CHECK(hipMalloc(&dbN,E*8));HIP_CHECK(hipMalloc(&dbS,E*8));
    HIP_CHECK(hipMemcpy(dbA,hbA.data(),E*8,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbN,hbN.data(),E*8,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbS,hbS.data(),E*8,hipMemcpyHostToDevice));
    std::vector<float> hx(K); for(int i=0;i<K;i++)hx[i]=xd(rng);
    float *dx,*dyA,*dyB; HIP_CHECK(hipMalloc(&dx,K*4));HIP_CHECK(hipMalloc(&dyA,E*M*4));HIP_CHECK(hipMalloc(&dyB,E*M*4));
    HIP_CHECK(hipMemcpy(dx,hx.data(),K*4,hipMemcpyHostToDevice));
    dim3 grid(M,E), block(32);
    auto run=[&](int mode)->double{
        hipEvent_t t0,t1;HIP_CHECK(hipEventCreate(&t0));HIP_CHECK(hipEventCreate(&t1));
        for(int i=0;i<warm;i++){ if(mode==0)moeA<<<grid,block>>>(dbA,dx,dyA,gpr,M); else moeB<<<grid,block>>>(dbN,dbS,dx,dyB,gpr,M);}
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize()); double best=1e30;
        for(int r=0;r<runs;r++){HIP_CHECK(hipEventRecord(t0));
            for(int i=0;i<iters;i++){ if(mode==0)moeA<<<grid,block>>>(dbA,dx,dyA,gpr,M); else moeB<<<grid,block>>>(dbN,dbS,dx,dyB,gpr,M);}
            HIP_CHECK(hipEventRecord(t1));HIP_CHECK(hipEventSynchronize(t1));float ms;HIP_CHECK(hipEventElapsedTime(&ms,t0,t1));
            double us=ms*1000.0/iters; if(us<best)best=us;}
        return best;};
    double tA=run(0), tB=run(1);
    std::vector<float> yA(E*M),yB(E*M);
    HIP_CHECK(hipMemcpy(yA.data(),dyA,E*M*4,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(yB.data(),dyB,E*M*4,hipMemcpyDeviceToHost));
    double maxabs=0; for(size_t i=0;i<yA.size();i++){double d=fabs((double)yA[i]-(double)yB[i]);if(d>maxabs)maxabs=d;}
    printf("PARITY max|A-B|=%.3e\n",maxabs);
    printf("A (136B interleaved): %.3f us/iter\n", tA);
    printf("B (128B aligned)    : %.3f us/iter\n", tB);
    printf("DELTA B vs A = %+.2f%% (%s)\n",(tA-tB)/tA*100.0, tB<tA?"B faster":"A faster");
    return 0;
}
