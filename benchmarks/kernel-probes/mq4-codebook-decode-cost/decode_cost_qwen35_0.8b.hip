// Qwen3.5-0.8B decode-cost verdict: current MQ4G256 vs the candidate
// (symmetric + baked codebook LUT + g64 E4M3 sub-scales), at IDENTICAL 136 B/group.
//
// Reports the TIME-WEIGHTED total over the real per-token GEMV stack, which is
// the only figure that answers "does this cost decode speed". Per-shape
// percentages on 8 us kernels are launch-overhead noise.
//
// config: hidden=1024 inter=3584 layers=24 heads=8 kv=2 head_dim=256 vocab=248320
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#define HC(x) do{ hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
__constant__ float GCB[16];

__device__ __forceinline__ float e4m3_dec(unsigned int b){
    unsigned int e=(b>>3)&0xFu, m=b&0x7u;
    return __int_as_float((int)(((e+120u)<<23)|(m<<20)));
}

__launch_bounds__(32,16)
extern "C" __global__ void kCur(const char* __restrict__ A,const float* __restrict__ xr,
                                float* __restrict__ y,int M,int K){
    const int row=blockIdx.x; if(row>=M) return; const int tid=threadIdx.x;
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*136); float acc=0.f;
    for(int g=0;g<gpr;g++){
        const char* gp=rp+g*136;
        float sc=__builtin_bit_cast(float,*(const unsigned int*)(gp));
        float ze=__builtin_bit_cast(float,*(const unsigned int*)(gp+4));
        const unsigned char* nb=(const unsigned char*)(gp+8);
        int bi=g*256+tid*8,bo=tid*4;
        unsigned char b0=nb[bo],b1=nb[bo+1],b2=nb[bo+2],b3=nb[bo+3];
        acc+=(sc*(float)(b0&0xF)+ze)*xr[bi]  +(sc*(float)(b0>>4)+ze)*xr[bi+1]
            +(sc*(float)(b1&0xF)+ze)*xr[bi+2]+(sc*(float)(b1>>4)+ze)*xr[bi+3]
            +(sc*(float)(b2&0xF)+ze)*xr[bi+4]+(sc*(float)(b2>>4)+ze)*xr[bi+5]
            +(sc*(float)(b3&0xF)+ze)*xr[bi+6]+(sc*(float)(b3>>4)+ze)*xr[bi+7];
    }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(tid==0) y[row]=acc;
}

// Candidate: [f32 master][4 x u8 E4M3 g64 sub-scale][128 B nibbles] = 136 B.
// The codebook absorbs the scale, so the per-weight multiply collapses to one
// FMUL per 8 weights instead of a per-weight FMA.
__launch_bounds__(32,16)
extern "C" __global__ void kNew(const char* __restrict__ A,const float* __restrict__ xr,
                                float* __restrict__ y,int M,int K){
    __shared__ float lut[16];
    const int tid=threadIdx.x;
    if(tid<16) lut[tid]=GCB[tid];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();      // blockDim==32 -> single wave32, no s_barrier
    const int row=blockIdx.x; if(row>=M) return;
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*136);
    const int sub=tid>>3; float acc=0.f;
    for(int g=0;g<gpr;g++){
        const char* gp=rp+g*136;
        float mst=__builtin_bit_cast(float,*(const unsigned int*)(gp));
        float eff=mst*e4m3_dec((unsigned char)gp[4+sub]);
        const unsigned char* nb=(const unsigned char*)(gp+8);
        int bi=g*256+tid*8,bo=tid*4;
        unsigned char b0=nb[bo],b1=nb[bo+1],b2=nb[bo+2],b3=nb[bo+3];
        acc+=(lut[b0&0xF]*xr[bi]  +lut[b0>>4]*xr[bi+1]
             +lut[b1&0xF]*xr[bi+2]+lut[b1>>4]*xr[bi+3]
             +lut[b2&0xF]*xr[bi+4]+lut[b2>>4]*xr[bi+5]
             +lut[b3&0xF]*xr[bi+6]+lut[b3>>4]*xr[bi+7])*eff;
    }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(tid==0) y[row]=acc;
}

struct Op{ int m,k,per_token; const char* name; };
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(int argc,char**argv){
    int iters=(argc>1)?atoi(argv[1]):600;
    int reps =(argc>2)?atoi(argv[2]):5;
    hipDeviceProp_t p; HC(hipGetDeviceProperties(&p,0));
    printf("GPU: %s (%s) CUs=%d  iters/shape=%d reps=%d\n",p.name,p.gcnArchName,p.multiProcessorCount,iters,reps);
    printf("Model: Qwen3.5-0.8B  hidden=1024 inter=3584 layers=24 head_dim=256 h=8 kv=2\n\n");
    float cb[16]; for(int i=0;i<16;i++) cb[i]=-1.f+2.f*(i+0.5f)/16.f;
    HC(hipMemcpyToSymbol(HIP_SYMBOL(GCB),cb,sizeof(cb)));

    const int L=24;
    Op ops[]={
        {3072,1024,L,"fused_qkv  [3072,1024]"},   // q 8*256 + k 2*256 + v 2*256
        {1024,2048,L,"o_proj     [1024,2048]"},
        {7168,1024,L,"fused_gate_up [7168,1024]"},
        {1024,3584,L,"down_proj  [1024,3584]"},
        {248320,1024,1,"lm_head  [248320,1024]"},
    };

    printf("%-26s %6s %10s %10s %10s %9s\n","op","x/tok","cur ms","new ms","delta","GB/s");
    printf("%s\n",std::string(78,'-').c_str());
    double totCur=0,totNew=0,totCurNoLM=0,totNewNoLM=0;
    std::vector<double> allDelta;

    for(auto o:ops){
        size_t g=(size_t)o.k/256, bytes=(size_t)o.m*g*136;
        char* A; float *x,*y;
        HC(hipMalloc(&A,bytes)); HC(hipMalloc(&x,(size_t)o.k*4)); HC(hipMalloc(&y,(size_t)o.m*4));
        std::vector<unsigned char> h(bytes); unsigned int st=0xC0FFEEu;
        for(size_t i=0;i<bytes;i++){ st=st*1664525u+1013904223u; h[i]=(unsigned char)(st>>24); }
        for(size_t gi=0;gi<(size_t)o.m*g;gi++) for(int j=0;j<4;j++) h[gi*136+4+j]|=0x40;
        HC(hipMemcpy(A,h.data(),bytes,hipMemcpyHostToDevice));
        std::vector<float> hx(o.k);
        for(int i=0;i<o.k;i++){ st=st*1664525u+1013904223u; hx[i]=((int)(st>>8)/8388608.f)-1.f; }
        HC(hipMemcpy(x,hx.data(),(size_t)o.k*4,hipMemcpyHostToDevice));

        dim3 gr(o.m),bk(32);
        auto fc=[&]{hipLaunchKernelGGL(kCur,gr,bk,0,0,A,x,y,o.m,o.k);};
        auto fn=[&]{hipLaunchKernelGGL(kNew,gr,bk,0,0,A,x,y,o.m,o.k);};
        for(int i=0;i<80;i++){fc();fn();} HC(hipDeviceSynchronize());

        hipEvent_t a,b; HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
        std::vector<double> RC,RN;
        // Batch B back-to-back launches per timed region: amortizes the ~3-5 us
        // ROCm launch overhead that otherwise dominates a ~10 us kernel, and
        // matches how decode actually enqueues (a stream of kernels, not
        // one-launch-then-sync).
        const int B = 64;
        for(int r=0;r<reps;r++){
            std::vector<double> TC,TN;
            auto burst=[&](auto&&k,std::vector<double>&out){
                HC(hipEventRecord(a));
                for(int j=0;j<B;j++) k();
                HC(hipEventRecord(b)); HC(hipEventSynchronize(b));
                float ms; HC(hipEventElapsedTime(&ms,a,b)); out.push_back(ms/B);
            };
            for(int i=0;i<iters/B+1;i++){ burst(fc,TC); burst(fn,TN); }
            RC.push_back(med(TC)); RN.push_back(med(TN));
        }
        HC(hipEventDestroy(a)); HC(hipEventDestroy(b));
        double tc=med(RC), tn=med(RN);
        double d=(tn/tc-1)*100; allDelta.push_back(d);
        printf("%-26s %6d %10.4f %10.4f %+9.2f%% %9.1f\n",o.name,o.per_token,tc,tn,d,bytes/(tc*1e-3)/1e9);
        totCur+=tc*o.per_token; totNew+=tn*o.per_token;
        if(std::string(o.name).find("lm_head")==std::string::npos){ totCurNoLM+=tc*o.per_token; totNewNoLM+=tn*o.per_token; }
        HC(hipFree(A));HC(hipFree(x));HC(hipFree(y));
    }
    printf("%s\n",std::string(78,'-').c_str());
    printf("\nPER-TOKEN WEIGHTED TOTAL (the number that matters)\n");
    printf("  incl. MQ4 lm_head : cur %.4f ms   new %.4f ms   %+.3f%%\n",
           totCur,totNew,(totNew/totCur-1)*100);
    printf("  excl. lm_head     : cur %.4f ms   new %.4f ms   %+.3f%%\n",
           totCurNoLM,totNewNoLM,(totNewNoLM/totCurNoLM-1)*100);
    printf("  (lm_head is commonly kept at Q8, hence both figures)\n");
    printf("\n  worst single op   : %+.2f%%\n",*std::max_element(allDelta.begin(),allDelta.end()));
    printf("  BUDGET            : +1.00%%  ->  %s\n",
           (totNew/totCur-1)*100 <= 1.0 && (totNewNoLM/totCurNoLM-1)*100 <= 1.0 ? "PASS" : "FAIL");
    return 0;
}
