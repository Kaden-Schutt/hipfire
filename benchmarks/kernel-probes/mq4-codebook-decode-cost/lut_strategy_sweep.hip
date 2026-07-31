// Round 2: A=current baseline vs three LUT strategies for the codebook variant.
// All read exactly 136 B/group -> identical memory traffic; delta is pure ALU/LDS.
// Variants are interleaved per round so DPM drift cannot alias onto one kernel.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#define HC(x) do{ hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)

__constant__ float GCB[16];

__device__ __forceinline__ float e4m3_dec(unsigned int b) {
    unsigned int e = (b >> 3) & 0xFu, m = b & 0x7u;
    return __int_as_float((int)(((e + 120u) << 23) | (m << 20)));
}

// ---- A: current production kernel, verbatim -------------------------------
__launch_bounds__(32, 16)
extern "C" __global__ void kA(const char* __restrict__ A, const float* __restrict__ xr,
                              float* __restrict__ y, int M, int K) {
    const int row=blockIdx.x; if(row>=M) return; const int tid=threadIdx.x;
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*136);
    float acc=0.f;
    for(int g=0;g<gpr;g++){
        const char* gp=rp+g*136;
        float sc=__builtin_bit_cast(float,*(const unsigned int*)(gp));
        float ze=__builtin_bit_cast(float,*(const unsigned int*)(gp+4));
        const unsigned char* nb=(const unsigned char*)(gp+8);
        int bi=g*256+tid*8, bo=tid*4;
        unsigned char b0=nb[bo],b1=nb[bo+1],b2=nb[bo+2],b3=nb[bo+3];
        acc+=(sc*(float)(b0&0xF)+ze)*xr[bi]  +(sc*(float)(b0>>4)+ze)*xr[bi+1]
            +(sc*(float)(b1&0xF)+ze)*xr[bi+2]+(sc*(float)(b1>>4)+ze)*xr[bi+3]
            +(sc*(float)(b2&0xF)+ze)*xr[bi+4]+(sc*(float)(b2>>4)+ze)*xr[bi+5]
            +(sc*(float)(b3&0xF)+ze)*xr[bi+6]+(sc*(float)(b3>>4)+ze)*xr[bi+7];
    }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(tid==0) y[row]=acc;
}

#define CB_BODY(LOOKUP)                                                        \
    const int row=blockIdx.x; if(row>=M) return;                               \
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*136);            \
    const int sub=tid>>3; float acc=0.f;                                       \
    for(int g=0;g<gpr;g++){                                                    \
        const char* gp=rp+g*136;                                               \
        float mst=__builtin_bit_cast(float,*(const unsigned int*)(gp));        \
        float eff=mst*e4m3_dec((unsigned char)gp[4+sub]);                      \
        const unsigned char* nb=(const unsigned char*)(gp+8);                  \
        int bi=g*256+tid*8, bo=tid*4;                                          \
        unsigned char b0=nb[bo],b1=nb[bo+1],b2=nb[bo+2],b3=nb[bo+3];           \
        acc+=(LOOKUP(b0&0xF)*xr[bi]  +LOOKUP(b0>>4)*xr[bi+1]                   \
             +LOOKUP(b1&0xF)*xr[bi+2]+LOOKUP(b1>>4)*xr[bi+3]                   \
             +LOOKUP(b2&0xF)*xr[bi+4]+LOOKUP(b2>>4)*xr[bi+5]                   \
             +LOOKUP(b3&0xF)*xr[bi+6]+LOOKUP(b3>>4)*xr[bi+7])*eff;             \
    }                                                                          \
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);                           \
    if(tid==0) y[row]=acc;

// ---- C: LDS LUT + __syncthreads (round-1 candidate) -----------------------
__launch_bounds__(32, 16)
extern "C" __global__ void kC_lds_sync(const char* __restrict__ A, const float* __restrict__ xr,
                                       float* __restrict__ y, int M, int K) {
    __shared__ float lut[16];
    const int tid=threadIdx.x;
    if(tid<16) lut[tid]=GCB[tid];
    __syncthreads();
#define L_LDS(i) lut[i]
    CB_BODY(L_LDS)
#undef L_LDS
}

// ---- E: LDS LUT, wave_barrier only (blockDim==32 is a single wave32, so a
//         full s_barrier is unnecessary; only the lgkmcnt wait is required) --
__launch_bounds__(32, 16)
extern "C" __global__ void kE_lds_wave(const char* __restrict__ A, const float* __restrict__ xr,
                                       float* __restrict__ y, int M, int K) {
    __shared__ float lut[16];
    const int tid=threadIdx.x;
    if(tid<16) lut[tid]=GCB[tid];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
#define L_LDSW(i) lut[i]
    CB_BODY(L_LDSW)
#undef L_LDSW
}

// ---- F: no LDS at all — index __constant__ memory directly ----------------
__launch_bounds__(32, 16)
extern "C" __global__ void kF_const(const char* __restrict__ A, const float* __restrict__ xr,
                                    float* __restrict__ y, int M, int K) {
    const int tid=threadIdx.x;
#define L_CONST(i) GCB[i]
    CB_BODY(L_CONST)
#undef L_CONST
}

static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
struct Shape{ int m,k; const char* name; };

int main(int argc,char**argv){
    int iters=(argc>1)?atoi(argv[1]):500;
    hipDeviceProp_t p; HC(hipGetDeviceProperties(&p,0));
    printf("GPU: %s (%s) CUs=%d iters=%d\n\n",p.name,p.gcnArchName,p.multiProcessorCount,iters);
    float cb[16]; for(int i=0;i<16;i++) cb[i]=-1.f+2.f*(i+0.5f)/16.f;
    HC(hipMemcpyToSymbol(HIP_SYMBOL(GCB),cb,sizeof(cb)));

    Shape shapes[]={
        {1536,1024,"0.8B q/o_proj  M=1536  K=1024"},
        {4096,1024,"0.8B gate_up   M=4096  K=1024"},
        {1024,4096,"0.8B down_proj M=1024  K=4096"},
        {2048,2048,"0.8B square    M=2048  K=2048"},
        {8192,2048,"mid dense      M=8192  K=2048"},
        {32768,2048,"large         M=32768 K=2048"},
        {65536,4096,"VRAM-bound    M=65536 K=4096"},
    };
    printf("%-31s %9s %10s %10s %10s %9s\n","shape","A ms","C lds+sync","E lds+wave","F const","A GB/s");
    printf("%s\n",std::string(85,'-').c_str());
    double wC=0,wE=0,wF=0; int n=0; double mxC=-9e9,mxE=-9e9,mxF=-9e9;

    for(auto s:shapes){
        size_t g=(size_t)s.k/256, bytes=(size_t)s.m*g*136;
        char* A; float *x,*y;
        HC(hipMalloc(&A,bytes)); HC(hipMalloc(&x,(size_t)s.k*4)); HC(hipMalloc(&y,(size_t)s.m*4));
        std::vector<unsigned char> h(bytes); unsigned int st=0x12345678u;
        for(size_t i=0;i<bytes;i++){ st=st*1664525u+1013904223u; h[i]=(unsigned char)(st>>24); }
        for(size_t gi=0;gi<(size_t)s.m*g;gi++) for(int j=0;j<4;j++) h[gi*136+4+j]|=0x40;
        HC(hipMemcpy(A,h.data(),bytes,hipMemcpyHostToDevice));
        std::vector<float> hx(s.k);
        for(int i=0;i<s.k;i++){ st=st*1664525u+1013904223u; hx[i]=((int)(st>>8)/8388608.f)-1.f; }
        HC(hipMemcpy(x,hx.data(),(size_t)s.k*4,hipMemcpyHostToDevice));

        dim3 gr(s.m),bk(32);
        auto fA=[&]{hipLaunchKernelGGL(kA,gr,bk,0,0,A,x,y,s.m,s.k);};
        auto fC=[&]{hipLaunchKernelGGL(kC_lds_sync,gr,bk,0,0,A,x,y,s.m,s.k);};
        auto fE=[&]{hipLaunchKernelGGL(kE_lds_wave,gr,bk,0,0,A,x,y,s.m,s.k);};
        auto fF=[&]{hipLaunchKernelGGL(kF_const,gr,bk,0,0,A,x,y,s.m,s.k);};
        for(int i=0;i<60;i++){fA();fC();fE();fF();}
        HC(hipDeviceSynchronize());
        hipEvent_t a,b; HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
        std::vector<double> TA,TC,TE,TF;
        auto one=[&](auto&&k,std::vector<double>&o){ HC(hipEventRecord(a)); k(); HC(hipEventRecord(b));
            HC(hipEventSynchronize(b)); float ms; HC(hipEventElapsedTime(&ms,a,b)); o.push_back(ms); };
        for(int i=0;i<iters;i++){ one(fA,TA); one(fC,TC); one(fE,TE); one(fF,TF); }
        HC(hipEventDestroy(a)); HC(hipEventDestroy(b));
        double ta=med(TA),tc=med(TC),te=med(TE),tf=med(TF);
        double dC=(tc/ta-1)*100,dE=(te/ta-1)*100,dF=(tf/ta-1)*100;
        printf("%-31s %9.4f %+9.2f%% %+9.2f%% %+9.2f%% %9.1f\n",s.name,ta,dC,dE,dF,bytes/(ta*1e-3)/1e9);
        wC+=dC;wE+=dE;wF+=dF;n++;
        mxC=std::max(mxC,dC);mxE=std::max(mxE,dE);mxF=std::max(mxF,dF);
        HC(hipFree(A));HC(hipFree(x));HC(hipFree(y));
    }
    printf("%s\n",std::string(85,'-').c_str());
    printf("%-31s %9s %+9.2f%% %+9.2f%% %+9.2f%%\n","MEAN","",wC/n,wE/n,wF/n);
    printf("%-31s %9s %+9.2f%% %+9.2f%% %+9.2f%%\n","WORST SHAPE","",mxC,mxE,mxF);
    printf("\nbudget = +1.00%% worst-case; negative = faster than current\n");
    return 0;
}
