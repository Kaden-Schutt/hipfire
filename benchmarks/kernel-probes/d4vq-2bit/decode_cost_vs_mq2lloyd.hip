// d=4 VQ (256 codewords, 2 bits/weight) vs MQ2-Lloyd, both 72 B per 256 weights.
// Identical memory traffic; the question is whether a 4 KB LDS codebook and its
// per-block fill are affordable versus MQ2-Lloyd's 4-entry in-header codebook.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#define HC(x) do{ hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)

__constant__ float CB4[256*4];
__constant__ float CB4H[256*2];   // same table, packed as half2 pairs

// ---- A: MQ2-Lloyd. Header = 4 x fp16 codebook. 64 B of 2-bit indices. ----
__launch_bounds__(32,16)
extern "C" __global__ void kLloyd(const char* __restrict__ A,const float* __restrict__ xr,
                                  float* __restrict__ y,int M,int K){
    const int row=blockIdx.x; if(row>=M) return; const int tid=threadIdx.x;
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*72);
    float acc=0.f;
    for(int g=0;g<gpr;g++){
        const char* gp=rp+g*72;
        const __half* cbh=(const __half*)gp;
        float c0=__half2float(cbh[0]),c1=__half2float(cbh[1]),
              c2=__half2float(cbh[2]),c3=__half2float(cbh[3]);
        float cb[4]={c0,c1,c2,c3};
        const unsigned char* nb=(const unsigned char*)(gp+8);
        int bi=g*256+tid*8, bo=tid*2;
        #pragma unroll
        for(int b=0;b<2;b++){
            unsigned char v=nb[bo+b];
            #pragma unroll
            for(int j=0;j<4;j++) acc+=cb[(v>>(2*j))&3]*xr[bi+b*4+j];
        }
    }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(tid==0) y[row]=acc;
}

__device__ __forceinline__ float e4m3d(unsigned int b){
    unsigned int e=(b>>3)&0xFu,m=b&0x7u;
    return __int_as_float((int)(((e+120u)<<23)|(m<<20)));
}

// ---- B: d4-VQ, f32 codebook staged in LDS (4 KB), filled per block ----
__launch_bounds__(32,16)
extern "C" __global__ void kVQ_lds32(const char* __restrict__ A,const float* __restrict__ xr,
                                     float* __restrict__ y,int M,int K){
    __shared__ float lut[256*4];
    const int tid=threadIdx.x;
    for(int i=tid;i<256*4;i+=32) lut[i]=CB4[i];
    __syncthreads();
    const int row=blockIdx.x; if(row>=M) return;
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*72);
    const int sub=tid>>3; float acc=0.f;
    for(int g=0;g<gpr;g++){
        const char* gp=rp+g*72;
        float mst=__builtin_bit_cast(float,*(const unsigned int*)(gp));
        float eff=mst*e4m3d((unsigned char)gp[4+sub]);
        const unsigned char* nb=(const unsigned char*)(gp+8);
        int bi=g*256+tid*8, bo=tid*2;
        #pragma unroll
        for(int b=0;b<2;b++){
            const float* c=&lut[(int)nb[bo+b]*4];
            acc+=(c[0]*xr[bi+b*4]+c[1]*xr[bi+b*4+1]
                 +c[2]*xr[bi+b*4+2]+c[3]*xr[bi+b*4+3])*eff;
        }
    }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(tid==0) y[row]=acc;
}

// ---- C: d4-VQ, fp16 codebook in LDS (2 KB) -> half the LDS pressure ----
__launch_bounds__(32,16)
extern "C" __global__ void kVQ_lds16(const char* __restrict__ A,const float* __restrict__ xr,
                                     float* __restrict__ y,int M,int K){
    __shared__ float lut[256*2];          // 256 codewords x 2 packed half2
    const int tid=threadIdx.x;
    for(int i=tid;i<256*2;i+=32) lut[i]=CB4H[i];
    __syncthreads();
    const int row=blockIdx.x; if(row>=M) return;
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*72);
    const int sub=tid>>3; float acc=0.f;
    for(int g=0;g<gpr;g++){
        const char* gp=rp+g*72;
        float mst=__builtin_bit_cast(float,*(const unsigned int*)(gp));
        float eff=mst*e4m3d((unsigned char)gp[4+sub]);
        const unsigned char* nb=(const unsigned char*)(gp+8);
        int bi=g*256+tid*8, bo=tid*2;
        #pragma unroll
        for(int b=0;b<2;b++){
            const __half2* c=(const __half2*)&lut[(int)nb[bo+b]*2];
            float2 a0=__half22float2(c[0]), a1=__half22float2(c[1]);
            acc+=(a0.x*xr[bi+b*4]+a0.y*xr[bi+b*4+1]
                 +a1.x*xr[bi+b*4+2]+a1.y*xr[bi+b*4+3])*eff;
        }
    }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(tid==0) y[row]=acc;
}

// ---- D: d4-VQ, codebook read straight from __constant__ (no LDS, no fill) ----
__launch_bounds__(32,16)
extern "C" __global__ void kVQ_const(const char* __restrict__ A,const float* __restrict__ xr,
                                     float* __restrict__ y,int M,int K){
    const int tid=threadIdx.x;
    const int row=blockIdx.x; if(row>=M) return;
    const int gpr=K/256; const char* rp=A+(long long)row*(gpr*72);
    const int sub=tid>>3; float acc=0.f;
    for(int g=0;g<gpr;g++){
        const char* gp=rp+g*72;
        float mst=__builtin_bit_cast(float,*(const unsigned int*)(gp));
        float eff=mst*e4m3d((unsigned char)gp[4+sub]);
        const unsigned char* nb=(const unsigned char*)(gp+8);
        int bi=g*256+tid*8, bo=tid*2;
        #pragma unroll
        for(int b=0;b<2;b++){
            const float* c=&CB4[(int)nb[bo+b]*4];
            acc+=(c[0]*xr[bi+b*4]+c[1]*xr[bi+b*4+1]
                 +c[2]*xr[bi+b*4+2]+c[3]*xr[bi+b*4+3])*eff;
        }
    }
    for(int o=16;o>0;o>>=1) acc+=__shfl_down(acc,o);
    if(tid==0) y[row]=acc;
}

static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
struct Op{ int m,k,per_token; const char* name; };

int main(int argc,char**argv){
    int iters=(argc>1)?atoi(argv[1]):640;
    hipDeviceProp_t p; HC(hipGetDeviceProperties(&p,0));
    printf("GPU: %s (%s) CUs=%d  LDS/block=%zu KB\n",p.name,p.gcnArchName,
           p.multiProcessorCount,(size_t)(p.sharedMemPerBlock/1024));
    printf("Qwen3.5-0.8B shapes; all variants 72 B/group (identical traffic)\n\n");

    std::vector<float> cb(256*4);
    unsigned int st=7u;
    for(auto&v:cb){ st=st*1664525u+1013904223u; v=((int)(st>>8)/8388608.0f)-1.0f; }
    HC(hipMemcpyToSymbol(HIP_SYMBOL(CB4),cb.data(),cb.size()*4));
    std::vector<unsigned short> cbh(256*4);
    for(int i=0;i<256*4;i++){ __half h=__float2half(cb[i]); cbh[i]=*(unsigned short*)&h; }
    HC(hipMemcpyToSymbol(HIP_SYMBOL(CB4H),cbh.data(),cbh.size()*2));

    const int L=24;
    Op ops[]={
        {3072,1024,L,"fused_qkv     [3072,1024]"},
        {1024,2048,L,"o_proj        [1024,2048]"},
        {7168,1024,L,"fused_gate_up [7168,1024]"},
        {1024,3584,L,"down_proj     [1024,3584]"},
        {248320,1024,1,"lm_head     [248320,1024]"},
    };
    printf("%-28s %9s %9s %9s %9s\n","op","Lloyd ms","VQ lds32","VQ lds16","VQ const");
    printf("%s\n",std::string(70,'-').c_str());
    double tl=0,t32=0,t16=0,tc=0, nl=0,n32=0,n16=0,nc=0;

    for(auto o:ops){
        size_t g=(size_t)o.k/256, bytes=(size_t)o.m*g*72;
        char* A; float *x,*y;
        HC(hipMalloc(&A,bytes)); HC(hipMalloc(&x,(size_t)o.k*4)); HC(hipMalloc(&y,(size_t)o.m*4));
        std::vector<unsigned char> h(bytes); unsigned int s2=0xBEEFu;
        for(size_t i=0;i<bytes;i++){ s2=s2*1664525u+1013904223u; h[i]=(unsigned char)(s2>>24); }
        for(size_t gi=0;gi<(size_t)o.m*g;gi++) for(int j=0;j<4;j++) h[gi*72+4+j]|=0x40;
        HC(hipMemcpy(A,h.data(),bytes,hipMemcpyHostToDevice));
        std::vector<float> hx(o.k);
        for(int i=0;i<o.k;i++){ s2=s2*1664525u+1013904223u; hx[i]=((int)(s2>>8)/8388608.f)-1.f; }
        HC(hipMemcpy(x,hx.data(),(size_t)o.k*4,hipMemcpyHostToDevice));

        dim3 gr(o.m),bk(32);
        auto fL=[&]{hipLaunchKernelGGL(kLloyd,gr,bk,0,0,A,x,y,o.m,o.k);};
        auto f32=[&]{hipLaunchKernelGGL(kVQ_lds32,gr,bk,0,0,A,x,y,o.m,o.k);};
        auto f16=[&]{hipLaunchKernelGGL(kVQ_lds16,gr,bk,0,0,A,x,y,o.m,o.k);};
        auto fC=[&]{hipLaunchKernelGGL(kVQ_const,gr,bk,0,0,A,x,y,o.m,o.k);};
        for(int i=0;i<80;i++){fL();f32();f16();fC();} HC(hipDeviceSynchronize());
        hipEvent_t a,b; HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
        std::vector<double> TL,T32,T16,TC;
        const int B=64;
        auto burst=[&](auto&&k,std::vector<double>&out){
            HC(hipEventRecord(a)); for(int j=0;j<B;j++) k();
            HC(hipEventRecord(b)); HC(hipEventSynchronize(b));
            float ms; HC(hipEventElapsedTime(&ms,a,b)); out.push_back(ms/B);
        };
        for(int i=0;i<iters/B+1;i++){ burst(fL,TL); burst(f32,T32); burst(f16,T16); burst(fC,TC); }
        HC(hipEventDestroy(a)); HC(hipEventDestroy(b));
        double L0=med(TL),A32=med(T32),A16=med(T16),AC=med(TC);
        printf("%-28s %9.4f %+8.2f%% %+8.2f%% %+8.2f%%\n",o.name,L0,
               (A32/L0-1)*100,(A16/L0-1)*100,(AC/L0-1)*100);
        tl+=L0*o.per_token; t32+=A32*o.per_token; t16+=A16*o.per_token; tc+=AC*o.per_token;
        if(std::string(o.name).find("lm_head")==std::string::npos){
            nl+=L0*o.per_token; n32+=A32*o.per_token; n16+=A16*o.per_token; nc+=AC*o.per_token; }
        HC(hipFree(A));HC(hipFree(x));HC(hipFree(y));
    }
    printf("%s\n",std::string(70,'-').c_str());
    printf("\nPER-TOKEN WEIGHTED vs MQ2-Lloyd\n");
    printf("  incl lm_head : lds32 %+.2f%%   lds16 %+.2f%%   const %+.2f%%\n",
           (t32/tl-1)*100,(t16/tl-1)*100,(tc/tl-1)*100);
    printf("  excl lm_head : lds32 %+.2f%%   lds16 %+.2f%%   const %+.2f%%\n",
           (n32/nl-1)*100,(n16/nl-1)*100,(nc/nl-1)*100);
    return 0;
}
