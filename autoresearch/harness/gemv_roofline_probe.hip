// gemv_roofline_probe.hip — isolate the "second bottleneck" that caps the mq4 decode GEMV below saturation.
// All at max occupancy (LDS=0). Ablations: strip reduction, strip dequant, NARROW(uint32) vs WIDE(uint4=128-bit)
// load, and prefetch depth. VGPR printed per variant (occupancy proxy). Whichever ablation jumps BW toward the
// arch's DRAM peak (gfx1100 960 / gfx1201 640 / gfx1030 512 / gfx1010 448 / gfx1151 ~256 GB/s) names the cap.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#define DEQ 1
#define RED 2
#define WIDE 4

template<int MODE,int DEPTH>
__global__ void gemv(const uint32_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ out, int K){
  extern __shared__ float lds[];
  int row=blockIdx.x, lane=threadIdx.x, kpack=K/8;
  const uint32_t* wr = W + (size_t)row*kpack;
  float acc=0.f;
  if constexpr (MODE & WIDE){
    const uint4* wr4=(const uint4*)wr; int kp4=kpack/4;
    for(int p=lane;p<kp4;p+=32){
      uint4 v=wr4[p]; uint32_t pk[4]={v.x,v.y,v.z,v.w};
      #pragma unroll
      for(int q=0;q<4;q++){
        if constexpr(MODE&DEQ){
          #pragma unroll
          for(int j=0;j<8;j++) acc+=((float)((pk[q]>>(j*4))&0xF)-8.f)*x[(p*4+q)*8+j];
        } else acc+=(float)pk[q];
      }
    }
  } else {
    for(int p=lane;p<kpack;p+=32*DEPTH){
      uint32_t pk[DEPTH];
      #pragma unroll
      for(int d=0;d<DEPTH;d++){ int pp=p+d*32; pk[d]= pp<kpack? wr[pp]:0u; }
      #pragma unroll
      for(int d=0;d<DEPTH;d++){ int pp=p+d*32; if(pp<kpack){
        if constexpr(MODE&DEQ){
          #pragma unroll
          for(int j=0;j<8;j++) acc+=((float)((pk[d]>>(j*4))&0xF)-8.f)*x[pp*8+j];
        } else acc+=(float)pk[d];
      }}
    }
  }
  if constexpr(MODE&RED){ for(int o=16;o>0;o>>=1) acc+=__shfl_xor(acc,o); }
  if(lane==0){ lds[0]=acc; out[row]=lds[0]; }
}

template<int MODE,int DEPTH>
void bench(const char* name, uint32_t* W, float* x, float* out, int M, int K){
  hipFuncAttributes fa; hipFuncGetAttributes(&fa,(const void*)gemv<MODE,DEPTH>);
  gemv<MODE,DEPTH><<<M,32,0>>>(W,x,out,K); hipDeviceSynchronize();
  float best=1e30f;
  for(int r=0;r<5;r++){
    hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);
    hipEventRecord(s); gemv<MODE,DEPTH><<<M,32,0>>>(W,x,out,K); hipEventRecord(e); hipEventSynchronize(e);
    float ms=0;hipEventElapsedTime(&ms,s,e); if(ms<best)best=ms;
    hipEventDestroy(s);hipEventDestroy(e);
  }
  double wb=(double)M*(K/2);
  printf("  %-26s vgpr=%-3d  %5.0f GB/s\n", name, fa.numRegs, wb/1e9/(best/1e3));
}

int main(){
  hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
  printf("# arch=%s WGP=%d maxClk=%dMHz\n", p.gcnArchName, p.multiProcessorCount, p.clockRate/1000);
  int M=131072,K=2048; size_t wb=(size_t)M*(K/2);
  uint32_t*W; hipMalloc(&W,wb); hipMemset(W,0x21,wb);
  float*x; hipMalloc(&x,K*4); hipMemset(x,1,K*4);
  float*o; hipMalloc(&o,M*4);
  // DPM WARMUP: spin ~4s to ramp the clock to steady max BEFORE any measurement (cold runs are 1.5-2x slow)
  for(int w=0; w<14000; w++) gemv<DEQ|RED,1><<<M,32,0>>>(W,x,o,K);
  hipDeviceSynchronize();
  puts("-- ablation @ max occ (LDS=0, post-DPM-warmup) --");
  bench<DEQ|RED,1>("baseline (A, drift-chk)", W,x,o,M,K);
  bench<DEQ|RED,1>("baseline narrow+dq+red", W,x,o,M,K);
  bench<DEQ,1>    ("  no-reduce (dq only)",  W,x,o,M,K);
  bench<RED,1>    ("  no-dequant (raw+red)", W,x,o,M,K);
  bench<0,1>      ("  raw, no-red, no-dq",   W,x,o,M,K);
  bench<DEQ|RED|WIDE,1>("WIDE uint4+dq+red", W,x,o,M,K);
  bench<RED|WIDE,1>    ("  WIDE raw+red",     W,x,o,M,K);
  puts("-- prefetch depth (narrow+dq+red); VGPR rises w/ depth (occ tradeoff) --");
  bench<DEQ|RED,1>("depth=1", W,x,o,M,K);
  bench<DEQ|RED,2>("depth=2", W,x,o,M,K);
  bench<DEQ|RED,4>("depth=4", W,x,o,M,K);
  bench<DEQ|RED,8>("depth=8", W,x,o,M,K);
  puts("-- drift check: baseline again at END (must match baseline A or the run is confounded) --");
  bench<DEQ|RED,1>("baseline (Z, drift-chk)", W,x,o,M,K);
  hipFree(W);hipFree(x);hipFree(o); return 0;
}
