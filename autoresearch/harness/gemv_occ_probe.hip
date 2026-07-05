// gemv_occ_probe.hip — real-shaped mq4 decode GEMV, sweep OCCUPANCY (waves-in-flight per CU) via a launch-time
// dynamic-LDS knob, measure throughput. Tests the waves-in-flight / latency-hiding thesis directly on the real
// 1-wave-per-row access pattern (not a streaming proxy). If throughput FALLS as occupancy drops, the GEMV is
// latency-hiding-bound; the arch that falls STEEPER is more starved for in-flight relative to its bandwidth
// (Little's law) — i.e. the gfx11-vs-gfx12 differentiator. DRAM-sized weight models the streamed decode weights.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void gemv(const uint32_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ out, int K){
  extern __shared__ float lds[];                 // size set at launch -> the occupancy knob (reserved per block)
  int row = blockIdx.x, lane = threadIdx.x;
  int kpack = K/8;                               // 8 x 4-bit weights per uint32
  const uint32_t* wr = W + (size_t)row*kpack;
  float acc = 0.f;
  for(int p=lane; p<kpack; p+=32){
    uint32_t pk = wr[p];
    #pragma unroll
    for(int j=0;j<8;j++) acc += ((float)((pk>>(j*4))&0xF) - 8.0f) * x[p*8+j];   // dequant + fma
  }
  for(int o=16;o>0;o>>=1) acc += __shfl_xor(acc,o);
  if(lane==0){ lds[0]=acc; out[row]=lds[0]; }    // touch lds[0] so the reservation isn't elided
}

int main(){
  hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
  printf("# arch=%s  WGP=%d  maxClk=%dMHz\n", p.gcnArchName, p.multiProcessorCount, p.clockRate/1000);
  int M=131072, K=2048;                          // 128MB weight (DRAM-streamed) ; x resident (8KB)
  size_t wbytes=(size_t)M*(K/2);
  uint32_t* W=nullptr; float *x=nullptr,*out=nullptr;
  if(hipMalloc(&W,wbytes)!=hipSuccess){printf("malloc fail\n");return 1;}
  hipMalloc(&x,K*sizeof(float)); hipMalloc(&out,M*sizeof(float));
  hipMemset(W,0x21,wbytes); hipMemset(x,1,K*sizeof(float));
  int lds[]={0,2048,4096,8192,16384,24576,32768};   // bytes/block -> ~waves/CU = min(32, 65536/lds)
  printf("LDS_B  ~occ(wv/CU)   ms(best3)   Grow/s   GB/s(weights)\n");
  for(int li=0; li<(int)(sizeof(lds)/sizeof(int)); ++li){
    int L=lds[li];
    gemv<<<M,32,L>>>(W,x,out,K); hipDeviceSynchronize();     // warm
    float best=1e30f;
    for(int r=0;r<3;r++){
      hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
      hipEventRecord(s); gemv<<<M,32,L>>>(W,x,out,K); hipEventRecord(e); hipEventSynchronize(e);
      float ms=0; hipEventElapsedTime(&ms,s,e); if(ms<best)best=ms;
      hipEventDestroy(s); hipEventDestroy(e);
    }
    int occ = L>0 ? (65536/L < 32 ? 65536/L : 32) : 32;
    printf("%5d  %8d      %8.3f   %6.1f   %6.0f\n", L, occ, best, M/1e9/(best/1e3)*1000.0, (double)wbytes/1e9/(best/1e3));
  }
  hipFree(W); hipFree(x); hipFree(out); return 0;
}
