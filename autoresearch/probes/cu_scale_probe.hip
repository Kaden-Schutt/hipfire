// cu_scale_probe.hip — find gfx1100's CU-saturation knee. A latency/BW-sensitive streaming kernel launched
// at increasing GRID sizes (blocks). Throughput rises as more blocks occupy more CUs and hide more latency,
// then plateaus once the hardware is saturated. The KNEE = how many blocks gfx1100 actually needs. If the
// real decode GEMV launches FEWER blocks than the knee, its 96 CUs are idle and the arch-generic kernel wastes
// the silicon (the gfx11-vs-gfx12 confound). gfx1100 = 6 SE x 8 WGP = 48 WGP = 96 CU; block=256thr = 8 wave32.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void stream(const float4* __restrict__ buf, uint64_t n4, int iters, float* out){
  uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x, stride=(uint64_t)gridDim.x*blockDim.x;
  float acc=0.f;
  for(int it=0; it<iters; ++it) for(uint64_t i=tid; i<n4; i+=stride){ float4 v=buf[i]; acc+=v.x+v.y+v.z+v.w; }
  if(tid<stride) out[tid&4095]=acc;
}

int main(int argc,char**argv){
  size_t bytes = 64ull*1024*1024;   // 64MB: MALL-resident (isolates latency-hiding from DRAM BW)
  uint64_t n4 = bytes/16;
  float4* d=nullptr; float* o=nullptr;
  if(hipMalloc(&d,bytes)!=hipSuccess){printf("malloc fail\n");return 1;}
  hipMalloc(&o,4096*sizeof(float)); hipMemset(d,1,bytes);
  int block=256;
  int grids[]={6,12,24,48,72,96,144,192,288,384,768,1536};
  printf("blocks  eff_GBps   (knee = blocks gfx1100 needs to saturate; 48=1/WGP, 96=1/CU)\n");
  for(int gi=0; gi<(int)(sizeof(grids)/sizeof(int)); ++gi){
    int g=grids[gi];
    stream<<<g,block>>>(d,n4,8,o); hipDeviceSynchronize();          // warm
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s); stream<<<g,block>>>(d,n4,32,o); hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    printf("%6d   %.0f\n", g, (double)bytes*32.0/1e9/(ms/1e3));
  }
  hipFree(d); hipFree(o); return 0;
}
