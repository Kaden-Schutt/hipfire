// mall_probe.hip — map the gfx1100 cache hierarchy (L2 6MB / MALL 96MB / DRAM) by measuring steady-state
// REREAD bandwidth vs buffer size. A mid-plateau above DRAM BW between 6MB and 96MB == the MALL is engaged
// and catching L2 misses (→ "DRAM-thrash" is false). A straight drop to DRAM BW after 6MB == MALL not engaged.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

__global__ void reread(const float4* __restrict__ buf, uint64_t n4, int iters, float* out){
  uint64_t tid = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x*blockDim.x;
  float acc = 0.f;
  for(int it=0; it<iters; ++it)
    for(uint64_t i=tid; i<n4; i+=stride){ float4 v=buf[i]; acc += v.x+v.y+v.z+v.w; }
  if(tid < stride) out[tid & 4095] = acc;   // defeat DCE
}

int main(int argc,char**argv){
  size_t mb  = argc>1 ? (size_t)atoll(argv[1]) : 32;
  int    iters = argc>2 ? atoi(argv[2]) : 64;
  int    devname = argc>3 ? atoi(argv[3]) : 0;   // print arch name once
  size_t bytes = mb*1024ull*1024ull;
  uint64_t n4  = bytes/sizeof(float4);
  if(devname){ hipDeviceProp_t p; hipGetDeviceProperties(&p,0); fprintf(stderr,"gfx=%s\n",p.gcnArchName); }
  float4* d=nullptr; float* out=nullptr;
  if(hipMalloc(&d,bytes)!=hipSuccess){ printf("{\"mb\":%zu,\"err\":\"malloc\"}\n",mb); return 1; }
  hipMalloc(&out,4096*sizeof(float));
  hipMemset(d,1,bytes);
  int block=256, grid=2048;
  reread<<<grid,block>>>(d,n4,2,out); hipDeviceSynchronize();   // warm: fill whatever cache holds it
  hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  hipEventRecord(s); reread<<<grid,block>>>(d,n4,iters,out); hipEventRecord(e); hipEventSynchronize(e);
  float ms=0; hipEventElapsedTime(&ms,s,e);
  double gb=(double)bytes*(double)iters/1e9;
  printf("{\"mb\":%zu,\"iters\":%d,\"ms\":%.3f,\"eff_GBps\":%.0f}\n", mb, iters, ms, gb/(ms/1e3));
  hipFree(d); hipFree(out); return 0;
}
