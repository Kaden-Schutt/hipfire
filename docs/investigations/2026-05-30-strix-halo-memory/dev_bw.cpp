#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
static double now_s(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#define CK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("ERR %s: %s\n",#x,hipGetErrorString(e));return 1;}}while(0)
__global__ void writek(uint8_t* p,size_t n){size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;for(;i<n;i+=s)p[i]=(uint8_t)(i&0x7f);}
__global__ void sumk(const uint8_t* p,size_t n,unsigned long long* o){size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;unsigned long long a=0;for(;i<n;i+=s)a+=p[i];atomicAdd(o,a);}
int main(int c,char**v){double gb=c>1?atof(v[1]):40.0;size_t n=(size_t)(gb*1073741824.0);
 uint8_t* p=nullptr; if(hipMalloc((void**)&p,n)!=hipSuccess){printf("hipMalloc FAIL\n");return 2;}
 int th=256,bl=8192; double t1=now_s();hipLaunchKernelGGL(writek,dim3(bl),dim3(th),0,0,p,n);CK(hipDeviceSynchronize());double ws=now_s()-t1;
 printf("CARVEOUT WRITE %.1fGB: %.1f GB/s\n",gb,gb/ws);
 unsigned long long*d;CK(hipMalloc((void**)&d,8));CK(hipMemset(d,0,8));double t2=now_s();hipLaunchKernelGGL(sumk,dim3(bl),dim3(th),0,0,p,n,d);CK(hipDeviceSynchronize());double rs=now_s()-t2;
 printf("CARVEOUT READ  %.1fGB: %.1f GB/s\n",gb,gb/rs); hipFree(p);return 0;}
