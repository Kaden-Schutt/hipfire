// GTT-GEM bandwidth probe for gfx1151 (the autorocm KMD path).
// Allocates an AMDGPU_GEM_DOMAIN_GTT buffer directly via libdrm_amdgpu (GART-
// mapped, GPU-local — the same allocation class radv uses), exports it as a
// dma-buf, imports it into HIP, and measures GPU compute read/write bandwidth.
// This is the path hipMalloc/hipMallocManaged/hipHostMalloc can't reach.
#include <hip/hip_runtime.h>
#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>

static double now_s(){ return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define HK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
    printf("HIP ERR %s: %s\n",#x,hipGetErrorString(e)); return 1; } }while(0)

__global__ void writek(uint8_t* p,size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
    for(;i<n;i+=s) p[i]=(uint8_t)(i&0x7f);
}
__global__ void sumk(const uint8_t* p,size_t n,unsigned long long* o){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
    unsigned long long a=0; for(;i<n;i+=s) a+=p[i]; atomicAdd(o,a);
}

int main(int argc,char**argv){
    double gb = argc>1?atof(argv[1]):10.0;
    size_t n = (size_t)(gb*1073741824.0);
    const char* node = argc>2?argv[2]:"/dev/dri/renderD129";

    int fd=open(node,O_RDWR|O_CLOEXEC);
    if(fd<0){ printf("open %s fail\n",node); return 1; }
    amdgpu_device_handle dev; uint32_t maj,mn;
    if(amdgpu_device_initialize(fd,&maj,&mn,&dev)){ printf("amdgpu_device_initialize fail\n"); return 1; }
    printf("amdgpu drm %u.%u on %s\n",maj,mn,node);

    struct amdgpu_bo_alloc_request req; memset(&req,0,sizeof(req));
    req.alloc_size=n; req.phys_alignment=0x1000;
    req.preferred_heap=AMDGPU_GEM_DOMAIN_GTT; req.flags=0;
    amdgpu_bo_handle bo;
    int r=amdgpu_bo_alloc(dev,&req,&bo);
    if(r){ printf("amdgpu_bo_alloc(GTT %.1fGB) fail rc=%d\n",gb,r); return 2; }
    printf("GTT-domain BO alloc %.1f GB ok\n",gb); fflush(stdout);

    uint32_t dmafd=0;
    r=amdgpu_bo_export(bo,amdgpu_bo_handle_type_dma_buf_fd,&dmafd);
    if(r){ printf("amdgpu_bo_export fail rc=%d\n",r); return 3; }
    printf("dma-buf fd=%u\n",dmafd); fflush(stdout);

    hipExternalMemory_t ext;
    hipExternalMemoryHandleDesc hd; memset(&hd,0,sizeof(hd));
    hd.type=hipExternalMemoryHandleTypeOpaqueFd; hd.handle.fd=(int)dmafd; hd.size=n;
    HK(hipImportExternalMemory(&ext,&hd));
    void* dptr=nullptr;
    hipExternalMemoryBufferDesc bd; memset(&bd,0,sizeof(bd));
    bd.offset=0; bd.size=n; bd.flags=0;
    HK(hipExternalMemoryGetMappedBuffer(&dptr,ext,&bd));
    printf("HIP imported GTT buffer -> dev ptr=%p\n",dptr); fflush(stdout);

    int th=256,bl=8192;
    double t1=now_s(); hipLaunchKernelGGL(writek,dim3(bl),dim3(th),0,0,(uint8_t*)dptr,n); HK(hipDeviceSynchronize());
    double ws=now_s()-t1; printf("GTT-GEM WRITE %.1fGB: %.2fs = %.1f GB/s\n",gb,ws,gb/ws); fflush(stdout);

    unsigned long long* d; HK(hipMalloc((void**)&d,8)); HK(hipMemset(d,0,8));
    double t2=now_s(); hipLaunchKernelGGL(sumk,dim3(bl),dim3(th),0,0,(const uint8_t*)dptr,n,d); HK(hipDeviceSynchronize());
    double rs=now_s()-t2; unsigned long long hs=0; HK(hipMemcpy(&hs,d,8,hipMemcpyDeviceToHost));
    printf("GTT-GEM READ  %.1fGB: %.2fs = %.1f GB/s  sum=%llu\n",gb,rs,gb/rs,hs);
    printf("OK GTT-GEM zero-copy %.1f GB\n",gb);
    return 0;
}
