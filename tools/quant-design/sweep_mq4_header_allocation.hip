// One-off design sweep for the MQ4-G256 header-allocation question.
// Sweeps: selection exponent p, sub-block count, indexed top-k refinement (k, bits).
// One 256-thread workgroup per 256-weight group; 64 candidate level profiles.
// Errors accumulated in RAW weight space; tail = coefficients with |w| >= threshold.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define NLEV 16
#define NPROF 64

__global__ __launch_bounds__(256) void eval_variant(
    const float* __restrict__ G, int nblk,
    const float* __restrict__ fam,
    float tailthr, float p, int nsub, int refine_k, int refine_bits,
    double* __restrict__ out_sse, double* __restrict__ out_tail_sse,
    unsigned long long* __restrict__ out_n, unsigned long long* __restrict__ out_tailn)
{
    __shared__ float sh_lev[NPROF * NLEV];
    __shared__ float sh_red[256];
    __shared__ float sh_max[8];
    __shared__ int   sh_best[8];
    __shared__ float sh_z[256];
    __shared__ float sh_zh[256];

    const int tid = threadIdx.x;
    for (int i = tid; i < NPROF * NLEV; i += 256) sh_lev[i] = fam[i];
    __syncthreads();

    const int sub_sz = 256 / nsub;
    const int sub    = tid / sub_sz;

    double sse = 0.0, tsse = 0.0;
    unsigned long long n = 0, tn = 0;

    for (int blk = blockIdx.x; blk < nblk; blk += gridDim.x) {
        const float w = G[(size_t)blk * 256 + tid];

        // ---- per-sub-block max|w| ----
        sh_red[tid] = fabsf(w);
        __syncthreads();
        for (int s = sub_sz >> 1; s > 0; s >>= 1) {
            if ((tid % sub_sz) < s) sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + s]);
            __syncthreads();
        }
        if ((tid % sub_sz) == 0) sh_max[sub] = sh_red[tid] > 0.0f ? sh_red[tid] : 1.0f;
        __syncthreads();
        const float scale = sh_max[sub];
        const float z = w / scale;
        sh_z[tid] = z;
        __syncthreads();

        // ---- pick the profile minimising sum |z|^p (z - zhat)^2 over this sub-block ----
        const float wz = (p == 0.0f) ? 1.0f : __powf(fabsf(z) + 1e-30f, p);
        float bestcost = 1e30f; int bestj = 0;
        for (int j = 0; j < NPROF; ++j) {
            const float* cb = &sh_lev[j * NLEV];
            float bd = 1e30f, bv = 0.0f;
            for (int l = 0; l < NLEV; ++l) {
                const float d = fabsf(z - cb[l]);
                if (d < bd) { bd = d; bv = cb[l]; }
            }
            sh_red[tid] = wz * (z - bv) * (z - bv);
            __syncthreads();
            for (int s = sub_sz >> 1; s > 0; s >>= 1) {
                if ((tid % sub_sz) < s) sh_red[tid] += sh_red[tid + s];
                __syncthreads();
            }
            const float cost = sh_red[sub * sub_sz];
            if (cost < bestcost) { bestcost = cost; bestj = j; }
            __syncthreads();
        }
        if ((tid % sub_sz) == 0) sh_best[sub] = bestj;
        __syncthreads();

        // ---- reconstruct with the chosen profile ----
        {
            const float* cb = &sh_lev[sh_best[sub] * NLEV];
            float bd = 1e30f, bv = 0.0f;
            for (int l = 0; l < NLEV; ++l) {
                const float d = fabsf(z - cb[l]);
                if (d < bd) { bd = d; bv = cb[l]; }
            }
            sh_zh[tid] = bv;
        }
        __syncthreads();

        // ---- indexed top-k refinement: k passes of masked max over the WHOLE group ----
        for (int pass = 0; pass < refine_k; ++pass) {
            sh_red[tid] = (sh_zh[tid] == sh_z[tid]) ? -1.0f : fabsf(sh_z[tid]);
            __syncthreads();
            // full 256-wide argmax by value, then tag by equality (ties -> lowest index wins below)
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + s]);
                __syncthreads();
            }
            const float mv = sh_red[0];
            __syncthreads();
            if (mv > 0.0f) {
                // lowest-index thread holding the max refines itself
                sh_red[tid] = (fabsf(sh_z[tid]) == mv && sh_zh[tid] != sh_z[tid]) ? (float)tid : 1e30f;
                __syncthreads();
                for (int s = 128; s > 0; s >>= 1) {
                    if (tid < s) sh_red[tid] = fminf(sh_red[tid], sh_red[tid + s]);
                    __syncthreads();
                }
                const int win = (int)sh_red[0];
                __syncthreads();
                if (tid == win) {
                    const float q = (float)((1 << (refine_bits - 1)) - 1);
                    sh_zh[tid] = rintf(sh_z[tid] * q) / q;
                }
                __syncthreads();
            }
        }

        // ---- accumulate in RAW space ----
        const float e = sh_zh[tid] * scale - w;
        sse += (double)e * (double)e;
        n   += 1;
        if (fabsf(w) >= tailthr) { tsse += (double)e * (double)e; tn += 1; }
        __syncthreads();
    }

    // block reduction of the four accumulators
    __shared__ double rs[256], rt[256];
    __shared__ unsigned long long rn[256], rtn[256];
    rs[tid]=sse; rt[tid]=tsse; rn[tid]=n; rtn[tid]=tn;
    __syncthreads();
    for (int s=128;s>0;s>>=1){ if(tid<s){rs[tid]+=rs[tid+s];rt[tid]+=rt[tid+s];rn[tid]+=rn[tid+s];rtn[tid]+=rtn[tid+s];} __syncthreads(); }
    if (tid==0){ atomicAdd(out_sse,rs[0]); atomicAdd(out_tail_sse,rt[0]);
                 atomicAdd(out_n,rn[0]); atomicAdd(out_tailn,rtn[0]); }
}

#define CK(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1);} }while(0)

int main(int argc, char** argv) {
    const char* gpath = "/home/kaden/sweep_G.bin";
    const char* fpath = "/home/kaden/sweep_fam.bin";
    const float tailthr = argc > 1 ? atof(argv[1]) : 2.869166e-02f;

    FILE* f = fopen(gpath, "rb");
    if (!f) { printf("cannot open %s\n", gpath); return 1; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    const int nblk = (int)(bytes / (256 * sizeof(float)));
    std::vector<float> hG((size_t)nblk * 256);
    if (fread(hG.data(), 1, bytes, f) != (size_t)bytes) { printf("short read\n"); return 1; }
    fclose(f);

    std::vector<float> hF(NPROF * NLEV);
    f = fopen(fpath, "rb");
    if (fread(hF.data(), sizeof(float), NPROF * NLEV, f) != NPROF * NLEV) { printf("fam short\n"); return 1; }
    fclose(f);

    float *dG, *dF; double *dS, *dT; unsigned long long *dN, *dTN;
    CK(hipMalloc(&dG, hG.size()*sizeof(float)));
    CK(hipMalloc(&dF, hF.size()*sizeof(float)));
    CK(hipMalloc(&dS, sizeof(double))); CK(hipMalloc(&dT, sizeof(double)));
    CK(hipMalloc(&dN, sizeof(unsigned long long))); CK(hipMalloc(&dTN, sizeof(unsigned long long)));
    CK(hipMemcpy(dG, hG.data(), hG.size()*sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dF, hF.data(), hF.size()*sizeof(float), hipMemcpyHostToDevice));

    printf("  blocks %d   tail threshold %.6e\n\n", nblk, tailthr);
    printf("  %-46s %5s %5s %7s   %-13s %-13s\n", "variant", "hdrB", "totB", "bpw", "overall MSE", "tail-1% MSE");
    printf("  %s\n", "--------------------------------------------------------------------------------------------------");

    struct Cfg { const char* name; float p; int nsub; int k; int bits; int hdr; int tot; };
    const Cfg cfgs[] = {
        {"qt=43 132B: fp16 max + 6b sel, p=0",        0.f, 1, 0, 8,  4, 132},
        {"qt=43 132B: fp16 max + 6b sel, p=1",        1.f, 1, 0, 8,  4, 132},
        {"qt=43 132B: fp16 max + 6b sel, p=2",        2.f, 1, 0, 8,  4, 132},
        {"qt=43 132B: fp16 max + 6b sel, p=4",        4.f, 1, 0, 8,  4, 132},
        {"A 136B: p=0 + top-2 idx @12b (40b)",        0.f, 1, 2, 12, 8, 136},
        {"A 136B: p=0 + top-3 idx @5b  (39b)",        0.f, 1, 3, 5,  8, 136},
        {"A 136B: p=2 + top-2 idx @12b (40b)",        2.f, 1, 2, 12, 8, 136},
        {"A 136B: p=2 + top-3 idx @5b  (39b)",        2.f, 1, 3, 5,  8, 136},
        {"B 136B: 2x128 sub-blk max+sel, p=0",        0.f, 2, 0, 8,  6, 136},
        {"B 136B: 2x128 sub-blk max+sel, p=2",        2.f, 2, 0, 8,  6, 136},
        {"B 136B: 4x64  sub-blk max+sel, p=2",        2.f, 4, 0, 8, 12, 144},
        {"C 140B: p=0 + top-4 idx @6b (56b)",         0.f, 1, 4, 6, 12, 140},
        {"C 140B: p=2 + top-4 idx @6b (56b)",         2.f, 1, 4, 6, 12, 140},
        {"C 140B: p=0 + top-4 idx @8b (64b)",         0.f, 1, 4, 8, 12, 140},
    };

    const int grid = 2048;
    for (const auto& c : cfgs) {
        double z = 0.0; unsigned long long zu = 0;
        CK(hipMemcpy(dS,&z,sizeof(double),hipMemcpyHostToDevice));
        CK(hipMemcpy(dT,&z,sizeof(double),hipMemcpyHostToDevice));
        CK(hipMemcpy(dN,&zu,sizeof(unsigned long long),hipMemcpyHostToDevice));
        CK(hipMemcpy(dTN,&zu,sizeof(unsigned long long),hipMemcpyHostToDevice));
        hipLaunchKernelGGL(eval_variant, dim3(grid), dim3(256), 0, 0,
                           dG, nblk, dF, tailthr, c.p, c.nsub, c.k, c.bits, dS, dT, dN, dTN);
        CK(hipDeviceSynchronize());
        double hs, ht; unsigned long long hn, htn;
        CK(hipMemcpy(&hs,dS,sizeof(double),hipMemcpyDeviceToHost));
        CK(hipMemcpy(&ht,dT,sizeof(double),hipMemcpyDeviceToHost));
        CK(hipMemcpy(&hn,dN,sizeof(unsigned long long),hipMemcpyDeviceToHost));
        CK(hipMemcpy(&htn,dTN,sizeof(unsigned long long),hipMemcpyDeviceToHost));
        printf("  %-46s %5d %5d %7.4f   %.4e    %.4e\n", c.name, c.hdr, c.tot,
               c.tot * 8.0 / 256.0, hs / (double)hn, htn ? ht / (double)htn : 0.0);
    }
    return 0;
}
