// Is the 64-profile selector just a cheap re-derivation of MQ4-Lloyd (qt=30)?
//
// MQ4-Lloyd stores a FULL per-block 16-entry fp16 codebook: 32 B/group, 160 B total,
// 5.0 bpw. The selector stores 6 bits naming one of 64 pre-fitted profiles.
// Prior verdict (4a018afdf) on per-block codebooks: "costs 8x a per-channel scale
// for 1.8%" -- measured at 2-bit under RMS normalisation.
//
// Hypothesis under test: normalisation decides whether a per-block codebook is worth
// anything. Under RMS normalisation the FWHT + CLT make every block the same Gaussian
// shape, so there is nothing to adapt to (hence 1.8%). Under MAX normalisation blocks
// differ by their max/rms ratio (mean 3.046, p99 4.058), which creates real per-block
// shape variance a codebook can exploit.
//
// Grid: {RMS, MAX} normalisation x {single global cb, 64-profile selector, per-block Lloyd}.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define NLEV 16
#define NPROF 64
#define CB_GLOBAL 0
#define CB_SELECT 1
#define CB_LLOYD  2
#define CB_AFFINE 3
#define CB_UCLIP  4
#define CB_SYM    5
#define CB_HIER   6
#define CB_HIER2  7
__device__ __constant__ float CLIP_MULT[64] = {
  0.50f,0.51f,0.52f,0.53f,0.54f,0.55f,0.56f,0.57f,0.58f,0.59f,0.60f,0.61f,0.62f,0.63f,
  0.64f,0.65f,0.66f,0.67f,0.68f,0.69f,0.70f,0.71f,0.72f,0.73f,0.74f,0.75f,0.76f,0.77f,
  0.78f,0.79f,0.80f,0.81f,0.82f,0.83f,0.84f,0.85f,0.86f,0.87f,0.88f,0.89f,0.90f,0.905f,
  0.91f,0.915f,0.92f,0.925f,0.93f,0.935f,0.94f,0.945f,0.95f,0.955f,0.96f,0.965f,0.97f,
  0.975f,0.98f,0.9825f,0.985f,0.9875f,0.99f,0.9925f,0.995f,1.00f};
#define NORM_MAX 0
#define NORM_RMS 1

// GL_CB4 -- the Lloyd-Max optimum for a unit Gaussian, i.e. the codebook that BELONGS
// in RMS-normalised space. fam64 lives in MAX-normalised space (outermost +/-1.0), so
// using it on RMS data clips everything at 1.0 and reports a units bug as a result.
__device__ __constant__ float RMS_CB4[NLEV] = {
    -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9423f,-0.6568f,-0.3880f,-0.1284f,
     0.1284f, 0.3880f, 0.6568f, 0.9423f, 1.2562f, 1.6180f, 2.0690f, 2.7326f};

__global__ __launch_bounds__(256) void eval_cb(
    const float* __restrict__ G, int nblk,
    const float* __restrict__ fam,
    float tailthr, float p, int nsub, int cb_mode, int norm_mode, int lloyd_iters, int fp16_levels, int fp16_hdr,
    double* __restrict__ out_sse, double* __restrict__ out_tail_sse,
    unsigned long long* __restrict__ out_n, unsigned long long* __restrict__ out_tailn)
{
    __shared__ float sh_lev[NPROF * NLEV];
    __shared__ float sh_red[256];
    __shared__ float sh_scale[16];
    __shared__ int   sh_best[16];
    __shared__ float sh_z[256];
    __shared__ float sh_zh[256];
    __shared__ float sh_cb[16][NLEV];      // per-sub-block working codebook (Lloyd)
    __shared__ float sh_sum[16][NLEV];
    __shared__ int   sh_cnt[16][NLEV];

    const int tid = threadIdx.x;
    for (int i = tid; i < NPROF * NLEV; i += 256) sh_lev[i] = fam[i];
    __syncthreads();

    const int sub_sz = 256 / nsub;
    const int sub    = tid / sub_sz;
    const int lane   = tid % sub_sz;

    double sse = 0.0, tsse = 0.0;
    unsigned long long n = 0, tn = 0;

    for (int blk = blockIdx.x; blk < nblk; blk += gridDim.x) {
        const float w = G[(size_t)blk * 256 + tid];

        // ---- scale: per-sub-block max|w| or rms ----
        sh_red[tid] = (norm_mode == NORM_MAX) ? fabsf(w) : w * w;
        __syncthreads();
        for (int s = sub_sz >> 1; s > 0; s >>= 1) {
            if (lane < s) {
                sh_red[tid] = (norm_mode == NORM_MAX) ? fmaxf(sh_red[tid], sh_red[tid + s])
                                                      : sh_red[tid] + sh_red[tid + s];
            }
            __syncthreads();
        }
        if (lane == 0) {
            float v = sh_red[sub * sub_sz];
            if (norm_mode == NORM_RMS) v = sqrtf(v / (float)sub_sz);
            sh_scale[sub] = (v > 0.0f) ? v : 1.0f;
        }
        __syncthreads();
        const float scale = sh_scale[sub];
        const float z = w / scale;
        sh_z[tid] = z;
        __syncthreads();

        if (cb_mode == CB_LLOYD) {
            // ---- per-block Lloyd, MQ4-Lloyd style: fit 16 levels to THIS sub-block ----
            if (lane < NLEV) sh_cb[sub][lane] = (norm_mode == NORM_RMS) ? RMS_CB4[lane] : sh_lev[lane];
            __syncthreads();
            for (int it = 0; it < lloyd_iters; ++it) {
                if (lane < NLEV) { sh_sum[sub][lane] = 0.0f; sh_cnt[sub][lane] = 0; }
                __syncthreads();
                int bk = 0; float bd = 1e30f;
                for (int l = 0; l < NLEV; ++l) {
                    const float d = fabsf(z - sh_cb[sub][l]);
                    if (d < bd) { bd = d; bk = l; }
                }
                atomicAdd(&sh_sum[sub][bk], z);
                atomicAdd(&sh_cnt[sub][bk], 1);
                __syncthreads();
                if (lane < NLEV && sh_cnt[sub][lane] > 0)
                    sh_cb[sub][lane] = sh_sum[sub][lane] / (float)sh_cnt[sub][lane];
                __syncthreads();
            }
            // MQ4-Lloyd stores the fitted levels as fp16 -- charge that rounding,
            // otherwise the comparison flatters it with f32 codebook precision.
            if (lane < NLEV && fp16_levels)
                sh_cb[sub][lane] = __half2float(__float2half(sh_cb[sub][lane]));
            __syncthreads();
            float bd = 1e30f, bv = 0.0f;
            for (int l = 0; l < NLEV; ++l) {
                const float d = fabsf(z - sh_cb[sub][l]);
                if (d < bd) { bd = d; bv = sh_cb[sub][l]; }
            }
            sh_zh[tid] = bv;
            __syncthreads();
        } else {
            const int nprof = (cb_mode == CB_SELECT) ? NPROF : 1;
            const bool use_rms_cb = (norm_mode == NORM_RMS) && (cb_mode == CB_GLOBAL);
            const float wz = (p == 0.0f) ? 1.0f : __powf(fabsf(z) + 1e-30f, p);
            float bestcost = 1e30f; int bestj = 0;
            for (int j = 0; j < nprof; ++j) {
                const float* cb = use_rms_cb ? RMS_CB4 : &sh_lev[j * NLEV];
                float bd = 1e30f, bv = 0.0f;
                for (int l = 0; l < NLEV; ++l) {
                    const float d = fabsf(z - cb[l]);
                    if (d < bd) { bd = d; bv = cb[l]; }
                }
                sh_red[tid] = wz * (z - bv) * (z - bv);
                __syncthreads();
                for (int s = sub_sz >> 1; s > 0; s >>= 1) {
                    if (lane < s) sh_red[tid] += sh_red[tid + s];
                    __syncthreads();
                }
                const float cost = sh_red[sub * sub_sz];
                if (cost < bestcost) { bestcost = cost; bestj = j; }
                __syncthreads();
            }
            if (lane == 0) sh_best[sub] = bestj;
            __syncthreads();
            const float* cb = use_rms_cb ? RMS_CB4 : &sh_lev[sh_best[sub] * NLEV];
            float bd = 1e30f, bv = 0.0f;
            for (int l = 0; l < NLEV; ++l) {
                const float d = fabsf(z - cb[l]);
                if (d < bd) { bd = d; bv = cb[l]; }
            }
            sh_zh[tid] = bv;
            __syncthreads();
        }

        if (cb_mode == CB_AFFINE || cb_mode == CB_UCLIP || cb_mode == CB_SYM || cb_mode == CB_HIER || cb_mode == CB_HIER2) {
            // per-sub-block min/max for AFFINE; per-sub-block max for UCLIP
            sh_red[tid] = w;
            __syncthreads();
            for (int s = sub_sz >> 1; s > 0; s >>= 1) {
                if (lane < s) sh_red[tid] = fminf(sh_red[tid], sh_red[tid + s]);
                __syncthreads();
            }
            const float lo = sh_red[sub * sub_sz];
            __syncthreads();
            sh_red[tid] = w;
            __syncthreads();
            for (int s = sub_sz >> 1; s > 0; s >>= 1) {
                if (lane < s) sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + s]);
                __syncthreads();
            }
            const float hi = sh_red[sub * sub_sz];
            __syncthreads();
            if (cb_mode == CB_HIER || cb_mode == CB_HIER2) {
                // Q4_K-style hierarchical header: per-sub-block scale and zero quantised to
                // `hbits` each, against a per-256 fp16 super-scale (d) and super-zero (dmin).
                // Cost = nsub*2*hbits bits + 4 B, versus nsub*4 B for raw fp16 pairs.
                __shared__ float sh_ss[16], sh_zz[16], sh_dd[2];
                if (lane == 0) { sh_ss[sub] = (hi - lo) / 15.0f; sh_zz[sub] = lo; }
                __syncthreads();
                if (tid == 0) {
                    float smax = 0.f, zmax = 0.f;
                    for (int q = 0; q < nsub; ++q) { smax = fmaxf(smax, sh_ss[q]); zmax = fmaxf(zmax, fabsf(sh_zz[q])); }
                    sh_dd[0] = __half2float(__float2half(smax));   // d    (fp16)
                    sh_dd[1] = __half2float(__float2half(zmax));   // dmin (fp16)
                }
                __syncthreads();
                const float d = sh_dd[0], dmin = sh_dd[1];
                const float LV = (float)((1 << lloyd_iters) - 1);    // lloyd_iters reused as hbits
                float st, z;
                if (cb_mode == CB_HIER2) {
                    // range may only GROW: ceil the step, floor the zero. Guarantees the
                    // reconstructed interval covers [lo,hi], so neither extreme clips.
                    st = (d > 0.f) ? (ceilf(sh_ss[sub] / d * LV) / LV) * d : 0.f;
                    z  = (dmin > 0.f) ? (floorf(sh_zz[sub] / dmin * LV) / LV) * dmin : 0.f;
                } else {
                    st = (d > 0.f) ? (rintf(sh_ss[sub] / d * LV) / LV) * d : 0.f;
                    z  = (dmin > 0.f) ? (rintf(sh_zz[sub] / dmin * LV) / LV) * dmin : 0.f;
                }
                const float q  = (st > 0.0f) ? fminf(fmaxf(rintf((w - z) / st), 0.0f), 15.0f) : 0.0f;
                sh_zh[tid] = (q * st + z) / scale;
            } else if (cb_mode == CB_SYM) {
                // symmetric: 16 uniform levels over [-amax, +amax], q=0 -> -amax, q=15 -> +amax.
                // Only an fp16 scale is stored; no zero-point. Costs 2 B per sub-block.
                const float amax = fmaxf(fabsf(lo), fabsf(hi));
                const float st0 = 2.0f * amax / 15.0f;
                const float st  = fp16_hdr ? __half2float(__float2half(st0)) : st0;
                const float q   = (st > 0.0f) ? fminf(fmaxf(rintf((w + amax) / st), 0.0f), 15.0f) : 0.0f;
                sh_zh[tid] = (q * st - amax) / scale;
            } else if (cb_mode == CB_AFFINE) {
                // store scale+zero as fp16, as a real 136 B format must
                const float st0 = (hi - lo) / 15.0f;
                const float st  = fp16_hdr ? __half2float(__float2half(st0)) : st0;
                const float z   = fp16_hdr ? __half2float(__float2half(lo))  : lo;
                const float q  = (st > 0.0f) ? fminf(fmaxf(rintf((w - z) / st), 0.0f), 15.0f) : 0.0f;
                sh_zh[tid] = (q * st + z) / scale;
            } else {
                // 6-bit clip-multiplier selector: uniform levels over [-max*m, +max*m]
                const float amax = fmaxf(fabsf(lo), fabsf(hi));
                float bestc = 1e30f; int bestk = 63;
                for (int k = 0; k < 64; ++k) {
                    const float r  = amax * CLIP_MULT[k];
                    const float st = 2.0f * r / 15.0f;
                    const float q  = (st > 0.0f) ? fminf(fmaxf(rintf((w + r) / st), 0.0f), 15.0f) : 0.0f;
                    const float rc = q * st - r;
                    const float wz = (p == 0.0f) ? 1.0f : __powf(fabsf(w / scale) + 1e-30f, p);
                    sh_red[tid] = wz * (rc - w) * (rc - w);
                    __syncthreads();
                    for (int s = sub_sz >> 1; s > 0; s >>= 1) {
                        if (lane < s) sh_red[tid] += sh_red[tid + s];
                        __syncthreads();
                    }
                    const float cost = sh_red[sub * sub_sz];
                    if (cost < bestc) { bestc = cost; bestk = k; }
                    __syncthreads();
                }
                const float r  = amax * CLIP_MULT[bestk];
                const float st = 2.0f * r / 15.0f;
                const float q  = (st > 0.0f) ? fminf(fmaxf(rintf((w + r) / st), 0.0f), 15.0f) : 0.0f;
                sh_zh[tid] = (q * st - r) / scale;
            }
            __syncthreads();
        }

        const float e = sh_zh[tid] * scale - w;
        sse += (double)e * (double)e;
        n += 1;
        if (fabsf(w) >= tailthr) { tsse += (double)e * (double)e; tn += 1; }
        __syncthreads();
    }

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
    const float tailthr = argc > 1 ? atof(argv[1]) : 2.869166e-02f;
    FILE* f = fopen("/home/kaden/sweep_G.bin", "rb");
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    const int nblk = (int)(bytes / (256 * sizeof(float)));
    std::vector<float> hG((size_t)nblk * 256);
    if (fread(hG.data(), 1, bytes, f) != (size_t)bytes) { printf("short read\n"); return 1; }
    fclose(f);
    std::vector<float> hF(NPROF * NLEV);
    f = fopen("/home/kaden/sweep_fam.bin", "rb");
    if (fread(hF.data(), sizeof(float), NPROF*NLEV, f) != NPROF*NLEV) { printf("fam short\n"); return 1; }
    fclose(f);

    float *dG,*dF; double *dS,*dT; unsigned long long *dN,*dTN;
    CK(hipMalloc(&dG,hG.size()*sizeof(float))); CK(hipMalloc(&dF,hF.size()*sizeof(float)));
    CK(hipMalloc(&dS,sizeof(double))); CK(hipMalloc(&dT,sizeof(double)));
    CK(hipMalloc(&dN,sizeof(unsigned long long))); CK(hipMalloc(&dTN,sizeof(unsigned long long)));
    CK(hipMemcpy(dG,hG.data(),hG.size()*sizeof(float),hipMemcpyHostToDevice));
    CK(hipMemcpy(dF,hF.data(),hF.size()*sizeof(float),hipMemcpyHostToDevice));

    printf("  %d blocks, tail threshold %.6e\n\n", nblk, tailthr);
    printf("  %-40s %-6s %5s %7s   %-13s %-13s\n","codebook","norm","totB","bpw","overall MSE","tail-1% MSE");
    printf("  %s\n","---------------------------------------------------------------------------------------------------");

    struct Cfg { const char* name; int cb; int norm; int nsub; float p; int tot; int iters; int fp16; int hdr; };
    const Cfg cfgs[] = {
        {"v2        asym 2x128 fp16      136B", CB_AFFINE,NORM_MAX, 2,0.f,136,0,0,1},
        {"Q4_K-rn   hier 8x32 4b          140B", CB_HIER, NORM_MAX, 8,0.f,140,4,0,1},
        {"NEW-ceil  hier 8x32 4b          140B", CB_HIER2,NORM_MAX, 8,0.f,140,4,0,1},
        {"Q4_K-rn   hier 8x32 5b          142B", CB_HIER, NORM_MAX, 8,0.f,142,5,0,1},
        {"NEW-ceil  hier 8x32 5b          142B", CB_HIER2,NORM_MAX, 8,0.f,142,5,0,1},
        {"Q4_K-rn   hier 8x32 6b          144B", CB_HIER, NORM_MAX, 8,0.f,144,6,0,1},
        {"NEW-ceil  hier 8x32 6b          144B", CB_HIER2,NORM_MAX, 8,0.f,144,6,0,1},
        {"          asym 8x32 fp16       160B", CB_AFFINE,NORM_MAX, 8,0.f,160,0,0,1},
    };

    for (const auto& c : cfgs) {
        double z=0.0; unsigned long long zu=0;
        CK(hipMemcpy(dS,&z,sizeof(double),hipMemcpyHostToDevice));
        CK(hipMemcpy(dT,&z,sizeof(double),hipMemcpyHostToDevice));
        CK(hipMemcpy(dN,&zu,sizeof(unsigned long long),hipMemcpyHostToDevice));
        CK(hipMemcpy(dTN,&zu,sizeof(unsigned long long),hipMemcpyHostToDevice));
        hipLaunchKernelGGL(eval_cb, dim3(2048), dim3(256), 0, 0,
                           dG,nblk,dF,tailthr,c.p,c.nsub,c.cb,c.norm,c.iters,c.fp16,c.hdr,dS,dT,dN,dTN);
        CK(hipDeviceSynchronize());
        double hs,ht; unsigned long long hn,htn;
        CK(hipMemcpy(&hs,dS,sizeof(double),hipMemcpyDeviceToHost));
        CK(hipMemcpy(&ht,dT,sizeof(double),hipMemcpyDeviceToHost));
        CK(hipMemcpy(&hn,dN,sizeof(unsigned long long),hipMemcpyDeviceToHost));
        CK(hipMemcpy(&htn,dTN,sizeof(unsigned long long),hipMemcpyDeviceToHost));
        printf("  %-40s %-6s %5d %7.4f   %.4e    %.4e\n", c.name,
               c.norm==NORM_RMS?"RMS":"MAX", c.tot, c.tot*8.0/256.0,
               hs/(double)hn, htn? ht/(double)htn : 0.0);
    }
    return 0;
}
