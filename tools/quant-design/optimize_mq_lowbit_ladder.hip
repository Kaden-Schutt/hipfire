// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// GPU format-design optimizer for the matched MagnumQuant/HFQ G256 ladder.
// Evaluates 2/3/4-bit matched-header candidates, then measures the neutral-size
// 5/6-bit legacy-affine and dual-G128 fp16-header (V2) hypotheses on the same fixture.
// Candidate codecs are reconstructed only from the bytes charged in `header_b`.
//
// Usage:
//   hipcc -O3 --offload-arch=gfx1201 optimize_mq_lowbit_ladder.hip -o /tmp/mqopt
//   /tmp/mqopt [.codeinsight+research/quant-design/fixtures/sweep_G.bin] [tail-threshold] [K]
// Fixture is post-FWHT contiguous row-major G256 blocks (default 4096 x K=17408).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define MAX_LEVELS 16
#define NPROF 64

#define MODE_AFFINE 0
#define MODE_GL_RMS 1
#define MODE_ANCHORED_MAX 2
#define MODE_SELECTOR_MAX 3
#define MODE_LLOYD 4
// Production-faithful affine for MQ5/MQ6 v1 and the MQ4V2-style dual-G128 header.
// Degenerate half (hi==lo or f16 scale underflows to 0): store scale 0, all q=0.
#define MODE_AFFINE_PROD 5
#define MODE_LLOYD_ANCHORED 6


#define CK(x) do { \
    hipError_t e_ = (x); \
    if (e_ != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); \
        std::exit(1); \
    } \
} while (0)

struct DeviceResult {
    double sse;
    double tail_sse;
    double max_relative_sum;
    unsigned long long n;
    unsigned long long tail_n;
    unsigned long long blocks;
};

// Production hipfire f32→f16 truncates (not RNE). Bit-identical to
// crates/hipfire-quantize/src/float16.rs::f32_to_f16 / IEEE f16 decode.
__device__ __forceinline__ unsigned short f32_to_f16_trunc(float val) {
    const unsigned bits = __float_as_uint(val);
    const unsigned sign = (bits >> 31) & 1u;
    const int exp = (int)((bits >> 23) & 0xFFu);
    const unsigned frac = bits & 0x7FFFFFu;
    if (exp == 0xFF) {
        const unsigned f16_frac = (frac == 0u) ? 0u : ((frac >> 13) | 1u);
        return (unsigned short)((sign << 15) | (0x1Fu << 10) | f16_frac);
    }
    const int new_exp = exp - 127 + 15;
    if (new_exp >= 31)
        return (unsigned short)((sign << 15) | (0x1Fu << 10));
    if (new_exp <= 0) {
        if (new_exp < -10)
            return (unsigned short)(sign << 15);
        const unsigned f = frac | 0x800000u;
        const unsigned shift = (unsigned)(1 - new_exp + 13);
        return (unsigned short)((sign << 15) | (f >> shift));
    }
    return (unsigned short)((sign << 15) | ((unsigned)new_exp << 10) | (frac >> 13));
}

__device__ __forceinline__ float f16_to_f32_bits(unsigned short h) {
    const unsigned sign = ((unsigned)h >> 15) & 1u;
    unsigned exp = ((unsigned)h >> 10) & 0x1Fu;
    unsigned frac = (unsigned)h & 0x3FFu;
    unsigned out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign << 31;
        } else {
            exp = 127u - 15u + 1u;
            while ((frac & 0x400u) == 0u) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x3FFu;
            out = (sign << 31) | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31u) {
        out = (sign << 31) | (0xFFu << 23) | (frac << 13);
    } else {
        out = (sign << 31) | ((exp + 127u - 15u) << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}

__device__ __forceinline__ float f16_trunc_rt(float x) {
    return f16_to_f32_bits(f32_to_f16_trunc(x));
}

// Production Lloyd centroids use RNE f16 (f32_to_fp16_bits), not trunc.
__device__ __forceinline__ unsigned short f32_to_fp16_bits_rne(float v) {
    const unsigned bits = __float_as_uint(v);
    const unsigned short sign = (unsigned short)((bits >> 16) & 0x8000u);
    int exp = (int)((bits >> 23) & 0xFFu);
    const unsigned mant = bits & 0x7FFFFFu;
    if (exp == 0xFF) {
        const unsigned short m16 = mant != 0u ? (unsigned short)0x200 : (unsigned short)0;
        return (unsigned short)(sign | 0x7C00 | m16);
    }
    exp -= 127 - 15;
    if (exp >= 0x1F)
        return (unsigned short)(sign | 0x7C00);
    if (exp <= 0) {
        if (exp < -10)
            return sign;
        const unsigned m = mant | 0x800000u;
        const unsigned shift = (unsigned)(1 - exp) + 13u;
        unsigned short m16 = (unsigned short)(m >> shift);
        const unsigned lost = m & ((1u << shift) - 1u);
        const unsigned half = 1u << (shift - 1u);
        if (lost > half || (lost == half && (m16 & 1u) == 1u))
            m16 = (unsigned short)(m16 + 1u);
        return (unsigned short)(sign | m16);
    }
    unsigned short m16 = (unsigned short)(mant >> 13);
    const unsigned lost = mant & 0x1FFFu;
    if (lost > 0x1000u || (lost == 0x1000u && (m16 & 1u) == 1u)) {
        m16 = (unsigned short)(m16 + 1u);
        if (m16 == 0x400) {
            m16 = 0;
            ++exp;
            if (exp >= 0x1F)
                return (unsigned short)(sign | 0x7C00);
        }
    }
    return (unsigned short)(sign | ((unsigned short)exp << 10) | m16);
}

__device__ __forceinline__ float f16_rne_rt(float x) {
    return f16_to_f32_bits(f32_to_fp16_bits_rne(x));
}


__global__ __launch_bounds__(256) void eval_candidate(
    const float* __restrict__ weights,
    int nblk,
    const float* __restrict__ table,
    float tail_threshold,
    int bits,
    int mode,
    int nsub,
    int fp16_affine_header,
    float selector_p,
    DeviceResult* __restrict__ out)
{
    __shared__ float sh_gl[MAX_LEVELS];
    __shared__ float sh_anchor[MAX_LEVELS];
    __shared__ float sh_fam[NPROF * MAX_LEVELS];
    __shared__ float sh_red[256];
    __shared__ float sh_aux[256];
    __shared__ float sh_sort[256];
    __shared__ float sh_scale[8];
    __shared__ float sh_zero[8];
    __shared__ int sh_best[8];
    __shared__ float sh_cb[8][MAX_LEVELS];
    __shared__ double sh_sum[8][MAX_LEVELS];
    __shared__ int sh_count[8][MAX_LEVELS];
    __shared__ int sh_assignment[256];

    const int tid = (int)threadIdx.x;
    const int nlev = 1 << bits;
    if (tid < MAX_LEVELS) {
        sh_gl[tid] = table[tid];
        sh_anchor[tid] = table[MAX_LEVELS + tid];
    }
    for (int i = tid; i < NPROF * MAX_LEVELS; i += 256)
        sh_fam[i] = table[2 * MAX_LEVELS + i];
    __syncthreads();

    const int sub_size = 256 / nsub;
    const int sub = tid / sub_size;
    const int lane = tid % sub_size;
    double sse = 0.0;
    double tail_sse = 0.0;
    double max_relative_sum = 0.0;
    unsigned long long count = 0;
    unsigned long long tail_count = 0;
    unsigned long long block_count = 0;

    for (int blk = (int)blockIdx.x; blk < nblk; blk += (int)gridDim.x) {
        const float w = weights[(size_t)blk * 256 + tid];
        float reconstructed = 0.0f;

        if (mode == MODE_AFFINE || mode == MODE_AFFINE_PROD) {
            sh_red[tid] = w;
            __syncthreads();
            for (int stride = sub_size >> 1; stride > 0; stride >>= 1) {
                if (lane < stride)
                    sh_red[tid] = fminf(sh_red[tid], sh_red[tid + stride]);
                __syncthreads();
            }
            const float lo = sh_red[sub * sub_size];
            sh_red[tid] = w;
            __syncthreads();
            for (int stride = sub_size >> 1; stride > 0; stride >>= 1) {
                if (lane < stride)
                    sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + stride]);
                __syncthreads();
            }
            const float hi = sh_red[sub * sub_size];
            if (lane == 0) {
                float step;
                float zero = lo;
                if (mode == MODE_AFFINE_PROD && fp16_affine_header) {
                    // MQ4V2 contract: [s0,z0,s1,z1] f16; hi==lo → scale bits 0.
                    const float step_f32 =
                        hi > lo ? (hi - lo) / (float)(nlev - 1) : 0.0f;
                    step = f16_trunc_rt(step_f32);
                    zero = f16_trunc_rt(lo);
                    if (hi == lo)
                        step = 0.0f;
                } else if (mode == MODE_AFFINE_PROD) {
                    // Legacy MQ5/MQ6 v1: one f32 scale+zero per G256.
                    step = hi > lo ? (hi - lo) / (float)(nlev - 1) : 1.0f;
                } else {
                    // Original 2/3/4 research affine (unchanged semantics).
                    step = hi > lo ? (hi - lo) / (float)(nlev - 1) : 1.0f;
                    if (fp16_affine_header) {
                        step = f16_trunc_rt(step);
                        zero = f16_trunc_rt(zero);
                    }
                }
                sh_scale[sub] = step;
                sh_zero[sub] = zero;
            }
            __syncthreads();
            const float step = sh_scale[sub];
            const float zero = sh_zero[sub];
            float q;
            if (!(step > 0.0f)) {
                q = 0.0f;
            } else if (mode == MODE_AFFINE_PROD) {
                // Production cast: floor(x + 0.5) then clamp (matches MQ4V2/MQ5/MQ6).
                q = floorf((w - zero) / step + 0.5f);
                q = fminf(fmaxf(q, 0.0f), (float)(nlev - 1));
            } else {
                q = fminf(fmaxf(rintf((w - zero) / step), 0.0f),
                          (float)(nlev - 1));
            }
            reconstructed = q * step + zero;
        } else if (mode == MODE_LLOYD || mode == MODE_LLOYD_ANCHORED) {
            // MODE_LLOYD: percentile init, 8 Lloyd iters; keep final assignment
            // indices; RNE fp16 centroid store; no reassignment after rounding
            // (production unconstrained mq2lloyd).
            // MODE_LLOYD_ANCHORED: production-faithful endpoint-anchored MQ2
            // Lloyd. Sort the sub-block; pin endpoints to sorted min/max
            // (rank 0 / sub_size-1); init interior k at percentile rank
            // rintf(((k+0.5)/nlev)*(sub_size-1)) — for nlev=4 that is
            // 37.5%/62.5%. Eight Lloyd iters with fixed endpoints; RNE-round
            // all levels then final nearest assignment so encoder matches
            // GPU decode.
            // `nsub=2` composes the Lloyd fit with V2's two G128 containers.
            const int anchored = (mode == MODE_LLOYD_ANCHORED);
            if (anchored) {
                sh_sort[tid] = w;
                __syncthreads();
                for (int width = 2; width <= sub_size; width <<= 1) {
                    for (int stride = width >> 1; stride > 0; stride >>= 1) {
                        const int peer_lane = lane ^ stride;
                        const int peer = sub * sub_size + peer_lane;
                        if (peer_lane > lane) {
                            const bool ascending = (lane & width) == 0;
                            const float a = sh_sort[tid];
                            const float b = sh_sort[peer];
                            if ((ascending && a > b) || (!ascending && a < b)) {
                                sh_sort[tid] = b;
                                sh_sort[peer] = a;
                            }
                        }
                        __syncthreads();
                    }
                }
                const float lo = sh_sort[sub * sub_size];
                const float hi = sh_sort[sub * sub_size + (sub_size - 1)];
                if (lane < nlev) {
                    if (lane == 0)
                        sh_cb[sub][lane] = lo;
                    else if (lane == nlev - 1)
                        sh_cb[sub][lane] = hi;
                    else {
                        // Interior percentiles match production mq2lloyd
                        // anchored init (nlev=4 → ranks at 37.5% / 62.5%).
                        const float fraction =
                            ((float)lane + 0.5f) / (float)nlev;
                        const int rank = sub * sub_size +
                            (int)rintf(fraction * (float)(sub_size - 1));
                        sh_cb[sub][lane] = sh_sort[rank];
                    }
                }
                __syncthreads();
                // Always take the same control flow (nsub may split hi/lo).
                // Degenerate hi==lo: endpoints already equal; interiors stay at lo.
                for (int iteration = 0; iteration < 8; ++iteration) {
                    if (lane < nlev) {
                        sh_sum[sub][lane] = 0.0;
                        sh_count[sub][lane] = 0;
                    }
                    __syncthreads();
                    int nearest = 0;
                    float best_distance = fabsf(w - sh_cb[sub][0]);
                    for (int level = 1; level < nlev; ++level) {
                        const float distance = fabsf(w - sh_cb[sub][level]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            nearest = level;
                        }
                    }
                    sh_assignment[tid] = nearest;
                    atomicAdd(&sh_sum[sub][nearest], (double)w);
                    atomicAdd(&sh_count[sub][nearest], 1);
                    __syncthreads();
                    // Update interiors only when the block has range; re-pin
                    // endpoints every iter so cp0/cp{n-1} stay exact extrema.
                    if (lane == 0)
                        sh_cb[sub][0] = lo;
                    else if (lane == nlev - 1)
                        sh_cb[sub][lane] = hi;
                    else if (lane > 0 && lane < nlev - 1 && hi > lo &&
                             sh_count[sub][lane] != 0) {
                        sh_cb[sub][lane] =
                            (float)(sh_sum[sub][lane] /
                                    (double)sh_count[sub][lane]);
                    }
                    __syncthreads();
                }
                // Round all four before final nearest-codepoint assignment.
                if (lane < nlev)
                    sh_cb[sub][lane] = f16_rne_rt(sh_cb[sub][lane]);
                __syncthreads();
                {
                    int nearest = 0;
                    float best_distance = fabsf(w - sh_cb[sub][0]);
                    for (int level = 1; level < nlev; ++level) {
                        const float distance = fabsf(w - sh_cb[sub][level]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            nearest = level;
                        }
                    }
                    reconstructed = sh_cb[sub][nearest];
                }
            } else {
                sh_sort[tid] = w;
                __syncthreads();
                for (int width = 2; width <= sub_size; width <<= 1) {
                    for (int stride = width >> 1; stride > 0; stride >>= 1) {
                        const int peer_lane = lane ^ stride;
                        const int peer = sub * sub_size + peer_lane;
                        if (peer_lane > lane) {
                            const bool ascending = (lane & width) == 0;
                            const float a = sh_sort[tid];
                            const float b = sh_sort[peer];
                            if ((ascending && a > b) || (!ascending && a < b)) {
                                sh_sort[tid] = b;
                                sh_sort[peer] = a;
                            }
                        }
                        __syncthreads();
                    }
                }
                if (lane < nlev) {
                    const float fraction = ((float)lane + 0.5f) / (float)nlev;
                    const int rank = sub * sub_size +
                        (int)rintf(fraction * (float)(sub_size - 1));
                    sh_cb[sub][lane] = sh_sort[rank];
                }
                __syncthreads();
                for (int iteration = 0; iteration < 8; ++iteration) {
                    if (lane < nlev) {
                        sh_sum[sub][lane] = 0.0;
                        sh_count[sub][lane] = 0;
                    }
                    __syncthreads();
                    int nearest = 0;
                    float best_distance = fabsf(w - sh_cb[sub][0]);
                    for (int level = 1; level < nlev; ++level) {
                        const float distance = fabsf(w - sh_cb[sub][level]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            nearest = level;
                        }
                    }
                    sh_assignment[tid] = nearest;
                    atomicAdd(&sh_sum[sub][nearest], (double)w);
                    atomicAdd(&sh_count[sub][nearest], 1);
                    __syncthreads();
                    if (lane < nlev && sh_count[sub][lane] != 0) {
                        sh_cb[sub][lane] =
                            (float)(sh_sum[sub][lane] / (double)sh_count[sub][lane]);
                    }
                    __syncthreads();
                }
                if (lane < nlev)
                    sh_cb[sub][lane] = f16_rne_rt(sh_cb[sub][lane]);
                __syncthreads();
                reconstructed = sh_cb[sub][sh_assignment[tid]];
            }
        } else {
            sh_red[tid] = mode == MODE_GL_RMS ? w * w : fabsf(w);
            __syncthreads();
            for (int stride = sub_size >> 1; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    sh_red[tid] = mode == MODE_GL_RMS
                        ? sh_red[tid] + sh_red[tid + stride]
                        : fmaxf(sh_red[tid], sh_red[tid + stride]);
                }
                __syncthreads();
            }
            if (lane == 0) {
                float scale = sh_red[sub * sub_size];
                if (mode == MODE_GL_RMS)
                    scale = sqrtf(scale / (float)sub_size);
                if (!(scale > 0.0f))
                    scale = 1.0f;
                sh_scale[sub] = f16_trunc_rt(scale);
            }
            __syncthreads();
            const float scale = sh_scale[sub];
            const float z = w / scale;
            const float* levels = mode == MODE_GL_RMS ? sh_gl : sh_anchor;

            if (mode == MODE_SELECTOR_MAX) {
                float best_cost = 1.0e30f;
                int best_profile = 0;
                const float weight = selector_p == 0.0f
                    ? 1.0f
                    : __powf(fabsf(z) + 1.0e-30f, selector_p);
                for (int profile = 0; profile < NPROF; ++profile) {
                    const float* candidate = &sh_fam[profile * MAX_LEVELS];
                    float nearest_value = candidate[0];
                    float nearest_distance = fabsf(z - nearest_value);
                    for (int level = 1; level < nlev; ++level) {
                        const float distance = fabsf(z - candidate[level]);
                        if (distance < nearest_distance) {
                            nearest_distance = distance;
                            nearest_value = candidate[level];
                        }
                    }
                    const float error = z - nearest_value;
                    sh_red[tid] = weight * error * error;
                    __syncthreads();
                    for (int stride = sub_size >> 1; stride > 0; stride >>= 1) {
                        if (lane < stride)
                            sh_red[tid] += sh_red[tid + stride];
                        __syncthreads();
                    }
                    const float cost = sh_red[sub * sub_size];
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_profile = profile;
                    }
                    __syncthreads();
                }
                if (lane == 0)
                    sh_best[sub] = best_profile;
                __syncthreads();
                levels = &sh_fam[sh_best[sub] * MAX_LEVELS];
            }

            float nearest_value = levels[0];
            float nearest_distance = fabsf(z - nearest_value);
            for (int level = 1; level < nlev; ++level) {
                const float distance = fabsf(z - levels[level]);
                if (distance < nearest_distance) {
                    nearest_distance = distance;
                    nearest_value = levels[level];
                }
            }
            reconstructed = nearest_value * scale;
        }

        const float error = reconstructed - w;
        sse += (double)error * (double)error;
        ++count;
        if (fabsf(w) >= tail_threshold) {
            tail_sse += (double)error * (double)error;
            ++tail_count;
        }
        sh_red[tid] = fabsf(w);
        sh_aux[tid] = fabsf(error) / fmaxf(fabsf(w), 1.0e-30f);
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride && sh_red[tid + stride] > sh_red[tid]) {
                sh_red[tid] = sh_red[tid + stride];
                sh_aux[tid] = sh_aux[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            max_relative_sum += (double)sh_aux[0];
            ++block_count;
        }
        __syncthreads();
    }

    __shared__ double reduce_sse[256];
    __shared__ double reduce_tail[256];
    __shared__ double reduce_max_relative[256];
    __shared__ unsigned long long reduce_n[256];
    __shared__ unsigned long long reduce_tail_n[256];
    __shared__ unsigned long long reduce_blocks[256];
    reduce_sse[tid] = sse;
    reduce_tail[tid] = tail_sse;
    reduce_max_relative[tid] = max_relative_sum;
    reduce_n[tid] = count;
    reduce_tail_n[tid] = tail_count;
    reduce_blocks[tid] = block_count;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sse[tid] += reduce_sse[tid + stride];
            reduce_tail[tid] += reduce_tail[tid + stride];
            reduce_n[tid] += reduce_n[tid + stride];
            reduce_tail_n[tid] += reduce_tail_n[tid + stride];
            reduce_max_relative[tid] += reduce_max_relative[tid + stride];
            reduce_blocks[tid] += reduce_blocks[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&out->sse, reduce_sse[0]);
        atomicAdd(&out->tail_sse, reduce_tail[0]);
        atomicAdd(&out->max_relative_sum, reduce_max_relative[0]);
        atomicAdd(&out->n, reduce_n[0]);
        atomicAdd(&out->tail_n, reduce_tail_n[0]);
        atomicAdd(&out->blocks, reduce_blocks[0]);
    }
}

// ── Production MFP{2,3,4}G32E8 reconstruction (mirrors hipfire-quantize e8.rs) ──
// Layout: 16-byte row header (f16 row_scale_a + meta) + (1+4n) B per G32
//   (E4M3 block scale + 4×n-byte narrow E8 codewords). Tuned centers n2=2/n3=4/n4=7;
//   quant steps 3.8/1.8/0.88. Fixture is already post-FWHT.

__device__ __forceinline__ float mfp_round_tie_away(float x) {
    return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
}

__device__ __forceinline__ void mfp_closest_d8(const float* u, float* r) {
    long long s = 0;
    int worst_idx = 0;
    float worst_abs = -1.0f;
    float worst_dir = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float ri = mfp_round_tie_away(u[i]);
        r[i] = ri;
        s += (long long)ri;
        const float e = u[i] - ri;
        const float a = fabsf(e);
        if (a > worst_abs) {
            worst_abs = a;
            worst_idx = i;
            worst_dir = e >= 0.0f ? 1.0f : -1.0f;
        }
    }
    if ((s & 1ll) != 0)
        r[worst_idx] += worst_dir;
}

__device__ __forceinline__ void mfp_closest_e8(const float* u, float* out) {
    float a[8], b[8], ush[8], bsh[8];
    mfp_closest_d8(u, a);
    for (int i = 0; i < 8; ++i)
        ush[i] = u[i] - 0.5f;
    mfp_closest_d8(ush, bsh);
    for (int i = 0; i < 8; ++i)
        b[i] = bsh[i] + 0.5f;
    float da = 0.0f, db = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float ea = u[i] - a[i];
        const float eb = u[i] - b[i];
        da += ea * ea;
        db += eb * eb;
    }
    if (da <= db) {
        for (int i = 0; i < 8; ++i)
            out[i] = a[i];
    } else {
        for (int i = 0; i < 8; ++i)
            out[i] = b[i];
    }
}

__device__ __forceinline__ int mfp_coord_bias_n(int n) {
    if (n == 4)
        return 7;
    if (n == 3)
        return 4;
    return 2;
}

__device__ __forceinline__ unsigned mfp_encode_index_n(const float* p, int n) {
    const int bias = mfp_coord_bias_n(n);
    const int cmax = (1 << n) - 1;
    const unsigned mask = (1u << n) - 1u;
    const float fr = p[0] - floorf(p[0]);
    const unsigned coset = (fabsf(fr - 0.5f) < 0.1f) ? 1u : 0u;
    int w[8];
    for (int i = 0; i < 8; ++i)
        w[i] = coset ? (int)roundf(p[i] - 0.5f) : (int)roundf(p[i]);
    unsigned e[8];
    for (int i = 0; i < 8; ++i) {
        int v = w[i] + bias;
        if (v < 0)
            v = 0;
        if (v > cmax)
            v = cmax;
        e[i] = (unsigned)v;
    }
    unsigned sl = 0;
    for (int i = 0; i < 8; ++i)
        sl += e[i];
    if ((sl & 1u) != 0u) {
        if ((int)e[7] < cmax)
            e[7] += 1u;
        else
            e[7] -= 1u;
    }
    unsigned idx = 0;
    for (int i = 0; i < 7; ++i)
        idx |= (e[i] & mask) << (i * n);
    const int e7_high_bits = n - 1;
    const unsigned e7_high_mask = (1u << e7_high_bits) - 1u;
    idx |= ((e[7] >> 1) & e7_high_mask) << (7 * n);
    idx |= coset << (8 * n - 1);
    return idx;
}

__device__ __forceinline__ void mfp_decode_index_n(unsigned idx, int n, float* p) {
    const int bias = mfp_coord_bias_n(n);
    const unsigned mask = (1u << n) - 1u;
    const unsigned coset = (idx >> (8 * n - 1)) & 1u;
    unsigned e[8];
    unsigned sl = 0;
    for (int i = 0; i < 7; ++i) {
        e[i] = (idx >> (i * n)) & mask;
        sl += e[i];
    }
    const int e7_high_bits = n - 1;
    const unsigned e7_high_mask = (1u << e7_high_bits) - 1u;
    const unsigned e7_high = (idx >> (7 * n)) & e7_high_mask;
    const unsigned p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; ++i) {
        const float c = (float)((int)e[i] - bias);
        p[i] = coset ? c + 0.5f : c;
    }
}

__device__ __forceinline__ float mfp_e4m3_scale_decode(unsigned char byte) {
    const int exp = (byte >> 3) & 0xf;
    const unsigned mant = (unsigned)(byte & 0x7);
    if (exp == 0)
        return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xf && mant == 7)
        return 448.0f;
    return exp2f((float)(exp - 7)) * (1.0f + (float)mant * 0.125f);
}

__device__ __forceinline__ unsigned char mfp_e4m3_scale_encode_roundup(float s) {
    if (!(s > 0.0f))
        return 0x00;
    if (s >= 448.0f)
        return 0x7E;
    for (int code = 0; code <= 0x7E; ++code) {
        if (mfp_e4m3_scale_decode((unsigned char)code) >= s)
            return (unsigned char)code;
    }
    return 0x7E;
}

// One workgroup per row; 256 threads walk K as G256 tiles. Reconstructs via
// production row-scale (f16), E4M3 roundup block scale, closest-E8 + narrow
// n-byte codeword pack/decode (clamping path included).
__global__ __launch_bounds__(256) void eval_mfp_e8(
    const float* __restrict__ weights,
    int n_rows,
    int K,
    int n,
    float quant_step,
    float tail_threshold,
    DeviceResult* __restrict__ out)
{
    __shared__ float sh_red[256];
    __shared__ float sh_aux[256];
    __shared__ float sh_row_scale;
    __shared__ float sh_inv_row;
    __shared__ float sh_g32_scale[8];

    const int tid = (int)threadIdx.x;
    double sse = 0.0;
    double tail_sse = 0.0;
    double max_relative_sum = 0.0;
    unsigned long long count = 0;
    unsigned long long tail_count = 0;
    unsigned long long block_count = 0;

    for (int row = (int)blockIdx.x; row < n_rows; row += (int)gridDim.x) {
        const float* row_w = weights + (size_t)row * (size_t)K;

        float local_max = 0.0f;
        for (int i = tid; i < K; i += 256)
            local_max = fmaxf(local_max, fabsf(row_w[i]));
        sh_red[tid] = local_max;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride)
                sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + stride]);
            __syncthreads();
        }
        if (tid == 0) {
            const float row_max = sh_red[0];
            const float encode_scale = row_max > 0.0f ? row_max / 6.0f : 1.0f;
            // Production encodes blocks against the unrounded row scale, then
            // stores a truncating-fp16 row scale used by the decoder.
            sh_row_scale = f16_trunc_rt(encode_scale);
            sh_inv_row = row_max > 0.0f ? (1.0f / encode_scale) : 0.0f;
        }
        __syncthreads();
        const float row_scale = sh_row_scale;
        const float inv_row = sh_inv_row;

        for (int g256 = 0; g256 < K / 256; ++g256) {
            const float* blk = row_w + g256 * 256;
            const float w = blk[tid];
            const int g32 = tid / 32;
            const int lane32 = tid % 32;

            sh_red[tid] = fabsf(w);
            __syncthreads();
            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane32 < stride)
                    sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + stride]);
                __syncthreads();
            }
            if (lane32 == 0) {
                const float bmax = sh_red[g32 * 32];
                const float bmn = bmax * inv_row;
                const float s = bmn > 0.0f ? bmn / 6.0f : 0.0f;
                const unsigned char sb = mfp_e4m3_scale_encode_roundup(s);
                sh_g32_scale[g32] = mfp_e4m3_scale_decode(sb);
            }
            __syncthreads();

            const float bsf = sh_g32_scale[g32];
            const float inv_bs = bsf > 0.0f ? (1.0f / bsf) : 0.0f;
            const float scale = row_scale * bsf;
            const int e8g = lane32 / 8;
            const int e8i = lane32 % 8;
            const int base = g32 * 32 + e8g * 8;

            float v8[8];
            for (int i = 0; i < 8; ++i)
                v8[i] = blk[base + i] * inv_row * inv_bs;
            float u8[8];
            for (int i = 0; i < 8; ++i)
                u8[i] = v8[i] / quant_step;
            float p[8];
            mfp_closest_e8(u8, p);
            const unsigned idx = mfp_encode_index_n(p, n);
            mfp_decode_index_n(idx, n, p);
            const float reconstructed = p[e8i] * quant_step * scale;

            const float error = reconstructed - w;
            sse += (double)error * (double)error;
            ++count;
            if (fabsf(w) >= tail_threshold) {
                tail_sse += (double)error * (double)error;
                ++tail_count;
            }
            sh_red[tid] = fabsf(w);
            sh_aux[tid] = fabsf(error) / fmaxf(fabsf(w), 1.0e-30f);
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {
                if (tid < stride && sh_red[tid + stride] > sh_red[tid]) {
                    sh_red[tid] = sh_red[tid + stride];
                    sh_aux[tid] = sh_aux[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                max_relative_sum += (double)sh_aux[0];
                ++block_count;
            }
            __syncthreads();
        }
    }

    __shared__ double reduce_sse[256];
    __shared__ double reduce_tail[256];
    __shared__ double reduce_max_relative[256];
    __shared__ unsigned long long reduce_n[256];
    __shared__ unsigned long long reduce_tail_n[256];
    __shared__ unsigned long long reduce_blocks[256];
    reduce_sse[tid] = sse;
    reduce_tail[tid] = tail_sse;
    reduce_max_relative[tid] = max_relative_sum;
    reduce_n[tid] = count;
    reduce_tail_n[tid] = tail_count;
    reduce_blocks[tid] = block_count;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sse[tid] += reduce_sse[tid + stride];
            reduce_tail[tid] += reduce_tail[tid + stride];
            reduce_n[tid] += reduce_n[tid + stride];
            reduce_tail_n[tid] += reduce_tail_n[tid + stride];
            reduce_max_relative[tid] += reduce_max_relative[tid + stride];
            reduce_blocks[tid] += reduce_blocks[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&out->sse, reduce_sse[0]);
        atomicAdd(&out->tail_sse, reduce_tail[0]);
        atomicAdd(&out->max_relative_sum, reduce_max_relative[0]);
        atomicAdd(&out->n, reduce_n[0]);
        atomicAdd(&out->tail_n, reduce_tail_n[0]);
        atomicAdd(&out->blocks, reduce_blocks[0]);
    }
}


struct Config {
    std::string name;
    int mode;
    int nsub;
    int header_b;
    bool fp16_affine;
    float selector_p;
};

struct Result {
    std::string name;
    int header_b;
    int total_b;
    double mse;
    double tail_mse;
    double max_relative;
};

static std::vector<float> gaussian_codebook(int bits) {
    if (bits == 2)
        return {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
    if (bits == 3)
        return {-2.1520f, -1.3439f, -0.7560f, -0.2451f,
                 0.2451f, 0.7560f, 1.3439f, 2.1520f};
    return {-2.7326f, -2.0690f, -1.6180f, -1.2562f,
            -0.9423f, -0.6568f, -0.3880f, -0.1284f,
             0.1284f, 0.3880f, 0.6568f, 0.9423f,
             1.2562f, 1.6180f, 2.0690f, 2.7326f};
}

static std::vector<float> build_table(int bits) {
    const int nlev = 1 << bits;
    const std::vector<float> gl = gaussian_codebook(bits);
    std::vector<float> table(2 * MAX_LEVELS + NPROF * MAX_LEVELS, 0.0f);
    const float outer = std::fabs(gl.front());
    for (int level = 0; level < nlev; ++level) {
        table[level] = gl[level];
        table[MAX_LEVELS + level] = gl[level] / outer;
    }
    for (int profile = 0; profile < NPROF; ++profile) {
        const float gamma = 0.50f + 1.50f * (float)profile / (float)(NPROF - 1);
        for (int level = 0; level < nlev; ++level) {
            const float x = table[MAX_LEVELS + level];
            table[2 * MAX_LEVELS + profile * MAX_LEVELS + level] =
                std::copysign(std::pow(std::fabs(x), gamma), x);
        }
    }
    return table;
}

// Legacy payload packing: 2/3/4 = bits*32 B; 5-bit = 8 vals/5 B → 160;
// 6-bit = 4 vals/3 B → 192. V2 keeps the same payload; only the header encoding
// changes (still 8 B: dual G128 fp16 s/z). MQ5V2=168, MQ6V2=200 B/G256.
static int legacy_payload_bytes(int bits) {
    if (bits == 5)
        return 160; // 32 chunks × 5 B
    if (bits == 6)
        return 192; // 64 chunks × 3 B
    return bits * 32;
}

static std::vector<Config> configs_for(int bits) {
    const int nlev = 1 << bits;
    std::vector<Config> configs = {
        {"genesis affine G256 f32", MODE_AFFINE, 1, 8, false, 0.0f},
        {"v2 affine G128x2 fp16", MODE_AFFINE, 2, 8, true, 0.0f},
        {"GL RMS G256 fp16", MODE_GL_RMS, 1, 2, false, 0.0f},
        {"GL RMS G128x2 fp16", MODE_GL_RMS, 2, 4, false, 0.0f},
        {"GL RMS G64x4 fp16", MODE_GL_RMS, 4, 8, false, 0.0f},
        {"anchored GL max G256 fp16", MODE_ANCHORED_MAX, 1, 2, false, 0.0f},
        {"anchored GL max G128x2 fp16", MODE_ANCHORED_MAX, 2, 4, false, 0.0f},
        {"anchored GL max G64x4 fp16", MODE_ANCHORED_MAX, 4, 8, false, 0.0f},
        {"selector max G256 p0 6b", MODE_SELECTOR_MAX, 1, 3, false, 0.0f},
        {"selector max G256 p1 6b", MODE_SELECTOR_MAX, 1, 3, false, 1.0f},
        {"selector max G256 p2 6b", MODE_SELECTOR_MAX, 1, 3, false, 2.0f},
        {"selector max G128x2 p0 6b", MODE_SELECTOR_MAX, 2, 6, false, 0.0f},
        {"selector max G128x2 p1 6b", MODE_SELECTOR_MAX, 2, 6, false, 1.0f},
        {"selector max G128x2 p2 6b", MODE_SELECTOR_MAX, 2, 6, false, 2.0f},
        {"per-block Lloyd G256 fp16", MODE_LLOYD, 1, 2 * nlev, false, 0.0f},
        {"per-half Lloyd G128x2 fp16", MODE_LLOYD, 2, 4 * nlev, false, 0.0f},
    };
    // MQ2 72-B ceiling only: endpoint-anchored Lloyd (4×fp16 header = 8 B).
    // Same wire budget as unconstrained per-block Lloyd / MQ2V2 affine header.
    if (bits == 2) {
        configs.push_back(
            {"anchored Lloyd G256 fp16", MODE_LLOYD_ANCHORED, 1, 8, false, 0.0f});
    }
    return configs;
}


// Neutral-size 5/6-bit arms only: legacy G256 f32 header vs MQ4V2 dual-G128 fp16.
static std::vector<Config> configs_highbit(int bits) {
    char v1[32], v2[32];
    std::snprintf(v1, sizeof(v1), "MQ%d-v1", bits);
    std::snprintf(v2, sizeof(v2), "MQ%dV2", bits);
    return {
        {v1, MODE_AFFINE_PROD, 1, 8, false, 0.0f},
        {v2, MODE_AFFINE_PROD, 2, 8, true, 0.0f},
    };
}

static bool dominates(const Result& a, const Result& b) {
    const bool no_worse = a.total_b <= b.total_b && a.mse <= b.mse &&
                          a.tail_mse <= b.tail_mse;
    const bool strictly_better = a.total_b < b.total_b || a.mse < b.mse ||
                                 a.tail_mse < b.tail_mse;
    return no_worse && strictly_better;
}

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : ".codeinsight+research/quant-design/fixtures/sweep_G.bin";
    const float tail_threshold = argc > 2 ? std::atof(argv[2]) : 2.869166e-02f;
    const int K = argc > 3 ? std::atoi(argv[3]) : 17408;
    FILE* file = std::fopen(weights_path, "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", weights_path);
        return 1;
    }
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (bytes <= 0 || bytes % (256 * (long)sizeof(float)) != 0) {
        std::fprintf(stderr, "invalid post-FWHT fixture size: %ld\n", bytes);
        return 1;
    }
    const int nblk = (int)(bytes / (256 * sizeof(float)));
    if (K <= 0 || (K % 256) != 0) {
        std::fprintf(stderr, "K must be a positive multiple of 256, got %d\n", K);
        return 1;
    }
    const int g256_per_row = K / 256;
    if ((nblk % g256_per_row) != 0) {
        std::fprintf(stderr,
                     "fixture blocks %d not divisible by K/256=%d (row boundary unknown)\n",
                     nblk, g256_per_row);
        return 1;
    }
    const int n_rows = nblk / g256_per_row;
    std::vector<float> host_weights((size_t)nblk * 256);
    if (std::fread(host_weights.data(), 1, (size_t)bytes, file) != (size_t)bytes) {
        std::fprintf(stderr, "short read\n");
        return 1;
    }
    std::fclose(file);

    float* device_weights = nullptr;
    float* device_table = nullptr;
    DeviceResult* device_result = nullptr;
    CK(hipMalloc(&device_weights, host_weights.size() * sizeof(float)));
    CK(hipMalloc(&device_table, (2 * MAX_LEVELS + NPROF * MAX_LEVELS) * sizeof(float)));
    CK(hipMalloc(&device_result, sizeof(DeviceResult)));
    CK(hipMemcpy(device_weights, host_weights.data(), host_weights.size() * sizeof(float), hipMemcpyHostToDevice));

    std::printf("MQ/HFQ matched-header optimizer: %d blocks (%d rows x K=%d), tail threshold %.6e\n",
                nblk, n_rows, K, tail_threshold);
    std::printf("Every matched genesis budget is payload + 8 header bytes.\n");
    std::printf("MFP{2,3,4}G32E8 production: 16 B/row hdr + (1+4n) B/G32; "
                "centers n2=2/n3=4/n4=7; steps 3.8/1.8/0.88; E4M3 roundup + closest E8.\n");
    std::printf("MQ5/MQ6 hypothesis: legacy affine G256 f32 (v1) vs dual-G128 fp16 "
                "header (V2, MQ4V2 contract); payload 5b=8vals/5B, 6b=4vals/3B; "
                "MQ5V2=168 B/G256, MQ6V2=200 B/G256.\n");
    for (int bits = 2; bits <= 4; ++bits) {
        const int payload_b = legacy_payload_bytes(bits);
        const int matched_b = payload_b + 8;
        const std::vector<float> table = build_table(bits);
        CK(hipMemcpy(device_table, table.data(), table.size() * sizeof(float), hipMemcpyHostToDevice));
        const std::vector<Config> configs = configs_for(bits);
        std::vector<Result> results;
        std::printf("\n=== MQ%d/HFQ%d: matched ceiling %d B/group (%.4f bpw) ===\n",
                    bits, bits, matched_b, matched_b * 8.0 / 256.0);
        std::printf("  %-34s %4s %5s %7s   %-13s %-13s %-12s\n",
                    "candidate", "hdr", "total", "bpw", "overall MSE",
                    "tail-1% MSE", "max-rel mean");
        std::printf("  --------------------------------------------------------------------------------------------------------------\n");
        for (const Config& config : configs) {
            const DeviceResult zero{};
            CK(hipMemcpy(device_result, &zero, sizeof(zero), hipMemcpyHostToDevice));
            hipLaunchKernelGGL(eval_candidate, dim3(std::min(nblk, 2048)), dim3(256), 0, 0,
                               device_weights, nblk, device_table, tail_threshold,
                               bits, config.mode, config.nsub,
                               config.fp16_affine ? 1 : 0, config.selector_p,
                               device_result);
            CK(hipDeviceSynchronize());
            DeviceResult result{};
            CK(hipMemcpy(&result, device_result, sizeof(result), hipMemcpyDeviceToHost));
            const int total_b = payload_b + config.header_b;
            const double mse = result.sse / (double)result.n;
            const double tail_mse = result.tail_sse / (double)result.tail_n;
            const double max_relative = result.max_relative_sum / (double)result.blocks;
            results.push_back({config.name, config.header_b, total_b, mse, tail_mse, max_relative});
            std::printf("  %-34s %4d %5d %7.4f   %.4e    %.4e    %10.4e%s\n",
                        config.name.c_str(), config.header_b, total_b,
                        total_b * 8.0 / 256.0, mse, tail_mse, max_relative,
                        total_b <= matched_b ? "" : "  [over budget control]");
        }

        // Production MFP{n}G32E8 arm — exact wire layout / decode semantics.
        {
            const int n = bits;
            const float quant_step = n == 2 ? 3.8f : (n == 3 ? 1.8f : 0.88f);
            const DeviceResult zero_mfp{};
            CK(hipMemcpy(device_result, &zero_mfp, sizeof(zero_mfp), hipMemcpyHostToDevice));
            hipLaunchKernelGGL(eval_mfp_e8, dim3(std::min(n_rows, 2048)), dim3(256), 0, 0,
                               device_weights, n_rows, K, n, quant_step, tail_threshold,
                               device_result);
            CK(hipDeviceSynchronize());
            DeviceResult mfp_result{};
            CK(hipMemcpy(&mfp_result, device_result, sizeof(mfp_result), hipMemcpyDeviceToHost));
            const int g32_per_row = K / 32;
            const int row_bytes = 16 + g32_per_row * (1 + 4 * n);
            const double eff_per_g256 = (double)row_bytes * 256.0 / (double)K;
            // Pareto admission is conservative: any fractional row-header
            // overhead exceeds the integer matched ceiling.
            const int total_b = (int)std::ceil(eff_per_g256);
            const int header_b = total_b - payload_b;
            const double mse = mfp_result.sse / (double)mfp_result.n;
            const double tail_mse = mfp_result.tail_sse / (double)mfp_result.tail_n;
            const double max_relative =
                mfp_result.max_relative_sum / (double)mfp_result.blocks;
            char name[48];
            std::snprintf(name, sizeof(name), "MFP%dG32E8 prod +16B/row", n);
            results.push_back({name, header_b, total_b, mse, tail_mse, max_relative});
            const double bpw_k = row_bytes * 8.0 / (double)K;
            std::printf("  %-34s %4d %5d %7.4f   %.4e    %.4e    %10.4e"
                        "  [row %d B @K=%d -> %.4f bpw; 1+4*%d B/G32]%s\n",
                        name, header_b, total_b, eff_per_g256 * 8.0 / 256.0, mse, tail_mse,
                        max_relative, row_bytes, K, bpw_k, n,
                        total_b <= matched_b ? "" : "  [over budget control]");
        }

        std::printf("\n  Pareto frontier at <= %d B (bytes, overall MSE, tail MSE):\n", matched_b);
        for (const Result& candidate : results) {
            if (candidate.total_b > matched_b)
                continue;
            bool dominated = false;
            for (const Result& other : results) {
                if (&candidate != &other && other.total_b <= matched_b && dominates(other, candidate)) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                std::printf("    %-34s %3d B  %.4e  %.4e  %.4e\n",
                            candidate.name.c_str(), candidate.total_b,
                            candidate.mse, candidate.tail_mse, candidate.max_relative);
            }
        }

        std::vector<const Result*> tail_safe;
        const double genesis_tail = results.front().tail_mse;
        for (const Result& candidate : results) {
            if (candidate.total_b <= matched_b && candidate.tail_mse <= genesis_tail)
                tail_safe.push_back(&candidate);
        }
        std::sort(tail_safe.begin(), tail_safe.end(),
                  [](const Result* a, const Result* b) { return a->mse < b->mse; });
        std::printf("\n  Tail-safe candidates at <= %d B (tail <= genesis), best overall MSE first:\n",
                    matched_b);
        for (const Result* candidate : tail_safe) {
            std::printf("    %-34s %3d B  %.4e  %.4e  %.4e\n",
                        candidate->name.c_str(), candidate->total_b,
                        candidate->mse, candidate->tail_mse, candidate->max_relative);
        }
    }

    // ── 5/6-bit neutral-size affine hypothesis (v1 f32 G256 vs V2 fp16 G128×2) ──
    for (int bits = 5; bits <= 6; ++bits) {
        const int payload_b = legacy_payload_bytes(bits);
        const int matched_b = payload_b + 8; // MQ5=168, MQ6=200
        const std::vector<Config> configs = configs_highbit(bits);
        std::printf("\n=== MQ%d hypothesis: %d B/group (%.4f bpw) — v1 f32 vs V2 fp16 ===\n",
                    bits, matched_b, matched_b * 8.0 / 256.0);
        std::printf("  %-34s %4s %5s %7s   %-13s %-13s %-12s\n",
                    "candidate", "hdr", "total", "bpw", "overall MSE",
                    "tail-1% MSE", "max-rel mean");
        std::printf("  --------------------------------------------------------------------------------------------------------------\n");
        // Affine path does not read the codebook table; pass the bits-4 table as a dummy.
        const std::vector<float> table = build_table(4);
        CK(hipMemcpy(device_table, table.data(), table.size() * sizeof(float),
                     hipMemcpyHostToDevice));
        for (const Config& config : configs) {
            const DeviceResult zero{};
            CK(hipMemcpy(device_result, &zero, sizeof(zero), hipMemcpyHostToDevice));
            hipLaunchKernelGGL(eval_candidate, dim3(std::min(nblk, 2048)), dim3(256), 0, 0,
                               device_weights, nblk, device_table, tail_threshold,
                               bits, config.mode, config.nsub,
                               config.fp16_affine ? 1 : 0, config.selector_p,
                               device_result);
            CK(hipDeviceSynchronize());
            DeviceResult result{};
            CK(hipMemcpy(&result, device_result, sizeof(result), hipMemcpyDeviceToHost));
            const int total_b = payload_b + config.header_b;
            const double mse = result.sse / (double)result.n;
            const double tail_mse = result.tail_sse / (double)result.tail_n;
            const double max_relative =
                result.max_relative_sum / (double)result.blocks;
            std::printf("  %-34s %4d %5d %7.4f   %.4e    %.4e    %10.4e\n",
                        config.name.c_str(), config.header_b, total_b,
                        total_b * 8.0 / 256.0, mse, tail_mse, max_relative);
        }
        std::printf("  (V2 header: [0..2) s0 [2..4) z0 [4..6) s1 [6..8) z1 f16; "
                    "payload +8; degenerate half → scale=0, q=0 like MQ4V2)\n");
    }

    CK(hipFree(device_result));
    CK(hipFree(device_table));
    CK(hipFree(device_weights));
    return 0;
}
