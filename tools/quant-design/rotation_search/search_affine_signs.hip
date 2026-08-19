// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// search_affine_signs.hip — standalone gfx1201 HIP tool that searches a single
// global D2 sign table for production MQ4V2 and hypothetical matched-header
// 2-bit/3-bit V2 research layouts (two G128 fp16 scale+zero pairs).
//
// Runtime contract (matches the engine):
//   R = D2 · H · D1 with one global 256-sign D1 (seed 42), one global 256-sign
//   D2 (canonical seed 1042), one unnormalized 8-stage FWHT-256, then ×1/16.
//   Candidate D2 tables MUST preserve that op count — only the 256 signs change.
//
// Production header encoding (exact, crates/hipfire-quantize/src/float16.rs):
//   f32→f16 is Hipfire TRUNCATION, NOT IEEE RNE / __float2half_rn.
//   st,z = f16_to_f32(f32_to_f16_trunc(.));
//   q = clamp(floor((v-z)/st+0.5), 0, qmax);   // floor+0.5, not rint
//
// Scores (F=1 fixture for a single INPUT; equal weight over bits n∈{2,3,4}):
//   U = D1→FWHT→×1/16 (no D2). |D2·U|=|U| so masks/energies are D2-invariant.
//   energy_b = Σ_i U[b,i]²
//   eps      = 2^{-24} · median_b(energy_b)
//   tail mask: fixed-count top ceil(1%) of |U| (pre-D2); ties by (block,index) asc
//
//   r_all_n(d) = mean_b[ SSE_n(d,b) / (energy_b + eps) ]
//   J_all(d)   = (1/3) Σ_n r_all_n(d) / r_all_n(canonical)
//   J_tail(d)  = (1/3) Σ_n tail_SSE_n(d) / tail_SSE_n(canonical)   // PRIMARY
//
// Selection (lexicographic, lower better):
//   1. feasible first (J_all <= 1.000 required of final pick)
//   2. minimize J_tail
//   3. minimize J_all
//   4. minimize worst_n max(rel_tail_n, rel_all_n)
//   5. lex-smaller packed 256 sign bits (8×u32 LE)
// Internal diagnostic scalar: J_scalar = J_tail + 100*max(0,J_all-1)^2
//
// Also accumulates plain SSE and magnitude-weighted SSE (weight 1+|u|/RMS).
//
// Build:
//   hipcc --offload-arch=gfx1201 -O3 search_affine_signs.hip -o search_affine_signs
//
// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
//   search_affine_signs --help
//   search_affine_signs random INPUT N RNG_SEED OUT_PREFIX
//   search_affine_signs coord  INPUT SIGNS PASSES OUT_PREFIX
//   search_affine_signs eval   INPUT SIGNS OUT_PREFIX
//
// INPUT: headerless LE f32 [nblocks,256] raw unrotated groups.
// SIGNS: 256 × f32 LE exact ±1.0 (candidate D2 only).
//
// ranked.csv:
//   rank,seed,label,feasible,J_tail,J_all,J_scalar,worst_rel,obj_wsse,
//   sse2,sse3,sse4,wsse2,wsse3,wsse4,tsse2,tsse3,tsse4,
//   r2,r3,r4,rel_all2,rel_all3,rel_all4,rel_tail2,rel_tail3,rel_tail4,
//   mse2,mse3,mse4,wmse2,wmse3,wmse4,tail_mse2,tail_mse3,tail_mse4,
//   signs_u32_0..signs_u32_7
// coord.csv:
//   pass,flip_idx,accepted,feasible,J_tail,J_all,delta_J_tail,worst_rel,obj_wsse,
//   sse*,wsse*,tsse*,r*,rel_all*,rel_tail*   (flip_idx=-1 end-of-pass)
// eval.csv:
//   who,bits,mse,wmse,r_nmse,rel_all,tail_mse,rel_tail,abs_p99,abs_p99_9,abs_max
// *.signs: 256×f32 LE ±1.0
// *.u32.txt: eight hex u32 (bit i set iff signs[i]==+1)
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static constexpr int G = 256;
static constexpr int HALF = 128;
static constexpr int NBITS = 3;
static constexpr int BITS[NBITS] = {2, 3, 4};
static constexpr int QMAX[NBITS] = {3, 7, 15};
static constexpr uint32_t D1_SEED = 42u;
static constexpr uint32_t D2_CANON_SEED = 1042u;
static constexpr float FWHT_SCALE = 0.0625f;
static constexpr double EPS_SCALE = 0x1.0p-24;

#define CK(x)                                                                 \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                         hipGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static void die(const char* msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

static void usage(const char* argv0, int code) {
    std::fprintf(code ? stderr : stdout,
R"(search_affine_signs — global D2 search for MQ4V2 plus hypothetical 2/3-bit V2 layouts

USAGE
  %s --help
  %s random INPUT N RNG_SEED OUT_PREFIX
  %s coord  INPUT SIGNS PASSES OUT_PREFIX
  %s eval   INPUT SIGNS OUT_PREFIX

INPUT
  Headerless little-endian f32 [nblocks,256] raw unrotated weight groups.
  Size must be a positive multiple of 1024 bytes. Non-finite values rejected.

SIGNS
  Exactly 256 little-endian f32 values, each ±1. D2 only (D1 always seed 42).
  Candidate signs must not alter FWHT operation count vs canonical.

MODES
  random  Canonical D2 (seed 1042) + N LCG-seeded random D2 tables.
          Writes ranked.csv, best.signs, best.u32.txt under OUT_PREFIX.

  coord   Coordinate descent from SIGNS for PASSES sweeps of 256 single-bit
          flips. Accepts only strict total-order improvement.
          Writes coord.csv, final.signs, final.u32.txt.

  eval    Compare SIGNS to canonical. Writes eval.csv, eval.u32.txt.

TRANSFORMS AND CODEC
  Pre-rotate once: D1(seed 42) → unnormalized 8-stage FWHT-256 → ×1/16.
  Search applies candidate D2 only. Per G256: two independent contiguous G128
  asymmetric affine grids for bits∈{2,3,4}:
    lo,hi = min/max of half; step=(hi-lo)/(2^b-1);
    st,z = f16_to_f32(f32_to_f16_TRUNC(.))  // float16.rs truncation, NOT RNE
    q = clamp(floor((v-z)/st + 0.5), 0, 2^b-1);  recon = q*st + z
  Degenerate half (hi==lo or st==0): q=0, recon=z.

SCORES (mq4-v2.md §5: tail-1%% MSE is the KLD-supported metric)
  Tail mask: fixed-count top ceil(1%%) of |U| (pre-D2; D2-invariant);
  equal-magnitude ties broken by (block, index) ascending so membership is unique.
  J_tail = (1/3) Σ_n tail_SSE_n(d)/tail_SSE_n(canonical)     // PRIMARY
  J_all  = (1/3) Σ_n mean_b[SSE_n(b)/(E_b+eps)] / same(canon) // SECONDARY
  Feasible iff J_all <= 1.000. Among feasible: min J_tail, then J_all,
  then worst per-bit relative, then lex-smaller packed sign bits.
  Also reports SSE and magnitude-weighted SSE (weight 1+|u|/RMS).

OUTPUT
  ranked.csv / coord.csv / eval.csv schemas documented in source header.
  *.signs: 256×f32 LE ±1.0; *.u32.txt: 8 hex u32 words (bit i set iff +1)

BUILD
  hipcc --offload-arch=gfx1201 -O3 search_affine_signs.hip -o search_affine_signs
)",
                 argv0, argv0, argv0, argv0);
    std::exit(code);
}

static uint32_t lcg_next(uint32_t& state) {
    state = (state * 1103515245u + 12345u) & 0x7fffffffu;
    return state;
}

static void gen_fwht_signs(uint32_t seed, float* signs, int n) {
    uint32_t state = seed;
    for (int i = 0; i < n; ++i) {
        uint32_t s = lcg_next(state);
        signs[i] = ((s >> 16) & 1u) ? 1.0f : -1.0f;
    }
}

// Bit-identical to crates/hipfire-quantize/src/float16.rs f32_to_f16.
// Truncation toward zero in the mantissa (NOT IEEE round-to-nearest-even).
__host__ __device__ static inline uint16_t f32_to_f16_trunc(float val) {
    uint32_t bits;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    bits = __float_as_uint(val);
#else
    std::memcpy(&bits, &val, sizeof(bits));
#endif
    const uint32_t sign = (bits >> 31) & 1u;
    const int32_t exp = (int32_t)((bits >> 23) & 0xFFu);
    const uint32_t frac = bits & 0x7FFFFFu;
    if (exp == 0xFF) {
        // Inf / NaN — quiet NaN if frac nonzero (matches Rust).
        const uint32_t f16_frac = (frac == 0u) ? 0u : ((frac >> 13) | 1u);
        return (uint16_t)((sign << 15) | (0x1Fu << 10) | f16_frac);
    }
    const int32_t new_exp = exp - 127 + 15;
    if (new_exp >= 31) {
        // Overflow → Inf (truncate path saturates to Inf, matching Rust).
        return (uint16_t)((sign << 15) | (0x1Fu << 10));
    }
    if (new_exp <= 0) {
        if (new_exp < -10) {
            return (uint16_t)(sign << 15); // underflow → signed zero
        }
        // Subnormal: shift with truncation (no round-up).
        const uint32_t f = frac | 0x800000u;
        const uint32_t shift = (uint32_t)(1 - new_exp + 13);
        return (uint16_t)((sign << 15) | (f >> shift));
    }
    // Normal: drop 13 LSBs (truncation).
    return (uint16_t)((sign << 15) | ((uint32_t)new_exp << 10) | (frac >> 13));
}

// IEEE binary16 bits → f32 (matches half crate / hardware cvt for finite values).
__host__ __device__ static inline float f16_to_f32_bits(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            int e = -1;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(out);
#else
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
#endif
}

// Production round-trip used on scale/zero headers.
__host__ __device__ static inline float f16_trunc_rt(float x) {
    return f16_to_f32_bits(f32_to_f16_trunc(x));
}

static void fwht_unnormalized_256(float* x) {
    for (int stride = 1; stride < G; stride <<= 1) {
        for (int i = 0; i < G; i += stride * 2) {
            for (int j = 0; j < stride; ++j) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
}

static void pre_rotate_block(float* blk, const float* d1) {
    for (int i = 0; i < G; ++i) blk[i] *= d1[i];
    fwht_unnormalized_256(blk);
    for (int i = 0; i < G; ++i) blk[i] *= FWHT_SCALE;
}

static std::vector<float> load_corpus(const char* path, int& nblocks_out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open INPUT %s\n", path);
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_END) != 0) die("fseek INPUT failed");
    long bytes = std::ftell(f);
    if (bytes < 0) die("ftell INPUT failed");
    if (std::fseek(f, 0, SEEK_SET) != 0) die("fseek INPUT rewind failed");
    if (bytes == 0) die("INPUT is empty");
    if (bytes % (long)(G * (int)sizeof(float)) != 0) {
        std::fprintf(stderr,
                     "error: INPUT size %ld is not a multiple of %d bytes\n",
                     bytes, G * (int)sizeof(float));
        std::exit(1);
    }
    const int nblocks = (int)(bytes / (long)(G * (int)sizeof(float)));
    if (nblocks <= 0) die("INPUT has zero blocks");
    std::vector<float> data((size_t)nblocks * G);
    if (std::fread(data.data(), 1, (size_t)bytes, f) != (size_t)bytes) {
        std::fclose(f);
        die("short read on INPUT");
    }
    std::fclose(f);
    for (size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            std::fprintf(stderr, "error: non-finite value at element %zu\n", i);
            std::exit(1);
        }
    }
    nblocks_out = nblocks;
    return data;
}

static std::vector<float> load_signs(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open SIGNS %s\n", path);
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_END) != 0) die("fseek SIGNS failed");
    long bytes = std::ftell(f);
    if (bytes != (long)(G * (int)sizeof(float))) {
        std::fprintf(stderr, "error: SIGNS size %ld, expected %d\n", bytes,
                     G * (int)sizeof(float));
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_SET) != 0) die("fseek SIGNS rewind failed");
    std::vector<float> s(G);
    if (std::fread(s.data(), sizeof(float), G, f) != (size_t)G) {
        std::fclose(f);
        die("short read on SIGNS");
    }
    std::fclose(f);
    for (int i = 0; i < G; ++i) {
        if (!(s[i] == 1.0f || s[i] == -1.0f)) {
            std::fprintf(stderr, "error: SIGNS[%d]=%g not ±1.0f\n", i,
                         (double)s[i]);
            std::exit(1);
        }
    }
    return s;
}

static void atomic_write_bytes(const std::string& path, const void* data,
                               size_t nbytes) {
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s\n", tmp.c_str());
        std::exit(1);
    }
    if (std::fwrite(data, 1, nbytes, f) != nbytes) {
        std::fclose(f);
        std::remove(tmp.c_str());
        die("short write");
    }
    if (std::fflush(f) != 0 || std::fclose(f) != 0) {
        std::remove(tmp.c_str());
        die("flush/close failed");
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        FILE* out = std::fopen(path.c_str(), "wb");
        FILE* in = std::fopen(tmp.c_str(), "rb");
        if (!out || !in) {
            if (out) std::fclose(out);
            if (in) std::fclose(in);
            std::remove(tmp.c_str());
            die("rename/fallback failed");
        }
        std::vector<char> buf(1 << 20);
        size_t n;
        while ((n = std::fread(buf.data(), 1, buf.size(), in)) > 0) {
            if (std::fwrite(buf.data(), 1, n, out) != n) {
                std::fclose(in);
                std::fclose(out);
                std::remove(tmp.c_str());
                die("fallback write failed");
            }
        }
        std::fclose(in);
        std::fclose(out);
        std::remove(tmp.c_str());
    }
}

static void write_signs(const std::string& path, const float* signs) {
    atomic_write_bytes(path, signs, (size_t)G * sizeof(float));
}

static void atomic_write_text(const std::string& path, const std::string& text) {
    atomic_write_bytes(path, text.data(), text.size());
}

static void pack_signs_u32(const float* signs, uint32_t out[8]) {
    for (int w = 0; w < 8; ++w) {
        uint32_t v = 0;
        for (int b = 0; b < 32; ++b)
            if (signs[w * 32 + b] > 0.0f) v |= (1u << b);
        out[w] = v;
    }
}

static void write_u32_txt(const std::string& path, const float* signs) {
    uint32_t w[8];
    pack_signs_u32(signs, w);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%08x %08x %08x %08x %08x %08x %08x %08x\n", w[0], w[1], w[2],
                  w[3], w[4], w[5], w[6], w[7]);
    atomic_write_text(path, std::string(buf));
}

static int cmp_u32_pack(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 0; i < 8; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// ── device kernel ───────────────────────────────────────────────────────────

__global__ __launch_bounds__(256) void eval_affine_batch(
    const float* __restrict__ U, const float* __restrict__ energy,
    const uint8_t* __restrict__ tail_mask, int nblocks,
    const float* __restrict__ d2_batch, int ncand, float inv_rms, double eps,
    double* __restrict__ out_sse, double* __restrict__ out_wsse,
    double* __restrict__ out_tsse, double* __restrict__ out_rsum) {
    const int tid = threadIdx.x;
    const int cand = (int)blockIdx.y;
    if (cand >= ncand) return;

    __shared__ float sh_u[G];
    __shared__ float sh_min[G];
    __shared__ float sh_max[G];
    __shared__ double sh_red[NBITS][G];

    const float* d2 = d2_batch + (size_t)cand * G;
    const float s = d2[tid];

    double loc_sse[NBITS] = {};
    double loc_wsse[NBITS] = {};
    double loc_tsse[NBITS] = {};
    double loc_rsum[NBITS] = {};

    for (int blk = (int)blockIdx.x; blk < nblocks; blk += (int)gridDim.x) {
        const size_t gi = (size_t)blk * G + (size_t)tid;
        const float u = U[gi] * s;
        const int in_tail = (int)tail_mask[gi];
        sh_u[tid] = u;
        sh_min[tid] = u;
        sh_max[tid] = u;
        __syncthreads();

        const int half = tid >> 7;
        const int base = half << 7;
        const int lane = tid & 127;
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) {
                sh_min[base + lane] =
                    fminf(sh_min[base + lane], sh_min[base + lane + stride]);
                sh_max[base + lane] =
                    fmaxf(sh_max[base + lane], sh_max[base + lane + stride]);
            }
            __syncthreads();
        }

        const float lo = sh_min[base];
        const float hi = sh_max[base];
        const float wgt = 1.0f + fabsf(u) * inv_rms;
        const double e_blk = (double)energy[blk] + eps;

        double e2_local[NBITS];
#pragma unroll
        for (int bi = 0; bi < NBITS; ++bi) {
            const float levels = (float)QMAX[bi];
            float st = 0.0f;
            float z = lo;
            bool degenerate = (hi == lo);
            if (!degenerate) {
                st = f16_trunc_rt((hi - lo) / levels);
                z = f16_trunc_rt(lo);
                if (st == 0.0f) degenerate = true;
            } else {
                z = f16_trunc_rt(lo);
            }

            float recon;
            if (degenerate) {
                recon = z;
            } else {
                // Production: floor((v-z)/st + 0.5), not rint.
                float q = floorf((u - z) / st + 0.5f);
                q = fminf(fmaxf(q, 0.0f), levels);
                recon = q * st + z;
            }
            const double e = (double)recon - (double)u;
            const double e2 = e * e;
            e2_local[bi] = e2;
            loc_sse[bi] += e2;
            loc_wsse[bi] += e2 * (double)wgt;
            if (in_tail) loc_tsse[bi] += e2;
        }

#pragma unroll
        for (int bi = 0; bi < NBITS; ++bi) sh_red[bi][tid] = e2_local[bi];
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int bi = 0; bi < NBITS; ++bi)
                    sh_red[bi][tid] += sh_red[bi][tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
#pragma unroll
            for (int bi = 0; bi < NBITS; ++bi)
                loc_rsum[bi] += sh_red[bi][0] / e_blk;
        }
        __syncthreads();
    }

    // Reduce sse
#pragma unroll
    for (int bi = 0; bi < NBITS; ++bi) sh_red[bi][tid] = loc_sse[bi];
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int bi = 0; bi < NBITS; ++bi)
                sh_red[bi][tid] += sh_red[bi][tid + stride];
        }
        __syncthreads();
    }
    double red_sse[NBITS];
    if (tid == 0) {
#pragma unroll
        for (int bi = 0; bi < NBITS; ++bi) red_sse[bi] = sh_red[bi][0];
    }
    __syncthreads();

#pragma unroll
    for (int bi = 0; bi < NBITS; ++bi) sh_red[bi][tid] = loc_wsse[bi];
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int bi = 0; bi < NBITS; ++bi)
                sh_red[bi][tid] += sh_red[bi][tid + stride];
        }
        __syncthreads();
    }
    double red_wsse[NBITS];
    if (tid == 0) {
#pragma unroll
        for (int bi = 0; bi < NBITS; ++bi) red_wsse[bi] = sh_red[bi][0];
    }
    __syncthreads();

#pragma unroll
    for (int bi = 0; bi < NBITS; ++bi) sh_red[bi][tid] = loc_tsse[bi];
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int bi = 0; bi < NBITS; ++bi)
                sh_red[bi][tid] += sh_red[bi][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int bi = 0; bi < NBITS; ++bi) {
            atomicAdd(out_sse + cand * NBITS + bi, red_sse[bi]);
            atomicAdd(out_wsse + cand * NBITS + bi, red_wsse[bi]);
            atomicAdd(out_tsse + cand * NBITS + bi, sh_red[bi][0]);
            atomicAdd(out_rsum + cand * NBITS + bi, loc_rsum[bi]);
        }
    }
}

// ── metrics / GPU ctx ───────────────────────────────────────────────────────

struct CandMetrics {
    double sse[NBITS]{};
    double wsse[NBITS]{};
    double tsse[NBITS]{};
    double r[NBITS]{};
    double rel_all[NBITS]{};
    double rel_tail[NBITS]{};
    double J_tail = 0.0;
    double J_all = 0.0;
    double J_scalar = 0.0;
    double worst_rel = 0.0;
    int feasible = 0;

    double obj_wsse() const { return wsse[0] + wsse[1] + wsse[2]; }

    void finalize(const CandMetrics& canon) {
        J_tail = 0.0;
        J_all = 0.0;
        worst_rel = 0.0;
        for (int bi = 0; bi < NBITS; ++bi) {
            const double ra = canon.r[bi] > 0.0 ? canon.r[bi] : 1e-300;
            const double ta = canon.tsse[bi] > 0.0 ? canon.tsse[bi] : 1e-300;
            rel_all[bi] = r[bi] / ra;
            rel_tail[bi] = tsse[bi] / ta;
            J_all += rel_all[bi];
            J_tail += rel_tail[bi];
            const double w =
                rel_all[bi] > rel_tail[bi] ? rel_all[bi] : rel_tail[bi];
            if (w > worst_rel) worst_rel = w;
        }
        J_all *= (1.0 / (double)NBITS);
        J_tail *= (1.0 / (double)NBITS);
        feasible = (J_all <= 1.0) ? 1 : 0;
        const double over = J_all > 1.0 ? (J_all - 1.0) : 0.0;
        J_scalar = J_tail + 100.0 * over * over;
    }
};

static bool better_than(const CandMetrics& a, const uint32_t ap[8],
                        const CandMetrics& b, const uint32_t bp[8]) {
    if (a.feasible != b.feasible) return a.feasible > b.feasible;
    if (a.J_tail < b.J_tail) return true;
    if (a.J_tail > b.J_tail) return false;
    if (a.J_all < b.J_all) return true;
    if (a.J_all > b.J_all) return false;
    if (a.worst_rel < b.worst_rel) return true;
    if (a.worst_rel > b.worst_rel) return false;
    return cmp_u32_pack(ap, bp) < 0;
}

struct GpuCtx {
    float* dU = nullptr;
    float* dEnergy = nullptr;
    uint8_t* dTail = nullptr;
    double* dSse = nullptr;
    double* dWsse = nullptr;
    double* dTsse = nullptr;
    double* dRsum = nullptr;
    float* dD2 = nullptr;
    int nblocks = 0;
    int max_cand = 0;
    float inv_rms = 0.0f;
    double eps = 0.0;
    float tail_thr = 0.0f;
    size_t tail_count = 0;
    int grid_x = 0;
    std::vector<float> hU;
    std::vector<float> hEnergy;
    std::vector<uint8_t> hTail;
};

static void gpu_init(GpuCtx& ctx, const std::vector<float>& U, int nblocks,
                     int max_cand) {
    ctx.nblocks = nblocks;
    ctx.max_cand = max_cand;
    ctx.hU = U;
    const size_t n = (size_t)nblocks * G;

    ctx.hEnergy.resize((size_t)nblocks);
    double sumsq_all = 0.0;
    for (int b = 0; b < nblocks; ++b) {
        double e = 0.0;
        for (int i = 0; i < G; ++i) {
            double v = (double)U[(size_t)b * G + i];
            e += v * v;
        }
        ctx.hEnergy[(size_t)b] = (float)e;
        sumsq_all += e;
    }
    double rms = std::sqrt(sumsq_all / (double)n);
    if (!(rms > 0.0) || !std::isfinite(rms)) rms = 1.0;
    ctx.inv_rms = (float)(1.0 / rms);

    std::vector<float> sorted_e = ctx.hEnergy;
    std::nth_element(sorted_e.begin(), sorted_e.begin() + nblocks / 2,
                     sorted_e.end());
    ctx.eps = EPS_SCALE * (double)sorted_e[(size_t)nblocks / 2];
    if (!(ctx.eps >= 0.0) || !std::isfinite(ctx.eps)) ctx.eps = 0.0;
    // Fixed-count top ceil(1%) of |U|; ties: smaller (block,index) preferred.
    // Avoids threshold-only ambiguity when many coeffs share the same magnitude.
    std::vector<float> abs_u(n);
    for (size_t i = 0; i < n; ++i) abs_u[i] = std::fabs(U[i]);
    size_t k = (n + 99) / 100; // ceil(n * 0.01)
    if (k < 1) k = 1;
    if (k > n) k = n;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + (std::ptrdiff_t)k, order.end(),
        [&](size_t a, size_t b) {
            if (abs_u[a] != abs_u[b]) return abs_u[a] > abs_u[b];
            return a < b; // (block, index) via linear layout [block*256+i]
        });
    ctx.hTail.assign(n, 0);
    ctx.tail_count = k;
    ctx.tail_thr = abs_u[order[k - 1]]; // min |U| among selected (diagnostic)
    for (size_t t = 0; t < k; ++t) ctx.hTail[order[t]] = 1;

    CK(hipMalloc(&ctx.dU, n * sizeof(float)));
    CK(hipMemcpy(ctx.dU, U.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dEnergy, (size_t)nblocks * sizeof(float)));
    CK(hipMemcpy(ctx.dEnergy, ctx.hEnergy.data(),
                 (size_t)nblocks * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dTail, n * sizeof(uint8_t)));
    CK(hipMemcpy(ctx.dTail, ctx.hTail.data(), n * sizeof(uint8_t),
                 hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dSse, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dWsse, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dTsse, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dRsum, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dD2, (size_t)max_cand * G * sizeof(float)));
    ctx.grid_x = std::min(nblocks, 4096);

    std::printf("corpus: nblocks=%d coeffs=%zu rms=%.6e eps=%.6e "
                "tail_thr=%.6e tail_n=%zu (%.3f%%)\n",
                nblocks, n, rms, ctx.eps, (double)ctx.tail_thr, ctx.tail_count,
                100.0 * (double)ctx.tail_count / (double)n);
}

static void gpu_free(GpuCtx& ctx) {
    if (ctx.dU) hipFree(ctx.dU);
    if (ctx.dEnergy) hipFree(ctx.dEnergy);
    if (ctx.dTail) hipFree(ctx.dTail);
    if (ctx.dSse) hipFree(ctx.dSse);
    if (ctx.dWsse) hipFree(ctx.dWsse);
    if (ctx.dTsse) hipFree(ctx.dTsse);
    if (ctx.dRsum) hipFree(ctx.dRsum);
    if (ctx.dD2) hipFree(ctx.dD2);
    ctx = GpuCtx{};
}

static std::vector<CandMetrics> gpu_eval_batch(GpuCtx& ctx, const float* d2_host,
                                               int ncand) {
    if (ncand <= 0 || ncand > ctx.max_cand) die("ncand out of range");
    const size_t zbytes = (size_t)ncand * NBITS * sizeof(double);
    CK(hipMemset(ctx.dSse, 0, zbytes));
    CK(hipMemset(ctx.dWsse, 0, zbytes));
    CK(hipMemset(ctx.dTsse, 0, zbytes));
    CK(hipMemset(ctx.dRsum, 0, zbytes));
    CK(hipMemcpy(ctx.dD2, d2_host, (size_t)ncand * G * sizeof(float),
                 hipMemcpyHostToDevice));

    dim3 grid((unsigned)ctx.grid_x, (unsigned)ncand, 1);
    dim3 block(G, 1, 1);
    hipLaunchKernelGGL(eval_affine_batch, grid, block, 0, 0, ctx.dU,
                       ctx.dEnergy, ctx.dTail, ctx.nblocks, ctx.dD2, ncand,
                       ctx.inv_rms, ctx.eps, ctx.dSse, ctx.dWsse, ctx.dTsse,
                       ctx.dRsum);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    std::vector<double> hs(ncand * NBITS), hw(ncand * NBITS), ht(ncand * NBITS),
        hr(ncand * NBITS);
    CK(hipMemcpy(hs.data(), ctx.dSse, zbytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hw.data(), ctx.dWsse, zbytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ht.data(), ctx.dTsse, zbytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hr.data(), ctx.dRsum, zbytes, hipMemcpyDeviceToHost));

    std::vector<CandMetrics> out(ncand);
    const double inv_nb = 1.0 / (double)ctx.nblocks;
    for (int c = 0; c < ncand; ++c) {
        for (int bi = 0; bi < NBITS; ++bi) {
            out[c].sse[bi] = hs[(size_t)c * NBITS + bi];
            out[c].wsse[bi] = hw[(size_t)c * NBITS + bi];
            out[c].tsse[bi] = ht[(size_t)c * NBITS + bi];
            out[c].r[bi] = hr[(size_t)c * NBITS + bi] * inv_nb;
        }
    }
    return out;
}

struct EvalRow {
    int bits;
    double mse, wmse, r_nmse, rel_all, tail_mse, rel_tail;
    double abs_p99, abs_p99_9, abs_max;
};

static std::vector<EvalRow> host_eval_detail(const GpuCtx& ctx, const float* d2,
                                             const CandMetrics* canon) {
    const int nblocks = ctx.nblocks;
    const size_t n = (size_t)nblocks * G;
    std::vector<EvalRow> rows;
    rows.reserve(NBITS);

    for (int bi = 0; bi < NBITS; ++bi) {
        const float levels = (float)QMAX[bi];
        double sse = 0.0, wsse = 0.0, rsum = 0.0, tsse = 0.0;
        size_t tcount = 0;
        std::vector<float> abs_err(n);
        double abs_max = 0.0;

        for (int blk = 0; blk < nblocks; ++blk) {
            float v[G];
            for (int i = 0; i < G; ++i)
                v[i] = ctx.hU[(size_t)blk * G + i] * d2[i];
            const double denom = (double)ctx.hEnergy[(size_t)blk] + ctx.eps;
            double blk_sse = 0.0;

            for (int h = 0; h < 2; ++h) {
                const int off = h * HALF;
                float lo = v[off], hi = v[off];
                for (int i = 1; i < HALF; ++i) {
                    lo = std::min(lo, v[off + i]);
                    hi = std::max(hi, v[off + i]);
                }
                float st = 0.0f, z = lo;
                bool degenerate = (hi == lo);
                if (!degenerate) {
                    st = f16_trunc_rt((hi - lo) / levels);
                    z = f16_trunc_rt(lo);
                    if (st == 0.0f) degenerate = true;
                } else {
                    z = f16_trunc_rt(lo);
                }
                for (int i = 0; i < HALF; ++i) {
                    const int li = off + i;
                    const float u = v[li];
                    float recon;
                    if (degenerate) {
                        recon = z;
                    } else {
                        float q = std::floor((u - z) / st + 0.5f);
                        if (q < 0.0f) q = 0.0f;
                        if (q > levels) q = levels;
                        recon = q * st + z;
                    }
                    const double e = (double)recon - (double)u;
                    const double e2 = e * e;
                    const size_t gi = (size_t)blk * G + (size_t)li;
                    const float ae = (float)std::fabs(e);
                    abs_err[gi] = ae;
                    if ((double)ae > abs_max) abs_max = (double)ae;
                    sse += e2;
                    wsse +=
                        e2 * (1.0 + (double)std::fabs(u) * (double)ctx.inv_rms);
                    blk_sse += e2;
                    if (ctx.hTail[gi]) {
                        tsse += e2;
                        ++tcount;
                    }
                }
            }
            rsum += blk_sse / denom;
        }

        std::vector<float> sorted = abs_err;
        std::sort(sorted.begin(), sorted.end());
        auto pct = [&](double p) -> double {
            double idx = p * (double)(sorted.size() - 1);
            size_t i0 = (size_t)idx;
            size_t i1 = std::min(i0 + 1, sorted.size() - 1);
            double t = idx - (double)i0;
            return (1.0 - t) * (double)sorted[i0] + t * (double)sorted[i1];
        };

        EvalRow row;
        row.bits = BITS[bi];
        row.mse = sse / (double)n;
        row.wmse = wsse / (double)n;
        row.r_nmse = rsum / (double)nblocks;
        row.tail_mse = tcount ? tsse / (double)tcount : 0.0;
        if (canon) {
            const double ra = canon->r[bi] > 0.0 ? canon->r[bi] : 1e-300;
            const double ta = canon->tsse[bi] > 0.0 ? canon->tsse[bi] : 1e-300;
            row.rel_all = row.r_nmse / ra;
            row.rel_tail = tsse / ta;
        } else {
            row.rel_all = 1.0;
            row.rel_tail = 1.0;
        }
        row.abs_p99 = pct(0.99);
        row.abs_p99_9 = pct(0.999);
        row.abs_max = abs_max;
        rows.push_back(row);
    }
    return rows;
}

static std::vector<float> pre_rotate_corpus(const std::vector<float>& raw,
                                            int nblocks) {
    float d1[G];
    gen_fwht_signs(D1_SEED, d1, G);
    std::vector<float> U = raw;
    for (int b = 0; b < nblocks; ++b)
        pre_rotate_block(U.data() + (size_t)b * G, d1);
    return U;
}

static void mode_random(const char* input, int N, uint32_t rng_seed,
                        const char* out_prefix) {
    if (N < 0) die("N must be >= 0");
    int nblocks = 0;
    auto raw = load_corpus(input, nblocks);
    auto U = pre_rotate_corpus(raw, nblocks);

    const int ncand = N + 1;
    const int batch_cap = std::min(std::max(ncand, 1), 512);
    GpuCtx ctx;
    gpu_init(ctx, U, nblocks, batch_cap);
    const size_t tail_count = ctx.tail_count;

    struct Row {
        int rank = 0;
        uint32_t seed = 0;
        const char* label = "";
        CandMetrics m;
        std::vector<float> signs;
        uint32_t pack[8]{};
    };
    std::vector<Row> rows;
    rows.reserve((size_t)ncand);

    CandMetrics canon_m;
    {
        std::vector<float> signs(G);
        gen_fwht_signs(D2_CANON_SEED, signs.data(), G);
        canon_m = gpu_eval_batch(ctx, signs.data(), 1)[0];
        CandMetrics self = canon_m;
        self.finalize(canon_m);
        Row r;
        r.seed = D2_CANON_SEED;
        r.label = "canonical";
        r.m = self;
        r.signs = std::move(signs);
        pack_signs_u32(r.signs.data(), r.pack);
        rows.push_back(std::move(r));
    }

    std::vector<float> batch_signs;
    std::vector<uint32_t> batch_seeds;
    auto flush = [&]() {
        if (batch_seeds.empty()) return;
        const int bsz = (int)batch_seeds.size();
        auto ms = gpu_eval_batch(ctx, batch_signs.data(), bsz);
        for (int i = 0; i < bsz; ++i) {
            Row r;
            r.seed = batch_seeds[(size_t)i];
            r.label = "random";
            r.signs.resize(G);
            std::memcpy(r.signs.data(), batch_signs.data() + (size_t)i * G,
                        (size_t)G * sizeof(float));
            r.m = ms[(size_t)i];
            r.m.finalize(canon_m);
            pack_signs_u32(r.signs.data(), r.pack);
            rows.push_back(std::move(r));
        }
        batch_signs.clear();
        batch_seeds.clear();
    };

    for (int i = 0; i < N; ++i) {
        uint32_t seed = rng_seed + (uint32_t)i + 1u;
        if (seed == D2_CANON_SEED) seed += 0x9e3779b9u;
        float s[G];
        gen_fwht_signs(seed, s, G);
        batch_seeds.push_back(seed);
        batch_signs.insert(batch_signs.end(), s, s + G);
        if ((int)batch_seeds.size() >= batch_cap) flush();
    }
    flush();
    gpu_free(ctx);

    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return better_than(a.m, a.pack, b.m, b.pack);
    });
    for (size_t i = 0; i < rows.size(); ++i) rows[i].rank = (int)i + 1;

    size_t best_i = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].m.feasible) {
            best_i = i;
            break;
        }
    }
    if (!rows[best_i].m.feasible) {
        std::fprintf(stderr,
                     "warning: no feasible candidate (J_all<=1); emitting "
                     "best infeasible by J_tail\n");
    }

    const double denom = (double)nblocks * (double)G;
    const double tden = (double)std::max<size_t>(1, tail_count);

    std::string csv;
    csv += "rank,seed,label,feasible,J_tail,J_all,J_scalar,worst_rel,obj_wsse,"
           "sse2,sse3,sse4,wsse2,wsse3,wsse4,tsse2,tsse3,tsse4,"
           "r2,r3,r4,rel_all2,rel_all3,rel_all4,rel_tail2,rel_tail3,rel_tail4,"
           "mse2,mse3,mse4,wmse2,wmse3,wmse4,tail_mse2,tail_mse3,tail_mse4,"
           "signs_u32_0,signs_u32_1,signs_u32_2,signs_u32_3,"
           "signs_u32_4,signs_u32_5,signs_u32_6,signs_u32_7\n";
    char line[1400];
    for (const auto& r : rows) {
        std::snprintf(
            line, sizeof(line),
            "%d,%u,%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
            r.rank, r.seed, r.label, r.m.feasible, r.m.J_tail, r.m.J_all,
            r.m.J_scalar, r.m.worst_rel, r.m.obj_wsse(), r.m.sse[0], r.m.sse[1],
            r.m.sse[2], r.m.wsse[0], r.m.wsse[1], r.m.wsse[2], r.m.tsse[0],
            r.m.tsse[1], r.m.tsse[2], r.m.r[0], r.m.r[1], r.m.r[2],
            r.m.rel_all[0], r.m.rel_all[1], r.m.rel_all[2], r.m.rel_tail[0],
            r.m.rel_tail[1], r.m.rel_tail[2], r.m.sse[0] / denom,
            r.m.sse[1] / denom, r.m.sse[2] / denom, r.m.wsse[0] / denom,
            r.m.wsse[1] / denom, r.m.wsse[2] / denom, r.m.tsse[0] / tden,
            r.m.tsse[1] / tden, r.m.tsse[2] / tden, r.pack[0], r.pack[1],
            r.pack[2], r.pack[3], r.pack[4], r.pack[5], r.pack[6], r.pack[7]);
        csv += line;
    }
    atomic_write_text(std::string(out_prefix) + ".ranked.csv", csv);
    write_signs(std::string(out_prefix) + ".best.signs",
                rows[best_i].signs.data());
    write_u32_txt(std::string(out_prefix) + ".best.u32.txt",
                  rows[best_i].signs.data());

    std::printf("random: candidates=%d best_rank=%d seed=%u (%s) feasible=%d "
                "J_tail=%.6e J_all=%.6e u32=%08x…\n",
                ncand, rows[best_i].rank, rows[best_i].seed, rows[best_i].label,
                rows[best_i].m.feasible, rows[best_i].m.J_tail,
                rows[best_i].m.J_all, rows[best_i].pack[0]);
}

static void mode_coord(const char* input, const char* signs_path, int passes,
                       const char* out_prefix) {
    if (passes < 1) die("PASSES must be >= 1");
    int nblocks = 0;
    auto raw = load_corpus(input, nblocks);
    auto U = pre_rotate_corpus(raw, nblocks);
    auto cur = load_signs(signs_path);

    GpuCtx ctx;
    gpu_init(ctx, U, nblocks, 1);

    float canon_s[G];
    gen_fwht_signs(D2_CANON_SEED, canon_s, G);
    CandMetrics canon_m = gpu_eval_batch(ctx, canon_s, 1)[0];

    auto eval_one = [&](const float* s) {
        CandMetrics m = gpu_eval_batch(ctx, s, 1)[0];
        m.finalize(canon_m);
        return m;
    };

    CandMetrics m_cur = eval_one(cur.data());
    uint32_t pack_cur[8];
    pack_signs_u32(cur.data(), pack_cur);

    std::string csv;
    csv += "pass,flip_idx,accepted,feasible,J_tail,J_all,delta_J_tail,worst_rel,"
           "obj_wsse,sse2,sse3,sse4,wsse2,wsse3,wsse4,tsse2,tsse3,tsse4,"
           "r2,r3,r4,rel_all2,rel_all3,rel_all4,rel_tail2,rel_tail3,rel_tail4\n";
    char line[900];
    auto emit = [&](int pass, int flip, int acc, const CandMetrics& m,
                    double dJt) {
        std::snprintf(
            line, sizeof(line),
            "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            pass, flip, acc, m.feasible, m.J_tail, m.J_all, dJt, m.worst_rel,
            m.obj_wsse(), m.sse[0], m.sse[1], m.sse[2], m.wsse[0], m.wsse[1],
            m.wsse[2], m.tsse[0], m.tsse[1], m.tsse[2], m.r[0], m.r[1], m.r[2],
            m.rel_all[0], m.rel_all[1], m.rel_all[2], m.rel_tail[0],
            m.rel_tail[1], m.rel_tail[2]);
        csv += line;
    };

    for (int p = 1; p <= passes; ++p) {
        int accepted_n = 0;
        for (int fi = 0; fi < G; ++fi) {
            std::vector<float> trial = cur;
            trial[(size_t)fi] = -trial[(size_t)fi];
            CandMetrics mt = eval_one(trial.data());
            uint32_t pack_t[8];
            pack_signs_u32(trial.data(), pack_t);
            if (better_than(mt, pack_t, m_cur, pack_cur)) {
                const double dJt = mt.J_tail - m_cur.J_tail;
                cur.swap(trial);
                m_cur = mt;
                std::memcpy(pack_cur, pack_t, sizeof(pack_cur));
                ++accepted_n;
                emit(p, fi, 1, m_cur, dJt);
            }
        }
        emit(p, -1, 0, m_cur, 0.0);
        std::printf("coord pass %d/%d: accepted=%d feasible=%d J_tail=%.6e "
                    "J_all=%.6e\n",
                    p, passes, accepted_n, m_cur.feasible, m_cur.J_tail,
                    m_cur.J_all);
    }
    gpu_free(ctx);

    atomic_write_text(std::string(out_prefix) + ".coord.csv", csv);
    write_signs(std::string(out_prefix) + ".final.signs", cur.data());
    write_u32_txt(std::string(out_prefix) + ".final.u32.txt", cur.data());
}

static void mode_eval(const char* input, const char* signs_path,
                      const char* out_prefix) {
    int nblocks = 0;
    auto raw = load_corpus(input, nblocks);
    auto U = pre_rotate_corpus(raw, nblocks);
    auto cand = load_signs(signs_path);
    float canon[G];
    gen_fwht_signs(D2_CANON_SEED, canon, G);

    GpuCtx ctx;
    gpu_init(ctx, U, nblocks, 1);

    CandMetrics cm = gpu_eval_batch(ctx, canon, 1)[0];
    CandMetrics cnd = gpu_eval_batch(ctx, cand.data(), 1)[0];
    cnd.finalize(cm);

    auto rows_c = host_eval_detail(ctx, cand.data(), &cm);
    auto rows_k = host_eval_detail(ctx, canon, &cm);

    uint32_t pack[8];
    pack_signs_u32(cand.data(), pack);

    std::string csv;
    csv += "who,bits,mse,wmse,r_nmse,rel_all,tail_mse,rel_tail,abs_p99,"
           "abs_p99_9,abs_max\n";
    char line[512];
    auto dump = [&](const char* who, const std::vector<EvalRow>& rows) {
        for (const auto& r : rows) {
            std::snprintf(
                line, sizeof(line),
                "%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                who, r.bits, r.mse, r.wmse, r.r_nmse, r.rel_all, r.tail_mse,
                r.rel_tail, r.abs_p99, r.abs_p99_9, r.abs_max);
            csv += line;
        }
    };
    dump("candidate", rows_c);
    dump("canonical", rows_k);
    atomic_write_text(std::string(out_prefix) + ".eval.csv", csv);
    write_u32_txt(std::string(out_prefix) + ".eval.u32.txt", cand.data());

    std::printf("eval: feasible=%d J_tail=%.6e J_all=%.6e J_scalar=%.6e "
                "worst_rel=%.6e\n"
                "signs_u32: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                cnd.feasible, cnd.J_tail, cnd.J_all, cnd.J_scalar, cnd.worst_rel,
                pack[0], pack[1], pack[2], pack[3], pack[4], pack[5], pack[6],
                pack[7]);
    for (int i = 0; i < NBITS; ++i) {
        std::printf("  bits=%d mse=%.6e/%.6e tail_mse=%.6e/%.6e "
                    "rel_tail=%.4f rel_all=%.4f\n",
                    BITS[i], rows_c[(size_t)i].mse, rows_k[(size_t)i].mse,
                    rows_c[(size_t)i].tail_mse, rows_k[(size_t)i].tail_mse,
                    rows_c[(size_t)i].rel_tail, rows_c[(size_t)i].rel_all);
    }
    gpu_free(ctx);
}

int main(int argc, char** argv) {
    if (argc < 2) usage(argv[0], 1);
    const std::string mode = argv[1];
    if (mode == "--help" || mode == "-h" || mode == "help") usage(argv[0], 0);

    CK(hipSetDevice(0));

    if (mode == "random") {
        if (argc != 6) usage(argv[0], 1);
        mode_random(argv[2], std::atoi(argv[3]),
                    (uint32_t)std::strtoul(argv[4], nullptr, 10), argv[5]);
        return 0;
    }
    if (mode == "coord") {
        if (argc != 6) usage(argv[0], 1);
        mode_coord(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
        return 0;
    }
    if (mode == "eval") {
        if (argc != 5) usage(argv[0], 1);
        mode_eval(argv[2], argv[3], argv[4]);
        return 0;
    }

    std::fprintf(stderr, "error: unknown mode '%s'\n", mode.c_str());
    usage(argv[0], 1);
    return 1;
}
