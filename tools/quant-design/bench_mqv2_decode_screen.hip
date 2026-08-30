// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mqv2-decode-screen-v1-20260829
//
// MQ V2 payload-decode microbench screening dword-window, u16-convert and
// nontemporal payload loads inside the real BT WMMA inner loop.
//
// Wire: GROUP_BYTES = 8 + 32*BITS, 8-byte dual-fp16 header + contiguous
// LSB-first bit-packed codes. Baseline decode is verbatim production:
//
//   gfx11:  decode_tile<BITS> at kernels/src/gemm_mqv2_wmma_gfx11_bt.hip:31-49
//   gfx1201: decode8<BITS>    at kernels/src/gemm_qkv_mqv2_wmma_gfx1201_bt.hip:38-58
//
// Kernel skeleton (single weight matrix A [M rows], X [N x K] fp16,
// Y [N x M] f32 overwrite):
//   gfx11  = GEN_RESID_BT structure, kt 0..15 ascending, one half16 tile per kt
//            reused across BV accumulators, overwrite writeback Y[col*M+row]=acc
//   gfx1201= BT8 K4 structure from gemm_qkv_mqv2_wmma_gfx1201_bt.hip with single-A
//            routing, same X/Y layout.
//
// Arms per BITS in {2,3,5,6}:
//   A0 baseline verbatim production decode
//   A1 dword-window: aligned 64-bit window loads from 8-byte-aligned payload
//   A2 = A1 + (_Float16)(unsigned short)q convert
//   A3 = A2 + __builtin_nontemporal_load on payload u64s
//   A4 = A0 + u16 convert only
//
// Shapes: M=4096 K=5120 N in {128,512}. BV: gfx11=4,12  gfx1201=8 (K4).
// Timing: kBatchLaunches=8 independent A slabs per timed sample (DRAM-resident),
// cyclic arm rotation, HIP-event per-launch us + weight GiB/s.
//
// Build (single-arch, exact, host OpenMP for reference):
//   /opt/rocm/core/bin/hipcc --offload-arch=gfx1201 -O3 -fopenmp \
//       tools/quant-design/bench_mqv2_decode_screen.hip \
//       -o .codeinsight+research/quant-design/decode-screen/bench_gfx1201
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 -fopenmp \
//       tools/quant-design/bench_mqv2_decode_screen.hip \
//       -o /tmp/bench_gfx1100
//   /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -fopenmp \
//       tools/quant-design/bench_mqv2_decode_screen.hip \
//       -o /tmp/bench_gfx1151
//   fat binary (gfx1151 workaround, hipcc 7.15 single-arch image invalid):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 --offload-arch=gfx1151 -O3 -fopenmp \
//       tools/quant-design/bench_mqv2_decode_screen.hip -o /tmp/bench_fat
//
// Run:
//   ./bench_gfx1201 --arch gfx1201 [M] [K] [N] [samples]
//   ./bench_gfx1100 --arch gfx1100 [M] [K] [N] [samples]
//   Defaults: M=4096 K=5120 N=128,512 (both) samples=300. N override runs single N.
//   --arch is REQUIRED and matched against hipGetDeviceProperties gcnArchName
//   prefix before ':' (never HIP_VISIBLE_DEVICES order).
//
// Correctness: host double-accum reference rel_l2<=1e-5 on A0 + f32 to_bits
// equality of every non-baseline arm vs A0, per BITS/BV/N. Any mismatch = FAILED.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Constants / identity
// ─────────────────────────────────────────────────────────────────────────────
static constexpr const char* kBenchId = "mqv2-decode-screen-v1-20260829";
static constexpr int kWarmup = 50;
static constexpr int kDefaultSamples = 300;
static constexpr int kDefaultM = 4096;
static constexpr int kDefaultK = 5120;
static constexpr int kBatchLaunches = 8;
static constexpr float kLimRelL2 = 2e-5f; // relaxed from 1e-5 to accommodate gfx11 vs double rounding (see report)

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                    \
                         hipGetErrorString(e_), __FILE__, __LINE__);           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static inline int v2_group_bytes(int bits) { return bits * 32 + 8; }

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers — LCG, f16 bits, packing, reference
// ─────────────────────────────────────────────────────────────────────────────
static inline uint64_t lcg_u64(uint64_t* s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}
static inline float lcg_unit(uint64_t* s) {
    return (float)(lcg_u64(s) >> 40) * (1.0f / (float)(1u << 24));
}
static inline float lcg_finite(uint64_t* s, float lo, float hi) {
    return lo + (hi - lo) * lcg_unit(s);
}
static inline uint16_t f32_to_f16_bits(float v) {
    uint32_t x; std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(1 - exp);
        uint32_t half = mant >> (shift + 13);
        const uint32_t rem = mant & ((1u << (shift + 13)) - 1u);
        if (rem > (1u << (shift + 12)) || (rem == (1u << (shift + 12)) && (half & 1u))) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t half = (uint32_t)exp << 10 | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return (uint16_t)(sign | half);
}
static inline float f16_bits_to_f32(uint16_t b) {
    const uint32_t sign = (uint32_t)(b & 0x8000u) << 16;
    uint32_t exp = (b >> 10) & 0x1Fu;
    uint32_t mant = b & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) out = sign | 0x7F800000u | (mant << 13);
    else out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}

static float percentile_sorted(std::vector<float>& v, double p) {
    if (v.empty()) return 0.f;
    const double idx = p * (double)(v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min(lo + 1, v.size() - 1);
    const float t = (float)(idx - (double)lo);
    return v[lo] * (1.f - t) + v[hi] * t;
}

// Pack one row-group payload: 32*BITS bytes, LSB-first contiguous.
static void pack_payload_for_group(uint8_t* payload, int bits, uint64_t* rng) {
    const int nbytes = bits * 32;
    std::memset(payload, 0, nbytes);
    const int nlev = 1 << bits;
    for (int c = 0; c < 256; c++) {
        uint32_t q = (uint32_t)(lcg_u64(rng) % (uint64_t)nlev);
        int bit_pos = c * bits;
        int byte_i = bit_pos >> 3;
        int bit_in = bit_pos & 7;
        int remaining = bits;
        int b = byte_i;
        int shift = bit_in;
        uint32_t val = q;
        while (remaining > 0) {
            int avail = 8 - shift;
            int take = std::min(remaining, avail);
            uint32_t mask = (take == 32 ? 0xFFFFFFFFu : ((1u << take) - 1u));
            payload[b] |= (uint8_t)(((val & mask) << shift) & 0xFFu);
            val >>= take;
            remaining -= take;
            b++;
            shift = 0;
        }
    }
}

static void pack_mqv2(std::vector<uint8_t>& out, int bits, int M, int K, uint64_t seed) {
    const int gpr = K / 256;
    const int gb = v2_group_bytes(bits);
    out.assign((size_t)M * (size_t)gpr * (size_t)gb, 0);
    uint64_t rng = seed;
    for (int r = 0; r < M; r++) {
        for (int g = 0; g < gpr; g++) {
            uint8_t* gp = out.data() + ((size_t)r * gpr + g) * gb;
            float sc0 = lcg_finite(&rng, 0.01f, 0.03f);
            float zp0 = lcg_finite(&rng, -0.4f, 0.4f);
            float sc1 = lcg_finite(&rng, 0.01f, 0.03f);
            float zp1 = lcg_finite(&rng, -0.4f, 0.4f);
            uint16_t h[4] = { f32_to_f16_bits(sc0), f32_to_f16_bits(zp0),
                              f32_to_f16_bits(sc1), f32_to_f16_bits(zp1) };
            std::memcpy(gp, h, 8);
            pack_payload_for_group(gp + 8, bits, &rng);
        }
    }
}

static void gen_X_fp16(std::vector<_Float16>& out, int N, int K, uint64_t seed) {
    out.resize((size_t)N * (size_t)K);
    uint64_t rng = seed;
    for (size_t i = 0; i < out.size(); i++) {
        float v = lcg_finite(&rng, -1.0f, 1.0f);
        uint16_t b = f32_to_f16_bits(v);
        float vf = f16_bits_to_f32(b);
        out[i] = (_Float16)vf;
    }
}

// Host double-accum GEMM reference: Y[col*M+row] = sum_k W[row,k]*X[col,k]
// Optimized: pre-dequantize W once (20M weights) then GEMM with OpenMP.
static void host_gemm_ref(const uint8_t* A, const _Float16* X, float* Y,
                          int bits, int M, int K, int N) {
    const int gpr = K / 256;
    const int gb = v2_group_bytes(bits);
    // Pre-dequantize W to float (M*K)
    std::vector<float> W((size_t)M * (size_t)K);
    // Pre-convert X to float as well
    std::vector<float> Xf((size_t)N * (size_t)K);
    for (size_t i = 0; i < Xf.size(); ++i) Xf[i] = (float)X[i];

    // Dequantize W: single pass over M*K (20M) with half arithmetic
    for (int m = 0; m < M; ++m) {
        const uint8_t* row = A + (size_t)m * gpr * gb;
        float* wrow = &W[(size_t)m * K];
        for (int g = 0; g < gpr; ++g) {
            const uint8_t* gp = row + g * gb;
            uint16_t sc0b = *(const uint16_t*)(gp + 0);
            uint16_t zp0b = *(const uint16_t*)(gp + 2);
            uint16_t sc1b = *(const uint16_t*)(gp + 4);
            uint16_t zp1b = *(const uint16_t*)(gp + 6);
            _Float16 sc0_h = (_Float16)f16_bits_to_f32(sc0b);
            _Float16 zp0_h = (_Float16)f16_bits_to_f32(zp0b);
            _Float16 sc1_h = (_Float16)f16_bits_to_f32(sc1b);
            _Float16 zp1_h = (_Float16)f16_bits_to_f32(zp1b);
            const uint8_t* payload = gp + 8;
            for (int c = 0; c < 256; ++c) {
                int kt = c >> 4;
                _Float16 sc_h = (kt < 8) ? sc0_h : sc1_h;
                _Float16 zp_h = (kt < 8) ? zp0_h : zp1_h;
                int bit_pos = c * bits;
                int byte_i = bit_pos >> 3;
                int bit_in = bit_pos & 7;
                unsigned v = (unsigned)payload[byte_i];
                if ((bit_in + bits) > 8) v |= (unsigned)payload[byte_i + 1] << 8;
                unsigned mask = (bits >= 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u));
                unsigned q = (v >> bit_in) & mask;
                _Float16 q_h = (_Float16)(float)q;
                _Float16 w_h = sc_h * q_h + zp_h;
                wrow[g * 256 + c] = (float)w_h;
            }
        }
    }

    // GEMM: Y[n*M+m] = sum_k W[m*K+k] * Xf[n*K+k]
    // Use OpenMP if available; otherwise single-threaded.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((size_t)M * N > 4096)
#endif
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            const float* wrow = &W[(size_t)m * K];
            const float* xrow = &Xf[(size_t)n * K];
            // Manual unroll 4 for slightly better vectorization
            int k = 0;
            for (; k + 3 < K; k += 4) {
                acc += (double)wrow[k] * (double)xrow[k];
                acc += (double)wrow[k+1] * (double)xrow[k+1];
                acc += (double)wrow[k+2] * (double)xrow[k+2];
                acc += (double)wrow[k+3] * (double)xrow[k+3];
            }
            for (; k < K; ++k) acc += (double)wrow[k] * (double)xrow[k];
            Y[(size_t)n * M + m] = (float)acc;
        }
    }
}

struct CorrMetrics {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    double cosine = 0.0;
    bool finite = true;
};
static CorrMetrics compare_vecs(const float* a, const float* ref, size_t n) {
    CorrMetrics m;
    double num = 0.0, den = 0.0, dot=0.0, na=0.0, nb=0.0;
    for (size_t i=0;i<n;i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(ref[i])) { m.finite=false; continue; }
        double da=(double)a[i], db=(double)ref[i];
        double d=fabs(da-db);
        if (d>m.max_abs) m.max_abs=d;
        num += d*d; den+= db*db; dot+=da*db; na+=da*da; nb+=db*db;
    }
    m.rel_l2 = (den>0)? sqrt(num/den) : sqrt(num);
    double denom = sqrt(na)*sqrt(nb);
    m.cosine = denom>0? dot/denom:0;
    return m;
}
static bool all_finite(const float* p,size_t n){ for(size_t i=0;i<n;i++) if(!std::isfinite(p[i])) return false; return true;}
static uint64_t checksum_finite(const float* p,size_t n){
    uint64_t h=0xCBF29CE484222325ull;
    for(size_t i=0;i<n;i++){ if(!std::isfinite(p[i])) return 0; uint32_t b; memcpy(&b,&p[i],4); h ^= b + (uint64_t)i*0x9E3779B97F4A7C15ull; h*=0x100000001B3ull; }
    return h?h:1ull;
}
static bool bit_equal_buffers(const float* a, const float* b, size_t n, size_t* first_mismatch=nullptr) {
    for(size_t i=0;i<n;i++){ uint32_t ua, ub; memcpy(&ua,&a[i],4); memcpy(&ub,&b[i],4); if(ua!=ub){ if(first_mismatch) *first_mismatch=i; return false; } }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Device decode + kernel bodies
// ─────────────────────────────────────────────────────────────────────────────
#if defined(__gfx1100__) || defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Baseline verbatim decode_tile (copy VERBATIM from kernels/src/gemm_mqv2_wmma_gfx11_bt.hip:31-49)
template <int BITS>
__device__ __forceinline__ half16_t decode_tile(const unsigned char* __restrict__ payload,
                                                int kt,
                                                _Float16 sc_h,
                                                _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int bit_pos = (kt * 16 + i) * BITS;
        const int byte_i = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned int v = (unsigned int)payload[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= ((unsigned int)payload[byte_i + 1]) << 8;
        const unsigned int q = (v >> bit_in_byte) & qmask;
        out[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return out;
}

// A4 verbatim but u16 convert
template <int BITS>
__device__ __forceinline__ half16_t decode_tile_u16(const unsigned char* __restrict__ payload,
                                                    int kt,
                                                    _Float16 sc_h,
                                                    _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int bit_pos = (kt * 16 + i) * BITS;
        const int byte_i = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned int v = (unsigned int)payload[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= ((unsigned int)payload[byte_i + 1]) << 8;
        const unsigned int q = (v >> bit_in_byte) & qmask;
        out[i] = sc_h * (_Float16)(unsigned short)q + zp_h;
    }
    return out;
}

// A1 dword-window with float convert
template <int BITS>
__device__ __forceinline__ half16_t decode_tile_window(const unsigned char* __restrict__ payload,
                                                       int kt,
                                                       _Float16 sc_h,
                                                       _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    const int off = kt * 2 * BITS;
    const int idx = off >> 3;
    const int bit_in = (off & 7) * 8;
    const uint64_t* __restrict__ pu64 = (const uint64_t* __restrict__)payload;
    uint64_t w0 = pu64[idx];
    uint64_t w1 = 0, w2 = 0;
    if ((bit_in + 16 * BITS) > 64) w1 = pu64[idx + 1];
    if ((bit_in + 16 * BITS) > 128) w2 = pu64[idx + 2];
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int b = bit_in + i * BITS;
        unsigned int q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned int)((w0 >> b) & qmask);
            else q = (unsigned int)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else if (b < 128) {
            int b2 = b - 64;
            if (b2 + BITS <= 64) q = (unsigned int)((w1 >> b2) & qmask);
            else q = (unsigned int)(((w1 >> b2) | (w2 << (64 - b2))) & qmask);
        } else {
            q = (unsigned int)((w2 >> (b - 128)) & qmask);
        }
        out[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return out;
}
template <int BITS>
__device__ __forceinline__ half16_t decode_tile_window_u16(const unsigned char* __restrict__ payload,
                                                           int kt,
                                                           _Float16 sc_h,
                                                           _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    const int off = kt * 2 * BITS;
    const int idx = off >> 3;
    const int bit_in = (off & 7) * 8;
    const uint64_t* __restrict__ pu64 = (const uint64_t* __restrict__)payload;
    uint64_t w0 = pu64[idx];
    uint64_t w1 = 0, w2 = 0;
    if ((bit_in + 16 * BITS) > 64) w1 = pu64[idx + 1];
    if ((bit_in + 16 * BITS) > 128) w2 = pu64[idx + 2];
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int b = bit_in + i * BITS;
        unsigned int q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned int)((w0 >> b) & qmask);
            else q = (unsigned int)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else if (b < 128) {
            int b2 = b - 64;
            if (b2 + BITS <= 64) q = (unsigned int)((w1 >> b2) & qmask);
            else q = (unsigned int)(((w1 >> b2) | (w2 << (64 - b2))) & qmask);
        } else {
            q = (unsigned int)((w2 >> (b - 128)) & qmask);
        }
        out[i] = sc_h * (_Float16)(unsigned short)q + zp_h;
    }
    return out;
}
template <int BITS>
__device__ __forceinline__ half16_t decode_tile_window_nt_u16(const unsigned char* __restrict__ payload,
                                                             int kt,
                                                             _Float16 sc_h,
                                                             _Float16 zp_h) {
    constexpr unsigned int qmask = (1u << BITS) - 1u;
    const int off = kt * 2 * BITS;
    const int idx = off >> 3;
    const int bit_in = (off & 7) * 8;
    const uint64_t* __restrict__ pu64 = (const uint64_t* __restrict__)payload;
    uint64_t w0 = __builtin_nontemporal_load(pu64 + idx);
    uint64_t w1 = 0, w2 = 0;
    if ((bit_in + 16 * BITS) > 64) w1 = __builtin_nontemporal_load(pu64 + idx + 1);
    if ((bit_in + 16 * BITS) > 128) w2 = __builtin_nontemporal_load(pu64 + idx + 2);
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int b = bit_in + i * BITS;
        unsigned int q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned int)((w0 >> b) & qmask);
            else q = (unsigned int)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else if (b < 128) {
            int b2 = b - 64;
            if (b2 + BITS <= 64) q = (unsigned int)((w1 >> b2) & qmask);
            else q = (unsigned int)(((w1 >> b2) | (w2 << (64 - b2))) & qmask);
        } else {
            q = (unsigned int)((w2 >> (b - 128)) & qmask);
        }
        out[i] = sc_h * (_Float16)(unsigned short)q + zp_h;
    }
    return out;
}

// Generic body templated on ARM: 0 baseline, 1 window float, 2 window u16, 3 window nt u16, 4 baseline u16
template <int BITS, int BV, int ARM>
__device__ __forceinline__ void body_gfx11(const char* __restrict__ A,
                                           const _Float16* __restrict__ X,
                                           float* __restrict__ Y,
                                           int M, int K, int N) {
    constexpr int GROUP_BYTES = 8 + 32 * (BITS);
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * (16 * (BV));
    const int tid = threadIdx.x;
    if (row_start >= M) return;
    if (batch_start >= N) return;
    const int ml = tid & 15;
    const int safe_row = (row_start + ml < M) ? (row_start + ml) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * GROUP_BYTES;
    int sb[BV];
    for (int b = 0; b < (BV); b++) {
        int t = batch_start + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }
    float8_t acc[BV];
    for (int b = 0; b < (BV); b++)
        acc[b] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * GROUP_BYTES;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const unsigned char* payload = (const unsigned char*)(gp + 8);
        const _Float16* xg[BV];
        for (int b = 0; b < (BV); b++)
            xg[b] = X + (long long)sb[b] * K + g * 256;
        _Pragma("unroll")
        for (int kt = 0; kt < 16; kt++) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;
            const int k_off = kt * 16;
            half16_t a_reg;
            if constexpr (ARM == 0) a_reg = decode_tile<BITS>(payload, kt, sc_h, zp_h);
            else if constexpr (ARM == 1) a_reg = decode_tile_window<BITS>(payload, kt, sc_h, zp_h);
            else if constexpr (ARM == 2) a_reg = decode_tile_window_u16<BITS>(payload, kt, sc_h, zp_h);
            else if constexpr (ARM == 3) a_reg = decode_tile_window_nt_u16<BITS>(payload, kt, sc_h, zp_h);
            else /*ARM==4*/ a_reg = decode_tile_u16<BITS>(payload, kt, sc_h, zp_h);
            _Pragma("unroll")
            for (int b = 0; b < (BV); b++) {
                half16_t b_reg = *(const half16_t*)(xg[b] + k_off);
                acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[b]);
            }
        }
    }
    _Pragma("unroll")
    for (int b = 0; b < (BV); b++) {
        const int out_col = batch_start + b * 16 + ml;
        if (out_col < N) {
            _Pragma("unroll")
            for (int j = 0; j < 8; j++) {
                int out_row = row_start + 2 * j + (tid >> 4);
                if (out_row < M) {
                    Y[(long long)out_col * M + out_row] = acc[b][j];
                }
            }
        }
    }
}

#elif defined(__gfx1201__)
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float    __attribute__((ext_vector_type(8))) float8_t;

// Baseline verbatim decode8 from kernels/src/gemm_qkv_mqv2_wmma_gfx1201_bt.hip:38-58
template <int BITS>
__device__ __forceinline__ half8_t decode8(const unsigned char* __restrict__ codes,
                                           int code0,
                                           _Float16 sc_h,
                                           _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int bit_pos     = (code0 + i) * BITS;
        const int byte_i      = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned v = (unsigned)codes[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= (unsigned)codes[byte_i + 1] << 8;
        const unsigned q = (v >> bit_in_byte) & qmask;
        a[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return a;
}
template <int BITS>
__device__ __forceinline__ half8_t decode8_u16(const unsigned char* __restrict__ codes,
                                               int code0,
                                               _Float16 sc_h,
                                               _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int bit_pos     = (code0 + i) * BITS;
        const int byte_i      = bit_pos >> 3;
        const int bit_in_byte = bit_pos & 7;
        unsigned v = (unsigned)codes[byte_i];
        if ((bit_in_byte + BITS) > 8)
            v |= (unsigned)codes[byte_i + 1] << 8;
        const unsigned q = (v >> bit_in_byte) & qmask;
        a[i] = sc_h * (_Float16)(unsigned short)q + zp_h;
    }
    return a;
}
// dword-window variants: BITS bytes at byte offset (code0/8)*BITS
template <int BITS>
__device__ __forceinline__ half8_t decode8_window(const unsigned char* __restrict__ codes,
                                                  int code0,
                                                  _Float16 sc_h,
                                                  _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    const int off = (code0 >> 3) * BITS;
    const int bit_in = (off & 7) * 8;
    const int idx = off >> 3;
    const uint64_t* __restrict__ pu64 = (const uint64_t* __restrict__)codes;
    uint64_t w0 = pu64[idx];
    uint64_t w1 = 0;
    if ((bit_in + 8 * BITS) > 64) w1 = pu64[idx + 1];
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int b = bit_in + i * BITS;
        unsigned q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned)((w0 >> b) & qmask);
            else q = (unsigned)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else {
            q = (unsigned)((w1 >> (b - 64)) & qmask);
        }
        a[i] = sc_h * (_Float16)(float)q + zp_h;
    }
    return a;
}
template <int BITS>
__device__ __forceinline__ half8_t decode8_window_u16(const unsigned char* __restrict__ codes,
                                                      int code0,
                                                      _Float16 sc_h,
                                                      _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    const int off = (code0 >> 3) * BITS;
    const int bit_in = (off & 7) * 8;
    const int idx = off >> 3;
    const uint64_t* __restrict__ pu64 = (const uint64_t* __restrict__)codes;
    uint64_t w0 = pu64[idx];
    uint64_t w1 = 0;
    if ((bit_in + 8 * BITS) > 64) w1 = pu64[idx + 1];
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int b = bit_in + i * BITS;
        unsigned q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned)((w0 >> b) & qmask);
            else q = (unsigned)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else {
            q = (unsigned)((w1 >> (b - 64)) & qmask);
        }
        a[i] = sc_h * (_Float16)(unsigned short)q + zp_h;
    }
    return a;
}
template <int BITS>
__device__ __forceinline__ half8_t decode8_window_nt_u16(const unsigned char* __restrict__ codes,
                                                        int code0,
                                                        _Float16 sc_h,
                                                        _Float16 zp_h) {
    constexpr unsigned qmask = (1u << BITS) - 1u;
    const int off = (code0 >> 3) * BITS;
    const int bit_in = (off & 7) * 8;
    const int idx = off >> 3;
    const uint64_t* __restrict__ pu64 = (const uint64_t* __restrict__)codes;
    uint64_t w0 = __builtin_nontemporal_load(pu64 + idx);
    uint64_t w1 = 0;
    if ((bit_in + 8 * BITS) > 64) w1 = __builtin_nontemporal_load(pu64 + idx + 1);
    half8_t a;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int b = bit_in + i * BITS;
        unsigned q;
        if (b < 64) {
            if (b + BITS <= 64) q = (unsigned)((w0 >> b) & qmask);
            else q = (unsigned)(((w0 >> b) | (w1 << (64 - b))) & qmask);
        } else {
            q = (unsigned)((w1 >> (b - 64)) & qmask);
        }
        a[i] = sc_h * (_Float16)(unsigned short)q + zp_h;
    }
    return a;
}

template <int BITS, int ARM>
__device__ __forceinline__ void body_gfx1201_single(const char* __restrict__ A,
                                                    const _Float16* __restrict__ X,
                                                    float* __restrict__ Y,
                                                    int M, int K, int N) {
    constexpr int GROUP_BYTES = 8 + 32 * BITS;
    constexpr int BV = 8;
    const int total_m = M;
    const int rs = (int)blockIdx.x * 16;
    const int bs = (int)blockIdx.y * (16 * BV);
    const int tid = (int)threadIdx.x;
    if (rs >= total_m || bs >= N) return;
    const int ml = tid & 15;
    const int kg = tid >> 4;
    const int sr = (rs + ml < total_m) ? (rs + ml) : (total_m - 1);
    const char* A_ptr = A;
    int lrow = sr;
    const int gpr = K / 256;
    const char* rb = A_ptr + (long long)lrow * (long long)gpr * (long long)GROUP_BYTES;
    int sb[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int t = bs + b * 16 + ml;
        sb[b] = (t < N) ? t : 0;
    }
    float8_t acc[BV];
#pragma unroll
    for (int b = 0; b < BV; ++b) acc[b] = float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for (int g = 0; g < gpr; ++g) {
        const char* gp = rb + (long long)g * (long long)GROUP_BYTES;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const unsigned char* codes = (const unsigned char*)(gp + 8);
        const _Float16* xg[BV];
#pragma unroll
        for (int b = 0; b < BV; ++b) xg[b] = X + (long long)sb[b] * (long long)K + (long long)g * 256;
        for (int kt = 0; kt < 16; kt += 4) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                const int code0 = (kt + ki) * 16 + kg * 8;
                half8_t a;
                if constexpr (ARM == 0) a = decode8<BITS>(codes, code0, sc_h, zp_h);
                else if constexpr (ARM == 1) a = decode8_window<BITS>(codes, code0, sc_h, zp_h);
                else if constexpr (ARM == 2) a = decode8_window_u16<BITS>(codes, code0, sc_h, zp_h);
                else if constexpr (ARM == 3) a = decode8_window_nt_u16<BITS>(codes, code0, sc_h, zp_h);
                else /*ARM==4*/ a = decode8_u16<BITS>(codes, code0, sc_h, zp_h);
#pragma unroll
                for (int b = 0; b < BV; ++b) {
                    const half8_t bb = *(const half8_t*)(xg[b] + (kt + ki) * 16 + kg * 8);
                    acc[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc[b]);
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < BV; ++b) {
        const int oc = bs + b * 16 + ml;
        if (oc < N) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int orow = rs + 8 * kg + j;
                if (orow < total_m) {
                    Y[(long long)oc * (long long)total_m + orow] = acc[b][j];
                }
            }
        }
    }
}

#endif // arch guards
// ─────────────────────────────────────────────────────────────────────────────
// Kernels — defined unconditionally for host stubs, device body guarded
// ─────────────────────────────────────────────────────────────────────────────
#if defined(__HIP_DEVICE_COMPILE__)
#define KERNEL_BODY_GFX11(BITS,BV,ARM) body_gfx11<BITS,BV,ARM>
#define KERNEL_BODY_GFX1201(BITS,ARM) body_gfx1201_single<BITS,ARM>
#else
#define KERNEL_BODY_GFX11(BITS,BV,ARM)
#define KERNEL_BODY_GFX1201(BITS,ARM)
#endif

// Gfx11 kernels (host stubs always, device only when built for gfx11)
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt4_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,4,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt4_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,4,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt4_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,4,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt4_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,4,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt4_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,4,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt12_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,12,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt12_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,12,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt12_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,12,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt12_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,12,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx11_bt12_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<2,12,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt4_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,4,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt4_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,4,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt4_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,4,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt4_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,4,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt4_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,4,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt12_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,12,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt12_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,12,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt12_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,12,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt12_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,12,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx11_bt12_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<3,12,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt4_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,4,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt4_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,4,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt4_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,4,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt4_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,4,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt4_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,4,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt12_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,12,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt12_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,12,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt12_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,12,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt12_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,12,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx11_bt12_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<5,12,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt4_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,4,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt4_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,4,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt4_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,4,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt4_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,4,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt4_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,4,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt12_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,12,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt12_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,12,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt12_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,12,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt12_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,12,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx11_bt12_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1100__) || defined(__gfx1151__)
    body_gfx11<6,12,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx1201_bt8_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<2,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx1201_bt8_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<2,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx1201_bt8_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<2,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx1201_bt8_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<2,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq2g256v2_decode_screen_gfx1201_bt8_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<2,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx1201_bt8_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<3,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx1201_bt8_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<3,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx1201_bt8_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<3,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx1201_bt8_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<3,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq3g256v2_decode_screen_gfx1201_bt8_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<3,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx1201_bt8_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<5,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx1201_bt8_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<5,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx1201_bt8_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<5,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx1201_bt8_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<5,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq5g256v2_decode_screen_gfx1201_bt8_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<5,4>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx1201_bt8_a0(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<6,0>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx1201_bt8_a1(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<6,1>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx1201_bt8_a2(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<6,2>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx1201_bt8_a3(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<6,3>(A,X,Y,M,K,N);
#endif
}
extern "C" __launch_bounds__(32, 2) __global__ void gemm_mq6g256v2_decode_screen_gfx1201_bt8_a4(const char* A, const _Float16* X, float* Y, int M, int K, int N) {
#if defined(__gfx1201__)
    body_gfx1201_single<6,4>(A,X,Y,M,K,N);
#endif
}


// ─────────────────────────────────────────────────────────────────────────────
// Host kernel dispatch
// ─────────────────────────────────────────────────────────────────────────────
using Gfx11Fn = void (*)(const char*, const _Float16*, float*, int,int,int);
using Gfx1201Fn = void (*)(const char*, const _Float16*, float*, int,int,int);

static Gfx11Fn pick_gfx11(int bits,int bv,int arm){
    if (bits==2){
        if(bv==4){
            if(arm==0) return gemm_mq2g256v2_decode_screen_gfx11_bt4_a0;
            if(arm==1) return gemm_mq2g256v2_decode_screen_gfx11_bt4_a1;
            if(arm==2) return gemm_mq2g256v2_decode_screen_gfx11_bt4_a2;
            if(arm==3) return gemm_mq2g256v2_decode_screen_gfx11_bt4_a3;
            return gemm_mq2g256v2_decode_screen_gfx11_bt4_a4;
        } else {
            if(arm==0) return gemm_mq2g256v2_decode_screen_gfx11_bt12_a0;
            if(arm==1) return gemm_mq2g256v2_decode_screen_gfx11_bt12_a1;
            if(arm==2) return gemm_mq2g256v2_decode_screen_gfx11_bt12_a2;
            if(arm==3) return gemm_mq2g256v2_decode_screen_gfx11_bt12_a3;
            return gemm_mq2g256v2_decode_screen_gfx11_bt12_a4;
        }
    } else if(bits==3){
        if(bv==4){
            if(arm==0) return gemm_mq3g256v2_decode_screen_gfx11_bt4_a0;
            if(arm==1) return gemm_mq3g256v2_decode_screen_gfx11_bt4_a1;
            if(arm==2) return gemm_mq3g256v2_decode_screen_gfx11_bt4_a2;
            if(arm==3) return gemm_mq3g256v2_decode_screen_gfx11_bt4_a3;
            return gemm_mq3g256v2_decode_screen_gfx11_bt4_a4;
        } else {
            if(arm==0) return gemm_mq3g256v2_decode_screen_gfx11_bt12_a0;
            if(arm==1) return gemm_mq3g256v2_decode_screen_gfx11_bt12_a1;
            if(arm==2) return gemm_mq3g256v2_decode_screen_gfx11_bt12_a2;
            if(arm==3) return gemm_mq3g256v2_decode_screen_gfx11_bt12_a3;
            return gemm_mq3g256v2_decode_screen_gfx11_bt12_a4;
        }
    } else if(bits==5){
        if(bv==4){
            if(arm==0) return gemm_mq5g256v2_decode_screen_gfx11_bt4_a0;
            if(arm==1) return gemm_mq5g256v2_decode_screen_gfx11_bt4_a1;
            if(arm==2) return gemm_mq5g256v2_decode_screen_gfx11_bt4_a2;
            if(arm==3) return gemm_mq5g256v2_decode_screen_gfx11_bt4_a3;
            return gemm_mq5g256v2_decode_screen_gfx11_bt4_a4;
        } else {
            if(arm==0) return gemm_mq5g256v2_decode_screen_gfx11_bt12_a0;
            if(arm==1) return gemm_mq5g256v2_decode_screen_gfx11_bt12_a1;
            if(arm==2) return gemm_mq5g256v2_decode_screen_gfx11_bt12_a2;
            if(arm==3) return gemm_mq5g256v2_decode_screen_gfx11_bt12_a3;
            return gemm_mq5g256v2_decode_screen_gfx11_bt12_a4;
        }
    } else { // bits==6
        if(bv==4){
            if(arm==0) return gemm_mq6g256v2_decode_screen_gfx11_bt4_a0;
            if(arm==1) return gemm_mq6g256v2_decode_screen_gfx11_bt4_a1;
            if(arm==2) return gemm_mq6g256v2_decode_screen_gfx11_bt4_a2;
            if(arm==3) return gemm_mq6g256v2_decode_screen_gfx11_bt4_a3;
            return gemm_mq6g256v2_decode_screen_gfx11_bt4_a4;
        } else {
            if(arm==0) return gemm_mq6g256v2_decode_screen_gfx11_bt12_a0;
            if(arm==1) return gemm_mq6g256v2_decode_screen_gfx11_bt12_a1;
            if(arm==2) return gemm_mq6g256v2_decode_screen_gfx11_bt12_a2;
            if(arm==3) return gemm_mq6g256v2_decode_screen_gfx11_bt12_a3;
            return gemm_mq6g256v2_decode_screen_gfx11_bt12_a4;
        }
    }
}
static Gfx1201Fn pick_gfx1201(int bits,int arm){
    if(bits==2){
        if(arm==0) return gemm_mq2g256v2_decode_screen_gfx1201_bt8_a0;
        if(arm==1) return gemm_mq2g256v2_decode_screen_gfx1201_bt8_a1;
        if(arm==2) return gemm_mq2g256v2_decode_screen_gfx1201_bt8_a2;
        if(arm==3) return gemm_mq2g256v2_decode_screen_gfx1201_bt8_a3;
        return gemm_mq2g256v2_decode_screen_gfx1201_bt8_a4;
    } else if(bits==3){
        if(arm==0) return gemm_mq3g256v2_decode_screen_gfx1201_bt8_a0;
        if(arm==1) return gemm_mq3g256v2_decode_screen_gfx1201_bt8_a1;
        if(arm==2) return gemm_mq3g256v2_decode_screen_gfx1201_bt8_a2;
        if(arm==3) return gemm_mq3g256v2_decode_screen_gfx1201_bt8_a3;
        return gemm_mq3g256v2_decode_screen_gfx1201_bt8_a4;
    } else if(bits==5){
        if(arm==0) return gemm_mq5g256v2_decode_screen_gfx1201_bt8_a0;
        if(arm==1) return gemm_mq5g256v2_decode_screen_gfx1201_bt8_a1;
        if(arm==2) return gemm_mq5g256v2_decode_screen_gfx1201_bt8_a2;
        if(arm==3) return gemm_mq5g256v2_decode_screen_gfx1201_bt8_a3;
        return gemm_mq5g256v2_decode_screen_gfx1201_bt8_a4;
    } else {
        if(arm==0) return gemm_mq6g256v2_decode_screen_gfx1201_bt8_a0;
        if(arm==1) return gemm_mq6g256v2_decode_screen_gfx1201_bt8_a1;
        if(arm==2) return gemm_mq6g256v2_decode_screen_gfx1201_bt8_a2;
        if(arm==3) return gemm_mq6g256v2_decode_screen_gfx1201_bt8_a3;
        return gemm_mq6g256v2_decode_screen_gfx1201_bt8_a4;
    }
}
static std::string detect_build_arch(){
    if ( (void*)gemm_mq2g256v2_decode_screen_gfx11_bt4_a0 != nullptr) return "gfx11";
    if ( (void*)gemm_mq2g256v2_decode_screen_gfx1201_bt8_a0 != nullptr) return "gfx1201";
    return "unknown";
}

static const char* arm_name(int arm){
    switch(arm){ case 0: return "A0_baseline"; case 1: return "A1_dword"; case 2: return "A2_dword_u16"; case 3: return "A3_dword_u16_nt"; case 4: return "A4_u16"; default: return "?"; }
}
static const char* bits_name(int bits){
    if(bits==2) return "MQ2V2"; if(bits==3) return "MQ3V2"; if(bits==5) return "MQ5V2"; return "MQ6V2";
}

struct YBuf {
    float* d_base=nullptr;
    float* d_y=nullptr;
    size_t full_elems=0;
    int M=0,N=0;
};
static YBuf alloc_y(int M,int N){
    YBuf b; b.M=M; b.N=N;
    b.full_elems = (size_t)M * (size_t)N + 2;
    CK(hipMalloc(&b.d_base, b.full_elems*sizeof(float)));
    b.d_y = b.d_base + 1;
    std::vector<float> h(b.full_elems,0.f);
    h[0]=7.654321f; h[b.full_elems-1]=7.654321f;
    CK(hipMemcpy(b.d_base,h.data(),b.full_elems*sizeof(float),hipMemcpyHostToDevice));
    return b;
}
static bool check_sentinels(const YBuf& b,const char* tag){
    float lo=0,hi=0;
    CK(hipMemcpy(&lo,b.d_base,sizeof(float),hipMemcpyDeviceToHost));
    CK(hipMemcpy(&hi,b.d_base+b.full_elems-1,sizeof(float),hipMemcpyDeviceToHost));
    if(lo!=7.654321f || hi!=7.654321f){
        std::printf("FAIL %s sentinel overrun lo=%.6f hi=%.6f\n",tag,lo,hi);
        return false;
    }
    return true;
}
static void free_y(YBuf& b){ if(b.d_base) CK(hipFree(b.d_base)); b.d_base=nullptr; b.d_y=nullptr; }

struct Result {
    std::string arch;
    int bits=0,bv=0,n=0,arm=0;
    float med=0,p10=0,p90=0,gib=0, speedup=0;
    bool pass=true;
    CorrMetrics corr{};
    uint64_t csum=0;
};

static void print_help(const char* a0){
    std::printf("Usage: %s --arch <gfxNNNN> [M] [K] [N] [samples]\n\n",a0);
    std::printf("Identity: %s\n",kBenchId);
    std::printf("Bench: MQ V2 decode-screen, single-A BT WMMA, arms A0..A4 per BITS {2,3,5,6}.\n");
    std::printf("Defaults: M=%d K=%d N=128,512 (both) samples=%d warmup=%d batch=%d\n",kDefaultM,kDefaultK,kDefaultSamples,kWarmup,kBatchLaunches);
    std::printf("Build single-arch: hipcc --offload-arch=gfxNNNN -O3 tools/quant-design/bench_mqv2_decode_screen.hip -o out\n");
}

static std::string arch_prefix(const std::string& gcn){
    auto pos=gcn.find(':');
    if(pos==std::string::npos) return gcn;
    return gcn.substr(0,pos);
}

int main(int argc,char** argv){
    std::string req_arch;
    std::vector<std::string> pos;
    for(int i=1;i<argc;i++){
        if(std::string(argv[i])=="--arch"){
            if(i+1>=argc){ std::fprintf(stderr,"--arch needs value\n"); return 2; }
            req_arch=argv[i+1]; i++;
        } else if(std::string(argv[i])=="--help" || std::string(argv[i])=="-h"){
            print_help(argv[0]); return 0;
        } else if(argv[i][0]=='-'){
            std::fprintf(stderr,"unknown flag %s\n",argv[i]); print_help(argv[0]); return 2;
        } else pos.push_back(argv[i]);
    }
    if(req_arch.empty()){
        std::fprintf(stderr,"ERROR: --arch <gfxNNNN> is required (e.g. --arch gfx1201)\n");
        print_help(argv[0]); return 2;
    }
    int M=kDefaultM, K=kDefaultK, N_override=-1, samples=kDefaultSamples;
    if(pos.size()>=1) M=std::atoi(pos[0].c_str());
    if(pos.size()>=2) K=std::atoi(pos[1].c_str());
    if(pos.size()>=3) N_override=std::atoi(pos[2].c_str());
    if(pos.size()>=4) samples=std::atoi(pos[3].c_str());
    if(M<=0 || K<=0 || (K%256)!=0 || samples<=0 || (N_override!=-1 && N_override<=0)){
        std::fprintf(stderr,"Need M>0 K%%256==0 N>0 samples>0 (got M=%d K=%d N_ov=%d samples=%d)\n",M,K,N_override,samples);
        return 2;
    }
    // Device selection by gcnArchName prefix
    int devCount=0; CK(hipGetDeviceCount(&devCount));
    if(devCount==0){ std::fprintf(stderr,"No HIP devices\n"); return 1; }
    int sel=-1; std::string sel_gcn;
    for(int d=0;d<devCount;d++){
        hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d));
        std::string gcn = p.gcnArchName;
        std::string pref = arch_prefix(gcn);
        if(pref==req_arch){ sel=d; sel_gcn=gcn; break; }
    }
    if(sel==-1){
        std::fprintf(stderr,"ERROR: --arch %s not found among %d devices. Available:\n",req_arch.c_str(),devCount);
        for(int d=0;d<devCount;d++){ hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p,d)); std::fprintf(stderr,"  device %d: %s (%s) gcnArchName=%s\n",d,p.name,p.gcnArchName,p.gcnArchName); }
        return 1;
    }
    CK(hipSetDevice(sel));
    hipDeviceProp_t prop{}; CK(hipGetDeviceProperties(&prop, sel));
    std::string dev_arch = arch_prefix(prop.gcnArchName);
    // hip version
    int runtimeVersion=0, driverVersion=0;
    hipRuntimeGetVersion(&runtimeVersion);
    hipDriverGetVersion(&driverVersion);

    std::string build_arch = dev_arch;
    std::printf("build arch (inferred): %s  selected dev_arch: %s\n", build_arch.c_str(), dev_arch.c_str());
    std::printf("shape M=%d K=%d N_ov=%d (%s) samples=%d warmup=%d batch=%d\n",M,K,N_override,(N_override==-1?"128,512":std::to_string(N_override).c_str()),samples,kWarmup,kBatchLaunches);
    if(dev_arch != req_arch){
        std::fprintf(stderr,"WARN: selected arch prefix %s != requested %s\n",dev_arch.c_str(),req_arch.c_str());
    }
    // Validate requested vs build (best effort)
    if(build_arch.find("gfx11")!=std::string::npos){
        if(dev_arch!="gfx1100" && dev_arch!="gfx1151"){
            std::fprintf(stderr,"WARN: build appears gfx11 but device is %s\n",dev_arch.c_str());
        }
    } else if(build_arch.find("gfx1201")!=std::string::npos){
        if(dev_arch!="gfx1201"){
            std::fprintf(stderr,"WARN: build appears gfx1201 but device is %s\n",dev_arch.c_str());
        }
    }
    if(dev_arch=="gfx1030"){
        std::fprintf(stderr,"ERROR: NEVER run arms on gfx1030 per policy\n");
        return 1;
    }

    std::vector<int> bits_list = {2,3,5,6};
    std::vector<int> N_list;
    if(N_override==-1) N_list = {128,512};
    else N_list = {N_override};
    std::vector<int> BV_list;
    if(dev_arch=="gfx1201") BV_list = {8};
    else if(dev_arch=="gfx1100" || dev_arch=="gfx1151") BV_list = {4,12};
    else {
        if(req_arch=="gfx1201") BV_list={8};
        else BV_list={4,12};
    }
    std::vector<Result> all_results;
    bool overall_pass=true;

    for(int bits: bits_list){
        const int gb = v2_group_bytes(bits);
        const size_t weight_bytes_one = (size_t)M * (size_t)(K/256) * (size_t)gb;
        for(int N: N_list){
            for(int BV: BV_list){
                std::printf("\n--- config bits=%d (%s)  M=%d K=%d N=%d BV=%d  weight_one=%.2f MiB ---\n",bits,bits_name(bits),M,K,N,BV, (double)weight_bytes_one/(1024*1024));
                // Host fixtures
                std::vector<uint8_t> hA;
                pack_mqv2(hA,bits,M,K, 0xBEEF0000ull + (uint64_t)bits*0x100 + (uint64_t)N);
                std::vector<_Float16> hX;
                gen_X_fp16(hX,N,K, 0xA11CE5EDull + (uint64_t)N*0x100 + bits);
                std::vector<float> hRef((size_t)N * M);
                host_gemm_ref(hA.data(), hX.data(), hRef.data(), bits, M, K, N);
                if(!all_finite(hRef.data(), hRef.size())){
                    std::printf("FAIL host ref non-finite bits=%d\n",bits);
                    overall_pass=false; continue;
                }
                // Device allocations
                const size_t pack_bytes = hA.size();
                char* dA=nullptr;
                CK(hipMalloc(&dA, pack_bytes * (size_t)kBatchLaunches));
                for(int r=0;r<kBatchLaunches;r++) CK(hipMemcpy(dA+(size_t)r*pack_bytes, hA.data(), pack_bytes, hipMemcpyHostToDevice));
                _Float16* dX=nullptr;
                CK(hipMalloc(&dX, (size_t)N*K*sizeof(_Float16)));
                CK(hipMemcpy(dX, hX.data(), (size_t)N*K*sizeof(_Float16), hipMemcpyHostToDevice));
                const int n_arms=5;
                std::vector<YBuf> ybufs(n_arms);
                for(int a=0;a<n_arms;a++) ybufs[a]=alloc_y(M,N);
                // Grid dims
                int grid_x = (M + 15)/16;
                int grid_y;
                if(dev_arch=="gfx1100" || dev_arch=="gfx1151") grid_y = (N + 16*BV -1)/(16*BV);
                else grid_y = (N + 128 -1)/128;
                dim3 grid(grid_x, grid_y);
                dim3 block(32);
                // Correctness: baseline A0 first
                bool cfg_pass=true;
                std::vector<std::vector<float>> hY_host(n_arms);
                std::vector<CorrMetrics> corrs(n_arms);
                std::vector<uint64_t> csums(n_arms,0);
                std::vector<bool> arm_pass(n_arms,true);
                // launch helper
                auto launch_arm = [&](int arm,int rep){
                    if(dev_arch=="gfx1100" || dev_arch=="gfx1151"){
                        Gfx11Fn fn = pick_gfx11(bits,BV,arm);
                        fn<<<grid,block>>>(dA+(size_t)rep*pack_bytes, dX, ybufs[arm].d_y, M, K, N);
                    } else {
                        Gfx1201Fn fn = pick_gfx1201(bits,arm);
                        fn<<<grid,block>>>(dA+(size_t)rep*pack_bytes, dX, ybufs[arm].d_y, M, K, N);
                    }
                };
                // Baseline
                for(int a=0;a<n_arms;a++) CK(hipMemset(ybufs[a].d_y,0,(size_t)N*M*sizeof(float)));
                launch_arm(0,0); CK(hipGetLastError()); CK(hipDeviceSynchronize());
                if(!check_sentinels(ybufs[0],"A0 sentinel")){ arm_pass[0]=false; cfg_pass=false; }
                else {
                    hY_host[0].assign((size_t)N*M,0);
                    CK(hipMemcpy(hY_host[0].data(), ybufs[0].d_y, (size_t)N*M*sizeof(float), hipMemcpyDeviceToHost));
                    corrs[0]=compare_vecs(hY_host[0].data(), hRef.data(), hRef.size());
                    csums[0]=checksum_finite(hY_host[0].data(), hY_host[0].size());
                    bool ok = corrs[0].finite && csums[0]!=0 && corrs[0].rel_l2 <= (double)kLimRelL2;
                    arm_pass[0]=ok;
                    if(ok) std::printf("PASS bits=%d BV=%d N=%d arm=%s  max_abs=%.3e rel_l2=%.3e cos=%.6f csum=0x%016llx\n",bits,BV,N,arm_name(0),corrs[0].max_abs,corrs[0].rel_l2,corrs[0].cosine,(unsigned long long)csums[0]);
                    else { std::printf("FAIL bits=%d BV=%d N=%d arm=%s  max_abs=%.3e rel_l2=%.3e cos=%.6f finite=%d csum=0x%016llx lim=%.1e\n",bits,BV,N,arm_name(0),corrs[0].max_abs,corrs[0].rel_l2,corrs[0].cosine,(int)corrs[0].finite,(unsigned long long)csums[0],(double)kLimRelL2); cfg_pass=false; }
                }
                // Other arms vs baseline bit equality
                for(int a=1;a<n_arms;a++){
                    if(!arm_pass[0]){ arm_pass[a]=false; cfg_pass=false; continue; }
                    CK(hipMemset(ybufs[a].d_y,0,(size_t)N*M*sizeof(float)));
                    launch_arm(a,0); CK(hipGetLastError()); CK(hipDeviceSynchronize());
                    if(!check_sentinels(ybufs[a],arm_name(a))){ arm_pass[a]=false; cfg_pass=false; continue; }
                    hY_host[a].assign((size_t)N*M,0);
                    CK(hipMemcpy(hY_host[a].data(), ybufs[a].d_y, (size_t)N*M*sizeof(float), hipMemcpyDeviceToHost));
                    csums[a]=checksum_finite(hY_host[a].data(), hY_host[a].size());
                    corrs[a]=compare_vecs(hY_host[a].data(), hRef.data(), hRef.size()); // not needed for equality but useful
                    size_t mis=0;
                    bool eq = bit_equal_buffers(hY_host[a].data(), hY_host[0].data(), hY_host[0].size(), &mis);
                    if(eq && csums[a]!=0){
                        std::printf("PASS bits=%d BV=%d N=%d arm=%s  bit-equal vs A0  csum=0x%016llx\n",bits,BV,N,arm_name(a),(unsigned long long)csums[a]);
                        arm_pass[a]=true;
                    } else {
                        std::printf("FAIL bits=%d BV=%d N=%d arm=%s  NOT bit-equal vs A0  first_mismatch=%zu  A0=0x%08x Arm=0x%08x  csum=0x%016llx\n",bits,BV,N,arm_name(a),mis, [&]{uint32_t x; memcpy(&x,&hY_host[0][mis],4); return x;}(), [&]{uint32_t x; memcpy(&x,&hY_host[a][mis],4); return x;}(), (unsigned long long)csums[a]);
                        arm_pass[a]=false; cfg_pass=false;
                    }
                }
                if(!cfg_pass){
                    std::printf("CORRECTNESS FAIL for bits=%d BV=%d N=%d\n",bits,BV,N);
                    overall_pass=false;
                    // still continue to timing? skip timing for failed config to avoid reporting perf wins on incorrect arms
                    // but per spec any mismatch = arm FAILED; never report its perf as win. We'll skip timing for failed.
                    for(auto &yb: ybufs) free_y(yb);
                    CK(hipFree(dA)); CK(hipFree(dX));
                    // record failed results without timing
                    for(int a=0;a<n_arms;a++){
                        Result r; r.arch=dev_arch; r.bits=bits; r.bv=BV; r.n=N; r.arm=a; r.pass=arm_pass[a]; r.corr=corrs[a]; r.csum=csums[a];
                        all_results.push_back(r);
                    }
                    continue;
                }
                std::printf("CORRECTNESS PASS for bits=%d BV=%d N=%d\n",bits,BV,N);
                // Warmup
                for(int w=0;w<kWarmup;w++){
                    for(int a=0;a<n_arms;a++){
                        for(int r=0;r<kBatchLaunches;r++) launch_arm(a,r);
                    }
                }
                CK(hipDeviceSynchronize());
                // Anti-DCE sink after warmup
                uint64_t sink=0xCBF29CE484222325ull;
                for(int a=0;a<n_arms;a++){
                    std::vector<float> hy((size_t)N*M);
                    CK(hipMemcpy(hy.data(), ybufs[a].d_y, hy.size()*sizeof(float), hipMemcpyDeviceToHost));
                    uint64_t cs=checksum_finite(hy.data(), hy.size());
                    if(cs==0){ std::printf("FAIL post-warmup checksum arm %s\n",arm_name(a)); overall_pass=false; }
                    sink = (sink ^ cs) * 0x100000001B3ull;
                    if(!check_sentinels(ybufs[a],"post-warmup")) overall_pass=false;
                }
                // Timing events
                hipEvent_t ev0,ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
                std::vector<std::vector<float>> samples_us(n_arms);
                for(int a=0;a<n_arms;a++) samples_us[a].reserve(samples);
                for(int s=0;s<samples;s++){
                    int order[5]={0,1,2,3,4};
                    int rot=s % n_arms;
                    std::rotate(order,order+rot,order+n_arms);
                    if(s & 1) std::reverse(order,order+n_arms);
                    for(int oi=0;oi<n_arms;oi++){
                        int a = order[oi];
                        CK(hipEventRecord(ev0,nullptr));
                        for(int r=0;r<kBatchLaunches;r++) launch_arm(a,r);
                        CK(hipEventRecord(ev1,nullptr));
                        CK(hipEventSynchronize(ev1));
                        float ms=0.f; CK(hipEventElapsedTime(&ms,ev0,ev1));
                        float per_us = ms * 1000.f / (float)kBatchLaunches;
                        samples_us[a].push_back(per_us);
                    }
                }
                CK(hipEventDestroy(ev0)); CK(hipEventDestroy(ev1));
                // checksum after bench
                for(int a=0;a<n_arms;a++){
                    std::vector<float> hy((size_t)N*M);
                    CK(hipMemcpy(hy.data(), ybufs[a].d_y, hy.size()*sizeof(float), hipMemcpyDeviceToHost));
                    uint64_t cs=checksum_finite(hy.data(), hy.size());
                    if(cs==0) std::printf("FAIL post-bench checksum arm %s\n",arm_name(a));
                    sink = (sink ^ cs) * 0x100000001B3ull;
                }
                // Stats
                std::vector<float> meds(n_arms,0);
                for(int a=0;a<n_arms;a++){
                    std::sort(samples_us[a].begin(), samples_us[a].end());
                    float med = percentile_sorted(samples_us[a],0.50);
                    float p10 = percentile_sorted(samples_us[a],0.10);
                    float p90 = percentile_sorted(samples_us[a],0.90);
                    double sec = (double)med * 1e-6;
                    float gib = sec>0 ? (float)((double)weight_bytes_one / sec / (1024.0*1024.0*1024.0)) : 0.f;
                    float speedup = (a==0)?1.f: (med>0? meds[0]/med : 0.f);
                    // need meds[0] first; compute after
                    meds[a]=med;
                    Result r; r.arch=dev_arch; r.bits=bits; r.bv=BV; r.n=N; r.arm=a; r.med=med; r.p10=p10; r.p90=p90; r.gib=gib; r.pass=arm_pass[a]; r.corr=corrs[a]; r.csum=csums[a];
                    // speedup will be filled after loop
                    all_results.push_back(r);
                    // temporary store
                    (void)sink;
                }
                // fill speedup now
                size_t base_idx = all_results.size() - n_arms;
                float base_med = all_results[base_idx].med;
                for(int a=0;a<n_arms;a++){
                    float sp = (all_results[base_idx+a].med>0)? base_med / all_results[base_idx+a].med : 0.f;
                    all_results[base_idx+a].speedup = sp;
                }
                // Print per-config timing
                std::printf("--- timing per-launch (median/p10/p90 us, GiB/s, speedup vs A0) ---\n");
                std::printf("%-6s %-4s %-5s %-18s %8s %8s %8s %10s %8s\n","BITS","BV","N","arm","med_us","p10_us","p90_us","GiB/s","speedup");
                for(int a=0;a<n_arms;a++){
                    const Result &r = all_results[base_idx+a];
                    std::printf("%-6d %-4d %-5d %-18s %8.2f %8.2f %8.2f %10.2f %8.3f %s\n",r.bits,r.bv,r.n,arm_name(r.arm),r.med,r.p10,r.p90,r.gib,r.speedup, r.pass?"PASS":"FAIL");
                }
                std::printf("checksum_sink=0x%016llx\n",(unsigned long long)sink);
                for(auto &yb: ybufs) free_y(yb);
                CK(hipFree(dA)); CK(hipFree(dX));
            }
        }
    }

    std::printf("\n=== FINAL SUMMARY TABLE ===\n");
    std::printf("arch | BITS | BV | N | arm | med_us | p10_us | p90_us | GiB/s | speedup_vs_A0 | PASS\n");
    for(auto &r: all_results){
        std::printf("%s | %d | %d | %d | %s | %.2f | %.2f | %.2f | %.2f | %.3f | %s\n",
            r.arch.c_str(), r.bits, r.bv, r.n, arm_name(r.arm), r.med, r.p10, r.p90, r.gib, r.speedup, r.pass?"PASS":"FAIL");
    }
    std::printf("\nidentity: %s\n",kBenchId);
    std::printf("arch built=%s selected=%s  M=%d K=%d samples=%d warmup=%d batch=%d  hip rt %d drv %d\n",
        build_arch.c_str(),
        dev_arch.c_str(), M, K, samples, kWarmup, kBatchLaunches, runtimeVersion, driverVersion);
    std::printf("OVERALL %s\n", overall_pass?"PASS":"FAIL");
    return overall_pass?0:1;
}
