// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Self-contained prefill microbenchmark: MQ4V2 F16-WMMA vs Q8_1 i8-WMMA MMQ.
// Fleet fatbin covers every current hipx architecture; unsupported gfx1030 exits cleanly.
//
// Derived from:
//   kernels/src/gemm_hfq4g256_residual_mmq.gfx12.hip   (gfx12 i8 WMMA MMQ body)
//   kernels/src/gemm_hfq4g256_residual_mmq.hip         (RDNA3 LDS-tile MMQ + quantize)
//   kernels/src/gemm_mq4g256v2_residual_wmma.gfx12.hip (gfx12 V2 F16-WMMA)
//   kernels/src/gemm_mq4g256v2_residual_wmma.hip       (gfx11 V2 F16-WMMA half16)
//   kernels/src/gemm_hfq4g256_residual_fp16.hip        (convert_f32_to_f16)
//
// Arms (CSV-stable names on every family):
//   1. mq4v2_f16_compute   — V2 F16-WMMA, preconverted F16 X (compute-only)
//   2. mq4v2_f16_e2e       — F32→F16 + V2 F16-WMMA (conversion every iter)
//   3. mq4v2_mmq_compute   — V2 Q8_1 i8-WMMA MMQ, prequantized X (compute-only)
//   4. mq4v2_mmq_e2e       — F32→Q8_1 + V2 MMQ (conversion every iter)
//   5. hfq4_mmq_legacy     — legacy HFQ4 f32-header MMQ compute-only (header isolate)
//
// Family mechanisms:
//   gfx12  (gfx1200/gfx1201): 16×16 tiles, block[32], half8 / int32x2 WMMA,
//          contiguous C rows; no LDS. Builtins *_w32_gfx12.
//   RDNA3  (gfx1100/01/02/03, gfx1150/51): F16 = half16 + interleaved C mapping;
//          MMQ = 128×128 LDS tile, block[32,8,1], int32x4 + iu8_w32 (no _gfx12).
//          V2 half select only change vs production load_hfq4_tile.
//
// V2 wire: 136 B/G256 = [fp16 s0,z0,s1,z1] + 128 B nibble payload at +8.
// MMQ half select: K 0..127 → half0, K 128..255 → half1.
// Affine: s_half * d_x * dot(qw,qx) + z_half * sum_real_x.
// Output: column-major Y[col * M + row], SET (overwrite), not residual add.
//
// Correctness gate (before timing): dedicated M=32 K=512 N=16 buffers (canary
// immediately after Y); disjoint V2 half headers for half-select; header-control
// pair (identical nibbles, legacy f32 s/z repeated on both V2 fp16 halves);
// F16 ref uses float X; MMQ refs use Q8_1 d*q activations (ds4.y = d*sum(q8));
// rel-RMS <= 0.01; exit nonzero on failure.
//
// Timing: HIP events, 32 warmups, 100 measured iters, 5 independent repeats.
// Report median us, effective weight+activation GiB/s, ratios vs f16 compute.
//
// Build (hipx fleet fatbin — never single-arch):
//   /opt/rocm/core/bin/hipcc -O3 \
//       --offload-arch=gfx1030 --offload-arch=gfx1100 \
//       --offload-arch=gfx1151 --offload-arch=gfx1201 \
//       tools/quant-design/bench_mq4v2_prefill_mmq.hip \
//       -o /tmp/bench_mq4v2_prefill_mmq
//
// Run:
//   /tmp/bench_mq4v2_prefill_mmq [M] [K] [Nlist]
//   Defaults: M=5120 K=5120 Nlist=16,32,64,128,256,512,1024,2048
//   K must be a multiple of 256. Nlist is comma-separated positive ints.
//
// CSV (stdout): arm,M,K,N,med_us,w_bytes,act_bytes,eff_gib_s,ratio_f16_compute
// Human table: stderr.

#include <hip/hip_fp16.h>
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

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time GPU family banner (device-side macros; host may see "host")
// ─────────────────────────────────────────────────────────────────────────────

#if defined(__gfx1200__) || defined(__gfx1201__)
#define BENCH_COMPILE_FAMILY "gfx12"
#define BENCH_HAS_GFX12_DEVICE 1
#pragma message("bench_mq4v2_prefill_mmq: device kernels = gfx12 (16x16 WMMA, no LDS)")
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define BENCH_COMPILE_FAMILY "rdna3"
#define BENCH_HAS_RDNA3_DEVICE 1
#pragma message("bench_mq4v2_prefill_mmq: device kernels = rdna3/gfx11 (F16 half16 + MMQ 128x128 LDS)")
#else
#define BENCH_COMPILE_FAMILY "host-or-other"
#endif

#ifndef BENCH_HAS_GFX12_DEVICE
#define BENCH_HAS_GFX12_DEVICE 0
#endif
#ifndef BENCH_HAS_RDNA3_DEVICE
#define BENCH_HAS_RDNA3_DEVICE 0
#endif

static constexpr int kGroupBytes = 136;
static constexpr int kGroupK = 256;
static constexpr int kQK8_1 = 32;
static constexpr int kWarmup = 32;
static constexpr int kIters = 100;
static constexpr int kRepeats = 5;
static constexpr float kRelRmsLimit = 0.01f;
static constexpr float kCanary = 7.654321f;

// RDNA3 MMQ LDS-tile geometry (gemm_hfq4g256_residual_mmq.hip)
static constexpr int kMmqX = 128;
static constexpr int kMmqY = 128;
static constexpr int kMmqNwarps = 8;
static constexpr int kMmqTileNeK = 32;
static constexpr int kMmqTileYK = 36;
static constexpr int kMmqTileXK = 76;
static constexpr int kQi8_1 = 8;
static constexpr size_t kMmqSharedBytes =
    (size_t)(kMmqX * kMmqTileYK + kMmqY * kMmqTileXK) * sizeof(int);

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                    \
                         hipGetErrorString(e_), __FILE__, __LINE__);           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

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
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(1 - exp);
        uint32_t half = mant >> (shift + 13);
        const uint32_t rem = mant & ((1u << (shift + 13)) - 1u);
        if (rem > (1u << (shift + 12)) ||
            (rem == (1u << (shift + 12)) && (half & 1u)))
            half++;
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
        if (mant == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

static float median_us(std::vector<float>& v) {
    if (v.empty()) return 0.f;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n & 1u) return v[n / 2];
    return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

enum class GpuFamily { Gfx12, Rdna3, Unknown };

// Device binary stamps its offload family so host launch geometry cannot
// disagree with the kernel body (LDS=0 + RDNA3 body → GPU hang on smem[]).
#if defined(__gfx1200__) || defined(__gfx1201__)
__device__ int bench_device_family_id = 12;
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
__device__ int bench_device_family_id = 11;
#else
__device__ int bench_device_family_id = 0;
#endif

static GpuFamily detect_family(const hipDeviceProp_t& prop) {
    // Prefer the family baked into the device code object.
    int fam = 0;
    if (hipMemcpyFromSymbol(&fam, bench_device_family_id, sizeof(fam)) ==
        hipSuccess) {
        if (fam == 12) return GpuFamily::Gfx12;
        if (fam == 11) return GpuFamily::Rdna3;
    }
    const char* a = prop.gcnArchName;
    if (!a) a = "";
    if (std::strstr(a, "gfx120")) return GpuFamily::Gfx12;
    if (std::strstr(a, "gfx110") || std::strstr(a, "gfx115")) return GpuFamily::Rdna3;
    if (std::strstr(prop.name, "gfx120")) return GpuFamily::Gfx12;
    if (std::strstr(prop.name, "gfx110") || std::strstr(prop.name, "gfx115"))
        return GpuFamily::Rdna3;
    return GpuFamily::Unknown;
}

static const char* family_name(GpuFamily f) {
    switch (f) {
    case GpuFamily::Gfx12: return "gfx12";
    case GpuFamily::Rdna3: return "rdna3";
    default: return "unknown";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device: Q8_1 block + conversion (arch-portable)
// ─────────────────────────────────────────────────────────────────────────────

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * kQK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// Q8_1 MMQ block: ds4[i] = half2(d, sum_real) over each 32-K sub-block where
// sum_real = d * sum(rounded_q8). This is the exact zero-point partner for
//   s * d * dot(qw,qx) + z * sum_real
// (sum of original F32 is only an approximation and must not be used here).
extern "C" __global__ void quantize_q8_1_mmq_ds4(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    constexpr int WARP_SIZE = 32;
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    // Reduce sum of rounded int8 codes over the 32-K sub-block, then scale by d.
    float sum_q = (float)q.x + (float)q.y + (float)q.z + (float)q.w;
    for (int offset = 4; offset > 0; offset >>= 1)
        sum_q += __shfl_xor(sum_q, offset, WARP_SIZE);
    const float sum_real = d * sum_q;

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;

    if ((iqs & 31) == 0)
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum_real);
}

extern "C" __launch_bounds__(256)
__global__ void convert_f32_to_f16(
    const float* __restrict__ in,
    _Float16* __restrict__ out,
    int n
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = (_Float16)in[i];
}

typedef float __attribute__((ext_vector_type(8))) float8_t;

// ─────────────────────────────────────────────────────────────────────────────
// gfx12 device kernels
// ─────────────────────────────────────────────────────────────────────────────

#if defined(__gfx1200__) || defined(__gfx1201__)

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef int8_t   __attribute__((ext_vector_type(8))) int8x8_t;
typedef int32_t  __attribute__((ext_vector_type(2))) int32x2_t;
typedef int32_t  __attribute__((ext_vector_type(8))) int32x8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void bench_mq4v2_f16_wmma_set(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;
    if (row_start >= M || col_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int safe_col = (col_start + m_lane < N) ? (col_start + m_lane) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * kGroupBytes;
    const _Float16* x_base = X + (long long)safe_col * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * kGroupBytes;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const _Float16* xg = x_base + g * 256;

        #define DQ8(pk) \
            a_reg[0] = sc_h * (_Float16)(float)(((pk) >>  0) & 0xFu) + zp_h; \
            a_reg[1] = sc_h * (_Float16)(float)(((pk) >>  4) & 0xFu) + zp_h; \
            a_reg[2] = sc_h * (_Float16)(float)(((pk) >>  8) & 0xFu) + zp_h; \
            a_reg[3] = sc_h * (_Float16)(float)(((pk) >> 12) & 0xFu) + zp_h; \
            a_reg[4] = sc_h * (_Float16)(float)(((pk) >> 16) & 0xFu) + zp_h; \
            a_reg[5] = sc_h * (_Float16)(float)(((pk) >> 20) & 0xFu) + zp_h; \
            a_reg[6] = sc_h * (_Float16)(float)(((pk) >> 24) & 0xFu) + zp_h; \
            a_reg[7] = sc_h * (_Float16)(float)(((pk) >> 28) & 0xFu) + zp_h

        for (int kt = 0; kt < 16; kt += 4) {
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;
            const int k_off_a = (kt + 0) * 16;
            const int k_off_b = (kt + 1) * 16;
            const int k_off_c = (kt + 2) * 16;
            const int k_off_d = (kt + 3) * 16;

            unsigned int pka = *(const unsigned int*)(gp + 8 + k_off_a / 2 + k_grp * 4);
            unsigned int pkb = *(const unsigned int*)(gp + 8 + k_off_b / 2 + k_grp * 4);
            unsigned int pkc = *(const unsigned int*)(gp + 8 + k_off_c / 2 + k_grp * 4);
            unsigned int pkd = *(const unsigned int*)(gp + 8 + k_off_d / 2 + k_grp * 4);

            half8_t b_a = *(const half8_t*)(xg + k_off_a + k_grp * 8);
            half8_t b_b = *(const half8_t*)(xg + k_off_b + k_grp * 8);
            half8_t b_c = *(const half8_t*)(xg + k_off_c + k_grp * 8);
            half8_t b_d = *(const half8_t*)(xg + k_off_d + k_grp * 8);

            half8_t a_reg;
            DQ8(pka);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_a, acc);
            DQ8(pkb);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_b, acc);
            DQ8(pkc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_c, acc);
            DQ8(pkd);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_reg, b_d, acc);
        }
        #undef DQ8
    }

    const int out_col = col_start + m_lane;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M)
                Y[(long long)out_col * M + out_row] = acc[j];
        }
    }
}

template <bool V2>
static __device__ __forceinline__ void bench_mmq_gfx12_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;
    if (row_start >= M || col_start >= N) return;

    const int m_lane = tid & 15;
    const int k_grp  = tid >> 4;
    const int col = col_start + m_lane;
    const bool col_valid = (col < N);

    const int safe_row = (row_start + m_lane < M) ? (row_start + m_lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * kGroupBytes;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int blocks128 = K / 128;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * kGroupBytes;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);

        float sc_legacy = 0.f, zp_legacy = 0.f;
        if constexpr (!V2) {
            sc_legacy = __builtin_bit_cast(float, hA);
            zp_legacy = __builtin_bit_cast(float, hB);
        }

        for (int kb_in_group = 0; kb_in_group < 2; kb_in_group++) {
            const int kb_q8 = 2 * g + kb_in_group;
            if (kb_q8 >= blocks128) break;

            float sc_self, zp_self;
            if constexpr (V2) {
                const unsigned int hs = (kb_in_group == 0) ? hA : hB;
                sc_self = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
                zp_self = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
            } else {
                sc_self = sc_legacy;
                zp_self = zp_legacy;
            }

            float sc_row[8];
            float zp_row[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int src_lane = 8 * k_grp + j;
                sc_row[j] = __shfl(sc_self, src_lane);
                zp_row[j] = __shfl(zp_self, src_lane);
            }

            const block_q8_1_mmq* xb = col_valid
                ? (Xq + (long long)kb_q8 * N + col)
                : nullptr;

            unsigned int pk_arr[8];
            int8x8_t b_arr[8];
            float d_x_arr[4];
            float sum_x_arr[4];

            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                if (xb != nullptr) {
                    half2 dm = xb->ds4[sb];
                    float2 dm_f = __half22float2(dm);
                    d_x_arr[sb]   = dm_f.x;
                    sum_x_arr[sb] = dm_f.y;
                } else {
                    d_x_arr[sb]   = 0.0f;
                    sum_x_arr[sb] = 0.0f;
                }
                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    const int slot = sb * 2 + kt;
                    const int k_off = kb_in_group * 128 + sb * 32 + kt * 16 + k_grp * 8;
                    pk_arr[slot] = *(const unsigned int*)(gp + 8 + k_off / 2);
                    if (xb != nullptr) {
                        const int8_t* qs_p = xb->qs + sb * 32 + kt * 16 + k_grp * 8;
                        b_arr[slot] = *(const int8x8_t*)qs_p;
                    } else {
                        b_arr[slot] = (int8x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }
            }

            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                int32x8_t cacc = {0, 0, 0, 0, 0, 0, 0, 0};
                #pragma unroll
                for (int kt = 0; kt < 2; kt++) {
                    const int slot = sb * 2 + kt;
                    const unsigned int pk = pk_arr[slot];
                    int8x8_t a_vec;
                    a_vec[0] = (int8_t)((pk >>  0) & 0xFu);
                    a_vec[1] = (int8_t)((pk >>  4) & 0xFu);
                    a_vec[2] = (int8_t)((pk >>  8) & 0xFu);
                    a_vec[3] = (int8_t)((pk >> 12) & 0xFu);
                    a_vec[4] = (int8_t)((pk >> 16) & 0xFu);
                    a_vec[5] = (int8_t)((pk >> 20) & 0xFu);
                    a_vec[6] = (int8_t)((pk >> 24) & 0xFu);
                    a_vec[7] = (int8_t)((pk >> 28) & 0xFu);
                    int32x2_t a_i32 = *(const int32x2_t*)&a_vec;
                    int32x2_t b_i32 = *(const int32x2_t*)&b_arr[slot];
                    cacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        false, a_i32, true, b_i32, cacc, true
                    );
                }
                const float d_x_self   = d_x_arr[sb];
                const float sum_x_self = sum_x_arr[sb];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float dot_f = (float)cacc[j];
                    acc[j] += sc_row[j] * d_x_self * dot_f
                            + zp_row[j] * sum_x_self;
                }
            }
        }
    }

    if (col_valid) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 8 * k_grp + j;
            if (out_row < M)
                Y[(long long)col * M + out_row] = acc[j];
        }
    }
}

__launch_bounds__(32, 2)
extern "C" __global__ void bench_mq4v2_mmq_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    bench_mmq_gfx12_body<true>(A, Xq, Y, M, K, N);
}

__launch_bounds__(32, 2)
extern "C" __global__ void bench_hfq4_mmq_legacy_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    bench_mmq_gfx12_body<false>(A, Xq, Y, M, K, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// RDNA3 / gfx11 device kernels
// ─────────────────────────────────────────────────────────────────────────────

#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;

// V2 F16-WMMA: half16 operands, interleaved C (gemm_mq4g256v2_residual_wmma.hip)
#if defined(__gfx1150__) || defined(__gfx1151__)
__launch_bounds__(32, 8)
#else
__launch_bounds__(32, 2)
#endif
extern "C" __global__ void bench_mq4v2_f16_wmma_set(
    const char* __restrict__ A,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int row_start = blockIdx.x * 16;
    const int col_start = blockIdx.y * 16;
    const int tid = threadIdx.x;
    if (row_start >= M || col_start >= N) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_col = (col_start + (tid & 15) < N) ? (col_start + (tid & 15)) : 0;

    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * kGroupBytes;
    const _Float16* x_base = X + (long long)safe_col * K;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * kGroupBytes;
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const _Float16 sc0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA & 0xFFFFu)));
        const _Float16 zp0 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hA >> 16)));
        const _Float16 sc1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB & 0xFFFFu)));
        const _Float16 zp1 = (_Float16)__half2float(__ushort_as_half((unsigned short)(hB >> 16)));
        const _Float16* xg = x_base + g * 256;

        #pragma unroll
        for (int kt = 0; kt < 16; kt++) {
            const int k_off = kt * 16;
            const _Float16 sc_h = (kt < 8) ? sc0 : sc1;
            const _Float16 zp_h = (kt < 8) ? zp0 : zp1;

            unsigned int pk0 = *(const unsigned int*)(gp + 8 + k_off / 2);
            unsigned int pk1 = *(const unsigned int*)(gp + 8 + k_off / 2 + 4);

            half16_t a_reg;
            #define DQ(i, pk, sh) a_reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(0,  pk0,  0); DQ(1,  pk0,  4); DQ(2,  pk0,  8); DQ(3,  pk0, 12);
            DQ(4,  pk0, 16); DQ(5,  pk0, 20); DQ(6,  pk0, 24); DQ(7,  pk0, 28);
            DQ(8,  pk1,  0); DQ(9,  pk1,  4); DQ(10, pk1,  8); DQ(11, pk1, 12);
            DQ(12, pk1, 16); DQ(13, pk1, 20); DQ(14, pk1, 24); DQ(15, pk1, 28);
            #undef DQ

            half16_t b_reg = *(const half16_t*)(xg + k_off);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
    }

    // RDNA3 interleaved C: acc[j] = C[2*j + (tid>>4)][tid & 15]
    const int out_col = col_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M)
                Y[(long long)out_col * M + out_row] = acc[j];
        }
    }
}

// ── RDNA3 LDS-tile MMQ infrastructure (from gemm_hfq4g256_residual_mmq.hip) ─

template <int nbytes>
static __device__ __forceinline__ void memcpy_aligned(void* __restrict__ dst,
                                                      const void* __restrict__ src) {
    if constexpr (nbytes == 16) {
        ((int4*)dst)[0] = ((const int4*)src)[0];
    } else if constexpr (nbytes == 8) {
        ((int2*)dst)[0] = ((const int2*)src)[0];
    } else if constexpr (nbytes == 4) {
        ((int*)dst)[0] = ((const int*)src)[0];
    }
}

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
    DATA_LAYOUT_I_MAJOR_MIRRORED = 20,
};

template <int I_, int J_, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
struct tile {};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }
    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 16 && J == 16) {
            return 2 * l + (threadIdx.x / 16);
        } else {
            return l;
        }
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32 * 2;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int) {
        return threadIdx.x % 16;
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return l;
    }
};

template <typename T, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_16x8(
    tile<16, 8, T, dl>& t, const T* __restrict__ xs0, const int stride
) {
    static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad layout");
    static_assert(sizeof(t.x) == 32, "bad mirrored tile size");
    memcpy_aligned<16>(t.x + 0, xs0 + t.get_i(0) * stride + 0);
    memcpy_aligned<16>(t.x + 4, xs0 + t.get_i(0) * stride + 4);
}

template <data_layout dl_d, data_layout dl_ab>
static __device__ __forceinline__ void mma_i8(
    tile<16, 16, int, dl_d>& D,
    const tile<16, 8, int, dl_ab>& A,
    const tile<16, 8, int, dl_ab>& B
) {
    using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    int32x8_t* acc = (int32x8_t*)D.x;
    const int32x4_t* a_vec = (const int32x4_t*)A.x;
    const int32x4_t* b_vec = (const int32x4_t*)B.x;
    // Proven RDNA3 path: int32x4 operands, two iu8 WMMA per 16-K (signed nibbles 0..15).
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
}

// V2=true: dual-fp16 half headers → dm slots 0..3 = half0, 4..7 = half1.
// V2=false: legacy f32 sc/zp replicated into all 8 dm slots.
template <bool V2>
static __device__ __forceinline__ void load_weight_tile_mmq(
    const char* __restrict__ A,
    int* __restrict__ tile_x,
    int row0,
    int kb,
    int M,
    int groups_per_row
) {
    int* x_qs = tile_x;
    half2* x_dm = (half2*)(x_qs + 2 * kMmqTileNeK);

    const int txi = threadIdx.x;

    for (int i0 = 0; i0 < kMmqY; i0 += kMmqNwarps) {
        const int i = i0 + threadIdx.y;
        const int row = min(row0 + i, M - 1);
        const char* gp = A + ((long long)row * groups_per_row + kb) * kGroupBytes;
        const unsigned int qs0 = *(const unsigned int*)(gp + 8 + txi * 4);

        const unsigned int q0 = (qs0 >>  0) & 0xFu;
        const unsigned int q1 = (qs0 >>  4) & 0xFu;
        const unsigned int q2 = (qs0 >>  8) & 0xFu;
        const unsigned int q3 = (qs0 >> 12) & 0xFu;
        const unsigned int q4 = (qs0 >> 16) & 0xFu;
        const unsigned int q5 = (qs0 >> 20) & 0xFu;
        const unsigned int q6 = (qs0 >> 24) & 0xFu;
        const unsigned int q7 = (qs0 >> 28) & 0xFu;
        x_qs[i * kMmqTileXK + 2 * txi + 0] = (int)(q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
        x_qs[i * kMmqTileXK + 2 * txi + 1] = (int)(q4 | (q5 << 8) | (q6 << 16) | (q7 << 24));
    }

    constexpr int rows_per_warp_meta = 16;
    for (int i0 = 0; i0 < kMmqY; i0 += kMmqNwarps * rows_per_warp_meta) {
        const int i = i0 + threadIdx.y * rows_per_warp_meta + threadIdx.x / 2;
        if (i < kMmqY) {
            const int row = min(row0 + i, M - 1);
            const char* gp = A + ((long long)row * groups_per_row + kb) * kGroupBytes;
            const int ksc = threadIdx.x & 1;
            half2 dm;
            if constexpr (V2) {
                const unsigned int hA = *(const unsigned int*)(gp);
                const unsigned int hB = *(const unsigned int*)(gp + 4);
                // ksc==0 → half0 (K 0..127 / dm slots 0..3)
                // ksc==1 → half1 (K 128..255 / dm slots 4..7)
                const unsigned int hs = (ksc == 0) ? hA : hB;
                const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
                const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
                dm = make_half2((_Float16)sc, (_Float16)zp);
            } else {
                const float sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                dm = make_half2((_Float16)sc, (_Float16)zp);
            }
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                x_dm[i * kMmqTileXK + 4 * ksc + l] = dm;
        }
    }
}

static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int* __restrict__ x,
    const int* __restrict__ y,
    float* __restrict__ sum,
    int k00
) {
    using tile_A = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_B = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;

    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * kMmqTileYK);

    const int* x_qs = x;
    const half2* x_dm = (const half2*)(x_qs + 2 * kMmqTileNeK);
    const int* y_qs = y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < kMmqTileNeK; k01 += kQi8_1) {
        const int k0 = k00 + k01;
        tile_A A_frag[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix_16x8(A_frag[n],
                               x_qs + (i0 + n * tile_A::I) * kMmqTileXK + k0,
                               kMmqTileXK);
        }

        #pragma unroll
        for (int j0 = 0; j0 < kMmqX; j0 += ntx * tile_C::J) {
            tile_B B_frag;
            load_ldmatrix_16x8(B_frag, y_qs + j0 * kMmqTileYK + k01, kMmqTileYK);
            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * kMmqTileYK + k01 / kQi8_1]);

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C_frag;
                mma_i8(C_frag, A_frag[n], B_frag);

                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    // dmA.x = s_half, dmA.y = z_half; dsB.x = d_x, dsB.y = sum_real_x
                    const float2 dmA = __half22float2(x_dm[i * kMmqTileXK + k0 / kQi8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
                        dmA.x * dsB.x * (float)C_frag.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB.y;
                }
            }
        }
    }
}

template <bool V2>
static __device__ __forceinline__ void bench_mmq_rdna3_body(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    // Host grid is ceil-mapped: every launched block has row0 < M and col0 < N.
    // Do not early-return: all 256 threads must take every __syncthreads.
    const int row0 = blockIdx.x * kMmqY;
    const int col0 = blockIdx.y * kMmqX;

    const int groups_per_row = K / 256;
    const int blocks128 = K / 128;

    extern __shared__ int smem[];
    int* tile_y = smem;
    int* tile_x = tile_y + kMmqX * kMmqTileYK;

    float sum[kMmqX * kMmqY / (kMmqNwarps * 32)] = {0.0f};

    for (int kb = 0; kb < groups_per_row; ++kb) {
        // Weights: OOB rows clamp to M-1 (production load_hfq4_tile FULL=false).
        load_weight_tile_mmq<V2>(A, tile_x, row0, kb, M, groups_per_row);

        // ---- half-K 0 (dm slots 0..3 / V2 half0) ----
        {
            const int kb_q8 = 2 * kb;
            const int* by = (kb_q8 < blocks128)
                ? (const int*)(Xq + (long long)kb_q8 * N + col0)
                : nullptr;
            for (int l0 = 0; l0 < kMmqX * kMmqTileYK; l0 += kMmqNwarps * 32) {
                const int l = l0 + threadIdx.y * 32 + threadIdx.x;
                const int j = l / kMmqTileYK;
                // Partial N: zero-pad columns beyond N; never return early.
                tile_y[l] = (by != nullptr && col0 + j < N) ? by[l] : 0;
            }
        }
        __syncthreads();
        vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, 0);
        __syncthreads();

        // ---- half-K 1 (dm slots 4..7 / V2 half1) ----
        // Barriers stay outside any K-predicate so participation is uniform.
        // When K%256==0 (bench contract) the second half always exists.
        {
            const int kb_q8 = 2 * kb + 1;
            const int* by = (kb_q8 < blocks128)
                ? (const int*)(Xq + (long long)kb_q8 * N + col0)
                : nullptr;
            for (int l0 = 0; l0 < kMmqX * kMmqTileYK; l0 += kMmqNwarps * 32) {
                const int l = l0 + threadIdx.y * 32 + threadIdx.x;
                const int j = l / kMmqTileYK;
                tile_y[l] = (by != nullptr && col0 + j < N) ? by[l] : 0;
            }
        }
        __syncthreads();
        if (2 * kb + 1 < blocks128) {
            vec_dot_q8_1_q8_1_mma(tile_x, tile_y, sum, kMmqTileNeK);
        }
        __syncthreads();
    }

    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int rows_per_warp = 32;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    #pragma unroll
    for (int j0 = 0; j0 < kMmqX; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (row0 + i < M && col0 + j < N) {
                    float* yp = Y + (long long)(col0 + j) * M + (row0 + i);
                    *yp = sum[(j0 / tile_C::J + n) * tile_C::ne + l]; // SET
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(kMmqNwarps * 32, 2)
void bench_mq4v2_mmq_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    bench_mmq_rdna3_body<true>(A, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(kMmqNwarps * 32, 2)
void bench_hfq4_mmq_legacy_set(
    const char* __restrict__ A,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int K, int N
) {
    bench_mmq_rdna3_body<false>(A, Xq, Y, M, K, N);
}

#else

// Stubs when compiling without a supported offload arch in this TU slice.
extern "C" __global__ void bench_mq4v2_f16_wmma_set(
    const char*, const _Float16*, float*, int, int, int) {}
extern "C" __global__ void bench_mq4v2_mmq_set(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}
extern "C" __global__ void bench_hfq4_mmq_legacy_set(
    const char*, const block_q8_1_mmq*, float*, int, int, int) {}

#endif

// ─────────────────────────────────────────────────────────────────────────────
// Host packing / reference
// ─────────────────────────────────────────────────────────────────────────────

static size_t weight_bytes_mnk(int M, int K) {
    return (size_t)M * (size_t)(K / kGroupK) * (size_t)kGroupBytes;
}

static size_t q8_bytes_nk(int N, int K) {
    return (size_t)(K / 128) * (size_t)N * sizeof(block_q8_1_mmq);
}

static void pack_nibbles(uint8_t* payload, const uint8_t q[256]) {
    for (int i = 0; i < 128; i++)
        payload[i] = (uint8_t)((q[2 * i] & 0xFu) | ((q[2 * i + 1] & 0xFu) << 4));
}

static void pack_v2_group(uint8_t* gp,
                          float s0, float z0, float s1, float z1,
                          const uint8_t q[256]) {
    const uint16_t hs0 = f32_to_f16_bits(s0);
    const uint16_t hz0 = f32_to_f16_bits(z0);
    const uint16_t hs1 = f32_to_f16_bits(s1);
    const uint16_t hz1 = f32_to_f16_bits(z1);
    std::memcpy(gp + 0, &hs0, 2);
    std::memcpy(gp + 2, &hz0, 2);
    std::memcpy(gp + 4, &hs1, 2);
    std::memcpy(gp + 6, &hz1, 2);
    pack_nibbles(gp + 8, q);
}

static void pack_legacy_group(uint8_t* gp, float s, float z, const uint8_t q[256]) {
    std::memcpy(gp + 0, &s, 4);
    std::memcpy(gp + 4, &z, 4);
    pack_nibbles(gp + 8, q);
}

static void fill_v2_weights(std::vector<uint8_t>& A, int M, int K, uint64_t seed,
                            bool disjoint_halves) {
    A.assign(weight_bytes_mnk(M, K), 0);
    uint64_t rng = seed;
    const int gpr = K / kGroupK;
    for (int m = 0; m < M; m++) {
        for (int g = 0; g < gpr; g++) {
            uint8_t q[256];
            for (int i = 0; i < 256; i++)
                q[i] = (uint8_t)(lcg_u64(&rng) & 0xFu);
            float s0, z0, s1, z1;
            if (disjoint_halves) {
                s0 = 0.02f + 0.03f * lcg_unit(&rng);
                z0 = -0.40f + 0.10f * lcg_unit(&rng);
                s1 = 0.25f + 0.20f * lcg_unit(&rng);
                z1 =  0.55f + 0.15f * lcg_unit(&rng);
            } else {
                s0 = lcg_finite(&rng, 0.01f, 0.20f);
                z0 = lcg_finite(&rng, -0.5f, 0.5f);
                s1 = lcg_finite(&rng, 0.01f, 0.20f);
                z1 = lcg_finite(&rng, -0.5f, 0.5f);
            }
            uint8_t* gp = A.data() + ((size_t)m * (size_t)gpr + (size_t)g) * kGroupBytes;
            pack_v2_group(gp, s0, z0, s1, z1, q);
        }
    }
}

static void fill_legacy_weights(std::vector<uint8_t>& A, int M, int K, uint64_t seed) {
    A.assign(weight_bytes_mnk(M, K), 0);
    uint64_t rng = seed;
    const int gpr = K / kGroupK;
    for (int m = 0; m < M; m++) {
        for (int g = 0; g < gpr; g++) {
            uint8_t q[256];
            for (int i = 0; i < 256; i++)
                q[i] = (uint8_t)(lcg_u64(&rng) & 0xFu);
            const float s = lcg_finite(&rng, 0.01f, 0.25f);
            const float z = lcg_finite(&rng, -0.5f, 0.5f);
            uint8_t* gp = A.data() + ((size_t)m * (size_t)gpr + (size_t)g) * kGroupBytes;
            pack_legacy_group(gp, s, z, q);
        }
    }
}

// Header-control pair: identical unsigned-4 nibble payload and equivalent affine.
// Legacy stores one f32 (s,z); V2 repeats that same (s,z) across both fp16 halves.
// Isolates header representation from algorithm/payload.
static void fill_header_control_pair(std::vector<uint8_t>& A_leg,
                                     std::vector<uint8_t>& A_v2,
                                     int M, int K, uint64_t seed) {
    A_leg.assign(weight_bytes_mnk(M, K), 0);
    A_v2.assign(weight_bytes_mnk(M, K), 0);
    uint64_t rng = seed;
    const int gpr = K / kGroupK;
    for (int m = 0; m < M; m++) {
        for (int g = 0; g < gpr; g++) {
            uint8_t q[256];
            for (int i = 0; i < 256; i++)
                q[i] = (uint8_t)(lcg_u64(&rng) & 0xFu);
            const float s = lcg_finite(&rng, 0.01f, 0.25f);
            const float z = lcg_finite(&rng, -0.5f, 0.5f);
            const size_t off = ((size_t)m * (size_t)gpr + (size_t)g) * kGroupBytes;
            pack_legacy_group(A_leg.data() + off, s, z, q);
            pack_v2_group(A_v2.data() + off, s, z, s, z, q);
        }
    }
}

static void fill_x_f32(std::vector<float>& X, int N, int K, uint64_t seed) {
    X.resize((size_t)N * (size_t)K);
    uint64_t rng = seed;
    for (size_t i = 0; i < X.size(); i++)
        X[i] = lcg_finite(&rng, -1.5f, 1.5f);
}

// Host Q8_1: per-32 sub-block d and sum_real = d * sum(rounded_q8); qs length N*K.
static void quantize_q8_1_host(const float* X, int N, int K,
                               std::vector<int8_t>& qs,
                               std::vector<float>& d32,
                               std::vector<float>& sum_real32) {
    const int nsub = K / 32;
    qs.assign((size_t)N * (size_t)K, 0);
    d32.assign((size_t)N * (size_t)nsub, 0.f);
    sum_real32.assign((size_t)N * (size_t)nsub, 0.f);
    for (int n = 0; n < N; n++) {
        const float* xrow = X + (size_t)n * (size_t)K;
        int8_t* qrow = qs.data() + (size_t)n * (size_t)K;
        for (int s = 0; s < nsub; s++) {
            const float* xs = xrow + s * 32;
            float amax = 0.f;
            for (int i = 0; i < 32; i++)
                amax = std::max(amax, std::fabs(xs[i]));
            const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
            const float id = d > 0.0f ? 1.0f / d : 0.0f;
            int sum_q = 0;
            for (int i = 0; i < 32; i++) {
                const int qi = (int)std::lrintf(std::max(-127.0f, std::min(127.0f, xs[i] * id)));
                qrow[s * 32 + i] = (int8_t)qi;
                sum_q += qi;
            }
            d32[(size_t)n * (size_t)nsub + (size_t)s] = d;
            sum_real32[(size_t)n * (size_t)nsub + (size_t)s] = d * (float)sum_q;
        }
    }
}

// F16 / float-X reference (weight dequant × original activation).
static void ref_v2_f64(const uint8_t* A, const float* X, std::vector<double>& Y,
                       int M, int K, int N) {
    Y.assign((size_t)N * (size_t)M, 0.0);
    const int gpr = K / kGroupK;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            double acc = 0.0;
            const uint8_t* row = A + (size_t)m * (size_t)gpr * kGroupBytes;
            const float* xrow = X + (size_t)n * (size_t)K;
            for (int g = 0; g < gpr; g++) {
                const uint8_t* gp = row + (size_t)g * kGroupBytes;
                uint16_t hs0, hz0, hs1, hz1;
                std::memcpy(&hs0, gp + 0, 2);
                std::memcpy(&hz0, gp + 2, 2);
                std::memcpy(&hs1, gp + 4, 2);
                std::memcpy(&hz1, gp + 6, 2);
                const double s0 = (double)f16_bits_to_f32(hs0);
                const double z0 = (double)f16_bits_to_f32(hz0);
                const double s1 = (double)f16_bits_to_f32(hs1);
                const double z1 = (double)f16_bits_to_f32(hz1);
                const uint8_t* pk = gp + 8;
                for (int k = 0; k < 256; k++) {
                    const uint8_t byte = pk[k >> 1];
                    const int q = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    const double s = (k < 128) ? s0 : s1;
                    const double z = (k < 128) ? z0 : z1;
                    acc += (s * (double)q + z) * (double)xrow[g * 256 + k];
                }
            }
            Y[(size_t)n * (size_t)M + (size_t)m] = acc;
        }
    }
}

// MMQ reference: weight dequant × Q8_1 dequant activation (d*q), matching GPU MMQ.
static void ref_v2_mmq_f64(const uint8_t* A, const float* X, std::vector<double>& Y,
                           int M, int K, int N) {
    std::vector<int8_t> qs;
    std::vector<float> d32, sum_real32;
    quantize_q8_1_host(X, N, K, qs, d32, sum_real32);
    (void)sum_real32;
    Y.assign((size_t)N * (size_t)M, 0.0);
    const int gpr = K / kGroupK;
    const int nsub = K / 32;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            double acc = 0.0;
            const uint8_t* row = A + (size_t)m * (size_t)gpr * kGroupBytes;
            const int8_t* qrow = qs.data() + (size_t)n * (size_t)K;
            for (int g = 0; g < gpr; g++) {
                const uint8_t* gp = row + (size_t)g * kGroupBytes;
                uint16_t hs0, hz0, hs1, hz1;
                std::memcpy(&hs0, gp + 0, 2);
                std::memcpy(&hz0, gp + 2, 2);
                std::memcpy(&hs1, gp + 4, 2);
                std::memcpy(&hz1, gp + 6, 2);
                const double s0 = (double)f16_bits_to_f32(hs0);
                const double z0 = (double)f16_bits_to_f32(hz0);
                const double s1 = (double)f16_bits_to_f32(hs1);
                const double z1 = (double)f16_bits_to_f32(hz1);
                const uint8_t* pk = gp + 8;
                for (int k = 0; k < 256; k++) {
                    const uint8_t byte = pk[k >> 1];
                    const int qw = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    const double s = (k < 128) ? s0 : s1;
                    const double z = (k < 128) ? z0 : z1;
                    const int kk = g * 256 + k;
                    const int sub = kk / 32;
                    const double d = (double)d32[(size_t)n * (size_t)nsub + (size_t)sub];
                    const double qx = (double)qrow[kk];
                    acc += (s * (double)qw + z) * (d * qx);
                }
            }
            Y[(size_t)n * (size_t)M + (size_t)m] = acc;
        }
    }
}

static void ref_legacy_mmq_f64(const uint8_t* A, const float* X, std::vector<double>& Y,
                               int M, int K, int N) {
    std::vector<int8_t> qs;
    std::vector<float> d32, sum_real32;
    quantize_q8_1_host(X, N, K, qs, d32, sum_real32);
    (void)sum_real32;
    Y.assign((size_t)N * (size_t)M, 0.0);
    const int gpr = K / kGroupK;
    const int nsub = K / 32;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            double acc = 0.0;
            const uint8_t* row = A + (size_t)m * (size_t)gpr * kGroupBytes;
            const int8_t* qrow = qs.data() + (size_t)n * (size_t)K;
            for (int g = 0; g < gpr; g++) {
                const uint8_t* gp = row + (size_t)g * kGroupBytes;
                float sf, zf;
                std::memcpy(&sf, gp + 0, 4);
                std::memcpy(&zf, gp + 4, 4);
                const double s = (double)sf;
                const double z = (double)zf;
                const uint8_t* pk = gp + 8;
                for (int k = 0; k < 256; k++) {
                    const uint8_t byte = pk[k >> 1];
                    const int qw = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    const int kk = g * 256 + k;
                    const int sub = kk / 32;
                    const double d = (double)d32[(size_t)n * (size_t)nsub + (size_t)sub];
                    const double qx = (double)qrow[kk];
                    acc += (s * (double)qw + z) * (d * qx);
                }
            }
            Y[(size_t)n * (size_t)M + (size_t)m] = acc;
        }
    }
}

static float rel_rms(const float* got, const double* ref, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)got[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    if (den < 1e-30) return (num < 1e-30) ? 0.f : 1e30f;
    return (float)std::sqrt(num / den);
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch helpers (family-aware MMQ geometry)
// ─────────────────────────────────────────────────────────────────────────────

static GpuFamily g_family = GpuFamily::Unknown;

static void launch_q8_quant(const float* dX, block_q8_1_mmq* dXq, int K, int N) {
    const int threads = 256;
    const int k_elems_per_block = threads * 4;
    const int gx = (K + k_elems_per_block - 1) / k_elems_per_block;
    quantize_q8_1_mmq_ds4<<<dim3(gx, N, 1), dim3(threads, 1, 1)>>>(dX, dXq, K, N);
    CK(hipGetLastError());
}

static void launch_f16_convert(const float* dX, _Float16* dXf, int N, int K) {
    const int n = N * K;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    convert_f32_to_f16<<<blocks, threads>>>(dX, dXf, n);
    CK(hipGetLastError());
}

static void launch_f16_gemm(const char* dA, const _Float16* dX, float* dY,
                            int M, int K, int N) {
    // Both families use 16×16 output tiles / block dim 32 for F16 WMMA.
    bench_mq4v2_f16_wmma_set<<<dim3((M + 15) / 16, (N + 15) / 16, 1), dim3(32, 1, 1)>>>(
        dA, dX, dY, M, K, N);
    CK(hipGetLastError());
}

static void launch_mmq(const char* dA, const block_q8_1_mmq* dXq, float* dY,
                       int M, int K, int N, bool v2) {
    if (g_family == GpuFamily::Rdna3) {
        // Must match gemm_hfq4g256_residual_mmq host launch: 128x128 tile,
        // block [32,8,1], dynamic LDS for tile_x + tile_y. Launching the
        // RDNA3 body with LDS=0 or block.y!=8 hangs on smem[] traffic.
        dim3 grid((unsigned)((M + kMmqY - 1) / kMmqY),
                  (unsigned)((N + kMmqX - 1) / kMmqX), 1u);
        dim3 block(32u, (unsigned)kMmqNwarps, 1u);
        if (v2) {
            bench_mq4v2_mmq_set<<<grid, block, kMmqSharedBytes>>>(dA, dXq, dY, M, K, N);
        } else {
            bench_hfq4_mmq_legacy_set<<<grid, block, kMmqSharedBytes>>>(dA, dXq, dY, M, K, N);
        }
    } else {
        // gfx12: 16x16 tiles, block [32,1,1], no LDS
        dim3 grid((unsigned)((M + 15) / 16), (unsigned)((N + 15) / 16), 1u);
        dim3 block(32u, 1u, 1u);
        if (v2) {
            bench_mq4v2_mmq_set<<<grid, block>>>(dA, dXq, dY, M, K, N);
        } else {
            bench_hfq4_mmq_legacy_set<<<grid, block>>>(dA, dXq, dY, M, K, N);
        }
    }
    CK(hipGetLastError());
}

enum class ArmId : int {
    F16Compute = 0,
    F16E2E = 1,
    MmqCompute = 2,
    MmqE2E = 3,
    LegacyMmq = 4,
    Count = 5
};

static const char* arm_name(ArmId a) {
    switch (a) {
    case ArmId::F16Compute: return "mq4v2_f16_compute";
    case ArmId::F16E2E:     return "mq4v2_f16_e2e";
    case ArmId::MmqCompute: return "mq4v2_mmq_compute";
    case ArmId::MmqE2E:     return "mq4v2_mmq_e2e";
    case ArmId::LegacyMmq:  return "hfq4_mmq_legacy";
    default: return "?";
    }
}

static size_t act_bytes_for(ArmId a, int N, int K) {
    switch (a) {
    case ArmId::F16Compute:
        return (size_t)N * (size_t)K * sizeof(_Float16);
    case ArmId::F16E2E:
        return (size_t)N * (size_t)K * (sizeof(float) + sizeof(_Float16) + sizeof(_Float16));
    case ArmId::MmqCompute:
        return q8_bytes_nk(N, K);
    case ArmId::MmqE2E:
        return (size_t)N * (size_t)K * sizeof(float) + 2 * q8_bytes_nk(N, K);
    case ArmId::LegacyMmq:
        return q8_bytes_nk(N, K);
    default:
        return 0;
    }
}

struct DeviceBufs {
    char* dA_v2 = nullptr;
    char* dA_leg = nullptr;
    float* dX = nullptr;
    _Float16* dXf = nullptr;
    block_q8_1_mmq* dXq = nullptr;
    float* dY = nullptr;
    float* dY_canary_tail = nullptr;
    size_t y_cap = 0;
    size_t canary_n = 64;
};

static void alloc_bufs(DeviceBufs& b, int M, int K, int Nmax) {
    const size_t wbytes = weight_bytes_mnk(M, K);
    const size_t xbytes = (size_t)Nmax * (size_t)K * sizeof(float);
    const size_t xfbytes = (size_t)Nmax * (size_t)K * sizeof(_Float16);
    const size_t xqbytes = q8_bytes_nk(Nmax, K);
    b.y_cap = (size_t)Nmax * (size_t)M;
    const size_t ybytes = (b.y_cap + b.canary_n) * sizeof(float);

    CK(hipMalloc(&b.dA_v2, wbytes));
    CK(hipMalloc(&b.dA_leg, wbytes));
    CK(hipMalloc(&b.dX, xbytes));
    CK(hipMalloc(&b.dXf, xfbytes));
    CK(hipMalloc(&b.dXq, xqbytes));
    CK(hipMalloc(&b.dY, ybytes));
    b.dY_canary_tail = b.dY + b.y_cap;
}

static void free_bufs(DeviceBufs& b) {
    if (b.dA_v2) CK(hipFree(b.dA_v2));
    if (b.dA_leg) CK(hipFree(b.dA_leg));
    if (b.dX) CK(hipFree(b.dX));
    if (b.dXf) CK(hipFree(b.dXf));
    if (b.dXq) CK(hipFree(b.dXq));
    if (b.dY) CK(hipFree(b.dY));
    b = {};
}

static void plant_canary(DeviceBufs& b) {
    std::vector<float> c(b.canary_n, kCanary);
    CK(hipMemcpy(b.dY_canary_tail, c.data(), b.canary_n * sizeof(float),
                 hipMemcpyHostToDevice));
}

static bool check_canary(DeviceBufs& b) {
    std::vector<float> c(b.canary_n);
    CK(hipMemcpy(c.data(), b.dY_canary_tail, b.canary_n * sizeof(float),
                 hipMemcpyDeviceToHost));
    for (float v : c)
        if (v != kCanary) return false;
    return true;
}

static void clear_y(DeviceBufs& b, int M, int N) {
    CK(hipMemset(b.dY, 0, (size_t)N * (size_t)M * sizeof(float)));
}

static void launch_arm(ArmId a, DeviceBufs& b, int M, int K, int N) {
    switch (a) {
    case ArmId::F16Compute:
        launch_f16_gemm(b.dA_v2, b.dXf, b.dY, M, K, N);
        break;
    case ArmId::F16E2E:
        launch_f16_convert(b.dX, b.dXf, N, K);
        launch_f16_gemm(b.dA_v2, b.dXf, b.dY, M, K, N);
        break;
    case ArmId::MmqCompute:
        launch_mmq(b.dA_v2, b.dXq, b.dY, M, K, N, /*v2=*/true);
        break;
    case ArmId::MmqE2E:
        launch_q8_quant(b.dX, b.dXq, K, N);
        launch_mmq(b.dA_v2, b.dXq, b.dY, M, K, N, /*v2=*/true);
        break;
    case ArmId::LegacyMmq:
        launch_mmq(b.dA_leg, b.dXq, b.dY, M, K, N, /*v2=*/false);
        break;
    default:
        break;
    }
}

static float time_arm_us(ArmId a, DeviceBufs& b, int M, int K, int N) {
    if (a == ArmId::F16Compute) {
        launch_f16_convert(b.dX, b.dXf, N, K);
        CK(hipDeviceSynchronize());
    } else if (a == ArmId::MmqCompute || a == ArmId::LegacyMmq) {
        launch_q8_quant(b.dX, b.dXq, K, N);
        CK(hipDeviceSynchronize());
    }

    hipEvent_t ev0, ev1;
    CK(hipEventCreate(&ev0));
    CK(hipEventCreate(&ev1));

    std::vector<float> samples;
    samples.reserve((size_t)kRepeats * (size_t)kIters);

    for (int rep = 0; rep < kRepeats; rep++) {
        for (int w = 0; w < kWarmup; w++) {
            clear_y(b, M, N);
            launch_arm(a, b, M, K, N);
        }
        CK(hipDeviceSynchronize());

        for (int it = 0; it < kIters; it++) {
            clear_y(b, M, N);
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(ev0, nullptr));
            launch_arm(a, b, M, K, N);
            CK(hipEventRecord(ev1, nullptr));
            CK(hipEventSynchronize(ev1));
            float ms = 0.f;
            CK(hipEventElapsedTime(&ms, ev0, ev1));
            samples.push_back(ms * 1000.f);
        }
    }

    CK(hipEventDestroy(ev0));
    CK(hipEventDestroy(ev1));
    return median_us(samples);
}

// ─────────────────────────────────────────────────────────────────────────────
// Correctness gate — dedicated exact-shape buffers (never perf slack)
// ─────────────────────────────────────────────────────────────────────────────

static int run_correctness() {
    constexpr int M = 32, K = 512, N = 16;
    std::fprintf(stderr,
                 "correctness: M=%d K=%d N=%d  rel-RMS limit=%.3f  family=%s\n"
                 "  buffers: dedicated y_cap=%d (canary immediately after)\n",
                 M, K, N, kRelRmsLimit, family_name(g_family), M * N);

    DeviceBufs b{};
    alloc_bufs(b, M, K, N); // y_cap = 32*16; canary at dY + 512

    std::vector<uint8_t> hA_v2_disjoint, hA_leg_ctrl, hA_v2_ctrl;
    std::vector<float> hX;
    // Half-select fixture: intentionally different (s0,z0) vs (s1,z1).
    fill_v2_weights(hA_v2_disjoint, M, K, /*seed=*/0xC0FFEEu, /*disjoint_halves=*/true);
    // Header-control pair: identical nibbles; legacy f32 (s,z) repeated on both V2 halves.
    fill_header_control_pair(hA_leg_ctrl, hA_v2_ctrl, M, K, /*seed=*/0xBEEF01u);
    fill_x_f32(hX, N, K, /*seed=*/0xA11CEu);

    CK(hipMemcpy(b.dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));

    std::vector<double> ref_v2_f16, ref_v2_mmq, ref_leg_mmq, ref_v2_ctrl_mmq;
    ref_v2_f64(hA_v2_disjoint.data(), hX.data(), ref_v2_f16, M, K, N);
    ref_v2_mmq_f64(hA_v2_disjoint.data(), hX.data(), ref_v2_mmq, M, K, N);
    ref_legacy_mmq_f64(hA_leg_ctrl.data(), hX.data(), ref_leg_mmq, M, K, N);
    ref_v2_mmq_f64(hA_v2_ctrl.data(), hX.data(), ref_v2_ctrl_mmq, M, K, N);

    std::vector<float> hY((size_t)N * (size_t)M);

    auto check_arm = [&](const char* name, ArmId a, const char* dA,
                         const std::vector<double>& ref) -> bool {
        plant_canary(b);
        clear_y(b, M, N);
        if (a == ArmId::F16Compute || a == ArmId::F16E2E) {
            launch_f16_convert(b.dX, b.dXf, N, K);
            CK(hipDeviceSynchronize());
            launch_f16_gemm(dA, b.dXf, b.dY, M, K, N);
        } else if (a == ArmId::MmqCompute || a == ArmId::MmqE2E) {
            launch_q8_quant(b.dX, b.dXq, K, N);
            CK(hipDeviceSynchronize());
            launch_mmq(dA, b.dXq, b.dY, M, K, N, /*v2=*/true);
        } else {
            launch_q8_quant(b.dX, b.dXq, K, N);
            CK(hipDeviceSynchronize());
            launch_mmq(dA, b.dXq, b.dY, M, K, N, /*v2=*/false);
        }
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(hY.data(), b.dY, hY.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (!check_canary(b)) {
            std::fprintf(stderr, "FAIL %s: output canary corrupted\n", name);
            return false;
        }
        for (float v : hY) {
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "FAIL %s: non-finite output\n", name);
                return false;
            }
        }
        const float r = rel_rms(hY.data(), ref.data(), hY.size());
        std::fprintf(stderr, "  %-28s rel-RMS=%.6f  %s\n", name, r,
                     (r <= kRelRmsLimit) ? "OK" : "FAIL");
        return r <= kRelRmsLimit;
    };

    bool ok = true;

    // --- Disjoint-half V2 fixture (half-select) ---
    CK(hipMemcpy(b.dA_v2, hA_v2_disjoint.data(), hA_v2_disjoint.size(),
                 hipMemcpyHostToDevice));
    ok = check_arm("mq4v2_f16_compute/disjoint", ArmId::F16Compute, b.dA_v2,
                   ref_v2_f16) &&
         ok;
    ok = check_arm("mq4v2_mmq_compute/disjoint", ArmId::MmqCompute, b.dA_v2,
                   ref_v2_mmq) &&
         ok;

    // --- Header-control pair (identical payload; legacy f32 vs V2 dual-half) ---
    CK(hipMemcpy(b.dA_leg, hA_leg_ctrl.data(), hA_leg_ctrl.size(),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(b.dA_v2, hA_v2_ctrl.data(), hA_v2_ctrl.size(),
                 hipMemcpyHostToDevice));
    ok = check_arm("hfq4_mmq_legacy/hdr-ctrl", ArmId::LegacyMmq, b.dA_leg,
                   ref_leg_mmq) &&
         ok;
    ok = check_arm("mq4v2_mmq_compute/hdr-ctrl", ArmId::MmqCompute, b.dA_v2,
                   ref_v2_ctrl_mmq) &&
         ok;

    free_bufs(b);

    if (!ok) {
        std::fprintf(stderr, "correctness gate FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "correctness gate passed\n");
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI / main
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<int> parse_nlist(const char* s) {
    std::vector<int> out;
    if (!s || !*s) return out;
    const char* p = s;
    while (*p) {
        char* end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) out.push_back((int)v);
        if (*end == ',') {
            p = end + 1;
            continue;
        }
        break;
    }
    return out;
}

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [M] [K] [Nlist]\n"
        "  M,K       default 5120 5120; K must be multiple of 256\n"
        "  Nlist     comma-separated, default 16,32,64,128,256,512,1024,2048\n"
        "\n"
        "Build (hipx fleet fatbin — never single-arch):\n"
        "  /opt/rocm/core/bin/hipcc -O3 \\\n"
        "      --offload-arch=gfx1030 --offload-arch=gfx1100 \\\n"
        "      --offload-arch=gfx1151 --offload-arch=gfx1201 \\\n"
        "      tools/quant-design/bench_mq4v2_prefill_mmq.hip \\\n"
        "      -o /tmp/bench_mq4v2_prefill_mmq\n"
        "\n"
        "CSV columns (stdout):\n"
        "  arm,M,K,N,med_us,w_bytes,act_bytes,eff_gib_s,ratio_f16_compute\n"
        "Compile-time family macro: BENCH_COMPILE_FAMILY=%s\n",
        argv0, BENCH_COMPILE_FAMILY);
}

int main(int argc, char** argv) {
    int M = 5120;
    int K = 5120;
    std::vector<int> ns = {16, 32, 64, 128, 256, 512, 1024, 2048};

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
    }
    if (argc >= 2 && argv[1][0] != '-') M = std::atoi(argv[1]);
    if (argc >= 3 && argv[2][0] != '-') K = std::atoi(argv[2]);
    if (argc >= 4 && argv[3][0] != '-') {
        auto parsed = parse_nlist(argv[3]);
        if (!parsed.empty()) ns = std::move(parsed);
    }

    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        std::fprintf(stderr, "Need M>0 and K%%256==0 (got M=%d K=%d)\n", M, K);
        return 2;
    }
    for (int n : ns) {
        if (n <= 0) {
            std::fprintf(stderr, "N values must be positive\n");
            return 2;
        }
    }
    const int Nmax = *std::max_element(ns.begin(), ns.end());

    CK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    g_family = detect_family(prop);

    std::fprintf(stderr,
                 "=== MQ4V2 prefill MMQ microbench ===\n"
                 "compile-time family: %s\n"
                 "runtime GPU: %s  gcnArchName=%s  family=%s\n"
                 "shape M=%d K=%d  Nmax=%d  warm=%d iters=%d reps=%d\n"
                 "MMQ launch: %s\n"
                 "arms: f16_compute f16_e2e mmq_compute mmq_e2e hfq4_mmq_legacy\n",
                 BENCH_COMPILE_FAMILY,
                 prop.name, prop.gcnArchName, family_name(g_family),
                 M, K, Nmax, kWarmup, kIters, kRepeats,
                 (g_family == GpuFamily::Rdna3)
                     ? "rdna3 128x128 LDS tile block[32,8] shared=57344"
                     : "gfx12 16x16 tile block[32] LDS=0");

    if (g_family == GpuFamily::Unknown) {
        std::fprintf(stderr, "ERROR: unsupported GPU family (need gfx12 or rdna3)\n");
        return 2;
    }

    // Correctness uses its own exact-shape allocations (freed inside).
    if (run_correctness() != 0)
        return 1;

    DeviceBufs bufs;
    alloc_bufs(bufs, M, K, Nmax);


    std::vector<uint8_t> hA_v2, hA_leg;
    std::vector<float> hX;
    fill_v2_weights(hA_v2, M, K, /*seed=*/0x515151u, /*disjoint_halves=*/false);
    fill_legacy_weights(hA_leg, M, K, /*seed=*/0x616161u);
    fill_x_f32(hX, Nmax, K, /*seed=*/0x717171u);

    CK(hipMemcpy(bufs.dA_v2, hA_v2.data(), hA_v2.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(bufs.dA_leg, hA_leg.data(), hA_leg.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(bufs.dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
    CK(hipDeviceSynchronize());

    const size_t wbytes = weight_bytes_mnk(M, K);

    std::printf("arm,M,K,N,med_us,w_bytes,act_bytes,eff_gib_s,ratio_f16_compute\n");

    std::fprintf(stderr,
                 "\n%-22s %6s %10s %12s %10s %8s\n",
                 "arm", "N", "med_us", "eff_GiB/s", "ratio", "w_MiB");
    std::fprintf(stderr,
                 "%-22s %6s %10s %12s %10s %8s\n",
                 "----------------------", "------", "----------", "------------",
                 "----------", "--------");

    for (int N : ns) {
        float med_f16 = 0.f;
        for (int ai = 0; ai < (int)ArmId::Count; ai++) {
            const ArmId a = (ArmId)ai;
            plant_canary(bufs);
            const float med = time_arm_us(a, bufs, M, K, N);
            if (!check_canary(bufs)) {
                std::fprintf(stderr, "FAIL canary after timing %s N=%d\n",
                             arm_name(a), N);
                free_bufs(bufs);
                return 1;
            }
            if (a == ArmId::F16Compute) med_f16 = med;

            const size_t abytes = act_bytes_for(a, N, K);
            const double total = (double)wbytes + (double)abytes;
            const double gib_s = (med > 0.f)
                ? (total / (med * 1e-6)) / (1024.0 * 1024.0 * 1024.0)
                : 0.0;
            const float ratio = (med_f16 > 0.f) ? (med / med_f16) : 0.f;

            std::printf("%s,%d,%d,%d,%.4f,%zu,%zu,%.4f,%.4f\n",
                        arm_name(a), M, K, N, med, wbytes, abytes, gib_s, ratio);

            std::fprintf(stderr, "%-22s %6d %10.3f %12.3f %10.3f %8.2f\n",
                         arm_name(a), N, med, gib_s, ratio,
                         (double)wbytes / (1024.0 * 1024.0));
        }
        std::fprintf(stderr, "\n");
    }

    free_bufs(bufs);
    std::fprintf(stderr, "done.\n");
    return 0;
}
