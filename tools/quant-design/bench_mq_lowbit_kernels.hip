// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// gfx1201 HIP microbenchmark: matched decode-GEMV cost for low-bit MQ formats.
//
// Arms (bits × format × regime):
//   bits     : 2, 3, 4
//   formats  : V2 (dual-G128 fp16 s/z), Lloyd (2^bits fp16 centroids), MFP-E8 (G32)
//   regimes  : B=1 decode GEMV; B=16 shared-weight format/decode proxy
//
// Labels:
//   MQ4V2 is production. MQ2V2 / MQ3V2 are HYPOTHETICAL (production MQ2/MQ3 are Lloyd).
//   B=16 is a shared-weight format microbench (decode each weight once, reuse across
//   16 activation rows). It is NOT a production WMMA throughput measurement.
//
// Layouts (exact):
//   V2   : per G256 group = bits*32 + 8 bytes
//          dual-G128 headers: [fp16 s0|fp16 z0|fp16 s1|fp16 z1] + packed payload
//          w = q * s + z  (tid<16 → half0, tid>=16 → half1)
//   Lloyd: per G256 group = 2^bits * sizeof(fp16) + bits*32 payload bytes
//          w = cb[q]
//   MFP  : per row = 16 B header (fp16 row_scale @0) + (K/32) × (1 + 4*bits) B
//          G32 block = [E4M3 scale][4 × n-byte E8 codewords]
//          w = row_scale * e4m3(scale) * QUANT_STEP * e8_coord
//          QUANT_STEP = {2:3.8, 3:1.8, 4:0.88}; E8 centers/bias = {2:2, 3:4, 4:7}
//
// Build (parent validates; this file is source-only in this pass):
//   hipcc -O3 --offload-arch=gfx1201 tools/quant-design/bench_mq_lowbit_kernels.hip \
//         -o /tmp/bench_mq_lowbit_kernels
//   /tmp/bench_mq_lowbit_kernels [--help] [M] [K]
//
// Default shape: M=512 K=2048 (K%256==0). 32-thread launch, one wave per row.

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
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kWarmup = 100;
static constexpr int kSamples = 1000;
#define BATCH_B 16
static constexpr int kBatchB = BATCH_B;
static constexpr int kSentinelM = 2;
static constexpr int kSentinelK = 256;

#define CK(x)                                                                 \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                         hipGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

enum class Fmt : int { V2 = 0, Lloyd = 1, MfpE8 = 2 };

static inline const char* fmt_name(Fmt f, int bits) {
    switch (f) {
    case Fmt::V2:
        if (bits == 4) return "MQ4V2";
        if (bits == 3) return "MQ3V2(hyp)";
        return "MQ2V2(hyp)";
    case Fmt::Lloyd:
        if (bits == 4) return "MQ4L";
        if (bits == 3) return "MQ3L";
        return "MQ2L";
    case Fmt::MfpE8:
        if (bits == 4) return "MFP4G32E8";
        if (bits == 3) return "MFP3G32E8";
        return "MFP2G32E8";
    }
    return "?";
}

static inline int v2_group_bytes(int bits) { return bits * 32 + 8; }
static inline int lloyd_cb_bytes(int bits) { return (1 << bits) * 2; }
static inline int lloyd_group_bytes(int bits) {
    return lloyd_cb_bytes(bits) + bits * 32;
}
static inline int mfp_block_bytes(int bits) { return 1 + 4 * bits; }
static inline int mfp_row_bytes(int bits, int K) {
    return 16 + (K / 32) * mfp_block_bytes(bits);
}
static inline size_t weight_bytes(Fmt f, int bits, int M, int K) {
    const int gpr = K / 256;
    switch (f) {
    case Fmt::V2:
        return (size_t)M * (size_t)gpr * (size_t)v2_group_bytes(bits);
    case Fmt::Lloyd:
        return (size_t)M * (size_t)gpr * (size_t)lloyd_group_bytes(bits);
    case Fmt::MfpE8:
        return (size_t)M * (size_t)mfp_row_bytes(bits, K);
    }
    return 0;
}
static inline float quant_step(int bits) {
    if (bits == 2) return 3.8f;
    if (bits == 3) return 1.8f;
    return 0.88f;
}
static inline int e8_center(int bits) {
    if (bits == 2) return 2;
    if (bits == 3) return 4;
    return 7;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers: fp16, E4M3, E8 decode (bit-identical to production paths)
// ─────────────────────────────────────────────────────────────────────────────

static inline uint16_t f32_to_f16_bits(float v) {
    // Round-to-nearest-even f32→f16 (host-side; finite inputs only in this bench).
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
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u); // inf
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

static inline float e4m3_scale_decode(uint8_t b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    return std::ldexp(1.0f + (float)mant * 0.125f, exp - 7);
}

// Decode low 8n bits of idx → 8 E8 coords (pre-QUANT_STEP).
static void e8_decode_index_n(uint32_t idx, int n, float p[8]) {
    const int bias = e8_center(n);
    const uint32_t mask = (1u << n) - 1u;
    const int coset = (int)((idx >> (8 * n - 1)) & 1u);
    uint32_t e[8];
    uint32_t sl = 0u;
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (unsigned)(n * i)) & mask;
        sl += e[i];
    }
    const uint32_t e7_high_bits = (uint32_t)(n - 1);
    const uint32_t e7_high_mask = (1u << e7_high_bits) - 1u;
    const uint32_t e7_high = (idx >> (unsigned)(7 * n)) & e7_high_mask;
    const uint32_t p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
    for (int i = 0; i < 8; i++) {
        const float c = (float)((int)e[i] - bias);
        p[i] = coset ? c + 0.5f : c;
    }
}

// Build a valid random E8 codeword in the low 8n bits (upper bits zero).
static uint32_t random_e8_codeword(uint64_t* rng, int n) {
    // LCG
    auto next = [&]() -> uint32_t {
        *rng = *rng * 6364136223846793005ull + 1ull;
        return (uint32_t)(*rng >> 32);
    };
    const uint32_t cmax = (1u << n) - 1u;
    const int bias = e8_center(n);
    // Random integer coords in box, force even sum, pack.
    uint32_t e[8];
    uint32_t sl = 0u;
    for (int i = 0; i < 8; i++) {
        e[i] = next() & cmax;
        sl += e[i];
    }
    if (sl & 1u) {
        if (e[7] < cmax) e[7]++;
        else e[7]--;
    }
    const uint32_t coset = next() & 1u;
    uint32_t idx = 0u;
    const uint32_t mask = cmax;
    for (int i = 0; i < 7; i++) idx |= (e[i] & mask) << (n * i);
    const uint32_t e7_high_bits = (uint32_t)(n - 1);
    const uint32_t e7_high_mask = (1u << e7_high_bits) - 1u;
    idx |= ((e[7] >> 1) & e7_high_mask) << (7 * n);
    idx |= coset << (8 * n - 1);
    (void)bias;
    return idx;
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Pack random valid weight tensors
// ─────────────────────────────────────────────────────────────────────────────

static void pack_v2(std::vector<uint8_t>& out, int bits, int M, int K, uint64_t seed) {
    const int gpr = K / 256;
    const int gb = v2_group_bytes(bits);
    const int payload = bits * 32;
    out.assign((size_t)M * (size_t)gpr * (size_t)gb, 0);
    uint64_t rng = seed;
    const int nlev = 1 << bits;
    for (int r = 0; r < M; r++) {
        for (int g = 0; g < gpr; g++) {
            uint8_t* gp = out.data() + ((size_t)r * gpr + g) * gb;
            // Dual G128 fp16 s/z — finite, non-denorm scales.
            float s0 = lcg_finite(&rng, 0.01f, 2.0f);
            float z0 = lcg_finite(&rng, -1.0f, 1.0f);
            float s1 = lcg_finite(&rng, 0.01f, 2.0f);
            float z1 = lcg_finite(&rng, -1.0f, 1.0f);
            uint16_t h[4] = {
                f32_to_f16_bits(s0), f32_to_f16_bits(z0),
                f32_to_f16_bits(s1), f32_to_f16_bits(z1),
            };
            std::memcpy(gp, h, 8);
            // Packed indices: 32 threads × 8 weights.
            uint8_t* data = gp + 8;
            if (bits == 4) {
                for (int t = 0; t < 32; t++) {
                    for (int b = 0; b < 4; b++) {
                        uint8_t lo = (uint8_t)(lcg_u64(&rng) % (unsigned)nlev);
                        uint8_t hi = (uint8_t)(lcg_u64(&rng) % (unsigned)nlev);
                        data[t * 4 + b] = (uint8_t)(lo | (hi << 4));
                    }
                }
            } else if (bits == 3) {
                for (int t = 0; t < 32; t++) {
                    uint32_t pk = 0;
                    for (int i = 0; i < 8; i++) {
                        uint32_t q = (uint32_t)(lcg_u64(&rng) % (unsigned)nlev);
                        pk |= q << (3 * i);
                    }
                    data[t * 3 + 0] = (uint8_t)(pk & 0xFFu);
                    data[t * 3 + 1] = (uint8_t)((pk >> 8) & 0xFFu);
                    data[t * 3 + 2] = (uint8_t)((pk >> 16) & 0xFFu);
                }
            } else { // bits == 2
                for (int t = 0; t < 32; t++) {
                    for (int b = 0; b < 2; b++) {
                        uint8_t v = 0;
                        for (int i = 0; i < 4; i++) {
                            uint8_t q = (uint8_t)(lcg_u64(&rng) % (unsigned)nlev);
                            v |= (uint8_t)(q << (2 * i));
                        }
                        data[t * 2 + b] = v;
                    }
                }
            }
            (void)payload;
        }
    }
}

static void pack_lloyd(std::vector<uint8_t>& out, int bits, int M, int K, uint64_t seed) {
    const int gpr = K / 256;
    const int gb = lloyd_group_bytes(bits);
    const int cb_b = lloyd_cb_bytes(bits);
    const int nlev = 1 << bits;
    out.assign((size_t)M * (size_t)gpr * (size_t)gb, 0);
    uint64_t rng = seed;
    for (int r = 0; r < M; r++) {
        for (int g = 0; g < gpr; g++) {
            uint8_t* gp = out.data() + ((size_t)r * gpr + g) * gb;
            // Sorted-ish finite centroids.
            for (int i = 0; i < nlev; i++) {
                float c = lcg_finite(&rng, -3.0f, 3.0f);
                uint16_t h = f32_to_f16_bits(c);
                std::memcpy(gp + i * 2, &h, 2);
            }
            uint8_t* data = gp + cb_b;
            if (bits == 4) {
                for (int t = 0; t < 32; t++) {
                    for (int b = 0; b < 4; b++) {
                        uint8_t lo = (uint8_t)(lcg_u64(&rng) % (unsigned)nlev);
                        uint8_t hi = (uint8_t)(lcg_u64(&rng) % (unsigned)nlev);
                        data[t * 4 + b] = (uint8_t)(lo | (hi << 4));
                    }
                }
            } else if (bits == 3) {
                for (int t = 0; t < 32; t++) {
                    uint32_t pk = 0;
                    for (int i = 0; i < 8; i++)
                        pk |= ((uint32_t)(lcg_u64(&rng) % (unsigned)nlev)) << (3 * i);
                    data[t * 3 + 0] = (uint8_t)(pk & 0xFFu);
                    data[t * 3 + 1] = (uint8_t)((pk >> 8) & 0xFFu);
                    data[t * 3 + 2] = (uint8_t)((pk >> 16) & 0xFFu);
                }
            } else {
                for (int t = 0; t < 32; t++) {
                    for (int b = 0; b < 2; b++) {
                        uint8_t v = 0;
                        for (int i = 0; i < 4; i++)
                            v |= (uint8_t)((lcg_u64(&rng) % (unsigned)nlev) << (2 * i));
                        data[t * 2 + b] = v;
                    }
                }
            }
        }
    }
}

static void pack_mfp(std::vector<uint8_t>& out, int bits, int M, int K, uint64_t seed) {
    const int rb = mfp_row_bytes(bits, K);
    const int bb = mfp_block_bytes(bits);
    const int n_blocks = K / 32;
    out.assign((size_t)M * (size_t)rb, 0);
    uint64_t rng = seed;
    for (int r = 0; r < M; r++) {
        uint8_t* row = out.data() + (size_t)r * rb;
        float rs = lcg_finite(&rng, 0.01f, 2.0f);
        uint16_t rs_h = f32_to_f16_bits(rs);
        std::memcpy(row + 0, &rs_h, 2);
        uint16_t zero = 0;
        std::memcpy(row + 2, &zero, 2);
        uint16_t nb = (uint16_t)n_blocks;
        std::memcpy(row + 4, &nb, 2);
        row[6] = 0x05;
        row[7] = 0;
        for (int b = 0; b < n_blocks; b++) {
            uint8_t* blk = row + 16 + b * bb;
            // Valid finite E4M3 (avoid NaN code 0x7F).
            blk[0] = (uint8_t)(lcg_u64(&rng) % 0x7Fu);
            for (int g = 0; g < 4; g++) {
                uint32_t cw = random_e8_codeword(&rng, bits);
                // Write only low n bytes LE.
                for (int k = 0; k < bits; k++)
                    blk[1 + g * bits + k] = (uint8_t)((cw >> (8 * k)) & 0xFFu);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host reference decode → GEMV (sentinel correctness)
// ─────────────────────────────────────────────────────────────────────────────

static void host_gemv_v2(const uint8_t* A, const float* x, float* y,
                         int bits, int M, int K, int B) {
    const int gpr = K / 256;
    const int gb = v2_group_bytes(bits);
    for (int r = 0; r < M; r++) {
        for (int bat = 0; bat < B; bat++) {
            double acc = 0.0;
            const uint8_t* row = A + (size_t)r * gpr * gb;
            const float* xb = x + (size_t)bat * K;
            for (int g = 0; g < gpr; g++) {
                const uint8_t* gp = row + g * gb;
                float s0 = f16_bits_to_f32(*(const uint16_t*)(gp + 0));
                float z0 = f16_bits_to_f32(*(const uint16_t*)(gp + 2));
                float s1 = f16_bits_to_f32(*(const uint16_t*)(gp + 4));
                float z1 = f16_bits_to_f32(*(const uint16_t*)(gp + 6));
                const uint8_t* data = gp + 8;
                for (int t = 0; t < 32; t++) {
                    const float s = (t < 16) ? s0 : s1;
                    const float z = (t < 16) ? z0 : z1;
                    const int base = g * 256 + t * 8;
                    uint32_t q[8];
                    if (bits == 4) {
                        for (int i = 0; i < 4; i++) {
                            uint8_t by = data[t * 4 + i];
                            q[2 * i] = by & 0xF;
                            q[2 * i + 1] = (by >> 4) & 0xF;
                        }
                    } else if (bits == 3) {
                        uint32_t pk = (uint32_t)data[t * 3]
                                    | ((uint32_t)data[t * 3 + 1] << 8)
                                    | ((uint32_t)data[t * 3 + 2] << 16);
                        for (int i = 0; i < 8; i++) q[i] = (pk >> (3 * i)) & 7u;
                    } else {
                        for (int i = 0; i < 2; i++) {
                            uint8_t by = data[t * 2 + i];
                            q[4 * i + 0] = by & 3u;
                            q[4 * i + 1] = (by >> 2) & 3u;
                            q[4 * i + 2] = (by >> 4) & 3u;
                            q[4 * i + 3] = (by >> 6) & 3u;
                        }
                    }
                    for (int i = 0; i < 8; i++)
                        acc += (double)(s * (float)q[i] + z) * (double)xb[base + i];
                }
            }
            y[(size_t)bat * M + r] = (float)acc;
        }
    }
}

static void host_gemv_lloyd(const uint8_t* A, const float* x, float* y,
                            int bits, int M, int K, int B) {
    const int gpr = K / 256;
    const int gb = lloyd_group_bytes(bits);
    const int cb_b = lloyd_cb_bytes(bits);
    const int nlev = 1 << bits;
    for (int r = 0; r < M; r++) {
        for (int bat = 0; bat < B; bat++) {
            double acc = 0.0;
            const uint8_t* row = A + (size_t)r * gpr * gb;
            const float* xb = x + (size_t)bat * K;
            for (int g = 0; g < gpr; g++) {
                const uint8_t* gp = row + g * gb;
                float cb[16];
                for (int i = 0; i < nlev; i++)
                    cb[i] = f16_bits_to_f32(*(const uint16_t*)(gp + i * 2));
                const uint8_t* data = gp + cb_b;
                for (int t = 0; t < 32; t++) {
                    const int base = g * 256 + t * 8;
                    uint32_t q[8];
                    if (bits == 4) {
                        for (int i = 0; i < 4; i++) {
                            uint8_t by = data[t * 4 + i];
                            q[2 * i] = by & 0xF;
                            q[2 * i + 1] = (by >> 4) & 0xF;
                        }
                    } else if (bits == 3) {
                        uint32_t pk = (uint32_t)data[t * 3]
                                    | ((uint32_t)data[t * 3 + 1] << 8)
                                    | ((uint32_t)data[t * 3 + 2] << 16);
                        for (int i = 0; i < 8; i++) q[i] = (pk >> (3 * i)) & 7u;
                    } else {
                        for (int i = 0; i < 2; i++) {
                            uint8_t by = data[t * 2 + i];
                            q[4 * i + 0] = by & 3u;
                            q[4 * i + 1] = (by >> 2) & 3u;
                            q[4 * i + 2] = (by >> 4) & 3u;
                            q[4 * i + 3] = (by >> 6) & 3u;
                        }
                    }
                    for (int i = 0; i < 8; i++)
                        acc += (double)cb[q[i]] * (double)xb[base + i];
                }
            }
            y[(size_t)bat * M + r] = (float)acc;
        }
    }
}

static void host_gemv_mfp(const uint8_t* A, const float* x, float* y,
                          int bits, int M, int K, int B) {
    const int rb = mfp_row_bytes(bits, K);
    const int bb = mfp_block_bytes(bits);
    const int n_blocks = K / 32;
    const float qs = quant_step(bits);
    for (int r = 0; r < M; r++) {
        const uint8_t* row = A + (size_t)r * rb;
        const float row_scale = f16_bits_to_f32(*(const uint16_t*)row);
        for (int bat = 0; bat < B; bat++) {
            double acc = 0.0;
            const float* xb = x + (size_t)bat * K;
            for (int b = 0; b < n_blocks; b++) {
                const uint8_t* blk = row + 16 + b * bb;
                const float ssc = row_scale * e4m3_scale_decode(blk[0]) * qs;
                for (int g = 0; g < 4; g++) {
                    uint32_t cw = 0;
                    for (int k = 0; k < bits; k++)
                        cw |= ((uint32_t)blk[1 + g * bits + k]) << (8 * k);
                    float p[8];
                    e8_decode_index_n(cw, bits, p);
                    const int base = b * 32 + g * 8;
                    for (int i = 0; i < 8; i++)
                        acc += (double)(ssc * p[i]) * (double)xb[base + i];
                }
            }
            y[(size_t)bat * M + r] = (float)acc;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device kernels — template on BITS; format specialized
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float dev_e4m3(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * (float)mant * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + (float)mant * 0.125f);
}

template <int BITS>
__device__ __forceinline__ void dev_e8_decode(unsigned int idx, float p[8]) {
    constexpr int bias = (BITS == 2) ? 2 : (BITS == 3) ? 4 : 7;
    constexpr unsigned mask = (1u << BITS) - 1u;
    const int coset = (int)((idx >> (8 * BITS - 1)) & 1u);
    unsigned e[8];
    unsigned sl = 0u;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        e[i] = (idx >> (unsigned)(BITS * i)) & mask;
        sl += e[i];
    }
    constexpr unsigned e7_high_mask = (1u << (BITS - 1)) - 1u;
    const unsigned e7_high = (idx >> (unsigned)(7 * BITS)) & e7_high_mask;
    const unsigned p7 = e7_high << 1;
    e[7] = p7 | ((sl + p7) & 1u);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        float c = (float)((int)e[i] - bias);
        p[i] = coset ? c + 0.5f : c;
    }
}

template <int BITS>
__device__ __forceinline__ void unpack8(const unsigned char* data, int tid,
                                         unsigned q[8]) {
    if constexpr (BITS == 4) {
        const int boff = tid * 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned by = data[boff + i];
            q[2 * i] = by & 0xFu;
            q[2 * i + 1] = (by >> 4) & 0xFu;
        }
    } else if constexpr (BITS == 3) {
        const int boff = tid * 3;
        unsigned pk = (unsigned)data[boff]
                    | ((unsigned)data[boff + 1] << 8)
                    | ((unsigned)data[boff + 2] << 16);
#pragma unroll
        for (int i = 0; i < 8; i++) q[i] = (pk >> (3 * i)) & 7u;
    } else {
        const int boff = tid * 2;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned by = data[boff + i];
            q[4 * i + 0] = by & 3u;
            q[4 * i + 1] = (by >> 2) & 3u;
            q[4 * i + 2] = (by >> 4) & 3u;
            q[4 * i + 3] = (by >> 6) & 3u;
        }
    }
}

// ─── V2 B=1 ──────────────────────────────────────────────────────────────────
template <int BITS>
__launch_bounds__(32, 16)
__global__ void gemv_v2_b1(const char* __restrict__ A,
                           const float* __restrict__ x,
                           float* __restrict__ y,
                           int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    const int gpr = K / 256;
    constexpr int gb = BITS * 32 + 8;
    const char* row_ptr = A + (long long)row * gpr * gb;
    float acc = 0.0f;
    for (int g = 0; g < gpr; g++) {
        const char* gp = row_ptr + g * gb;
        const unsigned hA = *(const unsigned*)(gp);
        const unsigned hB = *(const unsigned*)(gp + 4);
        const unsigned hs = (tid < 16) ? hA : hB;
        const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned q[8];
        unpack8<BITS>((const unsigned char*)(gp + 8), tid, q);
        const int base = g * 256 + tid * 8;
#pragma unroll
        for (int i = 0; i < 8; i++)
            acc += (sc * (float)q[i] + zp) * x[base + i];
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (tid == 0) y[row] = acc;
}

// ─── V2 B=16 shared-weight ───────────────────────────────────────────────────
template <int BITS>
__launch_bounds__(32, 8)
__global__ void gemv_v2_b16(const char* __restrict__ A,
                            const float* __restrict__ x, // [16 x K]
                            float* __restrict__ y,       // [16 x M]
                            int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    const int gpr = K / 256;
    constexpr int gb = BITS * 32 + 8;
    const char* row_ptr = A + (long long)row * gpr * gb;
    float acc[BATCH_B];
#pragma unroll
    for (int b = 0; b < BATCH_B; b++) acc[b] = 0.0f;
    for (int g = 0; g < gpr; g++) {
        const char* gp = row_ptr + g * gb;
        const unsigned hA = *(const unsigned*)(gp);
        const unsigned hB = *(const unsigned*)(gp + 4);
        const unsigned hs = (tid < 16) ? hA : hB;
        const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned q[8];
        unpack8<BITS>((const unsigned char*)(gp + 8), tid, q);
        float w[8];
#pragma unroll
        for (int i = 0; i < 8; i++) w[i] = sc * (float)q[i] + zp;
        const int base = g * 256 + tid * 8;
#pragma unroll
        for (int b = 0; b < BATCH_B; b++) {
            const float* xb = x + (size_t)b * K;
#pragma unroll
            for (int i = 0; i < 8; i++)
                acc[b] += w[i] * xb[base + i];
        }
    }
#pragma unroll
    for (int b = 0; b < BATCH_B; b++) {
        float a = acc[b];
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down(a, off);
        if (tid == 0) y[(size_t)b * M + row] = a;
    }
}

// ─── Lloyd B=1 ───────────────────────────────────────────────────────────────
template <int BITS>
__launch_bounds__(32, 16)
__global__ void gemv_lloyd_b1(const char* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    const int gpr = K / 256;
    constexpr int nlev = 1 << BITS;
    constexpr int cb_b = nlev * 2;
    constexpr int gb = cb_b + BITS * 32;
    const char* row_ptr = A + (long long)row * gpr * gb;
    __shared__ float cbt[16];
    float acc = 0.0f;
    for (int g = 0; g < gpr; g++) {
        const char* gp = row_ptr + g * gb;
        if (tid < nlev)
            cbt[tid] = __half2float(((const __half*)gp)[tid]);
        __syncthreads();
        unsigned q[8];
        unpack8<BITS>((const unsigned char*)(gp + cb_b), tid, q);
        const int base = g * 256 + tid * 8;
#pragma unroll
        for (int i = 0; i < 8; i++)
            acc += cbt[q[i]] * x[base + i];
        __syncthreads();
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (tid == 0) y[row] = acc;
}

// ─── Lloyd B=16 shared-weight ────────────────────────────────────────────────
template <int BITS>
__launch_bounds__(32, 8)
__global__ void gemv_lloyd_b16(const char* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    const int gpr = K / 256;
    constexpr int nlev = 1 << BITS;
    constexpr int cb_b = nlev * 2;
    constexpr int gb = cb_b + BITS * 32;
    const char* row_ptr = A + (long long)row * gpr * gb;
    __shared__ float cbt[16];
    float acc[BATCH_B];
#pragma unroll
    for (int b = 0; b < BATCH_B; b++) acc[b] = 0.0f;
    for (int g = 0; g < gpr; g++) {
        const char* gp = row_ptr + g * gb;
        if (tid < nlev)
            cbt[tid] = __half2float(((const __half*)gp)[tid]);
        __syncthreads();
        unsigned q[8];
        unpack8<BITS>((const unsigned char*)(gp + cb_b), tid, q);
        float w[8];
#pragma unroll
        for (int i = 0; i < 8; i++) w[i] = cbt[q[i]];
        const int base = g * 256 + tid * 8;
#pragma unroll
        for (int b = 0; b < BATCH_B; b++) {
            const float* xb = x + (size_t)b * K;
#pragma unroll
            for (int i = 0; i < 8; i++)
                acc[b] += w[i] * xb[base + i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < BATCH_B; b++) {
        float a = acc[b];
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down(a, off);
        if (tid == 0) y[(size_t)b * M + row] = a;
    }
}

// ─── MFP-E8 B=1 ──────────────────────────────────────────────────────────────
template <int BITS>
__launch_bounds__(32, 16)
__global__ void gemv_mfp_b1(const char* __restrict__ A,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    constexpr int bb = 1 + 4 * BITS;
    constexpr float QS = (BITS == 2) ? 3.8f : (BITS == 3) ? 1.8f : 0.88f;
    const int n_blocks = K / 32;
    const long long row_off = (long long)row * (16 + (long long)n_blocks * bb);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    // 8 G32 blocks per G256 group; tid maps to one codeword of 8 weights.
    const int block_in = tid >> 2;          // 0..7 within a 256-group
    const int cw_slot = tid & 3;            // 0..3 codeword in block
    float acc = 0.0f;
    for (int b = block_in; b < n_blocks; b += 8) {
        const char* gp = row_ptr + (long long)b * bb;
        const float ssc = row_scale * dev_e4m3(*(const unsigned char*)gp) * QS;
        unsigned int ew = 0u;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_slot * BITS);
#pragma unroll
        for (int k = 0; k < BITS; k++)
            ew |= ((unsigned)w[k]) << (8 * k);
        float pe[8];
        dev_e8_decode<BITS>(ew, pe);
        const int base = b * 32 + cw_slot * 8;
#pragma unroll
        for (int i = 0; i < 8; i++)
            acc += ssc * pe[i] * x[base + i];
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (tid == 0) y[row] = acc;
}

// ─── MFP-E8 B=16 shared-weight ───────────────────────────────────────────────
template <int BITS>
__launch_bounds__(32, 8)
__global__ void gemv_mfp_b16(const char* __restrict__ A,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    constexpr int bb = 1 + 4 * BITS;
    constexpr float QS = (BITS == 2) ? 3.8f : (BITS == 3) ? 1.8f : 0.88f;
    const int n_blocks = K / 32;
    const long long row_off = (long long)row * (16 + (long long)n_blocks * bb);
    const char* hdr = A + row_off;
    const char* row_ptr = hdr + 16;
    const float row_scale =
        (float)__builtin_bit_cast(_Float16, *(const unsigned short*)hdr);
    const int block_in = tid >> 2;
    const int cw_slot = tid & 3;
    float acc[BATCH_B];
#pragma unroll
    for (int b = 0; b < BATCH_B; b++) acc[b] = 0.0f;
    for (int blk = block_in; blk < n_blocks; blk += 8) {
        const char* gp = row_ptr + (long long)blk * bb;
        const float ssc = row_scale * dev_e4m3(*(const unsigned char*)gp) * QS;
        unsigned int ew = 0u;
        const unsigned char* w = (const unsigned char*)(gp + 1 + cw_slot * BITS);
#pragma unroll
        for (int k = 0; k < BITS; k++)
            ew |= ((unsigned)w[k]) << (8 * k);
        float pe[8];
        dev_e8_decode<BITS>(ew, pe);
        float wt[8];
#pragma unroll
        for (int i = 0; i < 8; i++) wt[i] = ssc * pe[i];
        const int base = blk * 32 + cw_slot * 8;
#pragma unroll
        for (int b = 0; b < BATCH_B; b++) {
            const float* xb = x + (size_t)b * K;
#pragma unroll
            for (int i = 0; i < 8; i++)
                acc[b] += wt[i] * xb[base + i];
        }
    }
#pragma unroll
    for (int b = 0; b < BATCH_B; b++) {
        float a = acc[b];
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down(a, off);
        if (tid == 0) y[(size_t)b * M + row] = a;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch + timing
// ─────────────────────────────────────────────────────────────────────────────

using KernelB1 = void (*)(const char*, const float*, float*, int, int);
using KernelB16 = void (*)(const char*, const float*, float*, int, int);

static KernelB1 pick_b1(Fmt f, int bits) {
    if (f == Fmt::V2) {
        if (bits == 2) return gemv_v2_b1<2>;
        if (bits == 3) return gemv_v2_b1<3>;
        return gemv_v2_b1<4>;
    }
    if (f == Fmt::Lloyd) {
        if (bits == 2) return gemv_lloyd_b1<2>;
        if (bits == 3) return gemv_lloyd_b1<3>;
        return gemv_lloyd_b1<4>;
    }
    if (bits == 2) return gemv_mfp_b1<2>;
    if (bits == 3) return gemv_mfp_b1<3>;
    return gemv_mfp_b1<4>;
}
static KernelB16 pick_b16(Fmt f, int bits) {
    if (f == Fmt::V2) {
        if (bits == 2) return gemv_v2_b16<2>;
        if (bits == 3) return gemv_v2_b16<3>;
        return gemv_v2_b16<4>;
    }
    if (f == Fmt::Lloyd) {
        if (bits == 2) return gemv_lloyd_b16<2>;
        if (bits == 3) return gemv_lloyd_b16<3>;
        return gemv_lloyd_b16<4>;
    }
    if (bits == 2) return gemv_mfp_b16<2>;
    if (bits == 3) return gemv_mfp_b16<3>;
    return gemv_mfp_b16<4>;
}

struct Arm {
    Fmt fmt;
    int bits;
    int B; // 1 or 16
    const char* regime;
    size_t wbytes;
    char* dA = nullptr;
    float* dX = nullptr;
    float* dY = nullptr;
    KernelB1 k1 = nullptr;
    KernelB16 k16 = nullptr;
    std::vector<float> samples_us;
    float med = 0, p10 = 0, p90 = 0, gib_s = 0, ratio_v2 = 0;
};

static float percentile_sorted(std::vector<float>& v, double p) {
    if (v.empty()) return 0.f;
    const double idx = p * (double)(v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min(lo + 1, v.size() - 1);
    const float t = (float)(idx - (double)lo);
    return v[lo] * (1.f - t) + v[hi] * t;
}

static bool all_finite(const float* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(p[i])) return false;
    return true;
}

static uint64_t checksum_finite(const float* p, size_t n) {
    // Stable bit-mix of finite floats; fails loudly if any non-finite.
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(p[i])) return 0;
        uint32_t bits;
        std::memcpy(&bits, &p[i], 4);
        h ^= bits + (uint64_t)i * 0x9E3779B97F4A7C15ull;
        h *= 0x100000001B3ull;
    }
    return h ? h : 1ull;
}

static void print_help(const char* argv0) {
    std::printf(
        "Usage: %s [--help] [M] [K]\n"
        "\n"
        "Matched gfx1201 HIP microbenchmark for MQ low-bit decode kernels.\n"
        "Reports 3 bits × 3 formats × 2 regimes (B=1 decode GEMV, B=16 shared-weight).\n"
        "\n"
        "Formats:\n"
        "  V2     dual-G128 fp16 s/z + packed payload (bits*32+8 B/G256)\n"
        "         MQ4V2 = production; MQ2V2/MQ3V2 = HYPOTHETICAL\n"
        "  Lloyd  2^bits fp16 centroids + packed payload (production MQ2/3/4L)\n"
        "  MFP-E8 row hdr16 + G32 blocks of 1+4*bits B, E4M3 scale, E8 lattice\n"
        "         centers {2,4,7}, QUANT_STEP {3.8,1.8,0.88} for bits {2,3,4}\n"
        "\n"
        "Regimes:\n"
        "  B=1    decode GEMV (one activation row)\n"
        "  B=16   SHARED-WEIGHT FORMAT/DECODE MICROBENCH — each wave decodes each\n"
        "         weight once and reuses it across 16 activation rows.\n"
        "         This is NOT a production WMMA throughput measurement.\n"
        "\n"
        "Timing: HIP events, %d warmup + %d interleaved samples; median/p10/p90 us,\n"
        "effective weight GiB/s, ratio to V2 at the same bits+regime.\n"
        "Correctness: host decode sentinel + finite/checksum guards.\n"
        "Default M=512 K=2048 (K must be a multiple of 256). Launch = 32 threads/row.\n",
        argv0, kWarmup, kSamples);
}

int main(int argc, char** argv) {
    int M = 512;
    int K = 2048;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_help(argv[0]);
            return 0;
        }
    }
    if (argc >= 2 && argv[1][0] != '-') M = std::atoi(argv[1]);
    if (argc >= 3 && argv[2][0] != '-') K = std::atoi(argv[2]);
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        std::fprintf(stderr, "Need M>0 and K%%256==0 (got M=%d K=%d)\n", M, K);
        return 2;
    }

    CK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("=== MQ low-bit decode microbench ===\n");
    std::printf("GPU: %s  arch-target: gfx1201  M=%d K=%d\n", prop.name, M, K);
    std::printf("Warmup=%d  Samples=%d (HIP events, interleaved)\n", kWarmup, kSamples);
    std::printf("B=16 = shared-weight format/decode microbench (NOT production WMMA)\n");
    std::printf("MQ2V2/MQ3V2 = HYPOTHETICAL; MQ4V2 and MQ2/3/4 Lloyd + MFP*E8 = production layouts\n\n");

    // ── Build arms ──────────────────────────────────────────────────────────
    std::vector<Arm> arms;
    arms.reserve(18);
    const Fmt fmts[3] = {Fmt::V2, Fmt::Lloyd, Fmt::MfpE8};
    const int bits_list[3] = {2, 3, 4};
    const int B_list[2] = {1, 16};

    // Host activations (shared across formats; B=16 uses first 16 rows of a buffer).
    std::vector<float> hX((size_t)kBatchB * (size_t)K);
    {
        uint64_t rng = 0xA11CE5EEDull;
        for (size_t i = 0; i < hX.size(); i++)
            hX[i] = lcg_finite(&rng, -1.0f, 1.0f);
    }
    float *dX1 = nullptr, *dX16 = nullptr;
    CK(hipMalloc(&dX1, (size_t)K * sizeof(float)));
    CK(hipMalloc(&dX16, (size_t)kBatchB * (size_t)K * sizeof(float)));
    CK(hipMemcpy(dX1, hX.data(), (size_t)K * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dX16, hX.data(), (size_t)kBatchB * (size_t)K * sizeof(float),
                 hipMemcpyHostToDevice));

    for (int bi = 0; bi < 3; bi++) {
        const int bits = bits_list[bi];
        std::vector<uint8_t> pv, pl, pm;
        pack_v2(pv, bits, M, K, 0xBEEF0000ull + (uint64_t)bits * 17);
        pack_lloyd(pl, bits, M, K, 0x10D00000ull + (uint64_t)bits * 19);
        pack_mfp(pm, bits, M, K, 0xE8000000ull + (uint64_t)bits * 23);
        const std::vector<uint8_t>* packs[3] = {&pv, &pl, &pm};
        for (int fi = 0; fi < 3; fi++) {
            const Fmt fmt = fmts[fi];
            char* dA = nullptr;
            const size_t nbytes = packs[fi]->size();
            CK(hipMalloc(&dA, nbytes));
            CK(hipMemcpy(dA, packs[fi]->data(), nbytes, hipMemcpyHostToDevice));
            for (int B : B_list) {
                Arm a;
                a.fmt = fmt;
                a.bits = bits;
                a.B = B;
                a.regime = (B == 1) ? "B1-decode" : "B16-shared-wt";
                a.wbytes = weight_bytes(fmt, bits, M, K);
                a.dA = dA;
                a.dX = (B == 1) ? dX1 : dX16;
                const size_t ybytes = (size_t)B * (size_t)M * sizeof(float);
                CK(hipMalloc(&a.dY, ybytes));
                CK(hipMemset(a.dY, 0, ybytes));
                a.k1 = pick_b1(fmt, bits);
                a.k16 = pick_b16(fmt, bits);
                a.samples_us.reserve(kSamples);
                arms.push_back(a);
            }
        }
    }


    // ── Correctness: sentinel host vs device ────────────────────────────────
    bool pass = true;
    std::printf("--- correctness (sentinel M=%d K=%d) ---\n", kSentinelM, kSentinelK);
    for (int bits : bits_list) {
        for (int fi = 0; fi < 3; fi++) {
            const Fmt fmt = fmts[fi];
            std::vector<uint8_t> pk;
            if (fmt == Fmt::V2) pack_v2(pk, bits, kSentinelM, kSentinelK, 0x51ull + bits);
            else if (fmt == Fmt::Lloyd)
                pack_lloyd(pk, bits, kSentinelM, kSentinelK, 0x52ull + bits);
            else
                pack_mfp(pk, bits, kSentinelM, kSentinelK, 0x53ull + bits);

            std::vector<float> hXsent((size_t)kBatchB * kSentinelK);
            {
                uint64_t rng = 0xC0FFEEull + (unsigned)bits;
                for (auto& v : hXsent) v = lcg_finite(&rng, -0.5f, 0.5f);
            }
            for (int B : B_list) {
                std::vector<float> href((size_t)B * kSentinelM, 0.f);
                if (fmt == Fmt::V2)
                    host_gemv_v2(pk.data(), hXsent.data(), href.data(), bits,
                                 kSentinelM, kSentinelK, B);
                else if (fmt == Fmt::Lloyd)
                    host_gemv_lloyd(pk.data(), hXsent.data(), href.data(), bits,
                                    kSentinelM, kSentinelK, B);
                else
                    host_gemv_mfp(pk.data(), hXsent.data(), href.data(), bits,
                                  kSentinelM, kSentinelK, B);

                if (!all_finite(href.data(), href.size())) {
                    std::printf("FAIL %s B=%d host non-finite\n", fmt_name(fmt, bits), B);
                    pass = false;
                    continue;
                }

                char* dA = nullptr;
                float *dXs = nullptr, *dYs = nullptr;
                CK(hipMalloc(&dA, pk.size()));
                CK(hipMemcpy(dA, pk.data(), pk.size(), hipMemcpyHostToDevice));
                CK(hipMalloc(&dXs, (size_t)B * kSentinelK * sizeof(float)));
                CK(hipMemcpy(dXs, hXsent.data(),
                             (size_t)B * kSentinelK * sizeof(float),
                             hipMemcpyHostToDevice));
                CK(hipMalloc(&dYs, (size_t)B * kSentinelM * sizeof(float)));
                CK(hipMemset(dYs, 0, (size_t)B * kSentinelM * sizeof(float)));

                if (B == 1)
                    pick_b1(fmt, bits)<<<kSentinelM, 32>>>(dA, dXs, dYs,
                                                           kSentinelM, kSentinelK);
                else
                    pick_b16(fmt, bits)<<<kSentinelM, 32>>>(dA, dXs, dYs,
                                                            kSentinelM, kSentinelK);
                CK(hipGetLastError());
                CK(hipDeviceSynchronize());

                std::vector<float> hdev(href.size());
                CK(hipMemcpy(hdev.data(), dYs, hdev.size() * sizeof(float),
                             hipMemcpyDeviceToHost));

                if (!all_finite(hdev.data(), hdev.size())) {
                    std::printf("FAIL %s B=%d device non-finite\n",
                                fmt_name(fmt, bits), B);
                    pass = false;
                } else {
                    double max_abs = 0.0, max_rel = 0.0;
                    for (size_t i = 0; i < href.size(); i++) {
                        const double d = std::fabs((double)hdev[i] - (double)href[i]);
                        max_abs = std::max(max_abs, d);
                        const double denom =
                            std::max(1e-6, std::fabs((double)href[i]));
                        max_rel = std::max(max_rel, d / denom);
                    }
                    // FP reduction order can differ slightly; tight but tolerant.
                    const bool ok = max_abs < 1e-3 || max_rel < 1e-4;
                    if (!ok) {
                        std::printf(
                            "FAIL %s B=%d mismatch max_abs=%.3e max_rel=%.3e\n",
                            fmt_name(fmt, bits), B, max_abs, max_rel);
                        pass = false;
                    } else {
                        const uint64_t cs = checksum_finite(hdev.data(), hdev.size());
                        if (cs == 0) {
                            std::printf("FAIL %s B=%d checksum\n",
                                        fmt_name(fmt, bits), B);
                            pass = false;
                        } else {
                            std::printf(
                                "PASS %s B=%d  max_abs=%.3e  csum=0x%016llx\n",
                                fmt_name(fmt, bits), B, max_abs,
                                (unsigned long long)cs);
                        }
                    }
                }
                hipFree(dA);
                hipFree(dXs);
                hipFree(dYs);
            }
        }
    }
    if (!pass) {
        std::printf("\nCORRECTNESS: FAIL\n");
        return 1;
    }
    std::printf("CORRECTNESS: PASS\n\n");

    // ── Warmup (all arms) ───────────────────────────────────────────────────
    for (int w = 0; w < kWarmup; w++) {
        for (Arm& a : arms) {
            if (a.B == 1)
                a.k1<<<M, 32>>>(a.dA, a.dX, a.dY, M, K);
            else
                a.k16<<<M, 32>>>(a.dA, a.dX, a.dY, M, K);
        }
    }
    CK(hipDeviceSynchronize());

    // Guard: main-arm outputs finite + nonzero checksum (kills DCE).
    for (Arm& a : arms) {
        std::vector<float> hy((size_t)a.B * (size_t)M);
        CK(hipMemcpy(hy.data(), a.dY, hy.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
        if (!all_finite(hy.data(), hy.size()) ||
            checksum_finite(hy.data(), hy.size()) == 0) {
            std::printf("FAIL post-warmup finite/checksum %s %s\n",
                        fmt_name(a.fmt, a.bits), a.regime);
            return 1;
        }
    }

    // ── Interleaved timing ──────────────────────────────────────────────────
    hipEvent_t ev0, ev1;
    CK(hipEventCreate(&ev0));
    CK(hipEventCreate(&ev1));
    for (int s = 0; s < kSamples; s++) {
        for (Arm& a : arms) {
            CK(hipEventRecord(ev0, nullptr));
            if (a.B == 1)
                a.k1<<<M, 32>>>(a.dA, a.dX, a.dY, M, K);
            else
                a.k16<<<M, 32>>>(a.dA, a.dX, a.dY, M, K);
            CK(hipEventRecord(ev1, nullptr));
            CK(hipEventSynchronize(ev1));
            float ms = 0.f;
            CK(hipEventElapsedTime(&ms, ev0, ev1));
            a.samples_us.push_back(ms * 1000.f);
        }
    }
    CK(hipEventDestroy(ev0));
    CK(hipEventDestroy(ev1));

    // Final checksum pass (outputs must remain finite after timed runs).
    uint64_t sink = 0;
    for (Arm& a : arms) {
        std::vector<float> hy((size_t)a.B * (size_t)M);
        CK(hipMemcpy(hy.data(), a.dY, hy.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
        const uint64_t cs = checksum_finite(hy.data(), hy.size());
        if (cs == 0) {
            std::printf("FAIL post-bench checksum %s %s\n",
                        fmt_name(a.fmt, a.bits), a.regime);
            return 1;
        }
        sink ^= cs;
    }

    // ── Stats + ratios ──────────────────────────────────────────────────────
    for (Arm& a : arms) {
        std::sort(a.samples_us.begin(), a.samples_us.end());
        a.med = percentile_sorted(a.samples_us, 0.50);
        a.p10 = percentile_sorted(a.samples_us, 0.10);
        a.p90 = percentile_sorted(a.samples_us, 0.90);
        const double sec = (double)a.med * 1e-6;
        a.gib_s = (sec > 0.0)
                      ? (float)((double)a.wbytes / sec / (1024.0 * 1024.0 * 1024.0))
                      : 0.f;
    }
    for (Arm& a : arms) {
        float v2med = 0.f;
        for (const Arm& b : arms) {
            if (b.fmt == Fmt::V2 && b.bits == a.bits && b.B == a.B) {
                v2med = b.med;
                break;
            }
        }
        a.ratio_v2 = (v2med > 0.f) ? (a.med / v2med) : 0.f;
    }

    std::printf("--- results (median/p10/p90 us, weight GiB/s, t/t_V2) ---\n");
    std::printf("%-12s %-14s %8s %8s %8s %10s %8s %10s\n",
                "format", "regime", "med_us", "p10_us", "p90_us", "GiB/s",
                "t/V2", "wbytes");
    for (const Arm& a : arms) {
        std::printf("%-12s %-14s %8.2f %8.2f %8.2f %10.2f %8.3f %10zu\n",
                    fmt_name(a.fmt, a.bits), a.regime, a.med, a.p10, a.p90,
                    a.gib_s, a.ratio_v2, a.wbytes);
    }
    std::printf("\nchecksum_sink=0x%016llx  (anti-DCE)\n",
                (unsigned long long)sink);
    std::printf("arms=%zu  (expect 18 = 3 bits × 3 formats × 2 regimes)\n",
                arms.size());
    std::printf("CORRECTNESS: PASS\n");

    // Cleanup: unique dA pointers (each fmt×bits shared by 2 regimes).
    std::vector<char*> freed;
    for (Arm& a : arms) {
        if (std::find(freed.begin(), freed.end(), a.dA) == freed.end()) {
            hipFree(a.dA);
            freed.push_back(a.dA);
        }
        hipFree(a.dY);
        a.dA = nullptr;
        a.dY = nullptr;
    }
    hipFree(dX1);
    hipFree(dX16);
    return 0;
}
