// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// search_activation_joint.hip — standalone gfx1201 HIP tool that jointly
// searches global D1/D2 tables against activation-weighted reconstruction for
// production MQ4V2 and hypothetical matched-header 2-bit/3-bit V2 research
// layouts (two G128 fp16 scale+zero pairs).
//
// Runtime contract (matches the engine; table sizes/loads/FWHT stages/launches
// unchanged — only the 256 D1 and/or 256 D2 signs may change):
//   Forward: R = D2 · H · D1 with unnormalized 8-stage FWHT-256, then ×1/16.
//   one global 256-sign D1 (canonical seed 42)
//   one global 256-sign D2 (canonical seed 1042)
//
//   Inverse of quantization error (exact f32, same candidate pair, same ops):
//     e_original = D1 · H · D2 · e_rot / 16
//   Activation metric: m1 = Σ importance[i] · e_original[i]²
//   Importance is the imatrix input-channel diagonal slice for that weight
//   block; RAW and IMPORTANCE share identical [nblocks,256] block ordering.
//
// Production header encoding (exact, crates/hipfire-quantize/src/float16.rs):
//   f32→f16 is Hipfire TRUNCATION, NOT IEEE RNE / __float2half_rn.
//   st,z = f16_to_f32(f32_to_f16_trunc(.));
//   q = clamp(floor((v-z)/st+0.5), 0, qmax);   // floor+0.5, not rint
//
// Scores (F=1 paired RAW/IMPORTANCE fixture; equal weight over bits n∈{2,3,4}):
//   Pair p = (D1,D2). U(p) = D1→FWHT→×1/16 (pre-D2). |D2·U|=|U|.
//   energy_b = Σ_i U[b,i]²   (Parseval: equal to raw block energy)
//   eps      = 2^{-24} · median_b(energy_b)
//   tail mask: fixed-count top ceil(1%) of |U| for that pair's U;
//              ties by (block,index) ascending. D2-only search keeps U/mask fixed.
//
//   act_n(p)     = Σ importance · e_original_n(p)²          // PRIMARY mass
//   sse_n(p)     = Σ e_rot_n(p)²                            // raw rotated MSE
//   r_n(p)       = mean_b[ SSE_rot_n(p,b) / (energy_b+eps) ] // block NMSE
//   tsse_n(p)    = Σ_{i in tail} e_rot_n(p,i)²              // tail-1% MSE guard
//
//   J_act(p)   = (1/3) Σ_n act_n(p)  / act_n(canonical)     // PRIMARY rank
//   J_tail(p)  = (1/3) Σ_n tsse_n(p) / tsse_n(canonical)
//   J_block(p) = (1/3) Σ_n r_n(p)    / r_n(canonical)
//   rel_raw_n  = sse_n(p) / sse_n(canonical)
//   canonical  = (D1 seed 42, D2 seed 1042)
//
// Feasible iff:
//   J_block <= 1.0005  AND  max_n rel_raw_n <= 1.0025
//
// Selection (lexicographic, lower better):
//   1. feasible first
//   2. minimize J_act
//   3. minimize J_tail
//   4. minimize J_block
//   5. lex-smaller packed 256 sign bits of the table being searched (8×u32 LE)
//
// D1 modes evaluate every candidate D1 through the exact host f32 FWHT with the
// supplied D2 (U/tail/d1 rebound per candidate). D2 modes precompute U once for
// the supplied D1, then only apply candidate D2 on device.
//
// No heldout / split awareness: CLI takes explicit RAW and IMPORTANCE paths.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O3 search_activation_joint.hip \
//         -o search_activation_joint
//
// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
//   search_activation_joint --help
//   search_activation_joint d1-random RAW IMP D2_SIGNS N RNG_SEED OUT_PREFIX
//   search_activation_joint d1-coord  RAW IMP D1_SIGNS D2_SIGNS PASSES OUT_PREFIX
//   search_activation_joint d2-random RAW IMP D1_SIGNS N RNG_SEED OUT_PREFIX
//   search_activation_joint d2-coord  RAW IMP D1_SIGNS D2_SIGNS PASSES OUT_PREFIX
//   search_activation_joint eval      RAW IMP D1_SIGNS D2_SIGNS OUT_PREFIX
//
// RAW / IMPORTANCE: paired headerless LE f32 [nblocks,256]; identical sizes.
// *_SIGNS: 256 × f32 LE exact ±1.0
//
// ranked.csv (d1-random / d2-random):
//   rank,seed,label,table,feasible,J_act,J_tail,J_block,worst_raw,
//   act2,act3,act4,sse2,sse3,sse4,tsse2,tsse3,tsse4,r2,r3,r4,
//   rel_act2,rel_act3,rel_act4,rel_tail2,rel_tail3,rel_tail4,
//   rel_raw2,rel_raw3,rel_raw4,rel_block2,rel_block3,rel_block4,
//   mse2,mse3,mse4,act_mse2,act_mse3,act_mse4,tail_mse2,tail_mse3,tail_mse4,
//   signs_u32_0..signs_u32_7
// coord.csv (d1-coord / d2-coord):
//   pass,flip_idx,accepted,feasible,J_act,J_tail,J_block,delta_J_act,worst_raw,
//   act*,sse*,tsse*,r*,rel_act*,rel_tail*,rel_raw*,rel_block*
//   (flip_idx=-1 end-of-pass)
// eval.csv:
//   who,bits,mse,act_mse,r_nmse,rel_act,rel_raw,rel_block,tail_mse,rel_tail,
//   abs_p99,abs_p99_9,abs_max,act_blk_p99,act_blk_p99_9,act_blk_max
// *.d1.signs / *.d2.signs: 256×f32 LE ±1.0
// *.d1.u32.txt / *.d2.u32.txt / *.best.*.u32.txt / *.final.*.u32.txt:
//   eight hex u32 (bit i set iff signs[i]==+1)
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
static constexpr uint32_t D1_CANON_SEED = 42u;
static constexpr uint32_t D2_CANON_SEED = 1042u;
static constexpr float FWHT_SCALE = 0.0625f; // 1/16
static constexpr double EPS_SCALE = 0x1.0p-24;
static constexpr double FEAS_BLOCK = 1.0005;
static constexpr double FEAS_RAW = 1.0025;
static constexpr double PARITY_TOL = 1e-4;

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
R"(search_activation_joint — joint D1/D2 search on activation-weighted error

USAGE
  %s --help
  %s d1-random RAW IMPORTANCE D2_SIGNS N RNG_SEED OUT_PREFIX
  %s d1-coord  RAW IMPORTANCE D1_SIGNS D2_SIGNS PASSES OUT_PREFIX
  %s d2-random RAW IMPORTANCE D1_SIGNS N RNG_SEED OUT_PREFIX
  %s d2-coord  RAW IMPORTANCE D1_SIGNS D2_SIGNS PASSES OUT_PREFIX
  %s eval      RAW IMPORTANCE D1_SIGNS D2_SIGNS OUT_PREFIX

INPUTS
  RAW, IMPORTANCE
    Paired headerless little-endian f32 [nblocks,256] tensors.
    Same byte size, same block ordering. RAW = unrotated weight groups;
    IMPORTANCE = imatrix input-channel diagonal for that weight block
    (non-negative; non-finite rejected). Size must be a positive multiple
    of 1024 bytes. No heldout/split logic — paths are explicit.

  SIGNS (D1_SIGNS / D2_SIGNS)
    Exactly 256 little-endian f32 values, each exact ±1.0.
    Malformed size or non-±1 entries are rejected.
    Candidate signs must not alter FWHT operation count vs canonical.

MODES
  d1-random  Canonical D1 (seed 42) + N LCG-seeded random D1 tables, with the
             supplied D2 held fixed. Each D1 is applied through the exact host
             f32 FWHT before D2/affine/activation scoring.
             Writes ranked.csv, best.d1.signs, best.d1.u32.txt, fixed.d2.signs,
             fixed.d2.u32.txt under OUT_PREFIX.

  d1-coord   Coordinate descent on D1 from D1_SIGNS for PASSES sweeps of 256
             single-bit flips; D2_SIGNS fixed. Strict total-order accepts only.
             Writes coord.csv, final.d1.signs, final.d1.u32.txt,
             fixed.d2.signs, fixed.d2.u32.txt.

  d2-random  Canonical D2 (seed 1042) + N LCG-seeded random D2 tables, with the
             supplied D1 held fixed. Precomputes U = D1→FWHT→×1/16 once.
             Writes ranked.csv, best.d2.signs, best.d2.u32.txt, fixed.d1.signs,
             fixed.d1.u32.txt under OUT_PREFIX.

  d2-coord   Coordinate descent on D2 from D2_SIGNS for PASSES sweeps; D1 fixed
             and pre-rotated once. Writes coord.csv, final.d2.signs,
             final.d2.u32.txt, fixed.d1.signs, fixed.d1.u32.txt.

  eval       Score (D1_SIGNS, D2_SIGNS) against canonical (42, 1042).
             Writes eval.csv, cand.d1.signs, cand.d2.signs, cand.d1.u32.txt,
             cand.d2.u32.txt, canon.d1.u32.txt, canon.d2.u32.txt.
             Includes per-bit ratios and p99/p99.9 activation-weighted
             per-block error mass.

TRANSFORMS AND CODEC
  R = D2 · H · D1 with unnormalized 8-stage FWHT-256 then ×1/16.
  D1 modes: exact host f32 path per candidate D1 (U/tail/d1 rebound).
  D2 modes: precompute U once for supplied D1; search applies D2 only.
  Per G256: two independent contiguous G128 asymmetric affine grids for
  bits∈{2,3,4}:
    lo,hi = min/max of half; step=(hi-lo)/(2^b-1);
    st,z = f16_to_f32(f32_to_f16_TRUNC(.))  // float16.rs truncation, NOT RNE
    q = clamp(floor((v-z)/st + 0.5), 0, 2^b-1);  recon = q*st + z
  Degenerate half (hi==lo or st==0): q=0, recon=z.
  e_rot = recon - v.
  Inverse (exact f32, same candidate pair): e_orig = D1·H·D2·e_rot/16.
  Activation mass: Σ importance · e_orig².

SCORES
  Primary: J_act = mean_n act_n(p)/act_n(canonical).
  Guards (also reported): raw rotated MSE, block NMSE, fixed pre-D2 tail-1%%
  MSE (top ceil(1%%) of |U|; ties by (block,index) ascending).
  Feasible iff J_block <= 1.0005 and every bit's raw-MSE ratio <= 1.0025.
  Among feasible: min J_act, then J_tail, then J_block, then lex-smaller
  packed sign bits of the searched table.
  Canonical reference pair is always D1 seed 42 + D2 seed 1042.

PARITY
  On each run, host-side inverse-transform parity checks confirm
  inv(fwd(x)) ≈ x for canonical and candidate pairs (shared op count).

OUTPUT
  ranked.csv / coord.csv / eval.csv schemas documented in source header.
  Sign tables: 256×f32 LE ±1.0; u32.txt: 8 hex u32 words (bit i set iff +1).

BUILD
  hipcc --offload-arch=gfx1201 -O3 search_activation_joint.hip \
        -o search_activation_joint
)",
                 argv0, argv0, argv0, argv0, argv0, argv0);
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
        const uint32_t f16_frac = (frac == 0u) ? 0u : ((frac >> 13) | 1u);
        return (uint16_t)((sign << 15) | (0x1Fu << 10) | f16_frac);
    }
    const int32_t new_exp = exp - 127 + 15;
    if (new_exp >= 31) {
        return (uint16_t)((sign << 15) | (0x1Fu << 10));
    }
    if (new_exp <= 0) {
        if (new_exp < -10) {
            return (uint16_t)(sign << 15);
        }
        const uint32_t f = frac | 0x800000u;
        const uint32_t shift = (uint32_t)(1 - new_exp + 13);
        return (uint16_t)((sign << 15) | (f >> shift));
    }
    return (uint16_t)((sign << 15) | ((uint32_t)new_exp << 10) | (frac >> 13));
}

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

// Full forward: v = D2 · (H/16) · D1 · w  (in-place).
static void forward_rotate_block(float* blk, const float* d1, const float* d2) {
    pre_rotate_block(blk, d1);
    for (int i = 0; i < G; ++i) blk[i] *= d2[i];
}

// Inverse error map: e_orig = D1 · (H/16) · D2 · e_rot  (in-place).
static void inverse_error_block(float* e, const float* d1, const float* d2) {
    for (int i = 0; i < G; ++i) e[i] *= d2[i];
    fwht_unnormalized_256(e);
    for (int i = 0; i < G; ++i) e[i] = d1[i] * (e[i] * FWHT_SCALE);
}

static void parity_check_pair(const float* d1_a, const float* d2_a,
                              const float* d1_b, const float* d2_b,
                              const char* tag_b) {
    float x[G], y[G], z[G];
    uint32_t st = 0xC0FFEEu;
    for (int i = 0; i < G; ++i) {
        uint32_t r = lcg_next(st);
        x[i] = ((float)(int)(r % 2001) - 1000.0f) / 1000.0f;
    }
    auto check_one = [&](const float* d1, const float* d2, const char* who) {
        std::memcpy(y, x, sizeof(y));
        forward_rotate_block(y, d1, d2);
        std::memcpy(z, y, sizeof(z));
        inverse_error_block(z, d1, d2);
        double max_abs = 0.0, num = 0.0, den = 0.0;
        for (int i = 0; i < G; ++i) {
            double e = (double)z[i] - (double)x[i];
            double ae = std::fabs(e);
            if (ae > max_abs) max_abs = ae;
            num += e * e;
            den += (double)x[i] * (double)x[i];
        }
        double rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
        if (!(max_abs < PARITY_TOL) || !(rel < PARITY_TOL)) {
            std::fprintf(stderr,
                         "error: inverse-transform parity failed for %s "
                         "(max_abs=%.6e rel_l2=%.6e tol=%.6e)\n",
                         who, max_abs, rel, PARITY_TOL);
            std::exit(1);
        }
    };
    check_one(d1_a, d2_a, "canonical");
    check_one(d1_b, d2_b, tag_b);

    float zero[G];
    std::memset(zero, 0, sizeof(zero));
    inverse_error_block(zero, d1_a, d2_a);
    for (int i = 0; i < G; ++i) {
        if (zero[i] != 0.0f) die("canonical inverse of zero error is nonzero");
    }
    std::memset(zero, 0, sizeof(zero));
    inverse_error_block(zero, d1_b, d2_b);
    for (int i = 0; i < G; ++i) {
        if (zero[i] != 0.0f) die("candidate inverse of zero error is nonzero");
    }
}

static std::vector<float> load_f32_matrix(const char* path, int& nblocks_out,
                                          const char* label) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s %s\n", label, path);
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_END) != 0) die("fseek failed");
    long bytes = std::ftell(f);
    if (bytes < 0) die("ftell failed");
    if (std::fseek(f, 0, SEEK_SET) != 0) die("fseek rewind failed");
    if (bytes == 0) {
        std::fprintf(stderr, "error: %s is empty\n", label);
        std::exit(1);
    }
    if (bytes % (long)(G * (int)sizeof(float)) != 0) {
        std::fprintf(stderr,
                     "error: %s size %ld is not a multiple of %d bytes\n", label,
                     bytes, G * (int)sizeof(float));
        std::exit(1);
    }
    const int nblocks = (int)(bytes / (long)(G * (int)sizeof(float)));
    if (nblocks <= 0) {
        std::fprintf(stderr, "error: %s has zero blocks\n", label);
        std::exit(1);
    }
    std::vector<float> data((size_t)nblocks * G);
    if (std::fread(data.data(), 1, (size_t)bytes, f) != (size_t)bytes) {
        std::fclose(f);
        std::fprintf(stderr, "error: short read on %s\n", label);
        std::exit(1);
    }
    std::fclose(f);
    for (size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            std::fprintf(stderr, "error: non-finite %s value at element %zu\n",
                         label, i);
            std::exit(1);
        }
    }
    nblocks_out = nblocks;
    return data;
}

static void load_paired(const char* raw_path, const char* imp_path,
                        std::vector<float>& raw, std::vector<float>& imp,
                        int& nblocks_out) {
    int nb_raw = 0, nb_imp = 0;
    raw = load_f32_matrix(raw_path, nb_raw, "RAW");
    imp = load_f32_matrix(imp_path, nb_imp, "IMPORTANCE");
    if (nb_raw != nb_imp || raw.size() != imp.size()) {
        std::fprintf(stderr,
                     "error: RAW/IMPORTANCE size mismatch: blocks %d vs %d\n",
                     nb_raw, nb_imp);
        std::exit(1);
    }
    for (size_t i = 0; i < imp.size(); ++i) {
        if (imp[i] < 0.0f) {
            std::fprintf(stderr,
                         "error: negative IMPORTANCE at element %zu (%g)\n", i,
                         (double)imp[i]);
            std::exit(1);
        }
    }
    nblocks_out = nb_raw;
}

static std::vector<float> load_signs(const char* path, const char* label) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s %s\n", label, path);
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fprintf(stderr, "error: fseek %s failed\n", label);
        std::exit(1);
    }
    long bytes = std::ftell(f);
    if (bytes != (long)(G * (int)sizeof(float))) {
        std::fprintf(stderr, "error: %s size %ld, expected %d\n", label, bytes,
                     G * (int)sizeof(float));
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_SET) != 0) {
        std::fprintf(stderr, "error: fseek %s rewind failed\n", label);
        std::exit(1);
    }
    std::vector<float> s(G);
    if (std::fread(s.data(), sizeof(float), G, f) != (size_t)G) {
        std::fclose(f);
        std::fprintf(stderr, "error: short read on %s\n", label);
        std::exit(1);
    }
    std::fclose(f);
    for (int i = 0; i < G; ++i) {
        if (!(s[i] == 1.0f || s[i] == -1.0f)) {
            std::fprintf(stderr, "error: %s[%d]=%g not ±1.0f\n", label, i,
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
// One block = one (candidate, corpus-block-tile). 256 threads cover one G256.
// For each corpus block: D2·U → dual-G128 affine → e_rot → inverse via
// D2→FWHT→×1/16→D1 → accumulate activation / rotated / tail / block metrics.

__global__ __launch_bounds__(256) void eval_activation_batch(
    const float* __restrict__ U, const float* __restrict__ Imp,
    const float* __restrict__ energy, const uint8_t* __restrict__ tail_mask,
    const float* __restrict__ d1, int nblocks, const float* __restrict__ d2_batch,
    int ncand, double eps, double* __restrict__ out_sse,
    double* __restrict__ out_act, double* __restrict__ out_tsse,
    double* __restrict__ out_rsum) {
    const int tid = threadIdx.x;
    const int cand = (int)blockIdx.y;
    if (cand >= ncand) return;

    __shared__ float sh_min[G];
    __shared__ float sh_max[G];
    __shared__ float sh_e[G];
    __shared__ double sh_red[NBITS][G];

    const float* d2 = d2_batch + (size_t)cand * G;
    const float s = d2[tid];
    const float d1v = d1[tid];

    double loc_sse[NBITS] = {};
    double loc_act[NBITS] = {};
    double loc_tsse[NBITS] = {};
    double loc_rsum[NBITS] = {};

    for (int blk = (int)blockIdx.x; blk < nblocks; blk += (int)gridDim.x) {
        const size_t gi = (size_t)blk * G + (size_t)tid;
        const float u0 = U[gi];
        const float imp = Imp[gi];
        const int in_tail = (int)tail_mask[gi];
        const float v = u0 * s;
        sh_min[tid] = v;
        sh_max[tid] = v;
        __syncthreads();

        // Per-half min/max (two contiguous G128 groups).
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
        const double e_blk = (double)energy[blk] + eps;

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
                float q = floorf((v - z) / st + 0.5f);
                q = fminf(fmaxf(q, 0.0f), levels);
                recon = q * st + z;
            }
            const float e_rot = recon - v;
            const double e2 = (double)e_rot * (double)e_rot;
            loc_sse[bi] += e2;
            if (in_tail) loc_tsse[bi] += e2;

            // Inverse: D2 * e_rot → FWHT → ×1/16 → D1
            sh_e[tid] = s * e_rot;
            __syncthreads();
            for (int stride = 1; stride < G; stride <<= 1) {
                const int pair = tid ^ stride;
                const float a = sh_e[tid];
                const float b = sh_e[pair];
                float out;
                if ((tid & stride) == 0) {
                    out = a + b;
                } else {
                    out = b - a;
                }
                __syncthreads();
                sh_e[tid] = out;
                __syncthreads();
            }
            const float e_orig = d1v * (sh_e[tid] * FWHT_SCALE);
            loc_act[bi] += (double)imp * (double)e_orig * (double)e_orig;

            sh_red[bi][tid] = e2;
            __syncthreads();
            for (int stride = 128; stride > 0; stride >>= 1) {
                if (tid < stride) sh_red[bi][tid] += sh_red[bi][tid + stride];
                __syncthreads();
            }
            if (tid == 0) loc_rsum[bi] += sh_red[bi][0] / e_blk;
            __syncthreads();
        }
    }

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
    if (tid == 0) {
#pragma unroll
        for (int bi = 0; bi < NBITS; ++bi)
            atomicAdd(out_sse + cand * NBITS + bi, sh_red[bi][0]);
    }
    __syncthreads();

#pragma unroll
    for (int bi = 0; bi < NBITS; ++bi) sh_red[bi][tid] = loc_act[bi];
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
            atomicAdd(out_act + cand * NBITS + bi, sh_red[bi][0]);
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
        for (int bi = 0; bi < NBITS; ++bi)
            atomicAdd(out_tsse + cand * NBITS + bi, sh_red[bi][0]);
    }
    __syncthreads();

    sh_red[0][tid] = loc_rsum[0];
    sh_red[1][tid] = loc_rsum[1];
    sh_red[2][tid] = loc_rsum[2];
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
            atomicAdd(out_rsum + cand * NBITS + bi, sh_red[bi][0]);
    }
}

// ── metrics / GPU ctx ───────────────────────────────────────────────────────

struct CandMetrics {
    double sse[NBITS]{};  // raw rotated SSE
    double act[NBITS]{};  // activation-weighted original-space SSE
    double tsse[NBITS]{}; // tail rotated SSE
    double r[NBITS]{};    // mean block NMSE (rotated)
    double rel_act[NBITS]{};
    double rel_tail[NBITS]{};
    double rel_raw[NBITS]{};
    double rel_block[NBITS]{};
    double J_act = 0.0;
    double J_tail = 0.0;
    double J_block = 0.0;
    double worst_raw = 0.0;
    int feasible = 0;

    void finalize(const CandMetrics& canon) {
        J_act = 0.0;
        J_tail = 0.0;
        J_block = 0.0;
        worst_raw = 0.0;
        int ok_raw = 1;
        for (int bi = 0; bi < NBITS; ++bi) {
            const double aa = canon.act[bi] > 0.0 ? canon.act[bi] : 1e-300;
            const double ta = canon.tsse[bi] > 0.0 ? canon.tsse[bi] : 1e-300;
            const double sa = canon.sse[bi] > 0.0 ? canon.sse[bi] : 1e-300;
            const double ra = canon.r[bi] > 0.0 ? canon.r[bi] : 1e-300;
            rel_act[bi] = act[bi] / aa;
            rel_tail[bi] = tsse[bi] / ta;
            rel_raw[bi] = sse[bi] / sa;
            rel_block[bi] = r[bi] / ra;
            J_act += rel_act[bi];
            J_tail += rel_tail[bi];
            J_block += rel_block[bi];
            if (rel_raw[bi] > worst_raw) worst_raw = rel_raw[bi];
            if (rel_raw[bi] > FEAS_RAW) ok_raw = 0;
        }
        J_act *= (1.0 / (double)NBITS);
        J_tail *= (1.0 / (double)NBITS);
        J_block *= (1.0 / (double)NBITS);
        feasible = (J_block <= FEAS_BLOCK && ok_raw) ? 1 : 0;
    }
};

static bool better_than(const CandMetrics& a, const uint32_t ap[8],
                        const CandMetrics& b, const uint32_t bp[8]) {
    if (a.feasible != b.feasible) return a.feasible > b.feasible;
    if (a.J_act < b.J_act) return true;
    if (a.J_act > b.J_act) return false;
    if (a.J_tail < b.J_tail) return true;
    if (a.J_tail > b.J_tail) return false;
    if (a.J_block < b.J_block) return true;
    if (a.J_block > b.J_block) return false;
    return cmp_u32_pack(ap, bp) < 0;
}

struct GpuCtx {
    float* dU = nullptr;
    float* dImp = nullptr;
    float* dEnergy = nullptr;
    uint8_t* dTail = nullptr;
    float* dD1 = nullptr;
    double* dSse = nullptr;
    double* dAct = nullptr;
    double* dTsse = nullptr;
    double* dRsum = nullptr;
    float* dD2 = nullptr;
    int nblocks = 0;
    int max_cand = 0;
    double eps = 0.0;
    float tail_thr = 0.0f;
    size_t tail_count = 0;
    int grid_x = 0;
    std::vector<float> hU;
    std::vector<float> hImp;
    std::vector<float> hEnergy;
    std::vector<uint8_t> hTail;
    float hD1[G]{};
};

static void build_energy_tail(GpuCtx& ctx, const std::vector<float>& U,
                              int nblocks) {
    ctx.nblocks = nblocks;
    ctx.hU = U;
    const size_t n = (size_t)nblocks * G;

    ctx.hEnergy.resize((size_t)nblocks);
    for (int b = 0; b < nblocks; ++b) {
        double e = 0.0;
        for (int i = 0; i < G; ++i) {
            double v = (double)U[(size_t)b * G + i];
            e += v * v;
        }
        ctx.hEnergy[(size_t)b] = (float)e;
    }

    std::vector<float> sorted_e = ctx.hEnergy;
    std::nth_element(sorted_e.begin(), sorted_e.begin() + nblocks / 2,
                     sorted_e.end());
    ctx.eps = EPS_SCALE * (double)sorted_e[(size_t)nblocks / 2];
    if (!(ctx.eps >= 0.0) || !std::isfinite(ctx.eps)) ctx.eps = 0.0;

    // Fixed-count top ceil(1%) of |U|; ties: smaller (block,index) preferred.
    std::vector<float> abs_u(n);
    for (size_t i = 0; i < n; ++i) abs_u[i] = std::fabs(U[i]);
    size_t k = (n + 99) / 100;
    if (k < 1) k = 1;
    if (k > n) k = n;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + (std::ptrdiff_t)k, order.end(),
        [&](size_t a, size_t b) {
            if (abs_u[a] != abs_u[b]) return abs_u[a] > abs_u[b];
            return a < b;
        });
    ctx.hTail.assign(n, 0);
    ctx.tail_count = k;
    ctx.tail_thr = abs_u[order[k - 1]];
    for (size_t t = 0; t < k; ++t) ctx.hTail[order[t]] = 1;
}

static void gpu_alloc(GpuCtx& ctx, int nblocks, int max_cand) {
    ctx.nblocks = nblocks;
    ctx.max_cand = max_cand;
    const size_t n = (size_t)nblocks * G;
    CK(hipMalloc(&ctx.dU, n * sizeof(float)));
    CK(hipMalloc(&ctx.dImp, n * sizeof(float)));
    CK(hipMalloc(&ctx.dEnergy, (size_t)nblocks * sizeof(float)));
    CK(hipMalloc(&ctx.dTail, n * sizeof(uint8_t)));
    CK(hipMalloc(&ctx.dD1, (size_t)G * sizeof(float)));
    CK(hipMalloc(&ctx.dSse, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dAct, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dTsse, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dRsum, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dD2, (size_t)max_cand * G * sizeof(float)));
    ctx.grid_x = std::min(nblocks, 4096);
}

static void gpu_upload_static_imp(GpuCtx& ctx) {
    const size_t n = (size_t)ctx.nblocks * G;
    CK(hipMemcpy(ctx.dImp, ctx.hImp.data(), n * sizeof(float),
                 hipMemcpyHostToDevice));
}

static void gpu_upload_U_d1(GpuCtx& ctx) {
    const size_t n = (size_t)ctx.nblocks * G;
    CK(hipMemcpy(ctx.dU, ctx.hU.data(), n * sizeof(float),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(ctx.dEnergy, ctx.hEnergy.data(),
                 (size_t)ctx.nblocks * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(ctx.dTail, ctx.hTail.data(), n * sizeof(uint8_t),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(ctx.dD1, ctx.hD1, (size_t)G * sizeof(float),
                 hipMemcpyHostToDevice));
}

static void gpu_init(GpuCtx& ctx, const std::vector<float>& U,
                     const std::vector<float>& Imp, const float* d1,
                     int nblocks, int max_cand) {
    build_energy_tail(ctx, U, nblocks);
    ctx.hImp = Imp;
    std::memcpy(ctx.hD1, d1, (size_t)G * sizeof(float));
    gpu_alloc(ctx, nblocks, max_cand);
    gpu_upload_static_imp(ctx);
    gpu_upload_U_d1(ctx);

    const size_t n = (size_t)nblocks * G;
    double imp_sum = 0.0;
    for (size_t i = 0; i < n; ++i) imp_sum += (double)Imp[i];
    std::printf("corpus: nblocks=%d coeffs=%zu eps=%.6e tail_thr=%.6e "
                "tail_n=%zu (%.3f%%) imp_sum=%.6e\n",
                nblocks, n, ctx.eps, (double)ctx.tail_thr, ctx.tail_count,
                100.0 * (double)ctx.tail_count / (double)n, imp_sum);
}

// Rebind U/tail/energy/d1 for a new D1 (importance stays).
static void gpu_rebind_pair(GpuCtx& ctx, const std::vector<float>& U,
                            const float* d1) {
    build_energy_tail(ctx, U, ctx.nblocks);
    std::memcpy(ctx.hD1, d1, (size_t)G * sizeof(float));
    gpu_upload_U_d1(ctx);
}

static void gpu_free(GpuCtx& ctx) {
    if (ctx.dU) hipFree(ctx.dU);
    if (ctx.dImp) hipFree(ctx.dImp);
    if (ctx.dEnergy) hipFree(ctx.dEnergy);
    if (ctx.dTail) hipFree(ctx.dTail);
    if (ctx.dD1) hipFree(ctx.dD1);
    if (ctx.dSse) hipFree(ctx.dSse);
    if (ctx.dAct) hipFree(ctx.dAct);
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
    CK(hipMemset(ctx.dAct, 0, zbytes));
    CK(hipMemset(ctx.dTsse, 0, zbytes));
    CK(hipMemset(ctx.dRsum, 0, zbytes));
    CK(hipMemcpy(ctx.dD2, d2_host, (size_t)ncand * G * sizeof(float),
                 hipMemcpyHostToDevice));

    dim3 grid((unsigned)ctx.grid_x, (unsigned)ncand, 1);
    dim3 block(G, 1, 1);
    hipLaunchKernelGGL(eval_activation_batch, grid, block, 0, 0, ctx.dU,
                       ctx.dImp, ctx.dEnergy, ctx.dTail, ctx.dD1, ctx.nblocks,
                       ctx.dD2, ncand, ctx.eps, ctx.dSse, ctx.dAct, ctx.dTsse,
                       ctx.dRsum);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    std::vector<double> hs(ncand * NBITS), ha(ncand * NBITS), ht(ncand * NBITS),
        hr(ncand * NBITS);
    CK(hipMemcpy(hs.data(), ctx.dSse, zbytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ha.data(), ctx.dAct, zbytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ht.data(), ctx.dTsse, zbytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hr.data(), ctx.dRsum, zbytes, hipMemcpyDeviceToHost));

    std::vector<CandMetrics> out(ncand);
    const double inv_nb = 1.0 / (double)ctx.nblocks;
    for (int c = 0; c < ncand; ++c) {
        for (int bi = 0; bi < NBITS; ++bi) {
            out[c].sse[bi] = hs[(size_t)c * NBITS + bi];
            out[c].act[bi] = ha[(size_t)c * NBITS + bi];
            out[c].tsse[bi] = ht[(size_t)c * NBITS + bi];
            out[c].r[bi] = hr[(size_t)c * NBITS + bi] * inv_nb;
        }
    }
    return out;
}

struct EvalRow {
    int bits;
    double mse, act_mse, r_nmse, rel_act, rel_raw, rel_block, tail_mse, rel_tail;
    double abs_p99, abs_p99_9, abs_max;
    double act_blk_p99, act_blk_p99_9, act_blk_max;
};

// Host reference path — same math as the kernel; used for eval detail.
static std::vector<EvalRow> host_eval_detail(const GpuCtx& ctx, const float* d2,
                                             const CandMetrics* canon) {
    const int nblocks = ctx.nblocks;
    const size_t n = (size_t)nblocks * G;
    std::vector<EvalRow> rows;
    rows.reserve(NBITS);

    for (int bi = 0; bi < NBITS; ++bi) {
        const float levels = (float)QMAX[bi];
        double sse = 0.0, act = 0.0, rsum = 0.0, tsse = 0.0;
        size_t tcount = 0;
        std::vector<float> abs_err(n);
        std::vector<double> blk_act((size_t)nblocks, 0.0);
        double abs_max = 0.0;

        for (int blk = 0; blk < nblocks; ++blk) {
            float v[G];
            float e_rot[G];
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
                    e_rot[li] = recon - u;
                    const double e2 = (double)e_rot[li] * (double)e_rot[li];
                    const size_t gi = (size_t)blk * G + (size_t)li;
                    const float ae = std::fabs(e_rot[li]);
                    abs_err[gi] = ae;
                    if ((double)ae > abs_max) abs_max = (double)ae;
                    sse += e2;
                    blk_sse += e2;
                    if (ctx.hTail[gi]) {
                        tsse += e2;
                        ++tcount;
                    }
                }
            }
            rsum += blk_sse / denom;

            float e_orig[G];
            std::memcpy(e_orig, e_rot, sizeof(e_orig));
            inverse_error_block(e_orig, ctx.hD1, d2);
            double ba = 0.0;
            for (int i = 0; i < G; ++i) {
                const size_t gi = (size_t)blk * G + (size_t)i;
                const double eo = (double)e_orig[i];
                const double term = (double)ctx.hImp[gi] * eo * eo;
                act += term;
                ba += term;
            }
            blk_act[(size_t)blk] = ba;
        }

        std::vector<float> sorted = abs_err;
        std::sort(sorted.begin(), sorted.end());
        auto pct_f = [&](double p) -> double {
            double idx = p * (double)(sorted.size() - 1);
            size_t i0 = (size_t)idx;
            size_t i1 = std::min(i0 + 1, sorted.size() - 1);
            double t = idx - (double)i0;
            return (1.0 - t) * (double)sorted[i0] + t * (double)sorted[i1];
        };

        std::vector<double> sblk = blk_act;
        std::sort(sblk.begin(), sblk.end());
        auto pct_b = [&](double p) -> double {
            if (sblk.empty()) return 0.0;
            double idx = p * (double)(sblk.size() - 1);
            size_t i0 = (size_t)idx;
            size_t i1 = std::min(i0 + 1, sblk.size() - 1);
            double t = idx - (double)i0;
            return (1.0 - t) * sblk[i0] + t * sblk[i1];
        };

        EvalRow row;
        row.bits = BITS[bi];
        row.mse = sse / (double)n;
        row.act_mse = act / (double)n;
        row.r_nmse = rsum / (double)nblocks;
        row.tail_mse = tcount ? tsse / (double)tcount : 0.0;
        if (canon) {
            const double aa = canon->act[bi] > 0.0 ? canon->act[bi] : 1e-300;
            const double sa = canon->sse[bi] > 0.0 ? canon->sse[bi] : 1e-300;
            const double ra = canon->r[bi] > 0.0 ? canon->r[bi] : 1e-300;
            const double ta = canon->tsse[bi] > 0.0 ? canon->tsse[bi] : 1e-300;
            row.rel_act = act / aa;
            row.rel_raw = sse / sa;
            row.rel_block = row.r_nmse / ra;
            row.rel_tail = tsse / ta;
        } else {
            row.rel_act = 1.0;
            row.rel_raw = 1.0;
            row.rel_block = 1.0;
            row.rel_tail = 1.0;
        }
        row.abs_p99 = pct_f(0.99);
        row.abs_p99_9 = pct_f(0.999);
        row.abs_max = abs_max;
        row.act_blk_p99 = pct_b(0.99);
        row.act_blk_p99_9 = pct_b(0.999);
        row.act_blk_max = sblk.empty() ? 0.0 : sblk.back();
        rows.push_back(row);
    }
    return rows;
}

static std::vector<float> pre_rotate_corpus(const std::vector<float>& raw,
                                            int nblocks, const float* d1) {
    std::vector<float> U = raw;
    for (int b = 0; b < nblocks; ++b)
        pre_rotate_block(U.data() + (size_t)b * G, d1);
    return U;
}

static CandMetrics eval_pair_on_ctx(GpuCtx& ctx, const float* d2) {
    return gpu_eval_batch(ctx, d2, 1)[0];
}

// Evaluate (d1,d2) by exact host f32 FWHT for d1 then GPU D2/affine/act path.
// Rebinds ctx U/tail/d1 to this d1's pre-D2 transform.
static CandMetrics eval_pair_rebind(GpuCtx& ctx, const std::vector<float>& raw,
                                    int nblocks, const float* d1,
                                    const float* d2) {
    auto U = pre_rotate_corpus(raw, nblocks, d1);
    gpu_rebind_pair(ctx, U, d1);
    return eval_pair_on_ctx(ctx, d2);
}

static CandMetrics eval_canonical(GpuCtx& scratch, const std::vector<float>& raw,
                                  int nblocks) {
    float d1[G], d2[G];
    gen_fwht_signs(D1_CANON_SEED, d1, G);
    gen_fwht_signs(D2_CANON_SEED, d2, G);
    return eval_pair_rebind(scratch, raw, nblocks, d1, d2);
}

static void write_fixed_pair(const std::string& prefix, const char* which_fixed,
                             const float* fixed_signs, const float* best_signs,
                             const char* which_best) {
    write_signs(prefix + ".fixed." + which_fixed + ".signs", fixed_signs);
    write_u32_txt(prefix + ".fixed." + which_fixed + ".u32.txt", fixed_signs);
    write_signs(prefix + ".best." + which_best + ".signs", best_signs);
    write_u32_txt(prefix + ".best." + which_best + ".u32.txt", best_signs);
}

static void write_final_pair(const std::string& prefix, const char* which_fixed,
                             const float* fixed_signs, const float* final_signs,
                             const char* which_final) {
    write_signs(prefix + ".fixed." + which_fixed + ".signs", fixed_signs);
    write_u32_txt(prefix + ".fixed." + which_fixed + ".u32.txt", fixed_signs);
    write_signs(prefix + ".final." + which_final + ".signs", final_signs);
    write_u32_txt(prefix + ".final." + which_final + ".u32.txt", final_signs);
}

static std::string ranked_header() {
    return "rank,seed,label,table,feasible,J_act,J_tail,J_block,worst_raw,"
           "act2,act3,act4,sse2,sse3,sse4,tsse2,tsse3,tsse4,r2,r3,r4,"
           "rel_act2,rel_act3,rel_act4,rel_tail2,rel_tail3,rel_tail4,"
           "rel_raw2,rel_raw3,rel_raw4,rel_block2,rel_block3,rel_block4,"
           "mse2,mse3,mse4,act_mse2,act_mse3,act_mse4,"
           "tail_mse2,tail_mse3,tail_mse4,"
           "signs_u32_0,signs_u32_1,signs_u32_2,signs_u32_3,"
           "signs_u32_4,signs_u32_5,signs_u32_6,signs_u32_7\n";
}

static void append_ranked_row(std::string& csv, int rank, uint32_t seed,
                              const char* label, const char* table,
                              const CandMetrics& m, const uint32_t pack[8],
                              int nblocks, size_t tail_count) {
    const double denom = (double)nblocks * (double)G;
    const double tden = (double)std::max<size_t>(1, tail_count);
    char line[1600];
    std::snprintf(
        line, sizeof(line),
        "%d,%u,%s,%s,%d,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
        rank, seed, label, table, m.feasible, m.J_act, m.J_tail, m.J_block,
        m.worst_raw, m.act[0], m.act[1], m.act[2], m.sse[0], m.sse[1], m.sse[2],
        m.tsse[0], m.tsse[1], m.tsse[2], m.r[0], m.r[1], m.r[2], m.rel_act[0],
        m.rel_act[1], m.rel_act[2], m.rel_tail[0], m.rel_tail[1], m.rel_tail[2],
        m.rel_raw[0], m.rel_raw[1], m.rel_raw[2], m.rel_block[0], m.rel_block[1],
        m.rel_block[2], m.sse[0] / denom, m.sse[1] / denom, m.sse[2] / denom,
        m.act[0] / denom, m.act[1] / denom, m.act[2] / denom, m.tsse[0] / tden,
        m.tsse[1] / tden, m.tsse[2] / tden, pack[0], pack[1], pack[2], pack[3],
        pack[4], pack[5], pack[6], pack[7]);
    csv += line;
}

static void run_parity_for_pair(const float* d1_cand, const float* d2_cand,
                                const char* tag) {
    float d1c[G], d2c[G];
    gen_fwht_signs(D1_CANON_SEED, d1c, G);
    gen_fwht_signs(D2_CANON_SEED, d2c, G);
    parity_check_pair(d1c, d2c, d1_cand, d2_cand, tag);
}

// ── modes ───────────────────────────────────────────────────────────────────

static void mode_d2_random(const char* raw_path, const char* imp_path,
                           const char* d1_path, int N, uint32_t rng_seed,
                           const char* out_prefix) {
    if (N < 0) die("N must be >= 0");
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto d1 = load_signs(d1_path, "D1_SIGNS");
    auto U = pre_rotate_corpus(raw, nblocks, d1.data());

    float d2_probe[G];
    gen_fwht_signs(rng_seed ? rng_seed : 1u, d2_probe, G);
    run_parity_for_pair(d1.data(), d2_probe, "rng-probe");

    const int ncand = N + 1;
    const int batch_cap = std::min(std::max(ncand, 1), 256);
    GpuCtx ctx;
    gpu_init(ctx, U, imp, d1.data(), nblocks, batch_cap);
    const size_t tail_count = ctx.tail_count;

    // Canonical reference pair (42, 1042) — may differ from supplied D1.
    GpuCtx ctx_canon;
    {
        float d1c[G];
        gen_fwht_signs(D1_CANON_SEED, d1c, G);
        auto U0 = pre_rotate_corpus(raw, nblocks, d1c);
        gpu_init(ctx_canon, U0, imp, d1c, nblocks, 1);
    }
    CandMetrics canon_m = eval_canonical(ctx_canon, raw, nblocks);
    gpu_free(ctx_canon);

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

    {
        std::vector<float> signs(G);
        gen_fwht_signs(D2_CANON_SEED, signs.data(), G);
        CandMetrics m = gpu_eval_batch(ctx, signs.data(), 1)[0];
        m.finalize(canon_m);
        Row r;
        r.seed = D2_CANON_SEED;
        r.label = "canonical";
        r.m = m;
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
                     "warning: no feasible candidate "
                     "(J_block<=%.4f & raw<=%.4f); emitting best infeasible "
                     "by J_act\n",
                     FEAS_BLOCK, FEAS_RAW);
    }

    std::string csv = ranked_header();
    for (const auto& r : rows)
        append_ranked_row(csv, r.rank, r.seed, r.label, "d2", r.m, r.pack,
                          nblocks, tail_count);
    atomic_write_text(std::string(out_prefix) + ".ranked.csv", csv);
    write_fixed_pair(out_prefix, "d1", d1.data(), rows[best_i].signs.data(),
                     "d2");

    std::printf("d2-random: candidates=%d best_rank=%d seed=%u (%s) feasible=%d "
                "J_act=%.6e J_tail=%.6e J_block=%.6e u32=%08x…\n",
                ncand, rows[best_i].rank, rows[best_i].seed, rows[best_i].label,
                rows[best_i].m.feasible, rows[best_i].m.J_act,
                rows[best_i].m.J_tail, rows[best_i].m.J_block,
                rows[best_i].pack[0]);
}

static void mode_d1_random(const char* raw_path, const char* imp_path,
                           const char* d2_path, int N, uint32_t rng_seed,
                           const char* out_prefix) {
    if (N < 0) die("N must be >= 0");
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto d2 = load_signs(d2_path, "D2_SIGNS");

    float d1_probe[G];
    gen_fwht_signs(rng_seed ? rng_seed : 1u, d1_probe, G);
    run_parity_for_pair(d1_probe, d2.data(), "rng-probe");

    // Scratch ctx; U/d1 rebound per D1 candidate (exact host f32 FWHT).
    GpuCtx ctx;
    {
        float d1tmp[G];
        gen_fwht_signs(D1_CANON_SEED, d1tmp, G);
        auto U0 = pre_rotate_corpus(raw, nblocks, d1tmp);
        gpu_init(ctx, U0, imp, d1tmp, nblocks, 1);
    }
    CandMetrics canon_m = eval_canonical(ctx, raw, nblocks);

    const int ncand = N + 1;
    struct Row {
        int rank = 0;
        uint32_t seed = 0;
        const char* label = "";
        CandMetrics m;
        std::vector<float> signs;
        uint32_t pack[8]{};
        size_t tail_count = 0;
    };
    std::vector<Row> rows;
    rows.reserve((size_t)ncand);

    auto eval_d1 = [&](const float* d1) -> CandMetrics {
        CandMetrics m = eval_pair_rebind(ctx, raw, nblocks, d1, d2.data());
        m.finalize(canon_m);
        return m;
    };

    {
        std::vector<float> signs(G);
        gen_fwht_signs(D1_CANON_SEED, signs.data(), G);
        Row r;
        r.seed = D1_CANON_SEED;
        r.label = "canonical";
        r.m = eval_d1(signs.data());
        r.tail_count = ctx.tail_count;
        r.signs = std::move(signs);
        pack_signs_u32(r.signs.data(), r.pack);
        rows.push_back(std::move(r));
    }

    for (int i = 0; i < N; ++i) {
        uint32_t seed = rng_seed + (uint32_t)i + 1u;
        if (seed == D1_CANON_SEED) seed += 0x9e3779b9u;
        float s[G];
        gen_fwht_signs(seed, s, G);
        Row r;
        r.seed = seed;
        r.label = "random";
        r.m = eval_d1(s);
        r.tail_count = ctx.tail_count;
        r.signs.assign(s, s + G);
        pack_signs_u32(r.signs.data(), r.pack);
        rows.push_back(std::move(r));
    }
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
                     "warning: no feasible candidate "
                     "(J_block<=%.4f & raw<=%.4f); emitting best infeasible "
                     "by J_act\n",
                     FEAS_BLOCK, FEAS_RAW);
    }

    std::string csv = ranked_header();
    for (const auto& r : rows)
        append_ranked_row(csv, r.rank, r.seed, r.label, "d1", r.m, r.pack,
                          nblocks, r.tail_count);
    atomic_write_text(std::string(out_prefix) + ".ranked.csv", csv);
    write_fixed_pair(out_prefix, "d2", d2.data(), rows[best_i].signs.data(),
                     "d1");

    std::printf("d1-random: candidates=%d best_rank=%d seed=%u (%s) feasible=%d "
                "J_act=%.6e J_tail=%.6e J_block=%.6e u32=%08x…\n",
                ncand, rows[best_i].rank, rows[best_i].seed, rows[best_i].label,
                rows[best_i].m.feasible, rows[best_i].m.J_act,
                rows[best_i].m.J_tail, rows[best_i].m.J_block,
                rows[best_i].pack[0]);
}

static void mode_d2_coord(const char* raw_path, const char* imp_path,
                          const char* d1_path, const char* d2_path, int passes,
                          const char* out_prefix) {
    if (passes < 1) die("PASSES must be >= 1");
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto d1 = load_signs(d1_path, "D1_SIGNS");
    auto cur = load_signs(d2_path, "D2_SIGNS");
    auto U = pre_rotate_corpus(raw, nblocks, d1.data());
    run_parity_for_pair(d1.data(), cur.data(), "coord-start");

    GpuCtx ctx;
    gpu_init(ctx, U, imp, d1.data(), nblocks, 1);

    GpuCtx ctx_canon;
    {
        float d1c[G];
        gen_fwht_signs(D1_CANON_SEED, d1c, G);
        auto U0 = pre_rotate_corpus(raw, nblocks, d1c);
        gpu_init(ctx_canon, U0, imp, d1c, nblocks, 1);
    }
    CandMetrics canon_m = eval_canonical(ctx_canon, raw, nblocks);
    gpu_free(ctx_canon);

    auto eval_one = [&](const float* s) {
        CandMetrics m = gpu_eval_batch(ctx, s, 1)[0];
        m.finalize(canon_m);
        return m;
    };

    CandMetrics m_cur = eval_one(cur.data());
    uint32_t pack_cur[8];
    pack_signs_u32(cur.data(), pack_cur);

    std::string csv;
    csv += "pass,flip_idx,accepted,feasible,J_act,J_tail,J_block,delta_J_act,"
           "worst_raw,act2,act3,act4,sse2,sse3,sse4,tsse2,tsse3,tsse4,"
           "r2,r3,r4,rel_act2,rel_act3,rel_act4,rel_tail2,rel_tail3,rel_tail4,"
           "rel_raw2,rel_raw3,rel_raw4,rel_block2,rel_block3,rel_block4\n";
    char line[1100];
    auto emit = [&](int pass, int flip, int acc, const CandMetrics& m,
                    double dJa) {
        std::snprintf(
            line, sizeof(line),
            "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            pass, flip, acc, m.feasible, m.J_act, m.J_tail, m.J_block, dJa,
            m.worst_raw, m.act[0], m.act[1], m.act[2], m.sse[0], m.sse[1],
            m.sse[2], m.tsse[0], m.tsse[1], m.tsse[2], m.r[0], m.r[1], m.r[2],
            m.rel_act[0], m.rel_act[1], m.rel_act[2], m.rel_tail[0],
            m.rel_tail[1], m.rel_tail[2], m.rel_raw[0], m.rel_raw[1],
            m.rel_raw[2], m.rel_block[0], m.rel_block[1], m.rel_block[2]);
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
                const double dJa = mt.J_act - m_cur.J_act;
                cur.swap(trial);
                m_cur = mt;
                std::memcpy(pack_cur, pack_t, sizeof(pack_cur));
                ++accepted_n;
                emit(p, fi, 1, m_cur, dJa);
            }
        }
        emit(p, -1, 0, m_cur, 0.0);
        std::printf("d2-coord pass %d/%d: accepted=%d feasible=%d J_act=%.6e "
                    "J_tail=%.6e J_block=%.6e\n",
                    p, passes, accepted_n, m_cur.feasible, m_cur.J_act,
                    m_cur.J_tail, m_cur.J_block);
    }
    gpu_free(ctx);

    atomic_write_text(std::string(out_prefix) + ".coord.csv", csv);
    write_final_pair(out_prefix, "d1", d1.data(), cur.data(), "d2");
}

static void mode_d1_coord(const char* raw_path, const char* imp_path,
                          const char* d1_path, const char* d2_path, int passes,
                          const char* out_prefix) {
    if (passes < 1) die("PASSES must be >= 1");
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto cur = load_signs(d1_path, "D1_SIGNS");
    auto d2 = load_signs(d2_path, "D2_SIGNS");
    run_parity_for_pair(cur.data(), d2.data(), "coord-start");

    GpuCtx ctx;
    {
        auto U0 = pre_rotate_corpus(raw, nblocks, cur.data());
        gpu_init(ctx, U0, imp, cur.data(), nblocks, 1);
    }
    CandMetrics canon_m = eval_canonical(ctx, raw, nblocks);

    auto eval_one = [&](const float* d1) {
        CandMetrics m = eval_pair_rebind(ctx, raw, nblocks, d1, d2.data());
        m.finalize(canon_m);
        return m;
    };

    CandMetrics m_cur = eval_one(cur.data());
    uint32_t pack_cur[8];
    pack_signs_u32(cur.data(), pack_cur);

    std::string csv;
    csv += "pass,flip_idx,accepted,feasible,J_act,J_tail,J_block,delta_J_act,"
           "worst_raw,act2,act3,act4,sse2,sse3,sse4,tsse2,tsse3,tsse4,"
           "r2,r3,r4,rel_act2,rel_act3,rel_act4,rel_tail2,rel_tail3,rel_tail4,"
           "rel_raw2,rel_raw3,rel_raw4,rel_block2,rel_block3,rel_block4\n";
    char line[1100];
    auto emit = [&](int pass, int flip, int acc, const CandMetrics& m,
                    double dJa) {
        std::snprintf(
            line, sizeof(line),
            "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            pass, flip, acc, m.feasible, m.J_act, m.J_tail, m.J_block, dJa,
            m.worst_raw, m.act[0], m.act[1], m.act[2], m.sse[0], m.sse[1],
            m.sse[2], m.tsse[0], m.tsse[1], m.tsse[2], m.r[0], m.r[1], m.r[2],
            m.rel_act[0], m.rel_act[1], m.rel_act[2], m.rel_tail[0],
            m.rel_tail[1], m.rel_tail[2], m.rel_raw[0], m.rel_raw[1],
            m.rel_raw[2], m.rel_block[0], m.rel_block[1], m.rel_block[2]);
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
                const double dJa = mt.J_act - m_cur.J_act;
                cur.swap(trial);
                m_cur = mt;
                std::memcpy(pack_cur, pack_t, sizeof(pack_cur));
                ++accepted_n;
                emit(p, fi, 1, m_cur, dJa);
            }
        }
        emit(p, -1, 0, m_cur, 0.0);
        std::printf("d1-coord pass %d/%d: accepted=%d feasible=%d J_act=%.6e "
                    "J_tail=%.6e J_block=%.6e\n",
                    p, passes, accepted_n, m_cur.feasible, m_cur.J_act,
                    m_cur.J_tail, m_cur.J_block);
    }
    gpu_free(ctx);

    atomic_write_text(std::string(out_prefix) + ".coord.csv", csv);
    write_final_pair(out_prefix, "d2", d2.data(), cur.data(), "d1");
}

static void mode_eval(const char* raw_path, const char* imp_path,
                      const char* d1_path, const char* d2_path,
                      const char* out_prefix) {
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto d1 = load_signs(d1_path, "D1_SIGNS");
    auto d2 = load_signs(d2_path, "D2_SIGNS");

    float d1c[G], d2c[G];
    gen_fwht_signs(D1_CANON_SEED, d1c, G);
    gen_fwht_signs(D2_CANON_SEED, d2c, G);
    run_parity_for_pair(d1.data(), d2.data(), "eval-candidate");

    GpuCtx ctx;
    auto U_cand = pre_rotate_corpus(raw, nblocks, d1.data());
    gpu_init(ctx, U_cand, imp, d1.data(), nblocks, 1);

    CandMetrics cnd = gpu_eval_batch(ctx, d2.data(), 1)[0];

    // Canonical metrics on canonical U/mask/d1.
    auto U_can = pre_rotate_corpus(raw, nblocks, d1c);
    gpu_rebind_pair(ctx, U_can, d1c);
    CandMetrics cm = gpu_eval_batch(ctx, d2c, 1)[0];
    // Keep cm raw (un-finalized) as denominator base.
    cnd.finalize(cm);

    // Detail rows: candidate on its U/d1; canonical on its U/d1.
    gpu_rebind_pair(ctx, U_cand, d1.data());
    auto rows_c = host_eval_detail(ctx, d2.data(), &cm);

    gpu_rebind_pair(ctx, U_can, d1c);
    auto rows_k = host_eval_detail(ctx, d2c, &cm);

    // Parity: host act totals vs GPU for candidate (relative).
    for (int bi = 0; bi < NBITS; ++bi) {
        const double g = cnd.act[bi];
        const double h =
            rows_c[(size_t)bi].act_mse * (double)nblocks * (double)G;
        const double den = std::max(std::fabs(g), 1e-300);
        const double rel = std::fabs(g - h) / den;
        if (rel > 1e-5 && std::fabs(g - h) > 1e-6) {
            std::fprintf(stderr,
                         "warning: host/GPU act mismatch bits=%d "
                         "gpu=%.6e host=%.6e rel=%.3e\n",
                         BITS[bi], g, h, rel);
        }
    }

    uint32_t pack_d1[8], pack_d2[8], pack_d1c[8], pack_d2c[8];
    pack_signs_u32(d1.data(), pack_d1);
    pack_signs_u32(d2.data(), pack_d2);
    pack_signs_u32(d1c, pack_d1c);
    pack_signs_u32(d2c, pack_d2c);

    std::string csv;
    csv += "who,bits,mse,act_mse,r_nmse,rel_act,rel_raw,rel_block,tail_mse,"
           "rel_tail,abs_p99,abs_p99_9,abs_max,act_blk_p99,act_blk_p99_9,"
           "act_blk_max\n";
    char line[640];
    auto dump = [&](const char* who, const std::vector<EvalRow>& erows) {
        for (const auto& r : erows) {
            std::snprintf(
                line, sizeof(line),
                "%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                who, r.bits, r.mse, r.act_mse, r.r_nmse, r.rel_act, r.rel_raw,
                r.rel_block, r.tail_mse, r.rel_tail, r.abs_p99, r.abs_p99_9,
                r.abs_max, r.act_blk_p99, r.act_blk_p99_9, r.act_blk_max);
            csv += line;
        }
    };
    dump("candidate", rows_c);
    dump("canonical", rows_k);
    atomic_write_text(std::string(out_prefix) + ".eval.csv", csv);

    write_signs(std::string(out_prefix) + ".cand.d1.signs", d1.data());
    write_signs(std::string(out_prefix) + ".cand.d2.signs", d2.data());
    write_u32_txt(std::string(out_prefix) + ".cand.d1.u32.txt", d1.data());
    write_u32_txt(std::string(out_prefix) + ".cand.d2.u32.txt", d2.data());
    write_u32_txt(std::string(out_prefix) + ".canon.d1.u32.txt", d1c);
    write_u32_txt(std::string(out_prefix) + ".canon.d2.u32.txt", d2c);

    std::printf("eval: feasible=%d J_act=%.6e J_tail=%.6e J_block=%.6e "
                "worst_raw=%.6e\n"
                "cand d1_u32: %08x %08x %08x %08x %08x %08x %08x %08x\n"
                "cand d2_u32: %08x %08x %08x %08x %08x %08x %08x %08x\n"
                "canon d1_u32: %08x %08x %08x %08x %08x %08x %08x %08x\n"
                "canon d2_u32: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                cnd.feasible, cnd.J_act, cnd.J_tail, cnd.J_block, cnd.worst_raw,
                pack_d1[0], pack_d1[1], pack_d1[2], pack_d1[3], pack_d1[4],
                pack_d1[5], pack_d1[6], pack_d1[7], pack_d2[0], pack_d2[1],
                pack_d2[2], pack_d2[3], pack_d2[4], pack_d2[5], pack_d2[6],
                pack_d2[7], pack_d1c[0], pack_d1c[1], pack_d1c[2], pack_d1c[3],
                pack_d1c[4], pack_d1c[5], pack_d1c[6], pack_d1c[7], pack_d2c[0],
                pack_d2c[1], pack_d2c[2], pack_d2c[3], pack_d2c[4], pack_d2c[5],
                pack_d2c[6], pack_d2c[7]);
    for (int i = 0; i < NBITS; ++i) {
        std::printf("  bits=%d mse=%.6e/%.6e act_mse=%.6e/%.6e "
                    "rel_act=%.4f rel_raw=%.4f rel_block=%.4f rel_tail=%.4f "
                    "act_blk_p99=%.6e p99=%.6e p99.9=%.6e max=%.6e\n",
                    BITS[i], rows_c[(size_t)i].mse, rows_k[(size_t)i].mse,
                    rows_c[(size_t)i].act_mse, rows_k[(size_t)i].act_mse,
                    rows_c[(size_t)i].rel_act, rows_c[(size_t)i].rel_raw,
                    rows_c[(size_t)i].rel_block, rows_c[(size_t)i].rel_tail,
                    rows_c[(size_t)i].act_blk_p99, rows_c[(size_t)i].abs_p99,
                    rows_c[(size_t)i].abs_p99_9, rows_c[(size_t)i].abs_max);
    }
    gpu_free(ctx);
}

int main(int argc, char** argv) {
    if (argc < 2) usage(argv[0], 1);
    const std::string mode = argv[1];
    if (mode == "--help" || mode == "-h" || mode == "help") usage(argv[0], 0);

    CK(hipSetDevice(0));

    if (mode == "d1-random") {
        if (argc != 8) usage(argv[0], 1);
        mode_d1_random(argv[2], argv[3], argv[4], std::atoi(argv[5]),
                       (uint32_t)std::strtoul(argv[6], nullptr, 10), argv[7]);
        return 0;
    }
    if (mode == "d1-coord") {
        if (argc != 8) usage(argv[0], 1);
        mode_d1_coord(argv[2], argv[3], argv[4], argv[5], std::atoi(argv[6]),
                      argv[7]);
        return 0;
    }
    if (mode == "d2-random") {
        if (argc != 8) usage(argv[0], 1);
        mode_d2_random(argv[2], argv[3], argv[4], std::atoi(argv[5]),
                       (uint32_t)std::strtoul(argv[6], nullptr, 10), argv[7]);
        return 0;
    }
    if (mode == "d2-coord") {
        if (argc != 8) usage(argv[0], 1);
        mode_d2_coord(argv[2], argv[3], argv[4], argv[5], std::atoi(argv[6]),
                      argv[7]);
        return 0;
    }
    if (mode == "eval") {
        if (argc != 7) usage(argv[0], 1);
        mode_eval(argv[2], argv[3], argv[4], argv[5], argv[6]);
        return 0;
    }

    std::fprintf(stderr, "error: unknown mode '%s'\n", mode.c_str());
    usage(argv[0], 1);
    return 1;
}
