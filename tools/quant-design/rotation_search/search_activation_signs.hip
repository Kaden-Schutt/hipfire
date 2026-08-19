// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// search_activation_signs.hip — standalone gfx1201 HIP tool that searches a
// global D2 sign table against activation-weighted reconstruction error
// for production MQ4V2 and hypothetical matched-header 2-bit/3-bit V2
// research layouts (two G128 fp16 scale+zero pairs).
//
// Runtime contract (matches the engine):
//   Forward: R = D2 · H · D1 with one global 256-sign D1 (seed 42), one global
//   256-sign D2 (canonical seed 1042), one unnormalized 8-stage FWHT-256, then
//   ×1/16. Candidate D2 tables MUST preserve that op count — only the 256 signs
//   change.
//
//   Inverse of quantization error (exact f32, same op count):
//     e_original = D1 · H · D2 · e_rot / 16
//   i.e. apply D2 to e_rot, unnormalized FWHT-256, ×1/16, then D1.
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
//   U = D1→FWHT→×1/16 (no D2). |D2·U|=|U| so masks/energies are D2-invariant.
//   energy_b = Σ_i U[b,i]²
//   eps      = 2^{-24} · median_b(energy_b)
//   tail mask: fixed-count top ceil(1%) of |U| (pre-D2); ties by (block,index)
//
//   act_n(d)     = Σ importance · e_original_n(d)²          // PRIMARY mass
//   sse_n(d)     = Σ e_rot_n(d)²                            // raw rotated MSE
//   r_n(d)       = mean_b[ SSE_rot_n(d,b) / (energy_b+eps) ] // block NMSE
//   tsse_n(d)    = Σ_{i in tail} e_rot_n(d,i)²              // tail-1% MSE guard
//
//   J_act(d)   = (1/3) Σ_n act_n(d)  / act_n(canonical)     // PRIMARY rank
//   J_tail(d)  = (1/3) Σ_n tsse_n(d) / tsse_n(canonical)
//   J_block(d) = (1/3) Σ_n r_n(d)    / r_n(canonical)
//   rel_raw_n  = sse_n(d) / sse_n(canonical)
//
// Feasible iff:
//   J_block <= 1.0005  AND  max_n rel_raw_n <= 1.0025
//
// Selection (lexicographic, lower better):
//   1. feasible first
//   2. minimize J_act
//   3. minimize J_tail
//   4. minimize J_block
//   5. lex-smaller packed 256 sign bits (8×u32 LE)
//
// No heldout / split awareness: CLI takes explicit RAW and IMPORTANCE paths.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O3 search_activation_signs.hip \
//         -o search_activation_signs
//
// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
//   search_activation_signs --help
//   search_activation_signs random RAW IMPORTANCE N RNG_SEED OUT_PREFIX
//   search_activation_signs coord  RAW IMPORTANCE SIGNS PASSES OUT_PREFIX
//   search_activation_signs eval   RAW IMPORTANCE SIGNS OUT_PREFIX
//
// RAW / IMPORTANCE: paired headerless LE f32 [nblocks,256]; identical sizes.
// SIGNS: 256 × f32 LE exact ±1.0 (candidate D2 only).
//
// ranked.csv:
//   rank,seed,label,feasible,J_act,J_tail,J_block,worst_raw,
//   act2,act3,act4,sse2,sse3,sse4,tsse2,tsse3,tsse4,r2,r3,r4,
//   rel_act2,rel_act3,rel_act4,rel_tail2,rel_tail3,rel_tail4,
//   rel_raw2,rel_raw3,rel_raw4,rel_block2,rel_block3,rel_block4,
//   mse2,mse3,mse4,act_mse2,act_mse3,act_mse4,tail_mse2,tail_mse3,tail_mse4,
//   signs_u32_0..signs_u32_7
// coord.csv:
//   pass,flip_idx,accepted,feasible,J_act,J_tail,J_block,delta_J_act,worst_raw,
//   act*,sse*,tsse*,r*,rel_act*,rel_tail*,rel_raw*,rel_block*
// eval.csv:
//   who,bits,mse,act_mse,r_nmse,rel_act,rel_raw,rel_block,tail_mse,rel_tail,
//   abs_p99,abs_p99_9,abs_max,act_blk_p99,act_blk_p99_9,act_blk_max
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
R"(search_activation_signs — global D2 search on activation-weighted error

USAGE
  %s --help
  %s random RAW IMPORTANCE N RNG_SEED OUT_PREFIX
  %s coord  RAW IMPORTANCE SIGNS PASSES OUT_PREFIX
  %s eval   RAW IMPORTANCE SIGNS OUT_PREFIX

INPUTS
  RAW, IMPORTANCE
    Paired headerless little-endian f32 [nblocks,256] tensors.
    Same byte size, same block ordering. RAW = unrotated weight groups;
    IMPORTANCE = imatrix input-channel diagonal for that weight block
    (non-negative; non-finite rejected). Size must be a positive multiple
    of 1024 bytes. No heldout/split logic — paths are explicit.

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
          Includes per-bit ratios and p99/p99.9 activation-weighted per-block
          error mass.

TRANSFORMS AND CODEC
  Pre-rotate once: D1(seed 42) → unnormalized 8-stage FWHT-256 → ×1/16 → U.
  Search applies candidate D2 only: v = D2·U.
  Per G256: two independent contiguous G128 asymmetric affine grids for
  bits∈{2,3,4}:
    lo,hi = min/max of half; step=(hi-lo)/(2^b-1);
    st,z = f16_to_f32(f32_to_f16_TRUNC(.))  // float16.rs truncation, NOT RNE
    q = clamp(floor((v-z)/st + 0.5), 0, 2^b-1);  recon = q*st + z
  Degenerate half (hi==lo or st==0): q=0, recon=z.
  e_rot = recon - v.
  Inverse (exact f32, candidate D2): e_orig = D1·H·D2·e_rot/16.
  Activation mass: Σ importance · e_orig².

SCORES
  Primary: J_act = mean_n act_n(d)/act_n(canonical).
  Guards (also reported): raw rotated MSE, block NMSE, fixed pre-D2 tail-1%%
  MSE (top ceil(1%%) of |U|; ties by (block,index) ascending).
  Feasible iff J_block <= 1.0005 and every bit's raw-MSE ratio <= 1.0025.
  Among feasible: min J_act, then J_tail, then J_block, then lex-smaller
  packed sign bits.

PARITY
  On each run, host-side inverse-transform parity checks confirm
  inv(fwd(x)) ≈ x for canonical and candidate D2 (shared op count).

OUTPUT
  ranked.csv / coord.csv / eval.csv schemas documented in source header.
  *.signs: 256×f32 LE ±1.0; *.u32.txt: 8 hex u32 words (bit i set iff +1)

BUILD
  hipcc --offload-arch=gfx1201 -O3 search_activation_signs.hip \
        -o search_activation_signs
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

// Full forward: v = D2 · (H/16) · D1 · w  (in-place on blk copy path).
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

static void parity_check_inverse(const float* d1, const float* d2_a,
                                 const float* d2_b, const char* tag_b) {
    // inv(fwd(x)) must recover x for both D2 tables (same op count).
    float x[G], y[G], z[G];
    uint32_t st = 0xC0FFEEu;
    for (int i = 0; i < G; ++i) {
        uint32_t r = lcg_next(st);
        // Deterministic quasi-normal-ish samples in [-1,1].
        x[i] = ((float)(int)(r % 2001) - 1000.0f) / 1000.0f;
    }
    auto check_one = [&](const float* d2, const char* who) {
        std::memcpy(y, x, sizeof(y));
        forward_rotate_block(y, d1, d2);
        // Inverse of full rotation on values equals inverse-error map.
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
    check_one(d2_a, "canonical");
    check_one(d2_b, tag_b);

    // Cross-path op-count sanity: applying inverse with the other table's D2
    // to a zero error yields zero for both (trivial but catches D1 mixups).
    float zero[G];
    std::memset(zero, 0, sizeof(zero));
    inverse_error_block(zero, d1, d2_a);
    for (int i = 0; i < G; ++i) {
        if (zero[i] != 0.0f) die("canonical inverse of zero error is nonzero");
    }
    std::memset(zero, 0, sizeof(zero));
    inverse_error_block(zero, d1, d2_b);
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

        // Quantize each bit-width independently; inverse error per bit needs
        // separate e_rot buffers — process one bit at a time.
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
                // Butterfly: lower index gets a+b, upper gets a-b where
                // "lower" has the stride bit clear.
                float out;
                if ((tid & stride) == 0) {
                    out = a + b;
                } else {
                    // a is upper (tid has bit), b is lower partner
                    out = b - a;
                }
                __syncthreads();
                sh_e[tid] = out;
                __syncthreads();
            }
            const float e_orig = d1v * (sh_e[tid] * FWHT_SCALE);
            loc_act[bi] += (double)imp * (double)e_orig * (double)e_orig;

            // Block SSE reduction for NMSE (rotated space).
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

    // Reduce per-thread locals across the block into global atomics.
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

    // rsum already only on tid==0 per block-iteration; reduce those.
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

static void gpu_init(GpuCtx& ctx, const std::vector<float>& U,
                     const std::vector<float>& Imp, int nblocks, int max_cand) {
    ctx.nblocks = nblocks;
    ctx.max_cand = max_cand;
    ctx.hU = U;
    ctx.hImp = Imp;
    const size_t n = (size_t)nblocks * G;

    gen_fwht_signs(D1_SEED, ctx.hD1, G);

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

    CK(hipMalloc(&ctx.dU, n * sizeof(float)));
    CK(hipMemcpy(ctx.dU, U.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dImp, n * sizeof(float)));
    CK(hipMemcpy(ctx.dImp, Imp.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dEnergy, (size_t)nblocks * sizeof(float)));
    CK(hipMemcpy(ctx.dEnergy, ctx.hEnergy.data(),
                 (size_t)nblocks * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dTail, n * sizeof(uint8_t)));
    CK(hipMemcpy(ctx.dTail, ctx.hTail.data(), n * sizeof(uint8_t),
                 hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dD1, (size_t)G * sizeof(float)));
    CK(hipMemcpy(ctx.dD1, ctx.hD1, (size_t)G * sizeof(float),
                 hipMemcpyHostToDevice));
    CK(hipMalloc(&ctx.dSse, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dAct, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dTsse, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dRsum, (size_t)max_cand * NBITS * sizeof(double)));
    CK(hipMalloc(&ctx.dD2, (size_t)max_cand * G * sizeof(float)));
    ctx.grid_x = std::min(nblocks, 4096);

    double imp_sum = 0.0;
    for (size_t i = 0; i < n; ++i) imp_sum += (double)Imp[i];
    std::printf("corpus: nblocks=%d coeffs=%zu eps=%.6e tail_thr=%.6e "
                "tail_n=%zu (%.3f%%) imp_sum=%.6e\n",
                nblocks, n, ctx.eps, (double)ctx.tail_thr, ctx.tail_count,
                100.0 * (double)ctx.tail_count / (double)n, imp_sum);
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

// Host reference path — same math as the kernel; used for eval detail +
// optional cross-check.
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
                                            int nblocks) {
    float d1[G];
    gen_fwht_signs(D1_SEED, d1, G);
    std::vector<float> U = raw;
    for (int b = 0; b < nblocks; ++b)
        pre_rotate_block(U.data() + (size_t)b * G, d1);
    return U;
}

static void run_parity_for_signs(const float* d2_cand, const char* tag) {
    float d1[G], d2c[G];
    gen_fwht_signs(D1_SEED, d1, G);
    gen_fwht_signs(D2_CANON_SEED, d2c, G);
    parity_check_inverse(d1, d2c, d2_cand, tag);
}

static void mode_random(const char* raw_path, const char* imp_path, int N,
                        uint32_t rng_seed, const char* out_prefix) {
    if (N < 0) die("N must be >= 0");
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto U = pre_rotate_corpus(raw, nblocks);

    float d2_probe[G];
    gen_fwht_signs(rng_seed ? rng_seed : 1u, d2_probe, G);
    run_parity_for_signs(d2_probe, "rng-probe");

    const int ncand = N + 1;
    const int batch_cap = std::min(std::max(ncand, 1), 256);
    GpuCtx ctx;
    gpu_init(ctx, U, imp, nblocks, batch_cap);
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
                     "warning: no feasible candidate "
                     "(J_block<=%.4f & raw<=%.4f); emitting best infeasible "
                     "by J_act\n",
                     FEAS_BLOCK, FEAS_RAW);
    }

    const double denom = (double)nblocks * (double)G;
    const double tden = (double)std::max<size_t>(1, tail_count);

    std::string csv;
    csv += "rank,seed,label,feasible,J_act,J_tail,J_block,worst_raw,"
           "act2,act3,act4,sse2,sse3,sse4,tsse2,tsse3,tsse4,r2,r3,r4,"
           "rel_act2,rel_act3,rel_act4,rel_tail2,rel_tail3,rel_tail4,"
           "rel_raw2,rel_raw3,rel_raw4,rel_block2,rel_block3,rel_block4,"
           "mse2,mse3,mse4,act_mse2,act_mse3,act_mse4,"
           "tail_mse2,tail_mse3,tail_mse4,"
           "signs_u32_0,signs_u32_1,signs_u32_2,signs_u32_3,"
           "signs_u32_4,signs_u32_5,signs_u32_6,signs_u32_7\n";
    char line[1600];
    for (const auto& r : rows) {
        std::snprintf(
            line, sizeof(line),
            "%d,%u,%s,%d,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
            "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
            r.rank, r.seed, r.label, r.m.feasible, r.m.J_act, r.m.J_tail,
            r.m.J_block, r.m.worst_raw, r.m.act[0], r.m.act[1], r.m.act[2],
            r.m.sse[0], r.m.sse[1], r.m.sse[2], r.m.tsse[0], r.m.tsse[1],
            r.m.tsse[2], r.m.r[0], r.m.r[1], r.m.r[2], r.m.rel_act[0],
            r.m.rel_act[1], r.m.rel_act[2], r.m.rel_tail[0], r.m.rel_tail[1],
            r.m.rel_tail[2], r.m.rel_raw[0], r.m.rel_raw[1], r.m.rel_raw[2],
            r.m.rel_block[0], r.m.rel_block[1], r.m.rel_block[2],
            r.m.sse[0] / denom, r.m.sse[1] / denom, r.m.sse[2] / denom,
            r.m.act[0] / denom, r.m.act[1] / denom, r.m.act[2] / denom,
            r.m.tsse[0] / tden, r.m.tsse[1] / tden, r.m.tsse[2] / tden,
            r.pack[0], r.pack[1], r.pack[2], r.pack[3], r.pack[4], r.pack[5],
            r.pack[6], r.pack[7]);
        csv += line;
    }
    atomic_write_text(std::string(out_prefix) + ".ranked.csv", csv);
    write_signs(std::string(out_prefix) + ".best.signs",
                rows[best_i].signs.data());
    write_u32_txt(std::string(out_prefix) + ".best.u32.txt",
                  rows[best_i].signs.data());

    std::printf("random: candidates=%d best_rank=%d seed=%u (%s) feasible=%d "
                "J_act=%.6e J_tail=%.6e J_block=%.6e u32=%08x…\n",
                ncand, rows[best_i].rank, rows[best_i].seed, rows[best_i].label,
                rows[best_i].m.feasible, rows[best_i].m.J_act,
                rows[best_i].m.J_tail, rows[best_i].m.J_block,
                rows[best_i].pack[0]);
}

static void mode_coord(const char* raw_path, const char* imp_path,
                       const char* signs_path, int passes,
                       const char* out_prefix) {
    if (passes < 1) die("PASSES must be >= 1");
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto U = pre_rotate_corpus(raw, nblocks);
    auto cur = load_signs(signs_path);
    run_parity_for_signs(cur.data(), "coord-start");

    GpuCtx ctx;
    gpu_init(ctx, U, imp, nblocks, 1);

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
        std::printf("coord pass %d/%d: accepted=%d feasible=%d J_act=%.6e "
                    "J_tail=%.6e J_block=%.6e\n",
                    p, passes, accepted_n, m_cur.feasible, m_cur.J_act,
                    m_cur.J_tail, m_cur.J_block);
    }
    gpu_free(ctx);

    atomic_write_text(std::string(out_prefix) + ".coord.csv", csv);
    write_signs(std::string(out_prefix) + ".final.signs", cur.data());
    write_u32_txt(std::string(out_prefix) + ".final.u32.txt", cur.data());
}

static void mode_eval(const char* raw_path, const char* imp_path,
                      const char* signs_path, const char* out_prefix) {
    int nblocks = 0;
    std::vector<float> raw, imp;
    load_paired(raw_path, imp_path, raw, imp, nblocks);
    auto U = pre_rotate_corpus(raw, nblocks);
    auto cand = load_signs(signs_path);
    float canon[G];
    gen_fwht_signs(D2_CANON_SEED, canon, G);
    run_parity_for_signs(cand.data(), "eval-candidate");

    GpuCtx ctx;
    gpu_init(ctx, U, imp, nblocks, 1);

    CandMetrics cm = gpu_eval_batch(ctx, canon, 1)[0];
    CandMetrics cnd = gpu_eval_batch(ctx, cand.data(), 1)[0];
    cnd.finalize(cm);

    // Host detail for percentiles; also cross-check GPU totals.
    auto rows_c = host_eval_detail(ctx, cand.data(), &cm);
    auto rows_k = host_eval_detail(ctx, canon, &cm);

    // Parity: host act totals vs GPU for candidate (relative).
    for (int bi = 0; bi < NBITS; ++bi) {
        const double g = cnd.act[bi];
        // Recompute host act mass from rows: act_mse * n
        const double h = rows_c[(size_t)bi].act_mse * (double)nblocks * (double)G;
        const double den = std::max(std::fabs(g), 1e-300);
        const double rel = std::fabs(g - h) / den;
        if (rel > 1e-5 && std::fabs(g - h) > 1e-6) {
            std::fprintf(stderr,
                         "warning: host/GPU act mismatch bits=%d "
                         "gpu=%.6e host=%.6e rel=%.3e\n",
                         BITS[bi], g, h, rel);
        }
    }

    uint32_t pack[8];
    pack_signs_u32(cand.data(), pack);

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
    write_signs(std::string(out_prefix) + ".eval.signs", cand.data());
    write_u32_txt(std::string(out_prefix) + ".eval.u32.txt", cand.data());

    std::printf("eval: feasible=%d J_act=%.6e J_tail=%.6e J_block=%.6e "
                "worst_raw=%.6e\n"
                "signs_u32: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                cnd.feasible, cnd.J_act, cnd.J_tail, cnd.J_block, cnd.worst_raw,
                pack[0], pack[1], pack[2], pack[3], pack[4], pack[5], pack[6],
                pack[7]);
    for (int i = 0; i < NBITS; ++i) {
        std::printf("  bits=%d mse=%.6e/%.6e act_mse=%.6e/%.6e "
                    "rel_act=%.4f rel_raw=%.4f rel_block=%.4f rel_tail=%.4f "
                    "act_blk_p99=%.6e\n",
                    BITS[i], rows_c[(size_t)i].mse, rows_k[(size_t)i].mse,
                    rows_c[(size_t)i].act_mse, rows_k[(size_t)i].act_mse,
                    rows_c[(size_t)i].rel_act, rows_c[(size_t)i].rel_raw,
                    rows_c[(size_t)i].rel_block, rows_c[(size_t)i].rel_tail,
                    rows_c[(size_t)i].act_blk_p99);
    }
    gpu_free(ctx);
}

int main(int argc, char** argv) {
    if (argc < 2) usage(argv[0], 1);
    const std::string mode = argv[1];
    if (mode == "--help" || mode == "-h" || mode == "help") usage(argv[0], 0);

    CK(hipSetDevice(0));

    if (mode == "random") {
        if (argc != 7) usage(argv[0], 1);
        mode_random(argv[2], argv[3], std::atoi(argv[4]),
                    (uint32_t)std::strtoul(argv[5], nullptr, 10), argv[6]);
        return 0;
    }
    if (mode == "coord") {
        if (argc != 7) usage(argv[0], 1);
        mode_coord(argv[2], argv[3], argv[4], std::atoi(argv[5]), argv[6]);
        return 0;
    }
    if (mode == "eval") {
        if (argc != 6) usage(argv[0], 1);
        mode_eval(argv[2], argv[3], argv[4], argv[5]);
        return 0;
    }

    std::fprintf(stderr, "error: unknown mode '%s'\n", mode.c_str());
    usage(argv[0], 1);
    return 1;
}
