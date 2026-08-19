// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// bench_runtime_rotate.hip — gfx1201 microbenchmark proving a D2 table-only
// candidate has identical standalone mq_rotate_x runtime cost vs canonical D2.
//
// Scope
//   Standalone mq_rotate_x only. Fused rotate paths are out of scope.
//
// Experiment
//   Production mq_rotate_x implements R = D2 · H · D1 (signs1 → FWHT → ×1/16
//   → signs2). Canonical and candidate differ ONLY in the contents of equal-size
//   256-f32 signs2 buffers. One compiled mq_rotate_x function; identical launch
//   geometry, K, input, signs1, kernarg order, warmup, and iteration count.
//   Only the signs2 device pointer (same size buffer, different ±1 contents)
//   differs between arms. Uploads are outside timed regions.
//
// Method
//   1. Generate D1 (seed 42) and canonical D2 (seed 1042) with engine LCG.
//   2. Load candidate D2: exactly 256 little-endian f32 values, each ±1.
//   3. Deterministic host input (LCG seed 7) uploaded once (untimed).
//   4. Warm up both signs2 pointers, then collect paired HIP-event samples:
//        - AB arm: canonical vs candidate (randomized AB/BA order per pair)
//        - AA control: canonical vs canonical (timing noise floor)
//   5. Report median / p10 / p90 / mean microseconds and paired ratios.
//      Equivalence uses median paired ratio (cand/canon), not mean alone.
//   6. Emit every paired sample (canon_us, cand_us, ratio_ab, aa_*, ratio_aa)
//      so the parent can bootstrap CIs externally.
//   7. Untimed correctness: pre-D2 core must match (cand_out == canon_out *
//      signs2_canon * signs2_cand elementwise); all outputs finite; checksum
//      printed so the backend cannot DCE the kernels.
//
// CLI
//   bench_runtime_rotate --help
//   bench_runtime_rotate CANDIDATE.signs [K] [BATCH] [ITERS]
//
// Defaults: K=8192, BATCH=1, ITERS=1000, warmup=100 (warmup not CLI).
// Suggested sweep (parent): K∈{2048,4096,5120,8192,12288,32768}, B∈{1,8,32}.
// K must be a positive multiple of 256. BATCH and ITERS must be >= 1.
//
// Output (stdout)
//   identity / config lines, then:
//     canonical_us              median=… p10=… p90=… mean=…
//     candidate_us              median=… p10=… p90=… mean=…
//     ratio_cand_over_canon     median=… p10=… p90=… mean=…   ← primary
//     aa_control_us_a           median=… p10=… p90=… mean=…
//     aa_control_us_b           median=… p10=… p90=… mean=…
//     ratio_aa_control          median=… p10=… p90=… mean=…
//     equivalence  cand_over_canon_median_pct=… aa_spread_median_pct=…
//                  within_0p5pct=yes|no  indist_from_aa=yes|no
//                  (decision on median paired ratio, not mean)
//     finite_ok  yes
//     pre_d2_core_ok  yes
//     checksum  …
//     samples_begin
//     # i canon_us cand_us ratio_ab aa_a_us aa_b_us ratio_aa
//     <ITERS TSV rows>
//     samples_end
//   Zero-cost claim (parent): within_0p5pct AND indist_from_aa on medians.
//
// Build
//   hipcc --offload-arch=gfx1201 -O3 bench_runtime_rotate.hip -o bench_runtime_rotate
//
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>

// Production mq_rotate_x (and sibling gemv_mq4g256) — unmodified source.
// Relative to tools/quant-design/rotation_search/.
#include "../../../kernels/src/gemv_mq4g256.hip"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr int G = 256;
static constexpr uint32_t D1_SEED = 42u;
static constexpr uint32_t D2_CANON_SEED = 1042u;
static constexpr uint32_t INPUT_SEED = 7u;
static constexpr uint32_t PAIR_ORDER_SEED = 99u;
static constexpr int DEFAULT_K = 8192;
static constexpr int DEFAULT_BATCH = 1;
static constexpr int DEFAULT_ITERS = 1000;
static constexpr int WARMUP = 100;
static constexpr float ZERO_COST_PCT = 0.5f;

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
    std::fprintf(
        code ? stderr : stdout,
        R"(bench_runtime_rotate — prove D2 table-only candidate has identical mq_rotate_x cost

EXPERIMENT
  Production mq_rotate_x applies R = D2 · H · D1 per 256-element group on
  gfx1201 (wave32, block=32). Canonical D2 uses engine LCG seed 1042; D1 always
  seed 42. Candidate supplies a different 256-entry ±1 signs2 table only.
  One compiled mq_rotate_x binary/function; launch geometry, K, input, signs1,
  kernarg order, warmup, and iteration count are identical. Only the signs2
  device pointer differs between arms (equal-size 256-f32 buffers). Uploads are
  outside timed regions. Scope: standalone rotate only (not fused paths).

USAGE
  %s --help
  %s CANDIDATE.signs [K] [BATCH] [ITERS]

ARGUMENTS
  CANDIDATE.signs  Exactly 256 little-endian f32 values, each exactly ±1.0.
  K                Elements per batch row; positive multiple of 256 (default %d).
  BATCH            Batch rows; positive integer (default %d).
  ITERS            Timed paired iterations after warmup (default %d).
  Warmup is fixed at %d untimed paired launches (not configurable).

  Suggested parent sweep: K in {2048,4096,5120,8192,12288,32768}, BATCH in
  {1,8,32}, ITERS>=1000.

METHOD
  Generate D1/D2 via engine LCG (state = seed*1103515245+12345 & 0x7fffffff;
  sign = +1 iff (state>>16)&1). Deterministic input from LCG seed %u.
  After warmup:
    AB pairs — canonical vs candidate, randomized AB/BA order per pair
    AA pairs — canonical vs canonical (negative control / noise floor)
  Each launch timed with hipEventRecord on the null stream.
  Grid = [(K/256)*BATCH, 1, 1], block = [32, 1, 1], sharedMemBytes = 0
  (matches production rotate_x_mq / rotate_x_mq_batched on non-gfx942).

OUTPUT
  identity  kernel=mq_rotate_x block=32 grid=… shared=0 signs_bytes=1024
  config    k=… batch=… iters=… warmup=… n_elems=…
  canonical_us           median=… p10=… p90=… mean=…
  candidate_us           median=… p10=… p90=… mean=…
  ratio_cand_over_canon  median=… p10=… p90=… mean=…   (PRIMARY paired ratio)
  aa_control_us_a        median=… p10=… p90=… mean=…
  aa_control_us_b        median=… p10=… p90=… mean=…
  ratio_aa_control       median=… p10=… p90=… mean=…
  equivalence  cand_over_canon_median_pct=… aa_spread_median_pct=…
               within_0p5pct=yes|no  indist_from_aa=yes|no
  finite_ok  yes
  pre_d2_core_ok  yes
  checksum  <hex>
  samples_begin
  # i canon_us cand_us ratio_ab aa_a_us aa_b_us ratio_aa
  <one TSV row per timed pair>
  samples_end

  Times are microseconds per kernel launch. Paired ratios are always
  cand/canon (AB) or A2/A1 (AA) after undoing AB/BA randomization.
  Equivalence decision uses the MEDIAN paired ratio (not mean): parent
  zero-cost claim requires within_0p5pct AND indist_from_aa
  (|median_ratio-1|*100 <= 0.5 and |cand/canon-1| <= |aa-1| at the median).
  Full per-pair samples are emitted for external bootstrap CIs.
  Untimed: cand_out[i] must equal canon_out[i]*s2_canon[j]*s2_cand[j] (pre-D2
  core identity); all outputs finite; checksum prevents DCE.
  Exits nonzero on HIP failure, bad args, malformed signs, non-finite output,
  or pre-D2 core mismatch.

BUILD
  hipcc --offload-arch=gfx1201 -O3 bench_runtime_rotate.hip -o bench_runtime_rotate
)",
        argv0, argv0, DEFAULT_K, DEFAULT_BATCH, DEFAULT_ITERS, WARMUP,
        INPUT_SEED);
    std::exit(code);
}

static uint32_t lcg_next(uint32_t& state) {
    state = (state * 1103515245u + 12345u) & 0x7fffffffu;
    return state;
}

// Engine-identical FWHT sign table (matches search_affine_signs / quantize).
static void gen_fwht_signs(uint32_t seed, float* signs, int n) {
    uint32_t state = seed;
    for (int i = 0; i < n; ++i) {
        uint32_t s = lcg_next(state);
        signs[i] = ((s >> 16) & 1u) ? 1.0f : -1.0f;
    }
}

static void gen_input(uint32_t seed, float* x, size_t n) {
    uint32_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        uint32_t s = lcg_next(state);
        const float u = (float)((s >> 8) & 0xffffu) * (1.0f / 65535.0f);
        x[i] = u * 2.0f - 1.0f;
    }
}

static std::vector<float> load_signs(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open CANDIDATE.signs %s\n", path);
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        die("fseek CANDIDATE.signs failed");
    }
    long bytes = std::ftell(f);
    if (bytes < 0) {
        std::fclose(f);
        die("ftell CANDIDATE.signs failed");
    }
    if (bytes != (long)(G * (int)sizeof(float))) {
        std::fprintf(stderr,
                     "error: CANDIDATE.signs size %ld, expected %d bytes\n",
                     bytes, G * (int)sizeof(float));
        std::fclose(f);
        std::exit(1);
    }
    if (std::fseek(f, 0, SEEK_SET) != 0) {
        std::fclose(f);
        die("fseek CANDIDATE.signs rewind failed");
    }
    std::vector<float> s((size_t)G);
    if (std::fread(s.data(), sizeof(float), (size_t)G, f) != (size_t)G) {
        std::fclose(f);
        die("short read on CANDIDATE.signs");
    }
    std::fclose(f);
    for (int i = 0; i < G; ++i) {
        if (!(s[(size_t)i] == 1.0f || s[(size_t)i] == -1.0f)) {
            std::fprintf(stderr, "error: CANDIDATE.signs[%d]=%g not ±1.0f\n", i,
                         (double)s[(size_t)i]);
            std::exit(1);
        }
    }
    return s;
}

static int parse_positive_int(const char* s, const char* name) {
    if (!s || !*s) {
        std::fprintf(stderr, "error: empty %s\n", name);
        std::exit(1);
    }
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        std::fprintf(stderr, "error: invalid %s '%s'\n", name, s);
        std::exit(1);
    }
    if (v < 1 || v > 0x7fffffffL) {
        std::fprintf(stderr, "error: %s out of range: %ld\n", name, v);
        std::exit(1);
    }
    return (int)v;
}

static float percentile_sorted(const std::vector<float>& sorted, double p) {
    if (sorted.empty()) return 0.0f;
    if (sorted.size() == 1) return sorted[0];
    const double idx = p * (double)(sorted.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - (double)lo;
    return (float)((1.0 - frac) * (double)sorted[lo] +
                   frac * (double)sorted[hi]);
}

struct Stats {
    float median;
    float p10;
    float p90;
    float mean;
};

static Stats compute_stats(std::vector<float> v) {
    if (v.empty()) die("internal: empty timing vector");
    double sum = 0.0;
    for (float x : v) sum += (double)x;
    const float mean = (float)(sum / (double)v.size());
    std::sort(v.begin(), v.end());
    Stats st;
    st.median = percentile_sorted(v, 0.50);
    st.p10 = percentile_sorted(v, 0.10);
    st.p90 = percentile_sorted(v, 0.90);
    st.mean = mean;
    return st;
}

static void print_stats(const char* label, const Stats& s) {
    std::printf("%s  median=%.6f p10=%.6f p90=%.6f mean=%.6f\n", label,
                (double)s.median, (double)s.p10, (double)s.p90, (double)s.mean);
}

// Single production entry point — never a second kernel binary.
static void launch_rotate(const float* d_x, float* d_y, const float* d_s1,
                          const float* d_s2, int K, int batch) {
    const int groups = K / G;
    const int grid = groups * batch;
    // kernarg order matches production: x_in, x_out, signs1, signs2, K
    hipLaunchKernelGGL(mq_rotate_x, dim3(grid), dim3(32), 0, 0, d_x, d_y, d_s1,
                       d_s2, K);
    CK(hipGetLastError());
}

// Time one launch; returns microseconds. Caller must not upload inside.
static float time_launch_us(hipEvent_t ev_start, hipEvent_t ev_stop,
                            const float* d_x, float* d_y, const float* d_s1,
                            const float* d_s2, int K, int batch) {
    CK(hipEventRecord(ev_start, 0));
    launch_rotate(d_x, d_y, d_s1, d_s2, K, batch);
    CK(hipEventRecord(ev_stop, 0));
    CK(hipEventSynchronize(ev_stop));
    float ms = 0.0f;
    CK(hipEventElapsedTime(&ms, ev_start, ev_stop));
    const float us = ms * 1000.0f;
    if (!(us > 0.0f) || !std::isfinite(us)) {
        std::fprintf(stderr, "error: non-positive or non-finite timing %g us\n",
                     (double)us);
        std::exit(1);
    }
    return us;
}

static uint64_t checksum_f32(const float* data, size_t n) {
    uint64_t h = 14695981039346656037ull; // FNV-1a 64
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &data[i], sizeof(bits));
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t)((bits >> (8 * b)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) usage(argv[0], 1);
    if (std::strcmp(argv[1], "--help") == 0 ||
        std::strcmp(argv[1], "-h") == 0) {
        usage(argv[0], 0);
    }
    if (argc > 5) {
        std::fprintf(stderr, "error: too many arguments\n");
        usage(argv[0], 1);
    }

    const char* signs_path = argv[1];
    const int K = (argc >= 3) ? parse_positive_int(argv[2], "K") : DEFAULT_K;
    const int batch =
        (argc >= 4) ? parse_positive_int(argv[3], "BATCH") : DEFAULT_BATCH;
    const int iters =
        (argc >= 5) ? parse_positive_int(argv[4], "ITERS") : DEFAULT_ITERS;

    if (K % G != 0) {
        std::fprintf(stderr, "error: K=%d is not a multiple of %d\n", K, G);
        std::exit(1);
    }
    if ((int64_t)batch * (int64_t)(K / G) > (int64_t)0x7fffffff) {
        die("grid dimension (K/256)*BATCH overflows int");
    }
    const size_t n_elems = (size_t)batch * (size_t)K;
    if (n_elems == 0 || n_elems / (size_t)K != (size_t)batch) {
        die("n_elems overflow");
    }
    const int grid = (K / G) * batch;

    std::vector<float> signs1(G), signs2_canon(G);
    gen_fwht_signs(D1_SEED, signs1.data(), G);
    gen_fwht_signs(D2_CANON_SEED, signs2_canon.data(), G);
    std::vector<float> signs2_cand = load_signs(signs_path);

    std::vector<float> h_x(n_elems);
    gen_input(INPUT_SEED, h_x.data(), n_elems);

    float *d_x = nullptr, *d_y_a = nullptr, *d_y_b = nullptr;
    float *d_s1 = nullptr, *d_s2_canon = nullptr, *d_s2_cand = nullptr;
    const size_t bytes_x = n_elems * sizeof(float);
    const size_t bytes_s = (size_t)G * sizeof(float);

    CK(hipMalloc(&d_x, bytes_x));
    CK(hipMalloc(&d_y_a, bytes_x));
    CK(hipMalloc(&d_y_b, bytes_x));
    CK(hipMalloc(&d_s1, bytes_s));
    CK(hipMalloc(&d_s2_canon, bytes_s));
    CK(hipMalloc(&d_s2_cand, bytes_s));

    // All H2D uploads untimed.
    CK(hipMemcpy(d_x, h_x.data(), bytes_x, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_s1, signs1.data(), bytes_s, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_s2_canon, signs2_canon.data(), bytes_s,
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(d_s2_cand, signs2_cand.data(), bytes_s, hipMemcpyHostToDevice));
    CK(hipMemset(d_y_a, 0, bytes_x));
    CK(hipMemset(d_y_b, 0, bytes_x));
    CK(hipDeviceSynchronize());

    // Warmup both arms and both output buffers.
    for (int i = 0; i < WARMUP; ++i) {
        launch_rotate(d_x, d_y_a, d_s1, d_s2_canon, K, batch);
        launch_rotate(d_x, d_y_b, d_s1, d_s2_cand, K, batch);
    }
    CK(hipDeviceSynchronize());

    hipEvent_t ev_start, ev_stop;
    CK(hipEventCreate(&ev_start));
    CK(hipEventCreate(&ev_stop));

    std::vector<float> t_canon((size_t)iters);
    std::vector<float> t_cand((size_t)iters);
    std::vector<float> ratio_ab((size_t)iters);
    std::vector<float> t_aa_a((size_t)iters);
    std::vector<float> t_aa_b((size_t)iters);
    std::vector<float> ratio_aa((size_t)iters);

    uint32_t order_state = PAIR_ORDER_SEED;

    for (int i = 0; i < iters; ++i) {
        // Randomized AB / BA order for the candidate comparison pair.
        const bool cand_first = (lcg_next(order_state) & 1u) != 0u;
        float us_c = 0.0f, us_d = 0.0f;
        if (!cand_first) {
            us_c = time_launch_us(ev_start, ev_stop, d_x, d_y_a, d_s1,
                                  d_s2_canon, K, batch);
            us_d = time_launch_us(ev_start, ev_stop, d_x, d_y_b, d_s1,
                                  d_s2_cand, K, batch);
        } else {
            us_d = time_launch_us(ev_start, ev_stop, d_x, d_y_b, d_s1,
                                  d_s2_cand, K, batch);
            us_c = time_launch_us(ev_start, ev_stop, d_x, d_y_a, d_s1,
                                  d_s2_canon, K, batch);
        }
        t_canon[(size_t)i] = us_c;
        t_cand[(size_t)i] = us_d;
        ratio_ab[(size_t)i] = us_d / us_c;

        // A/A negative control: identical signs2 pointer both times.
        // Still use two output buffers so write traffic matches AB arm.
        const bool swap_aa = (lcg_next(order_state) & 1u) != 0u;
        float us_a1, us_a2;
        if (!swap_aa) {
            us_a1 = time_launch_us(ev_start, ev_stop, d_x, d_y_a, d_s1,
                                   d_s2_canon, K, batch);
            us_a2 = time_launch_us(ev_start, ev_stop, d_x, d_y_b, d_s1,
                                   d_s2_canon, K, batch);
        } else {
            us_a2 = time_launch_us(ev_start, ev_stop, d_x, d_y_b, d_s1,
                                   d_s2_canon, K, batch);
            us_a1 = time_launch_us(ev_start, ev_stop, d_x, d_y_a, d_s1,
                                   d_s2_canon, K, batch);
        }
        t_aa_a[(size_t)i] = us_a1;
        t_aa_b[(size_t)i] = us_a2;
        ratio_aa[(size_t)i] = us_a2 / us_a1;
    }

    // Final untimed correctness launches into dedicated buffers.
    CK(hipMemset(d_y_a, 0, bytes_x));
    CK(hipMemset(d_y_b, 0, bytes_x));
    launch_rotate(d_x, d_y_a, d_s1, d_s2_canon, K, batch);
    launch_rotate(d_x, d_y_b, d_s1, d_s2_cand, K, batch);
    CK(hipDeviceSynchronize());

    std::vector<float> h_y_canon(n_elems), h_y_cand(n_elems);
    CK(hipMemcpy(h_y_canon.data(), d_y_a, bytes_x, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h_y_cand.data(), d_y_b, bytes_x, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < n_elems; ++i) {
        if (!std::isfinite(h_y_canon[i])) {
            std::fprintf(stderr,
                         "error: non-finite canonical output at element %zu\n",
                         i);
            std::exit(1);
        }
        if (!std::isfinite(h_y_cand[i])) {
            std::fprintf(stderr,
                         "error: non-finite candidate output at element %zu\n",
                         i);
            std::exit(1);
        }
        // pre-D2 core identity: out = core * signs2, signs2 = ±1
        // => cand = canon * s2_canon * s2_cand  (exact IEEE sign-bit toggles)
        const size_t j = i % (size_t)G;
        const float expect =
            h_y_canon[i] * signs2_canon[j] * signs2_cand[j];
        uint32_t got_bits = 0, exp_bits = 0;
        std::memcpy(&got_bits, &h_y_cand[i], sizeof(got_bits));
        std::memcpy(&exp_bits, &expect, sizeof(exp_bits));
        if (got_bits != exp_bits) {
            std::fprintf(stderr,
                         "error: pre-D2 core mismatch at element %zu: "
                         "cand=%g expect=%g (canon=%g s2c=%g s2d=%g)\n",
                         i, (double)h_y_cand[i], (double)expect,
                         (double)h_y_canon[i], (double)signs2_canon[j],
                         (double)signs2_cand[j]);
            std::exit(1);
        }
    }

    const uint64_t csum = checksum_f32(h_y_canon.data(), n_elems) ^
                          (checksum_f32(h_y_cand.data(), n_elems) << 1);

    const Stats sc = compute_stats(t_canon);
    const Stats sd = compute_stats(t_cand);
    const Stats sr = compute_stats(ratio_ab);
    const Stats sa1 = compute_stats(t_aa_a);
    const Stats sa2 = compute_stats(t_aa_b);
    const Stats sra = compute_stats(ratio_aa);

    const float ab_pct =
        (float)(std::fabs((double)sr.median - 1.0) * 100.0);
    const float aa_pct =
        (float)(std::fabs((double)sra.median - 1.0) * 100.0);
    const bool within = ab_pct <= ZERO_COST_PCT;
    const bool indist = ab_pct <= aa_pct + 1e-12f;

    std::printf(
        "identity  kernel=mq_rotate_x block=32 grid=%d shared=0 signs_bytes=%d "
        "kernarg=x,x_out,signs1,signs2,K\n",
        grid, G * (int)sizeof(float));
    std::printf("config  k=%d batch=%d iters=%d warmup=%d n_elems=%zu\n", K,
                batch, iters, WARMUP, n_elems);
    print_stats("canonical_us", sc);
    print_stats("candidate_us", sd);
    print_stats("ratio_cand_over_canon", sr);
    print_stats("aa_control_us_a", sa1);
    print_stats("aa_control_us_b", sa2);
    print_stats("ratio_aa_control", sra);
    std::printf(
        "equivalence  cand_over_canon_median_pct=%.6f aa_spread_median_pct=%.6f "
        "within_0p5pct=%s  indist_from_aa=%s  "
        "paired_ratio_median=%.8f  aa_ratio_median=%.8f\n",
        (double)ab_pct, (double)aa_pct, within ? "yes" : "no",
        indist ? "yes" : "no", (double)sr.median, (double)sra.median);
    std::printf("finite_ok  yes\n");
    std::printf("pre_d2_core_ok  yes\n");
    std::printf("checksum  %016llx\n", (unsigned long long)csum);

    // Every paired sample for parent-side bootstrap CI (no in-binary bootstrap).
    std::printf("samples_begin\n");
    std::printf("# i canon_us cand_us ratio_ab aa_a_us aa_b_us ratio_aa\n");
    for (int i = 0; i < iters; ++i) {
        std::printf("%d %.6f %.6f %.8f %.6f %.6f %.8f\n", i,
                    (double)t_canon[(size_t)i], (double)t_cand[(size_t)i],
                    (double)ratio_ab[(size_t)i], (double)t_aa_a[(size_t)i],
                    (double)t_aa_b[(size_t)i], (double)ratio_aa[(size_t)i]);
    }
    std::printf("samples_end\n");

    CK(hipEventDestroy(ev_start));
    CK(hipEventDestroy(ev_stop));
    CK(hipFree(d_x));
    CK(hipFree(d_y_a));
    CK(hipFree(d_y_b));
    CK(hipFree(d_s1));
    CK(hipFree(d_s2_canon));
    CK(hipFree(d_s2_cand));
    return 0;
}
