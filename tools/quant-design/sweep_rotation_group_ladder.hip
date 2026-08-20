// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// HIP quality microbench: independent signed-FWHT rotation scope R and affine
// quantization-group G ladder on a raw Qwen3.8 tensor fixture.
//
// Signed FWHT is D2 * H * D1 / sqrt(R) with LCG sign seeds 42 / 1042 (engine).
// Metrics are measured in rotated space (matching existing quant-design sweeps).
//
// Usage:
//   hipcc --offload-arch=gfx1201 -O3 sweep_rotation_group_ladder.hip -o .codeinsight+research/quant-design/sweep_rotation_group_ladder
//   .codeinsight+research/quant-design/sweep_rotation_group_ladder <raw-f32-path> [rows] [K]
//
// Default intended shape: first 4096 rows of
//   model.language_model.layers.20.mlp.down_proj.weight, K=17408.
// Input is contiguous row-major raw F32. No embedded $HOME path.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                          \
    do {                                                               \
        hipError_t e_ = (x);                                           \
        if (e_ != hipSuccess) {                                        \
            std::fprintf(stderr, "HIP error %s at %d\n",               \
                         hipGetErrorString(e_), __LINE__);             \
            std::exit(1);                                              \
        }                                                              \
    } while (0)

static constexpr int SEED1 = 42;
static constexpr int SEED2 = 1042;
static constexpr int CHUNK = 256;
static constexpr int DEFAULT_ROWS = 4096;
static constexpr int DEFAULT_K = 17408;

// ── Host sign LCG (matches engine gen_fwht_signs) ──────────────────────────
static std::vector<float> gen_fwht_signs(unsigned seed, int n) {
    std::vector<float> signs((size_t)n);
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = (state * 1103515245u + 12345u) & 0x7fffffffu;
        signs[(size_t)i] = ((state >> 16) & 1u) ? 1.0f : -1.0f;
    }
    return signs;
}

// ── GPU signed FWHT forward: out = D2 * H * D1(in) / sqrt(R) ───────────────
// One block per R-group. Dynamic shared memory holds the R-vector.
// blockDim.x is a power of two <= R (typically min(R, 1024)).
__global__ void fwht_forward_r(const float* __restrict__ in,
                               float* __restrict__ out,
                               const float* __restrict__ s1,
                               const float* __restrict__ s2,
                               int R,
                               int n_groups) {
    extern __shared__ float sh[];
    const int g = (int)blockIdx.x;
    if (g >= n_groups)
        return;
    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const float* src = in + (size_t)g * (size_t)R;
    float* dst = out + (size_t)g * (size_t)R;

    for (int i = tid; i < R; i += nthreads)
        sh[i] = src[i] * s1[i];
    __syncthreads();

    for (int stride = 1; stride < R; stride <<= 1) {
        // R/2 independent butterflies; each owns one (base, j) pair.
        for (int t = tid; t < (R >> 1); t += nthreads) {
            const int pair = t / stride;
            const int j = t % stride;
            const int base = pair * (stride << 1);
            const float a = sh[base + j];
            const float b = sh[base + j + stride];
            sh[base + j] = a + b;
            sh[base + j + stride] = a - b;
        }
        __syncthreads();
    }

    const float inv = rsqrtf((float)R);
    for (int i = tid; i < R; i += nthreads)
        dst[i] = sh[i] * inv * s2[i];
}

// ── GPU affine quantize + reconstruct over G-groups ────────────────────────
// Writes reconstructed values into `recon` (same length as `rot`).
// fp16_header: 1 → round scale+zero through __half; 0 → keep f32.
__global__ void affine_recon_g(const float* __restrict__ rot,
                               float* __restrict__ recon,
                               int n_groups,
                               int G,
                               int bits,
                               int fp16_header) {
    // One block covers many G-groups; 256 threads, each walks groups strided.
    // Within a group, threads 0..G-1 (or a subset) cooperatively min/max via shared.
    // Simpler: each block owns one G-group when G<=256; for generality use
    // blockIdx over groups and threadIdx for elements inside the group.
    const int grp = (int)blockIdx.x;
    if (grp >= n_groups)
        return;
    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const float* src = rot + (size_t)grp * (size_t)G;
    float* dst = recon + (size_t)grp * (size_t)G;

    // Parallel min/max reduction over the G-group.
    extern __shared__ float red[];
    float* sh_min = red;
    float* sh_max = red + nthreads;

    float lo = 1.0e30f;
    float hi = -1.0e30f;
    for (int i = tid; i < G; i += nthreads) {
        const float v = src[i];
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
    }
    sh_min[tid] = lo;
    sh_max[tid] = hi;
    __syncthreads();
    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sh_min[tid] = fminf(sh_min[tid], sh_min[tid + s]);
            sh_max[tid] = fmaxf(sh_max[tid], sh_max[tid + s]);
        }
        __syncthreads();
    }
    lo = sh_min[0];
    hi = sh_max[0];

    const int nlev = 1 << bits;
    const float steps = (float)(nlev - 1);
    float scale = (hi > lo) ? (hi - lo) / steps : 0.0f;
    float zero = lo;
    if (fp16_header) {
        scale = __half2float(__float2half(scale));
        zero = __half2float(__float2half(zero));
    }

    for (int i = tid; i < G; i += nthreads) {
        const float w = src[i];
        const float q = (scale > 0.0f)
            ? fminf(fmaxf(rintf((w - zero) / scale), 0.0f), steps)
            : 0.0f;
        dst[i] = q * scale + zero;
    }
}

struct Metrics {
    double mse;
    double p99_mse;
    double p999_mse;
    double max_rel_mean;
};

// Host-side metrics on rotated space, fixed 256-coefficient report chunks.
static Metrics compute_metrics(const float* rot,
                               const float* recon,
                               size_t n,
                               float thr99,
                               float thr999) {
    double sse = 0.0;
    double sse99 = 0.0;
    double sse999 = 0.0;
    size_t n99 = 0;
    size_t n999 = 0;
    double max_rel_sum = 0.0;
    size_t n_chunks = n / (size_t)CHUNK;

    for (size_t i = 0; i < n; ++i) {
        const double e = (double)recon[i] - (double)rot[i];
        const double ee = e * e;
        sse += ee;
        const float aw = fabsf(rot[i]);
        if (aw >= thr99) {
            sse99 += ee;
            ++n99;
        }
        if (aw >= thr999) {
            sse999 += ee;
            ++n999;
        }
    }

    for (size_t c = 0; c < n_chunks; ++c) {
        const size_t base = c * (size_t)CHUNK;
        int best = 0;
        float best_abs = fabsf(rot[base]);
        for (int i = 1; i < CHUNK; ++i) {
            const float a = fabsf(rot[base + (size_t)i]);
            if (a > best_abs) {
                best_abs = a;
                best = i;
            }
        }
        const float w = rot[base + (size_t)best];
        const float r = recon[base + (size_t)best];
        const float denom = fmaxf(fabsf(w), 1.0e-30f);
        max_rel_sum += (double)(fabsf(r - w) / denom);
    }

    Metrics m{};
    m.mse = sse / (double)n;
    m.p99_mse = n99 ? sse99 / (double)n99 : 0.0;
    m.p999_mse = n999 ? sse999 / (double)n999 : 0.0;
    m.max_rel_mean = n_chunks ? max_rel_sum / (double)n_chunks : 0.0;
    return m;
}

// Bytes charged per 256 coefficients for a (bits, G) affine layout.
// G256 uses f32 scale+zero (8 B / group). G128/G64 use fp16 scale+zero (4 B / group).
static void byte_accounting(int bits, int G, int* payload_b, int* header_b, int* total_b) {
    *payload_b = bits * 32; // bits * 256 / 8
    const int groups_per_256 = CHUNK / G;
    const int bytes_per_group = (G == 256) ? 8 : 4; // f32 pair vs fp16 pair
    *header_b = groups_per_256 * bytes_per_group;
    *total_b = *payload_b + *header_b;
}

struct Candidate {
    int bits;
    int R;
    int G;
    const char* tag; // optional baseline label, else ""
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <raw-f32-path> [rows=%d] [K=%d]\n",
                     argv[0], DEFAULT_ROWS, DEFAULT_K);
        return 2;
    }
    const char* path = argv[1];
    const int rows = argc > 2 ? std::atoi(argv[2]) : DEFAULT_ROWS;
    const int K = argc > 3 ? std::atoi(argv[3]) : DEFAULT_K;
    if (rows <= 0 || K <= 0) {
        std::fprintf(stderr, "rows and K must be positive\n");
        return 1;
    }
    const size_t n_elem = (size_t)rows * (size_t)K;
    const size_t expect_bytes = n_elem * sizeof(float);

    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fprintf(stderr, "seek failed\n");
        return 1;
    }
    const long file_bytes = std::ftell(f);
    if (file_bytes < 0) {
        std::fprintf(stderr, "ftell failed\n");
        return 1;
    }
    std::fseek(f, 0, SEEK_SET);
    if ((size_t)file_bytes != expect_bytes) {
        std::fprintf(stderr,
                     "input size mismatch: got %ld bytes, expected %zu "
                     "(%d rows * %d K * 4)\n",
                     file_bytes, expect_bytes, rows, K);
        std::fclose(f);
        return 1;
    }
    std::vector<float> host_w(n_elem);
    if (std::fread(host_w.data(), 1, expect_bytes, f) != expect_bytes) {
        std::fprintf(stderr, "short read of weight fixture\n");
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    // Candidate ladder: R outer, then G, then bits. G must divide R; rows*K ÷ R.
    // R-outer keeps ensure_rotated at one forward FWHT per R.
    static const int R_LIST[] = {128, 256, 512, 1024};
    static const int RG[][2] = {
        {128, 64},  {128, 128},
        {256, 64},  {256, 128}, {256, 256},
        {512, 64},  {512, 128}, {512, 256},
        {1024, 64}, {1024, 128}, {1024, 256},
    };
    static const int N_RG = (int)(sizeof(RG) / sizeof(RG[0]));
    static const int BITS[] = {2, 3, 4};

    for (int i = 0; i < N_RG; ++i) {
        const int R = RG[i][0];
        const int G = RG[i][1];
        if (G > R || (R % G) != 0) {
            std::fprintf(stderr, "internal: G=%d does not divide R=%d\n", G, R);
            return 1;
        }
        if (n_elem % (size_t)R != 0) {
            std::fprintf(stderr,
                         "rows*K=%zu not divisible by R=%d\n", n_elem, R);
            return 1;
        }
    }
    if (n_elem % (size_t)CHUNK != 0) {
        std::fprintf(stderr, "rows*K=%zu not divisible by 256\n", n_elem);
        return 1;
    }

    // Build ordered candidate list (R outer so each rotation is computed once).
    std::vector<Candidate> cands;
    cands.reserve(33);
    for (int i = 0; i < N_RG; ++i) {
        const int R = RG[i][0];
        const int G = RG[i][1];
        for (int bi = 0; bi < 3; ++bi) {
            const int bits = BITS[bi];
            const char* tag = "";
            if (bits == 4 && R == 256 && G == 256)
                tag = "genesis";
            else if (bits == 4 && R == 256 && G == 128)
                tag = "V2";
            cands.push_back({bits, R, G, tag});
        }
    }
    // 3 bits × 11 (R,G) = 33
    if ((int)cands.size() != 33) {
        std::fprintf(stderr, "internal: expected 33 candidates, got %zu\n",
                     cands.size());
        return 1;
    }

    float* d_w = nullptr;
    float* d_rot = nullptr;
    float* d_recon = nullptr;
    float* d_s1 = nullptr;
    float* d_s2 = nullptr;
    CK(hipMalloc(&d_w, n_elem * sizeof(float)));
    CK(hipMalloc(&d_rot, n_elem * sizeof(float)));
    CK(hipMalloc(&d_recon, n_elem * sizeof(float)));
    CK(hipMalloc(&d_s1, 1024 * sizeof(float)));
    CK(hipMalloc(&d_s2, 1024 * sizeof(float)));
    CK(hipMemcpy(d_w, host_w.data(), n_elem * sizeof(float),
                 hipMemcpyHostToDevice));

    std::vector<float> host_rot(n_elem);
    std::vector<float> host_recon(n_elem);

    std::printf(
        "rotation×group ladder: %d rows × K=%d = %zu weights from %s\n",
        rows, K, n_elem, path);
    std::printf(
        "signed FWHT seeds %d/%d; metrics in rotated space; "
        "budget = payload+8 B/256\n",
        SEED1, SEED2);
    std::printf(
        "header: G256=f32 scale+zero (8B/256); G128=fp16×2 (8B/256); "
        "G64=fp16×4 (16B/256)\n\n");

    std::printf(
        "%-4s %5s %5s %-10s %5s %5s %5s %7s  %-12s %-12s %-12s %-12s %s\n",
        "bits", "R", "G", "label", "payB", "hdrB", "totB", "bpw",
        "overall MSE", "p99 MSE", "p99.9 MSE", "max-rel mean", "flag");
    std::printf(
        "--------------------------------------------------------------------------------------------------"
        "-----------------------------------------------\n");

    // Cache rotated host buffer + thresholds per R.
    struct RotCache {
        int R;
        bool ready;
        float thr99;
        float thr999;
    };
    RotCache cache_list[4];
    for (int i = 0; i < 4; ++i) {
        cache_list[i].R = R_LIST[i];
        cache_list[i].ready = false;
        cache_list[i].thr99 = 0.0f;
        cache_list[i].thr999 = 0.0f;
    }
    int current_R = -1;

    auto ensure_rotated = [&](int R) {
        if (current_R == R)
            return;
        const int n_groups = (int)(n_elem / (size_t)R);
        const std::vector<float> s1 = gen_fwht_signs((unsigned)SEED1, R);
        const std::vector<float> s2 = gen_fwht_signs((unsigned)SEED2, R);
        CK(hipMemcpy(d_s1, s1.data(), (size_t)R * sizeof(float),
                     hipMemcpyHostToDevice));
        CK(hipMemcpy(d_s2, s2.data(), (size_t)R * sizeof(float),
                     hipMemcpyHostToDevice));

        int tpb = R;
        if (tpb > 1024)
            tpb = 1024;
        // Prefer power-of-two thread count that divides the butterfly work.
        while (tpb > 64 && (R % tpb) != 0)
            tpb >>= 1;
        if (tpb < 32)
            tpb = 32;
        if (tpb > R)
            tpb = R;

        const size_t shmem = (size_t)R * sizeof(float);
        hipLaunchKernelGGL(fwht_forward_r, dim3(n_groups), dim3(tpb), shmem, 0,
                           d_w, d_rot, d_s1, d_s2, R, n_groups);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(host_rot.data(), d_rot, n_elem * sizeof(float),
                     hipMemcpyDeviceToHost));

        // Per-R magnitude percentiles for tail thresholds.
        std::vector<float> abs_vals(n_elem);
        for (size_t i = 0; i < n_elem; ++i)
            abs_vals[i] = fabsf(host_rot[i]);
        std::sort(abs_vals.begin(), abs_vals.end());
        auto pct = [&](double p) -> float {
            if (n_elem == 0)
                return 0.0f;
            const double idx = p * (double)(n_elem - 1);
            const size_t lo = (size_t)idx;
            const size_t hi = std::min(lo + 1, n_elem - 1);
            const double frac = idx - (double)lo;
            return (float)((1.0 - frac) * (double)abs_vals[lo] +
                           frac * (double)abs_vals[hi]);
        };
        const float thr99 = pct(0.99);
        const float thr999 = pct(0.999);

        for (int i = 0; i < 4; ++i) {
            if (cache_list[i].R == R) {
                cache_list[i].ready = true;
                cache_list[i].thr99 = thr99;
                cache_list[i].thr999 = thr999;
            }
        }
        current_R = R;

        std::printf("# R=%-4d groups=%-8d |w| p99=%.6e  p99.9=%.6e\n",
                    R, n_groups, thr99, thr999);
    };

    auto thr_for = [&](int R, float* t99, float* t999) {
        for (int i = 0; i < 4; ++i) {
            if (cache_list[i].R == R && cache_list[i].ready) {
                *t99 = cache_list[i].thr99;
                *t999 = cache_list[i].thr999;
                return;
            }
        }
        *t99 = 0.0f;
        *t999 = 0.0f;
    };

    for (const Candidate& c : cands) {
        ensure_rotated(c.R);
        float thr99 = 0.0f, thr999 = 0.0f;
        thr_for(c.R, &thr99, &thr999);

        const int n_g = (int)(n_elem / (size_t)c.G);
        const int fp16_header = (c.G == 256) ? 0 : 1;
        int tpb = c.G;
        if (tpb > 256)
            tpb = 256;
        // Power-of-two for clean reduction; clamp to >=1.
        int red_tpb = 1;
        while ((red_tpb << 1) <= tpb)
            red_tpb <<= 1;
        tpb = red_tpb;
        const size_t shmem = (size_t)tpb * 2 * sizeof(float);
        hipLaunchKernelGGL(affine_recon_g, dim3(n_g), dim3(tpb), shmem, 0,
                           d_rot, d_recon, n_g, c.G, c.bits, fp16_header);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(host_recon.data(), d_recon, n_elem * sizeof(float),
                     hipMemcpyDeviceToHost));

        const Metrics m = compute_metrics(host_rot.data(), host_recon.data(),
                                          n_elem, thr99, thr999);

        int pay_b = 0, hdr_b = 0, tot_b = 0;
        byte_accounting(c.bits, c.G, &pay_b, &hdr_b, &tot_b);
        const int budget = pay_b + 8;
        const double bpw = tot_b * 8.0 / (double)CHUNK;
        const char* flag = (tot_b > budget) ? "[OVER budget]" : "";
        char label[16];
        if (c.tag[0] != '\0')
            std::snprintf(label, sizeof(label), "%s", c.tag);
        else
            label[0] = '\0';

        std::printf(
            "%-4d %5d %5d %-10s %5d %5d %5d %7.4f  %.4e   %.4e   %.4e   %.4e  %s\n",
            c.bits, c.R, c.G, label, pay_b, hdr_b, tot_b, bpw,
            m.mse, m.p99_mse, m.p999_mse, m.max_rel_mean, flag);
    }

    std::printf("\nBaselines: R256/G256 genesis (4b f32 hdr) and R256/G128 V2 "
                "(4b fp16 hdr) labelled above.\n");

    CK(hipFree(d_s2));
    CK(hipFree(d_s1));
    CK(hipFree(d_recon));
    CK(hipFree(d_rot));
    CK(hipFree(d_w));
    return 0;
}
