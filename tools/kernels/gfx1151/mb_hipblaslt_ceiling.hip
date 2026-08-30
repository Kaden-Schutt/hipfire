// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mb-hipblaslt-ceiling-gfx1151-v1-20260830
//
// QUESTION: what sustained I8I8 and F16 GEMM TOPS does hipBLASLt actually
// deliver on gfx1151 FFN shapes — is the silicon-ceiling >= 30 TOPS I8
// (2x headroom over MMQ ~15.5 TOPS)?
//
// This is a library-ceiling probe, not a production-kernel question.
// hipBLASLt is dlopened (HIPBLASLT_LIB / ROCM_PATH / soname fallback).
// No custom device kernels.
//
// Shapes (M,N,K) = FFN gate_up / down at chunk 512 and single-chunk 2048:
//   (17408, 512, 5120), (17408, 2048, 5120),
//   (5120, 512, 17408), (5120, 2048, 17408)
// Dtypes: i8×i8→i32 (HIPBLAS_COMPUTE_32I) and f16×f16→f32 (HIPBLAS_COMPUTE_32F).
// Layout: column-major, opN×opN, D = 1*A*B + 0*C.
// Algo: hipblasLtMatmulAlgoGetHeuristic, time top-N, keep fastest.
//
// Correctness: 128x64x128 CPU reference (I8 exact; F16 rel_l2 <= 5e-3) before
// full-shape timing. No I8 algo is a FINDING, not a failure.
//
// Build (host-only; fat gfx11 image for hipx ROCm 7.15):
//   /opt/rocm/bin/hipcc -O3 -std=c++17 -ldl \
//       --offload-arch=gfx1100 --offload-arch=gfx1151 \
//       tools/kernels/gfx1151/mb_hipblaslt_ceiling.hip -o mb_hipblaslt_ceiling
// Local compile-check (any arch; do not time on gfx1201):
//   /opt/rocm/core/bin/hipcc -O3 -std=c++17 -ldl \
//       --offload-arch=gfx1201 \
//       tools/kernels/gfx1151/mb_hipblaslt_ceiling.hip -o mb_hipblaslt_ceiling
// Run (pin gfx1151 as the only visible device, then arch-match):
//   HIP_VISIBLE_DEVICES=1 HIPBLASLT_LIB=/opt/rocm/lib/libhipblaslt.so \
//       ./mb_hipblaslt_ceiling --arch gfx1151
//   Defaults: warmup=20 timed=100 heuristic_n=16 workspace=256 MiB
// Env: HIPBLASLT_LIB, ROCM_PATH, HIPBLASLT_WORKSPACE_MB

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

static constexpr const char* kBenchId = "mb-hipblaslt-ceiling-gfx1151-v1-20260830";
static constexpr int kDefWarmup = 20;
static constexpr int kDefTimed = 100;
static constexpr int kDefHeurN = 16;
static constexpr int kPickWarmup = 3;
static constexpr int kPickTimed = 8;
static constexpr int kHostRefM = 128;
static constexpr int kHostRefN = 64;
static constexpr int kHostRefK = 128;

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ \
    std::fprintf(stderr,"HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1);} } while(0)

// ---------------------------------------------------------------------------
// hipBLASLt C ABI (dlopen). Values match hipblas-common / hipblaslt.h.
// ---------------------------------------------------------------------------
using hipblasStatus_t = int;
using hipblasLtHandle_t = void*;
using hipblasLtMatmulDesc_t = void*;
using hipblasLtMatrixLayout_t = void*;
using hipblasLtMatmulPreference_t = void*;
using hipDataType_i = int;
using hipblasComputeType_i = int;
using hipblasLtMatmulDescAttributes_i = int;
using hipblasLtMatmulPreferenceAttributes_i = int;
using hipblasOperation_i = int;

static constexpr hipblasStatus_t kLtSuccess = 0;
static constexpr hipblasStatus_t kLtNotSupported = 7;
static constexpr hipblasComputeType_i kCompute32F = 2;
static constexpr hipblasComputeType_i kCompute32I = 9;
static constexpr hipDataType_i kR32F = 0;
static constexpr hipDataType_i kR16F = 2;
static constexpr hipDataType_i kR8I = 3;
static constexpr hipDataType_i kR32I = 10;
static constexpr hipblasOperation_i kOpN = 111;
static constexpr int kDescTransA = 0;
static constexpr int kDescTransB = 1;
static constexpr int kPrefMaxWorkspace = 1;

struct hipblasLtMatmulAlgo_t {
    uint8_t data[16];
    size_t max_workspace_bytes;
};
struct hipblasLtMatmulHeuristicResult_t {
    hipblasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    hipblasStatus_t state;
    float wavesCount;
    int reserved[4];
};

using FnLtCreate = hipblasStatus_t (*)(hipblasLtHandle_t*);
using FnLtDestroy = hipblasStatus_t (*)(hipblasLtHandle_t);
using FnLtGetVersion = hipblasStatus_t (*)(hipblasLtHandle_t, int*);
using FnLayoutCreate = hipblasStatus_t (*)(hipblasLtMatrixLayout_t*, hipDataType_i, uint64_t, uint64_t, int64_t);
using FnLayoutDestroy = hipblasStatus_t (*)(hipblasLtMatrixLayout_t);
using FnDescCreate = hipblasStatus_t (*)(hipblasLtMatmulDesc_t*, hipblasComputeType_i, hipDataType_i);
using FnDescDestroy = hipblasStatus_t (*)(hipblasLtMatmulDesc_t);
using FnDescSet = hipblasStatus_t (*)(hipblasLtMatmulDesc_t, hipblasLtMatmulDescAttributes_i, const void*, size_t);
using FnPrefCreate = hipblasStatus_t (*)(hipblasLtMatmulPreference_t*);
using FnPrefDestroy = hipblasStatus_t (*)(hipblasLtMatmulPreference_t);
using FnPrefSet = hipblasStatus_t (*)(hipblasLtMatmulPreference_t, hipblasLtMatmulPreferenceAttributes_i, const void*, size_t);
using FnHeur = hipblasStatus_t (*)(hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t,
                                   hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
                                   hipblasLtMatmulPreference_t, int, hipblasLtMatmulHeuristicResult_t*, int*);
using FnMatmul = hipblasStatus_t (*)(hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void*,
                                     const void*, hipblasLtMatrixLayout_t, const void*, hipblasLtMatrixLayout_t,
                                     const void*, const void*, hipblasLtMatrixLayout_t, void*, hipblasLtMatrixLayout_t,
                                     const hipblasLtMatmulAlgo_t*, void*, size_t, hipStream_t);

struct Lt {
    void* lib = nullptr;
    FnLtCreate create = nullptr;
    FnLtDestroy destroy = nullptr;
    FnLtGetVersion get_version = nullptr;
    FnLayoutCreate layout_create = nullptr;
    FnLayoutDestroy layout_destroy = nullptr;
    FnDescCreate desc_create = nullptr;
    FnDescDestroy desc_destroy = nullptr;
    FnDescSet desc_set = nullptr;
    FnPrefCreate pref_create = nullptr;
    FnPrefDestroy pref_destroy = nullptr;
    FnPrefSet pref_set = nullptr;
    FnHeur heur = nullptr;
    FnMatmul matmul = nullptr;
};

static const char* lt_status_str(hipblasStatus_t s) {
    switch (s) {
        case 0: return "SUCCESS";
        case 1: return "NOT_INITIALIZED";
        case 2: return "ALLOC_FAILED";
        case 3: return "INVALID_VALUE";
        case 5: return "EXECUTION_FAILED";
        case 6: return "INTERNAL_ERROR";
        case 7: return "NOT_SUPPORTED";
        case 8: return "ARCH_MISMATCH";
        default: return "OTHER";
    }
}

static void* load_sym(void* lib, const char* name, bool required) {
    dlerror();
    void* p = dlsym(lib, name);
    const char* err = dlerror();
    if (!p || err) {
        if (required) {
            std::fprintf(stderr, "dlsym %s failed: %s\n", name, err ? err : "null");
            std::exit(1);
        }
        return nullptr;
    }
    return p;
}

static std::vector<std::string> lib_candidates() {
    std::vector<std::string> out;
    if (const char* e = std::getenv("HIPBLASLT_LIB")) {
        if (e[0]) out.emplace_back(e);
    }
    const char* roots[] = {nullptr, nullptr, "/opt/rocm", "/opt/rocm/lib", "/opt/rocm/hipblaslt"};
    if (const char* r = std::getenv("ROCM_PATH")) roots[0] = r;
    if (const char* r = std::getenv("HIPFIRE_ROCM_PATH")) roots[1] = r;
    const char* names[] = {
        "lib/libhipblaslt.so", "lib/libhipblaslt.so.0", "lib64/libhipblaslt.so",
        "libhipblaslt.so", "libhipblaslt.so.0"
    };
    for (const char* root : roots) {
        if (!root || !root[0]) continue;
        for (const char* n : names) {
            std::string p = root;
            if (p.back() == '/') p.pop_back();
            p.push_back('/');
            p += n;
            out.push_back(p);
        }
    }
    out.emplace_back("libhipblaslt.so");
    out.emplace_back("libhipblaslt.so.0");
    out.emplace_back("/opt/rocm/lib/libhipblaslt.so");
    out.emplace_back("/opt/rocm/lib/libhipblaslt.so.0");
    return out;
}

static Lt load_lt(std::string* used_path) {
    Lt lt{};
    auto cands = lib_candidates();
    for (const auto& p : cands) {
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (h) {
            lt.lib = h;
            *used_path = p;
            break;
        }
    }
    if (!lt.lib) {
        std::fprintf(stderr, "dlopen libhipblaslt.so failed. Tried:\n");
        for (const auto& p : cands) std::fprintf(stderr, "  %s\n", p.c_str());
        std::fprintf(stderr, "last dlerror: %s\n", dlerror());
        std::exit(1);
    }
    lt.create = (FnLtCreate)load_sym(lt.lib, "hipblasLtCreate", true);
    lt.destroy = (FnLtDestroy)load_sym(lt.lib, "hipblasLtDestroy", true);
    lt.get_version = (FnLtGetVersion)load_sym(lt.lib, "hipblasLtGetVersion", false);
    lt.layout_create = (FnLayoutCreate)load_sym(lt.lib, "hipblasLtMatrixLayoutCreate", true);
    lt.layout_destroy = (FnLayoutDestroy)load_sym(lt.lib, "hipblasLtMatrixLayoutDestroy", true);
    lt.desc_create = (FnDescCreate)load_sym(lt.lib, "hipblasLtMatmulDescCreate", true);
    lt.desc_destroy = (FnDescDestroy)load_sym(lt.lib, "hipblasLtMatmulDescDestroy", true);
    lt.desc_set = (FnDescSet)load_sym(lt.lib, "hipblasLtMatmulDescSetAttribute", true);
    lt.pref_create = (FnPrefCreate)load_sym(lt.lib, "hipblasLtMatmulPreferenceCreate", true);
    lt.pref_destroy = (FnPrefDestroy)load_sym(lt.lib, "hipblasLtMatmulPreferenceDestroy", true);
    lt.pref_set = (FnPrefSet)load_sym(lt.lib, "hipblasLtMatmulPreferenceSetAttribute", true);
    lt.heur = (FnHeur)load_sym(lt.lib, "hipblasLtMatmulAlgoGetHeuristic", true);
    lt.matmul = (FnMatmul)load_sym(lt.lib, "hipblasLtMatmul", true);
    return lt;
}

static uint64_t lcg_u64(uint64_t* s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}
static float lcg_unit(uint64_t* s) { return (float)(lcg_u64(s) >> 40) * (1.f / (float)(1u << 24)); }
static float lcg_finite(uint64_t* s, float lo, float hi) { return lo + (hi - lo) * lcg_unit(s); }

static uint16_t f32_to_f16_bits(float v) {
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
static float f16_bits_to_f32(uint16_t b) {
    const uint32_t sign = (uint32_t)(b & 0x8000u) << 16;
    uint32_t exp = (b >> 10) & 0x1Fu, mant = b & 0x3FFu, out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400u));
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) out = sign | 0x7F800000u | (mant << 13);
    else out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}

enum DType { kI8I32, kF16F32 };

static const char* dtype_name(DType d) { return d == kI8I32 ? "i8i8_i32" : "f16f16_f32"; }
static size_t ab_elem(DType d) { return d == kI8I32 ? 1 : 2; }
static size_t cd_elem(DType d) { return d == kI8I32 ? 4 : 4; }

static uint32_t algo_id(const hipblasLtMatmulAlgo_t& a) {
    uint32_t v = 0;
    std::memcpy(&v, a.data, 4);
    return v;
}
static void algo_hex(const hipblasLtMatmulAlgo_t& a, char out[33]) {
    static const char* hexd = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        out[2 * i] = hexd[(a.data[i] >> 4) & 0xF];
        out[2 * i + 1] = hexd[a.data[i] & 0xF];
    }
    out[32] = 0;
}

struct Matmul {
    const Lt* lt = nullptr;
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    DType dtype = kF16F32;
    int m = 0, n = 0, k = 0;
};

static void matmul_destroy(Matmul* g) {
    if (!g->lt) return;
    if (g->pref) g->lt->pref_destroy(g->pref);
    if (g->c) g->lt->layout_destroy(g->c);
    if (g->b) g->lt->layout_destroy(g->b);
    if (g->a) g->lt->layout_destroy(g->a);
    if (g->desc) g->lt->desc_destroy(g->desc);
    g->pref = nullptr; g->c = nullptr; g->b = nullptr; g->a = nullptr; g->desc = nullptr;
}

static bool matmul_init(Matmul* g, const Lt* lt, hipblasLtHandle_t handle, DType dtype,
                        int m, int n, int k, uint64_t max_ws) {
    matmul_destroy(g);
    g->lt = lt; g->handle = handle; g->dtype = dtype; g->m = m; g->n = n; g->k = k;
    hipblasComputeType_i compute = dtype == kI8I32 ? kCompute32I : kCompute32F;
    hipDataType_i scale = dtype == kI8I32 ? kR32I : kR32F;
    hipDataType_i ab = dtype == kI8I32 ? kR8I : kR16F;
    hipDataType_i cd = dtype == kI8I32 ? kR32I : kR32F;
    hipblasStatus_t st = lt->desc_create(&g->desc, compute, scale);
    if (st != kLtSuccess) {
        std::printf("  desc_create %s\n", lt_status_str(st));
        return false;
    }
    hipblasOperation_i op = kOpN;
    st = lt->desc_set(g->desc, kDescTransA, &op, sizeof(op));
    if (st != kLtSuccess) { std::printf("  set TRANSA %s\n", lt_status_str(st)); return false; }
    st = lt->desc_set(g->desc, kDescTransB, &op, sizeof(op));
    if (st != kLtSuccess) { std::printf("  set TRANSB %s\n", lt_status_str(st)); return false; }
    st = lt->layout_create(&g->a, ab, (uint64_t)m, (uint64_t)k, m);
    if (st != kLtSuccess) { std::printf("  layout A %s\n", lt_status_str(st)); return false; }
    st = lt->layout_create(&g->b, ab, (uint64_t)k, (uint64_t)n, k);
    if (st != kLtSuccess) { std::printf("  layout B %s\n", lt_status_str(st)); return false; }
    st = lt->layout_create(&g->c, cd, (uint64_t)m, (uint64_t)n, m);
    if (st != kLtSuccess) { std::printf("  layout C %s\n", lt_status_str(st)); return false; }
    st = lt->pref_create(&g->pref);
    if (st != kLtSuccess) { std::printf("  pref_create %s\n", lt_status_str(st)); return false; }
    st = lt->pref_set(g->pref, kPrefMaxWorkspace, &max_ws, sizeof(max_ws));
    if (st != kLtSuccess) { std::printf("  set workspace %s\n", lt_status_str(st)); return false; }
    return true;
}

static hipblasStatus_t do_matmul(const Matmul& g, const void* A, const void* B, void* D,
                                 const hipblasLtMatmulAlgo_t* algo, void* ws, size_t ws_bytes,
                                 hipStream_t stream) {
    int32_t ai = 1, bi = 0;
    float af = 1.f, bf = 0.f;
    const void* alpha = g.dtype == kI8I32 ? (const void*)&ai : (const void*)&af;
    const void* beta = g.dtype == kI8I32 ? (const void*)&bi : (const void*)&bf;
    return g.lt->matmul(g.handle, g.desc, alpha, A, g.a, B, g.b, beta, D, g.c, D, g.c,
                        algo, ws, ws_bytes, stream);
}

struct Row {
    int m = 0, n = 0, k = 0;
    const char* dtype = "";
    const char* status = "error";
    int heur_n = 0;
    int heur_tried = 0;
    uint32_t algo = 0;
    char hex[33]{};
    size_t workspace = 0;
    float waves = 0;
    const char* hostref = "SKIP";
    double hostref_metric = 0;
    double med_us = 0, p10_us = 0, p90_us = 0, mean_us = 0, tops = 0;
};

static double tops_of(int m, int n, int k, double us) {
    if (us <= 0) return 0;
    return (2.0 * (double)m * (double)n * (double)k) / (us * 1e-6) / 1e12;
}

static bool stats_us(const std::vector<float>& us, double* med, double* p10, double* p90, double* mean) {
    if (us.empty()) return false;
    std::vector<float> s = us;
    std::sort(s.begin(), s.end());
    *med = s[s.size() / 2];
    *p10 = s[(size_t)(s.size() * 0.10)];
    *p90 = s[(size_t)(s.size() * 0.90)];
    double acc = 0;
    for (float v : s) acc += v;
    *mean = acc / (double)s.size();
    return true;
}

static bool time_algo(const Matmul& g, const hipblasLtMatmulAlgo_t* algo, void* A, void* B, void* D,
                      void* ws, size_t ws_bytes, int warmup, int timed, hipEvent_t ev0, hipEvent_t ev1,
                      std::vector<float>* us_out, hipblasStatus_t* last) {
    us_out->clear();
    us_out->reserve(timed);
    for (int w = 0; w < warmup; w++) {
        hipblasStatus_t st = do_matmul(g, A, B, D, algo, ws, ws_bytes, 0);
        if (st != kLtSuccess) { *last = st; return false; }
    }
    CK(hipDeviceSynchronize());
    for (int t = 0; t < timed; t++) {
        CK(hipEventRecord(ev0, 0));
        hipblasStatus_t st = do_matmul(g, A, B, D, algo, ws, ws_bytes, 0);
        if (st != kLtSuccess) { *last = st; return false; }
        CK(hipEventRecord(ev1, 0));
        CK(hipEventSynchronize(ev1));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, ev0, ev1));
        us_out->push_back(ms * 1000.f);
    }
    *last = kLtSuccess;
    return true;
}

static void fill_host(DType dtype, int m, int n, int k, std::vector<uint8_t>* A,
                      std::vector<uint8_t>* B, uint64_t seed) {
    const size_t ab = ab_elem(dtype);
    A->assign((size_t)m * (size_t)k * ab, 0);
    B->assign((size_t)k * (size_t)n * ab, 0);
    uint64_t rng = seed;
    if (dtype == kI8I32) {
        for (size_t i = 0; i < A->size(); i++) (*A)[i] = (uint8_t)((int)(lcg_u64(&rng) % 17) - 8);
        for (size_t i = 0; i < B->size(); i++) (*B)[i] = (uint8_t)((int)(lcg_u64(&rng) % 17) - 8);
    } else {
        auto* Ah = reinterpret_cast<uint16_t*>(A->data());
        auto* Bh = reinterpret_cast<uint16_t*>(B->data());
        const size_t na = (size_t)m * (size_t)k, nb = (size_t)k * (size_t)n;
        for (size_t i = 0; i < na; i++) Ah[i] = f32_to_f16_bits(lcg_finite(&rng, -0.5f, 0.5f));
        for (size_t i = 0; i < nb; i++) Bh[i] = f32_to_f16_bits(lcg_finite(&rng, -0.5f, 0.5f));
    }
}

static bool hostref_i8(const std::vector<uint8_t>& A, const std::vector<uint8_t>& B,
                       const std::vector<int32_t>& C, int m, int n, int k, int tile, double* max_abs) {
    *max_abs = 0;
    const int tm = std::min(m, tile), tn = std::min(n, tile);
    for (int j = 0; j < tn; j++) {
        for (int i = 0; i < tm; i++) {
            int32_t acc = 0;
            for (int p = 0; p < k; p++) {
                const int8_t a = (int8_t)A[(size_t)i + (size_t)p * (size_t)m];
                const int8_t b = (int8_t)B[(size_t)p + (size_t)j * (size_t)k];
                acc += (int32_t)a * (int32_t)b;
            }
            const double d = std::abs((double)C[(size_t)i + (size_t)j * (size_t)m] - (double)acc);
            if (d > *max_abs) *max_abs = d;
        }
    }
    return *max_abs == 0;
}

static bool hostref_f16(const std::vector<uint8_t>& A, const std::vector<uint8_t>& B,
                        const std::vector<float>& C, int m, int n, int k, int tile, double* rel_l2) {
    const auto* Ah = reinterpret_cast<const uint16_t*>(A.data());
    const auto* Bh = reinterpret_cast<const uint16_t*>(B.data());
    const int tm = std::min(m, tile), tn = std::min(n, tile);
    double num = 0, den = 0;
    for (int j = 0; j < tn; j++) {
        for (int i = 0; i < tm; i++) {
            double acc = 0;
            for (int p = 0; p < k; p++) {
                const float a = f16_bits_to_f32(Ah[(size_t)i + (size_t)p * (size_t)m]);
                const float b = f16_bits_to_f32(Bh[(size_t)p + (size_t)j * (size_t)k]);
                acc += (double)a * (double)b;
            }
            const double d = (double)C[(size_t)i + (size_t)j * (size_t)m] - acc;
            num += d * d;
            den += acc * acc;
        }
    }
    *rel_l2 = den == 0 ? std::sqrt(num) : std::sqrt(num / den);
    return *rel_l2 <= 5e-3;
}

static bool run_hostref(const Lt& lt, hipblasLtHandle_t handle, DType dtype, uint64_t max_ws,
                        hipEvent_t ev0, hipEvent_t ev1, Row* row) {
    const int m = kHostRefM, n = kHostRefN, k = kHostRefK;
    Matmul g{};
    if (!matmul_init(&g, &lt, handle, dtype, m, n, k, max_ws)) {
        row->hostref = "SKIP";
        matmul_destroy(&g);
        return true;
    }
    std::vector<hipblasLtMatmulHeuristicResult_t> heur((size_t)8);
    int ret = 0;
    hipblasStatus_t st = lt.heur(handle, g.desc, g.a, g.b, g.c, g.c, g.pref, 8, heur.data(), &ret);
    if (st != kLtSuccess || ret <= 0) {
        std::printf("  HOSTREF %s 128x64x128 heuristic %s count=%d : SKIP\n",
                    dtype_name(dtype), lt_status_str(st), ret);
        row->hostref = "SKIP";
        matmul_destroy(&g);
        return true;
    }
    int pick = -1;
    for (int i = 0; i < ret; i++) if (heur[i].state == kLtSuccess) { pick = i; break; }
    if (pick < 0) {
        row->hostref = "SKIP";
        matmul_destroy(&g);
        return true;
    }
    std::vector<uint8_t> Ah, Bh;
    fill_host(dtype, m, n, k, &Ah, &Bh, 42);
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dWs = nullptr;
    CK(hipMalloc(&dA, Ah.size()));
    CK(hipMalloc(&dB, Bh.size()));
    CK(hipMalloc(&dC, (size_t)m * (size_t)n * cd_elem(dtype)));
    const size_t ws = heur[pick].workspaceSize;
    if (ws) CK(hipMalloc(&dWs, ws));
    CK(hipMemcpy(dA, Ah.data(), Ah.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, Bh.data(), Bh.size(), hipMemcpyHostToDevice));
    CK(hipMemset(dC, 0, (size_t)m * (size_t)n * cd_elem(dtype)));
    st = do_matmul(g, dA, dB, dC, &heur[pick].algo, dWs, ws, 0);
    CK(hipDeviceSynchronize());
    bool ok = false;
    if (st != kLtSuccess) {
        std::printf("  HOSTREF %s matmul %s : FAIL\n", dtype_name(dtype), lt_status_str(st));
        row->hostref = "FAIL";
        ok = false;
    } else if (dtype == kI8I32) {
        std::vector<int32_t> Ch((size_t)m * (size_t)n);
        CK(hipMemcpy(Ch.data(), dC, Ch.size() * 4, hipMemcpyDeviceToHost));
        double max_abs = 0;
        ok = hostref_i8(Ah, Bh, Ch, m, n, k, 8, &max_abs);
        row->hostref_metric = max_abs;
        row->hostref = ok ? "PASS" : "FAIL";
        std::printf("  HOSTREF %s 128x64x128 tile8 max_abs=%.0f : %s\n",
                    dtype_name(dtype), max_abs, row->hostref);
    } else {
        std::vector<float> Ch((size_t)m * (size_t)n);
        CK(hipMemcpy(Ch.data(), dC, Ch.size() * 4, hipMemcpyDeviceToHost));
        double rel = 0;
        ok = hostref_f16(Ah, Bh, Ch, m, n, k, 8, &rel);
        row->hostref_metric = rel;
        row->hostref = ok ? "PASS" : "FAIL";
        std::printf("  HOSTREF %s 128x64x128 tile8 rel_l2=%.2e : %s\n",
                    dtype_name(dtype), rel, row->hostref);
    }
    if (dWs) CK(hipFree(dWs));
    CK(hipFree(dC)); CK(hipFree(dB)); CK(hipFree(dA));
    matmul_destroy(&g);
    (void)ev0; (void)ev1;
    return ok;
}

static Row run_shape(const Lt& lt, hipblasLtHandle_t handle, DType dtype, int m, int n, int k,
                     int warmup, int timed, int heur_n, uint64_t max_ws,
                     hipEvent_t ev0, hipEvent_t ev1, const char* hostref_status) {
    Row row;
    row.m = m; row.n = n; row.k = k; row.dtype = dtype_name(dtype);
    row.hostref = hostref_status;
    std::printf("SHAPE M=%d N=%d K=%d dtype=%s\n", m, n, k, row.dtype);
    Matmul g{};
    if (!matmul_init(&g, &lt, handle, dtype, m, n, k, max_ws)) {
        row.status = "error";
        matmul_destroy(&g);
        return row;
    }
    std::vector<hipblasLtMatmulHeuristicResult_t> heur((size_t)heur_n);
    int ret = 0;
    hipblasStatus_t st = lt.heur(handle, g.desc, g.a, g.b, g.c, g.c, g.pref, heur_n, heur.data(), &ret);
    row.heur_n = ret;
    std::printf("  heuristic status=%s count=%d sizeof_heur=%zu\n",
                lt_status_str(st), ret, sizeof(hipblasLtMatmulHeuristicResult_t));
    if ((st != kLtSuccess && st != kLtNotSupported) && ret <= 0) {
        row.status = "error";
        matmul_destroy(&g);
        return row;
    }
    if (ret <= 0) {
        std::printf("  NO_ALGO (finding, not failure)\n");
        row.status = "no_algo";
        matmul_destroy(&g);
        return row;
    }

    std::vector<uint8_t> Ah, Bh;
    fill_host(dtype, m, n, k, &Ah, &Bh, 7);
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dWs = nullptr;
    CK(hipMalloc(&dA, Ah.size()));
    CK(hipMalloc(&dB, Bh.size()));
    CK(hipMalloc(&dC, (size_t)m * (size_t)n * cd_elem(dtype)));
    size_t need_ws = 0;
    int viable = 0;
    for (int i = 0; i < ret; i++) {
        if (heur[i].state != kLtSuccess) continue;
        viable++;
        if (heur[i].workspaceSize > need_ws) need_ws = heur[i].workspaceSize;
        char hx[33]; algo_hex(heur[i].algo, hx);
        std::printf("  cand[%d] algo_id=%u algo_hex=%s ws=%zu waves=%.2f\n",
                    i, algo_id(heur[i].algo), hx, heur[i].workspaceSize, heur[i].wavesCount);
    }
    if (viable == 0) {
        std::printf("  NO_ALGO (all heuristic.state != SUCCESS)\n");
        row.status = "no_algo";
        CK(hipFree(dC)); CK(hipFree(dB)); CK(hipFree(dA));
        matmul_destroy(&g);
        return row;
    }
    if (need_ws) CK(hipMalloc(&dWs, need_ws));
    CK(hipMemcpy(dA, Ah.data(), Ah.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, Bh.data(), Bh.size(), hipMemcpyHostToDevice));
    CK(hipMemset(dC, 0, (size_t)m * (size_t)n * cd_elem(dtype)));

    int best = -1;
    double best_med = 1e300;
    hipblasStatus_t last = kLtSuccess;
    std::vector<float> us;
    for (int i = 0; i < ret; i++) {
        if (heur[i].state != kLtSuccess) continue;
        row.heur_tried++;
        if (!time_algo(g, &heur[i].algo, dA, dB, dC, dWs, heur[i].workspaceSize,
                       kPickWarmup, kPickTimed, ev0, ev1, &us, &last)) {
            std::printf("  cand[%d] runtime %s — skip\n", i, lt_status_str(last));
            continue;
        }
        double med = 0, p10 = 0, p90 = 0, mean = 0;
        stats_us(us, &med, &p10, &p90, &mean);
        std::printf("  cand[%d] pick_med_us=%.2f tops=%.3f\n", i, med, tops_of(m, n, k, med));
        if (med < best_med) { best_med = med; best = i; }
    }
    if (best < 0) {
        std::printf("  NO_ALGO (heuristics returned but every launch failed)\n");
        row.status = "no_algo";
        if (dWs) CK(hipFree(dWs));
        CK(hipFree(dC)); CK(hipFree(dB)); CK(hipFree(dA));
        matmul_destroy(&g);
        return row;
    }

    if (!time_algo(g, &heur[best].algo, dA, dB, dC, dWs, heur[best].workspaceSize,
                   warmup, timed, ev0, ev1, &us, &last)) {
        std::printf("  timed run failed: %s\n", lt_status_str(last));
        row.status = "error";
        if (dWs) CK(hipFree(dWs));
        CK(hipFree(dC)); CK(hipFree(dB)); CK(hipFree(dA));
        matmul_destroy(&g);
        return row;
    }
    stats_us(us, &row.med_us, &row.p10_us, &row.p90_us, &row.mean_us);
    row.tops = tops_of(m, n, k, row.med_us);
    row.algo = algo_id(heur[best].algo);
    algo_hex(heur[best].algo, row.hex);
    row.workspace = heur[best].workspaceSize;
    row.waves = heur[best].wavesCount;
    row.status = "ok";
    std::printf("  BEST algo_id=%u algo_hex=%s ws=%zu waves=%.2f\n",
                row.algo, row.hex, row.workspace, row.waves);
    std::printf("  med_us=%.2f p10=%.2f p90=%.2f mean_us=%.2f  tops=%.3f\n",
                row.med_us, row.p10_us, row.p90_us, row.mean_us, row.tops);

    if (dWs) CK(hipFree(dWs));
    CK(hipFree(dC)); CK(hipFree(dB)); CK(hipFree(dA));
    matmul_destroy(&g);
    return row;
}

int main(int argc, char** argv) {
    std::string arch;
    int warmup = kDefWarmup, timed = kDefTimed, heur_n = kDefHeurN;
    int want_i8 = 1, want_f16 = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&] {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return std::atoi(argv[++i]);
        };
        if (a == "--arch" && i + 1 < argc) arch = argv[++i];
        else if (a == "--warmup") warmup = next();
        else if (a == "--timed" || a == "--samples") timed = next();
        else if (a == "--heuristic-n") heur_n = next();
        else if (a == "--dtype" && i + 1 < argc) {
            std::string d = argv[++i];
            want_i8 = (d == "all" || d == "i8");
            want_f16 = (d == "all" || d == "f16");
            if (!want_i8 && !want_f16) { std::fprintf(stderr, "--dtype all|i8|f16\n"); return 2; }
        } else {
            std::printf("Usage: %s --arch gfxNNNN [--warmup W] [--timed T] [--heuristic-n N] [--dtype all|i8|f16]\nIdentity: %s\n",
                        argv[0], kBenchId);
            return a == "--help" || a == "-h" ? 0 : 2;
        }
    }
    if (arch.empty()) { std::fprintf(stderr, "--arch required\n"); return 2; }
    if (warmup < 20) warmup = 20;
    if (timed < 100) timed = 100;
    if (heur_n < 1) heur_n = 1;

    int ndev = 0; CK(hipGetDeviceCount(&ndev));
    int dev = -1; hipDeviceProp_t prop{};
    for (int d = 0; d < ndev; d++) {
        hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, d));
        if (std::strncmp(p.gcnArchName, arch.c_str(), arch.size()) == 0) { dev = d; prop = p; break; }
    }
    if (dev < 0) { std::fprintf(stderr, "no device with gcnArchName %s*\n", arch.c_str()); return 2; }
    CK(hipSetDevice(dev));

    uint64_t max_ws = 256ull << 20;
    if (const char* e = std::getenv("HIPBLASLT_WORKSPACE_MB")) {
        int mb = std::atoi(e);
        if (mb > 0) max_ws = (uint64_t)mb << 20;
    }

    std::string libpath;
    Lt lt = load_lt(&libpath);
    hipblasLtHandle_t handle = nullptr;
    hipblasStatus_t st = lt.create(&handle);
    if (st != kLtSuccess || !handle) {
        std::fprintf(stderr, "hipblasLtCreate: %s\n", lt_status_str(st));
        return 1;
    }
    int ver = -1;
    if (lt.get_version) lt.get_version(handle, &ver);

    std::printf("%s\narch=%s dev=%d (%s) gcn=%s\nlib=%s hipblaslt_version=%d\n"
                "warmup=%d timed=%d heuristic_n=%d workspace_mb=%llu sizeof_algo=%zu sizeof_heur=%zu\n",
                kBenchId, arch.c_str(), dev, prop.name, prop.gcnArchName, libpath.c_str(), ver,
                warmup, timed, heur_n, (unsigned long long)(max_ws >> 20),
                sizeof(hipblasLtMatmulAlgo_t), sizeof(hipblasLtMatmulHeuristicResult_t));

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));

    const char* href_i8 = "SKIP";
    const char* href_f16 = "SKIP";
    bool pass = true;
    if (want_i8) {
        Row r{};
        if (!run_hostref(lt, handle, kI8I32, max_ws, ev0, ev1, &r)) pass = false;
        href_i8 = r.hostref;
    }
    if (want_f16) {
        Row r{};
        if (!run_hostref(lt, handle, kF16F32, max_ws, ev0, ev1, &r)) pass = false;
        href_f16 = r.hostref;
    }

    static const int kShapes[][3] = {
        {17408, 512, 5120},
        {17408, 2048, 5120},
        {5120, 512, 17408},
        {5120, 2048, 17408},
    };
    std::vector<Row> rows;
    DType dts[2] = {kI8I32, kF16F32};
    int want[2] = {want_i8, want_f16};
    const char* hrefs[2] = {href_i8, href_f16};
    for (int si = 0; si < 4; si++) {
        for (int di = 0; di < 2; di++) {
            if (!want[di]) continue;
            Row r = run_shape(lt, handle, dts[di], kShapes[si][0], kShapes[si][1], kShapes[si][2],
                              warmup, timed, heur_n, max_ws, ev0, ev1, hrefs[di]);
            if (std::strcmp(r.status, "error") == 0) pass = false;
            rows.push_back(r);
        }
    }

    std::printf("TABLE shape dtype status heur_n algo_id med_us tops hostref\n");
    for (const auto& r : rows) {
        std::printf("  %d,%d,%d %s %s heur=%d algo=%u med_us=%.2f tops=%.3f hostref=%s\n",
                    r.m, r.n, r.k, r.dtype, r.status, r.heur_n, r.algo, r.med_us, r.tops, r.hostref);
    }

    std::printf("JSON {\"id\":\"%s\",\"arch\":\"%s\",\"dev\":%d,\"name\":\"%s\",\"gcn\":\"%s\","
                "\"lib\":\"%s\",\"hipblaslt_version\":%d,\"warmup\":%d,\"timed\":%d,\"heuristic_n\":%d,"
                "\"results\":[",
                kBenchId, arch.c_str(), dev, prop.name, prop.gcnArchName, libpath.c_str(), ver,
                warmup, timed, heur_n);
    for (size_t i = 0; i < rows.size(); i++) {
        const auto& r = rows[i];
        std::printf("%s{\"m\":%d,\"n\":%d,\"k\":%d,\"dtype\":\"%s\",\"status\":\"%s\",\"heuristic_count\":%d,"
                    "\"heuristic_tried\":%d,\"algo_id\":%u,\"algo_hex\":\"%s\",\"workspace_bytes\":%zu,"
                    "\"waves\":%.3f,\"hostref\":\"%s\",\"hostref_metric\":%.6g,"
                    "\"med_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,\"mean_us\":%.3f,\"tops\":%.4f}",
                    i ? "," : "", r.m, r.n, r.k, r.dtype, r.status, r.heur_n, r.heur_tried, r.algo, r.hex,
                    r.workspace, r.waves, r.hostref, r.hostref_metric, r.med_us, r.p10_us, r.p90_us,
                    r.mean_us, r.tops);
    }
    std::printf("]}\n");
    std::printf("%s\n", pass ? "OVERALL PASS" : "OVERALL FAIL");

    CK(hipEventDestroy(ev0)); CK(hipEventDestroy(ev1));
    lt.destroy(handle);
    dlclose(lt.lib);
    return pass ? 0 : 1;
}
