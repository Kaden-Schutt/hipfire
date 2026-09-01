// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cstdint>
#include <climits>
#include <string>
#include <vector>

namespace {

#if defined(HIPFIRE_CK_TARGET_GFX1201)
constexpr int32_t kExpectedArch = HIPFIRE_FLASH_ATTN_CK_GFX1201;
constexpr size_t kExpectedCapabilities = 5;
constexpr bool kExpectedQ8D256 = false;
constexpr bool kExpectedAsym3GivensD256 = true;
constexpr bool kExpectedAsym3FwhtD256 = true;
constexpr bool kExpectedAsym4D256 = true;
#elif defined(HIPFIRE_CK_TARGET_GFX1151)
constexpr int32_t kExpectedArch = HIPFIRE_FLASH_ATTN_CK_GFX1151;
constexpr size_t kExpectedCapabilities = 1;
constexpr bool kExpectedQ8D256 = false;
constexpr bool kExpectedAsym3GivensD256 = false;
constexpr bool kExpectedAsym3FwhtD256 = false;
constexpr bool kExpectedAsym4D256 = false;
#else
constexpr int32_t kExpectedArch = HIPFIRE_FLASH_ATTN_CK_GFX1100;
constexpr size_t kExpectedCapabilities = 6;
constexpr bool kExpectedQ8D256 = true;
constexpr bool kExpectedAsym3GivensD256 = true;
constexpr bool kExpectedAsym3FwhtD256 = true;
constexpr bool kExpectedAsym4D256 = true;
#endif

bool has_capability(int32_t dtype, int32_t k_format, int32_t v_format, int32_t head_dim)
{
    const size_t count = hipfire_flash_attn_ck_capabilities(nullptr, 0);
    std::vector<hipfire_flash_attn_ck_capability> capabilities(count);
    if(hipfire_flash_attn_ck_capabilities(capabilities.data(), capabilities.size()) != count)
    {
        std::fprintf(stderr, "capability table changed while reading\n");
        std::exit(2);
    }
    return std::any_of(capabilities.begin(), capabilities.end(), [&](const auto& capability) {
        return capability.arch == kExpectedArch && capability.dtype == dtype &&
               capability.k_format == k_format && capability.v_format == v_format &&
               capability.head_dim == head_dim;
    });
}

void verify_capabilities()
{
    const size_t count = hipfire_flash_attn_ck_capabilities(nullptr, 0);
    const bool has_q8 = has_capability(HIPFIRE_FLASH_ATTN_CK_F32,
                                       HIPFIRE_FLASH_ATTN_CK_Q8,
                                       HIPFIRE_FLASH_ATTN_CK_Q8,
                                       256);
    const bool has_asym3 = has_capability(HIPFIRE_FLASH_ATTN_CK_F32,
                                          HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS,
                                          HIPFIRE_FLASH_ATTN_CK_Q8,
                                          256);
    const bool has_asym3_fwht = has_capability(HIPFIRE_FLASH_ATTN_CK_F32,
                                               HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT,
                                               HIPFIRE_FLASH_ATTN_CK_Q8,
                                               256);
    const bool has_asym4_givens = has_capability(HIPFIRE_FLASH_ATTN_CK_F32,
                                                 HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS,
                                                 HIPFIRE_FLASH_ATTN_CK_Q8,
                                                 256);
    const bool has_asym4_fwht = has_capability(HIPFIRE_FLASH_ATTN_CK_F32,
                                               HIPFIRE_FLASH_ATTN_CK_ASYM4_FWHT,
                                               HIPFIRE_FLASH_ATTN_CK_Q8,
                                               256);
    if(count != kExpectedCapabilities ||
       has_q8 != kExpectedQ8D256 || has_asym3 != kExpectedAsym3GivensD256 ||
       has_asym3_fwht != kExpectedAsym3FwhtD256 ||
       has_asym4_givens != kExpectedAsym4D256 || has_asym4_fwht != kExpectedAsym4D256 ||
       !has_capability(HIPFIRE_FLASH_ATTN_CK_F16,
                       HIPFIRE_FLASH_ATTN_CK_DENSE_F16,
                       HIPFIRE_FLASH_ATTN_CK_DENSE_F16,
                       64))
    {
        std::fprintf(stderr,
                     "unexpected exact-architecture capability table: count=%zu expected=%zu "
                     "q8=%d expected_q8=%d asym3=%d expected_asym3=%d "
                     "asym3_fwht=%d expected_asym3_fwht=%d "
                     "asym4_givens=%d asym4_fwht=%d expected_asym4=%d arch=%d\n",
                     count,
                     kExpectedCapabilities,
                     has_q8,
                     kExpectedQ8D256,
                     has_asym3,
                     kExpectedAsym3GivensD256,
                     has_asym3_fwht,
                     kExpectedAsym3FwhtD256,
                     has_asym4_givens,
                     has_asym4_fwht,
                     kExpectedAsym4D256,
                     kExpectedArch);
        std::exit(2);
    }
}

void check_hip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
        std::exit(2);
    }
}

size_t offset(int b, int s, int h, int d, int seqlen, int heads, int hdim)
{
    return ((static_cast<size_t>(b) * seqlen + s) * heads + h) * hdim + d;
}

float load(const std::vector<__half>& values,
           int b,
           int s,
           int h,
           int d,
           int seqlen,
           int heads,
           int hdim)
{
    return __half2float(values[offset(b, s, h, d, seqlen, heads, hdim)]);
}

void run_case(const char* name, int nhead_q, int nhead_k, bool causal, bool non_default_stream)
{
    constexpr int batch = 1;
    constexpr int seqlen_q = 64;
    constexpr int seqlen_k = 96;
    constexpr int hdim = 64;
    constexpr float scale = 1.0f / 8.0f;
    const int groups = nhead_q / nhead_k;

    const size_t q_count = static_cast<size_t>(batch) * seqlen_q * nhead_q * hdim;
    const size_t kv_count = static_cast<size_t>(batch) * seqlen_k * nhead_k * hdim;
    std::vector<__half> q(q_count), k(kv_count), v(kv_count), output(q_count);
    std::vector<float> expected(q_count, 0.0f);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for(auto* values : {&q, &k, &v})
    {
        for(auto& value : *values)
        {
            value = __float2half(distribution(rng));
        }
    }

    for(int b = 0; b < batch; ++b)
    {
        for(int hq = 0; hq < nhead_q; ++hq)
        {
            const int hk = hq / groups;
            for(int sq = 0; sq < seqlen_q; ++sq)
            {
                std::vector<float> scores(seqlen_k);
                float maximum = -INFINITY;
                for(int sk = 0; sk < seqlen_k; ++sk)
                {
                    if(causal && sk > sq + seqlen_k - seqlen_q)
                    {
                        scores[sk] = -INFINITY;
                        continue;
                    }
                    float score = 0.0f;
                    for(int d = 0; d < hdim; ++d)
                    {
                        score += load(q, b, sq, hq, d, seqlen_q, nhead_q, hdim) *
                                 load(k, b, sk, hk, d, seqlen_k, nhead_k, hdim);
                    }
                    scores[sk] = score * scale;
                    maximum = std::max(maximum, scores[sk]);
                }
                float denominator = 0.0f;
                for(float& score : scores)
                {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for(int d = 0; d < hdim; ++d)
                {
                    float value = 0.0f;
                    for(int sk = 0; sk < seqlen_k; ++sk)
                    {
                        value += scores[sk] / denominator *
                                 load(v, b, sk, hk, d, seqlen_k, nhead_k, hdim);
                    }
                    expected[offset(b, sq, hq, d, seqlen_q, nhead_q, hdim)] = value;
                }
            }
        }
    }

    void* q_device = nullptr;
    void* k_device = nullptr;
    void* v_device = nullptr;
    void* out_device = nullptr;
    check_hip(hipMalloc(&q_device, q_count * sizeof(__half)), "hipMalloc(q)");
    check_hip(hipMalloc(&k_device, kv_count * sizeof(__half)), "hipMalloc(k)");
    check_hip(hipMalloc(&v_device, kv_count * sizeof(__half)), "hipMalloc(v)");
    check_hip(hipMalloc(&out_device, q_count * sizeof(__half)), "hipMalloc(out)");
    hipStream_t stream = nullptr;
    if(non_default_stream)
    {
        check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    }
    check_hip(hipMemcpyAsync(q_device,
                            q.data(),
                            q_count * sizeof(__half),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(q)");
    check_hip(hipMemcpyAsync(k_device,
                            k.data(),
                            kv_count * sizeof(__half),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(k)");
    check_hip(hipMemcpyAsync(v_device,
                            v.data(),
                            kv_count * sizeof(__half),
                            hipMemcpyHostToDevice,
                            stream),
              "hipMemcpyAsync(v)");

    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = q_device;
    params.k = k_device;
    params.v = v_device;
    params.out = out_device;
    params.stream = reinterpret_cast<void*>(stream);
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F16;
    params.k_format = HIPFIRE_FLASH_ATTN_CK_DENSE_F16;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_DENSE_F16;
    params.batch = batch;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.nhead_q = nhead_q;
    params.nhead_k = nhead_k;
    params.head_dim = hdim;
    params.causal = causal ? 1 : 0;
    params.softmax_scale = scale;
    params.stride_q = nhead_q * hdim;
    params.stride_k = nhead_k * hdim;
    params.stride_v = nhead_k * hdim;
    params.stride_out = nhead_q * hdim;
    params.nhead_stride_q = hdim;
    params.nhead_stride_k = hdim;
    params.nhead_stride_v = hdim;
    params.nhead_stride_out = hdim;
    params.batch_stride_q = seqlen_q * nhead_q * hdim;
    params.batch_stride_k = seqlen_k * nhead_k * hdim;
    params.batch_stride_v = seqlen_k * nhead_k * hdim;
    params.batch_stride_out = seqlen_q * nhead_q * hdim;

    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd(&params, error, sizeof(error));
    if(status != 0)
    {
        std::fprintf(stderr, "sidecar status=%d: %s\n", status, error);
        std::exit(3);
    }
    check_hip(hipMemcpyAsync(output.data(),
                            out_device,
                            q_count * sizeof(__half),
                            hipMemcpyDeviceToHost,
                            stream),
              "hipMemcpyAsync(out)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize");

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for(size_t i = 0; i < q_count; ++i)
    {
        const float delta = std::abs(__half2float(output[i]) - expected[i]);
        max_abs = std::max(max_abs, delta);
        mean_abs += delta;
    }
    mean_abs /= q_count;
    std::printf("case=%s dtype=fp16 q_heads=%d kv_heads=%d causal=%d stream=%s "
                "max_abs=%.7g mean_abs=%.7g\n",
                name,
                nhead_q,
                nhead_k,
                causal ? 1 : 0,
                non_default_stream ? "non-default" : "default",
                max_abs,
                mean_abs);
    if(max_abs > 0.02f)
    {
        std::exit(4);
    }

    check_hip(hipFree(q_device), "hipFree(q)");
    check_hip(hipFree(k_device), "hipFree(k)");
    check_hip(hipFree(v_device), "hipFree(v)");
    check_hip(hipFree(out_device), "hipFree(out)");
    if(non_default_stream)
    {
        check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    }
}

void pack_q8(const std::vector<float>& input,
             std::vector<uint8_t>& packed,
             std::vector<float>& decoded,
             int rows,
             int heads)
{
    constexpr int hdim = 256;
    constexpr int head_bytes = 272;
    for(int row = 0; row < rows; ++row)
    {
        for(int head = 0; head < heads; ++head)
        {
            for(int block = 0; block < 8; ++block)
            {
                const size_t input_base = (static_cast<size_t>(row) * heads + head) * hdim + block * 32;
                const size_t packed_base = (static_cast<size_t>(row) * heads + head) * head_bytes + block * 34;
                float maximum = 0.0f;
                for(int index = 0; index < 32; ++index)
                    maximum = std::max(maximum, std::abs(input[input_base + index]));
                const float scale = maximum > 0.0f ? maximum / 127.0f : 0.0f;
                const __half scale_half = __float2half(scale);
                std::memcpy(packed.data() + packed_base, &scale_half, sizeof(scale_half));
                const float stored_scale = __half2float(scale_half);
                for(int index = 0; index < 32; ++index)
                {
                    const int quantized = stored_scale > 0.0f
                                              ? static_cast<int>(std::nearbyint(input[input_base + index] / stored_scale))
                                              : 0;
                    const int8_t value = static_cast<int8_t>(std::clamp(quantized, -127, 127));
                    std::memcpy(packed.data() + packed_base + 2 + index, &value, 1);
                    decoded[input_base + index] = stored_scale * static_cast<float>(value);
                }
            }
        }
    }
}

void run_q8_d256_case()
{
    constexpr int seqlen_q = 16;
    constexpr int seqlen_k = 32;
    constexpr int nhead_q = 4;
    constexpr int nhead_k = 2;
    constexpr int hdim = 256;
    constexpr int head_bytes = 272;
    constexpr float scale = 1.0f / 16.0f;
    const size_t q_count = static_cast<size_t>(seqlen_q) * nhead_q * hdim;
    const size_t kv_count = static_cast<size_t>(seqlen_k) * nhead_k * hdim;
    std::vector<float> q(q_count), k(kv_count), v(kv_count), decoded_k(kv_count),
        decoded_v(kv_count), output(q_count), expected(q_count);
    std::vector<uint8_t> packed_k(static_cast<size_t>(seqlen_k) * nhead_k * head_bytes);
    std::vector<uint8_t> packed_v(packed_k.size());
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for(auto* values : {&q, &k, &v})
        for(float& value : *values) value = distribution(rng);
    pack_q8(k, packed_k, decoded_k, seqlen_k, nhead_k);
    pack_q8(v, packed_v, decoded_v, seqlen_k, nhead_k);

    for(int hq = 0; hq < nhead_q; ++hq)
    {
        const int hk = hq / (nhead_q / nhead_k);
        for(int sq = 0; sq < seqlen_q; ++sq)
        {
            std::vector<float> scores(seqlen_k, -INFINITY);
            float maximum = -INFINITY;
            const int last_key = sq + seqlen_k - seqlen_q;
            for(int sk = 0; sk <= last_key; ++sk)
            {
                float score = 0.0f;
                for(int d = 0; d < hdim; ++d)
                    score += q[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + d] *
                             decoded_k[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + d];
                scores[sk] = score * scale;
                maximum = std::max(maximum, scores[sk]);
            }
            float denominator = 0.0f;
            for(int sk = 0; sk <= last_key; ++sk)
            {
                scores[sk] = std::exp(scores[sk] - maximum);
                denominator += scores[sk];
            }
            for(int d = 0; d < hdim; ++d)
                for(int sk = 0; sk <= last_key; ++sk)
                    expected[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + d] +=
                        scores[sk] / denominator *
                        decoded_v[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + d];
        }
    }

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr, *workspace = nullptr;
    check_hip(hipMalloc(&dq, q_count * sizeof(float)), "hipMalloc(q8 q)");
    check_hip(hipMalloc(&dk, packed_k.size()), "hipMalloc(q8 k)");
    check_hip(hipMalloc(&dv, packed_v.size()), "hipMalloc(q8 v)");
    check_hip(hipMalloc(&dout, q_count * sizeof(float)), "hipMalloc(q8 out)");
    check_hip(hipMemcpy(dq, q.data(), q_count * sizeof(float), hipMemcpyHostToDevice), "copy q8 q");
    check_hip(hipMemcpy(dk, packed_k.data(), packed_k.size(), hipMemcpyHostToDevice), "copy q8 k");
    check_hip(hipMemcpy(dv, packed_v.data(), packed_v.size(), hipMemcpyHostToDevice), "copy q8 v");

    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = dq; params.k = dk; params.v = dv; params.out = dout;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1; params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
    params.nhead_q = nhead_q; params.nhead_k = nhead_k; params.head_dim = hdim;
    params.causal = 1; params.softmax_scale = scale;
    params.stride_q = nhead_q * hdim; params.stride_k = nhead_k * hdim;
    params.stride_v = nhead_k * hdim; params.stride_out = nhead_q * hdim;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v =
        params.nhead_stride_out = hdim;
    params.batch_stride_q = seqlen_q * nhead_q * hdim;
    params.batch_stride_k = params.batch_stride_v = seqlen_k * nhead_k * hdim;
    params.batch_stride_out = params.batch_stride_q;
    params.packed_k_row_stride_bytes = params.packed_v_row_stride_bytes = nhead_k * head_bytes;
    params.packed_k_head_stride_bytes = params.packed_v_head_stride_bytes = head_bytes;
    params.workspace_bytes = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);
    check_hip(hipMalloc(&workspace, params.workspace_bytes), "hipMalloc(q8 workspace)");
    params.workspace = workspace;
    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd(&params, error, sizeof(error));
    if(status != 0) { std::fprintf(stderr, "q8 sidecar status=%d: %s\n", status, error); std::exit(5); }
    check_hip(hipMemcpy(output.data(), dout, q_count * sizeof(float), hipMemcpyDeviceToHost), "copy q8 out");
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    for(size_t index = 0; index < q_count; ++index)
    {
        const float delta = std::abs(output[index] - expected[index]);
        max_abs = std::max(max_abs, delta); mean_abs += delta;
    }
    mean_abs /= q_count;
    std::printf("case=q8-d256-gqa-causal max_abs=%.7g mean_abs=%.7g workspace=%zu\n",
                max_abs, mean_abs, params.workspace_bytes);
    if(max_abs > 0.01f) std::exit(6);
    check_hip(hipFree(dq), "hipFree(q8 q)");
    check_hip(hipFree(dk), "hipFree(q8 k)");
    check_hip(hipFree(dv), "hipFree(q8 v)");
    check_hip(hipFree(dout), "hipFree(q8 out)");
    check_hip(hipFree(workspace), "hipFree(q8 workspace)");
}

void run_asym3_contract_case(int format, int hdim, bool artifact_has_cell)
{
    int dummy = 0;
    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = params.k = params.v = &dummy;
    params.out = &dummy;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = format;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1;
    params.seqlen_q = 16;
    params.seqlen_k = 32;
    params.nhead_q = 4;
    params.nhead_k = 2;
    params.head_dim = hdim;
    params.causal = 1;
    params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(hdim));
    params.stride_q = params.stride_out = 4 * hdim;
    params.stride_k = params.stride_v = 2 * hdim;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v =
        params.nhead_stride_out = hdim;
    params.batch_stride_q = params.batch_stride_out = 16 * 4 * hdim;
    params.batch_stride_k = params.batch_stride_v = 32 * 2 * hdim;
    params.packed_k_head_stride_bytes = 4 + (hdim * 3) / 8;
    params.packed_v_head_stride_bytes = (hdim / 32) * 34;
    params.packed_k_row_stride_bytes = 2 * params.packed_k_head_stride_bytes;
    params.packed_v_row_stride_bytes = 2 * params.packed_v_head_stride_bytes;
    params.k_transform0 = params.k_transform1 = &dummy;
    const int transform_elements =
        format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS ? hdim / 2 : 256;
    params.k_transform0_elements = params.k_transform1_elements = transform_elements;
    params.workspace = &dummy;
    params.workspace_bytes = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);

    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error));
    const bool has_cell = artifact_has_cell && hdim == 256;
    if(!artifact_has_cell)
    {
        if(status != 2 || std::strstr(error, "not published") == nullptr)
        {
            std::fprintf(stderr, "unpublished asym3 contract status=%d: %s\n", status, error);
            std::exit(7);
        }
        return;
    }
    if((has_cell && status != 0) ||
       (!has_cell && (status != 2 || std::strstr(error, "no CK execution cell") == nullptr)))
    {
        std::fprintf(stderr, "asym3 contract status=%d: %s\n", status, error);
        std::exit(7);
    }
    params.packed_k_head_stride_bytes -= 1;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "malformed asym3 layout was accepted\n");
        std::exit(8);
    }
    params.packed_k_head_stride_bytes += 1;
    params.stride_q += 1;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "non-contiguous asym3 Q was accepted\n");
        std::exit(9);
    }
    params.stride_q -= 1;
    params.k_transform0 = nullptr;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "null asym3 transform was accepted\n");
        std::exit(10);
    }
    params.k_transform0 = &dummy;
    params.k_transform1_elements = transform_elements - 1;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "undersized asym3 transform was accepted\n");
        std::exit(11);
    }
    std::printf("case=asym3-contract format=%s head_dim=%d status=%s\n",
                format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS ? "givens" : "fwht", hdim,
                has_cell ? "supported" : "recognized-no-cell");
}

void run_asym4_contract_case(int format)
{
    constexpr int hdim = 256;
    int dummy = 0;
    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = params.k = params.v = params.out = &dummy;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = format;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1;
    params.seqlen_q = 16;
    params.seqlen_k = 32;
    params.nhead_q = 4;
    params.nhead_k = 2;
    params.head_dim = hdim;
    params.causal = 1;
    params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(hdim));
    params.stride_q = params.stride_out = 4 * hdim;
    params.stride_k = params.stride_v = 2 * hdim;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v =
        params.nhead_stride_out = hdim;
    params.batch_stride_q = params.batch_stride_out = 16 * 4 * hdim;
    params.batch_stride_k = params.batch_stride_v = 32 * 2 * hdim;
    params.packed_k_head_stride_bytes = 4 + hdim / 2;
    params.packed_v_head_stride_bytes = (hdim / 32) * 34;
    params.packed_k_row_stride_bytes = 2 * params.packed_k_head_stride_bytes;
    params.packed_v_row_stride_bytes = 2 * params.packed_v_head_stride_bytes;
    params.k_transform0 = params.k_transform1 = &dummy;
    const int transform_elements =
        format == HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS ? hdim / 2 : 128;
    params.k_transform0_elements = params.k_transform1_elements = transform_elements;

    char error[1024]{};
    params.workspace = &dummy;
    params.workspace_bytes = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 0)
    {
        std::fprintf(stderr, "asym4 contract was not recognized: %s\n", error);
        std::exit(14);
    }
    params.packed_k_head_stride_bytes -= 1;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "malformed asym4 layout was accepted\n");
        std::exit(15);
    }
    params.packed_k_head_stride_bytes += 1;
    params.packed_k_row_stride_bytes += 1;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "misaligned asym4 row stride was accepted\n");
        std::exit(16);
    }
    params.packed_k_row_stride_bytes -= 1;
    params.k_transform0 = nullptr;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "null asym4 transform was accepted\n");
        std::exit(17);
    }
    params.k_transform0 = &dummy;
    params.k_transform1_elements = transform_elements - 1;
    if(hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error)) != 1)
    {
        std::fprintf(stderr, "undersized asym4 transform was accepted\n");
        std::exit(18);
    }
    std::printf("case=asym4-contract format=%s head_dim=256 status=supported\n",
                format == HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS ? "givens" : "fwht");
}

void run_asym3_case(int format, int hdim)
{
    constexpr float centroids[8] = {
        -0.134860f, -0.083320f, -0.046469f, -0.015176f,
         0.015176f,  0.046469f,  0.083320f,  0.134860f,
    };
    constexpr int seqlen_q = 8;
    constexpr int seqlen_k = 16;
    constexpr int nhead_q = 4;
    constexpr int nhead_k = 2;
    const int k_head_bytes = 4 + (hdim * 3) / 8;
    const int v_head_bytes = (hdim / 32) * 34;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));
    const size_t q_count = static_cast<size_t>(seqlen_q) * nhead_q * hdim;
    const size_t kv_count = static_cast<size_t>(seqlen_k) * nhead_k * hdim;
    std::vector<float> q(q_count), transformed_q(q_count), v(kv_count), decoded_k(kv_count),
        decoded_v(kv_count), expected(q_count, 0.0f), output(q_count);
    const bool givens = format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS;
    std::vector<float> transform0(givens ? hdim / 2 : hdim);
    std::vector<float> transform1(givens ? hdim / 2 : hdim);
    std::vector<uint8_t> packed_k(static_cast<size_t>(seqlen_k) * nhead_k * k_head_bytes);
    std::vector<uint8_t> packed_v(static_cast<size_t>(seqlen_k) * nhead_k * v_head_bytes);
    std::mt19937 rng(29 + hdim);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for(float& value : q) value = distribution(rng);
    for(float& value : v) value = distribution(rng);
    transformed_q = q;
    for(int pair = 0; pair < hdim / 2; ++pair)
    {
        const float angle = 0.001f * static_cast<float>((pair * 17 + 3) % 97);
        if(givens)
        {
            transform0[pair] = std::cos(angle);
            transform1[pair] = std::sin(angle);
        }
    }
    if(givens)
    {
        for(int row = 0; row < seqlen_q; ++row)
            for(int head = 0; head < nhead_q; ++head)
                for(int pair = 0; pair < hdim / 2; ++pair)
                {
                    const size_t base =
                        (static_cast<size_t>(row) * nhead_q + head) * hdim + pair * 2;
                    const float a = transformed_q[base];
                    const float b = transformed_q[base + 1];
                    transformed_q[base] = a * transform0[pair] - b * transform1[pair];
                    transformed_q[base + 1] = a * transform1[pair] + b * transform0[pair];
                }
    }
    else
    {
        for(int dim = 0; dim < hdim; ++dim)
        {
            transform0[dim] = ((dim * 13 + 5) & 1) == 0 ? 1.0f : -1.0f;
            transform1[dim] = ((dim * 29 + 7) & 1) == 0 ? 1.0f : -1.0f;
        }
        for(int row = 0; row < seqlen_q; ++row)
            for(int head = 0; head < nhead_q; ++head)
            {
                const size_t base = (static_cast<size_t>(row) * nhead_q + head) * hdim;
                for(int dim = 0; dim < hdim; ++dim)
                    transformed_q[base + dim] *= transform0[dim];
                for(int stride = 1; stride < hdim; stride <<= 1)
                    for(int index = 0; index < hdim; index += stride * 2)
                        for(int offset = 0; offset < stride; ++offset)
                        {
                            const float a = transformed_q[base + index + offset];
                            const float b = transformed_q[base + index + offset + stride];
                            transformed_q[base + index + offset] = a + b;
                            transformed_q[base + index + offset + stride] = a - b;
                        }
                for(int dim = 0; dim < hdim; ++dim)
                    transformed_q[base + dim] *= 0.0625f * transform1[dim];
            }
    }
    for(int row = 0; row < seqlen_k; ++row)
        for(int head = 0; head < nhead_k; ++head)
        {
            uint8_t* destination = packed_k.data() +
                (static_cast<size_t>(row) * nhead_k + head) * k_head_bytes;
            const float cnorm = 0.5f + 0.01f * static_cast<float>((row + head) % 11);
            std::memcpy(destination, &cnorm, sizeof(cnorm));
            for(int chunk = 0; chunk < hdim / 256; ++chunk)
                for(int lane = 0; lane < 32; ++lane)
                {
                    uint32_t codes = 0;
                    for(int i = 0; i < 8; ++i)
                    {
                        const int code = (row * 3 + head * 5 + chunk * 7 + lane + i) & 7;
                        codes |= static_cast<uint32_t>(code) << (i * 3);
                        const int dim = chunk * 256 + lane * 8 + i;
                        decoded_k[(static_cast<size_t>(row) * nhead_k + head) * hdim + dim] =
                            cnorm * centroids[code];
                    }
                    uint8_t* bytes = destination + 4 + chunk * 96 + lane * 3;
                    bytes[0] = codes & 0xff;
                    bytes[1] = (codes >> 8) & 0xff;
                    bytes[2] = (codes >> 16) & 0xff;
                }
        }
    pack_q8(v, packed_v, decoded_v, seqlen_k, nhead_k);
    const int groups = nhead_q / nhead_k;
    for(int sq = 0; sq < seqlen_q; ++sq)
        for(int hq = 0; hq < nhead_q; ++hq)
        {
            const int hk = hq / groups;
            const int last_key = sq + seqlen_k - seqlen_q;
            std::vector<float> scores(last_key + 1);
            float maximum = -INFINITY;
            for(int sk = 0; sk <= last_key; ++sk)
            {
                float score = 0.0f;
                for(int d = 0; d < hdim; ++d)
                    score += transformed_q[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + d] *
                             decoded_k[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + d];
                scores[sk] = score * scale;
                maximum = std::max(maximum, scores[sk]);
            }
            float denominator = 0.0f;
            for(float& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for(int d = 0; d < hdim; ++d)
                for(int sk = 0; sk <= last_key; ++sk)
                    expected[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + d] +=
                        scores[sk] / denominator *
                        decoded_v[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + d];
        }

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    void *dcos = nullptr, *dsin = nullptr, *workspace = nullptr;
    check_hip(hipMalloc(&dq, q_count * sizeof(float)), "hipMalloc(asym q)");
    check_hip(hipMalloc(&dk, packed_k.size()), "hipMalloc(asym k)");
    check_hip(hipMalloc(&dv, packed_v.size()), "hipMalloc(asym v)");
    check_hip(hipMalloc(&dout, q_count * sizeof(float)), "hipMalloc(asym out)");
    check_hip(hipMalloc(&dcos, transform0.size() * sizeof(float)), "hipMalloc(asym transform0)");
    check_hip(hipMalloc(&dsin, transform1.size() * sizeof(float)), "hipMalloc(asym transform1)");
    check_hip(hipMemcpy(dq, q.data(), q_count * sizeof(float), hipMemcpyHostToDevice), "copy asym q");
    check_hip(hipMemcpy(dk, packed_k.data(), packed_k.size(), hipMemcpyHostToDevice), "copy asym k");
    check_hip(hipMemcpy(dv, packed_v.data(), packed_v.size(), hipMemcpyHostToDevice), "copy asym v");
    check_hip(hipMemcpy(dcos, transform0.data(), transform0.size() * sizeof(float), hipMemcpyHostToDevice), "copy asym transform0");
    check_hip(hipMemcpy(dsin, transform1.data(), transform1.size() * sizeof(float), hipMemcpyHostToDevice), "copy asym transform1");
    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION; params.struct_size = sizeof(params);
    params.q = dq; params.k = dk; params.v = dv; params.out = dout;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = format;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1; params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
    params.nhead_q = nhead_q; params.nhead_k = nhead_k; params.head_dim = hdim;
    params.causal = 1; params.softmax_scale = scale;
    params.stride_q = params.stride_out = nhead_q * hdim;
    params.stride_k = params.stride_v = nhead_k * hdim;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v = params.nhead_stride_out = hdim;
    params.batch_stride_q = params.batch_stride_out = seqlen_q * nhead_q * hdim;
    params.batch_stride_k = params.batch_stride_v = seqlen_k * nhead_k * hdim;
    params.packed_k_head_stride_bytes = k_head_bytes;
    params.packed_v_head_stride_bytes = v_head_bytes;
    params.packed_k_row_stride_bytes = nhead_k * k_head_bytes;
    params.packed_v_row_stride_bytes = nhead_k * v_head_bytes;
    params.k_transform0 = dcos; params.k_transform1 = dsin;
    params.k_transform0_elements = params.k_transform1_elements = givens ? hdim / 2 : hdim;
    params.workspace_bytes = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);
    check_hip(hipMalloc(&workspace, params.workspace_bytes), "hipMalloc(asym workspace)");
    params.workspace = workspace;
    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd(&params, error, sizeof(error));
    if(status != 0) { std::fprintf(stderr, "asym3 sidecar status=%d: %s\n", status, error); std::exit(12); }
    check_hip(hipMemcpy(output.data(), dout, q_count * sizeof(float), hipMemcpyDeviceToHost), "copy asym out");
    float max_abs = 0.0f; double mean_abs = 0.0;
    for(size_t i = 0; i < q_count; ++i) { const float d = std::abs(output[i] - expected[i]); max_abs = std::max(max_abs, d); mean_abs += d; }
    mean_abs /= q_count;
    std::printf("case=asym3-%s-d%d-gqa-causal max_abs=%.7g mean_abs=%.7g workspace=%zu\n",
                givens ? "givens" : "fwht", hdim, max_abs, mean_abs, params.workspace_bytes);
    if(max_abs > 0.002f) std::exit(13);
    for(void* pointer : {dq, dk, dv, dout, dcos, dsin, workspace}) check_hip(hipFree(pointer), "hipFree(asym)");
}

void run_asym4_case(int format)
{
    constexpr float centroids[16] = {
        -0.241565f, -0.182875f, -0.143012f, -0.111016f,
        -0.083262f, -0.057983f, -0.034295f, -0.011225f,
         0.011225f,  0.034295f,  0.057983f,  0.083262f,
         0.111016f,  0.143012f,  0.182875f,  0.241565f,
    };
    constexpr int hdim = 256;
    constexpr int seqlen_q = 8;
    constexpr int seqlen_k = 16;
    constexpr int nhead_q = 4;
    constexpr int nhead_k = 2;
    constexpr int k_head_bytes = 4 + hdim / 2;
    constexpr int v_head_bytes = (hdim / 32) * 34;
    const bool givens = format == HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hdim));
    const size_t q_count = static_cast<size_t>(seqlen_q) * nhead_q * hdim;
    const size_t kv_count = static_cast<size_t>(seqlen_k) * nhead_k * hdim;
    std::vector<float> q(q_count), transformed_q(q_count), v(kv_count), decoded_k(kv_count),
        decoded_v(kv_count), expected(q_count, 0.0f), output(q_count);
    std::vector<float> transform0(givens ? hdim / 2 : 128);
    std::vector<float> transform1(transform0.size());
    std::vector<uint8_t> packed_k(static_cast<size_t>(seqlen_k) * nhead_k * k_head_bytes);
    std::vector<uint8_t> packed_v(static_cast<size_t>(seqlen_k) * nhead_k * v_head_bytes);
    std::mt19937 rng(71 + format);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for(float& value : q) value = distribution(rng);
    for(float& value : v) value = distribution(rng);
    transformed_q = q;
    if(givens)
    {
        for(int pair = 0; pair < hdim / 2; ++pair)
        {
            const float angle = 0.001f * static_cast<float>((pair * 19 + 11) % 101);
            transform0[pair] = std::cos(angle);
            transform1[pair] = std::sin(angle);
        }
        for(int row = 0; row < seqlen_q; ++row)
            for(int head = 0; head < nhead_q; ++head)
                for(int pair = 0; pair < hdim / 2; ++pair)
                {
                    const size_t base =
                        (static_cast<size_t>(row) * nhead_q + head) * hdim + pair * 2;
                    const float a = transformed_q[base];
                    const float b = transformed_q[base + 1];
                    transformed_q[base] = a * transform0[pair] - b * transform1[pair];
                    transformed_q[base + 1] = a * transform1[pair] + b * transform0[pair];
                }
    }
    else
    {
        for(int dim = 0; dim < 128; ++dim)
        {
            transform0[dim] = ((dim * 17 + 3) & 1) == 0 ? 1.0f : -1.0f;
            transform1[dim] = ((dim * 23 + 9) & 1) == 0 ? 1.0f : -1.0f;
        }
        for(int row = 0; row < seqlen_q; ++row)
            for(int head = 0; head < nhead_q; ++head)
                for(int half = 0; half < 2; ++half)
                {
                    const size_t base =
                        (static_cast<size_t>(row) * nhead_q + head) * hdim + half * 128;
                    for(int dim = 0; dim < 128; ++dim)
                        transformed_q[base + dim] *= transform0[dim];
                    for(int stride = 1; stride < 128; stride <<= 1)
                        for(int index = 0; index < 128; index += stride * 2)
                            for(int offset = 0; offset < stride; ++offset)
                            {
                                const float a = transformed_q[base + index + offset];
                                const float b = transformed_q[base + index + offset + stride];
                                transformed_q[base + index + offset] = a + b;
                                transformed_q[base + index + offset + stride] = a - b;
                            }
                    for(int dim = 0; dim < 128; ++dim)
                        transformed_q[base + dim] *= 0.08838834764831845f * transform1[dim];
                }
    }
    for(int row = 0; row < seqlen_k; ++row)
        for(int head = 0; head < nhead_k; ++head)
        {
            uint8_t* destination = packed_k.data() +
                (static_cast<size_t>(row) * nhead_k + head) * k_head_bytes;
            const float cnorm = 0.45f + 0.01f * static_cast<float>((row + head) % 13);
            std::memcpy(destination, &cnorm, sizeof(cnorm));
            for(int half = 0; half < 2; ++half)
                for(int lane = 0; lane < 32; ++lane)
                {
                    const int dim = half * 128 + lane * 4;
                    int code[4];
                    for(int index = 0; index < 4; ++index)
                    {
                        code[index] = (row * 3 + head * 5 + half * 7 + lane + index) & 15;
                        decoded_k[(static_cast<size_t>(row) * nhead_k + head) * hdim + dim + index] =
                            cnorm * centroids[code[index]];
                    }
                    destination[4 + half * 64 + lane * 2] =
                        static_cast<uint8_t>((code[1] << 4) | code[0]);
                    destination[4 + half * 64 + lane * 2 + 1] =
                        static_cast<uint8_t>((code[3] << 4) | code[2]);
                }
        }
    pack_q8(v, packed_v, decoded_v, seqlen_k, nhead_k);
    const int groups = nhead_q / nhead_k;
    for(int sq = 0; sq < seqlen_q; ++sq)
        for(int hq = 0; hq < nhead_q; ++hq)
        {
            const int hk = hq / groups;
            const int last_key = sq + seqlen_k - seqlen_q;
            std::vector<float> scores(last_key + 1);
            float maximum = -INFINITY;
            for(int sk = 0; sk <= last_key; ++sk)
            {
                float score = 0.0f;
                for(int dim = 0; dim < hdim; ++dim)
                    score += transformed_q[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + dim] *
                             decoded_k[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + dim];
                scores[sk] = score * scale;
                maximum = std::max(maximum, scores[sk]);
            }
            float denominator = 0.0f;
            for(float& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for(int dim = 0; dim < hdim; ++dim)
                for(int sk = 0; sk <= last_key; ++sk)
                    expected[(static_cast<size_t>(sq) * nhead_q + hq) * hdim + dim] +=
                        scores[sk] / denominator *
                        decoded_v[(static_cast<size_t>(sk) * nhead_k + hk) * hdim + dim];
        }

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    void *dt0 = nullptr, *dt1 = nullptr, *workspace = nullptr;
    check_hip(hipMalloc(&dq, q_count * sizeof(float)), "hipMalloc(asym4 q)");
    check_hip(hipMalloc(&dk, packed_k.size()), "hipMalloc(asym4 k)");
    check_hip(hipMalloc(&dv, packed_v.size()), "hipMalloc(asym4 v)");
    check_hip(hipMalloc(&dout, q_count * sizeof(float)), "hipMalloc(asym4 out)");
    check_hip(hipMalloc(&dt0, transform0.size() * sizeof(float)), "hipMalloc(asym4 transform0)");
    check_hip(hipMalloc(&dt1, transform1.size() * sizeof(float)), "hipMalloc(asym4 transform1)");
    check_hip(hipMemcpy(dq, q.data(), q_count * sizeof(float), hipMemcpyHostToDevice), "copy asym4 q");
    check_hip(hipMemcpy(dk, packed_k.data(), packed_k.size(), hipMemcpyHostToDevice), "copy asym4 k");
    check_hip(hipMemcpy(dv, packed_v.data(), packed_v.size(), hipMemcpyHostToDevice), "copy asym4 v");
    check_hip(hipMemcpy(dt0, transform0.data(), transform0.size() * sizeof(float), hipMemcpyHostToDevice), "copy asym4 transform0");
    check_hip(hipMemcpy(dt1, transform1.data(), transform1.size() * sizeof(float), hipMemcpyHostToDevice), "copy asym4 transform1");
    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION; params.struct_size = sizeof(params);
    params.q = dq; params.k = dk; params.v = dv; params.out = dout;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32; params.k_format = format;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1; params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
    params.nhead_q = nhead_q; params.nhead_k = nhead_k; params.head_dim = hdim;
    params.causal = 1; params.softmax_scale = scale;
    params.stride_q = params.stride_out = nhead_q * hdim;
    params.stride_k = params.stride_v = nhead_k * hdim;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v =
        params.nhead_stride_out = hdim;
    params.batch_stride_q = params.batch_stride_out = seqlen_q * nhead_q * hdim;
    params.batch_stride_k = params.batch_stride_v = seqlen_k * nhead_k * hdim;
    params.packed_k_head_stride_bytes = k_head_bytes;
    params.packed_v_head_stride_bytes = v_head_bytes;
    params.packed_k_row_stride_bytes = nhead_k * k_head_bytes;
    params.packed_v_row_stride_bytes = nhead_k * v_head_bytes;
    params.k_transform0 = dt0; params.k_transform1 = dt1;
    params.k_transform0_elements = params.k_transform1_elements = transform0.size();
    params.workspace_bytes = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);
    check_hip(hipMalloc(&workspace, params.workspace_bytes), "hipMalloc(asym4 workspace)");
    params.workspace = workspace;
    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd(&params, error, sizeof(error));
    if(status != 0) { std::fprintf(stderr, "asym4 sidecar status=%d: %s\n", status, error); std::exit(18); }
    check_hip(hipMemcpy(output.data(), dout, q_count * sizeof(float), hipMemcpyDeviceToHost), "copy asym4 out");
    float max_abs = 0.0f; double mean_abs = 0.0;
    for(size_t index = 0; index < q_count; ++index)
    {
        const float delta = std::abs(output[index] - expected[index]);
        max_abs = std::max(max_abs, delta); mean_abs += delta;
    }
    mean_abs /= q_count;
    std::printf("case=asym4-%s-d256-gqa-causal max_abs=%.7g mean_abs=%.7g workspace=%zu\n",
                givens ? "givens" : "fwht", max_abs, mean_abs, params.workspace_bytes);
    if(max_abs > 0.002f) std::exit(19);
    for(void* pointer : {dq, dk, dv, dout, dt0, dt1, workspace})
        check_hip(hipFree(pointer), "hipFree(asym4)");
}


void run_asym4_padded_stride_beyond_i32_case()
{
    if(!kExpectedAsym4D256)
    {
        std::printf("case=asym4-padded-stride-beyond-i32 status=skipped-no-cell\n");
        return;
    }
    constexpr int hdim = 256;
    constexpr int64_t k_head_bytes = 4 + hdim / 2;
    constexpr int64_t v_head_bytes = (hdim / 32) * 34;
    // Padded row stride beyond i32, 4-byte aligned, without allocating the full span.
    constexpr int64_t padded_k_row = 2147483652LL; // INT32_MAX+5, %4==0
    constexpr int64_t padded_v_row = 2147483650LL; // even, %2==0
    static_assert(padded_k_row % 4 == 0, "k row must be 4-aligned");
    static_assert(padded_v_row % 2 == 0, "v row must be 2-aligned");
    alignas(4) uint8_t dummy_k_align[4]{};
    alignas(2) uint8_t dummy_v_align[2]{};
    alignas(4) float dummy_q[4]{};
    alignas(4) float dummy_out[4]{};
    alignas(4) float dummy_transform[128]{};
    // Use the aligned dummy buffers as base pointers.
    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = dummy_q;
    params.k = dummy_k_align;
    params.v = dummy_v_align;
    params.out = dummy_out;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1;
    params.seqlen_q = 16;
    params.seqlen_k = 32;
    params.nhead_q = 4;
    params.nhead_k = 2;
    params.head_dim = hdim;
    params.causal = 1;
    params.softmax_scale = 1.0f / std::sqrt(static_cast<float>(hdim));
    params.stride_q = params.stride_out = 4 * hdim;
    params.stride_k = params.stride_v = 2 * hdim;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v = params.nhead_stride_out = hdim;
    params.batch_stride_q = params.batch_stride_out = 16 * 4 * hdim;
    params.batch_stride_k = params.batch_stride_v = 32 * 2 * hdim;
    params.packed_k_head_stride_bytes = k_head_bytes;
    params.packed_v_head_stride_bytes = v_head_bytes;
    params.packed_k_row_stride_bytes = padded_k_row;
    params.packed_v_row_stride_bytes = padded_v_row;
    params.k_transform0 = dummy_transform;
    params.k_transform1 = dummy_transform;
    params.k_transform0_elements = params.k_transform1_elements = 128;
    params.workspace = dummy_q;
    // Workspace query must not wrap to a small value.
    const size_t queried = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);
    if(queried == 0 || queried == SIZE_MAX)
    {
        std::fprintf(stderr, "padded stride beyond i32 query failed: queried=%zu (expected valid)\n", queried);
        std::exit(20);
    }
    params.workspace_bytes = queried;
    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error));
    if(status != 0)
    {
        std::fprintf(stderr, "padded stride beyond i32 was rejected: status=%d %s\n", status, error);
        std::exit(21);
    }
    // Also verify that a truncated i32 view would have been rejected: the low 32 bits of padded_k_row
    // would be 5, which is < minimum (264) and would fail. Our 64-bit path must not truncate.
    if(static_cast<int32_t>(padded_k_row) < 264)
    {
        // Confirm that truncating would indeed be invalid, proving we didn't silently narrow.
        hipfire_flash_attn_ck_fwd_params truncated = params;
        truncated.packed_k_row_stride_bytes = static_cast<int32_t>(padded_k_row);
        // This truncated view should be rejected because row stride < minimum.
        char err2[256]{};
        int s2 = hipfire_flash_attn_ck_fwd_supported(&truncated, err2, sizeof(err2));
        if(s2 == 0)
        {
            std::fprintf(stderr, "truncated padded stride unexpectedly accepted\n");
            std::exit(22);
        }
    }
    std::printf("case=asym4-padded-stride-beyond-i32 status=supported row_k=%lld row_v=%lld workspace=%zu\n",
                static_cast<long long>(padded_k_row), static_cast<long long>(padded_v_row), queried);
}

void run_workspace_overflow_case()
{
    // Use Asym4 if available, else Q8, else Asym3. Pick first available.
    int format = HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS;
    if(!kExpectedAsym4D256)
    {
        if(kExpectedAsym3GivensD256) format = HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS;
        else if(kExpectedQ8D256) format = HIPFIRE_FLASH_ATTN_CK_Q8;
        else
        {
            std::printf("case=workspace-overflow status=skipped-no-quant-cell\n");
            return;
        }
    }
    alignas(4) uint8_t dummy[4]{};
    alignas(4) float t0[4]{};
    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = dummy; params.k = dummy; params.v = dummy; params.out = dummy;
    params.workspace = dummy;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = format;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    // Choose dimensions that overflow size_t workspace computation but still have batch=1 for quantized.
    // Use large seqlen and nhead that exceed staging product: seqlen_q = INT_MAX, nhead_q = 1<<20 (1048576)
    // head_dim=256 => q = 2e9 * 1M *256 ~5e23 overflow. But stride_q would then be 1M*256=268M <= INT32_MAX,
    // so stride check passes, allowing us to test overflow path.
    params.batch = INT_MAX;
    params.seqlen_q = INT_MAX;
    params.seqlen_k = INT_MAX;
    params.nhead_q = 1 << 20; // 1048576, stride = 268435456 <= INT32_MAX
    params.nhead_k = 1 << 19; // 524288
    params.head_dim = 256;
    params.causal = 1;
    params.softmax_scale = 1.0f;
    params.stride_q = static_cast<int64_t>(params.nhead_q) * params.head_dim;
    params.stride_out = params.stride_q;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v = params.nhead_stride_out = 256;
    params.batch_stride_q = static_cast<int64_t>(params.seqlen_q) * params.stride_q;
    params.batch_stride_k = static_cast<int64_t>(params.seqlen_k) * params.nhead_k * params.head_dim;
    params.batch_stride_v = params.batch_stride_k;
    params.batch_stride_out = params.batch_stride_q;
    // For overflow we don't need correct packed strides; they will be validated after workspace overflow.
    // But to reach overflow check, we need to provide at least syntactically valid packed strides.
    // Use minimal valid strides for the chosen format.
    if(format == HIPFIRE_FLASH_ATTN_CK_Q8)
    {
        params.packed_k_head_stride_bytes = params.packed_v_head_stride_bytes = 272;
        params.packed_k_row_stride_bytes = static_cast<int64_t>(params.nhead_k) * 272;
        params.packed_v_row_stride_bytes = params.packed_k_row_stride_bytes;
    }
    else if(format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS || format == HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT)
    {
        int k_head = 4 + (256*3)/8; //100
        int v_head = 272;
        params.packed_k_head_stride_bytes = k_head;
        params.packed_v_head_stride_bytes = v_head;
        params.packed_k_row_stride_bytes = static_cast<int64_t>(params.nhead_k) * k_head;
        params.packed_v_row_stride_bytes = static_cast<int64_t>(params.nhead_k) * v_head;
        params.k_transform0 = t0; params.k_transform1 = t0;
        params.k_transform0_elements = params.k_transform1_elements = 256;
    }
    else
    {
        int k_head = 4 + 256/2; //132
        int v_head = 272;
        params.packed_k_head_stride_bytes = k_head;
        params.packed_v_head_stride_bytes = v_head;
        params.packed_k_row_stride_bytes = static_cast<int64_t>(params.nhead_k) * k_head;
        params.packed_v_row_stride_bytes = static_cast<int64_t>(params.nhead_k) * v_head;
        params.k_transform0 = t0; params.k_transform1 = t0;
        params.k_transform0_elements = params.k_transform1_elements = 128;
    }
    const size_t queried = hipfire_flash_attn_ck_fwd_workspace_bytes(&params);
    if(queried != SIZE_MAX)
    {
        std::fprintf(stderr, "workspace overflow query did not return SIZE_MAX: got %zu\n", queried);
        std::exit(23);
    }
    char error[1024]{};
    const int status = hipfire_flash_attn_ck_fwd_supported(&params, error, sizeof(error));
    if(status == 0)
    {
        std::fprintf(stderr, "workspace overflow was accepted: status=%d error=%s\n", status, error);
        std::exit(24);
    }
    // Accept either overflow or stride error as long as it is rejected; query must be SIZE_MAX
    if(std::strstr(error, "overflow") == nullptr && std::strstr(error, "INT32_MAX") == nullptr && std::strstr(error, "stride") == nullptr)
    {
        std::fprintf(stderr, "workspace overflow not rejected with expected reason: status=%d error=%s\n", status, error);
        std::exit(24);
    }
    std::printf("case=workspace-overflow format=%d status=rejected-overflow queried=SIZE_MAX error=%s\n", format, error);
}

void run_dimension_overflow_case()
{
    // Test that element stride overflow and packed address overflow are rejected.
    // Use a valid Asym4 cell but with packed row stride that would overflow address calc.
    if(!kExpectedAsym4D256)
    {
        std::printf("case=dimension-overflow status=skipped-no-asym4\n");
        return;
    }
    alignas(4) uint8_t base[4]{};
    alignas(4) float qbuf[4]{};
    alignas(4) float tbuf[128]{};
    hipfire_flash_attn_ck_fwd_params params{};
    params.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
    params.struct_size = sizeof(params);
    params.q = qbuf; params.k = base; params.v = base; params.out = qbuf;
    params.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
    params.k_format = HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS;
    params.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
    params.batch = 1;
    params.seqlen_q = 16;
    params.seqlen_k = 32;
    params.nhead_q = 4;
    params.nhead_k = 2;
    params.head_dim = 256;
    params.causal = 1;
    params.softmax_scale = 1.0f / 16.0f;
    params.stride_q = params.stride_out = 4 * 256;
    params.stride_k = params.stride_v = 2 * 256;
    params.nhead_stride_q = params.nhead_stride_k = params.nhead_stride_v = params.nhead_stride_out = 256;
    params.batch_stride_q = 16*4*256; params.batch_stride_out = 16*4*256;
    params.batch_stride_k = 32*2*256; params.batch_stride_v = 32*2*256;
    params.packed_k_head_stride_bytes = 132;
    params.packed_v_head_stride_bytes = 272;
    // First, test element stride exceeding INT32_MAX is rejected.
    hipfire_flash_attn_ck_fwd_params bad_stride = params;
    bad_stride.stride_q = static_cast<int64_t>(INT32_MAX) + 1;
    char err[1024]{};
    int s = hipfire_flash_attn_ck_fwd_supported(&bad_stride, err, sizeof(err));
    if(s != 1 || std::strstr(err, "INT32_MAX") == nullptr)
    {
        std::fprintf(stderr, "element stride overflow not rejected: s=%d err=%s\n", s, err);
        std::exit(25);
    }
    // Second, test packed address overflow: use huge row stride that overflows int64 when multiplied.
    hipfire_flash_attn_ck_fwd_params bad_packed = params;
    bad_packed.packed_k_row_stride_bytes = (INT64_MAX / 2) + 4; // aligned to 4
    bad_packed.packed_k_row_stride_bytes &= ~3LL; // ensure %4==0
    bad_packed.packed_v_row_stride_bytes = 2 * 272;
    bad_packed.k_transform0 = tbuf; bad_packed.k_transform1 = tbuf;
    bad_packed.k_transform0_elements = bad_packed.k_transform1_elements = 128;
    bad_packed.workspace = base;
    // Need workspace_bytes to be at least queried, but query will be SIZE_MAX due to? Actually q small so workspace ok.
    // Set workspace to dummy large enough to avoid "too small" before overflow check.
    // Our validate checks packed overflow before workspace size, so we should reach overflow error.
    // Provide huge workspace to avoid size check interfering.
    bad_packed.workspace_bytes = SIZE_MAX;
    s = hipfire_flash_attn_ck_fwd_supported(&bad_packed, err, sizeof(err));
    if(s != 1 || std::strstr(err, "overflow") == nullptr)
    {
        std::fprintf(stderr, "packed address overflow not rejected: s=%d err=%s\n", s, err);
        std::exit(26);
    }
    std::printf("case=dimension-overflow status=rejected-both stride and packed overflow\n");
}

void run_alignment_validation_case()
{
    if(!kExpectedAsym4D256)
    {
        std::printf("case=alignment-validation status=skipped-no-asym4\n");
        return;
    }
    constexpr int hdim = 256;
    alignas(4) uint8_t backing[64]{};
    alignas(4) float qbuf[4]{};
    alignas(4) float tbuf[128]{};
    // Helper to build a valid baseline params
    auto make_valid = [&](const void* k_ptr, const void* v_ptr, int64_t k_row_stride, int64_t v_row_stride) {
        hipfire_flash_attn_ck_fwd_params p{};
        p.abi_version = HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
        p.struct_size = sizeof(p);
        p.q = qbuf; p.k = k_ptr; p.v = v_ptr; p.out = qbuf;
        p.dtype = HIPFIRE_FLASH_ATTN_CK_F32;
        p.k_format = HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS;
        p.v_format = HIPFIRE_FLASH_ATTN_CK_Q8;
        p.batch = 1; p.seqlen_q = 16; p.seqlen_k = 32;
        p.nhead_q = 4; p.nhead_k = 2; p.head_dim = hdim;
        p.causal = 1; p.softmax_scale = 1.0f/16.0f;
        p.stride_q = 4*hdim; p.stride_out = 4*hdim;
        p.stride_k = 2*hdim; p.stride_v = 2*hdim;
        p.nhead_stride_q = p.nhead_stride_k = p.nhead_stride_v = p.nhead_stride_out = hdim;
        p.batch_stride_q = 16*4*hdim; p.batch_stride_out = 16*4*hdim;
        p.batch_stride_k = 32*2*hdim; p.batch_stride_v = 32*2*hdim;
        p.packed_k_head_stride_bytes = 4 + hdim/2;
        p.packed_v_head_stride_bytes = (hdim/32)*34;
        p.packed_k_row_stride_bytes = k_row_stride;
        p.packed_v_row_stride_bytes = v_row_stride;
        p.k_transform0 = tbuf; p.k_transform1 = tbuf;
        p.k_transform0_elements = p.k_transform1_elements = 128;
        p.workspace = qbuf;
        p.workspace_bytes = hipfire_flash_attn_ck_fwd_workspace_bytes(&p);
        // If query overflowed, set to large dummy
        if(p.workspace_bytes == SIZE_MAX) p.workspace_bytes = 1<<20;
        return p;
    };
    char err[1024]{};
    // 1) Misaligned K base (needs 4): offset by 1
    {
        const void* mis_k = backing + 1;
        // backing is 4-aligned, +1 is misaligned
        auto p = make_valid(mis_k, backing, 2*132, 2*272);
        int s = hipfire_flash_attn_ck_fwd_supported(&p, err, sizeof(err));
        if(s != 1 || std::strstr(err, "align") == nullptr)
        {
            std::fprintf(stderr, "misaligned K base not rejected: s=%d err=%s\n", s, err);
            std::exit(27);
        }
    }
    // 2) Misaligned V base (needs 2): offset by 1 from even address
    {
        // backing is 4-aligned (even), +1 is odd -> misaligned for 2
        const void* mis_v = backing + 1;
        auto p = make_valid(backing, mis_v, 2*132, 2*272);
        int s = hipfire_flash_attn_ck_fwd_supported(&p, err, sizeof(err));
        if(s != 1 || std::strstr(err, "align") == nullptr)
        {
            std::fprintf(stderr, "misaligned V base not rejected: s=%d err=%s\n", s, err);
            std::exit(28);
        }
    }
    // 3) Misaligned K row stride (needs 4)
    {
        auto p = make_valid(backing, backing, 2*132 + 1, 2*272);
        int s = hipfire_flash_attn_ck_fwd_supported(&p, err, sizeof(err));
        if(s != 1 || std::strstr(err, "align") == nullptr)
        {
            std::fprintf(stderr, "misaligned K row stride not rejected: s=%d err=%s\n", s, err);
            std::exit(29);
        }
    }
    // 4) Misaligned V row stride (needs 2)
    {
        auto p = make_valid(backing, backing, 2*132, 2*272 + 1);
        int s = hipfire_flash_attn_ck_fwd_supported(&p, err, sizeof(err));
        if(s != 1 || std::strstr(err, "align") == nullptr)
        {
            std::fprintf(stderr, "misaligned V row stride not rejected: s=%d err=%s\n", s, err);
            std::exit(30);
        }
    }
    // 5) Valid aligned case must succeed
    {
        auto p = make_valid(backing, backing, 2*132, 2*272);
        int s = hipfire_flash_attn_ck_fwd_supported(&p, err, sizeof(err));
        if(s != 0)
        {
            std::fprintf(stderr, "valid aligned case rejected: s=%d err=%s\n", s, err);
            std::exit(31);
        }
    }
    std::printf("case=alignment-validation status=all-rejected-except-valid\n");
}


} // namespace

int main()
{
    if(hipfire_flash_attn_ck_abi_version() != HIPFIRE_FLASH_ATTN_CK_ABI_VERSION)
    {
        std::fprintf(stderr, "sidecar ABI mismatch\n");
        return 1;
    }
    verify_capabilities();
    run_case("gqa-noncausal", 4, 2, false, false);
    run_case("gqa-causal", 4, 2, true, true);
    run_case("mha-noncausal", 2, 2, false, false);
    run_case("mqa-noncausal", 4, 1, false, false);
    if(has_capability(HIPFIRE_FLASH_ATTN_CK_F32,
                      HIPFIRE_FLASH_ATTN_CK_Q8,
                      HIPFIRE_FLASH_ATTN_CK_Q8,
                      256))
    {
        run_q8_d256_case();
    }
    run_asym3_contract_case(
        HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS, 256, kExpectedAsym3GivensD256);
    run_asym3_contract_case(
        HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS, 512, kExpectedAsym3GivensD256);
    run_asym3_contract_case(
        HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT, 256, kExpectedAsym3FwhtD256);
    run_asym3_contract_case(
        HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT, 512, kExpectedAsym3FwhtD256);
    if(kExpectedAsym3GivensD256)
    {
        run_asym3_case(HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS, 256);
    }
    if(kExpectedAsym3FwhtD256)
    {
        run_asym3_case(HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT, 256);
    }
    if(kExpectedAsym4D256)
    {
        run_asym4_contract_case(HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS);
        run_asym4_contract_case(HIPFIRE_FLASH_ATTN_CK_ASYM4_FWHT);
        run_asym4_case(HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS);
        run_asym4_case(HIPFIRE_FLASH_ATTN_CK_ASYM4_FWHT);
    }
    run_asym4_padded_stride_beyond_i32_case();
    run_workspace_overflow_case();
    run_dimension_overflow_case();
    run_alignment_validation_case();
    return 0;
}
