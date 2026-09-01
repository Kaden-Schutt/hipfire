// SPDX-License-Identifier: Apache-2.0

#include "hipfire_flash_attn_ck.h"

#include "fmha_fwd.hpp"
#include "mask.hpp"
#include "turbo_common.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

static_assert(sizeof(hipfire_flash_attn_ck_fwd_params) == 272,
              "FlashAttention CK ABI parameter layout changed");
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, q) == 8);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, workspace) == 40);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, dtype) == 64);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, softmax_scale) == 104);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, stride_q) == 112);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, batch_stride_out) == 200);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, packed_k_row_stride_bytes) == 208);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, packed_k_head_stride_bytes) == 224);
static_assert(offsetof(hipfire_flash_attn_ck_fwd_params, k_transform0) == 240);
static_assert(sizeof(hipfire_flash_attn_ck_capability) == 32);

namespace {

constexpr size_t kWorkspaceAlignment = 256;

#if defined(HIPFIRE_CK_TARGET_GFX1100)
constexpr bool kHasQ8D256 = true;
#else
constexpr bool kHasQ8D256 = false;
#endif

#if defined(HIPFIRE_CK_TARGET_GFX1100) || defined(HIPFIRE_CK_TARGET_GFX1201)
constexpr bool kHasAsym3GivensD256 = true;
constexpr bool kHasAsym3FwhtD256 = true;
#else
constexpr bool kHasAsym3GivensD256 = false;
constexpr bool kHasAsym3FwhtD256 = false;
#endif

inline bool checked_mul_size_t(size_t a, size_t b, size_t* out)
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)
    return !__builtin_mul_overflow(a, b, out);
#endif
#endif
    if(a == 0 || b == 0)
    {
        *out = 0;
        return true;
    }
    if(a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
}

inline bool checked_add_size_t(size_t a, size_t b, size_t* out)
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)
    return !__builtin_add_overflow(a, b, out);
#endif
#endif
    if(a > SIZE_MAX - b) return false;
    *out = a + b;
    return true;
}

inline bool checked_mul_int64(int64_t a, int64_t b, int64_t* out)
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)
    return !__builtin_mul_overflow(a, b, out);
#endif
#endif
    __int128 prod = static_cast<__int128>(a) * static_cast<__int128>(b);
    if(prod < INT64_MIN || prod > INT64_MAX) return false;
    *out = static_cast<int64_t>(prod);
    return true;
}

inline bool checked_add_int64(int64_t a, int64_t b, int64_t* out)
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)
    return !__builtin_add_overflow(a, b, out);
#endif
#endif
    __int128 sum = static_cast<__int128>(a) + static_cast<__int128>(b);
    if(sum < INT64_MIN || sum > INT64_MAX) return false;
    *out = static_cast<int64_t>(sum);
    return true;
}

bool align_up_checked(size_t value, size_t* out)
{
    size_t tmp;
    if(!checked_add_size_t(value, kWorkspaceAlignment - 1, &tmp)) return false;
    *out = tmp & ~(kWorkspaceAlignment - 1);
    return true;
}

size_t align_up(size_t value)
{
    size_t out;
    if(!align_up_checked(value, &out)) return SIZE_MAX;
    return out;
}

bool checked_staging_workspace_bytes(const hipfire_flash_attn_ck_fwd_params* p, size_t* out)
{
    if(p == nullptr) return false;
    if(p->batch <= 0 || p->seqlen_q <= 0 || p->seqlen_k <= 0 ||
       p->nhead_q <= 0 || p->nhead_k <= 0 || p->head_dim <= 0)
        return false;
    size_t q, kv;
    if(!checked_mul_size_t(static_cast<size_t>(p->batch), static_cast<size_t>(p->seqlen_q), &q)) return false;
    if(!checked_mul_size_t(q, static_cast<size_t>(p->nhead_q), &q)) return false;
    if(!checked_mul_size_t(q, static_cast<size_t>(p->head_dim), &q)) return false;
    if(!checked_mul_size_t(static_cast<size_t>(p->batch), static_cast<size_t>(p->seqlen_k), &kv)) return false;
    if(!checked_mul_size_t(kv, static_cast<size_t>(p->nhead_k), &kv)) return false;
    if(!checked_mul_size_t(kv, static_cast<size_t>(p->head_dim), &kv)) return false;
    size_t q_bytes, kv_bytes;
    if(!checked_mul_size_t(q, sizeof(__half), &q_bytes)) return false;
    if(!checked_mul_size_t(kv, sizeof(__half), &kv_bytes)) return false;
    size_t aligned_q, aligned_kv;
    if(!align_up_checked(q_bytes, &aligned_q)) return false;
    if(!align_up_checked(kv_bytes, &aligned_kv)) return false;
    size_t total;
    size_t twice_kv;
    if(!checked_mul_size_t(aligned_kv, 2, &twice_kv)) return false;
    if(!checked_add_size_t(aligned_q, twice_kv, &total)) return false;
    if(!checked_add_size_t(total, aligned_q, &total)) return false;
    *out = total;
    return true;
}

size_t staging_workspace_bytes(const hipfire_flash_attn_ck_fwd_params* p)
{
    size_t out;
    if(!checked_staging_workspace_bytes(p, &out)) return SIZE_MAX;
    return out;
}

bool is_q8_cell(const hipfire_flash_attn_ck_fwd_params* p)
{
    return p->dtype == HIPFIRE_FLASH_ATTN_CK_F32 &&
           p->k_format == HIPFIRE_FLASH_ATTN_CK_Q8 &&
           p->v_format == HIPFIRE_FLASH_ATTN_CK_Q8;
}

bool is_asym3_cell(const hipfire_flash_attn_ck_fwd_params* p)
{
    return p->dtype == HIPFIRE_FLASH_ATTN_CK_F32 &&
           (p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS ||
            p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT) &&
           p->v_format == HIPFIRE_FLASH_ATTN_CK_Q8;
}

bool is_asym3_givens_cell(const hipfire_flash_attn_ck_fwd_params* p)
{
    return is_asym3_cell(p) && p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS &&
           p->head_dim == 256;
}

bool is_asym3_fwht_cell(const hipfire_flash_attn_ck_fwd_params* p)
{
    return is_asym3_cell(p) && p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT &&
           p->head_dim == 256;
}

bool is_asym3_execution_cell(const hipfire_flash_attn_ck_fwd_params* p)
{
    return is_asym3_givens_cell(p) || is_asym3_fwht_cell(p);
}

bool is_asym4_contract(const hipfire_flash_attn_ck_fwd_params* p)
{
    return p->dtype == HIPFIRE_FLASH_ATTN_CK_F32 &&
           (p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS ||
            p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM4_FWHT) &&
           p->v_format == HIPFIRE_FLASH_ATTN_CK_Q8;
}

bool is_asym4_execution_cell(const hipfire_flash_attn_ck_fwd_params* p)
{
#if defined(HIPFIRE_CK_TARGET_GFX1100) || defined(HIPFIRE_CK_TARGET_GFX1201)
    return is_asym4_contract(p) && p->head_dim == 256;
#else
    (void)p;
    return false;
#endif
}


__global__ void convert_f32_to_f16(const float* input, __half* output, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count) output[index] = __float2half_rn(input[index]);
}

__global__ void convert_f16_to_f32(const __half* input, float* output, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count) output[index] = __half2float(input[index]);
}

__global__ void decode_q8_kv_d256(const uint8_t* packed_k,
                                   const uint8_t* packed_v,
                                   __half* dense_k,
                                   __half* dense_v,
                                   int rows,
                                   int kv_heads,
                                   int64_t k_row_stride_bytes,
                                   int64_t v_row_stride_bytes)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= kv_heads || lane >= 32) return;

    const int block = lane >> 2;
    const int lane_in_block = lane & 3;
    const uint8_t* k_block = packed_k + static_cast<size_t>(row) * k_row_stride_bytes +
                             head * 272 + block * 34;
    const uint8_t* v_block = packed_v + static_cast<size_t>(row) * v_row_stride_bytes +
                             head * 272 + block * 34;
    const float k_scale = __half2float(*reinterpret_cast<const __half*>(k_block));
    const float v_scale = __half2float(*reinterpret_cast<const __half*>(v_block));
    const int8_t* k_values = reinterpret_cast<const int8_t*>(k_block + 2) + lane_in_block * 8;
    const int8_t* v_values = reinterpret_cast<const int8_t*>(v_block + 2) + lane_in_block * 8;
    const size_t output = (static_cast<size_t>(row) * kv_heads + head) * 256 + lane * 8;
#pragma unroll
    for(int index = 0; index < 8; ++index)
    {
        dense_k[output + index] = __float2half_rn(k_scale * static_cast<float>(k_values[index]));
        dense_v[output + index] = __float2half_rn(v_scale * static_cast<float>(v_values[index]));
    }
}

__device__ __forceinline__ void givens_rotate(float& a, float& b, float c, float s)
{
    const float a2 = a * c - b * s;
    b = a * s + b * c;
    a = a2;
}

__global__ void transform_q_givens_f32_to_f16(const float* input,
                                               __half* output,
                                               int rows,
                                               int heads,
                                               int head_dim,
                                               const float* cos_theta,
                                               const float* sin_theta)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= heads || lane >= 32) return;
    const int chunks = head_dim / 256;
    for(int chunk = 0; chunk < chunks; ++chunk)
    {
        const int dim = chunk * 256 + lane * 8;
        const int block = chunk * 128 + lane * 4;
        const size_t base = (static_cast<size_t>(row) * heads + head) * head_dim + dim;
        float values[8];
#pragma unroll
        for(int i = 0; i < 8; ++i) values[i] = input[base + i];
#pragma unroll
        for(int pair = 0; pair < 4; ++pair)
            givens_rotate(values[pair * 2], values[pair * 2 + 1],
                          cos_theta[block + pair], sin_theta[block + pair]);
#pragma unroll
        for(int i = 0; i < 8; ++i) output[base + i] = __float2half_rn(values[i]);
    }
}

__global__ void transform_q_fwht_f32_to_f16(const float* input,
                                             __half* output,
                                             int rows,
                                             int heads,
                                             int head_dim,
                                             const float* signs1,
                                             const float* signs2)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= heads || lane >= 32 || head_dim != 256) return;
    const size_t base = (static_cast<size_t>(row) * heads + head) * head_dim + lane * 8;
    float v0 = input[base + 0], v1 = input[base + 1];
    float v2 = input[base + 2], v3 = input[base + 3];
    float v4 = input[base + 4], v5 = input[base + 5];
    float v6 = input[base + 6], v7 = input[base + 7];
    fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, lane);
    output[base + 0] = __float2half_rn(v0); output[base + 1] = __float2half_rn(v1);
    output[base + 2] = __float2half_rn(v2); output[base + 3] = __float2half_rn(v3);
    output[base + 4] = __float2half_rn(v4); output[base + 5] = __float2half_rn(v5);
    output[base + 6] = __float2half_rn(v6); output[base + 7] = __float2half_rn(v7);
}

__global__ void transform_q_fwht128x2_f32_to_f16(const float* input,
                                                  __half* output,
                                                  int rows,
                                                  int heads,
                                                  int head_dim,
                                                  const float* signs1,
                                                  const float* signs2)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= heads || lane >= 32 || head_dim != 256) return;
    const size_t head_base = (static_cast<size_t>(row) * heads + head) * head_dim;
    for(int half = 0; half < 2; ++half)
    {
        const size_t base = head_base + half * 128 + lane * 4;
        float v0 = input[base + 0], v1 = input[base + 1];
        float v2 = input[base + 2], v3 = input[base + 3];
        fwht_shfl_forward(v0, v1, v2, v3, signs1, signs2, lane);
        output[base + 0] = __float2half_rn(v0);
        output[base + 1] = __float2half_rn(v1);
        output[base + 2] = __float2half_rn(v2);
        output[base + 3] = __float2half_rn(v3);
    }
}

__global__ void decode_asym4_k(const uint8_t* packed,
                               __half* dense,
                               int rows,
                               int heads,
                               int64_t row_stride_bytes,
                               int64_t head_stride_bytes,
                               int head_dim)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= heads || lane >= 32 || head_dim != 256) return;
    const uint8_t* source = packed + static_cast<size_t>(row) * row_stride_bytes +
                            static_cast<size_t>(head) * head_stride_bytes;
    const float cnorm = *reinterpret_cast<const float*>(source);
    __half* destination = dense + (static_cast<size_t>(row) * heads + head) * head_dim;
    for(int half = 0; half < 2; ++half)
    {
        const int byte_offset = 4 + half * 64 + lane * 2;
        const uint8_t packed01 = source[byte_offset];
        const uint8_t packed23 = source[byte_offset + 1];
        const int dim = half * 128 + lane * 4;
        destination[dim + 0] = __float2half_rn(cnorm * TURBO_C4[packed01 & 0xf]);
        destination[dim + 1] = __float2half_rn(cnorm * TURBO_C4[packed01 >> 4]);
        destination[dim + 2] = __float2half_rn(cnorm * TURBO_C4[packed23 & 0xf]);
        destination[dim + 3] = __float2half_rn(cnorm * TURBO_C4[packed23 >> 4]);
    }
}

__global__ void decode_asym3_k_givens(const uint8_t* packed,
                                      __half* dense,
                                      int rows,
                                      int heads,
                                      int64_t row_stride_bytes,
                                      int64_t head_stride_bytes,
                                      int head_dim)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= heads || lane >= 32) return;
    const uint8_t* source = packed + static_cast<size_t>(row) * row_stride_bytes +
                            head * head_stride_bytes;
    const float cnorm = *reinterpret_cast<const float*>(source);
    const int chunks = head_dim / 256;
    for(int chunk = 0; chunk < chunks; ++chunk)
    {
        const uint8_t* bytes = source + 4 + chunk * 96 + lane * 3;
        const uint32_t codes = static_cast<uint32_t>(bytes[0]) |
                               (static_cast<uint32_t>(bytes[1]) << 8) |
                               (static_cast<uint32_t>(bytes[2]) << 16);
        const int dim = chunk * 256 + lane * 8;
        const size_t base = (static_cast<size_t>(row) * heads + head) * head_dim + dim;
#pragma unroll
        for(int i = 0; i < 8; ++i)
            dense[base + i] = __float2half_rn(cnorm * TURBO_C3_256[(codes >> (i * 3)) & 7]);
    }
}

__global__ void decode_q8(const uint8_t* packed,
                          __half* dense,
                          int rows,
                          int heads,
                          int64_t row_stride_bytes,
                          int64_t head_stride_bytes,
                          int head_dim)
{
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if(row >= rows || head >= heads || lane >= head_dim / 8) return;
    const int block = lane >> 2;
    const int lane_in_block = lane & 3;
    const uint8_t* source = packed + static_cast<size_t>(row) * row_stride_bytes +
                            head * head_stride_bytes + block * 34;
    const float scale = __half2float(*reinterpret_cast<const __half*>(source));
    const int8_t* values = reinterpret_cast<const int8_t*>(source + 2) + lane_in_block * 8;
    const size_t base = (static_cast<size_t>(row) * heads + head) * head_dim + lane * 8;
#pragma unroll
    for(int i = 0; i < 8; ++i)
        dense[base + i] = __float2half_rn(scale * static_cast<float>(values[i]));
}

void set_error(char* error, size_t capacity, const std::string& message)
{
    if(error == nullptr || capacity == 0)
    {
        return;
    }
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(error, message.data(), count);
    error[count] = '\0';
}

int validate(const hipfire_flash_attn_ck_fwd_params* p, char* error, size_t error_capacity)
{
    if(p == nullptr)
    {
        set_error(error, error_capacity, "params is null");
        return 1;
    }
    if(p->abi_version != HIPFIRE_FLASH_ATTN_CK_ABI_VERSION)
    {
        set_error(error, error_capacity, "unsupported ABI version");
        return 1;
    }
    if(p->struct_size < sizeof(hipfire_flash_attn_ck_fwd_params))
    {
        set_error(error, error_capacity, "parameter struct is too small");
        return 1;
    }
    if(p->q == nullptr || p->k == nullptr || p->v == nullptr || p->out == nullptr)
    {
        set_error(error, error_capacity, "q, k, v, and out must be non-null");
        return 1;
    }
    const bool dense = p->dtype == HIPFIRE_FLASH_ATTN_CK_F16 &&
                       p->k_format == HIPFIRE_FLASH_ATTN_CK_DENSE_F16 &&
                       p->v_format == HIPFIRE_FLASH_ATTN_CK_DENSE_F16;
    const bool q8 = is_q8_cell(p);
    const bool asym3 = is_asym3_cell(p);
    const bool asym4 = is_asym4_contract(p);
    if(!dense && !q8 && !asym3 && !asym4)
    {
        set_error(error, error_capacity, "unsupported dtype and K/V format cell");
        return 1;
    }
    if((q8 && !kHasQ8D256) ||
       (p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS && !kHasAsym3GivensD256) ||
       (p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT && !kHasAsym3FwhtD256))
    {
        set_error(error, error_capacity, "selected quantized cell is not published by this artifact");
        return 2;
    }
    if(p->workspace_bytes > 0 && p->workspace == nullptr)
    {
        set_error(error, error_capacity, "workspace must be non-null when workspace_bytes is non-zero");
        return 1;
    }
    if(p->batch <= 0 || p->seqlen_q <= 0 || p->seqlen_k <= 0 ||
       p->nhead_q <= 0 || p->nhead_k <= 0)
    {
        set_error(error, error_capacity, "batch, sequence lengths, and head counts must be positive");
        return 1;
    }
    if((dense && p->head_dim != 64) ||
       (q8 && p->head_dim != 256) ||
       (asym3 && p->head_dim != 256 && p->head_dim != 512) ||
       (asym4 && p->head_dim != 256))
    {
        set_error(error, error_capacity, "unsupported head dimension for selected cell");
        return 1;
    }
    if(p->nhead_q % p->nhead_k != 0)
    {
        set_error(error, error_capacity, "nhead_k must divide nhead_q");
        return 1;
    }
    if(p->causal != 0 && p->causal != 1)
    {
        set_error(error, error_capacity, "causal must be 0 or 1");
        return 1;
    }
    if(!(p->softmax_scale > 0.0f))
    {
        set_error(error, error_capacity, "softmax_scale must be positive");
        return 1;
    }
    const int64_t strides[] = {
        p->stride_q,
        p->stride_k,
        p->stride_v,
        p->stride_out,
        p->nhead_stride_q,
        p->nhead_stride_k,
        p->nhead_stride_v,
        p->nhead_stride_out,
        p->batch_stride_q,
        p->batch_stride_k,
        p->batch_stride_v,
        p->batch_stride_out,
    };
    for(const int64_t stride : strides)
    {
        if(stride <= 0 || stride > INT32_MAX)
        {
            set_error(error, error_capacity, "all element strides must be in (0, INT32_MAX]");
            return 1;
        }
    }
    if(q8 || asym3 || asym4)
    {
        const int64_t row_elements = static_cast<int64_t>(p->nhead_q) * p->head_dim;
        const int64_t batch_elements = static_cast<int64_t>(p->seqlen_q) * row_elements;
        if(p->stride_q != row_elements || p->stride_out != row_elements ||
           p->nhead_stride_q != p->head_dim || p->nhead_stride_out != p->head_dim ||
           p->batch_stride_q != batch_elements || p->batch_stride_out != batch_elements)
        {
            set_error(error, error_capacity, "packed staging requires contiguous Q and output");
            return 1;
        }
    }
    if(q8 || asym3 || asym4)
    {
        size_t required;
        if(!checked_staging_workspace_bytes(p, &required))
        {
            set_error(error, error_capacity, "workspace byte computation overflowed");
            return 1;
        }
        auto check_packed_overflow = [&](int64_t row_stride, int64_t head_stride, int64_t head_bytes) -> bool {
            if(row_stride < 0 || head_stride < 0 || head_bytes < 0) return false;
            if(p->seqlen_k > 1)
            {
                int64_t row_max;
                if(!checked_mul_int64(static_cast<int64_t>(p->seqlen_k - 1), row_stride, &row_max)) return false;
                int64_t total = row_max;
                if(p->nhead_k > 1)
                {
                    int64_t head_max;
                    if(!checked_mul_int64(static_cast<int64_t>(p->nhead_k - 1), head_stride, &head_max)) return false;
                    if(!checked_add_int64(total, head_max, &total)) return false;
                }
                if(!checked_add_int64(total, head_bytes, &total)) return false;
                if(total < 0) return false;
                if(static_cast<unsigned long long>(total) > SIZE_MAX) return false;
            }
            else if(p->nhead_k > 1)
            {
                int64_t head_max;
                if(!checked_mul_int64(static_cast<int64_t>(p->nhead_k - 1), head_stride, &head_max)) return false;
                int64_t total = head_max;
                if(!checked_add_int64(total, head_bytes, &total)) return false;
                if(total < 0) return false;
                if(static_cast<unsigned long long>(total) > SIZE_MAX) return false;
            }
            else
            {
                if(head_bytes < 0 || static_cast<unsigned long long>(head_bytes) > SIZE_MAX) return false;
            }
            return true;
        };
        auto is_aligned = [](const void* ptr, size_t alignment) -> bool {
            return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
        };
        if(!is_aligned(p->q, alignof(float)) || !is_aligned(p->out, alignof(float)))
        {
            set_error(error, error_capacity, "Q and output base pointers must be 4-byte aligned for quantized staging");
            return 1;
        }
        if(q8)
        {
            const int64_t head_bytes = (p->head_dim / 32) * 34;
            const int64_t minimum_row = static_cast<int64_t>(p->nhead_k) * head_bytes;
            if(p->batch != 1 || p->causal != 1 ||
               p->packed_k_head_stride_bytes != head_bytes ||
               p->packed_v_head_stride_bytes != head_bytes ||
               p->packed_k_row_stride_bytes < minimum_row ||
               p->packed_v_row_stride_bytes < minimum_row)
            {
                set_error(error, error_capacity, "Q8 D256 requires batch=1, causal, and valid packed row strides");
                return 1;
            }
            if(!check_packed_overflow(p->packed_k_row_stride_bytes, p->packed_k_head_stride_bytes, head_bytes) ||
               !check_packed_overflow(p->packed_v_row_stride_bytes, p->packed_v_head_stride_bytes, head_bytes))
            {
                set_error(error, error_capacity, "packed row/head stride computation overflowed");
                return 1;
            }
            if(!is_aligned(p->k, alignof(__half)) || !is_aligned(p->v, alignof(__half)))
            {
                set_error(error, error_capacity, "Q8 packed base pointers must be 2-byte aligned");
                return 1;
            }
            if(p->packed_k_row_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0 ||
               p->packed_v_row_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0 ||
               p->packed_k_head_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0 ||
               p->packed_v_head_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0)
            {
                set_error(error, error_capacity, "Q8 packed row/head strides must be 2-byte aligned");
                return 1;
            }
            if(p->workspace == nullptr || required == SIZE_MAX || p->workspace_bytes < required)
            {
                set_error(error, error_capacity, "caller workspace is too small for Q8 staging");
                return 1;
            }
        }
        if(asym3)
        {
            const int64_t k_head_bytes = 4 + (p->head_dim * 3) / 8;
            const int64_t v_head_bytes = (p->head_dim / 32) * 34;
            if(p->batch != 1 || p->causal != 1 ||
               p->packed_k_head_stride_bytes != k_head_bytes ||
               p->packed_v_head_stride_bytes != v_head_bytes ||
               p->packed_k_row_stride_bytes < static_cast<int64_t>(p->nhead_k) * k_head_bytes ||
               p->packed_v_row_stride_bytes < static_cast<int64_t>(p->nhead_k) * v_head_bytes)
            {
                set_error(error, error_capacity, "Asym3 requires batch=1, causal, and exact head strides");
                return 1;
            }
            if(!check_packed_overflow(p->packed_k_row_stride_bytes, p->packed_k_head_stride_bytes, k_head_bytes) ||
               !check_packed_overflow(p->packed_v_row_stride_bytes, p->packed_v_head_stride_bytes, v_head_bytes))
            {
                set_error(error, error_capacity, "packed row/head stride computation overflowed");
                return 1;
            }
            if(!is_aligned(p->k, alignof(float)))
            {
                set_error(error, error_capacity, "Asym3 K packed base pointer must be 4-byte aligned");
                return 1;
            }
            if(!is_aligned(p->v, alignof(__half)))
            {
                set_error(error, error_capacity, "Asym3 V packed base pointer must be 2-byte aligned");
                return 1;
            }
            if(p->packed_k_row_stride_bytes % static_cast<int64_t>(alignof(float)) != 0 ||
               p->packed_k_head_stride_bytes % static_cast<int64_t>(alignof(float)) != 0 ||
               p->packed_v_row_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0 ||
               p->packed_v_head_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0)
            {
                set_error(error, error_capacity, "Asym3 packed strides have insufficient alignment");
                return 1;
            }
            const int64_t transform_elements =
                p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS ? p->head_dim / 2 : 256;
            if(p->k_transform0 == nullptr || p->k_transform1 == nullptr ||
               p->k_transform0_elements < transform_elements ||
               p->k_transform1_elements < transform_elements)
            {
                set_error(error, error_capacity, "Asym3 transform metadata is missing or undersized");
                return 1;
            }
            if(!is_aligned(p->k_transform0, alignof(float)) || !is_aligned(p->k_transform1, alignof(float)))
            {
                set_error(error, error_capacity, "Asym3 transform base pointers must be 4-byte aligned");
                return 1;
            }
            if(!is_asym3_execution_cell(p))
            {
                set_error(error, error_capacity, "Asym3 packed layout is valid but has no CK execution cell");
                return 2;
            }
            if(p->workspace == nullptr || required == SIZE_MAX || p->workspace_bytes < required)
            {
                set_error(error, error_capacity, "caller workspace is too small for Asym3 staging");
                return 1;
            }
        }
        if(asym4)
        {
            const int64_t k_head_bytes = 4 + p->head_dim / 2;
            const int64_t v_head_bytes = (p->head_dim / 32) * 34;
            if(p->batch != 1 || p->causal != 1 ||
               p->packed_k_head_stride_bytes != k_head_bytes ||
               p->packed_v_head_stride_bytes != v_head_bytes ||
               p->packed_k_row_stride_bytes < static_cast<int64_t>(p->nhead_k) * k_head_bytes ||
               p->packed_v_row_stride_bytes < static_cast<int64_t>(p->nhead_k) * v_head_bytes)
            {
                set_error(error, error_capacity, "Asym4 requires batch=1, causal, and exact head strides");
                return 1;
            }
            if(!check_packed_overflow(p->packed_k_row_stride_bytes, p->packed_k_head_stride_bytes, k_head_bytes) ||
               !check_packed_overflow(p->packed_v_row_stride_bytes, p->packed_v_head_stride_bytes, v_head_bytes))
            {
                set_error(error, error_capacity, "packed row/head stride computation overflowed");
                return 1;
            }
            if(!is_aligned(p->k, alignof(float)))
            {
                set_error(error, error_capacity, "Asym4 K packed base pointer must be 4-byte aligned");
                return 1;
            }
            if(!is_aligned(p->v, alignof(__half)))
            {
                set_error(error, error_capacity, "Asym4 V packed base pointer must be 2-byte aligned");
                return 1;
            }
            if(p->packed_k_row_stride_bytes % static_cast<int64_t>(alignof(float)) != 0 ||
               p->packed_k_head_stride_bytes % static_cast<int64_t>(alignof(float)) != 0 ||
               p->packed_v_row_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0 ||
               p->packed_v_head_stride_bytes % static_cast<int64_t>(alignof(__half)) != 0)
            {
                set_error(error, error_capacity, "Asym4 packed strides have insufficient alignment");
                return 1;
            }
            const int64_t transform_elements =
                p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS ? p->head_dim / 2 : 128;
            if(p->k_transform0 == nullptr || p->k_transform1 == nullptr ||
               p->k_transform0_elements < transform_elements ||
               p->k_transform1_elements < transform_elements)
            {
                set_error(error, error_capacity, "Asym4 transform metadata is missing or undersized");
                return 1;
            }
            if(!is_aligned(p->k_transform0, alignof(float)) || !is_aligned(p->k_transform1, alignof(float)))
            {
                set_error(error, error_capacity, "Asym4 transform base pointers must be 4-byte aligned");
                return 1;
            }
            if(!is_asym4_execution_cell(p))
            {
                set_error(error, error_capacity, "Asym4 packed layout is valid but has no CK execution cell");
                return 2;
            }
            if(p->workspace == nullptr || required == SIZE_MAX || p->workspace_bytes < required)
            {
                set_error(error, error_capacity, "caller workspace is too small for Asym4 staging");
                return 1;
            }
        }
    }
    set_error(error, error_capacity, "");
    return 0;
}

} // namespace

extern "C" uint32_t hipfire_flash_attn_ck_abi_version(void)
{
    return HIPFIRE_FLASH_ATTN_CK_ABI_VERSION;
}

extern "C" size_t hipfire_flash_attn_ck_capabilities(
    hipfire_flash_attn_ck_capability* capabilities,
    size_t capacity)
{
    static const hipfire_flash_attn_ck_capability cells[] = {
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
#if defined(HIPFIRE_CK_TARGET_GFX1201)
        HIPFIRE_FLASH_ATTN_CK_GFX1201,
#elif defined(HIPFIRE_CK_TARGET_GFX1151)
        HIPFIRE_FLASH_ATTN_CK_GFX1151,
#else
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
#endif
        HIPFIRE_FLASH_ATTN_CK_F16,
        HIPFIRE_FLASH_ATTN_CK_DENSE_F16,
        HIPFIRE_FLASH_ATTN_CK_DENSE_F16,
        64,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
#if defined(HIPFIRE_CK_TARGET_GFX1100) || defined(HIPFIRE_CK_TARGET_GFX1201)
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
#if defined(HIPFIRE_CK_TARGET_GFX1201)
        HIPFIRE_FLASH_ATTN_CK_GFX1201,
#else
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
#endif
        HIPFIRE_FLASH_ATTN_CK_F32,
        HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        256,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
#if defined(HIPFIRE_CK_TARGET_GFX1201)
        HIPFIRE_FLASH_ATTN_CK_GFX1201,
#else
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
#endif
        HIPFIRE_FLASH_ATTN_CK_F32,
        HIPFIRE_FLASH_ATTN_CK_ASYM4_FWHT,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        256,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
#if defined(HIPFIRE_CK_TARGET_GFX1201)
        HIPFIRE_FLASH_ATTN_CK_GFX1201,
#else
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
#endif
        HIPFIRE_FLASH_ATTN_CK_F32,
        HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        256,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
#if defined(HIPFIRE_CK_TARGET_GFX1201)
        HIPFIRE_FLASH_ATTN_CK_GFX1201,
#else
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
#endif
        HIPFIRE_FLASH_ATTN_CK_F32,
        HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        256,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
#endif
#if defined(HIPFIRE_CK_TARGET_GFX1100)
    {
        HIPFIRE_FLASH_ATTN_CK_ABI_VERSION,
        sizeof(hipfire_flash_attn_ck_capability),
        HIPFIRE_FLASH_ATTN_CK_GFX1100,
        HIPFIRE_FLASH_ATTN_CK_F32,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        HIPFIRE_FLASH_ATTN_CK_Q8,
        256,
        HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL | HIPFIRE_FLASH_ATTN_CK_CAP_GQA,
    },
#endif
    };
    constexpr size_t count = sizeof(cells) / sizeof(cells[0]);
    if(capabilities != nullptr && capacity > 0)
    {
        const size_t written = std::min(capacity, count);
        std::memcpy(capabilities, cells, written * sizeof(cells[0]));
        return written;
    }
    return count;
}

extern "C" size_t hipfire_flash_attn_ck_fwd_workspace_bytes(
    const hipfire_flash_attn_ck_fwd_params* params)
{
    if(params == nullptr) return 0;
    const bool q8 = kHasQ8D256 && is_q8_cell(params);
    const bool asym3_givens = kHasAsym3GivensD256 && is_asym3_givens_cell(params);
    const bool asym3_fwht = kHasAsym3FwhtD256 && is_asym3_fwht_cell(params);
    const bool asym4 = is_asym4_execution_cell(params);
    if(!(q8 || asym3_givens || asym3_fwht || asym4))
        return 0;
    size_t out;
    if(!checked_staging_workspace_bytes(params, &out))
        return SIZE_MAX;
    return out;
}

extern "C" int hipfire_flash_attn_ck_fwd_supported(
    const hipfire_flash_attn_ck_fwd_params* params,
    char* error,
    size_t error_capacity)
{
    return validate(params, error, error_capacity);
}

extern "C" int hipfire_flash_attn_ck_fwd(
    const hipfire_flash_attn_ck_fwd_params* p,
    char* error,
    size_t error_capacity)
{
    if(const int status = validate(p, error, error_capacity); status != 0)
    {
        return status;
    }

    try
    {
        const bool q8 = is_q8_cell(p);
        const bool asym3_givens = is_asym3_givens_cell(p);
        const bool asym3_fwht = is_asym3_fwht_cell(p);
        const bool asym4_givens =
            is_asym4_execution_cell(p) && p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS;
        const bool asym4_fwht =
            is_asym4_execution_cell(p) && p->k_format == HIPFIRE_FLASH_ATTN_CK_ASYM4_FWHT;
        const void* q_ptr = p->q;
        const void* k_ptr = p->k;
        const void* v_ptr = p->v;
        void* out_ptr = p->out;
        int64_t stride_q = p->stride_q;
        int64_t stride_k = p->stride_k;
        int64_t stride_v = p->stride_v;
        int64_t stride_out = p->stride_out;
        int64_t nhead_stride_q = p->nhead_stride_q;
        int64_t nhead_stride_k = p->nhead_stride_k;
        int64_t nhead_stride_v = p->nhead_stride_v;
        int64_t nhead_stride_out = p->nhead_stride_out;
        int64_t batch_stride_q = p->batch_stride_q;
        int64_t batch_stride_k = p->batch_stride_k;
        int64_t batch_stride_v = p->batch_stride_v;
        int64_t batch_stride_out = p->batch_stride_out;
        __half* staged_out = nullptr;
        hipStream_t stream = reinterpret_cast<hipStream_t>(p->stream);
        size_t q_count, kv_count;
        {
            size_t tmp;
            if(!checked_mul_size_t(static_cast<size_t>(p->batch), static_cast<size_t>(p->seqlen_q), &tmp) ||
               !checked_mul_size_t(tmp, static_cast<size_t>(p->nhead_q), &tmp) ||
               !checked_mul_size_t(tmp, static_cast<size_t>(p->head_dim), &q_count))
            {
                set_error(error, error_capacity, "q_count computation overflowed");
                return 1;
            }
            size_t kv_rows;
            if(!checked_mul_size_t(static_cast<size_t>(p->seqlen_k), static_cast<size_t>(p->nhead_k), &kv_rows) ||
               !checked_mul_size_t(kv_rows, static_cast<size_t>(p->head_dim), &kv_count))
            {
                set_error(error, error_capacity, "kv_count computation overflowed");
                return 1;
            }
        }
        if(q8 || asym3_givens || asym3_fwht || asym4_givens || asym4_fwht)
        {
            size_t q_bytes, kv_bytes, aligned_q, aligned_kv;
            if(!checked_mul_size_t(q_count, sizeof(__half), &q_bytes) ||
               !checked_mul_size_t(kv_count, sizeof(__half), &kv_bytes) ||
               !align_up_checked(q_bytes, &aligned_q) ||
               !align_up_checked(kv_bytes, &aligned_kv))
            {
                set_error(error, error_capacity, "workspace byte computation overflowed");
                return 1;
            }
            uint8_t* cursor = static_cast<uint8_t*>(p->workspace);
            __half* staged_q = reinterpret_cast<__half*>(cursor);
            cursor += aligned_q;
            __half* staged_k = reinterpret_cast<__half*>(cursor);
            cursor += aligned_kv;
            __half* staged_v = reinterpret_cast<__half*>(cursor);
            cursor += aligned_kv;
            staged_out = reinterpret_cast<__half*>(cursor);

            const int threads = 256;
            if(q8)
            {
                convert_f32_to_f16<<<(q_count + threads - 1) / threads, threads, 0, stream>>>(
                    static_cast<const float*>(p->q), staged_q, q_count);
                decode_q8_kv_d256<<<dim3(p->seqlen_k, p->nhead_k), 32, 0, stream>>>(
                    static_cast<const uint8_t*>(p->k), static_cast<const uint8_t*>(p->v),
                    staged_k, staged_v, p->seqlen_k, p->nhead_k,
                    p->packed_k_row_stride_bytes, p->packed_v_row_stride_bytes);
            }
            else if(asym3_givens || asym3_fwht)
            {
                if(asym3_givens)
                {
                    transform_q_givens_f32_to_f16<<<dim3(p->seqlen_q, p->nhead_q), 32, 0, stream>>>(
                        static_cast<const float*>(p->q), staged_q, p->seqlen_q, p->nhead_q,
                        p->head_dim, static_cast<const float*>(p->k_transform0),
                        static_cast<const float*>(p->k_transform1));
                }
                else
                {
                    transform_q_fwht_f32_to_f16<<<dim3(p->seqlen_q, p->nhead_q), 32, 0, stream>>>(
                        static_cast<const float*>(p->q), staged_q, p->seqlen_q, p->nhead_q,
                        p->head_dim, static_cast<const float*>(p->k_transform0),
                        static_cast<const float*>(p->k_transform1));
                }
                decode_asym3_k_givens<<<dim3(p->seqlen_k, p->nhead_k), 32, 0, stream>>>(
                    static_cast<const uint8_t*>(p->k), staged_k, p->seqlen_k, p->nhead_k,
                    p->packed_k_row_stride_bytes, p->packed_k_head_stride_bytes, p->head_dim);
                decode_q8<<<dim3(p->seqlen_k, p->nhead_k), 32, 0, stream>>>(
                    static_cast<const uint8_t*>(p->v), staged_v, p->seqlen_k, p->nhead_k,
                    p->packed_v_row_stride_bytes, p->packed_v_head_stride_bytes, p->head_dim);
            }
            else
            {
                if(asym4_givens)
                {
                    transform_q_givens_f32_to_f16<<<dim3(p->seqlen_q, p->nhead_q), 32, 0, stream>>>(
                        static_cast<const float*>(p->q), staged_q, p->seqlen_q, p->nhead_q,
                        p->head_dim, static_cast<const float*>(p->k_transform0),
                        static_cast<const float*>(p->k_transform1));
                }
                else
                {
                    transform_q_fwht128x2_f32_to_f16<<<dim3(p->seqlen_q, p->nhead_q), 32, 0, stream>>>(
                        static_cast<const float*>(p->q), staged_q, p->seqlen_q, p->nhead_q,
                        p->head_dim, static_cast<const float*>(p->k_transform0),
                        static_cast<const float*>(p->k_transform1));
                }
                decode_asym4_k<<<dim3(p->seqlen_k, p->nhead_k), 32, 0, stream>>>(
                    static_cast<const uint8_t*>(p->k), staged_k, p->seqlen_k, p->nhead_k,
                    p->packed_k_row_stride_bytes, p->packed_k_head_stride_bytes, p->head_dim);
                decode_q8<<<dim3(p->seqlen_k, p->nhead_k), 32, 0, stream>>>(
                    static_cast<const uint8_t*>(p->v), staged_v, p->seqlen_k, p->nhead_k,
                    p->packed_v_row_stride_bytes, p->packed_v_head_stride_bytes, p->head_dim);
            }
            if(const hipError_t status = hipGetLastError(); status != hipSuccess)
            {
                set_error(error, error_capacity, hipGetErrorString(status));
                return 3;
            }
            q_ptr = staged_q;
            k_ptr = staged_k;
            v_ptr = staged_v;
            out_ptr = staged_out;
            stride_q = p->nhead_q * p->head_dim;
            stride_k = p->nhead_k * p->head_dim;
            stride_v = p->nhead_k * p->head_dim;
            stride_out = p->nhead_q * p->head_dim;
            nhead_stride_q = nhead_stride_k = nhead_stride_v = nhead_stride_out = p->head_dim;
            batch_stride_q = p->seqlen_q * stride_q;
            batch_stride_k = p->seqlen_k * stride_k;
            batch_stride_v = p->seqlen_k * stride_v;
            batch_stride_out = p->seqlen_q * stride_out;
        }
        const std::string dtype = "fp16";
        const std::string mask_id = p->causal != 0 ? "b:-1,0" : "0";
        const mask_info mask = mask_info::decode(mask_id, p->seqlen_q, p->seqlen_k);

        fmha_fwd_traits traits{
            p->head_dim,
            p->head_dim,
            dtype,
            false,
            true,
            false,
            mask.type,
            bias_enum::no_bias,
            false,
            false,
            quant_scale_enum::no_scale,
            false,
        };

        fmha_fwd_args args{};
        args.q_ptr = q_ptr;
        args.k_ptr = k_ptr;
        args.v_ptr = v_ptr;
        args.o_ptr = out_ptr;
        args.seqlen_q = p->seqlen_q;
        args.seqlen_k = p->seqlen_k;
        args.batch = p->batch;
        args.max_seqlen_q = p->seqlen_q;
        args.hdim_q = p->head_dim;
        args.hdim_v = p->head_dim;
        args.nhead_q = p->nhead_q;
        args.nhead_k = p->nhead_k;
        args.scale_s = p->softmax_scale;
        args.logits_soft_cap = 0.0f;
        args.stride_q = static_cast<ck_tile::index_t>(stride_q);
        args.stride_k = static_cast<ck_tile::index_t>(stride_k);
        args.stride_v = static_cast<ck_tile::index_t>(stride_v);
        args.stride_o = static_cast<ck_tile::index_t>(stride_out);
        args.nhead_stride_q = static_cast<ck_tile::index_t>(nhead_stride_q);
        args.nhead_stride_k = static_cast<ck_tile::index_t>(nhead_stride_k);
        args.nhead_stride_v = static_cast<ck_tile::index_t>(nhead_stride_v);
        args.nhead_stride_o = static_cast<ck_tile::index_t>(nhead_stride_out);
        args.batch_stride_q = static_cast<ck_tile::index_t>(batch_stride_q);
        args.batch_stride_k = static_cast<ck_tile::index_t>(batch_stride_k);
        args.batch_stride_v = static_cast<ck_tile::index_t>(batch_stride_v);
        args.batch_stride_o = static_cast<ck_tile::index_t>(batch_stride_out);
        args.window_size_left = -1;
        args.window_size_right = p->causal != 0 ? 0 : -1;
        args.mask_type = static_cast<ck_tile::index_t>(mask.type);
        args.min_seqlen_q = 0;
        args.p_drop = 0.0f;
        args.s_randval = false;
        args.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

        ck_tile::stream_config stream_config{
            reinterpret_cast<hipStream_t>(p->stream),
        };
        const float result = fmha_fwd(traits, args, stream_config);
        if(result < 0.0f)
        {
            set_error(error, error_capacity, "CK found no matching forward kernel");
            return 2;
        }
        if(q8 || asym3_givens || asym3_fwht || asym4_givens || asym4_fwht)
        {
            const int threads = 256;
            convert_f16_to_f32<<<(q_count + threads - 1) / threads, threads, 0, stream>>>(
                staged_out, static_cast<float*>(p->out), q_count);
            if(const hipError_t status = hipGetLastError(); status != hipSuccess)
            {
                set_error(error, error_capacity, hipGetErrorString(status));
                return 3;
            }
        }
        set_error(error, error_capacity, "");
        return 0;
    }
    catch(const std::exception& exception)
    {
        set_error(error, error_capacity, exception.what());
        return 3;
    }
    catch(...)
    {
        set_error(error, error_capacity, "unknown C++ exception");
        return 3;
    }
}
