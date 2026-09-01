// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HIPFIRE_FLASH_ATTN_CK_ABI_VERSION 4u

enum hipfire_flash_attn_ck_dtype {
    HIPFIRE_FLASH_ATTN_CK_F16 = 1,
    HIPFIRE_FLASH_ATTN_CK_BF16 = 2,
    HIPFIRE_FLASH_ATTN_CK_F32 = 3,
};

enum hipfire_flash_attn_ck_arch {
    HIPFIRE_FLASH_ATTN_CK_GFX1100 = 1100,
    HIPFIRE_FLASH_ATTN_CK_GFX1151 = 1151,
    HIPFIRE_FLASH_ATTN_CK_GFX1201 = 1201,
};

enum hipfire_flash_attn_ck_kv_format {
    HIPFIRE_FLASH_ATTN_CK_DENSE_F16 = 1,
    HIPFIRE_FLASH_ATTN_CK_DENSE_BF16 = 2,
    HIPFIRE_FLASH_ATTN_CK_Q8 = 3,
    HIPFIRE_FLASH_ATTN_CK_ASYM3_GIVENS = 4,
    HIPFIRE_FLASH_ATTN_CK_ASYM3_FWHT = 5,
    HIPFIRE_FLASH_ATTN_CK_LLOYD = 6,
    HIPFIRE_FLASH_ATTN_CK_ASYM4_GIVENS = 7,
    HIPFIRE_FLASH_ATTN_CK_ASYM4_FWHT = 8,
};

#define HIPFIRE_FLASH_ATTN_CK_CAP_CAUSAL (1u << 0)
#define HIPFIRE_FLASH_ATTN_CK_CAP_GQA (1u << 1)

struct hipfire_flash_attn_ck_capability {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t arch;
    int32_t dtype;
    int32_t k_format;
    int32_t v_format;
    int32_t head_dim;
    uint32_t flags;
};

struct hipfire_flash_attn_ck_fwd_params {
    uint32_t abi_version;
    uint32_t struct_size;

    const void* q;
    const void* k;
    const void* v;
    void* out;
    void* workspace;
    size_t workspace_bytes;
    void* stream;

    int32_t dtype;
    int32_t k_format;
    int32_t v_format;
    int32_t batch;
    int32_t seqlen_q;
    int32_t seqlen_k;
    int32_t nhead_q;
    int32_t nhead_k;
    int32_t head_dim;
    int32_t causal;

    float softmax_scale;

    int64_t stride_q;
    int64_t stride_k;
    int64_t stride_v;
    int64_t stride_out;
    int64_t nhead_stride_q;
    int64_t nhead_stride_k;
    int64_t nhead_stride_v;
    int64_t nhead_stride_out;
    int64_t batch_stride_q;
    int64_t batch_stride_k;
    int64_t batch_stride_v;
    int64_t batch_stride_out;

    /* Packed KV row strides are bytes. Zero for dense K/V formats. */
    int64_t packed_k_row_stride_bytes;
    int64_t packed_v_row_stride_bytes;
    int64_t packed_k_head_stride_bytes;
    int64_t packed_v_head_stride_bytes;

    /* Asym transform metadata: cos/sin for Givens, signs1/signs2 for FWHT. */
    const void* k_transform0;
    const void* k_transform1;
    int64_t k_transform0_elements;
    int64_t k_transform1_elements;
};

uint32_t hipfire_flash_attn_ck_abi_version(void);

size_t hipfire_flash_attn_ck_capabilities(
    struct hipfire_flash_attn_ck_capability* capabilities,
    size_t capacity);

size_t hipfire_flash_attn_ck_fwd_workspace_bytes(
    const struct hipfire_flash_attn_ck_fwd_params* params);

int hipfire_flash_attn_ck_fwd_supported(
    const struct hipfire_flash_attn_ck_fwd_params* params,
    char* error,
    size_t error_capacity);

int hipfire_flash_attn_ck_fwd(
    const struct hipfire_flash_attn_ck_fwd_params* params,
    char* error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
