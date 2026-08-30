// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Runtime-compiled host shim: rocPRIM device top-K (largest, unordered) over
// f32 logits → (val, idx) pairs. Two-phase temp-storage convention.
// Build flags (caller): -O3 -fPIC -shared -DROCPRIM_WITH_TOPK --offload-arch=<arch>
//                       -I<rocm include>

#include <hip/hip_runtime.h>
#include <rocprim/device/device_topk.hpp>
#include <rocprim/iterator/counting_iterator.hpp>

#include <cstddef>
#include <cstdint>

extern "C" int hipfire_rocprim_topk_f32(
    void* stream,
    const float* d_logits,
    unsigned vocab,
    unsigned k,
    float* d_out_vals,
    unsigned* d_out_idx,
    void* d_temp,
    size_t* temp_bytes)
{
    if (temp_bytes == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (d_logits == nullptr || vocab == 0 || k == 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (d_temp != nullptr && (d_out_vals == nullptr || d_out_idx == nullptr)) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    // Identity indices via counting iterator — no vocab-sized index buffer.
    auto idx_in = rocprim::make_counting_iterator(0u);
    hipStream_t s = static_cast<hipStream_t>(stream);

    // Descending=true (largest), Ordered=false (unordered; Ordered unsupported).
    hipError_t err = rocprim::topk_pairs<rocprim::default_config,
                                          true /*Descending*/,
                                          false /*Ordered*/>(
        d_temp,
        *temp_bytes,
        d_logits,
        d_out_vals,
        idx_in,
        d_out_idx,
        static_cast<size_t>(vocab),
        static_cast<size_t>(k),
        {},
        s,
        false);
    return static_cast<int>(err);
}
