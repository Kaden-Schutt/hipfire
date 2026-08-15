// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

// Radiowave's first promoted lowering rule. The descriptor word is the untyped
// AMDGPU resource configuration derived from ROCm CK and correctness-checked
// on gfx1010, gfx1030, gfx1100, gfx1151, and gfx1201. Callers must keep each
// byte offset within the 32-bit resource window; larger allocations need a
// rebased descriptor. Performance promotion remains architecture-specific.
namespace radiowave {

using u32 = uint32_t;
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using BufferResource = __amdgpu_buffer_rsrc_t;

// AMDGPU cache-policy zero is the temporal/default path on the supported
// RDNA targets: retain ordinary L0/L1/L2 reuse. Coherence and dependency
// invalidation belong to Redline's certified dispatch boundary, not to every
// load instruction. Naming the policy here keeps Radiowave's default explicit
// and leaves room for correctness-gated non-temporal variants.
enum class CachePolicy : int {
    Temporal = 0,
};

inline constexpr int kDefaultCachePolicy =
    static_cast<int>(CachePolicy::Temporal);

// Preserve a distinct vector address expression through LLVM IR combining
// without emitting a machine instruction or imposing a memory-ordering edge.
// This is useful when several adjacent B32 requests should remain independently
// completable for the AMDGPU machine scheduler instead of becoming one B128.
__device__ __forceinline__ void opaque_vgpr_u32x4(
    u32& x0, u32& x1, u32& x2, u32& x3) {
    asm volatile("" : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3));
}

__device__ __forceinline__ BufferResource buffer_resource(const u32* p) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<u32*>(p), 0, 0xffffffffu, 0x31004000u);
}

__device__ __forceinline__ u32 buffer_load_u32(BufferResource resource,
                                                u32 word_offset) {
    return static_cast<u32>(__builtin_amdgcn_raw_buffer_load_b32(
        resource, word_offset * sizeof(u32), 0, kDefaultCachePolicy));
}

__device__ __forceinline__ u32 buffer_load_bytes_u32(BufferResource resource,
                                                      u32 byte_offset) {
    return static_cast<u32>(__builtin_amdgcn_raw_buffer_load_b32(
        resource, byte_offset, 0, kDefaultCachePolicy));
}

__device__ __forceinline__ u32x4 buffer_load_u32x4(BufferResource resource,
                                                    u32 word_offset) {
    const i32x4 words = __builtin_amdgcn_raw_buffer_load_b128(
        resource, word_offset * sizeof(u32), 0, kDefaultCachePolicy);
    return __builtin_bit_cast(u32x4, words);
}

__device__ __forceinline__ void buffer_store_u32(BufferResource resource,
                                                  u32 word_offset, u32 value) {
    __builtin_amdgcn_raw_buffer_store_b32(
        static_cast<int32_t>(value), resource, word_offset * sizeof(u32), 0,
        kDefaultCachePolicy);
}

} // namespace radiowave
