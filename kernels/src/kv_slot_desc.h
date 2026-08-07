// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Multi-slot KV descriptor — the single point of KV address translation for
// every batched attention kernel.
//
// A "slot" is one independent sequence in a batch. Today each slot owns a
// contiguous slab of `cap` tokens inside a per-layer arena, so translation is
// base + pos*stride. When this moves to paged block tables, ONLY
// kv_offset_for() changes — no kernel is touched. That is the entire reason
// this indirection exists; do not inline it away.
//
// Layout must stay byte-identical to the Rust mirror in
// crates/rdna-compute/src/kv_slots.rs. 24 bytes, 8-byte aligned.

#pragma once

#include <hip/hip_runtime.h>

struct KvSlotDesc {
    unsigned long long k_base;  // byte offset of this slot's K slab in the arena
    unsigned long long v_base;  // byte offset of this slot's V slab in the arena
    int seq_len;                // logical KV length; kernel reads [0, seq_len)
    int cap;                    // physical slab capacity in tokens; seq_len <= cap
};

// Byte offset of position `pos` within slot `s`'s K slab.
// `per_pos_bytes` is the per-position stride in bytes, uniform across slots
// (n_kv_heads * (head_dim/32) * 34 for Q8_0).
__device__ __forceinline__ unsigned long long kv_offset_for_k(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    return s.k_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

__device__ __forceinline__ unsigned long long kv_offset_for_v(
    const KvSlotDesc& s, int pos, int per_pos_bytes)
{
    return s.v_base + (unsigned long long)pos * (unsigned long long)per_pos_bytes;
}

// Single-slot fallback used when the descriptor pointer is null. Keeps the
// ported kernels on ONE code path: callers build this on the stack from the
// legacy scalar args, so there is no `if (descs) ... else ...` around every
// KV read. Behaviour is then bitwise identical to the pre-SP1 kernel.
__device__ __forceinline__ KvSlotDesc kv_slot_legacy(int seq_len, int max_seq)
{
    KvSlotDesc s;
    s.k_base = 0ULL;
    s.v_base = 0ULL;
    s.seq_len = seq_len;
    s.cap = max_seq;
    return s;
}
