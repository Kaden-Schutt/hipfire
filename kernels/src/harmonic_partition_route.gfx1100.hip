// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Exact-gfx1100 DS4HARM3 route partition. One lane preserves the CPU oracle's
// canonical slot order while removing the local-ID and slot-source H2D bounce.
// The layer-major map contains compact replica indices or UINT_MAX for experts
// owned only by gfx1151.
extern "C" __global__ void harmonic_partition_route_gfx1100(
    const unsigned int* __restrict__ expert_ids,
    const unsigned int* __restrict__ compact_index_map,
    unsigned int* __restrict__ local_expert_ids,
    unsigned int* __restrict__ slot_sources,
    unsigned int* __restrict__ local_count,
    int layer,
    int expert_count,
    int top_k
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int local = 0;
    unsigned int remote = 0;
    const unsigned int* layer_map =
        compact_index_map + (long long)layer * expert_count;
    for (int slot = 0; slot < top_k; ++slot) {
        const unsigned int expert = expert_ids[slot];
        const unsigned int compact = layer_map[expert];
        if (compact != 0xffffffffu) {
            local_expert_ids[local] = compact;
            slot_sources[slot] = local;
            ++local;
        } else {
            slot_sources[slot] = 0x80000000u | remote;
            ++remote;
        }
    }
    *local_count = local;
}
