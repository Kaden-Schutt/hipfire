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

// Retained-tape staging form. One owner-local dispatch publishes the complete
// activation packet and the device-resident local-route metadata. Its AQL
// packet carries the host checkpoint and a System release; later packets in
// the same batch may consume local_expert_ids/local_count without a host
// round-trip.
extern "C" __global__ void harmonic_stage_route_gfx1100(
    const float* __restrict__ x_rot,
    const unsigned int* __restrict__ expert_ids,
    const unsigned int* __restrict__ route_weight_bits,
    const unsigned int* __restrict__ compact_index_map,
    unsigned int* __restrict__ local_expert_ids,
    unsigned int* __restrict__ slot_sources,
    unsigned int* __restrict__ local_count,
    unsigned char* __restrict__ activation_packet,
    int layer,
    int expert_count,
    int hidden_size,
    int top_k
) {
    if (blockIdx.x != 0) return;
    const int tid = threadIdx.x;

    float* packet_x = reinterpret_cast<float*>(activation_packet);
    for (int index = tid; index < hidden_size; index += blockDim.x) {
        packet_x[index] = x_rot[index];
    }

    unsigned int* packet_ids = reinterpret_cast<unsigned int*>(
        activation_packet + (long long)hidden_size * sizeof(float));
    unsigned int* packet_weights = packet_ids + top_k;
    if (tid < top_k) {
        packet_ids[tid] = expert_ids[tid];
        packet_weights[tid] = route_weight_bits[tid];
    }

    if (tid == 0) {
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
}
