// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Graph-resident four-rank producer barrier for gfx1201 TP. hipStreamWriteValue
// and hipStreamWaitValue instantiate inside HIP graphs on ROCm 7.14, but their
// replay observes the prior epoch after the first launch. Kernel nodes do replay
// correctly, so publish and consume the epoch through system-scope atomics.

extern "C" __global__ void tp4_graph_signal_store_gfx1201(
    unsigned int *__restrict__ signals,
    int slot) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __hip_atomic_exchange(
            signals + slot, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

extern "C" __global__ void tp4_graph_signal_wait3_gfx1201(
    const unsigned int *__restrict__ signals0,
    const unsigned int *__restrict__ signals1,
    const unsigned int *__restrict__ signals2,
    int slot) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    while (__hip_atomic_load(
               signals0 + slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 1u ||
           __hip_atomic_load(
               signals1 + slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 1u ||
           __hip_atomic_load(
               signals2 + slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 1u) {
        __builtin_amdgcn_s_sleep(1);
    }
}
