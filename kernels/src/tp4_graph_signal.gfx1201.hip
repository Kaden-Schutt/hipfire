// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Graph-resident three/four-rank producer barrier for gfx1201 TP. hipStreamWriteValue
// and hipStreamWaitValue instantiate inside HIP graphs on ROCm 7.14, but their
// replay observes the prior epoch after the first launch. Kernel nodes do replay
// correctly, so publish and consume the epoch through system-scope atomics.

extern "C" __global__ void tp_graph_signal_store_gfx1201(
    unsigned int *__restrict__ signal,
    unsigned int epoch) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __hip_atomic_exchange(
            signal, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

extern "C" __global__ void tp_graph_signal_wait2_gfx1201(
    const unsigned int *__restrict__ signal0,
    const unsigned int *__restrict__ signal1,
    unsigned int epoch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    while (__hip_atomic_load(
               signal0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch ||
           __hip_atomic_load(
               signal1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        __builtin_amdgcn_s_sleep(1);
    }
}

extern "C" __global__ void tp_graph_signal_wait3_gfx1201(
    const unsigned int *__restrict__ signal0,
    const unsigned int *__restrict__ signal1,
    const unsigned int *__restrict__ signal2,
    unsigned int epoch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    while (__hip_atomic_load(
               signal0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch ||
           __hip_atomic_load(
               signal1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch ||
           __hip_atomic_load(
               signal2, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        __builtin_amdgcn_s_sleep(1);
    }
}

// TP4-only combined producer barrier. Each rank reaches this node after its
// local producer, publishes that completion, then waits for the other three
// ranks. This is semantically identical to a store node followed by a wait3
// node on every FIFO rank stream, while removing one graph node per rank per
// barrier.
extern "C" __global__ void tp_graph_signal_barrier3_gfx1201(
    unsigned int *__restrict__ signal,
    const unsigned int *__restrict__ signal0,
    const unsigned int *__restrict__ signal1,
    const unsigned int *__restrict__ signal2,
    unsigned int epoch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    __hip_atomic_exchange(
        signal, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    while (__hip_atomic_load(
               signal0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch ||
           __hip_atomic_load(
               signal1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch ||
           __hip_atomic_load(
               signal2, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < epoch) {
        __builtin_amdgcn_s_sleep(1);
    }
}
