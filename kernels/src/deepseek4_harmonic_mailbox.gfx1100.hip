// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// DeepSeek V4 harmonic gfx1100 mailbox operations.
//
// These kernels replace the per-layer host publication/poll loop with one
// producer and one bounded consumer operation on the dense owner's queue.
// The gfx1151 worker remains a separate process and continues to validate the
// authoritative epoch-typed HarmonicWireSlot before running an expert.

#include <hip/hip_runtime.h>

namespace {

constexpr unsigned kDenseOwner = 1;
constexpr unsigned kExpertOwner = 2;

constexpr int kHidden = 4096;
constexpr int kTopK = 6;
constexpr unsigned kActivationExtent = 16448;
constexpr unsigned kResultExtent = 16384;
constexpr int kExpertIdsPayloadOffset = 16384;
constexpr int kRouteWeightsPayloadOffset = 16408;
constexpr int kReservedPayloadOffset = 16432;

// HarmonicWireHeader ABI. Rust offset assertions live beside the wire type.
constexpr int kHeaderRouteIdentity = 24;
constexpr int kHeaderModelIdentity = 32;
constexpr int kHeaderSourceGeneration = 64;
constexpr int kHeaderDestinationGeneration = 72;
constexpr int kHeaderIsolatedOwners = 80;
constexpr int kHeaderPublishedEpochs = 128;
constexpr int kHeaderCompletedEpochs = 192;

// HarmonicWireSlot control-prefix ABI (payload pointers are passed separately).
constexpr int kSlotEpoch = 8;
constexpr int kSlotRouteIdentity = 16;
constexpr int kSlotModelIdentity = 24;
constexpr int kSlotLayer = 56;
constexpr int kSlotIndex = 60;
constexpr int kSlotSourceOwner = 64;
constexpr int kSlotDestinationOwner = 68;
constexpr int kSlotSourceGeneration = 72;
constexpr int kSlotDestinationGeneration = 80;
constexpr int kSlotExpertIds = 88;
constexpr int kSlotRouteWeights = 112;
constexpr int kSlotActivationExtent = 136;
constexpr int kSlotResultExtent = 140;
constexpr int kSlotDeadline = 144;
constexpr int kSlotActivationFingerprint = 152;
constexpr int kSlotResultFingerprint = 160;

enum MailboxError : unsigned {
  kMailboxOk = 0,
  kPublishEndpointIsolated = 1,
  kPublishSlotBusy = 2,
  kWaitTimeout = 3,
  kWaitTerminalFailure = 4,
  kWaitResultExtent = 5,
};

__device__ __forceinline__ unsigned sys_load_u32(const char *base, int offset,
                                                  int order) {
  return __hip_atomic_load(
      reinterpret_cast<const unsigned *>(base + offset), order,
      __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ unsigned long long
sys_load_u64(const char *base, int offset, int order) {
  return __hip_atomic_load(
      reinterpret_cast<const unsigned long long *>(base + offset), order,
      __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void sys_store_u32(char *base, int offset,
                                               unsigned value, int order) {
  __hip_atomic_store(reinterpret_cast<unsigned *>(base + offset), value, order,
                     __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void
sys_store_u64(char *base, int offset, unsigned long long value, int order) {
  __hip_atomic_store(reinterpret_cast<unsigned long long *>(base + offset),
                     value, order, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void set_first_error(unsigned *status,
                                                 unsigned error, int layer,
                                                 unsigned state,
                                                 unsigned flags) {
  const unsigned detail = error | ((static_cast<unsigned>(layer) & 0xffu) << 8) |
                          ((state & 0xfu) << 16) |
                          ((flags & 0xfu) << 20);
  atomicCAS(status, kMailboxOk, detail);
}

} // namespace

// One workgroup publishes one layer. It copies the 4096-element activation,
// mirrors route metadata, then release-publishes the packet to the CPU worker.
// `token_control = [base_epoch, common_deadline_tick]`.
extern "C" __global__ void deepseek4_harmonic_publish_gfx1100(
    char *__restrict__ header, char *__restrict__ slot0,
    char *__restrict__ slot1, char *__restrict__ activation0,
    char *__restrict__ activation1, const float *__restrict__ x_rot,
    const int *__restrict__ topk_indices,
    const float *__restrict__ topk_weights,
    const unsigned long long *__restrict__ token_control, int layer,
    unsigned *__restrict__ status) {
  __shared__ unsigned publish;
  __shared__ unsigned long long epoch;
  __shared__ char *slot;
  __shared__ char *activation;

  const int tid = threadIdx.x;
  if (tid == 0) {
    publish = 0;
    epoch = token_control[0] + static_cast<unsigned long long>(layer);
    const unsigned slot_index = static_cast<unsigned>(epoch & 1ull);
    slot = slot_index == 0 ? slot0 : slot1;
    activation = slot_index == 0 ? activation0 : activation1;

    if (atomicAdd(status, 0u) != kMailboxOk) {
      // A prior layer failed. Leave this epoch unpublished so the worker never
      // consumes state derived after a failed result.
    } else if (sys_load_u32(header, kHeaderIsolatedOwners, __ATOMIC_ACQUIRE) !=
               0) {
      set_first_error(status, kPublishEndpointIsolated, layer, 0, 0);
    } else {
      const unsigned long long prior_published =
          sys_load_u64(header, kHeaderPublishedEpochs + slot_index * 8,
                       __ATOMIC_ACQUIRE);
      const unsigned long long prior_completed =
          sys_load_u64(header, kHeaderCompletedEpochs + slot_index * 8,
                       __ATOMIC_ACQUIRE);
      if (prior_published != 0 && prior_completed != prior_published) {
        set_first_error(status, kPublishSlotBusy, layer,
                        static_cast<unsigned>(prior_published & 0xfu),
                        static_cast<unsigned>(prior_completed & 0xfu));
      } else {
        sys_store_u64(slot, kSlotEpoch, epoch, __ATOMIC_RELAXED);
        sys_store_u64(
            slot, kSlotRouteIdentity,
            sys_load_u64(header, kHeaderRouteIdentity, __ATOMIC_RELAXED),
            __ATOMIC_RELAXED);
        for (int word = 0; word < 4; ++word) {
          sys_store_u64(
              slot, kSlotModelIdentity + word * 8,
              sys_load_u64(header, kHeaderModelIdentity + word * 8,
                           __ATOMIC_RELAXED),
              __ATOMIC_RELAXED);
        }
        sys_store_u32(slot, kSlotLayer, static_cast<unsigned>(layer),
                      __ATOMIC_RELAXED);
        sys_store_u32(slot, kSlotIndex, slot_index, __ATOMIC_RELAXED);
        sys_store_u32(slot, kSlotSourceOwner, kDenseOwner, __ATOMIC_RELAXED);
        sys_store_u32(slot, kSlotDestinationOwner, kExpertOwner,
                      __ATOMIC_RELAXED);
        sys_store_u64(
            slot, kSlotSourceGeneration,
            sys_load_u64(header, kHeaderSourceGeneration, __ATOMIC_RELAXED),
            __ATOMIC_RELAXED);
        sys_store_u64(slot, kSlotDestinationGeneration,
                      sys_load_u64(header, kHeaderDestinationGeneration,
                                   __ATOMIC_RELAXED),
                      __ATOMIC_RELAXED);
        sys_store_u32(slot, kSlotActivationExtent, kActivationExtent,
                      __ATOMIC_RELAXED);
        sys_store_u32(slot, kSlotResultExtent, kResultExtent,
                      __ATOMIC_RELAXED);
        sys_store_u64(slot, kSlotDeadline, token_control[1],
                      __ATOMIC_RELAXED);
        sys_store_u64(slot, kSlotActivationFingerprint, 0,
                      __ATOMIC_RELAXED);
        sys_store_u64(slot, kSlotResultFingerprint, 0, __ATOMIC_RELAXED);
        publish = 1;
      }
    }
  }
  __syncthreads();

  if (publish == 0)
    return;

  for (int index = tid; index < kHidden; index += blockDim.x)
    reinterpret_cast<float *>(activation)[index] = x_rot[index];

  if (tid < kTopK) {
    const unsigned expert = static_cast<unsigned>(topk_indices[tid]);
    const unsigned weight = __builtin_bit_cast(unsigned, topk_weights[tid]);
    sys_store_u32(slot, kSlotExpertIds + tid * 4, expert, __ATOMIC_RELAXED);
    sys_store_u32(slot, kSlotRouteWeights + tid * 4, weight,
                  __ATOMIC_RELAXED);
    *reinterpret_cast<unsigned *>(activation + kExpertIdsPayloadOffset +
                                  tid * 4) = expert;
    *reinterpret_cast<unsigned *>(activation + kRouteWeightsPayloadOffset +
                                  tid * 4) = weight;
  }
  if (tid < 4)
    *reinterpret_cast<unsigned *>(activation + kReservedPayloadOffset +
                                  tid * 4) = 0;

  __syncthreads();
  if (tid == 0) {
    __threadfence_system();
    // This word has exactly one writer (dense gfx1100). The expert acquires it
    // before reading any slot metadata or activation bytes.
    sys_store_u64(header, kHeaderPublishedEpochs + (epoch & 1ull) * 8,
                  epoch, __ATOMIC_RELEASE);
  }
}

// One workgroup waits for exactly one terminal epoch. The wait is bounded in
// gfx1100 clock cycles, and the first failure latches `status`; later mailbox
// kernels then fail fast rather than stacking 43 timeouts. A successful wait
// copies the peer-owned 16 KiB result into device-local routed_partial and
// release-marks source quiescence.
extern "C" __global__ void deepseek4_harmonic_wait_copy_gfx1100(
    const char *__restrict__ header, const char *__restrict__ result0,
    const char *__restrict__ result1,
    float *__restrict__ routed_partial,
    const unsigned long long *__restrict__ token_control, int layer,
    unsigned long long max_wait_clocks, unsigned *__restrict__ status) {
  __shared__ unsigned copy_result;
  __shared__ const char *result;

  const int tid = threadIdx.x;
  if (tid == 0) {
    copy_result = 0;
    const unsigned long long epoch =
        token_control[0] + static_cast<unsigned long long>(layer);
    const unsigned slot_index = static_cast<unsigned>(epoch & 1ull);
    result = slot_index == 0 ? result0 : result1;

    if (atomicAdd(status, 0u) == kMailboxOk) {
      const unsigned long long start = clock64();
      for (;;) {
        const unsigned long long completed =
            sys_load_u64(header, kHeaderCompletedEpochs + slot_index * 8,
                         __ATOMIC_ACQUIRE);
        if (completed == epoch) {
          copy_result = 1;
          break;
        }
        if (completed > epoch) {
          set_first_error(status, kWaitTerminalFailure, layer,
                          static_cast<unsigned>(completed & 0xfu), 0);
          break;
        }
        if (clock64() - start >= max_wait_clocks) {
          set_first_error(status, kWaitTimeout, layer,
                          static_cast<unsigned>(completed & 0xfu), 0);
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();

  for (int index = tid; index < kHidden; index += blockDim.x) {
    routed_partial[index] =
        copy_result ? reinterpret_cast<const float *>(result)[index] : 0.0f;
  }
}
