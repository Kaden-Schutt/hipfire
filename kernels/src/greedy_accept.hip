// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Greedy speculative-accept prefix scan over device-resident verify argmaxes
// and MTP candidate tokens.
//
// result[0] = number of draft candidates accepted
// result[1] = bonus token id, or -1 when an accepted candidate was EOS and no
//             bonus should be committed
extern "C" __global__ void greedy_accept_from_argmax_i32(
    const int* __restrict__ argmax_per_pos,
    const int* __restrict__ candidates,
    int* __restrict__ result,
    int drafts_generated,
    int eos_token_id
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int accept_count = 0;
    for (int k = 0; k < drafts_generated; ++k) {
        int candidate = candidates[k];
        if (argmax_per_pos[k] != candidate) {
            result[0] = accept_count;
            result[1] = argmax_per_pos[k];
            return;
        }

        ++accept_count;
        if (candidate == eos_token_id) {
            result[0] = accept_count;
            result[1] = -1;
            return;
        }
    }

    result[0] = accept_count;
    result[1] = argmax_per_pos[drafts_generated];
}
