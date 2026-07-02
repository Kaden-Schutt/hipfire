// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1201 (RDNA4) dp4a Phase-B spike of the indexed MoE gate_up decode
// GEMV. Same expert-pointer / topk_indices gather and the same expanded
// gate/up output layout as the scalar FP32 kernel
// (gemv_hfq4g256_moe_gate_up_indexed.hip's
// `gemv_hfq4g256_moe_gate_up_k8_indexed`) — only the inner dot product
// changes: int8 (v_dot4_i32_iu8) instead of per-element FMA.
//
// ───────────────────────────────────────────────────────────────────────────
// STATUS: opt-in via HIPFIRE_GFX1201_DP4A=1 (default OFF — the scalar
// kernel is unchanged and stays the production path on every arch,
// including gfx1201). Per docs/gfx1201-native-surface.md, the roofline /
// occupancy pre-check called broad Phase-B dp4a NO-GO (100% static
// occupancy already, mixed/DRAM-leaning bound-class, direct ALU-vs-mem PMC
// unobtainable on this ROCm/box) — but the direct measurement never
// resolved either way, so this kernel exists to let the silicon decide via
// an on-device A/B rather than the roofline estimate alone. `moe_gate_up`
// (this kernel) was the doc's "best dp4a candidate" (~72.5% of DRAM peak,
// most credible ALU-headroom of the hot family). First iteration — MAY be
// a wash or a loss; that is an acceptable spike outcome.
// ───────────────────────────────────────────────────────────────────────────
//
// Math derivation (asymmetric HFQ4G256 dequant, PRESERVED EXACTLY from the
// scalar kernel's `DOG` macro convention):
//
//   weight_val = nibble * scale + zp        (nibble unsigned [0,15], no
//                                             bias/re-centering — this is
//                                             the exact convention read
//                                             from gemv_hfq4g256_moe_gate_up_
//                                             indexed.hip; do NOT flip the
//                                             zp sign or add an 8*scale
//                                             shift — that is a DIFFERENT
//                                             kernel family's convention,
//                                             see note below)
//
//   Σ_i(weight_val_i · act_i)
//     = scale · Σ_i(nibble_i · act_i) + zp · Σ_i(act_i)
//     = scale · [per-32-subblock: d_j · Σ_{i∈j}(nibble_i · q_i)] + zp · Σ_i(act_i)
//
// where q_i is the Q8_1-quantized (QK8_1=32) signed int8 activation and
// d_j is that subblock's f32 dequant scale (act_i ≈ d_j * q_i). The
// per-256-group weight scale/zp are applied to the SUM of the 8 per-32
// dp4a partial dot products the group spans (256/32 = 8 subblocks).
//
// dp4a mechanics: v_dot4_i32_iu8 (__builtin_amdgcn_sudot4 with
// a_signed=false, b_signed=true) takes an UNSIGNED packed-int8 "a" operand
// and a SIGNED packed-int8 "b" operand. Each weight nibble (0..15) is
// zero-extended into its own byte — genuinely unsigned, no bias needed —
// and packed 4-at-a-time (half of one 32-bit nibble-dword) against the
// matching 4 quantized (signed) activation bytes. This is DIFFERENT from
// the existing wave64 MMQ dp4a kernels (e.g.
// gemm_gate_up_hfq4g256_wave64_dp4a.hip), which use plain sdot4
// (signed×signed) and therefore re-center the nibble via `nibble - 8` plus
// a `zp_eff = zp + 8*scale` correction. Both are mathematically
// equivalent; this kernel takes the more direct iu8 route per the task
// spec instead of re-deriving the sdot4 bias trick.
//
// Activation reorder (in-kernel, NOT an offline repack): the weight nibble
// layout is fixed by the on-disk HFQ4G256 format (row-major nibble bytes,
// unchanged). Rather than repacking weights to match a natural int8
// activation order, this kernel reorders which quantized-activation bytes
// it reads per lane (`qs_byte_off`, `sub_in_block`, `block_within_group`)
// so they line up with the SAME 8 elements each lane's weight nibble-dword
// already covers (`base = g*256 + tid*8 .. +7`) — identical element
// coverage to the scalar kernel, just int8 instead of fp32.
//
// The `0.25f` factor on the zp·Σ(act) term is not a fudge: 4 lanes
// (tid, tid+1, tid+2, tid+3 within a 4-lane cluster) share one QK8_1=32
// activation subblock's precomputed Σ(act) (from `ds4[k].y`, the raw
// float sum computed by quantize_q8_1_mmq_ds4 — NOT a sum of quantized
// int8 values). Each of those 4 lanes independently reads the same
// Σ_j(act) and would otherwise 4x-overcount it once the final warp-wide
// __shfl_down reduction sums all 32 lanes together; dividing by 4 up
// front makes the post-reduction total exact. See
// gemm_gate_up_hfq4g256_wave64_dp4a.hip for the sibling derivation (same
// 4-lanes-per-Q8_1-subblock structure, same correction factor).

#define QK8_1 32

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// One (gp, group-index g, accumulator) dp4a dot-product step: one 136-byte
// weight group (256 4-bit weights) contributes exactly 8 elements per
// lane, matched against 8 quantized-activation bytes from the
// pre-quantized `Xq` (see rdna-compute's `ensure_q8_1_mmq_x`, N=1 for this
// single-token decode GEMV so `Xq + kblock` indexes directly with no
// per-token stride).
#define DP4A_DOG(gp, g, a) do { \
    const float sc__ = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
    const float zp__ = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4)); \
    const unsigned int pk__ = *(const unsigned int*)((gp) + 8 + boff); \
    const unsigned int ia__ = (pk__ & 0xFu) \
                             | (((pk__ >>  4) & 0xFu) << 8) \
                             | (((pk__ >>  8) & 0xFu) << 16) \
                             | (((pk__ >> 12) & 0xFu) << 24); \
    const unsigned int ib__ = ((pk__ >> 16) & 0xFu) \
                             | (((pk__ >> 20) & 0xFu) << 8) \
                             | (((pk__ >> 24) & 0xFu) << 16) \
                             | (((pk__ >> 28) & 0xFu) << 24); \
    const int kblock__ = 2 * (g) + block_within_group; \
    const block_q8_1_mmq* xb__ = Xq + kblock__; \
    const int* xqs__ = (const int*)(xb__->qs + qs_byte_off); \
    int sumi__ = 0; \
    sumi__ = __builtin_amdgcn_sudot4(false, (int)ia__, true, xqs__[0], sumi__, false); \
    sumi__ = __builtin_amdgcn_sudot4(false, (int)ib__, true, xqs__[1], sumi__, false); \
    const half2 ds__ = xb__->ds4[sub_in_block]; \
    const float2 dsf__ = __half22float2(ds__); \
    (a) += sc__ * dsf__.x * (float)sumi__ + zp__ * dsf__.y * 0.25f; \
} while (0)

// Own module name (kernels.rs) AND a `_gfx1201` symbol suffix — mandatory
// per the kernel-module-cache-collision lesson (JIT cache is keyed by
// module name only; an arch-variant reusing the base module name would be
// dead code on every other arch and could silently shadow the scalar
// kernel here).
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_gate_up_k8_indexed_dp4a_gfx1201(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const block_q8_1_mmq* __restrict__ Xq,                // [K/128] pre-quantized blocks (N=1)
    float* __restrict__ y_gate,                           // [K_TOP × MI]
    float* __restrict__ y_up,                             // [K_TOP × MI]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    // tid-only (group-independent — see derivation above): which half of
    // the 256-elem group (block_within_group), which 32-elem Q8_1
    // subblock within that half (sub_in_block), and the byte offset into
    // that subblock's 128-elem `qs` array (qs_byte_off).
    const int block_within_group = tid / 16;
    const int sub_in_block       = (tid / 4) % 4;
    const int qs_byte_off        = 8 * (tid % 16);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        DP4A_DOG(gp0, g,     acc0);
        DP4A_DOG(gp1, g + 1, acc1);
        DP4A_DOG(gp2, g + 2, acc2);
        DP4A_DOG(gp3, g + 3, acc3);
    }

    // Tail groups (groups_per_row not a multiple of 4): tail-into-acc[g%4]
    // discipline, same as the scalar kernel and every other DOG/TAIL_DOG
    // kernel in this family (commit 5302926 bug class — dumping all tail
    // groups into acc0 silently breaks output when groups_per_row isn't a
    // multiple of 4).
    if (tail >= 1) {
        const int g = quads << 2;
        DP4A_DOG(row_ptr + g * 136, g, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        DP4A_DOG(row_ptr + g * 136, g, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        DP4A_DOG(row_ptr + g * 136, g, acc2);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        if (row < mi) {
            y_gate[krank * mi + row] = acc;
        } else {
            y_up[krank * mi + (row - mi)] = acc;
        }
    }
}
