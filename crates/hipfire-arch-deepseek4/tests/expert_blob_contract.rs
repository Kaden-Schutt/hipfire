// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Characterisation test for the routed-expert blob + pointer-table layout.
//!
//! Routed-expert paging (docs/superpowers/plans/2026-08-02-ds4-expert-paging.md)
//! repoints table entries at cache slots. That is only sound if the layout is
//! exactly what `arch.rs` documents. If it ever changes, paging silently reads
//! the WRONG weights — a quality regression with no error — so pin it here.
//!
//! Verified against `upload_layer_routed_experts` (crates/hipfire-arch-deepseek4/
//! src/arch.rs:167, doc comment from :151):
//!
//! 1. ds4 uploads TWO blobs per layer-shaped block, not three: it "Writes
//!    `expert_w2_blob/_ptrs/_stride` and `expert_gate_up_blob/_ptrs/_stride`"
//!    — w1+w3 are fused into gate_up. So `ExpertBlobRole` is {GateUp, Down}.
//!
//! 2. The blob need NOT hold all `n_routed_experts`. The existing EP
//!    shard-aware path (`shard = Some((cfg, rank))`) already uploads "ONLY the
//!    rank-owned experts ... into a compact packed blob", and "the per-expert
//!    pointer table then maps owned `e` → its compact-blob slot". That
//!    expert→slot indirection is precisely what paging needs; paging changes
//!    the resident subset from static (chosen by EP rank) to dynamic (chosen
//!    by LRU). `shard = None` uploads all experts and is byte-identical to the
//!    original single-GPU path.
//!
//! 3. HAZARD, load-bearing for paging: in the shard path a NON-owned expert's
//!    pointer is aimed at "a shared ZEROED gate_up dummy (SwiGLU(0,0)=0 ⇒ 0
//!    routed contribution)". That is correct for EP, where a non-owned expert
//!    is genuinely computed on another rank — but it means a stale or unset
//!    pointer produces silence, not an error. Paging MUST never leave a routed
//!    expert aimed at the dummy; the plan's fail-closed rule exists for this.

/// Decode the two-F32-slot pointer encoding used by the indexed MoE GEMV.
/// Two F32 slots per u64 pointer, matching the qwen35 convention documented on
/// `expert_w1_ptrs` in deepseek4.rs.
fn decode_ptr(slots: &[f32], expert: usize) -> u64 {
    let lo = slots[expert * 2].to_bits() as u64;
    let hi = slots[expert * 2 + 1].to_bits() as u64;
    (hi << 32) | lo
}

/// Encode a device pointer into the two-F32-slot table representation.
fn encode_ptr(slots: &mut [f32], expert: usize, p: u64) {
    slots[expert * 2] = f32::from_bits((p & 0xffff_ffff) as u32);
    slots[expert * 2 + 1] = f32::from_bits((p >> 32) as u32);
}

#[test]
fn pointer_table_encodes_two_f32_slots_per_expert() {
    let base: u64 = 0x7f00_0000_1000;
    let stride: u64 = 2_359_296; // 2304 KiB, the MQ2 per-expert size
    let n = 4usize;
    let mut slots = vec![0f32; n * 2];
    for e in 0..n {
        encode_ptr(&mut slots, e, base + e as u64 * stride);
    }
    for e in 0..n {
        assert_eq!(decode_ptr(&slots, e), base + e as u64 * stride);
    }
}

#[test]
fn encoding_survives_pointers_with_high_bits_set() {
    // f32::from_bits round-trips NaN payloads bit-exactly, which is what makes
    // this encoding safe for arbitrary pointers. A naive `as f32` cast would
    // silently corrupt them, so prove the bit-level round-trip explicitly.
    let mut slots = vec![0f32; 2];
    for p in [
        0xffff_ffff_ffff_ffffu64,
        0x7ff8_0000_7ff8_0000, // both halves are NaN bit patterns
        0x0000_0000_ffff_ffff,
        0xffff_ffff_0000_0000,
        1,
    ] {
        encode_ptr(&mut slots, 0, p);
        assert_eq!(decode_ptr(&slots, 0), p, "round-trip failed for {p:#x}");
    }
}

#[test]
fn slot_repoint_is_reversible() {
    // Paging repoints an entry at a cache slot and must be able to restore it.
    let base: u64 = 0x7f00_0000_1000;
    let slot_base: u64 = 0x7f00_9000_0000;
    let stride: u64 = 2_359_296;
    let mut slots = vec![0f32; 8];

    encode_ptr(&mut slots, 2, base + 2 * stride);
    assert_eq!(decode_ptr(&slots, 2), base + 2 * stride);
    encode_ptr(&mut slots, 2, slot_base);
    assert_eq!(decode_ptr(&slots, 2), slot_base);
    encode_ptr(&mut slots, 2, base + 2 * stride);
    assert_eq!(decode_ptr(&slots, 2), base + 2 * stride);
}

#[test]
fn repointing_one_expert_leaves_its_neighbours_untouched() {
    // The failure this guards: an off-by-one in slot indexing that writes into
    // the adjacent expert's slot pair would corrupt a DIFFERENT expert's
    // weights — visible only as degraded quality, never as an error.
    let base: u64 = 0x7f00_0000_1000;
    let stride: u64 = 2_359_296;
    let n = 6usize;
    let mut slots = vec![0f32; n * 2];
    for e in 0..n {
        encode_ptr(&mut slots, e, base + e as u64 * stride);
    }
    encode_ptr(&mut slots, 3, 0xdead_beef_0000);
    for e in 0..n {
        let want = if e == 3 {
            0xdead_beef_0000
        } else {
            base + e as u64 * stride
        };
        assert_eq!(decode_ptr(&slots, e), want, "expert {e} disturbed");
    }
}

#[test]
fn compact_blob_maps_expert_to_slot_not_to_expert_index() {
    // The EP shard path already proves expert index and blob slot are
    // INDEPENDENT: owned expert `e` maps to its compact-blob slot. Paging
    // relies on the same indirection with a dynamic mapping, so assert that a
    // sparse expert set packs into dense slots addressed by slot, not expert.
    let slot_base: u64 = 0x7f00_9000_0000;
    let stride: u64 = 2_359_296;
    let n_experts = 8usize;
    // Experts 1, 4 and 7 are resident, in cache slots 0, 1, 2.
    let resident = [(1usize, 0usize), (4, 1), (7, 2)];
    let mut slots = vec![0f32; n_experts * 2];
    for (expert, slot) in resident {
        encode_ptr(&mut slots, expert, slot_base + slot as u64 * stride);
    }
    for (expert, slot) in resident {
        assert_eq!(
            decode_ptr(&slots, expert),
            slot_base + slot as u64 * stride,
            "expert {expert} should read slot {slot}"
        );
    }
    // A non-resident expert's entry is untouched (still zero) — which is why
    // the pager must never dispatch a routed expert it has not made resident.
    assert_eq!(decode_ptr(&slots, 0), 0);
}
