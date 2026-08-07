// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Host-side mirror of the multi-slot KV descriptor and the flat row-tile list
// that drives batched attention launches.
//
// A "row tile" is up to BR consecutive query rows belonging to ONE slot. No
// tile may span a slot boundary — a workgroup owns one tile and reads one
// slot's KV, so a straddling tile would read the wrong sequence's cache.

/// Byte-identical mirror of `struct KvSlotDesc` in `kernels/src/kv_slot_desc.h`.
/// 24 bytes, 8-byte aligned. Changing either side without the other silently
/// corrupts every KV address.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSlotDesc {
    /// Byte offset of this slot's K slab within the layer's K arena.
    pub k_base: u64,
    /// Byte offset of this slot's V slab within the layer's V arena.
    pub v_base: u64,
    /// Logical KV length. The kernel reads positions `[0, seq_len)`.
    pub seq_len: i32,
    /// Physical slab capacity in tokens. Invariant: `seq_len <= cap`.
    pub cap: i32,
}

/// Total query rows across all slots.
pub fn total_rows(slot_query_counts: &[usize]) -> usize {
    slot_query_counts.iter().sum()
}

/// Minimal f32 -> IEEE binary16 bit pattern (round-toward-zero mantissa).
/// Only needs to cover the small positive scales used by the correctness
/// harness and `test_q8_flash_prefill`. Moved here (from a private copy in
/// `examples/test_q8_flash_prefill.rs`) so Task 7's harness and Task 8's
/// benchmark share one implementation rather than drifting apart.
pub fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits & 0x007F_FFFF) >> 13;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}

/// Build one KV arena holding `seq_lens.len()` contiguous slabs and the
/// matching descriptor table. Each slab is `cap` tokens; `cap` is rounded up
/// so a future page size divides it (spec §6.4).
///
/// `poison_except`: when `Some(target)`, every slab other than `target` is
/// filled with NaN-producing bytes. Used by the isolation test.
///
/// Slab contents vary by slot index — identical slabs would let a cross-slot
/// addressing bug pass by symmetry.
///
/// Deviation from the task-7 brief's worked example: the brief's loop counts
/// 34-byte blocks as `cap * per_pos_bytes / 34` (floor division). For Q8_0
/// strides (always an exact multiple of 34) that is harmless, but asym3's K
/// stride (`n_kv_heads * (4 + head_dim*3/8)`, e.g. 200 or 400 B/pos at
/// hd=256) is NOT a multiple of 34, so floor division silently
/// under-allocates a slot's slab by up to 33 bytes. For a slot whose
/// `seq_len` lands exactly on `cap` (common in these shapes: 512, 8192,
/// 1024, 2048, 4096 are all multiples of PAGE_TOKENS), that shortfall is
/// inside the very last position this function will actually address,
/// which — for the LAST slot in the arena — is an out-of-bounds device
/// read past the end of the uploaded buffer. Using `div_ceil` instead
/// (over-allocating by at most 33 never-addressed padding bytes) removes
/// that risk without changing anything an addressing-correctness test can
/// observe.
pub fn build_arena(
    seq_lens: &[usize],
    per_pos_bytes: usize,
    poison_except: Option<usize>,
) -> (Vec<u8>, Vec<KvSlotDesc>) {
    const PAGE_TOKENS: usize = 128; // == TILE_SIZE, so pages divide slabs later
    let mut arena = Vec::new();
    let mut descs = Vec::with_capacity(seq_lens.len());
    for (slot, &sl) in seq_lens.iter().enumerate() {
        let cap = sl.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let base = arena.len() as u64;
        let poisoned = poison_except.is_some_and(|t| t != slot);
        let n_blocks = (cap * per_pos_bytes).div_ceil(34);
        for blk_idx in 0..n_blocks {
            // f16 scale: 0x7E00 is NaN; otherwise a per-slot varying value.
            let (lo, hi) = if poisoned {
                (0x00u8, 0x7Eu8)
            } else {
                let h = half_from_f32(0.02 + (((blk_idx + slot * 7) % 13) as f32) * 0.005);
                ((h & 0xFF) as u8, (h >> 8) as u8)
            };
            arena.push(lo);
            arena.push(hi);
            for j in 0..32 {
                arena.push((((blk_idx * 31 + j * 17 + slot * 101) % 251) as i32 - 125) as i8 as u8);
            }
        }
        descs.push(KvSlotDesc {
            k_base: base,
            v_base: base, // K and V arenas are separate buffers, same offsets
            seq_len: sl as i32,
            cap: cap as i32,
        });
    }
    (arena, descs)
}

/// Build an asym3 K arena. NOT the same generator as [`build_arena`] — found
/// necessary empirically while running the Task 7 harness, not part of the
/// brief's original design.
///
/// asym3's K layout (`kernels/src/attention_flash_asym3_tile_batched.hip`)
/// is `[4-byte cnorm f32][packed 3-bit body]` per (position, kv_head), with
/// `cnorm` read via `*(const float*)kb` at a byte offset that is NOT a
/// multiple of 34 (`k_bytes_per_head = 4 + head_dim*3/8`, e.g. 100 at
/// hd=256 — coprime-ish with 34). [`build_arena`]'s generic filler only
/// controls the FIRST 2 bytes of each 34-byte block (an f16 "scale" — safe
/// for Q8_0, whose scale read is always aligned to that same 2-byte field);
/// every other byte, including whichever 4 bytes a given `cnorm` read
/// lands on, comes from an unconstrained pseudo-random byte formula. Read
/// as an IEEE-754 f32, 4 essentially-random bytes have a real chance
/// (roughly 1/256 per read, from the exponent byte alone) of landing on a
/// subnormal, infinity, or NaN bit pattern. At the scale of hundreds of
/// `cnorm` reads per shape (`cap * n_kv_heads` positions), this fired in
/// practice: an isolation-test run reported "NaN leaked from a neighbouring
/// slot" that traced back to a `cnorm` value that was ALREADY non-finite in
/// the clean (unpoisoned) run — not a leak at all. The golden-equivalence
/// test did not catch it beforehand because `assert_close`'s comparison
/// (`(g - w).abs() / tol`) is silently vacuous when both `g` and `w` are
/// NaN (`NaN > worst` is always false in IEEE comparisons), so a candidate
/// and reference that independently compute the same garbage from the same
/// bytes both report a perfect match.
///
/// This generator instead constructs `cnorm` explicitly as a small, finite,
/// per-slot/position/head-varying f32 (so it can never land on a special
/// bit pattern by chance) and leaves the packed 3-bit body bytes
/// pseudo-random and unconstrained (safe regardless of value: they only
/// ever select one of 8 bounded entries from `TURBO_C3_256`).
pub fn build_asym3_k_arena(
    seq_lens: &[usize],
    n_kv_heads: usize,
    head_dim: usize,
    poison_except: Option<usize>,
) -> (Vec<u8>, Vec<KvSlotDesc>) {
    const PAGE_TOKENS: usize = 128;
    let head_bytes = 4 + (head_dim * 3) / 8;
    let mut arena = Vec::new();
    let mut descs = Vec::with_capacity(seq_lens.len());
    for (slot, &sl) in seq_lens.iter().enumerate() {
        let cap = sl.div_ceil(PAGE_TOKENS) * PAGE_TOKENS;
        let base = arena.len() as u64;
        let poisoned = poison_except.is_some_and(|t| t != slot);
        for pos in 0..cap {
            for kvh in 0..n_kv_heads {
                let cnorm: f32 = if poisoned {
                    f32::NAN
                } else {
                    0.02 + (((pos + kvh * 13 + slot * 7) % 13) as f32) * 0.005
                };
                arena.extend_from_slice(&cnorm.to_ne_bytes());
                for j in 0..(head_bytes - 4) {
                    arena.push((((pos * 31 + j * 17 + slot * 101 + kvh * 53) % 251) as i32 - 125) as i8 as u8);
                }
            }
        }
        descs.push(KvSlotDesc {
            k_base: base,
            v_base: base,
            seq_len: sl as i32,
            cap: cap as i32,
        });
    }
    (arena, descs)
}

/// Build the flat tile list. Returns `(tile_slot, tile_row0, tile_qbase)`:
///
/// - `tile_slot[t]`  — slot index owning tile `t`
/// - `tile_row0[t]`  — first query row of tile `t` *within its slot*
/// - `tile_qbase[t]` — first query row of tile `t` in the *global* flat row
///   space, which is how `q` and `out` are indexed
///
/// Both row indices are needed: KV addressing is slot-relative (via the
/// descriptor's `seq_len`) while Q/out addressing is global. Conflating them
/// makes slot 0 correct and every later slot read the wrong query.
///
/// Slots with zero query rows produce no tiles — an empty tile would read
/// uninitialised Q and write garbage into `out`.
pub fn build_tiles(
    slot_query_counts: &[usize],
    br: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    assert!(br > 0, "br must be positive");
    let mut tile_slot = Vec::new();
    let mut tile_row0 = Vec::new();
    let mut tile_qbase = Vec::new();
    let mut global = 0usize;
    for (slot, &m) in slot_query_counts.iter().enumerate() {
        let mut row0 = 0usize;
        while row0 < m {
            tile_slot.push(slot as i32);
            tile_row0.push(row0 as i32);
            tile_qbase.push((global + row0) as i32);
            row0 += br;
        }
        global += m;
    }
    (tile_slot, tile_row0, tile_qbase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc_is_24_bytes() {
        assert_eq!(std::mem::size_of::<KvSlotDesc>(), 24);
        assert_eq!(std::mem::align_of::<KvSlotDesc>(), 8);
    }

    #[test]
    fn tiles_never_span_a_slot() {
        // 3 slots with 1, 3 and 8 query rows; BR = 4.
        // Slot 0 -> 1 tile, slot 1 -> 1 tile, slot 2 -> 2 tiles. Total 4.
        let (tile_slot, tile_row0, _) = build_tiles(&[1, 3, 8], 4);
        assert_eq!(tile_slot, vec![0, 1, 2, 2]);
        assert_eq!(tile_row0, vec![0, 0, 0, 4]);
    }

    #[test]
    fn tile_qbase_is_the_global_flat_row() {
        // Same shape: global flat rows are 0 | 1,2,3 | 4..11, so the four
        // tiles start at global rows 0, 1, 4 and 8.
        let (_, _, tile_qbase) = build_tiles(&[1, 3, 8], 4);
        assert_eq!(tile_qbase, vec![0, 1, 4, 8]);
    }

    #[test]
    fn br_one_gives_one_tile_per_row() {
        let (tile_slot, tile_row0, tile_qbase) = build_tiles(&[1, 1, 1, 1], 1);
        assert_eq!(tile_slot, vec![0, 1, 2, 3]);
        assert_eq!(tile_row0, vec![0, 0, 0, 0]);
        assert_eq!(tile_qbase, vec![0, 1, 2, 3]);
    }

    #[test]
    fn zero_query_slots_produce_no_tiles() {
        // A slot with nothing to do this step must not get a tile — an empty
        // tile would read uninitialised Q and write garbage to out.
        // Slot 2's rows still start at global row 2, after slot 0's two rows.
        let (tile_slot, tile_row0, tile_qbase) = build_tiles(&[2, 0, 3], 4);
        assert_eq!(tile_slot, vec![0, 2]);
        assert_eq!(tile_row0, vec![0, 0]);
        assert_eq!(tile_qbase, vec![0, 2]);
    }

    #[test]
    fn total_rows_sums_query_counts() {
        assert_eq!(total_rows(&[1, 3, 8]), 12);
        assert_eq!(total_rows(&[]), 0);
    }

    #[test]
    fn mixed_prefill_and_decode_batch() {
        // The shape SP1 exists for: slot 0 verifies 8 draft tokens, slot 1
        // chunk-prefills 256, slots 2-3 decode 1 each. BR = 8.
        let (tile_slot, _, _) = build_tiles(&[8, 256, 1, 1], 8);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 0).count(), 1);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 1).count(), 32);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 2).count(), 1);
        assert_eq!(tile_slot.iter().filter(|&&s| s == 3).count(), 1);
        assert_eq!(tile_slot.len(), 35);
    }
}
