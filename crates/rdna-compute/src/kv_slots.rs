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
