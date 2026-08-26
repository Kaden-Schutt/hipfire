// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotBatch — one forward step's ragged work across N slots.
//
// A step mixes freely: one slot verifying 8 draft tokens, another
// chunk-prefilling 256, others decoding 1 each. That raggedness is what SP1's
// kernels were built for.

use rdna_compute::slot_pool::SlotId;

#[derive(Debug, Clone, Default)]
pub struct SlotBatch {
    /// Per-slot token counts for this step. 0 means the slot is idle.
    pub m_per_slot: Vec<usize>,
    /// Flat token ids, packed across slots in slot order.
    pub tokens: Vec<u32>,
    /// Per-row ABSOLUTE position within that row's own slot.
    ///
    /// Authoritative for the causal bound — never `desc.seq_len`. The two
    /// differ whenever a slot has more than one query row, and conflating them
    /// caused SP1's only Critical defect.
    pub positions: Vec<i32>,
    /// Slot index for each flat row.
    pub row_slot: Vec<i32>,
}

impl SlotBatch {
    /// Build a step from `(slot, tokens, start_pos)` triples. Slots with no
    /// tokens contribute no rows.
    pub fn build(per_slot: &[(SlotId, &[u32], usize)]) -> Self {
        let mut b = SlotBatch::default();
        for (slot, toks, start_pos) in per_slot {
            b.m_per_slot.push(toks.len());
            for (i, t) in toks.iter().enumerate() {
                b.tokens.push(*t);
                b.positions.push((start_pos + i) as i32);
                b.row_slot.push(slot.0 as i32);
            }
        }
        b
    }

    pub fn total_rows(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::slot_pool::SlotId;

    #[test]
    fn packs_tokens_in_slot_order() {
        let b = SlotBatch::build(&[
            (SlotId(0), &[10u32, 11][..], 100),
            (SlotId(1), &[20u32][..], 5),
        ]);
        assert_eq!(b.tokens, vec![10, 11, 20]);
        assert_eq!(b.m_per_slot, vec![2, 1]);
        assert_eq!(b.total_rows(), 3);
    }

    #[test]
    fn positions_advance_within_a_slot_from_its_own_start() {
        // Slot 0 verifying 3 tokens at start_pos 100 occupies 100,101,102.
        // Slot 1 decoding 1 token at start_pos 5 occupies 5. They are
        // independent -- positions are per-slot absolute, not batch-global.
        let b = SlotBatch::build(&[
            (SlotId(0), &[1u32, 2, 3][..], 100),
            (SlotId(1), &[9u32][..], 5),
        ]);
        assert_eq!(b.positions, vec![100, 101, 102, 5]);
    }

    #[test]
    fn row_slot_maps_every_flat_row_to_its_slot() {
        let b = SlotBatch::build(&[
            (SlotId(0), &[1u32, 2, 3][..], 0),
            (SlotId(2), &[7u32][..], 0),
        ]);
        assert_eq!(b.row_slot, vec![0, 0, 0, 2]);
    }

    #[test]
    fn idle_slots_contribute_no_rows() {
        let b = SlotBatch::build(&[(SlotId(0), &[][..], 0), (SlotId(1), &[5u32][..], 42)]);
        assert_eq!(b.m_per_slot, vec![0, 1]);
        assert_eq!(b.tokens, vec![5]);
        assert_eq!(b.positions, vec![42]);
        assert_eq!(b.row_slot, vec![1]);
    }

    #[test]
    fn an_all_idle_batch_is_empty() {
        let b = SlotBatch::build(&[(SlotId(0), &[][..], 0)]);
        assert!(b.is_empty());
        assert_eq!(b.total_rows(), 0);
    }

    #[test]
    fn mixed_prefill_and_decode_is_the_shape_this_exists_for() {
        // slot 0 verifies 8 draft tokens, slot 1 chunk-prefills 256,
        // slots 2-3 decode 1 each.
        let p0: Vec<u32> = (0..8).collect();
        let p1: Vec<u32> = (0..256).collect();
        let b = SlotBatch::build(&[
            (SlotId(0), &p0[..], 1000),
            (SlotId(1), &p1[..], 0),
            (SlotId(2), &[1u32][..], 50),
            (SlotId(3), &[2u32][..], 77),
        ]);
        assert_eq!(b.total_rows(), 266);
        assert_eq!(b.row_slot.iter().filter(|&&s| s == 1).count(), 256);
        assert_eq!(b.positions[b.positions.len() - 1], 77);
    }
}
