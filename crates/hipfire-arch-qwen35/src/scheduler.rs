// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Scheduler — decides what goes into each step's SlotBatch.
//
// Deliberately minimal: round-robin, chunked prefill mixed with decode, no
// preemption or priorities. Batching wins ~1.36x at 8 slots and recovers
// under a tenth of the available bandwidth headroom, so contorting the
// scheduler to maximise batch width is not where the remaining performance
// is. Correctness and not blocking other slots matter more.
//
// The one property that matters: a long prompt is chunked to `chunk_size`
// tokens per step so it cannot block other slots (in particular decoding
// slots) for its whole duration. See `prefill_and_decode_mix_in_one_batch`.

use crate::slot_batch::SlotBatch;
use rdna_compute::slot_pool::SlotId;

pub struct Scheduler {
    pub chunk_size: usize,
}

/// One slot's outstanding work as seen by the scheduler.
pub struct PendingWork {
    pub slot: SlotId,
    /// Prompt tokens not yet fed through prefill. Drained front-to-back as
    /// chunks are taken.
    pub remaining_prompt: Vec<u32>,
    /// Next absolute position this slot will occupy.
    pub next_pos: usize,
    /// Once prefill is complete, the slot decodes one token per step.
    pub decoding: bool,
}

impl Scheduler {
    /// Build the next step's `SlotBatch`, taking `min(chunk_size,
    /// remaining_prompt.len())` tokens from each prefilling slot and one
    /// token from each decoding slot, in slot order. Advances `next_pos`
    /// (and drains `remaining_prompt`) for whatever it takes.
    pub fn next_batch(&mut self, work: &mut [PendingWork]) -> SlotBatch {
        let mut b = SlotBatch::default();
        for w in work.iter_mut() {
            let take = if w.decoding {
                w.remaining_prompt.len().min(1)
            } else {
                w.remaining_prompt.len().min(self.chunk_size)
            };
            let toks: Vec<u32> = w.remaining_prompt.drain(..take).collect();
            let start_pos = w.next_pos;
            b.m_per_slot.push(toks.len());
            for (i, t) in toks.iter().enumerate() {
                b.tokens.push(*t);
                b.positions.push((start_pos + i) as i32);
                b.row_slot.push(w.slot.0 as i32);
            }
            w.next_pos += toks.len();
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::slot_pool::SlotId;

    fn prompt(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn a_long_prompt_is_chunked_not_run_whole() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(1000),
            next_pos: 0,
            decoding: false,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(
            b.total_rows(),
            256,
            "must take one chunk, not the whole prompt"
        );
        assert_eq!(work[0].remaining_prompt.len(), 744);
        assert_eq!(work[0].next_pos, 256);
    }

    #[test]
    fn prefill_and_decode_mix_in_one_batch() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![
            PendingWork {
                slot: SlotId(0),
                remaining_prompt: prompt(300),
                next_pos: 0,
                decoding: false,
            },
            PendingWork {
                slot: SlotId(1),
                remaining_prompt: vec![42],
                next_pos: 10,
                decoding: true,
            },
        ];
        let b = s.next_batch(&mut work);
        assert_eq!(
            b.m_per_slot,
            vec![256, 1],
            "a prefilling slot must not block a decoding one"
        );
    }

    #[test]
    fn a_prompt_shorter_than_a_chunk_completes_in_one_batch() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: prompt(10),
            next_pos: 0,
            decoding: false,
        }];
        let b = s.next_batch(&mut work);
        assert_eq!(b.total_rows(), 10);
        assert!(work[0].remaining_prompt.is_empty());
    }

    #[test]
    fn an_idle_slot_contributes_nothing() {
        let mut s = Scheduler { chunk_size: 256 };
        let mut work = vec![PendingWork {
            slot: SlotId(0),
            remaining_prompt: vec![],
            next_pos: 0,
            decoding: false,
        }];
        let b = s.next_batch(&mut work);
        assert!(b.is_empty());
    }
}
