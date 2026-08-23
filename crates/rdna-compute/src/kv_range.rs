// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// KvRangeAllocator — dynamic KV-range allocation over one shared arena.
//
// Instead of fixed per-slot slabs, the whole pool budget (e.g. 100k tokens)
// is ONE address space. Each admitted session claims a contiguous range
// `[byte_off, byte_off + cap*stride)` sized to its ACTUAL need
// (prompt + generation budget), and releases it when the request finishes.
//
// Consequences:
//   * a solo request can use nearly the entire pool;
//   * N concurrent requests split the pool by real demand, not by preset caps;
//   * no kernel changes: descriptors still resolve `k_base + pos*stride`.
//
// Fragmentation (freed gaps smaller than later requests) is handled by
// first-fit plus host-side compaction through the existing swap machinery —
// capture_slot/restore_slot relocate a live session's KV to any new range.

/// One allocated KV window inside the shared arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvRange {
    /// Byte offset of this range's start in each of the K and V arenas.
    pub byte_off: u64,
    /// Token capacity of this range (prompt + max generation budget).
    pub cap_tokens: usize,
}

#[derive(Debug)]
struct FreeInterval {
    off: u64,
    tokens: usize,
}

#[derive(Debug)]
pub struct KvRangeAllocator {
    total_tokens: usize,
    per_pos_bytes: usize,
    free: Vec<FreeInterval>, // sorted by off, non-overlapping, coalesced
}

impl KvRangeAllocator {
    pub fn new(total_tokens: usize, per_pos_bytes: usize) -> Self {
        Self {
            total_tokens,
            per_pos_bytes,
            free: vec![FreeInterval {
                off: 0,
                tokens: total_tokens,
            }],
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn per_pos_bytes(&self) -> usize {
        self.per_pos_bytes
    }

    /// Free tokens currently unallocated.
    pub fn free_tokens(&self) -> usize {
        self.free.iter().map(|f| f.tokens).sum()
    }

    /// Largest contiguous range that could be allocated right now.
    pub fn max_contiguous_tokens(&self) -> usize {
        self.free.iter().map(|f| f.tokens).max().unwrap_or(0)
    }

    /// Allocate a contiguous range of `need` tokens (first-fit). The range's
    /// byte offset is 128-token aligned so a future page size divides it.
    pub fn alloc(&mut self, need: usize) -> Option<KvRange> {
        let need = need.div_ceil(128) * 128;
        // First fit over sorted intervals.
        let idx = self.free.iter().position(|f| f.tokens >= need)?;
        let f = &mut self.free[idx];
        let byte_off = f.off * self.per_pos_bytes as u64;
        let range = KvRange {
            byte_off,
            cap_tokens: need,
        };
        // Shrink or remove the interval from its front.
        if f.tokens == need {
            self.free.remove(idx);
        } else {
            let taken = need as u64;
            f.off += taken;
            f.tokens -= need;
        }
        Some(range)
    }

    /// Return a previously allocated range to the pool. Coalesces with
    /// adjacent free intervals.
    pub fn free(&mut self, range: &KvRange) {
        let off = range.byte_off / self.per_pos_bytes as u64;
        let tokens = range.cap_tokens;
        // Insert sorted; coalesce with neighbours.
        let pos = self
            .free
            .iter()
            .position(|f| f.off > off)
            .unwrap_or(self.free.len());
        self.free.insert(
            pos,
            FreeInterval {
                off,
                tokens,
            },
        );
        self.coalesce_at(pos);
    }

    fn coalesce_at(&mut self, pos: usize) {
        // Coalesce with next.
        if pos + 1 < self.free.len() {
            let cur_end = self.free[pos].off + self.free[pos].tokens as u64;
            if cur_end == self.free[pos + 1].off {
                let next_tokens = self.free.remove(pos + 1).tokens;
                self.free[pos].tokens += next_tokens;
            }
        }
        // Coalesce with prev.
        if pos > 0 {
            let prev = &self.free[pos - 1];
            let prev_end = prev.off + prev.tokens as u64;
            if prev_end == self.free[pos].off {
                let cur_tokens = self.free.remove(pos).tokens;
                self.free[pos - 1].tokens += cur_tokens;
                self.coalesce_at(pos - 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_request_gets_whole_pool() {
        let mut a = KvRangeAllocator::new(100_000, 1088);
        let r = a.alloc(95_000).unwrap();
        assert_eq!(r.byte_off, 0);
        assert_eq!(r.cap_tokens, 95_104); // 95000 rounds up to 128-multiple
        assert_eq!(a.free_tokens(), 100_000 - 95_104);
    }

    #[test]
    fn two_requests_split_by_real_demand() {
        let mut a = KvRangeAllocator::new(100_000, 1088);
        let big = a.alloc(90_000).unwrap();
        let small = a.alloc(8_000).unwrap();
        // big rounded: 90_000 -> 90_112 tokens; small starts right after it.
        assert_eq!(small.byte_off as usize / 1088, 90_112);
        assert_eq!(small.cap_tokens, 8_064);
        assert!(a.alloc(3_000).is_none(), "only ~1.9k left");
        assert_eq!(a.max_contiguous_tokens(), 100_000 - 90_112 - 8_064);
        let _ = big;
    }

    #[test]
    fn free_then_realloc_reuses_space() {
        let mut a = KvRangeAllocator::new(10_000, 1088);
        let r1 = a.alloc(4_000).unwrap();
        let _r2 = a.alloc(4_000).unwrap();
        assert!(a.alloc(3_000).is_none());
        a.free(&r1);
        let r3 = a.alloc(3_000).unwrap();
        assert_eq!(r3.byte_off, 0, "must reuse the freed head");
    }

    #[test]
    fn free_coalesces_adjacent_intervals() {
        // 3900+3840... use sizes that round cleanly: 3 x 3968 (=128*31) = 11904 <= 12000
        let mut a = KvRangeAllocator::new(12_000, 1088);
        let r1 = a.alloc(3_900).unwrap();
        let r2 = a.alloc(3_900).unwrap();
        let _r3 = a.alloc(3_900).unwrap();
        assert!(
            a.alloc(1_200).is_none(),
            "fragmented before frees: only {} left",
            a.free_tokens()
        );
        a.free(&r2);
        a.free(&r1);
        assert_eq!(a.free_tokens(), 12_000 - 3_968);
        let r = a.alloc(7_900).unwrap();
        assert_eq!(r.byte_off, 0);
        let _ = r;
    }

    #[test]
    fn ranges_are_128_aligned_for_future_pages() {
        let mut a = KvRangeAllocator::new(10_000, 1088);
        let r = a.alloc(100).unwrap();
        assert_eq!(r.byte_off % 128, 0);
        assert_eq!(r.cap_tokens, 128);
    }

    #[test]
    fn oversubscribe_is_rejected_not_clamped() {
        let mut a = KvRangeAllocator::new(1_000, 1088);
        assert!(a.alloc(2_000).is_none());
    }
}
