// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Upstream-style CPU n-gram modular hash pool (llama.cpp `ngram-mod`).
//!
//! Direct-mapped table: hash(`n_match` prior tokens) → next token. Empty
//! sentinel is [`u32::MAX`]. Production defaults match the daemon-ready
//! primitive: 2²² slots (~16 MiB), `n_match=24`, draft gate `n_min=48`,
//! cap `n_max=64`. Occupancy above 25% and five consecutive low-acceptance
//! attempts (`drafted > 0 && accepted * 4 < drafted`) both clear the table.
//!
//! Pure host Rust. Existing [`crate::spec::PldMatcher`] / [`crate::spec::NgramCache`]
//! are unchanged for DFlash and legacy standalone users.

/// Multiplier for the modular rolling hash (wrapping `u64`).
pub const HASH_MUL: u64 = 6_364_136_223_846_793_005;

/// Empty direct-map slot.
pub const EMPTY: u32 = u32::MAX;

/// Configuration for [`NgramModPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramModConfig {
    /// Power-of-two direct-map capacity (number of `u32` slots).
    pub capacity: usize,
    /// Exact n-gram window length hashed as the key.
    pub n_match: usize,
    /// Minimum chained draft tokens required for a successful [`NgramModPool::draft`].
    pub n_min: usize,
    /// Hard cap on chained draft tokens (also limited by caller_max).
    pub n_max: usize,
}

impl Default for NgramModConfig {
    fn default() -> Self {
        Self {
            capacity: 1 << 22,
            n_match: 24,
            n_min: 48,
            n_max: 64,
        }
    }
}

/// Direct-mapped n-gram → next-token pool.
pub struct NgramModPool {
    config: NgramModConfig,
    /// `capacity` slots; `EMPTY` means unoccupied.
    table: Vec<u32>,
    /// Exact number of non-empty slots.
    occupied: usize,
    /// Bit mask for capacity (power of two).
    mask: usize,
    /// `HASH_MUL.wrapping_pow((n_match - 1) as u32)` for rolling updates.
    mul_pow: u64,
    /// Consecutive low-acceptance draft attempts; pool clears at 5.
    low_accept_streak: u32,
}

impl NgramModPool {
    /// Build a pool from `config`. Validates power-of-two capacity and draft bounds.
    pub fn new(config: NgramModConfig) -> Result<Self, &'static str> {
        if config.capacity == 0 || !config.capacity.is_power_of_two() {
            return Err("ngram_mod capacity must be a non-zero power of two");
        }
        if config.n_match == 0 {
            return Err("ngram_mod n_match must be >= 1");
        }
        if config.n_max == 0 {
            return Err("ngram_mod n_max must be >= 1");
        }
        if config.n_min > config.n_max {
            return Err("ngram_mod n_min must be <= n_max");
        }
        let mul_pow = if config.n_match <= 1 {
            1
        } else {
            HASH_MUL.wrapping_pow((config.n_match - 1) as u32)
        };
        Ok(Self {
            config,
            table: vec![EMPTY; config.capacity],
            occupied: 0,
            mask: config.capacity - 1,
            mul_pow,
            low_accept_streak: 0,
        })
    }

    /// Borrow the config used to construct this pool.
    #[inline]
    pub fn config(&self) -> &NgramModConfig {
        &self.config
    }

    /// Exact number of non-empty table slots.
    #[inline]
    pub fn occupied(&self) -> usize {
        self.occupied
    }

    /// Zero the table, occupancy, and low-acceptance streak.
    pub fn clear(&mut self) {
        self.table.fill(EMPTY);
        self.occupied = 0;
        self.low_accept_streak = 0;
    }

    /// Index every next-token position `j` in
    /// `max(next_token_start, n_match) .. context.len()` as
    /// `context[j - n_match .. j] → context[j]`.
    ///
    /// Uses a rolling hash across consecutive windows. After the batch, if
    /// occupancy is strictly above `capacity / 4`, the pool is cleared.
    pub fn insert_range(&mut self, context: &[u32], next_token_start: usize) {
        let n = self.config.n_match;
        let start = next_token_start.max(n);
        if start >= context.len() {
            return;
        }

        // Initial key window context[start - n .. start].
        let mut hash = hash_window(&context[start - n..start]);
        let mut j = start;
        loop {
            self.store(hash, context[j]);
            j += 1;
            if j >= context.len() {
                break;
            }
            // Slide: drop context[j - 1 - n], append context[j - 1].
            let old = context[j - 1 - n];
            let newly = context[j - 1];
            hash = roll_hash(hash, old, newly, self.mul_pow);
            debug_assert_eq!(
                hash,
                hash_window(&context[j - n..j]),
                "rolling hash must match full recompute"
            );
        }

        if self.occupied > self.config.capacity / 4 {
            // A storage reset also clears the low-acceptance streak.
            self.clear();
        }
    }

    /// Chain table hits by sliding the exact `n_match`-token window over
    /// `context`'s suffix. Returns `None` unless at least `n_min` tokens
    /// chain; length is capped by `n_max` and `caller_max`.
    ///
    /// The only heap allocation is the returned candidate `Vec` (capacity
    /// reserved once up front).
    pub fn draft(&self, context: &[u32], caller_max: usize) -> Option<Vec<u32>> {
        let n = self.config.n_match;
        let limit = self.config.n_max.min(caller_max);
        if limit == 0 || context.len() < n || self.config.n_min > limit {
            return None;
        }

        // Only the returned candidate Vec is heap-allocated. The sliding key is
        // tracked purely as a rolling hash; the token that ages out of the
        // window is read from `context` for the first `n` steps and from
        // previously drafted tokens thereafter.
        let base = context.len() - n;
        let mut hash = hash_window(&context[base..]);
        let mut out: Vec<u32> = Vec::with_capacity(limit);

        for k in 0..limit {
            let idx = (hash as usize) & self.mask;
            let tok = self.table[idx];
            if tok == EMPTY {
                break;
            }
            out.push(tok);

            let old = if k < n {
                context[base + k]
            } else {
                out[k - n]
            };
            hash = roll_hash(hash, old, tok, self.mul_pow);
        }

        if out.len() < self.config.n_min {
            None
        } else {
            Some(out)
        }
    }

    /// Record one draft attempt's `(drafted, accepted)` counts.
    ///
    /// A low-acceptance attempt is `drafted > 0 && accepted * 4 < drafted`.
    /// Each low attempt increments a consecutive streak; any other attempt
    /// resets the streak to zero. On streak 5 the pool is cleared and this
    /// returns `true`; otherwise it returns `false`.
    pub fn record_draft_result(&mut self, drafted: u32, accepted: u32) -> bool {
        // accepted/drafted < 1/4  ⇔  accepted * 4 < drafted (drafted > 0).
        let low = drafted > 0 && u64::from(accepted).saturating_mul(4) < u64::from(drafted);
        if !low {
            self.low_accept_streak = 0;
            return false;
        }
        self.low_accept_streak = self.low_accept_streak.saturating_add(1);
        if self.low_accept_streak < 5 {
            return false;
        }
        self.clear();
        true
    }

    #[inline]
    fn store(&mut self, hash: u64, next: u32) {
        let idx = (hash as usize) & self.mask;
        if self.table[idx] == EMPTY {
            self.occupied += 1;
        }
        self.table[idx] = next;
    }
}

/// Full modular hash of an `n_match`-token window.
#[inline]
fn hash_window(tokens: &[u32]) -> u64 {
    let mut h = 0u64;
    for &t in tokens {
        h = h.wrapping_mul(HASH_MUL).wrapping_add(u64::from(t));
    }
    h
}

/// Slide one token: drop `old` from the left, append `new_tok` on the right.
#[inline]
fn roll_hash(hash: u64, old: u32, new_tok: u32, mul_pow: u64) -> u64 {
    // h' = (h - old * M^{n-1}) * M + new
    hash.wrapping_sub(u64::from(old).wrapping_mul(mul_pow))
        .wrapping_mul(HASH_MUL)
        .wrapping_add(u64::from(new_tok))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(capacity: usize, n_match: usize, n_min: usize, n_max: usize) -> NgramModConfig {
        NgramModConfig {
            capacity,
            n_match,
            n_min,
            n_max,
        }
    }

    #[test]
    fn default_config_matches_production() {
        let d = NgramModConfig::default();
        assert_eq!(d.capacity, 1 << 22);
        assert_eq!(d.n_match, 24);
        assert_eq!(d.n_min, 48);
        assert_eq!(d.n_max, 64);
    }

    #[test]
    fn new_rejects_invalid_config() {
        assert!(NgramModPool::new(small_cfg(0, 2, 1, 2)).is_err());
        assert!(NgramModPool::new(small_cfg(3, 2, 1, 2)).is_err()); // not pow2
        assert!(NgramModPool::new(small_cfg(8, 0, 1, 2)).is_err());
        assert!(NgramModPool::new(small_cfg(8, 2, 3, 2)).is_err()); // n_min > n_max
        assert!(NgramModPool::new(small_cfg(8, 2, 1, 0)).is_err());
    }

    #[test]
    fn rolling_hash_equals_recompute() {
        let n = 5usize;
        let mul_pow = HASH_MUL.wrapping_pow((n - 1) as u32);
        let tokens: Vec<u32> = (1..30).collect();
        let mut h = hash_window(&tokens[0..n]);
        assert_eq!(h, hash_window(&tokens[0..n]));
        for j in n..tokens.len() {
            let old = tokens[j - n];
            let newly = tokens[j];
            h = roll_hash(h, old, newly, mul_pow);
            assert_eq!(
                h,
                hash_window(&tokens[j - n + 1..j + 1]),
                "mismatch at j={j}"
            );
        }
    }

    #[test]
    fn insert_range_rolling_matches_per_window_store() {
        // Capacity large enough that 27 inserts stay under capacity/4.
        let cfg = small_cfg(256, 3, 1, 8);
        let mut a = NgramModPool::new(cfg).unwrap();
        let mut b = NgramModPool::new(cfg).unwrap();
        let ctx: Vec<u32> = (10..40).collect();
        a.insert_range(&ctx, 0);
        let n = cfg.n_match;
        for j in n..ctx.len() {
            let h = hash_window(&ctx[j - n..j]);
            b.store(h, ctx[j]);
        }
        assert!(a.occupied() > 0, "must not have occupancy-reset");
        assert_eq!(a.table, b.table);
        assert_eq!(a.occupied(), b.occupied());
    }

    #[test]
    fn min_gate_closure() {
        let cfg = small_cfg(64, 2, 4, 8);
        let mut pool = NgramModPool::new(cfg).unwrap();
        // Train a short chain: only three next-tokens from [1,2].
        let ctx = vec![1, 2, 3, 4, 5];
        pool.insert_range(&ctx, 0);
        // Draft from [1,2] chains 3,4,5 then empty — length 3 < n_min=4.
        assert_eq!(pool.draft(&[1, 2], 8), None);
        // Extend chain to satisfy n_min=4.
        let ctx2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        pool.insert_range(&ctx2, 0);
        let d = pool.draft(&[1, 2], 8).expect("min gate should open");
        assert!(d.len() >= 4);
        assert_eq!(&d[..4], &[3, 4, 5, 6]);
    }

    #[test]
    fn exact_chain_and_caller_cap() {
        let cfg = small_cfg(128, 2, 2, 5);
        let mut pool = NgramModPool::new(cfg).unwrap();
        // Linear chain 0,1,2,...,20
        let ctx: Vec<u32> = (0..21).collect();
        pool.insert_range(&ctx, 0);
        // Full n_max=5
        let d = pool.draft(&[0, 1], 100).unwrap();
        assert_eq!(d, vec![2, 3, 4, 5, 6]);
        // caller_max trims below n_max
        let d2 = pool.draft(&[0, 1], 3).unwrap();
        assert_eq!(d2, vec![2, 3, 4]);
        // caller_max below n_min → None
        assert_eq!(pool.draft(&[0, 1], 1), None);
    }

    #[test]
    fn incremental_range_insertion() {
        let cfg = small_cfg(64, 3, 2, 8);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let ctx = vec![10, 11, 12, 13, 14, 15, 16, 17];
        // First half through index 4 inclusive as next-token positions.
        pool.insert_range(&ctx[..5], 0); // j = 3,4 → (10,11,12)->13, (11,12,13)->14
        assert_eq!(pool.occupied(), 2);
        // Resume from next_token_start = 5
        pool.insert_range(&ctx, 5); // j = 5,6,7
        assert_eq!(pool.occupied(), 5);
        let d = pool.draft(&[10, 11, 12], 8).unwrap();
        assert_eq!(&d[..5], &[13, 14, 15, 16, 17]);
    }

    #[test]
    fn collision_occupancy_exact() {
        // capacity 4 → only 4 slots; force many distinct windows into few buckets.
        let cfg = small_cfg(4, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let mut ctx: Vec<u32> = Vec::new();
        for i in 0..50u32 {
            ctx.push(i.wrapping_mul(17).wrapping_add(3));
        }
        pool.insert_range(&ctx, 0);
        // Either still filled (≤ capacity/4 = 1) or cleared by occupancy reset.
        let occ = pool.occupied();
        assert!(occ <= 4);
        let non_empty = pool.table.iter().filter(|&&t| t != EMPTY).count();
        assert_eq!(occ, non_empty);

        // Overwrite path: same keys must keep occupancy exact.
        if occ > 0 {
            pool.insert_range(&ctx[..4], 0);
            let non_empty2 = pool.table.iter().filter(|&&t| t != EMPTY).count();
            assert_eq!(pool.occupied(), non_empty2);
        }
    }

    #[test]
    fn occupancy_reset_above_quarter() {
        // capacity 16 → threshold capacity/4 = 4; reset when occupied > 4.
        let cfg = small_cfg(16, 1, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        for _ in 0..4 {
            assert!(!pool.record_draft_result(10, 0));
        }
        assert_eq!(pool.low_accept_streak, 4);
        // n_match=1: key is single prior token. Distinct keys 0..8 → 9 inserts.
        let ctx: Vec<u32> = (0..10).collect();
        pool.insert_range(&ctx, 0);
        // After batch occupancy would be 9 > 4 → cleared.
        assert_eq!(pool.occupied(), 0);
        assert!(pool.table.iter().all(|&t| t == EMPTY));
        assert_eq!(pool.low_accept_streak, 0);

        // Exactly 4 distinct keys via first four next-token positions only.
        // ctx2[..5] → j=1..4 keys = 0,10,1,11.
        let mut pool2 = NgramModPool::new(cfg).unwrap();
        let ctx2 = vec![0, 10, 1, 11, 2, 12, 3, 13];
        pool2.insert_range(&ctx2[..5], 0);
        assert_eq!(pool2.occupied(), 4);
        assert!(!pool2.table.iter().all(|&t| t == EMPTY));
    }

    #[test]
    fn consecutive_low_accept_streak_clears() {
        let cfg = small_cfg(32, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let ctx = vec![1, 2, 3, 4, 5, 6];
        pool.insert_range(&ctx, 0);
        assert!(pool.occupied() > 0);

        // Four low attempts — no clear yet.
        for _ in 0..4 {
            assert!(!pool.record_draft_result(10, 0));
        }
        assert!(pool.occupied() > 0);
        assert_eq!(pool.low_accept_streak, 4);

        // A healthy attempt resets the streak.
        assert!(!pool.record_draft_result(10, 5));
        assert_eq!(pool.low_accept_streak, 0);
        let filled = pool.occupied();
        assert!(filled > 0);

        // Five subsequent low attempts clear.
        for i in 0..5 {
            let cleared = pool.record_draft_result(8, 0);
            if i < 4 {
                assert!(!cleared);
                assert_eq!(pool.low_accept_streak, i + 1);
            } else {
                assert!(cleared);
            }
        }
        assert_eq!(pool.occupied(), 0);
        assert_eq!(pool.low_accept_streak, 0);
    }

    #[test]
    fn draft_requires_context_suffix() {
        let cfg = small_cfg(16, 4, 1, 4);
        let pool = NgramModPool::new(cfg).unwrap();
        assert_eq!(pool.draft(&[1, 2, 3], 4), None); // shorter than n_match
        assert_eq!(pool.draft(&[], 4), None);
    }

    #[test]
    fn next_token_start_skips_early_positions() {
        let cfg = small_cfg(64, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let ctx = vec![1, 2, 3, 4, 5, 6];
        // Only index j >= 4.
        pool.insert_range(&ctx, 4);
        // (1,2)->3 and (2,3)->4 must be absent; (3,4)->5 and (4,5)->6 present.
        assert_eq!(pool.draft(&[1, 2], 4), None);
        assert_eq!(pool.draft(&[3, 4], 4).unwrap(), vec![5, 6]);
    }

    #[test]
    fn clear_zeros_table_and_streak() {
        let cfg = small_cfg(16, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        pool.insert_range(&[1, 2, 3, 4], 0);
        assert!(pool.occupied() > 0);
        assert!(!pool.record_draft_result(10, 0));
        assert_eq!(pool.low_accept_streak, 1);
        pool.clear();
        assert_eq!(pool.occupied(), 0);
        assert!(pool.table.iter().all(|&t| t == EMPTY));
        assert_eq!(pool.low_accept_streak, 0);
        // Streak reset: need five more low attempts to clear again.
        pool.insert_range(&[1, 2, 3, 4], 0);
        for _ in 0..4 {
            assert!(!pool.record_draft_result(4, 0));
        }
        assert!(pool.record_draft_result(4, 0));
    }
}
