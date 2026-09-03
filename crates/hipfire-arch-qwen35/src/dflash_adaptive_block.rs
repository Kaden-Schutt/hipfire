// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Pure (no-GPU) adaptive verify-block controller for DFlash chain spec-decode.
//!
//! Long-context sessions stop paying a fixed B=16 full-KV verify scan per
//! cycle: the controller tracks the trailing [`ADAPTIVE_WINDOW`] verify cycles'
//! accepted-draft counts and shrinks the per-cycle proposal width to just past
//! the observed acceptance depth:
//!
//! ```text
//! τ̂ = Σ accepted / cycles              (trailing 8 verify cycles)
//! effective = clamp(ceil(τ̂) + 2, 2, full)
//! ```
//!
//! The `+2` headroom covers the +1 bonus token plus one row of accept-run
//! jitter, so a meeting-expectations window never starves the verify. The
//! controller seeds at full and holds full until [`ADAPTIVE_WINDOW`] cycles
//! are observed (no shrink on bootstrap noise), and returns full
//! unconditionally when the knob is off — fixed-B behaviour is then identical
//! to before.
//!
//! This generalizes the `dflash_spec_demo --adaptive-b` outer τ-window
//! trip-wire (τ < 2.5 → shrink block to 8) from a single threshold into a
//! continuous depth-following width.
//!
//! Greedy-parity (why shrinking never changes emitted tokens): the draft is
//! causal, so the first k rows of a width-B proposal are identical to a
//! width-k proposal; verify accepts the longest greedy prefix, and the bonus
//! token is the target argmax at the divergence point — the same token the
//! full-width run would have accepted there. Fewer proposed rows can only
//! shorten the accepted run, never change its tokens; only window boundaries
//! move.
//!
//! Env-free by construction: the `HIPFIRE_DFLASH_ADAPTIVE_B=0` kill-switch is
//! resolved by `build_dflash_speculator` (mirroring DSpark's
//! `HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0` in `build_dspark_speculator`), so this
//! controller is unit-testable without a device — see the tests below.

/// Trailing verify cycles of acceptance history before the width may shrink.
pub const ADAPTIVE_WINDOW: usize = 8;
/// Rows proposed past the expected accept run (bonus token + jitter row).
pub const ADAPTIVE_HEADROOM: usize = 2;
/// Floor proposal width: below 2 rows the verify saves nothing worth reseeding.
pub const ADAPTIVE_MIN_BLOCK: usize = 2;

/// Trailing-τ proposal-width controller owned by [`crate::dflash_spec::DflashSpeculator`].
///
/// Fed one `accepted` count per verify cycle (chain and ddtree alike — only
/// the chain proposal width shrinks; the tree budget is structural). Reset to
/// full at request start via `configure_request`; the 8-cycle seed gate then
/// re-arms, so a fresh request never opens shrunk on stale history.
pub struct DflashAdaptiveBlock {
    full: usize,
    enabled: bool,
    counts: [u32; ADAPTIVE_WINDOW],
    len: usize,
    head: usize,
    effective: usize,
}

impl DflashAdaptiveBlock {
    /// `full_block_size` is the configured runtime block (B=16);
    /// `enabled` is the already-resolved knob (load param AND env
    /// kill-switch folded by the constructor — never read env here).
    pub fn new(full_block_size: usize, enabled: bool) -> Self {
        Self {
            full: full_block_size,
            enabled,
            counts: [0; ADAPTIVE_WINDOW],
            len: 0,
            head: 0,
            effective: full_block_size,
        }
    }

    /// Configured full width (B).
    pub fn full(&self) -> usize {
        self.full
    }

    /// Current proposal width: full while disabled, seeding, or recovered.
    pub fn effective(&self) -> usize {
        if !self.enabled {
            return self.full;
        }
        self.effective
    }

    /// Mean accepted drafts over the current window (`None` while seeding).
    pub fn tau_hat(&self) -> Option<f64> {
        if self.len < ADAPTIVE_WINDOW {
            return None;
        }
        let sum: u64 = self.counts.iter().map(|&c| c as u64).sum();
        Some(sum as f64 / ADAPTIVE_WINDOW as f64)
    }

    /// Reseed to full with an empty window (request / conversation start).
    pub fn reset(&mut self) {
        self.counts = [0; ADAPTIVE_WINDOW];
        self.len = 0;
        self.head = 0;
        self.effective = self.full;
    }

    /// Feed one verify cycle's accepted-draft count; returns the proposal
    /// width for the NEXT cycle.
    pub fn observe(&mut self, accepted: usize) -> usize {
        if !self.enabled {
            return self.full;
        }
        self.counts[self.head] = u32::try_from(accepted).unwrap_or(u32::MAX);
        self.head = (self.head + 1) % ADAPTIVE_WINDOW;
        self.len = (self.len + 1).min(ADAPTIVE_WINDOW);
        if self.len < ADAPTIVE_WINDOW {
            // Seeding: hold full until a full window is observed.
            self.effective = self.full;
            return self.effective;
        }
        let tau_hat = self.tau_hat().unwrap_or(0.0);
        self.effective = (tau_hat.ceil() as usize + ADAPTIVE_HEADROOM)
            .clamp(ADAPTIVE_MIN_BLOCK, self.full.max(ADAPTIVE_MIN_BLOCK));
        self.effective
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_window_seeds_at_full() {
        let c = DflashAdaptiveBlock::new(16, true);
        assert_eq!(c.effective(), 16);
        assert_eq!(c.full(), 16);
        assert_eq!(c.tau_hat(), None);
    }

    #[test]
    fn holds_full_until_window_fills() {
        let mut c = DflashAdaptiveBlock::new(16, true);
        for _ in 0..7 {
            assert_eq!(c.observe(1), 16);
        }
        assert_eq!(c.effective(), 16);
        assert_eq!(c.tau_hat(), None);
    }

    #[test]
    fn shrinks_on_low_tau() {
        // Window all-1s: τ̂=1 → ceil(1)+2 = 3.
        let mut c = DflashAdaptiveBlock::new(16, true);
        for _ in 0..8 {
            c.observe(1);
        }
        assert_eq!(c.effective(), 3);
        assert_eq!(c.tau_hat(), Some(1.0));
    }

    #[test]
    fn follows_high_tau() {
        // Window all-8s at full 16: ceil(8)+2 = 10.
        let mut c = DflashAdaptiveBlock::new(16, true);
        for _ in 0..8 {
            c.observe(8);
        }
        assert_eq!(c.effective(), 10);
    }

    #[test]
    fn clamps_to_full_on_high_accept() {
        // ceil(16)+2 = 18 → clamped to full 16.
        let mut c = DflashAdaptiveBlock::new(16, true);
        for _ in 0..8 {
            c.observe(16);
        }
        assert_eq!(c.effective(), 16);
    }

    #[test]
    fn floors_at_two_on_total_reject() {
        // ceil(0)+2 = 2.
        let mut c = DflashAdaptiveBlock::new(16, true);
        for _ in 0..8 {
            c.observe(0);
        }
        assert_eq!(c.effective(), 2);
    }

    #[test]
    fn knob_off_stays_full() {
        let mut c = DflashAdaptiveBlock::new(16, false);
        for _ in 0..16 {
            assert_eq!(c.observe(0), 16);
        }
        assert_eq!(c.effective(), 16);
    }

    #[test]
    fn reset_reseeds_to_full() {
        let mut c = DflashAdaptiveBlock::new(16, true);
        for _ in 0..8 {
            c.observe(1);
        }
        assert_eq!(c.effective(), 3);
        c.reset();
        assert_eq!(c.effective(), 16);
        assert_eq!(c.tau_hat(), None);
        // ...and holds full for another 7 cycles (no instant re-shrink).
        for _ in 0..7 {
            assert_eq!(c.observe(0), 16);
        }
    }

    #[test]
    fn recovers_when_acceptance_returns() {
        let mut c = DflashAdaptiveBlock::new(16, true);
        for _ in 0..8 {
            c.observe(1);
        }
        assert_eq!(c.effective(), 3);
        for _ in 0..8 {
            c.observe(12);
        }
        // τ̂=12 → 14.
        assert_eq!(c.effective(), 14);
    }

    #[test]
    fn small_full_clamps_headroom() {
        // Full 4, all-1s: min(ceil(1)+2, 4) = 3.
        let mut c = DflashAdaptiveBlock::new(4, true);
        for _ in 0..8 {
            c.observe(1);
        }
        assert_eq!(c.effective(), 3);
    }
}
