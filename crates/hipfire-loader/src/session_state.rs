//! Per-request mutable session state factored out of `LoadedModel`.
//!
//! Owns every field that must be reset between request contexts. Part of the
//! god-struct-collapse (#462) — Inc 0 scaffolding; not yet wired into
//! `LoadedModel` (that is Inc 1).
//!
//! Concrete (no type params): `DeltaNetSnapshot` and `AsstTurnCache` are
//! both visible here at the loader layer, which sits at the top of the DAG
//! where both `hipfire-arch-qwen35` and this crate's own types are in scope.
//!
//! `clear()` routes every checkpoint drop through `DeltaNetSnapshot::free_gpu`
//! to avoid the hipMalloc-OOM leak that a bare `Vec::clear()` causes on
//! long-lived serves (same discipline as `free_checkpoints` in daemon.rs).

use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;

use crate::AsstTurnCache;

#[allow(dead_code)]
pub struct SessionState {
    pub seq_pos: usize,
    pub conversation_tokens: Vec<u32>,
    pub prefill_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub dflash_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub asst_turn_cache: AsstTurnCache,
}

#[allow(dead_code)]
impl SessionState {
    /// Construct with defaults read from environment variables.
    /// `AsstTurnCache::new_from_env()` only reads env vars — no GPU required.
    pub fn new_from_env() -> Self {
        SessionState {
            seq_pos: 0,
            conversation_tokens: Vec::new(),
            prefill_checkpoints: Vec::new(),
            dflash_checkpoints: Vec::new(),
            asst_turn_cache: AsstTurnCache::new_from_env(),
        }
    }

    /// CPU-side reset — zeros position and clears all checkpoint vecs.
    /// Safe to call in unit tests (no GPU). In production, prefer `clear()`
    /// which also frees GPU buffers in the checkpoint snapshots.
    pub fn clear_cpu_side(&mut self) {
        self.seq_pos = 0;
        self.conversation_tokens.clear();
        self.prefill_checkpoints.clear();
        self.dflash_checkpoints.clear();
    }

    /// Full reset: frees checkpoint GPU buffers then zeros CPU-side state.
    /// Mirrors `free_checkpoints` in daemon.rs — routes every drop through
    /// `DeltaNetSnapshot::free_gpu` so bare `Vec::clear()` cannot orphan
    /// HIP allocations (no `Drop` on `DeltaNetSnapshot`).
    pub fn clear(&mut self, _gpu: &mut rdna_compute::Gpu) {
        for (_, snap) in self.prefill_checkpoints.drain(..) {
            snap.free_gpu(_gpu);
        }
        for (_, snap) in self.dflash_checkpoints.drain(..) {
            snap.free_gpu(_gpu);
        }
        self.seq_pos = 0;
        self.conversation_tokens.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_zeroes_position_and_history() {
        let mut s = SessionState::new_from_env();
        s.seq_pos = 42;
        s.conversation_tokens.extend_from_slice(&[1, 2, 3]);
        s.clear_cpu_side();
        assert_eq!(s.seq_pos, 0);
        assert!(s.conversation_tokens.is_empty());
        assert!(s.prefill_checkpoints.is_empty());
        assert!(s.dflash_checkpoints.is_empty());
    }
}
