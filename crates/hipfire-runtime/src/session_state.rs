//! Owns every field that must be reset between request contexts. "Total by
//! ownership": a field not here is config or arch-owned. Part of the
//! god-struct-collapse #462 mechanism.
//!
//! `clear()` routes every checkpoint drop through `Snap::free_gpu` to avoid
//! the hipMalloc-OOM leak that a bare `Vec::clear()` causes on long-lived
//! serves (same discipline as `free_checkpoints` in daemon.rs).
//!
//! The two type parameters keep the lib crate free of circular deps: `Snap`
//! is instantiated with `hipfire_arch_qwen35::speculative::DeltaNetSnapshot`
//! and `Cache` with `hipfire_loader::AsstTurnCache` by the daemon/loader,
//! which sit at the top of the DAG where both crates are in scope.
//!
//! Inc 0: standalone, not yet wired into `LoadedModel`. Inc 1 wires it.

/// Every DeltaNet checkpoint snapshot must implement this so `SessionState`
/// can free GPU memory without taking a concrete dep on the arch crate.
pub trait FreeGpu {
    fn free_gpu(self, gpu: &mut rdna_compute::Gpu);
}

pub struct SessionState<Snap, Cache> {
    pub seq_pos: usize,
    pub conversation_tokens: Vec<u32>,
    pub prefill_checkpoints: Vec<(usize, Snap)>,
    pub dflash_checkpoints: Vec<(usize, Snap)>,
    pub asst_turn_cache: Cache,
}

impl<Snap: FreeGpu, Cache> SessionState<Snap, Cache> {
    pub fn new(cache: Cache) -> Self {
        SessionState {
            seq_pos: 0,
            conversation_tokens: Vec::new(),
            prefill_checkpoints: Vec::new(),
            dflash_checkpoints: Vec::new(),
            asst_turn_cache: cache,
        }
    }

    /// CPU-side reset — zeros position, clears conversation history, and
    /// drops the checkpoint vecs. In the unit-test path the vecs are always
    /// empty so no GPU buffers are leaked; in production use `clear()` which
    /// properly frees GPU memory via `FreeGpu::free_gpu`.
    pub fn clear_cpu_side(&mut self) {
        self.seq_pos = 0;
        self.conversation_tokens.clear();
        self.prefill_checkpoints.clear();
        self.dflash_checkpoints.clear();
    }

    /// Full reset: frees checkpoint GPU buffers then zeros CPU-side state.
    /// Mirrors `free_checkpoints` in daemon.rs — routes every drop through
    /// `FreeGpu::free_gpu` so bare `Vec::clear()` cannot orphan HIP allocations.
    pub fn clear(&mut self, gpu: &mut rdna_compute::Gpu) {
        for (_, snap) in self.prefill_checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
        for (_, snap) in self.dflash_checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
        self.seq_pos = 0;
        self.conversation_tokens.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub snapshot for unit tests — no GPU, no external deps.
    struct NoopSnap;
    impl FreeGpu for NoopSnap {
        fn free_gpu(self, _gpu: &mut rdna_compute::Gpu) {}
    }

    fn new_test_session() -> SessionState<NoopSnap, ()> {
        SessionState::new(())
    }

    #[test]
    fn clear_zeroes_position_and_history() {
        let mut s = new_test_session();
        s.seq_pos = 42;
        s.conversation_tokens.extend_from_slice(&[1, 2, 3]);
        s.clear_cpu_side(); // CPU-only variant, no Gpu needed in unit test
        assert_eq!(s.seq_pos, 0);
        assert!(s.conversation_tokens.is_empty());
        assert!(s.prefill_checkpoints.is_empty());
        assert!(s.dflash_checkpoints.is_empty());
    }
}
