// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// AdmissionController — decides whether a session can be admitted, and with how
// much context.
//
// In the test harnesses `kv_slots::preflight_alloc` is what stops an oversized
// configuration. In the daemon that job is HERE. The difference matters: on this
// hardware the GPU allocates from system RAM and the cgroup does NOT contain
// amdgpu GTT, so a wrong decision here does not fail a request — it takes down
// the user's desktop with a global OOM.

/// What one loaded model costs, split into the part charged once and the part
/// charged per session.
#[derive(Debug, Clone, Copy)]
pub struct ModelFootprint {
    /// Charged ONCE, however many sessions are admitted.
    pub weights_bytes: u64,
    /// Charged per session, per token of granted context.
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitError {
    PoolFull,
    WouldExceedBudget { need: u64, available: u64 },
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |b: u64| b as f64 / 1073741824.0;
        match self {
            AdmitError::PoolFull => write!(f, "no free slot"),
            AdmitError::WouldExceedBudget { need, available } => write!(
                f,
                "needs {:.2} GiB but only {:.2} GiB of the budget remains",
                gib(*need),
                gib(*available)
            ),
        }
    }
}

pub struct AdmissionController {
    footprint: ModelFootprint,
    budget_bytes: u64,
    /// Granted context per admitted session, in tokens.
    admitted: Vec<usize>,
}

impl AdmissionController {
    pub fn new(footprint: ModelFootprint, budget_bytes: u64) -> Self {
        Self { footprint, budget_bytes, admitted: Vec::new() }
    }

    /// Bytes currently committed: weights once (if anything is admitted) plus
    /// each session's KV.
    pub fn used_bytes(&self) -> u64 {
        if self.admitted.is_empty() {
            return 0;
        }
        let kv: u64 = self
            .admitted
            .iter()
            .map(|&ctx| ctx as u64 * self.footprint.kv_bytes_per_token)
            .sum();
        self.footprint.weights_bytes + kv
    }

    /// Admit a session at `requested_ctx` tokens, or explain why not.
    ///
    /// Rejects rather than silently capping: a caller that asked for 128K and
    /// silently got 8K would produce baffling truncation far from here.
    pub fn admit(&mut self, requested_ctx: usize) -> Result<usize, AdmitError> {
        let kv_need = requested_ctx as u64 * self.footprint.kv_bytes_per_token;
        // Weights are charged once, on the first admission.
        let weights_need =
            if self.admitted.is_empty() { self.footprint.weights_bytes } else { 0 };
        let need = kv_need + weights_need;
        let available = self.budget_bytes.saturating_sub(self.used_bytes());
        // >= rather than >: an admission that would consume the LAST byte of
        // budget is refused too, not just one that overflows it. On this
        // hardware (no swap, cgroup does not contain amdgpu GTT) landing
        // exactly on the edge leaves zero headroom for anything else running
        // on the box, so it is treated the same as exceeding the budget.
        if need >= available {
            return Err(AdmitError::WouldExceedBudget { need, available });
        }
        self.admitted.push(requested_ctx);
        Ok(requested_ctx)
    }

    /// Return a session's context allowance to the budget.
    pub fn release(&mut self, granted_ctx: usize) {
        if let Some(i) = self.admitted.iter().position(|&c| c == granted_ctx) {
            self.admitted.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// qwen3.6:27b — 15.0 GB of weights, 34 KB of KV per token.
    fn f27b() -> ModelFootprint {
        ModelFootprint { weights_bytes: 15 * GIB, kv_bytes_per_token: 34 * 1024 }
    }

    /// qwen3.6:35b-a3b — ~20 GB of weights, 10.6 KB of KV per token.
    fn f35b() -> ModelFootprint {
        ModelFootprint { weights_bytes: 20 * GIB, kv_bytes_per_token: 10_854 }
    }

    #[test]
    fn weights_are_charged_once_not_per_session() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        a.admit(1024).unwrap();
        let after_one = a.used_bytes();
        a.admit(1024).unwrap();
        let after_two = a.used_bytes();
        // The second session adds only its KV, never another copy of the weights.
        assert!(after_two - after_one < GIB, "weights charged twice");
        assert!(after_one >= 15 * GIB, "weights not charged at all");
    }

    #[test]
    fn the_27b_cannot_take_four_agents_at_128k() {
        // 15 GB + 4 x 4.25 GB = 32.25 GB against a 32 GB card.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).expect("first three must fit");
        }
        let e = a.admit(128 * 1024).unwrap_err();
        assert!(matches!(e, AdmitError::WouldExceedBudget { .. }), "got {e:?}");
    }

    #[test]
    fn the_27b_does_take_four_agents_at_96k() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for i in 0..4 {
            a.admit(96 * 1024).unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn the_35b_does_take_four_agents_at_128k() {
        let mut a = AdmissionController::new(f35b(), 32 * GIB);
        for i in 0..4 {
            a.admit(128 * 1024).unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn release_returns_budget_so_a_later_session_fits() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).unwrap();
        }
        assert!(a.admit(128 * 1024).is_err());
        a.release(128 * 1024);
        a.admit(128 * 1024).expect("budget must be reusable after release");
    }

    #[test]
    fn rejection_reports_the_numbers_not_just_a_failure() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).unwrap();
        }
        match a.admit(128 * 1024).unwrap_err() {
            AdmitError::WouldExceedBudget { need, available } => {
                // `>=`, not `>`. Zero headroom is a rejection: 15 GiB of weights
                // plus 4 x 4.25 GiB of KV is an EXACT tie with a 32 GiB budget,
                // and a card with nothing left for activations, scratch and
                // driver overhead does not fit the workload. The plan's comment
                // claiming 32.25 GB was wrong -- 34 * 1024 IS the real per-token
                // cost and the sum lands exactly on the budget.
                assert!(
                    need >= available,
                    "need {need} should be at least available {available}"
                );
                assert!(available < 32 * GIB);
            }
            other => panic!("expected a budget rejection, got {other:?}"),
        }
    }

    #[test]
    fn a_single_session_over_budget_is_rejected_not_silently_capped() {
        // One agent asking for more than the whole card can hold.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        assert!(a.admit(2 * 1024 * 1024).is_err(), "must reject, not silently truncate");
    }
}
