use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::{
    BindingLayoutFingerprint, CompiledPlan, PlanFingerprint, ReplayBindingError, ReplayBindings,
    Sha256Digest,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BackendKind {
    HipSerial,
    HipMultiStream,
    DirectAql,
}

/// One independently measured candidate run supplied to Auto selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendEvidence {
    pub backend: BackendKind,
    pub speedup_over_hip_serial: f64,
    pub parity_passed: bool,
    pub guards_passed: bool,
    pub same_artifact: bool,
    pub automatic_clocks: bool,
    pub gpu_timestamps_verified: bool,
    pub poisoned: bool,
}

impl BackendEvidence {
    pub fn certified(backend: BackendKind, speedup_over_hip_serial: f64) -> Self {
        Self {
            backend,
            speedup_over_hip_serial,
            parity_passed: true,
            guards_passed: true,
            same_artifact: true,
            automatic_clocks: true,
            gpu_timestamps_verified: true,
            poisoned: false,
        }
    }

    fn passes_gates(self) -> bool {
        self.speedup_over_hip_serial.is_finite()
            && self.speedup_over_hip_serial > 0.0
            && self.parity_passed
            && self.guards_passed
            && self.same_artifact
            && self.automatic_clocks
            && self.gpu_timestamps_verified
            && !self.poisoned
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoPolicy {
    pub minimum_speedup: f64,
    pub minimum_independent_runs: usize,
}

impl Default for AutoPolicy {
    fn default() -> Self {
        Self {
            minimum_speedup: 1.03,
            minimum_independent_runs: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutoDecision {
    pub selected: BackendKind,
    pub lower_bound_speedup: f64,
    pub reason: String,
}

impl AutoPolicy {
    /// Select only backends whose complete evidence set is valid and whose
    /// worst independent run clears the threshold. Any invalid observation for
    /// a candidate fails that candidate closed; HIP serial is unconditional.
    pub fn decide(self, evidence: &[BackendEvidence]) -> AutoDecision {
        let mut passing = Vec::new();
        for backend in [BackendKind::DirectAql, BackendKind::HipMultiStream] {
            let runs = evidence
                .iter()
                .copied()
                .filter(|run| run.backend == backend)
                .collect::<Vec<_>>();
            if runs.len() < self.minimum_independent_runs
                || runs.iter().any(|run| !run.passes_gates())
            {
                continue;
            }
            let lower_bound = runs
                .iter()
                .map(|run| run.speedup_over_hip_serial)
                .fold(f64::INFINITY, f64::min);
            if lower_bound >= self.minimum_speedup {
                passing.push((backend, lower_bound));
            }
        }
        passing.sort_by(|left, right| right.1.total_cmp(&left.1));
        match passing.first().copied() {
            Some((selected, lower_bound_speedup)) => AutoDecision {
                selected,
                lower_bound_speedup,
                reason: format!(
                    "{selected:?} cleared {:.3}x in every certified run",
                    self.minimum_speedup
                ),
            },
            None => AutoDecision {
                selected: BackendKind::HipSerial,
                lower_bound_speedup: 1.0,
                reason: "no accelerated backend cleared all fail-closed gates".to_owned(),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PlanCacheKey {
    plan: PlanFingerprint,
    binding_layout: BindingLayoutFingerprint,
    artifacts: Sha256Digest,
    device: String,
    backend: BackendKind,
}

impl PlanCacheKey {
    pub fn new(
        plan: &CompiledPlan,
        bindings: &ReplayBindings,
        device: impl Into<String>,
        backend: BackendKind,
    ) -> Result<Self, ReplayBindingError> {
        let mut hash = Sha256::new();
        hash.update(b"redline-plan-artifacts-v1\0");
        for (kernel, identity) in plan.artifact_identities() {
            hash.update((kernel.len() as u64).to_le_bytes());
            hash.update(kernel.as_bytes());
            hash.update(identity.code_object().as_bytes());
            hash.update(identity.symbol_text().as_bytes());
            hash.update(identity.generation().to_le_bytes());
        }
        Ok(Self {
            plan: plan.fingerprint(),
            binding_layout: bindings.layout_fingerprint(plan)?,
            artifacts: Sha256Digest::from_bytes(hash.finalize().into()),
            device: device.into(),
            backend,
        })
    }

    pub fn backend(&self) -> BackendKind {
        self.backend
    }
}

struct CacheEntry<T> {
    value: T,
    poisoned: bool,
}

/// In-process prepared-plan cache. A queue fault/timeout poisons the exact key;
/// poisoned values can never be retrieved or silently replaced.
pub struct PreparedPlanCache<T> {
    entries: BTreeMap<PlanCacheKey, CacheEntry<T>>,
}

impl<T> Default for PreparedPlanCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PreparedPlanCache<T> {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, key: PlanCacheKey, value: T) -> Result<(), PlanCacheError> {
        if self.entries.contains_key(&key) {
            return Err(PlanCacheError::AlreadyPresent);
        }
        self.entries.insert(
            key,
            CacheEntry {
                value,
                poisoned: false,
            },
        );
        Ok(())
    }

    pub fn get_mut(&mut self, key: &PlanCacheKey) -> Result<&mut T, PlanCacheError> {
        let entry = self.entries.get_mut(key).ok_or(PlanCacheError::Missing)?;
        if entry.poisoned {
            return Err(PlanCacheError::Poisoned);
        }
        Ok(&mut entry.value)
    }

    pub fn poison(&mut self, key: &PlanCacheKey) -> Result<(), PlanCacheError> {
        let entry = self.entries.get_mut(key).ok_or(PlanCacheError::Missing)?;
        entry.poisoned = true;
        Ok(())
    }

    pub fn remove(&mut self, key: &PlanCacheKey) -> Option<T> {
        self.entries.remove(key).map(|entry| entry.value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PlanCacheError {
    #[error("prepared plan is already cached")]
    AlreadyPresent,
    #[error("prepared plan cache entry is missing")]
    Missing,
    #[error("prepared plan cache entry is poisoned")]
    Poisoned,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_uses_worst_run_and_fails_closed() {
        let policy = AutoPolicy::default();
        let pass = [
            BackendEvidence::certified(BackendKind::DirectAql, 1.08),
            BackendEvidence::certified(BackendKind::DirectAql, 1.07),
        ];
        assert_eq!(policy.decide(&pass).selected, BackendKind::DirectAql);
        assert_eq!(policy.decide(&pass).lower_bound_speedup, 1.07);

        let mut invalid = pass;
        invalid[1].guards_passed = false;
        assert_eq!(policy.decide(&invalid).selected, BackendKind::HipSerial);

        let noisy = [
            BackendEvidence::certified(BackendKind::DirectAql, 1.20),
            BackendEvidence::certified(BackendKind::DirectAql, 1.02),
        ];
        assert_eq!(policy.decide(&noisy).selected, BackendKind::HipSerial);
    }

    #[test]
    fn cache_poison_is_sticky_until_explicit_removal() {
        let key = PlanCacheKey {
            plan: PlanFingerprint::from_bytes_for_test([1; 32]),
            binding_layout: BindingLayoutFingerprint::from_bytes_for_test([2; 32]),
            artifacts: Sha256Digest::from_bytes([3; 32]),
            device: "gfx1201:0000:03:00.0".to_owned(),
            backend: BackendKind::DirectAql,
        };
        let mut cache = PreparedPlanCache::new();
        cache.insert(key.clone(), 7_u32).unwrap();
        assert_eq!(*cache.get_mut(&key).unwrap(), 7);
        cache.poison(&key).unwrap();
        assert_eq!(cache.get_mut(&key).unwrap_err(), PlanCacheError::Poisoned);
        assert_eq!(cache.remove(&key), Some(7));
    }
}
