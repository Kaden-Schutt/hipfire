// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-agnostic speculative-decode policy + result seam types.
//!
//! Relocated from `hipfire-arch-qwen35::speculative` (P1): the KV-cache
//! layout selector, per-step result, rollback-replay / verify-graph modes,
//! the rollback parity decision, and the aggregate step stats. None name a
//! concrete architecture.

/// Which KV cache layout to use when allocating a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// Unquantized FP32 K/V cache. Gold-path verification only.
    Fp32,
    /// INT8 co-located K and V (default).
    Q8,
    /// Asym4: rotated 4-bit K + Q8 V (smaller than Q8, higher-fidelity than asym3).
    Asym4,
    /// Asym3: rotated 3-bit K + Q8 V. ~2.7× less KV BW than Q8, tightly-tuned
    /// kernel for the hot FA attention path. Good choice for long-context verify.
    Asym3,
    /// Asym2: rotated 2-bit K + Q8 V. Smallest but most lossy.
    Asym2,
    /// Fwht4: signed-FWHT-rotated 4-bit K + Q8 V. Byte-identical storage to
    /// Asym4 but with a Hadamard rotation (matches MQ4's weight-quant trick).
    /// Centroid LUTs were always Lloyd-Max-fit for post-FWHT N(0, 1/128) per
    /// turbo_common.h:13 — Fwht4 finally uses them on the distribution they
    /// were calibrated for. Opt-in via `--kv-mode fwht4`.
    Fwht4,
    /// Fwht3: signed-FWHT-256 rotated 3-bit K + Q8 V. Byte-identical storage to
    /// Asym3 (the canonical default). Single-pass 256-element FWHT — the
    /// natural fit for asym3's existing layout (8 dims/thread). Empirical
    /// prose-τ win on 3.5-27b at the 4-bit tier suggests the 3-bit tier
    /// should benefit even more from rotation. Opt-in via `--kv-mode fwht3`.
    Fwht3,
    /// Fwht2: signed-FWHT-128 rotated 2-bit K + Q8 V. Byte-identical storage
    /// to Asym2. 2-pass-over-128 structure matches fwht4. Highest theoretical
    /// leverage tier — Asym2 is doc'd "most lossy" and 2-bit centroid quant
    /// suffers most from outliers. Opt-in via `--kv-mode fwht2`.
    Fwht2,
}

impl Default for KvMode {
    fn default() -> Self {
        KvMode::Q8
    }
}

/// Result of one speculative decode step.
#[derive(Debug, Clone)]
pub struct SpecStepResult {
    /// Number of draft tokens accepted (0..=k).
    pub accepted: usize,
    /// Target's next-token prediction at the first rejection point (or after
    /// all drafted tokens if accepted == k). Appended to `committed`.
    pub bonus_token: u32,
    /// The full sequence of tokens the draft proposed this cycle.
    pub drafted: Vec<u32>,
    /// The tokens actually committed to both models: the seed token, accepted
    /// draft tokens, then `bonus_token` (length = accepted + 2).
    pub committed: Vec<u32>,
    /// DeltaNet rollback replay path used after the over-verifying target pass.
    pub rollback_replay: SpecRollbackReplayKind,
    /// Verify graph path used for the target verifier in this step.
    pub verify_graph_mode: SpecVerifyGraphMode,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpecRollbackReplayKind {
    GdnTape,
    BatchedPrefill,
    FullPrefill,
    PrefixVerify,
    SerialTape,
    VerifyComplete,
}

impl SpecRollbackReplayKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GdnTape => "gdn_tape",
            Self::BatchedPrefill => "batched_prefill",
            Self::FullPrefill => "full_prefill",
            Self::PrefixVerify => "prefix_verify",
            Self::SerialTape => "serial_tape",
            Self::VerifyComplete => "verify_complete",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpecVerifyGraphMode {
    NotApplicable,
    Direct,
    Warmup,
    Capture,
    Replay,
}

impl SpecVerifyGraphMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::Direct => "direct",
            Self::Warmup => "warmup",
            Self::Capture => "capture",
            Self::Replay => "replay",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VerifyGraphPolicy {
    Default,
    Disabled,
}

/// Conservative admission result for speculative verify rollback.
///
/// Qwen35 DFlash/MTP verify may write KV/logit state past the ultimately
/// accepted prefix. Single-session decode can tolerate this only when the next
/// AR/spec verify starts at the post-commit boundary and overwrites every slot
/// it will read. Cross-session verify batching is refused until a real
/// per-session scratch/commit protocol can prove KV, DeltaNet, logits, and next
/// token parity against AR replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecRollbackParityDecision {
    pub accepted: usize,
    pub committed_len: usize,
    pub next_position: usize,
    pub ar_replay_start: usize,
    pub allow_single_session: bool,
    pub allow_multi_request_verify_batch: bool,
    pub reason: &'static str,
}

pub fn spec_rollback_parity_decision(
    start_position: usize,
    accepted: usize,
    committed_len: usize,
    verify_len: usize,
    ar_replay_start: usize,
) -> SpecRollbackParityDecision {
    let next_position = start_position.saturating_add(committed_len.saturating_sub(1));
    if committed_len == 0 {
        return SpecRollbackParityDecision {
            accepted,
            committed_len,
            next_position,
            ar_replay_start,
            allow_single_session: false,
            allow_multi_request_verify_batch: false,
            reason: "empty_commit",
        };
    }
    if accepted + 1 > verify_len {
        return SpecRollbackParityDecision {
            accepted,
            committed_len,
            next_position,
            ar_replay_start,
            allow_single_session: false,
            allow_multi_request_verify_batch: false,
            reason: "accepted_prefix_exceeds_verify",
        };
    }
    if committed_len != accepted + 2 {
        return SpecRollbackParityDecision {
            accepted,
            committed_len,
            next_position,
            ar_replay_start,
            allow_single_session: false,
            allow_multi_request_verify_batch: false,
            reason: "commit_shape_mismatch",
        };
    }
    if ar_replay_start != next_position {
        return SpecRollbackParityDecision {
            accepted,
            committed_len,
            next_position,
            ar_replay_start,
            allow_single_session: false,
            allow_multi_request_verify_batch: false,
            reason: "ar_replay_does_not_start_at_commit_boundary",
        };
    }
    SpecRollbackParityDecision {
        accepted,
        committed_len,
        next_position,
        ar_replay_start,
        allow_single_session: true,
        allow_multi_request_verify_batch: false,
        reason: "multi_request_verify_batch_disabled_pending_rollback_parity",
    }
}

pub fn spec_rollback_parity_decision_for_step(
    start_position: usize,
    step: &SpecStepResult,
) -> SpecRollbackParityDecision {
    spec_rollback_parity_decision(
        start_position,
        step.accepted,
        step.committed.len(),
        step.drafted.len(),
        start_position + step.accepted + 1,
    )
}

/// Aggregated metrics for a sequence of speculative decode steps.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Total number of speculative cycles run.
    pub cycles: usize,
    /// Total number of tokens committed (sum of committed.len() across cycles).
    pub committed_tokens: usize,
    /// Total number of draft tokens accepted (sum of `accepted`).
    pub accepted_tokens: usize,
    /// Per-cycle acceptance count histogram, indexed by accepted count
    /// (0..=k). `acceptance_hist[i]` = number of cycles where exactly `i`
    /// draft tokens were accepted.
    pub acceptance_hist: Vec<usize>,
}

impl SpecStats {
    pub fn new(k: usize) -> Self {
        Self {
            cycles: 0,
            committed_tokens: 0,
            accepted_tokens: 0,
            acceptance_hist: vec![0; k + 1],
        }
    }

    pub fn record(&mut self, step: &SpecStepResult) {
        self.cycles += 1;
        self.committed_tokens += step.committed.len();
        self.accepted_tokens += step.accepted;
        if step.accepted < self.acceptance_hist.len() {
            self.acceptance_hist[step.accepted] += 1;
        }
    }

    /// Mean accepted draft tokens per cycle. This is τ from the Leviathan paper.
    pub fn tau(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.cycles as f32
        }
    }

    /// Mean committed tokens per cycle (tau + 1 on average, since each
    /// cycle always commits one bonus token).
    pub fn mean_committed(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.committed_tokens as f32 / self.cycles as f32
        }
    }
}
