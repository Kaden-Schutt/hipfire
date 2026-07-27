// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Append-only optimization-campaign accounting.
//!
//! GPU rounds are reconstructed from durable JSONL history.  Infrastructure
//! failures do not consume a round, while an identical candidate identity can
//! never consume two completed rounds.  A submitted configuration SHA joins
//! the exact code-object SHA as a paired identity; without one, the code-object
//! SHA remains the identity for backwards compatibility.

use crate::{ArchProfile, RESOURCE_CONTRACT_SCHEMA_VERSION, ResourceAssessment};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub const CAMPAIGN_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MAX_COMPLETED_GPU_BATTERIES_PER_TARGET: u8 = 3;

#[derive(Debug, Error)]
pub enum CampaignError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("campaign ledger line {line} is not valid JSON: {message}")]
    InvalidLedgerLine { line: usize, message: String },
    #[error("campaign ledger is empty")]
    EmptyLedger,
    #[error("campaign ledger must begin with exactly one campaign_started record")]
    InvalidStartRecord,
    #[error("campaign ledger already exists: {0}")]
    AlreadyExists(PathBuf),
    #[error("invalid campaign policy: {0}")]
    InvalidPolicy(String),
    #[error("invalid candidate evidence: {0}")]
    InvalidCandidate(String),
    #[error("candidate incumbent {actual} is stale; current campaign incumbent is {expected}")]
    StaleIncumbent { expected: String, actual: String },
    #[error("target {target:?} exhausted its {maximum} completed GPU batteries")]
    BatteryBudgetExhausted { target: String, maximum: u8 },
    #[error(
        "candidate {object_sha256} for target {target:?} exhausted its one infrastructure retry"
    )]
    InfrastructureRetriesExhausted {
        target: String,
        object_sha256: String,
    },
    #[error("candidate {object_sha256} for target {target:?} is not promotion eligible")]
    NotPromotionEligible {
        target: String,
        object_sha256: String,
    },
    #[error("candidate {object_sha256} for target {target:?} has already been promoted")]
    AlreadyPromoted {
        target: String,
        object_sha256: String,
    },
}

pub type CampaignResult<T> = std::result::Result<T, CampaignError>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignPolicy {
    pub max_completed_gpu_batteries_per_target: u8,
    /// Retries after the initial infrastructure-failed attempt.
    pub infrastructure_retries_per_object: u8,
    pub paired_turns: u8,
    pub minimum_median_gain_percent: f64,
    pub minimum_paired_wins: u8,
}

impl Default for CampaignPolicy {
    fn default() -> Self {
        Self {
            max_completed_gpu_batteries_per_target: DEFAULT_MAX_COMPLETED_GPU_BATTERIES_PER_TARGET,
            infrastructure_retries_per_object: 1,
            paired_turns: 8,
            minimum_median_gain_percent: 0.5,
            minimum_paired_wins: 5,
        }
    }
}

impl CampaignPolicy {
    pub fn validate(&self) -> CampaignResult<()> {
        if self.max_completed_gpu_batteries_per_target == 0
            || self.max_completed_gpu_batteries_per_target
                > DEFAULT_MAX_COMPLETED_GPU_BATTERIES_PER_TARGET
        {
            return Err(CampaignError::InvalidPolicy(
                "completed GPU battery limit must be between one and three".to_owned(),
            ));
        }
        if self.infrastructure_retries_per_object > 1 {
            return Err(CampaignError::InvalidPolicy(
                "at most one infrastructure retry is permitted".to_owned(),
            ));
        }
        if self.paired_turns != 8
            || self.minimum_paired_wins < 5
            || self.minimum_paired_wins > self.paired_turns
        {
            return Err(CampaignError::InvalidPolicy(
                "campaign batteries require eight paired turns and at least five wins".to_owned(),
            ));
        }
        if !self.minimum_median_gain_percent.is_finite() || self.minimum_median_gain_percent < 0.5 {
            return Err(CampaignError::InvalidPolicy(
                "minimum median gain must be finite and at least 0.5 percent".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn promotes(&self, median_gain_percent: f64, paired_wins: u8) -> bool {
        median_gain_percent >= self.minimum_median_gain_percent
            && paired_wins >= self.minimum_paired_wins
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignStarted {
    pub schema_version: u32,
    pub campaign_id: String,
    pub profile: ArchProfile,
    pub baseline_incumbent_sha256: String,
    pub policy: CampaignPolicy,
    pub recorded_unix_seconds: u64,
}

impl CampaignStarted {
    pub fn new(
        campaign_id: impl Into<String>,
        profile: ArchProfile,
        baseline_incumbent_sha256: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: CAMPAIGN_SCHEMA_VERSION,
            campaign_id: campaign_id.into(),
            profile,
            baseline_incumbent_sha256: baseline_incumbent_sha256.into(),
            policy: CampaignPolicy::default(),
            recorded_unix_seconds: now_unix_seconds(),
        }
    }

    pub fn policy(mut self, policy: CampaignPolicy) -> Self {
        self.policy = policy;
        self
    }

    fn validate(&self) -> CampaignResult<()> {
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION {
            return Err(CampaignError::InvalidPolicy(format!(
                "campaign schema {} is unsupported; expected {CAMPAIGN_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.campaign_id.trim().is_empty() || self.baseline_incumbent_sha256.trim().is_empty() {
            return Err(CampaignError::InvalidPolicy(
                "campaign id and baseline incumbent SHA must be non-empty".to_owned(),
            ));
        }
        self.policy.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateVerdict {
    StaticRejected {
        reason: String,
    },
    CorrectnessRejected {
        reason: String,
    },
    InfrastructureFailure {
        reason: String,
    },
    BatteryCompleted {
        median_gain_percent: f64,
        paired_wins: u8,
        paired_turns: u8,
    },
}

impl CandidateVerdict {
    fn is_terminal(&self) -> bool {
        !matches!(self, Self::InfrastructureFailure { .. })
    }

    fn reaches_gpu(&self) -> bool {
        matches!(
            self,
            Self::InfrastructureFailure { .. } | Self::BatteryCompleted { .. }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateSubmission {
    pub target: String,
    pub source_sha256: String,
    pub object_sha256: String,
    /// Optional identity for launch/replay configuration that can change while
    /// the compiled object remains byte-identical (for example tile geometry).
    /// When present, campaign de-duplication and retry accounting use its pair
    /// with `object_sha256` instead of the object SHA alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration_sha256: Option<String>,
    /// Whole-product executable or bundle that contains this object and is
    /// used by the correctness/timing harnesses.
    pub product_sha256: String,
    pub incumbent_sha256: String,
    pub resource_assessment: Option<ResourceAssessment>,
    pub correctness_artifact: Option<String>,
    pub timing_artifact: Option<String>,
    pub verdict: CandidateVerdict,
}

impl CandidateSubmission {
    pub fn new(
        target: impl Into<String>,
        source_sha256: impl Into<String>,
        object_sha256: impl Into<String>,
        incumbent_sha256: impl Into<String>,
        verdict: CandidateVerdict,
    ) -> Self {
        Self {
            target: target.into(),
            source_sha256: source_sha256.into(),
            object_sha256: object_sha256.into(),
            configuration_sha256: None,
            product_sha256: String::new(),
            incumbent_sha256: incumbent_sha256.into(),
            resource_assessment: None,
            correctness_artifact: None,
            timing_artifact: None,
            verdict,
        }
    }

    pub fn product_sha256(mut self, product_sha256: impl Into<String>) -> Self {
        self.product_sha256 = product_sha256.into();
        self
    }

    pub fn configuration_sha256(mut self, configuration_sha256: impl Into<String>) -> Self {
        self.configuration_sha256 = Some(configuration_sha256.into());
        self
    }

    pub fn resource_assessment(mut self, assessment: ResourceAssessment) -> Self {
        self.resource_assessment = Some(assessment);
        self
    }

    pub fn correctness_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.correctness_artifact = Some(artifact.into());
        self
    }

    pub fn timing_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.timing_artifact = Some(artifact.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub target: String,
    /// The next completed GPU round for infrastructure failures, or the actual
    /// completed round for a valid battery.  Static/correctness rejects have no
    /// GPU round.
    pub gpu_round: Option<u8>,
    /// Initial attempt is 1; the sole infrastructure retry is 2.
    pub attempt: u8,
    pub source_sha256: String,
    pub object_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration_sha256: Option<String>,
    pub product_sha256: String,
    pub incumbent_sha256: String,
    pub resource_assessment: Option<ResourceAssessment>,
    pub correctness_artifact: Option<String>,
    pub timing_artifact: Option<String>,
    pub verdict: CandidateVerdict,
    pub recorded_unix_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotionRecord {
    pub target: String,
    pub gpu_round: u8,
    pub source_sha256: String,
    pub object_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration_sha256: Option<String>,
    pub product_sha256: String,
    pub previous_incumbent_sha256: String,
    pub median_gain_percent: f64,
    pub paired_wins: u8,
    pub paired_turns: u8,
    pub timing_artifact: String,
    pub recorded_unix_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record_type", content = "record", rename_all = "snake_case")]
pub enum CampaignEvent {
    CampaignStarted(CampaignStarted),
    Candidate(CandidateRecord),
    Promotion(PromotionRecord),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "disposition", content = "record", rename_all = "snake_case")]
pub enum RecordDisposition {
    Recorded(CandidateRecord),
    DuplicateSkipped(CandidateRecord),
}

#[derive(Clone, Debug)]
pub struct CampaignLedger {
    path: PathBuf,
}

impl CampaignLedger {
    pub fn create(path: impl Into<PathBuf>, started: CampaignStarted) -> CampaignResult<Self> {
        started.validate()?;
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(CampaignError::AlreadyExists(path));
            }
            Err(error) => return Err(error.into()),
        };
        write_event(&mut file, &CampaignEvent::CampaignStarted(started))?;
        file.sync_data()?;
        Ok(Self { path })
    }

    pub fn open(path: impl Into<PathBuf>) -> CampaignResult<Self> {
        let ledger = Self { path: path.into() };
        let events = ledger.events()?;
        validate_start(&events)?;
        Ok(ledger)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn events(&self) -> CampaignResult<Vec<CampaignEvent>> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let event =
                serde_json::from_str(&line).map_err(|error| CampaignError::InvalidLedgerLine {
                    line: index + 1,
                    message: error.to_string(),
                })?;
            events.push(event);
        }
        validate_start(&events)?;
        Ok(events)
    }

    pub fn started(&self) -> CampaignResult<CampaignStarted> {
        let events = self.events()?;
        match &events[0] {
            CampaignEvent::CampaignStarted(started) => Ok(started.clone()),
            _ => Err(CampaignError::InvalidStartRecord),
        }
    }

    pub fn current_incumbent_sha256(&self) -> CampaignResult<String> {
        let events = self.events()?;
        Ok(current_incumbent(&events))
    }

    pub fn completed_batteries(&self, target: &str) -> CampaignResult<u8> {
        let events = self.events()?;
        Ok(completed_batteries(&events, target))
    }

    pub fn record_candidate(
        &self,
        submission: CandidateSubmission,
    ) -> CampaignResult<RecordDisposition> {
        validate_submission(&submission)?;
        let events = self.events()?;
        let started = started_from(&events);
        let matching = events
            .iter()
            .filter_map(|event| match event {
                CampaignEvent::Candidate(record)
                    if record.target == submission.target
                        && candidate_identity(record)
                            == submission_candidate_identity(&submission) =>
                {
                    Some(record)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if let Some(terminal) = matching.iter().find(|record| record.verdict.is_terminal()) {
            return Ok(RecordDisposition::DuplicateSkipped((*terminal).clone()));
        }

        let current_incumbent = current_incumbent(&events);
        if submission.incumbent_sha256 != current_incumbent {
            return Err(CampaignError::StaleIncumbent {
                expected: current_incumbent,
                actual: submission.incumbent_sha256,
            });
        }

        let infrastructure_failures = matching.len() as u8;
        if infrastructure_failures > started.policy.infrastructure_retries_per_object {
            return Err(CampaignError::InfrastructureRetriesExhausted {
                target: submission.target,
                object_sha256: submission.object_sha256,
            });
        }

        let completed = completed_batteries(&events, &submission.target);
        if submission.verdict.reaches_gpu()
            && completed >= started.policy.max_completed_gpu_batteries_per_target
        {
            return Err(CampaignError::BatteryBudgetExhausted {
                target: submission.target,
                maximum: started.policy.max_completed_gpu_batteries_per_target,
            });
        }

        validate_battery_evidence(&submission, started)?;
        let gpu_round = submission
            .verdict
            .reaches_gpu()
            .then_some(completed.saturating_add(1));
        let record = CandidateRecord {
            target: submission.target,
            gpu_round,
            attempt: infrastructure_failures.saturating_add(1),
            source_sha256: submission.source_sha256,
            object_sha256: submission.object_sha256,
            configuration_sha256: submission.configuration_sha256,
            product_sha256: submission.product_sha256,
            incumbent_sha256: submission.incumbent_sha256,
            resource_assessment: submission.resource_assessment,
            correctness_artifact: submission.correctness_artifact,
            timing_artifact: submission.timing_artifact,
            verdict: submission.verdict,
            recorded_unix_seconds: now_unix_seconds(),
        };
        self.append(&CampaignEvent::Candidate(record.clone()))?;
        Ok(RecordDisposition::Recorded(record))
    }

    pub fn promote(&self, target: &str, object_sha256: &str) -> CampaignResult<PromotionRecord> {
        self.promote_candidate(target, object_sha256, None)
    }

    /// Promote one exact candidate identity. Configured candidates must provide
    /// their configuration SHA; an object-only promotion deliberately matches
    /// only legacy/object-identified candidates so equal objects with distinct
    /// replay geometry cannot be confused.
    pub fn promote_candidate(
        &self,
        target: &str,
        object_sha256: &str,
        configuration_sha256: Option<&str>,
    ) -> CampaignResult<PromotionRecord> {
        let events = self.events()?;
        let started = started_from(&events);
        if events.iter().any(|event| {
            matches!(event, CampaignEvent::Promotion(record)
                if promotion_matches(record, target, object_sha256, configuration_sha256))
        }) {
            return Err(CampaignError::AlreadyPromoted {
                target: target.to_owned(),
                object_sha256: object_sha256.to_owned(),
            });
        }
        let candidate = events.iter().rev().find_map(|event| match event {
            CampaignEvent::Candidate(record)
                if candidate_matches_promotion(
                    record,
                    target,
                    object_sha256,
                    configuration_sha256,
                ) && matches!(record.verdict, CandidateVerdict::BatteryCompleted { .. }) =>
            {
                Some(record)
            }
            _ => None,
        });
        let Some(candidate) = candidate else {
            return Err(CampaignError::NotPromotionEligible {
                target: target.to_owned(),
                object_sha256: object_sha256.to_owned(),
            });
        };
        let current_incumbent = current_incumbent(&events);
        if candidate.incumbent_sha256 != current_incumbent {
            return Err(CampaignError::StaleIncumbent {
                expected: current_incumbent,
                actual: candidate.incumbent_sha256.clone(),
            });
        }
        let CandidateVerdict::BatteryCompleted {
            median_gain_percent,
            paired_wins,
            paired_turns,
        } = candidate.verdict
        else {
            unreachable!("candidate lookup requires a completed battery")
        };
        if !started.policy.promotes(median_gain_percent, paired_wins) {
            return Err(CampaignError::NotPromotionEligible {
                target: target.to_owned(),
                object_sha256: object_sha256.to_owned(),
            });
        }
        let timing_artifact = candidate
            .timing_artifact
            .clone()
            .expect("completed batteries require a timing artifact");
        let record = PromotionRecord {
            target: target.to_owned(),
            gpu_round: candidate
                .gpu_round
                .expect("completed batteries have a GPU round"),
            source_sha256: candidate.source_sha256.clone(),
            object_sha256: candidate.object_sha256.clone(),
            configuration_sha256: candidate.configuration_sha256.clone(),
            product_sha256: candidate.product_sha256.clone(),
            previous_incumbent_sha256: candidate.incumbent_sha256.clone(),
            median_gain_percent,
            paired_wins,
            paired_turns,
            timing_artifact,
            recorded_unix_seconds: now_unix_seconds(),
        };
        self.append(&CampaignEvent::Promotion(record.clone()))?;
        Ok(record)
    }

    fn append(&self, event: &CampaignEvent) -> CampaignResult<()> {
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        write_event(&mut file, event)?;
        file.sync_data()?;
        Ok(())
    }
}

fn validate_start(events: &[CampaignEvent]) -> CampaignResult<()> {
    let Some(CampaignEvent::CampaignStarted(started)) = events.first() else {
        return if events.is_empty() {
            Err(CampaignError::EmptyLedger)
        } else {
            Err(CampaignError::InvalidStartRecord)
        };
    };
    started.validate()?;
    if events
        .iter()
        .skip(1)
        .any(|event| matches!(event, CampaignEvent::CampaignStarted(_)))
    {
        return Err(CampaignError::InvalidStartRecord);
    }
    Ok(())
}

fn started_from(events: &[CampaignEvent]) -> &CampaignStarted {
    match &events[0] {
        CampaignEvent::CampaignStarted(started) => started,
        _ => unreachable!("event loading validates the start record"),
    }
}

fn current_incumbent(events: &[CampaignEvent]) -> String {
    let mut incumbent = started_from(events).baseline_incumbent_sha256.clone();
    for event in events {
        if let CampaignEvent::Promotion(record) = event {
            incumbent.clone_from(&record.product_sha256);
        }
    }
    incumbent
}

fn completed_batteries(events: &[CampaignEvent], target: &str) -> u8 {
    events
        .iter()
        .filter(|event| {
            matches!(event, CampaignEvent::Candidate(record)
                if record.target == target
                    && matches!(record.verdict, CandidateVerdict::BatteryCompleted { .. }))
        })
        .count()
        .try_into()
        .unwrap_or(u8::MAX)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateIdentity<'a> {
    Configured {
        object_sha256: &'a str,
        configuration_sha256: &'a str,
    },
    Object(&'a str),
}

fn candidate_identity(record: &CandidateRecord) -> CandidateIdentity<'_> {
    match record.configuration_sha256.as_deref() {
        Some(configuration_sha256) => CandidateIdentity::Configured {
            object_sha256: &record.object_sha256,
            configuration_sha256,
        },
        None => CandidateIdentity::Object(&record.object_sha256),
    }
}

fn submission_candidate_identity(submission: &CandidateSubmission) -> CandidateIdentity<'_> {
    match submission.configuration_sha256.as_deref() {
        Some(configuration_sha256) => CandidateIdentity::Configured {
            object_sha256: &submission.object_sha256,
            configuration_sha256,
        },
        None => CandidateIdentity::Object(&submission.object_sha256),
    }
}

fn candidate_matches_promotion(
    record: &CandidateRecord,
    target: &str,
    object_sha256: &str,
    configuration_sha256: Option<&str>,
) -> bool {
    record.target == target
        && record.object_sha256 == object_sha256
        && record.configuration_sha256.as_deref() == configuration_sha256
}

fn promotion_matches(
    record: &PromotionRecord,
    target: &str,
    object_sha256: &str,
    configuration_sha256: Option<&str>,
) -> bool {
    record.target == target
        && record.object_sha256 == object_sha256
        && record.configuration_sha256.as_deref() == configuration_sha256
}

fn validate_submission(submission: &CandidateSubmission) -> CampaignResult<()> {
    for (label, value) in [
        ("target", submission.target.as_str()),
        ("source SHA-256", submission.source_sha256.as_str()),
        ("object SHA-256", submission.object_sha256.as_str()),
        ("product SHA-256", submission.product_sha256.as_str()),
        ("incumbent SHA-256", submission.incumbent_sha256.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(CampaignError::InvalidCandidate(format!(
                "{label} must be non-empty"
            )));
        }
    }
    if submission
        .configuration_sha256
        .as_deref()
        .is_some_and(|sha| sha.trim().is_empty())
    {
        return Err(CampaignError::InvalidCandidate(
            "configuration SHA-256 must be non-empty when provided".to_owned(),
        ));
    }
    Ok(())
}

fn validate_battery_evidence(
    submission: &CandidateSubmission,
    started: &CampaignStarted,
) -> CampaignResult<()> {
    let policy = &started.policy;
    let accepted_resources = submission
        .resource_assessment
        .as_ref()
        .is_some_and(|assessment| {
            assessment.schema_version == RESOURCE_CONTRACT_SCHEMA_VERSION
                && assessment.profile == started.profile
                && matches!(assessment.required_wavefront_size, 32 | 64)
                && assessment.accepted
                && assessment.rejections.is_empty()
        });
    match &submission.verdict {
        CandidateVerdict::BatteryCompleted {
            median_gain_percent,
            paired_wins,
            paired_turns,
        } => {
            if !median_gain_percent.is_finite()
                || *paired_turns != policy.paired_turns
                || *paired_wins > *paired_turns
            {
                return Err(CampaignError::InvalidCandidate(format!(
                    "completed batteries require finite timing and exactly {} paired turns",
                    policy.paired_turns
                )));
            }
            if !accepted_resources
                || submission
                    .correctness_artifact
                    .as_deref()
                    .is_none_or(str::is_empty)
                || submission
                    .timing_artifact
                    .as_deref()
                    .is_none_or(str::is_empty)
            {
                return Err(CampaignError::InvalidCandidate(
                    "completed batteries require an accepted resource assessment plus correctness and timing artifacts"
                        .to_owned(),
                ));
            }
        }
        CandidateVerdict::InfrastructureFailure { .. } => {
            if !accepted_resources
                || submission
                    .correctness_artifact
                    .as_deref()
                    .is_none_or(str::is_empty)
            {
                return Err(CampaignError::InvalidCandidate(
                    "infrastructure failures require successful resource and correctness pre-gates"
                        .to_owned(),
                ));
            }
        }
        CandidateVerdict::StaticRejected { .. } | CandidateVerdict::CorrectnessRejected { .. } => {}
    }
    Ok(())
}

fn write_event(file: &mut File, event: &CampaignEvent) -> CampaignResult<()> {
    let mut encoded = serde_json::to_vec(event).map_err(|error| {
        CampaignError::InvalidCandidate(format!("event serialization failed: {error}"))
    })?;
    encoded.push(b'\n');
    file.write_all(&encoded)?;
    Ok(())
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_LEDGER: AtomicU64 = AtomicU64::new(0);

    fn path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "radiowave-campaign-{}-{}.jsonl",
            std::process::id(),
            NEXT_LEDGER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn accepted_resources() -> ResourceAssessment {
        ResourceAssessment {
            schema_version: RESOURCE_CONTRACT_SCHEMA_VERSION,
            profile: ArchProfile::Gfx1151,
            kernel: "kernel".to_owned(),
            required_wavefront_size: 32,
            accepted: true,
            incumbent_waves_per_simd: Some(16),
            candidate_waves_per_simd: Some(16),
            rejections: Vec::new(),
        }
    }

    fn completed(
        target: &str,
        object: &str,
        incumbent: &str,
        gain: f64,
        wins: u8,
    ) -> CandidateSubmission {
        CandidateSubmission::new(
            target,
            format!("source-{object}"),
            object,
            incumbent,
            CandidateVerdict::BatteryCompleted {
                median_gain_percent: gain,
                paired_wins: wins,
                paired_turns: 8,
            },
        )
        .product_sha256(format!("product-{object}"))
        .resource_assessment(accepted_resources())
        .correctness_artifact(format!("correct-{object}.json"))
        .timing_artifact(format!("timing-{object}.json"))
    }

    fn ledger() -> (PathBuf, CampaignLedger) {
        let path = path();
        let ledger = CampaignLedger::create(
            &path,
            CampaignStarted::new("gfx1151-test", ArchProfile::Gfx1151, "baseline"),
        )
        .unwrap();
        (path, ledger)
    }

    #[test]
    fn campaign_policy_cannot_weaken_the_agreed_budget_or_win_bar() {
        for policy in [
            CampaignPolicy {
                max_completed_gpu_batteries_per_target: 4,
                ..CampaignPolicy::default()
            },
            CampaignPolicy {
                infrastructure_retries_per_object: 2,
                ..CampaignPolicy::default()
            },
            CampaignPolicy {
                paired_turns: 4,
                ..CampaignPolicy::default()
            },
            CampaignPolicy {
                minimum_median_gain_percent: 0.49,
                ..CampaignPolicy::default()
            },
            CampaignPolicy {
                minimum_paired_wins: 4,
                ..CampaignPolicy::default()
            },
        ] {
            assert!(policy.validate().is_err());
        }
    }

    #[test]
    fn completed_rounds_are_sha_deduplicated_and_hard_capped() {
        let (path, ledger) = ledger();
        for (index, object) in ["a", "b", "c"].into_iter().enumerate() {
            let result = ledger
                .record_candidate(completed("tile", object, "baseline", 0.1, 4))
                .unwrap();
            let RecordDisposition::Recorded(record) = result else {
                panic!("new object was unexpectedly deduplicated")
            };
            assert_eq!(record.gpu_round, Some(index as u8 + 1));
        }
        let duplicate = ledger
            .record_candidate(completed("tile", "b", "baseline", 0.1, 4))
            .unwrap();
        assert!(matches!(duplicate, RecordDisposition::DuplicateSkipped(_)));
        assert!(matches!(
            ledger.record_candidate(completed("tile", "d", "baseline", 0.1, 4)),
            Err(CampaignError::BatteryBudgetExhausted { .. })
        ));
        assert_eq!(ledger.completed_batteries("tile").unwrap(), 3);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn configuration_sha_distinguishes_geometry_for_one_object() {
        let (path, ledger) = ledger();
        let tile32 = completed("flash-tile", "shared-object", "baseline", 0.1, 4)
            .configuration_sha256("config-tile32");
        let first = ledger.record_candidate(tile32.clone()).unwrap();
        let RecordDisposition::Recorded(first) = first else {
            panic!("first geometry was unexpectedly deduplicated")
        };
        assert_eq!(first.gpu_round, Some(1));
        assert_eq!(first.configuration_sha256.as_deref(), Some("config-tile32"));

        let tile64 = completed("flash-tile", "shared-object", "baseline", 0.1, 4)
            .configuration_sha256("config-tile64");
        let second = ledger.record_candidate(tile64).unwrap();
        let RecordDisposition::Recorded(second) = second else {
            panic!("distinct geometry was unexpectedly deduplicated")
        };
        assert_eq!(second.gpu_round, Some(2));
        assert_eq!(
            second.configuration_sha256.as_deref(),
            Some("config-tile64")
        );

        let duplicate = ledger.record_candidate(tile32).unwrap();
        assert!(matches!(
            duplicate,
            RecordDisposition::DuplicateSkipped(record)
                if record.configuration_sha256.as_deref() == Some("config-tile32")
        ));
        assert_eq!(ledger.completed_batteries("flash-tile").unwrap(), 2);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn recompiled_object_with_same_configuration_consumes_a_distinct_round() {
        let (path, ledger) = ledger();
        let object_a = completed("flash-tile", "object-a", "baseline", 0.1, 4)
            .configuration_sha256("config-tile32");
        let first = ledger.record_candidate(object_a.clone()).unwrap();
        let RecordDisposition::Recorded(first) = first else {
            panic!("first object was unexpectedly deduplicated")
        };
        assert_eq!(first.gpu_round, Some(1));

        let object_b = completed("flash-tile", "object-b", "baseline", 0.1, 4)
            .configuration_sha256("config-tile32");
        let second = ledger.record_candidate(object_b).unwrap();
        let RecordDisposition::Recorded(second) = second else {
            panic!("recompiled object was unexpectedly deduplicated")
        };
        assert_eq!(second.gpu_round, Some(2));

        assert!(matches!(
            ledger.record_candidate(object_a).unwrap(),
            RecordDisposition::DuplicateSkipped(_)
        ));
        assert_eq!(ledger.completed_batteries("flash-tile").unwrap(), 2);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn configured_promotion_requires_the_exact_object_and_configuration() {
        let (path, ledger) = ledger();
        ledger
            .record_candidate(
                completed("flash-tile", "shared-object", "baseline", 0.5, 5)
                    .configuration_sha256("config-tile32"),
            )
            .unwrap();

        assert!(matches!(
            ledger.promote("flash-tile", "shared-object"),
            Err(CampaignError::NotPromotionEligible { .. })
        ));
        assert!(matches!(
            ledger.promote_candidate("flash-tile", "different-object", Some("config-tile32")),
            Err(CampaignError::NotPromotionEligible { .. })
        ));
        let promotion = ledger
            .promote_candidate("flash-tile", "shared-object", Some("config-tile32"))
            .unwrap();
        assert_eq!(promotion.object_sha256, "shared-object");
        assert_eq!(
            promotion.configuration_sha256.as_deref(),
            Some("config-tile32")
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn configuration_identity_does_not_replace_source_or_object_provenance() {
        let (path, ledger) = ledger();
        for missing in ["source", "object"] {
            let mut submission = completed("flash-tile", "object", "baseline", 0.1, 4)
                .configuration_sha256("config-tile32");
            match missing {
                "source" => submission.source_sha256.clear(),
                "object" => submission.object_sha256.clear(),
                _ => unreachable!(),
            }
            assert!(matches!(
                ledger.record_candidate(submission),
                Err(CampaignError::InvalidCandidate(_))
            ));
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn infrastructure_failure_gets_one_free_retry_without_consuming_a_round() {
        let (path, ledger) = ledger();
        let failed = CandidateSubmission::new(
            "fusion",
            "source-a",
            "object-a",
            "baseline",
            CandidateVerdict::InfrastructureFailure {
                reason: "daemon disconnected".to_owned(),
            },
        )
        .product_sha256("product-a")
        .resource_assessment(accepted_resources())
        .correctness_artifact("correct-a.json");
        let first = ledger.record_candidate(failed.clone()).unwrap();
        let RecordDisposition::Recorded(first) = first else {
            panic!("first infrastructure attempt must be recorded")
        };
        assert_eq!(first.gpu_round, Some(1));
        assert_eq!(ledger.completed_batteries("fusion").unwrap(), 0);

        let second = ledger
            .record_candidate(completed("fusion", "object-a", "baseline", 0.6, 5))
            .unwrap();
        let RecordDisposition::Recorded(second) = second else {
            panic!("infrastructure retry must not be SHA-deduplicated")
        };
        assert_eq!(second.gpu_round, Some(1));
        assert_eq!(second.attempt, 2);

        let failed_b = CandidateSubmission::new(
            "other",
            "source-b",
            "object-b",
            "baseline",
            CandidateVerdict::InfrastructureFailure {
                reason: "invalid artifact".to_owned(),
            },
        )
        .product_sha256("product-b")
        .resource_assessment(accepted_resources())
        .correctness_artifact("correct-b.json");
        ledger.record_candidate(failed_b.clone()).unwrap();
        ledger.record_candidate(failed_b.clone()).unwrap();
        assert!(matches!(
            ledger.record_candidate(failed_b),
            Err(CampaignError::InfrastructureRetriesExhausted { .. })
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn promotion_requires_point_five_percent_and_five_of_eight() {
        let (path, ledger) = ledger();
        ledger
            .record_candidate(completed("conv", "weak", "baseline", 0.49, 8))
            .unwrap();
        assert!(matches!(
            ledger.promote("conv", "weak"),
            Err(CampaignError::NotPromotionEligible { .. })
        ));

        ledger
            .record_candidate(completed("conv", "winner", "baseline", 0.5, 5))
            .unwrap();
        let promotion = ledger.promote("conv", "winner").unwrap();
        assert_eq!(promotion.previous_incumbent_sha256, "baseline");
        assert_eq!(ledger.current_incumbent_sha256().unwrap(), "product-winner");
        assert!(matches!(
            ledger.promote("conv", "winner"),
            Err(CampaignError::AlreadyPromoted { .. })
        ));

        let reopened = CampaignLedger::open(&path).unwrap();
        assert_eq!(
            reopened.current_incumbent_sha256().unwrap(),
            "product-winner"
        );
        assert_eq!(reopened.events().unwrap().len(), 4);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn battery_rejects_internally_inconsistent_resource_evidence() {
        let (path, ledger) = ledger();
        let mut submission = completed("conv", "object", "baseline", 0.5, 5);
        submission.resource_assessment.as_mut().unwrap().accepted = true;
        submission
            .resource_assessment
            .as_mut()
            .unwrap()
            .rejections
            .push(crate::ResourceRejection::MissingCodeObjectIdentity);
        assert!(matches!(
            ledger.record_candidate(submission),
            Err(CampaignError::InvalidCandidate(_))
        ));
        fs::remove_file(path).unwrap();
    }
}
