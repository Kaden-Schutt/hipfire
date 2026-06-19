// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared `EvalResult` row builders and prompt references.
//!
//! `pass_row`/`skip_row`/`skip_row_with_metrics`/`row`/`row_for_model` are the
//! canonical constructors every battery/suite executor uses to emit results;
//! `PromptRef`/`prompt`/`combined_prompt_ref`/`structured_tools_prompt_ref`
//! capture the byte-identical prompt path + content hash that the provenance
//! contract requires. Extracted verbatim from the former
//! `hipfire-eval/src/lib.rs` monolith (no behavior change).

use crate::*;

pub(crate) fn pass_row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    mut metrics: BTreeMap<String, Value>,
) -> EvalResult {
    metrics.insert("implemented".to_string(), json!(true));
    row(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Pass,
        None,
        metrics,
        config,
        ctx,
        prompt,
        0,
    )
}

pub(crate) fn skip_row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    reason: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
) -> EvalResult {
    row(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Skip,
        Some(reason.to_string()),
        BTreeMap::new(),
        config,
        ctx,
        prompt,
        0,
    )
}

pub(crate) fn skip_row_with_metrics(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    reason: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    metrics: BTreeMap<String, Value>,
) -> EvalResult {
    row(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Skip,
        Some(reason.to_string()),
        metrics,
        config,
        ctx,
        prompt,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    status: EvalStatus,
    reason: Option<String>,
    metrics: BTreeMap<String, Value>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    elapsed_ms: u128,
) -> EvalResult {
    row_for_model(
        battery,
        suite,
        case_id,
        dataset_item_id,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt,
        elapsed_ms,
        config.model.clone(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn row_for_model(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    status: EvalStatus,
    reason: Option<String>,
    metrics: BTreeMap<String, Value>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    elapsed_ms: u128,
    model: String,
) -> EvalResult {
    EvalResult {
        schema: 2,
        battery,
        suite,
        case_id: case_id.to_string(),
        dataset_item_id,
        dataset_source: metric_string(&metrics, "dataset_source"),
        dataset_repo_id: metric_string(&metrics, "dataset_repo_id"),
        dataset_revision: metric_string(&metrics, "dataset_revision"),
        dataset_digest: metric_string(&metrics, "dataset_digest"),
        dataset_license: metric_string(&metrics, "dataset_license"),
        dataset_cache_path: metric_string(&metrics, "dataset_cache_path"),
        status,
        reason,
        metrics,
        prompt_hash: prompt.as_ref().map(|p| p.hash.clone()),
        prompt_path: prompt.map(|p| p.path),
        model_hash: model_hash(&model),
        model,
        draft: config.draft.clone(),
        baseline: config.baseline.clone(),
        reference: config.reference.clone(),
        draft_hash: config.draft.as_deref().and_then(model_hash),
        baseline_hash: config.baseline.as_deref().and_then(model_hash),
        reference_hash: config.reference.as_deref().and_then(model_hash),
        hipfire_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: ctx.commit_sha.clone(),
        commit_sha: ctx.commit_sha.clone(),
        git_branch: ctx.git_branch.clone(),
        git_describe: ctx.git_describe.clone(),
        git_dirty: ctx.git_dirty,
        binary_hash: ctx.binary_hash.clone(),
        arch: ctx.arch.clone(),
        rocm: ctx.rocm.clone(),
        host_profile_hash: ctx.host_profile.host_profile_hash.clone(),
        hardware_bucket: ctx.host_profile.hardware_bucket.clone(),
        kv_mode: config.kv_mode.clone(),
        started_utc: utc_now(),
        elapsed_ms,
    }
}

#[derive(Clone)]
pub(crate) struct PromptRef {
    path: String,
    pub(crate) hash: String,
}

impl PromptRef {
    pub(crate) fn from_content(path: String, content: &[u8]) -> Self {
        Self {
            path,
            hash: stable_hash_bytes(content),
        }
    }
}

pub(crate) fn prompt(path: &str) -> Option<PromptRef> {
    let p = Path::new(path);
    let owned;
    let p = if p.exists() {
        p
    } else {
        owned = repo_root()?.join(path);
        if !owned.exists() {
            return None;
        }
        &owned
    };
    Some(PromptRef {
        path: path.to_string(),
        hash: file_hash(p).unwrap_or_else(|| stable_hash_file_fallback(p)),
    })
}

pub(crate) fn combined_prompt_ref(system_path: &str, prompt_path: &str) -> Option<PromptRef> {
    let system = fs::read(resolve_repo_path(system_path)?).ok()?;
    let prompt = fs::read(resolve_repo_path(prompt_path)?).ok()?;
    let mut content = Vec::with_capacity(system.len() + prompt.len() + 5);
    content.extend_from_slice(&system);
    content.extend_from_slice(b"\n---\n");
    content.extend_from_slice(&prompt);
    Some(PromptRef::from_content(
        format!("{system_path}+{prompt_path}"),
        &content,
    ))
}

pub(crate) fn structured_tools_prompt_ref(
    system_path: &str,
    prompt_path: &str,
    tools: &Value,
) -> Option<PromptRef> {
    let system = fs::read(resolve_repo_path(system_path)?).ok()?;
    let prompt = fs::read(resolve_repo_path(prompt_path)?).ok()?;
    let tools = serde_json::to_vec(tools).ok()?;
    let mut content = Vec::with_capacity(system.len() + prompt.len() + tools.len() + 16);
    content.extend_from_slice(&system);
    content.extend_from_slice(b"\n---prompt---\n");
    content.extend_from_slice(&prompt);
    content.extend_from_slice(b"\n---tools---\n");
    content.extend_from_slice(&tools);
    Some(PromptRef::from_content(
        format!("{system_path}+{prompt_path}+structured_tools"),
        &content,
    ))
}
