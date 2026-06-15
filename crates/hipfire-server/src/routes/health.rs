use axum::{extract::State, response::Json};
use hipfire_scheduler::{
    parse_default_scheduler_priority, parse_server_prefill_policy_controls,
    scheduler_policy_for_priority, SchedulerPolicyEnv,
};
use serde_json::{json, Value};
use std::env;

use crate::state::SharedState;

pub async fn get_health(state: State<SharedState>) -> Json<Value> {
    let loaded = state.loaded_model_path.lock().await.clone();
    let scheduler_env = scheduler_env_from_process();
    Json(json!({
        "status": "ok",
        "model": loaded,
        "prefill_batch": prefill_batch_health(&scheduler_env),
        "decode_batch": decode_batch_health(&scheduler_env),
        "state_cache": state_cache_health(&scheduler_env),
        "batches": batch_health(),
    }))
}

fn scheduler_env_from_process() -> SchedulerPolicyEnv {
    SchedulerPolicyEnv::from_pairs(env::vars())
}

fn prefill_batch_enabled(env: &SchedulerPolicyEnv) -> bool {
    env.get("HIPFIRE_SERVER_PREFILL_BATCH")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "on" | "true"
            )
        })
        .unwrap_or(false)
}

fn prefill_batch_health(env: &SchedulerPolicyEnv) -> Value {
    if !prefill_batch_enabled(env) {
        return json!({ "enabled": false });
    }

    let priority = parse_default_scheduler_priority(env);
    let policy = scheduler_policy_for_priority(priority, env);
    let controls = parse_server_prefill_policy_controls(env);

    let mut payload = json!({
        "enabled": true,
        "queued": 0,
        "eligible": 0,
        "selected": 0,
        "skipped": 0,
        "total_batches": 0,
        "fused_batches": 0,
        "fallback_batches": 0,
        "batch_size_histogram": {},
        "cache_hits": 0,
        "cache_misses": 0,
        "metadata_cache_hits": 0,
        "runtime_cache_hits": 0,
        "queue_size": 0,
        "pending_requests": 0,
        "resident_runtime_sessions": 0,
        "resident_decode_sessions": 0,
        "resident_checkpoints": 0,
        "resident_checkpoint_max": controls.resident_checkpoint_max,
        "resident_state_cache": controls.resident_state_cache,
        "resident_state_limit": policy.resident_state_max,
        "spillable_batch_max": policy.spillable_batch_max,
        "spillable_sessions": 0,
        "state_cache_disk": controls.state_cache_disk,
        "state_cache_disk_min_priority": policy.disk_spill_min_priority,
        "disk_spill_allowed": policy.disk_spill_allowed,
        "state_cache_evictions_total": 0,
        "state_cache_recompute_required_total": 0,
        "generate_batch_prefill_capability": "unknown",
        "generate_batch_prefill_capability_reason": "rust_server_daemon_capability_not_probed",
        "queue_wait_reason": "disabled",
        "fallback_reason": "rust_server_scheduler_metadata_only",
        "runtime_dispatch_skipped_reason": "rust_server_prefill_queue_not_enabled",
        "selected_batch_size": 0,
        "last_prefill_tokens": 0,
        "last_prefill_ms": 0,
        "last_prefill_tok_s": 0,
    });
    payload["policy"] = json!({
        "priority": policy.priority,
        "priority_class": policy.priority_class.as_str(),
        "max_batch": policy.max_batch_size,
        "wait_ms": policy.coalesce_wait_ms,
        "target_pair_tokens": policy.target_pair_tokens,
        "max_processing_ms": policy.max_processing_ms,
    });
    payload
}

fn decode_batch_health(env: &SchedulerPolicyEnv) -> Value {
    if !prefill_batch_enabled(env) {
        return json!({ "enabled": false });
    }
    json!({
        "enabled": true,
        "eligible": 0,
        "selected": 0,
        "skipped": 0,
        "active_sessions": 0,
        "selected_batch_size": 0,
        "total_batches": 0,
        "serial_batches": 0,
        "fused_batches": 0,
        "last_skipped_reason": "rust_server_decode_scheduler_not_enabled",
        "fallback_reason": "rust_server_scheduler_metadata_only",
    })
}

fn state_cache_health(env: &SchedulerPolicyEnv) -> Value {
    if !prefill_batch_enabled(env) {
        return json!({ "enabled": false });
    }
    let controls = parse_server_prefill_policy_controls(env);
    json!({
        "enabled": controls.resident_state_cache || controls.state_cache_disk,
        "resident_enabled": controls.resident_state_cache,
        "resident_checkpoints": 0,
        "resident_checkpoint_max": controls.resident_checkpoint_max,
        "disk_enabled": controls.state_cache_disk,
        "daemon_prefix_hash": false,
        "daemon_prefix_hash_entries": 0,
        "semantic_boundary_checkpoints": false,
        "semantic_boundary_checkpoint_entries": 0,
        "prefix_hash_preflight_requests": 0,
        "prefix_hash_preflight_candidates": 0,
        "prefix_hash_preflight_matches": 0,
        "prefix_hash_preflight_boundary_matches": 0,
        "shared_prefix_fanout_groups": 0,
        "shared_prefix_fanout_followers": 0,
        "responses_previous_response_hits": 0,
        "responses_previous_response_misses": 0,
        "responses_stored_contexts": 0,
        "entries": 0,
        "bytes": 0,
        "metadata_hits": 0,
        "runtime_hits": 0,
        "evictions_total": 0,
        "recompute_required_total": 0,
    })
}

fn batch_health() -> Value {
    json!({
        "enabled": false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> SchedulerPolicyEnv {
        SchedulerPolicyEnv::from_pairs(pairs.iter().copied())
    }

    #[test]
    fn prefill_batch_health_is_disabled_by_default() {
        let payload = prefill_batch_health(&SchedulerPolicyEnv::empty());

        assert_eq!(payload, json!({ "enabled": false }));
    }

    #[test]
    fn prefill_batch_health_uses_shared_scheduler_policy() {
        let payload = prefill_batch_health(&env(&[
            ("HIPFIRE_SERVER_PREFILL_BATCH", "1"),
            ("HIPFIRE_SCHED_PRIORITY_DEFAULT", "128"),
            ("HIPFIRE_SCHED_PREFILL_BATCH_MAX", "8"),
            ("HIPFIRE_SCHED_RESIDENT_STATE_MAX", "3"),
            ("HIPFIRE_SCHED_SPILLABLE_BATCH_MAX", "12"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK", "1"),
            ("HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY", "64"),
            ("HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND", "11"),
        ]));

        assert_eq!(payload["enabled"], true);
        assert_eq!(payload["policy"]["priority"], 128);
        assert_eq!(payload["policy"]["priority_class"], "background");
        assert_eq!(payload["policy"]["max_batch"], 8);
        assert_eq!(payload["policy"]["wait_ms"], 11);
        assert_eq!(payload["resident_state_limit"], 3);
        assert_eq!(payload["spillable_batch_max"], 12);
        assert_eq!(payload["state_cache_disk"], true);
        assert_eq!(payload["disk_spill_allowed"], true);
        assert_eq!(
            payload["generate_batch_prefill_capability_reason"],
            "rust_server_daemon_capability_not_probed"
        );
    }

    #[test]
    fn health_state_cache_uses_shared_scheduler_controls() {
        let payload = state_cache_health(&env(&[
            ("HIPFIRE_SERVER_PREFILL_BATCH", "true"),
            ("HIPFIRE_SERVER_PREFILL_STATE_CACHE", "1"),
            ("HIPFIRE_STATE_CACHE_MAX_CHECKPOINTS", "5"),
            ("HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK", "1"),
        ]));

        assert_eq!(payload["enabled"], true);
        assert_eq!(payload["resident_enabled"], true);
        assert_eq!(payload["resident_checkpoint_max"], 5);
        assert_eq!(payload["disk_enabled"], true);
    }
}
