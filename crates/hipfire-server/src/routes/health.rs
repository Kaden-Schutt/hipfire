use axum::{extract::State, response::Json};
use hipfire_model::AcceleratorInventory;
use hipfire_scheduler::{
    server_decode_batch_health_json, server_prefill_batch_health_json,
    server_state_cache_health_json, SchedulerPolicyEnv,
};
use hipfire_state::runtime_workers_health_json_with_inventory;
use serde_json::{json, Value};
use std::env;

use crate::scheduler::server_accelerator_inventory;
use crate::state::SharedState;

pub async fn get_health(state: State<SharedState>) -> Json<Value> {
    let loaded = {
        let loaded = state.loaded_model_path.lock().await;
        loaded.clone()
    };
    let idle_timeout_sec = {
        let cfg = state.config.lock().await;
        cfg.idle_timeout
    };
    let prefill_queue_size = state.prefill_scheduler.lock().await.size();
    let selected_prefill_requests = state.selected_prefill_requests.lock().await.len();
    let accelerator_inventory = server_accelerator_inventory(&state).await;
    let scheduler_env = scheduler_env_from_process();
    let mut prefill_batch = server_prefill_batch_health_json(&scheduler_env);
    if let Some(obj) = prefill_batch.as_object_mut() {
        obj.insert("queue_size".to_string(), json!(prefill_queue_size));
        obj.insert("queued".to_string(), json!(prefill_queue_size));
        obj.insert(
            "selected_pending_dispatch".to_string(),
            json!(selected_prefill_requests),
        );
        if prefill_queue_size > 0 || selected_prefill_requests > 0 {
            obj.insert(
                "runtime_dispatch_skipped_reason".to_string(),
                json!("rust_server_requests_waiting_for_serial_daemon_dispatch"),
            );
        }
    }
    Json(json!({
        "status": "ok",
        "model": loaded,
        "idle_timeout_sec": idle_timeout_sec,
        "pid": std::process::id(),
        "prefill_batch": prefill_batch,
        "decode_batch": server_decode_batch_health_json(&scheduler_env),
        "state_cache": server_state_cache_health_json(&scheduler_env),
        "runtime_workers": runtime_workers_health_payload(&accelerator_inventory),
        "batches": batch_health_payload(&state).await,
    }))
}

fn scheduler_env_from_process() -> SchedulerPolicyEnv {
    SchedulerPolicyEnv::from_pairs(env::vars())
}

fn runtime_workers_health_payload(inventory: &AcceleratorInventory) -> serde_json::Value {
    runtime_workers_health_json_with_inventory(&[], 0, None, 0, 0, "none", inventory)
}

async fn batch_health_payload(state: &SharedState) -> serde_json::Value {
    let batches = state.batches.lock().await;
    let total = batches.len();
    let completed = batches
        .values()
        .filter(|batch| batch.status == "completed")
        .count();
    let failed = batches
        .values()
        .filter(|batch| batch.status == "failed")
        .count();
    let cancelled = batches
        .values()
        .filter(|batch| batch.status == "cancelled")
        .count();
    let queued = batches
        .values()
        .filter(|batch| {
            matches!(
                batch.status.as_str(),
                "validating" | "in_progress" | "finalizing"
            )
        })
        .count();
    json!({
        "enabled": true,
        "queued": queued,
        "selected": queued,
        "total": total,
        "failed": failed,
        "cancelled": cancelled,
        "completed": completed,
        "completion_window_supported": true,
        "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
        "execution_mode": "serial_fallback",
        "last_fallback_reason": "daemon_serialized_request_path",
        "batch_capability": "supported",
        "batch_capability_reason": "rust_axum_batch_control_plane",
        "selected_batch_execution_mode": "serial_fallback",
        "fallback_reason": "generate_batch_prefill_not_used_for_file_batches",
        "runtime_dispatch_skipped_reason": "batch_jobs_execute_via_blocking_routes",
        "unsupported_mode_hits_total": 0,
        "validation_errors_total": failed,
        "streaming_rejections_total": 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_route_uses_disabled_shared_scheduler_payloads() {
        let payload = json!({
            "prefill_batch": server_prefill_batch_health_json(&SchedulerPolicyEnv::empty()),
            "decode_batch": server_decode_batch_health_json(&SchedulerPolicyEnv::empty()),
            "state_cache": server_state_cache_health_json(&SchedulerPolicyEnv::empty()),
            "runtime_workers": runtime_workers_health_payload(&AcceleratorInventory::not_probed()),
            "batches": json!({ "enabled": true }),
        });

        assert_eq!(payload["prefill_batch"], json!({ "enabled": false }));
        assert_eq!(payload["decode_batch"], json!({ "enabled": false }));
        assert_eq!(payload["state_cache"], json!({ "enabled": false }));
        assert_eq!(payload["runtime_workers"]["resident_workers"], 0);
        assert_eq!(payload["runtime_workers"]["state_arena_backend"], "none");
        assert_eq!(
            payload["runtime_workers"]["accelerator_inventory"]["source"],
            "not_probed"
        );
        assert_eq!(
            payload["runtime_workers"]["accelerator_inventory"]["device_count"],
            0
        );
        assert_eq!(payload["runtime_workers"]["workers"], json!([]));
        assert_eq!(payload["batches"], json!({ "enabled": true }));
    }

    #[test]
    fn health_runtime_workers_can_embed_daemon_inventory() {
        let inventory = AcceleratorInventory::from_devices(
            "daemon",
            vec![hipfire_model::AcceleratorDeviceInfo::hip(
                "0",
                0,
                Some("gfx1201".to_string()),
                Some(24_000_000_000),
                Some(false),
                Some("HIP 6.4".to_string()),
            )],
        );
        let payload = runtime_workers_health_payload(&inventory);

        assert_eq!(payload["accelerator_inventory"]["source"], "daemon");
        assert_eq!(payload["accelerator_inventory"]["device_count"], 1);
        assert_eq!(
            payload["accelerator_inventory"]["devices"][0]["device_class"],
            "discrete"
        );
    }
}
