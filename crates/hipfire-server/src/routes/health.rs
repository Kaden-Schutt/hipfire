use axum::{extract::State, response::Json};
use hipfire_scheduler::{
    server_batch_health_json, server_decode_batch_health_json, server_prefill_batch_health_json,
    server_state_cache_health_json, SchedulerPolicyEnv,
};
use hipfire_state::runtime_workers_health_json;
use serde_json::{json, Value};
use std::env;

use crate::state::SharedState;

pub async fn get_health(state: State<SharedState>) -> Json<Value> {
    let loaded = state.loaded_model_path.lock().await.clone();
    let scheduler_env = scheduler_env_from_process();
    Json(json!({
        "status": "ok",
        "model": loaded,
        "prefill_batch": server_prefill_batch_health_json(&scheduler_env),
        "decode_batch": server_decode_batch_health_json(&scheduler_env),
        "state_cache": server_state_cache_health_json(&scheduler_env),
        "runtime_workers": runtime_workers_health_json(&[], 0, None, 0, 0, "none"),
        "batches": server_batch_health_json(),
    }))
}

fn scheduler_env_from_process() -> SchedulerPolicyEnv {
    SchedulerPolicyEnv::from_pairs(env::vars())
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
            "runtime_workers": runtime_workers_health_json(&[], 0, None, 0, 0, "none"),
            "batches": server_batch_health_json(),
        });

        assert_eq!(payload["prefill_batch"], json!({ "enabled": false }));
        assert_eq!(payload["decode_batch"], json!({ "enabled": false }));
        assert_eq!(payload["state_cache"], json!({ "enabled": false }));
        assert_eq!(payload["runtime_workers"]["resident_workers"], 0);
        assert_eq!(payload["runtime_workers"]["state_arena_backend"], "none");
        assert_eq!(payload["runtime_workers"]["workers"], json!([]));
        assert_eq!(payload["batches"], json!({ "enabled": false }));
    }
}
