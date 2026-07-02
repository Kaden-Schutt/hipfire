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
    let diffusion = diffusion_health_payload(&state).await;
    let active_model = loaded
        .clone()
        .or_else(|| diffusion_active_model(&diffusion));
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
        "version": hipfire_build_info::VERSION,
        "model": loaded,
        "active_model": active_model,
        "diffusion": diffusion,
        "idle_timeout_sec": idle_timeout_sec,
        "pid": std::process::id(),
        "prefill_batch": prefill_batch,
        "decode_batch": server_decode_batch_health_json(&scheduler_env),
        "state_cache": server_state_cache_health_json(&scheduler_env),
        "runtime_workers": runtime_workers_health_payload(&accelerator_inventory),
        "batches": batch_health_payload(&state).await,
    }))
}

async fn diffusion_health_payload(state: &SharedState) -> Value {
    let cache = state.diffusion_pipelines.lock().await;
    let mut models = cache
        .iter()
        .map(|(path, pipeline)| {
            let summary = pipeline.summary();
            json!({
                "path": path,
                "title": summary.title,
                "model_name": summary.model_name,
                "pipeline": summary.pipeline_class,
                "weight_format": summary.weight_format,
                "max_batch": summary.max_batch,
            })
        })
        .collect::<Vec<_>>();
    models.sort_by(|left, right| {
        left["path"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["path"].as_str().unwrap_or_default())
    });
    json!({
        "loaded": !models.is_empty(),
        "pipeline_count": models.len(),
        "models": models,
    })
}

fn diffusion_active_model(diffusion: &Value) -> Option<String> {
    diffusion
        .get("models")
        .and_then(Value::as_array)
        .and_then(|models| models.first())
        .and_then(|model| model.get("path"))
        .and_then(Value::as_str)
        .map(str::to_string)
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
    use axum::extract::State;
    use hipfire_diffusion::{
        DiffusionBatchMetadata, DiffusionHfqMetadata, DiffusionPipeline, DiffusionPipelineMetadata,
        DiffusionQuantizationMetadata, DiffusionTokenizerMetadata, DIFFUSION_ARTIFACT_KIND,
        DIFFUSION_SCHEMA_VERSION, HFQ_ARCH_DIFFUSION,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use std::collections::BTreeMap;
    use std::sync::Arc;

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

    #[tokio::test]
    async fn health_reports_cached_diffusion_pipeline_as_active_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-health-diffusion-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("metadata-only-diffusion.hfq");
        write_metadata_only_diffusion_hfq(&hfq_path);

        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let pipeline = Arc::new(DiffusionPipeline::open_hfq(&hfq_path).unwrap());
        state
            .diffusion_pipelines
            .lock()
            .await
            .insert(hfq_path.clone(), pipeline);

        let Json(payload) = get_health(State(state)).await;

        assert_eq!(payload["model"], Value::Null);
        assert_eq!(
            payload["active_model"].as_str().unwrap(),
            hfq_path.to_string_lossy()
        );
        assert_eq!(payload["diffusion"]["loaded"], true);
        assert_eq!(payload["diffusion"]["pipeline_count"], 1);
        assert_eq!(
            payload["diffusion"]["models"][0]["model_name"],
            "metadata-only-diffusion"
        );
        assert_eq!(
            payload["diffusion"]["models"][0]["pipeline"],
            "StableDiffusionPipeline"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_metadata_only_diffusion_hfq(path: &std::path::Path) {
        let metadata = DiffusionHfqMetadata {
            artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
            schema_version: DIFFUSION_SCHEMA_VERSION,
            pipeline: DiffusionPipelineMetadata {
                class_name: "StableDiffusionPipeline".to_string(),
                source: "/tmp/metadata-only-diffusion".to_string(),
                model_name: "metadata-only-diffusion".to_string(),
                latent_channels: Some(4),
                latent_height: Some(64),
                latent_width: Some(64),
                supported_widths: vec![512],
                supported_heights: vec![512],
            },
            tokenizer: DiffusionTokenizerMetadata::default(),
            tokenizer_2: None,
            batch: DiffusionBatchMetadata {
                max_batch: 1,
                batched_runtime: true,
            },
            quantization: DiffusionQuantizationMetadata {
                weight_format: "metadata-only".to_string(),
                activation_format: "fp16".to_string(),
                tensor_roles_version: 1,
            },
            components: BTreeMap::new(),
        };
        let tensors: Vec<HfqMemTensor> = Vec::new();
        write_hfqm_package_mem(
            path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
    }
}
