use hipfire_model::AcceleratorInventory;
use hipfire_scheduler::{PriorityPrefillScheduler, SchedulerPolicyEnv};

use crate::state::SharedState;

pub async fn server_accelerator_inventory(state: &SharedState) -> AcceleratorInventory {
    let mut engine = state.engine.lock().await;
    if let Some(engine) = engine.as_mut() {
        if let Ok(inventory) = engine.inventory().await {
            return inventory;
        }
    }
    AcceleratorInventory::not_probed()
}

pub fn server_prefill_scheduler_with_inventory(
    env: SchedulerPolicyEnv,
    inventory: AcceleratorInventory,
) -> PriorityPrefillScheduler {
    PriorityPrefillScheduler::with_accelerator_inventory(env, inventory)
}

pub async fn server_prefill_scheduler_from_state(
    state: &SharedState,
    env: SchedulerPolicyEnv,
) -> PriorityPrefillScheduler {
    server_prefill_scheduler_with_inventory(env, server_accelerator_inventory(state).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_model::{AcceleratorDeviceInfo, ModelWorkerKey};
    use hipfire_scheduler::{create_request_session_draft, CreateRequestSessionInput};

    fn worker(device_id: &str) -> ModelWorkerKey {
        ModelWorkerKey {
            artifact_path: "/models/qwen.hfq".to_string(),
            artifact_digest: Some("sha256:model".to_string()),
            arch_id: "qwen3.5".to_string(),
            quant_family: "mq4".to_string(),
            state_mode: "fp32".to_string(),
            max_seq_bucket: 4096,
            accelerator_kind: Some("hip".to_string()),
            device_id: Some(device_id.to_string()),
            feature_flags: vec!["serve".to_string(), "prefill_batch".to_string()],
        }
    }

    fn session(id: &str, device_id: &str) -> hipfire_scheduler::RequestSessionDraft {
        create_request_session_draft(CreateRequestSessionInput {
            id: id.to_string(),
            worker_key: worker(device_id),
            prompt_tokens: vec![1, 2, 3],
            cached_prefix_tokens: None,
            priority: None,
            state_kinds: vec!["kv".to_string(), "deltanet".to_string()],
        })
    }

    #[test]
    fn server_prefill_scheduler_uses_daemon_inventory_for_admission() {
        let inventory = AcceleratorInventory::from_devices(
            "daemon",
            vec![AcceleratorDeviceInfo::hip(
                "0",
                0,
                Some("gfx1201".to_string()),
                Some(24_000_000_000),
                Some(false),
                Some("HIP 6.4".to_string()),
            )],
        );
        let mut scheduler =
            server_prefill_scheduler_with_inventory(SchedulerPolicyEnv::empty(), inventory);

        scheduler.enqueue(session("ok", "0"), 0).unwrap();
        let err = scheduler.enqueue(session("missing", "1"), 0).unwrap_err();

        assert!(err.contains("worker_device_not_found"));
    }

    #[test]
    fn server_prefill_scheduler_preserves_unprobed_compatibility() {
        let mut scheduler = server_prefill_scheduler_with_inventory(
            SchedulerPolicyEnv::empty(),
            AcceleratorInventory::not_probed(),
        );

        scheduler.enqueue(session("unprobed", "7"), 0).unwrap();

        assert_eq!(scheduler.size(), 1);
    }
}
