use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

use hipfire_config::HipfireConfig;
use hipfire_daemon_adapter::DaemonEngine;
use hipfire_scheduler::{PriorityPrefillScheduler, SchedulerPolicyEnv};

pub struct AppState {
    /// Serializes all daemon I/O. Phase A: one request at a time.
    pub engine: Mutex<Option<DaemonEngine>>,
    pub config: Mutex<HipfireConfig>,
    /// Worker key ID of the currently loaded model, if any.
    pub loaded_model_path: Mutex<Option<String>>,
    /// Shared prefill scheduler used by Rust request paths when enabled.
    pub prefill_scheduler: Mutex<PriorityPrefillScheduler>,
    /// Request IDs selected by the scheduler and ready to enter daemon I/O.
    pub selected_prefill_requests: Mutex<HashSet<String>>,
    /// Serializes scheduler selection so one request path chooses batches at a time.
    pub prefill_dispatch: Mutex<()>,
    pub prefill_notify: Notify,
}

impl AppState {
    pub fn new(config: HipfireConfig) -> Arc<Self> {
        let scheduler_env = SchedulerPolicyEnv::from_pairs(std::env::vars());
        Arc::new(Self {
            engine: Mutex::new(None),
            config: Mutex::new(config),
            loaded_model_path: Mutex::new(None),
            prefill_scheduler: Mutex::new(PriorityPrefillScheduler::new(scheduler_env)),
            selected_prefill_requests: Mutex::new(HashSet::new()),
            prefill_dispatch: Mutex::new(()),
            prefill_notify: Notify::new(),
        })
    }
}

pub type SharedState = Arc<AppState>;
