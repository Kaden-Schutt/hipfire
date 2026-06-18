use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Notify};

use hipfire_config::HipfireConfig;
use hipfire_daemon_adapter::DaemonEngine;
use hipfire_prompt::Message;
use hipfire_scheduler::{PriorityPrefillScheduler, SchedulerPolicyEnv};

#[derive(Clone, Debug)]
pub struct StoredResponsesContext {
    pub messages: Vec<Message>,
}

#[derive(Clone, Debug)]
pub struct StoredFile {
    pub id: String,
    pub filename: String,
    pub bytes: usize,
    pub purpose: String,
    pub created_at: u64,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct StoredBatch {
    pub id: String,
    pub status: String,
    pub endpoint: String,
    pub completion_window: String,
    pub input_file_id: String,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub request_count: usize,
    pub completed_requests: usize,
    pub created_at: u64,
    pub in_progress_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub failed_reason: Option<String>,
}

pub struct AppState {
    /// Serializes all daemon I/O. Phase A: one request at a time.
    pub engine: Mutex<Option<DaemonEngine>>,
    pub config: Mutex<HipfireConfig>,
    /// Worker key ID of the currently loaded model, if any.
    pub loaded_model_path: Mutex<Option<String>>,
    /// Whether the current daemon worker can safely keep prefix-cache state across requests.
    pub loaded_model_cache_capable: Mutex<Option<bool>>,
    /// Effective max_seq used when the current model was loaded.
    pub loaded_model_max_seq: Mutex<Option<u32>>,
    /// Shared prefill scheduler used by Rust request paths when enabled.
    pub prefill_scheduler: Mutex<PriorityPrefillScheduler>,
    /// Request IDs selected by the scheduler and ready to enter daemon I/O.
    pub selected_prefill_requests: Mutex<HashSet<String>>,
    /// Serializes scheduler selection so one request path chooses batches at a time.
    pub prefill_dispatch: Mutex<()>,
    pub prefill_notify: Notify,
    pub responses_contexts: Mutex<HashMap<String, StoredResponsesContext>>,
    pub responses_order: Mutex<VecDeque<String>>,
    pub files: Mutex<HashMap<String, StoredFile>>,
    pub file_order: Mutex<VecDeque<String>>,
    pub batches: Mutex<HashMap<String, StoredBatch>>,
    pub batch_order: Mutex<VecDeque<String>>,
    pub last_request_unix_secs: Mutex<u64>,
}

impl AppState {
    pub fn new(config: HipfireConfig) -> Arc<Self> {
        let scheduler_env = SchedulerPolicyEnv::from_pairs(std::env::vars());
        Arc::new(Self {
            engine: Mutex::new(None),
            config: Mutex::new(config),
            loaded_model_path: Mutex::new(None),
            loaded_model_cache_capable: Mutex::new(None),
            loaded_model_max_seq: Mutex::new(None),
            prefill_scheduler: Mutex::new(PriorityPrefillScheduler::new(scheduler_env)),
            selected_prefill_requests: Mutex::new(HashSet::new()),
            prefill_dispatch: Mutex::new(()),
            prefill_notify: Notify::new(),
            responses_contexts: Mutex::new(HashMap::new()),
            responses_order: Mutex::new(VecDeque::new()),
            files: Mutex::new(HashMap::new()),
            file_order: Mutex::new(VecDeque::new()),
            batches: Mutex::new(HashMap::new()),
            batch_order: Mutex::new(VecDeque::new()),
            last_request_unix_secs: Mutex::new(now_secs()),
        })
    }
}

pub type SharedState = Arc<AppState>;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
