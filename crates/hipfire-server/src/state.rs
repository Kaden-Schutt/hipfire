use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Notify};

use hipfire_config::{HipfireConfig, LoadedConfig};
use hipfire_daemon_adapter::DaemonEngine;
use hipfire_diffusion::{DiffusionGenerationRuntimeOptions, DiffusionPipeline};
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

#[derive(Clone, Debug, Default)]
pub struct SdapiProgressState {
    pub active: bool,
    pub skipped: bool,
    pub interrupted: bool,
    pub task_id: Option<String>,
    pub mode: Option<String>,
    pub prompt: Option<String>,
    pub sampling_step: usize,
    pub sampling_steps: usize,
    pub current_image: Option<String>,
    pub textinfo: Option<String>,
    pub started_at_unix_secs: Option<u64>,
    pub completed_at_unix_secs: Option<u64>,
}

pub struct AppState {
    /// Serializes all daemon I/O. Phase A: one request at a time.
    pub engine: Mutex<Option<DaemonEngine>>,
    pub loaded_config: Mutex<LoadedConfig>,
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
    /// Opened diffusion HFQ pipelines keyed by resolved model path.
    pub diffusion_pipelines: Mutex<HashMap<PathBuf, Arc<DiffusionPipeline>>>,
    /// Daemon-resolved default diffusion runtime backend. Resolved once at
    /// `serve` launch (HIP/ROCm-first: the detected GPU, or the CPU reference
    /// oracle when `HIPFIRE_DIFFUSION_CPU_REFERENCE` is set). The bare
    /// constructor leaves this as the CPU reference so unit/route tests run
    /// without a GPU; a per-request `rocm_device_id` still overrides it.
    pub diffusion_runtime_default: StdMutex<DiffusionGenerationRuntimeOptions>,
    /// Stable-Diffusion-WebUI option compatibility values posted to
    /// `/sdapi/v1/options` that do not map to native Hipfire config fields.
    pub sdapi_options: Mutex<HashMap<String, serde_json::Value>>,
    pub sdapi_progress: Arc<StdMutex<SdapiProgressState>>,
    pub last_request_unix_secs: Mutex<u64>,
    pub training_runs_dir: PathBuf,
    /// Server-owned root for images saved by the SD API routes. Derived from
    /// config at construction; request `outdir_*` overrides never reach it.
    pub sdapi_output_root: PathBuf,
    /// Local admin bearer secret (`~/.hipfire/admin.secret`); same-box
    /// CLI/TUI present this to skip the `/admin` login flow.
    pub admin_secret: String,
    /// Active `/admin` browser sessions: token -> expiry (unix secs).
    pub admin_sessions: Mutex<HashMap<String, u64>>,
}

impl AppState {
    pub fn new(config: HipfireConfig) -> Arc<Self> {
        Self::new_loaded(LoadedConfig::from_config(config))
    }

    pub fn new_loaded(loaded_config: LoadedConfig) -> Arc<Self> {
        let training_runs_dir =
            hipfire_operator::training::training_runs_dir(hipfire_config::hipfire_dir());
        Self::new_loaded_with_training_runs_dir(loaded_config, training_runs_dir)
    }

    pub fn new_loaded_with_training_runs_dir(
        loaded_config: LoadedConfig,
        training_runs_dir: PathBuf,
    ) -> Arc<Self> {
        let scheduler_env = SchedulerPolicyEnv::from_pairs(std::env::vars());
        let config = loaded_config.config.clone();
        let sdapi_output_root = PathBuf::from(&config.sdapi_output_root);
        Arc::new(Self {
            engine: Mutex::new(None),
            loaded_config: Mutex::new(loaded_config),
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
            diffusion_pipelines: Mutex::new(HashMap::new()),
            diffusion_runtime_default: StdMutex::new(
                DiffusionGenerationRuntimeOptions::cpu_reference(),
            ),
            sdapi_options: Mutex::new(HashMap::new()),
            sdapi_progress: Arc::new(StdMutex::new(SdapiProgressState::default())),
            last_request_unix_secs: Mutex::new(now_secs()),
            training_runs_dir,
            sdapi_output_root,
            admin_secret: hipfire_config::ensure_admin_secret().unwrap_or_default(),
            admin_sessions: Mutex::new(HashMap::new()),
        })
    }

    /// Resolve the daemon's default diffusion backend once at `serve` launch.
    /// HIP/ROCm-first: detect the primary GPU and use it; honor
    /// `HIPFIRE_DIFFUSION_CPU_REFERENCE` for the CPU oracle; if no GPU is
    /// available and CPU was not requested, warn and keep the CPU reference so
    /// the daemon still serves (slowly) rather than failing every request.
    pub fn resolve_diffusion_runtime_default(&self) {
        let resolved = if DiffusionGenerationRuntimeOptions::cpu_reference_requested() {
            eprintln!(
                "[hipfire] diffusion: HIPFIRE_DIFFUSION_CPU_REFERENCE set; using the CPU reference oracle"
            );
            DiffusionGenerationRuntimeOptions::cpu_reference()
        } else {
            match hipfire_runtime::multi_gpu::resolve_primary_device(None) {
                Ok(device) => DiffusionGenerationRuntimeOptions::rocm_hybrid(device),
                Err(error) => {
                    eprintln!(
                        "[hipfire] diffusion: no ROCm device resolved ({error}); falling back to \
                         the slow CPU reference oracle. Set HIPFIRE_DIFFUSION_CPU_REFERENCE=1 to \
                         silence this warning."
                    );
                    DiffusionGenerationRuntimeOptions::cpu_reference()
                }
            }
        };
        *self.diffusion_runtime_default.lock().unwrap() = resolved;
    }

    /// The daemon's resolved default diffusion backend (see
    /// [`resolve_diffusion_runtime_default`]).
    pub fn diffusion_runtime_default(&self) -> DiffusionGenerationRuntimeOptions {
        *self.diffusion_runtime_default.lock().unwrap()
    }
}

pub type SharedState = Arc<AppState>;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
