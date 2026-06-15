use std::sync::Arc;
use tokio::sync::Mutex;

use hipfire_config::HipfireConfig;
use hipfire_daemon_adapter::DaemonEngine;

pub struct AppState {
    /// Serializes all daemon I/O. Phase A: one request at a time.
    pub engine: Mutex<Option<DaemonEngine>>,
    pub config: Mutex<HipfireConfig>,
    /// Worker key ID of the currently loaded model, if any.
    pub loaded_model_path: Mutex<Option<String>>,
}

impl AppState {
    pub fn new(config: HipfireConfig) -> Arc<Self> {
        Arc::new(Self {
            engine: Mutex::new(None),
            config: Mutex::new(config),
            loaded_model_path: Mutex::new(None),
        })
    }
}

pub type SharedState = Arc<AppState>;
