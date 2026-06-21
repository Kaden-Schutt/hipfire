//! Serde types shared between the hipfire server (`hipfire-server`) and the
//! WASM admin console (`hipfire-admin-ui`). Pure data + serde — no runtime,
//! no platform deps — so it compiles for both native and `wasm32`.

use serde::{Deserialize, Serialize};

/// Per-GPU telemetry snapshot. Every metric is independently optional so a
/// missing sysfs node degrades gracefully rather than dropping the card.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GpuTelemetry {
    /// DRM card name, e.g. "card1".
    pub card: String,
    /// GPU utilization 0–100.
    pub busy_percent: Option<u32>,
    /// Dedicated VRAM in use (bytes). On APUs this is the carveout, not GTT.
    pub vram_used_bytes: Option<u64>,
    /// Dedicated VRAM total (bytes). On APUs this is the carveout, not GTT.
    pub vram_total_bytes: Option<u64>,
    /// Edge/junction temperature (°C).
    pub temp_c: Option<f64>,
    /// Average board power draw (W).
    pub power_w: Option<f64>,
    /// Active shader clock (MHz).
    pub sclk_mhz: Option<u64>,
}

/// Top-level payload for `GET /admin/stats`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AdminStats {
    /// Server clock when the snapshot was taken (unix seconds).
    pub generated_unix: u64,
    /// One entry per AMD GPU visible to the host.
    pub gpus: Vec<GpuTelemetry>,
}
