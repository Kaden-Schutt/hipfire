//! NPU telemetry via the `hipfire-xdna` device layer.
//!
//! Wraps `XdnaDevice` (AMD XDNA / Ryzen AI, `/dev/accel/accelN`) into the
//! wasm-safe `NpuTelemetry` the admin surfaces render. Each sub-query is
//! best-effort: a missing `amd_pmf` (no sensors) shouldn't drop the clocks or
//! TOPS, and no NPU at all yields an empty vec rather than an error. Power is
//! converted mW → W to match the GPU side.

use hipfire_admin_types::NpuTelemetry;
use hipfire_xdna::XdnaDevice;

/// Collect telemetry for the default NPU accel node, if one is present and
/// accessible. Returns at most one entry today (single-NPU boxes); the vec
/// shape leaves room for multi-NPU hosts.
pub fn read_npus() -> Vec<NpuTelemetry> {
    let Ok(dev) = XdnaDevice::open_default() else {
        return Vec::new();
    };

    let mut t = NpuTelemetry {
        node: dev.path().to_string(),
        ..Default::default()
    };

    if let Ok(s) = dev.sensors() {
        t.power_w = s.power_mw.map(|mw| mw as f64 / 1000.0);
        t.temp_c = s.temp_c.map(|c| c as f64);
        t.mean_util_pct = s.mean_utilization_pct() as f64;
        t.columns_pct = s.column_utilization_pct;
    }
    if let Ok(r) = dev.resource_info() {
        t.tops_current = r.npu_tops_curr;
        t.tops_max = r.npu_tops_max;
        t.tasks_current = r.npu_task_curr;
        t.tasks_max = r.npu_task_max;
    }
    if let Ok(c) = dev.clocks() {
        t.mp_npu_mhz = c.mp_npu_mhz;
        t.h_mhz = c.h_mhz;
    }

    vec![t]
}
