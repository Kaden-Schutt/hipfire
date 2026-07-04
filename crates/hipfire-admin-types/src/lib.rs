//! Serde types shared between the hipfire server (`hipfire-server`) and the
//! WASM admin console (`hipfire-admin-ui`). Pure data + serde — no runtime,
//! no platform deps — so it compiles for both native and `wasm32`.
//!
//! The collection side (sysfs / `/proc` reading) lives in the native-only
//! `hipfire-sysinfo` crate. The render side — webUI (wasm) and TUI — consumes
//! these types and the derived-value helpers below (`primary_pool`,
//! `MemPool::percent`, [`fmt_bytes`]), which are pure and therefore identical
//! on both targets. Keep platform/`std::fs` access out of this crate.

use serde::{Deserialize, Serialize};

/// A single addressable GPU memory pool (VRAM or GTT) reduced to the two
/// numbers a UI actually renders, plus a human label. Derived from
/// [`GpuTelemetry`]; not collected directly.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemPool {
    /// Short pool label, e.g. "VRAM", "GTT", or "VRAM (carveout)".
    pub label: String,
    /// Bytes currently allocated in this pool.
    pub used_bytes: u64,
    /// Pool capacity in bytes.
    pub total_bytes: u64,
}

impl MemPool {
    /// Free bytes, saturating so a transient used>total race can't underflow.
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }

    /// Fill percentage in `0.0..=100.0`; `0.0` when capacity is unknown/zero.
    pub fn percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

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
    /// CPU-visible (BAR-mapped) VRAM in use (bytes). Equals VRAM on APUs and
    /// on ReBAR dGPUs; a smaller window on legacy-BAR discrete cards.
    pub vis_vram_used_bytes: Option<u64>,
    /// CPU-visible (BAR-mapped) VRAM total (bytes).
    pub vis_vram_total_bytes: Option<u64>,
    /// GTT in use (bytes) — the GART window into system RAM. This is the pool
    /// the runtime actually fills on UMA APUs.
    pub gtt_used_bytes: Option<u64>,
    /// GTT total (bytes) — capacity of the GART window into system RAM.
    pub gtt_total_bytes: Option<u64>,
    /// Whether this is an integrated/UMA device (APU). `None` when the
    /// collector could not determine it. When true the meaningful capacity is
    /// GTT, not the small VRAM carveout — see [`GpuTelemetry::primary_pool`].
    pub integrated: Option<bool>,
    /// Edge/junction temperature (°C).
    pub temp_c: Option<f64>,
    /// Average board power draw (W).
    pub power_w: Option<f64>,
    /// Active shader clock (MHz).
    pub sclk_mhz: Option<u64>,
    /// Firmware `gpu_metrics` extras (socket power, soc/gfx temp, throttle,
    /// DRAM bandwidth) when the binary table could be parsed. `None` if the
    /// device exposes no `gpu_metrics` node or an unrecognized version.
    pub metrics: Option<GpuMetrics>,
}

/// Decoded subset of the firmware `gpu_metrics` table — the fields plain sysfs
/// nodes don't expose. Every field is optional because layout and population
/// vary across the versioned struct (v1 dGPU, v2 APU, v3 Strix-class).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GpuMetrics {
    /// `(format_revision, content_revision)` of the parsed table.
    pub version: (u8, u8),
    /// Whole-package power draw (W) — `average_socket_power`.
    pub socket_power_w: Option<f64>,
    /// GFX die temperature (°C) — finer than the hwmon edge sensor.
    pub gfx_temp_c: Option<f64>,
    /// SoC temperature (°C).
    pub soc_temp_c: Option<f64>,
    /// Raw throttle bitmask; `0` means not throttling. `None` on v3 tables,
    /// which report per-reason residency accumulators instead.
    pub throttle_status: Option<u64>,
    /// DRAM read bandwidth (MB/s) — v3 tables only.
    pub dram_read_mbps: Option<f64>,
    /// DRAM write bandwidth (MB/s) — v3 tables only.
    pub dram_write_mbps: Option<f64>,
}

impl GpuMetrics {
    /// True when the throttle bitmask is present and nonzero.
    pub fn throttling(&self) -> bool {
        matches!(self.throttle_status, Some(s) if s != 0)
    }
}

/// One process holding GPU memory, parsed from `/proc/<pid>/fdinfo`. Engine
/// utilization is intentionally absent: it's a cumulative-nanosecond counter
/// that needs two samples to rate, and this kernel doesn't emit `drm-engine-*`
/// at all — so we surface the always-present per-process memory instead.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClientUsage {
    /// Owning process id.
    pub pid: u32,
    /// Process comm (short name).
    pub comm: String,
    /// DRM card this client's memory lives on (matched via PCI address), if
    /// resolvable.
    pub card: Option<String>,
    /// Resident VRAM (bytes).
    pub vram_bytes: u64,
    /// Resident GTT (bytes).
    pub gtt_bytes: u64,
}

/// Live NPU (AMD XDNA / Ryzen AI) telemetry, collected via `hipfire-xdna`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NpuTelemetry {
    /// Accel node path, e.g. "/dev/accel/accel0".
    pub node: String,
    /// Total NPU power (W).
    pub power_w: Option<f64>,
    /// NPU temperature (deg C), if the driver/PMF stack exposes it.
    pub temp_c: Option<f64>,
    /// Mean per-column utilization (%).
    pub mean_util_pct: f64,
    /// Per-column utilization (%), one entry per active column.
    pub columns_pct: Vec<u32>,
    /// Current / max throughput (TOPS).
    pub tops_current: u64,
    pub tops_max: u64,
    /// Active / max hardware contexts (tasks).
    pub tasks_current: u64,
    pub tasks_max: u64,
    /// MP-NPU clock (MHz).
    pub mp_npu_mhz: u32,
    /// H clock (MHz).
    pub h_mhz: u32,
}

impl NpuTelemetry {
    /// NPU "capacity" pool as a [`MemPool`]-style pair for uniform rendering:
    /// active vs max hardware contexts.
    pub fn tasks_pool(&self) -> MemPool {
        MemPool {
            label: "NPU tasks".to_string(),
            used_bytes: self.tasks_current,
            total_bytes: self.tasks_max,
        }
    }
}

impl GpuTelemetry {
    /// True when the card is integrated/UMA. Falls back to a sysfs-only
    /// heuristic when the collector left `integrated` unset: a VRAM carveout
    /// much smaller than the GTT window (≤ ¼) is the APU signature, since a
    /// discrete card's local VRAM is never dwarfed by its system-RAM window.
    pub fn is_integrated(&self) -> bool {
        if let Some(flag) = self.integrated {
            return flag;
        }
        match (self.vram_total_bytes, self.gtt_total_bytes) {
            (Some(vram), Some(gtt)) => gtt > 0 && vram < gtt / 4,
            _ => false,
        }
    }

    /// The VRAM pool as a [`MemPool`], if both numbers are present.
    pub fn vram_pool(&self) -> Option<MemPool> {
        Some(MemPool {
            label: if self.is_integrated() {
                "VRAM (carveout)".to_string()
            } else {
                "VRAM".to_string()
            },
            used_bytes: self.vram_used_bytes?,
            total_bytes: self.vram_total_bytes?,
        })
    }

    /// The GTT pool as a [`MemPool`], if both numbers are present.
    pub fn gtt_pool(&self) -> Option<MemPool> {
        Some(MemPool {
            label: "GTT".to_string(),
            used_bytes: self.gtt_used_bytes?,
            total_bytes: self.gtt_total_bytes?,
        })
    }

    /// The pool that actually governs out-of-memory on this device: GTT on
    /// integrated/UMA parts (the VRAM carveout is a stub there), local VRAM on
    /// discrete cards. This is the single bar a UI should lead with. Falls back
    /// to whichever pool is available if the preferred one is missing.
    pub fn primary_pool(&self) -> Option<MemPool> {
        if self.is_integrated() {
            self.gtt_pool().or_else(|| self.vram_pool())
        } else {
            self.vram_pool().or_else(|| self.gtt_pool())
        }
    }

    /// Total GPU-attributed bytes in flight across VRAM + GTT. On a dGPU a
    /// nonzero GTT component means VRAM has overflowed to host memory over
    /// PCIe; on an APU the two are distinct slices of the same DRAM. `None`
    /// when neither pool reported a `used` figure.
    pub fn total_gpu_used_bytes(&self) -> Option<u64> {
        match (self.vram_used_bytes, self.gtt_used_bytes) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
        }
    }
}

/// Host system-memory snapshot from the kernel's own accounting
/// (`/proc/meminfo`). On UMA APUs this is the ground truth for "how close to
/// OOM": GTT allocations are real system pages and already counted here.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HostMemory {
    /// `MemTotal` — usable RAM (already excludes any firmware VRAM carveout).
    pub total_bytes: u64,
    /// `MemAvailable` — the kernel's estimate of allocatable memory without
    /// swapping, accounting for reclaimable page cache. Prefer this over free.
    pub available_bytes: u64,
}

impl HostMemory {
    /// Used = total − available. This is the honest "used" figure; raw
    /// MemFree would ignore reclaimable cache and overstate pressure.
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.available_bytes)
    }

    /// Used percentage in `0.0..=100.0`.
    pub fn percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes() as f64 / self.total_bytes as f64) * 100.0
        }
    }

    /// This same memory rendered as a [`MemPool`] for uniform UI treatment.
    pub fn as_pool(&self) -> MemPool {
        MemPool {
            label: "System RAM".to_string(),
            used_bytes: self.used_bytes(),
            total_bytes: self.total_bytes,
        }
    }
}

/// Top-level payload for `GET /admin/stats`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AdminStats {
    /// Server clock when the snapshot was taken (unix seconds).
    pub generated_unix: u64,
    /// One entry per AMD GPU visible to the host.
    pub gpus: Vec<GpuTelemetry>,
    /// Host system memory, when readable. `None` if `/proc/meminfo` was
    /// unavailable (e.g. a non-Linux build target).
    pub host: Option<HostMemory>,
    /// Processes holding GPU memory (from `/proc/<pid>/fdinfo`), most-VRAM
    /// first. Empty when none are resolvable.
    #[serde(default)]
    pub clients: Vec<ClientUsage>,
    /// AMD XDNA NPUs with live telemetry. Empty when no NPU is present or the
    /// accel node is inaccessible.
    #[serde(default)]
    pub npus: Vec<NpuTelemetry>,
}

/// Format a byte count as a compact human string (e.g. `1.5 GiB`). Binary
/// units. Shared so the webUI and TUI render identical labels.
pub fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apu() -> GpuTelemetry {
        // gfx1103-style: 256 MiB carveout, 42 GiB GTT window.
        GpuTelemetry {
            vram_used_bytes: Some(91_074_560),
            vram_total_bytes: Some(268_435_456),
            gtt_used_bytes: Some(14_184_448),
            gtt_total_bytes: Some(45_097_156_608),
            ..Default::default()
        }
    }

    fn dgpu() -> GpuTelemetry {
        // W7800-style: 48 GiB local VRAM, ~16 GiB system-RAM GTT window.
        GpuTelemetry {
            vram_used_bytes: Some(8_000_000_000),
            vram_total_bytes: Some(51_539_607_552),
            gtt_used_bytes: Some(0),
            gtt_total_bytes: Some(17_179_869_184),
            ..Default::default()
        }
    }

    #[test]
    fn integrated_heuristic_distinguishes_apu_from_dgpu() {
        assert!(apu().is_integrated());
        assert!(!dgpu().is_integrated());
    }

    #[test]
    fn explicit_integrated_flag_overrides_heuristic() {
        let mut g = dgpu();
        g.integrated = Some(true);
        assert!(g.is_integrated());
    }

    #[test]
    fn primary_pool_picks_gtt_on_apu_and_vram_on_dgpu() {
        assert_eq!(apu().primary_pool().unwrap().label, "GTT");
        assert_eq!(apu().primary_pool().unwrap().total_bytes, 45_097_156_608);
        assert_eq!(dgpu().primary_pool().unwrap().label, "VRAM");
        assert_eq!(dgpu().primary_pool().unwrap().total_bytes, 51_539_607_552);
    }

    #[test]
    fn primary_pool_falls_back_when_preferred_missing() {
        // APU with no GTT numbers still yields the carveout pool.
        let g = GpuTelemetry {
            vram_used_bytes: Some(10),
            vram_total_bytes: Some(100),
            integrated: Some(true),
            ..Default::default()
        };
        assert_eq!(g.primary_pool().unwrap().label, "VRAM (carveout)");
    }

    #[test]
    fn total_gpu_used_sums_pools() {
        assert_eq!(apu().total_gpu_used_bytes(), Some(91_074_560 + 14_184_448));
        assert_eq!(GpuTelemetry::default().total_gpu_used_bytes(), None);
    }

    #[test]
    fn mempool_percent_and_free() {
        let p = MemPool {
            label: "X".into(),
            used_bytes: 25,
            total_bytes: 100,
        };
        assert_eq!(p.percent(), 25.0);
        assert_eq!(p.free_bytes(), 75);
        assert_eq!(MemPool::default().percent(), 0.0);
    }

    #[test]
    fn host_memory_used_prefers_available() {
        let h = HostMemory {
            total_bytes: 1000,
            available_bytes: 600,
        };
        assert_eq!(h.used_bytes(), 400);
        assert_eq!(h.percent(), 40.0);
        assert_eq!(h.as_pool().total_bytes, 1000);
    }

    #[test]
    fn fmt_bytes_units() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(1536), "1.5 KiB");
        assert_eq!(fmt_bytes(268_435_456), "256.0 MiB");
        assert_eq!(fmt_bytes(45_097_156_608), "42.0 GiB");
    }
}
