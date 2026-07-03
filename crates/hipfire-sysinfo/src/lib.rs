//! Portable host/GPU memory telemetry for hipfire's admin surfaces.
//!
//! Collects from the kernel directly — sysfs (`/sys/class/drm/card*/device`)
//! and `/proc/meminfo` — with no `rocm-smi` spawn, no root, and crucially **no
//! HIP/ROCm dependency**, so the same reader serves the HTTP server, the TUI,
//! and any non-GPU build target. The runtime-independent shape is deliberate:
//! the TUI must render memory without initializing a HIP context.
//!
//! Portability across the fleet (RDNA2/3/4 dGPUs and APUs) comes from reading
//! every AMD card under DRM rather than a hardcoded `card1`, and from surfacing
//! both VRAM and GTT so callers can pick the pool that governs OOM on each
//! device class — see [`hipfire_admin_types::GpuTelemetry::primary_pool`].
//!
//! Output is the wasm-safe serde types in [`hipfire_admin_types`]; the derived
//! "which pool matters / percent / human bytes" logic lives there so the webUI
//! (wasm) and TUI compute it identically.

use std::fs;
use std::path::Path;

use hipfire_admin_types::AdminStats;

mod fdinfo;
mod gpu;
mod gpu_metrics;
mod host;
mod host_profile;
mod npu;

pub use fdinfo::read_clients;
pub use gpu::read_gpu_telemetry;
pub use gpu_metrics::read_gpu_metrics;
pub use host::read_host_memory;
pub use host_profile::{
    collect_default_host_profile, collect_host_profile, detect_arch, parse_pp_dpm_mclk_max_mhz,
    HostProfileOverrides,
};
pub use npu::read_npus;

/// One-shot snapshot of every AMD GPU (with firmware `gpu_metrics` extras),
/// host system memory, per-process GPU memory, and any NPU. This is the single
/// call the server route and the TUI poll.
pub fn snapshot(generated_unix: u64) -> AdminStats {
    AdminStats {
        generated_unix,
        gpus: read_gpu_telemetry(),
        host: read_host_memory(),
        clients: read_clients(),
        npus: read_npus(),
    }
}

// ── Shared sysfs/proc primitives ────────────────────────────────────────────

/// Read a sysfs/proc file and parse it as a `u64`, returning `None` on any
/// missing node, permission error, or unparseable content.
pub(crate) fn read_u64(path: &Path) -> Option<u64> {
    read_trimmed(path)?.parse().ok()
}

/// Read a file and trim surrounding whitespace/newline, `None` on any error.
pub(crate) fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// Read a binary sysfs file (e.g. the `gpu_metrics` table) as raw bytes,
/// `None` on any missing node or read error.
pub(crate) fn read_bytes(path: &Path) -> Option<Vec<u8>> {
    fs::read(path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_stamps_time_and_never_panics() {
        // Real host; just assert structural sanity regardless of hardware.
        let s = snapshot(123);
        assert_eq!(s.generated_unix, 123);
        // host may be None on exotic targets, but on Linux CI it should read.
        if let Some(h) = &s.host {
            assert!(h.total_bytes >= h.available_bytes);
        }
    }

    // Re-export plumbing so the GPU module's synthetic-tree tests can build
    // fixtures with the same helpers the collector uses.
    #[test]
    fn read_u64_parses_and_degrades() {
        let dir = std::env::temp_dir().join(format!("hipfire-sysinfo-u64-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("n");
        fs::write(&p, "  42\n").unwrap();
        assert_eq!(read_u64(&p), Some(42));
        assert_eq!(read_u64(&dir.join("missing")), None);
        fs::write(&p, "not-a-number").unwrap();
        assert_eq!(read_u64(&p), None);
        let _ = fs::remove_dir_all(&dir);
    }
}
