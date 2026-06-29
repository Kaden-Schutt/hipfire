//! Per-GPU telemetry from `/sys/class/drm/card*/device`.
//!
//! Portable by construction: scans every DRM card and keeps the AMD ones
//! (vendor `0x1002`) rather than assuming `card1`, and reads both VRAM and the
//! GTT window so callers can pick the OOM-governing pool per device class. The
//! `integrated` flag is left `None` here — this crate is intentionally
//! HIP-free, so the APU/dGPU call is made by the carveout-vs-GTT heuristic in
//! [`hipfire_admin_types::GpuTelemetry::is_integrated`], which a HIP-aware
//! caller can override with the authoritative `hipDeviceAttributeIntegrated`.

use std::fs;
use std::path::Path;

use hipfire_admin_types::GpuTelemetry;

use crate::{read_trimmed, read_u64};

const DRM_ROOT: &str = "/sys/class/drm";
const AMD_VENDOR_ID: &str = "0x1002";

/// Read telemetry for every AMD GPU visible under `/sys/class/drm`.
pub fn read_gpu_telemetry() -> Vec<GpuTelemetry> {
    read_gpu_telemetry_from(Path::new(DRM_ROOT))
}

fn read_gpu_telemetry_from(drm_root: &Path) -> Vec<GpuTelemetry> {
    let Ok(entries) = fs::read_dir(drm_root) else {
        return Vec::new();
    };
    let mut cards: Vec<GpuTelemetry> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            // Match cardN but not cardN-DP-1 (connector subdirs).
            if !is_card_dir(&name) {
                return None;
            }
            let device = entry.path().join("device");
            if !is_amd(&device) {
                return None;
            }
            Some(read_card(&name, &device))
        })
        .collect();
    cards.sort_by(|a, b| a.card.cmp(&b.card));
    cards
}

fn is_card_dir(name: &str) -> bool {
    name.strip_prefix("card")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

fn is_amd(device: &Path) -> bool {
    read_trimmed(&device.join("vendor")).as_deref() == Some(AMD_VENDOR_ID)
}

fn read_card(card: &str, device: &Path) -> GpuTelemetry {
    GpuTelemetry {
        card: card.to_string(),
        busy_percent: read_u64(&device.join("gpu_busy_percent")).map(|v| v.min(100) as u32),
        vram_used_bytes: read_u64(&device.join("mem_info_vram_used")),
        vram_total_bytes: read_u64(&device.join("mem_info_vram_total")),
        vis_vram_used_bytes: read_u64(&device.join("mem_info_vis_vram_used")),
        vis_vram_total_bytes: read_u64(&device.join("mem_info_vis_vram_total")),
        gtt_used_bytes: read_u64(&device.join("mem_info_gtt_used")),
        gtt_total_bytes: read_u64(&device.join("mem_info_gtt_total")),
        // HIP-free collector: device class is inferred downstream from the
        // VRAM-carveout-vs-GTT ratio, not asserted here.
        integrated: None,
        temp_c: read_hwmon(device, "temp1_input").map(|milli| milli / 1000.0),
        power_w: read_hwmon(device, "power1_average").map(|micro| micro / 1_000_000.0),
        sclk_mhz: read_active_sclk(&device.join("pp_dpm_sclk")),
        metrics: crate::read_gpu_metrics(device),
    }
}

/// hwmon nodes live under `device/hwmon/hwmon*/`; scan for the first match.
fn read_hwmon(device: &Path, leaf: &str) -> Option<f64> {
    let hwmon_root = device.join("hwmon");
    let entries = fs::read_dir(&hwmon_root).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join(leaf);
        if let Some(value) = read_u64(&candidate) {
            return Some(value as f64);
        }
    }
    None
}

/// `pp_dpm_sclk` lines look like `0: 800Mhz` with the active level marked `*`.
fn read_active_sclk(path: &Path) -> Option<u64> {
    let text = read_trimmed(path)?;
    text.lines()
        .find(|line| line.contains('*'))
        .and_then(parse_sclk_mhz)
}

fn parse_sclk_mhz(line: &str) -> Option<u64> {
    let lower = line.to_ascii_lowercase();
    let mhz_pos = lower.find("mhz")?;
    let digits: String = lower[..mhz_pos]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
pub(crate) fn read_gpu_telemetry_at(root: std::path::PathBuf) -> Vec<GpuTelemetry> {
    read_gpu_telemetry_from(&root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_active_sclk_line() {
        assert_eq!(parse_sclk_mhz("1: 2200Mhz *"), Some(2200));
        assert_eq!(parse_sclk_mhz("0: 800Mhz"), Some(800));
        assert_eq!(parse_sclk_mhz("no clock here"), None);
    }

    #[test]
    fn card_dir_matching_ignores_connectors() {
        assert!(is_card_dir("card0"));
        assert!(is_card_dir("card12"));
        assert!(!is_card_dir("card1-DP-1"));
        assert!(!is_card_dir("renderD128"));
        assert!(!is_card_dir("card"));
    }

    #[test]
    fn reads_synthetic_amd_card_with_gtt_and_skips_non_amd() {
        let tmp = std::env::temp_dir().join(format!("hipfire-sysinfo-gpu-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);

        // AMD APU-shaped card: tiny VRAM carveout, large GTT window.
        let amd = tmp.join("card1").join("device");
        fs::create_dir_all(amd.join("hwmon").join("hwmon5")).unwrap();
        fs::write(amd.join("vendor"), "0x1002\n").unwrap();
        fs::write(amd.join("gpu_busy_percent"), "42\n").unwrap();
        fs::write(amd.join("mem_info_vram_used"), "91074560\n").unwrap();
        fs::write(amd.join("mem_info_vram_total"), "268435456\n").unwrap();
        fs::write(amd.join("mem_info_vis_vram_used"), "91074560\n").unwrap();
        fs::write(amd.join("mem_info_vis_vram_total"), "268435456\n").unwrap();
        fs::write(amd.join("mem_info_gtt_used"), "14184448\n").unwrap();
        fs::write(amd.join("mem_info_gtt_total"), "45097156608\n").unwrap();
        fs::write(amd.join("hwmon/hwmon5/temp1_input"), "27000\n").unwrap();
        fs::write(amd.join("hwmon/hwmon5/power1_average"), "7052000\n").unwrap();
        fs::write(amd.join("pp_dpm_sclk"), "0: 800Mhz\n1: 2200Mhz *\n").unwrap();

        // Non-AMD card should be skipped.
        let other = tmp.join("card0").join("device");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("vendor"), "0x8086\n").unwrap();

        let cards = read_gpu_telemetry_at(tmp.clone());
        let _ = fs::remove_dir_all(&tmp);

        assert_eq!(cards.len(), 1);
        let g = &cards[0];
        assert_eq!(g.card, "card1");
        assert_eq!(g.busy_percent, Some(42));
        assert_eq!(g.vram_used_bytes, Some(91_074_560));
        assert_eq!(g.vram_total_bytes, Some(268_435_456));
        assert_eq!(g.vis_vram_total_bytes, Some(268_435_456));
        assert_eq!(g.gtt_used_bytes, Some(14_184_448));
        assert_eq!(g.gtt_total_bytes, Some(45_097_156_608));
        assert_eq!(g.temp_c, Some(27.0));
        assert_eq!(g.power_w, Some(7.052));
        assert_eq!(g.sclk_mhz, Some(2200));
        // Derived classification: carveout ≪ GTT ⇒ integrated, GTT is primary.
        assert!(g.is_integrated());
        assert_eq!(g.primary_pool().unwrap().label, "GTT");
    }

    #[test]
    fn missing_gtt_nodes_degrade_to_none() {
        let tmp =
            std::env::temp_dir().join(format!("hipfire-sysinfo-nogtt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let amd = tmp.join("card1").join("device");
        fs::create_dir_all(&amd).unwrap();
        fs::write(amd.join("vendor"), "0x1002\n").unwrap();
        fs::write(amd.join("mem_info_vram_used"), "100\n").unwrap();
        fs::write(amd.join("mem_info_vram_total"), "200\n").unwrap();
        let cards = read_gpu_telemetry_at(tmp.clone());
        let _ = fs::remove_dir_all(&tmp);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].gtt_total_bytes, None);
        // No GTT info ⇒ heuristic can't prove integrated ⇒ VRAM is primary.
        assert!(!cards[0].is_integrated());
        assert_eq!(cards[0].primary_pool().unwrap().label, "VRAM");
    }
}
