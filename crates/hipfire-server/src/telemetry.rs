//! Host GPU telemetry for the `/admin` dashboard.
//!
//! Reads AMD GPU stats from sysfs (`/sys/class/drm/card*/device`) — no root,
//! no rocm-smi spawn. On APUs (e.g. gfx1151) the sysfs `mem_info_vram_total`
//! reports only the dedicated carveout (e.g. 512 MB), not the much larger GTT
//! pool the runtime actually allocates from; surface it as-is and let the UI
//! label it. Each field is independently `Option` so a missing sysfs node
//! degrades gracefully instead of dropping the whole card.

use std::fs;
use std::path::{Path, PathBuf};

use hipfire_admin_types::GpuTelemetry;

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
        temp_c: read_hwmon(device, "temp1_input").map(|milli| milli / 1000.0),
        power_w: read_hwmon(device, "power1_average").map(|micro| micro / 1_000_000.0),
        sclk_mhz: read_active_sclk(&device.join("pp_dpm_sclk")),
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

fn read_u64(path: &Path) -> Option<u64> {
    read_trimmed(path)?.parse().ok()
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

// Re-exported for tests that build synthetic sysfs trees.
#[allow(dead_code)]
pub(crate) fn read_gpu_telemetry_at(root: PathBuf) -> Vec<GpuTelemetry> {
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
    fn reads_synthetic_amd_card_and_skips_non_amd() {
        let tmp = std::env::temp_dir().join(format!("hipfire-telemetry-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);

        // AMD card with full telemetry.
        let amd = tmp.join("card1").join("device");
        fs::create_dir_all(amd.join("hwmon").join("hwmon5")).unwrap();
        fs::write(amd.join("vendor"), "0x1002\n").unwrap();
        fs::write(amd.join("gpu_busy_percent"), "42\n").unwrap();
        fs::write(amd.join("mem_info_vram_used"), "163282944\n").unwrap();
        fs::write(amd.join("mem_info_vram_total"), "536870912\n").unwrap();
        fs::write(
            amd.join("hwmon").join("hwmon5").join("temp1_input"),
            "27000\n",
        )
        .unwrap();
        fs::write(
            amd.join("hwmon").join("hwmon5").join("power1_average"),
            "7052000\n",
        )
        .unwrap();
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
        assert_eq!(g.vram_used_bytes, Some(163_282_944));
        assert_eq!(g.vram_total_bytes, Some(536_870_912));
        assert_eq!(g.temp_c, Some(27.0));
        assert_eq!(g.power_w, Some(7.052));
        assert_eq!(g.sclk_mhz, Some(2200));
    }
}
