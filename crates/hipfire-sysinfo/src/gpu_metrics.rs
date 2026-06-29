//! Parser for the firmware `gpu_metrics` binary table
//! (`/sys/class/drm/cardN/device/gpu_metrics`).
//!
//! The kernel exposes a *versioned* struct (`metrics_table_header` =
//! `u16 structure_size, u8 format_revision, u8 content_revision`) whose layout
//! changes materially between versions — not just appended fields. We read the
//! 4-byte header, dispatch on `(format_revision, content_revision)`, and pull
//! only the fields plain sysfs nodes lack: socket power, soc/gfx die temps,
//! throttle status, and (v3) DRAM bandwidth. Utilization deliberately comes
//! from `gpu_busy_percent` instead, sidestepping the table's centi-percent
//! ambiguity.
//!
//! Field offsets are validated against live hardware: this box (Phoenix
//! gfx1103, smu_v13_0_4) delivers `v2_1` (size 120) and halo (gfx1151,
//! smu_v14) delivers `v3_0`. Units per the kernel headers: temps centi-°C,
//! power milliWatts, bandwidth MB/s. Unknown versions yield `None` rather than
//! guessing offsets.

use std::path::Path;

use hipfire_admin_types::GpuMetrics;

use crate::read_bytes;

/// Read and decode `gpu_metrics` for one card's `device` dir, if present and a
/// recognized version.
pub fn read_gpu_metrics(device: &Path) -> Option<GpuMetrics> {
    let bytes = read_bytes(&device.join("gpu_metrics"))?;
    parse(&bytes)
}

/// u16 little-endian at `off`, or `None` if out of range.
fn u16le(b: &[u8], off: usize) -> Option<u16> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
}

/// u32 little-endian at `off`, or `None` if out of range.
fn u32le(b: &[u8], off: usize) -> Option<u32> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// centi-°C u16 → °C, treating the all-ones sentinel (0xFFFF) as "absent".
fn temp_c(b: &[u8], off: usize) -> Option<f64> {
    match u16le(b, off) {
        Some(0xFFFF) | None => None,
        Some(v) => Some(v as f64 / 100.0),
    }
}

/// milliWatts u16/u32 → W, treating the all-ones sentinel as "absent".
fn power_w_u16(b: &[u8], off: usize) -> Option<f64> {
    match u16le(b, off) {
        Some(0xFFFF) | None => None,
        Some(v) => Some(v as f64 / 1000.0),
    }
}
fn power_w_u32(b: &[u8], off: usize) -> Option<f64> {
    match u32le(b, off) {
        Some(0xFFFF_FFFF) | None => None,
        Some(v) => Some(v as f64 / 1000.0),
    }
}

fn parse(b: &[u8]) -> Option<GpuMetrics> {
    let format_rev = *b.get(2)?;
    let content_rev = *b.get(3)?;
    let mut m = GpuMetrics {
        version: (format_rev, content_rev),
        ..Default::default()
    };
    match (format_rev, content_rev) {
        // v2_0: system_clock_counter sits right after the header, shifting the
        // temperature/utilization block down by 8 bytes vs v2_1.
        (2, 0) => {
            m.gfx_temp_c = temp_c(b, 12);
            m.soc_temp_c = temp_c(b, 14);
            m.socket_power_w = power_w_u16(b, 48);
            m.throttle_status = u32le(b, 116).map(|v| v as u64);
        }
        // v2_1 and the layout-compatible v2_2/2_3/2_4 (which only *append*
        // fields after the shared prefix). Validated on gfx1103: temp_gfx@4,
        // temp_soc@6, socket_power@40, throttle@108.
        (2, _) => {
            m.gfx_temp_c = temp_c(b, 4);
            m.soc_temp_c = temp_c(b, 6);
            m.socket_power_w = power_w_u16(b, 40);
            m.throttle_status = u32le(b, 108).map(|v| v as u64);
        }
        // v3_0 (Strix-class APUs, smu_v14): wider power fields, 16 cores, and
        // first-class DRAM bandwidth. socket_power is u32. Throttle is split
        // into per-reason residency accumulators, so we leave it None here.
        (3, 0) => {
            m.gfx_temp_c = temp_c(b, 4);
            m.soc_temp_c = temp_c(b, 6);
            // temp_core[16]@8 (32B), temp_skin@40, gfx_act@42, vcn_act@44,
            // ipu_act[8]@46 (16B), core_c0[16]@62 (32B), then DRAM @94/@96.
            m.dram_read_mbps = u16le(b, 94).map(|v| v as f64);
            m.dram_write_mbps = u16le(b, 96).map(|v| v as f64);
            // system_clock_counter@102 (8B), average_socket_power u32 @110.
            m.socket_power_w = power_w_u32(b, 110);
        }
        // Unknown/unsupported version: report the version but no fields rather
        // than decode against the wrong layout.
        _ => {}
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal v2_1 buffer with known values at the validated offsets.
    fn v2_1_buf() -> Vec<u8> {
        let mut b = vec![0u8; 120];
        b[0..2].copy_from_slice(&120u16.to_le_bytes()); // structure_size
        b[2] = 2; // format_revision
        b[3] = 1; // content_revision
        b[4..6].copy_from_slice(&3237u16.to_le_bytes()); // temp_gfx → 32.37°C
        b[6..8].copy_from_slice(&3400u16.to_le_bytes()); // temp_soc → 34.0°C
        b[40..42].copy_from_slice(&5671u16.to_le_bytes()); // socket_power → 5.671W
        b[108..112].copy_from_slice(&0u32.to_le_bytes()); // throttle
        b
    }

    #[test]
    fn parses_v2_1_validated_offsets() {
        let m = parse(&v2_1_buf()).unwrap();
        assert_eq!(m.version, (2, 1));
        assert_eq!(m.gfx_temp_c, Some(32.37));
        assert_eq!(m.soc_temp_c, Some(34.0));
        assert_eq!(m.socket_power_w, Some(5.671));
        assert_eq!(m.throttle_status, Some(0));
        assert!(!m.throttling());
    }

    #[test]
    fn throttling_flag_reflects_nonzero_bitmask() {
        let mut b = v2_1_buf();
        b[108..112].copy_from_slice(&0b100u32.to_le_bytes());
        assert!(parse(&b).unwrap().throttling());
    }

    #[test]
    fn v3_0_reads_wide_power_and_dram_bw() {
        let mut b = vec![0u8; 200];
        b[0..2].copy_from_slice(&200u16.to_le_bytes());
        b[2] = 3;
        b[3] = 0;
        b[4..6].copy_from_slice(&3100u16.to_le_bytes()); // temp_gfx
        b[94..96].copy_from_slice(&1200u16.to_le_bytes()); // dram read MB/s
        b[96..98].copy_from_slice(&800u16.to_le_bytes()); // dram write MB/s
        b[110..114].copy_from_slice(&42000u32.to_le_bytes()); // socket power → 42W
        let m = parse(&b).unwrap();
        assert_eq!(m.version, (3, 0));
        assert_eq!(m.gfx_temp_c, Some(31.0));
        assert_eq!(m.dram_read_mbps, Some(1200.0));
        assert_eq!(m.dram_write_mbps, Some(800.0));
        assert_eq!(m.socket_power_w, Some(42.0));
    }

    #[test]
    fn unknown_version_yields_version_only() {
        let mut b = vec![0u8; 64];
        b[2] = 9;
        b[3] = 9;
        let m = parse(&b).unwrap();
        assert_eq!(m.version, (9, 9));
        assert_eq!(m.socket_power_w, None);
    }

    #[test]
    fn sentinel_temps_are_absent() {
        let mut b = v2_1_buf();
        b[4..6].copy_from_slice(&0xFFFFu16.to_le_bytes());
        assert_eq!(parse(&b).unwrap().gfx_temp_c, None);
    }
}
