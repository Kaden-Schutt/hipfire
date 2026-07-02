//! Host system memory from `/proc/meminfo`.
//!
//! On UMA APUs this is the ground truth for memory pressure: GTT allocations
//! are ordinary system pages and are already reflected in `MemAvailable`, so a
//! caller watching the host figure sees GPU pressure for free. We read
//! `MemTotal` and `MemAvailable` (not `MemFree`, which ignores reclaimable
//! page cache and would overstate "used").

use std::path::Path;

use hipfire_admin_types::HostMemory;

use crate::read_trimmed;

const MEMINFO: &str = "/proc/meminfo";

/// Read host system memory, or `None` when `/proc/meminfo` is absent
/// (non-Linux target) or lacks the required keys.
pub fn read_host_memory() -> Option<HostMemory> {
    parse_meminfo(&read_trimmed(Path::new(MEMINFO))?)
}

/// Parse the relevant `/proc/meminfo` keys. Values are in kibibytes per the
/// kernel format (`MemTotal:    47800200 kB`); we widen to bytes.
fn parse_meminfo(text: &str) -> Option<HostMemory> {
    let mut total_kib: Option<u64> = None;
    let mut avail_kib: Option<u64> = None;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest.split_whitespace().next().and_then(|v| v.parse().ok());
        match key.trim() {
            "MemTotal" => total_kib = value,
            "MemAvailable" => avail_kib = value,
            _ => {}
        }
        if total_kib.is_some() && avail_kib.is_some() {
            break;
        }
    }
    let total = total_kib?;
    // MemAvailable predates nothing modern but guard anyway: clamp to total.
    let available = avail_kib?.min(total);
    Some(HostMemory {
        total_bytes: total * 1024,
        available_bytes: available * 1024,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_meminfo_shape() {
        let sample = "\
MemTotal:       47800200 kB
MemFree:        38937000 kB
MemAvailable:   46892800 kB
Buffers:            3680 kB
Cached:          8310000 kB";
        let h = parse_meminfo(sample).unwrap();
        assert_eq!(h.total_bytes, 47_800_200 * 1024);
        assert_eq!(h.available_bytes, 46_892_800 * 1024);
        // Used derives from available, not free.
        assert_eq!(h.used_bytes(), (47_800_200 - 46_892_800) * 1024);
    }

    #[test]
    fn missing_available_key_yields_none() {
        let sample = "MemTotal:       47800200 kB\nMemFree: 100 kB";
        assert!(parse_meminfo(sample).is_none());
    }

    #[test]
    fn available_clamped_to_total() {
        let sample = "MemTotal: 100 kB\nMemAvailable: 999 kB";
        let h = parse_meminfo(sample).unwrap();
        assert_eq!(h.available_bytes, 100 * 1024);
        assert_eq!(h.used_bytes(), 0);
    }
}
