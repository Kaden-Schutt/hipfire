//! Per-process GPU memory from `/proc/<pid>/fdinfo`.
//!
//! Each open amdgpu DRM file exposes `drm-driver: amdgpu`, a `drm-pdev` PCI
//! address, a `drm-client-id`, and `drm-resident-{vram,gtt}` byte counters
//! (per-VM, so identical across fds sharing a `drm-client-id`). We walk every
//! readable pid, keep the amdgpu clients, dedup by `(pid, client-id)`, attribute
//! to a DRM card via the PCI address, and sort by VRAM then GTT.
//!
//! Engine utilization (`drm-engine-*`) is intentionally skipped: it's a
//! cumulative-nanosecond counter needing two samples to rate, and this kernel
//! doesn't emit it at all. Memory is always present, so that's what we surface.
//! Unreadable pids (permission, race) are skipped silently.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use hipfire_admin_types::ClientUsage;

/// Read per-process GPU memory usage across all readable processes.
pub fn read_clients() -> Vec<ClientUsage> {
    read_clients_from(Path::new("/proc"), &pci_to_card())
}

/// Map PCI address (e.g. "0000:04:00.0") → DRM card name ("card1") by resolving
/// each `/sys/class/drm/cardN/device` symlink to its PCI node.
fn pci_to_card() -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return map;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !is_card_dir(&name) {
            continue;
        }
        if let Ok(target) = fs::read_link(entry.path().join("device")) {
            if let Some(pci) = target.file_name().and_then(|s| s.to_str()) {
                map.insert(pci.to_string(), name);
            }
        }
    }
    map
}

fn is_card_dir(name: &str) -> bool {
    name.strip_prefix("card")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

fn read_clients_from(proc_root: &Path, pci_card: &BTreeMap<String, String>) -> Vec<ClientUsage> {
    let mut clients: Vec<ClientUsage> = Vec::new();
    let Ok(pids) = fs::read_dir(proc_root) else {
        return clients;
    };
    for pid_entry in pids.flatten() {
        let pid_name = pid_entry.file_name().to_string_lossy().to_string();
        let Ok(pid) = pid_name.parse::<u32>() else {
            continue;
        };
        if let Some(usage) = read_pid(&pid_entry.path(), pid, pci_card) {
            clients.push(usage);
        }
    }
    // Most VRAM first, then GTT, so a UI leading list shows the heaviest users.
    clients.sort_by(|a, b| {
        b.vram_bytes
            .cmp(&a.vram_bytes)
            .then(b.gtt_bytes.cmp(&a.gtt_bytes))
    });
    clients
}

/// Aggregate one pid's amdgpu DRM clients, deduping repeated client-ids.
fn read_pid(pid_dir: &Path, pid: u32, pci_card: &BTreeMap<String, String>) -> Option<ClientUsage> {
    let fdinfo_dir = pid_dir.join("fdinfo");
    let entries = fs::read_dir(&fdinfo_dir).ok()?;
    let mut seen_clients: HashSet<u64> = HashSet::new();
    let mut vram: u64 = 0;
    let mut gtt: u64 = 0;
    let mut card: Option<String> = None;
    let mut any = false;

    for entry in entries.flatten() {
        let Some(text) = fs::read_to_string(entry.path()).ok() else {
            continue;
        };
        let f = parse_fdinfo(&text);
        if f.driver.as_deref() != Some("amdgpu") {
            continue;
        }
        // Dedup: same client-id across fds reports identical per-VM memory.
        if let Some(id) = f.client_id {
            if !seen_clients.insert(id) {
                continue;
            }
        }
        any = true;
        vram += f.vram_bytes;
        gtt += f.gtt_bytes;
        if card.is_none() {
            card = f.pdev.and_then(|p| pci_card.get(&p).cloned());
        }
    }

    if !any {
        return None;
    }
    Some(ClientUsage {
        pid,
        comm: fs::read_to_string(pid_dir.join("comm"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
        card,
        vram_bytes: vram,
        gtt_bytes: gtt,
    })
}

#[derive(Default)]
struct Fdinfo {
    driver: Option<String>,
    pdev: Option<String>,
    client_id: Option<u64>,
    vram_bytes: u64,
    gtt_bytes: u64,
}

fn parse_fdinfo(text: &str) -> Fdinfo {
    let mut f = Fdinfo::default();
    for line in text.lines() {
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        let val = val.trim();
        match key.trim() {
            "drm-driver" => f.driver = Some(val.to_string()),
            "drm-pdev" => f.pdev = Some(val.to_string()),
            "drm-client-id" => f.client_id = val.parse().ok(),
            "drm-resident-vram" => f.vram_bytes = parse_mem_bytes(val),
            "drm-resident-gtt" => f.gtt_bytes = parse_mem_bytes(val),
            _ => {}
        }
    }
    f
}

/// Parse a DRM memory value like "6180 KiB" into bytes.
fn parse_mem_bytes(val: &str) -> u64 {
    let mut it = val.split_whitespace();
    let Some(num) = it.next().and_then(|n| n.parse::<u64>().ok()) else {
        return 0;
    };
    let mult = match it.next() {
        Some("KiB") => 1024,
        Some("MiB") => 1024 * 1024,
        Some("GiB") => 1024 * 1024 * 1024,
        _ => 1, // bytes / unknown
    };
    num * mult
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
pos:\t0
drm-driver:\tamdgpu
drm-pdev:\t0000:04:00.0
drm-client-id:\t107
drm-resident-vram:\t44 KiB
drm-resident-gtt:\t6180 KiB
";

    #[test]
    fn parses_amdgpu_fdinfo() {
        let f = parse_fdinfo(SAMPLE);
        assert_eq!(f.driver.as_deref(), Some("amdgpu"));
        assert_eq!(f.pdev.as_deref(), Some("0000:04:00.0"));
        assert_eq!(f.client_id, Some(107));
        assert_eq!(f.vram_bytes, 44 * 1024);
        assert_eq!(f.gtt_bytes, 6180 * 1024);
    }

    #[test]
    fn parse_mem_units() {
        assert_eq!(parse_mem_bytes("0"), 0);
        assert_eq!(parse_mem_bytes("512 KiB"), 512 * 1024);
        assert_eq!(parse_mem_bytes("2 MiB"), 2 * 1024 * 1024);
        assert_eq!(parse_mem_bytes("nonsense"), 0);
    }

    #[test]
    fn dedups_client_id_and_attributes_card() {
        let tmp = std::env::temp_dir().join(format!("hipfire-fdinfo-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let fdinfo = tmp.join("4242").join("fdinfo");
        fs::create_dir_all(&fdinfo).unwrap();
        // Two fds, same client-id → counted once.
        fs::write(fdinfo.join("3"), SAMPLE).unwrap();
        fs::write(fdinfo.join("4"), SAMPLE).unwrap();
        fs::write(tmp.join("4242").join("comm"), "hipfire\n").unwrap();

        let mut pci = BTreeMap::new();
        pci.insert("0000:04:00.0".to_string(), "card1".to_string());
        let clients = read_clients_from(&tmp, &pci);
        let _ = fs::remove_dir_all(&tmp);

        assert_eq!(clients.len(), 1);
        assert_eq!(clients[0].pid, 4242);
        assert_eq!(clients[0].comm, "hipfire");
        assert_eq!(clients[0].card.as_deref(), Some("card1"));
        assert_eq!(clients[0].vram_bytes, 44 * 1024); // not doubled
        assert_eq!(clients[0].gtt_bytes, 6180 * 1024);
    }
}
