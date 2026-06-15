// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Evidence provenance helpers shared by Hipfire eval and gate tooling.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfqMetadata {
    pub arch_id: u32,
    pub metadata_json: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceArtifactSpec {
    pub file: &'static str,
    pub kind: &'static str,
    pub expected_metrics: &'static [&'static str],
}

pub const STANDARD_EVIDENCE_ARTIFACT_SPECS: &[EvidenceArtifactSpec] = &[
    EvidenceArtifactSpec {
        file: "quality.json",
        kind: "quality",
        expected_metrics: &[
            "mean_kld",
            "p99_kld",
            "ppl",
            "argmax_match_rate",
            "accuracy",
            "exact_match",
        ],
    },
    EvidenceArtifactSpec {
        file: "performance.json",
        kind: "performance",
        expected_metrics: &["pp32_ms", "pp128_ms", "ttft_ms", "tok_s"],
    },
    EvidenceArtifactSpec {
        file: "phase_timings.json",
        kind: "phase_timings",
        expected_metrics: &["load_ms", "prefill_ms", "decode_ms", "teardown_ms"],
    },
    EvidenceArtifactSpec {
        file: "launch_counts.json",
        kind: "launch_counts",
        expected_metrics: &["kernel_launches", "graph_launches", "memcpy_ops"],
    },
    EvidenceArtifactSpec {
        file: "moe_router_histogram.json",
        kind: "moe_router_histogram",
        expected_metrics: &["expert_hits", "shared_expert_hits", "router_entropy"],
    },
    EvidenceArtifactSpec {
        file: "memory.json",
        kind: "memory",
        expected_metrics: &["vram_peak_bytes", "kv_bytes", "workspace_bytes"],
    },
    EvidenceArtifactSpec {
        file: "dflash_trace.json",
        kind: "dflash_trace",
        expected_metrics: &["ar_tok_s", "dflash_tok_s", "accept_rate", "tau"],
    },
    EvidenceArtifactSpec {
        file: "path_c_trace.json",
        kind: "path_c_trace",
        expected_metrics: &[
            "tok_s",
            "tau",
            "promotion_verdict",
            "tok_s_delta_pct",
            "tau_delta_pct",
        ],
    },
    EvidenceArtifactSpec {
        file: "module_evidence.json",
        kind: "module_evidence",
        expected_metrics: &[
            "module_kind",
            "module_id",
            "preferred_backend",
            "selected_backend",
            "oracle_backend",
            "fallback_reason",
        ],
    },
    EvidenceArtifactSpec {
        file: "profiling.json",
        kind: "profiling",
        expected_metrics: &["kernel_name", "duration_us", "occupancy", "waves"],
    },
    EvidenceArtifactSpec {
        file: "coherence.json",
        kind: "coherence",
        expected_metrics: &[
            "hard_fails",
            "soft_warns",
            "detector_count",
            "coherence_status",
        ],
    },
];

pub fn file_hash(path: &Path) -> Option<String> {
    command_digest("sha256sum", path).or_else(|| Some(stable_hash_file_fallback(path)))
}

pub fn model_hash(model: &str) -> Option<String> {
    static CACHE: OnceLock<Mutex<BTreeMap<String, Option<String>>>> = OnceLock::new();
    let key = model_hash_cache_key(model);
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(cache) = cache.lock() {
        if let Some(hash) = cache.get(&key) {
            return hash.clone();
        }
    }
    let hash = model_hash_uncached(model);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, hash.clone());
    }
    hash
}

pub fn stable_hash_file_fallback(path: &Path) -> String {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return "unavailable".to_string(),
    };
    let mut state = Fnv64::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => state.update(&buf[..n]),
            Err(_) => return "unavailable".to_string(),
        }
    }
    format!("fnv64:{:016x}", state.finish())
}

pub fn stable_hash_bytes(bytes: &[u8]) -> String {
    let mut state = Fnv64::new();
    state.update(bytes);
    format!("fnv64:{:016x}", state.finish())
}

pub fn stable_score(input: &str) -> f64 {
    let mut state = Fnv64::new();
    state.update(input.as_bytes());
    (state.finish() as f64) / (u64::MAX as f64)
}

pub fn directory_hash(path: &Path) -> Option<String> {
    let files = list_files(path);
    if files.is_empty() {
        return None;
    }
    Some(stable_hash_bytes(files.join("\n").as_bytes()))
}

pub fn read_hfq_metadata(path: &Path) -> Result<HfqMetadata, String> {
    let mut f = File::open(path).map_err(|e| format!("open model: {e}"))?;
    let mut header = [0u8; 32];
    f.read_exact(&mut header)
        .map_err(|e| format!("read HFQ header: {e}"))?;
    if &header[0..4] != b"HFQM" {
        return Err("not an HFQ container".to_string());
    }
    let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
    let span_len = data_offset.saturating_sub(metadata_offset);
    if span_len == 0 || span_len > 256 * 1024 * 1024 {
        return Err(format!(
            "invalid or too-large metadata span: {metadata_offset}..{data_offset}"
        ));
    }
    f.seek(SeekFrom::Start(metadata_offset as u64))
        .map_err(|e| format!("seek HFQ metadata span: {e}"))?;
    let mut span = vec![0u8; span_len];
    f.read_exact(&mut span)
        .map_err(|e| format!("read HFQ metadata span: {e}"))?;
    let json_end = find_json_object_end(&span)
        .ok_or_else(|| "HFQ metadata JSON object was not terminated".to_string())?;
    let metadata_json = String::from_utf8(span[..json_end].to_vec())
        .map_err(|e| format!("HFQ metadata is not UTF-8: {e}"))?;
    Ok(HfqMetadata {
        arch_id,
        metadata_json,
    })
}

fn model_hash_uncached(model: &str) -> Option<String> {
    let p = Path::new(model);
    if p.exists() {
        file_hash(p)
    } else {
        Some(format!("tag:{}", stable_hash_bytes(model.as_bytes())))
    }
}

fn model_hash_cache_key(model: &str) -> String {
    let p = Path::new(model);
    if let Ok(meta) = fs::metadata(p) {
        let modified = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let canonical = p
            .canonicalize()
            .unwrap_or_else(|_| p.to_path_buf())
            .display()
            .to_string();
        format!("file:{canonical}:{}:{modified}", meta.len())
    } else {
        format!("tag:{model}")
    }
}

fn command_digest(tool: &str, path: &Path) -> Option<String> {
    let out = Command::new(tool).arg(path).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
}

fn find_json_object_end(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

pub fn list_files(path: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect_files_relative(path, path, &mut out);
    out.sort();
    out
}

fn collect_files_relative(root: &Path, path: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_files_relative(root, &p, out);
        } else if p.is_file() {
            if let Ok(rel) = p.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn stable_hash_bytes_is_deterministic() {
        assert_eq!(stable_hash_bytes(b"abc"), "fnv64:e71fa2190541574b");
        assert_eq!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abc"));
        assert_ne!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abcd"));
    }

    #[test]
    fn model_hash_tags_non_files() {
        assert_eq!(
            model_hash("qwen3.5:9b").unwrap(),
            format!("tag:{}", stable_hash_bytes(b"qwen3.5:9b"))
        );
    }

    #[test]
    fn directory_hash_uses_relative_file_list() {
        let root = temp_dir("hipfire-evidence-dir-hash");
        fs::create_dir_all(root.join("nested")).unwrap();
        File::create(root.join("a.txt")).unwrap();
        File::create(root.join("nested/b.txt")).unwrap();
        assert_eq!(
            directory_hash(&root).unwrap(),
            stable_hash_bytes(b"a.txt\nnested/b.txt")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn standard_artifact_catalog_preserves_schema_names() {
        let specs: Vec<_> = STANDARD_EVIDENCE_ARTIFACT_SPECS
            .iter()
            .map(|spec| (spec.file, spec.kind))
            .collect();
        assert_eq!(
            specs,
            vec![
                ("quality.json", "quality"),
                ("performance.json", "performance"),
                ("phase_timings.json", "phase_timings"),
                ("launch_counts.json", "launch_counts"),
                ("moe_router_histogram.json", "moe_router_histogram"),
                ("memory.json", "memory"),
                ("dflash_trace.json", "dflash_trace"),
                ("path_c_trace.json", "path_c_trace"),
                ("module_evidence.json", "module_evidence"),
                ("profiling.json", "profiling"),
                ("coherence.json", "coherence"),
            ]
        );
        assert!(STANDARD_EVIDENCE_ARTIFACT_SPECS
            .iter()
            .all(|spec| !spec.expected_metrics.is_empty()));
    }

    #[test]
    fn read_hfq_metadata_extracts_json_span() {
        let root = temp_dir("hipfire-evidence-hfq");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("test.hfq");
        let metadata = br#"{"quantization_hash":{"kind":"test"}}"#;
        let metadata_offset = 32u64;
        let data_offset = metadata_offset + metadata.len() as u64 + 4;
        let mut header = [0u8; 32];
        header[0..4].copy_from_slice(b"HFQM");
        header[8..12].copy_from_slice(&42u32.to_le_bytes());
        header[16..24].copy_from_slice(&metadata_offset.to_le_bytes());
        header[24..32].copy_from_slice(&data_offset.to_le_bytes());
        let mut f = File::create(&path).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(metadata).unwrap();
        f.write_all(b"xxxx").unwrap();
        drop(f);

        let got = read_hfq_metadata(&path).unwrap();
        assert_eq!(got.arch_id, 42);
        assert_eq!(got.metadata_json, String::from_utf8_lossy(metadata));
        let _ = fs::remove_dir_all(root);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let unique = format!(
            "{}-{}",
            name,
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }
}
