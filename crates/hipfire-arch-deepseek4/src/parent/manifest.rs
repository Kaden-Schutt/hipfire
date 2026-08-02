// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mandatory evidence manifest for parent-logit / Hessian bundles.
//!
//! A previous 554-tensor Hessian capture had to be rejected because nothing
//! recorded which model produced it. This module makes that failure
//! structurally impossible: every artifact ships with byte-level provenance
//! (`schema = hipfire.ds4.parent.manifest/1`).
//!
//! SHA-256 is implemented in-module (FIPS 180-4). The workspace has `sha2` in
//! sibling crates (`hipfire-cli`, `radiowave`, `redline-dispatch`), but this
//! crate does not depend on it and the slice forbids editing `Cargo.toml`.
//! Hand-rolling keeps the fail-closed path dependency-free and matches the
//! in-tree fixture hasher in `hipfire-runtime/examples/ds4_prompt_fixture.rs`.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Wire schema version. Bump only with a coordinated consumer change.
pub const MANIFEST_SCHEMA: &str = "hipfire.ds4.parent.manifest/1";

/// Top-level evidence manifest. Field names and nesting match
/// `local://ds4-parent-contract.md` §6 exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParentManifest {
    pub schema: String,
    pub produced_utc: String,
    pub producer: ProducerInfo,
    pub engine: EngineInfo,
    pub source: SourceInfo,
    pub model: ModelInfo,
    /// `None` for producers that consume no corpus at all — an inventory or
    /// codec gate, for example. It is NOT a way to skip provenance: a
    /// manifest with no corpus may not carry outputs (see [`Self::validate`]),
    /// so nothing corpus-derived can ship unpinned.
    pub corpus: Option<CorpusInfo>,
    pub capture: CaptureInfo,
    pub outputs: Vec<OutputInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerInfo {
    pub binary: String,
    pub binary_sha256: String,
    pub argv: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    pub commit: String,
    /// `null` on the wire when the working tree is clean.
    pub dirty_diff_sha256: Option<String>,
    pub rocm_path: String,
    pub rocm_version: String,
    pub gpu_arch: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceInfo {
    pub root: String,
    pub index_sha256: String,
    pub shards: Vec<ShardInfo>,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardInfo {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub mtp_loaded: bool,
    pub rope_convention: String,
    pub quant: ModelQuantInfo,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelQuantInfo {
    pub quant_method: String,
    pub fmt: String,
    pub scale_fmt: String,
    pub expert_dtype: String,
    pub weight_block_size: [usize; 2],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusInfo {
    pub token_ids_sha256: String,
    pub n_tokens: usize,
    pub description: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaptureInfo {
    pub boundary: CaptureBoundary,
    pub tensors: Vec<CaptureTensor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureTensor {
    pub name: String,
    pub rows: usize,
    pub k: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputInfo {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub kind: OutputKind,
}

/// Activation capture boundary. Required even when `tensors` is empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureBoundary {
    PreQuant,
    PostDynamicFp8,
}

/// Kind of a produced artifact referenced from `outputs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    Logits,
    Hessian,
    Acts,
}

impl ParentManifest {
    /// Collect everything the process can determine on its own: producer
    /// binary path + hash + argv, engine git commit + dirty-diff hash, ROCm
    /// path/version, GPU arch. Fails if any of these cannot be determined —
    /// an unprovenanced manifest is worse than no manifest.
    pub fn probe_environment(gpu_arch: &str) -> Result<(ProducerInfo, EngineInfo), String> {
        if gpu_arch.trim().is_empty() {
            return Err("deepseek4 parent: gpu_arch must be non-empty".into());
        }

        let exe = std::env::current_exe().map_err(|e| {
            format!("deepseek4 parent: cannot resolve current executable: {e}")
        })?;
        let binary = exe
            .to_str()
            .ok_or_else(|| {
                "deepseek4 parent: current executable path is not valid UTF-8".to_string()
            })?
            .to_string();
        let binary_sha256 = sha256_file(&exe)?;
        let argv: Vec<String> = std::env::args().collect();
        if argv.is_empty() {
            return Err("deepseek4 parent: process argv is empty".into());
        }

        let commit = git_stdout(&["rev-parse", "HEAD"])?.trim().to_string();
        if commit.is_empty() {
            return Err("deepseek4 parent: git rev-parse HEAD returned empty commit".into());
        }

        let diff_bytes = git_stdout_bytes(&["diff", "HEAD"])?;
        let dirty_diff_sha256 = if diff_bytes.is_empty() {
            None
        } else {
            Some(sha256_bytes(&diff_bytes))
        };

        let (rocm_path, rocm_version) = probe_rocm()?;

        Ok((
            ProducerInfo {
                binary,
                binary_sha256,
                argv,
            },
            EngineInfo {
                commit,
                dirty_diff_sha256,
                rocm_path,
                rocm_version,
                gpu_arch: gpu_arch.to_string(),
            },
        ))
    }

    pub fn write_to(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            format!("deepseek4 parent: failed to serialize manifest: {e}")
        })?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "deepseek4 parent: failed to create manifest directory {}: {e}",
                        parent.display()
                    )
                })?;
            }
        }
        let mut f = File::create(path).map_err(|e| {
            format!(
                "deepseek4 parent: failed to create manifest {}: {e}",
                path.display()
            )
        })?;
        f.write_all(json.as_bytes()).map_err(|e| {
            format!(
                "deepseek4 parent: failed to write manifest {}: {e}",
                path.display()
            )
        })?;
        f.write_all(b"\n").map_err(|e| {
            format!(
                "deepseek4 parent: failed to write trailing newline on {}: {e}",
                path.display()
            )
        })?;
        Ok(())
    }

    pub fn read_from(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| {
            format!(
                "deepseek4 parent: failed to read manifest {}: {e}",
                path.display()
            )
        })?;
        let m: Self = serde_json::from_slice(&bytes).map_err(|e| {
            format!(
                "deepseek4 parent: failed to parse manifest {}: {e}",
                path.display()
            )
        })?;
        Ok(m)
    }

    /// Reject a manifest that cannot back a GPTQ or quality claim.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MANIFEST_SCHEMA {
            return Err(format!(
                "deepseek4 parent: unknown or missing schema version (got {:?}, expected {MANIFEST_SCHEMA})",
                self.schema
            ));
        }
        if self.engine.commit.trim().is_empty() {
            return Err("deepseek4 parent: engine.commit is empty".into());
        }
        if self.source.shards.is_empty() {
            return Err("deepseek4 parent: source.shards is empty".into());
        }
        for (i, shard) in self.source.shards.iter().enumerate() {
            if shard.sha256.trim().is_empty() {
                return Err(format!(
                    "deepseek4 parent: source.shards[{i}].sha256 is empty"
                ));
            }
        }
        match &self.corpus {
            Some(corpus) => {
                if corpus.n_tokens == 0 {
                    return Err("deepseek4 parent: corpus.n_tokens is zero".into());
                }
                if corpus.token_ids_sha256.trim().is_empty() {
                    return Err("deepseek4 parent: corpus.token_ids_sha256 is empty".into());
                }
            }
            // A corpus-free run is legitimate only when it produced nothing
            // corpus-derived. Logits, Hessians, and activations are all
            // functions of the tokens that drove them; shipping one without
            // naming its corpus is precisely the provenance failure that got
            // the previous 554-tensor capture rejected.
            None => {
                if !self.outputs.is_empty() {
                    return Err(format!(
                        "deepseek4 parent: corpus is null but {} output(s) are declared — \
                         a corpus-derived artifact must pin the corpus that produced it",
                        self.outputs.len()
                    ));
                }
                if !self.capture.tensors.is_empty() {
                    return Err(
                        "deepseek4 parent: corpus is null but capture.tensors is non-empty — \
                         captured activations must pin the corpus that produced them"
                            .into(),
                    );
                }
            }
        }
        for (i, t) in self.capture.tensors.iter().enumerate() {
            if t.rows == 0 {
                return Err(format!(
                    "deepseek4 parent: capture.tensors[{i}].rows is zero"
                ));
            }
            if t.k == 0 {
                return Err(format!(
                    "deepseek4 parent: capture.tensors[{i}].k is zero"
                ));
            }
        }
        for (i, out) in self.outputs.iter().enumerate() {
            if out.sha256.trim().is_empty() {
                return Err(format!(
                    "deepseek4 parent: outputs[{i}].sha256 is empty"
                ));
            }
            if out.bytes == 0 {
                return Err(format!(
                    "deepseek4 parent: outputs[{i}].bytes is zero"
                ));
            }
        }
        Ok(())
    }
}

/// Streaming SHA-256 over a file, for shard and output hashing.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| {
        format!(
            "deepseek4 parent: failed to open {} for sha256: {e}",
            path.display()
        )
    })?;
    let mut hasher = Sha256Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| {
            format!(
                "deepseek4 parent: failed to read {} for sha256: {e}",
                path.display()
            )
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex32(hasher.finalize()))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256Hasher::new();
    hasher.update(bytes);
    hex32(hasher.finalize())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn git_stdout(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("deepseek4 parent: failed to spawn git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "deepseek4 parent: git {} failed (status {}): {}",
            args.join(" "),
            out.status,
            stderr.trim()
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| {
        format!(
            "deepseek4 parent: git {} produced non-UTF-8 stdout: {e}",
            args.join(" ")
        )
    })
}

fn git_stdout_bytes(args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("deepseek4 parent: failed to spawn git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "deepseek4 parent: git {} failed (status {}): {}",
            args.join(" "),
            out.status,
            stderr.trim()
        ));
    }
    Ok(out.stdout)
}

/// Resolve ROCm install root + version. Fail closed — never invent a version.
fn probe_rocm() -> Result<(String, String), String> {
    let root = resolve_rocm_root().ok_or_else(|| {
        "deepseek4 parent: cannot determine ROCm path (set ROCM_PATH / HIPFIRE_ROCM_PATH or install under /opt/rocm)".to_string()
    })?;
    let version = read_rocm_version(&root).ok_or_else(|| {
        format!(
            "deepseek4 parent: cannot determine ROCm version under {} (missing .info/version and hipcc --version)",
            root.display()
        )
    })?;
    let path = root
        .to_str()
        .ok_or_else(|| "deepseek4 parent: ROCm path is not valid UTF-8".to_string())?
        .to_string();
    Ok((path, version))
}

fn resolve_rocm_root() -> Option<PathBuf> {
    // Prefer the workspace's shared resolver when it finds a root.
    if let Some(p) = hipfire_config::rocm::root() {
        if p.is_dir() {
            return Some(canonicalize_or_self(p));
        }
    }
    for var in ["HIPFIRE_ROCM_PATH", "ROCM_PATH", "HIP_PATH"] {
        if let Ok(v) = std::env::var(var) {
            let p = PathBuf::from(v.trim());
            if p.is_dir() {
                return Some(canonicalize_or_self(p));
            }
        }
    }
    for candidate in [
        "/opt/rocm/core",
        "/opt/rocm/core-7.14",
        "/opt/rocm",
    ] {
        let p = PathBuf::from(candidate);
        if p.is_dir() && p.join(".info").join("version").is_file() {
            return Some(canonicalize_or_self(p));
        }
    }
    None
}

fn canonicalize_or_self(p: PathBuf) -> PathBuf {
    fs::canonicalize(&p).unwrap_or(p)
}

fn read_rocm_version(root: &Path) -> Option<String> {
    let version_file = root.join(".info").join("version");
    if let Ok(s) = fs::read_to_string(&version_file) {
        let s = s.trim();
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    // Fallback: parse `HIP version: X.Y.Z...` from hipcc --version under the root.
    let hipcc = root.join("bin").join("hipcc");
    if hipcc.is_file() {
        if let Ok(out) = Command::new(&hipcc).arg("--version").output() {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Some(v) = parse_hipcc_version(&text) {
                    return Some(v);
                }
            }
        }
    }
    // Last resort: bare hipcc on PATH (still not a hardcoded version string).
    if let Ok(out) = Command::new("hipcc").arg("--version").output() {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(v) = parse_hipcc_version(&text) {
                return Some(v);
            }
        }
    }
    None
}

fn parse_hipcc_version(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("HIP version:") {
            let v = rest.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn hex32(digest: [u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// FIPS 180-4 SHA-256 (hand-rolled — see module docs for why)
// ---------------------------------------------------------------------------

struct Sha256Hasher {
    h: [u32; 8],
    /// Bytes buffered awaiting a full 64-byte block.
    buf: [u8; 64],
    buf_len: usize,
    /// Total message length in bytes (mod 2^64 is fine for our use).
    total_len: u64,
}

impl Sha256Hasher {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.saturating_mul(8);
        // Padding: 0x80, zeros, then 64-bit big-endian length.
        let mut pad = [0u8; 64 + 8];
        pad[0] = 0x80;
        let rem = self.buf_len;
        // Bytes needed so that (total_len + 1 + zeros) ≡ 56 (mod 64).
        let zeros = if rem < 56 {
            56 - rem - 1
        } else {
            56 + 64 - rem - 1
        };
        let pad_len = 1 + zeros + 8;
        pad[1 + zeros..pad_len].copy_from_slice(&bit_len.to_be_bytes());
        self.update(&pad[..pad_len]);
        debug_assert_eq!(self.buf_len, 0);

        let mut out = [0u8; 32];
        for (i, v) in self.h.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut hh = self.h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(hh);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_manifest() -> ParentManifest {
        ParentManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            produced_utc: "2026-08-01T12:00:00Z".to_string(),
            producer: ProducerInfo {
                binary: "/tmp/producer".into(),
                binary_sha256: "aa".repeat(32),
                argv: vec!["producer".into(), "--out".into(), "x.plog".into()],
            },
            engine: EngineInfo {
                commit: "deadbeef".into(),
                dirty_diff_sha256: None,
                rocm_path: "/opt/rocm/core".into(),
                rocm_version: "7.14.0".into(),
                gpu_arch: "gfx942".into(),
            },
            source: SourceInfo {
                root: "/mnt/scratch/models/DeepSeek-V4-Flash-0731".into(),
                index_sha256: "bb".repeat(32),
                shards: vec![ShardInfo {
                    file: "model-00001-of-00048.safetensors".into(),
                    sha256: "cc".repeat(32),
                    bytes: 1024,
                }],
                config_sha256: "dd".repeat(32),
                tokenizer_sha256: "ee".repeat(32),
            },
            model: ModelInfo {
                model_type: "deepseek_v4".into(),
                num_hidden_layers: 43,
                mtp_loaded: false,
                rope_convention: "yarn".into(),
                quant: ModelQuantInfo {
                    quant_method: "fp8".into(),
                    fmt: "e4m3".into(),
                    scale_fmt: "ue8m0".into(),
                    expert_dtype: "fp4".into(),
                    weight_block_size: [128, 128],
                },
            },
            corpus: Some(CorpusInfo {
                token_ids_sha256: "ff".repeat(32),
                n_tokens: 1024,
                description: "wikitext-1024".into(),
            }),
            capture: CaptureInfo {
                boundary: CaptureBoundary::PostDynamicFp8,
                tensors: vec![CaptureTensor {
                    name: "layers.0.attn.wq_a.weight".into(),
                    rows: 16,
                    k: 4096,
                }],
            },
            outputs: vec![OutputInfo {
                path: "parent.plog".into(),
                sha256: "11".repeat(32),
                bytes: 4096,
                kind: OutputKind::Logits,
            }],
        }
    }

    #[test]
    fn round_trip_preserves_contract_key_names() {
        let dir = std::env::temp_dir().join(format!(
            "ds4-parent-manifest-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");

        let original = sample_manifest();
        original.write_to(&path).expect("write_to");
        let loaded = ParentManifest::read_from(&path).expect("read_from");
        assert_eq!(loaded, original);

        // Assert against a literal expected-key list, not the struct fields,
        // so a serde rename silently diverging from §6 breaks the test.
        let raw: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let obj = raw.as_object().expect("top-level object");

        let top_keys: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
        assert_eq!(
            top_keys,
            vec![
                "schema",
                "produced_utc",
                "producer",
                "engine",
                "source",
                "model",
                "corpus",
                "capture",
                "outputs",
            ]
        );

        let producer = obj["producer"].as_object().unwrap();
        assert_eq!(
            producer.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["binary", "binary_sha256", "argv"]
        );

        let engine = obj["engine"].as_object().unwrap();
        assert_eq!(
            engine.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec![
                "commit",
                "dirty_diff_sha256",
                "rocm_path",
                "rocm_version",
                "gpu_arch",
            ]
        );

        let source = obj["source"].as_object().unwrap();
        assert_eq!(
            source.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec![
                "root",
                "index_sha256",
                "shards",
                "config_sha256",
                "tokenizer_sha256",
            ]
        );

        let shard = source["shards"][0].as_object().unwrap();
        assert_eq!(
            shard.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["file", "sha256", "bytes"]
        );

        let model = obj["model"].as_object().unwrap();
        assert_eq!(
            model.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec![
                "model_type",
                "num_hidden_layers",
                "mtp_loaded",
                "rope_convention",
                "quant",
            ]
        );

        let quant = model["quant"].as_object().unwrap();
        assert_eq!(
            quant.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec![
                "quant_method",
                "fmt",
                "scale_fmt",
                "expert_dtype",
                "weight_block_size",
            ]
        );

        let corpus = obj["corpus"].as_object().unwrap();
        assert_eq!(
            corpus.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["token_ids_sha256", "n_tokens", "description"]
        );

        let capture = obj["capture"].as_object().unwrap();
        assert_eq!(
            capture.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["boundary", "tensors"]
        );

        let tensor = capture["tensors"][0].as_object().unwrap();
        assert_eq!(
            tensor.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["name", "rows", "k"]
        );

        let output = obj["outputs"][0].as_object().unwrap();
        assert_eq!(
            output.keys().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec!["path", "sha256", "bytes", "kind"]
        );

        assert_eq!(obj["schema"], json!(MANIFEST_SCHEMA));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn capture_boundary_wire_strings() {
        assert_eq!(
            serde_json::to_value(CaptureBoundary::PreQuant).unwrap(),
            json!("pre_quant")
        );
        assert_eq!(
            serde_json::to_value(CaptureBoundary::PostDynamicFp8).unwrap(),
            json!("post_dynamic_fp8")
        );
    }

    #[test]
    fn output_kind_wire_strings() {
        assert_eq!(
            serde_json::to_value(OutputKind::Logits).unwrap(),
            json!("logits")
        );
        assert_eq!(
            serde_json::to_value(OutputKind::Hessian).unwrap(),
            json!("hessian")
        );
        assert_eq!(
            serde_json::to_value(OutputKind::Acts).unwrap(),
            json!("acts")
        );
    }

    #[test]
    fn validate_rejects_unknown_schema() {
        let mut m = sample_manifest();
        m.schema = "hipfire.ds4.parent.manifest/0".into();
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("unknown or missing schema"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_empty_shards() {
        let mut m = sample_manifest();
        m.source.shards.clear();
        let err = m.validate().unwrap_err();
        assert!(err.contains("source.shards is empty"), "unexpected err: {err}");
    }

    #[test]
    fn validate_rejects_shard_empty_sha256() {
        let mut m = sample_manifest();
        m.source.shards[0].sha256.clear();
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("source.shards[0].sha256 is empty"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_empty_engine_commit() {
        let mut m = sample_manifest();
        m.engine.commit.clear();
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("engine.commit is empty"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_zero_corpus_tokens() {
        let mut m = sample_manifest();
        m.corpus.as_mut().expect("fixture has a corpus").n_tokens = 0;
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("corpus.n_tokens is zero"),
            "unexpected err: {err}"
        );
    }

    /// A producer that consumes no corpus (the inventory and codec gates)
    /// must be able to emit a valid manifest. Forcing it to invent a token
    /// count would be a fabricated provenance field, which is worse than none.
    #[test]
    fn validate_accepts_null_corpus_when_nothing_was_produced() {
        let mut m = sample_manifest();
        m.corpus = None;
        m.outputs.clear();
        m.capture.tensors.clear();
        m.validate()
            .expect("an inventory-only run legitimately has no corpus");
    }

    /// The other half of that bargain: no corpus means no corpus-derived
    /// artifact. This is the exact hole that let the rejected 554-tensor
    /// capture ship without naming the model that drove it.
    #[test]
    fn validate_rejects_null_corpus_with_declared_outputs() {
        let mut m = sample_manifest();
        m.corpus = None;
        m.capture.tensors.clear();
        assert!(!m.outputs.is_empty(), "fixture must declare an output");
        let err = m
            .validate()
            .expect_err("outputs without a corpus must be refused");
        assert!(err.contains("corpus is null"), "unexpected err: {err}");
    }

    #[test]
    fn validate_rejects_null_corpus_with_captured_activations() {
        let mut m = sample_manifest();
        m.corpus = None;
        m.outputs.clear();
        assert!(
            !m.capture.tensors.is_empty(),
            "fixture must capture a tensor"
        );
        let err = m
            .validate()
            .expect_err("captured activations without a corpus must be refused");
        assert!(
            err.contains("capture.tensors is non-empty"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_empty_corpus_hash() {
        let mut m = sample_manifest();
        m.corpus
            .as_mut()
            .expect("fixture has a corpus")
            .token_ids_sha256 = String::new();
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("corpus.token_ids_sha256 is empty"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_output_empty_sha256() {
        let mut m = sample_manifest();
        m.outputs[0].sha256.clear();
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("outputs[0].sha256 is empty"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_output_zero_bytes() {
        let mut m = sample_manifest();
        m.outputs[0].bytes = 0;
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("outputs[0].bytes is zero"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_tensor_zero_rows() {
        let mut m = sample_manifest();
        m.capture.tensors[0].rows = 0;
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("capture.tensors[0].rows is zero"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_rejects_tensor_zero_k() {
        let mut m = sample_manifest();
        m.capture.tensors[0].k = 0;
        let err = m.validate().unwrap_err();
        assert!(
            err.contains("capture.tensors[0].k is zero"),
            "unexpected err: {err}"
        );
    }

    #[test]
    fn validate_accepts_empty_capture_tensors_with_boundary() {
        let mut m = sample_manifest();
        m.capture.tensors.clear();
        m.capture.boundary = CaptureBoundary::PreQuant;
        m.validate().expect("logits-only capture must be accepted");
    }

    #[test]
    fn sha256_nist_empty() {
        assert_eq!(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_nist_abc() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_nist_448bit() {
        // NIST CAVP: "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" (56 bytes = 448 bits)
        assert_eq!(
            sha256_bytes(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_file_matches_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "ds4-parent-sha-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        let data = b"hipfire parent manifest sha256_file streaming path";
        fs::write(&path, data).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_bytes(data));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_environment_gfx942_succeeds() {
        let (producer, engine) =
            ParentManifest::probe_environment("gfx942").expect("probe_environment must succeed");

        assert!(!producer.binary.is_empty(), "binary path empty");
        assert_eq!(producer.binary_sha256.len(), 64, "binary sha256 length");
        assert!(!producer.argv.is_empty(), "argv empty");

        assert_eq!(engine.commit.len(), 40, "git SHA length: {}", engine.commit);
        assert!(
            engine.commit.chars().all(|c| c.is_ascii_hexdigit()),
            "commit not hex: {}",
            engine.commit
        );
        if let Some(d) = &engine.dirty_diff_sha256 {
            assert_eq!(d.len(), 64, "dirty diff sha length");
        }
        assert!(!engine.rocm_path.is_empty(), "rocm_path empty");
        assert!(!engine.rocm_version.is_empty(), "rocm_version empty");
        assert_ne!(
            engine.rocm_version, "7.14",
            "must not hardcode bare 7.14; got full version string from install"
        );
        assert_eq!(engine.gpu_arch, "gfx942");

        // Print real values for the acceptance report (cargo test -- --nocapture).
        eprintln!(
            "probe_environment ok: commit={} dirty_diff_sha256={:?} rocm_path={} rocm_version={} binary={}",
            engine.commit,
            engine.dirty_diff_sha256,
            engine.rocm_path,
            engine.rocm_version,
            producer.binary
        );
    }
}
