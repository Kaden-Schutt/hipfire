// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! O3c-1 live serve Dashboard data layer.
//!
//! The PARSE logic (HTTP JSON → struct, rocm-smi text → VRAM) is factored into
//! pure, I/O-free helpers so it can be unit-tested on CPU without a TTY, a
//! running serve, or a GPU. The fetchers ([`fetch_dashboard`]) do the network /
//! subprocess I/O and then hand raw text to those pure helpers.
//!
//! Honesty contract: every field is either REAL data parsed from the live
//! serve / rocm-smi, or an explicit unavailable/offline state. We NEVER
//! fabricate placeholder numbers and present them as live.

use std::{
    env, fs,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use serde_json::Value;

use super::config::ConfigState;

/// Hard wall-clock bound on a single `rocm-smi` invocation. Even on the
/// background thread a zombie / hung rocm-smi must not accumulate forever, so
/// we spawn it, poll for completion, and `kill()` it past this deadline.
const ROCM_SMI_TIMEOUT: Duration = Duration::from_millis(2000);

/// Per-GPU VRAM reading parsed from `rocm-smi`. Values are bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuMem {
    /// rocm-smi card index (the `card0`/GPU[0] ordinal).
    pub index: u32,
    pub used_bytes: u64,
    pub total_bytes: u64,
}

impl GpuMem {
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }
    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / 1e9
    }
    pub fn free_gb(&self) -> f64 {
        self.free_bytes() as f64 / 1e9
    }
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / 1e9
    }
}

/// VRAM availability state — distinguishes "we have real per-GPU numbers" from
/// the honest "rocm-smi is absent / not Linux / returned garbage" case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VramState {
    /// Real per-GPU readings parsed from rocm-smi.
    Available(Vec<GpuMem>),
    /// rocm-smi missing, non-Linux, or unparseable. Carries an honest reason.
    Unavailable(String),
}

/// Per-GPU live load parsed from `rocm-smi --showuse --showtemp --showpower`.
/// Each field is optional — rocm-smi key names vary by ROCm version, so a
/// missing reading is honestly `None`, never fabricated.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuLoad {
    pub index: u32,
    pub util_pct: Option<f64>,
    pub temp_c: Option<f64>,
    pub power_w: Option<f64>,
}

/// GPU-load availability — real per-GPU readings, or an honest reason.
#[derive(Clone, Debug, PartialEq)]
pub enum LoadState {
    Available(Vec<GpuLoad>),
    Unavailable(String),
}

/// Stats parsed from the serve's GET /stats endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeStats {
    pub model: Option<String>,
    pub uptime_s: u64,
    pub queue_depth: u64,
    pub requests_served: u64,
    pub recent_tok_s: Option<f64>,
}

/// A full Dashboard snapshot. `serve_up` is the single source of truth for the
/// online/offline split the UI renders.
#[derive(Clone, Debug)]
pub struct Dashboard {
    pub serve_up: bool,
    pub endpoint: String,
    /// Raw GET /health body (or the transport-error string when serve is down).
    /// Mirrored into `StatusState.health_text` for the System tab so the UI
    /// thread never re-probes /health synchronously.
    pub health_text: String,
    /// Loaded model tag, from /stats or /health (None when idle / serve down).
    pub model: Option<String>,
    pub stats: Option<ServeStats>,
    /// Models advertised by GET /v1/models (ids).
    pub model_ids: Vec<String>,
    pub vram: VramState,
    /// Live per-GPU util / temp / power from rocm-smi (independent of serve).
    pub gpu_load: LoadState,
    /// When serve is down: an honest hint, e.g. how to start it.
    pub offline_hint: Option<String>,
    /// Live system diagnostics (GPU/HIP/kernel-cache/loaded-model). Probed off
    /// the UI thread by the worker. `None` until the first probe completes.
    pub system: Option<SystemInfo>,
}

impl Dashboard {
    /// Offline dashboard: serve unreachable. Carries a start hint and whatever
    /// VRAM we could still read (rocm-smi works without serve).
    pub fn offline(endpoint: String, vram: VramState, gpu_load: LoadState) -> Self {
        Self {
            serve_up: false,
            endpoint,
            health_text: String::new(),
            model: None,
            stats: None,
            model_ids: Vec::new(),
            vram,
            gpu_load,
            offline_hint: Some("serve offline — start it with `hipfire serve -d`".into()),
            system: None,
        }
    }
}

/// A single live system-diagnostic field. Each is either a real probed value
/// or an explicit honest `Unavailable(reason)` — we never fabricate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Probe {
    Value(String),
    Unavailable(String),
}

impl Probe {
    pub fn display(&self) -> &str {
        match self {
            Probe::Value(s) => s,
            Probe::Unavailable(s) => s,
        }
    }
    pub fn is_available(&self) -> bool {
        matches!(self, Probe::Value(_))
    }
}

/// Live system diagnostics for the System tab. Probed OFF the UI thread (by the
/// DashboardWorker) so a slow `rocm-smi` / `hipcc` invocation never blocks
/// render or input. Every field is an honest [`Probe`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemInfo {
    /// GPU product name(s) from `rocm-smi --showproductname`.
    pub gpu_name: Probe,
    /// GPU arch / gfx target (e.g. `gfx1100`) from `rocm-smi --showhw`/gcnArch.
    pub gpu_arch: Probe,
    /// HIP / ROCm runtime version (`hipconfig --version` or /opt/rocm/.info).
    pub hip_version: Probe,
    /// Kernel-cache directory presence + path.
    pub kernel_cache: Probe,
    /// Currently loaded serve model (from the live snapshot, None when idle).
    pub loaded_model: Probe,
    /// Checksum of the loaded model file when readable (size-based fast hash).
    pub model_checksum: Probe,
}

impl SystemInfo {
    /// Probe everything that is platform-static (GPU, HIP, kernel cache). The
    /// loaded-model + checksum are filled in separately from the live serve
    /// snapshot via [`SystemInfo::with_loaded_model`].
    pub fn probe() -> Self {
        if !cfg!(target_os = "linux") {
            let na = || Probe::Unavailable("unavailable: Linux-only probe".into());
            return Self {
                gpu_name: na(),
                gpu_arch: na(),
                hip_version: na(),
                kernel_cache: probe_kernel_cache(),
                loaded_model: Probe::Unavailable("no model loaded".into()),
                model_checksum: Probe::Unavailable("no model loaded".into()),
            };
        }
        let (gpu_name, gpu_arch) = probe_gpu();
        Self {
            gpu_name,
            gpu_arch,
            hip_version: probe_hip_version(),
            kernel_cache: probe_kernel_cache(),
            loaded_model: Probe::Unavailable("no model loaded".into()),
            model_checksum: Probe::Unavailable("no model loaded".into()),
        }
    }

    /// Fold the live loaded-model (from the serve snapshot) into the static
    /// probe, computing a cheap file checksum when the path is readable.
    pub fn with_loaded_model(mut self, model: Option<&str>) -> Self {
        match model {
            Some(m) if !m.is_empty() => {
                self.loaded_model = Probe::Value(m.to_string());
                self.model_checksum = probe_model_checksum(m);
            }
            _ => {
                self.loaded_model = Probe::Unavailable("no model loaded".into());
                self.model_checksum = Probe::Unavailable("no model loaded".into());
            }
        }
        self
    }
}

/// Probe GPU product name + arch via `rocm-smi`. Bounded so a hung rocm-smi
/// cannot wedge the worker. Returns honest unavailable reasons on failure.
fn probe_gpu() -> (Probe, Probe) {
    let mut cmd = Command::new("rocm-smi");
    cmd.args(["--showproductname", "--json"]);
    let name = match run_bounded(cmd, ROCM_SMI_TIMEOUT) {
        RocmSmiRun::Output(text) => parse_rocm_product_name(&text),
        RocmSmiRun::NotFound => Probe::Unavailable("unavailable: rocm-smi not installed".into()),
        RocmSmiRun::TimedOut => Probe::Unavailable("unavailable: rocm-smi timed out".into()),
        RocmSmiRun::SpawnError(e) => {
            Probe::Unavailable(format!("unavailable: rocm-smi error: {e}"))
        }
    };

    let cmd = Command::new("rocminfo");
    let arch = match run_bounded(cmd, ROCM_SMI_TIMEOUT) {
        RocmSmiRun::Output(text) => parse_rocminfo_arch(&text),
        RocmSmiRun::NotFound => Probe::Unavailable("unavailable: rocminfo not installed".into()),
        RocmSmiRun::TimedOut => Probe::Unavailable("unavailable: rocminfo timed out".into()),
        RocmSmiRun::SpawnError(e) => {
            Probe::Unavailable(format!("unavailable: rocminfo error: {e}"))
        }
    };
    (name, arch)
}

/// Parse `rocm-smi --showproductname --json`: ROCm emits per-card objects with
/// a `"Card Series"` / `"Card Model"` / `"Card SKU"` string field (key varies
/// by ROCm version). Returns the first non-empty product string found.
pub fn parse_rocm_product_name(body: &str) -> Probe {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        return Probe::Unavailable("unavailable: rocm-smi product JSON unparseable".into());
    };
    let Some(obj) = v.as_object() else {
        return Probe::Unavailable("unavailable: rocm-smi product JSON not an object".into());
    };
    let keys = [
        "Card Series",
        "Card Model",
        "Card SKU",
        "Device Name",
        "GPU ID",
    ];
    let mut names: Vec<String> = Vec::new();
    let mut cards: Vec<(u32, &Value)> = obj
        .iter()
        .filter_map(|(k, val)| {
            k.strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
                .map(|idx| (idx, val))
        })
        .collect();
    cards.sort_by_key(|(idx, _)| *idx);
    for (_, card) in cards {
        for key in keys {
            if let Some(Value::String(s)) = card.get(key) {
                let s = s.trim();
                if !s.is_empty() {
                    names.push(s.to_string());
                    break;
                }
            }
        }
    }
    if names.is_empty() {
        Probe::Unavailable("unavailable: no product name in rocm-smi output".into())
    } else {
        names.dedup();
        Probe::Value(names.join(", "))
    }
}

/// Parse `rocminfo` text output for the first GPU agent's `gfx` name (the
/// `Name:` line under an Agent of Device Type GPU). Returns the gfx target.
pub fn parse_rocminfo_arch(body: &str) -> Probe {
    // rocminfo lists agents; the GPU agent has a `Name:  gfxNNNN` line.
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Name:") {
            let name = rest.trim();
            if name.starts_with("gfx") {
                return Probe::Value(name.to_string());
            }
        }
        // Some builds use "gfx Name:" / "Marketing Name:" — catch the gfx token.
        if let Some(idx) = t.find("gfx") {
            let tok: String = t[idx..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect();
            if tok.len() > 3 && tok[3..].chars().any(|c| c.is_ascii_digit()) {
                return Probe::Value(tok);
            }
        }
    }
    Probe::Unavailable("unavailable: no gfx target in rocminfo output".into())
}

/// Probe the HIP/ROCm version. Prefers `hipconfig --version`, falls back to the
/// `/opt/rocm/.info/version` file.
fn probe_hip_version() -> Probe {
    let mut cmd = Command::new("hipconfig");
    cmd.arg("--version");
    if let RocmSmiRun::Output(text) = run_bounded(cmd, ROCM_SMI_TIMEOUT) {
        let v = text.trim();
        if !v.is_empty() {
            return Probe::Value(format!("HIP {v}"));
        }
    }
    // Fallback: ROCm version info file.
    for path in ["/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"] {
        if let Ok(v) = fs::read_to_string(path) {
            let v = v.trim();
            if !v.is_empty() {
                return Probe::Value(format!("ROCm {v}"));
            }
        }
    }
    Probe::Unavailable("unavailable: hipconfig absent and /opt/rocm/.info missing".into())
}

/// Resolve the kernel-cache directory the engine uses and report its presence.
/// Honors `HIPFIRE_KERNEL_CACHE` then falls back to `~/.hipfire_kernels` and the
/// HIP default `~/.cache/hip`.
fn kernel_cache_dir() -> Option<PathBuf> {
    if let Some(v) = env::var_os("HIPFIRE_KERNEL_CACHE") {
        return Some(PathBuf::from(v));
    }
    let home = env::var_os("HOME").map(PathBuf::from)?;
    let primary = home.join(".hipfire_kernels");
    if primary.exists() {
        return Some(primary);
    }
    Some(home.join(".cache").join("hip"))
}

fn probe_kernel_cache() -> Probe {
    match kernel_cache_dir() {
        Some(dir) => {
            let exists = dir.exists();
            let count = if exists {
                fs::read_dir(&dir).map(|rd| rd.count()).unwrap_or(0)
            } else {
                0
            };
            let disp = dir.display();
            if exists {
                Probe::Value(format!("{disp} (present, {count} entries)"))
            } else {
                Probe::Unavailable(format!("{disp} (absent — populated on first run)"))
            }
        }
        None => Probe::Unavailable("unavailable: cannot resolve cache dir (no HOME)".into()),
    }
}

/// Compute a cheap, stable checksum for the loaded model file. The model field
/// is often an absolute path; when it is and the file is readable we emit a
/// size + FNV-1a hash of the first/last 64 KiB (full hashing a multi-GB weight
/// file on the worker thread would be wasteful). Honest unavailable otherwise.
fn probe_model_checksum(model: &str) -> Probe {
    let path = PathBuf::from(model);
    if !path.is_absolute() || !path.exists() {
        return Probe::Unavailable("unavailable: model path not a readable file".into());
    }
    let meta = match fs::metadata(&path) {
        Ok(m) => m,
        Err(e) => return Probe::Unavailable(format!("unavailable: stat failed: {e}")),
    };
    if !meta.is_file() {
        return Probe::Unavailable("unavailable: model path is not a file".into());
    }
    let size = meta.len();
    match fnv_head_tail(&path, size) {
        Some(hash) => Probe::Value(format!("{hash:016x} ({:.2} GB)", size as f64 / 1e9)),
        None => Probe::Unavailable("unavailable: model file unreadable".into()),
    }
}

/// FNV-1a over (size ++ first 64 KiB ++ last 64 KiB) — fast, deterministic,
/// good enough to distinguish loaded weights without reading gigabytes.
fn fnv_head_tail(path: &std::path::Path, size: u64) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};
    const CHUNK: u64 = 64 * 1024;
    let mut f = fs::File::open(path).ok()?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let fold = |bytes: &[u8], hash: &mut u64| {
        for b in bytes {
            *hash ^= *b as u64;
            *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    fold(&size.to_le_bytes(), &mut hash);
    let mut head = vec![0u8; CHUNK.min(size) as usize];
    f.read_exact(&mut head).ok()?;
    fold(&head, &mut hash);
    if size > CHUNK {
        let tail_len = CHUNK.min(size - CHUNK);
        f.seek(SeekFrom::End(-(tail_len as i64))).ok()?;
        let mut tail = vec![0u8; tail_len as usize];
        f.read_exact(&mut tail).ok()?;
        fold(&tail, &mut hash);
    }
    Some(hash)
}

// ── Pure parse helpers (no I/O) ─────────────────────────────────────────────

/// Parse the GET /health JSON body. Returns the loaded-model field (which may
/// be JSON null → None). Returns None for garbage / non-object input rather
/// than fabricating a value.
pub fn parse_health_model(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    match v.get("model") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Parse the GET /stats JSON body into [`ServeStats`]. Returns None on garbage
/// / non-object input. Missing numeric fields default to 0 (the serve always
/// emits them, but be defensive against an older build); `recent_tok_s` stays
/// None unless present and finite/positive.
pub fn parse_stats(body: &str) -> Option<ServeStats> {
    let v: Value = serde_json::from_str(body).ok()?;
    if !v.is_object() {
        return None;
    }
    let model = match v.get("model") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let as_u64 = |key: &str| v.get(key).and_then(|x| x.as_u64()).unwrap_or(0);
    let recent_tok_s = v
        .get("recent_tok_s")
        .and_then(|x| x.as_f64())
        .filter(|f| f.is_finite() && *f > 0.0);
    Some(ServeStats {
        model,
        uptime_s: as_u64("uptime_s"),
        queue_depth: as_u64("queue_depth"),
        requests_served: as_u64("requests_served"),
        recent_tok_s,
    })
}

/// Parse the GET /v1/models JSON body into the list of model ids. Empty vec on
/// garbage input.
pub fn parse_model_ids(body: &str) -> Vec<String> {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse `rocm-smi --showmeminfo vram --json` output. ROCm emits a JSON object
/// keyed by `card0`, `card1`, … with `"VRAM Total Memory (B)"` and
/// `"VRAM Total Used Memory (B)"` string fields. Returns an honest
/// [`VramState::Unavailable`] when nothing parses.
pub fn parse_rocm_smi_json(body: &str) -> VramState {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return VramState::Unavailable(format!("rocm-smi JSON parse failed: {e}")),
    };
    let Some(obj) = v.as_object() else {
        return VramState::Unavailable("rocm-smi JSON not an object".into());
    };
    let mut gpus: Vec<GpuMem> = Vec::new();
    // Iterate cardN keys in numeric order.
    let mut cards: Vec<(u32, &Value)> = obj
        .iter()
        .filter_map(|(k, val)| {
            k.strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
                .map(|idx| (idx, val))
        })
        .collect();
    cards.sort_by_key(|(idx, _)| *idx);
    for (idx, card) in cards {
        let total = field_u64(card, "VRAM Total Memory (B)");
        let used = field_u64(card, "VRAM Total Used Memory (B)");
        if let (Some(total), Some(used)) = (total, used) {
            if total > 0 {
                gpus.push(GpuMem {
                    index: idx,
                    used_bytes: used,
                    total_bytes: total,
                });
            }
        }
    }
    if gpus.is_empty() {
        VramState::Unavailable("rocm-smi returned no VRAM fields".into())
    } else {
        VramState::Available(gpus)
    }
}

/// rocm-smi reports numbers as JSON strings; read either a string-encoded or a
/// raw number.
fn field_u64(card: &Value, key: &str) -> Option<u64> {
    match card.get(key)? {
        Value::String(s) => s.trim().parse::<u64>().ok(),
        Value::Number(n) => n.as_u64(),
        _ => None,
    }
}

// ── Fetchers (I/O; thin wrappers around the pure parsers) ───────────────────

fn get_text(agent: &ureq::Agent, url: &str) -> Option<String> {
    match agent.get(url).call() {
        Ok(mut resp) if resp.status().as_u16() < 400 => resp.body_mut().read_to_string().ok(),
        _ => None,
    }
}

/// Outcome of a bounded subprocess run. Separated from parsing so the timeout /
/// kill path stays testable in isolation. Shared with the doctor module.
pub(crate) enum RocmSmiRun {
    /// Process exited within the deadline; carries captured stdout.
    Output(String),
    /// Binary not present on PATH.
    NotFound,
    /// Spawn or wait failed for some other reason.
    SpawnError(String),
    /// Process exceeded [`ROCM_SMI_TIMEOUT`] and was killed.
    TimedOut,
}

/// Run `rocm-smi --showmeminfo vram --json` with a hard wall-clock deadline.
/// Spawns the child, polls `try_wait`, and `kill()`s it if it blows past
/// [`ROCM_SMI_TIMEOUT`] so a hung rocm-smi can never wedge the caller (even on
/// the background fetch thread, where it would otherwise leak a zombie probe).
fn run_rocm_smi_bounded() -> RocmSmiRun {
    let mut cmd = Command::new("rocm-smi");
    cmd.args(["--showmeminfo", "vram", "--json"]);
    run_bounded(cmd, ROCM_SMI_TIMEOUT)
}

fn run_rocm_smi_load_bounded() -> RocmSmiRun {
    let mut cmd = Command::new("rocm-smi");
    cmd.args(["--showuse", "--showtemp", "--showpower", "--json"]);
    run_bounded(cmd, ROCM_SMI_TIMEOUT)
}

/// Probe rocm-smi for per-GPU util / temperature / power. Honest
/// [`LoadState::Unavailable`] on non-Linux, missing binary, timeout, or
/// unparseable output. Never blocks indefinitely (bounded + killed).
pub fn fetch_gpu_load() -> LoadState {
    if !cfg!(target_os = "linux") {
        return LoadState::Unavailable("GPU load unavailable: rocm-smi is Linux-only".into());
    }
    match run_rocm_smi_load_bounded() {
        RocmSmiRun::Output(text) => parse_rocm_smi_load_json(&text),
        RocmSmiRun::NotFound => {
            LoadState::Unavailable("GPU load unavailable: rocm-smi not installed".into())
        }
        RocmSmiRun::TimedOut => LoadState::Unavailable(format!(
            "GPU load unavailable: rocm-smi timed out after {}ms (killed)",
            ROCM_SMI_TIMEOUT.as_millis()
        )),
        RocmSmiRun::SpawnError(e) => {
            LoadState::Unavailable(format!("GPU load unavailable: rocm-smi error: {e}"))
        }
    }
}

/// Parse `rocm-smi --showuse --showtemp --showpower --json`. Field key names
/// vary by ROCm version, so each metric is matched by substring (in preference
/// order) and a missing field is honestly `None`.
pub fn parse_rocm_smi_load_json(body: &str) -> LoadState {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return LoadState::Unavailable(format!("rocm-smi load JSON parse failed: {e}")),
    };
    let Some(obj) = v.as_object() else {
        return LoadState::Unavailable("rocm-smi load JSON not an object".into());
    };
    let mut cards: Vec<(u32, &Value)> = obj
        .iter()
        .filter_map(|(k, val)| {
            k.strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
                .map(|idx| (idx, val))
        })
        .collect();
    cards.sort_by_key(|(idx, _)| *idx);
    let mut loads = Vec::new();
    for (idx, card) in cards {
        // Specific needles only (no bare "Power"/"Temperature" fallback, which
        // could latch onto a cap/sensor key); each reading is range-validated so
        // an implausible value becomes an honest None.
        let util = field_f64_containing(card, &["GPU use (%)", "GPU use"])
            .filter(|u| (0.0..=100.0).contains(u));
        let temp = field_f64_containing(
            card,
            &["Temperature (Sensor edge)", "Temperature (Sensor junction)"],
        )
        .filter(|t| (0.0..=200.0).contains(t));
        let power = field_f64_containing(
            card,
            &[
                "Average Graphics Package Power",
                "Current Socket Graphics Package Power",
                "Current Graphics Package Power",
            ],
        )
        .filter(|p| (0.0..=2000.0).contains(p));
        if util.is_some() || temp.is_some() || power.is_some() {
            loads.push(GpuLoad {
                index: idx,
                util_pct: util,
                temp_c: temp,
                power_w: power,
            });
        }
    }
    if loads.is_empty() {
        LoadState::Unavailable("rocm-smi returned no load fields".into())
    } else {
        LoadState::Available(loads)
    }
}

/// First card field whose key CONTAINS one of `needles` (tried in order),
/// parsed as a FINITE f64. rocm-smi emits numbers as strings, and
/// `f64::from_str` accepts "NaN"/"inf" (which rocm-smi reports for unsupported
/// sensors on some SKUs) — we reject those so a non-finite reading falls through
/// to an honest `None` rather than rendering "NaN% util". Cap / Max / Min
/// sentinel keys are skipped so a "Power Cap" can't be mistaken for live draw.
fn field_f64_containing(card: &Value, needles: &[&str]) -> Option<f64> {
    let obj = card.as_object()?;
    for needle in needles {
        for (k, val) in obj {
            if k.contains("Cap") || k.contains("Max") || k.contains("Min") {
                continue;
            }
            if k.contains(needle) {
                let parsed = val
                    .as_str()
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .or_else(|| val.as_f64());
                if let Some(n) = parsed {
                    if n.is_finite() {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

/// Spawn `cmd`, capturing stdout, and enforce a hard `timeout`. Polls
/// `try_wait`; on deadline it `kill()`s and reaps the child so no zombie leaks.
/// Generic over the command so the timeout/kill path is unit-testable with a
/// stand-in (`sleep`) without depending on rocm-smi.
pub(crate) fn run_bounded(mut cmd: Command, timeout: Duration) -> RocmSmiRun {
    let mut child = match cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return RocmSmiRun::NotFound,
        Err(e) => return RocmSmiRun::SpawnError(e.to_string()),
    };

    // Drain stdout CONCURRENTLY on a reader thread. A verbose command (rocminfo)
    // can emit more than one OS pipe buffer (~64 KiB) of output BEFORE it exits;
    // if we only read after try_wait() reports exit, the child blocks on its
    // write() — full pipe, no reader — never exits, and we kill it as a FALSE
    // timeout. With a dedicated reader thread the pipe never backs up, so the
    // child can run to completion while the watchdog below still bounds the
    // wall-clock and kills a genuinely hung child.
    let stdout = child.stdout.take();
    let reader = stdout.map(|mut out| {
        thread::spawn(move || {
            let mut buf = String::new();
            let _ = out.read_to_string(&mut buf);
            buf
        })
    });

    // Watchdog: poll for exit, enforce the hard deadline (kill + reap on timeout).
    let deadline = Instant::now() + timeout;
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(_status)) => break None, // exited cleanly within deadline
            Ok(None) => {
                if Instant::now() >= deadline {
                    // Hung: kill and reap so we don't leak a zombie. Killing the
                    // child also closes the pipe, so the reader thread's
                    // read_to_string returns and we can join it without hanging.
                    let _ = child.kill();
                    let _ = child.wait();
                    break Some(RocmSmiRun::TimedOut);
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                // Reap best-effort so the reader thread unblocks, then report.
                let _ = child.kill();
                let _ = child.wait();
                break Some(RocmSmiRun::SpawnError(e.to_string()));
            }
        }
    };

    // Join the reader (it has finished once the pipe closed: clean exit OR kill).
    let captured = match reader {
        Some(h) => h.join().unwrap_or_default(),
        None => String::new(),
    };

    match outcome {
        Some(run) => run, // TimedOut / SpawnError — discard partial output
        None => RocmSmiRun::Output(captured),
    }
}

/// Probe rocm-smi for per-GPU VRAM. Honest [`VramState::Unavailable`] when the
/// platform is not Linux, the binary is absent, the output doesn't parse, or
/// the call exceeds [`ROCM_SMI_TIMEOUT`] (hard-killed). Never blocks
/// indefinitely.
pub fn fetch_vram() -> VramState {
    if !cfg!(target_os = "linux") {
        return VramState::Unavailable("VRAM unavailable: rocm-smi is Linux-only".into());
    }
    match run_rocm_smi_bounded() {
        RocmSmiRun::Output(text) => {
            let parsed = parse_rocm_smi_json(&text);
            // If JSON failed but the binary ran, surface a clean reason.
            match parsed {
                VramState::Available(_) => parsed,
                VramState::Unavailable(_) if text.trim().is_empty() => {
                    VramState::Unavailable("VRAM unavailable: rocm-smi produced no output".into())
                }
                other => other,
            }
        }
        RocmSmiRun::NotFound => {
            VramState::Unavailable("VRAM unavailable: rocm-smi not installed".into())
        }
        RocmSmiRun::TimedOut => VramState::Unavailable(format!(
            "VRAM unavailable: rocm-smi timed out after {}ms (killed)",
            ROCM_SMI_TIMEOUT.as_millis()
        )),
        RocmSmiRun::SpawnError(e) => {
            VramState::Unavailable(format!("VRAM unavailable: rocm-smi error: {e}"))
        }
    }
}

/// Build a full live Dashboard snapshot: probes /health, /stats, /v1/models,
/// and rocm-smi. Tolerant of serve being down (returns [`Dashboard::offline`])
/// and of rocm-smi being absent (VramState::Unavailable). All timeouts are
/// short so the refresh tick never wedges the UI.
pub fn fetch_dashboard(config: &ConfigState) -> Dashboard {
    let endpoint = format!("{}:{}", config.probe_host(), config.port);
    let vram = fetch_vram();
    let gpu_load = fetch_gpu_load();

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(450)))
        .http_status_as_error(false)
        .build()
        .into();
    let base = format!("http://{endpoint}");

    let health = get_text(&agent, &format!("{base}/health"));
    if health.is_none() {
        // Serve unreachable — honest offline state (VRAM/load may still read).
        return Dashboard::offline(endpoint, vram, gpu_load);
    }
    let health = health.unwrap();
    let health_model = parse_health_model(&health);

    let stats = get_text(&agent, &format!("{base}/stats")).and_then(|b| parse_stats(&b));
    let model_ids = get_text(&agent, &format!("{base}/v1/models"))
        .map(|b| parse_model_ids(&b))
        .unwrap_or_default();

    // Prefer the model from /stats (authoritative), fall back to /health.
    let model = stats
        .as_ref()
        .and_then(|s| s.model.clone())
        .or(health_model);

    Dashboard {
        serve_up: true,
        endpoint,
        health_text: health,
        model,
        stats,
        model_ids,
        vram,
        gpu_load,
        offline_hint: None,
        system: None,
    }
}

// ── Background fetch worker ─────────────────────────────────────────────────

/// How often the background thread re-fetches a snapshot while the Dashboard
/// tab is active.
const WORKER_REFRESH: Duration = Duration::from_millis(1500);
/// Poll cadence for the worker's "is the tab active?" / "should I exit?" flags
/// while it is idle (Dashboard tab not focused).
const WORKER_IDLE_POLL: Duration = Duration::from_millis(150);

/// Owns the background fetch thread. The UI thread NEVER calls
/// [`fetch_dashboard`] / rocm-smi / HTTP directly; it only reads the latest
/// snapshot via [`DashboardWorker::snapshot`]. The worker fetches on its own
/// thread every [`WORKER_REFRESH`] while the Dashboard tab is active, writing
/// the result into a shared `Arc<Mutex<Option<Dashboard>>>`, so a hung
/// rocm-smi or slow HTTP probe can stall only the worker — never render/input.
pub struct DashboardWorker {
    snapshot: Arc<Mutex<Option<Dashboard>>>,
    /// Whether the Dashboard tab is currently focused. The worker only fetches
    /// when this is true (plus an immediate fetch when it flips on).
    active: Arc<AtomicBool>,
    /// Set by the UI to force an immediate re-fetch (manual `r` refresh).
    force: Arc<AtomicBool>,
    /// Live config snapshot the worker reads each cycle (host/port). Updated by
    /// the UI on reload so a port change is picked up without restarting.
    config: Arc<Mutex<ConfigState>>,
    /// Signals the worker to exit; also wakes it via `wake`.
    shutdown: Arc<AtomicBool>,
    wake: mpsc::Sender<()>,
    handle: Option<thread::JoinHandle<()>>,
}

impl DashboardWorker {
    /// Spawn the background fetch thread. It starts idle (no fetch) until the
    /// Dashboard tab is marked active via [`DashboardWorker::set_active`].
    pub fn spawn(config: ConfigState) -> Self {
        let snapshot = Arc::new(Mutex::new(None));
        let active = Arc::new(AtomicBool::new(false));
        let force = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let config = Arc::new(Mutex::new(config));
        let (wake, wake_rx) = mpsc::channel::<()>();

        let handle = {
            let snapshot = Arc::clone(&snapshot);
            let active = Arc::clone(&active);
            let force = Arc::clone(&force);
            let shutdown = Arc::clone(&shutdown);
            let config = Arc::clone(&config);
            thread::spawn(move || {
                worker_loop(snapshot, active, force, shutdown, config, wake_rx);
            })
        };

        Self {
            snapshot,
            active,
            force,
            config,
            shutdown,
            wake,
            handle: Some(handle),
        }
    }

    /// Read the latest snapshot (cheap clone). `None` until the first fetch
    /// completes. This is the ONLY path the UI thread uses for dashboard data.
    pub fn snapshot(&self) -> Option<Dashboard> {
        // Poison-tolerant: this runs on the UI thread every frame, so a worker
        // panic while holding the lock must NOT crash render — recover the guard.
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Mark the Dashboard tab focused / unfocused. Flipping to active wakes the
    /// worker so the first fetch starts immediately instead of after an idle
    /// poll.
    pub fn set_active(&self, active: bool) {
        let was = self.active.swap(active, Ordering::SeqCst);
        if active && !was {
            let _ = self.wake.send(());
        }
    }

    /// Request an immediate re-fetch (manual refresh). No-op cost if the worker
    /// is mid-cycle; it picks the flag up at the next loop turn.
    pub fn force_refresh(&self) {
        self.force.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
    }

    /// Update the config the worker fetches against (e.g. after a port change
    /// on reload).
    pub fn update_config(&self, config: ConfigState) {
        *self.config.lock().unwrap_or_else(|e| e.into_inner()) = config;
    }
}

impl Drop for DashboardWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(
    snapshot: Arc<Mutex<Option<Dashboard>>>,
    active: Arc<AtomicBool>,
    force: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    config: Arc<Mutex<ConfigState>>,
    wake_rx: mpsc::Receiver<()>,
) {
    let mut last_fetch: Option<Instant> = None;
    // Static system diagnostics (GPU/HIP/kernel-cache) are probed once and
    // cached — they don't change within a session and re-running rocm-smi /
    // rocminfo every 1.5s would be wasteful. The loaded-model + checksum
    // overlay IS refreshed each cycle from the live serve snapshot.
    let mut system_base: Option<SystemInfo> = None;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }

        let is_active = active.load(Ordering::SeqCst);
        let forced = force.swap(false, Ordering::SeqCst);
        // A forced refresh (manual `r`) ALWAYS fetches, even when the live tab
        // is not focused — otherwise the force flag is swapped-then-dropped and
        // the snapshot goes stale until Dashboard/System is next opened. The
        // time-based cadence still only fires while active.
        let due = forced
            || (is_active
                && last_fetch
                    .map(|t| t.elapsed() >= WORKER_REFRESH)
                    .unwrap_or(true));

        if due {
            // Clone the config out of the lock so the (potentially slow) fetch
            // never holds it.
            let cfg = config.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let mut dash = fetch_dashboard(&cfg);
            // Probe static system info on first need; reuse thereafter.
            let base = system_base.get_or_insert_with(SystemInfo::probe).clone();
            dash.system = Some(base.with_loaded_model(dash.model.as_deref()));
            *snapshot.lock().unwrap_or_else(|e| e.into_inner()) = Some(dash);
            last_fetch = Some(Instant::now());
        }

        // Sleep until the next refresh is due (when active) or a short idle
        // poll (when inactive), but wake early on force/active/shutdown signals.
        let wait = if is_active {
            last_fetch
                .map(|t| WORKER_REFRESH.saturating_sub(t.elapsed()))
                .unwrap_or(WORKER_REFRESH)
                .max(Duration::from_millis(20))
        } else {
            WORKER_IDLE_POLL
        };
        // Drain a single wake (coalesces bursts) or time out.
        let _ = wake_rx.recv_timeout(wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_load_json_extracts_util_temp_power() {
        let body = r#"{
            "card0": {
                "GPU use (%)": "37",
                "Temperature (Sensor edge) (C)": "45.0",
                "Average Graphics Package Power (W)": "120.0"
            },
            "card1": {
                "GPU use (%)": "0",
                "Temperature (Sensor junction) (C)": "33.0",
                "Current Socket Graphics Package Power (W)": "18.0"
            }
        }"#;
        match parse_rocm_smi_load_json(body) {
            LoadState::Available(loads) => {
                assert_eq!(loads.len(), 2);
                assert_eq!(loads[0].util_pct, Some(37.0));
                assert_eq!(loads[0].temp_c, Some(45.0));
                assert_eq!(loads[0].power_w, Some(120.0));
                // Version-variant key names still resolve via substring match.
                assert_eq!(loads[1].temp_c, Some(33.0));
                assert_eq!(loads[1].power_w, Some(18.0));
            }
            other => panic!("expected Available, got {other:?}"),
        }
    }

    #[test]
    fn parse_load_json_tolerates_missing_fields_and_bad_input() {
        // Only util present -> Available with temp/power honestly None.
        match parse_rocm_smi_load_json(r#"{"card0": {"GPU use (%)": "12"}}"#) {
            LoadState::Available(loads) => {
                assert_eq!(loads[0].util_pct, Some(12.0));
                assert_eq!(loads[0].temp_c, None);
                assert_eq!(loads[0].power_w, None);
            }
            other => panic!("expected Available, got {other:?}"),
        }
        // No load fields -> Unavailable, not a fabricated zero.
        assert!(matches!(
            parse_rocm_smi_load_json(r#"{"card0": {}}"#),
            LoadState::Unavailable(_)
        ));
        // Bad JSON -> Unavailable.
        assert!(matches!(
            parse_rocm_smi_load_json("not json"),
            LoadState::Unavailable(_)
        ));
    }

    #[test]
    fn parse_load_json_rejects_nonfinite_and_cap_keys() {
        // rocm-smi reports "NaN"/"inf" for unsupported sensors on some SKUs;
        // f64::from_str accepts them, so they must be rejected to honest None.
        let body = r#"{"card0": {
            "GPU use (%)": "NaN",
            "Average Graphics Package Power (W)": "inf",
            "Temperature (Sensor edge) (C)": "45.0"
        }}"#;
        match parse_rocm_smi_load_json(body) {
            LoadState::Available(loads) => {
                assert_eq!(loads[0].util_pct, None, "NaN util rejected");
                assert_eq!(loads[0].power_w, None, "inf power rejected");
                assert_eq!(loads[0].temp_c, Some(45.0), "finite temp kept");
            }
            other => panic!("expected Available, got {other:?}"),
        }
        // A power CAP emitted before the draw key must not be read as live power.
        let body2 = r#"{"card0": {
            "Max Graphics Package Power (W)": "327",
            "Average Graphics Package Power (W)": "120"
        }}"#;
        match parse_rocm_smi_load_json(body2) {
            LoadState::Available(loads) => {
                assert_eq!(loads[0].power_w, Some(120.0), "live draw, not the cap");
            }
            other => panic!("expected Available, got {other:?}"),
        }
        // Out-of-range util (sensor sentinel) -> None.
        match parse_rocm_smi_load_json(r#"{"card0": {"GPU use (%)": "65535"}}"#) {
            LoadState::Unavailable(_) => {}
            LoadState::Available(loads) => assert_eq!(loads[0].util_pct, None),
        }
    }

    #[test]
    fn health_model_present() {
        let b = r#"{"status":"ok","model":"/home/u/.hipfire/models/q.mq4","pid":42}"#;
        assert_eq!(
            parse_health_model(b),
            Some("/home/u/.hipfire/models/q.mq4".to_string())
        );
    }

    #[test]
    fn health_model_null_or_idle() {
        assert_eq!(parse_health_model(r#"{"status":"ok","model":null}"#), None);
        assert_eq!(parse_health_model(r#"{"status":"ok","model":""}"#), None);
        assert_eq!(parse_health_model(r#"{"status":"ok"}"#), None);
    }

    #[test]
    fn health_garbage_is_none_not_panic() {
        assert_eq!(parse_health_model("not json"), None);
        assert_eq!(parse_health_model(""), None);
        assert_eq!(parse_health_model("[1,2,3]"), None);
    }

    #[test]
    fn stats_full_shape() {
        let b = r#"{"model":"q.mq4","uptime_s":63,"queue_depth":2,"requests_served":42,"recent_tok_s":151.24}"#;
        let s = parse_stats(b).expect("parse");
        assert_eq!(s.model.as_deref(), Some("q.mq4"));
        assert_eq!(s.uptime_s, 63);
        assert_eq!(s.queue_depth, 2);
        assert_eq!(s.requests_served, 42);
        assert_eq!(s.recent_tok_s, Some(151.24));
    }

    #[test]
    fn stats_idle_no_model_no_tok_s() {
        let b = r#"{"model":null,"uptime_s":5,"queue_depth":0,"requests_served":0}"#;
        let s = parse_stats(b).expect("parse");
        assert_eq!(s.model, None);
        assert_eq!(s.queue_depth, 0);
        assert_eq!(s.requests_served, 0);
        assert_eq!(s.recent_tok_s, None);
    }

    #[test]
    fn stats_garbage_is_none() {
        assert_eq!(parse_stats("not json"), None);
        assert_eq!(parse_stats("[1,2]"), None);
        assert_eq!(parse_stats(""), None);
    }

    #[test]
    fn stats_rejects_nonfinite_tok_s() {
        // serde won't parse NaN, but a zero/negative must be dropped.
        let b =
            r#"{"model":"m","uptime_s":1,"queue_depth":0,"requests_served":1,"recent_tok_s":0}"#;
        assert_eq!(parse_stats(b).unwrap().recent_tok_s, None);
        let b2 =
            r#"{"model":"m","uptime_s":1,"queue_depth":0,"requests_served":1,"recent_tok_s":-3}"#;
        assert_eq!(parse_stats(b2).unwrap().recent_tok_s, None);
    }

    #[test]
    fn model_ids_parsed() {
        let b = r#"{"data":[{"id":"qwen3.5:9b"},{"id":"qwen3.5:27b"}]}"#;
        assert_eq!(parse_model_ids(b), vec!["qwen3.5:9b", "qwen3.5:27b"]);
        assert!(parse_model_ids("garbage").is_empty());
        assert!(parse_model_ids(r#"{"data":[]}"#).is_empty());
    }

    #[test]
    fn rocm_smi_two_gpus() {
        let b = r#"{
            "card1": {"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "10000000000"},
            "card0": {"VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": "2000000000"}
        }"#;
        let vram = parse_rocm_smi_json(b);
        match vram {
            VramState::Available(gpus) => {
                assert_eq!(gpus.len(), 2);
                // Sorted by card index.
                assert_eq!(gpus[0].index, 0);
                assert_eq!(gpus[0].total_bytes, 34208743424);
                assert_eq!(gpus[0].used_bytes, 2000000000);
                assert_eq!(gpus[0].free_bytes(), 32208743424);
                assert_eq!(gpus[1].index, 1);
            }
            VramState::Unavailable(r) => panic!("expected Available, got {r}"),
        }
    }

    #[test]
    fn rocm_smi_numeric_fields_also_work() {
        let b = r#"{"card0":{"VRAM Total Memory (B)": 8000000000, "VRAM Total Used Memory (B)": 1000000000}}"#;
        match parse_rocm_smi_json(b) {
            VramState::Available(g) => assert_eq!(g[0].total_bytes, 8000000000),
            VramState::Unavailable(r) => panic!("expected Available, got {r}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_kills_hung_process() {
        // A process that sleeps far longer than the timeout must be killed and
        // reported as TimedOut within ~the timeout (not blocked for 60s).
        let mut cmd = Command::new("sleep");
        cmd.arg("60");
        let start = Instant::now();
        let run = run_bounded(cmd, Duration::from_millis(200));
        let elapsed = start.elapsed();
        assert!(matches!(run, RocmSmiRun::TimedOut), "expected TimedOut");
        assert!(
            elapsed < Duration::from_secs(5),
            "kill must be prompt, took {elapsed:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_captures_fast_output() {
        let mut cmd = Command::new("printf");
        cmd.arg("hello");
        match run_bounded(cmd, Duration::from_secs(5)) {
            RocmSmiRun::Output(s) => assert_eq!(s, "hello"),
            other => panic!("expected Output, got {:?}", run_kind(&other)),
        }
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_stdout_fill_is_not_falsely_timed_out() {
        // F2: a command that emits MORE than one OS pipe buffer (~64 KiB) of
        // stdout BEFORE exiting must run to completion, not be killed as a false
        // timeout. Before the concurrent-drain fix, the child would block on
        // write() against a full pipe (no reader until after exit) and get
        // killed. We emit ~1 MiB and assert we capture all of it.
        let mut cmd = Command::new("sh");
        // `yes` would never exit; use a bounded large emission via head.
        cmd.args(["-c", "yes ABCDEFGHIJ | head -c 1048576"]);
        let run = run_bounded(cmd, Duration::from_secs(10));
        match run {
            RocmSmiRun::Output(s) => {
                assert_eq!(
                    s.len(),
                    1_048_576,
                    "must capture the full 1 MiB without deadlocking the pipe"
                );
            }
            other => panic!(
                "expected Output (not a false timeout), got {:?}",
                run_kind(&other)
            ),
        }
    }

    #[test]
    fn bounded_run_missing_binary_is_notfound() {
        let cmd = Command::new("definitely-not-a-real-binary-xyz-123");
        assert!(matches!(
            run_bounded(cmd, Duration::from_secs(1)),
            RocmSmiRun::NotFound
        ));
    }

    #[cfg(unix)]
    fn run_kind(r: &RocmSmiRun) -> &'static str {
        match r {
            RocmSmiRun::Output(_) => "Output",
            RocmSmiRun::NotFound => "NotFound",
            RocmSmiRun::SpawnError(_) => "SpawnError",
            RocmSmiRun::TimedOut => "TimedOut",
        }
    }

    #[test]
    fn worker_spawns_and_drops_cleanly() {
        // Spawn the worker, leave it idle (Dashboard tab not active), and drop
        // it — Drop must signal shutdown and join without hanging. No GPU /
        // serve needed; idle worker performs no fetch.
        let cfg = ConfigState::load(&crate::hipfire::HipfirePaths::discover());
        let worker = DashboardWorker::spawn(cfg);
        // No snapshot before any fetch.
        assert!(worker.snapshot().is_none());
        // set_active toggling must not panic; we don't assert a fetch result
        // (no serve in CI) — just that the worker stays responsive and Drop
        // joins promptly.
        worker.set_active(true);
        worker.set_active(false);
        drop(worker); // joins; would hang the test if shutdown were broken.
    }

    #[test]
    fn rocm_smi_garbage_is_unavailable_not_panic() {
        assert!(matches!(
            parse_rocm_smi_json("not json"),
            VramState::Unavailable(_)
        ));
        assert!(matches!(
            parse_rocm_smi_json("{}"),
            VramState::Unavailable(_)
        ));
        // Object without the VRAM fields → unavailable, not a fake 0.
        assert!(matches!(
            parse_rocm_smi_json(r#"{"card0":{"GPU use (%)":"50"}}"#),
            VramState::Unavailable(_)
        ));
    }
}
