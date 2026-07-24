// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fail-closed XDNA projection control plane.
//!
//! The controller is dormant unless configured, opens amdxdna lazily, retains
//! imported HIP allocations for a model epoch, and poisons the route after any
//! timeout or device failure. It does not own or duplicate model weights.

use crate::{DType, GpuTensor};
use hipfire_config::XdnaMode;
use hsa_bridge::{ExportedDmaBuf, HsaRuntime};
use redline_xdna::{
    resolve_device_path, ArtifactBundle, Binding, BindingAccess, Bo, CommandRing, Device,
    HardwareContext, Program, ProjectionArithmetic, SubmissionTiming,
};
use serde::Serialize;
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_millis(1_000);
const MAX_TIMEOUT: Duration = Duration::from_millis(2_000);
const RECOVERY_HINT: &str = "stop the owning process gracefully (on hipx use gpukill <pid>), \
verify /dev/accel and /dev/kfd have no holders, then manually unbind/rebind only the amdxdna \
NPU function; Hipfire never resets, reloads, warmboots, or reboots hardware";
// Promoted only after the locked-hipx correctness, KLD, and end-to-end
// acceptance bundle passes. Shadow remains available to generate that proof.
const PRODUCTION_INTEROP_CERTIFIED: bool = false;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionKind {
    Qkvza,
    Qkv,
    Wo,
}

#[derive(Clone, Debug)]
pub struct ProjectionAdmission {
    pub kind: ProjectionKind,
    pub model_is_qwen36_a3b: bool,
    pub q8_kv: bool,
    pub weight_dtype: DType,
    pub k: usize,
    pub n: usize,
    pub chunk_tokens: usize,
    pub prompt_tokens: usize,
    pub gpu_count: usize,
    pub tp: usize,
    pub pp: usize,
    pub ep: usize,
    pub plain_prefill: bool,
    pub graph_capture: bool,
    pub dflash_or_tree: bool,
    pub hidden_capture: bool,
    pub mtp_tape: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "route")]
pub enum RouteDecision {
    Hip { reason: String },
    Error { reason: String },
    Shadow,
    Xdna,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ImportedBufferId(u64);

impl ImportedBufferId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImportedBufferSlice {
    pub id: ImportedBufferId,
    pub offset: u64,
    pub length: usize,
    pub access: BufferAccess,
}

impl ImportedBufferSlice {
    pub fn whole(id: ImportedBufferId, length: usize, access: BufferAccess) -> Self {
        Self {
            id,
            offset: 0,
            length,
            access,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct XdnaTimings {
    pub initialization_us: u64,
    pub registration_us: u64,
    pub submission_us: u64,
    pub synchronization_us: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct XdnaRouteCounts {
    pub admitted: u64,
    pub shadow: u64,
    pub submissions: u64,
    pub persistent_reuses: u64,
    pub fallbacks: u64,
    pub timeouts: u64,
    pub errors: u64,
    pub poisons: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct XdnaDiagnostics {
    pub available: bool,
    pub safety_state: &'static str,
    pub artifact_id: Option<String>,
    pub artifact_arithmetic: Option<ProjectionArithmetic>,
    pub device_path: Option<String>,
    pub mode: &'static str,
    pub admission_reason: String,
    pub poison_reason: Option<String>,
    pub recovery_required: bool,
    pub recovery_hint: Option<&'static str>,
    pub automatic_recovery: bool,
    pub owner_pid: u32,
    pub model_epoch: u64,
    pub imported_buffers: usize,
    pub imported_bytes: usize,
    pub quarantined_buffers: usize,
    pub quarantined_bytes: usize,
    pub route_counts: XdnaRouteCounts,
    pub timings: XdnaTimings,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct BufferIdentity {
    pointer: usize,
    bytes: usize,
    offset: usize,
    dtype: &'static str,
    shape: Vec<usize>,
    model_epoch: u64,
}

struct ImportedBuffer {
    #[allow(dead_code)]
    export: ExportedDmaBuf,
    bo: Bo,
    bytes: usize,
}

struct ReadyState {
    #[allow(dead_code)]
    hsa: Arc<HsaRuntime>,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    context: HardwareContext,
    program: Program,
    ring: CommandRing,
    identities: HashMap<BufferIdentity, ImportedBufferId>,
    buffers: HashMap<ImportedBufferId, ImportedBuffer>,
    next_buffer_id: u64,
}

struct PoisonedState {
    reason: String,
    quarantined: Option<Box<ReadyState>>,
}

impl Drop for PoisonedState {
    fn drop(&mut self) {
        if let Some(ready) = self.quarantined.take() {
            // A fault may leave a command active inside the driver. Explicitly
            // destroying its context/BOs from Drop would issue more ioctls on
            // the failed path. Keep the one poisoned runtime quarantined until
            // process exit; the kernel then closes all descriptors.
            std::mem::forget(ready);
        }
    }
}

enum ControllerState {
    Dormant,
    Ready(Box<ReadyState>),
    Unavailable(String),
    Poisoned(PoisonedState),
}

pub struct XdnaController {
    mode: XdnaMode,
    gpu_arch: String,
    gpu_device: i32,
    device_path_override: Option<PathBuf>,
    resolved_device_path: Option<PathBuf>,
    artifact_manifest: PathBuf,
    model_epoch: u64,
    state: ControllerState,
    last_admission_reason: String,
    route_counts: XdnaRouteCounts,
    timings: XdnaTimings,
}

impl XdnaController {
    pub fn from_active_config(gpu_arch: &str, gpu_device: i32) -> Self {
        let mode = hipfire_config::process_value("HIPFIRE_XDNA")
            .as_deref()
            .and_then(XdnaMode::parse)
            .unwrap_or_default();
        let device_path_override = hipfire_config::process_value("HIPFIRE_XDNA_DEVICE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let artifact_manifest = hipfire_config::process_value("HIPFIRE_XDNA_ARTIFACT")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../artifacts/xdna/gfx1151/q8-w8a16/manifest.json")
            });
        Self::new(
            mode,
            gpu_arch,
            gpu_device,
            device_path_override,
            artifact_manifest,
        )
    }

    pub fn new(
        mode: XdnaMode,
        gpu_arch: impl Into<String>,
        gpu_device: i32,
        device_path_override: Option<PathBuf>,
        artifact_manifest: PathBuf,
    ) -> Self {
        Self {
            mode,
            gpu_arch: gpu_arch.into(),
            gpu_device,
            device_path_override,
            resolved_device_path: None,
            artifact_manifest,
            model_epoch: 0,
            state: ControllerState::Dormant,
            last_admission_reason: if mode == XdnaMode::Off {
                "configured off".into()
            } else {
                "not evaluated".into()
            },
            route_counts: XdnaRouteCounts::default(),
            timings: XdnaTimings::default(),
        }
    }

    pub fn mode(&self) -> XdnaMode {
        self.mode
    }

    pub fn begin_model_epoch(&mut self) -> u64 {
        self.model_epoch = self.model_epoch.wrapping_add(1).max(1);
        if let ControllerState::Ready(ready) = &mut self.state {
            ready.identities.clear();
            ready.buffers.clear();
        }
        self.model_epoch
    }

    pub fn evaluate(&mut self, request: &ProjectionAdmission) -> RouteDecision {
        let rejection = self.rejection_reason(request);
        if let Some(reason) = rejection {
            self.last_admission_reason = reason.clone();
            if self.mode == XdnaMode::Force {
                self.route_counts.errors += 1;
                return RouteDecision::Error { reason };
            }
            self.route_counts.fallbacks += 1;
            return RouteDecision::Hip { reason };
        }
        self.route_counts.admitted += 1;
        self.last_admission_reason = "eligible".into();
        match self.mode {
            XdnaMode::Off => unreachable!("off is handled by rejection_reason"),
            XdnaMode::Shadow => {
                self.route_counts.shadow += 1;
                RouteDecision::Shadow
            }
            XdnaMode::Auto | XdnaMode::Force => RouteDecision::Xdna,
        }
    }

    fn rejection_reason(&self, request: &ProjectionAdmission) -> Option<String> {
        if self.mode == XdnaMode::Off {
            return Some("configured off".into());
        }
        if matches!(self.mode, XdnaMode::Auto | XdnaMode::Force) && !PRODUCTION_INTEROP_CERTIFIED {
            return Some("automatic XDNA adoption has not passed the locked-hipx gate".into());
        }
        match &self.state {
            ControllerState::Unavailable(reason) => {
                return Some(format!("unavailable: {reason}"));
            }
            ControllerState::Poisoned(poisoned) => {
                return Some(format!("poisoned: {}", poisoned.reason));
            }
            ControllerState::Dormant | ControllerState::Ready(_) => {}
        }
        if self.gpu_arch != "gfx1151" {
            return Some(format!("GPU architecture {} is not gfx1151", self.gpu_arch));
        }
        if self.gpu_device != 0 {
            return Some("XDNA route requires the active logical GPU to be device 0".into());
        }
        if request.gpu_count != 1 || request.tp != 1 || request.pp != 1 || request.ep != 1 {
            return Some("XDNA route requires single-GPU TP=PP=EP=1".into());
        }
        if !request.model_is_qwen36_a3b {
            return Some("model is not Qwen3.6 A3B".into());
        }
        if !request.q8_kv {
            return Some("KV cache is not Q8".into());
        }
        if request.weight_dtype != DType::Q8_0 {
            return Some(format!(
                "projection weight dtype {:?} is not Q8_0",
                request.weight_dtype
            ));
        }
        if request.k != 2048 {
            return Some(format!("projection K={} is not 2048", request.k));
        }
        if request.chunk_tokens == 0 || request.chunk_tokens > 256 {
            return Some(format!(
                "chunk batch {} is outside 1..=256",
                request.chunk_tokens
            ));
        }
        if !(512..=4096).contains(&request.prompt_tokens) {
            return Some(format!(
                "prompt length {} is outside 512..=4096",
                request.prompt_tokens
            ));
        }
        let output_supported = match request.kind {
            ProjectionKind::Wo => request.n == 2048,
            ProjectionKind::Qkv => request.n == 5120,
            ProjectionKind::Qkvza => request.n == 8224,
        };
        if !output_supported {
            return Some(format!(
                "{:?} output width {} is unsupported",
                request.kind, request.n
            ));
        }
        if !request.plain_prefill {
            return Some("request is not plain prefill".into());
        }
        if request.graph_capture {
            return Some("graph capture is active".into());
        }
        if request.dflash_or_tree {
            return Some("DFlash/tree verification is active".into());
        }
        if request.hidden_capture {
            return Some("hidden capture is active".into());
        }
        if request.mtp_tape {
            return Some("MTP tape is active".into());
        }
        None
    }

    pub fn register_tensor(
        &mut self,
        tensor: &GpuTensor,
    ) -> std::result::Result<ImportedBufferId, String> {
        if tensor.buf.as_ptr().is_null() || tensor.buf.size() == 0 {
            return Err("cannot register an empty GPU tensor".into());
        }
        let dtype = match tensor.dtype {
            DType::Q8_0 => "q8_0",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            other => return Err(format!("unsupported XDNA tensor dtype {other:?}")),
        };
        let identity = BufferIdentity {
            pointer: tensor.buf.as_ptr() as usize,
            bytes: tensor.buf.size(),
            offset: 0,
            dtype,
            shape: tensor.shape.clone(),
            model_epoch: self.model_epoch,
        };
        self.ensure_ready()?;
        let ControllerState::Ready(ready) = &mut self.state else {
            unreachable!("ensure_ready leaves the controller ready on success");
        };
        if let Some(id) = ready.identities.get(&identity).copied() {
            self.route_counts.persistent_reuses += 1;
            return Ok(id);
        }

        let started = std::time::Instant::now();
        let export = unsafe {
            ready
                .hsa
                .export_dmabuf(tensor.buf.as_ptr().cast_const(), tensor.buf.size())
        }
        .map_err(|error| error.to_string())?;
        let export_offset = export.offset();
        let bo = ready
            .device
            .import_dmabuf(export.as_raw_fd(), export_offset, tensor.buf.size())
            .map_err(|error| error.to_string())?;
        let id = ImportedBufferId(ready.next_buffer_id);
        ready.next_buffer_id = ready.next_buffer_id.wrapping_add(1).max(1);
        ready.identities.insert(identity, id);
        ready.buffers.insert(
            id,
            ImportedBuffer {
                export,
                bo,
                bytes: tensor.buf.size(),
            },
        );
        self.timings.registration_us += started.elapsed().as_micros() as u64;
        Ok(id)
    }

    pub fn submit_projection(
        &mut self,
        request: &ProjectionAdmission,
        bindings: &[ImportedBufferSlice],
        timeout: Option<Duration>,
    ) -> std::result::Result<SubmissionTiming, String> {
        if let Some(reason) = self.rejection_reason(request) {
            self.note_fallback(reason.clone());
            return Err(reason);
        }
        self.ensure_ready()?;
        let k = u32::try_from(request.k)
            .map_err(|_| format!("projection K={} does not fit the artifact ABI", request.k))?;
        let n = u32::try_from(request.n).map_err(|_| {
            format!(
                "projection output width {} does not fit the artifact ABI",
                request.n
            )
        })?;
        let batch = u32::try_from(request.chunk_tokens).map_err(|_| {
            format!(
                "projection batch {} does not fit the artifact ABI",
                request.chunk_tokens
            )
        })?;
        let supported = match &self.state {
            ControllerState::Ready(ready) => ready.program.supports_shape(k, n, batch),
            _ => unreachable!("ensure_ready leaves the controller ready on success"),
        };
        if !supported {
            let reason = format!(
                "artifact does not declare K={} N={} batch={}{}",
                request.k,
                request.n,
                request.chunk_tokens,
                if request.chunk_tokens < 256 {
                    " with masked-tail support"
                } else {
                    ""
                }
            );
            self.note_fallback(reason.clone());
            return Err(reason);
        }
        self.submit_imported(bindings, timeout)
    }

    fn submit_imported(
        &mut self,
        bindings: &[ImportedBufferSlice],
        timeout: Option<Duration>,
    ) -> std::result::Result<SubmissionTiming, String> {
        let timeout = validate_timeout(timeout)?;
        self.ensure_ready()?;
        let result = (|| -> std::result::Result<SubmissionTiming, String> {
            let ControllerState::Ready(ready) = &self.state else {
                unreachable!("ensure_ready leaves the controller ready on success");
            };
            let mut xdna_bindings = Vec::with_capacity(bindings.len());
            let sync_started = std::time::Instant::now();
            for binding in bindings {
                let imported = ready
                    .buffers
                    .get(&binding.id)
                    .ok_or_else(|| format!("unknown imported buffer {}", binding.id.get()))?;
                // Even write-only outputs are synchronized before submission
                // so their pages and dma-buf attachment are resident before
                // the NPU's first store.
                imported
                    .bo
                    .sync_to_device(binding.offset, binding.length)
                    .map_err(|error| error.to_string())?;
                xdna_bindings.push(Binding {
                    bo: &imported.bo,
                    offset: binding.offset,
                    length: binding.length,
                    access: match binding.access {
                        BufferAccess::Read => BindingAccess::Read,
                        BufferAccess::Write => BindingAccess::Write,
                        BufferAccess::ReadWrite => BindingAccess::ReadWrite,
                    },
                });
            }
            self.timings.synchronization_us += sync_started.elapsed().as_micros() as u64;
            let submit_started = std::time::Instant::now();
            let ticket = ready
                .ring
                .submit(&ready.program, &xdna_bindings)
                .map_err(|error| error.to_string())?;
            self.timings.submission_us += submit_started.elapsed().as_micros() as u64;
            let sync_started = std::time::Instant::now();
            let result = ticket.wait(timeout).map_err(|error| error.to_string());
            if result.is_ok() {
                for binding in bindings.iter().filter(|binding| {
                    matches!(
                        binding.access,
                        BufferAccess::Write | BufferAccess::ReadWrite
                    )
                }) {
                    let imported = ready
                        .buffers
                        .get(&binding.id)
                        .expect("binding was resolved before submission");
                    imported
                        .bo
                        .invalidate_cpu_cache(binding.offset, binding.length)
                        .map_err(|error| error.to_string())?;
                }
            }
            self.timings.synchronization_us += sync_started.elapsed().as_micros() as u64;
            result
        })();
        match result {
            Ok(timing) => {
                self.route_counts.submissions += 1;
                Ok(timing)
            }
            Err(error) => {
                if error.contains("timed out") {
                    self.route_counts.timeouts += 1;
                } else {
                    self.route_counts.errors += 1;
                }
                self.poison(error.clone());
                Err(error)
            }
        }
    }

    pub fn note_fallback(&mut self, reason: impl Into<String>) {
        self.route_counts.fallbacks += 1;
        self.last_admission_reason = reason.into();
    }

    pub fn poison(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        if let ControllerState::Poisoned(poisoned) = &self.state {
            self.last_admission_reason = format!("poisoned: {}", poisoned.reason);
            return;
        }
        let previous = std::mem::replace(&mut self.state, ControllerState::Dormant);
        let quarantined = match previous {
            ControllerState::Ready(ready) => Some(ready),
            ControllerState::Dormant | ControllerState::Unavailable(_) => None,
            ControllerState::Poisoned(_) => unreachable!("poisoned state returned above"),
        };
        self.last_admission_reason = format!("poisoned: {reason}");
        self.route_counts.poisons += 1;
        self.state = ControllerState::Poisoned(PoisonedState {
            reason,
            quarantined,
        });
    }

    fn ensure_ready(&mut self) -> std::result::Result<(), String> {
        match &self.state {
            ControllerState::Ready(_) => return Ok(()),
            ControllerState::Unavailable(reason) => {
                return Err(reason.clone());
            }
            ControllerState::Poisoned(poisoned) => return Err(poisoned.reason.clone()),
            ControllerState::Dormant => {}
        }
        if self.mode == XdnaMode::Off {
            return Err("XDNA is configured off".into());
        }
        let started = std::time::Instant::now();
        let device_path = match resolve_device_path(self.device_path_override.as_deref()) {
            Ok(path) => {
                self.resolved_device_path = Some(path.clone());
                path
            }
            Err(error) => {
                let reason = error.to_string();
                self.timings.initialization_us += started.elapsed().as_micros() as u64;
                self.last_admission_reason = format!("unavailable: {reason}");
                self.state = ControllerState::Unavailable(reason.clone());
                return Err(reason);
            }
        };
        let initialized = (|| {
            let hsa = HsaRuntime::load().map_err(|error| error.to_string())?;
            let device = Device::open(&device_path).map_err(|error| error.to_string())?;
            let bundle = ArtifactBundle::load(
                &self.artifact_manifest,
                &self.gpu_arch,
                device.metadata().firmware,
            )
            .map_err(|error| error.to_string())?;
            let context = device
                .create_context(2048)
                .map_err(|error| error.to_string())?;
            let program = context
                .load_program(&bundle)
                .map_err(|error| error.to_string())?;
            let ring = context.command_ring(4).map_err(|error| error.to_string())?;
            Ok::<_, String>(ReadyState {
                hsa,
                device,
                context,
                program,
                ring,
                identities: HashMap::new(),
                buffers: HashMap::new(),
                next_buffer_id: 1,
            })
        })();
        self.timings.initialization_us += started.elapsed().as_micros() as u64;
        match initialized {
            Ok(ready) => {
                self.state = ControllerState::Ready(Box::new(ready));
                Ok(())
            }
            Err(reason) => {
                self.last_admission_reason = format!("unavailable: {reason}");
                self.state = ControllerState::Unavailable(reason.clone());
                Err(reason)
            }
        }
    }

    pub fn diagnostics(&self) -> XdnaDiagnostics {
        let (
            available,
            safety_state,
            artifact_id,
            artifact_arithmetic,
            poison_reason,
            imported_buffers,
            imported_bytes,
            quarantined_buffers,
            quarantined_bytes,
        ) = match &self.state {
            ControllerState::Ready(ready) => (
                true,
                "ready",
                Some(ready.program.artifact_id().to_string()),
                Some(ready.program.arithmetic()),
                None,
                ready.buffers.len(),
                ready.buffers.values().map(|buffer| buffer.bytes).sum(),
                0,
                0,
            ),
            ControllerState::Poisoned(poisoned) => {
                let quarantined_buffers = poisoned
                    .quarantined
                    .as_ref()
                    .map_or(0, |ready| ready.buffers.len());
                let quarantined_bytes = poisoned.quarantined.as_ref().map_or(0, |ready| {
                    ready.buffers.values().map(|buffer| buffer.bytes).sum()
                });
                (
                    false,
                    "quarantined",
                    poisoned
                        .quarantined
                        .as_ref()
                        .map(|ready| ready.program.artifact_id().to_string()),
                    poisoned
                        .quarantined
                        .as_ref()
                        .map(|ready| ready.program.arithmetic()),
                    Some(poisoned.reason.clone()),
                    0,
                    0,
                    quarantined_buffers,
                    quarantined_bytes,
                )
            }
            ControllerState::Dormant => (false, "dormant", None, None, None, 0, 0, 0, 0),
            ControllerState::Unavailable(_) => (false, "unavailable", None, None, None, 0, 0, 0, 0),
        };
        XdnaDiagnostics {
            available,
            safety_state,
            artifact_id,
            artifact_arithmetic,
            device_path: self
                .resolved_device_path
                .as_ref()
                .or(self.device_path_override.as_ref())
                .map(|path| path.display().to_string()),
            mode: self.mode.as_str(),
            admission_reason: self.last_admission_reason.clone(),
            poison_reason,
            recovery_required: matches!(&self.state, ControllerState::Poisoned(_)),
            recovery_hint: matches!(&self.state, ControllerState::Poisoned(_))
                .then_some(RECOVERY_HINT),
            automatic_recovery: false,
            owner_pid: std::process::id(),
            model_epoch: self.model_epoch,
            imported_buffers,
            imported_bytes,
            quarantined_buffers,
            quarantined_bytes,
            route_counts: self.route_counts.clone(),
            timings: self.timings.clone(),
        }
    }
}

fn validate_timeout(timeout: Option<Duration>) -> std::result::Result<Duration, String> {
    let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
    if timeout.is_zero() || timeout > MAX_TIMEOUT {
        return Err(format!(
            "XDNA submission timeout must be in 1..={} ms, got {} ms",
            MAX_TIMEOUT.as_millis(),
            timeout.as_millis()
        ));
    }
    Ok(timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ProjectionAdmission {
        ProjectionAdmission {
            kind: ProjectionKind::Wo,
            model_is_qwen36_a3b: true,
            q8_kv: true,
            weight_dtype: DType::Q8_0,
            k: 2048,
            n: 2048,
            chunk_tokens: 256,
            prompt_tokens: 512,
            gpu_count: 1,
            tp: 1,
            pp: 1,
            ep: 1,
            plain_prefill: true,
            graph_capture: false,
            dflash_or_tree: false,
            hidden_capture: false,
            mtp_tape: false,
        }
    }

    fn controller(mode: XdnaMode) -> XdnaController {
        XdnaController::new(
            mode,
            "gfx1151",
            0,
            Some("/dev/accel/accel0".into()),
            "missing-manifest.json".into(),
        )
    }

    #[test]
    fn off_is_always_hip() {
        assert!(matches!(
            controller(XdnaMode::Off).evaluate(&request()),
            RouteDecision::Hip { .. }
        ));
    }

    #[test]
    fn shadow_admission_accepts_only_certified_shape() {
        assert_eq!(
            controller(XdnaMode::Shadow).evaluate(&request()),
            RouteDecision::Shadow
        );
        let mut bad = request();
        bad.prompt_tokens = 256;
        assert!(matches!(
            controller(XdnaMode::Shadow).evaluate(&bad),
            RouteDecision::Hip { reason } if reason.contains("512..=4096")
        ));
    }

    #[test]
    fn auto_is_closed_until_the_end_to_end_gate_passes() {
        assert!(matches!(
            controller(XdnaMode::Auto).evaluate(&request()),
            RouteDecision::Hip { reason } if reason.contains("locked-hipx")
        ));
    }

    #[test]
    fn graph_and_parallel_routes_are_rejected() {
        let mut graph = request();
        graph.graph_capture = true;
        assert!(matches!(
            controller(XdnaMode::Shadow).evaluate(&graph),
            RouteDecision::Hip { reason } if reason.contains("graph")
        ));

        let mut tp = request();
        tp.tp = 2;
        assert!(matches!(
            controller(XdnaMode::Shadow).evaluate(&tp),
            RouteDecision::Hip { reason } if reason.contains("single-GPU")
        ));
    }

    #[test]
    fn poison_is_sticky() {
        let mut controller = controller(XdnaMode::Shadow);
        controller.poison("timeout");
        assert!(matches!(
            controller.evaluate(&request()),
            RouteDecision::Hip { reason } if reason.contains("poisoned")
        ));
        assert_eq!(
            controller.diagnostics().poison_reason.as_deref(),
            Some("timeout")
        );
        assert_eq!(controller.diagnostics().safety_state, "quarantined");
        assert!(controller.diagnostics().recovery_required);
        assert!(!controller.diagnostics().automatic_recovery);
        assert!(controller
            .diagnostics()
            .recovery_hint
            .is_some_and(|hint| hint.contains("gpukill")));
        assert_eq!(controller.diagnostics().route_counts.poisons, 1);

        controller.poison("second error");
        assert_eq!(
            controller.diagnostics().poison_reason.as_deref(),
            Some("timeout"),
            "the first fault remains authoritative"
        );
        assert_eq!(controller.diagnostics().route_counts.poisons, 1);
    }

    #[test]
    fn model_epoch_never_reanimates_a_poisoned_controller() {
        let mut controller = controller(XdnaMode::Shadow);
        assert_eq!(controller.begin_model_epoch(), 1);
        controller.poison("device fault");
        assert_eq!(controller.begin_model_epoch(), 2);
        assert_eq!(controller.diagnostics().model_epoch, 2);
        assert_eq!(controller.diagnostics().safety_state, "quarantined");
        assert_eq!(
            controller.diagnostics().poison_reason.as_deref(),
            Some("device fault")
        );
    }

    #[test]
    fn force_turns_ineligibility_into_an_error() {
        let mut bad = request();
        bad.k = 4096;
        assert!(matches!(
            controller(XdnaMode::Force).evaluate(&bad),
            RouteDecision::Error { reason } if reason.contains("locked-hipx")
        ));
    }

    #[test]
    fn submission_timeout_is_tightly_bounded() {
        assert_eq!(validate_timeout(None).unwrap(), DEFAULT_TIMEOUT);
        assert_eq!(
            validate_timeout(Some(Duration::from_millis(1))).unwrap(),
            Duration::from_millis(1)
        );
        assert!(validate_timeout(Some(Duration::ZERO)).is_err());
        assert!(validate_timeout(Some(MAX_TIMEOUT + Duration::from_millis(1))).is_err());
    }
}
