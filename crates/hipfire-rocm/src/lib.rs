// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! ROCm backend contracts and evidence adapters.
//!
//! This crate intentionally does not own HIP dispatch yet. It gives callers a
//! typed boundary for describing which existing ROCm path handled a module.

use hipfire_cpu::{
    backend_selection_json, BackendSelection, DenseFfnBackend, DenseFfnBackendPreference,
    DenseFfnModuleInvocation, DiffStats, ProjectionModuleInvocation,
};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmDeviceIdentity {
    pub device_id: i32,
    pub arch: String,
    pub integrated: bool,
}

impl RocmDeviceIdentity {
    pub fn new(device_id: i32, arch: impl Into<String>, integrated: bool) -> Self {
        Self {
            device_id,
            arch: arch.into(),
            integrated,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmModuleKind {
    DenseFfnSwigluDown,
    ProjectionResidual,
}

impl RocmModuleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DenseFfnSwigluDown => "dense_ffn_swiglu_down",
            Self::ProjectionResidual => "projection_residual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocmBackendPath {
    HipRdnaCompute,
    CpuOracleBypass,
    NpuHybridFallback,
}

impl RocmBackendPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HipRdnaCompute => "hip_rdna_compute",
            Self::CpuOracleBypass => "cpu_oracle_bypass",
            Self::NpuHybridFallback => "npu_hybrid_fallback",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmModuleContract {
    pub module_kind: RocmModuleKind,
    pub module_id: String,
    pub layer_idx: usize,
    pub kernel_path: &'static str,
    pub device: RocmDeviceIdentity,
    pub mutates_residual: bool,
}

#[derive(Debug, Clone)]
pub struct RocmModuleOutput {
    pub contract: RocmModuleContract,
    pub preferred_backend: DenseFfnBackendPreference,
    pub selected_backend: DenseFfnBackend,
    pub backend_path: RocmBackendPath,
    pub drift: Option<DiffStats>,
    pub fallback_reason: Option<&'static str>,
}

pub fn rocm_device_identity(
    device_id: i32,
    arch: impl Into<String>,
    integrated: bool,
) -> RocmDeviceIdentity {
    RocmDeviceIdentity::new(device_id, arch, integrated)
}

pub fn rocm_dense_ffn_module_contract(
    invocation: &DenseFfnModuleInvocation,
    device: RocmDeviceIdentity,
    kernel_path: &'static str,
) -> RocmModuleContract {
    RocmModuleContract {
        module_kind: RocmModuleKind::DenseFfnSwigluDown,
        module_id: invocation.contract.module_id.clone(),
        layer_idx: invocation.contract.layer_idx,
        kernel_path,
        device,
        mutates_residual: invocation.contract.state.mutates_residual,
    }
}

pub fn rocm_projection_module_contract(
    invocation: &ProjectionModuleInvocation,
    device: RocmDeviceIdentity,
    kernel_path: &'static str,
) -> RocmModuleContract {
    RocmModuleContract {
        module_kind: RocmModuleKind::ProjectionResidual,
        module_id: invocation.contract.module_id.clone(),
        layer_idx: invocation.contract.layer_idx,
        kernel_path,
        device,
        mutates_residual: invocation.contract.state.mutates_residual,
    }
}

pub fn rocm_backend_path_for_selected_backend(
    selected_backend: DenseFfnBackend,
) -> RocmBackendPath {
    match selected_backend {
        DenseFfnBackend::CpuOracle => RocmBackendPath::CpuOracleBypass,
        DenseFfnBackend::GpuProduction => RocmBackendPath::HipRdnaCompute,
        DenseFfnBackend::NpuXdna => RocmBackendPath::NpuHybridFallback,
    }
}

pub fn rocm_dense_ffn_module_output(
    invocation: &DenseFfnModuleInvocation,
    device: RocmDeviceIdentity,
    kernel_path: &'static str,
    drift: Option<DiffStats>,
) -> RocmModuleOutput {
    let backend_path = rocm_backend_path_for_selected_backend(invocation.selected_backend);
    RocmModuleOutput {
        contract: rocm_dense_ffn_module_contract(invocation, device, kernel_path),
        preferred_backend: invocation.contract.preferred_backend,
        selected_backend: invocation.selected_backend,
        backend_path,
        drift,
        fallback_reason: invocation.fallback_reason,
    }
}

pub fn rocm_projection_module_output(
    invocation: &ProjectionModuleInvocation,
    device: RocmDeviceIdentity,
    kernel_path: &'static str,
) -> RocmModuleOutput {
    let backend_path = rocm_backend_path_for_selected_backend(invocation.selected_backend);
    RocmModuleOutput {
        contract: rocm_projection_module_contract(invocation, device, kernel_path),
        preferred_backend: invocation.contract.preferred_backend,
        selected_backend: invocation.selected_backend,
        backend_path,
        drift: None,
        fallback_reason: invocation.fallback_reason,
    }
}

pub fn diff_stats_json(stats: DiffStats) -> Value {
    json!({
        "n": stats.n,
        "max_abs": stats.max_abs,
        "mean_abs": stats.mean_abs,
        "rms": stats.rms,
        "n_nan": stats.n_nan,
        "n_inf": stats.n_inf,
    })
}

pub fn rocm_module_output_json(output: &RocmModuleOutput) -> Value {
    let selection = backend_selection_json(rocm_module_backend_selection(output));
    let mut value = json!({
        "module_kind": output.contract.module_kind.as_str(),
        "module_id": output.contract.module_id,
        "layer_idx": output.contract.layer_idx,
        "kernel_path": output.contract.kernel_path,
        "selected_backend": selection["selected_backend"].clone(),
        "backend_path": output.backend_path.as_str(),
        "fallback_reason": selection["fallback_reason"].clone(),
        "mutates_residual": output.contract.mutates_residual,
        "device": {
            "device_id": output.contract.device.device_id,
            "arch": output.contract.device.arch,
            "integrated": output.contract.device.integrated,
        },
    });
    if let Some(drift) = output.drift {
        value["drift"] = diff_stats_json(drift);
    }
    value
}

pub fn rocm_module_backend_selection(output: &RocmModuleOutput) -> BackendSelection {
    BackendSelection::new(
        output.preferred_backend,
        output.selected_backend,
        output.fallback_reason,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_cpu::{
        dense_ffn_module_invocation_from_shape, DenseFfnBackend, DenseFfnBackendPreference,
    };

    #[test]
    fn dense_ffn_rocm_output_records_device_and_kernel_path() {
        let invocation = dense_ffn_module_invocation_from_shape(
            2,
            4096,
            11008,
            DenseFfnBackendPreference::GpuProduction,
            false,
        );
        let output = rocm_dense_ffn_module_output(
            &invocation,
            rocm_device_identity(1, "gfx1100", false),
            "weight_gemv_swiglu_residual",
            None,
        );
        assert_eq!(output.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(output.backend_path, RocmBackendPath::HipRdnaCompute);
        assert_eq!(output.contract.device.arch, "gfx1100");

        let json = rocm_module_output_json(&output);
        assert_eq!(json["module_kind"], "dense_ffn_swiglu_down");
        assert_eq!(json["module_id"], "qwen35.layers.2.mlp.swiglu_down");
        assert_eq!(json["kernel_path"], "weight_gemv_swiglu_residual");
        assert_eq!(json["device"]["device_id"], 1);
        assert_eq!(json["device"]["arch"], "gfx1100");
    }

    #[test]
    fn backend_path_tracks_cpu_and_npu_bypass_cases() {
        assert_eq!(
            rocm_backend_path_for_selected_backend(DenseFfnBackend::CpuOracle),
            RocmBackendPath::CpuOracleBypass
        );
        assert_eq!(
            rocm_backend_path_for_selected_backend(DenseFfnBackend::NpuXdna),
            RocmBackendPath::NpuHybridFallback
        );
    }

    #[test]
    fn rocm_output_preserves_preferred_backend_for_npu_fallback_selection() {
        let invocation = dense_ffn_module_invocation_from_shape(
            4,
            4096,
            11008,
            DenseFfnBackendPreference::NpuOptIn,
            false,
        );
        let output = rocm_dense_ffn_module_output(
            &invocation,
            rocm_device_identity(0, "gfx1151", false),
            "weight_gemv_swiglu_residual",
            None,
        );
        let selection = rocm_module_backend_selection(&output);
        assert_eq!(
            selection.preferred_backend,
            DenseFfnBackendPreference::NpuOptIn
        );
        assert_eq!(selection.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(selection.oracle_backend, DenseFfnBackend::CpuOracle);
        assert_eq!(selection.fallback_reason, Some("npu_backend_unavailable"));

        let json = rocm_module_output_json(&output);
        assert_eq!(json["selected_backend"], "gpu_production");
        assert_eq!(json["fallback_reason"], "npu_backend_unavailable");
        assert!(json.get("preferred_backend").is_none());
    }
}
