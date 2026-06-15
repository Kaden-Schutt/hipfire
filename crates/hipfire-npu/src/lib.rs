// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! NPU module opt-in contracts.
//!
//! This crate intentionally does not own XDNA runtime dispatch yet. It owns the
//! typed policy boundary for deciding whether an NPU module has the artifacts
//! needed to be admitted by an architecture-specific caller.

use hipfire_cpu::{BackendSelection, DenseFfnBackend, DenseFfnBackendPreference, ModuleInvocation};
use serde_json::{json, Value};

pub const XDNA1_SWIGLU_BACKEND: &str = "npu_xdna1";
pub const NPU_ARTIFACTS_MISSING_FALLBACK: &str = "npu_artifacts_missing";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuModuleTarget {
    Xdna1Swiglu,
}

impl NpuModuleTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Xdna1Swiglu => "xdna1_swiglu",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xdna1ModuleArtifacts {
    pub xclbin: Option<String>,
    pub instr: Option<String>,
}

impl Xdna1ModuleArtifacts {
    pub fn new(xclbin: Option<String>, instr: Option<String>) -> Self {
        Self { xclbin, instr }
    }

    pub fn complete(&self) -> bool {
        self.xclbin.as_deref().is_some_and(|path| !path.is_empty())
            && self.instr.as_deref().is_some_and(|path| !path.is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpuModuleAdmission {
    pub target: NpuModuleTarget,
    pub module_kind: &'static str,
    pub module_id: String,
    pub opt_in: bool,
    pub artifacts_available: bool,
    pub selection: BackendSelection,
}

impl NpuModuleAdmission {
    pub fn admitted(&self) -> bool {
        self.opt_in
            && self.artifacts_available
            && self.selection.selected_backend == DenseFfnBackend::NpuXdna1
    }

    pub fn fallback_reason(&self) -> Option<&'static str> {
        self.selection.fallback_reason
    }
}

pub fn xdna1_swiglu_admission(
    invocation: &ModuleInvocation,
    artifacts: &Xdna1ModuleArtifacts,
) -> NpuModuleAdmission {
    let opt_in =
        invocation.backend_selection().preferred_backend == DenseFfnBackendPreference::NpuOptIn;
    let artifacts_available = artifacts.complete();
    let (selected_backend, fallback_reason) = if opt_in && artifacts_available {
        (DenseFfnBackend::NpuXdna1, None)
    } else if opt_in {
        (
            DenseFfnBackend::GpuProduction,
            Some(NPU_ARTIFACTS_MISSING_FALLBACK),
        )
    } else {
        (DenseFfnBackend::GpuProduction, None)
    };

    NpuModuleAdmission {
        target: NpuModuleTarget::Xdna1Swiglu,
        module_kind: invocation.module_kind(),
        module_id: invocation.module_id().to_string(),
        opt_in,
        artifacts_available,
        selection: BackendSelection::new(
            invocation.backend_selection().preferred_backend,
            selected_backend,
            fallback_reason,
        ),
    }
}

pub fn npu_module_admission_json(admission: &NpuModuleAdmission) -> Value {
    json!({
        "target": admission.target.as_str(),
        "module_kind": admission.module_kind,
        "module_id": admission.module_id,
        "opt_in": admission.opt_in,
        "artifacts_available": admission.artifacts_available,
        "admitted": admission.admitted(),
        "preferred_backend": admission.selection.preferred_backend.as_str(),
        "selected_backend": admission.selection.selected_backend.as_str(),
        "fallback_reason": admission.selection.fallback_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_cpu::{dense_ffn_module_invocation_from_shape, DenseFfnBackendPreference};

    #[test]
    fn xdna1_artifacts_require_both_paths() {
        assert!(!Xdna1ModuleArtifacts::new(None, None).complete());
        assert!(!Xdna1ModuleArtifacts::new(Some("a.xclbin".to_string()), None).complete());
        assert!(Xdna1ModuleArtifacts::new(
            Some("a.xclbin".to_string()),
            Some("a.instr".to_string())
        )
        .complete());
    }

    #[test]
    fn xdna1_swiglu_admission_keeps_npu_opt_in_explicit() {
        let invocation = ModuleInvocation::from(dense_ffn_module_invocation_from_shape(
            3,
            4096,
            11008,
            DenseFfnBackendPreference::NpuOptIn,
            false,
        ));

        let missing = xdna1_swiglu_admission(&invocation, &Xdna1ModuleArtifacts::new(None, None));
        assert!(!missing.admitted());
        assert_eq!(
            missing.selection.preferred_backend,
            DenseFfnBackendPreference::NpuOptIn
        );
        assert_eq!(
            missing.selection.selected_backend,
            DenseFfnBackend::GpuProduction
        );
        assert_eq!(
            missing.fallback_reason(),
            Some(NPU_ARTIFACTS_MISSING_FALLBACK)
        );

        let admitted = xdna1_swiglu_admission(
            &invocation,
            &Xdna1ModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
        );
        assert!(admitted.admitted());
        assert_eq!(
            admitted.selection.selected_backend,
            DenseFfnBackend::NpuXdna1
        );
        assert_eq!(
            npu_module_admission_json(&admitted)["target"],
            "xdna1_swiglu"
        );
    }

    #[test]
    fn xdna1_swiglu_admission_leaves_gpu_path_as_production() {
        let invocation = ModuleInvocation::from(dense_ffn_module_invocation_from_shape(
            3,
            4096,
            11008,
            DenseFfnBackendPreference::GpuProduction,
            false,
        ));

        let admission = xdna1_swiglu_admission(
            &invocation,
            &Xdna1ModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
        );
        assert!(!admission.opt_in);
        assert!(!admission.admitted());
        assert_eq!(
            admission.selection.preferred_backend,
            DenseFfnBackendPreference::GpuProduction
        );
        assert_eq!(
            admission.selection.selected_backend,
            DenseFfnBackend::GpuProduction
        );
        assert_eq!(admission.fallback_reason(), None);
    }
}
