// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! CPU BF16 oracle for the dense Qwen3.5 FFN SwiGLU/down epilogue.

use std::sync::OnceLock;

#[allow(unused_imports)]
pub(crate) use hipfire_cpu::{
    attention_wo_module_id, attention_wo_residual_contract_from_shape,
    attention_wo_residual_invocation_from_shape, bf16_bits_to_f32, decode_w_down_shadow,
    dense_ffn_backend_decision, dense_ffn_module_evidence, dense_ffn_module_evidence_json,
    dense_ffn_module_id, dense_ffn_module_invocation, dense_ffn_module_invocation_from_shape,
    dense_ffn_module_output, dense_ffn_module_output_json, dense_ffn_swiglu_down_contract,
    dense_ffn_swiglu_down_contract_from_shape, diff_stats, diff_stats_json, f32_to_bf16_bits_rne,
    projection_module_output, projection_module_output_json, round_f32_to_bf16,
    swiglu_down_bf16_cpu, Bf16DownShadow, DenseFfnBackend, DenseFfnBackendPreference,
    DenseFfnModuleContract, DenseFfnModuleEvidence, DenseFfnModuleInvocation, DenseFfnModuleOutput,
    DenseFfnStateContract, DenseFfnTensorContract, DiffStats, ModuleInvocation,
    ProjectionModuleContract, ProjectionModuleInvocation, ProjectionModuleOutput,
    ProjectionStateContract, ProjectionTensorContract,
};
pub(crate) use hipfire_npu::{xdna_swiglu_admission, XdnaModuleArtifacts};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnBf16Mode {
    Off,
    Compare,
    Cpu,
    Xdna1,
}

#[derive(Debug)]
pub struct FfnBf16Config {
    pub mode: FfnBf16Mode,
    pub layer: LayerSelect,
    pub trace: bool,
    /// Path to the xclbin file for the NPU SwiGLU kernel.
    /// Set via `HIPFIRE_QWEN35_XDNA1_XCLBIN`. Required when mode=xdna1.
    pub xdna1_xclbin: Option<String>,
    /// Path to the NPU instruction binary for the SwiGLU operation.
    /// Set via `HIPFIRE_QWEN35_XDNA1_INSTR`. Required when mode=xdna1.
    pub xdna1_instr: Option<String>,
    pub xdna1_artifacts: XdnaModuleArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSelect {
    One(usize),
    All,
}

pub fn config() -> &'static FfnBf16Config {
    static CONFIG: OnceLock<FfnBf16Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mode = match std::env::var("HIPFIRE_QWEN35_FFN_BF16")
            .unwrap_or_else(|_| "off".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "compare" => FfnBf16Mode::Compare,
            "cpu" => FfnBf16Mode::Cpu,
            "xdna1" => FfnBf16Mode::Xdna1,
            "off" | "0" | "false" | "" => FfnBf16Mode::Off,
            other => {
                panic!("HIPFIRE_QWEN35_FFN_BF16 must be off|compare|cpu|xdna1, got {other:?}")
            }
        };
        let layer = match std::env::var("HIPFIRE_QWEN35_FFN_BF16_LAYER")
            .unwrap_or_else(|_| "0".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "all" => LayerSelect::All,
            s => LayerSelect::One(
                s.parse::<usize>()
                    .expect("HIPFIRE_QWEN35_FFN_BF16_LAYER must be <n>|all"),
            ),
        };
        let trace = std::env::var("HIPFIRE_QWEN35_FFN_BF16_TRACE")
            .map(|v| !matches!(v.trim(), "" | "0" | "false" | "False" | "FALSE"))
            .unwrap_or(false);
        let xdna1_xclbin = std::env::var("HIPFIRE_QWEN35_XDNA1_XCLBIN").ok();
        let xdna1_instr = std::env::var("HIPFIRE_QWEN35_XDNA1_INSTR").ok();
        FfnBf16Config {
            mode,
            layer,
            trace,
            xdna1_artifacts: XdnaModuleArtifacts::new(xdna1_xclbin.clone(), xdna1_instr.clone()),
            xdna1_xclbin,
            xdna1_instr,
        }
    })
}

pub fn dense_ffn_backend_preference_for_mode(
    mode: FfnBf16Mode,
) -> Option<DenseFfnBackendPreference> {
    match mode {
        FfnBf16Mode::Off => None,
        FfnBf16Mode::Compare => Some(DenseFfnBackendPreference::GpuProduction),
        FfnBf16Mode::Cpu => Some(DenseFfnBackendPreference::CpuOracle),
        FfnBf16Mode::Xdna1 => Some(DenseFfnBackendPreference::NpuOptIn),
    }
}

pub(crate) fn xdna1_dense_ffn_module_invocation_from_shape(
    layer_idx: usize,
    residual_len: usize,
    hidden_size: usize,
    artifacts: &XdnaModuleArtifacts,
) -> DenseFfnModuleInvocation {
    let base = dense_ffn_module_invocation_from_shape(
        layer_idx,
        residual_len,
        hidden_size,
        DenseFfnBackendPreference::NpuOptIn,
        false,
    );
    let admission = xdna_swiglu_admission(&ModuleInvocation::from(base.clone()), artifacts);
    DenseFfnModuleInvocation {
        selected_backend: admission.selection.selected_backend,
        fallback_reason: admission.selection.fallback_reason,
        ..base
    }
}

pub fn enabled() -> bool {
    config().mode != FfnBf16Mode::Off
}

pub fn layer_selected(layer_idx: usize) -> bool {
    match config().layer {
        LayerSelect::One(n) => layer_idx == n,
        LayerSelect::All => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_ties_to_even() {
        let one = 1.0f32;
        let half_ulp = f32::from_bits(one.to_bits() + 0x8000);
        assert_eq!(f32_to_bf16_bits_rne(half_ulp), 0x3f80);

        let odd_lsb = f32::from_bits(0x3f81_0000);
        let odd_half_ulp = f32::from_bits(odd_lsb.to_bits() + 0x8000);
        assert_eq!(f32_to_bf16_bits_rne(odd_half_ulp), 0x3f82);
    }

    #[test]
    fn decode_bf16_and_f32_shadow() {
        let f32_bytes: Vec<u8> = [1.0001f32, -2.25, 3.5, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f32_shadow = decode_w_down_shadow(&f32_bytes, 2, 2, 2).unwrap();
        assert_eq!(f32_shadow.w_down[0], round_f32_to_bf16(1.0001));

        let bf16_bytes: Vec<u8> = [0x3f80u16, 0xc000, 0x4060, 0x4080]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16_shadow = decode_w_down_shadow(&bf16_bytes, 16, 2, 2).unwrap();
        assert_eq!(bf16_shadow.w_down, vec![1.0, -2.0, 3.5, 4.0]);
    }

    #[test]
    fn tiny_swiglu_down_bf16_cpu() {
        let shadow = Bf16DownShadow {
            w_down: vec![1.0, 2.0, -1.0, 0.5],
            m: 2,
            k: 2,
        };
        let out = swiglu_down_bf16_cpu(&[0.0, 1.0], &[2.0, -3.0], &[0.5, -0.5], &shadow);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dense_ffn_contract_names_oracle_gpu_and_opt_in_npu_backends() {
        let shadow = Bf16DownShadow {
            w_down: vec![1.0, 2.0, -1.0, 0.5],
            m: 2,
            k: 2,
        };
        let contract =
            dense_ffn_swiglu_down_contract(3, &shadow, DenseFfnBackendPreference::GpuProduction);
        assert_eq!(contract.module_kind, "qwen35_dense_ffn_swiglu_down");
        assert_eq!(contract.module_id, "qwen35.layers.3.mlp.swiglu_down");
        assert_eq!(contract.tensor.residual_len, 2);
        assert_eq!(contract.tensor.down_cols, 2);
        assert!(!contract.state.reads_kv);
        assert!(!contract.state.reads_deltanet);
        assert!(contract.state.mutates_residual);

        assert_eq!(
            dense_ffn_backend_decision(DenseFfnBackendPreference::CpuOracle, false),
            (DenseFfnBackend::CpuOracle, None)
        );
        assert_eq!(
            dense_ffn_backend_decision(DenseFfnBackendPreference::GpuProduction, false),
            (DenseFfnBackend::GpuProduction, None)
        );
        assert_eq!(
            dense_ffn_backend_decision(DenseFfnBackendPreference::NpuOptIn, true),
            (DenseFfnBackend::NpuXdna, None)
        );
        assert_eq!(
            dense_ffn_backend_decision(DenseFfnBackendPreference::NpuOptIn, false),
            (
                DenseFfnBackend::GpuProduction,
                Some("npu_backend_unavailable")
            )
        );
    }

    #[test]
    fn dense_ffn_evidence_records_backend_module_drift_and_fallback() {
        let shadow = Bf16DownShadow {
            w_down: vec![1.0],
            m: 1,
            k: 1,
        };
        let contract =
            dense_ffn_swiglu_down_contract(0, &shadow, DenseFfnBackendPreference::GpuProduction);
        let drift = DiffStats {
            n: 1,
            max_abs: 0.25,
            mean_abs: 0.25,
            rms: 0.25,
            n_nan: 0,
            n_inf: 0,
        };
        let evidence = dense_ffn_module_evidence(
            &contract,
            DenseFfnBackend::GpuProduction,
            Some(drift),
            Some("test_fallback"),
        );
        assert_eq!(evidence.module_kind, "qwen35_dense_ffn_swiglu_down");
        assert_eq!(evidence.module_id, "qwen35.layers.0.mlp.swiglu_down");
        assert_eq!(
            evidence.preferred_backend,
            DenseFfnBackendPreference::GpuProduction
        );
        assert_eq!(evidence.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(evidence.oracle_backend, DenseFfnBackend::CpuOracle);
        assert_eq!(evidence.fallback_reason, Some("test_fallback"));
        assert_eq!(evidence.drift.unwrap().max_abs, 0.25);

        let json = dense_ffn_module_evidence_json(&evidence);
        assert_eq!(json["module_kind"], "qwen35_dense_ffn_swiglu_down");
        assert_eq!(json["module_id"], "qwen35.layers.0.mlp.swiglu_down");
        assert_eq!(json["preferred_backend"], "gpu_production");
        assert_eq!(json["selected_backend"], "gpu_production");
        assert_eq!(json["oracle_backend"], "cpu_oracle");
        assert_eq!(json["fallback_reason"], "test_fallback");
        assert_eq!(json["drift"]["n"], 1);
        assert_eq!(json["drift"]["max_abs"], 0.25);
    }

    #[test]
    fn dense_ffn_invocation_and_output_bind_contract_to_backend_decision() {
        let shadow = Bf16DownShadow {
            w_down: vec![1.0, 2.0],
            m: 1,
            k: 2,
        };
        let invocation =
            dense_ffn_module_invocation(4, &shadow, DenseFfnBackendPreference::NpuOptIn, false);
        assert_eq!(
            invocation.contract.module_id,
            "qwen35.layers.4.mlp.swiglu_down"
        );
        assert_eq!(
            invocation.contract.preferred_backend,
            DenseFfnBackendPreference::NpuOptIn
        );
        assert_eq!(invocation.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(invocation.fallback_reason, Some("npu_backend_unavailable"));

        let output = dense_ffn_module_output(&invocation, None);
        assert_eq!(output.evidence.module_kind, "qwen35_dense_ffn_swiglu_down");
        assert_eq!(output.evidence.module_id, invocation.contract.module_id);
        assert_eq!(
            output.evidence.preferred_backend,
            DenseFfnBackendPreference::NpuOptIn
        );
        assert_eq!(
            output.evidence.selected_backend,
            DenseFfnBackend::GpuProduction
        );
        assert_eq!(
            output.evidence.fallback_reason,
            Some("npu_backend_unavailable")
        );
        assert!(output.mutates_residual);

        let json = dense_ffn_module_output_json(&output);
        assert_eq!(json["evidence"]["preferred_backend"], "npu_opt_in");
        assert_eq!(json["evidence"]["selected_backend"], "gpu_production");
        assert_eq!(json["evidence"]["oracle_backend"], "cpu_oracle");
        assert_eq!(
            json["evidence"]["fallback_reason"],
            "npu_backend_unavailable"
        );
        assert_eq!(json["mutates_residual"], true);
    }

    #[test]
    fn dense_ffn_mode_maps_xdna1_to_npu_opt_in_preference() {
        assert_eq!(
            dense_ffn_backend_preference_for_mode(FfnBf16Mode::Off),
            None
        );
        assert_eq!(
            dense_ffn_backend_preference_for_mode(FfnBf16Mode::Compare),
            Some(DenseFfnBackendPreference::GpuProduction)
        );
        assert_eq!(
            dense_ffn_backend_preference_for_mode(FfnBf16Mode::Cpu),
            Some(DenseFfnBackendPreference::CpuOracle)
        );
        assert_eq!(
            dense_ffn_backend_preference_for_mode(FfnBf16Mode::Xdna1),
            Some(DenseFfnBackendPreference::NpuOptIn)
        );
        let missing = xdna1_dense_ffn_module_invocation_from_shape(
            2,
            4096,
            11008,
            &XdnaModuleArtifacts::new(None, None),
        );
        assert_eq!(missing.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(missing.fallback_reason, Some("npu_artifacts_missing"));

        let admitted = xdna1_dense_ffn_module_invocation_from_shape(
            2,
            4096,
            11008,
            &XdnaModuleArtifacts::new(Some("a.xclbin".to_string()), Some("a.instr".to_string())),
        );
        assert_eq!(admitted.selected_backend, DenseFfnBackend::NpuXdna);
        assert_eq!(admitted.fallback_reason, None);
    }

    #[test]
    fn dense_ffn_invocation_can_use_production_weight_shape_without_bf16_shadow() {
        let invocation = dense_ffn_module_invocation_from_shape(
            5,
            4096,
            11008,
            DenseFfnBackendPreference::GpuProduction,
            false,
        );
        assert_eq!(
            invocation.contract.module_kind,
            "qwen35_dense_ffn_swiglu_down"
        );
        assert_eq!(
            invocation.contract.module_id,
            "qwen35.layers.5.mlp.swiglu_down"
        );
        assert_eq!(invocation.contract.tensor.residual_len, 4096);
        assert_eq!(invocation.contract.tensor.gate_len, 11008);
        assert_eq!(invocation.contract.tensor.up_len, 11008);
        assert_eq!(invocation.contract.tensor.down_rows, 4096);
        assert_eq!(invocation.contract.tensor.down_cols, 11008);
        assert_eq!(invocation.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(invocation.fallback_reason, None);

        let output = dense_ffn_module_output(&invocation, None);
        assert_eq!(
            output.evidence.preferred_backend,
            DenseFfnBackendPreference::GpuProduction
        );
        assert_eq!(
            output.evidence.selected_backend,
            DenseFfnBackend::GpuProduction
        );
        assert_eq!(output.evidence.oracle_backend, DenseFfnBackend::CpuOracle);
        assert!(output.mutates_residual);
    }

    #[test]
    fn attention_wo_invocation_records_projection_shape_and_gate_state() {
        let invocation = attention_wo_residual_invocation_from_shape(
            7,
            2048,
            4096,
            true,
            DenseFfnBackendPreference::GpuProduction,
            false,
        );
        assert_eq!(
            invocation.contract.module_kind,
            "qwen35_attention_wo_residual"
        );
        assert_eq!(
            invocation.contract.module_id,
            "qwen35.layers.7.attention.wo_residual"
        );
        assert_eq!(invocation.contract.tensor.input_len, 2048);
        assert_eq!(invocation.contract.tensor.output_len, 4096);
        assert_eq!(invocation.contract.tensor.weight_rows, 4096);
        assert_eq!(invocation.contract.tensor.weight_cols, 2048);
        assert!(invocation.contract.state.uses_attention_gate);
        assert!(invocation.contract.state.mutates_residual);
        assert!(!invocation.contract.state.reads_kv);
        assert_eq!(invocation.selected_backend, DenseFfnBackend::GpuProduction);

        let output = projection_module_output(&invocation);
        assert_eq!(output.module_kind, "qwen35_attention_wo_residual");
        assert_eq!(output.module_id, invocation.contract.module_id);
        assert_eq!(
            output.preferred_backend,
            DenseFfnBackendPreference::GpuProduction
        );
        assert_eq!(output.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(output.oracle_backend, DenseFfnBackend::CpuOracle);
        assert_eq!(output.fallback_reason, None);
        assert!(output.mutates_residual);

        let json = projection_module_output_json(&output);
        assert_eq!(json["module_kind"], "qwen35_attention_wo_residual");
        assert_eq!(json["module_id"], "qwen35.layers.7.attention.wo_residual");
        assert_eq!(json["preferred_backend"], "gpu_production");
        assert_eq!(json["selected_backend"], "gpu_production");
        assert_eq!(json["oracle_backend"], "cpu_oracle");
        assert!(json["fallback_reason"].is_null());
        assert_eq!(json["mutates_residual"], true);
    }
}
