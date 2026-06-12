// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! CPU BF16 oracle for the dense Qwen3.5 FFN SwiGLU/down epilogue.

use std::sync::OnceLock;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSelect {
    One(usize),
    All,
}

#[derive(Debug)]
pub struct Bf16DownShadow {
    pub w_down: Vec<f32>,
    pub m: usize,
    pub k: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DiffStats {
    pub n: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub rms: f32,
    pub n_nan: usize,
    pub n_inf: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseFfnBackendPreference {
    CpuOracle,
    GpuProduction,
    NpuOptIn,
}

impl DenseFfnBackendPreference {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOracle => "cpu_oracle",
            Self::GpuProduction => "gpu_production",
            Self::NpuOptIn => "npu_opt_in",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseFfnBackend {
    CpuOracle,
    GpuProduction,
    NpuXdna1,
}

impl DenseFfnBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOracle => "cpu_oracle",
            Self::GpuProduction => "gpu_production",
            Self::NpuXdna1 => "npu_xdna1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFfnTensorContract {
    pub residual_len: usize,
    pub gate_len: usize,
    pub up_len: usize,
    pub down_rows: usize,
    pub down_cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFfnStateContract {
    pub reads_kv: bool,
    pub reads_deltanet: bool,
    pub mutates_residual: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseFfnModuleContract {
    pub module_kind: &'static str,
    pub module_id: String,
    pub layer_idx: usize,
    pub tensor: DenseFfnTensorContract,
    pub state: DenseFfnStateContract,
    pub preferred_backend: DenseFfnBackendPreference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseFfnModuleInvocation {
    pub contract: DenseFfnModuleContract,
    pub selected_backend: DenseFfnBackend,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct DenseFfnModuleEvidence {
    pub module_id: String,
    pub selected_backend: DenseFfnBackend,
    pub oracle_backend: DenseFfnBackend,
    pub drift: Option<DiffStats>,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct DenseFfnModuleOutput {
    pub evidence: DenseFfnModuleEvidence,
    pub mutates_residual: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionTensorContract {
    pub input_len: usize,
    pub output_len: usize,
    pub weight_rows: usize,
    pub weight_cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionStateContract {
    pub reads_kv: bool,
    pub reads_deltanet: bool,
    pub uses_attention_gate: bool,
    pub mutates_residual: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionModuleContract {
    pub module_kind: &'static str,
    pub module_id: String,
    pub layer_idx: usize,
    pub tensor: ProjectionTensorContract,
    pub state: ProjectionStateContract,
    pub preferred_backend: DenseFfnBackendPreference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionModuleInvocation {
    pub contract: ProjectionModuleContract,
    pub selected_backend: DenseFfnBackend,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct ProjectionModuleOutput {
    pub module_id: String,
    pub selected_backend: DenseFfnBackend,
    pub oracle_backend: DenseFfnBackend,
    pub fallback_reason: Option<&'static str>,
    pub mutates_residual: bool,
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
        FfnBf16Config { mode, layer, trace }
    })
}

pub fn dense_ffn_module_id(layer_idx: usize) -> String {
    format!("qwen35.layers.{layer_idx}.mlp.swiglu_down")
}

pub fn attention_wo_module_id(layer_idx: usize) -> String {
    format!("qwen35.layers.{layer_idx}.attention.wo_residual")
}

pub fn dense_ffn_swiglu_down_contract(
    layer_idx: usize,
    shadow: &Bf16DownShadow,
    preferred_backend: DenseFfnBackendPreference,
) -> DenseFfnModuleContract {
    dense_ffn_swiglu_down_contract_from_shape(layer_idx, shadow.m, shadow.k, preferred_backend)
}

pub fn dense_ffn_swiglu_down_contract_from_shape(
    layer_idx: usize,
    residual_len: usize,
    hidden_len: usize,
    preferred_backend: DenseFfnBackendPreference,
) -> DenseFfnModuleContract {
    DenseFfnModuleContract {
        module_kind: "qwen35_dense_ffn_swiglu_down",
        module_id: dense_ffn_module_id(layer_idx),
        layer_idx,
        tensor: DenseFfnTensorContract {
            residual_len,
            gate_len: hidden_len,
            up_len: hidden_len,
            down_rows: residual_len,
            down_cols: hidden_len,
        },
        state: DenseFfnStateContract {
            reads_kv: false,
            reads_deltanet: false,
            mutates_residual: true,
        },
        preferred_backend,
    }
}

pub fn dense_ffn_backend_decision(
    preferred_backend: DenseFfnBackendPreference,
    npu_available: bool,
) -> (DenseFfnBackend, Option<&'static str>) {
    match preferred_backend {
        DenseFfnBackendPreference::CpuOracle => (DenseFfnBackend::CpuOracle, None),
        DenseFfnBackendPreference::GpuProduction => (DenseFfnBackend::GpuProduction, None),
        DenseFfnBackendPreference::NpuOptIn if npu_available => (DenseFfnBackend::NpuXdna1, None),
        DenseFfnBackendPreference::NpuOptIn => (
            DenseFfnBackend::GpuProduction,
            Some("npu_backend_unavailable"),
        ),
    }
}

pub fn dense_ffn_module_invocation(
    layer_idx: usize,
    shadow: &Bf16DownShadow,
    preferred_backend: DenseFfnBackendPreference,
    npu_available: bool,
) -> DenseFfnModuleInvocation {
    dense_ffn_module_invocation_from_shape(
        layer_idx,
        shadow.m,
        shadow.k,
        preferred_backend,
        npu_available,
    )
}

pub fn dense_ffn_module_invocation_from_shape(
    layer_idx: usize,
    residual_len: usize,
    hidden_len: usize,
    preferred_backend: DenseFfnBackendPreference,
    npu_available: bool,
) -> DenseFfnModuleInvocation {
    let contract = dense_ffn_swiglu_down_contract_from_shape(
        layer_idx,
        residual_len,
        hidden_len,
        preferred_backend,
    );
    let (selected_backend, fallback_reason) =
        dense_ffn_backend_decision(preferred_backend, npu_available);
    DenseFfnModuleInvocation {
        contract,
        selected_backend,
        fallback_reason,
    }
}

pub fn dense_ffn_module_evidence(
    contract: &DenseFfnModuleContract,
    selected_backend: DenseFfnBackend,
    drift: Option<DiffStats>,
    fallback_reason: Option<&'static str>,
) -> DenseFfnModuleEvidence {
    DenseFfnModuleEvidence {
        module_id: contract.module_id.clone(),
        selected_backend,
        oracle_backend: DenseFfnBackend::CpuOracle,
        drift,
        fallback_reason,
    }
}

pub fn dense_ffn_module_output(
    invocation: &DenseFfnModuleInvocation,
    drift: Option<DiffStats>,
) -> DenseFfnModuleOutput {
    DenseFfnModuleOutput {
        evidence: dense_ffn_module_evidence(
            &invocation.contract,
            invocation.selected_backend,
            drift,
            invocation.fallback_reason,
        ),
        mutates_residual: invocation.contract.state.mutates_residual,
    }
}

pub fn attention_wo_residual_contract_from_shape(
    layer_idx: usize,
    input_len: usize,
    output_len: usize,
    uses_attention_gate: bool,
    preferred_backend: DenseFfnBackendPreference,
) -> ProjectionModuleContract {
    ProjectionModuleContract {
        module_kind: "qwen35_attention_wo_residual",
        module_id: attention_wo_module_id(layer_idx),
        layer_idx,
        tensor: ProjectionTensorContract {
            input_len,
            output_len,
            weight_rows: output_len,
            weight_cols: input_len,
        },
        state: ProjectionStateContract {
            reads_kv: false,
            reads_deltanet: false,
            uses_attention_gate,
            mutates_residual: true,
        },
        preferred_backend,
    }
}

pub fn attention_wo_residual_invocation_from_shape(
    layer_idx: usize,
    input_len: usize,
    output_len: usize,
    uses_attention_gate: bool,
    preferred_backend: DenseFfnBackendPreference,
    npu_available: bool,
) -> ProjectionModuleInvocation {
    let contract = attention_wo_residual_contract_from_shape(
        layer_idx,
        input_len,
        output_len,
        uses_attention_gate,
        preferred_backend,
    );
    let (selected_backend, fallback_reason) =
        dense_ffn_backend_decision(preferred_backend, npu_available);
    ProjectionModuleInvocation {
        contract,
        selected_backend,
        fallback_reason,
    }
}

pub fn projection_module_output(invocation: &ProjectionModuleInvocation) -> ProjectionModuleOutput {
    ProjectionModuleOutput {
        module_id: invocation.contract.module_id.clone(),
        selected_backend: invocation.selected_backend,
        oracle_backend: DenseFfnBackend::CpuOracle,
        fallback_reason: invocation.fallback_reason,
        mutates_residual: invocation.contract.state.mutates_residual,
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

pub fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn round_f32_to_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits_rne(x))
}

pub fn decode_w_down_shadow(
    data: &[u8],
    quant_type: u8,
    m: usize,
    k: usize,
) -> Option<Bf16DownShadow> {
    let expected = m.checked_mul(k)?;
    let w_down = match quant_type {
        2 => {
            if data.len() != expected * 4 {
                return None;
            }
            data.chunks_exact(4)
                .map(|c| round_f32_to_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect()
        }
        16 => {
            if data.len() != expected * 2 {
                return None;
            }
            data.chunks_exact(2)
                .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        _ => return None,
    };
    Some(Bf16DownShadow { w_down, m, k })
}

pub fn swiglu_down_bf16_cpu(
    gate: &[f32],
    up: &[f32],
    residual: &[f32],
    shadow: &Bf16DownShadow,
) -> Vec<f32> {
    assert_eq!(gate.len(), shadow.k);
    assert_eq!(up.len(), shadow.k);
    assert_eq!(residual.len(), shadow.m);

    let mut hidden = vec![0.0f32; shadow.k];
    for i in 0..shadow.k {
        let g = round_f32_to_bf16(gate[i]);
        let u = round_f32_to_bf16(up[i]);
        let silu = g / (1.0 + (-g).exp());
        hidden[i] = round_f32_to_bf16(silu * u);
    }

    let mut out = residual.to_vec();
    for row in 0..shadow.m {
        let w_row = &shadow.w_down[row * shadow.k..(row + 1) * shadow.k];
        let mut acc = 0.0f32;
        for col in 0..shadow.k {
            acc += w_row[col] * hidden[col];
        }
        out[row] += acc;
    }
    out
}

pub fn diff_stats(a: &[f32], b: &[f32]) -> DiffStats {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_nan() || y.is_nan() {
            n_nan += 1;
            continue;
        }
        if x.is_infinite() || y.is_infinite() {
            n_inf += 1;
            continue;
        }
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_sq += (d as f64) * (d as f64);
    }
    let n = a.len();
    DiffStats {
        n,
        max_abs,
        mean_abs: (sum_abs / n.max(1) as f64) as f32,
        rms: (sum_sq / n.max(1) as f64).sqrt() as f32,
        n_nan,
        n_inf,
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
            (DenseFfnBackend::NpuXdna1, None)
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
        assert_eq!(evidence.module_id, "qwen35.layers.0.mlp.swiglu_down");
        assert_eq!(evidence.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(evidence.oracle_backend, DenseFfnBackend::CpuOracle);
        assert_eq!(evidence.fallback_reason, Some("test_fallback"));
        assert_eq!(evidence.drift.unwrap().max_abs, 0.25);
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
        assert_eq!(output.evidence.module_id, invocation.contract.module_id);
        assert_eq!(
            output.evidence.selected_backend,
            DenseFfnBackend::GpuProduction
        );
        assert_eq!(
            output.evidence.fallback_reason,
            Some("npu_backend_unavailable")
        );
        assert!(output.mutates_residual);
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
        assert_eq!(output.module_id, invocation.contract.module_id);
        assert_eq!(output.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(output.oracle_backend, DenseFfnBackend::CpuOracle);
        assert_eq!(output.fallback_reason, None);
        assert!(output.mutates_residual);
    }
}
