// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Deterministic CPU oracle backends and module evidence contracts.

use serde_json::{json, Value};

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
    NpuXdna,
}

impl DenseFfnBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOracle => "cpu_oracle",
            Self::GpuProduction => "gpu_production",
            Self::NpuXdna => "npu_xdna",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendSelection {
    pub preferred_backend: DenseFfnBackendPreference,
    pub selected_backend: DenseFfnBackend,
    pub oracle_backend: DenseFfnBackend,
    pub fallback_reason: Option<&'static str>,
}

impl BackendSelection {
    pub fn new(
        preferred_backend: DenseFfnBackendPreference,
        selected_backend: DenseFfnBackend,
        fallback_reason: Option<&'static str>,
    ) -> Self {
        Self {
            preferred_backend,
            selected_backend,
            oracle_backend: DenseFfnBackend::CpuOracle,
            fallback_reason,
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
    pub module_kind: &'static str,
    pub module_id: String,
    pub preferred_backend: DenseFfnBackendPreference,
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
    pub module_kind: &'static str,
    pub module_id: String,
    pub preferred_backend: DenseFfnBackendPreference,
    pub selected_backend: DenseFfnBackend,
    pub oracle_backend: DenseFfnBackend,
    pub fallback_reason: Option<&'static str>,
    pub mutates_residual: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleInvocation {
    DenseFfn(DenseFfnModuleInvocation),
    Projection(ProjectionModuleInvocation),
}

impl ModuleInvocation {
    pub fn module_kind(&self) -> &'static str {
        match self {
            Self::DenseFfn(invocation) => invocation.contract.module_kind,
            Self::Projection(invocation) => invocation.contract.module_kind,
        }
    }

    pub fn module_id(&self) -> &str {
        match self {
            Self::DenseFfn(invocation) => &invocation.contract.module_id,
            Self::Projection(invocation) => &invocation.contract.module_id,
        }
    }

    pub fn layer_idx(&self) -> usize {
        match self {
            Self::DenseFfn(invocation) => invocation.contract.layer_idx,
            Self::Projection(invocation) => invocation.contract.layer_idx,
        }
    }

    pub fn backend_selection(&self) -> BackendSelection {
        match self {
            Self::DenseFfn(invocation) => BackendSelection::new(
                invocation.contract.preferred_backend,
                invocation.selected_backend,
                invocation.fallback_reason,
            ),
            Self::Projection(invocation) => BackendSelection::new(
                invocation.contract.preferred_backend,
                invocation.selected_backend,
                invocation.fallback_reason,
            ),
        }
    }
}

impl From<DenseFfnModuleInvocation> for ModuleInvocation {
    fn from(invocation: DenseFfnModuleInvocation) -> Self {
        Self::DenseFfn(invocation)
    }
}

impl From<ProjectionModuleInvocation> for ModuleInvocation {
    fn from(invocation: ProjectionModuleInvocation) -> Self {
        Self::Projection(invocation)
    }
}

#[derive(Debug, Clone)]
pub enum ModuleOutput {
    DenseFfn(DenseFfnModuleOutput),
    Projection(ProjectionModuleOutput),
}

impl ModuleOutput {
    pub fn module_kind(&self) -> &'static str {
        match self {
            Self::DenseFfn(output) => output.evidence.module_kind,
            Self::Projection(output) => output.module_kind,
        }
    }

    pub fn module_id(&self) -> &str {
        match self {
            Self::DenseFfn(output) => &output.evidence.module_id,
            Self::Projection(output) => &output.module_id,
        }
    }

    pub fn mutates_residual(&self) -> bool {
        match self {
            Self::DenseFfn(output) => output.mutates_residual,
            Self::Projection(output) => output.mutates_residual,
        }
    }

    pub fn backend_selection(&self) -> BackendSelection {
        match self {
            Self::DenseFfn(output) => dense_ffn_module_backend_selection(&output.evidence),
            Self::Projection(output) => projection_module_backend_selection(output),
        }
    }
}

impl From<DenseFfnModuleOutput> for ModuleOutput {
    fn from(output: DenseFfnModuleOutput) -> Self {
        Self::DenseFfn(output)
    }
}

impl From<ProjectionModuleOutput> for ModuleOutput {
    fn from(output: ProjectionModuleOutput) -> Self {
        Self::Projection(output)
    }
}

#[derive(Debug)]
pub struct Bf16DownShadow {
    pub w_down: Vec<f32>,
    pub m: usize,
    pub k: usize,
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
        DenseFfnBackendPreference::NpuOptIn if npu_available => (DenseFfnBackend::NpuXdna, None),
        DenseFfnBackendPreference::NpuOptIn => (
            DenseFfnBackend::GpuProduction,
            Some("npu_backend_unavailable"),
        ),
    }
}

pub fn dense_ffn_backend_selection(
    preferred_backend: DenseFfnBackendPreference,
    npu_available: bool,
) -> BackendSelection {
    let (selected_backend, fallback_reason) =
        dense_ffn_backend_decision(preferred_backend, npu_available);
    BackendSelection::new(preferred_backend, selected_backend, fallback_reason)
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
        module_kind: contract.module_kind,
        module_id: contract.module_id.clone(),
        preferred_backend: contract.preferred_backend,
        selected_backend,
        oracle_backend: DenseFfnBackend::CpuOracle,
        drift,
        fallback_reason,
    }
}

pub fn dense_ffn_module_backend_selection(evidence: &DenseFfnModuleEvidence) -> BackendSelection {
    BackendSelection {
        preferred_backend: evidence.preferred_backend,
        selected_backend: evidence.selected_backend,
        oracle_backend: evidence.oracle_backend,
        fallback_reason: evidence.fallback_reason,
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

pub fn backend_selection_json(selection: BackendSelection) -> Value {
    json!({
        "preferred_backend": selection.preferred_backend.as_str(),
        "selected_backend": selection.selected_backend.as_str(),
        "oracle_backend": selection.oracle_backend.as_str(),
        "fallback_reason": selection.fallback_reason,
    })
}

pub fn dense_ffn_module_evidence_json(evidence: &DenseFfnModuleEvidence) -> Value {
    let selection = backend_selection_json(dense_ffn_module_backend_selection(evidence));
    let mut value = json!({
        "module_kind": evidence.module_kind,
        "module_id": evidence.module_id,
        "preferred_backend": selection["preferred_backend"].clone(),
        "selected_backend": selection["selected_backend"].clone(),
        "oracle_backend": selection["oracle_backend"].clone(),
        "fallback_reason": selection["fallback_reason"].clone(),
    });
    if let Some(drift) = evidence.drift {
        value["drift"] = diff_stats_json(drift);
    }
    value
}

pub fn dense_ffn_module_output_json(output: &DenseFfnModuleOutput) -> Value {
    json!({
        "evidence": dense_ffn_module_evidence_json(&output.evidence),
        "mutates_residual": output.mutates_residual,
    })
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
        module_kind: invocation.contract.module_kind,
        module_id: invocation.contract.module_id.clone(),
        preferred_backend: invocation.contract.preferred_backend,
        selected_backend: invocation.selected_backend,
        oracle_backend: DenseFfnBackend::CpuOracle,
        fallback_reason: invocation.fallback_reason,
        mutates_residual: invocation.contract.state.mutates_residual,
    }
}

pub fn projection_module_backend_selection(output: &ProjectionModuleOutput) -> BackendSelection {
    BackendSelection {
        preferred_backend: output.preferred_backend,
        selected_backend: output.selected_backend,
        oracle_backend: output.oracle_backend,
        fallback_reason: output.fallback_reason,
    }
}

pub fn projection_module_output_json(output: &ProjectionModuleOutput) -> Value {
    let selection = backend_selection_json(projection_module_backend_selection(output));
    json!({
        "module_kind": output.module_kind,
        "module_id": output.module_id,
        "preferred_backend": selection["preferred_backend"].clone(),
        "selected_backend": selection["selected_backend"].clone(),
        "oracle_backend": selection["oracle_backend"].clone(),
        "fallback_reason": selection["fallback_reason"].clone(),
        "mutates_residual": output.mutates_residual,
    })
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

        let selection = dense_ffn_backend_selection(DenseFfnBackendPreference::NpuOptIn, false);
        assert_eq!(
            selection.preferred_backend,
            DenseFfnBackendPreference::NpuOptIn
        );
        assert_eq!(selection.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(selection.oracle_backend, DenseFfnBackend::CpuOracle);
        assert_eq!(selection.fallback_reason, Some("npu_backend_unavailable"));
        assert_eq!(
            backend_selection_json(selection)["fallback_reason"],
            "npu_backend_unavailable"
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
        assert_eq!(evidence.selected_backend, DenseFfnBackend::GpuProduction);
        assert_eq!(evidence.oracle_backend, DenseFfnBackend::CpuOracle);
        assert_eq!(evidence.fallback_reason, Some("test_fallback"));
        assert_eq!(evidence.drift.unwrap().max_abs, 0.25);
        assert_eq!(
            dense_ffn_module_backend_selection(&evidence).selected_backend,
            DenseFfnBackend::GpuProduction
        );

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
    fn dense_ffn_invocation_can_use_production_weight_shape_without_shadow() {
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

        let generic = ModuleInvocation::from(invocation.clone());
        assert_eq!(generic.module_kind(), "qwen35_dense_ffn_swiglu_down");
        assert_eq!(generic.module_id(), "qwen35.layers.5.mlp.swiglu_down");
        assert_eq!(generic.layer_idx(), 5);
        assert_eq!(
            generic.backend_selection().selected_backend,
            DenseFfnBackend::GpuProduction
        );

        let generic_output = ModuleOutput::from(dense_ffn_module_output(&invocation, None));
        assert_eq!(generic_output.module_kind(), "qwen35_dense_ffn_swiglu_down");
        assert_eq!(
            generic_output.module_id(),
            "qwen35.layers.5.mlp.swiglu_down"
        );
        assert!(generic_output.mutates_residual());
        assert_eq!(
            generic_output.backend_selection().oracle_backend,
            DenseFfnBackend::CpuOracle
        );
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
        assert_eq!(invocation.selected_backend, DenseFfnBackend::GpuProduction);

        let output = projection_module_output(&invocation);
        assert_eq!(
            projection_module_backend_selection(&output).preferred_backend,
            DenseFfnBackendPreference::GpuProduction
        );
        let generic = ModuleInvocation::from(invocation);
        assert_eq!(generic.module_kind(), "qwen35_attention_wo_residual");
        assert_eq!(generic.module_id(), "qwen35.layers.7.attention.wo_residual");
        assert_eq!(generic.layer_idx(), 7);

        let generic_output = ModuleOutput::from(output.clone());
        assert_eq!(generic_output.module_kind(), "qwen35_attention_wo_residual");
        assert_eq!(
            generic_output.backend_selection().selected_backend,
            DenseFfnBackend::GpuProduction
        );
        let json = projection_module_output_json(&output);
        assert_eq!(json["module_kind"], "qwen35_attention_wo_residual");
        assert_eq!(json["module_id"], "qwen35.layers.7.attention.wo_residual");
        assert_eq!(json["preferred_backend"], "gpu_production");
        assert_eq!(json["selected_backend"], "gpu_production");
        assert_eq!(json["oracle_backend"], "cpu_oracle");
        assert!(json["fallback_reason"].is_null());
        assert_eq!(json["mutates_residual"], true);
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
}
