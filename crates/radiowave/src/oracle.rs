// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Compiler-neutral AMDGPU oracle reports.
//!
//! The oracle layer compares emitted shapes; it does not promote a compiler
//! option or source recipe. Correctness and measured performance remain the
//! promotion gates.

use crate::{
    CompileManifest, Error, InstructionStats, KernelReport, Result, inspect_amdgpu_instructions,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub const ORACLE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleCompiler {
    HipccLlvm,
    RadvAco,
    AmdvlkLlpc,
}

impl OracleCompiler {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HipccLlvm => "hipcc_llvm",
            Self::RadvAco => "radv_aco",
            Self::AmdvlkLlpc => "amdvlk_llpc",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputRelationship {
    Exact,
    #[default]
    Semantic,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterCountBasis {
    #[default]
    Unknown,
    CodeObjectMetadata,
    DriverAllocation,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OracleMetadata {
    pub kernel: String,
    pub compiler_version: String,
    pub target: String,
    pub input_sha256: String,
    pub input_relationship: InputRelationship,
    pub workgroup_size: [u32; 3],
    pub wavefront_size: u32,
}

impl OracleMetadata {
    pub fn new(
        kernel: impl Into<String>,
        target: impl Into<String>,
        workgroup_size: [u32; 3],
    ) -> Self {
        Self {
            kernel: kernel.into(),
            target: target.into(),
            workgroup_size,
            ..Self::default()
        }
    }

    pub fn input_artifact(mut self, path: &Path) -> Result<Self> {
        self.input_sha256 = sha256_path(path)?;
        Ok(self)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CompilerStatistics {
    pub static_instructions: Option<u64>,
    pub code_size_bytes: Option<u64>,
    pub latency: Option<u64>,
    pub inverse_throughput: Option<u64>,
    pub pre_sched_vgprs: Option<u64>,
    pub pre_sched_sgprs: Option<u64>,
    pub valu_instructions: Option<u64>,
    pub salu_instructions: Option<u64>,
    pub vmem_instructions: Option<u64>,
    pub smem_instructions: Option<u64>,
    pub vopd_instructions: Option<u64>,
    pub vmem_clause_score: Option<u64>,
    pub smem_clause_score: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OracleReport {
    pub schema_version: u32,
    pub compiler: OracleCompiler,
    pub compiler_version: String,
    pub kernel: String,
    pub target: String,
    pub input_sha256: String,
    pub input_relationship: InputRelationship,
    pub workgroup_size: [u32; 3],
    pub wavefront_size: u32,
    #[serde(default)]
    pub register_count_basis: RegisterCountBasis,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub vgpr_spill_count: u32,
    pub sgpr_spill_count: u32,
    pub scratch_size_bytes: u64,
    pub lds_size_bytes: u64,
    pub instructions: InstructionStats,
    #[serde(default)]
    pub compiler_statistics: CompilerStatistics,
}

impl OracleReport {
    pub fn from_hip_manifest(
        manifest: &CompileManifest,
        kernel: &str,
        workgroup_size: [u32; 3],
    ) -> Result<Self> {
        let inspection = manifest.inspection.as_ref().ok_or_else(|| {
            Error::InvalidOracle("HIP manifest contains no code-object inspection".to_owned())
        })?;
        let canonical = kernel.strip_suffix(".kd").unwrap_or(kernel);
        let report = inspection
            .kernel(kernel)
            .or_else(|| inspection.kernel(canonical))
            .ok_or_else(|| Error::InvalidOracle(format!("HIP kernel {kernel:?} is absent")))?;
        Self::from_hip_kernel(manifest, report, canonical.to_owned(), workgroup_size)
    }

    fn from_hip_kernel(
        manifest: &CompileManifest,
        kernel: &KernelReport,
        kernel_name: String,
        workgroup_size: [u32; 3],
    ) -> Result<Self> {
        let report = Self {
            schema_version: ORACLE_SCHEMA_VERSION,
            compiler: OracleCompiler::HipccLlvm,
            compiler_version: manifest.hipcc_version.clone(),
            kernel: kernel_name,
            target: manifest.arch.clone(),
            input_sha256: manifest.source_sha256.clone(),
            input_relationship: InputRelationship::Semantic,
            workgroup_size,
            wavefront_size: kernel.wavefront_size,
            register_count_basis: RegisterCountBasis::CodeObjectMetadata,
            vgpr_count: kernel.vgpr_count,
            sgpr_count: kernel.sgpr_count,
            vgpr_spill_count: kernel.vgpr_spill_count,
            sgpr_spill_count: kernel.sgpr_spill_count,
            scratch_size_bytes: kernel.private_segment_fixed_size.into(),
            lds_size_bytes: 0,
            instructions: kernel.instructions.clone(),
            compiler_statistics: CompilerStatistics::default(),
        };
        report.validate()?;
        Ok(report)
    }

    pub fn from_aco_dump(dump: &str, mut metadata: OracleMetadata) -> Result<Self> {
        let marker = "*** SHADER STATS ***";
        let marker_count = dump.match_indices(marker).count();
        if marker_count != 1 {
            return Err(Error::InvalidOracle(format!(
                "ACO dump must contain exactly one shader statistics block, found {marker_count}"
            )));
        }
        if metadata.workgroup_size.contains(&0) {
            metadata.workgroup_size = parse_aco_workgroup_size(dump)
                .ok_or_else(|| Error::InvalidOracle("ACO workgroup size is missing".to_owned()))?;
        }
        if metadata.wavefront_size == 0 {
            return Err(Error::InvalidOracle(
                "ACO subgroup size must be supplied by the pipeline executable properties"
                    .to_owned(),
            ));
        }

        let disasm = between(dump, "\ndisasm:\n", "\n\nCompute Shader:").ok_or_else(|| {
            Error::InvalidOracle("ACO dump contains no final assembly section".to_owned())
        })?;
        let stats = between(dump, marker, "********************").ok_or_else(|| {
            Error::InvalidOracle("ACO shader statistics block is unterminated".to_owned())
        })?;
        let report = Self {
            schema_version: ORACLE_SCHEMA_VERSION,
            compiler: OracleCompiler::RadvAco,
            compiler_version: metadata.compiler_version,
            kernel: metadata.kernel,
            target: metadata.target,
            input_sha256: metadata.input_sha256,
            input_relationship: metadata.input_relationship,
            workgroup_size: metadata.workgroup_size,
            wavefront_size: metadata.wavefront_size,
            register_count_basis: RegisterCountBasis::DriverAllocation,
            vgpr_count: labeled_u64(stats, "VGPRs:").unwrap_or_default() as u32,
            sgpr_count: labeled_u64(stats, "SGPRs:").unwrap_or_default() as u32,
            vgpr_spill_count: labeled_u64(stats, "Spilled VGPRs:").unwrap_or_default() as u32,
            sgpr_spill_count: labeled_u64(stats, "Spilled SGPRs:").unwrap_or_default() as u32,
            scratch_size_bytes: labeled_u64(stats, "Scratch size:").unwrap_or_default(),
            lds_size_bytes: labeled_u64(stats, "LDS size:").unwrap_or_default(),
            instructions: inspect_amdgpu_instructions(disasm),
            compiler_statistics: CompilerStatistics {
                static_instructions: labeled_u64(stats, "Instructions:"),
                code_size_bytes: labeled_u64(stats, "Code size:"),
                latency: labeled_u64(stats, "Latency:"),
                inverse_throughput: labeled_u64(stats, "Inverse Throughput:"),
                pre_sched_vgprs: labeled_u64(stats, "Pre-Sched VGPRs:"),
                pre_sched_sgprs: labeled_u64(stats, "Pre-Sched SGPRs:"),
                valu_instructions: labeled_u64(stats, "VALU:"),
                salu_instructions: labeled_u64(stats, "SALU:"),
                vmem_instructions: labeled_u64(stats, "VMEM:"),
                smem_instructions: labeled_u64(stats, "SMEM:"),
                vopd_instructions: labeled_u64(stats, "VOPD:"),
                vmem_clause_score: labeled_u64(stats, "VMEM Clause:"),
                smem_clause_score: labeled_u64(stats, "SMEM Clause:"),
            },
        };
        report.validate()?;
        Ok(report)
    }

    pub fn from_llpc_assembly(
        assembly: &str,
        symbol: &str,
        mut metadata: OracleMetadata,
    ) -> Result<Self> {
        let body = assembly_symbol(assembly, symbol).ok_or_else(|| {
            Error::InvalidOracle(format!("LLPC assembly symbol {symbol:?} is absent"))
        })?;
        let reported_wavefront = metadata_u64(assembly, ".wavefront_size:").unwrap_or_default();
        if metadata.wavefront_size == 0 {
            metadata.wavefront_size = reported_wavefront as u32;
        } else if reported_wavefront != 0 && metadata.wavefront_size != reported_wavefront as u32 {
            return Err(Error::InvalidOracle(format!(
                "LLPC wavefront mismatch: requested {}, assembly reports {reported_wavefront}",
                metadata.wavefront_size
            )));
        }
        if metadata.workgroup_size.contains(&0) {
            metadata.workgroup_size = parse_llpc_workgroup_size(assembly).ok_or_else(|| {
                Error::InvalidOracle("LLPC threadgroup dimensions are missing".to_owned())
            })?;
        }
        let report = Self {
            schema_version: ORACLE_SCHEMA_VERSION,
            compiler: OracleCompiler::AmdvlkLlpc,
            compiler_version: metadata.compiler_version,
            kernel: metadata.kernel,
            target: metadata.target,
            input_sha256: metadata.input_sha256,
            input_relationship: metadata.input_relationship,
            workgroup_size: metadata.workgroup_size,
            wavefront_size: metadata.wavefront_size,
            register_count_basis: RegisterCountBasis::CodeObjectMetadata,
            vgpr_count: metadata_u64(assembly, ".vgpr_count:").unwrap_or_default() as u32,
            sgpr_count: metadata_u64(assembly, ".sgpr_count:").unwrap_or_default() as u32,
            vgpr_spill_count: metadata_u64(assembly, ".vgpr_spill_count:").unwrap_or_default()
                as u32,
            sgpr_spill_count: metadata_u64(assembly, ".sgpr_spill_count:").unwrap_or_default()
                as u32,
            scratch_size_bytes: metadata_u64(assembly, ".scratch_memory_size:").unwrap_or_default(),
            lds_size_bytes: metadata_u64(assembly, ".group_segment_fixed_size:")
                .unwrap_or_default(),
            instructions: inspect_amdgpu_instructions(&body),
            compiler_statistics: CompilerStatistics::default(),
        };
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ORACLE_SCHEMA_VERSION {
            return Err(Error::InvalidOracle(format!(
                "unsupported oracle schema {}, expected {ORACLE_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.kernel.is_empty() || self.target.is_empty() {
            return Err(Error::InvalidOracle(
                "oracle kernel and target must be non-empty".to_owned(),
            ));
        }
        if self.workgroup_size.contains(&0) {
            return Err(Error::InvalidOracle(
                "oracle workgroup dimensions must be non-zero".to_owned(),
            ));
        }
        if !matches!(self.wavefront_size, 32 | 64) {
            return Err(Error::InvalidOracle(format!(
                "oracle wavefront size {} is neither 32 nor 64",
                self.wavefront_size
            )));
        }
        if self.input_relationship == InputRelationship::Exact && self.input_sha256.is_empty() {
            return Err(Error::InvalidOracle(
                "exact oracle inputs require an input SHA-256".to_owned(),
            ));
        }
        if self.instructions.static_instructions == 0 {
            return Err(Error::InvalidOracle(
                "oracle assembly contains no live machine instructions".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonConfidence {
    Exact,
    Semantic,
    Incomparable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OracleDelta {
    pub compiler: OracleCompiler,
    pub kernel: String,
    pub confidence: ComparisonConfidence,
    pub mismatches: Vec<String>,
    pub live_static_instructions: i64,
    pub compiler_static_instructions: Option<i64>,
    pub vgprs: Option<i64>,
    pub sgprs: Option<i64>,
    pub waits: i64,
    pub memory_clauses: i64,
    pub max_consecutive_vmem: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OracleComparison {
    pub schema_version: u32,
    pub baseline: OracleReport,
    pub candidates: Vec<OracleReport>,
    pub deltas: Vec<OracleDelta>,
}

impl OracleComparison {
    pub fn new(baseline: OracleReport, candidates: Vec<OracleReport>) -> Result<Self> {
        baseline.validate()?;
        for candidate in &candidates {
            candidate.validate()?;
        }
        let deltas = candidates
            .iter()
            .map(|candidate| delta(&baseline, candidate))
            .collect();
        Ok(Self {
            schema_version: ORACLE_SCHEMA_VERSION,
            baseline,
            candidates,
            deltas,
        })
    }
}

pub fn sha256_path(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn delta(baseline: &OracleReport, candidate: &OracleReport) -> OracleDelta {
    let mut mismatches = Vec::new();
    if baseline.target != candidate.target {
        mismatches.push(format!(
            "target {} != {}",
            baseline.target, candidate.target
        ));
    }
    if baseline.workgroup_size != candidate.workgroup_size {
        mismatches.push(format!(
            "workgroup {:?} != {:?}",
            baseline.workgroup_size, candidate.workgroup_size
        ));
    }
    if baseline.wavefront_size != candidate.wavefront_size {
        mismatches.push(format!(
            "wavefront {} != {}",
            baseline.wavefront_size, candidate.wavefront_size
        ));
    }
    let confidence = if !mismatches.is_empty() {
        ComparisonConfidence::Incomparable
    } else if !baseline.input_sha256.is_empty()
        && baseline.input_sha256 == candidate.input_sha256
        && baseline.input_relationship == InputRelationship::Exact
        && candidate.input_relationship == InputRelationship::Exact
    {
        ComparisonConfidence::Exact
    } else {
        ComparisonConfidence::Semantic
    };
    OracleDelta {
        compiler: candidate.compiler,
        kernel: candidate.kernel.clone(),
        confidence,
        mismatches,
        live_static_instructions: i64::from(candidate.instructions.static_instructions)
            - i64::from(baseline.instructions.static_instructions),
        compiler_static_instructions: option_delta(
            baseline.compiler_statistics.static_instructions,
            candidate.compiler_statistics.static_instructions,
        ),
        vgprs: (candidate.register_count_basis == baseline.register_count_basis)
            .then(|| i64::from(candidate.vgpr_count) - i64::from(baseline.vgpr_count)),
        sgprs: (candidate.register_count_basis == baseline.register_count_basis)
            .then(|| i64::from(candidate.sgpr_count) - i64::from(baseline.sgpr_count)),
        waits: i64::from(candidate.instructions.wait_instructions)
            - i64::from(baseline.instructions.wait_instructions),
        memory_clauses: i64::from(candidate.instructions.memory_clause_instructions)
            - i64::from(baseline.instructions.memory_clause_instructions),
        max_consecutive_vmem: i64::from(candidate.instructions.max_consecutive_vmem_instructions)
            - i64::from(baseline.instructions.max_consecutive_vmem_instructions),
    }
}

fn option_delta(baseline: Option<u64>, candidate: Option<u64>) -> Option<i64> {
    Some(i64::try_from(candidate?).ok()? - i64::try_from(baseline?).ok()?)
}

fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let body = text.split_once(start)?.1;
    Some(body.split_once(end)?.0)
}

fn labeled_u64(text: &str, label: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        line.trim()
            .strip_prefix(label)
            .and_then(|value| parse_u64(value.trim()))
    })
}

fn metadata_u64(text: &str, label: &str) -> Option<u64> {
    text.lines().rev().find_map(|line| {
        line.trim()
            .strip_prefix(label)
            .and_then(|value| parse_u64(value.trim()))
    })
}

fn parse_u64(value: &str) -> Option<u64> {
    let value = value
        .split_whitespace()
        .next()?
        .trim_end_matches([',', ']']);
    value.strip_prefix("0x").map_or_else(
        || value.parse().ok(),
        |hex| u64::from_str_radix(hex, 16).ok(),
    )
}

fn parse_aco_workgroup_size(text: &str) -> Option<[u32; 3]> {
    let values = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("workgroup_size:").map(str::trim))?;
    parse_dimensions(values.split(',').map(str::trim))
}

fn parse_llpc_workgroup_size(text: &str) -> Option<[u32; 3]> {
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.trim() == ".threadgroup_dimensions:" {
            let values = lines
                .by_ref()
                .take(3)
                .filter_map(|line| line.trim().strip_prefix('-').and_then(parse_u64));
            return parse_dimensions(values.map(|value| value.to_string()));
        }
    }
    None
}

fn parse_dimensions<I, S>(values: I) -> Option<[u32; 3]>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let values = values
        .into_iter()
        .filter_map(|value| parse_u64(value.as_ref()).and_then(|value| u32::try_from(value).ok()))
        .collect::<Vec<_>>();
    (values.len() == 3).then(|| [values[0], values[1], values[2]])
}

fn assembly_symbol(text: &str, symbol: &str) -> Option<String> {
    let label = format!("{symbol}:");
    let mut found = false;
    let mut body = String::new();
    for line in text.lines() {
        if !found {
            found = line.trim() == label;
            continue;
        }
        let trimmed = line.trim();
        if trimmed.starts_with(".Lfunc_end") || trimmed.starts_with(&format!(".size\t{symbol}")) {
            break;
        }
        body.push_str(line);
        body.push('\n');
    }
    (found && !body.is_empty()).then_some(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodeObjectInspection, MutableReadCache, SchedulerProfile, Wavefront};
    use std::path::PathBuf;

    fn metadata() -> OracleMetadata {
        OracleMetadata {
            kernel: "probe".to_owned(),
            compiler_version: "test".to_owned(),
            target: "gfx1201".to_owned(),
            input_sha256: "same-input".to_owned(),
            input_relationship: InputRelationship::Exact,
            workgroup_size: [64, 1, 1],
            wavefront_size: 64,
        }
    }

    #[test]
    fn parses_single_aco_shader_dump() {
        let dump = r#"
shader: MESA_SHADER_COMPUTE
workgroup_size: 64, 1, 1
disasm:
BB0:
    buffer_load_b32 v0, v0, s[0:3], null offen
    s_wait_loadcnt 0x0
    s_endpgm
    s_nop 0

Compute Shader:
*** SHADER STATS ***
SGPRs: 128
VGPRs: 24
Spilled SGPRs: 0
Spilled VGPRs: 0
Code size: 32
LDS size: 0
Scratch size: 0
Instructions: 3
Latency: 20
Inverse Throughput: 10
VMEM Clause: 1
SMEM Clause: 0
Pre-Sched SGPRs: 8
Pre-Sched VGPRs: 2
VALU: 0
SALU: 2
VMEM: 1
SMEM: 0
VOPD: 0
********************
"#;
        let report = OracleReport::from_aco_dump(dump, metadata()).unwrap();
        assert_eq!(report.instructions.static_instructions, 3);
        assert_eq!(report.instructions.buffer_loads, 1);
        assert_eq!(report.compiler_statistics.static_instructions, Some(3));
        assert_eq!(report.compiler_statistics.code_size_bytes, Some(32));
    }

    #[test]
    fn parses_llpc_metadata_and_live_body() {
        let assembly = r#"
_amdgpu_cs_main:
    buffer_load_b32 v0, v0, s[0:3], null offen
    s_endpgm
    s_nop 0
.Lfunc_end0:
    .sgpr_count: 0x10
    .vgpr_count: 0xc
    .wavefront_size: 0x40
    .threadgroup_dimensions:
      - 0x40
      - 0x1
      - 0x1
"#;
        let report =
            OracleReport::from_llpc_assembly(assembly, "_amdgpu_cs_main", metadata()).unwrap();
        assert_eq!(report.instructions.static_instructions, 2);
        assert_eq!(report.sgpr_count, 16);
        assert_eq!(report.vgpr_count, 12);
    }

    #[test]
    fn imports_a_hip_manifest_as_semantic_evidence() {
        let manifest = CompileManifest {
            schema_version: 3,
            compiler: "radiowave".to_owned(),
            generated_unix_seconds: 0,
            source: PathBuf::from("probe.hip"),
            output: PathBuf::from("probe.hsaco"),
            arch: "gfx1201".to_owned(),
            wavefront: Wavefront::Wave64,
            scheduler_profile: SchedulerProfile::Default,
            hipcc: PathBuf::from("hipcc"),
            hipcc_version: "ROCm test".to_owned(),
            command: Vec::new(),
            source_sha256: "hip-source".to_owned(),
            support_header_sha256: "header".to_owned(),
            output_sha256: "output".to_owned(),
            inspection: Some(CodeObjectInspection {
                bundle_target: "gfx1201".to_owned(),
                identity: None,
                kernels: vec![KernelReport {
                    name: "probe".to_owned(),
                    wavefront_size: 64,
                    vgpr_count: 12,
                    sgpr_count: 14,
                    mutable_read_cache: MutableReadCache::VmemOnly,
                    instructions: InstructionStats {
                        static_instructions: 12,
                        ..InstructionStats::default()
                    },
                    ..KernelReport::default()
                }],
            }),
        };
        let report = OracleReport::from_hip_manifest(&manifest, "probe", [64, 1, 1]).unwrap();
        assert_eq!(report.compiler, OracleCompiler::HipccLlvm);
        assert_eq!(report.input_relationship, InputRelationship::Semantic);
    }

    #[test]
    fn comparison_refuses_geometry_mismatch_and_recognizes_exact_input() {
        let dump = r#"
workgroup_size: 64, 1, 1
disasm:
    s_endpgm

Compute Shader:
*** SHADER STATS ***
SGPRs: 1
VGPRs: 1
Instructions: 1
********************
"#;
        let baseline = OracleReport::from_aco_dump(dump, metadata()).unwrap();
        let exact = OracleReport::from_llpc_assembly(
            "_amdgpu_cs_main:\n s_endpgm\n.Lfunc_end0:\n .sgpr_count: 1\n .vgpr_count: 1\n .wavefront_size: 64\n",
            "_amdgpu_cs_main",
            metadata(),
        )
        .unwrap();
        let mut mismatched = exact.clone();
        mismatched.workgroup_size = [32, 1, 1];
        let comparison = OracleComparison::new(baseline, vec![exact, mismatched]).unwrap();
        assert_eq!(comparison.deltas[0].confidence, ComparisonConfidence::Exact);
        assert_eq!(comparison.deltas[0].vgprs, None);
        assert_eq!(comparison.deltas[0].sgprs, None);
        assert_eq!(
            comparison.deltas[1].confidence,
            ComparisonConfidence::Incomparable
        );
    }
}
