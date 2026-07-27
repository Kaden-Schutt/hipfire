// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Architecture-neutral optimization recipes and architecture-specific proof.
//!
//! A recipe describes a semantic transformation without naming a GPU.  A
//! [`RecipeEvidence`] promotes that recipe for one concrete architecture after
//! correctness and performance certification.  This separation lets
//! autoresearch try the same candidate on a new target without silently
//! inheriting another GPU's winner.

use crate::{SchedulerProfile, Wavefront};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const RECIPE_SCHEMA_VERSION: u32 = 1;
pub const HIPFIRE_6409_EVIDENCE: &str =
    "crates/radiowave/tests/artifacts/recipes/hipfire-6409-gfx1201-2026-07-13-radiowave-all-losers-final-u8-aggregate.json";
pub const HIPX_PORTABILITY_EVIDENCE: &str =
    "crates/radiowave/tests/artifacts/hipx-portability-2026-07-14.json";

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RecipeError {
    #[error("recipe catalog schema {actual} is unsupported; expected {expected}")]
    UnsupportedSchema { actual: u32, expected: u32 },
    #[error("recipe {0:?} does not exist in the catalog")]
    UnknownRecipe(String),
    #[error("recipes select conflicting {field} values: {left:?} and {right:?}")]
    ConflictingActions {
        field: &'static str,
        left: String,
        right: String,
    },
    #[error("autoresearch ledger line {line} is not valid JSON: {message}")]
    InvalidLedgerLine { line: usize, message: String },
}

pub type RecipeResult<T> = std::result::Result<T, RecipeError>;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum KernelSelector {
    Exact(String),
    Prefix(String),
}

impl KernelSelector {
    fn matches(&self, kernel: &str) -> bool {
        match self {
            Self::Exact(value) => kernel == value,
            Self::Prefix(value) => kernel.starts_with(value),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecipePredicate {
    pub kernels: Vec<KernelSelector>,
    pub families: Vec<String>,
    pub required_tags: BTreeSet<String>,
    pub forbidden_tags: BTreeSet<String>,
}

impl RecipePredicate {
    pub fn matches(&self, workload: &WorkloadDescriptor) -> bool {
        let kernel_matches = self.kernels.is_empty()
            || self
                .kernels
                .iter()
                .any(|selector| selector.matches(&workload.kernel));
        let family_matches = self.families.is_empty()
            || self
                .families
                .iter()
                .any(|family| family == &workload.family);
        kernel_matches
            && family_matches
            && self.required_tags.is_subset(&workload.tags)
            && self.forbidden_tags.is_disjoint(&workload.tags)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SourceLowering {
    BufferResourceB32,
    TemporalBufferB32,
    AlignedBufferB128,
    IndependentBufferB32,
    BufferOutputRmw,
    Wave32XorExchange,
    LdsTranspose32x32,
    Unroll(u32),
    Chunk(u32),
    PairedIntegerHash,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecipeAction {
    pub wavefront: Option<Wavefront>,
    pub scheduler_profile: Option<SchedulerProfile>,
    pub workgroup_size: Option<u32>,
    pub kernel_variant: Option<String>,
    pub defines: BTreeSet<String>,
    pub lowerings: BTreeSet<SourceLowering>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceVerdict {
    /// Correctness and code-shape validation passed, but no performance WIN
    /// has promoted the recipe on this architecture.
    CorrectnessOnly,
    Promoted,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeEvidence {
    pub architecture: String,
    pub verdict: EvidenceVerdict,
    pub correctness_pass: bool,
    pub artifact: String,
    #[serde(default)]
    pub samples_per_row: Option<u32>,
    #[serde(default)]
    pub throughput_delta_pct: Option<f64>,
    #[serde(default)]
    pub duration_delta_pct: Option<f64>,
    #[serde(default)]
    pub note: String,
}

impl RecipeEvidence {
    fn promotes(&self, architecture: &str) -> bool {
        self.architecture == architecture
            && self.verdict == EvidenceVerdict::Promoted
            && self.correctness_pass
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationRecipe {
    pub id: String,
    pub revision: u32,
    pub summary: String,
    pub predicate: RecipePredicate,
    pub action: RecipeAction,
    #[serde(default)]
    pub evidence: Vec<RecipeEvidence>,
}

impl OptimizationRecipe {
    pub fn is_promoted_for(&self, architecture: &str) -> bool {
        self.evidence
            .iter()
            .any(|evidence| evidence.promotes(architecture))
    }

    pub fn is_rejected_for(&self, architecture: &str) -> bool {
        self.evidence.iter().any(|evidence| {
            evidence.architecture == architecture && evidence.verdict == EvidenceVerdict::Rejected
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkloadDescriptor {
    pub kernel: String,
    pub family: String,
    pub tags: BTreeSet<String>,
}

impl WorkloadDescriptor {
    pub fn new(kernel: impl Into<String>, family: impl Into<String>) -> Self {
        Self {
            kernel: kernel.into(),
            family: family.into(),
            tags: BTreeSet::new(),
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.insert(tag.into());
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    /// Apply only recipes carrying promoted, correctness-passing evidence for
    /// this exact architecture.
    Certified,
    /// Emit semantically applicable candidates for an autoresearch run.  No
    /// architecture evidence is required and nothing is promoted implicitly.
    Candidates,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizationPlan {
    pub wavefront: Option<Wavefront>,
    pub scheduler_profile: Option<SchedulerProfile>,
    pub workgroup_size: Option<u32>,
    pub kernel_variant: Option<String>,
    pub defines: BTreeSet<String>,
    pub lowerings: BTreeSet<SourceLowering>,
}

impl OptimizationPlan {
    fn merge(&mut self, action: &RecipeAction) -> RecipeResult<()> {
        merge_option(
            "wavefront",
            &mut self.wavefront,
            action.wavefront,
            |value| format!("{value:?}"),
        )?;
        merge_option(
            "scheduler_profile",
            &mut self.scheduler_profile,
            action.scheduler_profile,
            |value| format!("{value:?}"),
        )?;
        merge_option(
            "workgroup_size",
            &mut self.workgroup_size,
            action.workgroup_size,
            |value| value.to_string(),
        )?;
        merge_option(
            "kernel_variant",
            &mut self.kernel_variant,
            action.kernel_variant.clone(),
            |value| value.clone(),
        )?;
        self.defines.extend(action.defines.iter().cloned());
        self.lowerings.extend(action.lowerings.iter().cloned());
        Ok(())
    }
}

fn merge_option<T: Clone + Eq>(
    field: &'static str,
    destination: &mut Option<T>,
    incoming: Option<T>,
    render: impl Fn(&T) -> String,
) -> RecipeResult<()> {
    let Some(incoming) = incoming else {
        return Ok(());
    };
    if let Some(current) = destination {
        if current != &incoming {
            return Err(RecipeError::ConflictingActions {
                field,
                left: render(current),
                right: render(&incoming),
            });
        }
    } else {
        *destination = Some(incoming);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RecipeSelection {
    pub architecture: String,
    pub mode: SelectionMode,
    pub workload: WorkloadDescriptor,
    pub applied_recipes: Vec<String>,
    pub candidate_recipes: Vec<String>,
    pub rejected_recipes: Vec<String>,
    pub plan: OptimizationPlan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeCatalog {
    pub schema_version: u32,
    pub recipes: Vec<OptimizationRecipe>,
}

impl RecipeCatalog {
    pub fn new(recipes: Vec<OptimizationRecipe>) -> Self {
        Self {
            schema_version: RECIPE_SCHEMA_VERSION,
            recipes,
        }
    }

    pub fn from_json(encoded: &str) -> serde_json::Result<Self> {
        serde_json::from_str(encoded)
    }

    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self).map(|json| json + "\n")
    }

    pub fn validate(&self) -> RecipeResult<()> {
        if self.schema_version != RECIPE_SCHEMA_VERSION {
            return Err(RecipeError::UnsupportedSchema {
                actual: self.schema_version,
                expected: RECIPE_SCHEMA_VERSION,
            });
        }
        let mut ids = BTreeSet::new();
        for recipe in &self.recipes {
            if !ids.insert(&recipe.id) {
                return Err(RecipeError::ConflictingActions {
                    field: "recipe_id",
                    left: recipe.id.clone(),
                    right: recipe.id.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn select(
        &self,
        architecture: impl Into<String>,
        workload: WorkloadDescriptor,
        mode: SelectionMode,
    ) -> RecipeResult<RecipeSelection> {
        self.validate()?;
        let architecture = architecture.into();
        let mut applied_recipes = Vec::new();
        let mut candidate_recipes = Vec::new();
        let mut rejected_recipes = Vec::new();
        let mut plan = OptimizationPlan::default();
        for recipe in &self.recipes {
            if !recipe.predicate.matches(&workload) {
                continue;
            }
            if recipe.is_rejected_for(&architecture) {
                rejected_recipes.push(recipe.id.clone());
                continue;
            }
            let promoted = recipe.is_promoted_for(&architecture);
            if !promoted {
                candidate_recipes.push(recipe.id.clone());
            }
            if mode == SelectionMode::Certified && !promoted {
                continue;
            }
            plan.merge(&recipe.action)?;
            applied_recipes.push(recipe.id.clone());
        }
        Ok(RecipeSelection {
            architecture,
            mode,
            workload,
            applied_recipes,
            candidate_recipes,
            rejected_recipes,
            plan,
        })
    }

    /// Promote labeled autoresearch WIN rows into architecture-specific recipe
    /// evidence. Rows without `radiowave_recipe`/`radiowave_recipes`, or rows
    /// whose final verdict is not `WIN`, are intentionally ignored.
    pub fn ingest_autoresearch_jsonl(&mut self, encoded: &str) -> RecipeResult<usize> {
        self.validate()?;
        let mut inserted = 0;
        for (index, line) in encoded.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: serde_json::Value =
                serde_json::from_str(line).map_err(|error| RecipeError::InvalidLedgerLine {
                    line: index + 1,
                    message: error.to_string(),
                })?;
            if row.get("verdict").and_then(|value| value.as_str()) != Some("WIN") {
                continue;
            }
            let recipe_ids = ledger_recipe_ids(&row);
            if recipe_ids.is_empty() {
                continue;
            }
            let architecture = row
                .get("gpu_arch")
                .or_else(|| row.get("arch"))
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            if architecture.is_empty() {
                continue;
            }
            let artifact = row
                .get("measurement_hash")
                .or_else(|| row.get("variant_sha"))
                .or_else(|| row.get("var_sha"))
                .and_then(|value| value.as_str())
                .unwrap_or("autoresearch-win")
                .to_owned();
            for recipe_id in recipe_ids {
                let recipe = self
                    .recipes
                    .iter_mut()
                    .find(|recipe| recipe.id == recipe_id)
                    .ok_or_else(|| RecipeError::UnknownRecipe(recipe_id.clone()))?;
                let duplicate = recipe.evidence.iter().any(|evidence| {
                    evidence.architecture == architecture
                        && evidence.artifact == artifact
                        && evidence.verdict == EvidenceVerdict::Promoted
                });
                if duplicate {
                    continue;
                }
                recipe.evidence.push(RecipeEvidence {
                    architecture: architecture.to_owned(),
                    verdict: EvidenceVerdict::Promoted,
                    correctness_pass: true,
                    artifact: artifact.clone(),
                    samples_per_row: row
                        .get("seeds")
                        .and_then(|value| value.as_u64())
                        .and_then(|value| u32::try_from(value).ok()),
                    throughput_delta_pct: json_f64(&row, &["tok_delta_pct", "delta_pct"]),
                    duration_delta_pct: json_f64(&row, &["dur_delta_pct", "perf_delta"]),
                    note: row
                        .get("profile")
                        .or_else(|| row.get("profile_feedback"))
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_owned(),
                });
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    pub fn builtin_hipfire_6409() -> Self {
        let promoted = || vec![hipfire_6409_evidence()];
        let mut buffer_resource_evidence = promoted();
        buffer_resource_evidence.extend(hipx_portability_evidence());
        let mut dispatch_workgroup_evidence = promoted();
        dispatch_workgroup_evidence.extend(hipx_dispatch_workgroup_promotion_evidence());
        let mut wave64_evidence = promoted();
        wave64_evidence.extend(hipx_wave64_promotion_evidence());
        let mut dequant_chunk16_evidence = promoted();
        dequant_chunk16_evidence.extend(hipx_dequant_chunk16_promotion_evidence());
        let mut q4_wave64_evidence = promoted();
        q4_wave64_evidence.extend(hipx_wave64_rejection_evidence("q4_selected_dual"));
        let mut q6_wave64_evidence = promoted();
        q6_wave64_evidence.extend(hipx_wave64_rejection_evidence("q6_x8"));
        let wave64_kernels = [
            "dense_q8",
            "dense_q8_single",
            "vopd_dependent",
            "memory_interleave4",
            "vopd_independent",
            "vopd_mixed",
            "vopd_dequant",
        ];
        let buffer_rmw_kernels = [
            "memory_gather",
            "memory_interleave4",
            "dot_q8",
            "dot_q4",
            "dot_q6",
            "dot_scalar",
            "vopd_independent",
            "vopd_dependent",
            "vopd_mixed",
            "vopd_dequant",
            "q4_selected_dual",
            "q6_x8",
            "dense_q8",
            "dense_q8_single",
        ];
        Self::new(vec![
            recipe(
                "hipfire.dispatch.live_lane_workgroup",
                "Use one 32-lane workgroup when only lane zero is live.",
                exact_kernels(&["dispatch_tiny"]),
                &[],
                &[],
                &["phase:runtime"],
                RecipeAction {
                    workgroup_size: Some(32),
                    ..RecipeAction::default()
                },
                dispatch_workgroup_evidence,
            ),
            recipe(
                "hipfire.decode.wave64",
                "Select wave64 for the correctness-gated decode families that benefit from it.",
                exact_kernels(&wave64_kernels),
                &[],
                &[],
                &["phase:runtime", "shape:preserve_wave32"],
                RecipeAction {
                    wavefront: Some(Wavefront::Wave64),
                    ..RecipeAction::default()
                },
                wave64_evidence,
            ),
            recipe(
                "hipfire.decode.wave64.q4_selected_dual",
                "Select wave64 for the selected-dual Q4 kernel only where correctness is certified.",
                exact_kernels(&["q4_selected_dual"]),
                &[],
                &[],
                &["phase:runtime", "shape:preserve_wave32"],
                RecipeAction {
                    wavefront: Some(Wavefront::Wave64),
                    ..RecipeAction::default()
                },
                q4_wave64_evidence,
            ),
            recipe(
                "hipfire.decode.wave64.q6_x8",
                "Select wave64 for the selected-down Q6 kernel only where correctness is certified.",
                exact_kernels(&["q6_x8"]),
                &[],
                &[],
                &["phase:runtime", "shape:preserve_wave32"],
                RecipeAction {
                    wavefront: Some(Wavefront::Wave64),
                    ..RecipeAction::default()
                },
                q6_wave64_evidence,
            ),
            recipe(
                "hipfire.decode.workgroup64",
                "Pair the selected wave64 decode kernels with a 64-thread workgroup when geometry is tunable.",
                exact_kernels(&["q4_selected_dual", "q6_x8", "dense_q8"]),
                &[],
                &["shape:retunable_workgroup"],
                &["phase:runtime"],
                RecipeAction {
                    workgroup_size: Some(64),
                    ..RecipeAction::default()
                },
                promoted(),
            ),
            recipe(
                "hipfire.rmw.buffer_resource_b32",
                "Lower reviewed 32-bit-offset RMW traffic through AMDGPU buffer resources.",
                exact_kernels(&buffer_rmw_kernels),
                &[],
                &[],
                &["phase:runtime"],
                RecipeAction {
                    lowerings: BTreeSet::from([SourceLowering::BufferResourceB32]),
                    ..RecipeAction::default()
                },
                buffer_resource_evidence,
            ),
            recipe(
                "hipfire.cache.geometry_fma_vmem",
                "Keep geometry inputs on temporal buffer VMEM so Redline can use the lean certified cache boundary.",
                exact_kernels(&["geometry_fma"]),
                &["geometry"],
                &["phase:runtime"],
                &[],
                RecipeAction {
                    kernel_variant: Some("geometry_fma_buffer".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::BufferResourceB32]),
                    ..RecipeAction::default()
                },
                hipx_cache_vmem_promotion_evidence("geometry"),
            ),
            recipe(
                "hipfire.cache.reduction_wave_vmem",
                "Keep wave-reduction inputs on temporal buffer VMEM so Redline can use the lean certified cache boundary.",
                exact_kernels(&["reduction_wave"]),
                &["reduction"],
                &["phase:runtime"],
                &[],
                RecipeAction {
                    kernel_variant: Some("reduction_wave_buffer".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::BufferResourceB32]),
                    ..RecipeAction::default()
                },
                hipx_cache_vmem_promotion_evidence("reduction"),
            ),
            recipe(
                "hipfire.cache.mfp4_e8_u4_temporal_buffer",
                "Lower MFP4-E8 U4 decode weights through temporal cpol0 buffer loads.",
                exact_kernels(&["gemv_mfp4g32_e8_soa_u4"]),
                &["dense-e8"],
                &["phase:decode", "quant:mfp4-e8"],
                &[],
                RecipeAction {
                    kernel_variant: Some("temporal_buffer_cpol0".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::TemporalBufferB32]),
                    ..RecipeAction::default()
                },
                gfx1151_e8_temporal_buffer_evidence(),
            ),
            recipe(
                "hipfire.cross_lane.wave32_xor_exchange",
                "Lower wave32 XOR exchanges through architecture-reviewed DPP, ds_swizzle, and permlanex16 forms.",
                exact_kernels(&["deepseek4_attn_swa_topk_scoregrid_f32_buf"]),
                &["attention"],
                &["phase:decode", "reduction:max"],
                &[],
                RecipeAction {
                    kernel_variant: Some("scoregrid_xlane".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::Wave32XorExchange]),
                    ..RecipeAction::default()
                },
                gfx1151_xor_shuffle_evidence(),
            ),
            recipe(
                "hipfire.memory.topk_gather_lds_transpose",
                "Stage DeepSeek top-k gather tiles through padded LDS so row-major reads and dimension-major writes are both coalesced.",
                exact_kernels(&["deepseek4_topk_kv_gather_f32_buf"]),
                &["attention-gather"],
                &["phase:decode", "layout:row-to-column"],
                &[],
                RecipeAction {
                    workgroup_size: Some(256),
                    kernel_variant: Some("tiled_32x32_gfx1151".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::LdsTranspose32x32]),
                    ..RecipeAction::default()
                },
                gfx1151_topk_gather_tiled_evidence(),
            ),
            recipe(
                "hipfire.memory.interleave.independent_buffer_output",
                "Use buffer-resource output RMW for independent interleave throughput.",
                exact_kernels(&["memory_interleave4"]),
                &["memory-waitcnt"],
                &["timing:independent_throughput"],
                &[],
                RecipeAction {
                    kernel_variant: Some("memory_interleave4_buffer".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::BufferOutputRmw]),
                    ..RecipeAction::default()
                },
                promoted(),
            ),
            recipe(
                "hipfire.memory.interleave.single_b128_block64",
                "Use aligned B128 input loads and one 64-lane workgroup for single-dispatch interleave.",
                exact_kernels(&["memory_interleave4"]),
                &["memory-waitcnt"],
                &["timing:single_kernel_aggressive"],
                &["experiment:interleave_b32"],
                RecipeAction {
                    workgroup_size: Some(64),
                    kernel_variant: Some("memory_interleave4_block64".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::AlignedBufferB128]),
                    ..RecipeAction::default()
                },
                promoted(),
            ),
            recipe(
                "hipfire.memory.interleave.single_independent_b32",
                "Expose four independently completable B32 VMEM loads for the experimental interleave control.",
                exact_kernels(&["memory_interleave4"]),
                &["memory-waitcnt"],
                &[
                    "timing:single_kernel_aggressive",
                    "experiment:interleave_b32",
                ],
                &[],
                RecipeAction {
                    workgroup_size: Some(64),
                    kernel_variant: Some("memory_interleave4_block64_b32".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::IndependentBufferB32]),
                    ..RecipeAction::default()
                },
                Vec::new(),
            ),
            recipe(
                "hipfire.packed_dot.aligned_b128",
                "Load aligned packed dot tiles as B128 vectors before lane-local dot products.",
                exact_kernels(&["dot_q8", "dot_q4", "dot_q6", "dot_scalar"]),
                &["packed-dot"],
                &[],
                &["phase:runtime"],
                RecipeAction {
                    lowerings: BTreeSet::from([SourceLowering::AlignedBufferB128]),
                    ..RecipeAction::default()
                },
                promoted(),
            ),
            recipe(
                "hipfire.vopd.independent_unroll8",
                "Expose eight independent VOPD iterations to LLVM's machine scheduler.",
                exact_kernels(&["vopd_independent", "vopd_mixed"]),
                &["vopd"],
                &[],
                &["phase:runtime"],
                RecipeAction {
                    lowerings: BTreeSet::from([SourceLowering::Unroll(8)]),
                    ..RecipeAction::default()
                },
                promoted(),
            ),
            recipe(
                "hipfire.vopd.dequant_chunk16",
                "Use the sixteen-iteration dequant chunk variant.",
                exact_kernels(&["vopd_dequant"]),
                &["vopd"],
                &["phase:runtime"],
                &[],
                RecipeAction {
                    kernel_variant: Some("vopd_dequant_chunk16".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::Chunk(16)]),
                    ..RecipeAction::default()
                },
                dequant_chunk16_evidence,
            ),
            recipe(
                "hipfire.vopd.mixed_paired_hash",
                "Pair the independent integer hash chains before the mixed VOPD recurrence.",
                exact_kernels(&["vopd_mixed"]),
                &["vopd"],
                &["phase:runtime", "experiment:mixed_paired_hash"],
                &[],
                RecipeAction {
                    kernel_variant: Some("vopd_mixed_pair".to_owned()),
                    lowerings: BTreeSet::from([SourceLowering::PairedIntegerHash]),
                    ..RecipeAction::default()
                },
                Vec::new(),
            ),
        ])
    }
}

fn ledger_recipe_ids(row: &serde_json::Value) -> Vec<String> {
    if let Some(recipe) = row.get("radiowave_recipe").and_then(|value| value.as_str()) {
        return vec![recipe.to_owned()];
    }
    row.get("radiowave_recipes")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect()
}

fn json_f64(row: &serde_json::Value, fields: &[&str]) -> Option<f64> {
    fields
        .iter()
        .find_map(|field| row.get(field).and_then(|value| value.as_f64()))
}

fn exact_kernels(kernels: &[&str]) -> Vec<KernelSelector> {
    kernels
        .iter()
        .map(|kernel| KernelSelector::Exact((*kernel).to_owned()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn recipe(
    id: &str,
    summary: &str,
    kernels: Vec<KernelSelector>,
    families: &[&str],
    required_tags: &[&str],
    forbidden_tags: &[&str],
    action: RecipeAction,
    evidence: Vec<RecipeEvidence>,
) -> OptimizationRecipe {
    OptimizationRecipe {
        id: id.to_owned(),
        revision: 1,
        summary: summary.to_owned(),
        predicate: RecipePredicate {
            kernels,
            families: families.iter().map(|value| (*value).to_owned()).collect(),
            required_tags: required_tags
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            forbidden_tags: forbidden_tags
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
        },
        action,
        evidence,
    }
}

fn hipfire_6409_evidence() -> RecipeEvidence {
    RecipeEvidence {
        architecture: "gfx1201".to_owned(),
        verdict: EvidenceVerdict::Promoted,
        correctness_pass: true,
        artifact: HIPFIRE_6409_EVIDENCE.to_owned(),
        samples_per_row: Some(14),
        throughput_delta_pct: None,
        duration_delta_pct: None,
        note: "correctness-gated two-replicate microbenchmark promotion".to_owned(),
    }
}

fn hipx_portability_evidence() -> Vec<RecipeEvidence> {
    ["gfx1010", "gfx1030", "gfx1100", "gfx1151"]
        .into_iter()
        .map(|architecture| RecipeEvidence {
            architecture: architecture.to_owned(),
            verdict: EvidenceVerdict::CorrectnessOnly,
            correctness_pass: true,
            artifact: HIPX_PORTABILITY_EVIDENCE.to_owned(),
            samples_per_row: Some(1),
            throughput_delta_pct: None,
            duration_delta_pct: None,
            note: "live buffer B32 load/store probe passed; no performance promotion".to_owned(),
        })
        .collect()
}

fn hipx_wave64_promotion_evidence() -> Vec<RecipeEvidence> {
    [
        (
            "crates/radiowave/tests/artifacts/recipes/hipfire-6409-gfx1151-2026-07-14-radiowave-probe-wave64-vopd-results.json",
            -12.11,
            "wave64 improved all 16 VOPD rows; median Redline duration fell 12.11%",
        ),
        (
            "crates/radiowave/tests/artifacts/recipes/hipfire-6409-gfx1151-2026-07-14-radiowave-probe-wave64-interleave-results.json",
            -15.96,
            "wave64 improved all four interleave rows; median Redline duration fell 15.96%",
        ),
    ]
    .into_iter()
    .map(|(artifact, duration_delta_pct, note)| RecipeEvidence {
        architecture: "gfx1151".to_owned(),
        verdict: EvidenceVerdict::Promoted,
        correctness_pass: true,
        artifact: artifact.to_owned(),
        samples_per_row: Some(7),
        throughput_delta_pct: None,
        duration_delta_pct: Some(duration_delta_pct),
        note: note.to_owned(),
    })
    .collect()
}

fn hipx_dequant_chunk16_promotion_evidence() -> Vec<RecipeEvidence> {
    [
        ("gfx1151", -10.00),
        ("gfx1030", -14.39),
        ("gfx1010", -14.01),
    ]
    .into_iter()
    .map(|(architecture, duration_delta_pct)| RecipeEvidence {
        architecture: architecture.to_owned(),
        verdict: EvidenceVerdict::Promoted,
        correctness_pass: true,
        artifact: format!(
            "crates/radiowave/tests/artifacts/recipes/hipfire-6409-{architecture}-2026-07-14-radiowave-probe-dequant-chunk16-results.json"
        ),
        samples_per_row: Some(7),
        throughput_delta_pct: None,
        duration_delta_pct: Some(duration_delta_pct),
        note: "chunk16 improved all four dequant rows against the plain Redline baseline"
            .to_owned(),
    })
    .collect()
}

fn hipx_cache_vmem_promotion_evidence(family: &str) -> Vec<RecipeEvidence> {
    let measurements = match family {
        "geometry" => [("gfx1100", -8.80, -7.18), ("gfx1151", -4.43, -4.34)],
        "reduction" => [("gfx1100", -4.67, -6.25), ("gfx1151", -4.30, -4.46)],
        _ => unreachable!("built-in cache evidence has a fixed family"),
    };
    measurements
        .into_iter()
        .flat_map(|(architecture, first_delta, second_delta)| {
            [("radiowave-cache-vmem", first_delta),
             ("radiowave-cache-vmem-replicate2", second_delta)]
                .into_iter()
                .map(move |(run, duration_delta_pct)| RecipeEvidence {
                    architecture: architecture.to_owned(),
                    verdict: EvidenceVerdict::Promoted,
                    correctness_pass: true,
                    artifact: format!(
                        "crates/radiowave/tests/artifacts/recipes/hipfire-6409-{architecture}-2026-07-14-{run}-candidate-{family}-results.json"
                    ),
                    samples_per_row: Some(7),
                    throughput_delta_pct: None,
                    duration_delta_pct: Some(duration_delta_pct),
                    note: format!(
                        "temporal buffer VMEM improved every or all but one {family} row; reverse-order replicate confirmed the median duration win"
                    ),
                })
        })
        .collect()
}

fn hipx_dispatch_workgroup_promotion_evidence() -> Vec<RecipeEvidence> {
    [("gfx1100", -36.27), ("gfx1151", -58.34)]
        .into_iter()
        .flat_map(|(architecture, duration_delta_pct)| {
            [
                "radiowave-dispatch-wg32",
                "radiowave-dispatch-wg32-replicate2",
            ]
            .into_iter()
            .map(move |run| RecipeEvidence {
                architecture: architecture.to_owned(),
                verdict: EvidenceVerdict::Promoted,
                correctness_pass: true,
                artifact: format!(
                    "crates/radiowave/tests/artifacts/recipes/hipfire-6409-{architecture}-2026-07-14-{run}-candidate-results.json"
                ),
                samples_per_row: Some(7),
                throughput_delta_pct: None,
                duration_delta_pct: Some(duration_delta_pct),
                note: "32-thread live-lane workgroup preserved count-sweep latency and reduced the median large-grid Redline duration in both orderings".to_owned(),
            })
        })
        .collect()
}

fn gfx1151_e8_temporal_buffer_evidence() -> Vec<RecipeEvidence> {
    vec![RecipeEvidence {
        architecture: "gfx1151".to_owned(),
        verdict: EvidenceVerdict::Promoted,
        correctness_pass: true,
        artifact:
            "crates/radiowave/tests/artifacts/hipfire-deepseek4-mq2r-gfx1151-2026-07-25-e8-temporal-buffer.json"
                .to_owned(),
        samples_per_row: Some(3),
        throughput_delta_pct: Some(2.937766),
        duration_delta_pct: None,
        note: "2048/510 top-k-6 ABBA improved 25.87 to 26.63 tok/s with byte-identical output; temporal cpol0 won while non-temporal cpol20/cpol22 regressed large tensors by 40-50%"
            .to_owned(),
    }]
}

fn gfx1151_xor_shuffle_evidence() -> Vec<RecipeEvidence> {
    vec![RecipeEvidence {
        architecture: "gfx1151".to_owned(),
        verdict: EvidenceVerdict::CorrectnessOnly,
        correctness_pass: true,
        artifact:
            "crates/radiowave/tests/artifacts/hipfire-deepseek4-gfx1151-2026-07-25-xor-shuffle.json"
                .to_owned(),
        samples_per_row: Some(3),
        throughput_delta_pct: Some(0.637858),
        duration_delta_pct: Some(-0.633817),
        note: "41,120 raw-bit comparisons proved the stride table: xor1/xor2 DPP quad-perm, xor4/xor8 ds_swizzle, xor16 permlanex16 selectors 0x76543210/0xfedcba98; the score-grid screen was bit-exact but projected only 0.118% whole-token gain, so the lowering remains an unpromoted candidate"
            .to_owned(),
    }]
}

fn gfx1151_topk_gather_tiled_evidence() -> Vec<RecipeEvidence> {
    vec![RecipeEvidence {
        architecture: "gfx1151".to_owned(),
        verdict: EvidenceVerdict::Promoted,
        correctness_pass: true,
        artifact:
            "crates/radiowave/tests/artifacts/hipfire-deepseek4-mq2r-gfx1151-2026-07-25-topk-gather-tiled.json"
                .to_owned(),
        samples_per_row: Some(3),
        throughput_delta_pct: Some(3.363316),
        duration_delta_pct: Some(-72.760),
        note: "2048/510 top-k-6 ABBA improved 25.57 to 26.43 tok/s with byte-identical output; a 21-layer cold-set microbench reduced gather latency from 72.311 to 19.697 us/call"
            .to_owned(),
    }]
}

fn hipx_wave64_rejection_evidence(kernel: &str) -> Vec<RecipeEvidence> {
    ["gfx1010", "gfx1030", "gfx1100", "gfx1151"]
        .into_iter()
        .map(|architecture| RecipeEvidence {
            architecture: architecture.to_owned(),
            verdict: EvidenceVerdict::Rejected,
            correctness_pass: false,
            artifact: format!(
                "crates/radiowave/tests/artifacts/recipes/hipfire-6409-{architecture}-2026-07-14-radiowave-candidate-discovery-results.json"
            ),
            samples_per_row: Some(3),
            throughput_delta_pct: None,
            duration_delta_pct: None,
            note: format!(
                "{kernel} wave64 failed the HipEngine-parity CPU oracle; retain wave32"
            ),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promoted_arch_applies_but_unseen_arch_only_emits_candidates() {
        let catalog = RecipeCatalog::builtin_hipfire_6409();
        let workload = WorkloadDescriptor::new("dispatch_tiny", "dispatch-grid");
        let selected = catalog
            .select("gfx1201", workload.clone(), SelectionMode::Certified)
            .unwrap();
        assert_eq!(selected.plan.workgroup_size, Some(32));
        assert!(
            selected
                .applied_recipes
                .contains(&"hipfire.dispatch.live_lane_workgroup".to_owned())
        );

        let unseen = catalog
            .select("gfx942", workload, SelectionMode::Certified)
            .unwrap();
        assert_eq!(unseen.plan, OptimizationPlan::default());
        assert!(
            unseen
                .candidate_recipes
                .contains(&"hipfire.dispatch.live_lane_workgroup".to_owned())
        );
    }

    #[test]
    fn candidate_mode_is_architecture_neutral() {
        let selected = RecipeCatalog::builtin_hipfire_6409()
            .select(
                "future-architecture",
                WorkloadDescriptor::new("memory_interleave4", "memory-waitcnt")
                    .tag("timing:independent_throughput"),
                SelectionMode::Candidates,
            )
            .unwrap();
        assert_eq!(selected.plan.wavefront, Some(Wavefront::Wave64));
        assert_eq!(
            selected.plan.kernel_variant.as_deref(),
            Some("memory_interleave4_buffer")
        );
        assert!(
            selected
                .plan
                .lowerings
                .contains(&SourceLowering::BufferOutputRmw)
        );
    }

    #[test]
    fn architecture_rejection_blocks_an_unsafe_candidate_without_hiding_it() {
        let catalog = RecipeCatalog::builtin_hipfire_6409();
        let rejected = catalog
            .select(
                "gfx1030",
                WorkloadDescriptor::new("q4_selected_dual", "q4-selected-dual"),
                SelectionMode::Candidates,
            )
            .unwrap();
        assert_eq!(rejected.plan.wavefront, None);
        assert!(
            rejected
                .rejected_recipes
                .contains(&"hipfire.decode.wave64.q4_selected_dual".to_owned())
        );

        let promoted = catalog
            .select(
                "gfx1201",
                WorkloadDescriptor::new("q4_selected_dual", "q4-selected-dual"),
                SelectionMode::Certified,
            )
            .unwrap();
        assert_eq!(promoted.plan.wavefront, Some(Wavefront::Wave64));
    }

    #[test]
    fn correctness_only_evidence_never_promotes_a_recipe() {
        let catalog = RecipeCatalog::builtin_hipfire_6409();
        let recipe = catalog
            .recipes
            .iter()
            .find(|recipe| recipe.id == "hipfire.rmw.buffer_resource_b32")
            .unwrap();
        assert!(recipe.evidence.iter().any(|evidence| {
            evidence.architecture == "gfx1030"
                && evidence.verdict == EvidenceVerdict::CorrectnessOnly
                && evidence.correctness_pass
        }));
        assert!(!recipe.is_promoted_for("gfx1030"));
    }

    #[test]
    fn hipx_promotion_selects_chunk16_without_promoting_unroll8() {
        let catalog = RecipeCatalog::builtin_hipfire_6409();
        let dequant = catalog
            .select(
                "gfx1030",
                WorkloadDescriptor::new("vopd_dequant", "vopd").tag("phase:runtime"),
                SelectionMode::Certified,
            )
            .unwrap();
        assert_eq!(
            dequant.plan.kernel_variant.as_deref(),
            Some("vopd_dequant_chunk16")
        );
        assert!(
            dequant
                .applied_recipes
                .contains(&"hipfire.vopd.dequant_chunk16".to_owned())
        );

        let unroll = catalog
            .select(
                "gfx1030",
                WorkloadDescriptor::new("vopd_independent", "vopd"),
                SelectionMode::Certified,
            )
            .unwrap();
        assert!(unroll.applied_recipes.is_empty());
        assert!(unroll.plan.lowerings.is_empty());
    }

    #[test]
    fn cache_lowering_promotes_only_on_architectures_with_proof() {
        let selected = RecipeCatalog::builtin_hipfire_6409()
            .select(
                "gfx1100",
                WorkloadDescriptor::new("geometry_fma", "geometry").tag("phase:runtime"),
                SelectionMode::Candidates,
            )
            .unwrap();
        assert_eq!(
            selected.plan.kernel_variant.as_deref(),
            Some("geometry_fma_buffer")
        );
        assert!(
            selected
                .plan
                .lowerings
                .contains(&SourceLowering::BufferResourceB32)
        );

        let certified = RecipeCatalog::builtin_hipfire_6409()
            .select(
                "gfx1100",
                WorkloadDescriptor::new("geometry_fma", "geometry").tag("phase:runtime"),
                SelectionMode::Certified,
            )
            .unwrap();
        assert!(
            certified
                .applied_recipes
                .contains(&"hipfire.cache.geometry_fma_vmem".to_owned())
        );

        let unseen = RecipeCatalog::builtin_hipfire_6409()
            .select(
                "gfx1030",
                WorkloadDescriptor::new("geometry_fma", "geometry").tag("phase:runtime"),
                SelectionMode::Certified,
            )
            .unwrap();
        assert!(unseen.applied_recipes.is_empty());
    }

    #[test]
    fn gfx1151_e8_temporal_buffer_requires_exact_arch_proof() {
        let workload = WorkloadDescriptor::new("gemv_mfp4g32_e8_soa_u4", "dense-e8")
            .tag("phase:decode")
            .tag("quant:mfp4-e8");
        let certified = RecipeCatalog::builtin_hipfire_6409()
            .select("gfx1151", workload.clone(), SelectionMode::Certified)
            .unwrap();
        assert_eq!(
            certified.plan.kernel_variant.as_deref(),
            Some("temporal_buffer_cpol0")
        );
        assert!(
            certified
                .plan
                .lowerings
                .contains(&SourceLowering::TemporalBufferB32)
        );
        assert!(
            certified
                .applied_recipes
                .contains(&"hipfire.cache.mfp4_e8_u4_temporal_buffer".to_owned())
        );

        let unseen = RecipeCatalog::builtin_hipfire_6409()
            .select("gfx1201", workload, SelectionMode::Certified)
            .unwrap();
        assert!(unseen.applied_recipes.is_empty());
    }

    #[test]
    fn gfx1151_xor_shuffle_is_proven_but_not_performance_promoted() {
        let workload =
            WorkloadDescriptor::new("deepseek4_attn_swa_topk_scoregrid_f32_buf", "attention")
                .tag("phase:decode")
                .tag("reduction:max");
        let certified = RecipeCatalog::builtin_hipfire_6409()
            .select("gfx1151", workload.clone(), SelectionMode::Certified)
            .unwrap();
        assert!(certified.applied_recipes.is_empty());
        assert!(certified.plan.lowerings.is_empty());

        let candidate = RecipeCatalog::builtin_hipfire_6409()
            .select("gfx1151", workload, SelectionMode::Candidates)
            .unwrap();
        assert!(candidate
            .candidate_recipes
            .contains(&"hipfire.cross_lane.wave32_xor_exchange".to_owned()));
        assert!(candidate
            .plan
            .lowerings
            .contains(&SourceLowering::Wave32XorExchange));
    }

    #[test]
    fn gfx1151_topk_gather_lds_transpose_requires_exact_arch_proof() {
        let workload =
            WorkloadDescriptor::new("deepseek4_topk_kv_gather_f32_buf", "attention-gather")
                .tag("phase:decode")
                .tag("layout:row-to-column");
        let certified = RecipeCatalog::builtin_hipfire_6409()
            .select("gfx1151", workload.clone(), SelectionMode::Certified)
            .unwrap();
        assert_eq!(certified.plan.workgroup_size, Some(256));
        assert_eq!(
            certified.plan.kernel_variant.as_deref(),
            Some("tiled_32x32_gfx1151")
        );
        assert!(
            certified
                .plan
                .lowerings
                .contains(&SourceLowering::LdsTranspose32x32)
        );
        assert!(
            certified
                .applied_recipes
                .contains(&"hipfire.memory.topk_gather_lds_transpose".to_owned())
        );

        let unseen = RecipeCatalog::builtin_hipfire_6409()
            .select("gfx1201", workload, SelectionMode::Certified)
            .unwrap();
        assert!(unseen.applied_recipes.is_empty());
    }

    #[test]
    fn experimental_b32_does_not_conflict_with_promoted_b128() {
        let selected = RecipeCatalog::builtin_hipfire_6409()
            .select(
                "gfx1201",
                WorkloadDescriptor::new("memory_interleave4", "memory-waitcnt")
                    .tag("timing:single_kernel_aggressive")
                    .tag("experiment:interleave_b32"),
                SelectionMode::Candidates,
            )
            .unwrap();
        assert_eq!(selected.plan.workgroup_size, Some(64));
        assert_eq!(
            selected.plan.kernel_variant.as_deref(),
            Some("memory_interleave4_block64_b32")
        );
    }

    #[test]
    fn autoresearch_win_promotes_a_recipe_for_a_new_architecture() {
        let mut catalog = RecipeCatalog::builtin_hipfire_6409();
        let ledger = r#"{"arch":"gfx942","kernel":"dispatch_tiny","verdict":"WIN","radiowave_recipe":"hipfire.dispatch.live_lane_workgroup","measurement_hash":"abc123","tok_delta_pct":4.5,"dur_delta_pct":-4.2}"#;
        assert_eq!(catalog.ingest_autoresearch_jsonl(ledger).unwrap(), 1);
        let selected = catalog
            .select(
                "gfx942",
                WorkloadDescriptor::new("dispatch_tiny", "dispatch-grid"),
                SelectionMode::Certified,
            )
            .unwrap();
        assert_eq!(selected.plan.workgroup_size, Some(32));
    }

    #[test]
    fn catalog_json_round_trip_is_stable() {
        let catalog = RecipeCatalog::builtin_hipfire_6409();
        let encoded = catalog.to_json_pretty().unwrap();
        let decoded = RecipeCatalog::from_json(&encoded).unwrap();
        assert_eq!(decoded, catalog);
    }

    #[test]
    fn builtin_recipe_artifacts_are_tracked_fixtures() {
        const ARCHIVE_SHA: &str =
            "d4d0911e751e891b952f2983c0a78e05c304b854321ce5f438794ca7a8abda2b";
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root");
        let catalog = RecipeCatalog::builtin_hipfire_6409();
        let mut failures = Vec::new();

        for recipe in &catalog.recipes {
            for evidence in &recipe.evidence {
                let artifact = evidence.artifact.as_str();
                let normalized = artifact.replace('\\', "/");
                if normalized.contains("examples/") || normalized.contains("/results/") {
                    failures.push(format!(
                        "{}: artifact path still depends on examples/**/results: {artifact}",
                        recipe.id
                    ));
                    continue;
                }
                let path = repo_root.join(artifact);
                if !path.is_file() {
                    failures.push(format!("{}: missing fixture {artifact}", recipe.id));
                    continue;
                }
                // Compact provenance fixtures under recipes/; leave other tracked
                // artifacts (e.g. hipx portability) as plain existence checks.
                if !normalized.contains("/artifacts/recipes/") {
                    continue;
                }
                let raw = match std::fs::read_to_string(&path) {
                    Ok(raw) => raw,
                    Err(err) => {
                        failures.push(format!("{}: read {artifact}: {err}", recipe.id));
                        continue;
                    }
                };
                let value: serde_json::Value = match serde_json::from_str(&raw) {
                    Ok(value) => value,
                    Err(err) => {
                        failures.push(format!("{}: parse {artifact}: {err}", recipe.id));
                        continue;
                    }
                };
                let obj = match value.as_object() {
                    Some(obj) => obj,
                    None => {
                        failures.push(format!("{}: {artifact} is not a JSON object", recipe.id));
                        continue;
                    }
                };
                let schema_version = obj.get("schema_version").and_then(|v| v.as_u64());
                if schema_version != Some(1) {
                    failures.push(format!(
                        "{}: {artifact} schema_version must be 1, got {schema_version:?}",
                        recipe.id
                    ));
                }
                if obj.get("kind").and_then(|v| v.as_str()) != Some("radiowave_recipe_provenance")
                {
                    failures.push(format!(
                        "{}: {artifact} kind must be radiowave_recipe_provenance",
                        recipe.id
                    ));
                }
                let original_path = obj
                    .get("original_result_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if original_path.is_empty() {
                    failures.push(format!(
                        "{}: {artifact} missing original_result_path",
                        recipe.id
                    ));
                }
                let original_size = obj
                    .get("original_byte_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                if original_size == 0 {
                    failures.push(format!(
                        "{}: {artifact} original_byte_size must be nonzero",
                        recipe.id
                    ));
                }
                for field in ["original_sha256", "external_archive_sha256"] {
                    let hash = obj.get(field).and_then(|v| v.as_str()).unwrap_or("");
                    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                        failures.push(format!(
                            "{}: {artifact} {field} must be 64-hex, got {hash:?}",
                            recipe.id
                        ));
                    }
                }
                if obj.get("external_archive_sha256").and_then(|v| v.as_str()) != Some(ARCHIVE_SHA)
                {
                    failures.push(format!(
                        "{}: {artifact} external_archive_sha256 mismatch",
                        recipe.id
                    ));
                }
                let note = obj.get("note").and_then(|v| v.as_str()).unwrap_or("");
                if note.is_empty() {
                    failures.push(format!("{}: {artifact} missing note", recipe.id));
                }
                // Compact records must not re-embed full result payloads.
                if obj.contains_key("rows")
                    || obj.contains_key("measurements")
                    || obj.contains_key("results")
                    || raw.len() > 4096
                {
                    failures.push(format!(
                        "{}: {artifact} still looks like a raw result payload ({} bytes)",
                        recipe.id,
                        raw.len()
                    ));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "recipe artifact provenance contract failed:\n{}",
            failures.join("\n")
        );
    }
}
