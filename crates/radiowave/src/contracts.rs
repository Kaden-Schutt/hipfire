// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Fail-closed emitted-resource contracts for campaign candidates.

use crate::{ArchProfile, CodeObjectInspection, IsaVersion, KernelReport, Wavefront};
use serde::{Deserialize, Serialize};

pub const RESOURCE_CONTRACT_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResourceRejection {
    MissingCodeObjectIdentity,
    BundleArchitectureMismatch {
        expected: String,
        actual: String,
    },
    IdentityArchitectureMismatch {
        expected: String,
        actual: String,
    },
    MissingElfMachineId,
    ElfMachineIdMismatch {
        expected: u32,
        actual: u32,
    },
    MissingIsaVersion,
    IsaVersionMismatch {
        expected: IsaVersion,
        actual: IsaVersion,
    },
    MissingKernel {
        symbol: String,
    },
    IncumbentWavefrontMismatch {
        expected: u32,
        actual: u32,
    },
    WavefrontMismatch {
        expected: u32,
        actual: u32,
    },
    RegisterSpill {
        vgpr_spills: u32,
        sgpr_spills: u32,
    },
    ScratchMemory {
        bytes: u32,
    },
    OccupancyRegression {
        incumbent_waves_per_simd: u32,
        candidate_waves_per_simd: u32,
        incumbent_vgprs: u32,
        candidate_vgprs: u32,
    },
    StaticMemoryClauseLimit {
        maximum: u32,
        actual: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceAssessment {
    pub schema_version: u32,
    pub profile: ArchProfile,
    pub kernel: String,
    #[serde(default)]
    pub required_wavefront_size: u32,
    pub accepted: bool,
    pub incumbent_waves_per_simd: Option<u32>,
    pub candidate_waves_per_simd: Option<u32>,
    pub rejections: Vec<ResourceRejection>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceContract {
    pub profile: ArchProfile,
    /// A narrow per-target override. `None` preserves the architecture
    /// profile's default wavefront.
    #[serde(default)]
    pub required_wavefront: Option<Wavefront>,
}

impl ResourceContract {
    pub const fn new(profile: ArchProfile) -> Self {
        Self {
            profile,
            required_wavefront: None,
        }
    }

    /// Require a wavefront for one target without changing the architecture's
    /// default contract. The incumbent must use the same wavefront or the
    /// assessment fails before occupancy classes are compared.
    pub const fn require_wavefront(mut self, wavefront: Wavefront) -> Self {
        self.required_wavefront = Some(wavefront);
        self
    }

    pub const fn required_wavefront_size(self) -> u32 {
        match self.required_wavefront {
            Some(wavefront) => wavefront.width(),
            None => self.profile.required_wavefront_size(),
        }
    }

    /// Assess a candidate against the resource occupancy of the currently
    /// promoted kernel.  Register growth within one occupancy plateau is
    /// permitted; a fall to a lower resident-wave class is not.
    pub fn assess(
        self,
        inspection: &CodeObjectInspection,
        symbol: &str,
        incumbent: &KernelReport,
    ) -> ResourceAssessment {
        let canonical = symbol.strip_suffix(".kd").unwrap_or(symbol);
        let mut rejections = Vec::new();
        let required_wavefront_size = self.required_wavefront_size();
        let incumbent_wavefront_compatible = incumbent.wavefront_size == required_wavefront_size;
        if !incumbent_wavefront_compatible {
            rejections.push(ResourceRejection::IncumbentWavefrontMismatch {
                expected: required_wavefront_size,
                actual: incumbent.wavefront_size,
            });
        }

        let bundle_architecture = architecture_from_bundle_target(&inspection.bundle_target)
            .unwrap_or_else(|| inspection.bundle_target.clone());
        if bundle_architecture != self.profile.arch() {
            rejections.push(ResourceRejection::BundleArchitectureMismatch {
                expected: self.profile.arch().to_owned(),
                actual: bundle_architecture,
            });
        }

        if let Some(identity) = &inspection.identity {
            if identity.architecture != self.profile.arch() {
                rejections.push(ResourceRejection::IdentityArchitectureMismatch {
                    expected: self.profile.arch().to_owned(),
                    actual: identity.architecture.clone(),
                });
            }
            match identity.elf_machine_id {
                Some(actual) if actual != self.profile.elf_machine_id() => {
                    rejections.push(ResourceRejection::ElfMachineIdMismatch {
                        expected: self.profile.elf_machine_id(),
                        actual,
                    });
                }
                None => rejections.push(ResourceRejection::MissingElfMachineId),
                _ => {}
            }
            match identity.isa {
                Some(actual) if actual != self.profile.isa() => {
                    rejections.push(ResourceRejection::IsaVersionMismatch {
                        expected: self.profile.isa(),
                        actual,
                    });
                }
                None => rejections.push(ResourceRejection::MissingIsaVersion),
                _ => {}
            }
        } else {
            rejections.push(ResourceRejection::MissingCodeObjectIdentity);
        }

        let candidate = inspection
            .kernel(symbol)
            .or_else(|| inspection.kernel(canonical));
        let incumbent_waves = self
            .profile
            .vgpr_limited_waves(incumbent.vgpr_count, incumbent.wavefront_size);
        let incumbent_waves_per_simd = Some(incumbent_waves);
        let mut candidate_waves_per_simd = None;
        if let Some(candidate) = candidate {
            let candidate_wavefront_compatible =
                candidate.wavefront_size == required_wavefront_size;
            if !candidate_wavefront_compatible {
                rejections.push(ResourceRejection::WavefrontMismatch {
                    expected: required_wavefront_size,
                    actual: candidate.wavefront_size,
                });
            }
            if candidate.vgpr_spill_count != 0 || candidate.sgpr_spill_count != 0 {
                rejections.push(ResourceRejection::RegisterSpill {
                    vgpr_spills: candidate.vgpr_spill_count,
                    sgpr_spills: candidate.sgpr_spill_count,
                });
            }
            if candidate.private_segment_fixed_size != 0 {
                rejections.push(ResourceRejection::ScratchMemory {
                    bytes: candidate.private_segment_fixed_size,
                });
            }
            if candidate.instructions.memory_clause_instructions
                > self.profile.max_static_memory_clauses()
            {
                rejections.push(ResourceRejection::StaticMemoryClauseLimit {
                    maximum: self.profile.max_static_memory_clauses(),
                    actual: candidate.instructions.memory_clause_instructions,
                });
            }

            let candidate_waves = self
                .profile
                .vgpr_limited_waves(candidate.vgpr_count, candidate.wavefront_size);
            candidate_waves_per_simd = Some(candidate_waves);
            if incumbent_wavefront_compatible
                && candidate_wavefront_compatible
                && candidate_waves < incumbent_waves
            {
                rejections.push(ResourceRejection::OccupancyRegression {
                    incumbent_waves_per_simd: incumbent_waves,
                    candidate_waves_per_simd: candidate_waves,
                    incumbent_vgprs: incumbent.vgpr_count,
                    candidate_vgprs: candidate.vgpr_count,
                });
            }
        } else {
            rejections.push(ResourceRejection::MissingKernel {
                symbol: symbol.to_owned(),
            });
        }

        ResourceAssessment {
            schema_version: RESOURCE_CONTRACT_SCHEMA_VERSION,
            profile: self.profile,
            kernel: symbol.to_owned(),
            required_wavefront_size,
            accepted: rejections.is_empty(),
            incumbent_waves_per_simd,
            candidate_waves_per_simd,
            rejections,
        }
    }
}

fn architecture_from_bundle_target(target: &str) -> Option<String> {
    let start = target.rfind("gfx")?;
    let arch = target[start..]
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .next()
        .unwrap_or_default();
    (!arch.is_empty()).then(|| arch.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodeObjectIdentity, InstructionStats};

    fn report(vgprs: u32) -> KernelReport {
        report_for_wavefront(vgprs, 32)
    }

    fn report_for_wavefront(vgprs: u32, wavefront_size: u32) -> KernelReport {
        KernelReport {
            name: "candidate".to_owned(),
            wavefront_size,
            vgpr_count: vgprs,
            sgpr_count: 20,
            ..KernelReport::default()
        }
    }

    fn inspection(candidate: KernelReport) -> CodeObjectInspection {
        CodeObjectInspection {
            bundle_target: "hipv4-amdgcn-amd-amdhsa--gfx1151".to_owned(),
            identity: Some(CodeObjectIdentity {
                architecture: "gfx1151".to_owned(),
                elf_machine_id: Some(0x4a),
                isa: Some(IsaVersion::new(11, 5, 1)),
            }),
            kernels: vec![candidate],
        }
    }

    #[test]
    fn accepts_register_growth_within_the_same_gfx1151_plateau() {
        let assessment = ResourceContract::new(ArchProfile::Gfx1151).assess(
            &inspection(report(96)),
            "candidate",
            &report(82),
        );
        assert!(assessment.accepted, "{:?}", assessment.rejections);
        assert_eq!(assessment.incumbent_waves_per_simd, Some(16));
        assert_eq!(assessment.candidate_waves_per_simd, Some(16));
    }

    #[test]
    fn default_contract_remains_wave32() {
        let assessment = ResourceContract::new(ArchProfile::Gfx1151).assess(
            &inspection(report_for_wavefront(82, 64)),
            "candidate",
            &report_for_wavefront(82, 64),
        );
        assert!(!assessment.accepted);
        assert!(assessment.rejections.iter().any(|rejection| matches!(
            rejection,
            ResourceRejection::WavefrontMismatch {
                expected: 32,
                actual: 64
            }
        )));
    }

    #[test]
    fn explicit_wave64_override_accepts_an_incumbent_compatible_router() {
        let assessment = ResourceContract::new(ArchProfile::Gfx1151)
            .require_wavefront(Wavefront::Wave64)
            .assess(
                &inspection(report_for_wavefront(96, 64)),
                "candidate",
                &report_for_wavefront(82, 64),
            );
        assert!(assessment.accepted, "{:?}", assessment.rejections);
        assert_eq!(assessment.required_wavefront_size, 64);
        assert_eq!(assessment.incumbent_waves_per_simd, Some(8));
        assert_eq!(assessment.candidate_waves_per_simd, Some(8));
    }

    #[test]
    fn wavefront_override_cannot_change_a_wave32_incumbent_contract() {
        let assessment = ResourceContract::new(ArchProfile::Gfx1151)
            .require_wavefront(Wavefront::Wave64)
            .assess(
                &inspection(report_for_wavefront(82, 64)),
                "candidate",
                &report(82),
            );
        assert!(!assessment.accepted);
        assert!(assessment.rejections.iter().any(|rejection| matches!(
            rejection,
            ResourceRejection::IncumbentWavefrontMismatch {
                expected: 64,
                actual: 32
            }
        )));
    }

    #[test]
    fn rejects_occupancy_spills_scratch_and_clause_regressions_together() {
        let mut candidate = report(97);
        candidate.vgpr_spill_count = 1;
        candidate.private_segment_fixed_size = 16;
        candidate.instructions = InstructionStats {
            memory_clause_instructions: 33,
            ..InstructionStats::default()
        };
        let assessment = ResourceContract::new(ArchProfile::Gfx1151).assess(
            &inspection(candidate),
            "candidate",
            &report(96),
        );
        assert!(!assessment.accepted);
        assert!(
            assessment.rejections.iter().any(|rejection| matches!(
                rejection,
                ResourceRejection::OccupancyRegression { .. }
            ))
        );
        assert!(
            assessment
                .rejections
                .iter()
                .any(|rejection| matches!(rejection, ResourceRejection::RegisterSpill { .. }))
        );
        assert!(
            assessment
                .rejections
                .iter()
                .any(|rejection| matches!(rejection, ResourceRejection::ScratchMemory { .. }))
        );
        assert!(assessment.rejections.iter().any(|rejection| matches!(
            rejection,
            ResourceRejection::StaticMemoryClauseLimit { .. }
        )));
    }

    #[test]
    fn rejects_neighbouring_architecture_even_with_compatible_resources() {
        let mut candidate = inspection(report(82));
        candidate.bundle_target = "hipv4-amdgcn-amd-amdhsa--gfx1100".to_owned();
        candidate.identity = Some(CodeObjectIdentity {
            architecture: "gfx1100".to_owned(),
            elf_machine_id: Some(0x41),
            isa: Some(IsaVersion::new(11, 0, 0)),
        });
        let assessment = ResourceContract::new(ArchProfile::Gfx1151).assess(
            &candidate,
            "candidate",
            &report(82),
        );
        assert!(!assessment.accepted);
        assert!(assessment.rejections.len() >= 3);
    }
}
