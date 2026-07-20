// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One-time fixture evidence and cheap execution-mode eligibility for the
//! opt-in LFM2.5-350M dense-MQ4 retained route on exact gfx1201.

/// The caller-owned execution context for one LFM decode entry.
///
/// Only `PlainAr` can record or consume a retained tape. Every other public
/// forward route names its ineligible mode explicitly at the call boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeExecutionMode {
    PlainAr {
        pipeline_parallel: usize,
        tensor_parallel: usize,
    },
    Prefill,
    BatchedPrefill,
    Speculative,
    Oracle,
    Graph,
}

use hipfire_runtime::hfq::HfqFile;

pub const EXPECTED_RETAINED_MODEL_MD5: &str = "cb5284b8ad5c6f9e4ca859c0aff0bcd0";
pub const EXPECTED_RETAINED_MODEL_BYTES: u64 = 229_474_032;
const EXPECTED_RETAINED_MODEL_MD5_BYTES: [u8; 16] = [
    0xcb, 0x52, 0x84, 0xb8, 0xad, 0x5c, 0x6f, 0x9e, 0x4c, 0xa8, 0x59, 0xc0, 0xaf, 0xf0, 0xbc, 0xd0,
];

/// Opaque content identity for the one frozen HFQ artifact admitted to the
/// retained route. A caller can supply [`Self::ABSENT`] directly; authenticated
/// provenance is minted only by [`authenticate_retained_artifact`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetainedArtifactProvenance(bool);

impl RetainedArtifactProvenance {
    pub const ABSENT: Self = Self(false);

    const fn from_identity(has_overlay: bool, digest: Option<[u8; 16]>) -> Self {
        Self(!has_overlay && matches!(digest, Some(EXPECTED_RETAINED_MODEL_MD5_BYTES)))
    }

    #[cfg(test)]
    const fn authenticated() -> Self {
        Self(true)
    }

    const fn is_authenticated(self) -> bool {
        self.0
    }
}

/// Authenticate the exact base HFQ through its already-open file identity.
///
/// REAP overlays disqualify the artifact because tensor reads would no longer
/// reflect the base bytes covered by the digest.
pub fn authenticate_retained_artifact(
    hfq: &mut HfqFile,
) -> Result<RetainedArtifactProvenance, String> {
    if hfq.has_overlay() {
        return Ok(RetainedArtifactProvenance::ABSENT);
    }
    let digest = hfq
        .base_md5_if_len(EXPECTED_RETAINED_MODEL_BYTES)
        .map_err(|error| format!("lfm2moe: retained artifact identity: {error}"))?;
    Ok(RetainedArtifactProvenance::from_identity(false, digest))
}

/// Opaque proof that a bundle matched the frozen retained-route fixture at
/// construction time. External callers can supply only [`Self::ABSENT`];
/// verified evidence never leaves its owning bundle.
///
/// ```compile_fail
/// use hipfire_arch_lfm2moe::redline_plan::RetainedFixtureEvidence;
/// let forged = RetainedFixtureEvidence(true);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetainedFixtureEvidence(bool);

impl RetainedFixtureEvidence {
    pub const ABSENT: Self = Self(false);

    pub(crate) const fn from_validation(valid: bool, artifact: RetainedArtifactProvenance) -> Self {
        Self(valid && artifact.is_authenticated())
    }

    #[cfg(test)]
    const fn verified() -> Self {
        Self(true)
    }

    pub(crate) const fn is_verified(self) -> bool {
        self.0
    }
}

pub(crate) fn decode_fusion_eligible(
    fixture_evidence: RetainedFixtureEvidence,
    gpu_arch: &str,
    graph_enabled: bool,
    lowered_enabled: bool,
) -> bool {
    fixture_evidence.is_verified() && gpu_arch == "gfx1201" && !graph_enabled && lowered_enabled
}

pub(crate) fn retained_route_eligible(
    fixture_evidence: RetainedFixtureEvidence,
    gpu_arch: &str,
    opt_in: bool,
    mode: DecodeExecutionMode,
    position: u32,
    n_tokens: usize,
    graph_enabled: bool,
    lowered_enabled: bool,
    fusion_enabled: bool,
) -> bool {
    let DecodeExecutionMode::PlainAr {
        pipeline_parallel,
        tensor_parallel,
    } = mode
    else {
        return false;
    };
    fixture_evidence.is_verified()
        && gpu_arch == "gfx1201"
        && opt_in
        && pipeline_parallel == 1
        && tensor_parallel == 1
        && position as usize == n_tokens
        && !graph_enabled
        && lowered_enabled
        && fusion_enabled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_ar() -> DecodeExecutionMode {
        DecodeExecutionMode::PlainAr {
            pipeline_parallel: 1,
            tensor_parallel: 1,
        }
    }

    fn evidence() -> RetainedFixtureEvidence {
        RetainedFixtureEvidence::verified()
    }

    #[test]
    fn cached_fixture_evidence_requires_structure_and_authenticated_artifact() {
        let authenticated = RetainedArtifactProvenance::authenticated();
        assert!(RetainedFixtureEvidence::from_validation(true, authenticated).is_verified());
        assert!(!RetainedFixtureEvidence::from_validation(
            true,
            RetainedArtifactProvenance::ABSENT,
        )
        .is_verified());
        assert!(!RetainedFixtureEvidence::from_validation(false, authenticated).is_verified());
    }

    #[test]
    fn artifact_identity_rejects_overlays_and_wrong_digest() {
        let exact = Some(EXPECTED_RETAINED_MODEL_MD5_BYTES);
        assert!(RetainedArtifactProvenance::from_identity(false, exact).is_authenticated());
        assert!(!RetainedArtifactProvenance::from_identity(true, exact).is_authenticated());
        assert!(
            !RetainedArtifactProvenance::from_identity(false, Some([0_u8; 16])).is_authenticated()
        );
        assert!(!RetainedArtifactProvenance::from_identity(false, None).is_authenticated());
    }

    fn eligible(mode: DecodeExecutionMode) -> bool {
        retained_route_eligible(evidence(), "gfx1201", true, mode, 7, 7, false, true, true)
    }

    #[test]
    fn exact_fixture_gfx1201_lowered_route_admits_decode_fusion() {
        assert!(decode_fusion_eligible(evidence(), "gfx1201", false, true,));
    }

    #[test]
    fn decode_fusion_fails_closed_outside_exact_route() {
        assert!(!decode_fusion_eligible(
            RetainedFixtureEvidence::ABSENT,
            "gfx1201",
            false,
            true,
        ));
        assert!(!decode_fusion_eligible(evidence(), "gfx1200", false, true,));
        assert!(!decode_fusion_eligible(evidence(), "gfx1201", true, true,));
        assert!(!decode_fusion_eligible(evidence(), "gfx1201", false, false,));
    }

    #[test]
    fn exact_fixture_plain_ar_opt_in_is_eligible() {
        assert!(eligible(plain_ar()));
    }

    #[test]
    fn cached_fixture_evidence_is_required() {
        assert!(!retained_route_eligible(
            RetainedFixtureEvidence::ABSENT,
            "gfx1201",
            true,
            plain_ar(),
            7,
            7,
            false,
            true,
            true,
        ));
    }

    #[test]
    fn opt_in_is_required() {
        assert!(!retained_route_eligible(
            evidence(),
            "gfx1201",
            false,
            plain_ar(),
            7,
            7,
            false,
            true,
            true,
        ));
    }

    #[test]
    fn actual_decode_fusion_is_required() {
        assert!(!retained_route_eligible(
            evidence(),
            "gfx1201",
            true,
            plain_ar(),
            7,
            7,
            false,
            true,
            false,
        ));
    }

    #[test]
    fn exact_gfx1201_is_required() {
        for gpu_arch in ["gfx1200", "gfx12", "gfx1201:sramecc+"] {
            assert!(!retained_route_eligible(
                evidence(),
                gpu_arch,
                true,
                plain_ar(),
                7,
                7,
                false,
                true,
                true,
            ));
        }
    }

    #[test]
    fn single_gpu_topology_is_required() {
        for mode in [
            DecodeExecutionMode::PlainAr {
                pipeline_parallel: 2,
                tensor_parallel: 1,
            },
            DecodeExecutionMode::PlainAr {
                pipeline_parallel: 1,
                tensor_parallel: 2,
            },
        ] {
            assert!(!eligible(mode));
        }
    }

    #[test]
    fn sequential_position_is_required() {
        assert!(!retained_route_eligible(
            evidence(),
            "gfx1201",
            true,
            plain_ar(),
            8,
            7,
            false,
            true,
            true,
        ));
    }

    #[test]
    fn production_route_configuration_is_required() {
        for (graph, lowered) in [(true, true), (false, false)] {
            assert!(!retained_route_eligible(
                evidence(),
                "gfx1201",
                true,
                plain_ar(),
                7,
                7,
                graph,
                lowered,
                true,
            ));
        }
    }

    #[test]
    fn every_non_plain_ar_mode_is_ineligible() {
        for mode in [
            DecodeExecutionMode::Prefill,
            DecodeExecutionMode::BatchedPrefill,
            DecodeExecutionMode::Speculative,
            DecodeExecutionMode::Oracle,
            DecodeExecutionMode::Graph,
        ] {
            assert!(!eligible(mode));
        }
    }
}
