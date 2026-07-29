// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Exact architecture identities and resource geometry.
//!
//! A profile is intentionally narrower than a compiler target prefix.  A
//! candidate certified for `gfx1151` must not silently become a generic gfx11
//! candidate.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IsaVersion {
    pub major: u8,
    pub minor: u8,
    pub stepping: u8,
}

impl IsaVersion {
    pub const fn new(major: u8, minor: u8, stepping: u8) -> Self {
        Self {
            major,
            minor,
            stepping,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchProfile {
    Gfx1151,
}

impl ArchProfile {
    /// Resolve only an exact architecture name.  Prefixes and neighbouring
    /// gfx11 parts are deliberately rejected.
    pub fn from_arch(arch: &str) -> Option<Self> {
        match arch {
            "gfx1151" => Some(Self::Gfx1151),
            _ => None,
        }
    }

    pub const fn arch(self) -> &'static str {
        match self {
            Self::Gfx1151 => "gfx1151",
        }
    }

    pub const fn isa(self) -> IsaVersion {
        match self {
            Self::Gfx1151 => IsaVersion::new(11, 5, 1),
        }
    }

    /// `EF_AMDGPU_MACH_AMDGCN_GFX1151` in the low byte of the ELF flags.
    pub const fn elf_machine_id(self) -> u32 {
        match self {
            Self::Gfx1151 => 0x04a,
        }
    }

    pub const fn required_wavefront_size(self) -> u32 {
        match self {
            Self::Gfx1151 => 32,
        }
    }

    /// Campaign guardrail for the number of emitted static `s_clause`
    /// instructions in one kernel.
    pub const fn max_static_memory_clauses(self) -> u32 {
        match self {
            Self::Gfx1151 => 32,
        }
    }

    pub const fn max_waves_per_simd(self) -> u32 {
        match self {
            Self::Gfx1151 => 16,
        }
    }

    /// VGPR-limited resident waves for the exact target.  gfx1151 has the
    /// 1.5x VGPR envelope: wave32 uses 1536 registers in 24-register allocation
    /// quanta.  This captures the important 96 -> 97 VGPR cliff (16 -> 12
    /// resident waves) without rejecting register changes within a plateau.
    pub fn vgpr_limited_waves(self, vgpr_count: u32, wavefront_size: u32) -> u32 {
        let (capacity, granule, maximum_waves) = match (self, wavefront_size) {
            (Self::Gfx1151, 32) => (1536_u32, 24_u32, 16_u32),
            (Self::Gfx1151, 64) => (768_u32, 12_u32, 8_u32),
            _ => return 0,
        };
        if vgpr_count == 0 {
            return maximum_waves;
        }
        let allocated = vgpr_count.div_ceil(granule) * granule;
        (capacity / allocated).min(maximum_waves)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CodeObjectIdentity {
    pub architecture: String,
    pub elf_machine_id: Option<u32>,
    pub isa: Option<IsaVersion>,
}

impl CodeObjectIdentity {
    pub(crate) fn from_readobj(bundle_target: &str, readobj: &str) -> Self {
        let architecture = architecture_from_bundle_target(bundle_target).unwrap_or_default();
        Self {
            isa: isa_from_architecture(&architecture),
            architecture,
            elf_machine_id: elf_machine_id_from_readobj(readobj),
        }
    }
}

fn architecture_from_bundle_target(target: &str) -> Option<String> {
    let start = target.rfind("gfx")?;
    let architecture = target[start..]
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .next()
        .unwrap_or_default();
    (!architecture.is_empty()).then(|| architecture.to_owned())
}

fn isa_from_architecture(arch: &str) -> Option<IsaVersion> {
    let digits = arch.strip_prefix("gfx")?;
    let bytes = digits.as_bytes();
    match bytes {
        [major, minor, stepping]
            if major.is_ascii_digit() && minor.is_ascii_digit() && stepping.is_ascii_digit() =>
        {
            Some(IsaVersion::new(major - b'0', minor - b'0', stepping - b'0'))
        }
        [major_tens, major_ones, minor, stepping]
            if major_tens.is_ascii_digit()
                && major_ones.is_ascii_digit()
                && minor.is_ascii_digit()
                && stepping.is_ascii_digit() =>
        {
            Some(IsaVersion::new(
                (major_tens - b'0') * 10 + (major_ones - b'0'),
                minor - b'0',
                stepping - b'0',
            ))
        }
        _ => None,
    }
}

fn elf_machine_id_from_readobj(readobj: &str) -> Option<u32> {
    let mut in_flags = false;
    for line in readobj.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Flags [") || trimmed.starts_with("Flags:") {
            in_flags = true;
            if let Some(value) = first_hex_value(trimmed) {
                return Some(value & 0xff);
            }
            continue;
        }
        if in_flags && trimmed == "]" {
            break;
        }
        if in_flags && trimmed.contains("EF_AMDGPU_MACH_") {
            if let Some(value) = first_hex_value(trimmed) {
                return Some(value & 0xff);
            }
        }
    }
    None
}

fn first_hex_value(value: &str) -> Option<u32> {
    let start = value.find("0x")? + 2;
    let digits = value[start..]
        .chars()
        .take_while(|character| character.is_ascii_hexdigit())
        .collect::<String>();
    (!digits.is_empty())
        .then(|| u32::from_str_radix(&digits, 16).ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_profile_does_not_bleed_across_gfx11() {
        assert_eq!(
            ArchProfile::from_arch("gfx1151"),
            Some(ArchProfile::Gfx1151)
        );
        for arch in ["gfx1100", "gfx1150", "gfx1152", "gfx11", "gfx1151:xnack-"] {
            assert_eq!(ArchProfile::from_arch(arch), None, "{arch}");
        }
    }

    #[test]
    fn gfx1151_vgpr_occupancy_tracks_the_real_plateau() {
        let profile = ArchProfile::Gfx1151;
        assert_eq!(profile.vgpr_limited_waves(82, 32), 16);
        assert_eq!(profile.vgpr_limited_waves(96, 32), 16);
        assert_eq!(profile.vgpr_limited_waves(97, 32), 12);
        assert_eq!(profile.vgpr_limited_waves(120, 32), 12);
        assert_eq!(profile.vgpr_limited_waves(96, 64), 8);
        assert_eq!(profile.vgpr_limited_waves(97, 64), 7);
    }

    #[test]
    fn reads_exact_identity_from_unbundled_elf_header() {
        let readobj = r#"
ElfHeader {
  Machine: EM_AMDGPU (0xE0)
  Flags [ (0x40004A)
    EF_AMDGPU_MACH_AMDGCN_GFX1151 (0x4A)
  ]
}
"#;
        let identity =
            CodeObjectIdentity::from_readobj("hipv4-amdgcn-amd-amdhsa--gfx1151", readobj);
        assert_eq!(identity.architecture, "gfx1151");
        assert_eq!(identity.elf_machine_id, Some(0x4a));
        assert_eq!(identity.isa, Some(IsaVersion::new(11, 5, 1)));
    }
}
