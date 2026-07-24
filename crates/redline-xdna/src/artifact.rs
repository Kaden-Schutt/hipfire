// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{Result, XdnaError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub const SUPPORTED_MANIFEST_VERSION: u32 = 2;
pub const SUPPORTED_ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FirmwareVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    #[serde(default)]
    pub build: u32,
}

impl Ord for FirmwareVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch, self.build).cmp(&(
            other.major,
            other.minor,
            other.patch,
            other.build,
        ))
    }
}

impl PartialOrd for FirmwareVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for FirmwareVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}",
            self.major, self.minor, self.patch, self.build
        )
    }
}

impl FromStr for FirmwareVersion {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let fields: Vec<_> = value.split('.').collect();
        if !(3..=4).contains(&fields.len()) {
            return Err(format!("firmware version must have 3 or 4 fields: {value}"));
        }
        let parse = |index: usize| -> std::result::Result<u32, String> {
            fields[index]
                .parse()
                .map_err(|_| format!("invalid firmware component in {value}"))
        };
        Ok(Self {
            major: parse(0)?,
            minor: parse(1)?,
            patch: parse(2)?,
            build: if fields.len() == 4 { parse(3)? } else { 0 },
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FirmwareCompatibility {
    pub minimum: FirmwareVersion,
    pub maximum: FirmwareVersion,
}

impl FirmwareCompatibility {
    pub fn contains(&self, version: FirmwareVersion) -> bool {
        self.minimum <= version && version <= self.maximum
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IoLayout {
    pub activation: String,
    pub weight: String,
    pub accumulator: String,
    pub output: String,
    pub q8_block_elements: u32,
    pub q8_block_bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProjectionArithmetic {
    #[serde(rename = "q8_w8a16_f32")]
    Q8W8A16F32,
    #[serde(rename = "q8_w8a16_full_array_diagnostic")]
    Q8W8A16FullArrayDiagnostic,
    #[serde(rename = "q8_w8a16_microtile_diagnostic")]
    Q8W8A16MicrotileDiagnostic,
    #[serde(rename = "q8_decode_bf16_diagnostic")]
    Q8DecodeBf16Diagnostic,
    #[serde(rename = "bf16_bf16_f32_diagnostic")]
    Bf16Bf16F32Diagnostic,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProjectionShape {
    pub k: u32,
    pub n: u32,
    pub max_batch: u32,
    #[serde(default)]
    pub masked_batch_tail: bool,
    #[serde(default)]
    pub masked_output_tail: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BindingLayout {
    pub name: String,
    pub access: BindingAccess,
    pub minimum_bytes: u64,
    pub alignment: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactFile {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub manifest_version: u32,
    pub abi_version: u32,
    pub artifact_id: String,
    pub device: String,
    pub firmware: FirmwareCompatibility,
    pub arithmetic: ProjectionArithmetic,
    pub layout: IoLayout,
    pub shapes: Vec<ProjectionShape>,
    pub bindings: Vec<BindingLayout>,
    pub instruction_count: u32,
    pub pdi: ArtifactFile,
    pub instructions: ArtifactFile,
}

impl ArtifactManifest {
    pub fn supports_shape(&self, k: u32, n: u32, batch: u32) -> bool {
        batch > 0
            && self.shapes.iter().any(|shape| {
                shape.k == k
                    && (shape.n == n || (shape.masked_output_tail && n < shape.n))
                    && (shape.max_batch == batch
                        || (shape.masked_batch_tail && batch < shape.max_batch))
            })
    }

    pub fn validate(&self, path: &Path, device: &str, firmware: FirmwareVersion) -> Result<()> {
        let reject = |message: String| XdnaError::ArtifactManifest {
            path: path.to_path_buf(),
            message,
        };
        if self.manifest_version != SUPPORTED_MANIFEST_VERSION {
            return Err(reject(format!(
                "manifest_version {} != supported {}",
                self.manifest_version, SUPPORTED_MANIFEST_VERSION
            )));
        }
        if self.abi_version != SUPPORTED_ABI_VERSION {
            return Err(XdnaError::IncompatibleArtifact(format!(
                "ABI {} != supported {}",
                self.abi_version, SUPPORTED_ABI_VERSION
            )));
        }
        if self.artifact_id.trim().is_empty() {
            return Err(reject("artifact_id is empty".into()));
        }
        if self.device != device {
            return Err(XdnaError::IncompatibleArtifact(format!(
                "artifact targets {}, device is {device}",
                self.device
            )));
        }
        if !self.firmware.contains(firmware) {
            return Err(XdnaError::IncompatibleArtifact(format!(
                "firmware {firmware} outside supported {}..={}",
                self.firmware.minimum, self.firmware.maximum
            )));
        }
        if self.instruction_count == 0 {
            return Err(reject("instruction_count must be nonzero".into()));
        }
        let layout_matches_arithmetic = match self.arithmetic {
            ProjectionArithmetic::Q8W8A16F32 => {
                self.layout.activation == "bf16"
                    && self.layout.weight == "q8_0"
                    && self.layout.accumulator == "f32"
                    && self.layout.output == "f32"
                    && self.layout.q8_block_elements == 32
                    && self.layout.q8_block_bytes == 34
            }
            ProjectionArithmetic::Q8W8A16FullArrayDiagnostic => {
                self.layout.activation == "bf16"
                    && self.layout.weight == "q8_0"
                    && self.layout.accumulator == "f32"
                    && self.layout.output == "f32"
                    && self.layout.q8_block_elements == 32
                    && self.layout.q8_block_bytes == 34
            }
            ProjectionArithmetic::Q8W8A16MicrotileDiagnostic => {
                self.layout.activation == "bf16_aie_tile_4x8"
                    && self.layout.weight == "q8_0"
                    && self.layout.accumulator == "f32"
                    && self.layout.output == "f32_aie_tile_4x8"
                    && self.layout.q8_block_elements == 32
                    && self.layout.q8_block_bytes == 34
            }
            ProjectionArithmetic::Q8DecodeBf16Diagnostic => {
                self.layout.activation == "none"
                    && self.layout.weight == "q8_0"
                    && self.layout.accumulator == "none"
                    && self.layout.output == "bf16"
                    && self.layout.q8_block_elements == 32
                    && self.layout.q8_block_bytes == 34
            }
            ProjectionArithmetic::Bf16Bf16F32Diagnostic => {
                self.layout.activation == "bf16"
                    && self.layout.weight == "bf16"
                    && self.layout.accumulator == "f32"
                    && self.layout.output == "f32"
                    && self.layout.q8_block_elements == 0
                    && self.layout.q8_block_bytes == 0
            }
        };
        if !layout_matches_arithmetic {
            return Err(XdnaError::IncompatibleArtifact(format!(
                "layout {:?} does not match arithmetic contract {:?}",
                self.layout, self.arithmetic
            )));
        }
        if self.shapes.is_empty()
            || self
                .shapes
                .iter()
                .any(|shape| shape.k == 0 || shape.n == 0 || shape.max_batch == 0)
        {
            return Err(reject("shapes must contain nonzero dimensions".into()));
        }
        if self.bindings.is_empty() || self.bindings.len() > 5 {
            return Err(reject(format!(
                "bindings must contain 1..=5 entries, got {}",
                self.bindings.len()
            )));
        }
        let mut binding_names = std::collections::BTreeSet::new();
        for binding in &self.bindings {
            if binding.name.trim().is_empty() {
                return Err(reject("binding name is empty".into()));
            }
            if !binding_names.insert(binding.name.as_str()) {
                return Err(reject(format!("duplicate binding name {:?}", binding.name)));
            }
            if binding.minimum_bytes == 0 {
                return Err(reject(format!(
                    "binding {:?} minimum_bytes must be nonzero",
                    binding.name
                )));
            }
            if binding.alignment == 0
                || !binding.alignment.is_power_of_two()
                || binding.alignment > 4096
            {
                return Err(reject(format!(
                    "binding {:?} alignment {} must be a power of two in 1..=4096",
                    binding.name, binding.alignment
                )));
            }
        }
        validate_relative_file(path, &self.pdi)?;
        validate_relative_file(path, &self.instructions)?;
        Ok(())
    }
}

fn validate_relative_file(manifest_path: &Path, file: &ArtifactFile) -> Result<()> {
    if file.path.is_absolute()
        || file
            .path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(XdnaError::ArtifactManifest {
            path: manifest_path.to_path_buf(),
            message: format!("artifact path {:?} must stay under the bundle", file.path),
        });
    }
    if file.sha256.len() != 64 || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(XdnaError::ArtifactManifest {
            path: manifest_path.to_path_buf(),
            message: format!("invalid SHA-256 for {:?}", file.path),
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ArtifactBundle {
    pub manifest_path: PathBuf,
    pub manifest: ArtifactManifest,
    pub pdi: Vec<u8>,
    pub instructions: Vec<u8>,
}

impl ArtifactBundle {
    pub fn load(
        manifest_path: impl AsRef<Path>,
        device: &str,
        firmware: FirmwareVersion,
    ) -> Result<Self> {
        Self::load_for(
            manifest_path,
            device,
            firmware,
            ProjectionArithmetic::Q8W8A16F32,
        )
    }

    pub fn load_for(
        manifest_path: impl AsRef<Path>,
        device: &str,
        firmware: FirmwareVersion,
        expected_arithmetic: ProjectionArithmetic,
    ) -> Result<Self> {
        let manifest_path = manifest_path.as_ref().to_path_buf();
        let manifest_bytes = fs::read(&manifest_path).map_err(|source| XdnaError::Io {
            operation: "read XDNA artifact manifest",
            source,
        })?;
        let manifest: ArtifactManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                XdnaError::ArtifactManifest {
                    path: manifest_path.clone(),
                    message: error.to_string(),
                }
            })?;
        manifest.validate(&manifest_path, device, firmware)?;
        if manifest.arithmetic != expected_arithmetic {
            return Err(XdnaError::IncompatibleArtifact(format!(
                "arithmetic contract {:?} != required {:?}",
                manifest.arithmetic, expected_arithmetic
            )));
        }
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let pdi = read_checked(root, &manifest.pdi)?;
        let instructions = read_checked(root, &manifest.instructions)?;
        Ok(Self {
            manifest_path,
            manifest,
            pdi,
            instructions,
        })
    }
}

fn read_checked(root: &Path, artifact: &ArtifactFile) -> Result<Vec<u8>> {
    let path = root.join(&artifact.path);
    let bytes = fs::read(&path).map_err(|source| XdnaError::Io {
        operation: "read XDNA artifact",
        source,
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if !actual.eq_ignore_ascii_case(&artifact.sha256) {
        return Err(XdnaError::ArtifactChecksum {
            path,
            expected: artifact.sha256.clone(),
            actual,
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ArtifactManifest {
        ArtifactManifest {
            manifest_version: SUPPORTED_MANIFEST_VERSION,
            abi_version: SUPPORTED_ABI_VERSION,
            artifact_id: "q8-w8a16-test".into(),
            device: "gfx1151".into(),
            firmware: FirmwareCompatibility {
                minimum: "1.0.0.0".parse().unwrap(),
                maximum: "1.9.9.99".parse().unwrap(),
            },
            arithmetic: ProjectionArithmetic::Q8W8A16F32,
            layout: IoLayout {
                activation: "bf16".into(),
                weight: "q8_0".into(),
                accumulator: "f32".into(),
                output: "f32".into(),
                q8_block_elements: 32,
                q8_block_bytes: 34,
            },
            shapes: vec![ProjectionShape {
                k: 2048,
                n: 2048,
                max_batch: 256,
                masked_batch_tail: true,
                masked_output_tail: true,
            }],
            bindings: vec![
                BindingLayout {
                    name: "activation".into(),
                    access: BindingAccess::Read,
                    minimum_bytes: 2048 * 2,
                    alignment: 64,
                },
                BindingLayout {
                    name: "weight".into(),
                    access: BindingAccess::Read,
                    minimum_bytes: 2048 * 34 / 32,
                    alignment: 64,
                },
                BindingLayout {
                    name: "output".into(),
                    access: BindingAccess::Write,
                    minimum_bytes: 2048 * 4,
                    alignment: 64,
                },
            ],
            instruction_count: 123,
            pdi: ArtifactFile {
                path: "program.pdi".into(),
                sha256: "0".repeat(64),
            },
            instructions: ArtifactFile {
                path: "instructions.bin".into(),
                sha256: "0".repeat(64),
            },
        }
    }

    #[test]
    fn firmware_range_is_inclusive() {
        let range = fixture().firmware;
        assert!(range.contains("1.0.0.0".parse().unwrap()));
        assert!(range.contains("1.5.0.0".parse().unwrap()));
        assert!(range.contains("1.9.9.99".parse().unwrap()));
        assert!(!range.contains("2.0.0.0".parse().unwrap()));
    }

    #[test]
    fn rejects_parent_path() {
        let mut manifest = fixture();
        manifest.pdi.path = "../program.pdi".into();
        let error = manifest
            .validate(
                Path::new("manifest.json"),
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("must stay under"));
    }

    #[test]
    fn rejects_ambiguous_binding_abi() {
        let mut manifest = fixture();
        manifest.bindings[1].name = manifest.bindings[0].name.clone();
        assert!(manifest
            .validate(
                Path::new("manifest.json"),
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
            )
            .unwrap_err()
            .to_string()
            .contains("duplicate binding"));

        let mut manifest = fixture();
        manifest.bindings[0].alignment = 3;
        assert!(manifest
            .validate(
                Path::new("manifest.json"),
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
            )
            .unwrap_err()
            .to_string()
            .contains("power of two"));
    }

    #[test]
    fn arithmetic_contract_rejects_incompatible_layout() {
        let mut manifest = fixture();
        manifest.layout.weight = "i8".into();
        assert!(manifest
            .validate(
                Path::new("manifest.json"),
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
            )
            .unwrap_err()
            .to_string()
            .contains("does not match arithmetic contract"));

        let mut manifest = fixture();
        manifest.arithmetic = ProjectionArithmetic::Bf16Bf16F32Diagnostic;
        assert!(manifest
            .validate(
                Path::new("manifest.json"),
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
            )
            .unwrap_err()
            .to_string()
            .contains("does not match arithmetic contract"));

        let mut manifest = fixture();
        manifest.arithmetic = ProjectionArithmetic::Q8DecodeBf16Diagnostic;
        manifest.layout.activation = "none".into();
        manifest.layout.accumulator = "none".into();
        manifest.layout.output = "bf16".into();
        assert!(manifest
            .validate(
                Path::new("manifest.json"),
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
            )
            .is_ok());

        let mut manifest = fixture();
        manifest.arithmetic = ProjectionArithmetic::Q8W8A16MicrotileDiagnostic;
        manifest.layout.activation = "bf16_aie_tile_4x8".into();
        manifest.layout.output = "f32_aie_tile_4x8".into();
        assert!(manifest
            .validate(
                Path::new("manifest.json"),
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
            )
            .is_ok());
    }

    #[test]
    fn manifest_rejects_unknown_arithmetic_contract() {
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["arithmetic"] = serde_json::Value::String("i8_i8_i32".into());
        assert!(serde_json::from_value::<ArtifactManifest>(value).is_err());
    }

    #[test]
    fn shape_support_requires_explicit_tail_capability() {
        let manifest = fixture();
        assert!(manifest.supports_shape(2048, 2048, 256));
        assert!(manifest.supports_shape(2048, 2048, 1));
        assert!(manifest.supports_shape(2048, 16, 256));
        assert!(!manifest.supports_shape(4096, 2048, 256));
        assert!(!manifest.supports_shape(2048, 4096, 256));
        assert!(!manifest.supports_shape(2048, 2048, 257));

        let mut exact = manifest;
        exact.shapes[0].masked_batch_tail = false;
        exact.shapes[0].masked_output_tail = false;
        assert!(!exact.supports_shape(2048, 2048, 255));
        assert!(!exact.supports_shape(2048, 16, 256));
    }

    #[test]
    fn loads_only_checksum_verified_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        let pdi = b"test-pdi";
        let instructions = b"test-instructions";
        fs::write(directory.path().join("program.pdi"), pdi).unwrap();
        fs::write(directory.path().join("instructions.bin"), instructions).unwrap();

        let mut manifest = fixture();
        manifest.pdi.sha256 = format!("{:x}", Sha256::digest(pdi));
        manifest.instructions.sha256 = format!("{:x}", Sha256::digest(instructions));
        let manifest_path = directory.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let bundle =
            ArtifactBundle::load(&manifest_path, "gfx1151", "1.2.3.4".parse().unwrap()).unwrap();
        assert_eq!(bundle.pdi, pdi);
        assert_eq!(bundle.instructions, instructions);

        manifest.instructions.sha256 = "f".repeat(64);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(matches!(
            ArtifactBundle::load(&manifest_path, "gfx1151", "1.2.3.4".parse().unwrap()),
            Err(XdnaError::ArtifactChecksum { .. })
        ));
    }

    #[test]
    fn production_loader_rejects_diagnostic_arithmetic() {
        let directory = tempfile::tempdir().unwrap();
        let pdi = b"test-pdi";
        let instructions = b"test-instructions";
        fs::write(directory.path().join("program.pdi"), pdi).unwrap();
        fs::write(directory.path().join("instructions.bin"), instructions).unwrap();

        let manifest_path = directory.path().join("manifest.json");
        let diagnostic_layouts = [
            (
                ProjectionArithmetic::Bf16Bf16F32Diagnostic,
                IoLayout {
                    activation: "bf16".into(),
                    weight: "bf16".into(),
                    accumulator: "f32".into(),
                    output: "f32".into(),
                    q8_block_elements: 0,
                    q8_block_bytes: 0,
                },
            ),
            (
                ProjectionArithmetic::Q8DecodeBf16Diagnostic,
                IoLayout {
                    activation: "none".into(),
                    weight: "q8_0".into(),
                    accumulator: "none".into(),
                    output: "bf16".into(),
                    q8_block_elements: 32,
                    q8_block_bytes: 34,
                },
            ),
            (
                ProjectionArithmetic::Q8W8A16FullArrayDiagnostic,
                IoLayout {
                    activation: "bf16".into(),
                    weight: "q8_0".into(),
                    accumulator: "f32".into(),
                    output: "f32".into(),
                    q8_block_elements: 32,
                    q8_block_bytes: 34,
                },
            ),
            (
                ProjectionArithmetic::Q8W8A16MicrotileDiagnostic,
                IoLayout {
                    activation: "bf16_aie_tile_4x8".into(),
                    weight: "q8_0".into(),
                    accumulator: "f32".into(),
                    output: "f32_aie_tile_4x8".into(),
                    q8_block_elements: 32,
                    q8_block_bytes: 34,
                },
            ),
        ];
        for (arithmetic, layout) in diagnostic_layouts {
            let mut manifest = fixture();
            manifest.arithmetic = arithmetic;
            manifest.layout = layout;
            manifest.pdi.sha256 = format!("{:x}", Sha256::digest(pdi));
            manifest.instructions.sha256 = format!("{:x}", Sha256::digest(instructions));
            fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

            assert!(
                ArtifactBundle::load(&manifest_path, "gfx1151", "1.2.3.4".parse().unwrap())
                    .unwrap_err()
                    .to_string()
                    .contains("arithmetic contract")
            );
            assert!(ArtifactBundle::load_for(
                &manifest_path,
                "gfx1151",
                "1.2.3.4".parse().unwrap(),
                arithmetic,
            )
            .is_ok());
        }
    }
}
