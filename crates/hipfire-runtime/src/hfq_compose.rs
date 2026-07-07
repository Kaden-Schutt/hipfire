// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compose/decompose HFQM containers along role/feature sidecar boundaries.
//!
//! An `.hfq` model can be shipped either as a base container plus separate
//! sibling sidecar files (`<base>.mtp.hfq`, `.dflash.hfq`, `.triattn.hfq`,
//! `.calib.hfq`, discovered by `hipfire_model::detect_sidecars`) or as a single
//! bundled container carrying every feature's tensors (canonical name shape
//! `Family-Size.mtp.vl.mq4.hfq`).
//!
//! [`compose_hfq`] merges a base container and its sidecars into one bundle;
//! [`decompose_hfq`] splits a bundle back into its component files. They are a
//! lossless inverse pair: compose records a provenance manifest
//! ([`HFQM_COMPOSE_KEY`]) in the bundle metadata that stores, per component, the
//! original filename, `arch_id`, tensor name list, and verbatim metadata JSON —
//! so decompose reproduces each source file byte-for-byte without any per-arch
//! tensor-name inference. Neither operation transforms tensor payload bytes;
//! this is packaging granularity only, orthogonal to `hipfire optimize` (which
//! re-tiles weights into an arch-optimal layout).

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::hfq::{
    write_hfqm_package_streaming, HfqPackage, HfqStreamEntry, HFQM_ARCH_NON_WEIGHT_PACKAGE,
};

/// Metadata key under which [`compose_hfq`] stores the provenance manifest.
pub const HFQM_COMPOSE_KEY: &str = "hipfire_compose";
/// Format tag stamped into the manifest (versioned for forward compatibility).
pub const HFQM_COMPOSE_FORMAT: &str = "hipfire.hfqm.compose.v1";

/// Known role/feature tokens used to label a sidecar component. Purely
/// cosmetic (the exact reconstruction uses `filename`/`metadata_json`); this
/// only produces a friendly `tag` in the manifest.
const KNOWN_ROLES: &[&str] = &[
    "mtp", "dflash", "triattn", "vl", "calib", "hessian", "jinja",
];

/// One source container recorded in a bundle's provenance manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeComponent {
    /// Friendly role label (`base` for the first input, else the feature token).
    pub tag: String,
    /// Original file name (no directory), used as the decompose output name.
    pub filename: String,
    /// The component's own `arch_id` (weight sidecars match the base; role-only
    /// sidecars may use [`HFQM_ARCH_NON_WEIGHT_PACKAGE`]).
    pub arch_id: u32,
    /// Tensor names this component contributed, in original index order.
    pub tensors: Vec<String>,
    /// The component's original metadata JSON, stored verbatim so decompose can
    /// reproduce the source file's metadata bytes exactly.
    pub metadata_json: String,
}

/// Provenance manifest embedded in a composed bundle's metadata JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeManifest {
    pub format: String,
    pub components: Vec<ComposeComponent>,
}

/// The first known role token in a filename's dot-groups (e.g.
/// `Model.mtp.hfq` -> `mtp`), if any. Shared with the CLI so composed bundle
/// names are derived from the same role table.
pub fn sidecar_tag_from_filename(path: &Path) -> Option<String> {
    let fname = path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())?;
    let stem = fname.strip_suffix(".hfq").unwrap_or(&fname).to_string();
    stem.split('.')
        .find(|seg| KNOWN_ROLES.contains(seg))
        .map(|s| s.to_string())
}

/// Derive a friendly role tag for a sidecar from its filename dot-groups, then
/// its metadata, falling back to `"sidecar"`.
fn derive_tag(path: &Path, metadata_json: &str) -> String {
    if let Some(tag) = sidecar_tag_from_filename(path) {
        return tag;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(metadata_json) {
        for key in ["role", "artifact_kind", "package_schema"] {
            if let Some(s) = v.get(key).and_then(|x| x.as_str()) {
                return s.to_string();
            }
        }
    }
    "sidecar".to_string()
}

fn file_name_string(path: &Path) -> io::Result<String> {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("input path has no file name: {}", path.display()),
            )
        })
}

/// Merge a base container (first input) and its role/feature sidecars into a
/// single bundled `.hfq` written to `out`. The base's `arch_id` becomes the
/// bundle's; every sidecar must share that `arch_id` or use
/// [`HFQM_ARCH_NON_WEIGHT_PACKAGE`]. Tensor names must be unique across all
/// inputs. Returns the written bundle path.
pub fn compose_hfq(inputs: &[PathBuf], out: &Path) -> io::Result<PathBuf> {
    if inputs.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "compose needs a base container plus at least one sidecar (>= 2 inputs)",
        ));
    }

    let pkgs: Vec<HfqPackage> = inputs
        .iter()
        .map(|p| {
            HfqPackage::open(p)
                .map_err(|e| io::Error::new(e.kind(), format!("opening {}: {e}", p.display())))
        })
        .collect::<io::Result<_>>()?;

    let base_arch = pkgs[0].arch_id;
    for (pkg, path) in pkgs.iter().zip(inputs).skip(1) {
        if pkg.arch_id != base_arch && pkg.arch_id != HFQM_ARCH_NON_WEIGHT_PACKAGE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "sidecar {} arch_id {} is incompatible with base arch_id {} (must match or be {} for non-weight packages)",
                    path.display(),
                    pkg.arch_id,
                    base_arch,
                    HFQM_ARCH_NON_WEIGHT_PACKAGE
                ),
            ));
        }
    }

    // Flat (pkg_idx, entry_idx) map preserves per-input order and drives the
    // streaming payload writer; `seen` enforces globally unique tensor names.
    let mut seen: HashSet<&str> = HashSet::new();
    let mut flat: Vec<(usize, usize)> = Vec::new();
    let mut components: Vec<ComposeComponent> = Vec::with_capacity(pkgs.len());
    for (pi, pkg) in pkgs.iter().enumerate() {
        let mut names = Vec::with_capacity(pkg.entries().len());
        for (ei, e) in pkg.entries().iter().enumerate() {
            if !seen.insert(e.name.as_str()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "duplicate tensor name {:?} across inputs; cannot compose (HFQM index is keyed by name)",
                        e.name
                    ),
                ));
            }
            flat.push((pi, ei));
            names.push(e.name.clone());
        }
        let tag = if pi == 0 {
            "base".to_string()
        } else {
            derive_tag(&inputs[pi], &pkg.metadata_json)
        };
        components.push(ComposeComponent {
            tag,
            filename: file_name_string(&inputs[pi])?,
            arch_id: pkg.arch_id,
            tensors: names,
            metadata_json: pkg.metadata_json.clone(),
        });
    }

    // Bundle metadata = base metadata object + the provenance manifest.
    let mut bundle_meta = match serde_json::from_str::<serde_json::Value>(&pkgs[0].metadata_json) {
        Ok(v @ serde_json::Value::Object(_)) => v,
        _ => serde_json::Value::Object(serde_json::Map::new()),
    };
    let manifest = ComposeManifest {
        format: HFQM_COMPOSE_FORMAT.to_string(),
        components,
    };
    bundle_meta[HFQM_COMPOSE_KEY] = serde_json::to_value(&manifest).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("serializing manifest: {e}"),
        )
    })?;
    let bundle_meta = serde_json::to_string(&bundle_meta).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("serializing bundle metadata: {e}"),
        )
    })?;

    let stream_entries: Vec<HfqStreamEntry> = flat
        .iter()
        .map(|&(pi, ei)| {
            let e = &pkgs[pi].entries()[ei];
            HfqStreamEntry {
                name: e.name.clone(),
                quant_type: e.quant_type,
                shape: e.shape.clone(),
                group_size: e.group_size,
                data_len: e.data_size as u64,
            }
        })
        .collect();

    write_hfqm_package_streaming(out, base_arch, &bundle_meta, &stream_entries, |i, w| {
        let (pi, ei) = flat[i];
        let name = pkgs[pi].entries()[ei].name.as_str();
        let data = pkgs[pi]
            .blob_data(name)
            .expect("entry enumerated from this package must have blob data");
        w.write_all(data)
    })?;

    Ok(out.to_path_buf())
}

/// Split a composed bundle back into its component files under `out_dir`,
/// reproducing each source file (base + sidecars) byte-for-byte from the
/// embedded provenance manifest. Errors if the container has no
/// [`HFQM_COMPOSE_KEY`] manifest. Returns the written file paths.
pub fn decompose_hfq(bundle: &Path, out_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let pkg = HfqPackage::open(bundle)
        .map_err(|e| io::Error::new(e.kind(), format!("opening {}: {e}", bundle.display())))?;
    let meta: serde_json::Value = serde_json::from_str(&pkg.metadata_json).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bundle metadata is not valid JSON: {e}"),
        )
    })?;
    let Some(manifest_value) = meta.get(HFQM_COMPOSE_KEY) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} has no {HFQM_COMPOSE_KEY} manifest; decompose only supports containers produced by `hipfire model compose`",
                bundle.display()
            ),
        ));
    };
    let manifest: ComposeManifest =
        serde_json::from_value(manifest_value.clone()).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid {HFQM_COMPOSE_KEY} manifest: {e}"),
            )
        })?;
    if manifest.format != HFQM_COMPOSE_FORMAT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported compose manifest format {:?}", manifest.format),
        ));
    }

    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::with_capacity(manifest.components.len());
    for comp in &manifest.components {
        let out_path = out_dir.join(&comp.filename);
        let mut stream_entries = Vec::with_capacity(comp.tensors.len());
        for name in &comp.tensors {
            let e = pkg.entry(name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "manifest references tensor {name:?} absent from bundle {}",
                        bundle.display()
                    ),
                )
            })?;
            stream_entries.push(HfqStreamEntry {
                name: e.name.clone(),
                quant_type: e.quant_type,
                shape: e.shape.clone(),
                group_size: e.group_size,
                data_len: e.data_size as u64,
            });
        }
        write_hfqm_package_streaming(
            &out_path,
            comp.arch_id,
            &comp.metadata_json,
            &stream_entries,
            |i, w| {
                let data = pkg
                    .blob_data(&comp.tensors[i])
                    .expect("tensor validated present above");
                w.write_all(data)
            },
        )?;
        written.push(out_path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn scratch_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hfq_compose_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn mem_tensor(name: &str, data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: 1,
            shape: vec![1, data.len() as u32],
            group_size: 0,
            data,
        }
    }

    #[test]
    fn compose_then_decompose_round_trips_byte_identical() {
        let dir = scratch_dir();
        let base = dir.join("Model.mq4.hfq");
        let mtp = dir.join("Model.mtp.hfq");
        let bundle = dir.join("Model.mtp.mq4.hfq");

        let base_meta = r#"{"arch_id":5,"role":"base"}"#;
        let mtp_meta = r#"{"arch_id":5,"role":"mtp"}"#;
        write_hfqm_package_mem(
            &base,
            5,
            base_meta,
            &[mem_tensor("model.embed.weight", vec![1, 2, 3, 4])],
        )
        .unwrap();
        write_hfqm_package_mem(
            &mtp,
            5,
            mtp_meta,
            &[mem_tensor("mtp.head.weight", vec![9, 8, 7])],
        )
        .unwrap();

        compose_hfq(&[base.clone(), mtp.clone()], &bundle).unwrap();

        // Bundle holds the union of tensors + a valid manifest.
        let pkg = HfqPackage::open(&bundle).unwrap();
        assert_eq!(pkg.arch_id, 5);
        assert!(pkg.entry("model.embed.weight").is_some());
        assert!(pkg.entry("mtp.head.weight").is_some());
        let meta: serde_json::Value = serde_json::from_str(&pkg.metadata_json).unwrap();
        assert_eq!(meta["role"], "base");
        let manifest: ComposeManifest =
            serde_json::from_value(meta[HFQM_COMPOSE_KEY].clone()).unwrap();
        assert_eq!(manifest.format, HFQM_COMPOSE_FORMAT);
        assert_eq!(manifest.components.len(), 2);
        assert_eq!(manifest.components[0].tag, "base");
        assert_eq!(manifest.components[1].tag, "mtp");

        // Decompose reproduces both source files byte-for-byte.
        let out = dir.join("out");
        let written = decompose_hfq(&bundle, &out).unwrap();
        assert_eq!(written.len(), 2);
        assert_eq!(
            std::fs::read(out.join("Model.mq4.hfq")).unwrap(),
            std::fs::read(&base).unwrap()
        );
        assert_eq!(
            std::fs::read(out.join("Model.mtp.hfq")).unwrap(),
            std::fs::read(&mtp).unwrap()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_rejects_arch_mismatch() {
        let dir = scratch_dir();
        let base = dir.join("base.hfq");
        let side = dir.join("side.hfq");
        write_hfqm_package_mem(&base, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        write_hfqm_package_mem(&side, 7, "{}", &[mem_tensor("b", vec![2])]).unwrap();
        let err = compose_hfq(&[base, side], &dir.join("bundle.hfq")).unwrap_err();
        assert!(err.to_string().contains("incompatible"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_allows_non_weight_arch_zero_sidecar() {
        let dir = scratch_dir();
        let base = dir.join("base.hfq");
        let side = dir.join("side.jinja.hfq");
        write_hfqm_package_mem(&base, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        write_hfqm_package_mem(
            &side,
            HFQM_ARCH_NON_WEIGHT_PACKAGE,
            "{}",
            &[mem_tensor("b", vec![2])],
        )
        .unwrap();
        compose_hfq(&[base, side], &dir.join("bundle.hfq")).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compose_rejects_duplicate_tensor_names() {
        let dir = scratch_dir();
        let base = dir.join("base.hfq");
        let side = dir.join("side.hfq");
        write_hfqm_package_mem(&base, 5, "{}", &[mem_tensor("dup", vec![1])]).unwrap();
        write_hfqm_package_mem(&side, 5, "{}", &[mem_tensor("dup", vec![2])]).unwrap();
        let err = compose_hfq(&[base, side], &dir.join("bundle.hfq")).unwrap_err();
        assert!(err.to_string().contains("duplicate tensor name"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decompose_rejects_uncomposed_container() {
        let dir = scratch_dir();
        let plain = dir.join("plain.hfq");
        write_hfqm_package_mem(&plain, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        let err = decompose_hfq(&plain, &dir.join("out")).unwrap_err();
        assert!(err.to_string().contains("no hipfire_compose manifest"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
