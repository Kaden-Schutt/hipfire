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

/// Injected `role -> owned config-key list` map. Supplied by a caller that can
/// see the arch registry (`Arch::sidecar_config_keys`); this crate stays
/// arch-agnostic. Empty (the default) reproduces the pre-partition behavior:
/// no config keys move on compose/decompose.
pub type RoleConfigKeys = std::collections::BTreeMap<String, Vec<String>>;

/// Known role/feature tokens used to label a sidecar component. Purely
/// cosmetic (the exact reconstruction uses `filename`/`metadata_json`); this
/// only produces a friendly `tag` in the manifest.
pub const KNOWN_ROLES: &[&str] = &[
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
/// single bundled `.hfq` written to `out`. See [`compose_hfq_with_config_keys`];
/// this passes an empty [`RoleConfigKeys`] (no config-key merge).
pub fn compose_hfq(inputs: &[PathBuf], out: &Path) -> io::Result<PathBuf> {
    compose_hfq_with_config_keys(inputs, out, &RoleConfigKeys::new())
}

/// As [`compose_hfq`], but additionally merges each role sidecar's owned config
/// keys (per `role_keys`, keyed by the component's role tag) UP into the
/// bundle's top-level config — so the composed whole bundle advertises every
/// feature whose tensors it contains (e.g. `vision_config` travels up from the
/// `vl` sidecar). This is the inverse of the decompose-time move; together they
/// keep config claims and tensor presence consistent across a round trip.
///
/// The base's `arch_id` becomes the bundle's; every sidecar must share that
/// `arch_id` or use [`HFQM_ARCH_NON_WEIGHT_PACKAGE`]. Tensor names must be
/// unique across all inputs. Returns the written bundle path.
pub fn compose_hfq_with_config_keys(
    inputs: &[PathBuf],
    out: &Path,
    role_keys: &RoleConfigKeys,
) -> io::Result<PathBuf> {
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
    // Lift each sidecar's owned config keys into the bundle's top-level config
    // so the whole bundle advertises the features it actually contains.
    if let serde_json::Value::Object(bundle_obj) = &mut bundle_meta {
        for comp in components.iter().skip(1) {
            let Some(keys) = role_keys.get(&comp.tag) else {
                continue;
            };
            let Ok(serde_json::Value::Object(comp_obj)) =
                serde_json::from_str::<serde_json::Value>(&comp.metadata_json)
            else {
                continue;
            };
            for k in keys {
                if let Some(v) = comp_obj.get(k) {
                    bundle_obj.insert(k.clone(), v.clone());
                }
            }
        }
    }
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
        written.push(write_component(
            &pkg,
            &out_dir.join(&comp.filename),
            comp.arch_id,
            &comp.metadata_json,
            &comp.tensors,
        )?);
    }
    Ok(written)
}

/// Write one component `.hfq` (`tensor_names` pulled verbatim from `pkg`) with
/// the given `arch_id` and metadata. Shared by manifest-based and heuristic
/// decompose. Streams one tensor at a time out of the source mmap.
fn write_component(
    pkg: &HfqPackage,
    out_path: &Path,
    arch_id: u32,
    metadata_json: &str,
    tensor_names: &[String],
) -> io::Result<PathBuf> {
    let mut stream_entries = Vec::with_capacity(tensor_names.len());
    for name in tensor_names {
        let e = pkg.entry(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor {name:?} absent from bundle"),
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
    write_hfqm_package_streaming(out_path, arch_id, metadata_json, &stream_entries, |i, w| {
        let data = pkg
            .blob_data(&tensor_names[i])
            .expect("tensor validated present above");
        w.write_all(data)
    })?;
    Ok(out_path.to_path_buf())
}

/// True if `tensor_name` looks like it belongs to `role` (best-effort prefix
/// match used only by [`decompose_hfq_infer`]).
fn role_matches(role: &str, tensor_name: &str) -> bool {
    let n = tensor_name.to_ascii_lowercase();
    match role {
        "mtp" => n.contains("mtp"),
        "dflash" => n.contains("dflash") || n.contains("draft"),
        "triattn" => n.contains("triattn"),
        "vl" => [
            "vision",
            "visual",
            "siglip",
            "mm_projector",
            "multi_modal_projector",
        ]
        .iter()
        .any(|p| n.contains(p)),
        "calib" | "hessian" => {
            n.contains("calib") || n.contains("hessian") || n.contains("imatrix")
        }
        _ => false,
    }
}

/// All known role tokens present in a bundle filename's dot-groups, in order
/// (e.g. `Model.mtp.vl.mq4.hfq` -> `["mtp", "vl"]`).
fn role_tags_from_filename(path: &Path) -> Vec<String> {
    let fname = path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let stem = fname.strip_suffix(".hfq").unwrap_or(&fname).to_string();
    stem.split('.')
        .filter(|seg| KNOWN_ROLES.contains(seg))
        .map(|s| s.to_string())
        .collect()
}

/// Roles inferred from the bundle's *tensor names* alone, for legacy bundles
/// whose filename carries no role dot-groups. Restricted to the model-feature
/// roles that leave an unambiguous tensor-name fingerprint; the calibration
/// roles (`calib`/`hessian`/`imatrix`) are mutually indistinguishable by tensor
/// name (see [`role_matches`]) and so are only inferable from the filename.
/// Returns the matching roles in a stable order (matching partition precedence).
fn roles_from_tensor_names(pkg: &HfqPackage) -> Vec<String> {
    const TENSOR_INFERABLE_ROLES: &[&str] = &["mtp", "dflash", "triattn", "vl"];
    TENSOR_INFERABLE_ROLES
        .iter()
        .filter(|role| pkg.entries().iter().any(|e| role_matches(role, &e.name)))
        .map(|s| s.to_string())
        .collect()
}

/// Bundle filename with the given role dot-groups removed (case-insensitive):
/// `Model.mtp.vl.mq4.hfq` + `[mtp, vl]` -> `Model.mq4.hfq`.
fn strip_role_groups(fname: &str, roles: &[String]) -> String {
    let stem = fname.strip_suffix(".hfq").unwrap_or(fname);
    let kept: Vec<&str> = stem
        .split('.')
        .filter(|seg| !roles.iter().any(|r| r.eq_ignore_ascii_case(seg)))
        .collect();
    format!("{}.hfq", kept.join("."))
}

/// Best-effort split of a bundle that has NO [`HFQM_COMPOSE_KEY`] manifest,
/// driven by the role dot-groups in the bundle filename plus tensor-name prefix
/// matching ([`role_matches`]). Each declared role claims its matching tensors
/// (first role wins); the remainder become the base. This is LOSSY — output
/// files are not guaranteed byte-identical to any original sidecars (metadata
/// and per-sidecar `arch_id` are synthesized), unlike manifest-based decompose.
///
/// Legacy bundles whose filename carries no role dot-groups fall back to
/// inferring roles from tensor names alone ([`roles_from_tensor_names`]).
/// Errors only if neither the filename nor the tensor names reveal any role.
pub fn decompose_hfq_infer(bundle: &Path, out_dir: &Path) -> io::Result<Vec<PathBuf>> {
    decompose_hfq_infer_with_config_keys(bundle, out_dir, &RoleConfigKeys::new())
}

/// As [`decompose_hfq_infer`], but moves each split-off role's owned config
/// keys (per `role_keys`) OUT of the base metadata and INTO that role's
/// sidecar, so the reconstructed base never advertises a feature whose tensors
/// were carved away (e.g. a `vision_config` left behind with no vision tensors).
pub fn decompose_hfq_infer_with_config_keys(
    bundle: &Path,
    out_dir: &Path,
    role_keys: &RoleConfigKeys,
) -> io::Result<Vec<PathBuf>> {
    let pkg = HfqPackage::open(bundle)
        .map_err(|e| io::Error::new(e.kind(), format!("opening {}: {e}", bundle.display())))?;
    let mut roles = role_tags_from_filename(bundle);
    if roles.is_empty() {
        // Legacy bundle: no role dot-groups in the filename. Recover the split
        // from tensor-name fingerprints instead.
        roles = roles_from_tensor_names(&pkg);
    }
    if roles.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} has no role features in its filename and no role-tagged tensors (mtp/dflash/triattn/vl); heuristic decompose needs role dot-groups (e.g. .mtp.vl), tensor-name role fingerprints, or a composed bundle with a {HFQM_COMPOSE_KEY} manifest",
                bundle.display()
            ),
        ));
    }

    // Partition tensors: each declared role claims its matching, still-unclaimed
    // tensors (first role wins); everything left is the base.
    let mut claimed = vec![false; pkg.entries().len()];
    let mut role_tensors: Vec<(String, Vec<String>)> = Vec::new();
    for role in &roles {
        let mut names = Vec::new();
        for (i, e) in pkg.entries().iter().enumerate() {
            if !claimed[i] && role_matches(role, &e.name) {
                claimed[i] = true;
                names.push(e.name.clone());
            }
        }
        if !names.is_empty() {
            role_tensors.push((role.clone(), names));
        }
    }
    if role_tensors.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "no tensors in {} matched any declared role (mtp/dflash/triattn/vl/calib); cannot infer a split",
                bundle.display()
            ),
        ));
    }
    let base_names: Vec<String> = pkg
        .entries()
        .iter()
        .enumerate()
        .filter(|(i, _)| !claimed[*i])
        .map(|(_, e)| e.name.clone())
        .collect();

    let bundle_fname = bundle
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "bundle.hfq".to_string());
    let base_fname = strip_role_groups(&bundle_fname, &roles);
    let base_stem = base_fname.strip_suffix(".hfq").unwrap_or(&base_fname);
    // Sidecars are `<family>.<role>.hfq`, where family drops the quant token (the
    // base stem's last dot-group) — matching the compose naming (base
    // `Model.mq4.hfq` + `Model.mtp.hfq` <-> `Model.mtp.mq4.hfq`).
    let family_stem = base_stem
        .rsplit_once('.')
        .map(|(head, _)| head)
        .unwrap_or(base_stem);

    std::fs::create_dir_all(out_dir)?;

    // Move each split-off role's owned config keys out of the base metadata and
    // stash them per role, so the base no longer advertises carved-away features
    // and each sidecar carries its own config.
    let mut base_obj = match serde_json::from_str::<serde_json::Value>(&pkg.metadata_json) {
        Ok(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    let mut moved: std::collections::BTreeMap<String, serde_json::Map<String, serde_json::Value>> =
        std::collections::BTreeMap::new();
    for (role, _) in &role_tensors {
        if let Some(keys) = role_keys.get(role) {
            let dst = moved.entry(role.clone()).or_default();
            for k in keys {
                if let Some(v) = base_obj.remove(k) {
                    dst.insert(k.clone(), v);
                }
            }
        }
    }
    let base_meta_json = serde_json::to_string(&serde_json::Value::Object(base_obj))
        .unwrap_or_else(|_| pkg.metadata_json.clone());

    let mut written = Vec::new();
    if !base_names.is_empty() {
        written.push(write_component(
            &pkg,
            &out_dir.join(&base_fname),
            pkg.arch_id,
            &base_meta_json,
            &base_names,
        )?);
    }
    for (role, names) in &role_tensors {
        let mut side_obj = serde_json::Map::new();
        side_obj.insert("role".to_string(), serde_json::Value::String(role.clone()));
        side_obj.insert("arch_id".to_string(), serde_json::json!(pkg.arch_id));
        side_obj.insert(
            "hipfire_infer".to_string(),
            serde_json::Value::String("heuristic.v1".to_string()),
        );
        if let Some(mv) = moved.get(role) {
            for (k, v) in mv {
                side_obj.insert(k.clone(), v.clone());
            }
        }
        let side_meta = serde_json::Value::Object(side_obj).to_string();
        written.push(write_component(
            &pkg,
            &out_dir.join(format!("{family_stem}.{role}.hfq")),
            pkg.arch_id,
            &side_meta,
            names,
        )?);
    }
    Ok(written)
}

/// Decompose a bundle, preferring the lossless manifest path. See
/// [`decompose_hfq_auto_with_config_keys`]; this passes an empty
/// [`RoleConfigKeys`] (no config-key move on the heuristic path).
pub fn decompose_hfq_auto(bundle: &Path, out_dir: &Path, infer: bool) -> io::Result<Vec<PathBuf>> {
    decompose_hfq_auto_with_config_keys(bundle, out_dir, infer, &RoleConfigKeys::new())
}

/// As [`decompose_hfq_auto`], threading `role_keys` into the heuristic path so a
/// carved base drops the config keys its split-off sidecars now own. The lossless
/// manifest path is unaffected: it reproduces each component's stored metadata
/// verbatim, which is already role-consistent by construction.
pub fn decompose_hfq_auto_with_config_keys(
    bundle: &Path,
    out_dir: &Path,
    infer: bool,
    role_keys: &RoleConfigKeys,
) -> io::Result<Vec<PathBuf>> {
    let has_manifest = HfqPackage::open(bundle)
        .ok()
        .and_then(|pkg| serde_json::from_str::<serde_json::Value>(&pkg.metadata_json).ok())
        .map(|v| v.get(HFQM_COMPOSE_KEY).is_some())
        .unwrap_or(false);
    if has_manifest {
        decompose_hfq(bundle, out_dir)
    } else if infer {
        decompose_hfq_infer_with_config_keys(bundle, out_dir, role_keys)
    } else {
        decompose_hfq(bundle, out_dir) // reuses the clear "no manifest" error
    }
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
    fn infer_splits_manifestless_bundle_by_filename_roles() {
        let dir = scratch_dir();
        // A bundle with NO hipfire_compose manifest, name declaring `.mtp`.
        let bundle = dir.join("Model.mtp.mq4.hfq");
        write_hfqm_package_mem(
            &bundle,
            5,
            r#"{"arch_id":5}"#,
            &[
                mem_tensor("model.embed.weight", vec![1, 2, 3, 4]),
                mem_tensor("model.mtp.head.weight", vec![9, 8, 7]),
            ],
        )
        .unwrap();

        // Without --infer, a manifest-less bundle is a hard error.
        assert!(decompose_hfq(&bundle, &dir.join("no")).is_err());

        // --infer splits on the `.mtp` filename role + tensor-name prefix.
        let out = dir.join("out");
        let written = decompose_hfq_infer(&bundle, &out).unwrap();
        assert_eq!(written.len(), 2);
        let base = HfqPackage::open(&out.join("Model.mq4.hfq")).unwrap();
        assert!(base.entry("model.embed.weight").is_some());
        assert!(base.entry("model.mtp.head.weight").is_none());
        let mtp = HfqPackage::open(&out.join("Model.mtp.hfq")).unwrap();
        assert!(mtp.entry("model.mtp.head.weight").is_some());
        assert!(mtp.entry("model.embed.weight").is_none());
        assert!(mtp.metadata_json.contains("heuristic.v1"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn infer_errors_when_no_roles_in_filename_or_tensors() {
        let dir = scratch_dir();
        let bundle = dir.join("Model.mq4.hfq"); // no role dot-groups
                                                // Tensor names carry no role fingerprint either, so nothing to split.
        write_hfqm_package_mem(&bundle, 5, "{}", &[mem_tensor("a", vec![1])]).unwrap();
        let err = decompose_hfq_infer(&bundle, &dir.join("out")).unwrap_err();
        assert!(err.to_string().contains("no role-tagged tensors"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn infer_splits_legacy_bundle_by_tensor_names() {
        let dir = scratch_dir();
        // Legacy bundle: plain filename, NO role dot-groups and NO manifest,
        // but a vision tensor betrays a `vl` sidecar hiding inside.
        let bundle = dir.join("Model.mq4.hfq");
        write_hfqm_package_mem(
            &bundle,
            5,
            r#"{"arch_id":5}"#,
            &[
                mem_tensor("model.embed.weight", vec![1, 2, 3, 4]),
                mem_tensor("model.vision.patch_embed.weight", vec![9, 8, 7]),
            ],
        )
        .unwrap();

        // Filename declares no roles, so the split is recovered from tensor names.
        let out = dir.join("out");
        let written = decompose_hfq_infer(&bundle, &out).unwrap();
        assert_eq!(written.len(), 2);
        // Base keeps the original (unstripped) filename; the vl tensor is carved
        // out into `<family>.vl.hfq`.
        let base = HfqPackage::open(&out.join("Model.mq4.hfq")).unwrap();
        assert!(base.entry("model.embed.weight").is_some());
        assert!(base.entry("model.vision.patch_embed.weight").is_none());
        let vl = HfqPackage::open(&out.join("Model.vl.hfq")).unwrap();
        assert!(vl.entry("model.vision.patch_embed.weight").is_some());
        assert!(vl.entry("model.embed.weight").is_none());

        std::fs::remove_dir_all(&dir).ok();
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

    #[test]
    fn role_config_keys_move_out_of_base_and_compose_restores() {
        let dir = scratch_dir();
        // A VL monolith: metadata advertises `vision_config`, and it holds a
        // vision tensor (so infer carves a `vl` sidecar).
        let bundle = dir.join("Model.mq4.hfq");
        write_hfqm_package_mem(
            &bundle,
            5,
            r#"{"arch_id":5,"vision_config":{"depth":2},"text_only_field":true}"#,
            &[
                mem_tensor("model.embed.weight", vec![1, 2, 3, 4]),
                mem_tensor("model.vision.patch_embed.weight", vec![9, 8, 7]),
            ],
        )
        .unwrap();

        let mut keys = RoleConfigKeys::new();
        keys.insert("vl".to_string(), vec!["vision_config".to_string()]);

        let out = dir.join("out");
        let written = decompose_hfq_infer_with_config_keys(&bundle, &out, &keys).unwrap();
        assert_eq!(written.len(), 2);

        // Base no longer advertises vision_config, but keeps its other config.
        let base = HfqPackage::open(&out.join("Model.mq4.hfq")).unwrap();
        assert!(!base.metadata_json.contains("vision_config"));
        assert!(base.metadata_json.contains("text_only_field"));
        // The vl sidecar now owns vision_config.
        let vl = HfqPackage::open(&out.join("Model.vl.hfq")).unwrap();
        assert!(vl.metadata_json.contains("vision_config"));

        // Recompose base + vl → vision_config travels back to the bundle top level.
        let rebundled = dir.join("Rebundled.vl.mq4.hfq");
        compose_hfq_with_config_keys(
            &[out.join("Model.mq4.hfq"), out.join("Model.vl.hfq")],
            &rebundled,
            &keys,
        )
        .unwrap();
        assert!(HfqPackage::open(&rebundled)
            .unwrap()
            .metadata_json
            .contains("vision_config"));

        // But swap the vl sidecar out (compose base + a non-vl sidecar): the
        // bundle must NOT regain vision_config — no vision tensors, no claim.
        let mtp = dir.join("Model.mtp.hfq");
        write_hfqm_package_mem(
            &mtp,
            5,
            r#"{"role":"mtp"}"#,
            &[mem_tensor("model.mtp.w", vec![4])],
        )
        .unwrap();
        let swapped = dir.join("Swapped.mtp.mq4.hfq");
        compose_hfq_with_config_keys(&[out.join("Model.mq4.hfq"), mtp], &swapped, &keys).unwrap();
        assert!(!HfqPackage::open(&swapped)
            .unwrap()
            .metadata_json
            .contains("vision_config"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
