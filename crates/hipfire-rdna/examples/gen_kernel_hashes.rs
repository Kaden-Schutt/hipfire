#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

//! Generate .hash sidecar files for pre-compiled kernel blobs.
//! Reads kernel sources from kernels/src/*.hip and hashes them with
//! the same DefaultHasher(source + arch) algorithm as compiler.rs.
//!
//! Usage: cargo run --release -p hipfire-rdna --example gen_kernel_hashes
//! Run from the repo root after compile-kernels.sh.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

fn hash_source(source: &str, arch: &str) -> String {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    arch.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn main() {
    let src_dir = Path::new("kernels/src");
    assert!(
        src_dir.is_dir(),
        "Run from repo root (kernels/src/ not found)"
    );

    // Read common preambles prepended by dispatch helper paths.
    let turbo_common = std::fs::read_to_string(src_dir.join("turbo_common.h")).unwrap_or_default();
    let givens_common =
        std::fs::read_to_string(src_dir.join("givens_common.h")).unwrap_or_default();

    // Collect all generic kernel sources (skip arch-specific variants like *.gfx1100.hip)
    let mut kernel_sources: Vec<(String, String)> = Vec::new();
    // RDNA2 variant sources: (module_name, source) for precompiled blob generation
    let mut rdna2_variant_sources: Vec<(String, String)> = Vec::new();
    for entry in std::fs::read_dir(src_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map(|x| x == "hip").unwrap_or(false) {
            let stem = path.file_stem().unwrap().to_str().unwrap();
            // RDNA2 variant files: gemv_hfq4g256.gfx1030.v{N}.hip
            // These get module names like "gemv_hfq4g256_rdna2v{N}"
            if stem.starts_with("gemv_hfq4g256.gfx1030.v") {
                let v_num = stem.rsplit('v').next().unwrap_or("1");
                let module_name = format!("gemv_hfq4g256_rdna2v{v_num}");
                let raw_source = std::fs::read_to_string(&path).unwrap();
                rdna2_variant_sources.push((module_name, raw_source));
                continue;
            }
            // gfx906-specific dp4a MMQ kernel: hash for gfx906 only.
            // The file is not a variant override (no `.gfxNNN.` infix); it's a
            // distinct kernel only built and dispatched on gfx906.
            if stem == "gemm_hfq4g256_residual_mmq_gfx906" {
                let raw_source = std::fs::read_to_string(&path).unwrap();
                kernel_sources.push((stem.to_string(), raw_source));
                continue;
            }
            if stem.contains("gfx") {
                continue; // Skip other arch-specific variants
            }
            let raw_source = std::fs::read_to_string(&path).unwrap();
            // Replicate runtime source assembly. Givens kernels go through
            // ensure_givens4_kernel, which prepends both common headers.
            let source = if raw_source.contains("#include \"givens_common.h\"") {
                let stripped = raw_source
                    .replace("#include \"turbo_common.h\"", "")
                    .replace("#include \"givens_common.h\"", "");
                format!("{}\n{}\n{}", turbo_common, givens_common, stripped)
            } else if raw_source.contains("#include \"turbo_common.h\"") {
                let stripped = raw_source.replace("#include \"turbo_common.h\"", "");
                format!("{}\n{}", turbo_common, stripped)
            } else {
                raw_source
            };
            kernel_sources.push((stem.to_string(), source));
        }
    }
    kernel_sources.sort_by(|a, b| a.0.cmp(&b.0));
    rdna2_variant_sources.sort_by(|a, b| a.0.cmp(&b.0));

    let archs = [
        "gfx906", "gfx1010", "gfx1030", "gfx1100", "gfx1151", "gfx1200", "gfx1201",
    ];

    let mut written = 0;
    let mut skipped = 0;
    for arch in &archs {
        let dir = format!("kernels/compiled/{arch}");
        if !Path::new(&dir).is_dir() {
            continue;
        }
        eprintln!("--- {arch} ---");
        for (name, source) in &kernel_sources {
            let blob = format!("{dir}/{name}.hsaco");
            if !Path::new(&blob).exists() {
                continue;
            }

            // Check if compile-kernels.sh used an arch-specific variant for this blob.
            // If so, the blob was compiled from different source than what the runtime
            // hashes (runtime always uses generic). Don't write a hash — this forces
            // the runtime to recompile from the generic source (safe fallback).
            let arch_variant = src_dir.join(format!("{name}.{arch}.hip"));
            if arch_variant.exists() {
                let hash_file_str = format!("{dir}/{name}.hash");
                let hash_file_path = Path::new(&hash_file_str);
                if hash_file_path.exists() {
                    let _ = std::fs::remove_file(hash_file_path);
                    eprintln!("  {name}: REMOVED hash (arch-specific variant exists, blob is from different source)");
                }
                continue;
            }

            let hash = hash_source(source, arch);
            let hash_file = format!("{dir}/{name}.hash");

            if Path::new(&hash_file).exists() {
                let existing = std::fs::read_to_string(&hash_file).unwrap_or_default();
                if existing.trim() == hash {
                    skipped += 1;
                    continue;
                }
            }
            std::fs::write(&hash_file, &hash).unwrap();
            eprintln!("  {name}.hash = {hash}");
            written += 1;
        }

        // RDNA2 variant blobs: only for gfx1030/gfx1031
        if *arch == "gfx1030" || *arch == "gfx1031" {
            for (module_name, source) in &rdna2_variant_sources {
                let blob = format!("{dir}/{module_name}.hsaco");
                if !Path::new(&blob).exists() {
                    continue;
                }
                let hash = hash_source(source, arch);
                let hash_file = format!("{dir}/{module_name}.hash");
                if Path::new(&hash_file).exists() {
                    let existing = std::fs::read_to_string(&hash_file).unwrap_or_default();
                    if existing.trim() == hash {
                        skipped += 1;
                        continue;
                    }
                }
                std::fs::write(&hash_file, &hash).unwrap();
                eprintln!("  {module_name}.hash = {hash}");
                written += 1;
            }
        }
    }
    eprintln!("\nDone: {written} written, {skipped} unchanged.");
}
