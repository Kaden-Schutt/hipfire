//! oq4_repack — repackage a canonical OQ4 `.hfq` into the arch-optimal weight
//! layout ahead of load time.
//!
//! Canonical OQ4 stores each weight as quant_type 34 (`[f16 scale][128 nibbles]`
//! per 256-group, row-contiguous) — the portable, arch-independent form that
//! ships everywhere. At load the qwen3.5 loader repacks each such tensor into the
//! arch *combined* device layout (split nibbles + split scales + interleaved
//! decode records). This tool performs that transform ONCE, writing a derived
//! `.hfq` whose OQ4 tensors are stored pre-packed as quant_type 37; the loader
//! then uploads them verbatim with no per-load repack.
//!
//! Design (matches the "general on disk, arch-optimal in use" split):
//!   * The canonical file is the source of truth and is never modified. The
//!     derived file is a machine/arch artifact.
//!   * The quant_type code IS the layout version. A future change to the combined
//!     layout takes a NEW code, so a stale derived artifact is refused at load
//!     (honest capability gap) rather than read as garbage.
//!   * Loading is UMA-safe with no `mmap` double-copy: the qwen3.5 loader reads
//!     tensor data via `pread` (the index-only mmap is dropped after parsing,
//!     `FADV_DONTNEED`), so the pre-packed bytes stream straight into the single
//!     device allocation — no page-cache shadow copy in the shared pool.
//!
//! The transform itself is `hipfire_arch_qwen35::qwen35::oq4_pack_arch_combined`,
//! the SAME function the loader's qt=34 path calls, so the tool and the loader
//! can never drift.
//!
//! Usage:
//!   oq4_repack <in.hfq> [-o <out.hfq>] [--arch <gfx>]
//!
//! With no `-o`, the output is `<stem>-packed.hfq` beside the input.

use std::path::PathBuf;

use hipfire_arch_qwen35::qwen35::{oq4_pack_arch_combined, OQ4_ARCH_PACKED_QT};
use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};

/// Canonical (general, portable) on-disk OQ4 quant_type.
const CANONICAL_OQ4_QT: u8 = 34;

/// Best-effort live GPU arch probe (e.g. "gfx1103"). Read-only
/// `hipGetDeviceProperties` on device 0 — no GPU lock, no compute context, so it
/// coexists with a running daemon. Returns `None` if HIP can't be loaded or no
/// device is present (then the caller requires an explicit `--arch`).
fn probe_gpu_arch() -> Option<String> {
    let hip = hip_bridge::HipRuntime::load().ok()?;
    let arch = hip.get_arch(0).ok()?;
    if arch.starts_with("gfx") {
        Some(arch)
    } else {
        None
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut arch: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                output = Some(PathBuf::from(args.get(i).expect("-o needs a path")));
            }
            "--arch" => {
                i += 1;
                arch = Some(args.get(i).expect("--arch needs a value").clone());
            }
            "-h" | "--help" => {
                eprintln!("usage: oq4_repack <in.hfq> [-o <out.hfq>] [--arch <gfx>]");
                return;
            }
            other => {
                if input.is_none() {
                    input = Some(PathBuf::from(other));
                } else {
                    eprintln!("unexpected argument: {other}");
                    std::process::exit(2);
                }
            }
        }
        i += 1;
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("usage: oq4_repack <in.hfq> [-o <out.hfq>] [--arch <gfx>]");
        std::process::exit(2);
    });

    // Resolve the target arch: explicit --arch wins; otherwise probe the live
    // GPU (a read-only hipGetDeviceProperties — no GPU lock, no compute context).
    // This is just a device-name query, so it coexists with a running daemon.
    let arch = arch.unwrap_or_else(|| match probe_gpu_arch() {
        Some(a) => {
            eprintln!("detected GPU arch: {a} (override with --arch)");
            a
        }
        None => {
            eprintln!("could not detect a GPU arch; pass --arch <gfx> (e.g. --arch gfx1103)");
            std::process::exit(2);
        }
    });

    // Default output is arch-tagged beside the input: `<stem>.<arch>.hfq`
    // (e.g. qwen3.5-0.8b-oq4+.hfq -> qwen3.5-0.8b-oq4+.gfx1103.hfq). file_stem
    // strips only the trailing `.hfq`, so dotted model versions survive.
    let output = output.unwrap_or_else(|| {
        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());
        input.with_file_name(format!("{stem}.{arch}.hfq"))
    });

    // Index-only open: parse header/index without mmapping (let alone paging in)
    // the tensor payload — repack tooling must inspect arbitrarily large models
    // without a full-file copy. Per-tensor bytes are then read on demand below.
    let hfq = HfqFile::open(&input).unwrap_or_else(|e| {
        eprintln!("open {}: {e}", input.display());
        std::process::exit(1);
    });

    // v1 writes the whole derived package in memory (one Vec per tensor). Bundled
    // multi-module packages (MTP/DFlash sidecars) are not round-tripped by the
    // in-memory writer yet — refuse rather than silently drop them.
    if !hfq.modules().is_empty() {
        eprintln!(
            "refusing: {} bundles {} module(s); module round-trip is not supported yet",
            input.display(),
            hfq.modules().len()
        );
        std::process::exit(1);
    }

    // Informational for now: the combined layout is identical across current
    // RDNA/CDNA arches, so quant_type 37 is portable among them — the arch only
    // tags the output name. The flag/probe is the hook for future per-arch
    // layouts (group size / interleave width), at which point a distinct arch
    // will take a distinct quant_type code (and a header arch-gate at load).
    eprintln!("packing for {arch} (combined layout v1, qt {OQ4_ARCH_PACKED_QT})");

    let infos: Vec<_> = hfq.tensors().to_vec();
    let mut out_tensors: Vec<HfqMemTensor> = Vec::with_capacity(infos.len());
    let mut repacked = 0usize;
    let mut copied = 0usize;
    let mut canon_bytes = 0u64;
    let mut packed_bytes = 0u64;

    for info in &infos {
        let (_, data) = hfq
            .tensor_data_vec(&info.name)
            .unwrap_or_else(|| panic!("tensor data not found: {}", info.name));

        if info.quant_type == CANONICAL_OQ4_QT {
            // OQ4 weight matrices are stored [m, k] = [out, in].
            assert_eq!(
                info.shape.len(),
                2,
                "OQ4 tensor {} expected 2D [m,k], got shape {:?}",
                info.name,
                info.shape
            );
            let m = info.shape[0] as usize;
            let k = info.shape[1] as usize;
            let combined = oq4_pack_arch_combined(&data, m, k);
            canon_bytes += data.len() as u64;
            packed_bytes += combined.len() as u64;
            out_tensors.push(HfqMemTensor {
                name: info.name.clone(),
                quant_type: OQ4_ARCH_PACKED_QT,
                shape: info.shape.clone(),
                group_size: info.group_size,
                data: combined,
            });
            repacked += 1;
        } else {
            out_tensors.push(HfqMemTensor {
                name: info.name.clone(),
                quant_type: info.quant_type,
                shape: info.shape.clone(),
                group_size: info.group_size,
                data,
            });
            copied += 1;
        }
    }

    if repacked == 0 {
        eprintln!(
            "warning: no OQ4 (quant_type {CANONICAL_OQ4_QT}) tensors found in {} — nothing to repack",
            input.display()
        );
    }

    write_hfqm_package_mem(&output, hfq.arch_id, &hfq.metadata_json, &out_tensors).unwrap_or_else(
        |e| {
            eprintln!("write {}: {e}", output.display());
            std::process::exit(1);
        },
    );

    eprintln!(
        "oq4_repack: {} -> {}\n  repacked {repacked} OQ4 tensor(s) (qt {CANONICAL_OQ4_QT} -> {OQ4_ARCH_PACKED_QT}), copied {copied} other(s)\n  OQ4 weight bytes {canon_bytes} -> {packed_bytes} on disk (combined layout incl. interleaved decode region)",
        input.display(),
        output.display()
    );
}
