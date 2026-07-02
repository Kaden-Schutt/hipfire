// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `.hsaco` instruction histogram — a Rust port of
//! `scripts/kernel_atlas.py::isa_category`/`parse_disassembly_stats`, static
//! and NO-GPU (shells out to the ROCm LLVM tools — `clang-offload-bundler` +
//! `llvm-objdump` — which are the offline compiler toolchain, not the HIP
//! runtime; no GPU device is touched).
//!
//! Consumed by [`crate::roofline::Roofline::analyze`] (Task 5) as the static
//! half of the 3-bound-class roofline model.

use crate::profiler;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Instruction histogram for a single `.hsaco` code object, bucketed by the
/// same category grammar as `scripts/kernel_atlas.py::isa_category` (ported
/// to Rust, not re-invented) plus a handful of named opcode counts the
/// roofline model needs directly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IsaHistogram {
    /// `v_bfe_*` (bit-field extract) — the nibble/byte unpack step of scalar
    /// quant dequant (hfq4/hfq3/mq4 group formats).
    pub v_bfe: u32,
    /// `v_cvt_*` (format conversion, e.g. `v_cvt_f32_ubyte0`).
    pub v_cvt: u32,
    /// f32 FMA-family (`v_fma_f32`, `v_fmac_f32*`, `v_dual_fmac_f32`) — the
    /// GEMV accumulate. Deliberately excludes f16/pk_f16 FMA variants.
    pub f32_fma: u32,
    /// `v_dot4_*` (dp4a-style 4-way int8 dot product). Zero on every arch
    /// today — Phase A declares dp4a coverage as all-false
    /// ([`crate::arch_caps::Dp4aDecodeGemvCoverage`]); a nonzero count here
    /// would mean a native dp4a kernel is actually in use.
    pub v_dot4: u32,
    /// `v_wmma_*` (matrix-core instructions).
    pub v_wmma: u32,
    /// Exact `global_load_b128` count — the widest (16B/lane) global load
    /// form, used as the per-wave VMEM-footprint proxy in the roofline
    /// latency model.
    pub global_load_b128: u32,
    /// `s_delay_alu` — RDNA3/4 explicit scalar-pipe delay-slot marker;
    /// density here is a signal of serialized SALU dependency chains
    /// (the dequant offset/index arithmetic).
    pub s_delay_alu: u32,
    /// Any `s_wait*` opcode (`s_wait_alu`, `s_wait_loadcnt`,
    /// `s_wait_kmcnt`, `s_wait_dscnt`, legacy `s_waitcnt`).
    pub s_wait: u32,
    /// vmem-category instruction count / valu-category instruction count
    /// (`kernel_atlas.py::isa_category` buckets: vmem = `global_`/`buffer_`/
    /// `flat_`/`scratch_`; valu = `v_*`). Low ratio (lots of VALU per VMEM
    /// op) is the classic signature of a VALU-issue-bound dequant kernel.
    /// `0.0` when there is no VALU traffic at all (degenerate/empty case).
    pub vmem_valu_ratio: f64,
    /// `true` iff the kernel descriptor's `private_segment_fixed_size` > 0
    /// (register spill to the private/scratch segment) — sourced from the
    /// existing hsaco ELF parser ([`profiler::profile_kernels`]), not
    /// re-derived from the disassembly text.
    pub private_segment_red: bool,
}

impl IsaHistogram {
    /// Build a histogram for the device-arch object inside `path` (a raw
    /// `.hsaco` ELF or a `__CLANG_OFFLOAD_BUNDLE__` fat binary — both forms
    /// are produced by the hipcc JIT cache depending on ROCm version/flags).
    /// NO GPU required: shells out to `clang-offload-bundler`/`llvm-objdump`
    /// (the offline LLVM compiler toolchain) and parses `profile_kernels`
    /// (pure ELF/`.kd` parsing, already used by the live profiler).
    pub fn from_hsaco(path: impl AsRef<Path>, arch: &str) -> Result<Self, String> {
        let path = path.as_ref();
        let data = std::fs::read(path)
            .map_err(|e| format!("IsaHistogram::from_hsaco: failed to read {path:?}: {e}"))?;

        let objdump = find_llvm_tool("llvm-objdump").ok_or_else(|| {
            "IsaHistogram::from_hsaco: llvm-objdump not found (checked PATH and \
             $HIP_PATH/llvm/bin, default /opt/rocm/llvm/bin)"
                .to_string()
        })?;

        // A held TempExtract keeps its backing file alive (and cleans up on
        // drop) for the duration of the objdump call below when we had to
        // unbundle; the None arm just disassembles the raw ELF in place.
        let (disasm_input, _guard) = if is_offload_bundle(&data) {
            let bundler = find_llvm_tool("clang-offload-bundler").ok_or_else(|| {
                "IsaHistogram::from_hsaco: clang-offload-bundler not found (checked PATH \
                 and $HIP_PATH/llvm/bin, default /opt/rocm/llvm/bin)"
                    .to_string()
            })?;
            let targets = list_bundle_targets(&bundler, path)?;
            let chosen = choose_bundle_target(&targets, arch).ok_or_else(|| {
                format!(
                    "IsaHistogram::from_hsaco({path:?}): no bundle target for arch \
                     '{arch}' (available targets: {targets:?})"
                )
            })?;
            let extracted = unbundle(&bundler, path, &chosen)?;
            let guard = TempExtract(extracted.clone());
            (extracted, Some(guard))
        } else {
            (path.to_path_buf(), None)
        };

        let output = Command::new(&objdump)
            .arg("-d")
            .arg("--no-show-raw-insn")
            .arg(&disasm_input)
            .output()
            .map_err(|e| format!("IsaHistogram::from_hsaco: failed to run llvm-objdump: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "IsaHistogram::from_hsaco({path:?}): llvm-objdump failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let mut hist = Self::from_disassembly(&text);

        // Register/scratch metadata comes from the existing hsaco ELF
        // parser (profiler::profile_kernels — it internally handles both
        // raw ELF and offload-bundle forms), not re-derived here.
        let mut map = HashMap::new();
        map.insert("isa_histogram_probe".to_string(), path.to_path_buf());
        let (_cap, profiles) = profiler::profile_kernels(arch, 0, &map);
        hist.private_segment_red = profiles
            .first()
            .map(|p| p.scratch_bytes > 0)
            .unwrap_or(false);

        Ok(hist)
    }

    /// Parse `llvm-objdump -d --no-show-raw-insn` output into category
    /// counts. Instruction lines are always indented (a leading tab in
    /// AMD's LLVM objdump); symbol/section/file header lines are not —
    /// this is a more robust discriminator than the Python port's literal
    /// exclusion list (`"Disassembly"`/`"File"`/`"Format"`) since it can't
    /// be fooled by an opcode that happens to collide with a header word.
    fn from_disassembly(text: &str) -> Self {
        let mut hist = Self::default();
        let mut vmem = 0u32;
        let mut valu = 0u32;

        for line in text.lines() {
            if !line.starts_with(char::is_whitespace) {
                continue; // symbol header / "Disassembly of section" / file-format line
            }
            let opcode = match line.trim_start().split_whitespace().next() {
                Some(w) if w.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') => w,
                _ => continue,
            };

            if opcode.starts_with("v_bfe") {
                hist.v_bfe += 1;
            }
            if opcode.starts_with("v_cvt") {
                hist.v_cvt += 1;
            }
            if is_f32_fma(opcode) {
                hist.f32_fma += 1;
            }
            if opcode.starts_with("v_dot4") {
                hist.v_dot4 += 1;
            }
            if opcode.starts_with("v_wmma") {
                hist.v_wmma += 1;
            }
            if opcode == "global_load_b128" {
                hist.global_load_b128 += 1;
            }
            if opcode == "s_delay_alu" {
                hist.s_delay_alu += 1;
            }
            if opcode.starts_with("s_wait") {
                hist.s_wait += 1;
            }

            // kernel_atlas.py::isa_category vmem/valu buckets (subset needed
            // for the ratio — branch/smem/lds/salu/export/other are unused
            // by the roofline model today).
            if opcode.starts_with("global_")
                || opcode.starts_with("buffer_")
                || opcode.starts_with("flat_")
                || opcode.starts_with("scratch_")
            {
                vmem += 1;
            } else if opcode.starts_with("v_") {
                valu += 1;
            }
        }

        hist.vmem_valu_ratio = if valu > 0 {
            vmem as f64 / valu as f64
        } else {
            0.0
        };
        hist
    }
}

/// f32 FMA-family opcode match: `v_fma_f32`, `v_fmac_f32_e32`/`_e64`, and
/// the RDNA3/4 dual-issue-packed `v_dual_fmac_f32` (the FIRST mnemonic on a
/// dual-issue line — same first-token-only simplification as
/// `kernel_atlas.py::parse_disassembly_stats`). Deliberately excludes
/// `v_fma_f16`/`v_pk_fma_f16` (half-precision) and `v_fma_legacy_f32`.
fn is_f32_fma(opcode: &str) -> bool {
    opcode.starts_with("v_fma_f32")
        || opcode.starts_with("v_fmac_f32")
        || opcode.starts_with("v_dual_fmac_f32")
}

pub(crate) fn is_offload_bundle(data: &[u8]) -> bool {
    data.len() > 24 && &data[0..24] == b"__CLANG_OFFLOAD_BUNDLE__"
}

/// Resolve an LLVM tool binary: PATH first (matches
/// `scripts/kernel_atlas.py::find_tool`'s `shutil.which`), then
/// `$HIP_PATH/llvm/bin/<name>` (default `/opt/rocm/llvm/bin/<name>`) — the
/// standard ROCm install layout, where these tools live even when NOT on
/// `$PATH` (only `/opt/rocm/bin` typically is).
pub(crate) fn find_llvm_tool(name: &str) -> Option<String> {
    if Command::new(name).arg("--version").output().is_ok() {
        return Some(name.to_string());
    }
    let base = std::env::var("HIP_PATH")
        .or_else(|_| std::env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    let fallback = format!("{base}/llvm/bin/{name}");
    if std::path::Path::new(&fallback).is_file() {
        return Some(fallback);
    }
    None
}

/// `clang-offload-bundler --list --type=o --input=<path>` — the bundled
/// target-triple strings (e.g. `hipv4-amdgcn-amd-amdhsa--gfx1201`).
pub(crate) fn list_bundle_targets(bundler: &str, path: &Path) -> Result<Vec<String>, String> {
    let output = Command::new(bundler)
        .arg("--list")
        .arg("--type=o")
        .arg(format!("--input={}", path.display()))
        .output()
        .map_err(|e| format!("failed to run clang-offload-bundler --list: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "clang-offload-bundler --list failed for {path:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// Pick the bundle target for `arch` — prefer an exact `--{arch}` suffix
/// match (same rule as `kernel_atlas.py::choose_bundle_target`), falling
/// back to any `amdgcn-amd-amdhsa` device target, then the first entry.
pub(crate) fn choose_bundle_target(targets: &[String], arch: &str) -> Option<String> {
    if targets.is_empty() {
        return None;
    }
    if let Some(t) = targets.iter().find(|t| t.ends_with(&format!("--{arch}"))) {
        return Some(t.clone());
    }
    if let Some(t) = targets.iter().find(|t| t.contains("amdgcn-amd-amdhsa")) {
        return Some(t.clone());
    }
    Some(targets[0].clone())
}

/// Extract `target` from the bundle at `path` into a fresh scratch file
/// under `std::env::temp_dir()`, returning its path.
pub(crate) fn unbundle(
    bundler: &str,
    path: &Path,
    target: &str,
) -> Result<std::path::PathBuf, String> {
    // Unique-enough scratch name: pid + a cheap hash of the input path +
    // target, so concurrent `cargo test` runs / repeated calls don't clash.
    let mut hash: u64 = 1469598103934665603; // FNV offset basis
    for b in path
        .as_os_str()
        .to_string_lossy()
        .bytes()
        .chain(target.bytes())
    {
        hash ^= b as u64;
        hash = hash.wrapping_mul(1099511628211); // FNV prime
    }
    let out = std::env::temp_dir().join(format!(
        "hipfire-isa-histogram-{}-{:x}.o",
        std::process::id(),
        hash
    ));
    let output = Command::new(bundler)
        .arg("--unbundle")
        .arg("--type=o")
        .arg(format!("--targets={target}"))
        .arg(format!("--input={}", path.display()))
        .arg(format!("--output={}", out.display()))
        .output()
        .map_err(|e| format!("failed to run clang-offload-bundler --unbundle: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "clang-offload-bundler --unbundle failed for {path:?} target {target}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(out)
}

/// RAII cleanup for the scratch file written by [`unbundle`].
pub(crate) struct TempExtract(pub(crate) std::path::PathBuf);
impl Drop for TempExtract {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/kernel-fixtures/gfx1201")
            .join(name)
    }

    #[test]
    fn from_hsaco_gate_up_gfx1201() {
        let hist = IsaHistogram::from_hsaco(
            fixture_path("gemv_hfq4g256_moe_gate_up_indexed_batched.hsaco"),
            "gfx1201",
        )
        .expect("IsaHistogram::from_hsaco must succeed against the committed fixture");
        assert!(
            hist.v_bfe > 0,
            "hfq4g256 scalar dequant must show v_bfe_u32 (nibble extract), got {}",
            hist.v_bfe
        );
        assert!(
            hist.f32_fma > 0,
            "GEMV accumulate must show f32 FMA/FMAC, got {}",
            hist.f32_fma
        );
        assert_eq!(
            hist.v_dot4, 0,
            "no native dp4a kernel exists for gfx1201 yet (Phase A declare-only) \
             — this scalar kernel must show zero v_dot4"
        );
        assert!(
            !hist.private_segment_red,
            "no register spill expected on this committed-good fixture"
        );
    }

    #[test]
    fn from_hsaco_down_gfx1201() {
        // Second committed fixture — the down-projection sibling kernel.
        // Cross-checks the histogram isn't overfit to a single file: same
        // scalar-dequant ISA family (v_bfe/f32_fma present, zero v_dot4/
        // v_wmma, no register spill), different kernel/shape.
        let hist = IsaHistogram::from_hsaco(
            fixture_path("gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hsaco"),
            "gfx1201",
        )
        .expect("IsaHistogram::from_hsaco must succeed against the committed fixture");
        assert!(hist.v_bfe > 0, "got {}", hist.v_bfe);
        assert!(hist.f32_fma > 0, "got {}", hist.f32_fma);
        assert_eq!(hist.v_dot4, 0);
        assert_eq!(hist.v_wmma, 0);
        assert!(!hist.private_segment_red);
    }

    #[test]
    fn from_hsaco_missing_file_errors() {
        let result = IsaHistogram::from_hsaco(fixture_path("does_not_exist.hsaco"), "gfx1201");
        assert!(
            result.is_err(),
            "missing fixture must fail loud, not panic/silently-empty"
        );
    }
}
