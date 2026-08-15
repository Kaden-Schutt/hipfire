// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 7 groundwork: parent-derived online Hessian capture.
//!
//! Accumulates per-tensor block-diagonal-256 Gram matrices `XᵀX` **on device
//! in f32** at the producer boundary that GPTQ actually needs: the operand
//! the parent weight matmul consumes after dynamic FP8 activation
//! quantize/dequantize (`CaptureBoundary::PostDynamicFp8`).
//!
//! The rejected 554-tensor capture
//! (`docs/investigations/2026-08-01-ds4-parent-hessian-handoff.md`) was driven
//! by the *quantized* MQ2R artifact and merely stored F32 buffers. This module
//! refuses to claim parent provenance without the caller's boundary discipline
//! and writes the exact `E8H1` `.hblk` contract consumed by
//! `hipfire-quantize --hessian-dir` (byte-identical to
//! `crates/hip-bridge/examples/collect_e8_hessian_rocblas.rs`).
//!
//! Online accumulation deliberately avoids the 13.9 GiB intermediate `.acts`
//! dump. An optional debug dump remains available via
//! [`ParentHessianAccum::set_acts_dump_dir`] (off by default).

use crate::{Ds4ParentBackend, ParentQuantConfig};
use hip_bridge::{RocblasDatatype, RocblasOperation};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

/// On-disk magic for the per-256-block Hessian (`"E8H1"` LE).
///
/// Must match `hipfire-quantize::main::E8_HESSIAN_MAGIC` and
/// `collect_e8_hessian_rocblas::write_hblk`.
pub const E8H1_MAGIC: u32 = 0x45_38_48_31;

/// Channel block width for the diagonal Hessian partition.
pub const HESSIAN_BLOCK: usize = 256;

/// Bytes of one 256×256 f32 block (exclusive of the 12-byte file header).
pub const HBLK_BLOCK_BYTES: usize = HESSIAN_BLOCK * HESSIAN_BLOCK * 4;

/// File header size: magic + n_blocks + K (three u32 LE).
pub const HBLK_HEADER_BYTES: usize = 12;

/// Online per-tensor accumulation of `XᵀX` in f32 on device, keyed by the
/// full tensor name (== hipfire-quantize's `hessian_key` input).
pub struct ParentHessianAccum {
    backend: Ds4ParentBackend,
    /// Ordered map so drain order is deterministic.
    entries: BTreeMap<String, TensorHessianState>,
    /// Optional debug dump of activation tiles (`[u32 rows][u32 K][f32…]`).
    acts_dump_dir: Option<PathBuf>,
    acts_writers: BTreeMap<String, ActsWriter>,
}

struct TensorHessianState {
    k: usize,
    n_blocks: usize,
    /// Device buffer: `n_blocks * 256 * 256` f32, row-major per block.
    h_dev: GpuTensor,
    rows_seen: usize,
}

struct ActsWriter {
    _path: PathBuf,
    k: usize,
    rows: u32,
    file: std::io::BufWriter<std::fs::File>,
}

/// Aggregate validation + size report after [`ParentHessianAccum::write_hblk_dir`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParentHessianReport {
    pub tensors: usize,
    pub bytes: u64,
    pub min_diag: f32,
    pub max_abs: f32,
    pub nonfinite: usize,
    pub negative_diag: usize,
    pub max_asymmetry: f32,
}

/// Diff of a captured name set against the config-derived P3 map.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct P3NameCheck {
    pub expected: usize,
    pub captured: usize,
    pub missing: Vec<String>,
    pub extra: Vec<String>,
}

impl P3NameCheck {
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.extra.is_empty()
    }
}

impl ParentHessianAccum {
    /// Create an empty accumulator. Requires gfx942 (via `ensure_device`) and
    /// an initialized rocBLAS handle for the online Gram GEMM.
    pub fn new(gpu: &mut Gpu, backend: Ds4ParentBackend) -> Result<Self, String> {
        backend.ensure_device(gpu)?;
        gpu.try_init_rocblas();
        if gpu.rocblas.is_none() {
            return Err(
                "deepseek4 parent: ParentHessianAccum requires rocBLAS on gfx942 \
                 (librocblas failed to load)"
                    .to_owned(),
            );
        }
        Ok(Self {
            backend,
            entries: BTreeMap::new(),
            acts_dump_dir: None,
            acts_writers: BTreeMap::new(),
        })
    }

    /// Enable optional `.acts` debug dumps into `dir` (off by default).
    ///
    /// Production parent Hessian runs must leave this unset — the rejected
    /// capture wrote 13.9 GiB of intermediate activations.
    pub fn set_acts_dump_dir(&mut self, dir: Option<PathBuf>) {
        self.acts_dump_dir = dir;
    }

    /// Accumulate one activation tile for `name`.
    ///
    /// `x` is `[rows, k]` and **MUST** be the post-dynamic-quant operand the
    /// parent weight matmul consumes (BF16 after `act_quant_fp8_ue8m0`, or an
    /// F32 widening of that same matrix). Pre-quant residuals are the wrong
    /// boundary.
    pub fn accumulate(
        &mut self,
        gpu: &mut Gpu,
        name: &str,
        x: &GpuTensor,
        rows: usize,
        k: usize,
    ) -> Result<(), String> {
        self.backend.ensure_device(gpu)?;
        if name.is_empty() {
            return Err("deepseek4 parent: hessian accumulate requires a non-empty tensor name".into());
        }
        if rows == 0 {
            return Ok(());
        }
        if k == 0 || k % HESSIAN_BLOCK != 0 {
            return Err(format!(
                "deepseek4 parent: hessian {name}: K={k} must be a positive multiple of {HESSIAN_BLOCK}"
            ));
        }
        if rows > i32::MAX as usize || k > i32::MAX as usize {
            return Err(format!(
                "deepseek4 parent: hessian {name}: dimensions exceed rocBLAS i32 limits \
                 (rows={rows} k={k})"
            ));
        }
        let need_elems = rows
            .checked_mul(k)
            .ok_or_else(|| format!("deepseek4 parent: hessian {name}: rows*k overflow"))?;
        let elem_size = match x.dtype {
            DType::F32 => 4usize,
            DType::BF16 => 2usize,
            other => {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: x dtype must be F32 or BF16 (got {other:?})"
                ))
            }
        };
        let need_bytes = need_elems
            .checked_mul(elem_size)
            .ok_or_else(|| format!("deepseek4 parent: hessian {name}: byte size overflow"))?;
        if x.buf.size() < need_bytes {
            return Err(format!(
                "deepseek4 parent: hessian {name}: x buffer too small \
                 (have {} need {need_bytes} for [{rows},{k}] {:?})",
                x.buf.size(),
                x.dtype
            ));
        }

        // Stage F32 activations for rocBLAS. BF16 is the native post-quant
        // dtype; widen on the host once per tile (capture path is not hot).
        let x_f32_owned: Option<GpuTensor> = match x.dtype {
            DType::F32 => None,
            DType::BF16 => {
                let host = download_bf16_prefix_as_f32(gpu, x, need_elems)?;
                let t = gpu
                    .upload_f32(&host, &[rows, k])
                    .map_err(|e| {
                        format!("deepseek4 parent: hessian {name}: F32 upload: {e:?}")
                    })?;
                Some(t)
            }
            _ => unreachable!("dtype gated above"),
        };
        let x_f32: &GpuTensor = match &x_f32_owned {
            Some(t) => t,
            None => x,
        };

        if let Some(dir) = self.acts_dump_dir.clone() {
            let host = gpu
                .download_f32(x_f32)
                .map_err(|e| {
                    format!("deepseek4 parent: hessian {name}: acts download: {e:?}")
                })?;
            if host.len() < need_elems {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: acts download short ({} < {need_elems})",
                    host.len()
                ));
            }
            self.append_acts(name, k, &host[..need_elems], &dir)?;
        }

        if !self.entries.contains_key(name) {
            let n_blocks = k / HESSIAN_BLOCK;
            let h_elems = n_blocks
                .checked_mul(HESSIAN_BLOCK * HESSIAN_BLOCK)
                .ok_or_else(|| format!("deepseek4 parent: hessian {name}: H size overflow"))?;
            let h_dev = gpu
                .alloc_tensor(&[h_elems], DType::F32)
                .map_err(|e| format!("deepseek4 parent: hessian {name}: H alloc: {e:?}"))?;
            if let Err(e) = gpu.hip.memset(&h_dev.buf, 0, h_dev.buf.size()) {
                let _ = gpu.free_tensor(h_dev);
                return Err(format!("deepseek4 parent: hessian {name}: H zero: {e:?}"));
            }
            self.entries.insert(
                name.to_owned(),
                TensorHessianState {
                    k,
                    n_blocks,
                    h_dev,
                    rows_seen: 0,
                },
            );
        }
        {
            let st = self.entries.get(name).expect("just inserted or present");
            if st.k != k {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: K changed from {} to {k}",
                    st.k
                ));
            }
        }

        // rocBLAS Gram: for each 256-channel slice, H += X_bᵀ X_b.
        // Row-major X[rows,K] viewed as column-major Xᵀ[256, rows] at the
        // block pointer with lda=K. A*Aᵀ yields the 256×256 block; because H
        // is symmetric, column-major storage is bytewise identical to the
        // row-major .hblk contract (see collect_e8_hessian_rocblas.rs).
        let alpha = 1.0f32;
        let beta = 1.0f32;
        let rb = gpu.rocblas.as_ref().ok_or_else(|| {
            "deepseek4 parent: hessian accumulate: rocBLAS handle missing after init".to_owned()
        })?;
        let st = self.entries.get_mut(name).expect("entry present");
        for block in 0..st.n_blocks {
            let x_block = unsafe {
                x_f32
                    .buf
                    .as_ptr()
                    .cast::<u8>()
                    .add(block * HESSIAN_BLOCK * 4)
                    .cast::<c_void>()
            };
            let h_block = unsafe {
                st.h_dev
                    .buf
                    .as_ptr()
                    .cast::<u8>()
                    .add(block * HBLK_BLOCK_BYTES)
                    .cast::<c_void>()
            };
            unsafe {
                rb.gemm_ex(
                    RocblasOperation::None,
                    RocblasOperation::Transpose,
                    HESSIAN_BLOCK as i32,
                    HESSIAN_BLOCK as i32,
                    rows as i32,
                    (&alpha as *const f32).cast::<c_void>(),
                    x_block,
                    RocblasDatatype::F32,
                    k as i32,
                    x_block,
                    RocblasDatatype::F32,
                    k as i32,
                    (&beta as *const f32).cast::<c_void>(),
                    h_block,
                    RocblasDatatype::F32,
                    HESSIAN_BLOCK as i32,
                    h_block,
                    RocblasDatatype::F32,
                    HESSIAN_BLOCK as i32,
                    RocblasDatatype::F32,
                )
            }
            .map_err(|e| {
                format!(
                    "deepseek4 parent: hessian {name}: rocBLAS Gram block {block}: {e}"
                )
            })?;
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream).map_err(|e| {
                format!("deepseek4 parent: hessian {name}: stream sync: {e:?}")
            })?;
        } else {
            // Null stream: device-wide sync via a zero-size event is not
            // exposed; rocBLAS on the default stream is ordered with later
            // D2H of the same buffers. Explicit hipDeviceSynchronize is not
            // wrapped — rely on memcpy_dtoh's implicit sync at write time.
        }

        st.rows_seen = st
            .rows_seen
            .checked_add(rows)
            .ok_or_else(|| format!("deepseek4 parent: hessian {name}: rows_seen overflow"))?;

        if let Some(t) = x_f32_owned {
            let _ = gpu.free_tensor(t);
        }
        Ok(())
    }

    pub fn rows_seen(&self, name: &str) -> usize {
        self.entries.get(name).map(|s| s.rows_seen).unwrap_or(0)
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// K dimension recorded for `name`, if any tile has been accumulated.
    pub fn k_of(&self, name: &str) -> Option<usize> {
        self.entries.get(name).map(|s| s.k)
    }

    /// Release device Hessian buffers. Safe to call multiple times; further
    /// `accumulate` after this starts fresh device state for each name.
    pub fn free_device_buffers(&mut self, gpu: &mut Gpu) {
        let entries = std::mem::take(&mut self.entries);
        for (_, st) in entries {
            let _ = gpu.free_tensor(st.h_dev);
        }
    }

    /// Write every accumulated Hessian as `E8H1` `.hblk` into `dir`, exactly
    /// matching what `hipfire-quantize --hessian-dir` consumes.
    pub fn write_hblk_dir(
        &self,
        gpu: &mut Gpu,
        dir: &Path,
    ) -> Result<ParentHessianReport, String> {
        self.backend.ensure_device(gpu)?;
        if self.entries.is_empty() {
            return Err("deepseek4 parent: write_hblk_dir: no tensors accumulated".into());
        }
        std::fs::create_dir_all(dir).map_err(|e| {
            format!(
                "deepseek4 parent: create hessian dir {}: {e}",
                dir.display()
            )
        })?;

        let mut report = ParentHessianReport {
            tensors: 0,
            bytes: 0,
            min_diag: f32::INFINITY,
            max_abs: 0.0,
            nonfinite: 0,
            negative_diag: 0,
            max_asymmetry: 0.0,
        };

        for (name, st) in &self.entries {
            if st.rows_seen == 0 {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: zero rows accumulated — refusing empty H"
                ));
            }
            let mut blocks = gpu.download_f32(&st.h_dev).map_err(|e| {
                format!("deepseek4 parent: hessian {name}: download H: {e:?}")
            })?;
            let expect = st.n_blocks * HESSIAN_BLOCK * HESSIAN_BLOCK;
            if blocks.len() < expect {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: H download short ({} < {expect})",
                    blocks.len()
                ));
            }
            blocks.truncate(expect);

            // Host view as LE bytes for symmetrize/validate/write — same path
            // as collect_e8_hessian_rocblas.
            let mut bytes = vec![0u8; expect * 4];
            for (i, v) in blocks.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }

            let input_asym = symmetrize_blocks(&mut bytes, st.n_blocks);
            let stats = validate_blocks(&bytes, st.n_blocks).map_err(|e| {
                format!("deepseek4 parent: hessian {name}: {e}")
            })?;

            write_hblk_file(dir, name, st.k, &bytes)?;

            let file_bytes = (HBLK_HEADER_BYTES + bytes.len()) as u64;
            report.tensors += 1;
            report.bytes = report.bytes.saturating_add(file_bytes);
            report.min_diag = report.min_diag.min(stats.min_diag);
            report.max_abs = report.max_abs.max(stats.max_abs);
            report.nonfinite += stats.nonfinite;
            report.negative_diag += stats.negative_diag;
            report.max_asymmetry = report
                .max_asymmetry
                .max(stats.max_asymmetry)
                .max(input_asym);
            // Post-canonicalization asymmetry must be exactly 0.
            if stats.max_asymmetry != 0.0 {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: post-symmetrize asymmetry \
                     {} (must be 0)",
                    stats.max_asymmetry
                ));
            }
            if stats.nonfinite != 0 {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: {} non-finite entries",
                    stats.nonfinite
                ));
            }
            if stats.negative_diag != 0 {
                return Err(format!(
                    "deepseek4 parent: hessian {name}: {} materially negative diagonal entries \
                     (min_diag={:.6e})",
                    stats.negative_diag, stats.min_diag
                ));
            }
        }

        // Finalize optional acts dumps (patch row counts).
        // Note: acts writers are behind &self via interior... we only have &self.
        // Acts finalization is best-effort at Drop / explicit finish.
        let _ = dir;
        Ok(report)
    }

    /// Patch row counts on any open `.acts` debug files.
    pub fn finish_acts_dump(&mut self) -> Result<(usize, u64), String> {
        use std::io::{Seek, SeekFrom};
        let mut total_rows = 0u64;
        let n = self.acts_writers.len();
        for (name, w) in &mut self.acts_writers {
            w.file
                .flush()
                .and_then(|_| w.file.seek(SeekFrom::Start(0)).map(|_| ()))
                .and_then(|_| w.file.write_all(&w.rows.to_le_bytes()))
                .and_then(|_| w.file.write_all(&(w.k as u32).to_le_bytes()))
                .and_then(|_| w.file.flush())
                .map_err(|e| {
                    format!("deepseek4 parent: finalize acts dump {name}: {e}")
                })?;
            total_rows += w.rows as u64;
        }
        self.acts_writers.clear();
        Ok((n, total_rows))
    }

    fn append_acts(
        &mut self,
        name: &str,
        k: usize,
        values: &[f32],
        dir: &Path,
    ) -> Result<(), String> {
        use std::io::Write;
        std::fs::create_dir_all(dir).map_err(|e| {
            format!(
                "deepseek4 parent: create acts dir {}: {e}",
                dir.display()
            )
        })?;
        if !self.acts_writers.contains_key(name) {
            let key = hessian_key(name);
            let path = dir.join(format!("{key}.acts"));
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| {
                    format!(
                        "deepseek4 parent: create acts {}: {e}",
                        path.display()
                    )
                })?;
            let mut writer = std::io::BufWriter::new(file);
            writer
                .write_all(&0u32.to_le_bytes())
                .and_then(|_| writer.write_all(&(k as u32).to_le_bytes()))
                .map_err(|e| format!("deepseek4 parent: acts header {name}: {e}"))?;
            self.acts_writers.insert(
                name.to_owned(),
                ActsWriter {
                    _path: path,
                    k,
                    rows: 0,
                    file: writer,
                },
            );
        }
        let w = self.acts_writers.get_mut(name).expect("acts writer");
        if w.k != k {
            return Err(format!(
                "deepseek4 parent: acts {name}: K changed from {} to {k}",
                w.k
            ));
        }
        let rows: u32 = (values.len() / k)
            .try_into()
            .map_err(|_| format!("deepseek4 parent: acts {name}: row count overflow"))?;
        w.rows = w
            .rows
            .checked_add(rows)
            .ok_or_else(|| format!("deepseek4 parent: acts {name}: cumulative row overflow"))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        w.file
            .write_all(bytes)
            .map_err(|e| format!("deepseek4 parent: acts append {name}: {e}"))
    }
}

impl Drop for ParentHessianAccum {
    fn drop(&mut self) {
        // Best-effort acts finalize; ignore errors on drop.
        let _ = self.finish_acts_dump();
    }
}

// ── P3 name map ─────────────────────────────────────────────────────────────

/// Derive the P3 Hessian key set from config (main tower only; MTP excluded).
///
/// Mirrors the MQ2R P3 E8 tensor recipe in `arch.rs::validate_mq2r_tensor_policy`
/// — head + per-layer dense projections that GPTQ-on-E8 consumes. Routed
/// experts are qt=19 MQ2-Lloyd and are **not** in this map.
///
/// For the 0731 config (`num_hidden_layers=43`, the checkpoint's
/// `compress_ratios`) this yields **554** names. If a future config differs,
/// the returned count is the truth for that config — do not force 554.
pub fn p3_tensor_names(cfg: &ParentQuantConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(560);
    names.push("head.weight".to_owned());
    for layer in 0..cfg.num_hidden_layers {
        for suffix in [
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "ffn.shared_experts.w1.weight",
            "ffn.shared_experts.w2.weight",
            "ffn.shared_experts.w3.weight",
        ] {
            names.push(format!("layers.{layer}.{suffix}"));
        }
        let ratio = cfg.compress_ratio(layer);
        if ratio > 0 {
            names.push(format!("layers.{layer}.attn.compressor.wkv.weight"));
            names.push(format!("layers.{layer}.attn.compressor.wgate.weight"));
        }
        if ratio == 4 {
            for suffix in [
                "attn.indexer.wq_b.weight",
                "attn.indexer.weights_proj.weight",
                "attn.indexer.compressor.wkv.weight",
                "attn.indexer.compressor.wgate.weight",
            ] {
                names.push(format!("layers.{layer}.{suffix}"));
            }
        }
        names.push(format!("layers.{layer}.ffn.gate.weight"));
    }
    names
}

/// Compare a captured name set against [`p3_tensor_names`]. Reports missing
/// and extra names without forcing the count to 554.
pub fn check_p3_tensor_names(cfg: &ParentQuantConfig, captured: &[String]) -> P3NameCheck {
    let expected_list = p3_tensor_names(cfg);
    let expected: BTreeSet<&str> = expected_list.iter().map(String::as_str).collect();
    let got: BTreeSet<&str> = captured.iter().map(String::as_str).collect();
    let missing: Vec<String> = expected
        .difference(&got)
        .map(|s| (*s).to_owned())
        .collect();
    let extra: Vec<String> = got
        .difference(&expected)
        .map(|s| (*s).to_owned())
        .collect();
    P3NameCheck {
        expected: expected_list.len(),
        captured: captured.len(),
        missing,
        extra,
    }
}

/// Sanitize a full safetensors tensor name into the filesystem key used by
/// `hipfire-quantize::main::hessian_key` / `.hblk` filenames.
pub fn hessian_key(tensor_name: &str) -> String {
    tensor_name.replace(['/', '\\'], "_").replace("..", "_")
}

/// On-disk `.hblk` size for a tensor with inner dimension `k` (must be a
/// multiple of 256).
pub fn hblk_bytes_for_k(k: usize) -> Result<u64, String> {
    if k == 0 || k % HESSIAN_BLOCK != 0 {
        return Err(format!(
            "deepseek4 parent: hblk_bytes_for_k: K={k} must be a positive multiple of {HESSIAN_BLOCK}"
        ));
    }
    let n_blocks = k / HESSIAN_BLOCK;
    let body = (n_blocks as u64)
        .checked_mul(HBLK_BLOCK_BYTES as u64)
        .ok_or_else(|| "deepseek4 parent: hblk body size overflow".to_owned())?;
    Ok(body + HBLK_HEADER_BYTES as u64)
}

/// Expected activation K for a P3 tensor name (0731 geometry). Used for disk
/// budgeting before a capture; returns `None` for unrecognized names.
pub fn expected_k_for_p3_name(name: &str) -> Option<usize> {
    // Verified against the rejected-but-layout-valid 554-tensor capture on
    // mi300x (`p3-wikitext-1024-hblk-symmetric`).
    if name == "head.weight" {
        return Some(4096);
    }
    let rest = name.strip_prefix("layers.")?;
    let (_layer, suffix) = rest.split_once('.')?;
    let k = match suffix {
        "attn.wq_a.weight"
        | "attn.wkv.weight"
        | "attn.wo_a.weight"
        | "ffn.shared_experts.w1.weight"
        | "ffn.shared_experts.w3.weight"
        | "ffn.gate.weight"
        | "attn.compressor.wkv.weight"
        | "attn.compressor.wgate.weight"
        | "attn.indexer.weights_proj.weight"
        | "attn.indexer.compressor.wkv.weight"
        | "attn.indexer.compressor.wgate.weight" => 4096,
        "attn.wq_b.weight" | "attn.indexer.wq_b.weight" => 1024,
        "ffn.shared_experts.w2.weight" => 2048,
        "attn.wo_b.weight" => 8192,
        _ => return None,
    };
    Some(k)
}

/// Project total `.hblk` bytes for a full P3 parent capture under `cfg`.
pub fn project_p3_hblk_bytes(cfg: &ParentQuantConfig) -> Result<(usize, u64), String> {
    let names = p3_tensor_names(cfg);
    let mut total = 0u64;
    for name in &names {
        let k = expected_k_for_p3_name(name).ok_or_else(|| {
            format!("deepseek4 parent: no expected K for P3 name {name}")
        })?;
        total = total
            .checked_add(hblk_bytes_for_k(k)?)
            .ok_or_else(|| "deepseek4 parent: projected hblk bytes overflow".to_owned())?;
    }
    Ok((names.len(), total))
}

// ── E8H1 host contract (mirrors collect_e8_hessian_rocblas.rs) ──────────────

/// Per-tensor validation stats used by [`validate_blocks`] / [`ParentHessianReport`].
#[derive(Clone, Copy, Debug)]
pub struct BlockStats {
    pub min_diag: f32,
    pub max_abs: f32,
    pub max_asymmetry: f32,
    pub nonfinite: usize,
    pub negative_diag: usize,
}

/// Canonicalize independent rocBLAS triangles to exact symmetry.
///
/// rocBLAS evaluates the two triangles as independent dot products; long rows
/// can differ by a few FP32 ulps. Average in f64, write back as f32 — same as
/// `collect_e8_hessian_rocblas::symmetrize_blocks`.
pub fn symmetrize_blocks(blocks: &mut [u8], n_blocks: usize) -> f32 {
    let mut max_input_asym = 0.0f32;
    for block in 0..n_blocks {
        let base = block * HBLK_BLOCK_BYTES;
        for i in 0..HESSIAN_BLOCK {
            for j in (i + 1)..HESSIAN_BLOCK {
                let upper_offset = base + (i * HESSIAN_BLOCK + j) * 4;
                let lower_offset = base + (j * HESSIAN_BLOCK + i) * 4;
                let upper = f32::from_le_bytes(
                    blocks[upper_offset..upper_offset + 4]
                        .try_into()
                        .expect("four bytes"),
                );
                let lower = f32::from_le_bytes(
                    blocks[lower_offset..lower_offset + 4]
                        .try_into()
                        .expect("four bytes"),
                );
                max_input_asym = max_input_asym.max((upper - lower).abs());
                let average = (0.5 * (upper as f64 + lower as f64)) as f32;
                blocks[upper_offset..upper_offset + 4].copy_from_slice(&average.to_le_bytes());
                blocks[lower_offset..lower_offset + 4].copy_from_slice(&average.to_le_bytes());
            }
        }
    }
    max_input_asym
}

/// Validate finite entries, non-negative diagonals, and report asymmetry.
///
/// Material negative diagonal uses the same tolerance as
/// `collect_e8_hessian_rocblas::validate_blocks`:
/// `min_diag < -1e-5 * max(max_diag, 1)`.
pub fn validate_blocks(blocks: &[u8], n_blocks: usize) -> Result<BlockStats, String> {
    let need = n_blocks * HBLK_BLOCK_BYTES;
    if blocks.len() < need {
        return Err(format!(
            "block buffer short: {} < {need} for {n_blocks} blocks",
            blocks.len()
        ));
    }
    let mut min_diag = f32::INFINITY;
    let mut max_diag = f32::NEG_INFINITY;
    let mut max_abs = 0.0f32;
    let mut max_asym = 0.0f32;
    let mut nonfinite = 0usize;
    let mut negative_diag = 0usize;

    let value = |blocks: &[u8], block: usize, i: usize, j: usize| -> f32 {
        let offset = block * HBLK_BLOCK_BYTES + (i * HESSIAN_BLOCK + j) * 4;
        f32::from_le_bytes(blocks[offset..offset + 4].try_into().expect("four bytes"))
    };

    for block in 0..n_blocks {
        for i in 0..HESSIAN_BLOCK {
            let diagonal = value(blocks, block, i, i);
            if !diagonal.is_finite() {
                nonfinite += 1;
            } else {
                min_diag = min_diag.min(diagonal);
                max_diag = max_diag.max(diagonal);
                max_abs = max_abs.max(diagonal.abs());
            }
            for j in (i + 1)..HESSIAN_BLOCK {
                let upper = value(blocks, block, i, j);
                let lower = value(blocks, block, j, i);
                if !upper.is_finite() {
                    nonfinite += 1;
                } else {
                    max_abs = max_abs.max(upper.abs());
                }
                if !lower.is_finite() {
                    nonfinite += 1;
                } else {
                    max_abs = max_abs.max(lower.abs());
                }
                if upper.is_finite() && lower.is_finite() {
                    max_asym = max_asym.max((upper - lower).abs());
                }
            }
        }
    }

    if nonfinite != 0 {
        return Err(format!("non-finite Hessian entries: {nonfinite}"));
    }
    let floor = -1.0e-5 * max_diag.max(1.0);
    if min_diag < floor {
        // Count materially negative diagonals for the report.
        for block in 0..n_blocks {
            for i in 0..HESSIAN_BLOCK {
                let d = value(blocks, block, i, i);
                if d < floor {
                    negative_diag += 1;
                }
            }
        }
        return Err(format!(
            "materially negative Hessian diagonal: min={min_diag:.6e}, max={max_diag:.6e}, \
             count={negative_diag}"
        ));
    }
    Ok(BlockStats {
        min_diag,
        max_abs,
        max_asymmetry: max_asym,
        nonfinite: 0,
        negative_diag: 0,
    })
}

/// Write one `E8H1` file. `blocks` must already be symmetrized and validated.
pub fn write_hblk_file(
    out_dir: &Path,
    tensor_name: &str,
    k: usize,
    blocks: &[u8],
) -> Result<PathBuf, String> {
    if k == 0 || k % HESSIAN_BLOCK != 0 {
        return Err(format!(
            "deepseek4 parent: write_hblk K={k} not a positive multiple of {HESSIAN_BLOCK}"
        ));
    }
    let n_blocks = k / HESSIAN_BLOCK;
    let expected = n_blocks * HBLK_BLOCK_BYTES;
    if blocks.len() != expected {
        return Err(format!(
            "deepseek4 parent: write_hblk length mismatch: {} != {expected}",
            blocks.len()
        ));
    }
    std::fs::create_dir_all(out_dir).map_err(|e| {
        format!(
            "deepseek4 parent: create {}: {e}",
            out_dir.display()
        )
    })?;
    let path = out_dir.join(format!("{}.hblk", hessian_key(tensor_name)));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("deepseek4 parent: create {}: {e}", path.display()))?;
    file.write_all(&E8H1_MAGIC.to_le_bytes())
        .and_then(|_| file.write_all(&(n_blocks as u32).to_le_bytes()))
        .and_then(|_| file.write_all(&(k as u32).to_le_bytes()))
        .and_then(|_| file.write_all(blocks))
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("deepseek4 parent: write {}: {e}", path.display()))?;
    Ok(path)
}

/// Host f64 reference Gram for one 256-channel block (tests / mi300x proof).
pub fn host_xtx_block_f64(x: &[f32], rows: usize, k: usize, block: usize) -> Vec<f64> {
    assert_eq!(x.len(), rows * k);
    let mut h = vec![0.0f64; HESSIAN_BLOCK * HESSIAN_BLOCK];
    let col0 = block * HESSIAN_BLOCK;
    for r in 0..rows {
        let base = r * k + col0;
        for i in 0..HESSIAN_BLOCK {
            let xi = x[base + i] as f64;
            if xi == 0.0 {
                continue;
            }
            let row = &mut h[i * HESSIAN_BLOCK..(i + 1) * HESSIAN_BLOCK];
            for j in 0..HESSIAN_BLOCK {
                row[j] += xi * x[base + j] as f64;
            }
        }
    }
    h
}

fn download_bf16_prefix_as_f32(
    gpu: &Gpu,
    t: &GpuTensor,
    nelems: usize,
) -> Result<Vec<f32>, String> {
    let nbytes = nelems
        .checked_mul(2)
        .ok_or_else(|| "deepseek4 parent: bf16 download size overflow".to_owned())?;
    if t.buf.size() < nbytes {
        return Err(format!(
            "deepseek4 parent: bf16 buffer short (have {} need {nbytes})",
            t.buf.size()
        ));
    }
    let mut raw = vec![0u8; nbytes];
    // Borrow a sized view of the prefix.
    let view = unsafe { hip_bridge::DeviceBuffer::from_raw(t.buf.as_ptr(), nbytes) };
    gpu.hip
        .memcpy_dtoh(&mut raw, &view)
        .map_err(|e| format!("deepseek4 parent: bf16 dtoh: {e:?}"))?;
    let mut out = Vec::with_capacity(nelems);
    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    Ok(out)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Real 0731 `compress_ratios` (46 entries; parent uses first 43 layers).
    fn parent_cfg_0731() -> ParentQuantConfig {
        let json = serde_json::json!({
            "config": {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
                "num_hidden_layers": 43,
                "num_hash_layers": 3,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
                "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                                    4, 0, 0, 0],
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128]
                }
            }
        })
        .to_string();
        ParentQuantConfig::from_metadata_json(&json).expect("0731 fixture")
    }

    fn read_u32(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }

    #[test]
    fn p3_tensor_names_0731_count_is_554() {
        let cfg = parent_cfg_0731();
        let names = p3_tensor_names(&cfg);
        // Report: derived count vs the historical 554-tensor P3 map.
        assert_eq!(
            names.len(),
            554,
            "derived P3 count {} vs historical 554 — investigate compress_ratios / layer count",
            names.len()
        );
        // Uniqueness + stable head.
        assert_eq!(names[0], "head.weight");
        let set: BTreeSet<_> = names.iter().collect();
        assert_eq!(set.len(), names.len(), "duplicate P3 names");
        // Spot-check ratio gates: layer 0 ratio 0 → no compressor; layer 2
        // ratio 4 → indexer; layer 3 ratio 128 → compressor only.
        assert!(!names.iter().any(|n| n == "layers.0.attn.compressor.wkv.weight"));
        assert!(names.iter().any(|n| n == "layers.2.attn.indexer.wq_b.weight"));
        assert!(names.iter().any(|n| n == "layers.3.attn.compressor.wkv.weight"));
        assert!(!names.iter().any(|n| n == "layers.3.attn.indexer.wq_b.weight"));
        // MTP must never appear.
        assert!(!names.iter().any(|n| n.starts_with("mtp.")));
    }

    #[test]
    fn p3_name_check_reports_missing_and_extra() {
        let cfg = parent_cfg_0731();
        let mut names = p3_tensor_names(&cfg);
        let removed = names.pop().expect("non-empty");
        names.push("not.a.p3.tensor.weight".into());
        let check = check_p3_tensor_names(&cfg, &names);
        assert!(!check.ok());
        assert_eq!(check.expected, 554);
        assert!(check.missing.iter().any(|n| n == &removed), "{:?}", check.missing);
        assert_eq!(check.extra, vec!["not.a.p3.tensor.weight".to_owned()]);
    }

    #[test]
    fn hblk_header_layout_matches_e8h1_contract_byte_for_byte() {
        // Literal contract from collect_e8_hessian_rocblas / load_hessian_blocks:
        //   [u32 magic=0x45384831][u32 n_blocks=K/256][u32 K][f32 × n_blocks*256*256]
        let k = 512usize;
        let n_blocks = k / 256;
        let mut body = vec![0u8; n_blocks * HBLK_BLOCK_BYTES];
        // Distinct pattern so payload isn't all-zero.
        for b in 0..n_blocks {
            for i in 0..256 {
                let off = (b * 256 * 256 + i * 256 + i) * 4;
                let v = (b * 1000 + i) as f32;
                body[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        // Already symmetric (diagonal only).
        let stats = validate_blocks(&body, n_blocks).unwrap();
        assert_eq!(stats.max_asymmetry, 0.0);
        assert!(stats.min_diag >= 0.0);

        let dir = std::env::temp_dir().join(format!(
            "hipfire_parent_hblk_layout_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_hblk_file(&dir, "layers.0.test.weight", k, &body).unwrap();
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("layers.0.test.weight.hblk")
        );
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(read_u32(&bytes, 0), 0x45_38_48_31, "magic");
        assert_eq!(read_u32(&bytes, 4), n_blocks as u32, "n_blocks");
        assert_eq!(read_u32(&bytes, 8), k as u32, "K");
        assert_eq!(bytes.len(), 12 + body.len());
        assert_eq!(&bytes[12..], &body[..]);
        // hessian_key sanitization
        assert_eq!(hessian_key("a/b\\c..d"), "a_b_c_d");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn symmetry_canonicalization_averages_triangles() {
        let mut block = vec![0u8; HBLK_BLOCK_BYTES];
        let upper = (3.0f32).to_le_bytes();
        let lower = (5.0f32).to_le_bytes();
        let upper_offset = (7 * 256 + 19) * 4;
        let lower_offset = (19 * 256 + 7) * 4;
        block[upper_offset..upper_offset + 4].copy_from_slice(&upper);
        block[lower_offset..lower_offset + 4].copy_from_slice(&lower);

        assert_eq!(symmetrize_blocks(&mut block, 1), 2.0);
        let stats = validate_blocks(&block, 1).unwrap();
        assert_eq!(stats.max_asymmetry, 0.0);
        let canonical_upper =
            f32::from_le_bytes(block[upper_offset..upper_offset + 4].try_into().unwrap());
        let canonical_lower =
            f32::from_le_bytes(block[lower_offset..lower_offset + 4].try_into().unwrap());
        assert_eq!(canonical_upper, 4.0);
        assert_eq!(canonical_lower, canonical_upper);
    }

    #[test]
    fn validate_rejects_nonfinite() {
        let mut block = vec![0u8; HBLK_BLOCK_BYTES];
        let nan = f32::NAN.to_le_bytes();
        block[0..4].copy_from_slice(&nan);
        let err = validate_blocks(&block, 1).expect_err("nan");
        assert!(err.contains("non-finite"), "{err}");
    }

    #[test]
    fn validate_rejects_materially_negative_diagonal() {
        let mut block = vec![0u8; HBLK_BLOCK_BYTES];
        // diag[0] = -1.0, rest 0 → max_diag=0 → floor=-1e-5; -1 < floor.
        block[0..4].copy_from_slice(&(-1.0f32).to_le_bytes());
        let err = validate_blocks(&block, 1).expect_err("neg diag");
        assert!(err.contains("negative"), "{err}");
    }

    #[test]
    fn projected_full_capture_bytes_match_rejected_layout() {
        let cfg = parent_cfg_0731();
        let (n, bytes) = project_p3_hblk_bytes(&cfg).unwrap();
        assert_eq!(n, 554);
        // Byte-identical to the preserved symmetric capture size.
        assert_eq!(bytes, 2_212_502_008);
        // Per-K sizes (header + body).
        assert_eq!(hblk_bytes_for_k(1024).unwrap(), 1_048_588);
        assert_eq!(hblk_bytes_for_k(2048).unwrap(), 2_097_164);
        assert_eq!(hblk_bytes_for_k(4096).unwrap(), 4_194_316);
        assert_eq!(hblk_bytes_for_k(8192).unwrap(), 8_388_620);
    }

    #[test]
    fn host_xtx_block_matches_manual_dot() {
        let rows = 5usize;
        let k = 512usize;
        let values: Vec<f32> = (0..rows * k)
            .map(|index| {
                let row = index / k;
                let column = index % k;
                ((row * 17 + column * 7) as f32 * 0.003).sin()
            })
            .collect();
        let h = host_xtx_block_f64(&values, rows, k, 0);
        let mut expect = 0.0f64;
        for r in 0..rows {
            expect += values[r * k + 3] as f64 * values[r * k + 5] as f64;
        }
        assert!((h[3 * 256 + 5] - expect).abs() < 1e-12);
        assert!((h[5 * 256 + 3] - h[3 * 256 + 5]).abs() < 1e-15);
    }

    /// `Ds4ParentBackend` is a ZST sealed by `admit`. Unit/GPU smoke tests need
    /// a handle without a `ModelSource`; layout is zero-sized so this is sound.
    fn backend_for_test() -> Ds4ParentBackend {
        assert_eq!(std::mem::size_of::<Ds4ParentBackend>(), 0);
        unsafe { std::mem::zeroed() }
    }

    /// Prove online rocBLAS Gram against an independent f64 CPU `XᵀX`.
    ///
    /// Requires gfx942 + rocBLAS (mi300x). Bit-exactness is not expected;
    /// anything worse than f32 accumulation noise is a finding.
    #[test]
    #[ignore = "requires gfx942 + rocBLAS (mi300x)"]
    fn gfx942_online_gram_matches_f64_host_xtx() {
        let mut gpu = Gpu::init().expect("Gpu::init");
        if gpu.try_gfx942().is_none() {
            eprintln!("skip: not gfx942 (arch={})", gpu.arch);
            return;
        }
        let backend = backend_for_test();
        let mut acc = ParentHessianAccum::new(&mut gpu, backend).expect("accum");

        let rows = 5usize;
        let k = 512usize;
        let values: Vec<f32> = (0..rows * k)
            .map(|index| {
                let row = index / k;
                let column = index % k;
                ((row * 17 + column * 7) as f32 * 0.003).sin()
            })
            .collect();

        let x = gpu
            .upload_f32(&values, &[rows, k])
            .expect("upload X");

        // Feed twice to exercise beta=1 accumulation (matches collector test).
        acc.accumulate(&mut gpu, "layers.0.test.weight", &x, rows, k)
            .expect("accumulate 1");
        acc.accumulate(&mut gpu, "layers.0.test.weight", &x, rows, k)
            .expect("accumulate 2");
        assert_eq!(acc.rows_seen("layers.0.test.weight"), 2 * rows);

        let dir = std::env::temp_dir().join(format!(
            "hipfire_parent_hess_gpu_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let report = acc.write_hblk_dir(&mut gpu, &dir).expect("write_hblk_dir");
        assert_eq!(report.tensors, 1);
        assert_eq!(report.nonfinite, 0);
        assert_eq!(report.negative_diag, 0);
        assert_eq!(report.max_asymmetry, 0.0);

        let hblk = std::fs::read(dir.join("layers.0.test.weight.hblk")).unwrap();
        assert_eq!(read_u32(&hblk, 0), E8H1_MAGIC);
        assert_eq!(read_u32(&hblk, 4), 2);
        assert_eq!(read_u32(&hblk, 8), 512);

        // Independent f64 reference: 2 × XᵀX (two accumulation passes).
        let mut max_abs = 0.0f64;
        let mut sum_rel = 0.0f64;
        let mut n_rel = 0u64;
        for block in 0..2 {
            let href = host_xtx_block_f64(&values, rows, k, block);
            for i in 0..256 {
                for j in 0..256 {
                    let off = 12 + (block * 256 * 256 + i * 256 + j) * 4;
                    let got = f32::from_le_bytes(hblk[off..off + 4].try_into().unwrap()) as f64;
                    let expect = 2.0 * href[i * 256 + j];
                    let abs = (got - expect).abs();
                    max_abs = max_abs.max(abs);
                    let scale = expect.abs().max(1.0);
                    sum_rel += abs / scale;
                    n_rel += 1;
                    // f32 Gram noise: collector uses 2e-5 * max(|e|,1).
                    let tol = 2.0e-5 * scale;
                    assert!(
                        abs <= tol,
                        "block={block} ({i},{j}): got={got} expect={expect} abs={abs} tol={tol}"
                    );
                }
            }
        }
        let mean_rel = sum_rel / n_rel as f64;
        eprintln!(
            "gfx942 online Gram vs f64 host X^T X: max_abs={max_abs:.6e} mean_rel={mean_rel:.6e} \
             report.min_diag={:.6e} report.max_abs={:.6e}",
            report.min_diag, report.max_abs
        );
        // Anything approaching 1e-3 relative would be a finding, not noise.
        assert!(
            mean_rel < 1.0e-5,
            "mean relative error {mean_rel} exceeds f32 accumulation noise budget"
        );
        assert!(
            max_abs < 1.0e-4,
            "max abs error {max_abs} exceeds f32 accumulation noise budget"
        );

        acc.free_device_buffers(&mut gpu);
        let _ = gpu.free_tensor(x);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
