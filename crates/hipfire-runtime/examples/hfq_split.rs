// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Split an `.hfq` model file by tensor-name prefix.
//!
//! Produces two output files from one input:
//!   - BASE: every tensor whose name does NOT start with the prefix
//!   - ADDON: every tensor whose name DOES start with the prefix
//!
//! Both outputs are themselves valid `.hfq` files (full header + JSON
//! metadata + rebuilt tensor index + repacked tensor data). The JSON
//! metadata is copied verbatim into both — readers look up tensors by
//! name in the index, so unused JSON keys are inert. Tensor *bytes* are
//! copied 1:1 from the source; no quantization math runs.
//!
//! Common use: separate the optional MTP layer from a DeepSeek V4 base model.
//!
//!     hfq_split <input.hfq> \
//!         --base /tmp/base.hfq \
//!         --addon /tmp/addon.hfq \
//!         --addon-prefix mtp.0.
//!
//! The DeepSeek V4 runtime then loads `base` directly and picks up `addon` via
//! `HIPFIRE_DEEPSEEK4_MTP_ADDON=<addon path>` (explicit override) or the
//! sibling-file convention if you name the addon `<base>.mtp-addon.hfq`.
//!
//! HFQ on-disk format (read-side reference: `crates/hipfire-runtime/src/hfq.rs`):
//!
//!   Header (32 B):
//!     [0..4]    magic "HFQM"
//!     [4..8]    version (u32 le)
//!     [8..12]   arch_id (u32 le)
//!     [12..16]  n_tensors (u32 le)
//!     [16..24]  metadata_offset (u64 le) — typically 32
//!     [24..32]  data_offset (u64 le)
//!
//!   Metadata JSON @ metadata_offset (variable length, balanced braces)
//!
//!   Tensor index immediately after JSON:
//!     u32                n_tensors  (matches header)
//!     for each tensor:
//!       u16              name_len
//!       [name_len B]     name
//!       u8               quant_type
//!       u8               n_dims
//!       [n_dims × u32]   shape
//!       u32              group_size
//!       u64              data_size  (bytes)
//!
//!   Tensor data @ data_offset, packed in index order, no padding.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

struct TensorEntry {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_offset_src: u64,
    data_size: u64,
}

struct ParsedHfq {
    version: u32,
    arch_id: u32,
    metadata_offset: u64,
    metadata_json_bytes: Vec<u8>,
    tensors: Vec<TensorEntry>,
}

fn print_usage() {
    eprintln!(
        "usage: hfq_split <input.hfq> --base <base-out> --addon <addon-out> --addon-prefix <prefix>\n\
         \n\
         Splits an .hfq by tensor-name prefix. Tensors whose names start\n\
         with <prefix> go to <addon-out>; the rest go to <base-out>.\n\
         Tensor bytes are copied 1:1 — no quantization runs.\n\
         \n\
         Example (split DeepSeek V4's optional MTP layer into a sidecar):\n\
             hfq_split deepseek-v4-flash.mq2lloyd \\\n\
                 --base deepseek-v4-flash.mq2lloyd.new \\\n\
                 --addon deepseek-v4-flash-mtp.mq2lloyd \\\n\
                 --addon-prefix mtp.0."
    );
}

fn parse_hfq_header(file: &mut File) -> std::io::Result<ParsedHfq> {
    let mut header = [0u8; 32];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut header)?;
    if &header[0..4] != b"HFQM" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an .hfq file"));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap());

    // Read metadata JSON: from metadata_offset up to data_offset, but the
    // tensor index sits between JSON and data. Find JSON end via
    // balanced-brace scan (string-aware), identical to hfq.rs read path.
    let meta_region_len = (data_offset - metadata_offset) as usize;
    let mut meta_region = vec![0u8; meta_region_len];
    file.seek(SeekFrom::Start(metadata_offset))?;
    file.read_exact(&mut meta_region)?;

    let mut brace_depth: i32 = 0;
    let mut in_string = false;
    let mut escape = false;
    let mut json_end = 0;
    for (i, &b) in meta_region.iter().enumerate() {
        if escape { escape = false; continue; }
        if b == b'\\' && in_string { escape = true; continue; }
        if b == b'"' { in_string = !in_string; continue; }
        if !in_string {
            if b == b'{' { brace_depth += 1; }
            if b == b'}' {
                brace_depth -= 1;
                if brace_depth == 0 { json_end = i + 1; break; }
            }
        }
    }
    let metadata_json_bytes = meta_region[..json_end].to_vec();

    // Tensor index starts at metadata_offset + json_end
    let mut idx_buf = meta_region[json_end..].to_vec();
    let idx_n = u32::from_le_bytes(idx_buf[0..4].try_into().unwrap()) as usize;
    if idx_n != n_tensors {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("tensor count mismatch: header={n_tensors} index={idx_n}"),
        ));
    }
    let mut pos = 4usize;
    let mut tensors = Vec::with_capacity(n_tensors);
    let mut cumulative_offset = data_offset;
    while idx_buf.len() < (pos + 1) {
        // We may need more bytes from the file if data_offset is small but the
        // index spills past data_offset — shouldn't happen given the writer,
        // but defensively read more.
        let mut more = vec![0u8; 4096];
        let n_read = file.read(&mut more)?;
        if n_read == 0 { break; }
        idx_buf.extend_from_slice(&more[..n_read]);
    }
    for _ in 0..n_tensors {
        let name_len = u16::from_le_bytes(idx_buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&idx_buf[pos..pos + name_len]).to_string();
        pos += name_len;
        let quant_type = idx_buf[pos]; pos += 1;
        let n_dims = idx_buf[pos] as usize; pos += 1;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u32::from_le_bytes(idx_buf[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let group_size = u32::from_le_bytes(idx_buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let data_size = u64::from_le_bytes(idx_buf[pos..pos + 8].try_into().unwrap());
        pos += 8;

        tensors.push(TensorEntry {
            name,
            quant_type,
            shape,
            group_size,
            data_offset_src: cumulative_offset,
            data_size,
        });
        cumulative_offset += data_size;
    }

    Ok(ParsedHfq {
        version,
        arch_id,
        metadata_offset,
        metadata_json_bytes,
        tensors,
    })
}

fn write_partition(
    out_path: &Path,
    source: &mut File,
    parsed: &ParsedHfq,
    selected: &[&TensorEntry],
) -> std::io::Result<()> {
    let mut out = OpenOptions::new()
        .write(true).create(true).truncate(true)
        .open(out_path)?;

    // Compute the new tensor-index byte length so we can place data_offset.
    let mut index_len: usize = 4; // u32 n_tensors
    for t in selected {
        index_len += 2 + t.name.len() + 1 + 1 + t.shape.len() * 4 + 4 + 8;
    }
    let metadata_offset: u64 = 32;
    let data_offset: u64 = metadata_offset
        + parsed.metadata_json_bytes.len() as u64
        + index_len as u64;

    // Header
    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(b"HFQM");
    header[4..8].copy_from_slice(&parsed.version.to_le_bytes());
    header[8..12].copy_from_slice(&parsed.arch_id.to_le_bytes());
    header[12..16].copy_from_slice(&(selected.len() as u32).to_le_bytes());
    header[16..24].copy_from_slice(&metadata_offset.to_le_bytes());
    header[24..32].copy_from_slice(&data_offset.to_le_bytes());
    out.write_all(&header)?;

    // Metadata JSON (verbatim)
    out.write_all(&parsed.metadata_json_bytes)?;

    // Tensor index
    let mut idx = Vec::with_capacity(index_len);
    idx.extend_from_slice(&(selected.len() as u32).to_le_bytes());
    for t in selected {
        let name_bytes = t.name.as_bytes();
        idx.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        idx.extend_from_slice(name_bytes);
        idx.push(t.quant_type);
        idx.push(t.shape.len() as u8);
        for &d in &t.shape { idx.extend_from_slice(&d.to_le_bytes()); }
        idx.extend_from_slice(&t.group_size.to_le_bytes());
        idx.extend_from_slice(&t.data_size.to_le_bytes());
    }
    assert_eq!(idx.len(), index_len);
    out.write_all(&idx)?;

    // Tensor data — copy selected ranges from source in index order.
    // 16 MiB buffer balances syscalls vs RSS on UMA boxes.
    let mut buf = vec![0u8; 16 * 1024 * 1024];
    let mut total_bytes: u64 = 0;
    for (i, t) in selected.iter().enumerate() {
        source.seek(SeekFrom::Start(t.data_offset_src))?;
        let mut remaining = t.data_size;
        while remaining > 0 {
            let want = std::cmp::min(remaining as usize, buf.len());
            source.read_exact(&mut buf[..want])?;
            out.write_all(&buf[..want])?;
            remaining -= want as u64;
            total_bytes += want as u64;
        }
        if (i + 1) % 50 == 0 || i + 1 == selected.len() {
            eprintln!(
                "  wrote {}/{} tensors ({:.2} GB so far)",
                i + 1, selected.len(),
                total_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            );
        }
    }
    out.sync_data()?;
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut base: Option<PathBuf> = None;
    let mut addon: Option<PathBuf> = None;
    let mut prefix: Option<String> = None;
    let mut dry_run = false;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--base" => base = args.next().map(PathBuf::from),
            "--addon" => addon = args.next().map(PathBuf::from),
            "--addon-prefix" => prefix = args.next(),
            "--dry-run" => dry_run = true,
            "-h" | "--help" => { print_usage(); return ExitCode::from(0); }
            s if s.starts_with("--") => {
                eprintln!("error: unknown flag {s}");
                print_usage();
                return ExitCode::from(1);
            }
            s => {
                if input.is_some() {
                    eprintln!("error: multiple positional args");
                    return ExitCode::from(1);
                }
                input = Some(PathBuf::from(s));
            }
        }
    }

    let input = match input {
        Some(p) => p,
        None => { print_usage(); return ExitCode::from(1); }
    };
    let base = match base { Some(p) => p, None => { eprintln!("error: --base required"); return ExitCode::from(1); } };
    let addon = match addon { Some(p) => p, None => { eprintln!("error: --addon required"); return ExitCode::from(1); } };
    let prefix = match prefix { Some(p) => p, None => { eprintln!("error: --addon-prefix required"); return ExitCode::from(1); } };

    if base == input || addon == input {
        eprintln!("error: --base/--addon must differ from input path (write to a tmp + rename)");
        return ExitCode::from(1);
    }

    let mut src = match File::open(&input) {
        Ok(f) => f,
        Err(e) => { eprintln!("error: open {}: {e}", input.display()); return ExitCode::from(1); }
    };

    eprintln!("reading header from {}", input.display());
    let parsed = match parse_hfq_header(&mut src) {
        Ok(p) => p,
        Err(e) => { eprintln!("error: parse_hfq_header: {e}"); return ExitCode::from(1); }
    };
    eprintln!(
        "  version={} arch_id={} n_tensors={} metadata_offset={} metadata_json_len={}",
        parsed.version, parsed.arch_id, parsed.tensors.len(),
        parsed.metadata_offset, parsed.metadata_json_bytes.len(),
    );

    let (addon_tensors, base_tensors): (Vec<&TensorEntry>, Vec<&TensorEntry>) =
        parsed.tensors.iter().partition(|t| t.name.starts_with(&prefix));

    let base_bytes: u64 = base_tensors.iter().map(|t| t.data_size).sum();
    let addon_bytes: u64 = addon_tensors.iter().map(|t| t.data_size).sum();
    eprintln!(
        "partition: base = {} tensors / {:.2} GB | addon = {} tensors / {:.2} GB (prefix={:?})",
        base_tensors.len(), base_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        addon_tensors.len(), addon_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        prefix,
    );

    if addon_tensors.is_empty() {
        eprintln!("warning: no tensors matched prefix; addon file would be empty");
    }

    if dry_run {
        eprintln!("dry-run: would write");
        eprintln!("  base : {}", base.display());
        eprintln!("  addon: {}", addon.display());
        return ExitCode::from(0);
    }

    eprintln!("writing base → {}", base.display());
    if let Err(e) = write_partition(&base, &mut src, &parsed, &base_tensors) {
        eprintln!("error: write base: {e}");
        return ExitCode::from(1);
    }

    eprintln!("writing addon → {}", addon.display());
    if let Err(e) = write_partition(&addon, &mut src, &parsed, &addon_tensors) {
        eprintln!("error: write addon: {e}");
        return ExitCode::from(1);
    }

    eprintln!("done.");
    ExitCode::from(0)
}
