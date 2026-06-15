// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! convert_kldref — convert a legacy HFKLDR .kldref.bin into HFQM .kldref.hfq
//!
//! Legacy binary layout (all little-endian):
//!   [8]  magic: b"HFKLDR\0\0"
//!   [4]  version: u32 (must be 1)
//!   [4]  n_ctx: u32
//!   [4]  n_vocab: u32
//!   [4]  n_chunk: u32
//!   [2]  top_k: u16
//!   [6]  padding
//!   [n_chunk * n_ctx * 4]  tokens: u32[]
//!   [n_chunk * scored_per_chunk] blocks, each (8 + 8*top_k) bytes:
//!     [top_k * 4]  top_indices: u32[]
//!     [top_k * 4]  top_log_probs: f32[]
//!     [4]          residual_mass: f32
//!     [4]          padding
//!
//! scored_per_chunk is back-calculated from the file size.
//!
//! Usage:
//!   cargo run --release -p hipfire-runtime --example convert_kldref -- \
//!     --input  ~/qwen3.6-27b-bf16.kldref.bin \
//!     --output ~/.hipfire/kldrefs/qwen3.6-27b-bf16.kldref.hfq

fn main() {
    use hipfire_runtime::hfq::{
        write_hfqm_package_from_files, HfqPackageWriteEntry, HFQM_ARCH_NON_WEIGHT_PACKAGE,
    };
    use serde_json::json;
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" => {
                input = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            _ => {
                eprintln!("unknown argument: {}", argv[i]);
                eprintln!(
                    "usage: convert_kldref --input <path.kldref.bin> --output <path.kldref.hfq>"
                );
                std::process::exit(1);
            }
        }
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("--input required");
        std::process::exit(1);
    });
    let output = output.unwrap_or_else(|| {
        eprintln!("--output required");
        std::process::exit(1);
    });

    // ── Read header ──────────────────────────────────────────────────────────
    let file_size = std::fs::metadata(&input)
        .unwrap_or_else(|e| panic!("stat {}: {e}", input.display()))
        .len() as usize;

    let f = File::open(&input).unwrap_or_else(|e| panic!("open {}: {e}", input.display()));
    let mut r = BufReader::with_capacity(16 * 1024 * 1024, f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).expect("read magic");
    if &magic != b"HFKLDR\0\0" {
        eprintln!("bad magic {:?} — not a legacy .kldref.bin", magic);
        std::process::exit(1);
    }

    let mut hdr = [0u8; 24];
    r.read_exact(&mut hdr).expect("read header");
    let version = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    if version != 1 {
        eprintln!("unsupported legacy ref version {version} (expected 1)");
        std::process::exit(1);
    }
    let n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let n_vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let top_k = u16::from_le_bytes(hdr[16..18].try_into().unwrap()) as usize;

    // Back-calculate scored_per_chunk from file size
    let header_bytes = 8 + 24;
    let tokens_bytes = n_chunk * n_ctx * 4;
    let block_size = 8 + 8 * top_k; // 4*top_k indices + 4*top_k log_probs + 4 residual + 4 pad
    let blocks_total_bytes = file_size - header_bytes - tokens_bytes;
    if blocks_total_bytes % (n_chunk * block_size) != 0 {
        eprintln!(
            "file size {file_size} does not factor cleanly: \
             blocks_bytes={blocks_total_bytes} n_chunk={n_chunk} block_size={block_size}"
        );
        std::process::exit(1);
    }
    let scored_per_chunk = blocks_total_bytes / (n_chunk * block_size);

    eprintln!(
        "convert_kldref: n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk} \
         top_k={top_k} scored_per_chunk={scored_per_chunk}"
    );

    // ── Read tokens ──────────────────────────────────────────────────────────
    let mut tokens_raw = vec![0u8; tokens_bytes];
    r.read_exact(&mut tokens_raw).expect("read tokens");

    // ── Read blocks → separate tensor buffers ────────────────────────────────
    let mut indices_buf = vec![0u8; n_chunk * scored_per_chunk * top_k * 4];
    let mut log_probs_buf = vec![0u8; n_chunk * scored_per_chunk * top_k * 4];
    let mut residual_buf = vec![0u8; n_chunk * scored_per_chunk * 4];
    let mut block = vec![0u8; block_size];

    for c in 0..n_chunk {
        for s in 0..scored_per_chunk {
            r.read_exact(&mut block).expect("read block");
            let out_pos = (c * scored_per_chunk + s) * top_k;
            // top_indices: offsets 0 .. top_k*4
            indices_buf[out_pos * 4..(out_pos + top_k) * 4].copy_from_slice(&block[0..top_k * 4]);
            // top_log_probs: offsets top_k*4 .. top_k*8
            log_probs_buf[out_pos * 4..(out_pos + top_k) * 4]
                .copy_from_slice(&block[top_k * 4..top_k * 8]);
            // residual_mass (f32 at top_k*8): 4 bytes
            let resid_off = (c * scored_per_chunk + s) * 4;
            residual_buf[resid_off..resid_off + 4]
                .copy_from_slice(&block[top_k * 8..top_k * 8 + 4]);
        }
    }

    // ── Write temp files ─────────────────────────────────────────────────────
    let tmp_dir = output.parent().unwrap_or(std::path::Path::new("."));
    let mk_tmp = |suffix: &str| -> PathBuf {
        tmp_dir.join(format!(
            ".convert_kldref_tmp_{suffix}_{}.bin",
            std::process::id()
        ))
    };
    let tokens_path = mk_tmp("tokens");
    let indices_path = mk_tmp("indices");
    let log_probs_path = mk_tmp("log_probs");
    let residual_path = mk_tmp("residual");

    for (path, data) in [
        (&tokens_path, &tokens_raw),
        (&indices_path, &indices_buf),
        (&log_probs_path, &log_probs_buf),
        (&residual_path, &residual_buf),
    ] {
        let mut w = BufWriter::new(
            File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display())),
        );
        w.write_all(data).expect("write tmp");
    }

    // ── Build HFQM package ───────────────────────────────────────────────────
    let metadata = json!({
        "schema": 1,
        "artifact_kind": "hipfire.kldref",
        "package_schema": "hipfire.kldref.v1",
        "producer": "convert_kldref",
        "format": "HFQM",
        "format_version": 1u32,
        "source": input.display().to_string(),
        "n_ctx": n_ctx,
        "n_vocab": n_vocab,
        "n_chunk": n_chunk,
        "top_k": top_k,
        "scored_per_chunk": scored_per_chunk,
        "arch_id": HFQM_ARCH_NON_WEIGHT_PACKAGE,
        "payloads": {
            "kldref.tokens":       {"dtype": "u32", "shape": [n_chunk, n_ctx]},
            "kldref.top_indices":  {"dtype": "u32", "shape": [n_chunk, scored_per_chunk, top_k]},
            "kldref.top_log_probs":{"dtype": "f32", "shape": [n_chunk, scored_per_chunk, top_k]},
            "kldref.residual_mass":{"dtype": "f32", "shape": [n_chunk, scored_per_chunk]},
        },
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    let entries = vec![
        HfqPackageWriteEntry {
            name: "kldref.tokens".to_string(),
            quant_type: 0,
            shape: vec![n_chunk as u32, n_ctx as u32],
            group_size: 0,
            data_size: tokens_raw.len() as u64,
            source_path: tokens_path.clone(),
        },
        HfqPackageWriteEntry {
            name: "kldref.top_indices".to_string(),
            quant_type: 0,
            shape: vec![n_chunk as u32, scored_per_chunk as u32, top_k as u32],
            group_size: 0,
            data_size: indices_buf.len() as u64,
            source_path: indices_path.clone(),
        },
        HfqPackageWriteEntry {
            name: "kldref.top_log_probs".to_string(),
            quant_type: 0,
            shape: vec![n_chunk as u32, scored_per_chunk as u32, top_k as u32],
            group_size: 0,
            data_size: log_probs_buf.len() as u64,
            source_path: log_probs_path.clone(),
        },
        HfqPackageWriteEntry {
            name: "kldref.residual_mass".to_string(),
            quant_type: 0,
            shape: vec![n_chunk as u32, scored_per_chunk as u32],
            group_size: 0,
            data_size: residual_buf.len() as u64,
            source_path: residual_path.clone(),
        },
    ];

    write_hfqm_package_from_files(
        &output,
        HFQM_ARCH_NON_WEIGHT_PACKAGE,
        &metadata_json,
        &entries,
    )
    .unwrap_or_else(|e| panic!("write HFQM: {e}"));

    for p in [&tokens_path, &indices_path, &log_probs_path, &residual_path] {
        let _ = std::fs::remove_file(p);
    }

    let out_size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "convert_kldref: wrote {} ({:.3} GB)",
        output.display(),
        out_size as f64 / 1e9
    );
}
