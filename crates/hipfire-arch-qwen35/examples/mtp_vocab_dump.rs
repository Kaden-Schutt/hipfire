// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Export the exact compressed-vocabulary map embedded in a deployed `.mtp`
//! sidecar. This is deliberately CPU-only.

use hipfire_runtime::hfq::HfqFile;
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("Usage: mtp_vocab_dump <input.mtp> <output.json>");
        std::process::exit(2);
    }));
    let output = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("Usage: mtp_vocab_dump <input.mtp> <output.json>");
        std::process::exit(2);
    }));
    if args.next().is_some() {
        eprintln!("Usage: mtp_vocab_dump <input.mtp> <output.json>");
        std::process::exit(2);
    }
    let hfq =
        HfqFile::open(&input).unwrap_or_else(|error| panic!("open {}: {error}", input.display()));
    let meta: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).expect("parse .mtp metadata");
    let expected = meta
        .get("compressed_vocab_size")
        .and_then(|value| value.as_u64())
        .expect(".mtp has no compressed_vocab_size") as usize;
    assert!(expected > 0, ".mtp does not contain a compressed head");
    let (info, bytes) = hfq
        .tensor_data_vec("lm_head_draft.vocab_map")
        .expect(".mtp has no lm_head_draft.vocab_map");
    assert_eq!(info.shape, vec![expected as u32]);
    assert_eq!(bytes.len(), expected * 4);
    let draft_to_full: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let full_vocab_size = meta
        .get("vocab_size")
        .and_then(|value| value.as_u64())
        .expect(".mtp metadata has no vocab_size");
    let body = json!({
        "schema_version": 1,
        "source_mtp": input.display().to_string(),
        "compressed_vocab_size": expected,
        "full_vocab_size": full_vocab_size,
        "draft_to_full": draft_to_full,
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create output directory");
    }
    let mut file = File::create(&output)
        .unwrap_or_else(|error| panic!("create {}: {error}", output.display()));
    serde_json::to_writer_pretty(&mut file, &body).expect("write vocab JSON");
    file.write_all(b"\n").expect("terminate vocab JSON");
    file.sync_all().expect("sync vocab JSON");
    eprintln!(
        "wrote {}: {} / {} token ids",
        output.display(),
        expected,
        full_vocab_size
    );
}
