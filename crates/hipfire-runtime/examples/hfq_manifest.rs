// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Print an HFQ container's tensor manifest and per-quant-type census.
//!
//! Answers "what quantization policy did this artifact actually get?" without
//! loading weights onto a GPU — so it works on a box too small for the model.
//! Written while auditing Muse Glimmer's MQ4 artifact, where the requirement was
//! that norms stay F16 and embed/lm_head stay Q8; a census is the only way to
//! confirm that held for all 627 tensors rather than for the ones you spot-check.
//!
//! Usage:
//!   hfq_manifest <model.hfq|model.mq4> [name-substring-filter ...]

fn qt_name(qt: u8) -> &'static str {
    match qt {
        0 => "F32",
        1 => "F16",
        3 => "Q8_0",
        6 => "HFQ4G256",
        13 => "MQ4G256",
        15 => "MQ6G256",
        19 => "MQ4G256(alt)",
        30 => "MQ4G256Lloyd",
        31 => "MQ5G256",
        _ => "?",
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: hfq_manifest <model.hfq> [name-substring ...]");
            std::process::exit(2);
        }
    };
    let filters: Vec<String> = args.collect();

    let hfq = match hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(&path)) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("open {path}: {e:?}");
            std::process::exit(1);
        }
    };

    let tensors = hfq.tensors();
    println!("arch_id: {}", hfq.arch_id);
    println!("tensors: {}", tensors.len());

    let mut census: std::collections::BTreeMap<u8, (usize, u64)> = Default::default();
    for t in tensors {
        let e = census.entry(t.quant_type).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.data_size as u64;
    }
    println!("\ncensus:");
    for (qt, (count, bytes)) in &census {
        println!(
            "  {:<14} {:>5} tensors   {:>10.2} MiB",
            qt_name(*qt),
            count,
            *bytes as f64 / (1024.0 * 1024.0)
        );
    }

    if !filters.is_empty() {
        println!("\nmatching tensors:");
        for t in tensors {
            if filters.iter().any(|f| t.name.contains(f.as_str())) {
                println!(
                    "  {:<62} {:<14} {:?}",
                    t.name,
                    qt_name(t.quant_type),
                    t.shape
                );
            }
        }
    }
}
