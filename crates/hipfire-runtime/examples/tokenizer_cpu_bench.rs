// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::tokenizer::Tokenizer;
use std::hint::black_box;
use std::time::Instant;

fn token_hash(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tokenizer_cpu_bench <tokenizer.json> <input.txt> [iterations]");
        std::process::exit(2);
    }
    let iterations = args
        .get(3)
        .map(|value| {
            value
                .parse::<usize>()
                .expect("iterations must be an integer")
        })
        .unwrap_or(200);
    assert!(iterations > 0, "iterations must be non-zero");

    let tokenizer_json = std::fs::read_to_string(&args[1]).expect("read tokenizer.json");
    let input = std::fs::read_to_string(&args[2]).expect("read input text");
    let tokenizer = Tokenizer::from_hf_json(&tokenizer_json).expect("load tokenizer");

    let cold_start = Instant::now();
    let cold_tokens = tokenizer.encode(black_box(&input));
    let cold_us = cold_start.elapsed().as_secs_f64() * 1e6;
    if std::env::var_os("HIPFIRE_TOKENIZER_DUMP").is_some() {
        println!("tokens={cold_tokens:?}");
    }

    let warm_start = Instant::now();
    let mut warm_hash = 0u64;
    for _ in 0..iterations {
        let tokens = tokenizer.encode(black_box(&input));
        warm_hash ^= black_box(token_hash(&tokens));
    }
    let warm_us = warm_start.elapsed().as_secs_f64() * 1e6 / iterations as f64;

    println!(
        "bytes={} tokens={} hash={:016x} cold_us={:.3} warm_us={:.3} iterations={} guard={:016x}",
        input.len(),
        cold_tokens.len(),
        token_hash(&cold_tokens),
        cold_us,
        warm_us,
        iterations,
        warm_hash,
    );
}
