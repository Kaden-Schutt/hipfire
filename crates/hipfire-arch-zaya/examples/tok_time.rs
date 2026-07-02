// SPDX-License-Identifier: Apache-2.0
// hipfire — profiling probe: how long does the resident tokenizer take to encode
// increasing corpus sizes? Isolates the daemon kld_eval reference-load stall
// (full-corpus tokenization) from the GPU forward. No GPU.
//
// Run: cargo run --release -p hipfire-arch-zaya --example tok_time -- <hfq> <corpus.txt>

use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
use std::time::Instant;

fn main() {
    let model = std::env::args().nth(1).expect("hfq path");
    let corpus = std::env::args().nth(2).expect("corpus path");
    let hfq = HfqFile::open(Path::new(&model)).expect("open hfq");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let raw = std::fs::read(&corpus).expect("read corpus");
    let text = String::from_utf8_lossy(&raw).to_string();
    eprintln!(
        "corpus: {} bytes / {} chars",
        raw.len(),
        text.chars().count()
    );
    for take_chars in [16_000usize, 64_000, 256_000, 1_000_000, usize::MAX] {
        let slice: String = text.chars().take(take_chars).collect();
        let t = Instant::now();
        let toks = tok.encode(&slice);
        let dt = t.elapsed().as_secs_f64();
        eprintln!(
            "{:>10} chars -> {:>9} tokens in {:>8.3}s  ({:.0} tok/s)",
            slice.chars().count(),
            toks.len(),
            dt,
            toks.len() as f64 / dt.max(1e-9)
        );
        if slice.len() == text.len() {
            break;
        }
    }
}
