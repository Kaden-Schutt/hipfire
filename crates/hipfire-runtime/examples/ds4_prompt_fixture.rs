// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Build an exact-N token prompt fixture from a deterministic source corpus
//! using the model's embedded HFQ tokenizer.
//!
//! Used by the DS4 gfx942/MI300X G0 measurement contract (E3): take the first
//! N token IDs of a revisioned prose/chat source, decode them, require a
//! fail-closed re-encode round-trip, and write the UTF-8 text with no final LF.
//!
//! Usage:
//!   ds4_prompt_fixture --model <path-to-hfq> --source <path-to-txt> \
//!       --out <path-to-txt> [--count 2048] [--tokens-out <path-to-bin>]
//!
//! Output text: UTF-8, no trailing newline.
//! Optional --tokens-out: contiguous u32 little-endian token IDs (same as
//! `tokenize_slice`).

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut source: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut tokens_out: Option<PathBuf> = None;
    let mut count: usize = 2048;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(require_val(&argv, i, "--model")));
                i += 2;
            }
            "--source" => {
                source = Some(PathBuf::from(require_val(&argv, i, "--source")));
                i += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(require_val(&argv, i, "--out")));
                i += 2;
            }
            "--tokens-out" => {
                tokens_out = Some(PathBuf::from(require_val(&argv, i, "--tokens-out")));
                i += 2;
            }
            "--count" => {
                let raw = require_val(&argv, i, "--count");
                count = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("ds4_prompt_fixture: --count must be a positive integer, got {raw}");
                    std::process::exit(1);
                });
                if count == 0 {
                    eprintln!("ds4_prompt_fixture: --count must be >= 1");
                    std::process::exit(1);
                }
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: ds4_prompt_fixture --model <hfq> --source <txt> --out <txt> \
                     [--count 2048] [--tokens-out <bin>]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("ds4_prompt_fixture: unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    let model = model.unwrap_or_else(|| die("--model required"));
    let source = source.unwrap_or_else(|| die("--source required"));
    let out = out.unwrap_or_else(|| die("--out required"));

    let hfq = HfqFile::open(&model).unwrap_or_else(|e| {
        eprintln!(
            "ds4_prompt_fixture: open model {}: {e}",
            model.display()
        );
        std::process::exit(1);
    });
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).unwrap_or_else(|e| {
        eprintln!("ds4_prompt_fixture: load tokenizer from hfq metadata: {e}");
        std::process::exit(1);
    });

    let source_text = std::fs::read_to_string(&source).unwrap_or_else(|e| {
        eprintln!(
            "ds4_prompt_fixture: read source {}: {e}",
            source.display()
        );
        std::process::exit(1);
    });
    if !source_text.is_ascii() {
        eprintln!(
            "ds4_prompt_fixture: source {} is not pure ASCII",
            source.display()
        );
        std::process::exit(1);
    }

    eprintln!(
        "ds4_prompt_fixture: model={} source={} ({} bytes) count={count}",
        model.display(),
        source.display(),
        source_text.len()
    );

    let full_ids: Vec<u32> = tokenizer.encode(&source_text);
    if full_ids.len() < count {
        eprintln!(
            "ds4_prompt_fixture: source yielded {} tokens; need at least {count}",
            full_ids.len()
        );
        std::process::exit(1);
    }

    let prefix_ids: Vec<u32> = full_ids[..count].to_vec();
    // Tokenizer::decode — crates/hipfire-runtime/src/tokenizer.rs:651
    let mut decoded = tokenizer.decode(&prefix_ids);
    require_round_trip(&tokenizer, &prefix_ids, &decoded, "initial decode");

    // Strip trailing newlines only when the round-trip still holds exactly.
    if decoded.ends_with('\n') {
        let stripped = decoded.trim_end_matches(['\n', '\r']).to_string();
        match try_round_trip(&tokenizer, &prefix_ids, &stripped) {
            Ok(()) => {
                decoded = stripped;
            }
            Err(msg) => {
                eprintln!(
                    "ds4_prompt_fixture: trailing-newline strip conflicts with exact \
                     {count}-token round-trip ({msg}); refusing to strip or write a \
                     wrong-count fixture"
                );
                std::process::exit(1);
            }
        }
    }

    // Final guard before write (includes post-strip state).
    require_round_trip(&tokenizer, &prefix_ids, &decoded, "pre-write");

    if decoded.ends_with('\n') {
        eprintln!(
            "ds4_prompt_fixture: internal error: decoded fixture still ends with LF after strip pass"
        );
        std::process::exit(1);
    }

    let out_bytes = decoded.as_bytes();
    std::fs::write(&out, out_bytes).unwrap_or_else(|e| {
        eprintln!("ds4_prompt_fixture: write {}: {e}", out.display());
        std::process::exit(1);
    });

    if let Some(tok_path) = tokens_out.as_ref() {
        let mut f = std::fs::File::create(tok_path).unwrap_or_else(|e| {
            eprintln!(
                "ds4_prompt_fixture: create tokens-out {}: {e}",
                tok_path.display()
            );
            std::process::exit(1);
        });
        let bytes: Vec<u8> = prefix_ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        f.write_all(&bytes).unwrap_or_else(|e| {
            eprintln!(
                "ds4_prompt_fixture: write tokens-out {}: {e}",
                tok_path.display()
            );
            std::process::exit(1);
        });
        eprintln!(
            "ds4_prompt_fixture: wrote {} token bytes to {}",
            bytes.len(),
            tok_path.display()
        );
    }

    // md5 / sha2 are not dependencies of hipfire-runtime; print portable
    // digests via a tiny std-only SHA-256 and note MD5 unavailability.
    let sha256 = sha256_hex(out_bytes);
    println!("token_count={count}");
    println!("output_bytes={}", out_bytes.len());
    println!("md5=unavailable (hipfire-runtime has no md5 dependency; not added)");
    println!("sha256={sha256}");
    eprintln!(
        "ds4_prompt_fixture: wrote {} bytes (no final LF) to {}",
        out_bytes.len(),
        out.display()
    );
}

fn require_val<'a>(argv: &'a [String], i: usize, flag: &str) -> &'a str {
    argv.get(i + 1).map(|s| s.as_str()).unwrap_or_else(|| {
        eprintln!("ds4_prompt_fixture: {flag} requires a value");
        std::process::exit(1);
    })
}

fn die(msg: &str) -> ! {
    eprintln!("ds4_prompt_fixture: {msg}");
    std::process::exit(1);
}

fn require_round_trip(tokenizer: &Tokenizer, expected: &[u32], text: &str, phase: &str) {
    if let Err(msg) = try_round_trip(tokenizer, expected, text) {
        eprintln!("ds4_prompt_fixture: round-trip failed ({phase}): {msg}");
        std::process::exit(1);
    }
}

fn try_round_trip(tokenizer: &Tokenizer, expected: &[u32], text: &str) -> Result<(), String> {
    let again = tokenizer.encode(text);
    if again.len() != expected.len() {
        return Err(format!(
            "re-encode length {} != expected {}",
            again.len(),
            expected.len()
        ));
    }
    for (i, (a, b)) in again.iter().zip(expected.iter()).enumerate() {
        if a != b {
            return Err(format!(
                "first differing index {i}: re-encoded id {a} != expected id {b}"
            ));
        }
    }
    Ok(())
}

/// Minimal SHA-256 (FIPS 180-4) so the example stays dependency-free inside
/// hipfire-runtime. Not a general crypto module — fixture identity only.
fn sha256_hex(data: &[u8]) -> String {
    let hash = sha256(data);
    let mut s = String::with_capacity(64);
    for b in hash {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).saturating_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod sha256_smoke {
    #[test]
    fn empty_sha256() {
        // FIPS empty-string vector.
        let got = super::sha256_hex(b"");
        assert_eq!(
            got,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
