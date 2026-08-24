// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tokenize a pinned text corpus once into a flat `u32` LE `tokens.bin`.
//!
//! Both the parent baseline and every quantized capture consume this exact
//! file so neither side re-tokenizes. The corpus md5 is asserted against the
//! known wikitext2 slice pin before any encoding proceeds.
//!
//! Usage:
//! ```text
//! ds4_tokenize_corpus --tokenizer <checkpoint>/tokenizer.json \
//!                     --corpus <file.txt> --tokens 1024 --out tokens.bin
//! ```

use hipfire_ds4_parent::manifest::sha256_file;
use hipfire_runtime::tokenizer::Tokenizer;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::{Command, ExitCode};

/// Canonical wikitext2 slice pin (handoff / Gate 6).
const EXPECTED_CORPUS_MD5: &str = "83b0205a304bf4e52172ecdb05f2e895";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let tokenizer_path = Path::new(&args.tokenizer);
    let corpus_path = Path::new(&args.corpus);
    let out_path = Path::new(&args.out);

    if !tokenizer_path.is_file() {
        return Err(format!(
            "deepseek4 parent: --tokenizer is not a file: {}",
            tokenizer_path.display()
        ));
    }
    if !corpus_path.is_file() {
        return Err(format!(
            "deepseek4 parent: --corpus is not a file: {}",
            corpus_path.display()
        ));
    }
    if args.tokens == 0 {
        return Err("deepseek4 parent: --tokens must be > 0".into());
    }

    // ── Corpus pin ──────────────────────────────────────────────────────
    let corpus_bytes = fs::metadata(corpus_path)
        .map_err(|e| {
            format!(
                "deepseek4 parent: corpus metadata {}: {e}",
                corpus_path.display()
            )
        })?
        .len();
    let corpus_md5 = md5_file(corpus_path)?;
    if corpus_md5 != EXPECTED_CORPUS_MD5 {
        return Err(format!(
            "deepseek4 parent: corpus md5 mismatch for {}\n  expected: {EXPECTED_CORPUS_MD5}\n  actual:   {corpus_md5}\n\
             refusing to tokenize an unpinned corpus",
            corpus_path.display()
        ));
    }

    // ── Tokenizer ───────────────────────────────────────────────────────
    let tokenizer_sha = sha256_file(tokenizer_path)?;
    let tokenizer = Tokenizer::from_tokenizer_json(tokenizer_path)
        .map_err(|e| format!("deepseek4 parent: tokenizer load: {e}"))?
        .ok_or_else(|| {
            format!(
                "deepseek4 parent: missing tokenizer at {}",
                tokenizer_path.display()
            )
        })?;

    let text = fs::read_to_string(corpus_path).map_err(|e| {
        format!(
            "deepseek4 parent: read corpus {}: {e}",
            corpus_path.display()
        )
    })?;
    let all_ids = tokenizer.encode(&text);
    if all_ids.len() < args.tokens {
        return Err(format!(
            "deepseek4 parent: corpus produced only {} tokens; need {}",
            all_ids.len(),
            args.tokens
        ));
    }
    let ids: Vec<u32> = all_ids.into_iter().take(args.tokens).collect();
    let n = ids.len();
    debug_assert_eq!(n, args.tokens);

    // ── Write flat u32 LE ───────────────────────────────────────────────
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "deepseek4 parent: create out dir {}: {e}",
                    parent.display()
                )
            })?;
        }
    }
    let mut f = File::create(out_path).map_err(|e| {
        format!(
            "deepseek4 parent: create {}: {e}",
            out_path.display()
        )
    })?;
    let bytes = u32_slice_as_le_bytes(&ids);
    f.write_all(bytes).map_err(|e| {
        format!(
            "deepseek4 parent: write {}: {e}",
            out_path.display()
        )
    })?;
    f.flush().map_err(|e| {
        format!(
            "deepseek4 parent: flush {}: {e}",
            out_path.display()
        )
    })?;

    let token_ids_sha = sha256_bytes_local(bytes);
    let out_bytes = fs::metadata(out_path)
        .map_err(|e| format!("deepseek4 parent: out metadata: {e}"))?
        .len();
    let expect_out = (n as u64).saturating_mul(4);
    if out_bytes != expect_out {
        return Err(format!(
            "deepseek4 parent: wrote {out_bytes} bytes, expected {expect_out}"
        ));
    }

    // Round-trip self-check: read back and compare.
    let reread = read_token_ids(out_path)?;
    if reread != ids {
        return Err(
            "deepseek4 parent: tokens.bin round-trip mismatch after write".into(),
        );
    }

    let first16: Vec<u32> = ids.iter().copied().take(16).collect();
    let last16: Vec<u32> = if n >= 16 {
        ids[n - 16..].to_vec()
    } else {
        ids.clone()
    };

    println!("=== ds4_tokenize_corpus ===");
    println!("corpus_path: {}", corpus_path.display());
    println!("corpus_md5: {corpus_md5}");
    println!("corpus_bytes: {corpus_bytes}");
    println!("tokenizer: {}", tokenizer_path.display());
    println!("tokenizer_sha256: {tokenizer_sha}");
    println!("token_count: {n}");
    println!("token_ids_sha256: {token_ids_sha}");
    println!("out: {} ({out_bytes} bytes)", out_path.display());
    println!("first_16: {first16:?}");
    println!("last_16: {last16:?}");
    println!("bos_id={} eos_id={} add_bos={}", tokenizer.bos_id, tokenizer.eos_id, tokenizer.add_bos);
    println!("OK");
    Ok(())
}

/// Load a flat `u32` LE token-ids file (shared contract with the forward gate).
pub fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| {
        format!(
            "deepseek4 parent: read token-ids {}: {e}",
            path.display()
        )
    })?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "deepseek4 parent: token-ids file {} has {} bytes (not a multiple of 4)",
            path.display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn u32_slice_as_le_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn sha256_bytes_local(bytes: &[u8]) -> String {
    hipfire_ds4_parent::manifest::sha256_bytes(bytes)
}

fn md5_file(path: &Path) -> Result<String, String> {
    let out = Command::new("md5sum")
        .arg(path)
        .output()
        .map_err(|e| format!("deepseek4 parent: invoke md5sum: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "deepseek4 parent: md5sum failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace()
        .next()
        .map(|h| h.to_string())
        .ok_or_else(|| {
            format!(
                "deepseek4 parent: empty md5sum output for {}",
                path.display()
            )
        })
}

struct Args {
    tokenizer: String,
    corpus: String,
    tokens: usize,
    out: String,
}

fn parse_args() -> Result<Args, String> {
    let mut tokenizer: Option<String> = None;
    let mut corpus: Option<String> = None;
    let mut tokens: Option<usize> = None;
    let mut out: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--tokenizer" => {
                tokenizer = Some(
                    args.next()
                        .ok_or_else(|| "flag --tokenizer missing value".to_string())?,
                );
            }
            "--corpus" => {
                corpus = Some(
                    args.next()
                        .ok_or_else(|| "flag --corpus missing value".to_string())?,
                );
            }
            "--tokens" => {
                let v = args
                    .next()
                    .ok_or_else(|| "flag --tokens missing value".to_string())?;
                tokens = Some(v.parse().map_err(|e| format!("--tokens: {e}"))?);
            }
            "--out" => {
                out = Some(
                    args.next()
                        .ok_or_else(|| "flag --out missing value".to_string())?,
                );
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_tokenize_corpus --tokenizer <tokenizer.json> \
                     --corpus <file.txt> --tokens N --out tokens.bin"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    Ok(Args {
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        corpus: corpus.ok_or("--corpus is required")?,
        tokens: tokens.ok_or("--tokens is required")?,
        out: out.ok_or("--out is required")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn token_ids_round_trip_u32_le() {
        let ids: Vec<u32> = vec![0, 1, 42, 129_279, 0xDEAD_BEEF];
        let dir = std::env::temp_dir().join(format!(
            "ds4-tokenize-rt-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.bin");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(u32_slice_as_le_bytes(&ids)).unwrap();
        }
        let got = read_token_ids(&path).unwrap();
        assert_eq!(got, ids);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn token_ids_rejects_odd_byte_length() {
        let dir = std::env::temp_dir().join(format!(
            "ds4-tokenize-odd-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.bin");
        fs::write(&path, [0u8, 1, 2]).unwrap();
        let err = read_token_ids(&path).unwrap_err();
        assert!(err.contains("multiple of 4"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn expected_corpus_md5_is_canonical_pin() {
        assert_eq!(EXPECTED_CORPUS_MD5.len(), 32);
        assert_eq!(EXPECTED_CORPUS_MD5, "83b0205a304bf4e52172ecdb05f2e895");
    }
}
