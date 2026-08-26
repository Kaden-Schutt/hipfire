// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare a candidate `.plog` against a parent reference on the same tokens.
//!
//! Usage:
//!   ds4_parent_kld --reference parent.plog --candidate mq2r.plog \
//!                  --tokens tokens.bin [--json report.json]
//!
//! `--tokens` is a flat little-endian `u32` file of the token IDs the logits
//! were produced from (length must equal `n_tokens` in both plogs). Prints a
//! human-readable table; with `--json`, also writes the `DivergenceReport`
//! plus input paths and sha256 digests.

use hipfire_ds4_parent::plog::{compare, DivergenceReport, PlogReader};
use std::io::Read;
use std::path::Path;

fn main() {
    let mut reference: Option<String> = None;
    let mut candidate: Option<String> = None;
    let mut tokens: Option<String> = None;
    let mut json_out: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--reference" => {
                reference = Some(args.next().expect("flag --reference missing value"));
            }
            "--candidate" => {
                candidate = Some(args.next().expect("flag --candidate missing value"));
            }
            "--tokens" => {
                tokens = Some(args.next().expect("flag --tokens missing value"));
            }
            "--json" => {
                json_out = Some(args.next().expect("flag --json missing value"));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_kld --reference parent.plog --candidate mq2r.plog \
                     --tokens tokens.bin [--json report.json]"
                );
                std::process::exit(0);
            }
            _ => panic!("unknown flag: {flag}"),
        }
    }

    let reference = reference.expect(
        "usage: ds4_parent_kld --reference parent.plog --candidate mq2r.plog \
         --tokens tokens.bin [--json report.json]",
    );
    let candidate = candidate.expect("missing --candidate");
    let tokens_path = tokens.expect("missing --tokens");

    let ref_reader = PlogReader::open(Path::new(&reference)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let cand_reader = PlogReader::open(Path::new(&candidate)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let target_ids = read_token_ids(Path::new(&tokens_path)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let report = compare(&ref_reader, &cand_reader, &target_ids).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    print_table(&reference, &candidate, &tokens_path, &report);

    if let Some(out) = json_out.as_ref() {
        let ref_sha = sha256_file(Path::new(&reference)).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let cand_sha = sha256_file(Path::new(&candidate)).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let tok_sha = sha256_file(Path::new(&tokens_path)).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let payload = serde_json::json!({
            "reference": {"path": reference, "sha256": ref_sha},
            "candidate": {"path": candidate, "sha256": cand_sha},
            "tokens": {"path": tokens_path, "sha256": tok_sha},
            "report": report,
        });
        let text = serde_json::to_string_pretty(&payload).expect("serialize report");
        std::fs::write(out, text).unwrap_or_else(|e| {
            eprintln!("deepseek4 parent: write json {out}: {e}");
            std::process::exit(1);
        });
        eprintln!("Wrote JSON report to {out}");
    }
}

fn print_table(reference: &str, candidate: &str, tokens: &str, r: &DivergenceReport) {
    println!("ds4 parent KLD");
    println!("----------------------------------------");
    println!("reference:  {reference}");
    println!("candidate:  {candidate}");
    println!("tokens:     {tokens}");
    println!("n_tokens:   {}", r.n_tokens);
    println!("vocab:      {}", r.vocab);
    println!();
    println!("KLD (P_ref || Q_cand)");
    println!("  mean:     {:.10}", r.kld_mean);
    println!("  p50:      {:.10}", r.kld_p50);
    println!("  p95:      {:.10}", r.kld_p95);
    println!("  max:      {:.10}", r.kld_max);
    println!();
    println!("top-1 agreement:    {:.6}", r.top1_agreement);
    println!("top-5 overlap mean: {:.6}", r.top5_overlap_mean);
    println!();
    println!("PPL reference:  {:.6}", r.ppl_reference);
    println!("PPL candidate:  {:.6}", r.ppl_candidate);
    println!("max |d logit|:  {:.6}", r.max_abs_logit_delta);
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("deepseek4 parent: read tokens {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "deepseek4 parent: tokens {} size {} not multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
        out.push(u32::from_le_bytes(b));
    }
    Ok(out)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("deepseek4 parent: open {} for sha256: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("deepseek4 parent: read {} for sha256: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex32(hasher.finalize()))
}

// Minimal SHA-256 (stdlib only — matches the ds4_prompt_fixture pattern; no
// sha2 crate dep on this example path).
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    bit_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            buffer_len: 0,
            bit_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        let mut offset = 0;
        while offset < data.len() {
            let take = (64 - self.buffer_len).min(data.len() - offset);
            self.buffer[self.buffer_len..self.buffer_len + take]
                .copy_from_slice(&data[offset..offset + take]);
            self.buffer_len += take;
            offset += take;
            if self.buffer_len == 64 {
                self.compress();
                self.bit_len += 512;
                self.buffer_len = 0;
            }
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.bit_len + (self.buffer_len as u64) * 8;
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;
        if self.buffer_len > 56 {
            for i in self.buffer_len..64 {
                self.buffer[i] = 0;
            }
            self.compress();
            self.buffer_len = 0;
        }
        for i in self.buffer_len..56 {
            self.buffer[i] = 0;
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        self.compress();
        let mut out = [0u8; 32];
        for (i, &w) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    fn compress(&mut self) {
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
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(self.buffer[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
