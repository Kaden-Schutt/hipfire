// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

//! Stable hashing primitives shared by model identity and evidence contracts.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

pub fn file_hash(path: &Path) -> Option<String> {
    command_digest("sha256sum", path).or_else(|| Some(stable_hash_file_fallback(path)))
}

pub fn stable_hash_file_fallback(path: &Path) -> String {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return "unavailable".to_string(),
    };
    let mut state = Fnv64::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => state.update(&buf[..n]),
            Err(_) => return "unavailable".to_string(),
        }
    }
    format!("fnv64:{:016x}", state.finish())
}

pub fn stable_hash_bytes(bytes: &[u8]) -> String {
    let mut state = Fnv64::new();
    state.update(bytes);
    format!("fnv64:{:016x}", state.finish())
}

pub fn stable_score(input: &str) -> f64 {
    let mut state = Fnv64::new();
    state.update(input.as_bytes());
    (state.finish() as f64) / (u64::MAX as f64)
}

fn command_digest(tool: &str, path: &Path) -> Option<String> {
    let out = Command::new(tool).arg(path).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
}

struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_hash_bytes_is_deterministic() {
        assert_eq!(stable_hash_bytes(b"abc"), "fnv64:e71fa2190541574b");
        assert_eq!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abc"));
        assert_ne!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abcd"));
    }

    #[test]
    fn stable_hash_file_fallback_reports_unavailable() {
        assert_eq!(
            stable_hash_file_fallback(Path::new("/definitely/not/a/hipfire/file")),
            "unavailable"
        );
    }
}
