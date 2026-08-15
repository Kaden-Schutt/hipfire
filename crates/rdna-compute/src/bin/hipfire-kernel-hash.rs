// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Print the packaging hash for a kernel source file.
//!
//! Reuses the SAME `KernelCompiler::hash_parts` code path as the runtime
//! (`packaging_hash_for` → `hash_parts` with `toolchain_id = ""`) so
//! KERNEL_CACHE_ABI or field-order changes cannot silently diverge.
//!
//! Usage:
//!   hipfire-kernel-hash --arch <gfx> [--extra-flags <flags>] [--name <kernel>] <source.hip>
//! If --name is omitted, it is derived from the source path's file stem
//! (stripping `.gfxNNN` variant tags and `.hip`).

use std::env;
use std::path::Path;
use std::process::ExitCode;

fn print_help() {
    eprintln!(
        "Usage: hipfire-kernel-hash --arch <gfx> [--extra-flags <flags>] [--name <kernel>] <source.hip>"
    );
    eprintln!("");
    eprintln!("Prints the packaging hash (toolchain_id=\"\") for the given kernel source.");
    eprintln!("Reuses KernelCompiler::packaging_hash_for so the key matches a compiler-free runtime.");
}

fn derive_name(source_path: &str) -> String {
    let p = Path::new(source_path);
    let file = p.file_name().and_then(|s| s.to_str()).unwrap_or(source_path);
    // Strip .hip suffix if present
    let stem = if let Some(s) = file.strip_suffix(".hip") {
        s
    } else {
        file
    };
    // Strip variant tags like .gfx1201 or .gfx12
    // Check if stem ends with .gfx<digits>
    if let Some(dot) = stem.rfind(".gfx") {
        let suffix = &stem[dot + 1..]; // gfx...
        if suffix.starts_with("gfx") && suffix[3..].chars().all(|c| c.is_ascii_digit()) {
            return stem[..dot].to_string();
        }
    }
    stem.to_string()
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    let mut arch: Option<String> = None;
    let mut extra_flags: Option<String> = None;
    let mut name: Option<String> = None;
    let mut source_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--arch" => {
                if i + 1 >= args.len() {
                    eprintln!("--arch requires a value");
                    print_help();
                    return ExitCode::FAILURE;
                }
                arch = Some(args[i + 1].clone());
                i += 2;
            }
            "--extra-flags" => {
                if i + 1 >= args.len() {
                    eprintln!("--extra-flags requires a value");
                    return ExitCode::FAILURE;
                }
                extra_flags = Some(args[i + 1].clone());
                i += 2;
            }
            "--name" => {
                if i + 1 >= args.len() {
                    eprintln!("--name requires a value");
                    return ExitCode::FAILURE;
                }
                name = Some(args[i + 1].clone());
                i += 2;
            }
            "--help" | "-h" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            s if s.starts_with('-') => {
                eprintln!("unknown flag: {s}");
                print_help();
                return ExitCode::FAILURE;
            }
            pos => {
                if source_path.is_some() {
                    // If a second positional is given and --name not set, treat second as name?
                    // But prefer explicit --name. For compat, if two positionals: source and name
                    if name.is_none() {
                        name = Some(pos.to_string());
                    } else {
                        eprintln!("unexpected extra argument: {pos}");
                        print_help();
                        return ExitCode::FAILURE;
                    }
                } else {
                    source_path = Some(pos.to_string());
                }
                i += 1;
            }
        }
    }

    let Some(arch) = arch else {
        eprintln!("--arch is required");
        print_help();
        return ExitCode::FAILURE;
    };
    let Some(source_path) = source_path else {
        eprintln!("source file path is required");
        print_help();
        return ExitCode::FAILURE;
    };

    let kernel_name = name.unwrap_or_else(|| derive_name(&source_path));

    let source = match std::fs::read_to_string(&source_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read {}: {e}", source_path);
            return ExitCode::FAILURE;
        }
    };

    // extra_flags defaults to empty string (matches default runtime when no env is set).
    // Could also honor HIPFIRE_HIPCC_EXTRA_FLAGS from env if not explicitly passed,
    // but the packaging script is expected to produce default keys.
    let extra = extra_flags.unwrap_or_default();

    let hash = rdna_compute::KernelCompiler::packaging_hash_for(&arch, &kernel_name, &source, &extra);
    println!("{hash}");
    ExitCode::SUCCESS
}
