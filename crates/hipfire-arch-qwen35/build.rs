// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Runs when `--features npu-kernels` is set.  Compiles the SwiGLU, RMSNorm,
// and RoPE NPU kernels.  By default uses `--npu auto` to detect the installed
// NPU generation via pyxrt; set HIPFIRE_NPU_TARGETS to a comma-separated list
// (e.g. "npu1,npu2") to override.
// Soft-fails with a cargo warning if the MLIR-AIE toolchain is absent.
//
// Env vars consumed at build time:
//   HIPFIRE_NPU_HIDDEN_SIZES   — FFN intermediate sizes for SwiGLU (default: "8960")
//   HIPFIRE_NPU_RMSNORM_SIZES  — model hidden sizes for RMSNorm (default: "1536,3584")
//   HIPFIRE_NPU_ROPE_CONFIGS   — "n_heads:n_kv_heads:head_dim:n_rot" tuples,
//                                comma-separated (default: "8:2:256:64")
//   HIPFIRE_NPU_TARGETS        — comma-separated NPU targets: auto|npu1|npu2
//                                (default: "auto")
//   HIPFIRE_NPU_PYTHON         — Python interpreter to use (default: ~/.venv/bin/python
//                                falling back to python3)

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_NPU_KERNELS").is_err() {
        return;
    }

    let workspace = workspace_root();
    let build_script = workspace.join("tools/npu/build_qwen35_swiglu.py");
    let kernel_src = workspace.join("tools/npu/silu_mul_bf16.cc");

    println!("cargo:rerun-if-changed={}", build_script.display());
    println!("cargo:rerun-if-changed={}", kernel_src.display());
    println!("cargo:rerun-if-env-changed=HIPFIRE_NPU_HIDDEN_SIZES");
    println!("cargo:rerun-if-env-changed=HIPFIRE_NPU_TARGETS");
    println!("cargo:rerun-if-env-changed=HIPFIRE_NPU_PYTHON");

    if !build_script.exists() {
        println!(
            "cargo:warning=npu-kernels: build script not found at {} — skipping",
            build_script.display()
        );
        return;
    }

    let python = find_python();
    let out_dir = workspace.join("target/npu");

    let sizes: Vec<u32> = std::env::var("HIPFIRE_NPU_HIDDEN_SIZES")
        .unwrap_or_else(|_| "8960".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let targets: Vec<String> = std::env::var("HIPFIRE_NPU_TARGETS")
        .unwrap_or_else(|_| "auto".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    for npu in &targets {
        for &size in &sizes {
            run_build(&python, &build_script, &out_dir, npu, size);
        }
    }

    // ── RMSNorm kernel ────────────────────────────────────────────────────────
    let rmsnorm_script = workspace.join("tools/npu/build_qwen35_rmsnorm.py");
    let rmsnorm_src = workspace.join("tools/npu/rms_norm_weighted_bf16.cc");

    println!("cargo:rerun-if-changed={}", rmsnorm_script.display());
    println!("cargo:rerun-if-changed={}", rmsnorm_src.display());
    println!("cargo:rerun-if-env-changed=HIPFIRE_NPU_RMSNORM_SIZES");

    if !rmsnorm_script.exists() {
        println!(
            "cargo:warning=npu-kernels: rmsnorm build script not found at {} — skipping",
            rmsnorm_script.display()
        );
        return;
    }

    let rmsnorm_sizes: Vec<u32> = std::env::var("HIPFIRE_NPU_RMSNORM_SIZES")
        .unwrap_or_else(|_| "1536,3584".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    for npu in &targets {
        for &size in &rmsnorm_sizes {
            run_build(&python, &rmsnorm_script, &out_dir, npu, size);
        }
    }

    // ── RoPE kernel ───────────────────────────────────────────────────────────
    let rope_script = workspace.join("tools/npu/build_qwen35_rope.py");
    let rope_src = workspace.join("tools/npu/rope_rotate_bf16.cc");

    println!("cargo:rerun-if-changed={}", rope_script.display());
    println!("cargo:rerun-if-changed={}", rope_src.display());
    println!("cargo:rerun-if-env-changed=HIPFIRE_NPU_ROPE_CONFIGS");

    if !rope_script.exists() {
        println!(
            "cargo:warning=npu-kernels: rope build script not found at {} — skipping",
            rope_script.display()
        );
        return;
    }

    // Parse "n_heads:n_kv_heads:head_dim:n_rot" tuples.
    let rope_configs: Vec<(u32, u32, u32, u32)> = std::env::var("HIPFIRE_NPU_ROPE_CONFIGS")
        .unwrap_or_else(|_| "8:2:256:64".to_string())
        .split(',')
        .filter_map(|s| {
            let parts: Vec<u32> = s.trim().split(':').filter_map(|p| p.parse().ok()).collect();
            if parts.len() == 4 {
                Some((parts[0], parts[1], parts[2], parts[3]))
            } else {
                None
            }
        })
        .collect();

    for npu in &targets {
        for &(nh, nkv, hd, nr) in &rope_configs {
            run_rope_build(&python, &rope_script, &out_dir, npu, nh, nkv, hd, nr);
        }
    }

    // ── Head norm kernel ──────────────────────────────────────────────────────
    // HIPFIRE_NPU_HEADNORM_CONFIGS: same format as ROPE_CONFIGS but n_rot unused —
    // "n_heads:n_kv_heads:head_dim" (n_rot field ignored if present for compat).
    let headnorm_script = workspace.join("tools/npu/build_qwen35_headnorm.py");
    let headnorm_src = workspace.join("tools/npu/rms_norm_head_bf16.cc");

    println!("cargo:rerun-if-changed={}", headnorm_script.display());
    println!("cargo:rerun-if-changed={}", headnorm_src.display());
    println!("cargo:rerun-if-env-changed=HIPFIRE_NPU_HEADNORM_CONFIGS");

    if !headnorm_script.exists() {
        println!(
            "cargo:warning=npu-kernels: headnorm build script not found at {} — skipping",
            headnorm_script.display()
        );
        return;
    }

    // Parse "n_heads:n_kv_heads:head_dim" tuples (n_rot field ignored if present).
    let headnorm_configs: Vec<(u32, u32, u32)> = std::env::var("HIPFIRE_NPU_HEADNORM_CONFIGS")
        .unwrap_or_else(|_| "8:2:256".to_string())
        .split(',')
        .filter_map(|s| {
            let parts: Vec<u32> = s.trim().split(':').filter_map(|p| p.parse().ok()).collect();
            if parts.len() >= 3 {
                Some((parts[0], parts[1], parts[2]))
            } else {
                None
            }
        })
        .collect();

    for npu in &targets {
        for &(nh, nkv, hd) in &headnorm_configs {
            run_headnorm_build(&python, &headnorm_script, &out_dir, npu, nh, nkv, hd);
        }
    }

    // ── Attn output gate kernel ───────────────────────────────────────────────
    // HIPFIRE_NPU_ATTN_GATE_CONFIGS: "n_heads:head_dim" pairs (default "8:256").
    // Only needed when config.attn_output_gate=true.
    let attn_gate_script = workspace.join("tools/npu/build_qwen35_attn_gate.py");
    let attn_gate_src = workspace.join("tools/npu/sigmoid_mul_bf16.cc");

    println!("cargo:rerun-if-changed={}", attn_gate_script.display());
    println!("cargo:rerun-if-changed={}", attn_gate_src.display());
    println!("cargo:rerun-if-env-changed=HIPFIRE_NPU_ATTN_GATE_CONFIGS");

    if !attn_gate_script.exists() {
        println!(
            "cargo:warning=npu-kernels: attn-gate build script not found at {} — skipping",
            attn_gate_script.display()
        );
        return;
    }

    let attn_gate_configs: Vec<(u32, u32)> = std::env::var("HIPFIRE_NPU_ATTN_GATE_CONFIGS")
        .unwrap_or_else(|_| "8:256".to_string())
        .split(',')
        .filter_map(|s| {
            let parts: Vec<u32> = s.trim().split(':').filter_map(|p| p.parse().ok()).collect();
            if parts.len() >= 2 {
                Some((parts[0], parts[1]))
            } else {
                None
            }
        })
        .collect();

    for npu in &targets {
        for &(nh, hd) in &attn_gate_configs {
            run_attn_gate_build(&python, &attn_gate_script, &out_dir, npu, nh, hd);
        }
    }
}

fn run_build(python: &Path, script: &Path, out_dir: &Path, npu: &str, hidden_size: u32) {
    let result = Command::new(python)
        .arg(script)
        .arg("--hidden-size")
        .arg(hidden_size.to_string())
        .arg("--npu")
        .arg(npu)
        .arg("--out-dir")
        .arg(out_dir)
        .output();

    match result {
        Ok(out) if out.status.success() => {
            println!(
                "cargo:warning=npu-kernels: built {npu} hidden_size={hidden_size} → {}",
                out_dir.display()
            );
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            // Trim to first meaningful error line to keep cargo output readable
            let first_err = stderr
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("(no output)");
            println!(
                "cargo:warning=npu-kernels: {npu} hidden_size={hidden_size} failed \
                 (exit {}): {first_err}",
                out.status
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=npu-kernels: could not launch Python for {npu} \
                 hidden_size={hidden_size}: {e}"
            );
        }
    }
}

fn run_attn_gate_build(
    python: &Path,
    script: &Path,
    out_dir: &Path,
    npu: &str,
    n_heads: u32,
    head_dim: u32,
) {
    let result = Command::new(python)
        .arg(script)
        .arg("--n-heads")
        .arg(n_heads.to_string())
        .arg("--head-dim")
        .arg(head_dim.to_string())
        .arg("--npu")
        .arg(npu)
        .arg("--out-dir")
        .arg(out_dir)
        .output();

    match result {
        Ok(out) if out.status.success() => {
            println!(
                "cargo:warning=npu-kernels: attn-gate {npu} n_heads={n_heads} \
                 head_dim={head_dim} → {}",
                out_dir.display()
            );
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let first_err = stderr
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("(no output)");
            println!(
                "cargo:warning=npu-kernels: attn-gate {npu} n_heads={n_heads} \
                 head_dim={head_dim} failed (exit {}): {first_err}",
                out.status
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=npu-kernels: could not launch Python for attn-gate {npu}: {e}"
            );
        }
    }
}

fn run_headnorm_build(
    python: &Path,
    script: &Path,
    out_dir: &Path,
    npu: &str,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
) {
    let result = Command::new(python)
        .arg(script)
        .arg("--n-heads")
        .arg(n_heads.to_string())
        .arg("--n-kv-heads")
        .arg(n_kv_heads.to_string())
        .arg("--head-dim")
        .arg(head_dim.to_string())
        .arg("--npu")
        .arg(npu)
        .arg("--out-dir")
        .arg(out_dir)
        .output();

    match result {
        Ok(out) if out.status.success() => {
            println!(
                "cargo:warning=npu-kernels: headnorm {npu} n_heads={n_heads} \
                 n_kv_heads={n_kv_heads} head_dim={head_dim} → {}",
                out_dir.display()
            );
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let first_err = stderr
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("(no output)");
            println!(
                "cargo:warning=npu-kernels: headnorm {npu} n_heads={n_heads} \
                 n_kv_heads={n_kv_heads} head_dim={head_dim} \
                 failed (exit {}): {first_err}",
                out.status
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=npu-kernels: could not launch Python for headnorm {npu}: {e}"
            );
        }
    }
}

fn run_rope_build(
    python: &Path,
    script: &Path,
    out_dir: &Path,
    npu: &str,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_rot: u32,
) {
    let result = Command::new(python)
        .arg(script)
        .arg("--n-heads")
        .arg(n_heads.to_string())
        .arg("--n-kv-heads")
        .arg(n_kv_heads.to_string())
        .arg("--head-dim")
        .arg(head_dim.to_string())
        .arg("--n-rot")
        .arg(n_rot.to_string())
        .arg("--npu")
        .arg(npu)
        .arg("--out-dir")
        .arg(out_dir)
        .output();

    match result {
        Ok(out) if out.status.success() => {
            println!(
                "cargo:warning=npu-kernels: rope {npu} n_heads={n_heads} \
                 n_kv_heads={n_kv_heads} head_dim={head_dim} n_rot={n_rot} → {}",
                out_dir.display()
            );
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let first_err = stderr
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("(no output)");
            println!(
                "cargo:warning=npu-kernels: rope {npu} n_heads={n_heads} \
                 n_kv_heads={n_kv_heads} head_dim={head_dim} n_rot={n_rot} \
                 failed (exit {}): {first_err}",
                out.status
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=npu-kernels: could not launch Python for rope {npu}: {e}"
            );
        }
    }
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <workspace>/crates/hipfire-arch-qwen35
    Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // workspace root
        .to_path_buf()
}

fn find_python() -> PathBuf {
    if let Ok(p) = std::env::var("HIPFIRE_NPU_PYTHON") {
        return PathBuf::from(p);
    }
    let venv = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".venv/bin/python");
    if venv.exists() {
        return venv;
    }
    PathBuf::from("python3")
}
