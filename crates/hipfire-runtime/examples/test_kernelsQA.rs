// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QA mirror for the kernel harness.
//! Runs each kernel case in an isolated subprocess so hangs, panics, and leaks
//! do not collapse the rest of the sweep.

use rdna_compute::{DType, Gpu};
use std::env;
use std::process::{Command, ExitCode};
use std::thread;
use std::time::{Duration, Instant};

const SKIP_EXIT: u8 = 10;

struct CaseDef {
    name: &'static str,
    timeout: Duration,
}

const CASES: &[CaseDef] = &[
    CaseDef { name: "alloc_free", timeout: Duration::from_secs(20) },
    CaseDef { name: "upload_download_f32", timeout: Duration::from_secs(20) },
    CaseDef { name: "add_inplace_f32", timeout: Duration::from_secs(20) },
    CaseDef { name: "rmsnorm_f32", timeout: Duration::from_secs(20) },
    CaseDef { name: "softmax_f32", timeout: Duration::from_secs(20) },
    CaseDef { name: "attention_hd128_h8_kv2", timeout: Duration::from_secs(30) },
    CaseDef { name: "attention_hd256_h16_kv4", timeout: Duration::from_secs(30) },
    CaseDef { name: "attention_hd256_h10_kv2", timeout: Duration::from_secs(30) },
    CaseDef { name: "q8_write_attn_hd128_kv8", timeout: Duration::from_secs(30) },
    CaseDef { name: "q8_write_attn_hd256_kv4", timeout: Duration::from_secs(30) },
    CaseDef { name: "q8_write_attn_hd256_kv2", timeout: Duration::from_secs(30) },
    CaseDef { name: "gdn_q8_h32_hd128", timeout: Duration::from_secs(30) },
    CaseDef { name: "gdn_q8_h16_hd128", timeout: Duration::from_secs(30) },
    CaseDef { name: "vision_gemm_f16", timeout: Duration::from_secs(30) },
    CaseDef { name: "vision_layernorm_batched", timeout: Duration::from_secs(30) },
    CaseDef { name: "vision_transpose_f32", timeout: Duration::from_secs(30) },
];

enum CaseOutcome {
    Pass(String),
    Skip(String),
    Fail(String),
}

fn main() -> ExitCode {
    let mut case_name: Option<String> = None;
    let mut expected_arch: Option<String> = None;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--qa-case" => {
                i += 1;
                case_name = args.get(i).cloned();
            }
            "--expected-arch" => {
                i += 1;
                expected_arch = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(case) = case_name {
        return run_case(&case, expected_arch.as_deref());
    }

    supervisor(expected_arch.as_deref())
}

fn supervisor(expected_arch: Option<&str>) -> ExitCode {
    let exe = match env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("failed to resolve current executable: {err}");
            return ExitCode::from(1);
        }
    };

    let mut passed = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    eprintln!("=== hipfire kernel QA harness ===");
    if let Some(arch) = expected_arch {
        eprintln!("Expected arch: {arch}");
    }

    for case in CASES {
        eprintln!("\n--- {} ---", case.name);
        let mut cmd = Command::new(&exe);
        cmd.arg("--qa-case").arg(case.name);
        if let Some(arch) = expected_arch {
            cmd.arg("--expected-arch").arg(arch);
        }

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) => {
                failed += 1;
                eprintln!("spawn failed: {err}");
                continue;
            }
        };

        let start = Instant::now();
        let code = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status.code().unwrap_or(1),
                Ok(None) => {
                    if start.elapsed() > case.timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        break 124;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                Err(err) => {
                    eprintln!("wait failed: {err}");
                    break 1;
                }
            }
        };

        match code {
            0 => {
                passed += 1;
                eprintln!("SUPERVISOR PASS {} ({:.0}ms)", case.name, start.elapsed().as_secs_f64() * 1000.0);
            }
            x if x == SKIP_EXIT as i32 => {
                skipped += 1;
                eprintln!("SUPERVISOR SKIP {} ({:.0}ms)", case.name, start.elapsed().as_secs_f64() * 1000.0);
            }
            124 => {
                failed += 1;
                eprintln!("SUPERVISOR FAIL {} timed out after {:.1}s", case.name, case.timeout.as_secs_f64());
            }
            other => {
                failed += 1;
                eprintln!("SUPERVISOR FAIL {} rc={other}", case.name);
            }
        }
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  Passed:  {passed}");
    eprintln!("  Skipped: {skipped}");
    eprintln!("  Failed:  {failed}");

    if failed > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn run_case(case_name: &str, expected_arch: Option<&str>) -> ExitCode {
    let outcome = std::panic::catch_unwind(|| match case_name {
        "alloc_free" => alloc_free(expected_arch),
        "upload_download_f32" => upload_download_f32(expected_arch),
        "add_inplace_f32" => add_inplace_f32(expected_arch),
        "rmsnorm_f32" => rmsnorm_f32(expected_arch),
        "softmax_f32" => softmax_f32(expected_arch),
        "attention_hd128_h8_kv2" => attention_case(expected_arch, 8, 2, 128, 16),
        "attention_hd256_h16_kv4" => attention_case(expected_arch, 16, 4, 256, 16),
        "attention_hd256_h10_kv2" => attention_case(expected_arch, 10, 2, 256, 16),
        "q8_write_attn_hd128_kv8" => q8_kv_case(expected_arch, 8, 128),
        "q8_write_attn_hd256_kv4" => q8_kv_case(expected_arch, 4, 256),
        "q8_write_attn_hd256_kv2" => q8_kv_case(expected_arch, 2, 256),
        "gdn_q8_h32_hd128" => gdn_case(expected_arch, 32, 128),
        "gdn_q8_h16_hd128" => gdn_case(expected_arch, 16, 128),
        "vision_gemm_f16" => vision_gemm_f16(expected_arch),
        "vision_layernorm_batched" => vision_layernorm_batched(expected_arch),
        "vision_transpose_f32" => vision_transpose_f32(expected_arch),
        other => CaseOutcome::Fail(format!("unknown case: {other}")),
    });

    match outcome {
        Ok(CaseOutcome::Pass(msg)) => {
            eprintln!("QA PASS {case_name}: {msg}");
            ExitCode::SUCCESS
        }
        Ok(CaseOutcome::Skip(msg)) => {
            eprintln!("QA SKIP {case_name}: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Ok(CaseOutcome::Fail(msg)) => {
            eprintln!("QA FAIL {case_name}: {msg}");
            ExitCode::from(1)
        }
        Err(_) => {
            eprintln!("QA FAIL {case_name}: panic");
            ExitCode::from(1)
        }
    }
}

fn init_gpu(expected_arch: Option<&str>) -> Result<Gpu, CaseOutcome> {
    let gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(err) => {
            return Err(CaseOutcome::Skip(format!("GPU init unavailable: {err}")));
        }
    };

    if let Some(expected) = expected_arch {
        if gpu.arch != expected {
            return Err(CaseOutcome::Fail(format!(
                "expected arch {expected}, got {}",
                gpu.arch
            )));
        }
    }

    Ok(gpu)
}

fn ensure(cond: bool, msg: impl Into<String>) -> Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(msg.into())
    }
}

fn alloc_free(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let tensor = gpu.alloc_tensor(&[1024], DType::F32).map_err(|e| e.to_string())?;
        gpu.free_tensor(tensor).map_err(|e| e.to_string())?;
        Ok("allocated and released tensor".to_string())
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn upload_download_f32(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let data = vec![1.0f32; 256];
        let tensor = gpu.upload_f32(&data, &[256]).map_err(|e| e.to_string())?;
        let back = gpu.download_f32(&tensor).map_err(|e| e.to_string())?;
        ensure(back.len() == 256, format!("expected 256 values, got {}", back.len()))?;
        ensure((back[0] - 1.0).abs() < 1e-6, format!("expected first value 1.0, got {}", back[0]))?;
        gpu.free_tensor(tensor).map_err(|e| e.to_string())?;
        Ok(format!("round-tripped {} values", back.len()))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn add_inplace_f32(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let a = gpu.upload_f32(&vec![1.0f32; 64], &[64]).map_err(|e| e.to_string())?;
        let b = gpu.upload_f32(&vec![2.0f32; 64], &[64]).map_err(|e| e.to_string())?;
        gpu.add_inplace_f32(&a, &b).map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&a).map_err(|e| e.to_string())?;
        ensure((r[0] - 3.0).abs() < 1e-6, format!("expected 3.0, got {}", r[0]))?;
        gpu.free_tensor(a).map_err(|e| e.to_string())?;
        gpu.free_tensor(b).map_err(|e| e.to_string())?;
        Ok(format!("result[0]={:.3}", r[0]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn rmsnorm_f32(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let x = gpu.upload_f32(&vec![1.0f32; 128], &[128]).map_err(|e| e.to_string())?;
        let w = gpu.upload_f32(&vec![1.0f32; 128], &[128]).map_err(|e| e.to_string())?;
        let o = gpu.alloc_tensor(&[128], DType::F32).map_err(|e| e.to_string())?;
        gpu.rmsnorm_f32(&x, &w, &o, 1e-6).map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&o).map_err(|e| e.to_string())?;
        ensure(r[0].is_finite(), format!("rmsnorm produced non-finite value {}", r[0]))?;
        gpu.free_tensor(x).map_err(|e| e.to_string())?;
        gpu.free_tensor(w).map_err(|e| e.to_string())?;
        gpu.free_tensor(o).map_err(|e| e.to_string())?;
        Ok(format!("result[0]={:.4}", r[0]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn softmax_f32(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let x = gpu.upload_f32(&vec![1.0f32; 32], &[1, 32]).map_err(|e| e.to_string())?;
        gpu.softmax_f32(&x).map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&x).map_err(|e| e.to_string())?;
        let sum: f32 = r.iter().sum();
        ensure((sum - 1.0).abs() < 0.01, format!("softmax sum={sum}"))?;
        gpu.free_tensor(x).map_err(|e| e.to_string())?;
        Ok(format!("sum={sum:.4}"))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn attention_case(expected_arch: Option<&str>, n_heads: usize, n_kv: usize, hd: usize, seq: usize) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let q = gpu.upload_f32(&vec![0.1f32; n_heads * hd], &[n_heads * hd]).map_err(|e| e.to_string())?;
        let kv_dim = n_kv * hd;
        let k = gpu.upload_f32(&vec![0.1f32; seq * kv_dim], &[seq * kv_dim]).map_err(|e| e.to_string())?;
        let v = gpu.upload_f32(&vec![0.1f32; seq * kv_dim], &[seq * kv_dim]).map_err(|e| e.to_string())?;
        let o = gpu.alloc_tensor(&[n_heads * hd], DType::F32).map_err(|e| e.to_string())?;
        let pos_buf = gpu.hip.malloc(4).map_err(|e| e.to_string())?;
        let pos_val = (seq - 1) as i32;
        gpu.hip.memcpy_htod(&pos_buf, &pos_val.to_ne_bytes()).map_err(|e| e.to_string())?;
        gpu.attention_f32(&q, &k, &v, &o, &pos_buf, seq, n_heads, n_kv, hd, seq)
            .map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&o).map_err(|e| e.to_string())?;
        ensure(r[0].is_finite(), format!("attention produced non-finite value {}", r[0]))?;
        gpu.hip.free(pos_buf).map_err(|e| e.to_string())?;
        gpu.free_tensor(q).map_err(|e| e.to_string())?;
        gpu.free_tensor(k).map_err(|e| e.to_string())?;
        gpu.free_tensor(v).map_err(|e| e.to_string())?;
        gpu.free_tensor(o).map_err(|e| e.to_string())?;
        Ok(format!("output[0]={:.4}", r[0]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn q8_kv_case(expected_arch: Option<&str>, n_kv: usize, hd: usize) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let n_heads = n_kv * 4;
        let seq = 8usize;
        let q8_blocks = hd / 32;
        let q8_bytes_per_pos = n_kv * q8_blocks * 34;
        let cache_bytes = seq * q8_bytes_per_pos;
        let cache_elems = (cache_bytes + 3) / 4;
        let k_cache = gpu.zeros(&[cache_elems], DType::F32).map_err(|e| e.to_string())?;
        let v_cache = gpu.zeros(&[cache_elems], DType::F32).map_err(|e| e.to_string())?;
        let pos_buf = gpu.hip.malloc(4).map_err(|e| e.to_string())?;

        for p in 0..4 {
            let kv_data = gpu.upload_f32(&vec![0.1f32; n_kv * hd], &[n_kv * hd]).map_err(|e| e.to_string())?;
            let pv = p as i32;
            gpu.hip.memcpy_htod(&pos_buf, &pv.to_ne_bytes()).map_err(|e| e.to_string())?;
            gpu.kv_cache_write_q8_0(&k_cache, &kv_data, &pos_buf, n_kv, hd).map_err(|e| e.to_string())?;
            gpu.kv_cache_write_q8_0(&v_cache, &kv_data, &pos_buf, n_kv, hd).map_err(|e| e.to_string())?;
            gpu.free_tensor(kv_data).map_err(|e| e.to_string())?;
        }

        let q = gpu.upload_f32(&vec![0.1f32; n_heads * hd], &[n_heads * hd]).map_err(|e| e.to_string())?;
        let o = gpu.alloc_tensor(&[n_heads * hd], DType::F32).map_err(|e| e.to_string())?;
        let pv = 3i32;
        gpu.hip.memcpy_htod(&pos_buf, &pv.to_ne_bytes()).map_err(|e| e.to_string())?;
        gpu.attention_q8_0_kv(&q, &k_cache, &v_cache, &o, &pos_buf, 4, n_heads, n_kv, hd, seq)
            .map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&o).map_err(|e| e.to_string())?;
        ensure(r[0].is_finite(), format!("q8 attention produced non-finite value {}", r[0]))?;
        gpu.hip.free(pos_buf).map_err(|e| e.to_string())?;
        gpu.free_tensor(q).map_err(|e| e.to_string())?;
        gpu.free_tensor(o).map_err(|e| e.to_string())?;
        gpu.free_tensor(k_cache).map_err(|e| e.to_string())?;
        gpu.free_tensor(v_cache).map_err(|e| e.to_string())?;
        Ok(format!("output[0]={:.4}", r[0]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(feature = "deltanet")]
fn gdn_case(expected_arch: Option<&str>, n_heads: usize, hd: usize) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let s_size = n_heads * hd * hd;
        let scale_size = n_heads * hd;
        let s_q8 = gpu.zeros(&[s_size], DType::F32).map_err(|e| e.to_string())?;
        let s_scales = gpu.upload_f32(&vec![1.0f32; scale_size], &[scale_size]).map_err(|e| e.to_string())?;
        let q = gpu.upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd]).map_err(|e| e.to_string())?;
        let k = gpu.upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd]).map_err(|e| e.to_string())?;
        let v = gpu.upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd]).map_err(|e| e.to_string())?;
        let alpha = gpu.upload_f32(&vec![0.5f32; n_heads], &[n_heads]).map_err(|e| e.to_string())?;
        let beta = gpu.upload_f32(&vec![0.5f32; n_heads], &[n_heads]).map_err(|e| e.to_string())?;
        let o = gpu.alloc_tensor(&[n_heads * hd], DType::F32).map_err(|e| e.to_string())?;
        gpu.gated_delta_net_q8(&q, &k, &v, &alpha, &beta, &s_q8, &s_scales, &o, 1, n_heads, hd, None)
            .map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&o).map_err(|e| e.to_string())?;
        ensure(r[0].is_finite(), format!("gdn produced non-finite value {}", r[0]))?;
        for tensor in [q, k, v, alpha, beta, s_q8, s_scales, o] {
            gpu.free_tensor(tensor).map_err(|e| e.to_string())?;
        }
        Ok(format!("output[0]={:.4}", r[0]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

#[cfg(not(feature = "deltanet"))]
fn gdn_case(_: Option<&str>, _: usize, _: usize) -> CaseOutcome {
    CaseOutcome::Skip("gdn cases require --features deltanet".to_string())
}

fn vision_gemm_f16(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let (m, k, n) = (32usize, 64usize, 8usize);
        let w_data = vec![0u8; m * k * 2];
        let w = gpu.upload_raw(&w_data, &[w_data.len()]).map_err(|e| e.to_string())?;
        let x = gpu.upload_f32(&vec![1.0f32; n * k], &[n * k]).map_err(|e| e.to_string())?;
        let y = gpu.alloc_tensor(&[m * n], DType::F32).map_err(|e| e.to_string())?;
        gpu.gemm_f16(&w, &x, &y, m, k, n).map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&y).map_err(|e| e.to_string())?;
        ensure(r.len() == m * n, format!("expected {} outputs, got {}", m * n, r.len()))?;
        ensure(r[0].is_finite(), format!("gemm produced non-finite value {}", r[0]))?;
        gpu.free_tensor(w).map_err(|e| e.to_string())?;
        gpu.free_tensor(x).map_err(|e| e.to_string())?;
        gpu.free_tensor(y).map_err(|e| e.to_string())?;
        Ok(format!("output_len={} first={:.4}", r.len(), r[0]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn vision_layernorm_batched(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let batch = 4usize;
        let dim = 64usize;
        let x = gpu.upload_f32(&vec![1.0f32; batch * dim], &[batch * dim]).map_err(|e| e.to_string())?;
        let w = gpu.upload_f32(&vec![1.0f32; dim], &[dim]).map_err(|e| e.to_string())?;
        let b = gpu.upload_f32(&vec![0.0f32; dim], &[dim]).map_err(|e| e.to_string())?;
        let o = gpu.alloc_tensor(&[batch * dim], DType::F32).map_err(|e| e.to_string())?;
        gpu.layernorm_batched(&x, &w, &b, &o, batch, dim, 1e-6).map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&o).map_err(|e| e.to_string())?;
        ensure(r[0].is_finite(), format!("layernorm produced non-finite value {}", r[0]))?;
        gpu.free_tensor(x).map_err(|e| e.to_string())?;
        gpu.free_tensor(w).map_err(|e| e.to_string())?;
        gpu.free_tensor(b).map_err(|e| e.to_string())?;
        gpu.free_tensor(o).map_err(|e| e.to_string())?;
        Ok(format!("output[0]={:.4}", r[0]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}

fn vision_transpose_f32(expected_arch: Option<&str>) -> CaseOutcome {
    let mut gpu = match init_gpu(expected_arch) {
        Ok(gpu) => gpu,
        Err(outcome) => return outcome,
    };

    match (|| -> Result<String, String> {
        let rows = 4usize;
        let cols = 8usize;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let src = gpu.upload_f32(&data, &[rows * cols]).map_err(|e| e.to_string())?;
        let dst = gpu.alloc_tensor(&[rows * cols], DType::F32).map_err(|e| e.to_string())?;
        gpu.transpose_f32(&src, &dst, rows, cols).map_err(|e| e.to_string())?;
        let r = gpu.download_f32(&dst).map_err(|e| e.to_string())?;
        ensure((r[1] - 8.0).abs() < 0.01, format!("expected r[1]=8.0, got {}", r[1]))?;
        gpu.free_tensor(src).map_err(|e| e.to_string())?;
        gpu.free_tensor(dst).map_err(|e| e.to_string())?;
        Ok(format!("r[0]={:.1} r[1]={:.1}", r[0], r[1]))
    })() {
        Ok(msg) => CaseOutcome::Pass(msg),
        Err(err) => CaseOutcome::Fail(err),
    }
}
