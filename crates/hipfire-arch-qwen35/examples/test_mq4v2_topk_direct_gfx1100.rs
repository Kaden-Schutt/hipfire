// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! S8 gate: fused MQ4V2 LM-head + greedy top-K/log-sum-exp direct kernel
//! (`mq4v2_lmhead_topk_direct_gfx1100`) parity + timing on exact gfx1100.
//!
//! Loads the REAL lm_head (M=248320, K=5120, qt=44) from
//! qwen3.8-27b.mq4v2.xt.hfq. For each shape (B, K_TOP):
//!
//!   1. Baseline arm: `gemm_mq4g256v2_batched_lmhead` (default flags → the
//!      ks4 split-K LDS tier at K=5120) + `topk_logsumexp_batched_f32`,
//!      exactly the pre-change DDTree proposal path.
//!   2. Candidate arm: one `mq4v2_lmhead_topk_direct_gfx1100` launch.
//!
//! Gates:
//!   - Selected ids and their ORDER are exactly equal to the baseline on
//!     random inputs (distinct-valued top-K; the kernel's tile arithmetic
//!     replicates ks4's fixed-order split-K sum so logits are bit-identical).
//!   - Adversarial exact-tie inputs (X = 0, X = e0 basis): each arm is
//!     compared against a host simulation of ITS OWN partition/merge order.
//!     Ties crossing the baseline's 256-way thread partition are documented
//!     to order differently (uniform row: baseline [0..K), candidate
//!     [0,16,32,…]); distinct values are always exactly equal.
//!   - Log-probs, f64 floor (pattern of test_mq4v2_residual_ksplit_gfx1100):
//!     truth = f64 logsumexp over the baseline logits;
//!     relL2(cand,f64) <= max(5e-6, 1.10 * relL2(base,f64)) and
//!     maxAbs(cand,f64) <= max(5e-5, 1.10 * maxAbs(base,f64)).
//!   - Timing at the production shape (B=15, K=8): 32 warmups, 3×200
//!     interleaved launches. Fused must be >= 1.0 ms/launch faster than
//!     gemm+topk, and the selection tail (fused med − gemm-only med) must be
//!     <= 121 µs (10× the measured 1.21 ms topk ÷ 10 … i.e. 0.121 ms).
//!
//! On any other arch the harness SKIPs cleanly (exit 0, no GPU work).
//! Kill switch honored: HIPFIRE_DDTREE_TOPK_DIRECT_OFF=1 makes the fused
//! launcher refuse, so the harness asserts the flag is unset up front.

use rdna_compute::{DType, Gpu};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const MODEL_DEFAULT: &str = "/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.hfq";
const WARMUP: usize = 32;
const LAUNCHES: usize = 200;
const SAMPLES: usize = 3;

// ── minimal HFQ index parse (mirrors test_mq4v2_residual_ksplit_gfx1100) ────

struct HfqTensor {
    name: String,
    shape: Vec<u32>,
    data_off: usize,
    data_len: usize,
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

fn parse_hfq_index(path: &std::path::Path) -> (String, Vec<HfqTensor>) {
    let canon = std::fs::canonicalize(path)
        .unwrap_or_else(|e| panic!("canonicalize {}: {e}", path.display()));
    let mut f = File::open(&canon).expect("open hfq");
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr).expect("read hfq header");
    assert_eq!(&hdr[0..4], b"HFQM", "not an HFQ container");
    let n_tensors = u32le(&hdr[12..16]) as usize;
    let metadata_offset = u64le(&hdr[16..24]) as usize;
    let data_offset = u64le(&hdr[24..32]) as usize;
    let region_len = data_offset - metadata_offset;
    let mut region = vec![0u8; region_len];
    f.seek(SeekFrom::Start(metadata_offset as u64)).unwrap();
    f.read_exact(&mut region).expect("read hfq meta+index");
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let mut json_end = 0usize;
    for (i, &b) in region.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        if b == b'\\' && in_str {
            esc = true;
            continue;
        }
        if b == b'"' {
            in_str = !in_str;
            continue;
        }
        if !in_str {
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
    }
    assert!(json_end > 0, "metadata JSON not brace-terminated");
    let mut pos = json_end;
    let idx_n = u32le(&region[pos..pos + 4]) as usize;
    assert_eq!(idx_n, n_tensors, "index count != header count");
    pos += 4;
    let mut tensors = Vec::with_capacity(idx_n);
    let mut cum = data_offset;
    for _ in 0..idx_n {
        let nl = u16::from_le_bytes([region[pos], region[pos + 1]]) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&region[pos..pos + nl]).to_string();
        pos += nl + 1; // qt
        let nd = region[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd {
            shape.push(u32le(&region[pos..pos + 4]));
            pos += 4;
        }
        pos += 4; // group_size
        let data_len = u64le(&region[pos..pos + 8]) as usize;
        pos += 8;
        tensors.push(HfqTensor {
            name,
            shape,
            data_off: cum,
            data_len,
        });
        cum += data_len;
    }
    (canon.display().to_string(), tensors)
}

fn read_tensor_bytes(path: &str, t: &HfqTensor) -> Vec<u8> {
    let mut f = File::open(path).expect("reopen hfq for payload");
    f.seek(SeekFrom::Start(t.data_off as u64)).unwrap();
    let mut buf = vec![0u8; t.data_len];
    f.read_exact(&mut buf).expect("read tensor payload");
    buf
}

// ── host numerics helpers ────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_f32(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut st = seed | 1;
    (0..n)
        .map(|_| {
            let r = (xorshift64(&mut st) >> 11) as f64 / (u64::MAX >> 11) as f64;
            (lo + (r as f32) * (hi - lo)).clamp(lo, hi)
        })
        .collect()
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().unwrap();
}

fn htod_f32(gpu: &Gpu, t: &rdna_compute::GpuTensor, v: &[f32]) {
    gpu.hip
        .memcpy_htod(&t.buf, unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        })
        .expect("htod f32");
    sync(gpu);
}

fn rel_l2_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        num += d * d;
        den += y * y;
    }
    if den == 0.0 {
        if num == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (num / den).sqrt()
    }
}

fn max_abs_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f64, f64::max)
}

/// f64 truth for one column's top-K log-probs, computed over the baseline's
/// f32 logits (the shared input both arms' selection reads): exact top-K
/// selection by value with ascending-index tie-break (truth ordering is only
/// consumed for values whose selection both arms already agree on), and
/// log_z = max + ln(Σ exp(logit − max)) accumulated in f64.
fn truth_logp(logits: &[f32], vocab: usize, ids: &[i32], k: usize) -> Vec<f64> {
    let b = ids.len() / k;
    let mut out = Vec::with_capacity(b * k);
    for c in 0..b {
        let row = &logits[c * vocab..(c + 1) * vocab];
        let mut mx = f64::NEG_INFINITY;
        for &v in row {
            mx = mx.max(v as f64);
        }
        let mut se = 0f64;
        for &v in row {
            se += ((v as f64) - mx).exp();
        }
        let log_z = mx + se.ln();
        for j in 0..k {
            out.push(row[ids[c * k + j] as usize] as f64 - log_z);
        }
    }
    out
}

/// Host simulation of the baseline `topk_logsumexp_batched_f32` selection:
/// 256 threads, strided scan, per-thread replacement top-K, then thread-0
/// serial merge (t-major, j-minor, strict > insertion).
fn sim_baseline_ids(logits: &[f32], vocab: usize, b: usize, k: usize) -> Vec<i32> {
    const NTH: usize = 256;
    let mut out = vec![0i32; b * k];
    for c in 0..b {
        let row = &logits[c * vocab..(c + 1) * vocab];
        let mut tv = vec![vec![f32::NEG_INFINITY; k]; NTH];
        let mut ti = vec![vec![0i32; k]; NTH];
        for t in 0..NTH {
            let mut i = t;
            while i < vocab {
                let v = row[i];
                let (mut mj, mut mv) = (0usize, tv[t][0]);
                for j in 1..k {
                    if tv[t][j] < mv {
                        mv = tv[t][j];
                        mj = j;
                    }
                }
                if v > mv {
                    tv[t][mj] = v;
                    ti[t][mj] = i as i32;
                }
                i += NTH;
            }
        }
        let mut fv = vec![f32::NEG_INFINITY; k];
        let mut fi = vec![0i32; k];
        for t in 0..NTH {
            for j in 0..k {
                let v = tv[t][j];
                let i = ti[t][j];
                let mut ins = k;
                for q in (0..k).rev() {
                    if v > fv[q] {
                        ins = q;
                    }
                }
                if ins < k {
                    for q in (ins + 1..k).rev() {
                        fv[q] = fv[q - 1];
                        fi[q] = fi[q - 1];
                    }
                    fv[ins] = v;
                    fi[ins] = i;
                }
            }
        }
        out[c * k..(c + 1) * k].copy_from_slice(&fi);
    }
    out
}

/// Host simulation of the candidate kernel's selection: nwg single-wave
/// workgroups take strided 16-row tiles; lane c keeps a running replacement
/// top-K over rows tile*16 + 2j + (c>>4); thread 0 merges per-wg lists in
/// ascending wg order with strict-> insertion.
fn sim_candidate_ids(logits: &[f32], vocab: usize, b: usize, k: usize) -> Vec<i32> {
    let m_tiles = vocab.div_ceil(16);
    let nwg = m_tiles.min(rdna_compute::mq4v2_topk_direct::TDK_WG_MAX);
    let mut out = vec![0i32; b * k];
    for c in 0..b {
        let row = &logits[c * vocab..(c + 1) * vocab];
        let mut lists: Vec<Vec<(f32, i32)>> = Vec::with_capacity(nwg);
        for w in 0..nwg {
            let mut lv = vec![f32::NEG_INFINITY; k];
            let mut li = vec![0i32; k];
            let mut tile = w;
            while tile < m_tiles {
                let row0 = tile * 16;
                for j in 0..8 {
                    let r = row0 + 2 * j + (c >> 4);
                    if r >= vocab {
                        continue;
                    }
                    let v = row[r];
                    let (mut mj, mut mv) = (0usize, lv[0]);
                    for q in 1..k {
                        if lv[q] < mv {
                            mv = lv[q];
                            mj = q;
                        }
                    }
                    if v > mv {
                        lv[mj] = v;
                        li[mj] = r as i32;
                    }
                }
                tile += nwg;
            }
            lists.push(lv.into_iter().zip(li).collect());
        }
        let mut fv = vec![f32::NEG_INFINITY; k];
        let mut fi = vec![0i32; k];
        for lst in &lists {
            for &(v, i) in lst {
                let mut ins = k;
                for q in (0..k).rev() {
                    if v > fv[q] {
                        ins = q;
                    }
                }
                if ins < k {
                    for q in (ins + 1..k).rev() {
                        fv[q] = fv[q - 1];
                        fi[q] = fi[q - 1];
                    }
                    fv[ins] = v;
                    fi[ins] = i;
                }
            }
        }
        out[c * k..(c + 1) * k].copy_from_slice(&fi);
    }
    out
}

// ── timing ───────────────────────────────────────────────────────────────────

fn time_batch(gpu: &mut Gpu, launch: &mut dyn FnMut(&mut Gpu)) -> f64 {
    sync(gpu);
    let t0 = std::time::Instant::now();
    for _ in 0..LAUNCHES {
        launch(gpu);
    }
    sync(gpu);
    t0.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ── main ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    if !(gpu.arch_caps.is_gfx1100() && gpu.arch == "gfx1100") {
        eprintln!("SKIP: arch {} is not exact gfx1100", gpu.arch);
        return;
    }
    if gpu.flags.ddtree_topk_direct_off {
        eprintln!("SKIP: HIPFIRE_DDTREE_TOPK_DIRECT_OFF=1 — fused route disabled");
        return;
    }
    eprintln!("arch gfx1100 confirmed — running mq4v2 topk-direct parity + timing");

    let model_path = std::path::PathBuf::from(
        std::env::args().nth(1).as_deref().unwrap_or(MODEL_DEFAULT),
    );
    let (canon, tensors) = parse_hfq_index(&model_path);
    let t = tensors
        .iter()
        .find(|t| t.name.ends_with("lm_head.weight"))
        .expect("lm_head.weight not found");
    let m = t.shape[0] as usize;
    let k_dim = t.shape[1] as usize;
    assert_eq!((m, k_dim), (248320, 5120), "unexpected lm_head shape");
    assert_eq!(t.data_len, m * (k_dim / 256) * 136, "lm_head size mismatch");
    eprintln!("model: {canon}  lm_head M={m} K={k_dim} bytes={}", t.data_len);
    let payload = read_tensor_bytes(&canon, t);
    let d_a = gpu.upload_raw(&payload, &[m, k_dim]).expect("upload lm_head");

    // Persistent scratch (same layout as DdtreeTopkScratch, self-contained).
    let partials = gpu
        .alloc_tensor(
            &[rdna_compute::mq4v2_topk_direct::ddtree_topk_partials_bytes()],
            DType::Raw,
        )
        .expect("alloc partials");
    let ctl = gpu.alloc_tensor(&[8], DType::Raw).expect("alloc ctl");
    gpu.hip.memset(&ctl.buf, 0, 8).expect("zero ctl");

    let mut all_ok = true;

    // ── Parity shapes ────────────────────────────────────────────────────────
    for &(b, ktop) in &[(2usize, 8usize), (8, 8), (15, 8), (16, 8), (15, 4), (15, 1)] {
        // Random X with distinct-valued logits (tie probability ~0).
        let x_host = random_f32(b * k_dim, 0x51CE_5EED + b as u64, -1.0, 1.0);
        let d_x = gpu.alloc_tensor(&[b * k_dim], DType::F32).expect("alloc x");
        htod_f32(&gpu, &d_x, &x_host);

        // Baseline: logits GEMM + generic topk.
        let d_logits = gpu.alloc_tensor(&[b * m], DType::F32).expect("alloc logits");
        let d_bidx = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc bidx");
        let d_bval = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc bval");
        gpu.gemm_mq4g256v2_batched_lmhead(&d_a, &d_x, &d_logits, m, k_dim, b)
            .expect("baseline gemm");
        gpu.topk_logsumexp_batched_f32(&d_logits, &d_bidx, &d_bval, m, ktop, b)
            .expect("baseline topk");
        sync(&gpu);
        let logits_h = gpu.download_f32(&d_logits).expect("download logits");
        let bidx_h = gpu.download_f32(&d_bidx).expect("download bidx");
        let bval_h = gpu.download_f32(&d_bval).expect("download bval");
        let base_ids: Vec<i32> = bidx_h.iter().map(|f| f.to_bits() as i32).collect();

        // Candidate: fused kernel.
        let d_cidx = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc cidx");
        let d_cval = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc cval");
        gpu.mq4v2_lmhead_topk_direct_gfx1100(
            &d_a, &d_x, &partials, &ctl, &d_cidx, &d_cval, m, k_dim, b, ktop,
        )
        .expect("fused launch");
        sync(&gpu);
        let cidx_h = gpu.download_f32(&d_cidx).expect("download cidx");
        let cval_h = gpu.download_f32(&d_cval).expect("download cval");
        let cand_ids: Vec<i32> = cidx_h.iter().map(|f| f.to_bits() as i32).collect();

        // IDs: exact vs baseline; each arm vs its own partition simulation.
        let sim_b = sim_baseline_ids(&logits_h, m, b, ktop);
        let sim_c = sim_candidate_ids(&logits_h, m, b, ktop);
        let ids_ok = cand_ids == base_ids;
        let sim_ok = base_ids == sim_b && cand_ids == sim_c;
        if !ids_ok {
            all_ok = false;
            let bad = cand_ids
                .iter()
                .zip(base_ids.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            eprintln!("  FAIL ids B={b} K={ktop}: first diff at flat {bad}: cand={} base={}",
                cand_ids[bad], base_ids[bad]);
        }
        if !sim_ok {
            all_ok = false;
            eprintln!("  FAIL sim B={b} K={ktop}: baseline==sim_b {} candidate==sim_c {}",
                base_ids == sim_b, cand_ids == sim_c);
        }

        // Log-probs vs f64 floor.
        let truth = truth_logp(&logits_h, m, &base_ids, ktop);
        let base64: Vec<f64> = bval_h.iter().map(|&v| v as f64).collect();
        let cand64: Vec<f64> = cval_h.iter().map(|&v| v as f64).collect();
        let r_base = rel_l2_f64(&base64, &truth);
        let r_cand = rel_l2_f64(&cand64, &truth);
        let ma_base = max_abs_f64(&base64, &truth);
        let ma_cand = max_abs_f64(&cand64, &truth);
        let rl2_ok = r_cand <= 5e-6f64.max(1.10 * r_base);
        let mabs_ok = ma_cand <= 5e-5f64.max(1.10 * ma_base);
        if !(rl2_ok && mabs_ok) {
            all_ok = false;
            eprintln!("  FAIL logp B={b} K={ktop}: relL2 cand={r_cand:.3e} base={r_base:.3e} maxAbs cand={ma_cand:.3e} base={ma_base:.3e}");
        }
        println!(
            "B={b:>2} K={ktop}  ids_exact={ids_ok} sim_match={sim_ok} relL2(c,f64)={r_cand:.3e} relL2(b,f64)={r_base:.3e} maxAbs(c)={ma_cand:.3e} maxAbs(b)={ma_base:.3e} [{}]",
            if ids_ok && sim_ok && rl2_ok && mabs_ok { "OK" } else { "FAIL" }
        );

        let _ = gpu.free_tensor(d_x);
        let _ = gpu.free_tensor(d_logits);
        let _ = gpu.free_tensor(d_bidx);
        let _ = gpu.free_tensor(d_bval);
        let _ = gpu.free_tensor(d_cidx);
        let _ = gpu.free_tensor(d_cval);
    }

    // ── Adversarial exact-tie shapes ─────────────────────────────────────────
    // X = 0: every logit exactly 0. Baseline partition yields ids [0..K);
    // candidate partition yields [0,16,32,…] — documented divergence on
    // partition-crossing strict ties. Both must match their own simulations.
    let tie_cases: Vec<(Vec<f32>, &str)> = vec![
        (vec![0.0f32; 15 * k_dim], "X=0 uniform"),
        ({
            let mut v = vec![0.0f32; 15 * k_dim];
            for c in 0..15 {
                v[c * k_dim] = 1.0; // e0 basis: logits = W[:,0] — 4-bit levels, heavy ties
            }
            v
        }, "X=e0 weight-level ties"),
    ];
    for (x_host, label) in &tie_cases {
        let b = 15usize;
        let ktop = 8usize;
        let d_x = gpu.alloc_tensor(&[b * k_dim], DType::F32).expect("alloc x");
        htod_f32(&gpu, &d_x, x_host);
        let d_logits = gpu.alloc_tensor(&[b * m], DType::F32).expect("alloc logits");
        let d_bidx = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc bidx");
        let d_bval = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc bval");
        gpu.gemm_mq4g256v2_batched_lmhead(&d_a, &d_x, &d_logits, m, k_dim, b)
            .expect("baseline gemm");
        gpu.topk_logsumexp_batched_f32(&d_logits, &d_bidx, &d_bval, m, ktop, b)
            .expect("baseline topk");
        sync(&gpu);
        let logits_h = gpu.download_f32(&d_logits).expect("download logits");
        let bidx_h = gpu.download_f32(&d_bidx).expect("download bidx");
        let base_ids: Vec<i32> = bidx_h.iter().map(|f| f.to_bits() as i32).collect();

        let d_cidx = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc cidx");
        let d_cval = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc cval");
        gpu.mq4v2_lmhead_topk_direct_gfx1100(
            &d_a, &d_x, &partials, &ctl, &d_cidx, &d_cval, m, k_dim, b, ktop,
        )
        .expect("fused launch");
        sync(&gpu);
        let cidx_h = gpu.download_f32(&d_cidx).expect("download cidx");
        let cand_ids: Vec<i32> = cidx_h.iter().map(|f| f.to_bits() as i32).collect();

        let sim_b = sim_baseline_ids(&logits_h, m, b, ktop);
        let sim_c = sim_candidate_ids(&logits_h, m, b, ktop);
        let ok = base_ids == sim_b && cand_ids == sim_c;
        if !ok {
            all_ok = false;
            eprintln!(
                "  FAIL ties {label}: baseline==sim_b {} candidate==sim_c {}\n    base {:?}\n    simb {:?}\n    cand {:?}\n    simc {:?}",
                base_ids == sim_b,
                cand_ids == sim_c,
                &base_ids[..8],
                &sim_b[..8],
                &cand_ids[..8],
                &sim_c[..8]
            );
        }
        println!("ties {label:>24}: baseline==sim {} candidate==sim {} [{}]",
            base_ids == sim_b, cand_ids == sim_c, if ok { "OK" } else { "FAIL" });
        let _ = gpu.free_tensor(d_x);
        let _ = gpu.free_tensor(d_logits);
        let _ = gpu.free_tensor(d_bidx);
        let _ = gpu.free_tensor(d_bval);
        let _ = gpu.free_tensor(d_cidx);
        let _ = gpu.free_tensor(d_cval);
    }

    // ── Timing at the production shape (B=15, K=8) ──────────────────────────
    let (b, ktop) = (15usize, 8usize);
    let x_host = random_f32(b * k_dim, 0xBEEF_0042, -1.0, 1.0);
    let d_x = gpu.alloc_tensor(&[b * k_dim], DType::F32).expect("alloc x");
    htod_f32(&gpu, &d_x, &x_host);
    let d_logits = gpu.alloc_tensor(&[b * m], DType::F32).expect("alloc logits");
    let d_idx = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc idx");
    let d_val = gpu.alloc_tensor(&[b * ktop], DType::F32).expect("alloc val");

    // Warmup all arms.
    for _ in 0..WARMUP {
        gpu.gemm_mq4g256v2_batched_lmhead(&d_a, &d_x, &d_logits, m, k_dim, b)
            .unwrap();
        gpu.topk_logsumexp_batched_f32(&d_logits, &d_idx, &d_val, m, ktop, b)
            .unwrap();
        gpu.mq4v2_lmhead_topk_direct_gfx1100(
            &d_a, &d_x, &partials, &ctl, &d_idx, &d_val, m, k_dim, b, ktop,
        )
        .unwrap();
    }
    sync(&gpu);

    let mut s_base: Vec<f64> = Vec::new();
    let mut s_gemm: Vec<f64> = Vec::new();
    let mut s_cand: Vec<f64> = Vec::new();
    for _ in 0..SAMPLES {
        s_base.push(time_batch(&mut gpu, &mut |g: &mut Gpu| {
            g.gemm_mq4g256v2_batched_lmhead(&d_a, &d_x, &d_logits, m, k_dim, b)
                .unwrap();
            g.topk_logsumexp_batched_f32(&d_logits, &d_idx, &d_val, m, ktop, b)
                .unwrap();
        }));
        s_gemm.push(time_batch(&mut gpu, &mut |g: &mut Gpu| {
            g.gemm_mq4g256v2_batched_lmhead(&d_a, &d_x, &d_logits, m, k_dim, b)
                .unwrap();
        }));
        s_cand.push(time_batch(&mut gpu, &mut |g: &mut Gpu| {
            g.mq4v2_lmhead_topk_direct_gfx1100(
                &d_a, &d_x, &partials, &ctl, &d_idx, &d_val, m, k_dim, b, ktop,
            )
            .unwrap();
        }));
    }
    let med_base = median(s_base.clone());
    let med_gemm = median(s_gemm.clone());
    let med_cand = median(s_cand.clone());
    let speedup = med_base - med_cand;
    let tail = med_cand - med_gemm;
    println!(
        "timing B=15 K=8 us/launch: base(gemm+topk) med={med_base:.1} min={:.1} | gemm-only med={med_gemm:.1} | fused med={med_cand:.1} min={:.1} | saving={speedup:.1} tail={tail:.1}",
        s_base.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        s_cand.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
    );
    let perf_ok = speedup >= 1000.0 && tail <= 121.0;
    if !perf_ok {
        all_ok = false;
        eprintln!(
            "  FAIL perf: saving {speedup:.1} us (need >= 1000) tail {tail:.1} us (need <= 121)"
        );
    }

    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_logits);
    let _ = gpu.free_tensor(d_idx);
    let _ = gpu.free_tensor(d_val);
    let _ = gpu.free_tensor(partials);
    let _ = gpu.free_tensor(ctl);
    let _ = gpu.free_tensor(d_a);

    if all_ok {
        eprintln!("\nPASS: ids exact on distinct values, ties match per-partition simulations, logp f64-floor gate holds, fused >=1.0 ms faster with <=121 us tail");
    } else {
        eprintln!("\nFAIL: one or more S8 gates violated");
        std::process::exit(1);
    }
}
