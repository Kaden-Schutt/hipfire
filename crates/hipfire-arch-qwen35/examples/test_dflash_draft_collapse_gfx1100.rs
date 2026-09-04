// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S7 gate: `test_dflash_draft_collapse_gfx1100`.
//!
//! Compares the collapsed draft path (batched noise embeddings, F16-direct
//! rotate + overwrite WMMA GEMMs, dual-output RMSNorms, fused finish
//! conv+residual) against the legacy path with poisoned scratch:
//!
//! - Part A: 16× scalar `embedding_lookup_q8` vs one
//!   `embedding_lookup_q8_batched` over a synthetic Q8 table (exact memcmp).
//! - Part B: one full `draft_forward_opts` old-vs-new on the real MQ draft
//!   artifact, memcmp over embedding/residual/norm/projection/conv/final-x
//!   planes plus thlog watermarks.
//!
//! Any mismatch fails the process (nonzero exit). On non-gfx1100 the fast
//! path is dormant by construction, so the test skips gracefully.

use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f32_to_f16;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rand_f32(state: &mut u64) -> f32 {
    // Uniform in [-1, 1).
    let u = (xorshift(state) >> 11) as f64 / (1u64 << 53) as f64;
    (u * 2.0 - 1.0) as f32
}

fn f32_slice_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

fn i32_slice_bytes(data: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

fn check_eq(name: &str, a: &[f32], b: &[f32], failures: &mut Vec<String>) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    let mut bad = 0usize;
    let mut first = None;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            if first.is_none() {
                first = Some((i, x, y));
            }
            bad += 1;
        }
    }
    if bad > 0 {
        let (i, x, y) = first.unwrap();
        failures.push(format!(
            "{name}: {bad}/{} elements differ (first idx {i}: {x:e} vs {y:e})",
            a.len()
        ));
    } else {
        eprintln!("ok   {name} ({} elems, bit-identical)", a.len());
    }
}

// ── Part A: batched vs scalar Q8 embedding ──────────────────────────────
fn part_a_embedding(gpu: &mut Gpu) -> HipResult<()> {
    use hip_bridge::HipResult;
    const VOCAB: usize = 1024;
    const DIM: usize = 2048;
    const B: usize = 16;

    // Synthetic Q8_0 table: per-32 block f16 scale + 32 i8 quants.
    let mut rng = 0x1234_5678_9abc_def1u64;
    let blocks_per_row = DIM / 32;
    let row_bytes = blocks_per_row * 34;
    let mut table = vec![0u8; VOCAB * row_bytes];
    for v in 0..VOCAB {
        for blk in 0..blocks_per_row {
            let scale = 0.001 + (xorshift(&mut rng) % 1000) as f32 / 1_000_000.0;
            let off = v * row_bytes + blk * 34;
            table[off..off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for i in 0..32 {
                let q = (xorshift(&mut rng) % 256) as i8 as u8;
                table[off + 2 + i] = q;
            }
        }
    }
    let table_gpu = gpu.upload_raw(&table, &[table.len()])?;

    let ids: Vec<u32> = (0..B).map(|_| (xorshift(&mut rng) % VOCAB as u64) as u32).collect();

    // Old path: 16 scalar lookups into rows of one [B*DIM] plane.
    let out_old = gpu.alloc_tensor(&[B * DIM], DType::F32)?;
    for (i, &tok) in ids.iter().enumerate() {
        let dst = out_old.sub_offset(i * DIM, DIM);
        gpu.embedding_lookup_q8(&table_gpu, &dst, tok, DIM)?;
    }

    // New path: upload IDs once (i32 bits in an F32 plane, like noise_tokens)
    // and run a single batched lookup.
    let ids_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    let ids_gpu = gpu.alloc_tensor(&[B], DType::F32)?;
    gpu.hip.memcpy_htod(&ids_gpu.buf, i32_slice_bytes(&ids_i32))?;
    let out_new = gpu.alloc_tensor(&[B * DIM], DType::F32)?;
    gpu.embedding_lookup_q8_batched(&table_gpu, &out_new, &ids_gpu, B, DIM)?;

    gpu.hip.device_synchronize()?;
    let a = gpu.download_f32(&out_old)?;
    let b = gpu.download_f32(&out_new)?;
    let mut failures = Vec::new();
    check_eq("embedding_q8_batched_vs_scalar", &a, &b, &mut failures);

    // The ID plane itself must hold the exact uploaded bits.
    let ids_back = gpu.download_f32(&ids_gpu)?;
    let ids_back_i32: Vec<i32> = ids_back.iter().map(|&f| f.to_bits() as i32).collect();
    if ids_back_i32 != ids_i32 {
        failures.push("noise id plane round-trip mismatch".to_string());
    } else {
        eprintln!("ok   noise id plane round-trip ({B} ids)");
    }

    let _ = gpu.free_tensor(out_old);
    let _ = gpu.free_tensor(out_new);
    let _ = gpu.free_tensor(ids_gpu);
    let _ = gpu.free_tensor(table_gpu);
    if failures.is_empty() {
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        Err(hip_bridge::HipError::new(0, "part A embedding mismatch"))
    }
}

// Poison every data tensor in a scratch with 0xCD bytes (thlog/graph
// caches intentionally untouched — structural state, not data).
fn poison_scratch(gpu: &Gpu, s: &DflashScratch) -> HipResult<()> {
    use hip_bridge::HipResult;
    let mut all: Vec<&GpuTensor> = vec![
        &s.x,
        &s.x_norm,
        &s.q,
        &s.k_noise,
        &s.v_noise,
        &s.gate,
        &s.up,
        &s.gate_up,
        &s.attn_out,
        &s.residual,
        &s.target_hidden,
        &s.target_hidden_proj,
        &s.k_cat,
        &s.v_cat,
        &s.positions_q,
        &s.positions_k,
        &s.noise_tokens,
    ];
    for t in [&s.mq_x_rot, &s.mq_x_rot_f16].into_iter().flatten() {
        all.push(t);
    }
    for t in [
        &s.conv_temp,
        &s.conv_dynamic,
        &s.selector_proj,
        &s.topk_ids,
        &s.topk_vals,
    ]
    .into_iter()
    .flatten()
    {
        all.push(t);
    }
    for t in s.k_ctx_cached.iter().chain(s.v_ctx_cached.iter()) {
        all.push(t);
    }
    for t in [
        &s.k_full_cached,
        &s.v_full_cached,
        &s.k_cat_full,
        &s.v_cat_full,
    ]
    .into_iter()
    .flatten()
    {
        all.push(t);
    }
    for t in all {
        gpu.hip.memset(&t.buf, 0xCD, t.buf.size())?;
    }
    Ok(())
}

fn upload_inputs(
    gpu: &Gpu,
    s: &DflashScratch,
    noise: &[f32],
    th: &[f32],
    pos_q: &[i32],
    pos_k: &[i32],
) -> HipResult<()> {
    use hip_bridge::HipResult;
    gpu.hip.memcpy_htod(&s.x.buf, f32_slice_bytes(noise))?;
    gpu.hip.memcpy_htod(&s.target_hidden.buf, f32_slice_bytes(th))?;
    gpu.hip
        .memcpy_htod(&s.positions_q.buf, i32_slice_bytes(pos_q))?;
    gpu.hip
        .memcpy_htod(&s.positions_k.buf, i32_slice_bytes(pos_k))?;
    Ok(())
}

// ── Part B: full draft forward old-vs-new ───────────────────────────────
// `gpu.flags` is an immutable Arc, so old-vs-new runs in two processes
// (the kill switch is env-read at startup). `dump` runs one forward with
// the process's flag and serializes every compared plane; `cmp` byte-
// compares two dumps; default mode re-execs both dumps and compares.
fn dump_forward(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    outdir: &Path,
) -> HipResult<()> {
    use hip_bridge::HipResult;
    // Gate-matching geometry: the MERGESORT gate overrides --block-size 16
    // (the artifact declares 8), so run B=16 here too.
    let b = 16usize;
    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let ctx_cap = 256usize;
    let l = 64usize;
    eprintln!(
        "draft: n_layers={} hidden={h} inter={} b={b} l={l} collapse_off={}",
        cfg.n_layers,
        cfg.intermediate,
        gpu.flags.draft_collapse_off,
    );

    let mut s = if let Some(w) = cfg.declared_window {
        let w_full = if cfg.all_layers_sliding { w } else { ctx_cap };
        DflashScratch::new_windowed(gpu, cfg, b, w, w_full, ctx_cap, weights.has_mq)?
    } else {
        DflashScratch::new_with_mq(gpu, cfg, b, ctx_cap, weights.has_mq)?
    };

    // Deterministic synthetic inputs (fixed seed ⇒ identical across the
    // old/new dump processes).
    let mut rng = 0x2b7e_1516_28ae_d2a6u64;
    let noise: Vec<f32> = (0..b * h).map(|_| rand_f32(&mut rng)).collect();
    let th: Vec<f32> = (0..l * ne * h).map(|_| rand_f32(&mut rng) * 0.5).collect();
    let pos_q: Vec<i32> = (0..b).map(|i| 1000 + i as i32).collect();
    let pos_k: Vec<i32> = (0..l + b).map(|i| 1000 - l as i32 + i as i32).collect();

    poison_scratch(gpu, &s)?;
    upload_inputs(gpu, &s, &noise, &th, &pos_q, &pos_k)?;

    hipfire_runtime::dflash::draft_forward_opts(
        gpu, weights, cfg, None, None, &pos_q, &pos_k, b, l, &mut s, false,
    )?;
    gpu.hip.device_synchronize()?;

    std::fs::create_dir_all(outdir).expect("mkdir dump dir");
    let mut manifest = String::new();
    let mut dump = |name: &str, t: &GpuTensor| -> HipResult<()> {
        let v = gpu.download_f32(t)?;
        std::fs::write(outdir.join(format!("{name}.f32")), f32_slice_bytes(&v))
            .expect("write plane");
        manifest.push_str(&format!("{name} {}\n", v.len()));
        Ok(())
    };
    // Embedding entry plane is pre-loaded identically in both dumps; the
    // forward's first residual capture must see the same entry x.
    dump("final_x", &s.x)?;
    dump("residual", &s.residual)?;
    dump("x_norm", &s.x_norm)?;
    dump("q", &s.q)?;
    dump("k_noise", &s.k_noise)?;
    dump("v_noise", &s.v_noise)?;
    dump("gate", &s.gate)?;
    dump("up", &s.up)?;
    dump("gate_up", &s.gate_up)?;
    dump("attn_out", &s.attn_out)?;
    dump("target_hidden_proj", &s.target_hidden_proj)?;
    dump("k_cat", &s.k_cat)?;
    dump("v_cat", &s.v_cat)?;
    if let Some(t) = &s.conv_temp {
        dump("conv_temp", t)?;
    }
    if let Some(t) = &s.conv_dynamic {
        dump("conv_dynamic", t)?;
    }
    for (li, t) in s.k_ctx_cached.iter().enumerate() {
        dump(&format!("k_ctx_cached_{li}"), t)?;
    }
    for (li, t) in s.v_ctx_cached.iter().enumerate() {
        dump(&format!("v_ctx_cached_{li}"), t)?;
    }
    // NOTE: mq_x_rot (F32) vs mq_x_rot_f16 (F16) differ by design; their
    // consumers' outputs (all projections above) are the parity check.
    manifest.push_str(&format!("thlog_proj_cached_rows {}\n", s.thlog.proj_cached_rows()));
    manifest.push_str(&format!("thlog_uploaded_rows {}\n", s.thlog.uploaded_rows()));
    manifest.push_str(&format!("thlog_full_cached_rows {}\n", s.thlog.full_cached_rows()));
    std::fs::write(outdir.join("MANIFEST"), manifest).expect("write manifest");
    eprintln!("dumped forward (collapse_off={}) to {}", gpu.flags.draft_collapse_off, outdir.display());
    Ok(())
}

fn cmp_dumps(a: &Path, b: &Path) -> HipResult<()> {
    use hip_bridge::HipResult;
    let ma = std::fs::read_to_string(a.join("MANIFEST")).expect("read manifest A");
    let mb = std::fs::read_to_string(b.join("MANIFEST")).expect("read manifest B");
    if ma != mb {
        return Err(hip_bridge::HipError::new(0, "dump manifests differ"));
    }
    let mut fails = 0usize;
    for line in ma.lines() {
        let mut it = line.split_whitespace();
        let name = it.next().unwrap();
        if name.starts_with("thlog_") {
            eprintln!("ok   {name} = {}", it.next().unwrap());
            continue;
        }
        let fa = std::fs::read(a.join(format!("{name}.f32"))).expect("read plane A");
        let fb = std::fs::read(b.join(format!("{name}.f32"))).expect("read plane B");
        if fa != fb {
            let mut nbad = 0usize;
            for (x, y) in fa.chunks_exact(4).zip(fb.chunks_exact(4)) {
                if x != y {
                    nbad += 1;
                }
            }
            eprintln!("FAIL {name}: {nbad}/{} f32 differ", fa.len() / 4);
            fails += 1;
        } else {
            eprintln!("ok   {name} ({} f32, bit-identical)", fa.len() / 4);
        }
    }
    if fails > 0 {
        return Err(hip_bridge::HipError::new(0, "dump planes differ"));
    }
    eprintln!("PART B PASS: five-layer draft forward bit-identical");
    Ok(())
}

use hip_bridge::HipResult;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // `dump <outdir> [draft.hfq]`: single-process forward (used by re-exec).
    // `cmp <a> <b>`: host-side compare, no GPU needed.
    if args.get(1).map(|s| s.as_str()) == Some("cmp") {
        cmp_dumps(Path::new(&args[2]), Path::new(&args[3])).expect("cmp");
        return;
    }
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("gpu: {} (gfx1100={})", gpu.arch, gpu.arch_caps.is_gfx1100());
    if !gpu.arch_caps.is_gfx1100() {
        eprintln!("SKIP: S7 fast path is gfx1100-only and dormant here.");
        return;
    }
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
    }
    let default_draft = || {
        format!(
            "{}/.hipfire/models/qwen38-27b-dflash-mq4.hfq",
            std::env::var("HOME").unwrap()
        )
    };
    if args.get(1).map(|s| s.as_str()) == Some("dump") {
        let draft_path = args.get(3).cloned().unwrap_or_else(default_draft);
        let draft_hfq = HfqFile::open(Path::new(&draft_path)).expect("open draft artifact");
        let cfg = DflashConfig::from_hfq(&draft_hfq).expect("parse DflashConfig");
        let weights = DflashWeights::load(&mut gpu, &draft_hfq, &cfg).expect("load draft");
        dump_forward(&mut gpu, &weights, &cfg, Path::new(&args[2])).expect("dump");
        return;
    }

    // Default gate mode: Part A in-process, then re-exec old/new dumps.
    part_a_embedding(&mut gpu).expect("part A");

    let draft_path = default_draft();
    let draft_hfq = HfqFile::open(Path::new(&draft_path)).expect("open draft artifact");
    let cfg = DflashConfig::from_hfq(&draft_hfq).expect("parse DflashConfig");
    assert_eq!(cfg.n_layers, 5, "gate expects a five-layer draft");
    eprintln!("draft config ok (five layers)");

    let tmp = std::env::temp_dir().join(format!("s7-collapse-{}", std::process::id()));
    let new_dir = tmp.join("new");
    let old_dir = tmp.join("old");
    let exe = std::env::current_exe().expect("current exe");
    let run = |dir: &Path, off: bool| {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("dump").arg(dir).arg(&draft_path);
        if off {
            cmd.env("HIPFIRE_DRAFT_COLLAPSE_OFF", "1");
        }
        let st = cmd.status().expect("re-exec dump");
        assert!(st.success(), "dump failed (off={off})");
    };
    // Weights load inside each dump child; here only the config is checked.
    run(&new_dir, false);
    run(&old_dir, true);
    cmp_dumps(&new_dir, &old_dir).expect("part B");
    eprintln!("ALL S7 PARITY CHECKS PASS");
}
