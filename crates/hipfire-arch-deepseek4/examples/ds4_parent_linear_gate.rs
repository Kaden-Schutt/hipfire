// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 3 of the DS4 parent-checkpoint calibration path: prove the GPU
//! parent-linear path computes the same function as the checkpoint's bundled
//! `fp8_gemm` / `fp4_gemm` operator semantics on fixed inputs.
//!
//! The bundled reference cannot execute on mi300x (no torch/tilelang), so we
//! match a CPU oracle that implements those semantics exactly.
//!
//! # Acceptance (refined §4)
//!
//! Two independent signals — do **not** collapse them into one ratio cliff:
//!
//! 1. **Bias (hard fail).** `bias_z = |mean(GPU−Exact)| / std(GPU−Exact)` must
//!    be near zero (order `1/sqrt(count)`). A materially non-zero bias means a
//!    misplaced scale, regardless of magnitude. Small biased error is worse
//!    than larger unbiased error.
//! 2. **Magnitude (tree calibration).** ReferenceOrder rounds once per 128-
//!    term (fp8) or 32-term (fp4) block, so `err_ref` is far below the crude
//!    `√K·ε` bound. The GPU MFMA accumulates all K in f32 (~K/16 steps), so
//!    `err_gpu/err_ref ≈ √(chain_gpu/chain_ref)` can land near 2.8× and may
//!    exceed 4× on some shapes without being a semantic defect. When the
//!    ratio sits above 4× we calibrate against a third CPU mode —
//!    **SequentialF32** (all-K sequential f32 accumulation, no blocking) —
//!    and require `err_gpu` to match that magnitude rather than ReferenceOrder.
//! 3. **Self-check against vacuous passes.** If `err_ref == 0` the case cannot
//!    discriminate accumulation order (every term was exactly representable in
//!    f32). Such cases are reported **INCONCLUSIVE** (layout/indexing only),
//!    never PASS. Wide UE8M0 exponent spreads and real checkpoint tensors
//!    force real f32 rounding so the gate has teeth.
//!
//! Why the GPU path is the same mathematical function (contract §3): every
//! scale is a power of two. An E4M3 code × 2^e has a 3-bit mantissa and an
//! E2M1 code × 2^e has a 1-bit mantissa — both exactly representable in BF16.
//! Each product term is therefore the identical real number in both
//! formulations and exact in FP32 (≤ 8 significand bits). The *only*
//! difference is FP32 summation order and intermediate rounding.
//!
//! Note: `parent_linear_dense` / `parent_linear_expert` are **destructive** on
//! `x_bf16` (inplace act-quant). Every case allocates a fresh activation buffer.
//!
//! Usage (on mi300x):
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_linear_gate \
//!   --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!   [--manifest out/manifest.json] [--skip-shard-hashes] [--seed N]
//! ```
//!
//! Exit code 0 only if every meaningful case PASSes (INCONCLUSIVE is allowed).

use hipfire_arch_deepseek4::parent::codec::{
    e2m1_to_f32, e4m3_to_f32, round_to_bf16, ue8m0_to_f32,
};
use hipfire_arch_deepseek4::parent::gemm_ref::{
    act_quant_fp8_codes, linear_fp4_ref, linear_fp8_ref, AccumMode,
};
use hipfire_arch_deepseek4::parent::linear::{
    parent_linear_dense, parent_linear_expert, ParentDenseWeight, ParentExpertWeight,
};
use hipfire_arch_deepseek4::parent::manifest::{
    sha256_file, CaptureBoundary, CaptureInfo, ModelInfo, ModelQuantInfo, ParentManifest,
    ShardInfo, SourceInfo, MANIFEST_SCHEMA,
};
use hipfire_arch_deepseek4::parent::Ds4ParentBackend;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_SEED: u64 = 0x00D5_4CA7_E320_26u64;
const DENSE_BLOCK: usize = 128;
const EXPERT_GROUP: usize = 32;

/// Absolute bias ceiling. Main's fail example was bias_z≈0.4 (scale in the
/// wrong place). Unbiased f32 residual under wide dynamic range can still
/// show bias_z of a few percent of std because rounding is not white noise —
/// that is not a misplaced scale. 0.25 separates the two.
const BIAS_Z_LIMIT: f64 = 0.25;
/// Soft band vs ReferenceOrder — covers √(K/16 / K/128) ≈ 2.8× with slack.
const RATIO_SOFT: f64 = 4.0;
/// Hard ceiling vs ReferenceOrder even after sequential calibration.
const RATIO_HARD: f64 = 16.0;
/// When ratio > RATIO_SOFT, require err_gpu within this factor of err_seq.
const SEQ_MATCH: f64 = 2.0;
/// err_ref below this is treated as zero → INCONCLUSIVE (cannot discriminate).
const ERR_REF_FLOOR: f64 = 1e-15;

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

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
    let model_path = Path::new(&args.model);
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a safetensors directory, got {}",
            model_path.display()
        ));
    }

    println!("=== ds4_parent_linear_gate (Gate 3) ===");
    println!("model: {}", model_path.display());
    println!("seed:  {} (0x{:X})", args.seed, args.seed);
    println!("skip_shard_hashes: {}", args.skip_shard_hashes);
    if let Some(m) = args.manifest.as_ref() {
        println!("manifest: {}", m.display());
    }
    println!();
    println!(
        "acceptance (refined §4):\n\
           BIAS  hard fail: |mean(GPU−Exact)|/std < {BIAS_Z_LIMIT}\n\
           MAG   ratio=err_gpu/err_ref ≤ {RATIO_SOFT}× → magnitude OK;\n\
               {RATIO_SOFT}× < ratio ≤ {RATIO_HARD}× → OK only if err_gpu matches\n\
               sequential-f32 accumulation (within {SEQ_MATCH}× of err_seq);\n\
               ratio > {RATIO_HARD}× → FAIL\n\
           VACUOUS  err_ref == 0 → INCONCLUSIVE (layout check only, not evidence)"
    );
    println!(
        "reasoning: every UE8M0 scale is a power of two, so E4M3/E2M1 × scale is exact in BF16;\n\
         GPU BF16×BF16→FP32 and the reference block-scaled GEMM differ only by FP32 summation order.\n\
         ReferenceOrder rounds once per block (shallow tree); GPU MFMA is ~K/16 deep — expect\n\
         err_gpu/err_ref ≈ √(depth_gpu/depth_ref) ≈ 2.8× at K=4096, not 1.0×."
    );
    println!();

    // ── GPU + admit ──────────────────────────────────────────────────────
    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init failed: {e:?}"))?;
    let gfx = gpu
        .try_gfx942()
        .map(|_| "gfx942")
        .unwrap_or("not-gfx942");
    println!("gpu: {gfx}");
    if gfx != "gfx942" {
        return Err(format!(
            "deepseek4 parent: Gate 3 requires gfx942, got {gfx}"
        ));
    }

    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;
    println!(
        "opened SafetensorsSource: arch_id={} tensors={}",
        ModelSource::arch_id(&source),
        source.tensor_names().len()
    );

    let admit_t0 = Instant::now();
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    println!(
        "admit: OK ({:.1} ms)  model_type={} quant={} fmt={} scale_fmt={} expert={}",
        admit_t0.elapsed().as_secs_f64() * 1000.0,
        cfg.model_type,
        cfg.quant_method,
        cfg.fmt,
        cfg.scale_fmt,
        cfg.expert_dtype
    );
    println!();

    let mut cases: Vec<CaseResult> = Vec::new();
    let mut rng = XorShift64::new(args.seed);

    // ── A. Synthetic dense (FP8) ─────────────────────────────────────────
    println!("── Part A: synthetic dense FP8 linear ──");
    // tiny: intentionally narrow/exact → expected INCONCLUSIVE (layout only)
    // others: wide scales force real f32 rounding so the gate has teeth
    let dense_shapes: &[(usize, usize, usize, &str, ScaleMode)] = &[
        (1, 128, 128, "tiny_layout", ScaleMode::Mild),
        (32, 1024, 4096, "batch32_n1024_k4096", ScaleMode::Wide),
        (8, 4096, 2048, "batch8_n4096_k2048", ScaleMode::Wide),
        // n not multiple of 128 → ragged weight-scale grid (ceil(n/128))
        (4, 200, 256, "ragged_n200", ScaleMode::Wide),
        (16, 512, 512, "wide_scale_range", ScaleMode::Wide),
    ];
    for &(m, n, k, tag, smode) in dense_shapes {
        let name = format!("synth_dense_{tag}_m{m}_n{n}_k{k}");
        println!();
        println!("  case {name}  scale_mode={smode:?}");
        match run_synth_dense(&mut gpu, backend, &mut rng, m, n, k, smode, &name) {
            Ok(r) => {
                print_case(&r);
                cases.push(r);
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
                cases.push(CaseResult::error(name, "dense", e));
            }
        }
    }

    // ── B. Synthetic expert (FP4) ────────────────────────────────────────
    println!();
    println!("── Part B: synthetic expert FP4 linear ──");
    let expert_shapes: &[(usize, usize, usize, &str, ScaleMode)] = &[
        (1, 128, 128, "tiny_layout", ScaleMode::Mild),
        (32, 1024, 4096, "batch32_n1024_k4096", ScaleMode::Wide),
        (8, 4096, 2048, "batch8_n4096_k2048", ScaleMode::Wide),
        (4, 200, 256, "ragged_n200", ScaleMode::Wide),
        (16, 512, 512, "wide_scale_range", ScaleMode::Wide),
    ];
    for &(m, n, k, tag, smode) in expert_shapes {
        let name = format!("synth_expert_{tag}_m{m}_n{n}_k{k}");
        println!();
        println!("  case {name}  scale_mode={smode:?}");
        match run_synth_expert(&mut gpu, backend, &mut rng, m, n, k, smode, &name) {
            Ok(r) => {
                print_case(&r);
                cases.push(r);
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
                cases.push(CaseResult::error(name, "expert", e));
            }
        }
    }

    // ── C. Real checkpoint tensors ───────────────────────────────────────
    println!();
    println!("── Part C: real checkpoint tensors ──");
    {
        let name = "real_dense_layers3_attn_wq_a_m32".to_string();
        println!();
        println!("  case {name}");
        match run_real_dense(&mut gpu, backend, model_path, &mut rng, 32, &name) {
            Ok(r) => {
                print_case(&r);
                cases.push(r);
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
                cases.push(CaseResult::error(name, "dense", e));
            }
        }
    }
    {
        let name = "real_expert_layers3_ffn_experts0_w1_m32".to_string();
        println!();
        println!("  case {name}");
        match run_real_expert(&mut gpu, backend, model_path, &mut rng, 32, &name) {
            Ok(r) => {
                print_case(&r);
                cases.push(r);
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
                cases.push(CaseResult::error(name, "expert", e));
            }
        }
    }

    // ── Summary table ────────────────────────────────────────────────────
    println!();
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!(" Gate 3 linear summary");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!(
        "{:<48} {:<6} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {}",
        "CASE", "TIER", "err_ref", "err_gpu", "err_seq", "r/ref", "r/seq", "bias_z", "VERDICT"
    );
    println!("{}", "─".repeat(130));
    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_inconclusive = 0usize;
    for c in &cases {
        match c.verdict {
            Verdict::Pass => n_pass += 1,
            Verdict::Fail => n_fail += 1,
            Verdict::Inconclusive => n_inconclusive += 1,
        }
        let st = c.verdict.as_str();
        if c.error.is_some() {
            println!(
                "{:<48} {:<6} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {}  ({})",
                trunc(&c.name, 48),
                c.tier,
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                st,
                c.error.as_deref().unwrap_or("")
            );
        } else {
            println!(
                "{:<48} {:<6} {:>9.2e} {:>9.2e} {:>9.2e} {:>7.2} {:>7.2} {:>7.3} {}",
                trunc(&c.name, 48),
                c.tier,
                c.err_ref,
                c.err_gpu,
                c.err_seq,
                c.ratio_ref,
                c.ratio_seq,
                c.bias_z,
                st
            );
        }
    }
    println!("{}", "─".repeat(130));
    println!(
        "total={}  pass={}  fail={}  inconclusive={}",
        cases.len(),
        n_pass,
        n_fail,
        n_inconclusive
    );
    println!();

    println!("── interpretation rollup ──");
    for c in &cases {
        if let Some(e) = &c.error {
            println!("  {}: ERROR — {e}", c.name);
        } else {
            println!("  [{}] {}: {}", c.verdict.as_str(), c.name, c.interpretation);
        }
    }
    println!();

    // ── Manifest ─────────────────────────────────────────────────────────
    println!("=== manifest ===");
    let (producer, engine) = ParentManifest::probe_environment("gfx942")?;
    println!(
        "engine: commit={} dirty={} rocm={}@{} arch={}",
        engine.commit,
        engine.dirty_diff_sha256.as_deref().unwrap_or("clean"),
        engine.rocm_path,
        engine.rocm_version,
        engine.gpu_arch
    );
    let source_info = build_source_info(model_path, args.skip_shard_hashes)?;
    println!(
        "source: {} shards, index_sha256={}, config_sha256={}",
        source_info.shards.len(),
        &source_info.index_sha256[..16.min(source_info.index_sha256.len())],
        &source_info.config_sha256[..16.min(source_info.config_sha256.len())]
    );

    // This producer consumes no corpus and captures no activations — it only
    // compares GEMM outputs on fixed inputs. corpus=None is permitted by
    // validate() exactly when there are no outputs and no captured tensors.
    let corpus = None;
    let produced_utc = utc_now_rfc3339();
    let manifest = ParentManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        produced_utc,
        producer,
        engine,
        source: source_info,
        model: ModelInfo {
            model_type: cfg.model_type.clone(),
            num_hidden_layers: cfg.num_hidden_layers,
            mtp_loaded: false,
            rope_convention: "yarn".to_string(),
            quant: ModelQuantInfo {
                quant_method: cfg.quant_method.clone(),
                fmt: cfg.fmt.clone(),
                scale_fmt: cfg.scale_fmt.clone(),
                expert_dtype: cfg.expert_dtype.clone(),
                weight_block_size: cfg.weight_block_size,
            },
        },
        corpus,
        capture: CaptureInfo {
            boundary: CaptureBoundary::PostDynamicFp8,
            tensors: Vec::new(),
        },
        outputs: Vec::new(),
    };
    manifest
        .validate()
        .map_err(|e| format!("deepseek4 parent: manifest validate failed: {e}"))?;
    println!("manifest.validate(): OK (null corpus, no outputs declared)");
    if let Some(path) = args.manifest.as_ref() {
        manifest.write_to(path)?;
        println!("wrote {}", path.display());
        let written = fs::read_to_string(path).map_err(|e| {
            format!(
                "deepseek4 parent: re-read manifest {}: {e}",
                path.display()
            )
        })?;
        println!("--- manifest.json begin ---");
        print!("{written}");
        if !written.ends_with('\n') {
            println!();
        }
        println!("--- manifest.json end ---");
    }

    println!();
    println!("=== gate 3 summary ===");
    println!("admit:            PASS");
    println!(
        "cases:            {} pass / {} fail / {} inconclusive / {} total",
        n_pass,
        n_fail,
        n_inconclusive,
        cases.len()
    );
    // Require at least one meaningful (non-inconclusive) pass per tier.
    let meaningful_dense = cases
        .iter()
        .filter(|c| c.tier == "dense" && c.verdict == Verdict::Pass)
        .count();
    let meaningful_expert = cases
        .iter()
        .filter(|c| c.tier == "expert" && c.verdict == Verdict::Pass)
        .count();
    println!("meaningful dense PASS:   {meaningful_dense}");
    println!("meaningful expert PASS:  {meaningful_expert}");

    if n_fail == 0 && meaningful_dense > 0 && meaningful_expert > 0 {
        println!("GATE 3: PASS");
        println!(
            "assessment: GPU parent-linear path is semantically equivalent to the bundled\n\
             fp8_gemm/fp4_gemm reference. Bias is near zero on every meaningful case (no\n\
             misplaced scale). Magnitude residuals sit on the sequential-f32 accumulation\n\
             tree (deeper than ReferenceOrder's per-block reduction), not on a wrong\n\
             operator. Vacuous err_ref=0 cases are labelled INCONCLUSIVE, not PASS."
        );
        Ok(())
    } else if n_fail > 0 {
        println!("GATE 3: FAIL");
        Err(format!(
            "deepseek4 parent: Gate 3 failed {n_fail}/{} cases — see table above",
            cases.len()
        ))
    } else {
        println!("GATE 3: FAIL (no meaningful evidence)");
        Err(format!(
            "deepseek4 parent: Gate 3 produced no meaningful PASS \
             (dense={meaningful_dense} expert={meaningful_expert}); all cases vacuous?"
        ))
    }
}

// ---------------------------------------------------------------------------
// Case runners
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum ScaleMode {
    /// Mild exponents — may yield err_ref≈0 on tiny shapes (layout-only).
    Mild,
    /// Wide UE8M0 spread (2^-20 .. 2^10) so f32 addition loses bits.
    Wide,
}

fn run_synth_dense(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    rng: &mut XorShift64,
    m: usize,
    n: usize,
    k: usize,
    smode: ScaleMode,
    name: &str,
) -> Result<CaseResult, String> {
    let sn = (n + DENSE_BLOCK - 1) / DENSE_BLOCK;
    let sk = (k + DENSE_BLOCK - 1) / DENSE_BLOCK;
    let b_codes = gen_e4m3_codes(rng, n * k);
    let b_scales = match smode {
        ScaleMode::Wide => gen_ue8m0_wide(rng, sn * sk),
        ScaleMode::Mild => gen_ue8m0_mild(rng, sn * sk),
    };
    let x = gen_acts_bf16(rng, m * k, smode);
    run_dense_case(gpu, backend, &x, &b_codes, &b_scales, m, n, k, name)
}

fn run_synth_expert(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    rng: &mut XorShift64,
    m: usize,
    n: usize,
    k: usize,
    smode: ScaleMode,
    name: &str,
) -> Result<CaseResult, String> {
    if k % EXPERT_GROUP != 0 {
        return Err(format!(
            "deepseek4 parent: expert k={k} not multiple of {EXPERT_GROUP}"
        ));
    }
    if k % 2 != 0 {
        return Err(format!("deepseek4 parent: expert k={k} not even"));
    }
    let b_codes = gen_e2m1_packed(rng, n * (k / 2));
    let b_scales = match smode {
        ScaleMode::Wide => gen_ue8m0_wide(rng, n * (k / EXPERT_GROUP)),
        ScaleMode::Mild => gen_ue8m0_mild(rng, n * (k / EXPERT_GROUP)),
    };
    let x = gen_acts_bf16(rng, m * k, smode);
    run_expert_case(gpu, backend, &x, &b_codes, &b_scales, m, n, k, name)
}

fn run_real_dense(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    model: &Path,
    rng: &mut XorShift64,
    m: usize,
    name: &str,
) -> Result<CaseResult, String> {
    let shard = model.join("model-00005-of-00048.safetensors");
    let (_wd, ws, wbytes) = load_from_shard(&shard, "layers.3.attn.wq_a.weight")?;
    let (_sd, ss, sbytes) = load_from_shard(&shard, "layers.3.attn.wq_a.scale")?;
    if ws != [1024, 4096] {
        return Err(format!(
            "deepseek4 parent: unexpected wq_a.weight shape {ws:?}"
        ));
    }
    if ss != [8, 32] {
        return Err(format!(
            "deepseek4 parent: unexpected wq_a.scale shape {ss:?}"
        ));
    }
    let n = ws[0];
    let k = ws[1];
    println!("    weight F8_E4M3 {ws:?}  scale F8_E8M0 {ss:?}  act batch m={m}");
    // Wide activations so act-quant scales vary across the real weight scales.
    let x = gen_acts_bf16(rng, m * k, ScaleMode::Wide);
    run_dense_case(gpu, backend, &x, &wbytes, &sbytes, m, n, k, name)
}

fn run_real_expert(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    model: &Path,
    rng: &mut XorShift64,
    m: usize,
    name: &str,
) -> Result<CaseResult, String> {
    let shard = model.join("model-00005-of-00048.safetensors");
    let (_wd, ws, wbytes) = load_from_shard(&shard, "layers.3.ffn.experts.0.w1.weight")?;
    let (_sd, ss, sbytes) = load_from_shard(&shard, "layers.3.ffn.experts.0.w1.scale")?;
    if ws != [2048, 2048] {
        return Err(format!(
            "deepseek4 parent: unexpected expert w1.weight shape {ws:?}"
        ));
    }
    if ss != [2048, 128] {
        return Err(format!(
            "deepseek4 parent: unexpected expert w1.scale shape {ss:?}"
        ));
    }
    let n = ws[0];
    let k = ws[1] * 2; // packed along K
    println!(
        "    weight I8 {ws:?} logical [{n},{k}]  scale F8_E8M0 {ss:?}  act batch m={m}"
    );
    let x = gen_acts_bf16(rng, m * k, ScaleMode::Wide);
    run_expert_case(gpu, backend, &x, &wbytes, &sbytes, m, n, k, name)
}

fn run_dense_case(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    x: &[f32],
    b_codes: &[u8],
    b_scales: &[u8],
    m: usize,
    n: usize,
    k: usize,
    name: &str,
) -> Result<CaseResult, String> {
    let t0 = Instant::now();
    let exact = linear_fp8_ref(x, b_codes, b_scales, m, n, k, AccumMode::Exact)?;
    let t_exact = t0.elapsed();
    let t1 = Instant::now();
    let reference = linear_fp8_ref(x, b_codes, b_scales, m, n, k, AccumMode::ReferenceOrder)?;
    let t_ref = t1.elapsed();
    let t1b = Instant::now();
    let sequential = linear_fp8_sequential(x, b_codes, b_scales, m, n, k)?;
    let t_seq = t1b.elapsed();

    // Fresh x buffer every case — parent_linear_* is destructive (inplace quant).
    let t2 = Instant::now();
    let w = ParentDenseWeight::decode_resident(gpu, backend, b_codes, b_scales, n, k)?;
    let x_bytes = pack_f32_to_bf16_bytes(x);
    let x_t = upload_bf16(gpu, &x_bytes, &[m, k])?;
    let out_t = gpu
        .alloc_tensor(&[m, n], DType::F32)
        .map_err(|e| format!("deepseek4 parent: alloc out: {e:?}"))?;
    {
        let zeros = vec![0f32; m * n];
        let zb =
            unsafe { std::slice::from_raw_parts(zeros.as_ptr() as *const u8, zeros.len() * 4) };
        gpu.hip
            .memcpy_htod(&out_t.buf, zb)
            .map_err(|e| format!("deepseek4 parent: zero out: {e:?}"))?;
    }
    parent_linear_dense(gpu, backend, &w, &x_t, m, &out_t)?;
    let gpu_out = download_f32(gpu, &out_t, m * n)?;
    let t_gpu = t2.elapsed();

    let _ = gpu.free_tensor(x_t);
    let _ = gpu.free_tensor(out_t);
    drop(w);

    println!(
        "    timings: exact={:.1}ms ref={:.1}ms seq={:.1}ms gpu={:.1}ms",
        t_exact.as_secs_f64() * 1000.0,
        t_ref.as_secs_f64() * 1000.0,
        t_seq.as_secs_f64() * 1000.0,
        t_gpu.as_secs_f64() * 1000.0
    );
    Ok(score_case(
        name,
        "dense",
        m,
        n,
        k,
        &exact,
        &reference,
        &sequential,
        &gpu_out,
    ))
}

fn run_expert_case(
    gpu: &mut Gpu,
    backend: Ds4ParentBackend,
    x: &[f32],
    b_codes: &[u8],
    b_scales: &[u8],
    m: usize,
    n: usize,
    k: usize,
    name: &str,
) -> Result<CaseResult, String> {
    let t0 = Instant::now();
    let exact = linear_fp4_ref(x, b_codes, b_scales, m, n, k, AccumMode::Exact)?;
    let t_exact = t0.elapsed();
    let t1 = Instant::now();
    let reference = linear_fp4_ref(x, b_codes, b_scales, m, n, k, AccumMode::ReferenceOrder)?;
    let t_ref = t1.elapsed();
    let t1b = Instant::now();
    let sequential = linear_fp4_sequential(x, b_codes, b_scales, m, n, k)?;
    let t_seq = t1b.elapsed();

    let t2 = Instant::now();
    let w = ParentExpertWeight::upload_compressed(gpu, backend, b_codes, b_scales, n, k)?;
    let scratch = gpu
        .alloc_tensor(&[n, k], DType::BF16)
        .map_err(|e| format!("deepseek4 parent: alloc expert scratch: {e:?}"))?;
    w.decode_into(gpu, &scratch)?;

    // Fresh x every case — destructive act-quant.
    let x_bytes = pack_f32_to_bf16_bytes(x);
    let x_t = upload_bf16(gpu, &x_bytes, &[m, k])?;
    let out_t = gpu
        .alloc_tensor(&[m, n], DType::F32)
        .map_err(|e| format!("deepseek4 parent: alloc out: {e:?}"))?;
    {
        let zeros = vec![0f32; m * n];
        let zb =
            unsafe { std::slice::from_raw_parts(zeros.as_ptr() as *const u8, zeros.len() * 4) };
        gpu.hip
            .memcpy_htod(&out_t.buf, zb)
            .map_err(|e| format!("deepseek4 parent: zero out: {e:?}"))?;
    }
    parent_linear_expert(gpu, backend, &scratch, n, k, &x_t, m, &out_t)?;
    let gpu_out = download_f32(gpu, &out_t, m * n)?;
    let t_gpu = t2.elapsed();

    let _ = gpu.free_tensor(x_t);
    let _ = gpu.free_tensor(out_t);
    let _ = gpu.free_tensor(scratch);
    drop(w);

    println!(
        "    timings: exact={:.1}ms ref={:.1}ms seq={:.1}ms gpu={:.1}ms",
        t_exact.as_secs_f64() * 1000.0,
        t_ref.as_secs_f64() * 1000.0,
        t_seq.as_secs_f64() * 1000.0,
        t_gpu.as_secs_f64() * 1000.0
    );
    Ok(score_case(
        name,
        "expert",
        m,
        n,
        k,
        &exact,
        &reference,
        &sequential,
        &gpu_out,
    ))
}

// ---------------------------------------------------------------------------
// Sequential-f32 oracles (local; not in gemm_ref — GemmOracle sticks to §5)
// ---------------------------------------------------------------------------

/// All-K sequential f32 accumulation for dense FP8 linear.
///
/// Same operands as `linear_fp8_ref` (act-quant then scaled products), but
/// accumulates `Σ_k (sa·sb·a·b)` left-to-right in f32 with no block structure.
/// This is the shallowest-accuracy / deepest-chain CPU tree and is the right
/// magnitude comparator for an MFMA that also folds all K in f32.
fn linear_fp8_sequential(
    x: &[f32],
    b: &[u8],
    b_s: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    const GROUP: usize = 128;
    if x.len() != m * k {
        return Err(format!(
            "deepseek4 parent: sequential fp8 x len {} != m*k {}",
            x.len(),
            m * k
        ));
    }
    if k % GROUP != 0 {
        return Err(format!(
            "deepseek4 parent: sequential fp8 k={k} not divisible by {GROUP}"
        ));
    }
    let (a, a_s) = act_quant_fp8_codes(x, k, GROUP)?;
    let k_blocks = k / GROUP;
    let n_scale_rows = n.div_ceil(GROUP);
    if b.len() != n * k || b_s.len() != n_scale_rows * k_blocks {
        return Err(format!(
            "deepseek4 parent: sequential fp8 weight/scale size mismatch \
             b={} need {}  b_s={} need {}",
            b.len(),
            n * k,
            b_s.len(),
            n_scale_rows * k_blocks
        ));
    }
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            let n_sb = ni / GROUP;
            for kk in 0..k {
                let kb = kk / GROUP;
                let sa = ue8m0_to_f32(a_s[mi * k_blocks + kb]);
                let sb = ue8m0_to_f32(b_s[n_sb * k_blocks + kb]);
                let av = e4m3_to_f32(a[mi * k + kk]);
                let bv = e4m3_to_f32(b[ni * k + kk]);
                // Left-to-right f32: product then add. Matches a deep chain.
                acc += (sa * sb) * (av * bv);
            }
            out[mi * n + ni] = acc;
        }
    }
    Ok(out)
}

/// All-K sequential f32 accumulation for expert FP4 linear.
fn linear_fp4_sequential(
    x: &[f32],
    b: &[u8],
    b_s: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    const ACT_GROUP: usize = 128;
    const WGT_GROUP: usize = 32;
    if x.len() != m * k {
        return Err(format!(
            "deepseek4 parent: sequential fp4 x len {} != m*k {}",
            x.len(),
            m * k
        ));
    }
    if k % ACT_GROUP != 0 || k % 2 != 0 {
        return Err(format!(
            "deepseek4 parent: sequential fp4 k={k} must be multiple of {ACT_GROUP} and even"
        ));
    }
    let (a, a_s) = act_quant_fp8_codes(x, k, ACT_GROUP)?;
    let k_act_blocks = k / ACT_GROUP;
    let k_wgt_groups = k / WGT_GROUP;
    if b.len() != n * (k / 2) || b_s.len() != n * k_wgt_groups {
        return Err(format!(
            "deepseek4 parent: sequential fp4 weight/scale size mismatch \
             b={} need {}  b_s={} need {}",
            b.len(),
            n * (k / 2),
            b_s.len(),
            n * k_wgt_groups
        ));
    }
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let ka = kk / ACT_GROUP;
                let kw = kk / WGT_GROUP;
                let sa = ue8m0_to_f32(a_s[mi * k_act_blocks + ka]);
                let sb = ue8m0_to_f32(b_s[ni * k_wgt_groups + kw]);
                let av = e4m3_to_f32(a[mi * k + kk]);
                // Packed E2M1: low nibble = even k, high nibble = odd k.
                let byte = b[ni * (k / 2) + kk / 2];
                let nibble = if kk & 1 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let bv = e2m1_to_f32(nibble);
                acc += (sa * sb) * (av * bv);
            }
            out[mi * n + ni] = acc;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Scoring (refined §4)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Pass,
    Fail,
    Inconclusive,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Inconclusive => "INCONCLUSIVE",
        }
    }
}

#[derive(Clone, Debug)]
struct CaseResult {
    name: String,
    tier: &'static str,
    #[allow(dead_code)]
    m: usize,
    #[allow(dead_code)]
    n: usize,
    #[allow(dead_code)]
    k: usize,
    err_ref: f64,
    err_gpu: f64,
    err_seq: f64,
    ratio_ref: f64,
    ratio_seq: f64,
    max_abs: f64,
    #[allow(dead_code)]
    max_rel: f64,
    bias_z: f64,
    verdict: Verdict,
    interpretation: String,
    error: Option<String>,
}

impl CaseResult {
    fn error(name: String, tier: &'static str, e: String) -> Self {
        Self {
            name,
            tier,
            m: 0,
            n: 0,
            k: 0,
            err_ref: f64::NAN,
            err_gpu: f64::NAN,
            err_seq: f64::NAN,
            ratio_ref: f64::NAN,
            ratio_seq: f64::NAN,
            max_abs: f64::NAN,
            max_rel: f64::NAN,
            bias_z: f64::NAN,
            verdict: Verdict::Fail,
            interpretation: String::new(),
            error: Some(e),
        }
    }
}

fn score_case(
    name: &str,
    tier: &'static str,
    m: usize,
    n: usize,
    k: usize,
    exact: &[f32],
    reference: &[f32],
    sequential: &[f32],
    gpu: &[f32],
) -> CaseResult {
    assert_eq!(exact.len(), m * n);
    assert_eq!(reference.len(), exact.len());
    assert_eq!(sequential.len(), exact.len());
    assert_eq!(gpu.len(), exact.len());

    let err_ref = rel_l2(reference, exact);
    let err_gpu = rel_l2(gpu, exact);
    let err_seq = rel_l2(sequential, exact);
    let ratio_ref = if err_ref > 0.0 {
        err_gpu / err_ref
    } else if err_gpu == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let ratio_seq = if err_seq > 0.0 {
        err_gpu / err_seq
    } else if err_gpu == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };

    let (max_abs, max_rel, mean_signed, std_signed) = residual_stats(gpu, exact);
    let bias_z = if std_signed > 0.0 {
        mean_signed.abs() / std_signed
    } else {
        0.0
    };
    let count = (m * n) as f64;
    let bias_noise_floor = 1.0 / count.sqrt(); // order of unbiased mean fluctuation

    // ── Vacuous / layout-only ────────────────────────────────────────────
    // If ReferenceOrder itself has zero f32 rounding, the case cannot tell
    // accumulation trees apart. Bit-exact agreement is a layout/indexing
    // check only — label INCONCLUSIVE, never PASS.
    if err_ref < ERR_REF_FLOOR {
        let layout_ok = err_gpu < ERR_REF_FLOOR;
        let verdict = if layout_ok {
            Verdict::Inconclusive
        } else {
            Verdict::Fail
        };
        let interpretation = if layout_ok {
            format!(
                "INCONCLUSIVE layout/indexing check: err_ref=0 and err_gpu=0 — every term \
                 was exactly representable in f32, so no accumulation order was exercised. \
                 Not evidence of semantic equivalence. (bias_z={bias_z:.3})"
            )
        } else {
            format!(
                "FAIL layout/indexing: err_ref=0 but err_gpu={err_gpu:.3e} — GPU disagrees \
                 with Exact even though ReferenceOrder is bit-exact (indexing or dequant bug)"
            )
        };
        return CaseResult {
            name: name.to_string(),
            tier,
            m,
            n,
            k,
            err_ref,
            err_gpu,
            err_seq,
            ratio_ref,
            ratio_seq,
            max_abs,
            max_rel,
            bias_z,
            verdict,
            interpretation,
            error: None,
        };
    }

    // ── Bias is the hard fail ────────────────────────────────────────────
    // Also measure bias of the two CPU trees. Under wide UE8M0 spreads, f32
    // residual is correlated (not white), so GPU bias_z of ~0.05–0.08 is
    // normal and matches the CPU trees. A misplaced scale drives GPU bias_z
    // toward O(1) while the CPU trees stay near zero — that is the defect.
    let (_, _, _, _) = (mean_signed, std_signed, bias_noise_floor, count); // kept for reports
    let (bias_z_ref, mean_ref) = {
        let (_, _, mean, std) = residual_stats(reference, exact);
        (
            if std > 0.0 {
                mean.abs() / std
            } else {
                0.0
            },
            mean,
        )
    };
    let (bias_z_seq, mean_seq) = {
        let (_, _, mean, std) = residual_stats(sequential, exact);
        (
            if std > 0.0 {
                mean.abs() / std
            } else {
                0.0
            },
            mean,
        )
    };
    // Absolute material threshold OR GPU much more biased than both CPU trees.
    let cpu_bias_ceiling = bias_z_ref.max(bias_z_seq).max(bias_noise_floor) * 3.0 + 0.05;
    let bias_limit = BIAS_Z_LIMIT.max(cpu_bias_ceiling);
    let bias_ok = bias_z < bias_limit;

    // ── Magnitude calibration ────────────────────────────────────────────
    let mag_ok;
    let mag_note: String;
    if !err_gpu.is_finite() || err_gpu.is_nan() {
        mag_ok = false;
        mag_note = format!("err_gpu non-finite ({err_gpu})");
    } else if ratio_ref <= RATIO_SOFT {
        mag_ok = true;
        mag_note = format!(
            "ratio_ref={ratio_ref:.2} ≤ {RATIO_SOFT}× (within soft band of ReferenceOrder)"
        );
    } else if ratio_ref <= RATIO_HARD {
        // Must match sequential-f32 magnitude to claim "deeper tree, same op".
        let seq_ok = err_seq > ERR_REF_FLOOR
            && ratio_seq <= SEQ_MATCH
            && ratio_seq >= 1.0 / SEQ_MATCH;
        mag_ok = seq_ok;
        if seq_ok {
            mag_note = format!(
                "ratio_ref={ratio_ref:.2} in ({RATIO_SOFT},{RATIO_HARD}] but matches \
                 sequential-f32 (ratio_seq={ratio_seq:.2}, err_seq={err_seq:.3e}) — \
                 accumulation depth, not a semantic defect"
            );
        } else {
            mag_note = format!(
                "ratio_ref={ratio_ref:.2} elevated AND does not match sequential-f32 \
                 (ratio_seq={ratio_seq:.2}, err_seq={err_seq:.3e}, want within {SEQ_MATCH}×)"
            );
        }
    } else {
        mag_ok = false;
        mag_note = format!(
            "ratio_ref={ratio_ref:.2} > {RATIO_HARD}× hard ceiling vs ReferenceOrder"
        );
    }

    let verdict = if bias_ok && mag_ok {
        Verdict::Pass
    } else {
        Verdict::Fail
    };

    let interpretation = match (bias_ok, mag_ok) {
        (false, _) => format!(
            "SYSTEMATIC BIAS defect: bias_z={bias_z:.3} (limit {bias_limit:.3}; \
             ref_bias_z={bias_z_ref:.3} seq_bias_z={bias_z_seq:.3}); \
             mean_signed={mean_signed:.3e} (ref_mean={mean_ref:.3e} seq_mean={mean_seq:.3e}) — \
             a scale is applied in the wrong place. Magnitude note: {mag_note}"
        ),
        (true, false) => format!(
            "MAGNITUDE defect (unbiased): {mag_note}; bias_z={bias_z:.3} \
             (ref={bias_z_ref:.3} seq={bias_z_seq:.3}) OK"
        ),
        (true, true) => format!(
            "PASS: unbiased bias_z={bias_z:.3} (ref={bias_z_ref:.3} seq={bias_z_seq:.3}, \
             limit {bias_limit:.3}); {mag_note}; max|GPU−Exact|={max_abs:.3e}"
        ),
    };

    CaseResult {
        name: name.to_string(),
        tier,
        m,
        n,
        k,
        err_ref,
        err_gpu,
        err_seq,
        ratio_ref,
        ratio_seq,
        max_abs,
        max_rel,
        bias_z,
        verdict,
        interpretation,
        error: None,
    }
}

fn print_case(c: &CaseResult) {
    if let Some(e) = &c.error {
        println!("    FAIL  ERROR: {e}");
        return;
    }
    println!(
        "    {}  err_ref={:.6e}  err_gpu={:.6e}  err_seq={:.6e}  \
         ratio_ref={:.3}  ratio_seq={:.3}  max|d|={:.6e}  bias_z={:.4}",
        c.verdict.as_str(),
        c.err_ref,
        c.err_gpu,
        c.err_seq,
        c.ratio_ref,
        c.ratio_seq,
        c.max_abs,
        c.bias_z
    );
    println!("    → {}", c.interpretation);
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = f64::from(x) - f64::from(y);
        num += dx * dx;
        den += f64::from(y) * f64::from(y);
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        num.sqrt() / den.sqrt()
    }
}

fn residual_stats(a: &[f32], b: &[f32]) -> (f64, f64, f64, f64) {
    let n = a.len() as f64;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = f64::from(x) - f64::from(y);
        let ad = d.abs();
        if ad > max_abs {
            max_abs = ad;
        }
        let ay = f64::from(y).abs();
        if ay > 0.0 {
            let r = ad / ay;
            if r > max_rel {
                max_rel = r;
            }
        }
        sum += d;
        sumsq += d * d;
    }
    let mean = sum / n;
    let var = (sumsq / n) - mean * mean;
    let std = var.max(0.0).sqrt();
    (max_abs, max_rel, mean, std)
}

// ---------------------------------------------------------------------------
// Synthetic generators
// ---------------------------------------------------------------------------

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    fn next_f32_unit(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }
    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32_unit() * 2.0 - 1.0
    }
    fn next_byte(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }
}

fn gen_e4m3_codes(rng: &mut XorShift64, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        loop {
            let b = rng.next_byte();
            if b == 0x7f || b == 0xff {
                continue;
            }
            out.push(b);
            break;
        }
    }
    out
}

fn gen_e2m1_packed(rng: &mut XorShift64, n_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bytes);
    for _ in 0..n_bytes {
        out.push(rng.next_byte());
    }
    out
}

/// Mild UE8M0 exponents clustered around 2^-4 .. 2^4.
fn gen_ue8m0_mild(rng: &mut XorShift64, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let e = 123u8 + (rng.next_u32() % 9) as u8;
        out.push(e);
    }
    out
}

/// Wide UE8M0 dynamic range — 2^-20 .. 2^10. Spreads partial sums across
/// many binades so f32 addition actually loses bits (anti-vacuous).
fn gen_ue8m0_wide(rng: &mut XorShift64, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // 107..=137 → 2^-20 .. 2^10
        let e = 107u8 + (rng.next_u32() % 31) as u8;
        out.push(e.min(254));
    }
    out
}

/// BF16-representable activations. Wide mode mixes magnitudes so act-quant
/// block scales also span multiple exponents.
fn gen_acts_bf16(rng: &mut XorShift64, n: usize, smode: ScaleMode) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = match smode {
            ScaleMode::Mild => {
                rng.next_f32_signed() * if rng.next_u32() % 8 == 0 { 8.0 } else { 1.5 }
            }
            ScaleMode::Wide => {
                // Spatially varying magnitude along K so neighbouring act-scale
                // groups land in different binades after quant.
                let binade = ((i % 16) as i32) - 8; // 2^-8 .. 2^7
                let scale = 2.0f32.powi(binade);
                rng.next_f32_signed() * scale
            }
        };
        out.push(round_to_bf16(v));
    }
    out
}

// ---------------------------------------------------------------------------
// Host↔device helpers
// ---------------------------------------------------------------------------

fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        let sign = ((bits >> 16) & 0x8000) as u16;
        return sign | 0x7fc0;
    }
    let lsb = (bits >> 16) & 1;
    let lower = bits & 0xffff;
    let round_bit = (lower >> 15) & 1;
    let sticky = if (lower & 0x7fff) != 0 { 1 } else { 0 };
    let mut top = bits >> 16;
    if round_bit == 1 && (sticky == 1 || lsb == 1) {
        top = top.wrapping_add(1);
    }
    top as u16
}

fn pack_f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

fn upload_bf16(gpu: &mut Gpu, bytes: &[u8], shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(shape, DType::BF16)
        .map_err(|e| format!("deepseek4 parent: alloc bf16: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("deepseek4 parent: upload bf16: {e:?}"))?;
    Ok(t)
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<f32>, String> {
    let mut data = vec![0f32; n];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &t.buf)
        .map_err(|e| format!("deepseek4 parent: download f32: {e:?}"))?;
    Ok(data)
}

// ---------------------------------------------------------------------------
// Safetensors shard reader
// ---------------------------------------------------------------------------

fn load_from_shard(
    shard_path: &Path,
    tensor_name: &str,
) -> Result<(String, Vec<usize>, Vec<u8>), String> {
    let data = fs::read(shard_path).map_err(|e| {
        format!(
            "deepseek4 parent: read shard {}: {e}",
            shard_path.display()
        )
    })?;
    if data.len() < 8 {
        return Err("deepseek4 parent: shard too small".into());
    }
    let hdr_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if 8 + hdr_len > data.len() {
        return Err("deepseek4 parent: header overruns file".into());
    }
    let hdr: serde_json::Value = serde_json::from_slice(&data[8..8 + hdr_len])
        .map_err(|e| format!("deepseek4 parent: header json: {e}"))?;
    let meta = hdr.get(tensor_name).ok_or_else(|| {
        format!(
            "deepseek4 parent: tensor {tensor_name} not in {}",
            shard_path.display()
        )
    })?;
    let dtype = meta
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "deepseek4 parent: missing dtype".to_string())?
        .to_string();
    let shape: Vec<usize> = meta
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "deepseek4 parent: missing shape".to_string())?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let offs = meta
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "deepseek4 parent: missing data_offsets".to_string())?;
    let o0 = offs[0].as_u64().unwrap() as usize;
    let o1 = offs[1].as_u64().unwrap() as usize;
    let body = &data[8 + hdr_len..];
    if o1 > body.len() {
        return Err(format!(
            "deepseek4 parent: offset {o1} past body {}",
            body.len()
        ));
    }
    Ok((dtype, shape, body[o0..o1].to_vec()))
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    model: String,
    manifest: Option<PathBuf>,
    skip_shard_hashes: bool,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut skip_shard_hashes = false;
    let mut seed = DEFAULT_SEED;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model = Some(
                    args.next()
                        .ok_or_else(|| "flag --model missing value".to_string())?,
                );
            }
            "--manifest" => {
                let p = args
                    .next()
                    .ok_or_else(|| "flag --manifest missing value".to_string())?;
                manifest = Some(PathBuf::from(p));
            }
            "--skip-shard-hashes" => skip_shard_hashes = true,
            "--seed" => {
                let v = args
                    .next()
                    .ok_or_else(|| "flag --seed missing value".to_string())?;
                seed = if let Some(hex) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
                    u64::from_str_radix(hex, 16)
                        .map_err(|e| format!("--seed hex parse: {e}"))?
                } else {
                    v.parse::<u64>()
                        .map_err(|e| format!("--seed parse: {e}"))?
                };
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_linear_gate --model <dir> \
                     [--manifest out/manifest.json] [--skip-shard-hashes] [--seed N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    Ok(Args {
        model,
        manifest,
        skip_shard_hashes,
        seed,
    })
}

// ---------------------------------------------------------------------------
// Manifest source pinning (mirrors inventory_gate)
// ---------------------------------------------------------------------------

fn build_source_info(root: &Path, skip_shard_hashes: bool) -> Result<SourceInfo, String> {
    let root_str = root
        .to_str()
        .ok_or_else(|| "deepseek4 parent: model root is not valid UTF-8".to_string())?
        .to_string();

    let config_path = root.join("config.json");
    let index_path = root.join("model.safetensors.index.json");
    let tokenizer_path = root.join("tokenizer.json");

    let config_sha256 = sha256_file(&config_path)?;
    let index_sha256 = if index_path.is_file() {
        sha256_file(&index_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing index file {}",
            index_path.display()
        ));
    };
    let tokenizer_sha256 = if tokenizer_path.is_file() {
        sha256_file(&tokenizer_path)?
    } else {
        return Err(format!(
            "deepseek4 parent: missing tokenizer.json at {}",
            tokenizer_path.display()
        ));
    };

    let mut shard_paths: Vec<PathBuf> = fs::read_dir(root)
        .map_err(|e| format!("deepseek4 parent: read_dir {}: {e}", root.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err("deepseek4 parent: no .safetensors shards found".into());
    }

    let hash_t0 = Instant::now();
    let mut shards = Vec::with_capacity(shard_paths.len());
    for (i, p) in shard_paths.iter().enumerate() {
        let meta = fs::metadata(p).map_err(|e| {
            format!(
                "deepseek4 parent: metadata {}: {e}",
                p.display()
            )
        })?;
        let bytes = meta.len();
        let file = p
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("deepseek4 parent: bad shard name {}", p.display()))?
            .to_string();
        let sha256 = if skip_shard_hashes {
            format!("SKIPPED_SHARD_HASH_{i:02}")
        } else {
            let t0 = Instant::now();
            let h = sha256_file(p)?;
            println!(
                "  hashed {file} ({bytes} bytes) in {:.1} s → {}",
                t0.elapsed().as_secs_f64(),
                &h[..16]
            );
            h
        };
        shards.push(ShardInfo {
            file,
            sha256,
            bytes,
        });
    }
    if !skip_shard_hashes {
        println!(
            "hashed {} shards in {:.1} s",
            shards.len(),
            hash_t0.elapsed().as_secs_f64()
        );
    } else {
        println!(
            "SKIPPED hashing {} shards (--skip-shard-hashes); placeholders are not a pin",
            shards.len()
        );
    }

    Ok(SourceInfo {
        root: root_str,
        index_sha256,
        shards,
        config_sha256,
        tokenizer_sha256,
    })
}

fn utc_now_rfc3339() -> String {
    if let Ok(out) = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}
