// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! build_kld_ref_hipfire — first-party KLD reference producer.
//!
//! Loads a Hipfire HFQ model, tokenizes a slice, runs direct Hipfire forward
//! passes, top-K-reduces reference log-probs, and writes a metadata-rich
//! HFQM `.kldref.hfq` package consumed by `eval_hipfire`.
//!
//! This is intentionally separate from `build_kld_ref.rs`, which preserves
//! the historical llama.cpp BF16-GGUF reference path for cross-engine checks.
//!
//! Usage:
//!   cargo run --release --features deltanet -p hipfire-runtime \
//!     --example build_kld_ref_hipfire -- \
//!       --model ~/.hipfire/models/qwen3.5-0.8b-bf16.hfq \
//!       --slice benchmarks/quality-baselines/slice/slice.txt \
//!       --top-k 256 \
//!       --output ~/.hipfire/eval-results/refs/qwen3.5-0.8b-bf16.kldref.hfq \
//!       --n-ctx 2048 --kv-mode fp32

#![recursion_limit = "256"]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::{
        qwen35::{
            self, DeltaNetState, PrefillBatchScratch, Qwen35Config, Qwen35Scratch, Qwen35Weights,
        },
        speculative::{KvMode, ModelSlot, ModelSlotConfig},
    };
    use hipfire_runtime::hfq::{
        write_hfqm_package_from_files, HfqPackageWriteEntry, HFQM_ARCH_NON_WEIGHT_PACKAGE,
    };
    use hipfire_runtime::llama::weight_gemv;
    use hipfire_runtime::llama::KvCache;
    use rayon::prelude::*;
    use rdna_compute::{DType, Gpu, GpuTensor};
    use serde_json::json;
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;
    use std::fs::File;
    use std::hash::Hasher;
    use std::io::{BufReader, BufWriter, Read, Write};
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::Instant;
    use twox_hash::XxHash64;

    const KLDREF_SCHEMA_VERSION: u32 = 1;
    const KLDREF_ENTRY_QUANT_TYPE: u8 = 0;

    #[derive(Debug)]
    struct Args {
        model: PathBuf,
        slice: PathBuf,
        top_k: usize,
        output: PathBuf,
        n_ctx: usize,
        max_chunks: Option<usize>,
        kv_mode: KvMode,
        max_seq: Option<usize>,
        metadata_json: Option<PathBuf>,
    }

    fn print_usage() {
        eprintln!(
            "Usage:\n  build_kld_ref_hipfire --model <model.hfq> --slice <slice.txt> --top-k <N> --output <out.kldref.hfq> \\\n                         [--n-ctx <N>=2048] [--max-chunks N] [--kv-mode fp32|q8|asym4|asym3|asym2] [--max-seq N] [--metadata-json path]"
        );
    }

    fn parse_kv_mode(raw: &str) -> KvMode {
        match raw {
            "fp32" | "f32" => KvMode::Fp32,
            "q8" | "" => KvMode::Q8,
            "asym4" | "turbo4" => KvMode::Asym4,
            "asym3" | "turbo3" | "turbo" => KvMode::Asym3,
            "asym2" | "turbo2" => KvMode::Asym2,
            other => {
                eprintln!("unknown --kv-mode {other}; expected fp32|q8|asym4|asym3|asym2");
                std::process::exit(1);
            }
        }
    }

    fn parse_args() -> Args {
        let mut model: Option<PathBuf> = None;
        let mut slice: Option<PathBuf> = None;
        let mut top_k: usize = 256;
        let mut output: Option<PathBuf> = None;
        let mut n_ctx: usize = 2048;
        let mut max_chunks: Option<usize> = None;
        let mut kv_mode = KvMode::Fp32;
        let mut max_seq: Option<usize> = None;
        let mut metadata_json: Option<PathBuf> = None;

        let argv: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--model" => {
                    model = Some(PathBuf::from(&argv[i + 1]));
                    i += 2;
                }
                "--slice" => {
                    slice = Some(PathBuf::from(&argv[i + 1]));
                    i += 2;
                }
                "--top-k" => {
                    top_k = argv[i + 1].parse().expect("--top-k must be integer");
                    i += 2;
                }
                "--output" | "--out" => {
                    output = Some(PathBuf::from(&argv[i + 1]));
                    i += 2;
                }
                "--n-ctx" => {
                    n_ctx = argv[i + 1].parse().expect("--n-ctx must be integer");
                    i += 2;
                }
                "--max-chunks" => {
                    max_chunks = Some(argv[i + 1].parse().expect("--max-chunks must be integer"));
                    i += 2;
                }
                "--kv-mode" | "--kv" => {
                    kv_mode = parse_kv_mode(&argv[i + 1]);
                    i += 2;
                }
                "--max-seq" => {
                    max_seq = Some(argv[i + 1].parse().expect("--max-seq must be integer"));
                    i += 2;
                }
                "--metadata-json" => {
                    metadata_json = Some(PathBuf::from(&argv[i + 1]));
                    i += 2;
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    print_usage();
                    std::process::exit(1);
                }
            }
        }

        let model = model.unwrap_or_else(|| {
            print_usage();
            std::process::exit(1);
        });
        let slice = slice.unwrap_or_else(|| {
            print_usage();
            std::process::exit(1);
        });
        let output = output.unwrap_or_else(|| {
            print_usage();
            std::process::exit(1);
        });
        if n_ctx < 4 {
            eprintln!("--n-ctx must be >= 4");
            std::process::exit(1);
        }
        if top_k == 0 {
            eprintln!("--top-k must be > 0");
            std::process::exit(1);
        }
        Args {
            model,
            slice,
            top_k,
            output,
            n_ctx,
            max_chunks,
            kv_mode,
            max_seq,
            metadata_json,
        }
    }

    fn print_profile_summary(entries: &[rdna_compute::profile::ProfileEntry]) {
        #[derive(Default)]
        struct Agg {
            calls: usize,
            total_us: f64,
            total_bytes: usize,
        }

        let mut by_kernel: std::collections::BTreeMap<(&'static str, &'static str), Agg> =
            std::collections::BTreeMap::new();
        let mut total_us = 0.0f64;
        for entry in entries {
            let agg = by_kernel.entry((entry.category, entry.kernel)).or_default();
            agg.calls += 1;
            agg.total_us += entry.time_us;
            agg.total_bytes += entry.bytes;
            total_us += entry.time_us;
        }

        let mut rows: Vec<_> = by_kernel.into_iter().collect();
        rows.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

        eprintln!(
            "\n=== KLD PROFILE ({} kernel calls, {:.1}ms total timed kernel work) ===",
            entries.len(),
            total_us / 1000.0
        );
        eprintln!(
            "  {:<4} {:<10} {:<48} {:>8} {:>10} {:>10} {:>7} {:>10}",
            "rank", "category", "kernel", "calls", "total_ms", "us/call", "%", "MiB"
        );
        for (rank, ((category, kernel), agg)) in rows.iter().take(24).enumerate() {
            let avg_us = agg.total_us / agg.calls as f64;
            let pct = if total_us > 0.0 {
                agg.total_us * 100.0 / total_us
            } else {
                0.0
            };
            let mib = agg.total_bytes as f64 / (1024.0 * 1024.0);
            eprintln!(
                "  {:<4} {:<10} {:<48} {:>8} {:>9.2} {:>10.2} {:>6.1} {:>10.1}",
                rank + 1,
                category,
                kernel,
                agg.calls,
                agg.total_us / 1000.0,
                avg_us,
                pct,
                mib
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_kld_prefill(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KvCache,
        dn_state: &mut DeltaNetState,
        scratch: &Qwen35Scratch,
        pbs: Option<&PrefillBatchScratch>,
        hidden_out: &GpuTensor,
        graph_key: usize,
        use_graph: bool,
    ) -> hip_bridge::HipResult<()> {
        if !use_graph {
            return qwen35::forward_prefill_batch_with_pbs_opts(
                gpu,
                weights,
                config,
                tokens,
                start_pos,
                kv_cache,
                dn_state,
                scratch,
                None,
                Some(hidden_out),
                None,
                None,
                pbs,
                None,
                None,
                false,
            );
        }

        let pbs = pbs.expect("KLD graph prefill requires PrefillBatchScratch");
        if tokens.len() > pbs.max_batch {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "KLD graph prefill requires tokens.len={} <= pbs.max_batch={}. \
                     Set HIPFIRE_PREFILL_MAX_BATCH before model load or reduce --n-ctx.",
                    tokens.len(),
                    pbs.max_batch
                ),
            ));
        }
        qwen35::upload_prefill_batch_inputs(gpu, pbs, tokens, start_pos)?;
        let skip_active_stream_debug = std::env::var("HIPFIRE_KLD_NO_ACTIVE_STREAM")
            .ok()
            .as_deref()
            == Some("1");
        if gpu.active_stream.is_none() && !skip_active_stream_debug {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }

        if gpu.verify_has_graph(graph_key) {
            return gpu.verify_graph_launch(graph_key);
        }
        if gpu.verify_needs_warmup(graph_key) {
            gpu.verify_mark_warmup_done(graph_key);
            return qwen35::forward_prefill_batch_single_chunk_captured_opts(
                gpu,
                weights,
                config,
                tokens,
                start_pos,
                kv_cache,
                dn_state,
                scratch,
                pbs,
                None,
                Some(hidden_out),
                None,
                None,
                false,
            );
        }

        gpu.begin_verify_graph_capture(graph_key)?;
        let result = qwen35::forward_prefill_batch_single_chunk_captured_opts(
            gpu,
            weights,
            config,
            tokens,
            start_pos,
            kv_cache,
            dn_state,
            scratch,
            pbs,
            None,
            Some(hidden_out),
            None,
            None,
            false,
        );
        if let Err(err) = result {
            gpu.capture_mode = false;
            let _ = gpu
                .hip
                .stream_end_capture(gpu.active_stream.as_ref().unwrap());
            return Err(err);
        }
        let blob_count = gpu.capture_blobs.len();
        gpu.end_verify_graph_capture()?;
        gpu.verify_graph_launch(graph_key)?;
        eprintln!(
            "build_kld_ref_hipfire: captured prefill graph key={} tokens={} blobs={} cache_size={}",
            graph_key,
            tokens.len(),
            blob_count,
            gpu.verify_graph_count()
        );
        Ok(())
    }

    fn command_hash(cmd: &str, path: &Path) -> Option<String> {
        let out = std::process::Command::new(cmd).arg(path).output().ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .next()
            .map(String::from)
    }

    #[derive(Debug)]
    struct SourceModelHash {
        xxh64: Option<String>,
        sha256: Option<String>,
    }

    fn xxh64_file(path: &Path) -> Option<String> {
        let file = File::open(path).ok()?;
        let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);
        let mut buf = vec![0u8; 4 * 1024 * 1024];
        let mut h = XxHash64::with_seed(0);
        loop {
            let n = reader.read(&mut buf).ok()?;
            if n == 0 {
                break;
            }
            h.write(&buf[..n]);
        }
        Some(format!("{:016x}", h.finish()))
    }

    fn spawn_source_model_hash(
        path: PathBuf,
        include_sha256: bool,
    ) -> thread::JoinHandle<SourceModelHash> {
        thread::spawn(move || {
            let xxh64 = xxh64_file(&path);
            let sha256 = if include_sha256 {
                command_hash("sha256sum", &path)
            } else {
                None
            };
            SourceModelHash { xxh64, sha256 }
        })
    }

    fn join_source_model_hash(handle: thread::JoinHandle<SourceModelHash>) -> SourceModelHash {
        handle.join().unwrap_or(SourceModelHash {
            xxh64: None,
            sha256: None,
        })
    }

    #[derive(Clone, Copy, Debug)]
    struct HeapCandidate {
        logit: f32,
        idx: u32,
    }

    impl PartialEq for HeapCandidate {
        fn eq(&self, other: &Self) -> bool {
            self.idx == other.idx && self.logit.to_bits() == other.logit.to_bits()
        }
    }

    impl Eq for HeapCandidate {}

    impl PartialOrd for HeapCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HeapCandidate {
        fn cmp(&self, other: &Self) -> Ordering {
            // BinaryHeap is max-first. Reverse the ordering so `peek()` is the
            // smallest retained logit and can be replaced by a better one.
            other
                .logit
                .total_cmp(&self.logit)
                .then_with(|| other.idx.cmp(&self.idx))
        }
    }

    #[derive(Debug)]
    struct StreamingTopK {
        top_k: usize,
        max_logit: f32,
        sum_exp: f64,
        heap: BinaryHeap<HeapCandidate>,
    }

    impl StreamingTopK {
        fn new(top_k: usize) -> Self {
            Self {
                top_k,
                max_logit: f32::NEG_INFINITY,
                sum_exp: 0.0,
                heap: BinaryHeap::with_capacity(top_k),
            }
        }

        fn update_tile(&mut self, logits: &[f32], global_start: usize) {
            for (local_idx, &logit) in logits.iter().enumerate() {
                self.update_logsumexp(logit, 1.0);

                let cand = HeapCandidate {
                    logit,
                    idx: (global_start + local_idx) as u32,
                };
                self.update_candidate(cand);
            }
        }

        fn update_logsumexp(&mut self, chunk_max: f32, chunk_sum_exp: f32) {
            if !chunk_max.is_finite() || chunk_sum_exp <= 0.0 {
                return;
            }
            let chunk_sum_exp = chunk_sum_exp as f64;
            if self.max_logit == f32::NEG_INFINITY {
                self.max_logit = chunk_max;
                self.sum_exp = chunk_sum_exp;
            } else if chunk_max > self.max_logit {
                self.sum_exp =
                    self.sum_exp * ((self.max_logit - chunk_max) as f64).exp() + chunk_sum_exp;
                self.max_logit = chunk_max;
            } else {
                self.sum_exp += chunk_sum_exp * ((chunk_max - self.max_logit) as f64).exp();
            }
        }

        fn update_candidate(&mut self, cand: HeapCandidate) {
            if self.heap.len() < self.top_k {
                self.heap.push(cand);
            } else if let Some(&min_kept) = self.heap.peek() {
                if cand.logit > min_kept.logit
                    || (cand.logit == min_kept.logit && cand.idx < min_kept.idx)
                {
                    self.heap.pop();
                    self.heap.push(cand);
                }
            }
        }

        fn update_candidates(&mut self, indices: &[i32], logits: &[f32]) {
            for (&idx, &logit) in indices.iter().zip(logits.iter()) {
                if idx < 0 || !logit.is_finite() {
                    continue;
                }
                self.update_candidate(HeapCandidate {
                    logit,
                    idx: idx as u32,
                });
            }
        }

        fn write_final(self, indices_out: &mut [u32], log_probs_out: &mut [f32]) -> f32 {
            let log_z = (self.max_logit as f64) + self.sum_exp.ln();
            let mut top = self.heap.into_vec();
            top.sort_by(|a, b| b.logit.total_cmp(&a.logit).then_with(|| a.idx.cmp(&b.idx)));

            let mut top_p_sum = 0.0f64;
            for (i, cand) in top.iter().enumerate() {
                let log_p = ((cand.logit as f64) - log_z) as f32;
                indices_out[i] = cand.idx;
                log_probs_out[i] = log_p;
                top_p_sum += (log_p as f64).exp();
            }
            for i in top.len()..self.top_k {
                indices_out[i] = 0;
                log_probs_out[i] = f32::NEG_INFINITY;
            }
            (1.0 - top_p_sum).max(0.0) as f32
        }
    }

    fn log_softmax_top_k_row(
        logits: &[f32],
        top_k: usize,
        indices_out: &mut [u32],
        log_probs_out: &mut [f32],
    ) -> f32 {
        let k = top_k.min(logits.len());
        let mut max_logit = f32::NEG_INFINITY;
        let mut heap: BinaryHeap<HeapCandidate> = BinaryHeap::with_capacity(k);
        for (idx, &logit) in logits.iter().enumerate() {
            if logit > max_logit {
                max_logit = logit;
            }
            let cand = HeapCandidate {
                logit,
                idx: idx as u32,
            };
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(&min_kept) = heap.peek() {
                if logit > min_kept.logit {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }

        let mut sum_exp = 0.0f64;
        for &v in logits {
            sum_exp += ((v - max_logit) as f64).exp();
        }
        let log_z = (max_logit as f64) + sum_exp.ln();

        let mut top = heap.into_vec();
        top.sort_by(|a, b| b.logit.total_cmp(&a.logit).then_with(|| a.idx.cmp(&b.idx)));

        let mut top_p_sum = 0.0f64;
        for (i, cand) in top.iter().enumerate() {
            let log_p = ((cand.logit as f64) - log_z) as f32;
            indices_out[i] = cand.idx;
            log_probs_out[i] = log_p;
            top_p_sum += (log_p as f64).exp();
        }
        for i in top.len()..top_k {
            indices_out[i] = 0;
            log_probs_out[i] = f32::NEG_INFINITY;
        }
        (1.0 - top_p_sum).max(0.0) as f32
    }

    fn log_softmax_top_k(
        logits: &[f32],
        top_k: usize,
        indices_out: &mut BufWriter<File>,
        log_probs_out: &mut BufWriter<File>,
        residual_out: &mut BufWriter<File>,
    ) {
        let k = top_k.min(logits.len());
        let mut max_logit = f32::NEG_INFINITY;
        for &v in logits {
            if v > max_logit {
                max_logit = v;
            }
        }
        let mut sum_exp = 0.0f64;
        for &v in logits {
            sum_exp += ((v - max_logit) as f64).exp();
        }
        let log_z = (max_logit as f64) + sum_exp.ln();

        let mut top: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &logit)| (idx as u32, ((logit as f64) - log_z) as f32))
            .collect();
        let cmp_desc =
            |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
        if k < top.len() {
            top.select_nth_unstable_by(k - 1, cmp_desc);
        }
        top[..k].sort_by(cmp_desc);

        let top_p_sum: f64 = top[..k].iter().map(|&(_, lp)| (lp as f64).exp()).sum();
        let sum_p_residual = (1.0 - top_p_sum).max(0.0) as f32;

        for &(idx, _) in &top[..k] {
            indices_out.write_all(&idx.to_le_bytes()).unwrap();
        }
        for &(_, log_p) in &top[..k] {
            log_probs_out.write_all(&log_p.to_le_bytes()).unwrap();
        }
        residual_out
            .write_all(&sum_p_residual.to_le_bytes())
            .unwrap();
    }

    fn write_u32_slice(out: &mut BufWriter<File>, values: &[u32]) {
        for value in values {
            out.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    fn write_f32_slice(out: &mut BufWriter<File>, values: &[f32]) {
        for value in values {
            out.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    fn download_i32(gpu: &rdna_compute::Gpu, tensor: &rdna_compute::GpuTensor) -> Vec<i32> {
        gpu.bind_thread().expect("bind gpu for i32 download");
        let mut values = vec![0i32; tensor.numel()];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, values.len() * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &tensor.buf)
            .expect("download i32 tensor");
        values
    }

    fn write_metadata(path: &Path, value: serde_json::Value) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("create metadata parent");
            }
        }
        std::fs::write(path, serde_json::to_string_pretty(&value).unwrap())
            .expect("write metadata json");
    }

    let args = parse_args();
    if !args.model.exists() {
        eprintln!("--model path not found: {}", args.model.display());
        std::process::exit(1);
    }
    if !args.slice.exists() {
        eprintln!("--slice path not found: {}", args.slice.display());
        std::process::exit(1);
    }
    hipfire_runtime::eval_common::verify_slice_md5(&args.slice, "build_kld_ref_hipfire");

    let use_kld_graph = std::env::var("HIPFIRE_KLD_GRAPH").ok().as_deref() != Some("0");

    // Eval references should not silently include prompt-shape adaptation or
    // graph capture variance. Match `eval_hipfire`'s prompt determinism
    // default; KLD prefill capture is default-on for long reference builds
    // and can be disabled with HIPFIRE_KLD_GRAPH=0 for debugging.
    let use_kld_direct_f16kv_attn = match std::env::var("HIPFIRE_KLD_DIRECT_WMMA_ATTN")
        .or_else(|_| std::env::var("HIPFIRE_KLD_DIRECT_F16KV_ATTN"))
        .ok()
        .as_deref()
    {
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES") => true,
        _ => false,
    };
    let use_kld_fp32_gqa4_attn = match std::env::var("HIPFIRE_KLD_FP32_GQA4_ATTN").ok().as_deref() {
        Some("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO") => false,
        _ => true,
    };
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        if use_kld_direct_f16kv_attn {
            std::env::set_var("HIPFIRE_KLD_DIRECT_WMMA_ATTN", "1");
            std::env::set_var("HIPFIRE_KLD_DIRECT_F16KV_ATTN", "1");
        }
        if use_kld_fp32_gqa4_attn {
            std::env::set_var("HIPFIRE_KLD_FP32_GQA4_ATTN", "1");
        }
        if use_kld_graph {
            std::env::set_var("HIPFIRE_PREFILL_REUSE_PBS", "1");
            if std::env::var_os("HIPFIRE_PREFILL_MAX_BATCH").is_none() {
                std::env::set_var("HIPFIRE_PREFILL_MAX_BATCH", args.n_ctx.to_string());
            }
        } else {
            std::env::set_var("HIPFIRE_GRAPH", "0");
        }
    }

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!(
        "build_kld_ref_hipfire: loading {} on {}",
        args.model.display(),
        gpu.arch
    );
    let max_seq = args.max_seq.unwrap_or(args.n_ctx + 16);
    let cfg = ModelSlotConfig {
        max_seq,
        kv_mode: args.kv_mode,
        repeat_window: 128,
        state_quant: hipfire_arch_qwen35::qwen35::StateQuant::FP32,
    };
    let mut slot =
        ModelSlot::load(&mut gpu, &args.model, "ref", cfg).expect("failed to load model slot");
    let tokenizer = slot.load_tokenizer().expect("failed to load tokenizer");

    let slice_text = std::fs::read_to_string(&args.slice).expect("read slice");
    let all_tokens = tokenizer.encode(&slice_text);
    let available_chunks = all_tokens.len() / args.n_ctx;
    let n_chunk = args
        .max_chunks
        .map(|m| m.min(available_chunks))
        .unwrap_or(available_chunks);
    if n_chunk == 0 {
        eprintln!(
            "not enough tokens for one chunk: slice_tokens={} n_ctx={}",
            all_tokens.len(),
            args.n_ctx
        );
        std::process::exit(1);
    }
    let tokens = &all_tokens[..n_chunk * args.n_ctx];
    let scored_per_chunk = args.n_ctx - 1 - args.n_ctx / 2;
    let total_scored = scored_per_chunk * n_chunk;
    let top_k = args.top_k.min(slot.config.vocab_size);

    eprintln!(
        "build_kld_ref_hipfire: slice_tokens={} n_ctx={} n_chunk={} scored/chunk={} total_scored={} top_k={} kv_mode={:?}",
        all_tokens.len(),
        args.n_ctx,
        n_chunk,
        scored_per_chunk,
        total_scored,
        top_k,
        args.kv_mode
    );
    if use_kld_graph {
        eprintln!(
            "build_kld_ref_hipfire: graph prefill enabled with HIPFIRE_PREFILL_MAX_BATCH={}",
            std::env::var("HIPFIRE_PREFILL_MAX_BATCH").unwrap_or_else(|_| "<unset>".to_string())
        );
    }
    if use_kld_direct_f16kv_attn {
        eprintln!(
            "build_kld_ref_hipfire: direct causal WMMA attention enabled for FP32-KV prefill chunks"
        );
    } else if use_kld_fp32_gqa4_attn {
        eprintln!(
            "build_kld_ref_hipfire: direct FP32 GQA4 attention enabled for eligible KLD prefill chunks"
        );
    }
    if all_tokens.len() % args.n_ctx != 0 {
        eprintln!(
            "build_kld_ref_hipfire: dropping tail of {} token(s)",
            all_tokens.len() - tokens.len()
        );
    }

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output parent");
        }
    }
    let temp_stem = format!(
        ".{}.{}.kldref",
        args.output
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("output"),
        std::process::id()
    );
    let temp_dir = args
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let tokens_path = temp_dir.join(format!("{temp_stem}.tokens.tmp"));
    let indices_path = temp_dir.join(format!("{temp_stem}.top_indices.tmp"));
    let log_probs_path = temp_dir.join(format!("{temp_stem}.top_log_probs.tmp"));
    let residual_path = temp_dir.join(format!("{temp_stem}.residual_mass.tmp"));

    let mut tokens_out = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&tokens_path).expect("create tokens temp"),
    );
    for &token in tokens {
        tokens_out.write_all(&token.to_le_bytes()).unwrap();
    }
    tokens_out.flush().unwrap();
    drop(tokens_out);

    let mut indices_out = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&indices_path).expect("create top_indices temp"),
    );
    let mut log_probs_out = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&log_probs_path).expect("create top_log_probs temp"),
    );
    let mut residual_out = BufWriter::with_capacity(
        4 * 1024 * 1024,
        File::create(&residual_path).expect("create residual_mass temp"),
    );

    let started = Instant::now();
    let progress_interval = (total_scored / 100).max(1);
    let scoring_start = args.n_ctx / 2;
    let full_hidden_buf = gpu
        .alloc_tensor(&[args.n_ctx - 1, slot.config.dim], DType::F32)
        .expect("alloc full_hidden_buf");
    let hidden_buf = full_hidden_buf.sub_offset(
        scoring_start * slot.config.dim,
        scored_per_chunk * slot.config.dim,
    );
    let include_source_sha256 =
        std::env::var("HIPFIRE_KLD_SOURCE_SHA256").ok().as_deref() == Some("1");
    let source_model_hash = spawn_source_model_hash(args.model.clone(), include_source_sha256);
    let mut scored_done = 0usize;
    let do_profile = std::env::var("HIPFIRE_PROFILE").ok().as_deref() == Some("1");
    let prefill_only_debug = std::env::var("HIPFIRE_KLD_PREFILL_ONLY").ok().as_deref() == Some("1");
    for chunk_idx in 0..n_chunk {
        slot.reset_state(&mut gpu);
        let chunk = &tokens[chunk_idx * args.n_ctx..(chunk_idx + 1) * args.n_ctx];
        let profile_this_chunk = do_profile && chunk_idx == 0;
        if profile_this_chunk {
            rdna_compute::profile::start();
        }

        let t_prefill = Instant::now();
        forward_kld_prefill(
            &mut gpu,
            &slot.weights,
            &slot.config,
            &chunk[0..(args.n_ctx - 1)],
            0,
            &mut slot.kv_cache,
            &mut slot.dn_state,
            &slot.scratch,
            slot.scratch.prefill_batch.as_ref(),
            &full_hidden_buf,
            0x4b1d_0000usize,
            use_kld_graph,
        )
        .expect("forward_prefill_batch kld chunk");
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        if prefill_only_debug {
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip
                    .stream_synchronize(stream)
                    .expect("prefill-only debug stream sync");
            } else {
                gpu.hip
                    .device_synchronize()
                    .expect("prefill-only debug device sync");
            }
            eprintln!(
                "build_kld_ref_hipfire: prefill-only debug exit after chunk {} prefill={:.1}ms",
                chunk_idx + 1,
                prefill_ms
            );
            return;
        }

        let batched_logits = if slot.weights.output.gpu_dtype == DType::F16 {
            let logits = gpu
                .alloc_tensor(&[scored_per_chunk, slot.config.vocab_size], DType::F32)
                .expect("alloc batched lm_head logits");
            gpu.gemm_f16_batched_lmhead(
                &slot.weights.output.buf,
                &hidden_buf,
                &logits,
                slot.config.vocab_size,
                slot.config.dim,
                scored_per_chunk,
            )
            .expect("gemm_f16_batched_lmhead");
            Some(logits)
        } else {
            None
        };

        if let Some(ref all_logits) = batched_logits {
            let logits = gpu
                .download_f32(all_logits)
                .expect("download batched logits");
            let mut chunk_indices = vec![0u32; scored_per_chunk * top_k];
            let mut chunk_log_probs = vec![0.0f32; scored_per_chunk * top_k];
            let mut chunk_residuals = vec![0.0f32; scored_per_chunk];
            chunk_indices
                .par_chunks_mut(top_k)
                .zip(chunk_log_probs.par_chunks_mut(top_k))
                .zip(chunk_residuals.par_iter_mut())
                .enumerate()
                .for_each(|(j, ((idx_out, lp_out), residual_out))| {
                    let row = &logits[j * slot.config.vocab_size..(j + 1) * slot.config.vocab_size];
                    *residual_out = log_softmax_top_k_row(row, top_k, idx_out, lp_out);
                });
            write_u32_slice(&mut indices_out, &chunk_indices);
            write_f32_slice(&mut log_probs_out, &chunk_log_probs);
            write_f32_slice(&mut residual_out, &chunk_residuals);
            scored_done += scored_per_chunk;
            let pct = scored_done as f64 * 100.0 / total_scored as f64;
            let elapsed = started.elapsed().as_secs_f64();
            let rate = scored_done as f64 / elapsed.max(1e-9);
            eprint!(
                "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                chunk_idx + 1,
                n_chunk,
                scored_done,
                total_scored,
                pct,
                rate
            );
        } else if slot.weights.output.gpu_dtype == DType::BF16 {
            const VOCAB_TILE: usize = 32 * 1024;
            const KLD_GPU_TOPK: usize = 256;
            const KLD_GPU_CHUNK: usize = 2048;
            let mut lm_head_ms = 0.0f64;
            let mut gpu_topk_ms = 0.0f64;
            let mut download_ms = 0.0f64;
            let mut host_merge_ms = 0.0f64;
            let tile_logits = gpu
                .alloc_tensor(
                    &[scored_per_chunk, VOCAB_TILE.min(slot.config.vocab_size)],
                    DType::F32,
                )
                .expect("alloc tiled bf16 lm_head logits");
            let mut row_acc: Vec<StreamingTopK> = (0..scored_per_chunk)
                .map(|_| StreamingTopK::new(top_k))
                .collect();
            let use_gpu_tile_topk = top_k == KLD_GPU_TOPK
                && std::env::var("HIPFIRE_KLD_GPU_TOPK").ok().as_deref() != Some("0");
            for vocab_start in (0..slot.config.vocab_size).step_by(VOCAB_TILE) {
                let vocab_tile = (slot.config.vocab_size - vocab_start).min(VOCAB_TILE);
                let output_tile = slot
                    .weights
                    .output
                    .buf
                    .sub_offset(vocab_start * slot.config.dim, vocab_tile * slot.config.dim);
                let logits_tile = tile_logits.sub_offset(0, scored_per_chunk * vocab_tile);
                let t_lm_head = Instant::now();
                if gpu.arch == "gfx1151"
                    && vocab_tile >= 128
                    && scored_per_chunk >= 16
                    && slot.config.dim % 16 == 0
                {
                    gpu.gemm_bf16_x_bf16_wmma_gfx1151_m128_labeled(
                        &output_tile,
                        &hidden_buf,
                        &logits_tile,
                        vocab_tile,
                        slot.config.dim,
                        scored_per_chunk,
                        "gemm_bf16_x_bf16_wmma_lm_head_m128",
                    )
                    .expect("gemm_bf16_x_bf16_wmma_gfx1151_m128 lm_head tile");
                } else {
                    gpu.gemm_bf16_x_bf16_wmma_labeled(
                        &output_tile,
                        &hidden_buf,
                        &logits_tile,
                        vocab_tile,
                        slot.config.dim,
                        scored_per_chunk,
                        "gemm_bf16_x_bf16_wmma_lm_head",
                    )
                    .expect("gemm_bf16_x_bf16_wmma lm_head tile");
                }
                lm_head_ms += t_lm_head.elapsed().as_secs_f64() * 1000.0;
                if use_gpu_tile_topk {
                    let n_tile_chunks = (vocab_tile + KLD_GPU_CHUNK - 1) / KLD_GPU_CHUNK;
                    let top_vals = gpu
                        .alloc_tensor(&[scored_per_chunk, n_tile_chunks, KLD_GPU_TOPK], DType::F32)
                        .expect("alloc kld tile top vals");
                    let top_idx = gpu
                        .alloc_tensor(&[scored_per_chunk, n_tile_chunks, KLD_GPU_TOPK], DType::F32)
                        .expect("alloc kld tile top idx");
                    let chunk_max = gpu
                        .alloc_tensor(&[scored_per_chunk, n_tile_chunks], DType::F32)
                        .expect("alloc kld tile chunk max");
                    let chunk_sum = gpu
                        .alloc_tensor(&[scored_per_chunk, n_tile_chunks], DType::F32)
                        .expect("alloc kld tile chunk sum");
                    let t_gpu_topk = Instant::now();
                    gpu.kld_tile_topk_lse_f32(
                        &logits_tile,
                        &top_vals,
                        &top_idx,
                        &chunk_max,
                        &chunk_sum,
                        scored_per_chunk,
                        vocab_tile,
                        vocab_start,
                        n_tile_chunks,
                    )
                    .expect("kld_tile_topk_lse_f32");
                    gpu_topk_ms += t_gpu_topk.elapsed().as_secs_f64() * 1000.0;
                    let t_download = Instant::now();
                    let vals = gpu.download_f32(&top_vals).expect("download kld top vals");
                    let idx = download_i32(&gpu, &top_idx);
                    let maxes = gpu
                        .download_f32(&chunk_max)
                        .expect("download kld chunk max");
                    let sums = gpu
                        .download_f32(&chunk_sum)
                        .expect("download kld chunk sum");
                    download_ms += t_download.elapsed().as_secs_f64() * 1000.0;
                    let t_host_merge = Instant::now();
                    row_acc.par_iter_mut().enumerate().for_each(|(row, acc)| {
                        for chunk_id in 0..n_tile_chunks {
                            let stat = row * n_tile_chunks + chunk_id;
                            acc.update_logsumexp(maxes[stat], sums[stat]);
                            let off = stat * KLD_GPU_TOPK;
                            acc.update_candidates(
                                &idx[off..off + KLD_GPU_TOPK],
                                &vals[off..off + KLD_GPU_TOPK],
                            );
                        }
                    });
                    host_merge_ms += t_host_merge.elapsed().as_secs_f64() * 1000.0;
                    let _ = gpu.free_tensor(top_vals);
                    let _ = gpu.free_tensor(top_idx);
                    let _ = gpu.free_tensor(chunk_max);
                    let _ = gpu.free_tensor(chunk_sum);
                } else {
                    let t_download = Instant::now();
                    let tile = gpu
                        .download_f32(&logits_tile)
                        .expect("download bf16 lm_head tile");
                    download_ms += t_download.elapsed().as_secs_f64() * 1000.0;
                    let t_host_merge = Instant::now();
                    row_acc
                        .par_iter_mut()
                        .zip(tile.par_chunks(vocab_tile))
                        .for_each(|(acc, row)| acc.update_tile(row, vocab_start));
                    host_merge_ms += t_host_merge.elapsed().as_secs_f64() * 1000.0;
                }
            }
            let _ = gpu.free_tensor(tile_logits);

            let t_finalize = Instant::now();
            let mut chunk_indices = vec![0u32; scored_per_chunk * top_k];
            let mut chunk_log_probs = vec![0.0f32; scored_per_chunk * top_k];
            let mut chunk_residuals = vec![0.0f32; scored_per_chunk];
            row_acc
                .into_par_iter()
                .zip(chunk_indices.par_chunks_mut(top_k))
                .zip(chunk_log_probs.par_chunks_mut(top_k))
                .zip(chunk_residuals.par_iter_mut())
                .for_each(|(((acc, idx_out), lp_out), residual_out)| {
                    *residual_out = acc.write_final(idx_out, lp_out);
                });
            host_merge_ms += t_finalize.elapsed().as_secs_f64() * 1000.0;
            write_u32_slice(&mut indices_out, &chunk_indices);
            write_f32_slice(&mut log_probs_out, &chunk_log_probs);
            write_f32_slice(&mut residual_out, &chunk_residuals);
            scored_done += scored_per_chunk;
            let pct = scored_done as f64 * 100.0 / total_scored as f64;
            let elapsed = started.elapsed().as_secs_f64();
            let rate = scored_done as f64 / elapsed.max(1e-9);
            eprint!(
                "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                chunk_idx + 1,
                n_chunk,
                scored_done,
                total_scored,
                pct,
                rate
            );
            if profile_this_chunk {
                eprintln!(
                    "\nKLD stage wall chunk {}: prefill={:.1}ms lm_head={:.1}ms gpu_topk={:.1}ms download={:.1}ms host_merge_finalize={:.1}ms",
                    chunk_idx + 1,
                    prefill_ms,
                    lm_head_ms,
                    gpu_topk_ms,
                    download_ms,
                    host_merge_ms
                );
            }
        } else if slot.weights.output.gpu_dtype == DType::F32 {
            const VOCAB_TILE: usize = 32 * 1024;
            let tile_logits = gpu
                .alloc_tensor(
                    &[scored_per_chunk, VOCAB_TILE.min(slot.config.vocab_size)],
                    DType::F32,
                )
                .expect("alloc tiled lm_head logits");
            let mut row_acc: Vec<StreamingTopK> = (0..scored_per_chunk)
                .map(|_| StreamingTopK::new(top_k))
                .collect();
            for vocab_start in (0..slot.config.vocab_size).step_by(VOCAB_TILE) {
                let vocab_tile = (slot.config.vocab_size - vocab_start).min(VOCAB_TILE);
                let output_tile = slot
                    .weights
                    .output
                    .buf
                    .sub_offset(vocab_start * slot.config.dim, vocab_tile * slot.config.dim);
                let logits_tile = tile_logits.sub_offset(0, scored_per_chunk * vocab_tile);
                gpu.gemm_f32_register_tiled_b64(
                    &output_tile,
                    &hidden_buf,
                    &logits_tile,
                    vocab_tile,
                    slot.config.dim,
                    scored_per_chunk,
                )
                .expect("gemm_f32_register_tiled_b64 lm_head tile");
                let tile = gpu
                    .download_f32(&logits_tile)
                    .expect("download lm_head tile");
                row_acc
                    .par_iter_mut()
                    .zip(tile.par_chunks(vocab_tile))
                    .for_each(|(acc, row)| acc.update_tile(row, vocab_start));
            }
            let _ = gpu.free_tensor(tile_logits);

            let mut chunk_indices = vec![0u32; scored_per_chunk * top_k];
            let mut chunk_log_probs = vec![0.0f32; scored_per_chunk * top_k];
            let mut chunk_residuals = vec![0.0f32; scored_per_chunk];
            row_acc
                .into_par_iter()
                .zip(chunk_indices.par_chunks_mut(top_k))
                .zip(chunk_log_probs.par_chunks_mut(top_k))
                .zip(chunk_residuals.par_iter_mut())
                .for_each(|(((acc, idx_out), lp_out), residual_out)| {
                    *residual_out = acc.write_final(idx_out, lp_out);
                });
            write_u32_slice(&mut indices_out, &chunk_indices);
            write_f32_slice(&mut log_probs_out, &chunk_log_probs);
            write_f32_slice(&mut residual_out, &chunk_residuals);
            scored_done += scored_per_chunk;
            let pct = scored_done as f64 * 100.0 / total_scored as f64;
            let elapsed = started.elapsed().as_secs_f64();
            let rate = scored_done as f64 / elapsed.max(1e-9);
            eprint!(
                "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                chunk_idx + 1,
                n_chunk,
                scored_done,
                total_scored,
                pct,
                rate
            );
        } else {
            for j in 0..scored_per_chunk {
                let hidden_row = hidden_buf.sub_offset(j * slot.config.dim, slot.config.dim);
                weight_gemv(
                    &mut gpu,
                    &slot.weights.output,
                    &hidden_row,
                    &slot.scratch.logits,
                )
                .expect("weight_gemv lm_head");
                let logits_tensor = slot.scratch.logits.sub_offset(0, slot.config.vocab_size);
                let logits = gpu.download_f32(&logits_tensor).expect("download logits");
                log_softmax_top_k(
                    &logits,
                    top_k,
                    &mut indices_out,
                    &mut log_probs_out,
                    &mut residual_out,
                );
                scored_done += 1;
                if scored_done % progress_interval == 0 || scored_done == total_scored {
                    let pct = scored_done as f64 * 100.0 / total_scored as f64;
                    let elapsed = started.elapsed().as_secs_f64();
                    let rate = scored_done as f64 / elapsed.max(1e-9);
                    eprint!(
                        "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                        chunk_idx + 1,
                        n_chunk,
                        scored_done,
                        total_scored,
                        pct,
                        rate
                    );
                }
            }
        }
        if let Some(logits) = batched_logits {
            let _ = gpu.free_tensor(logits);
        }
        if profile_this_chunk {
            let entries = rdna_compute::profile::stop().unwrap_or_default();
            print_profile_summary(&entries);
        }
    }
    eprintln!();
    indices_out.flush().unwrap();
    log_probs_out.flush().unwrap();
    residual_out.flush().unwrap();
    drop(indices_out);
    drop(log_probs_out);
    drop(residual_out);

    let elapsed = started.elapsed().as_secs_f64();
    let producer_cmd = std::env::args().collect::<Vec<_>>().join(" ");
    let source_model_hash = join_source_model_hash(source_model_hash);
    let source_model_xxh64 = source_model_hash.xxh64.clone();

    let metadata = json!({
        "schema": 1,
        "artifact_kind": "hipfire.kldref",
        "package_schema": "hipfire.kldref.v1",
        "producer": "build_kld_ref_hipfire",
        "format": "HFQM",
        "format_version": KLDREF_SCHEMA_VERSION,
        "base_model_id": args.model.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown"),
        "reference_precision": "bf16",
        "model": args.model.display().to_string(),
        "source_model_xxh64": source_model_xxh64,
        "source_model_hash": {
            "algorithm": "xxh64",
            "seed": 0,
            "value": source_model_hash.xxh64,
        },
        "source_model_sha256": source_model_hash.sha256,
        "slice": args.slice.display().to_string(),
        "slice_md5": command_hash("md5sum", &args.slice),
        "output": args.output.display().to_string(),
        "n_ctx": args.n_ctx,
        "n_vocab": slot.config.vocab_size,
        "n_chunk": n_chunk,
        "top_k": top_k,
        "kv_mode": format!("{:?}", args.kv_mode),
        "deltanet_state_precision": "fp32",
        "deltanet_state_quant": "FP32",
        "kld_direct_wmma_attention": use_kld_direct_f16kv_attn,
        "kld_fp32_gqa4_attention": use_kld_fp32_gqa4_attn,
        "attention_kv_precision": if use_kld_direct_f16kv_attn {
            "direct_wmma_f32_kv"
        } else if use_kld_fp32_gqa4_attn {
            "direct_fp32_gqa4"
        } else {
            "cache_mode"
        },
        "kld_graph_prefill": use_kld_graph,
        "kld_graph_prefill_max_batch": std::env::var("HIPFIRE_PREFILL_MAX_BATCH").ok(),
        "producer_cmd": producer_cmd,
        "scoring_start": scoring_start,
        "scored_per_chunk": scored_per_chunk,
        "total_scored": total_scored,
        "slice_tokens": all_tokens.len(),
        "dropped_tail_tokens": all_tokens.len() - tokens.len(),
        "elapsed_sec": elapsed,
        "scored_tokens_per_sec": total_scored as f64 / elapsed.max(1e-9),
        "hipfire_version": env!("CARGO_PKG_VERSION"),
        "git_commit": option_env!("HIPFIRE_GIT_COMMIT"),
        "git_branch": option_env!("HIPFIRE_GIT_BRANCH"),
        "git_describe": option_env!("HIPFIRE_GIT_DESCRIBE"),
        "git_dirty": option_env!("HIPFIRE_GIT_DIRTY"),
        "gpu_arch": gpu.arch,
        "arch_id": HFQM_ARCH_NON_WEIGHT_PACKAGE,
        "arch_id_semantics": "0 is reserved for non-weight HFQM packages",
        "payloads": {
            "kldref.tokens": {"dtype": "u32", "shape": [n_chunk, args.n_ctx]},
            "kldref.top_indices": {"dtype": "u32", "shape": [n_chunk, scored_per_chunk, top_k]},
            "kldref.top_log_probs": {"dtype": "f32", "shape": [n_chunk, scored_per_chunk, top_k]},
            "kldref.residual_mass": {"dtype": "f32", "shape": [n_chunk, scored_per_chunk]}
        },
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();
    let entries = vec![
        HfqPackageWriteEntry {
            name: "kldref.tokens".to_string(),
            quant_type: KLDREF_ENTRY_QUANT_TYPE,
            shape: vec![n_chunk as u32, args.n_ctx as u32],
            group_size: 0,
            source_path: tokens_path.clone(),
            data_size: std::fs::metadata(&tokens_path)
                .expect("tokens metadata")
                .len(),
        },
        HfqPackageWriteEntry {
            name: "kldref.top_indices".to_string(),
            quant_type: KLDREF_ENTRY_QUANT_TYPE,
            shape: vec![n_chunk as u32, scored_per_chunk as u32, top_k as u32],
            group_size: 0,
            source_path: indices_path.clone(),
            data_size: std::fs::metadata(&indices_path)
                .expect("top_indices metadata")
                .len(),
        },
        HfqPackageWriteEntry {
            name: "kldref.top_log_probs".to_string(),
            quant_type: KLDREF_ENTRY_QUANT_TYPE,
            shape: vec![n_chunk as u32, scored_per_chunk as u32, top_k as u32],
            group_size: 0,
            source_path: log_probs_path.clone(),
            data_size: std::fs::metadata(&log_probs_path)
                .expect("top_log_probs metadata")
                .len(),
        },
        HfqPackageWriteEntry {
            name: "kldref.residual_mass".to_string(),
            quant_type: KLDREF_ENTRY_QUANT_TYPE,
            shape: vec![n_chunk as u32, scored_per_chunk as u32],
            group_size: 0,
            source_path: residual_path.clone(),
            data_size: std::fs::metadata(&residual_path)
                .expect("residual_mass metadata")
                .len(),
        },
    ];
    write_hfqm_package_from_files(
        &args.output,
        HFQM_ARCH_NON_WEIGHT_PACKAGE,
        &metadata_json,
        &entries,
    )
    .expect("write HFQM kldref package");

    let _ = std::fs::remove_file(&tokens_path);
    let _ = std::fs::remove_file(&indices_path);
    let _ = std::fs::remove_file(&log_probs_path);
    let _ = std::fs::remove_file(&residual_path);

    let out_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "build_kld_ref_hipfire: wrote {} ({} bytes = {:.3} GB) in {:.1}s ({:.1} scored tok/s)",
        args.output.display(),
        out_size,
        out_size as f64 / 1e9,
        elapsed,
        total_scored as f64 / elapsed.max(1e-9)
    );
    if let Some(path) = args.metadata_json.as_deref() {
        write_metadata(path, metadata);
    } else {
        write_metadata(&args.output.with_extension("json"), metadata);
    }
}
