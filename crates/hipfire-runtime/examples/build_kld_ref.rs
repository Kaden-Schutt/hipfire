//! build_kld_ref — KLD reference producer for the hipfire quant-quality eval.
//!
//! Spawns `llama-perplexity --kl-divergence-base <fifo>`, reads its full-vocab
//! uint16 KLD-base stream from a FIFO, top-K-reduces in flight, and writes
//! the hipfire β-format reference file (~2.15 GB at top_k=256, vs llama.cpp's
//! native ~318 GB).
//!
//! See docs/plans/issue-113-quant-quality-eval.md §"Hipfire-derived top-K
//! format" for the on-disk format spec.
//!
//! Pinned llama.cpp commit: 9dcf83552887bb898b4a98a5761361e504e31fc3.
//!
//! Usage:
//!   cargo run --release -p hipfire-runtime --example build_kld_ref -- \
//!       --bf16-gguf <path-to-bf16.gguf> \
//!       --slice    <path-to-slice.txt> \
//!       --top-k    256 \
//!       --output   <path-to-output.kldref.bin>

use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

const HIPFIRE_MAGIC: &[u8; 8] = b"HFKLDR\0\0";
const HIPFIRE_VERSION: u32 = 1;
const LLAMA_MAGIC: &[u8; 8] = b"_logits_";
const PINNED_LLAMACPP_COMMIT: &str = "9dcf83552887bb898b4a98a5761361e504e31fc3";

struct Args {
    bf16_gguf: PathBuf,
    slice: PathBuf,
    top_k: usize,
    output: PathBuf,
    n_ctx: usize,
    n_batch: usize,
    llama_perplexity_bin: String,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  build_kld_ref --bf16-gguf <path> --slice <path> --top-k <N> --output <path> \\\n                [--n-ctx <N>=2048] [--n-batch <N>=512] [--llama-perplexity-bin <path>=llama-perplexity]"
    );
}

fn parse_args() -> Args {
    let mut bf16_gguf: Option<PathBuf> = None;
    let mut slice: Option<PathBuf> = None;
    let mut top_k: usize = 256;
    let mut output: Option<PathBuf> = None;
    let mut n_ctx: usize = 2048;
    let mut n_batch: usize = 512;
    let mut llama_perplexity_bin: String = "llama-perplexity".to_string();

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--bf16-gguf"     => { bf16_gguf = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--slice"         => { slice = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--top-k"         => { top_k = argv[i + 1].parse().expect("--top-k must be u32"); i += 2; }
            "--output"        => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--n-ctx"         => { n_ctx = argv[i + 1].parse().expect("--n-ctx must be u32"); i += 2; }
            "--n-batch"       => { n_batch = argv[i + 1].parse().expect("--n-batch must be u32"); i += 2; }
            "--llama-perplexity-bin" => { llama_perplexity_bin = argv[i + 1].clone(); i += 2; }
            "-h" | "--help"   => { print_usage(); std::process::exit(0); }
            other             => { eprintln!("unknown arg: {other}"); print_usage(); std::process::exit(1); }
        }
    }
    let bf16_gguf = bf16_gguf.unwrap_or_else(|| { print_usage(); std::process::exit(1); });
    let slice = slice.unwrap_or_else(|| { print_usage(); std::process::exit(1); });
    let output = output.unwrap_or_else(|| { print_usage(); std::process::exit(1); });
    Args { bf16_gguf, slice, top_k, output, n_ctx, n_batch, llama_perplexity_bin }
}

/// Drop guard that unlinks the FIFO on scope exit. Survives panics.
struct FifoGuard(PathBuf);
impl Drop for FifoGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn main() {
    let args = parse_args();

    // 0. Sanity checks (H1 + M4): pinned llama.cpp commit + slice md5.
    hipfire_runtime::eval_common::verify_llama_commit(
        &args.llama_perplexity_bin, PINNED_LLAMACPP_COMMIT, "build_kld_ref",
    );
    hipfire_runtime::eval_common::verify_slice_md5(&args.slice, "build_kld_ref");

    // 1. mkfifo
    let fifo_path = PathBuf::from(format!("/tmp/hipfire-kldref-{}.fifo", std::process::id()));
    let _ = fs::remove_file(&fifo_path);
    let status = Command::new("mkfifo")
        .arg(&fifo_path)
        .status()
        .expect("failed to invoke mkfifo");
    if !status.success() {
        eprintln!("mkfifo exited with {status}");
        std::process::exit(1);
    }
    let _fifo_guard = FifoGuard(fifo_path.clone());
    eprintln!("build_kld_ref: created FIFO {}", fifo_path.display());

    // 2. Spawn llama-perplexity (writes to FIFO; blocks on FIFO open(write)
    //    until we open(read) below — Linux FIFO rendezvous semantics).
    eprintln!("build_kld_ref: spawning {} ...", args.llama_perplexity_bin);
    // --no-mmap: read all weights into one allocation upfront. Without
    // this, mmap demand-paging on a 50 GB BF16 model causes eviction
    // cycles when working-set ≈ available RAM (observed on gfx1151's
    // 124 GB UMA: 27B BF16 stalled in load_tensors for 10+ hours
    // because pages were paged in / evicted / re-paged repeatedly).
    // --no-mmap forces a single sequential read into pinned RAM at
    // startup, then no further IO during inference.
    let mut child = Command::new(&args.llama_perplexity_bin)
        .args(["-m", &args.bf16_gguf.display().to_string()])
        .args(["-f", &args.slice.display().to_string()])
        .args(["-c", &args.n_ctx.to_string()])
        .args(["-b", &args.n_batch.to_string()])
        .args(["--kl-divergence-base", &fifo_path.display().to_string()])
        .arg("--no-mmap")
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .spawn()
        .expect("failed to spawn llama-perplexity");

    // 3. Open FIFO for read. Will rendezvous with the child's open(write).
    eprintln!("build_kld_ref: opening FIFO for read (will block until llama-perplexity opens for write)...");
    let input_file = File::open(&fifo_path).expect("failed to open FIFO for read");
    let mut input = BufReader::with_capacity(8 * 1024 * 1024, input_file);
    eprintln!("build_kld_ref: FIFO open; reading llama.cpp header...");

    // 4. Read llama.cpp header (16 bytes).
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic).expect("read llama magic");
    if &magic != LLAMA_MAGIC {
        eprintln!("bad llama.cpp magic: got {:?}, want {:?}", magic, LLAMA_MAGIC);
        std::process::exit(2);
    }
    let mut hdr_rest = [0u8; 12];
    input.read_exact(&mut hdr_rest).expect("read llama header");
    let n_ctx_actual = u32::from_le_bytes(hdr_rest[0..4].try_into().unwrap()) as usize;
    let n_vocab = i32::from_le_bytes(hdr_rest[4..8].try_into().unwrap()) as usize;
    let n_chunk = i32::from_le_bytes(hdr_rest[8..12].try_into().unwrap()) as usize;

    eprintln!(
        "build_kld_ref: llama header → n_ctx={n_ctx_actual} n_vocab={n_vocab} n_chunk={n_chunk}"
    );

    if n_ctx_actual != args.n_ctx {
        eprintln!(
            "warning: --n-ctx {} but llama header says {}; trusting header",
            args.n_ctx, n_ctx_actual
        );
    }

    // 5. Read tokens (n_ctx_actual * n_chunk * 4 bytes, int32).
    let n_tokens = n_ctx_actual * n_chunk;
    let mut tokens = vec![0u8; n_tokens * 4];
    input.read_exact(&mut tokens).expect("read tokens");
    eprintln!("build_kld_ref: read {} tokens ({} bytes)", n_tokens, tokens.len());

    // 6. Open output, write hipfire β header + tokens.
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output parent dir");
        }
    }
    let output_file = File::create(&args.output).expect("failed to create output");
    let mut output = BufWriter::with_capacity(4 * 1024 * 1024, output_file);
    output.write_all(HIPFIRE_MAGIC).unwrap();
    output.write_all(&HIPFIRE_VERSION.to_le_bytes()).unwrap();
    output.write_all(&(n_ctx_actual as u32).to_le_bytes()).unwrap();
    output.write_all(&(n_vocab as u32).to_le_bytes()).unwrap();
    output.write_all(&(n_chunk as u32).to_le_bytes()).unwrap();
    output.write_all(&(args.top_k as u16).to_le_bytes()).unwrap();
    output.write_all(&0u16.to_le_bytes()).unwrap(); // flags
    output.write_all(&0u32.to_le_bytes()).unwrap(); // reserved
    // tokens: pass-through. Llama writes int32; hipfire reads as uint32 (same bit-pattern, no negatives expected for vocab IDs).
    output.write_all(&tokens).unwrap();

    // 7. Per-token loop: reduce to top-K + sum_p_residual.
    //    nv = 2*((n_vocab+1)/2) + 4   (uint16s per block)
    //    block_bytes = nv * 2
    let nv = 2 * ((n_vocab + 1) / 2) + 4;
    let block_bytes = nv * 2;
    let scored_per_chunk = n_ctx_actual - 1 - n_ctx_actual / 2;
    let total_scored = scored_per_chunk * n_chunk;
    let k = args.top_k;

    eprintln!(
        "build_kld_ref: reducing {} per-token blocks ({} bytes each) to top-K={}...",
        total_scored, block_bytes, k
    );

    let mut block_buf = vec![0u8; block_bytes];
    let mut log_probs: Vec<(u32, f32)> = Vec::with_capacity(n_vocab);

    let progress_interval = (total_scored / 100).max(1);
    let t0 = std::time::Instant::now();

    for i in 0..total_scored {
        input.read_exact(&mut block_buf).expect("read block");

        // First 8 bytes: scale (fp32) + min_log_prob (fp32), each as 2 uint16s.
        let scale = f32::from_le_bytes(block_buf[0..4].try_into().unwrap());
        let min_log_prob = f32::from_le_bytes(block_buf[4..8].try_into().unwrap());

        // n_vocab uint16 stored values follow. Reconstruct log-probs:
        //   log_p[i] = scale * stored[i] + min_log_prob
        // (verified against perplexity.cpp:222-225 on commit 9dcf83552 —
        // llama.cpp's encoding stores log-probs already, not raw logits.)
        log_probs.clear();
        for v in 0..n_vocab {
            let off = 8 + v * 2;
            let stored = u16::from_le_bytes(block_buf[off..off + 2].try_into().unwrap());
            let log_p = scale * (stored as f32) + min_log_prob;
            log_probs.push((v as u32, log_p));
        }

        // Top-K reduction: select_nth_unstable_by puts the K-th largest at
        // position k-1 in O(n) average; first k entries are the top-K but
        // unsorted. Then sort the first k descending — O(k log k) at k=256
        // is negligible.
        let cmp_desc = |a: &(u32, f32), b: &(u32, f32)| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        };
        if k < log_probs.len() {
            log_probs.select_nth_unstable_by(k - 1, cmp_desc);
        }
        log_probs[..k].sort_by(cmp_desc);

        // sum_p_residual: 1 - sum_{i in top_k} exp(log_p[i])  (fp64 accumulator)
        let top_p_sum_f64: f64 = log_probs[..k]
            .iter()
            .map(|&(_, lp)| (lp as f64).exp())
            .sum();
        let sum_p_residual = (1.0 - top_p_sum_f64).max(0.0) as f32;

        // Write hipfire β block.
        for &(idx, _) in &log_probs[..k] {
            output.write_all(&idx.to_le_bytes()).unwrap();
        }
        for &(_, lp) in &log_probs[..k] {
            output.write_all(&lp.to_le_bytes()).unwrap();
        }
        output.write_all(&sum_p_residual.to_le_bytes()).unwrap();
        output.write_all(&0f32.to_le_bytes()).unwrap(); // pad

        if (i + 1) % progress_interval == 0 || i + 1 == total_scored {
            let pct = (i + 1) as f64 * 100.0 / total_scored as f64;
            let elapsed = t0.elapsed().as_secs_f64();
            let toks_per_sec = (i + 1) as f64 / elapsed;
            eprint!(
                "\r  {:>6.2}%  ({}/{} tokens, {:.0} tok/s)   ",
                pct, i + 1, total_scored, toks_per_sec
            );
        }
    }
    eprintln!();

    output.flush().unwrap();
    drop(output);

    // 8. Wait for child, report.
    let status = child.wait().expect("failed to wait on child");
    if !status.success() {
        eprintln!(
            "warning: llama-perplexity exited with non-zero status: {}",
            status
        );
    }

    let out_size = fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "build_kld_ref: wrote {} ({} bytes = {:.2} GB) in {:.1}s",
        args.output.display(),
        out_size,
        out_size as f64 / 1e9,
        elapsed
    );
    eprintln!(
        "build_kld_ref: average reduction throughput: {:.0} tokens/sec",
        total_scored as f64 / elapsed
    );
}

// verify_llama_commit and verify_slice_md5 now live in
// hipfire_runtime::eval_common (crates/hipfire-runtime/src/eval_common.rs)
// so the same fix applies across eval_hipfire / eval_gguf / build_kld_ref.
