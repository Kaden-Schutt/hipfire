//! eval_gguf — KLD eval for GGUF quant variants against a hipfire-β BF16 reference.
//!
//! Spawns `llama-perplexity --kl-divergence-base <fifo>` on a GGUF candidate,
//! reads its full-vocab uint16 logit stream from a FIFO, computes per-token
//! KLD against the cached BF16 reference (in hipfire-β format) by looking up
//! the candidate's log-prob at the reference's top_indices, bins per-sequence,
//! emits HFKSEQ output that `kld_reduce.py` aggregates.
//!
//! KLD math is identical to `eval_hipfire.rs` — only the candidate-logit
//! source differs (FIFO stream from llama-perplexity vs hipfire forward).
//!
//! Plan: docs/plans/issue-113-quant-quality-eval.md §"GGUF anchor architecture
//! (rev-3.3)".
//!
//! Pinned llama.cpp commit: 9dcf83552887bb898b4a98a5761361e504e31fc3.
//!
//! Usage:
//!   cargo run --release -p hipfire-runtime --example eval_gguf -- \
//!       --candidate-gguf <path-to-candidate.gguf> \
//!       --ref            <path-to-hipfire-β-ref> \
//!       --slice          <path-to-slice.txt> \
//!       --output         <path-to-output.kldseq> \
//!       [--n-batch <N>=512] \
//!       [--llama-perplexity-bin <path>=llama-perplexity]

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

const HIPFIRE_MAGIC: &[u8; 8] = b"HFKLDR\0\0";
const SEQKLD_MAGIC: &[u8; 8] = b"HFKSEQ\0\0";
const LLAMA_MAGIC: &[u8; 8] = b"_logits_";
const PINNED_LLAMACPP_COMMIT: &str = "9dcf83552887bb898b4a98a5761361e504e31fc3";

struct Args {
    candidate_gguf: PathBuf,
    ref_path: PathBuf,
    slice: PathBuf,
    output: PathBuf,
    n_batch: usize,
    llama_perplexity_bin: String,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  eval_gguf --candidate-gguf <path> --ref <path> --slice <path> --output <path> \\\n            [--n-batch <N>=512] [--llama-perplexity-bin <path>=llama-perplexity]"
    );
}

fn parse_args() -> Args {
    let mut candidate_gguf: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut slice: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_batch: usize = 512;
    let mut llama_perplexity_bin: String = "llama-perplexity".to_string();

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--candidate-gguf" => { candidate_gguf = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref"            => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--slice"          => { slice = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--output"         => { output = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--n-batch"        => { n_batch = argv[i + 1].parse().expect("--n-batch must be u32"); i += 2; }
            "--llama-perplexity-bin" => { llama_perplexity_bin = argv[i + 1].clone(); i += 2; }
            "-h" | "--help"    => { print_usage(); std::process::exit(0); }
            other              => { eprintln!("unknown arg: {other}"); print_usage(); std::process::exit(1); }
        }
    }
    Args {
        candidate_gguf: candidate_gguf.unwrap_or_else(|| { print_usage(); std::process::exit(1); }),
        ref_path: ref_path.unwrap_or_else(|| { print_usage(); std::process::exit(1); }),
        slice: slice.unwrap_or_else(|| { print_usage(); std::process::exit(1); }),
        output: output.unwrap_or_else(|| { print_usage(); std::process::exit(1); }),
        n_batch,
        llama_perplexity_bin,
    }
}

struct FifoGuard(PathBuf);
impl Drop for FifoGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn main() {
    let args = parse_args();

    // ---------- Sanity (H1 + M1 + M4) ----------
    hipfire_runtime::eval_common::verify_llama_commit(
        &args.llama_perplexity_bin, PINNED_LLAMACPP_COMMIT, "eval_gguf",
    );
    hipfire_runtime::eval_common::verify_slice_md5(&args.slice, "eval_gguf");
    hipfire_runtime::eval_common::verify_ref_sha256(&args.ref_path, "eval_gguf");

    // ---------- Open ref file, read header + tokens ----------
    let ref_file = File::open(&args.ref_path).expect("open ref");
    let mut ref_in = BufReader::with_capacity(8 * 1024 * 1024, ref_file);

    let mut magic = [0u8; 8];
    ref_in.read_exact(&mut magic).expect("read ref magic");
    if &magic != HIPFIRE_MAGIC {
        eprintln!("bad ref magic: got {:?}, want {:?}", magic, HIPFIRE_MAGIC);
        std::process::exit(2);
    }
    let mut hdr = [0u8; 24];
    ref_in.read_exact(&mut hdr).expect("read ref header");
    let version = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    let ref_n_ctx = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let ref_n_vocab = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let ref_n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let top_k = u16::from_le_bytes(hdr[16..18].try_into().unwrap()) as usize;
    if version != 1 {
        eprintln!("unsupported ref version {version}"); std::process::exit(2);
    }
    let ref_per_token_bytes = 8 + 8 * top_k;
    let scored_per_chunk = ref_n_ctx - 1 - ref_n_ctx / 2;
    let total_scored = scored_per_chunk * ref_n_chunk;

    eprintln!(
        "eval_gguf: ref n_ctx={ref_n_ctx} n_vocab={ref_n_vocab} n_chunk={ref_n_chunk} top_k={top_k}"
    );

    // Hold ref's tokens block in memory; we'll byte-compare against the
    // candidate's tokens block once it lands on the FIFO (H7 — guard against
    // tokenizer drift between the BF16 ref's GGUF and the candidate GGUF).
    // Without this check, a different-tokenizer candidate produces blocks
    // for a different token sequence than the ref's, and every per-token
    // KLD is computed against the wrong reference distribution.
    let ref_tokens_bytes = ref_n_ctx * ref_n_chunk * 4;
    let mut ref_tokens = vec![0u8; ref_tokens_bytes];
    ref_in.read_exact(&mut ref_tokens).expect("read ref tokens");

    // ---------- Set up FIFO + spawn llama-perplexity on the candidate ----------
    let fifo_path = PathBuf::from(format!("/tmp/hipfire-eval-gguf-{}.fifo", std::process::id()));
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
    eprintln!("eval_gguf: created FIFO {}", fifo_path.display());

    eprintln!("eval_gguf: spawning {} on candidate {}...",
              args.llama_perplexity_bin, args.candidate_gguf.display());
    let mut child = Command::new(&args.llama_perplexity_bin)
        .args(["-m", &args.candidate_gguf.display().to_string()])
        .args(["-f", &args.slice.display().to_string()])
        .args(["-c", &ref_n_ctx.to_string()])
        .args(["-b", &args.n_batch.to_string()])
        .args(["--kl-divergence-base", &fifo_path.display().to_string()])
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .spawn()
        .expect("failed to spawn llama-perplexity");

    // ---------- Read candidate's llama.cpp header from FIFO ----------
    eprintln!("eval_gguf: opening FIFO for read...");
    let cand_file = File::open(&fifo_path).expect("failed to open FIFO for read");
    let mut cand_in = BufReader::with_capacity(8 * 1024 * 1024, cand_file);

    let mut cand_magic = [0u8; 8];
    cand_in.read_exact(&mut cand_magic).expect("read cand llama magic");
    if &cand_magic != LLAMA_MAGIC {
        eprintln!("bad llama.cpp magic from candidate: {:?}", cand_magic);
        std::process::exit(2);
    }
    let mut cand_hdr = [0u8; 12];
    cand_in.read_exact(&mut cand_hdr).expect("read cand llama header");
    let cand_n_ctx = u32::from_le_bytes(cand_hdr[0..4].try_into().unwrap()) as usize;
    let cand_n_vocab = i32::from_le_bytes(cand_hdr[4..8].try_into().unwrap()) as usize;
    let cand_n_chunk = i32::from_le_bytes(cand_hdr[8..12].try_into().unwrap()) as usize;
    eprintln!(
        "eval_gguf: cand n_ctx={cand_n_ctx} n_vocab={cand_n_vocab} n_chunk={cand_n_chunk}"
    );

    if cand_n_ctx != ref_n_ctx {
        eprintln!("ERROR: cand n_ctx {cand_n_ctx} != ref n_ctx {ref_n_ctx}");
        std::process::exit(2);
    }
    if cand_n_vocab != ref_n_vocab {
        eprintln!("ERROR: cand n_vocab {cand_n_vocab} != ref n_vocab {ref_n_vocab}");
        std::process::exit(2);
    }
    if cand_n_chunk != ref_n_chunk {
        eprintln!("ERROR: cand n_chunk {cand_n_chunk} != ref n_chunk {ref_n_chunk}");
        std::process::exit(2);
    }

    // Read candidate's tokens block and compare byte-for-byte against
    // ref's (H7). Same tokenizer.json on both sides produces identical
    // token IDs from identical slice bytes — anything else means the
    // candidate's GGUF was built from a different tokenizer, in which
    // case every subsequent KLD is meaningless.
    let cand_tokens_bytes = cand_n_ctx * cand_n_chunk * 4;
    let mut cand_tokens = vec![0u8; cand_tokens_bytes];
    cand_in.read_exact(&mut cand_tokens).expect("read cand tokens");
    if ref_tokens != cand_tokens {
        // Find first diverging position (in u32 space) for diagnostics.
        let mut first_diff = None;
        for (i, (r, c)) in ref_tokens.chunks_exact(4).zip(cand_tokens.chunks_exact(4)).enumerate() {
            if r != c {
                let r_id = u32::from_le_bytes(r.try_into().unwrap());
                let c_id = u32::from_le_bytes(c.try_into().unwrap());
                first_diff = Some((i, r_id, c_id));
                break;
            }
        }
        eprintln!("ERROR: candidate tokenization disagrees with ref.");
        eprintln!("  This means the BF16 ref's GGUF and the candidate GGUF use");
        eprintln!("  different tokenizers (or were built from different vocab");
        eprintln!("  snapshots). KLD math against this candidate is invalid.");
        if let Some((pos, r_id, c_id)) = first_diff {
            eprintln!("  first divergence at token index {pos}: ref={r_id} cand={c_id}");
        }
        std::process::exit(2);
    }
    drop(cand_tokens);
    // Decode ref_tokens once for actual-next-token lookups during NLL accumulation.
    let tokens: Vec<u32> = ref_tokens
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(ref_tokens);
    eprintln!("eval_gguf: verified cand tokens match ref tokens ({cand_tokens_bytes} bytes)");

    // ---------- Per-token loop ----------
    // For each scored token (n_chunk × scored_per_chunk):
    //   1. Read candidate's per-token block from FIFO (full vocab, llama.cpp format).
    //   2. Read ref's per-token block from ref file (top-K + residual, hipfire-β).
    //   3. Compute KLD = Σ_{i in ref_top_K} P_ref(i) * (log_p_ref - log_p_cand[i])
    //                  + residual cross-term
    //   4. Bin per-chunk; track mean + p99 per chunk.

    let cand_nv = 2 * ((cand_n_vocab + 1) / 2) + 4;
    let cand_block_bytes = cand_nv * 2;

    eprintln!(
        "eval_gguf: scoring {} tokens ({}/chunk × {} chunks)...",
        total_scored, scored_per_chunk, ref_n_chunk
    );
    eprintln!(
        "  cand block = {} B  ({} uint16),  ref block = {} B",
        cand_block_bytes, cand_nv, ref_per_token_bytes
    );

    let mut cand_block_buf = vec![0u8; cand_block_bytes];
    let mut ref_block_buf = vec![0u8; ref_per_token_bytes];

    let mut mean_kld_per_seq: Vec<f64> = Vec::with_capacity(ref_n_chunk);
    let mut p99_kld_per_seq: Vec<f64> = Vec::with_capacity(ref_n_chunk);
    let mut mean_nll_per_seq: Vec<f64> = Vec::with_capacity(ref_n_chunk);

    // Logit-positions for scored tokens within a chunk: [n_ctx/2, n_ctx-2].
    // The j-th scored token (0-indexed) has logit-pos = n_ctx/2 + j and
    // predicts the actual token at logit-pos+1, i.e. tokens[c*n_ctx + n_ctx/2 + j + 1].
    let scoring_start = ref_n_ctx / 2;

    let progress_interval = (total_scored / 100).max(1);
    let t0 = std::time::Instant::now();
    let mut total_done = 0usize;

    for c in 0..ref_n_chunk {
        let mut chunk_klds: Vec<f64> = Vec::with_capacity(scored_per_chunk);
        let mut chunk_nll_sum: f64 = 0.0;
        let mut chunk_nll_count: usize = 0;

        for j in 0..scored_per_chunk {
            // Read candidate's per-token block
            cand_in.read_exact(&mut cand_block_buf).expect("read cand block");

            // First 8 bytes: scale + min_log_prob (fp32 each, packed as 2× uint16)
            let cand_scale = f32::from_le_bytes(cand_block_buf[0..4].try_into().unwrap());
            let cand_min_log_prob = f32::from_le_bytes(cand_block_buf[4..8].try_into().unwrap());

            // Read ref's per-token block
            ref_in.read_exact(&mut ref_block_buf).expect("read ref block");
            // Parse β block: u32 indices[K] | f32 log_probs[K] | f32 residual | f32 pad
            let mut ref_top_indices: Vec<u32> = Vec::with_capacity(top_k);
            let mut ref_top_log_probs: Vec<f32> = Vec::with_capacity(top_k);
            for j in 0..top_k {
                ref_top_indices.push(u32::from_le_bytes(ref_block_buf[j * 4..j * 4 + 4].try_into().unwrap()));
            }
            let lp_off = top_k * 4;
            for j in 0..top_k {
                ref_top_log_probs.push(f32::from_le_bytes(ref_block_buf[lp_off + j * 4..lp_off + j * 4 + 4].try_into().unwrap()));
            }
            let resid_off = top_k * 8;
            let ref_sum_p_residual = f32::from_le_bytes(ref_block_buf[resid_off..resid_off + 4].try_into().unwrap());

            // KLD math (mirrors eval_hipfire.rs):
            //   Σ_{i in ref_top_K} P_ref(i) * (log_p_ref(i) - log_p_cand(i))
            //   + sum_p_residual_ref * (log sum_p_residual_ref - log sum_p_residual_cand)
            //
            // log_p_cand reconstructed from cand's stored uint16:
            //   log_p_cand[i] = scale * stored[i] + min_log_prob   (fp64)
            // (Verified against tools/perplexity/perplexity.cpp:222-225 on
            // commit 9dcf83552. Note that this is log-prob directly, not raw logit.)
            let mut kld_token = 0.0f64;
            let mut sum_p_cand_at_ref_top = 0.0f64;
            for j in 0..top_k {
                let ref_idx = ref_top_indices[j] as usize;
                if ref_idx >= cand_n_vocab {
                    eprintln!("warn: ref idx {ref_idx} >= cand_n_vocab {cand_n_vocab}");
                    continue;
                }
                let stored_off = 8 + ref_idx * 2;
                let stored = u16::from_le_bytes(
                    cand_block_buf[stored_off..stored_off + 2].try_into().unwrap()
                );
                let log_p_cand = (cand_scale as f64) * (stored as f64) + (cand_min_log_prob as f64);
                let log_p_ref = ref_top_log_probs[j] as f64;
                let p_ref = log_p_ref.exp();
                let p_cand = log_p_cand.exp();
                kld_token += p_ref * (log_p_ref - log_p_cand);
                sum_p_cand_at_ref_top += p_cand;
            }
            // Residual cross-term
            let sum_p_residual_ref = ref_sum_p_residual as f64;
            let sum_p_residual_cand = (1.0 - sum_p_cand_at_ref_top).max(0.0);
            if sum_p_residual_ref > 1e-9 && sum_p_residual_cand > 1e-9 {
                kld_token += sum_p_residual_ref
                    * (sum_p_residual_ref.ln() - sum_p_residual_cand.ln());
            }
            // KLD ≥ 0 by Gibbs' inequality. Same rationale as eval_hipfire.rs.
            debug_assert!(
                kld_token >= -1e-9,
                "negative KLD beyond fp roundoff: {kld_token}"
            );
            let kld_token = kld_token.max(0.0);

            // NLL: -log P_cand(actual_next_token). actual next token at
            // chunk c, logit-pos (scoring_start + j) is tokens[c*n_ctx + scoring_start + j + 1].
            let actual_next = tokens[c * ref_n_ctx + scoring_start + j + 1] as usize;
            if actual_next < cand_n_vocab {
                let stored_off = 8 + actual_next * 2;
                let stored = u16::from_le_bytes(
                    cand_block_buf[stored_off..stored_off + 2].try_into().unwrap()
                );
                let log_p_cand = (cand_scale as f64) * (stored as f64) + (cand_min_log_prob as f64);
                chunk_nll_sum += -log_p_cand;
                chunk_nll_count += 1;
            }

            chunk_klds.push(kld_token);
            total_done += 1;

            if total_done % progress_interval == 0 || total_done == total_scored {
                let pct = total_done as f64 * 100.0 / total_scored as f64;
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = total_done as f64 / elapsed.max(1e-9);
                eprint!(
                    "\r  chunk {:4}/{}  scored {:8}/{:8}  ({:5.1}%, {:.0} tok/s)   ",
                    c + 1, ref_n_chunk, total_done, total_scored, pct, rate
                );
            }
        }

        // Per-chunk aggregates
        if chunk_klds.is_empty() {
            mean_kld_per_seq.push(0.0);
            p99_kld_per_seq.push(0.0);
            mean_nll_per_seq.push(f64::NAN);
            continue;
        }
        let mean: f64 = chunk_klds.iter().copied().sum::<f64>() / chunk_klds.len() as f64;
        let mut sorted = chunk_klds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        let p99 = sorted[p99_idx];
        let mean_nll = if chunk_nll_count > 0 {
            chunk_nll_sum / chunk_nll_count as f64
        } else { f64::NAN };
        mean_kld_per_seq.push(mean);
        p99_kld_per_seq.push(p99);
        mean_nll_per_seq.push(mean_nll);
    }
    eprintln!();

    // ---------- Write HFKSEQ output (v2: adds mean_nll per chunk) ----------
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("create output parent dir");
        }
    }
    let out_file = File::create(&args.output).expect("create output");
    let mut out = BufWriter::new(out_file);
    out.write_all(SEQKLD_MAGIC).unwrap();
    out.write_all(&2u32.to_le_bytes()).unwrap();                  // version = 2
    out.write_all(&(ref_n_chunk as u32).to_le_bytes()).unwrap();   // n_chunk
    out.write_all(&0u32.to_le_bytes()).unwrap();                  // reserved
    for ((m, p), n) in mean_kld_per_seq.iter()
        .zip(p99_kld_per_seq.iter())
        .zip(mean_nll_per_seq.iter())
    {
        out.write_all(&m.to_le_bytes()).unwrap();
        out.write_all(&p.to_le_bytes()).unwrap();
        out.write_all(&n.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();

    let overall_mean: f64 = mean_kld_per_seq.iter().copied().sum::<f64>() / mean_kld_per_seq.len() as f64;
    let nll_finite: Vec<f64> = mean_nll_per_seq.iter().copied().filter(|x| x.is_finite()).collect();
    let overall_nll: f64 = if nll_finite.is_empty() {
        f64::NAN
    } else {
        nll_finite.iter().copied().sum::<f64>() / nll_finite.len() as f64
    };
    let overall_ppl = overall_nll.exp();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "eval_gguf: scored {total_done} tokens in {:.1}s ({:.0} tok/s)",
        elapsed, total_done as f64 / elapsed.max(1e-9)
    );
    eprintln!(
        "eval_gguf: slice-mean KLD = {:.6}  mean NLL = {:.6}  PPL = {:.4}",
        overall_mean, overall_nll, overall_ppl
    );

    // ---------- Wait for child ----------
    let status = child.wait().expect("failed to wait on child");
    if !status.success() {
        eprintln!("warning: llama-perplexity exited with status: {}", status);
    }

    eprintln!("eval_gguf: wrote {}", args.output.display());
}

// (verify_llama_commit / verify_slice_md5 / verify_ref_sha256 now live in
// hipfire_runtime::eval_common — see crates/hipfire-runtime/src/eval_common.rs)
