// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! kv_mode_kld — asym3-vs-q8 KV-mode quality gate (SP1 Task 9, spec §13).
//!
//! Loads a qwen35 model ONCE, then:
//!   1. Free-generates greedily under `--baseline` KV mode (default q8) from
//!      a REAL prompt, recording the exact token sequence T = prompt + gen
//!      and the full-vocab logits at every generated step. This is the
//!      "model's own generated output" the method constraint requires — not
//!      a canned reference completion.
//!   2. Drops the KV cache, builds a fresh one under `--candidate` mode
//!      (default asym3), and teacher-forces the SAME sequence T (same
//!      tokens, same positions), recording full-vocab logits at the same
//!      steps.
//!   3. Computes per-step KL(baseline || candidate) and KL(candidate ||
//!      baseline) plus top-1 agreement / top-5 overlap / first-divergence
//!      (candidate's own argmax vs the token the baseline actually chose at
//!      that step), and writes one JSON row to <out>.
//!
//! Teacher-forcing the candidate on the baseline's own trajectory (rather
//! than letting both free-run independently) removes the "two runs diverge
//! in token choice after step k, so what am I even comparing" confound and
//! is what makes a rigorous full-softmax KLD possible here.
//!
//! Usage:
//!   kv_mode_kld <model.hfq> <out.json> [--max-gen N] [--max-seq N]
//!               [--baseline MODE] [--candidate MODE] [--raw] PROMPT...

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache};
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: kv_mode_kld <model.hfq> <out.json> [--max-gen N] [--max-seq N] [--baseline MODE] [--candidate MODE] [--raw] PROMPT...");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let out_path = &args[2];

    let mut max_gen: usize = 150;
    let mut max_seq: usize = 4096;
    let mut baseline_mode = "q8".to_string();
    let mut candidate_mode = "asym3".to_string();
    let mut raw_prompt = false;
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--max-gen" => {
                i += 1;
                max_gen = args[i].parse().expect("--max-gen requires N");
            }
            "--max-seq" => {
                i += 1;
                max_seq = args[i].parse().expect("--max-seq requires N");
            }
            "--baseline" => {
                i += 1;
                baseline_mode = args[i].clone();
            }
            "--candidate" => {
                i += 1;
                candidate_mode = args[i].clone();
            }
            "--raw" => {
                raw_prompt = true;
            }
            other => prompt_parts.push(other.to_string()),
        }
        i += 1;
    }
    let prompt_text = if prompt_parts.is_empty() {
        "Explain how a hash table resolves collisions.".to_string()
    } else {
        prompt_parts.join(" ")
    };

    eprintln!(
        "kv_mode_kld: model={model_path} baseline={baseline_mode} candidate={candidate_mode} max_gen={max_gen} max_seq={max_seq}"
    );

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    let prompt_tokens0: Vec<u32> = if raw_prompt {
        tokenizer.encode(&prompt_text)
    } else {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user = tokenizer.encode("user");
        let asst = tokenizer.encode("assistant");
        let nl = tokenizer.encode("\n");
        let user_body = tokenizer.encode(&prompt_text);
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&user);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&user_body);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&asst);
        chat.extend_from_slice(&nl);
        // /no_think keeps generation on-topic within max_gen for the KLD scan.
        chat.extend_from_slice(&tokenizer.encode("<think>\n</think>\n\n"));
        chat
    };
    eprintln!("prompt: {} tokens", prompt_tokens0.len());

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    fn make_kv_cache(
        gpu: &mut rdna_compute::Gpu,
        config: &qwen35::Qwen35Config,
        mode: &str,
        kv_seq: usize,
    ) -> KvCache {
        match mode {
            "q8" => KvCache::new_gpu_q8(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            ),
            "asym4" | "turbo4" => KvCache::new_gpu_asym4(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            ),
            "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            ),
            "asym2" | "turbo2" => KvCache::new_gpu_asym2(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            ),
            other => panic!("unknown kv mode: {other} (q8|asym4|asym3|asym2)"),
        }
        .unwrap()
    }

    // top-5 helper, mirrors greedy_dump_top5.rs.
    fn top5(logits: &[f32]) -> [(u32, f32); 5] {
        let mut best: [(u32, f32); 5] = [(0, f32::NEG_INFINITY); 5];
        for (idx, &v) in logits.iter().enumerate() {
            if v <= best[4].1 {
                continue;
            }
            best[4] = (idx as u32, v);
            for j in (1..5).rev() {
                if best[j].1 > best[j - 1].1 {
                    best.swap(j, j - 1);
                } else {
                    break;
                }
            }
        }
        best
    }

    // Stable softmax KL divergence: KL(P||Q) = sum_i p_i * (log p_i - log q_i).
    fn kl_div(p_logits: &[f32], q_logits: &[f32]) -> f64 {
        let p_max = p_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let q_max = q_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut p_sum = 0.0f64;
        let mut q_sum = 0.0f64;
        for &v in p_logits {
            p_sum += ((v as f64) - p_max).exp();
        }
        for &v in q_logits {
            q_sum += ((v as f64) - q_max).exp();
        }
        let log_p_sum = p_sum.ln();
        let log_q_sum = q_sum.ln();
        let mut acc = 0.0f64;
        for (&pv, &qv) in p_logits.iter().zip(q_logits.iter()) {
            let log_p = (pv as f64) - p_max - log_p_sum;
            let log_q = (qv as f64) - q_max - log_q_sum;
            let p = log_p.exp();
            if p > 0.0 {
                acc += p * (log_p - log_q);
            }
        }
        acc
    }

    // ---------- Pass 1: baseline free-generation (the model's own output) ----------
    let mut kv_cache = make_kv_cache(&mut gpu, &config, &baseline_mode, max_seq);
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let mut all_tokens: Vec<u32> = prompt_tokens0.clone();
    for (pos, &token) in prompt_tokens0.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("baseline prefill failed");
    }
    let mut baseline_logits: Vec<Vec<f32>> = Vec::new();
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);
    baseline_logits.push(logits.clone());
    all_tokens.push(next_token);

    let max_gen_eff = max_gen.min(max_seq.saturating_sub(prompt_tokens0.len() + 8));
    for step in 1..max_gen_eff {
        let pos = all_tokens.len() - 1;
        if pos >= max_seq {
            break;
        }
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            next_token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("baseline gen failed");
        logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
        baseline_logits.push(logits.clone());
        all_tokens.push(next_token);
        if next_token == config.eos_token {
            break;
        }
        if step % 50 == 0 {
            eprintln!("  baseline step {step}");
        }
    }
    let baseline_text = tokenizer.decode(&all_tokens[prompt_tokens0.len()..]);
    eprintln!(
        "baseline ({baseline_mode}) generated {} tokens",
        baseline_logits.len()
    );
    drop(kv_cache);
    drop(dn_state);

    // ---------- Pass 2: candidate, teacher-forced on the EXACT same sequence ----------
    let mut kv_cache2 = make_kv_cache(&mut gpu, &config, &candidate_mode, max_seq);
    let mut dn_state2 = DeltaNetState::new(&mut gpu, &config).unwrap();

    // Prefill the prompt (positions 0..prompt_len-1).
    for (pos, &token) in prompt_tokens0.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut kv_cache2,
            &mut dn_state2,
            &scratch,
        )
        .expect("candidate prefill failed");
    }
    // First scored step: logits after consuming the prompt (predicts
    // all_tokens[prompt_len]), matching baseline_logits[0].
    let mut candidate_logits: Vec<Vec<f32>> = Vec::new();
    let l0 = gpu.download_f32(&scratch.logits).unwrap();
    candidate_logits.push(l0);
    // Remaining scored steps: feed all_tokens[prompt_len + k] at
    // pos = prompt_len + k, matching baseline_logits[k+1].
    let n_scored = baseline_logits.len();
    for k in 0..(n_scored - 1) {
        let pos = prompt_tokens0.len() + k;
        let token = all_tokens[pos];
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut kv_cache2,
            &mut dn_state2,
            &scratch,
        )
        .expect("candidate teacher-forced step failed");
        let l = gpu.download_f32(&scratch.logits).unwrap();
        candidate_logits.push(l);
        if k % 50 == 0 {
            eprintln!("  candidate step {k}");
        }
    }
    eprintln!(
        "candidate ({candidate_mode}) teacher-forced {} steps",
        candidate_logits.len()
    );

    // ---------- Compare ----------
    let n = baseline_logits.len().min(candidate_logits.len());
    let mut kld_pq: Vec<f64> = Vec::with_capacity(n); // KL(baseline || candidate)
    let mut kld_qp: Vec<f64> = Vec::with_capacity(n); // KL(candidate || baseline)
    let mut top1_match = 0usize;
    let mut top5_overlap_sum = 0usize;
    let mut first_divergence: Option<usize> = None;
    for i in 0..n {
        let p = &baseline_logits[i];
        let q = &candidate_logits[i];
        kld_pq.push(kl_div(p, q));
        kld_qp.push(kl_div(q, p));
        let base_tok = all_tokens[prompt_tokens0.len() + i];
        let cand_top1 = llama::argmax(q);
        if cand_top1 == base_tok {
            top1_match += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(i);
        }
        let bt = top5(p);
        let ct = top5(q);
        let base_ids: std::collections::HashSet<u32> = bt.iter().map(|x| x.0).collect();
        let cand_ids: std::collections::HashSet<u32> = ct.iter().map(|x| x.0).collect();
        top5_overlap_sum += base_ids.intersection(&cand_ids).count();
    }

    fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
    fn median(v: &[f64]) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = s.len() / 2;
        if s.len() % 2 == 0 {
            (s[mid - 1] + s[mid]) / 2.0
        } else {
            s[mid]
        }
    }

    let mean_kld_pq = mean(&kld_pq);
    let median_kld_pq = median(&kld_pq);
    let mean_kld_qp = mean(&kld_qp);
    let median_kld_qp = median(&kld_qp);
    let mut sorted_pq = kld_pq.clone();
    sorted_pq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99_kld_pq = if !sorted_pq.is_empty() {
        sorted_pq[(((sorted_pq.len() as f64) * 0.99) as usize).min(sorted_pq.len() - 1)]
    } else {
        f64::NAN
    };

    let out = serde_json::json!({
        "model": model_path,
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "prompt": prompt_text,
        "prompt_tokens": prompt_tokens0.len(),
        "max_seq": max_seq,
        "steps_compared": n,
        "mean_kld_baseline_vs_candidate": mean_kld_pq,
        "median_kld_baseline_vs_candidate": median_kld_pq,
        "p99_kld_baseline_vs_candidate": p99_kld_pq,
        "mean_kld_candidate_vs_baseline": mean_kld_qp,
        "median_kld_candidate_vs_baseline": median_kld_qp,
        "top1_agreement": (top1_match as f64) / (n as f64),
        "mean_top5_overlap": (top5_overlap_sum as f64) / (n as f64),
        "first_divergence": first_divergence,
        "baseline_generated_text": baseline_text,
        "kld_per_step_baseline_vs_candidate": kld_pq,
    });
    let mut f = std::fs::File::create(out_path).expect("create out.json");
    f.write_all(serde_json::to_string_pretty(&out).unwrap().as_bytes())
        .unwrap();
    eprintln!(
        "kv_mode_kld: n={n} mean_kld(P||Q)={mean_kld_pq:.6} median_kld(P||Q)={median_kld_pq:.6} top1_agree={:.4} first_div={:?}",
        (top1_match as f64) / (n as f64),
        first_divergence
    );
    eprintln!("wrote {out_path}");
}
