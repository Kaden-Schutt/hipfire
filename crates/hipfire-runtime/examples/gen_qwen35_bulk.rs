// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash bulk data-gen: ONE model load, MANY target-regenerated sequences.
// Seeds from (kldref chunk x offset) for diversity; greedy AR generation with
// per-layer hidden capture; dumps ONLY the 5 selected layers (52 MB/seq vs
// 640 MB) + a .toks file. Per-sequence GPU state is pool-scoped so memory is
// stable across thousands of sequences.
//
//   gen_qwen35_bulk --model <hfq> --ref <kldref> --out-dir <dir> --n-seqs M
//                   [--n 512] [--seed-len 32] [--start 0] [--kv-mode q8]

/// Extract top-K (descending) from a logit row, returns (token_id, logit) pairs.
/// Uses a linear min-heap scan: O(n*log(k)).
#[cfg(feature = "deltanet")]
fn partial_topk(lg: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(lg.len());
    // Simple partial sort: maintain a min-heap of (logit, idx) with capacity k
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (i, &v) in lg.iter().enumerate() {
        if heap.len() < k {
            heap.push((v, i as u32));
            if heap.len() == k {
                // sift into min-heap (by logit)
                heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if v > heap[0].0 {
            heap[0] = (v, i as u32);
            // re-sort to maintain min at front
            let mut j = 0usize;
            loop {
                let mut m = j;
                let l = 2 * j + 1; let r = 2 * j + 2;
                if l < heap.len() && heap[l].0 < heap[m].0 { m = l; }
                if r < heap.len() && heap[r].0 < heap[m].0 { m = r; }
                if m == j { break; }
                heap.swap(j, m);
                j = m;
            }
        }
    }
    // sort descending by logit
    heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    heap.into_iter().map(|(logit, tid)| (tid, logit)).collect()
}

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch, forward_scratch_with_hidden_ws};
    use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{weight_gemv, KvCache};
    use std::fs::File;
    use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
    use std::path::PathBuf;

    let mut sel: Vec<usize> = vec![2, 16, 31, 46, 61];
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None; let mut ref_path = None; let mut out_dir = None;
    let mut n = 512usize; let mut seed_len = 32usize; let mut n_seqs = 64usize; let mut start = 0usize; let mut kv_mode = "q8".to_string();
    let dump_topk = std::env::var("HIPFIRE_DUMP_TOPK").ok().as_deref() == Some("1");
    let topk_k: usize = std::env::var("HIPFIRE_KL_TOPK").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let mut rep_penalty = 1.3f32; let mut rep_window = 64usize;
    let mut temp = 0.0f32;
    let mut rng_seed: u64 = 0x9E3779B97F4A7C15;
    let ws_state_couple = std::env::var("HIPFIRE_WS_STATE_COUPLE").ok().as_deref() == Some("1");
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--ref" => { ref_path = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--out-dir" => { out_dir = Some(argv[i + 1].clone()); i += 2; }
            "--n" => { n = argv[i + 1].parse().unwrap(); i += 2; }
            "--seed-len" => { seed_len = argv[i + 1].parse().unwrap(); i += 2; }
            "--n-seqs" => { n_seqs = argv[i + 1].parse().unwrap(); i += 2; }
            "--start" => { start = argv[i + 1].parse().unwrap(); i += 2; }
            "--sel" => { sel = argv[i + 1].split(',').map(|x| x.parse().unwrap()).collect(); i += 2; }
            "--rep-penalty" => { rep_penalty = argv[i + 1].parse().unwrap(); i += 2; }
            "--rep-window" => { rep_window = argv[i + 1].parse().unwrap(); i += 2; }
            "--kv-mode" => { kv_mode = argv[i + 1].clone(); i += 2; }
            "--temp" => { temp = argv[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { rng_seed = argv[i + 1].parse().unwrap(); i += 2; }
            o => { eprintln!("unknown {o}"); std::process::exit(1); }
        }
    }
    let model = model.expect("--model"); let ref_path = ref_path.expect("--ref"); let out_dir = out_dir.expect("--out-dir");
    unsafe { std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0"); std::env::set_var("HIPFIRE_GRAPH", "0"); std::env::set_var("HIPFIRE_KV_MODE", &kv_mode); }
    std::fs::create_dir_all(&out_dir).ok();

    // load kldref tokens (all chunks) for seeds
    let mut rf = File::open(&ref_path).expect("ref");
    let mut magic = [0u8; 8]; rf.read_exact(&mut magic).unwrap();
    let mut hdr = [0u8; 24]; rf.read_exact(&mut hdr).unwrap();
    let n_ctx_ref = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let n_chunk = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let mut all_toks = vec![0u8; n_chunk * n_ctx_ref * 4];
    rf.seek(SeekFrom::Start(32)).unwrap(); rf.read_exact(&mut all_toks).unwrap();
    let toks_at = |chunk: usize, off: usize, len: usize| -> Vec<u32> {
        let base = (chunk * n_ctx_ref + off) * 4;
        all_toks[base..base + len * 4].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    };

    let mut hfq = HfqFile::open(&model).expect("hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let eps = config.norm_eps;
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch");

    // seed plan: distinct (chunk, offset). offsets stride so each is a different start.
    let max_off = n_ctx_ref.saturating_sub(seed_len + 1).max(1);
    let offs_per_chunk = ((n_seqs + n_chunk - 1) / n_chunk).max(1);
    let stride = (max_off / offs_per_chunk).max(1);

    // xorshift64* state for temperature sampling (seeded, deterministic)
    let mut rng_state: u64 = rng_seed | 1;
    #[inline]
    fn xorshift_unit(state: &mut u64) -> f32 {
        let mut s = *state;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *state = s;
        ((s >> 40) as f32) * (1.0 / 16_777_216.0)
    }
    // WS state: n_la_layers is determined from the config
    let n_la_layers = config.layer_types.iter().filter(|t| **t == hipfire_arch_qwen35::qwen35::LayerType::LinearAttention).count();
    let ws_summary_dim = n_la_layers * config.linear_num_value_heads; // alpha per LA layer
    if ws_state_couple {
        eprintln!("WS state-coupling ON: n_la_layers={n_la_layers} n_v_heads={} summary_dim={ws_summary_dim}", config.linear_num_value_heads);
    }
    gpu.pool_begin_scope();
    let t_all = std::time::Instant::now();
    for s in 0..n_seqs {
        let idx = start + s;
        let chunk = idx % n_chunk;
        let off = ((idx / n_chunk) * stride) % max_off;
        let seed = toks_at(chunk, off, seed_len);
        let ck = gpu.pool_checkpoint();
        let mut topk_rows: Vec<Vec<(u32, f32)>> = if dump_topk { Vec::with_capacity(n) } else { Vec::new() };
        let mut ws_alpha_per_pos: Vec<f32> = if ws_state_couple { Vec::with_capacity(n * ws_summary_dim) } else { Vec::new() };
        let mut ws_alpha_tmp: Vec<f32> = Vec::new();
        let mut kv_cache = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, n + 16).unwrap();
        let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
        let mut rb = HiddenStateRingBuffer::new(&mut gpu, config.n_layers, sel.len(), config.dim, n, 1).unwrap();
        rb.extract_layers = sel.clone();
        let mut generated: Vec<u32> = Vec::with_capacity(n);
        let mut tok = seed[0];
        for pos in 0..n {
            if ws_state_couple {
                forward_scratch_with_hidden_ws(&mut gpu, &weights, &config, tok, pos, &mut kv_cache, &mut dn_state, &scratch, &mut rb, &mut ws_alpha_tmp).expect("fwd_ws");
                ws_alpha_per_pos.extend_from_slice(&ws_alpha_tmp);
            } else {
                qwen35::forward_scratch_with_hidden(&mut gpu, &weights, &config, tok, pos, &mut kv_cache, &mut dn_state, &scratch, &mut rb).expect("fwd");
            }
            generated.push(tok);
            gpu.rmsnorm_f32(&scratch.x, &weights.output_norm, &scratch.tmp, eps).unwrap();
            weight_gemv(&mut gpu, &weights.output, &scratch.tmp, &scratch.logits).unwrap();
            let mut lg = gpu.download_f32(&scratch.logits).unwrap();
            // collect raw top-K logits for KL distillation (before rep penalty)
            if dump_topk { topk_rows.push(partial_topk(&lg, topk_k)); }
            // repetition penalty over the recent window (break greedy loops)
            let w0 = generated.len().saturating_sub(rep_window);
            for &t in &generated[w0..] {
                let v = &mut lg[t as usize];
                *v = if *v > 0.0 { *v / rep_penalty } else { *v * rep_penalty };
            }
            let next = if temp > 0.0 {
                // temperature sampling: softmax(logits/T) then categorical draw
                let inv_t = 1.0 / temp;
                let mut max_l = f32::NEG_INFINITY;
                for &v in &lg { if v * inv_t > max_l { max_l = v * inv_t; } }
                let mut sum = 0.0f32;
                let mut probs: Vec<f32> = lg.iter().map(|&v| { let e = (v * inv_t - max_l).exp(); sum += e; e }).collect();
                let inv_sum = 1.0 / sum;
                for p in probs.iter_mut() { *p *= inv_sum; }
                let u = xorshift_unit(&mut rng_state);
                let mut acc = 0.0f32;
                let mut sampled = (probs.len() - 1) as u32;
                for (j, &p) in probs.iter().enumerate() { acc += p; if u < acc { sampled = j as u32; break; } }
                sampled
            } else {
                let mut bi = 0usize; let mut bv = lg[0];
                for j in 1..lg.len() { if lg[j] > bv { bv = lg[j]; bi = j; } }
                bi as u32
            };
            tok = if pos + 1 < seed_len { seed[pos + 1] } else { next };
        }
        // dump 5 layers (HFHS n_layers=5, in SEL order) + toks
        let hp = format!("{out_dir}/seq{idx}.hfhs");
        let mut out = BufWriter::with_capacity(4 << 20, File::create(&hp).unwrap());
        out.write_all(b"HFHS\0\0\0\0").unwrap();
        out.write_all(&(sel.len() as u32).to_le_bytes()).unwrap();
        out.write_all(&(n as u32).to_le_bytes()).unwrap();
        out.write_all(&(config.dim as u32).to_le_bytes()).unwrap();
        out.write_all(&0u32.to_le_bytes()).unwrap();
        for buf in rb.layer_bufs.iter() {
            let d = gpu.download_f32(buf).unwrap();
            let raw: &[u8] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
            out.write_all(raw).unwrap();
        }
        out.flush().unwrap();
        let mut tf = BufWriter::new(File::create(format!("{out_dir}/seq{idx}.toks")).unwrap());
        for t in &generated { tf.write_all(&t.to_le_bytes()).unwrap(); }
        tf.flush().unwrap();
        // write .topk sibling: HFTK header + [n_pos][K] (u32 token_id, f32 logit) pairs
        if dump_topk && !topk_rows.is_empty() {
            let k_actual = topk_rows[0].len() as u32;
            let n_pos_actual = topk_rows.len() as u32;
            let mut kf = BufWriter::new(File::create(format!("{out_dir}/seq{idx}.topk")).unwrap());
            // 20-byte header: magic(8) + n_pos(4) + K(4) + vocab(4) + pad(4) = 24 bytes
            kf.write_all(b"HFTK\0\0\0\0").unwrap();
            kf.write_all(&n_pos_actual.to_le_bytes()).unwrap();
            kf.write_all(&k_actual.to_le_bytes()).unwrap();
            kf.write_all(&(config.vocab_size as u32).to_le_bytes()).unwrap();
            kf.write_all(&0u32.to_le_bytes()).unwrap();
            for row in &topk_rows {
                for &(tid, logit) in row {
                    kf.write_all(&tid.to_le_bytes()).unwrap();
                    kf.write_all(&logit.to_le_bytes()).unwrap();
                }
            }
            kf.flush().unwrap();
        }
        // write .wsstate sibling: WS state summary (alpha gates per LA layer)
        // Format: magic(8) + n_pos(4) + summary_dim(4) + n_la_layers(4) + n_v_heads(4) = 24 bytes header
        //         then [n_pos * summary_dim] f32 row-major
        if ws_state_couple && !ws_alpha_per_pos.is_empty() {
            let mut wf = BufWriter::new(File::create(format!("{out_dir}/seq{idx}.wsstate")).unwrap());
            wf.write_all(b"WSST    ").unwrap();
            wf.write_all(&(n as u32).to_le_bytes()).unwrap();
            wf.write_all(&(ws_summary_dim as u32).to_le_bytes()).unwrap();
            wf.write_all(&(n_la_layers as u32).to_le_bytes()).unwrap();
            wf.write_all(&(config.linear_num_value_heads as u32).to_le_bytes()).unwrap();
            let raw: &[u8] = unsafe { std::slice::from_raw_parts(ws_alpha_per_pos.as_ptr() as *const u8, ws_alpha_per_pos.len() * 4) };
            wf.write_all(raw).unwrap();
            wf.flush().unwrap();
        }
        gpu.pool_release_to(ck);
        if s % 8 == 0 || s + 1 == n_seqs {
            let uniq: std::collections::HashSet<u32> = generated[seed_len..].iter().copied().collect();
            eprintln!("  seq {idx} (chunk {chunk} off {off}): uniq {:.3}  [{}/{} {:.0}s]", uniq.len() as f32 / (n - seed_len) as f32, s + 1, n_seqs, t_all.elapsed().as_secs_f32());
        }
    }
    println!("gen_qwen35_bulk: OK {n_seqs} seqs -> {out_dir} ({:.0}s)", t_all.elapsed().as_secs_f32());
}
