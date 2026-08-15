// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tiny-oracle correctness harness for the Gemma 4 EAGLE drafter (arch_id 22).
//!
//! The drafter only affects spec-decode acceptance (τ), not output correctness
//! (the target verify guarantees correctness downstream), so the gate is
//! **logits/hidden cosine ≥ 0.99 vs the HF reference** — proving the math is
//! right. Pipeline:
//!
//!   1. Load tiny TARGET (arch 13) + tiny DRAFTER (arch 22) HFQs.
//!   2. Run the target through hipfire `decode_step` over the prompt tokens to
//!      populate its sliding+full KV caches. The target's NORMED final hidden
//!      (consumed by its lm_head) is left in `target_state.tmp`.
//!   3. `--inject-hf` (default ON): overwrite the target's LAST sliding + LAST
//!      full KV slot with the HF reference K/V (post-k_norm + RoPE + v_norm,
//!      re-quantized to Q8), and feed HF's exact target last-hidden as the
//!      step-0 hidden half. This ISOLATES the drafter forward math from the
//!      tiny target's per-layer numeric drift (a known scale artifact: even the
//!      validated arch-13 target only reaches ~0.88 min-cos by layer 5 at this
//!      tiny random-weight scale). Without `--inject-hf` the drafter consumes
//!      hipfire's own (drifted) target hidden+KV — a looser, end-to-end check.
//!   4. Run `drafter_step` for N steps, mirroring the HF candidate generator:
//!      step 0 → prev = last prompt token, hidden = target last-hidden;
//!      step i>0 → prev = prev step argmax, hidden = prev step post_proj.
//!      The drafter reads the target's LAST sliding + LAST full KV slot directly
//!      (no drafter KV cache, no writes), at the CONSTANT query_pos = n_ctx-1.
//!   5. Compare per step vs the HF oracle: normed-hidden cosine, logits cosine,
//!      argmax match, post_proj cosine.
//!
//! PASS: every step's logits cosine ≥ 0.99 and argmax matches.
//!
//! Usage:
//!   verify_drafter_gemma4 --target <target.hfq> --drafter <drafter.hfq> \
//!                         --oracle <drafter_oracle.bin> [--no-inject-hf]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::config::Gemma4Config;
    use hipfire_arch_gemma4::drafter::{
        drafter_step, drafter_step_from_concat, Gemma4DrafterConfig, Gemma4DrafterScratch,
        Gemma4DrafterWeights,
    };
    use hipfire_arch_gemma4::forward::decode_step;
    use hipfire_arch_gemma4::gemma4::{Gemma4State, Gemma4Weights};
    use hipfire_runtime::hfq::HfqFile;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut target_path: Option<PathBuf> = None;
    let mut drafter_path: Option<PathBuf> = None;
    let mut oracle_path: Option<PathBuf> = None;
    // Default gate = per-step isolation: each draft step gets HF's exact
    // target-hidden + the target's HF-reference shared KV, so we measure the
    // drafter FORWARD MATH (not the AR feedback chain, whose drift is corrected
    // by the target's verify step downstream — the drafter only sets τ).
    let mut inject_hf = true;
    let mut inject_every_step = true;
    let mut inject_pre_in = false;
    let mut feedback = false; // --feedback = run the realistic AR chain instead
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--target" => {
                target_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--drafter" => {
                drafter_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--oracle" => {
                oracle_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--no-inject-hf" => {
                inject_hf = false;
                i += 1;
            }
            "--inject-every-step" => {
                // Per-step isolation: feed HF's exact hidden half AND HF's prev
                // token each step (no feedback drift) — proves per-step math.
                inject_every_step = true;
                i += 1;
            }
            "--inject-pre-in" => {
                // Strongest isolation: feed HF's exact full concat (both the
                // embed half AND the hidden half) directly into pre_projection,
                // bypassing hipfire's Q8 embed lookup + feedback entirely. This
                // proves the pre_proj → backbone → norm → heads math in pure
                // isolation against the HF reference.
                inject_pre_in = true;
                inject_every_step = false;
                i += 1;
            }
            "--feedback" => {
                // Realistic AR chain: feed back the drafter's own argmax +
                // post_proj each step (drift is EXPECTED — corrected downstream
                // by the target verify; the drafter only affects acceptance τ).
                feedback = true;
                inject_every_step = false;
                i += 1;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let target_path = target_path.expect("--target required");
    let drafter_path = drafter_path.expect("--drafter required");
    let oracle_path = oracle_path.expect("--oracle required");

    // ── Read the HF oracle binary ──
    let raw = std::fs::read(&oracle_path).expect("read oracle");
    assert_eq!(&raw[0..8], b"HFDRAFT\0", "bad oracle magic");
    let mut off = 8usize;
    let rdu = |b: &[u8], o: &mut usize| -> usize {
        let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()) as usize;
        *o += 4;
        v
    };
    let n_ctx = rdu(&raw, &mut off);
    let n_draft = rdu(&raw, &mut off);
    let h = rdu(&raw, &mut off); // drafter hidden
    let bb = rdu(&raw, &mut off); // backbone hidden
    let vocab = rdu(&raw, &mut off);
    let query_pos = rdu(&raw, &mut off);
    let n_kv_s = rdu(&raw, &mut off);
    let hd_s = rdu(&raw, &mut off);
    let n_kv_f = rdu(&raw, &mut off);
    let hd_f = rdu(&raw, &mut off);
    let tokens: Vec<u32> = (0..n_ctx)
        .map(|_| {
            let v = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
            off += 4;
            v
        })
        .collect();
    let rdf = |b: &[u8], o: &mut usize, n: usize| -> Vec<f32> {
        let v: Vec<f32> = (0..n)
            .map(|k| f32::from_le_bytes(b[*o + k * 4..*o + k * 4 + 4].try_into().unwrap()))
            .collect();
        *o += n * 4;
        v
    };
    let target_last_hidden_hf = rdf(&raw, &mut off, bb);
    // shared K/V blocks ([seq, n_kv*hd] each), seq-major head-major. Written by
    // the gen script BEFORE the per-step records.
    let ks_hf = rdf(&raw, &mut off, n_ctx * n_kv_s * hd_s);
    let vs_hf = rdf(&raw, &mut off, n_ctx * n_kv_s * hd_s);
    let kf_hf = rdf(&raw, &mut off, n_ctx * n_kv_f * hd_f);
    let vf_hf = rdf(&raw, &mut off, n_ctx * n_kv_f * hd_f);
    struct OStep {
        prev_token: u32,
        argmax: u32,
        pre_in: Vec<f32>,        // full HF concat [2*bb]
        pre_in_hidden: Vec<f32>, // HF hidden half (second bb floats of pre_in)
        pre_out: Vec<f32>,
        normed: Vec<f32>,
        post_proj: Vec<f32>,
        logits: Vec<f32>,
    }
    let mut osteps = Vec::with_capacity(n_draft);
    for _ in 0..n_draft {
        let prev_token = rdu(&raw, &mut off) as u32;
        let argmax = rdu(&raw, &mut off) as u32;
        let pre_in = rdf(&raw, &mut off, 2 * bb);
        let pre_in_hidden = pre_in[bb..2 * bb].to_vec();
        let pre_out = rdf(&raw, &mut off, h);
        let normed = rdf(&raw, &mut off, h);
        let post_proj = rdf(&raw, &mut off, bb);
        let logits = rdf(&raw, &mut off, vocab);
        osteps.push(OStep {
            prev_token,
            argmax,
            pre_in,
            pre_in_hidden,
            pre_out,
            normed,
            post_proj,
            logits,
        });
    }
    eprintln!(
        "oracle: n_ctx={n_ctx} n_draft={n_draft} hidden={h} backbone={bb} vocab={vocab} query_pos={query_pos} \
         sliding(kv={n_kv_s},hd={hd_s}) full(kv={n_kv_f},hd={hd_f})  inject_hf={inject_hf}"
    );
    eprintln!("tokens: {tokens:?}");

    // ── Load models ──
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let target_hfq = HfqFile::open(&target_path).expect("open target");
    let tcfg = Gemma4Config::from_hfq(&target_hfq).expect("target config");
    eprintln!(
        "target: dim={} layers={} sliding={} full={} vocab={} softcap={}",
        tcfg.dim,
        tcfg.n_layers,
        tcfg.n_sliding_layers(),
        tcfg.n_full_layers(),
        tcfg.vocab_size,
        tcfg.final_logit_softcapping
    );
    let target_weights = Gemma4Weights::load(&target_hfq, &tcfg, &mut gpu).expect("target weights");

    let drafter_hfq = HfqFile::open(&drafter_path).expect("open drafter");
    let dcfg = Gemma4DrafterConfig::from_hfq(&drafter_hfq).expect("drafter config");
    eprintln!(
        "drafter: hidden={} layers={} n_heads={} sliding(hd={},kv={}) full(hd={},kv={}) backbone={} vocab={} softcap={} centroids={}/{}",
        dcfg.hidden, dcfg.n_layers, dcfg.n_heads,
        dcfg.sliding_head_dim, dcfg.sliding_n_kv_heads,
        dcfg.full_head_dim, dcfg.full_n_kv_heads,
        dcfg.backbone_hidden, dcfg.vocab_size, dcfg.final_logit_softcapping,
        dcfg.num_centroids, dcfg.centroid_top_k,
    );
    assert_eq!(dcfg.backbone_hidden, tcfg.dim, "drafter backbone must equal target hidden");
    let drafter_weights =
        Gemma4DrafterWeights::load(&drafter_hfq, &dcfg, &mut gpu).expect("drafter weights");
    let mut dscratch = Gemma4DrafterScratch::new(&mut gpu, &dcfg).expect("drafter scratch");

    // ── Run the target to populate its KV caches ──
    let max_seq = n_ctx + 8;
    let mut tstate = Gemma4State::new_with_max_seq(&mut gpu, &tcfg, max_seq).expect("target state");
    for (pos, &t) in tokens.iter().enumerate() {
        decode_step(&tcfg, &target_weights, &mut tstate, &mut gpu, t, pos as u32)
            .expect("target decode_step");
    }
    let target_normed = gpu.download_f32(&tstate.tmp).expect("download target tmp");

    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-30)
    };

    let tcos = cosine(&target_normed, &target_last_hidden_hf);
    eprintln!(
        "[sanity] hipfire target normed-final-hidden vs HF cosine = {tcos:.6}"
    );

    // ── Optional: inject HF-exact KV into the target's last slots ──
    if inject_hf {
        let pos_buf = gpu.hip.malloc(4).expect("inject pos_buf");
        // last sliding slot
        let s_slot = tcfg.n_sliding_layers() - 1;
        let f_slot = tcfg.n_full_layers() - 1;
        for pos in 0..n_ctx {
            // sliding K/V
            {
                let kvec = &ks_hf[pos * n_kv_s * hd_s..(pos + 1) * n_kv_s * hd_s];
                let vvec = &vs_hf[pos * n_kv_s * hd_s..(pos + 1) * n_kv_s * hd_s];
                let kt = gpu.upload_f32(kvec, &[n_kv_s * hd_s]).expect("up k");
                let vt = gpu.upload_f32(vvec, &[n_kv_s * hd_s]).expect("up v");
                let ph = [pos as i32];
                gpu.hip
                    .memcpy_htod(&pos_buf, unsafe {
                        std::slice::from_raw_parts(ph.as_ptr() as *const u8, 4)
                    })
                    .expect("pos htod");
                gpu.kv_cache_write_q8_0(&tstate.kv_sliding.k_gpu[s_slot], &kt, &pos_buf, n_kv_s, hd_s)
                    .expect("inject sliding k");
                gpu.kv_cache_write_q8_0(&tstate.kv_sliding.v_gpu[s_slot], &vt, &pos_buf, n_kv_s, hd_s)
                    .expect("inject sliding v");
            }
            // full K/V
            {
                let kvec = &kf_hf[pos * n_kv_f * hd_f..(pos + 1) * n_kv_f * hd_f];
                let vvec = &vf_hf[pos * n_kv_f * hd_f..(pos + 1) * n_kv_f * hd_f];
                let kt = gpu.upload_f32(kvec, &[n_kv_f * hd_f]).expect("up kf");
                let vt = gpu.upload_f32(vvec, &[n_kv_f * hd_f]).expect("up vf");
                let ph = [pos as i32];
                gpu.hip
                    .memcpy_htod(&pos_buf, unsafe {
                        std::slice::from_raw_parts(ph.as_ptr() as *const u8, 4)
                    })
                    .expect("pos htod f");
                gpu.kv_cache_write_q8_0(&tstate.kv_full.k_gpu[f_slot], &kt, &pos_buf, n_kv_f, hd_f)
                    .expect("inject full k");
                gpu.kv_cache_write_q8_0(&tstate.kv_full.v_gpu[f_slot], &vt, &pos_buf, n_kv_f, hd_f)
                    .expect("inject full v");
            }
        }
        eprintln!("[inject] overwrote target last-sliding (slot {s_slot}) + last-full (slot {f_slot}) KV with HF reference (Q8)");
    }

    // step-0 hidden half: HF-exact when injecting, else hipfire's normed hidden.
    let step0_hidden = if inject_hf {
        target_last_hidden_hf.clone()
    } else {
        target_normed.clone()
    };
    let mut hidden_backbone = gpu.upload_f32(&step0_hidden, &[bb]).expect("upload step0 hidden");

    // ── Drafter loop ──
    let last_prompt_token = *tokens.last().unwrap();
    let mut prev_token = last_prompt_token;
    let mut all_pass = true;
    let mut step0_pass = false;
    let _ = feedback;
    let mode = if inject_every_step {
        "per-step-isolation (HF hidden+KV, Q8 embed) [DEFAULT GATE]"
    } else if inject_pre_in {
        "pre-in-isolation (HF concat+KV)"
    } else {
        "feedback AR-chain (drift expected; corrected by target verify)"
    };
    println!("\n=== Drafter steps: {mode} (query_pos={query_pos}) ===");
    for step in 0..n_draft {
        let o = &osteps[step];
        if inject_every_step {
            // Feed HF's exact prev token + HF's exact hidden half each step.
            prev_token = o.prev_token;
            hidden_backbone = gpu
                .upload_f32(&o.pre_in_hidden, &[bb])
                .expect("upload hf hidden");
        } else if !inject_pre_in && prev_token != o.prev_token {
            eprintln!(
                "[warn] step {step}: hipfire prev_token {prev_token} != oracle {} — feeding hipfire's",
                o.prev_token
            );
        }
        let out = if inject_pre_in {
            drafter_step_from_concat(
                &mut gpu,
                &drafter_weights,
                &dcfg,
                &mut dscratch,
                &tstate,
                &tcfg,
                &o.pre_in,
                query_pos,
            )
            .expect("drafter_step_from_concat")
        } else {
            drafter_step(
                &mut gpu,
                &drafter_weights,
                &dcfg,
                &mut dscratch,
                &target_weights,
                &tstate,
                &tcfg,
                prev_token,
                &hidden_backbone,
                query_pos,
            )
            .expect("drafter_step")
        };

        let _ = &o.pre_out;
        let normed_cos = cosine(&out.normed_hidden, &o.normed);
        let logits_cos = cosine(&out.logits, &o.logits);
        let post_cos = cosine(&out.post_proj_hidden, &o.post_proj);
        let argmax_match = out.argmax == o.argmax;
        let pass = logits_cos >= 0.99 && normed_cos >= 0.99 && argmax_match;
        if step == 0 {
            step0_pass = pass;
        }
        all_pass &= pass;
        println!(
            "step {step}: prev={prev_token:<5} normed_cos={normed_cos:.6} logits_cos={logits_cos:.6} \
             post_cos={post_cos:.6} argmax=hip:{} hf:{} {} => {}",
            out.argmax,
            o.argmax,
            if argmax_match { "MATCH" } else { "MISS" },
            if pass { "PASS" } else { "FAIL" }
        );

        prev_token = out.argmax;
        hidden_backbone = gpu
            .upload_f32(&out.post_proj_hidden, &[bb])
            .expect("upload next hidden");
    }

    // Gate: per-step isolation requires EVERY step ≥0.99; the realistic
    // feedback chain only requires step 0 (drift downstream is corrected by the
    // target's verify — the drafter affects acceptance τ, not correctness).
    let overall = if feedback {
        step0_pass
    } else {
        all_pass
    };
    println!(
        "\n=== OVERALL ({mode}): {} ===",
        if overall { "PASS" } else { "FAIL" }
    );
    if !overall {
        std::process::exit(1);
    }
}
