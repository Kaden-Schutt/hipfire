// SPDX-License-Identifier: Apache-2.0
//! Batched verify parity harness for Muse Glimmer.
//! Proves that `verify_block_with_capture` (batched, one weight read per window)
//! is byte/argmax-identical to `verify_block` (B sequential decode_steps) and
//! that KV state after a batched verify is identical (next-token argmax matches).
//! For each B in {1,2,4,8,16}, does a 10-token prefill, then verifies a block
//! of B tokens and asserts picks and post-batch next-token argmax are identical.
use hipfire_arch_muse_glimmer::config::GlimmerConfig;
use hipfire_arch_muse_glimmer::forward::{decode_step, verify_block, verify_block_with_capture};
use hipfire_arch_muse_glimmer::glimmer::{GlimmerState, GlimmerWeights};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() {
    let model = std::env::args().nth(1).expect("model path: verify_batch_glimmer <model.mq4>");
    let mut gpu = Gpu::init().expect("gpu init");
    let hfq = HfqFile::open(std::path::Path::new(&model)).expect("open");
    let cfg = GlimmerConfig::from_hfq(&hfq).expect("cfg");
    eprintln!(
        "glimmer dim={} layers={} vocab={} window={} q8_lm={}",
        cfg.dim,
        cfg.n_layers,
        cfg.vocab_size,
        cfg.sliding_window,
        cfg.vocab_size // placeholder
    );
    let weights = GlimmerWeights::load(&hfq, &cfg, &mut gpu).expect("weights");
    // Deterministic token ids (avoid tokenizer dependency). First is BOS.
    let prompt_tokens: Vec<u32> = {
        let mut v = vec![200000];
        v.extend((0..100).map(|i| ((10 + i * 13) % 202040) as u32));
        v
    };
    for &b in &[1usize, 2, 4, 8, 16] {
        eprintln!("=== B={} ===", b);
        let mut state_seq = GlimmerState::new(&mut gpu, &cfg).expect("state");
        for (i, &tok) in prompt_tokens[..10].iter().enumerate() {
            decode_step(&cfg, &weights, &mut state_seq, &mut gpu, tok, i as u32).expect("decode");
        }
        let start_pos = 10;
        let block: Vec<u32> = prompt_tokens[10..10 + b].to_vec();
        let seq_picks = verify_block(&cfg, &weights, &mut state_seq, &mut gpu, &block, start_pos as u32).expect("seq verify");

        let mut state_bat = GlimmerState::new(&mut gpu, &cfg).expect("state");
        for (i, &tok) in prompt_tokens[..10].iter().enumerate() {
            decode_step(&cfg, &weights, &mut state_bat, &mut gpu, tok, i as u32).expect("decode");
        }
        let mut hidden_out = Vec::new();
        let bat_picks = verify_block_with_capture(&cfg, &weights, &mut state_bat, &mut gpu, &block, start_pos as u32, &[], &mut hidden_out).expect("bat verify");
        assert_eq!(seq_picks, bat_picks, "B={}: picks mismatch seq {:?} bat {:?}", b, seq_picks, bat_picks);
        eprintln!(" B={}: picks MATCH {:?}", b, &seq_picks[..seq_picks.len().min(4)]);

        // KV parity: decode one more token and check next-token argmax matches.
        let next_tok = 9999u32;
        let seq_next = decode_step(&cfg, &weights, &mut state_seq, &mut gpu, next_tok, (start_pos + b) as u32).expect("seq next");
        let seq_pick = seq_next.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap();
        let bat_next = decode_step(&cfg, &weights, &mut state_bat, &mut gpu, next_tok, (start_pos + b) as u32).expect("bat next");
        let bat_pick = bat_next.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap();
        assert_eq!(seq_pick, bat_pick, "B={}: post-batch next-token mismatch seq {} bat {}", b, seq_pick, bat_pick);
        eprintln!("  post-batch next token MATCH {}", seq_pick);

        state_seq.free_gpu(&mut gpu);
        state_bat.free_gpu(&mut gpu);
    }
    eprintln!("All B parity checks passed (picks and KV).");
}
