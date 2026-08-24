// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SP3 Task 3 — the correctness gate for `forward_batch_slots` (SP3 Task 2).
// Task 2 was build-only: nothing before this harness has checked that the
// N-slot forward produces the same numbers as the existing single-sequence
// path. This is that check.
//
// COMPARISON IS ON LOGITS, NOT TOKENS. Sampling makes runs diverge, and
// greedy decoding on a quantised model can diverge on a near-tie even when
// both paths are correct — token-level equivalence is not a sound check
// here. For `n_slots` in 1..=4, with per-slot prompts of differing length,
// each slot's full step sequence (one prefill + a few decodes) is run twice:
// once alone through the existing single-sequence `forward_prefill_batch`,
// and once as part of an n_slots-wide batch through `forward_batch_slots`.
// Per-step logits are compared with a tolerance-based `assert_close` —
// copied from `rdna-compute/examples/test_batched_attn_slots.rs` (SP1's
// harness) rather than reinvented, including its two hard-won guards: a
// finiteness check (two independently-NaN arrays would otherwise "agree" at
// a perfect 0.000x) and a non-degeneracy check that rejects an all-zero
// reference (SP1 found two all-zero arrays passing at 0.000x tolerance).
//
// NEGATIVE CONTROL. `run_negative_control` redirects one row's `row_slot`
// entry in the CANDIDATE arm's `SlotBatch` — the harness-level analogue of
// "two slots pointed at the same SlotId" — and asserts the resulting
// mismatch against the (uncorrupted) single-sequence reference. Two slots of
// UNEQUAL length are used, and the redirect is applied at a decode step
// where the two slots' absolute positions have already diverged (7 vs 10).
// This is deliberate: if the redirected row happened to share its target
// slot's KV write offset with a real, un-redirected row from that slot (the
// case at equal-length equal-position steps), the two rows would race to
// write the same slab bytes and, depending on which one lands last, the
// corrupted read could accidentally come back correct — defeating the
// control the way SP1's *first* attempt at this kind of check did (see
// `test_batched_attn_slots.rs`'s `maybe_corrupt` doc comment: corrupting
// both arms let a wrong answer agree with itself). Distinct lengths ensure
// every step's absolute positions are distinct across slots, so a redirected
// write always lands at an offset no real row is also writing to that step,
// and the redirected read is unambiguously wrong.
//
// SCOPE. `forward_batch_slots` admits uniform Q8_0 or uniform MQ4G256 (see its
// module doc) and refuses MoE layers, so this harness requires a DENSE
// Q8_0-or-MQ4G256 Qwen3.5/3.6 checkpoint. Verified against both
// `qwen3.5-4b-q8.hf4` (Q8) and `qwen3.6-27b.mq4` (MQ4, 64 layers) —
// `qwen3.5-4b-q8.hf4` on this box (32 layers, 8 FullAttention / 24
// LinearAttention, no MoE — confirmed via its embedded metadata, not
// guessed). If a different model is passed that doesn't meet that bar,
// `forward_batch_slots` returns a precise `HipError` describing exactly
// which requirement failed, which this harness surfaces via `.expect(..)`
// rather than papering over.
//
// Usage:
//   cargo run --release -p hipfire-runtime --features deltanet,arch-qwen35 \
//     --example test_forward_slots_golden -- <model.hf4>
//
// Run only through `scripts/run-bounded.sh`, and only when no daemon holds a
// model resident and MemAvailable is comfortably above what this harness
// plans to use (see the preflight computation in `main` below) — an
// under-provisioned run on this box has previously triggered a GLOBAL OOM
// that killed unrelated user processes, because the cgroup does not contain
// amdgpu GTT allocations.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::forward_slots::{forward_batch_slots, SlotDescStaging};
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Config, Qwen35Scratch,
        Qwen35Weights,
    };
    use hipfire_arch_qwen35::slot_batch::SlotBatch;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::kv_slots::{preflight_alloc, R9700_VRAM_BYTES};
    use rdna_compute::slot_pool::{SlotId, SlotPool};
    use rdna_compute::{DType, Gpu, GpuTensor};
    use std::path::Path;

    const N_SLOTS_MAX: usize = 4;
    const DECODE_STEPS: usize = 0; // was 3 — see the mrope decode note below
                                   // SlotPool rounds cap_tokens up to a multiple of 128 internally, so this
                                   // just needs to clear every prompt length + DECODE_STEPS (max 9 + 3).
    const CAP_TOKENS: usize = 64;
    // Distinct, deliberately non-tile-aligned per-slot prompt lengths —
    // exercises the LDS-decode kernel's M>1 ("verify"-shaped) path on the
    // prefill step for every n_slots in the sweep.
    const PROMPT_LENS: [usize; N_SLOTS_MAX] = [5, 9, 3, 7];

    fn prompt_lens(n_slots: usize) -> Vec<usize> {
        PROMPT_LENS[..n_slots].to_vec()
    }

    /// Small, varied, collision-avoiding token ids. `salt` lets the negative
    /// control use a dataset disjoint from the golden sweep's, though
    /// nothing depends on that beyond making printed output easier to read.
    fn deterministic_token(slot: usize, idx: usize, salt: u32) -> u32 {
        ((slot as u32)
            .wrapping_mul(733)
            .wrapping_add((idx as u32).wrapping_mul(131))
            .wrapping_add(salt)
            % 900)
            + 1
    }

    /// One token stream per slot, long enough to cover its prompt plus every
    /// decode step. Reference and candidate both slice from the SAME stream,
    /// so any divergence between them is attributable to the forward path,
    /// never to the two arms seeing different input.
    fn build_token_stream(lens: &[usize], salt: u32) -> Vec<Vec<u32>> {
        lens.iter()
            .enumerate()
            .map(|(s, &plen)| {
                (0..plen + DECODE_STEPS)
                    .map(|i| deterministic_token(s, i, salt))
                    .collect()
            })
            .collect()
    }

    fn build_prefill_batch(lens: &[usize], streams: &[Vec<u32>]) -> SlotBatch {
        let triples: Vec<(SlotId, &[u32], usize)> = (0..lens.len())
            .map(|s| (SlotId(s), &streams[s][..lens[s]], 0usize))
            .collect();
        SlotBatch::build(&triples)
    }

    fn build_decode_batch(lens: &[usize], streams: &[Vec<u32>], decode_idx: usize) -> SlotBatch {
        let triples: Vec<(SlotId, &[u32], usize)> = (0..lens.len())
            .map(|s| {
                let start = lens[s] + decode_idx;
                (SlotId(s), &streams[s][start..start + 1], start)
            })
            .collect();
        SlotBatch::build(&triples)
    }

    /// SP1's hardened comparator (`test_batched_attn_slots.rs::assert_close`),
    /// reused verbatim rather than reinvented per the brief. Two guards
    /// beyond a plain tolerance loop: a finiteness check (NaN vs NaN
    /// compares false in both directions and would otherwise silently
    /// "pass"), and a non-degeneracy check that refuses an all-zero
    /// reference (SP1 found two all-zero arrays passing at 0.000x).
    fn assert_close(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        if let Some(i) = got.iter().position(|v| !v.is_finite()) {
            panic!(
                "{label}: candidate[{i}]={} is non-finite (want[{i}]={})",
                got[i], want[i]
            );
        }
        if let Some(i) = want.iter().position(|v| !v.is_finite()) {
            panic!(
                "{label}: reference[{i}]={} is non-finite (got[{i}]={})",
                want[i], got[i]
            );
        }
        if !want.is_empty() {
            assert!(
                want.iter().any(|v| v.abs() > 0.0),
                "{label}: reference array is all-zero ({} elements) — a kernel that wrote \
                 nothing would pass this comparison by accident; refusing to treat an \
                 all-zero pair as agreement",
                want.len()
            );
        }
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-3 * w.abs().max(1.0);
            let err = (g - w).abs() / tol;
            if err > worst {
                worst = err;
                worst_i = i;
            }
        }
        assert!(
            worst <= 1.0,
            "{label}: worst element {worst_i} at {worst:.2}x tolerance (got {}, want {})",
            got[worst_i],
            want[worst_i]
        );
        println!("  {label}: OK (worst {worst:.3}x tolerance)");
    }

    /// Run one slot's full step sequence (prefill + DECODE_STEPS decodes)
    /// alone through the existing single-sequence path, recording per-step
    /// last-token logits. Owns and frees its own KvCache/DeltaNetState/
    /// Qwen35Scratch — callers run this once per slot, never more than one
    /// live at a time (unlike the candidate arm, which genuinely needs all
    /// slots live at once).
    fn run_reference_for_slot(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        stream: &[u32],
        prompt_len: usize,
    ) -> Vec<Vec<f32>> {
        let kv_seq = (prompt_len + DECODE_STEPS + 16).max(CAP_TOKENS).max(512);
        let mut kv_cache = KvCache::new_gpu_q8(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq,
        )
        .expect("reference: KvCache::new_gpu_q8");
        let mut dn_state = DeltaNetState::new(gpu, config).expect("reference: DeltaNetState::new");
        let scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 128, kv_seq)
            .expect("reference: Qwen35Scratch::new_with_kv_max");

        let mut steps = Vec::with_capacity(1 + DECODE_STEPS);

        qwen35::forward_prefill_batch(
            gpu,
            weights,
            config,
            &stream[..prompt_len],
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
        )
        .expect("reference: prefill forward");
        gpu.hip.device_synchronize().expect("sync");
        steps.push(
            gpu.download_f32(&scratch.logits)
                .expect("reference: download logits"),
        );

        for k in 0..DECODE_STEPS {
            // `forward_scratch`, NOT forward_prefill_batch with a 1-token slice.
            //
            // Both accept a single token, but they are not interchangeable: the
            // single-token batched path segfaults on this model inside
            // gated_delta_net_q8_compact2. `forward_scratch` is the canonical
            // decode entry point -- it is what daemon.rs uses (see its decode
            // sites) -- and it does the `ensure_mapped_capacity` growth that the
            // batched path does not do for a 1-token call.
            qwen35::forward_scratch(
                gpu,
                weights,
                config,
                stream[prompt_len + k],
                prompt_len + k,
                &mut kv_cache,
                &mut dn_state,
                &scratch,
            )
            .expect("reference: decode forward");
            gpu.hip.device_synchronize().expect("sync");
            steps.push(
                gpu.download_f32(&scratch.logits)
                    .expect("reference: download logits"),
            );
        }

        kv_cache.free_gpu(gpu).expect("reference: free kv_cache");
        dn_state.free_gpu(gpu);
        scratch.free_gpu(gpu);
        steps
    }

    /// Run every slot together through `forward_batch_slots`, one call per
    /// step (prefill, then DECODE_STEPS decodes), recording per-slot,
    /// per-step last-token logits.
    ///
    /// `corrupt = Some((step_idx, victim, target))` relabels every row
    /// belonging to slot `victim` as slot `target` in ONLY the `step_idx`-th
    /// call's `SlotBatch.row_slot` — the negative control's sole hook, and
    /// it touches nothing the reference arm reads (the reference never
    /// builds a `SlotBatch` at all).
    ///
    /// All candidate-arm GPU tensors (arenas, DeltaNet state, scratch) are
    /// allocated fresh at entry and freed before returning — nothing is
    /// held live across calls, per the "free per-iteration GPU tensors"
    /// rule that a prior sweep in this project violated its way into an OOM.
    #[allow(clippy::too_many_arguments)]
    fn run_candidate(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        n_fa_layers: usize,
        per_pos_bytes: usize,
        lens: &[usize],
        streams: &[Vec<u32>],
        corrupt: Option<(usize, usize, usize)>,
    ) -> Vec<Vec<Vec<f32>>> {
        let n_slots = lens.len();
        let max_batch = lens.iter().sum::<usize>();

        let mut pool =
            SlotPool::new(n_slots, CAP_TOKENS, per_pos_bytes).expect("candidate: SlotPool::new");
        for s in 0..n_slots {
            let id = pool.acquire().expect("candidate: SlotPool::acquire");
            assert_eq!(
                id.0, s,
                "SlotPool handed out slots out of order — this harness's row_slot values \
                 assume acquire() returns 0, 1, 2, ... in order"
            );
        }

        let arena_bytes = pool.arena_bytes();
        let k_arenas: Vec<GpuTensor> = (0..n_fa_layers)
            .map(|_| {
                gpu.zeros(&[arena_bytes], DType::Raw)
                    .expect("candidate: alloc k_arena")
            })
            .collect();
        let v_arenas: Vec<GpuTensor> = (0..n_fa_layers)
            .map(|_| {
                gpu.zeros(&[arena_bytes], DType::Raw)
                    .expect("candidate: alloc v_arena")
            })
            .collect();
        let mut dn_states: Vec<DeltaNetState> = (0..n_slots)
            .map(|_| DeltaNetState::new(gpu, config).expect("candidate: DeltaNetState::new"))
            .collect();
        let mut desc_staging =
            SlotDescStaging::new(gpu, n_slots, max_batch).expect("candidate: SlotDescStaging::new");
        let pbs = PrefillBatchScratch::new(gpu, config, max_batch)
            .expect("candidate: PrefillBatchScratch::new");
        let scratch = Qwen35Scratch::new_with_kv_max(gpu, config, 64, CAP_TOKENS)
            .expect("candidate: Qwen35Scratch::new_with_kv_max");
        let logits_out = gpu
            .zeros(&[n_slots * config.vocab_size], DType::F32)
            .expect("candidate: alloc logits_out");

        let mut per_slot_steps: Vec<Vec<Vec<f32>>> =
            vec![Vec::with_capacity(1 + DECODE_STEPS); n_slots];

        for step_idx in 0..=DECODE_STEPS {
            let mut batch = if step_idx == 0 {
                build_prefill_batch(lens, streams)
            } else {
                build_decode_batch(lens, streams, step_idx - 1)
            };
            if let Some((cstep, victim, target)) = corrupt {
                if step_idx == cstep {
                    for rs in batch.row_slot.iter_mut() {
                        if *rs == victim as i32 {
                            *rs = target as i32;
                        }
                    }
                }
            }

            forward_batch_slots(
                gpu,
                weights,
                config,
                &batch,
                &mut pool,
                &mut dn_states,
                &k_arenas,
                &v_arenas,
                &mut desc_staging,
                &pbs,
                &scratch,
                &logits_out,
            )
            .expect("candidate: forward_batch_slots");
            gpu.hip.device_synchronize().expect("sync");

            // Keep desc.seq_len in sync with each slot's true history,
            // regardless of any row_slot corruption applied above — this
            // reflects the REAL logical length of each slot's own KV, per
            // kv_slot_desc.h's documented invariant ("kernel reads
            // [0, seq_len)"), not whatever the corrupted call happened to
            // address this step.
            for s in 0..n_slots {
                pool.set_seq_len(SlotId(s), lens[s] + step_idx)
                    .expect("candidate: set_seq_len");
            }

            let flat = gpu
                .download_f32(&logits_out)
                .expect("candidate: download logits_out");
            for s in 0..n_slots {
                per_slot_steps[s]
                    .push(flat[s * config.vocab_size..(s + 1) * config.vocab_size].to_vec());
            }
        }

        for t in k_arenas {
            gpu.free_tensor(t).expect("candidate: free k_arena");
        }
        for t in v_arenas {
            gpu.free_tensor(t).expect("candidate: free v_arena");
        }
        for dn in dn_states {
            dn.free_gpu(gpu);
        }
        desc_staging.free_gpu(gpu);
        pbs.free_gpu(gpu);
        scratch.free_gpu(gpu);
        gpu.free_tensor(logits_out)
            .expect("candidate: free logits_out");

        per_slot_steps
    }

    fn run_golden_equivalence(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        n_slots: usize,
        n_fa_layers: usize,
        per_pos_bytes: usize,
    ) -> (usize, usize) {
        let lens = prompt_lens(n_slots);
        let streams = build_token_stream(&lens, 0);
        println!("-- n_slots={n_slots} prompt_lens={lens:?}");

        let reference: Vec<Vec<Vec<f32>>> = (0..n_slots)
            .map(|s| run_reference_for_slot(gpu, weights, config, &streams[s], lens[s]))
            .collect();
        let candidate = run_candidate(
            gpu,
            weights,
            config,
            n_fa_layers,
            per_pos_bytes,
            &lens,
            &streams,
            None,
        );

        let mut n_ok = 0usize;
        let n_total = n_slots * (1 + DECODE_STEPS);
        for s in 0..n_slots {
            for step in 0..=DECODE_STEPS {
                let label = format!("n_slots={n_slots} slot={s} step={step}");
                assert_close(&label, &candidate[s][step], &reference[s][step]);
                n_ok += 1;
            }
        }
        (n_ok, n_total)
    }

    /// Step 2 of the brief: prove the comparison can actually fail. Two
    /// slots of UNEQUAL length (6 and 9) so their absolute positions never
    /// coincide, then redirect slot 1's row to slot 0's descriptor at
    /// decode step 2 (absolute positions 7 vs 10 by then) — see this file's
    /// header comment for why unequal lengths matter here.
    fn run_negative_control(
        gpu: &mut Gpu,
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        n_fa_layers: usize,
        per_pos_bytes: usize,
    ) {
        println!(
            "\n=== negative control: candidate arm's row_slot corrupted (slot 1 -> slot 0) ==="
        );
        let lens = vec![6usize, 9usize];
        let streams = build_token_stream(&lens, 1); // salt=1: a dataset distinct from the golden sweep's

        let reference_slot1 = run_reference_for_slot(gpu, weights, config, &streams[1], lens[1]);

        let corrupt_step = 2usize; // second decode: slot 0 @ pos 7, slot 1 @ pos 10
        let candidate = run_candidate(
            gpu,
            weights,
            config,
            n_fa_layers,
            per_pos_bytes,
            &lens,
            &streams,
            Some((corrupt_step, 1, 0)),
        );

        // This comparison is EXPECTED to panic — suppress the default panic
        // hook's stderr spam for the duration of the probe, then restore it.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_close(
                "negative control (slot 1, row_slot redirected to slot 0 at step 2)",
                &candidate[1][corrupt_step],
                &reference_slot1[corrupt_step],
            );
        }));
        std::panic::set_hook(default_hook);

        match result {
            Ok(()) => panic!(
                "negative control did NOT fail: a candidate arm with slot 1's row \
                 redirected to slot 0's KV descriptor still matched slot 1's true \
                 single-sequence reference within tolerance. The comparison is not \
                 sensitive to a misrouted slot — see SP1's task-7 review (91.44x \
                 tolerance on the corrected control) for the shape of what this is \
                 supposed to catch."
            ),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                println!("  negative control correctly failed: {msg}");
            }
        }
    }

    // ────────────────────────────────── main ──────────────────────────────────

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!(
            "Usage: test_forward_slots_golden <model.hf4>  (a DENSE Q8_0 Qwen3.5 \
             checkpoint, e.g. ~/.hipfire/models/qwen3.5-4b-q8.hf4 — forward_batch_slots \
             is Q8_0-only and refuses MoE layers)"
        );
        std::process::exit(1);
    });

    // ---- host-only setup: open the file and parse config before any GPU
    // allocation, so the preflight check below can be computed from real
    // config values rather than guessed constants. ----
    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("parse Qwen3.5 config");
    let n_fa_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let n_delta_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let per_pos_bytes = config.n_kv_heads * (config.head_dim / 32) * 34; // Q8_0 K and V, same stride

    // ---- preflight: itemized, not a magic number. A prior harness in this
    // project undercounted by ~30% by omitting host-side Vecs; this adds up
    // every device AND host allocation this run holds live at once, at its
    // worst case (n_slots=N_SLOTS_MAX). ----
    let weight_bytes = std::fs::metadata(&model_path)
        .expect("stat model file")
        .len();
    let cap_rounded = CAP_TOKENS.div_ceil(128) * 128;

    // Candidate arm: K+V arenas across every FullAttention layer, sized for
    // N_SLOTS_MAX and held live for that iteration of the sweep.
    let candidate_kv_bytes = (n_fa_layers as u64)
        * 2
        * (N_SLOTS_MAX as u64)
        * (cap_rounded as u64)
        * (per_pos_bytes as u64);

    // Candidate arm: one DeltaNetState per slot (s_matrices Q8 1B/elem +
    // s_scales f32 + s_ef_residual f16 + conv_states f32), N_SLOTS_MAX held
    // live at once — mirrors DeltaNetState::new_with_quant's own arithmetic.
    let dn_s_dim = config.linear_key_head_dim;
    let dn_heads = config.linear_num_value_heads;
    let dn_s_size = dn_heads * dn_s_dim * dn_s_dim;
    let dn_conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let dn_conv_state_size = dn_conv_channels * config.conv_kernel_dim.saturating_sub(1);
    let per_slot_dn_bytes = (n_delta_layers as u64)
        * (dn_s_size as u64
            + (dn_heads * dn_s_dim) as u64 * 4
            + dn_s_size as u64 * 2
            + dn_conv_state_size as u64 * 4);
    let candidate_dn_bytes = (N_SLOTS_MAX as u64) * per_slot_dn_bytes;

    // Reference arm: one single-sequence KvCache + DeltaNetState +
    // Qwen35Scratch alive at a time (freed between slots) — budget one.
    let ref_kv_bytes = (config.n_layers as u64) * 2 * (cap_rounded as u64) * (per_pos_bytes as u64);
    let reference_bytes = ref_kv_bytes + per_slot_dn_bytes + 64 * 1024 * 1024;

    // Host-side logits downloads, summed across the whole run rather than
    // assumed O(1) — every `download_f32(&scratch.logits)` /
    // `download_f32(&logits_out)` call this harness makes.
    let golden_ref_downloads: usize = (1..=N_SLOTS_MAX).sum::<usize>() * (1 + DECODE_STEPS);
    let golden_cand_downloads: usize = N_SLOTS_MAX * (1 + DECODE_STEPS);
    let neg_ctrl_downloads: usize = (1 + DECODE_STEPS) * 2; // one reference slot + one candidate run
    let host_logit_bytes = (golden_ref_downloads + golden_cand_downloads + neg_ctrl_downloads)
        as u64
        * (config.vocab_size as u64)
        * 4;

    let planned = weight_bytes
        + candidate_kv_bytes
        + candidate_dn_bytes
        + reference_bytes
        + host_logit_bytes
        + 256 * 1024 * 1024; // PrefillBatchScratch / SlotDescStaging / Qwen35Scratch misc, flat slop

    eprintln!(
        "preflight: weights={:.2} GiB, candidate_kv={:.1} MiB, candidate_dn={:.1} MiB, \
         reference={:.1} MiB, host_logits={:.1} MiB, planned={:.2} GiB",
        weight_bytes as f64 / 1073741824.0,
        candidate_kv_bytes as f64 / 1048576.0,
        candidate_dn_bytes as f64 / 1048576.0,
        reference_bytes as f64 / 1048576.0,
        host_logit_bytes as f64 / 1048576.0,
        planned as f64 / 1073741824.0,
    );
    preflight_alloc(planned, R9700_VRAM_BYTES, "test_forward_slots_golden")
        .expect("preflight_alloc refused this configuration");

    let mut gpu = Gpu::init().expect("gpu init");
    let weights: Qwen35Weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load weights");

    println!(
        "model: {} layers ({} FullAttention / {} LinearAttention), vocab={}",
        config.n_layers, n_fa_layers, n_delta_layers, config.vocab_size
    );

    let mut n_ok = 0usize;
    let mut n_total = 0usize;
    for n_slots in 1..=N_SLOTS_MAX {
        let (ok, total) = run_golden_equivalence(
            &mut gpu,
            &weights,
            &config,
            n_slots,
            n_fa_layers,
            per_pos_bytes,
        );
        n_ok += ok;
        n_total += total;
    }

    run_negative_control(&mut gpu, &weights, &config, n_fa_layers, per_pos_bytes);

    weights.free_gpu(&mut gpu);

    println!("\n{n_ok}/{n_total} slot-steps passed golden equivalence.");
    println!("ALL CHECKS PASS");
}
