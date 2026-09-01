// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission probe for the asymmetric gfx1201 topology:
//! target TP3 on devices 0..2, dedicated DSpark drafter on device 3.
//!
//! This is deliberately not a serving path. It keeps the certified TP3 route
//! unchanged and measures the three serial costs that determine whether the
//! topology deserves product plumbing: target-hidden peer transfer, DSpark
//! draft, and TP3 batched verification.

use hipfire_arch_deepseek4::forward::{
    dspark_forward, final_norm_and_argmax_all_batched, forward_ep_prefill_batch_chunked,
    PrefillBatchScratch,
};
use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4State};
use hipfire_hardware::Gpus;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};
use std::time::Instant;

const TARGET_RANKS: usize = 3;
const DRAFTER_DEVICE: i32 = 3;

struct Args {
    model: PathBuf,
    sidecar: PathBuf,
    warmups: usize,
    samples: usize,
    verify_batch: usize,
    position: u32,
    tau: f64,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut sidecar = None;
    let mut warmups = 2usize;
    let mut samples = 5usize;
    let mut verify_batch = 3usize;
    let mut position = 2048u32;
    let mut tau = 3.02f64;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        let value = |i: usize, flag: &str| {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(value(i, "--model")?));
                i += 2;
            }
            "--sidecar" => {
                sidecar = Some(PathBuf::from(value(i, "--sidecar")?));
                i += 2;
            }
            "--warmups" => {
                warmups = value(i, "--warmups")?
                    .parse()
                    .map_err(|_| "invalid --warmups")?;
                i += 2;
            }
            "--samples" => {
                samples = value(i, "--samples")?
                    .parse()
                    .map_err(|_| "invalid --samples")?;
                i += 2;
            }
            "--verify-batch" => {
                verify_batch = value(i, "--verify-batch")?
                    .parse()
                    .map_err(|_| "invalid --verify-batch")?;
                i += 2;
            }
            "--position" => {
                position = value(i, "--position")?
                    .parse()
                    .map_err(|_| "invalid --position")?;
                i += 2;
            }
            "--tau" => {
                tau = value(i, "--tau")?.parse().map_err(|_| "invalid --tau")?;
                i += 2;
            }
            flag => return Err(format!("unknown argument {flag}")),
        }
    }
    if samples == 0 || verify_batch == 0 {
        return Err("--samples and --verify-batch must be nonzero".into());
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        sidecar: sidecar.ok_or("--sidecar is required")?,
        warmups,
        samples,
        verify_batch,
        position,
        tau,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn sync(gpu: &mut Gpu) -> Result<(), String> {
    gpu.bind_thread().map_err(|e| format!("bind: {e:?}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("device sync: {e:?}"))
}

fn sync_target(gpus: &mut Gpus) -> Result<(), String> {
    for (rank, gpu) in gpus.devices.iter_mut().enumerate() {
        gpu.bind_thread()
            .map_err(|e| format!("bind target rank {rank}: {e:?}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync target rank {rank}: {e:?}"))?;
    }
    Ok(())
}

fn peer_clone_tensor(
    src: &GpuTensor,
    src_device: i32,
    dst_gpu: &mut Gpu,
) -> Result<GpuTensor, String> {
    dst_gpu
        .bind_thread()
        .map_err(|e| format!("bind drafter clone: {e:?}"))?;
    let mut dst = dst_gpu
        .alloc_tensor(&[src.buf.size()], DType::Raw)
        .map_err(|e| format!("allocate peer clone: {e:?}"))?;
    dst_gpu
        .hip
        .memcpy_peer(
            &dst.buf,
            dst_gpu.device_id,
            &src.buf,
            src_device,
            src.buf.size(),
        )
        .map_err(|e| format!("peer clone: {e:?}"))?;
    dst.shape = src.shape.clone();
    dst.dtype = src.dtype;
    Ok(dst)
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let source = HfqFile::open(Path::new(&args.model))
        .map_err(|e| format!("open trunk {}: {e:?}", args.model.display()))?;
    let mut cfg = DeepseekV4::config_from_hfq(&source)?;
    if !cfg.mq2r || cfg.mq2rxt {
        return Err("probe requires the frozen DeepSeek4 MQ2R trunk".into());
    }
    if cfg.num_experts_per_tok != 6 {
        return Err(format!(
            "probe requires shipping top-k 6, got {}",
            cfg.num_experts_per_tok
        ));
    }
    // Never auto-load/replicate the sidecar into the target ranks.
    cfg.load_dspark = false;
    drop(source);

    eprintln!(
        "topology: target=TP3 devices 0,1,2 drafter=device 3 hidden={} layers={} verify_B={} position={}",
        cfg.hidden_size, cfg.num_hidden_layers, args.verify_batch, args.position
    );
    let device_opts = hipfire_runtime::config::get().device_resolve_opts();
    let mut gpus = Gpus::init_tp(&device_opts, TARGET_RANKS, cfg.num_hidden_layers)
        .map_err(|e| format!("initialize TP3 target: {e:?}"))?;
    if gpus.devices.len() != TARGET_RANKS
        || gpus.devices.iter().any(|gpu| !gpu.arch_caps.is_gfx1201())
    {
        return Err("target requires exactly three gfx1201 devices".into());
    }

    let shard = ShardConfig::new_uneven_experts(
        TARGET_RANKS,
        true,
        cfg.n_routed_experts,
        ExpertAssign::Stride,
    )
    .map_err(|e| format!("TP3 shard: {e}"))?;
    let mut weights_per_rank = Vec::with_capacity(TARGET_RANKS);
    for rank in 0..TARGET_RANKS {
        gpus.devices[rank]
            .bind_thread()
            .map_err(|e| format!("bind target rank {rank}: {e:?}"))?;
        let mut rank_source = HfqFile::open(Path::new(&args.model))
            .map_err(|e| format!("open trunk rank {rank}: {e:?}"))?;
        let started = Instant::now();
        let weights = DeepseekV4::load_weights_sharded(
            &mut rank_source,
            &cfg,
            &mut gpus.devices[rank],
            &shard,
            rank,
        )?;
        eprintln!(
            "target rank {rank} loaded on device {} in {:.1}s",
            gpus.devices[rank].device_id,
            started.elapsed().as_secs_f64()
        );
        weights_per_rank.push(weights);
    }
    if !gpus
        .enable_peer_all()
        .map_err(|e| format!("enable target peers: {e:?}"))?
    {
        return Err("TP3 target requires complete peer access".into());
    }
    hipfire_runtime::ep::ensure_rank_streams(&mut gpus)
        .map_err(|e| format!("target rank streams: {e:?}"))?;

    let mut state_per_rank = Vec::with_capacity(TARGET_RANKS);
    let mut pbs_per_rank = Vec::with_capacity(TARGET_RANKS);
    let mut partials = Vec::with_capacity(TARGET_RANKS);
    for rank in 0..TARGET_RANKS {
        gpus.devices[rank]
            .bind_thread()
            .map_err(|e| format!("bind target scratch rank {rank}: {e:?}"))?;
        state_per_rank.push(DeepseekV4State::new(&cfg)?);
        pbs_per_rank.push(PrefillBatchScratch::new(
            &mut gpus.devices[rank],
            &cfg,
            args.verify_batch.max(8),
        )?);
        partials.push(
            gpus.devices[rank]
                .zeros(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("allocate target partial rank {rank}: {e:?}"))?,
        );
    }
    gpus.prepare_tp_graph_signals(cfg.num_hidden_layers * 2)
        .map_err(|e| format!("prepare TP3 graph signals: {e:?}"))?;

    let mut drafter_gpu = Gpu::init_with_device(DRAFTER_DEVICE)
        .map_err(|e| format!("initialize drafter device 3: {e:?}"))?;
    if !drafter_gpu.arch_caps.is_gfx1201() {
        return Err(format!(
            "drafter device 3 is {}, expected gfx1201",
            drafter_gpu.arch
        ));
    }
    for rank in 0..TARGET_RANKS {
        let target_id = gpus.devices[rank].device_id;
        gpus.devices[rank]
            .bind_thread()
            .map_err(|e| format!("bind target peer rank {rank}: {e:?}"))?;
        gpus.devices[rank]
            .hip
            .enable_peer_access(drafter_gpu.device_id)
            .map_err(|e| format!("enable target rank {rank} -> drafter: {e:?}"))?;
        drafter_gpu
            .bind_thread()
            .map_err(|e| format!("bind drafter peer rank {rank}: {e:?}"))?;
        drafter_gpu
            .hip
            .enable_peer_access(target_id)
            .map_err(|e| format!("enable drafter -> target rank {rank}: {e:?}"))?;
    }

    let sidecar_source = HfqFile::open(Path::new(&args.sidecar))
        .map_err(|e| format!("open sidecar {}: {e:?}", args.sidecar.display()))?;
    let sidecar_started = Instant::now();
    let dspark = DeepseekV4::load_dspark(&sidecar_source, &mut drafter_gpu, &cfg)?
        .ok_or("sidecar has no DSpark payload")?;
    eprintln!(
        "drafter loaded on device 3 in {:.1}s: stages={} block={} target_layers={:?}",
        sidecar_started.elapsed().as_secs_f64(),
        dspark.stages.len(),
        dspark.cfg.block_size,
        dspark.cfg.target_layer_ids
    );
    if args.verify_batch > dspark.cfg.block_size {
        return Err(format!(
            "verify batch {} exceeds sidecar block {}",
            args.verify_batch, dspark.cfg.block_size
        ));
    }

    let rank0_device = gpus.devices[0].device_id;
    let token_embd = peer_clone_tensor(
        weights_per_rank[0]
            .token_embd
            .as_ref()
            .ok_or("rank0 token embedding missing")?,
        rank0_device,
        &mut drafter_gpu,
    )?;
    let head = peer_clone_tensor(
        weights_per_rank[0]
            .head
            .as_ref()
            .ok_or("rank0 head missing")?,
        rank0_device,
        &mut drafter_gpu,
    )?;
    // dspark_forward intentionally uses the stage final norm; this argument is
    // retained for API compatibility and is not dereferenced.
    let output_norm_dummy = drafter_gpu
        .zeros(&[1], DType::F32)
        .map_err(|e| format!("allocate output norm dummy: {e:?}"))?;

    let main_hidden_elems = dspark.cfg.target_layer_ids.len() * cfg.hidden_size;
    gpus.devices[0]
        .bind_thread()
        .map_err(|e| format!("bind rank0 hidden source: {e:?}"))?;
    let main_hidden_src = gpus.devices[0]
        .zeros(&[main_hidden_elems], DType::F32)
        .map_err(|e| format!("allocate target-hidden source: {e:?}"))?;
    let main_hidden = drafter_gpu
        .zeros(&[main_hidden_elems], DType::F32)
        .map_err(|e| format!("allocate drafter main-hidden: {e:?}"))?;
    let mut draft_state = DeepseekV4State::new(&cfg)?;

    let mut peer_us = Vec::with_capacity(args.samples);
    let mut draft_ms = Vec::with_capacity(args.samples);
    let mut last_draft = Vec::new();
    for iteration in 0..args.warmups + args.samples {
        drafter_gpu
            .bind_thread()
            .map_err(|e| format!("bind drafter sample: {e:?}"))?;
        let copy_started = Instant::now();
        drafter_gpu
            .hip
            .memcpy_peer(
                &main_hidden.buf,
                drafter_gpu.device_id,
                &main_hidden_src.buf,
                rank0_device,
                main_hidden_src.buf.size(),
            )
            .map_err(|e| format!("target-hidden peer copy: {e:?}"))?;
        let copy_elapsed = copy_started.elapsed().as_secs_f64() * 1e6;

        sync(&mut drafter_gpu)?;
        let draft_started = Instant::now();
        let result = dspark_forward(
            &cfg,
            &dspark,
            &mut draft_state,
            &mut drafter_gpu,
            &main_hidden,
            &token_embd,
            &head,
            &output_norm_dummy,
            1,
            args.position + iteration as u32 * dspark.cfg.block_size as u32,
        )?;
        sync(&mut drafter_gpu)?;
        let draft_elapsed = draft_started.elapsed().as_secs_f64() * 1e3;
        last_draft = result.tokens;
        if iteration >= args.warmups {
            peer_us.push(copy_elapsed);
            draft_ms.push(draft_elapsed);
        }
    }

    let mut verify_tokens = Vec::with_capacity(args.verify_batch);
    verify_tokens.push(1u32);
    verify_tokens.extend(last_draft.iter().copied().take(args.verify_batch - 1));
    if verify_tokens.len() != args.verify_batch {
        return Err("drafter returned too few tokens for verify batch".into());
    }
    let mut verify_trunk_ms = Vec::with_capacity(args.samples);
    let mut verify_heads_ms = Vec::with_capacity(args.samples);
    let mut verify_ms = Vec::with_capacity(args.samples);
    for iteration in 0..args.warmups + args.samples {
        let start_pos = args.position
            + (args.warmups + args.samples) as u32 * dspark.cfg.block_size as u32
            + iteration as u32 * args.verify_batch as u32;
        sync_target(&mut gpus)?;
        let verify_started = Instant::now();
        forward_ep_prefill_batch_chunked(
            &mut gpus,
            &weights_per_rank,
            &cfg,
            &mut state_per_rank,
            &mut pbs_per_rank,
            &verify_tokens,
            start_pos,
        )?;
        sync_target(&mut gpus)?;
        let trunk_elapsed = verify_started.elapsed().as_secs_f64() * 1e3;
        // The TP3 helper computes the last-row head. Add the exact all-row
        // greedy head used by DSpark verification. This deliberately counts
        // the last head twice, making the admission projection conservative.
        gpus.devices[0]
            .bind_thread()
            .map_err(|e| format!("bind rank0 verify head: {e:?}"))?;
        let heads_started = Instant::now();
        let _ = final_norm_and_argmax_all_batched(
            &cfg,
            &weights_per_rank[0],
            &mut state_per_rank[0],
            &pbs_per_rank[0],
            &mut gpus.devices[0],
            args.verify_batch,
        )?;
        sync_target(&mut gpus)?;
        let heads_elapsed = heads_started.elapsed().as_secs_f64() * 1e3;
        let verify_elapsed = trunk_elapsed + heads_elapsed;
        if iteration >= args.warmups {
            verify_trunk_ms.push(trunk_elapsed);
            verify_heads_ms.push(heads_elapsed);
            verify_ms.push(verify_elapsed);
        }
    }

    // Control: verify the same B tokens as independent calls through the
    // shipping TP3 retained hipGraph. This gives a hard fallback ceiling and
    // proves whether the small-B batched path, rather than TP3 itself, is the
    // blocker.
    let mut sequential_verify_ms = Vec::with_capacity(args.samples);
    for iteration in 0..args.warmups + args.samples {
        let start_pos = args.position
            + 2 * (args.warmups + args.samples) as u32 * dspark.cfg.block_size as u32
            + iteration as u32 * args.verify_batch as u32;
        sync_target(&mut gpus)?;
        let started = Instant::now();
        for (offset, &token) in verify_tokens.iter().enumerate() {
            hipfire_arch_deepseek4::forward::forward_ep(
                &mut gpus,
                &weights_per_rank,
                &cfg,
                &mut state_per_rank,
                &partials,
                token,
                start_pos + offset as u32,
            )?;
        }
        sync_target(&mut gpus)?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        if iteration >= args.warmups {
            sequential_verify_ms.push(elapsed);
        }
    }

    let peer_us_med = median(&mut peer_us);
    let draft_ms_med = median(&mut draft_ms);
    let verify_trunk_ms_med = median(&mut verify_trunk_ms);
    let verify_heads_ms_med = median(&mut verify_heads_ms);
    let verify_ms_med = median(&mut verify_ms);
    let sequential_verify_ms_med = median(&mut sequential_verify_ms);
    let window_ms = peer_us_med / 1e3 + draft_ms_med + verify_ms_med;
    let projected_tps = args.tau * 1e3 / window_ms;
    let sequential_window_ms = peer_us_med / 1e3 + draft_ms_med + sequential_verify_ms_med;
    let sequential_projected_tps = args.tau * 1e3 / sequential_window_ms;
    println!("=== TP3 target + device3 DSpark admission probe ===");
    println!("samples={} warmups={}", args.samples, args.warmups);
    println!(
        "hidden_peer_bytes={} peer_us_median={peer_us_med:.3}",
        main_hidden_src.buf.size()
    );
    println!("draft_ms_median={draft_ms_med:.3}");
    println!("verify_trunk_plus_last_head_ms_median={verify_trunk_ms_med:.3}");
    println!("verify_all_heads_ms_median={verify_heads_ms_med:.3}");
    println!(
        "verify_ms_median={verify_ms_med:.3} verify_batch={} (conservative: duplicate last-row head)",
        args.verify_batch
    );
    println!(
        "sequential_retained_verify_ms_median={sequential_verify_ms_med:.3} verify_batch={}",
        args.verify_batch
    );
    println!(
        "serial_window_ms={window_ms:.3} historical_tau={:.3} projected_tok_s={projected_tps:.3}",
        args.tau
    );
    println!(
        "sequential_serial_window_ms={sequential_window_ms:.3} historical_tau={:.3} projected_tok_s={sequential_projected_tps:.3}",
        args.tau
    );
    println!("draft_tokens={last_draft:?}");
    println!(
        "ADMISSION_ONLY: this is a component timing screen, not a serving or correctness claim"
    );
    Ok(())
}
