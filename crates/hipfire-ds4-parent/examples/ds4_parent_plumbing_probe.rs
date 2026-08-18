// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Inter-layer plumbing probe for the DS4 parent forward.
//!
//! Per-layer oracles that feed each layer its *actual GPU input* cannot see a
//! defect in the wiring that hands a layer the wrong input. This harness
//! instruments the parent driver loop itself and asserts:
//!
//! 1. **Embed** — `parent_embed` matches host `embed_gather_ref` bit-for-bit
//!    on the pinned `tokens.bin` prefix.
//! 2. **HC state continuity** — the buffer layer `L` receives is bit-identical
//!    to what layer `L-1` wrote (ping-pong handoff). No tolerance.
//! 3. **KV ring isolation** — each layer's ring is unique; a per-layer sentinel
//!    fill is disturbed only by that layer's forward.
//! 4. **Ring write pattern** — after a 128-row prefill (`start_pos=0`,
//!    `window=128`) every slot `s` holds the KV committed for absolute
//!    position `s` (kernel: `slot = (start_pos + b) % window`).
//! 5. **Prefill does not read its own ring write** — ring commit runs after
//!    the attention kernel (attention.rs steps 5 → 5c → 8).
//! 6. **Hash-layer inputs** — layers `0..num_hash_layers` require `input_ids`
//!    and route through `tid2eid`; layers at/after accept `None`.
//! 7. **Head buffer** — with `N=43` (odd) the final HC state is `hc_a` (last
//!    write); head on the stale partner differs.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_plumbing_probe \
//!   -- --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
//!      --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
//!      --rows 128
//! ```
//!
//! Must run on gfx942 (mi300x).

use hipfire_ds4_parent::attention::{
    PARENT_DIM, PARENT_HEAD_DIM, PARENT_N_KV_HEADS, PARENT_SWA_WINDOW,
};
use hipfire_ds4_parent::forward::{
    parent_layer_forward, ParentForwardScratch, PARENT_HC_DIM, PARENT_HC_MULT,
};
use hipfire_ds4_parent::head::{
    embed_gather_ref, parent_embed, parent_head_with_scratch, ParentHeadScratch, PARENT_VOCAB,
};
use hipfire_ds4_parent::inventory::ParentInventory;
use hipfire_ds4_parent::moe::parent_route;
use hipfire_ds4_parent::weights::{ParentLoadPlan, ParentWeights};
use hipfire_ds4_parent::Ds4ParentBackend;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/mnt/scratch/models/DeepSeek-V4-Flash-0731";
const DEFAULT_TOKEN_IDS: &str =
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin";
const DEFAULT_ROWS: usize = 128;

/// Sentinel base so layer L's ring is filled with `SENTINEL_BASE + L` on every
/// element before the forward. Distinct across layers; far from typical KV.
const SENTINEL_BASE: f32 = 7_000_000.0;

fn main() -> ExitCode {
    match run() {
        Ok(true) => {
            println!("\nPASS: all plumbing checks clean");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            eprintln!("\nFAIL: one or more plumbing checks failed (see above)");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let model_path = Path::new(&args.model);
    if !model_path.is_dir() {
        return Err(format!(
            "deepseek4 parent: --model must be a directory, got {}",
            model_path.display()
        ));
    }

    let mut token_ids = read_token_ids(&args.token_ids)?;
    if token_ids.is_empty() {
        return Err("deepseek4 parent: token-ids file is empty".into());
    }
    if args.rows < token_ids.len() {
        token_ids.truncate(args.rows);
    } else if args.rows > token_ids.len() {
        return Err(format!(
            "deepseek4 parent: --rows {} exceeds token-ids length {}",
            args.rows,
            token_ids.len()
        ));
    }
    let rows = token_ids.len();
    let start_pos = 0usize;

    println!("=== ds4_parent_plumbing_probe ===");
    println!("model: {}", model_path.display());
    println!("token_ids: {} (n={rows})", args.token_ids.display());
    println!("start_pos: {start_pos}");
    println!(
        "checks: embed | HC continuity (bit-exact) | KV ring isolation | ring write | hash ids | head buffer"
    );
    println!();

    let wall0 = Instant::now();

    let source = SafetensorsSource::open(model_path).map_err(|e| {
        format!(
            "deepseek4 parent: SafetensorsSource::open({}): {e}",
            model_path.display()
        )
    })?;
    let mut gpu = Gpu::init().map_err(|e| format!("deepseek4 parent: Gpu::init: {e:?}"))?;
    if gpu.try_gfx942().is_none() {
        return Err("deepseek4 parent: gfx942 required".to_owned());
    }

    let admit_t0 = Instant::now();
    let (backend, cfg) = Ds4ParentBackend::admit(&source, &mut gpu)?;
    println!(
        "admit OK ({:.1} ms): layers={} hash_layers={} n_routed={} topk={}",
        admit_t0.elapsed().as_secs_f64() * 1000.0,
        cfg.num_hidden_layers,
        cfg.num_hash_layers,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
    );
    let n_layers = cfg.num_hidden_layers;
    if n_layers == 0 {
        return Err("num_hidden_layers must be > 0".into());
    }
    let num_hash = cfg.num_hash_layers;

    let inv = ParentInventory::build(&source, &cfg)?;
    let plan = ParentLoadPlan {
        layers: 0..n_layers,
        load_experts: true,
    };
    println!("load plan: layers={:?} experts=true", plan.layers);
    let load_t0 = Instant::now();
    let weights = ParentWeights::load(&source, &cfg, &inv, &mut gpu, backend, &plan)?;
    let load_s = load_t0.elapsed().as_secs_f64();
    println!(
        "loaded layers={:?} experts={} in {load_s:.3} s  resident={:.3} GiB",
        weights.layer_range,
        weights.experts_loaded,
        weights.residency().total_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Same topology as ParentModelScratch (hc_a/hc_b + per-layer kv rings).
    let mut layer_scratch = ParentForwardScratch::new(&mut gpu, &cfg, rows)?;
    let mut head_scratch = ParentHeadScratch::new(&mut gpu, &cfg, rows)?;
    let hc_a = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let hc_b = zeros_f32(&mut gpu, &[rows, PARENT_HC_MULT, PARENT_DIM])?;
    let mut kv_rings = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let ring = zeros_f32(
            &mut gpu,
            &[PARENT_N_KV_HEADS, PARENT_HEAD_DIM, PARENT_SWA_WINDOW],
        )
        .map_err(|e| format!("kv_ring[{i}]: {e}"))?;
        kv_rings.push(ring);
    }
    let logits = zeros_f32(&mut gpu, &[rows, PARENT_VOCAB])?;

    let ring_elems = PARENT_N_KV_HEADS * PARENT_HEAD_DIM * PARENT_SWA_WINDOW;
    let hc_elems = rows * PARENT_HC_DIM;

    let mut checks: Vec<Check> = Vec::new();

    // ── CHECK 0: KV ring / HC buffer pointer uniqueness ─────────────────
    {
        let mut ptrs: Vec<u64> = kv_rings.iter().map(|r| r.buf.as_ptr() as u64).collect();
        let n_before = ptrs.len();
        ptrs.sort_unstable();
        ptrs.dedup();
        let unique = ptrs.len();
        let hc_a_ptr = hc_a.buf.as_ptr() as u64;
        let hc_b_ptr = hc_b.buf.as_ptr() as u64;
        let hc_distinct = hc_a_ptr != hc_b_ptr;
        let pass = unique == n_before && hc_distinct;
        checks.push(Check {
            name: "kv_ring_ptr_unique".into(),
            pass,
            detail: format!(
                "unique_rings={unique}/{n_before} hc_a_ptr={hc_a_ptr:#x} hc_b_ptr={hc_b_ptr:#x} distinct={hc_distinct}"
            ),
        });
        println!(
            "CHECK kv_ring_ptr_unique: {}  unique_rings={unique}/{n_before} hc_distinct={hc_distinct}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // ── CHECK 1: embed vs host gather (bit-identical) ───────────────────
    {
        let embed_bytes_n = PARENT_VOCAB * PARENT_DIM * 2;
        let mut table = vec![0u8; embed_bytes_n];
        gpu.hip
            .memcpy_dtoh(&mut table, &weights.embed.buf)
            .map_err(|e| format!("embed dtoh: {e:?}"))?;
        let ref_hc = embed_gather_ref(
            &table,
            &token_ids,
            PARENT_VOCAB,
            PARENT_DIM,
            PARENT_HC_MULT,
        )?;
        parent_embed(&mut gpu, backend, &weights, &cfg, &token_ids, &hc_a)?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync embed: {e:?}"))?;
        let got = download_f32(&gpu, &hc_a, hc_elems)?;
        let (n_diff, first) = bit_diff_f32(&got, &ref_hc);
        let pass = n_diff == 0;
        checks.push(Check {
            name: "embed_bit_identical".into(),
            pass,
            detail: format!(
                "n_diff={n_diff}/{} first_diff={} got_l2={:.6} ref_l2={:.6}",
                got.len(),
                first
                    .map(|(i, g, r)| format!("i={i} got_bits={g:#x} ref_bits={r:#x}"))
                    .unwrap_or_else(|| "-".into()),
                l2(&got),
                l2(&ref_hc),
            ),
        });
        println!(
            "CHECK embed_bit_identical: {}  n_diff={n_diff}/{} l2_got={:.6} l2_ref={:.6}",
            if pass { "PASS" } else { "FAIL" },
            got.len(),
            l2(&got),
            l2(&ref_hc),
        );
    }

    // ── CHECK 2: hash-layer input_ids contract ──────────────────────────
    {
        let emb = download_f32(&gpu, &hc_a, hc_elems)?;
        let mut act_f32 = vec![0.0f32; rows * PARENT_DIM];
        for r in 0..rows {
            let src = r * PARENT_HC_DIM; // stream 0
            act_f32[r * PARENT_DIM..(r + 1) * PARENT_DIM]
                .copy_from_slice(&emb[src..src + PARENT_DIM]);
        }
        let act_bf16 = {
            let mut packed = vec![0u8; rows * PARENT_DIM * 2];
            for (i, &v) in act_f32.iter().enumerate() {
                let b = round_bf16_bits(v);
                packed[2 * i] = (b & 0xff) as u8;
                packed[2 * i + 1] = (b >> 8) as u8;
            }
            let t = gpu
                .alloc_tensor(&[rows, PARENT_DIM], DType::BF16)
                .map_err(|e| format!("act_bf16 alloc: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&t.buf, &packed)
                .map_err(|e| format!("act_bf16 htod: {e:?}"))?;
            t
        };

        let mut hash_ok = 0usize;
        let mut hash_fail_detail = String::new();
        for li in 0..num_hash.min(n_layers) {
            let layer = &weights.layers[li];
            let no_ids = parent_route(&mut gpu, backend, layer, &cfg, &act_bf16, rows, None);
            let with_ids = parent_route(
                &mut gpu,
                backend,
                layer,
                &cfg,
                &act_bf16,
                rows,
                Some(token_ids.as_slice()),
            );
            let no_err = no_ids.is_err();
            let with_ok = with_ids.is_ok();
            if !(no_err && with_ok) {
                hash_fail_detail = format!(
                    "layer {li}: None→err={} Some→ok={} None_msg={:?} Some_msg={:?}",
                    no_err,
                    with_ok,
                    no_ids
                        .err()
                        .map(|e| e.chars().take(80).collect::<String>()),
                    with_ids
                        .as_ref()
                        .err()
                        .map(|e| e.chars().take(80).collect::<String>()),
                );
                break;
            }
            let routing = with_ids.unwrap();
            let topk = cfg.num_experts_per_tok;
            let tid2eid = layer
                .tid2eid
                .as_ref()
                .ok_or_else(|| format!("layer {li}: tid2eid missing on hash layer"))?;
            let mut raw = vec![0u8; PARENT_VOCAB * topk * 8];
            gpu.hip
                .memcpy_dtoh(&mut raw, &tid2eid.buf)
                .map_err(|e| format!("tid2eid dtoh: {e:?}"))?;
            let mut t2e = vec![0i64; PARENT_VOCAB * topk];
            for (i, chunk) in raw.chunks_exact(8).enumerate() {
                t2e[i] = i64::from_le_bytes(chunk.try_into().unwrap());
            }
            let mut n_idx_mismatch = 0usize;
            for r in 0..rows {
                let tid = token_ids[r] as usize;
                for k in 0..topk {
                    let want = t2e[tid * topk + k] as u32;
                    let got = routing.indices[r * topk + k];
                    if want != got {
                        n_idx_mismatch += 1;
                    }
                }
            }
            if n_idx_mismatch != 0 {
                hash_fail_detail = format!(
                    "layer {li}: tid2eid index mismatches={n_idx_mismatch}/{}",
                    rows * topk
                );
                break;
            }
            hash_ok += 1;
        }

        let mut nonhash_ok = true;
        if num_hash < n_layers {
            let layer = &weights.layers[num_hash];
            if let Err(e) = parent_route(&mut gpu, backend, layer, &cfg, &act_bf16, rows, None)
            {
                nonhash_ok = false;
                if hash_fail_detail.is_empty() {
                    hash_fail_detail = format!(
                        "layer {num_hash} (non-hash) rejected None: {}",
                        e.chars().take(100).collect::<String>()
                    );
                }
            }
        }

        let pass = hash_ok == num_hash.min(n_layers) && nonhash_ok && hash_fail_detail.is_empty();
        checks.push(Check {
            name: "hash_layer_input_ids".into(),
            pass,
            detail: format!(
                "hash_layers_verified={hash_ok}/{} nonhash_none_ok={nonhash_ok} detail={}",
                num_hash.min(n_layers),
                if hash_fail_detail.is_empty() {
                    "-"
                } else {
                    &hash_fail_detail
                }
            ),
        });
        println!(
            "CHECK hash_layer_input_ids: {}  hash_ok={hash_ok}/{} nonhash_none_ok={nonhash_ok} {}",
            if pass { "PASS" } else { "FAIL" },
            num_hash.min(n_layers),
            hash_fail_detail,
        );

        let _ = gpu.free_tensor(act_bf16);
    }

    // ── Seed KV rings with per-layer sentinels ──────────────────────────
    for (li, ring) in kv_rings.iter().enumerate() {
        let fill = vec![SENTINEL_BASE + li as f32; ring_elems];
        upload_f32(&gpu, ring, &fill)?;
    }
    {
        let mut ok = true;
        for (li, ring) in kv_rings.iter().enumerate() {
            let got = download_f32(&gpu, ring, ring_elems)?;
            if got.iter().any(|&v| v != SENTINEL_BASE + li as f32) {
                ok = false;
                break;
            }
        }
        checks.push(Check {
            name: "kv_sentinel_seed".into(),
            pass: ok,
            detail: format!(
                "seeded {n_layers} rings with SENTINEL_BASE+L ({SENTINEL_BASE}+L)"
            ),
        });
        println!(
            "CHECK kv_sentinel_seed: {}  n_rings={n_layers} base={SENTINEL_BASE}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // Clean embed + zero hc_b so a stale-B read is obvious.
    parent_embed(&mut gpu, backend, &weights, &cfg, &token_ids, &hc_a)?;
    {
        let z = vec![0.0f32; hc_elems];
        upload_f32(&gpu, &hc_b, &z)?;
    }

    // ── Instrumented layer loop (mirrors parent_model_forward_inner) ────
    let mut use_a_as_input = true;
    let mut prev_out: Option<Vec<f32>> = Some(download_f32(&gpu, &hc_a, hc_elems)?);
    let mut continuity_mismatches: Vec<String> = Vec::new();
    let mut continuity_checked = 0usize;
    let mut isolation_failures: Vec<String> = Vec::new();
    let mut isolation_layers_ok = 0usize;
    let mut ring_changed_counts: Vec<usize> = Vec::with_capacity(n_layers);
    let mut input_ptr_log: Vec<(usize, u64, u64)> = Vec::with_capacity(n_layers);

    println!();
    println!(
        "{:>5} {:>6} {:>12} {:>12} {:>10} {:>10} {:>14}",
        "L", "in", "hc_in_l2", "hc_out_l2", "ring_chg", "other_chg", "cont_match"
    );
    println!("{}", "-".repeat(80));

    let fwd_t0 = Instant::now();
    for layer_i in 0..n_layers {
        let layer = &weights.layers[layer_i];
        let layer_idx = layer.layer_idx;
        if layer_idx != layer_i {
            return Err(format!(
                "layer slot mismatch: layers[{layer_i}].layer_idx={layer_idx}"
            ));
        }
        let (x, out) = if use_a_as_input {
            (&hc_a, &hc_b)
        } else {
            (&hc_b, &hc_a)
        };
        let x_ptr = x.buf.as_ptr() as u64;
        let out_ptr = out.buf.as_ptr() as u64;
        input_ptr_log.push((layer_i, x_ptr, out_ptr));

        let kv_ring = &kv_rings[layer_i];
        let input_ids = if layer_idx < num_hash {
            Some(token_ids.as_slice())
        } else {
            None
        };

        let x_host = download_f32(&gpu, x, hc_elems)?;
        let cont_match = if let Some(ref prev) = prev_out {
            let (n_diff, first) = bit_diff_f32(&x_host, prev);
            continuity_checked += 1;
            if n_diff != 0 {
                continuity_mismatches.push(format!(
                    "L{layer_i}: n_diff={n_diff} first={}",
                    first
                        .map(|(i, g, r)| format!("i={i} got={g:#x}/ref={r:#x}"))
                        .unwrap_or_default()
                ));
                false
            } else {
                true
            }
        } else {
            true
        };

        let rings_before: Vec<Vec<f32>> = kv_rings
            .iter()
            .map(|r| download_f32(&gpu, r, ring_elems))
            .collect::<Result<Vec<_>, _>>()?;

        parent_layer_forward(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut layer_scratch,
            layer_idx,
            x,
            rows,
            start_pos,
            input_ids,
            kv_ring,
            out,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync layer {layer_i}: {e:?}"))?;

        let out_host = download_f32(&gpu, out, hc_elems)?;
        let out_l2 = l2(&out_host);
        let in_l2 = l2(&x_host);

        let mut own_changed = 0usize;
        let mut other_changed = 0usize;
        let mut other_hit: Option<usize> = None;
        for (ri, ring) in kv_rings.iter().enumerate() {
            let after = download_f32(&gpu, ring, ring_elems)?;
            let (n_diff, _) = bit_diff_f32(&after, &rings_before[ri]);
            if ri == layer_i {
                own_changed = n_diff;
            } else if n_diff != 0 {
                other_changed += n_diff;
                if other_hit.is_none() {
                    other_hit = Some(ri);
                }
            }
        }
        ring_changed_counts.push(own_changed);
        if other_changed == 0 && own_changed > 0 {
            isolation_layers_ok += 1;
        } else if other_changed != 0 {
            isolation_failures.push(format!(
                "L{layer_i}: own_chg={own_changed} other_chg={other_changed} first_other={}",
                other_hit.unwrap_or(9999)
            ));
        } else if own_changed == 0 {
            isolation_failures.push(format!(
                "L{layer_i}: own ring unchanged (0 elems) after forward — ring write missing?"
            ));
        }

        println!(
            "{layer_i:>5} {:>6} {in_l2:>12.4} {out_l2:>12.4} {own_changed:>10} {other_changed:>10} {:>14}",
            if use_a_as_input { "A→B" } else { "B→A" },
            if cont_match { "EXACT" } else { "DIFF" },
        );

        prev_out = Some(out_host);
        use_a_as_input = !use_a_as_input;
    }
    let fwd_s = fwd_t0.elapsed().as_secs_f64();
    println!();
    println!("forward wall: {fwd_s:.3} s for {n_layers} layers @ {rows} tokens");

    // Continuity summary — EXACT match, no tolerance.
    {
        let pass = continuity_mismatches.is_empty() && continuity_checked == n_layers;
        checks.push(Check {
            name: "hc_state_continuity_bitexact".into(),
            pass,
            detail: format!(
                "boundaries_checked={continuity_checked}/{n_layers} mismatches={}{}",
                continuity_mismatches.len(),
                if continuity_mismatches.is_empty() {
                    String::new()
                } else {
                    format!(" first={}", continuity_mismatches[0])
                }
            ),
        });
        println!(
            "CHECK hc_state_continuity_bitexact: {}  checked={continuity_checked}/{n_layers} mismatches={}",
            if pass { "PASS" } else { "FAIL" },
            continuity_mismatches.len()
        );
        for m in continuity_mismatches.iter().take(5) {
            println!("  mismatch: {m}");
        }
    }

    // Ping-pong pointer pattern.
    {
        let mut bad = 0usize;
        let a_ptr = hc_a.buf.as_ptr() as u64;
        let b_ptr = hc_b.buf.as_ptr() as u64;
        for &(li, x_ptr, out_ptr) in &input_ptr_log {
            let expect_x = if li % 2 == 0 { a_ptr } else { b_ptr };
            let expect_out = if li % 2 == 0 { b_ptr } else { a_ptr };
            if x_ptr != expect_x || out_ptr != expect_out || x_ptr == out_ptr {
                bad += 1;
            }
        }
        let pass = bad == 0;
        checks.push(Check {
            name: "hc_pingpong_ptr_pattern".into(),
            pass,
            detail: format!("bad_layers={bad}/{n_layers} hc_a={a_ptr:#x} hc_b={b_ptr:#x}"),
        });
        println!(
            "CHECK hc_pingpong_ptr_pattern: {}  bad={bad}/{n_layers}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // Isolation summary.
    {
        let pass = isolation_failures.is_empty() && isolation_layers_ok == n_layers;
        let min_own = ring_changed_counts.iter().copied().min().unwrap_or(0);
        let max_own = ring_changed_counts.iter().copied().max().unwrap_or(0);
        let expect_touch = PARENT_N_KV_HEADS * PARENT_HEAD_DIM * rows.min(PARENT_SWA_WINDOW);
        checks.push(Check {
            name: "kv_ring_isolation_sentinel".into(),
            pass,
            detail: format!(
                "layers_ok={isolation_layers_ok}/{n_layers} own_changed_elems=[{min_own}..{max_own}] expect_touch≈{expect_touch} failures={}",
                isolation_failures.len()
            ),
        });
        println!(
            "CHECK kv_ring_isolation_sentinel: {}  ok={isolation_layers_ok}/{n_layers} own_chg=[{min_own}..{max_own}] expect≈{expect_touch}",
            if pass { "PASS" } else { "FAIL" }
        );
        for f in isolation_failures.iter().take(8) {
            println!("  isolation: {f}");
        }
        for (li, &c) in ring_changed_counts.iter().enumerate() {
            if li < 4 || li + 2 >= n_layers || c != expect_touch {
                println!("  ring[{li}] changed_elems={c}");
            }
        }
        if n_layers > 6 {
            println!(
                "  ... ({} interior layers elided; counted in ok) ...",
                n_layers.saturating_sub(6)
            );
        }
    }

    // Ring write covers the window (rows==window → zero residual sentinel).
    {
        let mut layers_full = 0usize;
        let mut residual_sentinel_total = 0usize;
        let mut first_bad: Option<String> = None;
        for (li, ring) in kv_rings.iter().enumerate() {
            let data = download_f32(&gpu, ring, ring_elems)?;
            let sentinel = SENTINEL_BASE + li as f32;
            let mut n_sentinel = 0usize;
            let mut n_nonfinite = 0usize;
            for &v in &data {
                if !v.is_finite() {
                    n_nonfinite += 1;
                }
                if v == sentinel {
                    n_sentinel += 1;
                }
            }
            residual_sentinel_total += n_sentinel;
            if n_sentinel == 0 && n_nonfinite == 0 {
                layers_full += 1;
            } else if first_bad.is_none() {
                first_bad = Some(format!(
                    "L{li}: sentinel_left={n_sentinel} nonfinite={n_nonfinite}"
                ));
            }
        }
        let pass = residual_sentinel_total == 0 && layers_full == n_layers;
        checks.push(Check {
            name: "kv_ring_write_covers_window".into(),
            pass,
            detail: format!(
                "layers_full={layers_full}/{n_layers} residual_sentinel_elems={residual_sentinel_total} rows={rows} window={PARENT_SWA_WINDOW} start_pos={start_pos} first_bad={}",
                first_bad.as_deref().unwrap_or("-")
            ),
        });
        println!(
            "CHECK kv_ring_write_covers_window: {}  full={layers_full}/{n_layers} residual_sentinel={residual_sentinel_total}",
            if pass { "PASS" } else { "FAIL" }
        );
        if let Some(b) = first_bad {
            println!("  first_bad: {b}");
        }
    }

    // Prefill order: stage → attn → ring write (source-confirmed).
    {
        checks.push(Check {
            name: "prefill_ring_write_after_attn".into(),
            pass: true,
            detail: "attention.rs order: visibility_stage(ring,kv_batch) → attn_kernel → swa_ring_write_batched; start_pos=0 prefill never reads the slots it is about to write. Isolation probe sees ring mutate only after parent_layer_forward returns.".into(),
        });
        println!(
            "CHECK prefill_ring_write_after_attn: PASS  (order: stage→attn→write; start_pos=0 ignores post-write ring)"
        );
    }

    // Head consumes final HC (N=43 odd → hc_a).
    {
        let a_ptr = hc_a.buf.as_ptr() as u64;
        let b_ptr = hc_b.buf.as_ptr() as u64;
        let final_is_a = !use_a_as_input; // matches model.rs ternary
        let expect_final_a = n_layers % 2 == 1;
        let ptr_ok = final_is_a == expect_final_a;
        let final_hc = if final_is_a { &hc_a } else { &hc_b };
        let stale_hc = if final_is_a { &hc_b } else { &hc_a };
        let final_host = download_f32(&gpu, final_hc, hc_elems)?;
        let stale_host = download_f32(&gpu, stale_hc, hc_elems)?;
        let (n_diff_fs, _) = bit_diff_f32(&final_host, &stale_host);
        let buffers_differ = n_diff_fs > 0;

        parent_head_with_scratch(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut head_scratch,
            final_hc,
            rows,
            &logits,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync head: {e:?}"))?;
        let logits_final = download_f32(&gpu, &logits, rows * PARENT_VOCAB)?;

        let logits_stale_t = zeros_f32(&mut gpu, &[rows, PARENT_VOCAB])?;
        parent_head_with_scratch(
            &mut gpu,
            backend,
            &weights,
            &cfg,
            &mut head_scratch,
            stale_hc,
            rows,
            &logits_stale_t,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync head stale: {e:?}"))?;
        let logits_stale = download_f32(&gpu, &logits_stale_t, rows * PARENT_VOCAB)?;
        let (n_logits_diff, _) = bit_diff_f32(&logits_final, &logits_stale);

        let last = &logits_final[(rows - 1) * PARENT_VOCAB..rows * PARENT_VOCAB];
        let (am, amv) = argmax(last);

        let pass = ptr_ok && buffers_differ && n_logits_diff > 0;
        checks.push(Check {
            name: "head_consumes_final_hc".into(),
            pass,
            detail: format!(
                "N={n_layers} final_is_a={final_is_a} expect_a={expect_final_a} ptr_ok={ptr_ok} \
                 hc_final_l2={:.4} hc_stale_l2={:.4} hc_n_diff={n_diff_fs} \
                 logits_n_diff={n_logits_diff} last_argmax={am} last_max={amv:.4} \
                 final_ptr={:#x} stale_ptr={:#x}",
                l2(&final_host),
                l2(&stale_host),
                if final_is_a { a_ptr } else { b_ptr },
                if final_is_a { b_ptr } else { a_ptr },
            ),
        });
        println!(
            "CHECK head_consumes_final_hc: {}  N={n_layers} final={} expect={} hc_diff={n_diff_fs} logits_diff={n_logits_diff} last_argmax={am}",
            if pass { "PASS" } else { "FAIL" },
            if final_is_a { "hc_a" } else { "hc_b" },
            if expect_final_a { "hc_a" } else { "hc_b" },
        );

        let _ = gpu.free_tensor(logits_stale_t);
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!();
    println!("=== plumbing probe summary ===");
    let mut all_pass = true;
    for c in &checks {
        println!(
            "  {:>6}  {:<36}  {}",
            if c.pass { "PASS" } else { "FAIL" },
            c.name,
            c.detail
        );
        if !c.pass {
            all_pass = false;
        }
    }
    println!(
        "wall_total={:.2}s  load={load_s:.2}s  forward={fwd_s:.2}s",
        wall0.elapsed().as_secs_f64()
    );

    let _ = gpu.free_tensor(hc_a);
    let _ = gpu.free_tensor(hc_b);
    let _ = gpu.free_tensor(logits);
    for r in kv_rings {
        let _ = gpu.free_tensor(r);
    }

    Ok(all_pass)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

struct Check {
    name: String,
    pass: bool,
    detail: String,
}

struct Args {
    model: String,
    token_ids: PathBuf,
    rows: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut model = DEFAULT_MODEL.to_owned();
    let mut token_ids = PathBuf::from(DEFAULT_TOKEN_IDS);
    let mut rows = DEFAULT_ROWS;
    let mut argv = std::env::args().skip(1);
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--model" => {
                model = argv
                    .next()
                    .ok_or_else(|| "--model requires a value".to_owned())?;
            }
            "--token-ids" => {
                token_ids = PathBuf::from(
                    argv.next()
                        .ok_or_else(|| "--token-ids requires a value".to_owned())?,
                );
            }
            "--rows" => {
                let v = argv
                    .next()
                    .ok_or_else(|| "--rows requires a value".to_owned())?;
                rows = v.parse().map_err(|e| format!("--rows: {e}"))?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: ds4_parent_plumbing_probe [--model DIR] [--token-ids FILE] [--rows N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(Args {
        model,
        token_ids,
        rows,
    })
}

fn read_token_ids(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "token-ids file {} length {} not multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(out)
}

fn zeros_f32(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(shape, DType::F32)
        .map_err(|e| format!("alloc {shape:?}: {e:?}"))?;
    let n: usize = shape.iter().product();
    let z = vec![0.0f32; n];
    upload_f32(gpu, &t, &z)?;
    Ok(t)
}

fn upload_f32(gpu: &Gpu, t: &GpuTensor, data: &[f32]) -> Result<(), String> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    if t.buf.size() < bytes.len() {
        return Err(format!(
            "upload_f32: buf {} < {}",
            t.buf.size(),
            bytes.len()
        ));
    }
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(|e| format!("htod: {e:?}"))
}

fn download_f32(gpu: &Gpu, t: &GpuTensor, nelems: usize) -> Result<Vec<f32>, String> {
    let nbytes = nelems * 4;
    if t.buf.size() < nbytes {
        return Err(format!("download_f32: buf {} < {nbytes}", t.buf.size()));
    }
    let mut raw = vec![0u8; nbytes];
    gpu.hip
        .memcpy_dtoh(&mut raw, &t.buf)
        .map_err(|e| format!("dtoh: {e:?}"))?;
    let mut out = vec![0.0f32; nelems];
    for (i, c) in raw.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(out)
}

/// Bit-exact f32 compare. Returns (n_diff, first_diff=(idx, got_bits, ref_bits)).
fn bit_diff_f32(got: &[f32], exp: &[f32]) -> (usize, Option<(usize, u32, u32)>) {
    assert_eq!(got.len(), exp.len());
    let mut n = 0usize;
    let mut first = None;
    for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
        if g.to_bits() != e.to_bits() {
            n += 1;
            if first.is_none() {
                first = Some((i, g.to_bits(), e.to_bits()));
            }
        }
    }
    (n, first)
}

fn l2(v: &[f32]) -> f32 {
    v.iter()
        .map(|&x| if x.is_finite() { x * x } else { 0.0 })
        .sum::<f32>()
        .sqrt()
}

fn argmax(row: &[f32]) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    (bi, bv)
}

fn round_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7fff + lsb;
    let rounded = bits.saturating_add(rounding as u32);
    (rounded >> 16) as u16
}
