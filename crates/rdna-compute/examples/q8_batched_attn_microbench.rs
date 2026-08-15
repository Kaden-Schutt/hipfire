// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Microbench for the no-LDS-cap batched Q8 flash attention introduced in
// fix/q8-batched-masked-no-lds-cap. Compares, at a single FA-layer scale:
//
//   (A) NEW  attention_flash_q8_0_batched_masked   — one batched launch
//   (B) OLD  attention_flash_q8_0 looped per query  — the >15k fallback it replaces
//
// at a controlled (n, max_ctx_len) shape so rocprof / wall timing isn't
// drowned by 64 layers × many prefill chunks. Reports wall ms (median of 5)
// for each. The point: confirm NEW ≤ OLD (the replacement is not a perf
// regression) at long context, where OLD launches `n` separate kernels.
//
// Shapes default to Qwen3.5-9B FA: n_heads=40, n_kv_heads=8, head_dim=256.
// Override via env: NH, NKV, HD, N (batch/query rows), CTX (max_ctx_len).
//
// Run (gfx906): cargo run --release --example q8_batched_attn_microbench

use rdna_compute::{DType, Gpu};

/// The VRAM budget this harness measures against. `kv_slots::preflight_alloc`
/// takes it as a parameter because production `src/` may not read HIPFIRE_*
/// directly (scripts/check-env-docs.py); examples are exempt, so the override
/// is read here.
fn vram_budget_bytes() -> u64 {
    std::env::var("HIPFIRE_VRAM_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(rdna_compute::kv_slots::R9700_VRAM_BYTES)
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Median-of-`iters` wall time (ms) for one closure, after `warmups` untimed
/// runs, with exactly one `device_synchronize` per timed call — never per
/// kernel, since a per-op sync fabricates false GPU speedups by serializing
/// work that would otherwise overlap.
///
/// Module-level so it is callable from both `main()` and the multi-slot
/// bench helpers below (`bench_slots`, `bench_lds_path`). `main()`'s
/// original single-shape benchmarks build a local closure over this
/// function so their existing call sites are unchanged.
fn time_ms(gpu: &mut Gpu, warmups: usize, iters: usize, f: &dyn Fn(&mut Gpu)) -> f64 {
    for _ in 0..warmups {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut ts = vec![];
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        f(gpu);
        gpu.hip.device_synchronize().unwrap();
        ts.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[ts.len() / 2]
}

/// Preflight every allocation this benchmark makes through the shared
/// `kv_slots::preflight_alloc` gate — the 32 GiB R9700 deployment budget AND
/// `MemAvailable` headroom on this no-swap dev box, in one place, after the
/// 2026-08-07 global-OOM incident (nine of the user's applications killed by
/// harness memory pressure; see scripts/run-bounded.sh's header). `what` must
/// be the TOTAL held live at once for that call, not one buffer.
///
/// Returns `true` if the configuration may proceed. On refusal it prints why
/// and returns `false` — callers MUST skip the configuration, never proceed
/// anyway (a hand-rolled assert here would just re-introduce the exact
/// footgun this gate replaces).
fn preflight_or_skip(total_alloc_bytes: u64, what: &str) -> bool {
    match rdna_compute::kv_slots::preflight_alloc(total_alloc_bytes, vram_budget_bytes(), what) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("SKIP: {e}");
            false
        }
    }
}

fn main() {
    let nh = env_usize("NH", 40);
    let nkv = env_usize("NKV", 8);
    let hd = env_usize("HD", 256);
    let n = env_usize("N", 512); // query rows in the prefill chunk
    let ctx = env_usize("CTX", 20000); // max_ctx_len — above the 15k cliff
    let warmups = env_usize("WARMUPS", 3);
    let iters = env_usize("ITERS", 5);

    assert!(hd % 32 == 0, "head_dim must be a multiple of 32");
    let mut gpu = Gpu::init().expect("gpu init");

    // Q8 K/V cache layout (matches kv_cache.k_gpu): per position,
    // n_kv_heads * (head_dim/32) blocks of 34 bytes (fp16 scale + 32 i8).
    let blocks_per_head = hd / 32;
    let bytes_per_pos = nkv * blocks_per_head * 34;
    let cache_bytes = ctx * bytes_per_pos;

    // Resolved via kv_slots::attn_tile_size — the single source of truth
    // shared with launch_asym_flash_batched's own tile_size resolution
    // (attention.rs) and bench_slots below. HIPFIRE_ATTN_TILE_SIZE is a
    // process-global env var read inside the kernel launcher itself, so it
    // silently changes every call through this file, not just the
    // multi-slot sweep appended below. A value here that diverges from the
    // launcher's actual max_tiles undersizes `partials` and corrupts device
    // memory (found empirically before this was unified: TILE_SIZE=64
    // crashed with "illegal memory access" downstream, in the unrelated
    // multi-slot section, because this buffer was already too small
    // upstream). Resolved up front so both the preflight estimate below and
    // the actual `partials` allocation further down use the identical value.
    let tile = gpu.attn_tile_size();
    let max_tiles = ctx.div_ceil(tile);

    // This section's own allocations, checked against the same gate the new
    // multi-slot sweep below uses — see preflight_or_skip's doc comment.
    let single_shape_bytes = 2 * cache_bytes as u64 // k_cache + v_cache
        + 2 * (n * nh * hd * 4) as u64 // q + out
        + (n * nh * max_tiles * (2 + hd) * 4) as u64; // partials
    if !preflight_or_skip(single_shape_bytes, "main: single-shape section") {
        return;
    }

    // Fill K/V with a plausible-magnitude pattern: scale=1.0 (fp16 0x3C00),
    // codes = small ramp. Not numerically meaningful — we time, not verify
    // (correctness is the NIAH gate on the 32k fixture).
    let mut kv = vec![0u8; cache_bytes];
    for blk in kv.chunks_mut(34) {
        blk[0] = 0x00;
        blk[1] = 0x3C; // fp16 1.0 little-endian
        for (j, b) in blk[2..].iter_mut().enumerate() {
            *b = ((j as i32 % 7) - 3) as i8 as u8;
        }
    }
    let k_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("k upload");
    let v_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("v upload");

    // Q: [n × n_heads × head_dim] f32.
    let q_data: Vec<f32> = (0..n * nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q = gpu.upload_f32(&q_data, &[n * nh * hd]).expect("q upload");
    let out = gpu.zeros(&[n * nh * hd], DType::F32).expect("out");

    // positions: i32 bits in f32 slot — positions[b] = ctx - n + b (the
    // queries sit at the tail of the context, as in real tail-chunk prefill).
    let pos_data: Vec<i32> = (0..n).map(|b| (ctx - n + b) as i32).collect();
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, n * 4) };
    let positions = gpu.upload_raw(pos_bytes, &[n]).expect("pos upload");

    // flash_partials: [sub_batch × n_heads × max_tiles × (2+head_dim)].
    // Size it for the full batch so sub_batch == n (single chunk). `tile` /
    // `max_tiles` were already resolved above (before the preflight check),
    // matching launch_asym_flash_batched's own resolution — see that comment.
    let partials_numel = n * nh * max_tiles * (2 + hd);
    let partials = gpu.zeros(&[partials_numel], DType::F32).expect("partials");

    eprintln!(
        "shape: nh={nh} nkv={nkv} hd={hd} n={n} ctx={ctx} | cache={:.1} MiB partials={:.1} MiB",
        cache_bytes as f64 / 1048576.0,
        partials_numel as f64 * 4.0 / 1048576.0,
    );

    let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu)| -> f64 { time_ms(gpu, warmups, iters, f) };

    // Windowed batched flash. WINDOW=0 is full causal; WINDOW>0 clips to the
    // sliding window. The grid (max_tiles) AND the reduce (n_tiles) are sized by
    // max_ctx_len regardless of window — so sweeping CTX at a FIXED window tests
    // whether windowing actually reduces prefill cost, or only skips the dots
    // while still paying O(ctx) tile-launch + reduce overhead.
    let window = env_usize("WINDOW", 0) as i32;
    let new_ms = time(&mut gpu, &|g: &mut Gpu| {
        if window == 0 {
            g.attention_flash_q8_0_batched_masked(
                &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, ctx, n, &partials,
                None, 0, 0,
            )
            .expect("non-windowed batched");
        } else {
            g.attention_flash_q8_0_batched_masked_windowed(
                &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, ctx, n, &partials,
                None, 0, 0, window,
            )
            .expect("windowed batched");
        }
    });

    println!(
        "WINDOW={window:6} CTX={ctx:6} N={n:4} HD={hd} : batched flash {new_ms:8.2} ms  ({:6.1} us/query-row)",
        new_ms * 1000.0 / n as f64
    );

    // Query-tiled flash prefill. BR/BC swept via env; LDS is independent of ctx.
    let br = env_usize("BR", 16);
    let bc = env_usize("BC", 32);
    let flash_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_q8_0_flash_prefill(
            &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, n, br, bc,
        )
        .expect("flash prefill");
    });
    println!(
        "flash_prefill br={br} bc={bc} CTX={ctx} N={n}: {flash_ms:8.2} ms  \
         ({:6.1} us/query-row)  speedup_vs_tiled={:.2}x",
        flash_ms * 1000.0 / n as f64,
        new_ms / flash_ms
    );

    // WMMA (matrix-core) variant of the query-tiled kernel. Fixed 16x16 tiles.
    let wmma_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_q8_0_flash_prefill_wmma(
            &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, n,
        )
        .expect("wmma flash prefill");
    });
    println!(
        "flash_wmma       CTX={ctx} N={n}: {wmma_ms:8.2} ms  \
         ({:6.1} us/query-row)  vs_tiled={:.2}x  vs_scalar_flash={:.2}x",
        wmma_ms * 1000.0 / n as f64,
        new_ms / wmma_ms,
        flash_ms / wmma_ms
    );

    // The legacy LDS-backed kernel is only launchable while
    // (max_ctx_len + block + head_dim) * 4 <= 64 KB; above that it cannot run
    // at all, which is exactly why dispatch crosses over at 8192.
    let legacy_lds = (ctx + 256 + hd) * 4;
    if legacy_lds <= 64 * 1024 {
        let legacy_ms = time(&mut gpu, &|g: &mut Gpu| {
            g.attention_q8_0_kv_batched_masked(
                &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, ctx, n, None, 0, 0,
            )
            .expect("legacy lds kernel");
        });
        println!(
            "legacy_lds       CTX={ctx} N={n}: {legacy_ms:8.2} ms  \
             ({:6.1} us/query-row)  flash_speedup_vs_legacy={:.2}x",
            legacy_ms * 1000.0 / n as f64,
            legacy_ms / flash_ms
        );
    } else {
        println!("legacy_lds       CTX={ctx}: N/A (needs {legacy_lds} B LDS > 64 KB)");
    }

    // ── Multi-slot sweep: batched vs sequential ─────────────────────────────
    // (A) one batched launch over n_slots, versus
    // (B) n_slots sequential single-slot launches.
    // Spec §2 criterion 2: batched must beat sequential at every n_slots >= 2.
    // A regression here is a failure, not a tuning outcome.
    //
    // Every shape below is checked against MemAvailable and the 32 GiB R9700
    // target budget inside bench_slots/bench_lds_path (preflight_or_skip, via
    // kv_slots::preflight_alloc), before any upload happens.
    if let Some(avail) = rdna_compute::kv_slots::mem_available_bytes() {
        eprintln!(
            "preflight: {:.2} GiB MemAvailable (no swap on this box) before the multi-slot sweep",
            avail as f64 / 1073741824.0
        );
    }
    // Bytes moved per launch for GB/s reporting: K+V read, Q8_0 layout, one
    // "layer" (this microbench times a single kernel launch, not a full
    // multi-layer forward pass) — bytes = ctx * n_kv_heads * (head_dim/32) *
    // 34 * 2 (K and V), summed over every slot in the batch.
    //
    // Also counted: `partials` traffic. The tile kernel writes `rows *
    // n_heads * max_tiles * (2+head_dim)` f32s of partials and the reduce
    // kernel reads every one of them back (see launch_asym_flash_batched's
    // unconditional reduce-kernel launch in attention.rs) — that round trip
    // is real DRAM traffic this benchmark's own kernels perform, not an
    // artefact of measurement. Omitting it understated real traffic by
    // ~24% at n_slots=8/ctx=32768/TILE=128 (review finding I1). Since this
    // sweep uses one ctx per shape (`per_slot_ctx` for every slot), total
    // partials bytes are identical whether that ctx's rows arrive as one
    // batched launch or as `n_slots` sequential 1-row launches — same rows,
    // same max_tiles, same reduce cost — so one formula covers both arms.
    let bw_nkv = env_usize("NKV", 2);
    let bw_hd = env_usize("HD", 256);
    let bw_nh = env_usize("NH", 16); // MUST mirror bench_slots' own NH resolution
    let bw_per_pos_bytes = (bw_nkv * (bw_hd / 32) * 34) as f64;
    let bw_tile = gpu.attn_tile_size();
    for &n_slots in &[1usize, 2, 4, 8] {
        let per_slot_ctx = env_usize("SLOT_CTX", 32768);
        let shape = vec![per_slot_ctx; n_slots];
        let Some((batched_ms, seq_ms)) = bench_slots(&mut gpu, &shape, &vec![1usize; n_slots])
        else {
            println!("n_slots={n_slots:2} ctx={per_slot_ctx:6} : SKIPPED (preflight refused, see stderr)");
            continue;
        };
        let total_kv_bytes = n_slots as f64 * per_slot_ctx as f64 * bw_per_pos_bytes * 2.0;
        let max_tiles = per_slot_ctx.div_ceil(bw_tile);
        let partials_bytes =
            n_slots as f64 * bw_nh as f64 * max_tiles as f64 * (2.0 + bw_hd as f64) * 4.0 * 2.0; // write + read
        let total_bytes = total_kv_bytes + partials_bytes;
        let batched_gbs = total_bytes / (batched_ms * 1e6);
        let seq_gbs = total_bytes / (seq_ms * 1e6);
        println!(
            "n_slots={n_slots:2} ctx={per_slot_ctx:6} : batched {batched_ms:8.3} ms ({batched_gbs:6.1} GB/s)  \
             sequential {seq_ms:8.3} ms ({seq_gbs:6.1} GB/s)  speedup {:.2}x  \
             [KV {:.1} MB + partials {:.1} MB]",
            seq_ms / batched_ms,
            total_kv_bytes / 1e6,
            partials_bytes / 1e6,
        );
        if n_slots >= 2 {
            assert!(
                batched_ms < seq_ms,
                "batched ({batched_ms:.3} ms) must beat sequential \
                 ({seq_ms:.3} ms) at n_slots={n_slots} — spec §2 criterion 2"
            );
        }
    }

    // Ragged batch: max_tiles is derived from the batch MAXIMUM context, so
    // short slots launch tiles that immediately early-exit. Measure that waste
    // rather than assuming it is negligible (spec §7).
    {
        let ragged = vec![1024usize, 4096, 32768, 100_000];
        let uniform = vec![100_000usize; 4];
        let ragged_result = bench_slots(&mut gpu, &ragged, &vec![1usize; 4]);
        let uniform_result = bench_slots(&mut gpu, &uniform, &vec![1usize; 4]);
        match (ragged_result, uniform_result) {
            (Some((ragged_ms, _)), Some((uniform_ms, _))) => {
                let useful: usize = ragged.iter().sum();
                let launched = 100_000 * 4;
                // Partials round-trip: both shapes share max_ctx=100_000 and
                // 4 rows (m_per_slot is [1,1,1,1] for both), so max_tiles and
                // total partials bytes are IDENTICAL for ragged and
                // uniform-max — the waste being measured here is entirely in
                // KV tile launches that early-exit, not in partials traffic.
                // Included anyway for GB/s consistency with the n_slots
                // sweep above (review finding I1).
                let max_tiles = 100_000usize.div_ceil(bw_tile);
                let partials_bytes =
                    4.0 * bw_nh as f64 * max_tiles as f64 * (2.0 + bw_hd as f64) * 4.0 * 2.0;
                let useful_gbs =
                    (useful as f64 * bw_per_pos_bytes * 2.0 + partials_bytes) / (ragged_ms * 1e6);
                let uniform_gbs = (launched as f64 * bw_per_pos_bytes * 2.0 + partials_bytes)
                    / (uniform_ms * 1e6);
                // Time cost per 1000 USEFUL positions — the number that
                // actually answers "how much does raggedness cost", as
                // distinct from "what fraction of the launched grid
                // early-exits" (review finding I3: the 65.5% waste figure
                // below is real but is 6x the actual time cost).
                let ragged_us_per_1k = ragged_ms * 1000.0 / (useful as f64 / 1000.0);
                let uniform_us_per_1k = uniform_ms * 1000.0 / (launched as f64 / 1000.0);
                println!(
                    "ragged {ragged_ms:8.3} ms ({useful_gbs:6.1} GB/s useful-KV+partials basis) vs \
                     uniform-max {uniform_ms:8.3} ms ({uniform_gbs:6.1} GB/s)  (useful KV \
                     {useful}, tiles sized for {launched}, waste {:.1}%)  \
                     [time cost: {ragged_us_per_1k:.2} us/1000-useful-positions ragged vs \
                     {uniform_us_per_1k:.2} us/1000-useful-positions uniform, \
                     {:.1}% slower]",
                    100.0 * (1.0 - useful as f64 / launched as f64),
                    100.0 * (ragged_us_per_1k / uniform_us_per_1k - 1.0),
                );
            }
            _ => println!("ragged/uniform-max: SKIPPED (preflight refused, see stderr)"),
        }
    }

    // Which path wins below the crossover, at multi-slot batch? The router
    // sends ctx < LDS_CTX_LIMIT (15000) to the LDS kernel, whose grid is
    // [n_heads, batch] — thin. Measure both paths at the same shape rather
    // than assuming the existing single-sequence crossover still holds.
    for &ctx in &[2048usize, 8192, 14000] {
        for &n_slots in &[1usize, 4, 8] {
            let Some(lds_ms) = bench_lds_path(&mut gpu, ctx, n_slots) else {
                println!("ctx={ctx:6} n_slots={n_slots:2} : SKIPPED (LDS path preflight refused)");
                continue;
            };
            let Some(tile_ms) = bench_tile_path(&mut gpu, ctx, n_slots) else {
                println!("ctx={ctx:6} n_slots={n_slots:2} : SKIPPED (tile path preflight refused)");
                continue;
            };
            println!(
                "ctx={ctx:6} n_slots={n_slots:2} : LDS {lds_ms:8.3} ms  \
                 tile {tile_ms:8.3} ms  winner={}",
                if lds_ms < tile_ms { "LDS" } else { "TILE" }
            );
        }
    }
}

/// Time one batched multi-slot launch against n_slots sequential single-slot
/// launches over the same arena. Returns `Some((batched_ms, sequential_ms))`,
/// or `None` if `preflight_or_skip` refused this configuration (caller must
/// skip it, not substitute a default). Frees every tensor it allocates
/// before returning — this runs inside sweep loops and `GpuTensor` has no
/// `Drop` (see `GeneralBatch::free`'s doc comment in
/// test_batched_attn_slots.rs for the OOM this pattern avoids, and the
/// 2026-08-07 global-OOM incident for why "runs inside a loop" is not a
/// hypothetical here).
fn bench_slots(gpu: &mut Gpu, seq_lens: &[usize], m_per_slot: &[usize]) -> Option<(f64, f64)> {
    use rdna_compute::kv_slots::{build_arena, build_tiles, KvSlotDesc};

    let nh = env_usize("NH", 16);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let warmups = env_usize("WARMUPS", 3);
    let iters = env_usize("ITERS", 5);
    let per_pos_bytes = nkv * (hd / 32) * 34;

    let (arena, descs) = build_arena(seq_lens, per_pos_bytes, None);

    let rows: usize = m_per_slot.iter().sum();
    let (tile_slot, _, _) = build_tiles(m_per_slot, 1);
    let max_ctx = *seq_lens.iter().max().unwrap();
    // Resolved via kv_slots::attn_tile_size — the single source of truth
    // shared with launch_asym_flash_batched's own resolution (attention.rs)
    // and main()'s single-shape section above. A value that diverges from
    // the launcher's own resolution undersizes `partials` and corrupts
    // device memory (empirically: illegal memory access at TILE_SIZE=64
    // before this was unified).
    let max_tiles = max_ctx.div_ceil(gpu.attn_tile_size());

    // Preflight (host RAM + 32 GiB target budget), computed analytically
    // from host-side sizes BEFORE any upload — see module doc.
    let slabs_bytes: u64 = descs
        .iter()
        .map(|d| d.cap as u64 * per_pos_bytes as u64)
        .sum();
    let total_alloc_bytes: u64 = 2 * arena.len() as u64 // k_cache + v_cache
        + (descs.len() * std::mem::size_of::<KvSlotDesc>()) as u64 // d_descs
        + (tile_slot.len() * 4) as u64 // d_row_slot
        + (rows * 4) as u64 // positions
        + (rows * nh * hd * 4) as u64 * 2 // q + out (f32)
        + (rows * nh * max_tiles * (2 + hd) * 4) as u64 // partials
        + 2 * slabs_bytes // sequential-arm slabs: DISTINCT K and V buffers per
                           // slot, matching the batched arm's k_cache/v_cache
                           // — see the comment on `slabs` below for why K==V
                           // aliasing here would understate the batching win
        + (seq_lens.len() * 4) as u64; // sequential-arm per-slot positions
    if !preflight_or_skip(total_alloc_bytes, "bench_slots") {
        return None;
    }

    let k_cache = gpu.upload_raw(&arena, &[arena.len()]).expect("k arena");
    let v_cache = gpu.upload_raw(&arena, &[arena.len()]).expect("v arena");
    let desc_bytes = unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            descs.len() * std::mem::size_of::<KvSlotDesc>(),
        )
    };
    let d_descs = gpu
        .upload_raw(desc_bytes, &[descs.len() * 3])
        .expect("descs");
    let ts_bytes =
        unsafe { std::slice::from_raw_parts(tile_slot.as_ptr() as *const u8, tile_slot.len() * 4) };
    let d_row_slot = gpu
        .upload_raw(ts_bytes, &[tile_slot.len()])
        .expect("row_slot");

    // positions[r] = that row's own slot's seq_len - 1
    let mut pos_data: Vec<i32> = Vec::with_capacity(rows);
    for (slot, &m) in m_per_slot.iter().enumerate() {
        for _ in 0..m {
            pos_data.push(seq_lens[slot] as i32 - 1);
        }
    }
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, rows * 4) };
    let positions = gpu.upload_raw(pos_bytes, &[rows]).expect("positions");

    let q_data: Vec<f32> = (0..rows * nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q = gpu.upload_f32(&q_data, &[rows * nh * hd]).expect("q");
    let out = gpu.zeros(&[rows * nh * hd], DType::F32).expect("out");

    let partials = gpu
        .zeros(&[rows * nh * max_tiles * (2 + hd)], DType::F32)
        .expect("partials");

    let batched = time_ms(gpu, warmups, iters, &|g: &mut Gpu| {
        g.attention_flash_q8_0_batched_masked_slots(
            &q,
            &k_cache,
            &v_cache,
            &out,
            &positions,
            nh,
            nkv,
            hd,
            max_ctx,
            max_ctx,
            rows,
            &partials,
            None,
            0,
            0,
            Some(&d_descs),
            Some(&d_row_slot),
        )
        .expect("batched slots");
    });

    // Sequential arm: one legacy launch per slot, against that slot's slab.
    //
    // K and V are uploaded as two DISTINCT buffers per slot (same source
    // bytes, two separate device allocations) — mirroring exactly how the
    // batched arm above builds k_cache/v_cache as two distinct uploads of
    // `arena`. Passing one shared buffer as both K and V here would make
    // each sequential launch touch only HALF the distinct bytes the batched
    // arm touches (at ctx=32768 that's 17.8 MB vs 35.6 MB per slot): the
    // aliased half fits this box's MALL cache while the batched arm's full
    // working set does not, so the sequential arm would come out
    // artificially fast and the batching win would be understated. See
    // review finding C1.
    //
    // The per-slot slabs and positions are uploaded BEFORE the timed region.
    // Uploading inside the closure would charge the sequential arm a host->device
    // cost the batched arm never pays, flattering the batched path and making the
    // spec §2 criterion-2 assertion meaningless.
    let slabs: Vec<_> = seq_lens
        .iter()
        .enumerate()
        .map(|(slot, &sl)| {
            let off = descs[slot].k_base as usize;
            let len = descs[slot].cap as usize * per_pos_bytes;
            let slab_k = gpu
                .upload_raw(&arena[off..off + len], &[len])
                .expect("slab k");
            let slab_v = gpu
                .upload_raw(&arena[off..off + len], &[len])
                .expect("slab v");
            let pos = sl as i32 - 1;
            let pb = unsafe { std::slice::from_raw_parts(&pos as *const i32 as *const u8, 4) };
            let p = gpu.upload_raw(pb, &[1]).expect("slab pos");
            (slab_k, slab_v, p, sl)
        })
        .collect();

    let sequential = time_ms(gpu, warmups, iters, &|g: &mut Gpu| {
        for (slab_k, slab_v, p, sl) in &slabs {
            g.attention_flash_q8_0_batched_masked(
                &q, slab_k, slab_v, &out, p, nh, nkv, hd, *sl, *sl, 1, &partials, None, 0, 0,
            )
            .expect("sequential");
        }
    });

    gpu.free_tensor(k_cache).expect("free k_cache");
    gpu.free_tensor(v_cache).expect("free v_cache");
    gpu.free_tensor(d_descs).expect("free d_descs");
    gpu.free_tensor(d_row_slot).expect("free d_row_slot");
    gpu.free_tensor(positions).expect("free positions");
    gpu.free_tensor(q).expect("free q");
    gpu.free_tensor(out).expect("free out");
    gpu.free_tensor(partials).expect("free partials");
    for (slab_k, slab_v, p, _) in slabs {
        gpu.free_tensor(slab_k).expect("free slab k");
        gpu.free_tensor(slab_v).expect("free slab v");
        gpu.free_tensor(p).expect("free slab pos");
    }

    Some((batched, sequential))
}

/// LDS-decode-kernel time only, at a single-slot-per-row "decode" shape
/// (M=1 per slot) — the shape the LDS kernel's grid `[n_heads, batch]`
/// targets. Uses the same `kv_slots` arena/descriptor layout as `bench_slots`
/// so the LDS-vs-tile comparison in `main()` is not silently measuring two
/// different address spaces. `None` if `preflight_or_skip` refused.
fn bench_lds_path(gpu: &mut Gpu, ctx: usize, n_slots: usize) -> Option<f64> {
    use rdna_compute::kv_slots::{build_arena, KvSlotDesc};

    let nh = env_usize("NH", 16);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let warmups = env_usize("WARMUPS", 3);
    let iters = env_usize("ITERS", 5);
    let per_pos_bytes = nkv * (hd / 32) * 34;

    let seq_lens = vec![ctx; n_slots];
    let (arena, descs) = build_arena(&seq_lens, per_pos_bytes, None);
    let rows = n_slots; // one decode row per slot

    let total_alloc_bytes: u64 = 2 * arena.len() as u64
        + (descs.len() * std::mem::size_of::<KvSlotDesc>()) as u64
        + (rows * 4) as u64 * 2 // row_slot + positions
        + (rows * nh * hd * 4) as u64 * 2; // q + out
    if !preflight_or_skip(total_alloc_bytes, "bench_lds_path") {
        return None;
    }

    let k_cache = gpu.upload_raw(&arena, &[arena.len()]).expect("k arena");
    let v_cache = gpu.upload_raw(&arena, &[arena.len()]).expect("v arena");
    let desc_bytes = unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            descs.len() * std::mem::size_of::<KvSlotDesc>(),
        )
    };
    let d_descs = gpu.upload_raw(desc_bytes, &[descs.len()]).expect("descs");
    let row_slot: Vec<i32> = (0..n_slots as i32).collect();
    let rs_bytes =
        unsafe { std::slice::from_raw_parts(row_slot.as_ptr() as *const u8, row_slot.len() * 4) };
    let d_row_slot = gpu
        .upload_raw(rs_bytes, &[row_slot.len()])
        .expect("row_slot");

    let pos_data: Vec<i32> = seq_lens.iter().map(|&sl| sl as i32 - 1).collect();
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, rows * 4) };
    let positions = gpu.upload_raw(pos_bytes, &[rows]).expect("positions");

    let q_data: Vec<f32> = (0..rows * nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q = gpu.upload_f32(&q_data, &[rows * nh * hd]).expect("q");
    let out = gpu.zeros(&[rows * nh * hd], DType::F32).expect("out");

    let ms = time_ms(gpu, warmups, iters, &|g: &mut Gpu| {
        g.attention_q8_0_kv_batched_masked_slots(
            &q,
            &k_cache,
            &v_cache,
            &out,
            &positions,
            nh,
            nkv,
            hd,
            ctx,
            ctx,
            rows,
            None,
            0,
            0,
            Some(&d_descs),
            Some(&d_row_slot),
        )
        .expect("lds path");
    });

    gpu.free_tensor(k_cache).expect("free k_cache");
    gpu.free_tensor(v_cache).expect("free v_cache");
    gpu.free_tensor(d_descs).expect("free d_descs");
    gpu.free_tensor(d_row_slot).expect("free d_row_slot");
    gpu.free_tensor(positions).expect("free positions");
    gpu.free_tensor(q).expect("free q");
    gpu.free_tensor(out).expect("free out");

    Some(ms)
}

/// Tile-kernel (`attention_flash_q8_0_batched_masked_slots`) time only, at
/// the same M=1-per-slot decode shape as `bench_lds_path`. Reuses
/// `bench_slots` and discards its sequential-arm timing so both paths in the
/// LDS-vs-tile crossover share one arena/descriptor construction. `None` if
/// `bench_slots` was skipped by the preflight gate.
fn bench_tile_path(gpu: &mut Gpu, ctx: usize, n_slots: usize) -> Option<f64> {
    let seq_lens = vec![ctx; n_slots];
    let m_per_slot = vec![1usize; n_slots];
    bench_slots(gpu, &seq_lens, &m_per_slot).map(|(batched_ms, _seq_ms)| batched_ms)
}
