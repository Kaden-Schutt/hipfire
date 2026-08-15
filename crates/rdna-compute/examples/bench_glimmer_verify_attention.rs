// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Focused gfx1201 Muse Glimmer verifier attention microbench.
// Fixed shape B16 / H32 / KVH2 / HD128, Q8 cache, native window W=2048.
//
// Arms (exact vs approximate classification is printed on every RESULT row):
//   scalar_full     — exact full-causal F32 scalar Q8 flash (oracle)
//   wmma_full       — exact full-causal WMMA (reduced precision)
//   longspec_exact  — exact full-causal hybrid: prefix partial + suffix partial
//                     on K/V subview + partition merge (WMMA arithmetic)
//   window_2048     — WMMA sliding window; approximate vs global full attention,
//                     exact for Glimmer's designated SWA layers.
//   scalar_window   — scalar sliding-window oracle, timed as a viable SWA path.
//
// Env: CTX (default 16384, must be >= 2049), WARMUPS (10), GROUPS (7), ITERS (20).
// Requires gfx1201. Frozen APIs used for the hybrid:
//   Gpu::attention_q8_0_flash_prefill_wmma_partial_gfx1201_bench
//   Gpu::attention_partition_merge_f32_bench

use rdna_compute::{DType, Gpu, GpuTensor};

const B: usize = 16;
const H: usize = 32;
const KVH: usize = 2;
const HD: usize = 128;
const WINDOW: usize = 2048;
const PARTIAL_STRIDE: usize = 2 + HD; // [m, l, N[HD]]
const TILE: usize = 128;
const Q8_BLOCK: usize = 34;
const BYTES_PER_POS: usize = KVH * (HD / 32) * Q8_BLOCK; // 272

const REL_L2_MAX: f32 = 5e-3;
const COS_MIN: f32 = 1.0 - 1e-5;
const MERGE_ATOL: f32 = 1e-6;
const MERGE_RTOL: f32 = 1e-5;

#[derive(Clone, Copy)]
enum Arm {
    ScalarFull,
    WmmaFull,
    LongspecExact,
    WmmaWindow2048,
    ScalarWindow2048,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::ScalarFull => "scalar_full",
            Arm::WmmaFull => "wmma_full",
            Arm::LongspecExact => "longspec_exact",
            Arm::WmmaWindow2048 => "wmma_window_2048",
            Arm::ScalarWindow2048 => "scalar_window_2048",
        }
    }

    /// Exact full-causal under the Q8 fixture, or approximate vs full attention.
    fn classification(self) -> &'static str {
        match self {
            Arm::ScalarFull => "exact_full_causal_f32_scalar",
            Arm::WmmaFull => "exact_full_causal_wmma_reduced_precision",
            Arm::LongspecExact => "exact_full_causal_hybrid_wmma_partial_merge",
            Arm::WmmaWindow2048 => "exact_native_swa_w2048_wmma",
            Arm::ScalarWindow2048 => "exact_native_swa_w2048_f32_scalar",
        }
    }

    fn all() -> [Arm; 5] {
        [
            Arm::ScalarFull,
            Arm::WmmaFull,
            Arm::LongspecExact,
            Arm::WmmaWindow2048,
            Arm::ScalarWindow2048,
        ]
    }
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits & 0x007F_FFFF) >> 13;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}

fn upload_i32(gpu: &Gpu, data: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.upload_raw(bytes, &[data.len()])
        .expect("upload i32 positions")
}

/// Deterministic nondegenerate Q8 fixture patterned after test_q8_flash_prefill.rs:
/// varied per-block scales/codes; V codes differ from K.
fn build_kv(ctx: usize) -> (Vec<u8>, Vec<u8>) {
    let cache_bytes = ctx * BYTES_PER_POS;
    let mut k = vec![0u8; cache_bytes];
    for (bi, blk) in k.chunks_mut(Q8_BLOCK).enumerate() {
        let scale = 0.02 + ((bi % 13) as f32) * 0.005;
        let h = half_from_f32(scale);
        blk[0] = (h & 0xFF) as u8;
        blk[1] = (h >> 8) as u8;
        for (j, b) in blk[2..].iter_mut().enumerate() {
            *b = (((bi * 31 + j * 17) % 251) as i32 - 125) as i8 as u8;
        }
    }
    let mut v = k.clone();
    for (i, b) in v.iter_mut().enumerate() {
        if i % Q8_BLOCK >= 2 {
            *b = (*b).wrapping_add(7);
        }
    }
    (k, v)
}

fn build_q() -> Vec<f32> {
    (0..B * H * HD)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01)
        .collect()
}

struct CorrMetrics {
    max_abs: f32,
    max_rel_l2: f32,
    min_cos: f32,
    nonfinite: usize,
    degenerate: usize,
    compared: usize,
}

fn compare_outputs(ref_o: &[f32], cand: &[f32]) -> CorrMetrics {
    assert_eq!(ref_o.len(), cand.len());
    assert_eq!(ref_o.len(), B * H * HD);
    let mut max_abs = 0.0f32;
    let mut max_rel_l2 = 0.0f32;
    let mut min_cos = 1.0f32;
    let mut nonfinite = 0usize;
    let mut degenerate = 0usize;
    let mut compared = 0usize;
    for v in cand {
        if !v.is_finite() {
            nonfinite += 1;
        }
    }
    for vec_i in 0..(B * H) {
        let s = vec_i * HD;
        let (mut dot, mut na, mut nb, mut nd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for d in 0..HD {
            let a = ref_o[s + d] as f64;
            let b = cand[s + d] as f64;
            max_abs = max_abs.max((a - b).abs() as f32);
            dot += a * b;
            na += a * a;
            nb += b * b;
            nd += (a - b) * (a - b);
        }
        if na > 0.0 && nb > 0.0 {
            min_cos = min_cos.min((dot / (na.sqrt() * nb.sqrt())) as f32);
            max_rel_l2 = max_rel_l2.max((nd.sqrt() / na.sqrt()) as f32);
            compared += 1;
        } else if na > 0.0 {
            // Reference has signal; candidate is all-zero — vacuous-pass trap.
            degenerate += 1;
        }
    }
    CorrMetrics {
        max_abs,
        max_rel_l2,
        min_cos,
        nonfinite,
        degenerate,
        compared,
    }
}

fn gate_wmma_like(label: &str, m: &CorrMetrics) {
    println!(
        "CORR arm={label} max_abs={:.3e} max_rel_l2={:.3e} min_cos={:.9} \
         nonfinite={} degenerate={} compared={}",
        m.max_abs, m.max_rel_l2, m.min_cos, m.nonfinite, m.degenerate, m.compared
    );
    assert!(
        m.nonfinite == 0,
        "{label}: {n} nonfinite outputs",
        n = m.nonfinite
    );
    assert!(
        m.degenerate == 0,
        "{label}: {n} degenerate all-zero vectors while reference has signal",
        n = m.degenerate
    );
    assert!(
        m.compared == B * H,
        "{label}: only {} of {} vectors comparable",
        m.compared,
        B * H
    );
    assert!(
        m.max_rel_l2 <= REL_L2_MAX,
        "{label}: rel_l2 {:.3e} > {REL_L2_MAX:.3e}",
        m.max_rel_l2
    );
    assert!(
        m.min_cos >= COS_MIN,
        "{label}: min_cos {:.9} < {COS_MIN:.9}",
        m.min_cos
    );
}

/// Stable two-partition merge in host F64, matching the GPU merge contract.
fn host_merge_partials(prefix: &[f32], suffix: &[f32]) -> Vec<f32> {
    let n = B * H;
    assert_eq!(prefix.len(), n * PARTIAL_STRIDE);
    assert_eq!(suffix.len(), n * PARTIAL_STRIDE);
    let mut out = vec![0.0f32; n * HD];
    for i in 0..n {
        let pb = i * PARTIAL_STRIDE;
        let ob = i * HD;
        let mp = prefix[pb] as f64;
        let lp = prefix[pb + 1] as f64;
        let ms = suffix[pb] as f64;
        let ls = suffix[pb + 1] as f64;
        let m = mp.max(ms);
        let wp = if lp > 0.0 { (mp - m).exp() } else { 0.0 };
        let ws = if ls > 0.0 { (ms - m).exp() } else { 0.0 };
        let den = wp * lp + ws * ls;
        if den == 0.0 {
            for d in 0..HD {
                out[ob + d] = 0.0;
            }
            continue;
        }
        for d in 0..HD {
            let np = prefix[pb + 2 + d] as f64;
            let ns = suffix[pb + 2 + d] as f64;
            out[ob + d] = ((wp * np + ws * ns) / den) as f32;
        }
    }
    out
}

fn merge_allclose(gpu_o: &[f32], host_o: &[f32]) -> (f32, f32, bool) {
    let mut max_abs = 0.0f32;
    let mut worst_ratio = 0.0f32;
    let mut ok = true;
    for (g, h) in gpu_o.iter().zip(host_o.iter()) {
        if !g.is_finite() || !h.is_finite() {
            ok = false;
            continue;
        }
        let abs = (g - h).abs();
        max_abs = max_abs.max(abs);
        let budget = MERGE_ATOL + MERGE_RTOL * h.abs();
        let ratio = abs / budget;
        worst_ratio = worst_ratio.max(ratio);
        if abs > budget {
            ok = false;
        }
    }
    (max_abs, worst_ratio, ok)
}

struct Scratch {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    /// Non-owning views into k/v at position P; kept alive for the process.
    k_suf: GpuTensor,
    v_suf: GpuTensor,
    pos_full: GpuTensor,
    pos_prefix: GpuTensor,
    pos_suffix: GpuTensor,
    scalar_partials: GpuTensor,
    prefix_partials: GpuTensor,
    suffix_partials: GpuTensor,
    out_scalar: GpuTensor,
    out_wmma: GpuTensor,
    out_longspec: GpuTensor,
    out_window: GpuTensor,
    out_scalar_window: GpuTensor,
}

impl Scratch {
    fn new(gpu: &mut Gpu, ctx: usize) -> Self {
        let p = ctx - B;
        assert!(ctx >= B + 1, "CTX must leave a non-empty prefix");
        assert!(
            ctx >= WINDOW + 1,
            "CTX={ctx} must be > WINDOW={WINDOW} so the window arm is non-vacuous"
        );

        let (k_host, v_host) = build_kv(ctx);
        let q_host = build_q();
        let cache_bytes = ctx * BYTES_PER_POS;

        let q = gpu
            .upload_f32(&q_host, &[B * H * HD])
            .expect("upload Q");
        let k = gpu
            .upload_raw(&k_host, &[cache_bytes])
            .expect("upload K");
        let v = gpu
            .upload_raw(&v_host, &[cache_bytes])
            .expect("upload V");

        // Raw dtype: sub_offset counts bytes (dtype.size()==1).
        let suf_bytes = B * BYTES_PER_POS;
        let k_suf = k.sub_offset(p * BYTES_PER_POS, suf_bytes);
        let v_suf = v.sub_offset(p * BYTES_PER_POS, suf_bytes);

        let pos_full_h: Vec<i32> = (0..B).map(|i| (p + i) as i32).collect();
        let pos_prefix_h: Vec<i32> = vec![(p as i32) - 1; B];
        let pos_suffix_h: Vec<i32> = (0..B as i32).collect();
        let pos_full = upload_i32(gpu, &pos_full_h);
        let pos_prefix = upload_i32(gpu, &pos_prefix_h);
        let pos_suffix = upload_i32(gpu, &pos_suffix_h);

        let max_tiles = ctx.div_ceil(TILE);
        let scalar_partials = gpu
            .zeros(&[B * H * max_tiles * PARTIAL_STRIDE], DType::F32)
            .expect("scalar partials");
        let prefix_partials = gpu
            .zeros(&[B * H * PARTIAL_STRIDE], DType::F32)
            .expect("prefix partials");
        let suffix_partials = gpu
            .zeros(&[B * H * PARTIAL_STRIDE], DType::F32)
            .expect("suffix partials");

        let out_scalar = gpu.zeros(&[B * H * HD], DType::F32).expect("out_scalar");
        let out_wmma = gpu.zeros(&[B * H * HD], DType::F32).expect("out_wmma");
        let out_longspec = gpu.zeros(&[B * H * HD], DType::F32).expect("out_longspec");
        let out_window = gpu.zeros(&[B * H * HD], DType::F32).expect("out_window");
        let out_scalar_window = gpu
            .zeros(&[B * H * HD], DType::F32)
            .expect("out_scalar_window");

        Scratch {
            q,
            k,
            v,
            k_suf,
            v_suf,
            pos_full,
            pos_prefix,
            pos_suffix,
            scalar_partials,
            prefix_partials,
            suffix_partials,
            out_scalar,
            out_wmma,
            out_longspec,
            out_window,
            out_scalar_window,
        }
    }


    fn launch(&self, gpu: &mut Gpu, arm: Arm, ctx: usize) {
        match arm {
            Arm::ScalarFull => {
                gpu.attention_flash_q8_0_batched_masked(
                    &self.q,
                    &self.k,
                    &self.v,
                    &self.out_scalar,
                    &self.pos_full,
                    H,
                    KVH,
                    HD,
                    ctx,
                    ctx,
                    B,
                    &self.scalar_partials,
                    None,
                    0,
                    0,
                )
                .expect("scalar_full");
            }
            Arm::WmmaFull => {
                gpu.attention_q8_0_flash_prefill_wmma(
                    &self.q,
                    &self.k,
                    &self.v,
                    &self.out_wmma,
                    &self.pos_full,
                    H,
                    KVH,
                    HD,
                    B,
                )
                .expect("wmma_full");
            }
            Arm::LongspecExact => {
                // Prefix over [0, P) via every query position = P-1.
                gpu.attention_q8_0_flash_prefill_wmma_partial_gfx1201_bench(
                    &self.q,
                    &self.k,
                    &self.v,
                    &self.prefix_partials,
                    &self.pos_prefix,
                    H,
                    KVH,
                    HD,
                    B,
                )
                .expect("longspec prefix partial");
                // Suffix over local [0,i] on K/V starting at absolute P.
                gpu.attention_q8_0_flash_prefill_wmma_partial_gfx1201_bench(
                    &self.q,
                    &self.k_suf,
                    &self.v_suf,
                    &self.suffix_partials,
                    &self.pos_suffix,
                    H,
                    KVH,
                    HD,
                    B,
                )
                .expect("longspec suffix partial");
                gpu.attention_partition_merge_f32_bench(
                    &self.prefix_partials,
                    &self.suffix_partials,
                    &self.out_longspec,
                    B,
                    H,
                    HD,
                )
                .expect("longspec merge");
            }
            Arm::WmmaWindow2048 => {
                let ok = gpu
                    .attention_q8_0_flash_prefill_wmma_swa(
                        &self.q,
                        &self.k,
                        &self.v,
                        &self.out_window,
                        &self.pos_full,
                        H,
                        KVH,
                        HD,
                        B,
                        WINDOW,
                    )
                    .expect("wmma_window_2048 launch");
                assert!(ok, "attention_q8_0_flash_prefill_wmma_swa returned false");
            }
            Arm::ScalarWindow2048 => self.launch_scalar_window(gpu, ctx),
        }
    }

    fn launch_scalar_window(&self, gpu: &mut Gpu, ctx: usize) {
        gpu.attention_flash_q8_0_batched_masked_windowed(
            &self.q,
            &self.k,
            &self.v,
            &self.out_scalar_window,
            &self.pos_full,
            H,
            KVH,
            HD,
            ctx,
            ctx,
            B,
            &self.scalar_partials,
            None,
            0,
            0,
            WINDOW as i32,
        )
        .expect("scalar window reference");
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
    sorted[idx]
}

fn summarize_us(samples_us: &mut [f64]) -> (f64, f64, f64) {
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        percentile_sorted(samples_us, 0.5),
        percentile_sorted(samples_us, 0.10),
        percentile_sorted(samples_us, 0.90),
    )
}

/// Latin-square-ish rotation of the five arms per group.
fn arm_order(group: usize) -> [Arm; 5] {
    let mut arms = Arm::all();
    let rotation = group % arms.len();
    arms.rotate_left(rotation);
    arms
}

fn print_settings(gpu: &Gpu, ctx: usize, warmups: usize, groups: usize, iters: usize) {
    println!(
        "SETTINGS arch={} gfx1201_required=1 B={B} H={H} KVH={KVH} HD={HD} WINDOW={WINDOW} \
         CTX={ctx} WARMUPS={warmups} GROUPS={groups} ITERS={iters} \
         BYTES_PER_POS={BYTES_PER_POS} PARTIAL_STRIDE={PARTIAL_STRIDE}",
        gpu.arch
    );
    println!(
        "CLASSIFICATION scalar_full=exact wmma_full=exact longspec_exact=exact \
         wmma_window_2048=exact_native_swa scalar_window_2048=exact_native_swa \
         (both window arms truncate Glimmer global/NoPE attention)"
    );
    // Surface common attention tuning knobs that affect WMMA arithmetic.
    for k in [
        "HIPFIRE_FLASH_PREFILL_FIXED_HD",
        "HIPFIRE_FLASH_PREFILL_PREFETCH_V",
        "HIPFIRE_FLASH_PREFILL_SPLITQ",
        "HIPFIRE_GLIMMER_SWA_PREFETCH_V",
        "HIP_VISIBLE_DEVICES",
    ] {
        match std::env::var(k) {
            Ok(v) => println!("ENV {k}={v}"),
            Err(_) => println!("ENV {k}=<unset>"),
        }
    }
}

fn main() {
    let ctx = env_usize("CTX", 16_384);
    let warmups = env_usize("WARMUPS", 10);
    let groups = env_usize("GROUPS", 7);
    let iters = env_usize("ITERS", 20);

    assert!(
        ctx >= 2049,
        "CTX={ctx} must be >= 2049 so prefix P=CTX-B is non-empty past the native window edge"
    );
    assert!(iters > 0 && groups > 0, "GROUPS/ITERS must be positive");

    let mut gpu = Gpu::init().expect("gpu init");
    print_settings(&gpu, ctx, warmups, groups, iters);

    if !gpu.arch_caps.has_wmma_w32_gfx12() || !gpu.arch.eq_ignore_ascii_case("gfx1201") {
        // has_wmma_w32_gfx12 is true for all RDNA4; still require the exact gfx1201 string
        // for headline results as specified.
        if !gpu.arch.eq_ignore_ascii_case("gfx1201") {
            panic!(
                "bench_glimmer_verify_attention requires gfx1201; got arch={}",
                gpu.arch
            );
        }
    }

    let p = ctx - B;
    println!(
        "GEOMETRY P={p} prefix_keys={p} suffix_keys_local=0..{} full_pos=P+i bytes_per_pos={BYTES_PER_POS}",
        B - 1
    );

    let s = Scratch::new(&mut gpu, ctx);

    // ── JIT every arm once (untimed) ──────────────────────────────────
    for arm in Arm::all() {
        s.launch(&mut gpu, arm, ctx);
    }
    s.launch_scalar_window(&mut gpu, ctx);
    gpu.hip.device_synchronize().expect("sync after jit");

    // ── Correctness (before any timing) ───────────────────────────────
    s.launch(&mut gpu, Arm::ScalarFull, ctx);
    s.launch(&mut gpu, Arm::WmmaFull, ctx);
    s.launch(&mut gpu, Arm::LongspecExact, ctx);
    s.launch(&mut gpu, Arm::WmmaWindow2048, ctx);
    s.launch_scalar_window(&mut gpu, ctx);
    gpu.hip.device_synchronize().expect("sync after corr launches");

    let ref_full = gpu.download_f32(&s.out_scalar).expect("dl scalar_full");
    let wmma = gpu.download_f32(&s.out_wmma).expect("dl wmma_full");
    let longspec = gpu.download_f32(&s.out_longspec).expect("dl longspec");
    let win = gpu.download_f32(&s.out_window).expect("dl window");
    let ref_win = gpu
        .download_f32(&s.out_scalar_window)
        .expect("dl scalar_window");

    // Non-vacuous fixture: reference must carry signal.
    let ref_energy: f64 = ref_full.iter().map(|x| (*x as f64).powi(2)).sum();
    assert!(
        ref_energy > 0.0 && ref_full.iter().all(|v| v.is_finite()),
        "scalar_full reference is vacuous/nonfinite (energy={ref_energy})"
    );
    let ref_win_energy: f64 = ref_win.iter().map(|x| (*x as f64).powi(2)).sum();
    assert!(
        ref_win_energy > 0.0 && ref_win.iter().all(|v| v.is_finite()),
        "scalar window reference is vacuous/nonfinite (energy={ref_win_energy})"
    );

    gate_wmma_like("wmma_full", &compare_outputs(&ref_full, &wmma));
    gate_wmma_like("longspec_exact", &compare_outputs(&ref_full, &longspec));
    gate_wmma_like("wmma_window_2048", &compare_outputs(&ref_win, &win));

    // Merge oracle: host F64 stable merge of downloaded partials vs GPU out.
    let pref_h = gpu
        .download_f32(&s.prefix_partials)
        .expect("dl prefix partials");
    let suf_h = gpu
        .download_f32(&s.suffix_partials)
        .expect("dl suffix partials");
    assert!(
        pref_h.iter().all(|v| v.is_finite()) && suf_h.iter().all(|v| v.is_finite()),
        "prefix/suffix partials contain nonfinite values"
    );
    let mut any_l = false;
    for i in 0..(B * H) {
        if pref_h[i * PARTIAL_STRIDE + 1] > 0.0 || suf_h[i * PARTIAL_STRIDE + 1] > 0.0 {
            any_l = true;
            break;
        }
    }
    assert!(any_l, "all prefix/suffix l values are zero — vacuous partials");

    let host_merged = host_merge_partials(&pref_h, &suf_h);
    let (m_abs, m_ratio, m_ok) = merge_allclose(&longspec, &host_merged);
    println!(
        "CORR merge_oracle max_abs={m_abs:.3e} worst_tol_ratio={m_ratio:.3} \
         ATOL={MERGE_ATOL} RTOL={MERGE_RTOL} ok={m_ok}"
    );
    assert!(
        m_ok,
        "GPU merge diverges from host F64 stable merge (max_abs={m_abs:.3e} ratio={m_ratio:.3})"
    );

    println!(
        "CORRECTNESS_PASS CTX={ctx} P={p} exact_arms=scalar_full,wmma_full,longspec_exact \
         native_swa_arms=wmma_window_2048,scalar_window_2048 \
         (window arms gated against each other, not against global full attention)"
    );

    // ── Warmups (untimed, after correctness) ──────────────────────────
    for arm in Arm::all() {
        for _ in 0..warmups {
            s.launch(&mut gpu, arm, ctx);
        }
    }
    gpu.hip.device_synchronize().expect("sync after warmups");

    // ── Timing: GROUPS × ITERS complete windows; one sync per group ───
    // Rotate arm order each group. No alloc/download/JIT inside timed regions.
    let n_groups = groups;
    let mut samples: Vec<Vec<f64>> = vec![vec![0.0; n_groups]; 5];

    for g in 0..n_groups {
        let order = arm_order(g);
        for arm in order {
            gpu.hip.device_synchronize().expect("pre-group sync");
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                s.launch(&mut gpu, arm, ctx);
            }
            gpu.hip.device_synchronize().expect("post-group sync");
            let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
            let us_per_window = elapsed_us / iters as f64;
            let ai = match arm {
                Arm::ScalarFull => 0,
                Arm::WmmaFull => 1,
                Arm::LongspecExact => 2,
                Arm::WmmaWindow2048 => 3,
                Arm::ScalarWindow2048 => 4,
            };
            samples[ai][g] = us_per_window;
            println!(
                "SAMPLE arm={} class={} CTX={ctx} group={g} iters={iters} \
                 us_per_window={us_per_window:.3} us_per_query_row={:.3}",
                arm.name(),
                arm.classification(),
                us_per_window / B as f64
            );
        }
    }

    // Medians first so speedups are well-defined for every arm.
    let mut medians = [0.0f64; 5];
    let mut p10s = [0.0f64; 5];
    let mut p90s = [0.0f64; 5];
    for ai in 0..5 {
        let mut srow = samples[ai].clone();
        let (med, p10, p90) = summarize_us(&mut srow);
        medians[ai] = med;
        p10s[ai] = p10;
        p90s[ai] = p90;
    }

    for (ai, arm) in Arm::all().into_iter().enumerate() {
        let med = medians[ai];
        let p10 = p10s[ai];
        let p90 = p90s[ai];
        let us_q = med / B as f64;
        let sp_scalar = medians[0] / med;
        let sp_wmma = medians[1] / med;
        for g in 0..n_groups {
            println!(
                "RESULT kind=sample arm={} class={} CTX={ctx} group={g} \
                 us_per_window={:.6} us_per_query_row={:.6}",
                arm.name(),
                arm.classification(),
                samples[ai][g],
                samples[ai][g] / B as f64
            );
        }
        println!(
            "RESULT kind=summary arm={} class={} CTX={ctx} groups={n_groups} iters={iters} \
             median_us_per_window={med:.6} p10_us_per_window={p10:.6} p90_us_per_window={p90:.6} \
             median_us_per_query_row={us_q:.6} speedup_vs_scalar_full={sp_scalar:.4} \
             speedup_vs_wmma_full={sp_wmma:.4}",
            arm.name(),
            arm.classification()
        );
        println!(
            "SUMMARY arm={} class={} CTX={ctx}: median={med:.1} us/window \
             ({us_q:.2} us/query-row) p10={p10:.1} p90={p90:.1} \
             vs_scalar={sp_scalar:.3}x vs_wmma={sp_wmma:.3}x",
            arm.name(),
            arm.classification()
        );
    }

    // Attention-only estimates for Glimmer's native 39 SWA + 13 global mix.
    // Current dispatch uses WMMA for both; the candidate uses scalar flash for
    // both. Neither estimate is an end-to-end model measurement.
    let current_mix = 39.0 * medians[3] + 13.0 * medians[1];
    let scalar_mix = 39.0 * medians[4] + 13.0 * medians[0];
    println!(
        "RESULT kind=layer_mix_estimate note=attention_only_not_e2e \
         formula_current=39*T_wmma_window_2048+13*T_wmma_full \
         formula_candidate=39*T_scalar_window_2048+13*T_scalar_full CTX={ctx} \
         current_mix_us={current_mix:.3} scalar_mix_us={scalar_mix:.3} \
         speedup_scalar_mix_vs_current={:.4}",
        current_mix / scalar_mix
    );
    println!(
        "NOTE both W2048 arms are exact for Glimmer's native SWA layers but \
         must not replace the 13 global/NoPE layers."
    );
    println!(
        "NOTE longspec_exact times the full three-launch hybrid chain \
         (prefix_partial + suffix_partial + partition_merge)."
    );
    println!("DONE CTX={ctx} arch={}", gpu.arch);

    // Keep non-owning suffix views and owners live through the end; drop order
    // does not free views (DeviceBuffer::from_raw has no Drop free).
    let _keep = (&s.k_suf, &s.v_suf, &s.k, &s.v);
    let _ = _keep;
}
