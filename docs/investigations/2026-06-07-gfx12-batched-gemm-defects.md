# gfx12 (RDNA4) batched-GEMM defects — diagnosis (2026-06-07)

Box: `hiptrx`, GPU3 = gfx1201 / RDNA4 (AMD Radeon AI PRO R9700, 34.2 GB, HIP 7.2).
Worktree: `~/hipfire-gemma4`, branch `gemma4-rz`, base `a8c8bff3`.
Method: DIAGNOSE-ONLY. All probe edits reverted; tree clean except this doc.

Two pre-existing gfx12 batched-GEMM defects were investigated. **Neither root
cause is in the gfx12 WMMA kernel math.** Both are host-side dispatch / scratch
bugs that the named kernels merely sit downstream of.

| Bug | Symptom in brief | ACTUAL root cause | Kernel innocent? |
|-----|------------------|-------------------|------------------|
| A | `gemm_hfq4g256_residual_wmma_gfx12` illegal access (HipError 700) at "B>=5" | **Undersized `x_rot` scratch** in gemma4 `forward_batch` -> OOB write in the **o_proj** FWHT rotation on FULL attention layers (k=8192 > dim=3840) | YES — the GEMM kernel is correct |
| B | `gemm_q8_0_wmma_gfx12` wrong at m=vocab=262144 -> lm_head attractor | **Stale pointer-keyed fp16-x cache** (`ensure_fp16_x`): the WMMA path reuses a prior call's FP16 activations when the source x pointer is reused with new contents | YES — the GEMM kernel is numerically correct at m=262144 |

They DO NOT share a root cause, but they share a **theme**: the gfx12 WMMA
batched kernels are correct; the breakage is in the Rust host code that feeds
them (scratch sizing for A; FP16-conversion caching for B). Both are masked by
the corresponding scalar fallback (which is why `HIPFIRE_Q8_BATCHED_LEGACY=1`
"fixes" B, and why small B "fixes" A).

---

## Bug A — `gemm_hfq4g256_residual_wmma_gfx12` illegal access

### Repro

```
cd ~/hipfire-gemma4
cargo build --release -p hipfire-arch-gemma4 --features deltanet --example verify_batch_gemma4
HIP_VISIBLE_DEVICES=3 ./target/release/examples/verify_batch_gemma4 \
    --model ~/gemma4-12b-mq4attn.hfq --bs 4,5,6,8,16,32,64
```

Original observation (multi-B in one process): B=4 PASS, then crash attributed to B=5:
```
B=4   cosine=0.9995  ... => PASS
thread 'main' panicked: forward_batch: "gemma4 forward_batch download logits:
  HipError { code: 700, "hipMemcpy D2H: an illegal memory access was encountered" }"
```

### The "B>=5 threshold" is an artifact — true behaviour is non-monotonic

Isolated single-B runs (one process each) do NOT show a clean threshold:

| B (isolated) | result |
|---|---|
| 5, 6, 8, 16 | PASS |
| 24 | FAIL (cosine 0.9956, argmax/KV mismatch — corruption, not crash) |
| 31 | PASS |
| 32 | CRASH (HipError 700) |
| 33, 63 | CRASH |
| 48 | FAIL (cosine 0.37 — garbage logits) |
| 64 | PASS |

Non-monotonic fault + differing crash sites = **heap corruption that propagates**;
the observed symptom depends on the allocator layout for that particular B. The
multi-B "B=4 PASS then B=5 crash" was the async error from a *later* large-B
iteration surfacing at an *earlier* sync point.

### Localization — it is NOT the GEMM kernel

Under `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` the async error resolves to
a deterministic site (no longer the lazy D2H):

```
B=32: "gemma4 batch attn kv write k: HipError 700 hipModuleLaunchKernel ..."   (kv_cache_write_q8_0_batched)
B=33: "gemma4 batch o_proj rotate:  HipError 700 ..."                          (rotate_x_mq_batched_for)
B=63: "gemma4 batch o_proj rotate:  HipError 700 ..."
```

The faulting kernel is wherever the stomped pointer is next dereferenced — the
o_proj-rotate site points straight at the cause.

### Root cause — undersized `x_rot` scratch, OOB in o_proj rotation

`crates/hipfire-arch-gemma4/src/forward.rs:858`
```rust
let x_rot = alloc(gpu, b * dim, "x_rot")?; // FWHT scratch (MQ4 proj path)
```
`x_rot` is sized `b * dim` = `b * 3840`. It is shared by ALL projections in the
layer via `proj_gemm_batched` (forward.rs:746-768), which rotates `b * w.k`
floats into it:
```rust
rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)   // writes b*w.k floats into x_rot
```
For the o_proj on a FULL attention layer, `w.k = q_dim = n_heads * full_head_dim`.
From config: `n_heads = 16`, `full_head_dim = 512` (`crates/hipfire-arch-gemma4/src/config.rs`,
`max_q_dim()` at line 244). o_proj is loaded with `k = q_dim`
(`crates/hipfire-arch-gemma4/src/gemma4.rs:350-356` sliding, `:454-460` full):
```
o_proj: load_wt(hfq, gpu, ".self_attn.o_proj.weight", /*m*/ dim, /*k*/ q_dim)
```
So for FULL layers `w.k = 16 * 512 = 8192`, but `x_rot` holds only `dim = 3840`.
The rotation writes `b * 8192` floats into a `b * 3840` buffer ->
**OOB by `b * (8192-3840) = b*4352` floats = b*17408 bytes per call**, stomping
whatever allocation the bump-allocator placed after `x_rot` (KV cache, mask,
q/k/v scratch — layout shifts with B, hence non-monotonic).

(Sliding layers use `sliding_head_dim = 256` -> o_proj k = 16*256 = 4096, also
> 3840, so sliding layers ALSO overflow, just by a smaller margin: b*256 floats.)

The gemma4 model has 8 full + 40 sliding layers (dim=3840, vocab=262144). The
FFN scratch `ffn_rot` (`b * ffn_hd`, forward.rs:869) is correctly sized because
every FFN rotation k (`dim` for gate/up, `hidden_dim` for down) is <= ffn_hd.
Only `x_rot` is undersized, and only the o_proj exceeds dim.

### Proof

Probe: enlarge `x_rot` to `b * max_q.max(dim)` (max_q = 8192), rebuild, rerun.
Result: **every HipError 700 illegal-access crash disappears** for B=32/33/48/63;
remaining FAILs become purely numerical (cosine 0.96-0.998, the documented MQ4
fp16-WMMA argmax-tie drift at large B), with ZERO crashes across repeated
`--bs 4,5,6,8,16,32,64` runs. Probe reverted after confirmation.

### Fix plan (Bug A)

In `crates/hipfire-arch-gemma4/src/forward.rs` (`forward_batch_spec`):
```rust
// was: let x_rot = alloc(gpu, b * dim, "x_rot")?;
let x_rot = alloc(gpu, b * cfg.max_q_dim().max(dim), "x_rot")?;
```
`max_q_dim()` (= max(n_heads*sliding_hd, n_heads*full_hd) = 8192) is the largest
k any projection rotates into `x_rot` (the o_proj). Using `.max(dim)` keeps it
>= the q/k/v/gate/up case (k=dim). This is correct on ALL arches — the bug is
arch-independent; gfx12 just happened to take the WMMA proj path and grow the OOB
into a fault sooner. The eager single-token path never hits it because it rotates
per-projection into per-projection scratch.

Validation:
1. `verify_batch_gemma4 --bs 4,5,6,8,16,32,64` on gfx1201 -> no HipError 700.
2. Larger-B numerical FAILs are expected (MQ4 fp16-WMMA tie drift) and orthogonal
   to this fix; confirm they are cosine ~0.99+ argmax-tie flips, not crashes.
3. `./scripts/coherence-gate.sh` (gemma4 arch) + gemma4 spec-decode e2e
   (`infer_gemma4_spec`) byte-identical / coherent.
4. Cross-check the eager path is byte-inert (this only touches forward_batch).

Confidence: **very high.** Exact dimensions cited; oversizing the buffer
deterministically removes 100% of the illegal-access crashes.

---

## Bug B — `gemm_q8_0_wmma_gfx12` wrong as lm_head (single-token attractor)

### Microbench (standalone) does NOT reproduce a kernel-math error

A standalone WMMA-vs-scalar microbench (Appendix) builds a Q8 weight [m x k=3840]
and B activation rows, runs `gemm_q8_0_wmma` (WMMA path) vs `gemm_q8_0_batched`
(scalar reference), and sweeps m in {512,4096,32768,65536,131072,262144} at
B in {1,2,3,4,8}:

```
m= 262144 N=1  max_abs_diff=9.80e-2 max_rel=2.107e-2 nan=0 bad/262144=0 (0.00%)  bad_tiles=0/16384
m= 262144 N=8  max_abs_diff=9.80e-2 max_rel=1.840e-2 nan=0 bad/2097152=0 (0.00%) bad_tiles=0/16384
```

At EVERY m up to 262144 and every B: max relative error ~2% (pure fp16-WMMA
accumulation drift vs the fp32 scalar), ZERO out-of-tolerance elements, no NaN,
no tile-region corruption. Escalating activation magnitude (XSCALE up to 300,
max|x|=300) likewise stays ~2% rel with zero bad elements — **no fp16 overflow,
no large-m corruption.** The gfx12 WMMA kernel is numerically correct at vocab m.

Module-cache collision (the known gfx12 trap) was ruled out: `gemm_q8_0_wmma_gfx12`
and `gemm_q8_0_residual_wmma_gfx12` use DISTINCT module names
(`crates/rdna-compute/src/gemm.rs:16857` vs `:17603`), and their SRC constants
`include_str!` the correct distinct .hip files (`kernels.rs:1465` and `:2283`).

### Root cause — stale pointer-keyed FP16 activation cache

`gemm_q8_0_wmma` converts x to FP16 via `ensure_fp16_x`
(`crates/rdna-compute/src/gemm.rs:16862`):
```rust
let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
```
`ensure_fp16_x` (`crates/rdna-compute/src/scratch.rs:302-376`) caches the
FP32->FP16 conversion keyed on the SOURCE POINTER and SKIPS reconversion when the
pointer matches the last converted source:
```rust
let must_convert = capture_mode || self.fp16_x_source_ptr != src_ptr;   // scratch.rs:336
```
When the lm_head / prefill projections reuse the SAME x buffer pointer across
calls with NEW contents — exactly what llama prefill does, e.g. the Q8 q/k/v/
gate/up projections all read `pbs.x_rot_batch`
(`crates/hipfire-runtime/src/llama.rs:2079,2087,2095,2406,2503,2511`), and across
layers that buffer is refilled with new activations under the same pointer — the
WMMA path reads STALE fp16 from a prior call -> catastrophically wrong matmul ->
garbage logits -> single-token attractor when this is the lm_head.

The scalar reference `gemm_q8_0_batched` reads x in FP32 directly with no cache,
so it is always correct — which is precisely why `HIPFIRE_Q8_BATCHED_LEGACY=1`
(routing to scalar, gemm.rs:16812-16816) masks the bug, and why a synthetic
microbench with distinct per-call buffers cannot reproduce it.

The existing HFQ4 lmhead path already knows about this hazard and stomps the
cache pointer before dispatch (`crates/rdna-compute/src/gemm.rs:14507`,
`self.scratch.fp16_x_source_ptr = std::ptr::null_mut();` with a long comment at
:14480-14488 describing exactly this DFlash same-pointer-new-data trap). The Q8
WMMA path (`gemm_q8_0_wmma` / `gemm_q8_0_batched_chunked`) was NOT given the same
guard.

### Proof — stale-cache probe (decisive)

Allocate ONE x buffer; run WMMA on data A; `memcpy_htod` NEW data B into the
SAME pointer; run WMMA again; compare to scalar-on-B:
```
STALE-PROBE: cos(call2_wmma, scalar_dataB)  = -0.225469   <- WMMA on B is WRONG
STALE-PROBE: cos(call2_wmma, call1_dataA)   =  1.000000   <- WMMA on B == data-A result (STALE)
STALE-PROBE: cos(scalar_dataB, call1_dataA) = -0.225469   <- A and B are genuinely different
```
The WMMA second call returns EXACTLY the first call's (data-A) result: it never
reconverted. `m=vocab` is incidental — the bug fires whenever the x pointer is
reused, which the real lm_head/prefill always does.

Fix-direction probe: insert `self.scratch.fp16_x_source_ptr = std::ptr::null_mut();`
immediately before the `ensure_fp16_x` call in `gemm_q8_0_wmma`:
```
STALE-PROBE: cos(call2_wmma, scalar_dataB)  = 1.000000   <- fixed
STALE-PROBE: cos(call2_wmma, call1_dataA)   = -0.225467  <- no longer stale
```
Probe reverted after confirmation.

### Fix plan (Bug B)

In `crates/rdna-compute/src/gemm.rs`, `gemm_q8_0_wmma` (~line 16857), before the
`ensure_fp16_x` call:
```rust
// gfx12 WMMA Q8 path reads FP16-converted x; the pointer-keyed cache in
// ensure_fp16_x would skip reconversion when callers reuse the same x buffer
// with new contents (prefill x_rot_batch across layers; DFlash hidden reuse),
// serving STALE activations. Mirror gemm_hfq4g256_batched_lmhead (gemm.rs:14507).
self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
let x_f16_ptr = self.ensure_fp16_x(x, batch_size * k)?;
```

Considerations:
- This forces a reconversion on every `gemm_q8_0_wmma` call. That is correct and
  cheap (the convert kernel is bandwidth-bound on b*k elems, negligible vs the
  GEMM). The cache's intended win — skipping reconversion for q/k/v that share
  one input within a layer — is defeated for back-to-back same-pointer calls, but
  those are precisely the ones currently WRONG, so safety wins. (A more surgical
  option: make the cache content-aware via a generation counter, but the
  pointer-stomp matches the established HFQ4 pattern and is the minimal correct
  fix.)
- Audit the OTHER WMMA Q8 callers that go through `ensure_fp16_x` for the same
  hazard if they can be fed reused-pointer/new-content x: `gemm_qkv_q8_0_wmma`,
  `gemm_gate_up_q8_0_wmma`, `gemm_q8_0_residual_wmma(_gfx12)`, `gemm_qkvza_q8_0_wmma`.
  Within one layer the q/k/v share input AND contents (cache is correct there);
  the danger is strictly cross-call same-pointer-new-content. The lm_head and the
  cross-layer prefill reuse of `x_rot_batch` are the confirmed live triggers.
  Recommend either a single shared guard or making `ensure_fp16_x` content-aware
  rather than pointer-aware, but the minimal ship is the Q8-WMMA stomp above.

Validation:
1. Stale-cache probe (Appendix) -> `cos(call2_wmma, scalar_dataB)` == 1.0.
2. Q8 lm_head e2e on a Q8-lm_head model on gfx1201 with the WMMA path ON
   (`HIPFIRE_Q8_BATCHED_LEGACY` unset) -> coherent text, NO single-token attractor.
3. `./scripts/coherence-gate.sh` + `./scripts/coherence-gate-dflash.sh`
   (spec-decode lm_head) with the WMMA path active.
4. Confirm `HIPFIRE_Q8_BATCHED_LEGACY=1` (scalar) and the fixed WMMA path now
   produce argmax-identical logits on a real forward.

Confidence: **very high.** Direct cause/effect proof (stale read = exact data-A
result; pointer-stomp = exact scalar result).

---

## Shared cause?

**No shared root cause.** Bug A = undersized FWHT scratch buffer (`x_rot`,
arch-independent, gemma4-specific). Bug B = stale pointer-keyed FP16 cache
(`ensure_fp16_x`, affects any reused-pointer caller of the gfx12 Q8 WMMA path).

**Shared theme:** in BOTH, the gfx12 WMMA *kernel* is correct and the defect is
in the host-side Rust feeding it (scratch sizing / FP16 conversion caching). The
brief's kernel attributions (`gemm_hfq4g256_residual_wmma_gfx12`,
`gemm_q8_0_wmma_gfx12`) are innocent downstream consumers, not the bugs. The
"B>=5" and "wrong at m=vocab" framings are emergent symptoms (heap-layout-
dependent fault location; pointer-reuse in the lm_head), not the mechanisms.

Both fixes are 1-2 lines, behavior-preserving on the correct paths, and
arch-portable. They are independent and can land separately.

(Note: kernel occupancy/.hsaco metadata was not pulled — neither root cause is
register/LDS/occupancy-related; both are host-side and proven by direct
input/output comparison.)

---

## Appendix — Bug B microbench source

Saved during diagnosis as `crates/rdna-compute/examples/bug_b_q8_wmma_largem.rs`
(removed to keep the tree clean — recreate to reproduce). Run:

```
cargo build --release -p rdna-compute --example bug_b_q8_wmma_largem
HIP_VISIBLE_DEVICES=3 ./target/release/examples/bug_b_q8_wmma_largem            # sweep
HIP_VISIBLE_DEVICES=3 STALE_PROBE=1 ./target/release/examples/bug_b_q8_wmma_largem   # decisive probe
HIP_VISIBLE_DEVICES=3 XSCALE=300 ./target/release/examples/bug_b_q8_wmma_largem      # fp16-overflow probe
```

Key elements: builds the Q8 weight via `common/q8_test_utils::synth_q8`; WMMA
path = `gpu.gemm_q8_0_wmma(...)` direct; scalar reference = `gpu.gemm_q8_0_batched(...)`
direct (bypasses `gemm_q8_0_batched_chunked`'s RDNA4 auto-route to WMMA, and
avoids the `HIPFIRE_Q8_BATCHED_LEGACY` OnceLock which cannot be toggled
intra-process). `stale_cache_probe()` reuses one x pointer across two WMMA calls
with a `memcpy_htod` of fresh contents in between.

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Bug B diagnosis microbench: gemm_q8_0_wmma (gfx12 WMMA) vs gemm_q8_0_batched
// (scalar reference). Sweep m (vocab-like) at B; plus a stale-fp16-cache probe.

use rdna_compute::Gpu;

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::synth_q8;

fn synth_x(i: usize) -> f32 {
    let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
    (v * 1e-9) % 2.0 - 1.0
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== bug_b_q8_wmma_largem ===  arch = {arch}");
    if !arch.starts_with("gfx12") {
        eprintln!("SKIPPED: needs gfx12, got {arch}");
        std::process::exit(0);
    }

    let k = 3840usize;
    let ms: Vec<usize> = vec![512, 4096, 32768, 65536, 131072, 262144];
    let batches: Vec<usize> = vec![1, 2, 3, 4, 8];

    let max_b = *batches.iter().max().unwrap();
    let xscale: f32 = std::env::var("XSCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let x_host: Vec<f32> = (0..max_b * k).map(|i| synth_x(i) * xscale).collect();
    eprintln!("XSCALE={xscale}  max|x|={:.3}", x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max));
    let d_x = gpu.upload_f32(&x_host, &[max_b * k]).unwrap();

    for &m in &ms {
        let w = synth_q8(m, k, 0xA1B2C3D4);
        let d_a = gpu.upload_raw(&w, &[w.len()]).unwrap();
        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);
            let d_y_wmma = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
            gpu.gemm_q8_0_wmma(&d_a, &x_n, &d_y_wmma, m, k, n).unwrap();
            let d_y_ref = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
            gpu.gemm_q8_0_batched(&d_a, &x_n, &d_y_ref, m, k, n).unwrap();
            let yw = gpu.download_f32(&d_y_wmma).unwrap();
            let yr = gpu.download_f32(&d_y_ref).unwrap();
            // per-(row,col) stats; layout Y[n,m] row-major: y[col*m + row].
            let (mut max_abs_diff, mut max_rel) = (0.0f64, 0.0f64);
            let (mut n_nan, mut n_big) = (0usize, 0usize);
            let ref_max = yr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let thr = (ref_max * 0.01).max(1e-3);
            for col in 0..n {
                for row in 0..m {
                    let idx = col * m + row;
                    let (a, b) = (yw[idx], yr[idx]);
                    if a.is_nan() || b.is_nan() { n_nan += 1; }
                    let d = (a - b).abs() as f64;
                    if d > max_abs_diff { max_abs_diff = d; }
                    if b.abs() > thr { let r = d / b.abs() as f64; if r > max_rel { max_rel = r; } }
                    if d > thr as f64 { n_big += 1; }
                }
            }
            eprintln!("m={m:>7} N={n}  max_abs_diff={max_abs_diff:.4e} max_rel={max_rel:.3e} nan={n_nan} bad={n_big}");
        }
    }
    eprintln!("=== done ===");
    if std::env::var("STALE_PROBE").is_ok() { stale_cache_probe(); }
}

#[allow(dead_code)]
fn stale_cache_probe() {
    use rdna_compute::Gpu;
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch.starts_with("gfx12") { return; }
    let (m, k, n) = (4096usize, 3840usize, 4usize);
    let w = synth_q8(m, k, 0x1234);
    let d_a = gpu.upload_raw(&w, &[w.len()]).unwrap();
    // persistent x buffer (one pointer, reused — mimics pbs.x_rot_batch)
    let xa: Vec<f32> = (0..n*k).map(synth_x).collect();
    let d_x = gpu.upload_f32(&xa, &[n*k]).unwrap();
    let y1 = gpu.zeros(&[n*m], rdna_compute::DType::F32).unwrap();
    gpu.gemm_q8_0_wmma(&d_a, &d_x, &y1, m, k, n).unwrap();              // call 1, data A
    let xb: Vec<f32> = (0..n*k).map(|i| synth_x(i + 777) * 0.5).collect();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(xb.as_ptr() as *const u8, xb.len()*4) };
    gpu.hip.memcpy_htod(&d_x.buf, bytes).unwrap();                     // mutate SAME ptr -> data B
    let y2w = gpu.zeros(&[n*m], rdna_compute::DType::F32).unwrap();
    gpu.gemm_q8_0_wmma(&d_a, &d_x, &y2w, m, k, n).unwrap();            // call 2, WMMA (stale?)
    let y2r = gpu.zeros(&[n*m], rdna_compute::DType::F32).unwrap();
    gpu.gemm_q8_0_batched(&d_a, &d_x, &y2r, m, k, n).unwrap();         // ref on data B (fp32, fresh)
    let v1 = gpu.download_f32(&y1).unwrap();
    let v2w = gpu.download_f32(&y2w).unwrap();
    let v2r = gpu.download_f32(&y2r).unwrap();
    let cos = |a:&[f32],b:&[f32]| { let (mut d,mut na,mut nb)=(0f64,0f64,0f64);
        for (x,y) in a.iter().zip(b){d+=*x as f64* *y as f64;na+=(*x as f64).powi(2);nb+=(*y as f64).powi(2);}
        d/(na.sqrt()*nb.sqrt()+1e-30) };
    eprintln!("STALE-PROBE: cos(call2_wmma, scalar_dataB)  = {:.6}", cos(&v2w,&v2r));
    eprintln!("STALE-PROBE: cos(call2_wmma, call1_dataA)   = {:.6}", cos(&v2w,&v1));
    eprintln!("STALE-PROBE: cos(scalar_dataB, call1_dataA) = {:.6}", cos(&v2r,&v1));
}
```
