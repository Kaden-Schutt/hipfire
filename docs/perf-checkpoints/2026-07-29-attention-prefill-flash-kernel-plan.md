# Query-tiled Q8 Flash Prefill Attention — Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-workgroup-per-query Q8 prefill attention with a query-tiled FlashAttention-style kernel whose LDS is independent of context length.

**Architecture:** A new HIP kernel `attention_q8_0_flash_prefill` stages a tile of `BR` query rows in LDS once, then loops over `BC`-key tiles of the Q8_0 K/V cache staged in LDS in native block form, combining them with online softmax (running max/sum) and a register-resident output accumulator. K/V is read once per `BR` queries instead of once per query. Because LDS depends only on `BR`/`BC`, the kernel serves every context length and the 8192 crossover becomes unnecessary.

**Tech Stack:** HIP / ROCm 7.2.2, Rust, gfx1151 (RDNA3.5, wave32, 64 KB LDS/workgroup), Q8_0 KV cache (34 B per 32 dims: fp16 scale + 32 int8).

## Global Constraints

- Target arch: `gfx1151`. Wave size 32. LDS budget 64 KB per workgroup.
- Model shape under test: `n_heads=8, n_kv_heads=2 (kv_group=4), head_dim=256`.
- KV cache layout: `[position][n_kv_heads * (head_dim/32)][34 bytes]`, block = fp16 scale (2 B) + 32 int8 codes.
- Correctness bar is **numerically equivalent, not bit-identical**: max relative error ≤ 1e-4 where `|ref| > 1e-3`, max absolute error ≤ 1e-5 otherwise, cosine similarity ≥ 1 − 1e-6 per (query, head) output vector.
- Scope: causal, **non-tree** (`tree_bias == nullptr`) and **non-windowed** (`window <= 0`) prefill only. Decode (`batch_size == 1`) untouched.
- Benchmark prompts must use real in-distribution prose. Random-word filler makes the model degenerate and fabricates false corruption findings.
- Every task ends with a commit. Branch: `perf/flash-prefill-attention`.
- Do not modify `crates/hipfire-dispatch/src/families/attention.rs:1240` (`Q8_BATCHED_LDS_CROSSOVER`) until Task 6; the new kernel lands behind an env opt-in first.

---

### Task 1: Single-tile kernel, launcher, and correctness harness

Builds the kernel restricted to one K/V tile (`seq_len <= BC`). This isolates the Q8 dequant, GQA mapping, causal mask and output write before any online-softmax bookkeeping exists.

**Files:**
- Create: `kernels/src/attention_q8_0_flash_prefill.hip`
- Modify: `crates/rdna-compute/src/kernels.rs` (add `*_SRC` const beside `ATTENTION_Q8_0_KV_BATCHED_SRC` at line 3827)
- Modify: `crates/rdna-compute/src/attention.rs` (add launcher method beside `attention_q8_0_kv_batched_masked` at line 1717)
- Create: `crates/rdna-compute/examples/test_q8_flash_prefill.rs`

**Interfaces:**
- Consumes: `Gpu::ensure_kernel(module_name, source, func_name)`, `Gpu::launch_maybe_blob(func_name, grid, block, shared_mem, params, blob_builder)`, `Gpu::upload_f32`, `Gpu::upload_raw`, `Gpu::zeros`, `Gpu::download_f32`, `Gpu::attention_q8_0_kv_batched_masked` (the reference).
- Produces: `Gpu::attention_q8_0_flash_prefill(q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor, out: &GpuTensor, positions: &GpuTensor, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_ctx_len: usize, batch_size: usize, br: usize, bc: usize) -> HipResult<()>`. Later tasks call exactly this signature; the 9th argument is named `max_ctx_len` (unused for cache indexing, kept for call-site parity with the reference launcher).

- [ ] **Step 1: Write the failing test**

Create `crates/rdna-compute/examples/test_q8_flash_prefill.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// Correctness gate for attention_q8_0_flash_prefill vs attention_q8_0_kv_batched.
// Env: NH, NKV, HD, N (query rows), CTX (max_ctx_len), BR, BC.

use rdna_compute::{DType, Gpu};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let nh = env_usize("NH", 8);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let n = env_usize("N", 16);
    let ctx = env_usize("CTX", 32);
    let br = env_usize("BR", 16);
    let bc = env_usize("BC", 32);
    let mut gpu = Gpu::init().expect("gpu init");

    let bph = hd / 32;
    let bytes_per_pos = nkv * bph * 34;
    let cache_bytes = ctx * bytes_per_pos;

    // Deterministic pseudo-random KV: varied scales and codes so a wrong
    // dequant, wrong block stride or wrong GQA head cannot pass by symmetry.
    let mut kv = vec![0u8; cache_bytes];
    for (bi, blk) in kv.chunks_mut(34).enumerate() {
        let scale: f32 = 0.02 + ((bi % 13) as f32) * 0.005;
        let h = half_from_f32(scale);
        blk[0] = (h & 0xFF) as u8;
        blk[1] = (h >> 8) as u8;
        for (j, b) in blk[2..].iter_mut().enumerate() {
            *b = (((bi * 31 + j * 17) % 251) as i32 - 125) as i8 as u8;
        }
    }
    let k_cache = gpu.upload_raw(&kv, &[cache_bytes]).expect("k upload");
    let mut kv2 = kv.clone();
    for (i, b) in kv2.iter_mut().enumerate() { if i % 34 >= 2 { *b = (*b).wrapping_add(7); } }
    let v_cache = gpu.upload_raw(&kv2, &[cache_bytes]).expect("v upload");

    let q_data: Vec<f32> = (0..n * nh * hd)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01)
        .collect();
    let q = gpu.upload_f32(&q_data, &[n * nh * hd]).expect("q upload");

    // positions[b] = ctx - n + b : queries sit at the tail of the context.
    let pos_data: Vec<i32> = (0..n).map(|b| (ctx - n + b) as i32).collect();
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, n * 4) };
    let positions = gpu.upload_raw(pos_bytes, &[n]).expect("pos upload");

    let out_ref = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_ref");
    let out_new = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_new");

    gpu.attention_q8_0_kv_batched_masked(
        &q, &k_cache, &v_cache, &out_ref, &positions,
        nh, nkv, hd, ctx, ctx, n, None, 0, 0,
    ).expect("reference kernel");

    gpu.attention_q8_0_flash_prefill(
        &q, &k_cache, &v_cache, &out_new, &positions,
        nh, nkv, hd, ctx, n, br, bc,
    ).expect("flash prefill kernel");

    let a = gpu.download_f32(&out_ref).expect("dl ref");
    let b = gpu.download_f32(&out_new).expect("dl new");
    assert_eq!(a.len(), b.len());

    let (mut max_rel, mut max_abs) = (0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        let abs = (x - y).abs();
        if x.abs() > 1e-3 { max_rel = max_rel.max(abs / x.abs()); } else { max_abs = max_abs.max(abs); }
    }
    // Cosine similarity per (query, head) output vector.
    let mut min_cos = 1.0f32;
    for vec_i in 0..(n * nh) {
        let s = vec_i * hd;
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for d in 0..hd {
            dot += (a[s + d] as f64) * (b[s + d] as f64);
            na += (a[s + d] as f64).powi(2);
            nb += (b[s + d] as f64).powi(2);
        }
        if na > 0.0 && nb > 0.0 {
            min_cos = min_cos.min((dot / (na.sqrt() * nb.sqrt())) as f32);
        }
    }
    println!("nh={nh} nkv={nkv} hd={hd} n={n} ctx={ctx} br={br} bc={bc}");
    println!("max_rel={max_rel:.3e} max_abs={max_abs:.3e} min_cos={min_cos:.9}");
    assert!(max_rel <= 1e-4, "max relative error {max_rel:.3e} > 1e-4");
    assert!(max_abs <= 1e-5, "max absolute error {max_abs:.3e} > 1e-5");
    assert!(min_cos >= 1.0 - 1e-6, "min cosine {min_cos:.9} < 1-1e-6");
    println!("PASS");
}

/// Minimal f32 -> IEEE binary16 bit pattern (round-toward-zero mantissa).
/// Only needs to cover the small positive scales used above.
fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits & 0x007F_FFFF) >> 13;
    if exp <= 0 { return sign; }
    if exp >= 31 { return sign | 0x7C00; }
    sign | ((exp as u16) << 10) | (mant as u16)
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo build --release --example test_q8_flash_prefill 2>&1 | tail -20`
Expected: FAIL — `no method named 'attention_q8_0_flash_prefill' found for struct 'Gpu'`.

- [ ] **Step 3: Write the kernel (single K/V tile)**

Create `kernels/src/attention_q8_0_flash_prefill.hip`:

```cpp
// SPDX-License-Identifier: Apache-2.0
// hipfire — query-tiled Q8_0 flash prefill attention.
//
// Grid:  [ceil(batch_size / BR), n_heads]
// Block: 256 threads.
//
// One workgroup owns BR query rows for one head and streams the K/V cache in
// BC-key tiles staged in LDS in native Q8_0 block form. K/V is therefore read
// once per BR queries rather than once per query.
//
// TASK 1 SCOPE: exactly one K/V tile (host asserts seq_len <= BC). The tile
// loop and online softmax arrive in Task 2.
//
// LDS layout (all float regions first so the byte regions stay 4-aligned and
// each 34-byte block start stays 2-aligned for the fp16 scale load):
//   s[BR*BC] | m[BR] | l[BR] | corr[BR] | q[BR*head_dim] | kt[BC*bph*34] | vt[...]

#include <hip/hip_runtime.h>

#ifndef BR
#define BR 16
#endif
#ifndef BC
#define BC 32
#endif
#ifndef NTHREADS
#define NTHREADS 256
#endif

extern "C" __global__ __launch_bounds__(NTHREADS) void attention_q8_0_flash_prefill(
    const float* __restrict__ q,               // [batch_size × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache, // [max_seq × n_kv_heads × bph × 34]
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                   // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,         // [batch_size]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn
) {
    const int q_tile = blockIdx.x;
    const int h      = blockIdx.y;
    const int tid    = threadIdx.x;
    if (h >= n_heads) return;

    const int q_base = q_tile * BR;
    if (q_base >= batch_size) return;

    const int bph          = head_dim / 32;          // Q8_0 blocks per head
    const int total_blocks = n_kv_heads * bph;
    const int row_stride   = total_blocks * 34;      // bytes per cache position
    const int kv_group     = n_heads / n_kv_heads;
    const int kv_h         = h / kv_group;
    const int kv_blk       = kv_h * bph;
    const int q_stride     = n_heads * head_dim;

    extern __shared__ float smem[];
    float* s_lds    = smem;                       // [BR*BC]
    float* m_lds    = s_lds + BR * BC;            // [BR]
    float* l_lds    = m_lds + BR;                 // [BR]
    float* corr_lds = l_lds + BR;                 // [BR]
    float* q_lds    = corr_lds + BR;              // [BR*head_dim]
    unsigned char* kt = (unsigned char*)(q_lds + BR * head_dim);
    unsigned char* vt = kt + BC * bph * 34;

    // Rows present in this tile (last tile may be ragged).
    const int rows = min(BR, batch_size - q_base);

    // ── Stage Q tile ────────────────────────────────────────────────────
    for (int i = tid; i < rows * head_dim; i += NTHREADS) {
        const int r = i / head_dim;
        const int d = i % head_dim;
        q_lds[r * head_dim + d] = q[(long long)(q_base + r) * q_stride + h * head_dim + d];
    }
    // Per-row running state.
    if (tid < BR) { m_lds[tid] = -1e30f; l_lds[tid] = 0.0f; corr_lds[tid] = 1.0f; }
    __syncthreads();

    // Output accumulator: thread owns row r = tid / 16, dims [c*DPT, c*DPT+DPT)
    // where c = tid % 16 and DPT = head_dim / 16.
    const int lanes_per_row = NTHREADS / BR;         // 16
    const int acc_r  = tid / lanes_per_row;
    const int acc_c  = tid % lanes_per_row;
    const int dpt    = head_dim / lanes_per_row;     // 16 for head_dim=256
    const int d0     = acc_c * dpt;
    float acc[32];                                   // dpt <= 32
    for (int i = 0; i < dpt; i++) acc[i] = 0.0f;

    // Longest causal window in this tile decides how many keys to visit.
    int seq_len = 0;
    for (int r = 0; r < rows; r++) seq_len = max(seq_len, positions[q_base + r] + 1);

    // ── Single K/V tile (Task 1 scope) ──────────────────────────────────
    const int t_len = min(BC, seq_len);

    for (int i = tid; i < t_len * bph * 34; i += NTHREADS) {
        const int j  = i / (bph * 34);
        const int bo = i % (bph * 34);
        kt[i] = k_cache[(size_t)j * row_stride + kv_blk * 34 + bo];
        vt[i] = v_cache[(size_t)j * row_stride + kv_blk * 34 + bo];
    }
    __syncthreads();

    // S = Q · K^T with causal mask. BR*BC entries over NTHREADS threads.
    for (int e = tid; e < BR * BC; e += NTHREADS) {
        const int r = e / BC;
        const int j = e % BC;
        float sv = -INFINITY;
        if (r < rows && j < t_len && j <= positions[q_base + r]) {
            float dot = 0.0f;
            const unsigned char* kb = kt + (size_t)j * bph * 34;
            for (int bi = 0; bi < bph; bi++) {
                const float ks = (float)*((const _Float16*)(kb + bi * 34));
                const float* qb = q_lds + r * head_dim + bi * 32;
                for (int c = 0; c < 32; c++)
                    dot += qb[c] * (ks * (float)((signed char)kb[bi * 34 + 2 + c]));
            }
            sv = dot * scale_attn;
        }
        s_lds[e] = sv;
    }
    __syncthreads();

    // Row softmax (one thread per row; Task 2 keeps this shape).
    if (tid < rows) {
        float mx = -1e30f;
        for (int j = 0; j < BC; j++) mx = fmaxf(mx, s_lds[tid * BC + j]);
        float sum = 0.0f;
        for (int j = 0; j < BC; j++) {
            const float e = (s_lds[tid * BC + j] == -INFINITY) ? 0.0f : __expf(s_lds[tid * BC + j] - mx);
            s_lds[tid * BC + j] = e;
            sum += e;
        }
        m_lds[tid] = mx;
        l_lds[tid] = sum;
    }
    __syncthreads();

    // O += P · V
    if (acc_r < rows) {
        for (int j = 0; j < t_len; j++) {
            const float p = s_lds[acc_r * BC + j];
            if (p == 0.0f) continue;
            const unsigned char* vb = vt + (size_t)j * bph * 34;
            for (int i = 0; i < dpt; i++) {
                const int d  = d0 + i;
                const int bi = d >> 5;
                const int bj = d & 31;
                const float vs = (float)*((const _Float16*)(vb + bi * 34));
                acc[i] += p * (vs * (float)((signed char)vb[bi * 34 + 2 + bj]));
            }
        }
        float* o = out + (long long)(q_base + acc_r) * q_stride + h * head_dim;
        const float inv = 1.0f / l_lds[acc_r];
        for (int i = 0; i < dpt; i++) o[d0 + i] = acc[i] * inv;
    }
}
```

- [ ] **Step 4: Register the kernel source**

In `crates/rdna-compute/src/kernels.rs`, immediately after the
`ATTENTION_Q8_0_KV_BATCHED_SRC` definition (line 3827-3828), add:

```rust
/// Query-tiled Q8_0 flash prefill attention. LDS depends only on BR/BC,
/// never on context length, so one kernel serves every sequence length.
pub const ATTENTION_Q8_0_FLASH_PREFILL_SRC: &str =
    include_str!("../../../kernels/src/attention_q8_0_flash_prefill.hip");
```

- [ ] **Step 5: Add the launcher**

In `crates/rdna-compute/src/attention.rs`, after `attention_q8_0_kv_batched_masked` ends, add:

```rust
    /// Query-tiled Q8_0 flash prefill attention.
    /// `br`/`bc` are compile-time tile sizes; each (br, bc) pair compiles to
    /// its own module so they can be swept without touching the source.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_flash_prefill(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx_len: usize,
        batch_size: usize,
        br: usize,
        bc: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const NTHREADS: usize = 256;
        let module = format!("attention_q8_0_flash_prefill_br{br}_bc{bc}");
        let src = format!(
            "#define BR {br}\n#define BC {bc}\n#define NTHREADS {NTHREADS}\n{}",
            kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC
        );
        self.ensure_kernel(&module, &src, "attention_q8_0_flash_prefill")?;

        let bph = head_dim / 32;
        let lds = (br * bc + 3 * br + br * head_dim) * 4 + 2 * bc * bph * 34;
        assert!(lds <= 64 * 1024, "flash prefill LDS {lds} exceeds 64KB (br={br} bc={bc})");

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut sc = scale;
        let _ = max_ctx_len; // cache stride is derived from n_kv_heads/head_dim
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let grid_x = batch_size.div_ceil(br) as u32;
        self.launch_maybe_blob(
            "attention_q8_0_flash_prefill",
            [grid_x, n_heads as u32, 1],
            [NTHREADS as u32, 1, 1],
            lds as u32,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pos_ptr);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b.push_f32(sc);
                b
            },
        )
    }
```

If `ensure_kernel` is `pub(crate)` and this method sits in the same crate, no
visibility change is needed. `launch_maybe_blob` is likewise `pub(crate)`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo build --release --example test_q8_flash_prefill && CTX=32 N=16 ./target/release/examples/test_q8_flash_prefill`
Expected: prints `max_rel=... max_abs=... min_cos=...` then `PASS`.

If the reference kernel's grid requires `batch_size > 1` for the batched path,
keep `N=16`; that condition holds.

- [ ] **Step 7: Commit**

```bash
git add kernels/src/attention_q8_0_flash_prefill.hip \
        crates/rdna-compute/src/kernels.rs \
        crates/rdna-compute/src/attention.rs \
        crates/rdna-compute/examples/test_q8_flash_prefill.rs
git commit -m "feat(kernels): query-tiled Q8 flash prefill attention, single tile"
```

---

### Task 2: K/V tile loop with online softmax

**Files:**
- Modify: `kernels/src/attention_q8_0_flash_prefill.hip` (replace the single-tile section)
- Modify: `crates/rdna-compute/examples/test_q8_flash_prefill.rs` (no change needed; driven by env)

**Interfaces:**
- Consumes: `Gpu::attention_q8_0_flash_prefill` from Task 1 (unchanged signature).
- Produces: same signature, now correct for any `seq_len`.

- [ ] **Step 1: Write the failing test**

No new file. The failing case is multi-tile, driven by env:

Run: `CTX=512 N=64 ./target/release/examples/test_q8_flash_prefill`
Expected: FAIL — assertion on `max_rel`/`min_cos`, because Task 1 visits only the first `BC` keys.

- [ ] **Step 2: Confirm it fails**

Run the command above and record the failure output before changing code.

- [ ] **Step 3: Replace the single-tile section with the tile loop**

In `kernels/src/attention_q8_0_flash_prefill.hip`, delete everything from the
comment `// ── Single K/V tile (Task 1 scope) ──` through the end of the kernel
body, and replace with:

```cpp
    // ── K/V tile loop with online softmax ───────────────────────────────
    for (int t0 = 0; t0 < seq_len; t0 += BC) {
        const int t_len = min(BC, seq_len - t0);

        __syncthreads();                       // protect kt/vt from the previous iteration
        for (int i = tid; i < t_len * bph * 34; i += NTHREADS) {
            const int j  = i / (bph * 34);
            const int bo = i % (bph * 34);
            const size_t off = (size_t)(t0 + j) * row_stride + kv_blk * 34 + bo;
            kt[i] = k_cache[off];
            vt[i] = v_cache[off];
        }
        __syncthreads();

        for (int e = tid; e < BR * BC; e += NTHREADS) {
            const int r = e / BC;
            const int j = e % BC;
            float sv = -INFINITY;
            if (r < rows && j < t_len && (t0 + j) <= positions[q_base + r]) {
                float dot = 0.0f;
                const unsigned char* kb = kt + (size_t)j * bph * 34;
                for (int bi = 0; bi < bph; bi++) {
                    const float ks = (float)*((const _Float16*)(kb + bi * 34));
                    const float* qb = q_lds + r * head_dim + bi * 32;
                    for (int c = 0; c < 32; c++)
                        dot += qb[c] * (ks * (float)((signed char)kb[bi * 34 + 2 + c]));
                }
                sv = dot * scale_attn;
            }
            s_lds[e] = sv;
        }
        __syncthreads();

        // Online softmax: new running max, rescale factor, running sum.
        if (tid < rows) {
            float mx = m_lds[tid];
            for (int j = 0; j < BC; j++) mx = fmaxf(mx, s_lds[tid * BC + j]);
            const float corr = (m_lds[tid] <= -1e30f) ? 1.0f : __expf(m_lds[tid] - mx);
            float sum = 0.0f;
            for (int j = 0; j < BC; j++) {
                const float sv = s_lds[tid * BC + j];
                const float e  = (sv == -INFINITY) ? 0.0f : __expf(sv - mx);
                s_lds[tid * BC + j] = e;
                sum += e;
            }
            m_lds[tid]    = mx;
            l_lds[tid]    = l_lds[tid] * corr + sum;
            corr_lds[tid] = corr;
        }
        __syncthreads();

        // O = O * corr + P · V
        if (acc_r < rows) {
            const float corr = corr_lds[acc_r];
            if (corr != 1.0f) for (int i = 0; i < dpt; i++) acc[i] *= corr;
            for (int j = 0; j < t_len; j++) {
                const float p = s_lds[acc_r * BC + j];
                if (p == 0.0f) continue;
                const unsigned char* vb = vt + (size_t)j * bph * 34;
                for (int i = 0; i < dpt; i++) {
                    const int d  = d0 + i;
                    const int bi = d >> 5;
                    const int bj = d & 31;
                    const float vs = (float)*((const _Float16*)(vb + bi * 34));
                    acc[i] += p * (vs * (float)((signed char)vb[bi * 34 + 2 + bj]));
                }
            }
        }
    }

    // ── Write output ────────────────────────────────────────────────────
    if (acc_r < rows) {
        float* o = out + (long long)(q_base + acc_r) * q_stride + h * head_dim;
        const float denom = l_lds[acc_r];
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        for (int i = 0; i < dpt; i++) o[d0 + i] = acc[i] * inv;
    }
}
```

Note the entries with `j >= t_len` keep `-INFINITY` from the S pass, so they
contribute `0` to both `sum` and the P·V accumulation. The `t_len` bound in the
P·V loop makes that explicit.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo build --release --example test_q8_flash_prefill
for cfg in "32 16" "512 64" "1024 128" "4096 256" "8192 256" "12288 256"; do
  set -- $cfg; echo "CTX=$1 N=$2"; CTX=$1 N=$2 ./target/release/examples/test_q8_flash_prefill | tail -2
done
```
Expected: `PASS` for every configuration.

- [ ] **Step 5: Test the non-multiple-of-BC and short-sequence boundaries**

```bash
for cfg in "33 16" "31 8" "65 17" "97 33" "129 16"; do
  set -- $cfg; echo "CTX=$1 N=$2"; CTX=$1 N=$2 ./target/release/examples/test_q8_flash_prefill | tail -1
done
```
Expected: `PASS` for every configuration. These cover `seq_len % BC != 0`,
`seq_len < BC`, and `batch_size % BR != 0`.

- [ ] **Step 6: Commit**

```bash
git add kernels/src/attention_q8_0_flash_prefill.hip
git commit -m "feat(kernels): flash prefill K/V tile loop with online softmax"
```

---

### Task 3: Ragged positions and GQA coverage

Task 1/2 tests use a uniform tail window (`positions[b] = ctx - n + b`). Real
prefill chunks can carry per-row windows, and GQA (`n_kv_heads < n_heads`) must
map each query head to the right KV head.

**Files:**
- Modify: `crates/rdna-compute/examples/test_q8_flash_prefill.rs`

**Interfaces:**
- Consumes: `Gpu::attention_q8_0_flash_prefill` (unchanged).
- Produces: nothing new; this task only widens coverage.

- [ ] **Step 1: Write the failing test**

In `test_q8_flash_prefill.rs`, replace the `pos_data` line with an env-selected
pattern:

```rust
    // POS=tail  : positions[b] = ctx - n + b   (contiguous tail chunk)
    // POS=ragged: every row gets a different, non-monotonic causal window,
    //             which exercises per-row masking and the per-tile seq_len max.
    let pos_mode = std::env::var("POS").unwrap_or_else(|_| "tail".into());
    let pos_data: Vec<i32> = match pos_mode.as_str() {
        "ragged" => (0..n)
            .map(|b| {
                let span = ctx.max(2) - 1;
                (((b * 7919) % span) + 1) as i32
            })
            .collect(),
        _ => (0..n).map(|b| (ctx - n + b) as i32).collect(),
    };
```

- [ ] **Step 2: Run to verify the new coverage fails or passes explicitly**

```bash
cargo build --release --example test_q8_flash_prefill
POS=ragged CTX=1024 N=64 ./target/release/examples/test_q8_flash_prefill | tail -2
```
Expected: `PASS`. If it FAILS, the bug is the per-tile `seq_len` being a tile-wide
max while masking is per row — the `(t0 + j) <= positions[q_base + r]` guard in
Task 2 Step 3 is what makes this correct. Fix there, not in the test.

- [ ] **Step 3: Add the GQA and head-count sweep**

```bash
for cfg in "8 2" "8 8" "8 1" "4 2"; do
  set -- $cfg; echo "NH=$1 NKV=$2"
  NH=$1 NKV=$2 POS=ragged CTX=2048 N=64 ./target/release/examples/test_q8_flash_prefill | tail -1
done
```
Expected: `PASS` for every configuration. `NH=8 NKV=8` is kv_group=1,
`NH=8 NKV=1` is kv_group=8; both must map correctly.

- [ ] **Step 4: Run the full matrix once more**

```bash
for p in tail ragged; do for cfg in "512 64" "4096 256" "12288 256" "97 33"; do
  set -- $cfg; POS=$p CTX=$1 N=$2 ./target/release/examples/test_q8_flash_prefill | tail -1
done; done
```
Expected: eight `PASS` lines.

- [ ] **Step 5: Commit**

```bash
git add crates/rdna-compute/examples/test_q8_flash_prefill.rs
git commit -m "test(kernels): ragged positions and GQA coverage for flash prefill"
```

---

### Task 4: Microbench arm and BR/BC sweep

**Files:**
- Modify: `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`

**Interfaces:**
- Consumes: `Gpu::attention_q8_0_flash_prefill`, and the existing `time` closure and tensors in the microbench.
- Produces: a chosen `(BR, BC)` default, recorded in the commit message and used by Task 5.

- [ ] **Step 1: Add the new arm to the microbench**

In `crates/rdna-compute/examples/q8_batched_attn_microbench.rs`, after the
existing `new_ms` measurement, add:

```rust
    // Query-tiled flash prefill. BR/BC swept via env; LDS is independent of ctx.
    let br = env_usize("BR", 16);
    let bc = env_usize("BC", 32);
    let flash_ms = time(&mut gpu, &|g: &mut Gpu| {
        g.attention_q8_0_flash_prefill(
            &q, &k_cache, &v_cache, &out, &positions, nh, nkv, hd, ctx, n, br, bc,
        )
        .expect("flash prefill");
    });
    println!("flash_prefill br={br} bc={bc}: {flash_ms:.3} ms");
```

- [ ] **Step 2: Build and take a first reading**

```bash
cargo build --release --example q8_batched_attn_microbench
NH=8 NKV=2 HD=256 N=256 CTX=8192 ./target/release/examples/q8_batched_attn_microbench
```
Expected: prints the existing arm's ms and the new `flash_prefill` ms.

- [ ] **Step 3: Sweep BR/BC**

```bash
for br in 8 16 32; do for bc in 32 64; do
  echo -n "BR=$br BC=$bc  "
  NH=8 NKV=2 HD=256 N=256 CTX=8192 BR=$br BC=$bc \
    ./target/release/examples/q8_batched_attn_microbench 2>/dev/null | grep flash_prefill
done; done
```
Expected: a table of timings. Configurations whose LDS exceeds 64 KB will abort
on the `assert!` in the launcher — record them as N/A rather than removing the
assert. LDS = `(BR*BC + 3*BR + BR*256)*4 + 2*BC*8*34` bytes.

- [ ] **Step 4: Sweep context length at the winning BR/BC**

From the Step 3 table pick the `(BR, BC)` with the lowest `flash_prefill` ms.
If two are within 3% of each other, prefer the smaller `BR` (more workgroups,
better behaviour on short chunks). If the sweep is inconclusive or all
configurations abort on the LDS assert, use `BR=16 BC=32`. Export the choice so
the commands below are literal:

```bash
export WBR=16 WBC=32     # <- replace with the winning pair from Step 3
for ctx in 2048 4096 8192 12288 16000; do
  echo -n "CTX=$ctx  "
  NH=8 NKV=2 HD=256 N=256 CTX=$ctx BR=$WBR BC=$WBC \
    ./target/release/examples/q8_batched_attn_microbench 2>/dev/null | grep -E 'flash_prefill|windowed'
done
```
Expected: the flash arm's advantage grows with CTX, and it does not collapse at
CTX=16000 (where the old LDS kernel cannot run at all).

- [ ] **Step 5: Commit**

```bash
git add crates/rdna-compute/examples/q8_batched_attn_microbench.rs
git commit -m "bench(kernels): flash prefill arm and BR/BC sweep"
```

Record the winning `(BR, BC)` and the sweep table in the commit body.

---

### Task 5: Dispatch wiring behind an env opt-in

**Files:**
- Modify: `crates/hipfire-dispatch/src/families/attention.rs:1239-1281`
- Modify: `crates/rdna-compute/src/dispatch.rs:2545-2565` (precompile spec list)

**Interfaces:**
- Consumes: `Gpu::attention_q8_0_flash_prefill`.
- Produces: `HIPFIRE_FLASH_PREFILL=1` opt-in routing inside `KernelKey::AttnQ8_0KvBatchedMasked`.

- [ ] **Step 1: Add the opt-in branch**

In `crates/hipfire-dispatch/src/families/attention.rs`, inside
`KernelKey::AttnQ8_0KvBatchedMasked`, insert **before** the existing
`if io.max_ctx_len <= Q8_BATCHED_LDS_CROSSOVER` test:

```rust
                // Query-tiled flash prefill: LDS is independent of context, so
                // this path has no crossover. Opt-in until Stage A is validated;
                // causal non-tree only (tree/window keep the legacy kernels).
                let flash_optin = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL")
                    .ok()
                    .as_deref()
                    == Some("1");
                if flash_optin && io.tree_bias.is_none() && io.batch_size > 1 {
                    let br: usize = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_BR")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(16);
                    let bc: usize = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_BC")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(32);
                    return hip!(gpu.attention_q8_0_flash_prefill(
                        io.q,
                        io.k_cache,
                        io.v_cache,
                        io.output,
                        io.positions(),
                        io.n_heads,
                        io.n_kv_heads,
                        io.head_dim,
                        io.max_ctx_len,
                        io.batch_size,
                        br,
                        bc,
                    ));
                }
```

The `unwrap_or(16)` / `unwrap_or(32)` defaults above are the fallback pair. If
Task 4 Step 3 chose a different `(BR, BC)`, edit those two `unwrap_or` values to
match — they are the defaults used whenever `HIPFIRE_FLASH_PREFILL_BR`/`_BC` are
unset.

This branch sits inside `KernelKey::AttnQ8_0KvBatchedMasked`, which is already
the non-windowed key (sliding-window traffic uses the separate
`AttnQ8_0KvBatchedMaskedWindowed` key), so the `window <= 0` half of the spec's
scope boundary is satisfied structurally. The `tree_bias.is_none()` guard covers
the non-tree half, and `batch_size > 1` keeps decode on its existing path.

- [ ] **Step 2: Add the kernel to the precompile list**

In `crates/rdna-compute/src/dispatch.rs`, in the `"q8" | _ =>` arm alongside the
other `specs.push((...))` calls, add:

```rust
                specs.push((
                    "attention_q8_0_flash_prefill",
                    kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC.to_string(),
                ));
```

Note this precompiles the un-parameterised source; the BR/BC-specialised modules
still JIT on first use. That is acceptable for Stage A and shows up only as
first-chunk latency.

- [ ] **Step 3: Build and confirm the opt-in is inert when unset**

```bash
cargo build --release --bin hipfire --example daemon 2>&1 | tail -3
```
Expected: `Finished \`release\` profile`.

- [ ] **Step 4: Confirm identical output with the flag OFF**

```bash
SP=/tmp/claude-1000/-tmp/032fc170-e605-4f51-ad0d-cc474b95c351/scratchpad
./target/release/examples/coherence_probe \
  --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --prompt-file $SP/long_probe.txt --max-tokens 160 --temperature 0.0 \
  --max-seq 16384 --kv-mode q8 --emit-committed-jsonl $SP/tok_flagoff.jsonl
diff $SP/tok_8192.jsonl $SP/tok_flagoff.jsonl && echo "IDENTICAL (flag off is inert)"
```
Expected: `IDENTICAL (flag off is inert)`.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-dispatch/src/families/attention.rs crates/rdna-compute/src/dispatch.rs
git commit -m "feat(dispatch): opt-in routing for query-tiled flash prefill"
```

---

### Task 6: End-to-end validation gate

**Files:**
- Create: `docs/perf-checkpoints/2026-07-29-flash-prefill-stage-a-results.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the measured result table that decides whether Stage C proceeds.

- [ ] **Step 1: Token-stream equivalence with the flag ON**

```bash
SP=/tmp/claude-1000/-tmp/032fc170-e605-4f51-ad0d-cc474b95c351/scratchpad
HIPFIRE_FLASH_PREFILL=1 ./target/release/examples/coherence_probe \
  --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --prompt-file $SP/long_probe.txt --max-tokens 160 --temperature 0.0 \
  --max-seq 16384 --kv-mode q8 --emit-committed-jsonl $SP/tok_flash.jsonl
diff $SP/tok_8192.jsonl $SP/tok_flash.jsonl && echo "IDENTICAL" || echo "DIVERGENT (expected; inspect)"
```
Expected: all coherence rows `OK`. Token divergence is permitted by the
numerically-equivalent bar — if it diverges, record where and confirm the text
is still coherent in Step 2.

- [ ] **Step 2: Realistic-prose generation on both backends**

```bash
SP=/tmp/claude-1000/-tmp/032fc170-e605-4f51-ad0d-cc474b95c351/scratchpad
for BK in "" "hip"; do
  env ${BK:+HIPFIRE_REPLAY_BACKEND=$BK} HIPFIRE_FLASH_PREFILL=1 \
  python3 scripts/serve_harness.py --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
    --kv q8 --mtp off --thinking low --max-tokens 600 --max-seq 16384 \
    --sampling recipe:nothink --mode battery --prompts-file $SP/real_prompts.json \
    --out $SP/flash_real_${BK:-redline}.json --serve-log $SP/flash_real_${BK:-redline}.log
done
```
Expected: 6/6 `finish=stop` with coherent multi-word answers on each backend,
matching the baseline behaviour recorded in `real_8192_*.json`.

- [ ] **Step 3: Per-position curve — confirm the 8192 step is gone**

```bash
SP=/tmp/claude-1000/-tmp/032fc170-e605-4f51-ad0d-cc474b95c351/scratchpad
HIPFIRE_FLASH_PREFILL=1 HIPFIRE_REPLAY_BACKEND=hip \
python3 scripts/serve_harness.py --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
  --kv q8 --mtp off --thinking low --max-tokens 600 --max-seq 16384 \
  --sampling recipe:nothink --mode chain --prompts-file $SP/chain_prompts.json \
  --out $SP/chain_flash.json --serve-log $SP/chain_flash.log
```
Compare per-turn `prefill_ms / (ctx - cached)` against `chain_8192_hip.json`.
Expected: no step between p=7138 and p=8253, and lower ms/token at every p ≥ 7703.

- [ ] **Step 4: rocprofv3 exponent check**

```bash
SP=/tmp/claude-1000/-tmp/032fc170-e605-4f51-ad0d-cc474b95c351/scratchpad
for N in 2048 4096 8192 12288; do
  HIPFIRE_FLASH_PREFILL=1 rocprofv3 --kernel-trace --stats -S -f csv -d $SP/rp_flash -o p$N -- \
    ./target/release/examples/bench_qwen35_mq4 ~/.hipfire/models/qwen3.6-35b-a3b.mq4r \
    --prefill $N --prefill-runs 3 --gen 0 --warmup 0 2>&1 | grep PREFILL_SUMMARY
done
```
Expected: `attention_q8_0_flash_prefill` replaces both legacy attention kernels
in the stats, and its fitted exponent over 2048→12288 is below the 2.19 baseline.

- [ ] **Step 5: Write the results doc**

Create `docs/perf-checkpoints/2026-07-29-flash-prefill-stage-a-results.md` with:
the chosen (BR, BC) and the Task 4 sweep table; microbench ms at each CTX for
both arms; prefill tok/s at N ∈ {2048, 4096, 8192, 12288} before and after; the
attention exponent before (2.19) and after; the per-position curve comparison;
and an explicit statement of whether the Stage A gate passed.

Stage A gate: **flash prefill must be faster than the legacy path at every
CTX ≥ 2048 and must not regress prefill tok/s at N=2048.** If it regresses only
at small N, record it and note the mitigation (keep the legacy kernel for
`max_ctx_len` below the measured break-even) rather than declaring a pass.

- [ ] **Step 6: Commit**

```bash
git add docs/perf-checkpoints/2026-07-29-flash-prefill-stage-a-results.md
git commit -m "docs: Stage A flash prefill results"
```

---

## Stage gate

Do not begin Stage C (WMMA inner math) until Task 6 records a pass. If Stage A
fails its gate, the likely cause is the parallelism drop noted in the spec
(2048 workgroups → 128–256); re-run the Task 4 BR sweep at the failing N before
concluding the approach is wrong.
