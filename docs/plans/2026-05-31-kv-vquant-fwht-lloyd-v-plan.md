# FWHT-V → Lloyd-V V-cache Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantize the Qwen3.5/3.6 V-cache below its always-Q8 (272 B/head) layout by adding FWHT-rotated, centroid-LUT V modes (`lloyd2/3/4`), with K-bit and V-bit chosen independently, validated by a 3 K × 4 V KLD matrix.

**Architecture:** A runtime `v_mode` selector (bit-count: `8`=Q8 today, `2/3/4`=Lloyd-V) threaded through the existing KV write + flash-attention kernels — **no new per-(K×V) kernel files**. The write side reuses the existing `kv_cache_write_asym_k_fwht{2,3,4}` kernels pointed at the V buffer (they already rotate→unit-normalize→centroid-quantize into the exact `4 + head_dim·bits/8` layout). The genuinely new code is the attention V-read path: dequant `cnorm·TURBO_C*[idx]` accumulated in rotated space, then **one** `fwht_shfl_inverse_256` on the per-thread output registers before writeback (valid because V is softmax-summed, not dotted, and FWHT commutes with the linear combine — so the cross-tile reduce kernel needs no change).

**Tech Stack:** Rust (`crates/hipfire-runtime`, `crates/rdna-compute`, `crates/hipfire-arch-qwen35`), HIP kernels (`kernels/src/*.hip` + `turbo_common.h`), JIT-compiled via `ensure_givens4_kernel`. Validation: `eval_hipfire` KLD harness + `kld_reduce.py` + `scripts/coherence-gate.sh`. Build: `RUSTC_WRAPPER=sccache cargo build --release ...`. GPU coordination via `scripts/gpu-lock.sh`.

**Branch:** `feat/kv-vquant-fwht-lloyd-v` (stacked on PR #366). Design doc: `docs/plans/2026-05-31-kv-vquant-fwht-lloyd-v-design.md`.

---

## Orientation: how the pieces connect (read once before starting)

- `KvCache` struct + all ~50 ctors: `crates/hipfire-runtime/src/llama.rs` (struct at `:3306-3347`). Shared across archs.
- The fwht3 canonical single-GPU ctor: `new_gpu_fwht3_capped_filtered` (`llama.rs:3870-3911`); the byte sizing pattern `v_bpp = n_kv_heads * (head_dim/32) * 34` appears in 45 places (3 textual forms; see Task 1).
- Write launchers: `crates/rdna-compute/src/dispatch.rs` — `kv_cache_write_fwht3_fused` (`:23903-23942`, single) and `kv_cache_write_fwht3_batched` (`:24395-24444`). Both **end** by writing V via `kv_cache_write_q8_0[_batched]` (`:23941`, `:24443`). That tail call is what we replace for Lloyd-V.
- The K write kernels we reuse for V: `kernels/src/kv_cache_write_asym_k_fwht{2,3,4}.hip` (+ `_batched`). Registered as `KV_CACHE_WRITE_ASYM_K_FWHT{2,3,4}[_BATCHED]_SRC` in `crates/rdna-compute/src/kernels.rs:2018-2019` (and siblings).
- Attention tile kernel: `kernels/src/attention_flash_fwht3_tile.hip` (Phase D = V read at `:113-129`; output regs `out_vec[8]` per thread cover head_dim=256 across 32 lanes). Launcher `attention_flash_fwht3` (`dispatch.rs:24537-24628`); two-kernel pipeline (tile → `attention_flash_q8_0_reduce`). Batched variant via `launch_asym_flash_batched`.
- Quant primitives in `kernels/src/turbo_common.h`: `fwht_shfl_forward_256` (`:168`), `fwht_shfl_inverse_256` (`:220`), `TURBO_C{2,3,4}_256` (`:21-27`), `turbo_quantize_{2,3,4}bit_256` (`:330-342`). All `#include`d (textually inlined) by the kernels already.
- Runtime dispatch (the path eval/serve actually hit for 27B): `crates/hipfire-arch-qwen35/src/qwen35.rs` — write sites `kv_cache_write_fwht3_fused` at `:8438, 9259, 9776, 10255, 10645`, batched at `:7057, 7985`; attention sites at `:7135, 8053, 9262, 9779, 10258, 10648`. The fwht branch is gated by `kv_cache.quant_asym3 && kv_cache.quant_fwht` (the sign tables ride in `kv_cache.givens_cos/givens_sin`).
- `eval_hipfire`: `crates/hipfire-runtime/examples/eval_hipfire.rs` — `--kv-mode` parse `:77-85`, ctor `match` `:247-270`.

**Key reuse insight:** because `kv_cache_write_asym_k_fwht3(dst, src, pos, signs1, signs2, n_kv_heads, head_dim)` writes any vector into the `4 + head_dim·3/8` rotated-centroid layout, calling it on the **V** buffer produces a valid lloyd3-V. K and V can share the same sign tables (independent vectors, same randomized-Hadamard). So the write side is a dispatch change, not new kernel authoring.

**Validation reality:** there is no GPU unit-test harness. The "test" for each kernel task is: (a) `cargo build` green, (b) a smoke run that loads + generates without panic, and (c) the KLD/coherence gates. The end-to-end correctness signal for the whole novel mechanism is **Task 5** (fwht3-K/lloyd3-V smoke KLD ≈ fwht3/q8 baseline). If the inverse FWHT or layout is wrong, KLD explodes — it is a sharp check.

---

## Task 1: Scaffold — `VMode` enum, `KvCache.v_mode` field, sizing helper (byte-identical, builds green)

**Files:**
- Modify: `crates/hipfire-runtime/src/llama.rs` (struct `:3306-3347`; ~50 ctor `Self { ... }` literals; 45 V-sizing sites)

- [ ] **Step 1: Add the `VMode` enum**

Above the `KvCache` struct in `llama.rs` (e.g. just before `pub struct KvCache` at `:3306`):

```rust
/// V-cache quantization mode. The bit-count IS the kernarg value passed to
/// kernels: 8 = legacy Q8_0 (per-32-block fp16 scale + int8, 272 B/head at hd=256),
/// 2/3/4 = FWHT-rotated centroid-LUT V (Lloyd-V), layout identical to the K fwht
/// modes: `4 + head_dim*bits/8` B/head with one f32 cnorm per head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VMode {
    Q8,
    Lloyd2,
    Lloyd3,
    Lloyd4,
}

impl VMode {
    /// Kernarg value: the per-element bit count (8 for Q8). Drives both kernel
    /// dispatch branches and byte-layout arithmetic.
    pub fn bits(self) -> i32 {
        match self {
            VMode::Q8 => 8,
            VMode::Lloyd2 => 2,
            VMode::Lloyd3 => 3,
            VMode::Lloyd4 => 4,
        }
    }
}
```

- [ ] **Step 2: Add the struct field**

In `pub struct KvCache { ... }` (`llama.rs:3306-3347`), add after `pub quant_fwht: bool,` (`:3336`):

```rust
    /// V-cache quantization mode (independent of the K mode). Defaults to Q8.
    pub v_mode: VMode,
```

- [ ] **Step 3: Add the V-byte-sizing helper + a per-instance accessor**

Add as associated functions on `impl KvCache` (near the other helpers, e.g. after `alloc_k_v_filtered` at `:3454`):

```rust
    /// Bytes of V-cache per token-position (all heads) for a given V mode.
    /// Q8 = n_kv_heads * (head_dim/32) * 34. Lloyd = n_kv_heads * (4 + head_dim*bits/8).
    fn v_bytes_per_pos(n_kv_heads: usize, head_dim: usize, v_mode: VMode) -> usize {
        match v_mode {
            VMode::Q8 => n_kv_heads * (head_dim / 32) * 34,
            VMode::Lloyd2 | VMode::Lloyd3 | VMode::Lloyd4 => {
                n_kv_heads * (4 + (head_dim * v_mode.bits() as usize) / 8)
            }
        }
    }

    /// V-mode bit-count to pass as a kernarg.
    pub fn v_mode_bits(&self) -> i32 {
        self.v_mode.bits()
    }
```

- [ ] **Step 4: Set `v_mode: VMode::Q8` in every ctor literal, and parameterize the fwht filtered ctors' V sizing**

Every `Ok(Self { ... })` / `Self { ... }` literal must gain `v_mode: VMode::Q8,`. There are ~50. Do NOT hand-hunt — let the compiler enumerate them (Step 5). For the **three ctors eval uses** (`new_gpu_fwht2_filtered`, `new_gpu_fwht3_filtered`, `new_gpu_fwht4_filtered`), leave them defaulting to `VMode::Q8` here — V-mode override for the matrix is done post-construction via `set_v_mode_realloc` (Step 6), so their signatures are unchanged and their other callers (daemon) are untouched.

Representative edit (the fwht3 canonical ctor, `llama.rs:3905-3910`) — add the one field:

```rust
        Ok(Self {
            k_gpu, v_gpu, k_scales: vec![], v_scales: vec![], kv_dim,
            max_seq: max_seq_len, physical_cap, n_kv_heads, head_dim,
            quantized: true, quant_q8: false, quant_int8: false, quant_hfq4: false,
            quant_asym4: false, quant_asym3: true, quant_asym2: false, quant_fwht: true,
            boundary_layers: 0, givens_cos: Some(s1), givens_sin: Some(s2),
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        })
```

- [ ] **Step 5: Build and let the compiler list every remaining ctor**

Run: `RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime --features deltanet 2>&1 | grep -E "missing field .v_mode|missing structure field"`
Expected: a list of every `Self { ... }` literal missing `v_mode`. Add `v_mode: VMode::Q8,` to each, then rebuild until clean.

- [ ] **Step 6: Add `set_v_mode_realloc` (used by eval to size V for Lloyd without touching 50 ctor signatures)**

On `impl KvCache` in `llama.rs`:

```rust
    /// Reallocate the V buffers for a new V mode (used by eval/bench to set an
    /// independent V quant after construction). Re-sizes only real KV layers
    /// (placeholder 1-element buffers for non-KV layers are left as-is).
    /// K buffers and rotation tables are untouched.
    pub fn set_v_mode_realloc(&mut self, gpu: &mut Gpu, v_mode: VMode) -> HipResult<()> {
        let v_bpp = Self::v_bytes_per_pos(self.n_kv_heads, self.head_dim, v_mode);
        let v_elems = (self.physical_cap * v_bpp + 3) / 4;
        for t in self.v_gpu.iter_mut() {
            if t.numel() > 1 {
                *t = gpu.zeros(&[v_elems], DType::F32)?;
            }
        }
        self.v_mode = v_mode;
        Ok(())
    }
```

(If `GpuTensor` exposes element count under a different name than `numel()`, use whatever the crate uses — grep `impl GpuTensor`. The intent: distinguish real buffers from the 1-element non-KV placeholders.)

- [ ] **Step 7: Verify byte-identical behavior (regression guard)**

Run: `RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`
Expected: PASS. No runtime change yet — `v_mode` defaults to Q8 everywhere; this task only adds an unused field + helpers.

- [ ] **Step 8: Commit**

```bash
git add crates/hipfire-runtime/src/llama.rs
git commit -m "feat(kv): add VMode enum + KvCache.v_mode scaffold (default Q8, byte-identical)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Reuse the K fwht write kernels for V — V-write dispatch gated on `v_mode`

The existing `kv_cache_write_asym_k_fwht{2,3,4}` kernels write the exact Lloyd-V layout. We add small reusable launcher methods (mirroring the inline K launch already in the fused wrappers) and branch the fused wrappers' V-write tail on `v_mode`.

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs` (`kv_cache_write_fwht3_fused` `:23903-23942`, `kv_cache_write_fwht3_batched` `:24395-24444`, and the fwht2/fwht4 equivalents)
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs` (write sites `:7057, 7985, 8438, 9259, 9776, 10255, 10645`)

- [ ] **Step 1: Add a reusable single-token K-format writer (works on any buffer)**

In `dispatch.rs`, add a thin method that launches the existing fwht3 K kernel on an arbitrary dst/src (this is the inline body already at `:23914-23940`, factored out):

```rust
    /// Launch the fwht3 rotated-centroid write kernel on an arbitrary KV buffer.
    /// Used for K (always) and for V when v_mode == Lloyd3.
    pub fn kv_write_fwht3_vec(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht3",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_SRC,
            "kv_cache_write_asym_k_fwht3",
        )?;
        let func = &self.functions["kv_cache_write_asym_k_fwht3"];
        let mut kdp = dst.buf.as_ptr();
        let mut ksp = src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void, &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1],
                shared_mem, self.stream_ref(), &mut params)?;
        }
        Ok(())
    }
```

Add the batched twin `kv_write_fwht3_vec_batched` using `KV_CACHE_WRITE_ASYM_K_FWHT3_BATCHED_SRC` + `launch_maybe_blob` (copy the structure from `kv_cache_write_fwht3_batched` `:24407-24442`, but as a standalone vec writer taking `positions` + `batch_size`).

- [ ] **Step 2: Branch the fused wrappers' V tail on `v_mode`**

Change `kv_cache_write_fwht3_fused` (`dispatch.rs:23903`) to take a `v_mode_bits: i32` param and replace the tail `self.kv_cache_write_q8_0(v_dst, ...)` (`:23941`) with:

```rust
        if v_mode_bits == 3 {
            self.kv_write_fwht3_vec(v_dst, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim)
        } else {
            // v_mode_bits == 8 (Q8) — unchanged legacy path.
            self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
        }
```

Do the same for `kv_cache_write_fwht3_batched` (`:24443`) with `kv_write_fwht3_vec_batched`. (lloyd2/lloyd4 cases added in Task 6.)

- [ ] **Step 3: Thread `kv_cache.v_mode_bits()` through the qwen35 write sites**

At each `gpu.kv_cache_write_fwht3_fused(...)` / `..._batched(...)` call in `qwen35.rs` (`:7057, 7985, 8438, 9259, 9776, 10255, 10645`), append `kv_cache.v_mode_bits()` as the new last argument.

- [ ] **Step 4: Build**

Run: `RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`
Expected: PASS (V-write now branches, but `v_mode` is still Q8 everywhere → q8 tail taken → behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add crates/rdna-compute/src/dispatch.rs crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "feat(kv): v_mode-gated V write, reusing fwht3 K kernel for lloyd3-V

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Attention V-read — lloyd3 Phase D + tail inverse FWHT (the new kernel code)

**Files:**
- Modify: `kernels/src/attention_flash_fwht3_tile.hip` (signature + Phase D `:113-129` + writeback `:131-139`)
- Modify: `kernels/src/attention_flash_fwht3_tile_batched.hip` (same change)
- Modify: `crates/rdna-compute/src/dispatch.rs` (`attention_flash_fwht3` `:24537`, the `params` vec `:24569-24584`; and `launch_asym_flash_batched`)
- Modify: `crates/hipfire-arch-qwen35/src/qwen35.rs` (attention sites `:7135, 8053, 9262, 9779, 10258, 10648`)

- [ ] **Step 1: Add `int v_mode` to the tile kernel signature**

In `attention_flash_fwht3_tile.hip`, add a trailing param after `int max_tiles` (`:31`):

```cpp
    int max_tiles,
    int v_mode          // 8 = Q8_0 V (legacy); 3 = lloyd3 FWHT-rotated centroid V
) {
```

- [ ] **Step 2: Replace Phase D with a `v_mode` branch**

Replace the Phase D block (`attention_flash_fwht3_tile.hip:113-129`) with:

```cpp
    // Phase D: V — 8 output dims per thread (thread tid owns dims d0..d0+7).
    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    if (v_mode == 8) {
        // Legacy Q8_0 V (normal space) — unchanged.
        const int v_blocks_per_head = head_dim / 32;
        const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
        const int v_kv_head_blk     = kv_h * v_blocks_per_head;
        const int v_row_stride      = v_total_blocks * 34;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_row_stride;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int d = d0 + i;
                const int bi = d / 32;
                const int bj = d % 32;
                const int v_blk_off = (v_kv_head_blk + bi) * 34;
                const unsigned char* vbh = vb + v_blk_off;
                float vs = (float)*((const _Float16*)vbh);
                out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
            }
        }
    } else {
        // lloyd3 FWHT-rotated centroid V — layout identical to fwht3 K.
        const int v_bytes_per_head = 4 + (head_dim * 3) / 8;   // 100 B at hd=256
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
            float cnorm = *(const float*)vb;
            const unsigned char* base = vb + 4 + tid * 3;
            unsigned int packed = (unsigned int)base[0]
                                | ((unsigned int)base[1] << 8)
                                | ((unsigned int)base[2] << 16);
            out_vec[0] += w * (cnorm * TURBO_C3_256[ packed        & 7]);
            out_vec[1] += w * (cnorm * TURBO_C3_256[(packed >> 3)  & 7]);
            out_vec[2] += w * (cnorm * TURBO_C3_256[(packed >> 6)  & 7]);
            out_vec[3] += w * (cnorm * TURBO_C3_256[(packed >> 9)  & 7]);
            out_vec[4] += w * (cnorm * TURBO_C3_256[(packed >> 12) & 7]);
            out_vec[5] += w * (cnorm * TURBO_C3_256[(packed >> 15) & 7]);
            out_vec[6] += w * (cnorm * TURBO_C3_256[(packed >> 18) & 7]);
            out_vec[7] += w * (cnorm * TURBO_C3_256[(packed >> 21) & 7]);
        }
        // V was accumulated in rotated space: out_vec ≈ H·(Σ w·v). Undo once.
        fwht_shfl_inverse_256(out_vec[0], out_vec[1], out_vec[2], out_vec[3],
                              out_vec[4], out_vec[5], out_vec[6], out_vec[7],
                              signs1, signs2, tid);
    }
```

The writeback loop (`:131-139`) is unchanged — it writes `out_vec[i] → p[2 + d0 + i]`, now already in original space. The reduce kernel (`attention_flash_q8_0_reduce`) is unchanged (it sums per-dim partials, basis-agnostic).

- [ ] **Step 3: Mirror the change in the batched tile kernel**

Apply Steps 1-2 to `kernels/src/attention_flash_fwht3_tile_batched.hip` (same signature add + Phase D branch; the per-batch offset arithmetic stays, only the V-read region changes).

- [ ] **Step 4: Pass `v_mode` as a kernarg in the launchers**

In `attention_flash_fwht3` (`dispatch.rs:24537`), add a `v_mode_bits: i32` param to the fn signature, add `let mut vm = v_mode_bits;` near the other locals, and append `&mut vm as *mut _ as *mut c_void,` to the `params` vec (`:24569-24584`) as the final entry (matching the new kernel arg order). Do the same in `launch_asym_flash_batched` for the batched tile (thread `v_mode_bits` in and append to its kernarg list / `KernargBlob`).

- [ ] **Step 5: Thread `kv_cache.v_mode_bits()` through the qwen35 attention sites**

At each `gpu.attention_flash_fwht3(...)` / batched call in `qwen35.rs` (`:7135, 8053, 9262, 9779, 10258, 10648`), append `kv_cache.v_mode_bits()` as the new last arg.

- [ ] **Step 6: Build**

Run: `RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`
Expected: PASS. (Kernels JIT-compile on first launch at runtime, so a clean Rust build does not yet exercise the new HIP code.)

- [ ] **Step 7: Commit**

```bash
git add kernels/src/attention_flash_fwht3_tile.hip kernels/src/attention_flash_fwht3_tile_batched.hip crates/rdna-compute/src/dispatch.rs crates/hipfire-arch-qwen35/src/qwen35.rs
git commit -m "feat(kv): attention lloyd3-V read + tail inverse FWHT (v_mode kernarg)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `eval_hipfire --kv-v` flag → `set_v_mode_realloc`

**Files:**
- Modify: `crates/hipfire-runtime/examples/eval_hipfire.rs` (Args `:56-63`, parse `:72-105`, ctor `match` `:240-270`)

- [ ] **Step 1: Add `kv_v` to Args + default**

In the args struct add `kv_v: String,` and initialize `let mut kv_v = "q8".to_string();` near `kv_mode` (`:68`).

- [ ] **Step 2: Parse + validate `--kv-v`**

Add next to the `--kv-mode` arm (`:77`):

```rust
            "--kv-v" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "lloyd2" | "lloyd3" | "lloyd4") {
                    eprintln!("--kv-v must be one of: q8 lloyd2 lloyd3 lloyd4 (got {v})");
                    std::process::exit(1);
                }
                kv_v = v;
                i += 2;
            }
```

- [ ] **Step 3: Apply the V mode after cache construction**

After the `kv_mode` → ctor `match` (`:270`, the `let mut kv_cache = match ...`), add:

```rust
    let v_mode = match args.kv_v.as_str() {
        "q8" => llama::VMode::Q8,
        "lloyd2" => llama::VMode::Lloyd2,
        "lloyd3" => llama::VMode::Lloyd3,
        "lloyd4" => llama::VMode::Lloyd4,
        other => panic!("unknown --kv-v: {other}"),
    };
    if v_mode != llama::VMode::Q8 {
        kv_cache.set_v_mode_realloc(&mut gpu, v_mode).unwrap();
    }
```

(Use the actual import path for `VMode`/`KvCache` already in scope in this file.)

- [ ] **Step 4: Build**

Run: `RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-runtime/examples/eval_hipfire.rs
git commit -m "feat(eval): --kv-v flag for independent V quant selection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: End-to-end correctness — fwht3-K / lloyd3-V smoke KLD (GO/NO-GO)

This is the decisive correctness check for the entire mechanism (layout + rotation + inverse).

**Files:** none (validation only).

- [ ] **Step 1: Acquire GPU lock + ensure the 27B model and ref are present**

Run (local k9lin gfx1100 is fine for the smoke):
```bash
source scripts/gpu-lock.sh && gpu_acquire "kv-vquant-smoke"
ls -la ~/.hipfire/models/qwen3.6-27b.mq4 ~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin
```
Expected: both exist (model dir/file + 204,813,592-byte ref).

- [ ] **Step 2: Baseline cell — fwht3 K, Q8 V (4-chunk)**

```bash
mkdir -p benchmarks/quality-baselines/results/2026-05-31-kv-vquant
./target/release/examples/eval_hipfire \
    --model ~/.hipfire/models/qwen3.6-27b.mq4 \
    --ref ~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin \
    --output benchmarks/quality-baselines/results/2026-05-31-kv-vquant/smoke_fwht3_q8.kldseq \
    --kv-mode fwht3 --kv-v q8 --scoring-mode prefill --max-chunks 4 2>&1 | tee /tmp/smoke_fwht3_q8.log
grep "slice-mean KLD" /tmp/smoke_fwht3_q8.log
```
Expected: a `slice-mean KLD ≈ 0.011` line (matches the known fwht3 baseline; confirms harness + no regression from Tasks 1-4).

- [ ] **Step 3: New cell — fwht3 K, lloyd3 V (4-chunk)**

```bash
./target/release/examples/eval_hipfire \
    --model ~/.hipfire/models/qwen3.6-27b.mq4 \
    --ref ~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin \
    --output benchmarks/quality-baselines/results/2026-05-31-kv-vquant/smoke_fwht3_lloyd3.kldseq \
    --kv-mode fwht3 --kv-v lloyd3 --scoring-mode prefill --max-chunks 4 2>&1 | tee /tmp/smoke_fwht3_lloyd3.log
grep "slice-mean KLD" /tmp/smoke_fwht3_lloyd3.log
```
Expected (GO): KLD in the same ballpark as Step 2 — within roughly +0.000–0.003 of the Q8-V baseline (lloyd3-V should be close to Q8-V; some degradation is expected since V is sensitive, but not an explosion).
**NO-GO:** KLD ≫ baseline (e.g. >0.05) or NaN ⇒ the rotation/inverse or layout is wrong. Debug before proceeding: most likely the tail inverse basis (signs reused for V), the per-thread byte offset (`4 + tid*3`), or the `v_bytes_per_pos` mismatch between write (Task 2) and read (Task 3).

- [ ] **Step 4: Release lock + record**

```bash
source scripts/gpu-lock.sh && gpu_release
```
Record both KLD numbers in the design doc's results section (or a scratch note). Do not proceed to Task 6 until Step 3 is GO.

---

## Task 6: Replicate to lloyd2 and lloyd4

Write side already works for lloyd2/lloyd4 by reusing the fwht2/fwht4 K kernels; only the dispatch branches and the attention LUT cases are new.

**Files:**
- Modify: `crates/rdna-compute/src/dispatch.rs` (add `kv_write_fwht2_vec` / `kv_write_fwht4_vec` + `_batched`, mirroring Task 2 Step 1 with `KV_CACHE_WRITE_ASYM_K_FWHT{2,4}[_BATCHED]_SRC` and kernel names `kv_cache_write_asym_k_fwht{2,4}`; extend the fused-wrapper `v_mode_bits` branches to `2` and `4`)
- Modify: `kernels/src/attention_flash_fwht{2,3,4}_tile.hip` (+ `_batched`) — add lloyd2/lloyd4 cases to the Phase D branch

- [ ] **Step 1: Add lloyd2/lloyd4 write launchers + dispatch branches**

Mirror Task 2 Step 1 for fwht2 and fwht4 (`shared_mem` and grid identical; only the `_SRC` const + kernel-name string differ). Extend the fused-wrapper branch from Task 2 Step 2 to:
```rust
        match v_mode_bits {
            2 => self.kv_write_fwht2_vec(v_dst, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim),
            3 => self.kv_write_fwht3_vec(v_dst, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim),
            4 => self.kv_write_fwht4_vec(v_dst, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim),
            _ => self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim),
        }
```
Apply to all three fwht fused wrappers (fwht2/3/4, single + batched). **Note:** any K-mode can pair with any V-mode — e.g. fwht2-K with lloyd4-V — so each fwht{2,3,4} fused wrapper must handle all of `v_mode_bits ∈ {2,3,4,8}`.

- [ ] **Step 2: Add lloyd2 + lloyd4 cases to the attention Phase D branch**

In every `attention_flash_fwht{2,3,4}_tile.hip` (+ batched), generalize the `else` (lloyd) arm from Task 3 Step 2 to switch on `v_mode`:

```cpp
    } else {
        const int v_bytes_per_head = 4 + (head_dim * v_mode) / 8;
        const int v_bytes_per_pos  = n_kv_heads * v_bytes_per_head;
        const int v_head_off       = kv_h * v_bytes_per_head;
        const int vbytes_per_thread = v_mode;   // 8 dims * v_mode bits / 8 = v_mode bytes
        for (int t_local = 0; t_local < tile_len; t_local++) {
            float w = scores[t_local];
            int t = tile_start + t_local;
            const unsigned char* vb = v_cache + (size_t)t * v_bytes_per_pos + v_head_off;
            float cnorm = *(const float*)vb;
            const unsigned char* base = vb + 4 + tid * vbytes_per_thread;
            unsigned int packed = (unsigned int)base[0];
            if (v_mode >= 2) packed |= ((unsigned int)base[1] << 8);
            if (v_mode >= 3) packed |= ((unsigned int)base[2] << 16);
            if (v_mode >= 4) packed |= ((unsigned int)base[3] << 24);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float c;
                if (v_mode == 2)      c = TURBO_C2_256[(packed >> (i * 2)) & 3];
                else if (v_mode == 3) c = TURBO_C3_256[(packed >> (i * 3)) & 7];
                else                  c = TURBO_C4_256[(packed >> (i * 4)) & 15];
                out_vec[i] += w * (cnorm * c);
            }
        }
        fwht_shfl_inverse_256(out_vec[0], out_vec[1], out_vec[2], out_vec[3],
                              out_vec[4], out_vec[5], out_vec[6], out_vec[7],
                              signs1, signs2, tid);
    }
```

(`v_mode` is a uniform scalar kernarg → branch is warp-uniform, zero divergence. Replaces the lloyd3-only arm from Task 3.)

- [ ] **Step 3: Build + smoke both new V modes**

```bash
RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet
source scripts/gpu-lock.sh && gpu_acquire "kv-vquant-smoke2"
for V in lloyd2 lloyd4; do
  ./target/release/examples/eval_hipfire --model ~/.hipfire/models/qwen3.6-27b.mq4 \
    --ref ~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin \
    --output benchmarks/quality-baselines/results/2026-05-31-kv-vquant/smoke_fwht3_${V}.kldseq \
    --kv-mode fwht3 --kv-v $V --scoring-mode prefill --max-chunks 4 2>&1 | grep "slice-mean KLD"
done
source scripts/gpu-lock.sh && gpu_release
```
Expected: lloyd4-V KLD ≤ lloyd3-V ≤ (worse) lloyd2-V, all finite. lloyd2-V may be notably worse (expected; that's what the matrix decides).

- [ ] **Step 4: Commit**

```bash
git add crates/rdna-compute/src/dispatch.rs kernels/src/attention_flash_fwht2_tile.hip kernels/src/attention_flash_fwht2_tile_batched.hip kernels/src/attention_flash_fwht3_tile.hip kernels/src/attention_flash_fwht3_tile_batched.hip kernels/src/attention_flash_fwht4_tile.hip kernels/src/attention_flash_fwht4_tile_batched.hip
git commit -m "feat(kv): lloyd2 + lloyd4 V modes (write reuse + attention LUT cases)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: The 12-cell KLD matrix (MI300X)

**Files:**
- Create: `benchmarks/quality-baselines/results/2026-05-31-kv-vquant/` (`.kldseq` files + reduced table)
- Modify: `docs/plans/2026-05-31-kv-vquant-fwht-lloyd-v-design.md` (fill §5 with results)

- [ ] **Step 1: Sync the branch to MI300X and build there**

```bash
ssh mi300 'cd /root/hipfire && git fetch && git checkout feat/kv-vquant-fwht-lloyd-v && git pull && \
  RUSTC_WRAPPER=sccache cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet'
```
Expected: clean build on gfx942. Confirm `~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin` + the 27B model exist on mi300.

- [ ] **Step 2: Run all 12 cells (24-chunk paired)**

```bash
ssh mi300 'cd /root/hipfire && mkdir -p benchmarks/quality-baselines/results/2026-05-31-kv-vquant && \
for K in fwht2 fwht3 fwht4; do for V in q8 lloyd2 lloyd3 lloyd4; do \
  ./target/release/examples/eval_hipfire \
    --model ~/.hipfire/models/qwen3.6-27b.mq4 \
    --ref ~/.hipfire/kldref/qwen3.6-27b-MASTER-small.kldref.bin \
    --output benchmarks/quality-baselines/results/2026-05-31-kv-vquant/27b__${K}__${V}.kldseq \
    --kv-mode $K --kv-v $V --scoring-mode prefill --max-chunks 24 2>&1 \
    | grep "slice-mean KLD" | sed "s/^/${K} x ${V}: /"; \
done; done | tee benchmarks/quality-baselines/results/2026-05-31-kv-vquant/matrix.log'
```
Expected: 12 `K x V: slice-mean KLD = ...` lines. The `V=q8` column should reproduce the historical fwht2/3/4 anchors.

- [ ] **Step 3: Record the matrix + decision**

Fill the design doc §5 with the 12 numbers in the K×V grid alongside the byte-cost grid (§3). Identify: (a) lowest-byte cell with KLD ≈ Q8-V baseline; (b) the equal-byte K/V-split comparisons (e.g. fwht3-K/lloyd4-V vs fwht4-K/lloyd3-V at 232 B); (c) whether the V-bit gradient is steeper than the K-bit gradient (validates "V more sensitive"). Commit:

```bash
git add docs/plans/2026-05-31-kv-vquant-fwht-lloyd-v-design.md benchmarks/quality-baselines/results/2026-05-31-kv-vquant/matrix.log
git commit -m "docs(kv): 12-cell FWHT-K x Lloyd-V KLD matrix results

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(The `kld_reduce.py` table with bootstrap CIs is optional polish: `python3 benchmarks/quality-baselines/harness/kld_reduce.py --result-dir benchmarks/quality-baselines/results/2026-05-31-kv-vquant/ --out-md .../result-table.md` — but its filename parser expects `<variant>__<arch>__<scoring>.kldseq`; rename outputs to that form if you use it.)

---

## Task 8: Coherence gate (mandatory before any claim/default flip)

**Files:** none (validation only).

- [ ] **Step 1: Symlink the 27B mq4 into the gate's model dir if needed**

```bash
ls ~/.hipfire/models/qwen3.6-27b.mq4 || echo "ensure tool-call-27b model present for --full"
```

- [ ] **Step 2: Run the gate on the leading V-quant candidate**

Pick the matrix winner (likely fwht3-K/lloyd3-V or fwht3-K/lloyd4-V). Because the gate selects modes via its own config, run it with the candidate as the configured KV mode (set `HIPFIRE_KV_MODE` / the daemon config the gate uses). At minimum run the standard battery, and if spec-decode is untouched, the dflash gate is not required:

```bash
./scripts/coherence-gate.sh --full 2>&1 | tee /tmp/coherence-vquant.log
echo "exit=$?"
```
Expected: exit `0`; open the `/tmp/coherence-*.md` report and confirm each model is fluent, on-topic, no attractor/loop/special-token leak. **A KLD-parity win that fails coherence is not a win** (per CLAUDE.md + the falsification log).

- [ ] **Step 3: Record the gate result** in the design doc and the PR description.

---

## Task 9 (deferred until matrix + gate pass): production wiring + default decision

Only after Tasks 7-8 confirm a V-quant mode at KLD-parity + coherence. Scope of a follow-up commit/PR:

- Thread `v_mode` into the daemon/serve load path (`crates/hipfire-runtime/examples/daemon.rs`) and the config/CLI (`cli/index.ts`) — decide the user surface (composite mode name like `fwht3_lloyd4` vs a separate `kv_v` key); see design §7 open decision 1.
- Wire `v_mode` into the multi-GPU `*_multi_filtered` ctors (design §7 decision 5) and confirm the CASK `_capped` V path (decision 3).
- Add the single-token **decode** V write for Lloyd-V in the qwen35 decode path (the prefill/batched path is covered by Tasks 2-3; verify the decode site writes V via the same `v_mode` branch — design §7 decision 4).
- Decide and (if warranted) flip the per-arch default; update help text + `docs/plans/2026-05-31-kv-vquant-and-setup-wizard.md`.

---

## Self-review notes (coverage check vs design doc)

- Design §1 (FWHT-V→Lloyd-V, lloyd2/3/4, K/V decoupled): Tasks 1-6. ✅
- Design §2 (V-vs-K asymmetry, single tail inverse): Task 3 Step 2 (per-tile inverse, reduce unchanged). ✅
- Design §3 (byte layout, one cnorm/head): Task 1 Step 3 helper + Task 2 (write reuse) + Task 3 (read layout). ✅
- Design §4 components: 4a write (Task 2, via kernel reuse), 4b attention+inverse (Task 3), 4c allocation (Task 1), 4d KvCache V-mode (Task 1), 4e eval --kv-v (Task 4). ✅
- Design §5 (12-cell KLD matrix): Task 7. ✅
- Design §6 (KLD-parity + coherence guardrails): Tasks 5, 7, 8. ✅
- Design §7 open decisions: deferred to Task 9 (correct — they're production-surface choices, not matrix blockers). ✅

**Type/name consistency:** `VMode` enum + `.bits()`/`v_mode_bits()` (Task 1) used identically in Tasks 2-4; kernarg name `v_mode` (the bit-count) consistent across kernel signature (Task 3.1), Phase D branch (3.2/6.2), and launcher param `v_mode_bits` (3.4). Write methods `kv_write_fwht{2,3,4}_vec[_batched]` consistent between Task 2 (fwht3) and Task 6 (fwht2/4).

**Known risk flagged in-plan:** Task 5 is the GO/NO-GO for the inverse-FWHT/layout correctness; if KLD explodes, the suspects are enumerated there.
