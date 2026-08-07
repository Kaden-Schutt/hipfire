// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Attention, KV cache, DFlash, pflash, triattn, kv_compact, and vision attention dispatch.

use crate::kernels;
use crate::DType;
use crate::Gpu;
use crate::GpuTensor;
use hip_bridge::{DeviceBuffer, HipResult};
use std::ffi::c_void;
use std::sync::OnceLock;

fn q8_flash_default_tile_size(
    arch: &str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> usize {
    // Qwen3.5-0.8B's small full-attention shape is under-parallelized at
    // the generic gfx12 tile of 128. Keep this independently certified
    // specialization shape-bound so larger gfx12 models retain tile128.
    let gfx12_small_dense_shape =
        arch.starts_with("gfx12") && n_heads == 8 && n_kv_heads == 2 && head_dim == 256;
    let gfx1151_radiowave_shape = arch == "gfx1151"
        && n_heads == 16
        && n_kv_heads == 2
        && head_dim == 256
        && max_seq == 2_048;
    if gfx12_small_dense_shape {
        16
    } else if arch == "gfx1100" || gfx1151_radiowave_shape {
        32
    } else {
        128
    }
}

/// Architecture- and shape-aware tile geometry for scalar Q8 decode attention.
///
/// The kernel accepts the tile as a kernarg, so this does not fork codegen.
/// Restricting the surface to powers of two keeps the partial-buffer layout
/// and retained-grid contract straightforward. The gfx1100 default is the
/// stationary winner at tg128. The gfx1151 tile32 default is admitted only
/// for the certified Qwen3.6 A3B Q8 replay shape. The gfx12 tile16 default is
/// likewise limited to its certified Qwen3.5-0.8B shape.
pub fn q8_flash_tile_size(
    arch: &str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> usize {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_Q8_FLASH_TILE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| matches!(value, 16 | 32 | 64 | 128 | 256))
        })
        .unwrap_or_else(|| q8_flash_default_tile_size(arch, n_heads, n_kv_heads, head_dim, max_seq))
}

const V_MODE_Q8: i32 = 8;

#[inline]
fn replay_stable_tile_count(
    actual_tiles: usize,
    max_tiles: usize,
    graph_capture: bool,
    redline_recording: bool,
) -> usize {
    if graph_capture || redline_recording {
        max_tiles
    } else {
        actual_tiles
    }
}

/// Opt-in gate for the WMMA flash-attention prefill path.
fn is_wmma_fa_enabled() -> bool {
    use std::sync::OnceLock;
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_WMMA_FA").map_or(false, |v| v == "1")
    })
}

/// Minimum chunk size to engage the WMMA-FA route.
fn wmma_fa_min_batch() -> usize {
    use std::sync::OnceLock;
    static GATE: OnceLock<usize> = OnceLock::new();
    *GATE.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_WMMA_FA_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
    })
}

impl Gpu {
    /// DSpark bidirectional staging assembly (on-GPU; replaces a host
    /// d2h+assemble+h2d that forced ~2 stream syncs per stage).
    ///
    /// Builds `staged[block, head_dim, stage_w]` from the committed main_kv
    /// ring (`ring`, layout `[n_kv=1, head_dim, win]`) and the per-block KV
    /// (`block_kv`, `[block, kv_dim]`, `kv_dim == head_dim`). All `block` rows
    /// are identical (bidirectional); columns `>= n_committed+block` are zeroed.
    /// Runs on `active_stream` so it pipelines with the rest of the stage.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_stage_kv(
        &mut self,
        ring: &GpuTensor,
        block_kv: &GpuTensor,
        staged: &GpuTensor,
        win: usize,
        kv_dim: usize,
        head_dim: usize,
        n_committed: usize,
        block: usize,
        stage_w: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "dspark_stage_kv",
            kernels::DSPARK_STAGE_KV_SRC,
            "dspark_stage_kv",
        )?;

        let ring_ptr = ring.buf.as_ptr();
        let bkv_ptr = block_kv.buf.as_ptr();
        let st_ptr = staged.buf.as_ptr();
        let win_i = win as i32;
        let kvd = kv_dim as i32;
        let hd = head_dim as i32;
        let nc = n_committed as i32;
        let blk = block as i32;
        let sw = stage_w as i32;

        let mut params: Vec<*mut c_void> = vec![
            &ring_ptr as *const _ as *mut c_void,
            &bkv_ptr as *const _ as *mut c_void,
            &st_ptr as *const _ as *mut c_void,
            &win_i as *const _ as *mut c_void,
            &kvd as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &blk as *const _ as *mut c_void,
            &sw as *const _ as *mut c_void,
        ];

        let threads = 256u32;
        let grid_x = ((head_dim as u32) + threads - 1) / threads;
        self.launch_maybe_blob(
            "dspark_stage_kv",
            [grid_x, block as u32, 1],
            [threads, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ring_ptr);
                b.push_ptr(bkv_ptr);
                b.push_ptr(st_ptr);
                b.push_i32(win_i);
                b.push_i32(kvd);
                b.push_i32(hd);
                b.push_i32(nc);
                b.push_i32(blk);
                b.push_i32(sw);
                b
            },
        )
    }

    /// accs_count: [n_layers * n_heads * n_bands] u64 sample counters.
    /// All accs_* buffers persist across calls; the kernel ADDS into them.
    ///
    /// Grid = [n_heads, n_bands, 1]. Block = [64, 1, 1]. Zero cross-block
    /// contention since each (layer, head, band) is written by exactly one
    /// block at a time (called sequentially per layer per chunk).
    pub fn triattn_accumulate(
        &mut self,
        q_batch: &DeviceBuffer,
        accs_sum_re: &DeviceBuffer,
        accs_sum_im: &DeviceBuffer,
        accs_sum_abs: &DeviceBuffer,
        accs_count: &DeviceBuffer,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        layer_idx: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "triattn_accumulate",
            kernels::TRIATTN_ACCUMULATE_SRC,
            "triattn_accumulate_f32",
        )?;

        let n_bands = head_dim / 2;

        let mut q_ptr = q_batch.as_ptr();
        let mut sre_ptr = accs_sum_re.as_ptr();
        let mut sim_ptr = accs_sum_im.as_ptr();
        let mut sab_ptr = accs_sum_abs.as_ptr();
        let mut cnt_ptr = accs_count.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut li = layer_idx as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut sre_ptr as *mut _ as *mut c_void,
            &mut sim_ptr as *mut _ as *mut c_void,
            &mut sab_ptr as *mut _ as *mut c_void,
            &mut cnt_ptr as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "triattn_accumulate_f32",
            [n_heads as u32, n_bands as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(sre_ptr);
                b.push_ptr(sim_ptr);
                b.push_ptr(sab_ptr);
                b.push_ptr(cnt_ptr);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(li);
                b
            },
        )
    }

    /// GPU-side GQA attention.
    /// pos_buf: GPU buffer with single i32 position. Kernel computes seq_len = pos_buf[0] + 1.
    /// seq_len_hint: host-side seq_len for shared memory sizing (= pos + 1).
    pub fn attention_f32(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("attention", kernels::ATTENTION_SRC, "attention_f32")?;
        let func = &self.functions["attention_f32"];

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        // When a stream is active (graph capture mode), use max_seq for shared mem
        // so the captured graph works for all sequence lengths.
        let effective_seq = if self.active_stream.is_some() {
            max_seq
        } else {
            seq_len_hint
        };
        let block_size = (effective_seq.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((effective_seq + block_size as usize) * 4) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Flash-decoding attention: split KV scan for long sequences.
    /// Automatically chooses single-block or multi-block based on seq_len.
    pub fn attention_flash(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        partials: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Choose chunk size: aim for 4-16 chunks
        let chunk_size = if seq_len <= 128 { seq_len } else { 128 };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;

        // Phase 1: compute partial attention per chunk
        self.ensure_kernel(
            "attention_flash_partial",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_partial",
        )?;

        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let p_ptr = partials.buf.as_ptr();
        let sl = seq_len as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let cs = chunk_size as i32;

        let mut params1: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &p_ptr as *const _ as *mut c_void,
            &sl as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &cs as *const _ as *mut c_void,
        ];

        let block_size = 128u32.min(chunk_size as u32).next_power_of_two();
        let shared_mem = ((chunk_size + block_size as usize) * 4) as u32;

        self.launch_maybe_blob(
            "attention_flash_partial",
            [n_heads as u32, n_chunks as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(p_ptr);
                b.push_i32(sl);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(cs);
                b
            },
        )?;

        // Phase 2: reduce partials
        self.ensure_kernel(
            "attention_flash_reduce",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_reduce",
        )?;

        let p_ptr2 = partials.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let nh2 = n_heads as i32;
        let nc = n_chunks as i32;
        let hd2 = head_dim as i32;

        let mut params2: Vec<*mut c_void> = vec![
            &p_ptr2 as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &nh2 as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &hd2 as *const _ as *mut c_void,
        ];

        let reduce_block = head_dim.min(256) as u32;
        self.launch_maybe_blob(
            "attention_flash_reduce",
            [n_heads as u32, 1, 1],
            [reduce_block, 1, 1],
            0,
            &mut params2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p_ptr2);
                b.push_ptr(out_ptr);
                b.push_i32(nh2);
                b.push_i32(nc);
                b.push_i32(hd2);
                b
            },
        )
    }

    /// GQA-aware split-K flash decode: one phase-1 block per (kv_head, chunk)
    /// reuses a single K/V load across its query-head group (n_heads/n_kv_heads),
    /// so the KV cache is traversed n_kv_heads× not n_heads×. Phase-2 reuses
    /// `attention_flash_reduce`. Same partials buffer as `attention_flash`.
    pub fn attention_flash_gqa(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        partials: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let cs_cap = hipfire_config::developer_var("HIPFIRE_GQA_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let chunk_size = if seq_len <= cs_cap { seq_len } else { cs_cap };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;

        self.ensure_kernel(
            "attention_flash_gqa_partial",
            kernels::ATTENTION_FLASH_GQA_SRC,
            "attention_flash_gqa_partial",
        )?;
        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let p_ptr = partials.buf.as_ptr();
        let sl = seq_len as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let cs = chunk_size as i32;
        let mut p1: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &p_ptr as *const _ as *mut c_void,
            &sl as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &cs as *const _ as *mut c_void,
        ];
        let block = 128u32.min(chunk_size as u32).next_power_of_two();
        let shmem = ((chunk_size + block as usize) * 4) as u32;
        self.launch_maybe_blob(
            "attention_flash_gqa_partial",
            [n_kv_heads as u32, n_chunks as u32, 1],
            [block, 1, 1],
            shmem,
            &mut p1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(p_ptr);
                b.push_i32(sl);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(cs);
                b
            },
        )?;

        self.ensure_kernel(
            "attention_flash_reduce",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_reduce",
        )?;
        let p2_ptr = partials.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let nh2 = n_heads as i32;
        let nc = n_chunks as i32;
        let hd2 = head_dim as i32;
        let mut p2: Vec<*mut c_void> = vec![
            &p2_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &nh2 as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &hd2 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "attention_flash_reduce",
            [n_heads as u32, 1, 1],
            [head_dim.min(256) as u32, 1, 1],
            0,
            &mut p2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p2_ptr);
                b.push_ptr(o_ptr);
                b.push_i32(nh2);
                b.push_i32(nc);
                b.push_i32(hd2);
                b
            },
        )
    }

    /// Single-launch GQA decode: one block per kv_head streams all KV once,
    /// accumulates online-softmax for the group in LDS, writes O. No partials,
    /// no reduce. Grid = n_kv_heads. Probe of launch-vs-occupancy floor.
    pub fn attention_flash_gqa_fused(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        self.ensure_kernel(
            "attention_flash_gqa_fused",
            kernels::ATTENTION_FLASH_GQA_FUSED_SRC,
            "attention_flash_gqa_fused",
        )?;
        let f = &self.functions["attention_flash_gqa_fused"];
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut o_ptr = out.buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let kv_group = n_heads / n_kv_heads;
        let block = 128u32;
        let shmem = ((kv_group * head_dim + block as usize) * 4) as u32;
        let mut p: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                f,
                [n_kv_heads as u32, 1, 1],
                [block, 1, 1],
                shmem,
                self.stream_ref(),
                &mut p,
            )
        }
    }

    /// Warp-cooperative GQA decode attention. One warp per head in the kv-group,
    /// chunked KV processing with partials + reduce.
    /// 3.5x faster than scalar attention_flash on decode (271 -> 77 us).
    /// Grid=[n_kv_heads, n_chunks], block=[kv_group*32].
    pub fn attention_gqa_warp(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        partials: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let cs_cap = hipfire_config::developer_var("HIPFIRE_GQA_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let chunk_size = if seq_len <= cs_cap {
            seq_len
        } else {
            cs_cap.max(128)
        };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;
        let kv_group = n_heads / n_kv_heads;
        let block = (kv_group * 32) as u32;

        self.ensure_kernel(
            "attention_gqa_warp",
            kernels::ATTENTION_GQA_WARP_SRC,
            "attention_gqa_warp",
        )?;
        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let p_ptr = partials.buf.as_ptr();
        let sl = seq_len as i32;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let cs = chunk_size as i32;
        let mut p1: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &p_ptr as *const _ as *mut c_void,
            &sl as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &cs as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "attention_gqa_warp",
            [n_kv_heads as u32, n_chunks as u32, 1],
            [block, 1, 1],
            0,
            &mut p1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(p_ptr);
                b.push_i32(sl);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(cs);
                b
            },
        )?;

        self.ensure_kernel(
            "attention_flash_reduce",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_reduce",
        )?;
        let p2_ptr = partials.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let nh2 = n_heads as i32;
        let nc = n_chunks as i32;
        let hd2 = head_dim as i32;
        let mut p2: Vec<*mut c_void> = vec![
            &p2_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &nh2 as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &hd2 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "attention_flash_reduce",
            [n_heads as u32, 1, 1],
            [head_dim.min(256) as u32, 1, 1],
            0,
            &mut p2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p2_ptr);
                b.push_ptr(o_ptr);
                b.push_i32(nh2);
                b.push_i32(nc);
                b.push_i32(hd2);
                b
            },
        )
    }

    /// Warp-cooperative GQA decode attention with device-side seq_len.
    /// Identical to `attention_gqa_warp` but reads seq_len from a device
    /// pointer instead of a scalar kernarg. Used for hipGraph decode capture.
    pub fn attention_gqa_warp_dv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        partials: &GpuTensor,
        seq_len_buf: &DeviceBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        chunk_size: usize,
        n_chunks: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv_group = n_heads / n_kv_heads;
        let block = (kv_group * 32) as u32;

        self.ensure_kernel(
            "attention_gqa_warp_dv",
            kernels::ATTENTION_GQA_WARP_DV_SRC,
            "attention_gqa_warp_dv",
        )?;
        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let p_ptr = partials.buf.as_ptr();
        let sl_ptr = seq_len_buf.as_ptr();
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = scale;
        let cs = chunk_size as i32;
        let mut p1: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &p_ptr as *const _ as *mut c_void,
            &sl_ptr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &cs as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "attention_gqa_warp_dv",
            [n_kv_heads as u32, n_chunks as u32, 1],
            [block, 1, 1],
            0,
            &mut p1,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(p_ptr);
                b.push_ptr(sl_ptr);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(cs);
                b
            },
        )?;

        self.ensure_kernel(
            "attention_flash_reduce",
            kernels::ATTENTION_FLASH_SRC,
            "attention_flash_reduce",
        )?;
        let p2_ptr = partials.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let nh2 = n_heads as i32;
        let nc = n_chunks as i32;
        let hd2 = head_dim as i32;
        let mut p2: Vec<*mut c_void> = vec![
            &p2_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &nh2 as *const _ as *mut c_void,
            &nc as *const _ as *mut c_void,
            &hd2 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "attention_flash_reduce",
            [n_heads as u32, 1, 1],
            [head_dim.min(256) as u32, 1, 1],
            0,
            &mut p2,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p2_ptr);
                b.push_ptr(o_ptr);
                b.push_i32(nh2);
                b.push_i32(nc);
                b.push_i32(hd2);
                b
            },
        )
    }

    /// Write KV to HFQ4 co-located block (72 bytes per head: scale+zero+nibbles).
    pub fn kv_cache_write_hfq4(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_hfq4",
            kernels::KV_CACHE_WRITE_HFQ4_SRC,
            "kv_cache_write_hfq4",
        )?;
        let func = &self.functions["kv_cache_write_hfq4"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Attention with HFQ4 KV blocks (72 bytes per head, co-located).
    pub fn attention_hfq4_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_hfq4_kv",
            kernels::ATTENTION_HFQ4_KV_SRC,
            "attention_hfq4_kv",
        )?;
        let func = &self.functions["attention_hfq4_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        // scores[seq_len] + ws[block_size] + q_shared[head_dim]
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// INT8 co-located with f16 scale (matches Q8_0 precision, one block per head).
    pub fn kv_cache_write_int8c_f16(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_int8c_f16",
            kernels::KV_CACHE_WRITE_INT8C_F16_SRC,
            "kv_cache_write_int8c_f16",
        )?;
        let func = &self.functions["kv_cache_write_int8c_f16"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    pub fn attention_int8c_f16_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_int8c_f16_kv",
            kernels::ATTENTION_INT8C_F16_KV_SRC,
            "attention_int8c_f16_kv",
        )?;
        let func = &self.functions["attention_int8c_f16_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Write KV to INT8 co-located block (f32 scale + int8 data, symmetric).
    pub fn kv_cache_write_int8c(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_int8c",
            kernels::KV_CACHE_WRITE_INT8C_SRC,
            "kv_cache_write_int8c",
        )?;
        let func = &self.functions["kv_cache_write_int8c"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Attention with INT8 co-located KV blocks.
    pub fn attention_int8c_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_int8c_kv",
            kernels::ATTENTION_INT8C_KV_SRC,
            "attention_int8c_kv",
        )?;
        let func = &self.functions["attention_int8c_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Write KV to HFQ8 cache (FP32 scale+zero, contiguous uint8).
    pub fn kv_cache_write_hfq8(
        &mut self,
        dst_data: &GpuTensor,
        dst_scales: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_hfq8",
            kernels::KV_CACHE_WRITE_HFQ8_SRC,
            "kv_cache_write_hfq8",
        )?;
        let func = &self.functions["kv_cache_write_hfq8"];
        let mut dd = dst_data.buf.as_ptr();
        let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dd as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Attention with HFQ8 KV cache.
    pub fn attention_hfq8_kv(
        &mut self,
        q: &GpuTensor,
        k_data: &GpuTensor,
        k_scales: &GpuTensor,
        v_data: &GpuTensor,
        v_scales: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_hfq8_kv",
            kernels::ATTENTION_HFQ8_KV_SRC,
            "attention_hfq8_kv",
        )?;
        let func = &self.functions["attention_hfq8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kd = k_data.buf.as_ptr();
        let mut ks = k_scales.buf.as_ptr();
        let mut vd = v_data.buf.as_ptr();
        let mut vs = v_scales.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut vd as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Write KV to INT8 cache (separate scale array).
    pub fn kv_cache_write_int8(
        &mut self,
        dst_vals: &GpuTensor,
        dst_scales: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_int8",
            kernels::KV_CACHE_WRITE_INT8_SRC,
            "kv_cache_write_int8",
        )?;
        let func = &self.functions["kv_cache_write_int8"];
        let mut dv = dst_vals.buf.as_ptr();
        let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dv as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Attention with INT8 KV (separate scale array).
    pub fn attention_int8_kv(
        &mut self,
        q: &GpuTensor,
        k_vals: &GpuTensor,
        k_scales: &GpuTensor,
        v_vals: &GpuTensor,
        v_scales: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_int8_kv",
            kernels::ATTENTION_INT8_KV_SRC,
            "attention_int8_kv",
        )?;
        let func = &self.functions["attention_int8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut kv_ptr = k_vals.buf.as_ptr();
        let mut ks_ptr = k_scales.buf.as_ptr();
        let mut vv_ptr = v_vals.buf.as_ptr();
        let mut vs_ptr = v_scales.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void,
            &mut ks_ptr as *mut _ as *mut c_void,
            &mut vv_ptr as *mut _ as *mut c_void,
            &mut vs_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched causal attention: all query positions in one launch.
    /// Q: [seq_len × n_heads × head_dim], K/V: [seq_len × n_kv_heads × head_dim].
    pub fn attention_causal_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_causal_batched",
            kernels::ATTENTION_CAUSAL_BATCHED_SRC,
            "attention_causal_batched",
        )?;
        let func = &self.functions["attention_causal_batched"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // Block size: enough threads to cover head_dim and seq_len
        let block_size = 128u32.min((seq_len.max(head_dim) as u32).next_power_of_two());
        // Shared: scores[seq_len] + workspace[block_size]
        let shared_mem = ((seq_len + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, seq_len as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched decode-with-history attention over an F32 KV cache.
    ///
    /// `q` is a `[batch × n_heads × head_dim]` block of query rows; row `i` lives
    /// at absolute position `base + i`. K/V are read directly from the persistent
    /// F32 cache (`[max_seq × n_kv_heads × head_dim]`, indexed by absolute
    /// position), so each row attends to the FULL history `[0 .. base+i]`. The
    /// caller MUST have written the block's own post-RoPE K/V into the cache at
    /// positions `[base .. base+batch)` before calling, so the in-block causal
    /// prefix is present.
    ///
    /// One block per `(head, row)` running a single-pass softmax over `ctx_len =
    /// base + i + 1` — the same per-token flash-decode math `forward_step` runs
    /// sequentially, but block-parallel. Writes `out` `[batch × n_heads × head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_batched_history(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        batch: usize,
        base: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_decode_batched_history",
            kernels::ATTENTION_DECODE_BATCHED_HISTORY_SRC,
            "attention_decode_batched_history",
        )?;
        let func = &self.functions["attention_decode_batched_history"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut b = batch as i32;
        let mut base_i = base as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut base_i as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // Max context any row attends to (the last block row sees the most).
        let max_ctx = base + batch; // = base + (batch-1) + 1
        let block_size = 128u32.min((max_ctx.max(head_dim) as u32).next_power_of_two());
        // scores[max_ctx] + workspace[block_size]
        let shared_mem = ((max_ctx + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched Q8_0 KV cache write: quantize multiple positions in one launch.
    pub fn kv_cache_write_q8_0_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q8_0_batched",
            kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC,
            "kv_cache_write_q8_0_batched",
        )?;
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = positions.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        self.launch_maybe_blob(
            "kv_cache_write_q8_0_batched",
            [total_blocks, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(d);
                b.push_ptr(s);
                b.push_ptr(p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b
            },
        )
    }

    /// Write KV vector to Q8_0 quantized cache (same format as GGML Q8_0).
    pub fn kv_cache_write_q8_0(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q8_0",
            kernels::KV_CACHE_WRITE_Q8_0_SRC,
            "kv_cache_write_q8_0",
        )?;
        let d = dst.buf.as_ptr();
        let s = src.buf.as_ptr();
        let p = pos_buf.as_ptr();
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &d as *const _ as *mut c_void,
            &s as *const _ as *mut c_void,
            &p as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        let bytes = crate::profile::kv_cache_write_q8_0_bytes(n_kv_heads, head_dim);
        let timer =
            crate::profile::begin_timer(&self.hip, "kv_write", "kv_cache_write_q8_0", bytes);
        let result = self.launch_maybe_blob(
            "kv_cache_write_q8_0",
            [total_blocks, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(d);
                b.push_ptr(s);
                b.push_ptr(p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Exact paired K/V Q8_0 cache write for single-token decode. Uses the
    /// same 32-lane block quantizer as `kv_cache_write_q8_0` and concatenates
    /// the independent K and V block grids into one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_cache_write_q8_0_pair(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const KERNEL: &str = "kv_cache_write_q8_0_pair";
        self.ensure_kernel(
            KERNEL,
            kernels::KV_CACHE_WRITE_Q8_0_PAIR_GFX1100_SRC,
            KERNEL,
        )?;
        let kd = k_dst.buf.as_ptr();
        let vd = v_dst.buf.as_ptr();
        let ks = k_src.buf.as_ptr();
        let vs = v_src.buf.as_ptr();
        let p = pos_buf.as_ptr();
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &ks as *const _ as *mut c_void,
            &vs as *const _ as *mut c_void,
            &p as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        let bytes = crate::profile::kv_cache_write_q8_0_bytes(n_kv_heads, head_dim) * 2;
        let timer = crate::profile::begin_timer(&self.hip, "kv_write", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [total_blocks * 2, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kd);
                b.push_ptr(vd);
                b.push_ptr(ks);
                b.push_ptr(vs);
                b.push_ptr(p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched causal attention with Q8_0 quantized KV cache. Processes N
    /// queries in one launch; each query b has its own causal window read
    /// from positions[b] (i.e. attend to 0..positions[b]+1). Q and out are
    /// [batch_size × n_heads × head_dim] row-major; K/V caches are the same
    /// layout as `attention_q8_0_kv` and must already contain the prefix
    /// through positions[batch_size-1].
    ///
    /// Byte-exact with N single-token calls at batch_size=1, positions[0]=pos.
    ///
    /// `max_ctx_len` is the maximum seq_len = max(positions[b]) + 1 across
    /// the batch; used to size the shared memory allocation for scores[].
    pub fn attention_q8_0_kv_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_q8_0_kv_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            None,
            0,
            0,
        )
    }

    /// Tree-mask variant of `attention_q8_0_kv_batched`. When `tree_bias` is
    /// `Some`, the kernel ignores the causal cutoff and iterates over
    /// `[0, block_start + block_cols)`, applying an additive bias from
    /// `tree_bias[b × block_cols + (t - block_start)]` for in-block keys.
    /// Caller passes `-inf` on non-ancestor slots and `0.0` on ancestors
    /// (see `hipfire_runtime::ddtree::linearize_tree`).
    ///
    /// When `tree_bias` is `None`, `block_start` / `block_cols` are ignored
    /// and behavior is byte-identical to the legacy causal path.
    ///
    /// Shared memory: the tree-mode `seq_len` is always `block_start +
    /// block_cols`. Caller must pass `max_ctx_len` ≥ that value so the
    /// scores[] LDS slice is sized correctly.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_kv_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8_0_kv_batched",
            kernels::ATTENTION_Q8_0_KV_BATCHED_SRC,
            "attention_q8_0_kv_batched",
        )?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        // tree_bias = null when None; the kernel branches on bias != nullptr.
        let mut bias_ptr: *mut std::ffi::c_void = match tree_bias {
            Some(t) => t.buf.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut bs = block_start as i32;
        let mut bc = block_cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut bias_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut bc as *mut _ as *mut c_void,
        ];
        let block_size = (max_ctx_len.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        // Shared memory must accommodate the LARGEST batch row's seq_len for
        // scores[], plus nthreads workspace and head_dim q_shared.
        let shared_mem = ((max_ctx_len + block_size as usize + head_dim) * 4) as u32;
        let bytes =
            crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, max_ctx_len)
                * batch_size;
        let timer =
            crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv_batched", bytes);
        let bias_raw = bias_ptr; // alias for move into closure
        let result = self.launch_maybe_blob(
            "attention_q8_0_kv_batched",
            [n_heads as u32, batch_size as u32, 1],
            [block_size, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                b.push_ptr(k_ptr);
                b.push_ptr(v_ptr);
                b.push_ptr(out_ptr);
                b.push_ptr(pos_ptr);
                b.push_ptr(bias_raw);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b.push_i32(bs);
                b.push_i32(bc);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Query-tiled Q8_0 flash prefill attention.
    ///
    /// `br`/`bc` are compile-time tile sizes; each (br, bc) pair compiles to
    /// its own module so they can be swept without editing the source. LDS is
    /// a function of br/bc only — never of context length — so this kernel has
    /// no capacity crossover and no occupancy decay as the context grows.
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
        // The kernel's per-thread accumulator is a fixed float[32]; dpt must
        // fit it or the kernel would silently overrun its stack array.
        let dpt = head_dim / (NTHREADS / br);
        assert!(
            dpt <= 32,
            "flash prefill dpt {dpt} > 32 (br={br} head_dim={head_dim}); \
             raise NTHREADS or lower br"
        );
        let module = format!("attention_q8_0_flash_prefill_br{br}_bc{bc}");
        let src = format!(
            "#define BR {br}\n#define BC {bc}\n#define NTHREADS {NTHREADS}\n{}",
            kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC
        );
        self.ensure_kernel(&module, &src, "attention_q8_0_flash_prefill")?;

        let bph = head_dim / 32;
        let lds = (br * bc + 3 * br + br * head_dim) * 4 + 2 * bc * bph * 34;
        assert!(
            lds <= 64 * 1024,
            "flash prefill LDS {lds} exceeds 64KB (br={br} bc={bc})"
        );

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
        let _ = max_ctx_len; // cache stride derives from n_kv_heads/head_dim
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

    /// WMMA (matrix-core) variant of `attention_q8_0_flash_prefill`.
    ///
    /// Fixed 16-query / 16-key tiles (the WMMA fragment shape), one wave32 per
    /// workgroup. head_dim must be a multiple of 32 (Q8_0 block width) and at
    /// most 512 so `d_chunks <= MAX_D_CHUNKS`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_flash_prefill_wmma(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 32 == 0,
            "head_dim {head_dim} must be a multiple of 32"
        );
        assert!(
            head_dim <= 256,
            "head_dim {head_dim} exceeds MAX_D_CHUNKS*16"
        );
        // HIPFIRE_FLASH_PREFILL_SPLITQ=1 carries Q as a double-single (hi+lo)
        // pair through two WMMA passes per d-chunk. Q's f16 rounding is the
        // dominant error term — f16 Q alone through an otherwise-f32 kernel
        // gives rel_l2 5.36e-4 of a 5.48e-4 total at CTX=12288 — so this is the
        // accuracy lever. Costs one extra matrix op per d-chunk and a second Q
        // plane in LDS.
        let split_q = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_SPLITQ")
            .ok()
            .as_deref()
            == Some("1");
        let gfx12 = self.arch_caps.has_wmma_w32_gfx12();
        let fixed_head_dim = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_FIXED_HD")
            .ok()
            .as_deref()
            != Some("0");
        let fixed_hd = if fixed_head_dim { head_dim } else { 0 };
        let prefetch_v = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_PREFETCH_V")
            .ok()
            .as_deref()
            != Some("0");
        // gfx12 must see head_dim as a compile-time constant so the unrolled
        // float8_t output fragments stay in VGPRs. Leaving it dynamic spills
        // 544 B/thread to scratch (and reserves ~34 MiB of global scratch on
        // gfx1201), roughly halving this kernel's measured throughput.
        let pv_suffix = if prefetch_v { "" } else { "_pv0" };
        let tuning_suffix = pv_suffix;
        let module = if !fixed_head_dim && split_q {
            format!("attention_q8_0_flash_prefill_wmma_dynamic_splitq{tuning_suffix}")
        } else if !fixed_head_dim {
            format!("attention_q8_0_flash_prefill_wmma_dynamic{tuning_suffix}")
        } else if gfx12 && split_q {
            format!("attention_q8_0_flash_prefill_wmma_gfx12_hd{head_dim}_splitq{tuning_suffix}")
        } else if gfx12 {
            format!("attention_q8_0_flash_prefill_wmma_gfx12_hd{head_dim}{tuning_suffix}")
        } else if split_q {
            format!("attention_q8_0_flash_prefill_wmma_gfx11_hd{head_dim}_splitq{tuning_suffix}")
        } else {
            format!("attention_q8_0_flash_prefill_wmma_gfx11_hd{head_dim}{tuning_suffix}")
        };
        let kernel_src = if gfx12 {
            kernels::ATTENTION_Q8_0_FLASH_PREFILL_WMMA_GFX12_SRC
        } else {
            kernels::ATTENTION_Q8_0_FLASH_PREFILL_WMMA_SRC
        };
        let src = format!(
            "#define SPLIT_Q {}\n#define FIXED_HEAD_DIM {}\n#define PREFETCH_V {}\n{}",
            split_q as u32, fixed_hd, prefetch_v as u32, kernel_src
        );
        self.ensure_kernel(&module, &src, "attention_q8_0_flash_prefill_wmma")?;
        const M_TILE: usize = 16;
        const N_TILE: usize = 16;
        const S_STRIDE: usize = 18;
        const V_STRIDE: usize = 18;
        // MUST track the kernel's LDS layout: SPLIT_Q adds a second Q plane.
        // Under-allocating here overruns V_lds_T / S_lds / m_lds and the kernel
        // silently emits all zeros — observed, and it passed the accuracy gate
        // until that gate was hardened to reject degenerate output.
        let q_planes = if split_q { 2 } else { 1 };
        // Q(xN planes) f16 + V^T f16 + S f16 + (m,l,alpha) f32
        let lds = (q_planes * M_TILE * head_dim + head_dim * V_STRIDE + M_TILE * S_STRIDE) * 2
            + M_TILE * 3 * 4;
        assert!(
            lds <= 64 * 1024,
            "wmma flash prefill LDS {lds} exceeds 64KB"
        );

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
        let grid_x = batch_size.div_ceil(M_TILE) as u32;
        self.launch_maybe_blob(
            "attention_q8_0_flash_prefill_wmma",
            [grid_x, n_heads as u32, 1],
            [32, 1, 1],
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

    /// Batched flash attention for Q8_0 KV — tile + reduce two-kernel path.
    /// No LDS capacity limit: tiles seq_len into chunks of `tile_size` only,
    /// so shared memory is O(tile_size), not O(max_ctx_len). Replaces the
    /// old `attention_q8_0_kv_batched_masked` for long contexts (LDS would
    /// exceed ~64 KB hardware limit past ~15k ctx).
    ///
    /// Q8_0 has no per-quad rotation (unlike asym4/fwht), so `cos_theta` and
    /// `sin_theta` are not needed. The kernel never reads them, so `q` is
    /// passed as a dummy for both — matching the `launch_asym_flash_batched`
    /// signature without special-casing.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_q8_0_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_q8_0_tile_batched",
            kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC,
            "attention_flash_q8_0_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            q, // cos_theta dummy — kernel ignores
            q, // sin_theta dummy — kernel ignores
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            V_MODE_Q8,
            /*window=*/ 0,
            /*force_wmma_grid=*/ false,
        )
    }

    /// Sliding-window variant of [`attention_flash_q8_0_batched_masked`]. A
    /// query at position `p` attends only to keys in `[p-window+1, p]` (the last
    /// `window` keys); `window <= 0` is full causal (identical to the non-windowed
    /// method). Used by cohere2moe's `sliding_attention` layers so context beyond
    /// `sliding_window` clips correctly instead of running full-causal (degraded).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_q8_0_batched_masked_windowed(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_q8_0_tile_batched",
            kernels::ATTENTION_FLASH_Q8_0_TILE_BATCHED_SRC,
            "attention_flash_q8_0_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            q, // cos_theta dummy — kernel ignores
            q, // sin_theta dummy — kernel ignores
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            V_MODE_Q8,
            window,
            /*force_wmma_grid=*/ false,
        )
    }

    /// Flash attention with Q8_0 KV cache — tile + reduce two-kernel path.
    /// Tiles seq_len into chunks of `tile_size`, launches [n_heads, n_tiles]
    /// blocks for the tile kernel, then [n_heads] blocks for the reduce.
    /// Requires a pre-allocated `partials` buffer of size
    /// n_heads * max_tiles * (2 + head_dim) floats.
    pub fn attention_flash_q8_0(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.attention_flash_q8_0_windowed(
            q,
            k_cache,
            v_cache,
            out,
            pos_buf,
            seq_len_hint,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            partials,
            0,
        )
    }

    /// Sliding-window variant of [`attention_flash_q8_0`]: a query at position
    /// `p` attends only to keys in `[p-window+1, p]`. `window <= 0` is full
    /// causal (identical to the non-windowed method). Used by cohere2moe's
    /// `sliding_attention` decode steps once context exceeds `sliding_window`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_q8_0_windowed(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        window: i32,
    ) -> HipResult<()> {
        self.attention_flash_q8_0_windowed_impl(
            q,
            k_cache,
            v_cache,
            out,
            pos_buf,
            seq_len_hint,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            partials,
            window,
            None,
        )
    }

    /// Q8 flash-attention decode with the Qwen output gate and MQ rotation
    /// folded into the reduce epilogue. The tile pass is identical to
    /// `attention_flash_q8_0`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_q8_0_gated_mq_rotate_gfx1100(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        gate: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.attention_flash_q8_0_windowed_impl(
            q,
            k_cache,
            v_cache,
            out,
            pos_buf,
            seq_len_hint,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            partials,
            0,
            Some(gate),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_flash_q8_0_windowed_impl(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        window: i32,
        output_gate: Option<&GpuTensor>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let tile_size = q8_flash_tile_size(&self.arch, n_heads, n_kv_heads, head_dim, max_seq);
        // Graph-safe: use max_tiles so the grid is position-independent.
        // The tile kernel exits early for tiles beyond actual seq_len.
        let max_tiles = (max_seq + tile_size - 1) / tile_size;
        // For profiling / non-graph code paths, the actual tile count:
        let actual_tiles = (seq_len_hint + tile_size - 1) / tile_size;
        // Redline records an immutable launch sequence independently of
        // hipGraph's capture_mode. Its replay updates pos_buf but cannot grow a
        // recorded grid when seq_len crosses a tile boundary, so the
        // recording pass must capture the same max_tiles superset as hipGraph.
        let launch_tiles = replay_stable_tile_count(
            actual_tiles,
            max_tiles,
            self.graphs.capture_mode,
            self.replay.is_recording(),
        );

        // ── Tile kernel ──
        let gfx1151_tile_dpp = self.arch_caps.is_gfx1151()
            && hipfire_config::developer_var("HIPFIRE_GFX1151_ATTENTION_TILE_DPP").as_deref()
                == Ok("1");
        let (tile_module, tile_src) = if gfx1151_tile_dpp {
            (
                "attention_flash_q8_0_tile_dpp_gfx1151",
                kernels::ATTENTION_FLASH_Q8_0_TILE_DPP_GFX1151_SRC,
            )
        } else {
            (
                "attention_flash_q8_0_tile",
                kernels::ATTENTION_FLASH_Q8_0_TILE_SRC,
            )
        };
        self.ensure_kernel(tile_module, tile_src, "attention_flash_q8_0_tile")?;
        {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let q_ptr = q.buf.as_ptr();
            let k_ptr = k_cache.buf.as_ptr();
            let v_ptr = v_cache.buf.as_ptr();
            let p_ptr = partials.buf.as_ptr();
            let pos_ptr = pos_buf.as_ptr();
            let nh = n_heads as i32;
            let nkv = n_kv_heads as i32;
            let hd = head_dim as i32;
            let ms = max_seq as i32;
            let sc = scale;
            let ts = tile_size as i32;
            let wn = window;
            let grid = [n_heads as u32, launch_tiles as u32, 1];
            let shared = ((tile_size + head_dim) * 4) as u32;
            let mut params: Vec<*mut c_void> = vec![
                &q_ptr as *const _ as *mut c_void,
                &k_ptr as *const _ as *mut c_void,
                &v_ptr as *const _ as *mut c_void,
                &p_ptr as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &nkv as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &ms as *const _ as *mut c_void,
                &sc as *const _ as *mut c_void,
                &ts as *const _ as *mut c_void,
                &wn as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob_position_grid(
                "attention_flash_q8_0_tile",
                grid,
                [32, 1, 1],
                shared,
                &mut params,
                1,
                1,
                tile_size as u32,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr);
                    b.push_ptr(k_ptr);
                    b.push_ptr(v_ptr);
                    b.push_ptr(p_ptr);
                    b.push_ptr(pos_ptr);
                    b.push_i32(nh);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b.push_i32(ms);
                    b.push_f32(sc);
                    b.push_i32(ts);
                    b.push_i32(wn);
                    b
                },
            )?;
        }

        // ── Reduce kernel (reads seq_len from pos_buf, computes n_tiles) ──
        {
            let p_ptr = partials.buf.as_ptr();
            let o_ptr = out.buf.as_ptr();
            let nh = n_heads as i32;
            let hd = head_dim as i32;
            let pos_ptr = pos_buf.as_ptr();
            let ts = tile_size as i32;
            let mt = max_tiles as i32;
            if let Some(gate) = output_gate {
                let (module, src, kernel) = if self.arch_caps.is_gfx1151() {
                    (
                        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",
                        kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1151_SRC,
                        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",
                    )
                } else {
                    (
                        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",
                        kernels::ATTENTION_FLASH_Q8_0_REDUCE_GATED_MQ_ROTATE_GFX1100_SRC,
                        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",
                    )
                };
                self.ensure_kernel(module, src, kernel)?;
                self.ensure_mq_signs()?;
                let g_ptr = gate.buf.as_ptr();
                let s1_ptr = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
                let s2_ptr = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
                let mut params: Vec<*mut c_void> = vec![
                    &p_ptr as *const _ as *mut c_void,
                    &o_ptr as *const _ as *mut c_void,
                    &g_ptr as *const _ as *mut c_void,
                    &s1_ptr as *const _ as *mut c_void,
                    &s2_ptr as *const _ as *mut c_void,
                    &nh as *const _ as *mut c_void,
                    &hd as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &ts as *const _ as *mut c_void,
                    &mt as *const _ as *mut c_void,
                ];
                self.launch_maybe_blob(
                    kernel,
                    [n_heads as u32, 1, 1],
                    [256, 1, 1],
                    ((max_tiles + head_dim) * 4) as u32,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(p_ptr);
                        b.push_ptr(o_ptr);
                        b.push_ptr(g_ptr);
                        b.push_ptr(s1_ptr);
                        b.push_ptr(s2_ptr);
                        b.push_i32(nh);
                        b.push_i32(hd);
                        b.push_ptr(pos_ptr);
                        b.push_i32(ts);
                        b.push_i32(mt);
                        b
                    },
                )?;
            } else {
                const KERNEL: &str = "attention_flash_q8_0_reduce";
                self.ensure_kernel(KERNEL, kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC, KERNEL)?;
                let mut params: Vec<*mut c_void> = vec![
                    &p_ptr as *const _ as *mut c_void,
                    &o_ptr as *const _ as *mut c_void,
                    &nh as *const _ as *mut c_void,
                    &hd as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &ts as *const _ as *mut c_void,
                    &mt as *const _ as *mut c_void,
                ];
                self.launch_maybe_blob(
                    KERNEL,
                    [n_heads as u32, 1, 1],
                    [256, 1, 1],
                    (max_tiles * 4) as u32,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(p_ptr);
                        b.push_ptr(o_ptr);
                        b.push_i32(nh);
                        b.push_i32(hd);
                        b.push_ptr(pos_ptr);
                        b.push_i32(ts);
                        b.push_i32(mt);
                        b
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Compile a givens4 kernel — prepends turbo_common + givens_common headers.
    fn ensure_givens4_kernel(
        &mut self,
        name: &str,
        body_src: &str,
        func_name: &str,
    ) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }
        let stripped = body_src
            .replace("#include \"turbo_common.h\"", "")
            .replace("#include \"givens_common.h\"", "");
        let full_src = format!(
            "{}\n{}\n{}",
            kernels::TURBO_COMMON_H,
            kernels::GIVENS_COMMON_SRC,
            stripped
        );
        let obj_path = self.compiler.compile(name, &full_src)?;
        let obj_path_str = obj_path.to_str().unwrap().to_string();
        if !self.modules.contains_key(name) {
            let module = crate::scratch::module_load_or_recompile(
                &self.hip,
                &mut self.compiler,
                name,
                &full_src,
                &obj_path_str,
            )?;
            self.modules.insert(name.to_string(), module);
        }
        let module = &self.modules[name];
        let func = self.hip.module_get_function(module, func_name)?;
        self.functions.insert(func_name.to_string(), func);
        Ok(())
    }

    /// Shared FWHT attention reduction. These launches must use the blob seam:
    /// Redline records only explicit kernarg blobs, and omitting this dependent
    /// reducer silently records an incomplete attention tape.
    #[allow(clippy::too_many_arguments)]
    fn attention_flash_fwht_reduce(
        &mut self,
        partials: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        tile_size: usize,
        max_tiles: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        let p_ptr = partials.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let pos_ptr = pos_buf.as_ptr();
        let ts = tile_size as i32;
        let mt = max_tiles as i32;

        if v_mode_bits != V_MODE_Q8 {
            self.ensure_givens4_kernel(
                "attention_flash_lloyd_reduce",
                kernels::ATTENTION_FLASH_LLOYD_REDUCE_SRC,
                "attention_flash_lloyd_reduce",
            )?;
            let s1_ptr = signs1.buf.as_ptr();
            let s2_ptr = signs2.buf.as_ptr();
            let mut params: Vec<*mut c_void> = vec![
                &p_ptr as *const _ as *mut c_void,
                &o_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void,
                &ts as *const _ as *mut c_void,
                &mt as *const _ as *mut c_void,
                &s1_ptr as *const _ as *mut c_void,
                &s2_ptr as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_lloyd_reduce",
                [n_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(p_ptr);
                    b.push_ptr(o_ptr);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_ptr(pos_ptr);
                    b.push_i32(ts);
                    b.push_i32(mt);
                    b.push_ptr(s1_ptr);
                    b.push_ptr(s2_ptr);
                    b
                },
            )
        } else {
            self.ensure_kernel(
                "attention_flash_q8_0_reduce",
                kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
                "attention_flash_q8_0_reduce",
            )?;
            let mut params: Vec<*mut c_void> = vec![
                &p_ptr as *const _ as *mut c_void,
                &o_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &hd as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void,
                &ts as *const _ as *mut c_void,
                &mt as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_q8_0_reduce",
                [n_heads as u32, 1, 1],
                [256, 1, 1],
                (max_tiles * 4) as u32,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(p_ptr);
                    b.push_ptr(o_ptr);
                    b.push_i32(nh);
                    b.push_i32(hd);
                    b.push_ptr(pos_ptr);
                    b.push_i32(ts);
                    b.push_i32(mt);
                    b
                },
            )
        }
    }

    /// Fused K+V write for asym4: K at givens4 (rotated 4-bit), V at Q8_0 (normal space).
    /// Launches two kernels — K-only givens4 writer + standard Q8_0 writer.
    pub fn kv_cache_write_asym4_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // K: rotated 4-bit
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens4",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC,
            "kv_cache_write_asym_k_givens4",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens4"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        // V: standard Q8_0
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Fused K+V write for fwht4: K at signed-FWHT-rotated 4-bit, V at Q8_0.
    /// Byte-identical storage to asym4_fused — only the K-write kernel differs.
    /// `signs1` and `signs2` are 128-element FP32 ±1 vectors (occupy the same
    /// `givens_cos`/`givens_sin` slots on KvCache when `quant_fwht == true`).
    pub fn kv_cache_write_fwht4_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht4",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_SRC,
            "kv_cache_write_asym_k_fwht4",
        )?;
        {
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_maybe_blob(
                "kv_cache_write_asym_k_fwht4",
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                shared_mem,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(kdp);
                    b.push_ptr(ksp);
                    b.push_ptr(pp);
                    b.push_ptr(s1p);
                    b.push_ptr(s2p);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b
                },
            )?;
        }
        self.kv_write_v_by_mode(
            v_dst,
            v_src,
            pos_buf,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            v_mode_bits,
        )
    }

    /// Fused K+V write for asym3: K at 3-bit rotated (RotorQuant "planar3"), V at Q8_0.
    /// Best-quality rotated K per RotorQuant paper. Head geometry: 32 threads × 8
    /// values = 256 dims single-pass. 100 bytes/head for hd=256.
    pub fn kv_cache_write_asym3_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC,
            "kv_cache_write_asym_k_givens3",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens3"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Launch the fwht3 rotated-centroid write kernel on an arbitrary KV buffer.
    pub fn kv_cache_write_fwht3_vec(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht3",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_SRC,
            "kv_cache_write_asym_k_fwht3",
        )?;
        let mut kdp = dst.buf.as_ptr();
        let mut ksp = src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        self.launch_maybe_blob(
            "kv_cache_write_asym_k_fwht3",
            [n_kv_heads as u32, 1, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kdp);
                b.push_ptr(ksp);
                b.push_ptr(pp);
                b.push_ptr(s1p);
                b.push_ptr(s2p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b
            },
        )?;
        Ok(())
    }

    pub fn kv_cache_write_fwht3_vec_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht3_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht3_batched",
        )?;
        let mut kdp = dst.buf.as_ptr();
        let mut ksp = src.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        self.launch_maybe_blob(
            "kv_cache_write_asym_k_fwht3_batched",
            [n_kv_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kdp);
                b.push_ptr(ksp);
                b.push_ptr(pp);
                b.push_ptr(s1p);
                b.push_ptr(s2p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b
            },
        )
    }

    pub fn kv_cache_write_v256_2bit_vec(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_fwht256_2bit",
            kernels::KV_CACHE_WRITE_FWHT256_2BIT_SRC,
            "kv_cache_write_fwht256_2bit",
        )?;
        let func = &self.functions["kv_cache_write_fwht256_2bit"];
        let mut kdp = dst.buf.as_ptr();
        let mut ksp = src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    pub fn kv_cache_write_v256_2bit_vec_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_fwht256_2bit_batched",
            kernels::KV_CACHE_WRITE_FWHT256_2BIT_BATCHED_SRC,
            "kv_cache_write_fwht256_2bit_batched",
        )?;
        let mut kdp = dst.buf.as_ptr();
        let mut ksp = src.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        self.launch_maybe_blob(
            "kv_cache_write_fwht256_2bit_batched",
            [n_kv_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kdp);
                b.push_ptr(ksp);
                b.push_ptr(pp);
                b.push_ptr(s1p);
                b.push_ptr(s2p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b
            },
        )
    }

    pub fn kv_cache_write_v256_4bit_vec(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_fwht256_4bit",
            kernels::KV_CACHE_WRITE_FWHT256_4BIT_SRC,
            "kv_cache_write_fwht256_4bit",
        )?;
        let func = &self.functions["kv_cache_write_fwht256_4bit"];
        let mut kdp = dst.buf.as_ptr();
        let mut ksp = src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    pub fn kv_cache_write_v256_4bit_vec_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_fwht256_4bit_batched",
            kernels::KV_CACHE_WRITE_FWHT256_4BIT_BATCHED_SRC,
            "kv_cache_write_fwht256_4bit_batched",
        )?;
        let mut kdp = dst.buf.as_ptr();
        let mut ksp = src.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        self.launch_maybe_blob(
            "kv_cache_write_fwht256_4bit_batched",
            [n_kv_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kdp);
                b.push_ptr(ksp);
                b.push_ptr(pp);
                b.push_ptr(s1p);
                b.push_ptr(s2p);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b
            },
        )
    }

    fn kv_write_v_by_mode(
        &mut self,
        v_dst: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        match v_mode_bits {
            2 => self.kv_cache_write_v256_2bit_vec(
                v_dst, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim,
            ),
            3 => self.kv_cache_write_fwht3_vec(
                v_dst, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim,
            ),
            4 => self.kv_cache_write_v256_4bit_vec(
                v_dst, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim,
            ),
            _ => self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim),
        }
    }

    fn kv_write_v_by_mode_batched(
        &mut self,
        v_dst: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        match v_mode_bits {
            2 => self.kv_cache_write_v256_2bit_vec_batched(
                v_dst, v_src, positions, signs1, signs2, n_kv_heads, head_dim, batch_size,
            ),
            3 => self.kv_cache_write_fwht3_vec_batched(
                v_dst, v_src, positions, signs1, signs2, n_kv_heads, head_dim, batch_size,
            ),
            4 => self.kv_cache_write_v256_4bit_vec_batched(
                v_dst, v_src, positions, signs1, signs2, n_kv_heads, head_dim, batch_size,
            ),
            _ => self.kv_cache_write_q8_0_batched(
                v_dst, v_src, positions, n_kv_heads, head_dim, batch_size,
            ),
        }
    }

    pub fn transcode_v_q8_to_lloyd4(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        n_positions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_transcode_v_q8_to_lloyd4",
            kernels::KV_TRANSCODE_V_Q8_TO_LLOYD4_SRC,
            "kv_transcode_v_q8_to_lloyd4",
        )?;
        let func = &self.functions["kv_transcode_v_q8_to_lloyd4"];
        let mut dp = dst.buf.as_ptr();
        let mut sp = src.buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut np = n_positions as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, n_positions as u32, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    pub fn transcode_v_lloyd_down(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        n_positions: usize,
        src_bits: i32,
        dst_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_transcode_v_lloyd_down",
            kernels::KV_TRANSCODE_V_LLOYD_DOWN_SRC,
            "kv_transcode_v_lloyd_down",
        )?;
        let func = &self.functions["kv_transcode_v_lloyd_down"];
        let mut dp = dst.buf.as_ptr();
        let mut sp = src.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut np = n_positions as i32;
        let mut sb = src_bits;
        let mut db = dst_bits;
        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
            &mut sb as *mut _ as *mut c_void,
            &mut db as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, n_positions as u32, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    pub fn transcode_k_fwht4_to_fwht2(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        n_positions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_transcode_k_fwht4_to_fwht2",
            kernels::KV_TRANSCODE_K_FWHT4_TO_FWHT2_SRC,
            "kv_transcode_k_fwht4_to_fwht2",
        )?;
        let func = &self.functions["kv_transcode_k_fwht4_to_fwht2"];
        let mut dp = dst.buf.as_ptr();
        let mut sp = src.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut np = n_positions as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, n_positions as u32, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    pub fn transcode_k_fwht4_to_fwht3(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        n_positions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_transcode_k_fwht4_to_fwht3",
            kernels::KV_TRANSCODE_K_FWHT4_TO_FWHT3_SRC,
            "kv_transcode_k_fwht4_to_fwht3",
        )?;
        let func = &self.functions["kv_transcode_k_fwht4_to_fwht3"];
        let mut dp = dst.buf.as_ptr();
        let mut sp = src.buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr();
        let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut np = n_positions as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, n_positions as u32, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(())
    }

    /// Fused K+V write for fwht3: K at signed-FWHT-256 rotated 3-bit, V at Q8_0.
    /// Byte-identical storage to asym3 — only the K-write kernel differs.
    pub fn kv_cache_write_fwht3_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.kv_cache_write_fwht3_vec(k_dst, k_src, pos_buf, signs1, signs2, n_kv_heads, head_dim)?;
        self.kv_write_v_by_mode(
            v_dst,
            v_src,
            pos_buf,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            v_mode_bits,
        )
    }

    /// Shared helper: launch a batched K-only rotated write kernel.
    fn launch_asym_k_batched(
        &mut self,
        kernel_key: &str,
        src_const: &'static str,
        func_name: &'static str,
        k_dst: &GpuTensor,
        k_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_givens4_kernel(kernel_key, src_const, func_name)?;
        let mut kdp = k_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr();
        let mut pp = positions.buf.as_ptr();
        let mut ctp = cos_theta.buf.as_ptr();
        let mut stp = sin_theta.buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut ctp as *mut _ as *mut c_void,
            &mut stp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        self.launch_maybe_blob(
            func_name,
            [n_kv_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(kdp);
                b.push_ptr(ksp);
                b.push_ptr(pp);
                b.push_ptr(ctp);
                b.push_ptr(stp);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(bs);
                b
            },
        )
    }

    /// Shared helper: launch a batched asym flash tile + the shared asym reduce.
    ///
    /// `tree_bias` / `block_start` / `block_cols` activate DDTree tree-attention
    /// mode (bias added to in-block qk scores; seq_len extends to full cache
    /// including the tree block). When `tree_bias` is None and `block_cols` is
    /// 0, behavior is byte-identical to the legacy causal path.
    #[allow(clippy::too_many_arguments)]
    fn launch_asym_flash_batched(
        &mut self,
        tile_key: &'static str,
        tile_src: &'static str,
        tile_func_name: &'static str,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        v_mode_bits: i32,
        // Sliding-window span for the tile kernel: a query at position p attends
        // only to keys in [p-window+1, p]. <= 0 = full causal (legacy behavior).
        // Pushed as a trailing scalar kernarg; only the q8 tile kernel reads it
        // (asym/lloyd tile kernels declare fewer args and ignore the extra).
        window: i32,
        // When true, use the WMMA grid shape `[n_heads, ceil(chunk/BLOCK_M),
        // max_tiles]` and omit the `v_mode_bits` kernarg, even if the inline
        // `wmma_ok` ladder evaluates to false. Set by the WMMA dispatch
        // wrappers that already know their kernel is WMMA. Scalar callers
        // pass `false` (original behavior).
        force_wmma_grid: bool,
    ) -> HipResult<()> {
        const TILE_SIZE: usize = 128;
        const WMMA_BLOCK_M: usize = 16;
        let max_tiles = (max_ctx_len + TILE_SIZE - 1) / TILE_SIZE;
        let stride = 2 + head_dim;
        let per_pos_bytes = n_heads * max_tiles * stride * 4;
        let partials_capacity = partials.numel() * 4;
        let sub_batch = if per_pos_bytes > 0 {
            (partials_capacity / per_pos_bytes).max(1).min(batch_size)
        } else {
            batch_size
        };

        let wmma_fa_kernel = if self.arch_caps.has_wmma_w32_gfx12() {
            Some((
                "attention_flash_asym4_wmma_tile_batched_gfx12",
                kernels::ATTENTION_FLASH_ASYM4_WMMA_TILE_BATCHED_GFX12_SRC,
                "attention_flash_asym4_wmma_tile_batched_gfx12",
            ))
        } else if self.arch_caps.has_wmma_w32() {
            Some((
                "attention_flash_asym4_wmma_tile_batched",
                kernels::ATTENTION_FLASH_ASYM4_WMMA_TILE_BATCHED_SRC,
                "attention_flash_asym4_wmma_tile_batched",
            ))
        } else {
            None
        };
        let wmma_ok = is_wmma_fa_enabled()
            && wmma_fa_kernel.is_some()
            && (head_dim == 128 || head_dim == 256)
            && tree_bias.is_none()
            && v_mode_bits == V_MODE_Q8
            && tile_func_name == "attention_flash_asym4_tile_batched"
            && batch_size >= wmma_fa_min_batch()
            && batch_size % WMMA_BLOCK_M == 0
            && sub_batch % WMMA_BLOCK_M == 0;
        // `use_wmma_grid` controls grid shape, LDS, and kernarg layout.
        // True when either the inline env-gated ladder fires (scalar→WMMA
        // upgrade) OR the dispatch path explicitly routes to a WMMA variant.
        let use_wmma_grid = wmma_ok || force_wmma_grid;
        let (eff_tile_key, eff_tile_src, eff_tile_func): (
            &'static str,
            &'static str,
            &'static str,
        ) = if wmma_ok {
            wmma_fa_kernel.expect("wmma_ok requires a selected WMMA-FA kernel")
        } else {
            (tile_key, tile_src, tile_func_name)
        };

        self.ensure_givens4_kernel(eff_tile_key, eff_tile_src, eff_tile_func)?;
        if v_mode_bits != V_MODE_Q8 {
            self.ensure_givens4_kernel(
                "attention_flash_lloyd_reduce_batched",
                kernels::ATTENTION_FLASH_LLOYD_REDUCE_BATCHED_SRC,
                "attention_flash_lloyd_reduce_batched",
            )?;
        } else {
            self.ensure_kernel(
                "attention_flash_asym_reduce_batched",
                kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC,
                "attention_flash_asym_reduce_batched",
            )?;
        }

        let q_dim = n_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut offset = 0usize;
        while offset < batch_size {
            let chunk = (batch_size - offset).min(sub_batch);
            {
                let q_ptr =
                    unsafe { (q.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void };
                let k_ptr = k_cache.buf.as_ptr();
                let v_ptr = v_cache.buf.as_ptr();
                let p_ptr = partials.buf.as_ptr();
                let pos_ptr = positions.buf.as_ptr();
                let ct_ptr = cos_theta.buf.as_ptr();
                let st_ptr = sin_theta.buf.as_ptr();
                let bias_ptr: *mut std::ffi::c_void = match tree_bias {
                    Some(t) => t.buf.as_ptr(),
                    None => std::ptr::null_mut(),
                };
                let nh = n_heads as i32;
                let nkv = n_kv_heads as i32;
                let hd = head_dim as i32;
                let ms = max_seq as i32;
                let sc = scale;
                let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32;
                let bc = block_cols as i32;
                let vm = v_mode_bits;
                let wn = window;
                let mut params: Vec<*mut c_void> = vec![
                    &q_ptr as *const _ as *mut c_void,
                    &k_ptr as *const _ as *mut c_void,
                    &v_ptr as *const _ as *mut c_void,
                    &p_ptr as *const _ as *mut c_void,
                    &pos_ptr as *const _ as *mut c_void,
                    &ct_ptr as *const _ as *mut c_void,
                    &st_ptr as *const _ as *mut c_void,
                    &bias_ptr as *const _ as *mut c_void,
                    &nh as *const _ as *mut c_void,
                    &nkv as *const _ as *mut c_void,
                    &hd as *const _ as *mut c_void,
                    &ms as *const _ as *mut c_void,
                    &sc as *const _ as *mut c_void,
                    &ts as *const _ as *mut c_void,
                    &mt as *const _ as *mut c_void,
                    &bo as *const _ as *mut c_void,
                    &bs as *const _ as *mut c_void,
                    &bc as *const _ as *mut c_void,
                ];
                if !use_wmma_grid {
                    params.push(&vm as *const _ as *mut c_void);
                    params.push(&wn as *const _ as *mut c_void);
                }
                let (grid, lds_bytes): ([u32; 3], u32) = if use_wmma_grid {
                    let m_tiles = (chunk + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M;
                    ([n_heads as u32, m_tiles as u32, max_tiles as u32], 0)
                } else {
                    (
                        [n_heads as u32, max_tiles as u32, chunk as u32],
                        (TILE_SIZE * 4) as u32,
                    )
                };
                self.launch_maybe_blob(
                    eff_tile_func,
                    grid,
                    [32, 1, 1],
                    lds_bytes,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(q_ptr);
                        b.push_ptr(k_ptr);
                        b.push_ptr(v_ptr);
                        b.push_ptr(p_ptr);
                        b.push_ptr(pos_ptr);
                        b.push_ptr(ct_ptr);
                        b.push_ptr(st_ptr);
                        b.push_ptr(bias_ptr);
                        b.push_i32(nh);
                        b.push_i32(nkv);
                        b.push_i32(hd);
                        b.push_i32(ms);
                        b.push_f32(sc);
                        b.push_i32(ts);
                        b.push_i32(mt);
                        b.push_i32(bo);
                        b.push_i32(bs);
                        b.push_i32(bc);
                        if !use_wmma_grid {
                            b.push_i32(vm);
                            b.push_i32(wn);
                        }
                        b
                    },
                )?;
            }
            {
                let p_ptr = partials.buf.as_ptr();
                let o_ptr =
                    unsafe { (out.buf.as_ptr() as *mut u8).add(offset * q_dim * 4) as *mut c_void };
                let pos_ptr = positions.buf.as_ptr();
                let nh = n_heads as i32;
                let hd = head_dim as i32;
                let ts = TILE_SIZE as i32;
                let mt = max_tiles as i32;
                let bo = offset as i32;
                let bs = block_start as i32;
                let bc = block_cols as i32;
                if v_mode_bits != V_MODE_Q8 {
                    let s1_ptr = cos_theta.buf.as_ptr();
                    let s2_ptr = sin_theta.buf.as_ptr();
                    let mut params: Vec<*mut c_void> = vec![
                        &p_ptr as *const _ as *mut c_void,
                        &o_ptr as *const _ as *mut c_void,
                        &pos_ptr as *const _ as *mut c_void,
                        &nh as *const _ as *mut c_void,
                        &hd as *const _ as *mut c_void,
                        &ts as *const _ as *mut c_void,
                        &mt as *const _ as *mut c_void,
                        &bo as *const _ as *mut c_void,
                        &bs as *const _ as *mut c_void,
                        &bc as *const _ as *mut c_void,
                        &s1_ptr as *const _ as *mut c_void,
                        &s2_ptr as *const _ as *mut c_void,
                    ];
                    self.launch_maybe_blob(
                        "attention_flash_lloyd_reduce_batched",
                        [n_heads as u32, chunk as u32, 1],
                        [32, 1, 1],
                        0,
                        &mut params,
                        || {
                            let mut b = hip_bridge::KernargBlob::new();
                            b.push_ptr(p_ptr);
                            b.push_ptr(o_ptr);
                            b.push_ptr(pos_ptr);
                            b.push_i32(nh);
                            b.push_i32(hd);
                            b.push_i32(ts);
                            b.push_i32(mt);
                            b.push_i32(bo);
                            b.push_i32(bs);
                            b.push_i32(bc);
                            b.push_ptr(s1_ptr);
                            b.push_ptr(s2_ptr);
                            b
                        },
                    )?;
                } else {
                    let mut params: Vec<*mut c_void> = vec![
                        &p_ptr as *const _ as *mut c_void,
                        &o_ptr as *const _ as *mut c_void,
                        &pos_ptr as *const _ as *mut c_void,
                        &nh as *const _ as *mut c_void,
                        &hd as *const _ as *mut c_void,
                        &ts as *const _ as *mut c_void,
                        &mt as *const _ as *mut c_void,
                        &bo as *const _ as *mut c_void,
                        &bs as *const _ as *mut c_void,
                        &bc as *const _ as *mut c_void,
                    ];
                    self.launch_maybe_blob(
                        "attention_flash_asym_reduce_batched",
                        [n_heads as u32, chunk as u32, 1],
                        [32, 1, 1],
                        0,
                        &mut params,
                        || {
                            let mut b = hip_bridge::KernargBlob::new();
                            b.push_ptr(p_ptr);
                            b.push_ptr(o_ptr);
                            b.push_ptr(pos_ptr);
                            b.push_i32(nh);
                            b.push_i32(hd);
                            b.push_i32(ts);
                            b.push_i32(mt);
                            b.push_i32(bo);
                            b.push_i32(bs);
                            b.push_i32(bc);
                            b
                        },
                    )?;
                }
            }
            offset += chunk;
        }
        Ok(())
    }

    /// Batched K+V write for asym4 (K 4-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym4_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens4_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC,
            "kv_cache_write_asym_k_givens4_batched",
            k_dst,
            k_src,
            positions,
            cos_theta,
            sin_theta,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for fwht4 (K FWHT-rotated 4-bit + V Q8_0).
    /// Same launch geometry as asym4_batched; only the kernel name + sign-vector
    /// param semantics differ.
    pub fn kv_cache_write_fwht4_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_fwht4_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht4_batched",
            k_dst,
            k_src,
            positions,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_write_v_by_mode_batched(
            v_dst,
            v_src,
            positions,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            batch_size,
            v_mode_bits,
        )
    }

    /// Batched K+V write for asym2 (K 2-bit rotated + V Q8_0).
    pub fn kv_cache_write_asym2_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_givens2_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC,
            "kv_cache_write_asym_k_givens2_batched",
            k_dst,
            k_src,
            positions,
            cos_theta,
            sin_theta,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for fwht2 (K FWHT-rotated 2-bit + V Q8_0).
    pub fn kv_cache_write_fwht2_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_k_batched(
            "kv_cache_write_asym_k_fwht2_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_BATCHED_SRC,
            "kv_cache_write_asym_k_fwht2_batched",
            k_dst,
            k_src,
            positions,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            batch_size,
        )?;
        self.kv_write_v_by_mode_batched(
            v_dst,
            v_src,
            positions,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            batch_size,
            v_mode_bits,
        )
    }

    /// Batched flash attention for asym4 (K 4-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym4_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
        )
    }

    /// Tree-mask variant of `attention_flash_asym4_batched`. See
    /// `attention_q8_0_kv_batched_masked` and `ddtree::linearize_tree` for the
    /// bias layout. Passes `tree_bias` / `block_start` / `block_cols` into the
    /// tile + reduce kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym4_tile_batched",
            kernels::ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC,
            "attention_flash_asym4_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            V_MODE_Q8,
            /*window=*/ 0,
            /*force_wmma_grid=*/ false,
        )
    }

    /// WMMA-accelerated batched flash attention for asym4 + Q8-V.
    /// Same parameter layout as `attention_flash_asym4_batched_masked` but uses
    /// the WMMA tile kernel. Caller must ensure: head_dim in {128, 256},
    /// tree_bias is None, batch_size >= 16 and divisible by 16.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_wmma_tile_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym4_wmma_tile_batched",
            kernels::ATTENTION_FLASH_ASYM4_WMMA_TILE_BATCHED_SRC,
            "attention_flash_asym4_wmma_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            V_MODE_Q8,
            /*window=*/ 0,
            /*force_wmma_grid=*/ true,
        )
    }

    /// WMMA-accelerated batched flash attention for asym4 + Q8-V (gfx12 variant).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym4_wmma_tile_batched_gfx12(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym4_wmma_tile_batched_gfx12",
            kernels::ATTENTION_FLASH_ASYM4_WMMA_TILE_BATCHED_GFX12_SRC,
            "attention_flash_asym4_wmma_tile_batched_gfx12",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            V_MODE_Q8,
            /*window=*/ 0,
            /*force_wmma_grid=*/ true,
        )
    }

    /// Batched flash attention for fwht4 (K FWHT-rotated 4-bit + V Q8_0).
    /// `signs1` and `signs2` occupy the same slots as cos_theta/sin_theta on
    /// the asym4 path — the helper passes them opaquely to the tile kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht4_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_fwht4_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
            V_MODE_Q8,
        )
    }

    /// Tree-mask variant of `attention_flash_fwht4_batched`. Mirrors the asym4
    /// path one-for-one; the FA tile kernel is the only difference.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht4_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht4_tile_batched",
            kernels::ATTENTION_FLASH_FWHT4_TILE_BATCHED_SRC,
            "attention_flash_fwht4_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            v_mode_bits,
            /*window=*/ 0,
            /*force_wmma_grid=*/ false,
        )
    }

    /// Batched flash attention for asym2 (K 2-bit rotated + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym2_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym2_tile_batched",
            kernels::ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC,
            "attention_flash_asym2_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
            V_MODE_Q8,
            /*window=*/ 0,
            /*force_wmma_grid=*/ false,
        )
    }

    /// Batched flash attention for fwht2 (K FWHT-rotated 2-bit + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht2_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht2_tile_batched",
            kernels::ATTENTION_FLASH_FWHT2_TILE_BATCHED_SRC,
            "attention_flash_fwht2_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
            v_mode_bits,
            /*window=*/ 0,
            /*force_wmma_grid=*/ false,
        )
    }

    /// Batched K+V write for asym3 — processes N positions in one launch.
    /// K-only givens3 write (batched) + Q8_0 V write (batched).
    pub fn kv_cache_write_asym3_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // K: batched 3-bit rotated write.
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens3_batched",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC,
            "kv_cache_write_asym_k_givens3_batched",
        )?;
        {
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = positions.buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut bs = batch_size as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut bs as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_maybe_blob(
                "kv_cache_write_asym_k_givens3_batched",
                [n_kv_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                shared_mem,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(kdp);
                    b.push_ptr(ksp);
                    b.push_ptr(pp);
                    b.push_ptr(ctp);
                    b.push_ptr(stp);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b.push_i32(bs);
                    b
                },
            )?;
        }
        // V: batched Q8_0 write.
        self.kv_cache_write_q8_0_batched(v_dst, v_src, positions, n_kv_heads, head_dim, batch_size)
    }

    /// Batched K+V write for fwht3 (K FWHT-rotated 3-bit + V Q8_0).
    pub fn kv_cache_write_fwht3_batched(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.kv_cache_write_fwht3_vec_batched(
            k_dst, k_src, positions, signs1, signs2, n_kv_heads, head_dim, batch_size,
        )?;
        self.kv_write_v_by_mode_batched(
            v_dst,
            v_src,
            positions,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            batch_size,
            v_mode_bits,
        )
    }

    /// Batched flash attention for asym3 KV.
    /// Grid: [n_heads, max_tiles, sub_batch] tile + [n_heads, sub_batch] reduce,
    /// chunked by partials buffer capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_asym3_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
        )
    }

    /// Tree-mask variant of `attention_flash_asym3_batched`. asym3 is the
    /// default live KV path on 9B MQ4 — this is the primary target for
    /// DDTree batched verify on the hybrid arch.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_asym3_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_asym3_tile_batched",
            kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC,
            "attention_flash_asym3_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            cos_theta,
            sin_theta,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            V_MODE_Q8,
            /*window=*/ 0,
            /*force_wmma_grid=*/ false,
        )
    }

    /// Batched flash attention for fwht3 (K FWHT-rotated 3-bit + V Q8_0).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht3_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.attention_flash_fwht3_batched_masked(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            None,
            0,
            0,
            V_MODE_Q8,
        )
    }

    /// Tree-mask variant of `attention_flash_fwht3_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_fwht3_batched_masked(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        max_ctx_len: usize,
        batch_size: usize,
        partials: &GpuTensor,
        tree_bias: Option<&GpuTensor>,
        block_start: usize,
        block_cols: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.launch_asym_flash_batched(
            "attention_flash_fwht3_tile_batched",
            kernels::ATTENTION_FLASH_FWHT3_TILE_BATCHED_SRC,
            "attention_flash_fwht3_tile_batched",
            q,
            k_cache,
            v_cache,
            out,
            positions,
            signs1,
            signs2,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq,
            max_ctx_len,
            batch_size,
            partials,
            tree_bias,
            block_start,
            block_cols,
            v_mode_bits,
            /*window=*/ 0,
            /*force_wmma_grid=*/ false,
        )
    }

    /// Flash attention for asym3 KV (K at 3-bit rotated, V at Q8_0).
    /// Reuses Q8_0 flash reduce (output in normal space — V was un-rotated).
    /// Flash attention for fwht3 KV (K FWHT-rotated 3-bit, V at Q8_0).
    pub fn attention_flash_fwht3(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = replay_stable_tile_count(
            actual_tiles,
            max_tiles,
            self.graphs.capture_mode,
            self.replay.is_recording(),
        );

        self.ensure_givens4_kernel(
            "attention_flash_fwht3_tile",
            kernels::ATTENTION_FLASH_FWHT3_TILE_SRC,
            "attention_flash_fwht3_tile",
        )?;
        {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut s1_ptr = signs1.buf.as_ptr();
            let mut s2_ptr = signs2.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut vm = v_mode_bits;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut s1_ptr as *mut _ as *mut c_void,
                &mut s2_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
                &mut vm as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_fwht3_tile",
                [n_heads as u32, launch_tiles as u32, 1],
                [32, 1, 1],
                (TILE_SIZE * 4) as u32,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr);
                    b.push_ptr(k_ptr);
                    b.push_ptr(v_ptr);
                    b.push_ptr(p_ptr);
                    b.push_ptr(pos_ptr);
                    b.push_ptr(s1_ptr);
                    b.push_ptr(s2_ptr);
                    b.push_i32(nh);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b.push_i32(ms);
                    b.push_f32(sc);
                    b.push_i32(ts);
                    b.push_i32(mt);
                    b.push_i32(vm);
                    b
                },
            )?;
        }

        self.attention_flash_fwht_reduce(
            partials,
            out,
            pos_buf,
            signs1,
            signs2,
            n_heads,
            head_dim,
            TILE_SIZE,
            max_tiles,
            v_mode_bits,
        )?;
        Ok(())
    }

    pub fn attention_flash_asym3(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        self.ensure_givens4_kernel(
            "attention_flash_asym3_tile",
            kernels::ATTENTION_FLASH_ASYM3_TILE_SRC,
            "attention_flash_asym3_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym3_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let func = &self.functions["attention_flash_q8_0_reduce"];
            let mut p_ptr = partials.buf.as_ptr();
            let mut o_ptr = out.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, 1, 1],
                    [256, 1, 1],
                    (max_tiles * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Fused K+V write for asym2: K at givens2 (rotated 2-bit), V at Q8_0 (normal space).
    pub fn kv_cache_write_asym2_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_givens2",
            kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC,
            "kv_cache_write_asym_k_givens2",
        )?;
        {
            let func = &self.functions["kv_cache_write_asym_k_givens2"];
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut ctp = cos_theta.buf.as_ptr();
            let mut stp = sin_theta.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut ctp as *mut _ as *mut c_void,
                &mut stp as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_kv_heads as u32, 1, 1],
                    [32, 1, 1],
                    shared_mem,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        self.kv_cache_write_q8_0(v_dst, v_src, pos_buf, n_kv_heads, head_dim)
    }

    /// Fused K+V write for fwht2: K at FWHT-rotated 2-bit, V at Q8_0.
    pub fn kv_cache_write_fwht2_fused(
        &mut self,
        k_dst: &GpuTensor,
        v_dst: &GpuTensor,
        k_src: &GpuTensor,
        v_src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        n_kv_heads: usize,
        head_dim: usize,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "kv_cache_write_asym_k_fwht2",
            kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_SRC,
            "kv_cache_write_asym_k_fwht2",
        )?;
        {
            let mut kdp = k_dst.buf.as_ptr();
            let mut ksp = k_src.buf.as_ptr();
            let mut pp = pos_buf.as_ptr();
            let mut s1p = signs1.buf.as_ptr();
            let mut s2p = signs2.buf.as_ptr();
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut kdp as *mut _ as *mut c_void,
                &mut ksp as *mut _ as *mut c_void,
                &mut pp as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared_mem = ((head_dim + 32) * 4) as u32;
            self.launch_maybe_blob(
                "kv_cache_write_asym_k_fwht2",
                [n_kv_heads as u32, 1, 1],
                [32, 1, 1],
                shared_mem,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(kdp);
                    b.push_ptr(ksp);
                    b.push_ptr(pp);
                    b.push_ptr(s1p);
                    b.push_ptr(s2p);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b
                },
            )?;
        }
        self.kv_write_v_by_mode(
            v_dst,
            v_src,
            pos_buf,
            signs1,
            signs2,
            n_kv_heads,
            head_dim,
            v_mode_bits,
        )
    }

    /// Flash attention for asym4 KV (K at rotated 4-bit, V at Q8_0 normal space).
    /// Reuses the Q8_0 flash reduce since V was un-rotated — no inverse rotation needed.
    pub fn attention_flash_asym4(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        // Tile kernel
        self.ensure_givens4_kernel(
            "attention_flash_asym4_tile",
            kernels::ATTENTION_FLASH_ASYM4_TILE_SRC,
            "attention_flash_asym4_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym4_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32, // scores[tile_size]
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        // Reuse Q8_0 flash reduce (output already in normal space).
        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let func = &self.functions["attention_flash_q8_0_reduce"];
            let mut p_ptr = partials.buf.as_ptr();
            let mut o_ptr = out.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, 1, 1],
                    [256, 1, 1],
                    (max_tiles * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Flash attention for fwht4 KV (K FWHT-rotated 4-bit, V at Q8_0 normal space).
    /// Same launch geometry + Q8_0 reduce as asym4 — only the tile kernel differs.
    pub fn attention_flash_fwht4(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = replay_stable_tile_count(
            actual_tiles,
            max_tiles,
            self.graphs.capture_mode,
            self.replay.is_recording(),
        );

        self.ensure_givens4_kernel(
            "attention_flash_fwht4_tile",
            kernels::ATTENTION_FLASH_FWHT4_TILE_SRC,
            "attention_flash_fwht4_tile",
        )?;
        {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut s1_ptr = signs1.buf.as_ptr();
            let mut s2_ptr = signs2.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut vm = v_mode_bits;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut s1_ptr as *mut _ as *mut c_void,
                &mut s2_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
                &mut vm as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_fwht4_tile",
                [n_heads as u32, launch_tiles as u32, 1],
                [32, 1, 1],
                (TILE_SIZE * 4) as u32,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr);
                    b.push_ptr(k_ptr);
                    b.push_ptr(v_ptr);
                    b.push_ptr(p_ptr);
                    b.push_ptr(pos_ptr);
                    b.push_ptr(s1_ptr);
                    b.push_ptr(s2_ptr);
                    b.push_i32(nh);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b.push_i32(ms);
                    b.push_f32(sc);
                    b.push_i32(ts);
                    b.push_i32(mt);
                    b.push_i32(vm);
                    b
                },
            )?;
        }

        self.attention_flash_fwht_reduce(
            partials,
            out,
            pos_buf,
            signs1,
            signs2,
            n_heads,
            head_dim,
            TILE_SIZE,
            max_tiles,
            v_mode_bits,
        )?;
        Ok(())
    }

    /// Flash attention for asym2 KV (K at rotated 2-bit, V at Q8_0 normal space).
    /// Flash attention for fwht2 KV (K FWHT-rotated 2-bit, V at Q8_0).
    pub fn attention_flash_fwht2(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        signs1: &GpuTensor,
        signs2: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
        v_mode_bits: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = replay_stable_tile_count(
            actual_tiles,
            max_tiles,
            self.graphs.capture_mode,
            self.replay.is_recording(),
        );

        self.ensure_givens4_kernel(
            "attention_flash_fwht2_tile",
            kernels::ATTENTION_FLASH_FWHT2_TILE_SRC,
            "attention_flash_fwht2_tile",
        )?;
        {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut s1_ptr = signs1.buf.as_ptr();
            let mut s2_ptr = signs2.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut vm = v_mode_bits;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut s1_ptr as *mut _ as *mut c_void,
                &mut s2_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
                &mut vm as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                "attention_flash_fwht2_tile",
                [n_heads as u32, launch_tiles as u32, 1],
                [32, 1, 1],
                (TILE_SIZE * 4) as u32,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr);
                    b.push_ptr(k_ptr);
                    b.push_ptr(v_ptr);
                    b.push_ptr(p_ptr);
                    b.push_ptr(pos_ptr);
                    b.push_ptr(s1_ptr);
                    b.push_ptr(s2_ptr);
                    b.push_i32(nh);
                    b.push_i32(nkv);
                    b.push_i32(hd);
                    b.push_i32(ms);
                    b.push_f32(sc);
                    b.push_i32(ts);
                    b.push_i32(mt);
                    b.push_i32(vm);
                    b
                },
            )?;
        }

        self.attention_flash_fwht_reduce(
            partials,
            out,
            pos_buf,
            signs1,
            signs2,
            n_heads,
            head_dim,
            TILE_SIZE,
            max_tiles,
            v_mode_bits,
        )?;
        Ok(())
    }

    pub fn attention_flash_asym2(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const TILE_SIZE: usize = 128;
        let max_tiles = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let actual_tiles = (seq_len_hint + TILE_SIZE - 1) / TILE_SIZE;
        let launch_tiles = if self.graphs.capture_mode {
            max_tiles
        } else {
            actual_tiles
        };

        self.ensure_givens4_kernel(
            "attention_flash_asym2_tile",
            kernels::ATTENTION_FLASH_ASYM2_TILE_SRC,
            "attention_flash_asym2_tile",
        )?;
        {
            let func = &self.functions["attention_flash_asym2_tile"];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let mut q_ptr = q.buf.as_ptr();
            let mut k_ptr = k_cache.buf.as_ptr();
            let mut v_ptr = v_cache.buf.as_ptr();
            let mut p_ptr = partials.buf.as_ptr();
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ct_ptr = cos_theta.buf.as_ptr();
            let mut st_ptr = sin_theta.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut nkv = n_kv_heads as i32;
            let mut hd = head_dim as i32;
            let mut ms = max_seq as i32;
            let mut sc = scale;
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut k_ptr as *mut _ as *mut c_void,
                &mut v_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ct_ptr as *mut _ as *mut c_void,
                &mut st_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut ms as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, launch_tiles as u32, 1],
                    [32, 1, 1],
                    (TILE_SIZE * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        self.ensure_kernel(
            "attention_flash_q8_0_reduce",
            kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC,
            "attention_flash_q8_0_reduce",
        )?;
        {
            let func = &self.functions["attention_flash_q8_0_reduce"];
            let mut p_ptr = partials.buf.as_ptr();
            let mut o_ptr = out.buf.as_ptr();
            let mut nh = n_heads as i32;
            let mut hd = head_dim as i32;
            let mut pos_ptr = pos_buf.as_ptr();
            let mut ts = TILE_SIZE as i32;
            let mut mt = max_tiles as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut pos_ptr as *mut _ as *mut c_void,
                &mut ts as *mut _ as *mut c_void,
                &mut mt as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_heads as u32, 1, 1],
                    [256, 1, 1],
                    (max_tiles * 4) as u32,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        Ok(())
    }

    /// Attention with Q8_0 quantized KV cache.
    pub fn attention_q8_0_kv(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8_0_kv",
            kernels::ATTENTION_Q8_0_KV_SRC,
            "attention_q8_0_kv",
        )?;
        let func = &self.functions["attention_q8_0_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // Fixed-size launch: block 256, LDS = (ATT_Q8_TILE + head_dim + block)
        // floats regardless of seq_len (v2 kernel tiles positions with online
        // softmax — O(seq_len) LDS previously faulted past ~15.9k on gfx1100).
        const ATT_Q8_KV_TILE: usize = 2048;
        let block_size: u32 = 256;
        let shared_mem = ((ATT_Q8_KV_TILE + head_dim + block_size as usize) * 4) as u32;
        let bytes =
            crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, seq_len_hint);
        let timer = crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Phase-instrumented variant of attention_q8_0_kv. Identical to the
    /// baseline kernel but additionally writes per-head cycle counts for
    /// each internal phase into `cycle_counts` (layout: [n_heads * 3],
    /// per-head order = phase1(QK^T), phase2(softmax), phase3(V-weighted)).
    ///
    /// Uses __builtin_amdgcn_s_memrealtime() which returns a wall-clock
    /// counter. On gfx1100 the tick rate is approximately 1e8 Hz (10 ns
    /// per tick); confirm empirically by comparing against the kernel's
    /// total elapsed time from event timing.
    ///
    /// Use only for diagnostic profiling — the memrealtime reads serialize
    /// execution and inflate total time slightly.
    pub fn attention_q8_0_kv_timed(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        cycle_counts: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8_0_kv_timed",
            kernels::ATTENTION_Q8_0_KV_TIMED_SRC,
            "attention_q8_0_kv_timed",
        )?;
        let func = &self.functions["attention_q8_0_kv_timed"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut cc_ptr = cycle_counts.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cc_ptr as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over a Q8 post-RoPE K cache.
    /// Produces one score per cached position per query head; caller picks
    /// top-B for eviction (see arXiv:2604.04921 §4).
    ///
    /// `centers`: `[n_heads × n_bands × 3]` float32 packed as
    /// `(Re(E[q_f]), Im(E[q_f]), E[||q_f||])`. `scores`: `[n_heads × seq_len]`
    /// float32 output. One block per (pos, head); 32 threads reduce across
    /// the head's frequency bands.
    pub fn triattn_score_q8(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "triattn_score_q8",
            kernels::TRIATTN_SCORE_Q8_SRC,
            "triattn_score_q8",
        )?;
        let func = &self.functions["triattn_score_q8"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over an asym2 post-RoPE K cache.
    /// Same shape as `triattn_score_asym3` but reads the 2-bit packed
    /// layout (4 indices per byte) and the TURBO_C2_256 codebook.
    pub fn triattn_score_asym2(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym2",
            kernels::TRIATTN_SCORE_ASYM2_SRC,
            "triattn_score_asym2",
        )?;
        let func = &self.functions["triattn_score_asym2"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over an asym4 post-RoPE K cache.
    /// Same shape as `triattn_score_asym3` but reads the 4-bit nibble
    /// layout and the TURBO_C4 codebook.
    pub fn triattn_score_asym4(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym4",
            kernels::TRIATTN_SCORE_ASYM4_SRC,
            "triattn_score_asym4",
        )?;
        let func = &self.functions["triattn_score_asym4"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// TriAttention importance scoring over an asym3 post-RoPE K cache.
    /// Same contract as `triattn_score_q8` but reads asym3's Givens-rotated
    /// 3-bit layout and applies the inverse Givens rotation on the fly to
    /// recover post-RoPE K per band.
    pub fn triattn_score_asym3(
        &mut self,
        k_cache: &GpuTensor,
        centers: &GpuTensor,
        cos_theta: &GpuTensor,
        sin_theta: &GpuTensor,
        scores: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        p_q: f32,
        seq_len: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel(
            "triattn_score_asym3",
            kernels::TRIATTN_SCORE_ASYM3_SRC,
            "triattn_score_asym3",
        )?;
        let func = &self.functions["triattn_score_asym3"];
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut c_ptr = centers.buf.as_ptr();
        let mut ct_ptr = cos_theta.buf.as_ptr();
        let mut st_ptr = sin_theta.buf.as_ptr();
        let mut s_ptr = scores.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut nr = n_rot as i32;
        let mut th = rope_theta;
        let mut pq = p_q;
        let mut sl = seq_len as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut ct_ptr as *mut _ as *mut c_void,
            &mut st_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut th as *mut _ as *mut c_void,
            &mut pq as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [seq_len as u32, n_heads as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Compact a KV cache row by row: writes `dst[new_pos] = src[retain[new_pos]]`
    /// for `new_pos` in `[0, budget)`. Works for any byte-addressable cache
    /// layout — pass the layout's bytes-per-position.
    ///
    /// `retain_indices` must live on the device. Caller allocates `dst` with
    /// at least `budget × bytes_per_pos` bytes of capacity.
    pub fn kv_compact_gather(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        retain_indices: &GpuTensor,
        bytes_per_pos: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_compact_gather",
            kernels::KV_COMPACT_GATHER_SRC,
            "kv_compact_gather",
        )?;
        let func = &self.functions["kv_compact_gather"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut rp = retain_indices.buf.as_ptr();
        let mut bpp = bytes_per_pos as i32;
        let mut b = budget as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut bpp as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
        ];
        // Choose thread count to saturate per-row bandwidth: ~1 thread per
        // 16-byte chunk, capped at 256 threads per block.
        let threads = ((bytes_per_pos / 16) as u32).clamp(32, 256);
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, 1, 1],
                [threads, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for Q8_0 KV cache (arXiv:2604.10900).
    ///
    /// Computes `budget` output rows from `budget × m` source rows via
    /// weighted average + per-block requantization. Core (singleton)
    /// slots are handled uniformly by the caller: set `src_indices[s×m]`
    /// to the core source position and `src_weights[s×m] = 1.0`, rest = 0.
    ///
    /// All tensors live on the device. Caller allocates `dst` with at
    /// least `budget × n_kv × n_blocks × 34` bytes.
    pub fn kv_fold_q8(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor, // [budget × m] i32
        src_weights: &GpuTensor, // [budget × m] f32
        n_kv: usize,
        n_blocks: usize,
        m: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("kv_fold_q8", kernels::KV_FOLD_Q8_SRC, "kv_fold_q8")?;
        let func = &self.functions["kv_fold_q8"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut nb = n_blocks as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, n_kv as u32, n_blocks as u32],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym3 K (givens3).
    ///
    /// Same calling convention as `kv_fold_q8` but takes `head_dim` (whole head)
    /// since asym3 doesn't block-wise split. One thread block per
    /// (slot, kv_head), 32 threads.
    pub fn kv_fold_asym3(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,
        src_weights: &GpuTensor,
        n_kv: usize,
        head_dim: usize,
        m: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym3", kernels::KV_FOLD_ASYM3_SRC, "kv_fold_asym3")?;
        let func = &self.functions["kv_fold_asym3"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, n_kv as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym4 K (givens4).
    pub fn kv_fold_asym4(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,
        src_weights: &GpuTensor,
        n_kv: usize,
        head_dim: usize,
        m: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym4", kernels::KV_FOLD_ASYM4_SRC, "kv_fold_asym4")?;
        let func = &self.functions["kv_fold_asym4"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, n_kv as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// CASK m-folding merge for asym2 K (givens2).
    pub fn kv_fold_asym2(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        src_indices: &GpuTensor,
        src_weights: &GpuTensor,
        n_kv: usize,
        head_dim: usize,
        m: usize,
        budget: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_givens4_kernel("kv_fold_asym2", kernels::KV_FOLD_ASYM2_SRC, "kv_fold_asym2")?;
        let func = &self.functions["kv_fold_asym2"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ip = src_indices.buf.as_ptr();
        let mut wp = src_weights.buf.as_ptr();
        let mut nkv = n_kv as i32;
        let mut hd = head_dim as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [budget as u32, n_kv as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Write KV vector to Q8 (int8 symmetric) quantized cache.
    pub fn kv_cache_write_q8(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q8",
            kernels::KV_CACHE_WRITE_Q8_SRC,
            "kv_cache_write_q8",
        )?;
        let func = &self.functions["kv_cache_write_q8"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [block, 1, 1],
                shared,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Attention with Q8 quantized KV cache.
    pub fn attention_q8kv(
        &mut self,
        q: &GpuTensor,
        k_cache_q8: &GpuTensor,
        v_cache_q8: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q8kv",
            kernels::ATTENTION_Q8KV_SRC,
            "attention_q8kv",
        )?;
        let func = &self.functions["attention_q8kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache_q8.buf.as_ptr();
        let mut v_ptr = v_cache_q8.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Write KV vector to quantized HFQ4 cache.
    pub fn kv_cache_write_q4(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_q4",
            kernels::KV_CACHE_WRITE_Q4_SRC,
            "kv_cache_write_q4",
        )?;
        let func = &self.functions["kv_cache_write_q4"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 2 * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_kv_heads as u32, 1, 1],
                [block, 1, 1],
                shared,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Attention with quantized HFQ4 KV cache — dequantizes K/V on the fly.
    pub fn attention_q4kv(
        &mut self,
        q: &GpuTensor,
        k_cache_q4: &GpuTensor,
        v_cache_q4: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_q4kv",
            kernels::ATTENTION_Q4KV_SRC,
            "attention_q4kv",
        )?;
        let func = &self.functions["attention_q4kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache_q4.buf.as_ptr();
        let mut v_ptr = v_cache_q4.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32)
            .next_power_of_two()
            .min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// GPU-side KV cache write. Copies kv_dim floats from src to dst[pos_buf[0] * kv_dim].
    pub fn kv_cache_write(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        kv_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write",
            kernels::KV_CACHE_WRITE_SRC,
            "kv_cache_write",
        )?;
        let func = &self.functions["kv_cache_write"];

        let dst_ptr = dst.buf.as_ptr();
        let src_ptr = src.buf.as_ptr();
        let pos_ptr = pos_buf.as_ptr();
        let kd = kv_dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &dst_ptr as *const _ as *mut c_void,
            &src_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = (kv_dim as u32 + block - 1) / block;

        self.launch_maybe_blob(
            "kv_cache_write",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(dst_ptr);
                b.push_ptr(src_ptr);
                b.push_ptr(pos_ptr);
                b.push_i32(kd);
                b
            },
        )
    }

    /// Batched F32 KV-cache write: scatter `batch_size` rows of `src`
    /// (`[batch_size * kv_dim]`) into the F32 cache at the absolute
    /// positions in `positions` (`[batch_size]` i32), in one launch.
    /// Batched-prefill replacement for the per-position `kv_cache_write`.
    pub fn kv_cache_write_f32_batched(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        positions: &GpuTensor,
        kv_dim: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "kv_cache_write_f32_batched",
            kernels::KV_CACHE_WRITE_F32_BATCHED_SRC,
            "kv_cache_write_f32_batched",
        )?;
        let func = &self.functions["kv_cache_write_f32_batched"];

        let mut dst_ptr = dst.buf.as_ptr();
        let mut src_ptr = src.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        let mut kd = kv_dim as i32;
        let mut bs = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dst_ptr as *mut _ as *mut c_void,
            &mut src_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid_x = (kv_dim as u32 + block - 1) / block;

        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Fused ViT self-attention: reads QKV [N, 3*hidden], writes out [N, hidden].
    pub fn vit_attention_f32(
        &mut self,
        qkv: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "vit_attention_f32",
            kernels::VIT_ATTENTION_SRC,
            "vit_attention_f32",
        )?;
        let func = &self.functions["vit_attention_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = qkv.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = std::cmp::min(256, std::cmp::max(n, head_dim)) as u32;
        let block_size = block_size.next_power_of_two();
        // Shared memory: scores[N] + workspace[block_size]
        let shared_mem = ((n + block_size as usize) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, n as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Apply 2D rotary positional embedding to the Q and K halves of a packed
    /// QKV buffer for the Qwen3.5-VL vision tower (V is left untouched).
    ///
    /// `cos_t` and `sin_t` are shaped `[N, head_dim/2]` and are looked up
    /// per-(token, d) pair; the kernel reuses the same scalar for both
    /// `d < head_dim/2` and `d + head_dim/2` halves (HF concatenates
    /// `(rotary_pos_emb, rotary_pos_emb)` along the last dim before the
    /// trig table, so the two halves see the same angle).
    ///
    /// Grid=[num_heads, N], Block=[head_dim/2].
    pub fn apply_rope_2d_vision_f32(
        &mut self,
        qkv: &GpuTensor,
        cos_t: &GpuTensor,
        sin_t: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "apply_rope_2d_vision",
            kernels::APPLY_ROPE_2D_VISION_SRC,
            "apply_rope_2d_vision_f32",
        )?;
        let func = &self.functions["apply_rope_2d_vision_f32"];
        let mut qp = qkv.buf.as_ptr();
        let mut cp = cos_t.buf.as_ptr();
        let mut sp = sin_t.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let half = (head_dim / 2) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, n as u32, 1],
                [half, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Optimized vision attention with tiled K/V loading and 4 queries per block.
    /// ~3-5x faster than vit_attention_f32 via shared memory reuse.
    /// Grid=[num_heads, ceil(N/4)], Block=[256].
    pub fn vit_attention_opt(
        &mut self,
        qkv: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "vit_attention_opt",
            kernels::VIT_ATTENTION_OPT_SRC,
            "vit_attention_opt",
        )?;
        let func = &self.functions["vit_attention_opt"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = qkv.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut hi = hidden as i32;
        let mut nh = num_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let qpb = 2u32;
        let grid_y = ((n as u32 + qpb - 1) / qpb) as u32;
        // LDS: K_TILE * head_dim * 4 + N * 4 + 256 * 4
        let k_tile = 64u32;
        let shared_mem = (k_tile * head_dim as u32 * 4) + (n as u32 * 4) + (256 * 4);
        unsafe {
            self.hip.launch_kernel(
                func,
                [num_heads as u32, grid_y, 1],
                [256, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// DFlash draft cross-attention: `B` queries attend to `L` keys/values
    /// with NO causal mask (bidirectional). Supports GQA; `n_heads` must be
    /// a multiple of `n_kv_heads`. See `kernels/src/attention_dflash.hip`
    /// for the full contract.
    ///
    /// Layouts:
    ///   q : [B * n_heads    * head_dim]
    ///   k : [L * n_kv_heads * head_dim]
    ///   v : [L * n_kv_heads * head_dim]
    ///   out: [B * n_heads    * head_dim]
    pub fn attention_dflash_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "attention_dflash_f32",
            kernels::ATTENTION_DFLASH_SRC,
            "attention_dflash_f32",
        )?;
        let func = &self.functions["attention_dflash_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Tiled online-softmax (FlashAttention-style). LDS layout:
        //   tile_scores[tile_size] + ws[block_size] + out_run[head_dim]
        //
        // tile_size is chosen to keep LDS ≤ 56 KB (8 KB margin under gfx1100's
        // 64 KB hard limit for kernel launch overhead). Single-tile case
        // (l ≤ tile_size) is mathematically equivalent to the prior
        // single-pass softmax up to FP order; multi-tile carries (max, sum,
        // out) running state across tiles. Replaces the prior `scores[L]`
        // allocation that overflowed LDS at l > ~16128.
        let block_size = std::cmp::min(256, std::cmp::max(l, head_dim)) as u32;
        let block_size = block_size.next_power_of_two();
        const LDS_BUDGET_F32: usize = 14_336; // 56 KB / 4 bytes
        let fixed = block_size as usize + head_dim;
        let max_tile_room = LDS_BUDGET_F32.saturating_sub(fixed);
        let tile_size = std::cmp::min(l.max(1), max_tile_room.max(1));
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut ts = tile_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut ts as *mut _ as *mut c_void,
        ];
        let shared_mem = ((tile_size + block_size as usize + head_dim) * 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, b as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// WMMA-accelerated FlashAttention-style non-causal attention for
    /// the **large-B / large-L** case. Same Q/K/V layout and contract as
    /// [`Self::attention_dflash_f32`] — drop-in replacement.
    ///
    /// Grid:  `[n_heads, ceil(B / 16), 1]` (one block per (head, 16-Q-tile))
    /// Block: 32 threads (1 wave32 warp)
    /// LDS:   `(32 * head_dim + 256 + 48) * 4` bytes
    ///        — ≈ 17 KB for `head_dim=128`, fits comfortably under the
    ///        64 KB RDNA3 budget.
    ///
    /// Intended for `B >= 16` and `head_dim` a multiple of 16. The
    /// caller is responsible for picking between this and the scalar
    /// `attention_dflash_f32` based on workload shape.
    pub fn attention_dflash_wmma_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 16 == 0,
            "attention_dflash_wmma_f32: head_dim={head_dim} must be a multiple of 16 \
             (WMMA tiles K-axis in 16-element chunks)",
        );
        assert!(
            head_dim <= 256,
            "attention_dflash_wmma_f32: head_dim={head_dim} exceeds the 256 cap \
             — LDS budget is `3 * 16 * head_dim + 304` f32 slots, which overflows \
             the 64 KB RDNA3 wave32 limit above head_dim=256. Use \
             attention_dflash_f32 (scalar) for larger head_dim, or split this \
             kernel's LDS layout (drop Q_lds or O_lds) in a future variant.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        // gfx12/RDNA4 uses a distinct WMMA lowering (`_w32_gfx12` intrinsic);
        // the gfx11 `_w32` kernel does not compile on gfx12. Route to the
        // gfx12 sister there, base kernel on RDNA3/RDNA3.5.
        let (kernel_name, kernel_src, symbol) = if self.arch_caps.has_wmma_w32_gfx12() {
            (
                "attention_dflash_wmma_f32_gfx12",
                kernels::ATTENTION_DFLASH_WMMA_GFX12_SRC,
                "attention_dflash_wmma_f32_gfx12",
            )
        } else {
            (
                "attention_dflash_wmma_f32",
                kernels::ATTENTION_DFLASH_WMMA_SRC,
                "attention_dflash_wmma_f32",
            )
        };
        self.ensure_kernel(kernel_name, kernel_src, symbol)?;
        let func = &self.functions[kernel_name];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32 slots):
        //   Q_lds[16 * head_dim] + V_lds[16 * head_dim] + O_lds[16 * head_dim]
        //   + S_lds[16 * 16]
        //   + m_lds[16] + l_lds[16] + alpha_lds[16]
        let lds_f32 = 3 * 16 * head_dim + 16 * 16 + 16 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 15) / 16;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [32, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// FlashAttention-style WMMA with M=32 query tile (vs M=16 in
    /// `attention_dflash_wmma_f32`). Two waves per block; doubles the
    /// queries served per K-tile load, halving global-memory K
    /// traffic at vision-encoder shapes (large B, large L, head_dim ≤
    /// 128). Same head_dim ≤ 128 ceiling here — LDS budget is
    /// `(2*32 + 16) * head_dim + 32*16 + 96` f32 slots = 43 KB at
    /// hd=128, which is the largest tile that fits the 64 KB RDNA3
    /// wave32 SLM cap with full Q_lds + O_lds + V_lds.
    ///
    /// Caller responsibility: dispatch this when `B >= 32` AND
    /// `head_dim ≤ 128`; fall back to the M=16 variant or the scalar
    /// `attention_dflash_f32` otherwise.
    pub fn attention_dflash_wmma_m32_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 16 == 0,
            "attention_dflash_wmma_m32_f32: head_dim={head_dim} must be a multiple of 16",
        );
        assert!(
            head_dim <= 128,
            "attention_dflash_wmma_m32_f32: head_dim={head_dim} exceeds the 128 cap — \
             LDS budget at head_dim=160 is 53.4 KB and at head_dim=256 is 84 KB which \
             exceeds the 64 KB RDNA3 wave32 limit. Fall back to attention_dflash_wmma_f32 \
             (M=16) for larger head_dim.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m32_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        // gfx12/RDNA4 uses the `_w32_gfx12` WMMA lowering; the gfx11 kernel
        // does not compile on gfx12. Base kernel on RDNA3/RDNA3.5.
        let (m32_name, m32_src, m32_sym) = if self.arch_caps.has_wmma_w32_gfx12() {
            (
                "attention_dflash_wmma_m32_f32_gfx12",
                kernels::ATTENTION_DFLASH_WMMA_M32_GFX12_SRC,
                "attention_dflash_wmma_m32_f32_gfx12",
            )
        } else {
            (
                "attention_dflash_wmma_m32_f32",
                kernels::ATTENTION_DFLASH_WMMA_M32_SRC,
                "attention_dflash_wmma_m32_f32",
            )
        };
        self.ensure_kernel(m32_name, m32_src, m32_sym)?;
        let func = &self.functions[m32_name];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32 slots):
        //   Q_lds[32 * head_dim] + V_lds[16 * head_dim] + O_lds[32 * head_dim]
        //   + S_lds[32 * 16]
        //   + m_lds[32] + l_lds[32] + alpha_lds[32]
        let lds_f32 = (2 * 32 + 16) * head_dim + 32 * 16 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// FlashAttention-style WMMA with M=32 query tile and **N=64 K-tile
    /// width** (vs N=16 in `attention_dflash_wmma_m32_f32`). Q lives in
    /// registers across all K-tiles within a block; phase C fuses the
    /// alpha-scale of O with the SV epilogue.
    ///
    /// Targets the vision-encoder regime (large B, large L,
    /// head_dim ≤ 128) where rocprof shows the M=32 baseline is
    /// per-tile-fixed-cost bound (1220 K-tile visits at N=16 → 305 at
    /// N=64 means 4× fewer syncs / softmax passes / O-scaling passes).
    ///
    /// LDS at hd=128: V_lds[64*128] + O_lds[32*128] + S_lds[32*64] +
    /// scalars = 57.7 KB (under 64 KB RDNA3 wave32 cap). VGPR per lane
    /// ≈ 130 (Q_frags + s_acc + scratch).
    ///
    /// Caller responsibility: dispatch when `head_dim % 32 == 0`,
    /// `head_dim ≤ 128`. Falls back to M=32 or M=16 otherwise.
    pub fn attention_dflash_wmma_n64_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_n64_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128. The dc loop is fully unrolled with d_chunks=8 \
             so Q_frags[] gets register-promoted instead of spilled to scratch — making \
             it variable would re-introduce the 544 B/lane private segment that defeats \
             the Q-in-registers optimization (the v1 attempt regressed +19%). Fall back \
             to attention_dflash_wmma_m32_f32 (head_dim <= 128) or attention_dflash_wmma_f32 \
             (head_dim <= 256) for other head dims.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_n64_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_n64_f32",
            kernels::ATTENTION_DFLASH_WMMA_N64_SRC,
            "attention_dflash_wmma_n64_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_n64_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32 slots):
        //   V_lds[64 * head_dim] + O_lds[32 * head_dim]
        //   + S_lds[32 * 64]
        //   + m_lds[32] + l_lds[32] + alpha_lds[32]
        let lds_f32 = (64 + 32) * head_dim + 32 * 64 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// FlashAttention-style WMMA, M=32 query tile and N=64 K-tile,
    /// with **K and V already stored as f16 in DRAM** (Q and output
    /// stay f32). Halves the attention kernel's DRAM traffic for K and
    /// V — the dominant cost on memory-bound vision-encoder shapes.
    /// Caller must cast K and V to f16 once (via `cast_f32_to_f16`)
    /// before invoking this kernel.
    ///
    /// Same head_dim==128 restriction as `attention_dflash_wmma_n64_f32`
    /// (Q_frags register-promotion requires the dc loop fully unrolled).
    pub fn attention_dflash_wmma_n64_f16kv_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_n64_f16kv_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n64_f16kv_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n64_f16kv_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_n64_f16kv_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_n64_f16kv_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (same constraint as the f32-K/V sibling — the dc \
             loop is fully unrolled with d_chunks=8 so Q_frags register-promotes).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_n64_f16kv_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_n64_f16kv_f32",
            kernels::ATTENTION_DFLASH_WMMA_N64_F16KV_SRC,
            "attention_dflash_wmma_n64_f16kv_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_n64_f16kv_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout same as the f32-K/V sibling: V_lds stays f32 so
        // phase C is byte-identical between the two kernels.
        let lds_f32 = (64 + 32) * head_dim + 32 * 64 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// FlashAttention WMMA, M=32 query tile and **N=128 K-tile**, K and
    /// V f16 in DRAM, V_lds and S_lds in f16. Same shape as
    /// `attention_dflash_wmma_n64_f16kv_f32` but twice the K-tile width.
    /// Halves outer-loop iterations → halves __syncthreads / softmax /
    /// alpha-scale overhead per attention call.
    ///
    /// LDS at hd=128 ≈ 56.4 KB: V_lds[128*128] f16 (32 KB) +
    /// O_lds[32*128] f32 (16 KB) + S_lds[32*128] f16 (8 KB) + scalars.
    pub fn attention_dflash_wmma_n128_f16kv_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_n128_f16kv_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n128_f16kv_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_n128_f16kv_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_n128_f16kv_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_n128_f16kv_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (d_chunks=8 unroll for register-promoted Q_frags).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_n128_f16kv_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_n128_f16kv_f32",
            kernels::ATTENTION_DFLASH_WMMA_N128_F16KV_SRC,
            "attention_dflash_wmma_n128_f16kv_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_n128_f16kv_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32-equivalent slots — V_lds and S_lds are f16
        // so they take half the slot count of their nominal element
        // count):
        //   V_lds[128 * head_dim] f16     = 128 * head_dim / 2 f32 slots
        //   O_lds[32  * head_dim] f32     =  32 * head_dim     f32 slots
        //   S_lds[32  * 128]      f16     =  32 * 128 / 2      f32 slots
        //   m_lds + l_lds + alpha_lds     =  96                f32 slots
        let lds_f32 = (128 * head_dim) / 2 + 32 * head_dim + (32 * 128) / 2 + 32 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 31) / 32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [64, 1, 1], // 2 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// FlashAttention WMMA, **M=64** query tile + N=128 K-tile, K/V f16
    /// in DRAM, V_lds and S_lds in f16, **O register-resident**.
    /// 4-wave block (128 threads). Halves the query-block count vs
    /// M=32, which halves K and V DRAM traffic per attention call —
    /// the dominant cost on this DRAM-bound workload.
    ///
    /// LDS at hd=128 ≈ 48.8 KB: V_lds[128*128] f16 + S_lds[64*128] f16
    /// + scalars. No O_lds — O lives in per-lane register arrays
    /// (8 float8_t = 64 VGPRs/lane in WMMA frag_c layout).
    pub fn attention_dflash_wmma_m64_n128_f16kv_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (d_chunks=8 unroll, O_frags[8] register array).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (in f32-equivalent slots; V_lds and S_lds are f16
        // so they take half their nominal element count):
        //   V_lds[128 * head_dim]  f16   = 128 * head_dim / 2 f32 slots
        //   S_lds[64  * 128]       f16   =  64 * 128 / 2      f32 slots
        //   m + l + alpha (64 each)      = 192                f32 slots
        let lds_f32 = (128 * head_dim) / 2 + (64 * 128) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1], // 4 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// V3 of `attention_dflash_wmma_m64_n128_f16kv_f32`. Same shape
    /// as v2 (M=64, N=128, 4-wave block, f16 K/V, O in registers,
    /// padded S_lds, cooperative softmax) but with phase C reordered
    /// to outer c / inner dc so each `a_reg_sm` row chunk is read
    /// once per c instead of once per (dc, c). 8× reduction in phase
    /// C S_lds reads.
    pub fn attention_dflash_wmma_m64_n128_f16kv_v3_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_v3_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v3_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Same LDS layout as v2 (padded S_lds stride 130).
        let lds_f32 = (128 * head_dim) / 2 + (64 * 130) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Causal variant of `attention_dflash_wmma_m64_n128_f16kv_v3_f32`.
    /// Same tile shape (M=64, N=128, f16 K/V, 4-wave block, padded S_lds,
    /// cooperative softmax, phase C hoisted) but applies a causal mask
    /// during Phase A: S[q, k] = -inf when k > q. Tiles where all keys
    /// are in the future (kt_start >= q_start + m_tile) are skipped
    /// entirely.
    ///
    /// Intended for text-decoder prefill (causal self-attention with GQA).
    pub fn attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_module, kernel_src) = if self.arch_caps.has_wmma_w32_gfx12() {
            (
                "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32_rdna4",
                kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_CAUSAL_GFX12_SRC,
            )
        } else if self.arch_caps.has_wmma_w32() {
            (
                "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32_rdna3",
                kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V3_CAUSAL_SRC,
            )
        } else {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32 requires wave32 WMMA; \
                     arch={} has neither gfx11 has_wmma_w32 nor gfx12 has_wmma_w32_gfx12. \
                     Use attention_causal_batched for this arch.",
                    self.arch
                ),
            ));
        };
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            kernel_module,
            kernel_src,
            "attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let lds_f32 = (128 * head_dim) / 2 + (64 * 130) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut causal = 1i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut causal as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Vision attention winner (M=64, V_tile=32, f16 K/V, 2 WG/CU).
    /// V_tile=32 stages V in 4 v_chunks per K-tile, keeping LDS at 25.6 KB.
    pub fn attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n32_f16kv_v5_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n32_f16kv_v5_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n32_f16kv_v5_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n32_f16kv_v5_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n32_f16kv_v5_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n32_f16kv_v5_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        let (kernel_name, kernel_src, symbol) = if self.arch_caps.has_wmma_w32_gfx12() {
            (
                "attention_dflash_wmma_m64_n32_f16kv_v5_f32_gfx12",
                kernels::ATTENTION_DFLASH_WMMA_M64_N32_F16KV_V5_GFX12_SRC,
                "attention_dflash_wmma_m64_n32_f16kv_v5_f32_gfx12",
            )
        } else if self.arch_caps.has_wmma_w32() {
            (
                "attention_dflash_wmma_m64_n32_f16kv_v5_f32",
                kernels::ATTENTION_DFLASH_WMMA_M64_N32_F16KV_V5_SRC,
                "attention_dflash_wmma_m64_n32_f16kv_v5_f32",
            )
        } else {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "attention_dflash_wmma_m64_n32_f16kv_v5_f32 requires wave32 WMMA; \
                     arch={} does not support it.",
                    self.arch
                ),
            ));
        };
        self.ensure_kernel(kernel_name, kernel_src, symbol)?;
        let func = &self.functions[kernel_name];
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let v_tile = 32usize;
        let s_lds_stride = 130usize;
        let v_lds_bytes = v_tile * head_dim * 2;
        let s_lds_bytes = 64 * s_lds_stride * 2;
        let scaler_bytes = 64 * 4 * 3;
        let shared_mem = (v_lds_bytes + s_lds_bytes + scaler_bytes) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// V2 of `attention_dflash_wmma_m64_n128_f16kv_f32`. Same shape
    /// (M=64, N=128, 4-wave block, f16 K/V, O in registers) but adds
    /// (a) S_lds row stride 130 (was 128) to break a 16-way LDS bank
    /// conflict in phase C's S_lds reads, and (b) cooperative wave-32
    /// softmax via __shfl_xor butterfly.
    pub fn attention_dflash_wmma_m64_n128_f16kv_v2_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128 (d_chunks=8 unroll, O_frags[8] register array).",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V2_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_v2_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v2_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // LDS layout (f32-equivalent slots):
        //   V_lds[128 * head_dim] f16  = 128 * head_dim / 2 f32 slots
        //   S_lds[64  * 130]      f16  = 64 * 130 / 2       f32 slots (padded stride)
        //   m + l + alpha (64 each f32) = 192               f32 slots
        let lds_f32 = (128 * head_dim) / 2 + (64 * 130) / 2 + 64 * 3;
        let shared_mem = (lds_f32 * 4) as u32;

        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1], // 4 waves
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// V_lds transpose variant of v3 (M=64, N=128, f16 K/V).
    pub fn attention_dflash_wmma_m64_n128_f16kv_v4_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32: head_dim={head_dim} but this kernel is \
             hard-coded to head_dim==128.",
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32: n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}",
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V4_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_v4_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v4_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let v_t_stride = 130usize;
        let s_lds_stride = 130usize;
        let m_tile = 64usize;
        let v_lds_bytes = head_dim * v_t_stride * 2;
        let s_lds_bytes = m_tile * s_lds_stride * 2;
        let scaler_bytes = m_tile * 4 * 3;
        let shared_mem = (v_lds_bytes + s_lds_bytes + scaler_bytes) as u32;
        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// V_lds transpose variant of v5 (M=64, V_tile=32). Kept for bench only.
    pub fn attention_dflash_wmma_m64_n32_f16kv_v6_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            q.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n32_f16kv_v6_f32: q must be F32"
        );
        assert_eq!(
            k_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n32_f16kv_v6_f32: k must be F16"
        );
        assert_eq!(
            v_f16.dtype,
            DType::F16,
            "attention_dflash_wmma_m64_n32_f16kv_v6_f32: v must be F16"
        );
        assert_eq!(
            out.dtype,
            DType::F32,
            "attention_dflash_wmma_m64_n32_f16kv_v6_f32: out must be F32"
        );
        assert!(
            head_dim == 128,
            "attention_dflash_wmma_m64_n32_f16kv_v6_f32: head_dim must be 128"
        );
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "v6: n_heads must be divisible by n_kv_heads"
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n32_f16kv_v6_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N32_F16KV_V6_SRC,
            "attention_dflash_wmma_m64_n32_f16kv_v6_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n32_f16kv_v6_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let shared_mem = 25600u32;
        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// v7: M=128 K-shared sub-tiling. Kept for bench only.
    pub fn attention_dflash_wmma_m128_n32_f16kv_v7_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(q.dtype, DType::F32, "v7: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "v7: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "v7: v must be F16");
        assert_eq!(out.dtype, DType::F32, "v7: out must be F32");
        assert!(head_dim == 128, "v7: head_dim must be 128");
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "v7: n_heads must be divisible by n_kv_heads"
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m128_n32_f16kv_v7_f32",
            kernels::ATTENTION_DFLASH_WMMA_M128_N32_F16KV_V7_SRC,
            "attention_dflash_wmma_m128_n32_f16kv_v7_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m128_n32_f16kv_v7_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let shared_mem = 25600u32;
        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let q_tiles_128 = (b + 127) / 128;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles_128 as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// v7b: M=128 sequential sub-tiling, no K-sharing. Kept for bench only.
    pub fn attention_dflash_wmma_m128_n32_f16kv_v7b_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(q.dtype, DType::F32, "v7b: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "v7b: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "v7b: v must be F16");
        assert_eq!(out.dtype, DType::F32, "v7b: out must be F32");
        assert!(head_dim == 128, "v7b: head_dim must be 128");
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "v7b: n_heads must be divisible by n_kv_heads"
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m128_n32_f16kv_v7b_f32",
            kernels::ATTENTION_DFLASH_WMMA_M128_N32_F16KV_V7B_SRC,
            "attention_dflash_wmma_m128_n32_f16kv_v7b_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m128_n32_f16kv_v7b_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let shared_mem = 25600u32;
        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let q_tiles_128 = (b + 127) / 128;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles_128 as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// V_lds transpose variant of v3-causal. gfx11-only; falls back to
    /// v3-causal on gfx12.
    pub fn attention_dflash_wmma_m64_n128_f16kv_v4_causal_f32(
        &mut self,
        q: &GpuTensor,
        k_f16: &GpuTensor,
        v_f16: &GpuTensor,
        out: &GpuTensor,
        b: usize,
        l: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.arch_caps.has_wmma_w32_gfx12() {
            return self.attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32(
                q, k_f16, v_f16, out, b, l, n_heads, n_kv_heads, head_dim,
            );
        }
        if !self.arch_caps.has_wmma_w32() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "attention_dflash_wmma_m64_n128_f16kv_v4_causal_f32 requires wave32 WMMA; \
                     arch={} does not support it. Use attention_causal_batched for this arch.",
                    self.arch
                ),
            ));
        }
        assert_eq!(q.dtype, DType::F32, "v4_causal: q must be F32");
        assert_eq!(k_f16.dtype, DType::F16, "v4_causal: k must be F16");
        assert_eq!(v_f16.dtype, DType::F16, "v4_causal: v must be F16");
        assert_eq!(out.dtype, DType::F32, "v4_causal: out must be F32");
        assert!(head_dim == 128, "v4_causal: head_dim must be 128");
        assert!(b > 0 && l > 0 && n_heads > 0 && n_kv_heads > 0);
        assert!(
            n_heads % n_kv_heads == 0,
            "v4_causal: n_heads must be divisible by n_kv_heads"
        );
        self.ensure_kernel(
            "attention_dflash_wmma_m64_n128_f16kv_v4_causal_f32",
            kernels::ATTENTION_DFLASH_WMMA_M64_N128_F16KV_V4_CAUSAL_SRC,
            "attention_dflash_wmma_m64_n128_f16kv_v4_causal_f32",
        )?;
        let func = &self.functions["attention_dflash_wmma_m64_n128_f16kv_v4_causal_f32"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let v_t_stride = 130usize;
        let s_lds_stride = 130usize;
        let m_tile = 64usize;
        let v_lds_bytes = head_dim * v_t_stride * 2;
        let s_lds_bytes = m_tile * s_lds_stride * 2;
        let scaler_bytes = m_tile * 4 * 3;
        let shared_mem = (v_lds_bytes + s_lds_bytes + scaler_bytes) as u32;
        let mut qp = q.buf.as_ptr();
        let mut kp = k_f16.buf.as_ptr();
        let mut vp = v_f16.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut bi = b as i32;
        let mut li = l as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut li as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let q_tiles = (b + 63) / 64;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, q_tiles as u32, 1],
                [128, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PFlash scoring
    // ═══════════════════════════════════════════════════════════════════════════

    /// Per-block scoring on Q8_0 K cache. Reads `k_cache` (one layer's
    /// K-cache backing memory; the buffer must be the Q8_0-formatted slab
    /// produced by `KvCache::new_gpu_q8`) for the first `n_pos` positions,
    /// computes per-block mean K and cosine similarity vs the K at
    /// `last_pos`, and writes `n_blocks` f32 scores into `scores_out`.
    ///
    /// One workgroup per output block, 256 threads per workgroup. Each
    /// thread strides through `kv_dim` doing inline f16-scale + i8-value
    /// dequant; a 256-thread shared-memory reduction folds the partial
    /// (dot, ||block||^2, ||last||^2) fragments into one cosine score.
    ///
    /// Phase 2.1 of #93. Replaces the CPU-side dequant + mean-pool +
    /// cosine in `pflash::compute_scores_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_q8_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim % 32 == 0,
            "head_dim must be a multiple of 32 for Q8 KV cache"
        );
        assert!(n_blocks > 0 && block_size > 0 && n_pos > 0);
        assert!(last_pos < n_pos, "last_pos {last_pos} >= n_pos {n_pos}");
        self.ensure_kernel(
            "pflash_score_q8_kv",
            kernels::PFLASH_SCORE_Q8_KV_SRC,
            "pflash_score_q8_kv_blocks",
        )?;
        let func = &self.functions["pflash_score_q8_kv_blocks"];

        let k_ptr = k_cache.buf.as_ptr();
        let s_ptr = scores_out.buf.as_ptr();
        let mut np = n_pos as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;
        let mut nb = n_blocks as i32;
        let mut lp = last_pos as i32;

        let mut params: Vec<*mut c_void> = vec![
            &k_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// PFlash per-block scoring — fwht3 K-cache variant.
    ///
    /// Same input/output contract as `pflash_score_q8_kv`: takes a K
    /// cache buffer and emits one f32 cosine score per block. Only the
    /// K dequant path differs (fwht3 vs Q8). Used by
    /// `pflash::compute_scores_batched_gpu` when the drafter runs with
    /// fwht3 KV — that path's no-LDS-cap batched flash unblocks the >15K
    /// ctx regime that Q8 batched flash falls out of.
    ///
    /// Header prepend: the kernel uses `TURBO_C3_256` from
    /// `turbo_common.h`. Reusing `ensure_givens4_kernel` since it already
    /// prepends `turbo_common.h` + `givens_common.h`. The unused
    /// givens_common include is harmless (no symbols are referenced from
    /// it), and avoids adding another `ensure_*` variant.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_fwht3_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to pflash_score_fwht_kv_impl which binds
        self.pflash_score_fwht_kv_impl(
            "pflash_score_fwht3_kv",
            kernels::PFLASH_SCORE_FWHT3_KV_SRC,
            "pflash_score_fwht3_kv_blocks",
            8, // alignment: 8 dims per thread group (3-bit codes × 8 = 24 bits = 3 bytes)
            k_cache,
            scores_out,
            n_pos,
            n_kv_heads,
            head_dim,
            block_size,
            n_blocks,
            last_pos,
        )
    }

    /// PFlash per-block scoring — fwht4 K-cache variant.
    /// 4-bit codes packed into nibbles, two FWHT-128 halves per head at
    /// head_dim=256. Higher precision than fwht3 / larger K storage
    /// (132 B/head vs 100 B). Ablation variant.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_fwht4_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to pflash_score_fwht_kv_impl which binds
        self.pflash_score_fwht_kv_impl(
            "pflash_score_fwht4_kv",
            kernels::PFLASH_SCORE_FWHT4_KV_SRC,
            "pflash_score_fwht4_kv_blocks",
            // fwht4 thread-group = 4 dims (4-bit × 4 = 16 bits = 2 bytes)
            // plus head_dim must accommodate two FWHT-128 halves.
            4,
            k_cache,
            scores_out,
            n_pos,
            n_kv_heads,
            head_dim,
            block_size,
            n_blocks,
            last_pos,
        )
    }

    /// PFlash per-block scoring — fwht2 K-cache variant.
    /// 2-bit codes packed 4 per byte, two FWHT-128 halves per head at
    /// head_dim=256. Smallest K storage in the family (68 B/head).
    /// Ablation / lower-bound variant — likely NIAH-marginal.
    #[allow(clippy::too_many_arguments)]
    pub fn pflash_score_fwht2_kv(
        &mut self,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — delegates to pflash_score_fwht_kv_impl which binds
        self.pflash_score_fwht_kv_impl(
            "pflash_score_fwht2_kv",
            kernels::PFLASH_SCORE_FWHT2_KV_SRC,
            "pflash_score_fwht2_kv_blocks",
            4, // fwht2 thread-group = 4 dims (2-bit × 4 = 8 bits = 1 byte)
            k_cache,
            scores_out,
            n_pos,
            n_kv_heads,
            head_dim,
            block_size,
            n_blocks,
            last_pos,
        )
    }

    /// Shared launch body for fwht{2,3,4} scoring — same grid +
    /// argument shape, only the kernel binary + per-thread-group
    /// alignment vary.
    #[allow(clippy::too_many_arguments)]
    fn pflash_score_fwht_kv_impl(
        &mut self,
        cache_key: &str,
        src: &str,
        func_name: &str,
        tg_align: i32,
        k_cache: &GpuTensor,
        scores_out: &GpuTensor,
        n_pos: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        n_blocks: usize,
        last_pos: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            head_dim as i32 % tg_align == 0,
            "head_dim must be a multiple of {tg_align} for this fwht K cache layout",
        );
        assert!(n_blocks > 0 && block_size > 0 && n_pos > 0);
        assert!(last_pos < n_pos, "last_pos {last_pos} >= n_pos {n_pos}");
        self.ensure_givens4_kernel(cache_key, src, func_name)?;
        let func = &self.functions[func_name];

        let k_ptr = k_cache.buf.as_ptr();
        let s_ptr = scores_out.buf.as_ptr();
        let mut np = n_pos as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut bs = block_size as i32;
        let mut nb = n_blocks as i32;
        let mut lp = last_pos as i32;

        let mut params: Vec<*mut c_void> = vec![
            &k_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &mut np as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    pub fn compressor_compress_aligned_batched_f32(
        &mut self,
        prev_kv: &GpuTensor,
        prev_score: &GpuTensor,
        kv_batch: &GpuTensor,
        score_batch: &GpuTensor,
        kv_cache_out: &GpuTensor,
        r: i32,
        head_dim: i32,
        n_events: i32,
        overlap: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_compress_aligned_batched_f32",
            kernels::COMPRESSOR_COMPRESS_ALIGNED_BATCHED_SRC,
            "compressor_compress_aligned_batched_f32",
        )?;
        let func = &self.functions["compressor_compress_aligned_batched_f32"];
        let pk = prev_kv.buf.as_ptr();
        let ps = prev_score.buf.as_ptr();
        let kb = kv_batch.buf.as_ptr();
        let sb = score_batch.buf.as_ptr();
        let yo = kv_cache_out.buf.as_ptr();
        let mut rr = r;
        let mut hd = head_dim;
        let mut ne = n_events;
        let mut ov = overlap;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &pk as *const _ as *mut c_void,
            &ps as *const _ as *mut c_void,
            &kb as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &yo as *const _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut ov as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let grid_x = ((head_dim + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, n_events as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn compressor_add_ape_batched_f32(
        &mut self,
        score_batch: &GpuTensor, // [B, proj_dim] F32, in-place
        ape: &GpuTensor,         // [ratio, proj_dim] F32
        batch_size: i32,
        proj_dim: i32,
        ratio: i32,
        start_pos: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_add_ape_batched",
            kernels::COMPRESSOR_ADD_APE_BATCHED_SRC,
            "compressor_add_ape_batched_f32",
        )?;
        let func = &self.functions["compressor_add_ape_batched_f32"];
        let sb = score_batch.buf.as_ptr();
        let ap = ape.buf.as_ptr();
        let mut bs = batch_size;
        let mut pd = proj_dim;
        let mut rr = ratio;
        let mut sp = start_pos;
        let mut params: Vec<*mut c_void> = vec![
            &sb as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
        ];
        let grid_x = ((proj_dim + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn compressor_add_ape_f32_buf(
        &mut self,
        score: &GpuTensor,
        ape: &GpuTensor,
        ring_slot_buf: &GpuTensor,
        proj_dim: i32,
        ratio: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_add_ape_buf",
            kernels::COMPRESSOR_ADD_APE_BATCHED_SRC,
            "compressor_add_ape_f32_buf",
        )?;
        let sp = score.buf.as_ptr();
        let ap = ape.buf.as_ptr();
        let rp = ring_slot_buf.buf.as_ptr();
        let mut pd = proj_dim;
        let mut rr = ratio;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((proj_dim as u32) + block - 1) / block;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sp);
            b.push_ptr(ap);
            b.push_ptr(rp);
            b.push_i32(pd);
            b.push_i32(rr);
            b
        };
        self.launch_maybe_blob(
            "compressor_add_ape_f32_buf",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn compressor_overlap_concat_f32(
        &mut self,
        src: &GpuTensor, // [2*ratio, 2*head_dim] F32
        dst: &GpuTensor, // [2*ratio, head_dim] F32
        ratio: i32,
        head_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_overlap_concat",
            kernels::COMPRESSOR_OVERLAP_CONCAT_SRC,
            "compressor_overlap_concat_f32",
        )?;
        let sp = src.buf.as_ptr();
        let dp = dst.buf.as_ptr();
        let mut rv = ratio;
        let mut hd = head_dim;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &mut rv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "compressor_overlap_concat_f32",
            [(2 * ratio) as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(sp);
                b.push_ptr(dp);
                b.push_i32(rv);
                b.push_i32(hd);
                b
            },
        )
    }
    pub fn compressor_ring_write_batched_f32(
        &mut self,
        kv_batch: &GpuTensor,
        score_batch: &GpuTensor,
        kv_state: &GpuTensor,
        score_state: &GpuTensor,
        batch_size: i32,
        proj_dim: i32,
        r: i32,
        slot_base: i32,
        overlap: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_ring_write_batched_f32",
            kernels::COMPRESSOR_RING_WRITE_BATCHED_SRC,
            "compressor_ring_write_batched_f32",
        )?;
        let func = &self.functions["compressor_ring_write_batched_f32"];
        let kb = kv_batch.buf.as_ptr();
        let sb = score_batch.buf.as_ptr();
        let ks = kv_state.buf.as_ptr();
        let ss = score_state.buf.as_ptr();
        let mut bsv = batch_size;
        let mut pd = proj_dim;
        let mut rr = r;
        let mut sbase = slot_base;
        let mut ov = overlap;
        let mut params: Vec<*mut c_void> = vec![
            &kb as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &ks as *const _ as *mut c_void,
            &ss as *const _ as *mut c_void,
            &mut bsv as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut sbase as *mut _ as *mut c_void,
            &mut ov as *mut _ as *mut c_void,
        ];
        let grid_x = ((proj_dim + 255) / 256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn compressor_softmax_pool_f32(
        &mut self,
        kv_state: &GpuTensor,    // [T, head_dim] F32
        score_state: &GpuTensor, // [T, head_dim] F32
        output: &GpuTensor,      // [head_dim] F32
        t: i32,
        head_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_softmax_pool",
            kernels::COMPRESSOR_SOFTMAX_POOL_SRC,
            "compressor_softmax_pool_f32",
        )?;
        let func = &self.functions["compressor_softmax_pool_f32"];
        let kp = kv_state.buf.as_ptr();
        let sp = score_state.buf.as_ptr();
        let op = output.buf.as_ptr();
        let mut tv = t;
        let mut hd = head_dim;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut tv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((head_dim as u32) + block - 1) / block;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn compressor_softmax_pool_f32_buf(
        &mut self,
        kv_state: &GpuTensor,
        score_state: &GpuTensor,
        kv_cache: &GpuTensor, // base ptr [max_slots, head_dim]
        slot_buf: &GpuTensor,
        t: i32,
        head_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "compressor_softmax_pool_f32_buf",
            kernels::COMPRESSOR_SOFTMAX_POOL_BUF_SRC,
            "compressor_softmax_pool_f32_buf",
        )?;
        let kp = kv_state.buf.as_ptr();
        let sp = score_state.buf.as_ptr();
        let cp = kv_cache.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let mut tv = t;
        let mut hd = head_dim;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &mut tv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((head_dim as u32) + block - 1) / block;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(kp);
            b.push_ptr(sp);
            b.push_ptr(cp);
            b.push_ptr(sb);
            b.push_i32(tv);
            b.push_i32(hd);
            b
        };
        self.launch_maybe_blob(
            "compressor_softmax_pool_f32_buf",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn hc_apply_alpha(
        &mut self,
        c: &GpuTensor,
        alpha: &GpuTensor,
        base: &GpuTensor,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_apply_alpha",
            kernels::HC_APPLY_ALPHA_SRC,
            "hc_apply_alpha",
        )?;
        let cp = c.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_apply_alpha",
            [1, 1, 1],
            [24, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(ap);
                b.push_ptr(bp);
                b
            },
        )
    }
    pub fn hc_apply_alpha_batched(
        &mut self,
        c: &GpuTensor,
        alpha: &GpuTensor,
        base: &GpuTensor,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_apply_alpha_batched",
            kernels::HC_APPLY_ALPHA_BATCHED_SRC,
            "hc_apply_alpha_batched",
        )?;
        let cp = c.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_apply_alpha_batched",
            [batch_size as u32, 1, 1],
            [24, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(ap);
                b.push_ptr(bp);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn hc_compute_control(
        &mut self,
        deepseek4_gfx1151_route: bool,
        x_flat: &GpuTensor, // [x_dim] fp16
        w_fn: &GpuTensor,   // [n_ctrl, x_dim] fp16
        base: &GpuTensor,   // [n_ctrl] fp16
        c_out: &GpuTensor,  // [n_ctrl] fp32
        n_ctrl: i32,
        x_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let vec4 = self.arch == "gfx1151" && deepseek4_gfx1151_route;
        let (logical_name, symbol) = if vec4 {
            ("hc_compute_control_vec4", "hc_compute_control_vec4")
        } else {
            ("hc_compute_control", "hc_compute_control")
        };
        self.ensure_kernel(logical_name, kernels::HC_COMPUTE_CONTROL_SRC, symbol)?;
        let xp = x_flat.buf.as_ptr();
        let wp = w_fn.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let cp = c_out.buf.as_ptr();
        let mut nc = n_ctrl;
        let mut xd = x_dim;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut xd as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            logical_name,
            [n_ctrl as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(bp);
                b.push_ptr(cp);
                b.push_i32(nc);
                b.push_i32(xd);
                b
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_compute_control_vec4_finalize(
        &mut self,
        x_flat: &GpuTensor,
        w_fn: &GpuTensor,
        base: &GpuTensor,
        c_out: &GpuTensor,
        alpha: &GpuTensor,
        n_ctrl: i32,
        x_dim: i32,
        hc_eps: f32,
        post_scale: f32,
        sinkhorn_iters: i32,
        rsqrt_once: bool,
        prefer_t1024: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // `HIPFIRE_HC_CTRL_T1024=1` selects the 1024-thread variant. See
        // `kernels::HC_COMPUTE_CONTROL_T1024_SRC` — same algorithm, wider
        // block, NOT bit-exact (the LDS partial tree widens 8 -> 32).
        static T1024: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let t1024 = prefer_t1024
            || *T1024.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_HC_CTRL_T1024")
                    .ok()
                    .as_deref()
                    == Some("1")
            });
        let (logical_name, src, threads) = if t1024 {
            (
                "hc_compute_control_vec4_finalize_t1024",
                kernels::HC_COMPUTE_CONTROL_T1024_SRC,
                1024u32,
            )
        } else {
            (
                "hc_compute_control_vec4_finalize",
                kernels::HC_COMPUTE_CONTROL_SRC,
                256u32,
            )
        };
        self.ensure_kernel(logical_name, src, logical_name)?;
        let xp = x_flat.buf.as_ptr();
        let wp = w_fn.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let cp = c_out.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let mut nc = n_ctrl;
        let mut xd = x_dim;
        let mut eps = hc_eps;
        let mut ps = post_scale;
        let mut iters = sinkhorn_iters;
        let mut once = i32::from(rsqrt_once);
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut xd as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut iters as *mut _ as *mut c_void,
            &mut once as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            logical_name,
            [n_ctrl as u32, 1, 1],
            [threads, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(bp);
                b.push_ptr(cp);
                b.push_ptr(ap);
                b.push_i32(nc);
                b.push_i32(xd);
                b.push_f32(eps);
                b.push_f32(ps);
                b.push_i32(iters);
                b.push_i32(once);
                b
            },
        )
    }
    pub fn hc_compute_control_batched(
        &mut self,
        x_flat: &GpuTensor, // [batch, x_dim]
        w_fn: &GpuTensor,   // [n_ctrl, x_dim] fp16
        base: &GpuTensor,   // [n_ctrl] fp16
        c_out: &GpuTensor,  // [batch, n_ctrl] fp32
        n_ctrl: i32,
        x_dim: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_compute_control_batched",
            kernels::HC_COMPUTE_CONTROL_BATCHED_SRC,
            "hc_compute_control_batched",
        )?;
        let xp = x_flat.buf.as_ptr();
        let wp = w_fn.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let cp = c_out.buf.as_ptr();
        let mut nc = n_ctrl;
        let mut xd = x_dim;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut xd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_compute_control_batched",
            [n_ctrl as u32, batch_size as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(bp);
                b.push_ptr(cp);
                b.push_i32(nc);
                b.push_i32(xd);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn hc_head_compute_pre(
        &mut self,
        x_flat: &GpuTensor,  // [hc_mult * hidden] F32
        w_fn: &GpuTensor,    // [hc_mult, hc_mult * hidden] F16
        base: &GpuTensor,    // [hc_mult] F16
        pre_out: &GpuTensor, // [hc_mult] F32
        hc_mult: i32,
        x_dim: i32,
        scale: f32, // hc_head_scale (scalar)
        norm_eps: f32,
        hc_eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_head_compute_pre",
            kernels::HC_HEAD_COMPUTE_PRE_SRC,
            "hc_head_compute_pre",
        )?;
        let xp = x_flat.buf.as_ptr();
        let wp = w_fn.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let pp = pre_out.buf.as_ptr();
        let mut hm = hc_mult;
        let mut xd = x_dim;
        let mut sv = scale;
        let mut ne = norm_eps;
        let mut he = hc_eps;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut hm as *mut _ as *mut c_void,
            &mut xd as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut he as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_head_compute_pre",
            [hc_mult as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(bp);
                b.push_ptr(pp);
                b.push_i32(hm);
                b.push_i32(xd);
                b.push_f32(sv);
                b.push_f32(ne);
                b.push_f32(he);
                b
            },
        )
    }
    pub fn hc_input_map_4stream(
        &mut self,
        a_vec: &GpuTensor,
        streams: &GpuTensor,
        x_out: &GpuTensor,
        hidden: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_input_map_4stream",
            kernels::HC_INPUT_MAP_SRC,
            "hc_input_map_4stream",
        )?;
        let ap = a_vec.buf.as_ptr();
        let sp = streams.buf.as_ptr();
        let op = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_input_map_4stream",
            [((hidden + 255) / 256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(sp);
                b.push_ptr(op);
                b.push_i32(h);
                b
            },
        )
    }
    pub fn hc_input_map_4stream_batched(
        &mut self,
        a_vec: &GpuTensor,   // [batch, HC_MULT]
        streams: &GpuTensor, // [batch, HC_MULT, hidden]
        x_out: &GpuTensor,   // [batch, hidden]
        hidden: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_input_map_4stream_batched",
            kernels::HC_INPUT_MAP_BATCHED_SRC,
            "hc_input_map_4stream_batched",
        )?;
        let ap = a_vec.buf.as_ptr();
        let sp = streams.buf.as_ptr();
        let op = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_input_map_4stream_batched",
            [((hidden + 255) / 256) as u32, batch_size as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ap);
                b.push_ptr(sp);
                b.push_ptr(op);
                b.push_i32(h);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn hc_mix_4stream(
        &mut self,
        x_in: &GpuTensor,          // [4, hidden] fp16
        a_matrix: &GpuTensor,      // [4, 4] fp32 (post-Sinkhorn)
        scale: &GpuTensor,         // [4] fp32
        transform_out: &GpuTensor, // [hidden] fp16
        x_out: &GpuTensor,         // [4, hidden] fp16
        hidden: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_mix_4stream",
            kernels::HC_MIX_4STREAM_SRC,
            "hc_mix_4stream",
        )?;
        let xi = x_in.buf.as_ptr();
        let am = a_matrix.buf.as_ptr();
        let sc = scale.buf.as_ptr();
        let to = transform_out.buf.as_ptr();
        let xo = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut params: Vec<*mut c_void> = vec![
            &xi as *const _ as *mut c_void,
            &am as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &to as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_mix_4stream",
            [((hidden + 255) / 256) as u32, 4, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xi);
                b.push_ptr(am);
                b.push_ptr(sc);
                b.push_ptr(to);
                b.push_ptr(xo);
                b.push_i32(h);
                b
            },
        )
    }

    /// gfx1201 TP3 Hyper-Connection mix that consumes three peer-visible
    /// rank-local transform outputs in fixed rank order.
    pub fn hc_mix_4stream_peer3_gfx1201(
        &mut self,
        x_in: &GpuTensor,
        a_matrix: &GpuTensor,
        scale: &GpuTensor,
        transforms: [&GpuTensor; 3],
        x_out: &GpuTensor,
        hidden: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "hc_mix_4stream_peer3_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        let hidden_usize = hidden as usize;
        if transforms
            .iter()
            .any(|transform| transform.numel() != hidden_usize)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "hc_mix_4stream_peer3_gfx1201: transform shape mismatch",
            ));
        }
        self.ensure_kernel(
            "hc_mix_4stream_peer3_gfx1201",
            kernels::HC_MIX_4STREAM_PEER4_GFX1201_SRC,
            "hc_mix_4stream_peer3_gfx1201",
        )?;

        let xi = x_in.buf.as_ptr();
        let am = a_matrix.buf.as_ptr();
        let sc = scale.buf.as_ptr();
        let t0 = transforms[0].buf.as_ptr();
        let t1 = transforms[1].buf.as_ptr();
        let t2 = transforms[2].buf.as_ptr();
        let xo = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut params: Vec<*mut c_void> = vec![
            &xi as *const _ as *mut c_void,
            &am as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &t0 as *const _ as *mut c_void,
            &t1 as *const _ as *mut c_void,
            &t2 as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_mix_4stream_peer3_gfx1201",
            [((hidden + 255) / 256) as u32, 4, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xi);
                b.push_ptr(am);
                b.push_ptr(sc);
                b.push_ptr(t0);
                b.push_ptr(t1);
                b.push_ptr(t2);
                b.push_ptr(xo);
                b.push_i32(h);
                b
            },
        )
    }

    /// Micro-only exact-gfx1201 TP3 HC consumer with an embedded peer
    /// handshake and a caller-selected number of grid-stride blocks per HC
    /// stream. Product dispatch does not call this method.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_mix_4stream_peer3_sync_gridstride_gfx1201(
        &mut self,
        x_in: &GpuTensor,
        a_matrix: &GpuTensor,
        scale: &GpuTensor,
        transforms: [&GpuTensor; 3],
        own_signal: &DeviceBuffer,
        peer_signals: [&DeviceBuffer; 2],
        epoch: u32,
        blocks_per_stream: u32,
        x_out: &GpuTensor,
        hidden: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() || !matches!(blocks_per_stream, 1 | 2 | 4 | 8 | 16) {
            return Err(hip_bridge::HipError::new(
                0,
                "hc_mix_4stream_peer3_sync_gridstride_gfx1201 requires gfx1201 and 1/2/4/8/16 blocks per stream",
            ));
        }
        let hidden_usize = hidden as usize;
        if transforms
            .iter()
            .any(|transform| transform.numel() != hidden_usize)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "hc_mix_4stream_peer3_sync_gridstride_gfx1201: transform shape mismatch",
            ));
        }
        const KERNEL: &str = "hc_mix_4stream_peer3_sync_gridstride_gfx1201";
        self.ensure_kernel(KERNEL, kernels::HC_MIX_4STREAM_PEER4_GFX1201_SRC, KERNEL)?;

        let xi = x_in.buf.as_ptr();
        let am = a_matrix.buf.as_ptr();
        let sc = scale.buf.as_ptr();
        let t0 = transforms[0].buf.as_ptr();
        let t1 = transforms[1].buf.as_ptr();
        let t2 = transforms[2].buf.as_ptr();
        let own = own_signal.as_ptr();
        let peer0 = peer_signals[0].as_ptr();
        let peer1 = peer_signals[1].as_ptr();
        let xo = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut epoch_arg = epoch;
        let mut params: Vec<*mut c_void> = vec![
            &xi as *const _ as *mut c_void,
            &am as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &t0 as *const _ as *mut c_void,
            &t1 as *const _ as *mut c_void,
            &t2 as *const _ as *mut c_void,
            &own as *const _ as *mut c_void,
            &peer0 as *const _ as *mut c_void,
            &peer1 as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut epoch_arg as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [blocks_per_stream, 4, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xi);
                b.push_ptr(am);
                b.push_ptr(sc);
                b.push_ptr(t0);
                b.push_ptr(t1);
                b.push_ptr(t2);
                b.push_ptr(own);
                b.push_ptr(peer0);
                b.push_ptr(peer1);
                b.push_ptr(xo);
                b.push_i32(h);
                b.push_u32(epoch);
                b
            },
        )
    }

    /// gfx1201 TP4 Hyper-Connection mix that consumes four peer-visible
    /// rank-local transform outputs in fixed rank order.
    ///
    /// The caller must enable all-to-all peer access and order this launch
    /// after all four producer streams. This exact-gated primitive is not a
    /// generic collective; it exists to remove DS4's per-block RCCL
    /// materialization and subsequent HC launch boundary.
    pub fn hc_mix_4stream_peer4_gfx1201(
        &mut self,
        x_in: &GpuTensor,
        a_matrix: &GpuTensor,
        scale: &GpuTensor,
        transforms: [&GpuTensor; 4],
        x_out: &GpuTensor,
        hidden: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "hc_mix_4stream_peer4_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        let hidden_usize = hidden as usize;
        if transforms
            .iter()
            .any(|transform| transform.numel() != hidden_usize)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "hc_mix_4stream_peer4_gfx1201: transform shape mismatch",
            ));
        }
        self.ensure_kernel(
            "hc_mix_4stream_peer4_gfx1201",
            kernels::HC_MIX_4STREAM_PEER4_GFX1201_SRC,
            "hc_mix_4stream_peer4_gfx1201",
        )?;

        let xi = x_in.buf.as_ptr();
        let am = a_matrix.buf.as_ptr();
        let sc = scale.buf.as_ptr();
        let t0 = transforms[0].buf.as_ptr();
        let t1 = transforms[1].buf.as_ptr();
        let t2 = transforms[2].buf.as_ptr();
        let t3 = transforms[3].buf.as_ptr();
        let xo = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut params: Vec<*mut c_void> = vec![
            &xi as *const _ as *mut c_void,
            &am as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &t0 as *const _ as *mut c_void,
            &t1 as *const _ as *mut c_void,
            &t2 as *const _ as *mut c_void,
            &t3 as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_mix_4stream_peer4_gfx1201",
            [((hidden + 255) / 256) as u32, 4, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xi);
                b.push_ptr(am);
                b.push_ptr(sc);
                b.push_ptr(t0);
                b.push_ptr(t1);
                b.push_ptr(t2);
                b.push_ptr(t3);
                b.push_ptr(xo);
                b.push_i32(h);
                b
            },
        )
    }

    /// Publish one TP graph barrier epoch with system-scope release semantics.
    pub fn tp_graph_signal_store_gfx1201(
        &mut self,
        signal: &DeviceBuffer,
        epoch: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tp_graph_signal_store_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        self.ensure_kernel(
            "tp_graph_signal_store_gfx1201",
            kernels::TP_GRAPH_SIGNAL_GFX1201_SRC,
            "tp_graph_signal_store_gfx1201",
        )?;

        let ptr = signal.as_ptr();
        let mut epoch_arg = epoch;
        let mut params: Vec<*mut c_void> = vec![
            &ptr as *const _ as *mut c_void,
            &mut epoch_arg as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "tp_graph_signal_store_gfx1201",
            [1, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ptr);
                b.push_u32(epoch);
                b
            },
        )
    }

    /// Wait for the other two TP3 ranks with system-scope acquire semantics.
    pub fn tp_graph_signal_wait2_gfx1201(
        &mut self,
        signals: [&DeviceBuffer; 2],
        epoch: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tp_graph_signal_wait2_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        self.ensure_kernel(
            "tp_graph_signal_wait2_gfx1201",
            kernels::TP_GRAPH_SIGNAL_GFX1201_SRC,
            "tp_graph_signal_wait2_gfx1201",
        )?;

        let signal0 = signals[0].as_ptr();
        let signal1 = signals[1].as_ptr();
        let mut epoch_arg = epoch;
        let mut params: Vec<*mut c_void> = vec![
            &signal0 as *const _ as *mut c_void,
            &signal1 as *const _ as *mut c_void,
            &mut epoch_arg as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "tp_graph_signal_wait2_gfx1201",
            [1, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(signal0);
                b.push_ptr(signal1);
                b.push_u32(epoch);
                b
            },
        )
    }

    /// Wait for the other three TP4 ranks with system-scope acquire semantics.
    pub fn tp_graph_signal_wait3_gfx1201(
        &mut self,
        signals: [&DeviceBuffer; 3],
        epoch: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tp_graph_signal_wait3_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        self.ensure_kernel(
            "tp_graph_signal_wait3_gfx1201",
            kernels::TP_GRAPH_SIGNAL_GFX1201_SRC,
            "tp_graph_signal_wait3_gfx1201",
        )?;

        let signal0 = signals[0].as_ptr();
        let signal1 = signals[1].as_ptr();
        let signal2 = signals[2].as_ptr();
        let mut epoch_arg = epoch;
        let mut params: Vec<*mut c_void> = vec![
            &signal0 as *const _ as *mut c_void,
            &signal1 as *const _ as *mut c_void,
            &signal2 as *const _ as *mut c_void,
            &mut epoch_arg as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "tp_graph_signal_wait3_gfx1201",
            [1, 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(signal0);
                b.push_ptr(signal1);
                b.push_ptr(signal2);
                b.push_u32(epoch);
                b
            },
        )
    }

    pub fn hc_mix_4stream_batched(
        &mut self,
        x_in: &GpuTensor,          // [batch, 4, hidden]
        a_matrix: &GpuTensor,      // [batch, 4, 4]
        scale: &GpuTensor,         // [batch, 4]
        transform_out: &GpuTensor, // [batch, hidden]
        x_out: &GpuTensor,         // [batch, 4, hidden]
        hidden: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_mix_4stream_batched",
            kernels::HC_MIX_4STREAM_BATCHED_SRC,
            "hc_mix_4stream_batched",
        )?;
        let xi = x_in.buf.as_ptr();
        let am = a_matrix.buf.as_ptr();
        let sc = scale.buf.as_ptr();
        let to = transform_out.buf.as_ptr();
        let xo = x_out.buf.as_ptr();
        let mut h = hidden;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &xi as *const _ as *mut c_void,
            &am as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
            &to as *const _ as *mut c_void,
            &xo as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_mix_4stream_batched",
            [((hidden + 255) / 256) as u32, 4, batch_size as u32],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xi);
                b.push_ptr(am);
                b.push_ptr(sc);
                b.push_ptr(to);
                b.push_ptr(xo);
                b.push_i32(h);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn hc_pre_post_sigmoid_scale_f32(
        &mut self,
        hc_c: &GpuTensor,
        hc_eps: f32,
        post_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_pre_post_sigmoid_scale_f32",
            kernels::HC_PRE_POST_SIGMOID_SCALE_SRC,
            "hc_pre_post_sigmoid_scale_f32",
        )?;
        let xp = hc_c.buf.as_ptr();
        let mut eps = hc_eps;
        let mut ps = post_scale;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(xp);
            b.push_f32(eps);
            b.push_f32(ps);
            b
        };
        self.launch_maybe_blob(
            "hc_pre_post_sigmoid_scale_f32",
            [1, 1, 1],
            [8, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }

    pub fn hc_finalize_control(
        &mut self,
        c: &GpuTensor,
        alpha: &GpuTensor,
        base: &GpuTensor,
        hc_eps: f32,
        post_scale: f32,
        sinkhorn_iters: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_finalize_control",
            kernels::HC_FINALIZE_CONTROL_SRC,
            "hc_finalize_control",
        )?;
        let cp = c.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let mut eps = hc_eps;
        let mut ps = post_scale;
        let mut iters = sinkhorn_iters;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut iters as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_finalize_control",
            [1, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(ap);
                b.push_ptr(bp);
                b.push_f32(eps);
                b.push_f32(ps);
                b.push_i32(iters);
                b
            },
        )
    }

    pub fn hc_finalize_input_map(
        &mut self,
        c: &GpuTensor,
        alpha: &GpuTensor,
        base: &GpuTensor,
        streams: &GpuTensor,
        x_out: &GpuTensor,
        hidden: i32,
        hc_eps: f32,
        post_scale: f32,
        sinkhorn_iters: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_finalize_input_map",
            kernels::HC_FINALIZE_INPUT_MAP_SRC,
            "hc_finalize_input_map",
        )?;
        let cp = c.buf.as_ptr();
        let ap = alpha.buf.as_ptr();
        let bp = base.buf.as_ptr();
        let sp = streams.buf.as_ptr();
        let op = x_out.buf.as_ptr();
        let mut hidden_value = hidden;
        let mut eps = hc_eps;
        let mut post = post_scale;
        let mut iterations = sinkhorn_iters;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut hidden_value as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut post as *mut _ as *mut c_void,
            &mut iterations as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_finalize_input_map",
            [1, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(cp);
                blob.push_ptr(ap);
                blob.push_ptr(bp);
                blob.push_ptr(sp);
                blob.push_ptr(op);
                blob.push_i32(hidden_value);
                blob.push_f32(eps);
                blob.push_f32(post);
                blob.push_i32(iterations);
                blob
            },
        )
    }

    pub fn hc_sinkhorn_4x4(&mut self, matrix: &GpuTensor, eps: f32, iters: i32) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(matrix.numel(), 16, "hc_sinkhorn_4x4 expects a 4x4 matrix");
        self.ensure_kernel(
            "hc_sinkhorn_4x4",
            kernels::HC_SINKHORN_4X4_SRC,
            "hc_sinkhorn_4x4",
        )?;
        let m_ptr = matrix.buf.as_ptr();
        let mut eps_v = eps;
        let mut iters_v = iters;
        let mut params: Vec<*mut c_void> = vec![
            &m_ptr as *const _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
            &mut iters_v as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_sinkhorn_4x4",
            [1, 1, 1],
            [32, 1, 1], // single-warp variant: 16 active lanes for the 4x4 matrix
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(m_ptr);
                b.push_f32(eps_v);
                b.push_i32(iters_v);
                b
            },
        )
    }
    pub fn hc_sinkhorn_4x4_batched(
        &mut self,
        matrix: &GpuTensor,
        eps: f32,
        iters: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_sinkhorn_4x4_batched",
            kernels::HC_SINKHORN_4X4_BATCHED_SRC,
            "hc_sinkhorn_4x4_batched",
        )?;
        let m_ptr = matrix.buf.as_ptr();
        let mut eps_v = eps;
        let mut iters_v = iters;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &m_ptr as *const _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
            &mut iters_v as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_sinkhorn_4x4_batched",
            [batch_size as u32, 1, 1],
            [32, 1, 1], // single-warp variant: 16 active lanes per batch row
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(m_ptr);
                b.push_f32(eps_v);
                b.push_i32(iters_v);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn hc_split_finalize_batched(
        &mut self,
        c: &GpuTensor,    // [B, 24]
        pre: &GpuTensor,  // [B, 4]
        post: &GpuTensor, // [B, 4]
        comb: &GpuTensor, // [B, 16]
        post_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_split_finalize_batched",
            kernels::HC_SPLIT_FINALIZE_BATCHED_SRC,
            "hc_split_finalize_batched",
        )?;
        let cp = c.buf.as_ptr();
        let prp = pre.buf.as_ptr();
        let pop = post.buf.as_ptr();
        let cop = comb.buf.as_ptr();
        let mut ps = post_scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &prp as *const _ as *mut c_void,
            &pop as *const _ as *mut c_void,
            &cop as *const _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_split_finalize_batched",
            [batch_size as u32, 1, 1],
            [24, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(prp);
                b.push_ptr(pop);
                b.push_ptr(cop);
                b.push_f32(ps);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn hc_streams_init_from_embed_batched(
        &mut self,
        embed: &GpuTensor,
        streams: &GpuTensor,
        hidden: i32,
        hc_mult: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hc_streams_init_from_embed_batched",
            kernels::HC_STREAMS_INIT_FROM_EMBED_BATCHED_SRC,
            "hc_streams_init_from_embed_batched",
        )?;
        let ep = embed.buf.as_ptr();
        let sp = streams.buf.as_ptr();
        let mut h = hidden;
        let mut hm = hc_mult;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut hm as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "hc_streams_init_from_embed_batched",
            [((hidden + 255) / 256) as u32, batch_size as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(sp);
                b.push_i32(h);
                b.push_i32(hm);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn indexer_compressed_k_score(
        &mut self,
        q_idx: &GpuTensor,       // [H, D] fp16
        k_idx_cache: &GpuTensor, // [H, D, N] fp16
        scores: &GpuTensor,      // [H, N] fp32
        n_idx_heads: i32,
        idx_head_dim: i32,
        n_compressed: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_compressed_k_score",
            kernels::INDEXER_COMPRESSED_K_SCORE_SRC,
            "indexer_compressed_k_score",
        )?;
        let func = &self.functions["indexer_compressed_k_score"];
        let qp = q_idx.buf.as_ptr();
        let kp = k_idx_cache.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut d = idx_head_dim;
        let mut nc = n_compressed;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        // grid.x = heads, grid.y = ceil(N / TILE_POSITIONS=8)
        let grid_y = ((n_compressed + 7) / 8).max(1) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_idx_heads as u32, grid_y, 1],
                [64, 1, 1], // THREADS_PER_BLOCK
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn indexer_kv_gather(
        &mut self,
        k_main_cache: &GpuTensor,
        v_main_cache: &GpuTensor,
        unique_indices: &GpuTensor,
        k_gathered: &GpuTensor,
        v_gathered: &GpuTensor,
        n_kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
        n_unique: i32,
        compress_ratio: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_kv_gather",
            kernels::INDEXER_KV_GATHER_SRC,
            "indexer_kv_gather",
        )?;
        let func = &self.functions["indexer_kv_gather"];
        let kc = k_main_cache.buf.as_ptr();
        let vc = v_main_cache.buf.as_ptr();
        let ui = unique_indices.buf.as_ptr();
        let kg = k_gathered.buf.as_ptr();
        let vg = v_gathered.buf.as_ptr();
        let mut nh = n_kv_heads;
        let mut hd = head_dim;
        let mut ms = max_seq;
        let mut nu = n_unique;
        let mut cr = compress_ratio;
        let mut params: Vec<*mut c_void> = vec![
            &kc as *const _ as *mut c_void,
            &vc as *const _ as *mut c_void,
            &ui as *const _ as *mut c_void,
            &kg as *const _ as *mut c_void,
            &vg as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut cr as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_unique as u32, n_kv_heads as u32, 1],
                [64, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn indexer_relu_score_batched_f32(
        &mut self,
        q: &GpuTensor,           // [B, H, D]
        k_cache: &GpuTensor,     // [N_max, D] shared
        weights: &GpuTensor,     // [B, H]
        n_per_batch: &GpuTensor, // [B] i32
        scores: &GpuTensor,      // [B, N_max] output
        n_idx_heads: i32,        // H
        idx_head_dim: i32,       // D
        n_max: i32,              // N_max (cache slots considered)
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_relu_score_batched",
            kernels::INDEXER_RELU_SCORE_BATCHED_SRC,
            "indexer_relu_score_batched_f32",
        )?;
        let func = &self.functions["indexer_relu_score_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let np = n_per_batch.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut d = idx_head_dim;
        let mut nc = n_max;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_max as u32, batch_size as u32, 1],
                [n_idx_heads as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn indexer_relu_score_wmma_batched_f32(
        &mut self,
        q: &GpuTensor,           // [B, H, D]
        k_cache: &GpuTensor,     // [N_max, D] shared
        weights: &GpuTensor,     // [B, H]
        n_per_batch: &GpuTensor, // [B] i32
        scores: &GpuTensor,      // [B, N_max] output
        n_idx_heads: i32,
        idx_head_dim: i32,
        n_max: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            n_idx_heads, 64,
            "indexer_relu_score_wmma: requires H=64 (got {n_idx_heads})"
        );
        assert_eq!(
            idx_head_dim, 128,
            "indexer_relu_score_wmma: requires idx_head_dim=128 (got {idx_head_dim})"
        );
        self.ensure_kernel(
            "indexer_relu_score_wmma_batched",
            kernels::INDEXER_RELU_SCORE_WMMA_BATCHED_SRC,
            "indexer_relu_score_wmma_batched_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let np = n_per_batch.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut d = idx_head_dim;
        let mut nc = n_max;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let grid_n = (n_max as u32 + 15) / 16;
        self.launch_maybe_blob(
            "indexer_relu_score_wmma_batched_f32",
            [batch_size as u32, grid_n, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(wp);
                b.push_ptr(np);
                b.push_ptr(sp);
                b.push_i32(h);
                b.push_i32(d);
                b.push_i32(nc);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn indexer_relu_score_wmma_batched_klds_gfx1151(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        weights: &GpuTensor,
        n_per_batch: &GpuTensor,
        scores: &GpuTensor,
        n_idx_heads: i32,
        idx_head_dim: i32,
        n_max: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            self.arch, "gfx1151",
            "indexer_relu_score_wmma_batched_klds is gfx1151-only"
        );
        assert_eq!(
            n_idx_heads, 64,
            "indexer_relu_score_wmma_batched_klds: requires H=64 (got {n_idx_heads})"
        );
        assert_eq!(
            idx_head_dim, 128,
            "indexer_relu_score_wmma_batched_klds: requires D=128 (got {idx_head_dim})"
        );
        self.ensure_kernel(
            "indexer_relu_score_wmma_batched_klds_gfx1151",
            kernels::INDEXER_RELU_SCORE_WMMA_BATCHED_KLDS_GFX1151_SRC,
            "indexer_relu_score_wmma_batched_klds_gfx1151",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let np = n_per_batch.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut d = idx_head_dim;
        let mut nc = n_max;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let grid_n = (n_max as u32 + 15) / 16;
        self.launch_maybe_blob(
            "indexer_relu_score_wmma_batched_klds_gfx1151",
            [batch_size as u32, grid_n, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(wp);
                b.push_ptr(np);
                b.push_ptr(sp);
                b.push_i32(h);
                b.push_i32(d);
                b.push_i32(nc);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn indexer_relu_score_f32_buf(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        weights: &GpuTensor,
        scores: &GpuTensor,
        n_buf: &GpuTensor,
        max_n: i32,
        h: i32,
        d: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_relu_score_f32_buf",
            kernels::INDEXER_RELU_SCORE_BUF_SRC,
            "indexer_relu_score_f32_buf",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let nbp = n_buf.buf.as_ptr();
        let mut hi = h;
        let mut di = d;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nbp as *const _ as *mut c_void,
            &mut hi as *mut _ as *mut c_void,
            &mut di as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(qp);
            b.push_ptr(kp);
            b.push_ptr(wp);
            b.push_ptr(sp);
            b.push_ptr(nbp);
            b.push_i32(hi);
            b.push_i32(di);
            b
        };
        self.launch_maybe_blob(
            "indexer_relu_score_f32_buf",
            [max_n as u32, 1, 1],
            [h as u32, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn indexer_top_k(
        &mut self,
        scores: &GpuTensor,      // [H, N] fp32
        top_indices: &GpuTensor, // [H, K] i32
        n_idx_heads: i32,
        n_compressed: i32,
        k: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("indexer_top_k", kernels::INDEXER_TOP_K_SRC, "indexer_top_k")?;
        let func = &self.functions["indexer_top_k"];
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut nc = n_compressed;
        let mut kk = k;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        // shared mem = n_compressed bytes for the `taken` flag array.
        let smem = n_compressed as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_idx_heads as u32, 1, 1],
                [1, 1, 1], // stub single-thread per head
                smem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn indexer_top_k_batched(
        &mut self,
        scores: &GpuTensor,      // [B, H, N_stride] fp32
        top_indices: &GpuTensor, // [B, H, K_stride] i32
        n_idx_heads: i32,
        n_stride: i32, // score storage row stride
        n_iter: i32,   // actual iteration bound (≤ n_stride)
        k_stride: i32, // top_indices storage row stride
        k_fill: i32,   // ranks to fill (rest get -1)
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_top_k_batched",
            kernels::INDEXER_TOP_K_BATCHED_SRC,
            "indexer_top_k_batched",
        )?;
        let func = &self.functions["indexer_top_k_batched"];
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut ns = n_stride;
        let mut ni = n_iter;
        let mut ks = k_stride;
        let mut kf = k_fill;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut kf as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        // Both paths are block-parallel now: the fast path identity-writes
        // k_stride slots across threads; the slow path runs a parallel
        // threshold top-K (block min/max + binary search + compact) over
        // all 256 threads, using only static LDS — so no dynamic smem.
        let smem = 0u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_idx_heads as u32, batch_size as u32, 1],
                [256, 1, 1],
                smem,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// gfx1151-only exact bounded twin of `indexer_top_k_batched`.
    ///
    /// Kept as a separate symbol/method so the portable rank-count remains an
    /// in-process raw-parity reference and so no gfx1100/gfx12 route changes.
    pub fn indexer_top_k_batched_bounded_gfx1151(
        &mut self,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_idx_heads: i32,
        n_stride: i32,
        n_iter: i32,
        k_stride: i32,
        k_fill: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        debug_assert_eq!(self.arch, "gfx1151");
        self.bind_thread()?;
        const MODULE: &str = "indexer_top_k_batched_bounded_gfx1151";
        const SYMBOL: &str = "indexer_top_k_batched_bounded_gfx1151";
        self.ensure_kernel(
            MODULE,
            kernels::INDEXER_TOP_K_BATCHED_BOUNDED_GFX1151_SRC,
            SYMBOL,
        )?;
        let func = &self.functions[SYMBOL];
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut ns = n_stride;
        let mut ni = n_iter;
        let mut ks = k_stride;
        let mut kf = k_fill;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut kf as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_idx_heads as u32, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn indexer_top_k_batched_buf(
        &mut self,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_per_batch: &GpuTensor,
        n_idx_heads: i32,
        n_stride: i32,
        k_stride: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "indexer_top_k_batched_buf",
            kernels::INDEXER_TOP_K_BATCHED_SRC,
            "indexer_top_k_batched_buf",
        )?;
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let np = n_per_batch.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut ns = n_stride;
        let mut ks = k_stride;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &np as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sp);
            b.push_ptr(ti);
            b.push_ptr(np);
            b.push_i32(n_idx_heads);
            b.push_i32(n_stride);
            b.push_i32(k_stride);
            b.push_i32(batch_size);
            b
        };
        self.launch_maybe_blob(
            "indexer_top_k_batched_buf",
            [n_idx_heads as u32, batch_size as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn indexer_top_k_buf(
        &mut self,
        deepseek4_fast_route: bool,
        scores: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        n_idx_heads: i32,
        max_n_compressed: i32,
        max_k: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        static FORCE_SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let force_serial = *FORCE_SERIAL.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_INDEXER_TOPK_SERIAL")
                .ok()
                .as_deref()
                == Some("1")
        });
        static FORCE_BOUNDED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let force_bounded = *FORCE_BOUNDED.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BOUNDED")
                .ok()
                .as_deref()
                == Some("1")
        });
        static FORCE_BLOCK1024: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let force_block1024 = *FORCE_BLOCK1024.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BLOCK1024")
                .ok()
                .as_deref()
                == Some("1")
        });
        static FORCE_UNROLLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let force_unrolled = *FORCE_UNROLLED.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_INDEXER_TOPK_UNROLLED")
                .ok()
                .as_deref()
                == Some("1")
        });
        // gfx1151 keeps its certified route selection. gfx1201 reuses the
        // wave-size-independent bounded source, compiled into its own exact
        // device code object after the raw-i32 parity channel passed.
        let gfx1151_parallel = self.arch == "gfx1151" && deepseek4_fast_route && !force_serial;
        let gfx1201_bounded = self.arch == "gfx1201" && deepseek4_fast_route && !force_serial;
        let (logical_name, source, symbol, block, smem) = if gfx1151_parallel {
            if force_unrolled {
                (
                    "indexer_top_k_buf_unrolled",
                    kernels::INDEXER_TOP_K_BUF_UNROLLED_GFX1151_SRC,
                    "indexer_top_k_buf_parallel",
                    [if force_block1024 { 1024 } else { 256 }, 1, 1],
                    0,
                )
            } else if force_bounded {
                (
                    "indexer_top_k_buf_bounded",
                    kernels::INDEXER_TOP_K_BUF_BOUNDED_GFX1151_SRC,
                    "indexer_top_k_buf_parallel",
                    [if force_block1024 { 1024 } else { 256 }, 1, 1],
                    0,
                )
            } else {
                (
                    "indexer_top_k_buf_parallel",
                    kernels::INDEXER_TOP_K_BUF_PARALLEL_GFX1151_SRC,
                    "indexer_top_k_buf_parallel",
                    [256, 1, 1],
                    0,
                )
            }
        } else if gfx1201_bounded {
            (
                "indexer_top_k_buf_bounded_gfx1201",
                kernels::INDEXER_TOP_K_BUF_BOUNDED_GFX1151_SRC,
                "indexer_top_k_buf_parallel",
                [256, 1, 1],
                0,
            )
        } else {
            (
                "indexer_top_k_buf",
                kernels::INDEXER_TOP_K_BUF_SRC,
                "indexer_top_k_buf",
                [128, 1, 1],
                max_n_compressed as u32,
            )
        };
        self.ensure_kernel(logical_name, source, symbol)?;
        let sp = scores.buf.as_ptr();
        let ti = top_indices.buf.as_ptr();
        let nbp = n_compressed_buf.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let mut h = n_idx_heads;
        let mut mk = max_k;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &nbp as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut mk as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sp);
            b.push_ptr(ti);
            b.push_ptr(nbp);
            b.push_ptr(kbp);
            b.push_i32(h);
            b.push_i32(mk);
            b
        };
        self.launch_maybe_blob(
            symbol,
            [n_idx_heads as u32, 1, 1],
            block,
            smem,
            &mut params,
            blob_builder,
        )
    }
    /// F3/G1: multi-block two-stage merge-tree indexer top-K for the DeepSeek
    /// V4 long-context path. Stage 1 (`indexer_topk_chunk_sort`) bitonic-sorts
    /// each 512-candidate chunk of `scores` into a per-chunk descending run in
    /// the `ws_scores` / `ws_indices` workspace (run j at element offset
    /// j*512); `ceil(log2(n_runs))` rounds of `indexer_topk_merge_round` then
    /// pairwise-merge runs, keeping the better 512 of each pair;
    /// `indexer_topk_finalize` writes the surviving run to `top_indices`.
    /// Output is byte-identical IN ORDER to the incumbent rank-count kernel:
    /// total order (score desc, index asc), eligibility `score > -FLT_MAX`
    /// canonicalised at load so NaN never reaches the comparator, and the
    /// `n_compressed <= max_k` identity fast path (condition is `<= max_k`,
    /// not `<= k`). Correctness of the decomposition: a global top-K element
    /// is in the top-K of its own chunk, so the union of per-chunk top-512
    /// lists contains the global top-512 and every merge round preserves that.
    ///
    /// This lives on the generic `Gpu` surface — NOT behind `Gfx942Device`.
    /// The kernel uses only `__syncthreads()` and block-uniform loop bounds,
    /// so one portable code object is valid on wave32 (RDNA, gfx1151) and
    /// wave64 (CDNA, gfx942) alike, and BOTH arches select it (F3 on gfx942,
    /// G1 on gfx1151). `Gfx942Device` exists to gate CDNA-specific code
    /// objects; this is not one, so parking it there would needlessly fork
    /// the gfx1151 route off the shared implementation.
    ///
    /// Graph safety: `n_runs` and `rounds` are computed on the HOST from
    /// `max_compressed` alone, so the launch count (`1 + rounds + 1`) and
    /// every grid are fixed for a given model config — a captured hipGraph or
    /// Redline tape is exact. The live candidate count reaches the device
    /// solely through `n_compressed_buf`, identical to `indexer_top_k_buf`.
    /// Single-head only: the workspace has no head dimension; ds4 dispatches
    /// the indexer with `n_idx_heads == 1`.
    ///
    /// The sequence is LAUNCH-bound, not work-bound: measured per-launch cost
    /// is 7.3–13.5 us and barely moves with N (gfx1151: 0.0146 ms at N=512
    /// with 2 launches vs 0.1245 ms at N=262144 with 11 — 512x the data for
    /// 8.5x the time). The compute is already negligible, so no per-block
    /// work-reduction cleverness belongs here. The corollary: hipGraph
    /// capture is expected to make this materially faster still, because
    /// graph replay avoids the per-dispatch cost that dominates. Benchmark
    /// conclusions drawn with graphs disabled do not transfer.
    #[allow(clippy::too_many_arguments)]
    pub fn indexer_top_k_two_stage(
        &mut self,
        scores: &GpuTensor,
        ws_scores: &GpuTensor,
        ws_indices: &GpuTensor,
        top_indices: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        k_buf: &GpuTensor,
        max_compressed: i32,
        max_k: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // One module, three symbols: compile/load once, extract all three.
        self.ensure_kernel(
            "indexer_topk_twostage",
            kernels::INDEXER_TOPK_TWOSTAGE_SRC,
            "indexer_topk_chunk_sort",
        )?;
        self.ensure_kernel(
            "indexer_topk_twostage",
            kernels::INDEXER_TOPK_TWOSTAGE_SRC,
            "indexer_topk_merge_round",
        )?;
        self.ensure_kernel(
            "indexer_topk_twostage",
            kernels::INDEXER_TOPK_TWOSTAGE_SRC,
            "indexer_topk_finalize",
        )?;
        let sp = scores.buf.as_ptr();
        let wss = ws_scores.buf.as_ptr();
        let wsi = ws_indices.buf.as_ptr();
        let tip = top_indices.buf.as_ptr();
        let nbp = n_compressed_buf.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        // 512 candidates per chunk; rounds = ceil(log2(n_runs)) so repeated
        // pairwise merges converge to a single surviving run.
        let n_runs = (max_compressed as usize).div_ceil(512);
        let mut rounds = 0usize;
        while (1usize << rounds) < n_runs {
            rounds += 1;
        }
        // Stage 1: sort each chunk into its own run.
        {
            let mut nr = n_runs as i32;
            let mut params: Vec<*mut c_void> = vec![
                &sp as *const _ as *mut c_void,
                &wss as *const _ as *mut c_void,
                &wsi as *const _ as *mut c_void,
                &nbp as *const _ as *mut c_void,
                &mut nr as *mut _ as *mut c_void,
            ];
            let blob_builder = || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(sp);
                b.push_ptr(wss);
                b.push_ptr(wsi);
                b.push_ptr(nbp);
                b.push_i32(nr);
                b
            };
            self.launch_maybe_blob(
                "indexer_topk_chunk_sort",
                [n_runs as u32, 1, 1],
                [256, 1, 1],
                0,
                &mut params,
                blob_builder,
            )?;
        }
        // Rounds: pair p merges run p*2*run_step with run p*2*run_step+run_step
        // into run p*2*run_step. run_step doubles each round; n_pairs halves.
        for r in 0..rounds {
            let run_step = 1usize << r;
            let n_pairs = n_runs.div_ceil(2 << r);
            let mut nr = n_runs as i32;
            let mut rs = run_step as i32;
            let mut np = n_pairs as i32;
            let mut params: Vec<*mut c_void> = vec![
                &wss as *const _ as *mut c_void,
                &wsi as *const _ as *mut c_void,
                &mut nr as *mut _ as *mut c_void,
                &mut rs as *mut _ as *mut c_void,
                &mut np as *mut _ as *mut c_void,
            ];
            let blob_builder = || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wss);
                b.push_ptr(wsi);
                b.push_i32(nr);
                b.push_i32(rs);
                b.push_i32(np);
                b
            };
            self.launch_maybe_blob(
                "indexer_topk_merge_round",
                [n_pairs as u32, 1, 1],
                [256, 1, 1],
                0,
                &mut params,
                blob_builder,
            )?;
        }
        // Finalize: emit the surviving run as top_indices.
        {
            let mut mk = max_k;
            let mut params: Vec<*mut c_void> = vec![
                &wss as *const _ as *mut c_void,
                &wsi as *const _ as *mut c_void,
                &tip as *const _ as *mut c_void,
                &nbp as *const _ as *mut c_void,
                &kbp as *const _ as *mut c_void,
                &mut mk as *mut _ as *mut c_void,
            ];
            let blob_builder = || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wss);
                b.push_ptr(wsi);
                b.push_ptr(tip);
                b.push_ptr(nbp);
                b.push_ptr(kbp);
                b.push_i32(mk);
                b
            };
            self.launch_maybe_blob(
                "indexer_topk_finalize",
                [1, 1, 1],
                [256, 1, 1],
                0,
                &mut params,
                blob_builder,
            )
        }
    }
    pub fn rope_tail_interleaved(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_interleaved",
            kernels::ROPE_TAIL_INTERLEAVED_SRC,
            "rope_tail_interleaved_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = pos_buf.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        self.launch_maybe_blob(
            "rope_tail_interleaved_f32",
            [(half + 31) / 32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(pp);
                b.push_i32(nq);
                b.push_i32(nk);
                b.push_i32(hd);
                b.push_i32(nr);
                b.push_f32(fb);
                b
            },
        )
    }

    /// Exact-gfx1201 DS4 indexer-Q RoPE at H64/D128/R64.
    ///
    /// `head_waves` selects 8, 16, or 32 independent head-strided waves.
    /// Keeping this explicit lets the channel micro screen the occupancy/SFU balance;
    /// the product route passes the certified fixed geometry.
    pub fn rope_tail_interleaved_h64d128r64_gfx1201(
        &mut self,
        q: &GpuTensor,
        pos_buf: &GpuTensor,
        freq_base: f32,
        head_waves: u32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "rope_tail_interleaved_h64d128r64_gfx1201 requires gfx1201, got {}",
                    self.arch
                ),
            ));
        }
        if !matches!(head_waves, 8 | 16 | 32) {
            return Err(hip_bridge::HipError::new(
                0,
                "rope_tail_interleaved_h64d128r64_gfx1201 requires 8, 16, or 32 head waves",
            ));
        }
        if q.numel() != 64 * 128 || pos_buf.numel() != 1 {
            return Err(hip_bridge::HipError::new(
                0,
                "rope_tail_interleaved_h64d128r64_gfx1201 shape mismatch",
            ));
        }
        self.ensure_kernel(
            "rope_tail_interleaved_h64d128r64_gfx1201",
            kernels::ROPE_TAIL_INTERLEAVED_H64D128R64_GFX1201_SRC,
            "rope_tail_interleaved_h64d128r64_gfx1201",
        )?;
        let qp = q.buf.as_ptr();
        let pp = pos_buf.buf.as_ptr();
        let mut hd = 128_i32;
        let mut nr = 64_i32;
        let mut fb = freq_base;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "rope_tail_interleaved_h64d128r64_gfx1201",
            [1, head_waves, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(pp);
                b.push_i32(hd);
                b.push_i32(nr);
                b.push_f32(fb);
                b
            },
        )
    }

    pub fn rope_tail_interleaved_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_interleaved_batched",
            kernels::ROPE_TAIL_INTERLEAVED_BATCHED_SRC,
            "rope_tail_interleaved_batched_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        self.launch_maybe_blob(
            "rope_tail_interleaved_batched_f32",
            [(half + 31) / 32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(pp);
                b.push_i32(nq);
                b.push_i32(nk);
                b.push_i32(hd);
                b.push_i32(nr);
                b.push_f32(fb);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn rope_tail_yarn_interleaved(
        &mut self,
        deepseek4_gfx1151_route: bool,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        corr_low: f32,
        corr_high: f32,
        inverse: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let wide = self.arch == "gfx1151" && n_rot <= 128 && deepseek4_gfx1151_route;
        let (logical_name, symbol, block) = if wide {
            (
                "rope_tail_yarn_interleaved_wide",
                "rope_tail_yarn_interleaved_wide_f32",
                [256, 1, 1],
            )
        } else {
            (
                "rope_tail_yarn_interleaved",
                "rope_tail_yarn_interleaved_f32",
                [32, 1, 1],
            )
        };
        self.ensure_kernel(
            logical_name,
            kernels::ROPE_TAIL_YARN_INTERLEAVED_SRC,
            symbol,
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = pos_buf.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut fs = freq_scale;
        let mut ef = ext_factor;
        let mut af = attn_factor;
        let mut cl = corr_low;
        let mut ch = corr_high;
        let mut inv = inverse;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut fs as *mut _ as *mut c_void,
            &mut ef as *mut _ as *mut c_void,
            &mut af as *mut _ as *mut c_void,
            &mut cl as *mut _ as *mut c_void,
            &mut ch as *mut _ as *mut c_void,
            &mut inv as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(symbol, [1, 1, 1], block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(qp);
            b.push_ptr(kp);
            b.push_ptr(pp);
            b.push_i32(nq);
            b.push_i32(nk);
            b.push_i32(hd);
            b.push_i32(nr);
            b.push_f32(fb);
            b.push_f32(fs);
            b.push_f32(ef);
            b.push_f32(af);
            b.push_f32(cl);
            b.push_f32(ch);
            b.push_i32(inv);
            b
        })
    }
    pub fn rope_tail_yarn_interleaved_at_slot_buf(
        &mut self,
        base: &GpuTensor,
        pos_buf: &GpuTensor,
        slot_buf: &GpuTensor,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        corr_low: f32,
        corr_high: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_yarn_interleaved_at_slot_buf",
            kernels::ROPE_TAIL_YARN_INTERLEAVED_AT_SLOT_BUF_SRC,
            "rope_tail_yarn_interleaved_at_slot_buf_f32",
        )?;
        let bp = base.buf.as_ptr();
        let pp = pos_buf.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut fs = freq_scale;
        let mut ef = ext_factor;
        let mut af = attn_factor;
        let mut cl = corr_low;
        let mut ch = corr_high;
        let mut params: Vec<*mut c_void> = vec![
            &bp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut fs as *mut _ as *mut c_void,
            &mut ef as *mut _ as *mut c_void,
            &mut af as *mut _ as *mut c_void,
            &mut cl as *mut _ as *mut c_void,
            &mut ch as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        let block = 32u32;
        let grid = (half + block - 1) / block;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(bp);
            b.push_ptr(pp);
            b.push_ptr(sb);
            b.push_i32(hd);
            b.push_i32(nr);
            b.push_f32(fb);
            b.push_f32(fs);
            b.push_f32(ef);
            b.push_f32(af);
            b.push_f32(cl);
            b.push_f32(ch);
            b
        };
        self.launch_maybe_blob(
            "rope_tail_yarn_interleaved_at_slot_buf_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn rope_tail_yarn_interleaved_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        n_heads_q: i32,
        n_heads_k: i32,
        head_dim: i32,
        n_rot: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        corr_low: f32,
        corr_high: f32,
        inverse: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "rope_tail_yarn_interleaved_batched",
            kernels::ROPE_TAIL_YARN_INTERLEAVED_BATCHED_SRC,
            "rope_tail_yarn_interleaved_batched_f32",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let mut nq = n_heads_q;
        let mut nk = n_heads_k;
        let mut hd = head_dim;
        let mut nr = n_rot;
        let mut fb = freq_base;
        let mut fs = freq_scale;
        let mut ef = ext_factor;
        let mut af = attn_factor;
        let mut cl = corr_low;
        let mut ch = corr_high;
        let mut inv = inverse;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut fs as *mut _ as *mut c_void,
            &mut ef as *mut _ as *mut c_void,
            &mut af as *mut _ as *mut c_void,
            &mut cl as *mut _ as *mut c_void,
            &mut ch as *mut _ as *mut c_void,
            &mut inv as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let half = (n_rot / 2) as u32;
        self.launch_maybe_blob(
            "rope_tail_yarn_interleaved_batched_f32",
            [(half + 31) / 32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(pp);
                b.push_i32(nq);
                b.push_i32(nk);
                b.push_i32(hd);
                b.push_i32(nr);
                b.push_f32(fb);
                b.push_f32(fs);
                b.push_f32(ef);
                b.push_f32(af);
                b.push_f32(cl);
                b.push_f32(ch);
                b.push_i32(inv);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn state_overlap_shift_f32_buf(
        &mut self,
        state: &GpuTensor,
        commit_slot_buf: &GpuTensor,
        ratio: i32,
        proj_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "state_overlap_shift_f32_buf",
            kernels::STATE_OVERLAP_SHIFT_F32_BUF_SRC,
            "state_overlap_shift_f32_buf",
        )?;
        let stp = state.buf.as_ptr();
        let cp = commit_slot_buf.buf.as_ptr();
        let mut rv = ratio;
        let mut pd = proj_dim;
        let mut params: Vec<*mut c_void> = vec![
            &stp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut rv as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
        ];
        let total = (ratio * proj_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(stp);
            b.push_ptr(cp);
            b.push_i32(rv);
            b.push_i32(pd);
            b
        };
        self.launch_maybe_blob(
            "state_overlap_shift_f32_buf",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn state_ring_write_f32_buf(
        &mut self,
        src: &GpuTensor,
        state: &GpuTensor,
        ring_slot_buf: &GpuTensor,
        proj_dim: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "state_ring_write_f32_buf",
            kernels::STATE_RING_WRITE_F32_BUF_SRC,
            "state_ring_write_f32_buf",
        )?;
        let sp = src.buf.as_ptr();
        let stp = state.buf.as_ptr();
        let rp = ring_slot_buf.buf.as_ptr();
        let mut pd = proj_dim;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &stp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((proj_dim as u32) + block - 1) / block;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sp);
            b.push_ptr(stp);
            b.push_ptr(rp);
            b.push_i32(pd);
            b
        };
        self.launch_maybe_blob(
            "state_ring_write_f32_buf",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn swa_ring_write_batched_f32(
        &mut self,
        kv_batch: &GpuTensor, // [B, head_dim]
        cache: &GpuTensor,    // [n_kv_heads, head_dim, window]
        n_kv_heads: i32,
        head_dim: i32,
        window: i32,
        start_pos: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_ring_write_batched",
            kernels::SWA_RING_WRITE_BATCHED_SRC,
            "swa_ring_write_batched_f32",
        )?;
        let func = &self.functions["swa_ring_write_batched_f32"];
        let kp = kv_batch.buf.as_ptr();
        let cp = cache.buf.as_ptr();
        let mut nh = n_kv_heads;
        let mut hd = head_dim;
        let mut w = window;
        let mut sp = start_pos;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [((head_dim + 255) / 256) as u32, batch_size as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn swa_ring_write_batched_pos_f32(
        &mut self,
        kv_batch: &GpuTensor,
        cache: &GpuTensor,
        positions: &GpuTensor,
        n_kv_heads: i32,
        head_dim: i32,
        window: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_ring_write_batched_pos",
            kernels::SWA_RING_WRITE_BATCHED_SRC,
            "swa_ring_write_batched_pos_f32",
        )?;
        let kp = kv_batch.buf.as_ptr();
        let cp = cache.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let mut nh = n_kv_heads;
        let mut hd = head_dim;
        let mut w = window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(kp);
            b.push_ptr(cp);
            b.push_ptr(pp);
            b.push_i32(n_kv_heads);
            b.push_i32(head_dim);
            b.push_i32(window);
            b.push_i32(batch_size);
            b
        };
        self.launch_maybe_blob(
            "swa_ring_write_batched_pos_f32",
            [((head_dim + 255) / 256) as u32, batch_size as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn swa_ring_write_f32_buf(
        &mut self,
        kv: &GpuTensor,
        cache: &GpuTensor,
        slot_buf: &GpuTensor,
        n_kv_heads: i32,
        head_dim: i32,
        window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_ring_write_f32_buf",
            kernels::SWA_RING_WRITE_BUF_SRC,
            "swa_ring_write_f32_buf",
        )?;
        let kp = kv.buf.as_ptr();
        let cp = cache.buf.as_ptr();
        let sb = slot_buf.buf.as_ptr();
        let mut nh = n_kv_heads;
        let mut hd = head_dim;
        let mut wn = window;
        let mut params: Vec<*mut c_void> = vec![
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &sb as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
        ];
        let grid = ((head_dim + 255) / 256) as u32;
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(kp);
            b.push_ptr(cp);
            b.push_ptr(sb);
            b.push_i32(nh);
            b.push_i32(hd);
            b.push_i32(wn);
            b
        };
        self.launch_maybe_blob(
            "swa_ring_write_f32_buf",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn swa_visibility_stage_batched(
        &mut self,
        ring: &GpuTensor,     // [head_dim, swa_window] pre-chunk
        kv_batch: &GpuTensor, // [B, head_dim] within-chunk
        staged: &GpuTensor,   // [B, head_dim, swa_window] output
        start_pos: i32,
        swa_window: i32,
        head_dim: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_visibility_stage_batched",
            kernels::SWA_VISIBILITY_STAGE_BATCHED_SRC,
            "swa_visibility_stage_batched",
        )?;
        let func = &self.functions["swa_visibility_stage_batched"];
        let rp = ring.buf.as_ptr();
        let kp = kv_batch.buf.as_ptr();
        let sp = staged.buf.as_ptr();
        let mut sp_i = start_pos;
        let mut sw = swa_window;
        let mut hd = head_dim;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut sp_i as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [head_dim as u32, batch_size as u32, 1],
                [swa_window as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn swa_visibility_stage_batched_pos(
        &mut self,
        ring: &GpuTensor,
        kv_batch: &GpuTensor,
        staged: &GpuTensor,
        positions: &GpuTensor,
        swa_window: i32,
        head_dim: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "swa_visibility_stage_batched_pos",
            kernels::SWA_VISIBILITY_STAGE_BATCHED_SRC,
            "swa_visibility_stage_batched_pos",
        )?;
        let rp = ring.buf.as_ptr();
        let kp = kv_batch.buf.as_ptr();
        let sp = staged.buf.as_ptr();
        let pp = positions.buf.as_ptr();
        let mut sw = swa_window;
        let mut hd = head_dim;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(rp);
            b.push_ptr(kp);
            b.push_ptr(sp);
            b.push_ptr(pp);
            b.push_i32(swa_window);
            b.push_i32(head_dim);
            b.push_i32(batch_size);
            b
        };
        self.launch_maybe_blob(
            "swa_visibility_stage_batched_pos",
            [head_dim as u32, batch_size as u32, 1],
            [swa_window as u32, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn deepseek4_attn_pos0(
        &mut self,
        q: &GpuTensor,
        kv: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_pos0",
            kernels::V4F_ATTN_POS0_SRC,
            "deepseek4_attn_pos0",
        )?;
        let func = &self.functions["deepseek4_attn_pos0"];
        let qp = q.buf.as_ptr();
        let kp = kv.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_attn_swa(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        n_valid: i32,
        window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa",
            kernels::V4F_ATTN_SWA_SRC,
            "deepseek4_attn_swa",
        )?;
        let func = &self.functions["deepseek4_attn_swa"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut nv = n_valid;
        let mut wn = window;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_attn_swa_batched(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        window: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_batched",
            kernels::V4F_ATTN_SWA_BATCHED_SRC,
            "deepseek4_attn_swa_batched",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut wn = window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "deepseek4_attn_swa_batched",
            [n_heads as u32, batch_size as u32, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(vp);
                b.push_ptr(sp);
                b.push_ptr(nvp);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(og);
                b.push_i32(wn);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn deepseek4_attn_swa_batched_debug(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_arr: &GpuTensor,
        attn_out: &GpuTensor,
        debug_max: &GpuTensor,    // [batch, n_heads] f32
        debug_sumexp: &GpuTensor, // [batch, n_heads] f32
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        window: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_batched_debug",
            kernels::V4F_ATTN_SWA_BATCHED_DEBUG_SRC,
            "deepseek4_attn_swa_batched_debug",
        )?;
        let func = &self.functions["deepseek4_attn_swa_batched_debug"];
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let dmp = debug_max.buf.as_ptr();
        let dsp = debug_sumexp.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut wn = window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &dmp as *const _ as *mut c_void,
            &dsp as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_attn_swa_buf(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_valid_buf: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        o_groups: i32,
        window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_buf",
            kernels::V4F_ATTN_SWA_BUF_SRC,
            "deepseek4_attn_swa_buf",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k_cache.buf.as_ptr();
        let vp = v_cache.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let nvp = n_valid_buf.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut og = o_groups;
        let mut wn = window;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut og as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(qp);
            b.push_ptr(kp);
            b.push_ptr(vp);
            b.push_ptr(sp);
            b.push_ptr(op);
            b.push_ptr(nvp);
            b.push_i32(nh);
            b.push_i32(hd);
            b.push_i32(og);
            b.push_i32(wn);
            b
        };
        self.launch_maybe_blob(
            "deepseek4_attn_swa_buf",
            [n_heads as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn deepseek4_attn_swa_topk_batched_f32(
        &mut self,
        q: &GpuTensor,
        swa_k: &GpuTensor,
        swa_v: &GpuTensor,
        topk_k: &GpuTensor,
        topk_v: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_swa_arr: &GpuTensor,
        n_active_topk_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_topk_batched",
            kernels::V4F_ATTN_SWA_TOPK_BATCHED_SRC,
            "deepseek4_attn_swa_topk_batched_f32",
        )?;
        let func = &self.functions["deepseek4_attn_swa_topk_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = swa_k.buf.as_ptr();
        let vp = swa_v.buf.as_ptr();
        let tkp = topk_k.buf.as_ptr();
        let tvp = topk_v.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_swa_arr.buf.as_ptr();
        let nap = n_active_topk_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut sw = swa_window;
        let mut tw = topk_window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &tkp as *const _ as *mut c_void,
            &tvp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &nap as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut tw as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    pub fn deepseek4_attn_swa_topk_direct_batched_f32(
        &mut self,
        q: &GpuTensor,
        swa_k: &GpuTensor,
        swa_v: &GpuTensor,
        kv_cache: &GpuTensor,
        topk_idx: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_swa_arr: &GpuTensor,
        n_active_topk_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
        n_compressed: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_attn_swa_topk_direct_batched",
            kernels::V4F_ATTN_SWA_TOPK_DIRECT_BATCHED_SRC,
            "deepseek4_attn_swa_topk_direct_batched_f32",
        )?;
        let func = &self.functions["deepseek4_attn_swa_topk_direct_batched_f32"];
        let qp = q.buf.as_ptr();
        let kp = swa_k.buf.as_ptr();
        let vp = swa_v.buf.as_ptr();
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_swa_arr.buf.as_ptr();
        let nap = n_active_topk_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut sw = swa_window;
        let mut tw = topk_window;
        let mut nc = n_compressed;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &nap as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut tw as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [head_dim as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Head-batched f16-WMMA DSA attention (direct top-K) — faster sibling of
    /// `deepseek4_attn_swa_topk_direct_batched_f32`. K=V tied (single `swa_kv`);
    /// `max_n_total` (= max over batches of n_valid_swa + n_active_topk) sizes
    /// the per-block score LDS. Returns Err if the LDS would exceed 64 KB (the
    /// caller falls back to the f32 kernel). Requires n_heads%16==0, head_dim%16==0.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_topk_direct_wmma(
        &mut self,
        q: &GpuTensor,
        swa_kv: &GpuTensor,
        kv_cache: &GpuTensor,
        topk_idx: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_swa_arr: &GpuTensor,
        n_active_topk_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
        n_compressed: i32,
        batch_size: i32,
        max_n_total: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            n_heads % 16,
            0,
            "direct_wmma: n_heads must be %16 (got {n_heads})"
        );
        debug_assert_eq!(
            head_dim % 16,
            0,
            "direct_wmma: head_dim must be %16 (got {head_dim})"
        );
        let n_pad = ((max_n_total + 15) / 16) * 16;
        let lds_bytes = 16 * head_dim * 2 + 16 * n_pad * 4; // q f16 + s f32
        if lds_bytes > 64 * 1024 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("direct_wmma: LDS {lds_bytes} > 64KB (max_n_total={max_n_total})"),
            ));
        }
        self.ensure_kernel(
            "deepseek4_attn_swa_topk_direct_wmma",
            kernels::V4F_ATTN_SWA_TOPK_DIRECT_WMMA_SRC,
            "deepseek4_attn_swa_topk_direct_wmma",
        )?;
        let qp = q.buf.as_ptr();
        let kp = swa_kv.buf.as_ptr();
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_swa_arr.buf.as_ptr();
        let nap = n_active_topk_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut sw = swa_window;
        let mut tw = topk_window;
        let mut nc = n_compressed;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &nap as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut tw as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        // Capture-safe launch (blob path under the new base's prefill capture).
        self.launch_maybe_blob(
            "deepseek4_attn_swa_topk_direct_wmma",
            [(n_heads / 16) as u32, batch_size as u32, 1],
            [256, 1, 1], // 8 warps split the score n-tiles / output d-tiles
            lds_bytes as u32,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(sp);
                b.push_ptr(nvp);
                b.push_ptr(nap);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(sw);
                b.push_i32(tw);
                b.push_i32(nc);
                b.push_i32(bs);
                b
            },
        )
    }

    /// Head-batched f16-WMMA DSA attention (gathered top-K) — faster sibling of
    /// `deepseek4_attn_swa_topk_batched_f32`. K=V tied for both SWA (`swa_kv`)
    /// and top-K (`topk_kv`, the staged d-major buffer). Same LDS/fallback rules
    /// as `deepseek4_attn_swa_topk_direct_wmma`.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_attn_swa_topk_batched_wmma(
        &mut self,
        q: &GpuTensor,
        swa_kv: &GpuTensor,
        topk_kv: &GpuTensor,
        attn_sink: &GpuTensor,
        n_valid_swa_arr: &GpuTensor,
        n_active_topk_arr: &GpuTensor,
        attn_out: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
        batch_size: i32,
        max_n_total: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        debug_assert_eq!(
            n_heads % 16,
            0,
            "batched_wmma: n_heads must be %16 (got {n_heads})"
        );
        debug_assert_eq!(
            head_dim % 16,
            0,
            "batched_wmma: head_dim must be %16 (got {head_dim})"
        );
        let n_pad = ((max_n_total + 15) / 16) * 16;
        let lds_bytes = 16 * head_dim * 2 + 16 * n_pad * 4;
        if lds_bytes > 64 * 1024 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("batched_wmma: LDS {lds_bytes} > 64KB (max_n_total={max_n_total})"),
            ));
        }
        self.ensure_kernel(
            "deepseek4_attn_swa_topk_batched_wmma",
            kernels::V4F_ATTN_SWA_TOPK_BATCHED_WMMA_SRC,
            "deepseek4_attn_swa_topk_batched_wmma",
        )?;
        let qp = q.buf.as_ptr();
        let kp = swa_kv.buf.as_ptr();
        let tp = topk_kv.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let nvp = n_valid_swa_arr.buf.as_ptr();
        let nap = n_active_topk_arr.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut sw = swa_window;
        let mut tw = topk_window;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &nap as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut tw as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        // Capture-safe launch: the new base graph-captures the prefill, and
        // the void**-kernarg path records dangling stack pointers that break
        // on replay. launch_maybe_blob uses the blob path under capture.
        self.launch_maybe_blob(
            "deepseek4_attn_swa_topk_batched_wmma",
            [(n_heads / 16) as u32, batch_size as u32, 1],
            [256, 1, 1],
            lds_bytes as u32,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(nvp);
                b.push_ptr(nap);
                b.push_ptr(op);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(sw);
                b.push_i32(tw);
                b.push_i32(bs);
                b
            },
        )
    }

    pub fn deepseek4_attn_swa_topk_f32_buf(
        &mut self,
        deepseek4_scoregrid_route: bool,
        q: &GpuTensor,
        swa_k: &GpuTensor,
        swa_v: &GpuTensor,
        topk_k: &GpuTensor,
        topk_v: &GpuTensor,
        attn_sink: &GpuTensor,
        attn_out: &GpuTensor,
        n_valid_swa_buf: &GpuTensor,
        n_active_topk_buf: &GpuTensor,
        n_heads: i32,
        head_dim: i32,
        swa_window: i32,
        topk_window: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        static WARP_GFX1151: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static ILP4_GFX1151: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static SCOREGRID_XLANE_GFX1151: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static SCOREGRID_LARGE_SERIAL_GFX1151: std::sync::OnceLock<bool> =
            std::sync::OnceLock::new();
        let scoregrid_arch = self.arch == "gfx1151" || self.arch == "gfx1201";
        let scoregrid = scoregrid_arch && head_dim == 512 && deepseek4_scoregrid_route;
        let ilp4 = self.arch == "gfx1151"
            && head_dim == 512
            && *ILP4_GFX1151.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_ILP4")
                    .ok()
                    .as_deref()
                    == Some("1")
            });
        let warp = self.arch == "gfx1151"
            && head_dim == 512
            && *WARP_GFX1151.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_WARP")
                    .ok()
                    .as_deref()
                    == Some("1")
            });
        let scoregrid_xlane = scoregrid
            && *SCOREGRID_XLANE_GFX1151.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_XLANE")
                    .ok()
                    .as_deref()
                    == Some("1")
            });
        let scoregrid_large_serial = scoregrid
            && *SCOREGRID_LARGE_SERIAL_GFX1151.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_LARGE_SERIAL")
                    .ok()
                    .as_deref()
                    == Some("1")
            });
        let (logical_name, symbol, block, source) = if scoregrid_large_serial {
            (
                "deepseek4_attn_swa_topk_scoregrid_large_serial_gfx1151",
                "deepseek4_attn_swa_topk_scoregrid_f32_buf",
                [512, 1, 1],
                kernels::V4F_ATTN_SWA_TOPK_BUF_LARGE_SERIAL_GFX1151_SRC,
            )
        } else if scoregrid_xlane {
            (
                "deepseek4_attn_swa_topk_scoregrid_xlane_gfx1151",
                "deepseek4_attn_swa_topk_scoregrid_f32_buf",
                [512, 1, 1],
                kernels::V4F_ATTN_SWA_TOPK_BUF_XLANE_GFX1151_SRC,
            )
        } else if scoregrid {
            (
                "deepseek4_attn_swa_topk_scoregrid_f32_buf",
                "deepseek4_attn_swa_topk_scoregrid_f32_buf",
                [512, 1, 1],
                kernels::V4F_ATTN_SWA_TOPK_BUF_SRC,
            )
        } else if ilp4 {
            (
                "deepseek4_attn_swa_topk_ilp4_f32_buf",
                "deepseek4_attn_swa_topk_ilp4_f32_buf",
                [512, 1, 1],
                kernels::V4F_ATTN_SWA_TOPK_BUF_SRC,
            )
        } else if warp {
            (
                "deepseek4_attn_swa_topk_warp_f32_buf",
                "deepseek4_attn_swa_topk_warp_f32_buf",
                [256, 1, 1],
                kernels::V4F_ATTN_SWA_TOPK_BUF_SRC,
            )
        } else {
            (
                "deepseek4_attn_swa_topk_f32_buf",
                "deepseek4_attn_swa_topk_f32_buf",
                [head_dim as u32, 1, 1],
                kernels::V4F_ATTN_SWA_TOPK_BUF_SRC,
            )
        };
        self.ensure_kernel(logical_name, source, symbol)?;
        let qp = q.buf.as_ptr();
        let kp = swa_k.buf.as_ptr();
        let vp = swa_v.buf.as_ptr();
        let tkp = topk_k.buf.as_ptr();
        let tvp = topk_v.buf.as_ptr();
        let sp = attn_sink.buf.as_ptr();
        let op = attn_out.buf.as_ptr();
        let nvp = n_valid_swa_buf.buf.as_ptr();
        let nap = n_active_topk_buf.buf.as_ptr();
        let mut nh = n_heads;
        let mut hd = head_dim;
        let mut sw = swa_window;
        let mut tw = topk_window;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &tkp as *const _ as *mut c_void,
            &tvp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nvp as *const _ as *mut c_void,
            &nap as *const _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sw as *mut _ as *mut c_void,
            &mut tw as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(qp);
            b.push_ptr(kp);
            b.push_ptr(vp);
            b.push_ptr(tkp);
            b.push_ptr(tvp);
            b.push_ptr(sp);
            b.push_ptr(op);
            b.push_ptr(nvp);
            b.push_ptr(nap);
            b.push_i32(nh);
            b.push_i32(hd);
            b.push_i32(sw);
            b.push_i32(tw);
            b
        };
        self.launch_maybe_blob(
            symbol,
            [n_heads as u32, 1, 1],
            block,
            0,
            &mut params,
            blob_builder,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{q8_flash_default_tile_size, replay_stable_tile_count};

    #[test]
    fn q8_flash_gfx12_small_dense_shape_uses_tile16_only() {
        assert_eq!(q8_flash_default_tile_size("gfx1201", 8, 2, 256, 2_048), 16);
        assert_eq!(
            q8_flash_default_tile_size("gfx1201", 16, 2, 256, 2_048),
            128
        );
        assert_eq!(q8_flash_default_tile_size("gfx1201", 8, 2, 128, 2_048), 128);
    }

    #[test]
    fn gfx1151_tile32_default_is_exact_shape_and_arch_only() {
        assert_eq!(q8_flash_default_tile_size("gfx1151", 16, 2, 256, 2_048), 32);
        assert_eq!(
            q8_flash_default_tile_size("gfx1151", 16, 2, 256, 8_192),
            128
        );
        assert_eq!(q8_flash_default_tile_size("gfx1100", 1, 1, 64, 8_192), 32);
        assert_eq!(
            q8_flash_default_tile_size("gfx1200", 16, 2, 256, 2_048),
            128
        );
    }

    #[test]
    fn q8_flash_uses_max_tiles_for_both_capture_backends() {
        assert_eq!(replay_stable_tile_count(2, 64, false, false), 2);
        assert_eq!(replay_stable_tile_count(2, 64, true, false), 64);
        assert_eq!(replay_stable_tile_count(2, 64, false, true), 64);
    }
}
