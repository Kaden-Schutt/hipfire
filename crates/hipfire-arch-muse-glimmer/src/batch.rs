use crate::config::GlimmerConfig;
use crate::glimmer::GlimmerWeights;
use hipfire_runtime::llama::{EmbeddingFormat, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Fail-closed gate for batch decode. Only admits weight formats that have
/// batched GEMM kernels, identical rotation plans within each projection
/// family, and a Q8_0/F32/BF16 lm_head. `max_batch<=64`, `n_heads % n_kv_heads==0`,
/// `head_dim %32==0` are also enforced here so daemon attestation and forward
/// cannot diverge.
///
/// Batch embedding: HFQ4G256 / HFQ4G128 / Q8_0 / F32 (F32 via per-token gather loop).
/// Batch projections: Q8_0 / MQ4G256 / HFQ4G256 / MQ6G256 / F32 / BF16 (BF16 via GemmBf16Mfma on gfx942, F32 via GemmF32Batched).
/// lm_head: Q8_0 / F32 / BF16 (BF16 via GemmBf16Mfma, F32 via GemmF32Batched).
pub fn batch_weight_formats_supported(weights: &GlimmerWeights) -> Result<(), String> {
    // embedding must have a batched lookup kernel or F32 gather loop
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256
        | EmbeddingFormat::HFQ4G128
        | EmbeddingFormat::Q8_0
        | EmbeddingFormat::F32 => {}
        EmbeddingFormat::Q4K => return Err("glimmer batch: Q4K embed unsupported".to_string()),
    }
    // lm_head: Q8_0 batched chunked or F32/BF16 batched (GemmF32Batched / GemmBf16Mfma)
    if weights.lm_head.gpu_dtype != DType::Q8_0
        && weights.lm_head.gpu_dtype != DType::F32
        && weights.lm_head.gpu_dtype != DType::BF16
    {
        return Err(format!(
            "glimmer batch: lm_head must be Q8_0 or F32 or BF16 for batched decode (got {:?})",
            weights.lm_head.gpu_dtype
        ));
    }
    let allowed = |dt: DType| {
        matches!(
            dt,
            DType::Q8_0
                | DType::MQ4G256
                | DType::HFQ4G256
                | DType::MQ6G256
                | DType::F32
                | DType::BF16
        )
    };
    for (i, lw) in weights.layers.iter().enumerate() {
        for (label, w) in [
            ("q_proj", &lw.q_proj),
            ("k_proj", &lw.k_proj),
            ("v_proj", &lw.v_proj),
            ("attn_gate_proj", &lw.attn_gate_proj),
        ] {
            if !allowed(w.gpu_dtype) {
                return Err(format!(
                    "glimmer batch: layer {i} {label} dtype {:?} not in batch allowlist (Q8/MQ4/HFQ4/MQ6)",
                    w.gpu_dtype
                ));
            }
        }
        for (label, w) in [
            ("gate_proj", &lw.gate_proj),
            ("up_proj", &lw.up_proj),
            ("o_proj", &lw.o_proj),
            ("down_proj", &lw.down_proj),
        ] {
            if !allowed(w.gpu_dtype) {
                return Err(format!(
                    "glimmer batch: layer {i} {label} dtype {:?} not in batch allowlist",
                    w.gpu_dtype
                ));
            }
        }
        // rotation agreement within each projection family
        let plan = |w: &WeightTensor| hipfire_dispatch::types::dtype_rotation_plan(w.gpu_dtype);
        let q_plan = plan(&lw.q_proj);
        if plan(&lw.k_proj) != q_plan
            || plan(&lw.v_proj) != q_plan
            || plan(&lw.attn_gate_proj) != q_plan
        {
            return Err(format!(
                "glimmer batch: layer {i} q/k/v/attn_gate rotation plans must agree (q {:?}, k {:?}, v {:?}, gate {:?})",
                q_plan,
                plan(&lw.k_proj),
                plan(&lw.v_proj),
                plan(&lw.attn_gate_proj)
            ));
        }
        let gu_plan = plan(&lw.gate_proj);
        if plan(&lw.up_proj) != gu_plan {
            return Err(format!(
                "glimmer batch: layer {i} gate/up rotation plans must agree ({:?} vs {:?})",
                gu_plan,
                plan(&lw.up_proj)
            ));
        }
    }
    Ok(())
}

/// Independent continuous-batch decode state for Glimmer.
///
/// Owns the batched KV caches (lane-major absolute Q8, sliding + full),
/// the per-layer slot map, and persistent row-major tensors for scratch,
/// logits, and sampler state.
///
/// `max_batch <= 64`; VMM mapping uses `max(lane*lane_capacity + pos+1)`
/// across active physical lanes before each tick.
pub struct GlimmerDecodeBatchState {
    pub max_batch: usize,
    pub lane_capacity: usize,
    pub kv_sliding: KvCache,
    pub kv_full: KvCache,
    pub kv_slot_for_layer: Vec<usize>,

    pub dim: usize,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub vocab_size: usize,

    // batched scratch: [max_batch * dim] etc.
    pub x_batch: GpuTensor,
    pub residual_batch: GpuTensor,
    pub tmp_batch: GpuTensor,
    pub x_rot_batch: GpuTensor,
    pub q_batch: GpuTensor,
    pub k_batch: GpuTensor,
    pub v_batch: GpuTensor,
    pub attn_gate_batch: GpuTensor,
    pub attn_out_batch: GpuTensor,
    pub o_out_batch: GpuTensor,
    pub o_rot_batch: GpuTensor,
    pub gate_batch: GpuTensor,
    pub up_batch: GpuTensor,
    pub ffn_hidden_batch: GpuTensor,
    pub ffn_out_batch: GpuTensor,
    pub down_rot_batch: GpuTensor,
    pub final_hidden: GpuTensor,
    pub logits: GpuTensor,            // [max_batch * vocab]
    pub logits_scratch: GpuTensor,    // [vocab] scratch for per-row lm_head fallback
    pub sample_out: GpuTensor,        // [max_batch * 2] (token,new_rng) per lane
    pub sample_rng_states: GpuTensor, // [max_batch] u32-as-f32
    pub positions: GpuTensor,         // [max_batch] i32
    pub tokens: GpuTensor,            // [max_batch] i32
    // BF16 activation staging for calibration GEMMs (HIPFIRE_CALIB_BF16=1).
    // Persistent scratch sized once to max_batch*hidden_dim (max K), BF16.
    // Never per-call alloc; staged via gpu.convert_f32_to_bf16 then GemmBf16Mfma.
    pub bf16_scratch: GpuTensor, // [max_batch * hidden_dim] BF16

    // small neutral buffers for sampling
    pub repeat_dummy: GpuTensor,    // [1] zero
    pub qk_norm_ones: GpuTensor,    // [head_dim]
    pub embed_norm_ones: GpuTensor, // [dim]
}

impl GlimmerDecodeBatchState {
    /// Create a new batch state. Transactional — any partial allocation is
    /// freed before returning `Err`. Rejects `max_batch==0|>64`,
    /// `lane_capacity==0`, `n_heads%n_kv_heads!=0`,
    /// `head_dim%32!=0`, bad geometry, or LDS overflow.
    pub fn new(
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        max_batch: usize,
        lane_capacity: usize,
    ) -> Result<Self, String> {
        if max_batch == 0 || max_batch > 64 {
            return Err(format!(
                "glimmer batch: max_batch must be 1..=64 (got {max_batch})"
            ));
        }
        if lane_capacity == 0 {
            return Err("glimmer batch: lane_capacity must be non-zero".to_string());
        }
        if cfg.n_heads == 0
            || cfg.n_kv_heads == 0
            || cfg.head_dim == 0
            || cfg.dim == 0
            || cfg.hidden_dim == 0
            || cfg.vocab_size == 0
            || cfg.n_layers == 0
        {
            return Err("glimmer batch: zero or degenerate geometry".to_string());
        }
        if cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(format!(
                "glimmer batch: n_heads {} % n_kv_heads {} !=0",
                cfg.n_heads, cfg.n_kv_heads
            ));
        }
        if cfg.head_dim % 32 != 0 {
            return Err(format!(
                "glimmer batch: head_dim {} must be multiple of 32",
                cfg.head_dim
            ));
        }
        // LDS check before any allocation
        gpu.ensure_attention_q8_0_kv_independent_lds(lane_capacity, cfg.head_dim)
            .map_err(|e| format!("glimmer batch: LDS check: {e:?}"))?;
        gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();

        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let vocab = cfg.vocab_size;
        let total_capacity = max_batch
            .checked_mul(lane_capacity)
            .ok_or_else(|| "glimmer batch: KV capacity overflow".to_string())?;

        // overflow validation for all scratch products
        let _ = max_batch
            .checked_mul(dim)
            .ok_or_else(|| "glimmer batch: x size overflow".to_string())?;
        let _ = max_batch
            .checked_mul(q_dim)
            .ok_or_else(|| "glimmer batch: q size overflow".to_string())?;
        let _ = max_batch
            .checked_mul(kv_dim)
            .ok_or_else(|| "glimmer batch: kv size overflow".to_string())?;
        let _ = max_batch
            .checked_mul(hidden_dim)
            .ok_or_else(|| "glimmer batch: hidden size overflow".to_string())?;
        let _ = max_batch
            .checked_mul(vocab)
            .ok_or_else(|| "glimmer batch: vocab size overflow".to_string())?;

        // kv slot map mirrors GlimmerState
        let mut kv_slot_for_layer = Vec::with_capacity(cfg.n_layers);
        let mut s = 0usize;
        let mut f = 0usize;
        for &lt in cfg.layer_types.iter() {
            match lt {
                crate::config::GlimmerLayerType::Sliding => {
                    kv_slot_for_layer.push(s);
                    s += 1;
                }
                crate::config::GlimmerLayerType::Full => {
                    kv_slot_for_layer.push(f);
                    f += 1;
                }
            }
        }

        // KV caches: lane-major contiguous Q8, max_batch * lane_capacity per layer
        let kv_sliding = match KvCache::new_gpu_q8(
            gpu,
            cfg.n_sliding_layers(),
            cfg.n_kv_heads,
            cfg.head_dim,
            total_capacity,
        ) {
            Ok(v) => v,
            Err(e) => return Err(format!("glimmer batch: sliding kv: {e:?}")),
        };
        let kv_full = match KvCache::new_gpu_q8(
            gpu,
            cfg.n_full_layers(),
            cfg.n_kv_heads,
            cfg.head_dim,
            total_capacity,
        ) {
            Ok(v) => v,
            Err(e) => {
                let _ = kv_sliding.free_gpu(gpu);
                return Err(format!("glimmer batch: full kv: {e:?}"));
            }
        };

        // Transactional scratch: allocate owned tensors into ledger; on failure drain and free.
        let mut ledger: Vec<GpuTensor> = Vec::with_capacity(32);
        macro_rules! try_alloc {
            ($shape:expr, $label:expr) => {
                match gpu.zeros($shape, DType::F32) {
                    Ok(t) => ledger.push(t),
                    Err(e) => {
                        for prev in ledger.drain(..) {
                            let _ = gpu.release_tensor_immediate(prev);
                        }
                        let _ = kv_full.free_gpu(gpu);
                        let _ = kv_sliding.free_gpu(gpu);
                        return Err(format!("glimmer batch: alloc {}: {e:?}", $label));
                    }
                }
            };
        }
        macro_rules! try_alloc_dtype {
            ($shape:expr, $dtype:expr, $label:expr) => {
                match gpu.zeros($shape, $dtype) {
                    Ok(t) => ledger.push(t),
                    Err(e) => {
                        for prev in ledger.drain(..) {
                            let _ = gpu.release_tensor_immediate(prev);
                        }
                        let _ = kv_full.free_gpu(gpu);
                        let _ = kv_sliding.free_gpu(gpu);
                        return Err(format!("glimmer batch: alloc {}: {e:?}", $label));
                    }
                }
            };
        }
        try_alloc!(&[max_batch * dim], "x_batch");
        try_alloc!(&[max_batch * dim], "residual_batch");
        try_alloc!(&[max_batch * dim], "tmp_batch");
        try_alloc!(&[max_batch * dim], "x_rot_batch");
        try_alloc!(&[max_batch * q_dim], "q_batch");
        try_alloc!(&[max_batch * kv_dim], "k_batch");
        try_alloc!(&[max_batch * kv_dim], "v_batch");
        try_alloc!(&[max_batch * q_dim], "attn_gate_batch");
        try_alloc!(&[max_batch * q_dim], "attn_out_batch");
        try_alloc!(&[max_batch * dim], "o_out_batch");
        try_alloc!(&[max_batch * dim], "o_rot_batch");
        try_alloc!(&[max_batch * hidden_dim], "gate_batch");
        try_alloc!(&[max_batch * hidden_dim], "up_batch");
        try_alloc!(&[max_batch * hidden_dim], "ffn_hidden_batch");
        try_alloc!(&[max_batch * dim], "ffn_out_batch");
        try_alloc!(&[max_batch * hidden_dim], "down_rot_batch");
        try_alloc!(&[max_batch * dim], "final_hidden");
        try_alloc!(&[max_batch * vocab], "logits");
        try_alloc!(&[vocab], "logits_scratch");
        try_alloc!(&[max_batch * 2], "sample_out");
        try_alloc!(&[max_batch], "sample_rng_states");
        try_alloc!(&[max_batch], "positions");
        try_alloc!(&[max_batch], "tokens");
        try_alloc_dtype!(&[max_batch * hidden_dim], DType::BF16, "bf16_scratch");
        try_alloc!(&[1], "repeat_dummy");
        try_alloc!(&[cfg.head_dim], "qk_norm_ones");
        try_alloc!(&[dim], "embed_norm_ones");
        // drain in reverse allocation order into fields (ledger[0]=x_batch, etc.)
        let mut it = ledger.into_iter();
        let x_batch = it.next().unwrap();
        let residual_batch = it.next().unwrap();
        let tmp_batch = it.next().unwrap();
        let x_rot_batch = it.next().unwrap();
        let q_batch = it.next().unwrap();
        let k_batch = it.next().unwrap();
        let v_batch = it.next().unwrap();
        let attn_gate_batch = it.next().unwrap();
        let attn_out_batch = it.next().unwrap();
        let o_out_batch = it.next().unwrap();
        let o_rot_batch = it.next().unwrap();
        let gate_batch = it.next().unwrap();
        let up_batch = it.next().unwrap();
        let ffn_hidden_batch = it.next().unwrap();
        let ffn_out_batch = it.next().unwrap();
        let down_rot_batch = it.next().unwrap();
        let final_hidden = it.next().unwrap();
        let logits = it.next().unwrap();
        let logits_scratch = it.next().unwrap();
        let sample_out = it.next().unwrap();
        let sample_rng_states = it.next().unwrap();
        let positions = it.next().unwrap();
        let tokens = it.next().unwrap();
        let bf16_scratch = it.next().unwrap();
        let repeat_dummy = it.next().unwrap();
        let qk_norm_ones = it.next().unwrap();
        let embed_norm_ones = it.next().unwrap();
        // upload ones
        {
            let ones_hd: Vec<f32> = vec![1.0; cfg.head_dim];
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(ones_hd.as_ptr() as *const u8, ones_hd.len() * 4)
            };
            gpu.hip
                .memcpy_htod(&qk_norm_ones.buf, bytes)
                .map_err(|e| format!("glimmer batch: init qk_norm_ones: {e:?}"))?;
        }
        {
            let ones_dim: Vec<f32> = vec![1.0; dim];
            let bytes = unsafe {
                std::slice::from_raw_parts(ones_dim.as_ptr() as *const u8, ones_dim.len() * 4)
            };
            gpu.hip
                .memcpy_htod(&embed_norm_ones.buf, bytes)
                .map_err(|e| format!("glimmer batch: init embed_norm_ones: {e:?}"))?;
        }

        Ok(Self {
            max_batch,
            lane_capacity,
            kv_sliding,
            kv_full,
            kv_slot_for_layer,
            dim,
            hidden_dim,
            q_dim,
            kv_dim,
            vocab_size: vocab,
            x_batch,
            residual_batch,
            tmp_batch,
            x_rot_batch,
            q_batch,
            k_batch,
            v_batch,
            attn_gate_batch,
            attn_out_batch,
            o_out_batch,
            o_rot_batch,
            gate_batch,
            up_batch,
            ffn_hidden_batch,
            ffn_out_batch,
            down_rot_batch,
            final_hidden,
            logits,
            logits_scratch,
            sample_out,
            sample_rng_states,
            positions,
            tokens,
            bf16_scratch,
            repeat_dummy,
            qk_norm_ones,
            embed_norm_ones,
        })
    }

    /// Clear all lanes and scratch.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.kv_sliding
            .clear_gpu(gpu)
            .map_err(|e| format!("glimmer batch reset kv_sliding: {e:?}"))?;
        self.kv_full
            .clear_gpu(gpu)
            .map_err(|e| format!("glimmer batch reset kv_full: {e:?}"))?;
        for t in [
            &self.x_batch,
            &self.residual_batch,
            &self.tmp_batch,
            &self.x_rot_batch,
            &self.q_batch,
            &self.k_batch,
            &self.v_batch,
            &self.attn_gate_batch,
            &self.attn_out_batch,
            &self.o_out_batch,
            &self.o_rot_batch,
            &self.gate_batch,
            &self.up_batch,
            &self.ffn_hidden_batch,
            &self.ffn_out_batch,
            &self.down_rot_batch,
            &self.final_hidden,
            &self.logits,
            &self.logits_scratch,
            &self.sample_out,
            &self.sample_rng_states,
            &self.positions,
            &self.tokens,
            &self.bf16_scratch,
            &self.repeat_dummy,
        ] {
            gpu.hip
                .memset(&t.buf, 0, t.buf.size())
                .map_err(|e| format!("glimmer batch reset memset: {e:?}"))?;
        }
        gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
        Ok(())
    }

    /// Clear only one lane's KV slice and its output/scratch rows.
    /// Stale suffix bytes beyond the new generation's max position are left
    /// unreachable because attention reads are position-bounded.
    pub fn reset_lane(&mut self, gpu: &mut Gpu, lane: usize) -> Result<(), String> {
        if lane >= self.max_batch {
            return Err(format!(
                "glimmer batch reset_lane: lane {lane} >= max_batch {}",
                self.max_batch
            ));
        }
        let bytes_per_token = self.kv_sliding.n_kv_heads * (self.kv_sliding.head_dim / 32) * 34;
        let lane_bytes = self.lane_capacity * bytes_per_token;
        let byte_off = lane * lane_bytes;
        for t in self
            .kv_sliding
            .k_gpu
            .iter()
            .chain(self.kv_sliding.v_gpu.iter())
            .chain(self.kv_full.k_gpu.iter())
            .chain(self.kv_full.v_gpu.iter())
        {
            if t.numel() <= 1 {
                continue;
            }
            let parent_bytes = t.buf.size();
            let end = byte_off + lane_bytes;
            if end > parent_bytes {
                return Err("glimmer batch reset_lane: lane byte range exceeds parent".to_string());
            }
            let ptr = unsafe { (t.buf.as_ptr() as *mut u8).add(byte_off) as *mut std::ffi::c_void };
            let tmp_buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, lane_bytes) };
            gpu.hip
                .memset(&tmp_buf, 0, lane_bytes)
                .map_err(|e| format!("glimmer batch reset_lane kv memset: {e:?}"))?;
            std::mem::forget(tmp_buf);
        }
        let zero_row = |gpu: &mut Gpu, t: &GpuTensor, dim: usize| -> Result<(), String> {
            let view = t.sub_offset(lane * dim, dim);
            gpu.hip
                .memset(&view.buf, 0, view.buf.size())
                .map_err(|e| format!("glimmer batch reset_lane memset: {e:?}"))
        };
        zero_row(gpu, &self.x_batch, self.dim)?;
        zero_row(gpu, &self.residual_batch, self.dim)?;
        zero_row(gpu, &self.tmp_batch, self.dim)?;
        zero_row(gpu, &self.x_rot_batch, self.dim)?;
        zero_row(gpu, &self.q_batch, self.q_dim)?;
        zero_row(gpu, &self.k_batch, self.kv_dim)?;
        zero_row(gpu, &self.v_batch, self.kv_dim)?;
        zero_row(gpu, &self.attn_gate_batch, self.q_dim)?;
        zero_row(gpu, &self.attn_out_batch, self.q_dim)?;
        zero_row(gpu, &self.o_out_batch, self.dim)?;
        zero_row(gpu, &self.o_rot_batch, self.dim)?;
        zero_row(gpu, &self.gate_batch, self.hidden_dim)?;
        zero_row(gpu, &self.up_batch, self.hidden_dim)?;
        zero_row(gpu, &self.ffn_hidden_batch, self.hidden_dim)?;
        zero_row(gpu, &self.ffn_out_batch, self.dim)?;
        zero_row(gpu, &self.down_rot_batch, self.hidden_dim)?;
        zero_row(gpu, &self.final_hidden, self.dim)?;
        zero_row(gpu, &self.logits, self.vocab_size)?;
        // bf16_scratch holds BF16 activations — zero the lane's slice (bytes still zero)
        zero_row(gpu, &self.bf16_scratch, self.hidden_dim)?;
        {
            let v = self.sample_out.sub_offset(lane * 2, 2);
            gpu.hip
                .memset(&v.buf, 0, v.buf.size())
                .map_err(|e| format!("glimmer batch reset_lane sample_out: {e:?}"))?;
        }
        {
            let v = self.sample_rng_states.sub_offset(lane, 1);
            gpu.hip
                .memset(&v.buf, 0, v.buf.size())
                .map_err(|e| format!("glimmer batch reset_lane rng: {e:?}"))?;
        }
        {
            let v = self.positions.sub_offset(lane, 1);
            gpu.hip
                .memset(&v.buf, 0, v.buf.size())
                .map_err(|e| format!("glimmer batch reset_lane pos: {e:?}"))?;
        }
        {
            let v = self.tokens.sub_offset(lane, 1);
            gpu.hip
                .memset(&v.buf, 0, v.buf.size())
                .map_err(|e| format!("glimmer batch reset_lane tokens: {e:?}"))?;
        }
        gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
        Ok(())
    }

    /// Consumes self and returns all GPU resources.
    /// Consumes self and returns all GPU resources.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let ptrs: Vec<*mut std::ffi::c_void> = [
            &self.x_batch,
            &self.residual_batch,
            &self.tmp_batch,
            &self.x_rot_batch,
            &self.q_batch,
            &self.k_batch,
            &self.v_batch,
            &self.attn_gate_batch,
            &self.attn_out_batch,
            &self.o_out_batch,
            &self.o_rot_batch,
            &self.gate_batch,
            &self.up_batch,
            &self.ffn_hidden_batch,
            &self.ffn_out_batch,
            &self.down_rot_batch,
            &self.final_hidden,
            &self.logits,
            &self.logits_scratch,
            &self.sample_out,
            &self.sample_rng_states,
            &self.positions,
            &self.tokens,
            &self.bf16_scratch,
            &self.repeat_dummy,
            &self.qk_norm_ones,
            &self.embed_norm_ones,
        ]
        .iter()
        .map(|t| t.buf.as_ptr() as *mut std::ffi::c_void)
        .collect();
        let _ = self.kv_sliding.free_gpu(gpu);
        let _ = self.kv_full.free_gpu(gpu);
        for t in [
            self.x_batch,
            self.residual_batch,
            self.tmp_batch,
            self.x_rot_batch,
            self.q_batch,
            self.k_batch,
            self.v_batch,
            self.attn_gate_batch,
            self.attn_out_batch,
            self.o_out_batch,
            self.o_rot_batch,
            self.gate_batch,
            self.up_batch,
            self.ffn_hidden_batch,
            self.ffn_out_batch,
            self.down_rot_batch,
            self.final_hidden,
            self.logits,
            self.logits_scratch,
            self.sample_out,
            self.sample_rng_states,
            self.positions,
            self.tokens,
            self.bf16_scratch,
            self.repeat_dummy,
            self.qk_norm_ones,
            self.embed_norm_ones,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for ptr in ptrs {
            gpu.scratch.invalidate_x_caches_for(ptr);
        }
        gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
    }

    /// Admission-only sequential per-lane prefill with cancellation.
    /// For each prompt token, runs a single-row forward through all layers
    /// writing the lane's KV at absolute position `pos`. Skips lm_head except
    /// on the last token, so only the final logits row is produced.
    /// Checks `should_cancel()` between tokens; if cancelled returns `Ok(false)`
    /// (lane left for daemon reset). Final cancellation is also checked before
    /// returning.
    pub fn prefill_lane_cancellable(
        &mut self,
        gpu: &mut Gpu,
        weights: &GlimmerWeights,
        cfg: &GlimmerConfig,
        lane: usize,
        prompt: &[u32],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<bool, String> {
        if lane >= self.max_batch {
            return Err(format!(
                "glimmer prefill_lane: lane {lane} >= max_batch {}",
                self.max_batch
            ));
        }
        if prompt.is_empty() {
            return Err("glimmer prefill_lane: prompt is empty".to_string());
        }
        if prompt.len() >= self.lane_capacity {
            return Err(format!(
                "glimmer prefill_lane: prompt len {} leaves no decode capacity (lane_capacity {})",
                prompt.len(),
                self.lane_capacity
            ));
        }
        batch_weight_formats_supported(weights)?;
        // clear lane first (so KV and scratch start clean)
        self.reset_lane(gpu, lane)?;

        // lane-isolated scratch views (row slices)
        let dim = cfg.dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hidden_dim = cfg.hidden_dim;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let hd = cfg.head_dim;
        let rms_eps = cfg.rms_norm_eps;
        let post_eps = cfg.post_norm_eps;

        let x_lane = self.x_batch.sub_offset(lane * dim, dim);
        let residual_lane = self.residual_batch.sub_offset(lane * dim, dim);
        let tmp_lane = self.tmp_batch.sub_offset(lane * dim, dim);
        let x_rot_lane = self.x_rot_batch.sub_offset(lane * dim, dim);
        let q_lane = self.q_batch.sub_offset(lane * q_dim, q_dim);
        let k_lane = self.k_batch.sub_offset(lane * kv_dim, kv_dim);
        let v_lane = self.v_batch.sub_offset(lane * kv_dim, kv_dim);
        let attn_gate_lane = self.attn_gate_batch.sub_offset(lane * q_dim, q_dim);
        let attn_out_lane = self.attn_out_batch.sub_offset(lane * q_dim, q_dim);
        let gate_lane = self.gate_batch.sub_offset(lane * hidden_dim, hidden_dim);
        let up_lane = self.up_batch.sub_offset(lane * hidden_dim, hidden_dim);
        let ffn_hidden_lane = self
            .ffn_hidden_batch
            .sub_offset(lane * hidden_dim, hidden_dim);
        let ffn_out_lane = self.ffn_out_batch.sub_offset(lane * dim, dim);
        let final_lane = self.final_hidden.sub_offset(lane * dim, dim);
        let logits_lane = self
            .logits
            .sub_offset(lane * cfg.vocab_size, cfg.vocab_size);

        // use lane's positions slot as the scalar pos buf for per-token kernels
        let pos_slot = self.positions.sub_offset(lane, 1);

        for (pos_idx, &tok) in prompt.iter().enumerate() {
            if should_cancel() {
                return Ok(false);
            }
            gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
            gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
            // stage position
            let pos_i32 = pos_idx as i32;
            gpu.hip
                .memcpy_htod(&pos_slot.buf, &pos_i32.to_ne_bytes())
                .map_err(|e| format!("glimmer prefill htod pos: {e:?}"))?;
            // ensure VMM mapped for this lane/pos (max active = lane*lane_capacity + pos+1)
            // For sequential prefill only this lane is active, so max = lane*cap + pos+1.
            let required = lane * self.lane_capacity + pos_idx + 1;
            self.kv_sliding
                .ensure_mapped_capacity(gpu, required)
                .map_err(|e| format!("glimmer prefill kv_sliding map: {e:?}"))?;
            self.kv_full
                .ensure_mapped_capacity(gpu, required)
                .map_err(|e| format!("glimmer prefill kv_full map: {e:?}"))?;

            // embedding lookup (single token) into x_lane
            match weights.embd_format {
                EmbeddingFormat::Q8_0 => gpu
                    .embedding_lookup_q8(&weights.embed_tokens, &x_lane, tok, dim)
                    .map_err(|e| format!("glimmer prefill embed q8: {e:?}"))?,
                EmbeddingFormat::HFQ4G256 => gpu
                    .embedding_lookup_hfq4g256(&weights.embed_tokens, &x_lane, tok, dim)
                    .map_err(|e| format!("glimmer prefill embed hfq4g256: {e:?}"))?,
                EmbeddingFormat::HFQ4G128 => gpu
                    .embedding_lookup_hfq4g128(&weights.embed_tokens, &x_lane, tok, dim)
                    .map_err(|e| format!("glimmer prefill embed hfq4g128: {e:?}"))?,
                EmbeddingFormat::F32 => gpu
                    .embedding_lookup(&weights.embed_tokens, &x_lane, tok, dim)
                    .map_err(|e| format!("glimmer prefill embed f32: {e:?}"))?,
                EmbeddingFormat::Q4K => {
                    return Err("glimmer prefill: unsupported embed format Q4K".to_string())
                }
            }
            // scale-less embed_norm (treat ABI env as off for prefill parity)
            gpu.rmsnorm_f32(&x_lane, &self.embed_norm_ones, &x_lane, rms_eps)
                .map_err(|e| format!("glimmer prefill embed_norm: {e:?}"))?;

            for (layer_idx, lw) in weights.layers.iter().enumerate() {
                let slot = self.kv_slot_for_layer[layer_idx];
                // residual = x
                gpu.hip
                    .memcpy_dtod_at(&residual_lane.buf, 0, &x_lane.buf, 0, dim * 4)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} save residual: {e:?}"))?;
                // input norm -> tmp
                gpu.rmsnorm_f32(&x_lane, &lw.input_layernorm, &tmp_lane, rms_eps)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} input rmsnorm: {e:?}"))?;
                // q/k/v/gate (plain GEMM per lane)
                for (label, w, out) in [
                    ("q_proj", &lw.q_proj, &q_lane),
                    ("k_proj", &lw.k_proj, &k_lane),
                    ("v_proj", &lw.v_proj, &v_lane),
                    ("attn_gate", &lw.attn_gate_proj, &attn_gate_lane),
                ] {
                    // use shared rotation helper style: rotate if needed inside weight_gemv
                    // For simplicity use direct weight_gemv on tmp (which handles MQ rotation
                    // via internal rotate). The batch helpers do explicit rotate before GEMM,
                    // but per-row weight_gemv already includes rotation for MQ.
                    hipfire_runtime::llama::weight_gemv(gpu, w, &tmp_lane, out)
                        .map_err(|e| format!("glimmer prefill L{layer_idx} {label}: {e}"))?;
                }
                // scale-less QK norm + q scale
                gpu.rmsnorm_batched(&q_lane, &self.qk_norm_ones, &q_lane, n_heads, hd, rms_eps)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} q_norm: {e:?}"))?;
                gpu.rmsnorm_batched(&k_lane, &self.qk_norm_ones, &k_lane, n_kv, hd, rms_eps)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} k_norm: {e:?}"))?;
                gpu.scale_f32(&q_lane, cfg.qk_scale_factor)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} q scale: {e:?}"))?;
                if cfg.has_rope(layer_idx) {
                    let theta = cfg.rope_theta_for(layer_idx);
                    gpu.rope_f32(&q_lane, &k_lane, &pos_slot.buf, n_heads, n_kv, hd, theta)
                        .map_err(|e| format!("glimmer prefill L{layer_idx} rope: {e:?}"))?;
                }
                // KV write (single position, lane-major absolute via ensure_mapped + lane offset)
                // Use q8_lane_view for this lane's slice to get correct byte offset.
                // For per-token sequential we can write via lane view's buf with pos=slot offset 0..cap.
                // Instead construct a lane view for one layer and write with position = pos_idx (relative within lane)
                // But KV write for independent lane uses absolute position within lane (0..cap).
                // We can use kv_cache_write_q8_0 on the lane view with pos = pos_idx.
                let (k_cache, v_cache) = match cfg.layer_types[layer_idx] {
                    crate::config::GlimmerLayerType::Sliding => {
                        let kv = self
                            .kv_sliding
                            .q8_lane_view(lane, self.lane_capacity)
                            .map_err(|e| format!("glimmer prefill lane view sliding: {e:?}"))?;
                        (
                            kv.k_gpu[slot].shallow_clone(),
                            kv.v_gpu[slot].shallow_clone(),
                        )
                    }
                    crate::config::GlimmerLayerType::Full => {
                        let kv = self
                            .kv_full
                            .q8_lane_view(lane, self.lane_capacity)
                            .map_err(|e| format!("glimmer prefill lane view full: {e:?}"))?;
                        (
                            kv.k_gpu[slot].shallow_clone(),
                            kv.v_gpu[slot].shallow_clone(),
                        )
                    }
                };
                // write K and V at lane-local position pos_idx
                // need to set pos_slot to pos_idx (already), then call kv_cache_write
                gpu.kv_cache_write_q8_0(&k_cache, &k_lane, &pos_slot.buf, n_kv, hd)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} kv k: {e:?}"))?;
                gpu.kv_cache_write_q8_0(&v_cache, &v_lane, &pos_slot.buf, n_kv, hd)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} kv v: {e:?}"))?;

                // attention (single token, use swa path)
                let window = cfg.window_for(layer_idx);
                let seq_len = pos_idx + 1;
                let (k_g, v_g) = match cfg.layer_types[layer_idx] {
                    crate::config::GlimmerLayerType::Sliding => {
                        let kv = self
                            .kv_sliding
                            .q8_lane_view(lane, self.lane_capacity)
                            .map_err(|e| format!("glimmer prefill lane view attn: {e:?}"))?;
                        (
                            kv.k_gpu[slot].shallow_clone(),
                            kv.v_gpu[slot].shallow_clone(),
                        )
                    }
                    crate::config::GlimmerLayerType::Full => {
                        let kv = self
                            .kv_full
                            .q8_lane_view(lane, self.lane_capacity)
                            .map_err(|e| format!("glimmer prefill lane view attn2: {e:?}"))?;
                        (
                            kv.k_gpu[slot].shallow_clone(),
                            kv.v_gpu[slot].shallow_clone(),
                        )
                    }
                };
                gpu.attention_q8_0_kv_swa(
                    &q_lane,
                    &k_g,
                    &v_g,
                    &attn_out_lane,
                    &pos_slot.buf,
                    seq_len,
                    n_heads,
                    n_kv,
                    hd,
                    self.lane_capacity,
                    window,
                )
                .map_err(|e| format!("glimmer prefill L{layer_idx} attn: {e:?}"))?;

                gpu.sigmoid_mul_f32(&attn_out_lane, &attn_gate_lane)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} gate: {e:?}"))?;
                hipfire_runtime::llama::weight_gemv(gpu, &lw.o_proj, &attn_out_lane, &x_rot_lane)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} o_proj: {e}"))?;
                gpu.rmsnorm_f32(
                    &x_rot_lane,
                    &lw.post_attention_layernorm,
                    &x_rot_lane,
                    post_eps,
                )
                .map_err(|e| format!("glimmer prefill L{layer_idx} post_attn: {e:?}"))?;
                gpu.hip
                    .memcpy_dtod_at(&x_lane.buf, 0, &residual_lane.buf, 0, dim * 4)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} reset x: {e:?}"))?;
                gpu.add_inplace_f32(&x_lane, &x_rot_lane)
                    .map_err(|e| format!("glimmer prefill L{layer_idx} add: {e:?}"))?;

                gpu.hip
                    .memcpy_dtod_at(&residual_lane.buf, 0, &x_lane.buf, 0, dim * 4)
                    .map_err(|e| format!("glimmer prefill ffn residual: {e:?}"))?;
                gpu.rmsnorm_f32(&x_lane, &lw.pre_feedforward_layernorm, &tmp_lane, rms_eps)
                    .map_err(|e| format!("glimmer prefill pre_ffn: {e:?}"))?;
                gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
                gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
                hipfire_runtime::llama::weight_gemv(gpu, &lw.gate_proj, &tmp_lane, &gate_lane)
                    .map_err(|e| format!("glimmer prefill gate: {e}"))?;
                hipfire_runtime::llama::weight_gemv(gpu, &lw.up_proj, &tmp_lane, &up_lane)
                    .map_err(|e| format!("glimmer prefill up: {e}"))?;
                gpu.silu_mul_f32(&gate_lane, &up_lane, &ffn_hidden_lane)
                    .map_err(|e| format!("glimmer prefill silu: {e:?}"))?;
                hipfire_runtime::llama::weight_gemv(
                    gpu,
                    &lw.down_proj,
                    &ffn_hidden_lane,
                    &ffn_out_lane,
                )
                .map_err(|e| format!("glimmer prefill down: {e}"))?;
                gpu.rmsnorm_f32(
                    &ffn_out_lane,
                    &lw.post_feedforward_layernorm,
                    &ffn_out_lane,
                    post_eps,
                )
                .map_err(|e| format!("glimmer prefill post_ffn: {e:?}"))?;
                gpu.hip
                    .memcpy_dtod_at(&x_lane.buf, 0, &residual_lane.buf, 0, dim * 4)
                    .map_err(|e| format!("glimmer prefill reset2: {e:?}"))?;
                gpu.add_inplace_f32(&x_lane, &ffn_out_lane)
                    .map_err(|e| format!("glimmer prefill ffn add: {e:?}"))?;
            }

            // cancellation checked only between tokens, not mid-layer (same as LFM precedent)
            if pos_idx + 1 == prompt.len() {
                // final token: final norm + lm_head (only row)
                gpu.rmsnorm_f32(&x_lane, &weights.final_norm, &final_lane, rms_eps)
                    .map_err(|e| format!("glimmer prefill final norm: {e:?}"))?;
                hipfire_runtime::llama::weight_gemv(
                    gpu,
                    &weights.lm_head,
                    &final_lane,
                    &logits_lane,
                )
                .map_err(|e| format!("glimmer prefill lm_head: {e}"))?;
                if cfg.output_multiplier != 1.0 {
                    gpu.scale_f32(&logits_lane, cfg.output_multiplier)
                        .map_err(|e| format!("glimmer prefill scale: {e:?}"))?;
                }
                if cfg.final_logit_softcapping > 0.0 {
                    gpu.logit_softcap_f32(
                        &logits_lane,
                        cfg.vocab_size,
                        cfg.final_logit_softcapping,
                    )
                    .map_err(|e| format!("glimmer prefill softcap: {e:?}"))?;
                }
                // stage host mirror positions/tokens for this lane (lane_capacity stride)
                let pos_i32 = pos_idx as i32;
                gpu.hip
                    .memcpy_htod(
                        &self.positions.sub_offset(lane, 1).buf,
                        &pos_i32.to_ne_bytes(),
                    )
                    .map_err(|e| format!("glimmer prefill htod pos mirror: {e:?}"))?;
                // tokens mirror is updated by caller (host), GPU tokens tensor is for next decode tick.
            }
        }
        if should_cancel() {
            return Ok(false);
        }
        // update device tokens mirror for this lane's last position token (prompt tail)
        // Caller maintains host mirror; device tokens used only for forward batch embedding.
        Ok(true)
    }

    /// Sample one lane's resident logits with neutral penalties.
    /// Uses `sample_top_p_pf` with `repeat_penalty=1, presence=0, frequency=0, min_p=None`.
    /// `rng_state` is the per-lane host RNG; returns `(token, new_rng)`.
    pub fn sample_lane_product(
        &mut self,
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        lane: usize,
        temp: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng_state: u32,
    ) -> Result<(u32, u32), String> {
        if lane >= self.max_batch {
            return Err(format!("sample_lane_product lane {lane} out of range"));
        }
        let vocab = cfg.vocab_size;
        let logits_lane = self.logits.sub_offset(lane * vocab, vocab);
        let out_lane = self.sample_out.sub_offset(lane * 2, 2);
        if temp <= 1e-6 {
            let tok = gpu
                .argmax_f32(&logits_lane, vocab)
                .map_err(|e| format!("sample_lane argmax: {e:?}"))?;
            return Ok((tok, rng_state));
        }
        let top_p_eff = if top_p <= 0.0 || top_p > 1.0 {
            1.0
        } else {
            top_p
        };
        gpu.sample_top_p_pf(
            &logits_lane,
            &out_lane,
            &self.repeat_dummy,
            vocab,
            temp,
            top_p_eff,
            rng_state,
            0,
            1.0,
            0.0,
            0.0,
            top_k,
            None,
        )
        .map_err(|e| format!("sample_lane_product: {e:?}"))
    }

    /// Sample all active lanes with a single shared temp/top_p/top_k cohort.
    /// `rng_states` is a host slice of `max_batch` u32s (only active entries are read).
    /// Inactive rows' outputs are ignored and their RNG is not advanced.
    /// Returns `Vec<Option<(token,new_rng)>>` indexed by physical lane (None for inactive).
    pub fn sample_product(
        &mut self,
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        temp: f32,
        top_p: f32,
        top_k: Option<u32>,
        rng_states: &mut [u32],
        active_mask: u64,
    ) -> Result<Vec<Option<(u32, u32)>>, String> {
        if rng_states.len() < self.max_batch {
            return Err(format!(
                "sample_product rng_states len {} < max_batch {}",
                rng_states.len(),
                self.max_batch
            ));
        }
        let batch = self.max_batch;
        // fast path for greedy
        if temp <= 1e-6 {
            // batched argmax could batch active lanes, but per-lane loop preserves exact
            // byte identity and respects inactive holes (no extra launches for inactive).
            let mut out: Vec<Option<(u32, u32)>> = vec![None; batch];
            for lane in 0..batch {
                if (active_mask >> lane) & 1 == 0 {
                    continue;
                }
                let vocab = cfg.vocab_size;
                let logits_lane = self.logits.sub_offset(lane * vocab, vocab);
                let tok = gpu
                    .argmax_f32(&logits_lane, vocab)
                    .map_err(|e| format!("sample_product greedy lane {lane}: {e:?}"))?;
                out[lane] = Some((tok, rng_states[lane]));
            }
            return Ok(out);
        }
        // If all active lanes can be batched with sample_rows_pf_f32 and share
        // the same temp/top_p/top_k, use the batched sampler for one launch.
        // Otherwise fall back to per-lane loop (still correct, just more launches).
        // For v1, cohort batches are share-temp by BatchSamplingKey, so batched path
        // is the expected case. We implement the batched launch when top_k uniform
        // and all actives share the same hyperparams (single values).
        let active_cnt = active_mask.count_ones() as usize;
        if active_cnt == 0 {
            return Ok(vec![None; batch]);
        }
        // Check if we can use row kernel: need to stage logits contiguous for active only?
        // The row kernel operates on dense [batch * vocab] contiguous; holes would produce
        // garbage samples. Instead, per-lane fallback guarantees inactive rows never
        // influence sampling and never advance RNG.
        let mut out: Vec<Option<(u32, u32)>> = vec![None; batch];
        // Per-lane loop preserves inactive invariants exactly.
        for lane in 0..batch {
            if (active_mask >> lane) & 1 == 0 {
                continue;
            }
            let res =
                self.sample_lane_product(gpu, cfg, lane, temp, top_p, top_k, rng_states[lane])?;
            out[lane] = Some(res);
            // Only active lanes advance host RNG (copy back)
            rng_states[lane] = res.1;
        }
        Ok(out)
    }
}
