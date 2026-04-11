//! High-level GPU dispatch interface.
//! Manages compiled kernels, provides typed tensor operations.

use crate::compiler::KernelCompiler;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult, HipRuntime};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

/// gfx1100 multi-row GEMV tile selector.
/// HIPFIRE_GEMV_ROWS ∈ {1, 2, 4, 8}. Default 1 = single-row kernel (legacy).
/// Cached in a OnceLock — the env var is read exactly once per process.
/// Returns the runtime HIPFIRE_GEMV_ROWS override if set, otherwise None.
/// Valid values: 1, 2, 4, 8. Anything else is clamped to 1.
fn gemv_rows_override() -> Option<u32> {
    static CACHE: OnceLock<Option<u32>> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("HIPFIRE_GEMV_ROWS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|r| match r { 1 | 2 | 4 | 8 => r, _ => 1 })
    })
}

/// Per-arch default R for the multi-row HFQ4 GEMV kernel family.
///
/// - RDNA3 (gfx1100/1101/1102): R=1. Measured negative on 7900 XTX —
///   single-row is already near-BW-saturated (577 GiB/s on 9B ≈ 60% of
///   the 960 GiB/s peak) and multi-row under-subscribes the wave scheduler.
/// - RDNA2 (gfx1030/1031): R=1. These have their own arch-optimized narrow
///   kernels via gemv_hfq4g256_for_arch; the multi-row path is bypassed.
/// - Default (gfx1010 baseline, gfx1013 Cyan Skillfish / BC-250, others):
///   R=2. Measured +2.75% on BC-250 Qwen3.5 0.8B MQ4 in the session 1
///   perf work — the x-hoist amortization across 2 rows pays for the
///   minor occupancy drop from 20 → 18 waves/SIMD.
fn gemv_rows_default(arch: &str) -> u32 {
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => 1,
        "gfx1030" | "gfx1031" => 1,
        _ => 2,
    }
}

/// Tensor stored on the GPU. Tracks shape and element type.
pub struct GpuTensor {
    pub buf: DeviceBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q4K,  // 144 bytes per 256 elements
    Q6K,  // 210 bytes per 256 elements
    Q8_0,      // 34 bytes per 32 elements
    Q4F16G64,  // 36 bytes per 64 elements (RDNA-native FP16 dequant)
    Q4F16G32,  // 20 bytes per 32 elements (RDNA-native FP16 dequant)
    Q8HFQ,     // split-metadata: scales contiguous then values contiguous, 128B-aligned rows
    HFQ4G256,  // 136 bytes per 256 elements (flat 4-bit, f32 scale+zero, 18 VGPRs)
    HFQ4G128,  // 72 bytes per 128 elements (flat 4-bit, f32 scale+zero, 14 VGPRs)
    HFQ3G256,  // 104 bytes per 256 elements (flat 3-bit, f32 scale+zero)
    HFQ3G128,  // 56 bytes per 128 elements (flat 3-bit, f32 scale+zero)
    MQ4G256,   // MagnumQuant: FWHT-rotated HFQ4-G256 (136 bytes/group, same as HFQ4G256)
    MQ8G256,   // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target (258 bytes/group)
    HFQ2G256,  // 72 bytes per 256 elements (flat 2-bit, f32 scale+zero, ~19 VGPRs)
    HFQ2G128,  // 40 bytes per 128 elements (flat 2-bit, f32 scale+zero)
    HFQ6G256,  // 200 bytes per 256 elements (6-bit, f32 scale+zero)
    Raw,       // raw bytes, no element interpretation
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Q4F16G64 | DType::Q4F16G32 | DType::Q8HFQ | DType::HFQ4G256 | DType::HFQ4G128 | DType::HFQ3G256 | DType::HFQ3G128 | DType::HFQ2G256 | DType::HFQ2G128 | DType::HFQ6G256 | DType::MQ4G256 | DType::MQ8G256 | DType::Raw => 1, // byte-level
        }
    }
}

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
pub struct Gpu {
    pub hip: HipRuntime,
    pub arch: String,
    compiler: KernelCompiler,
    modules: HashMap<String, hip_bridge::Module>,
    functions: HashMap<String, hip_bridge::Function>,
    pool: crate::pool::GpuPool,
    /// When set, all kernel launches go to this stream instead of null stream.
    pub active_stream: Option<hip_bridge::Stream>,
    /// MagnumQuant FWHT signs (256 floats each) + rotation scratch buffer.
    pub mq_signs1: Option<GpuTensor>,
    pub mq_signs2: Option<GpuTensor>,
    pub mq_x_rot: Option<GpuTensor>,  // scratch for rotated x, sized to max K
    pub mq_x_q8: Option<hip_bridge::DeviceBuffer>,   // INT8 quantized rotated x for dp4a
    pub mq_x_scales: Option<hip_bridge::DeviceBuffer>, // per-group f32 scales for x quantization
}

impl Gpu {
    /// Returns the active stream ref for kernel launches (None = null stream).
    fn stream_ref(&self) -> Option<&hip_bridge::Stream> {
        self.active_stream.as_ref()
    }

    pub fn init() -> HipResult<Self> {
        let hip = HipRuntime::load()?;
        let count = hip.device_count()?;
        if count == 0 {
            return Err(hip_bridge::HipError::new(0, "no GPU devices found"));
        }
        hip.set_device(0)?;

        let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1010".to_string());
        let (vram_free, vram_total) = hip.get_vram_info().unwrap_or((0, 0));

        // Check HIP runtime version matches GPU arch requirements
        let (hip_major, hip_minor) = hip.runtime_version().unwrap_or((0, 0));
        let (min_major, min_minor) = match arch.as_str() {
            "gfx1200" | "gfx1201" => (6, 4), // RDNA4 needs ROCm 6.4+
            "gfx1100" | "gfx1101" | "gfx1102" => (5, 5), // RDNA3 needs ROCm 5.5+
            _ => (5, 0),
        };
        if hip_major > 0 && (hip_major < min_major || (hip_major == min_major && hip_minor < min_minor)) {
            eprintln!("WARNING: HIP runtime {}.{} may not support {}. Minimum: {}.{}", hip_major, hip_minor, arch, min_major, min_minor);
            eprintln!("  Update your HIP runtime or kernels may fail to load.");
        }
        eprintln!("GPU: {} ({:.1} GB VRAM, HIP {}.{})", arch, vram_total as f64 / 1e9, hip_major, hip_minor);

        let compiler = KernelCompiler::new(&arch)?;

        Ok(Self {
            hip,
            arch,
            compiler,
            modules: HashMap::new(),
            functions: HashMap::new(),
            pool: crate::pool::GpuPool::new(),
            active_stream: None,
            mq_signs1: None,
            mq_signs2: None,
            mq_x_rot: None,
            mq_x_q8: None,
            mq_x_scales: None,
        })
    }

    /// Compile and load a kernel if missing. Public variant of `ensure_kernel`
    /// for callers that need to JIT a kernel by name from outside the crate
    /// (primarily the hipGraph capture/replay path).
    pub fn ensure_kernel_public(
        &mut self,
        module_name: &str,
        source: &str,
        func_name: &str,
    ) -> HipResult<()> {
        self.ensure_kernel(module_name, source, func_name)
    }

    /// Launch a pre-loaded kernel by name using the `extra`-mode kernarg
    /// blob path. This is the only launch path that survives hipGraph
    /// capture on gfx1100 / ROCm 6.x — the traditional `kernelParams`
    /// (`void**`) path records stack pointers that dangle by the time the
    /// captured graph is replayed.
    ///
    /// Caller is responsible for:
    ///  - keeping `kernargs` alive across the life of any graph that
    ///    captured this launch (HIP records the blob pointer, not the data);
    ///  - building `kernargs` with the layout matching the kernel signature
    ///    (use `hip_bridge::KernargBlob` for correct alignment).
    pub fn launch_kernel_blob(
        &self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernargs: &mut [u8],
    ) -> HipResult<()> {
        let func = self.functions.get(func_name).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("launch_kernel_blob: function '{func_name}' not loaded"))
        })?;
        unsafe {
            self.hip.launch_kernel_blob(func, grid, block, shared_mem, self.stream_ref(), kernargs)
        }
    }

    /// Compile and load a kernel, caching the result.
    fn ensure_kernel(&mut self, module_name: &str, source: &str, func_name: &str) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }

        let obj_path = self.compiler.compile(module_name, source)?;
        let obj_path_str = obj_path.to_str().unwrap().to_string();

        if !self.modules.contains_key(module_name) {
            let module = self.hip.module_load(&obj_path_str)?;
            self.modules.insert(module_name.to_string(), module);
        }

        let module = &self.modules[module_name];
        let func = self.hip.module_get_function(module, func_name)?;
        self.functions.insert(func_name.to_string(), func);
        Ok(())
    }

    /// Pre-compile a batch of kernels in parallel (hipcc), then load modules + functions.
    /// Each entry is (module_name, source, func_name). Turbo kernels should have
    /// TURBO_COMMON_SRC already prepended in their source.
    pub fn precompile_kernels(&mut self, specs: &[(&str, &str, &str)]) -> HipResult<()> {
        // Collect (name, source) pairs for the compiler batch, skipping already-loaded
        let batch: Vec<(&str, &str)> = specs.iter()
            .filter(|(_, _, func)| !self.functions.contains_key(*func))
            .map(|(module, source, _)| (*module, *source))
            .collect();

        if batch.is_empty() {
            return Ok(());
        }

        // Parallel hipcc compilation
        self.compiler.compile_batch(&batch)?;

        // Now load modules + extract functions (must be sequential — GPU API calls)
        for &(module_name, source, func_name) in specs {
            if self.functions.contains_key(func_name) {
                continue;
            }
            let obj_path = self.compiler.compile(module_name, source)?;
            let obj_path_str = obj_path.to_str().unwrap().to_string();
            if !self.modules.contains_key(module_name) {
                let module = self.hip.module_load(&obj_path_str)?;
                self.modules.insert(module_name.to_string(), module);
            }
            let module = &self.modules[module_name];
            let func = self.hip.module_get_function(module, func_name)?;
            self.functions.insert(func_name.to_string(), func);
        }
        Ok(())
    }

    // ── Tensor allocation ───────────────────────────────────────

    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        let numel: usize = shape.iter().product();
        let byte_size = numel * dtype.size();
        let buf = self.pool.alloc(&self.hip, byte_size)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }

    pub fn upload_f32(&mut self, data: &[f32], shape: &[usize]) -> HipResult<GpuTensor> {
        let tensor = self.alloc_tensor(shape, DType::F32)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(tensor)
    }

    pub fn download_f32(&self, tensor: &GpuTensor) -> HipResult<Vec<f32>> {
        let numel = tensor.numel();
        let mut data = vec![0.0f32; numel];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 4)
        };
        self.hip.memcpy_dtoh(bytes, &tensor.buf)?;
        Ok(data)
    }

    pub fn zeros(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        let tensor = self.alloc_tensor(shape, dtype)?;
        self.hip.memset(&tensor.buf, 0, tensor.byte_size())?;
        Ok(tensor)
    }

    /// GPU-side embedding lookup: copy row `token_id` from embedding table to output.
    /// Avoids downloading the entire embedding table to CPU.
    pub fn embedding_lookup(
        &self,
        table: &GpuTensor,  // [vocab_size * dim] F32
        output: &GpuTensor, // [dim] F32
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        let byte_offset = (token_id as usize) * dim * 4;
        let byte_size = dim * 4;
        self.hip.memcpy_dtod_offset(&output.buf, &table.buf, byte_offset, byte_size)
    }

    /// Q4_LUT GEMV: 4-bit with LDS codebook lookup. 48 bytes per 32 elements.
    pub fn gemv_q4lut(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4lut", kernels::GEMV_Q4LUT_SRC, "gemv_q4lut")?;
        let func = &self.functions["gemv_q4lut"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // LDS: 8 codebooks × 16 entries × 2 bytes = 256 bytes
        let shared_mem = 256u32;
        unsafe {
            self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], shared_mem, None, &mut params)
        }
    }

    /// Wave-cooperative Q4 GEMV (Q4_F16_G32 format, 0.625 B/w). Shuffle-based nibble distribution.
    pub fn gemv_q4wave(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4wave", kernels::GEMV_Q4WAVE_SRC, "gemv_q4wave")?;
        let func = &self.functions["gemv_q4wave"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params) }
    }

    /// Q4-as-Q8 GEMV: 4-bit precision stored in Q8_0 format (1.0625 B/w). Gets Q8 occupancy.
    pub fn gemv_q4as8(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4as8", kernels::GEMV_Q4AS8_SRC, "gemv_q4as8")?;
        let func = &self.functions["gemv_q4as8"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params) }
    }

    /// Q8_0 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_q8(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("embedding_q8", kernels::EMBEDDING_Q8_SRC, "embedding_q8")?;
        let func = &self.functions["embedding_q8"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// Q4_K embedding lookup: dequantize one row on GPU, output F32.
    /// table is raw Q4_K bytes on GPU, output is [dim] F32.
    pub fn embedding_lookup_q4k(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("embedding_q4k", kernels::EMBEDDING_Q4K_SRC, "embedding_q4k")?;
        let func = &self.functions["embedding_q4k"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// HFQ4-G256 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g256(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("embedding_hfq4g256", kernels::EMBEDDING_HFQ4G256_SRC, "embedding_hfq4g256")?;
        let func = &self.functions["embedding_hfq4g256"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::embedding_hfq4g256_bytes(dim);
        let timer = crate::profile::begin_timer(&self.hip, "embedding", "embedding_lookup_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ4-G128 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g128(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("embedding_hfq4g128", kernels::EMBEDDING_HFQ4G128_SRC, "embedding_hfq4g128")?;
        let func = &self.functions["embedding_hfq4g128"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        }
    }

    /// Upload raw bytes to GPU (for quantized weights).
    pub fn upload_raw(&self, data: &[u8], shape: &[usize]) -> HipResult<GpuTensor> {
        let buf = self.hip.malloc(data.len())?;
        self.hip.memcpy_htod(&buf, data)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype: DType::Raw,
        })
    }

    pub fn free_tensor(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.pool.free(tensor.buf);
        Ok(())
    }

    /// Drain the GPU memory pool — actually calls hipFree on all pooled buffers.
    /// Call after model unload to return VRAM to the system.
    pub fn drain_pool(&mut self) {
        self.pool.drain(&self.hip);
    }

    // ── Kernel operations ───────────────────────────────────────

    /// y = A * x (matrix-vector multiply, A is [M, K], x is [K], y is [M])
    pub fn gemv_f32(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv", kernels::GEMV_SRC, "gemv_f32")?;
        let func = &self.functions["gemv_f32"];

        let m = a.shape[0] as i32;
        let k = a.shape[1] as i32;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m;
        let mut k_val = k;
        let mut alpha_val = alpha;
        let mut beta_val = beta;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut alpha_val as *mut _ as *mut c_void,
            &mut beta_val as *mut _ as *mut c_void,
        ];

        // One block per row, 256 threads per block with shared memory reduction
        let block_size = 256u32.min(k as u32);
        let shared_mem = block_size * 4; // one float per thread
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4k * x (quantized matrix-vector multiply, A stored as Q4_K on GPU)
    /// a_raw: raw Q4_K bytes on GPU, x: F32 input, y: F32 output
    /// m: number of output rows, k: number of input columns (must be multiple of 256)
    pub fn gemv_q4k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4k", kernels::GEMV_Q4K_SRC, "gemv_q4k")?;
        let func = &self.functions["gemv_q4k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory needed
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// HFQ4-G128 GEMV: flat 4-bit with 128-weight groups.
    /// K must be multiple of 128.
    pub fn gemv_hfq4g128(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq4g128", kernels::GEMV_HFQ4G128_SRC, "gemv_hfq4g128")?;
        let func = &self.functions["gemv_hfq4g128"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched HFQ4-G128 GEMM. Same tiled approach as G256.
    pub fn gemm_hfq4g128(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemm_hfq4g128", kernels::GEMM_HFQ4G128_SRC, "gemm_hfq4g128")?;
        let func = &self.functions["gemm_hfq4g128"];
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];
        let batch_tiles = ((batch_size + 7) / 8) as u32;
        unsafe {
            self.hip.launch_kernel(func, [m as u32, batch_tiles, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
        }
    }

    /// HFQ2-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq2g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq2g256", kernels::GEMV_HFQ2G256_SRC, "gemv_hfq2g256")?;
        let func = &self.functions["gemv_hfq2g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Lazily initialize MagnumQuant FWHT sign tables (256 floats each, seeds 42 and 1042).
    pub fn ensure_mq_signs(&mut self) -> HipResult<()> {
        if self.mq_signs1.is_some() { return Ok(()); }
        fn gen_signs(seed: u32) -> Vec<f32> {
            let mut state = seed;
            (0..256).map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
                if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
            }).collect()
        }
        let s1 = gen_signs(42);
        let s2 = gen_signs(1042);
        let s1b: Vec<u8> = s1.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2b: Vec<u8> = s2.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1t = self.alloc_tensor(&[256], DType::F32)?;
        let s2t = self.alloc_tensor(&[256], DType::F32)?;
        self.hip.memcpy_htod(&s1t.buf, &s1b)?;
        self.hip.memcpy_htod(&s2t.buf, &s2b)?;
        // Allocate scratch buffers — 32K elements covers K up to 32768
        let x_rot = self.alloc_tensor(&[32768], DType::F32)?;
        let x_q8 = self.hip.malloc(32768)?;  // INT8 buffer for dp4a
        let x_scales = self.hip.malloc(128 * 4)?; // up to 128 groups × f32
        self.mq_signs1 = Some(s1t);
        self.mq_signs2 = Some(s2t);
        self.mq_x_rot = Some(x_rot);
        self.mq_x_q8 = Some(x_q8);
        self.mq_x_scales = Some(x_scales);
        Ok(())
    }

    /// MagnumQuant GEMV: FWHT-rotated HFQ4-G256. Rotates x per group via ds_swizzle,
    /// then standard 4-bit dot product. signs1/signs2 are the FWHT sign tables (256 floats each).
    pub fn gemv_mq4g256(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        signs1: &GpuTensor, signs2: &GpuTensor,
        m: usize, k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC, "gemv_mq4g256")?;
        let func = &self.functions["gemv_mq4g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut s1_ptr = signs1.buf.as_ptr(); let mut s2_ptr = signs2.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut s1_ptr as *mut _ as *mut c_void, &mut s2_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void, &mut k_val as *mut _ as *mut c_void,
        ];
        // LDS for rotated x: 256 floats = 1024 bytes
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 1024, self.stream_ref(), &mut params) }
    }

    /// Fused RMSNorm + MagnumQuant FWHT rotation. Replaces the
    /// `rmsnorm_f32` + `rotate_x_mq` sequence with a single kernel launch.
    /// Reads unnormalized `x` + rmsnorm `weight`, computes rmsnorm in LDS,
    /// applies the same per-256-element FWHT as `mq_rotate_x`, and writes
    /// the rotated normalized vector into `x_rot`.
    ///
    /// Preconditions:
    /// - `k` is a multiple of 256 (enforced by callers via `config.dim`)
    /// - `k` ≤ 16384 (LDS ceiling; 16K floats = 64KB minus reduce buffer)
    pub fn fused_rmsnorm_rotate_mq(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate",
            kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
            "fused_rmsnorm_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let func = &self.functions["fused_rmsnorm_mq_rotate"];
        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        // Dynamic LDS: K floats for x_shared + 256 floats for reduce buffer.
        let shared_mem = ((k + 256) * 4) as u32;

        // Bandwidth: read x (K*4) + weight (K*4) + signs (2*256*4) + write x_rot (K*4)
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fused",
            "fused_rmsnorm_mq_rotate",
            bytes,
        );
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
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

    /// Batched `fused_rmsnorm_rotate_mq`. Grid.x is the batch dim — processes
    /// N tokens' [N × K] x into [N × K] x_rot in a single launch. Byte-exact
    /// against calling `fused_rmsnorm_rotate_mq` N times on separate x/x_rot
    /// buffers. Weight/signs are shared across the batch.
    pub fn fused_rmsnorm_rotate_mq_batched(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_rmsnorm_mq_rotate",
            kernels::FUSED_RMSNORM_MQ_ROTATE_SRC,
            "fused_rmsnorm_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();

        let func = &self.functions["fused_rmsnorm_mq_rotate"];
        let mut xp = x.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut eps_v = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = ((k + 256) * 4) as u32;
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_rmsnorm_mq_rotate_batched", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused SwiGLU + FWHT rotation. Reads gate/up, computes
    /// silu(gate[k])*up[k] on the fly, applies FWHT rotation, writes x_rot.
    /// Used as the w_down input stage for MQ4 — replaces the pair
    /// silu_mul_f32 + mq_rotate_x with one launch.
    pub fn fused_silu_mul_rotate_mq(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let func = &self.functions["fused_silu_mul_mq_rotate"];
        let mut gp = gate.buf.as_ptr();
        let mut up_p = up.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        // Bandwidth: read gate + up, 2x256 signs, write x_rot.
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [n_groups, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `fused_silu_mul_rotate_mq`. Grid.y is the batch dim — processes
    /// N tokens' [N × K] gate/up/x_rot in a single launch.
    pub fn fused_silu_mul_rotate_mq_batched(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            "fused_silu_mul_mq_rotate",
            kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC,
            "fused_silu_mul_mq_rotate",
        )?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let func = &self.functions["fused_silu_mul_mq_rotate"];
        let mut gp = gate.buf.as_ptr();
        let mut up_p = up.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up_p as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_silu_mul_mq_rotate_batched", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [n_groups, batch_size as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.
    /// Exposed so callers can batch one rotation across multiple GEMVs that share x
    /// (e.g., Q/K/V projections all consume the same post-RMSNorm x).
    pub fn rotate_x_mq(&mut self, x: &GpuTensor, x_rot: &GpuTensor, k: usize) -> HipResult<()> {
        self.ensure_mq_signs()?;
        self.ensure_kernel("mq_rotate_x", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let rot_func = &self.functions["mq_rotate_x"];
        let mut xp = x.buf.as_ptr(); let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr; let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut xrp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void, &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x", bytes);
        let result = unsafe { self.hip.launch_kernel(rot_func, [n_groups, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `rotate_x_mq`. Grid.y is the batch dim.
    pub fn rotate_x_mq_batched(
        &mut self,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_mq_signs()?;
        self.ensure_kernel("mq_rotate_x", kernels::GEMV_MQ4G256_SRC, "mq_rotate_x")?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let rot_func = &self.functions["mq_rotate_x"];
        let mut xp = x.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fwht", "mq_rotate_x_batched", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                rot_func,
                [n_groups, batch_size as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// MagnumQuant MQ4: rotate x once, then GEMV against rotated x.
    /// MQ4 weights are stored in HFQ4-G256 format with FWHT pre-applied, so the GEMV
    /// inner loop is identical to standard HFQ4 — we reuse the arch-tuned HFQ4 kernel.
    pub fn gemv_mq4g256_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        x_rot: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.rotate_x_mq(x, x_rot, k)?;
        // MQ4 = FWHT-rotated HFQ4-G256. dot(rot(W), rot(x)) = dot(W, x).
        // Route through the arch-specific HFQ4 kernel (4x unroll on gfx1100, etc).
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }

    /// MagnumQuant MQ4 with pre-rotated x. Skips the rotation step entirely —
    /// caller must have called `rotate_x_mq` into `x_rot` first.
    pub fn gemv_mq4g256_prerotated(
        &mut self, a_raw: &GpuTensor, x_rot: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.gemv_hfq4g256(a_raw, x_rot, y, m, k)
    }

    /// Standalone MQ8 rotate + INT8 quantize of x into internal `mq_x_q8`/`mq_x_scales`.
    /// After this, `gemv_mq8g256_prerotated` can be called multiple times with the same x.
    pub fn rotate_quantize_x_mq8(&mut self, x: &GpuTensor, k: usize) -> HipResult<()> {
        self.ensure_mq_signs()?;
        self.ensure_kernel("mq8_rotate_quantize_x", kernels::GEMV_MQ8G256_SRC, "mq8_rotate_quantize_x")?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;

        let rot_func = &self.functions["mq8_rotate_quantize_x"];
        let mut xp = x.buf.as_ptr();
        let mut xq = xq_ptr; let mut xs = xs_ptr;
        let mut s1 = s1_ptr; let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void, &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(rot_func, [n_groups, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// MQ8 dp4a GEMV using pre-rotated+quantized x. Caller must have called
    /// `rotate_quantize_x_mq8(x, k)` first — results use the internal `mq_x_q8`/`mq_x_scales`.
    pub fn gemv_mq8g256_prerotated(
        &mut self, a_raw: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_mq8g256", kernels::GEMV_MQ8G256_SRC, "gemv_mq8g256")?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();

        let func = &self.functions["gemv_mq8g256"];
        let mut ap = a_raw.buf.as_ptr();
        let mut xq = xq_ptr; let mut xs = xs_ptr;
        let mut yp = y.buf.as_ptr();
        let mut mv = m as i32; let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void, &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void, &mut kv as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// MagnumQuant MQ8: FWHT rotate + INT8 quantize x, then dp4a GEMV.
    pub fn gemv_mq8g256_with_rotate(
        &mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize,
    ) -> HipResult<()> {
        self.rotate_quantize_x_mq8(x, k)?;
        self.gemv_mq8g256_prerotated(a_raw, y, m, k)
    }

    /// HFQ3-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq3g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq3g256", kernels::GEMV_HFQ3G256_SRC, "gemv_hfq3g256")?;
        let func = &self.functions["gemv_hfq3g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ3-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq3g128(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq3g128", kernels::GEMV_HFQ3G128_SRC, "gemv_hfq3g128")?;
        let func = &self.functions["gemv_hfq3g128"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ2-G128 GEMV. K must be multiple of 128. Finer granularity than G256.
    pub fn gemv_hfq2g128(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq2g128", kernels::GEMV_HFQ2G128_SRC, "gemv_hfq2g128")?;
        let func = &self.functions["gemv_hfq2g128"];
        let mut ap = a_raw.buf.as_ptr(); let mut xp = x.buf.as_ptr(); let mut yp = y.buf.as_ptr();
        let mut mv = m as i32; let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ6-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq6g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC, "gemv_hfq6g256")?;
        let func = &self.functions["gemv_hfq6g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ8-G256 GEMV. K must be multiple of 256.
    pub fn gemv_hfq8g256(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq8g256", kernels::GEMV_HFQ8G256_SRC, "gemv_hfq8g256")?;
        let func = &self.functions["gemv_hfq8g256"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ4-G512 GEMV. K must be multiple of 512.
    pub fn gemv_hfq4g512(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq4g512", kernels::GEMV_HFQ4G512_SRC, "gemv_hfq4g512")?;
        let func = &self.functions["gemv_hfq4g512"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ4-G1024 GEMV. K must be multiple of 1024.
    pub fn gemv_hfq4g1024(&mut self, a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> HipResult<()> {
        self.ensure_kernel("gemv_hfq4g1024", kernels::GEMV_HFQ4G1024_SRC, "gemv_hfq4g1024")?;
        let func = &self.functions["gemv_hfq4g1024"];
        let mut a_ptr = a_raw.buf.as_ptr(); let mut x_ptr = x.buf.as_ptr(); let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32; let mut k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void, &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void, &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// HFQ4-G256 GEMV: flat 4-bit with 256-weight groups. K must be multiple of 256.
    pub fn gemv_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        let (hfq4g256_src, hfq4g256_module) = kernels::gemv_hfq4g256_for_arch(&self.arch);
        self.ensure_kernel(hfq4g256_module, hfq4g256_src, "gemv_hfq4g256")?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Multi-row GEMV: one warp computes R output rows, sharing x register
        // state across rows. Per-arch default picks R=1 on RDNA3 (negative)
        // and RDNA2 (has its own arch-specific narrow path), R=2 on the
        // default gfx1010-baseline path (gfx1010, gfx1013 Cyan Skillfish,
        // etc.). Override any arch with HIPFIRE_GEMV_ROWS ∈ {1, 2, 4, 8}.
        //
        // See gemv_rows_default() for the measurement data that motivates
        // the per-arch defaults.
        let rdna3 = matches!(self.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102");
        let rows = gemv_rows_override().unwrap_or_else(|| gemv_rows_default(self.arch.as_str()));
        let use_multirow = rows > 1;

        // RDNA2 (gfx1030/1031): always use the arch-optimized narrow kernel.
        // Other non-RDNA3 archs: use wide kernel (2 rows/block) for large M.
        let use_wide = !use_multirow
            && m >= 64
            && !matches!(self.arch.as_str(), "gfx1030" | "gfx1031" | "gfx1100" | "gfx1101" | "gfx1102");

        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256", bytes);
        let result = if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            let (mr_name, mr_src) = if rdna3 {
                ("gemv_hfq4g256_multirow_rdna3", kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC)
            } else {
                ("gemv_hfq4g256_multirow_default", kernels::GEMV_HFQ4G256_MULTIROW_SRC)
            };
            self.ensure_kernel(mr_name, mr_src, func_name)?;
            let mrfunc = &self.functions[func_name];
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            unsafe {
                self.hip.launch_kernel(mrfunc, [grid, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
            }
        } else if use_wide {
            self.ensure_kernel("gemv_hfq4g256_wide", kernels::GEMV_HFQ4G256_WIDE_SRC, "gemv_hfq4g256_wide")?;
            let wfunc = &self.functions["gemv_hfq4g256_wide"];
            let grid = ((m + 1) / 2) as u32;
            unsafe {
                self.hip.launch_kernel(wfunc, [grid, 1, 1], [64, 1, 1], 0, self.stream_ref(), &mut params)
            }
        } else {
            let func = &self.functions["gemv_hfq4g256"];
            unsafe {
                self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
            }
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    // HFQ2 GEMV dispatch already exists at line ~521 from the HFQ family

    /// 3-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_q=A_q·x, y_k=A_k·x, y_v=A_v·x in a single kernel launch
    /// for the Qwen3.5 FullAttention layer preamble. Same rationale and
    /// tail-handling guarantees as `fused_qkvza_hfq4g256`.
    pub fn fused_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "fused_qkv_hfq4g256",
            kernels::FUSED_QKV_HFQ4G256_SRC,
            "fused_qkv_hfq4g256",
        )?;
        let func = &self.functions["fused_qkv_hfq4g256"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let total_m = (q_m + k_m + v_m) as u32;
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkv_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [total_m, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// 4-way fused HFQ4-G256 projection — cross-arch.
    ///
    /// Performs y_qkv=A_qkv·x, y_z=A_z·x, y_beta=A_beta·x, y_alpha=A_alpha·x
    /// in a single kernel launch, where all four matrices share the same
    /// input `x` and the same K. Used by the Qwen3.5 DeltaNet LA layer
    /// preamble to collapse four launches (one per projection) into one.
    /// Bit-exact with four sequential `gemv_hfq4g256` calls.
    ///
    /// Works on every RDNA generation (gfx1010 / gfx1013 / gfx1030 /
    /// gfx1100+) because the inner loop and the standalone gemv_hfq4g256
    /// inner loop were unified onto the same 4-accumulator structure
    /// after commit 5302926.
    pub fn fused_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "fused_qkvza_hfq4g256",
            kernels::FUSED_QKVZA_HFQ4G256_SRC,
            "fused_qkvza_hfq4g256",
        )?;
        let func = &self.functions["fused_qkvza_hfq4g256"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;
        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qkvza_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [total_m, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 4-way fused HFQ4-G256 GEMM for the LA preamble.
    ///
    /// Processes N tokens × four projections (wqkv + wz + w_beta + w_alpha)
    /// in one launch. Bitwise-identical output to calling `fused_qkvza_hfq4g256`
    /// N times on the same x[b] — 4-accumulator interleave + pairwise combine
    /// are preserved per batch element.
    ///
    /// `x`: [N × K] row-major activation batch.
    /// `y_*`: [N × *_m] row-major outputs (overwrite semantics).
    pub fn gemm_qkvza_hfq4g256(
        &mut self,
        a_qkv: &GpuTensor, a_z: &GpuTensor, a_beta: &GpuTensor, a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor, y_z: &GpuTensor, y_beta: &GpuTensor, y_alpha: &GpuTensor,
        qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "gemm_qkvza_hfq4g256",
            kernels::GEMM_QKVZA_HFQ4G256_SRC,
            "gemm_qkvza_hfq4g256",
        )?;
        let func = &self.functions["gemm_qkvza_hfq4g256"];

        let mut aq = a_qkv.buf.as_ptr();
        let mut az = a_z.buf.as_ptr();
        let mut ab = a_beta.buf.as_ptr();
        let mut aa = a_alpha.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_qkv.buf.as_ptr();
        let mut yz = y_z.buf.as_ptr();
        let mut yb = y_beta.buf.as_ptr();
        let mut ya = y_alpha.buf.as_ptr();
        let mut q_m = qkv_m as i32;
        let mut z_m_val = z_m as i32;
        let mut b_m = beta_m as i32;
        let mut a_m = alpha_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut az as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
            &mut aa as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yz as *mut _ as *mut c_void,
            &mut yb as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut q_m as *mut _ as *mut c_void,
            &mut z_m_val as *mut _ as *mut c_void,
            &mut b_m as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (qkv_m + z_m + beta_m + alpha_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(z_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(alpha_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkvza_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 3-way fused HFQ4-G256 GEMM for the FA preamble.
    ///
    /// Processes N tokens × three projections (wq + wk + wv) in one launch.
    /// Bitwise-identical to calling `fused_qkv_hfq4g256` N times on the same
    /// x[b] — 4-accumulator interleave + pairwise combine preserved per
    /// batch element.
    pub fn gemm_qkv_hfq4g256(
        &mut self,
        a_q: &GpuTensor, a_k: &GpuTensor, a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor, y_k: &GpuTensor, y_v: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "gemm_qkv_hfq4g256",
            kernels::GEMM_QKV_HFQ4G256_SRC,
            "gemm_qkv_hfq4g256",
        )?;
        let func = &self.functions["gemm_qkv_hfq4g256"];

        let mut aq = a_q.buf.as_ptr();
        let mut ak = a_k.buf.as_ptr();
        let mut av = a_v.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yq = y_q.buf.as_ptr();
        let mut yk = y_k.buf.as_ptr();
        let mut yv = y_v.buf.as_ptr();
        let mut q_m_val = q_m as i32;
        let mut k_m_val = k_m as i32;
        let mut v_m_val = v_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yq as *mut _ as *mut c_void,
            &mut yk as *mut _ as *mut c_void,
            &mut yv as *mut _ as *mut c_void,
            &mut q_m_val as *mut _ as *mut c_void,
            &mut k_m_val as *mut _ as *mut c_void,
            &mut v_m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (q_m + k_m + v_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(k_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(v_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_qkv_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched 2-way fused HFQ4-G256 GEMM for the FFN preamble (gate + up).
    ///
    /// Processes N tokens × both projections (w_gate + w_up) in one launch.
    /// Bitwise-identical to calling `fused_gate_up_hfq4g256` N times on the
    /// same x[b] — 4-accumulator interleave + pairwise combine preserved
    /// per batch element.
    pub fn gemm_gate_up_hfq4g256(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "gemm_gate_up_hfq4g256",
            kernels::GEMM_GATE_UP_HFQ4G256_SRC,
            "gemm_gate_up_hfq4g256",
        )?;
        let func = &self.functions["gemm_gate_up_hfq4g256"];

        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut g_m = gate_m as i32;
        let mut u_m = up_m as i32;
        let mut k_val = k as i32;
        let mut n_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void,
            &mut g_m as *mut _ as *mut c_void,
            &mut u_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;
        let total_m = (gate_m + up_m) as u32;

        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
                  + crate::profile::gemv_hfq4g256_bytes(up_m, k);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_gate_up_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [total_m, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// HFQ4-G256 GEMV with fused residual add: y[row] += A[row] · x.
    /// Same math as `gemv_hfq4g256` but the final write accumulates into `y`
    /// instead of overwriting. Used for wo / w_down projections where the
    /// following step would have been `x += gemv_out` via add_inplace_f32.
    pub fn gemv_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        let (src, module) = kernels::gemv_hfq4g256_residual_for_arch(&self.arch);
        self.ensure_kernel(module, src, "gemv_hfq4g256_residual")?;

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // RDNA3 multi-row override path. Same selector as the non-residual
        // variant but there's currently no gfx1010-default multi-row residual
        // kernel, so non-RDNA3 archs still take the single-row residual path
        // regardless of HIPFIRE_GEMV_ROWS. (TODO: port the multi-row residual
        // kernel to the default path if/when the non-residual multi-row wins
        // scale to justify residual too.)
        let rdna3 = matches!(self.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102");
        let rows = if rdna3 { gemv_rows_override().unwrap_or(1) } else { 1 };
        let use_multirow = rdna3 && rows > 1;

        // Bandwidth: weight + x + y_read (for residual) + y_write.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemv_hfq4g256_residual", bytes);
        let result = if use_multirow {
            let (func_name, grid_div) = match rows {
                2 => ("gemv_hfq4g256_residual_multirow_r2", 2u32),
                4 => ("gemv_hfq4g256_residual_multirow_r4", 4u32),
                8 => ("gemv_hfq4g256_residual_multirow_r8", 8u32),
                _ => unreachable!(),
            };
            self.ensure_kernel(
                "gemv_hfq4g256_residual_multirow_rdna3",
                kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC,
                func_name,
            )?;
            let mrfunc = &self.functions[func_name];
            let grid = ((m as u32) + grid_div - 1) / grid_div;
            unsafe {
                self.hip.launch_kernel(mrfunc, [grid, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
            }
        } else {
            let func = &self.functions["gemv_hfq4g256_residual"];
            unsafe {
                self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
            }
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched HFQ4-G256 GEMM with fused residual add:
    ///   for b in 0..batch_size: y[b][row] += A[row] · x[b]
    ///
    /// Bitwise-identical output to calling `gemv_hfq4g256_residual` N times
    /// (preserves the 4-accumulator interleave and pairwise final combine),
    /// so safe to use in the quality-gated forward path. Each block handles
    /// one row × up to BATCH_TILE (=8) batch elements, amortizing the weight
    /// fetch across the batch loop.
    ///
    /// `x`: [batch_size × K] row-major, `y`: [batch_size × M] row-major.
    /// `y` must already hold the residual summand to accumulate into.
    pub fn gemm_hfq4g256_residual(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "gemm_hfq4g256_residual",
            kernels::GEMM_HFQ4G256_RESIDUAL_SRC,
            "gemm_hfq4g256_residual",
        )?;
        let func = &self.functions["gemm_hfq4g256_residual"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        // BATCH_TILE in the kernel — keep in sync.
        const BATCH_TILE: usize = 8;
        let batch_tiles = (batch_size + BATCH_TILE - 1) / BATCH_TILE;

        // Bandwidth: weight (read once, amortized across the batch loop on-chip
        // via L1/L2), per-batch x read, per-batch y read-modify-write.
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k)
            + batch_size * k * 4
            + batch_size * m * 4 * 2;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", "gemm_hfq4g256_residual", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched HFQ4-G256 GEMM: y[b][row] = A[row] · x[b] for all batch elements.
    /// x: [batch_size × K], y: [batch_size × M], both row-major.
    pub fn gemm_hfq4g256(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemm_hfq4g256", kernels::GEMM_HFQ4G256_SRC, "gemm_hfq4g256")?;
        let func = &self.functions["gemm_hfq4g256"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut bs_val = batch_size as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut bs_val as *mut _ as *mut c_void,
        ];

        let batch_tiles = ((batch_size + 7) / 8) as u32; // ceil(batch_size / BATCH_TILE=8)
        let bytes = crate::profile::gemm_hfq4g256_bytes(m, k, batch_size);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", "gemm_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_tiles, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Compute max softmax probability on GPU. Downloads 4 bytes instead of vocab×4.
    pub fn max_prob(
        &mut self, logits: &GpuTensor, result: &GpuTensor, vocab_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("max_prob", kernels::MAX_PROB_SRC, "max_prob")?;
        let func = &self.functions["max_prob"];
        let mut lp = logits.buf.as_ptr();
        let mut rp = result.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void, &mut rp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let shared = (block * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [1, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Fused QKV: three Q4_K GEMVs in one launch (saves 2 kernel launches per layer).
    /// q = Wq * x, k = Wk * x, v = Wv * x — all read the same input x.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_q4k(
        &mut self,
        wq: &GpuTensor, wk: &GpuTensor, wv: &GpuTensor,
        x: &GpuTensor,
        yq: &GpuTensor, yk: &GpuTensor, yv: &GpuTensor,
        q_m: usize, k_m: usize, v_m: usize, k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("fused_qkv_q4k", kernels::FUSED_QKV_Q4K_SRC, "fused_qkv_q4k")?;
        let func = &self.functions["fused_qkv_q4k"];

        let mut aq = wq.buf.as_ptr();
        let mut ak = wk.buf.as_ptr();
        let mut av = wv.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yqp = yq.buf.as_ptr();
        let mut ykp = yk.buf.as_ptr();
        let mut yvp = yv.buf.as_ptr();
        let mut qm = q_m as i32;
        let mut km = k_m as i32;
        let mut vm = v_m as i32;
        let mut kk = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut aq as *mut _ as *mut c_void,
            &mut ak as *mut _ as *mut c_void,
            &mut av as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yqp as *mut _ as *mut c_void,
            &mut ykp as *mut _ as *mut c_void,
            &mut yvp as *mut _ as *mut c_void,
            &mut qm as *mut _ as *mut c_void,
            &mut km as *mut _ as *mut c_void,
            &mut vm as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];

        let grid = (q_m + k_m + v_m) as u32;
        unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// Fused Gate+Up: two Q4_K GEMVs in one launch (saves 1 kernel launch per layer).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_q4k(
        &mut self,
        w_gate: &GpuTensor, w_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize, k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("fused_gate_up_q4k", kernels::FUSED_GATE_UP_Q4K_SRC, "fused_gate_up_q4k")?;
        let func = &self.functions["fused_gate_up_q4k"];

        let mut ag = w_gate.buf.as_ptr();
        let mut au = w_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut ygp = y_gate.buf.as_ptr();
        let mut yup = y_up.buf.as_ptr();
        let mut gm = gate_m as i32;
        let mut um = up_m as i32;
        let mut kk = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void,
            &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut ygp as *mut _ as *mut c_void,
            &mut yup as *mut _ as *mut c_void,
            &mut gm as *mut _ as *mut c_void,
            &mut um as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];

        let grid = (gate_m + up_m) as u32;
        unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// y = A_q8_0 * x (quantized GEMV for Q8_0)
    pub fn gemv_q8_0(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Adaptive dispatch: wide kernel for small K (more threads per row),
        // narrow kernel for large K (more blocks, better occupancy).
        if k <= 1536 {
            self.ensure_kernel("gemv_q8_0_wide", kernels::GEMV_Q8_0_WIDE_SRC, "gemv_q8_0_wide")?;
            let func = &self.functions["gemv_q8_0_wide"];
            let block_size = 64u32; // 2 warps, each processes one row
            let grid = ((m + 1) / 2) as u32; // ceil(M/2)
            return unsafe {
                self.hip.launch_kernel(func, [grid, 1, 1], [block_size, 1, 1], 0, None, &mut params)
            };
        }

        self.ensure_kernel("gemv_q8_0", kernels::GEMV_Q8_0_SRC, "gemv_q8_0")?;
        let func = &self.functions["gemv_q8_0"];
        let block_size = 32u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q8hfq * x (split-metadata Q8 GEMV, row_stride = padded row bytes)
    pub fn gemv_q8hfq(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        row_stride: usize,
    ) -> HipResult<()> {
        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;
        let mut rs_val = row_stride as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut rs_val as *mut _ as *mut c_void,
        ];

        if k <= 1536 {
            self.ensure_kernel("gemv_q8hfq_wide", kernels::GEMV_Q8HFQ_WIDE_SRC, "gemv_q8hfq_wide")?;
            let func = &self.functions["gemv_q8hfq_wide"];
            let block_size = 64u32;
            let grid = ((m + 1) / 2) as u32;
            return unsafe {
                self.hip.launch_kernel(func, [grid, 1, 1], [block_size, 1, 1], 0, None, &mut params)
            };
        }

        self.ensure_kernel("gemv_q8hfq", kernels::GEMV_Q8HFQ_SRC, "gemv_q8hfq")?;
        let func = &self.functions["gemv_q8hfq"];
        unsafe {
            self.hip.launch_kernel(func, [m as u32, 1, 1], [32, 1, 1], 0, None, &mut params)
        }
    }

    /// y = A_q6k * x (quantized GEMV for Q6_K)
    pub fn gemv_q6k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q6k", kernels::GEMV_Q6K_SRC, "gemv_q6k")?;
        let func = &self.functions["gemv_q6k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 64)
    /// a_raw: raw Q4_F16_G64 bytes on GPU, x: F32 input, y: F32 output
    /// Block: 36 bytes per 64 elements. K must be multiple of 64.
    /// Uses 128 threads (4 warps) with shared memory reduction for increased MLP.
    pub fn gemv_q4f16_g64(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4f16_g64", kernels::GEMV_Q4F16_G64_SRC, "gemv_q4f16_g64")?;
        let func = &self.functions["gemv_q4f16_g64"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (256-thread wide variant for occupancy testing)
    /// Element-strided access pattern matching F32 GEMV. Shared memory reduction.
    pub fn gemv_q4f16_g64_wide(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4f16_g64_wide", kernels::GEMV_Q4F16_G64_WIDE_SRC, "gemv_q4f16_g64_wide")?;
        let func = &self.functions["gemv_q4f16_g64_wide"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4; // one float per thread
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// y = A_q4f16 * x (RDNA-native Q4_F16 GEMV, group size 32)
    /// Block: 20 bytes per 32 elements. K must be multiple of 32.
    pub fn gemv_q4f16_g32(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4f16_g32", kernels::GEMV_Q4F16_G32_SRC, "gemv_q4f16_g32")?;
        let func = &self.functions["gemv_q4f16_g32"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// GPU-side argmax: returns index of max value. Avoids downloading full logits.
    pub fn argmax_f32(&mut self, data: &GpuTensor, n: usize) -> HipResult<u32> {
        self.ensure_kernel("argmax_f32", kernels::ARGMAX_SRC, "argmax_f32")?;
        let func = &self.functions["argmax_f32"];

        let result_buf = self.hip.malloc(4)?; // single int
        self.hip.memset(&result_buf, 0, 4)?;

        let mut dp = data.buf.as_ptr();
        let mut rp = result_buf.as_ptr();
        let mut nn = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared = block_size * 8; // float + int per thread
        unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [block_size, 1, 1], shared, None, &mut params)?;
        }

        let mut result = [0i32];
        let result_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 4)
        };
        self.hip.memcpy_dtoh(result_bytes, &result_buf)?;
        self.hip.free(result_buf)?;
        Ok(result[0] as u32)
    }

    /// out = rmsnorm(x, weight, eps)
    pub fn rmsnorm_f32(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        eps: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("rmsnorm", kernels::RMSNORM_SRC, "rmsnorm_f32")?;
        let func = &self.functions["rmsnorm_f32"];

        let batch = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let mut x_ptr = x.buf.as_ptr();
        let mut w_ptr = weight.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4; // float per thread

        let bytes = crate::profile::rmsnorm_bytes(batch * n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "rmsnorm_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [batch as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched RMSNorm: normalize `batch` vectors of length `n` independently.
    /// x and out can be the same buffer (in-place). Weight is [n], applied per vector.
    pub fn rmsnorm_batched(
        &mut self,
        x: &GpuTensor, weight: &GpuTensor, out: &GpuTensor,
        batch: usize, n: usize, eps: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("rmsnorm", kernels::RMSNORM_SRC, "rmsnorm_f32")?;
        let func = &self.functions["rmsnorm_f32"];

        let mut x_ptr = x.buf.as_ptr();
        let mut w_ptr = weight.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n as i32;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4;
        let bytes = crate::profile::rmsnorm_bytes(batch * n);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "rmsnorm_batched", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [batch as u32, 1, 1], [block_size, 1, 1], shared_mem, None, &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// c = a + b (element-wise)
    pub fn add_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("add", kernels::ADD_SRC, "add_f32")?;
        let func = &self.functions["add_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// a += b (in-place element-wise add)
    pub fn add_inplace_f32(&mut self, a: &GpuTensor, b: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("add_inplace", kernels::ADD_INPLACE_SRC, "add_inplace_f32")?;
        let func = &self.functions["add_inplace_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "add_inplace_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// c = a * b (element-wise)
    pub fn mul_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("mul", kernels::MUL_SRC, "mul_f32")?;
        let func = &self.functions["mul_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "mul_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// out = silu(x)
    pub fn silu_f32(&mut self, x: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("silu", kernels::SILU_SRC, "silu_f32")?;
        let func = &self.functions["silu_f32"];

        let n = x.numel() as i32;
        let mut x_ptr = x.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// out = silu(gate) * up — fused to avoid intermediate buffer
    pub fn silu_mul_f32(&mut self, gate: &GpuTensor, up: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("silu_mul", kernels::SILU_MUL_SRC, "silu_mul_f32")?;
        let func = &self.functions["silu_mul_f32"];

        let n = gate.numel() as i32;
        let mut gate_ptr = gate.buf.as_ptr();
        let mut up_ptr = up.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "silu_mul_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// In-place softmax over last dimension
    pub fn softmax_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("softmax", kernels::SOFTMAX_SRC, "softmax_f32")?;
        let func = &self.functions["softmax_f32"];

        let rows = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let mut x_ptr = x.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32.min(n as u32);
        let shared_mem = block * 4;

        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [block, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// GPU-side RoPE (rotary positional embedding) applied in-place to Q and K.
    /// pos_buf: GPU buffer containing a single i32 position value.
    pub fn rope_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos_buf: &DeviceBuffer,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("rope", kernels::ROPE_SRC, "rope_f32")?;
        let func = &self.functions["rope_f32"];

        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut fb = freq_base;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
        ];

        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid = (half + block - 1) / block;

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

    /// Batched RoPE: apply to [batch_size] positions in one launch.
    /// q: [batch_size × q_dim], k: [batch_size × kv_dim].
    /// positions: GPU buffer of [batch_size] i32 position indices.
    pub fn rope_batched_f32(
        &mut self, q: &GpuTensor, k: &GpuTensor, positions: &GpuTensor,
        n_heads_q: usize, n_heads_k: usize, head_dim: usize, freq_base: f32, batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("rope_batched", kernels::ROPE_BATCHED_SRC, "rope_batched_f32")?;
        let func = &self.functions["rope_batched_f32"];
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k.buf.as_ptr();
        let mut pos_ptr = positions.buf.as_ptr();
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut fb = freq_base;
        let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid_x = (half + block - 1) / block;
        unsafe {
            self.hip.launch_kernel(func, [grid_x, batch_size as u32, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        }
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
        let effective_seq = if self.active_stream.is_some() { max_seq } else { seq_len_hint };
        let block_size = (effective_seq.max(head_dim) as u32).next_power_of_two().min(256);
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
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Choose chunk size: aim for 4-16 chunks
        let chunk_size = if seq_len <= 128 { seq_len } else { 128 };
        let n_chunks = (seq_len + chunk_size - 1) / chunk_size;

        // Phase 1: compute partial attention per chunk
        self.ensure_kernel("attention_flash_partial", kernels::ATTENTION_FLASH_SRC, "attention_flash_partial")?;
        let func1 = &self.functions["attention_flash_partial"];

        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut p_ptr = partials.buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;
        let mut cs = chunk_size as i32;

        let mut params1: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
        ];

        let block_size = 128u32.min(chunk_size as u32).next_power_of_two();
        let shared_mem = ((chunk_size + block_size as usize) * 4) as u32;

        unsafe {
            self.hip.launch_kernel(
                func1,
                [n_heads as u32, n_chunks as u32, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params1,
            )?;
        }

        // Phase 2: reduce partials
        self.ensure_kernel("attention_flash_reduce", kernels::ATTENTION_FLASH_SRC, "attention_flash_reduce")?;
        let func2 = &self.functions["attention_flash_reduce"];

        let mut p_ptr2 = partials.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut nh2 = n_heads as i32;
        let mut nc = n_chunks as i32;
        let mut hd2 = head_dim as i32;

        let mut params2: Vec<*mut c_void> = vec![
            &mut p_ptr2 as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut nh2 as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void,
        ];

        let reduce_block = head_dim.min(256) as u32;
        unsafe {
            self.hip.launch_kernel(
                func2,
                [n_heads as u32, 1, 1],
                [reduce_block, 1, 1],
                0,
                self.stream_ref(),
                &mut params2,
            )
        }
    }

    /// Fused Gate+Up HFQ4-G256: two GEMVs in one launch.
    pub fn fused_gate_up_hfq4g256(
        &mut self,
        a_gate: &GpuTensor, a_up: &GpuTensor, x: &GpuTensor,
        y_gate: &GpuTensor, y_up: &GpuTensor,
        gate_m: usize, up_m: usize, k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("fused_gate_up_hfq4g256", kernels::FUSED_GATE_UP_HFQ4G256_SRC, "fused_gate_up_hfq4g256")?;
        let func = &self.functions["fused_gate_up_hfq4g256"];
        let mut ag = a_gate.buf.as_ptr();
        let mut au = a_up.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yg = y_gate.buf.as_ptr();
        let mut yu = y_up.buf.as_ptr();
        let mut gm = gate_m as i32;
        let mut um = up_m as i32;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ag as *mut _ as *mut c_void, &mut au as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void, &mut yg as *mut _ as *mut c_void,
            &mut yu as *mut _ as *mut c_void, &mut gm as *mut _ as *mut c_void,
            &mut um as *mut _ as *mut c_void, &mut kv as *mut _ as *mut c_void,
        ];
        let total_rows = (gate_m + up_m) as u32;
        unsafe { self.hip.launch_kernel(func, [total_rows, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Write KV to HFQ4 co-located block (72 bytes per head: scale+zero+nibbles).
    pub fn kv_cache_write_hfq4(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_hfq4", kernels::KV_CACHE_WRITE_HFQ4_SRC, "kv_cache_write_hfq4")?;
        let func = &self.functions["kv_cache_write_hfq4"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with HFQ4 KV blocks (72 bytes per head, co-located).
    pub fn attention_hfq4_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_hfq4_kv", kernels::ATTENTION_HFQ4_KV_SRC, "attention_hfq4_kv")?;
        let func = &self.functions["attention_hfq4_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        // scores[seq_len] + ws[block_size] + q_shared[head_dim]
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// HFQ4 KV write with sign-flip decorrelation. Same format, uses TURBO_SIGNS1.
    pub fn kv_cache_write_hfq4s(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("kv_cache_write_hfq4s", kernels::KV_CACHE_WRITE_HFQ4S_SRC, "kv_cache_write_hfq4s")?;
        let func = &self.functions["kv_cache_write_hfq4s"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with HFQ4+sign-flip KV. Uses TURBO_SIGNS1 for Q flip and V inverse.
    pub fn attention_hfq4s_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("attention_hfq4s_kv", kernels::ATTENTION_HFQ4S_KV_SRC, "attention_hfq4s_kv")?;
        let func = &self.functions["attention_hfq4s_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// INT8 co-located with f16 scale (matches Q8_0 precision, one block per head).
    pub fn kv_cache_write_int8c_f16(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_int8c_f16", kernels::KV_CACHE_WRITE_INT8C_F16_SRC, "kv_cache_write_int8c_f16")?;
        let func = &self.functions["kv_cache_write_int8c_f16"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    pub fn attention_int8c_f16_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_int8c_f16_kv", kernels::ATTENTION_INT8C_F16_KV_SRC, "attention_int8c_f16_kv")?;
        let func = &self.functions["attention_int8c_f16_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr(); let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to INT8 co-located block (f32 scale + int8 data, symmetric).
    pub fn kv_cache_write_int8c(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_int8c", kernels::KV_CACHE_WRITE_INT8C_SRC, "kv_cache_write_int8c")?;
        let func = &self.functions["kv_cache_write_int8c"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with INT8 co-located KV blocks.
    pub fn attention_int8c_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_int8c_kv", kernels::ATTENTION_INT8C_KV_SRC, "attention_int8c_kv")?;
        let func = &self.functions["attention_int8c_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to HFQ8 cache (FP32 scale+zero, contiguous uint8).
    pub fn kv_cache_write_hfq8(
        &mut self, dst_data: &GpuTensor, dst_scales: &GpuTensor, src: &GpuTensor,
        pos_buf: &DeviceBuffer, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_hfq8", kernels::KV_CACHE_WRITE_HFQ8_SRC, "kv_cache_write_hfq8")?;
        let func = &self.functions["kv_cache_write_hfq8"];
        let mut dd = dst_data.buf.as_ptr(); let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dd as *mut _ as *mut c_void, &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void, &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with HFQ8 KV cache.
    pub fn attention_hfq8_kv(
        &mut self, q: &GpuTensor,
        k_data: &GpuTensor, k_scales: &GpuTensor,
        v_data: &GpuTensor, v_scales: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_hfq8_kv", kernels::ATTENTION_HFQ8_KV_SRC, "attention_hfq8_kv")?;
        let func = &self.functions["attention_hfq8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr();
        let mut kd = k_data.buf.as_ptr(); let mut ks = k_scales.buf.as_ptr();
        let mut vd = v_data.buf.as_ptr(); let mut vs = v_scales.buf.as_ptr();
        let mut op = out.buf.as_ptr(); let mut pp = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void, &mut ks as *mut _ as *mut c_void,
            &mut vd as *mut _ as *mut c_void, &mut vs as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut pp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV to INT8 cache (separate scale array).
    pub fn kv_cache_write_int8(
        &mut self, dst_vals: &GpuTensor, dst_scales: &GpuTensor, src: &GpuTensor,
        pos_buf: &DeviceBuffer, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_int8", kernels::KV_CACHE_WRITE_INT8_SRC, "kv_cache_write_int8")?;
        let func = &self.functions["kv_cache_write_int8"];
        let mut dv = dst_vals.buf.as_ptr(); let mut ds = dst_scales.buf.as_ptr();
        let mut s = src.buf.as_ptr(); let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut dv as *mut _ as *mut c_void, &mut ds as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void, &mut p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Attention with INT8 KV (separate scale array).
    pub fn attention_int8_kv(
        &mut self, q: &GpuTensor,
        k_vals: &GpuTensor, k_scales: &GpuTensor,
        v_vals: &GpuTensor, v_scales: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_int8_kv", kernels::ATTENTION_INT8_KV_SRC, "attention_int8_kv")?;
        let func = &self.functions["attention_int8_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut kv_ptr = k_vals.buf.as_ptr(); let mut ks_ptr = k_scales.buf.as_ptr();
        let mut vv_ptr = v_vals.buf.as_ptr(); let mut vs_ptr = v_scales.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr(); let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut kv_ptr as *mut _ as *mut c_void, &mut ks_ptr as *mut _ as *mut c_void,
            &mut vv_ptr as *mut _ as *mut c_void, &mut vs_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void, &mut pos_ptr as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Batched causal attention: all query positions in one launch.
    /// Q: [seq_len × n_heads × head_dim], K/V: [seq_len × n_kv_heads × head_dim].
    pub fn attention_causal_batched(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, out: &GpuTensor,
        seq_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_causal_batched", kernels::ATTENTION_CAUSAL_BATCHED_SRC, "attention_causal_batched")?;
        let func = &self.functions["attention_causal_batched"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut sl = seq_len as i32; let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // Block size: enough threads to cover head_dim and seq_len
        let block_size = 128u32.min((seq_len.max(head_dim) as u32).next_power_of_two());
        // Shared: scores[seq_len] + workspace[block_size]
        let shared_mem = ((seq_len + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, seq_len as u32, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Batched Q8_0 KV cache write: quantize multiple positions in one launch.
    pub fn kv_cache_write_q8_0_batched(
        &mut self, dst: &GpuTensor, src: &GpuTensor, positions: &GpuTensor,
        n_kv_heads: usize, head_dim: usize, batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_q8_0_batched", kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC, "kv_cache_write_q8_0_batched")?;
        let func = &self.functions["kv_cache_write_q8_0_batched"];
        let mut d = dst.buf.as_ptr(); let mut s = src.buf.as_ptr();
        let mut p = positions.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32; let mut bs = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut bs as *mut _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        unsafe { self.hip.launch_kernel(func, [total_blocks, batch_size as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Write KV vector to Q8_0 quantized cache (same format as GGML Q8_0).
    pub fn kv_cache_write_q8_0(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_q8_0", kernels::KV_CACHE_WRITE_Q8_0_SRC, "kv_cache_write_q8_0")?;
        let func = &self.functions["kv_cache_write_q8_0"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let total_blocks = (n_kv_heads * head_dim / 32) as u32;
        let bytes = crate::profile::kv_cache_write_q8_0_bytes(n_kv_heads, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "kv_write", "kv_cache_write_q8_0", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [total_blocks, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Attention with Q8_0 quantized KV cache.
    pub fn attention_q8_0_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_q8_0_kv", kernels::ATTENTION_Q8_0_KV_SRC, "attention_q8_0_kv")?;
        let func = &self.functions["attention_q8_0_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        // Extra shared mem for Q head vector preloaded into shared memory
        let shared_mem = ((seq_len_hint + block_size as usize + head_dim) * 4) as u32;
        let bytes = crate::profile::attention_q8_0_kv_bytes(n_heads, n_kv_heads, head_dim, seq_len_hint);
        let timer = crate::profile::begin_timer(&self.hip, "attention", "attention_q8_0_kv", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Write KV vector to Q8 (int8 symmetric) quantized cache.
    pub fn kv_cache_write_q8(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_q8", kernels::KV_CACHE_WRITE_Q8_SRC, "kv_cache_write_q8")?;
        let func = &self.functions["kv_cache_write_q8"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Attention with Q8 quantized KV cache.
    pub fn attention_q8kv(
        &mut self, q: &GpuTensor, k_cache_q8: &GpuTensor, v_cache_q8: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_q8kv", kernels::ATTENTION_Q8KV_SRC, "attention_q8kv")?;
        let func = &self.functions["attention_q8kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr(); let mut k_ptr = k_cache_q8.buf.as_ptr();
        let mut v_ptr = v_cache_q8.buf.as_ptr(); let mut out_ptr = out.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Write KV vector to quantized HFQ4 cache.
    pub fn kv_cache_write_q4(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_q4", kernels::KV_CACHE_WRITE_Q4_SRC, "kv_cache_write_q4")?;
        let func = &self.functions["kv_cache_write_q4"];
        let mut d = dst.buf.as_ptr();
        let mut s = src.buf.as_ptr();
        let mut p = pos_buf.as_ptr();
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut d as *mut _ as *mut c_void, &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let block = 64u32.min(head_dim as u32);
        let shared = (block * 2 * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [block, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    /// Attention with quantized HFQ4 KV cache — dequantizes K/V on the fly.
    pub fn attention_q4kv(
        &mut self, q: &GpuTensor, k_cache_q4: &GpuTensor, v_cache_q4: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer, seq_len_hint: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_q4kv", kernels::ATTENTION_Q4KV_SRC, "attention_q4kv")?;
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
            &mut q_ptr as *mut _ as *mut c_void, &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void, &mut out_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = (seq_len_hint.max(head_dim) as u32).next_power_of_two().min(256);
        let shared_mem = ((seq_len_hint + block_size as usize) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// GPU-side KV cache write. Copies kv_dim floats from src to dst[pos_buf[0] * kv_dim].
    pub fn kv_cache_write(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos_buf: &DeviceBuffer,
        kv_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write", kernels::KV_CACHE_WRITE_SRC, "kv_cache_write")?;
        let func = &self.functions["kv_cache_write"];

        let mut dst_ptr = dst.buf.as_ptr();
        let mut src_ptr = src.buf.as_ptr();
        let mut pos_ptr = pos_buf.as_ptr();
        let mut kd = kv_dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut dst_ptr as *mut _ as *mut c_void,
            &mut src_ptr as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = (kv_dim as u32 + block - 1) / block;

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

    /// GPU-side top-K + top-P sampling. Returns (token_id, new_rng_state).
    /// Eliminates 600KB logits download per token.
    pub fn sample_top_p(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        repeat_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<(u32, u32)> {
        self.ensure_kernel("sample_top_p", kernels::SAMPLE_TOP_P_SRC, "sample_top_p")?;
        let func = &self.functions["sample_top_p"];

        let mut logits_ptr = logits.buf.as_ptr();
        let mut result_ptr = result_buf.buf.as_ptr();
        let mut repeat_ptr = repeat_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut rw = repeat_window as i32;
        let mut rp = repeat_penalty;

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut logits_ptr as *mut _ as *mut std::ffi::c_void,
            &mut result_ptr as *mut _ as *mut std::ffi::c_void,
            &mut repeat_ptr as *mut _ as *mut std::ffi::c_void,
            &mut vs as *mut _ as *mut std::ffi::c_void,
            &mut temp as *mut _ as *mut std::ffi::c_void,
            &mut tp as *mut _ as *mut std::ffi::c_void,
            &mut rng as *mut _ as *mut std::ffi::c_void,
            &mut rw as *mut _ as *mut std::ffi::c_void,
            &mut rp as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256u32;
        let shared_mem = (256 + 512 + 512 + 1) * 4;

        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )?;
        }

        let mut out = [0u8; 8];
        self.hip.memcpy_dtoh(&mut out, &result_buf.buf)?;
        let token_id = u32::from_ne_bytes([out[0], out[1], out[2], out[3]]);
        let new_rng = u32::from_ne_bytes([out[4], out[5], out[6], out[7]]);
        Ok((token_id, new_rng))
    }

    /// Launch sampling kernel only (no readback). For use during graph capture.
    pub fn sample_top_p_launch(
        &mut self,
        logits: &GpuTensor,
        result_buf: &GpuTensor,
        repeat_buf: &GpuTensor,
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        rng_state: u32,
        repeat_window: usize,
        repeat_penalty: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("sample_top_p", kernels::SAMPLE_TOP_P_SRC, "sample_top_p")?;
        let func = &self.functions["sample_top_p"];

        let mut logits_ptr = logits.buf.as_ptr();
        let mut result_ptr = result_buf.buf.as_ptr();
        let mut repeat_ptr = repeat_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut temp = temperature;
        let mut tp = top_p;
        let mut rng = rng_state;
        let mut rw = repeat_window as i32;
        let mut rp = repeat_penalty;

        let mut params: Vec<*mut std::ffi::c_void> = vec![
            &mut logits_ptr as *mut _ as *mut std::ffi::c_void,
            &mut result_ptr as *mut _ as *mut std::ffi::c_void,
            &mut repeat_ptr as *mut _ as *mut std::ffi::c_void,
            &mut vs as *mut _ as *mut std::ffi::c_void,
            &mut temp as *mut _ as *mut std::ffi::c_void,
            &mut tp as *mut _ as *mut std::ffi::c_void,
            &mut rng as *mut _ as *mut std::ffi::c_void,
            &mut rw as *mut _ as *mut std::ffi::c_void,
            &mut rp as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256u32;
        let shared_mem = (256 + 512 + 512 + 1) * 4;

        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    // ── DeltaNet ops (feature-gated) ─────────────────────────────────────

    /// Partial interleaved RoPE for Qwen3.5 full attention layers.
    #[cfg(feature = "deltanet")]
    pub fn rope_partial_interleaved_f32(
        &mut self, q: &GpuTensor, k: &GpuTensor, pos: i32,
        n_heads_q: usize, n_heads_k: usize, head_dim: usize, n_rot: usize, freq_base: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("rope_partial_interleaved", kernels::ROPE_PARTIAL_INTERLEAVED_SRC, "rope_partial_interleaved_f32")?;
        let func = &self.functions["rope_partial_interleaved_f32"];
        let mut qp = q.buf.as_ptr(); let mut kp = k.buf.as_ptr();
        let mut p = pos; let mut nhq = n_heads_q as i32; let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32; let mut nr = n_rot as i32; let mut fb = freq_base;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void, &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void, &mut fb as *mut _ as *mut c_void,
        ];
        let n_pairs = (n_rot / 2) as u32;
        let block = 32u32.min(n_pairs);
        let grid = (n_pairs + block - 1) / block;
        let bytes = crate::profile::rope_bytes(n_heads_q, n_heads_k, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rope", "rope_partial_interleaved_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Sigmoid activation, in-place.
    #[cfg(feature = "deltanet")]
    /// Repeat-interleave Q and K key heads up to value heads count.
    /// Replaces the per-head memcpy loop in DeltaNet for ratio>1 configs:
    /// `dst[(kh*ratio+r)*hd + d] = src[kh*hd + d]`. Does Q and K together
    /// in one launch. For Qwen3.5 9B (24 layers × 64 D2D each), this saves
    /// ~1500 hipMemcpy calls per forward.
    pub fn repeat_interleave_qk_f32(
        &mut self,
        q_src: &GpuTensor,
        k_src: &GpuTensor,
        q_dst: &GpuTensor,
        k_dst: &GpuTensor,
        n_key_heads: usize,
        ratio: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("repeat_interleave_qk", kernels::REPEAT_INTERLEAVE_QK_SRC, "repeat_interleave_qk_f32")?;
        let func = &self.functions["repeat_interleave_qk_f32"];
        let mut qsp = q_src.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr();
        let mut qdp = q_dst.buf.as_ptr();
        let mut kdp = k_dst.buf.as_ptr();
        let mut nkh = n_key_heads as i32;
        let mut r = ratio as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qsp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void,
            &mut qdp as *mut _ as *mut c_void,
            &mut kdp as *mut _ as *mut c_void,
            &mut nkh as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let total = (n_key_heads * ratio * head_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let bytes = (n_key_heads * head_dim * 4) * 2 // Q/K reads
                  + (n_key_heads * ratio * head_dim * 4) * 2; // Q/K writes
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "repeat_interleave_qk_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Deinterleave: split [A_h0(hd), B_h0(hd), A_h1(hd), B_h1(hd), ...] into A and B.
    /// Replaces per-head memcpy loop (n_heads × 2 ioctls → 1 dispatch).
    pub fn deinterleave_f32(&mut self, interleaved: &GpuTensor, out_a: &GpuTensor, out_b: &GpuTensor,
                            n_heads: usize, head_dim: usize) -> HipResult<()> {
        self.ensure_kernel("deinterleave", kernels::DEINTERLEAVE_SRC, "deinterleave_f32")?;
        let func = &self.functions["deinterleave_f32"];
        let mut inp = interleaved.buf.as_ptr();
        let mut ap = out_a.buf.as_ptr();
        let mut bp = out_b.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut inp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let total = (n_heads * head_dim) as u32;
        let block = 256u32;
        let grid = (total + block - 1) / block;
        let bytes = n_heads * head_dim * 4 * 3; // read interleaved, write both outputs
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "deinterleave_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    #[cfg(feature = "deltanet")]
    pub fn sigmoid_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("sigmoid", kernels::SIGMOID_SRC, "sigmoid_f32")?;
        let func = &self.functions["sigmoid_f32"];
        let mut xp = x.buf.as_ptr();
        let mut n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n as usize);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "sigmoid_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Softplus activation, in-place.
    #[cfg(feature = "deltanet")]
    pub fn softplus_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("softplus", kernels::SOFTPLUS_SRC, "softplus_f32")?;
        let func = &self.functions["softplus_f32"];
        let mut xp = x.buf.as_ptr();
        let mut n = x.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// L2 normalization per head, in-place. One warp per head.
    #[cfg(feature = "deltanet")]
    pub fn l2_norm_f32(&mut self, x: &GpuTensor, n_heads: usize, head_dim: usize, eps: f32) -> HipResult<()> {
        self.ensure_kernel("l2_norm", kernels::L2_NORM_SRC, "l2_norm_f32")?;
        let func = &self.functions["l2_norm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "l2_norm_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused `out *= sigmoid(gate)`. Replaces the sigmoid_f32+mul_f32 pair
    /// in the FA attention epilogue (one launch per full-attention layer).
    pub fn sigmoid_mul_f32(
        &mut self,
        out: &GpuTensor,
        gate: &GpuTensor,
    ) -> HipResult<()> {
        self.ensure_kernel("sigmoid_mul", kernels::SIGMOID_MUL_SRC, "sigmoid_mul_f32")?;
        let func = &self.functions["sigmoid_mul_f32"];
        let mut op = out.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut n = out.numel() as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n as usize) * 3;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "sigmoid_mul_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Top-K=1024 extraction over a logits vector. Populates an 8 KB
    /// buffer with [1024 × u32 indices | 1024 × f32 values]. One
    /// device→host copy pulls the whole thing. The host then runs its
    /// existing top-20 min-tracking loop over the 1024 candidates.
    ///
    /// Previous version used 1 wave of 32 threads and measured at ~1.4 ms
    /// because the compiler couldn't pipeline loads through the branchy
    /// min-tracking path. Current version uses 256 threads (8 waves) on
    /// a single workgroup — roughly 10× faster.
    pub fn topk_logits_f32(
        &mut self,
        logits: &GpuTensor,
        topk_buf: &GpuTensor,   // DType::F32 shape [2048] = 8192 bytes
        vocab_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("topk_logits", kernels::TOPK_LOGITS_SRC, "topk_logits_f32")?;
        let func = &self.functions["topk_logits_f32"];
        let mut lp = logits.buf.as_ptr();
        let mut bp = topk_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
        ];
        let bytes = vocab_size * 4 + 8192;
        let timer = crate::profile::begin_timer(&self.hip, "sampling", "topk_logits_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused sigmoid(dn_beta) + alpha_gate(dn_alpha). Both ops are element-wise
    /// scalar transforms applied to independent buffers of size n_v_heads in the
    /// DeltaNet preamble. Saves one launch per linear-attention layer.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let func = &self.functions["fused_sigmoid_alpha_gate_f32"];
        let mut bp = beta.buf.as_ptr();
        let mut ap = alpha.buf.as_ptr();
        let mut dp = dt_bias.buf.as_ptr();
        let mut lp = a_log.buf.as_ptr();
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut bp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_sigmoid_alpha_gate_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `fused_sigmoid_alpha_gate_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_sigmoid_alpha_gate_f32_batched(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "fused_sigmoid_alpha_gate",
            kernels::FUSED_SIGMOID_ALPHA_GATE_SRC,
            "fused_sigmoid_alpha_gate_f32",
        )?;
        let func = &self.functions["fused_sigmoid_alpha_gate_f32"];
        let mut bp = beta.buf.as_ptr();
        let mut ap = alpha.buf.as_ptr();
        let mut dp = dt_bias.buf.as_ptr();
        let mut lp = a_log.buf.as_ptr();
        let mut nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut bp as *mut _ as *mut c_void,
            &mut ap as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4 * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_sigmoid_alpha_gate_f32_batched", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [grid, batch_size as u32, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused L2-norm(Q) + L2-norm(K) + scale(Q). Replaces three back-to-back
    /// launches in DeltaNet's attention path with one — ~2 launches saved per
    /// linear-attention layer, so on Qwen3.5 (18-32 LA layers) we shave ~36-64
    /// launches per forward.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let func = &self.functions["fused_qk_l2_norm_scale_f32"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut qs = q_scale;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        // Covers both Q and K reads/writes.
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qk_l2_norm_scale_f32", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `fused_qk_l2_norm_scale_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn fused_qk_l2_norm_scale_f32_batched(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "fused_qk_l2_norm_scale",
            kernels::FUSED_QK_L2_NORM_SCALE_SRC,
            "fused_qk_l2_norm_scale_f32",
        )?;
        let func = &self.functions["fused_qk_l2_norm_scale_f32"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut qs = q_scale;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::elementwise1_bytes(n_heads * head_dim) * 2 * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "fused", "fused_qk_l2_norm_scale_f32_batched", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// 1D causal conv (kernel_size=4) for decode. Updates ring buffer state.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_decode_f32(
        &mut self, output: &GpuTensor, input: &GpuTensor, weight: &GpuTensor,
        state: &GpuTensor, n_channels: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("conv1d_decode", kernels::CONV1D_DECODE_SRC, "conv1d_decode_f32")?;
        let func = &self.functions["conv1d_decode_f32"];
        let mut op = output.buf.as_ptr();
        let mut ip = input.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut nc = n_channels as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void, &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Gated output norm: rmsnorm(x) * silu(z). Fused kernel.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32(
        &mut self, x: &GpuTensor, z: &GpuTensor, weight: &GpuTensor,
        out: &GpuTensor, n_heads: usize, head_dim: usize, eps: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let func = &self.functions["gated_norm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut zp = z.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut zp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Batched `gated_norm_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32_batched(
        &mut self,
        x: &GpuTensor, z: &GpuTensor, weight: &GpuTensor, out: &GpuTensor,
        n_heads: usize, head_dim: usize, eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let func = &self.functions["gated_norm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut zp = z.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut zp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim) * batch_size;
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32_batched", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, batch_size as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Gated Delta Net recurrence. S matrix in LDS. Processes all tokens sequentially.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
        gate: &GpuTensor, beta: &GpuTensor,
        state: &GpuTensor, output: &GpuTensor,
        n_tokens: usize, n_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gated_delta_net", kernels::GATED_DELTA_NET_SRC, "gated_delta_net_f32")?;
        let func = &self.functions["gated_delta_net_f32"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        // 32 threads, tiled S in LDS (4KB per tile). Grid: [n_heads, 128/8=16].
        let n_tiles = (128 / 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, n_tiles, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// GDN recurrence with Q8-quantized S state — tiled LDS + warp-shuffle.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q8(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
        gate: &GpuTensor, beta: &GpuTensor,
        s_q8: &GpuTensor, s_scales: &GpuTensor, output: &GpuTensor,
        n_tokens: usize, n_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gated_delta_net_q8", kernels::GATED_DELTA_NET_Q8_SRC, "gated_delta_net_q8")?;
        let func = &self.functions["gated_delta_net_q8"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = s_q8.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let n_tiles = (128 / 4) as u32;
        let bytes = crate::profile::gated_delta_net_q8_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "gated_delta_net_q8", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [n_heads as u32, n_tiles, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// GDN recurrence with Q4-quantized S state.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_q4(
        &mut self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
        gate: &GpuTensor, beta: &GpuTensor,
        s_q4: &GpuTensor, s_scales: &GpuTensor, output: &GpuTensor,
        n_tokens: usize, n_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gated_delta_net_q4", kernels::GATED_DELTA_NET_Q4_SRC, "gated_delta_net_q4")?;
        let func = &self.functions["gated_delta_net_q4"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = s_q4.buf.as_ptr();
        let mut scp = s_scales.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut scp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [128, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Alpha gate compute: alpha[i] = softplus(alpha[i] + dt_bias[i]) * (-exp(a_log[i])).
    /// Replaces 85µs CPU roundtrip with ~3µs GPU kernel.
    #[cfg(feature = "deltanet")]
    pub fn alpha_gate_f32(
        &mut self, alpha: &GpuTensor, dt_bias: &GpuTensor, a_log: &GpuTensor, n: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("alpha_gate", kernels::ALPHA_GATE_SRC, "alpha_gate_f32")?;
        let func = &self.functions["alpha_gate_f32"];
        let mut ap = alpha.buf.as_ptr();
        let mut dp = dt_bias.buf.as_ptr();
        let mut lp = a_log.buf.as_ptr();
        let mut nv = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut dp as *mut _ as *mut c_void,
            &mut lp as *mut _ as *mut c_void, &mut nv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = n * 4 * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "alpha_gate_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Scale vector by constant: x[i] *= scale. Replaces 48µs CPU roundtrip.
    #[cfg(feature = "deltanet")]
    pub fn scale_f32(&mut self, x: &GpuTensor, scale: f32) -> HipResult<()> {
        self.ensure_kernel("scale_f32", kernels::SCALE_F32_SRC, "scale_f32")?;
        let func = &self.functions["scale_f32"];
        let n = x.numel();
        let mut xp = x.buf.as_ptr();
        let mut nv = n as i32;
        let mut sv = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void, &mut nv as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let bytes = crate::profile::elementwise1_bytes(n);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", "scale_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused conv1d (kernel_size=4) + SiLU decode.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_f32(
        &mut self, output: &GpuTensor, input: &GpuTensor, weight: &GpuTensor,
        state: &GpuTensor, n_channels: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("conv1d_silu", kernels::CONV1D_SILU_SRC, "conv1d_silu_f32")?;
        let func = &self.functions["conv1d_silu_f32"];
        let mut op = output.buf.as_ptr();
        let mut ip = input.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut nc = n_channels as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void, &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels);
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_silu_f32", bytes);
        let result = unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params) };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    /// Fused conv1d+SiLU that writes directly to Q/K/V buffers, replacing
    /// the conv1d_silu_f32 + three DtoD split copies in the DeltaNet path.
    /// Channel layout: [Q (k_dim) | K (k_dim) | V (v_dim)] — matches the
    /// wqkv projection output layout.
    #[cfg(feature = "deltanet")]
    pub fn conv1d_silu_split_f32(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
    ) -> HipResult<()> {
        self.conv1d_silu_split_f32_n(q_out, k_out, v_out, input, weight, state, k_dim, v_dim, 1)
    }

    /// Batched conv1d + silu + Q/K/V split. Processes `n_tokens` tokens in
    /// order through the conv, advancing the ring-buffer state N times
    /// (identical state trajectory to calling the single-token variant N
    /// times). `input` / `q_out` / `k_out` / `v_out` are all [N × stride]
    /// row-major.
    pub fn conv1d_silu_split_f32_n(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_tokens: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            "conv1d_silu_split",
            kernels::CONV1D_SILU_SPLIT_SRC,
            "conv1d_silu_split_f32",
        )?;
        let func = &self.functions["conv1d_silu_split_f32"];
        let mut qp = q_out.buf.as_ptr();
        let mut kp = k_out.buf.as_ptr();
        let mut vp = v_out.buf.as_ptr();
        let mut ip = input.buf.as_ptr();
        let mut wp = weight.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut kd = k_dim as i32;
        let mut vd = v_dim as i32;
        let mut nt = n_tokens as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut ip as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut kd as *mut _ as *mut c_void,
            &mut vd as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
        ];
        let n_channels = 2 * k_dim + v_dim;
        let block = 256u32;
        let grid = ((n_channels as u32) + block - 1) / block;
        let bytes = crate::profile::conv1d_silu_bytes(n_channels) * n_tokens;
        let timer = crate::profile::begin_timer(&self.hip, "deltanet", "conv1d_silu_split_f32_n", bytes);
        let result = unsafe {
            self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, self.stream_ref(), &mut params)
        };
        if let Some(t) = timer { t.finish(&self.hip); }
        result
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TurboQuant KV cache kernels
    // ═══════════════════════════════════════════════════════════════════════════

    /// Compile a turbo kernel by prepending the shared preamble (FWHT + centroids).
    fn ensure_turbo_kernel(&mut self, name: &str, body_src: &str, func_name: &str) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }
        // Strip #include "turbo_common.h" since we prepend the source directly
        let stripped = body_src.replace("#include \"turbo_common.h\"", "");
        let full_src = format!("{}\n{}", kernels::TURBO_COMMON_SRC, stripped);
        let obj_path = self.compiler.compile(name, &full_src)?;
        let obj_path_str = obj_path.to_str().unwrap().to_string();
        if !self.modules.contains_key(name) {
            let module = self.hip.module_load(&obj_path_str)?;
            self.modules.insert(name.to_string(), module);
        }
        let module = &self.modules[name];
        let func = self.hip.module_get_function(module, func_name)?;
        self.functions.insert(func_name.to_string(), func);
        Ok(())
    }

    /// Fused K+V write for turbo4 (4-bit). 32 threads, parallel FWHT + quantize.
    pub fn kv_cache_write_turbo4_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        if head_dim == 256 {
            return self.kv_cache_write_turbo4_256_fused(k_dst, v_dst, k_src, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim);
        }
        self.ensure_turbo_kernel("kv_cache_write_turbo4", kernels::KV_CACHE_WRITE_TURBO4_SRC, "kv_cache_write_turbo4")?;
        let func = &self.functions["kv_cache_write_turbo4"];
        let mut kdp = k_dst.buf.as_ptr(); let mut vdp = v_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr(); let mut vsp = v_src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void, &mut vdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void, &mut vsp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    pub fn kv_cache_write_turbo4(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.kv_cache_write_turbo4_fused(dst, dst, src, src, pos_buf, signs1, signs2, n_kv_heads, head_dim)
    }

    /// Fused KV write for turbo_hfq3: writes both K and V in one kernel launch.
    /// 32 threads per kv_head, parallel FWHT + quantize.
    pub fn kv_cache_write_turbo3_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("kv_cache_write_turbo3", kernels::KV_CACHE_WRITE_TURBO3_SRC, "kv_cache_write_turbo3")?;
        let func = &self.functions["kv_cache_write_turbo3"];
        let mut kdp = k_dst.buf.as_ptr(); let mut vdp = v_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr(); let mut vsp = v_src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void, &mut vdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void, &mut vsp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        // shared: x[head_dim] + scratch[32] = (128 + 32) * 4 = 640 bytes
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Legacy single-tensor write (calls fused with same tensor for both).
    pub fn kv_cache_write_turbo3(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        // For backward compat: single-tensor write just uses the fused path once
        self.kv_cache_write_turbo3_fused(dst, dst, src, src, pos_buf, signs1, signs2, n_kv_heads, head_dim)
    }

    /// Fused K+V write for turbo_hfq2 (2-bit). 32 threads parallel FWHT.
    pub fn kv_cache_write_turbo2_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        if head_dim == 256 {
            return self.kv_cache_write_turbo2_256_fused(k_dst, v_dst, k_src, v_src, pos_buf, signs1, signs2, n_kv_heads, head_dim);
        }
        self.ensure_turbo_kernel("kv_cache_write_turbo2", kernels::KV_CACHE_WRITE_TURBO2_SRC, "kv_cache_write_turbo2")?;
        let func = &self.functions["kv_cache_write_turbo2"];
        let mut kdp = k_dst.buf.as_ptr(); let mut vdp = v_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr(); let mut vsp = v_src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void, &mut vdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void, &mut vsp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    pub fn kv_cache_write_turbo2(
        &mut self, dst: &GpuTensor, src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.kv_cache_write_turbo2_fused(dst, dst, src, src, pos_buf, signs1, signs2, n_kv_heads, head_dim)
    }

    /// Attention with turbo_hfq4 KV cache. Includes Q pre-rotation and V output inverse-rotation.
    pub fn attention_turbo4_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        if head_dim == 256 {
            return self.attention_turbo4_kv_256(q, k_cache, v_cache, out, pos_buf, signs1, signs2, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq);
        }
        self.ensure_turbo_kernel("attention_turbo4_kv", kernels::ATTENTION_TURBO4_KV_SRC, "attention_turbo4_kv")?;
        let func = &self.functions["attention_turbo4_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = 32u32;
        let shared_mem = (seq_len_hint * 4) as u32;  // scores only (FWHT is register-only)
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Attention with turbo_hfq3 KV cache.
    pub fn attention_turbo3_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("attention_turbo3_kv", kernels::ATTENTION_TURBO3_KV_SRC, "attention_turbo3_kv")?;
        let func = &self.functions["attention_turbo3_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = 32u32;
        let shared_mem = (seq_len_hint * 4) as u32;  // scores only (FWHT is register-only)
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Attention with turbo_hfq2 KV cache.
    pub fn attention_turbo2_kv(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        if head_dim == 256 {
            return self.attention_turbo2_kv_256(q, k_cache, v_cache, out, pos_buf, signs1, signs2, seq_len_hint, n_heads, n_kv_heads, head_dim, max_seq);
        }
        self.ensure_turbo_kernel("attention_turbo2_kv", kernels::ATTENTION_TURBO2_KV_SRC, "attention_turbo2_kv")?;
        let func = &self.functions["attention_turbo2_kv"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let block_size = 32u32;
        // Only scores[] in shared memory now (Q and V FWHT are register-only)
        let shared_mem = (seq_len_hint * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Compute cross-entropy loss for a single token on GPU.
    /// Returns -log(softmax(logits)[target]). Downloads 4 bytes instead of 600KB.
    pub fn cross_entropy_loss(
        &mut self, logits: &GpuTensor, target_buf: &DeviceBuffer, loss_buf: &GpuTensor,
        vocab_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("cross_entropy_loss", kernels::CROSS_ENTROPY_LOSS_SRC, "cross_entropy_loss")?;
        let func = &self.functions["cross_entropy_loss"];
        let mut lp = logits.buf.as_ptr();
        let mut tp = target_buf.as_ptr();
        let mut op = loss_buf.buf.as_ptr();
        let mut vs = vocab_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void, &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut vs as *mut _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared_mem = (block_size * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [1, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    // ═══ Vision encoder dispatch (GEMM, LayerNorm, GELU, bias-add) ═══

    /// Batched GEMV (GEMM) for F16 weights: Y[M,N] = W_f16[M,K] @ X_f32[N,K]^T
    pub fn gemm_f16(
        &mut self, w: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, n: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemm_f16", kernels::GEMM_F16_SRC, "gemm_f16")?;
        let func = &self.functions["gemm_f16"];
        let mut wp = w.buf.as_ptr();
        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, n as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Batched GEMM for F32: Y[M,N] = A[M,K] @ B[N,K]^T
    pub fn gemm_f32_batched(
        &mut self, a: &GpuTensor, b: &GpuTensor, y: &GpuTensor,
        m: usize, k: usize, n: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemm_f32_batched", kernels::GEMM_F32_SRC, "gemm_f32_batched")?;
        let func = &self.functions["gemm_f32_batched"];
        let mut ap = a.buf.as_ptr();
        let mut bp = b.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        unsafe { self.hip.launch_kernel(func, [m as u32, n as u32, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// LayerNorm with bias (batched): out = gamma * (x - mean) / sqrt(var + eps) + beta
    pub fn layernorm_batched(
        &mut self, x: &GpuTensor, gamma: &GpuTensor, beta: &GpuTensor,
        out: &GpuTensor, batch: usize, n: usize, eps: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("layernorm_f32", kernels::LAYERNORM_SRC, "layernorm_f32")?;
        let func = &self.functions["layernorm_f32"];
        let mut xp = x.buf.as_ptr();
        let mut gp = gamma.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut ep = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void,
        ];
        let block_size = std::cmp::min(256, n) as u32;
        // Round up to power of 2 for reduction
        let block_size = block_size.next_power_of_two();
        let shared_mem = block_size * 4;
        unsafe { self.hip.launch_kernel(func, [batch as u32, 1, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// GELU tanh approximation (in-place capable if x == out)
    pub fn gelu_tanh_f32(&mut self, x: &GpuTensor, out: &GpuTensor, n: usize) -> HipResult<()> {
        self.ensure_kernel("gelu_tanh_f32", kernels::GELU_TANH_SRC, "gelu_tanh_f32")?;
        let func = &self.functions["gelu_tanh_f32"];
        let mut xp = x.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut ni = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
        ];
        let blocks = ((n + 255) / 256) as u32;
        unsafe { self.hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Bias-add: x[batch, n] += bias[n] (in-place, broadcast over batch dim)
    pub fn bias_add_f32(&mut self, x: &GpuTensor, bias: &GpuTensor, batch: usize, n: usize) -> HipResult<()> {
        self.ensure_kernel("bias_add_f32", kernels::BIAS_ADD_SRC, "bias_add_f32")?;
        let func = &self.functions["bias_add_f32"];
        let mut xp = x.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut ni = n as i32;
        let total = (batch * n) as i32;
        let mut ti = total;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
        ];
        let blocks = ((total as usize + 255) / 256) as u32;
        unsafe { self.hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Transpose [rows, cols] → [cols, rows]
    pub fn transpose_f32(
        &mut self, src: &GpuTensor, dst: &GpuTensor, rows: usize, cols: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("transpose_f32", kernels::TRANSPOSE_SRC, "transpose_f32")?;
        let func = &self.functions["transpose_f32"];
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut ri = rows as i32;
        let mut ci = cols as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut ri as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
        ];
        let total = rows * cols;
        let blocks = ((total + 255) / 256) as u32;
        unsafe { self.hip.launch_kernel(func, [blocks, 1, 1], [256, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Fused ViT self-attention: reads QKV [N, 3*hidden], writes out [N, hidden].
    pub fn vit_attention_f32(
        &mut self, qkv: &GpuTensor, out: &GpuTensor,
        n: usize, hidden: usize, num_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("vit_attention_f32", kernels::VIT_ATTENTION_SRC, "vit_attention_f32")?;
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
        unsafe { self.hip.launch_kernel(func, [num_heads as u32, n as u32, 1], [block_size, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    // ═══ Asymmetric turbo: Q8 K + turbo4 V for head_dim=256 ═══

    /// Write V as turbo4 (FWHT-256 + 4-bit quantize). K written separately as Q8.
    pub fn kv_cache_write_turbo4_v256(
        &mut self, v_dst: &GpuTensor, v_src: &GpuTensor, pos_buf: &hip_bridge::DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("kv_cache_write_turbo4_v256", kernels::KV_CACHE_WRITE_TURBO4_V256_SRC, "kv_cache_write_turbo4_v256")?;
        let func = &self.functions["kv_cache_write_turbo4_v256"];
        let mut vd = v_dst.buf.as_ptr();
        let mut vs = v_src.buf.as_ptr();
        let mut pb = pos_buf.as_ptr();
        let mut s1 = signs1.buf.as_ptr();
        let mut s2 = signs2.buf.as_ptr();
        let mut nh = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut vd as *mut _ as *mut c_void, &mut vs as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void, &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let shared = (head_dim + 32) * 4; // x[head_dim] + scratch[32]
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared as u32, self.stream_ref(), &mut params) }
    }

    /// Asymmetric attention: Q8 K cache + turbo4 V cache, head_dim=256.
    pub fn attention_q8k_turbo4v_256(
        &mut self, q: &GpuTensor, k_q8: &GpuTensor, v_t4: &GpuTensor,
        out: &GpuTensor, pos_buf: &hip_bridge::DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("attention_q8k_turbo4v_256", kernels::ATTENTION_Q8K_TURBO4V_256_SRC, "attention_q8k_turbo4v_256")?;
        let func = &self.functions["attention_q8k_turbo4v_256"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k_q8.buf.as_ptr();
        let mut vp = v_t4.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pb = pos_buf.as_ptr();
        let mut s1 = signs1.buf.as_ptr();
        let mut s2 = signs2.buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nk = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void, &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void, &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
        ];
        // 32 threads (warp-cooperative), shared memory: scores[seq_len] only
        let shared = (seq_len * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    // ═══ HF4-V: hipfire-native 4-bit V cache (no FWHT, RDNA-optimized) ═══

    /// Write V to HF4 format: L2-normalize, affine 4-bit quantize, store norm+scale+min+nibbles.
    pub fn kv_cache_write_hf4v_256(
        &mut self, v_dst: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("kv_cache_write_hf4v_256", kernels::KV_CACHE_WRITE_HF4V_256_SRC, "kv_cache_write_hf4v_256")?;
        let func = &self.functions["kv_cache_write_hf4v_256"];
        let mut sp = v_src.buf.as_ptr();
        let mut dp = v_dst.buf.as_ptr();
        let mut pb = pos_buf.as_ptr();
        let mut nk = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as _, &mut dp as *mut _ as _,
            &mut pb as *mut _ as _, &mut nk as *mut _ as _, &mut hd as *mut _ as _,
        ];
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], 0, self.stream_ref(), &mut params) }
    }

    /// Asymmetric attention: Q8 K + HF4 V, head_dim=256.
    /// Dequant: 1 FMA per V element, no FWHT inverse. 32 VGPRs.
    pub fn attention_q8k_hf4v_256(
        &mut self, q: &GpuTensor, k_q8: &GpuTensor, v_hf4: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        seq_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention_q8k_hf4v_256", kernels::ATTENTION_Q8K_HF4V_256_SRC, "attention_q8k_hf4v_256")?;
        let func = &self.functions["attention_q8k_hf4v_256"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k_q8.buf.as_ptr();
        let mut vp = v_hf4.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut pb = pos_buf.as_ptr();
        let mut sl = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nk = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as _, &mut kp as *mut _ as _, &mut vp as *mut _ as _,
            &mut op as *mut _ as _, &mut pb as *mut _ as _,
            &mut sl as *mut _ as _, &mut nh as *mut _ as _, &mut nk as *mut _ as _,
            &mut hd as *mut _ as _, &mut ms as *mut _ as _, &mut scale as *mut _ as _,
        ];
        let shared = (seq_len * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], shared, self.stream_ref(), &mut params) }
    }

    // ═══ Symmetric turbo for head_dim=256 ═══

    /// Fused K+V turbo4 write for head_dim=256. 32 threads × 8 dims.
    pub fn kv_cache_write_turbo4_256_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("kv_cache_write_turbo4_256", kernels::KV_CACHE_WRITE_TURBO4_256_SRC, "kv_cache_write_turbo4_256")?;
        let func = &self.functions["kv_cache_write_turbo4_256"];
        let mut kdp = k_dst.buf.as_ptr(); let mut vdp = v_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr(); let mut vsp = v_src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void, &mut vdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void, &mut vsp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Turbo4 attention for head_dim=256. 32 threads × 8 dims, FWHT-256.
    pub fn attention_turbo4_kv_256(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("attention_turbo4_kv_256", kernels::ATTENTION_TURBO4_KV_256_SRC, "attention_turbo4_kv_256")?;
        let func = &self.functions["attention_turbo4_kv_256"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let shared_mem = (seq_len_hint * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Fused K+V turbo2 write for head_dim=256. 32 threads × 8 dims.
    pub fn kv_cache_write_turbo2_256_fused(
        &mut self, k_dst: &GpuTensor, v_dst: &GpuTensor,
        k_src: &GpuTensor, v_src: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        n_kv_heads: usize, head_dim: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("kv_cache_write_turbo2_256", kernels::KV_CACHE_WRITE_TURBO2_256_SRC, "kv_cache_write_turbo2_256")?;
        let func = &self.functions["kv_cache_write_turbo2_256"];
        let mut kdp = k_dst.buf.as_ptr(); let mut vdp = v_dst.buf.as_ptr();
        let mut ksp = k_src.buf.as_ptr(); let mut vsp = v_src.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nkv = n_kv_heads as i32; let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut kdp as *mut _ as *mut c_void, &mut vdp as *mut _ as *mut c_void,
            &mut ksp as *mut _ as *mut c_void, &mut vsp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
        ];
        let shared_mem = ((head_dim + 32) * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_kv_heads as u32, 1, 1], [32, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    /// Turbo2 attention for head_dim=256. 32 threads × 8 dims, FWHT-256.
    pub fn attention_turbo2_kv_256(
        &mut self, q: &GpuTensor, k_cache: &GpuTensor, v_cache: &GpuTensor,
        out: &GpuTensor, pos_buf: &DeviceBuffer,
        signs1: &GpuTensor, signs2: &GpuTensor,
        seq_len_hint: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_turbo_kernel("attention_turbo2_kv_256", kernels::ATTENTION_TURBO2_KV_256_SRC, "attention_turbo2_kv_256")?;
        let func = &self.functions["attention_turbo2_kv_256"];
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut qp = q.buf.as_ptr(); let mut kp = k_cache.buf.as_ptr();
        let mut vp = v_cache.buf.as_ptr(); let mut op = out.buf.as_ptr();
        let mut pp = pos_buf.as_ptr();
        let mut s1p = signs1.buf.as_ptr(); let mut s2p = signs2.buf.as_ptr();
        let mut nh = n_heads as i32; let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32; let mut ms = max_seq as i32; let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void, &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void, &mut s2p as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let shared_mem = (seq_len_hint * 4) as u32;
        unsafe { self.hip.launch_kernel(func, [n_heads as u32, 1, 1], [32, 1, 1], shared_mem, self.stream_ref(), &mut params) }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch precompilation — compile all kernels a model needs in parallel
    // ═══════════════════════════════════════════════════════════════════════════

    /// Helper: compose turbo kernel source (prepend TURBO_COMMON_SRC).
    fn turbo_source(body_src: &str) -> String {
        let stripped = body_src.replace("#include \"turbo_common.h\"", "");
        format!("{}\n{}", kernels::TURBO_COMMON_SRC, stripped)
    }

    /// Pre-compile all kernels needed for Qwen3.5 inference with a given
    /// weight quantization and KV cache type. Runs hipcc in parallel.
    #[cfg(feature = "deltanet")]
    pub fn precompile_qwen35(&mut self, weight_quant: &str, kv_type: &str, head_dim: usize) -> HipResult<()> {
        // Common kernels for all Qwen3.5 models (DeltaNet + FullAttn shared ops)
        let mut specs: Vec<(&str, String)> = vec![
            ("rmsnorm",                  kernels::RMSNORM_SRC.to_string()),
            ("add_inplace",              kernels::ADD_INPLACE_SRC.to_string()),
            ("mul",                      kernels::MUL_SRC.to_string()),
            ("silu_mul",                 kernels::SILU_MUL_SRC.to_string()),
            ("sigmoid",                  kernels::SIGMOID_SRC.to_string()),
            ("alpha_gate",               kernels::ALPHA_GATE_SRC.to_string()),
            ("conv1d_silu",              kernels::CONV1D_SILU_SRC.to_string()),
            ("l2_norm",                  kernels::L2_NORM_SRC.to_string()),
            ("fused_qk_l2_norm_scale",   kernels::FUSED_QK_L2_NORM_SCALE_SRC.to_string()),
            ("fused_sigmoid_alpha_gate", kernels::FUSED_SIGMOID_ALPHA_GATE_SRC.to_string()),
            ("conv1d_silu_split",        kernels::CONV1D_SILU_SPLIT_SRC.to_string()),
            ("sigmoid_mul",              kernels::SIGMOID_MUL_SRC.to_string()),
            ("topk_logits",              kernels::TOPK_LOGITS_SRC.to_string()),
            ("scale_f32",                kernels::SCALE_F32_SRC.to_string()),
            ("gated_norm",               kernels::GATED_NORM_SRC.to_string()),
            ("rope_partial_interleaved", kernels::ROPE_PARTIAL_INTERLEAVED_SRC.to_string()),
            // FullAttn: Q+gate deinterleave split
            ("deinterleave",             kernels::DEINTERLEAVE_SRC.to_string()),
            // DeltaNet: Q/K repeat-interleave for asymmetric MQA (replaces 64+ memcpy_dtod calls per layer on 4B/9B)
            ("repeat_interleave_qk",     kernels::REPEAT_INTERLEAVE_QK_SRC.to_string()),
        ];

        // Weight-format-specific GEMV
        match weight_quant {
            "hfq6" => {
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
            }
            "hfq4" => {
                let (src, module) = kernels::gemv_hfq4g256_for_arch(&self.arch);
                specs.push((module, src.to_string()));
                specs.push(("gemv_hfq4g256_wide", kernels::GEMV_HFQ4G256_WIDE_SRC.to_string()));
                // Multi-projection fused kernels (LA 4-way, FA 3-way, FFN
                // gate+up). Cross-arch — same 4-accumulator inner loop as
                // gemv_hfq4g256.hip; precompile on every arch that uses
                // the HFQ4 weight path.
                specs.push(("fused_qkvza_hfq4g256",
                            kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string()));
                specs.push(("fused_qkv_hfq4g256",
                            kernels::FUSED_QKV_HFQ4G256_SRC.to_string()));
                specs.push(("fused_gate_up_hfq4g256",
                            kernels::FUSED_GATE_UP_HFQ4G256_SRC.to_string()));
                // gfx1100 multi-row GEMV is opt-in via HIPFIRE_GEMV_ROWS={2,4,8}.
                // Empirically slower than the single-row kernel on gfx1100 at all
                // tested matrix sizes (see commit log / multi-row kernel header),
                // so we only precompile when the env var explicitly requests it.
                if matches!(self.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102")
                    && gemv_rows_override().unwrap_or(1) > 1
                {
                    specs.push(("gemv_hfq4g256_multirow_rdna3",
                                kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC.to_string()));
                    specs.push(("gemv_hfq4g256_residual_multirow_rdna3",
                                kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC.to_string()));
                }
            }
            "q8" => {
                specs.push(("gemv_q8_0", kernels::GEMV_Q8_0_SRC.to_string()));
            }
            _ => {}
        }

        // Embedding kernels — Q8_0 is most common, also cover HFQ4G256/G128 variants
        specs.push(("embedding_q8", kernels::EMBEDDING_Q8_SRC.to_string()));
        specs.push(("embedding_hfq4g256", kernels::EMBEDDING_HFQ4G256_SRC.to_string()));
        specs.push(("embedding_hfq4g128", kernels::EMBEDDING_HFQ4G128_SRC.to_string()));

        // DeltaNet kernels
        specs.push(("gated_delta_net_q8", kernels::GATED_DELTA_NET_Q8_SRC.to_string()));

        // KV cache kernels
        match kv_type {
            "turbo4" if head_dim == 256 => {
                specs.push(("kv_cache_write_turbo4_256", Self::turbo_source(kernels::KV_CACHE_WRITE_TURBO4_256_SRC)));
                specs.push(("attention_turbo4_kv_256",   Self::turbo_source(kernels::ATTENTION_TURBO4_KV_256_SRC)));
            }
            "turbo4" => {
                specs.push(("kv_cache_write_turbo4", Self::turbo_source(kernels::KV_CACHE_WRITE_TURBO4_SRC)));
                specs.push(("attention_turbo4_kv",   Self::turbo_source(kernels::ATTENTION_TURBO4_KV_SRC)));
            }
            "turbo3" => {
                specs.push(("kv_cache_write_turbo3", Self::turbo_source(kernels::KV_CACHE_WRITE_TURBO3_SRC)));
                specs.push(("attention_turbo3_kv",   Self::turbo_source(kernels::ATTENTION_TURBO3_KV_SRC)));
            }
            "turbo2" if head_dim == 256 => {
                specs.push(("kv_cache_write_turbo2_256", Self::turbo_source(kernels::KV_CACHE_WRITE_TURBO2_256_SRC)));
                specs.push(("attention_turbo2_kv_256",   Self::turbo_source(kernels::ATTENTION_TURBO2_KV_256_SRC)));
            }
            "turbo2" => {
                specs.push(("kv_cache_write_turbo2", Self::turbo_source(kernels::KV_CACHE_WRITE_TURBO2_SRC)));
                specs.push(("attention_turbo2_kv",   Self::turbo_source(kernels::ATTENTION_TURBO2_KV_SRC)));
            }
            "q8" | _ => {
                specs.push(("kv_cache_write_q8_0", kernels::KV_CACHE_WRITE_Q8_0_SRC.to_string()));
                specs.push(("attention_q8_0_kv",   kernels::ATTENTION_Q8_0_KV_SRC.to_string()));
            }
        }

        // Convert to (&str, &str) for the batch API
        let batch: Vec<(&str, &str)> = specs.iter()
            .map(|(name, src)| (*name, src.as_str()))
            .collect();
        self.compiler.compile_batch(&batch)?;

        // Now load all modules + functions sequentially (GPU API)
        for (name, src) in &specs {
            // Map module name → function name(s). Most modules expose exactly one
            // function; multirow modules expose three (r2/r4/r8).
            let func_names: Vec<&str> = match *name {
                "rmsnorm" => vec!["rmsnorm_f32"],
                "add_inplace" => vec!["add_inplace_f32"],
                "mul" => vec!["mul_f32"],
                "silu_mul" => vec!["silu_mul_f32"],
                "sigmoid" => vec!["sigmoid_f32"],
                "alpha_gate" => vec!["alpha_gate_f32"],
                "conv1d_silu" => vec!["conv1d_silu_f32"],
                "l2_norm" => vec!["l2_norm_f32"],
                "fused_qk_l2_norm_scale" => vec!["fused_qk_l2_norm_scale_f32"],
                "fused_sigmoid_alpha_gate" => vec!["fused_sigmoid_alpha_gate_f32"],
                "conv1d_silu_split" => vec!["conv1d_silu_split_f32"],
                "sigmoid_mul" => vec!["sigmoid_mul_f32"],
                "topk_logits"  => vec!["topk_logits_f32"],
                "scale_f32" => vec!["scale_f32"],
                "gated_norm" => vec!["gated_norm_f32"],
                "rope_partial_interleaved" => vec!["rope_partial_interleaved_f32"],
                "deinterleave" => vec!["deinterleave_f32"],
                "repeat_interleave_qk" => vec!["repeat_interleave_qk_f32"],
                "gated_delta_net_q8" => vec!["gated_delta_net_q8"],
                // RDNA2 variant module names → common function symbol
                n if n.starts_with("gemv_hfq4g256_rdna2") => vec!["gemv_hfq4g256"],
                // Multi-row RDNA3 modules expose three entry points per .hsaco
                "gemv_hfq4g256_multirow_rdna3" => vec![
                    "gemv_hfq4g256_multirow_r2",
                    "gemv_hfq4g256_multirow_r4",
                    "gemv_hfq4g256_multirow_r8",
                ],
                "gemv_hfq4g256_residual_multirow_rdna3" => vec![
                    "gemv_hfq4g256_residual_multirow_r2",
                    "gemv_hfq4g256_residual_multirow_r4",
                    "gemv_hfq4g256_residual_multirow_r8",
                ],
                other => vec![other],
            };
            // Compile and ensure the module is loaded once.
            let obj_path = self.compiler.compile(name, src)?;
            let obj_path_str = obj_path.to_str().unwrap().to_string();
            if !self.modules.contains_key(*name) {
                let module = self.hip.module_load(&obj_path_str)?;
                self.modules.insert(name.to_string(), module);
            }
            let module = &self.modules[*name];
            for func_name in &func_names {
                if self.functions.contains_key(*func_name) {
                    continue;
                }
                let func = self.hip.module_get_function(module, func_name)?;
                self.functions.insert(func_name.to_string(), func);
            }
        }

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Kernel profiler
    // ═══════════════════════════════════════════════════════════════════════════

    /// Profile all compiled kernels: hardware caps + ISA metadata + occupancy.
    pub fn profile(&self) -> (crate::profiler::GpuCapability, Vec<crate::profiler::KernelProfile>) {
        let vram = self.hip.get_vram_info().map(|(_, t)| t as u64).unwrap_or(0);
        crate::profiler::profile_kernels(&self.arch, vram, self.compiler.compiled_kernels())
    }
}
