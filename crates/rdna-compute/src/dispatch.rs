//! High-level GPU dispatch interface.
//! Manages compiled kernels, provides typed tensor operations.

use crate::compiler::KernelCompiler;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult, HipRuntime};
use std::collections::HashMap;
use std::ffi::c_void;

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
    Raw,       // raw bytes, no element interpretation
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Q4F16G64 | DType::Q4F16G32 | DType::Raw => 1, // byte-level
        }
    }
}

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
pub struct Gpu {
    pub hip: HipRuntime,
    compiler: KernelCompiler,
    modules: HashMap<String, hip_bridge::Module>,
    functions: HashMap<String, hip_bridge::Function>,
    pool: crate::pool::GpuPool,
}

impl Gpu {
    pub fn init() -> HipResult<Self> {
        let hip = HipRuntime::load()?;
        let count = hip.device_count()?;
        if count == 0 {
            return Err(hip_bridge::HipError::new(0, "no GPU devices found"));
        }
        hip.set_device(0)?;

        let compiler = KernelCompiler::new("gfx1010")?;

        Ok(Self {
            hip,
            compiler,
            modules: HashMap::new(),
            functions: HashMap::new(),
            pool: crate::pool::GpuPool::new(),
        })
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
                None,
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
                None,
                &mut params,
            )
        }
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
        self.ensure_kernel("gemv_q8_0", kernels::GEMV_Q8_0_SRC, "gemv_q8_0")?;
        let func = &self.functions["gemv_q8_0"];

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

        let block_size = 32u32; // single warp — warp shuffle only
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                None,
                &mut params,
            )
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
                None,
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
                None,
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
                None,
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
                None,
                &mut params,
            )
        }
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

        unsafe {
            self.hip.launch_kernel(
                func,
                [batch as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
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
        unsafe {
            self.hip.launch_kernel(func, [batch as u32, 1, 1], [block_size, 1, 1], shared_mem, None, &mut params)
        }
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
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
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
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
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
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
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
                None,
                &mut params,
            )
        }
    }

    /// GPU-side RoPE (rotary positional embedding) applied in-place to Q and K.
    pub fn rope_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos: usize,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("rope", kernels::ROPE_SRC, "rope_f32")?;
        let func = &self.functions["rope_f32"];

        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k.buf.as_ptr();
        let mut pos_val = pos as i32;
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut fb = freq_base;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut pos_val as *mut _ as *mut c_void,
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
                None,
                &mut params,
            )
        }
    }

    /// GPU-side GQA attention.
    pub fn attention_f32(
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
        self.ensure_kernel("attention", kernels::ATTENTION_SRC, "attention_f32")?;
        let func = &self.functions["attention_f32"];

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut seq_val = seq_len as i32;
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
            &mut seq_val as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(seq_len.max(head_dim) as u32);
        let shared_mem = ((seq_len + block_size as usize) * 4) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }
}
