//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

mod compiler;
mod dispatch;
mod kernels;
pub mod pool;
pub mod profile;
pub mod profiler;

pub use compiler::KernelCompiler;
pub use dispatch::{gemv_dp4a_enabled, has_wmma_f16, DType, Gpu, GpuTensor};
pub use kernels::GEMV_SRC;
