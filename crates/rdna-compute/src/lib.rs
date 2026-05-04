//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

mod compiler;
mod dispatch;
pub mod iu4_calibration;
mod kernels;
pub mod pool;
pub mod profile;
pub mod profiler;

pub use compiler::KernelCompiler;
pub use dispatch::{DType, Gpu, GpuTensor};
pub use iu4_calibration::{GpuIu4Calibration, GpuIu4CalSite};
pub use kernels::GEMV_SRC;
