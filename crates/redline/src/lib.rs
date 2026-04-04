//! Redline — direct-KMD GPU compute engine for AMD RDNA GPUs.
//!
//! Bypasses the HIP runtime entirely. Talks to /dev/dri/renderD128 via
//! libdrm_amdgpu.so (thin DRM userspace library, available on any system
//! with the amdgpu kernel driver — no ROCm needed).
//!
//! # Architecture
//!
//! ```text
//! hipfire engine
//! ├── hip-bridge (dlopen libamdhip64.so — requires ROCm)
//! └── redline    (dlopen libdrm_amdgpu.so — requires only amdgpu driver)
//!                 ├── device.rs    — GPU init, info queries
//!                 ├── memory.rs    — BO alloc, VA mapping, CPU mapping
//!                 ├── queue.rs     — compute queue, PM4 command buffers
//!                 ├── dispatch.rs  — kernel loading (.hsaco) + launch
//!                 └── sync.rs      — fences, synchronization
//! ```
//!
//! # What we replace
//!
//! | HIP function              | Redline equivalent            | DRM ioctl             |
//! |---------------------------|-------------------------------|-----------------------|
//! | hipInit + hipSetDevice    | Device::open()                | DRM_AMDGPU_INFO       |
//! | hipMalloc                 | Memory::alloc_vram()          | GEM_CREATE + GEM_VA   |
//! | hipFree                   | Memory::free()                | GEM_VA + close(fd)    |
//! | hipMemcpy (H→D)          | Memory::upload()              | GEM_MMAP + memcpy     |
//! | hipMemcpy (D→H)          | Memory::download()            | GEM_MMAP + memcpy     |
//! | hipMemcpy (D→D)          | Queue::copy()                 | DMA PM4 packets       |
//! | hipModuleLoad             | Dispatch::load_module()       | parse .hsaco ELF      |
//! | hipModuleLaunchKernel     | Dispatch::launch()            | CS submit PM4         |
//! | hipStreamSynchronize      | Sync::wait_fence()            | WAIT_CS               |
//! | hipDeviceSynchronize      | Sync::drain()                 | WAIT_CS (all)         |
//! | hipMemGetInfo             | Device::vram_info()           | DRM_AMDGPU_INFO       |

pub mod drm;
pub mod device;

/// Redline error type.
#[derive(Debug)]
pub struct RedlineError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for RedlineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "redline error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for RedlineError {}

pub type Result<T> = std::result::Result<T, RedlineError>;
