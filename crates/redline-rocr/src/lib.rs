//! Low-level public ROCr/HSA runtime and AQL packet mechanics for Redline.
//!
//! This crate owns the dynamically loaded ROCr ABI, public-HSA resource
//! lifetimes, architected 64-byte AQL packet encoding, and release-ordered
//! queue publication. It deliberately contains no dispatch-DAG scheduling or
//! backend-selection policy; those remain in `redline-dispatch`.
//!
//! # Provenance
//!
//! ABI values and layouts are transcribed from the MIT-licensed ROCm 7.2
//! public headers `/opt/rocm/include/hsa/{hsa.h,hsa_ext_amd.h}` and were
//! cross-checked against `rocm-systems` commit
//! `c0430a50286200ab0562f4733445cdee6e48d416`. Queue publication and resource
//! lifetime behavior are independently implemented against that public API;
//! the ROCm source tree is not vendored.

use std::ffi::c_void;
use std::fmt;
use std::ptr;
use std::sync::Arc;

use libloading::Library;

#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("the manual ROCr ABI currently supports only 64-bit little-endian hosts");

pub mod abi;
#[doc(hidden)]
pub mod packet;
mod pm4;
mod pm4_gfx10;
mod runtime;

pub use abi::{MissingSymbol, Symbols};
pub use packet::{
    BARRIER_DEPENDENCY_CAPACITY, FenceScope, HeaderPolicy, KernelMetadata, LaunchGeometry,
    PacketError,
};
pub use pm4::{Gfx12Pm4CommandBuffer, Pm4BuildError, Pm4DispatchSpanAttribution};
pub use pm4_gfx10::{
    Gfx10DispatchInitiatorPolicy, Gfx10KernelImage, Gfx10Pm4BuildError, Gfx10Pm4CommandBuffer,
    Gfx10SetShRegRecord, Gfx11ComputeResourceLimitsPolicy, Gfx11DispatchInterleave,
    Gfx11Pm4CommandBuffer,
};
pub use runtime::DEFAULT_WAIT_TIMEOUT;
pub use runtime::{
    AqlQueue, CompletionSignal, Executable, GpuDevice, GpuSelector, KernargBuffer, KernargPool,
    Kernel, KernelPm4Metadata, PciBusId, PciBusIdParseError, QueueDepthReport, QueueDepthSample,
    QueueDepthStats, QueueSet, Runtime, RuntimeError,
};

/// Load the installed public ROCr runtime without a link-time ROCm dependency.
pub fn load_symbols() -> Result<Arc<Symbols>, LoadError> {
    let candidates =
        hipfire_config::rocm::library_candidates(&["libhsa-runtime64.so", "libhsa-runtime64.so.1"]);
    let mut failures = Vec::new();
    let library = candidates
        .iter()
        .find_map(|candidate| {
            // SAFETY: loading the installed ROCr runtime is the purpose of this
            // crate. `Symbols` retains the successful library mapping.
            match unsafe { Library::new(candidate) } {
                Ok(library) => Some(Arc::new(library)),
                Err(error) => {
                    failures.push(format!("{candidate}: {error}"));
                    None
                }
            }
        })
        .ok_or_else(|| LoadError::Library {
            candidates: candidates.join(", "),
            detail: format!(
                "{}\n{}",
                failures.join("; "),
                hipfire_config::rocm::resolution_failure(
                    "the HSA runtime (libhsa-runtime64.so)",
                    &candidates,
                )
            ),
        })?;
    let keepalive: Arc<dyn Send + Sync> = library.clone();
    // SAFETY: every lookup is resolved from the public ROCr library retained by
    // `keepalive`; `Symbols::load` assigns each name its public-header ABI.
    unsafe {
        Symbols::load(keepalive, |name| {
            library
                .get::<*const c_void>(name.to_bytes_with_nul())
                .map_or(ptr::null(), |symbol| *symbol)
        })
        .map_err(LoadError::Symbol)
    }
}

#[derive(Debug)]
pub enum LoadError {
    Library { candidates: String, detail: String },
    Symbol(MissingSymbol),
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Library { candidates, detail } => write!(
                formatter,
                "could not load ROCr; tried {candidates}: {detail}"
            ),
            Self::Symbol(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for LoadError {}
