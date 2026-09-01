//! Direct public ROCr/HSA AQL record/replay.
//!
//! Low-level ABI, resource ownership, packet encoding, and queue publication
//! live in `redline-rocr`. This module retains dispatch graph semantics and
//! compatibility re-exports for existing `redline_dispatch::aql` callers.

mod generic;
mod queue_policy;
mod replay;

pub use generic::{AqlKernelCatalog, GenericAqlError, PreparedAqlPlan};
pub use queue_policy::{QueuePolicy, QueuePolicyParseError};
pub use redline_rocr::abi;
pub use redline_rocr::{
    AqlQueue, BARRIER_DEPENDENCY_CAPACITY, CompletionSignal, Executable, FenceScope,
    Gfx10DispatchInitiatorPolicy, Gfx10KernelImage, Gfx10Pm4BuildError, Gfx10Pm4CommandBuffer,
    Gfx10SetShRegRecord, Gfx11ComputeResourceLimitsPolicy, Gfx11DispatchInterleave,
    Gfx11Pm4CommandBuffer, Gfx12Pm4CommandBuffer, GpuDevice, GpuSelector, HeaderPolicy,
    KernargBuffer, KernargPool, Kernel, KernelMetadata, KernelPm4Metadata, LaunchGeometry,
    LoadError, MissingSymbol, PacketError, PciBusId, PciBusIdParseError, Pm4BuildError,
    QueueDepthReport, QueueDepthSample, QueueDepthStats, QueueSet, Runtime, RuntimeError, Symbols,
    load_symbols,
};
pub use replay::{
    BatchFencePolicy, GpuBatchTiming, GpuMultiQueueTiming, PhasedMultiQueuePm4Ib, Quiescence,
    RecordedDispatch, RecordedGraph, ReplayError, ReplaySubmission, SingleQueueBatchGraph,
    SingleQueueBatchSubmission, SingleQueuePm4Ib, TwoQueueBatchSubmission, TwoQueuePhase,
    TwoQueuePhasedGraph, TwoQueueSerializedBatchGraph, TwoQueueSubmission,
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn live_runtime_and_two_queue_smoke_when_enabled() {
        if std::env::var_os("REDLINE_TEST_AQL").is_none() {
            return;
        }
        let runtime = Runtime::initialize(load_symbols().unwrap()).unwrap();
        let device = runtime.select_gpu(GpuSelector::Ordinal(0)).unwrap();
        let queue_size = *device.queue_size_range().start();
        let queues = QueueSet::create(&device, 2, queue_size).unwrap();
        let ids = queues.queue_ids().collect::<Vec<_>>();
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn live_code_object_loader_smoke_when_enabled() {
        let Some(path) = std::env::var_os("REDLINE_TEST_AQL_HSACO") else {
            return;
        };
        let symbol = std::env::var("REDLINE_TEST_AQL_KERNEL")
            .unwrap_or_else(|_| "mq4gemv_exact.kd".to_owned());
        let code_object: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let runtime = Runtime::initialize(load_symbols().unwrap()).unwrap();
        let device = runtime.select_gpu(GpuSelector::Ordinal(0)).unwrap();
        let executable = Executable::load(&device, code_object).unwrap();
        let kernel = executable.kernel(&symbol).unwrap();
        let pool = KernargPool::discover(&device).unwrap();
        let kernarg = pool.allocate_for(kernel.metadata()).unwrap();

        assert_ne!(kernel.metadata().kernel_object, 0);
        assert_eq!(
            kernarg.len(),
            kernel.metadata().kernarg_segment_size as usize
        );
    }
}
