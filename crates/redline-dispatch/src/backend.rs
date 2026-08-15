use std::num::NonZeroUsize;

use crate::{Access, KernelLaunch, LaneId, NodeId};

/// Replay semantics selected when a plan is compiled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayMode {
    /// Complete every terminal signal before returning from `end_replay`.
    TokenLatency,
    /// Permit whole-token overlap, while limiting the number of outstanding
    /// tokens. A backend must throttle at `begin_replay` when the limit is hit.
    Throughput { max_tokens_in_flight: NonZeroUsize },
}

impl ReplayMode {
    pub fn throughput(max_tokens_in_flight: usize) -> Option<Self> {
        NonZeroUsize::new(max_tokens_in_flight).map(|max_tokens_in_flight| Self::Throughput {
            max_tokens_in_flight,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReplayToken(pub u64);

#[derive(Clone, Copy, Debug)]
pub struct BeginReplay {
    pub token: ReplayToken,
    pub mode: ReplayMode,
    pub lane_count: usize,
}

#[derive(Debug)]
pub struct DispatchRequest<'a, Signal> {
    pub token: ReplayToken,
    pub node: NodeId,
    pub lane: LaneId,
    pub launch: &'a KernelLaunch,
    pub accesses: &'a [Access],
    pub dependency_signals: &'a [Signal],
}

#[derive(Debug)]
pub struct EndReplay<'a, Signal> {
    pub token: ReplayToken,
    pub mode: ReplayMode,
    pub terminal_signals: &'a [Signal],
}

/// Concrete queue implementation driven by a validated [`crate::CompiledPlan`].
///
/// Implementations must preserve FIFO order within each [`LaneId`] and honor
/// every `dependency_signal` before executing the submitted dispatch. For
/// [`ReplayMode::TokenLatency`], `end_replay` must not return until all terminal
/// signals complete. Throughput mode may return an asynchronous completion, but
/// must enforce its `max_tokens_in_flight` bound.
pub trait DispatchBackend {
    type Signal: Clone;
    type Completion;
    type Error;

    fn begin_replay(&mut self, replay: BeginReplay) -> Result<(), Self::Error>;

    fn dispatch(
        &mut self,
        request: DispatchRequest<'_, Self::Signal>,
    ) -> Result<Self::Signal, Self::Error>;

    fn end_replay(
        &mut self,
        replay: EndReplay<'_, Self::Signal>,
    ) -> Result<Self::Completion, Self::Error>;
}
