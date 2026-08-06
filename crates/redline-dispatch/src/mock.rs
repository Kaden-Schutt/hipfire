//! Deterministic non-GPU backend for plan and integration tests.

use crate::{
    BeginReplay, DispatchBackend, DispatchRequest, EndReplay, LaneId, NodeId, ReplayMode,
    ReplayToken,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MockSignal(pub u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MockCompletion {
    pub token: ReplayToken,
    pub mode: ReplayMode,
    pub terminal_signals: Vec<MockSignal>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MockCommand {
    Begin {
        token: ReplayToken,
        mode: ReplayMode,
        lane_count: usize,
    },
    Dispatch {
        token: ReplayToken,
        node: NodeId,
        lane: LaneId,
        kernel: String,
        dependencies: Vec<MockSignal>,
        signal: MockSignal,
    },
    End(MockCompletion),
}

#[derive(Default)]
pub struct MockBackend {
    commands: Vec<MockCommand>,
    next_signal: u64,
    active: Option<ReplayToken>,
}

impl MockBackend {
    pub fn commands(&self) -> &[MockCommand] {
        &self.commands
    }
}

impl DispatchBackend for MockBackend {
    type Signal = MockSignal;
    type Completion = MockCompletion;
    type Error = MockError;

    fn begin_replay(&mut self, replay: BeginReplay) -> Result<(), Self::Error> {
        if let Some(active) = self.active {
            return Err(MockError::ReplayAlreadyActive(active));
        }
        self.active = Some(replay.token);
        self.commands.push(MockCommand::Begin {
            token: replay.token,
            mode: replay.mode,
            lane_count: replay.lane_count,
        });
        Ok(())
    }

    fn dispatch(
        &mut self,
        request: DispatchRequest<'_, Self::Signal>,
    ) -> Result<Self::Signal, Self::Error> {
        if self.active != Some(request.token) {
            return Err(MockError::ReplayNotActive(request.token));
        }
        let signal = MockSignal(self.next_signal);
        self.next_signal += 1;
        self.commands.push(MockCommand::Dispatch {
            token: request.token,
            node: request.node,
            lane: request.lane,
            kernel: request.launch.kernel().to_owned(),
            dependencies: request.dependency_signals.to_vec(),
            signal,
        });
        Ok(signal)
    }

    fn end_replay(
        &mut self,
        replay: EndReplay<'_, Self::Signal>,
    ) -> Result<Self::Completion, Self::Error> {
        if self.active != Some(replay.token) {
            return Err(MockError::ReplayNotActive(replay.token));
        }
        self.active = None;
        let completion = MockCompletion {
            token: replay.token,
            mode: replay.mode,
            terminal_signals: replay.terminal_signals.to_vec(),
        };
        self.commands.push(MockCommand::End(completion.clone()));
        Ok(completion)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MockError {
    #[error("replay {0:?} is already active")]
    ReplayAlreadyActive(ReplayToken),
    #[error("replay {0:?} is not active")]
    ReplayNotActive(ReplayToken),
}
