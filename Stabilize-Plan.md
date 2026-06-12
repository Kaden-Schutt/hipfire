# Plan To Clear Stabilize-First Blockers

  ## Summary

  Clear the remaining extraction blockers in dependency order, keeping Qwen35 V1 behavior stable while preparing the modular runtime split. Do not split
  qwen35.rs, dispatch.rs, or daemon state broadly until these blockers have focused parity and smoke coverage.

  Execution order:

  1. Native grouped-MoE decode chunks.
  2. Fused prefill interior checkpoint hooks.
  3. Generic sequence-state descriptors.
  4. MTP/DFlash rollback parity.
  5. Backend module contracts for CPU/GPU/NPU substitution.

  ## Key Changes

  - Add native grouped-MoE routed decode chunks under the existing Qwen35 decode-batch path. Keep serial_reference as oracle, keep auto conservative,
    and only promote grouped-MoE native decode after parity and latency gates pass.

  - Add backend-neutral prefill checkpoint hooks so fused prefill can emit semantic-boundary checkpoints, not only final checkpoints. The hook should
    carry: session id, logical token position, boundary kind, prefix hash input, and state handle.
    Status: the daemon now has a typed Qwen35 prefill checkpoint hook carrying that metadata. Final prefill checkpoints, serial semantic-boundary
    checkpoints, and synchronized multi-session fused dense/grouped semantic-boundary checkpoints can all emit attachable resident snapshots.
    Boundary layouts that would require a single-session fused interior segment fall back to serial_reference.

  - Replace Qwen35-only wrapped state assumptions with generic state descriptors:
      - StatePageKind: KV, DeltaNet, logits snapshot, backend-private.
      - StatePageDescriptor: kind, bytes, shape metadata, residency, owner session.
      - SequenceStateHandle: active session or checkpoint identity.
      - Keep current Qwen35 structs as the backing implementation until descriptors are proven.
    Status: model-worker descriptors now use typed page kinds, include shape metadata, and carry a sequence-state handle identity while still being
    backed by the current Qwen35 resident session map. Allocator-owned generic pages remain out of scope for this stabilize-first slice.

  - Add MTP/DFlash rollback parity before verify batching. Required invariant: after save, speculative advance, reject/restore, and AR replay, logits
    and next token match the serial reference within explicit tolerance. Keep multi-request verify batching disabled until this passes.
    Status: Qwen35 speculative decode now has a named rollback-admission guard for accept/reject commit shapes and AR replay boundary alignment.
    The daemon consults it before advancing DFlash state, and multi-request verify batching remains explicitly disabled pending full KV/DeltaNet/logits
    parity evidence.

  - Define the first backend module contract for one Qwen35 dense FFN/SwiGLU/down segment:
      - CPU backend is oracle.
      - GPU backend is current production path.
      - NPU backend remains opt-in.
      - Evidence records selected backend, module id, drift, and fallback reason.

  ## Interfaces

  - Add an internal state-arena API before creating new crates:
      - reserve_session_state(worker, state_kinds, max_seq)
      - attach_checkpoint(worker, checkpoint, prefix_hash)
      - fork_checkpoint(session, boundary)
      - release_state(handle)
      - describe_state(handle) -> Vec<StatePageDescriptor>

  - Add a generation module invocation API:
      - ModuleKind
      - TensorContract
      - StateContract
      - BackendPreference
      - ModuleInvocation
      - ModuleOutput

  - Add scheduler-visible batch metadata:
      - selected backend,
      - batch size,
      - compatible state kinds,
      - cached prefix tokens,
      - fallback reason.

  ## Test Plan

  - Preserve existing gates:
      - ./tests/smoke-generate-batch-prefill.sh
      - ./tests/smoke-server-prefill-batch.sh
      - ./tests/smoke-server-decode-batch.sh
      - prefix checkpoint smokes
      - focused cargo test for daemon/session/scheduler modules.

  - Add grouped-MoE decode tests:
      - serial vs native routed grouped-MoE decode parity,
      - batch sizes 2/4/8,
      - forced multi-chunk mode,
      - fallback when native grouped route is unsupported.

  - Add state-arena tests:
      - descriptor accounting,
      - attach/fork/release lifecycle,
      - stale handle rejection,
      - checkpoint cap eviction,
      - Qwen35 wrapped-state compatibility.

  - Add fused checkpoint tests:
      - final checkpoint still works,
      - interior semantic-boundary checkpoint attaches,
      - boundary reuse produces same continuation as full prefill.

  - Add MTP/DFlash rollback tests:
      - accept path unchanged,
      - reject path restores KV + DeltaNet + logits state,
      - AR replay parity after restore,
      - verify batching remains disabled until tests pass.

  ## Assumptions

  - Target architecture is Qwen35 first.
  - Streaming, tools/images, PFlash, CASK, multi-GPU batching, and cross-session MTP/DFlash verify batching remain out of scope.
  - Public OpenAI-compatible API shape does not change.
  - Existing daemon protocol can remain while the modular libraries are introduced behind it.
  - New crate extraction happens after these APIs pass in place.
