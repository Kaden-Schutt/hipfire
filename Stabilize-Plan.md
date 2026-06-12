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
    Status: explicit Qwen35-MoE `fused_grouped_moe_layer_chunked` decode now advances multi-session chunks through the native grouped-MoE row worker.
    `serial_reference` remains the oracle, internal parity can compare native and serial state, forced multi-chunk mode is covered, and real A3B
    server parity/latency smokes pass for B=2/4/8 with a forced chunk cap. `auto` now promotes grouped-MoE native decode only for B>=4 after
    capability and resident-state checks; B=2 remains serial because the latency gate is not clean enough. The decode-batch smoke now has an opt-in
    grouped-MoE parity matrix mode covering B=2/4/8 with chunk size 2 by default; B=4 and B=8 force multi-chunk native grouped-MoE decode while
    comparing responses against `serial_reference`.
    The same matrix now reports serial/native latency for the promotion gate and labels whether internal parity instrumentation is active.
    Instrumented latency is not used for promotion. On gfx1151 with `qwen3.6-35b-a3b-mq4.hfq` and internal parity off, three fresh-server matrix
    samples were: B=2 `15.449/11.528`, `15.664/10.133`, `15.382/15.918` ms serial/native; B=4 `40.927/31.047`,
    `41.251/39.145`, `40.858/38.763`; B=8 `91.668/81.663`, `91.946/82.022`, `91.586/82.485`. Native is promising but still
    not a clean promotion gate because small-B has one regression sample and B=4 sometimes sits within the expected noise band.
    Rechecking the promotion shape with chunk size 8 shows B=4/B=8 as stable one-chunk wins while B=2 remains noisy, so the scoped auto gate is
    B>=4: B=4 `40.955/12.742`, `40.588/17.593`, `40.988/14.446`; B=8 `91.834/16.616`, `91.369/17.535`,
    `91.879/14.655`; B=2 `15.359/13.223`, `16.382/15.607`, `15.576/16.404`.
    The full prefill smoke now passes on this host with canonical dense BF16 plus grouped-MoE MQ4 artifacts:
    `MODEL=$HOME/.hipfire/models/qwen3.5-0.8b-bf16.hfq MOE_MODEL=$HOME/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq
    UNSUPPORTED_MODEL=$HOME/.hipfire/models/llama-3.2-1b-instruct.mq4.hfq ./tests/smoke-generate-batch-prefill.sh`.
    The older dense MQ4 local spelling (`qwen3.5-0.8b.mq4.hfq`) remains unsuitable for this fused dense prefill smoke because its Q8_0 lm_head is
    intentionally rejected by the dense full-precision final-logits path.

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
    backed by the current Qwen35 resident session map. Runtime-view coverage now explicitly includes the KV, DeltaNet, and logits descriptor triplet
    needed to audit rollback state. Allocator-owned generic pages remain out of scope for this stabilize-first slice.

  - Add MTP/DFlash rollback parity before verify batching. Required invariant: after save, speculative advance, reject/restore, and AR replay, logits
    and next token match the serial reference within explicit tolerance. Keep multi-request verify batching disabled until this passes.
    Status: Qwen35 speculative decode now has a named rollback-admission guard for accept/reject commit shapes and AR replay boundary alignment.
    The daemon consults a centralized post-step decision helper before advancing DFlash state, decode batching refuses DFlash-loaded models at the
    runtime-surface boundary, and the DFlash coherence gate can now run same-prompt AR-token parity while reporting per-cycle rollback-admission and
    DeltaNet replay-path counts. First opt-in AR parity run passed the code prompt but failed the prose prompt at token index 62, so multi-request
    verify batching remains explicitly disabled pending full KV/DeltaNet/logits parity evidence. Cycle tracing localizes the prose mismatch to DFlash
    cycle 30 at position 120: `accepted=0`, `seed=27786`, `bonus=6511`, GDN tape replay, emitted range `62..63`; AR expected token `57874`.
    Verifier-row tracing shows DFlash row 0 ranks `6511` at logit `18.169245` above AR's `57874` at `18.144331`, so the next comparison is state/logit
    construction before position 120, not emission accounting. AR row tracing confirms pure `forward_scratch` at position 120 with `cur_token=27786`
    ranks `57874` over `6511` (`18.156204` vs `18.122902`). Disabling GDN tape replay and disabling verify graph capture both keep the DFlash row
    inverted, so the repro is not isolated to tape replay or graph capture. A `--block-size 2 --no-adaptive-b` DFlash control stays on the AR token
    path through this window (`tail=[27786, 57874]`), which narrows the next comparison to multi-position verify/rollback state accumulation at B=8+.
    Serial rollback replay via `forward_scratch` fixes the B=8 position-120 inversion and passes strict AR-token parity for the short DFlash gate, so
    it is now the conservative default. Fast GDN-tape/batched rollback replay remains diagnostic-only behind
    `HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY=0` until it can pass the same parity gate. The DFlash coherence gate now hard-fails opt-in AR parity
    runs if any DFlash row uses GDN-tape rollback replay, so rollback evidence cannot accidentally promote the diagnostic fast path. On gfx1151,
    `HIPFIRE_DFLASH_AR_PARITY=1 ./tests/coherence-gate-dflash.sh --fast` passed with conservative replay only: prose `replay_full_prefill=92`,
    code `replay_full_prefill=5`, and both rows `replay_gdn_tape=0`.
    A new opt-in diagnostic, `HIPFIRE_DFLASH_ROLLBACK_COMPARE=1`, compares fast GDN-tape rollback replay against serial replay at the same cycle.
    Re-running the known prose repro with `HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY=0 HIPFIRE_DFLASH_ROLLBACK_COMPARE=1
    HIPFIRE_DFLASH_TRACE_POSITION=120 HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN=57874` localizes the first recurrent-state mismatch to
    `s_matrix[0]`: `bytes=786432`, `differing_bytes=169285`, `first_offset=1`, `serial_byte=243`, `gdn_byte=242`. The verifier row at the same
    position still ranks DFlash token `6511` above AR token `57874`, so the next blocker is fixing first-layer GDN replay state parity before
    re-enabling fast replay.

  - Define the first backend module contract for one Qwen35 dense FFN/SwiGLU/down segment:
      - CPU backend is oracle.
      - GPU backend is current production path.
      - NPU backend remains opt-in.
      - Evidence records selected backend, module id, drift, and fallback reason.
    Status: the Qwen35 dense FFN BF16 oracle/probe path now exposes a typed `qwen35_dense_ffn_swiglu_down` contract with CPU-oracle,
    GPU-production, and NPU-opt-in backend preferences. The normal GPU production path and compare/cpu probe path now build the same in-place
    module invocation/output object tying the tensor/state contract to backend selection and output evidence. The adjacent Qwen35 attention `wo`
    residual projection now has the same in-place invocation shape on its production helper. Compare/cpu probe evidence records module id, selected
    backend, CPU oracle backend, drift stats when comparing GPU to CPU, and fallback reason. XDNA execution remains reserved until a real NPU backend
    lands.

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
    Status: `generate_batch_decode_step_done` now exposes selected backend, batch size, compatible state kinds, cached-prefix metadata, and fallback
    reason, and `/health.decode_batch` mirrors those fields for scheduler/status consumers. Active decode sessions now carry their resident
    prefix token count into the daemon envelope so decode telemetry can report real cached-prefix token totals.

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

  - Add backend module contract tests:
      - dense FFN/SwiGLU/down contract shape and statelessness,
      - CPU/GPU/NPU-opt-in backend selection,
      - evidence records selected backend, module id, drift, and fallback reason.

  ## Assumptions

  - Target architecture is Qwen35 first.
  - Streaming, tools/images, PFlash, CASK, multi-GPU batching, and cross-session MTP/DFlash verify batching remain out of scope.
  - Public OpenAI-compatible API shape does not change.
  - Existing daemon protocol can remain while the modular libraries are introduced behind it.
  - New crate extraction happens after these APIs pass in place.
