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
    position still ranks DFlash token `6511` above AR token `57874`. Switching Q8 GDN-tape replay to the same per-token
    `gated_delta_net_q8` recurrence cadence used by serial decode did not clear the known position-120 repro: the first mismatch is still in
    `s_matrix[0]` (`bytes=786432`, `differing_bytes=169288`, `first_offset=5`, `serial_byte=124`, `gdn_byte=123`), and the verifier row still
    chooses DFlash token `6511` over AR token `57874`. The next blocker is comparing the captured fast-replay `q/k/v/alpha/beta` inputs against
    serial per-token decode inputs before the first-layer GDN update, then fixing that input/capture drift before re-enabling fast replay.
    That comparison now shows the drift starts before replay recurrence: at the known position-120 repro, raw captured `qkv` for LA layer 0 differs
    before the first GDN update (`bytes=40960`, `differing_bytes=22466`, `first_offset=0`, `serial_byte=152`, `gdn_byte=255`). Disabling
    verify graph leaves the same raw `qkv` mismatch, so this is not graph replay. The first post-prefill DFlash cycle also differs at position 59
    (`bytes=81920`, `differing_bytes=45778`, `first_offset=0`, `serial_byte=132`, `gdn_byte=88`), which localizes the next blocker to batched
    verify/tape projection parity versus serial decode projection bytes rather than accumulated fast-rollback state.
    A follow-up serial-tape control shows replay is also not byte-exact when the tape is populated from serial decode projection bytes: position 120
    still mismatches after a one-step serial-captured tape replay (`s_matrix[0]`, `differing_bytes=169326`, `first_offset=8`, `serial_byte=190`,
    `serial_tape_byte=189`). Position 59 with two replay steps similarly mismatches. This rules out Q8 recurrence cadence alone and points at the
    pre-GDN replay segment: serial decode uses gfx1151's fused sigmoid/alpha-gate + conv1d/SiLU/split kernel, while GDN tape replay currently stores
    post-gate alpha/beta and replays through the separate conv-only path. Fast replay needs either raw pre-gate alpha/beta capture plus the exact
    fused serial kernel shape, or a proven byte-exact conv-only replay variant, before it can graduate from diagnostic mode.
    Adding raw pre-gate alpha/beta tape buffers and replaying the serial-captured tape through the fused gate+conv wrapper did not clear the control:
    position 120 remains at the same one-step serial-tape mismatch (`s_matrix[0]`, `differing_bytes=169326`, `first_offset=8`, `serial_byte=190`,
    `serial_tape_byte=189`), and position 59 remains mismatched as well. The next comparison needs to move after the fused gate+conv wrapper and
    compare serial decode's `q_raw/k_raw/v` plus post-gate alpha/beta against the fused replay outputs, before QK norm and GDN.
    That comparison now passes: both position 120 and position 59 report `dflash-rollback-fused-output-compare ... match` and
    `dflash-rollback-gdn-input-compare ... match`, while the following `s_matrix[0]` compare still mismatches. Matching the
    `GDN_REQUANT_FRAME` sequence proves the one-step position-120 serial-tape control is byte-exact
    (`dflash-rollback-serial-tape-compare ... match`). The two-step position-59 control still mismatches under layer-major replay, but a fresh
    token-major same-frame replay matches (`dflash-rollback-serial-tape-token-major-compare ... match`). This rules out the Q8 recurrence kernel
    and snapshot/restore as the serial-tape blocker: the replay must consume stochastic requant frames in the same token-major order as serial
    decode. The remaining fast-path blocker is still verify-tape projection parity (`qkv` mismatch at positions 59 and 120) plus a production-safe
    frame/order policy before GDN-tape rollback can replace conservative serial rollback. A projection-input tape diagnostic now proves the
    known position-120 repro reaches LA layer 0 with byte-identical normalized/rotated projection input, then diverges at that layer's batched
    verify `qkv` output (`bytes=40960`, `differing_bytes=22466`, `first_offset=0`, `serial_byte=152`, `gdn_byte=255`). The next blocker is
    therefore the batched LA `qkvza` projection family versus the serial decode projection family for identical input, not hidden-state drift
    before the first LA layer. A global diagnostic A/B with `HIPFIRE_FP16=0` clears that LA0 `qkv` mismatch but is too broad and slows prefill to
    ~22 tok/s, so it is not a production policy. The narrower `HIPFIRE_HFQ4_QKVZA_FAST=0` startup flag now bypasses only the HFQ4 qkvza fast
    projection family; on the same gfx1151 position-120 repro it keeps prefill near the default path (~57 tok/s), clears the LA0 `qkv` first
    mismatch, and exposes the next drift earlier in the network at `x_in index=1` (`bytes=20480`, `differing_bytes=11716`, `first_offset=0`,
    `serial_byte=108`, `gdn_byte=144`). The remaining blocker is now to locate the first non-qkvza batched verify drift before LA layer 1 before
    any fast replay path can be promoted. Extending the same tape diagnostic through LA0's post-projection stages shows `q_raw/k_raw/v`, repeated
    `q/k`, GDN output, and gated output norm all match serial, then the first mismatch appears after LA0 `wo` residual
    (`attn_residual[0]`, `bytes=20480`, `differing_bytes=12020`, `first_offset=0`, `serial_byte=90`, `gdn_byte=95`). The current blocker is
    therefore the batched HFQ4 residual projection family for LA `wo` versus serial `weight_gemv_residual`, not GDN recurrence, gated norm, or
    the next layer's RMSNorm input. Follow-up diagnostics did not clear that boundary: `HIPFIRE_HFQ4G256_MMQ_GFX1151=0` still reports the same
    `attn_residual[0]` mismatch, and a local per-row `weight_gemv_residual` experiment for the whole HFQ4 residual chunk also kept the first
    mismatch at `attn_residual[0]` while degrading output quality. The next cut should compare the residual input and residual destination rows
    immediately before the `wo` call, not swap residual kernels. Adding that `wo_residual_in` tape point confirms the destination residual row
    also matches serial immediately before LA0 `wo`; the first mismatch remains `attn_residual[0]`. That leaves the MQ4 `wo` projection output or
    fused residual epilogue itself as the active blocker. Capturing the rotated `wo_input` confirms the MQ activation rotation also matches serial,
    so both inputs to the fused residual projection match before the call. The remaining split is inside the batched `gemm_hfq4g256_residual`
    projection/epilogue path versus serial `gemv_hfq4g256_residual`. The backend-level residual parity harness reproduces the issue at the LA
    `wo` shape (`M=5120 K=2048 B=2`) under default gfx1151 dispatch; disabling MMQ alone is insufficient, while forcing the scalar batched
    fallback with `HIPFIRE_HFQ4G256_MMQ_GFX1151=0 HIPFIRE_MMQ=0 HIPFIRE_FP16=0` restores byte-exact parity for B=2/4/8. The narrow diagnostic
    flag is now `HIPFIRE_HFQ4_RESIDUAL_FAST=0`, which bypasses only the HFQ4/MQ4 residual fast branches and leaves the scalar batched residual
    kernel in place. Running the position-120 DFlash comparator with both `HIPFIRE_HFQ4_QKVZA_FAST=0` and `HIPFIRE_HFQ4_RESIDUAL_FAST=0` clears
    the LA `wo` residual split and moves the first mismatch to the post-FFN layer output (`layer_out[0]`, `bytes=20480`,
    `differing_bytes=10604`, `first_offset=0`, `serial_byte=65`, `gdn_byte=20`). The next cut is FFN-stage tape capture: FFN norm/rotated
    gate-up input, gate/up outputs, SwiGLU/down input, and the pre-`w_down` residual destination.
    Adding that FFN tape capture moves the first mismatch earlier to the FFN gate projection (`ffn_gate[0]`, `bytes=69632`,
    `differing_bytes=39124`, `first_offset=0`, `serial_byte=156`, `gdn_byte=59`) with qkvza/residual fast paths disabled. A broad
    `HIPFIRE_FP16=0` A/B clears `ffn_gate` and moves the split to `w_down_input[0]` (`bytes=69632`, `differing_bytes=69225`,
    `first_offset=0`, `serial_byte=0`, `gdn_byte=24`), so the gate/up split is in the batched HFQ4/MQ4 gate-up fast routing versus serial
    `fused_gate_up_hfq4g256`. The narrow diagnostic flag for that split is now `HIPFIRE_HFQ4_GATE_UP_FAST=0`. The initial `w_down_input`
    split was a serial tape-capture bug: MQ serial `weight_gemv_swiglu_residual` writes the rotated down input into `gpu.mq_x_rot`, not
    `s.ffn_hidden`. Capturing `gpu.mq_x_rot` for MQ `w_down` clears the LA0 FFN split under
    `HIPFIRE_HFQ4_QKVZA_FAST=0 HIPFIRE_HFQ4_GATE_UP_FAST=0 HIPFIRE_HFQ4_RESIDUAL_FAST=0`; the first mismatch moves to `x_in[3]`
    (`bytes=20480`, `differing_bytes=10071`, `first_offset=0`, `serial_byte=89`, `gdn_byte=196`), i.e. after the intervening FullAttention
    segment between LA index 2 and LA index 3. FullAttention bridge tape shows the hidden input to that segment matches serial, then the first
    mismatch appears at `fa_bridge_q[3]` after FullAttention q/k/v projection plus RoPE (`bytes=24576`, `differing_bytes=14526`,
    `first_offset=0`, `serial_byte=64`, `gdn_byte=8`) under
    `HIPFIRE_HFQ4_QKVZA_FAST=0 HIPFIRE_HFQ4_GATE_UP_FAST=0 HIPFIRE_HFQ4_RESIDUAL_FAST=0`. Finer bridge capture shows the normalized/rotated
    FA input matches, then the split occurs at `fa_bridge_q_full[3]` (`bytes=49152`, `differing_bytes=26488`, `first_offset=0`,
    `serial_byte=64`, `gdn_byte=16`), before q_norm/RoPE. A broad `HIPFIRE_FP16=0` A/B clears `fa_bridge_q_full` and moves the first
    mismatch to `fa_bridge_attn_out[3]` (`bytes=24576`, `differing_bytes=9335`, `first_offset=0`, `serial_byte=14`, `gdn_byte=10`). The
    narrower diagnostic flag `HIPFIRE_HFQ4_QKV_FAST=0` reproduces that move without disabling unrelated HFQ4 fast paths. The next cut is
    FullAttention output capture before/after the output gate now that q/k/v parity can be forced independently. Adding a pre-gate
    `fa_bridge_attn_raw` capture shows the split is already in the raw attention output (`fa_bridge_attn_raw[3]`, `bytes=24576`,
    `differing_bytes=6347`, `first_offset=0`, `serial_byte=69`, `gdn_byte=66`) under
    `HIPFIRE_HFQ4_QKV_FAST=0 HIPFIRE_HFQ4_QKVZA_FAST=0 HIPFIRE_HFQ4_GATE_UP_FAST=0 HIPFIRE_HFQ4_RESIDUAL_FAST=0`. The next blocker is
    batched Q8 FullAttention attention semantics: mask/bias, per-row position, KV write/read ordering, or the Q8 batched attention kernel.
    Three narrow Q8 FA attention diagnostics reproduce the same `fa_bridge_attn_raw[3]` mismatch without moving byte counts or first bytes:
    `HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP=1` routes through one-row masked launches, `HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS=1` removes the
    DDTree bias from the Q8 masked path, and `HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP=1` routes each row through the scalar causal
    `attention_q8_0_kv` path. That makes the next cut KV write/read ordering, row-position materialization, or replay/oracle semantics around
    the FA bridge rather than a pure `attention_q8_0_kv_batched_masked` grid/bias bug. A fourth diagnostic,
    `HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP=1`, defers Q8 FA KV writes and runs each row through serial `kv_cache_write_q8_0(K)`,
    `kv_cache_write_q8_0(V)`, and scalar `attention_q8_0_kv` in row order; it also reproduces the same `fa_bridge_attn_raw[3]` mismatch
    (`bytes=24576`, `differing_bytes=6347`, `first_offset=0`, `serial_byte=69`, `gdn_byte=66`). This rules out batched Q8 KV write timing as
    the active split. The next cut should compare the serial replay oracle semantics and FA bridge row/position materialization around the raw
    attention capture, or add an explicit serial-vs-batched raw attention oracle before the tape comparison.

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
