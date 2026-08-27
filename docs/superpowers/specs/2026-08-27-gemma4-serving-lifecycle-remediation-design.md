# Gemma4 serving and lifecycle remediation design

Date: 2026-08-27

## Goal

Complete the unfinished Gemma4 product path exposed by STEP-005 Task 7:

- make eager Gemma terminal events conform to the staged client-commit protocol;
- wire the Step-only lowered/MoE bundle into production generation;
- provide safe exact-prefix KV reuse for eager and lowered Gemma;
- make EOS, invalid-token, error, abort, and reset behavior conform to the
  intended shared lifecycle contract already used by the strongest DeepSeek and
  Qwen paths; and
- close proven lowered allocation-ownership defects before re-running lifecycle
  gates.

Existing limitations are not preserved merely because they exist. They are kept
only when evidence shows a deliberate product constraint.

## Current evidence

STEP-005 established deterministic parity between Gemma4's historical hand path
and typed Step execution, then deleted Gemma4's SuperOp route. Task 7 exposed
separate product gaps:

- eager dense Gemma emits visible token events and a legacy `done`, but omits
  `finish_reason` and never stages `commit_ready`; HTTP streaming therefore
  ends without a terminal chunk, usage, or `[DONE]`;
- canonical Gemma4 MoE correctly loads as `Gemma4LoweredBundle`, but
  `hipfire-generate` explicitly refuses every lowered bundle as unwired;
- lowered direct generation works because the example bypasses loader, daemon,
  product route selection, terminal handling, and reset ownership;
- lowered scratch allocates `pos_buf` while claiming nonexistent
  `DeviceBuffer::Drop` ownership;
- lowered weight teardown manually frees primary buffers and can omit optional
  sidecars;
- partial lowered construction is not transactional;
- MoE post-unload free memory decreased by 2 MB on each of four cycles, but the
  probe performed no MoE forward and existing diagnostics cannot attribute the
  drift;
- current dense and MoE artifacts produce malformed but deterministic decoded
  text, so numerical repeatability does not establish semantic quality; and
- the named E2B/E4B fixtures are absent.

## Scope

### Gate A: dense terminal contract

Update eager Gemma AR and EAGLE completion to:

- classify successful termination as `stop` or `length`;
- construct one rich pending terminal payload;
- call `await_client_terminal_commit`;
- publish request/cache effects only after `Commit`;
- emit the identical payload through `emit_staged_terminal_done`; and
- roll back on client abort or terminal timeout.

Gemma's carrier advertises the current semantic contract version so staged
terminal fields are projected by the HTTP/OpenAI layer. The common completion
adapter is not changed to synthesize a terminal from legacy `done`.

### Gate B: lowered/MoE generation and exact-prefix cache

Replace the explicit lowered-bundle refusal with a product adapter in
`hipfire-generate`. It borrows `Gemma4LoweredBundle`, uses the parity-proven
`lowered::forward_scratch`, and shares Gemma request framing, sampling, events,
stop decisions, accounting, terminal staging, and cache publication.

Initial prompt prefill is per-token Step execution. Batched prefill, adaptive
thresholds, and scheduler integration remain GEN-003 work.

Both eager and lowered Gemma implement exact-prefix KV reuse. Arbitrary LCP
rewind, cross-session cache lookup, persisted caches, and cross-model reuse are
out of scope.

### Gate C: lifecycle ownership

Fix explicit lowered allocation ownership, make reset total, make construction
transactional, add size-aware allocation telemetry, and re-run four-cycle
load/generate/reset/unload evidence.

## Non-goals

- Preserving `cache_capable=false` after Gemma satisfies the cache contract.
- Copying unfinished DeepSeek/MiniMax lifecycle quirks.
- A common HTTP compatibility fallback for direct legacy `done`.
- New Gemma kernels, quant formats, PP/TP/EP admission, speculation, EAGLE for
  lowered/MoE, or graph capture for the lowered route.
- General LCP rewind or cache snapshots.
- Waiving malformed-output or missing-fixture gates.

## Architecture

### Shared Gemma request shell

A narrow request shell in `hipfire-generate` owns behavior common to eager and
lowered Gemma:

- prompt framing and tokenization;
- exact-prefix cache matching;
- context and generation limits;
- sampling configuration and RNG;
- streamed token/reasoning events;
- stop and EOS handling;
- generated/prefill/cache/timing accounting;
- staged terminal transactions; and
- abort/error rollback.

Forward execution remains behind private local adapters. The concrete code may
use functions, an enum, or a sealed local trait to satisfy Rust borrowing. It
must not create a public pass-through architecture abstraction.

Each adapter provides only:

- reset request state;
- prefill one token;
- decode one committed token;
- read logits;
- report and set its materialized cursor; and
- expose cache rollback/invalidation capability.

### Eager adapter

The eager adapter retains the existing eager weights/state, EAGLE boundary, and
graph-capable decode. Its AR and EAGLE terminals use the shared staged protocol.
EAGLE remains eager-only.

### Lowered adapter

The lowered adapter owns:

- lower config and weights;
- `Gemma4Scratch`;
- sliding KV;
- full KV;
- one authoritative materialized cursor; and
- exact-prefix cache transaction metadata.

Per-token prefill and decode both call `lowered::forward_scratch`. The adapter
samples from `scratch.logits`; it does not reconstruct a second forward loop or
call eager execution.

## DeepSeek-aligned lifecycle contract

Gemma aligns with the intended neutral DeepSeek/Qwen lifecycle components:

- correlated attempt identity and cancellation;
- EOS-before-materialization;
- one pending terminal payload;
- `await_client_terminal_commit` followed by payload-identical staged `done`;
- commit effects only after client commit;
- attested rollback on abort/error; and
- `LoadedModel::reset_context` as cross-system reset authority.

Gemma does not copy DeepSeek-specific DSML, HC/MoE/EP reset policy, cache
fingerprints, DSpark/MTP behavior, EOS IDs, or known incomplete behavior.

Known defects that must not be copied include:

- missing sampled-token range checks;
- inconsistent handling of non-finite logits;
- ordinary forward errors without rollback;
- ignored user stop sequences;
- stop tokens emitted/appended without being forwarded;
- cursor-only reset for stateful caches; and
- semantic-contract legacy mode combined with staged-only fields.

## Token transaction

For each sampled token:

1. Require logits length to equal configured vocabulary size.
2. Sample only from finite logits. An all-non-finite distribution is an error,
   not token zero.
3. Require `token_id < vocab_size` before decode, parser, emission, or embedding.
4. Check EOS before parser, emission, history, or forward.
5. Feed a non-EOS token through the output/stop parser.
6. Forward the token into model state/KV.
7. Only after successful forward:
   - advance the materialized cursor;
   - append materialized token history;
   - release eligible visible/parser events;
   - increment completion count.

EOS is not emitted, counted, stored in materialized history, or forwarded. If a
future framed prompt includes an EOS/turn marker, that token belongs to the
uncached suffix and is prefetched normally.

### Stop sequences

Use the shared quarantine/parser pattern. Visible bytes may be held while a
possible stop prefix is unresolved. KV truth remains explicit:

- every successfully forwarded token enters materialized history;
- visible suppression does not imply an unrecorded KV rewind;
- if a later framed prompt does not contain those materialized stop tokens,
  exact-prefix matching fails and cold-resets.

No token may be visible or cached if its forward failed.

## Invalid logits and tokens

These fail closed before emission:

- logits length differs from `vocab_size`;
- all logits are non-finite;
- sampling returns an out-of-range token;
- tokenizer/parser cannot consume the token under the active contract.

Failure flow:

```text
no token emission
→ production_fail_closed_rollback_live
→ restore prior committed exact-prefix cache when safe, otherwise invalidate
→ one correlated terminal error
→ no successful done
```

## Exact-prefix cache

### Cache authority

Gemma session state stores:

- materialized token history;
- committed cursor;
- model/config identity;
- validity;
- request-start cursor/overwrite boundary for rollback.

`materialized_tokens.len() == cursor` and both KV families represent that cursor.

### Request start

Given framed input `N` and valid committed tokens `C`:

```text
if identity matches and N starts_with C:
    retain KV
    prefill N[C.len..]
    cached_tokens = C.len
else:
    total architecture reset
    prefill N
    cached_tokens = 0
```

The exact-prefix cache is available to both eager and lowered adapters. Once
implemented and validated, Gemma is advertised cache-capable; the old exclusion
is removed.

### Publication

Cache publication is a terminal commit effect:

```text
request start:
    retain prior committed transaction

during generation:
    mutate working KV/cursor
    track only successfully materialized rows

client Commit:
    publish working token history and cursor
    emit staged done

client Abort/error/discard:
    restore prior committed cursor when safe
    otherwise invalidate and cold-reset
```

A successful `length` terminal may publish an exact materialized prefix. Reuse is
still conditional on the next framed prompt containing that exact prefix; no
semantic completeness assumption is needed.

### Sliding-window rollback

A failed request may overwrite live committed rows in the sliding cache.
Snapshots are out of scope:

- if append-only work did not overwrite committed live-window rows, restore the
  committed cursor;
- if it did, invalidate and cold-reset;
- never reuse partially overwritten state.

## Terminal transaction

Build one `pending_done` containing:

- id and attempt id;
- explicit `finish_reason`;
- completion token count;
- total prompt token count;
- exact cached token count;
- prefill/decode/TTFT/total timings; and
- future staged semantic fields under the current contract version.

Flow:

```text
pending_done
→ commit_ready with identical fields
→ client Commit or Abort
→ done with identical fields, or attested rollback
```

Successful terminal emission occurs exactly once. HTTP streaming receives the
terminal chunk, usage, and `[DONE]`. EOF before terminal remains a client error.
The Python harness may be hardened separately so it cannot mask premature EOF,
but harness changes are not the product fix.

## Reset and error handling

### Architecture reset and request rollback

`ArchModel for Gemma4LoweredBundle` implements total cold reset:

- lowered cursor;
- sliding and full KV cursor/offset state;
- committed and working exact-prefix transactions;
- request-local scratch state owned by the architecture; and
- graph/capture validity request.

The eager adapter implements the equivalent cold reset for its state and cache.

The private Gemma request adapter separately implements
`rollback_working_request`. It may restore the prior committed cursor/cache only
when append-only rollback is attested safe. Otherwise it invalidates the cache
and requests the total cold reset. This distinction prevents a successful
cache restore from being immediately erased by a generic reset.

### Cross-system reset

`LoadedModel::reset_context` remains the only cross-system **cold-reset**
authority. Explicit reset, identity mismatch, unsafe overwrite rollback, and
unrecoverable request failure flow through it. It coordinates:

- host sequence/conversation/cache metadata;
- architecture cold reset;
- speculative state where present;
- graph/replay invalidation;
- pending terminal state; and
- synchronization attestation.

### Forward, sampling, or terminal failure

- discard uncommitted output;
- ask the Gemma adapter to roll back the working request;
- if rollback safely restores the committed cache, synchronize and attest that
  restoration without cold-resetting it;
- if rollback invalidates the cache, call `LoadedModel::reset_context` for total
  cold reset and synchronization;
- emit one correlated error or aborted terminal only after an attested result;
- emit one fail-closed error and no successful `done` when rollback cannot be
  attested; and
- never retry through eager, legacy, or another executor.

### Capacity

Before every forward, require cursor capacity. Overflow emits an explicit
context-length error and performs total rollback/reset. No wrapping or silent KV
overwrite is allowed.

## Resource ownership

### Scratch

`Gemma4Scratch::free_gpu` explicitly frees every `GpuTensor` and `pos_buf`.
No ownership comment may rely on nonexistent `DeviceBuffer::Drop`.

### Weights

Use complete weight teardown:

- primary buffer;
- AWQ scales;
- Paro/Givens sidecars;
- alias-versus-owner distinction;
- MoE pools once;
- no independent frees for expert pool subviews.

Prefer `WeightTensor::free_all` where the object owns its storage. Pool-backed
views remain documented non-owning exceptions.

### Transactional load

The lowered bundle is not published until config, weights, scratch, sliding KV,
full KV, and session/cache resources all succeed. Failure rolls back completed
owners in reverse order.

## Allocation telemetry

Before attributing the observed 2 MB/cycle drift, report:

- model-owned bytes;
- pool cached bytes;
- free device bytes;
- graph/module residency where observable;
- cycle and phase; and
- each owner freed.

The proven `pos_buf` leak is fixed regardless of its size. Remaining drift is
classified only from size-aware evidence. A monotonic free-memory decrease is
not declared JIT variance without proof.

## Verification gates

### Gate A: terminal contract

- eager AR stop → staged `finish_reason=stop`;
- eager AR cap → `length`;
- eager EAGLE equivalents;
- payload identity between `commit_ready` and `done`;
- HTTP terminal chunk, usage, and `[DONE]`;
- client abort with attested rollback;
- current semantic contract version advertised;
- `cached_tokens=0` only when no prefix was reused.

### Gate B: lowered/MoE adapter and cache

- production lowered adapter matches direct lowered greedy IDs/logits;
- per-token prompt prefill;
- EOS never emitted, counted, or materialized;
- invalid logits/token ID fail before emission and roll back;
- exact-prefix second turn reports `cached_tokens > 0`;
- unrelated turn cold-resets with `cached_tokens=0`;
- abort restores committed cache when safe;
- overwrite invalidates cache;
- stop, length, and successful terminal publication remain prefix-correct;
- both KV families and materialized cursor stay equal;
- HTTP staged terminal and usage are complete.

### Gate C: lifecycle

- fault injection at every construction owner boundary;
- normal unload frees all lowered owners;
- AWQ/Paro sidecar ownership coverage;
- four dense and MoE load/generate/reset/unload cycles;
- no monotonic owner or pool growth after warm cycle;
- reset then unrelated generation has no state bleed;
- validation uses isolated homes/ports and does not touch user processes.

## Quality and external fixture gates

Source remediation does not waive final Task 7 requirements:

- compare byte-identical token inputs against a known-coherent HF or eager
  reference;
- compare dense and MoE artifacts against a higher-precision control;
- provision the named E2B/E4B fixtures;
- inspect decoded production output.

Classification:

- Step/eager/reference divergence: implementation defect;
- all hipfire routes agree but HF differs: conversion/quantization/artifact
  defect;
- all references agree but output is poor: model/prompt quality.

## Completion

The remediation implementation completes when Gates A–C pass. Original STEP-005
Task 7 and final Gemma contract review remain blocked until quality and external
fixture gates also pass. No source-remediation result silently reduces those
acceptance criteria.
