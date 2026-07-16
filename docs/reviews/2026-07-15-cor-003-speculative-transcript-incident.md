# COR-003 speculative transcript incident

> **Historical incident report — superseded implementation guidance
> (2026-07-16).** This report preserves the transcript-ownership failure and
> its forensic findings from 2026-07-15. The bounded terminal scanner,
> parser/driver normal-versus-discard seams, sealed Qwen speculative-turn
> authority, cache/reset behavior, and native MTP cancellation described by the
> subsequent COR-003 implementation are now present. Fresh CPU, DFlash
> coherence, and multi-turn serving validation passed on 2026-07-16 (reports
> `/tmp/coherence-dflash-20260716-110721.md` and
> `/tmp/serve-multiturn-20260716-110919.md`, with no DFlash hard or soft
> warnings); `git diff --check` also passed. Transactional Qwen target
> loading is explicitly deferred to `SPEC-003`. Do not use the old
> recommended-design or missing-work sections as current status; use [the
> canonical tracker](../../.agent-progress/device-mesh-refactor-tracker.md).

## Executive summary

COR-003 exposed a broken ownership boundary in speculative Qwen output. The
filter, visible events, tool extraction, replay/cache tokens, and fingerprints
derive terminal output independently. A literal stop can therefore suppress
wire text while post-stop bytes still affect tool calls, cache replay, or the
assistant-turn fingerprint.

Do not continue patching `Qwen35Emit::finish()` in isolation. Replan Task 4
around one authoritative pre-stop transcript boundary.

## Confirmed failures

1. Initial `EosFilter` changes mixed literal-byte offsets with
reasoning-stripped offsets, allowing synthetic or leaked markers. This was
replaced with a bounded raw stop scanner upstream of stripping.
2. `FilterAction::Stop { emit }` required propagation through `StreamParser`;
without it safe pre-marker text was lost or generation continued silently.
3. Generic AR originally bypassed finalization on abort/errors and skipped its
ChatML trailer after a control-flow refactor. It now distinguishes normal
completion from discard, but this does not solve speculative transcript state.
4. Qwen speculative finalization filtered text but extracted tool calls from a
full token decode. A stop at `<tool_call>` could suppress text yet still emit a
tool call.
5. The attempted safe-token patch is unsound: it can remove valid pre-stop
tokens, retain post-stop tokens in daemon cache replay, and permit post-stop
contamination if the emitter is observed again after stopping.

## Required invariant

For every speculative turn there is exactly one sealed prefix. Visible text,
tool extraction, cache replay tokens, and fingerprinting consume that prefix
only. Bytes/tokens after the stop boundary are not observable or cacheable.
Abort/error seals nothing pending and never recovers tool calls.

## Recommended design

Ship a small stop-boundary capability plus a byte commit watermark:

- A bounded raw `StopQuarantine` sits upstream of UTF-8/reasoning filtering.
- It forwards only raw bytes proven not to form a future stop marker.
- A runtime-owned finalized-turn value owns the canonical safe transcript and
  replay-token cut. Its consuming seal/finalize operation is the sole source
  for visible events, tool extraction, cache replay, and fingerprints.
- Normal completion flushes safe pending bytes before sealing. Abort/error
  drops pending state without sealing/recovering tools.
- Do not add a ledger DAG, hash chain, global epoch, or cross-engine reason
  taxonomy; those are unjustified for COR-003.

## Plan requirements

1. Define byte/token boundary semantics for BPE, hidden reasoning, forced
   tokens, partial UTF-8, and stop markers.
2. Implement and exhaustively test the pure bounded quarantine scanner first.
3. Introduce one immutable finalized-turn projection used by Qwen emitter and
   daemon cache/fingerprint consumers.
4. Migrate `generate_spec` normal/discard outcomes to that projection.
5. Add integration tests proving a post-stop tool payload never reaches text,
   ToolCalls, replay tokens, cache fingerprint, or the next turn.
6. Keep AR/spec engine unification and model reset changes out of scope.

## Evidence status

Targeted filter/parser/AR tests have passed at earlier checkpoints. The latest
speculative safe-transcript change requires redesign and fresh red/green tests;
do not treat previous Task 4 results as acceptance evidence.

## Audit decision: retain, split, and scrap

### Retain in independently reviewed changes

- Bounded raw stop scanning, earliest-marker selection, terminal phases, and
  `FilterAction::Stop { emit }` in `eos_filter.rs`.
- `StreamParser` propagation of safe stop prefixes and idempotent normal
  finalization.
- Cohere one-shot pending-action recovery and forced-marker handling.
- Generic AR and generic speculative normal-versus-discard terminal seams.

These changes must be revalidated, but they do not depend on the broken Qwen
cache transcript bookkeeping.

### Scrap and replace as one sealed-turn migration

- Qwen `committed_tokens`, `streamed_tokens`, `transcript`, and the
  pop-until-decoded-length heuristic.
- `SpecEmit::streamed_tokens`, `SpecRun.streamed_tokens`, and
  `SpecRun.cache_replay_tokens` as competing sources of turn authority.
- Daemon-side re-decoding/re-parsing for tools and fingerprints.
- Cache replay built from generic `emitted` state.
- Immediate token-ID diagnostics before a token is proven sealed.
- Synthetic `</think>` wire output without corresponding admitted tokens.
- Duplicate byte-stop logic in Qwen emitter and static carrier terminator lists
  passed directly into it.

### Missing work after replacement

- Qwen PP raw-filter normal/discard epilogue.
- Bespoke DeepSeek AR normal/discard epilogue.
- A dedicated two-turn cache-contamination fixture.

## Required policy decisions

- An intra-token stop preserves valid visible bytes but makes replay/cache
  uncacheable; never drop a prior whole token to fake alignment.
- Normal completion flushes a valid partial stop prefix; abort/error does not.
- Incomplete UTF-8 emits its maximal valid prefix without U+FFFD.
- Synthetic think-close text is forbidden unless real admitted tokens produced
  it.
- Carrier framing owns typed protocol terminators/trailers; user stops are
  separate.
- Target state beyond a sealed cut is rolled back; where that is unavailable,
  reset and force a cold next turn.
- Token-ID diagnostics report sealed whole-token output only.
