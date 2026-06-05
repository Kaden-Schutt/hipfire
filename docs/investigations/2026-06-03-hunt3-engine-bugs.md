# Hunt-3 engine bug sweep (2026-06-03)

Adversarial multi-agent bug-hunt (`wf_f58cd575-2a6`, 109 agents, ~8.7M tokens,
~31 min) over 4 surfaces deliberately chosen to **avoid** the saturated
single-GPU A3B decode path and the separate quant track (#392):

1. **Multi-GPU / Pipeline-Parallel** (`generate_multi`, pp>1)
2. **Sampling + MoE dispatch**
3. **Daemon concurrency / streaming** (Rust JSON-lines)
4. **Bun CLI → backend entry** (TS + IPC boundary)

Method: 4 Explore scouts → 20 finder lenses (grounded by the map) → 3
perspective-diverse skeptics per finding (reachability / intended-vs-bug /
blast-radius, **refute-by-default**) → synthesis. Result: **20 confirmed
(≥2/3 affirm), 1 contested (1/3), 7 rejected**.

> ⚠️ **Line numbers in the raw swarm output are systematically unreliable**
> (agents drifted / approximated). Every entry below has been **re-anchored by
> symbol**. The CLAIMS are mostly sound when checked; **#5 is a falsified
> verification miss** (see below). Items marked **[src-verified]** were checked
> against live master `02634f4c` by hand; the rest rest on the 2/3 swarm verify
> and should be re-confirmed at fix time.

---

## Cross-cutting themes

1. **Sibling-path divergence (dominant).** `generate_multi` (pp>1) and
   `generate_vl` are systematically missing lifecycle/correctness fixes that
   the single-GPU `generate()` text path received: per-request Jinja
   rerender+reset (#389), state cold-reset on error/abort, grammar-constrained
   tool-call masking, generated-only repeat-penalty scope, committed-event
   emission, per-request deterministic RNG. **Fix strategy: treat `generate()`
   as the reference and audit each sibling against every invariant it enforces.**
2. **Single stdin/stdout pipe without a shared mutex.** Bun's one daemon pipe is
   serialized only by the busy/queue lock, but the idle-eviction timer,
   drain-restart, run-HTTP-fallback, and foreground-serve all drive send/recv or
   spawn outside that lock; `recv()`'s `EOF → process.exit` converts any pipe
   desync or daemon death into a full multi-client outage. Root cause: one-shot
   CLI design (process==request, flock singleton, exit-on-EOF) reused unchanged
   for a long-lived multi-client serve.
3. **Unchecked primitive coercion at boundaries.** `byte as char` re-UTF8s
   byte-fallback tokens to mojibake; JSON-string `max_tokens` string-concats into
   a 10M `max_seq`; JS-truthy `body.stream` diverges from OpenAI booleans.

---

## Confirmed (re-anchored + verification status)

### HIGH

- **H-A — `generate_multi` (pp>1) Jinja: drops system prompt + skips per-request cold-reset on turn 2+.** **[src-verified]**
  `daemon.rs:6320` still reads `try_jinja = jinja_enabled && m.seq_pos == 0 && m.chat_template.is_some()`
  — the `seq_pos==0` gate PR #389 *removed* from `generate()` (now `:7417` ungated) and `generate_dflash` (`:5091` ungated).
  Turn 2+ → `try_jinja=false` → Plain ChatML branch with `system=None` → extends turn-1 dirty KV/DeltaNet, no cold reset.
  This is the **explicitly-deferred #389 pp analogue**. cache_capable=true for arch 5/6/9 → CLI sends no per-turn reset, so it always fires.
  **Fix:** ungate `:6320` + add a `jinja && seq_pos>0` cold-reset (mirror `reset_pp_uncommitted_state!`) before the budget guard.

- **H-B — `recv()` EOF → `process.exit` kills the entire long-lived serve.** **[src-verified]**
  `cli/index.ts:1147` `process.exit(code === 0 ? 1 : code)` inside `recv()` (1131). One daemon panic/OOM (the ~13-req KV-OOM class)
  closes stdout → every queued/connected client dies; defeats the deliberate `drain()`-restart and the pre-warm try/catch.
  **Fix:** `recv()` throws a typed `DaemonClosedError` on `done`; one-shot `run()` catches→exit, serve handler catches→500 + `stop()/start()`.

- **H-C — tokenizer `byte as char` mojibake on SentencePiece byte-fallback.** **[src-verified]**
  `tokenizer.rs:1107` `result.push(byte as char)` in `decode_hex_escapes` (called by `decode`/`decode_bytes` at 604/639).
  `<0xHH>` → U+00HH → re-UTF8 to 2 bytes → mojibake. **Scope caveat:** only `is_gpt2_bpe==false` (Llama/SentencePiece) models —
  Qwen3.x is GPT2-BPE and uses `byte_to_id`, so the *primary* workload is unaffected (synthesis overstated "every model").
  Known-bad pattern: the same `byte as char` was already fixed elsewhere (see the regression comment at `:1969`).
  **Fix:** byte-emitting `decode_hex_escapes_bytes` that pushes raw `<0xHH>` bytes + `char::encode_utf8` for ordinary chars.

- **H-D — `max_tokens` string → string-concat → ~10M `max_seq` → unload-then-OOM (DoS).** **[src-verified]**
  `cli/index.ts:1968` `body.max_tokens ?? effective.max_tokens` (`??` guards null, *not* type) → `:1970` `requestMaxTokens + 1024`
  string-concats for a JSON-string `"100"` → `Math.max` coerces to ~10M → bumps load `max_seq`; daemon `:1499`
  `.as_u64().unwrap_or(4096)` has no upper clamp (bypasses the 524288 ceiling). One malformed request unloads the resident model then OOMs.
  **Fix:** type-validate/clamp `requestMaxTokens` at the boundary + clamp daemon-side `max_seq ≤ 524288`.

- **H-E — `forward_prefill_batch_multi` leaks already-allocated per-band `PrefillBatchScratch` on a later band's OOM.** [swarm-verified]
  `qwen35.rs:~11118` per-band `PrefillBatchScratch::new` loop is *outside* the result closure + free loop; band-1 OOM drops band-0's ~40
  GpuTensors with no `Drop` → monotonic VRAM leak. Same class as the fixed `PrefillBatchScratch::free_gpu` MoE-scratch leak; pp-specific
  (single-GPU sibling wraps `own_pbs` inside the closure). **Fix:** move per-band alloc inside a guarded closure whose Err arm frees pushed bands.

- **H-F — `generate_vl` think-cap force-close emits text tokens but no `{type:committed}` events → token-id stream desync.** [swarm-verified]
  VL force-close loop pushes close tokens + emits `{type:token}` + increments `generated` but never `emit_committed_event` → under
  `HIPFIRE_EMIT_TOKEN_IDS=1` the committed pos permanently diverges from `streamed_tokens.len()`, breaking coherence_probe / DFlash-gate
  detectors. AR `generate()` sibling emits correctly (`:6758`). **Fix:** add `emit_committed_event` in the VL force-close loop.

### MEDIUM

- **M-A — `generate_multi` forward-ERROR return paths leave DeltaNet/KV dirty across requests.** [swarm-verified; theme-consistent]
  Both abort paths call `reset_pp_uncommitted_state!`; the two forward-error returns (prefill Err, decode Err) emit `{type:error}` and
  return *without* it → partial-band failure leaves DeltaNet partially advanced; next cold turn prefills over non-zero state. **Fix:** call the
  reset macro before both error returns. (Note: single-GPU `generate()` forward errors `.unwrap()`→panic→clean restart, so it never carries this.)

- **M-B — deepseek4 greedy/top-k + llama argmax panic on NaN logit (`partial_cmp(..).unwrap()`).** [swarm-verified]
  `deepseek4/sampling.rs:62`(greedy)/`:72`(top-k) + `llama.rs:~5272`. A NaN logit (aggressive quant) panics → kills the daemon decode loop;
  the GPU kernel silently drops NaN, so there's a GPU/CPU split. **Fix:** `partial_cmp(..).unwrap_or(Ordering::Less)`, treat NaN as -inf.

- **M-C — `generate_multi` drops grammar-constrained tool-call sampling that pp=1 enforces.** [swarm-verified; theme-consistent]
  Its three sample sites are unconditional GPU `sampler::sample` with no grammar mask; `generate()` masks via the qwen35 `Matcher` on
  `<tool_call>`. pp>1 + tools → unconstrained → reopens the Pi-turn-12 ChatML-noise-in-tool_call attractor. **Fix:** lift `generate()`'s grammar wrapper into all three sites.

- **M-D — `generate_vl` repeat penalty scoped over prompt+generated, contradicting the text path's generated-only design.** [swarm-verified; theme-consistent]
  VL passes full `m.conversation_tokens` to `sample_cpu`; text path sets `ngram_scope_start = conversation_tokens.len()` after prefill
  (`:6371`, explicit MQ4/MQ6-recall comment). Prompt-dominated window suppresses names/numbers a transcription task must reproduce. **Fix:** capture VL scope start, pass generated-only slice.

- **M-E — Grammar/VL CPU sampling uses a process-global wall-clock-seeded RNG → non-determinism + cross-request state leak.** [swarm-verified]
  GPU path threads a per-request seed; `sample_cpu→llama::sample_top_p→simple_rand` reads/writes a process-global `AtomicU32` never reset.
  Inside one tool-call request the RNG source silently switches per-token; CPU output depends on prior/concurrent requests' sampling. **Fix:** thread the per-request seed through `sample_cpu` (or reset `SAMPLER_STATE` at the top of `generate()/generate_vl`).

- **M-F — `stop` sequences silently ignored end-to-end.** [swarm-verified]
  `cli/index.ts:~2187` `genParams` never reads `body.stop`; daemon has no stop-string matching. Halts only on EOS/`<|im_end|>`/max_tokens.
  **Fix:** parse+forward `body.stop`, add daemon-side stop-string match emitting `finish_reason="stop"` (min: stderr-warn so it's not silent).

### Bun single-pipe-without-mutex cluster (availability) [swarm-verified]

- **B-1 — idle-eviction timer drives `send/recv` without the request lock** → races a concurrent request on one pipe, cross-routes acks (consume unload-ack as reset-ack, `generate` to a null-model daemon). **Fix:** acquire the serve lock in eviction, re-validate idle precondition after the wait.
- **B-2 — drain-timeout restart races the machine-wide flock** (`stop()` doesn't await `proc.exited`; `LOCK_EX|LOCK_NB` → respawn hits FATAL-already-running → EOF → serve dies). **Fix:** `await proc.exited` before respawn; retry spawn/ping with backoff.
- **B-3 — foreground `serve` clobbers then deletes the live `serve -d` PID file**, orphaning a VRAM-holding daemon `hipfire stop` can't find. **Fix:** singleton-guard the foreground path; `cleanupPid` unlinks only on read-back-equals-own-pid.
- **B-4 — non-stream daemon-error `Response` built inside `ReadableStream.start()` is discarded** → client hangs on a 200 with empty body until the 255s idle timeout (intended 400/413/500 is dead code). **Fix:** emit the error through the open controller + close it.
- **B-5 — idle-eviction races the request handler's model reload** (request holds lock but `e.generating` still false during reload) → concurrent recv on one stdout, split JSON. **Fix:** eviction also respects a `busy` flag covering the whole lock-held window.
- **B-6 — `hipfire run` HTTP-fallback spawns a local daemon that collides with the running serve's flock** → FATAL → process.exit. **Fix:** when `runViaHttp` fails but `isServeUp()`, retry HTTP instead of spawning local.

---

## Contested (1/3 affirm — human eyeball)

- **C-1 — `body.stream` raw JS truthiness vs OpenAI strict-boolean.** `cli/index.ts:~2623` `if (body.stream)` — `"false"` (truthy) forces SSE; `0`/`""` forces non-stream. Contrast `stream_options.include_usage` which uses `=== true` (`:2097`). Real but low-confidence / client-mis-serialization only. **Fix:** `body.stream === true`.

---

## Falsified by hand-verification (do NOT action as ranked)

- **#5 (raw rank 5, "generate_dflash abort handlers never memset DeltaNet")** — **FALSE for the dominant trigger.** The decode abort handler (`daemon.rs:~5707`) *explicitly* memsets `s_matrices/s_scales/conv_states` + zeros `compact_offset`, with the comment *"stale mid-decode recurrent state from the aborted DFlash run corrupts the next generation (drift → premature EOS)"* — i.e. it **is** the shipped H1 fix, and V4 validated it. The finding's cited lines (4326/4560) point at `load_model_pp`. **Residual sliver (unconfirmed):** the *seed/prefill* abort (`:5426`) clears bookkeeping but does not memset DeltaNet — only a real bug if the seed phase advances *committed* `dn_state` and the next cold prefill doesn't memset. Worth a focused look, but narrow (brief seed-phase window) and not HIGH.

---

## Rejected: 7 (refuted by ≥2/3 skeptics as unreachable / intended / theoretical-only).

## Excluded by design (not hunted)
- Known-fixed: H1–H4, M1–M9, AR double-advance, #388, #389, the 6 teardown leaks.
- Separate track: quant-kernel numerical correctness / AWQ+GPTQ KLD (issue #392).
- Saturated: single-GPU A3B per-token decode coherence (2 prior swarms).
