# Plan — issue #173: bench harness, keep daemon alive across rows

> **Revision note (2026-05-06):** rewritten after adversarial reviews
> from glm-5-turbo and Gemini CLI.
> Key changes vs. v1:
> - Path A justification corrected (`hipfire serve` HTTP exists, but
>   request/response schema is too narrow — see §"Why Path B").
> - Per-row reset list is now an exhaustive enumeration (was hand-wavy).
> - `target.kv_cache.compact_offset = 0` and `target_hidden_host.clear()`
>   added (P0 corrections — both required for correctness).
> - Row separator changed from `=== ROW … ===` to `@@@ ROW … @@@` to
>   eliminate any future collision risk with engine-emitted lines.
> - Slice 1 shrunk: only the new flag + a small smoke harness.
>   coherence-gate migration moves to slice 2 (parser rewrite is
>   non-trivial; see §"Coherence-gate migration").
> - `--prompts-file` + `--cask-sidecar` is now a hard error, not a
>   "documented cumulative" caveat.
> - H2D math fixed (34 GB, not 28 GB).

## Problem (from issue)

`benchmarks/scripts/bench_dflash_27b_gfx906.sh` and similar dflash bench
scripts shell out to `dflash_spec_demo` once per (target, draft, prompt,
cutover-setting) tuple. Each invocation cold-loads ~17 GB of target+draft
weights and pays ~56 s of model H2D to measure ~1.5 s of decode. A 24-row
battery is ~25 min wallclock for ~36 s of actual GPU compute, and
`amdgpu_top` mostly shows the model H2D rather than the kernel under test.

Fix: load each (target, draft) pair **once** per process and run the
full row-set against the resident copy.

## Why Path B (extend `dflash_spec_demo`), not Path A (HTTP via `hipfire serve`)

`hipfire serve` exists and exposes `/v1/chat/completions` (Bun TS server,
`cli/index.ts:1280`). The issue's "HTTP API can issue a `POST /v1/completions`
per bench row" is feasible in principle. Path B is preferred because Path A
would require a much wider surface change for this PR's use case:

1. **Request schema is OpenAI-shaped** (messages array). Bench scripts
   need `--no-chatml` (raw prompt), `--ddtree-batched`, `--ddtree-budget`,
   `--ddtree-topk`, `--cask-*`, `--adaptive-b-range`, `--block-size`,
   `--ar-baseline`. Path A would need ~10 new request fields plumbed
   through the Bun→daemon JSON-lines bridge.
2. **Response schema is OpenAI SSE chunks.** No DFlash stats come back
   over HTTP — no `τ`, no `mean_committed`, no `accept_rate`, no
   `DFlash tokens: [...]` list. Path A would need a new response field
   carrying these and parsers updated for the new shape.
3. **`HIPFIRE_MMQ_MIN_BATCH` cutover.** Today it's read per-dispatch
   in `should_use_mmq` (rdna-compute/src/dispatch.rs:147). Flipping
   between resident-bench *invocations* (one per cutover value) is the
   natural fit — works equally for Path A or B.
4. **Existing parsers already understand `dflash_spec_demo`'s stderr.**
   Path B adds a row separator and keeps every per-row line byte-identical.
   Path A would require all parsers to switch to a new format.

If a future PR wants OpenAI-compatible HTTP for benching, that's a
separate piece of work. This PR unblocks the bench-harness pain today
without committing to that refactor.

## Scope

In scope (slice 1, this PR):

1. Add `--prompts-file <path>` flag to `dflash_spec_demo`.
2. Wrap the per-prompt block (tokenize → prefill → decode → report) in
   a loop with explicit between-row state reset (full list below).
3. Print `@@@ ROW <i>: <label> @@@` and `@@@ ROW <i> END @@@` separators
   on stderr in multi-prompt mode only. Single-prompt (`--prompt`) mode
   emits zero new lines — output stays byte-identical to current.
4. Add a small smoke harness `scripts/dflash_bench_resident_smoke.sh`
   that runs a 2-row manifest on the smallest available DFlash model and
   asserts both rows produce non-empty output. Validates the new flag
   end-to-end without depending on a 27 B model.

Out of scope, deferred to slice 2:

- `scripts/coherence-gate-dflash.sh` migration. The detector's
  `re.search(r"DFlash tokens: \[…\]")` and `grep … | head -3` both
  always match row 1; the parser needs a row-aware rewrite, not just a
  call-site change. Best done after slice 1's flag is proven on a small
  model.

Out of scope, separate work:

- `benchmarks/scripts/bench_dflash_27b_gfx906.sh` — does not exist on
  master (lives on PR #158's `feat/gfx906-mmq-redesign`). Should be
  authored on that branch using the new flag.
- `scripts/bench_humaneval_dflash.py`, `scripts/dflash_branch_bench.sh`,
  `scripts/sweep_dflash_full.sh`, `scripts/mq3-mq2-sweep.sh` — same
  fix applies; each is its own follow-up.
- Daemon JSON-lines `done` event additions (`mean_committed`, `mean_B`).
- HTTP wrapping of `hipfire serve` for benching.

## Per-row reset (exhaustive)

Anything per-prompt that isn't covered by `seed_target_hidden_from_prompt
→ target.reset_state(gpu)` (which only zeros `dn_state.s_matrices /
s_scales / conv_states`) must be reset explicitly. The list below is
the canonical reset block:

| Variable / call | Source line | Reason |
|---|---|---|
| `target_hidden_host.clear()` | demo:589 | `seed_target_hidden_from_prompt` *appends* via `extend_from_slice` (speculative.rs:4810). Without clear, row 2's host buffer = row 1 ++ row 2 → wrong slice uploaded by `scatter_hidden_block_to_interleaved`. **Silent corruption.** |
| `target.kv_cache.compact_offset = 0` | llama.rs:2648 | `reset_state` does not touch this. If row 1 triggered CASK eviction, row 2's RoPE phase = `position + compact_offset` is off. **Silent corruption.** Daemon does this explicitly at daemon.rs:812, :2100. |
| `draft_scratch.reset_upload_tracking()` | dflash.rs:488 | Clears `uploaded_target_hidden_rows`, `target_hidden_abs_positions`, `draft_ctx_cached_rows`. |
| Re-run lines 607–621 (scatter + abs_positions seed) | demo:607 | After `seed_target_hidden_from_prompt`, the demo manually scatters and seeds abs_positions. These two stmts are part of the per-row block, NOT the one-time setup. |
| `hidden_rb.reset()` | speculative.rs:1342 | `head=0; written=0`. GPU buffers stay (cheap; stale data unreadable). |
| `stats = SpecStats::new(draft_scratch_b)` | demo:778 | Fresh accept-len histogram and counters per row. |
| `position = prompt_tokens.len()` | demo:664 | Derived from new row's prompt tokens. |
| `emitted = vec![first_token]` | demo:707 | first_token re-derived from logits download per row. |
| `seed_token = first_token` | demo:710 | Same. |
| `accepts_window.clear()` | demo:793 | Rolling τ window — VecDeque from prior row poisons `live_tau`. |
| `window_hashes.clear()` | demo:759 | Loop-break detector HashSet — false-positive risk if a 32-tok suffix matches across rows. |
| `runtime_temp = temp` | demo:753 | Loop-break "temp" mode bumps this. |
| `runtime_repeat_penalty = repeat_penalty` | demo:754 | Loop-break "escalate" mode bumps this. |
| `loop_break_hits/consecutive/escalations/clean_streak = 0` | demo:755–758 | Detector counters. |
| `rng_state = seed | 1` | demo:797 | Determinism for `--temp > 0`. Without re-seed, identical-prompt rows produce different outputs. |
| `ngram_cache` reconstruct | demo:811–820 | Seeded from current prompt tokens (`c.observe_many(&prompt_tokens)`). Carries cross-row n-gram counts otherwise. |
| `pld_matcher` history clear | demo:830ish | If `--pld`, the PldMatcher carries n-gram match history across cycles. |
| `profile_cycle_count = 0; profile_armed = false` | demo:874–875 | Define semantics: "profile each row's first N cycles" matches single-prompt behavior best. |
| `pld_hits = 0; pld_accepted = 0` | demo:838–839 | Per-row PLD stats. |
| `adaptive_b_histogram.clear(); adaptive_b_changes = 0; adaptive_b_cycles_since_change = 0` | demo:960–963 | Adaptive-B scheduler state. |
| `ttft_ms = None` | demo (per row) | Re-set inside loop; `t_decode = Instant::now()` likewise. |
| `per_cycle_wall_us.clear(); per_cycle_api_us.clear()` | demo (per row) | Host-timing per-cycle vectors. |
| `reset_seed_oracle_stats()` | speculative.rs:73 | Already called at demo:954; needs to be inside the loop now. |
| `reset_ddtree_meta_stats()` | speculative.rs:129 | Already called at demo:955; ditto. |
| `hip_bridge::launch_counters::reset()` | hip-bridge/src/ffi.rs:61 | Only matters when `--host-timing`; the report divides cumulative counts by `total_cycles` for "mean per-cycle calls" → row 2's mean is inflated by `row1_cycles / row2_cycles` without reset. |
| `target_snap`, `post_seed_snap`, `path_c_*_snap` | — | DeltaNetSnapshots. Verify whether `spec_step_*` reads-before-writes on cycle 1. If yes, these need explicit invalidation. **Audit during implementation.** |

Counters that are P1-diagnostic and have no reset path today:

- `EvictionCtx.eviction_count` (`triattn.rs:758`, `Cell<usize>`, no reset) — only matters with `--cask-sidecar`. Mitigation: hard-error if `--prompts-file` and `--cask-sidecar` are both set. Cleaner than "documented cumulative."
- `DDTREE_FAST_COUNT` / `DDTREE_SLOW_COUNT` (`speculative.rs:4081–4084`, thread-locals, no reset). Already documented as cumulative in `[ddtree-tape]` line.
- `PATH_C_TOTAL` etc. (`speculative.rs:4557–4564`, same shape).

These three are noted but not blocking — out of scope for slice 1. Add
`pub fn reset()` for each in a follow-up if/when their per-row values
are wanted.

## File-level changes

### `crates/hipfire-runtime/examples/dflash_spec_demo.rs`

CLI surface added:

```
--prompts-file <path>   Mutually exclusive with --prompt and --prompt-file. JSON-lines:
                        {"label":"...","prompt":"...","max":N}
                        max is optional; defaults to global --max.
                        label must not contain '@@@' or newline.
                        Malformed line: log to stderr and continue
                        with next row (don't kill a long battery).
                        Mutually exclusive with --cask-sidecar and --pflash.
```

What stays one-time (outside the per-row loop):

- arg parse (lines 1–360)
- GPU init, draft load, target load, draft scratch alloc, vocab check
  (lines 361–471, ~56 s of H2D — this is what we're amortizing)
- tokenizer load (line 480)
- hidden_rb / snapshots / gdn_tape / ddtree_scratch / verify_scratch
  alloc, target_hidden_host `Vec::with_capacity` (lines 513–590, sized
  to ctx_capacity)

What moves into the per-row loop body:

- prompt encode + ChatML wrap + `eprintln!("prompt:")` (lines 484–511)
- The per-row reset block from the table above
- `seed_target_hidden_from_prompt` (line 595)
- `scatter_hidden_block_to_interleaved` + abs_positions seed (lines 607–621)
- Initial `first_logits` download + `first_token` argmax (lines 693–704)
- Decode setup (`emitted`, `seed_token`, `stats`, etc., lines 707–984)
- Decode loop body (lines 985–1474)
- Report block (lines 1476–1647)

Validation requirements that must hold:

- **Single-prompt mode regression-safety.** Running with `--prompt`
  produces byte-identical stderr and stdout to today's binary. No row
  separators emitted. The `for` loop runs once over a 1-element vec.
- **Determinism across rows.** Manifest = `[promptX, promptX]` with
  `--temp 0` produces byte-identical `DFlash tokens: [...]` lines for
  both rows. Timing differs (JIT warm), token IDs do not. This is the
  acceptance test for the reset block being complete.
- **Multi-prompt mode emits row separators.** `@@@ ROW i: label @@@`
  before each row's output, `@@@ ROW i END @@@` after. Choice of
  `@@@` (vs `===`) eliminates collision with engine's existing
  `=== dflash_spec_demo ===`, `=== BENCH METRICS ===` lines and any
  future `=== … ===` log additions.
- **`--prompts-file` + `--cask-sidecar` is a hard error.** `EvictionCtx
  .eviction_count` has no reset; cumulative counts are confusing and
  silently wrong-looking in reports.
- **Manifest error handling.** Malformed JSON line, missing `prompt`
  field, label containing `@@@` or newline → `eprintln!("[prompts-file]
  skipping line N: <reason>")` and continue. Don't panic and lose the
  battery.

### `scripts/dflash_bench_resident_smoke.sh` (new, slice 1)

~30 lines. Picks the smallest staged DFlash-compatible model from
`$HIPFIRE_MODELS_DIR` (e.g. 0.8B or 4B with matching DFlash drafter),
writes a 2-row JSON-lines manifest with two distinct short prompts
(or two copies of the same prompt for the determinism check), invokes
`dflash_spec_demo --prompts-file …`, asserts:

- exit 0
- both `@@@ ROW 0 END @@@` and `@@@ ROW 1 END @@@` in output
- both rows have non-empty `DFlash tokens: [...]` lines
- (optional) duplicate-prompt determinism: row 0 and row 1 token lists
  match byte-for-byte under `--temp 0`

Skips cleanly if no DFlash drafter is staged (mirrors
coherence-gate-dflash's skip pattern at line 142).

### `scripts/coherence-gate-dflash.sh` (slice 2, NOT this PR)

Migration is larger than v1 of the plan implied:

- The Python detector at lines 170–222 uses
  `re.search(r"DFlash tokens: \[…\]", out)` which returns the **first**
  match. In multi-row output it would always analyze row 1 and silently
  ignore row 2+. Rewrite required: split `out` on `@@@ ROW \d+: \S+ @@@`
  / `@@@ ROW \d+ END @@@` boundaries first, then run detector per row.

- Stats extraction at line 270 uses `grep … | head -3` which always
  yields row 1's stats. Same row-aware split required.
- Output-text extraction at line 295 (`sed -n '/--- OUTPUT ---/,
  /-------------/p'`) matches the **first** OUTPUT fence. Same fix.

- The hard-error counter aggregates across rows. Each row contributes
  independently; one bad row in a group still reports as one hard-error
  in the gate's exit code.
- Grouping: SHORT_TESTS becomes 2 invocations
  (1 dflash group of 2 rows, 1 ddtree-b12 group of 2 rows) vs 4 today.
  Wallclock saving: 2 model reloads avoided × 17 GB ≈ **34 GB H2D
  avoided ≈ ~2 min saved** at sustained ~300 MB/s. (Math fix vs v1's
  "28 GB".)
  `--full` adds b22-k4 and b8-k2 ddtree groups (1 row each), no
  additional savings — they stay as separate invocations. Net: 4 vs 6.

This is enough that it should be its own PR, after slice 1 is merged
and the flag is proven on the smoke harness.

## Validation (slice 1)

1. `cargo build --release --example dflash_spec_demo --features deltanet`
   compiles clean.
2. `--prompt` mode regression: pick any small staged model, run with
   `--prompt "Hello" --max 16`, capture stderr+stdout. Diff against the
   pre-change binary's output. Must be byte-identical.
3. Determinism: 2-row identical-prompt manifest with `--temp 0`, both
   `DFlash tokens: [...]` lines must match byte-for-byte. This is the
   linchpin test for state-reset completeness.
4. Different prompts: 2-row manifest with distinct prompts, both rows
   complete and produce sensible-looking output (eyeball decode).
5. `scripts/dflash_bench_resident_smoke.sh` exits 0 on whatever DFlash
   model is staged; skips cleanly otherwise.
6. `--prompts-file` + `--cask-sidecar` → exit 2 with clear error message.
7. Wallclock check: invoking the binary once with a 2-row manifest is
   measurably faster than two separate invocations. Should save one
   target-load worth of wallclock (~30–60 s on the smallest staged
   target).

## Risks remaining

- **DeltaNet snapshot reset semantics not yet audited.** If
  `spec_step_*` reads `target_snap` etc. before writing them on cycle
  1, row 2's cycle 1 sees row 1's leftover snapshot. Mitigation: trace
  read/write order in `spec_step_dflash` during implementation; if
  read-before-write, add explicit invalidation to the reset block.
- **PldMatcher state**: I don't yet know whether it's stateless across
  cycles or carries match history. The reset list assumes the latter
  (clear it). Verify during implementation; if stateless, the line is a
  no-op and the entry can be deleted.
- **First-row JIT warm-up bias.** The smoke test's wallclock comparison
  needs to account for `.hipfire_kernels/` cache state — if it's cold,
  row 1 pays JIT cost row 2 doesn't. Solution: run the smoke twice
  (pre-warm), report only the second. The flag's correctness doesn't
  depend on this — just the wallclock claim does.
