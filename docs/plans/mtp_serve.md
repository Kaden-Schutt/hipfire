# Wiring native MTP into `hipfire serve` — dual-source (bundle + sidecar)

**Status:** plan (rev 2 — incorporates adversarial review findings from
glm5/gemini/claude). **Branch:** `mtp-kevin`. **Prereq landed:** AWQ-aware
lm_head rotation fix (`llama::rotate_x_mq_batched_for` across all MTP
dispatch sites, commit 6bd2af2e — see `docs/plans/mtp_bug.md`).

## Goal

Let `hipfire serve` / `hipfire run` use native MTP speculative decode, loading
the MTP head from **either**:

1. a **separate `.mtp` sidecar** referenced like a DFlash `draft` — **no trunk
   re-download** (pull the ~515 MB head next to the existing `.mq4`), or
2. a **bundled `.mq4-mtp`** (trunk+head in one file, trailer-detected) — one file
   for users who prefer it.

Both must coexist with the existing DFlash-draft path and with plain AR. Existing
models keep working untouched (no trailer + no sidecar ⇒ AR/DFlash as today).

## Non-goals (v1)

- pp>1 / multi-GPU MTP (guard to pp=1, like DFlash).
- Replacing DFlash. MTP and DFlash are alternative spec paths; pick one per load.
- Re-tuning the kernels. This is plumbing around the working `mtp_spec` path.
- Sampling + p_min simultaneously. The K-chain sampling path uses `sample_top_p`;
  p_min uses `topk_logsumexp`. They don't compose — caller picks one or the
  other (already enforced at `mtp_spec.rs:1374`).

## What already exists (do NOT rebuild)

- **Working, coherent decode path:** `mtp_spec::spec_step_mtp` (greedy) and
  `spec_step_mtp_compressed_serial` (cvs head + p_min early-exit + **full
  sampling support** via GPU-side residual acceptance), validated on gfx1151
  post-AWQ-fix (LRU/merge_sort/agentic coherent; τ 2.4–3.2).
- **Sampling infrastructure in `spec_step_mtp_compressed_serial`:**
  `MtpSamplingConfig` (temp/top_p/top_k/min_p), `set_sampling`, GPU-side
  `sample_top_p` for draft chain + bonus, `softmax_prob_gather` for
  p_target/p_draft, residual acceptance loop (lines 1664–1759). Already
  exercised in `mtp_only_demo` with `--temp > 0`.
- **Reference integration:** `crates/hipfire-runtime/examples/mtp_only_demo.rs` —
  the exact load → `MtpSpecState::new_for_slot_with_kv_mode` → prefill →
  `set_sampling`/`set_p_min`/`max_n` → per-cycle `spec_step_*` → commit/EOS
  loop the daemon must port.
- **Both loaders:** `mtp_head::detect_bundled_mtp_offset` (trailer magic
  `HFBNDMTP`) + `load_mtp_head_bundled` for the bundle; the standalone sidecar
  loader for `.mtp`. The head's config carries `n_embd`/`vocab_size`/`rope_theta`
  for a trunk-compat check.
- **Producers:** `mtp_extract` (`--vocab-sidecar cvs.json` → compressed `.mtp`);
  `mq4_merge_mtp` (`--trunk --mtp` → `.mq4-mtp` bundle).

## MTP head resolution (daemon load) — the dual-source core

On `load`, after the trunk is loaded, resolve the MTP head in priority order:

```
1. params.mtp_head (explicit sidecar path)         → load_mtp_head(path)
2. registry/CLI-provided mtp.file sidecar          → load_mtp_head(path)   (== 1)
3. trunk file has a bundled trailer                → detect_bundled_mtp_offset(trunk)
                                                        → load_mtp_head_bundled(trunk, off)
4. none                                             → no MTP (AR / DFlash as today)
```

Implications:
- **No re-download:** paths 1–2 leave the trunk `.mq4` byte-identical; the user
  pulls only the small `.mtp`. This is the DFlash distribution model.
- **One-file convenience:** path 3 is opportunistic — `serve foo.mq4-mtp` just
  works, zero config.
- **Backward compatible:** unchanged `.mq4` files hit path 4.

**Compat guard (mandatory):** a sidecar `.mtp` can mismatch its trunk (wrong
arch/dims) and silently produce garbage. On load, verify the head's `arch_id ==
21`, and `n_embd`/`vocab_size`/`rope_theta`/`n_rot` match the trunk config;
degrade to AR with a loud warning on mismatch (don't hard-error: a typo'd
sidecar path shouldn't prevent the trunk from loading — see §Error policy).
Also gate: trunk arch_id ∈ {Qwen3.5/3.6 family}, `pp == 1`. Bundled heads are
co-produced so always match, but check anyway.

**Future hardening (Phase 2):** add a lineage/model-identifier tag in the
`.mtp` header and a load-time numerical self-check (one probe forward, assert
head argmax agrees with trunk AR argmax within tolerance). Structural-only
validation is how the AWQ bug shipped; the probe catches weight-distribution
mismatch that dims alone won't.

**Mutual exclusion:** MTP and DFlash `draft` are both spec paths. If both are
requested, error (mirror the existing cask+draft guard at daemon.rs:637). Add
a `mtp_mode` override (off/auto/on) so operators can explicitly choose — DFlash
has `dflash_mode`; MTP needs the same. Default: `auto` (use MTP when head is
present, DFlash when draft is present). Log which spec path is active on every
generation.

## Daemon changes (`crates/hipfire-runtime/examples/daemon.rs`)

### State & loading

1. **`LoadedModel` state (~365–426):** add `mtp: Option<MtpState>` alongside
   `dflash: Option<DflashState>`, where `MtpState` holds the `Qwen35MtpHead`,
   `MtpSpecState`, and resolved config (`max_n`, `p_min`, `compressed`,
   `kv_mode`).

2. **`load_model` (1483):** add `mtp_head: Option<&str>` param; implement the
   resolution above (sidecar load OR bundled-trailer detect); allocate
   `MtpSpecState::new_for_slot_with_kv_mode`; run the compat guard. Mirror the
   DFlash branch at 1532. **VRAM accounting:** update `min_vram_gb` to include
   the MTP head (~515 MB cvs sidecar; 0 additional for full-vocab mode since it
   reuses the trunk's lm_head). Fail early if trunk + MTP head + KV exceeds
  available VRAM.

3. **Param parsing (load handler, 574–658):** read `params.mtp_head` (path),
   `params.mtp_k` (→ `max_n`), `params.mtp_p_min`, `params.compressed`
   (default: infer from whether the head carries `lm_head_draft`), `mtp_mode`
   (off/auto/on). Mirror `draft`/`dflash_mode`/`kv_mode` parsing.

4. **Unload lifecycle:** `unload_model` must free MTP GPU state. DFlash has
   `df.take()` → free at daemon.rs:2230. MTP needs `mtp.take()` in the same
   block. One line, but must not be forgotten — `MtpSpecState` holds
   verify_hidden, verify_logits, mtp_kv, etc.

### Prefill

5. **`prev_hidden` seeding:** MTP's first cycle needs the trunk's post-output-norm
   hidden for the last prompt token. After the AR prefill loop completes,
   `scratch.tmp` contains the last token's post-norm hidden. Copy it into
   `MtpSpecState.prev_hidden` (one device-to-device memcpy, ~10 lines).

   Note: this is NOT the DFlash `hidden_rb` ring buffer (that captures
   per-layer rows for the drafter). MTP only needs the final output-norm
   hidden — simpler. The DFlash prefill's `seed_target_hidden_from_prompt`
   (daemon.rs:2662) re-runs a full per-token prefill; MTP does NOT need that.

### Decode loop

6. **Decode-loop dispatch:** the generate loop branches `if let Some(df) =
   m.dflash && temp <= 1e-6` → DFlash path (daemon.rs:3497–3539), else AR.
   Add a third branch: `else if let Some(mtp) = m.mtp` → MTP path.

   Unlike DFlash (greedy-only in serve), MTP supports both greedy AND
   sampling. The gate is:
   - `temp > 0` or `repeat_penalty != 1.0`: call `set_sampling` on the
     MtpSpecState, then enter the spec loop. The sampling path uses residual
     acceptance, NOT greedy argmax-match.
   - `temp == 0 && repeat_penalty == 1.0`: greedy path (current default,
     unchanged).
   - `p_min > 0 && temp == 0`: greedy + p_min early-exit (current behavior).

   `spec_step_*` returns `committed: Vec<u32>`, `accept_count`, `hit_eos`,
   `advance` — feed `committed` through the daemon's existing streaming
   (`token` events), stop-sequence, and EOS handling.

7. **Repeat penalty on trunk verify:** when `repeat_penalty > 1.0`, the trunk
   verify logits must be penalized before argmax (otherwise MTP acceptance
   silently diverges from AR output). Pattern already exists in DFlash at
   `speculative.rs:3034–3057`:
   - Download full verify logits `(K+1) × vocab × 4` bytes to host.
   - Per-row `apply_repeat_penalty(row, prev_committed, window, penalty)`.
   - Argmax the penalized row on host.
   - Cost: ~2.4 MB D2H for K=3 vocab=152K = ~0.2 ms; negligible vs ~80 ms
     cycle.

   Implementation: thread `repeat_penalty`, `repeat_window`, `prev_committed`
   through the three `spec_step_*` variants (~90-120 lines total). When
   `repeat_penalty == 1.0`, use the existing GPU-side batched argmax (no
   change). When `repeat_penalty > 1.0`, take the CPU download+penalty+argmax
   path.

   The MTP draft chain (`sample_top_p` at mtp_spec.rs:1465) currently
   hardcodes `repeat_penalty=1.0, repeat_window=0`. This is correct — the
   draft is a suggestion, not an authoritative answer; penalty is only needed
   on the trunk verify. Leave draft chain penalty at 1.0.

8. **Batch-commit streaming semantics:** MTP commits 1–K+1 tokens per cycle.
   The daemon's streaming layer must handle multi-token batches correctly:
   - **Tool-call boundaries:** `<|tool_call|>` may land at any position in
     `committed`. Scan the full vec; if a tool call fires at committed[2] of 5,
     suppress tokens 3–4.
   - **Stop sequences:** scan `committed` for stop strings; truncate at first
     match.
   - **EOS:** any EOS in `committed` terminates; subsequent tokens are dropped.
   - **`max_tokens` truncation:** if the committed batch would exceed the
     remaining token budget, truncate to `max_tokens - generated_so_far` and
     replay only the accepted prefix in KV.
   - **`max_think_tokens` enforcement:** the DFlash path (daemon.rs:2882) scans
     decoded text for `<think`/`</think` boundaries. MTP must do the same.
     Until implemented, gate MTP behind the `budgeted_thinking_needs_ar` guard
     (daemon.rs:3506).

### Error policy

9. **Degrade vs hard-fail:**
   - **Sidecar load failure / compat mismatch:** degrade to AR with a loud
     warning. Don't prevent the trunk from loading for a bad sidecar path.
   - **`spec_step_mtp` failure mid-generation:** propagate the error to the
     client. For greedy+penalty path, any failure is a real bug (not sampling
     variance), so surfacing it is correct. For sampling path, a transient
     failure could fall back to AR for the rest of the generation.

### hipGraph & eviction

10. **hipGraph:** MTP v1 runs graph-off. The variable advance (1..K+1 tokens/
    cycle) + DN snapshot/rollback doesn't fit a static graph. DFlash already
    has graph-off paths; this is consistent. Graph capture for MTP is Phase 2+.

11. **KV eviction cadence (CASK/TriAttn):** MTP verify writes `K+1` KV slots
    then rolls back to `accepted+1`. The physical position counter (`m.seq_pos`)
    must reflect the rolled-back position, not the peak write. Follow DFlash's
    pattern: `position += step.accepted + 1` (daemon.rs:2908), then fire
    eviction at the rolled-back position. The DN snapshot/restore already
    handles the state rollback; eviction just needs correct placement.

## CLI / registry changes

12. **Registry (`cli/registry.json`):** add an `mtp` sidecar field per model,
    mirroring the existing `triattn` field on `qwen3.6:27b` (line 12):
    ```json
    "qwen3.6:27b": { ..., "triattn": {...},
      "mtp": { "file": "qwen3.6-27b-cvs.mtp", "k": 3, "p_min": 0.65 } }
    ```
    Add a `pull` entry so `hipfire pull` fetches the sidecar.

13. **Plumbing (`cli/index.ts`):** resolve/download the `mtp.file`, and add
    `params.mtp_head` (+ `mtp_k`/`mtp_p_min`/`compressed`) to the daemon `load`
    message — exactly how `draft`/`triattn` are passed today.

## Distribution model (answers "do users re-download?")

| mode | producer | user action | trunk re-download |
|---|---|---|---|
| **sidecar** (default) | `mtp_extract … --vocab-sidecar` → `.mtp` | `hipfire pull <model>` fetches the small `.mtp` next to the existing `.mq4` | **No** |
| **bundle** (opt-in) | `mq4_merge_mtp --trunk --mtp` → `.mq4-mtp` | download/build the one-file bundle | Yes (new file) |

Recommend the **sidecar as the default shipped form** — it's small (~515 MB cvs),
re-uses the trunk, and matches the DFlash model users already understand. Offer
the bundle for air-gapped / single-file deploys.

## Defaults & corpus matching (from this session's benchmarks)

- **K (max_n):** per-arch default. gfx1151 (bw-limited) was fastest at **K=2**;
  gfx1100 (deploy target) / vLLM favor **K=3**. Default K=3 for the gfx1100 line,
  expose `params.mtp_k`. Precedence: registry value → arch default → user
  override.
- **p_min:** 0.65 (compressed-serial early-exit), expose to override. Mutually
  exclusive with `temp > 0`.
- **compressed:** on when the head carries `lm_head_draft` (cvs head). Default:
  `spec_step_mtp_compressed_serial` always (it handles both compressed and
  full-vocab heads; plain `spec_step_mtp` uses a lossy K-step chain with worse
  τ).
- **KV mode for MTP head:** Q8 (default from `MtpSpecState::new_for_slot`).
  Expose `params.mtp_kv_mode` override.
- **Corpus must match the serve workload.** Measured: a code-corpus head wins on
  code prompts, an agentic/workload-corpus head wins on agentic — the cvs corpus
  *is* the τ lever (see `mtp_bug.md` benchmark matrix). Ship a head whose corpus
  matches the model's intended workload, or document building a custom one
  (`scripts/traffic_dump_to_corpus.py` + `build_mtp_vocab_sidecar.py` +
  `mtp_extract`). **Note:** the default v1 corpus ingests
  `benchmarks/prompts/lru_cache_pep8_strict.txt`, so never bench corpora
  against that prompt (overfit).

## Coherence gate

`scripts/coherence-gate.sh` exercises only DFlash today. **Add a native-MTP row**
(sidecar + bundle) before shipping — load via the daemon, assert coherent (no
attractor/CJK leak/τ collapse). Use the same 4 prompts as
`coherence-gate-dflash.sh`. A test `.mtp` head must be distributable with the
gate (it skips absent models). Spec-decode change ⇒ also attractor checks on
the MTP output.

**Gate both greedy AND sampling paths.** The sampling path has residual
acceptance which can hide bugs (low τ looks like "draft disagreement" not
"corruption"). Run at least one sampling-mode gate with temp=1.0.

## Phasing

- **Phase 1 (MVP):** daemon items 1–6, 8 (batch-commit streaming), 10 (graph-off)
  + compat guard + bundle auto-detect (path 3) + sidecar path (1). Greedy AND
  sampling (wiring is trivial — just call `set_sampling`). Repeat penalty on
  trunk verify (item 7). Manual `params.mtp_head`. Coherence-gate row (greedy
  + sampling). `mtp_mode` override (item in mutual exclusion).
  → `serve foo.mq4-mtp` and `load {mtp_head: "x.mtp"}` both work for any
  temperature / repeat_penalty.

- **Phase 2:** registry `mtp` field + `hipfire pull`/`index.ts` plumbing (12–13)
  so `hipfire serve qwen3.6:27b` auto-uses the sidecar. Per-arch K default.
  Lineage tag + probe-forward self-check in compat guard. Budget-alert nudge
  and n-gram loop detection (deferred, same as DFlash today). `max_think_tokens`
  enforcement in MTP streaming (currently gated to AR).

- **Phase 3:** p_min + sampling composability (currently mutually exclusive).
  Corpus-per-workload head selection. hipGraph capture for MTP verify+replay.

## Sampling implementation detail

`spec_step_mtp_compressed_serial` already supports temp>0 via the residual-accept
rule (`MtpSamplingConfig`, `set_sampling`, GPU-side `sample_top_p` + gather).
The daemon wiring is:

```rust
// In generate(), MTP branch, before the spec loop:
if temp > 1e-6 || repeat_penalty > 1.0 {
    mtp.state.set_sampling(
        MtpSamplingConfig { temp, top_p, top_k: 20, min_p: 0.0 },
        rng_seed,
    );
}
```

Then call `spec_step_mtp_compressed_serial` as usual. When `sampling.is_greedy()`
is true (temp=0), the function uses the existing argmax path. When not, it uses
residual acceptance. No branching at the call site.

Repeat penalty (item 7) is separate: it applies to trunk verify logits
regardless of whether the path is greedy or sampling. The `spec_step_*`
functions need the penalty params threaded through for the verify step only.

## Risks / open questions

- **Silent trunk/head mismatch** → garbage. The compat guard (arch_id + dims +
  rope) is mandatory, not optional. Structural-only validation is necessary but
  not sufficient — a probe-forward self-check (Phase 2) catches the rest.
- **VRAM:** sidecar head ~515 MB (cvs) on top of the trunk; check `min_vram_gb`.
  Full-vocab mode uses the trunk's existing lm_head → 0 additional VRAM.
- **gfx1151 is near breakeven** (bw-limited); the real tok/s win is on gfx1100.
  Don't gate the feature on gfx1151 numbers.
- **Bundle staleness:** if the trunk is re-quantized, the bundled head must be
  re-spliced; the sidecar decouples this (re-extract just the `.mtp`).
- **GPU arch matrix:** MTP works on all supported archs but the τ/speedup is
  only meaningful on gfx11+. On gfx10/gfx906, batched verify falls back to
  per-token — correct but slow. The plan should state the full arch matrix.

## Telemetry

Surface per-request MTP metrics so operators/users can confirm the spec path is
active and helping:
- `spec_path`: "mtp" / "dflash" / "ar"
- `mtp_tau`: average tokens accepted per cycle
- `mtp_k`: effective K used
- `mtp_sampling`: true/false

Emit as part of the `done` event JSON, matching DFlash's existing telemetry
pattern.

## Review trail

Three adversarial reviews were produced 2026-05-26:
- `docs/mtp_serve_plan_rev_glm5.md` — primary review with cross-review synthesis
- `mtp_serve_plan_rev_gemini.md` — Gemini CLI review
- `mtp_serve_plan_rev_claude.md` — Claude review

Key findings incorporated in this revision:
| Finding | Source | Disposition |
|---|---|---|
| Sampling should be P1, not P3 | glm5 D1, claude §1 | **Accepted** — sampling already exists in code; daemon wiring is trivial |
| Repeat penalty diverges verify argmax | gemini §3.1, claude §1 | **Accepted** — item 7, pattern from DFlash speculative.rs:3034 |
| Multi-slot concurrency risk | gemini §2.1, claude §8 | **Rejected** — daemon is single-threaded (JSONL stdin/stdout) |
| DFlash has sampling, MTP greedy is regression | claude §1 | **Partially wrong** — DFlash in daemon is also greedy-only (temp≤1e-6 at 3508) |
| hipGraph capture needed | claude §2 | **Accepted, deferred** — item 10, graph-off for v1 |
| KV eviction cadence under rollback | claude §2 | **Accepted** — item 11 |
| prev_hidden seeding is hard | claude §4 | **Wrong** — one d2d copy after prefill, ~10 lines |
| Compat guard needs lineage tag + probe | claude §3 | **Accepted, Phase 2** — structural guard for v1, probe-forward in P2 |
| `mtp_mode` serve-time override | claude §6 | **Accepted** — part of mutual exclusion section |
| Batch-commit stop/tool-call/EOS handling | glm5 C1, gemini §2.5 | **Accepted** — item 8 |
| Unload lifecycle missing | glm5 C4 | **Accepted** — item 4 |
| `budgeted_thinking_needs_ar` must gate MTP | glm5 C2 | **Accepted** — item 8 |
| Degrade vs hard-fail policy | claude §5 | **Accepted** — item 9 |
| Telemetry missing | claude minor | **Accepted** — new §Telemetry |
