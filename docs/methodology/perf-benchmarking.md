# Perf benchmarking methodology

The protocol for measuring kernel-level tok/s honestly. Read this
before claiming any win in a commit message; the gates assume you've
followed it.

## Warm the cache and DPM state first

The single biggest cause of "noisy" benches is **measuring a cold
process before the kernel JIT cache and DPM state have warmed up**.
A first run after `cargo build` is 3-7× slower than steady state
because hipcc compiles kernels on first dispatch and the GPU
clocks haven't ramped. **That's not "DPM noise" — that's measurement
error.** Warm first, then measure.

Warmup options (any of these is sufficient):
- `HIPFIRE_DPM_WARMUP_SECS=10` env var on the bench binary
  (built into `bench_qwen35_mq4`, `dflash_spec_demo`, daemon)
- A throwaway warmup run with the same prompt before the timed run
- Use `scripts/probe_commits.sh` (handles warmup automatically)

## The within-session noise band

Once warm, the within-session A/B noise band on gfx1100 (7900 XTX)
is **±1–3%** on a fresh process. **A 3%+ delta is real signal worth
investigating, not noise to wave off.** Earlier versions of this doc
quoted ±10-15% — that figure conflated cold-start overhead with
true noise and let real regressions slip through. Don't repeat the
mistake: if you see a 3% delta, it's a real change. Bisect it.

Sources of REAL drift, ranked by impact (all distinct from the
cold-start overhead above):

1. **Stale build cache**. The speed-gate's `ensure_build()` is a
   no-op when the bench binary already exists. A "stash and re-bench"
   flow leaves the post-change binary in place, so both runs measure
   the same code. Always `rm target/release/examples/<bench>` before
   re-running.
2. **Firmware shadowing**. `/lib/firmware/updates/amdgpu` overrides
   the kernel's bundled firmware; if the dkms-installed firmware
   doesn't match the kernel-installed firmware, you get a SMU IF
   mismatch and ~50% prefill cratering. Fix:
   `sudo mv /lib/firmware/updates/amdgpu /lib/firmware/updates/amdgpu.bak
   && sudo reboot`. Symptoms in `dmesg | tail -40`.
3. **Thermal throttle**. After 10+ minutes of sustained DFlash runs
   with the case closed, the 7900 XTX throttles. Check
   `cat /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input`.
4. **DPM state ON COLD START** (mitigated by warmup). The GPU clocks
   ramp up over the first ~5 seconds of sustained load. This is the
   confound the old "±10-15%" claim was actually measuring. Solved by
   warmup; not a steady-state noise source.

## Cross-process verification

To trust a perf delta as real:

```bash
# 1. Pin to the candidate commit
git checkout <candidate>

# 2. Rebuild from clean
cargo clean -p rdna-compute
rm -f target/release/examples/bench_qwen35_mq4
cargo build --release --features deltanet -p hipfire-runtime \
    --example bench_qwen35_mq4

# 3. Run the gate
./scripts/speed-gate.sh --fast

# 4. Same procedure for the baseline
git checkout <baseline>
# ... (same build + bench)
```

Or via the harness directly:

```bash
./scripts/probe_commits.sh <baseline-sha> <candidate-sha>
```

`probe_commits.sh` does this end-to-end with fresh process per commit
+ a multi-run median. A delta that survives this protocol is real;
one that doesn't probably isn't.

## Daemon-driven perf measurement

The in-process `bench_qwen35_mq4` and `dflash_spec_demo` examples are
the canonical perf tools — they run a 10s `dpm_warmup` (memset loop
that pins the GPU at high sclk/mclk) before the timed gen window, so
their `gen_tok_s` numbers reflect steady-state silicon throughput.

Anything that talks to the daemon (`coherence_probe`, agent CLIs, eval
scripts) used to run *without* DPM warmup, leaving the GPU at a low
power state when the first request arrived. That produced ~5–10%
lower decode-rate measurements than the in-process bench tools. The
gap was especially confusing because `coherence_probe` reported
`tok_s = total_tokens / wall_ms`, which folds TTFT (and any new warmup
time) into the headline number — so short generations can look
catastrophically slow even when steady-state decode is fine.

Two fixes ship together:

- **`HIPFIRE_DPM_WARMUP_SECS=N` is honored by the daemon.** When set
  and positive, the daemon runs `gpu.dpm_warmup(N)` after weight
  upload but **before emitting the `loaded` ack**. The contract
  becomes: `loaded` means daemon is fully ready, including DPM-pinned.
  This ordering is load-bearing: if warmup ran *after* the ack, the
  probe would receive `loaded`, immediately send `generate`, and the
  daemon (still inside the load handler doing warmup) wouldn't
  process the `generate` until warmup finished. From the probe's POV
  that warmup time would fold into the measured TTFT, breaking
  `tok_s = total_tokens / wall_ms`. With warmup-before-ack, the probe
  sees `loaded` only when the daemon really is ready, and TTFT
  measures real prefill alone.
  Default OFF (production load latency unchanged). Recommended `10`
  for perf-bench runs, matching the bench tools.

- **`coherence_probe` reports both probe-derived and daemon-authoritative
  numbers.** Read the daemon ones for perf comparisons.
  - **Daemon-authoritative** (right number for perf): `daemon perf:
    prefill X tok/s (Yms / real ttft) | decode Z tok/s | overall W tok/s`.
    These come from the `done` event the daemon emits — `prefill_ms`
    is the real `forward_prefill_batch` timer, `decode_tok_s` is
    steady-state post-prefill, `ttft_ms` is real first-prefill-token
    latency. Apples-to-apples with `bench_qwen35_mq4 prefill_tok_s` /
    `gen_tok_s`.
  - **Probe-derived** (UX framing only, NOT perf-comparable): `tokens:
    N (probe wall A tok/s, probe gen B tok/s, probe ttft Cms)`. The
    probe sets `ttft` on the first non-synthetic `token` event it
    receives — but the daemon strips `<think>...</think>` content
    by default, so on thinking models (Qwen3.5, Qwen3.6) the first
    visible token only arrives *after* think closes. Probe `ttft`
    therefore folds prefill + entire think phase + `</think>` into
    one number, and `wall tok/s` / `gen tok/s` derived from it look
    catastrophically slow vs in-process bench. They aren't —
    they're just measuring something different (user-perceived
    time-to-visible-content).

  When in doubt: read the `daemon perf:` line, ignore `probe wall/gen`.
  The JSON report carries `daemon_prefill_tok_s`, `daemon_decode_tok_s`,
  `daemon_ttft_ms`, `daemon_tok_s` for downstream tooling.

Use:

```bash
HIPFIRE_DPM_WARMUP_SECS=10 \
  ./target/release/examples/coherence_probe \
    --model some.hfq --prompt-file p.txt --max-tokens 400
```

The expected residual gap between probe `gen tok/s` and bench
`gen_tok_s` after this is ~10% — that's daemon per-token overhead
(JSONL streaming, detokenize, stop-condition checks). Closing it
further is a separate workstream; it is uniform across quant formats
so does not interfere with format ranking.

## eval_hipfire wall-clock is NOT a kernel-speed meter

`eval_hipfire`'s reported `tok/s` (the trailing `(N tok/s)` line) is
**scored-tokens-per-wall-second across the entire eval loop**, not
forward-pass throughput. The eval loop runs, per chunk:

1. `forward_prefill_batch` (prefix, no logit capture)
2. `forward_prefill_batch` (scored region, with hidden-state capture)
3. **`weight_gemv` × `scored_per_chunk` (≈1023) for the lm_head fan-out**
4. `score_position` × scored_per_chunk (CPU-side KLD compute, F32→F16
   block I/O against the ref file)

For non-F16 lm_head dtypes (Q8_0, MQ4G256, …), step 3 issues one GEMV
call per scored position against the full lm_head matrix (vocab=248K
on Qwen3.5). On Q8 that's ~1 GB read per call × 1023 calls = ~1 TB/chunk
of HBM traffic just for lm_head. That dominates the wall-clock and
masks any projection-kernel win. Concrete numbers from `feat/q8-prefill-tier2`
validation on gfx1100 / RX 7900 XT:

| Path | eval_hipfire tok/s | daemon `prefill_tok_s` |
|---|---:|---:|
| q8f16 (Q8 lm_head, T3 WMMA projections) | 27 | 1069 |
| q8f16-f16lm (F16 lm_head batched, T3 WMMA) | 187 | 1069 |
| mq4 (MQ4 lm_head, MQ4 batched projections) | 248 | ~1100+ |

**Rules:**

- **For kernel-speed perf claims**, read `daemon prefill_tok_s` from
  the daemon's `done` event (`coherence_probe`, `coherence-gate.sh`
  matrix rows all expose it), or use `scripts/probe_commits.sh`
  which also uses the daemon. Do NOT cite eval_hipfire `tok/s`.
- **For KLD validation and silent-corruption checks**, eval_hipfire
  is the right tool — its wall-clock is incidental, only the slice-mean
  KLD and PPL matter.
- **For end-to-end "user-visible" throughput** (TTFT + decode), use
  the daemon. eval_hipfire's scoring loop is not representative of
  any production workload.

The lm_head fan-out itself is fixable — see PR #242 (F16 batched
fan-out via `gemm_f16_batched_lmhead`) for the pattern; a parallel
batched Q8 lm_head kernel is currently out of scope per
`docs/plans/q8-fused-prefill-kernels.md §Out of scope`.

## Speed-gate baselines

`tests/speed-baselines/<arch>.txt` records the "ground floor" decode
+ prefill numbers per arch. The pre-commit hook runs
`scripts/speed-gate.sh --fast` automatically when the staged diff
touches kernel / dispatch / forward-pass files. Tolerance is ±5% from
the committed baseline.

If a legitimate change trades a small regression on the baseline arch
for a much bigger win on another arch (rare, but happens), update the
baseline in the same commit:

```bash
./scripts/speed-gate.sh --update-baselines
git add tests/speed-baselines/
git commit
```

The baseline change stays in the same commit so a reviewer sees the
trade-off explicitly. Don't sneak baseline updates into separate
"chore" commits.

**Do not use `--no-verify` to bypass the gate.** A regression that
the gate catches is information you need; bypassing produces a commit
that masks the issue from `git bisect` later. Authorized exceptions
must be explicitly stated in writing by the maintainer for that
specific change.

## Prompt structure matters

Two prompts that tokenize to the same number of tokens but with
different whitespace patterns produce dramatically different DFlash τ.
Same model, same flags, same binary md5:

```
PEP-8 strict (\n\n\n between top-level defs):    27B-3.5  τ=8.07  → 161 tok/s
Single-blank (\n\n between top-level defs):      27B-3.5  τ=9.42  → 184 tok/s
```

A 14% perf swing from a whitespace cleanup is invisible in code
review but catastrophic for benchmarking. **All cross-session perf
comparisons MUST use byte-identical prompts**:

- Embed prompts as committed files (not heredocs in scripts that
  editors reformat).
- Record the prompt md5 alongside the result.
- If you can't verify the prompt md5, treat the comparison as
  unreliable.

The engine collapses `\n{3,}` → `\n\n` at prompt entry by default
(`prompt_normalize=true`). This eliminates the whitespace-variance
source for normal use, but bench scripts that bypass the engine entry
point still need the prompt-md5 discipline.

## Resident-bench mode (avoid per-row 17 GB model reload)

Bench scripts that shell out to `dflash_spec_demo` once per row pay
the full target+drafter H2D on every invocation. On a 27 B mq4 pair
that's ~56 s of model load to measure ~1.5 s of decode — a 24-row A/B
battery is ~25 min wallclock for ~36 s of GPU compute, and
`amdgpu_top` mostly shows the H2D rather than the kernel under test.

Use `--prompts-file <path>` instead of `--prompt`: the binary loads
target+drafter once and runs each manifest row against the resident
pair, with full state reset between rows. Manifest is JSON-lines:

```
{"label":"humaneval-0","prompt":"...","max":16}
{"label":"lru-cache","prompt":"...","max":16}
```

Each row's stderr output is bracketed by `@@@ ROW <i>: <label> @@@`
and `@@@ ROW <i> END @@@` markers so a downstream parser can split
the multi-row stream. Single `--prompt` mode is unchanged
(byte-identical output, no separators).

**Prompt-fixture discipline still applies** (§"Prompt structure
matters"): assemble the manifest at runtime from committed `.txt`
fixtures rather than inlining prompts in heredocs. Either `jq` or
`python3 -c` works — the only requirement is byte-stable output:

```bash
# jq form
jq -nR --arg p "$(cat benchmarks/prompts/humaneval_0_has_close_elements.txt)" \
   '{label:"humaneval-0", prompt:$p, max:16}' >  manifest.jsonl
jq -nR --arg p "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)" \
   '{label:"lru-cache",  prompt:$p, max:16}' >> manifest.jsonl

# python3 form (no jq dependency; see scripts/dflash_bench_resident_smoke.sh)
python3 -c '
import json, sys
for label, path in [("humaneval-0","benchmarks/prompts/humaneval_0_has_close_elements.txt"),
                    ("lru-cache", "benchmarks/prompts/lru_cache_pep8_strict.txt")]:
    print(json.dumps({"label":label, "prompt":open(path).read(), "max":16}))
' > manifest.jsonl

dflash_spec_demo --target T --draft D --prompts-file manifest.jsonl …
```

Or commit the JSONL fixture directly (see
`benchmarks/prompts/longcode_pflash.jsonl` for the shape).

Caveats:
- `--prompts-file` rejects `--cask-sidecar` — `EvictionCtx.eviction_count`
  has no per-row reset, and a cumulative count is silently misleading
  in the FlashCASK report. Run CASK benches as separate per-row
  invocations.
- `--prompts-file` rejects `--pflash` for now; PFlash drafter loading
  and compression state are single-prompt only. Run PFlash benches as
  separate invocations until a per-row reset path exists.
- Cross-row validation: pass the same prompt twice with `--temp 0`;
  the two `DFlash tokens: [...]` lines must match byte-for-byte.
  `scripts/dflash_bench_resident_smoke.sh` is the reference check.
- Per-process flags (e.g. `HIPFIRE_MMQ_MIN_BATCH` cutover, ddtree
  budget/topk, kv-mode) still vary across invocations, not within.
  Group manifest rows by these flags and run one invocation per group.

See issue #173 for the full design.

## DFlash speed gate

`scripts/coherence-gate-dflash.sh` runs the spec-decode coherence
battery: a fixed (model × prompt × spec-mode) matrix that catches
token-attractor regressions (output that passes the perf gate while
emitting `[1734 2357 2733 283 869 1734 2357 ...]` for 1500 tokens).

Three-tier thresholds:

| Tier | Window | Hard fail if |
|---|---|---|
| 1 | First 128 tokens | unique_token_ratio < 0.15 OR max_token_freq > 0.50 |
| 2 | Last 128 tokens | unique_token_ratio < 0.30 OR max_token_freq > 0.50 |
| 3 | Full output | 3-gram repetition density > 50% in second half |

Why both ends: single-token attractors show up in the first 128;
block-level structural loops (5+ token sequences repeating) appear
later. The CASK m-fold + DFlash 2026-04-26 case (τ=8.98, tight
stddev) passed the first-128 gate but emitted 76+ block-loop reps in
the last 1000 tokens.

Any DFlash perf bench that lacks the unique-token + max-frequency
checks AND visual eyeballing of output is unreliable. **Tight stddev
on a spec-decode bench is actively suspicious, not reassuring**:
real acceptance noise is wider; tight stddev correlates with
deterministic attractors.

## What the gates do NOT catch

- Per-lane WMMA mapping bugs (see commit `b7ac66a` — the gfx11 C-mapping
  was silently wrong for 6 weeks while the gates were green).
- Output drift that's coherent but worse on a benchmark task hipfire
  doesn't measure. The coherence gate is unique-token / repetition; it
  doesn't measure HumanEval pass-rate.
- Long-context drift past the prompts the gates exercise.

For correctness on a new arch port: run
`crates/hipfire-runtime/examples/test_kernels.rs` on the target hardware. It
runs each kernel through small golden cases against a CPU reference.
That's the load-bearing correctness gate; the perf and coherence
gates are optimization gates.
