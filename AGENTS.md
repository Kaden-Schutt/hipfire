# AGENTS.md — guide for agents testing hipfire v0.1.9-alpha

**Audience:** agents (or humans) running smoke / perf / correctness
tests on hipfire v0.1.9-alpha — particularly the production-ready MQ3
sub-4-bit Magnum Quant family, the DFlash MQ3 cross-quant matrix, and
the existing DFlash draft pull / prompt-shape adaptation paths
inherited from v0.1.8.

**Companion docs:** [`CLAUDE.md`](CLAUDE.md) holds project-wide rules
(non-negotiable hard rules, e.g. coherence-gate is the canonical gate).
This file holds the *testing playbook* — how to verify v0.1.9-alpha
works, what to measure, what counts as pass/fail.

**v0.1.9-alpha default behavior to be aware of:**
- **MQ3 is production on gfx11** (`gfx1100/1101/1102/1150/1151`) and
  gfx12 (`gfx1200/1201`). On gfx10 / gfx906 / gfx94x, MQ3 weights still
  load and run via per-token GEMV fallback — correct, just slower
  prefill. MoE/A3B + MQ3 is refused at load time (no MoE-branched WMMA
  path).
- **MQ2 is refused by default.** The quantizer requires
  `--format mq2 --i-know-this-is-broken` to opt in. Severe quality
  cliff confirmed; Lloyd-Max MQ2/MQ3 (qt=19/20) is the path forward.
- **`dflash_mode=off` default carries over from v0.1.8.** Any test
  exercising DFlash still needs `hipfire config set dflash_mode auto`
  or `HIPFIRE_DFLASH_DRAFT=<path>` first.

---

## 0 · Hard rules from CLAUDE.md (always apply)

1. **Coherence-gate-dflash is the canonical correctness gate.** Quality-
   gate.sh is deprecated — its byte-exact baselines drift faster than
   the engine evolves. Run `./scripts/coherence-gate-dflash.sh` after
   any change touching kernels, quant formats, dispatch, fusion,
   rotation, rmsnorm, or the spec-decode path.
2. **Prompt structure dictates τ.** One newline character can swing τ
   by 17%. Any tok/s comparison across sessions, agents, or commits
   MUST use **byte-identical prompts**. Embed prompts as committed
   files (`benchmarks/prompts/*.txt`); record the prompt md5 alongside
   results. Whitespace cleanups in scripts are forensic landmines.
3. **Tight stddev on a spec-decode bench is SUSPICIOUS, not reassuring.**
   Real acceptance noise is wider. Always eyeball the decoded output
   when τ comes back unusually high — single-token attractor failures
   pass every statistical gate as fake wins.
4. **Never store canonical bench prompts under `/tmp/`.** /tmp gets
   wiped on reboot. Use `benchmarks/prompts/`, `~/.hipfire/datasets/`,
   or a heredoc inside a committed script.
5. **No grep / find / glob inside `exec:bash`.** Use the `Grep` tool
   directly, or `exec:nodejs` with `execSync('rg -n PATTERN')`.
6. **`scripts/install.{sh,ps1}` copy the whole `cli/` directory recursively
   and prune dev/test artifacts by pattern.** New `.ts` files in `cli/`
   are auto-installed — no install-script edit required. Tests must
   follow `*.test.ts` / `test_*.ts` / `bench_*.ts` naming so the prune
   step excludes them; if you add a runtime helper that *looks* like a
   test name, rename it. The previous per-file enumeration grew stale
   silently after PR #129 (issue #163, naive fix #165, structural fix
   in this rule's enforcing PR).

---

## 1 · Setup (one-time)

### Pull the model + draft you want to test

Targets and drafts are independent pulls — drafts auto-discover their
target by filename when the daemon loads:

```bash
# 27B Qwen 3.5 (the canonical perf-test target):
hipfire pull qwen3.5:27b           # 15 GB target
hipfire pull qwen3.5:27b-draft     # 0.92 GB DFlash draft

# 27B Qwen 3.6 (refresh):
hipfire pull qwen3.6:27b           # 15 GB target
hipfire pull qwen3.6:27b-draft     # 0.92 GB DFlash draft

# 9B Qwen 3.5 (smaller, faster sanity-check):
hipfire pull qwen3.5:9b            # 5.3 GB target
hipfire pull qwen3.5:9b-draft      # 0.55 GB DFlash draft
```

Files land at `~/.hipfire/models/<canonical-name>` matching the
daemon's auto-discovery pattern (`qwen3{ver}-{size}-dflash-{quant}.hfq`).
**Do not rename.** Renaming breaks the auto-discovery and DFlash falls
back to AR silently.

### Verify md5s after pull (paranoid mode)

```
qwen35-9b-dflash-mq4.hfq    590f35403cd7f1d634945233234a12b7  557 MB
qwen35-27b-dflash-mq4.hfq   7b6df2a4ee1c8d933f0a52e187d1860b  919 MB
qwen36-27b-dflash-mq4.hfq   ecc64877dfe0a1312b6f4066c3920128  919 MB
qwen3.6-27b.mq4             9a6acdc49bcaa6a7b52ac161444cb769   15 GB
```

Any mismatch = re-pull or report.

### Build from source (if you're on a dev branch)

```bash
cargo build --release --features deltanet \
  --example daemon \
  --example dflash_spec_demo \
  --example encode_prompt \
  --example run \
  -p engine
```

---

## 2 · What v0.1.9-alpha added (test surface)

### A. MQ3 production (sub-4-bit Magnum Quant)

The headline of v0.1.9-alpha. MQ3 = FWHT-rotated 3-bit weight format,
104 B/group (3.25 bpw vs MQ4's 4 bpw at 136 B/group). Three new things
are now wired:

- **K4-unrolled GEMV decode + fused residual** on gfx1100. Decode
  matches MQ4 within 2% (9B 141 tok/s vs MQ4's 128.7).
- **WMMA prefill family** (`gemm_qkvza/qkv/gate_up/residual hfq3`)
  closing the 17× prefill gap that gated ship. Arch-gated to gfx11
  wave32 WMMA. gfx12 K4 variant ships in the same release.
- **DFlash cross-quant matrix.** MQ3↔MQ3, MQ3↔MQ4, MQ4↔MQ3 all valid
  for dense models. MoE/A3B + MQ3 still refused at daemon load.

Sweep harness for MQ3 quality + perf:
```bash
./scripts/mq3-mq2-sweep.sh   # 4-prompt × 5-model bench, md5-stamped
```

### B. Cache-invalidation lifecycle

`Gpu::unload_model` now drains `mmq_screen_cache` + `fp16_shadow_cache`
and tears down captured hipGraphs (verify, replay, AR forward). Three
Codex stop-time follow-ups, all pointer-keyed cache silent-corruption
class. Smoke test: rapid `hipfire serve` model swap loop should NOT
emit garbage on the new model's first decode.

### C. Defensive `parseToolCalls` (#111 stopgap)

Three known MQ4 attractor malformations are repaired before the
OpenAI shape returns: spec form, flat form, XML-tag corruption.
Token-attractor root cause (calibration retrain) deferred. Smoke
test: tool-calling prompt against `qwen3.5-9b.mq4` should never
return raw `<tool_call>` text in `message.content`.

### D. Inherited from v0.1.8 (still load-bearing)

- **Phase 1: prompt-shape adaptation — DEFAULT ON (2026-04-26)**

Engine-side `\n{3,}` → `\n\n` collapse before tokenize, eliminating the
rare BPE token 1358 (`\n\n\n`) in favor of HOT token 271 (`\n\n`) on
Qwen3.5/3.6 vocab.

**Default ON since 2026-04-26** — empirical 199 tok/s on 27B-3.5 LRU
DFlash (vs 159 with opt-out). The original v0.1.8-alpha ship had this
opt-in; it was promoted to default after the 2026-04-26 perf-regression
recovery confirmed +24% τ with zero correctness cost (commit 9a2c667).

To **opt out** (rare — only when raw `\n{3,}` whitespace is semantically
load-bearing):

- Env: `HIPFIRE_NORMALIZE_PROMPT=0`
- TUI: `hipfire config set prompt_normalize false`
- Per-model: `hipfire config qwen3.5:27b set prompt_normalize false`

**Expected lift over OPT-OUT baseline:** +14% to +27% tok/s on PEP-8-style
code prompts that contain `\n{3,}` patterns. Zero effect on prompts
without those patterns.

**Verify:** see §3 prompt-shape A/B test.

### B. Token heat diagnostic

`HIPFIRE_PROMPT_TOKEN_HEAT=1` triggers `Tokenizer::dump_prompt_heat()`
at every encode site. Output goes to stderr (pretty) or stdout (JSON
when `HIPFIRE_PROMPT_HEAT_JSON=1`).

Standalone tool: `./target/release/examples/encode_prompt MODEL.hfq
PROMPT.txt --heat`.

### C. EOT-stop fix

Daemon, run, and dflash_spec_demo now stop on `<|endoftext|>` token,
not just `<|im_end|>`. The Fibonacci-attractor loop in raw-text DFlash
is killed.

### D. DFlash drafts on HuggingFace

Three new HF endpoints (uploaded 2026-04-25, schuttdev account):
- `schuttdev/hipfire-qwen3.5-9b/qwen35-9b-dflash-mq4.hfq`
- `schuttdev/hipfire-qwen3.5-27b/qwen35-27b-dflash-mq4.hfq`
- `schuttdev/hipfire-qwen3.6-27b/qwen36-27b-dflash-mq4.hfq`

Plus the 3.6 27B target itself: `schuttdev/hipfire-qwen3.6-27b/qwen3.6-27b.mq4`.

Pullable via `hipfire pull qwen3.{5,6}:{9b,27b}-draft` and
`hipfire pull qwen3.6:27b`.

---

## 3 · Smoke tests (run these to validate)

### 3.1 — Fresh-process bench harness

Always run benches in a fresh process. Within-session A/B is noisy on
gfx1100 (±10–15 % drift from DPM/thermal state). For tight measurements:

```bash
# Use HIPFIRE_VERIFY_GRAPH=0 if you want deterministic measurements
# (graph capture adds 1.5-3% jitter; OFF gives 0.1% spread).
```

### 3.2 — Prompt-shape A/B test (Phase 1)

```bash
# A: PEP-8 prompt, normalize OFF (un-fixed)
./target/release/examples/dflash_spec_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --prompt "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)" \
  --max 120 --ctx 2048 --kv-mode asym3 --no-adaptive-b --no-chatml

# B: same prompt, normalize ON
HIPFIRE_NORMALIZE_PROMPT=1 ./target/release/examples/dflash_spec_demo ...
```

**Expected delta on 27B-3.5:** ~161 → ~199 tok/s (+24-27%), τ 8.07 → 10.36.
Run each ≥3 times in fresh processes. Median should land in the
expected range. Anything more than ±10% from the published numbers
is a regression — investigate before claiming a result.

### 3.3 — HumanEval/53 single-prompt peak

The `def add(x, y)` prompt is the canonical peak case (we beat 207
tok/s here, vs. Lucebox's RTX 3090 demo peak):

```bash
PROMPT=$(python3 -c "import json; print([json.loads(l) for l in open('/home/kaden/.hipfire/datasets/HumanEval.jsonl')][53]['prompt'])")
HIPFIRE_NORMALIZE_PROMPT=1 ./target/release/examples/dflash_spec_demo \
  --target ~/.hipfire/models/qwen3.5-27b.mq4 \
  --draft ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
  --prompt "$PROMPT" \
  --max 120 --ctx 2048 --kv-mode asym3 --no-adaptive-b --no-chatml
```

**Expected:** 5-run median 212.4 tok/s τ=10.90, 4/5 runs above 207.
If your median is below 200 or τ below 9.0, something has regressed
— open an issue with: GPU model, ROCm version, full bench output,
binary md5, prompt md5.

### 3.4 — DFlash-by-genre matrix (full sweep)

```bash
./scripts/sweep_dflash_full.sh   # 3 model × 2 mode × 3 genre × 3 runs
```

Reference numbers in `README.md` "DFlash speculative decode" section.
Code prompts: 4× win on 27B / 2.6-3× on 9B. Prose prompts: tie or
small loss on 9B (-20%, draft-target alignment issue, NOT a bug).

### 3.5 — Coherence gate (mandatory before any DFlash claim)

```bash
./scripts/coherence-gate-dflash.sh
```

Hard fails: zero tokens, panic, max_token_freq > 0.40,
unique_token_count / total < 0.30. The gate runs 4 tests in ~3 minutes:
27b-dflash-prose, 27b-dflash-code, 27b-ddtree-b12-prose, 27b-ddtree-b12-code.

If any test reports "soft_warn" but not "hard error" — read the report
text (path printed at end) and eyeball the decoded output. Numbers
alone aren't enough — check for token attractors visually.

### 3.6 — Pull flow end-to-end

If you're testing an actual user UX flow:

```bash
hipfire pull qwen3.5:9b
hipfire pull qwen3.5:9b-draft
hipfire config set dflash_mode auto    # opt in (default since 2026-04-26: off)
hipfire run qwen3.5:9b "Write a Python function to find the longest substring without repeating characters"
# expected: daemon logs '[hipfire] DFlash draft detected: ...'
# response generates at ≥250 tok/s on a 9B target with a paired draft
```

Without the `dflash_mode auto` config, `hipfire run` runs pure AR
even when a paired draft is on disk — the daemon explicitly logs
`[hipfire] DFlash disabled (dflash_mode=off).` This is the "I pulled
the draft but DFlash isn't firing" pitfall.

---

## 4 · DDTree caveats (gfx1100 only)

DDTree on gfx1100 is currently a **structural perf regression** —
the linearization-slot RoPE phase delta skew at FA layers (commit
[39aa358](https://github.com/Kaden-Schutt/hipfire/commit/39aa358))
makes our tree path slower than our linear path. Lucebox's DDTree
works on RTX 3090; ours doesn't (yet) on gfx1100.

If you're running DDTree benches and seeing regressions vs. linear
DFlash: **expected**, not a bug. Path C (trained custom draft) and
Path D (stale-context overlap) are the roadmap fixes. Don't open
issues for "DDTree slower than linear on gfx1100" unless you have
new data not already documented.

For dataclass benches:
- DDTree b12-k2 wins τ on prose / instruct (per memory) but loses
  wall-clock to per-cycle overhead.
- DDTree b22 with `--ddtree-batched` loses to plain linear on code.

---

## 5 · Reporting findings

### Where to put bench results

- **Numerical perf-checkpoints:** in the commit message body of the
  commit that produced the numbers, or in the PR description. The
  prior `docs/perf-checkpoints/` tree was archived 2026-04-27 — first-
  class artifacts now live in git history, not in a parallel doc tree
  that drifts.
- **Forensic discoveries (e.g. "I found X regresses Y"):** in the
  commit message of the fix (or the bisect commit). For longer
  writeups, the PR description. Local-only scratch goes to
  `.codeinsight+research/` (gitignored).
- **Coherence-gate failures:** include the gate's report path
  (`/tmp/coherence-dflash-*.md`) verbatim in the commit/PR.
  Investigate as numerical bug, NOT sampling variance.
- **Regression vs. last-shipped baseline:** include the binary md5
  (md5sum target/release/examples/dflash_spec_demo) and prompt md5.
  Without these, the result is unreproducible.

### Don't claim a perf win without

- ≥3 fresh-process runs
- Prompt md5 recorded
- Binary md5 recorded
- Coherence-gate-dflash pass
- Eyeball check on decoded output (especially when τ is unusually high)

### Don't claim a perf regression without

- ≥3 fresh-process runs (same prompt, same env)
- Bisect to a specific commit (use `scripts/probe_commits.sh COMMIT_BEFORE COMMIT_AFTER`)
- Confirmation that the regression appears across genres (not just one
  prompt that happens to hit a different distribution)

---

## 6 · Common pitfalls (history of what bit us)

| Symptom | Real cause | Fix |
|---|---|---|
| "DFlash got slower overnight" | Prompt structure changed (one newline added/removed) | Use byte-identical prompts via `benchmarks/prompts/*.txt` |
| `τ=9.42` on first run, `τ=8.07` on next | Different prompt — see above | Same fix |
| "0 evictions even though sidecar loaded" | `cask_beta` too high (default 128) means trigger is at budget+128 | Lower beta to 16 to actually exercise the eviction policy |
| "DFlash 102 tok/s on prose vs 124 AR" | Draft-target argmax disagreement on prose tokens, τ collapses to ~1.2 | This is expected with z-lab drafts; fix is Path C (train custom draft) |
| 3.6-A3B DFlash 68.6 tok/s vs AR 135 tok/s (50% loss) | 3.6 draft trained on 3.5 traces; target distribution mismatch on code. τ=1.22 on hard code. | Use AR mode for 3.6-A3B until Path C (custom 3.6 draft training) completes. 3.5-A3B DFlash works (τ=4.91) |
| `hipMalloc out of memory` at hidden_rb | Long ctx (≥16K real tokens) + 27B + asym3 = tight on 24 GB | Reduce ctx, use a smaller target, or wait for the bounded-rolling-buffer trick (roadmap) |
| `tok/s` below expected on long-ctx | KV cache growth — prefill is fine but decode slows past ~2K | Test at small ctx first, then scale |
| daemon doesn't auto-find draft | Filename doesn't match `qwen3{ver}-{size}-dflash-{quant}.hfq` | Don't rename the file after pull |
| `[hipfire] DFlash disabled (dflash_mode=off)` | Default flipped to `off` in 35265c6 (post-2026-04-26). Pulling a draft does NOT auto-enable DFlash anymore. | `hipfire config set dflash_mode auto` (or `on`); or per-model `hipfire config qwen3.5:9b set dflash_mode on` |
| "Numbers don't match the README" | Forgot `HIPFIRE_NORMALIZE_PROMPT=1` (pre-2026-04-26) | Now default ON. Pull latest. If you opted out via `prompt_normalize=false`, that overrides the default — flip back. |
| "27B DFlash regressed 30-40% suddenly" | PR #32 (cleanup-dead-wmma-kernels) on master removed `gemm_hfq4g256_residual_wmma{,2,_k4}.hip` thinking dead. Dispatch fell back to slower variants. | Verify against canonical 199 tok/s @ max=120 with default flags. If kernel files missing in `kernels/src/`, `git checkout` from a known-good commit (see commit 9a2c667 for the full recovery context). |
| `HIPFIRE_GRAPH=1` reports plausible tok/s but output is garbage | Dangling stack-pointer kernargs from raw `self.hip.launch_kernel(...)` calls in `forward_scratch_layers` (kv_cache_write_*, attention_flash_*, fused_qkv_hfq4g256, rmsnorm_batched, rope_partial_interleaved_f32, gated_delta_net_q8, etc.) — captured pointers dangle past `end_graph_capture` | Bench tok/s alone never proves graph correctness. Always coherence-gate or eyeball under `HIPFIRE_GRAPH=1`. Fix: migrate every raw-launch helper used in forward_scratch_layers to `launch_maybe_blob` (model after `conv1d_silu_split_f32_n`). |

---

## 7 · Quick-reference flag table

| Env var | Purpose | Default |
|---|---|---|
| `HIPFIRE_NORMALIZE_PROMPT` | Phase 1 `\n{3,}` collapse | **ON (since 2026-04-26)** — set `0` to opt out |
| `HIPFIRE_PROMPT_TOKEN_HEAT` | Per-position BPE merge-rank dump | OFF |
| `HIPFIRE_PROMPT_HEAT_JSON` | JSON output for heat dump | OFF |
| `HIPFIRE_PROMPT_HEAT_LIMIT` | Max rows in heat dump | 64 |
| `HIPFIRE_KV_MODE` | Override kv_cache config | (config) |
| `HIPFIRE_ATTN_FLASH` | Override flash_mode config | (config) |
| `HIPFIRE_DFLASH_DRAFT` | Force a specific draft path. Empty string = explicit opt-out | (filename auto-match alongside target) |
| `HIPFIRE_LOCAL` | Force local-spawn (skip serve HTTP) | OFF |
| `HIPFIRE_HOST_TIMING` | Per-cycle host timing probe | OFF |
| `HIPFIRE_VERIFY_GRAPH` | Verify-forward graph capture (0 = off) | ON |
| `HIPFIRE_DDTREE_*` | Various DDTree diagnostics | various |

| dflash_spec_demo flag | Purpose |
|---|---|
| `--ar-baseline` | Skip DFlash, greedy-decode via target only |
| `--no-chatml` | Bare prompts (raw-text drafts) |
| `--no-adaptive-b` | Fix B at the draft's trained block size |
| `--ddtree-batched` | Use batched tree verify (research) |
| `--ddtree-budget N` | Tree node budget |
| `--ddtree-topk K` | Tree fan-out |
| `--cask-sidecar PATH` | Load TriAttention sidecar |
| `--cask-budget N` | KV eviction target |
| `--cask-beta N` | Hysteresis (lower = more aggressive eviction) |

---

## 8 · Open questions agents can investigate

If you want to actively contribute findings, these are open:

1. **Phase 3 prompt-shape rules** — what other rare BPE tokens depress
   τ? Run `encode_prompt --heat` on a wide variety of prompts and look
   for patterns.
2. **Path C training**: a target-aligned custom DFlash draft. Recipe at
   `../dflash-fe/RECIPE_RedHat_DFlash_MI300X.md`.
3. **Path D engineering**: stale-context overlap pipelining — the only
   structural lever still on the table for 27B-3.5 code beyond +8.2%.
4. **DDTree gfx1100 fix**: linearization-slot RoPE phase delta skew
   (commit 39aa358). Per-genre data: `feedback_dflash_per_genre`
   memory. If you have an idea for the structural fix, the project
   memory has the relevant context.

---

*Last updated: 2026-05-02 (v0.1.9-alpha — MQ3 production-ready: K4
decode, WMMA prefill family, DFlash cross-quant matrix, gfx12 port,
cache-invalidation lifecycle, defensive parseToolCalls (#111 stopgap),
gfx906 + gfx1152 arch gating, speed-gate DPM warmup). When this doc
gets stale (more than 1-2 releases behind HEAD), update it as part of
the release PR.*
