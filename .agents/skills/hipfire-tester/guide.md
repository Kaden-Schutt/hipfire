# hipfire-tester guide

Agent checklist for bring-up, claim-scoped harnesses, and upstream-ready
reports on AMD GPUs. Works with any agent framework.

Mutable inventories (model tags, VRAM floors, default KV, format tables) live
in canonical owners — **do not copy them here**. Resolve tags from
[`registry/models.json`](../../../registry/models.json) and
[`docs/MODELS.md`](../../../docs/MODELS.md). Route every claim through
[`docs/VALIDATION.md`](../../../docs/VALIDATION.md).

## Rules

1. After identity/build, **branch immediately by claim class** — no fixed
   linear battery that always runs serve before channel/parity.
2. Only run commands whose paths exist in this checkout.
3. Harness green ≠ admission, numerical parity, or Redline product route proof.
4. Retired batteries are **not** current acceptance — use VALIDATION only.
5. Numeric `.hip`: channel (`test_kernels`) **then** applicable model/path
   manual route. Forward/fusion/KV numbers or state: path-specific parity
   oracle first (or **BLOCKED** if missing) → user-facing serve semantics only
   if needed → perf last. Never substitute serve/perf for a missing oracle.
6. Perf numbers need [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md):
   warm DPM/JIT, fresh process when comparing commits, binary + prompt identity.
   Matrix/profile CLI arms are **synthetic daemon probes**; production-path
   authority is native `serve_harness.py` JSON / daemon `done` fields.
7. Local serve/harness testing binds **loopback** (`127.0.0.1`). All-interface
   exposure needs explicit approval plus trusted firewall or authenticated
   reverse proxy ([`docs/SERVE.md`](../../../docs/SERVE.md) — no auth/TLS).
8. Chain `hipfire-diag` / `hipfire-autoheal` on environment or runtime failure;
   this skill stays on route selection and reporting.

## Phase 0 — identity

Record before any timed run:

```bash
hipfire --version 2>/dev/null || true
hipfire diag
# optional: rocminfo | rg -n "Name:|Marketing|gfx"
```

Note: gfx arch, VRAM free/total, HIP/ROCm line, kernel/blob status, local model
list. First-run kernel JIT can take minutes; cache is typically under
`.hipfire_kernels/` (worktree) or the install kernel dir.

## Phase 1 — install / build sanity

Prefer the path the tester actually uses:

- **Installed CLI:** `hipfire` on `PATH` (see [`docs/GETTING_STARTED.md`](../../../docs/GETTING_STARTED.md)).
- **Source checkout:** build only what the route needs, e.g.

```bash
cargo build --release --features deltanet --example daemon -p hipfire-runtime
# channel tests when the claim is kernel-numeric:
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
```

No-GPU merge bar (not GPU proof):

```bash
scripts/no-gpu-ci.sh
```

## Phase 2 — model selection

```bash
hipfire list          # local
hipfire list -r       # + remote registry tags
hipfire pull <tag>    # exact tag from registry; check min_vram_gb vs free VRAM
```

Do not hard-code a “standard matrix” of sizes. Pick one primary tag that fits
the card and the claim (e.g. `qwen3.5:4b` for dense MQ4 smoke; an `lfm2.5:*`
tag only for LFM routes). Confirm the on-disk file under `~/.hipfire/models/`
(or the path you pass to harnesses).

DFlash: pulling a draft does **not** enable speculation. Set
`hipfire config set dflash_mode auto` (or per-model overlay) and confirm logs
show the paired draft. Config authority: [`docs/CONFIG.md`](../../../docs/CONFIG.md).


## Phase 3 — branch by claim class

Do **not** run every subsection. Pick the VALIDATION claim class after
Phases 0–2, before any model execution. Ordering when multiple classes apply:

1. **Numeric `.hip` / channel** (3a) before model-level manual routes
2. **Parity / state** (3b) before user-facing serve semantics
3. **Serve / framing / multi-turn semantics** (3c–3e) only when the claim is
   user-visible behavior (and after parity when numbers/state can change)
4. **Redline / optional wrapper** (3f–3g) only for those claim classes
5. **Perf** (Phase 4) last — never a substitute for channel or parity

Missing required oracle or model/path route → **BLOCKED** (fail closed).

### 3a. Kernel channel (numeric `.hip`)

Required first leg when the claim is a new/changed numeric kernel:

```bash
cargo build --release --features deltanet --example test_kernels -p hipfire-runtime
./target/release/examples/test_kernels
```

Channel vs CPU reference on the detected arch. **Not** dispatch `bind_thread`
coverage (`scripts/verify-bind-thread.sh`). **Not** end-to-end serve proof.
**Not complete alone:** after channel, run the applicable model/path-level
manual route for that arch/surface ([`docs/VALIDATION.md`](../../../docs/VALIDATION.md)).
If that route does not exist → **BLOCKED**.

### 3b. Numerical / state parity

If the claim is forward/fusion/KV **numbers or state**, VALIDATION requires a
**path-specific oracle** for that arch/surface. Examples may live under
arch crates (e.g. LFM dump/parity examples) when present.

- Oracle exists → run it and record pass/fail **before** serve semantics or perf.
- No oracle for the surface → **BLOCKED** — do not substitute
  `serve_harness.py`, `hipfire bench`, or tok/s.

### 3c. Generic serve semantics

Model-agnostic battery / chain / session (finish reasons, empty/runaway,
timing hooks). **Not** numerical parity. **Not** LFM thinking-frame specifics.
**Not** Redline route proof. Run only after parity (3b) when the change can
break numbers/state.

For a user-visible generation/framing claim, start with a single-shot smoke:

```bash
hipfire run <tag> "Explain WMMA in one paragraph."
```

Expect coherent text and a tok/s footer. An environment/runtime failure here
routes to diag/autoheal; it does not bypass a required channel or parity route.
Optional CLI surface checks: `hipfire config list`, `hipfire ps`, and
`hipfire list`.

**Bind:** local testing must use loopback. Serve has **no authentication and
no TLS** ([`docs/SERVE.md`](../../../docs/SERVE.md)). Prefer:

```bash
hipfire serve 127.0.0.1:11435 -d
```

Do **not** use default all-interface bind or `scripts/serve-restart.sh`
(hardcodes `0.0.0.0`) for routine local harness work. If teardown is needed:
`scripts/serve-restart.sh --kill-only [port]` then the loopback serve command
above. All-interface exposure requires explicit user approval plus a trusted
firewall or authenticated reverse proxy.

```bash
# Portable route: attach an already-running **loopback** serve.
# Start serve separately first (loopback bind), then:
python3 scripts/serve_harness.py \
  --model "$HOME/.hipfire/models/<file-from-registry>" \
  --no-spawn \
  --port <actual-serve-port> \
  --mode battery \
  --sampling greedy \
  --max-tokens 128 \
  --out /tmp/serve_harness.json
```

Useful flags: `--tag`, `--mode {battery,chain,session}`, `--kv`, `--mtp`,
`--thinking`, `--seed`, `--prompts-file`. Spawn mode resolves the native
`hipfire` binary (or `HIPFIRE_CLI_BIN`). For `--mode session`, always
pass an explicit `--session PATH` (default is a machine-local fixture path).

### 3d. LFM2.5 framing / thinking

Only for registry tags under the `lfm2.5` family — resolve the exact tag and
on-disk file from [`registry/models.json`](../../../registry/models.json) /
[`docs/MODELS.md`](../../../docs/MODELS.md) at use time (do not copy a tag list
here).

```bash
python3 scripts/serve_harness.py \
  --model "$HOME/.hipfire/models/lfm2.5-1.2b-thinking.mq4" \
  --tag lfm2.5:1.2b-thinking \
  --sampling registry \
  --mode battery
# non-thinking framing: --sampling recipe:nothink
# HIPFIRE_DAEMON_BIN overrides target/release/examples/daemon
```

Requires a built native CLI and daemon (default
`target/release/examples/daemon`). Semantic smoke only — not logit/KV oracles. If the
claim also needs numerical/state proof, complete 3b first or stay **BLOCKED**.

### 3e. Multi-turn recall (manual HTTP)

Use when the claim is multi-turn serve behavior on a running daemon — not as a
substitute for path-specific parity oracles.

```bash
hipfire stop 2>/dev/null || true
# loopback only for local testing (no auth/TLS on serve)
hipfire serve 127.0.0.1:11435 -d
# wait until ready (warmup)
curl -s "http://127.0.0.1:11435/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"<tag>","messages":[
    {"role":"user","content":"My name is Kaden."},
    {"role":"assistant","content":"Hello Kaden!"},
    {"role":"user","content":"What is my name? One word."}
  ],"max_tokens":50,"temperature":0}'
hipfire stop
```

Report actual content. Prefer `serve_harness.py --mode chain|session` when the
harness covers the scenario. Teardown via `scripts/serve-restart.sh` must be
`--kill-only` then loopback relaunch — never the script’s default `0.0.0.0`
relaunch for local testing.

### 3f. Optional unified wrapper

```bash
# Safe harness-only (no worktree mutation): always pass --no-perf
scripts/gates.sh --model "$HOME/.hipfire/models/<file>" --no-perf
# --serve-only | --redline-only narrow redline/serve arms but do NOT disable perf
# --perf <ref> only changes the comparison ref (default HEAD~1); perf stays on
# --aql | --work-dir PATH
```

By default runs redline daemon harness + serve battery **and**
`probe_commits.sh` (stashes dirty/untracked state, force-checks out commits,
removes the bench binary). Use `--no-perf` unless the tree is clean and you
intentionally want that mutation. Explicit commit probes also require a clean
worktree willing to accept checkout churn. Does **not** treat retired
batteries as acceptance (route via VALIDATION). Not CI-default. Not universal.

### 3g. Redline capture evidence

```bash
python3 scripts/redline_daemon_harness.py \
  --model "$HOME/.hipfire/models/<file>" \
  --pm4 \
  --out .redline-work/redline-daemon-phases.json \
  --log .redline-work/redline-daemon-phases.log
```

Discovery/correctness evidence under manual-capture env. Promotion still needs
the certification ladder in [`docs/REDLINE.md`](../../../docs/REDLINE.md).
Harness fingerprint alone is not installed product PM4/AQL route proof.

## Phase 4 — performance reporting (after channel/parity when those apply)

Perf never replaces channel or parity. Missing oracle for a numbers/state
claim remains **BLOCKED** even if tok/s looks good.

### Product-CLI synthetic daemon probes

```bash
hipfire bench <tag> --runs 3
hipfire bench <tag> --matrix --pp 128,512,2048 --ctx 128,2048 --tg 128 --runs 3
# optional: --kv-mode <mode from docs/CLI.md / docs/CONFIG.md>  --redline  --json
hipfire profile <tag> --pp 128,512,2048 --tg 128,512 --runs 3
```

Matrix/profile arms drive daemon `bench_prefill` / `bench_decode` — **synthetic
daemon probes**, not production-path throughput authority
([`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md)).
Prefer `HIPFIRE_DPM_WARMUP_SECS=10` for stationary numbers.

### Production-path throughput

Report production-path numbers from native `serve_harness.py` JSON / daemon
`done` fields (`prefill_tok_s`, `decode_tok_s`, `ttft_ms`):

```bash
# see docs/methodology/bench-suite.md for the identity contract
HIPFIRE_DAEMON_BIN=./target/release/examples/daemon \
  python3 scripts/serve_harness.py --model <model_path> --tag <tag> \
  --sampling registry --mode battery --out /tmp/serve-harness.json
```

### Speed floor (MQ4 baselines when policy applies)

```bash
# needs target/release/examples/bench_qwen35_mq4 and tests/speed-baselines/<arch>.txt
scripts/speed-gate.sh --fast          # smaller subset when supported
scripts/speed-gate.sh                 # full available-model set
```

Baselines are committed floors for that script’s models/archs — not a global
perf admission for every tag.

### Fresh-process commit compare

```bash
scripts/probe_commits.sh <baseline-sha> <candidate-sha>
# or via gates.sh --perf <ref>
```

### What to refuse as promotion evidence

- Single dirty-worktree smoke without binary/prompt identity
- Atlas ISA Fit / suggest output as runtime proof (see `hipfire-kernel-atlas`)
- Historical tables in [`docs/BENCHMARKS.md`](../../../docs/BENCHMARKS.md) as live floors
- Retired battery results as current acceptance
- `hipfire bench --matrix` / `hipfire profile` synthetic arms as production-path authority
- Serve-harness green as numerical/state parity

## Phase 5 — quantize smoke (optional)

Only if the claim touches `hipfire quantize` or a freshly built artifact:

```bash
hipfire quantize Qwen/Qwen3.5-0.8B --format mq4 -o /tmp/test-qwen35-0.8b.mq4
hipfire run /tmp/test-qwen35-0.8b.mq4 "Hi"
```

Format authority: [`docs/QUANTIZATION.md`](../../../docs/QUANTIZATION.md),
operator guide: [`docs/QUANTIZE.md`](../../../docs/QUANTIZE.md).

## Report template

```markdown
## Tester report — {GPU marketing name}, {gfx}, {VRAM}

- Date / host:
- hipfire: `hipfire --version` →
- diag: (paste critical lines: arch, HIP, VRAM, kernel status)
- Model: {registry tag} → {on-disk path} (md5 optional)
- Config relevant: kv_cache=, dflash_mode=, thinking=
- Claim class(es) (branch immediately; do not invent a universal matrix):
- VALIDATION routes run (order: channel → parity → semantics → perf):
  - [ ] bring-up smoke (`hipfire run`) — pass/fail
  - [ ] test_kernels **then** model/path manual route — if numeric `.hip`
  - [ ] path-specific parity oracle — if numbers/state; else BLOCKED if required and missing
  - [ ] serve_harness / gates.sh — semantics only; loopback bind
  - [ ] redline harness / speed-gate / probe_commits — if claimed
- Bench (if any):
  - synthetic probe: `hipfire bench` / `--matrix` / `profile` — label synthetic
  - production-path: native `serve_harness.py` JSON and daemon `done` fields
  - command, binary md5, prompt id, warmup, table of pp/tg
- Evidence class per number: measured synthetic | measured production-path | not claimed as admission
- Failures / quirks / BLOCKED (missing oracle):
```

Submit via GitHub issue or PR. Keep raw logs out of git unless the project
already has a designated evidence path.

## Known operational notes (not acceptance gates)

- Runtime hangs, port conflicts, stale `serve.pid`, ROCm include paths →
  `hipfire-autoheal`.
- Spec-decode / DFlash quality regressions → claim-scoped serve or arch oracle
  via VALIDATION; retired batteries are not acceptance.
- Branch-only LFM or Redline surfaces: label as branch-implemented per
  [`docs/INDEX.md`](../../../docs/INDEX.md); do not phrase as `origin/beta`
  product defaults unless the comparison base already has them.
