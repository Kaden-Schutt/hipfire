# Validation routes

Maps claim class → validation route. **CI merge authority for hardware-relevant
diffs is [`hw-gate`](../.github/workflows/hw-gate.yml)** (see below). Executable
behavior lives in the scripts and workflows named in each route. Methodology
numbers and Redline certification prose live in their owners
([`INDEX.md`](INDEX.md)). Plan hardware coverage in the hw-gate section below
(the author's `<!-- hw-gate-request -->` block selects routes); no local
planning tool is CI evidence.

| Field | Value |
|---|---|
| Inventory date | 2026-07-22 |
| Audited source ref | `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Comparison base | `origin/beta` @ `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Integrated commit / tree hashes | External Git/CI only. |

## Rules

1. Pick the **narrowest** route that covers the changed surface.
2. Every route below names an **existing** path, or is marked **blocked**.
3. **Fail closed** on unknown claim classes: do not improvise a gate, and
   do not treat a green unrelated route as coverage.
4. There is **no universal replacement gate**.
5. Harness success without the proof the claim needs is insufficient
   (especially Redline route proof — see [`REDLINE.md`](REDLINE.md)).
6. **Admissions** are recorded only in [`admissions.yml`](admissions.yml).
   A passing route does not create an admission row.

## hw-gate (CI, required)

[`hw-gate`](../.github/workflows/hw-gate.yml) is the **required** CI check for
every PR and the repository's autonomous review rung: two model seats decide,
one human owns `master`. Diffs that touch no hardware-relevant surface pass
immediately. Every decision is announced on the PR under the seat's own
identity (`hipfire-sol[bot]`, `hipfire-fable[bot]`).

### Bucket selection (`scripts/hw-gate/select.py`)

Changed paths → buckets `load` / `serve` / `kernel` (first match wins per path;
`serve` and `kernel` also imply `load`):

| Bucket | Path rules (summary) |
|---|---|
| **kernel** | `kernels/**`, `crates/rdna-compute/**`, `crates/hipfire-dispatch/**`, `crates/hip-bridge/**`, `crates/saddle-core/**` |
| **serve** | `crates/hipfire-engine/**`, `crates/hipfire-generate/**`, daemon `slots.rs` / `serve*.rs`, runtime emit/eos/dflash/spec/reset/triattn surfaces, arch `serve`/`generate`/`spec` sources |
| **load** | loader/daemon/runtime load+config surfaces, arch `load*`/`weights*`/`carrier.rs`, `crates/hipfire-config/**`, `crates/hipfire-registry/**`, `registry/**`, workspace/`crates/*/Cargo.toml` and lockfile |
| **none** | everything else (docs, benchmarks, most scripts, tests, markdown, …) |

Touching **policy** paths (`.github/workflows/**`, `.github/CODEOWNERS`,
`scripts/hw-gate/**`, leanup/ratchet scripts, `registry/**`) is part of the
hard floor: no seat can merge those; a human does. Exec-sensitive paths (build
scripts, manifests, toolchain, CI, shell/python) are reported to Sol as input
to its execution-risk judgment; they are not a gate by themselves.

### The author's request (`<!-- hw-gate-request -->`)

A PR body may carry a fenced JSON block after the marker:
`{"routes":[{"mode":"battery"|"chain","tag":"registry:tag"}],"claim":"..."}`.
Sol treats the claim as a claim, runs the requested routes when the tag exists
on the runner, reports unknown or absent tags as unavailable (not failed), and
states in its verdict whether the claim was proven, disproven, or not
exercised. The PR template ships the skeleton.

### Seat 1 — Sol decides hardware and delivers the verdict

Sol (`openai-codex/gpt-5.6-sol`, read-only tools in a PR checkout) reads the
diff before anything runs and decides `run_hardware`: the hardware job builds
the PR and runs its daemon as the maintainer's user on their workstation, so
Sol refuses diffs that reach outside the process (network, filesystem beyond
model/cache/temp, env or credential reads, process spawning, unexplained build
or dependency changes, obfuscated blobs, unexplained `unsafe`). Nothing else
gates hardware; a maintainer's **`hw-run`** label only ever forces a run and
is removed after each run and on every push. Sol also composes the route list:
the mandatory fixtures for the touched buckets, the author's requested routes,
and its own additions.

After hardware, Sol reads every decoded turn and returns `greenlight` /
`needs-human` / `block` with regressions cited by `file:line` and fixture. Sol
never merges and never approves; its review is a comment.

### Fixtures and harnesses

Every fixture runs through [`scripts/serve_harness.py`](../scripts/serve_harness.py)
— never a single `hipfire run`, which is one request against a fresh daemon
and proves nothing about turn-to-turn state — with reasoning off. Mandatory
fixtures are registry tags pinned by sha256 in
[`scripts/hw-gate/fixtures.json`](../scripts/hw-gate/fixtures.json); a
missing or mismatched pinned fixture **fails the gate**. Requested extra tags
resolve through `registry/v1.json`; one absent from the runner is reported as
unavailable.

- **`load` bucket** — `battery` on every fixture (varied prompts, expect
  substrings, attractor / runaway / empty detection).
- **`serve` bucket** — `battery` + `chain` (related turns through the prefix
  cache: reset, cache, and terminal semantics).
- **`kernel` bucket** — `battery` plus
  [`scripts/redline_daemon_harness.py`](../scripts/redline_daemon_harness.py)
  capture + HIP/PM4 parity on the dense trunk.

Decoded assistant text is posted **verbatim** in the evidence comment. Reading
it is part of review. The per-turn prefill/decode rates in that table are
harness-side timings (HTTP streaming, sampling) and run well under
`hipfire bench`; they are context, never a performance claim. Perf claims go
through [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md).

### Seat 2 — Fable investigates and decides the merge

Fable (`anthropic/claude-fable-5-1`, thinking `xhigh`) reads the diff, the
evidence, and Sol's verdict — and then, when that evidence does not prove the
change, goes and gets the evidence itself. It runs with a real shell in a
sandboxed checkout of the PR head on the hardware host: every hiptrx GPU is
reserved for the session (a host-level lock serializes Fable sessions and
excludes the lane runs), the PR and the base branch are both built for A/B,
every registry artifact on the host is available read-only, and everything
Fable writes to `$HW_GATE_EVIDENCE` is uploaded with its decision. There is
no fixed route vocabulary: Fable chooses what proves the change — a
multi-GPU load for new topology code, a refused-load-then-generate sequence
for a moved refusal, Redline parity for a kernel, a base-vs-PR A/B for a
"no behavior change" refactor. The sandbox bounds reach, not judgment: no
network, no writes outside its home/evidence/build tree, no other GPUs, no
credentials, no `gh` (the script posts), a wall-clock budget
(`HW_GATE_MAX_MINUTES`, default 45), and only registry artifacts.

Fable returns `merge-staging` / `hold` / `block` with an `investigation`
table (question → route run → evidence file → result), an `unproven` list for
what this host could not exercise, and an announcement written for the
author. It may veto Sol's greenlight or override Sol's needs-human — expected
when it closed the gap itself — and must say why; overrides are recorded so
the maintainer can audit both seats against outcomes.

**Probation.** While the two-seat rung is on probation, `merge-staging` means
Fable merges the PR head into **`beta`** (the staging branch) under its own
identity and announces the merge SHA on the PR. `master` is promoted from
`beta` by the maintainer; GitHub marks the staged PRs merged when that
happens. The record of prelims, verdicts, decisions, and overrides is the
dataset for lifting probation.

### The floor (`scripts/hw-gate/review.py`)

The floor is the workflow's own rule, split in two:

| Tier | Fires on | Who can override |
|---|---|---|
| **hard** | a failed fixture or harness, an attractor, a policy-file change, a `RATCHET-RAISE:` commit without the `ratchet-raise` label | nobody — `block` (evidence) or `hold` (policy/ratchet) regardless of either seat |
| **soft** | coverage gaps, confidence < 0.8, Sol's `needs-human`, an unparseable verdict | Fable, with a stated reason |

### The `hw-gate` status

- `merge-staging` — green.
- `hold` — red until a maintainer who has read the seats' comments applies
  **`human-reviewed`** (a logged signature, cleared on every push; a label
  event re-evaluates the recorded decision without re-running hardware).
- `block` — red; only a new commit clears it.

Branch protection binds every maintainer except repository admins
(`enforce_admins` is off on purpose: the admin's judgment is the emergency
path). An admin merging past a red status is expected to have read the
comments first.

## Automatic checks vs manual evidence

| Class | When it runs | Authority |
|---|---|---|
| **Automatic (hw-gate, required)** | PR via [`.github/workflows/hw-gate.yml`](../.github/workflows/hw-gate.yml); hardware run after `hw-run` label when buckets need it | **Required** CI check for every PR. Select may pass with no HW surface; otherwise load/serve/kernel evidence + bounded reviewer floor. |
| **Automatic (no GPU CI)** | PR / push via the no-GPU workflow only | Merge bar for compile, native control-plane/unit tests, CPU tests, and env/docs reference coverage. **Not** model coherence, serve semantics, or perf admission. |
| **Automatic (path-gated hooks)** | Local `pre-commit` on matching staged runtime paths | Runs the hotspot guards selected by the staged path set. Documentation-only staged sets do not trigger a separate docs hook. **Not** a full product matrix. |
| **Manual local no-GPU equivalent** | Human/agent invokes `scripts/no-gpu-ci.sh` outside CI | Same checks as the workflow script body; still **manual invocation**, not automatic CI. |
| **Manual (GPU / model)** | Human or agent on hardware with an explicit model path | Still required for claim classes hw-gate does not cover (parity oracles, perf protocol, Redline promotion ladder, admissions). |

No-GPU CI green never substitutes for hw-gate or for a required manual route. hw-gate green does not create an admission or skip claim-specific oracles named below.

### Automatic entrypoints

| Route | Path | Role |
|---|---|---|
| **hw-gate (required)** | [`.github/workflows/hw-gate.yml`](../.github/workflows/hw-gate.yml) + [`scripts/hw-gate/`](../scripts/hw-gate/) | **Automatic** required CI: path → buckets, pinned fixtures, hardware run, reviewer floor. See [§ hw-gate](#hw-gate-ci-required). |
| No-GPU CI workflow | [`.github/workflows/no-gpu-ci.yml`](../.github/workflows/no-gpu-ci.yml) | **Automatic** CI entry that invokes the no-GPU script on PR/push. |
| Pre-commit hooks | [`.githooks/pre-commit`](../.githooks/pre-commit) | **Automatic** when hooks are installed (`scripts/install-hooks.sh`). Selects HOTSPOT / SERVE_HOTSPOT / PP_HOTSPOT runtime guards from staged paths; documentation-only staged sets exit without a separate docs gate. |
| Dispatch `bind_thread` invariant | [`scripts/verify-bind-thread.sh`](../scripts/verify-bind-thread.sh) (via pre-commit on matching paths) | **Automatic** when hooked: every public `dispatch.rs` entry must bind the HIP thread. Not a kernel numeric test. |
| Env/docs drift check | [`scripts/check-env-docs.py`](../scripts/check-env-docs.py) | **Automatic** through `scripts/no-gpu-ci.sh`; checks that referenced `HIPFIRE_*` names are documented and production reads are config-owned. |

### Manual local equivalents (not automatic)

| Route | Path | Role |
|---|---|---|
| No-GPU script | [`scripts/no-gpu-ci.sh`](../scripts/no-gpu-ci.sh) | Manual local run of the same body CI uses: `cargo check --workspace --examples`; selected no-GPU Rust and native control-plane tests; CPU pytest; env/docs coverage. |
| Env/docs drift check | [`scripts/check-env-docs.py`](../scripts/check-env-docs.py) | Fast standalone env-name coverage and ownership check. No GPU or model download. |

### Documentation checks

The shipped automated documentation-specific check is
[`scripts/check-env-docs.py`](../scripts/check-env-docs.py). It validates
environment-variable reference coverage and config ownership; it is not a
general Markdown link, lifecycle-label, or semantic-drift checker. Beta does
not currently ship a standalone `check-docs-reliability.py` command or a
documentation-only pre-commit gate. For documentation-only changes, run the
env/docs check plus `git diff --check`, and inspect changed links and commands
against the current tree.

### Maintained manual harness roles

Narrow roles. Do not widen a harness into a universal gate.

| Harness | Path | Role | Not this harness |
|---|---|---|---|
| **gates.sh** | [`scripts/gates.sh`](../scripts/gates.sh) | Maintained **manual** wrapper: optional Redline capture, generic serve battery, optional fresh-process perf compare (`probe_commits.sh`). Requires `--model`. | Not CI-default. Not universal. Does not call retired coherence-gate scripts. |
| **serve_harness.py** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | **Model-agnostic** user-facing serve behavior (battery / chain / session): finish reasons, runaway/empty, prefix cache, prefill/decode timing, recall hooks. | Not LFM thinking-frame specifics. Not Redline route proof. |
| **serve_harness.py (LFM tag)** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | LFM2.5 serve smoke with the exact registry tag; use registry sampling or `recipe:nothink` for non-thinking framing. | Not a substitute for numerical parity oracles. |
| **redline_daemon_harness.py** | [`scripts/redline_daemon_harness.py`](../scripts/redline_daemon_harness.py) | Resident-daemon **Redline** capture, phase fingerprint, shadow/parity, and timing evidence under manual-capture env. | Discovery/correctness evidence ≠ product timed-arm route proof by itself. Does not enable AQL routing. |
| **dispatch_profile** | [`tools/redline/dispatch_profile.py`](../tools/redline/dispatch_profile.py) | Manual **attribution-only** steady-state retained-PM4 per-dispatch span diagnostic (instrumented GFX12 tape). | **Not** route proof, certification/admission, or absolute/pure kernel timing: timestamp commands add observer overhead and spans include preceding PM4 boundary packets. |
| **tools.redline golden** | [`tools/redline/golden.py`](../tools/redline/golden.py) | Reproduce the exact checked-in MQ4R TG128 model/benchmark/route fixtures through the route-proof-capable product harness. | Not a universal GPU gate, a new-route certification shortcut, or an admission. |

### Supporting manual tools (existing; claim-scoped)

Use only when the claim class below names them. They are not universal.

| Tool | Path | Typical claim |
|---|---|---|
| Kernel channel tests | `target/release/examples/test_kernels` (build: `cargo build --release --features deltanet --example test_kernels -p hipfire-runtime`) | Kernel numeric vs CPU reference on the detected arch. **Not** dispatch bind coverage. |
| Dispatch bind check (manual) | [`scripts/verify-bind-thread.sh`](../scripts/verify-bind-thread.sh) | Same bind invariant as the hook; run explicitly if hooks are not installed. |
| Speed regression floor | [`scripts/speed-gate.sh`](../scripts/speed-gate.sh) | Prefill/decode vs committed `tests/speed-baselines/<arch>.txt` when that path’s policy applies. |
| Fresh-process commit probe | [`scripts/probe_commits.sh`](../scripts/probe_commits.sh) | Optional A/B invoked from `gates.sh --perf`. |
| Perf protocol | [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) | How to measure; not an executable gate. |
| Redline certification ladder | [`docs/REDLINE.md`](REDLINE.md) | What evidence is required before Redline-attributed promotion. |
| Path-specific parity / state oracle | Arch-owned example or test named by the change (when one exists) | Hidden-state, logit, KV/conv, or graph parity. If none exists for the surface → **blocked**. |

## Claim → route map

| Claim / change class | Minimum route(s) | Evidence kind |
|---|---|---|
| Docs, env-var tables, no-GPU Rust/Python/CLI only | Automatic: `.github/workflows/no-gpu-ci.yml` invokes `scripts/no-gpu-ci.sh`; for a fast docs-only local check run `python3 scripts/check-env-docs.py` plus `git diff --check` | Automatic CI and/or manual local equivalent — **not** GPU/model evidence |
| New/changed `.hip` kernel (numeric) | `test_kernels`; then model-level manual route for the arch under test | Manual + channel |
| Dispatch `bind_thread` / public `dispatch.rs` bind surface | `.githooks/pre-commit` → `scripts/verify-bind-thread.sh` (or run that script manually) | Automatic hook or manual bind check — **not** `test_kernels` |
| Forward / fusion / KV **numerical or state parity** | Path-specific parity/state oracle for that arch/surface; **blocked** if no oracle exists | Manual oracle — **not** `serve_harness.py` |
| Forward / fusion / sampling / KV **user-facing serve semantics** | `scripts/serve_harness.py` with the exact model (after parity route if the change can break numbers/state); add `scripts/gates.sh` when the Redline+serve+optional perf wrapper is desired | Manual serve (semantics only) |
| LFM2.5 chat framing / thinking output | `scripts/serve_harness.py` with an `lfm2.5:*` registry tag | Manual LFM |
| VL vision-tower forward numerical parity (arch-5 / arch-11 carriers) | Dump-and-diff vs an HF `transformers` reference for the exact checkpoint, pixel inputs pinned by hash (`benchmarks/vision/dump_hf_reference.py` precedent; family route: [`qwen35-vl-mq4v2-spec.md`](qwen35-vl-mq4v2-spec.md) §5, [`specs/2026-08-27-qwen35-vl-vision-serve.md`](specs/2026-08-27-qwen35-vl-vision-serve.md)); **blocked** for a checkpoint with no reference dump | Manual oracle — not `serve_harness.py`; a green VL serve battery is *not* parity evidence |
| VL image-bearing serve semantics (`generate_vl` over `/v1/chat/completions`) | Manual OpenAI-compatible battery through `hipfire serve` with the exact VL artifact: committed fixtures under [`../benchmarks/vision/images/`](../benchmarks/vision/images/) + the fixed desc/ocr prompts of `comparison-2026-05-23.md`, greedy temp 0; stream **and** non-stream typed-emission check (reasoning vs `content` deltas; no literal `<think>`/`<|im_end|>` chunks in content); client-disconnect probe mid-stream followed by an immediate follow-up turn (no slot wedge); eyeball every decoded output; record artifact sha256, fixture hashes, binary md5s. No scripted harness exists (**blocked** until one lands); `scripts/serve_harness.py` is text-only today and does not exercise this surface | Manual serve (semantics only) |
| Retained replay / PM4 / AQL graft | `scripts/redline_daemon_harness.py` **and** the certification steps in `docs/REDLINE.md` | Manual Redline; promotion still policy-gated |
| Perf improvement claim | Protocol in `methodology/perf-benchmarking.md` + stationary matched runs; `speed-gate.sh` or `gates.sh` perf arm when applicable | Measured; not admission |
| Existing sealed MQ4R Redline fixture reproduction | `python3 -m tools.redline golden` for its exact gfx1100/gfx1151/gfx1201 fixture only | Measured reproduction; exact identity + route proof + stationary floor; not admission. Does **not** imply that every default-eligible `.mq4r` model file is one of the sealed fixtures. |
| Arch port | `methodology/arch-port-validation.md` (channel + speed; no retired coherence battery as acceptance) | Manual |
| Model/route **admission** (registry evidence) | Row in [`admissions.yml`](admissions.yml) | Schema v2; exactly one evidence-bound record (LFM2.5-350M MQ4 gfx1201 retained-PM4). No inferred/wildcard rows. Registry admission/evidence is distinct from runtime wiring: the sealed LFM row does **not** select a runtime default and current automatic selection does not use it. |
| MQ4R **runtime** automatic Redline default | Source predicate `mq4r_redline_default` in `crates/hipfire-runtime/src/config.rs`; policy in [`REDLINE.md`](REDLINE.md) | **Only** current automatic runtime predicate. Runtime-only: exact GPU arch `gfx1100`, `gfx1151`, or `gfx1201`; PP=1; TP=1; case-insensitive `.mq4r` → retained PM4/Auto unless disabled with the config wizard's built-in `hip` profile, another explicit backend selection, or `HIPFIRE_REPLAY_BACKEND=hip`. Model-family agnostic (no `arch_id` gate). `gfx1200` and all other arches remain opt-in. Existing LFM `.mq4` registry evidence is not auto-selected because it is not `.mq4r`, not because LFM is categorically exempt; any usable non-default retained route must still prove route support and fail closed when unsupported. **Not** registry admission, **not** Section 7 certification, and **not** a sealed-fixture claim for every default-eligible `.mq4r` model. |
| Unknown surface | **Blocked** until an owner adds a row here | Fail closed |

## Retired coherence-gate scripts

The fixed `scripts/coherence-gate-*.sh` batteries are **retired as current
acceptance evidence**. They must not be required for merge, promotion, or
benchmark claims.

| Pattern | Status |
|---|---|
| `scripts/coherence-gate-*.sh` (e.g. `coherence-gate-dflash.sh`, `coherence-gate-qwen35-dspark.sh`, `coherence-gate-minimax.sh`, `coherence-gate-cohere2moe.sh`, `coherence-gate-deepseek4-*.sh`, …) | **Historical reproduction only.** Never promotion or acceptance. |
| Other gate scripts **not named anywhere in this selector** | Do not treat as canonical acceptance unless a future INDEX/VALIDATION revision names them. Supporting tools already listed above stay in force. |

Campaign-specific guidance (for example an LFM effort that omits coherence
gates) is **not** a universal rule for every arch or model. Universality is
blocked on purpose.

## Explicit non-routes

| Anti-pattern | Disposition |
|---|---|
| One script that “replaces all gates” | **Rejected** — no universal gate. |
| Green no-GPU CI as proof of GPU correctness | **Rejected** |
| `serve_harness` as numerical/state parity or universal forward proof | **Rejected** — semantics only; oracle or **blocked** |
| `serve_harness` success as Redline route proof | **Rejected** |
| `redline_daemon_harness` fingerprint as installed product PM4/AQL route | **Rejected** without `REDLINE.md` ladder |
| Coherence-gate pass as current acceptance | **Rejected** |
| Bench number without protocol + identity hashes | **Rejected** as promotion evidence |
| Inferred or “signed” `admissions.yml` row without earned fixture evidence | **Rejected** — schema v2 forbids inferred/wildcard rows; only the exact admitted record applies |

## Related owners

- Navigation / lifecycle: [`INDEX.md`](INDEX.md)
- Admission registry: [`admissions.yml`](admissions.yml)
- Redline policy: [`REDLINE.md`](REDLINE.md)
- Perf protocol: [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md)
