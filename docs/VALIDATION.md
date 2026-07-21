# Validation routes

Sole human **route selector** for hipfire validation evidence.
Executable behavior lives in the scripts and workflows named below — this
file only maps claim class → route. Methodology numbers and Redline
certification prose live in their owners ([`INDEX.md`](INDEX.md)).

| Field | Value |
|---|---|
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| Comparison base | `origin/beta` @ `9ffb18da9d1377dfbf759db82641ea039b2e522e` |
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

## Automatic checks vs manual evidence

| Class | When it runs | Authority |
|---|---|---|
| **Automatic (no GPU CI)** | PR / push via the no-GPU workflow only | Merge bar for compile, native control-plane/unit tests, and docs reliability. **Not** model coherence, serve semantics, or perf admission. |
| **Automatic (path-gated hooks)** | Local `pre-commit` on matching staged paths | Always runs documentation reliability first; documentation/governance/tooling-only staged sets exit before runtime/GPU hotspot gates; mixed commits continue through the hotspot guards that match remaining staged paths. **Not** a full product matrix. |
| **Manual local no-GPU equivalent** | Human/agent invokes `scripts/no-gpu-ci.sh` outside CI | Same checks as the workflow script body; still **manual invocation**, not automatic CI. |
| **Manual (GPU / model)** | Human or agent on hardware with an explicit model path | Required for kernel, dispatch, forward, quant, serve-behavior, Redline, and perf claims. |

Automatic CI green never substitutes for a required manual route. Running the no-GPU script locally is convenient parity with CI, not an automatic check.

### Automatic entrypoints

| Route | Path | Role |
|---|---|---|
| No-GPU CI workflow | [`.github/workflows/no-gpu-ci.yml`](../.github/workflows/no-gpu-ci.yml) | **Automatic** CI entry that invokes the no-GPU script on PR/push. |
| Pre-commit hooks | [`.githooks/pre-commit`](../.githooks/pre-commit) | **Automatic** when hooks are installed (`scripts/install-hooks.sh`). Always runs the staged documentation reliability checker, then separates documentation/governance/tooling-only staged sets from runtime paths before HOTSPOT / SERVE_HOTSPOT / PP_HOTSPOT. Docs-only sets exit before those runtime/GPU gates; mixed commits still run every applicable runtime gate. |
| Dispatch `bind_thread` invariant | [`scripts/verify-bind-thread.sh`](../scripts/verify-bind-thread.sh) (via pre-commit on matching paths) | **Automatic** when hooked: every public `dispatch.rs` entry must bind the HIP thread. Not a kernel numeric test. |
| Docs reliability checker | [`scripts/check-docs-reliability.py`](../scripts/check-docs-reliability.py) | **Automatic** in no-GPU CI and local pre-commit (staged mode, always before any docs-only early exit). Env-table coverage included. |

### Manual local equivalents (not automatic)

| Route | Path | Role |
|---|---|---|
| No-GPU script | [`scripts/no-gpu-ci.sh`](../scripts/no-gpu-ci.sh) | Manual local run of the same body CI uses: `cargo check --workspace --examples`; selected no-GPU Rust and native control-plane tests; CPU pytest; env/docs coverage. |

### Documentation reliability checker

Canonical checker: [`scripts/check-docs-reliability.py`](../scripts/check-docs-reliability.py).
Stdlib-only. No GPU, no model downloads. It reads a **git snapshot** (explicit
commit via `--target-ref`, or the current index via `--staged`) against an
explicit `--base-ref`. It does **not** guess a merge base, and it does **not**
write commit/tree identity into tracked docs.

The former standalone env-table gate is **subsumed**: env coverage over
`AGENTS.md` / `README.md` / `CONTRIBUTING.md` vs
[`env-vars.md`](env-vars.md) runs inside the canonical checker.

| Invocation | Command | When |
|---|---|---|
| Focused unit tests | `python3 -m unittest tests.test_docs_reliability` | Local or CI; no GPU. Exercises checker predicates in temp repos. |
| Staged (index) snapshot | `python3 scripts/check-docs-reliability.py --staged --base-ref HEAD` | Local pre-commit / before commit. Target = current index (`git write-tree`); ignores unstaged worktree noise. |
| Commit-tree snapshot | `python3 scripts/check-docs-reliability.py --target-ref <commit> --base-ref <commit>` | Local or CI against an explicit integrated commit. Example local same-tree structural check: `--target-ref HEAD --base-ref HEAD`. |

CLI contract (exactly one mode + required base):

```bash
python3 scripts/check-docs-reliability.py --staged --base-ref <ref>
python3 scripts/check-docs-reliability.py --target-ref <ref> --base-ref <ref>
```

**CI identity is external.** The no-GPU workflow fetches the explicit base and
supplies the final integrated commit and comparison base to the checker (via
the no-GPU script / `DOCS_DIFF_BASE`). Tracked docs never self-reference the
integrated commit or tree. CI records in its **job artifact**: commit SHA, tree
SHA, source refs (target + base), checker exit/results, and the semantic
matrix outcome for that run. Local invocation stays usable without GPU or
model downloads; when `DOCS_DIFF_BASE` is unset locally, same-tree checking
with `--base-ref HEAD` is the structural fallback only.

Enforcement surfaces (implementation lives in hooks/CI; this file only names
them):

- Pre-commit always runs the **staged** documentation reliability checker, then exits before runtime/GPU hotspot gates when the staged set is documentation/governance/tooling-only. Mixed commits continue through every applicable runtime gate after the checker.
- No-GPU CI / `scripts/no-gpu-ci.sh` run the focused unit tests and the
  **commit-tree** form with the externally supplied base.

### Maintained manual harness roles

Narrow roles. Do not widen a harness into a universal gate.

| Harness | Path | Role | Not this harness |
|---|---|---|---|
| **gates.sh** | [`scripts/gates.sh`](../scripts/gates.sh) | Maintained **manual** wrapper: optional Redline capture, generic serve battery, optional fresh-process perf compare (`probe_commits.sh`). Requires `--model`. | Not CI-default. Not universal. Does not call retired coherence-gate scripts. |
| **serve_harness.py** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | **Model-agnostic** user-facing serve behavior (battery / chain / session): finish reasons, runaway/empty, prefix cache, prefill/decode timing, recall hooks. | Not LFM thinking-frame specifics. Not Redline route proof. |
| **serve_harness.py (LFM tag)** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | LFM2.5 serve smoke with the exact registry tag; use registry sampling or `recipe:nothink` for non-thinking framing. | Not a substitute for numerical parity oracles. |
| **redline_daemon_harness.py** | [`scripts/redline_daemon_harness.py`](../scripts/redline_daemon_harness.py) | Resident-daemon **Redline** capture, phase fingerprint, shadow/parity, and timing evidence under manual-capture env. | Discovery/correctness evidence ≠ product timed-arm route proof by itself. Does not enable AQL routing. |

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
| Docs, env-var tables, no-GPU Rust/Python/CLI only | Automatic: `.github/workflows/no-gpu-ci.yml` in CI (focused `tests.test_docs_reliability` + `scripts/check-docs-reliability.py` with externally supplied target/base); optional manual local `scripts/no-gpu-ci.sh` or the checker commands above; pre-commit staged mode when hooks are installed | Automatic CI and/or manual local equivalent — **not** GPU/model evidence |
| New/changed `.hip` kernel (numeric) | `test_kernels`; then model-level manual route for the arch under test | Manual + channel |
| Dispatch `bind_thread` / public `dispatch.rs` bind surface | `.githooks/pre-commit` → `scripts/verify-bind-thread.sh` (or run that script manually) | Automatic hook or manual bind check — **not** `test_kernels` |
| Forward / fusion / KV **numerical or state parity** | Path-specific parity/state oracle for that arch/surface; **blocked** if no oracle exists | Manual oracle — **not** `serve_harness.py` |
| Forward / fusion / sampling / KV **user-facing serve semantics** | `scripts/serve_harness.py` with the exact model (after parity route if the change can break numbers/state); add `scripts/gates.sh` when the Redline+serve+optional perf wrapper is desired | Manual serve (semantics only) |
| LFM2.5 chat framing / thinking output | `scripts/serve_harness.py` with an `lfm2.5:*` registry tag | Manual LFM |
| Retained replay / PM4 / AQL graft | `scripts/redline_daemon_harness.py` **and** the certification steps in `docs/REDLINE.md` | Manual Redline; promotion still policy-gated |
| Perf improvement claim | Protocol in `methodology/perf-benchmarking.md` + stationary matched runs; `speed-gate.sh` or `gates.sh` perf arm when applicable | Measured; not admission |
| Arch port | `methodology/arch-port-validation.md` (channel + speed; no retired coherence battery as acceptance) | Manual |
| Model/route **admission** into product defaults | Row in [`admissions.yml`](admissions.yml) | Schema v2; exactly one evidence-bound record (LFM2.5-350M MQ4 gfx1201 retained-PM4). No inferred/wildcard rows. Runtime defaults must match loaded fixture evidence (`retained_fixture_evidence()` post-load) and fail closed on mismatch. |
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
