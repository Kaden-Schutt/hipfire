# hipfire: Rust-Native Inference Engine for AMD RDNA GPUs

## Mission

hipfire is a Rust-native ML inference engine (eventually training) for AMD RDNA
GPUs, built on a single HIP/ROCm-direct compute backend — no Python in the hot
path, no Vulkan/cross-vendor layer. It runs natively across RDNA generations
(RDNA1 gfx1010 → RDNA4 gfx1201) without lying about the hardware identity (no
`HSA_OVERRIDE_GFX_VERSION`). The project merges three efforts into one pipeline:

1. **autorocm** — map and unlock ROCm on consumer RDNA hardware
2. **autokernel** — optimize HIP/compute kernels per architecture
3. **hipfire** — the Rust-native inference engine itself

## Architecture (orientation)

User-facing entry is a Bun/TypeScript CLI (`cli/`) that posts to — or spawns —
the inference daemon (`crates/hipfire-runtime/examples/daemon.rs`). Kernels are
HIP under `kernels/src/`, JIT-compiled via `hipcc` and cached as `.hsaco` per
GPU arch.

Layer sketch (source wins on conflict; overview prose in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

- **hip-bridge / hsa-bridge** — `dlopen` FFI to HIP/HSA (no link-time ROCm).
- **rdna-compute** — kernel dispatch, hipGraph capture, JIT loader, arch predicates.
- **hipfire-dispatch** — per-family dispatch (quant × arch × flags).
- **hipfire-runtime** — KV, sampler, token loop, loaders, daemon.
- **hipfire-quantize** — CPU encoder (builds without GPU).
- **hipfire-arch-\*** — one crate per model family, keyed by `arch_id`
  ([docs/architecture-ids.md](docs/architecture-ids.md)).
- **redline / redline-dispatch / redline-rocr** — retained AQL/PM4 substrate and
  public ROCr lowering. Not the product HTTP transport. Normative procedure:
  [docs/REDLINE.md](docs/REDLINE.md).

Do not paste model registries, config key tables, or env-var matrices here —
owners: [docs/MODELS.md](docs/MODELS.md), [docs/CONFIG.md](docs/CONFIG.md),
[docs/env-vars.md](docs/env-vars.md). Navigation: [docs/INDEX.md](docs/INDEX.md).

## Building and local checks

```text
cargo build --release --workspace --all-targets --locked
cargo build --release --example daemon --features deltanet -p hipfire-runtime
./scripts/no-gpu-ci.sh
./scripts/install-hooks.sh
```

There is no GPU CI runner yet — GPU correctness and perf evidence run locally.
Automatic no-GPU CI is not model coherence, serve semantics, or perf admission.

Conventions:

- Don't `--no-verify`.
- Prefer `scripts/fmt-changed.sh` over bare workspace `cargo fmt`.
- Don't hand-edit `registry/v1.json` (generated from `cli/registry.json` via
  `scripts/registry_gen.py`).

## Validation (mandatory routing)

**Sole route selector:** [docs/VALIDATION.md](docs/VALIDATION.md).

- Fixed `scripts/coherence-gate*.sh` batteries are **retired** — never use them
  as acceptance, merge bar, or promotion evidence.
- There is **no universal replacement gate**. Pick the narrowest named route.
- Fail closed on unknown claim classes.
- Redline: manual capture/shadow is not timed product-arm route proof; full
  Redline-attributed promotion follows the ladder in
  [docs/REDLINE.md](docs/REDLINE.md).
- Admissions only via [docs/admissions.yml](docs/admissions.yml) (empty = none).

## Perf benchmarking (kernel / product claims)

Before claiming any tok/s win, read
[docs/methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).

Operational invariants (do not re-derive each session):

- **Warm kernel cache and DPM first** (throwaway forwards or
  `HIPFIRE_DPM_WARMUP_SECS=10`). Cold first runs are not representative.
- **JIT tax is per (config × kernel-shape).** Warm each matrix cell. A slowdown
  that survives a second pass is not JIT.
- **Cross-commit claims:** fresh process via `scripts/probe_commits.sh` (or the
  methodology's equivalent fresh-process protocol). Check the methodology
  negative-result log before restarting a known-dead experiment.
- **Δ ≥ 5% investigation rule.** Re-warm and re-median first. If the median
  holds, investigate (occupancy / rocprof / env / flags / bisect) — do not
  dismiss under inflated session-noise claims. Gains still need the path-specific
  evidence VALIDATION names for that claim class.
- **Byte-identical prompts** for any cross-session tok/s or τ comparison.
  Record prompt md5. Prefer committed files under `benchmarks/prompts/` — never
  canonical benches under `/tmp/`. One newline can swing τ dramatically; the
  engine collapses `\n{3,}` → `\n\n` by default (`HIPFIRE_NORMALIZE_PROMPT=0` /
  `prompt_normalize=false` to opt out when raw whitespace is load-bearing).
- **Historical bench tables stay historical**
  ([docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/perf-checkpoints/](docs/perf-checkpoints/)).
  They are not current floors or admissions.

Diagnosing memset pressure: `HIPFIRE_MEMSET_DUMP=1` (track_caller sites). Note
`memset_async` is gated on `active_stream` being `Some`.

## Skills

**Sole executable skill root:** [`.agents/skills/`](.agents/skills/).

Load skills from there (tester, diag, autoheal, arch-port, kernel-tuning,
kernel-atlas, astrea, rebase-onto-modular, redline-retained-replay, …).

Thin Redline discovery hook (workflow only; policy stays in REDLINE.md):
[`.agents/skills/redline-retained-replay/`](.agents/skills/redline-retained-replay/).

## Hard-won measurement and debugging rules

- **No `grep` / `find` / shell glob for codebase search.** Use the dedicated
  search/glob tools (or language-server / structured query paths). Do not
  shell out to `grep`, `rg`, `find`, or `ls **` for discovery.

- **Measure spec-decode durability on the daemon**, not demo harnesses
  (`dflash_spec_demo` / `mtp_only_demo` under-report serving behavior).
- **Offline spec-decode proxies mis-rank drafters** — confirm online before
  shipping a drafter decision.
- **Garbage output → swap to a known-good model first** before engine debugging.
  Wrong-recipe weights burn cycles.
- **Byte-parity under stochastic state is meaningless** — pin FP32 +
  `HIPFIRE_DETERMINISTIC=1` for bit-exact claims (e.g. Q8 DeltaNet state).
- **Tight stddev on a spec-decode bench is suspicious**, not reassuring. Eyeball
  decoded text when τ is unusually high (attractor failures fake statistical wins).
- **`scripts/install.{sh,ps1}` copy `cli/` recursively** and prune test/bench
  artifacts by pattern (`*.test.ts`, `test_*.ts`, `bench_*.ts`). New runtime
  helpers that look like tests must be renamed or they will not ship.

## GPU lock protocol (multi-agent)

Coordinate GPU access with `scripts/gpu-lock.sh`. Coordination is **manual** —
no committed auto-acquire hook.

- Lock file: `/tmp/hipfire-gpu.lock` (`HIPFIRE_GPU_LOCKFILE` override)
- `flock(1)` on an open fd — kernel releases on holder death. **Never `rm` the
  lockfile** (unlinking lets a second acquirer lock a fresh inode).
- Reentrant within a process tree (`HIPFIRE_GPU_LOCK_OWNER`).
- Manual: `source scripts/gpu-lock.sh && gpu_acquire "<branch>" && gpu_release`
- Status: `gpu_status`. Regression (no GPU): `bash scripts/test-gpu-lock.sh`

## Reference lineage

- ncdrone/rustane-style Rust-native `dlopen` FFI (their `ane-bridge` → our
  `hip-bridge` / `hsa-bridge`)
- Karpathy-style experiment + fixed-eval discipline (here: claim-scoped routes
  in VALIDATION, not a single universal battery)
- Mesa radeonsi/RADV, amdgpu KMD ioctl surface, ROCm HSA runtime as driver refs

## Orchestration and experiment tracking

Heavy or parallelizable work goes to subagents; the orchestrator synthesizes and
orders tests. **Git-commit every meaningful state change** — including failed
approaches — with structured results. The history is the research; document WHY
something failed so the search space narrows.

Project archaeology (phases 0–4 bootstrap) lives in `git log` and historical
findings trees. Day-to-day work is under `crates/` and the routes above.

## Rules

1. **No Python in the inference hot path.** Tooling/benchmarks only.
2. **Git commit everything.** Every experiment, finding, and failed approach.
3. **Document failures explicitly.** Concrete error codes beat "it didn't work."
4. **Portability matters.** Consider RDNA2/3/4. Single-arch-only is a hack
   unless explicitly scoped and gated.
5. **No `HSA_OVERRIDE_GFX_VERSION` as a permanent solution.** Throwaway local
   tests only; the engine must not depend on lying about hardware identity.
6. **When blocked, search.** GitHub issues, AMD docs, Mesa, forums.
7. **No Vulkan / wgpu / cross-vendor backend.** Out of scope (issue #44, closed
   2026-04-25). Single HIP/ROCm-direct backend; pivot HIP-side if a path stalls.

## Local / personal config

Machine-specific fleet notes do **not** belong in this committed file. Keep them
in a personal auto-loaded ignore path (recommended: `~/.claude/hipfire-fleet.md`).
Per-repo `CLAUDE.local.md` is also gitignored but worktree-local.

@~/.claude/hipfire-fleet.md
