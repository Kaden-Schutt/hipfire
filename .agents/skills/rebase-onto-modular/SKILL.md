---
name: rebase-onto-modular
description: Use when porting a hipfire feature/fix branch authored against pre-0.1.20 master onto post-modular master. Walks through the engine→hipfire-runtime + per-arch-crate split mechanically, then surfaces semantic conflicts that need human judgment.
---

# rebase-onto-modular

Hipfire 0.1.20 split the monolithic engine crate (historical/pre-modular path
`crates/engine/`, intentionally absent from the checkout) into
`hipfire-runtime` plus per-family `hipfire-arch-*` crates. Branches authored
against pre-modular trees need path, import, and sometimes dispatch rewrites
before they compile on current master.

**The old helper `scripts/rebase-onto-modular.sh` was removed** (see CHANGELOG
v0.3.0 — stale modular-rebase helper). This skill is the manual cutover
playbook. Do not search for or reintroduce that script.

## Reach for this when

- The branch still has historical/pre-modular `crates/engine/` (intentionally
  absent from the checkout), `use engine::`, or
- A PR was cut from pre-0.1.20 history and must land on post-modular master.

## Do not use when

- The branch was authored on post-0.1.20 modular master already.
- Changes touch only paths that never lived under `engine/` and need a plain
  `git rebase` onto current master (confirm with the map below first).
- You only need a new arch on an already-modular tree — copy
  `crates/hipfire-arch-toy/` and follow `docs/ARCHITECTURE.md` /
  `.agents/skills/hipfire-arch-port/` instead.

## Current topology (derive from tree; owners win on drift)

Authoritative overviews — **look these up; do not maintain a second crate
inventory in this skill:**

- [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md) — crate roles + request lifecycle
- [`docs/architecture-ids.md`](../../../docs/architecture-ids.md) — `arch_id` → crate
- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) — “Crate topology”
- CHANGELOG **v0.1.20** — original migration map
- Workspace members: root `Cargo.toml` `[workspace].members` and `ls crates/`

Mandatory before path rewrites on a live tree:

```bash
# confirm post-modular layout (no crates/engine/)
ls crates/
# optional: workspace member list
rg -n '^members' -A40 Cargo.toml
```

Bring-up contract: `hipfire_runtime::arch::Architecture` in
`crates/hipfire-runtime/src/arch.rs`. Forward stays **off** the trait (static
dispatch per concrete arch). Toy template for a new modular arch crate:
`crates/hipfire-arch-toy/`.

## Path / import cutover map

Historical/pre-modular source paths below are intentionally absent from the
checkout; map each to the post-modular destination (from CHANGELOG v0.1.20 —
apply cleanly, no shims, no `engine` re-exports):

| Pre-modular (historical; intentionally absent) | Post-modular |
|---|---|
| `crates/engine/src/lib.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-runtime/src/lib.rs` |
| `crates/engine/src/qwen35.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-arch-qwen35/src/qwen35.rs` |
| `crates/engine/src/qwen35_vl.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs` |
| `crates/engine/src/image.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-arch-qwen35-vl/src/image.rs` |
| `crates/engine/src/llama.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-runtime/src/llama.rs` (+ `hipfire_arch_llama::llama` re-export) |
| `crates/engine/src/speculative.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-arch-qwen35/src/speculative.rs` (and related spec modules) |
| `crates/engine/src/pflash.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-arch-qwen35/src/pflash.rs` |
| `crates/engine/src/loop_guard.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-runtime/src/loop_guard.rs` |
| `crates/engine/src/sampler.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-runtime/src/sampler.rs` |
| `crates/engine/src/prompt_frame.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-runtime/src/prompt_frame.rs` |
| `crates/engine/src/eos_filter.rs` (historical/pre-modular; intentionally absent from the checkout) | `crates/hipfire-runtime/src/eos_filter.rs` |

Import rewrites:

- `use engine::…` → `use hipfire_runtime::…` for runtime symbols
- Arch-specific symbols → `use hipfire_arch_<family>::…`
- Cargo: drop `engine` path dep; depend on `hipfire-runtime` and only the arch
  crates you call
- Shared GEMV / `KvCache` / dequant helpers that used to hang off
  `engine::qwen35` generally live under `hipfire_runtime::llama` (still the
  shared transformer home until a future physical split)

**Clean cutover:** migrate every caller in the branch. Leave no `engine`
alias, deprecated path, or dual-compile shim.

## Workflow

### 1. Backup and rebase onto current master

```bash
git tag rebase-onto-modular-backup-$(date -u +%Y%m%dT%H%M%SZ)
# working tree must be clean; do not run this on master itself
git fetch origin
git rebase origin/master   # or the repo's current default integration branch
```

On structural conflicts that are pure renames, prefer the post-modular side
for file location, then re-apply the branch’s additive logic onto the new
paths. Agents must not force-push or hard-reset without explicit human OK.

### 2. Mechanical rewrite

1. Move any remaining files still under historical/pre-modular `crates/engine/`
   (intentionally absent from the checkout) to the table above
   (or the correct newer arch crate if the code is family-specific).
2. Rewrite imports and Cargo.toml deps workspace-wide on the branch.
3. Grep until clean: `\bengine::`, historical/pre-modular `crates/engine`
   (intentionally absent from the checkout), `path = .*/engine`.
4. Update feature flags: consumers select arches via `hipfire-runtime`
   features / direct arch-crate deps — not a monolithic engine feature.

### 3. Semantic conflict triage

| Failure shape | What to do |
|---|---|
| Missed `engine::` import | Map symbol to runtime vs arch crate; fix call site (no blanket `pub use`). |
| New `arch_id` match arms in `daemon.rs` | Prefer arch crate + `Architecture` bring-up triple; register id per `docs/architecture-ids.md`. Do not grow a permanent parallel ladder when a trait hook exists. |
| Cross-arch helper via old `qwen35::*` | Use `hipfire_runtime::llama::{weight_gemv, KvCache, dequantize_*, …}` or the owning arch’s public API. |
| `sampler` / `loop_guard` / `prompt_frame` / `eos_filter` missing | Top-level modules on `hipfire_runtime`. |
| Missing `Architecture` trait | `use hipfire_runtime::arch::Architecture;` |
| Broken `image` / VL imports | `hipfire_arch_qwen35_vl::…` |
| Dispatch / kernel selection moved | Hot kernels route through `rdna-compute` + `hipfire-dispatch` families — do not re-embed arch-specific ISA ladders inside a random arch crate. |
| Spec / DFlash symbols | Check `hipfire-runtime` spec seams and `hipfire-arch-qwen35` (or the arch that owns the drafter). Inventory snapshot: `docs/speculation-support-inventory.md` (historical — verify in source). |

### 4. Verify (claim-scoped; no universal gate)

Build what you touched:

```bash
cargo build --release --features deltanet --workspace
# optional, if the branch had unit coverage:
cargo test --lib --features deltanet --workspace
```

Then pick **narrow** evidence from [`docs/VALIDATION.md`](../../../docs/VALIDATION.md)
for the change class (docs-only, kernel numeric, forward parity oracle, serve
semantics, perf protocol, arch-port procedure). Do **not**:

- require retired `scripts/coherence-gate-*.sh` as acceptance
- invent a one-script replacement gate
- treat green no-GPU CI as GPU correctness

Perf-sensitive branches: follow
`docs/methodology/perf-benchmarking.md` and, when applicable,
`scripts/speed-gate.sh` baselines — measured, not admission.

### 5. Push (human-approved)

```bash
git push --force-with-lease origin <branch>
```

`--force-with-lease` only after the human accepts the rewritten history.

## Rollback

```bash
git reset --hard rebase-onto-modular-backup-<timestamp>
git tag -d rebase-onto-modular-backup-<timestamp>
```

Agents must ask before destructive rollback or force-push.

## Related

- New GPU arch port (not model-family crate split):
  [`.agents/skills/hipfire-arch-port/`](../hipfire-arch-port/SKILL.md)
- Validation routes: [`docs/VALIDATION.md`](../../../docs/VALIDATION.md)
- Trait + overrides: `crates/hipfire-runtime/src/arch.rs`
- Toy template: `crates/hipfire-arch-toy/`
