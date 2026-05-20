---
name: rebase-onto-modular
description: Use when porting a hipfire feature/fix branch authored against pre-0.1.20 master onto post-modular master. Walks through the engine→hipfire-runtime + per-arch-crate split mechanically, then surfaces semantic conflicts that need human judgment.
---

# rebase-onto-modular

Hipfire's 0.1.20 release split the monolithic `engine` crate into a runtime
crate plus per-arch crates. Branches authored before 0.1.20 need their
import paths, Cargo deps, and (sometimes) trait-dispatch sites rewritten
before they compile against current master.

This skill runs the mechanical 80% via `scripts/rebase-onto-modular.sh`,
then guides human resolution of the remaining 20%.

## When to invoke

- A contributor's PR was authored against pre-modular master
  (look for `crates/engine/src/`, `use engine::`, or `engine = { path =`
  in the diff).
- They (or you, on their behalf) want to bring it onto post-modular master.

## Topology recap (post-0.1.20)

```
crates/
  hipfire-runtime/         ← was crates/engine/, plus 4 new modules
                             (loop_guard, sampler, prompt_frame, eos_filter)
  hipfire-arch-qwen35/     ← qwen35.rs + speculative.rs + pflash.rs
  hipfire-arch-qwen35-vl/  ← qwen35_vl.rs + image.rs
  hipfire-arch-llama/      ← facade over runtime::llama (real split = PR 14)
  hipfire-arch-toy/        ← reference template for new-arch contributors
  rdna-compute/            ← unchanged (kernel dispatch + RDNA-arch routing)
  hip-bridge/              ← unchanged (HIP/ROCm FFI)
  hipfire-quantize/        ← unchanged (quantizer CLI)
```

## Workflow

### 1. Run the mechanical rebase script

From the root of the contributor's worktree:

```bash
./scripts/rebase-onto-modular.sh
```

What it does:
- Creates a backup tag (`rebase-onto-modular-backup-<timestamp>`) so any
  step can be undone.
- Refuses to run if the working tree is dirty or you're on master.
- Rebases your branch onto current `origin/master`, favoring master on
  conflicts (the assumption: structural conflicts are about the rename,
  not about your changes; we re-apply your additive logic in the next step).
- Applies the path-rename map (engine/src/X.rs → hipfire-runtime or arch crate).
- Rewrites `use engine::*` → `use hipfire_runtime::*` (or arch crate), and
  Cargo dep references.
- Tries `cargo build --release --features deltanet --workspace` to surface
  any remaining issues.

If the script's mid-step says "rebase produced conflicts," follow its
on-screen instructions:
1. `git checkout --theirs <conflicted file>`
2. `git add <file>`
3. `git rebase --continue`
4. Once rebase completes, re-run the script.

### 2. Address common semantic conflicts

After the script's rewrites, `cargo build` may still fail. The usual
suspects:

| Failure shape | What to do |
|---|---|
| `engine::X` import the script missed | Check `scripts/rebase-onto-modular.sh`'s `PATH_MAP` and `REWRITES`. If your branch uses an unusual import path, manually rewrite or extend the script's map. |
| `arch_id` match-arm in your diff | Daemon's arch dispatch now goes through `<Architecture>::*` for the bring-up triple. If your branch added a new `arch_id => ...` arm in `daemon.rs::generate()`, port it into the new pattern: introduce a new arch crate (use `hipfire-arch-toy/` as template) or wire into an existing arch's dispatch. |
| Direct `qwen35::*` reach from non-qwen35 code | Most cross-arch helpers (`weight_gemv`, `KvCache`, `dequantize_*`, RoPE) live in `hipfire_runtime::llama` (still — physical split waits for PR 14 transformer extraction). Replace `engine::qwen35::weight_gemv` → `hipfire_runtime::llama::weight_gemv`. |
| `sampler` / `loop_guard` / `prompt_frame` / `eos_filter` not found | These moved to the runtime crate's top-level modules. `use hipfire_runtime::sampler::*` etc. |
| Missing `Architecture` trait import | `use hipfire_runtime::arch::Architecture;` |
| `image.rs` imports broken | Vision preprocessing moved to `hipfire-arch-qwen35-vl/src/image.rs`. Replace `engine::image::*` → `hipfire_arch_qwen35_vl::image::*`. |

### 3. Verify

```bash
cargo build --release --features deltanet --workspace
cargo test --lib --features deltanet --workspace
```

Both must pass before pushing.

If your branch had perf-sensitive changes:
```bash
./scripts/speed-gate.sh
```

If your branch touched kernels / quant / dispatch / fusion / rotation /
forward-pass:
```bash
./scripts/coherence-gate-dflash.sh
```

### 4. Push

```bash
git push --force-with-lease origin <your-branch>
```

`--force-with-lease` is safer than `--force` — it'll refuse if someone
else updated the branch concurrently.

## Rollback

If the rebase produces something worse than what you started with:

```bash
git reset --hard rebase-onto-modular-backup-<timestamp>
git tag -d rebase-onto-modular-backup-<timestamp>
```

The backup tag is created at the very start of the script and persists
until you delete it.

Agents must ask the user before running destructive rollback commands or
force-pushing a rewritten branch. The commands above are human recovery
instructions, not permission to discard unreviewed work.

## When NOT to use this skill

- Brand-new branches authored against post-modular master — they don't
  need rebase.
- Branches that touch ONLY `kernels/src/` or `crates/rdna-compute/` —
  those crate paths are unchanged; just `git rebase origin/master`.
- Branches that touch ONLY `crates/hip-bridge/` — same.
- Branches touching `crates/hipfire-quantize/` — same.

## Reference

- Migration map: `CHANGELOG.md` 0.1.20 entry
- Crate topology: `CONTRIBUTING.md` "Crate topology" section
- Architecture trait: `crates/hipfire-runtime/src/arch.rs`
- Toy arch template: `crates/hipfire-arch-toy/`
