<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# radiowave — vendored crate

This crate is a **vendored copy**, not hipfire-original code.

| | |
|---|---|
| Upstream | https://github.com/Kaden-Schutt/redline |
| Upstream path | `crates/radiowave` |
| Vendored at commit | `f4a0994e2315645b74eef8dec9305c58d252e699` |
| Vendored on | 2026-07-25 |
| Branch | `ds4-gfx1151-opt` |
| License | Apache-2.0 (NOT hipfire's workspace MIT) |

## Why vendored rather than a git dependency

hipfire's CI runs `cargo build --release --workspace --all-targets --locked`
with no network guarantee. A git dependency on a second repository would make
that gate depend on network reachability and on the upstream repo's visibility.
Vendoring keeps the required CI build self-contained.

The cost is drift: there are now two copies. Treat **upstream as the source of
truth**. Land changes there first, then re-vendor.

## Manifest divergence from upstream

The upstream `Cargo.toml` inherits seven keys from redline's
`[workspace.package]`. hipfire's workspace cannot supply them, so this copy
pins them explicitly:

| key | upstream (inherited) | here (explicit) | why |
|---|---|---|---|
| `edition` | 2024 | 2024 | hipfire workspace is 2021 |
| `license` | Apache-2.0 | Apache-2.0 | hipfire workspace is MIT — must not inherit |
| `rust-version` | 1.85 | 1.85 | not defined in hipfire workspace |
| `authors` / `repository` / `homepage` | redline | redline | not defined in hipfire workspace |
| `version` | 0.1.0 (redline) | 0.1.0 | keeps upstream version visible; hipfire is 0.3.0 |

Everything under `src/`, `include/` and `tests/` is byte-identical to upstream
at the pinned commit. If you change a file here, record it in this table or the
next re-vendor will silently revert it.

## Re-vendoring

```bash
SRC=../redline-pub                 # or a fresh clone of the upstream repo
rm -rf crates/radiowave/{src,include,tests,README.md}
cp -r "$SRC/crates/radiowave"/{src,include,tests,README.md} crates/radiowave/
# then update the commit hash above and in crates/radiowave/Cargo.toml
```

Do **not** overwrite `Cargo.toml` or this file — both carry hipfire-local
divergence.

## What this crate is for here

Radiowave owns the policy boundary between HIP source and LLVM/AMDGPU: it
injects reviewed source-level lowering rules, invokes hipcc, inspects the
emitted code object, and records reproducible build evidence.

`rdna-compute` now uses Radiowave's existing-code-object certification path to
bind gfx1151 JIT artifacts to exact hashes and fail-closed mutable-read cache
classes before retained replay consumes them. Recipe selection remains future
work — in particular moving the per-stride cross-lane lowering table into
`radiowave::recipes` so new arch ports inherit it instead of re-deriving it.
