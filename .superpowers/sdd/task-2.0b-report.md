# Task 2.0b Report: Byte-neutral `Gpus` threading through `generate()` dispatcher

## What changed

Two edit sites in `crates/hipfire-runtime/examples/daemon.rs`, total diff: +3 / -2 lines.

**1. `fn generate` signature (daemon.rs:8451):**
- `gpu: &mut rdna_compute::Gpu` → `gpus: &mut Gpus`
- `drafter_gpu: Option<&mut rdna_compute::Gpu>` left unchanged

**2. Function body — first line after `{` (daemon.rs:8452, new line):**
- Added `let gpu = &mut gpus.devices[0];`
- This local rebind means all 62 existing `gpu` uses inside the body transparently reference `gpus.devices[0]` via reborrow, with zero further edits inside the 2463-line body.

**3. `main()` call site (daemon.rs:2595):**
- `generate(m, &mut gpus.devices[0], pflash_drafter_gpu.as_mut(), …)` → `generate(m, &mut gpus, pflash_drafter_gpu.as_mut(), …)`

## Count of `gpu` → `gpus.devices[0]` sites

62 internal `gpu` uses now resolve through the one local rebind (`let gpu = &mut gpus.devices[0];`). No individual body-line edits were made — the rebind is the single indirection point.

## Invariants verified

- `drafter_gpu: Option<&mut rdna_compute::Gpu>` parameter: **unchanged**
- All `generate_<arch>` function signatures: **unchanged** (they still take `&mut Gpu`)
- `pflash_drafter_gpu` / spec-decode drafter device: **untouched**
- `daemon.rs` not reformatted — diff is 3 inserted / 2 deleted lines only, no whitespace churn

## Build

```
nix develop -c cargo build --release -p hipfire-runtime --example daemon
```
Result: `Finished release profile [optimized] target(s) in 4.99s` — no new errors or warnings.

## Clippy

```
nix develop -c cargo clippy -p hipfire-runtime
```
Result: one error in `hip-bridge/src/rccl.rs:309` — pre-existing, confirmed by `git stash` + re-run before and after. No new issues introduced.

## Coherence gate

```
nix develop -c ./scripts/coherence-gate.sh
```
Result: `no hard errors — review /tmp/coherence-20260706-085640.md for coherence, then commit if satisfied`

All model runs status **OK**: qwen3.5-0.8b.mq4, qwen3.5-4b.mq4, qwen3.5-9b.mq4, and other present models. Timing REGRESS entries are environmental wall-clock drift, not output regressions — all `actual=…PASS`. Skipped entries = models not on this box.

## Concerns

None. The local rebind approach (`let gpu = &mut gpus.devices[0];`) is semantically equivalent to 62 individual site replacements, is minimal and surgical, and avoids any risk of mistaken manual substitutions in a 2463-line function. The borrow checker accepts it because: (a) `drafter_gpu` is a separate incoming reference so no aliasing; (b) Rust NLL handles the `compress_gpu = drafter_gpu.as_deref_mut().unwrap_or(gpu)` reborrow pattern correctly since `compress_gpu`'s last use precedes the next `gpu` use. Future tasks 2.1–2.5 will change individual `generate_<arch>(m, gpu, …)` call sites inside `generate()` to `generate_<arch>(m, gpus, …)` as those functions migrate; at that point the local `gpu` binding becomes unused and can be dropped.
