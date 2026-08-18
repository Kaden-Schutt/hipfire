# DS4 DSpark / gfx1151 — handoff

**Date:** 2026-08-03
**Branch:** `ds4-cdna-test-fail` @ `cbcaad9cb` (origin = `warpfront/hipfire`)
**Worktree:** `/home/kaden/ClaudeCode/autorocm/hipfire/.claude/worktrees/ds4-mi300x-agentmaxx`
**Rig:** hipx, HIP dev 1 = Radeon 8060S gfx1151 (Strix Halo, 103 GB)
**Evidence:** `hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-dspark-genre-and-gates/`

---

## 1. Where the numbers actually are

Production daemon, `hipfire run`, `-n 128`, `--temp 0`, 0731 MQ2R trunk,
fresh daemon per cell, both orderings, all reproduced exactly:

|genre|AR|DSpark|tau|accept|
|---|---|---|---|---|
|**code**|27.5|**34.0** (E8 sidecar) / 33.4 (Q8)|3.024|67%|
|prose|27.7|27.0|2.051|35%|

**DSpark wins by 24% on code.** The "DSpark loses to AR" premise that drove
this workstream came from benchmarking a single prose prompt. Certification
bar is 35-40 tok/s; we are at 34.0 on code.

---

## 2. READ THIS FIRST: uncommitted state on hipx

`hipx:/home/kaden/hipfire-ds4-twostage` **is not a git repo** and has
diverged. It carries four anchor patches. Three have equivalents committed
on the branch; **one is hipx-only and must never be backported.**

|patch|branch commit|note|
|---|---|---|
|E8 arm in `dspark_wo_project` (`forward.rs` ~12519)|`7706e2bf2`|in sync|
|`hipfire run` `attempt_id` (`main.rs:1964`)|`3d74c9c47`|in sync|
|E8 batched-GEMV default `unwrap_or(0)`->`unwrap_or(8)` (`forward.rs:783`)|`cedcf379b`|in sync|
|**CDNA3 arm removed from `gemm_f16_x_f16_auto`**|**NONE — hipx only**|**do NOT backport**|

The last one exists because hipx's `rdna-compute` lacks the
`gemm_f16_x_f16_mfma_gfx942` binding that `398c3d176` added to the branch.
Removing that arm on the branch would break the CDNA3 DSpark port.

**Never wholesale-copy a file into the hipx tree.** Doing exactly that broke
its build mid-session (my `forward.rs` referenced the missing CDNA3
binding). Patch by anchor with a python script that asserts a unique match
and refuses otherwise.

---

## 3. Methodology rules earned the hard way

1. **Swapping a model file on disk does nothing to a running daemon.** It
   keeps serving the previously-loaded weights and yields a self-consistent,
   reproducible, WRONG answer. Every weight A/B MUST `pkill -x daemon`
   between cells and verify the on-disk md5 at run time. This invalidated
   two of my own results before I caught it.
2. **`pkill -f <path>` kills your own SSH session** — the session's command
   line contains that path. The `[d]aemon` bracket trick does NOT save you
   when the same command line also exports an unbracketed copy (e.g.
   `HIPFIRE_DAEMON_BIN=.../examples/daemon`). Use **`pkill -x daemon`**.
3. **The block controller is stateful across requests within a daemon
   lifetime.** Back-to-back configs contaminate each other; the second run
   inherits a converged controller.
4. Every A/B in **both orderings** (an 82 GB reload against a churned page
   cache once produced a fake 39% regression).
5. Genre dominates. Never quote a DSpark number without naming the prompt.

---

## 4. Settled — do not re-litigate

|question|verdict|
|---|---|
|Sidecar recipe mismatch (Q8F16 draft vs MQ2R trunk)|**Acceptance-neutral.** 67% with either sidecar. E8 is +1.8% on throughput only (smaller file, fewer bytes in `draft_block`).|
|Confidence threshold truncating proposals|**Inert.** `CONF_THRESHOLD=0` and the 0.3 default are byte-identical.|
|Deeper speculation / bigger block|**Counterproductive on MoE.** Fixed block=5 raises tau 3.02->4.10 but DROPS throughput 33.4->27.1. Verify traffic scales with batch; MoE is at ~72% of the ~256 GB/s roofline. Controller settling on block=2 is correct.|
|`--draft-max`|Does not reach DSpark (MTP/n-gram only).|
|`k_top` 6->4 in verify|Rejected — changes what the target emits.|
|V3 paper's 60-80% MTP acceptance as our ceiling|Invalid — different architecture (sequential enorm/hnorm), same-precision target.|
|Draft quality|**Not the problem.** Per-position agreement at block=5 is 100/74/63/40/37%.|

---

## 5. Next lever: graph-capture the verify

Verify is ~86% of the DSpark window. DFlash's verify IS hipGraph-captured
and measured **+14%** (25.6 -> 29.2 tok/s). At +14% on 34.0 that projects
to **~38.8 tok/s** — inside the 35-40 band.

**The only blocker:** DS4 uploads per-window values INSIDE the region that
would be captured, so replay would bake in capture-day contents forever.

|site|what|
|---|---|
|`forward.rs:11186`|tokens|
|`forward.rs:11192`|positions|
|`forward.rs:11208`|`n_valid_swa_arr`|
|`forward.rs:9532`|`n_active_topk_arr`, per mixed layer|
|`forward.rs:9351`|`n_per_batch` staging, per ratio-4 layer|
|`forward.rs:7601,7670`|compressor-fallback H2D, per ratio-4 layer|

The fix follows DFlash: hoist these into an
`upload_prefill_batch_inputs`-style pre-pass called OUTSIDE capture
(`crates/hipfire-arch-qwen35/src/qwen35.rs:6181-6197`, invoked at
`speculative.rs:2372`).

Everything else already qualifies: B=5 constant, PBS persistent
(`spec_impl.rs:280-291`), fixed-size KV rings (no per-window growth), lazy
allocs `is_none`-guarded and warmable outside capture, head/argmax outside.

Also unresolved: **~26.66 ms of the verify is untimed** (timed kernels sum
to 48.24 of 74.90 ms). Adding `profile::begin_timer` to the untimed
gfx1151 E8 batched/grouped GEMV launch sites in
`crates/rdna-compute/src/gemv.rs` is zero-risk and partitions that gap.

---

## 6. Repro

```bash
ssh hipx
cd /home/kaden/hipfire-ds4-twostage
export HIPFIRE_DAEMON_BIN=$PWD/target/release/examples/daemon
export HIP_VISIBLE_DEVICES=1
D=/home/kaden/.cache/hipfire-surgery
M=$D/deepseek-v4-flash-0731.mq2r
CODE='Write a Python function that merges two sorted lists into one sorted list, then explain how it works.'

pkill -x daemon; sleep 5           # MANDATORY between cells
HIPFIRE_DSPARK_DEBUG=1 ./target/release/hipfire run "$M" "$CODE" \
  --spec dspark -n 128 --temp 0 2>&1 | tee /tmp/run.log

python3 /home/kaden/ds4-gfx1151-evidence/2026-08-03-ds4-dspark-genre-and-gates/dspark_poscurve.py \
  /tmp/run.log mycell     # -> per-position accept curve + tau
```

Build after a patch (note: **no** `--features` for the arch crate on hipx;
the daemon DOES need `--features deltanet`):

```bash
cargo build --release -p hipfire-arch-deepseek4 --lib
cargo build --release --features deltanet -p hipfire-runtime --example daemon
```

Useful knobs: `HIPFIRE_DSPARK_PROFILE=1` (phase ms),
`HIPFIRE_DSPARK_DEBUG=1` (per-window drafts vs target_pick),
`HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0` (fixed block = `cfg.block_size` = 5),
`HIPFIRE_DSPARK_ZERO_CTX=1` (ctx-wiring sanity; healthy = accept collapses
to ~4%, drafts 0% identical).

---

## 7. Open

- Widen the genre result beyond one prompt per class before certifying —
  use the committed `benchmarks/prompts/` set. Prompt mix will dominate any
  headline figure.
- Reconcile the hipx anchor patches with the branch, or rebuild that
  checkout from git.
- Long-context (1M) blocker, unrelated to speed: **KV is hardcoded F32** on
  DS4 (`forward.rs:1674`); there is no `kv_mode` plumbing in the crate, so
  `--kv-mode q8` is silently ignored. Irrelevant at 128 tokens, decisive at 1M.
- Multi-turn + `reasoning_effort` hang (repro `/root/ds4_chain_think.sh` on
  mi300x); non-thinking multi-turn is 8/8.
