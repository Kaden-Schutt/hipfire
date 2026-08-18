# DS4 DSpark: the "loses to AR" result was a prose prompt

**Date:** 2026-08-03
**Arch:** gfx1151 (Strix Halo, Radeon 8060S), HIP dev 1
**Trunk:** `deepseek-v4-flash-0731.mq2r` (0731 checkpoint, MQ2R)
**Method:** production daemon via `hipfire run`, `-n 128`, `--temp 0`,
fresh daemon (`pkill -x daemon`) before every cell, every A/B in both
orderings.

## Headline

DSpark does **not** lose to autoregressive decode. It wins by 24% on code
and ties on prose. The long-standing "25.8 vs AR 27.9" figure that framed
this whole workstream was measured on a single **prose** prompt.

|prompt genre|AR|DSpark|tau|accept|verdict|
|---|---|---|---|---|---|
|code|27.5|**34.0**|3.024|67%|**1.24x win**|
|prose|27.7|27.0|2.051|35%|-2.5%, a wash|

Both orderings reproduce exactly (`prose_DSPARK` = `prose_DSPARK2` = 27.0;
`code_DSPARK` = `code_DSPARK2` = 33.4 on Q8, 34.0 on E8).

This is precisely the pattern [`AGENTS.md`](../../AGENTS.md) already
documents for DFlash — "Code prompts: 4x win on 27B / 2.6-3x on 9B. Prose
prompts: tie or small loss on 9B (-20%, draft-target alignment issue, NOT a
bug)." DSpark on DS4 is the same shape, milder.

Prompts used:
- code: `Write a Python function that merges two sorted lists into one sorted list, then explain how it works.`
- prose: `Explain in three sentences why the sky is blue.`

## The draft is good. The block controller is right.

Per-position argmax agreement, forced fixed block=5
(`HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0`):

```
 pos   match%   n
  0    100.0%    31
  1     74.2%    31
  2     63.3%    30
  3     40.0%    30
  4     36.7%    30
 mean accepted drafts/window = 3.097  (tau = 4.097)
```

Position 0 is essentially perfect. But raising the block to 5 **lowers**
throughput:

|block|tau|tok/s|
|---|---|---|
|2 (adaptive default)|3.024|**33.4**|
|5 (fixed)|4.097|27.1|

Verify traffic scales with batch and MoE is already at ~72% of Strix
Halo's ~256 GB/s roofline, so the extra 1.07 accepted tokens cost ~60 ms of
window. The `BlockController` (`crates/hipfire-runtime/src/dspark_block_controller.rs`,
start_block=2, p*=0.18) settling on 2 is the correct answer, not a bug.

**Deeper speculation is not the lever on an MoE target.** This is the
opposite of the WMMA-part result for dense targets, where B=1..13 is flat
and deeper speculation is free.

## Two of my own claims, retracted

Both were daemon-state artifacts. Recording them because the failure mode
is subtle and will recur.

1. **"The confidence threshold truncates good proposals."** RETRACTED.
   With fresh daemons, `HIPFIRE_DEEPSEEK4_DSPARK_CONF_THRESHOLD=0` and the
   0.3 default produce **byte-identical** output: 33.4 tok/s, 67% accept,
   tau 3.024, same accepted-prefix distribution. The earlier reading
   (47% accept, tau 2.42, only 14 of 52 windows proposing a second token)
   came from a run that did not restart the daemon first.

2. **"The E8 matched sidecar is slightly slower."** RETRACTED on
   throughput. Fresh daemons, md5-verified sidecar swaps:

   |sidecar|md5|tok/s|accept|tau|
   |---|---|---|---|---|
   |Q8F16|`92ed334c5afa`|33.4|67%|3.024|
   |E8-SoA matched|`bdccd8fc5321`|**34.0**|67%|3.024|
   |Q8F16 repeat|`92ed334c5afa`|33.4|67%|3.024|

   E8 is **+1.8%** — the smaller sidecar (5.79 GB vs 6.00 GB) moves fewer
   bytes in `draft_block`. The **acceptance** falsification from
   `7706e2bf2` stands unchanged: 67% either way, recipe matching does not
   move acceptance. The throughput direction in that commit message was
   wrong and is corrected here.

### The methodology rule this establishes

**Swapping a model file on disk does nothing to a running daemon.** Any
A/B over weights MUST `pkill -x daemon` between cells and verify the md5
of what is actually on disk at the moment of the run. A stale daemon
silently serves the previous artifact and produces a self-consistent,
reproducible, wrong answer.

Also: `pkill -f <path>` kills your own SSH session, because the session's
command line contains that path (e.g. via `export HIPFIRE_DAEMON_BIN=`).
The `[d]aemon` bracket trick does **not** save you when the same command
line also carries an unbracketed copy of the string. Use `pkill -x daemon`.

## DFlash review: what it does differently

Source review of the DFlash path against DSpark (agents: DraftArch,
VerifyPath, AcceptRootCause, Feasibility).

|axis|DFlash|DS4 DSpark|
|---|---|---|
|what the draft is|separately **trained** standalone small LM, arch_id=20, ~5-layer bidirectional, no own lm_head (`crates/hipfire-runtime/src/dflash.rs:7-14,94-126`)|three `mtp.{0,1,2}` blocks **extracted** from the checkpoint, ~2 GB/stage (`crates/hipfire-arch-deepseek4/src/arch.rs:1773-1914`)|
|proposal shape|**one** bidirectional block-diffusion pass over `[seed, MASK...]` giving **independent per-position marginals** (`dflash_generic.rs:30-34,663-730`)|3 chained stage passes on residual streams, then a **sequential Markov head fed its own predictions** (`dspark_core.rs:816-966`)|
|forward passes/window|1 draft + 1 verify|1 draft_block (3 stages) + run_heads + 1 verify|
|accept test|`accept_greedy_prefix`, strict argmax equality (`spec.rs:149-178`)|identical|

The Markov head compounding on its own predictions is the clearest
structural difference, and it is consistent with the measured decay
(100/74/63/40/37%). DFlash has no such autoregressive draft head.

**V3's documented 60-80% MTP acceptance is not a valid ceiling for DS4** —
that is a different architecture (sequential enorm/hnorm MTP) measured
against a same-precision target.

### Could DS4 use DFlash?

The generic harness is already target-agnostic:
`GenericDflashSpeculator` (`crates/hipfire-runtime/src/dflash_generic.rs:127,540`)
drives the drafter through the arch-generic `SpecTarget` trait and already
serves llama and qwen3 targets. Nothing in it assumes dense attention, a
KV layout, a tokenizer, or a non-MoE target.

DS4 cannot be paired **today** because:
- `Deepseek4Carrier` has no `draft_path` arm at all — it never reads
  `ctx.draft_path` (`crates/hipfire-loader/src/carriers.rs:984-1120`;
  contrast llama at `:781-817`).
- DS4's `SpecTarget` impl (`crates/hipfire-arch-deepseek4/src/spec_impl.rs:192-504`)
  lacks six hooks: `dflash_extract_layers`, `set_dflash_extract_layers`,
  `embed_row`, `lm_head_logits`, `verify_block_logits`, and `spec_advance`
  explicitly ignores `hidden_out` (`spec_impl.rs:230-236`).

DS4 already **has** the whole verify/commit half (`verify_block`,
`verify_block_capture_gpu`, `commit_prefix`, `capture_seed_main_hidden`,
`new_spec_scratch`). But adopting DFlash requires a **trained** DS4 draft,
which does not exist. That is the expensive path.

## The next lever: graph-capture the verify

DFlash's verify is hipGraph-captured and measured **+14%** (25.6 -> 29.2
tok/s). DS4's verify is not, and verify is ~86% of the DSpark window.

The blocker is specific and fixable. `forward_prefill_batch_single_chunk_captured_opts`
requires inputs uploaded **outside** the captured region
(`upload_prefill_batch_inputs`, `qwen35.rs:6181-6197`, called at
`speculative.rs:2372`). DS4 uploads inline, inside the region:

|site|what|
|---|---|
|`forward.rs:11186`|tokens|
|`forward.rs:11192`|positions|
|`forward.rs:11208`|`n_valid_swa_arr`|
|`forward.rs:9532`|`n_active_topk_arr`, per mixed layer|
|`forward.rs:9351`|`n_per_batch` staging, per ratio-4 layer|
|`forward.rs:7601,7670`|compressor-fallback H2D, per ratio-4 layer|

Replaying a graph over these would bake in capture-day values forever —
the exact failure DFlash documented for its tree-verify graph
(`speculative.rs:2316-2320`). Everything else checks out: B=5 constant,
PBS persistent (`spec_impl.rs:280-291`), fixed-size KV rings (no
per-window growth), lazy allocs are `is_none`-guarded and warm up outside
capture, head/argmax stay outside.

At +14% on today's 34.0 that projects to **~38.8 tok/s**, inside the
35-40 certification band.

## Status against the 35-40 bar

- **34.0 tok/s on code** (E8 sidecar), vs AR 27.5 — 1.24x.
- Prose is a wash and, per the DFlash precedent, expected to stay one.
- Remaining credible lever: verify graph capture, ~+14%, projecting ~38.8.
- Not levers: deeper blocks (verify is batch-scaled and near roofline),
  confidence threshold (inert), sidecar recipe (acceptance-neutral).
