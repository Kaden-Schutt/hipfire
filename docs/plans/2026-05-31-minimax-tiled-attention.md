<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# Task 2 — Tiled (online-softmax) attention for minimax's native context window

**Status:** IMPLEMENTED (2026-05-31). See "OUTCOME" at the bottom.
**Branch:** `lfm2moe/impl` (PR #365), based on `minimax/m2.7-impl` (`edf922db`).

> **OUTCOME (read first):** No new kernel was written. hipfire already ships a
> production-validated two-stage online-softmax flash path —
> `attention_flash_q8_0` (`attention_flash_q8_0_tile` + `_reduce`) — used as the
> default decode/prefill attention by qwen35 and llama, with the *identical*
> Q8_0 KV layout (34-byte blocks, GQA). Its LDS is `(128+head_dim)*4` ≈ const,
> independent of seq_len. minimax was simply switched onto it via a hybrid
> dispatch (mirroring qwen35). This is strictly lower-risk than a hand-written
> kernel — see the OUTCOME section for the exact edits and validation. The
> original from-scratch design below is kept for context.

## Why

`attention_q8_0_kv` (the GQA Q8-KV decode attention used by minimax, lfm2moe,
qwen35, llama) materializes the full `scores[seq_len]` array in **LDS**:

```
LDS = scores[seq_len] + workspace[nthreads] + q_shared[head_dim]   (floats)
host request = (sizing_seq + block_size + head_dim) * 4 bytes
```

gfx11/gfx12 LDS limit = **64 KB**. So this kernel can only serve
`seq_len ≲ 16K` ((X+256+64)*4 = 65536 → X ≈ 16064). Above that the launch is
rejected with `hipModuleLaunchKernel: invalid argument`.

We already shipped **fix (a)** (commit `1b227db8`): minimax now passes the real
`seq_len` (not `state.max_seq`) so the LDS is sized to the actual length — this
unblocked serve + hermes for prompts under ~16K. But minimax's *native* context
window is much larger, and a real hermes agent (≥64K advertised context, large
preamble) will exceed 16K. **Tiled attention is the proper fix.**

## Current kernel (the thing to replace/augment)

`kernels/src/attention_q8_0_kv.hip` — one workgroup per head (grid=[n_heads],
`nthreads` threads), `seq_len = pos_buf[0]+1`:
- Preload `q_head` → `q_shared[head_dim]` (LDS).
- **Phase 1:** each thread strides `t`, computes `scores[t] = (Q·K[t])*scale`
  (Q8_0 dequant: K block = 34 bytes = fp16 scale + 32 int8; `bi=d/32`,
  `bj=d%32`), tracks local max → reduce → `max_val`.
- Softmax in place: `scores[t]=exp(scores[t]-max_val)`, reduce → `sum_val`,
  `scores[t]/=sum_val`.
- **Phase 2:** each output dim `d` (thread-strided) does
  `out[d] = Σ_t scores[t] * V[t][d]` (V same Q8_0 layout).

Dispatch: `crates/rdna-compute/src/dispatch.rs:~25253` (`attention_q8_0_kv`).
It already has a `capture_mode` branch and sizes `shared_mem` by `sizing_seq`.
Kernel registration: `crates/rdna-compute/src/kernels.rs` (`ATTENTION_Q8_0_KV_SRC`).

## Tiled design (online softmax / flash-attention-decode)

Per head, maintain running scalars + accumulator instead of full `scores[]`:
- `m` = running max, `l` = running denom, `acc[head_dim]` = running output
  (acc + q_shared in LDS; both O(head_dim)).
- Process KV in tiles of `T` positions (e.g. `T = nthreads = 256`). Per tile:
  1. thread `t` computes `s = (Q·K[tile+t])*scale` into `sh_s[t]` (T floats LDS).
  2. reduce tile max → `tile_max`; `new_m = max(m, tile_max)`;
     `corr = expf(m - new_m)`.
  3. thread `t`: `sh_p[t] = expf(sh_s[t] - new_m)`; reduce → `tile_sum`;
     `l = l*corr + tile_sum`.
  4. each dim thread `d`: `acc[d] = acc[d]*corr + Σ_{t<tile_len} sh_p[t]*V[tile+t][d]`.
  5. `m = new_m`.
- Final: `out[d] = acc[d] / l`.

LDS = `q_shared[head_dim] + sh_s/sh_p[T] + acc[head_dim] + workspace[nthreads]`
≈ const (e.g. 64+256+64+256 = 640 floats = 2.5 KB) — **independent of seq_len**.
Preserve: Q8_0 dequant (34-byte blocks), GQA (`kv_h = h/(n_heads/n_kv_heads)`,
`kv_head_block_start`), and the exact V access pattern. New file:
`kernels/src/attention_q8_0_kv_tiled.hip`; register in `kernels.rs`; add
`attention_q8_0_kv_tiled` dispatch.

## Dispatch (hybrid — keep the validated fast path)

In `attention_q8_0_kv` dispatch: if `(sizing_seq + block_size + head_dim)*4 <=
LDS_BUDGET` (≈ 48–60 KB to leave headroom) → current kernel (fast, validated).
Else → tiled kernel. This way small contexts keep the fast LDS-resident path;
only large contexts pay the tiled path. (Alternatively: always tiled if perf is
acceptable — measure first.)

## Validation (do NOT trust the kernel without this)

1. **Local numerical parity on gfx1201** (4× R9700 present locally): extend
   `crates/hipfire-runtime/examples/test_q8kv.rs` — random Q/K/V, compare tiled
   vs current at small seq_len (cosine ≥ 0.9999 / max-abs-err tiny), and tiled
   vs a CPU reference at large seq_len (where current can't run). Build:
   `cargo build --release -p hipfire-runtime --example test_q8kv`.
2. **On-model on hipx (gfx1151):** rebuild daemon, restart serve, run minimax
   coherence + a **>16K-context** request that previously crashed; confirm
   coherent output. Then a hermes agent run with a large preamble.
3. Coherence gate / cosine for minimax must stay green.

## hipx environment (where minimax actually runs — 86 GB, gfx1151)

- Box: Strix Halo gfx1151 (96 GB VRAM carve-out; OS sees ~30 GB). Target the big
  GPU with **`ROCR_VISIBLE_DEVICES=1`** (idx0 = gfx1010 5700XT 8 GB, idx1 = gfx1151).
- Serve worktree: `~/hipfire-minimax-val` checked out on `lfm2moe/impl`. Update
  with `git fetch origin +refs/heads/lfm2moe/impl:refs/remotes/origin/lfm2moe/impl
  && git reset --hard origin/lfm2moe/impl && cargo build --release --example daemon`.
- Start serve: `ROCR_VISIBLE_DEVICES=1 HIPFIRE_JINJA_CHAT=1 bun cli/index.ts serve
  127.0.0.1:11435 -d` (from the worktree). Stop: `bun cli/index.ts stop`.
- minimax model: `~/minimax-tiny-val/MiniMax-M2.7.mq2-lloyd` (86 GB), symlinked as
  `~/.hipfire/models/minimax-m2.mq2lloyd` (the `.mq2-lloyd` dash ext is NOT
  recognized by `findModel`; `.mq2lloyd` is). serve auto-loads max_seq=32768.
- hermes: `~/.local/bin/hermes`, config `~/.hermes/config.yaml` → `custom:hipfire`
  provider, `base_url http://localhost:11435/v1`, **`context_length: 64000`** (its
  floor), `model.default: minimax-m2.mq2lloyd`, streaming off. Headless run:
  `hermes chat -q "..." -t terminal --max-turns 4 </dev/null` (NOT `-Q` — its
  q-to-quit handler KeyboardInterrupts under non-interactive ssh).
- ssh quoting: pipe scripts via `ssh hipx 'bash -l' <<'REMOTE' … REMOTE`; single
  quotes inside `ssh '…'` break the wrapper. `curl -d @~/x` does NOT expand `~`
  (use absolute path).

## Done already this session (baseline — don't redo)

- deepseek #363 (MMQLOAD default-ON attractor) cherry-picked to `minimax/m2.7-impl`
  (`edf922db`); branch rebased.
- minimax LDS fix (a): pass `seq_len` (`1b227db8`).
- jinja Chainable undefined-behavior (`930f47c3`) — minimax native template renders
  under hermes.
- tool-call parsers (LFM2 `<|tool_call_start|>` + MiniMax `<minimax:tool_call>`) +
  tojson(ensure_ascii) fix — minimax/lfm2 tool-calls round-trip via serve.
- Verified: hermes drives BOTH minimax + deepseek e2e (full tool loops) at ≤16K.

## OUTCOME — implemented 2026-05-31 (reuse `attention_flash_q8_0`, no new kernel)

**Approach.** The from-scratch tiled kernel above was unnecessary: the existing
`attention_flash_q8_0` (tile + reduce, `kernels/src/attention_flash_q8_0_tile.hip`
+ `attention_flash_q8_0_reduce.hip`, dispatch `crates/rdna-compute/src/dispatch.rs`
`attention_flash_q8_0`) is exactly an online-softmax flash path on the same Q8_0
KV layout, already the default attention for qwen35/llama on gfx11/gfx12. Switched
minimax onto it.

**Edits (all minimax-local; no kernel/dispatch/shared changes):**
1. `crates/hipfire-arch-minimax/src/minimax.rs`
   - Fixed `flash_partials` sizing: was `tile=256`, `max_seq/256+1` (under-allocated
     ~2×); now `TILE=128`, `ceil(physical_cap/128)` tiles × `n_heads × (2+head_dim)`
     — matches the dispatch exactly.
   - Added `flash_mode: u8` to `MiniMaxState` (env `HIPFIRE_ATTN_FLASH`, default 2
     on gfx11/gfx12 so warmup-eager and captured-replay decode use the SAME kernel).
2. `crates/hipfire-arch-minimax/src/forward.rs`
   - `decode_step_body`: hybrid `use_flash = capture_mode || flash_mode==2 ||
     (flash_mode==1 && seq_len>=2048) || seq_len>15000` → `attention_flash_q8_0`
     else `attention_q8_0_kv`. Fixes graph-capture LDS over-request too (capture
     forced the old kernel to size LDS to physical_cap → >64KB at ≳16K).
   - `forward_batch` (batched prefill): when `max_ctx > 15000`, loop per row calling
     single-position `attention_flash_q8_0` (reusing `state.flash_partials`); else
     keep `attention_q8_0_kv_batched`. Mirrors qwen35:7173.
3. `crates/hipfire-runtime/examples/test_q8kv.rs` — added flash-vs-baseline parity
   (Test 5) + a >16K flash-only sanity check.

**Validation.**
- Local gfx1201 parity (`test_q8kv`, GQA head_dim=128): flash vs baseline
  cosine=1.000000, max-abs-err ≤1.1e-6 at seq=4/64/512/2048; flash-only at
  seq=20000 (baseline can't run) finite + correct (max|out−V|=3.9e-3, Q8 noise).
- hipx gfx1151 on-model (86 GB MiniMax-M2.7.mq2-lloyd, serve, flash_mode=2 →
  every decode step uses flash): coherent reasoning + correct answer
  ("17 × 23 = 391"); no LDS/`invalid argument`/panic.
- >16K context (needle-in-haystack, ~19K-token prompt → prefill-flash >15K
  branch): _[fill result]_.
