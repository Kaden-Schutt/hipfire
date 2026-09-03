<!-- SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Kaden Schutt; hipfire — see LICENSE and NOTICE in the project root. -->

# DFlash speculative-decode audit — 2026-09-03

_Lifecycle: planned intent. Read-only audit of master `8cd15a62b`. Findings cite `path:line` at that commit._

**Scope.** Five read-only slices: draft runtime + VRAM (`hipfire-runtime/src/dflash.rs`, `admission.rs`, the three `carrier.rs` scratch-sizing commits), the generate loop (`hipfire-generate/src/qwen.rs` `generate_dflash`/`generate_spec` + helpers), verify/replay (`hipfire-arch-qwen35/src/dflash_spec.rs`, `dflash_verify_pm4.rs`, `speculative.rs`), the `attention_dflash*` kernel family + dispatch, and pairing/discovery/config (`dflash_generic.rs`, `dflash_convert.rs`, daemon/CLI/config). The headline finding (#1) was re-derived by the auditor from `crates/hipfire-cli/src/main.rs:1944-1965,2488+` and `crates/hipfire-daemon/src/main.rs:1194-1216`.

## Verdict

The speculative machinery itself is in good shape: the verify block is a batched target prefill with a DeltaNet snapshot + innovation tape replayed for `accept_len+1` (not per-row checkpoints), graph capture goes through owned kernarg blobs (`launch_maybe_blob`), retained PM4 is a pure phase machine at fixed B=16, the draft's attention is F32 end to end and never touches the target's quantised KV, masks and online-softmax are correct in the live kernels, and the client-cancel path runs the canonical rollback. The problems are around it: **reachability, accounting, and a handful of exits that skip rollback.**

## Broken

1. **DFlash is unreachable without an explicit draft path — the documented on-ramp does not exist.** high, verified.
   `AGENTS.md` ("pull the draft, `hipfire config set dflash_mode auto`, run; expect `[hipfire] DFlash draft detected`") and `docs/MODELS.md:116` describe sibling-filename auto-discovery (`qwen3{ver}-{size}-dflash-{quant}.hfq`). No such matcher exists in daemon, CLI, loader, or registry. The only draft sources are `--model-draft` (`cli/main.rs:1960-1965`), `developer.dflash_draft` / legacy `HIPFIRE_DFLASH_DRAFT` (`cli/main.rs:2579-2604`, daemon `main.rs:1194-1216`), and `params.draft`. `load_params` (`cli/main.rs:2488+`) never consults the registry entry, and `hipfire pull <tag>-draft` sets nothing. Consequence: `dflash_mode auto` (and `on`) with a pulled draft runs plain AR, silently — `on` behaves identically to `auto` on a missing/failed draft (daemon strips the path only for `off`; Qwen35 load `Err` → `eprintln` + AR, `loader/lib.rs:1960-1972`). The registry already carries the pairing (`qwen3.8:27b-draft` → `qwen3.8:27b-draft-mq4` → `qwen38-27b-dflash-mq4.hfq`), so a registry-driven resolution in `load_params` is the whole fix; `on` should then fail the load when no draft resolves, mirroring MTP (`lib.rs:1973-1976`). Measured stakes on a 7900 XTX: 27B mq4 AR is 46.8 tok/s; every "150 tok/s" number users quote is a speculator.
2. **Draft VRAM is charged nowhere.** high. `ModelFootprint` has two fields (`admission.rs:17-22`: target weights, KV bytes/token); nothing adds draft weights (0.92–1.66 GiB by quant, `docs/MODELS.md:106-113`) or the L-indexed draft planes. At the default `HIPFIRE_DFLASH_CTX_CAP=8192` on 27B (h=5120, ne=5, kvd≈1024, nL=5, B=16): `target_hidden` L·ne·h·4 ≈ 800 MiB, `target_hidden_proj` ≈ 160 MiB, per-layer K/V ctx caches ≈ 320 MiB, `k/v_cat` ≈ 160 MiB, `mq_x_rot` ≈ 100 MiB (chunked; was 800 MiB before the chunk fix) — **≈1.5–1.8 GiB scratch + weights ≈ 2.5–2.7 GiB** invisible to admission. Uncapped at 32k the caller's own comment puts it at ~11 GB (`dflash_spec.rs:117-122`). Windowed DFlash2 drafts (all-sliding, W=2048) are much smaller. Separately: the daemon never constructs `AdmissionController` at all — its only users are the slots path (`session_table.rs`) and tests — so `noslots` serving has no budget gate; the user gets a raw allocation failure after a 19 GB load.
3. **Partial-failure leaks in the draft constructors.** high. `DflashScratch::new_with_mq` (`dflash.rs:1403-1483`) and `new_windowed` (`1343-1356`) `?` out mid-way without freeing earlier `alloc_tensor`s; `DflashWeights::load` (`637-946`) likewise; `dflash_generic::build_generic_dflash_speculator` loads weights then `?` on scratch without freeing (`dflash_generic.rs:1048-1055`). `GpuTensor`/`DeviceBuffer` have no `Drop` (`rdna-compute/src/dispatch.rs:205-211`). The Qwen35 outer chain is transactional (`or_free!` in `load_dflash_state`), the ctors it calls are not.
4. **`make_spec_emitter` Err after a successful prefill has no rollback.** high. `qwen.rs:3113-3125` emits a validation error and returns `None`; the target's KV/DeltaNet/drafter hidden already advanced and, on `!cache_hit`, host `seq_pos`/`conversation_tokens` were cleared at `2929-2936`. Next turn can LCP against a dirty GPU. Every other error exit in the loop (`prefill` Err, `step` Err, realign, forced terminal, pending-seed flush, mid-loop abort) runs `production_fail_closed_rollback_live` — this one is the odd one out.
5. **Entry cap vs loop cap disagree.** high. `generate_dflash` falls through to AR only when `prompt + max_tokens > ctx_capacity` (`2078-2090`); `generate_spec` then hard-errors when `prompt + max_tokens + block_size > ctx` (`2980-3000`) — after `gen_start` was emitted. Requests in that `block_size` band get a started generation followed by an error instead of the promised AR fallback. Related: the mid-loop `position + block_size >= ctx_capacity` `break` (`3358-3360`) sets no flag, so if `generated < max_tokens` the epilogue reports `finish_reason=stop` and may store the cache — a silent early stop (med).
6. **Draft pairing checks nothing about the target.** high. Convert records draft geometry only (`dflash_convert.rs:1031-1088`: block size, mask token, `target_layer_ids`, `num_target_layers`, dims; no target family/size/quant/hash). Qwen35 load checks only `target_layer_ids[i] < n_layers` (`dflash_spec.rs:333-344`) — not even `draft.hidden == target.dim` (llama's generic path does, `carriers.rs:944-951`); `num_target_layers` is written and never compared. The known 3.5-draft-on-3.6-target τ≈1.2 failure is this. Also `vocab_size` is written as `config.get("vocab_size").cloned()` (`dflash_convert.rs:1040`) and can be JSON `null`, which `DflashConfig::from_hfq` (`dflash.rs:134`) then refuses — the writer can emit a draft the loader cannot read.

## Missing

- Registry-driven draft resolution at load (fixes 1); `dflash_mode=on` failing closed on a missing/failed draft.
- `ModelFootprint` (or a side charge at load) covering draft weights + `scratch(L)`; a "ceiling at this config is N tokens" line before the load; an admission test that `admit(64k)` with 27B + draft on 24 GiB fails closed (fixes 2).
- Transactional ctors (mirror `or_free!`) in `dflash.rs`/`dflash_generic.rs` (fixes 3).
- One shared predicate for the entry cap and the loop cap; a `ctx_exhausted` stop reason for the mid-loop break that suppresses cache store (fixes 5).
- Target identity in the draft header (arch, n_layers, dim, vocab, optional weight hash) + a loader refusal; `num_target_layers == target.n_layers` and `draft.hidden == target.dim` on the Qwen35 path; non-null `vocab_size` in convert; a schema version for the dflash metadata beyond `HFQ_VERSION=1` (fixes 6).
- `GenericDflashSpeculator` inherits no-op `rewind_to`/`on_evict` (`spec.rs:727-729, 807-818`; `dflash_generic.rs:951-968`) — prompt-cache resume / CASK on the llama path can desync `target_hidden_host` from the target KV. Either implement or refuse those combinations.
- Kernels: `attention_dflash.hip:254-259` finalizes with `1/l_run` and no `l_run == 0` guard (the sliding sibling guards, `:285-297`); the `n128_f16kv` entry is registered under `HasWmma` (includes gfx12) while its source is gfx11-only (`attention_table.rs:370-385`) — shadowed today by the higher-priority `DflashV5Gfx12`, a JIT failure if that ordering changes; the production `attention_dflash_wmma*` launches pass stack `void**` kernargs (`attention.rs:8588-8616, 9306-9336, 9440-9454`) — safe only because the draft captures FFN, never attention.

## Would-change

- `dflash_adaptive_b` is parsed by the daemon and dropped (`let _adaptive_b`, `daemon/main.rs`); the config key (`speculation.dflash_adaptive_b`, default true) is inert. Adaptive B lives only in `examples/dflash_spec_demo.rs`. Either wire or delete the key.
- AR fallback past the ctx cap leaves the full draft allocation resident (`qwen.rs:2078-2090`). Reasonable for speed; wrong on a 24 GB card after a load that only just fit.
- Windowed split drafts still pin one full-reach K/V layer at `w_full = requested_ctx` (`dflash_spec.rs:294-304`, `dflash.rs:1349-1352`) — the same "logical max_seq" class the carrier fix retired; consider `physical_cap`.
- `attention_dflash_wmma_m32_kstg_FAILED.hip` is unwired but exports the same C symbol as the live `m32` kernel; an accidental `include_str!` would shadow it.
- The "identical output" claim on AR fallback (`qwen.rs:2083`) is true for the main Qwen route (same request seed); the `Qwen2Spec` fallthrough uses a bare `generate_qwen2` without tools/seed parity (`ar.rs:1348-1372`).
- Docs: `AGENTS.md` §1/§3.6/§7 and `docs/MODELS.md:116` describe the non-existent auto-match; `docs/env-vars.md` should list `developer.dflash_draft` as the only production knob until (1) lands.

## Confirmed by design

- Verify forward: DeltaNet advances in place, then restores the pre-window snapshot and replays the `GdnTape` innovations for `accept_len+1` (`speculative.rs`); no per-row S checkpoint is needed.
- Graph capture: verify graphs are cached per B via `capture_mode → launch_maybe_blob` with owned blobs (`dispatch.rs`, `graph.rs`); the draft captures only its FFN tail with owned blobs (`dflash.rs:1969-1994`). Retained PM4 pins B=16, kv_mode/dn_state q8, `!tree`, `!full_logits` (`dflash_verify_pm4.rs`).
- Client cancel after handshake runs `production_fail_closed_rollback` (`qwen.rs:2557-2561, 2724-2727`) — not a bare `Abort => {}`.
- The three scratch-sizing commits end with the target's flash partials sized from `kv.physical_cap` (`carrier.rs:85-88`); `c305a34b6` was a bad intermediate that `a22b88d3f` restored. No sibling in `dflash.rs` is sized the old way.

## Not read

Full `draft_forward` attention body (~`dflash.rs:2500-3100`), the MTP speculator's commit path, the vision DFlash loop, research-only attention variants (`n64`, `v2/v3-noncausal/v4/v6/v7/v7b`) beyond status, `dflash_spec_demo.rs`, and the numerical tests under `rdna-compute/examples/test_attention_dflash.rs`. Nothing was run.

## Recommendation, in order

1. Registry-driven draft resolution + `on` fails closed + docs (hours; the largest measured user-facing win available: AR → speculative for every default user).
2. Charge the draft in the footprint and print the context ceiling at load (hours); decide whether `noslots` serving should construct `AdmissionController` at all (design).
3. Transactional draft ctors; `make_spec_emitter` rollback; unify the two ctx-cap predicates and add the `ctx_exhausted` reason (hours each, one PR).
4. Target identity in the draft header + loader refusal; non-null `vocab_size` (a day; needs a convert re-run for existing drafts or a permissive read of old headers).
5. Kernel nits (zero-`l` guard, `n128` predicate, blob-safe attention launches) in one small PR.
