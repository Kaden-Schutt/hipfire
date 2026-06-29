# OwnedTensor migration — halo (gfx1151) validation runbook

Companion to `docs/plans/2026-06-29-owned-tensor-raii-scratch.md` (the design +
what changed). This is the GPU-side validation to run on **halo** (Strix Halo,
gfx1151) before the OwnedTensor change is trusted/committed. nix2 (gfx1103) is
unsuitable: the migrated forwards include the full-precision paths that don't fit
its UMA, and it has the documented LDS/VMEM wedge hazard.

The change is **allocation-lifetime only** (kernel order, dtypes, and pooled
allocation order preserved), so the expectation is **numerics byte-unchanged**.
Two things actually need hardware:

1. **Correctness/numerics unchanged** across the migrated forwards.
2. **The `reclaim_pending()` graph-gating behaves** — and its peak-VRAM cost is
   acceptable on a memory-constrained APU.

## Why three "regimes"

`reclaim_pending()` self-gates on `graph_state_live()`
(`crates/rdna-compute/src/dispatch/mod.rs:2281`; live =
`capture_mode || graph_exec || captured_graph || verify_graph_cache || replay_graph_cache`).
That yields three behaviors the validation must each hit:

- **R1 — non-captured prefill**: reclaim actually returns scratch to the pool.
- **R2 — captured decode**: capture is **default-on for qwen35 on gfx11/gfx12**
  (`qwen35.rs:9312`, so it auto-engages on halo); reclaim is a no-op during/after
  capture, scratch is held in the mailbox.
- **R3 — replay-graph spec-decode**: a cached verify/replay graph keeps reclaim
  gated across steps — the "scratch held longer" window. This is the one to watch
  for peak-VRAM growth.

`HIPFIRE_GRAPH=0` forces direct decode (turns R2→R1) for A/B comparison;
`HIPFIRE_GRAPH=1` forces capture on.

## 0. Prerequisites & setup (on halo)

```bash
cd <hipfire repo on halo>
export HIP_VISIBLE_DEVICES=1          # gfx1151 is HIP device 1 on halo
git rev-parse --abbrev-ref HEAD       # confirm the OwnedTensor branch is checked out

# Build the CLI (provides `hipfire lock` + `hipfire detect` + `hipfire eval`/`chat`):
cargo build --release -p hipfire-cli --bin hipfire

# Confirm which model artifacts are actually present (commands below use PLACEHOLDER
# names from repo doc-headers/gates — substitute real ones):
ls ~/.hipfire/models ~/.hipfire/drafts 2>/dev/null   # or: ./target/release/hipfire list
```

### GPU lock protocol (mandatory for non-daemon GPU work)

Per repo invariant, examples / `hipfire eval` / `hipfire-quantize` do **not**
self-lock — wrap them. The two coherence gates *do* self-lock. Lockfile:
`~/.hipfire/locks/hip-gpu-0.lock` (flock(2), same inode the daemon uses).

```bash
HIPFIRE_BIN=./target/release/hipfire
"$HIPFIRE_BIN" lock status                                   # 'gpu is free' | 'gpu BUSY: <holder>'
"$HIPFIRE_BIN" gpu-lock acquire "owned-tensor-validate" --watch-pid "$$" \
  || { echo 'could not acquire GPU lock' >&2; exit 2; }
trap '"$HIPFIRE_BIN" gpu-lock release 2>/dev/null || true' EXIT
# ... GPU work here; lock auto-releases when this shell exits ...
```

## 1. No-GPU gate (no lock, run first)

```bash
./tests/no-gpu-ci.sh    # cargo check --workspace --examples -D warnings, lib tests, lints, doc-freshness
```
Already green in-session; re-run on halo's toolchain as a baseline.

## 2. Canonical correctness gates (self-locking)

### qwen35 spec-decode (R3) — the primary correctness gate
```bash
HIP_VISIBLE_DEVICES=1 HIPFIRE_DIR=$HOME/.hipfire ./tests/coherence-gate-dflash.sh          # short: 4 cases
HIP_VISIBLE_DEVICES=1 HIPFIRE_DIR=$HOME/.hipfire ./tests/coherence-gate-dflash.sh --fast    # 2 cases, ~1 min
HIP_VISIBLE_DEVICES=1 HIPFIRE_DIR=$HOME/.hipfire ./tests/coherence-gate-dflash.sh --full     # + ddtree b22-k4, b8-k2
```
- Self-acquires the GPU lock; auto-rebuilds `dflash_spec_demo` (`--features deltanet`).
- Needs a 27B target+draft pair under `~/.hipfire/models` + `~/.hipfire/drafts`
  (tries `qwen3.6-27b-mq4`, `qwen3.5-27b-mq4`, then `-mq3`). **Missing pair → SKIPS
  with exit 0** (not a failure) — so confirm a pair exists or this proves nothing.
- Pass = exit 0; fail (exit 1) = panic / zero tokens / token-attractor
  (`max_token_freq/total > 0.40` or `unique/total < 0.30`) via `hipfire detect`.
- Stricter AR↔DFlash token parity + rollback/state checks:
  prefix with `HIPFIRE_DFLASH_AR_PARITY=1`. **Run this** — it's the strongest
  "spec path still matches AR" evidence for a lifetime-only change.

### deepseek4 MTP spec-decode — BLOCKED on chaingun ⚠
`./tests/coherence-gate-deepseek4-mtp.sh` drives the daemon, but on `chaingun`
`hipfire-arch-deepseek4` is **commented out** of `crates/hipfire-runtime/Cargo.toml:118`
("needs swa_topk_wmma methods"). The deepseek4 crate compiles standalone (the `snap`
migration is verified by `cargo check -p hipfire-arch-deepseek4`) but is **not wired
into the daemon on this branch**, so its forward cannot be exercised on halo until
that dep is re-enabled. Same applies to minimax (line 117) and dots-ocr (line 119) —
those were not migrated, so no action, but note they're disabled.

## 3. Exercise each migrated forward across the regimes

Wrap every example below in the lock (§0). Substitute real `~/.hipfire/models/*.hfq`
names (placeholders are from repo doc-headers/gates — confirm with `hipfire list`).

### R1 — non-captured prefill (reclaim frees)
```bash
# gemma3 text prefill (arch 12) → forward_prefill_batch
cargo run --release -p hipfire-arch-gemma3 --example infer_gemma3 -- \
  --hfq ~/.hipfire/models/<gemma3-text>.hfq --prompt 'The Roman Empire, at its height,' --max-new-tokens 48
# llama/qwen2 dense prefill (arch 0/1/7) → llama prefill_forward / forward_prefill_batch
cargo run --release -p hipfire-runtime --example smoke_llama_prefill_batch -- ~/.hipfire/models/<llama>.hfq
cargo run --release -p hipfire-arch-qwen2 --example infer_qwen2 -- --hfq ~/.hipfire/models/<qwen2>.hfq --prompt-file benchmarks/prompts/qwen2_smoke.txt
# zaya prefill vs golden (arch 16) → gpu_forward_prefill (bit-exact cosine check)
cargo run --release -p hipfire-arch-zaya --example gpu_golden -- <zaya.bf16.hfq> <golden/raw_fp32 dir>
# qwen35 forced-direct decode (R2→R1 A/B): reclaim actually frees here
HIPFIRE_GRAPH=0 cargo run --release -p hipfire-runtime --example infer_qwen35 -- ~/.hipfire/models/<qwen35>.hfq 'One sentence about rivers.'
```

### R2 — captured decode (reclaim no-op; capture auto-on on gfx11)
```bash
# qwen35 dense decode (capture engages automatically on halo) → forward_scratch* path
cargo run --release -p hipfire-runtime --example infer_qwen35 -- ~/.hipfire/models/<qwen35>.hfq 'A short paragraph about the Roman Empire.'
# qwen35 MoE grouped paths → grouped_moe_*_final_logits / rq_apply_*
cargo run --release -p hipfire-runtime --example infer_qwen35 -- ~/.hipfire/models/qwen3.5-35b-a3b-mq4.hfq 'Explain speculative decoding in two sentences.'
# full prefill+decode through the daemon (also drives llama forward/forward_logits_gpu)
./target/release/hipfire chat --model ~/.hipfire/models/<model>.hfq --max-tokens 128 'Describe the fall of the Roman Empire.'
# zaya prefill + O(1) decode → gpu_forward_serve
cargo run --release -p hipfire-arch-zaya --example generate -- <zaya.hfq> 32
```

### R3 — replay-graph spec-decode (reclaim gated; watch peak VRAM)
```bash
# Canonical (see §2). Manual single run (wrap in lock yourself):
HIP_VISIBLE_DEVICES=1 ./target/release/examples/dflash_spec_demo \
  --target ~/.hipfire/models/qwen3.6-27b-mq4.hfq --draft ~/.hipfire/drafts/qwen3.6-27b-mq4.dflash.hfq \
  --prompt 'The Roman Empire, at its height, stretched from' --max 192
```

### VL vision forwards
```bash
# gemma3-vl vision_forward (arch 13) — dedicated example (most reliable)
cargo run --release -p hipfire-arch-gemma3-vl --example infer_gemma3_vl -- \
  --hfq ~/.hipfire/models/<gemma3-vl>.hfq --image benchmarks/vision/images/mri_human_brain.jpg \
  --prompt 'Describe this brain MRI.' --max-new-tokens 64
# qwen35-vl vision_forward — only via daemon/chat --attach
./target/release/hipfire chat --model ~/.hipfire/models/<qwen35-vl>.hfq \
  --attach benchmarks/vision/images/scene_1.jpg --max-tokens 64 'Describe this image.'
```

**What to check (R1/R2/R3 + VL):** clean exit, coherent output, no panic/page-fault,
and (zaya `gpu_golden`) the per-block cosine PASS (≥0.999). Crashes / garbage =
a migration regression (most likely a wrongly-converted view or an escaping tensor
reclaimed too early).

## 4. Numerics-unchanged evidence (KLD / battery compare)

Strongest signal for a lifetime-only change is bit/KLD parity vs the *pre-migration*
build. Pin the lock, then:
```bash
HIP_VISIBLE_DEVICES=1 ./target/release/hipfire eval <model>.hfq \
  --compare <model>.hfq --battery coherence,quality,speed
# higher-precision reference comparison (KLD wired via eval `kldref`):
HIP_VISIBLE_DEVICES=1 ./target/release/hipfire eval <candidate>.hfq --reference <bf16-or-q8>.hfq --battery coherence,perplexity
```
Cleanest A/B: build the pre-change commit into a second checkout, run the same
prompt set through both, and `diff` the emitted token ids / logits. `gpu_golden`
(zaya) already gives an absolute fp32-golden cosine without a second build.

## 5. Peak-VRAM under a cached replay graph (the gating cost)

The concern: in R3, dropped scratch stays in the mailbox until graphs invalidate,
so per-step transient scratch is held instead of reused → higher peak. Validate it's
**bounded and acceptable on the 8 GB-class budget**, not unbounded growth.

**There is no built-in high-water metric** (pool `total_allocated` only ever
increments; the mailbox has no depth getter; no `HIPFIRE_*` pool-debug knob). So
measure indirectly:

- `dflash_spec_demo` already prints `VRAM @ <label>: used/free GB` (init) and
  `vram_used_mb` BENCH METRICS per row (via `get_vram_info`/`hipMemGetInfo`):
  ```bash
  HIPFIRE_PROFILE=1 ./target/release/examples/dflash_spec_demo --target <t>.hfq --draft <d>.hfq \
    --prompt 'Write a haiku about GPUs' --max 128 --ctx 512 2>&1 | grep -E 'VRAM @|vram_used_mb'
  ```
- Compare the R3 `vram_used_mb` against an R1/R2 run of the same model; a large,
  step-monotonic delta in R3 is the held-scratch cost.

> ⚠ **APU measurement caveat (must confirm on halo first).** On gfx1151,
> `hipMemGetInfo` / `rocm-smi --showmeminfo vram` / sysfs `mem_info_vram_total`
> report only the small dedicated **carveout**, not the GTT pool the runtime
> actually allocates from (`crates/hipfire-server/src/telemetry.rs:3-6`). So the
> printed "VRAM used" may **not move** when scratch is held in GTT. The node that
> would actually show GTT occupancy is `mem_info_gtt_used`, which the repo reads
> **nowhere** (only `mem_info_gtt_total`). Before trusting §5, check on halo whether
> these move across model load:
> ```bash
> cat /sys/class/drm/card*/device/mem_info_gtt_used /sys/class/drm/card*/device/mem_info_gtt_total
> ```
> If `mem_info_gtt_used` exists and tracks occupancy, sample it in a `while` loop
> during an R3 run to catch the held-scratch high-water. If neither VRAM nor GTT
> readouts move, the held-scratch concern is only provable by code reading
> (the gate at `mod.rs:2281`) and the practical risk is low.

**If peak proves too high** on a constrained config: drain at graph-invalidation
points, or keep manual frees on the specific R3 (spec-decode) transients — those
sites are noted in the design doc and were the lowest-value conversions anyway.

## 6. Pass criteria

- §1 no-GPU gate green; §2 `coherence-gate-dflash.sh` exit 0 (with a real 27B pair
  present, ideally `HIPFIRE_DFLASH_AR_PARITY=1`).
- §3: every available arch runs clean across its regimes; zaya `gpu_golden` cosine PASS.
- §4: token-id / KLD parity vs the pre-migration build (lifetime-only ⇒ expect exact).
- §5: peak VRAM in R3 is bounded (constant-ish, not growing with generation length).

## 7. Open items to resolve on halo

- **Model filenames are unverified** — every `<...>.hfq` is a placeholder from
  repo doc-headers/gates. Confirm with `hipfire list` / `ls ~/.hipfire/models` and
  substitute. The coherence gates SKIP gracefully when their models are absent.
- **deepseek4 / minimax / dots-ocr disabled on chaingun** (runtime Cargo.toml
  117-119) — deepseek4's migration is crate-compile-verified only; its forward
  can't run via the daemon until the dep is re-enabled (needs `swa_topk_wmma`).
- **zaya example default paths** point at dev-box absolute paths
  (`/home/sadara/zaya1-8b-native.*`); the bf16 model + fp32 golden dump must be
  staged on halo for `gpu_golden`. `gpu_forward_calib` (zaya) is the calibration
  path, reached via `hipfire collect-artifacts` (exact invocation unconfirmed).
- **APU VRAM observability** (§5 caveat) — confirm which sysfs/HIP node actually
  reflects GTT occupancy before relying on the VRAM arm.
- **Follow-up (not part of this validation):** deepseek4 `attention_block_*`
  `debug_max`/`debug_sumexp` pre-existing dev-gated (`HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT`)
  pool leak.

## References
- Design + what changed: `docs/plans/2026-06-29-owned-tensor-raii-scratch.md`
- Steering rule: `crates/rdna-compute/AGENTS.md` (GPU Scratch Lifetime)
- Gate sources: `tests/no-gpu-ci.sh`, `tests/coherence-gate-dflash.sh`,
  `tests/coherence-gate-deepseek4-mtp.sh`
- Lock: `hipfire lock {acquire,release,status}` (`crates/hipfire-lock`,
  `crates/hipfire-cli/src/commands/lock.rs`)
