# Validation routes

Sole human **route selector** for hipfire validation evidence.
Executable behavior lives in the scripts and workflows named below — this
file only maps claim class → route. Methodology numbers and Redline
certification prose live in their owners ([`INDEX.md`](INDEX.md)).

| Field | Value |
|---|---|
| Inventory date | 2026-07-22 |
| Audited source ref | `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Comparison base | `origin/beta` @ `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Integrated commit / tree hashes | External Git/CI only. |

## Rules

1. Pick the **narrowest** route that covers the changed surface.
2. Every route below names an **existing** path, or is marked **blocked**.
3. **Fail closed** on unknown claim classes: do not improvise a gate, and
   do not treat a green unrelated route as coverage.
4. There is **no universal replacement gate**.
5. Harness success without the proof the claim needs is insufficient
   (especially Redline route proof — see [`REDLINE.md`](REDLINE.md)).
6. **Admissions** are recorded only in [`admissions.yml`](admissions.yml).
   A passing route does not create an admission row.

## Automatic checks vs manual evidence

| Class | When it runs | Authority |
|---|---|---|
| **Automatic (no GPU CI)** | PR / push via the no-GPU workflow only | Merge bar for compile, native control-plane/unit tests, CPU tests, and env/docs reference coverage. **Not** model coherence, serve semantics, or perf admission. |
| **Automatic (path-gated hooks)** | Local `pre-commit` on matching staged runtime paths | Runs the hotspot guards selected by the staged path set. Documentation-only staged sets do not trigger a separate docs hook. **Not** a full product matrix. |
| **Manual local no-GPU equivalent** | Human/agent invokes `scripts/no-gpu-ci.sh` outside CI | Same checks as the workflow script body; still **manual invocation**, not automatic CI. |
| **Manual (GPU / model)** | Human or agent on hardware with an explicit model path | Required for kernel, dispatch, forward, quant, serve-behavior, Redline, and perf claims. |

Automatic CI green never substitutes for a required manual route. Running the no-GPU script locally is convenient parity with CI, not an automatic check.

### Automatic entrypoints

| Route | Path | Role |
|---|---|---|
| No-GPU CI workflow | [`.github/workflows/no-gpu-ci.yml`](../.github/workflows/no-gpu-ci.yml) | **Automatic** CI entry that invokes the no-GPU script on PR/push. |
| Pre-commit hooks | [`.githooks/pre-commit`](../.githooks/pre-commit) | **Automatic** when hooks are installed (`scripts/install-hooks.sh`). Selects HOTSPOT / SERVE_HOTSPOT / PP_HOTSPOT runtime guards from staged paths; documentation-only staged sets exit without a separate docs gate. |
| Dispatch `bind_thread` invariant | [`scripts/verify-bind-thread.sh`](../scripts/verify-bind-thread.sh) (via pre-commit on matching paths) | **Automatic** when hooked: every public `dispatch.rs` entry must bind the HIP thread. Not a kernel numeric test. |
| Env/docs drift check | [`scripts/check-env-docs.py`](../scripts/check-env-docs.py) | **Automatic** through `scripts/no-gpu-ci.sh`; checks that referenced `HIPFIRE_*` names are documented and production reads are config-owned. |

### Manual local equivalents (not automatic)

| Route | Path | Role |
|---|---|---|
| No-GPU script | [`scripts/no-gpu-ci.sh`](../scripts/no-gpu-ci.sh) | Manual local run of the same body CI uses: `cargo check --workspace --examples`; selected no-GPU Rust and native control-plane tests; CPU pytest; env/docs coverage. |
| Env/docs drift check | [`scripts/check-env-docs.py`](../scripts/check-env-docs.py) | Fast standalone env-name coverage and ownership check. No GPU or model download. |

### Documentation checks

The shipped automated documentation-specific check is
[`scripts/check-env-docs.py`](../scripts/check-env-docs.py). It validates
environment-variable reference coverage and config ownership; it is not a
general Markdown link, lifecycle-label, or semantic-drift checker. Beta does
not currently ship a standalone `check-docs-reliability.py` command or a
documentation-only pre-commit gate. For documentation-only changes, run the
env/docs check plus `git diff --check`, and inspect changed links and commands
against the current tree.

### Maintained manual harness roles

Narrow roles. Do not widen a harness into a universal gate.

| Harness | Path | Role | Not this harness |
|---|---|---|---|
| **gates.sh** | [`scripts/gates.sh`](../scripts/gates.sh) | Maintained **manual** wrapper: optional Redline capture, generic serve battery, optional fresh-process perf compare (`probe_commits.sh`), and (opt-in, `--escha`) the Escha-W2 G1-G6 correctness battery. Requires `--model`. | Not CI-default. Not universal. Does not call retired coherence-gate scripts. The `--escha` arm is checkpoint-specific and is **off by default** — it is not a gate for any other model. |
| **serve_harness.py** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | **Model-agnostic** user-facing serve behavior (battery / chain / session): finish reasons, runaway/empty, prefix cache, prefill/decode timing, recall hooks. | Not LFM thinking-frame specifics. Not Redline route proof. |
| **serve_harness.py (LFM tag)** | [`scripts/serve_harness.py`](../scripts/serve_harness.py) | LFM2.5 serve smoke with the exact registry tag; use registry sampling or `recipe:nothink` for non-thinking framing. | Not a substitute for numerical parity oracles. |
| **redline_daemon_harness.py** | [`scripts/redline_daemon_harness.py`](../scripts/redline_daemon_harness.py) | Resident-daemon **Redline** capture, phase fingerprint, shadow/parity, and timing evidence under manual-capture env. | Discovery/correctness evidence ≠ product timed-arm route proof by itself. Does not enable AQL routing. |
| **dispatch_profile** | [`tools/redline/dispatch_profile.py`](../tools/redline/dispatch_profile.py) | Manual **attribution-only** steady-state retained-PM4 per-dispatch span diagnostic (instrumented GFX12 tape). | **Not** route proof, certification/admission, or absolute/pure kernel timing: timestamp commands add observer overhead and spans include preceding PM4 boundary packets. |
| **tools.redline golden** | [`tools/redline/golden.py`](../tools/redline/golden.py) | Reproduce the exact checked-in MQ4R TG128 model/benchmark/route fixtures through the route-proof-capable product harness. | Not a universal GPU gate, a new-route certification shortcut, or an admission. |

### Supporting manual tools (existing; claim-scoped)

Use only when the claim class below names them. They are not universal.

| Tool | Path | Typical claim |
|---|---|---|
| Kernel channel tests | `target/release/examples/test_kernels` (build: `cargo build --release --features deltanet --example test_kernels -p hipfire-runtime`) | Kernel numeric vs CPU reference on the detected arch. **Not** dispatch bind coverage. |
| Dispatch bind check (manual) | [`scripts/verify-bind-thread.sh`](../scripts/verify-bind-thread.sh) | Same bind invariant as the hook; run explicitly if hooks are not installed. |
| Speed regression floor | [`scripts/speed-gate.sh`](../scripts/speed-gate.sh) | Prefill/decode vs committed `tests/speed-baselines/<arch>.txt` when that path’s policy applies. |
| Fresh-process commit probe | [`scripts/probe_commits.sh`](../scripts/probe_commits.sh) | Optional A/B invoked from `gates.sh --perf`. |
| Perf protocol | [`docs/methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) | How to measure; not an executable gate. |
| Redline certification ladder | [`docs/REDLINE.md`](REDLINE.md) | What evidence is required before Redline-attributed promotion. |
| Path-specific parity / state oracle | Arch-owned example or test named by the change (when one exists) | Hidden-state, logit, KV/conv, or graph parity. If none exists for the surface → **blocked**. |

### Escha-W2 correctness gates (G1-G6)

Checkpoint-specific, **manual**, GPU + model. Run them with
`scripts/gates.sh --escha-only --model /path/to/escha-35b.hfq`, or
individually. These are the required route for any change to the escha codec,
the H128 transforms, the escha routed executors, or the escha loader — a green
serve battery is **not** evidence for any of them.

Measured values live in
[`escha-w2-port-design.md`](plans/escha-w2-port-design.md) §10.6. A gate that
passes with a number materially different from the one recorded there has not
passed; re-derive before re-recording.

| Gate | Command | Asserts | Recorded result |
|---|---|---|---|
| **G1** | `python3 scripts/escha-verify-roundtrip.py <src-dir> <model.hfq>` | Verbatim repack: every `escha_code` tensor byte-identical to the source safetensors. Count asserted against `model.safetensors.index.json`; fails on zero shards, zero code tensors, or a count mismatch. | 80/80 byte-identical |
| **G2** | `cargo run --release -p rdna-compute --example test_escha_decode_gpu_vs_cpu` | GPU tile decode == `escha_ref::reconstruct`, bit-exact in fp16, at golden and 89M-element shapes. | 0 mismatched |
| **G3** | `cargo run --release -p rdna-compute --example test_escha_h128_gpu_vs_cpu` | The H128 pair == `escha_ref`, bit-exact, every launch form (single, batched broadcast / per-slot / grouped, out_batched, swiglu). | 0 mismatched |
| **G4b** | `cargo run --release -p hipfire-arch-qwen35 --example escha_router_contract -- <model.hfq>` | arch-6 router selects the same experts as escha's reference routing. | 0/8 differing sets |
| **G4** | `cargo run --release -p hipfire-arch-qwen35 --example escha_moe_block_gate -- <model.hfq>` | Whole MoE block vs escha's `moeblk_out.f16`; plus indexed-vs-host and batched-vs-per-token **equality**. | F32 max 1.828e-4 / mean 9.673e-6; Q8_0 max 2.633e-4 / mean 3.027e-5; 0 differing floats on both route comparisons |
| **G5** | `scripts/escha-kld.sh <model.hfq>` | KLD vs the weight-exact escha arm on a fixed teacher-forced corpus, with an asserted negative control and an asserted upper bound. | 0.0027576 nats (CI 0.0019491-0.0038610), PPL 7.6585, control prints 0.000000 |
| **G6** | `cargo run --release -p hipfire-arch-qwen35 --example escha_prefill_batch_gate -- <model.hfq>` | Batched prefill vs the per-token route, whole model: argmax stable, logit deltas within measured bounds, no non-finite logits. | argmax stable; max\|delta\| 4.393e-1, mean\|delta\| 7.160e-2 |

`escha_ref` (`crates/hipfire-quantize/src/escha_ref.rs`) is the **frozen
oracle** every bit-exactness claim above rests on. It is a transcription of
EschaLabs' `ref.py` and must not be edited to make a gate pass; a gate that
disagrees with it is reporting a defect in hipfire.

G1-G4b need only the checkpoint and the fixtures committed under
`crates/hipfire-quantize/tests/data/escha/`. G5 and G6 load the whole model
(37.6 GB resident) and take minutes each.

## Claim → route map

| Claim / change class | Minimum route(s) | Evidence kind |
|---|---|---|
| Docs, env-var tables, no-GPU Rust/Python/CLI only | Automatic: `.github/workflows/no-gpu-ci.yml` invokes `scripts/no-gpu-ci.sh`; for a fast docs-only local check run `python3 scripts/check-env-docs.py` plus `git diff --check` | Automatic CI and/or manual local equivalent — **not** GPU/model evidence |
| New/changed `.hip` kernel (numeric) | `test_kernels`; then model-level manual route for the arch under test | Manual + channel |
| Dispatch `bind_thread` / public `dispatch.rs` bind surface | `.githooks/pre-commit` → `scripts/verify-bind-thread.sh` (or run that script manually) | Automatic hook or manual bind check — **not** `test_kernels` |
| Forward / fusion / KV **numerical or state parity** | Path-specific parity/state oracle for that arch/surface; **blocked** if no oracle exists | Manual oracle — **not** `serve_harness.py` |
| Forward / fusion / sampling / KV **user-facing serve semantics** | `scripts/serve_harness.py` with the exact model (after parity route if the change can break numbers/state); add `scripts/gates.sh` when the Redline+serve+optional perf wrapper is desired | Manual serve (semantics only) |
| LFM2.5 chat framing / thinking output | `scripts/serve_harness.py` with an `lfm2.5:*` registry tag | Manual LFM |
| VL vision-tower forward numerical parity (arch-5 / arch-11 carriers) | Dump-and-diff vs an HF `transformers` reference for the exact checkpoint, pixel inputs pinned by hash (`benchmarks/vision/dump_hf_reference.py` precedent; family route: [`qwen35-vl-mq4v2-spec.md`](qwen35-vl-mq4v2-spec.md) §5, [`specs/2026-08-27-qwen35-vl-vision-serve.md`](specs/2026-08-27-qwen35-vl-vision-serve.md)); **blocked** for a checkpoint with no reference dump | Manual oracle — not `serve_harness.py`; a green VL serve battery is *not* parity evidence |
| VL image-bearing serve semantics (`generate_vl` over `/v1/chat/completions`) | Manual OpenAI-compatible battery through `hipfire serve` with the exact VL artifact: committed fixtures under [`../benchmarks/vision/images/`](../benchmarks/vision/images/) + the fixed desc/ocr prompts of `comparison-2026-05-23.md`, greedy temp 0; stream **and** non-stream typed-emission check (reasoning vs `content` deltas; no literal `<think>`/`<|im_end|>` chunks in content); client-disconnect probe mid-stream followed by an immediate follow-up turn (no slot wedge); eyeball every decoded output; record artifact sha256, fixture hashes, binary md5s. No scripted harness exists (**blocked** until one lands); `scripts/serve_harness.py` is text-only today and does not exercise this surface | Manual serve (semantics only) |
| Retained replay / PM4 / AQL graft | `scripts/redline_daemon_harness.py` **and** the certification steps in `docs/REDLINE.md` | Manual Redline; promotion still policy-gated |
| Perf improvement claim | Protocol in `methodology/perf-benchmarking.md` + stationary matched runs; `speed-gate.sh` or `gates.sh` perf arm when applicable | Measured; not admission |
| Existing sealed MQ4R Redline fixture reproduction | `python3 -m tools.redline golden` for its exact gfx1100/gfx1151/gfx1201 fixture only | Measured reproduction; exact identity + route proof + stationary floor; not admission. Does **not** imply that every default-eligible `.mq4r` model file is one of the sealed fixtures. |
| Arch port | `methodology/arch-port-validation.md` (channel + speed; no retired coherence battery as acceptance) | Manual |
| Model/route **admission** (registry evidence) | Row in [`admissions.yml`](admissions.yml) | Schema v2; exactly one evidence-bound record (LFM2.5-350M MQ4 gfx1201 retained-PM4). No inferred/wildcard rows. Registry admission/evidence is distinct from runtime wiring: the sealed LFM row does **not** select a runtime default and current automatic selection does not use it. |
| MQ4R **runtime** automatic Redline default | Source predicate `mq4r_redline_default` in `crates/hipfire-runtime/src/config.rs`; policy in [`REDLINE.md`](REDLINE.md) | **Only** current automatic runtime predicate. Runtime-only: exact GPU arch `gfx1100`, `gfx1151`, or `gfx1201`; PP=1; TP=1; case-insensitive `.mq4r` → retained PM4/Auto unless disabled with the config wizard's built-in `hip` profile, another explicit backend selection, or `HIPFIRE_REPLAY_BACKEND=hip`. Model-family agnostic (no `arch_id` gate). `gfx1200` and all other arches remain opt-in. Existing LFM `.mq4` registry evidence is not auto-selected because it is not `.mq4r`, not because LFM is categorically exempt; any usable non-default retained route must still prove route support and fail closed when unsupported. **Not** registry admission, **not** Section 7 certification, and **not** a sealed-fixture claim for every default-eligible `.mq4r` model. |
| Escha-W2 codec / H128 transforms / escha routed executors / escha loader | `scripts/gates.sh --escha-only --model <escha .hfq>` (G1-G6; see the Escha-W2 table above), against the recorded values in [`escha-w2-port-design.md`](plans/escha-w2-port-design.md) §10.6 | Manual GPU + model. Bit-exactness claims are against the frozen `escha_ref` oracle. **Not** `serve_harness.py`, and **not** a subset — G4's two `0 differing floats` rows are equality claims and a change that turns either into a tolerance needs its own argument |
| Unknown surface | **Blocked** until an owner adds a row here | Fail closed |

## Retired coherence-gate scripts

The fixed `scripts/coherence-gate-*.sh` batteries are **retired as current
acceptance evidence**. They must not be required for merge, promotion, or
benchmark claims.

| Pattern | Status |
|---|---|
| `scripts/coherence-gate-*.sh` (e.g. `coherence-gate-dflash.sh`, `coherence-gate-qwen35-dspark.sh`, `coherence-gate-minimax.sh`, `coherence-gate-cohere2moe.sh`, `coherence-gate-deepseek4-*.sh`, …) | **Historical reproduction only.** Never promotion or acceptance. |
| Other gate scripts **not named anywhere in this selector** | Do not treat as canonical acceptance unless a future INDEX/VALIDATION revision names them. Supporting tools already listed above stay in force. |

Campaign-specific guidance (for example an LFM effort that omits coherence
gates) is **not** a universal rule for every arch or model. Universality is
blocked on purpose.

## Explicit non-routes

| Anti-pattern | Disposition |
|---|---|
| One script that “replaces all gates” | **Rejected** — no universal gate. |
| Green no-GPU CI as proof of GPU correctness | **Rejected** |
| `serve_harness` as numerical/state parity or universal forward proof | **Rejected** — semantics only; oracle or **blocked** |
| `serve_harness` success as Redline route proof | **Rejected** |
| `redline_daemon_harness` fingerprint as installed product PM4/AQL route | **Rejected** without `REDLINE.md` ladder |
| Coherence-gate pass as current acceptance | **Rejected** |
| Bench number without protocol + identity hashes | **Rejected** as promotion evidence |
| Inferred or “signed” `admissions.yml` row without earned fixture evidence | **Rejected** — schema v2 forbids inferred/wildcard rows; only the exact admitted record applies |

## Related owners

- Navigation / lifecycle: [`INDEX.md`](INDEX.md)
- Admission registry: [`admissions.yml`](admissions.yml)
- Redline policy: [`REDLINE.md`](REDLINE.md)
- Perf protocol: [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md)
