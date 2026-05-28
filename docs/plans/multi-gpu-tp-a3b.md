# Multi-GPU Tensor-Parallel (TP) — A3B MoE on hiptrx

**Status:** Scoping draft (2026-05-28)
**Target hardware:** hiptrx — 4× Radeon AI PRO R9700 (gfx1201/RDNA4, 34.2 GiB/card), HIP 7.2
**Target model:** Qwen3.6 35B-A3B (arch_id=6) — `num_experts=256`, `top_k=8`, `hidden=2048`, `head_dim=256`, `n_heads=8`, `n_kv_heads=2`, `moe_intermediate=512`
**Foundation for:** DSv4 (arch_id=9) TP — same EP shape, MLA attention swap
**Related:** `docs/plans/multi-gpu-pp.md` (PP v1 shipped, refuses TP), `docs/multi-gpu.md` (`Gpus` orchestrator)

---

## 1. Objective

Add tensor-parallel (TP) inference to the A3B MoE forward + MTP-spec serving
path on a 4× R9700 box. Layers stay fully present on every rank; weights are
sharded *within* each layer along the column dim (QKV, gate_up) or row dim
(wo, down). Routed-expert MoE FFN runs **expert-parallel (EP)**: 256/N_tp
experts owned per rank, tokens routed via all-to-all on `top_k=8` selection.

**What v1 unlocks:**

- Multi-user serving with shared KV / shared expert pages — DP=4 can't do
  this without 4× the KV footprint per concurrent user.
- A3B with `max_ctx ≥ 32K` where per-card KV pushes a DP=4 worker over
  the 34 GB R9700 budget at batch>1.
- Foundation for DSv4 TP (MLA attention is the only new piece — sharding
  shape stays the same for the MoE/EP path).

**What v1 does NOT give you:**

- Beats DP=4 on raw multi-user throughput for *short* contexts. If your
  workload fits 4 independent 23.5 GB model copies and you don't need
  long context per user, **DP is simpler and faster** — see §2.3.
- Eliminate PP. PP and TP are composable but not redundant; PP for
  models bigger than per-card+shard fit, TP for multi-user throughput.

---

## 2. Why TP — and why not DP

### 2.1 What we have now (post-PR #352)

- **PP v1** ships on master via `feat/multi-gpu-pp`. Splits layers across
  N devices, `s.x` cross-band copy per boundary, refuses
  `DFlash + pp>1`, `CASK + pp>1`, `arch_id ∉ {5,6} + pp>1`. PP-on-A3B
  decode is ~68% of single-card throughput per `docs/multi-gpu.md`
  (PR #58 baseline: 142 → 97 tok/s on A3B mq4 with PP=2 on 2× 7900 XTX).
- **MTP serial + DFlash** (PR #352 just landed) — A3B MTP runs at
  K=5 p=0.5 on R9700; DFlash + A3B target is the production code path.
- **No TP, no DP, no EP** in the current code.

### 2.2 The shape A3B-on-hiptrx wants

| Axis | A3B-friendly | A3B-hostile | Pick |
|---|---|---|---|
| Weight fit (mq4 35B-A3B) | 23.5 GB | 34 GB R9700 budget | Single-card OK |
| Single-user latency | TP all-reduce per layer + EP all-to-all | per-token per-layer comm cost | DP wins |
| Multi-user throughput | DP=4 = 4× linear (no comm) | TP=4 + larger effective batch per rank | DP wins for fitting models |
| Context > ~24K with batch > 1 | DP=4 each card OOMs on KV | TP=4 shares KV across ranks | TP wins |
| DSv4 (won't fit 34 GB) | TP=4 mandatory | DP=4 impossible | TP wins |
| MTP-spec | A3B path runs target+draft on one rank | TP=4 splits both, comm dominates | unclear — bench-driven |
| DFlash | Refused on pp>1 currently; TP-composition is open | per-step comm × draft K | bench-driven |

### 2.3 DP=4 alternative (mandatory honest accounting)

On 4× 34 GB R9700 with mq4 A3B (23.5 GB):

- **DP=4**: zero cross-device comm, four independent workers. ~4× linear
  throughput up to per-worker batch saturation. **Simpler to ship: ~1
  week vs. ~6+ weeks for TP+EP.** No `Gpus`-internal sharding changes.
  Wins for: short-context multi-user serving, ChatML batch APIs.
- **TP=4**: shared KV / shared expert pages, scales to longer context
  per concurrent user, foundation for DSv4 (which DP can't run at all).
  Wins for: extended context multi-user, models > 34 GB.

**Recommendation:** ship DP=4 first (~1 week, mostly daemon work — see
§Appendix A), defer TP to a deliberate "we need DSv4 OR extended-context
multi-user" trigger. If the user has already decided TP, proceed as below
and treat DP as an explicit non-goal.

For the rest of this doc we assume TP is the chosen path.

---

## 3. Architectural decisions

### 3.1 TP=4 with KV-replicated, GQA-aware Q-sharding

A3B has `n_heads=8`, `n_kv_heads=2` (GQA group size = 4). Clean splits:

| TP | Q heads/rank | KV heads/rank | KV strategy |
|---|---|---|---|
| 1 | 8 | 2 | single-card |
| 2 | 4 | 1 | clean split |
| 4 | 2 | **2 (replicated)** | each rank holds both KV heads |
| 8 | 1 | 2 (replicated) | unused — only 4 cards |

**Decision:** v1 ships **TP=4 with KV replicated** as the hiptrx-canonical
shape. KV-replicated wastes (n_kv_heads × head_dim × ctx × 2 bytes) per
extra rank — for A3B that's 2×256×4096×2 = 4 MB/layer KV per rank baseline,
or 1 GB/rank at 32K ctx with 64 layers — manageable. TP=2 + DP=2 is a
documented alternative (cleaner KV split, four-worker throughput).

Hard-fail at load when `tp > n_kv_heads` and `tp_kv_replicate=false`.
Default `tp_kv_replicate=true` on hiptrx (4 cards).

### 3.2 Sharding axes (per A3B layer block)

Following Megatron-LM convention:

| Tensor | Shape (A3B 35B) | Shard axis | Comm |
|---|---|---|---|
| `wq` | [n_heads × head_dim, hidden] = [2048, 2048] | col (dim 0) → [512, 2048]/rank | none after shard |
| `wk` | [n_kv_heads × head_dim, hidden] = [512, 2048] | replicated for TP=4 | none |
| `wv` | [n_kv_heads × head_dim, hidden] = [512, 2048] | replicated for TP=4 | none |
| `wo` | [hidden, n_heads × head_dim] = [2048, 2048] | row (dim 1) → [2048, 512]/rank | **all-reduce on residual** |
| Shared expert `gate_up` | [2 × moe_inter, hidden] = [1024, 2048] | col → [256, 2048]/rank | none after shard |
| Shared expert `down` | [hidden, moe_inter] = [2048, 512] | row → [2048, 128]/rank | **all-reduce on residual** |
| Routed `experts[e].gate_up` | [1024, 2048] × 256 | **expert-shard**: rank owns 64 experts | all-to-all on token activations |
| Routed `experts[e].down` | [2048, 512] × 256 | expert-shard, same 64 set/rank | all-to-all on expert outputs |
| `router` | [num_experts, hidden] = [256, 2048] | **replicated** | none — router runs on every rank |
| `token_embd` | [vocab, hidden] = [248K, 2048] | replicated v1 (~500 MB at int4) | none |
| `lm_head` | [vocab, hidden] | replicated v1 | none — optional col-shard in v1.1 |
| `output_norm`, per-layer norms | tiny | replicated | none |

### 3.3 Comm primitives and per-layer cost (measured 2026-05-28)

**Original estimates here (host-driven ring on `boundary_copy`) were
6.8× too optimistic for attn all-reduce.** See
`docs/investigations/2026-05-28-tp-comm-baseline-hiptrx.md` for the full
measurement. The numbers below are the measured floor on 4× R9700
gfx1201 hardware with RCCL 1.0.70202 used for all-reduce.

Hardware floor: peer-copy + stream sync is ~22 µs per call regardless
of payload up to ~30 KB — small payloads are latency-bound, not
bandwidth-bound. PCIe link is Gen4 x8 per card (~14 GB/s peer ceiling
at 512 KB payloads, not the ~25 GB/s the original estimate assumed).

Per token per layer at TP=4:

1. **Attn all-reduce on residual** (`[1, hidden=2048] × 2 B = 4 KB`):
   - host-driven ring on `boundary_copy`: **340 µs** (6 sequential steps × 22 µs floor + ~150 µs cross-stream event-chain serialization)
   - RCCL `ncclAllReduce`: **110 µs** (single in-kernel collective using peer-mapped loads + inline reduction; flat from 4 KB → 128 KB → bandwidth-bound only at 512 KB+)
   - **Adopted: RCCL** — 3.07× faster, no orchestrator changes needed beyond FFI wrapper.

2. **MoE all-to-all dispatch** (`top_k=8` activations × 4 KB = 32 KB):
   - host-driven `boundary_copy` per (src,dst) pair: **118 µs** (3 sequential sends per src stream × 4 src streams running in parallel = no event-chain serialization, unlike ring all-reduce)
   - RCCL `ncclAllToAll`: 115 µs at 32 KB, but **399 µs at 512 KB (1.78× worse than host-driven)** — RCCL's all-to-all bandwidth scaling on gfx1201 is poor.
   - **Adopted: host-driven** for now. Revisit with custom peer-read kernel if prefill-comm becomes a bottleneck.

3. **Shared expert all-reduce on residual**: same shape as (1), RCCL,
   **110 µs**.

**Per-MoE-layer comm cost (TP=4, mixed RCCL+host):** 110 + 118 + 110 = **335 µs**.
**Per-token comm @ 64 layers: 21.4 ms** (was 13 ms estimated, 51 ms
host-driven measured).

Single-card A3B decode runs ~7 ms/token (143 tok/s baseline).
**TP=4 single-user batch=1 decode is still comm-bound (3.06× ratio)** —
TP=4 cannot match single-card latency for single-user serving.

**However, batch amortization changes the verdict:**

| Batch | Comm/token | Compute/token (64 layers, single-card-equivalent) | Bound by |
|---|---:|---:|---|
| 1 | 21.4 ms | 7.0 ms | comm (3.06×) |
| 4 | ~21.4 ms (amortized) | 28 ms | **compute (1.3×)** |
| 16 | ~21.4 ms | 112 ms | strongly compute (5.2×) |
| 32 | ~21.4 ms | 224 ms | strongly compute (10×) |

At batch≥4 TP=4 should match DP=4 aggregate throughput while
additionally enabling extended-context via shared KV. **Bench gate
(Stage 8) now measures the batch-amortized regime**: pass requires
TP=4 batch=4 ≥ 0.9× DP=4 batch=4 OR extended-context-only justification.

### 3.4 Reusing the `Gpus` orchestrator

`crates/hipfire-runtime/src/multi_gpu.rs` already gives us:

- `devices: Vec<Gpu>` — per-rank device handle + rocBLAS handle
- `enable_peer_all`, `boundary_copy`, `wait_boundary` — pairwise async
  peer copy + event handoff
- `bind_thread()` invariant — every `Gpu::*` call bound to single thread
- VRAM preflight, arch-match check, `HIPFIRE_DEVICES` resolution

What TP adds (new file, parallel to `multi_gpu.rs`):

- `crates/hipfire-runtime/src/tp_shard.rs` —
  `ShardConfig { tp_size, tp_kv_replicate, expert_to_rank: Vec<u8> }`,
  `ColShard / RowShard` tensor wrappers, `all_reduce_sum(buf)`,
  `all_to_all_scatter(payloads, dst_rank_per_payload)`, etc.

When `tp_size == 1`, all of these degenerate to the single-rank path
(byte-identical to pre-TP code). When `tp_size > 1`, `Gpus.devices.len()
== tp_size` and `layer_to_device == vec![0; n_layers]` (PP=1 alongside
TP=N is v1; PP+TP composition is v1.1).

### 3.5 Router fp32 precision across ranks

Router runs on every rank with replicated `[num_experts, hidden]`
weights (no comm). Each rank computes the same logits → same softmax →
same top-k → same routing decision. **No all-reduce needed**, and the
fp32 cast for softmax (per `docs/plans/qwen35-moe-precision-vllm-comparison.md`)
holds locally on each rank.

The byte-equality requirement: replicated router weights MUST be
bit-identical across ranks (load once on dev 0, peer-copy to others —
NOT load independently per rank, which can hit different MQ4 dequant
codepaths depending on JIT cache state).

### 3.6 EP routing: who owns which expert

`ShardConfig.expert_to_rank: Vec<u8>` length 256. Default
contiguous-block assignment: rank `r` owns experts
`[r*64, (r+1)*64)`. Alternative: stride-N (expert `e` → rank `e % N`)
which load-balances better when top-k draws are non-uniform — gated
behind `HIPFIRE_TP_EXPERT_ASSIGN={contiguous,stride}` env var,
default `stride`.

Empirical comparison required (Stage 5 validation). MoE token
distribution skew matters: if 90% of tokens hit one expert, neither
scheme helps — but stride averages over hot indices, contiguous
clusters them.

---

## 4. Scope (v1)

### 4.1 In scope

- **TP=2 and TP=4** for A3B MoE decode + prefill on
  `crates/hipfire-arch-qwen35/src/qwen35.rs` (arch_id=6 FullAttnMoe and
  DeltaNetMoe layer branches).
- KV-replicated mode (TP=4 default on hiptrx).
- Column-sharded QKV, row-sharded wo with all-reduce on residual.
- Column-sharded shared-expert gate_up, row-sharded down, all-reduce.
- **Expert-parallel** routed MoE: all-to-all scatter + gather around
  `expert.gate_up_residual + expert.down`.
- Router replicated, fp32 softmax local per rank, byte-identical top-k
  selection across ranks (gated test).
- Embedding + lm_head replicated v1 (optional col-shard v1.1).
- `Gpus::init_tp(tp_size, tp_kv_replicate)` constructor parallel to
  `init_uniform`. PP=1 enforced (`layer_to_device == [0; n_layers]`).
- Daemon `tp` field in load message. Default `tp=1`.
- **Refuse at load** (v1):
  - `tp > 1` + `pp > 1` (composition is v1.1)
  - `tp > 1` + DFlash (composition is v1.2 — see §8)
  - `tp > 1` + CASK / TriAttention (single-device eviction state)
  - `tp > 1` + arch_id ∉ {5, 6} (LLaMA / Qwen2 / dots.ocr / DSv4)
- TP parity gate: TP=1 ↔ TP=2 ↔ TP=4 byte-identical greedy token stream
  ≥100 tokens on Qwen3.5 0.8B + Qwen3.6 35B-A3B.
- Per-rank bind_thread audit on new shard ops.
- Performance gate: TP=4 batch=32 decode tok/s vs DP=4 batch=32 — must
  exceed DP=4 by ≥10% OR justify ship anyway (e.g., context-extension
  unlock).

### 4.2 Out of scope (v1, deferred to v1.1+)

- **DP=N + serve manager** — separate roadmap (see §Appendix A).
- **PP + TP composition** (Megatron-style 2D parallelism) — v1.1.
- **DFlash + TP** — v1.2; draft + target ranks must coordinate
  speculative-accept rollback; see open question §8.4.
- **MTP-spec + TP** — v1.1 (MTP head is small, has own MoE; sharding
  the head's MoE doubles up on a 4-card box and may not help; bench
  first, scope later).
- **Vocab/lm_head sharding** — v1.1 (memory savings ~500 MB/rank;
  worth it only when other paths exist for KV-bound use cases).
- **DSv4 TP** — separate roadmap once the MLA-attention shard shape is
  validated against the reference (see §6 Stage 10).
- **TP on dense Qwen3.5 (arch_id=5)** — v1.1; A3B is the load-bearing
  case, dense Qwen3 fits single-card.
- **Pipelined comm overlap** (compute next-layer QKV while wo all-reduce
  in flight) — v1.2.
- **NCCL/RCCL adoption** — out. We own the comm path through
  `boundary_copy` + a thin all-reduce wrapper; pulling RCCL in for 4
  cards is premature.

---

## 5. Stages

### Stage 1 — `tp-1-shard-config` (~2d)

New: `crates/hipfire-runtime/src/tp_shard.rs`.

- `ShardConfig` struct (tp_size, tp_kv_replicate, expert_to_rank).
- `Gpus::init_tp(tp_size, tp_kv_replicate, n_layers, n_experts)` —
  parallels `init_uniform`; sets `layer_to_device = [0; n_layers]`;
  populates `expert_to_rank` per `HIPFIRE_TP_EXPERT_ASSIGN`.
- Pre-flight: hard-fail if `tp_size > devices.len()` or
  `tp_size > n_kv_heads && !tp_kv_replicate`.
- Tests: ShardConfig construction, expert assignment correctness,
  TP=1 degeneracy (byte-identical to single-card).

### Stage 2 — `tp-2-rccl-allreduce` (~2d) — **revised 2026-05-28**

Empirical comm bench (see investigation doc) shows host-driven ring
all-reduce on `boundary_copy` is 3× slower than RCCL on the same
hardware. The original Stage 2 (build our own ring) is replaced with
an RCCL FFI wrapper.

New: `crates/hip-bridge/src/rccl.rs` — dlopen-style FFI for librccl.

- Bind: `ncclCommInitAll`, `ncclCommDestroy`, `ncclAllReduce`,
  `ncclGroupStart`, `ncclGroupEnd`, `ncclGetErrorString`,
  `ncclGetVersion`. Optional later: `ncclSend`, `ncclRecv`,
  `ncclBroadcast`, `ncclReduceScatter` for future primitives.
- Safe Rust wrappers: `RcclComm` handle, `Drop` calls `ncclCommDestroy`,
  error type maps `ncclResult_t` → `RcclError`.
- Mirror `hip_bridge::HipRuntime`'s dlopen pattern so the runtime
  cleanly fails-soft when librccl is missing (`HIPFIRE_TP_USE_RCCL=0`
  forces the fallback path).

New on `Gpus`: `Gpus::all_reduce_sum(buffers: &[&mut DeviceBuffer],
n_elems, dtype)` backed by RCCL.

- Initializes RCCL communicators lazily on first all-reduce (cached
  for process lifetime in a new `Gpus.rccl_comms: Option<Vec<RcclComm>>`).
- Fallback path: if `RcclComm::init_all` fails OR
  `HIPFIRE_TP_USE_RCCL=0`, build the host-driven ring on `boundary_copy`
  (the original Stage 2 design, now demoted to fallback). Documented as
  3× slower but functionally correct — keeps the door open for arches
  where RCCL doesn't ship or fails to load.
- Test: `tp_allreduce_smoke` — RCCL path byte-correct (sum of 4 known
  per-rank vectors matches), fallback path byte-correct, both at
  payload sizes {4, 32, 128, 512} KB.

Stage 2 also brings the comm-microbench (`tp_comm_smoke.rs`) into the
test harness as a perf-regression guard — fails CI if the measured
RCCL all-reduce 4 KB drifts >20% from the 2026-05-28 baseline.

**Out of Stage 2:** all-to-all stays on host-driven `boundary_copy`
per §3.3. RCCL `ncclAllToAll` is benched in the investigation doc but
not adopted (worse at 512 KB; ties at small sizes).

### Stage 3 — `tp-3-attn-shard` (~4-5d)

`crates/hipfire-arch-qwen35/src/qwen35.rs`,
`crates/rdna-compute/src/dispatch.rs`.

- `Qwen35Weights::load_weights_tp(hfq, config, gpus, shard)` — each rank
  loads its column-slice of wq, row-slice of wo, full wk/wv (replicated).
- Shard-aware dispatch: `gpu.weight_gemv(rank_local_wq, x, q_local)`
  produces `q_local` of width `n_heads/tp × head_dim`. K, V dispatched
  on each rank from local-full wk/wv. FA / DFA runs locally on
  rank-local Q-head slice. wo GEMV produces local partial residual.
- All-reduce-sum on `s.x` after wo.
- `forward_scratch_tp`: parallels `forward_scratch` (line 4212); 4
  layer-type branches reproduced (DeltaNet, FullAttn, DeltaNetMoe,
  FullAttnMoe). FFN branches still single-rank in Stage 3.
- Parity acceptance: TP=2 ↔ TP=1 logits within 1e-4 on 0.8B greedy.

### Stage 4 — `tp-4-shared-expert-shard` (~2d)

Shared-expert gate_up col-shard, down row-shard, all-reduce on residual.
Parallels Stage 3 for the shared-expert path. No EP yet.

Acceptance: TP=2 byte-identical token stream ≥100 tokens on A3B for a
prompt that activates shared-expert only (skip routed top-k via a
debug env if needed for isolation).

### Stage 5 — `tp-5-expert-parallel` (~6-8d) — **largest stage**

`crates/hipfire-arch-qwen35/src/qwen35.rs` `moe_ffn_decode_impl`
(line 2974), `dispatch.rs` indexed-MoE kernels.

- Each rank loads only its 64 (TP=4) or 128 (TP=2) routed-expert
  weights. `expert_gate_up_ptrs`, `expert_down_ptrs` (qwen35.rs:427-428)
  populated with local-expert pointers only.
- Router replicated on every rank → identical top-k → each rank knows
  who-owns-what via `shard.expert_to_rank[expert_id]`.
- All-to-all primitive: per token, each rank sends 4-KB activation to
  up-to-`top_k`-distinct destination ranks; receives back the expert
  output. Schedule: K rounds of `boundary_copy` per layer where
  `K = max group size in routing decision`. For top_k=8, TP=4: most
  tokens fan out to all 4 ranks once → 1-3 rounds typically.
- `moe_down_combine_k8` (qwen35.rs:3953-ish, the existing per-token
  combine kernel) runs locally on the origin rank after gathering.
- Shared expert all-reduce composes with routed all-to-all gather.

**Risk:** highest in v1. Mitigations: synthetic 4-rank all-to-all bench
in `crates/rdna-compute/examples/tp_all2all_smoke.rs` before integration;
hard-fail if scatter dst-rank exceeds `tp_size`; debug dump of
per-rank expert counts per layer.

Acceptance: TP=4 byte-identical token stream ≥100 tokens on A3B
greedy temp=0 against TP=1.

### Stage 6 — `tp-6-prefill` (~3-4d)

Prefill loop multi-rank version. Boundary-copy sizes scale with
`batch_size`: for batch=128, residual all-reduce is `128 × 2048 × 2 =
512 KB` per layer → ~20 µs/layer. MoE all-to-all scales similarly.

Pipelined comm overlap is out of scope (v1.2).

### Stage 7 — `tp-7-daemon` (~2d)

`crates/hipfire-runtime/examples/daemon.rs`.

- Read `tp` from load command. Default `tp = gpus.devices.len()` only
  when explicitly `init_tp` (do NOT default-TP an init_uniform call).
- Refusal matrix at load: TP + PP, TP + DFlash, TP + CASK, TP + arch∉{6}.
- `LoadedModel` extension: `tp_gpus`, `tp_shard`, `tp_kv_caches`.
- Sample/argmax reads `s.logits` from rank 0 (replicated lm_head means
  every rank has it; pick rank 0 by convention).
- Reset routes per-rank state in parallel.

### Stage 8 — `tp-8-bench` (~2-3d) — **gate the ship**

New: `crates/hipfire-runtime/examples/tp_bench.rs`.

- Decode tok/s at batch ∈ {1, 4, 16, 32} for TP={1, 2, 4} and DP={1, 2,
  4} (DP simulated with multi-process daemon).
- Per-layer comm-time breakdown via `HIPFIRE_PROFILE_DECODE=1`.
- Output `tests/speed-baselines/hiptrx_tp.txt` for regression tracking.
- **Ship gate:** TP=4 batch=32 ≥ 1.1× DP=4 batch=32 OR documented
  context-extension unlock.

### Stage 9 — `tp-9-validation` (~2d)

`scripts/tp-gate.sh`, `scripts/coherence-gate.sh --tp N`,
`.githooks/pre-commit`, `crates/hipfire-arch-qwen35/tests/tp_parity.rs`.

- `tp-gate.sh`: env-gated on `HIPFIRE_HAVE_4_GPU=1`; runs the parity
  battery (Qwen3.5 0.8B TP=1↔2↔4 byte-identical at temp=0) +
  refusal-matrix tests + comm-primitive smoke.
- Pre-commit hotspot regex extension: `tp_|all_reduce|all_to_all|
  expert_to_rank|ShardConfig|init_tp`.
- Coherence gate `--tp 4` for any forward / dispatch / kernel change.

### Stage 10 — `tp-10-dsv4-bring-up` (separate PR, blocked on v1)

Once v1 lands, scope DSv4 TP. Sharding shape carries over almost
unchanged for the MoE/EP path; the new work is **MLA attention TP**
(latent-KV all-reduce shape differs from MHA/GQA — DSv4 latent_dim is
typically 512, sharded col-wise with all-reduce on the absorbed-K
projection). Out of scope for this doc.

---

## 6. Validation matrix

| # | Test | Hardware | Command |
|---|---|---|---|
| 1 | Single-GPU regression | 1× any | `cargo test --workspace && ./scripts/coherence-gate.sh` |
| 2 | TP=1 byte-identical to pre-TP | 1× any | `./scripts/coherence-gate.sh --tp 1` |
| 3 | TP=2 token parity (0.8B) | 2× hiptrx | `./scripts/coherence-gate.sh --tp 2 --model qwen3.5:0.8b` |
| 4 | TP=4 token parity (A3B) | 4× hiptrx | `HIPFIRE_HAVE_4_GPU=1 cargo test tp_parity_a3b --release` |
| 5 | Refusal: TP+PP, TP+DFlash, TP+CASK, TP+arch∉{6} | any | `./scripts/tp-gate.sh --refusal` |
| 6 | EP scatter correctness | 4× hiptrx | `tp_all2all_smoke` — 4-rank token routing matches expected |
| 7 | Comm-time breakdown | 4× hiptrx | `HIPFIRE_PROFILE_DECODE=1 hipfire bench qwen3.6:35b-a3b --tp 4` |
| 8 | TP=4 vs DP=4 throughput | 4× hiptrx | `tp_bench.rs` produces `hiptrx_tp.txt` regression baseline |
| 9 | Long-ctx KV fit (32K, batch=4) | 4× hiptrx | TP=4 succeeds where DP=4 OOMs (the actual TP-win condition) |
| 10 | Coherence gate (greedy A3B) | 4× hiptrx | `./scripts/coherence-gate.sh --tp 4 --model qwen3.6:35b-a3b` |

---

## 7. Critical files (cheat sheet)

| File | Role |
|---|---|
| `crates/hipfire-runtime/src/multi_gpu.rs` | `Gpus` orchestrator — reused; v1 adds `init_tp` constructor + `all_reduce_sum` (RCCL-backed) |
| `crates/hipfire-runtime/src/tp_shard.rs` | **new** — `ShardConfig`, expert-to-rank, all-to-all on `boundary_copy` |
| `crates/hip-bridge/src/rccl.rs` | **new (Stage 2)** — dlopen-style FFI for librccl: `ncclCommInitAll`, `ncclAllReduce`, group ops |
| `crates/hipfire-runtime/examples/tp_comm_smoke.rs` | **shipped 2026-05-28** — comm microbench (host-driven path) |
| `docs/investigations/2026-05-28-tp-comm-baseline-hiptrx.md` | comm-cost baseline + RCCL comparison; load-bearing data for §3.3 |
| `crates/hipfire-arch-qwen35/src/qwen35.rs:2974` | `moe_ffn_decode_impl` — EP shard target |
| `crates/hipfire-arch-qwen35/src/qwen35.rs:3787, 3849` | FullAttnMoe / DeltaNetMoe forward branches |
| `crates/hipfire-arch-qwen35/src/qwen35.rs:4212` | `forward_scratch` — `forward_scratch_tp` parallel |
| `crates/rdna-compute/src/dispatch.rs` | per-rank kernel dispatch; bind_thread audit applies |
| `crates/rdna-compute/examples/tp_all2all_smoke.rs` | **new** — Stage 5 comm-primitive smoke |
| `crates/hipfire-runtime/examples/tp_bench.rs` | **new** — Stage 8 ship-gate bench |
| `scripts/tp-gate.sh`, `scripts/coherence-gate.sh` | gate scripts |

---

## 8. Open questions (need maintainer decision before Stage 1)

### 8.1 DP=4 first? — softened after RCCL measurement

**Original recommendation (pre-RCCL bench):** ship DP=4 first; TP=4 is
hardware-comm-bound and can't match DP=4 throughput.

**Revised recommendation (post-RCCL bench, 2026-05-28):** the
DP-first call is no longer obvious. With RCCL all-reduce, TP=4 at
batch=4 should match DP=4 aggregate throughput AND adds extended-context
capability via shared KV. The DP-vs-TP decision now depends on:

- **Workload batch profile.** If your A3B traffic averages batch<2,
  DP=4 still wins (TP single-user is 3× slower per-token). If batch≥4
  is the steady state, TP=4 wins on the long-ctx axis at parity
  throughput.
- **Engineering cost.** DP=4 is ~1 week; TP=4 v1 is ~6+ weeks even with
  RCCL doing the heavy lifting. If you need to ship A3B multi-user
  serving in 2 weeks, ship DP=4 and revisit TP=4 next quarter.
- **DSv4 timeline.** TP infrastructure is mandatory for DSv4; if DSv4
  is the actual driver, build TP=4 now and use A3B as the v1 validation
  workload. DP=4 then becomes a smaller follow-up that wraps the
  daemon, not a separate code path.

**My current recommendation:** if DSv4 lands within ~2 months, **start
TP=4 with RCCL** (Stage 1 → Stage 9 as written, ~6 weeks). If DSv4
timeline is unclear and you want quick multi-user A3B throughput on
hiptrx, **ship DP=4 first** (~1 week, daemon-only) and let the TP
roadmap stretch.

### 8.2 TP=4 with KV-replicate or TP=2 ship target?

TP=2 has cleaner KV math (1 KV head/rank, no replication overhead),
runs on 2 cards, and 4×hiptrx can run TP=2 + DP=2. TP=4 requires KV
replication but uses all 4 cards in one model copy. **My recommendation**:
ship both, default to TP=2 in the v1 daemon (lower-risk, faster decode
per-rank), document TP=4 as the long-context option.

### 8.3 Expert assignment: contiguous block vs stride

§3.6. **My recommendation**: stride default, contiguous opt-in via env.
Validate empirically in Stage 5.

### 8.4 DFlash + TP composition scope

DFlash currently refuses pp>1; will refuse tp>1 in v1. The right v1.2
shape: draft model runs on rank 0 only (it's small), target model
sharded across all ranks. Verify-batch all-reduce coordinates accept/
reject decision. This is a substantial design — defer to v1.2 doc.

### 8.5 Refusal contract for MTP-spec + TP

PR #352's MTP head has its own MoE (independent of trunk MoE). MTP +
TP could shard both, but the MTP head is small enough that single-rank
may be fine and trunk-TP-only might compose cleanly. **My recommendation**:
v1 refuses MTP + TP at load; v1.1 PR explicitly benches both shapes and
picks one.

---

## Appendix A — DP=4 alternative shape (recommended ship-first)

For completeness, the DP=4 path that this TP doc displaces or precedes:

- Each card runs an independent `LoadedModel` (full A3B mq4, 23.5 GB).
- Daemon spawns 4 worker processes (or 4 inline forward threads, but
  the single-thread-per-HIP-work invariant means worker threads need
  separate processes — see `crates/hipfire-runtime/src/multi_gpu.rs:8`).
- Frontend round-robins or least-loaded-routes incoming requests across
  4 worker daemons.
- Per-request state (KV cache, MTP head state) lives on one worker for
  the request's lifetime.
- DFlash, CASK, MTP — all work unchanged per-worker (no refusal contract).
- Memory: 4 × (23.5 GB weights + KV per concurrent user). Loses to TP
  when concurrent KV pushes a single worker over 34 GB.

Estimated implementation: ~1 week. New crate: `hipfire-serve-router` or
extension to existing daemon. No `Gpus` / qwen35.rs changes. The fact
that this is so much cheaper than TP+EP is why §8.1 recommends shipping
it first.

---

## 9. References

- `docs/multi-gpu.md` — PP v1 user-facing reference
- `docs/plans/multi-gpu-pp.md` — PP roadmap; TP slots into the same `Gpus` shape
- `docs/plans/qwen35-moe-precision-vllm-comparison.md` — fp32 router cast justification
- `docs/investigations/2026-05-28-qwen36-27b-spec-r9700.md` — A3B serving baselines on hiptrx hardware
- `crates/hipfire-runtime/src/multi_gpu.rs` — Gpus orchestrator
- Megatron-LM Shoeybi et al. 2019 — TP column/row shard convention
- vLLM TP implementation — `pp_size × tp_size` shape, refusal matrix precedent
- DeepSeek-V3 paper §3.4 — EP all-to-all routing; DSv4 carries this forward
