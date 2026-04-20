# Agentic Sidecar Training — Cost & Strategy

**Date:** 2026-04-19
**Status:** Active priority. Replaces from-scratch DFlash draft training per `dflash-from-scratch-replication.md`.

## Background

TriAttention (hipfire's sidecar) is a per-target calibration that captures band-wise statistics of pre-RoPE Q vectors from the target model's attention heads on a representative corpus. The sidecar then pairs with any compatible draft at inference to improve τ — especially on OOD or domain-specific text where the target's vanilla attention patterns don't match the draft's assumptions.

Empirical result from 2026-04-18: **agentic-corpus-calibrated sidecars produce clean `<tool_call>` output on Hermes traces** while wikitext-calibrated sidecars degenerate on the same traces with the same models. That's the domain-specialization win.

## Cost comparison — where to run sidecar cal

Sidecar calibration is **per-target** (independent of draft). For our 4-target matrix:
- Qwen3.5-4B
- Qwen3.5-9B
- Qwen3.5-27B
- Qwen3.5-35B-A3B (or 3.6-A3B when we ingest it)

Current state: CPU-bottlenecked. Per `/root/chain_logs/9b_sidecar_cal.log` timing, 9B at 1M tokens took ~2hr (only 0.8% of that was GPU utilization per `rocm-smi`). CPU-side BandAccumulator loop dominates.

### Cost scenarios

| scenario | wall time per sidecar | × 4 sidecars | unit cost | **total** |
|---|---|---|---|---|
| Current code, sequential on MI300X | 2 hr | 8 hr | $2/hr | **$16** |
| Current code, sequential on 7900 XTX local | 3-4 hr | 12-16 hr | $0 | **$0** |
| Current code, parallel on 8× MI300X | 2 hr (wall) | 2 hr wall, 8 cards active | $48/hr (8×) | **$96** |
| With CPU-GPU split on 7900 XTX local | 30-40 min | 2-3 hr | $0 | **$0** |
| With CPU-GPU split on 8× MI300X (parallel) | 30-40 min | 40 min wall | $48/hr | **$32** |

**Conclusion: the cheapest path is implementing CPU-GPU split and running all 4 locally on the 7900 XTX.** Zero dollars, under 3 hours wall, hardware is ours forever.

Serial on MI300X at $16 is the "if CPU-GPU split is too much work" fallback. Still cheap but loses the free-local-hardware advantage.

Parallel on 8× MI300X makes no sense for sidecar cal alone — you pay for 8 cards, only need 4, and the wall-time improvement is small. Only worth it if co-locating with 3.6-A3B draft training.

## Empirical results — GPU kernel was a perf regression (2026-04-19)

Implemented the HIP reduce kernel as designed. Results on MI300X 9B + agentic
corpus (100k tokens):

| path | wall | throughput |
|---|---|---|
| CPU baseline | 720s | 139 tok/s |
| CPU + rayon-per-sample | 625s | 160 tok/s (+15%) |
| **GPU kernel (default)** | **1194s** | **84 tok/s (−40%)** |

Sidecar numerical correctness ✓ (max relative diff 3.6e-10 GPU vs CPU).

**Why the GPU kernel lost:**
1. The "99% CPU-bound" signal from `/usr/bin/time` was misleading — CPU at
   100% included time blocked on async GPU memcpy. The CPU was already
   overlapping accumulation with the next forward via the pipelined
   download pattern. Moving work to GPU eliminated that free overlap.
2. Per-chunk launch overhead: 8 kernel launches × 316k chunks = 2.5M HIP
   dispatches on the same stream as the forward — each dispatch serializes
   against forward due to implicit stream ordering.
3. Grid sizing problem on short chunks: for the first 100k tokens of the
   agentic corpus (avg ~3 tok/chunk), each (head × band) = 2048 blocks
   per kernel call only do ~3 sqrts each. Block setup + reduction overhead
   swamps the actual work.

**Outcome**: kernel kept in codebase for future experimentation (long-chunk
corpora may win), but `triattn_validate` defaults to CPU path. `--gpu-calib`
flag opts into the kernel. Rayon-per-sample change kept (+15% real gain).

For the 4-sidecar agentic_xl matrix, use CPU path — ~2hr per target on
MI300X, ~$8 for the matrix. Still cheaper than trying to optimize further.

## CPU-GPU split design (original implementation — NOT shipping as default)

### Current bottleneck

Per `crates/engine/examples/triattn_validate.rs` with CPU-side `TriAttnCalibState::add_sample`:

Per-prompt flow today:
1. GPU runs `qwen35::forward_prefill_batch` for chunk (~200-500 tokens).
2. For each FA layer, pre-RoPE Q tensor is **downloaded from GPU to host**.
3. CPU loops over Q: `16 heads × 128 bands × ~300 tokens × 8 FA layers = 4.9M BandAccumulator.add() calls` per chunk.
4. GPU is **idle** during step 3 (rocm-smi shows 0% most of the time).

For 316k chunks × 1M tokens, this totals:
- ~128 GB of Q data crossing PCIe (128 bytes per position × 1M positions × 8 layers)
- ~16 billion accumulator ops on CPU
- Wall time dominated by CPU phase, not GPU phase.

### Proposed pipelined design

Two approaches, in order of preference:

**Option A — GPU reduction kernel (cleanest, ~1 day work)**

Replace the CPU `add_sample` loop with a HIP kernel that runs on the Q tensor already in GPU memory. Kernel signature:

```c
__global__ void triattn_accumulate(
    const __half* q_device,              // [n_tokens, n_heads, head_dim]
    BandAccumulator* accs_device,         // [n_layers, n_heads, n_bands]
    int n_tokens, int n_heads, int head_dim,
    int layer_idx
);
```

Each block handles one (head, band) pair across all tokens, producing per-accumulator sum/sumsq/count via shared-memory reduction + single atomic add at the end. Classic parallel-reduce pattern.

Wire-up in `crates/engine/src/triattn.rs`:
- Replace `record_prerope_q` → add `record_prerope_q_device(layer_idx, q_device_buf)` variant.
- New tap state enum variant `CalibrateGpu` holds device-resident `accs_device` buffer.
- `finalize()` downloads the small `accs_device` buffer ONCE at end and runs the existing `BandAccumulator::finalize` per band to produce the `TriAttnCenters`.

Zero PCIe data transfer for Q during calibration. GPU stays at 100% during the accumulation. Bottleneck becomes pure target-forward throughput.

Expected speedup: **~5×** (from 100 tok/s to ~500 tok/s on MI300X, similar on 7900 XTX).

**Option B — Pipelined CPU accumulation with Rayon/SIMD (simpler, ~4hr work)**

Keep the CPU path but:
- Use `rayon::par_iter()` over heads, since accumulators for different heads are independent.
- SIMD the inner band loop (f32 sum + sumsq, complex — AVX2/AVX-512 via `std::simd` or `packed_simd`).
- Double-buffer: while CPU processes chunk N's downloaded Q, GPU runs chunk N+1's forward.

Expected speedup: ~3× from parallelization, +1-1.5× from overlap = **~4×** total.

Less clean, keeps all code in Rust, no HIP kernel debugging. Fallback if the GPU kernel hits unexpected issues on RDNA3 vs CDNA3.

### Recommendation

Do **Option A** (GPU kernel). The reduce pattern is standard enough that it's not much more risky than the pipelined version, and it gives the cleanest architecture: calibration becomes a pure GPU pipeline with a single sync at the end.

## Implementation plan

### Phase 1 — CPU-GPU split (Option A kernel)

1. Write `kernels/src/triattn_accumulate.hip` with the reduce kernel.
2. Add binding in `crates/rdna-compute/src/kernels.rs` and dispatch in `crates/rdna-compute/src/dispatch.rs`.
3. Refactor `crates/engine/src/triattn.rs`:
   - Keep existing CPU `TriAttnCalibState` as fallback (`--cpu-calib` flag).
   - New `TriAttnCalibStateGpu` holds device accs buffer.
   - Install GPU tap when `--gpu-calib` flag is set (default in MI300X + RDNA3 paths; keep CPU for RDNA1 where atomic add on fp32 might not work).
4. Validate: run agentic-corpus cal with CPU vs GPU paths on same seed corpus, verify resulting `TriAttnCenters` differ by < 1e-4 per band.
5. Bench: time 1M-token cal on 7900 XTX CPU vs GPU path. Target: ≤40 min GPU.

### Phase 2 — Run the 4-sidecar matrix locally

Once Phase 1 is proven:

```bash
# Build agentic corpus locally (first time; ~1GB cached afterwards)
bash scripts/fetch_calibration_corpus.sh --recipe agentic_xl \
    .dflash-runs/agentic_xl_corpus.txt

# Calibrate each target (sequential — no need to parallelize on 1 GPU)
for model in qwen3.5-4b.mq4 qwen3.5-9b.mq4 qwen3.5-27b.mq4 qwen3.5-35b-a3b.mq4; do
    target/release/examples/triattn_validate \
        ~/.hipfire/models/${model} \
        --corpus .dflash-runs/agentic_xl_corpus.txt \
        --sidecar ~/.hipfire/models/${model}.triattn.agentic_xl.bin \
        --max-tokens 1000000 --chunk-len 1024
done
```

Expected wall: 2-3 hours total, $0. Produces 4 agentic-specialized sidecars ready to pair with z-lab drafts.

### Phase 3 — Deploy + bench

Pair each new sidecar with z-lab's matching draft. Bench τ on:
- The 3 prompts already staged in `.dflash-runs/prompts/`
- Real Hermes agent traces (sample 20 from `agentic_corpus.txt` across the corpus)
- Production hipfire-serve with Hermes agent test harness (task #23 is already in flight)

Baseline to beat: z-lab draft alone (no sidecar) AND z-lab draft + wikitext-calibrated sidecar (existing `.triattn.bin` files). If agentic sidecar gives >10% τ lift on Hermes-style prompts, we ship.

## Parallel work — 3.6-A3B draft (when 8× cluster spins)

Only target without a z-lab baseline. Training this IS a unique research contribution regardless of where it lands on τ. Reserve for the 8× cluster window per `docs/plans/8x-cluster-launch.md`. Not blocking sidecar work.

## DDTree — future inference-time τ recovery

DDTree (arXiv:2604.12989) multiplexes drafted paths per cycle with tree-verification. Can stack on top of any draft. Our 2026-04-14 test (memory: `project_ddtree_tuning.md`) showed +130% τ on hard prompts but wall-clock SLOWER due to 30ms/cycle overhead from CPU-side tree management.

After sidecars are shipped, DDTree is the next τ lever. Likely requires porting tree-verify logic to GPU kernels; probably ~3-5 days of work. Defer until sidecar story is closed.
