# Daemon EP (expert-parallel) multi-GPU serving — implementation spec

Goal: make big MoE models (DeepSeek-V4-Flash 86GB, MiniMax, qwen3.5-A3B) **servable
multi-GPU through the daemon**, not just benchmarkable via standalone harnesses
(`ep_deepseek4`, `ep_decode_parity`). Wire the existing EP substrate
(`forward_ep` + `load_weights_sharded`) into the daemon serve loop for all 3
EP-capable arches.

Branch: `feat/daemon-ep` (off integration tip 12d2fc57, has PR #432).
Tracked: task #26.

## What already exists (do NOT rebuild)
- **EP substrate**: `forward_ep(gpus, weights_per_rank, cfg, state_per_rank, partials, tok, pos)`
  for ds4 (`hipfire-arch-deepseek4/src/forward.rs:2149`), qwen35 (`qwen35.rs`),
  minimax (`forward.rs`). `load_weights_sharded` per arch. `mtp_forward_ep` (ds4 spec).
- **Gpus orchestrator**: `multi_gpu::Gpus::init_tp(tp, n_layers)`, `enable_peer_all()`,
  `ep::ensure_rank_streams(&mut gpus)`. `Gpus::init_uniform/init_layers` (PP).
- **Daemon PP path**: `LoadedModel.pp_gpus: Option<Gpus>` + `pp_scratch_set` +
  `generate_multi` (daemon.rs:6632) + `*_multi` forward. This is the TEMPLATE — EP
  mirrors it but uses `forward_ep` (replicated-attn + sharded-experts + all-reduce)
  instead of PP layer-split.
- Reference invocation (proven on hiptrx 4×gfx1201): `ep_deepseek4 --tp 4` +
  chat template + `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=40`. See
  [[project_pr432_ds4_landed_2026_06_08]] / [[project_ship6_ep_substrate_validated_2026_06_07]].

## Design (mirror the PP wiring)

### 1. Config / CLI
- New `--tp N` (alias `--ep N`) and/or `HIPFIRE_EP_TP=N`. N>1 → EP mode.
- Mutually exclusive with PP (`--pp`). EP refuses DFlash/CASK/PFlash/VL/spec-decode
  for v1 (AR only) — gate exactly like the PP path refuses them.
- **Sampling IS supported** (temp/top-p/repeat/presence/frequency). EP/TP shard the
  FORWARD only; the sampler runs on the gathered rank-0 logits, orthogonal to the
  parallelism → reuse the daemon's existing sampler unchanged. Greedy is NOT
  required; it's only the deterministic case used for EP≡single-GPU byte-parity
  validation (fixed seed + HIPFIRE_DETERMINISTIC=1; per-rank stochastic Q8-state
  rounding otherwise makes logits differ by a hair run-to-run, harmless for
  sampling). "AR only" = no spec-decode, NOT greedy-only.
- `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` passthrough (ds4 ~30GB/rank on 32GB cards).

### 2. `LoadedModel` struct — ONE field `ep: Option<EpState>` (not 9 per-arch fields)
Avoid bloating LoadedModel + every load-arm constructor. Single field:
```rust
struct EpState { gpus: Gpus, inner: EpArch }
enum EpArch {
    Ds4   { weights: Vec<deepseek4::DeepseekV4Weights>, state: Vec<deepseek4::DeepseekV4State>, partials: Vec<GpuTensor> },
    Qwen35{ weights: Vec<qwen35::Qwen35Weights>,       state: Vec<…Qwen35 EP state…>,          partials: Vec<GpuTensor> },
    Minimax{ weights: Vec<minimax::…Weights>,          state: Vec<minimax::…State>,            partials: Vec<GpuTensor> },
}
```
- `LoadedModel.ep: Option<EpState>` — Some only when `ep>1`. Each load arm adds one
  line `ep: None,`. When `ep.is_some()`, the single-GPU arch fields stay None.
- (the `partials` Vec is the per-rank [hidden] F32 all-reduce buffer.)

### 3. Load path (per arch, EP arm)
When `ep_tp > 1`, in each arch's load arm:
- `let mut gpus = Gpus::init_tp(ep_tp, cfg.n_layers)?;`
- per rank r: `bind_thread`; `hfq = reopen`; `load_weights_sharded(&mut hfq, &cfg, &mut gpus.devices[r], &shard, r)`.
- alloc per-rank `State::new(&cfg)` + `partials = device.zeros([hidden], F32)`.
- `gpus.enable_peer_all()`; `ep::ensure_rank_streams(&mut gpus)`.
- store in the EP fields; set `ep_gpus = Some(gpus)`.
(This is exactly `ep_deepseek4.rs` main() lines ~83-105, lifted into the daemon.)

### 4. `generate_ep` (new fn, parallels generate_multi)
- Greedy/temp AR streaming via `forward_ep` per token:
  - prefill: `for (pos,&t) in prompt_ids { forward_ep(gpus, w_pr, cfg, s_pr, partials, t, pos) }`.
  - decode loop (`while generated < max_tokens`): `forward_ep(.., next, pos)`;
    download logits from rank-0 state (`s_pr[0].logits`); sample; stream; EOS/stop checks.
- Reuse the existing streaming machinery (EosFilter, StreamParser, LoopGuard,
  stop_at, think-cap) — same as generate_multi's tail. Factor the shared
  streaming tail if cheap; else copy the proven block.
- Conversation reset / context-full handling: zero per-rank states (mirror
  generate_multi's pp reset at 6667-6693, but per-rank).

### 5. Dispatch routing (serve handler)
- Where the handler picks generate vs generate_multi vs generate_dflash: add
  `if m.ep_gpus.is_some() { generate_ep(...) }` FIRST (highest precedence;
  EP refuses the others).

### 6. Feature-gate (v1)
Refuse on EP path (clear error, like PP): DFlash/spec-decode, CASK, PFlash,
VL, prefill-checkpoints. Full sampler (temp/top-p/penalties) supported — NOT
greedy-only. (MTP-EP spec-decode = later, folds in #26's original scope via
mtp_forward_ep.)

## Increments (build + verify each)
1. **ds4** (motivating, proven harness): config + struct + ds4 load arm +
   generate_ep + route. VERIFY: daemon serve ds4 on hiptrx `--tp 4`, coherent
   (the recall gate now works on gfx1201!). ← establishes the pattern.
2. **qwen35** (A3B EP): replicate load arm + generate_ep arch branch. VERIFY
   qwen3.5-A3B served on hiptrx `--tp 2`/`--tp 4`.
3. **minimax**: replicate. VERIFY on hiptrx `--tp 4` (minimax needs EP to fit
   across the 32GB cards).

## Test / gates — **EP IS hiptrx-ONLY**
EP needs ≥2 HOMOGENEOUS GPUs; the ONLY such box is **hiptrx (4× gfx1201/RDNA4)**.
- k9lin = single gfx1100 → no EP at all.
- hipx = gfx1010 5700XT (RDNA1) + gfx1151 Strix Halo (RDNA3.5) → heterogeneous,
  EP expert-offload across them is FRUITLESS (mismatched ISA, RDNA1 tiny/slow).
- ⇒ the #397 cross-arch mandate does NOT extend to EP (no RDNA3/3.5 multi-
  homogeneous-GPU box exists). daemon-EP validation = hiptrx/RDNA4 only.
- That's sufficient: `forward_ep` is already byte-validated on hiptrx
  ([[project_ship6_ep_substrate_validated_2026_06_07]]); the daemon wiring is
  arch-general (single-GPU paths untouched → compiles+runs unchanged on all
  arches). So this is a hiptrx serve-routing smoke test, NOT a re-validation of
  EP correctness.
- Per arch on hiptrx: daemon `--tp N` serve → coherent output; determinism =
  EP byte-parity vs the `ep_deepseek4` FNV (same prompt, fixed seed).
- No single-GPU regression: `ep` None when `tp<=1` → existing paths byte-identical
  (verify on k9lin/gfx1100 + the standard coherence gate).

## Gotchas
- `init_tp` uniform-VRAM preflight over-strict on near-full cards →
  HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB passthrough + a clearer daemon error.
- ds4 needs the chat template; the daemon already applies per-arch templates in
  the prompt-frame path, so generate_ep gets formatted prompts for free (unlike
  the raw `ep_deepseek4` harness).
- RCCL all-reduce count MUST == allocated buffer (n<buffer page-faults tp≥2;
  see [[project_ship6_ep_substrate_validated_2026_06_07]]).
