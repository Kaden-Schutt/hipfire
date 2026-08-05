# Runbook — a3b native F32 oracle + KLD reference on a rented MI300X

Produces a **hipfire-native** KLD reference for `qwen3.6-35b-a3b` at meaningful
scale. Native means the reference distribution comes from hipfire's own F32
forward, not llama.cpp — see `feedback_kldref_must_be_native_oracle` and the
0.29–0.36 nat cross-engine floor measured in F1.

**Precedent:** this is the F1/F2 path, re-run at a3b scale. Both were built and
validated on MI300X (gfx942, ROCm 7.0), 2026-06-03/04:

- `docs/plans/2026-06-02-hfqv2-implementation/experiments/self-sufficient-eval/F1-native-bf16-build.md`
- `.../F2-native-eval.md`

F1 added the `qt=2` / `qt=16` arms in `load_weight_tensor_raw`
(`qwen35.rs:1555`) and the `--format f32` passthrough (`main.rs:8408`). No new
code is required for this run.

---

## Why MI300X and not the local fleet

| oracle | size | fits? |
|---|---|---|
| F32, 1× MI300X (192 GB HBM3) | ~140 GB | **yes** |
| F32, 4× R9700 EP (128 GB) | 32.2 GB experts + 11.2 GB replicated dense = **43.4 GB/rank** | no (32 GB cards) |
| BF16 on disk | ~70 GB file, but `qwen35.rs:1555` widens to F32 at load → still 140 GB VRAM | no |

`DType::BF16` has exactly one reference in the runtime (a byte-size arm).
Resident-bf16 compute is F1's documented gap #2 and is **not** needed here.

---

## Preconditions

- **GPU:** 1× MI300X (gfx942), ROCm 7.x. Single card — no TP/EP.
- **Disk:** ≥ 400 GB (70 GB checkpoint + 140 GB oracle + headroom).
- **RAM:** ≥ 200 GB to use `HIPFIRE_NO_SPILL=1`. Otherwise the spill path needs
  ~2× the output on disk (280 GB).
- **No `HSA_OVERRIDE_GFX_VERSION`.** gfx942 is natively supported.

---

## Step 0 — bring up and build

```bash
rocminfo | grep -m1 gfx                      # expect gfx942
git clone <hipfire> && cd hipfire
git checkout a3b-lowbit-work                 # or master; the oracle path is on both

cargo build --release -p hipfire-quantize
cargo build --release --example build_kld_ref_native -p hipfire-runtime \
    --features arch-qwen35,deltanet
cargo build --release --example eval_hipfire -p hipfire-runtime \
    --features arch-qwen35,deltanet
```

`build_kld_ref_native` declares `required-features = ["arch-qwen35", "deltanet"]`
— omitting them silently skips the example.

## Step 1 — fetch the checkpoint

```bash
hf download Qwen/Qwen3.6-35B-A3B --local-dir /workspace/q36a3b-hf
```

Verify before encoding — the encoder keys off these:

```bash
python3 -c "import json;c=json.load(open('/workspace/q36a3b-hf/config.json'));\
t=c['text_config'];print(t['model_type'],t['num_hidden_layers'],t['num_experts'],\
t['hidden_size'],t['moe_intermediate_size'],c['text_config']['dtype'])"
# expect: qwen3_5_moe 40 256 2048 512 bfloat16
```

If the repo id differs, resolve it by those identifiers rather than by name —
the registry calls it `qwen3.6-35b-a3b` but the HF `model_type` is `qwen3_5_moe`.

## Step 2 — encode the F32 oracle (~140 GB, CPU only, no GPU needed)

```bash
HIPFIRE_NO_SPILL=1 \
./target/release/hipfire-quantize \
    --input  /workspace/q36a3b-hf \
    --output /workspace/q36a3b-f32-oracle.hfq \
    --format f32
```

- `--format f32` (aliases `f32-passthrough`, `oracle`) stores **every** tensor as
  `QuantType::F32` (qt=2), widening bf16→f32 losslessly. 3-D stacked MoE experts
  are split per-expert automatically (`[oracle split]` lines in the log) — the
  loader reads `...experts.{X}.{gate_up,down}_proj.weight` and would panic on the
  stacked form.
- `HIPFIRE_NO_SPILL=1` holds tensors in RAM and writes the output directly. If
  RAM is tight, drop it and set `HIPFIRE_SPILL_DIR=/workspace/spill` (needs 2×
  output on disk), or point it at `/dev/shm` if RAM is ample but disk is tight.

Expect ~140 GB out (35B × 4 B). F1's 9B oracle was 35.8 GB, same ratio.

## Step 3 — smoke ref (do this before the long run)

```bash
md5sum benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
cat    benchmarks/quality-baselines/slice/slice.md5      # must match

./target/release/examples/build_kld_ref_native \
    --model  /workspace/q36a3b-f32-oracle.hfq \
    --slice  benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
    --output /workspace/q36a3b-wt2-f32-native-SMOKE.kldref.bin \
    --top-k 256 --n-ctx 512 --tokenize-mode hipfire --max-chunks 8
```

`--tokenize-mode hipfire` is the llama-free path: hipfire's own BPE from the
oracle `.hfq` metadata. The tool internally forces `HIPFIRE_NORMALIZE_PROMPT=0`,
`HIPFIRE_GRAPH=0`, `HIPFIRE_KV_MODE=f32`.

Time this run — it sets the budget for Step 5. F1 measured ~35 tok/s on the 9B
dense F32 oracle; a3b is MoE (3B active) so expect the same order, but measure
rather than assume.

## Step 4 — soundness checks (gate the long run on these)

Scored tokens per chunk = `n_ctx - 1 - n_ctx/2` → 255 at n_ctx=512, 1023 at 2048.

1. **Oracle PPL.** `build_kld_ref_native` prints mean NLL / PPL over the scored
   window. The `.mq2` candidate measures PPL 6.2181, so the F32 oracle must come
   in **below** that. An oracle PPL above its own quantized candidate means the
   forward is broken — stop.
2. **Known-good candidate.** Score a high-precision SKU against the smoke ref:

   ```bash
   ./target/release/examples/eval_hipfire \
       --model ~/.hipfire/models/qwen3.6-35b-a3b.mq3p \
       --ref /workspace/q36a3b-wt2-f32-native-SMOKE.kldref.bin --kv-mode f32
   ```

   The reference flag is `--ref` (not `--kldref`); `--kv-mode f32` routes to
   `KvCache::new_gpu` — the true-FP32 KV path F1 added.

   F2's 9B Q8-vs-native-oracle was **0.0086 nats**. A high-precision a3b SKU
   should land in that neighbourhood — small, not ~0.2. If it comes out near the
   `.mq2` value, the oracle is not actually higher-precision than the candidate.
3. **Continuity.** Score `.mq2` and confirm it lands in the ballpark of the
   existing **0.238060**. Different token stream and count means it will not
   match exactly — an order-of-magnitude difference is the red flag, not a few
   percent.

## Step 5 — the real reference

Match the 27B MASTER's scale (2048 × 97 = 99,231 scored tokens):

```bash
./target/release/examples/build_kld_ref_native \
    --model  /workspace/q36a3b-f32-oracle.hfq \
    --slice  benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
    --output /workspace/q36a3b-wt2-f32-native.kldref.bin \
    --top-k 256 --n-ctx 2048 --tokenize-mode hipfire --max-chunks 97
```

That is **12×** the current a3b ref (8160 tokens) and moves to 2048-ctx, which
is more representative than 512. Scale `--max-chunks` to the measured rate from
Step 3; F2's note is that a full 1175-chunk canonical run is ~8.5 h at 35 tok/s.

F32 KV at n_ctx=2048 is negligible here (~82 MB — 10 full-attention layers of
40, 2 KV heads × 256 head_dim).

## Step 6 — retrieve and record

The artifact is small (~200 MB at 99k tokens; the current 8160-token ref is
16.8 MB). Pull only the `.bin` — leave the 140 GB oracle on the box.

```bash
scp <box>:/workspace/q36a3b-wt2-f32-native.kldref.bin ~/.hipfire/kldref/
```

Record alongside it, or the ref is unusable for cross-session comparison:

- oracle `.hfq` md5 + size, and the hipfire commit it was built from
- slice md5 (must equal `slice.md5`)
- `n_ctx`, `n_chunk`, `top_k`, scored-token count
- oracle mean NLL / PPL from Step 4.1
- `--tokenize-mode hipfire` (llama-free), ROCm version, gfx942

Then supersede `q36a3b-wt2-f32.kldref.bin` (8160 tokens) as the a3b baseline and
**re-measure `.mq2`** to establish the new baseline number. Every prior a3b KLD
in the investigation doc is tied to the old ref and does not transfer.

---

## Failure modes

- **`tensor not found: ...experts.0.gate_up_proj.weight`** — the 3-D expert split
  didn't fire. Confirm `[oracle split]` lines appear in the Step 2 log.
- **`_ => panic!` in `load_weight_tensor_raw`** — a qt with no arm. The oracle
  should be qt=2 throughout; check with
  `hfq_dump /workspace/q36a3b-f32-oracle.hfq | head -40`.
- **OOM during encode** — drop `HIPFIRE_NO_SPILL=1` and set `HIPFIRE_SPILL_DIR`.
- **OOM on load** — 140 GB against 192 GB is comfortable, but confirm nothing
  else holds the card (`rocm-smi`).
- **Oracle PPL ≈ candidate PPL** — the forward is not running at the precision
  you think. Do not proceed; the reference would be worthless.

## Cost

~$2–4/hr. Encode is CPU-bound (~1 h), ref generation is the GPU cost measured in
Step 3. Budget one working session; the artifact is reusable indefinitely.
