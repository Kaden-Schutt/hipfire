# scripts/arch-intake

Per-architecture intake harness: PyTorch is the oracle, hipfire is the
system-under-test, per-op NRMSE is the verdict. Used during a new model
arch port to bisect the first divergent op.

## Methodology (canonical)

1. **Reference dump (PyTorch side)** — load the HF model in bf16 (or
   fp16 to match what hipfire's f16 path will see), register
   `register_forward_hook` on every meaningful module (pre/post-norms,
   q/k/v projections, attn block, MLP gate/up/down, decoder block i/o),
   run one forward on a fixed canonical prompt, dump each captured
   tensor as a single-tensor safetensors file under
   `/tmp/<arch>-port/refs/<prompt-hash>/layer_NN/<step>.<side>.safetensors`.
   Write a `manifest.json` with shapes, dtypes, `input_ids`, prompt md5.

2. **(Optional) Dequant-patched reference** — once a `.mq4` / `.mg4`
   exists for the model, patch the HF model's weights with hipfire's
   numpy-dequantized values BEFORE running the reference forward. This
   removes the "different weights" confound; both runtimes hold
   byte-identical dequantized values.

3. **hipfire dump (Rust side)** — env-gated hooks in
   `forward_scratch`-style functions: `HIPFIRE_DUMP_DIR=/path` activates
   `dump_layer_tensor` calls, `HIPFIRE_DUMP_POS=NN` selects the position
   subdir. Each call: `hipMemcpyDtoH` the GpuTensor, write a 1-D f32
   safetensors file matching the ref dump's layout. Zero overhead when
   env unset.

4. **Diff** — `diff_layer_dumps.py` walks every `(pos, layer, side)`
   triple, computes `NRMSE = ||a - b|| / ||b||`, reports first
   divergence above the bf16 ULP threshold (5e-3) in execution order.

## Why first-divergence is the right rule

- pos 0 layer N input matches but output diverges → bug in layer N
  compute (no KV history).
- pos 0 layer N matches but pos 1 layer N output diverges → bug in
  KV-cache write/read between positions.
- All layer outputs match but final logits don't → bug in final norm /
  lm_head / softcap.

This is how the Gemma 4 V-norm bug got pinned (per CLAUDE.md context).

## Per-arch scripts

| Script | Status |
|---|---|
| `dump_zaya_reference.py` | Phase 1 scaffold; bf16 reference only, no dequant-patching (no ZAYA1 HFQ exists yet). Requires Zyphra/transformers @ zaya1 branch + trust_remote_code. |
| `prompts/zaya_canonical.txt` | Fixed canonical prompt (md5 captured in manifest) for repeatable comparisons. |

## Running the ZAYA1 reference dump

```
# On hiptrx, lane 2 (gpu 2 belongs to zaya port; 0,1 are gemma e-series)
ssh hiptrx
cd <hipfire repo>
git checkout feat/zaya1-port-intake

# One-time: install Zyphra's transformers fork
pip install --user "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1"

# Run dump
HIP_VISIBLE_DEVICES=2 python scripts/arch-intake/dump_zaya_reference.py \
    --model Zyphra/ZAYA1-8B \
    --prompt scripts/arch-intake/prompts/zaya_canonical.txt \
    --output /tmp/zaya-port/refs/canonical/
```
