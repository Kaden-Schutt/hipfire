# Krea2 numerical-parity harness (run on `halo`)

Validates hipfire's Krea-2-Turbo implementation against the diffusers reference,
tensor-by-tensor, and unblocks **full calibrated oq4++**. Run this on **halo**
(128 GB RAM) because the reference needs the full-precision model resident and
diffusers `>= 0.39.0.dev` (which carries `Krea2Pipeline`).

hipfire already validated the major *conventions* by source inspection (encoder
RoPE half-split, DiT RoPE interleaved, causal attention, QK-norm, VAE causal
conv / RMSNorm / latents) — see `memory/krea2-dit-diffusion.md`. This harness is
the *numerical* confirmation + catches the few Krea2-specific orderings that
aren't in an installable lib (adaLN chunk order, `select_layers` indexing,
text_fusion ordering).

## What it compares

hipfire dumps intermediates when `HIPFIRE_DIFFUSION_DUMP_DIR` is set (in
`Krea2TextConditioner::conditioning_from_token_ids`):

| file | shape | meaning |
|---|---|---|
| `encoder_layer_<N>.npy` | `[1, seq, 2560]` | Qwen3-VL hidden state after layer N (for N in `text_encoder_select_layers`) |
| `text_fusion_out.npy` | `[1, seq, 2560]` | fused conditioning fed to the DiT `txt_in` |

`reference.py` dumps the same from diffusers; `diff.py` reports per-tensor
max/mean abs diff + cosine similarity.

## Steps (on halo)

```bash
# 0. env: diffusers with Krea2 + a torch that imports cleanly
python -m venv ~/krea2-parity && . ~/krea2-parity/bin/activate
pip install "diffusers>=0.39.0.dev0" transformers torch numpy   # or the Krea2 branch
MODEL=/srv/huggingface/usb256GB/Krea-2-Turbo                     # diffusers snapshot
HFQ=/path/to/krea2-full.hfq                                      # or krea2-oq4pp.hfq

# 1. hipfire dump (release build). Same prompt/size for both sides.
cargo build --release -p hipfire-cli
rm -rf hipfire && mkdir hipfire
HIPFIRE_DIFFUSION_CPU_REFERENCE=1 HIPFIRE_DIFFUSION_DUMP_DIR=$PWD/hipfire \
  ./target/release/hipfire diffusion txt2img --model "$HFQ" \
    --prompt "a red cube" --steps 1 --width 64 --height 64 --cfg-scale 1.0 \
    --output /tmp/hipfire_krea2.png

# 2. diffusers reference dump (same prompt/size)
python scripts/krea2-parity/reference.py --model "$MODEL" --out $PWD/ref \
  --prompt "a red cube" --width 64 --height 64 --steps 1 --seed 0

# 3. diff
python scripts/krea2-parity/diff.py $PWD/ref $PWD/hipfire --rtol 1e-2 --atol 1e-3
```

## Interpreting the diff

- **All within tolerance** → the encoder + text_fusion match; the conditioning
  half is numerically correct. Extend the dumps (DiT out, VAE out) to finish the
  chain — add `dump_debug_tensor` calls in the denoiser `forward_krea` (predicted
  latent) and the VAE decode, and mirror them in `reference.py`.
- **`encoder_layer_<N>` mismatch, cos ≈ high but not 1** → a numeric detail
  (RoPE theta, norm eps). cos ≈ 0/negative → a real convention bug (RoPE
  half-split vs interleaved, causal masking).
- **`encoder_layer_<N>` shapes/values off by a whole layer** → `select_layers`
  indexing is off by one. `hidden_states[i]` in HF is *after* `i` layers
  (embeddings at 0); hipfire captures after `index+1` layers. If they disagree,
  shift the index in `reference.py` (or hipfire's `encode`) and note it in memory.
- **encoder matches but `text_fusion_out` mismatches** → the text_fusion
  ordering (layerwise → projector[12→1] → refiner) or projector orientation.

## Bonus on halo: full calibrated oq4++

With the model resident (128 GB), the full `[K,K]` Hessians fit, so:

```bash
HIPFIRE_DIFFUSION_CPU_REFERENCE=1 ./target/release/hipfire diffusion calibrate \
  "$HFQ_BF16" --output krea2.calib.hfq --steps 4 --width 256 --height 256 --hessian-max-k 16384
./target/release/hipfire diffusion quantize --format oq4++ --calib krea2.calib.hfq \
  --output krea2-oq4pp-ldlq.hfq "$HFQ_BF16"
```

Check the quantize summary's `ldlq_tensors` count — nonzero confirms LDLQ
Hessian error-feedback was applied (true oq4++, not the data-free fallback).
Note: verify `ResidentWeight::decode`'s calibration capture fires for all
converted linears (on nix1 only 1 Hessian was captured at a low `hessian-max-k`).
