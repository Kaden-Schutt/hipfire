#!/usr/bin/env python3
"""Krea2 diffusers REFERENCE dump for hipfire parity validation.

Runs the real Krea2 pipeline (needs diffusers >= 0.39.0.dev with
`Krea2Pipeline`/`Krea2Transformer2DModel`) on a fixed prompt+seed and dumps the
intermediate tensors hipfire also dumps (via HIPFIRE_DIFFUSION_DUMP_DIR):

  encoder_layer_<N>.npy   selected Qwen3-VL hidden states  [1, seq, text_hidden]
  text_fusion_out.npy     fused conditioning               [1, seq, text_hidden]
  dit_out.npy             transformer prediction           [1, C, H, W]  (optional)
  vae_out.npy             decoded pixels                   [1, 3, H, W]  (optional)

Then `diff.py ref_dir hipfire_dir` compares them. Because the diffusers API for
the private text_fusion / hidden-state extraction can shift between dev builds,
the hooks below are written defensively with clear TODO markers — adjust the two
marked spots to match the installed Krea2 modeling code if needed.

Usage:
  python reference.py --model /path/to/Krea-2-Turbo --out ref \
      --prompt "a red cube" --width 64 --height 64 --steps 1 --seed 0
"""
import argparse, os
import numpy as np
import torch


def save_npy(out_dir, name, tensor):
    os.makedirs(out_dir, exist_ok=True)
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    print(f"  saved {name}: {tuple(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Krea-2-Turbo diffusers dir")
    ap.add_argument("--out", required=True, help="output dump dir")
    ap.add_argument("--prompt", default="a red cube")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    from diffusers import DiffusionPipeline  # noqa: E402

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(args.device)

    select = getattr(pipe, "text_encoder_select_layers", None)
    if select is None:
        # stored on the pipeline config (model_index.json)
        select = pipe.config.get("text_encoder_select_layers", list(range(2, 36, 3)))
    print("select_layers:", select)

    tok = pipe.tokenizer(args.prompt, return_tensors="pt").to(args.device)
    print("token ids:", tok.input_ids[0].tolist())

    # --- encoder selected hidden states -------------------------------------
    # TODO(adjust): match how Krea2Pipeline runs its text encoder. The common
    # shape is: run the language_model with output_hidden_states=True and index
    # `hidden_states[layer]`. hidden_states has len(n_layers)+1 (embeddings first),
    # so `hidden_states[layer]` is the output *after* `layer` decoder layers.
    # hipfire dumps encoder_layer_<layer> = output after that many layers; if the
    # diff is off-by-one, shift the index here (or in hipfire) and note it.
    with torch.no_grad():
        te = pipe.text_encoder
        lm = getattr(te, "language_model", te)
        enc = lm(tok.input_ids, output_hidden_states=True)
        hs = enc.hidden_states  # tuple, [embeddings, layer0_out, layer1_out, ...]
        layers = []
        for layer in select:
            h = hs[layer]  # [1, seq, hidden]
            save_npy(args.out, f"encoder_layer_{layer}", h)
            layers.append(h)

    # --- text_fusion --------------------------------------------------------
    # TODO(adjust): call the transformer's text_fusion module on the stacked
    # selected hidden states. Expected input [B, seq, num_layers, dim]; output
    # [B, seq, dim]. If the module signature differs in the installed build,
    # adapt this call. hipfire dumps `text_fusion_out`.
    try:
        stacked = torch.stack(layers, dim=2)  # [1, seq, L, dim]
        fused = pipe.transformer.text_fusion(stacked)
        if isinstance(fused, (tuple, list)):
            fused = fused[0]
        save_npy(args.out, "text_fusion_out", fused)
    except Exception as e:  # noqa: BLE001
        print(f"  (text_fusion hook needs adjustment for this build: {e})")

    print("reference dump complete ->", args.out)


if __name__ == "__main__":
    main()
