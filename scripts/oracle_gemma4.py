# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.
#!/usr/bin/env python3
"""Gemma-4 long-context HF reference oracle.

Token IDs are the contract: both HF and hipfire read the SAME ids file, so
tokenizer differences never enter. Dumps, for the LAST position: final logits
top-k (post-softcap) + per-layer last-position hidden (first8 + L2 norm, for
cosine). Self-gate: run on a short known case and confirm hipfire agrees before
trusting the long case.

Usage:
  # make a >1024-token ids file from a text file (BOS-prefixed, raw — no chat frame)
  python3 scripts/oracle_gemma4.py --make-ids text.txt --ids-out ids.txt
  # run HF reference for an ids file
  python3 scripts/oracle_gemma4.py --ids-file ids.txt --out hf_ref.json
  # or an inline short case (self-gate)
  python3 scripts/oracle_gemma4.py --ids 2,9259 --out hf_short.json
Model selection is --model, then GEMMA4_MODEL, then the historical default.
"""
import argparse, json, os, sys

MODEL = "/local/models/google/gemma-4-12B-it"


def load_ids(args):
    if args.ids:
        return [int(x) for x in args.ids.split(",") if x.strip() != ""]
    if args.ids_file:
        txt = open(args.ids_file).read().replace(",", " ").split()
        return [int(x) for x in txt]
    raise SystemExit("need --ids or --ids-file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model path (overrides GEMMA4_MODEL)")
    ap.add_argument("--ids")
    ap.add_argument("--ids-file")
    ap.add_argument("--out")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--make-ids")
    ap.add_argument("--ids-out")
    ap.add_argument("--max-ids", type=int, default=1200)
    args = ap.parse_args()
    model_path = args.model or os.environ.get("GEMMA4_MODEL") or MODEL

    if args.make_ids:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        text = open(args.make_ids).read()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids = [2] + ids[: args.max_ids - 1]  # BOS + body
        with open(args.ids_out, "w") as f:
            f.write(" ".join(str(i) for i in ids))
        print(f"wrote {len(ids)} ids to {args.ids_out}")
        return

    ids = load_ids(args)
    print(f"Loading model (f32 CPU)…  ids={len(ids)} tokens", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map={"": "cpu"}
    ).eval()
    input_ids = torch.tensor([ids], device="cpu")
    print("forward…", file=sys.stderr)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    logits = out.logits[0, -1].float()               # [vocab], post-softcap
    hs = out.hidden_states                            # tuple len = n_layers+1
    topv, topi = torch.topk(logits, args.topk)
    layers = []
    for li, h in enumerate(hs):
        v = h[0, -1].float()                          # last-position hidden
        layers.append({
            "layer": li,                              # 0 = embeddings
            "first8": [round(x, 5) for x in v[:8].tolist()],
            "norm": round(v.norm().item(), 5),
        })
    rec = {
        "n_ids": len(ids),
        "logits_topk": [[int(i), round(float(x), 4)] for i, x in zip(topi.tolist(), topv.tolist())],
        "logit_argmax": int(topi[0]),
        "layers": layers,
    }
    if args.out:
        json.dump(rec, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    print("argmax:", rec["logit_argmax"], "top5:", rec["logits_topk"][:5])


if __name__ == "__main__":
    main()
