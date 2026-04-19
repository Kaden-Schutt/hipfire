#!/usr/bin/env python3
"""
dflash_diag_zlab_loss.py — Hermes cheap diagnostic #1 (2026-04-19).

Load z-lab's baseline Qwen3.5-4B-DFlash weights, run them through OUR
training forward pass (our data loader, our mask, our loss) at step 0.
Compute the loss on a batch from the same corpus.

Interpretation:
  loss < 3 (known-good draft produces good logits on our training task)
      → Our forward/mask/loss are CORRECT.
      → Bug is data distribution or config mis-instantiation (Hypotheses B, D).
  loss > 8 (known-good draft produces junk logits on our training task)
      → Our forward/mask/loss has a BUG (mask wrong, position_ids off, or
        config silently mis-instantiating the model).

Run on MI300X:
  python3 scripts/dflash_diag_zlab_loss.py \
      --target-repo Qwen/Qwen3.5-4B \
      --zlab-draft-repo z-lab/Qwen3.5-4B-DFlash \
      --corpus /root/calibration_corpus.txt \
      --num-batches 5

Needs ~20GB VRAM (4B target bf16 + 1B draft). ~5 min.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".dflash-reference"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dflash.model import DFlashDraftModel, extract_context_feature  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--zlab-draft-repo", default="z-lab/Qwen3.5-4B-DFlash",
                   help="HF repo of the z-lab reference draft (or local dir).")
    p.add_argument("--corpus", default="/root/calibration_corpus.txt")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--masked-blocks-per-seq", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--loss-gamma", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # ── Load target (frozen) ───────────────────────────────────────────
    print(f"[target] loading {args.target_repo}...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_repo)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_repo, torch_dtype=dtype, attn_implementation="eager",
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # ── Load z-lab draft ───────────────────────────────────────────────
    print(f"[draft] loading z-lab baseline {args.zlab_draft_repo}...")
    zlab_cfg = AutoConfig.from_pretrained(args.zlab_draft_repo, trust_remote_code=True)
    # z-lab's config already has dflash_config, block_size, num_target_layers.
    B_cfg = getattr(zlab_cfg, "block_size", args.block_size)
    if B_cfg != args.block_size:
        print(f"[draft] WARN z-lab block_size={B_cfg} != args.block_size={args.block_size} — using z-lab's value")
    B = B_cfg
    draft = DFlashDraftModel(zlab_cfg).to(device=device, dtype=dtype)
    # Load weights from safetensors via hf_hub_download + load_file, or use
    # from_pretrained's state_dict path. Simplest: use from_pretrained.
    try:
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        # Pick up first safetensors file in repo (their drafts are single-file).
        sf = hf_hub_download(repo_id=args.zlab_draft_repo, filename="model.safetensors")
        sd = load_file(sf)
        missing, unexpected = draft.load_state_dict(sd, strict=False)
        print(f"[draft] loaded weights; missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:5]:
            print(f"[draft]   missing keys (first 5): {missing[:5]}")
        if unexpected[:5]:
            print(f"[draft]   unexpected keys (first 5): {unexpected[:5]}")
    except Exception as e:
        print(f"[draft] load failed: {e}")
        return 2
    draft.eval()
    for p in draft.parameters():
        p.requires_grad_(False)
    print(f"[draft] {zlab_cfg.num_hidden_layers} layers, "
          f"hidden={zlab_cfg.hidden_size}, heads={zlab_cfg.num_attention_heads}, "
          f"head_dim={getattr(zlab_cfg, 'head_dim', zlab_cfg.hidden_size // zlab_cfg.num_attention_heads)}")

    # ── Load corpus (SAMPLED — we only need num_batches random slices) ─
    # Full 1GB corpus tokenize is 5+ min; cap at first ~2M chars then
    # tokenize only that. Plenty for 5 random L-length slices.
    print(f"[data] tokenizing head of {args.corpus}...", flush=True)
    head = Path(args.corpus).read_text()[:4_000_000]
    docs = [d.strip() for d in head.split("\n\n") if d.strip()][:200]
    bos = tokenizer.bos_token_id
    ids = []
    for d in docs:
        if bos is not None:
            ids.append(bos)
        ids.extend(tokenizer.encode(d, add_special_tokens=False))
    print(f"[data] {len(ids):,} tokens ({len(docs)} docs head-sample)", flush=True)

    mask_token_id = zlab_cfg.dflash_config["mask_token_id"]
    K = args.masked_blocks_per_seq
    L = args.seq_len

    if args.loss_gamma > 0:
        ks = torch.arange(1, B, device=device, dtype=torch.float32)
        pos_w = torch.exp(-(ks - 1) / args.loss_gamma)
        pos_w = pos_w / pos_w.sum()
    else:
        pos_w = torch.full((B - 1,), 1.0 / (B - 1), device=device)

    # ── Run num_batches forward passes, collect loss ───────────────────
    losses = []
    window_size = (L - B) // K
    print(f"\n[eval] running {args.num_batches} batches with z-lab weights through our forward...\n")
    with torch.no_grad():
        for b_idx in range(args.num_batches):
            start = random.randint(0, len(ids) - L - 1)
            clean = torch.tensor(ids[start:start + L], dtype=torch.long, device=device).unsqueeze(0)

            # Stratified random anchors
            anchors = torch.tensor([
                w * window_size + 1 + random.randint(0, max(0, window_size - B))
                for w in range(K)
            ], dtype=torch.long, device=device)

            # Target forward (clean)
            t_out = target(input_ids=clean, output_hidden_states=True, use_cache=False)
            tgt_ctx = extract_context_feature(t_out.hidden_states,
                                              zlab_cfg.dflash_config["target_layer_ids"])

            # Build concatenated noise blocks
            block_tok = torch.empty((1, K * B), dtype=torch.long, device=device)
            noise_positions = torch.empty((K * B,), dtype=torch.long, device=device)
            for k in range(K):
                s = int(anchors[k].item())
                blk = clean[:, s:s + B].clone()
                blk[:, 1:] = mask_token_id
                block_tok[:, k * B:(k + 1) * B] = blk
                noise_positions[k * B:(k + 1) * B] = torch.arange(s, s + B, device=device)
            noise_emb = target.model.embed_tokens(block_tok).to(dtype)

            ctx_positions = torch.arange(L, device=device)
            position_ids = torch.cat([ctx_positions, noise_positions]).unsqueeze(0)

            q_len = K * B
            q_block = torch.arange(q_len, device=device) // B
            q_anchor = anchors[q_block]
            ctx_idx = torch.arange(L, device=device).unsqueeze(0)
            ctx_visible = ctx_idx < q_anchor.unsqueeze(1)
            k_block = torch.arange(q_len, device=device) // B
            same_block = q_block.unsqueeze(1) == k_block.unsqueeze(0)
            mask_bool = torch.cat([ctx_visible, same_block], dim=1)
            attn_mask = torch.zeros_like(mask_bool, dtype=dtype)
            attn_mask.masked_fill_(~mask_bool, float("-inf"))
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            d_out = draft(
                noise_embedding=noise_emb,
                target_hidden=tgt_ctx,
                position_ids=position_ids,
                attention_mask=attn_mask,
                use_cache=False,
            )
            pred_idx = torch.tensor(
                [k * B + i for k in range(K) for i in range(1, B)],
                dtype=torch.long, device=device,
            )
            pred_hidden = d_out[:, pred_idx, :]
            logits = target.lm_head(pred_hidden)
            label_abs = torch.tensor(
                [int(anchors[k].item()) + i for k in range(K) for i in range(1, B)],
                dtype=torch.long, device=device,
            )
            labels = clean[:, label_abs]

            logits = logits.view(K, B - 1, -1)
            labels = labels.view(K, B - 1)

            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                reduction="none",
            ).view(K, B - 1)

            weighted = (ce * pos_w.unsqueeze(0)).sum(dim=1).mean()
            uniform = ce.mean()
            # Top-1 accuracy at position 1 (first predicted position, highest-weight)
            pred_top = logits.argmax(dim=-1)
            acc_pos1 = (pred_top[:, 0] == labels[:, 0]).float().mean()
            losses.append(weighted.item())

            print(f"  batch {b_idx+1}: weighted_loss={weighted.item():.3f}  "
                  f"uniform_loss={uniform.item():.3f}  "
                  f"pos1_top1_acc={acc_pos1.item():.3f}  "
                  f"anchors={anchors.tolist()}")

    mean_loss = sum(losses) / len(losses)
    print(f"\n{'='*60}")
    print(f"MEAN WEIGHTED LOSS (z-lab weights, our forward): {mean_loss:.3f}")
    print(f"{'='*60}")
    if mean_loss < 3.5:
        print("✓ Known-good draft produces low loss on our training task.")
        print("  → Our forward/mask/loss are CORRECT.")
        print("  → Bug is in the TRAINING DYNAMICS (data, config mis-instantiation, γ).")
    elif mean_loss < 6.0:
        print("~ Medium loss — inconclusive. Could be legit 'z-lab drafts are not")
        print("  optimal for agentic' data (their draft was wikitext-trained) or could be")
        print("  a real forward bug. Cross-check: run same diag on wikitext-sampled batches.")
    else:
        print("✗ Known-good draft produces HIGH loss on our training task.")
        print("  → Our forward/mask/loss has a BUG (Hypothesis A or D).")
        print("  → Investigate: mask construction, position_ids, config instantiation.")


if __name__ == "__main__":
    sys.exit(main())
