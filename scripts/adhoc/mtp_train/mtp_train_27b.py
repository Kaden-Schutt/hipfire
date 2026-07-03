"""
FastMTP fine-tune for Qwen3.6-27B MTP block on 1× MI300X.

Differences from 0.8B smoke:
- 27B trunk (much bigger forward time per step)
- tie_word_embeddings: False (use get_output_embeddings, not embed.weight)
- Larger MTP block (424.7M vs 20.5M params)
- More memory pressure (54 GB trunk + ~10 GB optimizer + activations)
- Reduced eval frequency + shorter eval seq to keep wall time bounded
"""
import sys, os, glob, time, json, random, math
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, "/tmp")
from mtp_module import (
    Qwen35MtpBlock, load_mtp_from_safetensors,
    hf_cache_dir, get_lm_head_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-27B"
PROMPT_DIR = "/workspace/hipfire/benchmarks/prompts"
MAX_LEN_TRAIN = 512
MAX_LEN_EVAL = 384
SEED = 42
N_TRAIN_STEPS = 500
LR = 5e-5
WARMUP_STEPS = 50
EVAL_EVERY = 50
N_WIKI = 500
HOLDOUT = {
    "lru_cache_pep8_strict.txt",
    "humaneval_0_has_close_elements.txt",
    "agentic_user_multistep.txt",
    "trains-meet.txt",
    "tool_call_system.txt",
}
CHECKPOINT = "/tmp/mtp_3p6_27b_trained.pt"

def load_prompts(prompt_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(prompt_dir, "*.txt"))):
        with open(p) as fh: t = fh.read()
        if len(t) >= 50:
            out.append((os.path.basename(p), t))
    return out

def load_wiki(n):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out, buf = [], ""
    for row in ds:
        buf += row["text"]
        if len(buf) >= 1500:
            out.append((f"wiki_{len(out):04d}", buf)); buf = ""
            if len(out) >= n: break
    return out

@torch.no_grad()
def eval_acceptance(model, mtp, embed, lm_head, tok, eval_prompts, max_len=MAX_LEN_EVAL):
    totals = {"agree": 0, "mtp_correct": 0, "total": 0, "tokens": 0}
    rows = []
    for name, txt in eval_prompts:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).input_ids.cuda()
        T = ids.shape[1]
        if T < 4: continue
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hid = out.hidden_states[-1]
        trunk_argmax = out.logits.argmax(-1)
        prev_emb = embed(ids)
        mtp_hid = mtp(prev_emb, trunk_hidden=hid)
        mtp_logits = lm_head(mtp_hid)
        mtp_argmax = mtp_logits.argmax(-1)
        labels = ids[0, 1:]
        a = int((mtp_argmax[0, :-1] == trunk_argmax[0, :-1]).sum())
        m = int((mtp_argmax[0, :-1] == labels).sum())
        totals["agree"] += a; totals["mtp_correct"] += m
        totals["total"] += int(labels.numel()); totals["tokens"] += T
        rows.append({"prompt": name, "T": T,
                     "agree": a/labels.numel(), "mtp": m/labels.numel()})
    if totals["total"] == 0: return 0.0, 0.0, rows
    return totals["agree"]/totals["total"], totals["mtp_correct"]/totals["total"], rows

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()
    print(f"=== FastMTP fine-tune on {MODEL_ID} (1× MI300X) ===\n")

    print(f"Load trunk {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    embed = model.get_input_embeddings()
    lm_head = get_lm_head_module(model)
    print(f"  trunk loaded in {time.time()-t0:.1f}s   hidden={text_cfg.hidden_size}  vocab={embed.weight.shape[0]}")
    print(f"  tied embeddings: {text_cfg.tie_word_embeddings if hasattr(text_cfg, 'tie_word_embeddings') else '?'}")
    print(f"  lm_head module: {type(lm_head).__name__}  weight shape: {tuple(lm_head.weight.shape)}")

    mem_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU mem after trunk load: {mem_after_load:.1f}GB\n")

    print("Load MTP weights from safetensors...")
    mtp_sd = load_mtp_from_safetensors(hf_cache_dir(MODEL_ID))
    print(f"  found {len(mtp_sd)} mtp.* tensors")

    print("Build Qwen35MtpBlock...")
    mtp = Qwen35MtpBlock(text_cfg).to(device="cuda:0", dtype=torch.bfloat16)
    missing, unexpected = mtp.load_pretrained_(mtp_sd)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        for k in missing[:5]: print(f"    missing: {k}")

    # Freeze trunk + lm_head, train MTP only
    for p in model.parameters(): p.requires_grad = False
    n_train = sum(p.numel() for p in mtp.parameters())
    print(f"  MTP trainable: {n_train:,} ({n_train/1e6:.1f}M)\n")

    print("Load prompts...")
    all_p = load_prompts(PROMPT_DIR)
    train_p = [p for p in all_p if p[0] not in HOLDOUT]
    eval_p = [p for p in all_p if p[0] in HOLDOUT]
    print(f"  hipfire train: {len(train_p)}, eval (held out): {len(eval_p)}")
    print(f"  loading {N_WIKI} wikitext samples...")
    wiki = load_wiki(N_WIKI)
    pool = train_p + wiki
    print(f"  training pool: {len(pool)}\n")

    optimizer = torch.optim.AdamW(mtp.parameters(), lr=LR, weight_decay=0.01)
    def lr_lambda(step):
        if step < WARMUP_STEPS: return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, N_TRAIN_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"BASELINE eval (pretrained MTP only)...")
    a0, m0, base_rows = eval_acceptance(model, mtp, embed, lm_head, tok, eval_p)
    print(f"  baseline: agree={100*a0:.2f}%  mtp_acc={100*m0:.2f}%")
    for r in base_rows:
        print(f"    {r['prompt'][:40]:<40} T={r['T']:>4}  agree={100*r['agree']:.1f}  mtp={100*r['mtp']:.1f}")
    print()

    mem_peak_baseline = torch.cuda.max_memory_allocated() / 1e9
    print(f"  mem peak after baseline: {mem_peak_baseline:.1f}GB\n")

    print(f"TRAIN {N_TRAIN_STEPS} steps...")
    print(f"{'step':<5} {'lr':<10} {'loss':<8} {'tok':<5} {'ms':<7} {'agree':<11}")
    losses, eval_hist = [], [(0, a0, m0)]
    train_t0 = time.time()
    for step in range(N_TRAIN_STEPS):
        _, prompt = random.choice(pool)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN_TRAIN).input_ids.cuda()
        T = ids.shape[1]
        if T < 4: continue

        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hid = out.hidden_states[-1]
            trunk_argmax = out.logits.argmax(-1)

        t_step = time.time()
        prev_emb = embed(ids)
        mtp_hid = mtp(prev_emb, trunk_hidden=hid)
        mtp_logits = lm_head(mtp_hid)
        loss = F.cross_entropy(
            mtp_logits[0, :-1, :].float(),
            trunk_argmax[0, :-1].long(),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mtp.parameters(), max_norm=1.0)
        optimizer.step()
        sched.step()
        torch.cuda.synchronize()
        dt = (time.time() - t_step) * 1000
        losses.append(loss.item())

        if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS - 1:
            ae, me, _ = eval_acceptance(model, mtp, embed, lm_head, tok, eval_p)
            eval_hist.append((step + 1, ae, me))
            print(f"{step:<5} {sched.get_last_lr()[0]:<10.2e} {loss.item():<8.4f} {T:<5} {dt:<7.1f} {100*ae:.2f} ({100*(ae-a0):+.2f})")
        elif step % 25 == 0:
            print(f"{step:<5} {sched.get_last_lr()[0]:<10.2e} {loss.item():<8.4f} {T:<5} {dt:<7.1f}")

    print()
    af, mf, final_rows = eval_acceptance(model, mtp, embed, lm_head, tok, eval_p)
    print(f"FINAL eval (post-train):")
    print(f"  baseline:  agree={100*a0:.2f}%  mtp_acc={100*m0:.2f}%")
    print(f"  final:     agree={100*af:.2f}%  mtp_acc={100*mf:.2f}%")
    print(f"  delta:     agree={100*(af-a0):+.2f}pp   mtp_acc={100*(mf-m0):+.2f}pp")
    print()
    print("  Per-prompt breakdown:")
    print(f"  {'prompt':<40} {'before':<8} {'after':<8} {'delta':<8}")
    for b, f in zip(base_rows, final_rows):
        d = 100 * (f["agree"] - b["agree"])
        print(f"  {b['prompt'][:40]:<40} {100*b['agree']:<8.2f} {100*f['agree']:<8.2f} {d:+.2f}")

    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nTraining wall: {time.time()-train_t0:.1f}s ({(time.time()-train_t0)/N_TRAIN_STEPS*1000:.1f} ms/step avg)")
    print(f"Total wall:    {time.time()-t0:.1f}s")
    print(f"GPU mem peak:  {mem_peak:.1f}GB")

    torch.save({
        "mtp_state_dict": mtp.state_dict(),
        "model_id": MODEL_ID,
        "baseline_agree": a0, "final_agree": af, "delta_agree": af - a0,
        "losses": losses, "eval_history": eval_hist,
    }, CHECKPOINT)
    print(f"Checkpoint: {CHECKPOINT}")
    json.dump({
        "model_id": MODEL_ID,
        "baseline_agree": a0, "final_agree": af,
        "delta_agree_pct_pts": 100*(af-a0),
        "baseline_mtp_acc": m0, "final_mtp_acc": mf,
        "training_steps": N_TRAIN_STEPS,
        "training_wall_seconds": time.time() - train_t0,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean_last10": sum(losses[-10:]) / 10,
        "eval_history": eval_hist,
        "mem_peak_gb": mem_peak,
        "per_prompt_baseline": base_rows,
        "per_prompt_final": final_rows,
    }, open("/tmp/mtp_3p6_27b_result.json", "w"), indent=2)

if __name__ == "__main__":
    main()
