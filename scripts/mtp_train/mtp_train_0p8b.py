"""
FastMTP-style fine-tune of Qwen3.5-0.8B MTP block.

- Freeze trunk + lm_head (tied embed)
- Train ONLY mtp.* params (~20.5M)
- Loss: CE(mtp_logits[t], trunk_argmax[t+1]) — single-step self-distill
- Data: 14 hipfire benchmark prompts + N wikitext samples for diversity
- Eval: held-out subset of hipfire prompts (LRU, HumanEval, agentic mix)
- Goal: post-train MTP/trunk argmax agreement > baseline 68.28% by ≥ 5%
"""
import sys, os, glob, time, json, random
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, "/tmp")
from mtp_module import Qwen35MtpBlock, load_mtp_from_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT_DIR = "/workspace/hipfire/benchmarks/prompts"
MAX_LEN = 768
SEED = 42
N_TRAIN_STEPS = 500
LR = 5e-5
EVAL_EVERY = 50
N_WIKI_SAMPLES = 500          # wiki samples added to training pool
HOLDOUT_PROMPTS = {            # NOT in training pool
    "lru_cache_pep8_strict.txt",
    "humaneval_0_has_close_elements.txt",
    "agentic_user_multistep.txt",
    "trains-meet.txt",
    "tool_call_system.txt",
}
CHECKPOINT = "/tmp/mtp_trained.pt"

def load_prompts(prompt_dir):
    all_p = []
    for p in sorted(glob.glob(os.path.join(prompt_dir, "*.txt"))):
        with open(p) as fh:
            txt = fh.read()
        if len(txt) < 50:
            continue
        all_p.append((os.path.basename(p), txt))
    return all_p

def load_wiki(n):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out = []
    # Aggregate consecutive lines until we get chunks of decent length
    buf = ""
    for row in ds:
        buf += row["text"]
        if len(buf) >= 1500:
            out.append((f"wiki_{len(out):04d}", buf))
            buf = ""
            if len(out) >= n:
                break
    return out

@torch.no_grad()
def eval_acceptance(model, mtp, embed, tok, eval_prompts):
    """Return (mtp_trunk_agreement, mtp_next_token_acc) averaged across eval set."""
    totals = {"agree": 0, "mtp_correct": 0, "total": 0}
    for name, txt in eval_prompts:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.cuda()
        T = ids.shape[1]
        if T < 2:
            continue
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hid = out.hidden_states[-1]
        trunk_argmax = out.logits.argmax(-1)
        prev_emb = embed(ids)
        mtp_hid = mtp(prev_emb, trunk_hidden=hid)
        mtp_logits = F.linear(mtp_hid, embed.weight)
        mtp_argmax = mtp_logits.argmax(-1)
        labels = ids[0, 1:]
        totals["agree"] += int((mtp_argmax[0, :-1] == trunk_argmax[0, :-1]).sum())
        totals["mtp_correct"] += int((mtp_argmax[0, :-1] == labels).sum())
        totals["total"] += int(labels.numel())
    if totals["total"] == 0:
        return 0.0, 0.0
    return totals["agree"]/totals["total"], totals["mtp_correct"]/totals["total"]

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    print("=== FastMTP fine-tune smoke on Qwen3.5-0.8B MTP block ===\n")

    print("Load trunk + MTP...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    embed = model.get_input_embeddings()
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B")
    mtp_sd = load_mtp_from_safetensors(cache_dir)
    mtp = Qwen35MtpBlock(text_cfg).to(device="cuda:0", dtype=torch.bfloat16)
    mtp.load_pretrained_(mtp_sd)
    for p in model.parameters(): p.requires_grad = False
    n_train = sum(p.numel() for p in mtp.parameters())
    print(f"  trunk frozen, MTP trainable: {n_train:,} params\n")

    print("Load data...")
    all_p = load_prompts(PROMPT_DIR)
    train_p = [(n, t) for n, t in all_p if n not in HOLDOUT_PROMPTS]
    eval_p  = [(n, t) for n, t in all_p if n in HOLDOUT_PROMPTS]
    print(f"  hipfire training prompts: {len(train_p)}")
    print(f"  hipfire held-out eval:    {len(eval_p)}")
    print(f"  loading {N_WIKI_SAMPLES} wikitext samples...")
    wiki = load_wiki(N_WIKI_SAMPLES)
    train_pool = train_p + wiki
    print(f"  total training pool:      {len(train_pool)}\n")

    optimizer = torch.optim.AdamW(mtp.parameters(), lr=LR, weight_decay=0.01)
    # Cosine LR with warmup
    def lr_lambda(step):
        warmup = 50
        if step < warmup: return step / warmup
        import math
        progress = (step - warmup) / max(1, N_TRAIN_STEPS - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Eval BEFORE training...")
    a0, m0 = eval_acceptance(model, mtp, embed, tok, eval_p)
    print(f"  baseline: agree={100*a0:.2f}%  mtp_acc={100*m0:.2f}%\n")

    print(f"{'step':<5} {'lr':<10} {'loss':<8} {'tokens':<7} {'wall_ms':<8}  {'eval_agree':<11}")
    print("-" * 60)
    losses = []
    eval_history = [(0, a0, m0)]
    train_t0 = time.time()
    for step in range(N_TRAIN_STEPS):
        prompt_name, prompt = random.choice(train_pool)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.cuda()
        T = ids.shape[1]
        if T < 4:
            continue

        # Forward trunk (no grad)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hid = out.hidden_states[-1]
            # Self-distill labels: trunk's own argmax
            trunk_argmax = out.logits.argmax(-1)  # [1, T]

        # MTP forward + loss
        t_step = time.time()
        prev_emb = embed(ids)
        mtp_hid = mtp(prev_emb, trunk_hidden=hid)
        mtp_logits = F.linear(mtp_hid, embed.weight)  # tied lm_head
        # Predict t+1 from logits at t. So mtp_logits[t] should match trunk_argmax[t]
        # (both predict the token after position t).
        # Skip last position (no t+1 label) and first position (degenerate)
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
            ae, me = eval_acceptance(model, mtp, embed, tok, eval_p)
            eval_history.append((step + 1, ae, me))
            lr_now = sched.get_last_lr()[0]
            print(f"{step:<5} {lr_now:<10.2e} {loss.item():<8.4f} {T:<7} {dt:<8.1f}  {100*ae:<5.2f} ({100*(ae-a0):+.2f})")
        else:
            lr_now = sched.get_last_lr()[0]
            if step % 10 == 0:  # less verbose
                print(f"{step:<5} {lr_now:<10.2e} {loss.item():<8.4f} {T:<7} {dt:<8.1f}")

    print("-" * 60)
    print(f"\nFinal eval...")
    af, mf = eval_acceptance(model, mtp, embed, tok, eval_p)
    print(f"  baseline:  agree={100*a0:.2f}%  mtp_acc={100*m0:.2f}%")
    print(f"  final:     agree={100*af:.2f}%  mtp_acc={100*mf:.2f}%")
    print(f"  delta:     agree={100*(af-a0):+.2f}%  mtp_acc={100*(mf-m0):+.2f}%\n")

    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Training wall: {time.time()-train_t0:.1f}s  ({(time.time()-train_t0)/N_TRAIN_STEPS*1000:.1f} ms/step avg)")
    print(f"Total wall:    {time.time()-t0:.1f}s")
    print(f"GPU mem peak:  {mem_peak:.1f}GB")

    # Save checkpoint
    torch.save({
        "mtp_state_dict": mtp.state_dict(),
        "config": {k:v for k,v in text_cfg.__dict__.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))},
        "baseline_agree": a0,
        "final_agree": af,
        "delta_agree": af - a0,
        "losses": losses,
        "eval_history": eval_history,
    }, CHECKPOINT)
    print(f"\nCheckpoint saved: {CHECKPOINT}")

    json.dump({
        "baseline_agree": a0,
        "final_agree": af,
        "delta_agree_pct_pts": 100*(af-a0),
        "baseline_mtp_acc": m0,
        "final_mtp_acc": mf,
        "training_steps": N_TRAIN_STEPS,
        "training_wall_seconds": time.time() - train_t0,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean_last10": sum(losses[-10:]) / 10,
        "eval_history": eval_history,
        "mem_peak_gb": mem_peak,
    }, open("/tmp/mtp_train_result.json", "w"), indent=2)

if __name__ == "__main__":
    main()
