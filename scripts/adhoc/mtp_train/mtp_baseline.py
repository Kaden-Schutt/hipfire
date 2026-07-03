"""
Measure baseline pretrained-MTP acceptance on a held-out prompt set.

Pre-training baseline for FastMTP. We want to know how well Qwen's
stock MTP block predicts the trunk's argmax at single-step (K=1) and
two-step (K=2) horizons. Post-training we'll measure the same and
compute Δ. Δ ≥ +5% on single-step would validate the recipe.
"""
import sys, os, glob, time, json
import torch
import torch.nn.functional as F

# Make our MTP module importable
sys.path.insert(0, "/tmp")
from mtp_module import Qwen35MtpBlock, load_mtp_from_safetensors

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT_DIR = "/workspace/hipfire/benchmarks/prompts"
MAX_LEN = 512  # cap per-prompt tokens to keep eval bounded

def load_prompts(prompt_dir, max_n=21):
    prompts = []
    for p in sorted(glob.glob(os.path.join(prompt_dir, "*.txt"))):
        with open(p) as fh:
            txt = fh.read()
        if len(txt) < 50:  # skip tiny prompts
            continue
        prompts.append((os.path.basename(p), txt))
        if len(prompts) >= max_n:
            break
    return prompts

@torch.no_grad()
def measure_one(model, mtp, embed, ids, mtp_k=1):
    """For one prompt, compute:
    - trunk single-step next-token accuracy
    - MTP single-step next-token accuracy
    - MTP K-step joint accuracy (k=2: probability that token t+1 AND t+2 both correct)
    Returns dict of metrics + counts.
    """
    T = ids.shape[1]
    trunk_out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    trunk_hidden = trunk_out.hidden_states[-1]   # [1, T, H]
    trunk_argmax = trunk_out.logits.argmax(-1)   # [1, T]

    # MTP single-step: at position t, predict ids[t+1] using emb(ids[t]) + trunk_hidden[t]
    prev_emb = embed(ids)
    mtp_hidden = mtp(prev_emb, trunk_hidden)
    mtp_logits = F.linear(mtp_hidden, embed.weight)
    mtp_argmax = mtp_logits.argmax(-1)   # [1, T]

    # Single-step next-token accuracy: pred[t] should match ids[t+1]
    if T > 1:
        labels = ids[0, 1:]
        trunk_correct = (trunk_argmax[0, :-1] == labels)
        mtp_correct = (mtp_argmax[0, :-1] == labels)
        agree = (mtp_argmax[0, :-1] == trunk_argmax[0, :-1])
    else:
        trunk_correct = mtp_correct = agree = torch.tensor([])

    return {
        "tokens": T,
        "trunk_correct": int(trunk_correct.sum()),
        "mtp_correct": int(mtp_correct.sum()),
        "mtp_trunk_agree": int(agree.sum()),
        "total": len(labels) if T > 1 else 0,
    }

def main():
    print(f"=== Pretrained MTP baseline on {MODEL_ID} ===\n")
    t0 = time.time()

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
    print(f"   loaded in {time.time()-t0:.1f}s\n")

    prompts = load_prompts(PROMPT_DIR)
    print(f"   {len(prompts)} prompts loaded\n")

    print(f"   {'prompt':<35} {'tok':<5} {'trunk':<7} {'mtp':<7} {'agree':<7}")
    print(f"   {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    totals = {"trunk_correct": 0, "mtp_correct": 0, "mtp_trunk_agree": 0, "total": 0, "tokens": 0}
    rows = []
    for name, txt in prompts:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.cuda()
        r = measure_one(model, mtp, embed, ids)
        for k in totals:
            totals[k] += r[k]
        tk_acc = r["trunk_correct"] / max(1, r["total"])
        mt_acc = r["mtp_correct"] / max(1, r["total"])
        ag_acc = r["mtp_trunk_agree"] / max(1, r["total"])
        print(f"   {name[:35]:<35} {r['tokens']:<5} {100*tk_acc:<7.1f} {100*mt_acc:<7.1f} {100*ag_acc:<7.1f}")
        rows.append({"prompt": name, **r, "trunk_acc": tk_acc, "mtp_acc": mt_acc, "agree": ag_acc})

    print(f"   {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    trunk_acc = totals["trunk_correct"] / max(1, totals["total"])
    mtp_acc = totals["mtp_correct"] / max(1, totals["total"])
    agree = totals["mtp_trunk_agree"] / max(1, totals["total"])
    print(f"   {'TOTAL':<35} {totals['tokens']:<5} {100*trunk_acc:<7.1f} {100*mtp_acc:<7.1f} {100*agree:<7.1f}")
    print()
    print(f"   trunk next-token accuracy: {100*trunk_acc:.2f}%")
    print(f"   MTP   next-token accuracy: {100*mtp_acc:.2f}%")
    print(f"   MTP/trunk argmax agree:    {100*agree:.2f}%")
    print(f"   gap (MTP vs trunk):        {100*(trunk_acc - mtp_acc):.2f}%")
    print()
    # The KEY METRIC for hipfire MTP solo: at K=1 with greedy verify,
    # acceptance = P(MTP_argmax == trunk_argmax). That's `agree`.
    # τ for K=1 = 1 + agree. For K=2 with conditional accept, τ ≈ 1 + agree + agree*conditional_2nd.
    print(f"   K=1 acceptance estimate: {100*agree:.2f}% → τ≈{1+agree:.3f}")
    print(f"   Total wall: {time.time()-t0:.1f}s")

    json.dump({
        "trunk_acc": trunk_acc,
        "mtp_acc": mtp_acc,
        "mtp_trunk_agree": agree,
        "total_tokens": totals["total"],
        "n_prompts": len(prompts),
        "rows": rows,
    }, open("/tmp/mtp_baseline.json", "w"), indent=2)
    print("   JSON: /tmp/mtp_baseline.json")

if __name__ == "__main__":
    main()
