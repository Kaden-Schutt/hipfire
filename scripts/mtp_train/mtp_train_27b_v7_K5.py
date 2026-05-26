"""
FastMTP v6 — full 5M calib + recursive K=3 CE.

Changes from v5:
- Corpus 10x: full 5M calib (~5000 chunks at 1024 chars) + HumanEval+ + distill
- Recursive depth K=3 (was K=2 in v5): train MTP to predict t+1, t+2, t+3
  with its own drafts feeding back. Decay 0.5, 0.25 (FastMTP-style).
- 1000 steps (was 500): more iterations on larger corpus.
- Same lr 5e-5 cosine, MAX_LEN 512 (memory-safe).
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
CORPUS_DIR = "/workspace/mtp-fastmtp/corpus_v2"
PROMPT_DIR = "/workspace/hipfire/benchmarks/prompts"
MAX_LEN_TRAIN = 384
MAX_LEN_EVAL = 384
SEED = 42
N_TRAIN_STEPS = 1500
LR = 5e-5
WARMUP_STEPS = 100
EVAL_EVERY = 100
N_RECURSIVE_STEPS = 5   # was 2 in v5
DECAYS = [1.0, 0.5, 0.25, 0.125, 0.0625]  # weights for step 0, 1, 2 losses
HOLDOUT = {
    "lru_cache_pep8_strict.txt",
    "humaneval_0_has_close_elements.txt",
    "agentic_user_multistep.txt",
    "trains-meet.txt",
    "tool_call_system.txt",
}
CHECKPOINT = "/tmp/mtp_3p6_27b_v7_K5.pt"

def load_eval_prompts():
    return [(os.path.basename(p), open(p).read())
            for p in sorted(glob.glob(os.path.join(PROMPT_DIR, "*.txt")))
            if os.path.basename(p) in HOLDOUT and len(open(p).read()) >= 50]

def load_humaneval_plus(path):
    out = []
    if not os.path.exists(path): return out
    with open(path) as fh:
        for i, line in enumerate(fh):
            try:
                d = json.loads(line)
                txt = d.get("prompt", "") + d.get("canonical_solution", "")
                if len(txt) >= 100 and "has_close_elements" not in d.get("entry_point", ""):
                    out.append((f"heval_{i:04d}", txt))
            except json.JSONDecodeError: pass
    return out

def load_distill(dir_path):
    out = []
    for f in sorted(glob.glob(os.path.join(dir_path, "*.stdout.txt"))):
        try:
            t = open(f).read()
            if len(t) >= 200: out.append((os.path.basename(f), t))
        except Exception: pass
    return out

def load_calib_chunks(path, chunk_chars=1024):
    """Chunk a large text file. Returns ~N chunks where N = file_size/chunk_chars."""
    if not os.path.exists(path): return []
    data = open(path).read()
    out = []
    for i in range(0, len(data) - chunk_chars, chunk_chars):
        chunk = data[i:i+chunk_chars]
        out.append((f"calib_{len(out):05d}", chunk))
    return out

@torch.no_grad()
def eval_recursive_K(model, mtp, embed, lm_head, tok, eval_prompts, K=3, max_len=MAX_LEN_EVAL):
    """Measure step0..step{K-1} agreement with recursive MTP-own chain."""
    totals = [{"agree": 0, "total": 0} for _ in range(K)]
    rows = []
    for name, txt in eval_prompts:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).input_ids.cuda()
        T = ids.shape[1]
        if T < K + 2: continue
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hid = out.hidden_states[-1]
        trunk_argmax = out.logits.argmax(-1)
        cur_hid = hid
        cur_emb = embed(ids)
        cur_pos = torch.arange(T, device=ids.device).unsqueeze(0)
        per_prompt = []
        for k in range(K):
            mtp_hid_k = mtp(cur_emb, trunk_hidden=cur_hid, position_ids=cur_pos + k)
            mtp_logits_k = lm_head(mtp_hid_k)
            mtp_pred_k = mtp_logits_k.argmax(-1)
            # At position t, mtp_pred_k predicts the token (k+1) steps ahead.
            # trunk_argmax[t+k] is what trunk predicts for position t+k+1.
            valid = T - 1 - k
            if valid > 0:
                a = int((mtp_pred_k[0, :valid] == trunk_argmax[0, k:k+valid]).sum())
                totals[k]["agree"] += a
                totals[k]["total"] += valid
                per_prompt.append(a / valid)
            else:
                per_prompt.append(0.0)
            # Setup next step
            cur_hid = mtp_hid_k
            cur_emb = embed(mtp_pred_k)
        rows.append({"prompt": name, "T": T, "steps": per_prompt})
    aggs = [(t["agree"] / max(1, t["total"])) for t in totals]
    return aggs, rows

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()
    print(f"=== FastMTP v6 K=3 RECURSIVE + LARGE CORPUS ({MODEL_ID}) ===\n")
    print(f"Loss = sum_k decay[k] * CE_step{0}..{N_RECURSIVE_STEPS-1}, decays={DECAYS}\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    embed = model.get_input_embeddings()
    lm_head = get_lm_head_module(model)
    mtp = Qwen35MtpBlock(text_cfg).to(device="cuda:0", dtype=torch.bfloat16)
    mtp.load_pretrained_(load_mtp_from_safetensors(hf_cache_dir(MODEL_ID)))
    for p in model.parameters(): p.requires_grad = False
    print(f"Loaded in {time.time()-t0:.1f}s. MTP {sum(p.numel() for p in mtp.parameters())/1e6:.1f}M params\n")

    heval = load_humaneval_plus(os.path.join(CORPUS_DIR, "humanevalplus.jsonl"))
    distill = load_distill(os.path.join(CORPUS_DIR, "distill_outputs"))
    calib = load_calib_chunks(os.path.join(CORPUS_DIR, "calib-5m-full.txt"))
    # Code-weight: 3x HE+, 3x distill, 1x calib
    pool = heval * 3 + distill * 3 + calib
    random.shuffle(pool)
    print(f"Corpus: {len(heval)} HE+ x3, {len(distill)} distill x3, {len(calib)} calib chunks → pool {len(pool)}\n")
    eval_p = load_eval_prompts()

    optimizer = torch.optim.AdamW(mtp.parameters(), lr=LR, weight_decay=0.01)
    def lr_lambda(step):
        if step < WARMUP_STEPS: return step / WARMUP_STEPS
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS) / max(1, N_TRAIN_STEPS - WARMUP_STEPS)))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("BASELINE recursive K=3 eval...")
    a0, _ = eval_recursive_K(model, mtp, embed, lm_head, tok, eval_p, K=N_RECURSIVE_STEPS)
    print(f"  step0: {100*a0[0]:.2f}%   step1: {100*a0[1]:.2f}%   step2: {100*a0[2]:.2f}%\n")

    print(f"TRAIN {N_TRAIN_STEPS} steps with K={N_RECURSIVE_STEPS} recursive CE...")
    print(f"{'step':<5} {'lr':<10} {'L0':<6} {'L1':<6} {'L2':<6} {'total':<7} {'tok':<5} {'ms':<6} {'s0':<6} {'s1':<6} {'s2':<6}")
    losses_total = []
    eval_hist = [(0,) + tuple(a0)]
    train_t0 = time.time()
    for step in range(N_TRAIN_STEPS):
        _, prompt = random.choice(pool)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN_TRAIN).input_ids.cuda()
        T = ids.shape[1]
        if T < N_RECURSIVE_STEPS + 4: continue

        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hid = out.hidden_states[-1]
            trunk_argmax = out.logits.argmax(-1)

        t_step = time.time()
        # Recursive K=3 forward
        cur_hid = hid
        cur_emb = embed(ids)
        cur_pos = torch.arange(T, device=ids.device).unsqueeze(0)
        step_losses = []
        for k in range(N_RECURSIVE_STEPS):
            mtp_hid_k = mtp(cur_emb, trunk_hidden=cur_hid, position_ids=cur_pos + k)
            mtp_logits_k = lm_head(mtp_hid_k)
            valid = T - 1 - k
            if valid <= 0: break
            loss_k = F.cross_entropy(
                mtp_logits_k[0, :valid, :].float(),
                trunk_argmax[0, k:k+valid].long(),
            )
            step_losses.append(loss_k)
            # Setup next step (stop grad at token argmax)
            with torch.no_grad():
                mtp_pred_k = mtp_logits_k.argmax(-1)
            cur_hid = mtp_hid_k  # keep gradient through hidden
            cur_emb = embed(mtp_pred_k)

        total_loss = sum(DECAYS[i] * step_losses[i] for i in range(len(step_losses)))
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mtp.parameters(), max_norm=1.0)
        optimizer.step()
        sched.step()
        torch.cuda.synchronize()
        dt = (time.time() - t_step) * 1000
        losses_total.append(total_loss.item())
        if step % 50 == 0: torch.cuda.empty_cache()
        if step > 0 and step % 200 == 0:
            torch.save({"mtp_state_dict": mtp.state_dict(), "step": step,
                       "model_id": MODEL_ID}, CHECKPOINT + f".step{step}")

        if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS - 1:
            ae, _ = eval_recursive_K(model, mtp, embed, lm_head, tok, eval_p, K=N_RECURSIVE_STEPS)
            eval_hist.append((step + 1,) + tuple(ae))
            l0 = step_losses[0].item() if len(step_losses) > 0 else 0
            l1 = step_losses[1].item() if len(step_losses) > 1 else 0
            l2 = step_losses[2].item() if len(step_losses) > 2 else 0
            print(f"{step:<5} {sched.get_last_lr()[0]:<10.2e} {l0:<6.3f} {l1:<6.3f} {l2:<6.3f} {total_loss.item():<7.3f} {T:<5} {dt:<6.1f} {100*ae[0]:5.1f} {100*ae[1]:5.1f} {100*ae[2]:5.1f}")

    af, final_rows = eval_recursive_K(model, mtp, embed, lm_head, tok, eval_p, K=N_RECURSIVE_STEPS)
    print(f"\nFINAL:")
    print(f"  baseline:  step0={100*a0[0]:.2f}%  step1={100*a0[1]:.2f}%  step2={100*a0[2]:.2f}%")
    print(f"  final:     step0={100*af[0]:.2f}%  step1={100*af[1]:.2f}%  step2={100*af[2]:.2f}%")
    print(f"  delta:     step0={100*(af[0]-a0[0]):+.2f}pp  step1={100*(af[1]-a0[1]):+.2f}pp  step2={100*(af[2]-a0[2]):+.2f}pp")
    print(f"\nPer-prompt step0/step1/step2:")
    for r in final_rows:
        s = "  ".join(f"{100*x:.1f}" for x in r["steps"])
        print(f"  {r['prompt'][:35]:<35} T={r['T']:>4}  {s}")
    print(f"\nTraining wall: {time.time()-train_t0:.1f}s")
    print(f"GPU mem peak:  {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    torch.save({
        "mtp_state_dict": mtp.state_dict(), "model_id": MODEL_ID,
        "baseline": a0, "final": af,
        "losses": losses_total, "eval_history": eval_hist,
        "loss_type": "recursive_K3_ce_decays", "decays": DECAYS,
    }, CHECKPOINT)
    print(f"Checkpoint: {CHECKPOINT}")

if __name__ == "__main__":
    main()
