"""
FastMTP v5 — RECURSIVE multi-step CE (the actual FastMTP recipe).

Hypothesis: v1-v4 all trained MTP for SINGLE-STEP prediction (given
ground-truth context). Runtime uses MTP's own drafts as context →
compounding error. We need to train MTP to predict t+2 USING ITS OWN
PRIOR DRAFT, not given truth.

Loss = CE_step0 + decay * CE_step1
  step0: MTP predicts trunk_argmax[t]   from (embed(ids[t]),       trunk_h[t])   at pos t
  step1: MTP predicts trunk_argmax[t+1] from (embed(mtp_pred_0[t]), mtp_h_0[t])   at pos t+1

Gradient flows through both forwards but stops at the argmax→embed
boundary (token selection is non-diff). Backprop teaches MTP to produce
hidden states + predictions that work for the NEXT step too.

Same v2 code-heavy corpus. 500 steps. CE loss (matches FastMTP paper).
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
MAX_LEN_TRAIN = 768
MAX_LEN_EVAL = 384
SEED = 42
N_TRAIN_STEPS = 500
LR = 5e-5
WARMUP_STEPS = 50
EVAL_EVERY = 50
N_RECURSIVE_STEPS = 2     # step 0 + step 1
RECURSIVE_DECAY = 0.5     # weight for step 1
HOLDOUT = {
    "lru_cache_pep8_strict.txt",
    "humaneval_0_has_close_elements.txt",
    "agentic_user_multistep.txt",
    "trains-meet.txt",
    "tool_call_system.txt",
}
CHECKPOINT = "/tmp/mtp_3p6_27b_v5_recursive.pt"

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

def load_calib_chunks(path, n_chunks=200, chunk_chars=1500):
    if not os.path.exists(path): return []
    data = open(path).read()
    return [(f"calib_{i:04d}", data[(i*chunk_chars)%max(1,len(data)-chunk_chars):(i*chunk_chars)%max(1,len(data)-chunk_chars)+chunk_chars])
            for i in range(n_chunks) if len(data[(i*chunk_chars)%max(1,len(data)-chunk_chars):(i*chunk_chars)%max(1,len(data)-chunk_chars)+chunk_chars]) >= 500]

@torch.no_grad()
def eval_recursive_acceptance(model, mtp, embed, lm_head, tok, eval_prompts, max_len=MAX_LEN_EVAL):
    """Measure BOTH step-0 and step-1 agreement, since v5 trains both."""
    totals = {"agree0": 0, "agree1": 0, "total0": 0, "total1": 0}
    rows = []
    for name, txt in eval_prompts:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).input_ids.cuda()
        T = ids.shape[1]
        if T < 4: continue
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hid = out.hidden_states[-1]
        trunk_argmax = out.logits.argmax(-1)
        # Step 0
        prev_emb_0 = embed(ids)
        pos_0 = torch.arange(T, device=ids.device).unsqueeze(0)
        mtp_hid_0 = mtp(prev_emb_0, trunk_hidden=hid, position_ids=pos_0)
        mtp_logits_0 = lm_head(mtp_hid_0)
        mtp_pred_0 = mtp_logits_0.argmax(-1)
        # Step 1 (recursive)
        prev_emb_1 = embed(mtp_pred_0)
        pos_1 = pos_0 + 1
        mtp_hid_1 = mtp(prev_emb_1, trunk_hidden=mtp_hid_0, position_ids=pos_1)
        mtp_logits_1 = lm_head(mtp_hid_1)
        mtp_pred_1 = mtp_logits_1.argmax(-1)
        # Compare
        agree0 = int((mtp_pred_0[0, :-1] == trunk_argmax[0, :-1]).sum())
        # step 1 at pos t predicts t+2. trunk_argmax[t+1] is what trunk predicts for t+2.
        agree1 = int((mtp_pred_1[0, :-2] == trunk_argmax[0, 1:-1]).sum())
        totals["agree0"] += agree0; totals["total0"] += T - 1
        totals["agree1"] += agree1; totals["total1"] += T - 2
        rows.append({
            "prompt": name, "T": T,
            "agree0": agree0 / max(1, T - 1),
            "agree1": agree1 / max(1, T - 2),
        })
    if totals["total0"] == 0: return 0.0, 0.0, rows
    return totals["agree0"]/totals["total0"], totals["agree1"]/totals["total1"], rows

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()
    print(f"=== FastMTP v5 RECURSIVE multi-step CE ({MODEL_ID}) ===\n")
    print(f"Loss = CE_step0 + {RECURSIVE_DECAY} * CE_step1\n")

    print(f"Load trunk + MTP...")
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
    print(f"  loaded in {time.time()-t0:.1f}s, MTP {sum(p.numel() for p in mtp.parameters())/1e6:.1f}M params\n")

    heval = load_humaneval_plus(os.path.join(CORPUS_DIR, "humanevalplus.jsonl"))
    distill = load_distill(os.path.join(CORPUS_DIR, "distill_outputs"))
    calib = load_calib_chunks(os.path.join(CORPUS_DIR, "calib_slice.txt"))
    pool = heval + heval + distill + distill + calib
    random.shuffle(pool)
    print(f"Corpus: pool {len(pool)}\n")
    eval_p = load_eval_prompts()

    optimizer = torch.optim.AdamW(mtp.parameters(), lr=LR, weight_decay=0.01)
    def lr_lambda(step):
        if step < WARMUP_STEPS: return step / WARMUP_STEPS
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS) / max(1, N_TRAIN_STEPS - WARMUP_STEPS)))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("BASELINE recursive eval...")
    a0_0, a0_1, _ = eval_recursive_acceptance(model, mtp, embed, lm_head, tok, eval_p)
    print(f"  step0: {100*a0_0:.2f}%   step1 (recursive): {100*a0_1:.2f}%\n")

    print(f"TRAIN {N_TRAIN_STEPS} steps with recursive loss...")
    print(f"{'step':<5} {'lr':<10} {'loss0':<7} {'loss1':<7} {'total':<7} {'tok':<5} {'ms':<6} {'agr0':<6} {'agr1':<6}")
    losses_total, losses_0, losses_1 = [], [], []
    eval_hist = [(0, a0_0, a0_1)]
    train_t0 = time.time()
    for step in range(N_TRAIN_STEPS):
        _, prompt = random.choice(pool)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN_TRAIN).input_ids.cuda()
        T = ids.shape[1]
        if T < 6: continue

        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hid = out.hidden_states[-1]
            trunk_argmax = out.logits.argmax(-1)

        t_step = time.time()
        # Step 0
        prev_emb_0 = embed(ids)
        pos_0 = torch.arange(T, device=ids.device).unsqueeze(0)
        mtp_hid_0 = mtp(prev_emb_0, trunk_hidden=hid, position_ids=pos_0)
        mtp_logits_0 = lm_head(mtp_hid_0)
        loss_0 = F.cross_entropy(mtp_logits_0[0, :-1, :].float(), trunk_argmax[0, :-1].long())

        # Step 1 (recursive — uses MTP's own prediction + hidden)
        # argmax breaks the gradient at the token level, but hidden state gradient still flows.
        with torch.no_grad():
            mtp_pred_0 = mtp_logits_0.argmax(-1)
        prev_emb_1 = embed(mtp_pred_0)  # detached at the token boundary
        pos_1 = pos_0 + 1
        mtp_hid_1 = mtp(prev_emb_1, trunk_hidden=mtp_hid_0, position_ids=pos_1)
        mtp_logits_1 = lm_head(mtp_hid_1)
        # Step 1 at pos t predicts t+2 → label = trunk_argmax[t+1]
        loss_1 = F.cross_entropy(mtp_logits_1[0, :-2, :].float(), trunk_argmax[0, 1:-1].long())

        loss = loss_0 + RECURSIVE_DECAY * loss_1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mtp.parameters(), max_norm=1.0)
        optimizer.step()
        sched.step()
        torch.cuda.synchronize()
        dt = (time.time() - t_step) * 1000
        losses_total.append(loss.item()); losses_0.append(loss_0.item()); losses_1.append(loss_1.item())

        if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS - 1:
            ae_0, ae_1, _ = eval_recursive_acceptance(model, mtp, embed, lm_head, tok, eval_p)
            eval_hist.append((step + 1, ae_0, ae_1))
            print(f"{step:<5} {sched.get_last_lr()[0]:<10.2e} {loss_0.item():<7.3f} {loss_1.item():<7.3f} {loss.item():<7.3f} {T:<5} {dt:<6.1f} {100*ae_0:<6.2f} {100*ae_1:<6.2f}")

    af_0, af_1, final_rows = eval_recursive_acceptance(model, mtp, embed, lm_head, tok, eval_p)
    print(f"\nFINAL:")
    print(f"  baseline:  step0={100*a0_0:.2f}%   step1={100*a0_1:.2f}%")
    print(f"  final:     step0={100*af_0:.2f}%   step1={100*af_1:.2f}%")
    print(f"  delta:     step0={100*(af_0-a0_0):+.2f}pp  step1={100*(af_1-a0_1):+.2f}pp")
    print("\n  Per-prompt step0/step1:")
    for r in final_rows:
        print(f"  {r['prompt'][:35]:<35} T={r['T']:>4}  step0={100*r['agree0']:5.1f}  step1={100*r['agree1']:5.1f}")
    print(f"\nTraining wall: {time.time()-train_t0:.1f}s   GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    torch.save({
        "mtp_state_dict": mtp.state_dict(), "model_id": MODEL_ID,
        "baseline_agree0": a0_0, "baseline_agree1": a0_1,
        "final_agree0": af_0, "final_agree1": af_1,
        "losses_total": losses_total, "losses_0": losses_0, "losses_1": losses_1,
        "eval_history": eval_hist,
        "loss_type": "recursive_multi_step_ce",
        "n_recursive_steps": N_RECURSIVE_STEPS, "recursive_decay": RECURSIVE_DECAY,
    }, CHECKPOINT)
    print(f"Checkpoint: {CHECKPOINT}")

if __name__ == "__main__":
    main()
