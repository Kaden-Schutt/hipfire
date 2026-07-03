"""
Qwen35MtpBlock — custom nn.Module that loads Qwen's stock MTP layer
from safetensors. HF Transformers ignores mtp.* weights via
_keys_to_ignore_on_load_unexpected — this module loads them properly.

Architecture (from inspecting Qwen3.5-0.8B safetensors):
  mtp.pre_fc_norm_embedding (RMSNorm, hidden)
  mtp.pre_fc_norm_hidden     (RMSNorm, hidden)
  mtp.fc                     (Linear, 2*hidden -> hidden, no bias)
  mtp.layers.0               (Qwen3_5DecoderLayer, full_attention, GQA, attn_output_gate)
  mtp.norm                   (RMSNorm, hidden)

Forward:
  inputs: trunk_hidden_t (frozen trunk's last hidden at position t),
          prev_token_emb (embedding of token t-1)
  emb_n   = pre_fc_norm_embedding(prev_token_emb)
  hid_n   = pre_fc_norm_hidden(trunk_hidden_t)
  x       = fc(concat([emb_n, hid_n], dim=-1))   # [B, T, 2H] -> [B, T, H]
  x       = layers[0](x, position_embeddings=...)   # decoder layer
  x       = norm(x)
  logits  = lm_head(x)   # tied with trunk's embed_tokens.weight
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
import glob, os

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)


class Qwen35MtpBlock(nn.Module):
    """Pretrained Qwen3.5 MTP block.

    Loads the mtp.* weights from a model's safetensors. The trunk
    (the main 24-layer model) is NOT included here — pass its
    embed_tokens.weight as the lm_head (tied) and the trunk's last
    hidden state to forward()."""

    def __init__(self, text_config):
        super().__init__()
        cfg = copy.deepcopy(text_config)
        # MTP layer is a SINGLE full_attention layer regardless of trunk's
        # layer_types pattern. Force it for the decoder layer construction.
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        self.config = cfg
        H = cfg.hidden_size

        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        # fc: concat(emb, hidden) -> hidden, so input dim = 2H
        self.fc = nn.Linear(2 * H, H, bias=False)

        # The actual transformer layer (full_attention)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, layer_idx=0)])

        self.norm = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)

        # Rotary embedding for the decoder layer (shared with trunk in
        # principle, but cheap to re-instantiate)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)

    def load_pretrained_(self, mtp_state_dict: dict):
        """Load weights from a state_dict with keys like 'mtp.fc.weight'.
        Strips 'mtp.' prefix to match our local module hierarchy."""
        stripped = {}
        for k, v in mtp_state_dict.items():
            if k.startswith("mtp."):
                stripped[k[len("mtp."):]] = v
            else:
                stripped[k] = v
        missing, unexpected = self.load_state_dict(stripped, strict=False)
        return missing, unexpected

    def forward(self, prev_token_emb, trunk_hidden, position_ids=None):
        """
        prev_token_emb: [B, T, H] — embeddings of input tokens shifted by -1
                        (or whatever convention the caller uses; we just
                        concat them with trunk_hidden).
        trunk_hidden:   [B, T, H] — frozen trunk's last hidden state at
                        the same positions as prev_token_emb.
        position_ids:   [B, T] — position indices (defaults to arange)

        Returns: hidden output of MTP block, [B, T, H]. Caller computes
                 logits = F.linear(hidden, lm_head_weight) with tied embed.
        """
        B, T, H = trunk_hidden.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=trunk_hidden.device).unsqueeze(0).expand(B, -1)

        emb_n = self.pre_fc_norm_embedding(prev_token_emb)
        hid_n = self.pre_fc_norm_hidden(trunk_hidden)
        x = torch.cat([emb_n, hid_n], dim=-1)  # [B, T, 2H]
        x = self.fc(x)                          # [B, T, H]

        # Build position embeddings via rotary
        position_embeddings = self.rotary_emb(x, position_ids)

        # Single decoder layer
        x = self.layers[0](
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=None,  # causal handled inside if needed
            position_ids=position_ids,
            past_key_values=None,
        )
        x = self.norm(x)
        return x


def load_mtp_from_safetensors(model_dir: str):
    """Find safetensors in a HF model dir, return dict of mtp.* tensors.
    Handles both HF cache layout (snapshots/<hash>/) and flat directory."""
    snaps = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    if snaps:
        snap_dir = snaps[0]
    else:
        snap_dir = model_dir
    sft_files = sorted(glob.glob(os.path.join(snap_dir, "*.safetensors")))
    mtp = {}
    for f in sft_files:
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if k.startswith("mtp."):
                    mtp[k] = fh.get_tensor(k)
    return mtp


def hf_cache_dir(model_id: str) -> str:
    """Return the HF cache root for a model ID like 'Qwen/Qwen3.6-27B'."""
    safe = model_id.replace("/", "--")
    return os.path.expanduser(f"~/.cache/huggingface/hub/models--{safe}")


def get_lm_head_module(model):
    """Return the lm_head module (handles tied + untied embeddings).
    For tied: returns a wrapper that uses embed.weight.
    For untied: returns the actual nn.Linear lm_head."""
    out = model.get_output_embeddings()
    if out is None:
        # Strictly tied — fall back to input embeddings
        return model.get_input_embeddings()
    return out


def smoke_test():
    """Verify the module loads, accepts trunk hidden, and produces logits."""
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    MODEL_ID = "Qwen/Qwen3.5-0.8B"
    print(f"=== Smoke test: load + forward Qwen35MtpBlock on {MODEL_ID} ===\n")
    t0 = time.time()

    print("1. Load trunk model (HF will skip mtp.* weights)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"   trunk loaded in {time.time()-t0:.1f}s")
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    embed = model.get_input_embeddings()
    print(f"   hidden={text_cfg.hidden_size}, vocab={embed.weight.shape[0]}")

    print("\n2. Load mtp.* tensors from safetensors...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B")
    mtp_sd = load_mtp_from_safetensors(cache_dir)
    print(f"   found {len(mtp_sd)} mtp.* tensors")

    print("\n3. Build Qwen35MtpBlock and load pretrained weights...")
    mtp = Qwen35MtpBlock(text_cfg).to(device="cuda:0", dtype=torch.bfloat16)
    missing, unexpected = mtp.load_pretrained_(mtp_sd)
    print(f"   missing keys (in module, not loaded): {len(missing)}")
    for k in missing[:5]: print(f"     - {k}")
    print(f"   unexpected keys (in sd, not in module): {len(unexpected)}")
    for k in unexpected[:5]: print(f"     - {k}")

    n_params = sum(p.numel() for p in mtp.parameters())
    print(f"   MTP block params: {n_params:,} ({n_params/1e6:.1f}M)")

    print("\n4. Forward: trunk + MTP, compare argmax to trunk's lm_head...")
    prompt = "def fibonacci(n):\n    if n < 2:\n        return n\n    return"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    print(f"   prompt: {prompt!r}")
    print(f"   tokens: {ids.shape[1]}")

    with torch.no_grad():
        trunk_out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        trunk_hidden = trunk_out.hidden_states[-1]   # [1, T, H]
        trunk_logits = trunk_out.logits               # [1, T, V]
        trunk_argmax = trunk_logits.argmax(-1)        # [1, T]

        # Per hipfire mtp_spec.rs runtime trace (agent #2):
        #   Input to MTP block: embed(last_committed_token) + prev_hidden
        # In training: at position t, "last_committed" = ids[t], "prev_hidden" = trunk_hidden[t].
        # MTP then predicts the NEXT token (ground truth = ids[t+1]).
        # So NO shift on the embedding side.
        prev_emb = embed(ids)  # [1, T, H] — embed of token AT position t

        # Try BOTH conventions and report
        mtp_hidden = mtp(prev_emb, trunk_hidden)   # [1, T, H]
        mtp_logits = F.linear(mtp_hidden, embed.weight)    # tied lm_head
        mtp_argmax = mtp_logits.argmax(-1)                 # [1, T]

    # Compute agreement: at each position, MTP's argmax should ideally
    # predict the NEXT token, same as trunk's lm_head predicts the next
    # token from the current hidden. Let's compare both.
    print("\n   Trunk argmax (predicts next token at each position):")
    print(f"     positions 0..10:  {trunk_argmax[0,:11].tolist()}")
    print(f"     decoded: {tok.decode(trunk_argmax[0,:11].tolist())!r}")
    print("\n   MTP argmax (predicts ??? from hidden + prev_emb):")
    print(f"     positions 0..10:  {mtp_argmax[0,:11].tolist()}")
    print(f"     decoded: {tok.decode(mtp_argmax[0,:11].tolist())!r}")

    # If MTP is meant to predict t+1 from hidden_t + emb_{t-1}, then
    # MTP[t-1] should predict ids[t] (same as trunk[t-1] does).
    # Agreement: count where mtp_argmax[t] == trunk_argmax[t]
    agree = (mtp_argmax == trunk_argmax).float().mean().item()
    print(f"\n   MTP/trunk argmax agreement: {100*agree:.1f}%")
    # Also: does MTP predict the actual next-token in the sequence?
    # ids[t+1] should = mtp_argmax[t]? Position t MTP sees emb_{t-1} + hidden_t,
    # so it should predict ids[t+1] (the token AFTER position t)
    if ids.shape[1] > 1:
        labels = ids[0, 1:]
        mtp_pred = mtp_argmax[0, :-1]
        trunk_pred = trunk_argmax[0, :-1]
        mtp_acc = (mtp_pred == labels).float().mean().item()
        trunk_acc = (trunk_pred == labels).float().mean().item()
        print(f"   Next-token accuracy vs actual prompt tokens:")
        print(f"     trunk: {100*trunk_acc:.1f}%   mtp: {100*mtp_acc:.1f}%")

    print(f"\n   total wall: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    smoke_test()
