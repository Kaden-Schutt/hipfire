import torch
from transformers import AutoModelForCausalLM

model_path = "/local/models/google/gemma-4-12B-it"

print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
)
model.eval()

print("Model loaded.")

# Let's intercept the first layer (layer 0) using a hook or forward modification
language_model = model.model.language_model
layer0 = language_model.layers[0]

# Let's save intermediates
intermediates = {}

def hook_fn(module, input, output):
    intermediates['layer0_input'] = input[0].detach()
    intermediates['layer0_output'] = output[0].detach()

hook = layer0.register_forward_hook(hook_fn)

# Capture input norm and attention projections
def make_hook(name):
    def h(module, input, output):
        intermediates[name] = output.detach()
    return h

language_model.embed_tokens.register_forward_hook(make_hook('embed_tokens'))
layer0.input_layernorm.register_forward_hook(make_hook('input_layernorm'))
layer0.self_attn.q_proj.register_forward_hook(make_hook('q_proj'))
layer0.self_attn.k_proj.register_forward_hook(make_hook('k_proj'))
layer0.self_attn.v_proj.register_forward_hook(make_hook('v_proj'))
layer0.self_attn.o_proj.register_forward_hook(make_hook('o_proj'))
layer0.post_attention_layernorm.register_forward_hook(make_hook('post_attention_layernorm'))
layer0.pre_feedforward_layernorm.register_forward_hook(make_hook('pre_feedforward_layernorm'))
layer0.mlp.gate_proj.register_forward_hook(make_hook('gate_proj'))
layer0.mlp.up_proj.register_forward_hook(make_hook('up_proj'))
layer0.mlp.down_proj.register_forward_hook(make_hook('down_proj'))
layer0.post_feedforward_layernorm.register_forward_hook(make_hook('post_feedforward_layernorm'))

class CaptureInputHook:
    def __init__(self):
        self.val = None
    def __call__(self, module, input, output):
        self.val = input[0].detach()

o_proj_input_capture = CaptureInputHook()
layer0.self_attn.o_proj.register_forward_hook(o_proj_input_capture)

# Sequence is [2, 9259] (BOS, Hello)
input_ids = torch.tensor([[2, 9259]], device="cpu")

print("Running forward pass...")
with torch.no_grad():
    outputs = model(input_ids)

print("\n--- HF INTERMEDIATES ---")
for k, v in intermediates.items():
    v_flat = v.view(-1)
    print(f"[{k}] shape={v.shape} sum={v_flat.sum().item():>+14.4e} first4={v_flat[:4].tolist()}")

attn_output = o_proj_input_capture.val
print("\n--- ATTENTION OUTPUT (pre-o_proj) ---")
print(f"shape={attn_output.shape} sum={attn_output.sum().item():>+14.4e} first4={attn_output.view(-1)[:4].tolist()}")

gqa_ratio = model.config.num_attention_heads // model.config.num_key_value_heads
for pos in [0, 1]:
    print(f"\nPosition {pos}:")
    pos_data = attn_output[0, pos] # [4096]
    for h in range(model.config.num_attention_heads):
        start = h * model.config.head_dim
        end = start + model.config.head_dim
        h_sum = pos_data[start:end].sum().item()
        first2 = pos_data[start:start+2].tolist()
        print(f"  head {h:2} (kv={h // gqa_ratio}): sum={h_sum:>+10.4f} first2={first2}")
