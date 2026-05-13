#!/usr/bin/env python3
"""PyTorch layer/tensor attribution for Qwen3.5 HFQ quant errors.

This is intentionally independent of Astrea. It uses the upstream HF/PyTorch
Qwen3.5 implementation as the execution oracle, captures BF16 layer inputs and
outputs, then replays selected layers with one HFQ-dequantized tensor swapped in
at a time. The output ranks which quantized tensors perturb the BF16 layer
output most on a real token sequence.
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Local HF snapshot or repo id")
    p.add_argument("--hfq-model", required=True, help="Candidate HFQ/MQ model")
    p.add_argument("--ref", help="Optional HFKLDR ref; uses chunk tokens as prompt")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--tokens", help="Comma-separated token ids; overrides --ref")
    p.add_argument("--layers", required=True, help="Comma list, ranges, or all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--tensor-filter", action="append", default=[], help="Substring filter; repeatable")
    p.add_argument("--include-f16", action="store_true", help="Also test F16/F32 tensors such as norms")
    p.add_argument("--layer-all", action="store_true", help="Also replay each layer with all matching HFQ tensors swapped")
    p.add_argument("--out")
    p.add_argument("--allow-remote", action="store_true")
    p.add_argument("--max-print", type=int, default=40)
    return p.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [x for x in out if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"layer(s) out of range for {n_layers}: {bad}")
    return sorted(set(out))


def find_json_end(buf: mmap.mmap, offset: int, end: int) -> int:
    brace = 0
    in_string = False
    esc = False
    for pos in range(offset, end):
        b = buf[pos]
        if esc:
            esc = False
            continue
        if b == 0x5C and in_string:
            esc = True
            continue
        if b == 0x22:
            in_string = not in_string
            continue
        if not in_string:
            if b == 0x7B:
                brace += 1
            elif b == 0x7D:
                brace -= 1
                if brace == 0:
                    return pos + 1
    raise ValueError("metadata JSON end not found")


class Hfq:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.f = self.path.open("rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.version, self.arch_id, self.n_tensors, meta_off, data_off = struct.unpack_from("<4sIIIQQ", self.mm, 0)
        if magic != b"HFQM":
            raise ValueError(f"not HFQ: {self.path}")
        json_end = find_json_end(self.mm, meta_off, data_off)
        self.metadata = json.loads(self.mm[meta_off:json_end])
        pos = json_end
        idx_n = struct.unpack_from("<I", self.mm, pos)[0]
        pos += 4
        if idx_n != self.n_tensors:
            raise ValueError(f"tensor count mismatch: {idx_n} != {self.n_tensors}")
        self.tensors: dict[str, dict[str, Any]] = {}
        off = data_off
        for _ in range(idx_n):
            name_len = struct.unpack_from("<H", self.mm, pos)[0]
            pos += 2
            name = bytes(self.mm[pos:pos + name_len]).decode("utf-8")
            pos += name_len
            qt = self.mm[pos]
            pos += 1
            nd = self.mm[pos]
            pos += 1
            shape = list(struct.unpack_from("<" + "I" * nd, self.mm, pos))
            pos += 4 * nd
            group = struct.unpack_from("<I", self.mm, pos)[0]
            pos += 4
            size = struct.unpack_from("<Q", self.mm, pos)[0]
            pos += 8
            self.tensors[name] = {"quant_type": qt, "shape": shape, "group_size": group, "offset": off, "size": size}
            off += size

    def tensor_bytes(self, name: str) -> tuple[dict[str, Any], bytes]:
        info = self.tensors[name]
        return info, self.mm[info["offset"]:info["offset"] + info["size"]]


def gen_fwht_signs(seed: int, n: int = 256) -> np.ndarray:
    state = seed
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = 1.0 if ((state >> 16) & 1) == 1 else -1.0
    return out


SIGNS1 = gen_fwht_signs(42)
SIGNS2 = gen_fwht_signs(1042)


def inverse_fwht_256(group: np.ndarray) -> np.ndarray:
    g = group.astype(np.float32, copy=True)
    g *= SIGNS2
    stride = 1
    while stride < 256:
        for j in range(0, 256, stride * 2):
            a = g[j:j + stride].copy()
            b = g[j + stride:j + stride * 2].copy()
            g[j:j + stride] = a + b
            g[j + stride:j + stride * 2] = a - b
        stride <<= 1
    g *= np.float32(0.0625)
    g *= SIGNS1
    return g


def inverse_fwht_rows_256(vals: np.ndarray) -> np.ndarray:
    g = vals.astype(np.float32, copy=True)
    g *= SIGNS2[None, :]
    stride = 1
    while stride < 256:
        v = g.reshape(g.shape[0], -1, stride * 2)
        a = v[:, :, :stride].copy()
        b = v[:, :, stride:stride * 2].copy()
        v[:, :, :stride] = a + b
        v[:, :, stride:stride * 2] = a - b
        stride <<= 1
    g *= np.float32(0.0625)
    g *= SIGNS1[None, :]
    return g


def dequant_4bit(data: bytes, group_size: int, rotated: bool) -> np.ndarray:
    bytes_per_group = 8 + group_size // 2
    if len(data) % bytes_per_group:
        raise ValueError(f"bad 4-bit byte count {len(data)} for group={group_size}")
    if rotated and group_size != 256:
        raise ValueError("rotated 4-bit dequant currently expects group_size=256")
    block_dtype = np.dtype([("scale", "<f4"), ("zero", "<f4"), ("packed", "u1", (group_size // 2,))])
    blocks = np.frombuffer(data, dtype=block_dtype)
    packed = blocks["packed"]
    vals = np.empty((blocks.shape[0], group_size), dtype=np.float32)
    vals[:, 0::2] = (packed & 0xF).astype(np.float32)
    vals[:, 1::2] = (packed >> 4).astype(np.float32)
    vals *= blocks["scale"][:, None]
    vals += blocks["zero"][:, None]
    if rotated:
        vals = inverse_fwht_rows_256(vals)
    return vals.reshape(-1)


def dequant_6bit(data: bytes, rotated: bool) -> np.ndarray:
    group_size = 256
    bytes_per_group = 200
    if len(data) % bytes_per_group:
        raise ValueError(f"bad 6-bit byte count {len(data)}")
    block_dtype = np.dtype([("scale", "<f4"), ("zero", "<f4"), ("packed", "u1", (192,))])
    blocks = np.frombuffer(data, dtype=block_dtype)
    packed = blocks["packed"].reshape(blocks.shape[0], 64, 3).astype(np.uint32)
    vals = np.empty((blocks.shape[0], group_size), dtype=np.float32)
    b0 = packed[:, :, 0]
    b1 = packed[:, :, 1]
    b2 = packed[:, :, 2]
    vals[:, 0::4] = (b0 & 0x3F).astype(np.float32)
    vals[:, 1::4] = (((b0 >> 6) | (b1 << 2)) & 0x3F).astype(np.float32)
    vals[:, 2::4] = (((b1 >> 4) | (b2 << 4)) & 0x3F).astype(np.float32)
    vals[:, 3::4] = ((b2 >> 2) & 0x3F).astype(np.float32)
    vals *= blocks["scale"][:, None]
    vals += blocks["zero"][:, None]
    if rotated:
        vals = inverse_fwht_rows_256(vals)
    return vals.reshape(-1)


def dequant_tensor(info: dict[str, Any], data: bytes) -> np.ndarray:
    qt = int(info["quant_type"])
    n = int(np.prod(info["shape"]))
    if qt == 1:
        arr = np.frombuffer(data, dtype="<f2").astype(np.float32)
    elif qt == 2:
        arr = np.frombuffer(data, dtype="<f4").astype(np.float32)
    elif qt == 6:
        arr = dequant_4bit(data, 256, rotated=False)
    elif qt == 7:
        arr = dequant_4bit(data, 128, rotated=False)
    elif qt == 8:
        arr = dequant_6bit(data, rotated=False)
    elif qt == 13:
        arr = dequant_4bit(data, 256, rotated=True)
    elif qt == 15:
        arr = dequant_6bit(data, rotated=True)
    else:
        raise ValueError(f"unsupported quant_type={qt}")
    if arr.size < n:
        raise ValueError(f"dequant produced {arr.size} values, need {n}")
    return arr[:n].reshape(tuple(info["shape"]))


def state_key_to_hfq_name(state_key: str) -> str:
    if state_key.startswith("model."):
        return "model.language_model." + state_key[len("model."):]
    return state_key


def parse_ref_tokens(path: str | Path, chunk: int, seq_len: int) -> list[int]:
    p = Path(path)
    with p.open("rb") as f:
        if f.read(8) != b"HFKLDR\0\0":
            raise ValueError(f"not HFKLDR: {p}")
        hdr = f.read(24)
        version = int.from_bytes(hdr[0:4], "little")
        n_ctx = int.from_bytes(hdr[4:8], "little")
        n_chunk = int.from_bytes(hdr[12:16], "little")
        if version != 1:
            raise ValueError(f"unsupported HFKLDR version {version}")
        if chunk < 0 or chunk >= n_chunk:
            raise ValueError(f"chunk {chunk} out of range 0..{n_chunk - 1}")
        if seq_len > n_ctx:
            raise ValueError(f"--seq-len {seq_len} exceeds ref ctx {n_ctx}")
        f.seek(8 + 24 + chunk * n_ctx * 4)
        raw = f.read(seq_len * 4)
    return [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]


def tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def clone_tree(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(clone_tree(v) for v in x)
    if isinstance(x, list):
        return [clone_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: clone_tree(v) for k, v in x.items()}
    return x


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    bf = b.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    d = af - bf
    mse = float(np.mean(d * d))
    rmse = math.sqrt(mse)
    ref = math.sqrt(float(np.mean(bf * bf)))
    denom = math.sqrt(float(np.dot(af, af))) * math.sqrt(float(np.dot(bf, bf)))
    return {
        "cosine": float(np.dot(af, bf) / denom) if denom else float("nan"),
        "mse": mse,
        "rmse": rmse,
        "rel_rmse": float(rmse / ref) if ref else float("nan"),
        "mae": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
    }


def copy_hfq_tensor(param: torch.nn.Parameter, arr: np.ndarray) -> torch.Tensor:
    original = param.detach().clone()
    src = torch.from_numpy(arr).to(device=param.device, dtype=param.dtype)
    with torch.no_grad():
        param.copy_(src)
    del src
    return original


def restore_tensor(param: torch.nn.Parameter, original: torch.Tensor) -> None:
    with torch.no_grad():
        param.copy_(original)


def main() -> None:
    args = parse_args()
    dtype = torch_dtype(args.dtype)
    if args.tokens:
        tokens = [int(x) for x in args.tokens.split(",") if x.strip()]
    elif args.ref:
        tokens = parse_ref_tokens(args.ref, args.chunk, args.seq_len)
    else:
        raise SystemExit("--tokens or --ref is required for tokenizer-free exact replay")
    if len(tokens) < 2:
        raise SystemExit("need at least two tokens")
    if len(tokens) > args.seq_len:
        tokens = tokens[: args.seq_len]

    print(f"loading HF oracle {args.hf_model} dtype={args.dtype} device={args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=not args.allow_remote,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(args.device)
    text_model = getattr(model, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise SystemExit("HF model does not expose model.layers")
    layers = parse_layers(args.layers, len(text_model.layers))
    hfq = Hfq(args.hfq_model)

    captures: dict[int, dict[str, Any]] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, hook_args, hook_kwargs, output):
            captures[layer_idx] = {
                "args": clone_tree(hook_args),
                "kwargs": clone_tree(hook_kwargs),
                "ref": tensor_output(output).detach().clone(),
            }
        return hook

    for layer_idx in layers:
        handles.append(text_model.layers[layer_idx].register_forward_hook(make_hook(layer_idx), with_kwargs=True))

    input_ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
    try:
        with torch.inference_mode():
            _ = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    missing = [x for x in layers if x not in captures]
    if missing:
        raise SystemExit(f"missing layer captures: {missing}")

    module_map = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def module_tensor_items(layer_idx: int) -> list[tuple[str, torch.nn.Module, dict[str, Any], bytes]]:
        prefix = f"model.layers.{layer_idx}."
        items = []
        for module_name, module in module_map.items():
            if not module_name.startswith(prefix) or not hasattr(module, "weight"):
                continue
            state_key = module_name + ".weight"
            if args.tensor_filter and not any(f in state_key for f in args.tensor_filter):
                continue
            hfq_name = state_key_to_hfq_name(state_key)
            if hfq_name not in hfq.tensors:
                skipped.append({"state_key": state_key, "reason": "not_in_hfq"})
                continue
            info, data = hfq.tensor_bytes(hfq_name)
            if int(info["quant_type"]) in (1, 2) and not args.include_f16:
                continue
            if tuple(info["shape"]) != tuple(module.weight.shape):
                skipped.append({"state_key": state_key, "hfq_name": hfq_name, "reason": "shape_mismatch", "hfq_shape": info["shape"], "param_shape": list(module.weight.shape)})
                continue
            items.append((state_key, module, info, data))
        return items

    for layer_idx in layers:
        cap = captures[layer_idx]
        layer = text_model.layers[layer_idx]
        items = module_tensor_items(layer_idx)
        for state_key, module, info, data in items:
            original = None
            try:
                arr = dequant_tensor(info, data)
                original = copy_hfq_tensor(module.weight, arr)
                with torch.inference_mode():
                    out = tensor_output(layer(*cap["args"], **cap["kwargs"]))
                restore_tensor(module.weight, original)
                original = None
                row = {
                    "layer": layer_idx,
                    "state_key": state_key,
                    "hfq_name": state_key_to_hfq_name(state_key),
                    "quant_type": int(info["quant_type"]),
                    "shape": list(info["shape"]),
                    "all": metrics(out, cap["ref"]),
                    "last": metrics(out[0, -1], cap["ref"][0, -1]) if out.ndim == 3 else metrics(out[-1], cap["ref"][-1]),
                }
                rows.append(row)
                del arr, out
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as e:
                if original is not None:
                    restore_tensor(module.weight, original)
                skipped.append({"state_key": state_key, "reason": "exception", "error": str(e)})

        if args.layer_all and items:
            originals: list[tuple[torch.nn.Module, torch.Tensor]] = []
            try:
                for _state_key, module, info, data in items:
                    arr = dequant_tensor(info, data)
                    originals.append((module, copy_hfq_tensor(module.weight, arr)))
                    del arr
                with torch.inference_mode():
                    out = tensor_output(layer(*cap["args"], **cap["kwargs"]))
                rows.append({
                    "layer": layer_idx,
                    "state_key": f"model.layers.{layer_idx}.*",
                    "hfq_name": "layer_all_matching_tensors",
                    "quant_type": "mixed",
                    "shape": None,
                    "all": metrics(out, cap["ref"]),
                    "last": metrics(out[0, -1], cap["ref"][0, -1]) if out.ndim == 3 else metrics(out[-1], cap["ref"][-1]),
                })
                del out
            except Exception as e:
                skipped.append({"state_key": f"model.layers.{layer_idx}.*", "reason": "exception", "error": str(e)})
            finally:
                for module, original in reversed(originals):
                    restore_tensor(module.weight, original)
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    rows_sorted = sorted(rows, key=lambda r: r["last"]["rel_rmse"], reverse=True)
    result = {
        "schema": "hipfire.qwen35.torch_hfq_attribution.v0",
        "hf_model": args.hf_model,
        "hfq_model": args.hfq_model,
        "ref": args.ref,
        "chunk": args.chunk,
        "seq_len": len(tokens),
        "layers": layers,
        "dtype": args.dtype,
        "device": args.device,
        "rows": rows_sorted,
        "skipped": skipped,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))

    print("rank layer quant rel_last  rel_all   max_last   tensor")
    print("---- ----- ----- --------- --------- ---------- ----------------")
    for i, row in enumerate(rows_sorted[: args.max_print], 1):
        print(
            f"{i:4d} {row['layer']:5d} {str(row['quant_type']):>5s} "
            f"{row['last']['rel_rmse']:9.6f} {row['all']['rel_rmse']:9.6f} "
            f"{row['last']['max_abs']:10.6f} {row['state_key']}"
        )
    if skipped:
        print(f"skipped={len(skipped)} (see JSON for details)")


if __name__ == "__main__":
    main()
