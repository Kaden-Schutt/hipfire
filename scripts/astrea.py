#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Astrea: agent-native model calibration planning for hipfire.

Astrea is intentionally a Python CLI. The stable contract is JSON artifacts
that both humans and agents can inspect, rerun, and hand to Atlas.
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import math
import mmap
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None


INSPECT_SCHEMA = "hipfire.astrea.inspect.v0"
PLAN_SCHEMA = "hipfire.astrea.plan.v0"
CALIBRATION_SCHEMA = "hipfire.astrea.calibration.v0"
EVAL_SCHEMA = "hipfire.astrea.eval.v0"
METRICS_SCHEMA = "hipfire.astrea.metrics.v0"
REPORT_SCHEMA = "hipfire.astrea.report.v0"
ENGINE_SCHEMA = "hipfire.astrea.engine_fingerprint.v0"
POLICY_SCHEMA = "hipfire.astrea.policy.v0"
PROMOTION_SCHEMA = "hipfire.astrea.promotion.v0"
KV_PROFILE_SCHEMA = "hipfire.astrea.kv_profile.v0"
BUNDLE_PLAN_SCHEMA = "hipfire.astrea.bundle_plan.v0"
GGUF_SUMMARY_SCHEMA = "hipfire.astrea.gguf_summary.v0"
HFQ_SUMMARY_SCHEMA = "hipfire.astrea.hfq_summary.v0"
IMATRIX_HFQ_JOIN_SCHEMA = "hipfire.astrea.imatrix_hfq_join.v0"
SAFETENSORS_SUMMARY_SCHEMA = "hipfire.astrea.safetensors_summary.v0"
PARO_PROBE_SCHEMA = "hipfire.astrea.paro_probe.v0"
PARO_IMPORT_SCHEMA = "hipfire.astrea.paro_import.v0"

SUPPORTED_FORMATS = {
    "mq3",
    "mq4",
    "mq6",
    "hfq4",
    "hfq6",
    "hfp4",
    "mfp4",
    "paro4",
    "q8",
    "f16",
}

SUPPORTED_METHODS = {
    "awq",
    "mq4-ls",
    "gptq",
    "mse",
    "minmax",
    "percentile",
    "imatrix",
    "imatrix-scale",
    "imatrix-kmap",
    "kmap",
    "fwht",
    "quarot",
    "paroquant",
    "roundtrip-eval",
    "gptq-probe",
    "awq-probe",
    "moe-probe",
    "model-ingress",
    "policy-search",
}

RECIPE_STAGES = {
    "ingress",
    "probe",
    "scale_search",
    "activation_aware",
    "rounding",
    "promotion",
    "transform",
    "eval",
}

DEFAULT_METHOD_STAGE = {
    "mse": "scale_search",
    "minmax": "scale_search",
    "percentile": "scale_search",
    "imatrix": "scale_search",
    "imatrix-scale": "scale_search",
    "awq": "activation_aware",
    "mq4-ls": "scale_search",
    "awq-probe": "activation_aware",
    "gptq": "rounding",
    "gptq-probe": "rounding",
    "kmap": "promotion",
    "imatrix-kmap": "promotion",
    "fwht": "transform",
    "quarot": "transform",
    "paroquant": "transform",
    "roundtrip-eval": "eval",
    "moe-probe": "probe",
    "model-ingress": "ingress",
    "policy-search": "promotion",
}

SUPPORTED_POLICY_OBJECTIVES = {
    "dynamic-tensor-policy",
    "moe-probe",
    "model-ingress",
    "kv-policy",
}

SUPPORTED_POLICY_DOMAINS = {
    "weights",
    "kv",
}

SUPPORTED_KV_MODES = {
    "fp16",
    "q8",
    "asym2",
    "asym3",
    "asym4",
    "triattn",
    "cask",
    "turbo",
    "turbo2",
    "turbo3",
    "turbo4",
    "rotor",
    "planar",
    "iso",
}

SUPPORTED_BUNDLE_INCLUDES = {
    "weights",
    "paro",
    "kv-policy",
    "triattn",
    "evidence",
}

ENGINE_HASH_PATHS = [
    "crates/rdna-compute/src/dispatch.rs",
    "crates/rdna-compute/src/kernels.rs",
    "crates/hipfire-arch-qwen35/src/qwen35.rs",
    "crates/hipfire-runtime/examples/eval_hipfire.rs",
    "kernels/src/rope_partial_interleaved.hip",
    "kernels/src/rope_partial_interleaved_batched.hip",
    "kernels/src/rope_partial_halfsplit.hip",
    "kernels/src/rope_partial_halfsplit_batched.hip",
]

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    30: "BF16",
}

HFQ_QUANT_TYPE_NAMES = {
    0: "Q4F16G64",
    1: "F16",
    2: "F32",
    3: "Q8F16",
    4: "Q4_K",
    5: "Q8HFQ",
    6: "HFQ4G256",
    7: "HFQ4G128",
    8: "HFQ6G256",
    9: "HFQ2G256",
    10: "HFQ2G128",
    11: "HFQ3G256",
    12: "HFQ3G128",
    13: "MQ4G256",
    14: "MQ8G256",
    17: "MQ3G256",
    18: "MQ2G256",
    19: "MQ2G256_LLOYD",
    20: "MQ3G256_LLOYD",
    21: "HFP4G32",
    24: "MFP4G32",
}

HFQ_QUANT_TYPE_FORMATS = {
    "F16": "f16",
    "Q8F16": "q8",
    "Q8HFQ": "q8",
    "MQ8G256": "q8",
    "HFQ4G256": "hfq4",
    "HFQ4G128": "hfq4",
    "HFQ6G256": "hfq6",
    "MQ4G256": "mq4",
    "MQ3G256": "mq3",
    "MQ3G256_LLOYD": "mq3",
    "HFP4G32": "hfp4",
    "MFP4G32": "mfp4",
}


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def md5_file(path):
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_summary(path):
    if not path:
        return None
    p = Path(path)
    item = {
        "path": str(p),
        "exists": p.exists(),
        "is_file": p.is_file(),
        "bytes": None,
        "md5": None,
    }
    if p.is_file():
        item["bytes"] = p.stat().st_size
        item["md5"] = md5_file(p)
    return item


class BinaryReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        if self.pos + n > len(self.data):
            raise ValueError("unexpected EOF while reading GGUF")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        values = struct.unpack(fmt, self.read(size))
        return values[0] if len(values) == 1 else values

    def u8(self):
        return self.unpack("<B")

    def i8(self):
        return self.unpack("<b")

    def u16(self):
        return self.unpack("<H")

    def i16(self):
        return self.unpack("<h")

    def u32(self):
        return self.unpack("<I")

    def i32(self):
        return self.unpack("<i")

    def u64(self):
        return self.unpack("<Q")

    def i64(self):
        return self.unpack("<q")

    def f32(self):
        return self.unpack("<f")

    def f64(self):
        return self.unpack("<d")

    def string(self):
        length = self.u64()
        return self.read(length).decode("utf-8")


def read_gguf_value(reader, value_type):
    if value_type == 0:
        return reader.u8()
    if value_type == 1:
        return reader.i8()
    if value_type == 2:
        return reader.u16()
    if value_type == 3:
        return reader.i16()
    if value_type == 4:
        return reader.u32()
    if value_type == 5:
        return reader.i32()
    if value_type == 6:
        return reader.f32()
    if value_type == 7:
        return bool(reader.u8())
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        elem_type = reader.u32()
        count = reader.u64()
        values = [read_gguf_value(reader, elem_type) for _ in range(min(count, 64))]
        for _ in range(count - len(values)):
            read_gguf_value(reader, elem_type)
        if count > len(values):
            return {"values": values, "truncated": True, "count": count}
        return values
    if value_type == 10:
        return reader.u64()
    if value_type == 11:
        return reader.i64()
    if value_type == 12:
        return reader.f64()
    raise ValueError(f"unknown GGUF metadata type {value_type}")


def json_object_end(data):
    depth = 0
    in_string = False
    escape = False
    for i, b in enumerate(data):
        if escape:
            escape = False
            continue
        if b == ord("\\") and in_string:
            escape = True
            continue
        if b == ord('"'):
            in_string = not in_string
            continue
        if in_string:
            continue
        if b == ord("{"):
            depth += 1
        elif b == ord("}"):
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("could not find end of JSON object")


def summarize_gguf(path, *, max_tensors=32):
    data = Path(path).read_bytes()
    reader = BinaryReader(data)
    magic = reader.read(4)
    if magic != b"GGUF":
        raise ValueError("not a GGUF file")
    version = reader.u32()
    tensor_count = reader.u64()
    metadata_kv_count = reader.u64()

    metadata = {}
    for _ in range(metadata_kv_count):
        key = reader.string()
        value_type = reader.u32()
        metadata[key] = read_gguf_value(reader, value_type)

    tensors = []
    dtype_counts = {}
    all_names = []
    imatrix_suffix_counts = {}
    imatrix_logical_names = set()
    imatrix_logical_tensors = {}
    for i in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        shape = [reader.u64() for _ in range(n_dims)]
        dtype_raw = reader.u32()
        dtype = GGML_TYPE_NAMES.get(dtype_raw, f"UNKNOWN_{dtype_raw}")
        offset = reader.u64()
        all_names.append(name)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        if name.endswith(".in_sum2"):
            logical_name = name[: -len(".in_sum2")]
            imatrix_suffix_counts["in_sum2"] = imatrix_suffix_counts.get("in_sum2", 0) + 1
            imatrix_logical_names.add(logical_name)
            imatrix_logical_tensors.setdefault(logical_name, {})["in_sum2"] = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "offset": offset,
            }
        elif name.endswith(".counts"):
            logical_name = name[: -len(".counts")]
            imatrix_suffix_counts["counts"] = imatrix_suffix_counts.get("counts", 0) + 1
            imatrix_logical_names.add(logical_name)
            imatrix_logical_tensors.setdefault(logical_name, {})["counts"] = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "offset": offset,
            }
        else:
            imatrix_suffix_counts["other"] = imatrix_suffix_counts.get("other", 0) + 1
            imatrix_logical_names.add(name)
            imatrix_logical_tensors.setdefault(name, {})["other"] = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "offset": offset,
            }
        if i < max_tensors:
            tensors.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "offset": offset,
                }
            )

    alignment = int(metadata.get("general.alignment", 32) or 32)
    tensor_data_offset = ((reader.pos + alignment - 1) // alignment) * alignment
    names_md5 = hashlib.md5("\n".join(all_names).encode("utf-8")).hexdigest()
    return {
        "schema": GGUF_SUMMARY_SCHEMA,
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
        "metadata": metadata,
        "tensor_data_offset": tensor_data_offset,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "tensor_names_md5": names_md5,
        "imatrix_logical_tensor_count": len(imatrix_logical_names),
        "imatrix_logical_names": sorted(imatrix_logical_names),
        "imatrix_logical_tensors": {
            name: imatrix_logical_tensors[name]
            for name in sorted(imatrix_logical_tensors)
        },
        "imatrix_suffix_counts": dict(sorted(imatrix_suffix_counts.items())),
        "tensors": tensors,
        "tensors_truncated": tensor_count > len(tensors),
    }


def metadata_summary(metadata):
    summary = {"keys": sorted(metadata.keys())}
    if "architecture" in metadata:
        summary["architecture"] = metadata["architecture"]
    config = metadata.get("config")
    if isinstance(config, dict):
        keep = [
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
        ]
        summary["config"] = {key: config[key] for key in keep if key in config}
    return summary


def read_hfq_index(path, *, max_tensors=32):
    p = Path(path)
    with p.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if len(mm) < 32:
                raise ValueError("HFQ file is smaller than the 32-byte header")
            magic = mm[0:4]
            if magic != b"HFQM":
                raise ValueError("not an HFQ file")
            version = struct.unpack_from("<I", mm, 4)[0]
            arch_id = struct.unpack_from("<I", mm, 8)[0]
            n_tensors = struct.unpack_from("<I", mm, 12)[0]
            metadata_offset = struct.unpack_from("<Q", mm, 16)[0]
            data_offset = struct.unpack_from("<Q", mm, 24)[0]
            if metadata_offset > data_offset or data_offset > len(mm):
                raise ValueError("invalid HFQ metadata/data offsets")

            metadata_bytes = mm[metadata_offset:data_offset]
            json_end = json_object_end(metadata_bytes)
            metadata_text = metadata_bytes[:json_end].decode("utf-8")
            metadata = json.loads(metadata_text)

            pos = metadata_offset + json_end
            if pos + 4 > data_offset:
                raise ValueError("missing HFQ tensor index")
            idx_n = struct.unpack_from("<I", mm, pos)[0]
            pos += 4
            if idx_n != n_tensors:
                raise ValueError(f"HFQ index count {idx_n} does not match header count {n_tensors}")

            tensors = []
            tensor_map = {}
            all_names = []
            quant_type_counts = {}
            cumulative_offset = data_offset
            for i in range(n_tensors):
                name_len = struct.unpack_from("<H", mm, pos)[0]
                pos += 2
                name = mm[pos : pos + name_len].decode("utf-8")
                pos += name_len
                quant_type = mm[pos]
                pos += 1
                n_dims = mm[pos]
                pos += 1
                shape = []
                for _ in range(n_dims):
                    shape.append(struct.unpack_from("<I", mm, pos)[0])
                    pos += 4
                group_size = struct.unpack_from("<I", mm, pos)[0]
                pos += 4
                data_size = struct.unpack_from("<Q", mm, pos)[0]
                pos += 8

                quant_type_name = HFQ_QUANT_TYPE_NAMES.get(quant_type, f"UNKNOWN_{quant_type}")
                item = {
                    "name": name,
                    "quant_type": quant_type,
                    "quant_type_name": quant_type_name,
                    "shape": shape,
                    "group_size": group_size,
                    "data_offset": cumulative_offset,
                    "data_size": data_size,
                }
                all_names.append(name)
                tensor_map[name] = item
                quant_type_counts[quant_type_name] = quant_type_counts.get(quant_type_name, 0) + 1
                if i < max_tensors:
                    tensors.append(dict(item))
                cumulative_offset += data_size

            names_md5 = hashlib.md5("\n".join(all_names).encode("utf-8")).hexdigest()
            summary = {
                "schema": HFQ_SUMMARY_SCHEMA,
                "magic": magic.decode("ascii"),
                "version": version,
                "arch_id": arch_id,
                "tensor_count": n_tensors,
                "metadata_offset": metadata_offset,
                "data_offset": data_offset,
                "data_end": cumulative_offset,
                "file_bytes": len(mm),
                "data_end_matches_file_size": cumulative_offset == len(mm),
                "metadata": metadata_summary(metadata),
                "quant_type_counts": dict(sorted(quant_type_counts.items())),
                "tensor_names_md5": names_md5,
                "tensors": tensors,
                "tensors_truncated": n_tensors > len(tensors),
            }
            return summary, tensor_map


def summarize_hfq(path, *, max_tensors=32):
    summary, _ = read_hfq_index(path, max_tensors=max_tensors)
    return summary


def imatrix_logical_name(name):
    if name.endswith(".in_sum2"):
        return name[: -len(".in_sum2")]
    if name.endswith(".counts"):
        return name[: -len(".counts")]
    return name


def gguf_to_hfq_candidates(gguf_name):
    top_level = {
        "token_embd.weight": [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ],
        "output.weight": ["lm_head.weight"],
        "output_norm.weight": [
            "model.language_model.norm.weight",
            "model.norm.weight",
        ],
    }
    if gguf_name in top_level:
        return top_level[gguf_name]

    if not gguf_name.startswith("blk."):
        return [gguf_name]
    rest = gguf_name[len("blk.") :]
    dot = rest.find(".")
    if dot < 0:
        return [gguf_name]
    layer_idx = rest[:dot]
    slot_full = rest[dot + 1 :]
    slot = slot_full[: -len(".weight")] if slot_full.endswith(".weight") else slot_full
    slot_map = {
        "attn_norm": ["input_layernorm"],
        "ffn_norm": ["post_attention_layernorm"],
        "attn_q": ["self_attn.q_proj"],
        "attn_k": ["self_attn.k_proj"],
        "attn_v": ["self_attn.v_proj"],
        "attn_output": ["self_attn.o_proj"],
        "attn_q_norm": ["self_attn.q_norm"],
        "attn_k_norm": ["self_attn.k_norm"],
        "ffn_gate": ["mlp.gate_proj"],
        "ffn_up": ["mlp.up_proj"],
        "ffn_down": ["mlp.down_proj"],
        # Qwen3.5 hybrid linear-attention aliases emitted by llama.cpp imatrix.
        "attn_gate": ["linear_attn.in_proj_z"],
        "attn_qkv": ["linear_attn.in_proj_qkv"],
        "ssm_alpha": ["linear_attn.in_proj_a"],
        "ssm_beta": ["linear_attn.in_proj_b"],
        "ssm_out": ["linear_attn.out_proj"],
    }
    translated = slot_map.get(slot, [slot])
    candidates = []
    for prefix in ("model.language_model.layers", "model.layers"):
        for item in translated:
            candidates.append(f"{prefix}.{layer_idx}.{item}.weight")
    return candidates


def gguf_expert_to_hfq_candidates(gguf_name, expert_index):
    if not gguf_name.startswith("blk."):
        return []
    rest = gguf_name[len("blk.") :]
    dot = rest.find(".")
    if dot < 0:
        return []
    layer_idx = rest[:dot]
    slot_full = rest[dot + 1 :]
    slot = slot_full[: -len(".weight")] if slot_full.endswith(".weight") else slot_full
    slot_map = {
        "ffn_gate_exps": ["gate_up_proj"],
        "ffn_up_exps": ["gate_up_proj"],
        "ffn_down_exps": ["down_proj"],
    }
    translated = slot_map.get(slot)
    if not translated:
        return []
    candidates = []
    for prefix in ("model.language_model.layers", "model.layers"):
        for item in translated:
            candidates.append(f"{prefix}.{layer_idx}.mlp.experts.{expert_index}.{item}.weight")
    return candidates


def hfq_k_dim(tensor):
    shape = tensor.get("shape") or []
    if len(shape) >= 2:
        return int(shape[-1])
    if shape:
        return int(shape[0])
    return None


def awq_runtime_eligible(name):
    return (
        name.endswith("q_proj.weight")
        or name.endswith("k_proj.weight")
        or name.endswith("v_proj.weight")
        or name.endswith("qkv_proj.weight")
        or name.endswith("wqkv.weight")
        or name.endswith("gate_proj.weight")
        or name.endswith("up_proj.weight")
        or name.endswith("w_gate.weight")
        or name.endswith("w_up.weight")
        or name.endswith("gate_up_proj.weight")
        or ".in_proj_" in name
        or name.endswith("mlp.gate.weight")
        or name.endswith("router.weight")
    )


def imatrix_slot(logical_name):
    parts = logical_name.split(".")
    if logical_name.startswith("blk.") and len(parts) > 2:
        return parts[2]
    return "top_level"


def imatrix_dims(logical_info):
    in_sum2 = logical_info.get("in_sum2") or {}
    shape = in_sum2.get("shape") or []
    k = int(shape[0]) if shape else None
    columns = int(shape[1]) if len(shape) >= 2 else 1
    return k, columns


def match_imatrix_to_hfq(model, imatrix, *, max_tensors=32):
    hfq_summary, hfq_tensors = read_hfq_index(model, max_tensors=max_tensors)
    imatrix_summary = summarize_gguf(imatrix, max_tensors=max_tensors)

    matches = []
    unmatched = []
    skipped = []
    k_dim_mismatches = []
    expert_coverage = []
    matched_quant_type_counts = {}
    matched_by_slot = {}
    for logical_name in imatrix_summary["imatrix_logical_names"]:
        logical_info = imatrix_summary.get("imatrix_logical_tensors", {}).get(logical_name, {})
        in_sum2 = logical_info.get("in_sum2")
        if not in_sum2:
            skipped.append({"imatrix_name": logical_name, "reason": "missing_in_sum2"})
            continue
        if in_sum2.get("dtype") != "F32":
            skipped.append({
                "imatrix_name": logical_name,
                "reason": "non_f32_in_sum2",
                "dtype": in_sum2.get("dtype"),
            })
            continue

        imatrix_k, imatrix_columns = imatrix_dims(logical_info)
        slot = imatrix_slot(logical_name)
        is_expert_matrix = imatrix_columns > 1 or slot.endswith("_exps")
        if is_expert_matrix:
            matched_experts = []
            missing_experts = []
            for expert_index in range(imatrix_columns):
                candidates = gguf_expert_to_hfq_candidates(logical_name, expert_index)
                if not candidates:
                    skipped.append({
                        "imatrix_name": logical_name,
                        "reason": "unsupported_multi_matrix_name",
                        "shape": in_sum2.get("shape"),
                    })
                    break
                hfq_name = next((name for name in candidates if name in hfq_tensors), None)
                if hfq_name is None:
                    missing_experts.append(expert_index)
                    unmatched.append({
                        "imatrix_name": logical_name,
                        "expert_index": expert_index,
                        "candidates": candidates,
                    })
                    continue
                tensor = hfq_tensors[hfq_name]
                quant_type_name = tensor["quant_type_name"]
                matched_quant_type_counts[quant_type_name] = matched_quant_type_counts.get(quant_type_name, 0) + 1
                matched_by_slot[slot] = matched_by_slot.get(slot, 0) + 1
                tensor_k = hfq_k_dim(tensor)
                k_dim_match = imatrix_k is None or tensor_k == imatrix_k
                if not k_dim_match:
                    k_dim_mismatches.append({
                        "imatrix_name": logical_name,
                        "expert_index": expert_index,
                        "hfq_name": hfq_name,
                        "imatrix_k": imatrix_k,
                        "hfq_k": tensor_k,
                    })
                matched_experts.append(expert_index)
                matches.append(
                    {
                        "imatrix_name": logical_name,
                        "hfq_name": hfq_name,
                        "expert_index": expert_index,
                        "quant_type": tensor["quant_type"],
                        "quant_type_name": quant_type_name,
                        "shape": tensor["shape"],
                        "group_size": tensor["group_size"],
                        "data_offset": tensor["data_offset"],
                        "data_size": tensor["data_size"],
                        "imatrix_shape": in_sum2.get("shape"),
                        "counts_shape": (logical_info.get("counts") or {}).get("shape"),
                        "imatrix_k": imatrix_k,
                        "imatrix_columns": imatrix_columns,
                        "hfq_k": tensor_k,
                        "k_dim_match": k_dim_match,
                        "awq_eligible": awq_runtime_eligible(hfq_name),
                        "calibration_supported": False,
                    }
                )
            if imatrix_columns > 1:
                expert_coverage.append({
                    "imatrix_name": logical_name,
                    "slot": slot,
                    "expert_count": imatrix_columns,
                    "matched_expert_count": len(matched_experts),
                    "missing_expert_count": len(missing_experts),
                    "matched_experts": matched_experts[:32],
                    "missing_experts": missing_experts[:32],
                    "truncated": len(matched_experts) > 32 or len(missing_experts) > 32,
                })
            continue

        candidates = gguf_to_hfq_candidates(logical_name)
        hfq_name = next((name for name in candidates if name in hfq_tensors), None)
        if hfq_name is None:
            unmatched.append({"imatrix_name": logical_name, "candidates": candidates})
            continue
        tensor = hfq_tensors[hfq_name]
        quant_type_name = tensor["quant_type_name"]
        matched_quant_type_counts[quant_type_name] = matched_quant_type_counts.get(quant_type_name, 0) + 1
        matched_by_slot[slot] = matched_by_slot.get(slot, 0) + 1
        tensor_k = hfq_k_dim(tensor)
        k_dim_match = imatrix_k is None or tensor_k == imatrix_k
        if not k_dim_match:
            k_dim_mismatches.append({
                "imatrix_name": logical_name,
                "hfq_name": hfq_name,
                "imatrix_k": imatrix_k,
                "hfq_k": tensor_k,
            })
        matches.append(
            {
                "imatrix_name": logical_name,
                "hfq_name": hfq_name,
                "quant_type": tensor["quant_type"],
                "quant_type_name": quant_type_name,
                "shape": tensor["shape"],
                "group_size": tensor["group_size"],
                "data_offset": tensor["data_offset"],
                "data_size": tensor["data_size"],
                "imatrix_shape": in_sum2.get("shape"),
                "counts_shape": (logical_info.get("counts") or {}).get("shape"),
                "imatrix_k": imatrix_k,
                "imatrix_columns": imatrix_columns,
                "hfq_k": tensor_k,
                "k_dim_match": k_dim_match,
                "awq_eligible": awq_runtime_eligible(hfq_name),
                "calibration_supported": True,
            }
        )

    return {
        "schema": IMATRIX_HFQ_JOIN_SCHEMA,
        "captured_at_utc": utc_now(),
        "model": str(model),
        "imatrix": str(imatrix),
        "hfq": {
            "tensor_count": hfq_summary["tensor_count"],
            "quant_type_counts": hfq_summary["quant_type_counts"],
            "tensor_names_md5": hfq_summary["tensor_names_md5"],
            "data_end_matches_file_size": hfq_summary["data_end_matches_file_size"],
        },
        "imatrix_tensor_count": imatrix_summary["tensor_count"],
        "imatrix_logical_tensor_count": imatrix_summary["imatrix_logical_tensor_count"],
        "imatrix_suffix_counts": imatrix_summary["imatrix_suffix_counts"],
        "matched_count": len(matches),
        "unmatched_count": len(unmatched),
        "skipped_count": len(skipped),
        "k_dim_mismatch_count": len(k_dim_mismatches),
        "awq_eligible_count": sum(1 for item in matches if item.get("awq_eligible")),
        "matched_quant_type_counts": dict(sorted(matched_quant_type_counts.items())),
        "matched_by_slot": dict(sorted(matched_by_slot.items())),
        "expert_coverage": expert_coverage,
        "k_dim_mismatches": k_dim_mismatches,
        "matches": matches,
        "unmatched": unmatched,
        "skipped": skipped,
        "ready": len(matches) > 0 and len(unmatched) == 0 and len(skipped) == 0 and len(k_dim_mismatches) == 0,
    }


def read_safetensors_index(path):
    p = Path(path)
    with p.open("rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise ValueError(f"{p} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            continue
        tensors[name] = {
            "name": name,
            "dtype": meta.get("dtype"),
            "shape": meta.get("shape", []),
            "data_offsets": meta.get("data_offsets", []),
            "file": str(p),
        }
    return tensors


def read_safetensors_dir_index(path, *, max_tensors=32):
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"source dir does not exist or is not a directory: {root}")
    files = sorted(p for p in root.iterdir() if p.name.endswith(".safetensors"))
    tensors = {}
    dtype_counts = {}
    preview = []
    for file_path in files:
        file_tensors = read_safetensors_index(file_path)
        for name, tensor in sorted(file_tensors.items()):
            tensors[name] = tensor
            dtype = tensor.get("dtype") or "UNKNOWN"
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            if len(preview) < max_tensors:
                preview.append(dict(tensor))
    names_md5 = hashlib.md5("\n".join(sorted(tensors)).encode("utf-8")).hexdigest()
    summary = {
        "schema": SAFETENSORS_SUMMARY_SCHEMA,
        "path": str(root),
        "file_count": len(files),
        "files": [str(p) for p in files],
        "tensor_count": len(tensors),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "tensor_names_md5": names_md5,
        "tensors": preview,
        "tensors_truncated": len(tensors) > len(preview),
    }
    return summary, tensors


def summarize_safetensors_dir(path, *, max_tensors=32):
    summary, _ = read_safetensors_dir_index(path, max_tensors=max_tensors)
    return summary


E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def f32_to_f16_bits(value):
    bits = f32_bits(value)
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        f16_frac = 0 if frac == 0 else ((frac >> 13) | 1)
        return ((sign << 15) | (0x1F << 10) | f16_frac) & 0xFFFF
    new_exp = int(exp) - 127 + 15
    if new_exp >= 31:
        return ((sign << 15) | (0x1F << 10)) & 0xFFFF
    if new_exp <= 0:
        if new_exp < -10:
            return (sign << 15) & 0xFFFF
        f = frac | 0x800000
        shift = 1 - new_exp + 13
        return ((sign << 15) | (f >> shift)) & 0xFFFF
    return ((sign << 15) | (new_exp << 10) | (frac >> 13)) & 0xFFFF


def bf16_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def gen_fwht_signs(seed, n=256):
    state = seed
    signs = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        signs.append(1.0 if ((state >> 16) & 1) == 1 else -1.0)
    return signs


FWHT_SIGNS1 = gen_fwht_signs(42)
FWHT_SIGNS2 = gen_fwht_signs(1042)


def cpu_fwht_256(values):
    if len(values) != 256:
        raise ValueError(f"FWHT requires 256 values, got {len(values)}")
    x = [float(values[i]) * FWHT_SIGNS1[i] for i in range(256)]
    stride = 1
    while stride < 256:
        i = 0
        while i < 256:
            for j in range(stride):
                a = x[i + j]
                b = x[i + j + stride]
                x[i + j] = a + b
                x[i + j + stride] = a - b
            i += stride * 2
        stride <<= 1
    return [x[i] * 0.0625 * FWHT_SIGNS2[i] for i in range(256)]


def e2m1_round(value):
    best_idx = 0
    best_err = float("inf")
    for i, code in enumerate(E2M1_LUT):
        err = abs(code - value)
        if err < best_err:
            best_err = err
            best_idx = i
    return best_idx


def weighted_abs_quantile(row, importance, quantile):
    if not importance or quantile >= 1.0:
        return None
    pairs = sorted((abs(v), max(float(w), 0.0)) for v, w in zip(row, importance))
    total = sum(w for _, w in pairs)
    if total <= 0.0:
        return None
    target = total * quantile
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= target:
            return value
    return pairs[-1][0] if pairs else None


def quantize_hfp4g32_row(row, *, format_flags=0):
    k = len(row)
    if k % 32 != 0:
        raise ValueError(f"HFP4G32 requires K%32==0, got K={k}")
    n_blocks = k // 32
    out = bytearray(16 + n_blocks * 17)
    row_max_abs = max((abs(float(v)) for v in row), default=0.0)
    row_scale_a = row_max_abs / 6.0 if row_max_abs > 0.0 else 1.0
    inv_row_scale = 1.0 / row_scale_a if row_max_abs > 0.0 else 0.0

    out[0:2] = struct.pack("<H", f32_to_f16_bits(row_scale_a))
    out[2:4] = struct.pack("<H", 0)
    out[4:6] = struct.pack("<H", n_blocks)
    out[6] = format_flags
    out[7] = 0

    for b in range(n_blocks):
        block = row[b * 32 : (b + 1) * 32]
        block_max_abs = max((abs(float(v)) for v in block), default=0.0)
        block_max_normalized = block_max_abs * inv_row_scale
        if block_max_normalized > 0.0:
            e_signed = math.ceil(math.log2(block_max_normalized / 6.0)) + 127
            block_e = max(0, min(254, e_signed))
        else:
            block_e = 0
        block_scale_factor = math.exp2(block_e - 127)
        inv_block_scale = 1.0 / block_scale_factor if block_scale_factor > 0.0 else 0.0

        payload_off = 16 + b * 17
        out[payload_off] = block_e
        for i in range(16):
            lo = float(block[2 * i]) * inv_row_scale * inv_block_scale
            hi = float(block[2 * i + 1]) * inv_row_scale * inv_block_scale
            out[payload_off + 1 + i] = e2m1_round(lo) | (e2m1_round(hi) << 4)
    return bytes(out)


def quantize_mfp4g32_values(values, *, importance=None, clip_quantile=0.999):
    if not values:
        return b""
    k = len(values[0])
    if k % 256 != 0:
        raise ValueError(f"MFP4G32 requires K%256==0, got K={k}")
    if any(len(row) != k for row in values):
        raise ValueError("all MFP4 rows must have the same K")
    if importance is not None and len(importance) != k:
        raise ValueError(f"imatrix vector length {len(importance)} does not match K={k}")

    out = bytearray()
    for row in values:
        row_buf = [float(v) for v in row]
        threshold = weighted_abs_quantile(row_buf, importance, clip_quantile)
        if threshold is not None:
            row_buf = [max(-threshold, min(threshold, v)) for v in row_buf]
        rotated = []
        for seg in range(k // 256):
            rotated.extend(cpu_fwht_256(row_buf[seg * 256 : (seg + 1) * 256]))
        out.extend(quantize_hfp4g32_row(rotated, format_flags=0x05))
    return bytes(out)


def f32_to_f16_bits_numpy(values):
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    sign = (bits >> np.uint32(31)) & np.uint32(1)
    exp = (bits >> np.uint32(23)) & np.uint32(0xFF)
    frac = bits & np.uint32(0x7FFFFF)
    out = np.zeros(bits.shape, dtype=np.uint16)

    inf_nan = exp == 0xFF
    if np.any(inf_nan):
        f16_frac = np.where(frac == 0, 0, (frac >> np.uint32(13)) | np.uint32(1))
        out[inf_nan] = (
            (sign[inf_nan] << np.uint32(15))
            | np.uint32(0x1F << 10)
            | f16_frac[inf_nan]
        ).astype(np.uint16)

    finite = ~inf_nan
    new_exp = exp.astype(np.int32) - 127 + 15
    overflow = finite & (new_exp >= 31)
    if np.any(overflow):
        out[overflow] = ((sign[overflow] << np.uint32(15)) | np.uint32(0x1F << 10)).astype(np.uint16)

    normal = finite & (new_exp > 0) & (new_exp < 31)
    if np.any(normal):
        out[normal] = (
            (sign[normal] << np.uint32(15))
            | (new_exp[normal].astype(np.uint32) << np.uint32(10))
            | (frac[normal] >> np.uint32(13))
        ).astype(np.uint16)

    subnormal = finite & (new_exp <= 0) & (new_exp >= -10)
    if np.any(subnormal):
        f = frac[subnormal] | np.uint32(0x800000)
        shifts = (1 - new_exp[subnormal] + 13).astype(np.uint32)
        out[subnormal] = (
            (sign[subnormal] << np.uint32(15))
            | (f >> shifts)
        ).astype(np.uint16)

    underflow = finite & (new_exp < -10)
    if np.any(underflow):
        out[underflow] = (sign[underflow] << np.uint32(15)).astype(np.uint16)

    return out


def weighted_abs_quantile_numpy(rows, importance, quantile):
    if importance is None or quantile >= 1.0:
        return None
    weights = np.asarray(importance, dtype=np.float32)
    if weights.shape != (rows.shape[1],):
        raise ValueError(f"imatrix vector length {weights.shape[0]} does not match K={rows.shape[1]}")
    if float(weights.sum()) <= 0.0:
        return None
    abs_rows = np.abs(rows)
    order = np.argsort(abs_rows, axis=1)
    sorted_abs = np.take_along_axis(abs_rows, order, axis=1)
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, axis=1)
    targets = cumulative[:, -1] * quantile
    idx = (cumulative >= targets[:, None]).argmax(axis=1)
    return sorted_abs[np.arange(rows.shape[0]), idx]


def fwht_256_numpy(values):
    x = np.asarray(values, dtype=np.float32).reshape(-1, 256).copy()
    x *= np.asarray(FWHT_SIGNS1, dtype=np.float32)
    stride = 1
    while stride < 256:
        for i in range(0, 256, stride * 2):
            a = x[:, i : i + stride].copy()
            b = x[:, i + stride : i + 2 * stride].copy()
            x[:, i : i + stride] = a + b
            x[:, i + stride : i + 2 * stride] = a - b
        stride <<= 1
    x *= np.float32(0.0625)
    x *= np.asarray(FWHT_SIGNS2, dtype=np.float32)
    return x.reshape(values.shape)


def inverse_fwht_256_numpy(values):
    x = np.asarray(values, dtype=np.float32).reshape(-1, 256).copy()
    x *= np.asarray(FWHT_SIGNS2, dtype=np.float32)
    stride = 1
    while stride < 256:
        for i in range(0, 256, stride * 2):
            a = x[:, i : i + stride].copy()
            b = x[:, i + stride : i + 2 * stride].copy()
            x[:, i : i + stride] = a + b
            x[:, i + stride : i + 2 * stride] = a - b
        stride <<= 1
    x *= np.float32(0.0625)
    x *= np.asarray(FWHT_SIGNS1, dtype=np.float32)
    return x.reshape(values.shape)


def clipped_rows_for_ratio(rows, clip_ratio):
    rows = np.asarray(rows, dtype=np.float32)
    ratio = float(clip_ratio)
    if ratio >= 1.0:
        return rows.copy()
    if ratio <= 0.0:
        raise ValueError(f"clip ratio must be positive, got {clip_ratio}")
    row_max_abs = np.max(np.abs(rows), axis=1)
    thresholds = row_max_abs * np.float32(ratio)
    return np.clip(rows, -thresholds[:, None], thresholds[:, None]).astype(np.float32, copy=False)


def mq4_affine_ls_refit_numpy(rotated, q, scale, zero, *, iters=3):
    """Refit MQ4 affine block scale/zero by least squares for fixed nibbles.

    Min/max is robust but not MSE-optimal. For the existing MQ4 wire format we
    can improve the block fit without changing kernels by alternating between
    (a) solving `rotated ~= scale * q + zero` for each block and (b) requantizing
    to the nearest 4-bit code under that affine fit.
    """
    r = np.asarray(rotated, dtype=np.float32)
    q_u8 = np.asarray(q, dtype=np.uint8)
    best_scale = np.asarray(scale, dtype=np.float32).copy()
    best_zero = np.asarray(zero, dtype=np.float32).copy()
    best_q = q_u8.copy()
    for _ in range(max(1, int(iters))):
        qf = best_q.astype(np.float32)
        mean_q = np.mean(qf, axis=1)
        mean_r = np.mean(r, axis=1)
        var_q = np.mean(qf * qf, axis=1) - mean_q * mean_q
        cov_qr = np.mean(qf * r, axis=1) - mean_q * mean_r
        candidate_scale = np.where(var_q > np.float32(1.0e-12), cov_qr / var_q, best_scale)
        candidate_zero = mean_r - candidate_scale * mean_q
        valid = np.isfinite(candidate_scale) & np.isfinite(candidate_zero) & (candidate_scale > 0.0)
        best_scale = np.where(valid, candidate_scale, best_scale).astype(np.float32)
        best_zero = np.where(valid, candidate_zero, best_zero).astype(np.float32)
        inv = np.where(best_scale > 0.0, np.float32(1.0) / best_scale, np.float32(0.0))
        best_q = np.clip(np.floor((r - best_zero[:, None]) * inv[:, None] + np.float32(0.5)), 0, 15).astype(np.uint8)
    return best_q, best_scale, best_zero


def mq4_quantized_rotated_blocks_numpy(rows, *, clip_ratio=1.0, fit="minmax", ls_iters=3):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("MQ4 values must be a 2D matrix")
    m, k = rows.shape
    if k % 256 != 0:
        raise ValueError(f"MQ4G256 requires K%256==0, got K={k}")
    clipped = clipped_rows_for_ratio(rows, clip_ratio)
    rotated = fwht_256_numpy(clipped.reshape(m, k // 256, 256)).reshape(-1, 256)
    min_val = np.min(rotated, axis=1).astype(np.float32)
    max_val = np.max(rotated, axis=1).astype(np.float32)
    value_range = (max_val - min_val).astype(np.float32)
    scale = np.where(value_range > 0.0, value_range / np.float32(15.0), np.float32(1.0)).astype(np.float32)
    inv_scale = np.where(value_range > 0.0, np.float32(1.0) / scale, np.float32(0.0)).astype(np.float32)
    q = np.floor((rotated - min_val[:, None]) * inv_scale[:, None] + np.float32(0.5))
    q = np.clip(q, 0, 15).astype(np.uint8)
    if fit == "minmax":
        return q, scale, min_val
    if fit == "ls":
        return mq4_affine_ls_refit_numpy(rotated, q, scale, min_val, iters=ls_iters)
    raise ValueError(f"unsupported MQ4 fit mode: {fit}")


def quantize_mq4g256_values_numpy(values, *, clip_ratio=1.0, fit="minmax", ls_iters=3):
    if np is None:
        raise RuntimeError("numpy is required for MQ4 calibration")
    rows = np.asarray(values, dtype=np.float32)
    m, k = rows.shape
    q, scale, min_val = mq4_quantized_rotated_blocks_numpy(
        rows, clip_ratio=clip_ratio, fit=fit, ls_iters=ls_iters
    )
    n_blocks = q.shape[0]
    out = np.zeros((n_blocks, 136), dtype=np.uint8)
    out[:, 0:4] = scale.astype("<f4").view(np.uint8).reshape(n_blocks, 4)
    out[:, 4:8] = min_val.astype("<f4").view(np.uint8).reshape(n_blocks, 4)
    out[:, 8:] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    expected_blocks = m * (k // 256)
    if n_blocks != expected_blocks:
        raise ValueError(f"MQ4 block count mismatch: {n_blocks} vs {expected_blocks}")
    return out.tobytes()


def dequantize_mq4g256_from_values_numpy(values, *, clip_ratio=1.0, fit="minmax", ls_iters=3):
    rows = np.asarray(values, dtype=np.float32)
    m, k = rows.shape
    q, scale, min_val = mq4_quantized_rotated_blocks_numpy(
        rows, clip_ratio=clip_ratio, fit=fit, ls_iters=ls_iters
    )
    rotated = q.astype(np.float32) * scale[:, None] + min_val[:, None]
    return inverse_fwht_256_numpy(rotated.reshape(m, k // 256, 256)).reshape(m, k)


def awq_mq4_clip_ratio_grid(extra_ratio=None):
    ratios = [1.0, 0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.925, 0.90]
    if extra_ratio is not None:
        try:
            value = float(extra_ratio)
            if value > 0.0:
                ratios.append(min(value, 1.0))
        except Exception:
            pass
    return sorted({round(value, 6) for value in ratios if value > 0.0}, reverse=True)


def select_awq_mq4_clip_ratio(values, importance, *, clip_ratio_grid=None, sample_rows=128):
    if np is None:
        raise RuntimeError("numpy is required for MQ4 AWQ calibration")
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("MQ4 AWQ values must be a 2D matrix")
    m, k = rows.shape
    if k % 256 != 0:
        raise ValueError(f"MQ4G256 requires K%256==0, got K={k}")
    if m > sample_rows:
        idx = np.linspace(0, m - 1, int(sample_rows), dtype=np.int64)
        sample = rows[idx, :]
    else:
        sample = rows
    weights = None
    if importance is not None:
        weights = np.asarray(importance, dtype=np.float32)
        if weights.shape != (k,):
            raise ValueError(f"imatrix vector length {weights.shape[0]} does not match K={k}")
        mean = float(np.mean(weights))
        if mean > 0.0:
            weights = weights / np.float32(mean)
        else:
            weights = None
    best_ratio = None
    best_mse = float("inf")
    grid = clip_ratio_grid or awq_mq4_clip_ratio_grid()
    for ratio in grid:
        deq = dequantize_mq4g256_from_values_numpy(sample, clip_ratio=ratio)
        err = (deq - sample) ** 2
        if weights is not None:
            err = err * weights[None, :]
        mse = float(np.mean(err))
        if mse < best_mse:
            best_mse = mse
            best_ratio = float(ratio)
    return best_ratio if best_ratio is not None else 1.0, best_mse


def e2m1_round_numpy(values):
    x = np.asarray(values, dtype=np.float32)
    abs_x = np.abs(x)
    idx = np.zeros(x.shape, dtype=np.uint8)
    idx = np.where(abs_x > 0.25, 1, idx)
    idx = np.where(abs_x > 0.75, 2, idx)
    idx = np.where(abs_x > 1.25, 3, idx)
    idx = np.where(abs_x > 1.75, 4, idx)
    idx = np.where(abs_x > 2.5, 5, idx)
    idx = np.where(abs_x > 3.5, 6, idx)
    idx = np.where(abs_x > 5.0, 7, idx)
    negative = (x < 0.0) & (idx > 0)
    return (idx + np.where(negative, 8, 0).astype(np.uint8)).astype(np.uint8)


def quantize_mfp4g32_values_numpy(values, *, importance=None, clip_quantile=0.999):
    if np is None:
        raise RuntimeError("numpy is required for the vectorized MFP4 path")
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("MFP4 values must be a 2D matrix")
    m, k = rows.shape
    if k % 256 != 0:
        raise ValueError(f"MFP4G32 requires K%256==0, got K={k}")

    thresholds = weighted_abs_quantile_numpy(rows, importance, clip_quantile)
    if thresholds is not None:
        rows = np.clip(rows, -thresholds[:, None], thresholds[:, None])
    else:
        rows = rows.copy()

    rotated = fwht_256_numpy(rows.reshape(m, k // 256, 256)).reshape(m, k)
    n_blocks = k // 32
    row_bytes = 16 + 17 * n_blocks
    out = np.zeros((m, row_bytes), dtype=np.uint8)

    row_max_abs = np.max(np.abs(rotated), axis=1)
    row_scale = np.where(row_max_abs > 0.0, row_max_abs / 6.0, 1.0).astype(np.float32)
    inv_row_scale = np.where(row_max_abs > 0.0, 1.0 / row_scale, 0.0).astype(np.float32)
    row_scale_bits = f32_to_f16_bits_numpy(row_scale)
    out[:, 0] = (row_scale_bits & 0xFF).astype(np.uint8)
    out[:, 1] = (row_scale_bits >> 8).astype(np.uint8)
    out[:, 4] = n_blocks & 0xFF
    out[:, 5] = (n_blocks >> 8) & 0xFF
    out[:, 6] = 0x05

    blocks = rotated.reshape(m, n_blocks, 32)
    block_max_abs = np.max(np.abs(blocks), axis=2)
    block_max_normalized = block_max_abs * inv_row_scale[:, None]
    block_e = np.zeros((m, n_blocks), dtype=np.uint8)
    mask = block_max_normalized > 0.0
    if np.any(mask):
        e = np.ceil(np.log2(block_max_normalized[mask] / 6.0)).astype(np.int32) + 127
        block_e[mask] = np.clip(e, 0, 254).astype(np.uint8)
    block_scale = np.exp2(block_e.astype(np.int32) - 127).astype(np.float32)
    values_scaled = blocks * inv_row_scale[:, None, None] / block_scale[:, :, None]
    nibbles = e2m1_round_numpy(values_scaled)
    packed = nibbles[:, :, 0::2] | (nibbles[:, :, 1::2] << 4)
    for b in range(n_blocks):
        off = 16 + b * 17
        out[:, off] = block_e[:, b]
        out[:, off + 1 : off + 17] = packed[:, b, :]
    return out.tobytes()


def quantize_q8f16_values(values):
    flat = [float(value) for row in values for value in row]
    group_size = 32
    block_bytes = 34
    n_blocks = ceil_div(len(flat), group_size)
    out = bytearray(n_blocks * block_bytes)
    for b in range(n_blocks):
        start = b * group_size
        end = min(start + group_size, len(flat))
        group = flat[start:end]
        max_abs = max((abs(value) for value in group), default=0.0)
        scale = max_abs / 127.0 if max_abs > 0.0 else 0.0
        inv_scale = 1.0 / scale if scale > 0.0 else 0.0
        off = b * block_bytes
        out[off : off + 2] = struct.pack("<H", f32_to_f16_bits(scale))
        for i in range(group_size):
            value = flat[start + i] if start + i < len(flat) else 0.0
            q = int(round(value * inv_scale)) if inv_scale > 0.0 else 0
            q = max(-128, min(127, q))
            out[off + 2 + i] = q & 0xFF
    return bytes(out)


def quantize_q8f16_values_numpy(values):
    if np is None:
        raise RuntimeError("numpy is required for the vectorized Q8F16 path")
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    group_size = 32
    block_bytes = 34
    n_blocks = ceil_div(flat.size, group_size)
    padded = np.zeros(n_blocks * group_size, dtype=np.float32)
    padded[: flat.size] = flat
    blocks = padded.reshape(n_blocks, group_size)
    max_abs = np.max(np.abs(blocks), axis=1)
    scale = np.where(max_abs > 0.0, max_abs / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    inv_scale = np.where(scale > 0.0, np.float32(1.0) / scale, np.float32(0.0)).astype(np.float32)
    q = np.rint(blocks * inv_scale[:, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    out = np.zeros((n_blocks, block_bytes), dtype=np.uint8)
    scale_bits = f32_to_f16_bits_numpy(scale)
    out[:, 0] = (scale_bits & 0xFF).astype(np.uint8)
    out[:, 1] = (scale_bits >> 8).astype(np.uint8)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def parse_gguf_index(path):
    data = Path(path).read_bytes()
    reader = BinaryReader(data)
    if reader.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = reader.u32()
    tensor_count = reader.u64()
    metadata_kv_count = reader.u64()
    metadata = {}
    for _ in range(metadata_kv_count):
        key = reader.string()
        value_type = reader.u32()
        metadata[key] = read_gguf_value(reader, value_type)
    tensors = {}
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        shape = [reader.u64() for _ in range(n_dims)]
        dtype_raw = reader.u32()
        offset = reader.u64()
        tensors[name] = {
            "name": name,
            "shape": shape,
            "dtype_raw": dtype_raw,
            "dtype": GGML_TYPE_NAMES.get(dtype_raw, f"UNKNOWN_{dtype_raw}"),
            "offset": offset,
        }
    alignment = int(metadata.get("general.alignment", 32) or 32)
    tensor_data_offset = ((reader.pos + alignment - 1) // alignment) * alignment
    return {
        "version": version,
        "metadata": metadata,
        "tensor_data_offset": tensor_data_offset,
        "tensors": tensors,
        "data": data,
    }


def read_gguf_f32_tensor(path, name):
    index = parse_gguf_index(path)
    tensor = index["tensors"].get(name)
    if tensor is None:
        raise ValueError(f"GGUF tensor not found: {name}")
    if tensor["dtype"] != "F32":
        raise ValueError(f"GGUF tensor {name} is {tensor['dtype']}, expected F32")
    count = 1
    for dim in tensor["shape"]:
        count *= int(dim)
    start = index["tensor_data_offset"] + tensor["offset"]
    end = start + count * 4
    raw = index["data"][start:end]
    if len(raw) != count * 4:
        raise ValueError(f"GGUF tensor {name} is truncated")
    return list(struct.unpack(f"<{count}f", raw))


def load_safetensors_values(source_tensors, name):
    tensor = source_tensors.get(name)
    if tensor is None:
        raise ValueError(f"source tensor not found: {name}")
    shape = tensor.get("shape") or []
    if len(shape) != 2:
        raise ValueError(f"source tensor {name} must be 2D, got shape={shape}")
    offsets = tensor.get("data_offsets") or []
    if len(offsets) != 2:
        raise ValueError(f"source tensor {name} missing data offsets")
    file_path = Path(tensor["file"])
    with file_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + header_len + offsets[0])
        raw = f.read(offsets[1] - offsets[0])

    dtype = tensor.get("dtype")
    count = int(shape[0]) * int(shape[1])
    if dtype == "BF16":
        if len(raw) != count * 2:
            raise ValueError(f"source tensor {name} byte count mismatch for BF16")
        flat = [bf16_to_f32(struct.unpack_from("<H", raw, i * 2)[0]) for i in range(count)]
    elif dtype == "F16":
        if len(raw) != count * 2:
            raise ValueError(f"source tensor {name} byte count mismatch for F16")
        flat = [float(struct.unpack_from("<e", raw, i * 2)[0]) for i in range(count)]
    elif dtype == "F32":
        if len(raw) != count * 4:
            raise ValueError(f"source tensor {name} byte count mismatch for F32")
        flat = list(struct.unpack(f"<{count}f", raw))
    else:
        raise ValueError(f"unsupported source tensor dtype for {name}: {dtype}")

    k = int(shape[1])
    return [flat[i * k : (i + 1) * k] for i in range(int(shape[0]))]


def load_safetensors_array(source_tensors, name):
    if np is None:
        return None
    tensor = source_tensors.get(name)
    if tensor is None:
        raise ValueError(f"source tensor not found: {name}")
    shape = [int(dim) for dim in (tensor.get("shape") or [])]
    if not shape:
        raise ValueError(f"source tensor {name} missing shape")
    offsets = tensor.get("data_offsets") or []
    if len(offsets) != 2:
        raise ValueError(f"source tensor {name} missing data offsets")
    file_path = Path(tensor["file"])
    with file_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + header_len + offsets[0])
        raw = f.read(offsets[1] - offsets[0])

    dtype = tensor.get("dtype")
    count = 1
    for dim in shape:
        count *= dim
    if dtype == "BF16":
        if len(raw) != count * 2:
            raise ValueError(f"source tensor {name} byte count mismatch for BF16")
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
        arr = bits.view(np.float32)
    elif dtype == "F16":
        if len(raw) != count * 2:
            raise ValueError(f"source tensor {name} byte count mismatch for F16")
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        if len(raw) != count * 4:
            raise ValueError(f"source tensor {name} byte count mismatch for F32")
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported source tensor dtype for {name}: {dtype}")
    return arr.reshape(tuple(shape))


def copy_candidate_file(model, candidate):
    src = Path(model)
    dst = Path(candidate)
    if src.resolve() == dst.resolve():
        raise ValueError("candidate output must not be the same path as the input model")
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["cp", "--reflink=auto", "--sparse=always", str(src), str(dst)],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        shutil.copy2(src, dst)


def read_hfq_layout(path):
    p = Path(path)
    data = p.read_bytes()
    if len(data) < 32 or data[0:4] != b"HFQM":
        raise ValueError("not an HFQ file")
    version = struct.unpack_from("<I", data, 4)[0]
    arch_id = struct.unpack_from("<I", data, 8)[0]
    n_tensors = struct.unpack_from("<I", data, 12)[0]
    metadata_offset = struct.unpack_from("<Q", data, 16)[0]
    data_offset = struct.unpack_from("<Q", data, 24)[0]
    metadata_region = data[metadata_offset:data_offset]
    json_end = json_object_end(metadata_region)
    metadata_bytes = bytes(metadata_region[:json_end])
    pos = metadata_offset + json_end
    idx_n = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    if idx_n != n_tensors:
        raise ValueError(f"HFQ index count {idx_n} does not match header count {n_tensors}")
    records = []
    cumulative_offset = data_offset
    for _ in range(n_tensors):
        name_len = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        quant_type = data[pos]
        pos += 1
        n_dims = data[pos]
        pos += 1
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack_from("<I", data, pos)[0])
            pos += 4
        group_size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        data_size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        payload = bytes(data[cumulative_offset : cumulative_offset + data_size])
        if len(payload) != data_size:
            raise ValueError(f"HFQ tensor {name} payload is truncated")
        quant_type_name = HFQ_QUANT_TYPE_NAMES.get(quant_type, f"UNKNOWN_{quant_type}")
        records.append(
            {
                "name": name,
                "quant_type": quant_type,
                "quant_type_name": quant_type_name,
                "shape": shape,
                "group_size": group_size,
                "data_size": data_size,
                "data": payload,
            }
        )
        cumulative_offset += data_size
    if cumulative_offset != len(data):
        raise ValueError(f"HFQ data end {cumulative_offset} does not match file size {len(data)}")
    return {
        "version": version,
        "arch_id": arch_id,
        "metadata_bytes": metadata_bytes,
        "records": records,
    }


def write_hfq_layout(path, layout):
    records = layout["records"]
    metadata = layout["metadata_bytes"]
    index = bytearray()
    index += struct.pack("<I", len(records))
    payloads = []
    for record in records:
        payload = record["data"]
        if len(payload) != int(record["data_size"]):
            raise ValueError(f"payload size mismatch for {record['name']}")
        raw_name = record["name"].encode("utf-8")
        index += struct.pack("<H", len(raw_name))
        index += raw_name
        index += struct.pack("<B", int(record["quant_type"]))
        index += struct.pack("<B", len(record["shape"]))
        for dim in record["shape"]:
            index += struct.pack("<I", int(dim))
        index += struct.pack("<I", int(record["group_size"]))
        index += struct.pack("<Q", int(record["data_size"]))
        payloads.append(payload)
    metadata_offset = 32
    data_offset = metadata_offset + len(metadata) + len(index)
    out = bytearray()
    out += b"HFQM"
    out += struct.pack("<I", int(layout["version"]))
    out += struct.pack("<I", int(layout["arch_id"]))
    out += struct.pack("<I", len(records))
    out += struct.pack("<Q", metadata_offset)
    out += struct.pack("<Q", data_offset)
    out += metadata
    out += index
    for payload in payloads:
        out += payload
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)


def q8f16_data_size_for_shape(shape):
    return ceil_div(tensor_element_count(shape), 32) * 34


ROTATED_RUNTIME_BASE_FORMATS = {"mq2", "mq3", "mq4", "mq6", "mfp4"}


def runtime_promotion_bundles_enabled(base_format, promotion_format):
    return base_format in ROTATED_RUNTIME_BASE_FORMATS and promotion_format in {"q8", "f16"}


def runtime_promotion_anchor_for_name(name, *, base_format, promotion_format):
    if not runtime_promotion_bundles_enabled(base_format, promotion_format):
        return None
    replacements = [
        (".self_attn.k_proj.weight", ".self_attn.q_proj.weight"),
        (".self_attn.v_proj.weight", ".self_attn.q_proj.weight"),
        (".linear_attn.in_proj_z.weight", ".linear_attn.in_proj_qkv.weight"),
        (".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_qkv.weight"),
        (".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_qkv.weight"),
        (".mlp.up_proj.weight", ".mlp.gate_proj.weight"),
    ]
    for suffix, anchor_suffix in replacements:
        if name.endswith(suffix):
            return name[: -len(suffix)] + anchor_suffix
    return None


def runtime_bundle_summary(base_format, promotion_format, added_anchors):
    return {
        "enabled": runtime_promotion_bundles_enabled(base_format, promotion_format),
        "added_anchor_count": len(added_anchors),
        "added_anchors": added_anchors,
        "rules": [
            "self_attn k/v require q anchor",
            "linear_attn z/a/b require qkv anchor",
            "mlp up requires gate anchor",
        ],
    }


def expand_runtime_promotion_selection(selected, available_items, *, base_format, promotion_format, strict=True):
    expanded = []
    expanded_names = set()
    added_anchors = []

    def add_item(item, role=None, trigger=None, anchor=None):
        name = item.get("hfq_name")
        if not name or name in expanded_names:
            return
        copy = dict(item)
        if role:
            copy["runtime_bundle_role"] = role
        if trigger:
            copy["runtime_bundle_trigger"] = trigger
        if anchor:
            copy["runtime_bundle_anchor"] = anchor
        expanded.append(copy)
        expanded_names.add(name)

    for item in selected:
        name = item.get("hfq_name")
        if not name:
            continue
        anchor = runtime_promotion_anchor_for_name(
            name,
            base_format=base_format,
            promotion_format=promotion_format,
        )
        if anchor and anchor not in expanded_names:
            anchor_item = available_items.get(anchor)
            if anchor_item is None:
                if strict:
                    raise ValueError(f"selected tensor {name} requires runtime anchor {anchor}, but it is unavailable")
            else:
                add_item(anchor_item, role="anchor", trigger=name)
                added_anchors.append({"anchor": anchor, "trigger": name})
        add_item(item, anchor=anchor)

    return expanded, runtime_bundle_summary(base_format, promotion_format, added_anchors)


def selected_policy_items(policy, *, max_tensors=None, tensor_filter=None):
    filters = [item.strip() for item in (tensor_filter or "").split(",") if item.strip()]
    selected = []
    for item in policy.get("selected") or []:
        name = item.get("hfq_name")
        if not name:
            continue
        if filters and not any(f in name or f in str(item.get("sensitivity_alias", "")) for f in filters):
            continue
        selected.append(item)
        if max_tensors is not None and len(selected) >= max_tensors:
            break
    return selected


def quantize_source_tensor_to_q8(source_tensors, record):
    name = record["name"]
    values_array = load_safetensors_array(source_tensors, name)
    expected_shape = tuple(int(dim) for dim in record["shape"])
    if values_array is not None:
        if tuple(values_array.shape) != expected_shape:
            raise ValueError(f"source shape mismatch for {name}: {values_array.shape} vs {expected_shape}")
        return quantize_q8f16_values_numpy(values_array)
    values = load_safetensors_values(source_tensors, name)
    if (len(values), len(values[0]) if values else 0) != expected_shape:
        raise ValueError(f"source shape mismatch for {name}")
    return quantize_q8f16_values(values)


def write_policy_promotion_candidate(
    policy,
    *,
    source_dir,
    output,
    max_tensors=None,
    tensor_filter=None,
):
    if policy.get("promotion_format") != "q8":
        raise ValueError("policy promotion writer currently supports promotion_format=q8 only")
    model = policy.get("model")
    if not model:
        raise ValueError("policy is missing model path")
    requested_selected = selected_policy_items(policy, max_tensors=max_tensors, tensor_filter=tensor_filter)
    if not requested_selected:
        raise ValueError("policy has no selected tensors to promote")
    layout = read_hfq_layout(model)
    available_items = {record["name"]: {"hfq_name": record["name"]} for record in layout["records"]}
    for item in (policy.get("selected") or []) + (policy.get("skipped") or []):
        name = item.get("hfq_name")
        if name:
            available_items[name] = item
    selected, bundle_summary = expand_runtime_promotion_selection(
        requested_selected,
        available_items,
        base_format=policy.get("base_format"),
        promotion_format=policy.get("promotion_format"),
    )
    selected_names = {item["hfq_name"] for item in selected}
    source_summary, source_tensors = read_safetensors_dir_index(source_dir)
    promoted = []
    missing = sorted(name for name in selected_names if name not in source_tensors)
    if missing:
        raise ValueError(f"source tensors missing for selected policy entries: {missing[:8]}")

    for record in layout["records"]:
        if record["name"] not in selected_names:
            continue
        tensor_format = HFQ_QUANT_TYPE_FORMATS.get(record["quant_type_name"], "unknown")
        if tensor_format != policy.get("base_format"):
            raise ValueError(
                f"selected tensor {record['name']} is {tensor_format}, expected {policy.get('base_format')}"
            )
        packed = quantize_source_tensor_to_q8(source_tensors, record)
        expected_size = q8f16_data_size_for_shape(record["shape"])
        if len(packed) != expected_size:
            raise ValueError(f"Q8F16 packed size mismatch for {record['name']}: {len(packed)} vs {expected_size}")
        old = {
            "quant_type": record["quant_type"],
            "quant_type_name": record["quant_type_name"],
            "group_size": record["group_size"],
            "data_size": record["data_size"],
        }
        record["quant_type"] = 3
        record["quant_type_name"] = "Q8F16"
        record["group_size"] = 32
        record["data_size"] = len(packed)
        record["data"] = packed
        promoted.append(
            {
                "hfq_name": record["name"],
                "shape": record["shape"],
                "old": old,
                "new": {
                    "quant_type": record["quant_type"],
                    "quant_type_name": record["quant_type_name"],
                    "group_size": record["group_size"],
                    "data_size": record["data_size"],
                },
                "extra_bytes": record["data_size"] - old["data_size"],
            }
        )

    if len(promoted) != len(selected_names):
        promoted_names = {item["hfq_name"] for item in promoted}
        missing_model = sorted(selected_names - promoted_names)
        raise ValueError(f"selected tensors missing from HFQ model: {missing_model[:8]}")

    write_hfq_layout(output, layout)
    return {
        "schema": PROMOTION_SCHEMA,
        "captured_at_utc": utc_now(),
        "policy_id": policy.get("policy_id"),
        "status": "candidate_written",
        "model": model,
        "candidate": str(output),
        "candidate_bytes": Path(output).stat().st_size,
        "source": {
            "path": source_summary["path"],
            "file_count": source_summary["file_count"],
            "tensor_count": source_summary["tensor_count"],
            "dtype_counts": source_summary["dtype_counts"],
            "tensor_names_md5": source_summary["tensor_names_md5"],
        },
        "base_format": policy.get("base_format"),
        "promotion_format": policy.get("promotion_format"),
        "requested_selected_count": len(requested_selected),
        "effective_selected_count": len(selected),
        "runtime_promotion_bundles": bundle_summary,
        "promoted_tensor_count": len(promoted),
        "promoted_extra_bytes": sum(item["extra_bytes"] for item in promoted),
        "promotions": promoted,
        "next_step": "run KLD/PPL against BF16, then Atlas AR/DFlash perf gates if quality improves",
    }


def patch_hfq_tensor(candidate, tensor_info, packed):
    if len(packed) != tensor_info["data_size"]:
        raise ValueError(
            f"packed tensor size mismatch for {tensor_info['name']}: "
            f"{len(packed)} vs {tensor_info['data_size']}"
        )
    with Path(candidate).open("r+b") as f:
        f.seek(tensor_info["data_offset"])
        f.write(packed)


def patch_hfq_tensor_from_file(candidate, tensor_info, patch_path):
    patch_path = Path(patch_path)
    if patch_path.stat().st_size != tensor_info["data_size"]:
        raise ValueError(
            f"packed tensor size mismatch for {tensor_info['name']}: "
            f"{patch_path.stat().st_size} vs {tensor_info['data_size']}"
        )
    with Path(candidate).open("r+b") as dst, patch_path.open("rb") as src:
        dst.seek(tensor_info["data_offset"])
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def select_imatrix_scale_matches(join, *, max_tensors=None, tensor_filter=None):
    filters = [item.strip() for item in (tensor_filter or "").split(",") if item.strip()]
    selected = []
    for item in join["matches"]:
        if not item.get("calibration_supported", True):
            continue
        if item["quant_type_name"] != "MFP4G32":
            continue
        if filters and not any(f in item["imatrix_name"] or f in item["hfq_name"] for f in filters):
            continue
        selected.append(item)
        if max_tensors is not None and len(selected) >= max_tensors:
            break
    return selected


def select_awq_mq4_matches(join, *, max_tensors=None, tensor_filter=None):
    filters = [item.strip() for item in (tensor_filter or "").split(",") if item.strip()]
    selected = []
    for item in join["matches"]:
        if not item.get("calibration_supported", True):
            continue
        if item["quant_type_name"] != "MQ4G256":
            continue
        if filters and not any(f in item["imatrix_name"] or f in item["hfq_name"] for f in filters):
            continue
        selected.append(item)
        if max_tensors is not None and len(selected) >= max_tensors:
            break
    return selected


def build_imatrix_scale_patch(task):
    item = task["item"]
    hfq_name = item["hfq_name"]
    k = item["shape"][1]
    importance = read_gguf_f32_tensor(task["imatrix"], f"{item['imatrix_name']}.in_sum2")
    counts = read_gguf_f32_tensor(task["imatrix"], f"{item['imatrix_name']}.counts")
    denom = counts[0] if counts and counts[0] > 0.0 else 1.0
    importance = [max(v / denom, 0.0) for v in importance]
    source_tensors = {hfq_name: task["source_tensor"]}
    values_array = load_safetensors_array(source_tensors, hfq_name)
    if values_array is not None:
        if values_array.shape != (item["shape"][0], k):
            raise ValueError(f"source shape mismatch for {hfq_name}")
        packed = quantize_mfp4g32_values_numpy(
            values_array,
            importance=importance,
            clip_quantile=task["clip_quantile"],
        )
    else:
        values = load_safetensors_values(source_tensors, hfq_name)
        if len(values) != item["shape"][0] or len(values[0]) != k:
            raise ValueError(f"source shape mismatch for {hfq_name}")
        packed = quantize_mfp4g32_values(
            values,
            importance=importance,
            clip_quantile=task["clip_quantile"],
        )
    patch_path = Path(task["patch_path"])
    patch_path.write_bytes(packed)
    return {
        "index": task["index"],
        "patch_path": str(patch_path),
        "mutation": {
            "imatrix_name": item["imatrix_name"],
            "hfq_name": hfq_name,
            "shape": item["shape"],
            "data_offset": task["tensor_info"]["data_offset"],
            "data_size": task["tensor_info"]["data_size"],
            "clip_quantile": task["clip_quantile"],
        },
    }


def run_patch_tasks(tasks, worker_fn, parallel_workers):
    if parallel_workers == 1:
        return [worker_fn(task) for task in tasks], "serial"

    def run_with_executor(executor_cls):
        by_index = {}
        with executor_cls(max_workers=parallel_workers) as pool:
            futures = {pool.submit(worker_fn, task): task["index"] for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                by_index[result["index"]] = result
        return [by_index[i] for i in range(len(tasks))]

    try:
        return run_with_executor(concurrent.futures.ProcessPoolExecutor), "process"
    except PermissionError:
        return run_with_executor(concurrent.futures.ThreadPoolExecutor), "thread"
    except OSError as exc:
        if getattr(exc, "errno", None) == 1:
            return run_with_executor(concurrent.futures.ThreadPoolExecutor), "thread"
        raise


def write_imatrix_scale_candidate(
    plan,
    join,
    source_tensors,
    *,
    max_tensors=None,
    tensor_filter=None,
    clip_quantile=0.999,
    workers=1,
):
    model = plan.get("model")
    candidate = plan.get("output")
    imatrix = plan.get("imatrix")
    if not candidate:
        raise ValueError("plan output is required when writing a candidate")
    copy_candidate_file(model, candidate)
    _, candidate_tensors = read_hfq_index(candidate, max_tensors=0)

    selected = select_imatrix_scale_matches(join, max_tensors=max_tensors, tensor_filter=tensor_filter)
    if not selected:
        raise ValueError("no MFP4G32 tensors selected for imatrix-scale calibration")

    patch_dir = Path(str(candidate) + ".patches")
    shutil.rmtree(patch_dir, ignore_errors=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    requested_workers = max(1, int(workers or 1))
    parallel_workers = min(requested_workers, len(selected))
    tasks = []
    for index, item in enumerate(selected):
        tensor_info = candidate_tensors[item["hfq_name"]]
        tasks.append(
            {
                "index": index,
                "item": item,
                "tensor_info": tensor_info,
                "source_tensor": source_tensors[item["hfq_name"]],
                "imatrix": imatrix,
                "clip_quantile": clip_quantile,
                "patch_path": str(patch_dir / f"{index:04d}.bin"),
            }
        )

    worker_mode = "serial"
    try:
        results, worker_mode = run_patch_tasks(tasks, build_imatrix_scale_patch, parallel_workers)

        mutations = []
        for result in results:
            patch_hfq_tensor_from_file(
                candidate,
                candidate_tensors[result["mutation"]["hfq_name"]],
                result["patch_path"],
            )
            mutations.append(result["mutation"])
    except Exception:
        shutil.rmtree(patch_dir, ignore_errors=True)
        raise
    shutil.rmtree(patch_dir, ignore_errors=True)

    return {
        "candidate": str(candidate),
        "candidate_bytes": Path(candidate).stat().st_size,
        "mutated_tensor_count": len(mutations),
        "parallel_workers": parallel_workers,
        "worker_mode": worker_mode,
        "mutations": mutations,
    }


def build_awq_mq4_patch(task):
    item = task["item"]
    hfq_name = item["hfq_name"]
    k = item["shape"][1]
    importance = read_gguf_f32_tensor(task["imatrix"], f"{item['imatrix_name']}.in_sum2")
    counts = read_gguf_f32_tensor(task["imatrix"], f"{item['imatrix_name']}.counts")
    denom = counts[0] if counts and counts[0] > 0.0 else 1.0
    importance = [max(v / denom, 0.0) for v in importance]
    source_tensors = {hfq_name: task["source_tensor"]}
    values_array = load_safetensors_array(source_tensors, hfq_name)
    if values_array is None:
        raise RuntimeError("numpy is required for MQ4 AWQ calibration")
    if values_array.shape != (item["shape"][0], k):
        raise ValueError(f"source shape mismatch for {hfq_name}")
    ratio_grid = task["clip_ratio_grid"]
    clip_ratio, weighted_mse = select_awq_mq4_clip_ratio(
        values_array,
        importance,
        clip_ratio_grid=ratio_grid,
        sample_rows=task["sample_rows"],
    )
    packed = quantize_mq4g256_values_numpy(values_array, clip_ratio=clip_ratio)
    if len(packed) != task["tensor_info"]["data_size"]:
        raise ValueError(
            f"packed MQ4 tensor size mismatch for {hfq_name}: "
            f"{len(packed)} vs {task['tensor_info']['data_size']}"
        )
    patch_path = Path(task["patch_path"])
    patch_path.write_bytes(packed)
    return {
        "index": task["index"],
        "patch_path": str(patch_path),
        "mutation": {
            "method": "awq",
            "imatrix_name": item["imatrix_name"],
            "hfq_name": hfq_name,
            "shape": item["shape"],
            "quant_type_name": item["quant_type_name"],
            "data_offset": task["tensor_info"]["data_offset"],
            "data_size": task["tensor_info"]["data_size"],
            "clip_ratio": clip_ratio,
            "clip_ratio_grid": ratio_grid,
            "weighted_mse": weighted_mse,
            "sample_rows": task["sample_rows"],
        },
    }


def build_mq4_ls_patch(task):
    item = task["item"]
    hfq_name = item["hfq_name"]
    k = item["shape"][1]
    source_tensors = {hfq_name: task["source_tensor"]}
    values_array = load_safetensors_array(source_tensors, hfq_name)
    if values_array is None:
        raise RuntimeError("numpy is required for MQ4 LS calibration")
    if values_array.shape != (item["shape"][0], k):
        raise ValueError(f"source shape mismatch for {hfq_name}")
    packed = quantize_mq4g256_values_numpy(
        values_array,
        clip_ratio=task["clip_ratio"],
        fit="ls",
        ls_iters=task["ls_iters"],
    )
    if len(packed) != task["tensor_info"]["data_size"]:
        raise ValueError(
            f"packed MQ4 tensor size mismatch for {hfq_name}: "
            f"{len(packed)} vs {task['tensor_info']['data_size']}"
        )
    sample = values_array
    if sample.shape[0] > task["sample_rows"]:
        idx = np.linspace(0, sample.shape[0] - 1, int(task["sample_rows"]), dtype=np.int64)
        sample = sample[idx, :]
    minmax_deq = dequantize_mq4g256_from_values_numpy(sample, fit="minmax")
    ls_deq = dequantize_mq4g256_from_values_numpy(
        sample,
        clip_ratio=task["clip_ratio"],
        fit="ls",
        ls_iters=task["ls_iters"],
    )
    minmax_mse = float(np.mean((minmax_deq - sample) ** 2))
    ls_mse = float(np.mean((ls_deq - sample) ** 2))
    patch_path = Path(task["patch_path"])
    patch_path.write_bytes(packed)
    return {
        "index": task["index"],
        "patch_path": str(patch_path),
        "mutation": {
            "method": "mq4-ls",
            "hfq_name": hfq_name,
            "shape": item["shape"],
            "quant_type_name": item["quant_type_name"],
            "data_offset": task["tensor_info"]["data_offset"],
            "data_size": task["tensor_info"]["data_size"],
            "ls_iters": task["ls_iters"],
            "clip_ratio": task["clip_ratio"],
            "sample_rows": task["sample_rows"],
            "sample_minmax_mse": minmax_mse,
            "sample_ls_mse": ls_mse,
            "sample_mse_delta_pct": ((minmax_mse - ls_mse) / minmax_mse) if minmax_mse > 0.0 else 0.0,
        },
    }


def write_mq4_ls_candidate(
    plan,
    join,
    source_tensors,
    *,
    max_tensors=None,
    tensor_filter=None,
    workers=1,
    ls_iters=3,
):
    model = plan.get("model")
    candidate = plan.get("output")
    clip_ratio = float(plan.get("clip_ratio", 1.0))
    if not candidate:
        raise ValueError("plan output is required when writing a candidate")
    copy_candidate_file(model, candidate)
    _, candidate_tensors = read_hfq_index(candidate, max_tensors=0)

    selected = select_awq_mq4_matches(join, max_tensors=max_tensors, tensor_filter=tensor_filter)
    if not selected:
        raise ValueError("no MQ4G256 tensors selected for LS calibration")

    patch_dir = Path(str(candidate) + ".patches")
    shutil.rmtree(patch_dir, ignore_errors=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    requested_workers = max(1, int(workers or 1))
    parallel_workers = min(requested_workers, len(selected))
    tasks = []
    for index, item in enumerate(selected):
        tensor_info = candidate_tensors[item["hfq_name"]]
        tasks.append(
            {
                "index": index,
                "item": item,
                "tensor_info": tensor_info,
                "source_tensor": source_tensors[item["hfq_name"]],
                "ls_iters": ls_iters,
                "clip_ratio": clip_ratio,
                "sample_rows": 128,
                "patch_path": str(patch_dir / f"{index:04d}.bin"),
            }
        )

    worker_mode = "serial"
    try:
        results, worker_mode = run_patch_tasks(tasks, build_mq4_ls_patch, parallel_workers)

        mutations = []
        for result in results:
            patch_hfq_tensor_from_file(
                candidate,
                candidate_tensors[result["mutation"]["hfq_name"]],
                result["patch_path"],
            )
            mutations.append(result["mutation"])
    except Exception:
        shutil.rmtree(patch_dir, ignore_errors=True)
        raise
    shutil.rmtree(patch_dir, ignore_errors=True)

    mean_delta = sum(m["sample_mse_delta_pct"] for m in mutations) / len(mutations)
    return {
        "candidate": str(candidate),
        "candidate_bytes": Path(candidate).stat().st_size,
        "mutated_tensor_count": len(mutations),
        "parallel_workers": parallel_workers,
        "worker_mode": worker_mode,
        "mean_sample_mse_delta_pct": mean_delta,
        "clip_ratio": clip_ratio,
        "mutations": mutations,
    }


def write_awq_mq4_candidate(
    plan,
    join,
    source_tensors,
    *,
    max_tensors=None,
    tensor_filter=None,
    clip_quantile=0.999,
    workers=1,
):
    model = plan.get("model")
    candidate = plan.get("output")
    imatrix = plan.get("imatrix")
    if not candidate:
        raise ValueError("plan output is required when writing a candidate")
    copy_candidate_file(model, candidate)
    _, candidate_tensors = read_hfq_index(candidate, max_tensors=0)

    selected = select_awq_mq4_matches(join, max_tensors=max_tensors, tensor_filter=tensor_filter)
    if not selected:
        raise ValueError("no MQ4G256 tensors selected for AWQ calibration")

    patch_dir = Path(str(candidate) + ".patches")
    shutil.rmtree(patch_dir, ignore_errors=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    requested_workers = max(1, int(workers or 1))
    parallel_workers = min(requested_workers, len(selected))
    ratio_grid = awq_mq4_clip_ratio_grid(clip_quantile)
    tasks = []
    for index, item in enumerate(selected):
        tensor_info = candidate_tensors[item["hfq_name"]]
        tasks.append(
            {
                "index": index,
                "item": item,
                "tensor_info": tensor_info,
                "source_tensor": source_tensors[item["hfq_name"]],
                "imatrix": imatrix,
                "clip_ratio_grid": ratio_grid,
                "sample_rows": 128,
                "patch_path": str(patch_dir / f"{index:04d}.bin"),
            }
        )

    worker_mode = "serial"
    try:
        results, worker_mode = run_patch_tasks(tasks, build_awq_mq4_patch, parallel_workers)

        mutations = []
        for result in results:
            patch_hfq_tensor_from_file(
                candidate,
                candidate_tensors[result["mutation"]["hfq_name"]],
                result["patch_path"],
            )
            mutations.append(result["mutation"])
    except Exception:
        shutil.rmtree(patch_dir, ignore_errors=True)
        raise
    shutil.rmtree(patch_dir, ignore_errors=True)

    return {
        "candidate": str(candidate),
        "candidate_bytes": Path(candidate).stat().st_size,
        "mutated_tensor_count": len(mutations),
        "parallel_workers": parallel_workers,
        "worker_mode": worker_mode,
        "awq_clip_ratio_grid": ratio_grid,
        "mutations": mutations,
    }


def git_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def git_value(root, *args):
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_rope_convention(root):
    root = Path(root)
    dispatch_path = root / "crates" / "rdna-compute" / "src" / "dispatch.rs"
    text = dispatch_path.read_text(encoding="utf-8", errors="ignore") if dispatch_path.exists() else ""
    has_halfsplit_source = (
        (root / "kernels" / "src" / "rope_partial_halfsplit.hip").exists()
        or (root / "kernels" / "src" / "rope_partial_halfsplit_batched.hip").exists()
    )
    dispatches_halfsplit = "rope_partial_halfsplit" in text
    has_legacy_escape = "HIPFIRE_ROPE_INTERLEAVED_LEGACY" in text
    if has_halfsplit_source and dispatches_halfsplit and has_legacy_escape:
        return "halfsplit"
    if "rope_partial_interleaved" in text or (root / "kernels" / "src" / "rope_partial_interleaved.hip").exists():
        return "interleaved_legacy"
    return "unknown"


def engine_fingerprint(engine_root=None):
    root = Path(engine_root) if engine_root else Path(__file__).resolve().parents[1]
    source_hashes = {}
    for rel in ENGINE_HASH_PATHS:
        path = root / rel
        if path.is_file():
            source_hashes[rel] = file_sha256(path)
    env_legacy = os.environ.get("HIPFIRE_ROPE_INTERLEAVED_LEGACY")
    dirty = git_value(root, "status", "--short")
    payload = {
        "schema": ENGINE_SCHEMA,
        "captured_at_utc": utc_now(),
        "root": str(root),
        "git": {
            "sha": git_value(root, "rev-parse", "HEAD"),
            "short": git_value(root, "rev-parse", "--short", "HEAD"),
            "branch": git_value(root, "branch", "--show-current"),
            "dirty": bool(dirty),
            "dirty_paths": dirty.splitlines()[:64] if dirty else [],
        },
        "rope_convention_default": detect_rope_convention(root),
        "env": {
            "HIPFIRE_ROPE_INTERLEAVED_LEGACY": env_legacy,
        },
        "source_hashes": source_hashes,
    }
    stable = {
        "git": payload["git"],
        "rope_convention_default": payload["rope_convention_default"],
        "source_hashes": payload["source_hashes"],
    }
    payload["fingerprint_id"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def infer_format(model_path):
    suffix = Path(model_path).suffix.lower().lstrip(".")
    if suffix in SUPPORTED_FORMATS:
        return suffix
    if suffix in {"hfq", "gguf"}:
        return suffix
    return "unknown"


def validate_values(values, supported, kind):
    unknown = [value for value in values if value not in supported]
    if unknown:
        raise ValueError(f"unsupported {kind}: {', '.join(unknown)}")


def parse_recipe_stage(text, order):
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"recipe stage must be stage:method, got {text!r}")
    stage, method = parts[0].strip(), parts[1].strip()
    if stage not in RECIPE_STAGES:
        raise ValueError(f"unsupported recipe stage: {stage}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported recipe method: {method}")
    item = {"order": order, "stage": stage, "method": method}
    if len(parts) == 3 and parts[2].strip():
        params = {}
        for pair in parts[2].split(","):
            if not pair.strip():
                continue
            if "=" not in pair:
                raise ValueError(f"recipe params must be key=value pairs, got {pair!r}")
            key, value = pair.split("=", 1)
            params[key.strip()] = value.strip()
        item["params"] = params
    return item


def build_recipe(methods, recipe_stages=None):
    if recipe_stages:
        recipe = [parse_recipe_stage(text, i) for i, text in enumerate(recipe_stages)]
        recipe_methods = [item["method"] for item in recipe]
        missing = [method for method in recipe_methods if method not in methods]
        if missing:
            raise ValueError(
                "recipe methods must also be listed with --method: "
                + ", ".join(missing)
            )
        return recipe
    recipe = []
    for i, method in enumerate(methods):
        stage = DEFAULT_METHOD_STAGE.get(method)
        if stage is None:
            raise ValueError(f"no default recipe stage for method: {method}")
        recipe.append({"order": i, "stage": stage, "method": method})
    return recipe


def inspect_model(model, *, imatrix=None, quant_format=None):
    result = {
        "schema": INSPECT_SCHEMA,
        "captured_at_utc": utc_now(),
        "host": socket.gethostname(),
        "git": git_sha(),
        "model": file_summary(model),
        "imatrix": file_summary(imatrix),
        "format": quant_format or infer_format(model),
    }
    if imatrix and Path(imatrix).is_file():
        try:
            with Path(imatrix).open("rb") as f:
                magic = f.read(4)
            if magic == b"GGUF":
                result["imatrix_gguf"] = summarize_gguf(imatrix)
        except Exception as exc:
            result["imatrix_gguf_error"] = str(exc)
    if model and Path(model).is_file():
        try:
            with Path(model).open("rb") as f:
                magic = f.read(4)
            if magic == b"HFQM":
                result["hfq"] = summarize_hfq(model)
        except Exception as exc:
            result["hfq_error"] = str(exc)
    return result


def build_plan(
    *,
    model,
    formats,
    methods,
    imatrix=None,
    calibration_corpus=None,
    source_dir=None,
    output=None,
    eval_commands=None,
    atlas_commands=None,
    recipe_stages=None,
    plan_id=None,
):
    if not formats:
        raise ValueError("at least one --format is required")
    if not methods:
        raise ValueError("at least one --method is required")
    validate_values(formats, SUPPORTED_FORMATS, "format")
    validate_values(methods, SUPPORTED_METHODS, "method")
    recipe = build_recipe(methods, recipe_stages=recipe_stages)

    model_name = Path(model).name.replace(".", "-").replace("/", "-")
    generated_plan_id = f"astrea-{model_name}-{int(time.time())}"
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": plan_id or generated_plan_id,
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "git": git_sha(),
        "intended_runners": ["human", "agent"],
        "model": model,
        "formats": formats,
        "methods": methods,
        "recipe": recipe,
        "recipe_stackable": len(recipe) > 1,
        "imatrix": imatrix,
        "calibration_corpus": calibration_corpus,
        "source_dir": source_dir,
        "output": output,
        "eval_commands": eval_commands or [],
        "atlas_commands": atlas_commands or [],
        "quality_gates": [
            "KLD or PPL must improve against a BF16 or accepted higher-precision reference before promotion.",
            "Matched-size comparisons must report KLD/PPL delta and output artifact size.",
            "Reports must include the reference model, dataset/chunk count, and eval command.",
        ],
        "runtime_constraints": [
            "Do not change an on-disk quant contract without matching runtime loader and kernel dispatch changes.",
            "HFP/MFP-family candidates must preserve the current fast-path block-size contract until producer and runtime move together.",
            "MQ-family imatrix work should focus on data-driven k-map/promotion sensitivity unless eval evidence shows within-group benefit.",
        ],
        "atlas_bridge": {
            "emit_rows": True,
            "required_when": "Run Atlas AR and DFlash perf rows before claiming a calibrated format is ship-ready.",
        },
        "notes": [
            "This plan is an experiment artifact. It does not mutate weights by itself.",
            "Calibration methods are stackable recipe stages; compare stacks empirically before promotion.",
        ],
    }


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload, pretty=False, out=None):
    text = (
        json.dumps(payload, indent=2, sort_keys=True)
        if pretty
        else json.dumps(payload, sort_keys=True)
    )
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        return
    if pretty:
        print(text)
    else:
        print(text)


def calibrate_plan(
    plan_path,
    *,
    dry_run=True,
    source_dir=None,
    write_candidate=False,
    max_tensors=None,
    tensor_filter=None,
    clip_quantile=0.999,
    workers=1,
):
    plan = load_json(plan_path)
    result = {
        "schema": CALIBRATION_SCHEMA,
        "captured_at_utc": utc_now(),
        "plan_id": plan.get("plan_id"),
        "status": "dry_run" if dry_run else "blocked",
        "candidate": plan.get("output"),
        "formats": plan.get("formats", []),
        "methods": plan.get("methods", []),
        "message": "weight mutation is not implemented in this scaffold; use this artifact to review the candidate plan",
    }
    model = plan.get("model")
    imatrix = plan.get("imatrix")
    source_dir = source_dir or plan.get("source_dir")
    if model and imatrix and Path(model).is_file() and Path(imatrix).is_file():
        try:
            join = match_imatrix_to_hfq(model, imatrix)
            result["join"] = join
            source_ready = None
            if source_dir:
                source_summary, source_tensors = read_safetensors_dir_index(source_dir)
                missing_source = [
                    item["hfq_name"]
                    for item in join["matches"]
                    if item["hfq_name"] not in source_tensors
                ]
                source_ready = len(missing_source) == 0 and len(join["matches"]) > 0
                result["source"] = {
                    "path": source_summary["path"],
                    "file_count": source_summary["file_count"],
                    "tensor_count": source_summary["tensor_count"],
                    "dtype_counts": source_summary["dtype_counts"],
                    "tensor_names_md5": source_summary["tensor_names_md5"],
                }
                result["source_match_count"] = len(join["matches"]) - len(missing_source)
                result["missing_source_count"] = len(missing_source)
                result["missing_source"] = missing_source[:32]
                result["source_ready"] = source_ready
                if write_candidate and source_ready and "imatrix-scale" in result["methods"] and "mfp4" in result["formats"]:
                    mutation = write_imatrix_scale_candidate(
                        plan,
                        join,
                        source_tensors,
                        max_tensors=max_tensors,
                        tensor_filter=tensor_filter,
                        clip_quantile=clip_quantile,
                        workers=workers,
                    )
                    result.update(mutation)
                    result["status"] = "candidate_written"
                    result["message"] = "wrote imatrix-scale MFP4 candidate with same HFQ tensor byte ranges"
                    result["next_step"] = "run KLD/PPL against BF16, then Atlas AR/DFlash perf gates"
                    return result
                if write_candidate and source_ready and "awq" in result["methods"] and "mq4" in result["formats"]:
                    mutation = write_awq_mq4_candidate(
                        plan,
                        join,
                        source_tensors,
                        max_tensors=max_tensors,
                        tensor_filter=tensor_filter,
                        clip_quantile=clip_quantile,
                        workers=workers,
                    )
                    result.update(mutation)
                    result["status"] = "candidate_written"
                    result["message"] = "wrote AWQ MQ4 candidate with same HFQ tensor byte ranges"
                    result["next_step"] = "run KLD/PPL against BF16, then Atlas AR/DFlash perf gates"
                    return result
                if write_candidate and source_ready and "mq4-ls" in result["methods"] and "mq4" in result["formats"]:
                    mutation = write_mq4_ls_candidate(
                        plan,
                        join,
                        source_tensors,
                        max_tensors=max_tensors,
                        tensor_filter=tensor_filter,
                        workers=workers,
                    )
                    result.update(mutation)
                    result["status"] = "candidate_written"
                    result["message"] = "wrote MQ4 LS candidate with same HFQ tensor byte ranges"
                    result["next_step"] = "run KLD/PPL against BF16, then Atlas AR/DFlash perf gates"
                    return result
            if dry_run and join["ready"]:
                if source_ready is False:
                    result["status"] = "blocked_missing_source_tensors"
                    result["message"] = "imatrix tensors match HFQ tensors, but BF16 source tensors are missing"
                    result["next_step"] = "point Astrea at the original BF16 safetensors source for this HFQ model"
                else:
                    result["status"] = "ready_for_calibration_dry_run"
                    result["message"] = "imatrix tensors match HFQ tensors; no weight bytes were mutated"
                    result["next_step"] = (
                        "implement weight mutation by re-quantizing matched tensors from BF16 source weights, "
                        "then run KLD/PPL and Atlas AR/DFlash gates"
                    )
            elif not join["ready"]:
                result["status"] = "blocked_unmatched_tensors"
                result["next_step"] = "fix imatrix-to-HFQ tensor aliases before attempting weight mutation"
        except Exception as exc:
            result["status"] = "blocked_join_error"
            result["join_error"] = str(exc)
            result["next_step"] = "inspect the HFQ model and imatrix files separately"
    else:
        result["next_step"] = "provide existing HFQ model and GGUF imatrix paths in the plan"
    return result


def eval_plan(plan_path, *, run=False):
    plan = load_json(plan_path)
    commands = list(plan.get("eval_commands") or [])
    results = []
    for command in commands:
        if not run:
            results.append({"command": command, "status": "not_run"})
            continue
        proc = subprocess.run(command, shell=True, text=True, capture_output=True)
        results.append(
            {
                "command": command,
                "status": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        )
    return {
        "schema": EVAL_SCHEMA,
        "captured_at_utc": utc_now(),
        "plan_id": plan.get("plan_id"),
        "ran_commands": run,
        "results": results,
    }


def load_quality_rows(path):
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif isinstance(payload, dict) and isinstance(payload.get("quality_rows"), list):
        rows = payload["quality_rows"]
    else:
        raise ValueError(f"unsupported quality JSON shape: {path}")
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        for key in ["mean_kld", "p99_kld", "ppl", "mean_kld_ci_lo", "mean_kld_ci_hi"]:
            if item.get(key) is not None:
                item[key] = float(item[key])
        if item.get("n_chunks") is not None:
            item["n_chunks"] = int(item["n_chunks"])
        normalized.append(item)
    return normalized


def select_quality_row(rows, variant, *, arch=None, scoring_mode=None):
    matches = []
    for row in rows:
        if row.get("variant") != variant:
            continue
        if arch and row.get("arch") != arch:
            continue
        if scoring_mode and row.get("scoring_mode") != scoring_mode:
            continue
        matches.append(row)
    if not matches:
        qualifiers = []
        if arch:
            qualifiers.append(f"arch={arch}")
        if scoring_mode:
            qualifiers.append(f"scoring_mode={scoring_mode}")
        suffix = " " + " ".join(qualifiers) if qualifiers else ""
        raise ValueError(f"quality row not found for variant {variant!r}{suffix}")
    if len(matches) > 1:
        raise ValueError(f"quality row is ambiguous for variant {variant!r}; pass --arch or --scoring-mode")
    return dict(matches[0])


def attach_above_floor(row, floor):
    item = dict(row)
    item["above_floor_kld"] = item["mean_kld"] - floor["mean_kld"]
    return item


def collect_metrics(
    *,
    quality_json,
    candidate_variant,
    baseline_variant=None,
    floor_variant=None,
    arch=None,
    scoring_mode=None,
    candidate_model=None,
    baseline_model=None,
    reference_model=None,
    dataset=None,
    engine_root=None,
):
    rows = load_quality_rows(quality_json)
    candidate = select_quality_row(rows, candidate_variant, arch=arch, scoring_mode=scoring_mode)
    baseline = (
        select_quality_row(rows, baseline_variant, arch=arch, scoring_mode=scoring_mode)
        if baseline_variant
        else None
    )
    floor = (
        select_quality_row(rows, floor_variant, arch=arch, scoring_mode=scoring_mode)
        if floor_variant
        else None
    )
    if floor:
        candidate = attach_above_floor(candidate, floor)
        if baseline:
            baseline = attach_above_floor(baseline, floor)

    quality = {
        "source": str(quality_json),
        "candidate": candidate,
        "baseline": baseline,
        "floor": floor,
    }
    if baseline:
        quality["kld_delta"] = candidate["mean_kld"] - baseline["mean_kld"]
        if candidate.get("ppl") is not None and baseline.get("ppl") is not None:
            quality["ppl_delta"] = candidate["ppl"] - baseline["ppl"]
        if floor:
            recovered = baseline["above_floor_kld"] - candidate["above_floor_kld"]
            quality["kld_recovered"] = recovered
            quality["kld_recovered_pct"] = (
                recovered / baseline["above_floor_kld"]
                if baseline["above_floor_kld"] > 0.0
                else None
            )

    verdict = "needs_baseline"
    if baseline:
        ppl_delta = quality.get("ppl_delta")
        kld_delta = quality["kld_delta"]
        if kld_delta < 0.0 and (ppl_delta is None or ppl_delta <= 0.0):
            verdict = "quality_improved"
        elif kld_delta < 0.0:
            verdict = "kld_improved_ppl_regressed"
        else:
            verdict = "no_quality_gain"

    return {
        "schema": METRICS_SCHEMA,
        "captured_at_utc": utc_now(),
        "host": socket.gethostname(),
        "git": git_sha(),
        "candidate_model": file_summary(candidate_model),
        "baseline_model": file_summary(baseline_model),
        "reference_model": reference_model,
        "dataset": dataset,
        "engine": engine_fingerprint(engine_root),
        "quality": quality,
        "weight": None,
        "verdict": verdict,
        "next_step": (
            "run Atlas AR/DFlash perf gates before promotion"
            if verdict == "quality_improved"
            else "iterate calibration recipe or collect a better baseline"
        ),
    }


def tensor_element_count(shape):
    count = 1
    for dim in shape or []:
        count *= int(dim)
    return count


def ceil_div(value, denom):
    return (int(value) + int(denom) - 1) // int(denom)


def estimate_format_data_size(shape, quant_format):
    fmt = quant_format.lower()
    elements = tensor_element_count(shape)
    if fmt == "f16":
        return elements * 2
    if fmt == "q8":
        return q8f16_data_size_for_shape(shape)
    if fmt in {"mq4", "hfq4"}:
        return ceil_div(elements, 256) * 136
    if fmt == "mq3":
        return ceil_div(elements, 256) * 104
    if fmt in {"mq6", "hfq6"}:
        return ceil_div(elements, 256) * 200
    if fmt in {"hfp4", "mfp4"}:
        if len(shape or []) != 2:
            return ceil_div(elements, 2)
        rows, k = int(shape[0]), int(shape[1])
        if k <= 0:
            return 0
        return rows * (16 + 17 * ceil_div(k, 32))
    raise ValueError(f"unsupported policy format size estimate: {quant_format}")


def load_json_sensitivity(path):
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tensors"), list):
        rows = payload["tensors"]
    elif isinstance(payload, dict) and isinstance(payload.get("scores"), dict):
        rows = [{"name": name, "score": score} for name, score in payload["scores"].items()]
    else:
        raise ValueError(f"unsupported sensitivity JSON shape: {path}")

    scores = {}
    aliases = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("hfq_name") or row.get("name") or row.get("tensor")
        if not name:
            continue
        score = (
            row.get("score")
            if row.get("score") is not None
            else row.get("sensitivity", row.get("importance"))
        )
        if score is None:
            continue
        score = max(float(score), 0.0)
        scores[name] = score
        aliases[name] = row.get("alias") or name
        for candidate in gguf_to_hfq_candidates(name):
            scores.setdefault(candidate, score)
            aliases.setdefault(candidate, name)
    return {
        "source": "json",
        "path": str(path),
        "score_count": len(scores),
        "scores": scores,
        "aliases": aliases,
    }


def load_imatrix_sensitivity(model, imatrix):
    join = match_imatrix_to_hfq(model, imatrix, max_tensors=0)
    scores = {}
    aliases = {}
    errors = []
    for item in join["matches"]:
        try:
            importance = read_gguf_f32_tensor(imatrix, f"{item['imatrix_name']}.in_sum2")
            counts = read_gguf_f32_tensor(imatrix, f"{item['imatrix_name']}.counts")
        except Exception as exc:
            errors.append({"imatrix_name": item["imatrix_name"], "error": str(exc)})
            continue
        denom = counts[0] if counts and counts[0] > 0.0 else 1.0
        normalized = [max(float(value) / denom, 0.0) for value in importance]
        score = sum(normalized) / len(normalized) if normalized else 0.0
        scores[item["hfq_name"]] = score
        aliases[item["hfq_name"]] = item["imatrix_name"]
    return {
        "source": "imatrix",
        "path": str(imatrix),
        "matched_count": join["matched_count"],
        "unmatched_count": join["unmatched_count"],
        "errors": errors[:16],
        "score_count": len(scores),
        "scores": scores,
        "aliases": aliases,
    }


def build_ingress_summary(hfq_tensors, model_family=None):
    family = (model_family or "").lower()
    expert_names = []
    router_names = []
    for name in sorted(hfq_tensors):
        lowered = name.lower()
        if ".experts." in lowered or ".expert." in lowered:
            expert_names.append(name)
        if (
            "router" in lowered
            or lowered.endswith(".mlp.gate.weight")
            or lowered.endswith(".block_sparse_moe.gate.weight")
        ):
            router_names.append(name)
    moe_detected = bool(
        expert_names
        or router_names
        or "moe" in family
        or "a3b" in family
        or "mixture" in family
    )
    return {
        "model_family": model_family,
        "moe_detected": moe_detected,
        "expert_tensor_count": len(expert_names),
        "router_tensor_count": len(router_names),
        "expert_tensors_preview": expert_names[:16],
        "router_tensors_preview": router_names[:16],
    }


def build_policy_probe_plan(objectives, ingress):
    plan = {}
    if "dynamic-tensor-policy" in objectives:
        plan["dynamic_tensor_policy"] = [
            "rank tensors by sensitivity per added byte",
            "emit mixed-format candidate recipe under the requested byte budget",
            "evaluate KLD/PPL against a fixed reference before promotion",
            "send quality-passing candidates through Atlas AR and DFlash perf gates",
        ]
    if "moe-probe" in objectives or ingress.get("moe_detected"):
        plan["moe"] = [
            "classify router tensors, expert tensors, and shared dense tensors separately",
            "collect expert-hit distribution on calibration and eval prompts",
            "score per-expert sensitivity so rarely-used experts are not over-promoted blindly",
            "compare AR and DFlash quality/perf by expert-hit bucket before shipping a MoE policy",
        ]
    if "model-ingress" in objectives:
        plan["model_ingress"] = [
            "fingerprint config, tensor names, RoPE convention, attention layout, and MoE/router structure",
            "build an alias map from source tensor names to HFQ runtime tensor names",
            "run tiny KLD/PPL smoke before any full calibration sweep",
            "record unsupported kernels or quant contracts as policy blockers, not silent fallbacks",
        ]
    if "kv-policy" in objectives:
        plan["kv_policy"] = [
            "treat asym3 as the current quality/perf baseline",
            "profile q8, asym2, asym3, asym4, TriAttention/CASK, and turbo/rotor research modes separately",
            "score AR and DFlash independently because eviction and speculative decode have different failure modes",
            "package persistent KV calibration data inside the HFQ model artifact before promotion",
        ]
    return plan


def build_weight_transform_plan(methods):
    plan = {}
    if "paroquant" in methods:
        plan["paroquant"] = {
            "status": "planned",
            "transform": "channel-wise scaling plus independent pairwise Givens rotations",
            "first_probe": "optimize a single layer or small model in PyTorch, then compare KLD/PPL before any runtime mutation",
            "runtime_requirement": "fused inverse transform plus quant matvec kernel must pass Atlas AR/DFlash perf gates",
            "package_section": "transform.paro",
        }
    if "quarot" in methods:
        plan["quarot"] = {
            "status": "candidate",
            "transform": "orthogonal rotation baseline for comparison against ParoQuant",
            "package_section": "transform.quarot",
        }
    return plan


def build_policy(
    *,
    model,
    base_format,
    promotion_format,
    sensitivity_json=None,
    imatrix=None,
    max_extra_bytes,
    methods=None,
    objectives=None,
    domains=None,
    model_family=None,
    policy_id=None,
):
    validate_values([base_format, promotion_format], SUPPORTED_FORMATS, "format")
    methods = methods or []
    validate_values(methods, SUPPORTED_METHODS, "method")
    objectives = objectives or ["dynamic-tensor-policy"]
    validate_values(objectives, SUPPORTED_POLICY_OBJECTIVES, "policy objective")
    domains = domains or ["weights"]
    validate_values(domains, SUPPORTED_POLICY_DOMAINS, "policy domain")
    if max_extra_bytes is None or max_extra_bytes < 0:
        raise ValueError("--max-extra-bytes must be non-negative")
    if not sensitivity_json and not imatrix:
        raise ValueError("policy requires --sensitivity-json or --imatrix")

    hfq_summary, hfq_tensors = read_hfq_index(model, max_tensors=32)
    if sensitivity_json:
        sensitivity = load_json_sensitivity(sensitivity_json)
    else:
        sensitivity = load_imatrix_sensitivity(model, imatrix)
    scores = sensitivity["scores"]
    aliases = sensitivity["aliases"]
    ingress = build_ingress_summary(hfq_tensors, model_family=model_family)

    base_data_bytes = sum(int(item["data_size"]) for item in hfq_tensors.values())
    candidates = []
    unscored_tensor_count = 0
    format_mismatch_count = 0
    for name, tensor in sorted(hfq_tensors.items()):
        tensor_format = HFQ_QUANT_TYPE_FORMATS.get(tensor["quant_type_name"], "unknown")
        if tensor_format != base_format:
            format_mismatch_count += 1
            continue
        score = scores.get(name)
        if score is None:
            unscored_tensor_count += 1
            continue
        promoted_size = estimate_format_data_size(tensor["shape"], promotion_format)
        base_size = int(tensor["data_size"])
        extra = promoted_size - base_size
        if extra <= 0:
            continue
        candidates.append(
            {
                "hfq_name": name,
                "sensitivity_alias": aliases.get(name),
                "quant_type_name": tensor["quant_type_name"],
                "shape": tensor["shape"],
                "base_data_size": base_size,
                "promoted_data_size": promoted_size,
                "extra_bytes": extra,
                "score": float(score),
                "score_per_extra_byte": float(score) / float(extra),
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score_per_extra_byte"],
            -item["score"],
            item["extra_bytes"],
            item["hfq_name"],
        )
    )

    candidate_by_name = {item["hfq_name"]: item for item in candidates}
    selected = []
    selected_names = set()
    skipped = []
    remaining = int(max_extra_bytes)
    added_runtime_anchors = []
    for item in candidates:
        name = item["hfq_name"]
        if name in selected_names:
            continue
        bundle = []
        anchor = runtime_promotion_anchor_for_name(
            name,
            base_format=base_format,
            promotion_format=promotion_format,
        )
        if anchor and anchor not in selected_names:
            anchor_item = candidate_by_name.get(anchor)
            if anchor_item is None:
                skipped_item = dict(item)
                skipped_item["reason"] = "runtime_anchor_missing"
                skipped_item["runtime_bundle_anchor"] = anchor
                skipped.append(skipped_item)
                continue
            bundle.append(("anchor", anchor_item, name))
        bundle.append(("selected", item, anchor))

        unique_bundle = []
        bundle_names = set()
        for role, bundle_item, trigger in bundle:
            bundle_name = bundle_item["hfq_name"]
            if bundle_name in selected_names or bundle_name in bundle_names:
                continue
            unique_bundle.append((role, bundle_item, trigger))
            bundle_names.add(bundle_name)

        bundle_extra = sum(int(bundle_item["extra_bytes"]) for _, bundle_item, _ in unique_bundle)
        if bundle_extra <= remaining:
            for role, bundle_item, trigger in unique_bundle:
                selected_item = dict(bundle_item)
                if role == "anchor":
                    selected_item["runtime_bundle_role"] = "anchor"
                    selected_item["runtime_bundle_trigger"] = trigger
                    added_runtime_anchors.append({"anchor": selected_item["hfq_name"], "trigger": trigger})
                elif trigger:
                    selected_item["runtime_bundle_anchor"] = trigger
                selected.append(selected_item)
                selected_names.add(selected_item["hfq_name"])
                remaining -= int(selected_item["extra_bytes"])
        else:
            skipped_item = dict(item)
            skipped_item["reason"] = "over_budget"
            if anchor:
                skipped_item["runtime_bundle_anchor"] = anchor
                skipped_item["required_runtime_bundle_extra_bytes"] = bundle_extra
            skipped.append(skipped_item)

    selected_extra = sum(item["extra_bytes"] for item in selected)
    model_name = Path(model).name.replace(".", "-").replace("/", "-")
    return {
        "schema": POLICY_SCHEMA,
        "policy_id": policy_id or f"astrea-policy-{model_name}-{int(time.time())}",
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "git": git_sha(),
        "model": model,
        "model_family": model_family,
        "hfq": {
            "tensor_count": hfq_summary["tensor_count"],
            "quant_type_counts": hfq_summary["quant_type_counts"],
            "tensor_names_md5": hfq_summary["tensor_names_md5"],
        },
        "base_format": base_format,
        "promotion_format": promotion_format,
        "methods": methods,
        "objectives": objectives,
        "domains": domains,
        "ingress": ingress,
        "probe_plan": build_policy_probe_plan(objectives, ingress),
        "weight_transform_plan": build_weight_transform_plan(methods),
        "runtime_mutation_status": "deferred_to_loader_and_kernel_work",
        "runtime_promotion_bundles": runtime_bundle_summary(
            base_format,
            promotion_format,
            added_runtime_anchors,
        ),
        "sensitivity": {
            key: value
            for key, value in sensitivity.items()
            if key not in {"scores", "aliases"}
        },
        "base_data_bytes": base_data_bytes,
        "max_extra_bytes": int(max_extra_bytes),
        "selected_extra_bytes": selected_extra,
        "candidate_data_bytes": base_data_bytes + selected_extra,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "unscored_tensor_count": unscored_tensor_count,
        "format_mismatch_count": format_mismatch_count,
        "selected": selected,
        "skipped": skipped,
        "format_cost_model": {
            "mode": "estimated_data_bytes",
            "notes": [
                "Base tensor bytes come from the HFQ index.",
                "Promotion bytes are estimated from the target format contract; verify final artifact size after writing weights.",
            ],
        },
        "next_step": "write candidate weights, then collect KLD/PPL and Atlas AR/DFlash rows",
    }


def triattn_summary(path):
    item = file_summary(path)
    if item and item["is_file"]:
        try:
            with Path(path).open("rb") as f:
                item["magic"] = f.read(4).decode("ascii", errors="replace")
        except Exception as exc:
            item["read_error"] = str(exc)
    return item


def describe_kv_mode(mode, *, triattn=None):
    mode = mode.lower()
    if mode not in SUPPORTED_KV_MODES:
        raise ValueError(f"unsupported KV mode: {mode}")
    if mode == "fp16":
        return {
            "family": "reference",
            "status": "implemented",
            "persistent_artifact": False,
            "notes": ["highest memory cost; useful as a correctness reference"],
        }
    if mode == "q8":
        return {
            "family": "uniform",
            "status": "implemented",
            "persistent_artifact": False,
            "notes": ["byte-stable higher-precision KV reference path"],
        }
    if mode in {"asym2", "asym3", "asym4"}:
        bits = mode[-1]
        return {
            "family": "asym_rotated",
            "status": "implemented",
            "persistent_artifact": False,
            "notes": [f"rotated K at {bits} bits with V stored as Q8_0"],
        }
    if mode == "triattn":
        exists = bool(triattn and triattn.get("exists"))
        return {
            "family": "triattention",
            "status": "implemented" if exists else "needs_calibration_artifact",
            "persistent_artifact": True,
            "package_section": "triattn.centers",
            "notes": ["drop-eviction policy; package centers inside the model artifact before promotion"],
        }
    if mode == "cask":
        return {
            "family": "triattention",
            "status": "implemented_guarded",
            "persistent_artifact": True,
            "package_section": "triattn.centers",
            "notes": ["m-folding policy has known DFlash and A3B hazards; require explicit gates"],
        }
    if mode.startswith("turbo"):
        return {
            "family": "turboquant",
            "status": "research_candidate",
            "persistent_artifact": False,
            "notes": ["online random/vector quantization candidate; needs HIP kernel and quality probes"],
        }
    if mode in {"rotor", "planar", "iso"}:
        return {
            "family": "rotor",
            "status": "research_candidate",
            "persistent_artifact": False,
            "notes": ["block-rotation KV candidate close to hipfire's existing Givens/asym machinery"],
        }
    raise ValueError(f"unsupported KV mode: {mode}")


def build_kv_profile(
    *,
    model,
    modes=None,
    triattn=None,
    model_family=None,
    profile_id=None,
    engine_root=None,
):
    modes = modes or ["q8", "asym3", "triattn", "turbo3", "rotor"]
    validate_values(modes, SUPPORTED_KV_MODES, "KV mode")
    triattn_info = triattn_summary(triattn)
    hfq = summarize_hfq(model) if model and Path(model).is_file() else None
    mode_map = {
        mode: describe_kv_mode(mode, triattn=triattn_info)
        for mode in modes
    }
    if "asym3" not in mode_map:
        mode_map["asym3"] = describe_kv_mode("asym3", triattn=triattn_info)
    model_name = Path(model).name.replace(".", "-").replace("/", "-")
    return {
        "schema": KV_PROFILE_SCHEMA,
        "profile_id": profile_id or f"astrea-kv-{model_name}-{int(time.time())}",
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "git": git_sha(),
        "model": file_summary(model),
        "model_family": model_family,
        "engine": engine_fingerprint(engine_root),
        "hfq": {
            "tensor_count": hfq["tensor_count"],
            "quant_type_counts": hfq["quant_type_counts"],
            "tensor_names_md5": hfq["tensor_names_md5"],
        } if hfq else None,
        "baseline_mode": "asym3",
        "modes": mode_map,
        "triattn": triattn_info,
        "quality_gates": [
            "Collect AR KLD/PPL and long-context recall for each KV mode.",
            "Collect DFlash KLD/PPL, tau, and decoded-output checks separately from AR.",
            "Reject CASK m-folding on DFlash or A3B unless the dedicated coherence and recall gates pass.",
        ],
        "atlas_bridge": {
            "emit_rows": True,
            "required_metrics": ["tok/s", "tau", "eviction_count", "decode_ms/token", "prefill_ms/token"],
        },
        "package_target": {
            "external_sidecars": False,
            "triattn_section": "triattn.centers",
            "kv_policy_section": "kv.policy",
        },
        "next_step": "run Atlas AR/DFlash rows for implemented modes; keep turbo/rotor as research candidates until kernels exist",
    }


def build_bundle_plan(
    *,
    model,
    output,
    include=None,
    triattn=None,
    policy_id=None,
    bundle_id=None,
):
    include = include or ["weights", "kv-policy", "evidence"]
    validate_values(include, SUPPORTED_BUNDLE_INCLUDES, "bundle include")
    model_name = Path(model).name.replace(".", "-").replace("/", "-")
    sections = {
        "manifest": {
            "required": True,
            "description": "section table, schema versions, checksums, runtime requirements",
        }
    }
    if "weights" in include:
        sections["weights"] = {
            "required": True,
            "source": file_summary(model),
            "description": "existing HFQ tensor index and contiguous tensor payloads",
        }
    if "paro" in include:
        sections["transform.paro"] = {
            "required": False,
            "policy_id": policy_id,
            "description": "ParoQuant channel scales and independent pairwise rotation parameters",
            "runtime_status": "deferred_until_loader_and_fused_kernel_exist",
        }
    if "kv-policy" in include:
        sections["kv.policy"] = {
            "required": False,
            "policy_id": policy_id,
            "description": "selected KV mode, safety gates, profile provenance, and runtime constraints",
            "runtime_status": "deferred_until_loader_policy_dispatch_exists",
        }
    if "triattn" in include:
        sections["triattn.centers"] = {
            "required": False,
            "source": triattn_summary(triattn),
            "description": "TriAttention band centers embedded in the model package instead of stored as a loose sidecar",
            "runtime_status": "deferred_until_loader_embedded_tria_read_exists",
        }
    if "evidence" in include:
        sections["evidence.summary"] = {
            "required": False,
            "description": "Astrea KLD/PPL evidence and Atlas AR/DFlash perf row references",
        }
    return {
        "schema": BUNDLE_PLAN_SCHEMA,
        "bundle_id": bundle_id or f"astrea-bundle-{model_name}-{int(time.time())}",
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "git": git_sha(),
        "model": str(model),
        "output": str(output),
        "policy_id": policy_id,
        "container": {
            "format": "hfq-package-v0",
            "is_zip": False,
            "external_sidecars": False,
            "layout": "HFQ-compatible typed section table with embedded policy/calibration sections",
        },
        "external_sidecars_target": False,
        "sections": sections,
        "deferred_runtime_work": [
            "loader must parse and validate the package section table",
            "runtime must reject transform or KV policy sections when required kernels are unavailable",
            "CLI pull/run should prefer embedded triattn.centers over loose .triattn.bin files",
            "Atlas must record AR and DFlash perf rows before package promotion",
        ],
        "next_step": "treat this as a package contract artifact; do not write a packaged model until loader support exists",
    }


def build_report(paths):
    items = [load_json(path) for path in paths]
    metric_items = [item for item in items if item.get("schema") == METRICS_SCHEMA]
    if any(item.get("verdict") == "quality_improved" for item in metric_items):
        recommendation = "quality improved; run Atlas AR/DFlash perf gates before promotion"
    elif metric_items:
        recommendation = "quality evidence does not justify promotion yet"
    else:
        recommendation = "collect KLD/PPL evidence and Atlas perf rows before promotion"
    return {
        "schema": REPORT_SCHEMA,
        "captured_at_utc": utc_now(),
        "artifact_count": len(items),
        "artifacts": [
            {
                "schema": item.get("schema"),
                "plan_id": item.get("plan_id"),
                "status": item.get("status"),
                "formats": item.get("formats"),
                "methods": item.get("methods"),
                "verdict": item.get("verdict"),
            }
            for item in items
        ],
        "metric_verdicts": [item.get("verdict") for item in metric_items],
        "recommendation": recommendation,
    }


def load_paroquant_import_module():
    import importlib.util

    script_path = Path(__file__).with_name("paroquant_import.py")
    spec = importlib.util.spec_from_file_location("hipfire_paroquant_import", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_paroquant_oracle_module():
    import importlib.util

    script_path = Path(__file__).with_name("paroquant_oracle.py")
    spec = importlib.util.spec_from_file_location("hipfire_paroquant_oracle", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paro_probe_model(model, *, local_only=False, max_modules=None):
    module = load_paroquant_import_module()
    return module.probe_model(model, local_only=local_only, max_modules=max_modules)


def paro_import_model(model, output, *, local_only=False, max_modules=None, layout="native", copy_floats="f16"):
    module = load_paroquant_import_module()
    return module.import_model(
        model,
        output,
        local_only=local_only,
        max_modules=max_modules,
        layout=layout,
        copy_floats=copy_floats,
    )


def paro_oracle_model(source, hfq, *, module_name=None, local_only=False, samples=2, seed=1234, input_scale=0.125, atol=0.0):
    module = load_paroquant_oracle_module()
    args = argparse.Namespace(
        source=source,
        hfq=hfq,
        module=module_name,
        local_only=local_only,
        samples=samples,
        seed=seed,
        input_scale=input_scale,
        atol=atol,
        pretty=False,
    )
    return module.run_oracle(args)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="astrea",
        description="Agent-native model calibration planning for hipfire.",
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Summarize model/imatrix inputs.")
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--imatrix")
    inspect.add_argument("--format", dest="quant_format")
    inspect.add_argument("--pretty", action="store_true")
    inspect.add_argument("--out", help="Write JSON to this path instead of stdout.")

    imatrix_join = sub.add_parser("imatrix-join", help="Report imatrix to HFQ tensor coverage.")
    imatrix_join.add_argument("--model", required=True)
    imatrix_join.add_argument("--imatrix", required=True)
    imatrix_join.add_argument("--max-tensors", type=int, default=32)
    imatrix_join.add_argument("--pretty", action="store_true")
    imatrix_join.add_argument("--out", help="Write JSON to this path instead of stdout.")

    fingerprint = sub.add_parser("fingerprint", help="Fingerprint the hipfire engine path.")
    fingerprint.add_argument("--engine-root")
    fingerprint.add_argument("--pretty", action="store_true")
    fingerprint.add_argument("--out", help="Write JSON to this path instead of stdout.")

    plan = sub.add_parser("plan", help="Emit an Astrea calibration plan.")
    plan.add_argument("--model", required=True)
    plan.add_argument("--format", dest="formats", action="append", required=True)
    plan.add_argument("--method", dest="methods", action="append", required=True)
    plan.add_argument("--imatrix")
    plan.add_argument("--calibration-corpus")
    plan.add_argument("--source-dir")
    plan.add_argument("--output")
    plan.add_argument("--eval-command", dest="eval_commands", action="append", default=[])
    plan.add_argument("--atlas-command", dest="atlas_commands", action="append", default=[])
    plan.add_argument("--recipe-stage", dest="recipe_stages", action="append", default=[])
    plan.add_argument("--plan-id")
    plan.add_argument("--pretty", action="store_true")
    plan.add_argument("--out", help="Write JSON to this path instead of stdout.")

    calibrate = sub.add_parser("calibrate", help="Create a calibration run artifact.")
    calibrate.add_argument("--plan", required=True)
    calibrate.add_argument("--source-dir")
    calibrate.add_argument("--write-candidate", action="store_true")
    calibrate.add_argument("--max-tensors", type=int)
    calibrate.add_argument("--tensor-filter")
    calibrate.add_argument("--clip-quantile", type=float, default=0.999)
    calibrate.add_argument("--workers", type=int, default=1)
    calibrate.add_argument("--dry-run", action="store_true", default=True)
    calibrate.add_argument("--pretty", action="store_true")
    calibrate.add_argument("--out", help="Write JSON to this path instead of stdout.")

    eval_cmd = sub.add_parser("eval", help="Render or run eval commands from a plan.")
    eval_cmd.add_argument("--plan", required=True)
    eval_cmd.add_argument("--run", action="store_true")
    eval_cmd.add_argument("--pretty", action="store_true")
    eval_cmd.add_argument("--out", help="Write JSON to this path instead of stdout.")

    metrics = sub.add_parser("metrics", help="Collect structured quality metrics.")
    metrics.add_argument("--quality-json", required=True)
    metrics.add_argument("--candidate-variant", required=True)
    metrics.add_argument("--baseline-variant")
    metrics.add_argument("--floor-variant")
    metrics.add_argument("--arch")
    metrics.add_argument("--scoring-mode")
    metrics.add_argument("--candidate-model")
    metrics.add_argument("--baseline-model")
    metrics.add_argument("--reference-model")
    metrics.add_argument("--dataset")
    metrics.add_argument("--engine-root")
    metrics.add_argument("--pretty", action="store_true")
    metrics.add_argument("--out", help="Write JSON to this path instead of stdout.")

    policy = sub.add_parser("policy", help="Emit a dynamic mixed-format quant policy.")
    policy.add_argument("--model", required=True)
    policy.add_argument("--base-format", required=True)
    policy.add_argument("--promotion-format", required=True)
    policy.add_argument("--sensitivity-json")
    policy.add_argument("--imatrix")
    policy.add_argument("--max-extra-bytes", type=int, required=True)
    policy.add_argument("--method", dest="methods", action="append", default=[])
    policy.add_argument("--objective", dest="objectives", action="append", default=[])
    policy.add_argument("--domain", dest="domains", action="append", default=[])
    policy.add_argument("--model-family")
    policy.add_argument("--policy-id")
    policy.add_argument("--pretty", action="store_true")
    policy.add_argument("--out", help="Write JSON to this path instead of stdout.")

    promote = sub.add_parser("promote", help="Write a mixed-format candidate from an Astrea policy artifact.")
    promote.add_argument("--policy", required=True)
    promote.add_argument("--source-dir", required=True)
    promote.add_argument("--output", required=True)
    promote.add_argument("--max-tensors", type=int)
    promote.add_argument("--tensor-filter")
    promote.add_argument("--pretty", action="store_true")
    promote.add_argument("--out", help="Write JSON to this path instead of stdout.")

    kv_profile = sub.add_parser("kv-profile", help="Emit a KV cache policy/profile artifact.")
    kv_profile.add_argument("--model", required=True)
    kv_profile.add_argument("--mode", dest="modes", action="append", default=[])
    kv_profile.add_argument("--triattn")
    kv_profile.add_argument("--model-family")
    kv_profile.add_argument("--engine-root")
    kv_profile.add_argument("--profile-id")
    kv_profile.add_argument("--pretty", action="store_true")
    kv_profile.add_argument("--out", help="Write JSON to this path instead of stdout.")

    bundle_plan = sub.add_parser("bundle-plan", help="Emit an HFQ package/bundle plan artifact.")
    bundle_plan.add_argument("--model", required=True)
    bundle_plan.add_argument("--output", required=True)
    bundle_plan.add_argument("--include", dest="include", action="append", default=[])
    bundle_plan.add_argument("--triattn")
    bundle_plan.add_argument("--policy-id")
    bundle_plan.add_argument("--bundle-id")
    bundle_plan.add_argument("--pretty", action="store_true")
    bundle_plan.add_argument("--out", help="Write JSON to this path instead of stdout.")

    paro_probe = sub.add_parser("paro-probe", help="Inspect a ParoQuant checkpoint and runtime contract.")
    paro_probe.add_argument("--model", required=True)
    paro_probe.add_argument("--local-only", action="store_true")
    paro_probe.add_argument("--max-modules", type=int)
    paro_probe.add_argument("--pretty", action="store_true")
    paro_probe.add_argument("--out", help="Write JSON to this path instead of stdout.")

    paro_import = sub.add_parser("paro-import", help="Write a runtime-loadable PARO4G128 HFQ model.")
    paro_import.add_argument("--model", required=True)
    paro_import.add_argument("--output", required=True)
    paro_import.add_argument("--local-only", action="store_true")
    paro_import.add_argument("--max-modules", type=int)
    paro_import.add_argument("--layout", choices=("native", "engine"), default="native")
    paro_import.add_argument("--copy-floats", choices=("f16", "q8"), default="f16")
    paro_import.add_argument("--pretty", action="store_true")
    paro_import.add_argument("--out", help="Write JSON to this path instead of stdout.")

    paro_oracle = sub.add_parser("paro-oracle", help="Compare PARO4G128 HFQ bytes against a PyTorch Paro decode oracle.")
    paro_oracle.add_argument("--source", required=True, help="ParoQuant safetensors directory or HuggingFace repo id.")
    paro_oracle.add_argument("--hfq", required=True, help="HFQ file produced by astrea paro-import.")
    paro_oracle.add_argument("--module", dest="module_name", help="Paro module base name; defaults to the first complete module.")
    paro_oracle.add_argument("--local-only", action="store_true")
    paro_oracle.add_argument("--samples", type=int, default=2)
    paro_oracle.add_argument("--seed", type=int, default=1234)
    paro_oracle.add_argument("--input-scale", type=float, default=0.125)
    paro_oracle.add_argument("--atol", type=float, default=0.0)
    paro_oracle.add_argument("--pretty", action="store_true")
    paro_oracle.add_argument("--out", help="Write JSON to this path instead of stdout.")

    report = sub.add_parser("report", help="Summarize Astrea JSON artifacts.")
    report.add_argument("artifacts", nargs="+")
    report.add_argument("--pretty", action="store_true")
    report.add_argument("--out", help="Write JSON to this path instead of stdout.")

    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        write_json(
            inspect_model(args.model, imatrix=args.imatrix, quant_format=args.quant_format),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "imatrix-join":
        write_json(
            match_imatrix_to_hfq(args.model, args.imatrix, max_tensors=args.max_tensors),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "fingerprint":
        write_json(
            engine_fingerprint(args.engine_root),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "plan":
        write_json(
            build_plan(
                model=args.model,
                formats=args.formats,
                methods=args.methods,
                imatrix=args.imatrix,
                calibration_corpus=args.calibration_corpus,
                source_dir=args.source_dir,
                output=args.output,
                eval_commands=args.eval_commands,
                atlas_commands=args.atlas_commands,
                recipe_stages=args.recipe_stages,
                plan_id=args.plan_id,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "calibrate":
        write_json(
            calibrate_plan(
                args.plan,
                dry_run=args.dry_run and not args.write_candidate,
                source_dir=args.source_dir,
                write_candidate=args.write_candidate,
                max_tensors=args.max_tensors,
                tensor_filter=args.tensor_filter,
                clip_quantile=args.clip_quantile,
                workers=args.workers,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "eval":
        write_json(eval_plan(args.plan, run=args.run), pretty=args.pretty, out=args.out)
    elif args.command == "metrics":
        write_json(
            collect_metrics(
                quality_json=args.quality_json,
                candidate_variant=args.candidate_variant,
                baseline_variant=args.baseline_variant,
                floor_variant=args.floor_variant,
                arch=args.arch,
                scoring_mode=args.scoring_mode,
                candidate_model=args.candidate_model,
                baseline_model=args.baseline_model,
                reference_model=args.reference_model,
                dataset=args.dataset,
                engine_root=args.engine_root,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "policy":
        write_json(
            build_policy(
                model=args.model,
                base_format=args.base_format,
                promotion_format=args.promotion_format,
                sensitivity_json=args.sensitivity_json,
                imatrix=args.imatrix,
                max_extra_bytes=args.max_extra_bytes,
                methods=args.methods,
                objectives=args.objectives or None,
                domains=args.domains or None,
                model_family=args.model_family,
                policy_id=args.policy_id,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "promote":
        write_json(
            write_policy_promotion_candidate(
                load_json(args.policy),
                source_dir=args.source_dir,
                output=args.output,
                max_tensors=args.max_tensors,
                tensor_filter=args.tensor_filter,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "kv-profile":
        write_json(
            build_kv_profile(
                model=args.model,
                modes=args.modes or None,
                triattn=args.triattn,
                model_family=args.model_family,
                profile_id=args.profile_id,
                engine_root=args.engine_root,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "bundle-plan":
        write_json(
            build_bundle_plan(
                model=args.model,
                output=args.output,
                include=args.include or None,
                triattn=args.triattn,
                policy_id=args.policy_id,
                bundle_id=args.bundle_id,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "paro-probe":
        write_json(
            paro_probe_model(
                args.model,
                local_only=args.local_only,
                max_modules=args.max_modules,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "paro-import":
        write_json(
            paro_import_model(
                args.model,
                args.output,
                local_only=args.local_only,
                max_modules=args.max_modules,
                layout=args.layout,
                copy_floats=args.copy_floats,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "paro-oracle":
        write_json(
            paro_oracle_model(
                args.source,
                args.hfq,
                module_name=args.module_name,
                local_only=args.local_only,
                samples=args.samples,
                seed=args.seed,
                input_scale=args.input_scale,
                atol=args.atol,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "report":
        write_json(build_report(args.artifacts), pretty=args.pretty, out=args.out)
    return 0


def main_for_test(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = run(argv)
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception as exc:
        code = 1
        print(str(exc), file=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def main():
    return run()


if __name__ == "__main__":
    sys.exit(main())
