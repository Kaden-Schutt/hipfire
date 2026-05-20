#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASTREA_PATH = ROOT / "scripts" / "astrea.py"


def load_astrea():
    spec = importlib.util.spec_from_file_location("astrea", ASTREA_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["astrea"] = module
    spec.loader.exec_module(module)
    return module


class AstreaTests(unittest.TestCase):
    def write_minimal_gguf(self, path, tensor_names=None):
        def gguf_string(text):
            raw = text.encode("utf-8")
            return struct.pack("<Q", len(raw)) + raw

        tensor_names = tensor_names or [
            "blk.0.attn_q.weight.in_sum2",
            "blk.0.attn_q.weight.counts",
        ]
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", len(tensor_names))  # tensor count
        buf += struct.pack("<Q", 2)  # metadata kv count
        buf += gguf_string("general.alignment")
        buf += struct.pack("<I", 4)  # u32
        buf += struct.pack("<I", 32)
        buf += gguf_string("general.name")
        buf += struct.pack("<I", 8)  # string
        buf += gguf_string("synthetic-imatrix")
        for i, name in enumerate(tensor_names):
            shape = [256] if name.endswith(".in_sum2") else [1]
            offset = 0 if i == 0 else sum(
                (256 if prior.endswith(".in_sum2") else 1) * 4
                for prior in tensor_names[:i]
            )
            buf += gguf_string(name)
            buf += struct.pack("<I", len(shape))
            for dim in shape:
                buf += struct.pack("<Q", dim)
            buf += struct.pack("<I", 0)  # F32
            buf += struct.pack("<Q", offset)
        pad = (-len(buf)) % 32
        buf += b"\0" * pad
        payload_bytes = sum((256 if name.endswith(".in_sum2") else 1) * 4 for name in tensor_names)
        buf += b"\0" * payload_bytes
        path.write_bytes(buf)

    def write_minimal_hfq(self, path, tensors=None):
        tensors = tensors or [
            ("model.language_model.layers.0.mlp.gate_proj.weight", 24, [128, 256], 32, 64),
            ("model.language_model.layers.0.self_attn.q_proj.weight", 24, [64, 256], 32, 32),
            ("model.language_model.layers.0.linear_attn.in_proj_z.weight", 24, [64, 256], 32, 32),
        ]
        metadata = json.dumps(
            {"architecture": "qwen3", "config": {"model_type": "qwen3", "num_hidden_layers": 1}},
            separators=(",", ":"),
        ).encode("utf-8")
        index = bytearray()
        index += struct.pack("<I", len(tensors))
        for name, quant_type, shape, group_size, data_size in tensors:
            raw_name = name.encode("utf-8")
            index += struct.pack("<H", len(raw_name))
            index += raw_name
            index += struct.pack("<B", quant_type)
            index += struct.pack("<B", len(shape))
            for dim in shape:
                index += struct.pack("<I", dim)
            index += struct.pack("<I", group_size)
            index += struct.pack("<Q", data_size)
        header_size = 32
        data_offset = header_size + len(metadata) + len(index)
        buf = bytearray()
        buf += b"HFQM"
        buf += struct.pack("<I", 1)  # version
        buf += struct.pack("<I", 5)  # arch id
        buf += struct.pack("<I", len(tensors))
        buf += struct.pack("<Q", header_size)
        buf += struct.pack("<Q", data_offset)
        buf += metadata
        buf += index
        buf += b"\0" * sum(item[4] for item in tensors)
        path.write_bytes(buf)

    def write_minimal_safetensors(self, path, tensors=None):
        tensors = tensors or {
            "model.language_model.layers.0.mlp.gate_proj.weight": {
                "dtype": "BF16",
                "shape": [128, 256],
                "data_offsets": [0, 65536],
            },
            "model.language_model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16",
                "shape": [64, 256],
                "data_offsets": [65536, 98304],
            },
        }
        header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
        data_size = max(meta["data_offsets"][1] for meta in tensors.values())
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * data_size)

    def write_f32_safetensors(self, path, tensor_values):
        header_items = {}
        payload = bytearray()
        for name, rows in tensor_values.items():
            flat = [float(value) for row in rows for value in row]
            start = len(payload)
            payload.extend(struct.pack(f"<{len(flat)}f", *flat))
            end = len(payload)
            header_items[name] = {
                "dtype": "F32",
                "shape": [len(rows), len(rows[0]) if rows else 0],
                "data_offsets": [start, end],
            }
        header = json.dumps(header_items, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(header)) + header + payload)

    def write_minimal_imatrix_gguf(self, path, logical_name, k):
        logical_names = logical_name if isinstance(logical_name, list) else [logical_name]

        def gguf_string(text):
            raw = text.encode("utf-8")
            return struct.pack("<Q", len(raw)) + raw

        names = []
        for item in logical_names:
            names.extend([f"{item}.in_sum2", f"{item}.counts"])
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", len(names))
        buf += struct.pack("<Q", 1)
        buf += gguf_string("general.alignment")
        buf += struct.pack("<I", 4)
        buf += struct.pack("<I", 32)
        offsets = []
        tensor_payloads = []
        cursor = 0
        for name in names:
            offsets.append(cursor)
            if name.endswith(".in_sum2"):
                payload = b"".join(struct.pack("<f", 1.0 + i / k) for i in range(k))
            else:
                payload = struct.pack("<f", 1.0)
            tensor_payloads.append(payload)
            cursor += len(payload)
        for name, offset in zip(names, offsets):
            shape = [k] if name.endswith(".in_sum2") else [1]
            buf += gguf_string(name)
            buf += struct.pack("<I", len(shape))
            for dim in shape:
                buf += struct.pack("<Q", dim)
            buf += struct.pack("<I", 0)  # F32
            buf += struct.pack("<Q", offset)
        pad = (-len(buf)) % 32
        buf += b"\0" * pad
        for payload in tensor_payloads:
            buf += payload
        path.write_bytes(buf)

    def write_expert_imatrix_gguf(self, path, logical_name, k, n_experts):
        def gguf_string(text):
            raw = text.encode("utf-8")
            return struct.pack("<Q", len(raw)) + raw

        names = [f"{logical_name}.in_sum2", f"{logical_name}.counts"]
        payloads = [
            b"".join(struct.pack("<f", 1.0 + i / max(k, 1)) for i in range(k * n_experts)),
            b"".join(struct.pack("<f", 8.0) for _ in range(n_experts)),
        ]
        shapes = [[k, n_experts], [1, n_experts]]
        offsets = []
        cursor = 0
        for payload in payloads:
            offsets.append(cursor)
            cursor += len(payload)

        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", len(names))
        buf += struct.pack("<Q", 1)
        buf += gguf_string("general.alignment")
        buf += struct.pack("<I", 4)
        buf += struct.pack("<I", 32)
        for name, shape, offset in zip(names, shapes, offsets):
            buf += gguf_string(name)
            buf += struct.pack("<I", len(shape))
            for dim in shape:
                buf += struct.pack("<Q", dim)
            buf += struct.pack("<I", 0)
            buf += struct.pack("<Q", offset)
        pad = (-len(buf)) % 32
        buf += b"\0" * pad
        for payload in payloads:
            buf += payload
        path.write_bytes(buf)

    def test_inspect_records_model_and_imatrix_fingerprints(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "qwen3.5-9b.mfp4"
            imatrix = root / "imatrix.gguf"
            model.write_bytes(b"model-bytes")
            imatrix.write_bytes(b"imatrix-bytes")

            result = astrea.inspect_model(str(model), imatrix=str(imatrix), quant_format="mfp4")

        self.assertEqual(result["schema"], "hipfire.astrea.inspect.v0")
        self.assertEqual(result["model"]["path"], str(model))
        self.assertEqual(result["model"]["bytes"], len(b"model-bytes"))
        self.assertEqual(result["model"]["md5"], "f98e2e1154e289fb796b948e8ccd5063")
        self.assertEqual(result["imatrix"]["bytes"], len(b"imatrix-bytes"))
        self.assertEqual(result["format"], "mfp4")

    def test_inspect_parses_gguf_imatrix_tensor_index(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "qwen3.5-9b.mfp4"
            imatrix = root / "imatrix.gguf"
            model.write_bytes(b"model-bytes")
            self.write_minimal_gguf(imatrix)

            result = astrea.inspect_model(str(model), imatrix=str(imatrix), quant_format="mfp4")

        gguf = result["imatrix_gguf"]
        self.assertEqual(gguf["schema"], "hipfire.astrea.gguf_summary.v0")
        self.assertEqual(gguf["version"], 3)
        self.assertEqual(gguf["tensor_count"], 2)
        self.assertEqual(gguf["metadata_kv_count"], 2)
        self.assertEqual(gguf["metadata"]["general.name"], "synthetic-imatrix")
        self.assertEqual(gguf["dtype_counts"], {"F32": 2})
        self.assertEqual(gguf["imatrix_logical_tensor_count"], 1)
        self.assertEqual(gguf["imatrix_suffix_counts"], {"counts": 1, "in_sum2": 1})
        self.assertEqual(gguf["tensors"][0]["name"], "blk.0.attn_q.weight.in_sum2")
        self.assertEqual(gguf["tensors"][1]["shape"], [1])
        self.assertEqual(gguf["imatrix_logical_names"], ["blk.0.attn_q.weight"])

    def test_summarize_hfq_reads_tensor_index_without_tensor_payloads(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "synthetic.mfp4"
            self.write_minimal_hfq(model)

            result = astrea.summarize_hfq(model)

        self.assertEqual(result["schema"], "hipfire.astrea.hfq_summary.v0")
        self.assertEqual(result["magic"], "HFQM")
        self.assertEqual(result["arch_id"], 5)
        self.assertEqual(result["tensor_count"], 3)
        self.assertEqual(result["quant_type_counts"], {"MFP4G32": 3})
        self.assertEqual(
            result["tensors"][0]["name"],
            "model.language_model.layers.0.mlp.gate_proj.weight",
        )
        self.assertEqual(result["tensors"][0]["data_offset"], result["data_offset"])

    def test_match_imatrix_to_hfq_tensors_uses_qwen35_name_aliases(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mfp4"
            imatrix = root / "imatrix.gguf"
            self.write_minimal_hfq(model)
            self.write_minimal_gguf(
                imatrix,
                [
                    "blk.0.ffn_gate.weight.in_sum2",
                    "blk.0.ffn_gate.weight.counts",
                    "blk.0.attn_q.weight.in_sum2",
                    "blk.0.attn_q.weight.counts",
                    "blk.0.attn_gate.weight.in_sum2",
                    "blk.0.attn_gate.weight.counts",
                ],
            )

            result = astrea.match_imatrix_to_hfq(str(model), str(imatrix))

        self.assertEqual(result["schema"], "hipfire.astrea.imatrix_hfq_join.v0")
        self.assertEqual(result["imatrix_logical_tensor_count"], 3)
        self.assertEqual(result["matched_count"], 3)
        self.assertEqual(result["unmatched_count"], 0)
        self.assertEqual(result["matched_quant_type_counts"], {"MFP4G32": 3})
        matched = {item["imatrix_name"]: item["hfq_name"] for item in result["matches"]}
        self.assertEqual(
            matched["blk.0.ffn_gate.weight"],
            "model.language_model.layers.0.mlp.gate_proj.weight",
        )
        self.assertEqual(
            matched["blk.0.attn_gate.weight"],
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        )
        self.assertEqual(result["k_dim_mismatch_count"], 0)

    def test_match_imatrix_expert_matrix_expands_per_expert(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4"
            imatrix = root / "imatrix.gguf"
            self.write_minimal_hfq(
                model,
                tensors=[
                    ("model.language_model.layers.0.mlp.experts.0.gate_up_proj.weight", 13, [512, 256], 256, 272),
                    ("model.language_model.layers.0.mlp.experts.1.gate_up_proj.weight", 13, [512, 256], 256, 272),
                ],
            )
            self.write_expert_imatrix_gguf(
                imatrix,
                "blk.0.ffn_gate_exps.weight",
                k=256,
                n_experts=2,
            )

            result = astrea.match_imatrix_to_hfq(str(model), str(imatrix))

        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["unmatched_count"], 0)
        self.assertEqual(result["k_dim_mismatch_count"], 0)
        self.assertEqual(result["awq_eligible_count"], 2)
        self.assertEqual(result["matched_by_slot"], {"ffn_gate_exps": 2})
        self.assertEqual(result["expert_coverage"][0]["matched_expert_count"], 2)
        self.assertFalse(result["matches"][0]["calibration_supported"])
        self.assertEqual(result["matches"][0]["imatrix_columns"], 2)

    def test_calibrate_dry_run_reports_matching_readiness(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mfp4"
            imatrix = root / "imatrix.gguf"
            plan = root / "plan.json"
            self.write_minimal_hfq(model)
            self.write_minimal_gguf(
                imatrix,
                [
                    "blk.0.ffn_gate.weight.in_sum2",
                    "blk.0.ffn_gate.weight.counts",
                    "blk.0.attn_q.weight.in_sum2",
                    "blk.0.attn_q.weight.counts",
                ],
            )
            plan.write_text(
                json.dumps(
                    {
                        "schema": "hipfire.astrea.plan.v0",
                        "plan_id": "calibration-ready-smoke",
                        "model": str(model),
                        "formats": ["mfp4"],
                        "methods": ["imatrix-scale"],
                        "imatrix": str(imatrix),
                        "output": str(root / "candidate.mfp4"),
                    }
                ),
                encoding="utf-8",
            )

            result = astrea.calibrate_plan(str(plan), dry_run=True)

        self.assertEqual(result["status"], "ready_for_calibration_dry_run")
        self.assertEqual(result["join"]["matched_count"], 2)
        self.assertEqual(result["join"]["unmatched_count"], 0)
        self.assertIn("weight mutation", result["next_step"])

    def test_calibrate_dry_run_checks_bf16_source_tensors(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mfp4"
            imatrix = root / "imatrix.gguf"
            plan = root / "plan.json"
            self.write_minimal_hfq(model)
            self.write_minimal_safetensors(source_dir / "model-00001-of-00001.safetensors")
            self.write_minimal_gguf(
                imatrix,
                [
                    "blk.0.ffn_gate.weight.in_sum2",
                    "blk.0.ffn_gate.weight.counts",
                    "blk.0.attn_q.weight.in_sum2",
                    "blk.0.attn_q.weight.counts",
                ],
            )
            plan.write_text(
                json.dumps(
                    {
                        "schema": "hipfire.astrea.plan.v0",
                        "plan_id": "source-ready-smoke",
                        "model": str(model),
                        "source_dir": str(source_dir),
                        "formats": ["mfp4"],
                        "methods": ["imatrix-scale"],
                        "imatrix": str(imatrix),
                    }
                ),
                encoding="utf-8",
            )

            result = astrea.calibrate_plan(str(plan), dry_run=True)

        self.assertEqual(result["status"], "ready_for_calibration_dry_run")
        self.assertEqual(result["source"]["tensor_count"], 2)
        self.assertEqual(result["source_match_count"], 2)
        self.assertEqual(result["missing_source_count"], 0)
        self.assertTrue(result["source_ready"])

    def test_quantize_mfp4g32_tensor_preserves_layout_and_rotation_flags(self):
        astrea = load_astrea()
        values = [[float(i - 128) / 64.0 for i in range(256)]]

        packed = astrea.quantize_mfp4g32_values(values)

        self.assertEqual(len(packed), 16 + 17 * 8)
        self.assertEqual(packed[4:6], struct.pack("<H", 8))
        self.assertEqual(packed[6], 0x05)
        self.assertNotEqual(packed[16:], b"\0" * (17 * 8))

    def test_numpy_mfp4_quantizer_matches_scalar_reference(self):
        astrea = load_astrea()
        if astrea.np is None:
            self.skipTest("numpy is not installed")
        values = [
            [math.sin(i * 0.173) + 0.25 * math.cos(i * 0.071) for i in range(256)],
            [math.cos(i * 0.119) - 0.30 * math.sin(i * 0.053) for i in range(256)],
        ]
        importance = [1.0 + (i % 13) / 13.0 for i in range(256)]

        scalar = astrea.quantize_mfp4g32_values(values, importance=importance, clip_quantile=1.0)
        fast = astrea.quantize_mfp4g32_values_numpy(values, importance=importance, clip_quantile=1.0)

        self.assertEqual(fast, scalar)

    def test_calibrate_write_candidate_patches_same_size_mfp4_tensor(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mfp4"
            imatrix = root / "imatrix.gguf"
            plan = root / "plan.json"
            candidate = root / "candidate.mfp4"
            tensor_name = "model.language_model.layers.0.mlp.gate_proj.weight"
            self.write_minimal_hfq(model, [(tensor_name, 24, [1, 256], 32, 16 + 17 * 8)])
            self.write_minimal_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {
                    tensor_name: {
                        "dtype": "BF16",
                        "shape": [1, 256],
                        "data_offsets": [0, 512],
                    }
                },
            )
            self.write_minimal_imatrix_gguf(imatrix, "blk.0.ffn_gate.weight", 256)
            plan.write_text(
                json.dumps(
                    {
                        "schema": "hipfire.astrea.plan.v0",
                        "plan_id": "write-candidate-smoke",
                        "model": str(model),
                        "source_dir": str(source_dir),
                        "formats": ["mfp4"],
                        "methods": ["imatrix-scale"],
                        "imatrix": str(imatrix),
                        "output": str(candidate),
                    }
                ),
                encoding="utf-8",
            )

            result = astrea.calibrate_plan(str(plan), dry_run=False, write_candidate=True, max_tensors=1)

            self.assertEqual(result["status"], "candidate_written")
            self.assertEqual(result["mutated_tensor_count"], 1)
            self.assertEqual(candidate.stat().st_size, model.stat().st_size)
            _, tensor_map = astrea.read_hfq_index(candidate)
            patched = candidate.read_bytes()[
                tensor_map[tensor_name]["data_offset"] : tensor_map[tensor_name]["data_offset"]
                + tensor_map[tensor_name]["data_size"]
            ]
            self.assertEqual(patched[6], 0x05)
            self.assertNotEqual(patched, b"\0" * len(patched))

    def test_calibrate_write_candidate_accepts_comma_tensor_filters(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mfp4"
            imatrix = root / "imatrix.gguf"
            plan = root / "plan.json"
            candidate = root / "candidate.mfp4"
            gate = "model.language_model.layers.0.mlp.gate_proj.weight"
            q_proj = "model.language_model.layers.0.self_attn.q_proj.weight"
            self.write_minimal_hfq(
                model,
                [
                    (gate, 24, [1, 256], 32, 16 + 17 * 8),
                    (q_proj, 24, [1, 256], 32, 16 + 17 * 8),
                ],
            )
            self.write_minimal_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {
                    gate: {"dtype": "BF16", "shape": [1, 256], "data_offsets": [0, 512]},
                    q_proj: {"dtype": "BF16", "shape": [1, 256], "data_offsets": [512, 1024]},
                },
            )
            self.write_minimal_imatrix_gguf(
                imatrix,
                ["blk.0.ffn_gate.weight", "blk.0.attn_q.weight"],
                256,
            )
            plan.write_text(
                json.dumps(
                    {
                        "schema": "hipfire.astrea.plan.v0",
                        "plan_id": "comma-filter-smoke",
                        "model": str(model),
                        "source_dir": str(source_dir),
                        "formats": ["mfp4"],
                        "methods": ["imatrix-scale"],
                        "imatrix": str(imatrix),
                        "output": str(candidate),
                    }
                ),
                encoding="utf-8",
            )

            result = astrea.calibrate_plan(
                str(plan),
                dry_run=False,
                write_candidate=True,
                max_tensors=2,
                tensor_filter="ffn_gate,attn_q",
            )

            self.assertEqual(result["status"], "candidate_written")
            self.assertEqual(result["mutated_tensor_count"], 2)

    def test_parallel_write_candidate_matches_serial_bytes(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mfp4"
            imatrix = root / "imatrix.gguf"
            serial_plan = root / "serial-plan.json"
            parallel_plan = root / "parallel-plan.json"
            serial_candidate = root / "serial.mfp4"
            parallel_candidate = root / "parallel.mfp4"
            gate = "model.language_model.layers.0.mlp.gate_proj.weight"
            q_proj = "model.language_model.layers.0.self_attn.q_proj.weight"
            self.write_minimal_hfq(
                model,
                [
                    (gate, 24, [2, 256], 32, 2 * (16 + 17 * 8)),
                    (q_proj, 24, [2, 256], 32, 2 * (16 + 17 * 8)),
                ],
            )
            self.write_minimal_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {
                    gate: {"dtype": "BF16", "shape": [2, 256], "data_offsets": [0, 1024]},
                    q_proj: {"dtype": "BF16", "shape": [2, 256], "data_offsets": [1024, 2048]},
                },
            )
            self.write_minimal_imatrix_gguf(
                imatrix,
                ["blk.0.ffn_gate.weight", "blk.0.attn_q.weight"],
                256,
            )
            base_plan = {
                "schema": "hipfire.astrea.plan.v0",
                "plan_id": "parallel-write-candidate-smoke",
                "model": str(model),
                "source_dir": str(source_dir),
                "formats": ["mfp4"],
                "methods": ["imatrix-scale"],
                "imatrix": str(imatrix),
            }
            serial_plan.write_text(json.dumps(dict(base_plan, output=str(serial_candidate))), encoding="utf-8")
            parallel_plan.write_text(json.dumps(dict(base_plan, output=str(parallel_candidate))), encoding="utf-8")

            serial = astrea.calibrate_plan(
                str(serial_plan),
                dry_run=False,
                write_candidate=True,
                max_tensors=2,
                tensor_filter="ffn_gate,attn_q",
                clip_quantile=1.0,
                workers=1,
            )
            parallel = astrea.calibrate_plan(
                str(parallel_plan),
                dry_run=False,
                write_candidate=True,
                max_tensors=2,
                tensor_filter="ffn_gate,attn_q",
                clip_quantile=1.0,
                workers=2,
            )

            self.assertEqual(serial_candidate.read_bytes(), parallel_candidate.read_bytes())
            self.assertEqual(serial["mutated_tensor_count"], 2)
            self.assertEqual(parallel["mutated_tensor_count"], 2)
            self.assertEqual(parallel["parallel_workers"], 2)

    def test_plan_keeps_agent_human_contract_and_guardrails(self):
        astrea = load_astrea()

        plan = astrea.build_plan(
            model="/mnt/nas/qwen3.5-9b",
            formats=["mfp4"],
            methods=["imatrix-scale"],
            imatrix="/mnt/nas/imatrix.gguf",
            eval_commands=["python3 crates/hipfire-runtime/examples/eval_hipfire.rs"],
            atlas_commands=["python3 scripts/kernel_atlas.py collect-ar"],
            output="/tmp/qwen3.5-9b.mfp4.astrea",
            plan_id="astrea-mfp4-smoke",
        )

        self.assertEqual(plan["schema"], "hipfire.astrea.plan.v0")
        self.assertEqual(plan["plan_id"], "astrea-mfp4-smoke")
        self.assertEqual(plan["formats"], ["mfp4"])
        self.assertIn("human", plan["intended_runners"])
        self.assertIn("agent", plan["intended_runners"])
        self.assertIn("imatrix-scale", plan["methods"])
        self.assertTrue(any("KLD" in gate for gate in plan["quality_gates"]))
        self.assertTrue(any("runtime" in item.lower() for item in plan["runtime_constraints"]))

    def test_plan_records_stackable_recipe_stages(self):
        astrea = load_astrea()

        plan = astrea.build_plan(
            model="/mnt/nas/qwen3.5-9b.mq4",
            formats=["mq4"],
            methods=["imatrix-scale", "awq", "gptq"],
            recipe_stages=[
                "scale_search:imatrix-scale",
                "activation_aware:awq",
                "rounding:gptq",
            ],
        )

        self.assertEqual(plan["methods"], ["imatrix-scale", "awq", "gptq"])
        self.assertEqual(
            [(stage["stage"], stage["method"]) for stage in plan["recipe"]],
            [
                ("scale_search", "imatrix-scale"),
                ("activation_aware", "awq"),
                ("rounding", "gptq"),
            ],
        )
        self.assertTrue(all("order" in stage for stage in plan["recipe"]))
        self.assertTrue(plan["recipe_stackable"])

    def test_mq4_ls_fit_reduces_reconstruction_mse(self):
        astrea = load_astrea()
        if astrea.np is None:
            self.skipTest("numpy unavailable")
        rows = astrea.np.asarray(
            [
                [math.sin(i * 0.13) * (1.0 + (i % 23) / 7.0) + (0.75 if i == 17 else 0.0) for i in range(256)],
                [math.cos(i * 0.19) * (1.0 + (i % 29) / 9.0) - (0.50 if i == 211 else 0.0) for i in range(256)],
            ],
            dtype=astrea.np.float32,
        )

        minmax = astrea.dequantize_mq4g256_from_values_numpy(rows, fit="minmax")
        ls = astrea.dequantize_mq4g256_from_values_numpy(rows, fit="ls", ls_iters=3)
        minmax_mse = float(astrea.np.mean((minmax - rows) ** 2))
        ls_mse = float(astrea.np.mean((ls - rows) ** 2))

        self.assertLess(ls_mse, minmax_mse)

    def test_calibrate_mq4_ls_writes_same_format_candidate(self):
        astrea = load_astrea()
        if astrea.np is None:
            self.skipTest("numpy unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mq4.hfq"
            candidate = root / "synthetic.ls.mq4.hfq"
            imatrix = root / "imatrix.gguf"
            plan_path = root / "plan.json"
            tensor_name = "model.language_model.layers.0.mlp.gate_proj.weight"
            self.write_minimal_hfq(model, tensors=[(tensor_name, 13, [2, 256], 256, 272)])
            self.write_minimal_imatrix_gguf(imatrix, "blk.0.ffn_gate.weight", 256)
            self.write_f32_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {
                    tensor_name: [
                        [math.sin(i * 0.17) * (1.0 + (i % 17) / 8.0) for i in range(256)],
                        [math.cos(i * 0.11) * (1.0 + (i % 13) / 7.0) for i in range(256)],
                    ],
                },
            )
            plan_path.write_text(
                json.dumps(
                    {
                        "schema": "hipfire.astrea.plan.v0",
                        "plan_id": "mq4-ls-smoke",
                        "model": str(model),
                        "output": str(candidate),
                        "formats": ["mq4"],
                        "methods": ["mq4-ls"],
                        "imatrix": str(imatrix),
                    }
                ),
                encoding="utf-8",
            )

            result = astrea.calibrate_plan(
                str(plan_path),
                dry_run=False,
                source_dir=str(source_dir),
                write_candidate=True,
                max_tensors=1,
                workers=1,
            )
            summary, tensor_map = astrea.read_hfq_index(candidate)
            candidate_size = candidate.stat().st_size
            model_size = model.stat().st_size

        self.assertEqual(result["status"], "candidate_written")
        self.assertEqual(result["message"], "wrote MQ4 LS candidate with same HFQ tensor byte ranges")
        self.assertEqual(result["mutated_tensor_count"], 1)
        self.assertEqual(result["mutations"][0]["method"], "mq4-ls")
        self.assertEqual(result["mutations"][0]["quant_type_name"], "MQ4G256")
        self.assertLessEqual(result["mutations"][0]["sample_ls_mse"], result["mutations"][0]["sample_minmax_mse"])
        self.assertEqual(candidate_size, model_size)
        self.assertEqual(summary["quant_type_counts"], {"MQ4G256": 1})
        self.assertEqual(tensor_map[tensor_name]["data_size"], 272)

    def test_calibrate_awq_writes_same_format_mq4_candidate(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mq4.hfq"
            candidate = root / "synthetic.awq.mq4.hfq"
            imatrix = root / "imatrix.gguf"
            plan_path = root / "plan.json"
            tensor_name = "model.language_model.layers.0.mlp.gate_proj.weight"
            self.write_minimal_hfq(model, tensors=[(tensor_name, 13, [2, 256], 256, 272)])
            self.write_minimal_imatrix_gguf(imatrix, "blk.0.ffn_gate.weight", 256)
            self.write_f32_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {
                    tensor_name: [
                        [math.sin(i * 0.17) * (1.0 + (i % 17) / 8.0) for i in range(256)],
                        [math.cos(i * 0.11) * (1.0 + (i % 13) / 7.0) for i in range(256)],
                    ],
                },
            )
            plan_path.write_text(
                json.dumps(
                    {
                        "schema": "hipfire.astrea.plan.v0",
                        "plan_id": "awq-mq4-smoke",
                        "model": str(model),
                        "output": str(candidate),
                        "formats": ["mq4"],
                        "methods": ["awq"],
                        "imatrix": str(imatrix),
                    }
                ),
                encoding="utf-8",
            )

            result = astrea.calibrate_plan(
                str(plan_path),
                dry_run=False,
                source_dir=str(source_dir),
                write_candidate=True,
                max_tensors=1,
                workers=1,
            )
            summary, tensor_map = astrea.read_hfq_index(candidate)
            candidate_size = candidate.stat().st_size
            model_size = model.stat().st_size
            patched_payload = candidate.read_bytes()[
                tensor_map[tensor_name]["data_offset"] : tensor_map[tensor_name]["data_offset"] + 272
            ]

        self.assertEqual(result["status"], "candidate_written")
        self.assertEqual(result["message"], "wrote AWQ MQ4 candidate with same HFQ tensor byte ranges")
        self.assertEqual(result["mutated_tensor_count"], 1)
        self.assertEqual(result["mutations"][0]["method"], "awq")
        self.assertEqual(result["mutations"][0]["quant_type_name"], "MQ4G256")
        self.assertIn(result["mutations"][0]["clip_ratio"], result["mutations"][0]["clip_ratio_grid"])
        self.assertEqual(candidate_size, model_size)
        self.assertEqual(summary["quant_type_counts"], {"MQ4G256": 1})
        self.assertEqual(tensor_map[tensor_name]["data_size"], 272)
        self.assertNotEqual(patched_payload, b"\0" * 272)

    def test_policy_selects_highest_sensitivity_promotions_under_extra_byte_budget(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            sensitivity = root / "sensitivity.json"
            self.write_minimal_hfq(
                model,
                tensors=[
                    ("model.language_model.layers.0.mlp.gate_proj.weight", 13, [1, 256], 256, 136),
                    ("model.language_model.layers.0.mlp.up_proj.weight", 13, [1, 256], 256, 136),
                    ("model.language_model.layers.0.mlp.down_proj.weight", 13, [1, 256], 256, 136),
                ],
            )
            sensitivity.write_text(
                json.dumps(
                    {
                        "tensors": [
                            {
                                "name": "model.language_model.layers.0.mlp.gate_proj.weight",
                                "score": 0.90,
                            },
                            {
                                "name": "model.language_model.layers.0.mlp.up_proj.weight",
                                "score": 0.40,
                            },
                            {
                                "name": "model.language_model.layers.0.mlp.down_proj.weight",
                                "score": 0.80,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            policy = astrea.build_policy(
                model=str(model),
                base_format="mq4",
                promotion_format="q8",
                sensitivity_json=str(sensitivity),
                max_extra_bytes=272,
                methods=["awq", "gptq"],
                policy_id="policy-smoke",
            )

        self.assertEqual(policy["schema"], "hipfire.astrea.policy.v0")
        self.assertEqual(policy["policy_id"], "policy-smoke")
        self.assertEqual(policy["base_format"], "mq4")
        self.assertEqual(policy["promotion_format"], "q8")
        self.assertEqual(policy["base_data_bytes"], 408)
        self.assertEqual(policy["selected_extra_bytes"], 272)
        self.assertEqual(policy["candidate_data_bytes"], 680)
        self.assertEqual(policy["methods"], ["awq", "gptq"])
        self.assertEqual(
            [item["hfq_name"] for item in policy["selected"]],
            [
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "model.language_model.layers.0.mlp.down_proj.weight",
            ],
        )
        self.assertEqual(policy["skipped"][0]["hfq_name"], "model.language_model.layers.0.mlp.up_proj.weight")
        self.assertEqual(policy["next_step"], "write candidate weights, then collect KLD/PPL and Atlas AR/DFlash rows")

    def test_policy_can_score_hfq_tensors_from_imatrix_aliases(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            imatrix = root / "imatrix.gguf"
            self.write_minimal_hfq(
                model,
                tensors=[
                    ("model.language_model.layers.0.mlp.gate_proj.weight", 13, [1, 256], 256, 136),
                    ("model.language_model.layers.0.self_attn.q_proj.weight", 13, [1, 256], 256, 136),
                ],
            )
            self.write_minimal_imatrix_gguf(
                imatrix,
                ["blk.0.ffn_gate.weight", "blk.0.attn_q.weight"],
                256,
            )

            policy = astrea.build_policy(
                model=str(model),
                base_format="mq4",
                promotion_format="q8",
                imatrix=str(imatrix),
                max_extra_bytes=136,
                methods=["imatrix-kmap"],
                policy_id="imatrix-policy-smoke",
            )

        self.assertEqual(policy["sensitivity"]["source"], "imatrix")
        self.assertEqual(policy["candidate_count"], 2)
        selected = policy["selected"][0]
        self.assertIn(selected["hfq_name"], {
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        })
        self.assertTrue(selected["sensitivity_alias"])
        self.assertEqual(selected["extra_bytes"], 136)

    def test_policy_q8_cost_model_matches_q8_0_storage(self):
        astrea = load_astrea()

        self.assertEqual(astrea.estimate_format_data_size([1, 256], "q8"), 272)
        self.assertEqual(astrea.estimate_format_data_size([2, 32], "q8"), 68)

    def test_policy_bundles_runtime_anchor_for_rotated_q8_projection(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            sensitivity = root / "sensitivity.json"
            anchor_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            dependent_name = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
            self.write_minimal_hfq(
                model,
                tensors=[
                    (anchor_name, 13, [1, 256], 256, 136),
                    (dependent_name, 13, [1, 256], 256, 136),
                ],
            )
            sensitivity.write_text(
                json.dumps(
                    {
                        "tensors": [
                            {"name": anchor_name, "score": 0.01},
                            {"name": dependent_name, "score": 1.00},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            policy = astrea.build_policy(
                model=str(model),
                base_format="mq4",
                promotion_format="q8",
                sensitivity_json=str(sensitivity),
                max_extra_bytes=272,
            )

        self.assertEqual(
            [item["hfq_name"] for item in policy["selected"]],
            [anchor_name, dependent_name],
        )
        self.assertEqual(policy["selected_extra_bytes"], 272)
        self.assertEqual(policy["runtime_promotion_bundles"]["added_anchor_count"], 1)
        self.assertEqual(policy["selected"][0]["runtime_bundle_role"], "anchor")
        self.assertEqual(policy["selected"][1]["runtime_bundle_anchor"], anchor_name)

    def test_promote_policy_candidate_rebuilds_hfq_with_q8_tensor(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mq4.hfq"
            candidate = root / "candidate.mixed.hfq"
            selected_name = "model.language_model.layers.0.mlp.gate_proj.weight"
            untouched_name = "model.language_model.layers.0.mlp.down_proj.weight"
            self.write_minimal_hfq(
                model,
                tensors=[
                    (selected_name, 13, [2, 32], 256, 136),
                    (untouched_name, 13, [2, 32], 256, 136),
                ],
            )
            self.write_f32_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {
                    selected_name: [
                        [math.sin(i * 0.11) for i in range(32)],
                        [math.cos(i * 0.07) for i in range(32)],
                    ],
                    untouched_name: [
                        [0.25 for _ in range(32)],
                        [-0.25 for _ in range(32)],
                    ],
                },
            )
            policy = {
                "schema": "hipfire.astrea.policy.v0",
                "policy_id": "promotion-writer-smoke",
                "model": str(model),
                "base_format": "mq4",
                "promotion_format": "q8",
                "selected": [{"hfq_name": selected_name}],
            }

            result = astrea.write_policy_promotion_candidate(
                policy,
                source_dir=str(source_dir),
                output=str(candidate),
            )

            self.assertEqual(result["schema"], "hipfire.astrea.promotion.v0")
            self.assertEqual(result["status"], "candidate_written")
            self.assertEqual(result["promoted_tensor_count"], 1)
            summary, tensor_map = astrea.read_hfq_index(candidate)
            self.assertTrue(summary["data_end_matches_file_size"])
            self.assertEqual(summary["quant_type_counts"], {"MQ4G256": 1, "Q8F16": 1})
            self.assertEqual(tensor_map[selected_name]["quant_type_name"], "Q8F16")
            self.assertEqual(tensor_map[selected_name]["group_size"], 32)
            self.assertEqual(tensor_map[selected_name]["data_size"], 68)
            self.assertEqual(tensor_map[untouched_name]["quant_type_name"], "MQ4G256")
            q8_payload = candidate.read_bytes()[
                tensor_map[selected_name]["data_offset"] : tensor_map[selected_name]["data_offset"]
                + tensor_map[selected_name]["data_size"]
            ]
            self.assertNotEqual(q8_payload[2:], b"\0" * (len(q8_payload) - 2))

    def test_promote_expands_runtime_anchor_for_legacy_policy(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mq4.hfq"
            candidate = root / "candidate.mixed.hfq"
            anchor_name = "model.language_model.layers.0.self_attn.q_proj.weight"
            dependent_name = "model.language_model.layers.0.self_attn.k_proj.weight"
            self.write_minimal_hfq(
                model,
                tensors=[
                    (anchor_name, 13, [1, 32], 256, 17),
                    (dependent_name, 13, [1, 32], 256, 17),
                ],
            )
            self.write_f32_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {
                    anchor_name: [[float(i) / 31.0 for i in range(32)]],
                    dependent_name: [[math.sin(i) for i in range(32)]],
                },
            )
            policy = {
                "schema": "hipfire.astrea.policy.v0",
                "policy_id": "legacy-dependent-only",
                "model": str(model),
                "base_format": "mq4",
                "promotion_format": "q8",
                "selected": [{"hfq_name": dependent_name}],
                "skipped": [{"hfq_name": anchor_name}],
            }

            result = astrea.write_policy_promotion_candidate(
                policy,
                source_dir=str(source_dir),
                output=str(candidate),
            )

            summary, tensor_map = astrea.read_hfq_index(candidate)

        self.assertEqual(result["promoted_tensor_count"], 2)
        self.assertEqual(result["runtime_promotion_bundles"]["added_anchor_count"], 1)
        self.assertEqual(summary["quant_type_counts"], {"Q8F16": 2})
        self.assertEqual(tensor_map[anchor_name]["quant_type_name"], "Q8F16")
        self.assertEqual(tensor_map[dependent_name]["quant_type_name"], "Q8F16")

    def test_cli_promote_policy_candidate_emits_json(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "source"
            source_dir.mkdir()
            model = root / "synthetic.mq4.hfq"
            candidate = root / "candidate.mixed.hfq"
            policy_path = root / "policy.json"
            tensor_name = "model.language_model.layers.0.mlp.gate_proj.weight"
            self.write_minimal_hfq(model, tensors=[(tensor_name, 13, [1, 32], 256, 136)])
            self.write_f32_safetensors(
                source_dir / "model-00001-of-00001.safetensors",
                {tensor_name: [[float(i) / 31.0 for i in range(32)]]},
            )
            policy_path.write_text(
                json.dumps(
                    {
                        "schema": "hipfire.astrea.policy.v0",
                        "policy_id": "promotion-cli-smoke",
                        "model": str(model),
                        "base_format": "mq4",
                        "promotion_format": "q8",
                        "selected": [{"hfq_name": tensor_name}],
                    }
                ),
                encoding="utf-8",
            )

            code, stdout, stderr = astrea.main_for_test(
                [
                    "promote",
                    "--policy",
                    str(policy_path),
                    "--source-dir",
                    str(source_dir),
                    "--output",
                    str(candidate),
                ]
            )

        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema"], "hipfire.astrea.promotion.v0")
        self.assertEqual(payload["status"], "candidate_written")
        self.assertEqual(payload["promoted_tensor_count"], 1)

    def test_policy_emits_moe_and_new_model_ingress_work_items(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "qwen3.6-a3b.mq4.hfq"
            sensitivity = root / "sensitivity.json"
            self.write_minimal_hfq(
                model,
                tensors=[
                    ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", 13, [1, 256], 256, 136),
                    ("model.language_model.layers.0.mlp.experts.1.gate_proj.weight", 13, [1, 256], 256, 136),
                    ("model.language_model.layers.0.mlp.gate.weight", 3, [2, 256], 1, 512),
                ],
            )
            sensitivity.write_text(
                json.dumps(
                    {
                        "tensors": [
                            {
                                "name": "model.language_model.layers.0.mlp.experts.1.gate_proj.weight",
                                "score": 0.75,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            policy = astrea.build_policy(
                model=str(model),
                base_format="mq4",
                promotion_format="q8",
                sensitivity_json=str(sensitivity),
                max_extra_bytes=120,
                methods=["awq"],
                objectives=["moe-probe", "model-ingress"],
                model_family="qwen3.6-a3b",
            )

        self.assertEqual(policy["objectives"], ["moe-probe", "model-ingress"])
        self.assertTrue(policy["ingress"]["moe_detected"])
        self.assertEqual(policy["ingress"]["expert_tensor_count"], 2)
        self.assertEqual(policy["ingress"]["router_tensor_count"], 1)
        self.assertIn("moe", policy["probe_plan"])
        self.assertIn("model_ingress", policy["probe_plan"])
        self.assertTrue(any("expert" in item for item in policy["probe_plan"]["moe"]))
        self.assertEqual(policy["model_family"], "qwen3.6-a3b")

    def test_policy_wires_paro_and_kv_domains_without_runtime_mutation(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            sensitivity = root / "sensitivity.json"
            self.write_minimal_hfq(
                model,
                tensors=[
                    ("model.language_model.layers.0.self_attn.q_proj.weight", 13, [1, 256], 256, 136),
                    ("model.language_model.layers.0.self_attn.k_proj.weight", 13, [1, 256], 256, 136),
                    ("model.language_model.layers.0.mlp.gate_proj.weight", 13, [1, 256], 256, 136),
                ],
            )
            sensitivity.write_text(
                json.dumps(
                    {
                        "tensors": [
                            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "score": 0.01},
                            {"name": "model.language_model.layers.0.self_attn.k_proj.weight", "score": 0.95},
                            {"name": "model.language_model.layers.0.mlp.gate_proj.weight", "score": 0.40},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            policy = astrea.build_policy(
                model=str(model),
                base_format="mq4",
                promotion_format="q8",
                sensitivity_json=str(sensitivity),
                max_extra_bytes=272,
                methods=["paroquant", "awq"],
                domains=["weights", "kv"],
                objectives=["dynamic-tensor-policy", "kv-policy"],
                policy_id="combined-policy-smoke",
            )

        self.assertEqual(policy["domains"], ["weights", "kv"])
        self.assertIn("paroquant", policy["methods"])
        self.assertIn("paroquant", policy["weight_transform_plan"])
        self.assertIn("kv_policy", policy["probe_plan"])
        self.assertEqual(policy["runtime_mutation_status"], "deferred_to_loader_and_kernel_work")
        self.assertEqual(
            [item["hfq_name"] for item in policy["selected"]],
            [
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.0.self_attn.k_proj.weight",
            ],
        )

    def test_kv_profile_records_asym_triattn_turbo_and_rotor_candidates(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            triattn = root / "synthetic.mq4.hfq.triattn.bin"
            self.write_minimal_hfq(model)
            triattn.write_bytes(b"TRIA" + b"\0" * 32)

            profile = astrea.build_kv_profile(
                model=str(model),
                modes=["q8", "asym3", "triattn", "turbo3", "rotor"],
                triattn=str(triattn),
                model_family="qwen3.5-9b",
                profile_id="kv-profile-smoke",
            )

        self.assertEqual(profile["schema"], "hipfire.astrea.kv_profile.v0")
        self.assertEqual(profile["profile_id"], "kv-profile-smoke")
        self.assertEqual(profile["baseline_mode"], "asym3")
        self.assertTrue(profile["triattn"]["exists"])
        self.assertEqual(profile["modes"]["asym3"]["status"], "implemented")
        self.assertEqual(profile["modes"]["turbo3"]["status"], "research_candidate")
        self.assertEqual(profile["modes"]["rotor"]["family"], "rotor")
        self.assertIn("DFlash", " ".join(profile["quality_gates"]))
        self.assertIn("atlas", profile["next_step"].lower())

    def test_bundle_plan_keeps_triattn_and_kv_policy_inside_model_package(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            triattn = root / "synthetic.mq4.hfq.triattn.bin"
            self.write_minimal_hfq(model)
            triattn.write_bytes(b"TRIA" + b"\0" * 32)

            plan = astrea.build_bundle_plan(
                model=str(model),
                output=str(root / "synthetic.mq4.astrea.hfq"),
                include=["weights", "paro", "kv-policy", "triattn"],
                triattn=str(triattn),
                policy_id="combined-policy-smoke",
                bundle_id="bundle-plan-smoke",
            )

        self.assertEqual(plan["schema"], "hipfire.astrea.bundle_plan.v0")
        self.assertEqual(plan["bundle_id"], "bundle-plan-smoke")
        self.assertFalse(plan["external_sidecars_target"])
        self.assertEqual(plan["container"]["format"], "hfq-package-v0")
        self.assertIn("transform.paro", plan["sections"])
        self.assertIn("kv.policy", plan["sections"])
        self.assertIn("triattn.centers", plan["sections"])
        self.assertEqual(plan["sections"]["triattn.centers"]["source"]["path"], str(triattn))
        self.assertIn("loader", " ".join(plan["deferred_runtime_work"]))

    def test_cli_policy_emits_json(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            sensitivity = root / "sensitivity.json"
            self.write_minimal_hfq(
                model,
                tensors=[
                    ("model.language_model.layers.0.mlp.gate_proj.weight", 13, [1, 256], 256, 136),
                ],
            )
            sensitivity.write_text(
                json.dumps(
                    {
                        "tensors": [
                            {
                                "name": "model.language_model.layers.0.mlp.gate_proj.weight",
                                "score": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr = astrea.main_for_test(
                [
                    "policy",
                    "--model",
                    str(model),
                    "--base-format",
                    "mq4",
                    "--promotion-format",
                    "q8",
                    "--sensitivity-json",
                    str(sensitivity),
                    "--max-extra-bytes",
                    "136",
                    "--method",
                    "awq",
                    "--policy-id",
                    "astrea-policy-cli-smoke",
                ]
            )

        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema"], "hipfire.astrea.policy.v0")
        self.assertEqual(payload["policy_id"], "astrea-policy-cli-smoke")
        self.assertEqual(payload["selected"][0]["hfq_name"], "model.language_model.layers.0.mlp.gate_proj.weight")

    def test_cli_kv_profile_and_bundle_plan_emit_json(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "synthetic.mq4.hfq"
            triattn = root / "synthetic.mq4.hfq.triattn.bin"
            self.write_minimal_hfq(model)
            triattn.write_bytes(b"TRIA" + b"\0" * 32)

            code, stdout, stderr = astrea.main_for_test(
                [
                    "kv-profile",
                    "--model",
                    str(model),
                    "--mode",
                    "asym3",
                    "--mode",
                    "turbo3",
                    "--triattn",
                    str(triattn),
                    "--profile-id",
                    "kv-cli-smoke",
                ]
            )
            self.assertEqual(code, 0, stderr)
            profile = json.loads(stdout)
            self.assertEqual(profile["schema"], "hipfire.astrea.kv_profile.v0")
            self.assertEqual(profile["modes"]["turbo3"]["status"], "research_candidate")

            code, stdout, stderr = astrea.main_for_test(
                [
                    "bundle-plan",
                    "--model",
                    str(model),
                    "--output",
                    str(root / "out.hfq"),
                    "--include",
                    "paro",
                    "--include",
                    "kv-policy",
                    "--include",
                    "triattn",
                    "--triattn",
                    str(triattn),
                    "--bundle-id",
                    "bundle-cli-smoke",
                ]
            )
            self.assertEqual(code, 0, stderr)
            bundle = json.loads(stdout)
            self.assertEqual(bundle["schema"], "hipfire.astrea.bundle_plan.v0")
            self.assertIn("triattn.centers", bundle["sections"])

    def test_metrics_ingests_kld_reduce_rows_and_computes_q8_floor_recovery(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            quality_json = Path(td) / "result-data.json"
            quality_json.write_text(
                json.dumps(
                    [
                        {
                            "variant": "Q8 floor (256ch)",
                            "arch": "gfx1151",
                            "scoring_mode": "prefill",
                            "n_chunks": 10,
                            "mean_kld": 0.5735,
                            "mean_kld_ci_lo": 0.55,
                            "mean_kld_ci_hi": 0.59,
                            "p99_kld": 1.0,
                            "ppl": 13.383,
                        },
                        {
                            "variant": "9B mq4-base",
                            "arch": "gfx1151",
                            "scoring_mode": "prefill",
                            "n_chunks": 10,
                            "mean_kld": 0.8165,
                            "mean_kld_ci_lo": 0.79,
                            "mean_kld_ci_hi": 0.84,
                            "p99_kld": 1.4,
                            "ppl": 15.063,
                        },
                        {
                            "variant": "9B mq4-awa",
                            "arch": "gfx1151",
                            "scoring_mode": "prefill",
                            "n_chunks": 10,
                            "mean_kld": 0.7373,
                            "mean_kld_ci_lo": 0.71,
                            "mean_kld_ci_hi": 0.76,
                            "p99_kld": 1.2,
                            "ppl": 14.303,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            metrics = astrea.collect_metrics(
                quality_json=str(quality_json),
                floor_variant="Q8 floor (256ch)",
                baseline_variant="9B mq4-base",
                candidate_variant="9B mq4-awa",
            )

        self.assertEqual(metrics["schema"], "hipfire.astrea.metrics.v0")
        self.assertEqual(metrics["quality"]["candidate"]["mean_kld"], 0.7373)
        self.assertAlmostEqual(metrics["quality"]["baseline"]["above_floor_kld"], 0.2430, places=4)
        self.assertAlmostEqual(metrics["quality"]["candidate"]["above_floor_kld"], 0.1638, places=4)
        self.assertAlmostEqual(metrics["quality"]["kld_recovered"], 0.0792, places=4)
        self.assertAlmostEqual(metrics["quality"]["kld_recovered_pct"], 0.3259, places=3)
        self.assertAlmostEqual(metrics["quality"]["ppl_delta"], -0.7600, places=4)
        self.assertEqual(metrics["verdict"], "quality_improved")

    def test_engine_fingerprint_detects_rope_convention_from_engine_root(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dispatch = root / "crates" / "rdna-compute" / "src" / "dispatch.rs"
            kernels = root / "kernels" / "src"
            dispatch.parent.mkdir(parents=True)
            kernels.mkdir(parents=True)
            dispatch.write_text(
                "pub fn rope_partial_interleaved_f32() { /* legacy */ }\n",
                encoding="utf-8",
            )
            (kernels / "rope_partial_interleaved.hip").write_text("legacy\n", encoding="utf-8")

            legacy = astrea.engine_fingerprint(root)
            self.assertEqual(legacy["rope_convention_default"], "interleaved_legacy")

            dispatch.write_text(
                "HIPFIRE_ROPE_INTERLEAVED_LEGACY rope_partial_halfsplit_f32\n",
                encoding="utf-8",
            )
            (kernels / "rope_partial_halfsplit.hip").write_text("halfsplit\n", encoding="utf-8")

            halfsplit = astrea.engine_fingerprint(root)
            self.assertEqual(halfsplit["rope_convention_default"], "halfsplit")
            self.assertIn("crates/rdna-compute/src/dispatch.rs", halfsplit["source_hashes"])

    def test_metrics_embeds_engine_fingerprint(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dispatch = root / "crates" / "rdna-compute" / "src" / "dispatch.rs"
            kernels = root / "kernels" / "src"
            dispatch.parent.mkdir(parents=True)
            kernels.mkdir(parents=True)
            dispatch.write_text("HIPFIRE_ROPE_INTERLEAVED_LEGACY rope_partial_halfsplit_f32\n", encoding="utf-8")
            (kernels / "rope_partial_halfsplit.hip").write_text("halfsplit\n", encoding="utf-8")
            quality_json = root / "result-data.json"
            quality_json.write_text(
                json.dumps(
                    [
                        {"variant": "floor", "mean_kld": 0.5, "p99_kld": 1.0, "ppl": 10.0},
                        {"variant": "base", "mean_kld": 0.8, "p99_kld": 1.0, "ppl": 12.0},
                        {"variant": "candidate", "mean_kld": 0.7, "p99_kld": 1.0, "ppl": 11.0},
                    ]
                ),
                encoding="utf-8",
            )

            metrics = astrea.collect_metrics(
                quality_json=str(quality_json),
                floor_variant="floor",
                baseline_variant="base",
                candidate_variant="candidate",
                engine_root=str(root),
            )

        self.assertEqual(metrics["engine"]["rope_convention_default"], "halfsplit")
        self.assertEqual(metrics["engine"]["root"], str(root))

    def test_cli_plan_emits_json(self):
        astrea = load_astrea()
        code, stdout, stderr = astrea.main_for_test(
            [
                "plan",
                "--model",
                "/mnt/nas/qwen3.5-9b",
                "--format",
                "mfp4",
                "--method",
                "imatrix-scale",
                "--plan-id",
                "astrea-cli-smoke",
            ]
        )

        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["plan_id"], "astrea-cli-smoke")
        self.assertEqual(payload["methods"], ["imatrix-scale"])

    def test_cli_out_writes_json_and_creates_parent_dirs(self):
        astrea = load_astrea()
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "runs" / "astrea" / "plan.json"
            code, stdout, stderr = astrea.main_for_test(
                [
                    "plan",
                    "--model",
                    "/mnt/nas/qwen3.5-9b",
                    "--format",
                    "mfp4",
                    "--method",
                    "imatrix-scale",
                    "--plan-id",
                    "astrea-out-smoke",
                    "--out",
                    str(out_path),
                ]
            )

            self.assertEqual(code, 0, stderr)
            self.assertEqual(stdout, "")
            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "hipfire.astrea.plan.v0")
            self.assertEqual(payload["plan_id"], "astrea-out-smoke")


if __name__ == "__main__":
    unittest.main()
