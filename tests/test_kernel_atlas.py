#!/usr/bin/env python3
import importlib.util
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS_PATH = REPO_ROOT / "scripts" / "kernel_atlas.py"

spec = importlib.util.spec_from_file_location("kernel_atlas", ATLAS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load kernel_atlas module from {ATLAS_PATH}")
kernel_atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_atlas)


class KernelAtlasTest(unittest.TestCase):
    def test_base_metadata_records_binary_and_diff_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td) / "bench_qwen35_mq4"
            binary.write_bytes(b"fake binary")
            diff = " M scripts/kernel_atlas.py\n--- diff body\n"
            diff_md5 = hashlib.md5(diff.encode("utf-8")).hexdigest()

            row = kernel_atlas.base_metadata(
                arch="gfx1201",
                hostname="hiptrx",
                git_sha="a4e42c5a",
                phase="ar",
                workload="qwen3.5-27b",
                model_path="/models/qwen3.5-27b.mq4",
                command=[str(binary)],
                env={},
                status="ok",
                binary_path=str(binary),
                git_dirty=True,
                diff_text=diff,
            )

        self.assertEqual(row["provenance"]["binary_path"], str(binary))
        self.assertEqual(row["provenance"]["binary_md5"], "2705a45681f2b74083dda1e3972714b1")
        self.assertTrue(row["provenance"]["git_dirty"])
        self.assertEqual(row["provenance"]["diff_md5"], diff_md5)

    def test_parse_bench_summary_extracts_ar_prefill_and_decode_metrics(self):
        text = """
noise
SUMMARY  gen_tok_s=101.5  bw_gib_s=1512.4  prefill_tok_s=1262.2  avg_ms=9.85  p50_ms=9.81
"""

        metrics = kernel_atlas.parse_bench_summary(text)

        self.assertEqual(metrics["gen_tok_s"], 101.5)
        self.assertEqual(metrics["bw_gib_s"], 1512.4)
        self.assertEqual(metrics["prefill_tok_s"], 1262.2)
        self.assertEqual(metrics["avg_ms"], 9.85)
        self.assertEqual(metrics["p50_ms"], 9.81)

    def test_parse_bench_profile_sections_extracts_prefill_and_decode_kernels(self):
        text = """
=== PROFILE (96 launches, 10.0ms wall) ===
  gemm_hfq4g256_residual                         24x  3.6ms  (150µs/call)  36.0%  900.0 GiB/s
  fused_qkvza_hfq4g256                            8x  1.8ms  (225µs/call)  18.0%  700.0 GiB/s
  TOTAL (serialized)                                10.0ms

=== DECODE PROFILE (64 launches, 5.0ms wall) ===
  gemv_hfq4g256_multirow_r4                      24x  2.5ms  (104µs/call)  50.0%  400.0 GiB/s
  fused_qk_l2_norm_scale_f32                      8x  0.5ms  (62µs/call)  10.0%  10.0 GiB/s
  TOTAL (serialized)                                 5.0ms
"""

        sections = kernel_atlas.parse_bench_profile_sections(text)

        self.assertEqual(
            [k["name"] for k in sections["prefill"]],
            ["gemm_hfq4g256_residual", "fused_qkvza_hfq4g256"],
        )
        self.assertEqual(sections["prefill"][0]["calls"], 24)
        self.assertEqual(sections["prefill"][0]["total_ms"], 3.6)
        self.assertEqual(sections["prefill"][0]["pct"], 36.0)
        self.assertEqual(
            [k["name"] for k in sections["decode_ar"]],
            ["gemv_hfq4g256_multirow_r4", "fused_qk_l2_norm_scale_f32"],
        )

    def test_ar_rows_are_phase_aware_and_keep_variant_env(self):
        metrics = {
            "prefill_tok_s": 574.3,
            "gen_tok_s": 35.8,
            "bw_gib_s": 536.4,
            "avg_ms": 27.91,
            "p50_ms": 27.88,
        }
        base = kernel_atlas.base_metadata(
            arch="gfx1201",
            hostname="hiptrx",
            git_sha="a4e42c5a",
            phase="unused",
            workload="qwen3.5-27b",
            model_path="/models/qwen3.5-27b.mq4",
            command=["bench_qwen35_mq4", "/models/qwen3.5-27b.mq4"],
            env={"HIPFIRE_KV_MODE": "asym3", "HIPFIRE_GEMV_ROWS": "4"},
            status="ok",
        )

        rows = kernel_atlas.ar_rows_from_metrics(
            base=base,
            metrics=metrics,
            model_size="27b",
            quant="mq4",
            prefill=128,
            gen=50,
            run_index=1,
            profile_sections={
                "prefill": [{"name": "gemm_hfq4g256_residual", "calls": 24}],
                "decode_ar": [{"name": "gemv_hfq4g256", "calls": 50}],
            },
        )

        self.assertEqual([row["phase"] for row in rows], ["prefill", "decode_ar"])
        self.assertEqual(rows[0]["shape_bucket"], "pp128")
        self.assertEqual(rows[0]["metrics"]["prefill_tok_s"], 574.3)
        self.assertEqual(rows[1]["shape_bucket"], "decode_ar_pp128_gen50")
        self.assertEqual(rows[1]["metrics"]["gen_tok_s"], 35.8)
        self.assertEqual(rows[1]["variant"]["env"]["HIPFIRE_GEMV_ROWS"], "4")
        self.assertEqual(rows[0]["artifacts"]["profile_kernels"][0]["name"], "gemm_hfq4g256_residual")
        self.assertEqual(rows[1]["artifacts"]["profile_kernels"][0]["name"], "gemv_hfq4g256")

    def test_profile_kernels_get_op_attribution(self):
        kernels = kernel_atlas.annotate_profile_kernels(
            [
                {"name": "gemv_hfq4g256_residual", "calls": 24},
                {"name": "fused_qkvza_hfq4g256", "calls": 8},
                {"name": "fused_qk_l2_norm_scale_f32", "calls": 2},
            ]
        )

        self.assertEqual(kernels[0]["op"]["family"], "linear")
        self.assertEqual(kernels[0]["op"]["role"], "residual_gemv")
        self.assertEqual(kernels[0]["op"]["phase_hint"], "decode")
        self.assertEqual(kernels[1]["op"]["family"], "attention")
        self.assertEqual(kernels[1]["op"]["role"], "qkvza_projection")
        self.assertEqual(kernels[2]["op"]["role"], "qk_norm_scale")

    def test_collect_dispatch_manifest_finds_source_and_dispatch_refs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kernel_src = root / "kernels" / "src" / "gemv_hfq4g256_multirow.hip"
            dispatch = root / "crates" / "rdna-compute" / "src" / "dispatch.rs"
            kernel_src.parent.mkdir(parents=True)
            dispatch.parent.mkdir(parents=True)
            kernel_src.write_text(
                'extern "C" __global__ void gemv_hfq4g256_multirow_r4() {}\n',
                encoding="utf-8",
            )
            dispatch.write_text(
                'if arch == "gfx1201" && rows == 4 { launch("gemv_hfq4g256_multirow_r4"); }\n',
                encoding="utf-8",
            )

            manifest = kernel_atlas.collect_dispatch_manifest(
                root=root,
                kernel_names=["gemv_hfq4g256_multirow_r4"],
            )

        entry = manifest["entries"][0]
        self.assertEqual(entry["name"], "gemv_hfq4g256_multirow_r4")
        self.assertEqual(entry["op"]["role"], "multirow_gemv")
        self.assertEqual(entry["source_files"][0]["path"], "kernels/src/gemv_hfq4g256_multirow.hip")
        self.assertEqual(entry["dispatch_refs"][0]["path"], "crates/rdna-compute/src/dispatch.rs")
        self.assertEqual(entry["dispatch_refs"][0]["line"], 1)

    def test_collect_dispatch_manifest_ranks_exact_kernel_sources_first(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "kernels" / "src"
            src.mkdir(parents=True)
            (src / "gemm_hfq4g256_residual.hip").write_text(
                "// mentions gemv_hfq4g256_residual in stale docs\n",
                encoding="utf-8",
            )
            (src / "fused_gate_up_hfq4g256.hip").write_text(
                "// mentions gemv_hfq4g256 as a related decode kernel\n",
                encoding="utf-8",
            )
            (src / "gemv_hfq4g256_residual.hip").write_text(
                'extern "C" __global__ void gemv_hfq4g256_residual() {}\n',
                encoding="utf-8",
            )
            (src / "gemv_hfq4g256.hip").write_text(
                'extern "C" __global__ void gemv_hfq4g256() {}\n',
                encoding="utf-8",
            )

            manifest = kernel_atlas.collect_dispatch_manifest(
                root=root,
                kernel_names=["gemv_hfq4g256_residual", "gemv_hfq4g256"],
            )

        by_name = {entry["name"]: entry for entry in manifest["entries"]}
        self.assertEqual(
            by_name["gemv_hfq4g256_residual"]["source_files"][0]["path"],
            "kernels/src/gemv_hfq4g256_residual.hip",
        )
        self.assertEqual(
            by_name["gemv_hfq4g256"]["source_files"][0]["path"],
            "kernels/src/gemv_hfq4g256.hip",
        )

    def test_collect_dispatch_manifest_prefers_target_arch_source_when_available(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "kernels" / "src"
            src.mkdir(parents=True)
            (src / "gemv_hfq4g256.hip").write_text(
                'extern "C" __global__ void gemv_hfq4g256() {}\n',
                encoding="utf-8",
            )
            (src / "gemv_hfq4g256.gfx1201.hip").write_text(
                'extern "C" __global__ void gemv_hfq4g256() {}\n',
                encoding="utf-8",
            )
            (src / "gemv_hfq4g256.gfx1030.v5.hip").write_text(
                'extern "C" __global__ void gemv_hfq4g256() {}\n',
                encoding="utf-8",
            )

            manifest = kernel_atlas.collect_dispatch_manifest(
                root=root,
                kernel_names=["gemv_hfq4g256"],
                arch="gfx1201",
            )

        entry = manifest["entries"][0]
        self.assertEqual(entry["arch"], "gfx1201")
        self.assertEqual(
            entry["source_files"][0]["path"],
            "kernels/src/gemv_hfq4g256.gfx1201.hip",
        )

    def test_collect_dispatch_manifest_ranks_active_dispatch_refs_before_comments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dispatch = root / "crates" / "hipfire-runtime" / "src" / "llama.rs"
            dispatch.parent.mkdir(parents=True)
            dispatch.write_text(
                "\n".join(
                    [
                        "/// gemv_hfq4g256_residual appears in old notes",
                        "// gemv_hfq4g256_residual comment-only mention",
                        "gpu.gemv_hfq4g256_residual(&w.buf, x, y, w.m, w.k);",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = kernel_atlas.collect_dispatch_manifest(
                root=root,
                kernel_names=["gemv_hfq4g256_residual"],
            )

        ref = manifest["entries"][0]["dispatch_refs"][0]
        self.assertEqual(ref["path"], "crates/hipfire-runtime/src/llama.rs")
        self.assertEqual(ref["line"], 3)
        self.assertEqual(ref["kind"], "code")
        self.assertTrue(ref["text"].startswith("gpu."))

    def test_collect_dispatch_manifest_ranks_runtime_src_refs_before_examples(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            example = root / "crates" / "hipfire-runtime" / "examples" / "bench.rs"
            runtime = root / "crates" / "hipfire-runtime" / "src" / "llama.rs"
            example.parent.mkdir(parents=True)
            runtime.parent.mkdir(parents=True)
            example.write_text(
                "gpu.gemv_hfq4g256(&d_a, &d_x, &d_y, m, k).unwrap();\n",
                encoding="utf-8",
            )
            runtime.write_text(
                "DType::HFQ4G256 => gpu.gemv_hfq4g256(&w.buf, x, y, w.m, w.k),\n",
                encoding="utf-8",
            )

            manifest = kernel_atlas.collect_dispatch_manifest(
                root=root,
                kernel_names=["gemv_hfq4g256"],
            )

        ref = manifest["entries"][0]["dispatch_refs"][0]
        self.assertEqual(ref["path"], "crates/hipfire-runtime/src/llama.rs")

    def test_parse_dflash_summary_prefers_authoritative_metric_lines(self):
        text = """
emitted: 256 tokens in 1.45s (176.55 tok/s)
decode_tok_s: 192.69
decode_tau: 13.2727
ttft_ms: 142.18
cycles: 18
accepted: 239
"""

        metrics = kernel_atlas.parse_dflash_summary(text)

        self.assertEqual(metrics["decode_tok_s"], 192.69)
        self.assertEqual(metrics["tau"], 13.2727)
        self.assertEqual(metrics["ttft_ms"], 142.18)
        self.assertEqual(metrics["cycles"], 18)
        self.assertEqual(metrics["accepted"], 239)
        self.assertEqual(metrics["emitted_tokens"], 256)

    def test_dflash_row_records_prompt_hash_and_phase(self):
        with tempfile.TemporaryDirectory() as td:
            prompt_path = Path(td) / "prompt.txt"
            target_path = Path(td) / "qwen3.5-27b.mq4"
            draft_path = Path(td) / "qwen35-27b-dflash-mq4.hfq"
            prompt_path.write_text("def merge_sort(xs):\n    pass\n", encoding="utf-8")
            target_path.write_text("small fake target", encoding="utf-8")
            draft_path.write_text("small fake draft", encoding="utf-8")
            metrics = {
                "decode_tok_s": 192.69,
                "tau": 13.2727,
                "ttft_ms": 142.18,
                "emitted_tokens": 256,
                "elapsed_s": 1.329,
            }
            base = kernel_atlas.base_metadata(
                arch="gfx1201",
                hostname="hiptrx",
                git_sha="a4e42c5a",
                phase="unused",
                workload="qwen3.5-27b-dflash",
                model_path="/models/qwen3.5-27b.mq4",
                command=["dflash_spec_demo"],
                env={"HIPFIRE_KV_MODE": "asym3"},
                status="ok",
            )

            row = kernel_atlas.dflash_row_from_metrics(
                base=base,
                metrics=metrics,
                target=str(target_path),
                draft=str(draft_path),
                prompt_file=str(prompt_path),
                max_tokens=256,
                ctx=2048,
                kv_mode="asym3",
                run_index=2,
            )

        self.assertEqual(row["phase"], "decode_dflash")
        self.assertEqual(row["shape_bucket"], "dflash_max256_ctx2048")
        self.assertEqual(row["run_index"], 2)
        self.assertEqual(row["metrics"]["decode_tok_s"], 192.69)
        self.assertEqual(row["artifacts"]["prompt_md5"], "412c4140301469b6745a0294a8352c15")
        self.assertIsNone(row["artifacts"]["target_md5"])
        self.assertIsNone(row["artifacts"]["draft_md5"])

    def test_parse_isa_metadata_extracts_real_amdhsa_kernel_fields(self):
        text = """
AMDGPU Metadata: ---
amdhsa.kernels:
  - .group_segment_fixed_size: 512
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 256
    .name:           gemm_hfq4g256_residual
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .sgpr_spill_count: 0
    .symbol:         gemm_hfq4g256_residual.kd
    .vgpr_count:     158
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
...
"""

        parsed = kernel_atlas.parse_isa_metadata(text)

        self.assertEqual(parsed["amdhsa_target"], "amdgcn-amd-amdhsa--gfx1100")
        self.assertEqual(len(parsed["kernels"]), 1)
        kernel = parsed["kernels"][0]
        self.assertEqual(kernel["name"], "gemm_hfq4g256_residual")
        self.assertEqual(kernel["vgpr_count"], 158)
        self.assertEqual(kernel["sgpr_count"], 52)
        self.assertEqual(kernel["group_segment_fixed_size"], 512)
        self.assertEqual(kernel["wavefront_size"], 32)

    def test_parse_disassembly_stats_counts_isa_categories(self):
        text = """
0000000000001700 <gemm_hfq4g256_residual>:
    s_load_b128 s[4:7], s[0:1], 0x18
    s_waitcnt lgkmcnt(0)
    global_load_b128 v[4:7], v[0:1], off
    v_mfma_f32_16x16x16f16 a[0:7], v[0:1], v[2:3], a[0:7]
    v_add_f32_e32 v0, v1, v2
    ds_write_b32 v0, v1
    s_cbranch_scc1 42
"""

        stats = kernel_atlas.parse_disassembly_stats(text)

        self.assertEqual(stats["instruction_count"], 7)
        self.assertEqual(stats["kernel_symbols"], ["gemm_hfq4g256_residual"])
        self.assertEqual(stats["opcode_counts"]["v_mfma_f32_16x16x16f16"], 1)
        self.assertEqual(stats["category_counts"]["matrix"], 1)
        self.assertEqual(stats["category_counts"]["vmem"], 1)
        self.assertEqual(stats["category_counts"]["lds"], 1)
        self.assertEqual(stats["category_counts"]["branch"], 1)

    def test_render_fit_view_shows_arch_quant_and_left_on_table(self):
        row = {
            "arch": "gfx1201",
            "workload": "qwen3.5-27b",
            "model_size": "27b",
            "quant": "mq3",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen50",
            "metrics": {"gen_tok_s": 35.8, "bw_gib_s": 536.4},
        }
        manifest = {
            "objects": [
                {
                    "kernels": [
                        {
                            "name": "gemv_hfq3g256_residual",
                            "vgpr_count": 132,
                            "sgpr_count": 48,
                            "vgpr_spill_count": 0,
                            "sgpr_spill_count": 0,
                            "wavefront_size": 32,
                            "group_segment_fixed_size": 0,
                        }
                    ],
                    "instruction_summary": {
                        "instruction_count": 1000,
                        "category_counts": {
                            "valu": 500,
                            "salu": 250,
                            "vmem": 200,
                            "lds": 25,
                            "branch": 25,
                        },
                    },
                }
            ]
        }

        view = kernel_atlas.render_fit_view(row, manifest)

        self.assertIn("ISA FIT VIEW", view)
        self.assertIn("gfx1201 | qwen3.5-27b | MQ3", view)
        self.assertIn("matrix/wmma       [################]  [................]", view)
        self.assertIn("bytes saved       high", view)
        self.assertIn("unpack tax        medium/high", view)
        self.assertIn("likely limit      memory/launch, not matrix throughput", view)
        self.assertIn("left on table     fusion / launch reduction / lower bytes moved", view)

    def test_render_fit_view_marks_matrix_unavailable_on_gfx1010(self):
        row = {
            "arch": "gfx1010",
            "workload": "qwen3.5-4b",
            "model_size": "4b",
            "quant": "mq4",
            "phase": "prefill",
            "shape_bucket": "pp128",
            "metrics": {"prefill_tok_s": 800.0},
        }
        manifest = {
            "objects": [
                {
                    "kernels": [{"name": "gemm_hfq4g256", "vgpr_count": 96, "sgpr_count": 40}],
                    "instruction_summary": {
                        "instruction_count": 100,
                        "category_counts": {"matrix": 0, "valu": 70, "vmem": 20, "salu": 10},
                    },
                }
            ]
        }

        view = kernel_atlas.render_fit_view(row, manifest)

        self.assertIn("gfx1010 | qwen3.5-4b | MQ4", view)
        self.assertIn("matrix/wmma       [................]  [................]   unavailable", view)
        self.assertIn("native matrix     no", view)

    def test_render_fit_view_joins_profiled_kernels_to_isa_objects(self):
        row = {
            "arch": "gfx1010",
            "workload": "qwen3.5-0.8b",
            "model_size": "0.8b",
            "quant": "mq4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen5",
            "metrics": {"gen_tok_s": 214.2},
            "artifacts": {
                "profile_kernels": [
                    {"name": "gemv_hfq4g256_multirow_r4", "calls": 24, "pct": 50.0},
                    {"name": "missing_hot_kernel", "calls": 8, "pct": 10.0},
                ]
            },
        }
        manifest = {
            "objects": [
                {
                    "path": ".hipfire_kernels/gfx1010/gemv_hfq4g256_multirow_default.hsaco",
                    "kernels": [
                        {
                            "name": "gemv_hfq4g256_multirow_r4",
                            "vgpr_count": 56,
                            "sgpr_count": 42,
                            "vgpr_spill_count": 10,
                            "sgpr_spill_count": 0,
                            "private_segment_fixed_size": 44,
                            "wavefront_size": 32,
                        }
                    ],
                    "instruction_summary": {
                        "instruction_count": 100,
                        "category_counts": {"valu": 70, "salu": 10, "vmem": 20},
                    },
                },
                {
                    "path": ".hipfire_kernels/gfx1010/unprofiled_candidate.hsaco",
                    "kernels": [
                        {
                            "name": "unprofiled_candidate",
                            "vgpr_count": 200,
                            "sgpr_count": 90,
                            "vgpr_spill_count": 0,
                            "sgpr_spill_count": 0,
                            "wavefront_size": 32,
                        }
                    ],
                    "instruction_summary": {
                        "instruction_count": 1000,
                        "category_counts": {"valu": 1000},
                    },
                },
            ]
        }

        joined = kernel_atlas.join_profile_to_isa(row, manifest)
        view = kernel_atlas.render_fit_view(row, manifest)

        self.assertEqual(joined["scope"], "profile-matched")
        self.assertEqual(joined["matched_profile_names"], ["gemv_hfq4g256_multirow_r4"])
        self.assertEqual(joined["unmatched_profile_names"], ["missing_hot_kernel"])
        self.assertEqual(joined["matched_object_count"], 1)
        self.assertIn("ISA SCOPE", view)
        self.assertIn("profile matched   1/2 names, 1 objects", view)
        self.assertIn("unmatched hot     missing_hot_kernel", view)
        self.assertIn("spills            [##########......]    10", view)
        self.assertNotIn("  200", view)

    def test_render_fit_view_shows_dispatch_and_op_attribution_for_hot_kernels(self):
        row = {
            "arch": "gfx1201",
            "workload": "qwen3.5-27b",
            "model_size": "27b",
            "quant": "mq4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen50",
            "metrics": {"gen_tok_s": 192.69},
            "artifacts": {
                "profile_kernels": kernel_atlas.annotate_profile_kernels(
                    [{"name": "gemv_hfq4g256_multirow_r4", "calls": 50, "pct": 36.3}]
                ),
                "dispatch": {
                    "schema": "hipfire.kernel_atlas.dispatch.v0",
                    "entries": [
                        {
                            "name": "gemv_hfq4g256_multirow_r4",
                            "op": {
                                "family": "linear",
                                "role": "multirow_gemv",
                                "phase_hint": "decode",
                            },
                            "source_files": [
                                {"path": "kernels/src/gemv_hfq4g256_multirow.hip"}
                            ],
                            "dispatch_refs": [
                                {
                                    "path": "crates/rdna-compute/src/dispatch.rs",
                                    "line": 42,
                                    "text": "HIPFIRE_GEMV_ROWS == 4",
                                }
                            ],
                            "env_controls": ["HIPFIRE_GEMV_ROWS"],
                        }
                    ],
                },
            },
        }
        manifest = {
            "objects": [
                {
                    "path": ".hipfire_kernels/gfx1201/gemv_hfq4g256_multirow_default.hsaco",
                    "kernels": [{"name": "gemv_hfq4g256_multirow_r4", "vgpr_count": 56}],
                    "instruction_summary": {
                        "instruction_count": 10,
                        "category_counts": {"valu": 6, "vmem": 4},
                    },
                }
            ]
        }

        view = kernel_atlas.render_fit_view(row, manifest)

        self.assertIn("HOT KERNELS", view)
        self.assertIn("gemv_hfq4g256_multirow_r4  linear.multirow_gemv", view)
        self.assertIn("src kernels/src/gemv_hfq4g256_multirow.hip", view)
        self.assertIn("dispatch crates/rdna-compute/src/dispatch.rs:42", view)
        self.assertIn("env HIPFIRE_GEMV_ROWS", view)

    def test_build_task_bundle_from_row_carries_hot_kernel_contract(self):
        row = {
            "arch": "gfx1201",
            "hostname": "hiptrx",
            "workload": "qwen3.5-27b",
            "model_size": "27b",
            "quant": "mq4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen50",
            "metrics": {"gen_tok_s": 192.69, "bw_gib_s": 536.4},
            "command": ["bench_qwen35_mq4", "/models/qwen3.5-27b.mq4"],
            "variant": {"env": {"HIPFIRE_GEMV_ROWS": "4"}},
            "provenance": {"binary_md5": "abc", "diff_md5": "def", "git_dirty": True},
            "artifacts": {
                "profile_kernels": kernel_atlas.annotate_profile_kernels(
                    [{"name": "gemv_hfq4g256_multirow_r4", "pct": 36.3, "calls": 50}]
                ),
                "dispatch": {
                    "entries": [
                        {
                            "name": "gemv_hfq4g256_multirow_r4",
                            "source_files": [{"path": "kernels/src/gemv_hfq4g256_multirow.hip"}],
                            "dispatch_refs": [{"path": "crates/rdna-compute/src/dispatch.rs", "line": 42}],
                            "env_controls": ["HIPFIRE_GEMV_ROWS"],
                        }
                    ]
                },
            },
        }
        manifest = {
            "objects": [
                {
                    "path": ".hipfire_kernels/gfx1201/gemv_hfq4g256_multirow_default.hsaco",
                    "kernels": [{"name": "gemv_hfq4g256_multirow_r4", "vgpr_count": 56}],
                    "instruction_summary": {"instruction_count": 10, "category_counts": {"valu": 6}},
                }
            ]
        }

        task = kernel_atlas.build_task_bundle_from_row(
            row,
            manifest,
            allowed_files=["kernels/src/gemv_hfq4g256_multirow.hip"],
            correctness_commands=[["./scripts/coherence-gate-dflash.sh"]],
            task_id="atlas-test-task",
        )

        self.assertEqual(task["schema"], "hipfire.kernel_atlas.task.v0")
        self.assertEqual(task["task_id"], "atlas-test-task")
        self.assertEqual(task["target"]["host"], "hiptrx")
        self.assertEqual(task["hot_kernel"]["name"], "gemv_hfq4g256_multirow_r4")
        self.assertEqual(task["hot_kernel"]["op"]["role"], "multirow_gemv")
        self.assertEqual(task["hot_kernel"]["source_files"][0]["path"], "kernels/src/gemv_hfq4g256_multirow.hip")
        self.assertEqual(task["hot_kernel"]["isa_objects"][0]["path"], ".hipfire_kernels/gfx1201/gemv_hfq4g256_multirow_default.hsaco")
        self.assertEqual(task["constraints"]["allowed_files"], ["kernels/src/gemv_hfq4g256_multirow.hip"])
        self.assertEqual(task["eval"]["metric"], "gen_tok_s")
        self.assertEqual(task["eval"]["goal"], "maximize")
        self.assertEqual(task["eval"]["benchmark_command"], ["bench_qwen35_mq4", "/models/qwen3.5-27b.mq4"])

    def test_build_task_bundle_strips_profile_env_and_requires_clean_baseline(self):
        row = {
            "arch": "gfx1201",
            "hostname": "hiptrx",
            "workload": "qwen3.5-27b",
            "quant": "mq4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen50",
            "metrics": {"gen_tok_s": 22.1},
            "command": ["bench"],
            "variant": {
                "env": {
                    "HIPFIRE_KV_MODE": "asym3",
                    "HIPFIRE_PROFILE": "1",
                    "HIPFIRE_PROFILE_DECODE": "1",
                    "HIPFIRE_HOST_TIMING": "1",
                }
            },
            "artifacts": {
                "profile_kernels": kernel_atlas.annotate_profile_kernels(
                    [{"name": "gemv_hfq4g256", "pct": 36.3}]
                ),
                "dispatch": {
                    "entries": [
                        {
                            "name": "gemv_hfq4g256",
                            "source_files": [{"path": "kernels/src/gemv_hfq4g256.gfx1201.hip"}],
                        }
                    ]
                },
            },
        }

        task = kernel_atlas.build_task_bundle_from_row(row, manifest={})

        self.assertEqual(task["baseline"]["env"], {"HIPFIRE_KV_MODE": "asym3"})
        self.assertEqual(task["baseline"]["row_env"]["HIPFIRE_PROFILE"], "1")
        self.assertEqual(
            task["baseline"]["stripped_env_keys"],
            ["HIPFIRE_HOST_TIMING", "HIPFIRE_PROFILE", "HIPFIRE_PROFILE_DECODE"],
        )
        self.assertTrue(task["eval"]["requires_fresh_baseline"])
        self.assertEqual(task["constraints"]["allowed_files"], ["kernels/src/gemv_hfq4g256.gfx1201.hip"])

    def test_build_task_bundle_defaults_allowed_files_to_top_ranked_source_only(self):
        row = {
            "arch": "gfx1201",
            "hostname": "hiptrx",
            "workload": "qwen3.5-27b",
            "quant": "mq4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen50",
            "metrics": {"gen_tok_s": 35.2},
            "command": ["bench"],
            "variant": {"env": {}},
            "artifacts": {
                "profile_kernels": kernel_atlas.annotate_profile_kernels(
                    [{"name": "gemv_hfq4g256", "pct": 36.3}]
                ),
                "dispatch": {
                    "entries": [
                        {
                            "name": "gemv_hfq4g256",
                            "source_files": [
                                {"path": "kernels/src/gemv_hfq4g256.gfx1201.hip"},
                                {"path": "kernels/src/gemv_hfq4g256.hip"},
                                {"path": "kernels/src/gemv_hfq4g256_wide.hip"},
                            ],
                        }
                    ]
                },
            },
        }

        task = kernel_atlas.build_task_bundle_from_row(row, manifest={})

        self.assertEqual(task["constraints"]["allowed_files"], ["kernels/src/gemv_hfq4g256.gfx1201.hip"])

    def test_build_task_bundle_from_suggestion_records_experiment_contract(self):
        row = {
            "arch": "gfx1100",
            "hostname": "k9lin",
            "workload": "qwen3.5-0.8b-paro4-native",
            "quant": "paro4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen16",
            "metrics": {"gen_tok_s": 72.5},
            "command": ["bench_qwen35_mq4", "/models/paro4.hfq"],
            "variant": {"env": {"HIPFIRE_PROFILE_DECODE": "1", "HIPFIRE_GRAPH": "0"}},
            "artifacts": {
                "profile_kernels": kernel_atlas.annotate_profile_kernels(
                    [
                        {"name": "rmsnorm_f32", "pct": 24.3},
                        {"name": "fused_qk_l2_norm_scale_f32", "pct": 8.8},
                    ]
                ),
                "dispatch": {
                    "entries": [
                        {
                            "name": "rmsnorm_f32",
                            "source_files": [{"path": "kernels/src/fused_rmsnorm_mq_rotate.hip"}],
                            "dispatch_refs": [{"path": "crates/hipfire-runtime/src/llama.rs", "line": 679}],
                        }
                    ]
                },
            },
        }
        suggestion = {
            "id": "sug-fusion",
            "rank": 1,
            "title": "Prioritize decode launch/fusion experiments over matrix-path changes",
            "lever_type": "fusion",
            "risk": "medium",
            "expected_impact": "high",
            "score": 92.0,
            "history_key": "hist-fusion",
            "hot_kernel": None,
            "allowed_files": [
                "crates/hipfire-runtime/src/llama.rs",
                "kernels/src/fused_rmsnorm_mq_rotate.hip",
                "crates/hipfire-runtime/src/llama.rs",
            ],
            "rationale": ["launch dominated"],
            "candidate_steps": ["fuse one adjacent transform"],
            "correctness_commands": [["./scripts/coherence-gate-dflash.sh"]],
        }

        task = kernel_atlas.build_task_bundle_from_row(row, manifest={}, suggestion=suggestion)
        text = kernel_atlas.render_task_markdown(task)

        self.assertEqual(task["producer"]["kind"], "hipfire-atlas-suggestion")
        self.assertEqual(task["producer"]["suggestion"]["id"], "sug-fusion")
        self.assertEqual(task["experiment"]["lever_type"], "fusion")
        self.assertEqual(
            task["constraints"]["allowed_files"],
            ["crates/hipfire-runtime/src/llama.rs", "kernels/src/fused_rmsnorm_mq_rotate.hip"],
        )
        self.assertEqual(task["eval"]["correctness_commands"], [["./scripts/coherence-gate-dflash.sh"]])
        self.assertIn("## Suggested Experiment", text)
        self.assertIn("Prioritize decode launch/fusion experiments", text)

    def test_build_suggestion_queue_emits_multiple_ranked_experiments(self):
        row = {
            "arch": "gfx1201",
            "hostname": "hiptrx",
            "workload": "qwen3.5-27b",
            "quant": "mq4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen50",
            "metrics": {"gen_tok_s": 35.2, "bw_gib_s": 491.0},
            "command": ["bench"],
            "variant": {
                "env": {
                    "HIPFIRE_KV_MODE": "asym3",
                    "HIPFIRE_PROFILE": "1",
                    "HIPFIRE_PROFILE_DECODE": "1",
                }
            },
            "artifacts": {
                "profile_kernels": kernel_atlas.annotate_profile_kernels(
                    [
                        {"name": "gemv_hfq4g256_residual", "pct": 36.3, "calls": 3200},
                        {"name": "fused_qkvza_hfq4g256", "pct": 18.3, "calls": 1200},
                        {"name": "fused_rmsnorm_mq_rotate", "pct": 10.2, "calls": 3200},
                        {"name": "mq_rotate_x", "pct": 2.0, "calls": 3200},
                    ]
                )
            },
        }
        dispatch = {
            "entries": [
                {
                    "name": "gemv_hfq4g256_residual",
                    "source_files": [{"path": "kernels/src/gemv_hfq4g256_residual.hip"}],
                    "dispatch_refs": [{"path": "crates/hipfire-runtime/src/llama.rs", "line": 790}],
                    "env_controls": ["HIPFIRE_GEMV_ROWS"],
                },
                {
                    "name": "fused_qkvza_hfq4g256",
                    "source_files": [{"path": "kernels/src/fused_qkvza_hfq4g256.hip"}],
                    "dispatch_refs": [{"path": "crates/hipfire-arch-qwen35/src/qwen35.rs", "line": 1944}],
                },
                {
                    "name": "fused_rmsnorm_mq_rotate",
                    "source_files": [{"path": "kernels/src/fused_rmsnorm_mq_rotate.hip"}],
                    "dispatch_refs": [{"path": "crates/rdna-compute/src/kernels.rs", "line": 269}],
                },
            ]
        }
        manifest = {
            "objects": [
                {
                    "kernels": [{"name": "gemv_hfq4g256_residual", "vgpr_count": 84}],
                    "instruction_summary": {
                        "instruction_count": 100,
                        "category_counts": {"valu": 60, "vmem": 20, "salu": 20},
                    },
                }
            ]
        }

        queue = kernel_atlas.build_suggestion_queue(
            row,
            manifest,
            dispatch,
            max_suggestions=8,
        )

        self.assertEqual(queue["schema"], "hipfire.kernel_atlas.suggestions.v0")
        self.assertEqual(queue["target"]["arch"], "gfx1201")
        self.assertGreaterEqual(queue["suggestion_count"], 4)
        ranks = [item["rank"] for item in queue["suggestions"]]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))
        self.assertTrue(any(item["lever_type"] == "dispatch" for item in queue["suggestions"]))
        self.assertTrue(any(item["lever_type"] == "source_sweep" for item in queue["suggestions"]))
        self.assertTrue(any(item["lever_type"] == "fusion" for item in queue["suggestions"]))
        self.assertTrue(any(item["lever_type"] == "measurement" for item in queue["suggestions"]))
        first = queue["suggestions"][0]
        self.assertIn(first["risk"], {"low", "medium", "high"})
        self.assertIn("eval", first)
        self.assertEqual(first["eval"]["metric"], "gen_tok_s")
        self.assertEqual(first["eval"]["env"], {"HIPFIRE_KV_MODE": "asym3"})
        self.assertTrue(first["eval"]["requires_fresh_baseline"])

    def test_build_suggestion_queue_demotes_rejected_history(self):
        row = {
            "arch": "gfx1201",
            "hostname": "hiptrx",
            "workload": "qwen3.5-27b",
            "quant": "mq4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen50",
            "metrics": {"gen_tok_s": 35.3},
            "command": ["bench"],
            "variant": {"env": {"HIPFIRE_KV_MODE": "asym3"}},
            "artifacts": {
                "profile_kernels": kernel_atlas.annotate_profile_kernels(
                    [
                        {"name": "gemv_hfq4g256_residual", "pct": 36.3, "calls": 3200},
                        {"name": "fused_qkvza_hfq4g256", "pct": 18.3, "calls": 1200},
                        {"name": "fused_rmsnorm_mq_rotate", "pct": 10.2, "calls": 3200},
                    ]
                )
            },
        }
        dispatch = {
            "entries": [
                {
                    "name": "gemv_hfq4g256_residual",
                    "source_files": [{"path": "kernels/src/gemv_hfq4g256_residual.hip"}],
                    "dispatch_refs": [{"path": "crates/rdna-compute/src/dispatch.rs", "line": 6231}],
                    "env_controls": ["HIPFIRE_GEMV_ROWS"],
                },
                {
                    "name": "fused_rmsnorm_mq_rotate",
                    "source_files": [{"path": "kernels/src/fused_rmsnorm_mq_rotate.hip"}],
                    "dispatch_refs": [{"path": "crates/rdna-compute/src/kernels.rs", "line": 269}],
                },
            ]
        }
        history = [
            {
                "task_id": "hiptrx-gfx1201-27b-ar-gemv-residual-atlas-ready-suggest-r2",
                "status": "pass",
                "metric": "gen_tok_s",
                "speedup": 0.997,
                "target": {
                    "arch": "gfx1201",
                    "workload": "qwen3.5-27b",
                    "phase": "decode_ar",
                    "shape_bucket": "decode_ar_pp32_gen50",
                    "quant": "mq4",
                },
                "hot_kernel": "gemv_hfq4g256_residual",
                "env": {"HIPFIRE_GEMV_ROWS": "2", "HIPFIRE_KV_MODE": "asym3"},
            }
        ]

        queue = kernel_atlas.build_suggestion_queue(row, {}, dispatch, max_suggestions=8, history=history)
        rejected = next(item for item in queue["suggestions"] if item["title"].startswith("Port/test gfx12 residual"))

        self.assertGreaterEqual(queue["history"]["demoted_suggestions"], 1)
        self.assertEqual(rejected["history"]["status"], "rejected")
        self.assertEqual(rejected["history"]["best_speedup"], 0.997)
        self.assertGreater(rejected["rank"], 1)
        self.assertFalse(queue["suggestions"][0]["title"].startswith("Port/test gfx12 residual"))

    def test_render_suggestion_markdown_lists_titles_and_files(self):
        queue = {
            "schema": "hipfire.kernel_atlas.suggestions.v0",
            "target": {"arch": "gfx1201", "workload": "qwen3.5-27b", "phase": "decode_ar"},
            "suggestions": [
                {
                    "rank": 1,
                    "title": "Try residual multirow GEMV",
                    "lever_type": "dispatch",
                    "risk": "medium",
                    "expected_impact": "medium",
                    "allowed_files": ["crates/rdna-compute/src/dispatch.rs"],
                    "history": {"status": "rejected", "match_count": 1, "best_speedup": 0.97},
                    "rationale": ["hot residual GEMV"],
                    "candidate_steps": ["wire candidate"],
                }
            ],
        }

        text = kernel_atlas.render_suggestion_markdown(queue)

        self.assertIn("# Atlas Suggestions", text)
        self.assertIn("1. Try residual multirow GEMV", text)
        self.assertIn("crates/rdna-compute/src/dispatch.rs", text)
        self.assertIn("history: rejected over 1 prior eval(s), best speedup 0.970x", text)

    def test_write_task_bundle_creates_json_and_agent_markdown(self):
        task = {
            "schema": "hipfire.kernel_atlas.task.v0",
            "task_id": "atlas-test-task",
            "target": {"arch": "gfx1201", "phase": "decode_ar", "workload": "qwen3.5-27b"},
            "hot_kernel": {
                "name": "gemv_hfq4g256_multirow_r4",
                "op": {"family": "linear", "role": "multirow_gemv"},
            },
            "baseline": {"metrics": {"gen_tok_s": 192.69}},
            "constraints": {"allowed_files": ["kernels/src/gemv_hfq4g256_multirow.hip"]},
            "eval": {"metric": "gen_tok_s", "benchmark_command": ["bench"]},
        }
        with tempfile.TemporaryDirectory() as td:
            paths = kernel_atlas.write_task_bundle(task, td)
            task_json = Path(paths["task_json"])
            task_md = Path(paths["task_md"])

            loaded = kernel_atlas.load_json_or_jsonl(str(task_json))
            text = task_md.read_text(encoding="utf-8")

        self.assertEqual(loaded["task_id"], "atlas-test-task")
        self.assertIn("Atlas Optimization Task", text)
        self.assertIn("gemv_hfq4g256_multirow_r4", text)
        self.assertIn("kernels/src/gemv_hfq4g256_multirow.hip", text)

    def test_evaluate_task_bundle_records_result_and_ledger_entry(self):
        task = {
            "schema": "hipfire.kernel_atlas.task.v0",
            "task_id": "atlas-test-task",
            "baseline": {"metrics": {"gen_tok_s": 100.0}},
            "eval": {
                "metric": "gen_tok_s",
                "goal": "maximize",
                "benchmark_command": [
                    "python3",
                    "-c",
                    "print('SUMMARY gen_tok_s=125.0 bw_gib_s=12.0 prefill_tok_s=1.0 avg_ms=2.0 p50_ms=2.0')",
                ],
                "correctness_commands": [["python3", "-c", "print('ok')"]],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            result = kernel_atlas.evaluate_task_bundle(task, output_dir=td)
            ledger = Path(td) / "ledger.jsonl"

            ledger_rows = [kernel_atlas.load_json_or_jsonl(str(ledger))]

        self.assertEqual(result["schema"], "hipfire.kernel_atlas.eval.v0")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["gen_tok_s"], 125.0)
        self.assertEqual(result["delta"]["speedup"], 1.25)
        self.assertEqual(ledger_rows[0]["task_id"], "atlas-test-task")
        self.assertEqual(ledger_rows[0]["status"], "pass")

    def test_run_task_command_supports_shell_chains_for_correctness_commands(self):
        result = kernel_atlas.run_task_command("python3 -c 'print(1)' && python3 -c 'print(2)'")

        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["shell"])
        self.assertIn("1\n2", result["output_tail"])

    def test_run_graph_ab_for_row_records_second_pass_lift(self):
        code = (
            "import os; "
            "g=os.environ.get('HIPFIRE_GRAPH'); "
            "v=100.0 if g=='0' else 125.0; "
            "print(f'SUMMARY gen_tok_s={v} bw_gib_s=1.0 prefill_tok_s={v/10} avg_ms=2.0 p50_ms=2.0')"
        )
        row = {
            "arch": "gfx1100",
            "hostname": "k9lin",
            "workload": "qwen3.5-0.8b-paro4-native",
            "quant": "paro4",
            "phase": "decode_ar",
            "shape_bucket": "decode_ar_pp32_gen16",
            "command": ["python3", "-c", code],
            "variant": {"env": {"HIPFIRE_PROFILE_DECODE": "1", "HIPFIRE_KV_MODE": "q8"}},
        }

        result = kernel_atlas.run_graph_ab_for_row(row)

        self.assertEqual(result["schema"], "hipfire.kernel_atlas.graph_ab.v0")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["stripped_env_keys"], ["HIPFIRE_PROFILE_DECODE"])
        self.assertEqual(result["graph_off"]["headline_pass"], 2)
        self.assertEqual(len(result["graph_off"]["passes"]), 2)
        self.assertEqual(result["graph_off"]["metrics"]["gen_tok_s"], 100.0)
        self.assertEqual(result["graph_on"]["metrics"]["gen_tok_s"], 125.0)
        self.assertEqual(result["delta"]["lift"], 1.25)
        self.assertEqual(result["graph_on"]["route"]["graph_enabled"], True)

    def test_evaluate_task_bundle_uses_median_across_repeated_runs(self):
        command = (
            "from pathlib import Path; "
            "p=Path('counter.txt'); "
            "n=int(p.read_text()) if p.exists() else 0; "
            "vals=[20.0,100.0,104.0,106.0]; "
            "print(f'SUMMARY gen_tok_s={vals[n]} bw_gib_s=1.0 prefill_tok_s=1.0 avg_ms=1.0 p50_ms=1.0'); "
            "p.write_text(str(n+1))"
        )
        task = {
            "schema": "hipfire.kernel_atlas.task.v0",
            "task_id": "atlas-repeat-eval",
            "baseline": {"metrics": {"gen_tok_s": 100.0}, "env": {}},
            "eval": {
                "metric": "gen_tok_s",
                "goal": "maximize",
                "benchmark_command": ["python3", "-c", command],
                "correctness_commands": [],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            result = kernel_atlas.evaluate_task_bundle(
                task,
                output_dir=td,
                runs=3,
                warmup_runs=1,
                cwd=td,
            )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(result["warmup_runs"]), 1)
        self.assertEqual(len(result["benchmark_runs"]), 3)
        self.assertEqual(result["metrics"]["gen_tok_s"], 104.0)
        self.assertEqual(result["summary"]["gen_tok_s"]["median"], 104.0)
        self.assertEqual(result["delta"]["speedup"], 1.04)
        self.assertEqual(result["stability"]["status"], "stable")

    def test_evaluate_task_bundle_marks_high_spread_unstable(self):
        command = (
            "from pathlib import Path; "
            "p=Path('counter.txt'); "
            "n=int(p.read_text()) if p.exists() else 0; "
            "vals=[100.0,200.0,300.0]; "
            "print(f'SUMMARY gen_tok_s={vals[n]} bw_gib_s=1.0 prefill_tok_s=1.0 avg_ms=1.0 p50_ms=1.0'); "
            "p.write_text(str(n+1))"
        )
        task = {
            "schema": "hipfire.kernel_atlas.task.v0",
            "task_id": "atlas-unstable-eval",
            "baseline": {"metrics": {"gen_tok_s": 200.0}, "env": {}},
            "eval": {
                "metric": "gen_tok_s",
                "goal": "maximize",
                "benchmark_command": ["python3", "-c", command],
                "correctness_commands": [],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            result = kernel_atlas.evaluate_task_bundle(
                task,
                output_dir=td,
                runs=3,
                cwd=td,
                max_rel_spread=0.25,
            )

        self.assertEqual(result["status"], "unstable")
        self.assertEqual(result["metrics"]["gen_tok_s"], 200.0)
        self.assertGreater(result["summary"]["gen_tok_s"]["rel_spread"], 0.25)
        self.assertEqual(result["stability"]["status"], "unstable")

    def test_evaluate_task_bundle_refreshes_baseline_json(self):
        task = {
            "schema": "hipfire.kernel_atlas.task.v0",
            "task_id": "atlas-refresh-baseline",
            "baseline": {"metrics": {"gen_tok_s": 10.0}, "env": {}},
            "eval": {
                "metric": "gen_tok_s",
                "goal": "maximize",
                "benchmark_command": [
                    "python3",
                    "-c",
                    "print('SUMMARY gen_tok_s=104.0 bw_gib_s=1.0 prefill_tok_s=1.0 avg_ms=1.0 p50_ms=1.0')",
                ],
                "correctness_commands": [],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            result = kernel_atlas.evaluate_task_bundle(
                task,
                output_dir=td,
                runs=3,
                refresh_baseline=True,
            )
            baseline = kernel_atlas.load_json_or_jsonl(str(Path(td) / "baseline.json"))

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["baseline"]["source"], "refreshed")
        self.assertEqual(baseline["schema"], "hipfire.kernel_atlas.baseline.v0")
        self.assertEqual(baseline["metrics"]["gen_tok_s"], 104.0)
        self.assertEqual(baseline["summary"]["gen_tok_s"]["median"], 104.0)
        self.assertEqual(result["delta"]["speedup"], 1.0)

    def test_evaluate_task_bundle_requires_fresh_baseline_when_task_env_was_cleaned(self):
        task = {
            "schema": "hipfire.kernel_atlas.task.v0",
            "task_id": "atlas-needs-baseline",
            "baseline": {"metrics": {"gen_tok_s": 22.1}, "env": {"HIPFIRE_KV_MODE": "asym3"}},
            "eval": {
                "metric": "gen_tok_s",
                "goal": "maximize",
                "requires_fresh_baseline": True,
                "benchmark_command": [
                    "python3",
                    "-c",
                    "print('SUMMARY gen_tok_s=35.2 bw_gib_s=1.0 prefill_tok_s=1.0 avg_ms=1.0 p50_ms=1.0')",
                ],
                "correctness_commands": [],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            result = kernel_atlas.evaluate_task_bundle(task, output_dir=td)

        self.assertEqual(result["status"], "needs_baseline")
        self.assertEqual(result["baseline"]["source"], "missing-required")
        self.assertIsNone(result["delta"]["speedup"])

    def test_build_pytorch_task_bundle_records_shape_eval_contract(self):
        task = kernel_atlas.build_pytorch_task_bundle(
            name="llama-rmsnorm-shape",
            op="rmsnorm",
            input_shapes=["1,2048,4096"],
            dtype="float16",
            eval_command=["python3", "bench_rmsnorm.py"],
            allowed_files=["kernels/src/rmsnorm_candidate.hip"],
            task_id="atlas-pytorch-task",
        )

        self.assertEqual(task["schema"], "hipfire.kernel_atlas.task.v0")
        self.assertEqual(task["producer"]["kind"], "pytorch")
        self.assertEqual(task["producer"]["op"], "rmsnorm")
        self.assertEqual(task["producer"]["input_shapes"], ["1,2048,4096"])
        self.assertEqual(task["producer"]["dtype"], "float16")
        self.assertEqual(task["eval"]["benchmark_command"], ["python3", "bench_rmsnorm.py"])
        self.assertEqual(task["constraints"]["allowed_files"], ["kernels/src/rmsnorm_candidate.hip"])

    def test_task_cli_writes_bundle_from_row_and_isa(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row_path = root / "row.jsonl"
            isa_path = root / "isa.json"
            out_dir = root / "task-out"
            row = {
                "arch": "gfx1201",
                "hostname": "hiptrx",
                "workload": "qwen3.5-27b",
                "phase": "decode_ar",
                "shape_bucket": "decode_ar_pp32_gen50",
                "metrics": {"gen_tok_s": 100.0},
                "command": ["python3", "-c", "print('gen_tok_s=101.0')"],
                "variant": {"env": {}},
                "artifacts": {
                    "profile_kernels": kernel_atlas.annotate_profile_kernels(
                        [{"name": "gemv_hfq4g256_multirow_r4", "pct": 36.3}]
                    )
                },
            }
            isa = {
                "objects": [
                    {
                        "path": "kernel.hsaco",
                        "kernels": [{"name": "gemv_hfq4g256_multirow_r4"}],
                        "instruction_summary": {"instruction_count": 1, "category_counts": {"valu": 1}},
                    }
                ]
            }
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            isa_path.write_text(json.dumps(isa), encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    str(ATLAS_PATH),
                    "task",
                    "--row",
                    str(row_path),
                    "--isa",
                    str(isa_path),
                    "--allowed-file",
                    "kernels/src/gemv_hfq4g256_multirow.hip",
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            task = kernel_atlas.load_json_or_jsonl(str(out_dir / "task.json"))

        self.assertEqual(task["schema"], "hipfire.kernel_atlas.task.v0")
        self.assertEqual(task["target"]["host"], "hiptrx")
        self.assertEqual(task["constraints"]["allowed_files"], ["kernels/src/gemv_hfq4g256_multirow.hip"])

    def test_task_cli_without_allowed_file_defaults_to_top_ranked_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row_path = root / "row.jsonl"
            isa_path = root / "isa.json"
            dispatch_path = root / "dispatch.json"
            out_dir = root / "task-out"
            row = {
                "arch": "gfx1201",
                "hostname": "hiptrx",
                "workload": "qwen3.5-27b",
                "phase": "decode_ar",
                "shape_bucket": "decode_ar_pp32_gen50",
                "metrics": {"gen_tok_s": 100.0},
                "command": ["python3", "-c", "print('gen_tok_s=101.0')"],
                "variant": {"env": {}},
                "artifacts": {
                    "profile_kernels": kernel_atlas.annotate_profile_kernels(
                        [{"name": "gemv_hfq4g256", "pct": 36.3}]
                    )
                },
            }
            isa = {"objects": []}
            dispatch = {
                "entries": [
                    {
                        "name": "gemv_hfq4g256",
                        "source_files": [
                            {"path": "kernels/src/gemv_hfq4g256.gfx1201.hip"},
                            {"path": "kernels/src/gemv_hfq4g256.hip"},
                        ],
                    }
                ]
            }
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            isa_path.write_text(json.dumps(isa), encoding="utf-8")
            dispatch_path.write_text(json.dumps(dispatch), encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    str(ATLAS_PATH),
                    "task",
                    "--row",
                    str(row_path),
                    "--isa",
                    str(isa_path),
                    "--dispatch",
                    str(dispatch_path),
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            task = kernel_atlas.load_json_or_jsonl(str(out_dir / "task.json"))

        self.assertEqual(task["constraints"]["allowed_files"], ["kernels/src/gemv_hfq4g256.gfx1201.hip"])

    def test_task_cli_can_build_from_ranked_suggestion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row_path = root / "row.jsonl"
            isa_path = root / "isa.json"
            dispatch_path = root / "dispatch.json"
            out_dir = root / "task-out"
            row = {
                "arch": "gfx1100",
                "hostname": "k9lin",
                "workload": "qwen3.5-0.8b-paro4-native",
                "quant": "paro4",
                "phase": "decode_ar",
                "shape_bucket": "decode_ar_pp32_gen16",
                "metrics": {"gen_tok_s": 72.5},
                "command": ["python3", "-c", "print('gen_tok_s=73.0')"],
                "variant": {"env": {"HIPFIRE_PROFILE_DECODE": "1", "HIPFIRE_GRAPH": "0"}},
                "artifacts": {
                    "profile_kernels": kernel_atlas.annotate_profile_kernels(
                        [{"name": "rmsnorm_f32", "pct": 24.3}]
                    )
                },
            }
            isa = {
                "objects": [
                    {
                        "kernels": [{"name": "gemv_paro4g128"}],
                        "instruction_summary": {"instruction_count": 10, "category_counts": {"valu": 10}},
                    }
                ]
            }
            dispatch = {
                "entries": [
                    {
                        "name": "rmsnorm_f32",
                        "source_files": [{"path": "kernels/src/fused_rmsnorm_mq_rotate.hip"}],
                        "dispatch_refs": [{"path": "crates/hipfire-runtime/src/llama.rs", "line": 679}],
                    }
                ]
            }
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            isa_path.write_text(json.dumps(isa), encoding="utf-8")
            dispatch_path.write_text(json.dumps(dispatch), encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    str(ATLAS_PATH),
                    "task",
                    "--row",
                    str(row_path),
                    "--isa",
                    str(isa_path),
                    "--dispatch",
                    str(dispatch_path),
                    "--suggestion-rank",
                    "1",
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            task = kernel_atlas.load_json_or_jsonl(str(out_dir / "task.json"))
            text = (out_dir / "TASK.md").read_text(encoding="utf-8")

        self.assertEqual(task["producer"]["kind"], "hipfire-atlas-suggestion")
        self.assertEqual(task["producer"]["suggestion"]["rank"], 1)
        self.assertEqual(task["experiment"]["lever_type"], "fusion")
        self.assertIn("kernels/src/fused_rmsnorm_mq_rotate.hip", task["constraints"]["allowed_files"])
        self.assertIn("## Suggested Experiment", text)

    def test_suggest_cli_outputs_ranked_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row_path = root / "row.jsonl"
            isa_path = root / "isa.json"
            dispatch_path = root / "dispatch.json"
            row = {
                "arch": "gfx1201",
                "hostname": "hiptrx",
                "workload": "qwen3.5-27b",
                "quant": "mq4",
                "phase": "decode_ar",
                "shape_bucket": "decode_ar_pp32_gen50",
                "metrics": {"gen_tok_s": 35.2},
                "command": ["bench"],
                "variant": {"env": {"HIPFIRE_PROFILE": "1"}},
                "artifacts": {
                    "profile_kernels": kernel_atlas.annotate_profile_kernels(
                        [{"name": "gemv_hfq4g256_residual", "pct": 36.3}]
                    )
                },
            }
            isa = {"objects": []}
            dispatch = {
                "entries": [
                    {
                        "name": "gemv_hfq4g256_residual",
                        "source_files": [{"path": "kernels/src/gemv_hfq4g256_residual.hip"}],
                        "dispatch_refs": [{"path": "crates/hipfire-runtime/src/llama.rs", "line": 790}],
                    }
                ]
            }
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            isa_path.write_text(json.dumps(isa), encoding="utf-8")
            dispatch_path.write_text(json.dumps(dispatch), encoding="utf-8")

            proc = subprocess.run(
                [
                    "python3",
                    str(ATLAS_PATH),
                    "suggest",
                    "--row",
                    str(row_path),
                    "--isa",
                    str(isa_path),
                    "--dispatch",
                    str(dispatch_path),
                    "--max-suggestions",
                    "4",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            queue = json.loads(proc.stdout)

        self.assertEqual(queue["schema"], "hipfire.kernel_atlas.suggestions.v0")
        self.assertGreaterEqual(queue["suggestion_count"], 2)
        self.assertEqual(queue["suggestions"][0]["rank"], 1)

    def test_suggest_cli_auto_loads_default_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row_path = root / "row.jsonl"
            isa_path = root / "isa.json"
            dispatch_path = root / "dispatch.json"
            history_root = root / ".codeinsight+research" / "kernel-atlas" / "tasks" / "residual"
            eval_dir = history_root / "eval-suggest-r2-3x"
            eval_dir.mkdir(parents=True)
            task_id = "hiptrx-gfx1201-27b-ar-gemv-residual-atlas-ready-suggest-r2"
            row = {
                "arch": "gfx1201",
                "hostname": "hiptrx",
                "workload": "qwen3.5-27b",
                "quant": "mq4",
                "phase": "decode_ar",
                "shape_bucket": "decode_ar_pp32_gen50",
                "metrics": {"gen_tok_s": 35.3},
                "command": ["bench"],
                "variant": {"env": {"HIPFIRE_KV_MODE": "asym3"}},
                "artifacts": {
                    "profile_kernels": kernel_atlas.annotate_profile_kernels(
                        [
                            {"name": "gemv_hfq4g256_residual", "pct": 36.3},
                            {"name": "fused_rmsnorm_mq_rotate", "pct": 10.2},
                        ]
                    )
                },
            }
            dispatch = {
                "entries": [
                    {
                        "name": "gemv_hfq4g256_residual",
                        "source_files": [{"path": "kernels/src/gemv_hfq4g256_residual.hip"}],
                        "dispatch_refs": [{"path": "crates/rdna-compute/src/dispatch.rs", "line": 6231}],
                        "env_controls": ["HIPFIRE_GEMV_ROWS"],
                    },
                    {
                        "name": "fused_rmsnorm_mq_rotate",
                        "source_files": [{"path": "kernels/src/fused_rmsnorm_mq_rotate.hip"}],
                        "dispatch_refs": [{"path": "crates/rdna-compute/src/kernels.rs", "line": 269}],
                    },
                ]
            }
            task = {
                "task_id": task_id,
                "target": {
                    "arch": "gfx1201",
                    "workload": "qwen3.5-27b",
                    "phase": "decode_ar",
                    "shape_bucket": "decode_ar_pp32_gen50",
                    "quant": "mq4",
                },
                "hot_kernel": {"name": "gemv_hfq4g256_residual"},
                "baseline": {"env": {"HIPFIRE_GEMV_ROWS": "2", "HIPFIRE_KV_MODE": "asym3"}},
                "constraints": {"allowed_files": ["kernels/src/gemv_hfq4g256_residual.hip"]},
            }
            result = {
                "schema": "hipfire.kernel_atlas.eval.v0",
                "task_id": task_id,
                "status": "pass",
                "metric": "gen_tok_s",
                "metrics": {"gen_tok_s": 35.2},
                "delta": {"speedup": 0.997},
                "stability": {"status": "stable"},
            }
            row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            isa_path.write_text(json.dumps({"objects": []}), encoding="utf-8")
            dispatch_path.write_text(json.dumps(dispatch), encoding="utf-8")
            (history_root / "task-suggest-r2.json").write_text(json.dumps(task), encoding="utf-8")
            (eval_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

            proc = subprocess.run(
                [
                    "python3",
                    str(ATLAS_PATH),
                    "suggest",
                    "--row",
                    str(row_path),
                    "--isa",
                    str(isa_path),
                    "--dispatch",
                    str(dispatch_path),
                    "--max-suggestions",
                    "6",
                ],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            )
            queue = json.loads(proc.stdout)

        rejected = next(item for item in queue["suggestions"] if item["title"].startswith("Port/test gfx12 residual"))
        self.assertGreaterEqual(queue["history"]["demoted_suggestions"], 1)
        self.assertEqual(rejected["history"]["status"], "rejected")
        self.assertGreater(rejected["rank"], 1)

    def test_eval_cli_runs_task_and_writes_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_path = root / "task.json"
            out_dir = root / "eval-out"
            task = {
                "schema": "hipfire.kernel_atlas.task.v0",
                "task_id": "atlas-cli-eval",
                "baseline": {"metrics": {"gen_tok_s": 100.0}, "env": {}},
                "eval": {
                    "metric": "gen_tok_s",
                    "goal": "maximize",
                    "benchmark_command": [
                        "python3",
                        "-c",
                        "print('SUMMARY gen_tok_s=110.0 bw_gib_s=1.0 prefill_tok_s=1.0 avg_ms=1.0 p50_ms=1.0')",
                    ],
                    "correctness_commands": [],
                },
            }
            task_path.write_text(json.dumps(task), encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    str(ATLAS_PATH),
                    "eval",
                    "--task",
                    str(task_path),
                    "--output-dir",
                    str(out_dir),
                    "--runs",
                    "2",
                    "--warmup-runs",
                    "1",
                    "--refresh-baseline",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            result = kernel_atlas.load_json_or_jsonl(str(out_dir / "result.json"))
            baseline_exists = (out_dir / "baseline.json").exists()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["gen_tok_s"], 110.0)
        self.assertEqual(result["summary"]["gen_tok_s"]["median"], 110.0)
        self.assertTrue(baseline_exists)

    def test_task_pytorch_cli_writes_shape_task(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "task-pytorch"

            subprocess.run(
                [
                    "python3",
                    str(ATLAS_PATH),
                    "task-pytorch",
                    "--name",
                    "rmsnorm-shape",
                    "--op",
                    "rmsnorm",
                    "--input-shape",
                    "1,2048,4096",
                    "--dtype",
                    "float16",
                    "--eval-command",
                    "python3 bench_rmsnorm.py",
                    "--allowed-file",
                    "kernels/src/rmsnorm_candidate.hip",
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            task = kernel_atlas.load_json_or_jsonl(str(out_dir / "task.json"))

        self.assertEqual(task["producer"]["kind"], "pytorch")
        self.assertEqual(task["producer"]["input_shapes"], ["1,2048,4096"])
        self.assertEqual(task["eval"]["benchmark_command"], ["python3", "bench_rmsnorm.py"])

    def test_load_json_or_jsonl_reads_indexed_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as td:
            rows = Path(td) / "rows.jsonl"
            rows.write_text('{"phase":"prefill"}\n{"phase":"decode_ar"}\n', encoding="utf-8")

            row = kernel_atlas.load_json_or_jsonl(str(rows), index=1)

        self.assertEqual(row["phase"], "decode_ar")

    def test_load_json_or_jsonl_reads_pretty_json_documents(self):
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "manifest.json"
            doc.write_text('{\n  "schema": "hipfire.kernel_atlas.isa.v0",\n  "objects": []\n}\n', encoding="utf-8")

            manifest = kernel_atlas.load_json_or_jsonl(str(doc))

        self.assertEqual(manifest["schema"], "hipfire.kernel_atlas.isa.v0")


if __name__ == "__main__":
    unittest.main()
