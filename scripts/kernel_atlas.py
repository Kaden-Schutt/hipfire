#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Phase-aware Kernel Atlas collector for hipfire benches.

This intentionally starts as a measurement harness. It does not rewrite kernels
or choose winners; it turns existing AR and DFlash benchmark output into JSONL
rows that can become the Atlas corpus.
"""

import argparse
import collections
import copy
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCHEMA = "hipfire.kernel_atlas.v0"
TAU = chr(0x03C4)
TUNING_STRIP_ENV_KEYS = {
    "HIPFIRE_HOST_TIMING",
    "HIPFIRE_PROFILE",
    "HIPFIRE_PROFILE_DECODE",
    "HIPFIRE_PROMPT_HEAT_JSON",
    "HIPFIRE_PROMPT_TOKEN_HEAT",
}
DEFAULT_HISTORY_DIR = Path(".codeinsight+research") / "kernel-atlas" / "tasks"
HISTORY_REJECT_SPEEDUP = 1.0
HISTORY_REJECT_DEMOTE = 48.0


def parse_bench_summary(text):
    """Parse the bench_qwen35_mq4 SUMMARY line."""
    summary = None
    for line in text.splitlines():
        if line.startswith("SUMMARY"):
            summary = line
    if summary is None:
        raise ValueError("bench output did not contain a SUMMARY line")

    values = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([0-9.]+)", summary):
        values[key] = float(value)
    required = ["gen_tok_s", "bw_gib_s", "prefill_tok_s", "avg_ms", "p50_ms"]
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"bench SUMMARY missing keys: {', '.join(missing)}")
    return values


def parse_bench_profile_sections(text):
    """Parse HIPFIRE_PROFILE/HIPFIRE_PROFILE_DECODE kernel tables."""
    sections = {"prefill": [], "decode_ar": []}
    current = None
    line_re = re.compile(
        r"^\s+([A-Za-z0-9_.$]+)\s+"
        r"([0-9]+)x\s+"
        r"([0-9.]+)ms\s+"
        r"\(([0-9.]+)(?:\u00b5|u)s/call\)\s+"
        r"([0-9.]+)%\s+"
        r"([0-9.]+)\s+GiB/s"
    )

    for line in text.splitlines():
        if line.startswith("=== DECODE PROFILE"):
            current = "decode_ar"
            continue
        if line.startswith("=== PROFILE"):
            current = "prefill"
            continue
        if line.startswith("===") and "PROFILE" not in line:
            current = None
            continue
        if current is None:
            continue
        match = line_re.match(line)
        if not match:
            continue
        name, calls, total_ms, avg_us, pct, gib_s = match.groups()
        sections[current].append(
            {
                "name": name,
                "calls": int(calls),
                "total_ms": float(total_ms),
                "avg_us": float(avg_us),
                "pct": float(pct),
                "gib_s": float(gib_s),
            }
        )
    return sections


def classify_kernel_op(name):
    """Best-effort op attribution from hipfire kernel naming conventions."""
    lowered = (name or "").lower()
    if "fused_qkvza" in lowered:
        return {"family": "attention", "role": "qkvza_projection", "phase_hint": "prefill/decode"}
    if "fused_qkv" in lowered:
        return {"family": "attention", "role": "qkv_projection", "phase_hint": "prefill/decode"}
    if "qk_l2_norm" in lowered:
        return {"family": "attention", "role": "qk_norm_scale", "phase_hint": "decode"}
    if "attention_flash" in lowered or "flash" in lowered:
        return {"family": "attention", "role": "flash_attention", "phase_hint": "prefill/decode"}
    if "kv_cache" in lowered:
        return {"family": "attention", "role": "kv_cache", "phase_hint": "decode"}
    if "rope" in lowered or "rotate" in lowered:
        return {"family": "position", "role": "rope_rotate", "phase_hint": "prefill/decode"}
    if "rmsnorm" in lowered or "norm" in lowered:
        return {"family": "norm", "role": "normalization", "phase_hint": "prefill/decode"}
    if "gate_up" in lowered:
        return {"family": "mlp", "role": "gate_up_projection", "phase_hint": "prefill/decode"}
    if "swiglu" in lowered:
        return {"family": "mlp", "role": "swiglu", "phase_hint": "prefill/decode"}
    if "gemv" in lowered and "residual" in lowered:
        return {"family": "linear", "role": "residual_gemv", "phase_hint": "decode"}
    if "gemv" in lowered and "multirow" in lowered:
        return {"family": "linear", "role": "multirow_gemv", "phase_hint": "decode"}
    if "gemv" in lowered:
        return {"family": "linear", "role": "gemv", "phase_hint": "decode"}
    if "gemm" in lowered and "residual" in lowered:
        return {"family": "linear", "role": "residual_gemm", "phase_hint": "prefill"}
    if "gemm" in lowered:
        return {"family": "linear", "role": "gemm", "phase_hint": "prefill"}
    return {"family": "unknown", "role": "unknown", "phase_hint": "unknown"}


def annotate_profile_kernels(kernels):
    annotated = []
    for item in kernels or []:
        row = copy.deepcopy(item) if isinstance(item, dict) else {"name": str(item)}
        row.setdefault("op", classify_kernel_op(row.get("name")))
        annotated.append(row)
    return annotated


def annotate_rocprof_coverage(row):
    """Add BLINDSPOT tag to rows where rocprof_blindspot_count > 0."""
    if not isinstance(row, dict):
        return row
    metrics = row.get("metrics", {}) or {}
    count = metrics.get("rocprof_blindspot_count", 0)
    if count and count > 0:
        row.setdefault("tags", [])
        if "BLINDSPOT" not in row["tags"]:
            row["tags"].append("BLINDSPOT")
    return row


def parse_rocprof_coverage_section(text):
    """Parse the rocprofv3 coverage block from bench stderr output."""
    coverage = {}
    match = re.search(r"=== ROCPROF COVERAGE \(([0-9.]+)%\) ===", text)
    if match:
        coverage["coverage_pct"] = float(match.group(1))
    match = re.search(r"rocprof total:\s+([0-9.]+)ms", text)
    if match:
        coverage["rocprof_total_ms"] = float(match.group(1))
    match = re.search(r"internal total:\s+([0-9.]+)ms", text)
    if match:
        coverage["internal_total_ms"] = float(match.group(1))
    match = re.search(r"blindspot total:\s+([0-9.]+)ms\s+\(([0-9]+) (?:un-tracked )?kernels?\)", text)
    if match:
        coverage["blindspot_total_ms"] = float(match.group(1))
        coverage["blindspot_count"] = int(match.group(2))
    return coverage


def parse_dflash_summary(text):
    """Parse dflash_spec_demo metrics from authoritative lines or fallback text."""
    metrics = {}

    line_patterns = {
        "decode_tok_s": r"^decode_tok_s:\s*([0-9.]+)",
        "tau": r"^decode_tau:\s*([0-9.]+)",
        "ttft_ms": r"^ttft_ms:\s*([0-9.]+)",
    }
    for key, pattern in line_patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            metrics[key] = float(match.group(1))

    emitted = re.search(
        r"emitted:\s*([0-9]+)\s+tokens\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+tok/s\)",
        text,
    )
    if emitted:
        metrics["emitted_tokens"] = int(emitted.group(1))
        metrics["elapsed_s"] = float(emitted.group(2))
        metrics.setdefault("decode_tok_s", float(emitted.group(3)))

    if "tau" not in metrics:
        tau_match = re.search(r"(?:tau|" + re.escape(TAU) + r")=([0-9.]+)", text)
        if tau_match:
            metrics["tau"] = float(tau_match.group(1))

    for key in ("cycles", "accepted"):
        match = re.search(r"^" + key + r":\s*([0-9]+)", text, flags=re.MULTILINE)
        if match:
            metrics[key] = int(match.group(1))

    required = ["decode_tok_s", "tau"]
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"dflash output missing metrics: {', '.join(missing)}")
    return metrics


def md5_file(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_text(text):
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def parse_graph_blob_count(text):
    match = re.search(r"\[hipGraph\]\s+captured\s+([0-9]+)\s+blobs", text or "")
    return int(match.group(1)) if match else None


def normalize_flash_requested(env):
    raw = str((env or {}).get("HIPFIRE_ATTN_FLASH", "auto")).strip().lower()
    if raw in ("2", "always", "force", "forced"):
        return "always"
    if raw in ("0", "never", "off", "false"):
        return "never"
    return "auto"


def infer_attention_route(kv_mode, env):
    kv = str(kv_mode or (env or {}).get("HIPFIRE_KV_MODE", "")).strip().lower()
    flash_requested = normalize_flash_requested(env)
    if kv in ("asym2", "asym3", "asym4"):
        return {
            "attention_impl": f"attention_flash_{kv}",
            "flash_active": True,
            "flash_requested": flash_requested,
            "route_confidence": "static_kv_policy",
        }
    if kv in ("fwht2", "fwht3", "fwht4"):
        return {
            "attention_impl": f"attention_flash_{kv}",
            "flash_active": True,
            "flash_requested": flash_requested,
            "route_confidence": "static_kv_policy",
        }
    if kv == "q8":
        if flash_requested == "always":
            return {
                "attention_impl": "attention_flash_q8_0",
                "flash_active": True,
                "flash_requested": flash_requested,
                "route_confidence": "forced_env",
            }
        return {
            "attention_impl": "attention_q8_0_kv",
            "flash_active": False,
            "flash_requested": flash_requested,
            "route_confidence": "short_context_default",
        }
    return {
        "attention_impl": "unknown",
        "flash_active": None,
        "flash_requested": flash_requested,
        "route_confidence": "unknown",
    }


def build_route_manifest(*, kv_mode, env, output, graph_enabled):
    route = infer_attention_route(kv_mode, env)
    route.update(
        {
            "kv_mode": kv_mode,
            "graph_enabled": bool(graph_enabled),
            "graph_blob_count": parse_graph_blob_count(output),
        }
    )
    if route["graph_enabled"] and route["attention_impl"] == "attention_flash_q8_0":
        route["warnings"] = [
            "q8 graph capture routed to q8 flash attention; verify logits/coherence before trusting perf"
        ]
    return route


def git_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return "unknown"


def git_diff_text():
    parts = []
    for command in (
        ["git", "status", "--porcelain=v1"],
        ["git", "diff", "--binary", "HEAD", "--"],
    ):
        try:
            out = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            out = ""
        parts.append(out)
    return "\n".join(parts)


def git_is_dirty(diff_text=None):
    if diff_text is None:
        diff_text = git_diff_text()
    return bool(diff_text.strip())


def detect_arch():
    for name in ("HIPFIRE_BASELINE_ARCH", "HIPFIRE_TARGET_ARCH"):
        value = os.environ.get(name)
        if value:
            return value
    for probe in ("amdgpu-arch", "offload-arch", "/opt/rocm/bin/amdgpu-arch"):
        try:
            out = subprocess.check_output([probe], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for line in out.splitlines():
            if line.startswith("gfx"):
                return line.strip()
    return "unknown"


def find_tool(candidates):
    for candidate in candidates:
        if "/" in candidate:
            path = Path(candidate)
            if path.exists() and os.access(path, os.X_OK):
                return str(path)
            continue
        found = shutil.which(candidate)
        if found:
            return found
    return None


def parse_isa_scalar(value):
    value = value.strip()
    if not value:
        return None
    if value in ("true", "false"):
        return value == "true"
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    return value


def parse_isa_metadata(text):
    """Extract kernel resource metadata from llvm-readobj --notes output."""
    target = None
    target_match = re.search(r"amdhsa\.target:\s+(\S+)", text)
    if target_match:
        target = target_match.group(1)

    wanted = {
        "name",
        "symbol",
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "kernarg_segment_size",
        "max_flat_workgroup_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "wavefront_size",
        "workgroup_processor_mode",
        "uses_dynamic_stack",
    }
    kernels = []
    current = None
    in_kernels = False

    for line in text.splitlines():
        if re.match(r"\s*amdhsa\.kernels:", line):
            in_kernels = True
            continue
        if in_kernels and re.match(r"\s*amdhsa\.target:", line):
            if current and current.get("name"):
                kernels.append(current)
            current = None
            in_kernels = False
            continue
        if not in_kernels:
            continue

        item = re.match(r"^\s{2}-\s+\.([A-Za-z0-9_]+):\s*(.*)$", line)
        if item:
            if current and current.get("name"):
                kernels.append(current)
            current = {}
            key, value = item.group(1), item.group(2)
            if key in wanted and value.strip():
                current[key] = parse_isa_scalar(value)
            continue

        field = re.match(r"^\s{4}\.([A-Za-z0-9_]+):\s*(.*)$", line)
        if current is not None and field:
            key, value = field.group(1), field.group(2)
            if key in wanted:
                parsed = parse_isa_scalar(value)
                if parsed is not None:
                    current[key] = parsed

    if current and current.get("name"):
        kernels.append(current)

    return {
        "amdhsa_target": target,
        "kernels": kernels,
    }


def isa_category(opcode):
    if "mfma" in opcode or "wmma" in opcode:
        return "matrix"
    if "dot" in opcode:
        return "dot"
    if opcode.startswith(("global_", "buffer_", "flat_", "scratch_")):
        return "vmem"
    if opcode.startswith(("s_load", "s_buffer")):
        return "smem"
    if opcode.startswith("ds_"):
        return "lds"
    if opcode.startswith(("s_cbranch", "s_branch")) or opcode.endswith("branch"):
        return "branch"
    if opcode.startswith("s_waitcnt"):
        return "wait"
    if opcode.startswith("s_"):
        return "salu"
    if opcode.startswith("v_"):
        return "valu"
    if opcode.startswith("exp"):
        return "export"
    return "other"


def parse_disassembly_stats(text):
    """Summarize llvm-objdump disassembly into stable ISA counters."""
    opcodes = collections.Counter()
    categories = collections.Counter()
    kernel_symbols = []

    for line in text.splitlines():
        symbol = re.search(r"^[0-9a-fA-F]+\s+<([^>]+)>:", line.strip())
        if symbol:
            kernel_symbols.append(symbol.group(1))
            continue
        stripped = line.strip()
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*)\b", stripped)
        if not match:
            continue
        opcode = match.group(1)
        if opcode in ("Disassembly", "File", "Format"):
            continue
        opcodes[opcode] += 1
        categories[isa_category(opcode)] += 1

    return {
        "instruction_count": sum(opcodes.values()),
        "kernel_symbols": kernel_symbols,
        "opcode_counts": dict(sorted(opcodes.items())),
        "category_counts": dict(sorted(categories.items())),
    }


def run_tool(command):
    proc = subprocess.run(command, text=True, capture_output=True)
    return proc.returncode, proc.stdout + proc.stderr


def list_bundle_targets(path, bundler):
    if not bundler:
        return []
    code, out = run_tool([bundler, "--list", "--type=o", f"--input={path}"])
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def choose_bundle_target(targets, arch):
    if not targets:
        return None
    for target in targets:
        if arch and target.endswith(f"--{arch}"):
            return target
    for target in targets:
        if "amdgcn-amd-amdhsa" in target:
            return target
    return targets[0]


def inspect_isa_object(path, arch=None, objdump=None, readobj=None, bundler=None):
    """Inspect one HSACO/code object with ROCm LLVM tools."""
    objdump = objdump or find_tool(["llvm-objdump", "/opt/rocm/llvm/bin/llvm-objdump"])
    readobj = readobj or find_tool(["llvm-readobj", "/opt/rocm/llvm/bin/llvm-readobj"])
    bundler = bundler or find_tool(["clang-offload-bundler", "/opt/rocm/llvm/bin/clang-offload-bundler"])

    path = str(path)
    result = {
        "path": path,
        "md5": md5_file(path),
        "bundle_targets": [],
        "bundle_target": None,
        "amdhsa_target": None,
        "kernels": [],
        "instruction_summary": {
            "instruction_count": 0,
            "kernel_symbols": [],
            "opcode_counts": {},
            "category_counts": {},
        },
        "tool_errors": [],
    }
    if not objdump or not readobj:
        result["tool_errors"].append("missing llvm-objdump or llvm-readobj")
        return result

    with tempfile.TemporaryDirectory(prefix="hipfire-atlas-isa-") as td:
        input_path = path
        targets = list_bundle_targets(path, bundler)
        result["bundle_targets"] = targets
        chosen = choose_bundle_target(targets, arch)
        if chosen:
            extracted = str(Path(td) / "amdgpu.o")
            code, out = run_tool(
                [
                    bundler,
                    "--unbundle",
                    "--type=o",
                    f"--targets={chosen}",
                    f"--input={path}",
                    f"--output={extracted}",
                ]
            )
            if code == 0:
                input_path = extracted
                result["bundle_target"] = chosen
            else:
                result["tool_errors"].append(out.strip()[-500:])

        code, out = run_tool([readobj, "--notes", input_path])
        if code == 0:
            metadata = parse_isa_metadata(out)
            result["amdhsa_target"] = metadata["amdhsa_target"]
            result["kernels"] = metadata["kernels"]
        else:
            result["tool_errors"].append(out.strip()[-500:])

        code, out = run_tool([objdump, "-d", "--no-show-raw-insn", input_path])
        if code == 0:
            result["instruction_summary"] = parse_disassembly_stats(out)
        else:
            result["tool_errors"].append(out.strip()[-500:])

    return result


def discover_isa_paths(files, dirs, name_filter=None, limit=None):
    seen = set()
    paths = []
    for file_name in files or []:
        p = Path(file_name)
        if p.is_file() and str(p) not in seen:
            paths.append(p)
            seen.add(str(p))
    for dir_name in dirs or []:
        root = Path(dir_name)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix not in (".hsaco", ".co", ".o"):
                continue
            if str(p) in seen:
                continue
            if name_filter and not re.search(name_filter, str(p)):
                continue
            paths.append(p)
            seen.add(str(p))
            if limit and len(paths) >= limit:
                return paths
    return paths[:limit] if limit else paths


def collect_isa_manifest(*, arch, files=None, dirs=None, name_filter=None, limit=None):
    paths = discover_isa_paths(files, dirs, name_filter=name_filter, limit=limit)
    objects = [inspect_isa_object(path, arch=arch) for path in paths]
    kernel_count = sum(len(obj.get("kernels", [])) for obj in objects)
    instruction_count = sum(
        obj.get("instruction_summary", {}).get("instruction_count", 0) for obj in objects
    )
    return {
        "schema": "hipfire.kernel_atlas.isa.v0",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": arch,
        "object_count": len(objects),
        "kernel_count": kernel_count,
        "instruction_count": instruction_count,
        "objects": objects,
    }


def kernel_name_variants(name):
    variants = []
    for candidate in (name or "", str(name or "").split(".kd", 1)[0]):
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
    for pattern in (r"_r[0-9]+$", r"_k[0-9]+$", r"_v[0-9]+$", r"_default$"):
        for candidate in list(variants):
            stripped = re.sub(pattern, "", candidate)
            if stripped and stripped not in variants:
                variants.append(stripped)
    return variants


def relpath(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def token_occurs(text, needle):
    if not text or not needle:
        return False
    return (
        re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(needle) + r"(?![A-Za-z0-9_])",
            text,
        )
        is not None
    )


@functools.lru_cache(maxsize=4096)
def read_text_lossy(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def text_contains_any(path, needles):
    text = read_text_lossy(path)
    return any(token_occurs(text, needle) for needle in needles)


@functools.lru_cache(maxsize=128)
def candidate_file_paths(search_roots, suffixes):
    paths = []
    suffix_set = set(suffixes)
    for search_root_name in search_roots:
        search_root = Path(search_root_name)
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.rglob("*")):
            if path.is_file() and path.suffix in suffix_set:
                paths.append(str(path))
    return tuple(paths)


def source_file_score(path, root, variants, arch=None):
    primary = variants[0] if variants else ""
    stem = Path(path).stem
    rel = relpath(path, root)
    text = read_text_lossy(path)
    score = 0
    reasons = []

    if arch and primary and (stem == f"{primary}.{arch}" or stem.startswith(f"{primary}.{arch}.")):
        score += 1200
        reasons.append("target-arch-stem")
    elif arch and any(
        stem == f"{variant}.{arch}" or stem.startswith(f"{variant}.{arch}.")
        for variant in variants[1:]
    ):
        score += 1050
        reasons.append("variant-target-arch-stem")
    elif primary and stem == primary:
        score += 1000
        reasons.append("exact-stem")
    elif primary and stem.startswith(primary + "."):
        score += 950
        reasons.append("arch-stem")
    else:
        for variant in variants[1:]:
            if stem == variant:
                score += 900
                reasons.append("variant-stem")
                break
            if stem.startswith(variant + "."):
                score += 850
                reasons.append("variant-arch-stem")
                break

    if any(token_occurs(stem, variant) for variant in variants):
        score += 250
        reasons.append("stem-token")
    elif any(variant and variant in stem for variant in variants):
        score += 10
        reasons.append("stem-substring")

    if primary and token_occurs(text, primary):
        score += 140
        reasons.append("primary-token")
    elif any(token_occurs(text, variant) for variant in variants[1:]):
        score += 100
        reasons.append("variant-token")
    elif any(variant and variant in text for variant in variants):
        score += 5
        reasons.append("text-substring")

    if Path(path).suffix == ".hip":
        score += 25
        reasons.append("hip-source")
    if rel.startswith("kernels/src/"):
        score += 15
        reasons.append("kernel-tree")
    if re.search(r"(^|/)(test|tests|benchmarks?)(/|_)", rel):
        score -= 200
        reasons.append("test-penalty")

    return score, reasons


def find_source_files(root, variants, limit=8, arch=None):
    root = Path(root)
    candidates = []
    search_roots = [root / "kernels" / "src"]
    if not search_roots[0].is_dir():
        search_roots = [root]
    suffixes = {".hip", ".cu", ".cpp", ".cc", ".c", ".h", ".hpp"}
    for path_name in candidate_file_paths(tuple(str(p) for p in search_roots), tuple(sorted(suffixes))):
        path = Path(path_name)
        score, reasons = source_file_score(path, root, variants, arch=arch)
        if score <= 0:
            continue
        candidates.append(
            {
                "path": relpath(path, root),
                "md5": md5_file(path),
                "match_score": score,
                "match_reason": reasons,
            }
        )
    candidates.sort(key=lambda item: (-item["match_score"], item["path"]))
    return candidates[:limit]


def dispatch_line_kind(line):
    stripped = line.strip()
    if not stripped:
        return "blank"
    if stripped.startswith(("///", "//", "/*", "*", "*/")):
        return "comment"
    return "code"


def dispatch_ref_score(path, root, line, variants):
    rel = relpath(path, root)
    primary = variants[0] if variants else ""
    kind = dispatch_line_kind(line)
    stripped = line.strip()
    score = 0
    reasons = []

    if kind == "code":
        score += 1000
        reasons.append("code")
    elif kind == "comment":
        score += 100
        reasons.append("comment")

    if primary and token_occurs(line, primary):
        score += 220
        reasons.append("primary-token")
    elif any(token_occurs(line, variant) for variant in variants[1:]):
        score += 160
        reasons.append("variant-token")
    elif any(variant and variant in line for variant in variants):
        score += 5
        reasons.append("substring")

    if re.search(r"\b(launch|dispatch|kernel|call)\b", stripped) or "gpu." in stripped:
        score += 60
        reasons.append("call-like")
    if "(" in stripped and ")" in stripped:
        score += 25
        reasons.append("call-syntax")
    if rel.endswith("dispatch.rs"):
        score += 40
        reasons.append("dispatch-file")
    elif rel.startswith("crates/hipfire-runtime/"):
        score += 30
        reasons.append("runtime")
    elif rel.startswith("crates/"):
        score += 20
        reasons.append("crate")
    elif rel.startswith("kernels/src/"):
        score += 10
        reasons.append("kernel-source")
    if "/src/" in rel:
        score += 45
        reasons.append("src-tree")
    if "/examples/" in rel:
        score -= 150
        reasons.append("example-penalty")
    if re.search(r"(^|/)(test|tests|benchmarks?)(/|_)", rel):
        score -= 200
        reasons.append("test-penalty")

    return score, kind, reasons


def find_dispatch_refs(root, variants, limit=8):
    root = Path(root)
    refs = []
    search_roots = [root / "crates", root / "cli", root / "kernels" / "src"]
    suffixes = {".rs", ".ts", ".js", ".hip", ".h", ".hpp", ".cpp"}
    for path_name in candidate_file_paths(tuple(str(p) for p in search_roots), tuple(sorted(suffixes))):
        path = Path(path_name)
        lines = read_text_lossy(path).splitlines()
        for line_no, line in enumerate(lines, 1):
            if not any(variant and variant in line for variant in variants):
                continue
            score, kind, reasons = dispatch_ref_score(path, root, line, variants)
            if score <= 0:
                continue
            refs.append(
                {
                    "path": relpath(path, root),
                    "line": line_no,
                    "text": line.strip()[:240],
                    "kind": kind,
                    "match_score": score,
                    "match_reason": reasons,
                }
            )
    refs.sort(key=lambda item: (-item["match_score"], item["path"], item["line"]))
    return refs[:limit]


def infer_env_controls(name, dispatch_refs=None):
    controls = set()
    lowered = (name or "").lower()
    if "multirow" in lowered or "hfq4g256_residual" in lowered:
        controls.add("HIPFIRE_GEMV_ROWS")
    if "attention" in lowered or "flash" in lowered:
        controls.add("HIPFIRE_ATTN_FLASH")
    if "kv" in lowered:
        controls.add("HIPFIRE_KV_MODE")
    for ref in dispatch_refs or []:
        for match in re.findall(r"HIPFIRE_[A-Z0-9_]+", ref.get("text", "")):
            controls.add(match)
    return sorted(controls)


def collect_dispatch_manifest(*, root=".", kernel_names=None, ref_limit=8, arch=None):
    root = Path(root)
    entries = []
    for name in kernel_names or []:
        variants = kernel_name_variants(name)
        dispatch_refs = find_dispatch_refs(root, variants, limit=ref_limit)
        entries.append(
            {
                "name": name,
                "arch": arch,
                "op": classify_kernel_op(name),
                "name_variants": variants,
                "source_files": find_source_files(root, variants, limit=ref_limit, arch=arch),
                "dispatch_refs": dispatch_refs,
                "env_controls": infer_env_controls(name, dispatch_refs),
            }
        )
    return {
        "schema": "hipfire.kernel_atlas.dispatch.v0",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "arch": arch,
        "entry_count": len(entries),
        "entries": entries,
    }


def maybe_build_isa_artifact(args, arch):
    files = getattr(args, "isa_file", None) or []
    dirs = getattr(args, "isa_dir", None) or []
    if not files and not dirs:
        return None
    manifest = collect_isa_manifest(
        arch=arch,
        files=files,
        dirs=dirs,
        name_filter=getattr(args, "isa_filter", None),
        limit=getattr(args, "isa_limit", None),
    )
    output = getattr(args, "isa_output", None)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "manifest_path": str(path),
            "manifest_md5": md5_file(path),
            "object_count": manifest["object_count"],
            "kernel_count": manifest["kernel_count"],
            "instruction_count": manifest["instruction_count"],
        }
    return {"manifest": manifest}


def profile_names_from_rows(rows):
    names = []
    for row in rows:
        for name in row_profile_names(row):
            if name not in names:
                names.append(name)
    return names


def maybe_build_dispatch_artifact(args, rows):
    if not getattr(args, "dispatch_provenance", False) and not getattr(
        args, "dispatch_output", None
    ):
        return None
    manifest = collect_dispatch_manifest(
        root=getattr(args, "dispatch_root", "."),
        kernel_names=profile_names_from_rows(rows),
        ref_limit=getattr(args, "dispatch_ref_limit", 8),
        arch=getattr(args, "arch", None) or (rows[0].get("arch") if rows else None),
    )
    output = getattr(args, "dispatch_output", None)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "manifest_path": str(path),
            "manifest_md5": md5_file(path),
            "entry_count": manifest["entry_count"],
        }
    return manifest


def attach_isa_artifact(rows, artifact):
    if not artifact:
        return
    for row in rows:
        row.setdefault("artifacts", {})["isa"] = artifact


def attach_dispatch_artifact(rows, artifact):
    if not artifact:
        return
    for row in rows:
        row.setdefault("artifacts", {})["dispatch"] = artifact


ARCH_CAPABILITIES = {
    "gfx1010": {
        "native_matrix": False,
        "wavefront": "32/64",
        "notes": "RDNA1: no WMMA path; fit is mostly VALU/VMEM/launch.",
    },
    "gfx1030": {
        "native_matrix": False,
        "wavefront": "32",
        "notes": "RDNA2: no WMMA path; packed-int/VALU fit matters.",
    },
    "gfx1100": {
        "native_matrix": True,
        "wavefront": "32",
        "notes": "RDNA3: wave32 WMMA available for prefill-family kernels.",
    },
    "gfx1101": {
        "native_matrix": True,
        "wavefront": "32",
        "notes": "RDNA3: wave32 WMMA available for prefill-family kernels.",
    },
    "gfx1102": {
        "native_matrix": True,
        "wavefront": "32",
        "notes": "RDNA3: wave32 WMMA available for prefill-family kernels.",
    },
    "gfx1151": {
        "native_matrix": True,
        "wavefront": "32",
        "notes": "RDNA3.5: wave32 WMMA available; bandwidth and launch count still dominate many decode paths.",
    },
    "gfx1200": {
        "native_matrix": True,
        "wavefront": "32",
        "notes": "RDNA4: wave32 WMMA available; arch-specific dispatch choices matter.",
    },
    "gfx1201": {
        "native_matrix": True,
        "wavefront": "32",
        "notes": "RDNA4: wave32 WMMA available; arch-specific dispatch choices matter.",
    },
}


QUANT_PROFILES = {
    "mq4": {
        "label": "MQ4",
        "bytes_saved": "medium",
        "unpack_tax": "low",
        "fit_question": "is bandwidth still the limit after simple unpack?",
    },
    "hfq4": {
        "label": "HFQ4",
        "bytes_saved": "medium",
        "unpack_tax": "low",
        "fit_question": "is bandwidth still the limit after simple unpack?",
    },
    "mq3": {
        "label": "MQ3",
        "bytes_saved": "high",
        "unpack_tax": "medium/high",
        "fit_question": "did saved bandwidth beat unpack/register pressure?",
    },
    "hfq3": {
        "label": "HFQ3",
        "bytes_saved": "high",
        "unpack_tax": "medium/high",
        "fit_question": "did saved bandwidth beat unpack/register pressure?",
    },
    "q8": {
        "label": "Q8",
        "bytes_saved": "low",
        "unpack_tax": "low",
        "fit_question": "is cache/KV bandwidth or launch overhead dominant?",
    },
    "paro4": {
        "label": "PARO4",
        "bytes_saved": "medium",
        "unpack_tax": "high",
        "fit_question": "is pair-rotation repeated across output packs instead of amortized?",
    },
}


def quant_profile(quant):
    key = (quant or "").lower()
    for prefix, profile in QUANT_PROFILES.items():
        if key.startswith(prefix):
            return profile
    return {
        "label": (quant or "unknown").upper(),
        "bytes_saved": "unknown",
        "unpack_tax": "unknown",
        "fit_question": "collect ISA plus profile rows before interpreting fit.",
    }


def arch_capability(arch):
    return ARCH_CAPABILITIES.get(
        arch or "",
        {
            "native_matrix": False,
            "wavefront": "unknown",
            "notes": "Unknown arch; Atlas can still render observed ISA but avoids capability claims.",
        },
    )


def fit_bar(fraction, width=16):
    fraction = max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def aggregate_manifest(manifest):
    objects = manifest.get("objects", []) if manifest else []
    categories = collections.Counter()
    opcodes = collections.Counter()
    kernels = []
    instruction_count = 0

    for obj in objects:
        summary = obj.get("instruction_summary", {})
        instruction_count += int(summary.get("instruction_count", 0) or 0)
        categories.update(summary.get("category_counts", {}) or {})
        opcodes.update(summary.get("opcode_counts", {}) or {})
        kernels.extend(obj.get("kernels", []) or [])

    return {
        "instruction_count": instruction_count,
        "category_counts": dict(categories),
        "opcode_counts": dict(opcodes),
        "kernels": kernels,
    }


def max_kernel_field(kernels, key, default=0):
    values = [k.get(key) for k in kernels if isinstance(k.get(key), int)]
    return max(values) if values else default


def first_kernel_field(kernels, key, default="unknown"):
    for kernel in kernels:
        value = kernel.get(key)
        if value is not None:
            return value
    return default


def observed_category_counts(categories):
    return {
        "matrix/wmma": int(categories.get("matrix", 0) or 0),
        "valu": int(categories.get("valu", 0) or 0) + int(categories.get("dot", 0) or 0),
        "vmem": int(categories.get("vmem", 0) or 0),
        "lds": int(categories.get("lds", 0) or 0),
        "salu/control": (
            int(categories.get("salu", 0) or 0)
            + int(categories.get("branch", 0) or 0)
            + int(categories.get("wait", 0) or 0)
            + int(categories.get("smem", 0) or 0)
        ),
    }


def object_kernel_names(obj):
    """Return every kernel name/symbol string that can identify an ISA object."""
    names = set()
    for kernel in obj.get("kernels", []) or []:
        name = kernel.get("name")
        symbol = kernel.get("symbol")
        if name:
            names.add(name)
        if symbol:
            names.add(symbol)
            names.add(str(symbol).split(".kd", 1)[0])
    for symbol in obj.get("instruction_summary", {}).get("kernel_symbols", []) or []:
        if symbol:
            names.add(symbol)
            names.add(str(symbol).split(".kd", 1)[0])
    return names


def row_profile_names(row):
    names = []
    for item in row.get("artifacts", {}).get("profile_kernels", []) or []:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = str(item)
        if name and name not in names:
            names.append(name)
    return names


def join_profile_to_isa(row, manifest):
    """Scope an ISA manifest to objects matching the row's profiled kernels."""
    objects = list((manifest or {}).get("objects", []) or [])
    profile_names = row_profile_names(row)
    if not profile_names:
        return {
            "scope": "all-inspected",
            "objects": objects,
            "profile_names": [],
            "matched_profile_names": [],
            "unmatched_profile_names": [],
            "matched_object_count": len(objects),
            "inspected_object_count": len(objects),
        }

    profile_set = set(profile_names)
    matched_objects = []
    matched_names = set()
    for obj in objects:
        hits = profile_set.intersection(object_kernel_names(obj))
        if not hits:
            continue
        scoped = copy.deepcopy(obj)
        scoped["profile_matches"] = sorted(hits)
        matched_objects.append(scoped)
        matched_names.update(hits)

    matched_profile_names = [name for name in profile_names if name in matched_names]
    unmatched_profile_names = [name for name in profile_names if name not in matched_names]
    if matched_objects:
        scope = "profile-matched"
        scoped_objects = matched_objects
    else:
        scope = "profile-unmatched"
        scoped_objects = objects

    return {
        "scope": scope,
        "objects": scoped_objects,
        "profile_names": profile_names,
        "matched_profile_names": matched_profile_names,
        "unmatched_profile_names": unmatched_profile_names,
        "matched_object_count": len(matched_objects),
        "inspected_object_count": len(objects),
    }


def dispatch_entries_by_name(dispatch_manifest):
    entries = {}
    for entry in (dispatch_manifest or {}).get("entries", []) or []:
        name = entry.get("name")
        if name:
            entries[name] = entry
    return entries


def profile_kernel_op(item):
    if isinstance(item, dict):
        return item.get("op") or classify_kernel_op(item.get("name"))
    return classify_kernel_op(str(item))


def render_hot_kernel_lines(row, dispatch_manifest, limit=6):
    kernels = row.get("artifacts", {}).get("profile_kernels", []) or []
    if not kernels:
        return []
    dispatch_by_name = dispatch_entries_by_name(dispatch_manifest)
    lines = ["", "HOT KERNELS"]
    for item in kernels[:limit]:
        name = item.get("name") if isinstance(item, dict) else str(item)
        op = profile_kernel_op(item)
        entry = dispatch_by_name.get(name, {})
        source = "unknown"
        if entry.get("source_files"):
            source = entry["source_files"][0].get("path", "unknown")
        dispatch = "unknown"
        if entry.get("dispatch_refs"):
            ref = entry["dispatch_refs"][0]
            dispatch = f"{ref.get('path', 'unknown')}:{ref.get('line', '?')}"
        env = ""
        if entry.get("env_controls"):
            env = "  env " + ",".join(entry["env_controls"])
        lines.append(
            f"{name}  {op['family']}.{op['role']}  src {source}  dispatch {dispatch}{env}"
        )
    return lines


def fit_interpretation(row, manifest_summary, arch_info, quant_info):
    categories = manifest_summary["category_counts"]
    observed = observed_category_counts(categories)
    total = max(1, manifest_summary["instruction_count"])
    matrix = observed["matrix/wmma"]
    vmem_frac = observed["vmem"] / total
    valu_frac = observed["valu"] / total
    phase = row.get("phase", "")
    hot = row.get("artifacts", {}).get("profile_kernels", []) or []
    paro_gemv_pct = sum(
        profile_pct(item)
        for item in hot
        if "gemv_paro4g128" in ((item.get("name") if isinstance(item, dict) else str(item)) or "")
    )

    if paro_gemv_pct >= 35.0 and phase in ("decode_ar", "decode_dflash"):
        likely = "Paro rotated-GEMV dominates"
        left = "amortize pair rotation / prerotate once per projection / improve packed-output tiling"
    elif arch_info["native_matrix"] and matrix == 0 and phase in ("decode_ar", "decode_dflash"):
        likely = "memory/launch, not matrix throughput"
        left = "fusion / launch reduction / lower bytes moved"
    elif arch_info["native_matrix"] and matrix == 0 and phase == "prefill":
        likely = "prefill is missing matrix utilization"
        left = "WMMA prefill path / tile remap / dispatch routing"
    elif vmem_frac >= 0.18 and valu_frac >= 0.35 and quant_info["unpack_tax"] != "low":
        likely = "bandwidth plus unpack/register pressure"
        left = "reduce unpack tax or register footprint"
    elif vmem_frac >= 0.18:
        likely = "memory bandwidth"
        left = "lower bytes moved / cache reuse / fusion"
    elif valu_frac >= 0.50:
        likely = "VALU/unpack throughput"
        left = "native packed op mapping / instruction reduction"
    else:
        likely = "mixed or launch dominated"
        left = "profile with launch counts and counters"

    return likely, left


def render_fit_view(row, manifest):
    arch = row.get("arch", "unknown")
    workload = row.get("workload", "unknown")
    quant = row.get("quant", "unknown")
    phase = row.get("phase", "unknown")
    shape = row.get("shape_bucket", "unknown")
    metrics = row.get("metrics", {})
    arch_info = arch_capability(arch)
    quant_info = quant_profile(quant)
    dispatch_manifest = row.get("artifacts", {}).get("dispatch", {})
    joined = join_profile_to_isa(row, manifest or {})
    summary = aggregate_manifest({"objects": joined["objects"]})
    categories = summary["category_counts"]
    observed = observed_category_counts(categories)
    total = max(1, summary["instruction_count"])
    kernels = summary["kernels"]
    likely, left = fit_interpretation(row, summary, arch_info, quant_info)

    quant_label = quant_info["label"]
    native_matrix = arch_info["native_matrix"]
    available = {
        "matrix/wmma": 1.0 if native_matrix else 0.0,
        "valu": 1.0,
        "vmem": 1.0,
        "lds": 1.0,
        "salu/control": 1.0,
    }

    lines = [
        "ISA FIT VIEW",
        f"{arch} | {workload} | {quant_label} | {phase} | {shape}",
        "",
        "ISA SCOPE",
    ]
    if joined["scope"] == "profile-matched":
        lines.append(
            f"profile matched   {len(joined['matched_profile_names'])}/{len(joined['profile_names'])} names, {joined['matched_object_count']} objects"
        )
        if joined["unmatched_profile_names"]:
            unmatched = ", ".join(joined["unmatched_profile_names"][:6])
            lines.append(f"unmatched hot     {unmatched}")
    elif joined["scope"] == "profile-unmatched":
        lines.append(
            f"profile matched   0/{len(joined['profile_names'])} names, using all {joined['inspected_object_count']} objects"
        )
        unmatched = ", ".join(joined["unmatched_profile_names"][:6])
        lines.append(f"unmatched hot     {unmatched}")
    else:
        lines.append(
            f"profile matched   no profile names, using all {joined['inspected_object_count']} objects"
        )

    lines.extend(render_hot_kernel_lines(row, dispatch_manifest))
    lines.extend(
        [
            "",
            "ARCH LANES        AVAILABLE           OBSERVED ISA",
        ]
    )
    for label in ("matrix/wmma", "valu", "vmem", "lds", "salu/control"):
        avail_bar = fit_bar(available[label])
        obs_bar = fit_bar(observed[label] / total)
        suffix = f"{observed[label]:5d}"
        if label == "matrix/wmma" and not native_matrix:
            suffix = "  unavailable"
        lines.append(f"{label:<17} {avail_bar}  {obs_bar} {suffix}")

    lines.extend(
        [
            "",
            "RESOURCE SHAPE",
            f"vgpr              {fit_bar(max_kernel_field(kernels, 'vgpr_count') / 256.0)} {max_kernel_field(kernels, 'vgpr_count'):5d}",
            f"sgpr              {fit_bar(max_kernel_field(kernels, 'sgpr_count') / 128.0)} {max_kernel_field(kernels, 'sgpr_count'):5d}",
            f"spills            {fit_bar(min(1.0, (max_kernel_field(kernels, 'vgpr_spill_count') + max_kernel_field(kernels, 'sgpr_spill_count')) / 16.0))} {max_kernel_field(kernels, 'vgpr_spill_count') + max_kernel_field(kernels, 'sgpr_spill_count'):5d}",
            f"wavefront                         {first_kernel_field(kernels, 'wavefront_size', arch_info['wavefront'])}",
            f"lds bytes                      {max_kernel_field(kernels, 'group_segment_fixed_size'):5d}",
            "",
            "QUANT INTENT",
            f"bytes saved       {quant_info['bytes_saved']}",
            f"unpack tax        {quant_info['unpack_tax']}",
            f"native matrix     {'yes' if native_matrix else 'no'}",
            f"fit question      {quant_info['fit_question']}",
            "",
            "RUNTIME",
        ]
    )
    for key in ("prefill_tok_s", "gen_tok_s", "decode_tok_s", "tau", "bw_gib_s"):
        if key in metrics:
            lines.append(f"{key:<17} {metrics[key]}")

    lines.extend(
        [
            "",
            "FIT READ",
            f"likely limit      {likely}",
            f"left on table     {left}",
            f"arch note         {arch_info['notes']}",
        ]
    )
    return "\n".join(lines)


def safe_id(text):
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(text or "").lower()).strip("-")
    return text or "unknown"


def row_eval_contract(row):
    row_env = copy.deepcopy(row.get("variant", {}).get("env", {}))
    eval_env, stripped = tuning_eval_env(row_env)
    return {
        "metric": preferred_metric_for_row(row),
        "goal": "maximize",
        "benchmark_command": copy.deepcopy(row.get("command", [])),
        "env": eval_env,
        "stripped_env_keys": stripped,
        "requires_fresh_baseline": bool(stripped),
    }


def source_path_for_entry(entry):
    source_files = entry.get("source_files", []) or []
    if source_files:
        return source_files[0].get("path")
    return None


def dispatch_paths_for_entry(entry, limit=2):
    paths = []
    for ref in entry.get("dispatch_refs", []) or []:
        path = ref.get("path")
        if path and path not in paths:
            paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def profile_pct(item):
    if isinstance(item, dict) and isinstance(item.get("pct"), (int, float)):
        return float(item["pct"])
    return 0.0


def suggestion_record(
    *,
    title,
    lever_type,
    hot_kernel=None,
    op=None,
    score=0,
    risk="medium",
    expected_impact="medium",
    allowed_files=None,
    rationale=None,
    candidate_steps=None,
    eval_contract=None,
    correctness_commands=None,
):
    payload = {
        "title": title,
        "lever_type": lever_type,
        "hot_kernel": hot_kernel or "global",
        "allowed_files": sorted(dict.fromkeys(allowed_files or [])),
        "candidate_steps": list(candidate_steps or []),
    }
    return {
        "id": "sug-" + md5_text(json.dumps(payload, sort_keys=True))[:12],
        "rank": None,
        "score": round(float(score), 3),
        "title": title,
        "lever_type": lever_type,
        "hot_kernel": hot_kernel,
        "op": op or {"family": "unknown", "role": "unknown", "phase_hint": "unknown"},
        "risk": risk,
        "expected_impact": expected_impact,
        "allowed_files": payload["allowed_files"],
        "rationale": list(rationale or []),
        "candidate_steps": payload["candidate_steps"],
        "eval": copy.deepcopy(eval_contract or {}),
        "correctness_commands": list(correctness_commands or []),
    }


def add_suggestion(suggestions, seen, item):
    key = (
        item.get("lever_type"),
        item.get("hot_kernel"),
        item.get("title"),
        tuple(item.get("allowed_files", [])),
    )
    if key in seen:
        return
    seen.add(key)
    suggestions.append(item)


def read_jsonl_objects(path):
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def history_candidate_paths(paths=None, root=None):
    root_path = Path(root or ".")
    candidates = [root_path / DEFAULT_HISTORY_DIR]
    for item in paths or []:
        p = Path(item)
        if not p.is_absolute():
            p = root_path / p
        candidates.append(p)
    deduped = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def scan_history_files(paths=None, root=None):
    result_files = []
    ledger_files = []
    task_files = []
    for path in history_candidate_paths(paths, root=root):
        if path.is_file():
            if path.name == "ledger.jsonl":
                ledger_files.append(path)
            elif path.name == "result.json":
                result_files.append(path)
            elif path.name.startswith("task") and path.suffix == ".json":
                task_files.append(path)
            continue
        if not path.is_dir():
            continue
        result_files.extend(sorted(path.rglob("result.json")))
        ledger_files.extend(sorted(path.rglob("ledger.jsonl")))
        task_files.extend(sorted(path.rglob("task*.json")))
    return result_files, ledger_files, task_files


def load_task_history_index(task_files):
    index = {}
    for path in task_files:
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = task.get("task_id")
        if task_id:
            index.setdefault(task_id, task)
    return index


def normalize_history_entry(row, task_index=None, source=None):
    task_index = task_index or {}
    task_id = row.get("task_id")
    task = task_index.get(task_id, {})
    task_producer = task.get("producer", {}) if isinstance(task.get("producer"), dict) else {}
    suggestion = row.get("suggestion", {}) if isinstance(row.get("suggestion"), dict) else {}
    producer_suggestion = (
        task_producer.get("suggestion", {}) if isinstance(task_producer.get("suggestion"), dict) else {}
    )
    hot_kernel = None
    if isinstance(row.get("hot_kernel"), dict):
        hot_kernel = row["hot_kernel"].get("name")
    elif isinstance(row.get("hot_kernel"), str):
        hot_kernel = row.get("hot_kernel")
    if hot_kernel is None and isinstance(task.get("hot_kernel"), dict):
        hot_kernel = task["hot_kernel"].get("name")
    target = row.get("target") if isinstance(row.get("target"), dict) else None
    if target is None:
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
    baseline = task.get("baseline", {}) if isinstance(task.get("baseline"), dict) else {}
    env = baseline.get("env", {}) if isinstance(baseline.get("env"), dict) else {}
    constraints = task.get("constraints", {}) if isinstance(task.get("constraints"), dict) else {}
    speedup = row.get("speedup")
    if speedup is None and isinstance(row.get("delta"), dict):
        speedup = row["delta"].get("speedup")
    return {
        "task_id": task_id,
        "captured_at_utc": row.get("captured_at_utc"),
        "source": source,
        "status": row.get("status"),
        "metric": row.get("metric"),
        "speedup": speedup,
        "stability": copy.deepcopy(row.get("stability", {})),
        "target": copy.deepcopy(target or {}),
        "hot_kernel": hot_kernel,
        "allowed_files": copy.deepcopy(constraints.get("allowed_files", []) or []),
        "env": copy.deepcopy(env),
        "suggestion_id": suggestion.get("id") or producer_suggestion.get("id"),
        "history_key": suggestion.get("history_key") or producer_suggestion.get("history_key"),
        "lever_type": suggestion.get("lever_type") or producer_suggestion.get("lever_type"),
        "title": suggestion.get("title") or producer_suggestion.get("title"),
    }


def load_suggestion_history(paths=None, root=None):
    result_files, ledger_files, task_files = scan_history_files(paths, root=root)
    task_index = load_task_history_index(task_files)
    entries = []
    seen = set()
    for path in result_files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entry = normalize_history_entry(row, task_index=task_index, source=str(path))
        key = (entry.get("task_id"), entry.get("source"))
        if key not in seen:
            seen.add(key)
            entries.append(entry)
    for path in ledger_files:
        for row in read_jsonl_objects(path):
            entry = normalize_history_entry(row, task_index=task_index, source=str(path))
            key = (entry.get("task_id"), entry.get("captured_at_utc"), entry.get("source"))
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return entries


def history_matches_target(row, entry):
    target = entry.get("target") or {}
    for key in ("arch", "workload", "phase", "shape_bucket", "quant"):
        expected = row.get(key)
        actual = target.get(key)
        if expected and actual and expected != actual:
            return False
    return True


def suggestion_history_key(item):
    payload = {
        "title": item.get("title"),
        "lever_type": item.get("lever_type"),
        "hot_kernel": item.get("hot_kernel"),
        "allowed_files": item.get("allowed_files", []),
        "candidate_steps": item.get("candidate_steps", []),
    }
    return "hist-" + md5_text(json.dumps(payload, sort_keys=True))[:12]


def entry_matches_suggestion(entry, item):
    if entry.get("suggestion_id") and entry.get("suggestion_id") == item.get("id"):
        return True
    if entry.get("history_key") and entry.get("history_key") == item.get("history_key"):
        return True
    hot_kernel = item.get("hot_kernel")
    if not hot_kernel or entry.get("hot_kernel") != hot_kernel:
        return False
    if entry.get("lever_type") and entry.get("lever_type") == item.get("lever_type"):
        return True
    title = " ".join(
        [
            str(item.get("title", "")),
            " ".join(str(step) for step in item.get("candidate_steps", []) or []),
        ]
    ).lower()
    task_id = str(entry.get("task_id") or "").lower()
    env = entry.get("env") if isinstance(entry.get("env"), dict) else {}
    if "multirow" in title and ("multirow" in task_id or "HIPFIRE_GEMV_ROWS" in env):
        return True
    if item.get("lever_type") == "env_sweep" and any(str(key).startswith("HIPFIRE_") for key in env):
        return True
    return False


def history_rejects_entry(entry):
    if entry.get("status") not in ("pass", "unstable"):
        return False
    speedup = entry.get("speedup")
    return isinstance(speedup, (int, float)) and speedup < HISTORY_REJECT_SPEEDUP


def apply_history_to_suggestions(row, suggestions, history_entries):
    target_history = [entry for entry in history_entries or [] if history_matches_target(row, entry)]
    matched_count = 0
    demoted_count = 0
    for item in suggestions:
        item["history_key"] = suggestion_history_key(item)
        matches = [entry for entry in target_history if entry_matches_suggestion(entry, item)]
        if not matches:
            continue
        matched_count += len(matches)
        speedups = [entry.get("speedup") for entry in matches if isinstance(entry.get("speedup"), (int, float))]
        rejected = [entry for entry in matches if history_rejects_entry(entry)]
        best_speedup = max(speedups) if speedups else None
        item["history"] = {
            "status": "rejected" if rejected and (best_speedup is None or best_speedup < HISTORY_REJECT_SPEEDUP) else "seen",
            "match_count": len(matches),
            "best_speedup": best_speedup,
            "task_ids": sorted(dict.fromkeys(entry.get("task_id") for entry in matches if entry.get("task_id"))),
        }
        if item["history"]["status"] == "rejected":
            item["score_before_history"] = item["score"]
            item["score"] = round(max(0.0, float(item["score"]) - HISTORY_REJECT_DEMOTE), 3)
            demoted_count += 1
            if best_speedup is not None:
                item.setdefault("rationale", []).append(
                    f"Atlas history rejected this lever on the same target; best prior speedup was {best_speedup:.3f}x."
                )
    return {
        "entries": len(history_entries or []),
        "target_entries": len(target_history),
        "matched_entries": matched_count,
        "demoted_suggestions": demoted_count,
    }


def build_suggestion_queue(
    row,
    manifest=None,
    dispatch_manifest=None,
    *,
    max_suggestions=12,
    hot_limit=8,
    correctness_commands=None,
    history=None,
):
    dispatch_manifest = dispatch_manifest or row.get("artifacts", {}).get("dispatch", {}) or {}
    dispatch_by_name = dispatch_entries_by_name(dispatch_manifest)
    eval_contract = row_eval_contract(row)
    arch = row.get("arch", "unknown")
    phase = row.get("phase", "unknown")
    quant = row.get("quant", "unknown")
    workload = row.get("workload", "unknown")
    shape = row.get("shape_bucket", "unknown")
    hot = row.get("artifacts", {}).get("profile_kernels", []) or []
    joined = join_profile_to_isa(row, manifest or {})
    summary = aggregate_manifest({"objects": joined.get("objects", [])})
    arch_info = arch_capability(arch)
    quant_info = quant_profile(quant)
    likely, left = fit_interpretation(row, summary, arch_info, quant_info)
    observed = observed_category_counts(summary.get("category_counts", {}))

    suggestions = []
    seen = set()

    unmatched = joined.get("unmatched_profile_names", []) or []
    paro_gemv_pct = sum(
        profile_pct(item)
        for item in hot[:hot_limit]
        if "gemv_paro4g128" in ((item.get("name") if isinstance(item, dict) else str(item)) or "")
    )
    if paro_gemv_pct >= 35.0:
        paro_files = []
        for item in hot[:hot_limit]:
            name = item.get("name") if isinstance(item, dict) else str(item)
            if "gemv_paro4g128" not in name:
                continue
            entry = dispatch_by_name.get(name, {})
            src = source_path_for_entry(entry)
            if src:
                paro_files.append(src)
            paro_files.extend(dispatch_paths_for_entry(entry, limit=1))
        add_suggestion(
            suggestions,
            seen,
            suggestion_record(
                title="Split Paro activation rotation from packed GEMV",
                lever_type="format_kernel",
                score=96 + min(8, paro_gemv_pct / 16.0),
                risk="medium/high",
                expected_impact="high",
                allowed_files=paro_files or ["kernels/src/gemv_paro4g128.hip", "crates/rdna-compute/src/dispatch.rs"],
                rationale=[
                    f"Paro GEMV kernels account for {paro_gemv_pct:.1f}% of the captured profile.",
                    "The current native kernel rotates each G128 activation group inside every packed-output block.",
                    "A prerotate-once plus packed-GEMV path can amortize pairs/theta/channel-scale work across all output rows.",
                ],
                candidate_steps=[
                    "Add a Paro G128 activation-rotation kernel that writes an x_rot scratch once per projection.",
                    "Add a prerotated packed W4 GEMV/residual kernel that consumes x_rot without pair rotation.",
                    "Gate the route, then compare against a clean Atlas baseline before making it default.",
                ],
                eval_contract=eval_contract,
                correctness_commands=correctness_commands,
            ),
        )
    if unmatched:
        add_suggestion(
            suggestions,
            seen,
            suggestion_record(
                title="Expand ISA capture for unmatched hot kernels",
                lever_type="measurement",
                score=83,
                risk="low",
                expected_impact="indirect",
                allowed_files=[],
                rationale=[
                    f"{len(unmatched)} profiled kernel(s) are missing from the ISA manifest.",
                    "A tuning queue is safer when every hot source has matching VGPR/SGPR/opcode evidence.",
                ],
                candidate_steps=[
                    "Regenerate the ISA manifest with a wider --isa-filter or higher --isa-limit.",
                    "Re-run render-fit before choosing source-level mutations for unmatched kernels.",
                ],
                eval_contract=eval_contract,
                correctness_commands=correctness_commands,
            ),
        )

    if (
        arch_info.get("native_matrix")
        and phase in ("decode_ar", "decode_dflash")
        and observed["matrix/wmma"] == 0
        and paro_gemv_pct < 35.0
    ):
        low_byte_sources = []
        for item in hot[:hot_limit]:
            name = item.get("name") if isinstance(item, dict) else str(item)
            role = profile_kernel_op(item).get("role", "")
            if any(token in name for token in ("rmsnorm", "rotate", "qk_l2_norm", "gated_norm")) or role in (
                "rope_rotate",
                "normalization",
                "qk_norm_scale",
            ):
                entry = dispatch_by_name.get(name, {})
                src = source_path_for_entry(entry)
                if src:
                    low_byte_sources.append(src)
                low_byte_sources.extend(dispatch_paths_for_entry(entry, limit=1))
        add_suggestion(
            suggestions,
            seen,
            suggestion_record(
                title="Prioritize decode launch/fusion experiments over matrix-path changes",
                lever_type="fusion",
                score=92,
                risk="medium",
                expected_impact="high",
                allowed_files=low_byte_sources,
                rationale=[
                    f"Fit read says {likely}; left on table: {left}.",
                    "Native matrix units are available, but the observed profiled ISA scope has zero matrix instructions.",
                    "This points at launch count, fusion, and bytes moved before another pure GEMV micro-tweak.",
                ],
                candidate_steps=[
                    "Pick one adjacent low-byte transform chain from the hot profile.",
                    "Create one fused candidate that removes a launch or one full-memory round trip.",
                    "Evaluate against a clean Atlas baseline with the same prompt/shape.",
                ],
                eval_contract=eval_contract,
                correctness_commands=correctness_commands,
            ),
        )

    for item in hot[:hot_limit]:
        name = item.get("name") if isinstance(item, dict) else str(item)
        op = profile_kernel_op(item)
        entry = dispatch_by_name.get(name, {})
        src = source_path_for_entry(entry)
        dispatch_paths = dispatch_paths_for_entry(entry)
        pct = profile_pct(item)
        op_role = op.get("role", "unknown")
        allowed = [path for path in [src] if path]

        if src:
            add_suggestion(
                suggestions,
                seen,
                suggestion_record(
                    title=f"Sweep source-level variants for {name}",
                    lever_type="source_sweep",
                    hot_kernel=name,
                    op=op,
                    score=70 + min(20, pct / 2.0),
                    risk="medium",
                    expected_impact="medium",
                    allowed_files=allowed,
                    rationale=[
                        f"{name} accounts for {pct:.1f}% of the captured profile." if pct else f"{name} is in the hot profile.",
                        "Use Atlas eval to test one controlled variant at a time.",
                    ],
                    candidate_steps=[
                        "Try exactly one source-level knob per candidate: launch_bounds, unroll factor, load grouping, prefetch distance, or accumulator live range.",
                        "Rebuild the benchmark binary so embedded HIP source changes take effect.",
                        "Reject stable candidates that do not beat the clean baseline.",
                    ],
                    eval_contract=eval_contract,
                    correctness_commands=correctness_commands,
                ),
            )

        if entry.get("env_controls"):
            add_suggestion(
                suggestions,
                seen,
                suggestion_record(
                    title=f"Sweep dispatch/env controls for {name}",
                    lever_type="env_sweep",
                    hot_kernel=name,
                    op=op,
                    score=62 + min(14, pct / 4.0),
                    risk="low",
                    expected_impact="low/medium",
                    allowed_files=dispatch_paths,
                    rationale=[
                        f"Dispatch provenance exposes {', '.join(entry.get('env_controls', []))}.",
                        "An env sweep can find an existing fast path before mutating kernel source.",
                    ],
                    candidate_steps=[
                        "Run one clean Atlas eval per env setting.",
                        "Keep the fastest stable setting only if it holds against the same baseline.",
                    ],
                    eval_contract=eval_contract,
                    correctness_commands=correctness_commands,
                ),
            )

        if op_role == "residual_gemv" and arch.startswith("gfx12"):
            files = allowed + [
                "crates/rdna-compute/src/kernels.rs",
                "crates/rdna-compute/src/dispatch.rs",
            ]
            add_suggestion(
                suggestions,
                seen,
                suggestion_record(
                    title=f"Port/test gfx12 residual multirow path for {name}",
                    lever_type="dispatch",
                    hot_kernel=name,
                    op=op,
                    score=88 + min(8, pct / 8.0),
                    risk="high",
                    expected_impact="medium/high",
                    allowed_files=files,
                    rationale=[
                        f"{name} is the top residual GEMV hot path on {arch}.",
                        "The existing residual multirow branch is not obviously enabled for gfx12, so this can change launch geometry instead of only compiler scheduling.",
                    ],
                    candidate_steps=[
                        "Add or route a gfx12-specific residual candidate behind an explicit dispatch knob.",
                        "Compare rows=1/2/4/8-style variants with clean Atlas eval.",
                        "Run the relevant correctness gate before keeping any candidate.",
                    ],
                    eval_contract=eval_contract,
                    correctness_commands=correctness_commands,
                ),
            )

        if op_role in ("qkvza_projection", "qkv_projection"):
            add_suggestion(
                suggestions,
                seen,
                suggestion_record(
                    title=f"Compare projection fusion/routing variants for {name}",
                    lever_type="dispatch",
                    hot_kernel=name,
                    op=op,
                    score=82 + min(8, pct / 8.0),
                    risk="medium",
                    expected_impact="medium",
                    allowed_files=allowed + dispatch_paths,
                    rationale=[
                        f"{name} is a hot fused attention projection.",
                        "Projection kernels can expose bytes-moved and launch-count wins when routing changes avoid fallback or duplicated reads.",
                    ],
                    candidate_steps=[
                        "Check whether the runtime path has an arch-specific gfx12 source or fallback.",
                        "Test one routing/source variant under Atlas eval before broader fusion work.",
                    ],
                    eval_contract=eval_contract,
                    correctness_commands=correctness_commands,
                ),
            )

        if op_role in ("rope_rotate", "normalization", "qk_norm_scale") or any(
            token in name for token in ("mq_rotate", "gated_norm", "qk_l2_norm")
        ):
            add_suggestion(
                suggestions,
                seen,
                suggestion_record(
                    title=f"Fuse or batch low-byte transform around {name}",
                    lever_type="fusion",
                    hot_kernel=name,
                    op=op,
                    score=86 + min(8, pct / 8.0),
                    risk="medium",
                    expected_impact="medium/high",
                    allowed_files=allowed + dispatch_paths,
                    rationale=[
                        f"{name} is a transform/norm hot kernel rather than a large matrix kernel.",
                        "These kernels are good launch-reduction candidates because they often move little data per launch.",
                    ],
                    candidate_steps=[
                        "Find the immediate producer/consumer in the dispatch path.",
                        "Fuse one adjacent transform or batch repeated per-layer calls.",
                        "Evaluate tok/s and inspect output if the row is speculative decode.",
                    ],
                    eval_contract=eval_contract,
                    correctness_commands=correctness_commands,
                ),
            )

    history_summary = apply_history_to_suggestions(row, suggestions, history or [])
    suggestions.sort(key=lambda item: (-item["score"], item["lever_type"], item["title"]))
    suggestions = suggestions[: max(0, int(max_suggestions))]
    for idx, item in enumerate(suggestions, 1):
        item["rank"] = idx

    return {
        "schema": "hipfire.kernel_atlas.suggestions.v0",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": {
            "host": row.get("hostname"),
            "arch": arch,
            "workload": workload,
            "phase": phase,
            "shape_bucket": shape,
            "quant": quant,
        },
        "metric": eval_contract["metric"],
        "fit_read": {
            "likely_limit": likely,
            "left_on_table": left,
            "quant_question": quant_info.get("fit_question"),
        },
        "history": history_summary,
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
    }


def render_suggestion_markdown(queue):
    target = queue.get("target", {})
    lines = [
        "# Atlas Suggestions",
        "",
        f"Target: {target.get('arch', 'unknown')} {target.get('workload', 'unknown')} {target.get('phase', 'unknown')} {target.get('shape_bucket', '')}".rstrip(),
        f"Metric: {queue.get('metric', 'unknown')}",
        "",
    ]
    fit = queue.get("fit_read", {})
    if fit:
        lines.extend(
            [
                "## Fit Read",
                f"- likely limit: {fit.get('likely_limit', 'unknown')}",
                f"- left on table: {fit.get('left_on_table', 'unknown')}",
                "",
            ]
        )
    lines.append("## Queue")
    for item in queue.get("suggestions", []) or []:
        lines.extend(
            [
                f"{item.get('rank')}. {item.get('title')}",
                f"- type: {item.get('lever_type')}  risk: {item.get('risk')}  expected: {item.get('expected_impact')}",
            ]
        )
        hot = item.get("hot_kernel")
        if hot:
            lines.append(f"- hot kernel: {hot}")
        files = item.get("allowed_files", []) or []
        if files:
            lines.append("- files: " + ", ".join(files))
        hist = item.get("history", {})
        if hist:
            detail = f"- history: {hist.get('status', 'seen')} over {hist.get('match_count', 0)} prior eval(s)"
            if hist.get("best_speedup") is not None:
                detail += f", best speedup {hist['best_speedup']:.3f}x"
            lines.append(detail)
        rationale = item.get("rationale", []) or []
        if rationale:
            lines.append("- rationale: " + " ".join(rationale))
        steps = item.get("candidate_steps", []) or []
        if steps:
            lines.append("- first steps: " + " ".join(steps[:2]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def preferred_metric_for_row(row):
    phase = row.get("phase", "")
    metrics = row.get("metrics", {})
    for key in ("gen_tok_s", "decode_tok_s", "prefill_tok_s", "tau"):
        if key in metrics:
            return key
    if phase == "prefill":
        return "prefill_tok_s"
    if phase == "decode_dflash":
        return "decode_tok_s"
    return "gen_tok_s"


def first_profile_kernel(row):
    kernels = row.get("artifacts", {}).get("profile_kernels", []) or []
    if kernels:
        return kernels[0]
    return None


def matched_dispatch_entry(row, kernel_name):
    for entry in row.get("artifacts", {}).get("dispatch", {}).get("entries", []) or []:
        if entry.get("name") == kernel_name:
            return entry
    return {}


def task_id_from_payload(prefix, payload):
    text = json.dumps(payload, sort_keys=True, default=str)
    return f"{prefix}-{md5_text(text)[:12]}"


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items or []:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def tuning_eval_env(row_env):
    row_env = dict(row_env or {})
    stripped = sorted(key for key in row_env if key in TUNING_STRIP_ENV_KEYS)
    env = {key: value for key, value in row_env.items() if key not in TUNING_STRIP_ENV_KEYS}
    return env, stripped


def build_task_bundle_from_row(
    row,
    manifest=None,
    *,
    allowed_files=None,
    correctness_commands=None,
    task_id=None,
    suggestion=None,
):
    kernel = first_profile_kernel(row) or {"name": "unknown", "op": classify_kernel_op("unknown")}
    suggestion = copy.deepcopy(suggestion) if isinstance(suggestion, dict) else None
    if suggestion and suggestion.get("hot_kernel"):
        suggested_kernel = suggestion["hot_kernel"]
        for item in row.get("artifacts", {}).get("profile_kernels", []) or []:
            name = item.get("name") if isinstance(item, dict) else str(item)
            if name == suggested_kernel:
                kernel = item
                break
        else:
            kernel = {
                "name": suggested_kernel,
                "op": suggestion.get("op") or classify_kernel_op(suggested_kernel),
            }
    kernel_name = kernel.get("name", "unknown") if isinstance(kernel, dict) else str(kernel)
    dispatch_entry = matched_dispatch_entry(row, kernel_name)
    joined = join_profile_to_isa(row, manifest or {})
    isa_objects = [
        {
            "path": obj.get("path"),
            "md5": obj.get("md5"),
            "kernels": obj.get("kernels", []),
            "instruction_summary": obj.get("instruction_summary", {}),
        }
        for obj in joined.get("objects", [])
    ]
    source_files = copy.deepcopy(dispatch_entry.get("source_files", []) or [])
    if allowed_files is None:
        if suggestion is not None:
            allowed_files = suggestion.get("allowed_files", []) or []
        elif source_files and source_files[0].get("path"):
            allowed_files = [source_files[0]["path"]]
        else:
            allowed_files = []
    allowed_files = dedupe_preserve_order(allowed_files)
    suggestion_correctness = suggestion.get("correctness_commands", []) if suggestion else []
    correctness_commands = list(correctness_commands or [])
    if not correctness_commands and suggestion_correctness:
        correctness_commands = copy.deepcopy(suggestion_correctness)
    metric = preferred_metric_for_row(row)
    row_env = copy.deepcopy(row.get("variant", {}).get("env", {}))
    eval_env, stripped_env_keys = tuning_eval_env(row_env)
    payload = {
        "row": {
            "arch": row.get("arch"),
            "workload": row.get("workload"),
            "phase": row.get("phase"),
            "shape_bucket": row.get("shape_bucket"),
        },
        "kernel": kernel_name,
        "metric": metric,
    }
    if suggestion:
        payload["suggestion_id"] = suggestion.get("id")
        payload["suggestion_title"] = suggestion.get("title")
    task_id = task_id or task_id_from_payload("atlas-task", payload)
    producer = {"kind": "hipfire-atlas-row"}
    if suggestion:
        producer = {
            "kind": "hipfire-atlas-suggestion",
            "suggestion": {
                "id": suggestion.get("id"),
                "rank": suggestion.get("rank"),
                "title": suggestion.get("title"),
                "lever_type": suggestion.get("lever_type"),
                "hot_kernel": suggestion.get("hot_kernel"),
                "op": copy.deepcopy(suggestion.get("op") or {}),
                "risk": suggestion.get("risk"),
                "expected_impact": suggestion.get("expected_impact"),
                "score": suggestion.get("score"),
                "history_key": suggestion.get("history_key"),
                "history": copy.deepcopy(suggestion.get("history") or {}),
                "rationale": copy.deepcopy(suggestion.get("rationale") or []),
                "candidate_steps": copy.deepcopy(suggestion.get("candidate_steps") or []),
            },
        }
    return {
        "schema": "hipfire.kernel_atlas.task.v0",
        "task_id": task_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": producer,
        "target": {
            "host": row.get("hostname"),
            "arch": row.get("arch"),
            "workload": row.get("workload"),
            "phase": row.get("phase"),
            "shape_bucket": row.get("shape_bucket"),
            "model_size": row.get("model_size"),
            "quant": row.get("quant"),
        },
        "hot_kernel": {
            "name": kernel_name,
            "profile": copy.deepcopy(kernel) if isinstance(kernel, dict) else {"name": kernel_name},
            "op": profile_kernel_op(kernel),
            "source_files": source_files,
            "dispatch_refs": copy.deepcopy(dispatch_entry.get("dispatch_refs", []) or []),
            "env_controls": copy.deepcopy(dispatch_entry.get("env_controls", []) or []),
            "isa_objects": isa_objects,
        },
        "baseline": {
            "metrics": copy.deepcopy(row.get("metrics", {})),
            "command": copy.deepcopy(row.get("command", [])),
            "env": eval_env,
            "row_env": row_env,
            "stripped_env_keys": stripped_env_keys,
            "requires_fresh_baseline": bool(stripped_env_keys),
            "provenance": copy.deepcopy(row.get("provenance", {})),
        },
        "experiment": {
            "title": suggestion.get("title") if suggestion else f"Optimize {kernel_name}",
            "lever_type": suggestion.get("lever_type") if suggestion else "source_sweep",
            "risk": suggestion.get("risk") if suggestion else "medium",
            "expected_impact": suggestion.get("expected_impact") if suggestion else "unknown",
            "rationale": copy.deepcopy(suggestion.get("rationale") or []) if suggestion else [],
            "candidate_steps": copy.deepcopy(suggestion.get("candidate_steps") or []) if suggestion else [],
        },
        "constraints": {
            "allowed_files": allowed_files,
        },
        "eval": {
            "metric": metric,
            "goal": "maximize",
            "benchmark_command": copy.deepcopy(row.get("command", [])),
            "correctness_commands": list(correctness_commands or []),
            "requires_fresh_baseline": bool(stripped_env_keys),
        },
    }


def select_suggestion(queue, *, rank=None, suggestion_id=None):
    suggestions = queue.get("suggestions", []) or []
    if rank is not None and suggestion_id:
        raise ValueError("use either --suggestion-rank or --suggestion-id, not both")
    if rank is not None:
        for item in suggestions:
            if item.get("rank") == rank:
                return item
        raise ValueError(f"suggestion rank {rank} not found in queue")
    if suggestion_id:
        for item in suggestions:
            if item.get("id") == suggestion_id:
                return item
        raise ValueError(f"suggestion id {suggestion_id} not found in queue")
    return None


def build_pytorch_task_bundle(
    *,
    name,
    op,
    input_shapes,
    dtype,
    eval_command,
    allowed_files=None,
    task_id=None,
):
    payload = {"name": name, "op": op, "input_shapes": input_shapes, "dtype": dtype}
    task_id = task_id or task_id_from_payload("atlas-pytorch", payload)
    return {
        "schema": "hipfire.kernel_atlas.task.v0",
        "task_id": task_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": {
            "kind": "pytorch",
            "name": name,
            "op": op,
            "input_shapes": list(input_shapes or []),
            "dtype": dtype,
        },
        "target": {
            "workload": name,
            "phase": "pytorch_probe",
            "shape_bucket": "|".join(input_shapes or []),
            "quant": "unknown",
        },
        "hot_kernel": {
            "name": op,
            "op": {"family": "pytorch", "role": op, "phase_hint": "probe"},
            "source_files": [],
            "dispatch_refs": [],
            "env_controls": [],
            "isa_objects": [],
        },
        "baseline": {"metrics": {}, "command": list(eval_command or []), "env": {}, "provenance": {}},
        "constraints": {"allowed_files": list(allowed_files or [])},
        "eval": {
            "metric": "score",
            "goal": "maximize",
            "benchmark_command": list(eval_command or []),
            "correctness_commands": [],
        },
    }


def render_task_markdown(task):
    hot = task.get("hot_kernel", {})
    target = task.get("target", {})
    eval_cfg = task.get("eval", {})
    baseline = task.get("baseline", {})
    allowed = task.get("constraints", {}).get("allowed_files", []) or []
    experiment = task.get("experiment", {}) or {}
    lines = [
        "# Atlas Optimization Task",
        "",
        f"Task: {task.get('task_id')}",
        f"Producer: {task.get('producer', {}).get('kind', 'unknown')}",
        f"Target: {target.get('arch', 'unknown')} {target.get('workload', 'unknown')} {target.get('phase', 'unknown')} {target.get('shape_bucket', 'unknown')}",
        "",
        "## Hot Kernel",
        f"- name: {hot.get('name', 'unknown')}",
        f"- op: {hot.get('op', {}).get('family', 'unknown')}.{hot.get('op', {}).get('role', 'unknown')}",
        "",
        "## Suggested Experiment",
        f"- title: {experiment.get('title', 'Optimize profiled kernel')}",
        f"- lever: {experiment.get('lever_type', 'unknown')}",
        f"- risk: {experiment.get('risk', 'unknown')}",
        f"- expected impact: {experiment.get('expected_impact', 'unknown')}",
    ]
    rationale = experiment.get("rationale", []) or []
    if rationale:
        lines.append("- rationale:")
        lines.extend(f"  - {item}" for item in rationale)
    steps = experiment.get("candidate_steps", []) or []
    if steps:
        lines.append("- candidate steps:")
        lines.extend(f"  - {item}" for item in steps)
    lines.extend(
        [
            "",
        "## Allowed Files",
        ]
    )
    if allowed:
        lines.extend(f"- {path}" for path in allowed)
    else:
        lines.append("- none specified")
    lines.extend(
        [
            "",
            "## Benchmark",
            "```bash",
            " ".join(shlex.quote(str(part)) for part in eval_cfg.get("benchmark_command", [])),
            "```",
            "",
            "## Metric",
            f"{eval_cfg.get('metric', 'unknown')} ({eval_cfg.get('goal', 'maximize')})",
            "",
            "## Eval Environment",
        ]
    )
    env = baseline.get("env", {}) or {}
    if env:
        lines.extend(f"- {key}={value}" for key, value in sorted(env.items()))
    else:
        lines.append("- default process environment")
    stripped = baseline.get("stripped_env_keys", []) or []
    if stripped:
        lines.append(f"- stripped instrumentation: {', '.join(stripped)}")
    if eval_cfg.get("requires_fresh_baseline"):
        lines.append("- fresh clean baseline required before candidate comparison")
    lines.extend(
        [
            "",
            "## Constraints",
            "Only edit allowed files. Preserve correctness gates. Record any changed assumptions in the result notes.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_task_bundle(task, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    task_json = root / "task.json"
    task_md = root / "TASK.md"
    task_json.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_md.write_text(render_task_markdown(task), encoding="utf-8")
    return {"task_json": str(task_json), "task_md": str(task_md)}


def normalize_command(command):
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def command_needs_shell(command):
    if isinstance(command, str):
        return True
    control = {"&&", "||", ";", "|", ">", ">>", "<"}
    return any(str(part) in control for part in command or [])


def command_for_subprocess(command):
    if command_needs_shell(command):
        if isinstance(command, str):
            return command, True
        return shlex.join(normalize_command(command)), True
    return normalize_command(command), False


def parse_metric_output(text, metric):
    parsers = (parse_bench_summary, parse_dflash_summary)
    for parser in parsers:
        try:
            parsed = parser(text)
        except ValueError:
            continue
        if metric in parsed:
            return parsed
    match = re.search(rf"(?:^|\s){re.escape(metric)}=([0-9.]+)", text)
    if match:
        return {metric: float(match.group(1))}
    match = re.search(rf"^{re.escape(metric)}:\s*([0-9.]+)", text, flags=re.MULTILINE)
    if match:
        return {metric: float(match.group(1))}
    return {}


def run_task_command(command, env=None, timeout=None, cwd=None):
    proc_command, use_shell = command_for_subprocess(command)
    proc = subprocess.run(
        proc_command,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
        timeout=timeout,
        cwd=cwd,
        shell=use_shell,
        executable="/bin/bash" if use_shell else None,
    )
    return {
        "command": proc_command,
        "shell": use_shell,
        "returncode": proc.returncode,
        "output_tail": (proc.stdout + proc.stderr)[-4000:],
    }


def summarize_values(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "stdev": None,
            "mad": None,
            "rel_spread": None,
        }
    median = statistics.median(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    mad = statistics.median([abs(value - median) for value in values])
    rel_spread = ((max(values) - min(values)) / abs(median)) if median else None
    return {
        "count": len(values),
        "median": median,
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "stdev": stdev,
        "mad": mad,
        "rel_spread": rel_spread,
    }


def summarize_metric_runs(runs):
    keys = sorted({key for run in runs for key in run.get("metrics", {})})
    summary = {}
    medians = {}
    for key in keys:
        values = [
            run["metrics"][key]
            for run in runs
            if isinstance(run.get("metrics", {}).get(key), (int, float))
        ]
        if not values:
            continue
        summary[key] = summarize_values(values)
        medians[key] = summary[key]["median"]
    return medians, summary


def run_metric_command(command, metric, env=None, timeout=None, cwd=None):
    result = run_task_command(command, env=env, timeout=timeout, cwd=cwd)
    result["metrics"] = parse_metric_output(result["output_tail"], metric)
    return result


def run_benchmark_series(command, metric, *, env=None, timeout=None, cwd=None, runs=1, warmup_runs=0):
    warmup = []
    benchmark = []
    for _ in range(max(0, int(warmup_runs))):
        warmup.append(run_metric_command(command, metric, env=env, timeout=timeout, cwd=cwd))
    for _ in range(max(1, int(runs))):
        benchmark.append(run_metric_command(command, metric, env=env, timeout=timeout, cwd=cwd))
    return warmup, benchmark


def load_baseline_file(path, metric):
    if not path:
        return None
    baseline = load_json_or_jsonl(path)
    if metric in baseline.get("metrics", {}):
        return {
            "source": str(path),
            "metrics": baseline.get("metrics", {}),
            "summary": baseline.get("summary", {}),
        }
    if metric in baseline.get("summary", {}):
        return {
            "source": str(path),
            "metrics": {metric: baseline["summary"][metric].get("median")},
            "summary": baseline.get("summary", {}),
        }
    return None


def write_baseline_json(task, output_path, metric, metrics, summary, stability, benchmark_runs, warmup_runs):
    baseline = {
        "schema": "hipfire.kernel_atlas.baseline.v0",
        "task_id": task.get("task_id"),
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric": metric,
        "metrics": metrics,
        "summary": summary,
        "stability": stability,
        "run_count": len(benchmark_runs),
        "warmup_run_count": len(warmup_runs),
    }
    (Path(output_path) / "baseline.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return baseline


def evaluate_task_bundle(
    task,
    output_dir,
    *,
    timeout=None,
    cwd=None,
    runs=1,
    warmup_runs=0,
    max_rel_spread=0.20,
    refresh_baseline=False,
    baseline_path=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eval_cfg = task.get("eval", {})
    metric = eval_cfg.get("metric", "score")
    env = task.get("baseline", {}).get("env", {})
    correctness_results = []
    status = "pass"
    for command in eval_cfg.get("correctness_commands", []) or []:
        result = run_task_command(command, env=env, timeout=timeout, cwd=cwd)
        correctness_results.append(result)
        if result["returncode"] != 0:
            status = "fail"

    warmup_results, benchmark_results = run_benchmark_series(
        eval_cfg.get("benchmark_command", []),
        metric,
        env=env,
        timeout=timeout,
        cwd=cwd,
        runs=runs,
        warmup_runs=warmup_runs,
    )
    if any(result["returncode"] != 0 for result in benchmark_results):
        status = "fail"
    metrics, summary = summarize_metric_runs(benchmark_results)
    if metric not in metrics:
        status = "fail"
    metric_summary = summary.get(metric, {})
    stability = {
        "status": "stable",
        "max_rel_spread": max_rel_spread,
        "rel_spread": metric_summary.get("rel_spread"),
    }
    if (
        status == "pass"
        and metric_summary.get("rel_spread") is not None
        and metric_summary["rel_spread"] > max_rel_spread
    ):
        status = "unstable"
        stability["status"] = "unstable"

    requires_fresh_baseline = bool(eval_cfg.get("requires_fresh_baseline"))
    baseline = load_baseline_file(baseline_path, metric)
    missing_required_baseline = requires_fresh_baseline and not refresh_baseline and baseline is None
    if missing_required_baseline:
        baseline = {
            "source": "missing-required",
            "metrics": {},
            "summary": {},
        }
        if status == "pass":
            status = "needs_baseline"
    elif baseline is None:
        baseline = {
            "source": "task",
            "metrics": copy.deepcopy(task.get("baseline", {}).get("metrics", {})),
            "summary": {},
        }
    if refresh_baseline:
        baseline = write_baseline_json(
            task,
            output_path,
            metric,
            metrics,
            summary,
            stability,
            benchmark_results,
            warmup_results,
        )
        baseline["source"] = "refreshed"

    baseline_value = baseline.get("metrics", {}).get(metric)
    current_value = metrics.get(metric)
    speedup = None
    if (
        not missing_required_baseline
        and isinstance(baseline_value, (int, float))
        and isinstance(current_value, (int, float))
        and baseline_value
    ):
        if eval_cfg.get("goal", "maximize") == "minimize":
            speedup = baseline_value / current_value if current_value else None
        else:
            speedup = current_value / baseline_value

    result = {
        "schema": "hipfire.kernel_atlas.eval.v0",
        "task_id": task.get("task_id"),
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "metric": metric,
        "metrics": metrics,
        "summary": summary,
        "stability": stability,
        "baseline": baseline,
        "delta": {"speedup": speedup},
        "provenance": {
            "git_dirty": git_is_dirty(),
            "diff_md5": md5_text(git_diff_text()),
            "binary_md5": md5_file(eval_cfg.get("benchmark_command", [None])[0]),
        },
        "correctness": correctness_results,
        "warmup_runs": warmup_results,
        "benchmark_runs": benchmark_results,
        "benchmark": benchmark_results[-1] if benchmark_results else {},
    }
    (output_path / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger_entry = {
        "task_id": task.get("task_id"),
        "captured_at_utc": result["captured_at_utc"],
        "status": status,
        "metric": metric,
        "metrics": metrics,
        "summary": summary,
        "stability": stability,
        "baseline_source": baseline.get("source"),
        "speedup": speedup,
        "diff_md5": result["provenance"]["diff_md5"],
    }
    with (output_path / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry, sort_keys=True) + "\n")
    return result


def graph_ab_delta(off_metrics, on_metrics, metric):
    off_value = off_metrics.get(metric)
    on_value = on_metrics.get(metric)
    delta = {}
    if isinstance(off_value, (int, float)) and isinstance(on_value, (int, float)):
        delta["metric"] = metric
        delta["graph_off"] = off_value
        delta["graph_on"] = on_value
        delta["absolute"] = on_value - off_value
        delta["lift"] = (on_value / off_value) if off_value else None
    for key in ("p50_ms", "avg_ms", "prefill_tok_s", "bw_gib_s"):
        off = off_metrics.get(key)
        on = on_metrics.get(key)
        if isinstance(off, (int, float)) and isinstance(on, (int, float)):
            delta[f"{key}_delta"] = on - off
    return delta


def run_graph_ab_for_row(row, *, correctness_commands=None, timeout=None, cwd=None):
    metric = preferred_metric_for_row(row)
    row_env = copy.deepcopy(row.get("variant", {}).get("env", {}))
    base_env, stripped_env_keys = tuning_eval_env(row_env)
    command = copy.deepcopy(row.get("command", []))
    status = "pass"
    modes = {}

    for graph_value, label in (("0", "graph_off"), ("1", "graph_on")):
        env = dict(base_env)
        env["HIPFIRE_GRAPH"] = graph_value
        passes = []
        for pass_index in (1, 2):
            result = run_metric_command(command, metric, env=env, timeout=timeout, cwd=cwd)
            result["pass_index"] = pass_index
            passes.append(result)
            if result["returncode"] != 0 or metric not in result.get("metrics", {}):
                status = "fail"
        headline = passes[1] if len(passes) > 1 else passes[-1]
        modes[label] = {
            "env": env,
            "passes": passes,
            "headline_pass": 2,
            "metrics": copy.deepcopy(headline.get("metrics", {})),
            "route": build_route_manifest(
                kv_mode=env.get("HIPFIRE_KV_MODE"),
                env=env,
                output=headline.get("output_tail", ""),
                graph_enabled=(graph_value == "1"),
            ),
        }

    correctness = []
    graph_on_env = modes["graph_on"]["env"]
    for command_item in correctness_commands or []:
        result = run_task_command(command_item, env=graph_on_env, timeout=timeout, cwd=cwd)
        correctness.append(result)
        if result["returncode"] != 0:
            status = "fail"

    return {
        "schema": "hipfire.kernel_atlas.graph_ab.v0",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "metric": metric,
        "target": {
            "host": row.get("hostname"),
            "arch": row.get("arch"),
            "workload": row.get("workload"),
            "phase": row.get("phase"),
            "shape_bucket": row.get("shape_bucket"),
            "quant": row.get("quant"),
        },
        "command": command,
        "row_env": row_env,
        "base_env": base_env,
        "stripped_env_keys": stripped_env_keys,
        "graph_off": modes["graph_off"],
        "graph_on": modes["graph_on"],
        "delta": graph_ab_delta(modes["graph_off"]["metrics"], modes["graph_on"]["metrics"], metric),
        "correctness": correctness,
        "provenance": {
            "git_dirty": git_is_dirty(),
            "diff_md5": md5_text(git_diff_text()),
            "binary_md5": md5_file(command[0] if command else None),
        },
    }


def load_json_or_jsonl(path, index=0):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    rows_text = [line for line in text.splitlines() if line.strip()]
    rows = [json.loads(line) for line in rows_text]
    if not rows:
        raise SystemExit(f"no JSON rows found in {path}")
    if index < 0 or index >= len(rows):
        raise SystemExit(f"row index {index} out of range for {path} ({len(rows)} rows)")
    return rows[index]


def load_manifest_for_row(row, explicit_path=None):
    if explicit_path:
        return load_json_or_jsonl(explicit_path)
    isa = row.get("artifacts", {}).get("isa", {})
    if "manifest" in isa:
        return isa["manifest"]
    if "manifest_path" in isa:
        return load_json_or_jsonl(isa["manifest_path"])
    raise SystemExit("render-fit needs --isa or a row with artifacts.isa.manifest_path")


def load_dispatch_for_row(row, explicit_path=None):
    if explicit_path:
        return load_json_or_jsonl(explicit_path)
    dispatch = row.get("artifacts", {}).get("dispatch", {})
    if "entries" in dispatch:
        return dispatch
    if "manifest" in dispatch:
        return dispatch["manifest"]
    if "manifest_path" in dispatch:
        return load_json_or_jsonl(dispatch["manifest_path"])
    return None


def base_metadata(
    *,
    arch,
    hostname,
    git_sha,
    phase,
    workload,
    model_path,
    command,
    env,
    status,
    hash_model=False,
    binary_path=None,
    git_dirty=None,
    diff_text=None,
):
    if diff_text is None:
        diff_text = git_diff_text()
    if git_dirty is None:
        git_dirty = git_is_dirty(diff_text)
    return {
        "schema": SCHEMA,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": arch,
        "hostname": hostname,
        "git_sha": git_sha,
        "phase": phase,
        "workload": workload,
        "model": {
            "path": model_path,
            "md5": md5_file(model_path) if hash_model else None,
        },
        "provenance": {
            "binary_path": binary_path,
            "binary_md5": md5_file(binary_path),
            "git_dirty": git_dirty,
            "diff_md5": md5_text(diff_text) if diff_text else None,
        },
        "command": command,
        "variant": {
            "env": dict(sorted(env.items())),
        },
        "status": status,
        "artifacts": {},
    }


def ar_rows_from_metrics(
    *,
    base,
    metrics,
    model_size,
    quant,
    prefill,
    gen,
    run_index,
    profile_sections=None,
    rocprof_coverage=None,
):
    profile_sections = profile_sections or {}
    rocprof_coverage = rocprof_coverage or {}
    prefill_row = copy.deepcopy(base)
    prefill_row.update(
        {
            "phase": "prefill",
            "workload_kind": "ar",
            "model_size": model_size,
            "quant": quant,
            "shape_bucket": f"pp{prefill}",
            "run_index": run_index,
            "metrics": {
                "prefill_tokens": prefill,
                "prefill_tok_s": metrics["prefill_tok_s"],
            },
        }
    )
    if profile_sections.get("prefill"):
        prefill_row.setdefault("artifacts", {})["profile_kernels"] = annotate_profile_kernels(
            profile_sections["prefill"]
        )

    decode_row = copy.deepcopy(base)
    decode_row.update(
        {
            "phase": "decode_ar",
            "workload_kind": "ar",
            "model_size": model_size,
            "quant": quant,
            "shape_bucket": f"decode_ar_pp{prefill}_gen{gen}",
            "run_index": run_index,
            "metrics": {
                "gen_tokens": gen,
                "gen_tok_s": metrics["gen_tok_s"],
                "bw_gib_s": metrics["bw_gib_s"],
                "avg_ms": metrics["avg_ms"],
                "p50_ms": metrics["p50_ms"],
            },
        }
    )
    if profile_sections.get("decode_ar"):
        decode_row.setdefault("artifacts", {})["profile_kernels"] = annotate_profile_kernels(
            profile_sections["decode_ar"]
        )
    if rocprof_coverage:
        coverage_keys = (
            "coverage_pct",
            "rocprof_total_ms",
            "internal_total_ms",
            "blindspot_total_ms",
            "blindspot_count",
        )
        rc_metrics = {
            f"rocprof_{k}": rocprof_coverage[k]
            for k in coverage_keys
            if k in rocprof_coverage
        }
        if rc_metrics:
            prefill_row.setdefault("metrics", {}).update(rc_metrics)
    annotate_rocprof_coverage(prefill_row)
    annotate_rocprof_coverage(decode_row)
    return [prefill_row, decode_row]


def dflash_row_from_metrics(
    *,
    base,
    metrics,
    target,
    draft,
    prompt_file,
    max_tokens,
    ctx,
    kv_mode,
    run_index,
    hash_models=False,
):
    row = copy.deepcopy(base)
    row.update(
        {
            "phase": "decode_dflash",
            "workload_kind": "dflash",
            "shape_bucket": f"dflash_max{max_tokens}_ctx{ctx}",
            "run_index": run_index,
            "metrics": metrics,
        }
    )
    row["artifacts"].update(
        {
            "target": target,
            "target_md5": md5_file(target) if hash_models else None,
            "draft": draft,
            "draft_md5": md5_file(draft) if hash_models else None,
            "prompt_file": prompt_file,
            "prompt_md5": md5_file(prompt_file),
            "kv_mode": kv_mode,
        }
    )
    return row


def parse_env(values):
    env = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"--env key cannot be empty: {item!r}")
        env[key] = value
    return env


def run_capture(command, env, timeout):
    full_env = os.environ.copy()
    full_env.update(env)
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=full_env,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


def write_rows(rows, output):
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True))


def collect_ar(args):
    rows = []
    prefills = args.prefill or [32, 128]
    diff_text = git_diff_text()
    dirty = git_is_dirty(diff_text)
    env = {
        "HIPFIRE_KV_MODE": args.kv_mode,
        "HIPFIRE_DPM_WARMUP_SECS": str(args.dpm_warmup_secs),
    }
    if args.graph:
        env["HIPFIRE_GRAPH"] = "1"
    if args.profile_prefill:
        env["HIPFIRE_PROFILE"] = "1"
    if args.profile_decode:
        env["HIPFIRE_PROFILE_DECODE"] = "1"
    env.update(parse_env(args.env))

    arch = args.arch or detect_arch()
    host = args.hostname or socket.gethostname()
    sha = args.git_sha or git_sha()

    for prefill in prefills:
        for run_index in range(1, args.runs + 1):
            command = [
                args.bench,
                args.model,
                "--prefill",
                str(prefill),
                "--prefill-runs",
                str(args.prefill_runs),
                "--warmup",
                str(args.warmup),
                "--gen",
                str(args.gen),
            ]
            code, output = run_capture(command, env, args.timeout)
            base = base_metadata(
                arch=arch,
                hostname=host,
                git_sha=sha,
                phase="ar",
                workload=args.workload,
                model_path=args.model,
                command=command,
                env=env,
                status="ok" if code == 0 else "error",
                hash_model=args.hash_models,
                binary_path=args.bench,
                git_dirty=dirty,
                diff_text=diff_text,
            )
            base["artifacts"]["runtime_route"] = build_route_manifest(
                kv_mode=args.kv_mode,
                env=env,
                output=output,
                graph_enabled=args.graph,
            )
            if code != 0:
                base["stderr_tail"] = output[-4000:]
                rows.append(base)
                continue
            metrics = parse_bench_summary(output)
            profile_sections = parse_bench_profile_sections(output)
            rocprof_coverage = parse_rocprof_coverage_section(output)
            rows.extend(
                ar_rows_from_metrics(
                    base=base,
                    metrics=metrics,
                    model_size=args.model_size,
                    quant=args.quant,
                    prefill=prefill,
                    gen=args.gen,
                    run_index=run_index,
                    profile_sections=profile_sections,
                    rocprof_coverage=rocprof_coverage,
                )
            )

    attach_isa_artifact(rows, maybe_build_isa_artifact(args, arch))
    attach_dispatch_artifact(rows, maybe_build_dispatch_artifact(args, rows))
    write_rows(rows, args.output)
    return 0


def collect_dflash(args):
    kv_mode = str(args.kv_mode).strip().lower()
    if kv_mode not in ("q8", "fwht2", "fwht3", "fwht4"):
        print("collect-dflash: DFlash perf collection requires --kv-mode q8|fwht2|fwht3|fwht4", file=sys.stderr)
        return 2
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    diff_text = git_diff_text()
    dirty = git_is_dirty(diff_text)
    env = {
        "HIPFIRE_KV_MODE": args.kv_mode,
        "HIPFIRE_DPM_WARMUP_SECS": str(args.dpm_warmup_secs),
    }
    env.update(parse_env(args.env))
    arch = args.arch or detect_arch()
    host = args.hostname or socket.gethostname()
    sha = args.git_sha or git_sha()
    rows = []

    for run_index in range(1, args.runs + 1):
        command = [
            args.demo,
            "--target",
            args.target,
            "--draft",
            args.draft,
            "--prompt",
            prompt,
            "--max",
            str(args.max_tokens),
            "--ctx",
            str(args.ctx),
            "--kv-mode",
            args.kv_mode,
        ]
        if args.no_chatml:
            command.append("--no-chatml")
        code, output = run_capture(command, env, args.timeout)
        base = base_metadata(
            arch=arch,
            hostname=host,
            git_sha=sha,
            phase="dflash",
            workload=args.workload,
            model_path=args.target,
            command=command[:],
            env=env,
            status="ok" if code == 0 else "error",
            hash_model=args.hash_models,
            binary_path=args.demo,
            git_dirty=dirty,
            diff_text=diff_text,
        )
        base["artifacts"]["runtime_route"] = build_route_manifest(
            kv_mode=args.kv_mode,
            env=env,
            output=output,
            graph_enabled=env.get("HIPFIRE_GRAPH") == "1",
        )
        base["command"] = [part if part != prompt else f"@{args.prompt_file}" for part in command]
        if code != 0:
            base["stderr_tail"] = output[-4000:]
            rows.append(base)
            continue
        metrics = parse_dflash_summary(output)
        rows.append(
            dflash_row_from_metrics(
                base=base,
                metrics=metrics,
                target=args.target,
                draft=args.draft,
                prompt_file=args.prompt_file,
                max_tokens=args.max_tokens,
                ctx=args.ctx,
                kv_mode=args.kv_mode,
                run_index=run_index,
                hash_models=args.hash_models,
            )
        )

    attach_isa_artifact(rows, maybe_build_isa_artifact(args, arch))
    attach_dispatch_artifact(rows, maybe_build_dispatch_artifact(args, rows))
    write_rows(rows, args.output)
    return 0


def parse_file(args):
    text = Path(args.input).read_text(encoding="utf-8")
    if args.kind == "bench":
        data = parse_bench_summary(text)
    else:
        data = parse_dflash_summary(text)
    print(json.dumps(data, sort_keys=True))
    return 0


def render_fit(args):
    row = load_json_or_jsonl(args.row, args.row_index)
    manifest = load_manifest_for_row(row, args.isa)
    dispatch = load_dispatch_for_row(row, args.dispatch)
    if dispatch:
        row = copy.deepcopy(row)
        row.setdefault("artifacts", {})["dispatch"] = dispatch
    print(render_fit_view(row, manifest))
    return 0


def suggest_from_row(args):
    row = load_json_or_jsonl(args.row, args.row_index)
    manifest = load_manifest_for_row(row, args.isa)
    dispatch = load_dispatch_for_row(row, args.dispatch)
    if dispatch:
        row = copy.deepcopy(row)
        row.setdefault("artifacts", {})["dispatch"] = dispatch
    correctness = list(args.correctness_command or [])
    history = load_suggestion_history(args.history, root=Path.cwd())
    queue = build_suggestion_queue(
        row,
        manifest,
        dispatch,
        max_suggestions=args.max_suggestions,
        hot_limit=args.hot_limit,
        correctness_commands=correctness,
        history=history,
    )
    if args.format == "markdown":
        text = render_suggestion_markdown(queue)
    else:
        text = json.dumps(queue, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def task_from_row(args):
    row = load_json_or_jsonl(args.row, args.row_index)
    manifest = load_manifest_for_row(row, args.isa)
    dispatch = load_dispatch_for_row(row, args.dispatch)
    if dispatch:
        row = copy.deepcopy(row)
        row.setdefault("artifacts", {})["dispatch"] = dispatch
    correctness = list(args.correctness_command or [])
    suggestion = None
    if args.suggestion_rank is not None or args.suggestion_id:
        history = load_suggestion_history(args.history, root=Path.cwd())
        queue = build_suggestion_queue(
            row,
            manifest,
            dispatch,
            max_suggestions=args.max_suggestions,
            hot_limit=args.hot_limit,
            correctness_commands=correctness,
            history=history,
        )
        try:
            suggestion = select_suggestion(
                queue,
                rank=args.suggestion_rank,
                suggestion_id=args.suggestion_id,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    task = build_task_bundle_from_row(
        row,
        manifest,
        allowed_files=args.allowed_file or None,
        correctness_commands=correctness,
        task_id=args.task_id,
        suggestion=suggestion,
    )
    output_dir = args.output_dir or str(
        Path(".codeinsight+research") / "kernel-atlas" / "tasks" / task["task_id"]
    )
    paths = write_task_bundle(task, output_dir)
    print(json.dumps(paths, sort_keys=True))
    return 0


def task_from_pytorch(args):
    task = build_pytorch_task_bundle(
        name=args.name,
        op=args.op,
        input_shapes=args.input_shape,
        dtype=args.dtype,
        eval_command=normalize_command(args.eval_command),
        allowed_files=args.allowed_file,
        task_id=args.task_id,
    )
    output_dir = args.output_dir or str(
        Path(".codeinsight+research") / "kernel-atlas" / "tasks" / task["task_id"]
    )
    paths = write_task_bundle(task, output_dir)
    print(json.dumps(paths, sort_keys=True))
    return 0


def eval_task(args):
    task = load_json_or_jsonl(args.task)
    result = evaluate_task_bundle(
        task,
        output_dir=args.output_dir,
        timeout=args.timeout,
        cwd=args.cwd,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        max_rel_spread=args.max_rel_spread,
        refresh_baseline=args.refresh_baseline,
        baseline_path=args.baseline,
    )
    print(json.dumps({"result_path": str(Path(args.output_dir) / "result.json"), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] in ("pass", "unstable") else 1


def graph_ab_from_row(args):
    row = load_json_or_jsonl(args.row, args.row_index)
    correctness = list(args.correctness_command or [])
    result = run_graph_ab_for_row(
        row,
        correctness_commands=correctness,
        timeout=args.timeout,
        cwd=args.cwd,
    )
    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if result["status"] == "pass" else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_isa_args(p):
        p.add_argument("--isa-dir", action="append", default=[], help="scan a directory for .hsaco/.co/.o files")
        p.add_argument("--isa-file", action="append", default=[], help="inspect one HSACO/code object")
        p.add_argument("--isa-filter", help="regex filter applied to ISA object paths")
        p.add_argument("--isa-limit", type=int, help="maximum number of ISA objects to inspect")
        p.add_argument("--isa-output", help="write ISA manifest here and reference it from each row")

    def add_dispatch_args(p):
        p.add_argument("--dispatch-provenance", action="store_true", help="scan source/dispatch refs for profiled kernels")
        p.add_argument("--dispatch-root", default=".", help="repo root for dispatch provenance scans")
        p.add_argument("--dispatch-ref-limit", type=int, default=8, help="max source/dispatch refs per kernel")
        p.add_argument("--dispatch-output", help="write dispatch manifest here and reference it from each row")

    parse = sub.add_parser("parse", help="parse saved bench output")
    parse.add_argument("kind", choices=("bench", "dflash"))
    parse.add_argument("input")
    parse.set_defaults(func=parse_file)

    fit = sub.add_parser(
        "render-fit",
        help="render an ASCII ISA/quant fit view",
        description="render an ASCII ISA/quant fit view",
    )
    fit.add_argument("--row", required=True, help="Atlas JSONL/JSON row file")
    fit.add_argument("--row-index", type=int, default=0, help="row index when --row is JSONL")
    fit.add_argument("--isa", help="ISA manifest JSON; defaults to row artifacts.isa.manifest_path")
    fit.add_argument("--dispatch", help="dispatch manifest JSON; defaults to row artifacts.dispatch.manifest_path")
    fit.set_defaults(func=render_fit)

    suggest = sub.add_parser("suggest", help="emit ranked Atlas tuning suggestions")
    suggest.add_argument("--row", required=True, help="Atlas JSONL/JSON row file")
    suggest.add_argument("--row-index", type=int, default=0, help="row index when --row is JSONL")
    suggest.add_argument("--isa", help="ISA manifest JSON; defaults to row artifacts.isa.manifest_path")
    suggest.add_argument("--dispatch", help="dispatch manifest JSON; defaults to row artifacts.dispatch.manifest_path")
    suggest.add_argument("--max-suggestions", type=int, default=12)
    suggest.add_argument("--hot-limit", type=int, default=8)
    suggest.add_argument("--history", action="append", default=[], help="add an Atlas history file or directory")
    suggest.add_argument("--correctness-command", action="append", default=[], help="command string for candidate correctness gate")
    suggest.add_argument("--format", choices=("json", "markdown"), default="json")
    suggest.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    suggest.add_argument("--output", help="write suggestions here instead of stdout")
    suggest.set_defaults(func=suggest_from_row)

    task = sub.add_parser("task", help="emit an optimization task bundle from an Atlas row")
    task.add_argument("--row", required=True, help="Atlas JSONL/JSON row file")
    task.add_argument("--row-index", type=int, default=0, help="row index when --row is JSONL")
    task.add_argument("--isa", help="ISA manifest JSON; defaults to row artifacts.isa.manifest_path")
    task.add_argument("--dispatch", help="dispatch manifest JSON; defaults to row artifacts.dispatch.manifest_path")
    task.add_argument("--allowed-file", action="append", default=[], help="file an agent may edit")
    task.add_argument("--correctness-command", action="append", default=[], help="command string for correctness gate")
    task.add_argument("--suggestion-rank", type=int, help="build the task from the ranked suggestion N")
    task.add_argument("--suggestion-id", help="build the task from a specific suggestion id")
    task.add_argument("--max-suggestions", type=int, default=12, help="suggestion queue size when using --suggestion-rank/id")
    task.add_argument("--hot-limit", type=int, default=8, help="hot profile rows considered when using --suggestion-rank/id")
    task.add_argument("--history", action="append", default=[], help="add Atlas history when selecting a suggestion")
    task.add_argument("--task-id")
    task.add_argument("--output-dir")
    task.set_defaults(func=task_from_row)

    task_pt = sub.add_parser("task-pytorch", help="emit a PyTorch-shape optimization task bundle")
    task_pt.add_argument("--name", required=True)
    task_pt.add_argument("--op", required=True)
    task_pt.add_argument("--input-shape", action="append", required=True)
    task_pt.add_argument("--dtype", default="float16")
    task_pt.add_argument("--eval-command", required=True, help="benchmark/eval command string")
    task_pt.add_argument("--allowed-file", action="append", default=[], help="file an agent may edit")
    task_pt.add_argument("--task-id")
    task_pt.add_argument("--output-dir")
    task_pt.set_defaults(func=task_from_pytorch)

    ev = sub.add_parser("eval", help="run an Atlas task benchmark/correctness contract")
    ev.add_argument("--task", required=True, help="task.json")
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--timeout", type=int)
    ev.add_argument("--cwd")
    ev.add_argument("--runs", type=int, default=1, help="fresh benchmark runs to summarize")
    ev.add_argument("--warmup-runs", type=int, default=0, help="fresh benchmark runs to discard before measurement")
    ev.add_argument("--max-rel-spread", type=float, default=0.20, help="mark unstable when (max-min)/median exceeds this")
    ev.add_argument("--refresh-baseline", action="store_true", help="write baseline.json from this run series and compare against it")
    ev.add_argument("--baseline", help="baseline.json to compare against instead of task baseline")
    ev.set_defaults(func=eval_task)

    graph = sub.add_parser("graph-ab", help="run HIPFIRE_GRAPH=0/1 A/B from an Atlas row")
    graph.add_argument("--row", required=True, help="Atlas JSONL/JSON row file")
    graph.add_argument("--row-index", type=int, default=0, help="row index when --row is JSONL")
    graph.add_argument("--correctness-command", action="append", default=[], help="graph-on correctness command string")
    graph.add_argument("--timeout", type=float)
    graph.add_argument("--cwd")
    graph.add_argument("--pretty", action="store_true")
    graph.add_argument("--output", help="write graph A/B JSON here instead of stdout")
    graph.set_defaults(func=graph_ab_from_row)

    ar = sub.add_parser("collect-ar", help="collect AR prefill/decode rows")
    ar.add_argument("--bench", default="./target/release/examples/bench_qwen35_mq4")
    ar.add_argument("--model", required=True)
    ar.add_argument("--workload", default="qwen3.5")
    ar.add_argument("--model-size", default="unknown")
    ar.add_argument("--quant", default="mq4")
    ar.add_argument("--prefill", type=int, action="append")
    ar.add_argument("--prefill-runs", type=int, default=1)
    ar.add_argument("--gen", type=int, default=50)
    ar.add_argument("--warmup", type=int, default=5)
    ar.add_argument("--runs", type=int, default=1)
    ar.add_argument("--kv-mode", default="asym3")
    ar.add_argument("--dpm-warmup-secs", default="3")
    ar.add_argument("--graph", action="store_true")
    ar.add_argument("--profile-prefill", action="store_true", help="capture prefill per-kernel profile table")
    ar.add_argument("--profile-decode", action="store_true", help="capture decode per-kernel profile table")
    ar.add_argument("--hash-models", action="store_true")
    ar.add_argument("--env", action="append", default=[])
    ar.add_argument("--arch")
    ar.add_argument("--hostname")
    ar.add_argument("--git-sha")
    ar.add_argument("--timeout", type=int, default=600)
    ar.add_argument("--output")
    add_isa_args(ar)
    add_dispatch_args(ar)
    ar.set_defaults(func=collect_ar)

    df = sub.add_parser("collect-dflash", help="collect DFlash decode rows")
    df.add_argument("--demo", default="./target/release/examples/dflash_spec_demo")
    df.add_argument("--target", required=True)
    df.add_argument("--draft", required=True)
    df.add_argument("--prompt-file", required=True)
    df.add_argument("--workload", default="qwen3.5-dflash")
    df.add_argument("--max-tokens", type=int, default=256)
    df.add_argument("--ctx", type=int, default=2048)
    df.add_argument("--runs", type=int, default=1)
    df.add_argument("--kv-mode", default="q8", help="DFlash KV mode; use q8 or fwht*, not asym* for perf gates")
    df.add_argument("--dpm-warmup-secs", default="10")
    df.add_argument("--no-chatml", action="store_true", default=True)
    df.add_argument("--chatml", action="store_false", dest="no_chatml")
    df.add_argument("--hash-models", action="store_true")
    df.add_argument("--env", action="append", default=[])
    df.add_argument("--arch")
    df.add_argument("--hostname")
    df.add_argument("--git-sha")
    df.add_argument("--timeout", type=int, default=900)
    df.add_argument("--output")
    add_isa_args(df)
    add_dispatch_args(df)
    df.set_defaults(func=collect_dflash)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
