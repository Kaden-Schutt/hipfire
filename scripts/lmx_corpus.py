#!/usr/bin/env python3
"""
lmx_corpus.py — LocalMax corpus sealer (developer-only orchestration)

Three subcommands:
  init --root ROOT [--machine-out PATH]
  seal-case --root ROOT --family FAMILY --case-id ID --metadata FILE --summary FILE --command FILE --environment FILE [--raw FILE ...] [--decoded FILE ...] [--validation FILE ...]
  manifest --root ROOT

Atomic JSON/text writes via temp-file + os.replace, SHA-256 streaming,
deterministic sorted outputs, symlink refusal, idempotent reseal.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile

REQUIRED_TOP_DIRS = [
    "qwen-0.8b-mq4",
    "qwen-a3b-moe",
    "qwen-a3b-batched",
    "deepseek-v4",
    "lfm2.5-230m",
    "lfm2.5-350m",
    "lmx-packages",
    "raw",
    "validation",
]

# families that can hold cases (the 6 model families). Validation for seal-case
# is intentionally traversal-only per spec, but we keep the list for discovery.
CASE_FAMILIES = [
    "qwen-0.8b-mq4",
    "qwen-a3b-moe",
    "qwen-a3b-batched",
    "deepseek-v4",
    "lfm2.5-230m",
    "lfm2.5-350m",
]

SCHEMA_VERSION = "1"
SCHEMA = "lmx-corpus/1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def error(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def utc_now() -> str:
    # RFC3339 Zulu, deterministic format
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def atomic_write_text(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def atomic_copy_stream(src: str, dst: str) -> None:
    d = os.path.dirname(os.path.abspath(dst)) or "."
    os.makedirs(d, exist_ok=True)
    # refuse to follow destination symlink
    if os.path.lexists(dst) and os.path.islink(dst):
        error(f"destination is a symlink (refusing to follow): {dst}")
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(dst) + ".tmp.")
    try:
        with open(src, "rb") as sf:
            with os.fdopen(fd, "wb") as df:
                for chunk in iter(lambda: sf.read(8192), b""):
                    df.write(chunk)
                df.flush()
                os.fsync(df.fileno())
        os.replace(tmp, dst)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def validate_safe_name(name: str, label: str) -> None:
    if not isinstance(name, str) or not name:
        error(f"{label} must be non-empty")
    if "/" in name or "\\" in name or "\x00" in name:
        error(f"{label} must not contain path separators or null: {name!r}")
    if name in (".", ".."):
        error(f"{label} must not be '.' or '..'")
    if ".." in name:
        error(f"{label} must not contain '..': {name!r}")
    if os.path.isabs(name):
        error(f"{label} must not be absolute: {name!r}")
    # control chars / newline
    if any(ord(c) < 0x20 for c in name):
        error(f"{label} must not contain control characters: {name!r}")


def capture_tool(name: str) -> dict:
    info: dict = {"available": False, "output": None, "version": None}
    # try variants
    candidates = [
        [name, "--version"],
        [name, "-v"],
        [name, "--help"],
        [name],
    ]
    for argv in candidates:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=3)
            out = (proc.stdout or "") + (proc.stderr or "")
            out = out.strip()
            if out:
                info["available"] = True
                info["output"] = out[:8192]
                first = out.splitlines()[0] if out else ""
                info["version"] = first[:512]
                return info
            if proc.returncode == 0:
                info["available"] = True
                return info
        except FileNotFoundError:
            return info
        except subprocess.TimeoutExpired:
            info["output"] = "timeout"
            return info
        except Exception as e:
            info["output"] = str(e)[:512]
            return info
    return info


def capture_machine_info() -> dict:
    info: dict = {}
    info["generated_utc"] = utc_now()
    try:
        info["hostname"] = socket.gethostname()
    except Exception:
        info["hostname"] = ""
    try:
        info["platform"] = platform.platform()
    except Exception:
        info["platform"] = ""
    try:
        u = platform.uname()
        info["uname"] = {
            "system": u.system,
            "node": u.node,
            "release": u.release,
            "version": u.version,
            "machine": u.machine,
            "processor": u.processor,
        }
        info["uname_str"] = " ".join([u.system, u.node, u.release, u.version, u.machine])
    except Exception:
        try:
            info["uname_str"] = " ".join(os.uname())  # type: ignore
        except Exception:
            info["uname_str"] = ""
    # /etc/os-release
    try:
        if os.path.isfile("/etc/os-release") and os.access("/etc/os-release", os.R_OK):
            with open("/etc/os-release", "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            info["os_release"] = content[:8192]
            parsed: dict = {}
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    parsed[k] = v
            info["os_release_parsed"] = parsed
    except Exception:
        pass
    # best-effort tool captures
    for tool in ["rocminfo", "rocm-smi", "hipconfig"]:
        # use dash-preserving key but also underscore alias for convenience
        data = capture_tool(tool)
        info[tool] = data
        # also store underscore variant
        alias = tool.replace("-", "_")
        if alias != tool:
            info[alias] = data
    return info


def flatten_appended(value) -> list:
    if value is None:
        return []
    # value is list of lists (due to action=append + nargs=*)
    out: list = []
    for entry in value:
        if entry is None:
            continue
        if isinstance(entry, list):
            out.extend(entry)
        else:
            out.append(entry)
    return out


def compute_case_checksums(case_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (rel_posix, sha256) for every file except checksums.sha256."""
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(case_dir, topdown=True, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for d in list(dirnames):
            dp = os.path.join(dirpath, d)
            if os.path.islink(dp):
                error(f"symlink directory not allowed: {dp}")
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if os.path.islink(fpath):
                error(f"symlink file not allowed: {fpath}")
            rel = os.path.relpath(fpath, case_dir).replace(os.sep, "/")
            if rel == "checksums.sha256":
                continue
            files.append(rel)
    files.sort()
    result: list[tuple[str, str]] = []
    for rel in files:
        abs_path = os.path.join(case_dir, rel.replace("/", os.sep))
        h = sha256_file(abs_path)
        result.append((rel, h))
    return result


def write_checksums_file(path: str, checksums: list[tuple[str, str]]) -> None:
    # deterministic: already sorted
    lines = "".join(f"{h}  {rel}\n" for rel, h in checksums)
    atomic_write_text(path, lines)


def cases_equal(staged: str, existing: str) -> bool:
    # compare file inventories via hashes (excluding checksums.sha256 for now)
    staged_sums = compute_case_checksums(staged)
    existing_sums = compute_case_checksums(existing)
    if staged_sums != existing_sums:
        return False
    # compare checksums.sha256 content
    s_cs = os.path.join(staged, "checksums.sha256")
    e_cs = os.path.join(existing, "checksums.sha256")
    try:
        with open(s_cs, "r", encoding="utf-8") as a, open(e_cs, "r", encoding="utf-8") as b:
            if a.read() != b.read():
                return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_init(args) -> None:
    root = os.path.abspath(args.root)
    # refuse to follow symlink for root if it already exists as symlink
    if os.path.lexists(root) and os.path.islink(root):
        error(f"root is a symlink (refusing to follow): {root}")
    os.makedirs(root, exist_ok=True)
    if os.path.islink(root):
        error(f"root is a symlink (refusing to follow): {root}")

    for d in REQUIRED_TOP_DIRS:
        p = os.path.join(root, d)
        if os.path.lexists(p):
            if os.path.islink(p):
                error(f"required path is a symlink (refusing to follow): {p}")
            if not os.path.isdir(p):
                error(f"required path exists but is not a directory: {p}")
        else:
            os.makedirs(p, exist_ok=True)
            if os.path.islink(p):
                error(f"required path is a symlink after creation: {p}")

    info = capture_machine_info()
    machine_path = os.path.join(root, "machine.json")
    atomic_write_json(machine_path, info)
    if args.machine_out:
        out = os.path.abspath(args.machine_out)
        parent = os.path.dirname(out) or "."
        os.makedirs(parent, exist_ok=True)
        # also refuse symlink for destination parent? only file itself
        if os.path.lexists(out) and os.path.islink(out):
            error(f"machine-out destination is a symlink: {out}")
        atomic_write_json(out, info)


def cmd_seal_case(args) -> None:
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        error(f"root does not exist or is not a directory: {root}")
    if os.path.islink(root):
        error(f"root is a symlink (refusing to follow): {root}")

    family = args.family
    case_id = args.case_id
    validate_safe_name(family, "family")
    validate_safe_name(case_id, "case-id")

    family_path = os.path.join(root, family)
    case_path = os.path.join(family_path, case_id)

    # never follow destination symlink
    for p in (family_path, case_path):
        if os.path.lexists(p) and os.path.islink(p):
            error(f"destination is a symlink (refusing to follow): {p}")

    # validate input files existence
    for label, path in [
        ("metadata", args.metadata),
        ("summary", args.summary),
        ("command", args.command),
        ("environment", args.environment),
    ]:
        if not path:
            error(f"missing required --{label} argument")
        if not os.path.isfile(path):
            error(f"{label} file not found: {path}")

    # validate JSON objects
    metadata_obj = None
    summary_obj = None
    for label, path in [("metadata", args.metadata), ("summary", args.summary)]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            error(f"{label} is not valid JSON: {e}")
        if not isinstance(obj, dict):
            error(f"{label} must be a JSON object (got {type(obj).__name__})")
        if label == "metadata":
            metadata_obj = obj
        else:
            summary_obj = obj

    raw_files = flatten_appended(args.raw)
    decoded_files = flatten_appended(args.decoded)
    validation_files = flatten_appended(args.validation)

    for category, lst in [("raw", raw_files), ("decoded", decoded_files), ("validation", validation_files)]:
        for p in lst:
            if not os.path.isfile(p):
                error(f"{category} file not found: {p}")
        basenames = [os.path.basename(p) for p in lst]
        # empty basename (e.g., path ends with /) is invalid
        for b in basenames:
            if not b:
                error(f"{category} file has empty basename: {p!r}")
            validate_safe_name(b, f"{category} basename")
        if len(basenames) != len(set(basenames)):
            seen = set()
            dups = set()
            for b in basenames:
                if b in seen:
                    dups.add(b)
                seen.add(b)
            error(f"colliding basenames in --{category}: {', '.join(sorted(dups))}")

    # stage in temp directory inside root for same-filesystem atomic rename
    staging_root = tempfile.mkdtemp(prefix=".tmp-seal-", dir=root)
    staged_case = os.path.join(staging_root, "case")
    try:
        os.makedirs(staged_case, exist_ok=True)
        for sub in ["raw", "decoded", "validation"]:
            os.makedirs(os.path.join(staged_case, sub), exist_ok=True)

        # copy artifacts via atomic streaming
        for src in raw_files:
            dst = os.path.join(staged_case, "raw", os.path.basename(src))
            atomic_copy_stream(src, dst)
        for src in decoded_files:
            dst = os.path.join(staged_case, "decoded", os.path.basename(src))
            atomic_copy_stream(src, dst)
        for src in validation_files:
            dst = os.path.join(staged_case, "validation", os.path.basename(src))
            atomic_copy_stream(src, dst)

        # install canonical files atomically
        assert metadata_obj is not None and summary_obj is not None
        atomic_write_json(os.path.join(staged_case, "metadata.json"), metadata_obj)
        atomic_write_json(os.path.join(staged_case, "summary.json"), summary_obj)
        atomic_copy_stream(args.command, os.path.join(staged_case, "command.txt"))
        atomic_copy_stream(args.environment, os.path.join(staged_case, "environment.txt"))

        # compute and write checksums
        checksums = compute_case_checksums(staged_case)
        write_checksums_file(os.path.join(staged_case, "checksums.sha256"), checksums)

        # if destination exists, enforce exact-idempotent
        if os.path.lexists(case_path):
            if os.path.islink(case_path):
                error(f"destination is a symlink (refusing to follow): {case_path}")
            if not os.path.isdir(case_path):
                error(f"destination exists but is not a directory: {case_path}")
            if cases_equal(staged_case, case_path):
                # idempotent — clean up and succeed
                shutil.rmtree(staging_root, ignore_errors=True)
                return
            else:
                error(f"sealed case already exists with different contents: {case_path} (refusing to overwrite)")

        # ensure family dir exists (and not symlink)
        if os.path.lexists(family_path) and os.path.islink(family_path):
            error(f"destination is a symlink (refusing to follow): {family_path}")
        os.makedirs(family_path, exist_ok=True)
        if os.path.islink(family_path):
            error(f"destination is a symlink after creation: {family_path}")

        # atomic move of staged case to final location
        # os.rename is atomic on same filesystem
        os.rename(staged_case, case_path)
        # remove empty staging root
        try:
            os.rmdir(staging_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
    except SystemExit:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(staging_root, ignore_errors=True)
        if isinstance(e, SystemExit):
            raise
        error(str(e))


def cmd_manifest(args) -> None:
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        error(f"root does not exist or is not a directory: {root}")
    if os.path.islink(root):
        error(f"root is a symlink (refusing to follow): {root}")

    # discover cases under CASE_FAMILIES (plus any family that looks like a case holder
    # but we limit to CASE_FAMILIES for determinism; also scan any top-level dir that contains cases
    # to avoid missing families not in the list, we scan all top-level entries that are dirs
    # except the known non-family dirs.
    non_family = {"lmx-packages", "raw", "validation"}
    cases: list[tuple[str, str, str]] = []
    # first scan CASE_FAMILIES deterministically
    for fam in CASE_FAMILIES:
        fam_path = os.path.join(root, fam)
        if not os.path.isdir(fam_path):
            continue
        if os.path.islink(fam_path):
            error(f"family path is a symlink: {fam_path}")
        try:
            entries = os.listdir(fam_path)
        except Exception as e:
            error(f"cannot list family {fam}: {e}")
        entries.sort()
        for entry in entries:
            if entry.startswith("."):
                continue
            case_id = entry
            case_path = os.path.join(fam_path, case_id)
            if os.path.islink(case_path):
                error(f"case path is a symlink: {case_path}")
            if not os.path.isdir(case_path):
                continue
            cases.append((fam, case_id, case_path))

    # also discover any other top-level dirs that might hold cases but are not in CASE_FAMILIES
    # (e.g., if user used a different family). We include them if they are not in non_family
    # and not already covered, and contain at least one subdir with metadata.json — to avoid
    # false positives, we only add those that look like case dirs.
    try:
        top_entries = os.listdir(root)
    except Exception as e:
        error(f"cannot list root: {e}")
    for entry in sorted(top_entries):
        if entry in CASE_FAMILIES or entry in non_family:
            continue
        if entry.startswith("."):
            continue
        fam_path = os.path.join(root, entry)
        if os.path.islink(fam_path) or not os.path.isdir(fam_path):
            continue
        # check if it contains any case-like subdir
        try:
            subs = os.listdir(fam_path)
        except Exception:
            continue
        for sub in sorted(subs):
            if sub.startswith("."):
                continue
            case_path = os.path.join(fam_path, sub)
            if os.path.islink(case_path) or not os.path.isdir(case_path):
                continue
            # if it has metadata.json, treat as case
            if os.path.isfile(os.path.join(case_path, "metadata.json")):
                # ensure not already added
                if not any(c[0] == entry and c[1] == sub for c in cases):
                    validate_safe_name(entry, "family")
                    validate_safe_name(sub, "case-id")
                    cases.append((entry, sub, case_path))

    cases.sort(key=lambda x: (x[0], x[1]))

    # verify each case
    for fam, cid, cpath in cases:
        # canonical files
        for fname in ["metadata.json", "summary.json", "command.txt", "environment.txt", "checksums.sha256"]:
            fpath = os.path.join(cpath, fname)
            if not os.path.isfile(fpath):
                error(f"case {fam}/{cid} missing canonical file: {fname}")
            if os.path.islink(fpath):
                error(f"case {fam}/{cid} file is a symlink: {fname}")
        for sub in ["raw", "decoded", "validation"]:
            sp = os.path.join(cpath, sub)
            if not os.path.isdir(sp):
                error(f"case {fam}/{cid} missing directory: {sub}")
            if os.path.islink(sp):
                error(f"case {fam}/{cid} directory is a symlink: {sub}")

        # JSON object check
        for fname in ["metadata.json", "summary.json"]:
            fpath = os.path.join(cpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception as e:
                error(f"case {fam}/{cid} {fname} invalid JSON: {e}")
            if not isinstance(obj, dict):
                error(f"case {fam}/{cid} {fname} must be a JSON object")

        # checksums verification
        cs_path = os.path.join(cpath, "checksums.sha256")
        # parse
        expected: dict[str, str] = {}
        file_order: list[str] = []
        try:
            with open(cs_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    # parse "<hash>  <path>" — hash and path separated by whitespace (usually two spaces)
                    if "  " in raw_line:
                        h, _, rel = raw_line.strip().partition("  ")
                        h = h.strip()
                        rel = rel.strip()
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            error(f"case {fam}/{cid} malformed checksums line: {raw_line!r}")
                        h, rel = parts[0], parts[1]
                        # if path contained spaces, re-join remainder
                        if len(parts) > 2:
                            # reconstruct by splitting on hash
                            # fallback: take everything after hash in raw_line
                            idx = raw_line.find(h)
                            rel = raw_line[idx + len(h):].strip()
                    if len(h) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
                        error(f"case {fam}/{cid} invalid hash in checksums: {h!r}")
                    h = h.lower()
                    if rel in expected:
                        error(f"case {fam}/{cid} duplicate checksums entry: {rel}")
                    expected[rel] = h
                    file_order.append(rel)
        except SystemExit:
            raise
        except Exception as e:
            error(f"case {fam}/{cid} cannot read checksums.sha256: {e}")

        # compute actual
        actual: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(cpath, topdown=True, followlinks=False):
            dirnames.sort()
            filenames.sort()
            for d in dirnames:
                dp = os.path.join(dirpath, d)
                if os.path.islink(dp):
                    error(f"case {fam}/{cid} directory is a symlink: {dp}")
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if os.path.islink(fpath):
                    error(f"case {fam}/{cid} file is a symlink: {fpath}")
                rel = os.path.relpath(fpath, cpath).replace(os.sep, "/")
                if rel == "checksums.sha256":
                    continue
                actual[rel] = sha256_file(fpath)

        if set(expected.keys()) != set(actual.keys()):
            missing = sorted(set(actual.keys()) - set(expected.keys()))
            extra = sorted(set(expected.keys()) - set(actual.keys()))
            msgs = []
            if missing:
                msgs.append(f"missing from checksums: {missing}")
            if extra:
                msgs.append(f"extra in checksums: {extra}")
            error(f"case {fam}/{cid} checksums inventory mismatch: {'; '.join(msgs)}")

        for rel, h in expected.items():
            if actual.get(rel) != h:
                error(f"case {fam}/{cid} checksum mismatch for {rel}: expected {h}, got {actual.get(rel)}")

        # ensure deterministic sorted order in file
        if file_order != sorted(file_order):
            error(f"case {fam}/{cid} checksums.sha256 not sorted")

    # machine identity hash
    machine_path = os.path.join(root, "machine.json")
    machine_hash = None
    if os.path.isfile(machine_path):
        if os.path.islink(machine_path):
            error("machine.json is a symlink")
        machine_hash = sha256_file(machine_path)

    # build manifest cases
    manifest_cases = []
    for fam, cid, cpath in cases:
        with open(os.path.join(cpath, "metadata.json"), "r", encoding="utf-8") as f:
            md = json.load(f)
        with open(os.path.join(cpath, "summary.json"), "r", encoding="utf-8") as f:
            sm = json.load(f)
        # file hashes (same as actual, but recompute to be safe and sorted)
        file_hashes: dict[str, str] = {}
        for dirpath, _, filenames in os.walk(cpath, topdown=True, followlinks=False):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if os.path.islink(fpath):
                    continue
                rel = os.path.relpath(fpath, cpath).replace(os.sep, "/")
                if rel == "checksums.sha256":
                    continue
                file_hashes[rel] = sha256_file(fpath)
        file_hashes = dict(sorted(file_hashes.items()))
        manifest_cases.append({
            "family": fam,
            "id": cid,
            "metadata": md,
            "summary": sm,
            "file_checksums": file_hashes,
        })

    # top-level package hashes: all files under lmx-packages, deterministic
    package_hashes: dict[str, str] = {}
    pkg_dir = os.path.join(root, "lmx-packages")
    if os.path.isdir(pkg_dir):
        if os.path.islink(pkg_dir):
            error("lmx-packages is a symlink")
        for dirpath, dirnames, filenames in os.walk(pkg_dir, topdown=True, followlinks=False):
            dirnames.sort()
            filenames.sort()
            for d in dirnames:
                dp = os.path.join(dirpath, d)
                if os.path.islink(dp):
                    error(f"package directory is a symlink: {dp}")
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if os.path.islink(fpath):
                    error(f"package file is a symlink: {fpath}")
                rel = os.path.relpath(fpath, root).replace(os.sep, "/")
                # exclude MANIFEST files if they somehow appear under lmx-packages (shouldn't)
                if rel in ("MANIFEST.json", "MANIFEST.sha256"):
                    continue
                package_hashes[rel] = sha256_file(fpath)
        package_hashes = dict(sorted(package_hashes.items()))

    generated_utc = utc_now()
    manifest_obj = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA,
        "generated_utc": generated_utc,
        "machine_identity_hash": machine_hash,
        "cases": manifest_cases,
        "package_hashes": package_hashes,
    }

    manifest_path = os.path.join(root, "MANIFEST.json")
    # exclude MANIFEST files from their own inventory — we never include them
    atomic_write_json(manifest_path, manifest_obj)
    # MANIFEST.sha256 over MANIFEST.json
    h = sha256_file(manifest_path)
    sha_path = os.path.join(root, "MANIFEST.sha256")
    atomic_write_text(sha_path, f"{h}  MANIFEST.json\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lmx_corpus.py",
        description="LocalMax corpus sealer — developer-only orchestration for deterministic corpus layout",
    )
    subs = p.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    pi = subs.add_parser("init", help="create required top-level family directories and atomic machine.json")
    pi.add_argument("--root", required=True, help="corpus root directory")
    pi.add_argument("--machine-out", dest="machine_out", default=None, help="optional additional path to write machine.json")
    pi.set_defaults(func=cmd_init)

    ps = subs.add_parser("seal-case", help="validate and seal a case into ROOT/FAMILY/CASE_ID")
    ps.add_argument("--root", required=True, help="corpus root")
    ps.add_argument("--family", required=True, help="family name (no traversal)")
    ps.add_argument("--case-id", required=True, dest="case_id", help="case identifier (no traversal)")
    ps.add_argument("--metadata", required=True, help="path to metadata JSON (must be object)")
    ps.add_argument("--summary", required=True, help="path to summary JSON (must be object)")
    ps.add_argument("--command", required=True, help="path to command text file")
    ps.add_argument("--environment", required=True, help="path to environment text file")
    ps.add_argument("--raw", dest="raw", action="append", nargs="*", default=None, help="raw artifact files (repeatable, or multiple per flag)")
    ps.add_argument("--decoded", dest="decoded", action="append", nargs="*", default=None, help="decoded artifact files")
    ps.add_argument("--validation", dest="validation", action="append", nargs="*", default=None, help="validation artifact files")
    ps.set_defaults(func=cmd_seal_case)

    pm = subs.add_parser("manifest", help="verify cases and emit deterministic MANIFEST.json + MANIFEST.sha256")
    pm.add_argument("--root", required=True, help="corpus root")
    pm.set_defaults(func=cmd_manifest)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:
        error(str(e))


if __name__ == "__main__":
    main()
