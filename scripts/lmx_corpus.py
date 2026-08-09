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
import io
import subprocess
import sys
import tarfile
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

PACKAGE_SCHEMA = "lmx-package/1"
PACKAGE_SCHEMA_VERSION = "1"


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
# deterministic packaging helpers (pure, testable)
# ---------------------------------------------------------------------------

def _normalize_tarinfo(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalize TarInfo for deterministic archives."""
    ti.mtime = 0
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    # stable modes: directories 0o755, regular files 0o644
    if ti.isdir():
        ti.mode = 0o755
        ti.size = 0
    elif ti.isreg():
        ti.mode = 0o644
    else:
        error(f"unsupported tar entry type (only regular files and directories): {ti.name}")
    # clear pax headers for determinism
    ti.pax_headers = {}
    return ti


def _sanitize_value(value):
    """Recursively sanitize values to avoid absolute paths and credentials.

    Returns sanitized copy or None if value should be omitted.
    Strings that are absolute paths are replaced with their basename.
    Credential-like keys/values are omitted.
    """
    if isinstance(value, str):
        # reject credential-like values
        low = value.lower()
        if "api_key" in low or "credential" in low or "secret" in low or "token" in low:
            # check if looks like credential: contains sk- or key
            return None
        if value.startswith("/") or value.startswith("\\"):
            # absolute path -> use basename to avoid leaking absolute
            base = os.path.basename(value)
            return base if base else None
        # also reject strings that look like absolute paths with \x00 or ..
        if ".." in value and "/" in value:
            # still sanitize: return basename
            return os.path.basename(value)
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            # omit credential-like keys
            kl = k.lower()
            if "credential" in kl or "secret" in kl or "api_key" in kl or "token" in kl:
                continue
            # also omit password
            if "password" in kl:
                continue
            sanitized = _sanitize_value(v)
            # keep sanitized even if None? For dict values that were absolute paths we replaced with basename, so not None
            # If sanitized is None due to credential, omit the key
            if sanitized is None and isinstance(v, str):
                # check if it was credential vs absolute path: absolute path returns basename not None, so None means credential
                # omit
                continue
            out[k] = sanitized
        return out
    if isinstance(value, list):
        out_list = []
        for item in value:
            s = _sanitize_value(item)
            if s is not None:
                out_list.append(s)
            else:
                # if item was absolute path string, _sanitize returns basename not None, so None means credential -> omit
                if isinstance(item, str) and (item.startswith("/") or item.startswith("\\")):
                    base = os.path.basename(item)
                    if base:
                        out_list.append(base)
        return out_list
    # numbers, bools, None remain
    return value


def _build_package_index(
    family: str,
    case_id: str,
    hostname: str,
    machine_identity_hash: str | None,
    source_manifest_hash: str | None,
    case_file_checksums: dict,
    metadata: dict,
    summary: dict,
) -> dict:
    """Build deterministic PACKAGE.json index.

    Contains schema, hostname/machine identity, family/case, source MANIFEST hash,
    case file checksums, and normalized identity/route/metrics pointers without
    inventing missing data (only includes keys present in source).
    """
    # ensure sorted checksums
    checksums_sorted = dict(sorted(case_file_checksums.items())) if case_file_checksums else {}
    index: dict = {
        "schema": PACKAGE_SCHEMA,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "id": case_id,
        "hostname": hostname or "",
        "machine_identity_hash": machine_identity_hash,
        "source_manifest_hash": source_manifest_hash,
        "case_file_checksums": checksums_sorted,
    }
    # normalized identity pointer - only if source has relevant data
    identity = None
    if isinstance(metadata, dict) and metadata:
        candidate_keys = ["host", "hardware", "model", "platform", "binary", "execution", "family", "case_id"]
        present = {k: metadata[k] for k in candidate_keys if k in metadata}
        if present:
            sanitized = _sanitize_value(present)
            # filter empty
            if isinstance(sanitized, dict) and sanitized:
                identity = sanitized
    if identity is not None:
        index["identity"] = identity

    # route pointer
    route = None
    if isinstance(metadata, dict):
        if "route_proof" in metadata:
            route = _sanitize_value(metadata["route_proof"])
        elif "execution" in metadata:
            route = _sanitize_value(metadata["execution"])
        elif "route" in metadata:
            route = _sanitize_value(metadata["route"])
    if route is None and isinstance(summary, dict):
        if "route" in summary:
            route = _sanitize_value(summary["route"])
        elif "runtime_route" in summary:
            route = _sanitize_value(summary["runtime_route"])
    if route is not None:
        # avoid empty dict
        if isinstance(route, dict) and not route:
            pass
        else:
            index["route"] = route

    # metrics pointer
    metrics = None
    if isinstance(metadata, dict) and "measurements" in metadata:
        metrics = _sanitize_value(metadata["measurements"])
    elif isinstance(summary, dict):
        # select known metric keys if present
        metric_keys = [
            "median_tok_s", "median_hip_tok_s", "median_pm4_tok_s",
            "median_speedup", "median_speedup_median", "per_process_hip", "per_process_pm4",
            "per_process_hip_tok_s", "per_process_pm4_tok_s", "per_process_speedup",
            "process_runs", "prompt_md5", "decoded_md5", "decoded_sha256", "valid", "median_speedup_ratio"
        ]
        present_m = {k: summary[k] for k in metric_keys if k in summary}
        if present_m:
            metrics = _sanitize_value(present_m)
    if metrics is not None:
        if isinstance(metrics, dict) and not metrics:
            pass
        else:
            index["metrics"] = metrics

    return index


def _discover_cases(root: str) -> list[tuple[str, str, str]]:
    """Discover cases under ROOT using same logic as manifest (preserving discovery).

    Returns sorted list of (family, case_id, case_path).
    """
    non_family = {"lmx-packages", "raw", "validation"}
    cases: list[tuple[str, str, str]] = []
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
            if os.path.isfile(os.path.join(case_path, "metadata.json")):
                if not any(c[0] == entry and c[1] == sub for c in cases):
                    validate_safe_name(entry, "family")
                    validate_safe_name(sub, "case-id")
                    cases.append((entry, sub, case_path))
    cases.sort(key=lambda x: (x[0], x[1]))
    return cases


def _verify_all_cases(root: str, cases: list[tuple[str, str, str]]) -> None:
    """Verify every case's canonical files, JSON, and checksums. Fail closed."""
    for fam, cid, cpath in cases:
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
        for fname in ["metadata.json", "summary.json"]:
            fpath = os.path.join(cpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception as e:
                error(f"case {fam}/{cid} {fname} invalid JSON: {e}")
            if not isinstance(obj, dict):
                error(f"case {fam}/{cid} {fname} must be a JSON object")
        cs_path = os.path.join(cpath, "checksums.sha256")
        expected: dict[str, str] = {}
        file_order: list[str] = []
        try:
            with open(cs_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if "  " in raw_line:
                        h, _, rel = raw_line.strip().partition("  ")
                        h = h.strip()
                        rel = rel.strip()
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            error(f"case {fam}/{cid} malformed checksums line: {raw_line!r}")
                        h, rel = parts[0], parts[1]
                        if len(parts) > 2:
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
        if file_order != sorted(file_order):
            error(f"case {fam}/{cid} checksums.sha256 not sorted")


def _collect_case_tar_entries(case_dir: str, package_index: dict) -> list[tuple[str, str, bool, bytes | None]]:
    """Collect sorted entries for per-case tar: (arcname, fs_path_or_None, is_dir, data_bytes).

    For regular files, fs_path is the source file; for PACKAGE.json, data_bytes is provided and fs_path is None.
    Directories are marked is_dir True with no data.
    """
    entries: list[tuple[str, str, bool, bytes | None]] = []
    # Ensure case_dir has no symlink
    if os.path.islink(case_dir):
        error(f"case directory is a symlink: {case_dir}")
    # walk case_dir, collect dirs and files sorted
    all_dirs: set[str] = set()
    all_files: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(case_dir, topdown=True, followlinks=False):
        dirnames.sort()
        filenames.sort()
        # validate no symlink dirs
        for d in list(dirnames):
            dp = os.path.join(dirpath, d)
            if os.path.islink(dp):
                error(f"symlink directory not allowed: {dp}")
            # record dir for later adding (we will add dir entries via all_dirs)
            rel_dir = os.path.relpath(dp, case_dir).replace(os.sep, "/")
            all_dirs.add(rel_dir)
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if os.path.islink(fpath):
                error(f"symlink file not allowed: {fpath}")
            rel = os.path.relpath(fpath, case_dir).replace(os.sep, "/")
            # reject absolute or traversal in rel (should not happen)
            if rel.startswith("/") or ".." in rel.split("/"):
                error(f"invalid case file path: {rel}")
            all_files.append((rel, fpath))
    # add directory entries sorted
    for d in sorted(all_dirs):
        # validate dir name safe (no absolute)
        if d.startswith("/") or ".." in d.split("/"):
            error(f"invalid case dir path: {d}")
        entries.append((d, os.path.join(case_dir, d.replace("/", os.sep)), True, None))
    # add file entries sorted by rel
    for rel, fpath in sorted(all_files, key=lambda x: x[0]):
        entries.append((rel, fpath, False, None))
    # add PACKAGE.json at root
    pkg_json_bytes = json.dumps(package_index, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    entries.append(("PACKAGE.json", None, False, pkg_json_bytes))
    # sort all entries by arcname for deterministic tar (dirs and files intermixed deterministically)
    # To ensure deterministic, sort by arcname; dirs already have distinct names, but we must ensure stable order
    # However we have already sorted dirs and files separately; now we need to sort combined
    # The current combined list already has dirs sorted and files sorted but not interleaved sorted; we need final sort
    entries.sort(key=lambda x: x[0])
    return entries


def _write_deterministic_case_tar(case_dir: str, dest_tmp: str, package_index: dict) -> None:
    """Write uncompressed deterministic tar for a case to dest_tmp (temp path).

    Sorted POSIX paths, mtime 0, uid/gid 0, empty owner/group, stable modes, no symlink traversal.
    Archive exactly the case directory contents plus generated PACKAGE.json.
    Atomically fsync.
    """
    entries = _collect_case_tar_entries(case_dir, package_index)
    # Open dest_tmp via fileobj for fsync control
    # Use PAX_FORMAT for determinism
    with open(dest_tmp, "wb") as f:
        with tarfile.open(fileobj=f, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for arcname, fs_path, is_dir, data_bytes in entries:
                # reject unexpected members: ensure arcname does not contain lmx-packages, absolute, etc.
                if arcname.startswith("lmx-packages"):
                    error(f"per-case package must not contain lmx-packages recursively: {arcname}")
                if arcname.startswith("/") or arcname.startswith("\\") or ".." in arcname.split("/"):
                    error(f"per-case package contains absolute or traversal path: {arcname}")
                if "\x00" in arcname:
                    error(f"per-case package contains null byte in path: {arcname!r}")
                ti = tarfile.TarInfo(name=arcname)
                if is_dir:
                    ti.type = tarfile.DIRTYPE
                    ti.size = 0
                    _normalize_tarinfo(ti)
                    tar.addfile(ti)
                else:
                    if data_bytes is not None:
                        # PACKAGE.json
                        ti.size = len(data_bytes)
                        _normalize_tarinfo(ti)
                        tar.addfile(ti, io.BytesIO(data_bytes))
                    else:
                        # regular file
                        assert fs_path is not None
                        # validate file is regular (not symlink already checked)
                        st = os.stat(fs_path)
                        # ensure regular file (not fifo, device)
                        import stat as _stat
                        if not _stat.S_ISREG(st.st_mode):
                            error(f"per-case package only supports regular files: {arcname}")
                        ti.size = st.st_size
                        _normalize_tarinfo(ti)
                        with open(fs_path, "rb") as sf:
                            tar.addfile(ti, sf)
        f.flush()
        os.fsync(f.fileno())


def _write_deterministic_host_tar(root: str, dest_tmp: str) -> None:
    """Write deterministic host archive.

    Includes machine.json, final MANIFEST files, six family trees, and case packages.
    Excludes top-level raw/validation, host-archives itself, non-family/quarantine/staging.
    Sorted POSIX, mtime 0, uid/gid 0, stable modes, no symlink traversal, no absolute paths.
    """
    # Collect entries as (arcname, fs_path, is_dir)
    entries: list[tuple[str, str, bool]] = []

    machine_path = os.path.join(root, "machine.json")
    manifest_path = os.path.join(root, "MANIFEST.json")
    manifest_sha_path = os.path.join(root, "MANIFEST.sha256")

    if os.path.isfile(machine_path):
        if os.path.islink(machine_path):
            error("machine.json is a symlink")
        entries.append(("machine.json", machine_path, False))
    if os.path.isfile(manifest_path):
        if os.path.islink(manifest_path):
            error("MANIFEST.json is a symlink")
        entries.append(("MANIFEST.json", manifest_path, False))
    if os.path.isfile(manifest_sha_path):
        if os.path.islink(manifest_sha_path):
            error("MANIFEST.sha256 is a symlink")
        entries.append(("MANIFEST.sha256", manifest_sha_path, False))

    # six family trees
    for fam in CASE_FAMILIES:
        fam_path = os.path.join(root, fam)
        if not os.path.isdir(fam_path) or os.path.islink(fam_path):
            continue
        # add family dir
        entries.append((fam, fam_path, True))
        for dirpath, dirnames, filenames in os.walk(fam_path, topdown=True, followlinks=False):
            dirnames.sort()
            filenames.sort()
            for d in list(dirnames):
                dp = os.path.join(dirpath, d)
                if os.path.islink(dp):
                    error(f"family directory symlink not allowed: {dp}")
            for d in dirnames:
                dp = os.path.join(dirpath, d)
                rel = os.path.relpath(dp, root).replace(os.sep, "/")
                if rel.startswith("/") or ".." in rel.split("/"):
                    error(f"host archive contains invalid path: {rel}")
                entries.append((rel, dp, True))
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if os.path.islink(fpath):
                    error(f"family file symlink not allowed: {fpath}")
                rel = os.path.relpath(fpath, root).replace(os.sep, "/")
                if rel.startswith("/") or ".." in rel.split("/"):
                    error(f"host archive contains invalid path: {rel}")
                # ensure file is regular
                import stat as _stat
                st = os.stat(fpath)
                if not _stat.S_ISREG(st.st_mode):
                    error(f"host archive only supports regular files: {rel}")
                entries.append((rel, fpath, False))

    # case packages under lmx-packages/cases
    cases_pkg_root = os.path.join(root, "lmx-packages", "cases")
    lmx_pkg_dir = os.path.join(root, "lmx-packages")
    if os.path.isdir(cases_pkg_root) and not os.path.islink(cases_pkg_root):
        # ensure lmx-packages dir entry
        if os.path.isdir(lmx_pkg_dir) and not os.path.islink(lmx_pkg_dir):
            # avoid duplicate if already added? not yet
            if not any(e[0] == "lmx-packages" for e in entries):
                entries.append(("lmx-packages", lmx_pkg_dir, True))
        entries.append(("lmx-packages/cases", cases_pkg_root, True))
        # list families under cases
        try:
            families = os.listdir(cases_pkg_root)
        except Exception as e:
            error(f"cannot list lmx-packages/cases: {e}")
        families = sorted(f for f in families if not f.startswith("."))
        for fam in families:
            if fam not in CASE_FAMILIES:
                # exclude non-family entries under cases (should not happen)
                continue
            fam_pkg_path = os.path.join(cases_pkg_root, fam)
            if os.path.islink(fam_pkg_path):
                error(f"package family dir is symlink: {fam_pkg_path}")
            if not os.path.isdir(fam_pkg_path):
                continue
            # validate archive names stay within CASE_FAMILIES/case IDs
            validate_safe_name(fam, "family")
            arc_fam = f"lmx-packages/cases/{fam}"
            if not any(e[0] == arc_fam for e in entries):
                entries.append((arc_fam, fam_pkg_path, True))
            try:
                pkg_entries = os.listdir(fam_pkg_path)
            except Exception as e:
                error(f"cannot list package family {fam}: {e}")
            pkg_entries.sort()
            for entry in pkg_entries:
                if entry.startswith("."):
                    continue
                fpath = os.path.join(fam_pkg_path, entry)
                if os.path.islink(fpath):
                    error(f"package file is symlink: {fpath}")
                # validate archive names
                if entry.endswith(".tar"):
                    case_id = entry[:-4]
                    validate_safe_name(case_id, "case-id")
                elif entry.endswith(".tar.sha256"):
                    case_id = entry[:-10]
                    validate_safe_name(case_id, "case-id")
                else:
                    error(f"unexpected file in lmx-packages/cases/{fam}: {entry}")
                rel = f"lmx-packages/cases/{fam}/{entry}"
                # ensure no absolute/path traversal
                if rel.startswith("/") or ".." in rel.split("/"):
                    error(f"host archive contains invalid package path: {rel}")
                import stat as _stat
                st = os.stat(fpath)
                if not _stat.S_ISREG(st.st_mode):
                    error(f"host archive only supports regular files: {rel}")
                entries.append((rel, fpath, False))

    # Explicitly exclude: top-level raw, validation, host-archives itself, quarantine, staging
    # We simply never added them, so they are excluded. Ensure we didn't accidentally add them via family walk (families are only CASE_FAMILIES, not raw)
    # Also ensure we never added lmx-packages/host-archives
    # Double-check no entry starts with forbidden prefixes
    forbidden_prefixes = ["raw/", "raw", "validation/", "validation", "lmx-packages/host-archives", "quarantine", "staging"]
    for arcname, _, _ in entries:
        for fp in forbidden_prefixes:
            if arcname == fp or arcname.startswith(fp + "/"):
                error(f"host archive must not include {fp}: {arcname}")
        if arcname.startswith("lmx-packages/host-archives"):
            error(f"host archive cannot include itself: {arcname}")
        if arcname.startswith("/"):
            error(f"host archive contains absolute path: {arcname}")

    # Deduplicate and sort deterministically by arcname
    # Use dict to deduplicate keeping first is_dir True if any
    dedup: dict[str, tuple[str, bool]] = {}
    for arcname, fs_path, is_dir in entries:
        if arcname not in dedup:
            dedup[arcname] = (fs_path, is_dir)
        else:
            # if existing is file and new is dir, keep dir? But should not happen duplicate with different type
            pass
    sorted_entries = sorted(dedup.items(), key=lambda x: x[0])

    with open(dest_tmp, "wb") as f:
        with tarfile.open(fileobj=f, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for arcname, (fs_path, is_dir) in sorted_entries:
                ti = tarfile.TarInfo(name=arcname)
                if is_dir:
                    ti.type = tarfile.DIRTYPE
                    ti.size = 0
                    _normalize_tarinfo(ti)
                    tar.addfile(ti)
                else:
                    st = os.stat(fs_path)
                    import stat as _stat
                    if not _stat.S_ISREG(st.st_mode):
                        error(f"host archive only supports regular files: {arcname}")
                    ti.size = st.st_size
                    _normalize_tarinfo(ti)
                    with open(fs_path, "rb") as sf:
                        tar.addfile(ti, sf)
        f.flush()
        os.fsync(f.fileno())


def _regenerate_manifest(root: str) -> None:
    """Regenerate MANIFEST.json and MANIFEST.sha256 with tightened package_hash scan.

    Package hashes cover lmx-packages/cases only, explicitly excluding host-archives
    to avoid self-referential cycle. Preserves sealed-case discovery logic.
    """
    # Reuse discovery and verification logic from cmd_manifest but with tightened package scan
    cases = _discover_cases(root)
    # verify again (fail closed)
    _verify_all_cases(root, cases)

    machine_path = os.path.join(root, "machine.json")
    machine_hash = None
    if os.path.isfile(machine_path):
        if os.path.islink(machine_path):
            error("machine.json is a symlink")
        machine_hash = sha256_file(machine_path)

    manifest_cases = []
    for fam, cid, cpath in cases:
        with open(os.path.join(cpath, "metadata.json"), "r", encoding="utf-8") as f:
            md = json.load(f)
        with open(os.path.join(cpath, "summary.json"), "r", encoding="utf-8") as f:
            sm = json.load(f)
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

    # tightened package_hashes: only lmx-packages/cases subtree, exclude host-archives
    package_hashes: dict[str, str] = {}
    cases_pkg_dir = os.path.join(root, "lmx-packages", "cases")
    if os.path.isdir(cases_pkg_dir):
        if os.path.islink(cases_pkg_dir):
            error("lmx-packages/cases is a symlink")
        for dirpath, dirnames, filenames in os.walk(cases_pkg_dir, topdown=True, followlinks=False):
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
                # Tighten: only include if under lmx-packages/cases, explicitly exclude host-archives
                if not rel.startswith("lmx-packages/cases/"):
                    continue
                if rel.startswith("lmx-packages/host-archives"):
                    continue
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
    atomic_write_json(manifest_path, manifest_obj)
    h = sha256_file(manifest_path)
    sha_path = os.path.join(root, "MANIFEST.sha256")
    atomic_write_text(sha_path, f"{h}  MANIFEST.json\n")


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

    # top-level package hashes: deterministic, tightened to lmx-packages/cases only
    # Explicitly exclude host-archives to avoid self-referential cycle
    package_hashes: dict[str, str] = {}
    cases_pkg_dir = os.path.join(root, "lmx-packages", "cases")
    if os.path.isdir(cases_pkg_dir):
        if os.path.islink(cases_pkg_dir):
            error("lmx-packages/cases is a symlink")
        for dirpath, dirnames, filenames in os.walk(cases_pkg_dir, topdown=True, followlinks=False):
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
                if not rel.startswith("lmx-packages/cases/"):
                    continue
                if rel.startswith("lmx-packages/host-archives"):
                    continue
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

def cmd_package_cases(args) -> None:
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        error(f"root does not exist or is not a directory: {root}")
    if os.path.islink(root):
        error(f"root is a symlink (refusing to follow): {root}")

    cases = _discover_cases(root)
    if not cases:
        error("no sealed cases found (nothing to package)")
    _verify_all_cases(root, cases)

    manifest_path = os.path.join(root, "MANIFEST.json")
    source_manifest_hash = None
    if os.path.isfile(manifest_path):
        if os.path.islink(manifest_path):
            error("MANIFEST.json is a symlink")
        source_manifest_hash = sha256_file(manifest_path)

    machine_path = os.path.join(root, "machine.json")
    hostname = ""
    machine_hash = None
    if os.path.isfile(machine_path):
        if os.path.islink(machine_path):
            error("machine.json is a symlink")
        machine_hash = sha256_file(machine_path)
        try:
            with open(machine_path, "r", encoding="utf-8") as f:
                mj = json.load(f)
                hostname = mj.get("hostname") or ""
                if not hostname:
                    uname = mj.get("uname")
                    if isinstance(uname, dict):
                        hostname = uname.get("node") or ""
                if not hostname:
                    hostname = mj.get("uname_str", "").split()[1] if mj.get("uname_str") else ""
        except Exception:
            hostname = ""
    if not hostname:
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "unknown"
    hostname = hostname.split(".")[0] if hostname else "unknown"

    temp_entries: list[tuple[str, str, str, str]] = []
    try:
        for fam, cid, cpath in sorted(cases, key=lambda x: (x[0], x[1])):
            if fam not in CASE_FAMILIES:
                error(f"case family not in CASE_FAMILIES: {fam}")
            validate_safe_name(fam, "family")
            validate_safe_name(cid, "case-id")

            checksums = compute_case_checksums(cpath)
            checksums_dict = dict(checksums)

            metadata_obj: dict = {}
            summary_obj: dict = {}
            try:
                with open(os.path.join(cpath, "metadata.json"), "r", encoding="utf-8") as f:
                    metadata_obj = json.load(f)
                    if not isinstance(metadata_obj, dict):
                        metadata_obj = {}
            except Exception:
                metadata_obj = {}
            try:
                with open(os.path.join(cpath, "summary.json"), "r", encoding="utf-8") as f:
                    summary_obj = json.load(f)
                    if not isinstance(summary_obj, dict):
                        summary_obj = {}
            except Exception:
                summary_obj = {}

            package_index = _build_package_index(
                family=fam,
                case_id=cid,
                hostname=hostname,
                machine_identity_hash=machine_hash,
                source_manifest_hash=source_manifest_hash,
                case_file_checksums=checksums_dict,
                metadata=metadata_obj,
                summary=summary_obj,
            )

            pkg_dir = os.path.join(root, "lmx-packages", "cases", fam)
            if os.path.lexists(pkg_dir) and os.path.islink(pkg_dir):
                error(f"package directory is a symlink: {pkg_dir}")
            os.makedirs(pkg_dir, exist_ok=True)
            if os.path.islink(pkg_dir):
                error(f"package directory is a symlink after creation: {pkg_dir}")

            final_tar = os.path.join(pkg_dir, f"{cid}.tar")
            final_sha = final_tar + ".sha256"
            if os.path.lexists(final_tar) and os.path.islink(final_tar):
                error(f"destination is a symlink (refusing to follow): {final_tar}")
            if os.path.lexists(final_sha) and os.path.islink(final_sha):
                error(f"destination is a symlink (refusing to follow): {final_sha}")

            fd, tmp_tar = tempfile.mkstemp(dir=pkg_dir, prefix=f".{cid}.tar.tmp.")
            os.close(fd)
            _write_deterministic_case_tar(cpath, tmp_tar, package_index)

            tar_hash = sha256_file(tmp_tar)
            fd2, tmp_sha = tempfile.mkstemp(dir=pkg_dir, prefix=f".{cid}.tar.sha256.tmp.")
            try:
                with os.fdopen(fd2, "w", encoding="utf-8", newline="\n") as f:
                    f.write(f"{tar_hash}  lmx-packages/cases/{fam}/{cid}.tar\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                try:
                    os.unlink(tmp_tar)
                except Exception:
                    pass
                try:
                    os.unlink(tmp_sha)
                except Exception:
                    pass
                raise

            temp_entries.append((final_tar, final_sha, tmp_tar, tmp_sha))

        for final_tar, final_sha, tmp_tar, tmp_sha in temp_entries:
            os.replace(tmp_tar, final_tar)
            try:
                dfd = os.open(os.path.dirname(final_tar), os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except Exception:
                pass
            os.replace(tmp_sha, final_sha)
            try:
                dfd = os.open(os.path.dirname(final_sha), os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except Exception:
                pass

        _regenerate_manifest(root)

    except SystemExit:
        for _, _, tmp_tar, tmp_sha in temp_entries:
            try:
                if os.path.exists(tmp_tar):
                    os.unlink(tmp_tar)
            except Exception:
                pass
            try:
                if os.path.exists(tmp_sha):
                    os.unlink(tmp_sha)
            except Exception:
                pass
        raise
    except Exception as e:
        for _, _, tmp_tar, tmp_sha in temp_entries:
            try:
                if os.path.exists(tmp_tar):
                    os.unlink(tmp_tar)
            except Exception:
                pass
            try:
                if os.path.exists(tmp_sha):
                    os.unlink(tmp_sha)
            except Exception:
                pass
        if isinstance(e, SystemExit):
            raise
        error(str(e))


def cmd_package_host(args) -> None:
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        error(f"root does not exist or is not a directory: {root}")
    if os.path.islink(root):
        error(f"root is a symlink (refusing to follow): {root}")

    manifest_path = os.path.join(root, "MANIFEST.json")
    manifest_sha_path = os.path.join(root, "MANIFEST.sha256")
    if not os.path.isfile(manifest_path):
        error("MANIFEST.json missing -- run manifest or package-cases first")
    if os.path.islink(manifest_path):
        error("MANIFEST.json is a symlink")
    if not os.path.isfile(manifest_sha_path):
        error("MANIFEST.sha256 missing")
    if os.path.islink(manifest_sha_path):
        error("MANIFEST.sha256 is a symlink")
    try:
        with open(manifest_sha_path, "r", encoding="utf-8") as f:
            line = f.read().strip()
            expected = line.split()[0] if line else ""
    except Exception as e:
        error(f"cannot read MANIFEST.sha256: {e}")
    actual = sha256_file(manifest_path)
    if actual.lower() != expected.lower():
        error(f"MANIFEST.sha256 mismatch: expected {expected}, got {actual}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_obj = json.load(f)
    except Exception as e:
        error(f"MANIFEST.json invalid JSON: {e}")
    if not isinstance(manifest_obj, dict):
        error("MANIFEST.json must be a JSON object")

    cases = _discover_cases(root)
    if not cases:
        error("no sealed cases found")
    _verify_all_cases(root, cases)

    for fam, cid, _ in sorted(cases, key=lambda x: (x[0], x[1])):
        pkg_tar = os.path.join(root, "lmx-packages", "cases", fam, f"{cid}.tar")
        pkg_sha = pkg_tar + ".sha256"
        if not os.path.isfile(pkg_tar):
            error(f"case package missing: lmx-packages/cases/{fam}/{cid}.tar")
        if os.path.islink(pkg_tar):
            error(f"case package is a symlink: {pkg_tar}")
        if not os.path.isfile(pkg_sha):
            error(f"case package sidecar missing: lmx-packages/cases/{fam}/{cid}.tar.sha256")
        if os.path.islink(pkg_sha):
            error(f"case package sidecar is a symlink: {pkg_sha}")
        try:
            with open(pkg_sha, "r", encoding="utf-8") as f:
                line = f.read().strip()
                if not line:
                    error(f"case package sidecar empty: {pkg_sha}")
                parts = line.split()
                expected_hash = parts[0]
                if len(expected_hash) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_hash):
                    error(f"case package sidecar invalid hash: {expected_hash!r}")
                if len(parts) >= 2:
                    path_part = parts[-1]
                    if path_part != f"lmx-packages/cases/{fam}/{cid}.tar":
                        if path_part not in (f"{fam}/{cid}.tar", f"lmx-packages/cases/{fam}/{cid}.tar"):
                            error(f"case package sidecar path mismatch for {fam}/{cid}: {path_part!r}")
        except SystemExit:
            raise
        except Exception as e:
            error(f"cannot read case package sidecar {fam}/{cid}: {e}")
        actual_hash = sha256_file(pkg_tar)
        if actual_hash.lower() != expected_hash.lower():
            error(f"case package sidecar mismatch for {fam}/{cid}: expected {expected_hash}, got {actual_hash}")
        try:
            with tarfile.open(pkg_tar, "r") as tar:
                has_package_json = False
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in member.name.split("/"):
                        error(f"package {fam}/{cid} contains absolute or traversal path: {member.name}")
                    if member.issym() or member.islnk():
                        error(f"package {fam}/{cid} contains symlink: {member.name}")
                    if not (member.isreg() or member.isdir()):
                        error(f"package {fam}/{cid} contains non-regular member: {member.name}")
                    if member.name.startswith("lmx-packages"):
                        error(f"package {fam}/{cid} contains lmx-packages recursively: {member.name}")
                    if member.name == "PACKAGE.json":
                        has_package_json = True
                if not has_package_json:
                    error(f"package {fam}/{cid} missing PACKAGE.json")
        except SystemExit:
            raise
        except tarfile.TarError as e:
            error(f"case package {fam}/{cid} is corrupt tar: {e}")
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            error(f"case package {fam}/{cid} verification failed: {e}")

    machine_path = os.path.join(root, "machine.json")
    hostname = ""
    if os.path.isfile(machine_path) and not os.path.islink(machine_path):
        try:
            with open(machine_path, "r", encoding="utf-8") as f:
                mj = json.load(f)
                hostname = mj.get("hostname") or ""
                if not hostname:
                    uname = mj.get("uname")
                    if isinstance(uname, dict):
                        hostname = uname.get("node") or ""
        except Exception:
            hostname = ""
    if not hostname:
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "unknown"
    hostname = hostname.split(".")[0] if hostname else "unknown"
    safe_hostname = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in hostname)
    if not safe_hostname:
        safe_hostname = "unknown"
    validate_safe_name(safe_hostname, "hostname")

    manifest_hash = sha256_file(manifest_path)
    manifest_sha12 = manifest_hash[:12]

    host_dir = os.path.join(root, "lmx-packages", "host-archives")
    if os.path.lexists(host_dir) and os.path.islink(host_dir):
        error(f"host-archives is a symlink: {host_dir}")
    os.makedirs(host_dir, exist_ok=True)
    if os.path.islink(host_dir):
        error(f"host-archives is a symlink after creation: {host_dir}")

    final_name = f"lmx-corpus-{safe_hostname}-{manifest_sha12}.tar"
    validate_safe_name(final_name, "host-archive")
    final_tar = os.path.join(host_dir, final_name)
    final_sha = final_tar + ".sha256"
    if os.path.lexists(final_tar) and os.path.islink(final_tar):
        error(f"destination is a symlink (refusing to follow): {final_tar}")
    if os.path.lexists(final_sha) and os.path.islink(final_sha):
        error(f"destination is a symlink (refusing to follow): {final_sha}")

    fd, tmp_tar = tempfile.mkstemp(dir=host_dir, prefix=f".{final_name}.tmp.")
    os.close(fd)
    tmp_sha = None
    try:
        _write_deterministic_host_tar(root, tmp_tar)
        tar_hash = sha256_file(tmp_tar)
        fd2, tmp_sha = tempfile.mkstemp(dir=host_dir, prefix=f".{final_name}.sha256.tmp.")
        try:
            with os.fdopen(fd2, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"{tar_hash}  lmx-packages/host-archives/{final_name}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                if tmp_sha and os.path.exists(tmp_sha):
                    os.unlink(tmp_sha)
            except Exception:
                pass
            raise
        os.replace(tmp_tar, final_tar)
        try:
            dfd = os.open(host_dir, os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass
        os.replace(tmp_sha, final_sha)
        try:
            dfd = os.open(host_dir, os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass
    except SystemExit:
        try:
            if os.path.exists(tmp_tar):
                os.unlink(tmp_tar)
        except Exception:
            pass
        try:
            if tmp_sha and os.path.exists(tmp_sha):
                os.unlink(tmp_sha)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            if os.path.exists(tmp_tar):
                os.unlink(tmp_tar)
        except Exception:
            pass
        try:
            if tmp_sha and os.path.exists(tmp_sha):
                os.unlink(tmp_sha)
        except Exception:
            pass
        if isinstance(e, SystemExit):
            raise
        error(str(e))


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

    pc = subs.add_parser("package-cases", help="verify cases, write deterministic per-case packages, and regenerate MANIFEST")
    pc.add_argument("--root", required=True, help="corpus root")
    pc.set_defaults(func=cmd_package_cases)

    ph = subs.add_parser("package-host", help="verify MANIFEST and case packages, write deterministic host archive")
    ph.add_argument("--root", required=True, help="corpus root")
    ph.set_defaults(func=cmd_package_host)

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
