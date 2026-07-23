# Copyright (c) Kaden Schutt
"""Bounded, canonical source capsules for one pull request target."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import base64
import binascii
import hashlib
import re
from typing import Any

from .canonical import DEFAULT_MAX_BYTES, canonical_digest, canonical_json
from .models import ReviewTarget


MAX_PATH_BYTES = 4096
MAX_CHANGED_PATHS = 3000
MAX_TREE_ENTRIES = 65536
MAX_TOTAL_SOURCE_BYTES = 8 * 1024 * 1024
MAX_BLOB_BYTES = 2 * 1024 * 1024
MAX_TREE_DEPTH = 64
MAX_CANONICAL_BYTES = DEFAULT_MAX_BYTES
_GITHUB_STANDARD_REQUEST_QUOTA = 5000
_CAPSULE_NON_BLOB_REQUESTS = 4
MAX_BLOB_REQUESTS = 4096
assert MAX_BLOB_REQUESTS + _CAPSULE_NON_BLOB_REQUESTS < _GITHUB_STANDARD_REQUEST_QUOTA
_SHA1_OID = re.compile(r"[0-9a-f]{40}")


class ReviewCapsuleError(ValueError):
    """Raised when a capsule cannot be constructed from a trusted boundary."""


@dataclass(frozen=True)
class ReviewManifestEntry:
    path: str
    base_mode: str | None
    head_mode: str | None
    base_blob_oid: str | None
    head_blob_oid: str | None
    base_byte_size: int | None
    head_byte_size: int | None


@dataclass(frozen=True)
class ReviewFile:
    path: str
    base_source: str | None
    head_source: str | None


@dataclass(frozen=True)
class ReviewCapsule:
    target: ReviewTarget
    target_key: str
    merge_base_tree_oid: str
    head_tree_oid: str
    manifest: tuple[ReviewManifestEntry, ...]
    files: tuple[ReviewFile, ...]
    complete: bool
    coverage: tuple[str, ...]
    rejections: tuple[str, ...]
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, ReviewTarget) or self.target_key != self.target.target_key():
            raise ReviewCapsuleError("capsule target binding is invalid")
        expected = canonical_digest(
            {key: value for key, value in self.to_mapping().items() if key != "digest"},
            max_bytes=MAX_CANONICAL_BYTES,
        )
        if self.digest != "sha256:" + expected:
            raise ReviewCapsuleError("capsule digest does not match canonical content")
        if tuple(item.path for item in self.manifest) != tuple(sorted(item.path for item in self.manifest)):
            raise ReviewCapsuleError("capsule manifest is not canonically ordered")
        if self.complete and self.rejections:
            raise ReviewCapsuleError("complete capsule cannot contain rejection reasons")

    def to_mapping(self) -> dict[str, Any]:
        target = {
            "repository": self.target.repository,
            "number": self.target.number,
            "head_repository": self.target.head_repository,
            "head_sha": self.target.head_sha,
            "base_ref": self.target.base_ref,
            "base_sha": self.target.base_sha,
            "merge_base_sha": self.target.merge_base_sha,
        }
        return {
            "schema": "agentic-review/review-capsule-v1",
            "target": target,
            "target_key": self.target_key,
            "merge_base_tree_oid": self.merge_base_tree_oid,
            "head_tree_oid": self.head_tree_oid,
            "manifest": [vars(item) for item in self.manifest],
            "files": [vars(item) for item in self.files],
            "complete": self.complete,
            "coverage": list(self.coverage),
            "rejections": list(self.rejections),
            "digest": self.digest,
        }

    def canonical_json(self) -> bytes:
        return canonical_json(self.to_mapping(), max_bytes=MAX_CANONICAL_BYTES)


def capsule_coverage(capsule: ReviewCapsule) -> dict[str, Any]:
    """Derive the exact protocol coverage evidence from an authenticated capsule."""
    if not isinstance(capsule, ReviewCapsule):
        raise ValueError("coverage requires a typed review capsule")
    expected_file_count = len(capsule.manifest)
    retrieved_file_count = len(capsule.files)
    expected_blob_count = sum(
        int(entry.base_blob_oid is not None) + int(entry.head_blob_oid is not None)
        for entry in capsule.manifest
    )
    retrieved_content_count = sum(
        int(item.base_source is not None) + int(item.head_source is not None)
        for item in capsule.files
    )
    retrieved_blob_count = retrieved_content_count
    expected_content_count = expected_blob_count
    return {
        "retrieved_file_count": retrieved_file_count,
        "expected_file_count": expected_file_count,
        "retrieved_blob_count": retrieved_blob_count,
        "expected_blob_count": expected_blob_count,
        "retrieved_content_count": retrieved_content_count,
        "expected_content_count": expected_content_count,
        "coverage_complete": (
            capsule.complete
            and retrieved_file_count == expected_file_count
            and retrieved_blob_count == expected_blob_count
            and retrieved_content_count == expected_content_count
        ),
    }


def _data(response: Any) -> Mapping[str, Any]:
    value = getattr(response, "data", response)
    if not isinstance(value, Mapping):
        raise ReviewCapsuleError("GitHub response is not an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReviewCapsuleError(f"{name} is missing")
    return value


def _trees(
    client: Any,
    target: ReviewTarget,
    repository: str,
    commit_sha: str,
    label: str,
) -> tuple[str, dict[str, Mapping[str, Any]], list[str]]:
    reasons: list[str] = []
    try:
        commit = _data(client.get_commit(repository, commit_sha))
        if commit.get("sha") != commit_sha:
            reasons.append(f"{label} commit identity mismatch")
        tree = commit.get("tree")
        if not isinstance(tree, Mapping):
            reasons.append(f"{label} commit tree is unavailable")
            return "", {}, reasons
        tree_oid = _text(tree.get("sha"), f"{label} tree OID")
        raw_tree = _data(client.get_tree(repository, tree_oid, recursive=True))
        if raw_tree.get("sha") != tree_oid:
            reasons.append(f"{label} tree identity mismatch")
        if raw_tree.get("truncated") is not False:
            reasons.append(f"{label} recursive tree truncation marker is missing or true")
        entries = raw_tree.get("tree")
        if not isinstance(entries, list):
            reasons.append(f"{label} tree entries are unavailable")
            return tree_oid, {}, reasons
        if len(entries) > MAX_TREE_ENTRIES:
            reasons.append(f"{label} tree exceeds item cap")
            entries = entries[:MAX_TREE_ENTRIES]
        result: dict[str, Mapping[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                reasons.append(f"{label} tree contains a malformed entry")
                continue
            path = entry.get("path")
            if (
                not isinstance(path, str)
                or not path
                or len(path.encode("utf-8", "surrogatepass")) > MAX_PATH_BYTES
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or any(ord(char) < 0x20 for char in path)
            ):
                reasons.append(f"{label} tree contains an invalid path")
                continue
            if len(path.split("/")) > MAX_TREE_DEPTH:
                reasons.append(f"{label} tree path exceeds depth limit: {path}")
                continue
            if path in result:
                reasons.append(f"{label} tree contains duplicate path: {path}")
                continue
            if not all(isinstance(entry.get(field), str) and entry[field] for field in ("mode", "type", "sha")):
                reasons.append(f"{label} tree entry is missing identity: {path}")
                continue
            if entry["type"] == "tree":
                continue
            result[path] = entry
        return tree_oid, result, reasons
    except Exception as exc:
        reasons.append(f"{label} tree unavailable: {type(exc).__name__}")
        return "", {}, reasons


def _blob(
    client: Any,
    target: ReviewTarget,
    repository: str,
    oid: str,
    path: str,
    side: str,
) -> tuple[str | None, int | None, list[str]]:
    reasons: list[str] = []
    try:
        data = _data(client.get_blob(repository, oid))
        if data.get("sha") != oid:
            reasons.append(f"{side} blob identity mismatch: {path}")
            return None, None, reasons
        if _SHA1_OID.fullmatch(oid) is None:
            reasons.append(f"{side} blob OID is unsupported (expected SHA-1): {path}")
            return None, None, reasons
        declared = data.get("size")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            reasons.append(f"{side} blob size is invalid: {path}")
            return None, None, reasons
        if declared > MAX_BLOB_BYTES:
            reasons.append(f"{side} blob exceeds byte cap: {path}")
            return None, declared, reasons
        if data.get("encoding") != "base64" or not isinstance(data.get("content"), str):
            reasons.append(f"{side} blob has opaque or invalid encoding: {path}")
            return None, declared, reasons
        try:
            raw = base64.b64decode(
                data["content"].encode("ascii").replace(b"\n", b"").replace(b"\r", b""),
                validate=True,
            )
        except (UnicodeEncodeError, binascii.Error) as exc:
            reasons.append(f"{side} blob has invalid base64: {path}")
            return None, declared, reasons
        if len(raw) != declared:
            reasons.append(f"{side} blob byte size mismatch: {path}")
            return None, declared, reasons
        actual_oid = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if actual_oid != oid:
            reasons.append(f"{side} blob Git object hash mismatch: {path}")
            return None, declared, reasons
        if b"\x00" in raw:
            reasons.append(f"{side} blob is binary: {path}")
            return None, declared, reasons
        try:
            return raw.decode("utf-8"), declared, reasons
        except UnicodeDecodeError:
            reasons.append(f"{side} blob is binary or opaque: {path}")
            return None, declared, reasons
    except Exception as exc:
        reasons.append(f"{side} blob unavailable: {path} ({type(exc).__name__})")
        return None, None, reasons


def build_review_capsule(client: Any, target: ReviewTarget) -> ReviewCapsule:
    """Compare ``merge_base_sha`` to ``head_sha`` and return a bounded capsule."""
    if not isinstance(target, ReviewTarget):
        raise ReviewCapsuleError("target must be a ReviewTarget")
    base_oid, base_tree, reasons = _trees(client, target, target.repository, target.merge_base_sha, "base")
    head_oid, head_tree, head_reasons = _trees(client, target, target.head_repository, target.head_sha, "head")
    reasons.extend(head_reasons)
    changed_paths = sorted(
        path for path in set(base_tree) | set(head_tree)
        if base_tree.get(path, {}).get("sha") != head_tree.get(path, {}).get("sha")
        or base_tree.get(path, {}).get("mode") != head_tree.get(path, {}).get("mode")
        or base_tree.get(path, {}).get("type") != head_tree.get(path, {}).get("type")
    )
    changed_path_cap_hit = len(changed_paths) > MAX_CHANGED_PATHS
    if changed_path_cap_hit:
        reasons.append("changed path count exceeds item cap")
        changed_paths = changed_paths[:MAX_CHANGED_PATHS]
    blob_keys: set[tuple[str, str]] = set()
    for path in changed_paths:
        base = base_tree.get(path)
        head = head_tree.get(path)
        if base is not None and base.get("type") == "blob":
            blob_keys.add((target.repository, base["sha"]))
        if head is not None and head.get("type") == "blob":
            blob_keys.add((target.head_repository, head["sha"]))
    if len(blob_keys) > MAX_BLOB_REQUESTS:
        reasons.append("blob request budget exceeds fixed capsule limit")
    manifest: list[ReviewManifestEntry] = []
    files: list[ReviewFile] = []
    total_bytes = 0
    blob_cache: dict[tuple[str, str], tuple[str | None, int | None]] = {}
    blob_request_budget_reported = False

    def load_blob(
        repository: str, oid: str, path: str, side: str
    ) -> tuple[str | None, int | None, list[str]]:
        nonlocal blob_request_budget_reported
        key = (repository, oid)
        if key in blob_cache:
            source, size = blob_cache[key]
            return source, size, []
        if len(blob_cache) >= MAX_BLOB_REQUESTS:
            if not blob_request_budget_reported:
                reasons.append("blob request budget exhausted before full capsule coverage")
                blob_request_budget_reported = True
            return None, None, []
        source, size, blob_reasons = _blob(client, target, repository, oid, path, side)
        blob_cache[key] = (source, size)
        return source, size, blob_reasons

    for path in changed_paths:
        base = base_tree.get(path)
        head = head_tree.get(path)
        if changed_path_cap_hit:
            manifest.append(ReviewManifestEntry(
                path,
                base.get("mode") if base else None,
                head.get("mode") if head else None,
                base.get("sha") if base and base.get("type") == "blob" else None,
                head.get("sha") if head and head.get("type") == "blob" else None,
                None,
                None,
            ))
            files.append(ReviewFile(path, None, None))
            continue
        if (base and base.get("type") != "blob") or (head and head.get("type") != "blob"):
            entries = [entry for entry in (base, head) if entry is not None]
            if any(entry.get("type") == "commit" or entry.get("mode") == "160000" for entry in entries):
                reasons.append(f"submodule commit entry is unsupported: {path}")
            else:
                reasons.append(f"unsupported or opaque tree leaf: {path}")
            manifest.append(ReviewManifestEntry(
                path,
                base.get("mode") if base else None,
                head.get("mode") if head else None,
                base.get("sha") if base and base.get("type") == "blob" else None,
                head.get("sha") if head and head.get("type") == "blob" else None,
                None,
                None,
            ))
            files.append(ReviewFile(path, None, None))
            continue
        unsupported_mode = (base and base.get("mode") not in {"100644", "100755", "120000"}) or (
            head and head.get("mode") not in {"100644", "100755", "120000"}
        )
        base_source = head_source = None
        base_size = head_size = None
        if base is not None:
            base_source, base_size, blob_reasons = load_blob(target.repository, base["sha"], path, "base")
            reasons.extend(blob_reasons)
        if head is not None:
            head_source, head_size, blob_reasons = load_blob(target.head_repository, head["sha"], path, "head")
            reasons.extend(blob_reasons)
        if unsupported_mode:
            reasons.append(f"binary or opaque file mode: {path}")
        for size in (base_size, head_size):
            if size is not None:
                total_bytes += size
        if total_bytes > MAX_TOTAL_SOURCE_BYTES:
            reasons.append("total source bytes exceed cap")
            base_source = head_source = None
        manifest.append(ReviewManifestEntry(
            path,
            base.get("mode") if base else None,
            head.get("mode") if head else None,
            base.get("sha") if base else None,
            head.get("sha") if head else None,
            base_size,
            head_size,
        ))
        files.append(ReviewFile(path, base_source, head_source))
        if total_bytes > MAX_TOTAL_SOURCE_BYTES:
            break
    manifest.sort(key=lambda item: item.path)
    files.sort(key=lambda item: item.path)
    complete = not reasons and len(manifest) == len(changed_paths)
    coverage = (
        "merge-base tree compared to head tree",
        f"{len(manifest)} changed paths represented",
        f"{total_bytes} source bytes inspected",
    )
    values = {
        "target": target,
        "target_key": target.target_key(),
        "merge_base_tree_oid": base_oid,
        "head_tree_oid": head_oid,
        "manifest": tuple(manifest),
        "files": tuple(files),
        "complete": complete,
        "coverage": coverage,
        "rejections": tuple(sorted(set(reasons))),
    }
    try:
        digest = "sha256:" + canonical_digest(
            {"schema": "agentic-review/review-capsule-v1", **values}, max_bytes=MAX_CANONICAL_BYTES
        )
    except ValueError:
        # A rejected capsule must remain representable and auditable; do not
        # leak canonical_json's size exception at this trust boundary.
        values = {
            **values,
            "manifest": tuple(values["manifest"]),
            "files": (),
            "complete": False,
            "coverage": ("canonical byte cap prevented full file coverage",),
            "rejections": tuple(sorted(set((*values["rejections"], "canonical capsule byte limit exceeded")))),
        }
        while True:
            try:
                digest = "sha256:" + canonical_digest(
                    {"schema": "agentic-review/review-capsule-v1", **values}, max_bytes=MAX_CANONICAL_BYTES
                )
                break
            except ValueError:
                manifest = values["manifest"]
                if not manifest:
                    values = {**values, "coverage": ("canonical byte cap prevented file manifest coverage",)}
                    digest = "sha256:" + canonical_digest(
                        {"schema": "agentic-review/review-capsule-v1", **values}, max_bytes=MAX_CANONICAL_BYTES
                    )
                    break
                values = {**values, "manifest": manifest[:-1]}
    return ReviewCapsule(digest=digest, **values)
