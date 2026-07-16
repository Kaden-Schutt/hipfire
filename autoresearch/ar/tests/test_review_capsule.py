# Copyright (c) Kaden Schutt
import base64
import hashlib
import json

import pytest

from autoresearch.ar.review.capsule import MAX_BLOB_REQUESTS, ReviewCapsuleError, build_review_capsule
from autoresearch.ar.review.models import ReviewTarget


TARGET = ReviewTarget("owner/repo", 42, "fork/repo", "head", "main", "base", "merge")


def git_blob_oid(payload):
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


OLD_OID = git_blob_oid(b"old\n")
NEW_OID = git_blob_oid(b"new\n")
A_OID = git_blob_oid(b"a\n")
B_OID = git_blob_oid(b"b\n")


def response(data):
    return type("Response", (), {"data": data})()


def tree(sha, entries, *, truncated=False):
    return response({"sha": sha, "tree": entries, "truncated": truncated})


def commit(tree_sha):
    return response({"sha": "merge" if tree_sha == "merge-tree" else "head", "tree": {"sha": tree_sha}})


def blob(sha, payload, *, encoding="base64", size=None):
    return response({
        "sha": sha,
        "encoding": encoding,
        "content": base64.b64encode(payload).decode() if encoding == "base64" else payload,
        "size": len(payload) if size is None else size,
    })


class FakeGitHub:
    def __init__(self, trees, blobs):
        self.trees = trees
        self.blobs = blobs
        self.tree_calls = []
        self.blob_calls = []
        self.commit_calls = []

    def get_commit(self, repository, sha):
        self.commit_calls.append((repository, sha))
        return commit("merge-tree" if sha == TARGET.merge_base_sha else "head-tree")

    def get_tree(self, repository, sha, *, recursive=False):
        self.tree_calls.append((repository, sha, recursive))
        return self.trees[sha]

    def get_blob(self, repository, sha):
        self.blob_calls.append((repository, sha))
        return self.blobs[sha]


def test_capsule_uses_merge_base_tree_not_base_tip_and_retrieves_changed_blobs():
    client = FakeGitHub(
        {
            "merge-tree": tree("merge-tree", [{"path": "z.py", "mode": "100644", "type": "blob", "sha": OLD_OID}]),
            "head-tree": tree("head-tree", [
                {"path": "a.py", "mode": "100644", "type": "blob", "sha": A_OID},
                {"path": "z.py", "mode": "100644", "type": "blob", "sha": NEW_OID},
            ]),
        },
        {OLD_OID: blob(OLD_OID, b"old\n"), NEW_OID: blob(NEW_OID, b"new\n"), A_OID: blob(A_OID, b"a\n")},
    )
    capsule = build_review_capsule(client, TARGET)

    assert capsule.complete
    assert [item.path for item in capsule.manifest] == ["a.py", "z.py"]
    assert capsule.manifest[0].base_blob_oid is None
    assert capsule.manifest[0].head_blob_oid == A_OID
    assert capsule.manifest[1].base_blob_oid == OLD_OID
    assert capsule.manifest[1].head_blob_oid == NEW_OID
    assert capsule.files[0].head_source == "a\n"
    assert client.commit_calls == [("owner/repo", "merge"), ("fork/repo", "head")]
    assert client.tree_calls == [("owner/repo", "merge-tree", True), ("fork/repo", "head-tree", True)]
    assert client.blob_calls == [("fork/repo", A_OID), ("owner/repo", OLD_OID), ("fork/repo", NEW_OID)]


def test_capsule_order_and_digest_are_stable_across_api_order():
    entries = [
        {"path": "b.txt", "mode": "100644", "type": "blob", "sha": B_OID},
        {"path": "a.txt", "mode": "100644", "type": "blob", "sha": A_OID},
    ]
    first = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", entries)},
        {A_OID: blob(A_OID, b"a\n"), B_OID: blob(B_OID, b"b\n")},
    )
    second = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", list(reversed(entries)))},
        {A_OID: blob(A_OID, b"a\n"), B_OID: blob(B_OID, b"b\n")},
    )

    left = build_review_capsule(first, TARGET)
    right = build_review_capsule(second, TARGET)
    assert left.digest == right.digest
    assert left.to_mapping() == right.to_mapping()
    assert json.dumps(left.to_mapping(), sort_keys=False) == json.dumps(right.to_mapping(), sort_keys=False)


def test_truncated_tree_is_explicitly_incomplete():
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", [], truncated=True), "head-tree": tree("head-tree", [])}, {}
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("truncat" in reason for reason in capsule.rejections)


def test_directory_entries_are_not_changed_files():
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", [
            {"path": "src", "mode": "040000", "type": "tree", "sha": "old-dir"},
            {"path": "src/a.py", "mode": "100644", "type": "blob", "sha": OLD_OID},
        ]), "head-tree": tree("head-tree", [
            {"path": "src", "mode": "040000", "type": "tree", "sha": "new-dir"},
            {"path": "src/a.py", "mode": "100644", "type": "blob", "sha": NEW_OID},
        ])},
        {OLD_OID: blob(OLD_OID, b"old\n"), NEW_OID: blob(NEW_OID, b"new\n")},
    )
    capsule = build_review_capsule(client, TARGET)
    assert capsule.complete
    assert [item.path for item in capsule.manifest] == ["src/a.py"]


def test_missing_truncated_marker_is_incomplete():
    client = FakeGitHub(
        {"merge-tree": response({"sha": "merge-tree", "tree": []}), "head-tree": tree("head-tree", [])}, {}
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("truncat" in reason for reason in capsule.rejections)


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"\x00binary", "binary"),
        (b"x", "size"),
    ],
)
def test_binary_and_declared_size_rejection(payload, message):
    oid = git_blob_oid(payload)
    blob_data = blob(oid, payload, size=2 if payload == b"x" else None)
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.bin", "mode": "100644", "type": "blob", "sha": oid},
        ])},
        {oid: blob_data},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any(message in reason.lower() for reason in capsule.rejections)


def test_invalid_base64_and_encoding_are_rejected():
    oid = git_blob_oid(b"not-base64")
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.py", "mode": "100644", "type": "blob", "sha": oid},
        ])},
        {oid: response({"sha": oid, "encoding": "utf-8", "content": "not-base64", "size": 3})},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("encoding" in reason or "opaque" in reason for reason in capsule.rejections)


def test_symlink_blob_is_retrieved_but_submodule_is_explicitly_incomplete():
    link_oid = git_blob_oid(b"target")
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "link", "mode": "120000", "type": "blob", "sha": link_oid},
            {"path": "vendor", "mode": "160000", "type": "commit", "sha": "submodule"},
        ])},
        {link_oid: blob(link_oid, b"target")},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert client.blob_calls == [("fork/repo", link_oid)]
    assert {item.path for item in capsule.manifest} == {"link", "vendor"}
    assert any("submodule" in reason or "opaque" in reason or "binary" in reason for reason in capsule.rejections)


def test_blob_sha_mismatch_is_incomplete():
    expected = git_blob_oid(b"x\n")
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.py", "mode": "100644", "type": "blob", "sha": expected},
        ])},
        {expected: blob("returned", b"x\n")},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("identity mismatch" in reason for reason in capsule.rejections)


def test_supported_sha1_oid_must_match_git_blob_object_hash():
    payload = b"x = 1\n"
    expected = git_blob_oid(payload)
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.py", "mode": "100644", "type": "blob", "sha": expected},
        ])},
        {expected: blob(expected, b"different\n")},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("object" in reason or "hash" in reason for reason in capsule.rejections)


def test_non_sha1_blob_oid_is_explicitly_incomplete():
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.py", "mode": "100644", "type": "blob", "sha": "short-oid"},
        ])},
        {"short-oid": blob("short-oid", b"x\n")},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("OID" in reason or "oid" in reason for reason in capsule.rejections)


def test_unsupported_blob_mode_is_retrieved_before_incompleteness():
    payload = b"opaque-mode\n"
    oid = git_blob_oid(payload)
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "mode.bin", "mode": "100640", "type": "blob", "sha": oid},
        ])},
        {oid: blob(oid, payload)},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert client.blob_calls == [("fork/repo", oid)]
    assert any("mode" in reason for reason in capsule.rejections)


def test_canonical_byte_limit_returns_rejected_capsule(monkeypatch):
    monkeypatch.setattr("autoresearch.ar.review.capsule.MAX_CANONICAL_BYTES", 2048)
    large_oid = git_blob_oid(b"x" * 5000)
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "large.py", "mode": "100644", "type": "blob", "sha": large_oid},
        ])},
        {large_oid: blob(large_oid, b"x" * 5000)},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("canonical" in reason for reason in capsule.rejections)
    assert len(capsule.canonical_json()) <= 2048


def test_missing_blob_and_manifest_mismatch_never_claim_complete():
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.py", "mode": "100644", "type": "blob", "sha": "missing"},
            {"path": "x.py", "mode": "100644", "type": "blob", "sha": "other"},
        ])},
        {},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert capsule.rejections


def test_capsule_rejects_oversized_paths_before_blob_fetch():
    path = "x" * 5000
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": path, "mode": "100644", "type": "blob", "sha": "x"},
        ])},
        {"x": blob("x", b"ok\n")},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("path" in reason for reason in capsule.rejections)
    assert client.blob_calls == []


def test_changed_file_cap_stops_before_any_blob_retrieval():
    old_oid = git_blob_oid(b"old\n")
    new_oid = git_blob_oid(b"new\n")
    paths = [f"file-{index}.py" for index in range(4096)]
    client = FakeGitHub(
        {
            "merge-tree": tree("merge-tree", [
                {"path": path, "mode": "100644", "type": "blob", "sha": old_oid} for path in paths
            ]),
            "head-tree": tree("head-tree", [
                {"path": path, "mode": "100644", "type": "blob", "sha": new_oid} for path in paths
            ]),
        },
        {old_oid: blob(old_oid, b"old\n"), new_oid: blob(new_oid, b"new\n")},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert len(capsule.manifest) == 3000
    assert client.blob_calls == []
    assert any("count" in reason or "cap" in reason for reason in capsule.rejections)


def test_total_source_byte_overflow_stops_remaining_blob_retrieval(monkeypatch):
    monkeypatch.setattr("autoresearch.ar.review.capsule.MAX_TOTAL_SOURCE_BYTES", 5)
    payloads = [b"one\n", b"two\n", b"three\n"]
    oids = [git_blob_oid(payload) for payload in payloads]
    paths = [f"file-{index}.py" for index in range(3)]
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": path, "mode": "100644", "type": "blob", "sha": oid}
            for path, oid in zip(paths, oids)
        ])},
        {oid: blob(oid, payload) for oid, payload in zip(oids, payloads)},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert len(client.blob_calls) == 2
    assert client.blob_calls[-1][1] == oids[1]
    assert any("total source bytes" in reason for reason in capsule.rejections)


def test_blob_request_budget_bounds_three_thousand_changed_paths():
    base_entries = []
    head_entries = []
    blobs = {}
    for index in range(3000):
        old_payload = f"old-{index}\n".encode()
        new_payload = f"new-{index}\n".encode()
        old_oid = git_blob_oid(old_payload)
        new_oid = git_blob_oid(new_payload)
        path = f"file-{index}.py"
        base_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": old_oid})
        head_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": new_oid})
        blobs[old_oid] = blob(old_oid, old_payload)
        blobs[new_oid] = blob(new_oid, new_payload)
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", base_entries), "head-tree": tree("head-tree", head_entries)}, blobs
    )

    capsule = build_review_capsule(client, TARGET)

    assert not capsule.complete
    assert len(client.blob_calls) == MAX_BLOB_REQUESTS
    assert any("blob request budget" in reason for reason in capsule.rejections)


def test_repeated_blob_oids_are_fetched_once():
    payload = b"shared\n"
    oid = git_blob_oid(payload)
    paths = [f"file-{index}.py" for index in range(3000)]
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": path, "mode": "100644", "type": "blob", "sha": oid} for path in paths
        ])},
        {oid: blob(oid, payload)},
    )

    capsule = build_review_capsule(client, TARGET)

    assert capsule.complete
    assert client.blob_calls == [("fork/repo", oid)]
