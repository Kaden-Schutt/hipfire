# Copyright (c) Kaden Schutt
import base64
import json

import pytest

from autoresearch.ar.review.capsule import ReviewCapsuleError, build_review_capsule
from autoresearch.ar.review.models import ReviewTarget


TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head", "main", "base", "merge")


def response(data):
    return type("Response", (), {"data": data})()


def tree(sha, entries, *, truncated=False):
    return response({"sha": sha, "tree": entries, "truncated": truncated})


def commit(tree_sha):
    return response({"sha": "merge" if tree_sha == "merge-tree" else "head", "commit": {"tree": {"sha": tree_sha}}})


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

    def get_commit(self, repository, sha):
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
            "merge-tree": tree("merge-tree", [{"path": "z.py", "mode": "100644", "type": "blob", "sha": "z0"}]),
            "head-tree": tree("head-tree", [
                {"path": "a.py", "mode": "100644", "type": "blob", "sha": "a1"},
                {"path": "z.py", "mode": "100644", "type": "blob", "sha": "z1"},
            ]),
        },
        {"z0": blob("z0", b"old\n"), "z1": blob("z1", b"new\n"), "a1": blob("a1", b"a\n")},
    )
    capsule = build_review_capsule(client, TARGET)

    assert capsule.complete
    assert [item.path for item in capsule.manifest] == ["a.py", "z.py"]
    assert capsule.manifest[0].base_blob_oid is None
    assert capsule.manifest[0].head_blob_oid == "a1"
    assert capsule.manifest[1].base_blob_oid == "z0"
    assert capsule.manifest[1].head_blob_oid == "z1"
    assert capsule.files[0].head_source == "a\n"
    assert client.tree_calls == [("owner/repo", "merge-tree", True), ("owner/repo", "head-tree", True)]


def test_capsule_order_and_digest_are_stable_across_api_order():
    entries = [
        {"path": "b.txt", "mode": "100644", "type": "blob", "sha": "b"},
        {"path": "a.txt", "mode": "100644", "type": "blob", "sha": "a"},
    ]
    first = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", entries)},
        {"a": blob("a", b"a\n"), "b": blob("b", b"b\n")},
    )
    second = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", list(reversed(entries)))},
        {"a": blob("a", b"a\n"), "b": blob("b", b"b\n")},
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


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"\x00binary", "binary"),
        (b"x", "size"),
    ],
)
def test_binary_and_declared_size_rejection(payload, message):
    blob_data = blob("x", payload, size=2 if payload == b"x" else None)
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.bin", "mode": "100644", "type": "blob", "sha": "x"},
        ])},
        {"x": blob_data},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any(message in reason.lower() for reason in capsule.rejections)


def test_invalid_base64_and_encoding_are_rejected():
    client = FakeGitHub(
        {"merge-tree": tree("merge-tree", []), "head-tree": tree("head-tree", [
            {"path": "x.py", "mode": "100644", "type": "blob", "sha": "x"},
        ])},
        {"x": response({"sha": "x", "encoding": "utf-8", "content": "not-base64", "size": 3})},
    )
    capsule = build_review_capsule(client, TARGET)
    assert not capsule.complete
    assert any("encoding" in reason or "opaque" in reason for reason in capsule.rejections)


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
