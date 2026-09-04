#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Fake gh for tests: records calls and answers pr view, api comments, pr review, label, merges."""

import json
import os
import sys
from pathlib import Path

def _log(args):
    log_path = os.environ.get("FAKE_GH_LOG")
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                json.dump({"args": args}, f)
                f.write("\n")
        except Exception:
            pass

def _comments_path():
    return os.environ.get("FAKE_GH_COMMENTS", "/tmp/fake-gh-comments.json")

def _load_comments():
    p = _comments_path()
    if Path(p).is_file():
        try:
            return json.loads(Path(p).read_text())
        except Exception:
            return []
    return []

def _save_comments(comments):
    p = _comments_path()
    try:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(json.dumps(comments))
    except Exception:
        pass

def main():
    args = sys.argv[1:]
    _log(args)

    # pr view
    if len(args) >= 2 and args[0] == "pr" and args[1] == "view":
        override = os.environ.get("FAKE_GH_PR_VIEW")
        if override and Path(override).is_file():
            sys.stdout.write(Path(override).read_text())
        else:
            sys.stdout.write(json.dumps({
                "title": "Test PR",
                "body": "Test body from author\n<!-- hw-gate-request -->\n```json\n{\"routes\":[],\"claim\":\"test claim\"}\n```",
                "author": {"login": "testuser"},
                "url": "https://github.com/o/r/pull/1"
            }))
        sys.exit(0)

    # api merges
    if args and args[0] == "api" and len(args) > 1 and "merges" in args[1]:
        if os.environ.get("FAKE_GH_MERGE_409") == "1":
            sys.stderr.write("409 Conflict: merge conflict or already merged\n")
            sys.exit(1)
        # success
        # parse base/head for logging
        sys.stdout.write(json.dumps({"sha": "abc123mergedsha", "commit": {"message": "merged"}}))
        sys.exit(0)

    # api repos/.../issues/.../comments etc
    if args and args[0] == "api" and "issues" in args[1] if len(args) > 1 else False:
        endpoint = args[1] if len(args) > 1 else ""
        method = "GET"
        for i, a in enumerate(args):
            if a == "--method" and i + 1 < len(args):
                method = args[i+1]
        has_paginate = "--paginate" in args
        if has_paginate and method == "GET" and "comments" in endpoint and "labels" not in endpoint:
            comments = _load_comments()
            sys.stdout.write(json.dumps(comments))
            sys.exit(0)
        if method == "POST" and "comments" in endpoint and "labels" not in endpoint:
            body = ""
            for a in args:
                if a.startswith("body="):
                    body = a[len("body="):]
            comments = _load_comments()
            new_id = 1000 + len(comments) + 1
            new_comment = {"id": new_id, "body": body, "html_url": f"https://github.com/o/r/pull/1#issuecomment-{new_id}"}
            comments.append(new_comment)
            _save_comments(comments)
            sys.stdout.write(json.dumps(new_comment))
            sys.exit(0)
        if method == "PATCH" and "issues/comments/" in endpoint:
            body = ""
            for a in args:
                if a.startswith("body="):
                    body = a[len("body="):]
            try:
                cid = int(endpoint.rstrip("/").split("/")[-1])
            except Exception:
                cid = None
            comments = _load_comments()
            for c in comments:
                if c.get("id") == cid:
                    c["body"] = body
                    _save_comments(comments)
                    sys.stdout.write(json.dumps(c))
                    sys.exit(0)
            sys.stdout.write(json.dumps({"id": cid, "body": body, "html_url": f"https://github.com/o/r/pull/1#issuecomment-{cid}"}))
            sys.exit(0)
        if method == "POST" and "labels" in endpoint:
            sys.stdout.write(json.dumps({}))
            sys.exit(0)
        if method == "DELETE" and "labels" in endpoint:
            sys.stdout.write("")
            sys.exit(0)
        sys.stdout.write(json.dumps({}))
        sys.exit(0)

    # pr review
    if len(args) >= 2 and args[0] == "pr" and args[1] == "review":
        sys.stdout.write("https://github.com/o/r/pull/1#pullrequestreview-1")
        sys.exit(0)

    # label create
    if len(args) >= 1 and args[0] == "label":
        sys.stdout.write("")
        sys.exit(0)

    sys.stdout.write("")
    sys.exit(0)

if __name__ == "__main__":
    main()
