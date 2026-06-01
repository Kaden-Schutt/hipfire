#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

# Regression test for issue #376:
# `hipfire update` copied only cli/index.ts + cli/registry.json, so after
# index.ts gained helper imports (`./chat_pure.ts`, `./chat.ts`) installed
# CLIs could be stranded with a new entrypoint and missing sibling modules.

set -euo pipefail

TEST_NAME="update-cli-payload"

fail() { echo "FAIL [$TEST_NAME]: $*" >&2; exit 1; }
pass() { echo "PASS [$TEST_NAME]: $*"; }

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
[ -n "$REPO_ROOT" ] || fail "not in a git checkout"

CLI="$REPO_ROOT/cli/index.ts"
[ -f "$CLI" ] || fail "cli/index.ts not found at $CLI"

# Bootstrap safety: an older installed updater may copy only the new index.ts.
# The new entrypoint must still start so the user can run `hipfire update`
# again and let the fixed updater copy the rest of cli/.
if grep -E '^import .*"\./(chat|chat_pure)\.ts";' "$CLI" >/dev/null; then
    fail "cli/index.ts has a startup import of chat/chat_pure; legacy index-only updates can strand the CLI"
fi

# Future safety: the fixed updater must copy the whole runtime CLI tree, then
# prune dev-only files, matching scripts/install.{sh,ps1}.
grep -q 'function syncCliRuntimePayload' "$CLI" \
    || fail "cli/index.ts is missing syncCliRuntimePayload()"
grep -q 'cpSync(cliSrcDir, tmpDir, { recursive: true' "$CLI" \
    || fail "hipfire update does not recursively copy cli/"
grep -q 'pruneCliRuntimePayload' "$CLI" \
    || fail "hipfire update does not prune dev/test CLI artifacts after copy"

# Forbidden: this is the exact stale-copy pattern that caused #376.
if grep -q 'copyFileSync(indexSrc,.*cli/index\.ts' "$CLI"; then
    fail "hipfire update still copies only cli/index.ts instead of the CLI payload"
fi

pass "update keeps CLI payload dependencies together"
