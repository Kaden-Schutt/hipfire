#!/usr/bin/env bash

set -eu

cd "$(dirname "$0")/.."

scripts_dir="$(cd "$(dirname "$0")" && pwd)"
python3 "${scripts_dir}/gen-env-docs.py"

echo "regen-env-vars-doc: regenerated docs/env-vars.md and crates/hipfire-runtime/src/env_docs.rs"
