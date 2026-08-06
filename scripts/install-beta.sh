#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Vanity entry point for the beta channel: curl -fsSL beta.hipfire.dev | bash
# Always installs/updates from the `beta` branch so users never need to pass
# --branch beta themselves. Equivalent to:
#   curl -fsSL .../scripts/install.sh | bash -s -- --branch beta
# Any extra args are forwarded through to install.sh / hipfire setup.
set -euo pipefail

exec bash -c "$(curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/beta/scripts/install.sh)" -- --branch beta "$@"
