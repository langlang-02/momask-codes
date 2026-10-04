#!/usr/bin/env bash
# Compatibility wrapper. Prefer g1_visualize/reconstruct_g1_local.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/reconstruct_g1_local.sh" "$@"
