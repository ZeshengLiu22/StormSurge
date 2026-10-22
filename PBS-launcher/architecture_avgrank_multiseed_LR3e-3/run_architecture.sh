#!/usr/bin/env bash
# Shared implementation keeps both LR groups on the same execution protocol.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../architecture_avgrank_multiseed/run_architecture.sh" LR3e-3 "$@"
