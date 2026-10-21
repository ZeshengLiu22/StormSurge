#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /glade/work/zel/conda-envs/derecho-ml/bin/python "${SCRIPT_DIR}/sweep.py" "$@"
