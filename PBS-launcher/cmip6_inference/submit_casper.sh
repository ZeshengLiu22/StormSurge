#!/usr/bin/env bash
set -euo pipefail
runner_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-/glade/work/zel/conda-envs/derecho-ml/bin/python}" "$runner_dir/submit.py" casper --manifest "$runner_dir/manifest.json" "$@"
