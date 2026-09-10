#!/usr/bin/env bash
# Matched to Stable Dual; only the named mechanism and artifact directory change.
source "$(dirname "${BASH_SOURCE[0]}")/../train_config_NCEP_Battery_Stable_Dual.sh"
DUAL_ABLATION="no_gate_bce"
ALL_RESULTS_ROOT="./All_Results/Dual_Ablations/no_gate_bce"
