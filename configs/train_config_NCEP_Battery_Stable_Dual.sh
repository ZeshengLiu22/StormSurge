#!/usr/bin/env bash
# Matched body/exceedance head: only the prediction head/supervision differ.
source "$(dirname "${BASH_SOURCE[0]}")/train_config_NCEP_Battery_Stable_Single.sh"
HEAD_TYPE="dual"
DUAL_MODE="exceedance"
