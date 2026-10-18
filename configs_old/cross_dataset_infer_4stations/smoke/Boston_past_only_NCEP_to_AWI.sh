#!/usr/bin/env bash
# One real cross-dataset transfer, isolated from the 264 production pairs.
source "$(dirname "${BASH_SOURCE[0]}")/../Boston/past_only/NCEP/NCEP_to_AWI.sh"
YEARS="2008_2009"
EXPECTED_YEAR_COUNT=1
EXPERIMENT_GROUP="smoke_past_only"
INFERENCE_RESULTS_ROOT="${FOUR_STATION_RESULTS_ROOT}/smoke/Boston"
