#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

EXPERIMENT_GROUP="past_only"
YEARS="2008_2009,2009_2010,2010_2011,2011_2012,2012_2013,2013_2014,2014_2015"
EXPECTED_YEAR_COUNT=7
INFERENCE_RESULTS_ROOT="/home/exouser/media/share/PACT/FormalRuns_0925/CrossDatasetInference_4Stations/Boston/past_only"
