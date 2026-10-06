#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="future_year_Boston_AWI_to_EC_EARTH"
SOURCE_NAME="AWI"
TARGET_NAME="EC_EARTH"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_AWI_Boston_future_year_G0_T0500_EP0100__20260929_054511/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/CMIP6_AWI/graphs"
TARGET_ROOT="./Data/Grid4_New/CMIP6_EC_EARTH/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
