#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="future_year_Boston_EC_EARTH_to_MPI"
SOURCE_NAME="EC_EARTH"
TARGET_NAME="MPI"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0100__20260930_142355/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/CMIP6_EC_EARTH/graphs"
TARGET_ROOT="./Data/Grid4_New/CMIP6_MPI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
