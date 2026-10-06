#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="future_year_Battery_NCEP_to_MPI"
SOURCE_NAME="NCEP"
TARGET_NAME="MPI"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/NCEP_future_transfer/NCEP_Battery_future_transfer_G0_T0500_EP0100__20261005_235201/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/NCEP/graphs"
TARGET_ROOT="./Data/Grid4_New/CMIP6_MPI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
