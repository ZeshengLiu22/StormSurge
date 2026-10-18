#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="NCEP_to_MRI_future_year"
SOURCE_NAME="NCEP"
TARGET_NAME="MRI"

CKPT_PATH=/media/share/PACT/FormalRuns_0925/NCEP_future_transfer/NCEP_Boston_future_transfer_G0_T0500_EP0100__20261001_165726/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/NCEP/graphs"
TARGET_ROOT="./Data/Grid4_New/CMIP6_MRI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
