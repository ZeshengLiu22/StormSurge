#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="future_year_Lewes_NCEP_to_MRI"
SOURCE_NAME="NCEP"
TARGET_NAME="MRI"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/NCEP_future_transfer/NCEP_Lewes_future_transfer_G0_T0500_EP0100__20261006_010226/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/NCEP/graphs"
TARGET_ROOT="./Data/Grid4_New/CMIP6_MRI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
