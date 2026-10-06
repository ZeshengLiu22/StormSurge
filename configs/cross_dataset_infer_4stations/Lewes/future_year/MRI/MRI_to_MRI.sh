#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="future_year_Lewes_MRI_to_MRI"
SOURCE_NAME="MRI"
TARGET_NAME="MRI"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Lewes_QuickCheck_0929/future_year/CMIP6_MRI_Lewes_future_year_G0_T0500_EP0100__20261006_051633/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/CMIP6_MRI/graphs"
TARGET_ROOT="./Data/Grid4_New/CMIP6_MRI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
