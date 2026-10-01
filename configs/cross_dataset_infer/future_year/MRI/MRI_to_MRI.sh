#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="MRI_to_MRI_future_year"
SOURCE_NAME="MRI"
TARGET_NAME="MRI"

CKPT_PATH=/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MRI_Boston_future_year_G0_T0500_EP0100__20260930_142355/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/CMIP6_MRI/graphs"
TARGET_ROOT="./Data/Grid4_New/CMIP6_MRI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
