#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="AWI_to_MRI_past_only"
SOURCE_NAME="AWI"
TARGET_NAME="MRI"

CKPT_PATH=/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_AWI_Boston_past_only_G0_T0500_EP0100__20260929_054511/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New_PastOnly/CMIP6_AWI/graphs"
TARGET_ROOT="./Data/Grid4_New_PastOnly/CMIP6_MRI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
