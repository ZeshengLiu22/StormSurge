#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="past_only_Lewes_MRI_to_EC_EARTH"
SOURCE_NAME="MRI"
TARGET_NAME="EC_EARTH"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Lewes_QuickCheck_0929/past_only/CMIP6_MRI_Lewes_past_only_G0_T0500_EP0100__20261003_051753/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New_PastOnly/CMIP6_MRI/graphs"
TARGET_ROOT="./Data/Grid4_New_PastOnly/CMIP6_EC_EARTH/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
