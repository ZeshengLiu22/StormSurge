#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="past_only_Lewes_NCEP_to_MRI"
SOURCE_NAME="NCEP"
TARGET_NAME="MRI"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/NCEP_Lewes_G0_T0500_EP0100__20260927_115826/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/NCEP/graphs"
TARGET_ROOT="./Data/Grid4_New_PastOnly/CMIP6_MRI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
