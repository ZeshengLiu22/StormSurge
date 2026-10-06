#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="past_only_Boston_NCEP_to_AWI"
SOURCE_NAME="NCEP"
TARGET_NAME="AWI"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/NCEP_Boston_G0_T0500_EP0100__20260927_191110/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/NCEP/graphs"
TARGET_ROOT="./Data/Grid4_New_PastOnly/CMIP6_AWI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
