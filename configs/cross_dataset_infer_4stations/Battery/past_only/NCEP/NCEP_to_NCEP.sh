#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="past_only_Battery_NCEP_to_NCEP"
SOURCE_NAME="NCEP"
TARGET_NAME="NCEP"

CKPT_PATH=/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/NCEP_Battery_G0_T0500_EP0100__20260928_094059/best_overall.pt

SOURCE_ROOT="./Data/Grid4_New/NCEP/graphs"
TARGET_ROOT="./Data/Grid4_New/NCEP/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
