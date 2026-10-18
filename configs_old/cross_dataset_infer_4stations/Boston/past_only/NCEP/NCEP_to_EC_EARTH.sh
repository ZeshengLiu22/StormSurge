#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="past_only_Boston_NCEP_to_EC_EARTH"
SOURCE_NAME="NCEP"
TARGET_NAME="EC_EARTH"

CKPT_PATH=/glade/u/home/zel/scratch/PACT-Derecho/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/NCEP_Boston_G0_T0500_EP0100__20260927_191110/best_overall.pt

SOURCE_ROOT="/glade/u/home/zel/scratch/PACT-Derecho/PACT/Data/Grid4_New/NCEP/graphs"
TARGET_ROOT="/glade/u/home/zel/scratch/PACT-Derecho/PACT/Data/Grid4_New_PastOnly/CMIP6_EC_EARTH/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
