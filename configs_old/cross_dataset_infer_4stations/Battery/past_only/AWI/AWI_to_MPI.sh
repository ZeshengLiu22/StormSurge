#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

NAME="past_only_Battery_AWI_to_MPI"
SOURCE_NAME="AWI"
TARGET_NAME="MPI"

CKPT_PATH=/glade/u/home/zel/scratch/PACT-Derecho/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/past_only/CMIP6_AWI_Battery_past_only_G0_T0500_EP0100__20261003_051753/best_overall.pt

SOURCE_ROOT="/glade/u/home/zel/scratch/PACT-Derecho/PACT/Data/Grid4_New_PastOnly/CMIP6_AWI/graphs"
TARGET_ROOT="/glade/u/home/zel/scratch/PACT-Derecho/PACT/Data/Grid4_New_PastOnly/CMIP6_MPI/graphs"

ROOT_DIR="${SOURCE_ROOT}"
TEST_ROOT_DIR="${TARGET_ROOT}"
