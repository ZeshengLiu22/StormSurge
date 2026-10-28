#!/usr/bin/env bash
# Generated assignment-only config; station_runner.py consumes the pinned manifest.
MANIFEST=/glade/u/home/zel/scratch/StormSurge/PBS-launcher/cmip6_inference/manifest.json
RUN_ID=CMIP6_MIROC/Battery/seed42/PACT_Peak
CMIP6=CMIP6_MIROC
STATION=Battery
SEED=42
MODEL_LABEL=PACT_Peak
MODEL=perceiver3
CKPT=/glade/derecho/scratch/zel/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/cmip6/CMIP6_MIROC/Battery/seed42/PACT_Peak/20261024_114446_CMIP6_MIROC_Battery_seed42_PACT_Peak_36H_lr5e-3_clip0.25/best_overall.pt
ROOT_DIR=/glade/u/home/zel/scratch/PACT-Data-Final/CMIP6_MIROC_PastOnly/graphs
TEST_ROOT_DIR=/glade/u/home/zel/scratch/StormSurge/dataset/CMIP6_MIROC/graphs
OUT_DIR=/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/cmip6_inference/CMIP6_MIROC/Battery/seed42/PACT_Peak/outputs
YEARS=2070_2071,2071_2072,2072_2073,2073_2074,2074_2075,2075_2076,2076_2077,2077_2078,2078_2079,2079_2080,2080_2081,2081_2082,2082_2083,2083_2084,2084_2085,2085_2086,2086_2087,2087_2088,2088_2089,2089_2090,2090_2091,2091_2092,2092_2093,2093_2094,2094_2095,2095_2096,2096_2097,2097_2098,2098_2099,2099_2100
STRICT_YEARS=1
SAVE_NPZ=1
