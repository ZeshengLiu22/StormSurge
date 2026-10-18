# Run table

Config paths are relative to `/home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929`.

Result paths are relative to `/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929`.

The original 20 runs are followed by 20 additional Tail Only / Peak Only configs.
Use `qsub_ablations.txt` to submit the additional runs manually.

| Group | Dataset | Loss variant | Config path | Result path |
| --- | --- | --- | --- | --- |
| past_only | CMIP6_AWI | T0000_EP0000 | past_only/train_config_CMIP6_AWI_Boston_past_only_G0_T0000_EP0000.sh | past_only/CMIP6_AWI_Boston_past_only_G0_T0000_EP0000__20260929_054511 |
| past_only | CMIP6_AWI | T0500_EP0100 | past_only/train_config_CMIP6_AWI_Boston_past_only_G0_T0500_EP0100.sh | past_only/CMIP6_AWI_Boston_past_only_G0_T0500_EP0100__20260929_054511 |
| past_only | CMIP6_CNRM | T0000_EP0000 | past_only/train_config_CMIP6_CNRM_Boston_past_only_G0_T0000_EP0000.sh | past_only/CMIP6_CNRM_Boston_past_only_G0_T0000_EP0000__20260929_054511 |
| past_only | CMIP6_CNRM | T0500_EP0100 | past_only/train_config_CMIP6_CNRM_Boston_past_only_G0_T0500_EP0100.sh | past_only/CMIP6_CNRM_Boston_past_only_G0_T0500_EP0100__20260929_054511 |
| past_only | CMIP6_EC_EARTH | T0000_EP0000 | past_only/train_config_CMIP6_EC_EARTH_Boston_past_only_G0_T0000_EP0000.sh | past_only/CMIP6_EC_EARTH_Boston_past_only_G0_T0000_EP0000__20260929_054511 |
| past_only | CMIP6_EC_EARTH | T0500_EP0100 | past_only/train_config_CMIP6_EC_EARTH_Boston_past_only_G0_T0500_EP0100.sh | past_only/CMIP6_EC_EARTH_Boston_past_only_G0_T0500_EP0100__20260929_054511 |
| past_only | CMIP6_MPI | T0000_EP0000 | past_only/train_config_CMIP6_MPI_Boston_past_only_G0_T0000_EP0000.sh | past_only/CMIP6_MPI_Boston_past_only_G0_T0000_EP0000__20260929_054511 |
| past_only | CMIP6_MPI | T0500_EP0100 | past_only/train_config_CMIP6_MPI_Boston_past_only_G0_T0500_EP0100.sh | past_only/CMIP6_MPI_Boston_past_only_G0_T0500_EP0100__20260929_054511 |
| past_only | CMIP6_MRI | T0000_EP0000 | past_only/train_config_CMIP6_MRI_Boston_past_only_G0_T0000_EP0000.sh | past_only/CMIP6_MRI_Boston_past_only_G0_T0000_EP0000__20260929_054511 |
| past_only | CMIP6_MRI | T0500_EP0100 | past_only/train_config_CMIP6_MRI_Boston_past_only_G0_T0500_EP0100.sh | past_only/CMIP6_MRI_Boston_past_only_G0_T0500_EP0100__20260929_054511 |
| future_year | CMIP6_AWI | T0000_EP0000 | future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0000_EP0000.sh | future_year/CMIP6_AWI_Boston_future_year_G0_T0000_EP0000__20260929_054511 |
| future_year | CMIP6_AWI | T0500_EP0100 | future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0500_EP0100.sh | future_year/CMIP6_AWI_Boston_future_year_G0_T0500_EP0100__20260929_054511 |
| future_year | CMIP6_CNRM | T0000_EP0000 | future_year/train_config_CMIP6_CNRM_Boston_future_year_G0_T0000_EP0000.sh | future_year/CMIP6_CNRM_Boston_future_year_G0_T0000_EP0000__20260929_054511 |
| future_year | CMIP6_CNRM | T0500_EP0100 | future_year/train_config_CMIP6_CNRM_Boston_future_year_G0_T0500_EP0100.sh | future_year/CMIP6_CNRM_Boston_future_year_G0_T0500_EP0100__20260929_054511 |
| future_year | CMIP6_EC_EARTH | T0000_EP0000 | future_year/train_config_CMIP6_EC_EARTH_Boston_future_year_G0_T0000_EP0000.sh | future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0000_EP0000__20260929_054511 |
| future_year | CMIP6_EC_EARTH | T0500_EP0100 | future_year/train_config_CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0100.sh | future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0100__20260929_054511 |
| future_year | CMIP6_MPI | T0000_EP0000 | future_year/train_config_CMIP6_MPI_Boston_future_year_G0_T0000_EP0000.sh | future_year/CMIP6_MPI_Boston_future_year_G0_T0000_EP0000__20260929_054511 |
| future_year | CMIP6_MPI | T0500_EP0100 | future_year/train_config_CMIP6_MPI_Boston_future_year_G0_T0500_EP0100.sh | future_year/CMIP6_MPI_Boston_future_year_G0_T0500_EP0100__20260929_054511 |
| future_year | CMIP6_MRI | T0000_EP0000 | future_year/train_config_CMIP6_MRI_Boston_future_year_G0_T0000_EP0000.sh | future_year/CMIP6_MRI_Boston_future_year_G0_T0000_EP0000__20260929_054511 |
| future_year | CMIP6_MRI | T0500_EP0100 | future_year/train_config_CMIP6_MRI_Boston_future_year_G0_T0500_EP0100.sh | future_year/CMIP6_MRI_Boston_future_year_G0_T0500_EP0100__20260929_054511 |
| past_only | CMIP6_AWI | T0500_EP0000 | past_only/train_config_CMIP6_AWI_Boston_past_only_G0_T0500_EP0000.sh | past_only/CMIP6_AWI_Boston_past_only_G0_T0500_EP0000__20260930_233312 |
| past_only | CMIP6_AWI | T0000_EP0100 | past_only/train_config_CMIP6_AWI_Boston_past_only_G0_T0000_EP0100.sh | past_only/CMIP6_AWI_Boston_past_only_G0_T0000_EP0100__20260930_233312 |
| past_only | CMIP6_CNRM | T0500_EP0000 | past_only/train_config_CMIP6_CNRM_Boston_past_only_G0_T0500_EP0000.sh | past_only/CMIP6_CNRM_Boston_past_only_G0_T0500_EP0000__20260930_233312 |
| past_only | CMIP6_CNRM | T0000_EP0100 | past_only/train_config_CMIP6_CNRM_Boston_past_only_G0_T0000_EP0100.sh | past_only/CMIP6_CNRM_Boston_past_only_G0_T0000_EP0100__20260930_233312 |
| past_only | CMIP6_EC_EARTH | T0500_EP0000 | past_only/train_config_CMIP6_EC_EARTH_Boston_past_only_G0_T0500_EP0000.sh | past_only/CMIP6_EC_EARTH_Boston_past_only_G0_T0500_EP0000__20260930_233312 |
| past_only | CMIP6_EC_EARTH | T0000_EP0100 | past_only/train_config_CMIP6_EC_EARTH_Boston_past_only_G0_T0000_EP0100.sh | past_only/CMIP6_EC_EARTH_Boston_past_only_G0_T0000_EP0100__20260930_233312 |
| past_only | CMIP6_MPI | T0500_EP0000 | past_only/train_config_CMIP6_MPI_Boston_past_only_G0_T0500_EP0000.sh | past_only/CMIP6_MPI_Boston_past_only_G0_T0500_EP0000__20260930_233312 |
| past_only | CMIP6_MPI | T0000_EP0100 | past_only/train_config_CMIP6_MPI_Boston_past_only_G0_T0000_EP0100.sh | past_only/CMIP6_MPI_Boston_past_only_G0_T0000_EP0100__20260930_233312 |
| past_only | CMIP6_MRI | T0500_EP0000 | past_only/train_config_CMIP6_MRI_Boston_past_only_G0_T0500_EP0000.sh | past_only/CMIP6_MRI_Boston_past_only_G0_T0500_EP0000__20260930_233312 |
| past_only | CMIP6_MRI | T0000_EP0100 | past_only/train_config_CMIP6_MRI_Boston_past_only_G0_T0000_EP0100.sh | past_only/CMIP6_MRI_Boston_past_only_G0_T0000_EP0100__20260930_233312 |
| future_year | CMIP6_AWI | T0500_EP0000 | future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0500_EP0000.sh | future_year/CMIP6_AWI_Boston_future_year_G0_T0500_EP0000__20260930_233312 |
| future_year | CMIP6_AWI | T0000_EP0100 | future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0000_EP0100.sh | future_year/CMIP6_AWI_Boston_future_year_G0_T0000_EP0100__20260930_233312 |
| future_year | CMIP6_CNRM | T0500_EP0000 | future_year/train_config_CMIP6_CNRM_Boston_future_year_G0_T0500_EP0000.sh | future_year/CMIP6_CNRM_Boston_future_year_G0_T0500_EP0000__20260930_233312 |
| future_year | CMIP6_CNRM | T0000_EP0100 | future_year/train_config_CMIP6_CNRM_Boston_future_year_G0_T0000_EP0100.sh | future_year/CMIP6_CNRM_Boston_future_year_G0_T0000_EP0100__20260930_233312 |
| future_year | CMIP6_EC_EARTH | T0500_EP0000 | future_year/train_config_CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0000.sh | future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0000__20260930_233312 |
| future_year | CMIP6_EC_EARTH | T0000_EP0100 | future_year/train_config_CMIP6_EC_EARTH_Boston_future_year_G0_T0000_EP0100.sh | future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0000_EP0100__20260930_233312 |
| future_year | CMIP6_MPI | T0500_EP0000 | future_year/train_config_CMIP6_MPI_Boston_future_year_G0_T0500_EP0000.sh | future_year/CMIP6_MPI_Boston_future_year_G0_T0500_EP0000__20260930_233312 |
| future_year | CMIP6_MPI | T0000_EP0100 | future_year/train_config_CMIP6_MPI_Boston_future_year_G0_T0000_EP0100.sh | future_year/CMIP6_MPI_Boston_future_year_G0_T0000_EP0100__20260930_233312 |
| future_year | CMIP6_MRI | T0500_EP0000 | future_year/train_config_CMIP6_MRI_Boston_future_year_G0_T0500_EP0000.sh | future_year/CMIP6_MRI_Boston_future_year_G0_T0500_EP0000__20260930_233312 |
| future_year | CMIP6_MRI | T0000_EP0100 | future_year/train_config_CMIP6_MRI_Boston_future_year_G0_T0000_EP0100.sh | future_year/CMIP6_MRI_Boston_future_year_G0_T0000_EP0100__20260930_233312 |
