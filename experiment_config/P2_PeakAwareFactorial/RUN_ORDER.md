# P2 suggested run order

Run from the StormSurge repository root in the existing shell that defines
`qsub_local`. These are deferred execution instructions; generation/validation
submits nothing. Use one station list at a time with the known single-H100 queue.

| Batch | Station | Experiment IDs | Direct | Severity-shape | Command list |
| --- | --- | --- | ---: | ---: | --- |
| 1 | CBBT | 1–24 | 8 | 16 | `commands_CBBT.txt` |
| 2 | Lewes | 25–48 | 8 | 16 | `commands_Lewes.txt` |
| 3 | Battery | 49–72 | 8 | 16 | `commands_Battery.txt` |
| 4 | Boston | 73–96 | 8 | 16 | `commands_Boston.txt` |

Within each station, direct precedes severity-shape. Binary order is Tail,
Amplitude, Shape (always zero for direct), Peak, with Peak varying fastest.
The first config is direct T0 A0 S0 P0. All interaction combinations follow.

`commands_all.txt` concatenates the four station lists in the order above.
Choose the combined list or the station lists to avoid duplicate submissions.
The command form is `qsub_local train.sh <label> <config_path>`.

Scientific settings do not change between batches. Each config is a single
seed-42 run, with one history, learning rate and loss-mode entry. The queue's
existing single-slot behavior serializes jobs; no queue settings were changed.
Artifact directories use the semantic name followed by the execution timestamp.
