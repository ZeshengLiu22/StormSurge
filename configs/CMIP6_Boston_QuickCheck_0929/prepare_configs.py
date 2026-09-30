#!/usr/bin/env python3
"""Prepare the original ten configs per group; use prepare_ablations.py to extend them."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import re
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS = Path('/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929')
GROUPS = {'past_only': ('Grid4_New_PastOnly', '0.6', '0.2', 'P'),
          'future_year': ('Grid4_New', '0.4545454545', '0.0909090909', 'F')}
VARIANTS = [('T0000_EP0000','0.0','0.0'), ('T0500_EP0100','0.05','0.01')]


def write_manifest(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not list(ROOT.rglob('train_config_*.sh')), 'Refuse to overwrite prepared configs'
    assert not RESULTS.exists(), 'Refuse to reuse a results directory'
    provenance = json.loads((ROOT/'template_provenance.json').read_text())
    for details in provenance.values():
        template = Path(details['template_path'])
        snapshot = Path(details['successful_formal_run'])/'config_used.sh'
        keys = re.findall(r'^([A-Za-z_][A-Za-z_0-9]*)=', template.read_text(), re.M)
        def values(path):
            result = subprocess.run(['bash','-c','source "$1"; shift; for key in "$@"; do declare -p "$key"; done',
                                     'bash',str(path),*keys],capture_output=True,text=True,check=True)
            return {key:line.split('=',1)[1] for key,line in zip(keys,result.stdout.splitlines())}
        assert values(template) == values(snapshot)
        assert (snapshot.parent/'exit_status').read_text().strip() == '0'
        assert all((snapshot.parent/f'best_{role}.pt').is_file() for role in ['overall','exceedance'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    all_rows = []
    for group,(data_folder,train_ratio,val_ratio,short) in GROUPS.items():
        folder, results = ROOT/group, RESULTS/group
        data = json.loads((folder/'dataset_audit.json').read_text())
        discovered = sorted(p.name for p in (REPO/'Data'/data_folder).glob('CMIP6_*') if p.is_dir())
        assert sorted(data['datasets']) == discovered, f'{group}: incomplete data audit'
        included = sorted(k for k,v in data['datasets'].items() if v['included'])
        assert len(included) == 5 and 'CMIP6_Cane5' not in included, f'{group}: expected five usable datasets'
        rows = []
        for dataset in included:
            for variant,tail,peak in VARIANTS:
                template = Path(provenance[variant]['template_path'])
                run_name = f'{dataset}_Boston_{group}_G0_{variant}'
                config = folder/f'train_config_{run_name}.sh'
                replacements = {'ROOT_DIR':f'"./Data/{data_folder}/{dataset}/graphs"',
                                'CHECKPOINT_SELECTION':'"overall"','ALL_RESULTS_ROOT':f'"{results}"',
                                'PACT_RUN_NAME':f'"{run_name}"','PYTHON_RUN_TAG_BASE':f'"{run_name}"'}
                if group == 'future_year':
                    replacements.update(TRAIN_RATIO=f'"{train_ratio}"',VAL_RATIO=f'"{val_ratio}"')
                text = template.read_text()
                for key,value in replacements.items():
                    text,n = re.subn(rf'^{key}=.*$', f'{key}={value}',text,flags=re.M)
                    assert n == 1
                text = text.replace('# Four-station gradient recommendation: KEEP GRID AS PROPOSED.',
                                    f'# Boston CMIP6 quick check: {dataset}, {group}; matching NCEP formal template.')
                text = text.replace('# Primary selector: exceedance; secondary: overall (both retained by active code).',
                                    '# Primary selector: overall; secondary: exceedance (both retained by active code).')
                config.write_text(text)
                rows.append(dict(group=group,dataset=dataset,variant=variant,tail_mse_weight=tail,
                                 episode_gt_aligned_peak_mse_weight=peak,config_path=str(config),
                                 result_path=str(results/f'{run_name}__{stamp}'),template_path=str(template),
                                 primary_checkpoint='overall',runstamp=stamp,
                                 queue_label=f'QB0929{short}_{dataset.removeprefix("CMIP6_")}_{variant}'))
        write_manifest(folder/'manifest.csv', rows)
        all_rows.extend(rows)
        launcher = '''#!/usr/bin/env bash
if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v qsub_local >/dev/null 2>&1 && [[ $- != *i* ]]; then
  exec bash -i "$0" "$@"
fi
set -euo pipefail
cd /home/exouser/StormSurge
CONFIG_ROOT=/home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929
CONFIG_FOLDER=CONFIG_FOLDER_VALUE
RESULTS_ROOT=RESULTS_ROOT_VALUE
PYTHON_BIN="${S0_PYTHON_BIN:-/home/exouser/.conda/envs/torchpyg-cu12x/bin/python}"
export PACT_RUNSTAMP=STAMP_VALUE
export USE_TMUX=1
export QSUB_LOCAL_SLOTS=1
"$PYTHON_BIN" "$CONFIG_ROOT/audit_configs.py" GROUP_VALUE
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  qsub_local() { DRY_RUN=1 USE_TMUX=0 bash "$1" "$3"; }
else
  "$PYTHON_BIN" - <<'CHECK_RUNTIME'
import sys, torch, torch_geometric, train
assert torch.cuda.is_available(), 'CUDA is required'
assert torch.cuda.is_bf16_supported(), 'BF16 is required'
print(f'[Preflight OK] Python={sys.executable}; torch={torch.__version__}; GPU={torch.cuda.get_device_name(0)}; BF16 available')
CHECK_RUNTIME
  if [[ -e "$RESULTS_ROOT/_orchestration" ]]; then
    echo '[FATAL] This group was already submitted; inspect submissions.tsv before any retry.' >&2
    exit 2
  fi
  mkdir -p "$RESULTS_ROOT"
  mkdir "$RESULTS_ROOT/_orchestration"
  cp "$CONFIG_FOLDER/manifest.csv" "$CONFIG_FOLDER/config_audit.json" "$CONFIG_FOLDER/dataset_audit.json" "$RESULTS_ROOT/_orchestration/"
  printf 'task_id\\tlabel\\tconfig_path\\tresult_path\\n' > "$RESULTS_ROOT/_orchestration/submissions.tsv"
  exec > >(tee -a "$RESULTS_ROOT/_orchestration/launch.log") 2>&1
fi
submit() {
  local label="$1" config="$2" result="$3" task_id
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    qsub_local train.sh "$label" "$config"
  else
    [[ ! -e "$result" ]] || { echo "[FATAL] Result path already exists: $result" >&2; return 2; }
    task_id=$(qsub_local train.sh "$label" "$config")
    [[ "$task_id" =~ ^[0-9]+$ ]] || { echo "[FATAL] Unexpected queue reply: $task_id" >&2; return 2; }
    printf '%s\\t%s\\t%s\\t%s\\n' "$task_id" "$label" "$config" "$result" >> "$RESULTS_ROOT/_orchestration/submissions.tsv"
    printf '[Submitted] id=%s label=%s result=%s\\n' "$task_id" "$label" "$result"
  fi
}
'''
        for token,value in [('CONFIG_FOLDER_VALUE',str(folder)),('RESULTS_ROOT_VALUE',str(results)),
                            ('STAMP_VALUE',stamp),('GROUP_VALUE',group)]:
            launcher = launcher.replace(token,value)
        for row in rows:
            launcher += 'submit '+' '.join(shlex.quote(row[k]) for k in ['queue_label','config_path','result_path'])+'\n'
        (folder/'launch_all.sh').write_text(launcher)
        splits = data['datasets'][included[0]]['split']
        split_text = '\n'.join(f'| {part.upper()} | {len(values["year_groups"])} | {values["year_groups"][0]}–{values["year_groups"][-1]} |' for part,values in splits.items())
        (folder/'README.md').write_text(f'''# Boston CMIP6 — {group}

10 configs: AWI, CNRM, EC_EARTH, MPI, MRI × `T0000_EP0000` / `T0500_EP0100`.
Cane5 is excluded by request. All five selected datasets passed the data audit.

Data source: `./Data/{data_folder}/<dataset>/graphs`.
TRAIN_RATIO=`{train_ratio}`, VAL_RATIO=`{val_ratio}`; chronological year groups;
`SHUFFLE_YEARS=0`, `FUTURE_ONLY=0`.

| Split | Year groups | Range |
| --- | ---: | --- |
{split_text}

Results root: `{results}`.
Exact per-run paths and the fixed batch stamp `{stamp}` are in `manifest.csv`.
Both checkpoint roles are saved; **overall is primary**.
Formulations and unchanged training settings are in [the shared README](../README.md).

Audit: `python ../audit_configs.py {group}` (use the configured training Python).
Launch from the repository: `bash configs/CMIP6_Boston_QuickCheck_0929/{group}/launch_all.sh`.
Use `DRY_RUN=1` for a preview. Queue receipts: `_orchestration/submissions.tsv` under the results root.
''')
    write_manifest(ROOT/'manifest.csv', all_rows)
    (ROOT/'launch_all.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /home/exouser/StormSurge
CONFIG_ROOT=/home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929
PYTHON_BIN="${S0_PYTHON_BIN:-/home/exouser/.conda/envs/torchpyg-cu12x/bin/python}"
# Audit both groups before any submission.
"$PYTHON_BIN" "$CONFIG_ROOT/audit_configs.py" past_only
"$PYTHON_BIN" "$CONFIG_ROOT/audit_configs.py" future_year
bash "$CONFIG_ROOT/past_only/launch_all.sh"
bash "$CONFIG_ROOT/future_year/launch_all.sh"
''')
    (ROOT/'README.md').write_text(f'''# Boston CMIP6 quick check — 0929

Test the NCEP extreme-aware objective using **two groups of 10 configs** (20 runs).
Included in each group: `CMIP6_AWI`, `CMIP6_CNRM`, `CMIP6_EC_EARTH`, `CMIP6_MPI`, `CMIP6_MRI`.
Excluded: `CMIP6_Cane5`, by request. All five selected datasets passed completeness,
tensor/history, target/timestamp, and split-population checks.

## Groups and chronological splits

| Group | Data source | TRAIN / VAL ratios | TRAIN / VAL / TEST year groups |
| --- | --- | --- | --- |
| [past_only](past_only/README.md) | `Data/Grid4_New_PastOnly` | 0.6 / 0.2 | 22 / 7 / 7, exactly matching NCEP |
| [future_year](future_year/README.md) | `Data/Grid4_New` | 0.4545454545 / 0.0909090909 | 30 / 6 / 30; TEST is all 2070_2071–2099_2100 |

`SHUFFLE_YEARS=0` and `FUTURE_ONLY=0` in both groups. The latter keeps historical
years available for TRAIN/VAL; future years appear only in the future-year TEST split.
Past-only folders contain unused future dictionaries; the loader reads only their 36 historical graph files.

## Two loss formulations

- `T0000_EP0000`: Global MSE.
- `T0500_EP0100`: Global MSE + 0.05 × Tail-MSE + 0.01 × EpisodeGTAlignedPeak-MSE.

Tail uses strict GT > TRAIN Q95; episode peak supervises one GT peak timestamp per
physical TRAIN episode with the existing fixed TRAIN prevalence normalization.

## Common settings and checkpoints

Boston; Single; GraphSAGE + Transformer; 24 h history; hidden 128; LR 5e-3;
300 epochs; batch 256; accumulation 4; seed 42; MSE global loss; cosine scheduler;
BF16 AMP and TF32. All other architecture, Adam, warmup, normalization, loader,
and evaluation settings match the successful current Boston NCEP formal templates.
TRAIN-only fitting and the existing eleven final metrics are unchanged.
Both `best_overall.pt` and `best_exceedance.pt` are retained and reevaluated on VAL/TEST.
**Overall is primary** for reporting, aliases, and primary prediction exports.

## Exact paths and launch

Config root: `{ROOT}`; separate `past_only/` and `future_year/` folders.

Results root: `{RESULTS}`; separate `past_only/` and `future_year/` folders.

`manifest.csv` records all 20 exact config/result paths; each group also has its own
manifest, audit report, template diff, and dry-run commands. No NCEP config/result is modified.

From `/home/exouser/StormSurge`:

```bash
DRY_RUN=1 bash configs/CMIP6_Boston_QuickCheck_0929/launch_all.sh
bash configs/CMIP6_Boston_QuickCheck_0929/launch_all.sh
```

The launcher audits both groups before submission and uses existing `qsub_local` →
`train.sh` → tmux, one GPU job at a time. Past-only jobs are queued first.
Queue IDs are saved in each results folder's `_orchestration/submissions.tsv`.
Use `qstat_local` in an interactive shell for status. The fixed batch cannot be submitted twice.
''')
    print(f'Prepared {len(all_rows)} configs: 10 past_only + 10 future_year; runstamp={stamp}')


if __name__ == '__main__':
    main()
