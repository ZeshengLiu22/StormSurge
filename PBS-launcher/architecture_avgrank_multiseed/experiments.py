#!/usr/bin/env python3
"""Generate and validate the isolated AvgRank study using only the standard library."""

import argparse
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path
import re
import shlex
import subprocess


REPOSITORY = Path("/glade/u/home/zel/scratch/StormSurge")
STATIONS = ("Battery", "Boston", "CBBT", "Lewes")
SEEDS = (42, 123, 2026)
MUTABLE = {"SEED", "ALL_RESULTS_ROOT", "PACT_RUN_NAME", "PYTHON_RUN_TAG_BASE"}
ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)=(.*)$")


@dataclass(frozen=True)
class Experiment:
    group: str
    architecture: str
    minutes: str
    walltime: str
    job_name: str

    @property
    def history(self):
        return self.architecture.rsplit("_", 1)[1]

    @property
    def temporal(self):
        name = self.architecture.split("_")[1]
        return "Transformer" if name == "Attn" else name

    @property
    def letter(self):
        return "A" if self.group == "LR3e-3" else "C"


EXPERIMENTS = (
    Experiment("LR3e-3", "CNN_GRU_48H", "46.557", "03:30:00", "arA_GRU48"),
    Experiment("LR3e-3", "CNN_LSTM_42H", "42.606", "03:15:00", "arA_LSTM42"),
    Experiment("LR3e-3", "CNN_GRU_30H", "35.194", "02:45:00", "arA_GRU30"),
    Experiment("LR5e-3_Clip0.25", "CNN_Attn_48H", "46.727", "03:30:00", "arC_Attn48"),
    Experiment("LR5e-3_Clip0.25", "CNN_MLP_36H", "37.910", "03:00:00", "arC_MLP36"),
    Experiment("LR5e-3_Clip0.25", "CNN_LSTM_30H", "35.193", "02:45:00", "arC_LSTM30"),
    Experiment("LR5e-3_Clip0.25", "CNN_Attn_24H", "33.087", "02:45:00", "arC_Attn24"),
    Experiment("LR5e-3_Clip0.25", "CNN_LSTM_24H", "32.251", "02:45:00", "arC_LSTM24"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family(experiment):
    return f"architecture_avgrank_multiseed_{experiment.group}"


def source_path(experiment, station):
    return (Path("configs") / f"architecture_ablation_{experiment.group}"
            / experiment.history / station
            / f"train_config_NCEP_{station}_{experiment.architecture}_G0_T0000_EP0000.sh")


def config_path(experiment, station, seed):
    return (Path("configs") / family(experiment) / experiment.architecture / station
            / f"seed{seed}"
            / f"train_config_NCEP_{station}_{experiment.architecture}_G0_T0000_EP0000_seed{seed}.sh")


def result_path(experiment, station, seed):
    return (REPOSITORY.parent / f"PACT_StormSurge_FormalRuns_1021_{experiment.group}"
            / "architecture_avgrank_multiseed" / experiment.architecture / station / f"seed{seed}")


def pbs_path(experiment):
    return Path("PBS-launcher") / family(experiment) / experiment.architecture / "run_derecho.pbs"


def assignments(text):
    """Require assignment-only configs, retaining the literal shell values."""
    values = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = ASSIGNMENT.fullmatch(line)
        require(match is not None, f"Not an assignment-only config: {line!r}")
        key, value = match.groups()
        require(key not in values, f"Duplicate config assignment: {key}")
        values[key] = value
    return values


def normalized(text):
    return [line for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
            and line.split("=", 1)[0] not in MUTABLE]


def replacements(experiment, station, seed, source):
    values = assignments(source)
    return {
        "SEED": str(seed),
        "ALL_RESULTS_ROOT": f'"{result_path(experiment, station, seed)}"',
        "PACT_RUN_NAME": f'"{shlex.split(values["PACT_RUN_NAME"])[0]}_seed{seed}"',
        "PYTHON_RUN_TAG_BASE": f'"{shlex.split(values["PYTHON_RUN_TAG_BASE"])[0]}_seed{seed}"',
    }


def config_text(experiment, station, seed, source):
    text = source
    for key, value in replacements(experiment, station, seed, source).items():
        text, count = re.subn(rf"^{key}=.*$", lambda _: f"{key}={value}", text, flags=re.M)
        require(count == 1, f"Expected exactly one {key} in {source_path(experiment, station)}")
    comment = (f"# AvgRank multiseed: group {experiment.letter}, {experiment.architecture}, "
               f"{station}, seed {seed}.\n# Source: {source_path(experiment, station)}\n")
    return text.replace("#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n" + comment, 1)


def wrapper_text(group):
    return f'''#!/usr/bin/env bash
# Shared implementation keeps both LR groups on the same execution protocol.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
exec bash "${{SCRIPT_DIR}}/../architecture_avgrank_multiseed/run_architecture.sh" {group} "$@"
'''


def pbs_text(experiment):
    return f'''#!/bin/bash -l
# Twelve sequential four-GPU DDP runs: Battery, Boston, CBBT, Lewes; seeds 42, 123, 2026.
# Measured estimate: {Decimal(experiment.minutes) * 3} minutes for three seeds.
#PBS -N {experiment.job_name}
#PBS -A ULHI0005
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=64:mem=487GB:mpiprocs=4:ompthreads=1:ngpus=4
#PBS -l walltime={experiment.walltime}
#PBS -j oe
#PBS -r n
#PBS -o {REPOSITORY / pbs_path(experiment).parent / 'pbs_logs'}/

set -euo pipefail
cd -- {REPOSITORY}
exec bash PBS-launcher/{family(experiment)}/run_architecture.sh {experiment.architecture}
'''


def write_new(path, text, executable=False):
    """Never replace differing existing files, including previously generated configs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text() == text, f"Refusing to overwrite differing file: {path}")
    else:
        with path.open("x") as stream:
            stream.write(text)
    if executable:
        path.chmod(0o755)


def generate(repo, destination):
    # Validate every source before writing any configs.
    sources = {(e, station): (repo / source_path(e, station)).read_text()
               for e in EXPERIMENTS for station in STATIONS}
    for (experiment, station), source in sources.items():
        validate_settings(assignments(source), experiment, station)
    for experiment in EXPERIMENTS:
        for station in STATIONS:
            for seed in SEEDS:
                write_new(destination / config_path(experiment, station, seed),
                          config_text(experiment, station, seed, sources[experiment, station]))
        write_new(destination / pbs_path(experiment), pbs_text(experiment), executable=True)
        write_new(destination / pbs_path(experiment).parent / "pbs_logs" / ".gitkeep", "")
        write_new(destination / "PBS-launcher" / family(experiment) / "run_architecture.sh",
                  wrapper_text(experiment.group), executable=True)
        write_new(destination / "PBS-launcher" / family(experiment) / ".gitignore",
                  "*/execution_logs/\n*/manifests/\n*/pbs_logs/*\n!*/pbs_logs/.gitkeep\n")
    print(f"Generated 96 configs, 8 PBS scripts, 2 group launchers, 8 PBS log directories in {destination}")


def validate_settings(values, experiment, station):
    expected = {
        "TRAIN_PY": '"train.py"', "ROOT_DIR": '"./dataset/NCEP/graphs"',
        "TEST_ROOT_DIR": '""', "MODEL": '"perceiver3"', "STATION": f'"{station}"',
        "ENCODER_TYPE": '"CNN"', "PRESSURE_MODE": '"raw"',
        "NODE_ATTENTION_BACKEND": '"math"', "WEIGHT_DECAY_POLICY": '"module"',
        "EPOCHS": "300", "SHUFFLE_YEARS": "0", "EXCEEDANCE_LOSS_WEIGHT": "0.0",
        "EPISODE_GT_ALIGNED_PEAK_WEIGHT": "0.0", "CHECKPOINT_SELECTION": '"overall"',
        "RUN_DIR_NAME_STYLE": '"runname_timestamp"', "num_gpus": "4",
        "TEMPORAL_BLOCK": f'"{experiment.temporal}"',
        "HISTORY_HOURS_LIST": f"({experiment.history[:-1]})",
        "LR_LIST": '("3e-3")' if experiment.letter == "A" else '("5e-3")',
        "MAX_GRAD_NORM": "0" if experiment.letter == "A" else "0.25",
        "TRAIN_RATIO": '"0.6"', "VAL_RATIO": '"0.2"',
        "FUTURE_ONLY": "0", "FUTURE_YEAR_THRESHOLD": "2030",
    }
    for key, expected_value in expected.items():
        require(values.get(key) == expected_value,
                f"{experiment.group}/{experiment.architecture}/{station}: "
                f"{key}={values.get(key)!r}; expected {expected_value!r}")


def validate_configs(repo, experiments):
    paths = []
    split_settings = set()
    for experiment in experiments:
        arch_root = repo / "configs" / family(experiment) / experiment.architecture
        require({p.name for p in arch_root.iterdir()} == set(STATIONS),
                f"Unexpected station entries: {arch_root}")
        for station in STATIONS:
            require({p.name for p in (arch_root / station).iterdir()} == {f"seed{s}" for s in SEEDS},
                    f"Unexpected seed entries: {arch_root / station}")
            original = (repo / source_path(experiment, station)).read_text()
            assignments(original)
            for seed in SEEDS:
                path = repo / config_path(experiment, station, seed)
                require({p.name for p in path.parent.iterdir()} == {path.name},
                        f"Unexpected config entries: {path.parent}")
                text = path.read_text()
                values = assignments(text)
                validate_settings(values, experiment, station)
                require(normalized(text) == normalized(original), f"Reference settings changed: {path}")
                for key, value in replacements(experiment, station, seed, original).items():
                    require(values.get(key) == value, f"Incorrect {key}: {path}")
                split_settings.add(tuple(values[key] for key in
                                         ("ROOT_DIR", "TEST_ROOT_DIR", "TRAIN_RATIO", "VAL_RATIO",
                                          "SHUFFLE_YEARS", "FUTURE_ONLY", "FUTURE_YEAR_THRESHOLD")))
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
                paths.append(path)
    require(len(split_settings) == 1, "Fixed year-split settings differ across configs")
    require(len(paths) == len(set(paths)) == 12 * len(experiments), "Duplicate or missing configs")
    return paths


def validate_all(repo, dry_run):
    paths = validate_configs(repo, EXPERIMENTS)
    for group in dict.fromkeys(e.group for e in EXPERIMENTS):
        experiments = [e for e in EXPERIMENTS if e.group == group]
        config_root = repo / "configs" / family(experiments[0])
        require({p.name for p in config_root.iterdir()} == {e.architecture for e in experiments},
                f"Unexpected architectures: {config_root}")
        launcher_root = repo / "PBS-launcher" / family(experiments[0])
        expected_pbs = {repo / pbs_path(e) for e in experiments}
        require(set(launcher_root.rglob("*.pbs")) == expected_pbs, f"Unexpected PBS scripts: {launcher_root}")
        require((launcher_root / "run_architecture.sh").read_text() == wrapper_text(group),
                f"Unexpected group wrapper: {launcher_root}")
    covered = []
    for experiment in EXPERIMENTS:
        pbs = repo / pbs_path(experiment)
        require(pbs.read_text() == pbs_text(experiment), f"Unexpected PBS directives or command: {pbs}")
        require((pbs.parent / "pbs_logs").is_dir(), f"Missing PBS log destination: {pbs}")
        if dry_run:
            result = subprocess.run(["bash", str(pbs)], cwd=repo, text=True,
                                    capture_output=True, check=True,
                                    env={**os.environ, "DRY_RUN": "1",
                                         "SLURM_NTASKS_PER_NODE": "999", "SLURM_JOB_ID": "not-a-number"})
            sequence = [Path(line.split("\t")[4]).resolve() for line in result.stdout.splitlines()
                        if line.startswith("CONFIG\t")]
            expected = [(repo / config_path(experiment, station, seed)).resolve()
                        for station in STATIONS for seed in SEEDS]
            require(sequence == expected, f"Wrong dry-run order or config coverage: {pbs}\n{result.stdout}")
            covered.extend(sequence)
            print(f"PASS dry-run: {experiment.group}/{experiment.architecture}: 12 configs")
    shell_files = [repo / "PBS-launcher/architecture_avgrank_multiseed/run_architecture.sh"]
    for group in dict.fromkeys(e.group for e in EXPERIMENTS):
        root = repo / "PBS-launcher" / f"architecture_avgrank_multiseed_{group}"
        shell_files.extend(root.rglob("*.sh"))
        shell_files.extend(root.rglob("*.pbs"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
    require(len({e.job_name for e in EXPERIMENTS}) == 8, "Duplicate PBS job names")
    if dry_run:
        require(Counter(covered) == Counter(path.resolve() for path in paths),
                "The eight PBS jobs do not cover all 96 configs exactly once")
    gpu_hours = sum(Decimal(e.minutes) for e in EXPERIMENTS) * 3 * 4 / 60
    reserved = sum(int(e.walltime[:2]) * 60 + int(e.walltime[3:5]) for e in EXPERIMENTS) * Decimal(4) / 60
    print("PASS: 96 configs (A=36, C=60), 8 independent PBS jobs, exact reference settings, fixed splits, Bash syntax")
    print(f"Measured estimate: {gpu_hours:.4f} GPU-hours; economy x0.7: {gpu_hours * Decimal('0.7'):.4f}")
    print(f"Requested walltime capacity: {reserved:.2f} GPU-hours; economy x0.7: {reserved * Decimal('0.7'):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, help="Generation destination; defaults to --repo")
    parser.add_argument("--group", choices=("LR3e-3", "LR5e-3_Clip0.25"))
    parser.add_argument("--architecture")
    parser.add_argument("--dry-run", action="store_true", help="Also execute all eight PBS bodies with DRY_RUN=1")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            require(not (args.group or args.architecture or args.dry_run), "Generation always covers the entire matrix")
            generate(args.repo, args.output_root or args.repo)
        elif args.group or args.architecture:
            require(args.group and args.architecture and not args.dry_run,
                    "Config preflight requires both --group and --architecture, without --dry-run")
            experiments = [e for e in EXPERIMENTS if (e.group, e.architecture) == (args.group, args.architecture)]
            require(len(experiments) == 1, "Unknown architecture/group combination")
            validate_configs(args.repo, experiments)
            print(f"[PREFLIGHT] {args.group}/{args.architecture}: all 12 configs match their references")
        else:
            validate_all(args.repo, args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"[FATAL] {error}\n")


if __name__ == "__main__":
    main()
