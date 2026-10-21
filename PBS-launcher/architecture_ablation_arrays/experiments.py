"""Deterministic sweep identities and conservative, CPU-only artifact verification."""
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import zipfile

GROUPS = {"LR3e-3": ("3e-3", "0"), "LR1e-3": ("1e-3", "0"),
          "LR5e-3_Clip0.25": ("5e-3", "0.25")}
STATIONS = ("Boston", "CBBT", "Lewes", "Battery")
HISTORIES = tuple(f"{h}H" for h in range(0, 49, 6))
TEMPORALS = ("MLP", "LSTM", "GRU", "Attn")
ENCODERS = ("CNN", "GNN")
JOB_NAMES = {"LR3e-3": "ss3", "LR1e-3": "ss1", "LR5e-3_Clip0.25": "ss5c"}
PYTHON = "/glade/work/zel/conda-envs/derecho-ml/bin/python"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignments(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)=(.*)", line)
        if match:
            key, value = match.groups()
            if value.startswith("(") and value.endswith(")"):
                value = value[1:-1]
            parts = shlex.split(value)
            result[key] = " ".join(parts)
    return result


@dataclass(frozen=True)
class Experiment:
    root: Path
    group: str
    encoder: str
    index: int

    def __post_init__(self):
        if self.group not in GROUPS or self.encoder not in ENCODERS or not 0 <= self.index < 144:
            raise ValueError("Expected a known LR group, CNN/GNN, and index 0..143")

    @property
    def station(self):
        return STATIONS[self.index // 36]

    @property
    def history(self):
        return HISTORIES[(self.index % 36) // 4]

    @property
    def temporal(self):
        return TEMPORALS[self.index % 4]

    @property
    def name(self):
        return f"NCEP_{self.station}_{self.encoder}_{self.temporal}_{self.history}_G0_T0000_EP0000"

    @property
    def identity(self):
        return f"{self.group}/{self.encoder}/{self.station}/{self.history}/{self.temporal}"

    @property
    def config(self):
        return self.root / "configs" / f"architecture_ablation_{self.group}" / self.history / self.station / f"train_config_{self.name}.sh"

    @property
    def launcher(self):
        return self.root / "PBS-launcher" / f"architecture_ablation_{self.group}" / f"run_{self.encoder}_array.pbs"

    @property
    def state_base(self):
        return self.root / "PBS-launcher" / "architecture_ablation_arrays" / "state"

    @property
    def state(self):
        return self.state_base / self.identity

    @cached_property
    def settings(self):
        return assignments(self.config)

    @property
    def results(self):
        path = Path(self.settings["ALL_RESULTS_ROOT"])
        return path if path.is_absolute() else self.root / path

    def validate(self):
        expected = {"STATION": self.station, "ENCODER_TYPE": "CNN" if self.encoder == "CNN" else "GraphSAGE",
                    "TEMPORAL_BLOCK": "Transformer" if self.temporal == "Attn" else self.temporal,
                    "HISTORY_HOURS_LIST": self.history[:-1], "LR_LIST": GROUPS[self.group][0],
                    "MAX_GRAD_NORM": GROUPS[self.group][1], "EPOCHS": "300", "num_gpus": "4",
                    "RUN_DIR_NAME_STYLE": "runname_timestamp", "PACT_RUN_NAME": self.name,
                    "PYTHON_RUN_TAG_BASE": self.name}
        for key, value in expected.items():
            if self.settings.get(key) != value:
                raise ValueError(f"{self.config}: expected {key}={value}, found {self.settings.get(key)}")
        expected_root = self.root.parent / f"PACT_StormSurge_FormalRuns_1021_{self.group}" / "architecture_ablation" / self.history / self.station
        if self.results.resolve() != expected_root.resolve():
            raise ValueError(f"Unexpected output root in {self.config}")
        return self


def runtime_env(stamp, dry=False):
    env = os.environ.copy()
    env.update(PYTHON_BIN=PYTHON, S0_PYTHON_BIN=PYTHON, USE_TMUX="0", DRY_RUN="1" if dry else "0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PACT_RUNSTAMP=stamp)
    for name in ("SLURM_NTASKS_PER_NODE", "SLURM_JOB_ID", "SESSION_NAME"):
        env.pop(name, None)
    if dry:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def preview(exp, stamp="PREVIEW"):
    proc = subprocess.run(["bash", str(exp.root / "train.sh"), str(exp.config)], cwd=exp.root,
                          env=runtime_env(stamp, True), text=True, capture_output=True, timeout=30, check=True)
    commands = [shlex.split(line[5:]) for line in proc.stdout.splitlines() if line.startswith("CMD: ")]
    launchers = [line for line in proc.stdout.splitlines() if line.startswith("LAUNCHER:")]
    if len(commands) != 1 or len(launchers) != 1 or "--nproc_per_node=4" not in launchers[0]:
        raise ValueError(f"{exp.identity}: expected exactly one four-rank DDP command")
    args = {}
    tokens = commands[0][1:]
    while tokens:
        key = tokens.pop(0).removeprefix("--")
        args[key] = tokens.pop(0) if tokens and not tokens[0].startswith("--") else True
    return proc.stdout, args


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


def same_argument(key, wanted, actual):
    if isinstance(actual, bool):
        return wanted is actual or str(wanted).lower() == str(actual).lower()
    if isinstance(actual, (int, float)):
        try:
            return math.isclose(float(wanted), actual, rel_tol=1e-12, abs_tol=0)
        except (TypeError, ValueError):
            return False
    if key in ("root_dir", "test_root_dir", "station_json_dir"):
        return os.path.normpath(str(wanted)) == os.path.normpath(str(actual))
    return wanted == actual


def verify_result(exp, directory, expected=None):
    """Require clean shell exit, all 300 epochs, matching arguments and final reports.

    No checkpoint deserialization or GPU imports. ZIP directory checks reject empty
    or truncated checkpoints/predictions; reports prove both roles were reloaded.
    """
    directory = Path(directory)
    try:
        if directory.parent.resolve() != exp.results.resolve() or not directory.name.startswith(exp.name + "__"):
            return False, "wrong output identity"
        if (directory / "exit_status").read_text().strip() != "0":
            return False, "unclean exit"
        required = ["launcher.log", "config_used.sh", f"train_{exp.name}.log", "best.pt", "best_overall.pt",
                    "best_exceedance.pt", "threshold_metadata.json", "episode_peak_metadata.json",
                    "checkpoint_comparison.json", "checkpoint_comparison.md"]
        for name in required:
            if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
                return False, f"missing/empty {name}"
        config_files = list(directory.glob("config_*.json"))
        if len(config_files) != 1:
            return False, "expected one effective training config"
        config_file = config_files[0]
        stem = config_file.stem.removeprefix("config_")
        config = json.loads(config_file.read_text())
        if expected is None:
            _, expected = preview(exp)
        for key, value in expected.items():
            if key not in ("output_dir", "run_tag") and not same_argument(key, value, config.get(key)):
                return False, f"config mismatch: {key}"
        if Path(config["output_dir"]).resolve() != directory.resolve():
            return False, "output provenance mismatch"
        if config["run_tag"] != exp.name + "_" + directory.name.split("__", 1)[1]:
            return False, "run tag mismatch"
        metrics = [json.loads(line) for line in (directory / f"metrics_{stem}.jsonl").read_text().splitlines() if line.strip()]
        if [row.get("epoch") for row in metrics] != list(range(1, 301)):
            return False, "expected contiguous epochs 1..300"
        for row in metrics:
            if not finite_tree(row):
                return False, "nonfinite epoch metrics"
            for split in ("train", "val"):
                if not isinstance(row.get(split, {}).get("all_rmse"), (int, float)):
                    return False, "missing epoch RMSE"
        summary = json.loads((directory / f"summary_{stem}.json").read_text())
        comparison = json.loads((directory / "checkpoint_comparison.json").read_text())
        if not finite_tree(summary) or not finite_tree(comparison):
            return False, "nonfinite final metrics"
        if summary.get("run_tag") != config["run_tag"] or not 1 <= summary.get("best_epoch", 0) <= 300:
            return False, "invalid final summary"
        if not isinstance(summary.get("wall_seconds"), (int, float)) or summary["wall_seconds"] <= 0:
            return False, "missing final elapsed time"
        primary = config.get("checkpoint_selection", "overall")
        if summary.get("checkpoint_selection", {}).get("checkpoint_selection_mode") != primary:
            return False, "missing primary checkpoint selection report"
        for split in ("val", "test"):
            if not isinstance(summary.get(split, {}).get("all_rmse"), (int, float)):
                return False, "missing primary final evaluation"
        for name in ("threshold_metadata.json", "episode_peak_metadata.json"):
            if not finite_tree(json.loads((directory / name).read_text())):
                return False, "nonfinite training metadata"
        for role in ("overall", "exceedance"):
            report = comparison[role]
            if summary.get("checkpoint_comparison", {}).get(role) != report:
                return False, "inconsistent final checkpoint reports"
            if not 1 <= report["epoch"] <= 300 or report.get("selection_split") != "val":
                return False, "invalid checkpoint provenance"
            if Path(report["path"]).resolve() != (directory / f"best_{role}.pt").resolve():
                return False, "checkpoint path mismatch"
            for split in ("val", "test"):
                if not isinstance(report[split].get("all_rmse"), (int, float)):
                    return False, "missing final evaluation"
                with zipfile.ZipFile(directory / f"{split}_predictions_{role}.npz") as archive:
                    if not {"y_true.npy", "y_pred.npy"}.issubset(archive.namelist()):
                        return False, "missing prediction arrays"
        if summary["best_epoch"] != comparison[primary]["epoch"]:
            return False, "inconsistent selected epoch"
        for name in ("best.pt", "best_overall.pt", "best_exceedance.pt", f"best_{stem}.pth"):
            with zipfile.ZipFile(directory / name) as archive:
                if not any(n.endswith("/data.pkl") for n in archive.namelist()):
                    return False, "invalid checkpoint archive"
        with zipfile.ZipFile(directory / f"test_preds_{stem}.npz") as archive:
            if not {"y_true.npy", "y_pred.npy"}.issubset(archive.namelist()):
                return False, "missing primary predictions"
        log = (directory / "launcher.log").read_text(errors="replace")
        if "Epoch 300/300" not in log or "FINAL MULTI-CHECKPOINT RE-EVALUATION" not in log or "Wall time (all combinations):" not in log:
            return False, "missing training/reporting completion in launcher.log"
        return True, "verified 300 epochs, clean exit, matching config and complete reports"
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        return False, str(exc)


def successful_result(exp):
    expected = None
    for directory in sorted(exp.results.glob(exp.name + "__*"), reverse=True):
        if not (directory / "exit_status").exists():
            continue
        if expected is None:
            _, expected = preview(exp)
        valid, _ = verify_result(exp, directory, expected)
        if valid:
            return str(directory)
    return None
