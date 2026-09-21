#!/usr/bin/env python3
"""Validate the fixed factorial using the unchanged launcher/parser and CPU models.

Never invoke train.main, an optimizer, a queue, or the diagnosis. Reuse the
preserved TRAIN-fitted statistics after checking source-file metadata and splits.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPECTED_SHA = "af53de756860765a9809b0c5c44bc828c2a08359"
RESULTS_ROOT = Path("/media/share/PACT/Results/All_results_0921_true_peak_amp_factorial")
STAMP = "20000101_000000"  # Fixed dry-run identity only; never a training seed.
STATIONS = ("CBBT", "Lewes", "Battery", "Boston")
LABELS = dict(S0="Single", D0="DualBase", D1="Tail", D2="Amp", D3="TailAmp")
OBSOLETE = ("PEAK_LOSS_WEIGHT", "PEAK_POOL", "PEAK_POOL_BETA", "EXCESS_AMP_POOL",
            "EXCESS_AMP_BETA", "peak_magnitude", "constrained_peak_magnitude")
LOSS_DIFFERENCES = dict(D0=set(), D1={"loss_mode", "tail_lambda"},
                        D2={"excess_amp_loss_weight"},
                        D3={"loss_mode", "tail_lambda", "excess_amp_loss_weight"})
SHELL_DIFFERENCES = dict(D0=set(), D1={"LOSS_MODE_LIST", "TAIL_LAMBDA_LIST"},
                         D2={"EXCESS_AMP_LOSS_WEIGHT"},
                         D3={"LOSS_MODE_LIST", "TAIL_LAMBDA_LIST", "EXCESS_AMP_LOSS_WEIGHT"})
COMMON = dict(
    root_dir="./Data/Grid4_New/NCEP/graphs", test_root_dir="", model="perceiver3",
    encoder_type="GraphSAGE", temporal_block="Transformer", history_hours=24,
    hidden_channels=128, num_layers=2, dropout=.05, head_dropout=.05,
    node_read_heads=8, time_read_heads=8, transformer_layers=2,
    transformer_ff_mult=4., transformer_dropout=0., max_time_steps=32,
    train_ratio=.6, val_ratio=.2, shuffle_years=0, future_only=0, seed=42,
    use_site_elevation=0, use_bathymetry=0, x_norm="zscore", x_clip=0., x_aug=0,
    x_aug_prob=0., x_aug_scale=0., x_aug_bias=0., x_nodes_per_graph=0,
    batch_size=256, grad_accum_steps=4, lr=.005, epochs=300, scheduler="cosine",
    warmup_epochs=5, warmup_start_factor=.1, min_lr=1e-6, max_grad_norm=0.,
    deterministic=0, amp=True, amp_dtype="bf16", tf32=True, num_workers=0,
    torch_threads=1, pin_memory=False, persistent_workers=False,
    prefetch_factor=0, mp_context="fork", excess_formulation="direct",
    exceedance_head_experiment=None, exceedance_gate_pooling="mean",
    gate_mode="window", dual_mode="exceedance", dual_ablation="none",
    exceedance_percentile=95., body_loss_weight=1., shape_loss_weight=0.,
    tail_frac=.05, checkpoint_selection="overall", checkpoint_overall_tol=.01,
    save_aux_checkpoints=0,
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def differences(left, right, ignored=()):
    return {key for key in left.keys() | right.keys()
            if key not in ignored and left.get(key) != right.get(key)}


def shell_values(path, names, repo, env):
    script = '''source "$1"; shift
for name in "$@"; do
  declare -n value="$name"
  printf '%s\\0%s\\0' "$name" "${#value[@]}"
  printf '%s\\0' "${value[@]}"
done'''
    raw = subprocess.check_output(["bash", "-c", script, "bash", str(path), *names],
                                  cwd=repo, env=env).decode().split("\0")[:-1]
    result = {}
    while raw:
        name, count = raw[:2]
        count = int(count)
        result[name] = raw[2:2 + count]
        raw = raw[2 + count:]
    return result


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--experiment-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    repo, experiment = options.repo.resolve(), options.experiment_dir.resolve()
    output = (options.output_dir or experiment / "validation").resolve()
    sys.path.insert(0, str(repo))
    import torch
    from emulator.common import configure_runtime
    from emulator.data import ForcingGraphStore, load_station_json, station_features_from_json
    from emulator.models import ModelConfig, build_model, count_model_parameters
    from emulator.models.heads import ExceedanceHead, SingleHead
    from emulator.training import ForecastLoss, LossConfig
    from emulator.training.arguments import parse_args
    from emulator.training.checkpoints import CheckpointSelector, SELECTION_METRICS

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    assert git("branch", "--show-current") == "main"
    assert git("rev-parse", "HEAD") == git("rev-parse", "main") == git("rev-parse", "origin/main") == EXPECTED_SHA
    assert not git("diff", "HEAD", "--name-only"), "Tracked production files changed"
    audit = json.loads((experiment / "initial_audit.json").read_text())
    assert audit["head"] == audit["origin_main"] == EXPECTED_SHA and audit["branch"] == "main"
    for relative, digest in audit["preserved_files"].items():
        assert sha(repo / relative) == digest, f"Preserved diagnostic changed: {relative}"
    for line in git("status", "--short", "--untracked-files=all").splitlines():
        assert line.removeprefix("?? ").startswith(experiment.name + "/"), line
    results_before = sorted(str(p) for p in RESULTS_ROOT.glob("**/*")) if RESULTS_ROOT.exists() else None
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((experiment / "manifest.csv").open()))
    configs = sorted((experiment / "configs").glob("*.sh"))
    assert len(rows) == len(configs) == 20
    assert Counter(row["station"] for row in rows) == dict.fromkeys(STATIONS, 5)
    assert {(row["station"], row["condition"]) for row in rows} == {(s, c) for s in STATIONS for c in LABELS}
    assert {Path(row["config_path"]).name for row in rows} == {p.name for p in configs}
    assert len({row["run_name"] for row in rows}) == 20
    assert (experiment / "commands_all.txt").read_text().splitlines() == [row["qsub_local_command"] for row in rows]
    for station in STATIONS:
        assert (experiment / f"commands_{station}.txt").read_text().splitlines() == [
            row["qsub_local_command"] for row in rows if row["station"] == station]

    # A clean local launcher environment prevents an enclosing Slurm job or an
    # unrelated experiment's exports from changing a dry-run identity.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER", "LANG", "LD_LIBRARY_PATH") if key in os.environ}
    env.update(DRY_RUN="1", PACT_RUNSTAMP=STAMP, USE_TMUX="1", PYTHONDONTWRITEBYTECODE="1")
    parsed, shells, run_reports = {}, {}, []
    defaults = vars(parse_args(["--model", "perceiver3"]))
    for row in rows:
        station, condition, run_name = row["station"], row["condition"], row["run_name"]
        dual, tail, amp = condition != "S0", condition in {"D1", "D3"}, condition in {"D2", "D3"}
        expected_name = f"0921_{station}_{condition}_{LABELS[condition]}"
        assert run_name == expected_name
        relative = f"{experiment.name}/configs/train_config_{run_name}.sh"
        assert row["config_path"] == relative
        assert row["qsub_local_command"] == f"qsub_local train.sh {run_name} {relative}"
        config = experiment / "configs" / Path(relative).name
        content = config.read_text()
        assert all(name not in content for name in OBSOLETE)
        assert not re.search(r"^\s*(source|\.)\s", content, re.M), "Configs must be standalone"
        subprocess.run(["bash", "-n", str(config)], check=True)
        names = re.findall(r"^([A-Za-z_]\w*)=", content, re.M)
        assert len(names) == len(set(names)), "Duplicate shell assignments"
        shell = shell_values(config, names, repo, env)
        assert all(len(value) == 1 for value in shell.values()), "No sweep axes are permitted"
        shell_expected = dict(num_gpus="1", DISABLE_OOD="1", DO_CONDA="0", USE_TMUX="1",
                              TRAIN_PY="train.py", ALL_RESULTS_ROOT=str(RESULTS_ROOT),
                              RUN_DIR_NAME_STYLE="runname_timestamp", PACT_RUN_NAME=run_name,
                              PYTHON_RUN_TAG_BASE=run_name,
                              PYTHON_BIN="/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python",
                              DUAL_LOSS="1" if dual else "0",
                              TAIL_LAMBDA_LIST="0.025" if tail else "0.10",
                              SLOPE_LAMBDA_LIST="0.01", EXCEEDANCE_HEAD_EXPERIMENT="")
        assert {key: shell[key][0] for key in shell_expected} == shell_expected
        assert Path(shell["PYTHON_BIN"][0]).samefile(sys.executable), "Use the configured production Python"
        # Source the unchanged launcher, then call its read-only snapshot helper.
        # DRY_RUN=1 takes the actual launcher's command-generation branch.
        snapshot = output / f"{run_name}.resolved.txt"
        invocation = ["bash", "-c", 'source ./train.sh "$1"; write_resolved_config "$2"',
                      "bash", str(config), str(snapshot)]
        result = subprocess.run(invocation, cwd=repo, env=dict(env, SESSION_NAME=run_name),
                                text=True, capture_output=True, check=True)
        (output / f"{run_name}.dry_run.txt").write_text(result.stdout + result.stderr)
        commands = [shlex.split(line[5:]) for line in result.stdout.splitlines() if line.startswith("CMD: ")]
        assert len(commands) == 1 and commands[0][0] == "train.py"
        resolved = shell_values(snapshot, names, repo, env)
        assert resolved == shell, differences(resolved, shell)
        args = parse_args(commands[0][1:])
        values = vars(args)
        expected = dict(defaults, **COMMON)
        expected.update(station=station, head_type="dual" if dual else "single", dual_loss=int(dual),
                        loss_mode="mse_tail" if tail else "mse", tail_lambda=.025 if tail else .1,
                        excess_amp_loss_weight=.003 if amp else 0., excess_loss_weight=2. if dual else 1.,
                        gate_loss_weight=.5 if dual else 1., run_tag=f"{run_name}_{STAMP}",
                        output_dir=str(RESULTS_ROOT / f"{run_name}__{STAMP}"))
        assert values == expected, {key: (values.get(key), expected.get(key)) for key in differences(values, expected)}
        manifest_expected = dict(experiment_id="0921_true_peak_amp_factorial", head_type=args.head_type,
            loss_mode=args.loss_mode, tail_enabled=str(int(tail)), tail_lambda="0.025" if tail else "0",
            amp_enabled=str(int(amp)), amp_weight="0.003" if amp else "0",
            exceedance_percentile="95" if dual else "N/A", body_loss_weight="1" if dual else "N/A",
            excess_loss_weight="2" if dual else "N/A", gate_loss_weight="0.5" if dual else "N/A",
            lr="5e-3", seed="42", checkpoint_selection="overall", results_root=str(RESULTS_ROOT),
            tail_lambda_config="0.025" if tail else "0.10")
        assert {key: row[key] for key in manifest_expected} == manifest_expected
        selector = CheckpointSelector(args.checkpoint_selection, args.checkpoint_overall_tol, args.save_aux_checkpoints)
        assert not selector.needs_candidates and SELECTION_METRICS[args.checkpoint_selection] == "rmse_all"
        key = (station, condition)
        parsed[key], shells[key] = args, shell
        run_reports.append(dict(station=station, condition=condition, run_name=run_name,
            config=relative, config_sha256=sha(config), syntax="PASS", command_count=1,
            dry_run="PASS", argparse="PASS", resolved_shell="PASS", parsed_args=values.copy()))
        print(f"PASS {run_name}: shell, one launcher command, argparse, manifest, frozen protocol", flush=True)

    for station in STATIONS:
        base = vars(parsed[station, "D0"])
        for condition in LOSS_DIFFERENCES:
            assert differences(base, vars(parsed[station, condition]), {"run_tag", "output_dir"}) == LOSS_DIFFERENCES[condition]
            assert differences(shells[station, "D0"], shells[station, condition],
                               {"PACT_RUN_NAME", "PYTHON_RUN_TAG_BASE"}) == SHELL_DIFFERENCES[condition]
        single_changes = differences(base, vars(parsed[station, "S0"]), {"run_tag", "output_dir"})
        assert single_changes == {"head_type", "dual_loss", "excess_loss_weight", "gate_loss_weight"}
    all_dual_ignored = {"station", "run_tag", "output_dir", "loss_mode", "tail_lambda", "excess_amp_loss_weight"}
    for station in STATIONS:
        for condition in LOSS_DIFFERENCES:
            assert not differences(vars(parsed["CBBT", "D0"]), vars(parsed[station, condition]), all_dual_ignored)

    models, datasets = [], []
    for station in STATIONS:
        metadata = json.loads((repo / experiment.name / "diagnosis" / station / "metadata.json").read_text())
        args = parsed[station, "D0"]
        data_root = (repo / args.root_dir).resolve()
        assert data_root == Path(metadata["training_args"]["root_dir"])
        files = [p for p in sorted(data_root.glob("*graphs.pt")) if p.name.split("_")[2] == station]
        file_store = ForcingGraphStore.__new__(ForcingGraphStore)
        file_store.year_to_indices = defaultdict(list)
        for index, path in enumerate(files):
            file_store.year_to_indices["_".join(path.name.split("_")[:2])].append(index)
        split_options = {key: getattr(args, key) for key in
                         ("train_ratio", "val_ratio", "shuffle_years", "seed", "future_only", "future_year_threshold")}
        splits = file_store.split(**split_options)
        file_splits = {part: [files[index].name for index in ids] for part, ids in splits.items()}
        assert file_splits == metadata["file_splits"]
        for saved in metadata["opened_graph_files"]:
            stat = Path(saved["path"]).stat()
            assert stat.st_size == saved["bytes"] and stat.st_mtime_ns == saved["mtime_ns"]
        first_train_path = files[splits["train"][0]]
        graphs = torch.load(first_train_path, map_location="cpu", weights_only=False)
        graph = graphs[0]
        in_channels, out_channels = graph.x.size(-1), graph.y.numel()
        assert graph.x_hist.size(0) >= args.history_hours // 6 + 1
        del graph, graphs
        assert (in_channels, out_channels) == (5, 6)
        stats = {key: torch.tensor(value, dtype=torch.float32) for key, value in metadata["normalization"].items()}
        fitted = metadata["thresholds"]
        assert fitted["exceedance_percentile"] == 95 and metadata["training_args"]["tail_frac"] == .05
        assert fitted["event_prior"] == fitted["event_count"] / fitted["train_windows"]
        baseline_state, baseline_config, baseline_loss, baseline_counts = None, None, None, None
        for condition in ("D0", "D1", "D2", "D3", "S0"):
            args = parsed[station, condition]
            # One production seed call for each independent model construction,
            # as in train.main. No extra resets, RNG forks or copied parameters.
            configure_runtime(args.seed, args.torch_threads, bool(args.deterministic), bool(args.tf32))
            station_json = load_station_json(repo / args.station_json_dir, args.station)
            station_feat = station_features_from_json(station_json, use_site_elevation=args.use_site_elevation,
                                                       use_bathymetry=args.use_bathymetry)
            assert station_feat.numel() == 6
            values = {field.name: getattr(args, field.name) for field in fields(ModelConfig) if hasattr(args, field.name)}
            values.update(in_channels=in_channels, out_channels=out_channels, model="pact",
                temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                temporal_dropout=args.transformer_dropout, history_steps=args.history_hours // 6,
                station_feat_dim=station_feat.numel(), peak_prior=fitted["event_prior"],
                peak_threshold_norm=((fitted["tau_phys"] - stats["y_mean"]) / stats["y_std"]).tolist()
                if args.head_type == "dual" else None)
            model_config = ModelConfig(**values)
            model = build_model(model_config)
            assert type(model.head) is (SingleHead if condition == "S0" else ExceedanceHead)
            counts = count_model_parameters(model)
            loss_config = LossConfig(**{field.name: getattr(args, field.name) for field in fields(LossConfig)})
            ForecastLoss(loss_config, stats, fitted["tail_threshold"], fitted["wmse_threshold"],
                         event_prior=fitted["event_prior"], event_threshold=fitted["tau_phys"])
            state = model.state_dict()
            if condition == "D0":
                baseline_state = {key: value.clone() for key, value in state.items()}
                baseline_config, baseline_loss, baseline_counts = asdict(model_config), asdict(loss_config), counts
            elif condition != "S0":
                assert asdict(model_config) == baseline_config and counts == baseline_counts
                assert state.keys() == baseline_state.keys()
                assert all(torch.equal(value, baseline_state[key]) for key, value in state.items())
                assert differences(asdict(loss_config), baseline_loss) == LOSS_DIFFERENCES[condition]
            else:
                assert counts["total"] != baseline_counts["total"]
                assert counts["backbone"] == baseline_counts["backbone"]
                assert all(torch.equal(value, baseline_state[key]) for key, value in state.items() if not key.startswith("head."))
            models.append(dict(station=station, condition=condition, head_class=type(model.head).__name__,
                               counts=counts, state_sha256=state_digest(state), model_config=asdict(model_config)))
            del model
        datasets.append(dict(station=station, file_splits=file_splits,
            first_train_file_opened=str(first_train_path), in_channels=in_channels, out_channels=out_channels,
            station_features=6, train_windows=fitted["train_windows"], thresholds=fitted,
            cached_train_metadata_sha256=sha(repo / experiment.name / "diagnosis" / station / "metadata.json"),
            unchanged_train_files_checked=len(metadata["opened_graph_files"])))
        print(f"PASS {station}: chronological splits; D0-D3 architecture, count and exact initial state match", flush=True)

    source = (repo / "train.py").read_text()
    metric_map = dict(AllRMSE="rmse_all", AllMAE="mae_all", Top5RMSE="rmse_peak5", Top5MAE="mae_peak5",
        TruePeakRMSE="true_peak_rmse_top5", TruePeakMAE="true_peak_mae_top5", TruePeakBias="true_peak_bias_top5",
        **{"TruePeakUnder%": "true_peak_underprediction_fraction_top5", "TimingSteps": "peak_timing_mae_steps_top5"})
    for label, key in metric_map.items():
        assert f'("{label}", "{key}")' in source
    assert "torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)" in source
    assert not git("diff", "HEAD", "--name-only")
    for relative, digest in audit["preserved_files"].items():
        assert sha(repo / relative) == digest
    results_after = sorted(str(p) for p in RESULTS_ROOT.glob("**/*")) if RESULTS_ROOT.exists() else None
    assert results_after == results_before
    report = dict(status="PASS", main_sha=git("rev-parse", "HEAD"), branch="main", origin_main=git("rev-parse", "origin/main"),
        production_changes=[], preserved_diagnostic_files=len(audit["preserved_files"]), configs=20,
        configs_per_station=dict.fromkeys(STATIONS, 5), dry_runs_passed=20, argparse_passed=20,
        dual_only_intended_differences="PASS", dual_identical_initialized_states="PASS",
        no_obsolete_controls="PASS", training_launched=False, results_root=str(RESULTS_ROOT),
        results_root_unchanged=True, optimizer="production Adam", weight_decay=1e-5,
        python=sys.executable, torch=torch.__version__, metrics=metric_map,
        initialization_method="CPU construction from parsed launcher args and preserved TRAIN statistics; one production seed call per model; exact tensor comparison including buffers. TRAIN sources checked by size/mtime and filename splits; one TRAIN file per station opened for dimensions. No refitting, forward pass, optimizer, VAL/TEST file loading or diagnosis execution.",
        metadata_note="Requested USE_SITE_ELEVATION=0 gives six station features. Preserved diagnosis used elevation=1 (seven features); its model counts are not used for this factorial.",
        runs=run_reports, models=models, datasets=datasets)
    write_json(output / "validation_report.json", report)
    counts_by_head = {m["model_config"]["head_type"]: m["counts"]["total"] for m in models}
    lines = ["# Fixed factorial validation", "", f"PASS against production main `{EXPECTED_SHA}`.", "",
        "20/20 configs passed shell syntax, singleton axes, actual train.sh dry-run command generation, resolved shell snapshots, production argparse, manifest and all prescribed settings.", "",
        "Five configs per station: S0 plus D0/D1/D2/D3. Dual pairs differ only by loss_mode/tail_lambda and/or excess_amp_loss_weight, after excluding run identity. All other arguments and runtime settings match across all Dual runs.", "",
        f"Parameters: Single **{counts_by_head['single']:,}**; Dual **{counts_by_head['dual']:,}**. D0–D3 have identical ModelConfig, counts, and every initialized state tensor within each station. S0 retains the identical initialized backbone.", "",
        report["initialization_method"], "", report["metadata_note"], "",
        "Canonical nine reporting columns verified, including the five `_top5` true-peak-time metrics. Every run selects by VAL overall RMSE, tolerance 0.01, with auxiliary checkpoints disabled. Adam weight decay remains 1e-5.", "",
        f"All {len(audit['preserved_files'])} pre-existing diagnosis files passed SHA-256 preservation checks. Production tracked files are unchanged; no obsolete controls appear in configs. No jobs, training, checkpoint writes or results folders were created.", "",
        "The JSON report contains every parsed argument, model count, initial-state hash and filename split. Per-run dry-run transcripts and resolved snapshots are beside this report.", ""]
    (output / "validation_report.md").write_text("\n".join(lines))
    print(f"PASS: 20 configs; Single={counts_by_head['single']:,}; Dual={counts_by_head['dual']:,}; no training.")


if __name__ == "__main__":
    main()
