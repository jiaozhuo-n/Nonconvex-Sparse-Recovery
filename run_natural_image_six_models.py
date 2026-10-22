"""Six-method natural-image restoration: redraw, evaluate, or train and evaluate."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
METHODS = ("alepom", "proposed", "nestd", "usbnet", "csformer", "damp")
FOUR_NETWORKS = METHODS[:4]
LABELS = dict(alepom="ALePOM", proposed="Proposed", nestd="NesTD-Net", usbnet="USB-Net", csformer="CSformer", damp="D-AMP")

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def invoke(function, arguments):
    previous = sys.argv
    try:
        sys.argv = [function.__module__] + [str(arg) for arg in arguments]
        function()
    finally:
        sys.argv = previous

def checkpoint_path(directory, selection):
    path = Path(selection["checkpoint"])
    return path if path.is_absolute() else directory / path

def freeze_checkpoints(directory):
    records = []
    for method in METHODS[:-1]:
        selection = read(directory / f"{method}_selection.json")
        checkpoint = checkpoint_path(directory, selection)
        if sha(checkpoint) != selection["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed: {method}")
        records.append(dict(method=method, checkpoint=checkpoint.name, sha256=sha(checkpoint)))
    write_json(directory / "CHECKPOINTS.json", dict(methods=list(METHODS), seed=2025,
        sensor_sha256=sha(directory / "sensor.pt"), selected=records))

def train_all(directory):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Training uses the original CUDA training implementation; a CUDA GPU is required.")
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "train_four_models.py"),
                    "--output", str(directory), "--methods", ",".join(FOUR_NETWORKS)],
                   cwd=str(HERE), check=True)
    import run_missing_baselines as extra
    import run_study as run
    settings = SimpleNamespace(**run.read(directory / "protocol.json")["settings"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    extra.train_csformer(directory, directory)
    sensor = {key: value.to("cuda") if torch.is_tensor(value) else value
              for key, value in torch.load(directory / "sensor.pt", map_location="cpu", weights_only=True).items()}
    extra.tune_damp(run.development(settings)["validation_selected"], settings, sensor, float(sensor["scale"]), directory)
    freeze_checkpoints(directory)

def evaluate_all(directory, output):
    import numpy as np
    import torch
    import run_study as run
    import run_highpass_learned_decoder as strong
    import confirm_impulse_report as evaluation
    import run_missing_baselines as extra
    if not torch.cuda.is_available():
        raise RuntimeError("The original full-resolution inference implementation requires a CUDA GPU.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    settings = SimpleNamespace(**run.read(directory / "protocol.json")["settings"])
    if settings.seed != 2025:
        raise ValueError("The supplied six-model plotting protocol is the declared seed-2025 experiment.")
    audit = read(directory / "CHECKPOINTS.json")
    if sha(directory / "sensor.pt") != audit["sensor_sha256"]:
        raise ValueError("The shared sensing matrix or normalization changed.")
    for item in audit["selected"]:
        if sha(directory / item["checkpoint"]) != item["sha256"]:
            raise ValueError(f"Checkpoint checksum failed: {item['method']}")
    sensor = {key: value.to("cuda") if torch.is_tensor(value) else value
              for key, value in torch.load(directory / "sensor.pt", map_location="cpu", weights_only=True).items()}
    manifest = read(HERE / "data/test_manifest.json")
    for photo in manifest["images"]:
        if sha(HERE / photo["path"]) != photo["sha256"]:
            raise ValueError(f"Test photograph changed: {photo['id']}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "test_manifest.json", manifest)
    photos = manifest["images"]
    rows = []
    for method in FOUR_NETWORKS:
        selection = read(directory / f"{method}_selection.json")
        checkpoint = torch.load(checkpoint_path(directory, selection), map_location="cpu", weights_only=True)
        model = strong.make_model(method, sensor, settings, checkpoint["trial"])
        model.load_state_dict(checkpoint["state"])
        own = evaluation.evaluate(model, photos, sensor, settings, dict(seed=2025, method=method), output)
        rows.extend(own)
        run.save(output / "completed" / f"2025_{method}.json", own)
        del model, checkpoint
        torch.cuda.empty_cache()
    selection = read(directory / "csformer_selection.json")
    checkpoint = torch.load(checkpoint_path(directory, selection), map_location="cpu", weights_only=True)
    model = extra.CSformerError(sensor).to("cuda")
    model.load_state_dict(checkpoint["state"])
    model.eval()
    with torch.no_grad():
        own = extra.evaluate_csformer(model, photos, settings, sensor, float(sensor["scale"]), output, seed=2025)
    rows.extend(own)
    run.save(output / "completed/2025_csformer.json", own)
    del model, checkpoint
    torch.cuda.empty_cache()
    params = read(directory / "damp_selection.json")["selected"]
    with torch.no_grad():
        own = extra.evaluate_damp(photos, settings, sensor, float(sensor["scale"]), params, output, seed=2025)
    rows.extend(own)
    run.save(output / "completed/2025_damp.json", own)
    expected = {(2025, photo["id"], method, condition["family"], condition["probability"], condition["amplitude"])
                for photo in photos for method in METHODS for condition in evaluation.conditions()}
    actual = [(row["seed"], row["image"], row["method"], row["family"], row["probability"], row["amplitude"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing or duplicate six-method inference results.")
    if not all(np.isfinite(row[key]) for row in rows for key in ("psnr", "ssim", "mse", "raw_psnr", "raw_mse")):
        raise ValueError("Nonfinite image restoration metric.")
    run.csvsave(output / "all_runs.csv", rows)
    write_json(output / "evaluation_protocol.json", dict(methods=list(METHODS), seed=2025,
        checkpoint_manifest_sha256=sha(directory / "CHECKPOINTS.json"),
        common_observations=True, image_count=24, record_count=len(rows),
        metric_summary="Per-photograph mean and sample standard deviation; not training-seed variation."))

def table_summary(output):
    with (output / "all_runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if float(row["probability"]) == .02 and float(row["amplitude"]) == .5]
    table = []
    for method in METHODS:
        own = [row for row in rows if row["method"] == method]
        if len(own) != 24 or len({row["image"] for row in own}) != 24 or {int(row["seed"]) for row in own} != {2025}:
            raise ValueError(f"Expected 24 unique seed-2025 photographs: {method}")
        item = dict(method=LABELS[method], photographs=24, seed=2025)
        for key in ("psnr", "ssim"):
            values = [float(row[key]) for row in own]
            item[key + "_mean"] = statistics.mean(values)
            item[key + "_sample_std"] = statistics.stdev(values)
        table.append(item)
    with (output / "table_mean_std.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for item in table:
        print(f"{item['method']}: PSNR {item['psnr_mean']:.3f} +/- {item['psnr_sample_std']:.3f}; SSIM {item['ssim_mean']:.4f} +/- {item['ssim_sample_std']:.4f}")

def redraw(output):
    import plot_natural_image_six_models as curves
    invoke(curves.main, ["--source", output, "--output", output / "six_models_figures"])
    import make_restoration_comparison as montage
    montage.SOURCE = output
    montage.OUTPUT_PNG = output / "restoration_comparison_6_models.png"
    montage.OUTPUT_PDF = output / "restoration_comparison_6_models.pdf"
    montage.main()
    table_summary(output)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replot", "evaluate", "train", "all"), default="replot")
    parser.add_argument("--checkpoints", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--training-output", type=Path, default=HERE / "runs/seed_2025")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode in ("train", "all"):
        source = args.training_output.resolve()
        if source == (HERE / "checkpoints").resolve():
            raise ValueError("Choose a new training-output directory for new training runs.")
        train_all(source)
    else:
        source = args.checkpoints.resolve()
    if args.mode in ("evaluate", "all"):
        evaluate_all(source, args.output)
    if args.mode in ("replot", "evaluate", "all"):
        redraw(args.output)
    print("COMPLETE", args.mode, str(args.output), flush=True)

if __name__ == "__main__":
    main()
