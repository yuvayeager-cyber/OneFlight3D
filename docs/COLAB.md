# First GPU run on Google Colab

This is the runbook for the first **real** VGGT execution. It does not claim
that the hard targets have been met: the first run records facts, including
VGGT's actual output schema and wall-clock timing.

Before starting, select **Runtime → Change runtime type → T4 GPU** (or any GPU
runtime) in Colab. This runbook uses the VGGT project's published CUDA 12.1
PyTorch example, with the exact VGGT commit used by `Dockerfile.vggt`. VGGT
weights are downloaded on first inference; network download time is outside the
pipeline timing report.

The requested `banana` sample is downloaded first because it is small, but it
is deliberately a negative test for this pipeline: ODMdata marks its GPS EXIF
as absent. The EXIF adapter must reject it rather than manufacture location
data. The run then uses Brighton Beach, a small ODMdata-listed image set with
GPS EXIF, for the actual VGGT attempt.

## Cell 1 — clone this exact branch

```bash
!git clone https://github.com/yuvayeager-cyber/OneFlight3D.git
%cd /content/OneFlight3D
!git checkout codex/oneflight3d-build
!git rev-parse HEAD
```

Record the commit hash in the run notes.

## Cell 2 — install the pinned GPU environment

```bash
!python --version
!nvidia-smi
!python -m pip install --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0
!python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
!python -m pip install -r requirements/gpu-runtime.txt
!python -m pip install --no-deps git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce
!python scripts/check_environment.py --output_dir runs/colab-preflight --strict
```

Stop if `nvidia-smi` fails or if `ready_for_full_pipeline` is `false`. Keep the
generated `runs/colab-preflight/environment_report.json` with the run artifacts.

## Cell 3 — required banana check (expected to reject missing GPS)

```bash
!git clone --depth 1 https://github.com/pierotofy/dataset_banana.git data/banana
!python scripts/extract_exif_metadata.py --input_dir data/banana/images --output_dir runs/banana
```

Expected result: non-zero exit with an EXIF-GPS error. Do not add synthetic or
guessed GPS to make the next stages run. That would invalidate metric alignment.

## Cell 4 — download a GPS-EXIF set and adapt it

```bash
!git clone --depth 1 https://github.com/pierotofy/drone_dataset_brighton_beach.git data/brighton_beach
!find data/brighton_beach -maxdepth 2 -type f | head -30
```

The repository layout has changed in the past. Set `IMAGE_DIR` to the directory
that directly contains the JPEGs (the command below finds the first candidate):

```python
from pathlib import Path
candidates = [p for p in Path("data/brighton_beach").rglob("*") if p.is_dir() and any(x.suffix.lower() in {".jpg", ".jpeg"} for x in p.iterdir())]
assert candidates, "No JPEG directory found; inspect the Cell 4 output."
IMAGE_DIR = candidates[0]
print(IMAGE_DIR)
```

```bash
!python scripts/extract_exif_metadata.py --input_dir "$IMAGE_DIR" --output_dir runs/brighton
!head -5 runs/brighton/frames_meta.csv
```

## Cell 5 — dynamic masking, then VGGT schema verification

Start with every other frame to control GPU memory. **Do not continue to Stage
4 if Stage 3 does not produce the schema file.**

```bash
!python scripts/02_mask_dynamic_objects.py \
  --input_dir runs/brighton/frames \
  --output_dir runs/brighton/frames_masked \
  --device cuda

!time python scripts/03_run_vggt.py \
  --input_dir runs/brighton/frames_masked \
  --output_file runs/brighton/vggt_predictions.npz \
  --schema_file runs/brighton/vggt_output_schema.json \
  --device cuda --max_frames 18

!cat runs/brighton/vggt_output_schema.json
```

Compare the exact keys and shapes in `vggt_output_schema.json` against Stage 4
(`extrinsics`, `intrinsics`, `world_points`, `world_points_conf`) before moving
on. If any differ, save the schema and fix the code in a commit; do not rename
the JSON or fake an NPZ field to conceal the mismatch.

## Cell 6 — run the remaining pipeline stages and record timings

`run_pipeline.py` starts at Stage 1, while this data path has already created
frames/metadata. Run the remaining stages individually, timing each real
command. (The Stage 3 `time` measurement in Cell 5 is its timing record.)

```bash
!time python scripts/04_gps_alignment.py \
  --predictions runs/brighton/vggt_predictions.npz \
  --frames_meta runs/brighton/frames_meta.csv \
  --output_file runs/brighton/aligned_points.npz

!time python scripts/05_export_formats.py \
  --aligned_points runs/brighton/aligned_points.npz \
  --frames_meta runs/brighton/frames_meta.csv \
  --output_dir runs/brighton/exports --formats all

!time python scripts/06_confidence_export.py \
  --aligned_points runs/brighton/aligned_points.npz \
  --output_dir runs/brighton/exports

!find runs/brighton -maxdepth 3 -type f | sort
```

For a video run, invoke `run_pipeline.py`; it writes the required consolidated
`timing_report.json`:

```bash
!time python scripts/run_pipeline.py \
  --video /content/your_drone_video.mp4 \
  --telemetry /content/telemetry.csv \
  --output_dir runs/video --device cuda --formats all
!cat runs/video/timing_report.json
```

Do not compare an image-set timing to the <15-minute / 10-minute-video target.
Only a timed, 10-minute video run can test that requirement.

`run_pipeline.py` also accepts `--image_dir` and runs the GPS-EXIF adapter as
Stage 1. This is the one-command image-set mode for a clean rerun, and it
persists `timing_report.json` even if a later stage fails:

```bash
!python scripts/run_pipeline.py \
  --image_dir "$IMAGE_DIR" --output_dir runs/brighton-full --device cuda --formats all
!cat runs/brighton-full/timing_report.json
```

## Cell 7 — benchmark only with surveyed checkpoints

Run this only after supplying a GCP CSV in the exact CRS of the aligned points:

```bash
!python scripts/benchmark_accuracy.py \
  --aligned_points runs/brighton/aligned_points.npz \
  --gcp_file /content/surveyed_gcps.csv \
  --output_report runs/brighton/accuracy_report.json
!cat runs/brighton/accuracy_report.json
```

Brighton Beach has no GCP file in ODMdata, so it cannot produce an accuracy
number. Never infer RMSE from the GPS-alignment residual: that residual uses
the same camera/GPS correspondences that fitted the transform.

## Docker alternative

On a Linux host with NVIDIA Container Toolkit:

```bash
docker build -f Dockerfile.vggt -t oneflight3d-vggt .
docker run --rm --gpus all -it -v "$PWD:/workspace" oneflight3d-vggt \
  bash -lc 'python scripts/check_environment.py --output_dir runs/docker-preflight --strict'
```

Then repeat Cells 3–7 inside the container. The base image includes CUDA but
does not include the model checkpoint; VGGT obtains the checkpoint on its first
real inference.
