# OneFlight3D

**Single-Pass Drone Video to Accurate 3D Model Generation System**

SIH 2026 · Problem Statement 17 (NTRO) · Theme: Drone/Robotics · Category: Software

From a single continuous drone flight (video + GPS telemetry), produce a
georeferenced, metrically accurate, textured 3D model — terrain, structures,
facades, rooftops, roads, vegetation — without multiple flight passes, heavy
image overlap, or physical Ground Control Points.

## Hard Targets (PS-17)

| Target | Value |
|---|---|
| Processing time | < 15 minutes for a 10-minute video |
| Spatial accuracy | ≤ 1 m |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |

## Verified status (2026-09-22)

This repository now contains the six-stage implementation, a static viewer,
an EXIF image adapter, an ODM Docker wrapper, and a synthetic smoke-test path.
The latest local verification deliberately made **no** accuracy, completeness,
or speed claim:

- Passed: Umeyama solver, northern/southern UTM selection, synthetic Stage 4
  alignment, PLY/COLMAP/confidence export, Python compilation, and viewer JS
  syntax.
- Not verified: a real video/telemetry run, YOLO inference, VGGT output schema,
  mesh reconstruction, GLB/glTF/FBX, Docker/ODM, real LAS/GeoTIFF output, and
  a surveyed-GCP accuracy benchmark.
- Local blocker: this Windows host has no CUDA GPU or Docker, less than the
  10 GB recommended free space, and its Application Control policy blocks the
  native `pyproj` and GDAL/rasterio DLLs. The pipeline uses a built-in WGS84
  ENU/UTM fallback for alignment, but a production machine should use a normal
  supported Python/GIS environment.

Do not claim the PS targets until a real flight data set and independently
surveyed checkpoints have been processed and recorded.

## Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (VGGT and YOLOv8 require GPU for practical speed)
- **Docker** (optional — only for the OpenDroneMap fallback pipeline)

Run this before a demo. It writes a transparent machine-readable report and
does not install or download anything:

```bash
python scripts/check_environment.py --output_dir output/preflight --strict
```

## Installation

```bash
git clone <this-repo>
cd OneFlight3D
pip install -r requirements.txt

# VGGT — install separately per their docs:
pip install vggt
# OR from source:
# git clone https://github.com/facebookresearch/vggt && cd vggt && pip install -e .
```

## Quick Start — One Command

```bash
python scripts/run_pipeline.py \
    --video path/to/drone_video.mp4 \
    --telemetry path/to/gps_telemetry.csv \
    --output_dir output/
```

This runs the entire six-stage pipeline and produces all required output
formats in `output/exports/`.

### Telemetry CSV Format

The GPS telemetry CSV must have these columns (header row required):

```csv
timestamp,latitude,longitude,altitude
1694000000.0,28.6139,77.2090,120.5
1694000001.0,28.6140,77.2091,121.0
...
```

- `timestamp`: Unix epoch seconds (float)
- `latitude`, `longitude`: WGS84 decimal degrees
- `altitude`: meters above sea level (WGS84 ellipsoid or MSL)

## Pipeline Stages

Each stage can also be run independently:

### Stage 1 — Frame Extraction + GPS Sync

```bash
python scripts/01_extract_frames.py \
    --video drone_video.mp4 \
    --telemetry gps_telemetry.csv \
    --output_dir output/ \
    --fps 2 \
    --max_frames 300
```

### Image-set adapter (GPS in EXIF)

For a UAV image set such as an ODM sample, create the same `frames/` and
`frames_meta.csv` contract that Stage 1 produces. The adapter stages images in
`output/frames` using hard links by default (and safely falls back to copies
when hard links are unavailable), so Stages 2–6 remain unchanged.

```bash
python scripts/extract_exif_metadata.py \
    --input_dir path/to/gps_tagged_images \
    --output_dir output/
```

### Synthetic smoke test (not a benchmark)

This lets Stages 4–6 be exercised without UAV imagery, GPU, or VGGT. Its
outputs are explicitly labelled synthetic and must never be presented as a
reconstruction or accuracy result.

```bash
python scripts/generate_synthetic_smoke_data.py --output_dir work/smoke
python scripts/04_gps_alignment.py --predictions work/smoke/vggt_predictions.npz --frames_meta work/smoke/frames_meta.csv --output_file work/smoke/aligned_points.npz
python scripts/05_export_formats.py --aligned_points work/smoke/aligned_points.npz --frames_meta work/smoke/frames_meta.csv --output_dir work/smoke/exports --formats ply,colmap
python scripts/06_confidence_export.py --aligned_points work/smoke/aligned_points.npz --output_dir work/smoke/exports
```

### Stage 2 — Dynamic Object Masking (YOLOv8n)

```bash
python scripts/02_mask_dynamic_objects.py \
    --input_dir output/frames/ \
    --output_dir output/frames_masked/
```

### Stage 3 — VGGT Reconstruction

```bash
python scripts/03_run_vggt.py \
    --input_dir output/frames_masked/ \
    --output_file output/vggt_predictions.npz \
    --device cuda
```

### Stage 4 — GPS/Metric Alignment

```bash
python scripts/04_gps_alignment.py \
    --predictions output/vggt_predictions.npz \
    --frames_meta output/frames_meta.csv \
    --output_file output/aligned_points.npz
```

### Stage 5 — Export All Formats

```bash
python scripts/05_export_formats.py \
    --aligned_points output/aligned_points.npz \
    --frames_meta output/frames_meta.csv \
    --output_dir output/exports/ \
    --formats all
```

### Stage 6 — Confidence Overlay

```bash
python scripts/06_confidence_export.py \
    --aligned_points output/aligned_points.npz \
    --output_dir output/exports/
```

### OpenDroneMap Fallback (independent)

```bash
python scripts/odm_fallback.py \
    --frames_dir output/frames/ \
    --output_dir output/odm_output/
```

The input images must retain GPS EXIF. The wrapper stages them into the ODM
project, runs `opendronemap/odm`, and writes `odm_report.json` with its measured
runtime and output checks. Validate the command first without Docker:

```bash
python scripts/odm_fallback.py --frames_dir output/frames --output_dir output/odm_output --dry_run
```

### Web viewer

The offline static viewer is in [`viewer/`](viewer/README.md). It loads a
selected run folder, prioritizes GLB/glTF then falls back to PLY, supports the
confidence overlay, orbit/pan/zoom, and two-point metre measurements. See its
README for serving instructions; all required Three.js files are vendored.

## Output Directory Structure

```
output/
├── frames/                     # Extracted video frames
│   └── frame_0001.jpg ...
├── frames_meta.csv             # Frame-to-GPS correspondence
├── frames_masked/              # Frames with dynamic objects masked
├── vggt_predictions.npz        # Raw VGGT output
├── vggt_output_schema.json     # Verified VGGT output key names & shapes
├── aligned_points.npz          # Georeferenced point cloud + metadata
├── exports/
│   ├── model.obj               # Mesh (OBJ format)
│   ├── model.mtl               # Material file for OBJ
│   ├── cloud.ply               # Point cloud (PLY format)
│   ├── cloud.las               # Point cloud (LAS format, geospatial)
│   ├── dsm.tif                 # Digital Surface Model (GeoTIFF)
│   ├── model.glb               # Mesh (glTF binary)
│   ├── model.gltf              # Mesh (glTF)
│   ├── model.fbx               # Mesh (FBX)
│   ├── colmap/                 # COLMAP text format
│   │   ├── cameras.txt
│   │   ├── images.txt
│   │   └── points3D.txt
│   └── confidence_cloud.ply    # Confidence-colored point cloud
├── timing_report.json          # Per-stage and total timing
└── mask_report.json            # YOLOv8 detection summary
```

## Accuracy Benchmarking

```bash
python scripts/benchmark_accuracy.py \
    --aligned_points output/aligned_points.npz \
    --gcp_file path/to/surveyed_gcps.csv \
    --output_report output/accuracy_report.json
```

GCP CSV format: `id,easting,northing,elevation` (UTM meters, same zone as
the pipeline's auto-detected UTM zone).

## Known Limitations

- **VGGT output keys are unverified** until Stage 3 actually runs on real data
  with a GPU. The code logs the real schema on first run.
- **No accuracy number has been measured yet.** The benchmark harness exists
  but requires the Shahbazi et al. gravel-pit dataset or equivalent.
- **Mesh/FBX export** requires Open3D plus either a functioning trimesh FBX
  writer or Blender on `PATH`. The exporter now tries trimesh then a real
  Blender-headless OBJ-to-FBX fallback. If neither route works, it skips FBX
  with a precise warning rather than producing a corrupt placeholder.
- **VGGT licensing**: the research checkpoint is non-commercial; the commercial
  checkpoint excludes military applications. See CLAUDE.md for details.

## Architecture

See [CLAUDE.md](CLAUDE.md) for full architecture documentation, key decisions,
and rationale.
