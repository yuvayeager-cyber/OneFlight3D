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

## Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (VGGT and YOLOv8 require GPU for practical speed)
- **Docker** (optional — only for the OpenDroneMap fallback pipeline)

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
- **FBX export** requires either `pyfbx` or Blender installed headlessly.
  If neither is available, FBX export is skipped with a warning.
- **VGGT licensing**: the research checkpoint is non-commercial; the commercial
  checkpoint excludes military applications. See CLAUDE.md for details.

## Architecture

See [CLAUDE.md](CLAUDE.md) for full architecture documentation, key decisions,
and rationale.
