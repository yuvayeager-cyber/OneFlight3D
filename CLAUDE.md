# CLAUDE.md — OneFlight3D

Context for any Claude Code / AI session working in this repo. Read this before writing or changing code.

## What this project is

SIH 2026 submission — **Problem Statement 17** (National Technical Research Organisation / NTRO):
**"Single-Pass Drone Video to Accurate 3D Model Generation System."**

Goal: from a single continuous drone flight (video + GPS + flight metadata), produce a
georeferenced, metrically accurate, textured 3D model — terrain, structures, facades,
rooftops, roads, vegetation — **without** multiple flight passes, heavy image overlap, or
physical Ground Control Points (GCPs).

- Theme: Drone/Robotics · Category: Software
- Dataset: will be provided by NTRO in real time (not yet in hand)
- Project/pitch name: **OneFlight3D**

## Hard targets (from the official PS — these are graded, not optional)

| Target | Value |
|---|---|
| Processing time | < 15 minutes for a 10-minute video |
| Spatial accuracy | ≤ 1 m |
| Output formats | OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx |
| Visualization | Web-based or desktop viewer |

Evaluation weights: Reconstruction Accuracy 30% · Model Completeness 20% ·
Processing Speed 20% · Innovation 15% · Scalability 10% · User Interface 5%.
**70% of the score is measured (accuracy/completeness/speed), not pitched.** Prioritize
accordingly — the confidence-overlay story is only 15%.

## Architecture — six-stage pipeline

```
Drone video + GPS/telemetry
        │
        ▼
[1] Frame extraction + GPS sync         scripts/01_extract_frames.py
        │   video + telemetry.csv → frames/ + frames_meta.csv
        ▼
[2] Dynamic object masking (YOLOv8n)    scripts/02_mask_dynamic_objects.py
        │   frames/ → frames_masked/ (people/vehicles/animals zeroed out)
        ▼
[3] VGGT reconstruction (core engine)   scripts/03_run_vggt.py
        │   frames_masked/ → vggt_predictions.npz
        │   (extrinsics, intrinsics, depth, world_points, confidence)
        ▼
[4] GPS / metric alignment (Umeyama)    scripts/04_gps_alignment.py + utils/geo_utils.py
        │   vggt_predictions.npz + frames_meta.csv → aligned_points.npz
        ├──────────────────────────────┬─────────────────────────────┐
        ▼                              ▼
[5] Multi-format export           [6] Confidence overlay export
    scripts/05_export_formats.py      scripts/06_confidence_export.py
    → OBJ/PLY/LAS/GeoTIFF/glTF/        → confidence_cloud.ply
      GLB/FBX/COLMAP (dependency-       (red=low, green=high)
      gated; see verification status)
        │
        ▼
    Static web viewer (viewer/) — implemented; browser interaction unverified

Parallel safety net (independent of the pipeline above):
    OpenDroneMap (ODM), Docker — scripts/odm_fallback.py; container run unverified
    video frames + GPS → point cloud/mesh/orthomosaic
    Used ONLY as a fallback demo if the VGGT pipeline breaks
```

## Key decisions (and why — don't relitigate these without a real reason)

- **VGGT over DUSt3R/MASt3R** — VGGT (Visual Geometry Grounded Transformer, CVPR 2025
  Best Paper) jointly infers pose/depth/points/confidence for many-to-hundreds of views
  in one forward pass. DUSt3R/MASt3R are pairwise-only and need a separate global
  alignment step — worse fit for a continuous video sequence.
- **GPS + Umeyama similarity transform for metric scale, not physical GCPs** — VGGT's
  output is correct in relative shape but arbitrary in scale. Fit a closed-form
  similarity transform (scale + rotation + translation) between VGGT's camera positions
  and the same frames' GPS positions (converted to local ENU meters), then apply it to
  the whole cloud. Standard, well-tested algorithm — implemented in `utils/geo_utils.py`.
- **YOLOv8n bounding-box masking, not pixel-level segmentation** — VGGT's own docs say
  simple bounding-box masking of dynamic objects (people/vehicles/animals) is sufficient
  before reconstruction. Don't add SAM or precise segmentation; it's unneeded complexity.
- **Confidence overlay instead of generative inpainting for occluded surfaces** — VGGT
  gives a native per-point confidence score. Color by confidence (green=observed,
  red=inferred/occluded) rather than hallucinating plausible geometry for unseen
  surfaces. This is the core "innovation" claim in the pitch — do not undermine it by
  quietly smoothing over low-confidence regions in post-processing.
- **OpenDroneMap kept as a fully separate fallback, not a fused hybrid** — ODM only
  runs as an independent backup pipeline in case VGGT breaks before a demo. Do not
  try to merge ODM and VGGT outputs into one algorithm.
- **Custom COLMAP-format writer, not VGGT's own export script** — the exact behavior of
  VGGT's own COLMAP export for the installed version was never verified, so a
  from-scratch writer was built instead. Revisit only if VGGT's own script is confirmed
  reliable.

## Current state / known gaps (be honest about these — don't paper over them)

- **Local synthetic verification completed 2026-09-22:** Stage 4 alignment
  and PLY/COLMAP/confidence exports passed on generated test data. This is a
  crash/interface smoke test only, never an accuracy or performance result.
- **Real image-set Stage 1 was run 2026-09-22.** The 17-image ODM Banana set
  was correctly rejected because every image lacked the required GPS EXIF;
  no coordinates were invented. The 18-image Brighton Beach set produced
  `frames_meta.csv` successfully. The first orchestrated local run passed its
  EXIF stage in 0.70 s and then stopped at Stage 2 because `ultralytics` was
  absent; `run_pipeline.py` recorded that failure in `timing_report.json`.
  After installing `ultralytics` 8.4.158 with CPU PyTorch 2.14.0, a direct
  Stage 2 run completed on all 18 Brighton frames, wrote `frames_masked/` and
  `mask_report.json`, and detected six dynamic objects across five frames
  (three cars, one person, one bird, and one bus). This verifies CPU masking
  on real images only; it does not verify GPU execution or any later stage.
- **Real video Stage 1 has not run.** No video plus telemetry input has been
  supplied on this host. Stage 2 has masked real still images on CPU, but
  no real video pipeline has been run end-to-end.
- **Real VGGT inference/schema was run on a Colab Tesla T4 on 2026-09-23.**
  The Colab preflight reported CUDA available with PyTorch 2.11.0+cu128 and
  VGGT 0.0.1. Stage 3 emitted a raw-model schema with the keys `depth`,
  `depth_conf`, `images`, `pose_enc`, `pose_enc_list`, `world_points`, and
  `world_points_conf`. The real dense shapes were `world_points`
  `(1, 18, 294, 518, 3)`, `world_points_conf` `(1, 18, 294, 518)`, and
  `images` `(1, 18, 3, 294, 518)`. This verifies raw model inference and
  establishes that the model uses `pose_enc`, not raw `extrinsics`/
  `intrinsics` fields.
- **The first schema gate correctly exposed a Stage 4 boundary mismatch.**
  Stage 3 decodes `pose_enc` before saving `extrinsics`/`intrinsics`, but the
  old verifier incorrectly demanded those decoded fields in the raw-model
  schema; Stage 4 also assumed no batch axis. Stage 4 now normalizes a
  verified singleton leading batch axis from every consumed saved array and
  rejects any other batch size or malformed shape. The verifier now checks
  both the raw schema and the actual `vggt_predictions.npz` archive. The
  corrected gate and Stage 4 adapter have passed synthetic batch-first tests,
  but the real saved archive has not yet been inspected and Stage 4 has not
  yet run on the Colab output.
- **Required output writers are implemented but not all locally exercised.**
  Stage 5 has paths for OBJ, PLY, LAS, GeoTIFF DSM, GLB/glTF, FBX, and COLMAP.
  The local synthetic run validated PLY/COLMAP only. The initial strict
  preflight saw transient 15-second `pyproj`/`rasterio` import timeouts, but a
  repeat strict preflight passed both packages. Open3D remains absent; LAS,
  GeoTIFF, and mesh formats still require a real Stage 3-data test.
- **No accuracy number has been measured.** Plan: benchmark against the Shahbazi et al.
  UAV gravel-pit dataset (158 images, 109 real surveyed GCPs/checkpoints), subsampled to
  simulate single-pass overlap, and report real RMSE. Never assert an accuracy figure
  that hasn't actually been measured. Brighton Beach has no surveyed-GCP file,
  so this session could not run `benchmark_accuracy.py` or report RMSE.
- **This local host is not a GPU/ODM execution environment.** Its repeated
  2026-09-22 strict preflight found 8.59 GB free of the requested 10 GB, no
  PyTorch/VGGT, Docker, `nvidia-smi`, or CUDA GPU. Colab T4 has now run real
  VGGT inference, but Docker/ODM has not run on either host.
- **gsplat/Nerfstudio training pass** is designed (COLMAP export exists) but has never
  been run.
- **Viewer is implemented as an offline static tool** (`viewer/`): GLB/glTF
  primary loading, PLY fallback, confidence toggle, orbit/pan/zoom, and
  two-point metre measurements. It has not been visually exercised on this
  host because browser automation was unavailable.
- **ODM wrapper is implemented but Docker is unavailable locally.** It needs
  one verified run against a public GPS-EXIF sample before it is demo-ready.
- **IMU / barometric altitude / RTK-PPK** — listed as optional PS inputs, not consumed
  by any code (GPS lat/lon/alt only, matched to frames by nearest timestamp).

## Conventions

- CLI scripts live in `scripts/`, numbered by pipeline stage (`01_...` through `06_...`).
  Shared math/helpers live in `utils/` (e.g. `geo_utils.py` for ENU conversion and the
  Umeyama solver).
- Every script takes real CLI args (`--video`, `--telemetry`, `--image_dir`, `--fps`,
  `--max_frames`, etc.) — never hardcode a path or filename. The point of this
  system is that it works on a single-pass drone video or a GPS-EXIF image set,
  not one pre-tested input.
- Before trusting any third-party model's output structure (VGGT, YOLO, etc.), print and
  verify its actual keys/shape on real data — don't assume field names from
  documentation or memory.
- Camera intrinsics are currently assumed fixed across all frames (typical continuous
  flight, non-zooming camera). Flag clearly in code comments if a dataset breaks this
  assumption.
- GPS-to-frame correspondence uses nearest-neighbor timestamp matching (fine for
  1–10 Hz GPS logs). No IMU-based interpolation between GPS fixes exists — don't assume
  sub-GPS-interval precision anywhere downstream.

## Licensing constraint — disclose, don't hide

VGGT's original research checkpoint is non-commercial. A separate commercial checkpoint
(VGGT-1B-Commercial) exists but **explicitly excludes military applications**. Since this
PS is from NTRO and lists military reconnaissance as a potential application, this is a
real conflict. Disclose it proactively in the pitch and propose a custom-trained or
properly licensed model as the path for an actual defense deployment — don't present the
system as demo-ready for military use as-is.

## Recommended next steps (in order)

1. Pull the batch-normalization fix into the existing Colab checkout and run
   `verify_vggt_schema.py` against both the saved raw schema and
   `vggt_predictions.npz`; do not continue unless it passes.
2. Run Stage 4 on the verified real archive, then inspect its alignment output
   and diagnostics.
3. Complete Stages 5–6 with that aligned output and record generated files and
   the consolidated timing report.
4. Run OpenDroneMap against the Brighton Beach image set in Docker as the
   independent fallback.
5. Run the pipeline against the Shahbazi gravel-pit GCP dataset (subsampled) and compute
   a real RMSE.
6. Wire Stage 5's `write_colmap_text()` into a real glue script and run an actual
   gsplat/splatfacto training pass end-to-end.
7. Visually exercise the existing static viewer with a real exported asset.
8. Only if time remains: progressive/live batch-based reconstruction.
