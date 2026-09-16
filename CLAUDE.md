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
[5] COLMAP export                [6] Confidence overlay export
    scripts/05_export_and_splat.py    scripts/06_confidence_export.py
    → cameras/images/points3D.txt     → confidence_cloud.ply (red=low, green=high)
        │
        ▼
    gsplat / Nerfstudio splatfacto (designed, not yet run)
        → textured, renderable Gaussian Splat scene
        ▼
    Web viewer (Three.js/Potree) + AI copilot — not yet built

Parallel safety net (independent of the pipeline above):
    OpenDroneMap (ODM), Docker — not yet set up
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

- **None of the six scripts have been run yet.** Everything below is unverified until
  Stage 1–3 actually execute on real data.
- **VGGT's output dict key names are unverified** against the installed package version.
  `03_run_vggt.py` must print `predictions.keys()` and Stage 4/5 key names must be
  checked against the real output before being trusted.
- **Missing required output formats.** The PS requires OBJ, PLY, LAS, GeoTIFF, and
  .glb/.gltf/.fbx. The pipeline currently only produces COLMAP text files and a
  confidence-colored `.ply`. GeoTIFF (orthomosaic/DSM) and mesh formats (.glb/.fbx) are
  not implemented anywhere yet.
- **No accuracy number has been measured.** Plan: benchmark against the Shahbazi et al.
  UAV gravel-pit dataset (158 images, 109 real surveyed GCPs/checkpoints), subsampled to
  simulate single-pass overlap, and report real RMSE. Never assert an accuracy figure
  that hasn't actually been measured.
- **No GPU access secured yet** (Colab/Kaggle/RunPod/organizer credits — undecided).
- **gsplat/Nerfstudio training pass** is designed (COLMAP export exists) but has never
  been run.
- **No web layer at all** — no upload UI, job/queue system, browser viewer, or AI
  copilot. Today this is a CLI pipeline of scripts run one video at a time, not the
  generic multi-user product the long-term vision describes.
- **ODM fallback is not set up** — this should happen in parallel with VGGT work, not
  after, per the "keep a working fallback for research-grade code under deadline" rule.
- **IMU / barometric altitude / RTK-PPK** — listed as optional PS inputs, not consumed
  by any code (GPS lat/lon/alt only, matched to frames by nearest timestamp).

## Conventions

- CLI scripts live in `scripts/`, numbered by pipeline stage (`01_...` through `06_...`).
  Shared math/helpers live in `utils/` (e.g. `geo_utils.py` for ENU conversion and the
  Umeyama solver).
- Every script takes real CLI args (`--video`, `--telemetry`, `--fps`, `--max_frames`,
  etc.) — never hardcode a path or filename. The point of this system is that it works
  on *any* single-pass drone video, not one pre-tested file.
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

1. Test the VGGT Hugging Face demo on a small public sample set (e.g. OpenDroneMap's
   `banana` set) — a five-minute feasibility gut-check before more engineering time.
2. Secure GPU access (Colab/Kaggle/RunPod, or ask organizers for sponsor credits).
3. In parallel, get a basic OpenDroneMap run working on sample data — the safety net,
   not optional.
4. Run Stages 1–3 on a small real/sample dataset; print and record VGGT's actual output
   keys; fix Stage 4/5 key names to match.
5. Add the missing required export formats (GeoTIFF, .glb/.gltf, .fbx, LAS) — currently
   the single biggest gap against the official grading rubric.
6. Run the pipeline against the Shahbazi gravel-pit GCP dataset (subsampled) and compute
   a real RMSE.
7. Wire Stage 5's `write_colmap_text()` into a real glue script and run an actual
   gsplat/splatfacto training pass end-to-end.
8. Build the browser viewer (Three.js or Potree) with the confidence overlay toggle.
9. Add the LLM copilot chat layer over scene metadata.
10. Only if time remains: progressive/live batch-based reconstruction.
