# Colab artifact handoff

Use one directory per actual GPU run:

```text
runs/colab/<dataset>-<gpu>-<UTC timestamp>/
```

For example, a first Brighton Beach T4 run on 22 September 2026 belongs in
`runs/colab/brighton-beach-t4-20260922T174500Z/`. Raw contents under
`runs/colab/` are Git-ignored; this README is the only tracked file there.
Do not rename files from the pipeline when collecting them.

## What to bring back

For the first Stage 3 schema check, place these files in the run directory:

```text
environment_report.json        # Colab strict preflight result
frames_meta.csv                # GPS/image correspondence used for the run
mask_report.json               # Stage 2 detections
vggt_output_schema.json        # required: actual prediction keys and shapes
stage3_stdout.log              # required: unedited Stage 3 console output
```

For a successful full pipeline, also include:

```text
timing_report.json             # required for the consolidated pipeline timing
vggt_predictions.npz           # preserves the fields consumed by Stage 4
aligned_points.npz             # Stage 4 result
exports/                       # every produced PLY/COLMAP/mesh/LAS/GeoTIFF file
accuracy_report.json           # only when real surveyed GCPs were supplied
artifact_manifest.txt          # `find . -type f | sort` from the run directory
```

`vggt_output_schema.json`, `stage3_stdout.log`, and `timing_report.json` are
small and must be included even if a later stage fails. `vggt_predictions.npz`
and export artifacts may be large, but retain them when transfer is possible:
they are the evidence required to reproduce any Stage 4–5 mismatch.

## Colab collection commands

After the commands in `docs/COLAB.md`, run the following in Colab. Replace the
example `RUN_ID` with the real dataset, GPU, and UTC start time. This copies
the documented Colab outputs into the handoff directory without changing their
filenames.

```bash
RUN_ID="brighton-beach-t4-20260922T174500Z"
HANDOFF_DIR="runs/colab/${RUN_ID}"
mkdir -p "$HANDOFF_DIR"

cp runs/colab-preflight/environment_report.json "$HANDOFF_DIR/"
cp runs/brighton/frames_meta.csv "$HANDOFF_DIR/"
cp runs/brighton/mask_report.json "$HANDOFF_DIR/" 2>/dev/null || true
cp runs/brighton/vggt_output_schema.json "$HANDOFF_DIR/" 2>/dev/null || true
cp runs/brighton/vggt_predictions.npz "$HANDOFF_DIR/" 2>/dev/null || true
cp runs/brighton/aligned_points.npz "$HANDOFF_DIR/" 2>/dev/null || true
cp runs/brighton/timing_report.json "$HANDOFF_DIR/" 2>/dev/null || true
cp runs/brighton/accuracy_report.json "$HANDOFF_DIR/" 2>/dev/null || true
cp -a runs/brighton/exports "$HANDOFF_DIR/" 2>/dev/null || true
find "$HANDOFF_DIR" -type f | sort > "$HANDOFF_DIR/artifact_manifest.txt"
```

Save the unedited Stage 3 notebook-cell output as
`$HANDOFF_DIR/stage3_stdout.log`, then download the entire `$HANDOFF_DIR`
folder and copy it into the same `runs/colab/` location in this checkout.

On receipt, compare the saved schema and NPZ fields with the exact Stage 4
assumptions (`extrinsics`, `intrinsics`, `world_points`, and
`world_points_conf`) before changing any adapter or exporter code. A key or
shape mismatch is a code fix; do not rename artifacts or fabricate NPZ fields
to make the current consumer pass.
