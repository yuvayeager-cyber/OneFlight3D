#!/usr/bin/env python3
"""
Stage 3 — VGGT Reconstruction (Core Engine)

Runs VGGT (Visual Geometry Grounded Transformer) inference on masked frames
to produce camera extrinsics, intrinsics, dense depth maps, and 3D point maps.

THIS IS THE HIGHEST-RISK UNVERIFIED STEP — see CLAUDE.md.
The script FIRST prints predictions.keys() and all tensor shapes before
doing anything else. If the real output differs from assumptions, fix
downstream code and update CLAUDE.md.

Input:
    --input_dir     Directory containing masked frames (from Stage 2)
    --output_file   Path to write predictions .npz

Output:
    vggt_predictions.npz    Contains: world_points, world_points_conf,
                            extrinsics, intrinsics, depth_maps, depth_conf,
                            images, plus the raw pose_enc
    vggt_output_schema.json Schema documenting actual key names and shapes
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("03_run_vggt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: Run VGGT reconstruction on masked frames"
    )
    parser.add_argument(
        "--input_dir", required=True, type=str,
        help="Directory containing masked frames (from Stage 2)"
    )
    parser.add_argument(
        "--output_file", required=True, type=str,
        help="Output path for predictions .npz file"
    )
    parser.add_argument(
        "--max_frames", type=int, default=300,
        help="Maximum frames to process (default: 300)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference: cuda or cpu (default: cuda)"
    )
    parser.add_argument(
        "--model_name", type=str, default="facebook/VGGT-1B",
        help="HuggingFace model name or local path (default: facebook/VGGT-1B)"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Inference precision (default: auto — bfloat16 if Ampere+, else float16)"
    )
    parser.add_argument(
        "--schema_file", type=str, default=None,
        help="Path for output schema JSON (default: alongside output_file)"
    )
    return parser.parse_args()


def get_image_paths(input_dir: str, max_frames: int) -> list:
    """Find and sort all image files in the input directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    input_path = Path(input_dir)

    files = sorted([
        str(f) for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not files:
        log.error("No image files found in %s", input_dir)
        sys.exit(1)

    if len(files) > max_frames:
        log.info("Limiting to %d frames (found %d)", max_frames, len(files))
        files = files[:max_frames]

    log.info("Found %d frames for VGGT inference", len(files))
    return files


def log_prediction_schema(predictions: dict, schema_path: str):
    """
    CRITICAL: Log the actual VGGT output key names and tensor shapes.
    This is the verification step flagged in CLAUDE.md — do not skip.
    """
    schema = {}

    log.info("=" * 60)
    log.info("VGGT OUTPUT SCHEMA (verify these against downstream code):")
    log.info("=" * 60)

    for key in sorted(predictions.keys()):
        val = predictions[key]
        if hasattr(val, "shape"):
            info = {
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "min": float(val.min()) if val.numel() > 0 else None,
                "max": float(val.max()) if val.numel() > 0 else None,
            }
        elif isinstance(val, np.ndarray):
            info = {
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "min": float(val.min()) if val.size > 0 else None,
                "max": float(val.max()) if val.size > 0 else None,
            }
        else:
            info = {"type": str(type(val).__name__), "value": str(val)[:200]}

        schema[key] = info
        log.info("  %-25s %s", key, info)

    log.info("=" * 60)

    # Save schema to JSON
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    log.info("Schema saved to %s", schema_path)

    return schema


def run_vggt_inference(args):
    """
    Run VGGT model inference.

    The model workflow (from VGGT docs):
    1. Load model via VGGT.from_pretrained()
    2. Load & preprocess images via load_and_preprocess_images()
    3. Forward pass → raw predictions dict containing pose_enc, depth, etc.
    4. Decode pose_enc → extrinsics + intrinsics via pose_encoding_to_extri_intri()
    5. Optionally unproject depth → 3D point maps

    TODO/UNVERIFIED: This code has not been run on a GPU with real data.
    The import structure and key names are based on the VGGT repository
    (facebookresearch/vggt) as of 2025. If the installed version differs,
    the schema logging step (above) will reveal the actual structure.
    """

    # -------------------------------------------------------------------------
    # Step 0: Attempt to import VGGT
    # -------------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        log.error("PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as e:
        log.error(
            "VGGT package not installed or importable: %s\n"
            "Install with: pip install vggt\n"
            "Or from source: git clone https://github.com/facebookresearch/vggt "
            "&& cd vggt && pip install -e .",
            e,
        )
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 1: Setup device and precision
    # -------------------------------------------------------------------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU (will be very slow)")
        device = "cpu"

    if args.dtype == "auto":
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map[args.dtype]

    log.info("Device: %s, dtype: %s", device, dtype)

    # -------------------------------------------------------------------------
    # Step 2: Load model
    # -------------------------------------------------------------------------
    log.info("Loading VGGT model: %s", args.model_name)
    model = VGGT.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    log.info("Model loaded successfully")

    # -------------------------------------------------------------------------
    # Step 3: Load and preprocess images
    # -------------------------------------------------------------------------
    image_paths = get_image_paths(args.input_dir, args.max_frames)
    log.info("Loading and preprocessing %d images...", len(image_paths))
    images = load_and_preprocess_images(image_paths).to(device)
    log.info("Images tensor shape: %s", images.shape)

    # -------------------------------------------------------------------------
    # Step 4: Run inference
    # -------------------------------------------------------------------------
    log.info("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # -------------------------------------------------------------------------
    # Step 5: CRITICAL — Log actual output schema before doing anything else
    # -------------------------------------------------------------------------
    schema_path = args.schema_file
    if schema_path is None:
        schema_path = str(Path(args.output_file).parent / "vggt_output_schema.json")

    schema = log_prediction_schema(predictions, schema_path)

    # -------------------------------------------------------------------------
    # Step 6: Decode pose encodings → extrinsics + intrinsics
    # -------------------------------------------------------------------------
    # VGGT's raw output contains 'pose_enc' which must be decoded.
    # pose_encoding_to_extri_intri(pose_enc, image_size) → (extrinsics, intrinsics)
    #   extrinsics: (S, 3, 4) — [R | t] per frame
    #   intrinsics: (S, 3, 3) — camera matrix per frame

    if "pose_enc" in predictions:
        log.info("Decoding pose encodings → extrinsics + intrinsics...")
        # Image size from the images tensor (H, W)
        image_h, image_w = images.shape[-2], images.shape[-1]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], (image_h, image_w)
        )
        log.info("Extrinsics shape: %s", extrinsics.shape)
        log.info("Intrinsics shape: %s", intrinsics.shape)
    elif "extrinsic" in predictions:
        # Some versions may directly output decoded matrices
        extrinsics = predictions["extrinsic"]
        intrinsics = predictions["intrinsic"]
        log.info("Using pre-decoded extrinsics/intrinsics from model output")
    else:
        log.error(
            "Cannot find pose data in predictions. "
            "Available keys: %s", list(predictions.keys())
        )
        log.error(
            "This is the unverified-key-names issue flagged in CLAUDE.md. "
            "Update the code to use the correct key names from the schema above."
        )
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 7: Extract point maps and confidence
    # -------------------------------------------------------------------------
    # Expected keys (from research): world_points (S,H,W,3), world_points_conf (S,H,W)
    # Also: depth_map (S,H,W,1), depth_conf (S,H,W)

    world_points = None
    world_points_conf = None
    depth_maps = None
    depth_conf = None

    # Try known key names — adapt based on what the schema reveals
    for key_candidate in ["world_points", "point_map", "world_points_3d"]:
        if key_candidate in predictions:
            world_points = predictions[key_candidate]
            log.info("World points found as '%s': shape %s", key_candidate, world_points.shape)
            break

    for key_candidate in ["world_points_conf", "point_conf", "conf"]:
        if key_candidate in predictions:
            world_points_conf = predictions[key_candidate]
            log.info("Point confidence found as '%s': shape %s", key_candidate, world_points_conf.shape)
            break

    for key_candidate in ["depth_map", "depth", "depth_maps"]:
        if key_candidate in predictions:
            depth_maps = predictions[key_candidate]
            log.info("Depth maps found as '%s': shape %s", key_candidate, depth_maps.shape)
            break

    for key_candidate in ["depth_conf", "depth_confidence"]:
        if key_candidate in predictions:
            depth_conf = predictions[key_candidate]
            log.info("Depth confidence found as '%s': shape %s", key_candidate, depth_conf.shape)
            break

    if world_points is None:
        log.warning(
            "No world_points/point_map found in predictions. "
            "Will need to unproject from depth maps if available."
        )
        if depth_maps is not None and "unproject_depth_map_to_point_map" in dir():
            log.info("Attempting to unproject depth maps to point maps...")
            # TODO/UNVERIFIED: Uncomment and verify import path
            # from vggt.utils.geometry import unproject_depth_map_to_point_map
            # world_points = unproject_depth_map_to_point_map(depth_maps, intrinsics)

    # -------------------------------------------------------------------------
    # Step 8: Save predictions to .npz
    # -------------------------------------------------------------------------
    log.info("Saving predictions to %s", args.output_file)

    save_dict = {}

    # Convert tensors to numpy for .npz storage
    def to_numpy(tensor_or_array):
        if hasattr(tensor_or_array, "cpu"):
            return tensor_or_array.cpu().float().numpy()
        return np.array(tensor_or_array)

    save_dict["extrinsics"] = to_numpy(extrinsics)      # (S, 3, 4)
    save_dict["intrinsics"] = to_numpy(intrinsics)      # (S, 3, 3)

    if world_points is not None:
        save_dict["world_points"] = to_numpy(world_points)      # (S, H, W, 3)
    if world_points_conf is not None:
        save_dict["world_points_conf"] = to_numpy(world_points_conf)  # (S, H, W)
    if depth_maps is not None:
        save_dict["depth_maps"] = to_numpy(depth_maps)          # (S, H, W, 1)
    if depth_conf is not None:
        save_dict["depth_conf"] = to_numpy(depth_conf)          # (S, H, W)

    # Save the input images for color extraction later
    save_dict["images"] = to_numpy(images.squeeze(0) if images.dim() == 5 else images)

    # Store image file paths for metadata linkage
    image_paths_list = get_image_paths(args.input_dir, args.max_frames)
    # Can't store strings in .npz directly — save separately
    save_dict["num_frames"] = np.array([len(image_paths_list)])

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **save_dict)

    # Also save image paths list alongside
    paths_file = output_path.parent / "vggt_image_paths.json"
    with open(paths_file, "w") as f:
        json.dump(image_paths_list, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Predictions saved: %.1f MB", file_size_mb)
    log.info("Image paths saved to %s", paths_file)

    # Print summary
    log.info("--- VGGT Output Summary ---")
    for key, arr in save_dict.items():
        if isinstance(arr, np.ndarray):
            log.info("  %-25s shape=%-20s dtype=%s", key, arr.shape, arr.dtype)
    log.info("---------------------------")

    return str(output_path)


if __name__ == "__main__":
    args = parse_args()
    output = run_vggt_inference(args)
    log.info("Stage 3 complete: predictions saved to %s", output)
