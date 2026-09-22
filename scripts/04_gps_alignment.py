#!/usr/bin/env python3
"""
Stage 4 — GPS / Metric Alignment (Umeyama)

Aligns VGGT's arbitrary-scale reconstruction to real-world GPS coordinates
using a closed-form similarity transform (Umeyama algorithm).

Input:
    --predictions   Path to vggt_predictions.npz (from Stage 3)
    --frames_meta   Path to frames_meta.csv (from Stage 1)
    --output_file   Path to write aligned_points.npz

Output:
    aligned_points.npz      Points, colors, confidence, extrinsics, intrinsics,
                            transform params, alignment metadata
    alignment_meta.json     Alignment diagnostics (RMSE, scale, UTM zone, etc.)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.geo_utils import (
    wgs84_to_enu,
    wgs84_to_utm,
    umeyama_similarity,
    apply_transform,
    check_spatial_spread,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("04_gps_alignment")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4: Align VGGT reconstruction to GPS coordinates"
    )
    parser.add_argument(
        "--predictions", required=True, type=str,
        help="Path to vggt_predictions.npz (from Stage 3)"
    )
    parser.add_argument(
        "--frames_meta", required=True, type=str,
        help="Path to frames_meta.csv (from Stage 1)"
    )
    parser.add_argument(
        "--output_file", required=True, type=str,
        help="Output path for aligned_points.npz"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.5,
        help="Minimum confidence to keep a point (default: 0.5)"
    )
    parser.add_argument(
        "--max_points", type=int, default=5_000_000,
        help="Maximum points to retain after filtering (default: 5M)"
    )
    parser.add_argument(
        "--coordinate_system", type=str, default="utm",
        choices=["enu", "utm"],
        help="Output coordinate system (default: utm)"
    )
    return parser.parse_args()


def extract_camera_positions(extrinsics: np.ndarray) -> np.ndarray:
    """
    Extract camera world positions from [R|t] extrinsic matrices.

    Assumes world-to-camera convention: extrinsics maps world → camera.
    Camera center in world frame = -R^T @ t

    Args:
        extrinsics: (S, 3, 4) array of [R|t] matrices

    Returns:
        positions: (S, 3) camera positions in VGGT's coordinate system
    """
    S = extrinsics.shape[0]
    positions = np.zeros((S, 3))
    for i in range(S):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        positions[i] = -R.T @ t
    return positions


def transform_extrinsics(extrinsics: np.ndarray, s: float,
                         R_sim: np.ndarray, t_sim: np.ndarray) -> np.ndarray:
    """
    Transform world-to-camera extrinsics into the aligned coordinate system.

    Given similarity transform p_aligned = s * R_sim @ p_orig + t_sim,
    the new extrinsics satisfy: p_cam_new = s * p_cam_orig (scale cancels
    in perspective division, so intrinsics stay unchanged).

    Formulas:
        c_aligned = s * R_sim @ c_orig + t_sim
        R_cam_new = R_cam @ R_sim^T
        t_cam_new = -R_cam_new @ c_aligned
    """
    S = extrinsics.shape[0]
    out = np.zeros_like(extrinsics)
    for i in range(S):
        R_cam = extrinsics[i, :3, :3]
        t_cam = extrinsics[i, :3, 3]
        c_orig = -R_cam.T @ t_cam
        c_aligned = s * R_sim @ c_orig + t_sim
        R_cam_new = R_cam @ R_sim.T
        t_cam_new = -R_cam_new @ c_aligned
        out[i, :3, :3] = R_cam_new
        out[i, :3, 3] = t_cam_new
    return out


def load_colors(pred: dict, pred_path: Path, H: int, W: int, S: int) -> np.ndarray:
    """
    Load per-pixel RGB colors matching the VGGT output resolution.
    Tries original frames first, falls back to stored images.

    Returns (S, H, W, 3) uint8.
    """
    # Try original frame paths
    paths_file = pred_path.parent / "vggt_image_paths.json"
    if paths_file.exists():
        with open(paths_file) as f:
            frame_paths = json.load(f)
        log.info("Loading colors from %d original frames...", len(frame_paths))
        colors = np.zeros((S, H, W, 3), dtype=np.uint8)
        for i, fp in enumerate(frame_paths[:S]):
            img = cv2.imread(fp)
            if img is None:
                log.warning("Cannot read frame %s — black fill", fp)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != H or img.shape[1] != W:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            colors[i] = img
        return colors

    # Fallback: stored images from predictions
    if "images" in pred:
        log.warning("Frame paths not found — using stored images (may be normalized)")
        imgs = pred["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)  # (S,3,H,W) → (S,H,W,3)
        imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
        if imgs.shape[1:3] != (H, W):
            log.warning("Image shape %s doesn't match points (%d,%d) — resizing", imgs.shape, H, W)
            resized = np.zeros((S, H, W, 3), dtype=np.uint8)
            for i in range(min(S, len(imgs))):
                resized[i] = cv2.resize(imgs[i], (W, H))
            return resized
        return imgs

    log.warning("No color source available — using grey fill")
    return np.full((S, H, W, 3), 128, dtype=np.uint8)


def run_alignment(args):
    """Main alignment logic."""

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        log.error("Predictions file not found: %s", pred_path)
        sys.exit(1)

    log.info("Loading predictions from %s", pred_path)
    pred = np.load(str(pred_path))
    log.info("Prediction keys: %s", list(pred.keys()))

    extrinsics = pred["extrinsics"]   # (S, 3, 4)
    intrinsics = pred["intrinsics"]   # (S, 3, 3)
    S = extrinsics.shape[0]
    log.info("Loaded %d frames", S)

    if "world_points" not in pred:
        log.error("world_points not found in predictions. Keys: %s", list(pred.keys()))
        sys.exit(1)

    world_points = pred["world_points"]   # (S, H, W, 3)
    _, H, W, _ = world_points.shape
    log.info("World points shape: %s", world_points.shape)

    confidence = (
        pred["world_points_conf"] if "world_points_conf" in pred
        else np.ones(world_points.shape[:3], dtype=np.float32)
    )
    if "world_points_conf" not in pred:
        log.warning("No confidence data — treating all points as high confidence")

    meta_df = pd.read_csv(args.frames_meta)
    if len(meta_df) != S:
        log.error("Frame count mismatch: predictions %d vs metadata %d", S, len(meta_df))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Camera positions from VGGT extrinsics
    # ------------------------------------------------------------------
    cam_pos_vggt = extract_camera_positions(extrinsics)
    log.info("VGGT camera positions — min: %s, max: %s",
             cam_pos_vggt.min(axis=0).round(3), cam_pos_vggt.max(axis=0).round(3))

    # ------------------------------------------------------------------
    # 2. GPS → local metric coordinates
    # ------------------------------------------------------------------
    lat = meta_df["latitude"].values
    lon = meta_df["longitude"].values
    alt = meta_df["altitude"].values
    ref_lat, ref_lon, ref_alt = float(lat[0]), float(lon[0]), float(alt[0])

    utm_zone, utm_crs_definition, utm_epsg = None, "", None
    if args.coordinate_system == "enu":
        gps_local = wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
    else:
        gps_local, utm_zone, utm_crs_definition = wgs84_to_utm(lat, lon, alt)
        # Preserve the exact CRS chosen upstream.  Exporters must never infer
        # a hemisphere from an elevation or an arbitrary point-cloud axis.
        utm_epsg = (32600 if float(np.mean(lat)) >= 0 else 32700) + utm_zone

    log.info("GPS local — min: %s, max: %s",
             gps_local.min(axis=0).round(3), gps_local.max(axis=0).round(3))

    # ------------------------------------------------------------------
    # 3. Validate spatial distribution
    # ------------------------------------------------------------------
    is_valid, spread_info = check_spatial_spread(gps_local)
    if not is_valid:
        log.warning("Spatial distribution issues — alignment may be unreliable")

    # ------------------------------------------------------------------
    # 4. Umeyama similarity transform
    # ------------------------------------------------------------------
    s, R, t, rmse = umeyama_similarity(cam_pos_vggt, gps_local)
    log.info("Alignment — scale: %.6f, RMSE: %.4f m", s, rmse)
    if rmse > 5.0:
        log.warning("RMSE %.2f m (>5m) — check data quality", rmse)

    # ------------------------------------------------------------------
    # 5. Transform world points and extrinsics
    # ------------------------------------------------------------------
    log.info("Applying similarity transform to world points...")
    points_aligned = apply_transform(world_points, s, R, t)  # (S,H,W,3)
    extrinsics_aligned = transform_extrinsics(extrinsics, s, R, t)  # (S,3,4)

    # ------------------------------------------------------------------
    # 6. Load colors
    # ------------------------------------------------------------------
    colors_4d = load_colors(dict(pred), Path(args.predictions), H, W, S)

    # ------------------------------------------------------------------
    # 7. Flatten, filter by confidence, subsample
    # ------------------------------------------------------------------
    points_flat = points_aligned.reshape(-1, 3)
    conf_flat = confidence.reshape(-1)
    colors_flat = colors_4d.reshape(-1, 3)

    mask = (conf_flat >= args.conf_threshold) & np.isfinite(points_flat).all(axis=1)
    points_f = points_flat[mask]
    conf_f = conf_flat[mask]
    colors_f = colors_flat[mask]

    log.info("Points: %d total → %d after filter (%.1f%%)",
             len(points_flat), len(points_f),
             100 * len(points_f) / max(len(points_flat), 1))

    if len(points_f) > args.max_points:
        log.info("Subsampling %d → %d points", len(points_f), args.max_points)
        idx = np.random.default_rng(42).choice(len(points_f), args.max_points, replace=False)
        idx.sort()
        points_f, conf_f, colors_f = points_f[idx], conf_f[idx], colors_f[idx]

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "points": points_f.astype(np.float64),
        "colors": colors_f.astype(np.uint8),
        "confidence": conf_f.astype(np.float32),
        "extrinsics": extrinsics_aligned.astype(np.float64),
        "intrinsics": intrinsics.astype(np.float64),
        "transform_s": np.array([s]),
        "transform_R": R,
        "transform_t": t,
        "alignment_rmse": np.array([rmse]),
        "ref_lat": np.array([ref_lat]),
        "ref_lon": np.array([ref_lon]),
        "ref_alt": np.array([ref_alt]),
    }
    if utm_zone is not None:
        save_dict["utm_zone"] = np.array([utm_zone])
        save_dict["utm_epsg"] = np.array([utm_epsg], dtype=np.int32)
        save_dict["crs_definition"] = np.array([utm_crs_definition])

    np.savez_compressed(str(output_path), **save_dict)
    file_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Saved: %s (%.1f MB, %d points)", output_path, file_mb, len(points_f))

    # Alignment metadata sidecar
    meta_json = {
        "num_points": int(len(points_f)),
        "num_frames": int(S),
        "alignment_rmse_m": float(rmse),
        "transform_scale": float(s),
        "conf_threshold": args.conf_threshold,
        "coordinate_system": args.coordinate_system,
        "ref_lat": ref_lat, "ref_lon": ref_lon, "ref_alt": ref_alt,
        "spatial_spread": spread_info,
    }
    if utm_zone is not None:
        meta_json["utm_zone"] = utm_zone
        meta_json["utm_epsg"] = utm_epsg
        meta_json["utm_crs_definition"] = utm_crs_definition

    meta_path = output_path.parent / "alignment_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)
    log.info("Metadata written to %s", meta_path)

    return str(output_path)


if __name__ == "__main__":
    args = parse_args()
    output = run_alignment(args)
    log.info("Stage 4 complete: %s", output)
