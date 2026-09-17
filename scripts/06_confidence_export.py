#!/usr/bin/env python3
"""
Stage 6 — Confidence Overlay Export

Creates a confidence-colored point cloud where color encodes reconstruction
confidence: green = high confidence (well-observed), red = low confidence
(inferred/occluded). This is OneFlight3D's core "innovation" visual.

Input:
    --aligned_points  Path to aligned_points.npz (from Stage 4)
    --output_dir      Directory for output

Output:
    output_dir/confidence_cloud.ply    Confidence-colored point cloud
    output_dir/confidence_stats.json   Distribution statistics
"""

import argparse
import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("06_confidence_export")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 6: Export confidence-colored point cloud"
    )
    parser.add_argument(
        "--aligned_points", required=True, type=str,
        help="Path to aligned_points.npz (from Stage 4)"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str,
        help="Output directory"
    )
    parser.add_argument(
        "--colormap", type=str, default="rg",
        choices=["rg", "rygb"],
        help="Color mapping: rg = red→green, rygb = red→yellow→green→blue (default: rg)"
    )
    return parser.parse_args()


def confidence_to_rgb_rg(conf: np.ndarray) -> np.ndarray:
    """
    Map confidence [0, 1] → RGB color.
    0.0 = red (low confidence), 1.0 = green (high confidence).
    Linear interpolation in RGB space.
    """
    conf = np.clip(conf, 0, 1)
    r = ((1.0 - conf) * 255).astype(np.uint8)
    g = (conf * 255).astype(np.uint8)
    b = np.zeros_like(r)
    return np.stack([r, g, b], axis=-1)


def confidence_to_rgb_rygb(conf: np.ndarray) -> np.ndarray:
    """
    Map confidence [0, 1] → RGB via red→yellow→green→blue ramp.
    Gives more visual differentiation in the mid-range.
    """
    conf = np.clip(conf, 0, 1)
    r = np.zeros(len(conf), dtype=np.uint8)
    g = np.zeros(len(conf), dtype=np.uint8)
    b = np.zeros(len(conf), dtype=np.uint8)

    # 0.0–0.33: red → yellow
    mask = conf < 0.33
    t = conf[mask] / 0.33
    r[mask] = 255
    g[mask] = (t * 255).astype(np.uint8)

    # 0.33–0.66: yellow → green
    mask = (conf >= 0.33) & (conf < 0.66)
    t = (conf[mask] - 0.33) / 0.33
    r[mask] = ((1 - t) * 255).astype(np.uint8)
    g[mask] = 255

    # 0.66–1.0: green → blue
    mask = conf >= 0.66
    t = (conf[mask] - 0.66) / 0.34
    g[mask] = ((1 - t) * 255).astype(np.uint8)
    b[mask] = (t * 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def write_ply(points, colors, output_path):
    """Write binary PLY with colored vertices."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *points[i].astype(np.float32)))
            f.write(struct.pack("<BBB", *colors[i].astype(np.uint8)))


def run_confidence_export(args):
    """Main confidence export logic."""

    data_path = Path(args.aligned_points)
    if not data_path.exists():
        log.error("Aligned points not found: %s", data_path)
        sys.exit(1)

    log.info("Loading aligned points from %s", data_path)
    data = np.load(str(data_path))
    points = data["points"]
    confidence = data["confidence"]

    log.info("Loaded %d points", len(points))
    log.info("Confidence range: [%.4f, %.4f], mean: %.4f",
             confidence.min(), confidence.max(), confidence.mean())

    # Color by confidence
    if args.colormap == "rygb":
        colors = confidence_to_rgb_rygb(confidence)
    else:
        colors = confidence_to_rgb_rg(confidence)

    # Write PLY
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "confidence_cloud.ply"
    write_ply(points, colors, ply_path)
    log.info("Confidence cloud exported: %s (%d points)", ply_path, len(points))

    # Write statistics
    stats = {
        "num_points": int(len(points)),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "confidence_mean": float(confidence.mean()),
        "confidence_median": float(np.median(confidence)),
        "confidence_std": float(confidence.std()),
        "pct_high_conf": float((confidence >= 0.8).sum() / len(confidence) * 100),
        "pct_mid_conf": float(((confidence >= 0.4) & (confidence < 0.8)).sum() / len(confidence) * 100),
        "pct_low_conf": float((confidence < 0.4).sum() / len(confidence) * 100),
        "colormap": args.colormap,
    }

    stats_path = output_dir / "confidence_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Stats: %.1f%% high, %.1f%% mid, %.1f%% low confidence",
             stats["pct_high_conf"], stats["pct_mid_conf"], stats["pct_low_conf"])
    log.info("Statistics written to %s", stats_path)


if __name__ == "__main__":
    args = parse_args()
    run_confidence_export(args)
    log.info("Stage 6 complete")
