#!/usr/bin/env python3
"""
benchmark_accuracy.py — Accuracy Benchmarking Against Surveyed GCPs

Computes RMSE, mean error, max error, and per-axis error between the
aligned point cloud and independently surveyed Ground Control Points.

Input:
    --aligned_points  Path to aligned_points.npz (from Stage 4)
    --gcp_file        CSV of surveyed GCPs (columns: id, easting, northing, elevation)
    --output_report   Path for JSON accuracy report

Output:
    accuracy_report.json    Per-point and aggregate error statistics
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark reconstruction accuracy against surveyed GCPs"
    )
    parser.add_argument(
        "--aligned_points", required=True, type=str,
        help="Path to aligned_points.npz (from Stage 4)"
    )
    parser.add_argument(
        "--gcp_file", required=True, type=str,
        help="CSV of surveyed GCPs (columns: id, easting, northing, elevation)"
    )
    parser.add_argument(
        "--output_report", required=True, type=str,
        help="Path for JSON accuracy report"
    )
    parser.add_argument(
        "--search_radius", type=float, default=2.0,
        help="Max distance (m) to search for nearest reconstructed point (default: 2.0)"
    )
    return parser.parse_args()


def find_nearest_point(query, cloud):
    """Find the nearest point in cloud to query. Returns (distance, index)."""
    dists = np.linalg.norm(cloud - query, axis=1)
    idx = np.argmin(dists)
    return float(dists[idx]), int(idx)


def run_benchmark(args):
    """Main benchmarking logic."""

    # Load aligned points
    data = np.load(args.aligned_points)
    points = data["points"]  # (N, 3) in UTM/ENU
    log.info("Loaded %d reconstructed points", len(points))

    # Load GCPs
    gcp_df = pd.read_csv(args.gcp_file)
    required = {"id", "easting", "northing", "elevation"}
    missing = required - set(gcp_df.columns)
    if missing:
        log.error("GCP CSV missing columns: %s. Found: %s", missing, list(gcp_df.columns))
        sys.exit(1)

    log.info("Loaded %d GCPs", len(gcp_df))

    # Match each GCP to nearest reconstructed point
    results = []
    for _, row in gcp_df.iterrows():
        gcp = np.array([row["easting"], row["northing"], row["elevation"]])
        dist, idx = find_nearest_point(gcp, points)

        matched = points[idx]
        error_3d = dist
        error_east = matched[0] - gcp[0]
        error_north = matched[1] - gcp[1]
        error_elev = matched[2] - gcp[2]
        error_horiz = np.sqrt(error_east**2 + error_north**2)

        result = {
            "gcp_id": str(row["id"]),
            "gcp_easting": float(gcp[0]),
            "gcp_northing": float(gcp[1]),
            "gcp_elevation": float(gcp[2]),
            "matched_easting": float(matched[0]),
            "matched_northing": float(matched[1]),
            "matched_elevation": float(matched[2]),
            "error_3d_m": round(error_3d, 4),
            "error_horiz_m": round(error_horiz, 4),
            "error_vertical_m": round(abs(error_elev), 4),
            "within_search_radius": dist <= args.search_radius,
        }
        results.append(result)

        if dist > args.search_radius:
            log.warning("GCP %s: nearest point is %.2f m away (> %.1f m radius)",
                        row["id"], dist, args.search_radius)

    # Aggregate statistics
    valid = [r for r in results if r["within_search_radius"]]
    if not valid:
        log.error("No GCPs matched within search radius — cannot compute accuracy")
        report = {"error": "No GCPs matched", "per_gcp": results}
    else:
        errors_3d = np.array([r["error_3d_m"] for r in valid])
        errors_h = np.array([r["error_horiz_m"] for r in valid])
        errors_v = np.array([r["error_vertical_m"] for r in valid])

        report = {
            "num_gcps_total": len(results),
            "num_gcps_matched": len(valid),
            "search_radius_m": args.search_radius,
            "rmse_3d_m": round(float(np.sqrt(np.mean(errors_3d**2))), 4),
            "rmse_horizontal_m": round(float(np.sqrt(np.mean(errors_h**2))), 4),
            "rmse_vertical_m": round(float(np.sqrt(np.mean(errors_v**2))), 4),
            "mean_error_3d_m": round(float(np.mean(errors_3d)), 4),
            "max_error_3d_m": round(float(np.max(errors_3d)), 4),
            "median_error_3d_m": round(float(np.median(errors_3d)), 4),
            "per_gcp": results,
        }

        log.info("--- Accuracy Report ---")
        log.info("GCPs matched: %d / %d", len(valid), len(results))
        log.info("RMSE 3D:         %.4f m", report["rmse_3d_m"])
        log.info("RMSE horizontal: %.4f m", report["rmse_horizontal_m"])
        log.info("RMSE vertical:   %.4f m", report["rmse_vertical_m"])
        log.info("Mean error:      %.4f m", report["mean_error_3d_m"])
        log.info("Max error:       %.4f m", report["max_error_3d_m"])

    # Write report
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report written to %s", output_path)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
    log.info("Benchmark complete")
