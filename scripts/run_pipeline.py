#!/usr/bin/env python3
"""
run_pipeline.py — OneFlight3D End-to-End Pipeline Orchestrator

Runs the full six-stage pipeline from drone video + GPS telemetry to
all required output formats. Times each stage and writes a timing report.

Usage:
    python scripts/run_pipeline.py \\
        --video path/to/drone_video.mp4 \\
        --telemetry path/to/gps_telemetry.csv \\
        --output_dir output/
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OneFlight3D: Full pipeline from drone video to 3D exports"
    )
    parser.add_argument(
        "--video", required=True, type=str,
        help="Path to input drone video file"
    )
    parser.add_argument(
        "--telemetry", required=True, type=str,
        help="Path to GPS telemetry CSV"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str,
        help="Output directory for all pipeline artifacts"
    )
    # Stage 1 options
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frame extraction rate (default: 2.0)")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Max frames to extract (default: 300)")
    # Stage 3 options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B",
                        help="VGGT model name (default: facebook/VGGT-1B)")
    # Stage 4 options
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for point filtering (default: 0.5)")
    parser.add_argument("--coordinate_system", type=str, default="utm",
                        choices=["enu", "utm"],
                        help="Output coordinate system (default: utm)")
    # Stage 5 options
    parser.add_argument("--formats", type=str, default="all",
                        help="Export formats, comma-separated or 'all' (default: all)")
    # Skip stages
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated stage numbers to skip (e.g., '2,6')")
    return parser.parse_args()


def run_stage(name: str, cmd: list, timing: dict) -> bool:
    """Run a pipeline stage as a subprocess. Returns True on success."""
    log.info("=" * 60)
    log.info("STAGE: %s", name)
    log.info("CMD: %s", " ".join(cmd))
    log.info("=" * 60)

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    elapsed = time.time() - start
    timing[name] = round(elapsed, 2)

    if result.returncode != 0:
        log.error("STAGE FAILED: %s (exit code %d, %.1f sec)",
                  name, result.returncode, elapsed)
        return False

    log.info("STAGE DONE: %s (%.1f sec)", name, elapsed)
    return True


def main():
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip = set()
    if args.skip:
        skip = {int(s.strip()) for s in args.skip.split(",")}
        log.info("Skipping stages: %s", skip)

    python = sys.executable
    timing = {}
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1: Frame Extraction + GPS Sync
    # ------------------------------------------------------------------
    if 1 not in skip:
        ok = run_stage("1_extract_frames", [
            python, str(scripts_dir / "01_extract_frames.py"),
            "--video", args.video,
            "--telemetry", args.telemetry,
            "--output_dir", str(output_dir),
            "--fps", str(args.fps),
            "--max_frames", str(args.max_frames),
        ], timing)
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 2: Dynamic Object Masking
    # ------------------------------------------------------------------
    if 2 not in skip:
        ok = run_stage("2_mask_dynamic_objects", [
            python, str(scripts_dir / "02_mask_dynamic_objects.py"),
            "--input_dir", str(output_dir / "frames"),
            "--output_dir", str(output_dir / "frames_masked"),
        ], timing)
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 3: VGGT Reconstruction
    # ------------------------------------------------------------------
    if 3 not in skip:
        ok = run_stage("3_run_vggt", [
            python, str(scripts_dir / "03_run_vggt.py"),
            "--input_dir", str(output_dir / "frames_masked"),
            "--output_file", str(output_dir / "vggt_predictions.npz"),
            "--device", args.device,
            "--model_name", args.model_name,
            "--max_frames", str(args.max_frames),
        ], timing)
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 4: GPS / Metric Alignment
    # ------------------------------------------------------------------
    if 4 not in skip:
        ok = run_stage("4_gps_alignment", [
            python, str(scripts_dir / "04_gps_alignment.py"),
            "--predictions", str(output_dir / "vggt_predictions.npz"),
            "--frames_meta", str(output_dir / "frames_meta.csv"),
            "--output_file", str(output_dir / "aligned_points.npz"),
            "--conf_threshold", str(args.conf_threshold),
            "--coordinate_system", args.coordinate_system,
        ], timing)
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 5: Export All Formats
    # ------------------------------------------------------------------
    if 5 not in skip:
        cmd = [
            python, str(scripts_dir / "05_export_formats.py"),
            "--aligned_points", str(output_dir / "aligned_points.npz"),
            "--output_dir", str(output_dir / "exports"),
            "--formats", args.formats,
        ]
        frames_meta = output_dir / "frames_meta.csv"
        if frames_meta.exists():
            cmd.extend(["--frames_meta", str(frames_meta)])
        ok = run_stage("5_export_formats", cmd, timing)
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 6: Confidence Overlay
    # ------------------------------------------------------------------
    if 6 not in skip:
        ok = run_stage("6_confidence_export", [
            python, str(scripts_dir / "06_confidence_export.py"),
            "--aligned_points", str(output_dir / "aligned_points.npz"),
            "--output_dir", str(output_dir / "exports"),
        ], timing)
        if not ok:
            sys.exit(1)

    # ------------------------------------------------------------------
    # Timing report
    # ------------------------------------------------------------------
    total = time.time() - pipeline_start
    timing["total"] = round(total, 2)

    report = {
        "timing_seconds": timing,
        "total_seconds": round(total, 2),
        "total_minutes": round(total / 60, 2),
        "args": {
            "video": args.video,
            "telemetry": args.telemetry,
            "output_dir": str(output_dir),
            "fps": args.fps,
            "max_frames": args.max_frames,
            "device": args.device,
            "model_name": args.model_name,
            "conf_threshold": args.conf_threshold,
            "coordinate_system": args.coordinate_system,
            "formats": args.formats,
        },
    }

    report_path = output_dir / "timing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("Total time: %.1f sec (%.1f min)", total, total / 60)
    for stage, secs in timing.items():
        if stage != "total":
            log.info("  %-30s %6.1f sec", stage, secs)
    log.info("Timing report: %s", report_path)
    log.info("Exports: %s", output_dir / "exports")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
