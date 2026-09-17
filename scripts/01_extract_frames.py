#!/usr/bin/env python3
"""
Stage 1 — Frame Extraction + GPS Sync

Extracts frames from a drone video at a configurable FPS, matches each frame
to the nearest GPS telemetry record by timestamp, and outputs a metadata CSV
linking frames to their geolocation.

Input:
    --video         Path to drone video file (MP4, MOV, etc.)
    --telemetry     Path to GPS telemetry CSV (columns: timestamp, latitude, longitude, altitude)
    --output_dir    Directory to write frames/ and frames_meta.csv

Output:
    output_dir/frames/frame_NNNN.jpg    Extracted frames
    output_dir/frames_meta.csv          Frame-to-GPS correspondence table
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("01_extract_frames")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Extract frames from drone video and sync with GPS telemetry"
    )
    parser.add_argument(
        "--video", required=True, type=str,
        help="Path to input drone video file"
    )
    parser.add_argument(
        "--telemetry", required=True, type=str,
        help="Path to GPS telemetry CSV (columns: timestamp, latitude, longitude, altitude)"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str,
        help="Output directory (will create frames/ subdirectory)"
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Frame extraction rate in frames per second (default: 2.0)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=300,
        help="Maximum number of frames to extract (default: 300)"
    )
    parser.add_argument(
        "--start_time", type=float, default=0.0,
        help="Start extracting from this video time in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--image_format", type=str, default="jpg", choices=["jpg", "png"],
        help="Output image format (default: jpg)"
    )
    parser.add_argument(
        "--jpg_quality", type=int, default=95,
        help="JPEG quality 0-100 (default: 95)"
    )
    return parser.parse_args()


def load_telemetry(telemetry_path: str) -> pd.DataFrame:
    """
    Load GPS telemetry CSV. Expected columns: timestamp, latitude, longitude, altitude.
    Timestamp should be Unix epoch seconds (float).
    """
    df = pd.read_csv(telemetry_path)

    # Validate required columns
    required_cols = {"timestamp", "latitude", "longitude", "altitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Telemetry CSV missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Sort by timestamp for binary search
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic validation
    if df["timestamp"].isna().any():
        raise ValueError("Telemetry CSV contains NaN timestamps")
    if len(df) < 2:
        raise ValueError(f"Telemetry CSV has only {len(df)} rows — need at least 2")

    log.info(
        "Loaded telemetry: %d records, time range %.1f–%.1f (%.1f sec)",
        len(df),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
        df["timestamp"].iloc[-1] - df["timestamp"].iloc[0],
    )
    return df


def find_nearest_gps(frame_time_sec: float, telemetry_df: pd.DataFrame,
                     video_start_timestamp: float) -> dict:
    """
    Find the GPS record with the closest timestamp to the given frame time.

    frame_time_sec: seconds since video start
    video_start_timestamp: Unix epoch timestamp of the first telemetry record
                          (used to convert video time to absolute time)
    """
    # Convert video-relative time to absolute timestamp
    absolute_time = video_start_timestamp + frame_time_sec

    # Binary search for nearest timestamp
    timestamps = telemetry_df["timestamp"].values
    idx = np.searchsorted(timestamps, absolute_time)

    # Check neighbors to find the actual nearest
    if idx == 0:
        nearest_idx = 0
    elif idx >= len(timestamps):
        nearest_idx = len(timestamps) - 1
    else:
        # Compare distance to left and right neighbors
        if abs(timestamps[idx - 1] - absolute_time) <= abs(timestamps[idx] - absolute_time):
            nearest_idx = idx - 1
        else:
            nearest_idx = idx

    row = telemetry_df.iloc[nearest_idx]
    dt = abs(row["timestamp"] - absolute_time)

    return {
        "gps_timestamp": row["timestamp"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "altitude": row["altitude"],
        "gps_dt_sec": dt,
    }


def extract_frames(args):
    """Main frame extraction and GPS sync logic."""

    video_path = Path(args.video)
    telemetry_path = Path(args.telemetry)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        log.error("Video file not found: %s", video_path)
        sys.exit(1)
    if not telemetry_path.exists():
        log.error("Telemetry file not found: %s", telemetry_path)
        sys.exit(1)

    # Create output directories
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load telemetry
    telemetry_df = load_telemetry(str(telemetry_path))
    video_start_timestamp = telemetry_df["timestamp"].iloc[0]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames_in_video / video_fps if video_fps > 0 else 0

    log.info(
        "Video: %.1f fps, %d frames, %.1f sec duration",
        video_fps, total_frames_in_video, video_duration,
    )

    # Calculate frame extraction interval
    frame_interval = 1.0 / args.fps  # seconds between extracted frames
    frame_interval_video_frames = int(video_fps / args.fps)

    log.info(
        "Extracting at %.1f fps (every %d video frames), max %d frames",
        args.fps, frame_interval_video_frames, args.max_frames,
    )

    # Skip to start time
    if args.start_time > 0:
        start_frame = int(args.start_time * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        log.info("Skipping to %.1f sec (frame %d)", args.start_time, start_frame)

    # Extract frames
    metadata_rows = []
    extracted_count = 0
    frame_idx = 0
    gps_warnings = 0

    # Calculate total expected frames for progress bar
    remaining_duration = video_duration - args.start_time
    expected_frames = min(
        int(remaining_duration * args.fps),
        args.max_frames,
    )

    pbar = tqdm(total=expected_frames, desc="Extracting frames", unit="frame")

    while extracted_count < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Check if this frame should be extracted (based on target FPS)
        if frame_idx % max(1, frame_interval_video_frames) != 0:
            frame_idx += 1
            continue

        # Video time relative to start
        video_time_sec = current_time - args.start_time

        # Find nearest GPS record
        gps_match = find_nearest_gps(video_time_sec, telemetry_df, video_start_timestamp)

        # Warn if GPS match is poor
        if gps_match["gps_dt_sec"] > 1.0:
            if gps_warnings < 5:
                log.warning(
                    "Frame %04d: nearest GPS record is %.2f sec away (>1s threshold)",
                    extracted_count, gps_match["gps_dt_sec"],
                )
            gps_warnings += 1

        # Save frame
        frame_filename = f"frame_{extracted_count:04d}.{args.image_format}"
        frame_path = frames_dir / frame_filename

        if args.image_format == "jpg":
            cv2.imwrite(str(frame_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
        else:
            cv2.imwrite(str(frame_path), frame)

        # Record metadata
        metadata_rows.append({
            "frame_file": frame_filename,
            "frame_index": extracted_count,
            "video_time_sec": round(video_time_sec, 3),
            "gps_timestamp": gps_match["gps_timestamp"],
            "latitude": gps_match["latitude"],
            "longitude": gps_match["longitude"],
            "altitude": gps_match["altitude"],
            "gps_dt_sec": round(gps_match["gps_dt_sec"], 4),
        })

        extracted_count += 1
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if gps_warnings > 5:
        log.warning(
            "Total frames with GPS match >1s: %d/%d",
            gps_warnings, extracted_count,
        )

    # Write metadata CSV
    meta_path = output_dir / "frames_meta.csv"
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(meta_path, index=False)

    log.info("Extracted %d frames to %s", extracted_count, frames_dir)
    log.info("Metadata written to %s", meta_path)

    # Summary statistics
    if metadata_rows:
        dt_values = [r["gps_dt_sec"] for r in metadata_rows]
        log.info(
            "GPS match quality — mean dt: %.3f sec, max dt: %.3f sec, "
            "%d/%d frames within 1s",
            np.mean(dt_values),
            np.max(dt_values),
            sum(1 for dt in dt_values if dt <= 1.0),
            len(dt_values),
        )

    return extracted_count


if __name__ == "__main__":
    args = parse_args()
    n = extract_frames(args)
    if n == 0:
        log.error("No frames extracted — check video file and parameters")
        sys.exit(1)
    log.info("Stage 1 complete: %d frames extracted and GPS-synced", n)
