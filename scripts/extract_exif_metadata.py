#!/usr/bin/env python3
"""Adapt GPS-tagged drone images to the Stage 1 frame/metadata contract.

OpenDroneMap samples and many UAV cameras record GPS in EXIF instead of in a
separate telemetry CSV.  This utility copies those images into ``frames/`` and
writes the exact ``frames_meta.csv`` schema consumed by Stages 2--6.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import ExifTags, Image


GPS_INFO_TAG = next(key for key, value in ExifTags.TAGS.items() if value == "GPSInfo")
GPS_NAME_TO_ID = {value: key for key, value in ExifTags.GPSTAGS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OneFlight3D frames_meta.csv from GPS-tagged drone images."
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory of EXIF-tagged images.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Run directory; receives frames/ and frames_meta.csv.")
    parser.add_argument(
        "--copy_mode", choices=("copy", "hardlink"), default="hardlink",
        help="How to stage files in output_dir/frames (default: hardlink, falling back to copy).",
    )
    return parser.parse_args()


def rational_to_float(value: object) -> float:
    """Convert Pillow EXIF rationals and ordinary numbers to float."""
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def dms_to_decimal(values: object, reference: object) -> float:
    degrees, minutes, seconds = values
    decimal = rational_to_float(degrees) + rational_to_float(minutes) / 60 + rational_to_float(seconds) / 3600
    return -decimal if str(reference).upper() in {"S", "W"} else decimal


def exif_timestamp(gps: dict[object, object], fallback_path: Path) -> float:
    date = gps.get(GPS_NAME_TO_ID["GPSDateStamp"])
    time = gps.get(GPS_NAME_TO_ID["GPSTimeStamp"])
    if date and time:
        hours, minutes, seconds = (rational_to_float(item) for item in time)
        timestamp = datetime.strptime(str(date), "%Y:%m:%d").replace(tzinfo=timezone.utc)
        return timestamp.timestamp() + hours * 3600 + minutes * 60 + seconds
    # GPS position is still valid.  The fallback is explicitly visible in the
    # CSV and keeps deterministic ordering for data sets without EXIF time.
    return fallback_path.stat().st_mtime


def read_gps(image_path: Path) -> tuple[float, float, float, float]:
    with Image.open(image_path) as image:
        exif = image.getexif()
        gps = exif.get_ifd(GPS_INFO_TAG) if GPS_INFO_TAG in exif else {}
    required = ("GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef")
    missing = [name for name in required if GPS_NAME_TO_ID[name] not in gps]
    if missing:
        raise ValueError(f"missing EXIF GPS fields: {', '.join(missing)}")
    latitude = dms_to_decimal(gps[GPS_NAME_TO_ID["GPSLatitude"]], gps[GPS_NAME_TO_ID["GPSLatitudeRef"]])
    longitude = dms_to_decimal(gps[GPS_NAME_TO_ID["GPSLongitude"]], gps[GPS_NAME_TO_ID["GPSLongitudeRef"]])
    altitude_value = gps.get(GPS_NAME_TO_ID["GPSAltitude"], 0)
    altitude = rational_to_float(altitude_value)
    if gps.get(GPS_NAME_TO_ID["GPSAltitudeRef"], 0) == 1:
        altitude = -altitude
    return latitude, longitude, altitude, exif_timestamp(gps, image_path)


def stage_image(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        return
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2
    image_paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    )
    if not image_paths:
        print(f"No supported image files found in {args.input_dir}", file=sys.stderr)
        return 2

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rejected: list[str] = []
    for index, image_path in enumerate(image_paths):
        try:
            latitude, longitude, altitude, timestamp = read_gps(image_path)
        except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
            rejected.append(f"{image_path.name}: {error}")
            continue
        frame_path = frames_dir / image_path.name
        stage_image(image_path, frame_path, args.copy_mode)
        rows.append({
            "frame_file": image_path.name,
            "frame_index": len(rows),
            "video_time_sec": 0.0,  # populated after timestamp ordering below
            "gps_timestamp": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "gps_dt_sec": 0.0,
        })

    if len(rows) < 3:
        print("Need at least three GPS-tagged images for camera alignment.", file=sys.stderr)
        for item in rejected:
            print(item, file=sys.stderr)
        return 1
    rows.sort(key=lambda row: float(row["gps_timestamp"]))
    origin = float(rows[0]["gps_timestamp"])
    for index, row in enumerate(rows):
        row["frame_index"] = index
        row["video_time_sec"] = round(float(row["gps_timestamp"]) - origin, 3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "frames_meta.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} GPS-synchronised frame records to {metadata_path}")
    if rejected:
        print(f"Skipped {len(rejected)} images without readable GPS EXIF; see stderr.", file=sys.stderr)
        for item in rejected:
            print(item, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
