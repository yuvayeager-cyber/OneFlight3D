#!/usr/bin/env python3
"""Generate a tiny non-benchmark data set for offline pipeline smoke tests.

The generated prediction archive is deliberately labelled synthetic and must
never be used to claim reconstruction accuracy or speed on real UAV data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OneFlight3D synthetic smoke-test inputs.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 3 or args.width < 2 or args.height < 2:
        raise SystemExit("Use at least 3 frames and a 2x2 image grid.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, args.width), np.linspace(-1, 1, args.height))
    world_points, confidences, extrinsics, images, metadata = [], [], [], [], []
    for index in range(args.frames):
        camera = np.array([index * 1.8, np.sin(index * 0.7), 2.0 + 0.15 * index])
        point_map = np.stack((camera[0] + x_grid, camera[1] + y_grid, 0.15 * np.sin(x_grid * 3 + index)), axis=-1)
        world_points.append(point_map.astype(np.float32))
        confidences.append(np.clip(0.55 + 0.4 * (1 - x_grid**2 - y_grid**2), 0.05, 1).astype(np.float32))
        extrinsic = np.zeros((3, 4), dtype=np.float32)
        extrinsic[:, :3] = np.eye(3)
        extrinsic[:, 3] = -camera
        extrinsics.append(extrinsic)
        rgb = np.stack(((x_grid + 1) * 120, (y_grid + 1) * 120, np.full_like(x_grid, 50 + index * 20)), axis=-1).clip(0, 255).astype(np.uint8)
        image_name = f"frame_{index:04d}.png"
        image_path = frames_dir / image_name
        Image.fromarray(rgb).save(image_path)
        images.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255)
        # Small, non-collinear route near New Delhi. Stage 4 converts it to UTM.
        metadata.append({
            "frame_file": image_name, "frame_index": index, "video_time_sec": float(index),
            "gps_timestamp": 1_700_000_000.0 + index,
            "latitude": 28.6139 + np.sin(index * 0.7) * 0.00001,
            "longitude": 77.2090 + index * 0.000018,
            "altitude": 120.0 + 0.15 * index,
            "gps_dt_sec": 0.0,
        })
    with (args.output_dir / "frames_meta.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    np.savez_compressed(
        args.output_dir / "vggt_predictions.npz",
        world_points=np.asarray(world_points),
        world_points_conf=np.asarray(confidences),
        extrinsics=np.asarray(extrinsics),
        intrinsics=np.repeat(np.array([[[30, 0, args.width / 2], [0, 30, args.height / 2], [0, 0, 1]]], dtype=np.float32), args.frames, axis=0),
        images=np.asarray(images),
    )
    (args.output_dir / "vggt_image_paths.json").write_text(
        json.dumps([str(path.resolve()) for path in sorted(frames_dir.iterdir())], indent=2), encoding="utf-8"
    )
    (args.output_dir / "SYNTHETIC_ONLY.txt").write_text(
        "This data exists solely for smoke tests. It is not drone imagery and cannot support an accuracy or runtime claim.\n",
        encoding="utf-8",
    )
    print(f"Created synthetic smoke data in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
