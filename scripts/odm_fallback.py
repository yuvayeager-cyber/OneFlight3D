#!/usr/bin/env python3
"""Run OpenDroneMap as a separate, reproducible fallback pipeline.

ODM uses GPS in each input image's EXIF metadata.  It intentionally does not
read or combine VGGT output: it is independent insurance for a demo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent OpenDroneMap fallback in Docker.")
    parser.add_argument("--frames_dir", required=True, type=Path, help="GPS-EXIF-tagged UAV image directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to receive ODM project files and report.")
    parser.add_argument("--project_name", default="odm", help="ODM project folder name (default: odm).")
    parser.add_argument("--image", default="opendronemap/odm:latest", help="ODM Docker image.")
    parser.add_argument("--orthophoto_resolution", type=float, default=5.0, help="Orthophoto resolution in cm/pixel.")
    parser.add_argument("--fast_orthophoto", action="store_true", help="Use ODM's faster orthophoto mode.")
    parser.add_argument("--copy_mode", choices=("copy", "hardlink"), default="hardlink", help="How to stage input images (default: hardlink, falling back to copy).")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print the Docker command without staging or running ODM.")
    return parser.parse_args()


def image_files(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def stage_images(images: list[Path], destination: Path, mode: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in images:
        target = destination / source.name
        if target.exists():
            continue
        if mode == "hardlink":
            try:
                target.hardlink_to(source)
                continue
            except OSError:
                pass
        shutil.copy2(source, target)


def write_report(output_dir: Path, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "odm_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    args = parse_args()
    if not args.frames_dir.is_dir():
        print(f"Frame directory does not exist: {args.frames_dir}", file=sys.stderr)
        return 2
    images = image_files(args.frames_dir)
    if len(images) < 3:
        print("ODM needs at least three images; no run was started.", file=sys.stderr)
        return 2

    project_dir = args.output_dir.resolve() / args.project_name
    command = [
        "docker", "run", "--rm",
        "-v", f"{args.output_dir.resolve()}:/datasets",
        args.image,
        "--project-path", "/datasets",
        args.project_name,
        "--orthophoto-resolution", str(args.orthophoto_resolution),
    ]
    if args.fast_orthophoto:
        command.append("--fast-orthophoto")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_frames_dir": str(args.frames_dir.resolve()),
        "input_image_count": len(images),
        "project_dir": str(project_dir),
        "docker_command": command,
        "dry_run": args.dry_run,
        "verified": False,
    }
    if args.dry_run:
        print("Docker command (not run):\n" + " ".join(command))
        return 0
    if not shutil.which("docker"):
        report["failure_reason"] = "Docker was not found on PATH."
        report_path = write_report(args.output_dir, report)
        print(f"Docker was not found on PATH. Install/start Docker Desktop, then rerun this command. Report: {report_path}", file=sys.stderr)
        return 1
    try:
        docker_info = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    except OSError as error:
        print(f"Cannot start Docker: {error}", file=sys.stderr)
        return 1
    if docker_info.returncode:
        report["failure_reason"] = "Docker is installed but its daemon is unavailable."
        report_path = write_report(args.output_dir, report)
        print(f"Docker is installed but its daemon is unavailable. Start Docker Desktop, then retry. Report: {report_path}", file=sys.stderr)
        return 1

    stage_images(images, project_dir / "images", args.copy_mode)
    started = time.monotonic()
    result = subprocess.run(command, text=True)
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["docker_exit_code"] = result.returncode
    expected = {
        "point_cloud": project_dir / "odm_georeferencing" / "odm_georeferenced_model.laz",
        "orthophoto": project_dir / "odm_orthophoto" / "odm_orthophoto.tif",
        "textured_model": project_dir / "odm_texturing" / "odm_textured_model.obj",
    }
    report["outputs"] = {name: {"path": str(path), "exists": path.exists()} for name, path in expected.items()}
    report["verified"] = result.returncode == 0 and any(item["exists"] for item in report["outputs"].values())
    report_path = write_report(args.output_dir, report)
    if report["verified"]:
        print(f"ODM fallback verified in {report['elapsed_seconds']} s. Report: {report_path}")
        return 0
    print(f"ODM did not produce a verified output. Report: {report_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
