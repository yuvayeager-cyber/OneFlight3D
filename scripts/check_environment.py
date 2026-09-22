#!/usr/bin/env python3
"""Report whether this host can run each OneFlight3D pipeline path.

This script never installs packages or downloads model weights.  It is safe to
run before a demo and emits a JSON report suitable for attaching to a run log.
"""

from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CORE_PACKAGES = ("numpy", "pandas", "cv2", "scipy", "pyproj", "PIL")
EXPORT_PACKAGES = ("laspy", "rasterio", "open3d", "trimesh")
VGGT_PACKAGES = ("torch", "vggt")
YOLO_PACKAGES = ("ultralytics",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check OneFlight3D runtime prerequisites without changing the host."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Directory for environment_report.json (default: print only).",
    )
    parser.add_argument(
        "--min_disk_gb", type=float, default=10.0,
        help="Minimum free disk space to recommend for a run (default: 10).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero unless core, VGGT, YOLO, exporters, and ODM are ready.",
    )
    return parser.parse_args()


def package_status(name: str) -> dict[str, object]:
    available = importlib.util.find_spec(name) is not None
    result: dict[str, object] = {"available": available}
    if available:
        try:
            # Importing OpenCV or a CUDA package merely to get a version can
            # initialize native runtimes, so use package metadata instead.
            distribution = {"cv2": "opencv-python", "PIL": "Pillow"}.get(name, name)
            result["version"] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result["version"] = "unknown"
        # A package can have metadata while a required native DLL is blocked.
        # Probe in a child so a bad native import cannot hang this preflight.
        try:
            probe = subprocess.run(
                [sys.executable, "-c", f"import {name}"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if probe.returncode:
                result.update(
                    available=False,
                    error=(probe.stderr or probe.stdout).strip().splitlines()[-1],
                )
        except subprocess.TimeoutExpired:
            result.update(available=False, error="import probe timed out after 15 seconds")
    return result


def group_status(names: tuple[str, ...]) -> dict[str, object]:
    packages = {name: package_status(name) for name in names}
    return {
        "ready": all(bool(entry["available"]) for entry in packages.values()),
        "packages": packages,
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    disk = shutil.disk_usage(root)
    report: dict[str, object] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "disk": {
            "free_gb": round(disk.free / 1024**3, 2),
            "required_gb": args.min_disk_gb,
            "ready": disk.free >= args.min_disk_gb * 1024**3,
        },
        "core": group_status(CORE_PACKAGES),
        "exporters": group_status(EXPORT_PACKAGES),
        "vggt": group_status(VGGT_PACKAGES),
        "yolo": group_status(YOLO_PACKAGES),
        "tools": {
            name: {"available": shutil.which(name) is not None}
            for name in ("docker", "blender", "nvidia-smi")
        },
    }

    torch_status = report["vggt"]["packages"].get("torch")  # type: ignore[index]
    if torch_status["available"]:  # type: ignore[index]
        try:
            import torch

            report["gpu"] = {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            }
        except Exception as error:
            report["gpu"] = {"cuda_available": False, "error": str(error)}
    else:
        report["gpu"] = {"cuda_available": False, "reason": "PyTorch is unavailable"}

    report["odm"] = {
        "ready": bool(report["tools"]["docker"]["available"]),  # type: ignore[index]
        "reason": "Docker is required to run the independent ODM fallback.",
    }
    report["ready_for_full_pipeline"] = bool(
        report["disk"]["ready"]  # type: ignore[index]
        and report["core"]["ready"]  # type: ignore[index]
        and report["exporters"]["ready"]  # type: ignore[index]
        and report["vggt"]["ready"]  # type: ignore[index]
        and report["yolo"]["ready"]  # type: ignore[index]
        and report["gpu"]["cuda_available"]  # type: ignore[index]
    )

    print(json.dumps(report, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / "environment_report.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Environment report written to {destination}")

    return 0 if (not args.strict or report["ready_for_full_pipeline"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
