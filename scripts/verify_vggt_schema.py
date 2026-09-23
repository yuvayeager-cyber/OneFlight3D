#!/usr/bin/env python3
"""Gate a recorded VGGT output schema before attempting Stage 4.

``03_run_vggt.py`` writes ``vggt_output_schema.json`` from the model's actual
prediction dictionary.  This script compares that record with the key names
and array ranks currently consumed by Stages 4 and 5.  It intentionally does
not modify, infer, or rename any prediction fields.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SourceReference:
    """A source snippet whose current line number is reported to the operator."""

    path: str
    snippet: str


@dataclass(frozen=True)
class KeyExpectation:
    key: str
    required: bool
    dimensions: int
    trailing_dimensions: tuple[int, ...]
    description: str
    references: tuple[SourceReference, ...]


EXPECTED_RAW_VGGT_KEYS = (
    KeyExpectation(
        key="pose_enc",
        required=True,
        dimensions=3,
        trailing_dimensions=(),
        description="raw pose encoding shaped (batch, frames, pose values)",
        references=(
            SourceReference(
                "scripts/03_run_vggt.py", 'predictions["pose_enc"]'
            ),
        ),
    ),
    KeyExpectation(
        key="world_points",
        required=True,
        dimensions=5,
        trailing_dimensions=(3,),
        description="raw dense points shaped (batch, frames, height, width, 3)",
        references=(
            SourceReference(
                "scripts/03_run_vggt.py", 'save_dict["world_points"]'
            ),
        ),
    ),
    KeyExpectation(
        key="world_points_conf",
        required=False,
        dimensions=4,
        trailing_dimensions=(),
        description="raw confidence shaped (batch, frames, height, width)",
        references=(
            SourceReference(
                "scripts/03_run_vggt.py", 'save_dict["world_points_conf"]'
            ),
        ),
    ),
    KeyExpectation(
        key="images",
        required=False,
        dimensions=5,
        trailing_dimensions=(),
        description="raw images with batch, frame, and three-channel axes",
        references=(
            SourceReference("scripts/03_run_vggt.py", 'save_dict["images"]'),
        ),
    ),
)

EXPECTED_STAGE4_ARCHIVE_KEYS = (
    KeyExpectation(
        key="extrinsics",
        required=True,
        dimensions=3,
        trailing_dimensions=(3, 4),
        description="decoded camera transforms shaped (frames, 3, 4)",
        references=(
            SourceReference(
                "scripts/04_gps_alignment.py", '"extrinsics": (3, (3, 4))'
            ),
            SourceReference(
                "scripts/05_export_formats.py", 'extrinsics = data["extrinsics"]'
            ),
        ),
    ),
    KeyExpectation(
        key="intrinsics",
        required=True,
        dimensions=3,
        trailing_dimensions=(3, 3),
        description="decoded camera matrices shaped (frames, 3, 3)",
        references=(
            SourceReference(
                "scripts/04_gps_alignment.py", '"intrinsics": (3, (3, 3))'
            ),
            SourceReference(
                "scripts/05_export_formats.py", 'intrinsics = data["intrinsics"]'
            ),
        ),
    ),
    KeyExpectation(
        key="world_points",
        required=True,
        dimensions=4,
        trailing_dimensions=(3,),
        description="dense points shaped (frames, height, width, 3)",
        references=(
            SourceReference(
                "scripts/04_gps_alignment.py", '"world_points": (4, (3,))'
            ),
        ),
    ),
    KeyExpectation(
        key="world_points_conf",
        required=False,
        dimensions=3,
        trailing_dimensions=(),
        description="per-point confidence shaped (frames, height, width)",
        references=(
            SourceReference(
                "scripts/04_gps_alignment.py", '"world_points_conf", canonical['
            ),
        ),
    ),
    KeyExpectation(
        key="images",
        required=False,
        dimensions=4,
        trailing_dimensions=(),
        description="optional images with a three-channel axis",
        references=(
            SourceReference(
                "scripts/04_gps_alignment.py", 'canonicalize_vggt_array("images"'
            ),
        ),
    ),
)

STAGE5_ALIGNED_KEYS = (
    "points",
    "colors",
    "confidence",
    "extrinsics",
    "intrinsics",
    "crs_definition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a real VGGT schema before attempting Stage 4."
    )
    parser.add_argument(
        "schema_file",
        type=Path,
        help="Path to vggt_output_schema.json emitted by 03_run_vggt.py",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Path to vggt_predictions.npz written by Stage 3. Defaults to a "
            "same-directory file with that name."
        ),
    )
    return parser.parse_args()


def source_locations(references: tuple[SourceReference, ...]) -> list[str]:
    """Find current source lines instead of hard-coding line numbers."""
    locations: list[str] = []
    for reference in references:
        source_path = ROOT / reference.path
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            locations.append(f"{reference.path}: unavailable ({error})")
            continue

        matched = False
        for line_number, line in enumerate(lines, start=1):
            if reference.snippet in line:
                locations.append(f"{reference.path}:{line_number}: {line.strip()}")
                matched = True
        if not matched:
            locations.append(
                f"{reference.path}: snippet not found: {reference.snippet}"
            )
    return locations


def get_shape(value: Any) -> tuple[list[int] | None, str | None]:
    """Validate the shape representation written by Stage 3's schema logger."""
    if not isinstance(value, dict):
        return None, "schema entry is not an object"

    shape = value.get("shape")
    if not isinstance(shape, list):
        return None, "schema entry has no list-valued 'shape'"
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in shape):
        return None, "shape contains a non-integer dimension"
    if any(dimension <= 0 for dimension in shape):
        return None, "shape contains a non-positive dimension"
    return shape, None


def validate_shape(expectation: KeyExpectation, shape: list[int]) -> list[str]:
    """Return human-readable incompatibilities for a present schema key."""
    errors: list[str] = []
    if len(shape) != expectation.dimensions:
        errors.append(
            f"expected {expectation.dimensions} dimensions, found {len(shape)}"
        )
        return errors

    if expectation.trailing_dimensions:
        expected_suffix = list(expectation.trailing_dimensions)
        actual_suffix = shape[-len(expected_suffix):]
        if actual_suffix != expected_suffix:
            errors.append(
                f"expected trailing dimensions {expected_suffix}, found {actual_suffix}"
            )

    channel_index = 2 if len(shape) == 5 else 1
    if expectation.key == "images" and 3 not in (shape[channel_index], shape[-1]):
        errors.append(
            "expected a three-channel axis after the frame axis or at the final dimension"
        )
    return errors


def print_references(references: tuple[SourceReference, ...]) -> None:
    for location in source_locations(references):
        print(f"       consumer: {location}")


def validate_raw_schema(schema: dict[str, Any]) -> bool:
    """Validate the model dictionary recorded before Stage 3 adapts its tensors."""
    passed = True
    known_shapes: dict[str, list[int]] = {}

    print("\nRaw VGGT model schema:")
    for expectation in EXPECTED_RAW_VGGT_KEYS:
        if expectation.key not in schema:
            if expectation.required:
                print(
                    f"[FAIL] {expectation.key}: missing; required {expectation.description}."
                )
                print_references(expectation.references)
                passed = False
            else:
                fallback = "Stage 4 supplies all-ones confidence." if expectation.key == "world_points_conf" else "Stage 4 can use file paths or a grey fallback."
                print(f"[PASS] {expectation.key}: missing but optional; {fallback}")
            continue

        shape, shape_error = get_shape(schema[expectation.key])
        if shape_error:
            print(f"[FAIL] {expectation.key}: present but {shape_error}.")
            print_references(expectation.references)
            passed = False
            continue

        errors = validate_shape(expectation, shape)
        if shape and shape[0] != 1:
            errors.append(
                f"expected singleton batch dimension [0] == 1, found {shape[0]}"
            )
        if errors:
            print(
                f"[FAIL] {expectation.key}: present with shape {shape}; "
                + "; ".join(errors)
                + "."
            )
            print_references(expectation.references)
            passed = False
            continue

        known_shapes[expectation.key] = shape
        optional_label = " optional" if not expectation.required else ""
        print(
            f"[PASS] {expectation.key}:{optional_label} present with verified raw shape {shape} "
            f"({expectation.description})."
        )

    points_shape = known_shapes.get("world_points")
    confidence_shape = known_shapes.get("world_points_conf")
    if points_shape and confidence_shape and confidence_shape != points_shape[:4]:
        print(
            "[FAIL] world_points_conf: shape "
            f"{confidence_shape} does not match world_points leading shape "
            f"{points_shape[:4]}."
        )
        print_references(
            next(
                expectation.references
                for expectation in EXPECTED_RAW_VGGT_KEYS
                if expectation.key == "world_points_conf"
            )
        )
        passed = False

    images_shape = known_shapes.get("images")
    if points_shape and images_shape and images_shape[:2] != points_shape[:2]:
        print(
            "[FAIL] images: batch/frame dimensions "
            f"{images_shape[:2]} do not match world_points {points_shape[:2]}."
        )
        print_references(
            next(
                expectation.references
                for expectation in EXPECTED_RAW_VGGT_KEYS
                if expectation.key == "images"
            )
        )
        passed = False
    return passed


def canonical_archive_shape(expectation: KeyExpectation, shape: list[int]) -> tuple[list[int] | None, list[str]]:
    """Apply the exact singleton-batch rule Stage 4 uses to an NPZ field."""
    canonical_shape = shape
    errors: list[str] = []
    if len(canonical_shape) == expectation.dimensions + 1:
        if canonical_shape[0] != 1:
            return None, [
                f"has batch shape {shape}; Stage 4 accepts only batch size 1"
            ]
        canonical_shape = canonical_shape[1:]
        print(
            f"       normalized singleton batch axis: {shape} -> {canonical_shape}"
        )
    errors.extend(validate_shape(expectation, canonical_shape))
    return canonical_shape if not errors else None, errors


def validate_prediction_archive(predictions_path: Path) -> bool:
    """Validate the concrete Stage 3 archive consumed by Stage 4."""
    try:
        with np.load(predictions_path, allow_pickle=False) as archive:
            archive_shapes = {name: list(archive[name].shape) for name in archive.files}
    except (OSError, ValueError) as error:
        print(f"[FAIL] Cannot read Stage 3 prediction archive {predictions_path}: {error}")
        return False

    print(f"\nStage 3 prediction archive: {predictions_path}")
    print("Recorded keys: " + (", ".join(sorted(archive_shapes)) or "<none>"))
    passed = True
    known_shapes: dict[str, list[int]] = {}
    for expectation in EXPECTED_STAGE4_ARCHIVE_KEYS:
        shape = archive_shapes.get(expectation.key)
        if shape is None:
            if expectation.required:
                print(
                    f"[FAIL] {expectation.key}: missing; required {expectation.description}."
                )
                print_references(expectation.references)
                passed = False
            else:
                fallback = "Stage 4 supplies all-ones confidence." if expectation.key == "world_points_conf" else "Stage 4 uses image paths or a grey fallback."
                print(f"[PASS] {expectation.key}: missing but optional; {fallback}")
            continue

        canonical_shape, errors = canonical_archive_shape(expectation, shape)
        if errors:
            print(
                f"[FAIL] {expectation.key}: saved shape {shape}; "
                + "; ".join(errors)
                + "."
            )
            print_references(expectation.references)
            passed = False
            continue
        known_shapes[expectation.key] = canonical_shape
        optional_label = " optional" if not expectation.required else ""
        print(
            f"[PASS] {expectation.key}:{optional_label} saved shape {shape} is compatible "
            f"as {canonical_shape} ({expectation.description})."
        )

    points_shape = known_shapes.get("world_points")
    confidence_shape = known_shapes.get("world_points_conf")
    if points_shape and confidence_shape and confidence_shape != points_shape[:3]:
        print(
            "[FAIL] world_points_conf: canonical shape "
            f"{confidence_shape} does not match world_points leading shape "
            f"{points_shape[:3]}."
        )
        print_references(
            next(
                expectation.references
                for expectation in EXPECTED_STAGE4_ARCHIVE_KEYS
                if expectation.key == "world_points_conf"
            )
        )
        passed = False

    frame_shapes = [
        known_shapes[key][0]
        for key in ("extrinsics", "intrinsics", "world_points")
        if key in known_shapes
    ]
    if len(frame_shapes) > 1 and len(set(frame_shapes)) != 1:
        print(
            "[FAIL] frame count: decoded extrinsics, intrinsics, and world_points "
            f"disagree: {frame_shapes}."
        )
        print_references((
            SourceReference("scripts/04_gps_alignment.py", "len(meta_df) != S"),
        ))
        passed = False
    return passed


def print_stage5_boundary() -> None:
    """Explain why Stage 5 itself is not changed by this raw-VGGT mismatch."""

    print("\nStage 5 boundary (informational):")
    print(
        "  Stage 5 does not consume vggt_output_schema.json or vggt_predictions.npz "
        "directly; it reads the aligned_points.npz created by Stage 4."
    )
    for key in STAGE5_ALIGNED_KEYS:
        locations = source_locations((
            SourceReference("scripts/05_export_formats.py", f'data["{key}"]'),
        ))
        if locations and "snippet not found" not in locations[0]:
            print(f"  - {key}: {locations[0]}")
        else:
            print(f"  - {key}: optional or conditional Stage 5 field")

def main() -> int:
    args = parse_args()
    try:
        with args.schema_file.open(encoding="utf-8") as schema_file:
            schema = json.load(schema_file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[FAIL] Cannot read schema {args.schema_file}: {error}")
        return 2

    if not isinstance(schema, dict):
        print("[FAIL] Schema root must be a JSON object keyed by prediction name.")
        return 2

    print(f"VGGT schema compatibility gate: {args.schema_file}")
    print("Recorded keys: " + (", ".join(sorted(schema)) or "<none>"))
    raw_schema_passed = validate_raw_schema(schema)
    predictions_path = args.predictions or args.schema_file.parent / "vggt_predictions.npz"
    if not predictions_path.exists():
        print(
            "\n[FAIL] Stage 3 prediction archive not found: "
            f"{predictions_path}. The raw schema alone cannot verify the decoded "
            "extrinsics/intrinsics that Stage 4 reads."
        )
        print_stage5_boundary()
        print("\nRESULT: FAIL — do not attempt Stage 4 until the archive is available and passes.")
        return 1

    archive_passed = validate_prediction_archive(predictions_path)
    print_stage5_boundary()
    if raw_schema_passed and archive_passed:
        print("\nRESULT: PASS — raw VGGT output and the saved Stage 3 archive satisfy Stage 4.")
        return 0
    print("\nRESULT: FAIL — do not attempt Stage 4 until the listed mismatch is resolved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
