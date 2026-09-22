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


EXPECTED_STAGE4_KEYS = (
    KeyExpectation(
        key="extrinsics",
        required=True,
        dimensions=3,
        trailing_dimensions=(3, 4),
        description="camera transforms shaped (frames, 3, 4)",
        references=(
            SourceReference(
                "scripts/04_gps_alignment.py", 'extrinsics = pred["extrinsics"]'
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
        description="camera matrices shaped (frames, 3, 3)",
        references=(
            SourceReference(
                "scripts/04_gps_alignment.py", 'intrinsics = pred["intrinsics"]'
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
                "scripts/04_gps_alignment.py", 'if "world_points" not in pred'
            ),
            SourceReference(
                "scripts/04_gps_alignment.py", 'world_points = pred["world_points"]'
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
                "scripts/04_gps_alignment.py", 'pred["world_points_conf"]'
            ),
        ),
    ),
    KeyExpectation(
        key="images",
        required=False,
        dimensions=4,
        trailing_dimensions=(),
        description="optional image fallback with a three-channel axis",
        references=(
            SourceReference("scripts/04_gps_alignment.py", 'imgs = pred["images"]'),
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

    if expectation.key == "images" and 3 not in (shape[1], shape[-1]):
        errors.append(
            "expected a three-channel axis at dimension 1 or the final dimension"
        )
    return errors


def print_references(references: tuple[SourceReference, ...]) -> None:
    for location in source_locations(references):
        print(f"       consumer: {location}")


def validate_schema(schema: dict[str, Any]) -> bool:
    """Print the compatibility report and return True only for a safe Stage 4 gate."""
    passed = True
    known_shapes: dict[str, list[int]] = {}

    for expectation in EXPECTED_STAGE4_KEYS:
        if expectation.key not in schema:
            if expectation.required:
                print(
                    f"[FAIL] {expectation.key}: missing; required {expectation.description}."
                )
                print_references(expectation.references)
                passed = False
            else:
                fallback = (
                    "Stage 4 supplies all-ones confidence."
                    if expectation.key == "world_points_conf"
                    else "Stage 4 uses image paths or a grey fallback."
                )
                print(f"[PASS] {expectation.key}: missing but optional; {fallback}")
            continue

        shape, shape_error = get_shape(schema[expectation.key])
        if shape_error:
            print(f"[FAIL] {expectation.key}: present but {shape_error}.")
            print_references(expectation.references)
            passed = False
            continue

        errors = validate_shape(expectation, shape)
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
            f"[PASS] {expectation.key}:{optional_label} present with sane shape {shape} "
            f"({expectation.description})."
        )

    points_shape = known_shapes.get("world_points")
    confidence_shape = known_shapes.get("world_points_conf")
    if points_shape and confidence_shape and confidence_shape != points_shape[:3]:
        print(
            "[FAIL] world_points_conf: shape "
            f"{confidence_shape} does not match world_points leading shape "
            f"{points_shape[:3]}."
        )
        print_references(
            next(
                expectation.references
                for expectation in EXPECTED_STAGE4_KEYS
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
            "[FAIL] frame count: extrinsics, intrinsics, and world_points disagree: "
            f"{frame_shapes}."
        )
        print("       consumer: scripts/04_gps_alignment.py:208 compares metadata frames to extrinsics.")
        passed = False

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

    if passed:
        print("\nRESULT: PASS — schema satisfies the current Stage 4 input assumptions.")
    else:
        print("\nRESULT: FAIL — do not attempt Stage 4 until the listed mismatch is resolved.")
    return passed


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
    return 0 if validate_schema(schema) else 1


if __name__ == "__main__":
    raise SystemExit(main())
