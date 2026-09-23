import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
VERIFIER = ROOT / "scripts" / "verify_vggt_schema.py"


def batch_first_raw_schema():
    """A raw VGGT schema using the verified batch-first dense tensor layout."""
    return {
        "pose_enc": {"shape": [1, 18, 9]},
        "world_points": {"shape": [1, 18, 4, 6, 3]},
        "world_points_conf": {"shape": [1, 18, 4, 6]},
        "images": {"shape": [1, 18, 3, 4, 6]},
    }


def decoded_prediction_arrays():
    return {
        "extrinsics": np.zeros((1, 18, 3, 4), dtype=np.float32),
        "intrinsics": np.zeros((1, 18, 3, 3), dtype=np.float32),
        "world_points": np.zeros((1, 18, 4, 6, 3), dtype=np.float32),
        "world_points_conf": np.ones((1, 18, 4, 6), dtype=np.float32),
        # Stage 3 squeezes the original image batch before saving this field.
        "images": np.zeros((18, 3, 4, 6), dtype=np.float32),
    }


class VerifyVggtSchemaTests(unittest.TestCase):
    def run_verifier(self, schema, predictions=None):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            schema_path = run_dir / "vggt_output_schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            if predictions is not None:
                np.savez_compressed(run_dir / "vggt_predictions.npz", **predictions)
            return subprocess.run(
                [sys.executable, str(VERIFIER), str(schema_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

    def test_accepts_real_batch_first_schema_and_archive(self):
        result = self.run_verifier(batch_first_raw_schema(), decoded_prediction_arrays())

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Raw VGGT model schema", result.stdout)
        self.assertIn("normalized singleton batch axis", result.stdout)
        self.assertIn("[PASS] extrinsics", result.stdout)
        self.assertIn("RESULT: PASS", result.stdout)
        self.assertIn("scripts/05_export_formats.py", result.stdout)

    def test_allows_stage_four_confidence_fallback(self):
        schema = batch_first_raw_schema()
        del schema["world_points_conf"]
        predictions = decoded_prediction_arrays()
        del predictions["world_points_conf"]

        result = self.run_verifier(schema, predictions)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[PASS] world_points_conf: missing but optional", result.stdout)

    def test_missing_decoded_key_fails_with_stage_four_location(self):
        predictions = decoded_prediction_arrays()
        del predictions["extrinsics"]

        result = self.run_verifier(batch_first_raw_schema(), predictions)

        self.assertEqual(result.returncode, 1)
        self.assertIn("[FAIL] extrinsics: missing", result.stdout)
        self.assertIn("scripts/04_gps_alignment.py:", result.stdout)
        self.assertIn("RESULT: FAIL", result.stdout)

    def test_invalid_saved_shape_fails(self):
        predictions = decoded_prediction_arrays()
        predictions["extrinsics"] = np.zeros((1, 18, 4), dtype=np.float32)

        result = self.run_verifier(batch_first_raw_schema(), predictions)

        self.assertEqual(result.returncode, 1)
        self.assertIn("[FAIL] extrinsics: saved shape [1, 18, 4]", result.stdout)
        self.assertIn("scripts/04_gps_alignment.py:", result.stdout)

    def test_raw_schema_without_prediction_archive_fails(self):
        result = self.run_verifier(batch_first_raw_schema())

        self.assertEqual(result.returncode, 1)
        self.assertIn("prediction archive not found", result.stdout)
