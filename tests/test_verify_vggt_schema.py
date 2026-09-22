import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VERIFIER = ROOT / "scripts" / "verify_vggt_schema.py"


class VerifyVggtSchemaTests(unittest.TestCase):
    def run_verifier(self, schema):
        with tempfile.TemporaryDirectory() as temporary:
            schema_path = Path(temporary) / "vggt_output_schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(VERIFIER), str(schema_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

    def test_accepts_canonical_stage_four_shapes(self):
        schema = {
            "extrinsics": {"shape": [18, 3, 4]},
            "intrinsics": {"shape": [18, 3, 3]},
            "world_points": {"shape": [18, 480, 640, 3]},
            "world_points_conf": {"shape": [18, 480, 640]},
            "images": {"shape": [18, 3, 480, 640]},
        }

        result = self.run_verifier(schema)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[PASS] world_points", result.stdout)
        self.assertIn("RESULT: PASS", result.stdout)
        self.assertIn("scripts/05_export_formats.py", result.stdout)

    def test_allows_stage_four_confidence_fallback(self):
        schema = {
            "extrinsics": {"shape": [18, 3, 4]},
            "intrinsics": {"shape": [18, 3, 3]},
            "world_points": {"shape": [18, 480, 640, 3]},
        }

        result = self.run_verifier(schema)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[PASS] world_points_conf: missing but optional", result.stdout)
        self.assertIn("Recorded keys: extrinsics, intrinsics, world_points", result.stdout)

    def test_missing_required_key_fails_with_consumer_location(self):
        schema = {
            "extrinsics": {"shape": [18, 3, 4]},
            "intrinsics": {"shape": [18, 3, 3]},
        }

        result = self.run_verifier(schema)

        self.assertEqual(result.returncode, 1)
        self.assertIn("[FAIL] world_points: missing", result.stdout)
        self.assertIn("scripts/04_gps_alignment.py:", result.stdout)
        self.assertIn("RESULT: FAIL", result.stdout)

    def test_invalid_shape_fails(self):
        schema = {
            "extrinsics": {"shape": [18, 4]},
            "intrinsics": {"shape": [18, 3, 3]},
            "world_points": {"shape": [18, 480, 640, 3]},
        }

        result = self.run_verifier(schema)

        self.assertEqual(result.returncode, 1)
        self.assertIn("[FAIL] extrinsics: present with shape [18, 4]", result.stdout)
        self.assertIn("scripts/04_gps_alignment.py:", result.stdout)
