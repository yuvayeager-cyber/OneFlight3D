import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


class SmokePipelineTests(unittest.TestCase):
    def run_script(self, *arguments):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_synthetic_alignment_and_portable_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_script(SCRIPTS / "generate_synthetic_smoke_data.py", "--output_dir", run_dir)
            self.run_script(
                SCRIPTS / "04_gps_alignment.py",
                "--predictions", run_dir / "vggt_predictions.npz",
                "--frames_meta", run_dir / "frames_meta.csv",
                "--output_file", run_dir / "aligned_points.npz",
            )
            self.run_script(
                SCRIPTS / "05_export_formats.py",
                "--aligned_points", run_dir / "aligned_points.npz",
                "--frames_meta", run_dir / "frames_meta.csv",
                "--output_dir", run_dir / "exports",
                "--formats", "ply,colmap",
            )
            self.run_script(
                SCRIPTS / "06_confidence_export.py",
                "--aligned_points", run_dir / "aligned_points.npz",
                "--output_dir", run_dir / "exports",
            )
            with np.load(run_dir / "aligned_points.npz") as data:
                self.assertGreater(len(data["points"]), 100)
                self.assertEqual(str(data["crs_definition"][0]), "EPSG:32643")
            self.assertTrue((run_dir / "exports" / "cloud.ply").is_file())
            self.assertTrue((run_dir / "exports" / "confidence_cloud.ply").is_file())
            self.assertTrue((run_dir / "exports" / "colmap" / "cameras.txt").is_file())
