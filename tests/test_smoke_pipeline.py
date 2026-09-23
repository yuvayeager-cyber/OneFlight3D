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
                # The no-pyproj fallback stores an EPSG token while pyproj
                # serializes the same CRS as WKT.  Validate the authoritative
                # numeric EPSG value and accept either interoperable encoding.
                self.assertEqual(int(data["utm_epsg"][0]), 32643)
                self.assertIn("32643", str(data["crs_definition"][0]))
            self.assertTrue((run_dir / "exports" / "cloud.ply").is_file())
            self.assertTrue((run_dir / "exports" / "confidence_cloud.ply").is_file())
            self.assertTrue((run_dir / "exports" / "colmap" / "cameras.txt").is_file())

    def test_alignment_normalizes_singleton_vggt_batch_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_script(SCRIPTS / "generate_synthetic_smoke_data.py", "--output_dir", run_dir)
            with np.load(run_dir / "vggt_predictions.npz") as unbatched:
                batched = {name: unbatched[name][None, ...] for name in unbatched.files}
            batched_path = run_dir / "vggt_predictions_batched.npz"
            np.savez_compressed(batched_path, **batched)
            # Force Stage 4 to use the batch-normalized image tensor fallback.
            (run_dir / "vggt_image_paths.json").unlink()

            self.run_script(
                SCRIPTS / "04_gps_alignment.py",
                "--predictions", batched_path,
                "--frames_meta", run_dir / "frames_meta.csv",
                "--output_file", run_dir / "aligned_points_batched.npz",
            )
            self.run_script(
                SCRIPTS / "05_export_formats.py",
                "--aligned_points", run_dir / "aligned_points_batched.npz",
                "--frames_meta", run_dir / "frames_meta.csv",
                "--output_dir", run_dir / "exports_batched",
                "--formats", "ply,colmap",
            )
            with np.load(run_dir / "aligned_points_batched.npz") as data:
                self.assertEqual(data["extrinsics"].shape, (6, 3, 4))
                self.assertEqual(data["intrinsics"].shape, (6, 3, 3))
            self.assertTrue((run_dir / "exports_batched" / "cloud.ply").is_file())
            self.assertTrue((run_dir / "exports_batched" / "colmap" / "cameras.txt").is_file())

    def test_alignment_rejects_multiple_vggt_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_script(SCRIPTS / "generate_synthetic_smoke_data.py", "--output_dir", run_dir)
            with np.load(run_dir / "vggt_predictions.npz") as unbatched:
                multi_batch = {
                    name: np.repeat(unbatched[name][None, ...], 2, axis=0)
                    for name in unbatched.files
                }
            predictions_path = run_dir / "vggt_predictions_multi_batch.npz"
            np.savez_compressed(predictions_path, **multi_batch)
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPTS / "04_gps_alignment.py"),
                    "--predictions", str(predictions_path),
                    "--frames_meta", str(run_dir / "frames_meta.csv"),
                    "--output_file", str(run_dir / "should_not_exist.npz"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("accepts only a single image sequence", result.stdout + result.stderr)
