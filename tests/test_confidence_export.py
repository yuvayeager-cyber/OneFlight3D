import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "06_confidence_export.py"
spec = importlib.util.spec_from_file_location("confidence_export", SCRIPT)
confidence_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confidence_export)


class ConfidenceDisplayTests(unittest.TestCase):
    def test_unbounded_vggt_confidence_is_not_clamped_to_green(self):
        raw = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=np.float32)

        normalized, metadata = confidence_export.normalize_confidence_for_display(raw)
        colors = confidence_export.confidence_to_rgb_rg(normalized)

        self.assertGreater(normalized.max(), normalized.min())
        self.assertTrue((colors[:, 0] > 0).any())
        self.assertTrue((colors[:, 1] > 0).any())
        self.assertEqual(metadata["method"], "linear_percentile_clip")

    def test_constant_confidence_has_neutral_display_value(self):
        normalized, _ = confidence_export.normalize_confidence_for_display(
            np.full(4, 7.0, dtype=np.float32)
        )

        np.testing.assert_array_equal(normalized, np.full(4, 0.5, dtype=np.float32))
