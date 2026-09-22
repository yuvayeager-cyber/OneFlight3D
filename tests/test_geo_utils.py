import unittest
import importlib.util
from pathlib import Path

import pandas as pd

import numpy as np

from utils.geo_utils import umeyama_similarity, wgs84_to_utm


def load_stage_one_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "01_extract_frames.py"
    spec = importlib.util.spec_from_file_location("extract_frames", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeoUtilsTests(unittest.TestCase):
    def test_umeyama_recovers_known_similarity(self):
        source = np.array([[0, 0, 0], [2, 0, 1], [0, 3, 1], [1, 2, 4]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        scale = 2.5
        translation = np.array([100, -20, 7], dtype=float)
        destination = scale * (source @ rotation.T) + translation

        found_scale, found_rotation, found_translation, rmse = umeyama_similarity(source, destination)

        np.testing.assert_allclose(found_scale, scale, atol=1e-10)
        np.testing.assert_allclose(found_rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(found_translation, translation, atol=1e-10)
        self.assertLess(rmse, 1e-10)

    def test_utm_fallback_has_southern_epsg_and_false_northing(self):
        coordinates, zone, crs = wgs84_to_utm(
            np.array([-33.8688]), np.array([151.2093]), np.array([30.0])
        )
        self.assertEqual(zone, 56)
        self.assertIn("32756", crs)
        self.assertGreater(coordinates[0, 1], 1_000_000)

    def test_gps_match_uses_video_time_from_zero(self):
        stage_one = load_stage_one_module()
        telemetry = pd.DataFrame({
            "timestamp": [1000.0, 1060.0, 1120.0],
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [4.0, 5.0, 6.0],
            "altitude": [7.0, 8.0, 9.0],
        })
        matched = stage_one.find_nearest_gps(60.0, telemetry, video_start_timestamp=1000.0)
        self.assertEqual(matched["gps_timestamp"], 1060.0)
