import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent


def load_preflight_module():
    path = ROOT / "scripts" / "check_environment.py"
    spec = importlib.util.spec_from_file_location("check_environment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckEnvironmentTests(unittest.TestCase):
    def test_import_timeout_retries_and_recovers(self):
        preflight = load_preflight_module()
        first_timeout = subprocess.TimeoutExpired("python", 15)
        success = subprocess.CompletedProcess(["python"], 0, "", "")

        with (
            mock.patch.object(preflight.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(preflight.subprocess, "run", side_effect=[first_timeout, success]) as run,
        ):
            status = preflight.package_status("example_package")

        self.assertTrue(status["available"])
        self.assertEqual(status["import_probe_recovered_on_attempt"], 2)
        self.assertEqual(run.call_count, 2)

    def test_import_timeout_fails_only_after_retry(self):
        preflight = load_preflight_module()
        timeout = subprocess.TimeoutExpired("python", 15)

        with (
            mock.patch.object(preflight.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(preflight.subprocess, "run", side_effect=[timeout, timeout]) as run,
        ):
            status = preflight.package_status("example_package")

        self.assertFalse(status["available"])
        self.assertIn("on 2 attempts", status["error"])
        self.assertEqual(run.call_count, 2)
