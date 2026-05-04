#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import hashlib
from datetime import datetime
from types import SimpleNamespace
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("eevee_hwrt_production_scene_probe.py")
EXPECTATIONS_PATH = Path(__file__).with_name("eevee_hwrt_production_scene_expectations.json")
SPEC = importlib.util.spec_from_file_location("eevee_hwrt_production_scene_probe", MODULE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


class ProductionSceneProbeExpectationTest(unittest.TestCase):
    def test_load_expectations_defaults_and_scenes(self):
        payload = """
{
  "defaults": {"require_exit_code_zero": true},
  "scenes": {"BARBERSHOP_INTERIOR": {"min_global_luma": 0.2}}
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "expectations.json"
            path.write_text(payload, encoding="utf-8")
            defaults, scenes = PROBE.load_expectations(path)

        self.assertEqual(defaults["require_exit_code_zero"], True)
        self.assertEqual(scenes["BARBERSHOP_INTERIOR"]["min_global_luma"], 0.2)

    def test_checked_in_expectations_file_loads(self):
        defaults, scenes = PROBE.load_expectations(EXPECTATIONS_PATH)

        self.assertTrue(defaults["require_exit_code_zero"])
        self.assertTrue(defaults["require_metrics"])
        self.assertTrue(defaults["require_stats"])
        self.assertIn("BARBERSHOP_INTERIOR", scenes)
        self.assertIn("POOL", scenes)
        self.assertEqual(scenes["BARBERSHOP_INTERIOR"]["expected_fast_gi_scene"], "mixed")
        self.assertEqual(scenes["BARBERSHOP_INTERIOR"]["expected_fast_gi_budget"], "balanced")
        self.assertEqual(scenes["POOL"]["expected_fast_gi_scene"], "interior")
        self.assertEqual(scenes["POOL"]["expected_fast_gi_budget"], "indirect")
        self.assertFalse(scenes["POOL"]["fail_if_effectively_black"])

    def test_compute_file_sha256_matches_known_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.txt"
            path.write_text("nuru", encoding="utf-8")
            expected = hashlib.sha256(b"nuru").hexdigest()

            self.assertEqual(PROBE.compute_file_sha256(path), expected)
            self.assertIsNone(PROBE.compute_file_sha256(path.with_name("missing.txt")))

    def test_load_expectations_rejects_unknown_key(self):
        payload = """
{
  "defaults": {"unknown_setting": true}
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "expectations.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown expectation keys"):
                PROBE.load_expectations(path)

    def test_load_expectations_rejects_unknown_top_level_key(self):
        payload = """
{
  "defaults": {"require_exit_code_zero": true},
  "extra": {}
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "expectations.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown top-level keys"):
                PROBE.load_expectations(path)

    def test_load_expectations_rejects_non_object_scene_entry(self):
        payload = """
{
  "scenes": {"BARBERSHOP_INTERIOR": 3}
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "expectations.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                PROBE.load_expectations(path)

    def test_load_expectations_rejects_non_object_scenes_map(self):
        payload = """
{
  "scenes": []
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "expectations.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scenes must be a JSON object"):
                PROBE.load_expectations(path)

    def test_load_expectations_rejects_wrong_boolean_type(self):
        payload = """
{
  "defaults": {"require_exit_code_zero": "yes"}
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "expectations.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                PROBE.load_expectations(path)

    def test_load_expectations_rejects_wrong_numeric_type(self):
        payload = """
{
  "scenes": {"BARBERSHOP_INTERIOR": {"min_global_luma": "bright"}}
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "expectations.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a number"):
                PROBE.load_expectations(path)

    def test_scene_label_from_path_normalizes_name(self):
        label = PROBE.scene_label_from_path(Path("/tmp/My Scene-name.blend"))
        self.assertEqual(label, "MY_SCENE_NAME")

    def test_parse_stats_line_parses_numbers_and_strings(self):
        parsed = PROBE.parse_stats_line(
            "scene=mixed budget=balanced smoothed_traced_ms=123.45 memory_limited=0 "
            "active_cascades=3/3 density=1.23e-2 offset=+7"
        )

        self.assertEqual(parsed["scene"], "mixed")
        self.assertEqual(parsed["budget"], "balanced")
        self.assertEqual(parsed["smoothed_traced_ms"], 123.45)
        self.assertEqual(parsed["memory_limited"], 0)
        self.assertEqual(parsed["active_cascades"], "3/3")
        self.assertEqual(parsed["density"], 1.23e-2)
        self.assertEqual(parsed["offset"], 7)

    def test_extract_last_stats_uses_last_matching_line(self):
        text = "\n".join(
            [
                "noise line",
                "EEVEE HWRT FastGI stats scene=interior smoothed_traced_ms=80.0",
                "other line",
                "EEVEE HWRT FastGI stats scene=mixed smoothed_traced_ms=42.5",
            ]
        )

        parsed = PROBE.extract_last_stats(text)

        self.assertEqual(parsed["scene"], "mixed")
        self.assertEqual(parsed["smoothed_traced_ms"], 42.5)

    def test_trim_output_tail_keeps_only_trailing_lines(self):
        text = "\n".join(f"line-{index}" for index in range(6))
        trimmed = PROBE.trim_output_tail(text, max_lines=3)
        self.assertEqual(trimmed, "line-3\nline-4\nline-5")

    def test_persist_scene_artifacts_copies_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            artifacts_dir = tmpdir_path / "artifacts"
            output_image = tmpdir_path / "render.png"
            metrics_json = tmpdir_path / "metrics.json"
            output_image.write_bytes(b"png")
            metrics_json.write_text('{"global_luma": 0.25}', encoding="utf-8")

            artifact_paths = PROBE.persist_scene_artifacts(
                artifacts_dir, "barbershop_interior", output_image, metrics_json, "stdout", "stderr"
            )

            self.assertEqual((artifacts_dir / "barbershop_interior_render.png").read_bytes(), b"png")
            self.assertEqual(
                (artifacts_dir / "barbershop_interior_metrics.json").read_text(encoding="utf-8"),
                '{"global_luma": 0.25}',
            )
            self.assertEqual((artifacts_dir / "barbershop_interior_stdout.log").read_text(encoding="utf-8"), "stdout")
            self.assertEqual((artifacts_dir / "barbershop_interior_stderr.log").read_text(encoding="utf-8"), "stderr")
            self.assertEqual(
                artifact_paths["render_image"], str(artifacts_dir / "barbershop_interior_render.png")
            )

    def test_build_artifact_stem_is_stable_for_unique_labels(self):
        stem = PROBE.build_artifact_stem("BARBERSHOP_INTERIOR", 0, set())
        self.assertEqual(stem, "barbershop_interior")

    def test_build_artifact_stem_disambiguates_duplicate_labels(self):
        stem = PROBE.build_artifact_stem("BARBERSHOP_INTERIOR", 3, {"BARBERSHOP_INTERIOR"})
        self.assertEqual(stem, "barbershop_interior_03")

    def test_build_request_id_is_stable_for_unique_labels(self):
        request_id = PROBE.build_request_id("BARBERSHOP_INTERIOR", 0, set())
        self.assertEqual(request_id, "BARBERSHOP_INTERIOR")

    def test_build_request_id_disambiguates_duplicate_labels(self):
        request_id = PROBE.build_request_id("BARBERSHOP_INTERIOR", 3, {"BARBERSHOP_INTERIOR"})
        self.assertEqual(request_id, "BARBERSHOP_INTERIOR[03]")

    def test_build_summary_reports_pass_and_fail_counts(self):
        results = [
            {"label": "BARBERSHOP_INTERIOR", "request_id": "BARBERSHOP_INTERIOR", "failures": []},
            {"label": "POOL", "request_id": "POOL", "failures": ["POOL: expected EXIT_CODE=0, got 1"]},
        ]
        failures = ["POOL: expected EXIT_CODE=0, got 1"]
        metadata = {
            "expectations_json": "/tmp/expectations.json",
            "scene_labels_using_defaults_only": [],
            "unused_scene_expectation_labels": [],
        }

        summary = PROBE.build_summary(results, failures, metadata)

        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["scene_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["failed_labels"], ["POOL"])
        self.assertEqual(summary["failed_request_ids"], ["POOL"])
        self.assertEqual(summary["warning_count"], 0)
        self.assertEqual(summary["warnings"], [])

    def test_build_summary_reports_non_fatal_configuration_warnings(self):
        results = [{"label": "BARBERSHOP_INTERIOR", "request_id": "BARBERSHOP_INTERIOR", "failures": []}]
        failures = []
        metadata = {
            "expectations_json": None,
            "scene_labels_using_defaults_only": ["BARBERSHOP_INTERIOR"],
            "unused_scene_expectation_labels": ["UNUSED"],
            "duplicate_scene_paths": ["/tmp/a.blend"],
            "duplicate_scene_labels": ["BARBERSHOP_INTERIOR"],
        }

        summary = PROBE.build_summary(results, failures, metadata)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["warning_count"], 5)
        self.assertTrue(any("telemetry-only" in warning for warning in summary["warnings"]))
        self.assertTrue(any("defaults only" in warning for warning in summary["warnings"]))
        self.assertTrue(any("unused scene expectation labels" in warning for warning in summary["warnings"]))
        self.assertTrue(any("duplicate scene paths requested" in warning for warning in summary["warnings"]))
        self.assertTrue(any("duplicate scene labels requested" in warning for warning in summary["warnings"]))

    def test_build_payload_metadata_records_invocation_paths(self):
        args = SimpleNamespace(
            blender_bin=Path("/tmp/blender"),
            scene=["/tmp/a.blend", "/tmp/b.blend", "/tmp/a.blend"],
            expectations_json=EXPECTATIONS_PATH,
            artifacts_dir=Path("/tmp/artifacts"),
        )

        metadata = PROBE.build_payload_metadata(args, {"A": {"min_global_luma": 0.2}, "UNUSED": {}})

        self.assertEqual(metadata["probe_name"], "eevee_hwrt_production_scene_probe")
        self.assertEqual(metadata["payload_schema_version"], 1)
        datetime.fromisoformat(metadata["generated_at_utc"])
        self.assertTrue(metadata["generated_at_utc"].endswith("+00:00"))
        self.assertEqual(metadata["python_executable"], sys.executable)
        self.assertEqual(metadata["python_version"], sys.version.split()[0])
        self.assertEqual(len(metadata["probe_source_sha256"]), 64)
        self.assertEqual(metadata["expectations_sha256"], PROBE.compute_file_sha256(EXPECTATIONS_PATH))
        self.assertEqual(metadata["blender_bin"], "/tmp/blender")
        self.assertEqual(metadata["scene_paths"], ["/tmp/a.blend", "/tmp/b.blend", "/tmp/a.blend"])
        self.assertEqual(metadata["scene_labels"], ["A", "B", "A"])
        self.assertEqual(metadata["expectations_json"], str(EXPECTATIONS_PATH))
        self.assertEqual(metadata["artifacts_dir"], "/tmp/artifacts")
        self.assertEqual(metadata["scene_labels_with_overrides"], ["A"])
        self.assertEqual(metadata["scene_labels_using_defaults_only"], ["B"])
        self.assertEqual(metadata["unused_scene_expectation_labels"], ["UNUSED"])
        self.assertEqual(metadata["duplicate_scene_paths"], ["/tmp/a.blend"])
        self.assertEqual(metadata["duplicate_scene_labels"], ["A"])

    def test_build_effective_expectations_merges_defaults_and_scene_values(self):
        defaults = {"require_exit_code_zero": True, "require_metrics": True, "expected_fast_gi_scene": "mixed"}
        scene_expectations = {"require_metrics": False, "max_fast_gi_traced_ms": 120.0}

        expectations = PROBE.build_effective_expectations(defaults, scene_expectations)

        self.assertEqual(
            expectations,
            {
                "require_exit_code_zero": True,
                "require_metrics": False,
                "expected_fast_gi_scene": "mixed",
                "max_fast_gi_traced_ms": 120.0,
            },
        )

    def test_build_expectation_provenance_tracks_default_and_override_keys(self):
        defaults = {"require_exit_code_zero": True, "require_metrics": True}
        scene_expectations = {"max_fast_gi_traced_ms": 120.0}

        provenance = PROBE.build_expectation_provenance(defaults, scene_expectations)

        self.assertEqual(
            provenance,
            {
                "used_scene_override": True,
                "default_keys": ["require_exit_code_zero", "require_metrics"],
                "override_keys": ["max_fast_gi_traced_ms"],
            },
        )

    def test_evaluate_result_accepts_known_dark_allowlisted_scene(self):
        result = {
            "label": "POOL",
            "exit_code": 0,
            "metrics": {"global_luma": 0.0, "center_luma": 0.0},
            "stats": {"scene": "interior", "budget": "indirect", "memory_limited": 0, "smoothed_traced_ms": 44.0},
        }
        defaults = {
            "require_exit_code_zero": True,
            "require_metrics": True,
            "require_stats": True,
            "fail_if_effectively_black": True,
            "require_memory_limited_zero": True,
        }
        scene_expectations = {
            "fail_if_effectively_black": False,
            "expected_fast_gi_scene": "interior",
            "expected_fast_gi_budget": "indirect",
            "max_fast_gi_traced_ms": 120.0,
        }

        expectations = PROBE.build_effective_expectations(defaults, scene_expectations)
        failures = PROBE.evaluate_result(result, expectations)
        self.assertEqual(failures, [])

    def test_evaluate_result_reports_multiple_failures(self):
        result = {
            "label": "BARBERSHOP_INTERIOR",
            "exit_code": 2,
            "metrics": {"global_luma": 0.05, "center_luma": 0.04},
            "stats": {"scene": "interior", "budget": "indirect", "memory_limited": 1, "smoothed_traced_ms": 300.0},
        }
        defaults = {
            "require_exit_code_zero": True,
            "require_memory_limited_zero": True,
        }
        scene_expectations = {
            "min_global_luma": 0.15,
            "min_center_luma": 0.15,
            "expected_fast_gi_scene": "mixed",
            "expected_fast_gi_budget": "balanced",
            "max_fast_gi_traced_ms": 250.0,
        }

        expectations = PROBE.build_effective_expectations(defaults, scene_expectations)
        failures = PROBE.evaluate_result(result, expectations)

        self.assertTrue(any("expected EXIT_CODE=0" in failure for failure in failures))
        self.assertTrue(any("GLOBAL_LUMA" in failure for failure in failures))
        self.assertTrue(any("CENTER_LUMA" in failure for failure in failures))
        self.assertTrue(any("FAST_GI_SCENE" in failure for failure in failures))
        self.assertTrue(any("FAST_GI_BUDGET" in failure for failure in failures))
        self.assertTrue(any("FAST_GI_MEMORY_LIMITED" in failure for failure in failures))
        self.assertTrue(any("FAST_GI_TRACED_MS" in failure for failure in failures))

    def test_evaluate_result_reports_missing_metrics_and_stats(self):
        result = {
            "label": "BARBERSHOP_INTERIOR",
            "exit_code": 0,
            "metrics": {},
            "stats": {},
        }
        defaults = {
            "require_exit_code_zero": True,
            "require_metrics": True,
            "require_stats": True,
        }

        expectations = PROBE.build_effective_expectations(defaults, {})
        failures = PROBE.evaluate_result(result, expectations)

        self.assertTrue(any("missing metrics payload" in failure for failure in failures))
        self.assertTrue(any("missing stats payload" in failure for failure in failures))


if __name__ == "__main__":
    remaining = [arg for arg in sys.argv[1:] if arg != "--"]
    unittest.main(argv=sys.argv[0:1] + remaining)
