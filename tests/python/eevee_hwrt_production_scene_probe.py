#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone


STATS_PREFIX = "EEVEE HWRT FastGI stats "
PAIR_RE = re.compile(r"(\w+)=([^\s]+)")
EFFECTIVELY_BLACK_LUMA = 1.0e-3
OUTPUT_TAIL_LINE_COUNT = 40
PROBE_NAME = "eevee_hwrt_production_scene_probe"
PAYLOAD_SCHEMA_VERSION = 1
ALLOWED_EXPECTATION_KEYS = {
    "require_exit_code_zero",
    "require_metrics",
    "require_stats",
    "fail_if_effectively_black",
    "require_memory_limited_zero",
    "min_global_luma",
    "min_center_luma",
    "expected_fast_gi_scene",
    "expected_fast_gi_budget",
    "max_fast_gi_traced_ms",
}
BOOLEAN_EXPECTATION_KEYS = {
    "require_exit_code_zero",
    "require_metrics",
    "require_stats",
    "fail_if_effectively_black",
    "require_memory_limited_zero",
}
NUMERIC_EXPECTATION_KEYS = {
    "min_global_luma",
    "min_center_luma",
    "max_fast_gi_traced_ms",
}
STRING_EXPECTATION_KEYS = {
    "expected_fast_gi_scene",
    "expected_fast_gi_budget",
}
ALLOWED_EXPECTATIONS_TOP_LEVEL_KEYS = {"defaults", "scenes"}


def _inside_blender():
    try:
        import bpy  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run focused HWRT production-scene renders and capture coarse image/runtime telemetry."
    )
    if _inside_blender():
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []
        parser.add_argument("--inner", action="store_true")
        parser.add_argument("--scene-label", type=str, required=True)
        parser.add_argument("--metrics-json", type=Path, required=True)
        parser.add_argument("--output-image", type=Path, required=True)
    else:
        parser.add_argument("--blender-bin", type=Path, required=True)
        parser.add_argument("--scene", action="append", required=True)
        parser.add_argument("--output-json", type=Path, default=None)
        parser.add_argument("--expectations-json", type=Path, default=None)
        parser.add_argument("--artifacts-dir", type=Path, default=None)
    return parser.parse_args(argv if _inside_blender() else None)


def parse_stats_line(text: str):
    def parse_scalar(value: str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    parsed = {}
    for key, raw_value in PAIR_RE.findall(text):
        value = raw_value.rstrip(",")
        parsed[key] = parse_scalar(value)
    return parsed


def extract_last_stats(stdout: str):
    last = None
    for line in stdout.splitlines():
        if STATS_PREFIX in line:
            last = line[line.index(STATS_PREFIX) + len(STATS_PREFIX) :]
    return parse_stats_line(last) if last is not None else {}


def scene_label_from_path(path: Path):
    return path.stem.replace("-", "_").replace(" ", "_").upper()


def trim_output_tail(text: str, max_lines: int = OUTPUT_TAIL_LINE_COUNT):
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def compute_file_sha256(path: Path | None):
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_expectations_map(label: str, data):
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    unknown_keys = sorted(set(data.keys()) - ALLOWED_EXPECTATION_KEYS)
    if unknown_keys:
        raise ValueError(f"{label} contains unknown expectation keys: {', '.join(unknown_keys)}")
    for key, value in data.items():
        if key in BOOLEAN_EXPECTATION_KEYS and not isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be a boolean")
        if key in NUMERIC_EXPECTATION_KEYS and not _is_number(value):
            raise ValueError(f"{label}.{key} must be a number")
        if key in STRING_EXPECTATION_KEYS and not isinstance(value, str):
            raise ValueError(f"{label}.{key} must be a string")


def load_expectations(path: Path | None):
    if path is None:
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expectations payload must be a JSON object")
    unknown_top_level_keys = sorted(set(payload.keys()) - ALLOWED_EXPECTATIONS_TOP_LEVEL_KEYS)
    if unknown_top_level_keys:
        raise ValueError(
            f"expectations payload contains unknown top-level keys: {', '.join(unknown_top_level_keys)}"
        )
    defaults = payload.get("defaults", {})
    scenes = payload.get("scenes", {})
    validate_expectations_map("defaults", defaults)
    if not isinstance(scenes, dict):
        raise ValueError("scenes must be a JSON object")
    for scene_label, scene_expectations in scenes.items():
        if not isinstance(scene_label, str):
            raise ValueError("scene labels must be strings")
        validate_expectations_map(f"scenes.{scene_label}", scene_expectations)
    return defaults, scenes


def build_effective_expectations(defaults, scene_expectations):
    expectations = dict(defaults)
    expectations.update(scene_expectations or {})
    return expectations


def build_expectation_provenance(defaults, scene_expectations):
    scene_expectations = scene_expectations or {}
    return {
        "used_scene_override": bool(scene_expectations),
        "default_keys": sorted(defaults.keys()),
        "override_keys": sorted(scene_expectations.keys()),
    }


def evaluate_result(result, expectations):

    failures = []
    label = result["label"]
    metrics = result["metrics"]
    stats = result["stats"]

    if expectations.get("require_exit_code_zero", False) and result["exit_code"] != 0:
        failures.append(f"{label}: expected EXIT_CODE=0, got {result['exit_code']}")

    if expectations.get("require_metrics", False) and not metrics:
        failures.append(f"{label}: missing metrics payload")

    if expectations.get("require_stats", False) and not stats:
        failures.append(f"{label}: missing stats payload")

    if expectations.get("fail_if_effectively_black", False):
        global_luma = float(metrics.get("global_luma", 0.0))
        center_luma = float(metrics.get("center_luma", 0.0))
        if global_luma < EFFECTIVELY_BLACK_LUMA and center_luma < EFFECTIVELY_BLACK_LUMA:
            failures.append(
                f"{label}: render is effectively black "
                f"(global={global_luma:.6f}, center={center_luma:.6f})"
            )

    if "min_global_luma" in expectations and not metrics:
        failures.append(f"{label}: missing metrics for min_global_luma check")
    elif "min_global_luma" in expectations:
        global_luma = float(metrics.get("global_luma", 0.0))
        minimum = float(expectations["min_global_luma"])
        if global_luma < minimum:
            failures.append(f"{label}: GLOBAL_LUMA {global_luma:.6f} < {minimum:.6f}")

    if "min_center_luma" in expectations and not metrics:
        failures.append(f"{label}: missing metrics for min_center_luma check")
    elif "min_center_luma" in expectations:
        center_luma = float(metrics.get("center_luma", 0.0))
        minimum = float(expectations["min_center_luma"])
        if center_luma < minimum:
            failures.append(f"{label}: CENTER_LUMA {center_luma:.6f} < {minimum:.6f}")

    if "expected_fast_gi_scene" in expectations and not stats:
        failures.append(f"{label}: missing stats for expected_fast_gi_scene check")
    elif "expected_fast_gi_scene" in expectations:
        scene_value = stats.get("scene", "unknown")
        expected = expectations["expected_fast_gi_scene"]
        if scene_value != expected:
            failures.append(f"{label}: FAST_GI_SCENE {scene_value!r} != {expected!r}")

    if "expected_fast_gi_budget" in expectations and not stats:
        failures.append(f"{label}: missing stats for expected_fast_gi_budget check")
    elif "expected_fast_gi_budget" in expectations:
        budget_value = stats.get("budget", "unknown")
        expected = expectations["expected_fast_gi_budget"]
        if budget_value != expected:
            failures.append(f"{label}: FAST_GI_BUDGET {budget_value!r} != {expected!r}")

    if expectations.get("require_memory_limited_zero", False):
        if not stats:
            failures.append(f"{label}: missing stats for memory_limited check")
        elif int(stats.get("memory_limited", -1)) != 0:
            failures.append(f"{label}: FAST_GI_MEMORY_LIMITED={stats.get('memory_limited')}")

    if "max_fast_gi_traced_ms" in expectations:
        if not stats:
            failures.append(f"{label}: missing stats for traced_ms check")
        else:
            traced_ms = float(stats.get("smoothed_traced_ms", float("inf")))
            maximum = float(expectations["max_fast_gi_traced_ms"])
            if traced_ms > maximum:
                failures.append(f"{label}: FAST_GI_TRACED_MS {traced_ms:.6f} > {maximum:.6f}")

    return failures


def persist_scene_artifacts(
    artifacts_dir: Path | None,
    artifact_stem: str,
    output_image: Path,
    metrics_json: Path,
    stdout_text: str,
    stderr_text: str,
):
    if artifacts_dir is None:
        return {}

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {}

    image_target = artifacts_dir / f"{artifact_stem}_render.png"
    if output_image.exists():
        shutil.copy2(output_image, image_target)
        artifact_paths["render_image"] = str(image_target)

    metrics_target = artifacts_dir / f"{artifact_stem}_metrics.json"
    if metrics_json.exists():
        shutil.copy2(metrics_json, metrics_target)
        artifact_paths["metrics_json"] = str(metrics_target)

    stdout_target = artifacts_dir / f"{artifact_stem}_stdout.log"
    stdout_target.write_text(stdout_text, encoding="utf-8")
    artifact_paths["stdout_log"] = str(stdout_target)

    stderr_target = artifacts_dir / f"{artifact_stem}_stderr.log"
    stderr_target.write_text(stderr_text, encoding="utf-8")
    artifact_paths["stderr_log"] = str(stderr_target)

    return artifact_paths


def build_summary(results, failures, metadata=None):
    metadata = metadata or {}
    failed_labels = [result["label"] for result in results if result.get("failures")]
    failed_request_ids = [result["request_id"] for result in results if result.get("failures")]
    warnings = []
    if metadata.get("expectations_json") is None:
        warnings.append("expectations-json not provided; results are telemetry-only")
    defaults_only = metadata.get("scene_labels_using_defaults_only", [])
    if defaults_only:
        warnings.append(f"scene labels using defaults only: {', '.join(defaults_only)}")
    unused_overrides = metadata.get("unused_scene_expectation_labels", [])
    if unused_overrides:
        warnings.append(f"unused scene expectation labels: {', '.join(unused_overrides)}")
    duplicate_scene_paths = metadata.get("duplicate_scene_paths", [])
    if duplicate_scene_paths:
        warnings.append(f"duplicate scene paths requested: {', '.join(duplicate_scene_paths)}")
    duplicate_scene_labels = metadata.get("duplicate_scene_labels", [])
    if duplicate_scene_labels:
        warnings.append(f"duplicate scene labels requested: {', '.join(duplicate_scene_labels)}")
    return {
        "status": "fail" if failures else "pass",
        "scene_count": len(results),
        "failure_count": len(failures),
        "failed_labels": failed_labels,
        "failed_request_ids": failed_request_ids,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def build_payload_metadata(args, scene_expectations=None):
    scene_expectations = scene_expectations or {}
    scene_paths = [str(Path(scene)) for scene in args.scene]
    scene_labels = [scene_label_from_path(Path(scene)) for scene in args.scene]
    requested_label_set = set(scene_labels)
    override_label_set = set(scene_expectations.keys())
    duplicate_scene_paths = sorted({path for path in scene_paths if scene_paths.count(path) > 1})
    duplicate_scene_labels = sorted({label for label in scene_labels if scene_labels.count(label) > 1})
    expectations_path = Path(args.expectations_json) if args.expectations_json is not None else None

    return {
        "probe_name": PROBE_NAME,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "probe_source_sha256": compute_file_sha256(Path(__file__).resolve()),
        "expectations_sha256": compute_file_sha256(expectations_path),
        "blender_bin": str(args.blender_bin),
        "scene_paths": scene_paths,
        "scene_labels": scene_labels,
        "expectations_json": str(args.expectations_json) if args.expectations_json is not None else None,
        "artifacts_dir": str(args.artifacts_dir) if args.artifacts_dir is not None else None,
        "scene_labels_with_overrides": sorted(requested_label_set & override_label_set),
        "scene_labels_using_defaults_only": sorted(requested_label_set - override_label_set),
        "unused_scene_expectation_labels": sorted(override_label_set - requested_label_set),
        "duplicate_scene_paths": duplicate_scene_paths,
        "duplicate_scene_labels": duplicate_scene_labels,
    }


def build_artifact_stem(label: str, request_index: int, duplicate_labels):
    artifact_stem = label.lower()
    if label in duplicate_labels:
        artifact_stem = f"{artifact_stem}_{request_index:02d}"
    return artifact_stem


def build_request_id(label: str, request_index: int, duplicate_labels):
    if label in duplicate_labels:
        return f"{label}[{request_index:02d}]"
    return label


def run_scene(
    blender_bin: Path,
    scene_path: Path,
    artifacts_dir: Path | None = None,
    artifact_stem: str | None = None,
    request_index: int | None = None,
    request_id: str | None = None,
):
    with tempfile.TemporaryDirectory(prefix="eevee_hwrt_production_scene_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        metrics_json = tmpdir_path / "metrics.json"
        output_image = tmpdir_path / "render.png"
        label = scene_label_from_path(scene_path)
        cmd = [
            str(blender_bin),
            "-b",
            str(scene_path),
            "-P",
            str(Path(__file__).resolve()),
            "--",
            "--inner",
            "--scene-label",
            label,
            "--metrics-json",
            str(metrics_json),
            "--output-image",
            str(output_image),
        ]
        env = os.environ.copy()
        env["BLENDER_EEVEE_HWRT_FAST_GI_STATS"] = "1"
        env["BLENDER_GPU_METAL_IGNORE_TEXTURE_POOL_ASSERT"] = "1"
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
        combined_output = completed.stdout + "\n" + completed.stderr
        stats = extract_last_stats(combined_output)
        metrics = {}
        if metrics_json.exists():
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        artifact_paths = persist_scene_artifacts(
            artifacts_dir,
            artifact_stem or label.lower(),
            output_image,
            metrics_json,
            completed.stdout,
            completed.stderr,
        )
        return {
            "scene": str(scene_path),
            "label": label,
            "request_index": request_index,
            "request_id": request_id or label,
            "exit_code": completed.returncode,
            "stats": stats,
            "metrics": metrics,
            "stdout_tail": trim_output_tail(completed.stdout),
            "stderr_tail": trim_output_tail(completed.stderr),
            "artifacts": artifact_paths,
        }


def outer_main():
    args = _parse_args()
    defaults, scene_expectations = load_expectations(args.expectations_json)
    results = []
    failures = []
    scene_labels = [scene_label_from_path(Path(scene)) for scene in args.scene]
    duplicate_labels = {label for label in scene_labels if scene_labels.count(label) > 1}

    for index, scene in enumerate(args.scene):
        scene_path = Path(scene)
        label = scene_label_from_path(scene_path)
        artifact_stem = build_artifact_stem(label, index, duplicate_labels)
        request_id = build_request_id(label, index, duplicate_labels)
        result = run_scene(
            args.blender_bin,
            scene_path,
            args.artifacts_dir,
            artifact_stem,
            index,
            request_id,
        )
        local_scene_expectations = scene_expectations.get(result["label"], {})
        result["effective_expectations"] = build_effective_expectations(
            defaults, local_scene_expectations
        )
        result["expectation_provenance"] = build_expectation_provenance(defaults, local_scene_expectations)
        result["failures"] = evaluate_result(result, result["effective_expectations"])
        result["status"] = "fail" if result["failures"] else "pass"
        results.append(result)
        failures.extend(result["failures"])

    metadata = build_payload_metadata(args, scene_expectations)
    payload = {
        "metadata": metadata,
        "results": results,
        "failures": failures,
        "summary": build_summary(results, failures, metadata),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for result in results:
        label = result["label"]
        request_id = result["request_id"]
        metrics = result["metrics"]
        stats = result["stats"]
        print(f"{label}_EXIT_CODE={result['exit_code']}")
        print(f"{label}_REQUEST_ID={request_id}")
        if metrics:
            print(f"{label}_GLOBAL_LUMA={metrics.get('global_luma', 0.0):.6f}")
            print(f"{label}_CENTER_LUMA={metrics.get('center_luma', 0.0):.6f}")
        if stats:
            print(f"{label}_FAST_GI_SCENE={stats.get('scene', 'unknown')}")
            print(f"{label}_FAST_GI_BUDGET={stats.get('budget', 'unknown')}")
            print(f"{label}_FAST_GI_TIER={stats.get('tier', 'unknown')}")
            print(f"{label}_FAST_GI_ACTIVE_CASCADES={stats.get('active_cascades', 'unknown')}")
            print(f"{label}_FAST_GI_MEMORY_LIMITED={stats.get('memory_limited', -1)}")
            print(f"{label}_FAST_GI_MEM_MIB={float(stats.get('fast_gi_mem_mib', 0.0)):.6f}")
            print(f"{label}_FAST_GI_TRACED_MS={float(stats.get('smoothed_traced_ms', 0.0)):.6f}")
        for artifact_name, artifact_path in result["artifacts"].items():
            print(f"{label}_{artifact_name.upper()}={artifact_path}")
        for failure in result["failures"]:
            print(f"FAILURE={failure}")

    print(f"SUMMARY_STATUS={payload['summary']['status']}")
    print(f"SUMMARY_SCENE_COUNT={payload['summary']['scene_count']}")
    print(f"SUMMARY_FAILURE_COUNT={payload['summary']['failure_count']}")
    print(f"SUMMARY_WARNING_COUNT={payload['summary']['warning_count']}")
    if payload["summary"]["failed_labels"]:
        print(f"SUMMARY_FAILED_LABELS={','.join(payload['summary']['failed_labels'])}")
    if payload["summary"]["failed_request_ids"]:
        print(f"SUMMARY_FAILED_REQUEST_IDS={','.join(payload['summary']['failed_request_ids'])}")
    for warning in payload["summary"]["warnings"]:
        print(f"WARNING={warning}")

    return 1 if failures else 0


def configure_scene(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = max(scene.render.resolution_x, 960)
    scene.render.resolution_y = max(scene.render.resolution_y, 540)
    scene.render.resolution_percentage = 100

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.use_hardware_raytracing_gi = True
    eevee.hardware_raytracing_reflection_mode = "FULL"
    eevee.hardware_raytracing_refraction_mode = "HYBRID"
    eevee.use_hardware_raytracing_environment = True
    eevee.use_hardware_raytracing_shadows = True
    eevee.use_hardware_raytracing_caustics = False
    eevee.taa_render_samples = 8
    eevee.taa_samples = 8

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.use_denoise = True
    ray_tracing.denoise_spatial = True
    ray_tracing.denoise_temporal = True
    ray_tracing.denoise_bilateral = True


def image_luma_metrics(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)

    total_luma = 0.0
    total_count = 0
    center_luma = 0.0
    center_count = 0
    min_x = width // 4
    max_x = (width * 3) // 4
    min_y = height // 4
    max_y = (height * 3) // 4

    for y in range(height):
        row = y * width * 4
        for x in range(width):
            base = row + x * 4
            r = pixels[base + 0]
            g = pixels[base + 1]
            b = pixels[base + 2]
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            total_luma += luma
            total_count += 1
            if min_x <= x < max_x and min_y <= y < max_y:
                center_luma += luma
                center_count += 1

    return {
        "width": width,
        "height": height,
        "global_luma": total_luma / max(1, total_count),
        "center_luma": center_luma / max(1, center_count),
    }


def inner_main():
    global bpy
    import bpy as blender_bpy

    bpy = blender_bpy

    args = _parse_args()
    scene = bpy.context.scene
    configure_scene(scene)
    scene.render.filepath = str(args.output_image)
    bpy.ops.render.render(write_still=True)
    metrics = image_luma_metrics(args.output_image)
    metrics["scene_label"] = args.scene_label
    args.metrics_json.write_text(json.dumps(metrics, separators=(",", ":")), encoding="utf-8")
    print(f"METRICS_CAPTURED={args.scene_label} path={args.metrics_json}", flush=True)


def main():
    return inner_main() if _inside_blender() else outer_main()


if __name__ == "__main__":
    raise SystemExit(main())
