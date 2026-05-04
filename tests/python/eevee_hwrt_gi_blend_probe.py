#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _inside_blender():
    try:
        import bpy  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Probe HWRT GI on test.blend with shadows off/on."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--blender-bin", type=Path, default=None)
    parser.add_argument("--scene-path", type=Path, default=Path("scenes/test.blend"))
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--radius-px", type=int, default=56)
    parser.add_argument("--redraw-iterations", type=int, default=20)
    parser.add_argument("--inner-case", choices=("gi_off", "gi_on_shadows_off", "gi_on_shadows_on"), default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--min-fast-wall-luma", type=float, default=None)
    parser.add_argument("--max-gi-off-wall-luma", type=float, default=None)
    parser.add_argument("--min-shadow-retention", type=float, default=None)
    return parser.parse_args(argv)


PATCH_POINTS = {
    "left_wall": (-7.2, 0.0, 0.5),
    "back_wall": (0.0, 7.2, 0.5),
}


def print_metric(name: str, value: float):
    print(f"{name}={value:.6f}", flush=True)


def run_case(blender_bin: Path, scene_path: Path, script_path: Path, output_dir: Path, case: str, args):
    metrics_path = output_dir / f"{case}_metrics.json"
    command = [
        str(blender_bin),
        str(scene_path),
        "-P",
        str(script_path),
        "--",
        "--inner-case",
        case,
        "--metrics-json",
        str(metrics_path),
        "--output-dir",
        str(output_dir),
        "--samples",
        str(args.samples),
        "--radius-px",
        str(args.radius_px),
        "--redraw-iterations",
        str(args.redraw_iterations),
    ]
    env = os.environ.copy()
    env.setdefault("BLENDER_EEVEE_HWRT_FAST_GI_ALLOW_SCREEN_SEED", "0")
    result = subprocess.run(command, env=env, check=False)
    if not metrics_path.exists():
        raise RuntimeError(f"{case} run did not produce metrics at {metrics_path} (exit={result.returncode})")
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, result.returncode


def run_outer(args):
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_gi_blend_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    blender_bin = args.blender_bin or Path(os.environ.get("BLENDER_BIN", ""))
    if not blender_bin or not blender_bin.exists():
        raise RuntimeError("--blender-bin is required when running outside Blender.")

    scene_path = args.scene_path.resolve()
    script_path = Path(__file__).resolve()
    gi_off, gi_off_code = run_case(blender_bin, scene_path, script_path, output_dir, "gi_off", args)
    gi_on_off, gi_on_off_code = run_case(
        blender_bin, scene_path, script_path, output_dir, "gi_on_shadows_off", args
    )
    gi_on_on, gi_on_on_code = run_case(
        blender_bin, scene_path, script_path, output_dir, "gi_on_shadows_on", args
    )

    shadow_retention = min(gi_on_off["wall_luma"], gi_on_on["wall_luma"]) / max(
        max(gi_on_off["wall_luma"], gi_on_on["wall_luma"]), 1.0e-6
    )

    print(f"PATCH_BOUNDS={gi_off['patch_bounds']}", flush=True)
    print(f"GI_OFF_EXIT_CODE={gi_off_code}", flush=True)
    print(f"GI_ON_SHADOWS_OFF_EXIT_CODE={gi_on_off_code}", flush=True)
    print(f"GI_ON_SHADOWS_ON_EXIT_CODE={gi_on_on_code}", flush=True)
    print_metric("GI_BLEND_GI_OFF_WALL_LUMA", gi_off["wall_luma"])
    print_metric("GI_BLEND_GI_ON_SHADOWS_OFF_WALL_LUMA", gi_on_off["wall_luma"])
    print_metric("GI_BLEND_GI_ON_SHADOWS_ON_WALL_LUMA", gi_on_on["wall_luma"])
    print_metric("GI_BLEND_SHADOW_TOGGLE_RETENTION", shadow_retention)
    for name in PATCH_POINTS:
        key = f"{name}_luma"
        print_metric(f"GI_BLEND_GI_OFF_{name.upper()}_LUMA", gi_off[key])
        print_metric(f"GI_BLEND_GI_ON_SHADOWS_OFF_{name.upper()}_LUMA", gi_on_off[key])
        print_metric(f"GI_BLEND_GI_ON_SHADOWS_ON_{name.upper()}_LUMA", gi_on_on[key])
    print(f"OUTPUT_DIR={output_dir}", flush=True)

    failures = []
    if args.max_gi_off_wall_luma is not None and gi_off["wall_luma"] > args.max_gi_off_wall_luma:
        failures.append(
            f"GI off wall luma expected <= {args.max_gi_off_wall_luma:.6f}, got {gi_off['wall_luma']:.6f}"
        )
    if args.min_fast_wall_luma is not None:
        if gi_on_off["wall_luma"] < args.min_fast_wall_luma:
            failures.append(
                f"GI on wall luma with shadows off expected >= {args.min_fast_wall_luma:.6f}, got {gi_on_off['wall_luma']:.6f}"
            )
        if gi_on_on["wall_luma"] < args.min_fast_wall_luma:
            failures.append(
                f"GI on wall luma with shadows on expected >= {args.min_fast_wall_luma:.6f}, got {gi_on_on['wall_luma']:.6f}"
            )
    if args.min_shadow_retention is not None and shadow_retention < args.min_shadow_retention:
        failures.append(
            f"Shadow-toggle retention expected >= {args.min_shadow_retention:.6f}, got {shadow_retention:.6f}"
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


if _inside_blender():
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    def configure_scene(scene, samples: int, *, gi_mode: str, shadows: bool):
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"

        eevee = scene.eevee
        eevee.use_raytracing = True
        eevee.ray_tracing_method = "HARDWARE"
        eevee.use_hardware_raytracing_gi = (gi_mode == "ON")
        eevee.hardware_raytracing_reflection_mode = "OFF"
        eevee.hardware_raytracing_refraction_mode = "OFF"
        eevee.use_hardware_raytracing_environment = False
        eevee.use_hardware_raytracing_caustics = False
        eevee.use_hardware_raytracing_shadows = shadows
        eevee.taa_samples = max(1, samples)
        eevee.taa_render_samples = max(1, samples)

        ray_tracing = eevee.ray_tracing_options
        ray_tracing.resolution_scale = "1"
        ray_tracing.use_denoise = False
        ray_tracing.denoise_spatial = False
        ray_tracing.denoise_temporal = False
        ray_tracing.denoise_bilateral = False


    def find_view3d_context():
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type != "VIEW_3D":
                    continue
                region = next((region for region in area.regions if region.type == "WINDOW"), None)
                if region is None:
                    continue
                return window, area, region, area.spaces.active
        raise RuntimeError("No VIEW_3D area available for GI blend probe.")


    def force_viewport_camera(window, area, region, space):
        space.shading.type = "RENDERED"
        space.overlay.show_overlays = False
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.view3d.view_camera()


    def redraw(iterations: int):
        for _ in range(iterations):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


    def capture_viewport(window, area, region, output_dir: Path, tag: str):
        path = output_dir / f"{tag}.png"
        bpy.context.scene.render.filepath = str(path)
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.render.opengl(write_still=True, view_context=True)
        image = bpy.data.images.load(str(path), check_existing=False)
        width, height = image.size
        pixels = list(image.pixels[:])
        bpy.data.images.remove(image)
        return width, height, pixels


    def projected_crop_bounds(scene, camera, point: Vector, width: int, height: int, radius_px: int):
        co = world_to_camera_view(scene, camera, point)
        center_x = int(round(co.x * width))
        center_y = int(round(co.y * height))
        min_x = max(0, center_x - radius_px)
        max_x = min(width - 1, center_x + radius_px)
        min_y = max(0, center_y - radius_px)
        max_y = min(height - 1, center_y + radius_px)
        return (min_x, min_y, max_x, max_y)


    def crop_mean_rgb(pixels, width: int, bounds):
        min_x, min_y, max_x, max_y = bounds
        total = [0.0, 0.0, 0.0]
        count = 0
        for y in range(min_y, max_y + 1):
            row = y * width * 4
            for x in range(min_x, max_x + 1):
                base = row + x * 4
                total[0] += pixels[base + 0]
                total[1] += pixels[base + 1]
                total[2] += pixels[base + 2]
                count += 1
        return tuple(channel / max(1, count) for channel in total)


    def mean_luma(rgb):
        return (rgb[0] + rgb[1] + rgb[2]) / 3.0


    def wall_metrics(scene, camera, width: int, height: int, pixels, radius_px: int):
        patch_lumas = {}
        patch_bounds = {}
        for name, point in PATCH_POINTS.items():
            bounds = projected_crop_bounds(scene, camera, Vector(point), width, height, radius_px)
            patch_bounds[name] = bounds
            patch_lumas[name] = mean_luma(crop_mean_rgb(pixels, width, bounds))
        return {
            "wall_luma": sum(patch_lumas.values()) / max(1, len(patch_lumas)),
            "left_wall_luma": patch_lumas["left_wall"],
            "back_wall_luma": patch_lumas["back_wall"],
            "patch_bounds": patch_bounds,
        }


    def run_inner(args):
        output_dir = args.output_dir
        if output_dir is None:
            raise RuntimeError("--output-dir is required for inner GI blend probe runs.")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.metrics_json is None:
            raise RuntimeError("--metrics-json is required for inner GI blend probe runs.")
        if args.inner_case is None:
            raise RuntimeError("--inner-case is required for inner GI blend probe runs.")

        scene = bpy.context.scene
        camera = scene.camera
        if camera is None:
            raise RuntimeError("The selected validation scene requires an active camera.")
        window, area, region, space = find_view3d_context()

        if args.inner_case == "gi_off":
            configure_scene(scene, args.samples, gi_mode="OFF", shadows=False)
        elif args.inner_case == "gi_on_shadows_off":
            configure_scene(scene, args.samples, gi_mode="ON", shadows=False)
        else:
            configure_scene(scene, args.samples, gi_mode="ON", shadows=True)

        bpy.context.view_layer.update()
        force_viewport_camera(window, area, region, space)
        redraw(args.redraw_iterations)
        width, height, pixels = capture_viewport(window, area, region, output_dir, args.inner_case)
        payload = wall_metrics(scene, camera, width, height, pixels, args.radius_px)
        with args.metrics_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        bpy.ops.wm.quit_blender()


    run_inner(_parse_args())
else:
    run_outer(_parse_args())
