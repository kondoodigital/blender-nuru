#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import tempfile
from pathlib import Path


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Warm Fast GI, move the camera, and let sparse brick updates settle."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--redraw-iterations", type=int, default=16)
    parser.add_argument("--move-distance", type=float, default=20.0)
    parser.add_argument("--taa-samples", type=int, default=1)
    return parser.parse_args(argv)


def _inside_blender():
    try:
        import bpy  # noqa: F401

        return True
    except ImportError:
        return False


if not _inside_blender():
    raise RuntimeError("This script must run inside Blender.")

import bpy


def configure_scene(scene, taa_samples: int):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = "HARDWARE"
    scene.eevee.hardware_raytracing_reflection_mode = "FULL"
    scene.eevee.hardware_raytracing_refraction_mode = "FULL"
    scene.eevee.use_hardware_raytracing_environment = True
    scene.eevee.use_hardware_raytracing_shadows = True
    scene.eevee.use_hardware_raytracing_gi = True
    scene.eevee.use_hardware_raytracing_caustics = True
    scene.eevee.ray_tracing_caustics_samples = 32
    scene.eevee.taa_samples = max(1, taa_samples)
    scene.eevee.taa_render_samples = max(1, taa_samples)


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
    raise RuntimeError("No VIEW_3D area available for viewport probe.")


def force_viewport_camera(window, area, region, space):
    space.shading.type = "RENDERED"
    space.overlay.show_overlays = False
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_camera()


def move_viewport(space, distance: float):
    region_3d = space.region_3d
    if region_3d is None:
        raise RuntimeError("VIEW_3D region_3d is required for sparse live probe.")
    region_3d.view_location.x += distance


def redraw(iterations: int):
    for _ in range(iterations):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def capture_viewport(window, area, region, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=str(path))
    return path


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_fast_gi_sparse_live_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    configure_scene(scene, args.taa_samples)
    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    # Warm one traced field before the viewport move so camera invalidation is measured against
    # settled state instead of startup/bootstrap noise.
    redraw(args.redraw_iterations)
    capture_viewport(window, area, region, output_dir, "warm_start")

    move_viewport(space, args.move_distance)
    redraw(args.redraw_iterations)
    capture_viewport(window, area, region, output_dir, "after_move")

    print(f"OUTPUT_DIR={output_dir}")
    bpy.ops.wm.quit_blender()


main()
