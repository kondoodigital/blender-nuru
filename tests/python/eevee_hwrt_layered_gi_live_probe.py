#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
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
        description="Probe live viewport response for layered Hardware RT GI controls on scenes/test.blend."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--redraw-iterations", type=int, default=8)
    parser.add_argument("--taa-samples", type=int, default=1)
    parser.add_argument("--hold-open", action="store_true")
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
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


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


def redraw(iterations: int):
    for _ in range(iterations):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def load_pixels(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


def capture_viewport(window, area, region, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    bpy.context.scene.render.filepath = str(path)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.render.opengl(write_still=True, view_context=True)
    width, height, pixels = load_pixels(path)
    return width, height, pixels


def object_bbox_pixels(scene, camera, obj, width: int, height: int, margin_px: int = 12):
    xs = []
    ys = []
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        co = world_to_camera_view(scene, camera, world_corner)
        xs.append(co.x)
        ys.append(co.y)

    min_x = max(0, math.floor(min(xs) * width) - margin_px)
    max_x = min(width - 1, math.ceil(max(xs) * width) + margin_px)
    min_y = max(0, math.floor(min(ys) * height) - margin_px)
    max_y = min(height - 1, math.ceil(max(ys) * height) + margin_px)
    return (min_x, min_y, max_x, max_y)


def crop_abs_diff_mean(pixels_a, pixels_b, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    total = 0.0
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            for channel in range(3):
                total += abs(pixels_a[base + channel] - pixels_b[base + channel])
                count += 1
    return total / max(1, count)


def apply_layered_state(scene, enabled: bool):
    eevee = scene.eevee
    eevee.use_hardware_raytracing_gi = enabled
    eevee.use_hardware_raytracing_caustics = enabled
    eevee.ray_tracing_caustics_samples = 32 if enabled else 8


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_layered_gi_live_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    configure_scene(scene, args.taa_samples)
    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    camera = scene.camera
    mirror = bpy.data.objects["Plane.001"]
    bpy.context.view_layer.update()

    apply_layered_state(scene, enabled=False)
    redraw(args.redraw_iterations)
    width, height, pixels_off = capture_viewport(window, area, region, output_dir, "layered_off")

    apply_layered_state(scene, enabled=True)
    redraw(args.redraw_iterations)
    _width, _height, pixels_on = capture_viewport(window, area, region, output_dir, "layered_on")

    mirror_bounds = object_bbox_pixels(scene, camera, mirror, width, height)
    mirror_diff = crop_abs_diff_mean(pixels_off, pixels_on, width, height, mirror_bounds)
    frame_diff = crop_abs_diff_mean(pixels_off, pixels_on, width, height, (0, 0, width - 1, height - 1))

    print(f"MIRROR_BOUNDS={mirror_bounds}")
    print(f"LAYERED_VIEWPORT_FRAME_DIFF_MEAN={frame_diff:.6f}")
    print(f"LAYERED_VIEWPORT_MIRROR_DIFF_MEAN={mirror_diff:.6f}")
    print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None and not args.hold_open:
        cleanup_dir.cleanup()

    if not args.hold_open:
        bpy.ops.wm.quit_blender()


main()
