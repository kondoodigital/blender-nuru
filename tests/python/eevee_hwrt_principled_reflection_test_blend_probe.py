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
        description="Probe Principled mirror replay in scenes/test.blend using a slight camera move."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--redraw-iterations", type=int, default=12)
    parser.add_argument("--camera-dx", type=float, default=0.15)
    parser.add_argument("--camera-dy", type=float, default=0.0)
    parser.add_argument("--camera-dz", type=float, default=0.0)
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


def configure_scene(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = "HARDWARE"
    scene.eevee.hardware_raytracing_reflection_mode = "FULL"
    scene.eevee.hardware_raytracing_refraction_mode = "FULL"
    scene.eevee.use_hardware_raytracing_environment = True
    scene.eevee.use_hardware_raytracing_shadows = True
    scene.eevee.taa_samples = 1
    scene.eevee.taa_render_samples = 1
    scene.eevee.ray_tracing_reflection_bounces = 4
    scene.eevee.ray_tracing_refraction_bounces = 4


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


def redraw(window, area, region, iterations: int):
    for _ in range(iterations):
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def load_pixels(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


def capture_viewport(scene, window, area, region, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    with bpy.context.temp_override(window=window, area=area, region=region):
        scene.render.filepath = str(path)
        bpy.ops.render.opengl(write_still=True, view_context=True)
    width, height, pixels = load_pixels(path)
    return {"path": path, "width": width, "height": height, "pixels": pixels}


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


def crop_rgb_mean(pixels, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    total = [0.0, 0.0, 0.0]
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            for channel in range(3):
                total[channel] += pixels[base + channel]
            count += 1
    return tuple(channel / max(1, count) for channel in total)


def lower_reflection_band_bounds(
    bounds, inset_x_frac: float = 0.18, start_y_frac: float = 0.56, end_y_frac: float = 0.94
):
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    inset_x = int(round(width * inset_x_frac))
    band_min_y = min_y + int(round((height - 1) * start_y_frac))
    band_max_y = min_y + int(round((height - 1) * end_y_frac))
    return (
        min(max_x, min_x + inset_x),
        max(min_y, min(max_y, band_min_y)),
        max(min_x, max_x - inset_x),
        max(min_y, min(max_y, band_max_y)),
    )


def crop_luma_mean_stddev(pixels, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    total = 0.0
    total_sq = 0.0
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            luma = (
                pixels[base + 0] * 0.2126
                + pixels[base + 1] * 0.7152
                + pixels[base + 2] * 0.0722
            )
            total += luma
            total_sq += luma * luma
            count += 1
    mean = total / max(1, count)
    variance = max(total_sq / max(1, count) - mean * mean, 0.0)
    return mean, math.sqrt(variance)


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


def projected_crop_bounds(scene, camera, point: Vector, width: int, height: int, radius_px: int = 90):
    co = world_to_camera_view(scene, camera, point)
    x = int(round(co.x * width))
    y = int(round(co.y * height))
    return (
        max(0, x - radius_px),
        max(0, y - radius_px),
        min(width - 1, x + radius_px),
        min(height - 1, y + radius_px),
    )


def mirror_reflection_point(point: Vector, plane_obj) -> Vector:
    plane_normal = (plane_obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0))).normalized()
    plane_origin = plane_obj.location
    offset = point - plane_origin
    return point - 2.0 * offset.dot(plane_normal) * plane_normal


def run_probe():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_principled_test_blend_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    configure_scene(scene)

    camera = bpy.data.objects["Camera"]
    sphere = bpy.data.objects["Sphere"]
    mirror = bpy.data.objects["Plane.004"]
    principled = next(
        node
        for node in bpy.data.materials["Material.003"].node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )

    original_camera_location = camera.location.copy()
    original_metallic = principled.inputs["Metallic"].default_value
    original_transmission = principled.inputs["Transmission Weight"].default_value
    original_roughness = principled.inputs["Roughness"].default_value
    original_ior = principled.inputs["IOR"].default_value

    camera.location = original_camera_location + Vector((args.camera_dx, args.camera_dy, args.camera_dz))
    principled.inputs["Roughness"].default_value = 0.0
    principled.inputs["IOR"].default_value = max(1.0, original_ior)
    bpy.context.view_layer.update()

    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)
    redraw(window, area, region, args.redraw_iterations)

    reflection_point = mirror_reflection_point(sphere.location, mirror)

    def set_principled(metallic: float, transmission: float):
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Transmission Weight"].default_value = transmission
        bpy.data.materials["Material.003"].node_tree.update_tag()
        sphere.update_tag()
        bpy.context.view_layer.update()

    def capture_case(tag: str):
        redraw(window, area, region, args.redraw_iterations)
        return capture_viewport(scene, window, area, region, output_dir, tag)

    set_principled(0.0, 0.0)
    metallic_base = capture_case("metallic_base")
    sphere_bounds = object_bbox_pixels(
        scene, camera, sphere, metallic_base["width"], metallic_base["height"], margin_px=16
    )
    mirror_bounds = object_bbox_pixels(
        scene, camera, mirror, metallic_base["width"], metallic_base["height"], margin_px=16
    )
    reflection_bounds = projected_crop_bounds(
        scene, camera, reflection_point, metallic_base["width"], metallic_base["height"]
    )
    ground_reflection_bounds = lower_reflection_band_bounds(reflection_bounds)

    set_principled(0.49, 0.0)
    metallic_mid_a = capture_case("metallic_mid_a")
    set_principled(0.51, 0.0)
    metallic_mid_b = capture_case("metallic_mid_b")
    set_principled(0.88, 0.0)
    metallic_target = capture_case("metallic_088")
    set_principled(0.99, 0.0)
    metallic_near_full = capture_case("metallic_099")
    set_principled(1.0, 0.0)
    metallic_full = capture_case("metallic_full")

    set_principled(0.0, 0.0)
    transmission_base = capture_case("transmission_base")
    set_principled(0.0, 0.05)
    transmission_mid = capture_case("transmission_mid")
    set_principled(0.0, 1.0)
    transmission_full = capture_case("transmission_full")

    metallic_mid_diff = crop_abs_diff_mean(
        metallic_mid_a["pixels"],
        metallic_mid_b["pixels"],
        metallic_mid_a["width"],
        metallic_mid_a["height"],
        reflection_bounds,
    )
    metallic_full_diff = crop_abs_diff_mean(
        metallic_base["pixels"],
        metallic_full["pixels"],
        metallic_base["width"],
        metallic_base["height"],
        reflection_bounds,
    )
    transmission_mid_diff = crop_abs_diff_mean(
        transmission_base["pixels"],
        transmission_mid["pixels"],
        transmission_base["width"],
        transmission_base["height"],
        reflection_bounds,
    )
    transmission_full_diff = crop_abs_diff_mean(
        transmission_base["pixels"],
        transmission_full["pixels"],
        transmission_base["width"],
        transmission_base["height"],
        reflection_bounds,
    )
    metallic_direct_diff = crop_abs_diff_mean(
        metallic_base["pixels"],
        metallic_full["pixels"],
        metallic_base["width"],
        metallic_base["height"],
        sphere_bounds,
    )
    metallic_target_diff = crop_abs_diff_mean(
        metallic_target["pixels"],
        metallic_full["pixels"],
        metallic_target["width"],
        metallic_target["height"],
        reflection_bounds,
    )
    metallic_near_full_diff = crop_abs_diff_mean(
        metallic_near_full["pixels"],
        metallic_full["pixels"],
        metallic_near_full["width"],
        metallic_near_full["height"],
        reflection_bounds,
    )
    transmission_direct_diff = crop_abs_diff_mean(
        transmission_base["pixels"],
        transmission_full["pixels"],
        transmission_base["width"],
        transmission_base["height"],
        sphere_bounds,
    )

    metallic_full_rgb = crop_rgb_mean(
        metallic_full["pixels"], metallic_full["width"], metallic_full["height"], reflection_bounds
    )
    transmission_full_rgb = crop_rgb_mean(
        transmission_full["pixels"],
        transmission_full["width"],
        transmission_full["height"],
        reflection_bounds,
    )
    metallic_target_ground_mean, metallic_target_ground_stddev = crop_luma_mean_stddev(
        metallic_target["pixels"],
        metallic_target["width"],
        metallic_target["height"],
        ground_reflection_bounds,
    )
    metallic_full_ground_mean, metallic_full_ground_stddev = crop_luma_mean_stddev(
        metallic_full["pixels"],
        metallic_full["width"],
        metallic_full["height"],
        ground_reflection_bounds,
    )

    print(f"CAMERA_OFFSET=({args.camera_dx:.4f}, {args.camera_dy:.4f}, {args.camera_dz:.4f})")
    print(f"SPHERE_BOUNDS={sphere_bounds}")
    print(f"MIRROR_BOUNDS={mirror_bounds}")
    print(f"REFLECTION_BOUNDS={reflection_bounds}")
    print(f"GROUND_REFLECTION_BOUNDS={ground_reflection_bounds}")
    print(f"TEST_BLEND_METALLIC_DIRECT_DIFF_MEAN={metallic_direct_diff:.6f}")
    print(f"TEST_BLEND_METALLIC_MID_DIFF_MEAN={metallic_mid_diff:.6f}")
    print(f"TEST_BLEND_METALLIC_FULL_DIFF_MEAN={metallic_full_diff:.6f}")
    print(f"TEST_BLEND_METALLIC_MID_RATIO={metallic_mid_diff / max(metallic_full_diff, 1.0e-6):.6f}")
    print(f"TEST_BLEND_METALLIC_088_TO_100_DIFF_MEAN={metallic_target_diff:.6f}")
    print(f"TEST_BLEND_METALLIC_099_TO_100_DIFF_MEAN={metallic_near_full_diff:.6f}")
    print(f"TEST_BLEND_METALLIC_088_GROUND_LUMA_MEAN={metallic_target_ground_mean:.6f}")
    print(
        f"TEST_BLEND_METALLIC_088_GROUND_LUMA_STDDEV={metallic_target_ground_stddev:.6f}"
    )
    print(f"TEST_BLEND_METALLIC_100_GROUND_LUMA_MEAN={metallic_full_ground_mean:.6f}")
    print(
        f"TEST_BLEND_METALLIC_100_GROUND_LUMA_STDDEV={metallic_full_ground_stddev:.6f}"
    )
    print(
        "TEST_BLEND_METALLIC_FULL_RGB="
        f"({metallic_full_rgb[0]:.6f}, {metallic_full_rgb[1]:.6f}, {metallic_full_rgb[2]:.6f})"
    )
    print(f"TEST_BLEND_TRANSMISSION_DIRECT_DIFF_MEAN={transmission_direct_diff:.6f}")
    print(f"TEST_BLEND_TRANSMISSION_MID_DIFF_MEAN={transmission_mid_diff:.6f}")
    print(f"TEST_BLEND_TRANSMISSION_FULL_DIFF_MEAN={transmission_full_diff:.6f}")
    print(
        "TEST_BLEND_TRANSMISSION_MID_RATIO="
        f"{transmission_mid_diff / max(transmission_full_diff, 1.0e-6):.6f}"
    )
    print(
        "TEST_BLEND_TRANSMISSION_FULL_RGB="
        f"({transmission_full_rgb[0]:.6f}, {transmission_full_rgb[1]:.6f}, {transmission_full_rgb[2]:.6f})"
    )
    print(f"OUTPUT_DIR={output_dir}")

    camera.location = original_camera_location
    principled.inputs["Metallic"].default_value = original_metallic
    principled.inputs["Transmission Weight"].default_value = original_transmission
    principled.inputs["Roughness"].default_value = original_roughness
    principled.inputs["IOR"].default_value = original_ior
    bpy.context.view_layer.update()

    if cleanup_dir is not None:
        cleanup_dir.cleanup()

    def _quit_blender():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(_quit_blender, first_interval=0.1)


run_probe()
