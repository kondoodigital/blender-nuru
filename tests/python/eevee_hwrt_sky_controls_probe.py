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
        description="Probe Sky Disk visibility for Eevee Hardware RT."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=24)
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
from mathutils import Vector


def configure_scene(scene, samples: int):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "OFF"
    eevee.use_hardware_raytracing_environment = True
    eevee.use_hardware_raytracing_shadows = False
    eevee.taa_render_samples = max(1, samples)

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.use_denoise = False
    ray_tracing.denoise_spatial = False
    ray_tracing.denoise_temporal = False
    ray_tracing.denoise_bilateral = False


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_sky_world(scene):
    world = bpy.data.worlds.new("HWRTSkyControlsProbeWorld")
    world.use_nodes = True
    world.sun_angle = math.radians(8.0)
    scene.world = world

    ntree = world.node_tree
    ntree.nodes.clear()
    sky = ntree.nodes.new("ShaderNodeTexSky")
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")

    sky.sky_type = "SINGLE_SCATTERING"
    sky.sun_disc = True
    sky.sun_size = math.radians(1.2)
    sky.sun_intensity = 5.0
    sky.sun_elevation = 0.0
    sky.sun_rotation = 0.0
    background.inputs["Strength"].default_value = 1.0

    ntree.links.new(sky.outputs["Color"], background.inputs["Color"])
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return sky


def render_to_image(scene, output_dir: Path, tag: str):
    scene.render.filepath = str(output_dir / f"{tag}.png")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


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


def assert_metric(name: str, value: float, minimum: float, failures):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_sky_controls_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    sky = ensure_sky_world(scene)

    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    sky_camera = bpy.context.active_object
    look_at(sky_camera, Vector((0.0, -1.0, 0.0)))
    scene.camera = sky_camera

    sky.sun_disc = False
    width, height, disk_off = render_to_image(scene, output_dir, "sun_disk_off")
    sky.sun_disc = True
    width, height, disk_on = render_to_image(scene, output_dir, "sun_disk_on")
    center_bounds = (
        width // 2 - 80,
        height // 2 - 80,
        width // 2 + 80,
        height // 2 + 80,
    )

    failures = []
    sun_disk_diff = crop_abs_diff_mean(disk_off, disk_on, width, height, center_bounds)
    print(f"SUN_DISK_BOUNDS={center_bounds}")
    assert_metric("SUN_DISK_DIFF_MEAN", sun_disk_diff, 0.01, failures)
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT sky controls probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT sky controls probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
