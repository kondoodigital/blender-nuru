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
        description="Probe Hardware Fast GI near-field leak rejection across a thin wall."
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
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def configure_scene(scene, samples: int):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.use_hardware_raytracing_gi = True
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "OFF"
    eevee.use_hardware_raytracing_environment = False
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


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTFastGINearfieldLeakWorld")
        scene.world = world
    world.use_nodes = True
    ntree = world.node_tree
    background = next((node for node in ntree.nodes if node.bl_idname == "ShaderNodeBackground"), None)
    output = next((node for node in ntree.nodes if node.bl_idname == "ShaderNodeOutputWorld"), None)
    if background is None:
        background = ntree.nodes.new("ShaderNodeBackground")
    if output is None:
        output = ntree.nodes.new("ShaderNodeOutputWorld")
    if not background.outputs["Background"].is_linked:
        ntree.links.new(background.outputs["Background"], output.inputs["Surface"])
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0


def make_diffuse_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    diffuse = ntree.nodes.new("ShaderNodeBsdfDiffuse")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    diffuse.inputs["Color"].default_value = color
    ntree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_emission_material(name: str, strength: float):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    emission = ntree.nodes.new("ShaderNodeEmission")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Color"].default_value = (1.0, 0.9, 0.7, 1.0)
    emission.inputs["Strength"].default_value = strength
    ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def create_plane(name: str, location, scale, material):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = Vector(scale)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def create_cube(name: str, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = Vector(scale)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def clear_scene(scene):
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def create_scene_layout(scene):
    clear_scene(scene)
    ensure_world(scene)

    diffuse = make_diffuse_material("ProbeDiffuseMat", (0.9, 0.9, 0.9, 1.0))
    emissive = make_emission_material("ProbeEmitterMat", 24.0)

    floor = create_plane("ProbeFloor", (0.0, 0.0, 0.0), (2.5, 2.5, 2.5), diffuse)
    _back = create_plane("ProbeBack", (0.0, 1.6, 1.2), (2.5, 1.2, 1.0), diffuse)
    _back.rotation_euler.x = math.pi * 0.5
    wall = create_cube("ProbeThinWall", (0.0, 0.0, 1.0), (1.0, 0.03, 1.0), diffuse)
    layered_wall = create_cube("ProbeLayeredWall", (0.0, -0.18, 1.0), (0.92, 0.025, 0.95), diffuse)
    emitter = create_plane("ProbeEmitter", (0.0, 0.45, 1.0), (0.45, 0.45, 0.45), emissive)
    emitter.rotation_euler.x = math.pi * 0.5

    patch = Vector((0.0, -0.8, 0.0))
    bpy.ops.object.camera_add(location=(0.0, -3.2, 1.4))
    camera = bpy.context.active_object
    look_at(camera, patch)
    scene.camera = camera

    return floor, wall, layered_wall, emitter, camera, patch


def render_to_image(scene, output_dir: Path, tag: str):
    scene.render.filepath = str(output_dir / f"{tag}.png")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
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


def crop_mean_luma(pixels, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    total = 0.0
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            r = pixels[base]
            g = pixels[base + 1]
            b = pixels[base + 2]
            total += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
    return total / max(1, count)


def assert_metric(name: str, value: float, minimum: float | None, maximum: float | None, failures: list[str]):
    threshold_text = []
    if minimum is not None:
        threshold_text.append(f">= {minimum:.6f}")
    if maximum is not None:
        threshold_text.append(f"<= {maximum:.6f}")
    print(f"{name}={value:.6f} threshold={' and '.join(threshold_text)}")
    if minimum is not None and value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")
    if maximum is not None and value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_fast_gi_nearfield_leak_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    _floor, wall, layered_wall, _emitter, camera, patch = create_scene_layout(scene)

    layered_wall.hide_render = True
    layered_wall.hide_viewport = True
    bpy.context.view_layer.update()

    width, height, blocked_pixels = render_to_image(scene, output_dir, "nearfield_leak_wall_present")

    layered_wall.hide_render = False
    layered_wall.hide_viewport = False
    bpy.context.view_layer.update()
    _, _, layered_pixels = render_to_image(scene, output_dir, "nearfield_leak_layered_wall_present")

    wall.hide_render = True
    wall.hide_viewport = True
    layered_wall.hide_render = True
    layered_wall.hide_viewport = True
    bpy.context.view_layer.update()
    _, _, open_pixels = render_to_image(scene, output_dir, "nearfield_leak_wall_removed")

    patch_bounds = projected_crop_bounds(scene, camera, patch, width, height, radius_px=52)
    blocked_mean = crop_mean_luma(blocked_pixels, width, height, patch_bounds)
    layered_mean = crop_mean_luma(layered_pixels, width, height, patch_bounds)
    open_mean = crop_mean_luma(open_pixels, width, height, patch_bounds)
    diff_mean = open_mean - blocked_mean
    layered_diff_mean = open_mean - layered_mean

    failures = []
    print(f"PATCH_BOUNDS={patch_bounds}")
    assert_metric("NEARFIELD_LEAK_BLOCKED_MEAN", blocked_mean, None, 0.18, failures)
    assert_metric("NEARFIELD_LAYERED_BLOCKED_MEAN", layered_mean, None, 0.14, failures)
    assert_metric("NEARFIELD_LEAK_OPEN_MEAN", open_mean, 0.08, None, failures)
    assert_metric("NEARFIELD_LEAK_DIFF_MEAN", diff_mean, 0.03, None, failures)
    assert_metric("NEARFIELD_LAYERED_DIFF_MEAN", layered_diff_mean, 0.03, None, failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("EEVEE HWRT Fast GI nearfield leak probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


if __name__ == "__main__":
    main()
