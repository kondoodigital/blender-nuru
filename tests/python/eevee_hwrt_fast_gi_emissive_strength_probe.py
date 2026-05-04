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
        description="Probe Hardware Fast GI response to emissive strength changes."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--low-strength", type=float, default=1.0)
    parser.add_argument("--high-strength", type=float, default=20.0)
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
        world = bpy.data.worlds.new("HWRTFastGIEmissiveStrengthWorld")
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
    emission.inputs["Color"].default_value = (1.0, 0.85, 0.55, 1.0)
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


def create_scene_layout(scene):
    floor = create_plane("ProbeFloor", (0.0, 0.0, 0.0), (3.0, 3.0, 3.0), make_diffuse_material("ProbeFloorMat", (0.9, 0.9, 0.9, 1.0)))
    back = create_plane("ProbeBackWall", (0.0, 2.5, 1.8), (3.0, 1.8, 1.0), make_diffuse_material("ProbeBackWallMat", (0.9, 0.9, 0.9, 1.0)))
    back.rotation_euler.x = 1.57079632679
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 1.6, 1.8))
    emitter = bpy.context.active_object
    emitter.name = "ProbeEmitter"
    emitter.scale = Vector((0.45, 0.45, 0.45))
    emitter.rotation_euler.x = 1.57079632679
    emitter.data.materials.clear()
    emitter.data.materials.append(make_emission_material("ProbeEmitterMat", 1.0))

    bpy.ops.object.camera_add(location=(0.0, -5.5, 2.6))
    camera = bpy.context.active_object
    target_patch = Vector((0.0, 0.6, 0.0))
    look_at(camera, target_patch)
    scene.camera = camera

    return floor, back, emitter, camera, target_patch


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


def projected_crop_bounds(scene, camera, point: Vector, width: int, height: int, radius_px: int):
    co = world_to_camera_view(scene, camera, point)
    center_x = int(round(co.x * width))
    center_y = int(round(co.y * height))
    min_x = max(0, center_x - radius_px)
    max_x = min(width - 1, center_x + radius_px)
    min_y = max(0, center_y - radius_px)
    max_y = min(height - 1, center_y + radius_px)
    return (min_x, min_y, max_x, max_y)


def assert_metric(name: str, value: float, minimum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def set_emitter_strength(emitter, strength: float):
    material = emitter.data.materials[0]
    emission = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeEmission")
    emission.inputs["Strength"].default_value = strength


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_fast_gi_emissive_strength_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    ensure_world(scene)
    _floor, _back, emitter, camera, target_patch = create_scene_layout(scene)

    set_emitter_strength(emitter, args.low_strength)
    width, height, low_pixels = render_to_image(scene, output_dir, "fast_gi_low")

    set_emitter_strength(emitter, args.high_strength)
    _, _, high_pixels = render_to_image(scene, output_dir, "fast_gi_high")

    patch_bounds = projected_crop_bounds(scene, camera, target_patch, width, height, radius_px=48)
    failures = []
    diff_mean = crop_abs_diff_mean(low_pixels, high_pixels, width, height, patch_bounds)
    print(f"PATCH_BOUNDS={patch_bounds}")
    assert_metric("FAST_GI_EMISSIVE_STRENGTH_DIFF_MEAN", diff_mean, 0.010, failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("EEVEE HWRT fast GI emissive strength probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


if __name__ == "__main__":
    main()
