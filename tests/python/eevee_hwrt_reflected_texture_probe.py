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
        description="Probe mirrored image-texture response for Eevee Hardware RT."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=16)
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
    eevee.hardware_raytracing_reflection_mode = "FULL"
    eevee.hardware_raytracing_refraction_mode = "FULL"
    eevee.use_hardware_raytracing_environment = True
    eevee.use_hardware_raytracing_shadows = True
    eevee.taa_render_samples = max(1, samples)

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.use_denoise = False
    ray_tracing.denoise_spatial = False
    ray_tracing.denoise_temporal = False
    ray_tracing.denoise_bilateral = False
    ray_tracing.screen_trace_quality = 1.0
    ray_tracing.screen_trace_thickness = 1.0


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def align_object_normal(obj, normal: Vector):
    obj.rotation_euler = normal.normalized().to_track_quat("Z", "Y").to_euler()


def make_mirror_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    glossy = ntree.nodes.new("ShaderNodeBsdfGlossy")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    glossy.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs["Roughness"].default_value = 0.0
    ntree.links.new(glossy.outputs["BSDF"], output.inputs["Surface"])
    return material


def ensure_quadrant_image(name: str, quadrant_colors, size: int = 16):
    image = bpy.data.images.new(name=name, width=size, height=size, alpha=False, float_buffer=False)
    pixels = []
    half = size // 2
    for y in range(size):
        for x in range(size):
            is_top = y >= half
            is_right = x >= half
            quadrant_index = (0 if is_top else 2) + (1 if is_right else 0)
            color = quadrant_colors[quadrant_index]
            pixels.extend((color[0], color[1], color[2], 1.0))
    image.pixels = pixels
    image.update()
    return image


def make_image_emission_material(name: str):
    image_a = ensure_quadrant_image(
        f"{name}ImageA",
        ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
    )
    image_b = ensure_quadrant_image(
        f"{name}ImageB",
        ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.5, 0.0), (0.2, 0.2, 1.0)),
    )

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()

    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    image_node = ntree.nodes.new("ShaderNodeTexImage")
    emission = ntree.nodes.new("ShaderNodeEmission")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")

    image_node.interpolation = "Closest"
    emission.inputs["Strength"].default_value = 4.0

    ntree.links.new(texcoord.outputs["UV"], image_node.inputs["Vector"])
    ntree.links.new(image_node.outputs["Color"], emission.inputs["Color"])
    ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    image_node.image = image_a
    return material, image_node, image_a, image_b


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTReflectedTextureProbeWorld")
        scene.world = world
    world.use_nodes = True
    ntree = world.node_tree
    ntree.nodes.clear()
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")
    background.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    background.inputs["Strength"].default_value = 0.2
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])


def render_to_image(scene, output_dir: Path, tag: str):
    scene.render.filepath = str(output_dir / f"{tag}.png")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


def projected_crop_bounds(scene, camera, point: Vector, width: int, height: int, radius_px: int = 80):
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
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_reflected_texture_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    ensure_world(scene)

    mirror_material = make_mirror_material("HWRTTextureProbeMirror")
    texture_material, image_node, image_a, image_b = make_image_emission_material("HWRTTextureProbeImage")

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 1.0))
    cube = bpy.context.active_object
    cube.scale = Vector((0.7, 0.7, 0.7))
    cube.data.materials.clear()
    cube.data.materials.append(texture_material)

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(2.4, 0.3, 1.2))
    mirror = bpy.context.active_object
    mirror.scale = Vector((1.2, 1.2, 1.2))
    mirror.data.materials.clear()
    mirror.data.materials.append(mirror_material)

    bpy.ops.object.camera_add(location=(0.0, -6.0, 3.0))
    camera = bpy.context.active_object
    look_at(camera, Vector((0.5, 0.2, 1.0)))
    scene.camera = camera

    mirror_view_dir = (camera.location - mirror.location).normalized()
    mirror_target_dir = (cube.location - mirror.location).normalized()
    align_object_normal(mirror, mirror_view_dir + mirror_target_dir)

    image_node.image = image_a
    width, height, pixels_a = render_to_image(scene, output_dir, "texture_a")
    image_node.image = image_b
    width, height, pixels_b = render_to_image(scene, output_dir, "texture_b")

    direct_bounds = projected_crop_bounds(scene, camera, cube.location, width, height, 80)
    reflected_point = mirror_reflection_point(cube.location, mirror)
    reflection_bounds = projected_crop_bounds(scene, camera, reflected_point, width, height, 80)

    failures = []
    direct_diff = crop_abs_diff_mean(pixels_a, pixels_b, width, height, direct_bounds)
    reflection_diff = crop_abs_diff_mean(pixels_a, pixels_b, width, height, reflection_bounds)
    print(f"DIRECT_BOUNDS={direct_bounds}")
    print(f"REFLECTION_BOUNDS={reflection_bounds}")
    assert_metric("DIRECT_TEXTURE_DIFF_MEAN", direct_diff, 0.01, failures)
    assert_metric("REFLECTED_TEXTURE_DIFF_MEAN", reflection_diff, 0.01, failures)
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT reflected texture probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT reflected texture probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
