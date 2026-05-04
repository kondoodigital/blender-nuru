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
        description="Probe Principled clearcoat response in Eevee Hardware RT reflections."
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
    eevee.hardware_raytracing_reflection_mode = "FULL"
    eevee.hardware_raytracing_refraction_mode = "FULL"
    eevee.use_hardware_raytracing_environment = True
    eevee.use_hardware_raytracing_shadows = True
    eevee.taa_render_samples = max(1, samples)
    eevee.ray_tracing_reflection_bounces = 4
    eevee.ray_tracing_refraction_bounces = 4

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


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTPrincipledCoatProbeWorld")
        scene.world = world
    world.use_nodes = True
    ntree = world.node_tree
    ntree.nodes.clear()

    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    mapping = ntree.nodes.new("ShaderNodeMapping")
    gradient = ntree.nodes.new("ShaderNodeTexGradient")
    ramp = ntree.nodes.new("ShaderNodeValToRGB")
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")

    mapping.inputs["Rotation"].default_value[2] = math.radians(90.0)
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = (0.02, 0.10, 0.45, 1.0)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (1.0, 0.35, 0.05, 1.0)
    background.inputs["Strength"].default_value = 1.3

    ntree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    ntree.links.new(mapping.outputs["Vector"], gradient.inputs["Vector"])
    ntree.links.new(gradient.outputs["Fac"], ramp.inputs["Fac"])
    ntree.links.new(ramp.outputs["Color"], background.inputs["Color"])
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])


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


def make_principled_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    principled = ntree.nodes.new("ShaderNodeBsdfPrincipled")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    principled.inputs["Base Color"].default_value = (0.75, 0.72, 0.68, 1.0)
    principled.inputs["Roughness"].default_value = 0.12
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Coat Weight"].default_value = 0.0
    principled.inputs["Coat Roughness"].default_value = 0.03
    principled.inputs["Coat Tint"].default_value = (1.0, 0.15, 0.15, 1.0)
    ntree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material, principled


def make_floor_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    checker = ntree.nodes.new("ShaderNodeTexChecker")
    mapping = ntree.nodes.new("ShaderNodeMapping")
    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    principled = ntree.nodes.new("ShaderNodeBsdfPrincipled")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    checker.inputs["Color1"].default_value = (0.75, 0.75, 0.75, 1.0)
    checker.inputs["Color2"].default_value = (0.15, 0.15, 0.15, 1.0)
    mapping.inputs["Scale"].default_value = (5.0, 5.0, 5.0)
    principled.inputs["Roughness"].default_value = 0.9
    ntree.links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    ntree.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    ntree.links.new(checker.outputs["Color"], principled.inputs["Base Color"])
    ntree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_backlight_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    emission = ntree.nodes.new("ShaderNodeEmission")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 8.0
    ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


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
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_principled_coat_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    ensure_world(scene)

    mirror_material = make_mirror_material("HWRTPrincipledCoatMirror")
    floor_material = make_floor_material("HWRTPrincipledCoatFloor")
    sphere_material, sphere_node = make_principled_material("HWRTPrincipledCoatSphere")
    backlight_material = make_backlight_material("HWRTPrincipledCoatBacklight")

    bpy.ops.mesh.primitive_uv_sphere_add(location=(0.0, 0.0, 1.05), segments=64, ring_count=32)
    sphere = bpy.context.active_object
    sphere.scale = Vector((0.9, 0.9, 0.9))
    sphere.data.materials.clear()
    sphere.data.materials.append(sphere_material)

    bpy.ops.mesh.primitive_plane_add(size=7.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.data.materials.clear()
    floor.data.materials.append(floor_material)

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(2.5, 0.25, 1.2))
    mirror = bpy.context.active_object
    mirror.scale = Vector((1.2, 1.2, 1.2))
    mirror.data.materials.clear()
    mirror.data.materials.append(mirror_material)

    bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0.0, 2.8, 1.2))
    backlight = bpy.context.active_object
    backlight.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    backlight.data.materials.clear()
    backlight.data.materials.append(backlight_material)

    bpy.ops.object.camera_add(location=(0.0, -6.0, 2.8))
    camera = bpy.context.active_object
    look_at(camera, Vector((0.5, 0.15, 1.0)))
    scene.camera = camera

    mirror_view_dir = (camera.location - mirror.location).normalized()
    mirror_target_dir = (sphere.location - mirror.location).normalized()
    align_object_normal(mirror, mirror_view_dir + mirror_target_dir)

    sphere_reflection_point = mirror_reflection_point(sphere.location, mirror)
    failures = []

    def render_case(tag: str):
        return render_to_image(scene, output_dir, tag)

    sphere_node.inputs["Metallic"].default_value = 0.0
    sphere_node.inputs["Base Color"].default_value = (0.8, 0.72, 0.65, 1.0)
    sphere_node.inputs["Roughness"].default_value = 0.12
    sphere_node.inputs["Coat Weight"].default_value = 0.0
    sphere_node.inputs["Coat Tint"].default_value = (1.0, 0.15, 0.15, 1.0)
    sphere_node.inputs["Coat Roughness"].default_value = 0.03
    width, height, diffuse_coat_a = render_case("coat_diffuse_a")
    sphere_direct_bounds = projected_crop_bounds(scene, camera, sphere.location, width, height, 90)
    sphere_reflection_bounds = projected_crop_bounds(scene, camera, sphere_reflection_point, width, height, 90)

    sphere_node.inputs["Coat Weight"].default_value = 1.0
    width, height, diffuse_coat_b = render_case("coat_diffuse_b")
    diffuse_direct_diff = crop_abs_diff_mean(diffuse_coat_a, diffuse_coat_b, width, height, sphere_direct_bounds)
    diffuse_reflection_diff = crop_abs_diff_mean(
        diffuse_coat_a, diffuse_coat_b, width, height, sphere_reflection_bounds
    )
    assert_metric("COAT_DIFFUSE_DIRECT_DIFF_MEAN", diffuse_direct_diff, 0.01, failures)
    assert_metric("COAT_DIFFUSE_REFLECTION_DIFF_MEAN", diffuse_reflection_diff, 0.01, failures)

    sphere_node.inputs["Metallic"].default_value = 1.0
    sphere_node.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    sphere_node.inputs["Roughness"].default_value = 0.08
    sphere_node.inputs["Coat Weight"].default_value = 1.0
    sphere_node.inputs["Coat Tint"].default_value = (1.0, 0.1, 0.1, 1.0)
    sphere_node.inputs["Coat Roughness"].default_value = 0.02
    width, height, metal_coat_a = render_case("coat_metal_a")

    sphere_node.inputs["Coat Tint"].default_value = (0.1, 1.0, 0.1, 1.0)
    sphere_node.inputs["Coat Roughness"].default_value = 0.45
    width, height, metal_coat_b = render_case("coat_metal_b")
    metal_direct_diff = crop_abs_diff_mean(metal_coat_a, metal_coat_b, width, height, sphere_direct_bounds)
    metal_reflection_diff = crop_abs_diff_mean(
        metal_coat_a, metal_coat_b, width, height, sphere_reflection_bounds
    )
    assert_metric("COAT_METAL_DIRECT_DIFF_MEAN", metal_direct_diff, 0.01, failures)
    assert_metric("COAT_METAL_REFLECTION_DIFF_MEAN", metal_reflection_diff, 0.01, failures)

    print(f"SPHERE_DIRECT_BOUNDS={sphere_direct_bounds}")
    print(f"SPHERE_REFLECTION_BOUNDS={sphere_reflection_bounds}")
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT Principled coat probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT Principled coat probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
