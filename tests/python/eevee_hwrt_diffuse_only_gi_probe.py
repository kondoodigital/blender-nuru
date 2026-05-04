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
        description="Probe that Hardware RT GI stays diffuse-only and does not brighten specular hits."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=32)
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
    eevee.use_hardware_raytracing_gi = False
    eevee.hardware_raytracing_reflection_mode = "FULL"
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
    ray_tracing.screen_trace_quality = 1.0
    ray_tracing.screen_trace_thickness = 1.0


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def align_object_normal(obj, normal: Vector):
    obj.rotation_euler = normal.normalized().to_track_quat("Z", "Y").to_euler()


def ensure_world(scene, strength: float):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTDiffuseOnlyGIProbeWorld")
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
    background.inputs["Color"].default_value = (0.85, 0.92, 1.0, 1.0)
    background.inputs["Strength"].default_value = strength


def make_diffuse_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    diffuse = ntree.nodes.new("ShaderNodeBsdfDiffuse")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    diffuse.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    ntree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


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


def create_plane(name: str, location, scale, material):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = Vector(scale)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def create_scene_layout(scene):
    diffuse_mat = make_diffuse_material("ProbeDiffuseMat")
    mirror_mat = make_mirror_material("ProbeMirrorMat")

    floor = create_plane("ProbeFloor", (0.0, 0.0, 0.0), (4.0, 4.0, 4.0), diffuse_mat)
    mirror_target = Vector((-0.8, -10.0, 1.0))
    hidden_patch = create_plane("ProbeHiddenPatch", mirror_target, (1.4, 1.4, 1.4), diffuse_mat)
    mirror = create_plane("ProbeMirror", (2.4, 0.6, 1.2), (1.2, 1.2, 1.2), mirror_mat)

    bpy.ops.object.camera_add(location=(0.0, -6.0, 3.0))
    camera = bpy.context.active_object
    look_at(camera, Vector((0.4, 0.2, 0.4)))
    scene.camera = camera

    mirror_view_dir = (camera.location - mirror.location).normalized()
    mirror_target_dir = (mirror_target - mirror.location).normalized()
    align_object_normal(mirror, mirror_view_dir + mirror_target_dir)
    align_object_normal(hidden_patch, mirror.location - hidden_patch.location)

    visible_patch = Vector((0.9, 0.35, 0.0))
    return floor, mirror, hidden_patch, camera, visible_patch


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


def projected_point(scene, camera, point: Vector):
    co = world_to_camera_view(scene, camera, point)
    return (co.x, co.y, co.z)


def mirror_reflection_point(camera_pos: Vector, target_point: Vector, mirror_obj):
    mirror_origin = mirror_obj.matrix_world.translation
    mirror_normal = (mirror_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    mirrored_target = target_point - 2.0 * (target_point - mirror_origin).dot(mirror_normal) * mirror_normal
    ray = mirrored_target - camera_pos
    denom = ray.dot(mirror_normal)
    if abs(denom) < 1.0e-8:
        return mirror_origin
    distance = (mirror_origin - camera_pos).dot(mirror_normal) / denom
    return camera_pos + ray * distance


def assert_metric_min(name: str, value: float, minimum: float, failures: list[str]):
    print(f"{name}={value:.6f} min={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def assert_metric_max(name: str, value: float, maximum: float, failures: list[str]):
    print(f"{name}={value:.6f} max={maximum:.6f}")
    if value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_diffuse_only_gi_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    ensure_world(scene, 0.0)
    _floor, mirror, hidden_patch, camera, visible_patch = create_scene_layout(scene)

    width, height, gi_off = render_to_image(scene, output_dir, "gi_off")

    ensure_world(scene, 3.0)
    scene.eevee.use_hardware_raytracing_gi = True
    width, height, gi_on = render_to_image(scene, output_dir, "gi_on")

    direct_bounds = projected_crop_bounds(scene, camera, visible_patch, width, height, 70)
    reflected_patch = mirror_reflection_point(camera.location, hidden_patch.location, mirror)
    reflection_bounds = projected_crop_bounds(scene, camera, reflected_patch, width, height, 70)

    failures = []
    direct_diff = crop_abs_diff_mean(gi_off, gi_on, width, height, direct_bounds)
    reflection_diff = crop_abs_diff_mean(gi_off, gi_on, width, height, reflection_bounds)
    print(f"DIRECT_BOUNDS={direct_bounds}")
    print(f"HIDDEN_PATCH_VIEW={projected_point(scene, camera, hidden_patch.location)}")
    print(f"REFLECTED_PATCH_VIEW={projected_point(scene, camera, reflected_patch)}")
    print(f"REFLECTION_BOUNDS={reflection_bounds}")
    assert_metric_min("DIFFUSE_GI_DIRECT_DIFF_MEAN", direct_diff, 0.01, failures)
    assert_metric_max("SPECULAR_GI_REFLECTION_DIFF_MEAN", reflection_diff, 0.01, failures)

    if failures:
        print("EEVEE HWRT diffuse-only GI probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT diffuse-only GI probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
