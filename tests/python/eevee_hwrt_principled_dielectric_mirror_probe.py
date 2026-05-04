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
        description="Probe Principled dielectric mirror response in Eevee Hardware RT."
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
        world = bpy.data.worlds.new("HWRTPrincipledDielectricMirrorWorld")
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
    ramp.color_ramp.elements[0].position = 0.2
    ramp.color_ramp.elements[0].color = (0.02, 0.10, 0.45, 1.0)
    ramp.color_ramp.elements[1].position = 0.8
    ramp.color_ramp.elements[1].color = (1.0, 0.35, 0.05, 1.0)
    background.inputs["Strength"].default_value = 1.3

    ntree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    ntree.links.new(mapping.outputs["Vector"], gradient.inputs["Vector"])
    ntree.links.new(gradient.outputs["Fac"], ramp.inputs["Fac"])
    ntree.links.new(ramp.outputs["Color"], background.inputs["Color"])
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])


def ensure_quadrant_image(name: str, quadrant_colors, size: int = 16):
    image = bpy.data.images.get(name)
    if image is None:
        image = bpy.data.images.new(name=name, width=size, height=size, alpha=False, float_buffer=False)
    elif image.size[0] != size or image.size[1] != size:
        image.scale(size, size)

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


def make_dielectric_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    principled = ntree.nodes.new("ShaderNodeBsdfPrincipled")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    principled.inputs["Base Color"].default_value = (0.8, 0.72, 0.65, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.05
    principled.inputs["Specular IOR Level"].default_value = 0.5
    principled.inputs["Specular Tint"].default_value = (1.0, 1.0, 1.0, 1.0)
    ntree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material, principled


def make_floor_material(name: str):
    image_a = ensure_quadrant_image(
        f"{name}ImageA",
        ((1.0, 0.1, 0.1), (1.0, 1.0, 0.1), (0.1, 0.3, 1.0), (0.1, 0.9, 0.5)),
    )
    image_b = ensure_quadrant_image(
        f"{name}ImageB",
        ((0.0, 1.0, 0.2), (0.1, 0.8, 1.0), (1.0, 0.2, 0.8), (1.0, 0.4, 0.1)),
    )

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()

    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    image_node = ntree.nodes.new("ShaderNodeTexImage")
    principled = ntree.nodes.new("ShaderNodeBsdfPrincipled")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")

    image_node.interpolation = "Closest"
    principled.inputs["Roughness"].default_value = 0.85
    principled.inputs["Specular IOR Level"].default_value = 0.45

    ntree.links.new(texcoord.outputs["UV"], image_node.inputs["Vector"])
    ntree.links.new(image_node.outputs["Color"], principled.inputs["Base Color"])
    ntree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    image_node.image = image_a
    return material, image_node, image_a, image_b


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


def refresh_material_binding(material, obj):
    material.node_tree.update_tag()
    obj.update_tag()
    bpy.context.view_layer.update()


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


def crop_mean_rgb(pixels, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    accum = [0.0, 0.0, 0.0]
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            accum[0] += pixels[base + 0]
            accum[1] += pixels[base + 1]
            accum[2] += pixels[base + 2]
            count += 1
    return tuple(channel / max(1, count) for channel in accum)


def assert_metric(name: str, value: float, minimum: float, failures):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def assert_metric_max(name: str, value: float, maximum: float, failures):
    print(f"{name}={value:.6f} threshold={maximum:.6f}")
    if value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def assert_channel_order(name: str, rgb, dominant_idx: int, failures, minimum_gap: float = 0.005):
    dominant = rgb[dominant_idx]
    others = [rgb[index] for index in range(3) if index != dominant_idx]
    gap = dominant - max(others)
    print(f"{name}={rgb} dominant_gap={gap:.6f}")
    if gap < minimum_gap:
        failures.append(f"{name} expected dominant channel gap >= {minimum_gap:.6f}, got {gap:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_principled_dielectric_mirror_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    ensure_world(scene)

    mirror_material = make_mirror_material("HWRTPrincipledDielectricMirror")
    dielectric_material, dielectric_node = make_dielectric_material("HWRTPrincipledDielectric")
    floor_material, floor_image_node, floor_image_a, floor_image_b = make_floor_material("HWRTPrincipledFloor")
    backlight_material = make_backlight_material("HWRTPrincipledDielectricBacklight")

    bpy.ops.mesh.primitive_uv_sphere_add(location=(0.0, 0.0, 1.05), segments=64, ring_count=32)
    sphere = bpy.context.active_object
    sphere.scale = Vector((0.9, 0.9, 0.9))
    sphere.data.materials.clear()
    sphere.data.materials.append(dielectric_material)

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
    floor_sample_point = Vector((0.8, 0.65, 0.0))
    floor_reflection_point = mirror_reflection_point(floor_sample_point, mirror)

    failures = []

    def render_case(tag: str):
        return render_to_image(scene, output_dir, tag)

    dielectric_node.inputs["Roughness"].default_value = 0.05
    dielectric_node.inputs["Specular IOR Level"].default_value = 0.5
    dielectric_node.inputs["Specular Tint"].default_value = (1.0, 1.0, 1.0, 1.0)
    floor_image_node.image = floor_image_a
    refresh_material_binding(floor_material, floor)
    width, height, roughness_a = render_case("dielectric_roughness_a")
    sphere_direct_bounds = projected_crop_bounds(scene, camera, sphere.location, width, height, 90)
    sphere_reflection_bounds = projected_crop_bounds(scene, camera, sphere_reflection_point, width, height, 90)
    floor_direct_bounds = projected_crop_bounds(scene, camera, floor_sample_point, width, height, 90)
    floor_reflection_bounds = projected_crop_bounds(scene, camera, floor_reflection_point, width, height, 90)

    dielectric_node.inputs["Roughness"].default_value = 0.6
    width, height, roughness_b = render_case("dielectric_roughness_b")
    roughness_direct_diff = crop_abs_diff_mean(roughness_a, roughness_b, width, height, sphere_direct_bounds)
    roughness_reflection_diff = crop_abs_diff_mean(
        roughness_a, roughness_b, width, height, sphere_reflection_bounds
    )
    assert_metric("DIELECTRIC_ROUGHNESS_DIRECT_DIFF_MEAN", roughness_direct_diff, 0.01, failures)
    assert_metric("DIELECTRIC_ROUGHNESS_REFLECTION_DIFF_MEAN", roughness_reflection_diff, 0.01, failures)

    dielectric_node.inputs["Roughness"].default_value = 0.05
    dielectric_node.inputs["Specular IOR Level"].default_value = 0.0
    width, height, spec_level_a = render_case("dielectric_spec_level_a")
    dielectric_node.inputs["Specular IOR Level"].default_value = 1.0
    width, height, spec_level_b = render_case("dielectric_spec_level_b")
    spec_level_reflection_diff = crop_abs_diff_mean(
        spec_level_a, spec_level_b, width, height, sphere_reflection_bounds
    )
    assert_metric("DIELECTRIC_SPECULAR_LEVEL_REFLECTION_DIFF_MEAN", spec_level_reflection_diff, 0.004, failures)

    dielectric_node.inputs["Specular IOR Level"].default_value = 1.0
    dielectric_node.inputs["Specular Tint"].default_value = (1.0, 0.1, 0.1, 1.0)
    width, height, tint_a = render_case("dielectric_tint_a")
    tint_reflection_rgb_a = crop_mean_rgb(tint_a, width, height, sphere_reflection_bounds)

    dielectric_node.inputs["Specular Tint"].default_value = (0.1, 1.0, 0.1, 1.0)
    width, height, tint_b = render_case("dielectric_tint_b")
    tint_reflection_rgb_b = crop_mean_rgb(tint_b, width, height, sphere_reflection_bounds)
    tint_reflection_diff = crop_abs_diff_mean(tint_a, tint_b, width, height, sphere_reflection_bounds)
    tint_red_drop = tint_reflection_rgb_a[0] - tint_reflection_rgb_b[0]
    tint_green_lift = tint_reflection_rgb_b[1] - tint_reflection_rgb_a[1]
    assert_metric("DIELECTRIC_SPECULAR_TINT_REFLECTION_DIFF_MEAN", tint_reflection_diff, 0.004, failures)
    assert_metric("DIELECTRIC_SPECULAR_TINT_RED_DROP", tint_red_drop, 0.02, failures)
    assert_metric("DIELECTRIC_SPECULAR_TINT_GREEN_LIFT", tint_green_lift, 0.02, failures)

    dielectric_node.inputs["Specular Tint"].default_value = (1.0, 1.0, 1.0, 1.0)
    floor_image_node.image = floor_image_a
    refresh_material_binding(floor_material, floor)
    width, height, floor_a = render_case("dielectric_floor_a")
    floor_image_node.image = floor_image_b
    refresh_material_binding(floor_material, floor)
    width, height, floor_b = render_case("dielectric_floor_b")
    floor_direct_diff = crop_abs_diff_mean(floor_a, floor_b, width, height, floor_direct_bounds)
    floor_reflection_diff = crop_abs_diff_mean(floor_a, floor_b, width, height, floor_reflection_bounds)
    assert_metric("DIELECTRIC_FLOOR_DIRECT_DIFF_MEAN", floor_direct_diff, 0.02, failures)
    assert_metric("DIELECTRIC_FLOOR_REFLECTION_DIFF_MEAN", floor_reflection_diff, 0.01, failures)

    dielectric_node.inputs["Transmission Weight"].default_value = 0.0
    dielectric_node.inputs["Metallic"].default_value = 0.0
    dielectric_node.inputs["Roughness"].default_value = 0.05
    width, height, metallic_base = render_case("principled_metallic_base")
    dielectric_node.inputs["Metallic"].default_value = 0.45
    width, height, metallic_mid_a = render_case("principled_metallic_mid_a")
    dielectric_node.inputs["Metallic"].default_value = 0.55
    width, height, metallic_mid_b = render_case("principled_metallic_mid_b")
    dielectric_node.inputs["Metallic"].default_value = 1.0
    width, height, metallic_full = render_case("principled_metallic_full")
    metallic_mid_reflection_diff = crop_abs_diff_mean(
        metallic_mid_a, metallic_mid_b, width, height, sphere_reflection_bounds
    )
    metallic_full_reflection_diff = crop_abs_diff_mean(
        metallic_base, metallic_full, width, height, sphere_reflection_bounds
    )
    metallic_mid_ratio = metallic_mid_reflection_diff / max(metallic_full_reflection_diff, 1.0e-6)
    assert_metric("PRINCIPLED_METALLIC_FULL_REFLECTION_DIFF_MEAN", metallic_full_reflection_diff, 0.01, failures)
    assert_metric_max("PRINCIPLED_METALLIC_MID_REFLECTION_RATIO", metallic_mid_ratio, 0.7, failures)

    dielectric_node.inputs["Metallic"].default_value = 0.0
    dielectric_node.inputs["Roughness"].default_value = 0.0
    dielectric_node.inputs["Transmission Weight"].default_value = 0.0
    width, height, transmission_base = render_case("principled_transmission_base")
    dielectric_node.inputs["Transmission Weight"].default_value = 0.05
    width, height, transmission_mid = render_case("principled_transmission_mid")
    dielectric_node.inputs["Transmission Weight"].default_value = 1.0
    width, height, transmission_full = render_case("principled_transmission_full")
    transmission_mid_reflection_diff = crop_abs_diff_mean(
        transmission_base, transmission_mid, width, height, sphere_reflection_bounds
    )
    transmission_full_reflection_diff = crop_abs_diff_mean(
        transmission_base, transmission_full, width, height, sphere_reflection_bounds
    )
    transmission_mid_ratio = transmission_mid_reflection_diff / max(transmission_full_reflection_diff, 1.0e-6)
    assert_metric(
        "PRINCIPLED_TRANSMISSION_FULL_REFLECTION_DIFF_MEAN",
        transmission_full_reflection_diff,
        0.01,
        failures,
    )
    assert_metric_max("PRINCIPLED_TRANSMISSION_MID_REFLECTION_RATIO", transmission_mid_ratio, 0.7, failures)

    print(f"SPHERE_DIRECT_BOUNDS={sphere_direct_bounds}")
    print(f"SPHERE_REFLECTION_BOUNDS={sphere_reflection_bounds}")
    print(f"FLOOR_DIRECT_BOUNDS={floor_direct_bounds}")
    print(f"FLOOR_REFLECTION_BOUNDS={floor_reflection_bounds}")
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT Principled dielectric mirror probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT Principled dielectric mirror probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
