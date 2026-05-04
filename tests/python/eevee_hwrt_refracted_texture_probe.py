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
        description="Probe that HWRT refraction picks up textured scene hits behind glass."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--lighting-mode", choices=("sun", "environment"), default="environment")
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


def configure_scene(scene, samples: int, *, environment_enabled: bool, gi_enabled: bool):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.use_hardware_raytracing_gi = gi_enabled
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "FULL"
    eevee.use_hardware_raytracing_environment = environment_enabled
    eevee.use_hardware_raytracing_shadows = False
    eevee.taa_render_samples = max(1, samples)
    eevee.ray_tracing_refraction_bounces = 6

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.use_denoise = False
    ray_tracing.denoise_spatial = False
    ray_tracing.denoise_temporal = False
    ray_tracing.denoise_bilateral = False
    ray_tracing.screen_trace_quality = 1.0
    ray_tracing.screen_trace_thickness = 1.0


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.images,
        bpy.data.worlds,
    ):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def ensure_black_world(scene):
    world = bpy.data.worlds.new("HWRTRefractedTextureProbeWorld")
    world.use_nodes = True
    scene.world = world
    ntree = world.node_tree
    ntree.nodes.clear()
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])


def ensure_environment_world(scene):
    world = bpy.data.worlds.new("HWRTRefractedTextureProbeEnvWorld")
    world.use_nodes = True
    scene.world = world
    ntree = world.node_tree
    ntree.nodes.clear()
    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    mapping = ntree.nodes.new("ShaderNodeMapping")
    environment = ntree.nodes.new("ShaderNodeTexEnvironment")
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")
    image = bpy.data.images.new("HWRTRefractedTextureProbeHDRI", width=128, height=64, float_buffer=True)
    pixels = []
    for y in range(64):
        v = y / 63.0
        for x in range(128):
            u = x / 127.0
            warm = max(0.0, 1.0 - abs(u - 0.75) * 5.0)
            cool = max(0.0, 1.0 - abs(u - 0.20) * 6.0)
            horizon = v
            r = 0.03 + 0.10 * horizon + 2.8 * warm
            g = 0.03 + 0.12 * horizon + 1.2 * warm + 0.4 * cool
            b = 0.04 + 0.16 * horizon + 2.2 * cool
            pixels.extend((r, g, b, 1.0))
    image.pixels = pixels
    image.update()
    environment.image = image
    background.inputs["Strength"].default_value = 1.0
    ntree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    ntree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    ntree.links.new(environment.outputs["Color"], background.inputs["Color"])
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_checker_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    checker = ntree.nodes.new("ShaderNodeTexChecker")
    diffuse = ntree.nodes.new("ShaderNodeBsdfDiffuse")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    checker.inputs["Scale"].default_value = 12.0
    ntree.links.new(texcoord.outputs["UV"], checker.inputs["Vector"])
    ntree.links.new(checker.outputs["Color"], diffuse.inputs["Color"])
    ntree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material, checker


def make_glass_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    glass = ntree.nodes.new("ShaderNodeBsdfGlass")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["IOR"].default_value = 1.45
    ntree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_layout(scene, lighting_mode: str):
    floor_mat, checker = make_checker_material("HWRTRefractedTextureProbeFloor")
    glass_mat = make_glass_material("HWRTRefractedTextureProbeGlass")

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.scale = Vector((5.0, 5.0, 5.0))
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_uv_sphere_add(location=(0.0, 0.0, 0.65), segments=64, ring_count=32)
    sphere = bpy.context.active_object
    sphere.scale = Vector((0.9, 0.9, 0.9))
    sphere.data.materials.append(glass_mat)

    if lighting_mode == "sun":
        bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 3.0))
        light = bpy.context.active_object
        light.rotation_euler = (0.75, 0.0, 0.55)
        light.data.energy = 2.0

    bpy.ops.object.camera_add(location=(0.0, -4.8, 2.6))
    camera = bpy.context.active_object
    look_at(camera, Vector((0.0, 0.0, 0.55)))
    scene.camera = camera
    return checker, sphere, camera


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
    return (
        max(0, center_x - radius_px),
        max(0, center_y - radius_px),
        min(width - 1, center_x + radius_px),
        min(height - 1, center_y + radius_px),
    )


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


def crop_mean_luma(pixels, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    total = 0.0
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            r = pixels[base + 0]
            g = pixels[base + 1]
            b = pixels[base + 2]
            total += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
    return total / max(1, count)


def assert_metric(name: str, value: float, minimum: float, failures):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def assert_upper_bound(name: str, value: float, maximum: float, failures):
    print(f"{name}={value:.6f} ceiling={maximum:.6f}")
    if value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def render_checker_swap_pair(scene, checker, output_dir: Path, tag_prefix: str):
    checker.inputs["Color1"].default_value = (1.0, 0.0, 0.0, 1.0)
    checker.inputs["Color2"].default_value = (0.0, 1.0, 0.0, 1.0)
    width, height, image_a = render_to_image(scene, output_dir, f"{tag_prefix}_a")

    checker.inputs["Color1"].default_value = (0.0, 1.0, 0.0, 1.0)
    checker.inputs["Color2"].default_value = (1.0, 0.0, 0.0, 1.0)
    width, height, image_b = render_to_image(scene, output_dir, f"{tag_prefix}_b")
    return width, height, image_a, image_b


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_refracted_texture_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    clear_scene()
    scene = bpy.context.scene
    if args.lighting_mode == "environment":
        ensure_environment_world(scene)
    else:
        ensure_black_world(scene)
    checker, sphere, camera = create_layout(scene, args.lighting_mode)
    configure_scene(
        scene,
        args.samples,
        environment_enabled=(args.lighting_mode == "environment"),
        gi_enabled=False,
    )
    bpy.context.view_layer.update()

    width, height, image_a, image_b = render_checker_swap_pair(
        scene, checker, output_dir, "refract_env_only"
    )

    sphere_bounds = projected_crop_bounds(scene, camera, sphere.location, width, height, 110)
    refract_diff = crop_abs_diff_mean(image_a, image_b, width, height, sphere_bounds)

    failures = []
    print(f"SPHERE_BOUNDS={sphere_bounds}")
    assert_metric("REFRACTED_TEXTURE_DIFF_MEAN", refract_diff, 0.01, failures)
    if args.lighting_mode == "environment":
        env_only_luma = 0.5 * (
            crop_mean_luma(image_a, width, height, sphere_bounds)
            + crop_mean_luma(image_b, width, height, sphere_bounds)
        )
        configure_scene(scene, args.samples, environment_enabled=True, gi_enabled=True)
        bpy.context.view_layer.update()
        _width, _height, gi_env_a, gi_env_b = render_checker_swap_pair(
            scene, checker, output_dir, "refract_gi_env"
        )
        gi_env_diff = crop_abs_diff_mean(gi_env_a, gi_env_b, width, height, sphere_bounds)
        gi_env_luma = 0.5 * (
            crop_mean_luma(gi_env_a, width, height, sphere_bounds)
            + crop_mean_luma(gi_env_b, width, height, sphere_bounds)
        )
        print(f"REFRACTED_TEXTURE_ENV_ONLY_LUMA={env_only_luma:.6f}")
        print(f"REFRACTED_TEXTURE_GI_ENV_LUMA={gi_env_luma:.6f}")
        assert_metric("REFRACTED_TEXTURE_GI_ENV_DIFF_MEAN", gi_env_diff, 0.01, failures)
        assert_upper_bound(
            "REFRACTED_TEXTURE_GI_ENV_DARKEN_DELTA",
            env_only_luma - gi_env_luma,
            0.03,
            failures,
        )
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT refracted texture probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT refracted texture probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
