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
        description="Probe GI-off Hardware RT environment fallback on a diffuse HDRI scene."
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


def configure_scene(scene, samples: int, *, method: str, environment_enabled: bool):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = method
    eevee.use_hardware_raytracing_gi = False
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "OFF"
    eevee.use_hardware_raytracing_environment = environment_enabled
    eevee.use_hardware_raytracing_shadows = False
    eevee.taa_samples = max(1, samples)
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


def create_environment_image(name: str, width: int = 256, height: int = 128):
    image = bpy.data.images.new(name=name, width=width, height=height, float_buffer=True)
    image.colorspace_settings.name = "scene_linear"

    warm_dir = Vector((0.45, -0.75, 0.48)).normalized()
    cool_dir = Vector((-0.85, 0.10, 0.52)).normalized()
    pixels = [0.0] * (width * height * 4)

    for y in range(height):
        v = (y + 0.5) / height
        theta = v * math.pi
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for x in range(width):
            u = (x + 0.5) / width
            phi = (u - 0.5) * 2.0 * math.pi
            direction = Vector(
                (
                    sin_theta * math.cos(phi),
                    sin_theta * math.sin(phi),
                    cos_theta,
                )
            )
            horizon = max(direction.z, 0.0)
            ambient = 0.03 + 0.08 * horizon
            warm = max(direction.dot(warm_dir), 0.0) ** 14
            cool = max(direction.dot(cool_dir), 0.0) ** 18
            r = ambient + warm * 5.0 + cool * 0.2
            g = ambient * 1.2 + warm * 2.2 + cool * 0.5
            b = ambient * 1.4 + warm * 0.8 + cool * 3.4
            base = (y * width + x) * 4
            pixels[base + 0] = r
            pixels[base + 1] = g
            pixels[base + 2] = b
            pixels[base + 3] = 1.0

    image.pixels[:] = pixels
    return image


def ensure_environment_world(scene):
    world = bpy.data.worlds.new("HWRTGIEnvironmentProbeWorld")
    world.use_nodes = True
    scene.world = world

    ntree = world.node_tree
    ntree.nodes.clear()
    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    mapping = ntree.nodes.new("ShaderNodeMapping")
    environment = ntree.nodes.new("ShaderNodeTexEnvironment")
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")

    environment.image = create_environment_image("HWRTGIEnvironmentProbeHDRI")
    background.inputs["Strength"].default_value = 1.0

    ntree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    ntree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    ntree.links.new(environment.outputs["Color"], background.inputs["Color"])
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])


def make_diffuse_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 1.0
    return material


def create_scene_layout(scene):
    floor_mat = make_diffuse_material("HWRTGIEnvironmentProbeFloor", (0.78, 0.78, 0.78, 1.0))
    blocker_mat = make_diffuse_material("HWRTGIEnvironmentProbeBlocker", (0.18, 0.18, 0.18, 1.0))

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 1.4, 0.0))
    floor = bpy.context.active_object
    floor.name = "ProbeFloor"
    floor.scale = Vector((5.5, 5.5, 5.5))
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 1.45, 1.20))
    blocker = bpy.context.active_object
    blocker.name = "ProbeBlocker"
    blocker.scale = Vector((1.20, 1.20, 0.12))
    blocker.data.materials.append(blocker_mat)

    bpy.ops.object.camera_add(location=(0.0, -6.0, 3.2))
    camera = bpy.context.active_object
    camera.name = "ProbeCamera"
    look_at(camera, Vector((0.0, 1.4, 0.6)))
    scene.camera = camera

    shadow_patch = Vector((0.0, 1.45, 0.0))
    light_patch = Vector((1.9, 1.45, 0.0))
    return floor, blocker, camera, shadow_patch, light_patch


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


def object_crop_bounds(scene, camera, obj, width: int, height: int, pad_px: int = 12):
    coords = []
    world_matrix = obj.matrix_world
    for corner in obj.bound_box:
        co = world_to_camera_view(scene, camera, world_matrix @ Vector(corner))
        coords.append((co.x, co.y))

    min_x = int(math.floor(min(co[0] for co in coords) * width)) - pad_px
    max_x = int(math.ceil(max(co[0] for co in coords) * width)) + pad_px
    min_y = int(math.floor(min(co[1] for co in coords) * height)) - pad_px
    max_y = int(math.ceil(max(co[1] for co in coords) * height)) + pad_px
    return (
        max(0, min_x),
        max(0, min_y),
        min(width - 1, max_x),
        min(height - 1, max_y),
    )


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


def crop_tile_luminance_range(pixels, width: int, height: int, bounds, tiles: int = 5):
    min_x, min_y, max_x, max_y = bounds
    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    min_tile = float("inf")
    max_tile = 0.0

    for tile_y in range(tiles):
        for tile_x in range(tiles):
            tile_min_x = min_x + (tile_x * span_x) // tiles
            tile_max_x = min_x + ((tile_x + 1) * span_x) // tiles - 1
            tile_min_y = min_y + (tile_y * span_y) // tiles
            tile_max_y = min_y + ((tile_y + 1) * span_y) // tiles - 1
            mean_luma = crop_mean_luma(
                pixels, width, height, (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
            )
            min_tile = min(min_tile, mean_luma)
            max_tile = max(max_tile, mean_luma)

    return max_tile - min_tile


def crop_abs_diff_mean(pixels_a, pixels_b, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    total = 0.0
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            total += abs(pixels_a[base + 0] - pixels_b[base + 0])
            total += abs(pixels_a[base + 1] - pixels_b[base + 1])
            total += abs(pixels_a[base + 2] - pixels_b[base + 2])
            count += 3
    return total / max(1, count)


def assert_metric(name: str, value: float, minimum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_gi_off_environment_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    clear_scene()
    scene = bpy.context.scene
    ensure_environment_world(scene)
    floor, _blocker, camera, shadow_patch, light_patch = create_scene_layout(scene)
    bpy.context.view_layer.update()

    configure_scene(scene, args.samples, method="SCREEN", environment_enabled=False)
    width, height, screen_classic = render_to_image(scene, output_dir, "screen_classic")

    configure_scene(scene, args.samples, method="HARDWARE", environment_enabled=True)
    width, height, hwrt_env_on = render_to_image(scene, output_dir, "hardware_gi_off_env_on")

    floor_bounds = object_crop_bounds(scene, camera, floor, width, height, pad_px=10)
    shadow_bounds = projected_crop_bounds(scene, camera, shadow_patch, width, height, 56)
    light_bounds = projected_crop_bounds(scene, camera, light_patch, width, height, 56)

    screen_shadow_luma = crop_mean_luma(screen_classic, width, height, shadow_bounds)
    hw_shadow_luma = crop_mean_luma(hwrt_env_on, width, height, shadow_bounds)
    shadow_diff = crop_abs_diff_mean(screen_classic, hwrt_env_on, width, height, shadow_bounds)
    floor_variation = crop_tile_luminance_range(hwrt_env_on, width, height, floor_bounds, tiles=6)

    failures = []
    print(f"FLOOR_BOUNDS={floor_bounds}")
    print(f"SHADOW_BOUNDS={shadow_bounds}")
    print(f"LIGHT_BOUNDS={light_bounds}")
    assert_metric("HWRT_ENV_SHADOW_LUMA", hw_shadow_luma, 0.0200, failures)
    assert_metric("GI_OFF_ENV_SHADOW_DIFF_MEAN", shadow_diff, 0.0080, failures)
    assert_metric("HWRT_ENV_FLOOR_VARIATION", floor_variation, 0.0100, failures)
    assert_metric("SCREEN_CLASSIC_SHADOW_LUMA", screen_shadow_luma, 0.0400, failures)
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT GI-off environment probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT GI-off environment probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
