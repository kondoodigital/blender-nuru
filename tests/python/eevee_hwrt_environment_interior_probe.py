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
        description="Probe that HWRT environment light stays aperture-limited indoors and GI carries it deeper."
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


def configure_scene(scene, samples: int, *, gi_enabled: bool):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.hardware_raytracing_gi_mode = "ON" if gi_enabled else "OFF"
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "OFF"
    eevee.use_hardware_raytracing_environment = True
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

    warm_dir = Vector((0.15, 0.92, 0.35)).normalized()
    cool_dir = Vector((-0.70, 0.10, 0.70)).normalized()
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
            ambient = 0.02 + 0.04 * horizon
            warm = max(direction.dot(warm_dir), 0.0) ** 18
            cool = max(direction.dot(cool_dir), 0.0) ** 16
            r = ambient + warm * 7.0 + cool * 0.1
            g = ambient * 1.2 + warm * 3.2 + cool * 0.4
            b = ambient * 1.4 + warm * 0.9 + cool * 4.0
            base = (y * width + x) * 4
            pixels[base + 0] = r
            pixels[base + 1] = g
            pixels[base + 2] = b
            pixels[base + 3] = 1.0

    image.pixels[:] = pixels
    return image


def ensure_environment_world(scene):
    world = bpy.data.worlds.new("HWRTEnvironmentInteriorProbeWorld")
    world.use_nodes = True
    scene.world = world

    ntree = world.node_tree
    ntree.nodes.clear()
    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    mapping = ntree.nodes.new("ShaderNodeMapping")
    environment = ntree.nodes.new("ShaderNodeTexEnvironment")
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")

    environment.image = create_environment_image("HWRTEnvironmentInteriorProbeHDRI")
    background.inputs["Strength"].default_value = 1.0
    mapping.inputs["Rotation"].default_value[2] = math.radians(90.0)

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


def add_box(name: str, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = Vector(scale)
    obj.data.materials.append(material)
    return obj


def create_scene_layout(scene):
    wall_mat = make_diffuse_material("HWRTEnvironmentInteriorProbeWall", (0.84, 0.84, 0.84, 1.0))
    floor_mat = make_diffuse_material("HWRTEnvironmentInteriorProbeFloor", (0.74, 0.74, 0.74, 1.0))

    add_box("RoomFloor", (0.0, 0.0, -0.05), (4.0, 4.0, 0.05), floor_mat)
    add_box("RoomCeiling", (0.0, 0.0, 3.05), (4.0, 4.0, 0.05), wall_mat)
    add_box("RoomBack", (0.0, -4.05, 1.5), (4.0, 0.05, 1.5), wall_mat)
    add_box("RoomLeft", (-4.05, 0.0, 1.5), (0.05, 4.0, 1.5), wall_mat)
    add_box("RoomRight", (4.05, 0.0, 1.5), (0.05, 4.0, 1.5), wall_mat)
    add_box("FrontLeft", (-2.525, 4.05, 1.5), (1.475, 0.05, 1.5), wall_mat)
    add_box("FrontRight", (2.525, 4.05, 1.5), (1.475, 0.05, 1.5), wall_mat)
    add_box("FrontBottom", (0.0, 4.05, 0.45), (1.05, 0.05, 0.45), wall_mat)
    add_box("FrontTop", (0.0, 4.05, 2.55), (1.05, 0.05, 0.45), wall_mat)

    bpy.ops.object.camera_add(location=(0.0, -3.2, 1.75))
    camera = bpy.context.active_object
    camera.name = "ProbeCamera"
    look_at(camera, Vector((0.0, 3.2, 0.15)))
    scene.camera = camera

    return camera


def render_to_image(scene, output_dir: Path, tag: str):
    scene.render.filepath = str(output_dir / f"{tag}.png")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


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


def assert_metric(name: str, value: float, minimum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def assert_upper_bound(name: str, value: float, maximum: float, failures: list[str]):
    print(f"{name}={value:.6f} ceiling={maximum:.6f}")
    if value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_environment_interior_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    clear_scene()
    scene = bpy.context.scene
    ensure_environment_world(scene)
    camera = create_scene_layout(scene)
    bpy.context.view_layer.update()

    configure_scene(scene, args.samples, gi_enabled=False)
    width, height, gi_off = render_to_image(scene, output_dir, "gi_off")

    configure_scene(scene, args.samples, gi_enabled=True)
    width, height, gi_on = render_to_image(scene, output_dir, "gi_on")

    del camera
    interior_bounds = (280, 260, 680, 430)

    interior_off = crop_mean_luma(gi_off, width, height, interior_bounds)
    interior_on = crop_mean_luma(gi_on, width, height, interior_bounds)

    failures = []
    print(f"INTERIOR_BOUNDS={interior_bounds}")
    assert_upper_bound("ENV_INTERIOR_DIRECT_GI_OFF_LUMA", interior_off, 0.0250, failures)
    assert_metric("ENV_INTERIOR_GI_ON_LUMA", interior_on, 0.0200, failures)
    assert_metric("ENV_INTERIOR_GI_LIFT", interior_on - interior_off, 0.0100, failures)
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT environment interior probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT environment interior probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
