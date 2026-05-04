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
        description="Probe that textured diffuse materials feed Hardware RT GI."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=48)
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
    eevee.use_hardware_raytracing_gi = gi_enabled
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "OFF"
    eevee.use_hardware_raytracing_environment = False
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


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_black_world(scene):
    world = bpy.data.worlds.new("HWRTTexturedGIProbeWorld")
    world.use_nodes = True
    scene.world = world
    ntree = world.node_tree
    ntree.nodes.clear()
    background = ntree.nodes.new("ShaderNodeBackground")
    output = ntree.nodes.new("ShaderNodeOutputWorld")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    ntree.links.new(background.outputs["Background"], output.inputs["Surface"])


def make_diffuse_material(name: str, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 1.0
    return material


def make_texture_image(name: str, color):
    image = bpy.data.images.new(name=name, width=4, height=4, alpha=False, float_buffer=False)
    pixels = []
    for _ in range(16):
      pixels.extend((color[0], color[1], color[2], 1.0))
    image.pixels = pixels
    image.update()
    return image


def make_textured_diffuse_material(name: str):
    image_green = make_texture_image(f"{name}Green", (0.0, 1.0, 0.0))
    image_red = make_texture_image(f"{name}Red", (1.0, 0.0, 0.0))

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    image_node = ntree.nodes.new("ShaderNodeTexImage")
    bsdf = ntree.nodes.new("ShaderNodeBsdfDiffuse")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    image_node.interpolation = "Closest"
    ntree.links.new(texcoord.outputs["UV"], image_node.inputs["Vector"])
    ntree.links.new(image_node.outputs["Color"], bsdf.inputs["Color"])
    ntree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    image_node.image = image_green
    return material, image_node, image_green, image_red


def create_probe_layout(scene):
    floor_mat = make_diffuse_material("HWRTTexturedGIProbeFloor", (1.0, 1.0, 1.0, 1.0))
    cube_mat, image_node, image_green, image_red = make_textured_diffuse_material(
        "HWRTTexturedGIProbeCube"
    )

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "ProbeFloor"
    floor.scale = Vector((5.0, 5.0, 5.0))
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.55))
    cube = bpy.context.active_object
    cube.name = "ProbeCube"
    cube.scale = Vector((0.55, 0.55, 0.55))
    cube.data.materials.append(cube_mat)

    bpy.ops.object.light_add(type="POINT", location=(1.2, -1.6, 2.4))
    light = bpy.context.active_object
    light.data.energy = 6000.0
    light.data.shadow_soft_size = 0.05

    bpy.ops.object.camera_add(location=(2.7, -4.1, 2.2))
    camera = bpy.context.active_object
    look_at(camera, Vector((0.2, 0.1, 0.25)))
    scene.camera = camera

    receiver_patch = Vector((-0.82, 0.52, 0.0))
    return image_node, image_green, image_red, camera, receiver_patch


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


def assert_upper_bound(name: str, value: float, maximum: float, failures):
    print(f"{name}={value:.6f} ceiling={maximum:.6f}")
    if value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def assert_delta(name: str, value_a: float, value_b: float, minimum_delta: float, failures):
    delta = value_a - value_b
    print(f"{name}={delta:.6f} threshold={minimum_delta:.6f}")
    if delta < minimum_delta:
        failures.append(f"{name} expected delta >= {minimum_delta:.6f}, got {delta:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_textured_gi_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    clear_scene()
    scene = bpy.context.scene
    ensure_black_world(scene)
    image_node, image_green, image_red, camera, receiver_patch = create_probe_layout(scene)
    bpy.context.view_layer.update()

    configure_scene(scene, args.samples, gi_enabled=False)
    image_node.image = image_green
    width, height, gi_off_green = render_to_image(scene, output_dir, "gi_off_green")
    image_node.image = image_red
    width, height, gi_off_red = render_to_image(scene, output_dir, "gi_off_red")

    configure_scene(scene, args.samples, gi_enabled=True)
    image_node.image = image_green
    width, height, gi_on_green = render_to_image(scene, output_dir, "gi_on_green")
    image_node.image = image_red
    width, height, gi_on_red = render_to_image(scene, output_dir, "gi_on_red")

    del camera, receiver_patch
    receiver_bounds = (120, 180, 240, 300)
    off_diff = crop_abs_diff_mean(gi_off_green, gi_off_red, width, height, receiver_bounds)
    on_diff = crop_abs_diff_mean(gi_on_green, gi_on_red, width, height, receiver_bounds)

    failures = []
    print(f"RECEIVER_BOUNDS={receiver_bounds}")
    assert_upper_bound("TEXTURED_GI_OFF_DIFF_MEAN", off_diff, 0.03, failures)
    assert_metric("TEXTURED_GI_ON_DIFF_MEAN", on_diff, 0.03, failures)
    assert_delta("TEXTURED_GI_DIFF_DELTA", on_diff, off_diff, 0.008, failures)
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT textured GI probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT textured GI probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
