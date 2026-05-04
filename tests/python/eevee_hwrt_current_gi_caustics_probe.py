#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import subprocess
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
        description="Probe current traced GI plus Hardware RT caustics coexistence on a diffuse receiver."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to keep rendered outputs. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Render samples for each probe image.",
    )
    parser.add_argument(
        "--low-caustic-samples",
        type=int,
        default=2,
        help="Hardware caustics sample budget for the low-sample pass.",
    )
    parser.add_argument(
        "--high-caustic-samples",
        type=int,
        default=32,
        help="Hardware caustics sample budget for the high-sample pass.",
    )
    parser.add_argument("--child-render", action="store_true")
    parser.add_argument("--image-tag", type=str, default="")
    parser.add_argument("--caustics-enabled", type=int, default=0)
    parser.add_argument("--caustics-samples", type=int, default=1)
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
    eevee.hardware_raytracing_gi_mode = "ON"
    eevee.hardware_raytracing_reflection_mode = "FULL"
    eevee.hardware_raytracing_refraction_mode = "FULL"
    eevee.use_hardware_raytracing_environment = False
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


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def align_object_normal(obj, normal: Vector):
    obj.rotation_euler = normal.normalized().to_track_quat("Z", "Y").to_euler()


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTCurrentGICausticsProbeWorld")
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


def create_plane(name: str, location, scale, material):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = Vector(scale)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def create_scene_layout(scene):
    floor_mat = make_diffuse_material("HWRTCurrentGICausticsProbeFloor", (0.95, 0.95, 0.95, 1.0))
    wall_mat = make_diffuse_material("HWRTCurrentGICausticsProbeWall", (0.80, 0.80, 0.80, 1.0))
    mirror_mat = make_mirror_material("HWRTCurrentGICausticsProbeMirror")
    glass_mat = make_glass_material("HWRTCurrentGICausticsProbeGlass")

    create_plane("ProbeFloor", (0.0, 0.0, 0.0), (4.0, 4.0, 4.0), floor_mat)
    target_patch = Vector((0.9, 0.35, 0.0))

    back_wall = create_plane("ProbeBackWall", (0.0, 3.5, 1.6), (4.0, 1.6, 1.0), wall_mat)
    back_wall.rotation_euler.x = 1.57079632679

    mirror = create_plane("ProbeMirror", (2.4, 0.6, 1.2), (1.2, 1.2, 1.2), mirror_mat)

    bpy.ops.mesh.primitive_uv_sphere_add(location=(0.0, 0.0, 1.0), segments=64, ring_count=32, radius=1.0)
    sphere = bpy.context.active_object
    sphere.name = "ProbeGlassSphere"
    sphere.data.materials.clear()
    sphere.data.materials.append(glass_mat)

    bpy.ops.object.light_add(type="POINT", location=(-1.1, -3.0, 3.6))
    light = bpy.context.active_object
    light.name = "ProbePoint"
    light.data.energy = 8000.0
    light.data.shadow_soft_size = 0.05

    bpy.ops.object.camera_add(location=(0.0, -6.0, 3.0))
    camera = bpy.context.active_object
    camera.name = "ProbeCamera"
    look_at(camera, Vector((0.4, 0.2, 0.4)))
    scene.camera = camera

    mirror_view_dir = (camera.location - mirror.location).normalized()
    mirror_target_dir = (target_patch - mirror.location).normalized()
    align_object_normal(mirror, mirror_view_dir + mirror_target_dir)

    return back_wall, mirror, camera, target_patch


def render_to_image(scene, output_dir: Path, tag: str):
    scene.render.filepath = str(output_dir / f"{tag}.png")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


def load_image_pixels(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
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


def crop_max_tile_abs_diff_mean(pixels_a, pixels_b, width: int, height: int, bounds, tiles: int = 4):
    min_x, min_y, max_x, max_y = bounds
    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    best = 0.0
    for tile_y in range(tiles):
        for tile_x in range(tiles):
            tile_min_x = min_x + (tile_x * span_x) // tiles
            tile_max_x = min_x + ((tile_x + 1) * span_x) // tiles - 1
            tile_min_y = min_y + (tile_y * span_y) // tiles
            tile_max_y = min_y + ((tile_y + 1) * span_y) // tiles - 1
            best = max(
                best,
                crop_abs_diff_mean(
                    pixels_a, pixels_b, width, height, (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
                ),
            )
    return best


def projected_crop_bounds(scene, camera, point: Vector, width: int, height: int, radius_px: int):
    co = world_to_camera_view(scene, camera, point)
    center_x = int(round(co.x * width))
    center_y = int(round(co.y * height))
    min_x = max(0, center_x - radius_px)
    max_x = min(width - 1, center_x + radius_px)
    min_y = max(0, center_y - radius_px)
    max_y = min(height - 1, center_y + radius_px)
    return (min_x, min_y, max_x, max_y)


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


def assert_metric(name: str, value: float, minimum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def assert_upper_bound(name: str, value: float, maximum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold={maximum:.6f}")
    if value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def set_caustics(scene, enabled: bool, sample_count: int):
    scene.eevee.use_hardware_raytracing_caustics = enabled
    scene.eevee.ray_tracing_caustics_samples = max(1, sample_count)


def run_child_render(output_dir: Path, samples: int, tag: str, enabled: bool, caustics_samples: int):
    command = [
        bpy.app.binary_path,
        "-b",
        "--factory-startup",
        "-P",
        __file__,
        "--",
        "--output-dir",
        str(output_dir),
        "--samples",
        str(samples),
        "--child-render",
        "--image-tag",
        tag,
        "--caustics-enabled",
        "1" if enabled else "0",
        "--caustics-samples",
        str(caustics_samples),
    ]
    subprocess.run(command, check=True)


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_current_gi_caustics_probe_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args.samples)
    ensure_world(scene)
    _back_wall, mirror, camera, target_patch = create_scene_layout(scene)
    bpy.context.view_layer.update()

    if args.child_render:
        set_caustics(scene, bool(args.caustics_enabled), args.caustics_samples)
        render_to_image(scene, output_dir, args.image_tag)
        print(
            f"CHILD_RENDER tag={args.image_tag} enabled={bool(args.caustics_enabled)} "
            f"samples={args.caustics_samples} gi_backend=CURRENT"
        )
        return

    failures = []

    run_child_render(output_dir, args.samples, "caustics_off", False, args.low_caustic_samples)
    run_child_render(output_dir, args.samples, "caustics_low", True, args.low_caustic_samples)
    run_child_render(output_dir, args.samples, "caustics_high", True, args.high_caustic_samples)
    bpy.context.view_layer.update()

    width, height, caustics_off = load_image_pixels(output_dir / "caustics_off.png")
    _width, _height, caustics_low = load_image_pixels(output_dir / "caustics_low.png")
    _width, _height, caustics_high = load_image_pixels(output_dir / "caustics_high.png")

    direct_bounds = projected_crop_bounds(scene, camera, target_patch, width, height, 70)
    direct_toggle_diff = crop_max_tile_abs_diff_mean(
        caustics_off, caustics_high, width, height, direct_bounds, tiles=5
    )
    reflected_patch = mirror_reflection_point(camera.location, target_patch, mirror)
    reflection_bounds = projected_crop_bounds(scene, camera, reflected_patch, width, height, 70)
    control_patch = Vector((-1.8, 0.35, 0.0))
    control_bounds = projected_crop_bounds(scene, camera, control_patch, width, height, 70)
    direct_sample_diff = crop_max_tile_abs_diff_mean(
        caustics_low, caustics_high, width, height, direct_bounds, tiles=5
    )
    reflected_toggle_diff = crop_max_tile_abs_diff_mean(
        caustics_off, caustics_high, width, height, reflection_bounds, tiles=5
    )
    reflected_sample_diff = crop_max_tile_abs_diff_mean(
        caustics_low, caustics_high, width, height, reflection_bounds, tiles=5
    )
    control_toggle_diff = crop_max_tile_abs_diff_mean(
        caustics_off, caustics_high, width, height, control_bounds, tiles=5
    )

    print(f"DIRECT_BOUNDS={direct_bounds}")
    print(f"REFLECTION_BOUNDS={reflection_bounds}")
    print(f"CONTROL_BOUNDS={control_bounds}")
    assert_metric("CURRENT_GI_CAUSTICS_DIRECT_TOGGLE_DIFF_MEAN", direct_toggle_diff, 0.002, failures)
    assert_metric("CURRENT_GI_CAUSTICS_DIRECT_SAMPLE_DIFF_MEAN", direct_sample_diff, 0.0005, failures)
    assert_metric("CURRENT_GI_CAUSTICS_REFLECTED_TOGGLE_DIFF_MEAN", reflected_toggle_diff, 0.002, failures)
    assert_metric("CURRENT_GI_CAUSTICS_REFLECTED_SAMPLE_DIFF_MEAN", reflected_sample_diff, 0.0005, failures)
    assert_upper_bound("CURRENT_GI_CAUSTICS_CONTROL_TOGGLE_DIFF_MEAN", control_toggle_diff, 0.0015, failures)
    print(f"OUTPUT_DIR={output_dir}")

    if failures:
        print("EEVEE HWRT current GI caustics probe failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT current GI caustics probe passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
