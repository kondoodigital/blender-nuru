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
        description="Probe Rendered viewport Principled reflection continuity in Eevee Hardware RT."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--redraw-iterations", type=int, default=12)
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


def configure_scene(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = "HARDWARE"
    scene.eevee.hardware_raytracing_reflection_mode = "FULL"
    scene.eevee.hardware_raytracing_refraction_mode = "FULL"
    scene.eevee.use_hardware_raytracing_environment = True
    scene.eevee.use_hardware_raytracing_shadows = True
    scene.eevee.taa_samples = 1
    scene.eevee.taa_render_samples = 1
    scene.eevee.ray_tracing_reflection_bounces = 4
    scene.eevee.ray_tracing_refraction_bounces = 4
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100


def find_view3d_context():
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((region for region in area.regions if region.type == "WINDOW"), None)
            if region is None:
                continue
            return window, area, region, area.spaces.active
    raise RuntimeError("No VIEW_3D area available for viewport probe.")


def force_viewport_camera(window, area, region, space):
    space.shading.type = "RENDERED"
    space.overlay.show_overlays = False
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.view3d.view_camera()


def redraw(window, area, region, iterations: int):
    for _ in range(iterations):
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def align_object_normal(obj, normal: Vector):
    obj.rotation_euler = normal.normalized().to_track_quat("Z", "Y").to_euler()


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTPrincipledReflectionLiveWorld")
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
    principled.inputs["Base Color"].default_value = (0.8, 0.72, 0.65, 1.0)
    principled.inputs["Roughness"].default_value = 0.0
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Transmission Weight"].default_value = 0.0
    principled.inputs["IOR"].default_value = 1.45
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


def load_pixels(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


def capture_viewport(scene, window, area, region, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    with bpy.context.temp_override(window=window, area=area, region=region):
        scene.render.filepath = str(path)
        bpy.ops.render.opengl(write_still=True, view_context=True)
    width, height, pixels = load_pixels(path)
    return {"path": path, "width": width, "height": height, "pixels": pixels}


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


def lower_reflection_band_bounds(
    bounds, inset_x_frac: float = 0.18, start_y_frac: float = 0.56, end_y_frac: float = 0.94
):
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    inset_x = int(round(width * inset_x_frac))
    band_min_y = min_y + int(round((height - 1) * start_y_frac))
    band_max_y = min_y + int(round((height - 1) * end_y_frac))
    return (
        min(max_x, min_x + inset_x),
        max(min_y, min(max_y, band_min_y)),
        max(min_x, max_x - inset_x),
        max(min_y, min(max_y, band_max_y)),
    )


def crop_luma_mean_stddev(pixels, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    total = 0.0
    total_sq = 0.0
    count = 0
    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            luma = (
                pixels[base + 0] * 0.2126
                + pixels[base + 1] * 0.7152
                + pixels[base + 2] * 0.0722
            )
            total += luma
            total_sq += luma * luma
            count += 1
    mean = total / max(1, count)
    variance = max(total_sq / max(1, count) - mean * mean, 0.0)
    return mean, math.sqrt(variance)


def create_object(add_op):
    existing = set(bpy.data.objects.keys())
    add_op()
    created = [name for name in bpy.data.objects.keys() if name not in existing]
    if not created:
        raise RuntimeError("Expected the add operator to create an object.")
    return bpy.data.objects[created[-1]]


def run_probe():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_principled_reflection_live_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene)
    ensure_world(scene)

    mirror_material = make_mirror_material("HWRTPrincipledReflectionLiveMirror")
    sphere_material, sphere_node = make_principled_material("HWRTPrincipledReflectionLiveSphere")
    floor_material = make_floor_material("HWRTPrincipledReflectionLiveFloor")
    backlight_material = make_backlight_material("HWRTPrincipledReflectionLiveBacklight")

    sphere = create_object(
        lambda: bpy.ops.mesh.primitive_uv_sphere_add(
            location=(0.0, 0.0, 1.05), segments=64, ring_count=32
        )
    )
    sphere.scale = Vector((0.9, 0.9, 0.9))
    sphere.data.materials.clear()
    sphere.data.materials.append(sphere_material)

    floor = create_object(lambda: bpy.ops.mesh.primitive_plane_add(size=7.0, location=(0.0, 0.0, 0.0)))
    floor.data.materials.clear()
    floor.data.materials.append(floor_material)

    mirror = create_object(
        lambda: bpy.ops.mesh.primitive_plane_add(size=2.0, location=(2.5, 0.25, 1.2))
    )
    mirror.scale = Vector((1.2, 1.2, 1.2))
    mirror.data.materials.clear()
    mirror.data.materials.append(mirror_material)

    backlight = create_object(
        lambda: bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0.0, 2.8, 1.2))
    )
    backlight.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    backlight.data.materials.clear()
    backlight.data.materials.append(backlight_material)

    camera = create_object(lambda: bpy.ops.object.camera_add(location=(0.0, -6.0, 2.8)))
    look_at(camera, Vector((0.5, 0.15, 1.0)))
    scene.camera = camera

    mirror_view_dir = (camera.location - mirror.location).normalized()
    mirror_target_dir = (sphere.location - mirror.location).normalized()
    align_object_normal(mirror, mirror_view_dir + mirror_target_dir)

    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)
    redraw(window, area, region, args.redraw_iterations)

    reflection_point = mirror_reflection_point(sphere.location, mirror)

    def capture_case(tag: str):
        redraw(window, area, region, args.redraw_iterations)
        return capture_viewport(scene, window, area, region, output_dir, tag)

    sphere_node.inputs["Transmission Weight"].default_value = 0.0
    sphere_node.inputs["Metallic"].default_value = 0.0
    base_case = capture_case("metallic_base")
    reflection_bounds = projected_crop_bounds(
        scene, camera, reflection_point, base_case["width"], base_case["height"], 90
    )
    ground_reflection_bounds = lower_reflection_band_bounds(reflection_bounds)

    sphere_node.inputs["Metallic"].default_value = 0.49
    metallic_mid_a = capture_case("metallic_mid_a")
    sphere_node.inputs["Metallic"].default_value = 0.51
    metallic_mid_b = capture_case("metallic_mid_b")
    sphere_node.inputs["Metallic"].default_value = 0.88
    metallic_target = capture_case("metallic_088")
    sphere_node.inputs["Metallic"].default_value = 0.99
    metallic_near_full = capture_case("metallic_099")
    sphere_node.inputs["Metallic"].default_value = 1.0
    metallic_full = capture_case("metallic_full")

    metallic_mid_diff = crop_abs_diff_mean(
        metallic_mid_a["pixels"],
        metallic_mid_b["pixels"],
        metallic_mid_a["width"],
        metallic_mid_a["height"],
        reflection_bounds,
    )
    metallic_full_diff = crop_abs_diff_mean(
        base_case["pixels"],
        metallic_full["pixels"],
        base_case["width"],
        base_case["height"],
        reflection_bounds,
    )
    metallic_target_diff = crop_abs_diff_mean(
        metallic_target["pixels"],
        metallic_full["pixels"],
        metallic_target["width"],
        metallic_target["height"],
        reflection_bounds,
    )
    metallic_near_full_diff = crop_abs_diff_mean(
        metallic_near_full["pixels"],
        metallic_full["pixels"],
        metallic_near_full["width"],
        metallic_near_full["height"],
        reflection_bounds,
    )
    metallic_target_ground_mean, metallic_target_ground_stddev = crop_luma_mean_stddev(
        metallic_target["pixels"],
        metallic_target["width"],
        metallic_target["height"],
        ground_reflection_bounds,
    )
    metallic_full_ground_mean, metallic_full_ground_stddev = crop_luma_mean_stddev(
        metallic_full["pixels"],
        metallic_full["width"],
        metallic_full["height"],
        ground_reflection_bounds,
    )

    sphere_node.inputs["Metallic"].default_value = 0.0
    sphere_node.inputs["Transmission Weight"].default_value = 0.0
    transmission_base = capture_case("transmission_base")
    sphere_node.inputs["Transmission Weight"].default_value = 0.05
    transmission_mid = capture_case("transmission_mid")
    sphere_node.inputs["Transmission Weight"].default_value = 1.0
    transmission_full = capture_case("transmission_full")

    transmission_mid_diff = crop_abs_diff_mean(
        transmission_base["pixels"],
        transmission_mid["pixels"],
        transmission_base["width"],
        transmission_base["height"],
        reflection_bounds,
    )
    transmission_full_diff = crop_abs_diff_mean(
        transmission_base["pixels"],
        transmission_full["pixels"],
        transmission_base["width"],
        transmission_base["height"],
        reflection_bounds,
    )

    print(f"REFLECTION_BOUNDS={reflection_bounds}")
    print(f"GROUND_REFLECTION_BOUNDS={ground_reflection_bounds}")
    print(f"LIVE_PRINCIPLED_METALLIC_MID_DIFF_MEAN={metallic_mid_diff:.6f}")
    print(f"LIVE_PRINCIPLED_METALLIC_FULL_DIFF_MEAN={metallic_full_diff:.6f}")
    print(
        f"LIVE_PRINCIPLED_METALLIC_MID_RATIO={metallic_mid_diff / max(metallic_full_diff, 1.0e-6):.6f}"
    )
    print(f"LIVE_PRINCIPLED_METALLIC_088_TO_100_DIFF_MEAN={metallic_target_diff:.6f}")
    print(f"LIVE_PRINCIPLED_METALLIC_099_TO_100_DIFF_MEAN={metallic_near_full_diff:.6f}")
    print(f"LIVE_PRINCIPLED_METALLIC_088_GROUND_LUMA_MEAN={metallic_target_ground_mean:.6f}")
    print(
        f"LIVE_PRINCIPLED_METALLIC_088_GROUND_LUMA_STDDEV={metallic_target_ground_stddev:.6f}"
    )
    print(f"LIVE_PRINCIPLED_METALLIC_100_GROUND_LUMA_MEAN={metallic_full_ground_mean:.6f}")
    print(
        f"LIVE_PRINCIPLED_METALLIC_100_GROUND_LUMA_STDDEV={metallic_full_ground_stddev:.6f}"
    )
    print(f"LIVE_PRINCIPLED_TRANSMISSION_MID_DIFF_MEAN={transmission_mid_diff:.6f}")
    print(f"LIVE_PRINCIPLED_TRANSMISSION_FULL_DIFF_MEAN={transmission_full_diff:.6f}")
    print(
        f"LIVE_PRINCIPLED_TRANSMISSION_MID_RATIO={transmission_mid_diff / max(transmission_full_diff, 1.0e-6):.6f}"
    )
    print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()

    def _quit_blender():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(_quit_blender, first_interval=0.1)


run_probe()
