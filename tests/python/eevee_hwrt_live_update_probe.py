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
        description="Probe live viewport update propagation for Eevee Hardware RT reflections."
    )
    parser.add_argument(
        "--mode",
        choices=("live_probe", "snapshot"),
        default="live_probe",
        help="Either run the in-session update probe or capture a single configured snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to keep viewport captures. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--checker-location-x",
        type=float,
        default=0.0,
        help="X translation for the checker mapping in snapshot mode.",
    )
    parser.add_argument(
        "--modifier-enabled",
        action="store_true",
        help="Enable the probe Subsurf modifier in snapshot mode.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="snapshot",
        help="Output tag for snapshot mode.",
    )
    parser.add_argument(
        "--sphere-mode",
        choices=("glass", "mirror"),
        default="glass",
        help="Use the scene glass material or override the sphere with a pure mirror material.",
    )
    parser.add_argument(
        "--redraw-iterations",
        type=int,
        default=8,
        help="Number of redraw iterations to wait after each edit before capture.",
    )
    parser.add_argument(
        "--taa-samples",
        type=int,
        default=1,
        help="Viewport TAA sample count to use for the probe. Use >1 for deferred accumulation checks.",
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        help="Keep Blender open instead of auto-quitting after the probe finishes.",
    )
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


def configure_scene(scene, taa_samples: int):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = "HARDWARE"
    scene.eevee.hardware_raytracing_reflection_mode = "FULL"
    scene.eevee.hardware_raytracing_refraction_mode = "FULL"
    scene.eevee.use_hardware_raytracing_environment = True
    scene.eevee.use_hardware_raytracing_shadows = True
    scene.eevee.taa_samples = max(1, taa_samples)
    scene.eevee.taa_render_samples = max(1, taa_samples)


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


def redraw(iterations: int = 8):
    for _ in range(iterations):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def object_bbox_pixels(scene, camera, obj, width: int, height: int, margin_px: int = 12):
    xs = []
    ys = []
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        co = world_to_camera_view(scene, camera, world_corner)
        xs.append(co.x)
        ys.append(co.y)

    min_x = max(0, math.floor(min(xs) * width) - margin_px)
    max_x = min(width - 1, math.ceil(max(xs) * width) + margin_px)
    min_y = max(0, math.floor(min(ys) * height) - margin_px)
    max_y = min(height - 1, math.ceil(max(ys) * height) + margin_px)
    return (min_x, min_y, max_x, max_y)


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


def load_pixels(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels


def capture_viewport(scene, window, area, region, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=str(path))
    width, height, pixels = load_pixels(path)
    return {"path": path, "width": width, "height": height, "pixels": pixels}


def ensure_checker_emission_material(name: str = "HWRTLiveUpdateChecker"):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        ntree = material.node_tree
        ntree.nodes.clear()

        texcoord = ntree.nodes.new("ShaderNodeTexCoord")
        mapping = ntree.nodes.new("ShaderNodeMapping")
        checker = ntree.nodes.new("ShaderNodeTexChecker")
        emission = ntree.nodes.new("ShaderNodeEmission")
        output = ntree.nodes.new("ShaderNodeOutputMaterial")

        texcoord.location = (-800, 0)
        mapping.location = (-600, 0)
        checker.location = (-350, 0)
        emission.location = (-100, 0)
        output.location = (150, 0)

        checker.inputs["Color1"].default_value = (1.0, 0.0, 0.0, 1.0)
        checker.inputs["Color2"].default_value = (0.0, 0.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 16.0

        ntree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
        ntree.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
        ntree.links.new(checker.outputs["Color"], emission.inputs["Color"])
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    mapping_node = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeMapping")
    return material, mapping_node


def ensure_mirror_material(name: str = "HWRTLiveUpdateMirror"):
    material = bpy.data.materials.get(name)
    if material is None:
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


def run_probe():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_live_update_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    camera = scene.camera
    cube = bpy.data.objects["Cube"]
    sphere = bpy.data.objects["Sphere"]
    original_cube_material = cube.material_slots[0].material
    original_sphere_material = sphere.material_slots[0].material
    checker_material, checker_mapping = ensure_checker_emission_material()
    mirror_material = ensure_mirror_material()
    modifier = cube.modifiers.get("HWRTLiveUpdateSubsurf")
    if modifier is None:
        modifier = cube.modifiers.new(name="HWRTLiveUpdateSubsurf", type="SUBSURF")
    modifier.show_viewport = False
    modifier.show_render = False

    configure_scene(scene, args.taa_samples)
    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    cube.material_slots[0].material = checker_material
    sphere.material_slots[0].material = (
        mirror_material if args.sphere_mode == "mirror" else original_sphere_material
    )
    checker_mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)

    if args.mode == "snapshot":
        checker_mapping.inputs["Location"].default_value = (args.checker_location_x, 0.0, 0.0)
        modifier.show_viewport = args.modifier_enabled
        modifier.show_render = args.modifier_enabled
        checker_material.node_tree.update_tag()
        cube.update_tag()
        redraw(args.redraw_iterations)
        capture_viewport(scene, window, area, region, output_dir, args.tag)
        print(f"OUTPUT_DIR={output_dir}")
    else:
        checker_mapping.inputs["Location"].default_value = (0.0, 0.0, 0.0)
        redraw(args.redraw_iterations)
        phase_a = capture_viewport(scene, window, area, region, output_dir, "checker_phase_a")

        sphere_bounds = object_bbox_pixels(scene, camera, sphere, phase_a["width"], phase_a["height"])
        cube_bounds = object_bbox_pixels(scene, camera, cube, phase_a["width"], phase_a["height"])

        checker_mapping.inputs["Location"].default_value = (0.125, 0.0, 0.0)
        checker_material.node_tree.update_tag()
        cube.update_tag()
        redraw(args.redraw_iterations)
        phase_b = capture_viewport(scene, window, area, region, output_dir, "checker_phase_b")

        checker_cube_diff = crop_abs_diff_mean(
            phase_a["pixels"], phase_b["pixels"], phase_a["width"], phase_a["height"], cube_bounds
        )
        checker_sphere_diff = crop_abs_diff_mean(
            phase_a["pixels"], phase_b["pixels"], phase_a["width"], phase_a["height"], sphere_bounds
        )
        print(f"CHECKER_DIRECT_CUBE_DIFF_MEAN={checker_cube_diff:.6f}")
        print(f"CHECKER_REFLECTION_SPHERE_DIFF_MEAN={checker_sphere_diff:.6f}")

        modifier.show_viewport = True
        modifier.show_render = True
        cube.update_tag()
        redraw(args.redraw_iterations)
        modifier_on = capture_viewport(scene, window, area, region, output_dir, "modifier_on")

        modifier.show_viewport = False
        modifier.show_render = False
        cube.update_tag()
        redraw(args.redraw_iterations)
        modifier_off = capture_viewport(scene, window, area, region, output_dir, "modifier_off")

        modifier_cube_diff = crop_abs_diff_mean(
            modifier_on["pixels"],
            modifier_off["pixels"],
            modifier_on["width"],
            modifier_on["height"],
            cube_bounds,
        )
        modifier_sphere_diff = crop_abs_diff_mean(
            modifier_on["pixels"],
            modifier_off["pixels"],
            modifier_on["width"],
            modifier_on["height"],
            sphere_bounds,
        )
        print(f"MODIFIER_DIRECT_CUBE_DIFF_MEAN={modifier_cube_diff:.6f}")
        print(f"MODIFIER_REFLECTION_SPHERE_DIFF_MEAN={modifier_sphere_diff:.6f}")
        print(f"OUTPUT_DIR={output_dir}")

    cube.material_slots[0].material = original_cube_material
    sphere.material_slots[0].material = original_sphere_material
    modifier.show_viewport = False
    modifier.show_render = False

    if cleanup_dir is not None:
        cleanup_dir.cleanup()

    if not args.hold_open:
        def _quit_blender():
            bpy.ops.wm.quit_blender()
            return None

        bpy.app.timers.register(_quit_blender, first_interval=0.1)


run_probe()
