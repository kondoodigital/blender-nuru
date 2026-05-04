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
        description="Probe live checker updates in a strong planar mirror setup on scenes/test.blend."
    )
    parser.add_argument(
        "--mode",
        choices=("live_probe", "snapshot"),
        default="live_probe",
        help="Either run the in-session mirror probe or capture a single configured snapshot.",
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
        help="X translation for the checker mapping.",
    )
    parser.add_argument(
        "--probe",
        choices=("checker_shift", "modifier_toggle", "image_swap_generated", "image_swap_uv"),
        default="checker_shift",
        help="Which reflected update to exercise in the planar mirror setup.",
    )
    parser.add_argument(
        "--image-state",
        choices=("A", "B"),
        default="A",
        help="Image selection for snapshot mode when using an image-swap probe.",
    )
    parser.add_argument(
        "--modifier-enabled",
        action="store_true",
        help="Enable the probe Subsurf modifier in snapshot mode for modifier-toggle probes.",
    )
    parser.add_argument(
        "--redraw-iterations",
        type=int,
        default=1,
        help="Number of redraw iterations to wait after each edit before capture.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="snapshot",
        help="Output tag for snapshot mode.",
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


def redraw(iterations: int = 1):
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


def ensure_checker_emission_material(name: str = "HWRTPlaneMirrorChecker"):
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

        checker.inputs["Color1"].default_value = (1.0, 0.0, 0.0, 1.0)
        checker.inputs["Color2"].default_value = (0.0, 0.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 16.0

        ntree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
        ntree.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
        ntree.links.new(checker.outputs["Color"], emission.inputs["Color"])
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    mapping = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeMapping")
    return material, mapping


def ensure_mirror_material(name: str = "HWRTPlaneMirrorMaterial"):
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


def ensure_quadrant_image(name: str, quadrant_colors: tuple[tuple[float, float, float], ...], size: int = 8):
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


def ensure_image_emission_material(name: str, coord_output: str):
    image_a = ensure_quadrant_image(
        f"{name}ImageA",
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
    )
    image_b = ensure_quadrant_image(
        f"{name}ImageB",
        (
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.5, 0.0),
            (0.2, 0.2, 1.0),
        ),
    )

    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        ntree = material.node_tree
        ntree.nodes.clear()

        texcoord = ntree.nodes.new("ShaderNodeTexCoord")
        mapping = ntree.nodes.new("ShaderNodeMapping")
        image_node = ntree.nodes.new("ShaderNodeTexImage")
        emission = ntree.nodes.new("ShaderNodeEmission")
        output = ntree.nodes.new("ShaderNodeOutputMaterial")

        image_node.interpolation = "Closest"
        emission.inputs["Strength"].default_value = 16.0

        ntree.links.new(texcoord.outputs[coord_output], mapping.inputs["Vector"])
        ntree.links.new(mapping.outputs["Vector"], image_node.inputs["Vector"])
        ntree.links.new(image_node.outputs["Color"], emission.inputs["Color"])
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    image_node = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage")
    image_node.image = image_a
    return material, image_node, image_a, image_b


def ensure_probe_modifier(cube, name: str = "HWRTPlaneMirrorSubsurf"):
    modifier = cube.modifiers.get(name)
    if modifier is None:
        modifier = cube.modifiers.new(name=name, type="SUBSURF")
        modifier.levels = 2
        modifier.render_levels = 2
    return modifier


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_probe_plane(name: str = "HWRTPlaneMirrorTarget"):
    plane = bpy.data.objects.get(name)
    if plane is None:
        bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
        plane = bpy.context.active_object
        plane.name = name
        plane.data.name = f"{name}Mesh"
    return plane


def setup_probe_layout(scene):
    cube = bpy.data.objects["Cube"]
    camera = scene.camera
    light = bpy.data.objects["Point"]
    plane = ensure_probe_plane()

    for obj in bpy.data.objects:
        if obj.name not in {cube.name, camera.name, light.name, plane.name}:
            obj.hide_viewport = True
            obj.hide_render = True

    plane.hide_viewport = False
    plane.hide_render = False
    plane.location = Vector((0.0, 0.0, 0.0))
    plane.scale = Vector((2.5, 2.5, 2.5))

    cube.hide_viewport = False
    cube.hide_render = False
    cube.location = Vector((0.0, 1.5, 1.0))
    cube.scale = Vector((0.8, 0.8, 0.8))
    cube.rotation_euler = (0.0, 0.0, 0.0)

    camera.location = Vector((0.0, -6.0, 3.0))
    look_at(camera, Vector((0.0, 0.0, 0.7)))

    light.location = Vector((0.0, -3.0, 5.0))
    light.data.energy = 2500.0
    return cube, plane


def run_probe():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_plane_mirror_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    configure_scene(scene, args.taa_samples)
    cube, plane = setup_probe_layout(scene)
    camera = scene.camera

    checker_material, checker_mapping = ensure_checker_emission_material()
    mirror_material = ensure_mirror_material()
    generated_material, generated_image_node, generated_image_a, generated_image_b = ensure_image_emission_material(
        "HWRTPlaneMirrorGenerated", "Generated"
    )
    uv_material, uv_image_node, uv_image_a, uv_image_b = ensure_image_emission_material(
        "HWRTPlaneMirrorUV", "UV"
    )
    modifier = ensure_probe_modifier(cube)
    cube.data.materials.clear()
    plane.data.materials.clear()
    plane.data.materials.append(mirror_material)

    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    def configure_probe_state(*, live_state_a: bool):
        modifier.show_viewport = False
        modifier.show_render = False

        if args.probe == "checker_shift":
            cube.data.materials.clear()
            cube.data.materials.append(checker_material)
            checker_mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
            checker_mapping.inputs["Location"].default_value = (
                0.0 if live_state_a else 0.125,
                0.0,
                0.0,
            )
            checker_material.node_tree.update_tag()
        elif args.probe == "modifier_toggle":
            cube.data.materials.clear()
            cube.data.materials.append(checker_material)
            checker_mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
            checker_mapping.inputs["Location"].default_value = (0.125, 0.0, 0.0)
            modifier_enabled = False if live_state_a else True
            modifier.show_viewport = modifier_enabled
            modifier.show_render = modifier_enabled
            checker_material.node_tree.update_tag()
        elif args.probe == "image_swap_generated":
            cube.data.materials.clear()
            cube.data.materials.append(generated_material)
            generated_image_node.image = generated_image_a if live_state_a else generated_image_b
            generated_material.node_tree.update_tag()
        else:
            cube.data.materials.clear()
            cube.data.materials.append(uv_material)
            uv_image_node.image = uv_image_a if live_state_a else uv_image_b
            uv_material.node_tree.update_tag()

        cube.update_tag()

    def configure_snapshot_state():
        modifier.show_viewport = False
        modifier.show_render = False

        if args.probe in {"checker_shift", "modifier_toggle"}:
            cube.data.materials.clear()
            cube.data.materials.append(checker_material)
            checker_mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
            checker_mapping.inputs["Location"].default_value = (args.checker_location_x, 0.0, 0.0)
            modifier.show_viewport = args.modifier_enabled
            modifier.show_render = args.modifier_enabled
            checker_material.node_tree.update_tag()
        elif args.probe == "image_swap_generated":
            cube.data.materials.clear()
            cube.data.materials.append(generated_material)
            generated_image_node.image = generated_image_a if args.image_state == "A" else generated_image_b
            generated_material.node_tree.update_tag()
        else:
            cube.data.materials.clear()
            cube.data.materials.append(uv_material)
            uv_image_node.image = uv_image_a if args.image_state == "A" else uv_image_b
            uv_material.node_tree.update_tag()

        cube.update_tag()

    if args.mode == "snapshot":
        configure_snapshot_state()
        redraw(args.redraw_iterations)
        capture = capture_viewport(scene, window, area, region, output_dir, args.tag)
        cube_bounds = object_bbox_pixels(scene, camera, cube, capture["width"], capture["height"])
        plane_bounds = object_bbox_pixels(scene, camera, plane, capture["width"], capture["height"])
        print(f"CUBE_BOUNDS={cube_bounds}")
        print(f"PLANE_BOUNDS={plane_bounds}")
        print(f"OUTPUT_DIR={output_dir}")
    else:
        configure_probe_state(live_state_a=True)
        redraw(args.redraw_iterations)
        phase_a = capture_viewport(scene, window, area, region, output_dir, "plane_phase_a")

        cube_bounds = object_bbox_pixels(scene, camera, cube, phase_a["width"], phase_a["height"])
        plane_bounds = object_bbox_pixels(scene, camera, plane, phase_a["width"], phase_a["height"])

        configure_probe_state(live_state_a=False)
        redraw(args.redraw_iterations)
        phase_b = capture_viewport(scene, window, area, region, output_dir, "plane_phase_b")

        cube_diff = crop_abs_diff_mean(
            phase_a["pixels"], phase_b["pixels"], phase_a["width"], phase_a["height"], cube_bounds
        )
        plane_diff = crop_abs_diff_mean(
            phase_a["pixels"], phase_b["pixels"], phase_a["width"], phase_a["height"], plane_bounds
        )
        print(f"CUBE_BOUNDS={cube_bounds}")
        print(f"PLANE_BOUNDS={plane_bounds}")
        print(f"PLANE_MIRROR_{args.probe.upper()}_DIRECT_CUBE_DIFF_MEAN={cube_diff:.6f}")
        print(f"PLANE_MIRROR_{args.probe.upper()}_REFLECTION_DIFF_MEAN={plane_diff:.6f}")
        print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()

    if not args.hold_open:
        def _quit_blender():
            bpy.ops.wm.quit_blender()
            return None

        bpy.app.timers.register(_quit_blender, first_interval=0.1)


run_probe()
