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
        description="Probe live image-texture reflection updates on scenes/test.blend."
    )
    parser.add_argument(
        "--mode",
        choices=("live_probe", "snapshot"),
        default="live_probe",
        help="Either run the in-session texture swap probe or capture a single configured snapshot.",
    )
    parser.add_argument(
        "--coord-output",
        choices=("Generated", "UV"),
        default="Generated",
        help="Coordinate source used for the cube image texture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to keep viewport captures. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--image-state",
        choices=("A", "B"),
        default="A",
        help="Which image to bind in snapshot mode.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="snapshot",
        help="Output tag for snapshot mode.",
    )
    parser.add_argument(
        "--redraw-iterations",
        type=int,
        default=8,
        help="Number of redraw iterations to wait after each edit before capture.",
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


def configure_scene(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = "HARDWARE"
    scene.eevee.hardware_raytracing_reflection_mode = "FULL"
    scene.eevee.hardware_raytracing_refraction_mode = "FULL"
    scene.eevee.use_hardware_raytracing_environment = True
    scene.eevee.use_hardware_raytracing_shadows = True
    scene.eevee.taa_samples = 1
    scene.eevee.taa_render_samples = 1


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


def ensure_mirror_material(name: str = "HWRTLiveTextureMirror"):
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

        texcoord.location = (-800, 0)
        mapping.location = (-600, 0)
        image_node.location = (-350, 0)
        emission.location = (-100, 0)
        output.location = (150, 0)

        image_node.interpolation = "Closest"
        emission.inputs["Strength"].default_value = 16.0

        ntree.links.new(texcoord.outputs[coord_output], mapping.inputs["Vector"])
        ntree.links.new(mapping.outputs["Vector"], image_node.inputs["Vector"])
        ntree.links.new(image_node.outputs["Color"], emission.inputs["Color"])
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    image_node = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage")
    image_node.image = image_a
    return material, image_node, image_a, image_b


def run_probe():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_live_texture_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    camera = scene.camera
    cube = bpy.data.objects["Cube"]
    sphere = bpy.data.objects["Sphere"]
    original_cube_material = cube.material_slots[0].material
    original_sphere_material = sphere.material_slots[0].material

    material, image_node, image_a, image_b = ensure_image_emission_material(
        name=f"HWRTLiveTexture{args.coord_output}",
        coord_output=args.coord_output,
    )
    mirror_material = ensure_mirror_material()

    configure_scene(scene)
    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    cube.material_slots[0].material = material
    sphere.material_slots[0].material = mirror_material

    if args.mode == "snapshot":
        image_node.image = image_a if args.image_state == "A" else image_b
        material.node_tree.update_tag()
        cube.update_tag()
        redraw(args.redraw_iterations)
        capture = capture_viewport(scene, window, area, region, output_dir, args.tag)
        sphere_bounds = object_bbox_pixels(scene, camera, sphere, capture["width"], capture["height"])
        cube_bounds = object_bbox_pixels(scene, camera, cube, capture["width"], capture["height"])
        print(f"CUBE_BOUNDS={cube_bounds}")
        print(f"SPHERE_BOUNDS={sphere_bounds}")
        print(f"OUTPUT_DIR={output_dir}")
    else:
        image_node.image = image_a
        redraw(args.redraw_iterations)
        phase_a = capture_viewport(scene, window, area, region, output_dir, "image_a")
        sphere_bounds = object_bbox_pixels(scene, camera, sphere, phase_a["width"], phase_a["height"])
        cube_bounds = object_bbox_pixels(scene, camera, cube, phase_a["width"], phase_a["height"])

        image_node.image = image_b
        material.node_tree.update_tag()
        cube.update_tag()
        redraw(args.redraw_iterations)
        phase_b = capture_viewport(scene, window, area, region, output_dir, "image_b")

        cube_diff = crop_abs_diff_mean(
            phase_a["pixels"], phase_b["pixels"], phase_a["width"], phase_a["height"], cube_bounds
        )
        sphere_diff = crop_abs_diff_mean(
            phase_a["pixels"], phase_b["pixels"], phase_a["width"], phase_a["height"], sphere_bounds
        )
        print(f"CUBE_BOUNDS={cube_bounds}")
        print(f"SPHERE_BOUNDS={sphere_bounds}")
        print(f"IMAGE_DIRECT_CUBE_DIFF_MEAN={cube_diff:.6f}")
        print(f"IMAGE_REFLECTION_SPHERE_DIFF_MEAN={sphere_diff:.6f}")
        print(f"OUTPUT_DIR={output_dir}")

    cube.material_slots[0].material = original_cube_material
    sphere.material_slots[0].material = original_sphere_material

    if cleanup_dir is not None:
        cleanup_dir.cleanup()

    def _quit_blender():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(_quit_blender, first_interval=0.1)


run_probe()
