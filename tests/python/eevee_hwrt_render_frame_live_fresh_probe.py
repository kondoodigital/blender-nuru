#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
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
        description="Compare same-session render-frame edits against a fresh identical render."
    )
    parser.add_argument(
        "--mode",
        choices=("live_vs_fresh", "snapshot"),
        default="live_vs_fresh",
        help="Run the same-session edit probe or render a single configured snapshot.",
    )
    parser.add_argument(
        "--probe",
        choices=("checker_shift", "modifier_toggle", "image_swap_generated", "image_swap_uv"),
        default="image_swap_uv",
        help="Which reflected change to exercise in the planar mirror setup.",
    )
    parser.add_argument(
        "--state",
        choices=("A", "B"),
        default="B",
        help="Target state for snapshot mode.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Render samples to use for each frame render.",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to scene render resolution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to keep rendered outputs. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="snapshot",
        help="Output tag for snapshot mode.",
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


def configure_scene(scene, samples: int, resolution_scale: float):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_percentage = max(1, int(round(100 * resolution_scale)))

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.hardware_raytracing_reflection_mode = "FULL"
    eevee.hardware_raytracing_refraction_mode = "FULL"
    eevee.use_hardware_raytracing_environment = True
    eevee.use_hardware_raytracing_shadows = True
    eevee.taa_render_samples = max(1, samples)
    eevee.taa_samples = max(1, samples)

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.screen_trace_quality = 1.0
    ray_tracing.screen_trace_thickness = 1.0


def ensure_checker_emission_material(name: str = "HWRTRenderLiveFreshChecker"):
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
    return material, image_node, image_a, image_b


def ensure_mirror_material(name: str = "HWRTRenderLiveFreshMirror"):
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


def ensure_probe_modifier(cube, name: str = "HWRTRenderLiveFreshSubsurf"):
    modifier = cube.modifiers.get(name)
    if modifier is None:
        modifier = cube.modifiers.new(name=name, type="SUBSURF")
        modifier.levels = 2
        modifier.render_levels = 2
    return modifier


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_probe_plane(name: str = "HWRTRenderLiveFreshPlane"):
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


def render_to_image(scene, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return load_image_pixels(path)


def load_image_pixels(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return {"path": path, "width": width, "height": height, "pixels": pixels}


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
            tile_bounds = (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
            best = max(best, crop_abs_diff_mean(pixels_a, pixels_b, width, height, tile_bounds))

    return best


def diff_bounds(pixels_a, pixels_b, width: int, height: int, search_bounds, threshold: float = 0.01):
    min_x, min_y, max_x, max_y = search_bounds
    hit_min_x = width
    hit_min_y = height
    hit_max_x = -1
    hit_max_y = -1

    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            channel_diff = max(
                abs(pixels_a[base + 0] - pixels_b[base + 0]),
                abs(pixels_a[base + 1] - pixels_b[base + 1]),
                abs(pixels_a[base + 2] - pixels_b[base + 2]),
            )
            if channel_diff > threshold:
                hit_min_x = min(hit_min_x, x)
                hit_min_y = min(hit_min_y, y)
                hit_max_x = max(hit_max_x, x)
                hit_max_y = max(hit_max_y, y)

    if hit_max_x < hit_min_x or hit_max_y < hit_min_y:
        return search_bounds

    margin = 6
    return (
        max(min_x, hit_min_x - margin),
        max(min_y, hit_min_y - margin),
        min(max_x, hit_max_x + margin),
        min(max_y, hit_max_y + margin),
    )


def configure_probe_state(
    probe: str,
    state: str,
    cube,
    modifier,
    checker_material,
    checker_mapping,
    generated_material,
    generated_image_node,
    generated_image_a,
    generated_image_b,
    uv_material,
    uv_image_node,
    uv_image_a,
    uv_image_b,
):
    modifier.show_viewport = False
    modifier.show_render = False

    if probe == "checker_shift":
        cube.data.materials.clear()
        cube.data.materials.append(checker_material)
        checker_mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
        checker_mapping.inputs["Location"].default_value = (0.0 if state == "A" else 0.125, 0.0, 0.0)
        checker_material.node_tree.update_tag()
    elif probe == "modifier_toggle":
        cube.data.materials.clear()
        cube.data.materials.append(checker_material)
        checker_mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
        checker_mapping.inputs["Location"].default_value = (0.125, 0.0, 0.0)
        modifier_enabled = state == "B"
        modifier.show_viewport = modifier_enabled
        modifier.show_render = modifier_enabled
        checker_material.node_tree.update_tag()
    elif probe == "image_swap_generated":
        cube.data.materials.clear()
        cube.data.materials.append(generated_material)
        generated_image_node.image = generated_image_a if state == "A" else generated_image_b
        generated_material.node_tree.update_tag()
    else:
        cube.data.materials.clear()
        cube.data.materials.append(uv_material)
        uv_image_node.image = uv_image_a if state == "A" else uv_image_b
        uv_material.node_tree.update_tag()

    cube.update_tag()


def spawn_fresh_snapshot(args, output_dir: Path, tag: str):
    cmd = [
        bpy.app.binary_path,
        "-b",
        bpy.data.filepath,
        "-P",
        __file__,
        "--",
        "--mode",
        "snapshot",
        "--probe",
        args.probe,
        "--state",
        "B",
        "--samples",
        str(args.samples),
        "--resolution-scale",
        str(args.resolution_scale),
        "--output-dir",
        str(output_dir),
        "--tag",
        tag,
    ]
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_render_live_fresh_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    configure_scene(scene, args.samples, args.resolution_scale)

    cube, plane = setup_probe_layout(scene)
    camera = scene.camera

    checker_material, checker_mapping = ensure_checker_emission_material()
    generated_material, generated_image_node, generated_image_a, generated_image_b = ensure_image_emission_material(
        "HWRTRenderLiveFreshGenerated", "Generated"
    )
    uv_material, uv_image_node, uv_image_a, uv_image_b = ensure_image_emission_material(
        "HWRTRenderLiveFreshUV", "UV"
    )
    modifier = ensure_probe_modifier(cube)
    mirror_material = ensure_mirror_material()
    plane.data.materials.clear()
    plane.data.materials.append(mirror_material)

    if args.mode == "snapshot":
        configure_probe_state(
            args.probe,
            args.state,
            cube,
            modifier,
            checker_material,
            checker_mapping,
            generated_material,
            generated_image_node,
            generated_image_a,
            generated_image_b,
            uv_material,
            uv_image_node,
            uv_image_a,
            uv_image_b,
        )
        snapshot = render_to_image(scene, output_dir, args.tag)
        plane_bounds = object_bbox_pixels(scene, camera, plane, snapshot["width"], snapshot["height"])
        cube_bounds = object_bbox_pixels(scene, camera, cube, snapshot["width"], snapshot["height"])
        print(f"PLANE_BOUNDS={plane_bounds}")
        print(f"CUBE_BOUNDS={cube_bounds}")
        print(f"OUTPUT_DIR={output_dir}")
        if cleanup_dir is not None:
            cleanup_dir.cleanup()
        return

    configure_probe_state(
        args.probe,
        "A",
        cube,
        modifier,
        checker_material,
        checker_mapping,
        generated_material,
        generated_image_node,
        generated_image_a,
        generated_image_b,
        uv_material,
        uv_image_node,
        uv_image_a,
        uv_image_b,
    )
    live_a = render_to_image(scene, output_dir, "live_state_a")

    configure_probe_state(
        args.probe,
        "B",
        cube,
        modifier,
        checker_material,
        checker_mapping,
        generated_material,
        generated_image_node,
        generated_image_a,
        generated_image_b,
        uv_material,
        uv_image_node,
        uv_image_a,
        uv_image_b,
    )
    live_b = render_to_image(scene, output_dir, "live_state_b")

    plane_bounds = object_bbox_pixels(scene, camera, plane, live_a["width"], live_a["height"])
    cube_bounds = object_bbox_pixels(scene, camera, cube, live_a["width"], live_a["height"])
    changed_bounds = diff_bounds(
        live_a["pixels"], live_b["pixels"], live_a["width"], live_a["height"], plane_bounds
    )

    spawn_fresh_snapshot(args, output_dir, "fresh_state_b")
    fresh_b = load_image_pixels(output_dir / "fresh_state_b.png")

    direct_diff = crop_abs_diff_mean(
        live_a["pixels"], live_b["pixels"], live_a["width"], live_a["height"], cube_bounds
    )
    reflection_diff = crop_abs_diff_mean(
        live_a["pixels"], live_b["pixels"], live_a["width"], live_a["height"], changed_bounds
    )
    live_fresh_diff = crop_abs_diff_mean(
        live_b["pixels"], fresh_b["pixels"], live_b["width"], live_b["height"], changed_bounds
    )
    live_fresh_tile_diff = crop_max_tile_abs_diff_mean(
        live_b["pixels"], fresh_b["pixels"], live_b["width"], live_b["height"], changed_bounds
    )

    print(f"PLANE_BOUNDS={plane_bounds}")
    print(f"CUBE_BOUNDS={cube_bounds}")
    print(f"CHANGED_BOUNDS={changed_bounds}")
    print(f"LIVE_EDIT_DIRECT_DIFF_MEAN={direct_diff:.6f}")
    print(f"LIVE_EDIT_REFLECTION_DIFF_MEAN={reflection_diff:.6f}")
    print(f"LIVE_VS_FRESH_REFLECTION_DIFF_MEAN={live_fresh_diff:.6f}")
    print(f"LIVE_VS_FRESH_REFLECTION_MAX_TILE_DIFF_MEAN={live_fresh_tile_diff:.6f}")
    print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
