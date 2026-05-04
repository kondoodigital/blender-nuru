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
        description="Probe live Eevee Hardware RT mirror updates on the barbershop scene."
    )
    parser.add_argument(
        "--mode",
        choices=("live_probe", "snapshot"),
        default="live_probe",
        help="Either run an in-session mirror update probe or capture a single configured snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to keep viewport captures. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--cube-visible",
        action="store_true",
        help="Show the temporary emissive cube in snapshot mode.",
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
        help="Number of redraw iterations to wait before capturing each viewport state.",
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
    scene.render.filepath = str(path)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.render.opengl(write_still=True, view_context=True)
    width, height, pixels = load_pixels(path)
    return {"path": path, "width": width, "height": height, "pixels": pixels}


def ensure_probe_material(name: str = "HWRTBarbershopProbeEmission"):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        ntree = material.node_tree
        ntree.nodes.clear()
        emission = ntree.nodes.new("ShaderNodeEmission")
        output = ntree.nodes.new("ShaderNodeOutputMaterial")
        emission.inputs["Color"].default_value = (0.05, 1.0, 0.05, 1.0)
        emission.inputs["Strength"].default_value = 50.0
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def visible_mirror_candidate(scene, camera):
    width = scene.render.resolution_x * scene.render.resolution_percentage // 100
    height = scene.render.resolution_y * scene.render.resolution_percentage // 100
    candidates = []
    for obj in bpy.data.objects:
        name = obj.name.lower()
        if "mirror" not in name:
            continue
        try:
            bounds = object_bbox_pixels(scene, camera, obj, width, height, margin_px=0)
        except Exception:
            continue
        min_x, min_y, max_x, max_y = bounds
        area = max(0, max_x - min_x + 1) * max(0, max_y - min_y + 1)
        if area > 0:
            candidates.append((area, obj))
    if not candidates:
        raise RuntimeError("No visible mirror object found in current camera view.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def ensure_probe_cube(mirror_obj, camera, name: str = "HWRTBarbershopProbeCube"):
    probe = bpy.data.objects.get(name)
    if probe is None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        probe = bpy.context.active_object
        probe.name = name
        probe.data.name = f"{name}Mesh"
    probe.hide_select = True
    probe.data.materials.clear()
    probe.data.materials.append(ensure_probe_material())

    local_corners = [Vector(corner) for corner in mirror_obj.bound_box]
    local_dims = [max(c[i] for c in local_corners) - min(c[i] for c in local_corners) for i in range(3)]
    normal_axis = min(range(3), key=lambda axis: local_dims[axis])
    tangent_axis = max(range(3), key=lambda axis: local_dims[axis])
    world_basis = mirror_obj.matrix_world.to_3x3()

    center = sum((mirror_obj.matrix_world @ corner for corner in local_corners), Vector()) / 8.0
    normal = world_basis.col[normal_axis].normalized()
    tangent = world_basis.col[tangent_axis].normalized()
    if normal.dot((camera.location - center).normalized()) < 0.0:
        normal.negate()

    mirror_span = max(local_dims)
    probe.scale = Vector((mirror_span * 0.08, mirror_span * 0.08, mirror_span * 0.08))
    probe.location = center + normal * (mirror_span * 0.20) + tangent * (mirror_span * 0.18)
    return probe


def run_probe():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_barbershop_live_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    camera = scene.camera
    configure_scene(scene)

    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    mirror = visible_mirror_candidate(scene, camera)
    probe = ensure_probe_cube(mirror, camera)
    probe.hide_viewport = True
    probe.hide_render = True

    redraw(args.redraw_iterations)

    if args.mode == "snapshot":
        probe.hide_viewport = not args.cube_visible
        probe.hide_render = not args.cube_visible
        probe.update_tag()
        redraw(args.redraw_iterations)
        capture = capture_viewport(scene, window, area, region, output_dir, args.tag)
        mirror_bounds = object_bbox_pixels(scene, camera, mirror, capture["width"], capture["height"])
        print(f"MIRROR_OBJECT={mirror.name}")
        print(f"MIRROR_BOUNDS={mirror_bounds}")
        print(f"OUTPUT_DIR={output_dir}")
    else:
        before = capture_viewport(scene, window, area, region, output_dir, "mirror_before")
        mirror_bounds = object_bbox_pixels(scene, camera, mirror, before["width"], before["height"])

        probe.hide_viewport = False
        probe.hide_render = False
        probe.update_tag()
        redraw(args.redraw_iterations)
        after = capture_viewport(scene, window, area, region, output_dir, "mirror_after")

        mirror_diff = crop_abs_diff_mean(
            before["pixels"], after["pixels"], before["width"], before["height"], mirror_bounds
        )
        print(f"MIRROR_OBJECT={mirror.name}")
        print(f"MIRROR_BOUNDS={mirror_bounds}")
        print(f"MIRROR_DIFF_MEAN={mirror_diff:.6f}")
        print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()

    def _quit_blender():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(_quit_blender, first_interval=0.1)


run_probe()
