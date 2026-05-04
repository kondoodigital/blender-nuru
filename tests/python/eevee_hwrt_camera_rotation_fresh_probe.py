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
        description="Compare same-session camera-rotation return renders against a fresh identical rerun."
    )
    parser.add_argument(
        "--mode",
        choices=("live_vs_fresh", "snapshot"),
        default="live_vs_fresh",
        help="Run the same-session rotation probe or render a single configured snapshot.",
    )
    parser.add_argument(
        "--state",
        choices=("A", "B"),
        default="A",
        help="Camera state for snapshot mode.",
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    parser.add_argument("--emissive-strength", type=float, default=80.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="snapshot")
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


def configure_scene(scene, args):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.use_hardware_raytracing_gi = True
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "OFF"
    eevee.use_hardware_raytracing_environment = False
    eevee.use_hardware_raytracing_shadows = False
    eevee.taa_render_samples = max(1, args.samples)
    eevee.taa_samples = max(1, args.samples)

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.use_denoise = False
    ray_tracing.denoise_spatial = False
    ray_tracing.denoise_temporal = False
    ray_tracing.denoise_bilateral = False


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTCameraRotationWorld")
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


def make_emission_material(name: str, strength: float):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()
    emission = ntree.nodes.new("ShaderNodeEmission")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Color"].default_value = (1.0, 0.84, 0.60, 1.0)
    emission.inputs["Strength"].default_value = strength
    ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def create_plane(name: str, location, scale, material):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = Vector(scale)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def create_room(scene, emissive_strength: float):
    wall = make_diffuse_material("CameraRotationWallMat", (0.90, 0.90, 0.90, 1.0))
    floor_mat = make_diffuse_material("CameraRotationFloorMat", (0.82, 0.82, 0.82, 1.0))
    blocker_mat = make_diffuse_material("CameraRotationBlockerMat", (0.65, 0.65, 0.65, 1.0))
    emitter_mat = make_emission_material("CameraRotationEmitterMat", emissive_strength)

    create_plane("ProbeFloor", (0.0, 0.0, 0.0), (2.2, 2.8, 1.0), floor_mat)
    ceiling = create_plane("ProbeCeiling", (0.0, 0.0, 3.2), (2.2, 2.8, 1.0), wall)
    ceiling.rotation_euler.x = math.pi

    back = create_plane("ProbeBackWall", (0.0, 2.8, 1.6), (2.2, 1.6, 1.0), wall)
    back.rotation_euler.x = math.pi * 0.5

    left = create_plane("ProbeLeftWall", (-2.2, 0.0, 1.6), (2.8, 1.6, 1.0), wall)
    left.rotation_euler.y = math.pi * 0.5

    right = create_plane("ProbeRightWall", (2.2, 0.0, 1.6), (2.8, 1.6, 1.0), wall)
    right.rotation_euler.y = -math.pi * 0.5

    blocker = create_plane("ProbeBaffle", (0.0, 0.35, 0.9), (1.1, 0.9, 1.0), blocker_mat)
    blocker.rotation_euler.x = math.pi * 0.5

    emitter = create_plane("ProbeEmitter", (1.95, 1.05, 2.25), (0.55, 0.55, 1.0), emitter_mat)
    emitter.rotation_euler = Vector((0.0, math.radians(90.0), math.radians(180.0)))

    bpy.ops.object.camera_add(location=(0.0, -5.8, 1.65))
    camera = bpy.context.active_object
    patch = Vector((0.0, -0.95, 0.0))
    scene.camera = camera
    return camera, patch


def apply_camera_state(camera, patch: Vector, state: str):
    if state == "A":
        camera.location = Vector((0.0, -5.8, 1.65))
        look_at(camera, Vector((0.0, -0.5, 0.45)))
    elif state == "B":
        camera.location = Vector((2.35, -4.6, 1.85))
        look_at(camera, Vector((1.35, 1.05, 1.9)))
    else:
        raise RuntimeError(f"Unsupported camera state: {state}")
    camera.update_tag()


def render_to_image(scene, output_dir: Path, tag: str):
    scene.render.filepath = str(output_dir / f"{tag}.png")
    bpy.ops.render.render(write_still=True)
    return load_image_pixels(Path(scene.render.filepath))


def load_image_pixels(path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return {"path": path, "width": width, "height": height, "pixels": pixels}


def projected_crop_bounds(scene, camera, point: Vector, width: int, height: int, radius_px: int):
    co = world_to_camera_view(scene, camera, point)
    center_x = int(round(co.x * width))
    center_y = int(round(co.y * height))
    min_x = max(0, center_x - radius_px)
    max_x = min(width - 1, center_x + radius_px)
    min_y = max(0, center_y - radius_px)
    max_y = min(height - 1, center_y + radius_px)
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


def diff_bounds(pixels_a, pixels_b, width: int, height: int, fallback_bounds, threshold: float = 0.015):
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for y in range(height):
        row = y * width * 4
        for x in range(width):
            base = row + x * 4
            diff = max(
                abs(pixels_a[base + 0] - pixels_b[base + 0]),
                abs(pixels_a[base + 1] - pixels_b[base + 1]),
                abs(pixels_a[base + 2] - pixels_b[base + 2]),
            )
            if diff > threshold:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x == -1:
        return fallback_bounds
    margin = 12
    fb_min_x, fb_min_y, fb_max_x, fb_max_y = fallback_bounds
    return (
        max(fb_min_x, min_x - margin),
        max(fb_min_y, min_y - margin),
        min(fb_max_x, max_x + margin),
        min(fb_max_y, max_y + margin),
    )


def spawn_fresh_snapshot(args, output_dir: Path, tag: str, state: str):
    cmd = [
        bpy.app.binary_path,
        "-b",
        "--factory-startup",
        "-P",
        __file__,
        "--",
        "--mode",
        "snapshot",
        "--state",
        state,
        "--samples",
        str(args.samples),
        "--resolution-x",
        str(args.resolution_x),
        "--resolution-y",
        str(args.resolution_y),
        "--emissive-strength",
        str(args.emissive_strength),
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
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_camera_rotation_fresh_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args)
    ensure_world(scene)
    camera, patch = create_room(scene, args.emissive_strength)

    if args.mode == "snapshot":
        apply_camera_state(camera, patch, args.state)
        snapshot = render_to_image(scene, output_dir, args.tag)
        patch_bounds = projected_crop_bounds(scene, camera, patch, snapshot["width"], snapshot["height"], 42)
        print(f"PATCH_BOUNDS={patch_bounds}")
        print(f"OUTPUT_DIR={output_dir}")
        if cleanup_dir is not None:
            cleanup_dir.cleanup()
        return

    apply_camera_state(camera, patch, "A")
    live_a = render_to_image(scene, output_dir, "live_a")

    apply_camera_state(camera, patch, "B")
    live_b = render_to_image(scene, output_dir, "live_b")

    apply_camera_state(camera, patch, "A")
    live_return_a = render_to_image(scene, output_dir, "live_return_a")

    patch_bounds = projected_crop_bounds(scene, camera, patch, live_a["width"], live_a["height"], 42)
    changed_bounds = diff_bounds(
        live_a["pixels"], live_b["pixels"], live_a["width"], live_a["height"], (0, 0, live_a["width"] - 1, live_a["height"] - 1)
    )

    spawn_fresh_snapshot(args, output_dir, "fresh_a", "A")
    fresh_a = load_image_pixels(output_dir / "fresh_a.png")

    rotate_diff = crop_abs_diff_mean(
        live_a["pixels"], live_b["pixels"], live_a["width"], live_a["height"], changed_bounds
    )
    return_fresh_diff = crop_abs_diff_mean(
        live_return_a["pixels"], fresh_a["pixels"], live_return_a["width"], live_return_a["height"], patch_bounds
    )
    return_original_diff = crop_abs_diff_mean(
        live_return_a["pixels"], live_a["pixels"], live_return_a["width"], live_return_a["height"], patch_bounds
    )

    print(f"PATCH_BOUNDS={patch_bounds}")
    print(f"CHANGED_BOUNDS={changed_bounds}")
    print(f"CAMERA_ROTATION_FRAME_DIFF_MEAN={rotate_diff:.6f}")
    print(f"CAMERA_ROTATION_RETURN_VS_FRESH_DIFF_MEAN={return_fresh_diff:.6f}")
    print(f"CAMERA_ROTATION_RETURN_VS_INITIAL_DIFF_MEAN={return_original_diff:.6f}")
    print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


if __name__ == "__main__":
    main()
