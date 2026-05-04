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
        description="Compare same-session post-edit Fast GI renders against a fresh identical B-state rerun."
    )
    parser.add_argument(
        "--mode",
        choices=("live_vs_fresh", "snapshot"),
        default="live_vs_fresh",
        help="Run the same-session response probe or render a single configured snapshot.",
    )
    parser.add_argument("--probe", choices=("light", "geometry"), default="light")
    parser.add_argument("--state", choices=("A", "B"), default="A")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
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
    eevee.use_hardware_raytracing_caustics = False
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
        world = bpy.data.worlds.new("HWRTAnimationResponseWorld")
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
    emission.inputs["Color"].default_value = (1.0, 0.86, 0.62, 1.0)
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


def create_cube(name: str, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = Vector(scale)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def create_scene_layout(scene):
    ensure_world(scene)

    wall_mat = make_diffuse_material("AnimationResponseWallMat", (0.88, 0.88, 0.88, 1.0))
    floor_mat = make_diffuse_material("AnimationResponseFloorMat", (0.84, 0.84, 0.84, 1.0))
    blocker_mat = make_diffuse_material("AnimationResponseBlockerMat", (0.65, 0.65, 0.65, 1.0))
    emitter_mat = make_emission_material("AnimationResponseEmitterMat", 24.0)

    create_plane("ProbeFloor", (0.0, 0.0, 0.0), (2.4, 2.9, 1.0), floor_mat)
    ceiling = create_plane("ProbeCeiling", (0.0, 0.0, 3.1), (2.4, 2.9, 1.0), wall_mat)
    ceiling.rotation_euler.x = math.pi

    back = create_plane("ProbeBackWall", (0.0, 2.9, 1.55), (2.4, 1.55, 1.0), wall_mat)
    back.rotation_euler.x = math.pi * 0.5

    left = create_plane("ProbeLeftWall", (-2.4, 0.0, 1.55), (2.9, 1.55, 1.0), wall_mat)
    left.rotation_euler.y = math.pi * 0.5

    right = create_plane("ProbeRightWall", (2.4, 0.0, 1.55), (2.9, 1.55, 1.0), wall_mat)
    right.rotation_euler.y = -math.pi * 0.5

    emitter = create_plane("ProbeEmitter", (0.0, 1.45, 2.1), (0.55, 0.55, 1.0), emitter_mat)
    emitter.rotation_euler.x = math.pi * 0.5

    blocker = create_cube("ProbeGeometryBlocker", (0.0, 0.35, 1.05), (0.85, 0.08, 0.95), blocker_mat)

    bpy.ops.object.camera_add(location=(0.0, -5.7, 1.75))
    camera = bpy.context.active_object
    patch = Vector((0.0, -0.95, 0.0))
    look_at(camera, Vector((0.0, -0.35, 0.35)))
    scene.camera = camera
    return emitter, blocker, patch


def set_emitter_strength(emitter, strength: float):
    material = emitter.data.materials[0]
    emission = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeEmission")
    emission.inputs["Strength"].default_value = strength
    material.node_tree.update_tag()


def configure_probe_state(probe: str, state: str, emitter, blocker):
    if probe == "light":
        set_emitter_strength(emitter, 24.0 if state == "A" else 96.0)
        blocker.hide_render = False
        blocker.hide_viewport = False
    elif probe == "geometry":
        set_emitter_strength(emitter, 60.0)
        blocker.hide_render = state == "A"
        blocker.hide_viewport = state == "A"
    else:
        raise RuntimeError(f"Unsupported probe: {probe}")
    emitter.update_tag()
    blocker.update_tag()
    bpy.context.view_layer.update()


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


def spawn_fresh_snapshot(args, output_dir: Path, tag: str):
    cmd = [
        bpy.app.binary_path,
        "-b",
        "--factory-startup",
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
        "--resolution-x",
        str(args.resolution_x),
        "--resolution-y",
        str(args.resolution_y),
        "--output-dir",
        str(output_dir),
        "--tag",
        tag,
    ]
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)


def metric_prefix(probe: str):
    return "LIGHT_ANIMATION" if probe == "light" else "GEOMETRY_ANIMATION"


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_animation_response_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene, args)
    emitter, blocker, patch = create_scene_layout(scene)
    camera = scene.camera

    if args.mode == "snapshot":
        configure_probe_state(args.probe, args.state, emitter, blocker)
        snapshot = render_to_image(scene, output_dir, args.tag)
        patch_bounds = projected_crop_bounds(scene, camera, patch, snapshot["width"], snapshot["height"], 46)
        print(f"PATCH_BOUNDS={patch_bounds}")
        print(f"OUTPUT_DIR={output_dir}")
        if cleanup_dir is not None:
            cleanup_dir.cleanup()
        return

    configure_probe_state(args.probe, "A", emitter, blocker)
    live_a = render_to_image(scene, output_dir, f"{args.probe}_live_a")

    configure_probe_state(args.probe, "B", emitter, blocker)
    live_b_frame1 = render_to_image(scene, output_dir, f"{args.probe}_live_b_frame1")
    live_b_frame2 = render_to_image(scene, output_dir, f"{args.probe}_live_b_frame2")

    spawn_fresh_snapshot(args, output_dir, f"{args.probe}_fresh_b")
    fresh_b = load_image_pixels(output_dir / f"{args.probe}_fresh_b.png")

    patch_bounds = projected_crop_bounds(scene, camera, patch, live_a["width"], live_a["height"], 46)
    prefix = metric_prefix(args.probe)

    state_change_diff = crop_abs_diff_mean(
        live_a["pixels"], live_b_frame1["pixels"], live_a["width"], live_a["height"], patch_bounds
    )
    frame1_fresh_diff = crop_abs_diff_mean(
        live_b_frame1["pixels"], fresh_b["pixels"], live_b_frame1["width"], live_b_frame1["height"], patch_bounds
    )
    frame2_fresh_diff = crop_abs_diff_mean(
        live_b_frame2["pixels"], fresh_b["pixels"], live_b_frame2["width"], live_b_frame2["height"], patch_bounds
    )

    print(f"PATCH_BOUNDS={patch_bounds}")
    print(f"{prefix}_STATE_CHANGE_DIFF_MEAN={state_change_diff:.6f}")
    print(f"{prefix}_FRAME1_VS_FRESH_DIFF_MEAN={frame1_fresh_diff:.6f}")
    print(f"{prefix}_FRAME2_VS_FRESH_DIFF_MEAN={frame2_fresh_diff:.6f}")
    print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


if __name__ == "__main__":
    main()
