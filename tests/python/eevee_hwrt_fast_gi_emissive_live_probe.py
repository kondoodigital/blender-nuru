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
        description="Warm Fast GI, edit an emissive material, and let the next sparse refresh settle."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--redraw-iterations", type=int, default=16)
    parser.add_argument("--strength-scale", type=float, default=4.0)
    parser.add_argument("--taa-samples", type=int, default=1)
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
from mathutils import Vector


def configure_scene(scene, taa_samples: int):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_method = "HARDWARE"
    scene.eevee.hardware_raytracing_reflection_mode = "OFF"
    scene.eevee.hardware_raytracing_refraction_mode = "OFF"
    scene.eevee.use_hardware_raytracing_environment = False
    scene.eevee.use_hardware_raytracing_shadows = False
    scene.eevee.use_hardware_raytracing_gi = True
    scene.eevee.use_hardware_raytracing_caustics = False
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


def redraw(iterations: int):
    for _ in range(iterations):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def capture_viewport(window, area, region, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=str(path))
    return path


def clear_scene(scene):
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTFastGIEmissiveLiveWorld")
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
    emission.inputs["Color"].default_value = (1.0, 0.85, 0.55, 1.0)
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


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_scene_layout(scene):
    clear_scene(scene)
    ensure_world(scene)

    create_plane(
        "ProbeFloor",
        (0.0, 0.0, 0.0),
        (3.0, 3.0, 3.0),
        make_diffuse_material("ProbeFloorMat", (0.9, 0.9, 0.9, 1.0)),
    )
    back = create_plane(
        "ProbeBackWall",
        (0.0, 2.5, 1.8),
        (3.0, 1.8, 1.0),
        make_diffuse_material("ProbeBackWallMat", (0.9, 0.9, 0.9, 1.0)),
    )
    back.rotation_euler.x = 1.57079632679

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 1.6, 1.8))
    emitter = bpy.context.active_object
    emitter.name = "ProbeEmitter"
    emitter.scale = Vector((0.45, 0.45, 0.45))
    emitter.rotation_euler.x = 1.57079632679
    emitter.data.materials.clear()
    emitter.data.materials.append(make_emission_material("ProbeEmitterMat", 1.0))

    bpy.ops.object.camera_add(location=(0.0, -5.5, 2.6))
    camera = bpy.context.active_object
    look_at(camera, Vector((0.0, 0.6, 0.0)))
    scene.camera = camera

    return emitter


def set_emitter_strength(emitter, strength: float):
    material = emitter.data.materials[0]
    emission = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeEmission")
    emission.inputs["Strength"].default_value = strength
    material.node_tree.update_tag()
    emitter.update_tag()


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_fast_gi_emissive_live_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    configure_scene(scene, args.taa_samples)
    emitter = create_scene_layout(scene)
    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    # Warm one traced field before the emissive edit so the invalidation stats reflect the edit
    # instead of startup/bootstrap state.
    redraw(args.redraw_iterations)
    capture_viewport(window, area, region, output_dir, "warm_start")

    material = emitter.data.materials[0]
    emission = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeEmission")
    new_strength = max(1.0e-4, emission.inputs["Strength"].default_value * args.strength_scale)
    set_emitter_strength(emitter, new_strength)
    bpy.context.view_layer.update()
    redraw(args.redraw_iterations)
    capture_viewport(window, area, region, output_dir, "after_emissive_edit")

    print(f"OUTPUT_DIR={output_dir}")
    bpy.ops.wm.quit_blender()


main()
