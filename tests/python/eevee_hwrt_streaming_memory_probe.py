#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


STATS_PREFIX = "EEVEE HWRT FastGI stats "
PAIR_RE = re.compile(r"(\w+)=([^\s]+)")


def _inside_blender():
    try:
        import bpy  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate large-scene Fast GI memory fitting by comparing default and tight budgets."
    )
    if _inside_blender():
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []
        parser.add_argument("--inner", action="store_true")
        parser.add_argument("--output-dir", type=Path, default=None)
    else:
        parser.add_argument("--blender-bin", type=Path, required=True)
        parser.add_argument("--output-json", type=Path, default=None)
        parser.add_argument("--default-budget-mb", type=int, default=2)
        parser.add_argument("--tight-budget-mb", type=int, default=1)
    return parser.parse_args(argv if _inside_blender() else None)


def parse_stats_line(text: str):
    parsed = {}
    for key, raw_value in PAIR_RE.findall(text):
        value = raw_value.rstrip(",")
        if value.replace(".", "", 1).replace("-", "", 1).isdigit():
            if "." in value:
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
        else:
            parsed[key] = value
    return parsed


def extract_last_stats(stdout: str):
    last = None
    for line in stdout.splitlines():
        if STATS_PREFIX in line:
            last = line[line.index(STATS_PREFIX) + len(STATS_PREFIX) :]
    if last is None:
        raise RuntimeError("Fast GI stats line not found in Blender output.")
    return parse_stats_line(last)


def leading_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and "/" in value:
        head = value.split("/", 1)[0]
        if head.isdigit():
            return int(head)
    return 0


def run_case(blender_bin: Path, budget_mb: int):
    cmd = [
        str(blender_bin),
        "-b",
        "--factory-startup",
        "-P",
        str(Path(__file__).resolve()),
        "--",
        "--inner",
    ]
    env = os.environ.copy()
    env["BLENDER_EEVEE_HWRT_FAST_GI_STATS"] = "1"
    env["BLENDER_EEVEE_HWRT_FAST_GI_BUDGET_MB"] = str(budget_mb)
    env["BLENDER_GPU_METAL_IGNORE_TEXTURE_POOL_ASSERT"] = "1"
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return extract_last_stats(completed.stdout + "\n" + completed.stderr)


def outer_main():
    args = _parse_args()
    default_case = run_case(args.blender_bin, args.default_budget_mb)
    tight_case = run_case(args.blender_bin, args.tight_budget_mb)
    default_active = leading_int(default_case.get("active_cascades", 0))
    tight_active = leading_int(tight_case.get("active_cascades", 0))

    result = {
        "default_budget_mb": args.default_budget_mb,
        "tight_budget_mb": args.tight_budget_mb,
        "default": default_case,
        "tight": tight_case,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"LARGE_SCENE_DEFAULT_MEMORY_LIMITED={default_case.get('memory_limited', -1)}")
    print(f"LARGE_SCENE_DEFAULT_ACTIVE_CASCADES={default_case.get('active_cascades', -1)}")
    print(f"LARGE_SCENE_DEFAULT_FAST_GI_MEM_MIB={default_case.get('fast_gi_mem_mib', -1.0):.6f}")
    print(f"LARGE_SCENE_DEFAULT_BUDGET_MIB={default_case.get('budget_mib', -1.0):.6f}")
    print(f"LARGE_SCENE_TIGHT_MEMORY_LIMITED={tight_case.get('memory_limited', -1)}")
    print(f"LARGE_SCENE_TIGHT_ACTIVE_CASCADES={tight_case.get('active_cascades', -1)}")
    print(f"LARGE_SCENE_TIGHT_FAST_GI_MEM_MIB={tight_case.get('fast_gi_mem_mib', -1.0):.6f}")
    print(f"LARGE_SCENE_TIGHT_BUDGET_MIB={tight_case.get('budget_mib', -1.0):.6f}")
    print(f"LARGE_SCENE_ACTIVE_CASCADES_DROP={default_active - tight_active}")


if not _inside_blender():
    outer_main()
    raise SystemExit(0)


import bpy
from mathutils import Vector


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
    emission.inputs["Color"].default_value = (1.0, 0.82, 0.58, 1.0)
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


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_scene(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    eevee = scene.eevee
    eevee.use_raytracing = True
    eevee.ray_tracing_method = "HARDWARE"
    eevee.use_hardware_raytracing_gi = True
    eevee.hardware_raytracing_reflection_mode = "OFF"
    eevee.hardware_raytracing_refraction_mode = "OFF"
    eevee.use_hardware_raytracing_environment = False
    eevee.use_hardware_raytracing_shadows = False
    eevee.use_hardware_raytracing_caustics = False
    eevee.fast_gi_distance = 0.0
    eevee.fast_gi_resolution = "1"
    eevee.taa_render_samples = 1
    eevee.taa_samples = 1

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.use_denoise = False
    ray_tracing.denoise_spatial = False
    ray_tracing.denoise_temporal = False
    ray_tracing.denoise_bilateral = False


def ensure_world(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTStreamingMemoryWorld")
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


def build_large_scene(scene):
    ensure_world(scene)
    diffuse = make_diffuse_material("StreamingMemoryDiffuse", (0.82, 0.82, 0.82, 1.0))
    emitter = make_emission_material("StreamingMemoryEmitter", 32.0)

    create_plane("ProbeFloor", (0.0, 0.0, 0.0), (60.0, 60.0, 1.0), diffuse)

    index = 0
    for x in range(-48, 49, 12):
        for y in range(-48, 49, 12):
            height = 1.6 + ((x + y) % 3) * 0.7
            create_cube(f"ScatterBlock_{index}", (x, y, height * 0.5), (2.2, 2.2, height), diffuse)
            if (x + y) % 24 == 0:
                light_panel = create_plane(
                    f"EmitterPanel_{index}",
                    (x + 1.5, y - 1.5, height + 1.2),
                    (0.9, 0.9, 1.0),
                    emitter,
                )
                light_panel.rotation_euler.x = 1.57079632679
            index += 1

    bpy.ops.object.camera_add(location=(0.0, -54.0, 18.0))
    camera = bpy.context.active_object
    look_at(camera, Vector((0.0, 0.0, 5.0)))
    scene.camera = camera


def inner_main():
    _args = _parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene)
    build_large_scene(scene)
    output_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_streaming_memory_inner_")
    scene.render.filepath = str(Path(output_dir.name) / "streaming_memory_probe.png")
    bpy.ops.render.render(write_still=True)
    output_dir.cleanup()


inner_main()
