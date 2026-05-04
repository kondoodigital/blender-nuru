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
        description="Render clear HWRT reflection proof images on scenes/test.blend."
    )
    parser.add_argument(
        "--proof",
        choices=(
            "uv_reflection",
            "uv_gradient_reflection",
            "uvmap_explicit_reflection",
            "uvmap_explicit_gradient_reflection",
            "modifier_reflection",
        ),
        default="uv_reflection",
        help="Which reflected proof image to render.",
    )
    parser.add_argument(
        "--view-mode",
        choices=("reflection", "direct"),
        default="reflection",
        help="Render the object only in reflection or directly for material reference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to keep rendered outputs. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Render samples for the proof image.",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to scene render resolution.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="proof",
        help="Output tag for the rendered image.",
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

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.use_denoise = False
    ray_tracing.denoise_spatial = False
    ray_tracing.denoise_temporal = False
    ray_tracing.denoise_bilateral = False
    ray_tracing.screen_trace_quality = 1.0
    ray_tracing.screen_trace_thickness = 1.0


def ensure_quadrant_image(name: str, quadrant_colors: tuple[tuple[float, float, float], ...], size: int = 16):
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


def ensure_uv_image_emission_material(name: str = "HWRTVisualProofUV"):
    image = ensure_quadrant_image(
        f"{name}Image",
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        ),
    )

    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        ntree = material.node_tree
        ntree.nodes.clear()

        texcoord = ntree.nodes.new("ShaderNodeTexCoord")
        image_node = ntree.nodes.new("ShaderNodeTexImage")
        emission = ntree.nodes.new("ShaderNodeEmission")
        output = ntree.nodes.new("ShaderNodeOutputMaterial")

        image_node.interpolation = "Closest"
        emission.inputs["Strength"].default_value = 1.0

        ntree.links.new(texcoord.outputs["UV"], image_node.inputs["Vector"])
        ntree.links.new(image_node.outputs["Color"], emission.inputs["Color"])
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    image_node = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage")
    image_node.image = image
    return material


def ensure_uvmap_image_emission_material(uv_map_name: str, name: str = "HWRTVisualProofUVExplicit"):
    image = ensure_quadrant_image(
        f"{name}Image",
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        ),
    )

    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()

    uvmap = ntree.nodes.new("ShaderNodeUVMap")
    image_node = ntree.nodes.new("ShaderNodeTexImage")
    emission = ntree.nodes.new("ShaderNodeEmission")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")

    uvmap.uv_map = uv_map_name
    image_node.interpolation = "Closest"
    image_node.image = image
    emission.inputs["Strength"].default_value = 1.0

    ntree.links.new(uvmap.outputs["UV"], image_node.inputs["Vector"])
    ntree.links.new(image_node.outputs["Color"], emission.inputs["Color"])
    ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def ensure_uv_gradient_emission_material(name: str = "HWRTVisualProofUVGradient"):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()

    texcoord = ntree.nodes.new("ShaderNodeTexCoord")
    separate = ntree.nodes.new("ShaderNodeSeparateXYZ")
    combine = ntree.nodes.new("ShaderNodeCombineXYZ")
    emission = ntree.nodes.new("ShaderNodeEmission")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")

    emission.inputs["Strength"].default_value = 1.0

    ntree.links.new(texcoord.outputs["UV"], separate.inputs["Vector"])
    ntree.links.new(separate.outputs["X"], combine.inputs["X"])
    ntree.links.new(separate.outputs["Y"], combine.inputs["Y"])
    ntree.links.new(separate.outputs["Z"], combine.inputs["Z"])
    ntree.links.new(combine.outputs["Vector"], emission.inputs["Color"])
    ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    return material


def ensure_uvmap_gradient_emission_material(
    uv_map_name: str, name: str = "HWRTVisualProofUVGradientExplicit"
):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    ntree = material.node_tree
    ntree.nodes.clear()

    uvmap = ntree.nodes.new("ShaderNodeUVMap")
    separate = ntree.nodes.new("ShaderNodeSeparateXYZ")
    combine = ntree.nodes.new("ShaderNodeCombineXYZ")
    emission = ntree.nodes.new("ShaderNodeEmission")
    output = ntree.nodes.new("ShaderNodeOutputMaterial")

    uvmap.uv_map = uv_map_name
    emission.inputs["Strength"].default_value = 1.0

    ntree.links.new(uvmap.outputs["UV"], separate.inputs["Vector"])
    ntree.links.new(separate.outputs["X"], combine.inputs["X"])
    ntree.links.new(separate.outputs["Y"], combine.inputs["Y"])
    ntree.links.new(separate.outputs["Z"], combine.inputs["Z"])
    ntree.links.new(combine.outputs["Vector"], emission.inputs["Color"])
    ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def ensure_checker_emission_material(name: str = "HWRTVisualProofChecker"):
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

        checker.inputs["Color1"].default_value = (1.0, 0.0, 1.0, 1.0)
        checker.inputs["Color2"].default_value = (0.1, 0.3, 1.0, 1.0)
        checker.inputs["Scale"].default_value = 12.0
        emission.inputs["Strength"].default_value = 1.0

        ntree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
        ntree.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
        ntree.links.new(checker.outputs["Color"], emission.inputs["Color"])
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    mapping = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
    return material


def ensure_mirror_material(name: str = "HWRTVisualProofMirror"):
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


def ensure_vertical_mirror_plane(name: str = "HWRTVisualProofMirrorPlane"):
    plane = bpy.data.objects.get(name)
    if plane is None:
        bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 1.5))
        plane = bpy.context.active_object
        plane.name = name
        plane.data.name = f"{name}Mesh"
    plane.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    plane.scale = Vector((3.0, 3.0, 3.0))
    return plane


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_layout(scene, view_mode: str):
    cube = bpy.data.objects["Cube"]
    camera = scene.camera
    light = bpy.data.objects["Point"]
    plane = ensure_vertical_mirror_plane()

    for obj in bpy.data.objects:
        if obj.name not in {cube.name, camera.name, light.name, plane.name}:
            obj.hide_viewport = True
            obj.hide_render = True

    cube.hide_viewport = False
    cube.hide_render = False
    cube.scale = Vector((1.0, 1.0, 1.0))
    cube.rotation_euler = (0.0, 0.0, 0.0)

    if view_mode == "reflection":
        plane.hide_viewport = False
        plane.hide_render = False
        plane.location = Vector((0.0, 0.0, 1.5))

        camera.location = Vector((0.0, -6.0, 1.5))
        look_at(camera, plane.location)

        cube.location = Vector((0.0, -10.0, 1.5))
    else:
        plane.hide_viewport = True
        plane.hide_render = True

        cube.location = Vector((0.0, 0.0, 1.5))
        camera.location = Vector((0.0, -6.0, 1.5))
        look_at(camera, cube.location)

    light.location = Vector((2.0, -5.0, 5.0))
    light.data.energy = 2000.0
    return cube, plane, camera


def render_to_image(scene, output_dir: Path, tag: str):
    path = output_dir / f"{tag}.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"OUTPUT_IMAGE={path}")
    return path


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_visual_proof_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    configure_scene(scene, args.samples, args.resolution_scale)
    cube, plane, camera = setup_layout(scene, args.view_mode)

    mirror_material = ensure_mirror_material()
    plane.data.materials.clear()
    plane.data.materials.append(mirror_material)

    modifier = cube.modifiers.get("HWRTVisualProofSubsurf")
    if modifier is None:
        modifier = cube.modifiers.new(name="HWRTVisualProofSubsurf", type="SUBSURF")
        modifier.levels = 2
        modifier.render_levels = 2
    modifier.show_viewport = False
    modifier.show_render = False

    uv_map_name = cube.data.uv_layers[0].name if cube.data.uv_layers else ""

    if args.proof == "uv_reflection":
        cube_material = ensure_uv_image_emission_material()
        cube.data.materials.clear()
        cube.data.materials.append(cube_material)
    elif args.proof == "uv_gradient_reflection":
        cube_material = ensure_uv_gradient_emission_material()
        cube.data.materials.clear()
        cube.data.materials.append(cube_material)
    elif args.proof == "uvmap_explicit_reflection":
        cube_material = ensure_uvmap_image_emission_material(uv_map_name)
        cube.data.materials.clear()
        cube.data.materials.append(cube_material)
    elif args.proof == "uvmap_explicit_gradient_reflection":
        cube_material = ensure_uvmap_gradient_emission_material(uv_map_name)
        cube.data.materials.clear()
        cube.data.materials.append(cube_material)
    else:
        cube_material = ensure_checker_emission_material()
        cube.data.materials.clear()
        cube.data.materials.append(cube_material)
        modifier.show_viewport = True
        modifier.show_render = True

    cube.update_tag()
    render_to_image(scene, output_dir, args.tag)
    print(f"PROOF={args.proof}")
    print(f"VIEW_MODE={args.view_mode}")
    print(f"OUTPUT_DIR={output_dir}")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
