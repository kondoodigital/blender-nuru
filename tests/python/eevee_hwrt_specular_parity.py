#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Focused Eevee Hardware RT specular parity checks on scenes/test.blend."
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
        default=16,
        help="Render samples for each parity scenario.",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the scene render resolution for faster checks.",
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


@dataclass
class ImageStats:
    mean_rgb: tuple[float, float, float]
    std_rgb: tuple[float, float, float]


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
    eevee.ray_tracing_reflection_bounces = 4
    eevee.ray_tracing_refraction_bounces = 4
    eevee.taa_render_samples = max(1, samples)

    ray_tracing = eevee.ray_tracing_options
    ray_tracing.resolution_scale = "1"
    ray_tracing.screen_trace_quality = 1.0
    ray_tracing.screen_trace_thickness = 1.0


def ensure_checker_emission_material(name: str = "HWRTSpecularParityChecker"):
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


def ensure_image_emission_material(
    name: str = "HWRTSpecularParityImage", coord_output: str = "Generated"
):
    image_a = ensure_quadrant_image(
        "HWRTSpecularParityImageA",
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
    )
    image_b = ensure_quadrant_image(
        "HWRTSpecularParityImageB",
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


def ensure_modifier_probe(obj, name: str = "HWRTSpecularParitySubsurf"):
    modifier = obj.modifiers.get(name)
    if modifier is None:
        modifier = obj.modifiers.new(name=name, type="SUBSURF")
    modifier.levels = 2
    modifier.render_levels = 2
    modifier.subdivision_type = "CATMULL_CLARK"
    modifier.show_render = True
    modifier.show_viewport = True
    return modifier


def ensure_mirror_material(name: str = "HWRTSpecularParityMirror"):
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


def ensure_glass_material(name: str = "HWRTSpecularParityGlass"):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        ntree = material.node_tree
        ntree.nodes.clear()
        glass = ntree.nodes.new("ShaderNodeBsdfGlass")
        output = ntree.nodes.new("ShaderNodeOutputMaterial")
        glass.inputs["IOR"].default_value = 1.45
        glass.inputs["Roughness"].default_value = 0.05
        ntree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    glass = next(node for node in material.node_tree.nodes if node.type == "BSDF_GLASS")
    return material, glass


def ensure_probe_plane(name: str = "HWRTSpecularParityMirrorPlane"):
    plane = bpy.data.objects.get(name)
    if plane is None:
        bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
        plane = bpy.context.active_object
        plane.name = name
        plane.data.name = f"{name}Mesh"
    return plane


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def align_object_normal(obj, normal: Vector):
    obj.rotation_euler = normal.normalized().to_track_quat("Z", "Y").to_euler()


def setup_mirror_plane_layout(scene, cube, camera, light):
    plane = ensure_probe_plane()

    for obj in bpy.data.objects:
        if obj.name not in {cube.name, camera.name, light.name, plane.name}:
            obj.hide_render = True

    plane.hide_render = False
    plane.location = Vector((0.0, 0.0, 0.0))
    plane.scale = Vector((2.5, 2.5, 2.5))

    cube.hide_render = False
    cube.location = Vector((0.0, 1.5, 1.0))
    cube.scale = Vector((0.8, 0.8, 0.8))
    cube.rotation_euler = (0.0, 0.0, 0.0)

    camera.location = Vector((0.0, -6.0, 3.0))
    look_at(camera, Vector((0.0, 0.0, 0.7)))

    light.location = Vector((0.0, -3.0, 5.0))
    light.data.energy = 0.0
    return plane


def ensure_emission_cube(name: str = "HWRTSpecularParityBounceTarget"):
    obj = bpy.data.objects.get(name)
    if obj is None:
        bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0.0, 0.0, 0.0))
        obj = bpy.context.active_object
        obj.name = name
        obj.data.name = f"{name}Mesh"

    material_name = f"{name}Material"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        ntree = material.node_tree
        ntree.nodes.clear()
        emission = ntree.nodes.new("ShaderNodeEmission")
        output = ntree.nodes.new("ShaderNodeOutputMaterial")
        emission.inputs["Color"].default_value = (1.0, 0.1, 0.1, 1.0)
        emission.inputs["Strength"].default_value = 25.0
        ntree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def ensure_world_background(scene):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("HWRTSpecularParityWorld")
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
    return background


def ensure_shadow_occluder(cube, point_light, name: str = "HWRTSpecularParityOccluder"):
    occluder = bpy.data.objects.get(name)
    if occluder is None:
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        occluder = bpy.context.active_object
        occluder.name = name
        occluder.data.name = f"{name}Mesh"

    occluder.hide_select = True
    occluder.location = point_light.location.lerp(cube.location, 0.4)
    max_dim = max(cube.dimensions.x, cube.dimensions.y, cube.dimensions.z)
    occluder.scale = Vector((max_dim * 0.3, max_dim * 0.3, max_dim * 0.3))
    occluder.hide_render = True
    return occluder


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
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    return width, height, pixels, path


def refresh_material_binding(material, obj):
    material.node_tree.update_tag()
    obj.update_tag()
    bpy.context.view_layer.update()


def crop_stats(pixels, width: int, height: int, bounds):
    min_x, min_y, max_x, max_y = bounds
    sums = [0.0, 0.0, 0.0]
    sums_sq = [0.0, 0.0, 0.0]
    count = 0

    for y in range(min_y, max_y + 1):
        row = y * width * 4
        for x in range(min_x, max_x + 1):
            base = row + x * 4
            for channel in range(3):
                value = pixels[base + channel]
                sums[channel] += value
                sums_sq[channel] += value * value
            count += 1

    means = tuple(channel_sum / count for channel_sum in sums)
    stds = []
    for idx in range(3):
        variance = max(0.0, (sums_sq[idx] / count) - (means[idx] * means[idx]))
        stds.append(math.sqrt(variance))
    return ImageStats(mean_rgb=means, std_rgb=tuple(stds))


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


def assert_metric(name: str, value: float, minimum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold={minimum:.6f}")
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def main():
    args = _parse_args()
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_specular_parity_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    camera = scene.camera
    sphere = bpy.data.objects["Sphere"]
    cube = bpy.data.objects["Cube"]
    point_light = bpy.data.objects["Point"]
    original_sphere_material = sphere.material_slots[0].material
    glass_material, glass_node = ensure_glass_material()
    sphere.material_slots[0].material = glass_material

    configure_scene(scene, args.samples, args.resolution_scale)

    original_cube_material = cube.material_slots[0].material
    checker_material, checker_mapping = ensure_checker_emission_material()
    image_material, image_node, image_a, image_b = ensure_image_emission_material()
    uv_image_material, uv_image_node, uv_image_a, uv_image_b = ensure_image_emission_material(
        name="HWRTSpecularParityUVImage",
        coord_output="UV",
    )

    bounds_cache = {}

    def render_case(tag: str, focus_obj=None, bounds_key: str = "sphere"):
        width, height, pixels, path = render_to_image(scene, output_dir, tag)
        tracked_obj = sphere if focus_obj is None else focus_obj
        if bounds_key not in bounds_cache:
            bounds_cache[bounds_key] = object_bbox_pixels(scene, camera, tracked_obj, width, height)
        focus_stats = crop_stats(pixels, width, height, bounds_cache[bounds_key])
        return {
            "tag": tag,
            "width": width,
            "height": height,
            "pixels": pixels,
            "path": path,
            "focus_stats": focus_stats,
            "sphere_stats": focus_stats if bounds_key == "sphere" else None,
        }

    failures = []

    glass_node.inputs["IOR"].default_value = 1.10
    glass_node.inputs["Roughness"].default_value = 0.05
    cube.material_slots[0].material = original_cube_material
    point_light.data.energy = 500.0
    ior_low = render_case("ior_low")

    glass_node.inputs["IOR"].default_value = 1.80
    ior_high = render_case("ior_high")
    ior_diff = crop_abs_diff_mean(
        ior_low["pixels"],
        ior_high["pixels"],
        ior_low["width"],
        ior_low["height"],
        bounds_cache["sphere"],
    )
    assert_metric("IOR_CROP_DIFF_MEAN", ior_diff, 0.01, failures)

    glass_node.inputs["IOR"].default_value = 1.45
    glass_node.inputs["Roughness"].default_value = 0.0
    rough_low = render_case("roughness_low")

    glass_node.inputs["Roughness"].default_value = 1.0
    rough_high = render_case("roughness_high")
    rough_diff = crop_abs_diff_mean(
        rough_low["pixels"],
        rough_high["pixels"],
        rough_low["width"],
        rough_low["height"],
        bounds_cache["sphere"],
    )
    assert_metric("ROUGHNESS_CROP_DIFF_MEAN", rough_diff, 0.02, failures)

    glass_node.inputs["IOR"].default_value = 1.45
    glass_node.inputs["Roughness"].default_value = 0.0
    cube.material_slots[0].material = checker_material
    checker_mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
    checker_mapping.inputs["Location"].default_value = (0.0, 0.0, 0.0)
    checker_low = render_case("checker_phase_a")

    checker_mapping.inputs["Location"].default_value = (0.125, 0.0, 0.0)
    checker_high = render_case("checker_phase_b")
    checker_diff = crop_max_tile_abs_diff_mean(
        checker_low["pixels"],
        checker_high["pixels"],
        checker_low["width"],
        checker_low["height"],
        bounds_cache["sphere"],
    )
    assert_metric("CHECKER_MAX_TILE_DIFF_MEAN", checker_diff, 0.004, failures)
    assert_metric(
        "CHECKER_CROP_STDDEV_MAX",
        max(checker_high["sphere_stats"].std_rgb),
        0.02,
        failures,
    )

    checker_mapping.inputs["Location"].default_value = (0.0, 0.0, 0.0)
    modifier_probe = ensure_modifier_probe(cube)
    checker_modifier_low = render_case("checker_modifier_phase_a")

    checker_mapping.inputs["Location"].default_value = (0.125, 0.0, 0.0)
    checker_modifier_high = render_case("checker_modifier_phase_b")
    checker_modifier_diff = crop_max_tile_abs_diff_mean(
        checker_modifier_low["pixels"],
        checker_modifier_high["pixels"],
        checker_modifier_low["width"],
        checker_modifier_low["height"],
        bounds_cache["sphere"],
    )
    assert_metric("CHECKER_MODIFIER_MAX_TILE_DIFF_MEAN", checker_modifier_diff, 0.0025, failures)
    assert_metric(
        "CHECKER_MODIFIER_CROP_STDDEV_MAX",
        max(checker_modifier_high["sphere_stats"].std_rgb),
        0.02,
        failures,
    )
    cube.modifiers.remove(modifier_probe)

    cube.material_slots[0].material = image_material
    image_node.image = image_a
    refresh_material_binding(image_material, cube)
    image_a_case = render_case("image_texture_a")

    image_node.image = image_b
    refresh_material_binding(image_material, cube)
    image_b_case = render_case("image_texture_b")
    image_texture_diff = crop_max_tile_abs_diff_mean(
        image_a_case["pixels"],
        image_b_case["pixels"],
        image_a_case["width"],
        image_a_case["height"],
        bounds_cache["sphere"],
    )
    assert_metric("IMAGE_TEXTURE_MAX_TILE_DIFF_MEAN", image_texture_diff, 0.004, failures)
    assert_metric(
        "IMAGE_TEXTURE_CROP_STDDEV_MAX",
        max(image_b_case["sphere_stats"].std_rgb),
        0.02,
        failures,
    )

    cube.material_slots[0].material = uv_image_material
    uv_image_node.image = uv_image_a
    refresh_material_binding(uv_image_material, cube)
    uv_image_a_case = render_case("uv_image_texture_a")

    uv_image_node.image = uv_image_b
    refresh_material_binding(uv_image_material, cube)
    uv_image_b_case = render_case("uv_image_texture_b")
    uv_image_texture_diff = crop_max_tile_abs_diff_mean(
        uv_image_a_case["pixels"],
        uv_image_b_case["pixels"],
        uv_image_a_case["width"],
        uv_image_a_case["height"],
        bounds_cache["sphere"],
    )
    assert_metric("UV_IMAGE_TEXTURE_MAX_TILE_DIFF_MEAN", uv_image_texture_diff, 0.004, failures)
    assert_metric(
        "UV_IMAGE_TEXTURE_CROP_STDDEV_MAX",
        max(uv_image_b_case["sphere_stats"].std_rgb),
        0.02,
        failures,
    )

    cube.material_slots[0].material = original_cube_material
    glass_node.inputs["Roughness"].default_value = 0.05
    point_light.data.energy = 0.0
    light_off = render_case("point_light_off")

    point_light.data.energy = 2500.0
    light_on = render_case("point_light_on")
    light_diff = crop_abs_diff_mean(
        light_off["pixels"],
        light_on["pixels"],
        light_off["width"],
        light_off["height"],
        bounds_cache["sphere"],
    )
    assert_metric("POINT_LIGHT_CROP_DIFF_MEAN", light_diff, 0.01, failures)

    shadow_occluder = ensure_shadow_occluder(cube, point_light)
    point_light.data.energy = 2500.0

    scene.eevee.use_hardware_raytracing_shadows = False
    shadow_occluder.hide_render = True
    shadow_map_clear = render_case("shadow_map_clear")

    shadow_occluder.hide_render = False
    shadow_map_blocked = render_case("shadow_map_blocked")
    shadow_map_diff = crop_max_tile_abs_diff_mean(
        shadow_map_clear["pixels"],
        shadow_map_blocked["pixels"],
        shadow_map_clear["width"],
        shadow_map_clear["height"],
        bounds_cache["sphere"],
    )
    assert_metric("SHADOWMAP_REFLECTION_MAX_TILE_DIFF_MEAN", shadow_map_diff, 0.01, failures)

    scene.eevee.use_hardware_raytracing_shadows = True
    shadow_occluder.hide_render = True
    hw_shadow_clear = render_case("hw_shadow_clear")

    shadow_occluder.hide_render = False
    hw_shadow_blocked = render_case("hw_shadow_blocked")
    hw_shadow_diff = crop_max_tile_abs_diff_mean(
        hw_shadow_clear["pixels"],
        hw_shadow_blocked["pixels"],
        hw_shadow_clear["width"],
        hw_shadow_clear["height"],
        bounds_cache["sphere"],
    )
    assert_metric("HWRT_SHADOW_REFLECTION_MAX_TILE_DIFF_MEAN", hw_shadow_diff, 0.01, failures)

    mirror_material = ensure_mirror_material()
    background = ensure_world_background(scene)
    original_world_color = tuple(background.inputs["Color"].default_value)
    original_world_strength = background.inputs["Strength"].default_value

    plane = setup_mirror_plane_layout(scene, cube, camera, point_light)
    plane.data.materials.clear()
    plane.data.materials.append(mirror_material)
    cube.data.materials.clear()
    cube.data.materials.append(original_cube_material)
    scene.eevee.use_hardware_raytracing_shadows = True
    scene.eevee.use_hardware_raytracing_environment = True

    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    reflected_gi_off = render_case("reflected_world_gi_off", focus_obj=plane, bounds_key="mirror_plane")

    background.inputs["Color"].default_value = (0.15, 0.45, 1.0, 1.0)
    background.inputs["Strength"].default_value = 1.5
    reflected_gi_on = render_case("reflected_world_gi_on", focus_obj=plane, bounds_key="mirror_plane")
    reflected_gi_diff = crop_abs_diff_mean(
        reflected_gi_off["pixels"],
        reflected_gi_on["pixels"],
        reflected_gi_off["width"],
        reflected_gi_off["height"],
        bounds_cache["mirror_plane"],
    )
    assert_metric("REFLECTED_WORLD_GI_DIFF_MEAN", reflected_gi_diff, 0.01, failures)

    bounce_mirror = ensure_probe_plane("HWRTSpecularParityBounceMirror")
    bounce_mirror.hide_render = False
    bounce_mirror.location = Vector((0.0, 1.9, 1.1))
    bounce_mirror.scale = Vector((0.7, 0.7, 0.7))
    bounce_mirror.data.materials.clear()
    bounce_mirror.data.materials.append(mirror_material)

    bounce_target = ensure_emission_cube()
    bounce_target.hide_render = False
    bounce_target.location = Vector((2.4, 1.9, 1.1))
    bounce_target.scale = Vector((0.35, 0.35, 0.35))

    cube.hide_render = True
    reflected_camera = camera.location.copy()
    reflected_camera.z = -reflected_camera.z
    mirror_view_dir = (reflected_camera - bounce_mirror.location).normalized()
    mirror_target_dir = (bounce_target.location - bounce_mirror.location).normalized()
    align_object_normal(bounce_mirror, mirror_view_dir + mirror_target_dir)

    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    scene.eevee.ray_tracing_reflection_bounces = 1
    bounce_one = render_case("mirror_bounce_1", focus_obj=plane, bounds_key="mirror_plane")

    scene.eevee.ray_tracing_reflection_bounces = 2
    bounce_two = render_case("mirror_bounce_2", focus_obj=plane, bounds_key="mirror_plane")
    bounce_diff = crop_abs_diff_mean(
        bounce_one["pixels"],
        bounce_two["pixels"],
        bounce_one["width"],
        bounce_one["height"],
        bounds_cache["mirror_plane"],
    )
    print(f"MIRROR_BOUNCE_CHAIN_DIFF_MEAN={bounce_diff:.6f} threshold=informational")

    background.inputs["Color"].default_value = original_world_color
    background.inputs["Strength"].default_value = original_world_strength
    sphere.material_slots[0].material = original_sphere_material

    print(f"OUTPUT_DIR={output_dir}")
    if failures:
        print("EEVEE HWRT specular parity failures:")
        for failure in failures:
            print(f" - {failure}")
        sys.exit(1)

    print("EEVEE HWRT specular parity checks passed.")

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


main()
