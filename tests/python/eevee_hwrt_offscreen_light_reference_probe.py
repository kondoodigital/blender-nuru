#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _inside_blender():
    try:
        import bpy  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Compare Eevee HWRT off-screen and behind-camera emissive response against Cycles diffuse references."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--blender-bin", type=Path, default=None)
    parser.add_argument("--eevee-samples", type=int, default=32)
    parser.add_argument("--cycles-samples", type=int, default=128)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    parser.add_argument("--emissive-strength", type=float, default=80.0)
    parser.add_argument("--inner-engine", choices=("eevee", "cycles"), default=None)
    parser.add_argument("--case", choices=("offscreen", "behind_camera"), default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    return parser.parse_args(argv)


def assert_metric_min(name: str, value: float, minimum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold_min={minimum:.6f}", flush=True)
    if value < minimum:
        failures.append(f"{name} expected >= {minimum:.6f}, got {value:.6f}")


def assert_metric_max(name: str, value: float, maximum: float, failures: list[str]):
    print(f"{name}={value:.6f} threshold_max={maximum:.6f}", flush=True)
    if value > maximum:
        failures.append(f"{name} expected <= {maximum:.6f}, got {value:.6f}")


def crop_abs_diff_mean(crop_a, crop_b):
    total = 0.0
    count = 0
    for pixel_a, pixel_b in zip(crop_a, crop_b):
        total += abs(pixel_a - pixel_b)
        count += 1
    return total / max(1, count)


def run_subprocess_probe(blender_bin: Path, script_path: Path, args, engine: str, case_id: str, output_dir: Path):
    metrics_path = output_dir / f"{case_id}_{engine}_metrics.json"
    command = [
        str(blender_bin),
        "-b",
        "--factory-startup",
        "-P",
        str(script_path),
        "--",
        "--inner-engine",
        engine,
        "--case",
        case_id,
        "--metrics-json",
        str(metrics_path),
        "--output-dir",
        str(output_dir),
        "--eevee-samples",
        str(args.eevee_samples),
        "--cycles-samples",
        str(args.cycles_samples),
        "--resolution-x",
        str(args.resolution_x),
        "--resolution-y",
        str(args.resolution_y),
        "--emissive-strength",
        str(args.emissive_strength),
    ]
    print(f"RUN_{case_id.upper()}_{engine.upper()}={' '.join(command)}", flush=True)
    result = subprocess.run(command, check=False)
    if not metrics_path.exists():
        raise RuntimeError(
            f"{case_id}/{engine} run did not produce metrics at {metrics_path} (exit={result.returncode})"
        )
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, result.returncode


def evaluate_case(case_label: str, eevee_payload, cycles_payload, eevee_code: int, cycles_code: int):
    failures = []
    diff_mean = crop_abs_diff_mean(eevee_payload["crop_rgb"], cycles_payload["crop_rgb"])
    luma_ratio = eevee_payload["patch_luma"] / max(cycles_payload["patch_luma"], 1.0e-6)
    prefix = case_label.upper()

    print(f"{prefix}_PATCH_BOUNDS={tuple(eevee_payload['patch_bounds'])}", flush=True)
    print(f"{prefix}_EEVEE_EXIT_CODE={eevee_code}", flush=True)
    print(f"{prefix}_CYCLES_EXIT_CODE={cycles_code}", flush=True)
    assert_metric_min(f"{prefix}_EEVEE_PATCH_LUMA", eevee_payload["patch_luma"], 0.02, failures)
    assert_metric_min(f"{prefix}_CYCLES_PATCH_LUMA", cycles_payload["patch_luma"], 0.03, failures)
    assert_metric_min(f"{prefix}_PATCH_LUMA_RATIO", luma_ratio, 0.35, failures)
    assert_metric_max(f"{prefix}_PATCH_ABS_DIFF_MEAN", diff_mean, 0.25, failures)
    return failures


def run_outer(args):
    output_dir = args.output_dir
    cleanup_dir = None
    if output_dir is None:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="eevee_hwrt_offscreen_light_reference_")
        output_dir = Path(cleanup_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    blender_bin = args.blender_bin or Path(os.environ.get("BLENDER_BIN", ""))
    if not blender_bin or not blender_bin.exists():
        raise RuntimeError("--blender-bin is required when running outside Blender.")

    script_path = Path(__file__).resolve()
    failures = []
    for case_id in ("offscreen", "behind_camera"):
        eevee_payload, eevee_code = run_subprocess_probe(blender_bin, script_path, args, "eevee", case_id, output_dir)
        cycles_payload, cycles_code = run_subprocess_probe(blender_bin, script_path, args, "cycles", case_id, output_dir)
        failures.extend(evaluate_case(case_id, eevee_payload, cycles_payload, eevee_code, cycles_code))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    print("EEVEE HWRT off-screen light reference probe passed.", flush=True)

    if cleanup_dir is not None:
        cleanup_dir.cleanup()


if _inside_blender():
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    def look_at(obj, target: Vector):
        direction = target - obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


    def ensure_world(scene):
        world = scene.world
        if world is None:
            world = bpy.data.worlds.new("HWRTOffscreenReferenceWorld")
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
        emission.inputs["Color"].default_value = (1.0, 0.85, 0.62, 1.0)
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


    def create_room(scene, emissive_strength: float, case_id: str):
        wall = make_diffuse_material(f"ProbeWallMat_{case_id}", (0.90, 0.90, 0.90, 1.0))
        floor_mat = make_diffuse_material(f"ProbeFloorMat_{case_id}", (0.82, 0.82, 0.82, 1.0))
        blocker_mat = make_diffuse_material(f"ProbeBlockerMat_{case_id}", (0.65, 0.65, 0.65, 1.0))
        emitter_mat = make_emission_material(f"ProbeEmitterMat_{case_id}", emissive_strength)

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

        bpy.ops.object.camera_add(location=(0.0, -5.8, 1.65))
        camera = bpy.context.active_object
        target_patch = Vector((0.0, -0.95, 0.0))
        look_at(camera, Vector((0.0, -0.5, 0.45)))
        scene.camera = camera

        emitter = create_plane("ProbeEmitter", (0.0, 0.0, 0.0), (0.55, 0.55, 1.0), emitter_mat)
        if case_id == "offscreen":
            emitter.location = Vector((1.95, 1.05, 2.25))
            emitter.rotation_euler = Vector((0.0, math.radians(90.0), math.radians(180.0)))
        elif case_id == "behind_camera":
            emitter.location = Vector((0.0, -6.45, 1.85))
            emitter.rotation_euler = Vector((math.radians(90.0), 0.0, 0.0))
        else:
            raise RuntimeError(f"Unsupported case: {case_id}")

        return camera, target_patch


    def ensure_cycles_engine_available():
        engine_items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        if any(item.identifier == "CYCLES" for item in engine_items):
            return

        addon_dir = Path(__file__).resolve().parents[2] / "intern" / "cycles" / "blender" / "addon"
        spec = importlib.util.spec_from_file_location(
            "cycles",
            addon_dir / "__init__.py",
            submodule_search_locations=[str(addon_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load Cycles addon from {addon_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.register()


    def configure_render_common(scene, args):
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.resolution_x = args.resolution_x
        scene.render.resolution_y = args.resolution_y
        scene.render.resolution_percentage = 100


    def configure_eevee(scene, args):
        scene.render.engine = "BLENDER_EEVEE"
        eevee = scene.eevee
        eevee.use_raytracing = True
        eevee.ray_tracing_method = "HARDWARE"
        eevee.use_hardware_raytracing_gi = True
        eevee.hardware_raytracing_reflection_mode = "OFF"
        eevee.hardware_raytracing_refraction_mode = "OFF"
        eevee.use_hardware_raytracing_environment = False
        eevee.use_hardware_raytracing_shadows = False
        eevee.taa_render_samples = max(1, args.eevee_samples)

        ray_tracing = eevee.ray_tracing_options
        ray_tracing.resolution_scale = "1"
        ray_tracing.use_denoise = False
        ray_tracing.denoise_spatial = False
        ray_tracing.denoise_temporal = False
        ray_tracing.denoise_bilateral = False


    def configure_cycles(scene, args):
        ensure_cycles_engine_available()
        scene.render.engine = "CYCLES"
        cycles = scene.cycles
        cycles.device = "CPU"
        cycles.samples = max(1, args.cycles_samples)
        cycles.use_denoising = False
        cycles.max_bounces = 8
        cycles.diffuse_bounces = 8
        cycles.glossy_bounces = 0
        cycles.transmission_bounces = 0
        cycles.transparent_max_bounces = 0
        cycles.volume_bounces = 0


    def projected_crop_bounds(scene, camera, point: Vector, width: int, height: int, radius_px: int):
        co = world_to_camera_view(scene, camera, point)
        center_x = int(round(co.x * width))
        center_y = int(round(co.y * height))
        min_x = max(0, center_x - radius_px)
        max_x = min(width - 1, center_x + radius_px)
        min_y = max(0, center_y - radius_px)
        max_y = min(height - 1, center_y + radius_px)
        return (min_x, min_y, max_x, max_y)


    def crop_payload_from_image(image, scene, camera, point: Vector):
        width, height = image.size
        pixels = list(image.pixels[:])
        bounds = projected_crop_bounds(scene, camera, point, width, height, radius_px=42)
        min_x, min_y, max_x, max_y = bounds
        crop_rgb = []
        total_luma = 0.0
        count = 0
        for y in range(min_y, max_y + 1):
            row = y * width * 4
            for x in range(min_x, max_x + 1):
                base = row + x * 4
                r = pixels[base + 0]
                g = pixels[base + 1]
                b = pixels[base + 2]
                crop_rgb.extend((r, g, b))
                total_luma += 0.2126 * r + 0.7152 * g + 0.0722 * b
                count += 1
        return {
            "patch_bounds": list(bounds),
            "patch_luma": total_luma / max(1, count),
            "crop_rgb": crop_rgb,
        }


    def install_metrics_handler(scene, image_path: Path, metrics_path: Path, camera, point: Vector, engine: str):
        captured = {"done": False}

        def _capture():
            if captured["done"]:
                return
            render_result = bpy.data.images.get("Render Result")
            if render_result is not None and render_result.size[0] > 0 and render_result.size[1] > 0:
                payload = crop_payload_from_image(render_result, scene, camera, point)
            else:
                if not image_path.exists():
                    return
                image = bpy.data.images.load(str(image_path), check_existing=False)
                payload = crop_payload_from_image(image, scene, camera, point)
                bpy.data.images.remove(image)
            metrics_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            print(f"METRICS_CAPTURED={engine} path={metrics_path}", flush=True)
            captured["done"] = True
            if engine == "eevee":
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)

        def _render_post(_scene):
            _capture()

        def _render_stats(_stats):
            _capture()

        bpy.app.handlers.render_post.append(_render_post)
        if engine == "eevee":
            bpy.app.handlers.render_stats.append(_render_stats)
        return _render_post, _render_stats if engine == "eevee" else None


    def run_inner(args):
        if args.inner_engine is None or args.metrics_json is None or args.case is None:
            raise RuntimeError("Inner Blender mode requires --inner-engine, --case, and --metrics-json.")
        if args.output_dir is None:
            raise RuntimeError("Inner Blender mode requires --output-dir.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        configure_render_common(scene, args)
        ensure_world(scene)
        camera, target_patch = create_room(scene, args.emissive_strength, args.case)

        if args.inner_engine == "eevee":
            configure_eevee(scene, args)
            image_path = args.output_dir / f"{args.case}_eevee.png"
        else:
            configure_cycles(scene, args)
            image_path = args.output_dir / f"{args.case}_cycles.png"

        scene.render.filepath = str(image_path)
        post_handler, stats_handler = install_metrics_handler(
            scene, image_path, args.metrics_json, camera, target_patch, args.inner_engine
        )

        print(f"RENDER_BEGIN={args.case}_{args.inner_engine} path={scene.render.filepath}", flush=True)
        bpy.ops.render.render(write_still=True)
        print(f"RENDER_DONE={args.case}_{args.inner_engine} path={scene.render.filepath}", flush=True)

        if args.inner_engine == "cycles" and not args.metrics_json.exists():
            render_result = bpy.data.images.get("Render Result")
            if render_result is None:
                raise RuntimeError("Cycles render completed without a Render Result image.")
            payload = crop_payload_from_image(render_result, scene, camera, target_patch)
            args.metrics_json.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

        if post_handler in bpy.app.handlers.render_post:
            bpy.app.handlers.render_post.remove(post_handler)
        if stats_handler is not None and stats_handler in bpy.app.handlers.render_stats:
            bpy.app.handlers.render_stats.remove(stats_handler)


def main():
    args = _parse_args()
    if _inside_blender():
        run_inner(args)
    else:
        run_outer(args)


if __name__ == "__main__":
    main()
