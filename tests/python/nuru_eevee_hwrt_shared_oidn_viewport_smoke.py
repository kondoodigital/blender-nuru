# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Smoke-test Nuru HWRT/OIDN combinations in a live Rendered viewport.

Run from a GUI Blender process, not background mode. The script cycles viewport
settings and relies on process stability plus log markers as the crash signal.
"""

from __future__ import annotations

import itertools
import math
import sys

import bpy


def _set_enum_if_valid(owner, attr: str, value: str) -> bool:
    if not hasattr(owner, attr):
        return False
    prop = owner.bl_rna.properties[attr]
    valid_values = {item.identifier for item in prop.enum_items}
    if value not in valid_values:
        return False
    setattr(owner, attr, value)
    return True


def _set_render_engine(scene: bpy.types.Scene) -> None:
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"


def _create_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _create_scene(scene: bpy.types.Scene) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    mat_floor = _create_material("smoke_diffuse_floor", (0.8, 0.16, 0.08, 1.0))
    mat_reflect = _create_material("smoke_reflective_blue", (0.08, 0.18, 0.9, 1.0))
    mat_sss = _create_material("smoke_sss_green", (0.25, 0.85, 0.45, 1.0))

    bpy.ops.mesh.primitive_plane_add(size=7.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.data.materials.append(mat_floor)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.95, 0.0, 0.62))
    cube = bpy.context.object
    cube.name = "Smoke Reflective Receiver"
    cube.data.materials.append(mat_reflect)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.38, location=(0.9, -0.2, 0.45))
    sphere = bpy.context.object
    sphere.name = "Smoke SSS Receiver"
    sphere.data.materials.append(mat_sss)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 4.2))
    light = bpy.context.object
    light.name = "Smoke Area Light"
    light.data.energy = 500.0
    light.data.size = 4.0

    bpy.ops.object.camera_add(location=(0.0, -5.2, 2.4), rotation=(math.radians(66.0), 0.0, 0.0))
    scene.camera = bpy.context.object


def _configure_eevee(scene: bpy.types.Scene) -> None:
    eevee = scene.eevee
    eevee.use_raytracing = True
    _set_enum_if_valid(eevee, "ray_tracing_method", "HARDWARE")

    for samples_attr in ("taa_render_samples", "taa_samples"):
        if hasattr(eevee, samples_attr):
            setattr(eevee, samples_attr, 4)

    for attr, value in (
        ("use_hardware_raytracing_gi", True),
        ("use_hardware_raytracing_shadows", True),
        ("use_hardware_raytracing_indirect_gi_cache", False),
    ):
        if hasattr(eevee, attr):
            setattr(eevee, attr, value)

    for attr in ("hardware_raytracing_reflection_mode", "hardware_raytracing_refraction_mode"):
        _set_enum_if_valid(eevee, attr, "FULL")

    opts = eevee.ray_tracing_options
    if hasattr(opts, "denoise_use_gpu"):
        opts.denoise_use_gpu = True
    _set_enum_if_valid(opts, "denoise_input_passes", "RGB")
    _set_enum_if_valid(opts, "denoise_prefilter", "FAST")
    _set_enum_if_valid(opts, "denoise_quality", "BALANCED")


def _configure_rendered_viewport(scene: bpy.types.Scene) -> None:
    if bpy.context.window is None:
        raise RuntimeError("Viewport smoke requires a GUI window; do not run with --background")

    for area in bpy.context.window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type != "VIEW_3D":
                continue
            space.shading.type = "RENDERED"
            space.region_3d.view_perspective = "CAMERA"
            space.overlay.show_overlays = False
            area.tag_redraw()

    scene.frame_set(1)
    bpy.context.view_layer.update()


def _set_combo(scene: bpy.types.Scene, combo: tuple[str, bool, str, str, str, str]) -> None:
    label, use_oidn, resolution_scale, gi_spatial, reflection_mode, refraction_mode = combo
    eevee = scene.eevee
    opts = eevee.ray_tracing_options

    _set_enum_if_valid(opts, "denoise_filter", "OIDN" if use_oidn else "BILATERAL")
    _set_enum_if_valid(opts, "resolution_scale", resolution_scale)
    _set_enum_if_valid(opts, "gi_spatial_samples", gi_spatial)
    _set_enum_if_valid(eevee, "hardware_raytracing_reflection_mode", reflection_mode)
    _set_enum_if_valid(eevee, "hardware_raytracing_refraction_mode", refraction_mode)

    print(f"VIEWPORT_HWRT_SMOKE_BEGIN {label}", flush=True)


def _redraw_viewport() -> None:
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    bpy.context.view_layer.update()
    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
    except Exception as ex:
        print(f"VIEWPORT_HWRT_SMOKE_REDRAW_TIMER_WARNING {ex}", flush=True)


def _combo_list() -> list[tuple[str, bool, str, str, str, str]]:
    combos = []
    for use_oidn, resolution_scale, gi_spatial, reflection_mode, refraction_mode in itertools.product(
        (True, False),
        ("1", "2", "4"),
        ("8", "16", "32"),
        ("OFF", "FULL"),
        ("OFF", "FULL"),
    ):
        label = (
            f"denoise_{'oidn' if use_oidn else 'bilateral'}"
            f"_scale_{resolution_scale}"
            f"_gi_{gi_spatial}"
            f"_refl_{reflection_mode.lower()}"
            f"_refr_{refraction_mode.lower()}"
        )
        combos.append((label, use_oidn, resolution_scale, gi_spatial, reflection_mode, refraction_mode))
    return combos


def main() -> None:
    scene = bpy.context.scene
    _set_render_engine(scene)
    scene.render.resolution_x = 128
    scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    _create_scene(scene)
    _configure_eevee(scene)
    _configure_rendered_viewport(scene)

    combos = _combo_list()
    state = {"index": 0, "phase": "set"}

    print(f"VIEWPORT_HWRT_SMOKE_TOTAL {len(combos)}", flush=True)

    def step() -> float | None:
        index = state["index"]
        if index >= len(combos):
            print("VIEWPORT_HWRT_SMOKE_DONE", flush=True)
            bpy.ops.wm.quit_blender()
            return None

        combo = combos[index]
        if state["phase"] == "set":
            _set_combo(scene, combo)
            state["phase"] = "draw"
            return 0.15

        _redraw_viewport()
        print(f"VIEWPORT_HWRT_SMOKE_OK {combo[0]}", flush=True)
        state["index"] += 1
        state["phase"] = "set"
        return 0.05

    bpy.app.timers.register(step, first_interval=0.25)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
