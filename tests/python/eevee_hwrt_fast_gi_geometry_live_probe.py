#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Warm Fast GI, edit scene geometry, and let the next sparse refresh settle."
    )
    parser.add_argument("--redraw-iterations", type=int, default=16)
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


def configure_scene(scene, taa_samples: int):
    scene.render.engine = "BLENDER_EEVEE"
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


def edit_geometry():
    bpy.ops.mesh.primitive_uv_sphere_add(location=(0.0, 0.0, 1.25), radius=0.45)
    obj = bpy.context.active_object
    obj.name = "FastGIGeometryProbeSphere"
    obj.update_tag()
    bpy.context.view_layer.update()


def main():
    args = _parse_args()
    scene = bpy.context.scene
    configure_scene(scene, args.taa_samples)
    window, area, region, space = find_view3d_context()
    force_viewport_camera(window, area, region, space)

    # Warm one traced field before the geometry edit so the invalidation stats reflect the edit
    # itself instead of startup/bootstrap state.
    redraw(args.redraw_iterations)
    edit_geometry()
    redraw(args.redraw_iterations)

    bpy.ops.wm.quit_blender()


main()
