#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed validation corpus for the RT-Only Eevee lighting work.

This is a lightweight manifest, not a monolithic runner. It gives the branch one
stable place to enumerate which focused probes and production checks define the
current superplan validation envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests" / "python"

REFERENCE_POLICIES = {
    "cycles_diffuse": {
        "renderer": "CYCLES",
        "integrator": "diffuse-only path tracing",
        "output": "saved reference image set",
        "notes": "Use for GI correctness, emissive-only interiors, leak checks, and production-scene bounce validation.",
    },
    "saved_delta": {
        "renderer": "BLENDER_EEVEE",
        "integrator": "paired saved-image delta",
        "output": "saved comparison images plus measured crops",
        "notes": "Use where the regression is about ownership or mode deltas rather than absolute diffuse GI truth.",
    },
    "same_session_fresh": {
        "renderer": "BLENDER_EEVEE",
        "integrator": "same-session versus fresh rerun",
        "output": "paired live/fresh renders or screenshots",
        "notes": "Use for viewport and render history hygiene regressions.",
    },
}

VALIDATION_MODES = {
    "headless-deterministic": {
        "execution": "background or scripted render",
        "notes": "Use when the probe can run without an interactive VIEW_3D session and should stay repeatable across reruns.",
    },
    "rendered-viewport-only": {
        "execution": "interactive rendered VIEW_3D session",
        "notes": "Use when the probe requires a live rendered viewport and does not have a meaningful headless equivalent.",
    },
    "same-session stability": {
        "execution": "interactive same-session edit and settle",
        "notes": "Use when the probe's main goal is history hygiene or post-edit refresh behavior inside one persistent session.",
    },
}

TARGET_BUCKETS = {
    "emissive_only_interior": {
        "reference_policy": "cycles_diffuse",
        "goal": "Diffuse GI must stay visibly lit from emissive-only or emissive-dominant scenes without depending on camera-visible direct lighting.",
    },
    "thin_wall_leak": {
        "reference_policy": "cycles_diffuse",
        "goal": "Thin barriers must keep the blocked side dark enough while still showing a clear lift when the barrier is removed.",
    },
    "reflected_diffuse_gi": {
        "reference_policy": "cycles_diffuse",
        "goal": "Indirect diffuse transport must survive mirror or secondary-hit views instead of dropping back to camera-dependent black.",
    },
    "direct_indirect_separation": {
        "reference_policy": "cycles_diffuse",
        "goal": "The On/Off Hardware GI contract must preserve clear indirect response when enabled without obvious direct/indirect double counting.",
    },
    "production_scene": {
        "reference_policy": "cycles_diffuse",
        "goal": "Large artist scenes must keep the intended GI/emissive response without falling over on stability or history hygiene.",
    },
}


CORPUS = [
    {
        "id": "specular-parity",
        "script": "eevee_hwrt_specular_parity.py",
        "category": "specular",
        "reference_policy": "saved_delta",
        "reference": "saved-image parity thresholds on generated reflection/refraction scenes",
        "covers": [
            "roughness response",
            "ior response",
            "modifier-driven material replay",
            "image texture reflection bindings",
        ],
    },
    {
        "id": "specular-material-color",
        "script": "eevee_hwrt_specular_material_color_probe.py",
        "category": "specular",
        "reference_policy": "saved_delta",
        "reference": "mirror reflection parity for tinted and textured metallic secondary materials",
        "metrics": [
            {"name": "METAL_TINT_DIRECT_DIFF_MEAN", "relation": ">=", "value": 0.02},
            {"name": "METAL_TINT_REFLECTION_DIFF_MEAN", "relation": ">=", "value": 0.01},
            {"name": "METAL_TEXTURE_DIRECT_DIFF_MEAN", "relation": ">=", "value": 0.02},
            {"name": "METAL_TEXTURE_REFLECTION_DIFF_MEAN", "relation": ">=", "value": 0.01},
        ],
        "covers": [
            "colored metallic secondary-hit response",
            "textured metallic secondary-hit response",
            "specular material color retention in mirror reflections",
            "texture-driven specular color retention",
        ],
    },
    {
        "id": "fast-gi-mirror",
        "script": "eevee_hwrt_fast_gi_mirror_probe.py",
        "category": "gi",
        "reference_policy": "cycles_diffuse",
        "reference": "direct and mirrored crop deltas with Fast GI on/off",
        "targets": ["emissive_only_interior", "reflected_diffuse_gi"],
        "metrics": [
            {"name": "FAST_GI_DIRECT_DIFF_MEAN", "relation": ">=", "value": 0.01},
            {"name": "FAST_GI_REFLECTED_DIFF_MEAN", "relation": ">=", "value": 0.01},
        ],
        "covers": [
            "world-space fast GI visibility",
            "reflected indirect lighting",
            "emissive bounce visibility in mirrors",
        ],
    },
    {
        "id": "fast-gi-emissive-strength",
        "script": "eevee_hwrt_fast_gi_emissive_strength_probe.py",
        "category": "gi",
        "reference_policy": "saved_delta",
        "reference": "saved-image crop delta for low versus high emissive intensity",
        "targets": ["emissive_only_interior"],
        "metrics": [
            {"name": "FAST_GI_EMISSIVE_STRENGTH_DIFF_MEAN", "relation": ">=", "value": 0.01},
        ],
        "covers": [
            "emissive strength response",
            "fast gi indirect energy scaling",
        ],
    },
    {
        "id": "fast-gi-emissive-reference",
        "script": "eevee_hwrt_fast_gi_emissive_reference_probe.py",
        "category": "gi",
        "reference_policy": "cycles_diffuse",
        "reference": "paired Eevee HWRT Fast GI versus Cycles diffuse-only emissive-interior crop",
        "targets": ["emissive_only_interior"],
        "metrics": [
            {"name": "EMISSIVE_INTERIOR_EEVEE_PATCH_LUMA", "relation": ">=", "value": 0.02},
            {"name": "EMISSIVE_INTERIOR_CYCLES_PATCH_LUMA", "relation": ">=", "value": 0.03},
            {"name": "EMISSIVE_INTERIOR_PATCH_LUMA_RATIO", "relation": ">=", "value": 0.45},
            {"name": "EMISSIVE_INTERIOR_PATCH_ABS_DIFF_MEAN", "relation": "<=", "value": 0.12},
        ],
        "covers": [
            "emissive-only interior parity against diffuse path tracing",
            "non-black indirect response behind an interior baffle",
            "same-scene Eevee versus Cycles crop comparison",
        ],
    },
    {
        "id": "fast-gi-nearfield-leak",
        "script": "eevee_hwrt_fast_gi_nearfield_leak_probe.py",
        "category": "gi",
        "reference_policy": "saved_delta",
        "reference": "paired saved-image delta for thin-wall barrier present versus removed",
        "targets": ["thin_wall_leak"],
        "metrics": [
            {"name": "NEARFIELD_LEAK_BLOCKED_MEAN", "relation": "<=", "value": 0.18},
            {"name": "NEARFIELD_LAYERED_BLOCKED_MEAN", "relation": "<=", "value": 0.14},
            {"name": "NEARFIELD_LEAK_OPEN_MEAN", "relation": ">=", "value": 0.08},
            {"name": "NEARFIELD_LEAK_DIFF_MEAN", "relation": ">=", "value": 0.03},
            {"name": "NEARFIELD_LAYERED_DIFF_MEAN", "relation": ">=", "value": 0.03},
        ],
        "covers": [
            "near-field leak rejection",
            "thin-wall barrier behavior",
            "layered occlusion barrier behavior",
            "occupancy and thickness gating regressions",
        ],
    },
    {
        "id": "many-light-reference",
        "script": "eevee_hwrt_many_light_reference_probe.py",
        "category": "gi",
        "reference_policy": "cycles_diffuse",
        "reference": "paired Eevee HWRT many-light interior direct-light crop versus Cycles direct-light-style reference",
        "targets": ["direct_indirect_separation"],
        "metrics": [
            {"name": "MANY_LIGHT_EEVEE_PATCH_LUMA", "relation": ">=", "value": 0.05},
            {"name": "MANY_LIGHT_CYCLES_PATCH_LUMA", "relation": ">=", "value": 0.08},
            {"name": "MANY_LIGHT_PATCH_LUMA_RATIO", "relation": ">=", "value": 0.65},
            {"name": "MANY_LIGHT_PATCH_ABS_DIFF_MEAN", "relation": "<=", "value": 0.18},
        ],
        "covers": [
            "many-light interior direct-light parity",
            "analytic local-light accumulation under HWRT",
            "black-world direct-light reference behavior",
        ],
    },
    {
        "id": "offscreen-light-reference",
        "script": "eevee_hwrt_offscreen_light_reference_probe.py",
        "category": "gi",
        "reference_policy": "cycles_diffuse",
        "reference": "paired Eevee HWRT off-screen and behind-camera emissive crop versus Cycles diffuse-only references",
        "targets": ["emissive_only_interior"],
        "metrics": [
            {"name": "OFFSCREEN_EEVEE_PATCH_LUMA", "relation": ">=", "value": 0.02},
            {"name": "OFFSCREEN_CYCLES_PATCH_LUMA", "relation": ">=", "value": 0.03},
            {"name": "OFFSCREEN_PATCH_LUMA_RATIO", "relation": ">=", "value": 0.35},
            {"name": "OFFSCREEN_PATCH_ABS_DIFF_MEAN", "relation": "<=", "value": 0.25},
            {"name": "BEHIND_CAMERA_EEVEE_PATCH_LUMA", "relation": ">=", "value": 0.02},
            {"name": "BEHIND_CAMERA_CYCLES_PATCH_LUMA", "relation": ">=", "value": 0.03},
            {"name": "BEHIND_CAMERA_PATCH_LUMA_RATIO", "relation": ">=", "value": 0.35},
            {"name": "BEHIND_CAMERA_PATCH_ABS_DIFF_MEAN", "relation": "<=", "value": 0.25},
        ],
        "covers": [
            "off-screen emissive contribution",
            "behind-camera emissive contribution",
            "camera-independent Fast GI response",
        ],
    },
    {
        "id": "reflected-gi",
        "script": "eevee_hwrt_reflected_gi_probe.py",
        "category": "gi",
        "reference_policy": "cycles_diffuse",
        "reference": "direct floor patch and mirrored floor patch deltas",
        "targets": ["reflected_diffuse_gi"],
        "metrics": [
            {"name": "DIRECT_FLOOR_PATCH_DIFF_MEAN", "relation": ">=", "value": 0.01},
            {"name": "REFLECTED_FLOOR_PATCH_DIFF_MEAN", "relation": ">=", "value": 0.01},
        ],
        "covers": [
            "reflected diffuse GI",
            "secondary-hit indirect reuse",
            "mirror-visible indirect response",
        ],
    },
    {
        "id": "layered-gi-modes",
        "script": "eevee_hwrt_layered_gi_modes_probe.py",
        "category": "gi",
        "reference_policy": "cycles_diffuse",
        "reference": "GI on/off luminance thresholds and no-direct-light leakage regressions",
        "targets": ["direct_indirect_separation"],
        "metrics": [
            {"name": "GI_ON_DIRECT_LUMA", "relation": ">=", "value": 0.05},
        ],
        "covers": [
            "Hardware GI on/off ownership",
            "direct/indirect separation",
            "double-lighting regressions",
        ],
    },
    {
        "id": "gi-off-environment",
        "script": "eevee_hwrt_gi_off_environment_probe.py",
        "category": "environment",
        "reference_policy": "saved_delta",
        "reference": "screen-space classic baseline versus Hardware RT GI-off environment-on diffuse world probe",
        "metrics": [
            {"name": "HWRT_ENV_SHADOW_LUMA", "relation": ">=", "value": 0.02},
            {"name": "GI_OFF_ENV_SHADOW_DIFF_MEAN", "relation": ">=", "value": 0.008},
            {"name": "HWRT_ENV_FLOOR_VARIATION", "relation": ">=", "value": 0.01},
        ],
        "covers": [
            "GI-off environment-on diffuse world fallback",
            "hardware environment visibility modulation on classic probe lighting",
            "non-black diffuse environment ownership",
        ],
    },
    {
        "id": "environment-interior",
        "script": "eevee_hwrt_environment_interior_probe.py",
        "category": "environment",
        "reference_policy": "saved_delta",
        "reference": "small room with one opening under HDRI world, checking dark GI-off interior versus GI-lit interior",
        "metrics": [
            {"name": "ENV_INTERIOR_DIRECT_GI_OFF_LUMA", "relation": "<=", "value": 0.025},
            {"name": "ENV_INTERIOR_GI_ENV_OFF_LUMA", "relation": "<=", "value": 0.025},
            {"name": "ENV_INTERIOR_GI_ON_LUMA", "relation": ">=", "value": 0.02},
            {"name": "ENV_INTERIOR_GI_LIFT", "relation": ">=", "value": 0.01},
            {"name": "ENV_INTERIOR_ENV_GATING_DELTA", "relation": ">=", "value": 0.01},
        ],
        "covers": [
            "dark interior when only direct environment is active",
            "no world-fed GI when environment admission is disabled",
            "indoor environment transport through Hardware GI",
            "reduced broad interior environment fill",
        ],
    },
    {
        "id": "textured-gi",
        "script": "eevee_hwrt_textured_gi_probe.py",
        "category": "gi",
        "reference_policy": "saved_delta",
        "reference": "shadowed receiver patch delta for UV-textured diffuse bounce with GI off versus on",
        "metrics": [
            {"name": "TEXTURED_GI_OFF_DIFF_MEAN", "relation": "<=", "value": 0.03},
            {"name": "TEXTURED_GI_ON_DIFF_MEAN", "relation": ">=", "value": 0.03},
            {"name": "TEXTURED_GI_DIFF_DELTA", "relation": ">=", "value": 0.008},
        ],
        "covers": [
            "textured diffuse secondary-hit replay",
            "uv image contribution to Hardware GI",
            "proxy-versus-hit-eval color regressions",
        ],
    },
    {
        "id": "refracted-texture",
        "script": "eevee_hwrt_refracted_texture_probe.py",
        "category": "specular",
        "reference_policy": "saved_delta",
        "reference": "glass sphere over checkerboard floor with color-swapped texture under env-lit and GI+ENV HWRT refraction",
        "metrics": [
            {"name": "REFRACTED_TEXTURE_DIFF_MEAN", "relation": ">=", "value": 0.01},
            {"name": "REFRACTED_TEXTURE_GI_ENV_DIFF_MEAN", "relation": ">=", "value": 0.01},
            {"name": "REFRACTED_TEXTURE_GI_ENV_DARKEN_DELTA", "relation": "<=", "value": 0.03},
        ],
        "covers": [
            "scene-final HWRT refraction resolved-surface reuse",
            "textured scene hits behind glass",
            "environment-only refraction through clear sphere",
            "combined GI plus environment refraction darkening regression",
            "checkerboard transmission through clear sphere",
        ],
    },
    {
        "id": "principled-reflection-live",
        "script": "eevee_hwrt_principled_reflection_live_probe.py",
        "category": "specular",
        "validation_mode": "rendered-viewport-only",
        "reference_policy": "saved_delta",
        "reference": "generated mirror scene for near-mirror Principled reflection continuity and reflected-ground retention",
        "metrics": [],
        "covers": [
            "near-mirror Principled metallic reflection continuity",
            "layered scene-final reflection restore regressions",
            "reflected ground retention on mirrored Principled secondaries",
            "0.99 versus 1.0 metallic threshold regressions",
        ],
    },
    {
        "id": "principled-reflection-test-blend",
        "script": "eevee_hwrt_principled_reflection_test_blend_probe.py",
        "category": "specular",
        "validation_mode": "rendered-viewport-only",
        "reference_policy": "saved_delta",
        "reference": "test.blend mirror scene for near-mirror Principled reflection continuity and reflected-ground retention",
        "metrics": [],
        "covers": [
            "near-mirror Principled mirror regression on scene content",
            "reflected-ground lower-band contrast on mirrored Principled hits",
            "0.88 metallic darkening regressions in reflected view",
            "near-full-metal threshold regressions at 0.99 versus 1.0",
        ],
    },
    {
        "id": "layered-gi-live",
        "script": "eevee_hwrt_layered_gi_live_probe.py",
        "category": "viewport",
        "reference_policy": "same_session_fresh",
        "reference": "same-session rendered viewport capture deltas",
        "covers": [
            "live viewport settle behavior",
            "amortized Fast GI updates",
            "interactive reset follow-up samples",
        ],
    },
    {
        "id": "fast-gi-sparse-live",
        "script": "eevee_hwrt_fast_gi_sparse_live_probe.py",
        "category": "viewport",
        "validation_mode": "rendered-viewport-only",
        "reference_policy": "saved_delta",
        "reference": "same-session rendered viewport captures before and after viewport motion",
        "covers": [
            "viewport-motion sparse brick refresh",
            "camera-invalidated Fast GI brick updates",
            "rendered VIEW_3D settle behavior after navigation",
        ],
    },
    {
        "id": "fast-gi-light-live",
        "script": "eevee_hwrt_fast_gi_light_live_probe.py",
        "category": "viewport",
        "validation_mode": "same-session stability",
        "reference_policy": "saved_delta",
        "reference": "same-session rendered viewport captures before and after a light edit",
        "covers": [
            "light invalidation",
            "same-session Fast GI refresh after light edits",
            "rendered viewport post-edit settle behavior",
        ],
    },
    {
        "id": "fast-gi-emissive-live",
        "script": "eevee_hwrt_fast_gi_emissive_live_probe.py",
        "category": "viewport",
        "validation_mode": "same-session stability",
        "reference_policy": "saved_delta",
        "reference": "same-session rendered viewport captures before and after an emissive-material edit",
        "covers": [
            "emissive invalidation",
            "same-session Fast GI refresh after emissive edits",
            "emissive edit propagation in rendered viewport",
        ],
    },
    {
        "id": "fast-gi-geometry-live",
        "script": "eevee_hwrt_fast_gi_geometry_live_probe.py",
        "category": "viewport",
        "validation_mode": "same-session stability",
        "reference_policy": "saved_delta",
        "reference": "same-session rendered viewport behavior before and after a geometry edit",
        "covers": [
            "geometry invalidation",
            "same-session Fast GI refresh after geometry edits",
            "post-edit rendered viewport settle behavior",
        ],
    },
    {
        "id": "caustics",
        "script": "eevee_hwrt_caustics_probe.py",
        "category": "indirect",
        "reference_policy": "cycles_diffuse",
        "reference": "receiver-side direct and reflected deltas across caustics sample levels",
        "metrics": [
            {"name": "CAUSTICS_DIRECT_TOGGLE_DIFF_MEAN", "relation": ">=", "value": 0.002},
            {"name": "CAUSTICS_DIRECT_SAMPLE_DIFF_MEAN", "relation": ">=", "value": 0.0005},
            {"name": "CAUSTICS_REFLECTED_TOGGLE_DIFF_MEAN", "relation": ">=", "value": 0.002},
            {"name": "CAUSTICS_REFLECTED_SAMPLE_DIFF_MEAN", "relation": ">=", "value": 0.0005},
            {"name": "CAUSTICS_CONTROL_TOGGLE_DIFF_MEAN", "relation": "<=", "value": 0.0015},
        ],
        "covers": [
            "caustics toggle behavior",
            "focused direct glass/refraction receiver energy",
            "caustics sample scaling",
            "focused reflected caustic receiver energy",
            "caustics no-diffuse-contamination control region",
        ],
    },
    {
        "id": "current-gi-caustics",
        "script": "eevee_hwrt_current_gi_caustics_probe.py",
        "category": "indirect",
        "reference_policy": "cycles_diffuse",
        "reference": "current traced GI direct and reflected caustic deltas across caustics sample levels",
        "metrics": [
            {"name": "CURRENT_GI_CAUSTICS_DIRECT_TOGGLE_DIFF_MEAN", "relation": ">=", "value": 0.002},
            {"name": "CURRENT_GI_CAUSTICS_DIRECT_SAMPLE_DIFF_MEAN", "relation": ">=", "value": 0.0005},
            {"name": "CURRENT_GI_CAUSTICS_REFLECTED_TOGGLE_DIFF_MEAN", "relation": ">=", "value": 0.002},
            {"name": "CURRENT_GI_CAUSTICS_REFLECTED_SAMPLE_DIFF_MEAN", "relation": ">=", "value": 0.0005},
            {"name": "CURRENT_GI_CAUSTICS_CONTROL_TOGGLE_DIFF_MEAN", "relation": "<=", "value": 0.0015},
        ],
        "covers": [
            "current GI on plus caustics off coexistence baseline",
            "current GI on plus caustics on direct receiver energy",
            "current GI on plus caustics on reflected receiver energy",
            "current GI plus caustics no-diffuse-contamination control region",
        ],
    },
    {
        "id": "caustics-feature-toggle",
        "script": "eevee_hwrt_caustics_feature_toggle_probe.py",
        "category": "indirect",
        "reference_policy": "cycles_diffuse",
        "reference": "caustics scene with reflection/refraction toggled against the full caustics path",
        "metrics": [
            {"name": "CAUSTICS_REFRACTION_TOGGLE_DIRECT_DIFF_MEAN", "relation": ">=", "value": 0.002},
            {"name": "CAUSTICS_REFLECTION_TOGGLE_REFLECTED_DIFF_MEAN", "relation": ">=", "value": 0.002},
        ],
        "covers": [
            "caustics plus refraction toggle fallback behavior",
            "caustics plus reflection toggle fallback behavior",
            "feature toggles with caustics enabled",
        ],
    },
    {
        "id": "caustics-environment-toggle",
        "script": "eevee_hwrt_caustics_environment_toggle_probe.py",
        "category": "indirect",
        "reference_policy": "cycles_diffuse",
        "reference": "caustics scene with environment on versus off under a directional HDRI-style world",
        "metrics": [
            {"name": "CAUSTICS_ENVIRONMENT_TOGGLE_DIRECT_DIFF_MEAN", "relation": ">=", "value": 0.0010},
            {"name": "CAUSTICS_ENVIRONMENT_TOGGLE_REFLECTED_DIFF_MEAN", "relation": ">=", "value": 0.0010},
        ],
        "covers": [
            "caustics plus environment toggle fallback behavior",
            "direct caustic receiver response to environment enablement",
            "reflected caustic receiver response to environment enablement",
        ],
    },
    {
        "id": "live-update",
        "script": "eevee_hwrt_live_update_probe.py",
        "category": "viewport",
        "reference_policy": "same_session_fresh",
        "reference": "same-session viewport response after scene edits",
        "covers": [
            "history invalidation",
            "post-edit update propagation",
        ],
    },
    {
        "id": "live-texture",
        "script": "eevee_hwrt_live_texture_probe.py",
        "category": "viewport",
        "reference_policy": "same_session_fresh",
        "reference": "same-session viewport response after texture/material edits",
        "covers": [
            "material invalidation",
            "texture-driven update propagation",
        ],
    },
    {
        "id": "camera-rotation-fresh",
        "script": "eevee_hwrt_camera_rotation_fresh_probe.py",
        "category": "render",
        "reference_policy": "same_session_fresh",
        "reference": "same-session camera rotate-away/return parity against a fresh A-state rerun",
        "metrics": [
            {"name": "CAMERA_ROTATION_FRAME_DIFF_MEAN", "relation": ">=", "value": 0.05},
            {"name": "CAMERA_ROTATION_RETURN_VS_FRESH_DIFF_MEAN", "relation": "<=", "value": 0.05},
            {"name": "CAMERA_ROTATION_RETURN_VS_INITIAL_DIFF_MEAN", "relation": "<=", "value": 0.05},
        ],
        "covers": [
            "camera-rotation stability",
            "same-session return-to-view parity",
            "no lighting truth rewrite on camera motion",
        ],
    },
    {
        "id": "animation-response-light",
        "script": "eevee_hwrt_animation_response_probe.py",
        "category": "render",
        "reference_policy": "same_session_fresh",
        "reference": "same-session light-edit frame 1 and frame 2 versus fresh B-state rerun (`--probe light`)",
        "metrics": [
            {"name": "LIGHT_ANIMATION_STATE_CHANGE_DIFF_MEAN", "relation": ">=", "value": 0.03},
            {"name": "LIGHT_ANIMATION_FRAME1_VS_FRESH_DIFF_MEAN", "relation": "<=", "value": 0.02},
            {"name": "LIGHT_ANIMATION_FRAME2_VS_FRESH_DIFF_MEAN", "relation": "<=", "value": 0.02},
        ],
        "covers": [
            "animated-light response time",
            "same-session Fast GI update catch-up after a light edit",
            "post-edit render-frame freshness for lighting changes",
        ],
    },
    {
        "id": "animation-response-geometry",
        "script": "eevee_hwrt_animation_response_probe.py",
        "category": "render",
        "reference_policy": "same_session_fresh",
        "reference": "same-session geometry-edit frame 1 and frame 2 versus fresh B-state rerun (`--probe geometry`)",
        "metrics": [
            {"name": "GEOMETRY_ANIMATION_STATE_CHANGE_DIFF_MEAN", "relation": ">=", "value": 0.05},
            {"name": "GEOMETRY_ANIMATION_FRAME1_VS_FRESH_DIFF_MEAN", "relation": "<=", "value": 0.02},
            {"name": "GEOMETRY_ANIMATION_FRAME2_VS_FRESH_DIFF_MEAN", "relation": "<=", "value": 0.02},
        ],
        "covers": [
            "animated-geometry response time",
            "same-session Fast GI update catch-up after a geometry edit",
            "post-edit render-frame freshness for occluder changes",
        ],
    },
    {
        "id": "streaming-memory",
        "script": "eevee_hwrt_streaming_memory_probe.py",
        "category": "render",
        "reference_policy": "saved_delta",
        "reference": "renderer telemetry under default versus tight Fast GI memory budgets on a large procedural scene",
        "targets": ["production_scene"],
        "metrics": [
            {"name": "LARGE_SCENE_DEFAULT_MEMORY_LIMITED", "relation": "<=", "value": 0},
            {"name": "LARGE_SCENE_TIGHT_MEMORY_LIMITED", "relation": ">=", "value": 1},
            {"name": "LARGE_SCENE_ACTIVE_CASCADES_DROP", "relation": ">=", "value": 1},
            {"name": "LARGE_SCENE_TIGHT_FAST_GI_MEM_MIB", "relation": "<=", "value": 1.0},
        ],
        "covers": [
            "large-scene Fast GI residency fitting",
            "memory-budget-driven cascade eviction",
            "streaming and memory telemetry regressions",
        ],
    },
    {
        "id": "render-frame-live-fresh",
        "script": "eevee_hwrt_render_frame_live_fresh_probe.py",
        "category": "render",
        "reference_policy": "same_session_fresh",
        "reference": "same-session render frame parity against a fresh identical render",
        "covers": [
            "history reset correctness",
            "render-frame state hygiene",
        ],
    },
    {
        "id": "plane-mirror-live",
        "script": "eevee_hwrt_plane_mirror_live_probe.py",
        "category": "viewport",
        "reference_policy": "same_session_fresh",
        "reference": "viewport screenshot-class reflected update deltas",
        "covers": [
            "mirror live updates",
            "reflected modifier response",
        ],
    },
    {
        "id": "barbershop-live",
        "script": "eevee_hwrt_barbershop_live_probe.py",
        "category": "production",
        "reference_policy": "cycles_diffuse",
        "reference": "rendered viewport behavior on the emissive-heavy barbershop scene",
        "targets": ["emissive_only_interior", "production_scene"],
        "metrics": [],
        "covers": [
            "production-scene stability",
            "emissive-only interior response",
            "settled viewport lag regressions",
        ],
    },
    {
        "id": "reflection-visual-proof",
        "script": "eevee_hwrt_reflection_visual_proof.py",
        "category": "production",
        "reference_policy": "saved_delta",
        "reference": "saved-image visual proof on reflection-heavy content",
        "covers": [
            "reflection ownership",
            "continuation lighting sanity",
        ],
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List the fixed Eevee HWRT validation corpus.")
    parser.add_argument(
        "--category",
        choices=sorted({entry["category"] for entry in CORPUS}),
        default=None,
        help="Limit the output to one corpus category.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Choose text or json output.",
    )
    return parser.parse_args()


def _filtered_entries(category: str | None):
    if category is None:
        return CORPUS
    return [entry for entry in CORPUS if entry["category"] == category]


def _entry_with_path(entry):
    resolved = dict(entry)
    resolved["path"] = str(TESTS_DIR / entry["script"])
    resolved["exists"] = (TESTS_DIR / entry["script"]).exists()
    resolved["reference_policy_details"] = REFERENCE_POLICIES[entry["reference_policy"]]
    if "validation_mode" in entry:
        resolved["validation_mode_details"] = VALIDATION_MODES[entry["validation_mode"]]
    resolved["targets"] = entry.get("targets", [])
    resolved["target_details"] = [TARGET_BUCKETS[target] for target in resolved["targets"]]
    resolved["metrics"] = entry.get("metrics", [])
    return resolved


def _print_text(entries):
    for entry in entries:
        print(f"{entry['id']}: {entry['script']}")
        print(f"  category: {entry['category']}")
        print(f"  reference: {entry['reference']}")
        print(f"  reference_policy: {entry['reference_policy']}")
        print(f"  reference_renderer: {entry['reference_policy_details']['renderer']}")
        print(f"  reference_integrator: {entry['reference_policy_details']['integrator']}")
        if "validation_mode" in entry:
            print(f"  validation_mode: {entry['validation_mode']}")
            print(f"  validation_execution: {entry['validation_mode_details']['execution']}")
        print(f"  exists: {entry['exists']}")
        if entry["targets"]:
            print(f"  targets: {', '.join(entry['targets'])}")
        if entry["metrics"]:
            print("  metrics:")
            for metric in entry["metrics"]:
                print(f"    {metric['name']} {metric['relation']} {metric['value']:.6f}")
        print(f"  covers: {', '.join(entry['covers'])}")


def main() -> int:
    args = _parse_args()
    entries = [_entry_with_path(entry) for entry in _filtered_entries(args.category)]
    if args.format == "json":
        print(json.dumps(entries, indent=2, sort_keys=True))
    else:
        _print_text(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
