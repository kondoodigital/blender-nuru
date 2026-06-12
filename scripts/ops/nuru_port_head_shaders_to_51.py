#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH
# SPDX-License-Identifier: GPL-2.0-or-later
"""Port eevee .glsl from HEAD (DIAMOND 45) with 5.1.1 mechanical edits only."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAD = "HEAD"
UPSTREAM = "origin/blender-v5.1-release"
SHADER_GLOB = "source/blender/draw/engines/eevee/shaders/*.glsl"


def git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{HEAD}:{path}"],
        cwd=ROOT,
        text=True,
    )


def port_thickness_value_calls(content: str) -> str:
    """Map BSL Thickness.value() to 5.1 signed float thickness without breaking slab/sphere."""
    gbuffer_thickness = re.compile(
        r"gbuffer::read_thickness\((.+?)\)\.value\(\)", re.DOTALL
    )
    while True:
        content, count = gbuffer_thickness.subn(
            r"gbuffer::read_thickness(\1)", content
        )
        if count == 0:
            break
    gbuffer_read_thickness = re.compile(
        r"gbuffer_read_thickness\((.+?)\)\.value\(\)", re.DOTALL
    )
    while True:
        content, count = gbuffer_read_thickness.subn(
            r"gbuffer_read_thickness(\1)", content
        )
        if count == 0:
            break
    content = re.sub(
        r"(\w+)\.value\(\)(\s*!=\s*0(?:\.0f)?f?)",
        r"\1\2",
        content,
    )
    content = re.sub(
        r"(\w+)\.value\(\)(\s*==\s*0(?:\.0f)?f?)",
        r"\1\2",
        content,
    )
    content = re.sub(
        r"hit_distance > (\w+)\.value\(\)",
        r"hit_distance > abs(\1)",
        content,
    )
    content = re.sub(
        r"hit_distance < (\w+)\.value\(\)",
        r"hit_distance < abs(\1)",
        content,
    )
    content = re.sub(
        r"shadow_thickness < (\w+)\.value\(\)\)",
        r"shadow_thickness < abs(\1))",
        content,
    )
    content = re.sub(
        r"subsurface_transmission\(([^,]+),\s*(\w+)\.value\(\)\)",
        r"subsurface_transmission(\1, abs(\2))",
        content,
    )
    content = re.sub(
        r"\* (\w+)\.value\(\) \*",
        r"* abs(\1) *",
        content,
    )
    content = re.sub(
        r"min\((\w+)\.value\(\),",
        r"min(abs(\1),",
        content,
    )
    content = re.sub(
        r"is_directional \? (\w+)\.value\(\)",
        r"is_directional ? abs(\1)",
        content,
    )
    content = re.sub(r"\b(\w+)\.value\(\)", r"abs(\1)", content)
    return content


def list_nuru_shaders() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", HEAD, UPSTREAM, "--name-only", "--", SHADER_GLOB],
        cwd=ROOT,
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def mechanical_port(content: str) -> str:
    content = content.replace(
        '"eevee_spherical_harmonics.bsl.hh"',
        '"eevee_spherical_harmonics_lib.glsl"',
    )
    content = content.replace('.bsl.hh"', '.glsl"')
    content = content.replace(
        '"eevee_spherical_harmonics.glsl"',
        '"eevee_spherical_harmonics_lib.glsl"',
    )

    content = content.replace(
        "ThicknessMode thickness_mode = ThicknessMode::Slab;",
        "float thickness_mode = -1.0f;",
    )
    content = content.replace(
        "ThicknessMode thickness_mode = ThicknessMode::Sphere;",
        "float thickness_mode = 1.0f;",
    )
    content = re.sub(
        r"float::from\(\s*([^,]+)\s*,\s*thickness_mode\s*\)",
        r"\1 * thickness_mode",
        content,
    )
    content = re.sub(
        r"Thickness::from\(\s*([^,]+)\s*,\s*ThicknessMode::Sphere\s*\)",
        r"\1",
        content,
    )
    content = re.sub(
        r"Thickness::from\(\s*([^,]+)\s*,\s*ThicknessMode::Slab\s*\)",
        r"(-(\1))",
        content,
    )
    content = re.sub(
        r"Thickness::from\(\s*([^,]+)\s*,\s*thickness_mode\s*\)",
        r"(\1 * thickness_mode)",
        content,
    )
    content = re.sub(
        r"Thickness::from\(\s*([^,]+)\s*,\s*([\w]+)\.mode\(\)\s*\)",
        r"((\2 > 0.0f) ? (\1) : ((\2 < 0.0f) ? -(\1) : 0.0f))",
        content,
    )
    content = content.replace("Thickness::zero()", "0.0f")
    content = re.sub(
        r"([\w]+)\.mode\(\)\s*==\s*ThicknessMode::Slab",
        r"(\1 < 0.0f)",
        content,
    )
    content = re.sub(
        r"([\w]+)\.mode\(\)\s*==\s*ThicknessMode::Sphere",
        r"(\1 > 0.0f)",
        content,
    )
    content = re.sub(
        r"return\s+([\w]+)\.mode\(\)\s*==\s*ThicknessMode::Slab",
        r"return (\1 < 0.0f)",
        content,
    )

    content = re.sub(r"\bThickness\b(?!Isect|Mode)", "float", content)
    content = re.sub(
        r"([\w]+(?:\.[\w]+)*)\.evaluate_lambert\(([^)]+)\)",
        r"spherical_harmonics_evaluate_lambert(\2, \1)",
        content,
    )
    content = port_thickness_value_calls(content)

    content = re.sub(r"SphericalHarmonicL([012])<float4>", r"SphericalHarmonicL\1", content)
    def _encode_method_replace(match: re.Match[str]) -> str:
        var = match.group(1)
        args = match.group(2)
        comma = args.rfind(",")
        if comma == -1:
            return match.group(0)
        direction = args[:comma].strip()
        amplitude = args[comma + 1 :].strip()
        return f"spherical_harmonics_encode_signal_sample({direction}, {amplitude}, {var})"

    content = re.sub(
        r"(\w+)\.encode_signal_sample\((.+)\)",
        _encode_method_replace,
        content,
    )
    content = content.replace(
        "SphericalHarmonicL1 sh = {};",
        "SphericalHarmonicL1 sh;\n"
        "  sh.L0.M0 = float4(0.0f);\n"
        "  sh.L1.Mn1 = float4(0.0f);\n"
        "  sh.L1.M0 = float4(0.0f);\n"
        "  sh.L1.Mp1 = float4(0.0f);",
    )

    content = content.replace(
        "spherical_harmonics::clamp_energy",
        "spherical_harmonics_clamp",
    )
    content = re.sub(
        r"spherical_harmonics::(\w+)_coef",
        r"spherical_harmonics_\1",
        content,
    )
    content = re.sub(
        r"spherical_harmonics::(\w+)",
        r"spherical_harmonics_\1",
        content,
    )
    content = content.replace(
        "spherical_harmonics_clamp_energy",
        "spherical_harmonics_clamp",
    )
    return content


def post_check_fixes(path: str, content: str) -> str:
    """Fixes that depend on final file content after bulk transforms."""
    if path.endswith("eevee_lightprobe_eval_lib.glsl"):
        content = content.replace(
            "return lightprobe_load(gl_GlobalInvocationID.xy, P, Ng, V);",
            "return lightprobe_load(P, Ng, V);",
        )
    return content


def main() -> int:
    paths = list_nuru_shaders()
    for rel in paths:
        raw = git_show(rel)
        ported = post_check_fixes(rel, mechanical_port(raw))
        dest = ROOT / rel
        dest.write_text(ported, encoding="utf-8")
        print(f"ported {rel}")
    print(f"nuru_port_head_shaders_to_51: OK ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
