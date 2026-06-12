#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Fail if macos-metal drifts from vanilla 5.1.1 in volumetric paths or carries
# forbidden 5.2-alpha BSL bleed. Run from repo root:
#   ./scripts/ops/nuru_51_diff_gate.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

UPSTREAM="${NURU_51_UPSTREAM:-origin/blender-v5.1-release}"
EEVEE_SHADERS="source/blender/draw/engines/eevee/shaders"
MAX_BSL_HH="${NURU_MAX_EEVEE_BSL_HH:-4}"

errors=0

err() {
  echo "nuru_51_diff_gate: ERROR: $*" >&2
  errors=$((errors + 1))
}

if ! git rev-parse --verify "$UPSTREAM" >/dev/null 2>&1; then
  err "upstream ref '$UPSTREAM' not found"
  exit 1
fi

# Forbidden: Nuru-only occupancy BSL (5.1 uses .glsl only).
if [[ -f "$EEVEE_SHADERS/eevee_occupancy_lib.bsl.hh" ]]; then
  err "$EEVEE_SHADERS/eevee_occupancy_lib.bsl.hh must not exist (use eevee_occupancy_lib.glsl)"
fi

# Nuru-modified eevee shaders must not be accidentally reverted to vanilla 5.1.1.
HEAD_REF="${NURU_HEAD_REF:-HEAD}"
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  if ! git cat-file -e "$HEAD_REF:$rel" 2>/dev/null; then
    continue
  fi
  if git diff --quiet "$HEAD_REF" "$UPSTREAM" -- "$rel" 2>/dev/null; then
    continue
  fi
  if git diff --quiet "$UPSTREAM" -- "$rel" 2>/dev/null; then
    # HEAD may differ from upstream only by mechanical 5.1 port lines; matching vanilla is OK then.
    non_mech="$(git diff "$HEAD_REF" "$UPSTREAM" -- "$rel" 2>/dev/null |
      grep -E '^[+-]' | grep -v '^[+-][+-][+-]' |
      grep -vE 'bsl\.hh|\.glsl|Thickness|read_thickness|0\.0f|SphericalHarmonic|spherical_harmonics' || true)"
    if [[ -n "$non_mech" ]]; then
      err "$rel matches vanilla $UPSTREAM but $HEAD_REF had non-mechanical Nuru changes"
    fi
  fi
done < <(git diff "$HEAD_REF" "$UPSTREAM" --name-only -- "$EEVEE_SHADERS"/*.glsl 2>/dev/null || true)

# eevee .bsl.hh count ceiling (5.1 has 4 shadow page files + Nuru HWRT minimum).
bsl_count="$(find "$EEVEE_SHADERS" -maxdepth 1 -name '*.bsl.hh' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$bsl_count" -gt "$MAX_BSL_HH" ]]; then
  err "eevee .bsl.hh count $bsl_count exceeds limit $MAX_BSL_HH"
fi

# Forbidden 5.2-era GLSL APIs in eevee (see docs/nuru_shaders.md in NuruNew).
# Use fixed-string grep for patterns containing regex metacharacters like '('.
declare -a FORBIDDEN_FIXED=(
  'Thickness::'
  'Thickness thickness'
  'spherical_harmonics::'
  'lightprobe_load(float2'
  'eevee_light_culling_cull'
  'eevee_raytracing_denoise'
)
for pattern in "${FORBIDDEN_FIXED[@]}"; do
  if grep -rF "$pattern" "$EEVEE_SHADERS" --include='*.glsl' -q 2>/dev/null; then
    err "forbidden 5.2 pattern in eevee .glsl: $pattern"
    grep -rFn "$pattern" "$EEVEE_SHADERS" --include='*.glsl' 2>/dev/null | head -5 >&2
  fi
done
if grep -rE '#include ".*\.bsl\.hh"' "$EEVEE_SHADERS" --include='*.glsl' -q 2>/dev/null; then
  err 'forbidden 5.2 pattern in eevee .glsl: #include "*.bsl.hh"'
  grep -rEn '#include ".*\.bsl\.hh"' "$EEVEE_SHADERS" --include='*.glsl' 2>/dev/null | head -5 >&2
fi

# HEAD naming: shader registration must match 5.1 create-info / file basenames.
EEVEE_SHADER_CC="source/blender/draw/engines/eevee/eevee_shader.cc"
for pattern in 'eevee_light_culling_cull' 'eevee_raytracing_denoise'; do
  if grep -q "$pattern" "$EEVEE_SHADER_CC" 2>/dev/null; then
    err "$EEVEE_SHADER_CC uses 5.2 registration name '$pattern' (HEAD bug; use eevee_light_culling_select / eevee_ray_denoise_*)"
  fi
done

# HWRT hit-eval requires extended MeshVertex (vertex_indices + barycentric_weights).
if ! grep -q 'vertex_indices' "$EEVEE_SHADERS/eevee_geom_types_lib.glsl" 2>/dev/null; then
  err "$EEVEE_SHADERS/eevee_geom_types_lib.glsl missing HWRT MeshVertex fields"
fi

# Lint-only BSL must not remain in tree.
for forbidden in \
  eevee_camera_lib.bsl.hh \
  eevee_light_culling.bsl.hh \
  eevee_light_shadow_setup.bsl.hh \
  eevee_lut_comp.bsl.hh \
  eevee_ray_denoise.bsl.hh \
  eevee_subsurface.bsl.hh; do
  if [[ -f "$EEVEE_SHADERS/$forbidden" ]]; then
    err "$EEVEE_SHADERS/$forbidden must be removed (lint-only 5.2 bleed)"
  fi
done

if [[ "$errors" -gt 0 ]]; then
  echo "nuru_51_diff_gate: $errors check(s) failed" >&2
  exit 1
fi

echo "nuru_51_diff_gate: OK (upstream=$UPSTREAM, eevee_bsl_hh=$bsl_count)"
exit 0
