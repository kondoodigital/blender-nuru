/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/eevee_tracing_infos.hh"

COMPUTE_SHADER_CREATE_INFO(eevee_ray_hit_eval_prefix)

void main()
{
  const uint entry_index = gl_GlobalInvocationID.x;
  if (entry_index >= uint(scene_entry_count)) {
    return;
  }

  uint prefix_sum = 0u;
  for (uint i = 0u; i < entry_index; i++) {
    prefix_sum += hit_eval_count_buf[i];
  }

  const uint hit_count = hit_eval_count_buf[entry_index];
  hit_eval_offset_buf[entry_index] = prefix_sum;
  hit_eval_cursor_buf[entry_index] = 0u;

  DrawCommandArray draw_cmd;
  draw_cmd.vertex_len = hit_count * 3u;
  draw_cmd.instance_len = (hit_count > 0u) ? 1u : 0u;
  draw_cmd.vertex_first = prefix_sum * 3u;
  draw_cmd.instance_first = 0u;
  draw_cmd._pad0 = 0u;
  draw_cmd._pad1 = 0u;
  draw_cmd._pad2 = 0u;
  draw_cmd._pad3 = 0u;
  hit_eval_indirect_draw_buf[entry_index] = draw_cmd;
}
