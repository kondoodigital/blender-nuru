/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Compile shader files as C++ inside one compilation unit to lint syntax and get IDE integration.
 * Vanilla 5.1.1: only the four shadow-page BSL files upstream ships.
 */

#include "eevee_shadow_page_allocate.bsl.hh" /* IWYU pragma: export */
#include "eevee_shadow_page_defrag.bsl.hh"   /* IWYU pragma: export */
#include "eevee_shadow_page_free.bsl.hh"     /* IWYU pragma: export */
#include "eevee_shadow_page_ops.bsl.hh"      /* IWYU pragma: export */

void main() {}
