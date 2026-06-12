/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * NVIDIA OptiX denoiser backend for the Nuru HWRT denoise step (Windows/Vulkan).
 *
 * This is a drop-in alternative for the OIDN filter execution only: the pack/unpack kernels,
 * the exportable Vulkan buffers, and every call site stay identical. The OptiX denoiser imports
 * the same `OPAQUE_WIN32` allocations through CUDA external memory and denoises them in place,
 * zero-copy. On any unavailability or failure the caller falls back to the OIDN filter.
 */

#pragma once

#ifdef WITH_NURU_OPTIX_DENOISER

#  include <cstddef>

namespace blender::gpu::nuru_optix {

/** One packed float3 plane backed by an exportable Vulkan allocation. */
struct DenoisePlane {
  /** NT handle exported via `vkGetMemoryWin32HandleKHR` (owned by the RTBuffer; not adopted). */
  void *win32_handle = nullptr;
  /** Size of the backing `VkDeviceMemory` allocation in bytes. */
  size_t alloc_size = 0;
};

struct DenoiseParams {
  DenoisePlane color;
  DenoisePlane output;
  /** Optional guides; a null `win32_handle` disables the guide. */
  DenoisePlane albedo;
  DenoisePlane normal;
  int width = 0;
  int height = 0;
};

/**
 * Run the OptiX HDR denoiser on the packed planes. Returns false when OptiX/CUDA is unavailable
 * or any step fails; the first failure disables the path for the session (single stderr notice)
 * so the caller's OIDN fallback takes over permanently.
 *
 * Must be called with the Vulkan queue idle and externally serialized against queue submission
 * (same Xid 109 contract as the GPU OIDN filter).
 */
bool denoise(const DenoiseParams &params);

/** Release every CUDA/OptiX resource (denoiser, context, stream, scratch). Safe to call when
 * nothing was initialized. */
void free_resources();

}  // namespace blender::gpu::nuru_optix

#endif /* WITH_NURU_OPTIX_DENOISER */
