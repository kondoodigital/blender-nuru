# Nuru On Linux

Blender-Nuru for Linux runs Eevee on the Vulkan graphics backend with Nuru hardware ray tracing on NVIDIA RTX GPUs. This page covers requirements, installation, and Linux-specific notes.

## Requirements

| Item | Requirement |
| --- | --- |
| Operating system | 64-bit Linux. The `.deb` package targets Ubuntu 22.04+ / Debian 12+; the portable archive runs on any current distribution |
| GPU | NVIDIA GeForce RTX class with hardware ray tracing (RTX 20 series or newer); validated on RTX 50 series |
| GPU driver | NVIDIA proprietary driver, 595 series or newer |
| Graphics backend | Vulkan, built into the package; no extra setup is needed |
| Other GPUs | AMD and Intel GPUs are not yet validated for Nuru hardware ray tracing |

## Install

### Debian package (recommended on Ubuntu and Debian)

1. Download the Linux `.deb` (`Blender-Nuru-linux-<version>.deb`) from the [Releases page](https://github.com/kondoodigital/blender-nuru/releases).
2. Install it: `sudo apt install ./Blender-Nuru-linux-<version>.deb`
3. Start **Blender-Nuru** from your application menu, or run `blender-nuru` in a terminal. The package installs to `/opt/blender-nuru` and can be removed any time with `sudo apt remove blender-nuru`.

### Portable archive

1. Download the Linux `.tar.gz` from the [Releases page](https://github.com/kondoodigital/blender-nuru/releases).
2. Unpack it anywhere you like, for example `~/blender-nuru`.
3. Start **`./blender-nuru`** from the unpacked folder. Running it from a terminal also shows log output, which is useful for reporting problems.

## Enable Nuru

1. Open **Render Properties**.
2. Set the render engine to **Eevee**.
3. Enable **Raytracing**.
4. Set **Method** to **Nuru Raytracing**.
5. Use **Quick Settings** to enable the RT features you need.

The Nuru **Quick Settings** appear when the active GPU supports the Nuru backend. If they stay hidden, update the NVIDIA driver and confirm the RTX GPU is the active render device.

## GPU Denoising

Denoising on Linux runs on the GPU, and the denoiser shares its working memory directly with the renderer (zero-copy). The **Denoise** option is fast at every trace resolution and is safe to keep enabled for interactive work and final renders.

## Cycles On Linux

This package also includes Cycles GPU kernels for NVIDIA **CUDA** and **OptiX**. Enable them under **Edit > Preferences > System > Cycles Render Devices**.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Nuru Quick Settings do not appear | Update the NVIDIA driver to the 595 series or newer and confirm an RTX GPU is active. |
| The app does not start, or reports a Vulkan device error | Update the NVIDIA driver and make sure the proprietary driver (not Nouveau) is in use, then try again. |
| The first rendered frame is slow | Shaders and kernels warm up on the first render after installation; later frames are much faster. |
| The `.deb` fails to install on an older distribution | The package uses zstd compression and needs Ubuntu 22.04+ / Debian 12+. Use the portable `.tar.gz` instead. |

## Related

- [Nuru Overview](nuru_overview.html)
- [UI Reference](nuru_ui_reference.html)
- [Shadows And Lights](nuru_shadows_and_lights.html)
- [Materials](nuru_materials.html)
