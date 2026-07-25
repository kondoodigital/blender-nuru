# Nuru On macOS

Blender-Nuru for macOS runs Eevee on the Metal graphics backend with Nuru hardware ray tracing on Apple Silicon. This page covers requirements, installation, and macOS-specific notes.

## Requirements

| Item | Requirement |
| --- | --- |
| Operating system | macOS 14 or newer |
| Hardware | Apple Silicon M3 or newer |
| Unsupported hardware | M1 and M2 do not provide the required hardware ray tracing support; Intel Macs are not supported |
| Graphics backend | Metal, built into the package; no extra setup is needed |

## Install

### DMG (recommended)

1. Download **`Blender-Nuru-macos-<version>.dmg`** from the [Releases page](https://github.com/kondoodigital/blender-nuru/releases) and open it.
2. Drag **`Blender-Nuru.app`** into your **Applications** folder.
3. Eject the disk image.
4. Start **Blender-Nuru** from Applications or Launchpad.

The disk image and application are Developer ID signed by **Kondoo Digital GmbH** and notarized by Apple.

### Portable zip

1. Download the macOS zip from the [Releases page](https://github.com/kondoodigital/blender-nuru/releases) and unzip it.
2. Drag **`Blender-Nuru.app`** into your **Applications** folder.
3. Start **Blender-Nuru** from Applications or Launchpad.

### Gatekeeper

The macOS application uses Apple Developer ID signing, hardened runtime, a secure timestamp, and Apple notarization. The DMG is separately signed, notarized, and stapled. Gatekeeper should identify the developer as **Kondoo Digital GmbH** and open the application normally. If macOS cannot verify a download, delete it and download a fresh copy from the official GitHub Releases page rather than bypassing Gatekeeper.

## Enable Nuru

1. Open **Render Properties**.
2. Set the render engine to **Eevee**.
3. Enable **Raytracing**.
4. Set **Method** to **Nuru Raytracing**.
5. Use **Quick Settings** to enable the RT features you need.

The Nuru **Quick Settings** appear when the active Mac supports the Nuru backend. If they stay hidden, confirm the machine is an Apple Silicon M3 or newer on macOS 14 or newer.

## GPU Denoising

Denoising on macOS runs through **OpenImageDenoise** on the GPU. The **Denoiser** option's **OptiX Denoiser** entry is NVIDIA-only; selecting it on a Mac simply keeps OpenImageDenoise active, so the setting is always safe to touch.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Nuru Quick Settings do not appear | Confirm Apple Silicon M3 or newer and macOS 14 or newer. M1/M2 are not supported. |
| Gatekeeper cannot verify the DMG or app | Delete the downloaded copy and download it again from the official GitHub Releases page. Do not use a copy modified by a third party. |
| macOS identifies an unexpected developer | Cancel the launch and confirm that the download came from the official `kondoodigital/blender-nuru` release. The expected developer is **Kondoo Digital GmbH**. |
| The first rendered frame is slow | Shaders and pipelines warm up on the first render after installation; later frames are much faster. |

## Related

- [Nuru Overview](nuru_overview.html)
- [UI Reference](nuru_ui_reference.html)
- [Shadows And Lights](nuru_shadows_and_lights.html)
- [Dos And Don'ts](nuru_dos_and_donts.html)
- [Materials](nuru_materials.html)
