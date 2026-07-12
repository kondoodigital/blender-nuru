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

### Installer DMG (recommended)

1. Download **`Blender-Nuru-macos-<version>.dmg`** from the [Releases page](https://github.com/kondoodigital/blender-nuru/releases) and open it.
2. Double-click **Install Blender-Nuru**.
3. The installer copies **`Blender-Nuru.app`** to **`/Applications`**, clears the download quarantine flag, and registers the app with Launchpad.
4. Start **Blender-Nuru** from Applications or Launchpad.

If macOS blocks the installer itself on first open, right-click **Install Blender-Nuru** and choose **Open** once.

### Portable zip

1. Download the macOS zip from the [Releases page](https://github.com/kondoodigital/blender-nuru/releases) and unzip it.
2. Double-click **Install Blender-Nuru** inside the unzipped folder, or install manually: move **`Blender-Nuru.app`** to **`/Applications`** and clear the quarantine attribute once from a terminal (see below).
3. Start **Blender-Nuru** from Applications or Launchpad.

### Gatekeeper

The build is not code-signed, so Gatekeeper may report the app as damaged on first start if the quarantine flag was not cleared. The installer clears it for you; for a manual install, remove it once from a terminal:

```sh
xattr -dr com.apple.quarantine /Applications/Blender-Nuru.app
```

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
| Gatekeeper reports the app as damaged | Run the `xattr` command above once, then start the app again. |
| macOS still refuses to open the app after the `xattr` command | Re-download the archive, replace the app in `/Applications` with a fresh copy, and run the command again before the first launch. |
| The first rendered frame is slow | Shaders and pipelines warm up on the first render after installation; later frames are much faster. |

## Related

- [Nuru Overview](nuru_overview.html)
- [UI Reference](nuru_ui_reference.html)
- [Dos And Don'ts](nuru_dos_and_donts.html)
- [Materials](nuru_materials.html)
