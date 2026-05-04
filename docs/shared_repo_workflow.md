<!-- SPDX-FileCopyrightText: 2026 Kondoo Digital GmbH -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Shared Repo Workflow

Use this guide when the same Nuru source tree is accessed from macOS, Linux, or
Windows, including over SMB.

## Current Support Boundary

- Shared source is supported.
- Shared build directories are not supported.
- Nuru Hardware RT viewport support in Diamond 1 is Metal-only.
- OptiX and CUDA Nuru backend support are Work in Progress.
- Linux and Windows can use the source tree for ordinary Blender development,
  inspection, and non-HWRT work.

## Core Rules

- Treat the source tree and build trees as separate concerns.
- Only one machine should actively write the working tree at a time.
- Keep generated build trees under `builds/`.
- Never reuse one build directory across operating systems.
- Preserve unrelated platform-specific outputs.
- Never rewrite another OS's CMake cache in place.
- Keep validation outputs outside the shared repo when possible.

## Standard Build Roots

- macOS development: `builds/macos-dev`
- macOS release: `builds/macos-release`
- Linux development: `builds/linux-dev`
- Linux release: `builds/linux-release`
- Windows development: `builds/windows-dev`
- Windows release: `builds/windows-release`

The older generic names `builds/dev` and `builds/release` are not the documented
defaults for this shared tree because they do not encode the owning OS.

## macOS

The active Nuru HWRT backend is macOS Metal.

Configure:

```sh
cmake -S . -B builds/macos-dev -G Ninja
```

Focused Eevee/GPU build:

```sh
cmake --build builds/macos-dev --target bf_gpu bf_draw blender -j 12
```

Launch path:

```sh
builds/macos-dev/bin/Blender.app/Contents/MacOS/Blender
```

Install runtime assets after `scripts/startup/*` edits:

```sh
cmake --install builds/macos-dev
```

## Linux

Linux build trees are for Linux-local Blender work in this shared source tree.
They are not active Nuru HWRT validation targets in Diamond 1 while OptiX and
CUDA Nuru backend support are Work in Progress.

Configure:

```sh
cmake -S . -B builds/linux-dev -G Ninja
```

Focused Eevee/GPU build:

```sh
cmake --build builds/linux-dev --target bf_gpu bf_draw blender -j 12
```

Launch path:

```sh
builds/linux-dev/bin/blender
```

Do not point Linux CMake at `builds/macos-dev` or `builds/macos-release`.

## Windows

Windows build trees are for Windows-local Blender work in this shared source
tree. They are not active Nuru HWRT validation targets in Diamond 1 while OptiX
and CUDA Nuru backend support are Work in Progress.

Prefer Ninja unless a Windows-only workflow explicitly requires Visual Studio.

Configure:

```sh
cmake -S Z:\Nuru -B Z:\Nuru\builds\windows-dev -G Ninja
```

Focused Eevee/GPU build:

```sh
cmake --build Z:\Nuru\builds\windows-dev --target bf_gpu bf_draw blender -j 12
```

Launch path:

```sh
Z:\Nuru\builds\windows-dev\bin\blender.exe
```

Do not write Windows generator files into macOS or Linux build directories.

## Practical Safety Checks

- Stop Blender, editors, and build jobs on one machine before switching the
  shared tree to another machine.
- If a path or command is macOS-specific, derive the Linux/Windows equivalent
  instead of reusing it blindly.
- Keep generated validation images, benchmark payloads, and temporary render
  outputs outside the repo unless explicitly requested.
