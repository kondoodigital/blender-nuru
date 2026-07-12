#!/usr/bin/env python3
"""Generate the Blender-Nuru NSIS installer branding assets from images/.

Inputs (clean RGBA PNGs):
  nuru_logo_256.png   - square logo; becomes the installer/uninstaller/shortcut icon.
                        (The wide logo padded to square rendered as an unreadable
                        ~19 px sliver at shell icon sizes - "missing icon" in 0.9.8.)
  nuru_logo_wide.png  - wide logo; header strip.
  nuru_logo.png       - UI logo with alpha; becomes the installer splash screen.
  nuru_logo_512.png   - square logo with "powered by Kondoo Digital"; becomes the
                        welcome/finish sidebar image.

Outputs (written to --out-dir):
  blender_nuru.ico    - multi-size icon (256..16), square logo content-cropped and
                        padded to square.
  splash.bmp          - 24-bit BMP for advsplash, magenta (FF00FF) transparency key,
                        soft-alpha pixels pre-blended onto a dark neutral backdrop.
  welcome.bmp         - 164x314 MUI welcome/finish sidebar, white background.
  header.bmp          - 150x57 MUI header image, white background.

Usage:
  python generate_assets.py --images-dir ../../../images --out-dir <dir>
"""

import argparse
import os

from PIL import Image

SPLASH_KEY = (255, 0, 255)
SPLASH_BACKDROP = (28, 28, 34)


def load_rgba(path):
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def make_icon(square_png, out_path):
    img = load_rgba(square_png)
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    master = canvas.resize((256, 256), Image.LANCZOS)
    # Explicit per-size frames: Pillow's single-image `sizes=` export keeps one
    # 256 px PNG-compressed entry that some shell views refuse to scale, which
    # also contributed to the blank installer icon. Store real resampled frames.
    frames = [master.resize((s, s), Image.LANCZOS)
              for s in (256, 128, 64, 48, 32, 24, 16)]
    frames[0].save(
        out_path,
        format="ICO",
        append_images=frames[1:],
        sizes=[(f.width, f.height) for f in frames],
    )


def make_splash(ui_png, out_path, width=560):
    """Color-key transparency: fully/mostly opaque pixels keep their color (soft
    edges pre-blended onto a dark neutral), everything else becomes the key."""
    img = load_rgba(ui_png)
    height = round(img.height * width / img.width)
    img = img.resize((width, height), Image.LANCZOS)
    out = Image.new("RGB", (width, height), SPLASH_KEY)
    src = img.load()
    dst = out.load()
    for y in range(height):
        for x in range(width):
            r, g, b, a = src[x, y]
            if a < 96:
                continue
            f = a / 255.0
            rgb = (
                round(r * f + SPLASH_BACKDROP[0] * (1.0 - f)),
                round(g * f + SPLASH_BACKDROP[1] * (1.0 - f)),
                round(b * f + SPLASH_BACKDROP[2] * (1.0 - f)),
            )
            if rgb == SPLASH_KEY:  # avoid accidental key-color holes
                rgb = (rgb[0], 1, rgb[2])
            dst[x, y] = rgb
    out.save(out_path, "BMP")


def paste_on_white(canvas_size, logo, max_size, center_y=None):
    canvas = Image.new("RGB", canvas_size, (255, 255, 255))
    fitted = logo.copy()
    fitted.thumbnail(max_size, Image.LANCZOS)
    x = (canvas_size[0] - fitted.width) // 2
    y = center_y - fitted.height // 2 if center_y else (canvas_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y), fitted.split()[3])
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wide = os.path.join(args.images_dir, "nuru_logo_wide.png")
    square = os.path.join(args.images_dir, "nuru_logo_256.png")
    ui = os.path.join(args.images_dir, "nuru_logo.png")
    powered = os.path.join(args.images_dir, "nuru_logo_512.png")

    make_icon(square, os.path.join(args.out_dir, "blender_nuru.ico"))
    make_splash(ui, os.path.join(args.out_dir, "splash.bmp"))

    welcome = paste_on_white((164, 314), load_rgba(powered), (150, 150), center_y=110)
    welcome.save(os.path.join(args.out_dir, "welcome.bmp"), "BMP")

    header = paste_on_white((150, 57), load_rgba(wide), (142, 48))
    header.save(os.path.join(args.out_dir, "header.bmp"), "BMP")
    print("installer assets written to", args.out_dir)


if __name__ == "__main__":
    main()
