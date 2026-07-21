#!/usr/bin/env python3
"""Build packaging/qw35.icns from assets/app_icon.png.

The source art is a white squircle with a baked drop shadow on a pure-white
1254x1254 canvas and no alpha channel. Shipping that as-is would render a
white square tile in the Dock, so this script rebuilds a proper macOS icon:

  1. locate the squircle edge-to-edge (the baked shadow dips darkest just
     outside the edge, so each side reads as "shadow band, then bright again");
  2. crop it and LANCZOS-resize onto the 824x824 Apple icon-grid area of a
     1024x1024 transparent canvas;
  3. alpha-mask with the Apple squircle (superellipse, drawn 2 px inset for
     clean anti-aliased borders);
  4. regenerate the drop shadow from the mask (the baked one is unusable once
     the white canvas becomes transparent);
  5. emit the 10-size .iconset and run iconutil.

Usage:
  python3 make_icns.py SRC.png OUT.icns [--bbox L,T,R,B] [--no-shadow]

--bbox overrides edge detection (exclusive right/bottom, source-image pixels).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageFilter

CANVAS = 1024
GRID = 824  # Apple icon-grid squircle size on a 1024 canvas
# Superellipse exponent: |x|^n + |y|^n = 1 with n=5 tracks Apple's icon shape
# closely enough that the difference is invisible at icon sizes.
SUPER_N = 5.0

SHADOW_OFFSET_Y = 12
SHADOW_BLUR = 24
SHADOW_ALPHA = 77  # 30% black

ICONSET_SIZES = (16, 32, 128, 256, 512)


def detect_bbox(im: Image.Image) -> tuple[int, int, int, int]:
    """Squircle bbox from the center row/column shadow profile.

    Walking inward from each side, pixels first dip below the shadow
    threshold and then return to near-white at the squircle edge; that
    recovery point is the edge.
    """
    w, h = im.size
    px = im.convert("RGB").load()
    row = [min(px[x, h // 2]) for x in range(w)]
    col = [min(px[w // 2, y]) for y in range(h)]

    def edge(vals: list[int]) -> int:
        in_shadow = False
        for i, v in enumerate(vals):
            if v < 245:
                in_shadow = True
            elif in_shadow and v >= 252:
                return i
        raise SystemExit(
            "make_icns: could not detect the squircle edge (no shadow-band -> "
            "bright transition on the center axis); pass --bbox L,T,R,B"
        )

    left = edge(row)
    right = len(row) - edge(list(reversed(row)))
    top = edge(col)
    bottom = len(col) - edge(list(reversed(col)))
    if not (right - left > w // 2 and bottom - top > h // 2):
        raise SystemExit(
            f"make_icns: implausible detected bbox ({left},{top},{right},{bottom}); "
            "pass --bbox L,T,R,B"
        )
    return left, top, right, bottom


def squircle_mask(size: int, inset: int, scale: int = 4) -> Image.Image:
    """Anti-aliased superellipse alpha mask, drawn at `scale`x and downsampled."""
    big = (size - 2 * inset) * scale
    half = big / 2
    mask = Image.new("L", (big, big), 0)
    px = mask.load()
    for y in range(big):
        dy = abs((y + 0.5) - half) / half
        # |x/h|^n = 1 - |y/h|^n  ->  half-width of the row
        span = half * (max(0.0, 1.0 - dy**SUPER_N)) ** (1.0 / SUPER_N)
        x0, x1 = int(half - span), int(half + span)
        for x in range(x0, x1):
            px[x, y] = 255
    mask = mask.resize((size - 2 * inset, size - 2 * inset), Image.LANCZOS)
    out = Image.new("L", (size, size), 0)
    out.paste(mask, (inset, inset))
    return out


def build_master(src: Image.Image, bbox: tuple[int, int, int, int], shadow: bool) -> Image.Image:
    art = src.convert("RGB").crop(bbox).resize((GRID, GRID), Image.LANCZOS)
    mask = squircle_mask(GRID, inset=2)

    tile = Image.new("RGBA", (GRID, GRID))
    tile.paste(art, (0, 0), mask)

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    origin = (CANVAS - GRID) // 2
    if shadow:
        sh = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        black = Image.new("L", (GRID, GRID), SHADOW_ALPHA)
        sh.paste(Image.merge("RGBA", (black.point(lambda _: 0),) * 3 + (mask.point(lambda v: v * SHADOW_ALPHA // 255),)), (origin, origin + SHADOW_OFFSET_Y))
        sh = sh.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
        canvas = Image.alpha_composite(canvas, sh)
    canvas.paste(tile, (origin, origin), tile)
    return canvas


def write_icns(master: Image.Image, out_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "qw35.iconset")
        os.mkdir(iconset)
        for size in ICONSET_SIZES:
            master.resize((size, size), Image.LANCZOS).save(
                os.path.join(iconset, f"icon_{size}x{size}.png")
            )
            two = size * 2
            img = master if two == CANVAS else master.resize((two, two), Image.LANCZOS)
            img.save(os.path.join(iconset, f"icon_{size}x{size}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out_path], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bbox", help="override squircle bbox as L,T,R,B (source pixels)")
    ap.add_argument("--no-shadow", dest="shadow", action="store_false")
    args = ap.parse_args()

    src = Image.open(args.src)
    if args.bbox:
        bbox = tuple(int(v) for v in args.bbox.split(","))
        if len(bbox) != 4:
            sys.exit("make_icns: --bbox needs four comma-separated integers")
    else:
        bbox = detect_bbox(src)
        print(f"make_icns: detected squircle bbox {bbox}")

    master = build_master(src, bbox, shadow=args.shadow)
    write_icns(master, args.out)
    master.save(os.path.splitext(args.out)[0] + "-1024.png")  # inspection copy
    print(f"make_icns: wrote {args.out}")


if __name__ == "__main__":
    main()
