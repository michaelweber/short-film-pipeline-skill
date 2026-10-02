"""Split a four-view turnaround sheet into one image per view, for multi-view -> 3D model tools.

    python tools/split_views.py film/<name>/stills/turn_hero.png [more.png ...] [--size 1024]

Finds the backdrop colour from the sheet's border, then the columns that contain anything else. The sheet must
contain exactly four separate runs of such columns (front, left, back, right, left to right); anything else
means views touch or overlap, and the sheet is rejected. All four views keep one shared scale and ground line
(the sheet's full vertical span, one square size), centred on squares of the backdrop colour, and are written next
to the sheet as <stem>_front/_left/_back/_right.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VIEWS = ("front", "left", "back", "right")


def split(path: Path, size: int, tol: float = 18.0, min_gap: int = 6) -> list[Path]:
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    h, w, _ = img.shape
    border = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3),
                             img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    fg = np.abs(img - bg).max(axis=2) > tol
    cols = fg.sum(axis=0) > max(2, h // 200)  # ignore speckle
    runs, start, gap = [], None, 0
    for x, on in enumerate(cols):
        if on:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                runs.append((start, x - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, w))
    runs = [r for r in runs if r[1] - r[0] > w // 40]  # drop stray specks
    if len(runs) != 4:
        raise ValueError(f"{path.name}: found {len(runs)} separate views, need 4 (views touch or overlap)")
    # One scale and one ground line for all four views (multi-view reconstruction needs consistent scale):
    # every view keeps the sheet's shared vertical span, and every square has the same side.
    spans = []
    for view, (x0, x1) in zip(VIEWS, runs):
        rows = np.where(fg[:, x0:x1].any(axis=1))[0]
        if rows[0] <= 1 or rows[-1] >= h - 2:
            raise ValueError(f"{path.name}: the {view} view touches the top or bottom edge (cropped)")
        spans.append((int(rows[0]), int(rows[-1]) + 1))
    y0, y1 = min(s[0] for s in spans), max(s[1] for s in spans)
    side = int(max(y1 - y0, *(x1 - x0 for x0, x1 in runs)) * 1.12)
    out = []
    for view, (x0, x1) in zip(VIEWS, runs):
        crop = Image.fromarray(img[y0:y1, x0:x1].astype(np.uint8))
        canvas = Image.new("RGB", (side, side), tuple(int(v) for v in bg))
        canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        dest = path.with_name(f"{path.stem}_{view}.png")
        canvas.resize((size, size), Image.Resampling.LANCZOS).save(dest)
        out.append(dest)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheets", nargs="+", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    bad = 0
    for sheet in a.sheets:
        try:
            for p in split(sheet, a.size):
                print(f"+ {p}")
        except ValueError as e:
            print(f"!! {e}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
