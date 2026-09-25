"""Contact sheets for reviewing renders and stills.

    python tools/sheet.py film/<name>/shots.json                        # every rendered shot, 3 frames each
    python tools/sheet.py film/<name>/shots.json --ids h10,p10 --cols 5 --tag fix
    python tools/sheet.py film/<name>/shots.json --stills --tag stills  # stills/<id>.png, 3 per row
    python tools/sheet.py film/<name>/shots.json --ids h10 --cols 16 --crop 0.55,0.1,0.9,0.7 --tag mouth

Shots: one row per shot, `cols` frames sampled at 8%..92% of its length, 7 rows per sheet.
--crop x0,y0,x1,y1 (fractions of the frame) zooms every frame to that box and samples from 0% to 100% instead:
close-up strips for checking a silent character's mouth or a hand frame by frame.
Sheets land in <film>/edit/sheets/<tag>_<k>.png; each path is printed.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SHEET_W = 1152
ROWS_PER_SHEET = 7
FONT = ImageFont.truetype("arialbd.ttf", 20)
SMALL = ImageFont.truetype("arialbd.ttf", 11)


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def grab(video: Path, t: float) -> Image.Image:
    png = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(png)).convert("RGB")


def label(img: Image.Image, text: str, at_top: bool = False) -> None:
    """Bottom-left by default, so the label never covers faces (subjects sit in the upper half of most framings);
    top-left on close-up strips, where the mouth or hand sits low in the box."""
    d = ImageDraw.Draw(img)
    font = FONT if img.width >= 160 else SMALL
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    top = 0 if at_top else img.height - (y1 - y0) - 8
    d.rectangle((0, top, x1 + 12, img.height), fill="black")
    d.text((6, top + 4 - y0), text, font=font, fill="white")


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Letterbox into w×h (keeps aspect; stills may be square)."""
    img = img.copy()
    img.thumbnail((w, h), Image.LANCZOS)
    tile = Image.new("RGB", (w, h), "black")
    tile.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return tile


def write_sheets(rows: list[list[Image.Image]], cols: int, out_dir: Path, tag: str) -> None:
    if not rows:
        return
    tw, th = rows[0][0].size
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(rows), ROWS_PER_SHEET):
        chunk = rows[k:k + ROWS_PER_SHEET]
        sheet = Image.new("RGB", (tw * cols, th * len(chunk)), "black")
        for r, row in enumerate(chunk):
            for c, tile in enumerate(row):
                sheet.paste(tile, (c * tw, r * th))
        path = out_dir / f"{tag}_{k // ROWS_PER_SHEET}.png"
        sheet.save(path)
        print(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--ids", help="comma-separated shot (or still) ids; default all, in order")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--tag", default="sheet")
    ap.add_argument("--stills", action="store_true", help="tile stills/<id>.png instead of renders")
    ap.add_argument("--crop", help="x0,y0,x1,y1 fractions: close-up box, frames sampled from 0%% to 100%%")
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    root = a.shotlist.parent
    out_dir = root / "edit" / "sheets"
    wanted = a.ids.split(",") if a.ids else None

    if a.stills:
        cols = 3
        tw, th = SHEET_W // cols, round(SHEET_W // cols * 9 / 16)
        tiles = []
        for sid in wanted or list(film.get("stills", {})):
            p = root / "stills" / f"{sid}.png"
            if not p.exists():
                print(f"skip {sid}: missing")
                continue
            tile = fit(Image.open(p).convert("RGB"), tw, th)
            label(tile, sid)
            tiles.append(tile)
        write_sheets([tiles[i:i + cols] for i in range(0, len(tiles), cols)], cols, out_dir, a.tag)
        return

    if a.cols < 2:
        raise SystemExit("--cols must be at least 2 (frames are spread from 8% to 92% of each shot)")
    tw, th = SHEET_W // a.cols, round(SHEET_W // a.cols * 9 / 16)
    box = [float(v) for v in a.crop.split(",")] if a.crop else None
    if box and (len(box) != 4 or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise SystemExit("--crop takes x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if box:  # tiles take the box's aspect
        th = round(tw * (box[3] - box[1]) * film.get("height", 768) / ((box[2] - box[0]) * film.get("width", 1344)))
    ids = wanted or [s["id"] for s in film["shots"]]
    rows = []
    for sid in ids:
        video = root / "renders" / f"{sid}.mp4"
        if not video.exists():
            print(f"skip {sid}: missing")
            continue
        d = probe_duration(video)
        row = []
        for i in range(a.cols):
            t = d * i / a.cols if box else d * (0.08 + 0.84 * i / (a.cols - 1))
            frame = grab(video, t)
            if box:
                frame = frame.crop((round(box[0] * frame.width), round(box[1] * frame.height),
                                    round(box[2] * frame.width), round(box[3] * frame.height)))
            tile = fit(frame, tw, th)
            label(tile, f"{t:.1f}" if box and i else f"{sid} {t:.1f}s" if i else sid, at_top=bool(box))
            row.append(tile)
        rows.append(row)
    write_sheets(rows, a.cols, out_dir, a.tag)


if __name__ == "__main__":
    sys.exit(main())
