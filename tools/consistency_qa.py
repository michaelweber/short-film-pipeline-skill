"""Keyframe consistency check: within each shot, how much do the set and the characters drift between keyframes?

    python tools/consistency_qa.py film/<f>/shots.json [--stills stills] [--out edit/consistency.json]

Keyframes are the stills named k_<shot>_<n> (their "init" is the greybox frame they were drawn over, whose
<shot>_t<sec>_depth.png marks what is near: the cast). Each later keyframe is compared with the shot's first:
  set    SSIM of the two keyframes' greyscale background, where the depth render says neither frame has a figure
         (the camera may move slightly; a redrawn corridor scores well below a restyled one)
  cast   colour-histogram distance (Hellinger, 0 = same palette) over the pixels the depth render marks as cast
A keyframe is flagged when set < --set-min or cast > --cast-max. Prints a table; writes the numbers as JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

KEY = re.compile(r"k_(p?\w+?)_(\d+)$")


def load(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32) / 255


def cast_mask(depth: Path, size: tuple[int, int], cut: float) -> np.ndarray:
    """Pixels nearer than `cut` of the depth range (white = near): the figures, not the walls behind them."""
    d = np.asarray(Image.open(depth).convert("L").resize(size, Image.BILINEAR), dtype=np.float32) / 255
    return d > cut


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray, win: int = 7) -> float:
    """Mean SSIM of greyscale a, b over `mask` (box-filtered local statistics)."""
    from numpy.lib.stride_tricks import sliding_window_view
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2

    def box(x):
        x = np.pad(x, pad, mode="edge")
        return sliding_window_view(x, (win, win)).mean(axis=(-1, -2))

    ma, mb = box(a), box(b)
    va, vb = box(a * a) - ma * ma, box(b * b) - mb * mb
    cov = box(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s[mask].mean()) if mask.any() else float("nan")


def hist_dist(a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray, bins: int = 8) -> float:
    """Hellinger distance between the RGB histograms of a[ma] and b[mb]."""
    if not ma.any() or not mb.any():
        return float("nan")
    ha = np.histogramdd(a[ma], bins=bins, range=[(0, 1)] * 3)[0].ravel()
    hb = np.histogramdd(b[mb], bins=bins, range=[(0, 1)] * 3)[0].ravel()
    ha, hb = ha / ha.sum(), hb / hb.sum()
    return float(np.sqrt(max(0.0, 1 - np.sqrt(ha * hb).sum())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--stills", default="stills", help="folder of keyframe PNGs, relative to the shot list")
    ap.add_argument("--out", default="edit/consistency.json")
    ap.add_argument("--set-min", type=float, default=0.55)
    ap.add_argument("--cast-max", type=float, default=0.45)
    ap.add_argument("--cut", type=float, default=0.35, help="depth above which a pixel counts as cast")
    a = ap.parse_args()
    root = a.shotlist.parent
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    size = (336, 192)
    shots: dict[str, list] = {}
    for sid, spec in film.get("stills", {}).items():
        m = KEY.fullmatch(sid)
        if m and (spec.get("init") or spec.get("control")):
            shots.setdefault(m.group(1), []).append((int(m.group(2)), sid, spec))
    rows, flagged = [], 0
    for shot, keys in sorted(shots.items()):
        keys.sort()
        imgs, masks = {}, {}
        for _, sid, spec in keys:
            png = root / a.stills / f"{sid}.png"
            src = spec.get("init") or spec["control"]["image"].replace("_depth.png", ".png")
            depth = root / src.replace(".png", "_depth.png")
            if not png.exists() or not depth.exists():
                print(f"  {sid}: missing {png if not png.exists() else depth}", file=sys.stderr)
                continue
            imgs[sid], masks[sid] = load(png, size), cast_mask(depth, size, a.cut)
        ids = [sid for _, sid, _ in keys if sid in imgs]
        if len(ids) < 2:
            continue
        first = ids[0]
        for sid in ids[1:]:
            bg = ~(masks[first] | masks[sid])
            grey = [imgs[x] @ np.array([0.299, 0.587, 0.114], dtype=np.float32) for x in (first, sid)]
            s = ssim(grey[0], grey[1], bg)
            c = hist_dist(imgs[first], imgs[sid], masks[first], masks[sid])
            bad = (s == s and s < a.set_min) or (c == c and c > a.cast_max)
            flagged += bad
            rows.append({"shot": shot, "a": first, "b": sid, "set_ssim": round(s, 3), "cast_dist": round(c, 3),
                         "flag": bool(bad)})
            print(f"{'!' if bad else ' '} {shot:4} {first:9} -> {sid:9}  set {s:5.2f}  cast {c:5.2f}")
    out = root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set_min": a.set_min, "cast_max": a.cast_max, "rows": rows}, indent=1), encoding="utf-8")
    sets = [r["set_ssim"] for r in rows if r["set_ssim"] == r["set_ssim"]]
    casts = [r["cast_dist"] for r in rows if r["cast_dist"] == r["cast_dist"]]
    print(f"{len(rows)} pairs, {flagged} flagged; mean set {np.mean(sets):.2f}, mean cast {np.mean(casts):.2f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
