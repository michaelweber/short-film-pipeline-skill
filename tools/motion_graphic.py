"""Animated infographic shots ("mode": "motion_graphic") rendered frame by frame with PIL, no GPU.

    python tools/motion_graphic.py film/x/shots.json --only mg01 --preview   # contact strip of one graphic
    (h3_render.py renders every motion_graphic shot like a clip: renders/<id>.mp4, silent track, cached)

Shot fields: "duration" (s) and "graphic": {"template": ..., ...}. Every template keeps its content inside the
2.39:1 letterbox safe area and animates in (slides, pops, counters, fills), holds, and stays on its final state.
Image paths are relative to the film dir; "cutout": true keys a white sheet background to transparency.

Templates and their fields:
  blister_card  collector blister card: "series", "title", "subtitle", "image", "features" [str], "sticker" (starburst),
                "badge" (corner age badge, "\\n" for a second line), "warning" (bottom strip)
  stat_card  trading card: "title", "subtitle", "image", "stats" [{"label", "fill" 0-1.2, "display"}]
  bar_chart  "title", "subtitle", "unit", "bars" [{"label", "value"}], "callout" (sticker on the last bar)
  meter      horizontal gauge: "title", "subtitle", "marks" [{"label", "tag", "pos" 0-1}], "needle_to" (>1 = past
             the end), "verdict" (stamp when the needle lands)
  counter    dashboard: "label", "from", "to", "roll" (s), "milestones" [{"at", "label"}] (starbursts as they pass)
  line_crash dashboard graph that climbs and falls to zero: "label", "peak", "crash_at" 0-1, "draw" (s), "delta",
             "verdict" (stamp after the fall)
  empty_feed comment feed of placeholder rows that never loads: "label", "banner", "banner_at" (s)
  error_stack stalled copy dialog + stacking error windows, screen tears then dies: "task", "app", "errors" [str],
             "black_at" (s)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FPS = 24
FONTS = Path("C:/Windows/Fonts")
BLACK = "seguibl.ttf"   # Segoe UI Black: headline
BOLD = "segoeuib.ttf"
IMPACT = "impact.ttf"
PAL = {"bg0": (18, 22, 38), "bg1": (44, 20, 60), "gold": (255, 196, 46), "red": (230, 44, 52),
       "white": (250, 248, 240), "ink": (20, 20, 28), "teal": (40, 200, 190), "card": (255, 214, 64)}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


# ------------------------------------------------------------------ easing
def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def seg(t: float, t0: float, dur: float) -> float:
    return clamp((t - t0) / dur)


def out_cubic(x: float) -> float:
    return 1 - (1 - x) ** 3


def out_back(x: float, s: float = 1.9) -> float:
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2 if x > -1 else 0.0


def out_elastic(x: float) -> float:
    if x <= 0 or x >= 1:
        return clamp(x)
    return 2 ** (-10 * x) * math.sin((x * 10 - 0.75) * (2 * math.pi) / 3) + 1


# ------------------------------------------------------------------ drawing helpers
class Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        bar = round((h - w / 2.39) / 2)  # letterbox bars laid by the cut
        self.top, self.bottom = bar + 14, h - bar - 14


def background(cv: Canvas, t: float, c0=PAL["bg0"], c1=PAL["bg1"]) -> Image.Image:
    y = np.linspace(0, 1, cv.h)[:, None]
    x = np.linspace(0, 1, cv.w)[None, :]
    g = np.clip(0.55 * y + 0.45 * x + 0.08 * np.sin(2 * math.pi * (x * 1.3 + t * 0.15)), 0, 1)[..., None]
    arr = (np.array(c0) * (1 - g) + np.array(c1) * g)
    # slow rotating light rays from the centre (TV-commercial energy)
    yy, xx = np.mgrid[0:cv.h, 0:cv.w]
    ang = np.arctan2(yy - cv.h / 2, xx - cv.w / 2) + t * 0.35
    rays = (np.sin(ang * 14) > 0.55) * 14.0
    arr = np.clip(arr + rays[..., None], 0, 255)
    # halftone dots
    dots = ((xx % 18 - 9) ** 2 + (yy % 18 - 9) ** 2 < 5) * 10.0
    arr = np.clip(arr + dots[..., None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


def text(d: ImageDraw.ImageDraw, xy, s: str, f, fill, anchor="la", stroke=0, stroke_fill=PAL["ink"], shadow=True):
    if shadow:
        d.text((xy[0] + 4, xy[1] + 5), s, font=f, fill=(0, 0, 0, 150), anchor=anchor, stroke_width=stroke,
               stroke_fill=(0, 0, 0, 150))
    d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=stroke_fill)


def fit(s: str, name: str, size: int, width: int) -> ImageFont.FreeTypeFont:
    while size > 10:
        f = font(name, size)
        if f.getlength(s) <= width:
            return f
        size -= 2
    return font(name, size)


def starburst(size: int, spikes: int, fill, outline=PAL["ink"]) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, r0, r1 = size / 2, size * 0.36, size * 0.49
    pts = [(c + (r1 if i % 2 == 0 else r0) * math.cos(math.pi * i / spikes),
            c + (r1 if i % 2 == 0 else r0) * math.sin(math.pi * i / spikes)) for i in range(2 * spikes)]
    d.polygon(pts, fill=fill, outline=outline, width=5)
    return im


def paste_scaled(base: Image.Image, im: Image.Image, center, scale: float, rot: float = 0.0, alpha: float = 1.0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im2 = im.resize((w, h), Image.LANCZOS)
    if rot:
        im2 = im2.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        a = im2.getchannel("A").point(lambda v: int(v * alpha))
        im2.putalpha(a)
    base.alpha_composite(im2, (int(center[0] - im2.width / 2), int(center[1] - im2.height / 2)))


def load_image(root: Path, spec: dict) -> Image.Image:
    im = Image.open(root / spec["image"]).convert("RGBA")
    if spec.get("cutout"):
        rgb = np.asarray(im.convert("RGB")).astype(int)
        key = (255 - rgb).max(axis=2) > 22
        a = Image.fromarray((key * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(
            ImageFilter.GaussianBlur(1.2))
        im.putalpha(a)
        im = im.crop(im.getbbox())
    return im


def glint(base: Image.Image, box, t: float, period: float = 3.0):
    """Diagonal light sweep across a box, once per period."""
    x0, y0, x1, y1 = box
    p = (t % period) / period
    if p > 0.4:
        return
    cx = x0 + (x1 - x0 + 400) * (p / 0.4) - 200
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.polygon([(cx - 60, y0), (cx, y0), (cx - 120, y1), (cx - 180, y1)], fill=(255, 255, 255, 70))
    mask = Image.new("L", base.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, 28, fill=255)
    ov.putalpha(ImageChops.multiply(ov.getchannel("A"), mask))
    base.alpha_composite(ov)


def header(base: Image.Image, cv: Canvas, t: float, title: str, subtitle: str | None, t0: float = 0.0):
    d = ImageDraw.Draw(base)
    k = out_back(seg(t, t0, 0.6))
    x = -600 + 660 * k
    f = fit(title, BLACK, 58, 820)
    text(d, (x, cv.top + 6), title, f, PAL["white"], stroke=3)
    if subtitle:
        k2 = out_cubic(seg(t, t0 + 0.3, 0.5))
        text(d, (-600 + 664 * k2, cv.top + 78), subtitle, font(BOLD, 28), PAL["gold"], stroke=2)


# ------------------------------------------------------------------ templates
def blister_card(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    base = background(cv, t, (14, 30, 70), (70, 14, 40))
    d = ImageDraw.Draw(base)
    # the card slides up with a bounce
    k = out_back(seg(t, 0.0, 0.8), 1.4)
    cw, ch = 560, cv.bottom - cv.top - 20
    cx0 = 90
    cy0 = cv.top + 10 + (1 - k) * 700
    card = (cx0, cy0, cx0 + cw, cy0 + ch)
    d.rounded_rectangle(card, 30, fill=PAL["red"], outline=PAL["ink"], width=6)
    d.rounded_rectangle((cx0 + 18, cy0 + 16, cx0 + cw - 18, cy0 + 96), 20, fill=PAL["gold"], outline=PAL["ink"], width=4)
    text(d, (cx0 + cw / 2, cy0 + 34), g.get("series", "COLLECTOR SERIES"), font(BOLD, 22), PAL["ink"], "mm",
         shadow=False)
    f = fit(g["title"], BLACK, 40, cw - 60)
    text(d, (cx0 + cw / 2, cy0 + 70), g["title"], f, PAL["ink"], "mm", shadow=False)
    # plastic bubble with the figure
    bx = (cx0 + 40, cy0 + 116, cx0 + cw - 40, cy0 + ch - 70)
    d.rounded_rectangle(bx, 40, fill=(235, 245, 255, 70), outline=(255, 255, 255, 200), width=4)
    if "img" not in cache:
        cache["img"] = load_image(root, g)
    im = cache["img"]
    s = min((bx[2] - bx[0] - 40) / im.width, (bx[3] - bx[1] - 30) / im.height)
    pop = out_back(seg(t, 0.5, 0.7))
    paste_scaled(base, im, ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2 + 8), s * pop)
    glint(base, bx, t - 1.2)
    d = ImageDraw.Draw(base)
    text(d, (cx0 + cw / 2, cy0 + ch - 38), g.get("subtitle", ""), fit(g.get("subtitle", ""), BOLD, 26, cw - 50),
         PAL["white"], "mm", stroke=2)
    # feature list with check marks, one after another
    fx, fy = cx0 + cw + 60, cv.top + 40
    for i, feat in enumerate(g.get("features", [])):
        kk = out_back(seg(t, 1.1 + 0.45 * i, 0.5))
        if kk <= 0:
            continue
        y = fy + i * 74
        x = fx + (1 - kk) * 500
        d.rounded_rectangle((x, y, x + 560, y + 60), 16, fill=(255, 255, 255, 235), outline=PAL["ink"], width=4)
        d.ellipse((x + 12, y + 10, x + 52, y + 50), fill=PAL["teal"], outline=PAL["ink"], width=3)
        d.line([(x + 22, y + 31), (x + 30, y + 40), (x + 44, y + 20)], fill=PAL["white"], width=6)
        text(d, (x + 66, y + 30), feat, fit(feat, BOLD, 30, 480), PAL["ink"], "lm", shadow=False)
    # starburst sticker slaps on
    if g.get("sticker"):
        ks = seg(t, 1.3 + 0.45 * len(g.get("features", [])), 0.45)
        if ks > 0:
            if "star" not in cache:
                st = starburst(300, 16, PAL["gold"])
                sd = ImageDraw.Draw(st)
                lines = g["sticker"].split("\n")
                for j, line in enumerate(lines):
                    text(sd, (150, 150 + (j - (len(lines) - 1) / 2) * 38), line, fit(line, IMPACT, 40, 200),
                         PAL["red"], "mm", stroke=2, stroke_fill=PAL["white"], shadow=False)
                cache["star"] = st
            wob = 4 * math.sin(t * 5)
            paste_scaled(base, cache["star"], (cx0 + cw - 30, cy0 + 290), out_elastic(ks) * 0.8, -14 + wob)
    # age badge
    if g.get("badge"):
        kb = out_back(seg(t, 0.9, 0.5))
        r = 62 * kb
        bx0, by0 = cx0 + 60, cy0 + ch - 150
        if r > 2:
            d = ImageDraw.Draw(base)
            d.ellipse((bx0 - r, by0 - r, bx0 + r, by0 + r), fill=PAL["white"], outline=PAL["ink"], width=5)
            lines = g["badge"].split("\n")
            text(d, (bx0, by0 - 12 * (len(lines) > 1)), lines[0], font(BLACK, int(34 * kb) + 1), PAL["red"], "mm",
                 shadow=False)
            if len(lines) > 1:
                text(d, (bx0, by0 + 24), lines[1], font(BOLD, int(15 * kb) + 1), PAL["ink"], "mm", shadow=False)
    # warning strip
    if g.get("warning"):
        kw = out_cubic(seg(t, 2.6 + 0.45 * len(g.get("features", [])), 0.5))
        if kw > 0:
            d = ImageDraw.Draw(base)
            y = cv.bottom - 70
            x1 = fx - 10 + 600 * kw
            d.rectangle((fx - 10, y, x1, y + 56), fill=(255, 225, 0), outline=PAL["ink"], width=4)
            for sx in range(int(fx - 10), int(x1), 36):
                d.polygon([(sx, y + 56), (sx + 18, y + 56), (sx + 36, y), (sx + 18, y)], fill=(30, 30, 30))
            if x1 - 40 > fx + 40:
                d.rectangle((fx + 40, y + 8, x1 - 40, y + 48), fill=(255, 225, 0))
            if kw > 0.95:
                text(d, ((fx + x1) / 2, y + 28), g["warning"], fit(g["warning"], BLACK, 26, 500), PAL["ink"], "mm",
                     shadow=False)
    return base


def stat_card(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    base = background(cv, t, (10, 40, 44), (20, 16, 60))
    header(base, cv, t, g["title"], g.get("subtitle"))
    d = ImageDraw.Draw(base)
    # portrait card flips in
    k = seg(t, 0.2, 0.7)
    sx = abs(math.cos((1 - out_cubic(k)) * math.pi / 2)) if k > 0 else 0
    cx, cy, cw, ch = 300, (cv.top + cv.bottom) / 2 + 40, 360, cv.bottom - cv.top - 150
    if "img" not in cache:
        cache["img"] = load_image(root, g)
    if sx > 0.02:
        card = Image.new("RGBA", (cw, int(ch)), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle((0, 0, cw - 1, ch - 1), 26, fill=PAL["card"], outline=PAL["ink"], width=6)
        cd.rounded_rectangle((18, 18, cw - 19, ch - 70), 18, fill=(30, 70, 90), outline=PAL["ink"], width=4)
        im = cache["img"]
        s = min((cw - 60) / im.width, (ch - 110) / im.height)
        im2 = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        card.alpha_composite(im2, ((cw - im2.width) // 2, 28 + (ch - 110 - im2.height) // 2))
        text(cd, (cw / 2, ch - 36), g.get("card_name", "LEGENDARY"), font(BLACK, 30), PAL["ink"], "mm", shadow=False)
        card = card.resize((max(1, int(cw * sx)), int(ch)), Image.LANCZOS)
        base.alpha_composite(card, (int(cx - card.width / 2), int(cy - ch / 2)))
        glint(base, (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2), t - 1.0)
    # stat bars fill one after another; fills over 1.0 burst past the end
    d = ImageDraw.Draw(base)
    x0, y0, bw = 560, cv.top + 140, 560
    for i, st in enumerate(g["stats"]):
        tt = 0.9 + 0.55 * i
        ki = out_cubic(seg(t, tt, 0.35))
        if ki <= 0:
            continue
        y = y0 + i * 84
        text(d, (x0 + (1 - ki) * 300, y), st["label"], font(BLACK, 30), PAL["white"], stroke=2)
        fill = st["fill"] * out_cubic(seg(t, tt + 0.15, 0.9))
        d.rounded_rectangle((x0, y + 42, x0 + bw, y + 70), 14, fill=(255, 255, 255, 50), outline=PAL["ink"], width=3)
        w = min(fill, 1.0) * bw
        col = PAL["red"] if st["fill"] > 0.99 else PAL["teal"]
        if w > 20:
            d.rounded_rectangle((x0, y + 42, x0 + w, y + 70), 14, fill=col, outline=PAL["ink"], width=3)
        if fill > 1.0:  # overflow spark
            r = 10 + 30 * (fill - 1) / max(0.01, st["fill"] - 1)
            d.ellipse((x0 + bw - r, y + 56 - r, x0 + bw + r, y + 56 + r), fill=(255, 240, 120, 200))
        disp = st.get("display", "")
        if seg(t, tt + 0.6, 0.2) > 0:
            text(d, (x0 + bw + 46, y + 56), disp, fit(disp, IMPACT, 40, 170), PAL["gold"], "lm", stroke=2)
    return base


def bar_chart(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    base = background(cv, t, (12, 24, 30), (10, 50, 60))
    header(base, cv, t, g["title"], g.get("subtitle"))
    d = ImageDraw.Draw(base)
    bars = g["bars"]
    vmax = max(b["value"] for b in bars) * 1.15
    x0, x1 = 140, cv.w - 140
    y_base, y_top = cv.bottom - 60, cv.top + 150
    ka = out_cubic(seg(t, 0.3, 0.5))
    d.line([(x0, y_base), (x0 + (x1 - x0) * ka, y_base)], fill=PAL["white"], width=4)
    for gl in range(1, 4):
        y = y_base - (y_base - y_top) * gl / 4
        d.line([(x0, y), (x0 + (x1 - x0) * ka, y)], fill=(255, 255, 255, 40), width=2)
    n = len(bars)
    slot = (x1 - x0) / n
    for i, b in enumerate(bars):
        tt = 0.7 + 0.35 * i
        k = out_back(seg(t, tt, 0.7), 1.2)
        h = (y_base - y_top) * b["value"] / vmax * k
        bx = x0 + slot * i + slot * 0.18
        bw = slot * 0.64
        col = PAL["red"] if i == n - 1 else PAL["teal"]
        if h > 2:
            d.rounded_rectangle((bx, y_base - h, bx + bw, y_base), 10, fill=col, outline=PAL["ink"], width=4)
            val = int(round(b["value"] * min(1, seg(t, tt, 0.7))))
            text(d, (bx + bw / 2, y_base - h - 26), f"{val}", font(IMPACT, 40), PAL["white"], "mm", stroke=2)
        elif b["value"] == 0 and seg(t, tt + 0.4, 0.3) > 0:  # an empty slot still reads as a (sad) zero
            text(d, (bx + bw / 2, y_base - 26), "0", font(IMPACT, 40), PAL["white"], "mm", stroke=2)
        if seg(t, tt, 0.3) > 0:
            text(d, (bx + bw / 2, y_base + 28), b["label"], fit(b["label"], BOLD, 24, int(slot)), PAL["gold"], "mm",
                 stroke=2)
    if g.get("unit"):
        text(d, (x1, y_top - 30), g["unit"], font(BOLD, 24), PAL["white"], "rm", stroke=2)
    if g.get("callout"):
        kc = seg(t, 0.9 + 0.35 * n, 0.5)
        if kc > 0:
            if "star" not in cache:
                st = starburst(240, 14, PAL["gold"])
                sd = ImageDraw.Draw(st)
                lines = g["callout"].split("\n")
                for j, line in enumerate(lines):
                    text(sd, (120, 120 + (j - (len(lines) - 1) / 2) * 32), line, fit(line, IMPACT, 34, 160),
                         PAL["red"], "mm", stroke=2, stroke_fill=PAL["white"], shadow=False)
                cache["star"] = st
            last_x = x0 + slot * (n - 1) + slot * 0.5
            paste_scaled(base, cache["star"], (last_x - 150, y_top + 40), out_elastic(kc) * 0.9,
                         12 + 3 * math.sin(t * 5))
    return base


def meter(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    base = background(cv, t, (30, 12, 20), (60, 20, 10))
    header(base, cv, t, g["title"], g.get("subtitle"))
    d = ImageDraw.Draw(base)
    x0, x1 = 120, cv.w - 170
    y = (cv.top + cv.bottom) / 2 + 60
    k = out_cubic(seg(t, 0.3, 0.6))
    # gradient gauge green -> red
    gw = int((x1 - x0) * k)
    if gw > 4:
        grad = np.linspace(0, 1, gw)[None, :, None]
        arr = (np.array([60, 200, 90]) * (1 - grad) + np.array([235, 40, 40]) * grad) * np.ones((44, 1, 1))
        base.paste(Image.fromarray(arr.astype(np.uint8), "RGB"), (x0, int(y - 22)))
        d.rounded_rectangle((x0, y - 22, x0 + gw, y + 22), 10, outline=PAL["ink"], width=5)
    marks = g["marks"]
    for i, m in enumerate(marks):
        tt = 0.8 + 0.45 * i
        km = out_back(seg(t, tt, 0.45))
        if km <= 0:
            continue
        mx = x0 + (x1 - x0) * m["pos"]
        d.line([(mx, y - 34), (mx, y + 34)], fill=PAL["white"], width=4)
        up = i % 2 == 0
        ly = y - 90 if up else y + 90
        text(d, (mx, ly + (1 - km) * (-40 if up else 40)), m["label"], fit(m["label"], BLACK, 30, 260), PAL["white"],
             "mm", stroke=2)
        if m.get("tag"):
            tag = m["tag"]
            tf = fit(tag, IMPACT, 26, 240)
            tw = tf.getlength(tag) + 24
            ty = ly + (38 if up else 38)
            col = PAL["red"] if "NOT" in tag else PAL["gold"]
            d.rounded_rectangle((mx - tw / 2, ty - 18, mx + tw / 2, ty + 18), 8, fill=col, outline=PAL["ink"],
                                width=3)
            text(d, (mx, ty), tag, tf, PAL["ink"], "mm", shadow=False)
    # needle sweeps, overshoots, rattles, then breaks through the end
    t_needle = 0.8 + 0.45 * len(marks) + 0.3
    kn = seg(t, t_needle, 1.4)
    target = g.get("needle_to", 1.0)
    p = out_elastic(kn) * target if kn > 0 else 0.0
    if kn >= 1:
        p = target + 0.01 * math.sin(t * 40)
    nx = x0 + (x1 - x0) * p
    d.polygon([(nx, y - 44), (nx - 18, y - 80), (nx + 18, y - 80)], fill=PAL["gold"], outline=PAL["ink"])
    d.line([(nx, y - 44), (nx, y + 44)], fill=PAL["gold"], width=8)
    if target > 1 and p > 1:  # cracks where it smashed through the end
        rnd = np.random.default_rng(7)
        for _ in range(9):
            a = rnd.uniform(0, 2 * math.pi)
            r = rnd.uniform(40, 120)
            d.line([(x1, y), (x1 + r * math.cos(a), y + r * math.sin(a))], fill=PAL["white"], width=3)
    if g.get("verdict") and kn >= 1:
        kv = seg(t, t_needle + 1.5, 0.35)
        if kv > 0:
            if "stamp" not in cache:
                s = g["verdict"]
                f = font(IMPACT, 96)
                w = int(f.getlength(s)) + 80
                st = Image.new("RGBA", (w, 150), (0, 0, 0, 0))
                sd = ImageDraw.Draw(st)
                sd.rounded_rectangle((6, 6, w - 6, 144), 16, outline=PAL["red"], width=10)
                sd.text((w / 2, 75), s, font=f, fill=PAL["red"], anchor="mm")
                cache["stamp"] = st
            sc = 1.0 + 1.4 * (1 - out_cubic(kv))
            paste_scaled(base, cache["stamp"], (x0 + 0.2 * (x1 - x0), y + 130), sc * 0.8, -8, alpha=out_cubic(kv))
    return base


# ------------------------------------------------------------------ "screen" templates (a creator's dashboard)
UI = {"bg": (16, 18, 24), "panel": (30, 33, 42), "line": (70, 76, 92), "text": (225, 228, 235), "dim": (130, 136, 150),
      "green": (60, 220, 120), "red": (240, 60, 60), "amber": (255, 184, 40)}


def screen_base(cv: Canvas, t: float, glitch: float = 0.0) -> Image.Image:
    """Dark dashboard backdrop with a subtle scanline shimmer; `glitch` 0-1 adds horizontal tearing."""
    base = Image.new("RGB", (cv.w, cv.h), UI["bg"])
    d = ImageDraw.Draw(base, "RGBA")  # blends only onto an RGB image (on RGBA it writes opaque white rows)
    for y in range(0, cv.h, 4):
        d.line([(0, y), (cv.w, y)], fill=(255, 255, 255, 5))
    base = base.convert("RGBA")
    if glitch > 0:
        rnd = np.random.default_rng(int(t * 24))
        arr = np.array(base)
        for _ in range(int(12 * glitch)):
            y0 = int(rnd.integers(0, cv.h - 20))
            hh = int(rnd.integers(2, 18))
            arr[y0:y0 + hh] = np.roll(arr[y0:y0 + hh], int(rnd.integers(-60, 60)), axis=1)
        base = Image.fromarray(arr)
    return base


def panel(d, box, title=None):
    d.rounded_rectangle(box, 18, fill=UI["panel"], outline=UI["line"], width=2)
    if title:
        text(d, (box[0] + 28, box[1] + 18), title, font(BOLD, 32), UI["dim"], shadow=False)


def counter(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    """Subscriber counter rolling up through milestones, with a climbing sparkline."""
    base = screen_base(cv, t)
    d = ImageDraw.Draw(base)
    box = (120, cv.top + 20, cv.w - 120, cv.bottom - 20)
    panel(d, box, g.get("label", "SUBSCRIBERS"))
    a, b, dur = g["from"], g["to"], g.get("roll", 4.0)
    k = out_cubic(seg(t, 0.3, dur))
    val = int(a + (b - a) * (k ** 1.6))
    text(d, (cv.w / 2, box[1] + 170), f"{val:,}", font(BLACK, 150), UI["text"], "mm", shadow=False)
    # sparkline: rising curve drawn up to k
    x0, x1, y0, y1 = box[0] + 60, box[2] - 60, box[3] - 40, box[1] + 280
    pts = [(x0 + (x1 - x0) * i / 120, y0 - (y0 - y1) * ((i / 120) ** 2.2) + 6 * math.sin(i / 5)) for i in range(121)]
    n = max(2, int(121 * k))
    d.line(pts[:n], fill=UI["green"], width=6, joint="curve")
    d.ellipse((pts[n - 1][0] - 9, pts[n - 1][1] - 9, pts[n - 1][0] + 9, pts[n - 1][1] + 9), fill=UI["green"])
    for m in g.get("milestones", []):
        if val >= m["at"]:
            km = out_elastic(seg(t, 0.3 + dur * ((m["at"] - a) / max(1, b - a)) ** (1 / 1.6), 0.5))
            if km > 0:
                if m["label"] not in cache:
                    cache[m["label"]] = starburst(200, 14, PAL["gold"])
                    ImageDraw.Draw(cache[m["label"]]).text((100, 100), m["label"], font=font(IMPACT, 44),
                                                          fill=PAL["red"], anchor="mm")
                paste_scaled(base, cache[m["label"]], (box[2] - 150 - 190 * g["milestones"].index(m), box[1] + 95),
                             km * 0.8, 10 * math.sin(t * 4))
    return base


def line_crash(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    """Views graph that climbs, then falls off a cliff to a flat zero; the stat turns red; optional verdict stamp."""
    k = seg(t, 0.2, g.get("draw", 4.0))
    crash_at = g.get("crash_at", 0.55)
    base = screen_base(cv, t, glitch=0.6 if 0 < k - crash_at < 0.08 else 0.0)
    d = ImageDraw.Draw(base)
    box = (100, cv.top + 20, cv.w - 100, cv.bottom - 20)
    panel(d, box, g.get("label", "VIEWS (LAST 90 DAYS)"))
    x0, x1, yb, yt = box[0] + 60, box[2] - 60, box[3] - 50, box[1] + 150
    for gl in range(4):
        y = yb - (yb - yt) * gl / 3
        d.line([(x0, y), (x1, y)], fill=UI["line"], width=1)

    def v(u):
        return 0.55 + 0.4 * u / crash_at + 0.04 * math.sin(u * 40) if u < crash_at else max(0.0, 0.95 * (1 - (u - crash_at) / 0.06)) if u < crash_at + 0.06 else 0.004
    n = max(2, int(200 * k))
    pts = [(x0 + (x1 - x0) * i / 199, yb - (yb - yt) * v(i / 199)) for i in range(n)]
    crashed = k > crash_at
    d.line(pts, fill=UI["red"] if crashed else UI["green"], width=6, joint="curve")
    cur = int(g.get("peak", 48210) * v(min(k, 0.999)))
    text(d, (box[2] - 40, box[1] + 60), f"{cur:,}", font(BLACK, 64), UI["red"] if crashed else UI["text"], "rm",
         shadow=False)
    if crashed:
        text(d, (box[2] - 40, box[1] + 112), g.get("delta", "-99.8%"), font(BLACK, 36), UI["red"], "rm", shadow=False)
    if g.get("verdict") and k >= 1:
        kv = seg(t, 0.2 + g.get("draw", 4.0) + 0.3, 0.35)
        if kv > 0:
            if "stamp" not in cache:
                f = font(IMPACT, 90)
                w = int(f.getlength(g["verdict"])) + 80
                st = Image.new("RGBA", (w, 140), (0, 0, 0, 0))
                sd = ImageDraw.Draw(st)
                sd.rounded_rectangle((6, 6, w - 6, 134), 16, outline=UI["red"], width=10)
                sd.text((w / 2, 70), g["verdict"], font=f, fill=UI["red"], anchor="mm")
                cache["stamp"] = st
            paste_scaled(base, cache["stamp"], (cv.w / 2, (yt + yb) / 2), 0.8 * (1 + 1.4 * (1 - out_cubic(kv))), -8,
                         alpha=out_cubic(kv))
    return base


def empty_feed(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    """A comment feed of grey placeholder rows that never loads; a spinner turns forever, then the banner."""
    base = screen_base(cv, t)
    d = ImageDraw.Draw(base)
    box = (220, cv.top + 20, cv.w - 220, cv.bottom - 20)
    panel(d, box, g.get("label", "COMMENTS (0)"))
    shimmer = (t * 0.6) % 1.0
    for i in range(5):
        y = box[1] + 80 + i * 80
        c = 60 + int(25 * max(0.0, 1 - abs(shimmer * 6 - i) / 1.5))
        d.ellipse((box[0] + 30, y, box[0] + 80, y + 50), fill=(c, c + 4, c + 12))
        d.rounded_rectangle((box[0] + 100, y + 6, box[0] + 100 + 420 - 40 * (i % 3), y + 20), 7, fill=(c, c + 4, c + 12))
        d.rounded_rectangle((box[0] + 100, y + 30, box[0] + 100 + 260 + 30 * (i % 2), y + 44), 7,
                            fill=(c - 10, c - 6, c + 2))
    cx, cy = box[2] - 60, box[1] + 36
    d.arc((cx - 16, cy - 16, cx + 16, cy + 16), int(t * 360) % 360, int(t * 360) % 360 + 270, fill=UI["dim"], width=4)
    if g.get("banner"):
        kb = out_back(seg(t, g.get("banner_at", 2.0), 0.5))
        if kb > 0:
            y = box[3] - 70 + (1 - kb) * 120
            d.rounded_rectangle((box[0] + 30, y, box[2] - 30, y + 50), 10, fill=(60, 20, 24), outline=UI["red"], width=2)
            text(d, ((box[0] + box[2]) / 2, y + 25), g["banner"], font(BOLD, 24), UI["text"], "mm", shadow=False)
    return base


def error_stack(root: Path, cv: Canvas, g: dict, t: float, cache: dict) -> Image.Image:
    """Error windows pop up one on top of another over a failing progress bar; the screen tears, then goes dark."""
    errs = g["errors"]
    t_black = g.get("black_at", 0.9 + 0.55 * len(errs) + 0.8)
    base = screen_base(cv, t, glitch=clamp((t - t_black + 0.8) / 0.8))
    d = ImageDraw.Draw(base)
    # background copy dialog with a progress bar that stalls and turns red
    bx = (cv.w / 2 - 330, cv.top + 60, cv.w / 2 + 330, cv.top + 230)
    panel(d, bx, g.get("task", "Copying 5 years of footage…"))
    p = min(0.62, 0.62 * out_cubic(seg(t, 0.1, 0.8)))
    stalled = t > 1.0
    d.rounded_rectangle((bx[0] + 30, bx[1] + 90, bx[2] - 30, bx[1] + 120), 8, fill=UI["line"])
    d.rounded_rectangle((bx[0] + 30, bx[1] + 90, bx[0] + 30 + (bx[2] - bx[0] - 60) * p, bx[1] + 120), 8,
                        fill=UI["red"] if stalled else UI["green"])
    text(d, (bx[2] - 30, bx[1] + 145), "0 bytes/s" if stalled else "58 MB/s", font(BOLD, 20), UI["dim"], "rm",
         shadow=False)
    for i, e in enumerate(errs):
        ki = out_back(seg(t, 0.9 + 0.55 * i, 0.3))
        if ki <= 0:
            continue
        w, h = 520, 190
        cx, cy = cv.w / 2 - 150 + 70 * i, (cv.top + cv.bottom) / 2 + 10 + 34 * i
        win = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        wd = ImageDraw.Draw(win)
        wd.rounded_rectangle((0, 0, w - 1, h - 1), 12, fill=(238, 238, 240), outline=(90, 90, 100), width=2)
        wd.rounded_rectangle((0, 0, w - 1, 36), 12, fill=UI["red"])
        wd.rectangle((0, 20, w - 1, 36), fill=UI["red"])
        text(wd, (16, 18), g.get("app", "Disk Utility"), font(BOLD, 18), (255, 255, 255), "lm", shadow=False)
        wd.polygon([(40, 150), (80, 80), (120, 150)], fill=UI["amber"], outline=(40, 40, 40))
        text(wd, (80, 128), "!", font(BLACK, 40), (30, 30, 30), "mm", shadow=False)
        text(wd, (150, 80), e, fit(e, BOLD, 26, 350), (30, 30, 36), "lm", shadow=False)
        wd.rounded_rectangle((w - 140, h - 50, w - 20, h - 16), 8, fill=(210, 212, 218), outline=(120, 120, 130))
        text(wd, (w - 80, h - 33), "OK", font(BOLD, 18), (30, 30, 36), "mm", shadow=False)
        paste_scaled(base, win, (cx, cy), (0.6 + 0.4 * ki) * 1.3)
    if t > t_black:  # the screen dies
        a = int(255 * clamp((t - t_black) / 0.25))
        base.alpha_composite(Image.new("RGBA", base.size, (0, 0, 0, a)))
    return base


TEMPLATES = {"blister_card": blister_card, "stat_card": stat_card, "bar_chart": bar_chart, "meter": meter,
             "counter": counter, "line_crash": line_crash, "empty_feed": empty_feed, "error_stack": error_stack}


def frames(root: Path, g: dict, w: int, h: int, duration: float):
    cv = Canvas(w, h)
    fn = TEMPLATES[g["template"]]
    cache: dict = {}
    for i in range(round(duration * FPS)):
        yield fn(root, cv, g, i / FPS, cache).convert("RGB")


def digest(root: Path, shot: dict, w: int, h: int) -> str:
    g = shot["graphic"]
    img = hashlib.sha1((root / g["image"]).read_bytes()).hexdigest() if g.get("image") else None
    src = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()  # template code changes re-render
    return hashlib.sha1(json.dumps({"mode": "motion_graphic", "graphic": g, "image": img, "code": src, "w": w, "h": h,
                                    "duration": shot["duration"]}, sort_keys=True).encode()).hexdigest()


def render(root: Path, shot: dict, w: int, h: int, dest: Path) -> None:
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                             f"{w}x{h}", "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                             "-t", str(shot["duration"]), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                             "-c:a", "aac", "-shortest", str(dest)], stdin=subprocess.PIPE)
    for fr in frames(root, shot["graphic"], w, h, shot["duration"]):
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise SystemExit(f"{shot['id']}: ffmpeg failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--only", required=True, help="comma-separated motion_graphic shot ids")
    ap.add_argument("--preview", action="store_true", help="write edit/sheets/mg_<id>.png (6 frames) instead")
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    root = a.shotlist.parent
    w, h = film.get("width", 1344), film.get("height", 768)
    for shot in film["shots"]:
        if shot["id"] not in a.only.split(","):
            continue
        if a.preview:
            ts = np.linspace(0.5, shot["duration"] - 0.05, 6)
            cv, cache = Canvas(w, h), {}
            fn = TEMPLATES[shot["graphic"]["template"]]
            tiles = [fn(root, cv, shot["graphic"], float(t), cache).convert("RGB").resize((w // 2, h // 2)) for t in ts]
            sheet = Image.new("RGB", (w, h // 2 * 3))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i % 2) * (w // 2), (i // 2) * (h // 2)))
            out = root / "edit" / "sheets" / f"mg_{shot['id']}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(out)
            print(out)
        else:
            dest = root / "renders" / f"{shot['id']}.mp4"
            render(root, shot, w, h, dest)
            print(f"+ {shot['id']}: {dest}")


if __name__ == "__main__":
    sys.exit(main())
