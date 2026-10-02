"""Cut the film in DaVinci Resolve Studio: build a timeline from the shot list, mix, render <film>.mp4.

    python tools/resolve_edit.py film/<name>/shots.json             # new timeline, render, export OTIO
    python tools/resolve_edit.py film/<name>/shots.json --no-render # timeline only, for hand editing

Resolve must be running with Preferences > General > External scripting using = Local.

One Resolve project per film ("<project_prefix><film dir>"; project_prefix defaults to the resolve_project_prefix
setting, see tools/pipeline_settings.py, else ""). Every run adds a new
timeline "<title> NN", so earlier cuts stay in the project. Layout:
  V1 "Picture"    every shot's render
  A1 "Shot audio" each shot's H3 audio, or its vocal-free stem for shots with "bed": "instruments"; nothing for
                  shots with "bed": "none" (e.g. silent B-roll that should carry only music and narration)
  A2 "Narration"  the narration clips from narrate.py, each at its shot time + "at"
  A3 "Music"      only if the film has "music": the tools/music.py score, ducked under narration and dialogue
  next "SFX"      only if some shot has "sfx": the tools/sfx.py clips at shot start + "at", "gain_db" each, not
                  ducked (one track; clips must not overlap)
  V2 "Holds"      only if some shot has a "hold": a looping, cropped stretch of the shot's own render over itself
  next "Letterbox" only if the film has "letterbox": one full-length clip of black bars over everything below
  next "Titles"   per-shot text overlays (shot "titles"; title k of a shot on its own track, since they overlap):
                  transparent PNGs in edit/titles/, wrapped as exact-length QuickTime Animation clips (Resolve
                  ignores endFrame for stills)
Film-level extras:
  "letterbox": 2.39                                                      # bars for this aspect ratio (top/bottom)
  "music": {"style": ..., "lyrics": ..., "seed": 1, "max_duration": 150, # the score (see tools/music.py);
            "gain_db": -6, "duck_db": -9, "start": 0.0, "in": 0.0,      #  level, extra dip under narration and
            "file": "music/x.mp3"}                                       #  on-screen dialogue, where it starts in
                                                                         #  the cut, how far into the score (1 s
                                                                         #  fade in when > 0), optional supplied
                                                                         #  score used instead of the YuE2 render
Per-shot extras:
  "titles": [{"text": "#10", "style": "rank", "at": 1.0, "dur": 3.0}]   # at/dur in seconds from the shot start;
            styles: rank (big yellow number, top right; "align": "left" for top left), lower (lower third on a
            dark bar), card (centred magenta lines, "\n" separated) in Impact; place (small upper-case location
            super inside the bottom letterbox bar) and doc (centred title card: big first line, smaller further
            lines, gold rules above and below) in Bahnschrift.
  "zoom": 1.25                                                           # punch-in on the V1 item (ZoomX/ZoomY)
  "pillarbox": 1.3333                                                    # crop the V1 item's sides to this visible
                                                                         #  aspect ratio (4:3 archival footage); the
                                                                         #  black timeline background shows through
  "out": 2.4                                                             # end the shot here (seconds), e.g. to drop
                                                                         #  words H3 invented after a short line
  "in": 0.9                                                              # start the shot here (render seconds), e.g.
                                                                         #  to drop words H3 invented before a line
  "hold": {"from": 3.0, "dur": 2.0, "crop": {"left": 0.58, "softness": 30}}
            # for the whole shot, lay [from, from+dur) of the same render over it, played forward and backward on a
            # loop and cropped ("crop": fractions of the frame removed per side; softness feathers the edge). Use:
            # H3 lip-syncs every mouth in frame, so a character who must stay silent keeps moving their lips; hold
            # their side on a stretch where the mouth is shut (tools/mouth_hold.py picks it). Same locked-off
            # camera and light, so the seam doesn't show. Holds inherit the shot's zoom.
Pre-made footage (a channel intro) comes in as an h3_render.py "clip" shot and is cut like any other shot; still
photographs as "still" shots, with any Ken Burns move ("kenburns") rendered into the shot by h3_render.py.
Ducking is done on the timeline, not baked into files: A1 is split around every narration line, the pieces
under the voice drop to "narration_ducking_db" (-14), and 0 dB audio cross fades make the ramps. It all stays
editable in Resolve. Loudness: render and deliver, measure the delivered file with ffmpeg ebur128, shift every
audio item by the difference to "loudness_lufs" (-16), repeat (up to LOUDNESS_PASSES) until within 0.3 LU; the
delivery limiter shaves loudness off the peaks, so one shift measured on the master lands short. A previous
<film>.mp4 is kept as <film>_iterN.mp4.

Resolve references media by path and does not notice a file rewritten in place, so every clip is imported
from a content-addressed copy, edit/media/<name>.<sha1[:10]><ext>. A re-rendered shot or re-rolled narration
take becomes a new media-pool item, and older timelines keep pointing at the takes they were cut with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import music
from narrate import cut_speech, duration, instruments_stem, plan, report, tts
from pipeline_settings import setting
from sfx import sfx_clip

os.environ.setdefault("RESOLVE_SCRIPT_API",
                      r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting")
# normpath: Windows resolves fusionscript.dll's own dependencies from its folder only for a backslash path
os.environ.setdefault("RESOLVE_SCRIPT_LIB", os.path.normpath(setting(
    "resolve_script_lib", r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll")))
sys.path.append(os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules"))

FPS = 24
LEAD_S, TAIL_S, HOLD_S = 0.15, 0.25, 0.35   # duck ahead of the first word, release after the last, bridge gaps
XFADE = 6                                    # frames per ducking ramp (0 dB audio cross fade, centred on the cut)
MIN_SEG = 2 * XFADE                          # shorter bed pieces are folded into their neighbour
NARRATION_GAIN_DB = 4.0                      # clips are normalised to -20 LUFS at TTS time; sit them above the bed
LOUDNESS_PASSES = 6                          # render/measure/shift rounds to land the delivery on "loudness_lufs"
                                             # (4 left an episode-length cut at -14.7: SFX peaks pull the limiter down)


def connect():
    import DaVinciResolveScript as dvr
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise SystemExit("Resolve not reachable: start DaVinci Resolve Studio with External scripting = Local")
    return resolve


class Media:
    """Media-pool items for content-addressed copies of the film's files (see module docstring)."""

    def __init__(self, project, root: Path):
        self.mp = project.GetMediaPool()
        self.store = root / "edit" / "media"
        self.store.mkdir(parents=True, exist_ok=True)
        self.items = {}
        stack = [self.mp.GetRootFolder()]
        while stack:
            folder = stack.pop()
            stack += folder.GetSubFolderList()
            for clip in folder.GetClipList():
                self.items[os.path.normcase(clip.GetClipProperty("File Path"))] = clip

    def get(self, path: Path):
        data = path.read_bytes()
        copy = self.store / f"{path.stem}.{hashlib.sha1(data).hexdigest()[:10]}{path.suffix}"
        if not copy.exists():
            copy.write_bytes(data)
        key = os.path.normcase(str(copy.resolve()))
        if key not in self.items:
            imported = self.mp.ImportMedia([str(copy.resolve())])
            if len(imported) != 1:
                raise SystemExit(f"Resolve could not import {copy}")
            self.items[key] = imported[0]
        return self.items[key]


def open_project(resolve, film: dict, root: Path):
    pm = resolve.GetProjectManager()
    name = film.get("project_prefix", setting("resolve_project_prefix", "")) + root.name
    project = pm.LoadProject(name)
    if not project:
        project = pm.CreateProject(name)
        for key, value in (("timelineFrameRate", str(FPS)),
                           ("timelineResolutionWidth", str(film.get("width", 1344))),
                           ("timelineResolutionHeight", str(film.get("height", 768)))):
            if not project.SetSetting(key, value):
                raise SystemExit(f"could not set project {key}={value}")
    return project


def video_tracks(film: dict) -> dict[str, int | None]:
    """Video track indices: V1 Picture, then Holds (if any shot has a "hold"), then Letterbox (if the film has
    "letterbox"), then the titles tracks from "titles" up."""
    i, tracks = 2, {"holds": None, "letterbox": None}
    if any(s.get("hold") for s in film["shots"]):
        tracks["holds"], i = i, i + 1
    if film.get("letterbox"):
        tracks["letterbox"], i = i, i + 1
    return tracks | {"titles": i}


def new_timeline(project, film: dict, media: Media):
    # Resolve refuses timeline names with some punctuation (a ":" made CreateEmptyTimeline return None).
    title = re.sub(r"[^\w .-]+", " ", film.get("title", "film")).strip()
    title = re.sub(r"\s+", " ", title)
    names = [project.GetTimelineByIndex(i + 1).GetName() for i in range(project.GetTimelineCount())]
    taken = [int(m.group(1)) for n in names if (m := re.fullmatch(re.escape(title) + r" (\d+)", n))]
    tl = media.mp.CreateEmptyTimeline(f"{title} {max(taken, default=0) + 1:02d}")
    if tl is None:
        raise SystemExit(f"Resolve could not create timeline {title!r}")
    project.SetCurrentTimeline(tl)
    if not tl.AddTrack("audio", "mono"):
        raise SystemExit("could not add the narration track")
    if "music" in film:
        if not tl.AddTrack("audio", "stereo"):
            raise SystemExit("could not add the music track")
        tl.SetTrackName("audio", 3, "Music")
    if any(s.get("sfx") for s in film["shots"]):
        if not tl.AddTrack("audio", "mono"):
            raise SystemExit("could not add the SFX track")
        tl.SetTrackName("audio", sfx_track(film), "SFX")
    tracks = video_tracks(film)
    for key, label in (("holds", "Holds"), ("letterbox", "Letterbox")):
        if tracks[key] is not None:
            if not tl.AddTrack("video"):
                raise SystemExit(f"could not add the {key} track")
            tl.SetTrackName("video", tracks[key], label)
    # A shot's titles overlap in time (rank + lower third), and one track can't hold overlapping items, so title k
    # of every shot goes on its own track: "Titles", "Titles 2", ...
    n_titles = max(1, max((len(s.get("titles", [])) for s in film["shots"]), default=0))
    for k in range(n_titles):
        if not tl.AddTrack("video"):
            raise SystemExit("could not add a titles track")
        tl.SetTrackName("video", tracks["titles"] + k, "Titles" if k == 0 else f"Titles {k + 1}")
    for kind, idx, label in (("video", 1, "Picture"), ("audio", 1, "Shot audio"), ("audio", 2, "Narration")):
        tl.SetTrackName(kind, idx, label)
    return tl


TITLE_FONT = "C:/Windows/Fonts/impact.ttf"
DOC_FONT = "C:/Windows/Fonts/bahnschrift.ttf"     # variable font: weight picked by variation name
DOC_FALLBACK = "C:/Windows/Fonts/segoeuib.ttf"    # when Pillow can't set font variations
DOC_GOLD = "#F2C230"


def doc_font(size: int, weight: str):
    try:
        font = ImageFont.truetype(DOC_FONT, size)
        font.set_variation_by_name(weight)
        return font
    except (OSError, ValueError):
        return ImageFont.truetype(DOC_FALLBACK, size)


def letterbox_bar(film: dict) -> int:
    """Rows of black bar at the top and at the bottom for the film's "letterbox" ratio (0 without one)."""
    if not film.get("letterbox"):
        return 0
    w, h = film.get("width", 1344), film.get("height", 768)
    return max(0, round((h - w / float(film["letterbox"])) / 2))


def letterbox_png(root: Path, film: dict) -> Path:
    w, h = film.get("width", 1344), film.get("height", 768)
    path = root / "edit" / "titles" / f"letterbox_{film['letterbox']}_{w}x{h}.png"
    if not path.exists():
        bar = letterbox_bar(film)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, w - 1, bar - 1), fill="black")
        d.rectangle((0, h - bar, w - 1, h - 1), fill="black")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    return path


def title_png(root: Path, film: dict, text: str, style: str, align: str = "right") -> Path:
    """Full-frame transparent PNG with the title drawn in one of the fixed styles (cached by content). `align`
    places the rank badge in the top right (default: clear of a presenter sitting left of centre) or top left."""
    w, h = film.get("width", 1344), film.get("height", 768)
    bar = letterbox_bar(film) or 52
    # place titles sit inside the letterbox bar, so their key includes it (other styles keep their old keys)
    extra = f"|{bar}" if style == "place" else ""
    key = hashlib.sha1(f"{style}|{text}|{w}|{h}|{align}{extra}".encode()).hexdigest()[:10]
    path = root / "edit" / "titles" / f"{key}.png"
    if path.exists():
        return path
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if style == "rank":
        if align not in ("right", "left"):
            raise SystemExit(f"unknown title align {align!r} (right, left)")
        d.text((w - 50, 30) if align == "right" else (60, 40), text, font=ImageFont.truetype(TITLE_FONT, 170),
               fill="#FFE600", stroke_width=12, stroke_fill="black", anchor="ra" if align == "right" else "la")
    elif style == "lower":
        bar = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(bar).rectangle((0, h - 170, w, h - 50), fill=(0, 0, 0, 160))
        img = Image.alpha_composite(img, bar)
        d = ImageDraw.Draw(img)
        d.text((60, h - 110), text, font=ImageFont.truetype(TITLE_FONT, 72), fill="white", stroke_width=8,
               stroke_fill="black", anchor="lm")
    elif style == "card":
        font = ImageFont.truetype(TITLE_FONT, 110)
        d.multiline_text((w / 2, h / 2), text, font=font, fill="#FF2BD6", stroke_width=10, stroke_fill="black",
                         anchor="mm", align="center", spacing=round(110 * 0.1))
    elif style == "place":
        d.text((60, h - bar / 2), text.upper(), font=doc_font(30, "SemiBold"), fill="white", anchor="lm")
    elif style == "doc":
        lines = text.split("\n")
        fonts = [doc_font(104 if i == 0 else 44, "Bold") for i in range(len(lines))]
        boxes = [d.textbbox((0, 0), line, font=f, anchor="lt") for line, f in zip(lines, fonts)]
        gap = 18
        block = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
        y = top = (h - block) / 2
        for line, f, b in zip(lines, fonts, boxes):
            d.text(((w - (b[2] - b[0])) / 2 - b[0], y - b[1]), line, font=f, fill="white", anchor="lt")
            y += b[3] - b[1] + gap
        bottom = top + block
        rule = max(b[2] - b[0] for b in boxes) + 80
        x0 = (w - rule) / 2
        d.rectangle((x0, top - 24 - 4, x0 + rule, top - 24 - 1), fill=DOC_GOLD)
        d.rectangle((x0, bottom + 24, x0 + rule, bottom + 24 + 3), fill=DOC_GOLD)
    else:
        raise SystemExit(f"unknown title style {style!r} (rank, lower, card, place, doc)")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def png_clip(png: Path, frames: int) -> Path:
    """The PNG as a `frames`-long QuickTime Animation (alpha) clip. Resolve places a PNG still at its default
    still duration and ignores endFrame, so overlays go in as movies of the exact length."""
    mov = png.with_name(f"{png.stem}_{frames}.mov")
    if not mov.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-frames:v", str(frames),
                        "-r", str(FPS), "-c:v", "qtrle", "-pix_fmt", "argb", str(mov)], check=True)
    return mov


def title_clip(root: Path, film: dict, text: str, style: str, frames: int, align: str = "right") -> Path:
    return png_clip(title_png(root, film, text, style, align), frames)


def music_clip(root: Path, film: dict, total_s: float) -> Path:
    """The score (tools/music.py: the JSON seed's render, or the supplied "file") from "in", cut to the programme
    after "start", with a 1.5 s fade out, and a 1 s fade in when it enters mid-track ("in" > 0)."""
    m = film["music"]
    src = music.music_path(film, root)
    if not src.exists():
        raise SystemExit(f"missing score {src}" + ("" if m.get("file") else ": run tools/music.py"))
    start, cue = float(m.get("start", 0.0)), float(m.get("in", 0.0))
    length = round(min(duration(src) - cue, total_s - start), 3)
    if length <= 1.5:
        raise SystemExit(f"music: only {length}s of score left after in={cue}, start={start}")
    out = root / "edit" / "music" / f"{src.stem}_{cue}_{length}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fades = f"afade=t=out:st={length - 1.5}:d=1.5" + (",afade=t=in:d=1" if cue > 0 else "")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(cue), "-t", str(length), "-i", str(src),
                        "-af", fades, "-ar", "48000", "-ac", "2", str(out)],
                       check=True)
    return out


def duck_windows(events: list[dict]) -> list[tuple[float, float]]:
    spans = sorted((e["start"] - LEAD_S, e["end"] + TAIL_S) for e in events)
    merged: list[list[float]] = []
    for s, t in spans:
        if merged and s - merged[-1][1] < HOLD_S:
            merged[-1][1] = max(merged[-1][1], t)
        else:
            merged.append([s, t])
    return [(s, t) for s, t in merged]


def bed_pieces(a: int, b: int, windows: list[tuple[float, float]]) -> list[tuple[int, int, bool]]:
    """Split shot frames [a, b) into (start, end, ducked) runs; runs shorter than MIN_SEG join the ducked side."""
    marks = [False] * (b - a)
    for s, t in windows:
        for f in range(max(a, round(s * FPS)), min(b, round(t * FPS))):
            marks[f - a] = True
    runs: list[list] = []
    for i, m in enumerate(marks):
        if runs and runs[-1][2] == m:
            runs[-1][1] = a + i + 1
        else:
            runs.append([a + i, a + i + 1, m])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if r[1] - r[0] < MIN_SEG and not r[2]:
                r[2] = True
                changed = True
        folded = [runs[0]]
        for r in runs[1:]:
            if r[2] == folded[-1][2]:
                folded[-1][1] = r[1]
            else:
                folded.append(r)
        runs = folded
    return [(s, t, d) for s, t, d in runs]


def place_pieces(mp, clip, origin: int, pieces: list[tuple[int, int, bool]], track: int, t0: int,
                 levels: tuple[float, float], label: str) -> list:
    """Lay (start, end, ducked) pieces of an audio clip whose frame 0 sits at timeline frame `origin` on `track`,
    at levels (normal, ducked) dB, with a 0 dB cross fade between pieces (the ducking ramps)."""
    items = mp.AppendToTimeline([{"mediaPoolItem": clip, "startFrame": s - origin, "endFrame": t - origin,
                                  "mediaType": 2, "trackIndex": track, "recordFrame": t0 + s} for s, t, _ in pieces])
    if len(items) != len(pieces):
        raise SystemExit(f"{label}: Resolve placed {len(items)} of {len(pieces)} audio pieces")
    for item, (_, _, ducked) in zip(items, pieces):
        item.SetProperty("AudioVolume", float(levels[1] if ducked else levels[0]))
    for item in items[1:]:
        if not item.AddTransition({"type": "Cross Fade 0 dB", "category": "audio", "position": "start",
                                   "alignment": "center", "duration": XFADE}):
            raise SystemExit(f"{label}: could not add a ducking cross fade")
    return items


def build(project, film: dict, root: Path, clips: list[dict], events: list[dict], media: Media):
    tl = new_timeline(project, film, media)
    shots = {s["id"]: s for s in film["shots"]}
    mp, t0 = media.mp, tl.GetStartFrame()
    duck_db = film.get("narration_ducking_db", -14)
    windows = duck_windows(events)
    frame = 0
    tracks = video_tracks(film)
    for c in clips:
        video = media.get(c["video"])
        shot = shots[c["id"]]
        n = int(video.GetClipProperty("Frames"))
        if "out" in shot:
            n = min(n, round(shot["out"] * FPS))
        head = round(shot.get("in", 0.0) * FPS)  # "in": frames dropped from the head (invented words before a line)
        (vitem,) = mp.AppendToTimeline([{"mediaPoolItem": video, "startFrame": head, "endFrame": n, "mediaType": 1,
                                         "trackIndex": 1, "recordFrame": t0 + frame}])
        n -= head
        if "zoom" in shot:
            set_zoom(vitem, float(shot["zoom"]), c["id"])
        if "pillarbox" in shot:
            set_pillarbox(vitem, film, float(shot["pillarbox"]), c["id"])
        if "hold" in shot:
            place_hold(film, root, shot, c["video"], n, t0 + frame, media, tracks["holds"])
        for k, title in enumerate(shot.get("titles", [])):
            length = round(title["dur"] * FPS)
            clip = media.get(title_clip(root, film, title["text"], title["style"], length, title.get("align", "right")))
            placed = mp.AppendToTimeline([{"mediaPoolItem": clip, "startFrame": 0, "endFrame": length,
                                           "mediaType": 1, "trackIndex": tracks["titles"] + k,
                                           "recordFrame": t0 + frame + round(title["at"] * FPS)}])
            if len(placed) != 1 or placed[0].GetDuration() != length:
                got = placed[0].GetDuration() if placed else "nothing"
                raise SystemExit(f"{c['id']}: title {title['text']!r} placed as {got}, wanted {length} frames")
        items = []
        if c["bed"] != "none":
            bed_path = instruments_stem(c["video"]) if c["bed"] == "instruments" else c["video"]
            bed = media.get(bed_path)
            usable = min(n, int(duration(bed_path) * FPS) - head)
            pieces = bed_pieces(frame, frame + usable, windows)
            items = place_pieces(mp, bed, frame - head, pieces, 1, t0, (0.0, duck_db), c["id"])
        tl.SetClipsLinked([vitem, *items], True)
        frame += n
    if tracks["letterbox"] is not None:
        clip = media.get(png_clip(letterbox_png(root, film), frame))
        placed = mp.AppendToTimeline([{"mediaPoolItem": clip, "startFrame": 0, "endFrame": frame, "mediaType": 1,
                                       "trackIndex": tracks["letterbox"], "recordFrame": t0}])
        if len(placed) != 1 or placed[0].GetDuration() != frame:
            got = placed[0].GetDuration() if placed else "nothing"
            raise SystemExit(f"letterbox placed as {got}, wanted {frame} frames")
    for e in events:
        clip = media.get(Path(e["clip"]))
        (item,) = mp.AppendToTimeline([{"mediaPoolItem": clip, "mediaType": 2, "trackIndex": 2,
                                        "recordFrame": t0 + round(e["start"] * FPS)}])
        item.SetProperty("AudioVolume", NARRATION_GAIN_DB)
    if "music" in film:
        place_music(film, root, clips, events, media, t0, frame)
    place_sfx(film, root, clips, shots, media, t0)
    return tl


def sfx_track(film: dict) -> int:
    return 4 if "music" in film else 3


def place_sfx(film: dict, root: Path, clips: list[dict], shots: dict, media: Media, t0: int) -> None:
    """Each shot "sfx" clip (tools/sfx.py) on the SFX track at the shot's cut start + "at", at "gain_db"."""
    placed_until = -1
    for c in clips:
        shot = shots[c["id"]]
        for entry in shot.get("sfx", []):
            path = sfx_clip(film, root, shot, entry)
            start = round((c["start"] + float(entry["at"])) * FPS)
            if start < placed_until:
                raise SystemExit(f"{c['id']}: SFX at {entry['at']}s overlaps the previous SFX clip")
            (item,) = media.mp.AppendToTimeline([{"mediaPoolItem": media.get(path), "mediaType": 2,
                                                  "trackIndex": sfx_track(film), "recordFrame": t0 + start}])
            item.SetProperty("AudioVolume", float(entry.get("gain_db", 0.0)))
            placed_until = start + item.GetDuration()


def place_music(film: dict, root: Path, clips: list[dict], events: list[dict], media: Media, t0: int,
                total: int) -> None:
    """The score on A3 from film music "start", ducked by "duck_db" under narration and on-screen dialogue."""
    m = film["music"]
    path = music_clip(root, film, total / FPS)
    # On-screen speech from each shot's vocal stem, only the part inside its "in"/"out" trim.
    speech = [{"start": s, "end": t} for s, t, _ in cut_speech(clips)]
    a = round(float(m.get("start", 0.0)) * FPS)
    pieces = bed_pieces(a, min(total, a + int(duration(path) * FPS)), duck_windows(events + speech))
    gain = float(m.get("gain_db", -6.0))
    place_pieces(media.mp, media.get(path), a, pieces, 3, t0, (gain, gain + float(m.get("duck_db", -9.0))), "music")


def set_zoom(item, z: float, sid: str) -> None:
    item.SetProperty("ZoomX", z)
    item.SetProperty("ZoomY", z)
    if abs(float(item.GetProperty("ZoomX")) - z) > 1e-3:
        raise SystemExit(f"{sid}: zoom not applied")


def set_pillarbox(item, film: dict, ratio: float, sid: str) -> None:
    """Crop both sides of the item so `ratio` (width/height) of the frame stays visible, centred."""
    w, h = film.get("width", 1344), film.get("height", 768)
    px = (w - h * ratio) / 2
    for side in ("Left", "Right"):
        item.SetProperty(f"Crop{side}", px)
        if abs(float(item.GetProperty(f"Crop{side}")) - px) > 0.5:
            raise SystemExit(f"{sid}: pillarbox crop not applied")


def hold_clip(root: Path, video: Path, hold: dict, frames: int) -> Path:
    """`frames` of video [from, from+dur) played forward, backward, forward... (picture only, cached by content)."""
    key = hashlib.sha1(f"{hashlib.sha1(video.read_bytes()).hexdigest()}|{hold['from']}|{hold['dur']}|{frames}"
                       .encode()).hexdigest()[:10]
    out = root / "edit" / "holds" / f"{video.stem}_{key}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        size = 2 * round(hold["dur"] * FPS)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex",
                        f"[0:v]trim=start={hold['from']}:duration={hold['dur']},setpts=PTS-STARTPTS,split[f][b];"
                        f"[b]reverse[r];[f][r]concat=n=2:v=1:a=0,loop=loop=-1:size={size},"
                        f"trim=end_frame={frames},setpts=N/{FPS}/TB[v]", "-map", "[v]", "-an", "-r", str(FPS),
                        "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def place_hold(film: dict, root: Path, shot: dict, video: Path, n: int, shot_start: int, media: Media,
               track: int) -> None:
    """The shot's hold (module docstring) over the whole shot on the Holds track, cropped, with the shot's zoom."""
    hold = shot["hold"]
    if hold["from"] + hold["dur"] > duration(video):
        raise SystemExit(f"{shot['id']}: hold {hold} runs past the end of {video.name}")
    clip = media.get(hold_clip(root, video, hold, n))
    placed = media.mp.AppendToTimeline([{"mediaPoolItem": clip, "startFrame": 0, "endFrame": n, "mediaType": 1,
                                         "trackIndex": track, "recordFrame": shot_start}])
    if len(placed) != 1 or placed[0].GetDuration() != n:
        raise SystemExit(f"{shot['id']}: Resolve did not place the hold")
    item = placed[0]
    if "zoom" in shot:
        set_zoom(item, float(shot["zoom"]), shot["id"])
    crop = hold.get("crop", {})
    w, h = film.get("width", 1344), film.get("height", 768)
    for side, size in (("Left", w), ("Right", w), ("Top", h), ("Bottom", h)):
        if side.lower() in crop:
            px = float(crop[side.lower()]) * size
            item.SetProperty(f"Crop{side}", px)
            if abs(float(item.GetProperty(f"Crop{side}")) - px) > 0.5:
                raise SystemExit(f"{shot['id']}: hold crop {side.lower()} not applied")
    if "softness" in crop:
        item.SetProperty("CropSoftness", float(crop["softness"]))


def audio_items(tl) -> list:
    return [i for k in range(1, tl.GetTrackCount("audio") + 1) for i in tl.GetItemListInTrack("audio", k)
            if i.GetType() != "transition"]


def render_master(project, edit_dir: Path, name: str) -> Path:
    """H.264 picture + 24-bit PCM audio. Resolve's own AAC encoder puts a +24 dBFS click in the first 50 ms,
    so audio leaves Resolve uncompressed and deliver() encodes it."""
    project.SetCurrentRenderFormatAndCodec("mov", "H264")
    if not project.SetRenderSettings({"SelectAllFrames": 1, "TargetDir": str(edit_dir.resolve()),
                                      "CustomName": f"{name}_master", "ExportVideo": True, "ExportAudio": True,
                                      "AudioCodec": "lpcm", "AudioSampleRate": 48000, "AudioBitDepth": 24}):
        raise SystemExit("Resolve rejected the render settings")
    job = project.AddRenderJob()
    project.StartRendering([job])
    while project.IsRenderingInProgress():
        time.sleep(1)
    status = project.GetRenderJobStatus(job)
    project.DeleteRenderJob(job)
    if status.get("JobStatus") != "Complete":
        raise SystemExit(f"render failed: {status}")
    return edit_dir / f"{name}_master.mov"


def deliver(master: Path, out: Path, crf: int = 16, ceiling_db: float = -2.0) -> None:
    """x264 at `crf` (film "crf", default 16; Resolve's API rejects every VideoQuality value, so its H.264 comes out
    at ~14 Mbps) and AAC through a peak limiter at `ceiling_db` dBFS (Fairlight's master limiter has no API). The
    limiter runs at 4x oversampling so it catches inter-sample peaks: at -14 LUFS a 48 kHz limiter still left
    -0.4 dBTP after the AAC encode. main() lowers the ceiling when the encode still overshoots -1 dBTP."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(master), "-map", "0:v:0", "-map", "0:a:0",
                    "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-af", f"aresample=192000,alimiter=limit={10 ** (ceiling_db / 20):.4f}:attack=2:release=60:"
                    "level=disabled,aresample=48000",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)], check=True)


def loudness(path: Path) -> tuple[float, float]:
    log = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-map", "0:a:0", "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = log[log.rindex("Summary:"):]
    return (float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1)),
            float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1)))


def archive_previous(root: Path) -> None:
    out = root / f"{root.name}.mp4"
    if out.exists():
        n = 1
        while (root / f"{root.name}_iter{n}.mp4").exists():
            n += 1
        out.rename(root / f"{root.name}_iter{n}.mp4")
        print(f"kept previous cut as {root.name}_iter{n}.mp4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--no-render", action="store_true", help="build the timeline only")
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    root = a.shotlist.parent
    clips, events = plan(film, root)
    for e in events:
        e["clip"] = tts(film, root, e["text"], e["seed"], e["file"], e["voice"], e["tempo"], e["delivery"])
        e["end"] = e["start"] + duration(e["clip"])
    report(film, root, clips, events)
    resolve = connect()
    project = open_project(resolve, film, root)
    media = Media(project, root)
    tl = build(project, film, root, clips, events, media)
    resolve.GetProjectManager().SaveProject()
    print(f"timeline '{tl.GetName()}' in project '{project.GetName()}': {len(clips)} shots, {len(events)} lines")
    edit_dir = root / "edit"
    edit_dir.mkdir(exist_ok=True)
    otio = edit_dir / f"{tl.GetName().replace(' ', '_')}.otio"
    if not tl.Export(str(otio.resolve()), resolve.EXPORT_OTIO, resolve.EXPORT_NONE):
        raise SystemExit("OTIO export failed")
    print(f"cut list -> {otio}")
    if a.no_render:
        return
    archive_previous(root)
    target = film.get("loudness_lufs", -16.0)
    out = root / f"{root.name}.mp4"
    # Measure the delivered file, not the master: the delivery limiter takes loudness off the peaks, so a single
    # shift computed on the master lands short. Shift every audio item by the remaining error and re-deliver.
    total, ceiling = 0.0, -2.0
    for attempt in range(LOUDNESS_PASSES):
        master = render_master(project, edit_dir, root.name)
        deliver(master, out, film.get("crf", 16), ceiling)
        lufs, peak = loudness(out)
        # AAC overshoots the limiter by more when the mix is dense: lower the ceiling by the excess and re-encode.
        while peak > -1.0 and ceiling > -6.0:
            ceiling -= peak + 1.0 + 0.2
            deliver(master, out, film.get("crf", 16), ceiling)
            lufs, peak = loudness(out)
        shift = target - lufs
        if abs(shift) <= 0.3 or attempt == LOUDNESS_PASSES - 1:
            break
        for item in audio_items(tl):
            item.SetProperty("AudioVolume", float(item.GetProperty("AudioVolume")) + shift)
        resolve.GetProjectManager().SaveProject()
        total += shift
    warn = "  !! true peak above -1 dBTP" if peak > -1.0 else ""
    warn += f"  !! {lufs:.1f} LUFS, target {target}" if abs(target - lufs) > 0.5 else ""
    print(f"rendered -> {out}  ({duration(out):.1f}s, {lufs:.1f} LUFS, true peak {peak:.1f} dBTP, "
          f"gain shift {total:+.1f} dB, limiter {ceiling:.1f} dBFS; master {master.name}){warn}")


if __name__ == "__main__":
    sys.exit(main())
