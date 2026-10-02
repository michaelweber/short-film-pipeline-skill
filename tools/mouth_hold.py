"""Pick the stretch of a render where a silent character's mouth is shut, and write it as the shot's "hold".

    python tools/mouth_hold.py film/<name>/shots.json --ids h10 --box 0.6,0.2,0.82,0.68
    python tools/mouth_hold.py film/<name>/shots.json --ids p10 --box 0.05,0,0.4,0.55 --side left --metric motion

H3 lip-syncs every mouth in frame, so a character who must stay silent (the puppet while the host talks) keeps
opening and moving their mouth. resolve_edit.py hides that with a hold: for the whole shot, that character's side
of the frame shows one short stretch of the same render, played forward and backward on a loop. Same locked-off
camera and light, so the seam doesn't show, and the character still sways a little, as if watching.

--box (fractions of the frame) goes tightly around the silent character's head; the crop edge is set --margin
beyond its inner side (--side right: the character is on the right, so the hold's left is cropped; --side left:
its right is cropped). The stretch minimises, over --dur seconds at 8 fps inside the box:
  --metric red     strongly red pixels: an open felt mouth (puppets). Static red in the box (a shirt logo) only
                   adds a constant; keep moving red props out of the box.
  --metric motion  mean frame-to-frame change: a still face (people; mouths, blinks and nods all count)
Written to the shot:  "hold": {"from": <start>, "dur": <dur>, "crop": {"left"|"right": <edge>, "softness": ...}}
Check the result with a close-up strip (sheet.py --crop) on the cut, not only on the raw render.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FPS = 8
W, H = 336, 192


def box_frames(video: Path, box: tuple[float, float, float, float]) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"fps={FPS},scale={W}:{H}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.int16)
    x0, y0, x1, y1 = round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H)
    return frames[:, y0:y1, x0:x1]


def red_score(f: np.ndarray) -> np.ndarray:
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return ((r > 120) & (r - g > 80) & (r - b > 60)).sum(axis=(1, 2)).astype(float)


def motion_score(f: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))
    return np.concatenate([[d[0]], d])


def best_window(score: np.ndarray, n: int) -> tuple[int, float, float]:
    """(start frame, window max, window mean) of the n-frame window with the lowest (max, mean)."""
    best = None
    for k in range(len(score) - n + 1):
        w = score[k:k + n]
        cand = (float(w.max()), float(w.mean()), k)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return best[2], best[0], best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--ids", required=True, help="comma-separated shots; all share --box")
    ap.add_argument("--box", required=True, help="x0,y0,x1,y1 around the silent character's head (fractions)")
    ap.add_argument("--side", choices=("right", "left"), default="right", help="which side the character is on")
    ap.add_argument("--metric", choices=("red", "motion"), default="red")
    ap.add_argument("--dur", type=float, default=2.0, help="seconds of footage in the loop")
    ap.add_argument("--margin", type=float, default=0.03, help="crop edge this far outside the box")
    ap.add_argument("--softness", type=float, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    box = tuple(float(v) for v in a.box.split(","))
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    shots = {s["id"]: s for s in film["shots"]}
    crop = ({"left": round(max(0.0, box[0] - a.margin), 3)} if a.side == "right"
            else {"right": round(max(0.0, 1 - box[2] - a.margin), 3)})
    for sid in a.ids.split(","):
        frames = box_frames(a.shotlist.parent / "renders" / f"{sid}.mp4", box)
        score = red_score(frames) if a.metric == "red" else motion_score(frames)
        n = round(a.dur * FPS)
        if len(score) < n:
            raise SystemExit(f"{sid}: shorter than --dur {a.dur}s")
        k, peak, mean = best_window(score, n)
        shots[sid]["hold"] = {"from": round(k / FPS, 3), "dur": a.dur, "crop": crop | {"softness": a.softness}}
        print(f"{sid}: hold {k / FPS:.2f}-{(k + n) / FPS:.2f}s  {a.metric} max {peak:.1f} mean {mean:.1f} "
              f"(shot median {float(np.median(score)):.1f}, max {float(score.max()):.1f})  crop {crop}")
    if not a.dry_run:
        a.shotlist.write_text(json.dumps(film, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
