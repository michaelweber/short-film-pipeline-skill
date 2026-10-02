"""Sound effects for a cut: MiniMax H3 audio-only takes of the shot's own scene, trimmed and levelled.

    python tools/sfx.py film/<name>/shots.json                  # make missing SFX (seed from the entry or picks)
    python tools/sfx.py film/<name>/shots.json --audition 3     # render seeds 1..3 per SFX, keep the CLAP-best

shots.json, per shot:
  "sfx": [{"sound": "a big brown bear's deep, rumbling roar",      # the audio prompt: what is heard
           "action": "@bear rears up and roars at the camera.",   # what happens in the take's picture (sync)
           "at": 1.2, "dur": 2.5,                                  # seconds into the shot / clip length
           "gain_db": -2, "seed": 2}]                              # level on the SFX track; seed optional

Each take is a small-picture (448x256) H3 ref-mode render of the shot's subjects (or its keyframe as framing) and
style doing `action`, with `sound` as the only audio; only the audio is decoded (~15 s on the h3 card, front of
the queue). The clip starts at the take's first sound (leading silence removed), is `dur` long with a 0.3 s fade
out, and is loudness-normalised to -20 LUFS (raise or lower it with gain_db). Cache: sfx/<hash>.wav (+ .take.flac).
--audition scores each seed's clip with CLAP (tools/clap_score.py) against `sound` vs speech/music and stores the
winner in sfx/picks.json; an entry's own "seed" always wins over the pick. resolve_edit.py places the clips on an
"SFX" audio track (not ducked).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import h3_render
from h3_render import H3_URL, Graph, upload_image
from narrate import ASR_PY, aux_env, fetch_audio, run_graph

SFX_V = "sfx_v1"
TARGET_LUFS = -20.0


def _picks(root: Path) -> dict:
    p = root / "sfx" / "picks.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _key(entry: dict) -> str:
    return entry["sound"] + " | " + entry["action"]


def seed_of(root: Path, entry: dict) -> int:
    return int(entry.get("seed") or _picks(root).get(_key(entry), 1))


def sfx_path(shot: dict, entry: dict, root: Path, seed: int | None = None) -> Path:
    seed = seed_of(root, entry) if seed is None else seed
    key = hashlib.sha1(json.dumps([SFX_V, shot.get("subjects"), shot.get("framing") or shot.get("first_frame"),
                                   shot.get("style"), entry["sound"], entry["action"], entry["dur"], seed])
                       .encode()).hexdigest()[:16]
    return root / "sfx" / f"{key}.wav"


def _graph(film: dict, root: Path, shot: dict, entry: dict, seed: int, prefix: str) -> Graph:
    f = dict(film, prompt_format="h3_ref")
    frame = shot.get("framing") or shot.get("first_frame")
    take = {"id": f"sfx_{shot['id']}", "mode": "ref", "subjects": list(shot.get("subjects", [])), "voices": [],
            "style": shot.get("style"), "duration": max(4, math.ceil(entry["dur"] + 1.5)),
            "summary": "A short moment recorded for its sound effect.",
            "prompt": f"[Shot 1] {entry['action']} From the very first moment the sound is heard: {entry['sound']}.",
            "audio": f"{entry['sound']}. No speech, no voices, no music.", "music": "N/A", "_seed": seed}
    if frame and not take["subjects"]:
        take["framing"] = frame
    take["width"], take["height"] = h3_render.draft_size(film.get("width", 1344), film.get("height", 768), 256)
    take["_prompt"], refs = h3_render.compose_prompt(take, f)
    take["_refs"] = [str(root / p) for p in refs]
    take["_voices"] = []
    images = {p: upload_image(Path(p), base=H3_URL) for p in take["_refs"]}
    g = Graph()
    g.nodes = h3_render.build_graph(take, f, images, prefix)
    decoded = next(nid for nid, n in g.nodes.items() if n["class_type"] == "VAEDecodeAudio")
    g.add("SaveAudio", audio=[decoded, 0], filename_prefix=prefix)
    for nid in [nid for nid, n in g.nodes.items() if n["class_type"] in ("SaveVideo", "CreateVideo", "VAEDecode")]:
        del g.nodes[nid]
    return g


def sfx_clip(film: dict, root: Path, shot: dict, entry: dict, seed: int | None = None) -> Path:
    """The finished clip for one "sfx" entry (rendered and cached on first use)."""
    seed = seed_of(root, entry) if seed is None else seed
    dest = sfx_path(shot, entry, root, seed)
    if dest.exists():
        return dest
    dest.parent.mkdir(exist_ok=True)
    take = dest.with_suffix(".take.flac")
    if not take.exists():
        g = _graph(film, root, shot, entry, seed, f"h3film/{root.name}/sfx/{dest.stem}")
        fetch_audio(run_graph(g, f"sfx {shot['id']} s{seed}", H3_URL, front=True), take, H3_URL)
    d = float(entry["dur"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(take), "-af",
                    "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.05,"
                    f"atrim=0:{d},afade=t=out:st={max(0.0, d - 0.3):.3f}:d=0.3,"
                    f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=20", "-ar", "48000", "-ac", "1", str(dest)], check=True)
    return dest


def clap_rank(entry: dict, clips: list[Path]) -> list[tuple[float, Path]]:
    """(cosine similarity to `sound`, clip), best first. The softmax against speech/music/silence/ambience saturates
    at 1.00 for any take that is clearly the effect, so the ranking uses the raw similarity; a take whose
    probability for `sound` is under 0.5 sorts last. CLAP runs in the py_torch Python on the aux card."""
    env = aux_env()
    out = subprocess.run([ASR_PY, str(Path(__file__).with_name("clap_score.py")), "--label", entry["sound"],
                          "--label", "people talking", "--label", "background music", "--label", "silence",
                          "--label", "wind and river ambience", *map(str, clips)],
                         capture_output=True, text=True, env=env).stdout
    rows = [json.loads(ln) for ln in out.splitlines() if ln.startswith("{")]
    return sorted(((r["sims"][entry["sound"]] - (1.0 if r["scores"][entry["sound"]] < 0.5 else 0.0), Path(r["file"]))
                   for r in rows), key=lambda t: -t[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--only", help="comma-separated shot ids")
    ap.add_argument("--audition", type=int, default=0, help="render seeds 1..N per SFX and keep the CLAP-best")
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    root = a.shotlist.parent
    only = set(a.only.split(",")) if a.only else None
    picks = _picks(root)
    for shot in film["shots"]:
        if only and shot["id"] not in only:
            continue
        for entry in shot.get("sfx", []):
            if a.audition and not entry.get("seed"):
                seeds = range(1, a.audition + 1)
                clips = {sfx_clip(film, root, shot, entry, s): s for s in seeds}
                ranked = clap_rank(entry, list(clips))
                p, best = ranked[0]
                picks[_key(entry)] = clips[next(c for c in clips if c.resolve() == best.resolve())]
                (root / "sfx" / "picks.json").write_text(json.dumps(picks, indent=1))
                print(f"{shot['id']:6s} pick seed {picks[_key(entry)]}  CLAP {p:.2f}  "
                      f"({', '.join(f'{q:.2f}' for q, _ in ranked)})  {entry['sound'][:50]}", flush=True)
            else:
                clip = sfx_clip(film, root, shot, entry)
                print(f"{shot['id']:6s} {clip.name}  {entry['sound'][:60]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
