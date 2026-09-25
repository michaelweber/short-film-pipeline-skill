"""Generate a film's score locally with YuE2 (instrumental AR LoRA + NAR LoRA) and report where its big hit lands.

    python tools/music.py film/<name>/shots.json             # render the score for the JSON seed
    python tools/music.py film/<name>/shots.json --seed 2    # render/report a candidate seed
    python tools/music.py film/<name>/shots.json --force     # re-render even if cached

shots.json (film level):
  "music": {"style": "instrumental, epic ...", "lyrics": "[intro 0:00-0:20]\n[verse 0:20-0:50]...",
            "seed": 1, "max_duration": 150,       # required
            "gain_db": -6, "duck_db": -9,        # music level, and the extra dip under narration/dialogue
            "start": 0.0, "in": 0.0,             # where the score starts in the cut / how far into the score
            "file": "music/remaster.mp3"}        # optional: a supplied score used instead of the YuE2 render
The score is cached as <film>/music/score_<hash>.flac (hash of style, lyrics, seed, max_duration and models);
resolve_edit.py lays it on A3 "Music" with the JSON seed, or the "file" when one is set. --seed N renders a
candidate without touching the JSON; adopt it by setting "seed" (and removing "file"). After a render (and on a
cached run) the tool prints the length and the hit onsets: every t >= 20 s whose momentary loudness is >= 6 LU
above the median of the previous 20 s ("hit" = the first; "hits" = all, with their rise). A fresh render, and
every run on a "file", also gets an ASR vocal check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from h3_render import Graph
from narrate import duration, fetch_audio, run_graph
from speech_qa import transcribe, words

CKPT = "yue2_3b_bf16.safetensors"
AR_LORA = "ar_lora_inst_v3abc_comfyui.safetensors"
NAR_LORA = "nar_lora_joint_v4_comfyui.safetensors"
HIT_AFTER_S, HIT_WINDOW_S, HIT_RISE_LU = 20.0, 20.0, 6.0


def music_path(film: dict, root: Path, seed: int | None = None) -> Path:
    """The score the cut uses: the supplied "file" if set, else the YuE2 render for `seed` (default: JSON seed)."""
    m = film["music"]
    if seed is None and m.get("file"):
        return root / m["file"]
    seed = m["seed"] if seed is None else seed
    key = hashlib.sha1(json.dumps([m["style"], m["lyrics"], seed, m["max_duration"], CKPT, AR_LORA, NAR_LORA])
                       .encode()).hexdigest()[:12]
    return root / "music" / f"score_{key}.flac"


def build_graph(m: dict, seed: int, prefix: str) -> Graph:
    """The Comfy template audio_yue2_text2music with its ABC (melody/chords) stage on, plus the two LoRAs."""
    g = Graph()
    ck = g.add("CheckpointLoaderSimple", ckpt_name=CKPT)
    ar = g.add("LoraLoader", model=[ck, 0], clip=[ck, 1], lora_name=AR_LORA, strength_model=1.0, strength_clip=1.0)
    nar = g.add("LoraLoader", model=[ar, 0], clip=[ar, 1], lora_name=NAR_LORA, strength_model=1.0,
                strength_clip=1.0)
    abc = g.add("YuE2GenerateABC", clip=[nar, 1], style=m["style"], lyrics=m["lyrics"], seed=seed, mode="full",
                max_abc_tokens=8192)
    gm = g.add("YuE2GenerateMusic", clip=[nar, 1], style=m["style"], lyrics=m["lyrics"], abc=[abc, 0], seed=seed,
               mode="full", max_duration=float(m["max_duration"]), temperature=1.0, top_p=0.95, top_k=100,
               repetition_penalty=1.2)
    neg = g.add("ConditioningZeroOut", conditioning=[gm, 0])
    lat = g.add("EmptyYuE2LatentAudio", seconds=[gm, 1], batch_size=1)
    ks = g.add("KSampler", model=[nar, 0], positive=[gm, 0], negative=[neg, 0], latent_image=[lat, 0], seed=seed,
               steps=32, cfg=1.0, sampler_name="dpm_2", scheduler="sgm_uniform", denoise=1.0)
    dec = g.add("VAEDecodeAudio", samples=[ks, 0], vae=[ck, 2])
    g.add("SaveAudio", audio=[dec, 0], filename_prefix=prefix)
    return g


def momentary(path: Path) -> list[tuple[float, float]]:
    """(t, momentary loudness LUFS) every 0.1 s."""
    log = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=framelog=info", "-f", "null",
                          "-"], capture_output=True, text=True).stderr
    return [(float(t), float(m)) for t, m in re.findall(r"t:\s*([\d.]+)\s+TARGET.*?M:\s*(-?[\d.]+|-inf)", log)
            if m != "-inf"]


def hits(path: Path) -> list[tuple[float, float]]:
    """(t, rise LU) of every hit onset: t >= 20 s where the momentary loudness is >= 6 LU above the median of the
    previous 20 s, keeping the first frame of each cluster (onsets less than 2 s apart are one hit)."""
    series, found = momentary(path), []
    for i, (t, m) in enumerate(series):
        if t < HIT_AFTER_S:
            continue
        before = [v for u, v in series[:i] if u >= t - HIT_WINDOW_S]
        rise = m - statistics.median(before) if before else 0.0
        if rise >= HIT_RISE_LU and (not found or t - found[-1][0] > 2.0):
            found.append((round(t, 1), round(rise, 1)))
    return found


def hit_report(path: Path) -> str:
    """The first hit onset plus every onset with its rise (the cut aligns one of them with the title card)."""
    found = hits(path)
    if not found:
        return "hit none"
    return f"hit {found[0][0]:.1f}  hits " + " ".join(f"{t:.1f}(+{r:.1f})" for t, r in found)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--seed", type=int, help="render a candidate seed (the JSON is not changed)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    root = a.shotlist.parent
    if "music" not in film:
        raise SystemExit(f'no "music" block in {a.shotlist}')
    m = film["music"]
    if a.seed is None and m.get("file"):
        src = music_path(film, root)
        if not src.exists():
            raise SystemExit(f"missing score file {src}")
        print(f"= {src} (supplied)  {duration(src):.1f}s  {hit_report(src)}  {vocal_check(src)}")
        return
    seed = m["seed"] if a.seed is None else a.seed
    dest = music_path(film, root, seed)
    if dest.exists() and not a.force:
        print(f"= {dest} (cached)  seed {seed}  {duration(dest):.1f}s  {hit_report(dest)}")
        return
    dest.parent.mkdir(exist_ok=True)
    g = build_graph(m, seed, f"h3film/{root.name}/music/{dest.stem}")
    fetch_audio(run_graph(g, f"score seed {seed}"), dest)
    report = hit_report(dest)
    print(f"\r+ {dest}  seed {seed}  {duration(dest):.1f}s  {report}  {vocal_check(dest)}")


def vocal_check(path: Path) -> str:
    """ASR on the separated vocal stem (written next to the score): clean when at most 3 words are heard."""
    heard = transcribe(path, path.with_name(path.stem + ".vocals.flac"))
    return f"heard {heard!r}  " + ("vocals: clean" if len(words(heard)) <= 3 else f"!! vocals: {heard}")


if __name__ == "__main__":
    sys.exit(main())
