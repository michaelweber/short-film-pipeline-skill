"""Generate the narration (voice-over) clips with a cloned voice and check their timing against the cut.

    python tools/narrate.py film/<name>/shots.json         # TTS missing lines, print the timing report
    python tools/narrate.py film/<name>/shots.json --plan  # timing report only, no GPU work

The film itself is cut, mixed and rendered in DaVinci Resolve by tools/resolve_edit.py, which uses the
plan/tts/report functions here.

shots.json additions:
  "narration_voice": "hero"                     # key into "voices" ({desc, audio, ref_text?})
  "narration_engine": "vibevoice"               # see ENGINES below, or "h3" (below)
  "narration_subject": "host"                   # h3 only (optional): key into "subjects"; the narrator is staged on
                                                #  camera with these photos (only the audio is kept)
  "narration_ducking_db": -14                   # how far the shot audio dips under the voice (resolve_edit.py)
  "loudness_lufs": -16                          # programme loudness target (resolve_edit.py)
  shot "narration": [{"at": 1.5, "text": "...", "seed": 7}]   # seconds from the start of that shot; may run into
                                                #  the next. "seed" (optional) overrides "narration_seed" (42) for
                                                #  one line, e.g. to re-roll a take whose voice drifted; "file"
                                                #  (optional) uses that clip instead (e.g. "narration/picked/a01.wav");
                                                #  "voice" (optional) reads the line in another "voices" key (e.g. a
                                                #  character's letter), default "narration_voice"; "tempo" (optional)
                                                #  overrides "narration_tempo"
  "narration_tempo": 1.15                       # speed factor for the finished clips (ffmpeg atempo, pitch kept;
                                                #  default 1.0); a sped-up clip is cached as <hash>_x<tempo>.wav
  "narration_isolate": true                     # keep only the voice of each clip (Mel-Band RoFormer vocal stem;
                                                #  VibeVoice sometimes adds a music bed); cached as <hash>_iso.wav
  shot "bed": "instruments"                     # use the shot's H3 audio with vocals removed (for shots whose
                                                #  H3 audio had voice-over baked in); "none" drops the shot's audio
                                                #  from the cut (music and narration only); default "full"
Engine "h3": each line is a MiniMax H3 ref-mode take (tools/h3_render.py graph, film turbo/size settings) of the
narrator reading it into a studio microphone, with the voice clip as <Audio 1>; only the audio is decoded.
Length is 1 s + words / 2.6 (min 5 s). H3 pads a line with invented words, so each take is cut to the scripted
line at its pauses (split_take: ASR on candidate spans); the raw take stays as <hash>.take.flac.
A line's "file" (e.g. a piece cut from a longer take, see split_take) is used as-is instead of any engine.
The report flags lines that overlap each other, run far into the next shot, or sit on top of on-screen
dialogue (speech spans from the vocal stem speech_qa.py writes; shots on the instrumental bed are skipped).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import h3_render
from h3_render import COMFY, Graph, http_json, upload_image, wait_for

SEPARATOR = "MelBandRoformer_fp32.safetensors"


def _vibevoice(g: Graph, text: str, ref: list, voice: dict, seed: int) -> str:
    return g.add("VibeVoiceSingleSpeakerNode", text=text, model="VibeVoice-Large", attention_type="auto",
                 quantize_llm="full precision", free_memory_after_generate=False, diffusion_steps=20, seed=seed,
                 cfg_scale=1.3, use_sampling=False, voice_to_clone=ref)


def _suite(node: str, params: dict):
    """TTS Audio Suite engine (voice cloned from the reference clip + its transcript)."""
    def build(g: Graph, text: str, ref: list, voice: dict, seed: int) -> str:
        eng = g.add(node, **params)
        cv = g.add("CharacterVoicesNode", voice_name="none", reference_text=voice.get("ref_text", ""),
                   trim_start=0.0, trim_end=0.0, customized=False, opt_audio_input=ref)
        return g.add("UnifiedTTSTextNode", TTS_engine=[eng, 0], text=text, narrator_voice="none", seed=seed,
                     opt_narrator=[cv, 0])
    return build


ENGINES = {
    "vibevoice": _vibevoice,
    "indextts2": _suite("IndexTTSEngineNode", {
        "model_path": "IndexTTS-2", "device": "auto", "emotion_alpha": 1.0, "use_random": False,
        "max_text_tokens_per_segment": 120, "interval_silence": 200, "temperature": 0.8, "top_p": 0.8, "top_k": 30,
        "do_sample": True, "length_penalty": 0.0, "num_beams": 3, "repetition_penalty": 10.0,
        "max_mel_tokens": 1500, "use_fp16": True, "use_deepspeed": False}),
}


H3_VO_SCENE = ("A quiet voice-over recording booth with dark acoustic foam walls and a warm desk lamp. {who} sits at a "
               "studio microphone on a boom arm, framed from the chest up, and reads documentary narration into it "
               "from the first frame to the last, earnest and intimate. Their hands rest on the desk; they are "
               "natural human hands with five fingers each.")
H3_VO_AUDIO = ("Dry, close-miked studio voice-over with no room echo and no music. {who}, in the voice of <Audio 1>, "
               "says exactly {what} and nothing else, in an earnest, intimate documentary-narrator tone: {lines} "
               "That is the only speech in the shot; after it, silence until the end. No other voices.")
H3_TAKE_MAX_S = 15  # H3's trained range is ~124-362 frames (5-15 s)


def h3_take_seconds(texts: list[str]) -> int:
    """Render length for an H3 take of `texts`: 1 s lead + words / 2.6 + 0.7 s per pause between lines + 3 s of
    headroom (H3 reads narration slowly and often pads the start; a 9 s take cut the last word of its second line)."""
    words = sum(len(t.split()) for t in texts)
    return max(5, math.ceil(1 + words / 2.6 + 0.7 * (len(texts) - 1) + 3))


def _h3_vo(film: dict, root: Path, texts: list[str], seed: int, prefix: str, voice_key: str | None = None) -> Graph:
    """H3 ref-mode take of the narrator reading `texts` in order (see module docstring), saving only its audio.
    `voice_key` reads them in another voice; the narration_subject stages only the narration voice."""
    voice_key = voice_key or film["narration_voice"]
    subject = film.get("narration_subject") if voice_key == film["narration_voice"] else None
    who = film["voices"][voice_key]["desc"]
    what = "one line" if len(texts) == 1 else "these lines in order, with a short pause between them,"
    audio = H3_VO_AUDIO.format(who=who, what=what, lines=" ".join(f'"{t}"' for t in texts))
    shot = {"id": "narration", "mode": "ref", "duration": min(H3_TAKE_MAX_S, h3_take_seconds(texts)),
            "subjects": [subject] if subject else [], "voices": [voice_key],
            "prompt": H3_VO_SCENE.format(who=who), "audio": audio, "_seed": seed}
    shot["_prompt"], refs = h3_render.compose_prompt(shot, film)
    shot["_refs"] = [str(root / p) for p in refs]
    shot["_voices"] = [str(root / film["voices"][voice_key]["audio"])]
    images = {p: upload_image(Path(p)) for p in shot["_refs"] + shot["_voices"]}
    g = Graph()
    g.nodes = h3_render.build_graph(shot, film, images, prefix)
    decoded = next(nid for nid, n in g.nodes.items() if n["class_type"] == "VAEDecodeAudio")
    g.add("SaveAudio", audio=[decoded, 0], filename_prefix=prefix)
    for nid in [nid for nid, n in g.nodes.items() if n["class_type"] in ("SaveVideo", "CreateVideo", "VAEDecode")]:
        del g.nodes[nid]  # picture outputs: never decoded
    return g


def run_graph(g: Graph, label: str) -> dict:
    resp = http_json("/prompt", {"prompt": g.nodes, "client_id": "narrate"})
    if resp.get("node_errors"):
        raise RuntimeError(f"{label}: {json.dumps(resp['node_errors'])[:2000]}")
    return wait_for(resp["prompt_id"], label)


def fetch_audio(entry: dict, dest: Path) -> None:
    item = next(a for o in entry["outputs"].values() for a in o.get("audio", []))
    q = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
    with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=120) as r:
        dest.write_bytes(r.read())


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out) if out not in ("", "N/A") else 0.0


def clip_path(film: dict, root: Path, text: str, seed: int | None = None, file: str | None = None,
              voice: str | None = None, tempo: float = 1.0, raw: bool = False) -> Path:
    """Where the line's clip lives: the line's own "file" if set, else the engine's cache path. The engine's clip is
    <hash>.wav (`raw`); "narration_isolate" derives <hash>_iso.wav from it, a tempo other than 1 <...>_x<tempo>.wav."""
    if file:
        return root / file
    voice_key = voice or film["narration_voice"]
    ref = root / film["voices"][voice_key]["audio"]
    seed = film.get("narration_seed", 42) if seed is None else seed
    engine = film.get("narration_engine", "vibevoice")
    extra = [H3_VO_SCENE, H3_VO_AUDIO, film.get("narration_subject"), h3_render.render_settings({"mode": "ref"}, film)
             ] if engine == "h3" else []
    if voice_key != film["narration_voice"]:
        extra.append(voice_key)  # only for other voices, so the narration voice's cached clips keep their names
    key = hashlib.sha1(json.dumps([text, engine, seed, hashlib.sha1(ref.read_bytes()).hexdigest(), *extra])
                       .encode())
    path = root / "narration" / f"{key.hexdigest()[:16]}.wav"
    if raw:
        return path
    if film.get("narration_isolate"):
        path = path.with_name(f"{path.stem}_iso.wav")
    return path if tempo == 1.0 else path.with_name(f"{path.stem}_x{tempo:g}.wav")


def dialogue_spans(video: Path) -> list[tuple[float, float]]:
    """On-screen speech intervals (seconds into the shot) from the vocal stem speech_qa.py writes."""
    stem = video.with_name(video.stem + ".vocals.flac")
    if not stem.exists() or stem.stat().st_mtime < video.stat().st_mtime:
        return []
    log = subprocess.run(["ffmpeg", "-i", str(stem), "-af", "silencedetect=n=-35dB:d=0.3", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [0.0] + [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    total = duration(stem)
    spans = [(e, s) for e, s in zip(ends, starts + [total]) if s - e > 0.25]
    return spans


def tts(film: dict, root: Path, text: str, seed: int | None = None, file: str | None = None,
        voice: str | None = None, tempo: float = 1.0) -> Path:
    """The line's finished clip: the engine's take (cached), then its isolated voice ("narration_isolate") and tempo
    as cached derivatives, so neither re-renders the take."""
    if file:
        dest = root / file
        if not dest.exists():
            raise SystemExit(f"missing narration file {dest}")
        return dest
    dest = clip_path(film, root, text, seed, None, voice, tempo)
    if dest.exists():
        return dest
    clip = _engine_clip(film, root, text, seed, voice)
    if film.get("narration_isolate"):
        iso = clip.with_name(f"{clip.stem}_iso.wav")
        if not iso.exists():
            isolate_voice(clip, iso)
        clip = iso
    if tempo != 1.0:
        # Sped-up copy (pitch kept); the slower clips stay cached beside it.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-af", f"atempo={tempo}", str(dest)],
                       check=True)
        clip = dest
    return clip


def _engine_clip(film: dict, root: Path, text: str, seed: int | None, voice_key: str | None) -> Path:
    voice = film["voices"][voice_key or film["narration_voice"]]
    engine = film.get("narration_engine", "vibevoice")
    ref = root / voice["audio"]
    seed = film.get("narration_seed", 42) if seed is None else seed
    dest = clip_path(film, root, text, seed, None, voice_key, raw=True)
    if dest.exists():
        return dest
    dest.parent.mkdir(exist_ok=True)
    prefix = f"h3film/{root.name}/narration/{dest.stem}"
    flac = dest.with_suffix(".flac")
    if engine == "h3":
        # The raw take is kept (<hash>.take.flac), so re-trimming never re-renders it.
        take = dest.with_suffix(".take.flac")
        if not take.exists():
            fetch_audio(run_graph(_h3_vo(film, root, [text], seed, prefix, voice_key), f"h3 {text[:30]}"), take)
        cut(take, *split_take(take, [text])[0], flac)
    else:
        g = Graph()
        audio = ENGINES[engine](g, text, [g.add("LoadAudio", audio=upload_image(ref)), 0], voice, seed)
        g.add("SaveAudio", audio=[audio, 0], filename_prefix=prefix)
        fetch_audio(run_graph(g, f"{engine} {text[:30]}"), flac)
    finish_clip(flac, dest)
    flac.unlink()
    print(f"\r+ narration {dest.name}: {duration(dest):.1f}s  {text[:60]}")
    return dest


def isolate_voice(src: Path, dest: Path) -> None:
    """The vocal stem of a narration clip (Mel-Band RoFormer), re-normalised and trimmed like any clip. VibeVoice
    sometimes invents a music/ambience bed under the voice; this drops it."""
    g = Graph()
    a = g.add("LoadAudio", audio=upload_image(src))
    m = g.add("MelBandRoFormerModelLoader", model_name=SEPARATOR)
    s = g.add("MelBandRoFormerSampler", model=[m, 0], audio=[a, 0])
    g.add("SaveAudio", audio=[s, 0], filename_prefix=f"h3film/stems/{src.stem}_voice")
    flac = dest.with_suffix(".flac")
    fetch_audio(run_graph(g, f"isolate {src.stem}"), flac)
    finish_clip(flac, dest)
    flac.unlink()
    print(f"\r+ narration {dest.name}: voice isolated")


def finish_clip(src: Path, dest: Path) -> None:
    """Normalise loudness first (whispers are otherwise below any fixed gate), then trim leading/trailing silence
    so "at" means "the first word lands here". Falls back to the untrimmed clip if trimming ate it."""
    norm = dest.with_name(dest.stem + ".norm.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af", "loudnorm=I=-20:TP=-2",
                    "-ar", "48000", "-ac", "1", str(norm)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(norm), "-af",
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse", str(dest)], check=True)
    if duration(dest) < 0.3:
        norm.replace(dest)
    else:
        norm.unlink()


def cut(src: Path, a: float, b: float, dest: Path) -> None:
    """[a, b) of src with 10 ms fades (a and b sit in envelope dips, see dips())."""
    length = b - a
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-t", f"{length:.3f}", "-i", str(src),
                    "-af", f"afade=t=in:d=0.01,afade=t=out:st={max(0.0, length - 0.01):.3f}:d=0.01", str(dest)],
                   check=True)


def dips(take: Path) -> list[float]:
    """Candidate cut points in a voice take: local minima of the speech envelope (10 ms hop, 100 ms smoothing) at
    least 6 dB under the take's median speech level and 0.12 s apart, plus both ends. Relative, because TTS takes
    barely pause between sentences (VibeVoice's gaps sit around -30 dBFS, not silence)."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(take), "-ac", "1", "-ar", "16000", "-f", "f32le",
                          "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop = 160
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
    db = 20 * np.log10(np.convolve(rms, np.ones(10) / 10, mode="same") + 1e-9)
    speech = np.median(db[db > db.max() - 40])
    minima = [i for i in range(1, n - 1) if db[i] <= db[i - 1] and db[i] <= db[i + 1] and db[i] < speech - 6]
    picked: list[int] = []
    for i in sorted(minima, key=lambda i: db[i]):
        if all(abs(i - j) >= 12 for j in picked):
            picked.append(i)
    return sorted({0.0, duration(take), *(i * hop / 16000 for i in picked)})


def split_take(take: Path, texts: list[str], min_recall: float = 0.9) -> list[tuple[float, float]]:
    """Find each scripted line of `texts` (in order) in one take; TTS and H3 takes pad or run lines together.
    Cut points are envelope dips (dips()). For each line, the target is the recall of the line's words in the whole
    rest of the take (ASR); the line starts at the latest dip whose rest still reaches that target and ends at the
    earliest dip after it whose span does, so no word is dropped at either end. Both are binary searches (recall
    only grows as a span widens). A line whose rest of the take holds under `min_recall` of its words gets the
    whole remaining span, with a warning."""
    from speech_qa import score, transcribe
    points = dips(take)
    total = points[-1]
    heard: dict[tuple[str, float, float], float] = {}
    spans, done = [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        def recall(text: str, a: float, b: float) -> float:
            if (text, a, b) not in heard:
                part = Path(tmp) / f"{len(heard)}.wav"
                cut(take, a, b, part)
                heard[(text, a, b)] = score(text, transcribe(part, None))[0]
            return heard[(text, a, b)]

        def last_true(xs: list[float], ok) -> float | None:
            lo, hi, best = 0, len(xs) - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                if ok(xs[mid]):
                    best, lo = xs[mid], mid + 1
                else:
                    hi = mid - 1
            return best

        for text in texts:
            target = recall(text, done, total)
            if target < min_recall:
                print(f"\n!! take holds only {target:.2f} of the line: {text[:60]}")
                spans.append((done, total))
                continue
            a = last_true([p for p in points if done <= p < total], lambda s: recall(text, s, total) >= target)
            a = done if a is None else a
            ends = [p for p in points if p > a][::-1]  # descending: "ok" holds for a prefix
            b = last_true(ends, lambda e: recall(text, a, e) >= target) or total
            spans.append((a, b))
            done = b
    # A dip can sit inside a soft word ending ("babies" -> "baby") or before a soft onset, and ASR still hears the
    # word whole. Err long (overlaps are cheap to fix in the edit): LEAD_S before and TAIL_S after each line, never
    # past the neighbouring lines' own boundaries.
    prev_ends = [0.0] + [b for _, b in spans[:-1]]
    next_starts = [a for a, _ in spans[1:]] + [total]
    return [(max(pe, a - LEAD_S) if a > pe else a, min(ns, b + TAIL_S) if ns > b else b)
            for (a, b), pe, ns in zip(spans, prev_ends, next_starts)]


LEAD_S, TAIL_S = 0.15, 0.6


def instruments_stem(video: Path) -> Path:
    stem = video.with_name(video.stem + ".instruments.flac")
    if stem.exists() and stem.stat().st_mtime >= video.stat().st_mtime:
        return stem
    wav = video.with_name(video.stem + ".bed_src.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
                    str(wav)], check=True)
    g = Graph()
    a = g.add("LoadAudio", audio=upload_image(wav))
    m = g.add("MelBandRoFormerModelLoader", model_name=SEPARATOR)
    s = g.add("MelBandRoFormerSampler", model=[m, 0], audio=[a, 0])
    g.add("SaveAudio", audio=[s, 1], filename_prefix=f"h3film/stems/{video.stem}_instruments")
    fetch_audio(run_graph(g, f"stem {video.stem}"), stem)
    wav.unlink()
    return stem


def plan(film: dict, root: Path) -> tuple[list[dict], list[dict]]:
    """Returns (clips with start/end, narration events with absolute start). A shot's "out" (seconds) ends it
    early in the cut, e.g. to drop words H3 invented after the scripted line."""
    clips, events, t = [], [], 0.0
    for shot in film["shots"]:
        video = root / "renders" / f"{shot['id']}.mp4"
        if not video.exists():
            raise SystemExit(f"missing render {video} (run h3_render.py first)")
        d = duration(video)
        if "out" in shot:
            d = min(d, round(shot["out"] * 24) / 24)
        clips.append({"id": shot["id"], "video": video, "start": t, "end": t + d, "bed": shot.get("bed", "full")})
        for line in shot.get("narration", []):
            events.append({"shot": shot["id"], "start": t + line["at"], "text": line["text"], "seed": line.get("seed"),
                           "file": line.get("file"), "voice": line.get("voice"),
                           "tempo": float(line.get("tempo", film.get("narration_tempo", 1.0)))})
        t += d
    return clips, events


def report(film: dict, root: Path, clips: list[dict], events: list[dict]) -> None:
    """Print the narration timeline with overlap / dialogue warnings. Events with an "end" are checked."""
    ends = {c["id"]: c["end"] for c in clips}
    # Speech after a shot's "out" never reaches the cut, so spans are clipped to the trimmed length.
    speech = [(c["start"] + s, c["start"] + min(t, c["end"] - c["start"]), c["id"]) for c in clips
              if c["bed"] == "full" for s, t in dialogue_spans(c["video"]) if s < c["end"] - c["start"]]
    for i, e in enumerate(events):
        warn = ""
        if "end" in e:
            if i + 1 < len(events) and e["end"] > events[i + 1]["start"] - 0.2:
                warn = f"  !! overlaps next line by {e['end'] - events[i + 1]['start'] + 0.2:.1f}s"
            elif e["end"] > ends[e["shot"]] + 2.5:
                warn = f"  (runs {e['end'] - ends[e['shot']]:.1f}s into the next shot)"
            hits = [(s, t, sid) for s, t, sid in speech if min(e["end"], t) - max(e["start"], s) > 0.15]
            if hits:
                warn += "  !! over on-screen dialogue " + ", ".join(f"{sid}@{s:.1f}-{t:.1f}" for s, t, sid in hits)
        span = f"{e['start']:6.1f}-{e.get('end', e['start']):6.1f}"
        print(f"{span}  {e['shot']:>4}  {e['text'][:70]}{warn}")
    print(f"film length {clips[-1]['end']:.1f}s, {len(events)} narration lines")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--plan", action="store_true", help="timing report only; no TTS")
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    root = a.shotlist.parent
    clips, events = plan(film, root)
    for e in events:
        clip = (clip_path(film, root, e["text"], e["seed"], e["file"], e["voice"], e["tempo"]) if a.plan
                else tts(film, root, e["text"], e["seed"], e["file"], e["voice"], e["tempo"]))
        if clip.exists():
            e["end"] = e["start"] + duration(clip)
    report(film, root, clips, events)


if __name__ == "__main__":
    sys.exit(main())
