"""Generate the narration (voice-over) clips with a cloned voice and check their timing against the cut.

    python tools/narrate.py film/<name>/shots.json         # TTS missing lines, print the timing report
    python tools/narrate.py film/<name>/shots.json --plan  # timing report only, no GPU work

The film itself is cut, mixed and rendered in DaVinci Resolve by tools/resolve_edit.py, which uses the
plan/tts/report functions here.

shots.json additions:
  "narration_voice": "narrator"                 # key into "voices" ({desc, audio, ref_text?})
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
                                                #  overrides "narration_tempo"; "delivery" (optional, h3 guide
                                                #  format) is the tone the line is read in, e.g. "a slow, solemn,
                                                #  dramatic tone" (a "slow" delivery gets a longer take)
  "narration_delivery": "..."                   # default tone for every line (unset: an earnest, intimate
                                                #  documentary-narrator tone at a natural pace)
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
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import h3_render
from h3_render import AUX_URL, H3_URL, Graph, http_json, upload_image, view, wait_for
from pipeline_settings import setting

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
               "from the first frame to the last, earnest and intimate. His hands rest on the desk; they are "
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


DEFAULT_DELIVERY = "an earnest, intimate documentary-narrator tone at a natural pace"


def _h3_vo(film: dict, root: Path, texts: list[str], seed: int, prefix: str, voice_key: str | None = None,
           delivery: str | None = None) -> Graph:
    """H3 ref-mode take of the narrator reading `texts` in order (see module docstring), saving only its audio.
    `voice_key` reads them in another voice; the narration_subject stages only the narration voice."""
    voice_key = voice_key or film["narration_voice"]
    subject = film.get("narration_subject") if voice_key == film["narration_voice"] else None
    who = film["voices"][voice_key]["desc"]
    seconds = h3_take_seconds(texts)
    if film.get("narration_h3_guide"):
        # MiniMax's official Ref2VA format (docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md): spoken words only inside
        # <d>[English] ...</d>, speaker IDs, <Audio 1> as a timbre reference that is not copied, and a take barely
        # longer than the line: H3 fills spare time with babble / the reference clip's own words.
        f = dict(film, prompt_format="h3_ref")
        spk = "@" + subject if subject else who
        lines = " ".join(f"<d>[English] {t}</d>" for t in texts)
        delivery = delivery or film.get("narration_delivery") or DEFAULT_DELIVERY
        wps = 2.0 if "slow" in delivery else 2.6  # a slow, solemn read needs room, or H3 rushes it
        from speech_qa import words as spoken  # "fifty-six" / "56" count as two spoken words, not one
        n_words = sum(len(spoken(t)) for t in texts)
        shot = {"id": "narration", "mode": "ref", "subjects": [subject] if subject else [], "voices": [voice_key],
                # +1 s: a slow read ran past a tight take's end ("fifty thousa|"); babble after a pause in the spare
                # second is cut off at the pause (split_take)
                "duration": min(H3_TAKE_MAX_S, max(4, math.ceil(1.8 + n_words / wps + 0.5 * (len(texts) - 1)))),
                "summary": f"A static close-up of {spk} (S1) recording a voice-over in a quiet studio booth, using "
                           "<Audio 1> as the voice-timbre reference.",
                "prompt": f"The target video is a plain, static documentary-interview shot with a warm desk lamp.\n"
                          f"[Shot 1] A static medium close-up of {spk} (S1) sitting at a desk in a quiet recording "
                          f"booth with dark acoustic foam walls, looking into the lens, his hands resting on the desk. "
                          f"From the very first moment, using the voice timbre referenced from <Audio 1>, {spk} (S1) "
                          f"says in {delivery}: {lines} "
                          f"He then stays silent with his lips closed until the end.",
                "audio": "Dry, quiet voice-over booth room tone with no echo and no other sounds.",
                "music": "N/A", "_seed": seed}
        film = f
    else:
        what = "one line" if len(texts) == 1 else "these lines in order, with a short pause between them,"
        audio = H3_VO_AUDIO.format(who=who, what=what, lines=" ".join(f'"{t}"' for t in texts))
        shot = {"id": "narration", "mode": "ref", "duration": min(H3_TAKE_MAX_S, seconds),
                "subjects": [subject] if subject else [], "voices": [voice_key],
                "prompt": H3_VO_SCENE.format(who=who), "audio": audio, "_seed": seed}
    if film.get("narration_h3_short_side"):  # only the audio is kept: denoise a small picture (audio latent unchanged)
        shot["width"], shot["height"] = h3_render.draft_size(film.get("width", 1344), film.get("height", 768),
                                                             int(film["narration_h3_short_side"]))
    shot["_prompt"], refs = h3_render.compose_prompt(shot, film)
    shot["_refs"] = [str(root / p) for p in refs]
    shot["_voices"] = [str(root / film["voices"][voice_key]["audio"])]
    images = {p: upload_image(Path(p), base=H3_URL) for p in shot["_refs"] + shot["_voices"]}
    g = Graph()
    g.nodes = h3_render.build_graph(shot, film, images, prefix)
    decoded = next(nid for nid, n in g.nodes.items() if n["class_type"] == "VAEDecodeAudio")
    g.add("SaveAudio", audio=[decoded, 0], filename_prefix=prefix)
    for nid in [nid for nid, n in g.nodes.items() if n["class_type"] in ("SaveVideo", "CreateVideo", "VAEDecode")]:
        del g.nodes[nid]  # picture outputs: never decoded
    return g


def run_graph(g: Graph, label: str, base: str = AUX_URL, front: bool = False) -> dict:
    """Runs `g` on `base`: the aux instance (:8189) for TTS/separation/music; H3 takes pass H3_URL. `front` jumps the
    queue (a small-picture VO take runs ~15 s; waiting behind multi-minute video renders would dominate it)."""
    resp = http_json("/prompt", {"prompt": g.nodes, "client_id": "narrate", "front": front}, base=base)
    if resp.get("node_errors"):
        raise RuntimeError(f"{label}: {json.dumps(resp['node_errors'])[:2000]}")
    return wait_for(resp["prompt_id"], label, base=base)


def fetch_audio(entry: dict, dest: Path, base: str = AUX_URL) -> None:
    """Downloads the job's audio from `base`, the instance that ran it (each has its own output dir)."""
    item = next(a for o in entry["outputs"].values() for a in o.get("audio", []))
    dest.write_bytes(view(item, base))


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out) if out not in ("", "N/A") else 0.0


def clip_path(film: dict, root: Path, text: str, seed: int | None = None, file: str | None = None,
              voice: str | None = None, tempo: float = 1.0, raw: bool = False, delivery: str | None = None) -> Path:
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
    if engine == "h3" and film.get("narration_h3_guide"):
        extra.append("guide_v1")  # official-format takes; the old free-text takes stay cached under their names
        delivery = delivery or film.get("narration_delivery")
        if delivery:  # a line's "delivery" (tone) or the film's "narration_delivery"; unset keeps the default's names
            extra.append(["delivery", delivery])
    def cached(parts: list) -> Path:
        key = hashlib.sha1(json.dumps([text, engine, seed, hashlib.sha1(ref.read_bytes()).hexdigest(), *parts])
                           .encode())
        return root / "narration" / f"{key.hexdigest()[:16]}.wav"

    path = cached(extra)
    if engine == "h3" and film.get("narration_h3_short_side"):
        # Small-picture takes only for lines without a clip yet: a full-size take already made stays in use.
        small = cached([*extra, ["short_side", int(film["narration_h3_short_side"])]])
        path = path if path.exists() and not small.exists() else small
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
        voice: str | None = None, tempo: float = 1.0, delivery: str | None = None) -> Path:
    """The line's finished clip: the engine's take (cached), then its isolated voice ("narration_isolate") and tempo
    as cached derivatives, so neither re-renders the take."""
    if file:
        dest = root / file
        if not dest.exists():
            raise SystemExit(f"missing narration file {dest}")
        return dest
    clip = _engine_clip(film, root, text, seed, voice, delivery)  # cached; re-cuts (dropping derived copies) if stale
    dest = clip_path(film, root, text, seed, None, voice, tempo, delivery=delivery)
    if dest.exists():
        return dest
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

CUT_VERSION = 9  # 7: cut at the real pauses (>= PAUSE_S) around the first/last words; v2-6 cut from Whisper's
#                     word edges and clipped final consonants ("ea|st", "giant|s") and "fifty [thousand]"
#                  8: number/compound words align by all their tokens (v7 sent mg03 to the dip search)
#                  9: "50" + ",000" word tokens merged (v8 sent bs08 to the dip search; "thousand" clipped)


def cut_info(clip: Path) -> dict:
    """The .cutv sidecar of an H3 narration clip: cutter version, cut points in the take, and the take's level just
    inside the cut end (dB re its peak). {} when missing or written by an older cutter (a bare version number)."""
    try:
        info = json.loads(Path(clip).with_suffix(".cutv").read_text())
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _engine_clip(film: dict, root: Path, text: str, seed: int | None, voice_key: str | None,
                 delivery: str | None = None) -> Path:
    voice = film["voices"][voice_key or film["narration_voice"]]
    engine = film.get("narration_engine", "vibevoice")
    ref = root / voice["audio"]
    seed = film.get("narration_seed", 42) if seed is None else seed
    dest = clip_path(film, root, text, seed, None, voice_key, raw=True, delivery=delivery)
    take = dest.with_suffix(".take.flac")
    marker = dest.with_suffix(".cutv")  # {"v": CUT_VERSION, "end_db": take level just inside the cut end}
    stale_cut = engine == "h3" and take.exists() and cut_info(dest).get("v") != CUT_VERSION
    if dest.exists() and not stale_cut:
        return dest
    for derived in dest.parent.glob(f"{dest.stem}_*.wav"):  # tempo/isolated copies of an old cut
        derived.unlink()
    dest.parent.mkdir(exist_ok=True)
    prefix = f"h3film/{root.name}/narration/{dest.stem}"
    flac = dest.with_suffix(".flac")
    if engine == "h3":
        # The raw take is kept (<hash>.take.flac), so re-trimming never re-renders it; a clip cut by an older
        # cutter (CUT_VERSION) is re-cut from it.
        if not take.exists():
            small = bool(film.get("narration_h3_short_side"))
            g = _h3_vo(film, root, [text], seed, prefix, voice_key, delivery)
            fetch_audio(run_graph(g, f"h3 {text[:30]}", H3_URL, front=small), take, H3_URL)
        a, b = split_take(take, [text])[0]
        cut(take, a, b, flac)
        lv = _levels(take)
        end_db = max(lv[min(max(i, 0), len(lv) - 1)] for i in (int(b / 0.02) - 2, int(b / 0.02) - 1))
        timed = word_times(take)
        aligned = bool(timed) and split_take_words(take, [text], timed) is not None
        marker.write_text(json.dumps({"v": CUT_VERSION, "a": round(a, 3), "b": round(b, 3),
                                      "end_db": round(end_db, 1), "aligned": aligned}))
    else:
        g = Graph()
        audio = ENGINES[engine](g, text, [g.add("LoadAudio", audio=upload_image(ref, base=AUX_URL)), 0], voice, seed)
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
    a = g.add("LoadAudio", audio=upload_image(src, base=AUX_URL))
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
    """[a, b) of src with a 10 ms fade in and a 60 ms fade out (an end cut can fall inside a held sound after the
    last word, see _snap; 10 ms clicked there)."""
    length = b - a
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-t", f"{length:.3f}", "-i", str(src),
                    "-af", f"afade=t=in:d=0.01,afade=t=out:st={max(0.0, length - 0.06):.3f}:d=0.06", str(dest)],
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


ASR_PY = setting("py_torch", sys.executable)  # a Python with torch + transformers
AUX_GPU = setting("aux_gpu_uuid")  # the aux card's nvidia-smi UUID (tools/gpu_queue.py status); None: no pinning


def aux_env(gpu: str | None = None) -> dict:
    """os.environ with CUDA_VISIBLE_DEVICES pinned to `gpu` (default AUX_GPU); unpinned when neither is set."""
    env = dict(os.environ)
    if gpu or AUX_GPU:
        env["CUDA_VISIBLE_DEVICES"] = gpu or AUX_GPU
    return env


_ASR_WORKER: subprocess.Popen | None = None


_ASR_MEMO: dict[tuple[str, int], dict] = {}


def _asr(take: Path) -> dict | None:
    """tools/asr_words.py output for `take` (memoised by path + mtime), from a persistent worker on the aux card (model
    loaded once; ~2.5 s per take for both decodes), or None if that path is unavailable."""
    global _ASR_WORKER
    key = (str(take.resolve()), take.stat().st_mtime_ns)
    if key in _ASR_MEMO:
        return _ASR_MEMO[key]
    try:
        if _ASR_WORKER is None or _ASR_WORKER.poll() is not None:
            env = aux_env(os.environ.get("ASR_WORDS_GPU"))
            _ASR_WORKER = subprocess.Popen([ASR_PY, str(Path(__file__).with_name("asr_words.py")), "--serve"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           text=True, env=env)
            while True:  # skip anything printed before the ready line
                line = _ASR_WORKER.stdout.readline()
                if not line or line.startswith('{"ready"'):
                    break
        _ASR_WORKER.stdin.write(f"{take.resolve()}\n")
        _ASR_WORKER.stdin.flush()
        while True:
            line = _ASR_WORKER.stdout.readline()
            if not line:
                return None
            if line.startswith("{"):
                out = json.loads(line)
                if "error" in out:
                    return None
                _ASR_MEMO[key] = out
                return out
    except (OSError, ValueError):
        return None


def word_times(take: Path, plain: bool = False) -> list[tuple[str, float, float]] | None:
    """Words of a take with start/end seconds: the verbatim-prompted decode (stutters, repeats, fillers and babble
    kept as words), or with `plain` the default decode. None if the local ASR is unavailable. Whisper's word tokens
    split a number at its thousands comma ("50" + ",000"); they are merged back into one word ("50,000"), since
    ",000" alone normalises to "zero" and broke alignment (bs08 fell back to the dip search and lost "thousand")."""
    out = _asr(take)
    if out is None:
        return None
    merged: list[tuple[str, float, float]] = []
    for w, a, b in out["plain_words" if plain else "words"]:
        if merged and re.match(r"^,\d{3}", w) and re.search(r"\d$", merged[-1][0]):
            merged[-1] = (merged[-1][0] + w, merged[-1][1], b)
        else:
            merged.append((w, a, b))
    return merged


def split_take_words(take: Path, texts: list[str], words: list[tuple[str, float, float]]
                     ) -> list[tuple[float, float, float, float]] | None:
    """Each line's (first word start, first word end, last word start, last word end) from word timestamps: align
    the script's words to the heard words (difflib, in order). None when a line is not found (fewer than 75 % of
    its words), so the caller falls back to the dip search."""
    import difflib
    from speech_qa import words as norm
    # every spoken token of every heard word, mapped back to the word ("56" -> fifty six, "100%" -> one hundred
    # percent; keeping only the first token dropped mg03 below the match threshold and into the dip search)
    toks = [(t, i) for i, (w, _, _) in enumerate(words) for t in (norm(w) or [""])]
    heard = [t for t, _ in toks]
    spans, pos = [], 0
    for text in texts:
        want = norm(text)
        m = difflib.SequenceMatcher(None, heard[pos:], want, autojunk=False)
        hits = [(blk.a + pos + k) for blk in m.get_matching_blocks() for k in range(blk.size)]
        if len(hits) < 0.75 * len(want):
            return None
        first, last = toks[min(hits)][1], toks[max(hits)][1]
        spans.append((words[first][1], words[first][2], words[last][1], words[last][2]))
        pos = max(hits) + 1
    return spans


PAUSE_S = 0.12  # a real pause between phrases; stop closures inside a word ("gian|ts", "ea|st") are 40-60 ms


def _pause_cut(levels: list[float], anchor: float, direction: int, reach: float, floor_db: float = -38.0,
               win: float = 0.02) -> float | None:
    """The line's edge at the first real pause (>= PAUSE_S below `floor_db`) found from `anchor` moving away from
    the line (+1: from the last word's start, forward; -1: from the first word's end, backward), within `reach`
    seconds. Returns a point 60 ms into the pause (the clip keeps the word's whole decay and final consonants),
    the take's edge if the take ends first, or None if no pause comes within reach."""
    n = len(levels)
    need = round(PAUSE_S / win)
    i, run = int(anchor / win), 0
    steps = round(reach / win)
    for _ in range(steps):
        if not 0 <= i < n:
            return 0.0 if direction < 0 else n * win
        run = run + 1 if levels[i] < floor_db else 0
        if run >= need:
            start = i - direction * (need - 1)  # first quiet window of the run, on the line's side
            return max(0.0, min(n * win, (start + direction * 3) * win + (win if direction > 0 else 0)))
        i += direction
    return None


def _snap(levels: list[float], t: float, direction: int, limit: float, sound_max: float, quiet_pad: float,
          floor_db: float = -38.0, win: float = 0.02) -> float:
    """A cut point near Whisper's word edge `t`, moving away from the line (`direction` +1 = later for an end, -1 =
    earlier for a start), never past `limit` (the neighbouring heard word) or the take's edge. `levels`: dB of each
    20 ms window relative to the take's peak (_levels). It follows the sound itself while it stays above
    `floor_db`, up to `sound_max` seconds (Whisper's edges land inside a decaying last word or a soft onset, and
    cutting there clipped words), then adds up to `quiet_pad` of silence, stopping at the next sound. When no pause
    comes before the cap, the limit or the take's edge, the sound running on is not the word (a held hum or babble
    Whisper leaves out; the take's own fade-out at its edge is no dip to cut at): the cut goes at the quietest
    window within 0.12 s of Whisper's edge, where cut()'s fade ends it."""
    n = len(levels)
    idx = lambda x: min(n - 1, max(0, int(x / win)))  # noqa: E731
    x, moved = t, 0.0
    while levels[idx(x)] >= floor_db:
        nxt = x + direction * win
        if moved >= sound_max or (nxt - limit) * direction >= 0 or not 0 <= nxt <= n * win:
            if levels[idx(x)] < -24.0:  # a decay still fading out at the stop point (e.g. the take's edge): keep it
                return max(0.0, min(n * win, x))
            near = [t + k * win for k in range(-6, 7) if 0 <= t + k * win <= n * win]
            return min(near, key=lambda p: (levels[idx(p)], abs(p - t)))
        x, moved = nxt, moved + win
    pad = 0.0
    while pad < quiet_pad and levels[idx(x)] < floor_db:
        nxt = x + direction * win
        if (nxt - limit) * direction >= -0.05 or not 0 <= nxt <= n * win:
            break
        x, pad = nxt, pad + win
    return max(0.0, min(n * win, x))


def check_clip(text: str, clip: Path) -> dict:
    """Gate for a finished narration clip. ok when:
    - recall >= 0.9 (better of the verbatim and plain decodes; the prompted decode sometimes stops early);
    - no extra words in the verbatim decode (repeats, stutters, fillers, babble: speech_qa.extra_words) and no
      cut-off fragment (a verbatim token ending in "-": "No expla -" is a clipped word the plain decode completes);
    - no voiced audio (>= 0.1 s, frames within 26 dB of the peak) apart from the words: a word covers its span
      +-0.1 s plus any sound running on from it without a pause (Whisper ends drawn-out words early, "expla|nation"),
      so what is left is sound after a pause Whisper leaves out entirely (e.g. a babble onset after the line);
    - no stretched word: one whose span holds more than 0.65 s + 0.08 s per letter of *voiced* audio (pauses don't
      count). Whisper folds a repeated phrase or gibberish into a neighbouring word's span ("back" holding 1.44 s
      of speech over "the bridge is out the- the bridge is out, we turn back"); real words in a full episode's
      narration stayed under +0.2 s of it."""
    from speech_qa import extra_words, score, words as norm
    verb, plain = word_times(clip), word_times(clip, plain=True)
    if verb is None or plain is None:
        raise RuntimeError(f"local ASR unavailable for {clip}")
    heard = " ".join(w for w, _, _ in verb)
    recall = max(score(text, heard)[0], score(text, " ".join(w for w, _, _ in plain))[0])
    extra = extra_words(text, heard) + [w for w, _, _ in verb if w.rstrip(".,!?").endswith("-") or w == "-"]
    # A clipped first/last word ("babies" -> "bait") costs one word of recall and passes 0.9; the edge words must
    # be heard (either decode, close spelling) for the clip to pass.
    import difflib
    want = norm(text)
    def edge_ok(k: int) -> bool:
        return any(difflib.SequenceMatcher(None, want[k], got).ratio() >= 0.8
                   for dec in (verb, plain) for got in [(norm(" ".join(w for w, _, _ in dec)) or [""])[k]])
    clipped = [want[k] for k in (0, -1) if want and not edge_ok(k)]
    levels = _levels(clip)
    # A cut through a word leaves sound at the cut, and Whisper papers over it ("then fifty" heard as "then fifty
    # thousand"). Measured on the take at the cut (the .cutv sidecar), since finish_clip's loudnorm and silence
    # trim change the clip's own end: the take must be near-silent (or done decaying) just inside the cut end.
    info = cut_info(clip)
    end_db = info.get("end_db")
    if end_db is not None and end_db > -30.0:
        clipped.append(f"<ends mid-sound {end_db:.0f} dB>")
    if info.get("aligned") is False:  # dip-search fallback: its cuts clipped "thousand" and "percent"
        clipped.append("<unaligned cut>")
    start_in_sound = bool(levels) and max(levels[0:2]) > -24.0  # H3 often speaks from frame 0: reported only
    voiced_mask = [lv >= -26.0 for lv in levels]
    sounding = [lv >= -38.0 for lv in levels]
    n = len(levels)
    covered = [False] * n
    stretched = []
    for w, a, b in verb:
        lo, hi = max(0, int((a - 0.1) / 0.02)), min(n - 1, int((b + 0.1) / 0.02))
        while lo > 0 and sounding[lo - 1]:
            lo -= 1
        while hi < n - 1 and sounding[hi + 1]:
            hi += 1
        for i in range(lo, hi + 1):
            covered[i] = True
        voiced = sum(1 for i in range(int(a / 0.02), min(n, int(b / 0.02))) if voiced_mask[i]) * 0.02
        if voiced > 0.65 + 0.08 * len("".join(norm(w))):
            stretched.append(f"{w}={voiced:.2f}s")
    uncovered = round(sum(1 for v, c in zip(voiced_mask, covered) if v and not c) * 0.02, 2)
    ok = recall >= 0.9 and not extra and not clipped and uncovered < 0.1 and not stretched
    return {"ok": ok, "recall": round(recall, 2), "extra": extra, "clipped": clipped, "uncovered_s": uncovered,
            "stretched": stretched, "starts_in_sound": start_in_sound, "heard": heard}


def split_take(take: Path, texts: list[str], min_recall: float = 0.75) -> list[tuple[float, float]]:
    """Find each scripted line of `texts` (in order) in one take; TTS and H3 takes pad or run lines together.
    Fast path: word timestamps (word_times / split_take_words). Fallback: cut points are envelope dips (dips()).
    For each line, the target is the recall of the line's words in the whole
    rest of the take (ASR); the line starts at the latest dip whose rest still reaches that target and ends at the
    earliest dip after it whose span does, so no word is dropped at either end. Both are binary searches (recall
    only grows as a span widens). A line whose rest of the take holds under `min_recall` of its words gets the
    whole remaining span, with a warning."""
    from speech_qa import score, transcribe
    points = dips(take)
    total = points[-1]
    timed = word_times(take)
    fast = split_take_words(take, texts, timed) if timed else None
    if fast:
        # Whisper's edges can't place the cut: it ends drawn-out words early ("expla|nation"), reports a final
        # "-st"/"-ts" as the next (babble) word, and completes a clipped "fifty" to "fifty thousand". Its words only
        # choose the pause; the line runs between the real pauses around its first and last words.
        levels = _levels(take)

        def anchor(a: float, b: float, outward: int) -> float:
            """A point inside the edge word: within 6 dB of the loudest window of Whisper's span [a, b], the one
            furthest out (latest for the last word, earliest for the first). Whisper's span can spill into the
            pause before the word ("Next season, … harder") or hold the end of the word before it ("for…
            months"), so neither its edges nor the plain peak are safe."""
            lo, hi = max(0, int(a / 0.02)), max(int(a / 0.02) + 1, min(len(levels), int(b / 0.02)))
            idx = [min(i, len(levels) - 1) for i in range(lo, hi)]
            peak = max(levels[i] for i in idx)
            near = [i for i in idx if levels[i] >= peak - 6.0]
            return (max(near) if outward > 0 else min(near)) * 0.02

        cuts = []
        for k, (fs, fe, ls, le) in enumerate(fast):
            a = _pause_cut(levels, anchor(fs, fe, -1), -1, fe - fs + 0.8)
            b = _pause_cut(levels, anchor(ls, le, +1), +1, le - ls + 1.0)
            a = _snap(levels, fs, -1, 0.0, 0.4, 0.12) if a is None else a
            b = _snap(levels, min(le, total), +1, total, 0.6, 0.25) if b is None else b
            if k + 1 < len(fast):
                b = min(b, fast[k + 1][0])
            cuts.append((a, min(b, total)))
        return cuts
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
            if target >= 0.99:
                # Dips can be seconds apart, so babble run together with the line survives the dip search (H3
                # takes often babble before/after it). Refine both ends on a 0.05 s grid; ASR must still hear every
                # word. Only with full recall: a partial target let the search cut real words ("my name is").
                a = _last_ok(a, b - 0.4, lambda s: recall(text, s, b) >= target)
                b = _first_ok(a + 0.4, b, lambda e: recall(text, a, e) >= target)
            spans.append((a, b))
            done = b
    # A cut can sit inside a soft word ending ("babies" -> "baby") or before a soft onset, and ASR still hears the
    # word whole. Pad by up to LEAD_S / TAIL_S, but only across quiet audio (babble right next to the line stays out),
    # and never past the neighbouring lines' own boundaries.
    quiet = _quiet_mask(take)
    prev_ends = [0.0] + [b for _, b in spans[:-1]]
    next_starts = [a for a, _ in spans[1:]] + [total]
    return [(_pad(quiet, a, -LEAD_S, pe), _pad(quiet, b, TAIL_S, ns)) for (a, b), pe, ns in
            zip(spans, prev_ends, next_starts)]


def _last_ok(lo: float, hi: float, ok, step: float = 0.05) -> float:
    """Latest t in [lo, hi] (step grid) with ok(t), assuming ok holds for a prefix; lo if none past it."""
    while hi - lo > step:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _first_ok(lo: float, hi: float, ok, step: float = 0.05) -> float:
    """Earliest t in [lo, hi] with ok(t), assuming ok holds for a suffix; hi if none before it."""
    while hi - lo > step:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def _levels(take: Path, win: float = 0.02) -> list[float]:
    """Per-20 ms window: level in dB relative to the take's loudest window."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(take), "-ac", "1", "-ar", "16000", "-f", "s16le",
                          "-"], capture_output=True, check=True).stdout
    import array
    s = array.array("h", raw)
    n = int(16000 * win)
    rms = [math.sqrt(sum(x * x for x in s[i:i + n]) / max(1, len(s[i:i + n]))) + 1e-9 for i in range(0, len(s), n)]
    peak = max(rms)
    return [20 * math.log10(r / peak) for r in rms]


def _quiet_mask(take: Path, win: float = 0.02, floor_db: float = -38.0) -> list[bool]:
    """Per-20 ms window: True where the take is below `floor_db` relative to its peak window (a pause)."""
    return [lv < floor_db for lv in _levels(take, win)]


def _pad(quiet: list[bool], t: float, pad: float, limit: float, win: float = 0.02) -> float:
    """Move t by up to `pad` seconds (negative = earlier) while the audio stays quiet, never past `limit`."""
    step = win if pad > 0 else -win
    moved = 0.0
    while abs(moved) < abs(pad):
        i = int((t + moved + step / 2) / win)
        nxt = t + moved + step
        if not (0 <= i < len(quiet)) or not quiet[i] or (pad > 0 and nxt > limit) or (pad < 0 and nxt < limit):
            break
        moved += step
    # a small fixed margin keeps soft onsets/endings even when they are not below the floor
    edge = t + moved + (0.04 if pad > 0 else -0.04)
    return min(limit, edge) if pad > 0 else max(limit, edge)


LEAD_S, TAIL_S = 0.15, 0.6


def instruments_stem(video: Path) -> Path:
    stem = video.with_name(video.stem + ".instruments.flac")
    if stem.exists() and stem.stat().st_mtime >= video.stat().st_mtime:
        return stem
    wav = video.with_name(video.stem + ".bed_src.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
                    str(wav)], check=True)
    g = Graph()
    a = g.add("LoadAudio", audio=upload_image(wav, base=AUX_URL))
    m = g.add("MelBandRoFormerModelLoader", model_name=SEPARATOR)
    s = g.add("MelBandRoFormerSampler", model=[m, 0], audio=[a, 0])
    g.add("SaveAudio", audio=[s, 1], filename_prefix=f"h3film/stems/{video.stem}_instruments")
    fetch_audio(run_graph(g, f"stem {video.stem}"), stem)
    wav.unlink()
    return stem


def plan(film: dict, root: Path) -> tuple[list[dict], list[dict]]:
    """Returns (clips with start/end, narration events with absolute start). A shot's "out" (seconds) ends it
    early in the cut, e.g. to drop words H3 invented after the scripted line; "in" (seconds) starts it later,
    e.g. to drop words invented before the line. Both are render times; "at" stays relative to the cut shot."""
    clips, events, t = [], [], 0.0
    for shot in film["shots"]:
        video = root / "renders" / f"{shot['id']}.mp4"
        if not video.exists():
            raise SystemExit(f"missing render {video} (run h3_render.py first)")
        d = duration(video)
        if "out" in shot:
            d = min(d, round(shot["out"] * 24) / 24)
        head = round(shot.get("in", 0.0) * 24) / 24
        d -= head
        clips.append({"id": shot["id"], "video": video, "start": t, "end": t + d, "in": head,
                      "bed": shot.get("bed", "full")})
        for line in shot.get("narration", []):
            events.append({"shot": shot["id"], "start": t + line["at"], "text": line["text"], "seed": line.get("seed"),
                           "file": line.get("file"), "voice": line.get("voice"), "delivery": line.get("delivery"),
                           "tempo": float(line.get("tempo", film.get("narration_tempo", 1.0)))})
        t += d
    return clips, events


def cut_speech(clips: list[dict]) -> list[tuple[float, float, str]]:
    """On-screen speech spans (absolute cut time, shot id) from each full-bed shot's vocal stem, shifted by its "in"
    and clipped to its trimmed length: speech outside [in, out) never reaches the cut."""
    spans = []
    for c in clips:
        if c["bed"] != "full":
            continue
        length = c["end"] - c["start"]
        for s, t in dialogue_spans(c["video"]):
            s, t = s - c.get("in", 0.0), t - c.get("in", 0.0)
            if t > 0 and s < length:
                spans.append((c["start"] + max(0.0, s), c["start"] + min(t, length), c["id"]))
    return spans


def report(film: dict, root: Path, clips: list[dict], events: list[dict]) -> None:
    """Print the narration timeline with overlap / dialogue warnings. Events with an "end" are checked."""
    ends = {c["id"]: c["end"] for c in clips}
    speech = cut_speech(clips)
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
        clip = (clip_path(film, root, e["text"], e["seed"], e["file"], e["voice"], e["tempo"],
                          delivery=e["delivery"]) if a.plan
                else tts(film, root, e["text"], e["seed"], e["file"], e["voice"], e["tempo"], e["delivery"]))
        if clip.exists():
            e["end"] = e["start"] + duration(clip)
    report(film, root, clips, events)


if __name__ == "__main__":
    sys.exit(main())
