"""Render a shot list with MiniMax H3 on a local ComfyUI (the cut is made in Resolve by tools/resolve_edit.py).

    python tools/h3_render.py film/<name>/shots.json                 # render every shot not yet rendered
    python tools/h3_render.py film/<name>/shots.json --only s01,s03  # subset
    python tools/h3_render.py film/<name>/shots.json --force         # ignore cache
    python tools/h3_render.py film/<name>/shots.json --dry-run       # write API graphs only

Shot-list JSON (paths relative to the shot-list file):
{
  "title": "My Film", "style": "<global look, prepended to every prompt>",
  "audio": "<global soundscape note, appended>",
  "width": 1344, "height": 768, "steps": 20, "seed": 1000, "turbo": false,   # turbo: step-distilled LoRA
  "subjects": {"hero": {"desc": "<age, build, hair, clothes>", "ref": "refs/hero.png"},        # "ref": a path or a list of
               "guest": {"desc": "...", "ref": ["refs/g1.png", "refs/g2.png"]}},    #  paths (several real photos)
  "voices": {"hero": {"desc": "Hero", "audio": "voices/hero.wav"}},
  "styles": {"human": "<alternative look>"},
  "shots": [
    {"id": "s01", "mode": "fl",  "duration": 5, "prompt": "...", "first_frame": "keys/s01.png", "last_frame": null},
    {"id": "s02", "mode": "fl",  "duration": 5, "prompt": "...", "first_frame": "@prev", "style": "human"},
    {"id": "s03", "mode": "ref", "duration": 8, "prompt": "...", "subjects": ["hero"], "voices": ["hero"],
     "first_frame": "keys/s03a.png", "guides": [{"t": 4.5, "image": "keys/s03b.png"}]},
    {"id": "s04", "mode": "ref", "duration": 8, "prompt": "...", "subjects": ["guest"], "framing": "keys/s04.png"}
  ]
}
Modes:
  fl  — fl2va model (MiniMaxH3ImageToVideo). No frames = text-to-video; first_frame = image-to-video;
        first+last = first/last-frame interpolation. "@prev" = last frame of the previous shot's render.
  ref — ref2va model (MiniMaxH3ReferenceToVideo). Subject reference images keep characters on-model,
        reference voices keep a character's voice (ref_audios; say "in the voice of <Audio N>" in the
        audio line). first_frame/last_frame become guides at t=0 / t=duration; extra guides pin keyframes
        at timestamps (MiniMaxH3AddGuide). "framing" is a soft alternative to frame guides: the still goes in
        as one more reference image, described as a framing/layout reference only, so the camera, set and
        placement follow it while every face and likeness comes from the subjects' own pictures. The prompt
        gets a subject_definitions block mapping <Subject N>/<Picture N>/<Audio N> tags, as in the Comfy
        templates. At most 9 reference images per shot.
  clip — copy an existing video ("source", relative to the shot list) into renders/<id>.mp4, scaled/padded to
        the film size at 24 fps (no GPU work, no "duration"; keeps the source's audio).
  still — render a still image ("source", relative to the shot list) as renders/<id>.mp4 of "duration" seconds,
        cover-fitted to the film size at 24 fps with a silent audio track (no GPU work, no "prompt"). Static,
        or a Ken Burns move with "kenburns": {"from": [cx, cy, zoom], "to": [cx, cy, zoom]} — the centre of the
        visible window as fractions of the cover-fitted frame (0-1) and zoom >= 1, interpolated linearly over the
        shot (Resolve 21.1's API has no keyframe calls, so the move is rendered here, supersampled 2x).
Renders land in <shotlist dir>/renders/<id>.mp4 with a <id>.json sidecar; a shot re-renders only when its
effective spec (prompt, frames' content hashes, params) changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zlib
from pathlib import Path

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
FPS = 24
MODELS = {
    "fl": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "ref": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}
# Step-distilled LoRA presets. "turbo": true picks the mode's default; "turbo": "<preset>" picks one explicitly
# (film-level or per shot). "shift" = (video, audio) sigma shift via MiniMaxH3SigmaShift; None = model default.
TURBO_PRESETS = {
    "fl8": {"mode": "fl", "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "steps": 8},
    "fl8_768": {"mode": "fl", "lora": "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "steps": 8},
    "fl4": {"mode": "fl", "lora": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", "steps": 4,
            "shift": (6.0, 3.0)},
    "ref4": {"mode": "ref", "lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "steps": 4},
    "ref8": {"mode": "ref", "lora": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "steps": 8},
}
TURBO_DEFAULT = {"fl": "fl8", "ref": "ref4"}
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def frames_for(seconds: float) -> int:
    """H3 wants 17k+5 frames at 24 fps (same formula as the Comfy templates)."""
    n = max(5, round(seconds * FPS))
    return n + (5 - n % 17) % 17


# ---- ComfyUI HTTP ------------------------------------------------------------------
def http_json(path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(COMFY + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:4000]}") from None


def upload_image(path: Path) -> str:
    """Upload to ComfyUI's input/ under a content-addressed name; returns the LoadImage name."""
    data = path.read_bytes()
    name = f"h3film_{hashlib.sha1(data).hexdigest()[:12]}{path.suffix.lower()}"
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{boundary}--\r\n'
    ).encode()
    req = urllib.request.Request(COMFY + "/upload/image", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["name"]


# ---- graph -------------------------------------------------------------------------
class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add(self, cls: str, **inputs) -> str:
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": cls, "inputs": inputs}
        return nid


def render_settings(shot: dict, film: dict) -> dict:
    """Effective model settings for a shot (resolves film/shot turbo presets and step overrides)."""
    mode = shot.get("mode", "fl")
    turbo = shot.get("turbo", film.get("turbo", False))
    if isinstance(turbo, dict):  # film-level per-mode map, e.g. {"fl": "fl4", "ref": "ref8"}
        turbo = turbo.get(mode, False)
    lora, shift, steps = None, None, film.get("steps", 20)
    if turbo:
        preset = TURBO_PRESETS[TURBO_DEFAULT[mode] if turbo is True else turbo]
        if preset["mode"] != mode:
            raise ValueError(f"{shot['id']}: turbo preset {turbo!r} is for {preset['mode']!r} shots, not {mode!r}")
        lora, shift, steps = preset["lora"], preset.get("shift"), preset["steps"]
    settings = {"mode": mode, "lora": lora, "shift": shift, "steps": shot.get("steps", steps),
                "width": shot.get("width", film.get("width", 1344)),
                "height": shot.get("height", film.get("height", 768))}
    if shot.get("loras"):  # extra style/motion LoRAs, e.g. [["minimax_h3_wushu_action_v5_fl2va.safetensors", 0.5]]
        settings["loras"] = [[name, float(strength)] for name, strength in shot["loras"]]
    return settings


def build_graph(shot: dict, film: dict, images: dict[str, str], prefix: str) -> dict:
    """images maps local path -> uploaded LoadImage name."""
    g = Graph()
    settings = render_settings(shot, film)
    mode = settings["mode"]
    w, h, length = settings["width"], settings["height"], frames_for(shot["duration"])
    unet = g.add("UNETLoader", unet_name=MODELS[mode], weight_dtype="default")
    if settings["lora"]:
        unet = g.add("LoraLoaderModelOnly", model=[unet, 0], lora_name=settings["lora"], strength_model=1.0)
    for name, strength in settings.get("loras", []):
        unet = g.add("LoraLoaderModelOnly", model=[unet, 0], lora_name=name, strength_model=strength)
    if settings["shift"]:
        unet = g.add("MiniMaxH3SigmaShift", model=[unet, 0], shift_video=settings["shift"][0],
                     shift_audio=settings["shift"][1])
    steps = settings["steps"]
    clip = g.add("CLIPLoader", clip_name=TEXT_ENCODER, type="minimax", device="default")
    vvae = g.add("VAELoader", vae_name=VIDEO_VAE)
    avae = g.add("VAELoader", vae_name=AUDIO_VAE)
    load = lambda p: [g.add("LoadImage", image=images[p]), 0]  # noqa: E731
    load_audio = lambda p: [g.add("LoadAudio", audio=images[p]), 0]  # noqa: E731

    if mode == "fl":
        kw = {}
        if shot.get("first_frame"):
            kw["first_frame"] = load(shot["first_frame"])
        if shot.get("last_frame"):
            kw["last_frame"] = load(shot["last_frame"])
        cond = g.add("MiniMaxH3ImageToVideo", clip=[clip, 0], vae=[vvae, 0], prompt=shot["_prompt"],
                     width=w, height=h, length=length, **kw)
        positive, latent = [cond, 0], [cond, 1]
    elif mode == "ref":
        kw = {f"ref_images.ref_image_{i}": load(p) for i, p in enumerate(shot["_refs"])}
        kw |= {f"ref_audios.ref_audio_{i}": load_audio(p) for i, p in enumerate(shot["_voices"])}
        cond = g.add("MiniMaxH3ReferenceToVideo", clip=[clip, 0], vae=[vvae, 0], audio_vae=[avae, 0],
                     prompt=shot["_prompt"], width=w, height=h, length=length, ref_image_size="match", **kw)
        positive, latent = [cond, 0], [cond, 1]
        for guide in shot.get("guides", []):
            idx = min(round(guide["t"] * FPS), length - 1)
            gid = g.add("MiniMaxH3AddGuide", positive=positive, latent=latent, frame_idx=idx,
                        vae=[vvae, 0], audio_vae=[avae, 0], image=load(guide["image"]))
            positive = [gid, 0]
    else:
        raise ValueError(f"{shot['id']}: unknown mode {mode!r}")

    noise = g.add("RandomNoise", noise_seed=shot["_seed"])
    sampler = g.add("KSamplerSelect", sampler_name="res_multistep")
    sigmas = g.add("BasicScheduler", model=[unet, 0], scheduler="simple", steps=steps, denoise=1.0)
    guider = g.add("BasicGuider", model=[unet, 0], conditioning=positive)
    out = g.add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0],
                sigmas=[sigmas, 0], latent_image=latent)
    frames = g.add("VAEDecode", samples=[out, 0], vae=[vvae, 0])
    audio = g.add("VAEDecodeAudio", samples=[out, 0], vae=[avae, 0])
    video = g.add("CreateVideo", images=[frames, 0], audio=[audio, 0], fps=float(FPS))
    g.add("SaveVideo", video=[video, 0], filename_prefix=prefix, format="mp4", **{"format.codec": "h264"})
    return g.nodes


# ---- prompts -----------------------------------------------------------------------
def compose_prompt(shot: dict, film: dict) -> tuple[str, list[str]]:
    """Returns (prompt, ordered reference image paths: each subject's pictures, then the framing still)."""
    parts: list[str] = []
    refs: list[str] = []
    if shot.get("mode") == "ref":
        defs = ["subject_definitions:"]
        for n, key in enumerate(shot.get("subjects", []), start=1):
            subj = film["subjects"][key]
            own = subj["ref"] if isinstance(subj["ref"], list) else [subj["ref"]]
            tags = [f"<Picture {len(refs) + k}>" for k in range(1, len(own) + 1)]
            refs += own
            shown = tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]
            defs.append(f"<Subject {n}> is {subj['desc']}, shown in {shown}.")
        if shot.get("framing"):
            refs.append(shot["framing"])
            f = f"<Picture {len(refs)}>"
            defs.append(f"{f} is a framing reference only: match its camera angle, composition, set, lighting and "
                        f"where everyone and everything sits, but take every subject's face, likeness, body and "
                        f"details from that subject's own pictures, never from {f}.")
        if len(refs) > 9:
            raise ValueError(f"{shot['id']}: {len(refs)} reference images; H3 takes at most 9")
        for m, guide in enumerate(shot.get("guides", []), start=len(refs) + 1):
            defs.append(f"<Picture {m}> is the keyframe at {guide['t']:.1f}s: {guide.get('desc', 'match it exactly')}.")
        for n, key in enumerate(shot.get("voices", []), start=1):
            voice = film["voices"][key]
            defs.append(f"<Audio {n}> is the voice of {voice['desc']}. Every line {voice['desc']} speaks uses exactly "
                        f"this voice (same timbre, pitch, age and accent) while saying the new words from the script; "
                        f"do not play <Audio {n}> itself.")
        if len(defs) > 1:
            parts.append("\n".join(defs))
    style = film.get("styles", {}).get(shot["style"]) if shot.get("style") else film.get("style")
    if shot.get("style") and style is None:
        raise ValueError(f"{shot['id']}: unknown style {shot['style']!r}")
    if style:
        parts.append(style)
    parts.append(shot["prompt"])
    audio = shot.get("audio") or film.get("audio")
    if audio:
        parts.append(f"Audio: {audio}")
    if shot.get("trigger"):  # LoRA trigger words must lead the prompt (e.g. "wushu_action,")
        parts.insert(0, shot["trigger"])
    return "\n\n".join(parts), refs


# ---- run -----------------------------------------------------------------------------
def last_frame(video: Path) -> Path:
    png = video.with_name(video.stem + "_last.png")
    if not png.exists() or png.stat().st_mtime < video.stat().st_mtime:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.1", "-i", str(video),
                        "-frames:v", "1", "-update", "1", str(png)], check=True)
    return png


def kenburns_frames(src: Path, w: int, h: int, kb: dict, n: int, sid: str):
    """Yields `n` rgb24 frames (w x h) of `src` cover-fitted to the frame, panning/zooming linearly from kb["from"]
    to kb["to"] ([cx, cy, zoom]; see the module docstring). Centres are clamped so the window stays inside the
    picture. Rendered at 2x and reduced, so slow sub-pixel moves stay smooth."""
    from PIL import Image

    def clamp(cx: float, cy: float, z: float) -> tuple[float, float, float]:
        z = max(1.0, z)
        m = 0.5 / z
        return min(max(cx, m), 1 - m), min(max(cy, m), 1 - m), z

    ends = [[float(v) for v in kb[k]] for k in ("from", "to")]
    fixed = [list(clamp(*e)) for e in ends]
    if fixed != ends:
        print(f"{sid}: kenburns centre clamped")
    img = Image.open(src).convert("RGB")
    big_w, big_h = 2 * w, 2 * h
    s = max(big_w / img.width, big_h / img.height)
    img = img.resize((max(big_w, round(img.width * s)), max(big_h, round(img.height * s))), Image.Resampling.LANCZOS)
    x0, y0 = (img.width - big_w) // 2, (img.height - big_h) // 2
    img = img.crop((x0, y0, x0 + big_w, y0 + big_h))
    (ax, ay, az), (bx, by, bz) = fixed
    for t in range(n):
        f = t / (n - 1) if n > 1 else 0.0
        cx, cy, z = clamp(ax + (bx - ax) * f, ay + (by - ay) * f, az + (bz - az) * f)
        box = ((cx - 0.5 / z) * big_w, (cy - 0.5 / z) * big_h, (cx + 0.5 / z) * big_w, (cy + 0.5 / z) * big_h)
        yield img.transform((big_w, big_h), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC).reduce(2).tobytes()


def wait_for(prompt_id: str, label: str) -> dict:
    t0 = time.time()
    while True:
        hist = http_json(f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"{label}: execution error: {json.dumps(msgs)[:3000]}")
            if status.get("completed"):
                return entry
        print(f"\r  {label}: rendering… {time.time() - t0:5.0f}s", end="", flush=True)
        time.sleep(5)


def fetch_video(entry: dict, dest: Path) -> None:
    for node_out in entry["outputs"].values():
        for item in node_out.get("images", []) + node_out.get("videos", []) + node_out.get("gifs", []):
            if item.get("filename", "").endswith((".mp4", ".webm", ".mkv")):
                q = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
                with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=300) as r:
                    dest.write_bytes(r.read())
                return
    raise RuntimeError(f"no video in outputs: {json.dumps(entry['outputs'])[:1000]}")


def render(shotlist: Path, only: set[str] | None, force: bool, dry_run: bool, restamp: bool = False) -> None:
    film = json.loads(shotlist.read_text(encoding="utf-8"))
    root = shotlist.parent
    renders = root / "renders"
    renders.mkdir(exist_ok=True)
    slug = shotlist.parent.name
    prev_video: Path | None = None
    for i, shot in enumerate(film["shots"]):
        sid = shot["id"]
        dest = renders / f"{sid}.mp4"
        prev, prev_video = prev_video, dest
        if only and sid not in only:
            continue
        # Default seed keys off the shot id, so inserting or reordering shots doesn't re-roll the others.
        shot["_seed"] = shot.get("seed", film.get("seed", 0) + zlib.crc32(sid.encode()) % 100_000)
        if shot.get("mode") == "clip":
            src = root / shot["source"]
            if not src.exists():
                raise SystemExit(f"{sid}: missing source {src}")
            settings = render_settings(shot, film)
            w, h = settings["width"], settings["height"]
            digest = hashlib.sha1(json.dumps({"mode": "clip", "source": hashlib.sha1(src.read_bytes()).hexdigest(),
                                              "w": w, "h": h}, sort_keys=True).encode()).hexdigest()
            sidecar = dest.with_suffix(".json")
            if not force and dest.exists() and sidecar.exists() and json.loads(sidecar.read_text())["digest"] == digest:
                print(f"= {sid}: up to date")
                continue
            if restamp and dest.exists() and sidecar.exists():
                meta = json.loads(sidecar.read_text())
                sidecar.write_text(json.dumps(meta | {"digest": digest}, indent=2), encoding="utf-8")
                print(f"# {sid}: restamped existing render as current")
                continue
            if dry_run:
                print(f"~ {sid}: clip (dry run)")
                continue
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf",
                            f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                            f"fps={FPS},setsar=1", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-ar", "48000", "-ac", "2", str(dest)], check=True)
            sidecar.write_text(json.dumps({"digest": digest, "source": shot["source"]}, indent=2), encoding="utf-8")
            print(f"+ {sid}: clip {shot['source']}")
            continue
        if shot.get("mode") == "still":
            src = root / shot["source"]
            if not src.exists():
                raise SystemExit(f"{sid}: missing source {src}")
            settings = render_settings(shot, film)
            w, h = settings["width"], settings["height"]
            digest = hashlib.sha1(json.dumps({"mode": "still", "source": hashlib.sha1(src.read_bytes()).hexdigest(),
                                              "w": w, "h": h, "duration": shot["duration"],
                                              "kenburns": shot.get("kenburns")}, sort_keys=True).encode()).hexdigest()
            sidecar = dest.with_suffix(".json")
            if not force and dest.exists() and sidecar.exists() and json.loads(sidecar.read_text())["digest"] == digest:
                print(f"= {sid}: up to date")
                continue
            if restamp and dest.exists() and sidecar.exists():
                meta = json.loads(sidecar.read_text())
                sidecar.write_text(json.dumps(meta | {"digest": digest}, indent=2), encoding="utf-8")
                print(f"# {sid}: restamped existing render as current")
                continue
            if dry_run:
                print(f"~ {sid}: still (dry run)")
                continue
            enc = ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(dest)]
            if "kenburns" in shot:
                n = round(shot["duration"] * FPS)
                proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i",
                                         "anullsrc=r=48000:cl=stereo", "-t", str(shot["duration"]), *enc],
                                        stdin=subprocess.PIPE)
                for frame in kenburns_frames(src, w, h, shot["kenburns"], n, sid):
                    proc.stdin.write(frame)
                proc.stdin.close()
                if proc.wait():
                    raise SystemExit(f"{sid}: ffmpeg failed")
            else:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(src), "-f", "lavfi",
                                "-i", "anullsrc=r=48000:cl=stereo", "-t", str(shot["duration"]), "-vf",
                                f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},setsar=1",
                                *enc], check=True)
            sidecar.write_text(json.dumps({"digest": digest, "source": shot["source"]}, indent=2), encoding="utf-8")
            print(f"+ {sid}: still {shot['source']}")
            continue
        # resolve image paths (+ "@prev" chaining to the previous shot's last frame)
        for key in ("first_frame", "last_frame"):
            if shot.get(key) == "@prev":
                if prev is None or not prev.exists():
                    if not dry_run:
                        raise SystemExit(f"{sid}: {key}=@prev but the previous shot has no render yet")
                    shot[key] = None
                    continue
                shot[key] = str(last_frame(prev))
            elif shot.get(key):
                shot[key] = str(root / shot[key])
        for guide in shot.get("guides", []):
            guide["image"] = str(root / guide["image"])
        if shot.get("mode") == "ref":
            # ref2va has no first/last-frame inputs: pin them as timeline guides instead.
            if shot.get("first_frame"):
                shot.setdefault("guides", []).insert(0, {"t": 0, "image": shot.pop("first_frame"),
                                                         "desc": "the opening frame; start exactly from it"})
            if shot.get("last_frame"):
                shot.setdefault("guides", []).append({"t": shot["duration"], "image": shot.pop("last_frame"),
                                                      "desc": "the closing frame; end exactly on it"})
        shot["_prompt"], shot["_refs"] = compose_prompt(shot, film)
        shot["_refs"] = [str(root / p) for p in shot["_refs"]]
        shot["_voices"] = [str(root / film["voices"][k]["audio"]) for k in shot.get("voices", [])]
        local = [Path(p) for p in (shot.get("first_frame"), shot.get("last_frame"), *shot["_refs"],
                                   *(gd["image"] for gd in shot.get("guides", [])), *shot["_voices"]) if p]
        for p in local:
            if not p.exists():
                raise SystemExit(f"{sid}: missing input {p}")
        # Cache key = everything that changes the rendered pixels/audio. Narration, bed, the edit-only
        # titles/zoom/hold/out/pillarbox and the literal seed/turbo/steps keys are excluded; their effective
        # values are covered by "_seed" and "_render".
        spec = {k: v for k, v in shot.items() if k not in (
            "seed", "turbo", "steps", "width", "height", "narration", "bed", "titles", "zoom", "hold", "out",
            "pillarbox", "first_frame", "last_frame", "guides", "_refs", "_voices")}
        spec["_images"] = [hashlib.sha1(p.read_bytes()).hexdigest() for p in local]
        spec["_guides"] = [g["t"] for g in shot.get("guides", [])]
        spec["_render"] = render_settings(shot, film)
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        sidecar = dest.with_suffix(".json")
        if not force and dest.exists() and sidecar.exists() and json.loads(sidecar.read_text())["digest"] == digest:
            print(f"= {sid}: up to date")
            continue
        if restamp and dest.exists() and sidecar.exists():
            meta = json.loads(sidecar.read_text())
            sidecar.write_text(json.dumps(meta | {"digest": digest}, indent=2), encoding="utf-8")
            print(f"# {sid}: restamped existing render as current")
            continue
        if shot.get("mode") == "black":
            # Picture-free beat (e.g. a character losing sight): black frames + a faint room-tone bed.
            if not dry_run:
                w, h = render_settings(shot, film)["width"], render_settings(shot, film)["height"]
                d = frames_for(shot["duration"]) / FPS
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                                f"color=c=black:s={w}x{h}:r={FPS}:d={d:.3f}", "-f", "lavfi", "-i",
                                f"anoisesrc=color=brown:amplitude=0.01:r=48000:d={d:.3f}", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-shortest", str(dest)], check=True)
                sidecar.write_text(json.dumps({"digest": digest, "seed": shot["_seed"], "prompt": "black"}, indent=2),
                                   encoding="utf-8")
            print(f"+ {sid}: black {shot['duration']}s")
            continue
        uploaded = {str(p): (upload_image(p) if not dry_run else p.name) for p in local}
        graph = build_graph(shot, film, uploaded, prefix=f"h3film/{slug}/{sid}")
        (renders / f"{sid}.api.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
        if dry_run:
            print(f"~ {sid}: graph written ({shot.get('mode', 'fl')}, {frames_for(shot['duration'])} frames)")
            continue
        t0 = time.time()
        resp = http_json("/prompt", {"prompt": graph, "client_id": "h3_render"})
        if resp.get("node_errors"):
            raise RuntimeError(f"{sid}: {json.dumps(resp['node_errors'])[:3000]}")
        entry = wait_for(resp["prompt_id"], sid)
        fetch_video(entry, dest)
        sidecar.write_text(json.dumps({"digest": digest, "prompt_id": resp["prompt_id"], "seed": shot["_seed"],
                                       "seconds": round(time.time() - t0), "prompt": shot["_prompt"]}, indent=2),
                           encoding="utf-8")
        print(f"\r+ {sid}: {dest} ({time.time() - t0:.0f}s)          ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--only", help="comma-separated shot ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restamp", action="store_true",
                    help="accept existing renders of the selected shots as current (after a cache-key-only change)")
    a = ap.parse_args()
    render(a.shotlist, set(a.only.split(",")) if a.only else None, a.force, a.dry_run, a.restamp)


if __name__ == "__main__":
    sys.exit(main())
