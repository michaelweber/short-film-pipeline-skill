"""Render a shot list with MiniMax H3 on a local ComfyUI (the cut is made in Resolve by tools/resolve_edit.py).

    python tools/h3_render.py film/<name>/shots.json                 # render every shot not yet rendered
    python tools/h3_render.py film/<name>/shots.json --only s01,s03  # subset
    python tools/h3_render.py film/<name>/shots.json --force         # ignore cache
    python tools/h3_render.py film/<name>/shots.json --dry-run       # write API graphs only

Shot-list JSON (paths relative to the shot-list file):
{
  "title": "My Short", "style": "<global look, prepended to every prompt>",
  "audio": "<global soundscape note, appended>",
  "width": 1344, "height": 768, "steps": 20, "seed": 1000, "turbo": false,   # turbo: step-distilled LoRA
  "subjects": {"kid": {"desc": "10-year-old boy ...", "ref": "refs/kid.png"},        # "ref": a path or a list of
               "host": {"desc": "...", "ref": ["refs/h1.png", "refs/h2.png"]}},    #  paths (several real photos)
  "voices": {"kid": {"desc": "the boy", "audio": "voices/kid.wav"}},
  "styles": {"human": "<alternative look>"},
  "shots": [
    {"id": "s01", "mode": "fl",  "duration": 5, "prompt": "...", "first_frame": "keys/s01.png", "last_frame": null},
    {"id": "s02", "mode": "fl",  "duration": 5, "prompt": "...", "first_frame": "@prev", "style": "human"},
    {"id": "s03", "mode": "ref", "duration": 8, "prompt": "...", "subjects": ["kid"], "voices": ["kid"],
     "first_frame": "keys/s03a.png", "guides": [{"t": 4.5, "image": "keys/s03b.png"}]},
    {"id": "s04", "mode": "ref", "duration": 8, "prompt": "...", "subjects": ["host"], "framing": "keys/s04.png"}
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
        "control": {"video": "blocking/p08_depth.mp4", "strength": 1.0, "start": 0.0, "end": 1.0} adds the
        MiniMax H3 Fun ControlNet Union (models/model_patches, CONTROL_PATCH): the depth/canny/pose video steers
        every frame (greybox.py writes <id>_depth.mp4 from the blocking). Works in both fl and ref mode.
        "ref_video": {"video": "blocking/p08_anim.mp4", "desc": "..."} (ref mode, "prompt_format": "h3_ref") passes
        a blocking animatic as <Video 1>, a camera/motion storyboard ("@anim" in the prompt). Such a shot is
        non-turbo unless it sets "turbo" itself (the film's turbo is not inherited: turbo makes H3 ignore the video),
        and first_frame/last_frame are not turned into guides: pins override the video, so a ref-video shot gets
        guides only from its own explicit "guides" list (blocking_ref2vid research: IoU 0.917 non-turbo unpinned).
  clip — copy an existing video ("source", relative to the shot list) into renders/<id>.mp4, scaled/padded to
        the film size at 24 fps (no GPU work, no "duration"; keeps the source's audio).
  still — render a still image ("source", relative to the shot list) as renders/<id>.mp4 of "duration" seconds,
        cover-fitted to the film size at 24 fps with a silent audio track (no GPU work, no "prompt"). Static,
        or a Ken Burns move with "kenburns": {"from": [cx, cy, zoom], "to": [cx, cy, zoom]} — the centre of the
        visible window as fractions of the cover-fitted frame (0-1) and zoom >= 1, interpolated linearly over the
        shot (Resolve 21.1's API has no keyframe calls, so the move is rendered here, supersampled 2x).
Renders land in <shotlist dir>/renders/<id>.mp4 with a <id>.json sidecar; a shot re-renders only when its
effective spec (prompt, frames' content hashes, params) changes.
Tiers ("tier": "draft"|"final" on the film or a shot, default final; --tier overrides both):
  final — the settings as written (unchanged behaviour). A final with a control video and a turbo LoRA warns: the
          turbo+control contour-rim artifact (.claude/skills/short-film-pipeline/prompting.md).
  draft — a settings preview: short side scaled to 512 (aspect kept, snapped to 32), always turbo (the shot's own
          preset, else fl8 / ref4, ref8 with a control video), "draft_seeds" seeds (default 3: seed, +1, +2), written
          to renders/_draft/<id>_s<seed>.mp4 so drafts never touch finals. clip/still/black shots are skipped.
          Ref-video shots draft NON-turbo at short side 384 (672x384), 20 steps (~80 s per take on the h3 card).
  Turbo on a ref-video shot warns in either tier.
"audio_lock": true feeds the shot's "guide_audio" file ("audio_lock": "<path>" names it directly) into
MiniMaxH3AddGuide's audio input at frame 0, so the dialogue timing is the same in draft and final.
ComfyUI instances: H3 renders go to H3_URL (env H3_COMFY_URL, default :8188, the h3 card); stills/audio/ASR tools
use AUX_URL (env AUX_COMFY_URL, default :8189, the aux card). Outputs are fetched from the instance that ran the job.
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

H3_URL = os.environ.get("H3_COMFY_URL", "http://127.0.0.1:8188")  # H3 video only (h3 card)
AUX_URL = os.environ.get("AUX_COMFY_URL", "http://127.0.0.1:8189")  # stills, depth, TTS/ASR/CLAP (aux card)
COMFY = H3_URL  # old name, kept for importers
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
CONTROL_PATCH = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
TIERS = ("draft", "final")
DRAFT_SHORT_SIDE = 512
DRAFT_SHORT_SIDE_REF_VIDEO = 384  # ref-video drafts: non-turbo, 672x384 (blocking_ref2vid research, runs A/C)
DRAFT_TURBO = {"fl": "fl8", "ref": "ref4"}  # ref8 when the shot has a control video (throughput report §4.6)
SIZE_MULTIPLE = 32  # width/height step of MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo


def frames_for(seconds: float) -> int:
    """H3 wants 17k+5 frames at 24 fps (same formula as the Comfy templates)."""
    n = max(5, round(seconds * FPS))
    return n + (5 - n % 17) % 17


# ---- ComfyUI HTTP ------------------------------------------------------------------
def http_json(path: str, payload: dict | None = None, base: str = H3_URL) -> dict:
    req = urllib.request.Request(base + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:4000]}") from None


def upload_image(path: Path, base: str = H3_URL) -> str:
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
    req = urllib.request.Request(base + "/upload/image", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["name"]


def view(item: dict, base: str = H3_URL, timeout: int = 120) -> bytes:
    """An output file of a finished job, via /view on the instance that ran it (each instance has its own output dir)."""
    q = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
    with urllib.request.urlopen(f"{base}/view?{q}", timeout=timeout) as r:
        return r.read()


# ---- graph -------------------------------------------------------------------------
class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add(self, cls: str, **inputs) -> str:
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": cls, "inputs": inputs}
        return nid


def draft_size(w: int, h: int, short_side: int = DRAFT_SHORT_SIDE) -> tuple[int, int]:
    """(w, h) scaled so the short side is `short_side`, aspect kept, snapped to the model's size step."""
    scale = short_side / min(w, h)
    return tuple(max(SIZE_MULTIPLE, round(v * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE) for v in (w, h))


def has_ref_video(shot: dict) -> bool:
    return shot.get("mode", "fl") == "ref" and bool(shot.get("ref_video"))


def render_settings(shot: dict, film: dict) -> dict:
    """Effective model settings for a shot (resolves film/shot turbo presets and step overrides, and the draft tier
    that render() marks with shot["_tier"] = "draft")."""
    mode = shot.get("mode", "fl")
    draft = shot.get("_tier") == "draft"
    ref_video = has_ref_video(shot)
    # A reference video only steers non-turbo H3, so the film's turbo is not inherited; an explicit shot turbo is kept.
    turbo = shot.get("turbo", False if ref_video else film.get("turbo", False))
    if isinstance(turbo, dict):  # film-level per-mode map, e.g. {"fl": "fl4", "ref": "ref8"}
        turbo = turbo.get(mode, False)
    if draft and not turbo and not ref_video:  # drafts are turbo; the rim artifact is tolerated with control
        turbo = "ref8" if mode == "ref" and shot.get("control") else DRAFT_TURBO.get(mode, False)
    lora, shift, steps = None, None, film.get("steps", 20)
    if turbo:
        preset = TURBO_PRESETS[TURBO_DEFAULT[mode] if turbo is True else turbo]
        if preset["mode"] != mode:
            raise ValueError(f"{shot['id']}: turbo preset {turbo!r} is for {preset['mode']!r} shots, not {mode!r}")
        lora, shift, steps = preset["lora"], preset.get("shift"), preset["steps"]
    settings = {"mode": mode, "lora": lora, "shift": shift,
                "steps": steps if draft and lora else shot.get("steps", steps),
                "width": shot.get("width", film.get("width", 1344)),
                "height": shot.get("height", film.get("height", 768))}
    if draft:
        settings["width"], settings["height"] = draft_size(
            settings["width"], settings["height"], DRAFT_SHORT_SIDE_REF_VIDEO if ref_video else DRAFT_SHORT_SIDE)
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
    control = shot.get("control")
    if control:
        # H3 Fun ControlNet Union: a depth/canny/pose video (e.g. greybox.py's <id>_depth.mp4) steers every frame.
        patch = g.add("ModelPatchLoader", name=control.get("patch", CONTROL_PATCH))
        video = g.add("LoadVideo", file=images[control["video"]])
        frames_in = g.add("GetVideoComponents", video=[video, 0])
        unet = g.add("MiniMaxH3FunControlNetApply", model=[unet, 0], model_patch=[patch, 0], vae=[vvae, 0],
                     strength=control.get("strength", 1.0), start_percent=control.get("start", 0.0),
                     end_percent=control.get("end", 1.0), control_video=[frames_in, 0])
    load = lambda p: [g.add("LoadImage", image=images[p]), 0]  # noqa: E731
    load_audio = lambda p: [g.add("LoadAudio", audio=images[p]), 0]  # noqa: E731
    lock = shot.get("_audio_lock")  # locked guide audio (audio_lock): anchored at frame 0, as P01/P15 did
    locked = False

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
        if shot.get("ref_video"):
            rv = g.add("LoadVideo", file=images[shot["ref_video"]["video"]])
            kw["ref_videos.ref_video_0"] = [g.add("GetVideoComponents", video=[rv, 0]), 0]
        cond = g.add("MiniMaxH3ReferenceToVideo", clip=[clip, 0], vae=[vvae, 0], audio_vae=[avae, 0],
                     prompt=shot["_prompt"], width=w, height=h, length=length, ref_image_size="match", **kw)
        positive, latent = [cond, 0], [cond, 1]
        for guide in shot.get("guides", []):
            idx = min(round(guide["t"] * FPS), length - 1)
            gid = g.add("MiniMaxH3AddGuide", positive=positive, latent=latent, frame_idx=idx,
                        vae=[vvae, 0], audio_vae=[avae, 0], image=load(guide["image"]),
                        **({"audio": load_audio(lock)} if lock and idx == 0 and not locked else {}))
            locked = locked or bool(lock and idx == 0)
            positive = [gid, 0]
    else:
        raise ValueError(f"{shot['id']}: unknown mode {mode!r}")
    if lock and not locked:  # no frame-0 guide to carry it: an audio-only guide
        gid = g.add("MiniMaxH3AddGuide", positive=positive, latent=latent, frame_idx=0, audio_vae=[avae, 0],
                    audio=load_audio(lock))
        positive = [gid, 0]

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
def stamp(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def compose_h3_ref(shot: dict, film: dict) -> tuple[str, list[str]]:
    """A ref-mode prompt in MiniMax's own six-section Ref2VA format (subject_definitions, summary,
    retention_analysis, detailed_description, overall_soundscape, non_diegetic_music). In "prompt" (the
    detailed_description body, starting "[Shot 1]"), "@<subject key>" becomes <Subject N>, "@key<n>" the n-th
    guide keyframe's <Picture N> and "@frame" the framing still. Speakers (S1, S2, ... in "speakers" order,
    default the shot's voices) are written by hand in the prompt; each voice is bound to the subject of the same
    key. "audio" is the overall_soundscape, "music" the non_diegetic_music (default N/A).
    With a "ref_video" the wording follows the blocking_ref2vid research (runs A/C/D): each subject in
    ref_video["motion"] ({key: "position, step and hand signal"}; default every subject, "position, movement and
    timing") takes its appearance from its Pictures and that motion from <Video 1>; other subjects are sets redrawn
    from <Video 1>'s viewpoint. ref_video["desc"] says what the video is, ref_video["defines"] overrides what it
    defines; <Video 1> is partially_preserved and its placeholder surfaces become the subjects. A subject's optional
    "look" (a detail list) follows its source clause, as in the research prompts."""
    import re
    refs: list[str] = []
    subject_tag: dict[str, str] = {}
    defs, keep = [], []
    video = shot.get("ref_video")
    motion = (video.get("motion") or {k: "position, movement and timing" for k in shot.get("subjects", [])}
              if video else {})
    for n, key in enumerate(shot.get("subjects", []), start=1):
        subj = film["subjects"][key]
        own = subj["ref"] if isinstance(subj["ref"], list) else [subj["ref"]]
        tags = [f"<Picture {len(refs) + k}>" for k in range(1, len(own) + 1)]
        refs += own
        shown = tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]
        subject_tag[key] = f"<Subject {n}>"
        look = f": {subj['look']}" if subj.get("look") else ""  # optional detail list, as in the research prompts
        if key in motion:  # appearance from its pictures, motion from the blocking video
            defs.append(f"<Subject {n}> is {subj['desc']} whose appearance comes from {shown} and whose {motion[key]} "
                        f"come from <Video 1>{look}. Its face, build, costume and colours come only from {shown}.")
            keep.append(f"<Subject {n}> (appears in [Shot 1]): fully_preserved - face, build, costume, colours and "
                        f"carried props from {shown}.")
        elif video:  # a set: design from its picture, viewpoint from the video
            defs.append(f"<Subject {n}> is {subj['desc']} in {shown}{look}. {shown} supplies the set design, palette and "
                        f"drawing style; it is a design reference, not a shot to cut to.")
            keep.append(f"<Subject {n}> (appears in [Shot 1]): partially_preserved - the design, palette and drawing "
                        f"style of {shown}, redrawn from the viewpoint of <Video 1>.")
        else:
            defs.append(f"<Subject {n}> is {subj['desc']}, shown in {shown}.")
            keep.append(f"<Subject {n}> (appears in [Shot 1]): fully_preserved - face, build, costume, colours and "
                        f"carried props are retained exactly as in {shown}.")
    pictures = {}
    if shot.get("framing"):
        refs.append(shot["framing"])
        pictures["frame"] = f"<Picture {len(refs)}>"
        defs.append(f"{pictures['frame']} is a storyboard reference for [Shot 1], defining its viewpoint, subject "
                    f"placement and composition only; every face and costume comes from the subjects' own pictures.")
        keep.append(f"{pictures['frame']} ([Shot 1] storyboard): weak_reference - viewpoint and placement only.")
    for m, guide in enumerate(shot.get("guides", []), start=1):
        if guide["image"] not in refs:
            refs.append(guide["image"])
        tag = f"<Picture {refs.index(guide['image']) + 1}>"
        pictures[f"key{m}"] = tag
        role = ("the first frame" if guide["t"] == 0 else "the last frame" if guide["t"] >= shot["duration"]
                else f"the keyframe at {stamp(guide['t'])}")
        defs.append(f"{tag} is {role} of [Shot 1], showing {guide.get('desc', 'the frame at that moment')}.")
        keep.append(f"{tag} ([Shot 1] {role}): fully_preserved - composition, poses, flat cel-shaded colours and "
                    f"linework are matched exactly at {stamp(guide['t'])}.")
    if len(refs) > 9:
        raise ValueError(f"{shot['id']}: {len(refs)} reference images; H3 takes at most 9")
    if video:
        pictures["anim"] = "<Video 1>"
        subjects = [subject_tag[k] for k in shot.get("subjects", [])]
        became = (subjects[0] if len(subjects) == 1 else ", ".join(subjects[:-1]) + " and " + subjects[-1]
                  ) if subjects else "the subjects"
        defs.append(f"<Video 1> is {video.get('desc', 'a low-resolution 3D blocking render of [Shot 1]')}. "
                    + video.get("defines", "It defines the camera position and camera path, the framing, where each "
                                "figure stands and which way it faces, and the timing of every movement.")
                    + " Its untextured surfaces, stand-in figures and lighting are placeholders only.")
        keep.append(f"<Video 1> (camera path and staging): partially_preserved - camera path, framing, figure "
                    f"placement and movement timing are followed; every placeholder surface becomes {became}.")
    speakers = shot.get("speakers", shot.get("voices", []))
    for n, key in enumerate(shot.get("voices", []), start=1):
        who = subject_tag.get(key, film["voices"][key]["desc"])
        defs.append(f"<Audio {n}> is the voice-timbre reference for {who} (S{speakers.index(key) + 1}).")
        keep.append(f"<Audio {n}>: reference - {who} speaks new words in exactly this voice timbre, pitch and "
                    f"accent without copying the original signal.")

    def expand(text: str) -> str:
        return re.sub(r"@(\w+)", lambda m: pictures.get(m.group(1)) or subject_tag.get(m.group(1)) or m.group(0),
                      text)

    task = "keyframe completion + reference generation" if shot.get("guides") else "reference generation"
    if shot.get("voices"):
        task += " + audio reference"
    style = film.get("styles", {}).get(shot["style"]) if shot.get("style") else film.get("style")
    body = expand(shot["prompt"])
    unresolved = re.findall(r"@\w+", body)
    if unresolved:
        raise ValueError(f"{shot['id']}: unknown tokens {unresolved}")
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(defs),
        f"summary:\n[{task}] {expand(shot.get('summary', ''))}".rstrip(),
        "retention_analysis:\n" + "\n".join(keep),
        f"detailed_description:\n{style}\n{body}" if style else f"detailed_description:\n{body}",
        f"overall_soundscape:\n{expand(shot.get('audio', 'N/A'))}",
        f"non_diegetic_music:\n{shot.get('music', film.get('music_line', 'N/A'))}",
    ]), refs


def compose_prompt(shot: dict, film: dict) -> tuple[str, list[str]]:
    """Returns the prompt and ordered picture paths: subjects, framing, then ref-mode guide frames."""
    if shot.get("mode") == "ref" and film.get("prompt_format") == "h3_ref":
        return compose_h3_ref(shot, film)
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




def wait_for(prompt_id: str, label: str, base: str = H3_URL) -> dict:
    t0 = time.time()
    while True:
        hist = http_json(f"/history/{prompt_id}", base=base)
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


def fetch_video(entry: dict, dest: Path, base: str = H3_URL) -> None:
    for node_out in entry["outputs"].values():
        for item in node_out.get("images", []) + node_out.get("videos", []) + node_out.get("gifs", []):
            if item.get("type") == "output" and item.get("filename", "").endswith((".mp4", ".webm", ".mkv")):
                dest.write_bytes(view(item, base, timeout=300))
                return
    raise RuntimeError(f"no video in outputs: {json.dumps(entry['outputs'])[:1000]}")


def render(shotlist: Path, only: set[str] | None, force: bool, dry_run: bool, restamp: bool = False,
           tier: str | None = None, draft_seeds: int | None = None) -> None:
    film = json.loads(shotlist.read_text(encoding="utf-8"))
    root = shotlist.resolve().parent
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
        shot_tier = tier or shot.get("tier", film.get("tier", "final"))
        if shot_tier not in TIERS:
            raise SystemExit(f"{sid}: unknown tier {shot_tier!r} (one of {', '.join(TIERS)})")
        if shot_tier == "draft" and shot.get("mode") in ("clip", "still", "black", "motion_graphic"):
            print(f"- {sid}: {shot['mode']} shot, no draft tier")
            continue
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
        if shot.get("mode") == "motion_graphic":
            import motion_graphic  # PIL frame renderer; no GPU
            settings = render_settings(shot, film)
            w, h = settings["width"], settings["height"]
            digest = motion_graphic.digest(root, shot, w, h)
            sidecar = dest.with_suffix(".json")
            if not force and dest.exists() and sidecar.exists() and json.loads(sidecar.read_text())["digest"] == digest:
                print(f"= {sid}: up to date")
                continue
            if dry_run:
                print(f"~ {sid}: motion graphic (dry run)")
                continue
            motion_graphic.render(root, shot, w, h, dest)
            sidecar.write_text(json.dumps({"digest": digest, "graphic": shot["graphic"]}, indent=2), encoding="utf-8")
            print(f"+ {sid}: motion graphic {shot['graphic']['template']}")
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
        if has_ref_video(shot):
            # The reference video carries camera and staging; pinned frames would override it. Guides are opt-in:
            # only the shot's own "guides" list, never first_frame/last_frame turned into pins.
            dropped = [k for k in ("first_frame", "last_frame") if shot.pop(k, None)]
            if dropped:
                print(f"! {sid}: ref-video shot, ignoring {'/'.join(dropped)} (list pins in \"guides\" to keep them)",
                      file=sys.stderr)
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
        if shot.get("control"):
            shot["control"] = dict(shot["control"], video=str(root / shot["control"]["video"]))
        if shot.get("ref_video"):
            shot["ref_video"] = dict(shot["ref_video"], video=str(root / shot["ref_video"]["video"]))
        if shot.get("audio_lock"):
            lock = shot["audio_lock"] if isinstance(shot["audio_lock"], str) else shot.get("guide_audio")
            if not lock:
                raise SystemExit(f"{sid}: audio_lock needs a \"guide_audio\" file (or \"audio_lock\": \"<path>\")")
            shot["_audio_lock"] = str(root / lock)
        local = [Path(p) for p in (shot.get("first_frame"), shot.get("last_frame"), *shot["_refs"],
                                   *(gd["image"] for gd in shot.get("guides", [])), *shot["_voices"],
                                   (shot.get("control") or {}).get("video"), (shot.get("ref_video") or {}).get("video"),
                                   shot.get("_audio_lock")) if p]
        for p in local:
            if not p.exists():
                raise SystemExit(f"{sid}: missing input {p}")
        draft = shot_tier == "draft"
        if draft:
            # Settings preview: low-res takes of seed, seed+1, ... under renders/_draft, never over the final.
            shot["_tier"] = "draft"
            n = draft_seeds or int(shot.get("draft_seeds", film.get("draft_seeds", 3)))
            (renders / "_draft").mkdir(exist_ok=True)
            takes = [(shot["_seed"] + k, renders / "_draft" / f"{sid}_s{shot['_seed'] + k}.mp4") for k in range(n)]
        else:
            takes = [(shot["_seed"], dest)]
            if shot.get("control") and render_settings(shot, film)["lora"]:
                print(f"! {sid}: final with a control video and turbo LoRA {render_settings(shot, film)['lora']}: "
                      "expect the turbo+control contour-rim artifact (prompting.md); render finals with control "
                      "non-turbo", file=sys.stderr)
        if has_ref_video(shot) and render_settings(shot, film)["lora"]:
            print(f"! {sid}: ref-video shot with turbo LoRA {render_settings(shot, film)['lora']}: turbo makes H3 ignore "
                  "the reference video (blocking_ref2vid research: bbox IoU 0.0, invented camera); drop \"turbo\"",
                  file=sys.stderr)
        image_hashes = [hashlib.sha1(p.read_bytes()).hexdigest() for p in local]
        uploaded: dict[str, str] | None = None
        for seed, out in takes:
            shot["_seed"] = seed
            # Cache key = everything that changes the rendered pixels/audio. Narration, bed, the edit-only
            # titles/zoom/hold/out/pillarbox and the literal seed/turbo/steps/tier keys are excluded; their effective
            # values are covered by "_seed", "_tier" (drafts only, so final keys are unchanged) and "_render".
            spec = {k: v for k, v in shot.items() if k not in (
                "seed", "turbo", "steps", "width", "height", "narration", "bed", "titles", "zoom", "hold", "out", "in",
                "sfx",
                "pillarbox", "first_frame", "last_frame", "guides", "_refs", "_voices", "tier", "draft_seeds",
                "_audio_lock")}
            spec["_images"] = image_hashes
            spec["_guides"] = [g["t"] for g in shot.get("guides", [])]
            spec["_render"] = render_settings(shot, film)
            if shot.get("ref_video") or shot.get("control"):
                # Older versions could cache LoadVideo's input preview instead of SaveVideo's generated output.
                spec["_video_output"] = "generated"
            digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            sidecar = out.with_suffix(".json")
            label = f"{sid} s{seed}" if draft else sid
            if not force and out.exists() and sidecar.exists() and json.loads(sidecar.read_text())["digest"] == digest:
                print(f"= {label}: up to date")
                continue
            if restamp and out.exists() and sidecar.exists():
                meta = json.loads(sidecar.read_text())
                sidecar.write_text(json.dumps(meta | {"digest": digest}, indent=2), encoding="utf-8")
                print(f"# {label}: restamped existing render as current")
                continue
            if shot.get("mode") == "black":
                # Picture-free beat (e.g. a character losing sight): black frames + a faint room-tone bed.
                if not dry_run:
                    w, h = render_settings(shot, film)["width"], render_settings(shot, film)["height"]
                    d = frames_for(shot["duration"]) / FPS
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                                    f"color=c=black:s={w}x{h}:r={FPS}:d={d:.3f}", "-f", "lavfi", "-i",
                                    f"anoisesrc=color=brown:amplitude=0.01:r=48000:d={d:.3f}", "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-shortest", str(out)],
                                   check=True)
                    sidecar.write_text(json.dumps({"digest": digest, "seed": seed, "prompt": "black"}, indent=2),
                                       encoding="utf-8")
                print(f"+ {sid}: black {shot['duration']}s")
                continue
            if uploaded is None:
                uploaded = {str(p): (upload_image(p, base=H3_URL) if not dry_run else p.name) for p in local}
            prefix = f"h3film/{slug}/_draft/{out.stem}" if draft else f"h3film/{slug}/{sid}"
            graph = build_graph(shot, film, uploaded, prefix=prefix)
            api = out.with_suffix(".api.json")
            api.write_text(json.dumps(graph, indent=2), encoding="utf-8")
            if dry_run:
                if draft:
                    r = render_settings(shot, film)
                    print(f"~ {label}: draft graph written ({shot.get('mode', 'fl')}, {frames_for(shot['duration'])} "
                          f"frames, {r['width']}x{r['height']}, {r['lora']}, {r['steps']} steps) -> {api}")
                else:
                    print(f"~ {sid}: graph written ({shot.get('mode', 'fl')}, {frames_for(shot['duration'])} frames)")
                continue
            t0 = time.time()
            resp = http_json("/prompt", {"prompt": graph, "client_id": "h3_render"}, base=H3_URL)
            if resp.get("node_errors"):
                raise RuntimeError(f"{label}: {json.dumps(resp['node_errors'])[:3000]}")
            entry = wait_for(resp["prompt_id"], label, base=H3_URL)
            fetch_video(entry, out, base=H3_URL)
            sidecar.write_text(json.dumps({"digest": digest, "prompt_id": resp["prompt_id"], "seed": seed,
                                           **({"tier": "draft"} if draft else {}),
                                           "seconds": round(time.time() - t0), "prompt": shot["_prompt"]}, indent=2),
                               encoding="utf-8")
            print(f"\r+ {label}: {out} ({time.time() - t0:.0f}s)          ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--only", help="comma-separated shot ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restamp", action="store_true",
                    help="accept existing renders of the selected shots as current (after a cache-key-only change)")
    ap.add_argument("--tier", choices=TIERS, help="override every shot's tier (draft: 512p turbo seed batch)")
    ap.add_argument("--draft-seeds", type=int, help="seeds per draft shot (default: the shot/film draft_seeds, else 3)")
    a = ap.parse_args()
    render(a.shotlist, set(a.only.split(",")) if a.only else None, a.force, a.dry_run, a.restamp, a.tier,
           a.draft_seeds)


if __name__ == "__main__":
    sys.exit(main())
