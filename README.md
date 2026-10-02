# short-film-pipeline-skill

An agent skill (Claude Code / any `SKILL.md`-aware harness) for turning a script or story into a short AI film
on local hardware, with an editable DaVinci Resolve cut as the deliverable:

- **Stills**: Fizgig/H3 reference-guided stills, character and location sheets; Krea-2 Turbo sheets and Flux.2
  Klein keyframes as the older route.
- **Shots**: MiniMax H3 image/reference-to-video with generated dialogue, cloned voices and lip-sync, written in
  MiniMax's official prompt format; PIL motion graphics for screens and infographics.
- **Voice-over**: H3 narration takes (small-picture, audio only) with per-line delivery, cut at real pauses and
  gated by Whisper large-v3 (verbatim decode: extra words, clipped edges, stretched words, stray audio).
- **Sound**: YuE2 instrumental score cues (CLAP-ranked), H3 audio-only sound-effect takes (CLAP-picked).
- **QA**: vocal separation + ASR against the script, speaker similarity, contact sheets, dialogue-trim checks.
- **Cut**: DaVinci Resolve Studio via its scripting API: timeline, titles, letterbox, holds, narration ducking,
  SFX track, loudness to target, H.264/AAC delivery, OTIO export.

`SKILL.md` is the workflow the agent follows; `prompting.md`, `qa.md`, `fizgig.md` and `troubleshooting.md` hold
the rules learned the hard way; `art-animation.md` and `models.md` cover optional stylized animation and
Blender/rig staging; `reference.md` documents every `shots.json` field and tool; `templates/` has a starter
treatment and shot list; `tools/` has the scripts.

## Install
Clone into your harness's skills directory, e.g. for Claude Code:

```bash
git clone https://github.com/michaelweber/short-film-pipeline-skill .claude/skills/short-film-pipeline
```

Then ask the agent to make a film; it reads `SKILL.md` and runs `tools/*.py` against `film/<name>/shots.json`
in your project.

## Machine settings
Machine-specific values are read by `tools/pipeline_settings.py`: an environment variable (the upper-case key)
wins, then `tools/local_settings.json` (untracked), then a neutral default.

```json
{
  "py_torch": "C:/path/to/python_with_torch.exe",
  "h3_gpu_uuid": "GPU-…", "h3_gpu_name": "part of the H3 card's name",
  "aux_gpu_uuid": "GPU-…", "aux_gpu_name": "part of the aux card's name",
  "gpu_lease_dir": "C:/path/to/.gpu_leases",
  "resolve_project_prefix": "myfilms_",
  "resolve_script_lib": "C:/path/to/DaVinci Resolve/fusionscript.dll"
}
```

## Requirements
- **ComfyUI**: an H3 instance (default `http://127.0.0.1:8188`, `H3_COMFY_URL`) and optionally an aux instance on
  a second GPU (default `:8189`, `AUX_COMFY_URL`) for stills, separation, ASR and music. Models and nodes:
  - MiniMax H3 nodes and models (`minimax_h3_fl2va_*`, `minimax_h3_ref2va_*`, the H3 text encoder and video/audio
    VAEs) plus the turbo LoRAs named in `tools/h3_render.py`.
  - Flux.2 Klein 9B and Krea-2 Turbo (model names in `tools/keyframes.py`) for the keyframe route.
  - YuE2 (`yue2_3b_bf16.safetensors`) and the instrumental AR / NAR LoRAs named in `tools/music.py` (music only).
  - Custom nodes: TTS Audio Suite (Granite ASR, IndexTTS-2), VibeVoice-ComfyUI, MelBandRoFormer.
- A large-VRAM GPU for H3 (developed on a 96 GB card; 1344×768 needs a lot of memory).
- **ffmpeg / ffprobe** on `PATH`.
- **Python 3.10+** with Pillow and numpy for the tools; a Python with torch, transformers and librosa (the
  `py_torch` setting; ComfyUI's embedded Python works) for `asr_words.py` (Whisper large-v3), `clap_score.py`
  (LAION CLAP), `voice_similarity.py`; `yt-dlp` for `voice_ref.py` with URLs. The Hugging Face models download on
  first use.
- **DaVinci Resolve Studio** with Preferences > General > External scripting using = Local, for the cut. Set
  `RESOLVE_SCRIPT_API` / `resolve_script_lib` if it isn't installed in Blackmagic's default location.
- Optional: Blender 5.x for greybox blocking (`tools/greybox.py`) and rig staging.
- Title fonts (Impact, Bahnschrift, Segoe UI Bold, Arial Bold) are looked up by filename; they ship with Windows.

## Notes
- You're responsible for the rights to any real person's or character's likeness or voice you clone (consent,
  parody/fair use, local law). Check before publishing or monetising.
- Check the licences of the models you use; the YuE2 instrumental LoRAs, for example, are non-commercial.
