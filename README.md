# short-film-pipeline-skill

An agent skill (Claude Code / any `SKILL.md`-aware harness) for turning a script or story into a short AI film
on local hardware:

- **Stills**: Krea-2 Turbo character/set sheets, Flux.2 Klein reference-edited keyframes.
- **Shots**: MiniMax H3 image/reference-to-video with generated dialogue, cloned voices and lip-sync.
- **Voice-over**: VibeVoice-Large, IndexTTS-2 or H3 narration, split from long takes with ASR-checked cuts.
- **Score**: YuE2 instrumental music with hit detection for aligning the title card.
- **QA**: vocal separation + ASR against the script, speaker similarity, contact sheets.
- **Cut**: DaVinci Resolve Studio via its scripting API: timeline, titles, letterbox, holds, ducking, loudness to
  target, H.264/AAC delivery, OTIO export.

`SKILL.md` is the workflow the agent follows; `prompting.md`, `qa.md` and `troubleshooting.md` hold the rules
learned the hard way; `reference.md` documents every `shots.json` field; `templates/` has a starter treatment and
shot list; `tools/` has the scripts.

## Install
Clone into your harness's skills directory, e.g. for Claude Code:

```bash
git clone https://github.com/michaelweber/short-film-pipeline-skill .claude/skills/short-film-pipeline
```

Then ask the agent to make a film; it reads `SKILL.md` and runs `tools/*.py` against `film/<name>/shots.json`
in your project.

## Requirements
- **ComfyUI** running locally (default `http://127.0.0.1:8188`, override with `COMFYUI_URL`) with:
  - MiniMax H3 nodes and models (`minimax_h3_fl2va_*`, `minimax_h3_ref2va_*`, the H3 text encoder and video/audio
    VAEs) plus the turbo LoRAs named in `tools/h3_render.py`.
  - Flux.2 Klein 9B and Krea-2 Turbo (model names in `tools/keyframes.py`).
  - YuE2 (`yue2_3b_bf16.safetensors`) and the instrumental AR / NAR LoRAs named in `tools/music.py` (music only).
  - Custom nodes: TTS Audio Suite (Granite ASR, IndexTTS-2), VibeVoice-ComfyUI, MelBandRoFormer.
- A large-VRAM GPU (developed on a 96 GB card; H3 at 1344×768 needs a lot of memory).
- **ffmpeg / ffprobe** on `PATH`.
- **Python 3.10+** with Pillow and numpy for the tools; a Python with torch, transformers and librosa for
  `voice_similarity.py` (ComfyUI's embedded Python works); `yt-dlp` for `voice_ref.py` with URLs.
- **DaVinci Resolve Studio** with Preferences > General > External scripting using = Local, for the cut. Set
  `RESOLVE_SCRIPT_API` / `RESOLVE_SCRIPT_LIB` if it isn't installed in Blackmagic's default location.
- Title fonts (Impact, Bahnschrift, Segoe UI Bold, Arial Bold) are looked up by filename; they ship with Windows.

## Notes
- You're responsible for the rights to any real person's or character's likeness or voice you clone (consent,
  parody/fair use, local law). Check before publishing or monetising.
- Check the licences of the models you use; the YuE2 instrumental LoRAs, for example, are non-commercial.
