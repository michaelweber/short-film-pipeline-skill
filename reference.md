# Reference: tools and shots.json

Each film lives in `film/<name>/` with its `shots.json`; every path in it is relative to that file. Outputs land
next to it: `stills/`, `renders/`, `narration/`, `music/`, `edit/`, and `<name>.mp4`.

```
film/<name>/treatment.md ──► film/<name>/shots.json
                                   │
  tools/keyframes.py    (Krea-2 sheets → Flux.2 Klein keyframes)          ──► stills/*.png
  tools/h3_render.py    (MiniMax H3 image/ref → video + audio, per shot)   ──► renders/*.mp4
  tools/speech_qa.py    (vocal stem + ASR vs script, flags babble)         ──► renders/speech_qa.json
  tools/narrate.py      (cloned-voice voice-over clips + timing report)    ──► narration/*.wav
  tools/music.py        (YuE2 instrumental score, hit onsets)              ──► music/score_*.flac
  tools/resolve_edit.py (DaVinci Resolve: timeline, ducking, loudness)     ──► <name>.mp4
```

## Run
ComfyUI must be running (default `http://127.0.0.1:8188`; set `COMFYUI_URL` to change it). On large-VRAM cards
start it with `--disable-dynamic-vram` (see troubleshooting.md).

```bash
python tools/keyframes.py film/<name>/shots.json            # character/set sheets, then keyframes
python tools/h3_render.py film/<name>/shots.json            # every shot not yet rendered
python tools/h3_render.py film/<name>/shots.json --only s11 --force
python tools/speech_qa.py film/<name>/shots.json            # transcribe dialogue, flag mismatches
$PY_TORCH tools/voice_similarity.py film/<name>/voices/hero.wav film/<name>/renders/s04.vocals.flac
python tools/narrate.py film/<name>/shots.json --plan       # narration timing report, no GPU
python tools/narrate.py film/<name>/shots.json              # TTS missing lines, then the report
python tools/music.py film/<name>/shots.json                # score, hit onsets, vocal check
python tools/resolve_edit.py film/<name>/shots.json         # cut + mix + render in Resolve -> <name>.mp4
python tools/resolve_edit.py film/<name>/shots.json --no-render   # timeline only, to finish by hand
python tools/sheet.py film/<name>/shots.json --stills --tag stills         # contact sheets for review
python tools/sheet.py film/<name>/shots.json --ids s10 --cols 16 --crop 0.55,0.1,0.9,0.7   # close-up strip
python tools/mouth_hold.py film/<name>/shots.json --ids s10 --box 0.58,0.15,0.86,0.68     # silent-character hold
python tools/voice_ref.py <video url or file> --out film/<name>/voices/v.wav --keep "phrase|phrase"
```
`$PY_TORCH` is any Python with torch, transformers and librosa (ComfyUI's embedded Python works).
keyframes.py and h3_render.py cache results: a still or shot re-renders only when its prompt, parameters or input
images change, so you can edit one prompt and rerun the whole list. The API graph for each shot is written next
to its render (`renders/<id>.api.json`); load it into the ComfyUI UI to tweak it by hand.

## shots.json
- Film level: `title`, `project_prefix` (Resolve project = prefix + film dir), `style`, `styles`, `still_style`,
  `width`, `height`, `seed`, `turbo`, `steps`, `subjects`, `voices`, `stills`, `shots`, plus the narration,
  music and edit fields below.
- `stills.<id>`: `prompt`, `width`/`height`, `seed`, optional `refs` (`"@hero"` = another still, or a file path),
  optional `engine` (`krea2` or `klein`). A still with no refs uses Krea-2 Turbo; a still with refs uses
  Flux.2 Klein 9B reference editing. Build character and set sheets first, then each shot's keyframe from them.
- `shots[]`: `id`, `duration` (seconds; H3 is trained on 5–15 s), `mode`, `prompt`, `audio`,
  `first_frame` / `last_frame` (a still path, or `"@prev"` for the last frame of the previous shot), `seed`.
  - `mode: "fl"`: the fl2va model; text→video, image→video, or first+last-frame interpolation.
  - `mode: "ref"`: the ref2va model; `subjects` (keys of top-level `subjects` → `{desc, ref}`; `ref` is one path
    or a list of several real photos), `voices` (keys of top-level `voices` → `{desc, audio, ref_text}`), and
    `guides` (`[{t, image, desc}]`) that pin keyframes to timestamps. `first_frame`/`last_frame` become guides.
    `framing` (a still path) is the soft alternative: one more reference image, described as a framing/layout
    reference only, so likeness comes from the subjects' photos. At most 9 reference images per shot.
  - `mode: "black"`: black picture with a faint room-tone bed and no GPU work (title cards, narration-only beats).
  - `mode: "clip"`: `"source": "media/intro.mp4"` drops pre-made footage into the cut with its own audio.
  - `mode: "still"`: `"source": "stills/photo.png"`, `"duration"`: a photograph as a shot, cover-fitted to the film
    size, with a silent track and no GPU work. `"kenburns": {"from": [cx, cy, zoom], "to": [cx, cy, zoom]}` pans
    and zooms linearly across the shot: `cx`/`cy` are the centre of the visible window as fractions of the
    cover-fitted frame, `zoom` ≥ 1, centres clamped so the window stays inside the picture. The move is rendered
    into the file (2× supersampled), because Resolve 21.1's scripting API has no keyframe calls.
  - `styles` (top level) plus `"style": "<name>"` on a shot replaces the film-wide `style` for that shot.
  - `"loras": [["file.safetensors", strength], ...]` stacks extra LoRAs after the turbo LoRA; `"trigger"` puts
    trigger words at the very start of the prompt. Both are part of the cache key. A/B them per shot.
- Turbo: `"turbo": true` uses each mode's default preset; name a preset per shot, or set a film-level map such as
  `"turbo": {"fl": "fl4", "ref": "ref8"}`. Presets (LoRAs in `models/loras`): `fl8`, `fl8_768`, `fl4` (4-step
  v1.2 768p with video/audio shift 6/3; the audio-fix release), `ref4`, `ref8`.
- Consistent voices: pick a clean line from any render (`speech_qa.py` writes `renders/<id>.vocals.flac`),
  trim/normalise it into `voices/<name>.wav`, then put the shots where that character speaks in `ref` mode with
  `"voices": ["<name>"]` and write "in the voice of <Audio 1>" in the `audio` line. `fl` shots invent a new voice
  every time.
- Prompt style: H3 follows timestamped beats (`[0s-3s] …`), camera language, and an explicit audio line. Put
  dialogue in quotes in `audio`; H3 generates the voices and lip-syncs them.

## Narration (voice-over)
- `"narration_voice": "hero"` points at `voices.hero` (`audio` is the reference clip, `ref_text` its transcript).
- `"narration_engine"`: `vibevoice` (VibeVoice-Large, default), `indextts2`, or `h3` (a MiniMax H3 take of the
  narrator reading into a microphone; `"narration_subject"` stages them on camera, only the audio is kept).
- Each shot can carry `"narration": [{"at": 1.5, "text": "...", "seed": 7, "file": "..."}]`. `at` is seconds into
  that shot; a line may run over into the next shot. `seed` re-rolls one take. `file` uses a ready-made clip,
  e.g. a piece cut from a long take with `narrate.split_take(take, [line, line, ...])`, which finds each line at
  envelope dips, checks it with ASR, and pads 0.15 s before / 0.6 s after. `voice` reads the line in another
  `voices` entry (a character's letter home in their own voice); `tempo` overrides `narration_tempo`.
- `"narration_tempo": 1.1` speeds every finished clip up with ffmpeg `atempo` (pitch kept), cached as
  `<hash>_x1.1.wav`. Cloned narrators tend to read slowly; 1.1–1.2 sounds natural.
- `"narration_isolate": true` keeps only the voice of every clip (Mel-Band RoFormer vocal stem), cached as
  `<hash>_iso.wav`. VibeVoice sometimes adds a music bed, especially when the reference clip had one; isolation
  lowers it, and a dry reference avoids it.
- `--plan` warns when lines overlap each other or sit on on-screen dialogue (speech spans after a shot's `out`
  are ignored).
- `"narration_ducking_db"` (-14): how far shot audio dips under the voice.
- Clips are cached in `narration/<hash>.wav` (hash of text, voice, engine settings and seed); isolation and tempo
  are derived files beside it, so changing them never re-renders a take.

## Music
`"music": {"style", "lyrics", "seed", "max_duration", "gain_db": -6, "duck_db": -9, "start": 0, "in": 0, "file"}`.
`tools/music.py` renders the score with YuE2 (checkpoint `yue2_3b_bf16`, an instrumental AR LoRA and a NAR LoRA;
see the constants at the top of the file) to `music/score_<hash>.flac` and prints its length, the hit onsets
(momentary loudness ≥ 6 LU over the previous 20 s median) and an ASR vocal check. For an instrumental score, use
section tags only as `lyrics` (`[intro 0:00-0:20]\n[verse 0:20-0:50]…`). `--seed N` renders a candidate without
editing the JSON. `"file"` uses a supplied track instead. The cut lays the score on A3 from `start` (seconds
into the cut), `in` seconds into the score (1 s fade in when `in` > 0), at `gain_db`, dipping a further
`duck_db` under narration and on-screen dialogue, with a 1.5 s fade out.

## Edit (DaVinci Resolve Studio)
`tools/resolve_edit.py` drives Resolve through the official scripting API (`DaVinciResolveScript`). Resolve must be
running with Preferences > General > External scripting using = Local. If Resolve is not in the default location,
set `RESOLVE_SCRIPT_API` and `RESOLVE_SCRIPT_LIB`.
- One project per film, `<project_prefix><film dir>`. Every run adds a timeline `<title> NN`, so earlier cuts stay.
- Tracks: V1 "Picture" (shot renders); A1 "Shot audio"; A2 "Narration"; A3 "Music" (films with `"music"`).
  Video tracks above V1: "Holds" (if any shot has a hold), "Letterbox" (if the film has `"letterbox"`), then
  "Titles" tracks (title k of a shot on its own track, since titles overlap).
- Film-level edit fields: `"letterbox": 2.39` (black bars for that aspect ratio), `"loudness_lufs"` (-16),
  `"crf"` (16; 23 halves the file).
- Per-shot edit fields (they never re-render a shot):
  - `"titles": [{"text", "style", "at", "dur", "align"}]`: transparent overlays. Styles: `rank` (big yellow number,
    top right; `"align": "left"`), `lower` (lower third), `card` (centred magenta lines), `place` (small upper-case
    location super inside the bottom letterbox bar), `doc` (centred title card: big first line, smaller further
    lines, gold rules above and below).
  - `"zoom": 1.25` punches in on the V1 item.
  - `"pillarbox": 1.3333` crops the V1 item's sides so only that aspect ratio stays visible (4:3 archival TV); the
    black timeline background shows through. Keep people in the central 4:3 (say so in `style`).
  - `"hold": {"from", "dur", "crop": {"left": 0.57, "softness": 30}}` lays a short stretch of the shot's own
    render, looped forward and backward and cropped, over the whole shot. It keeps a silent character's mouth
    shut (H3 lip-syncs every mouth in frame). `tools/mouth_hold.py` picks the stretch.
  - `"out": 2.4` ends the shot early, e.g. to drop words H3 invented after a short line.
  - `"bed"`: `"full"` (default), `"instruments"` (the shot's audio with vocals removed), or `"none"` (no shot audio:
    music and narration only; use it for every shot without a spoken line).
- Ducking lives on the timeline: A1 and A3 are split around each narration line (0.15 s lead, 0.25 s tail, gaps
  under 0.35 s bridged) and the pieces under the voice drop, joined by 6-frame "Cross Fade 0 dB" transitions.
  It stays editable in Resolve like any hand-made duck.
- Loudness: render and deliver, measure the delivered file with ffmpeg `ebur128`, shift every audio item by the
  error to `loudness_lufs`, and repeat (up to 4 passes) until within 0.3 LU. If AAC pushes the true peak above
  −1 dBTP, the limiter ceiling is lowered and the file re-encoded.
- Resolve renders `edit/<film>_master.mov` (H.264 + 24-bit PCM); ffmpeg encodes `<film>.mp4` (x264 + AAC through a
  4x-oversampled limiter). Resolve's own AAC encoder puts a click in the first 50 ms, and its master limiter,
  automation curves and bus routing have no scripting API.
- The previous `<film>.mp4` is kept as `_iterN.mp4`. The cut list is exported to `edit/<timeline>.otio`, which also
  opens in Premiere or Kdenlive.
- Media is imported from content-addressed copies in `edit/media/`: Resolve doesn't notice a file rewritten in
  place, so a re-rendered shot becomes a new media-pool item and older timelines keep their takes.

## Timing (one 96 GB workstation GPU, turbo presets)
- H3 at 1344×768, 5 s: `fl4` ~1–2 min, `ref8` ~2–4 min per shot; switching models adds load time, so run all
  stills first and then all shots.
- Stills: Krea-2 ~5–30 s, Klein with refs ~10–40 s.
- H3 narration takes (≤ 15 s): ~5–10 min each. VibeVoice-Large: ~10 min for a 35 s whole-script take.
- YuE2 score, 150 s: ~3.5 min.

## Known issues
- H3 sometimes adds invented words before or after a line, or reads a short line slowly across the whole shot.
  `speech_qa.py` catches the first as low precision; "says exactly one line and nothing else" plus "No other
  voices" usually fixes it, and `out` trims the rest.
- Colour priming: any colour named in the prompt gets attached to faces or props. Describe what *is* there.
- H3 pushes into close-ups even when told "static". `last_frame` = `first_frame` pins composition and scale.
