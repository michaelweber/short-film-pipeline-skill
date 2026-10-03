# Film pipeline (local ComfyUI → MiniMax H3)

```
book/NN-*.md  ──► film/<story>/treatment.md ──► film/<story>/shots.json
                                                   │
          tools/greybox.py    (Blender, headless: set greybox + per-shot blocking) ──► blocking/*.png, *.blend
          tools/keyframes.py  (Krea-2 sheets → Flux.2 Klein keyframes)  ──► stills/*.png
          tools/h3_render.py  (MiniMax H3 image→video + audio, per shot) ──► renders/*.mp4
          tools/speech_qa.py  (vocal stem + ASR vs script, flags babble)  ──► renders/speech_qa.json
          tools/narrate.py    (cloned-voice voice-over clips + timing report) ──► narration/*.wav
          tools/resolve_edit.py (DaVinci Resolve: timeline, ducking, loudness, render) ──► <story>.mp4
```

## Run
The H3 instance of ComfyUI must be running (default :8188) on the big GPU, started with
`--disable-dynamic-vram` (see Troubleshooting). Stills, ASR and repairs can run on a
second, aux instance (default :8189) on another GPU; `tools/gpu_queue.py --card h3|aux` routes jobs, and the
cards' GPU UUIDs come from the `h3_gpu_uuid`/`aux_gpu_uuid` settings (see "Machine settings").

```bash
python tools/keyframes.py film/<name>/shots.json          # character/set sheets, then keyframes
python tools/h3_render.py film/<name>/shots.json          # every shot not yet rendered
python tools/h3_render.py film/<name>/shots.json --only s11 --force
python tools/speech_qa.py film/<name>/shots.json          # transcribe dialogue, flag mismatches
# voice consistency (needs torch, so use the py_torch Python):
$PY_TORCH tools/voice_similarity.py \
    film/<name>/voices/narrator.wav film/<name>/renders/s04.vocals.flac
python tools/narrate.py film/<name>/shots.json --plan  # narration timing report, no GPU
python tools/narrate.py film/<name>/shots.json         # TTS missing lines, then the report
python tools/resolve_edit.py film/<name>/shots.json    # cut + mix + render in Resolve -> <name>.mp4
python tools/resolve_edit.py film/<name>/shots.json --no-render   # timeline only, to finish by hand
python tools/sheet.py film/<name>/shots.json --stills --tag stills       # contact sheets for review
python tools/sheet.py film/<name>/shots.json --ids h10 --cols 16 --crop 0.55,0.1,0.9,0.7   # close-up strip
python tools/mouth_hold.py film/<name>/shots.json --ids h10 --box 0.58,0.15,0.86,0.68   # silent puppet hold
python tools/voice_ref.py <youtube url> --out film/<name>/voices/v.wav --keep "phrase|phrase"   # voice ref from a video
```

### Machine settings
Machine-specific values are read by `tools/pipeline_settings.py` (`setting(key, default)`): first the environment
variable named after the upper-cased key, then `tools/local_settings.json` (untracked, this machine's values), then
the built-in default. Keys:
- `py_torch` (`PY_TORCH`): a Python with torch/transformers (e.g. ComfyUI's embedded Python), used for
  `voice_similarity.py`, `asr_words.py` and other torch tools. Commands here write it as `$PY_TORCH`.
- `h3_gpu_uuid` / `aux_gpu_uuid` (`H3_GPU_UUID` / `AUX_GPU_UUID`): the GPU UUIDs (`nvidia-smi -L`) of the H3 and
  aux ComfyUI instances.
- `gpu_lease_dir` (`GPU_LEASE_DIR`): directory for the GPU queue's lease files.
- `resolve_project_prefix` (`RESOLVE_PROJECT_PREFIX`): prefix of the per-film Resolve project name.
- `resolve_script_lib` (`RESOLVE_SCRIPT_LIB`): path to Resolve's `fusionscript.dll`; defaults to the standard
  install location.

```json
{"py_torch": "C:/path/to/ComfyUI/python_embeded/python.exe", "h3_gpu_uuid": "GPU-…", "aux_gpu_uuid": "GPU-…"}
```
Both tools cache results. A still or shot re-renders only when its prompt, parameters or input images change,
so you can edit one prompt and rerun the whole list. The API graph for each shot is written next to its
render (`renders/<id>.api.json`); you can load that file into the ComfyUI UI to tweak it by hand.

## Narration (voice-over) track
Voice-over is generated separately rather than by H3, so the narrator sounds the same across the whole film.
- `"narration_voice": "narrator"` points at `voices.narrator` (`audio` is the reference clip, `ref_text` is its transcript).
  `"narration_engine"` is `vibevoice` (VibeVoice-Large wrapper, the default) or `indextts2`.
- Each shot can carry `"narration": [{"at": 1.5, "text": "..."}]`. `at` is seconds into that shot, and a
  line may run over into the next shot. `--plan` warns when lines overlap.
- `"narration_engine": "h3"` voices each line as a MiniMax H3 take (`"narration_subject"` stages the narrator on
  camera; only the audio is kept), cut to the scripted line at its pauses. A line's `"file"` uses a ready-made clip
  instead, e.g. a piece cut from a long take with `narrate.split_take()` (ASR-checked boundaries at envelope dips).
  `"narration_h3_short_side": 256` renders those takes' picture at a 256-pixel short side (only the audio is kept):
  ~18 s instead of ~5 min per take at 1344×768. Lines that already have a full-size take keep it.
  `"narration_h3_guide": true` writes the take in MiniMax's official Ref2VA format (`<d>[English] …</d>`, speaker
  IDs, `<Audio 1>` as an uncopied timbre reference, take length fitted to the words); the free-text format made H3
  read prompt text aloud before the line.
  Each take is cut to its line from word timestamps (`tools/asr_words.py`, local Whisper large-v3 on the aux GPU via
  the `py_torch` Python, kept running as a worker by `narrate.py`: ~1.2 s per take); the ComfyUI ASR dip search is the
  fallback.
  A line's `"delivery"` sets the tone it is read in (guide format: "says in <delivery>: …"), e.g. `"a slow, solemn,
  dramatic tone, quiet and heavy with loss"`; a "slow" delivery gets a longer take (2.0 instead of 2.6 words/s).
  `"narration_delivery"` is the film default. Changing a line's delivery makes a new take (it is in the cache key).
  Gate every take with `narrate.check_clip`: recall ≥ 0.9, no extra words or cut-off fragments in large-v3's
  verbatim-prompted decode (repeats, stutters, fillers, gibberish), first and last words heard, no word holding far
  more speech than its length (a repeat folded into it), no voiced audio after a pause outside the words, and the
  take near-silent at the cut end (`.cutv` sidecar). Cuts sit 60 ms into the real pauses (≥ 120 ms) around the
  line's first and last words; Whisper's word edges only choose the pause (they clip final consonants). Clips are
  re-cut from their saved takes when `CUT_VERSION` changes. `speech_qa.score` spells
  digits as words ("10,000" = "ten thousand", years as spoken) and matches compounds either way ("Starfield" =
  "Star field").
  A line's `"voice": "<key>"` reads it in another `voices` entry (a character's letter read in their own voice).
  `"narration_tempo": 1.2` (film) or a line's `"tempo"` speeds the finished clips up with ffmpeg `atempo` (pitch
  kept); the sped-up copy is cached beside the 1.0 clip as `<hash>_x1.2.wav`, so changing it never re-renders a take.
- `"bed": "instruments"` on a shot plays its H3 audio with the vocals separated out. Use it for shots whose
  H3 audio already had voice-over baked in. `"bed": "none"` drops the shot's audio from the cut entirely, so
  silent B-roll carries only music and narration. `"bed": "full"` (default) keeps a render's own sound effects
  (prompted in its `audio`, e.g. an explosion); check them with `tools/clap_score.py` first.
- Shot `"sfx": [{"sound", "action", "at", "dur", "gain_db", "seed"}]` adds generated sound effects:
  `tools/sfx.py` renders each as an audio-only H3 take of the shot's subjects (or keyframe) doing `action` with
  `sound` as the only audio (~15 s each), trims it from its first sound to `dur` and normalises it to -20 LUFS.
  `--audition N` renders seeds 1..N and keeps the one CLAP (`tools/clap_score.py`, laion/clap-htsat-unfused)
  rates most like `sound` against speech/music (`sfx/picks.json`). `resolve_edit.py` lays them on an "SFX" track
  at shot start + `at`, not ducked.
- The cut and mix happen in DaVinci Resolve Studio. `tools/resolve_edit.py` works through its scripting API:
  - One project per film, `<project_prefix><film dir>`. Every run adds a timeline `<title> NN`, so earlier cuts stay.
  - Tracks: V1 "Picture" holds the shot renders. A1 "Shot audio" holds each shot's H3 audio, or its instrumental
    stem. A2 "Narration" holds the narration clips, placed to the frame. A3 "Music" (films with `"music"`) holds
    the score. Video tracks above V1, in order: "Holds" (if any shot has a hold), "Letterbox" (if the film has
    `"letterbox"`), then the "Titles" tracks. An "SFX" audio track follows "Music" when any shot has `"sfx"`.
  - Film-level edit fields:
    - `"letterbox": 2.39` lays one full-length clip of black bars (top and bottom) for that aspect ratio.
    - `"music": {"style", "lyrics", "seed", "max_duration", "gain_db": -6, "duck_db": -9, "start": 0, "in": 0}`:
      `tools/music.py` renders the score with YuE2 (instrumental LoRA) to `music/score_<hash>.flac` and prints
      its length, the hit time and a vocal check; `--seed N` renders a candidate without editing the JSON. The
      cut lays the score for the JSON `seed` on A3 from `start` (seconds into the cut), `in` seconds into the
      score (1 s fade in when `in` > 0), at `gain_db`, dipping a further `duck_db` under narration and on-screen
      dialogue, with a 1.5 s fade out. `"file": "music/x.mp3"` uses a supplied score instead of the YuE2 render
      (`music.py` then only reports its length, hits and vocal check).
  - Per-shot edit fields (they never re-render a shot):
    - `"titles": [{"text", "style": "rank"|"lower"|"card"|"place"|"doc", "at", "dur"}]` become transparent
      overlays on the "Titles" tracks. `place` is a small upper-case location super inside the bottom letterbox
      bar; `doc` is a centred title card (big first line, smaller further lines, gold rules above and below).
    - `"zoom": 1.25` punches in on the V1 item.
    - `"pillarbox": 1.3333` crops the V1 item's sides to that visible aspect ratio (4:3 archival footage); the
      black timeline background shows through.
    - `"hold": {"from", "dur", "crop": {"left": 0.57, "softness": 30}}` lays a short stretch of the shot's own
      render, looped forward and backward and cropped, over the whole shot on V2 "Holds". It keeps a silent
      character's mouth shut (H3 lip-syncs every mouth in frame). `tools/mouth_hold.py` picks the stretch.
      It's the same camera and light, so the seam doesn't show.
    - `"out": 2.4` ends the shot early, e.g. to drop words H3 invented after a short line. `"in": 0.9` starts it
      later (render seconds) to drop words invented before the line; narration `at` stays relative to the cut shot.
    - `"titles"` rank badges sit top right; add `"align": "left"` when the right side is busy.
  - `"mode": "clip"` shots (`"source": "media/intro.mp4"`) drop pre-made footage into the cut with its own audio.
  - `"mode": "still"` shots (`"source": "stills/x.png"`, `"duration"`) turn a photograph into a shot with a silent
    track, cover-fitted to the film size. `"kenburns": {"from": [cx, cy, zoom], "to": [cx, cy, zoom]}` (window
    centre as fractions of the cover-fitted frame, zoom ≥ 1) pans/zooms linearly across the shot. The move is
    rendered by `h3_render.py` (2× supersampled): Resolve 21.1's scripting API has no keyframe calls
    (`TimelineItem.AddKeyframe` is absent), so it can't be laid as editable keyframes.
  - Ducking lives on the timeline: A1 is split around each narration line (0.15 s lead, 0.25 s tail, gaps under
    0.35 s bridged). The pieces under the voice sit at `narration_ducking_db`, joined by 6-frame "Cross Fade 0 dB"
    transitions. It's editable in Resolve like any hand-made duck.
  - Loudness: Resolve renders the master at the timeline's levels; while its sample peak is above -0.5 dBFS every
    audio item drops 6 dB and it renders again (the 24-bit master clamps overs, and no later limiter undoes that).
    The delivery encode then adds the make-up gain to `loudness_lufs` (-16) ahead of its limiter, found on
    audio-only test encodes until within 0.3 LU. Raising every item instead clipped 457 of 1010 s of one master.
  - Resolve renders `edit/<film>_master.mov` (H.264 + 24-bit PCM). ffmpeg then encodes `<film>.mp4`: x264 at
    CRF 16 (film-level `"crf"` overrides it; 23 halves the file), and AAC through a -2 dBFS limiter run at 4x
    oversampling (true peak stays under -1 dBTP).
    - Resolve's own AAC encoder puts a +24 dBFS click in the first 50 ms.
    - Fairlight's master limiter has no API.
    - The 21.1 API rejects every `VideoQuality` value, so Resolve's H.264 can't be capped (it comes out at ~14 Mbps).
  - The previous `<film>.mp4` is kept as `_iterN.mp4`. The cut list is exported to `edit/<timeline>.otio`.
  - Film `"mono_dialogue": true` lays the A1 audio of shots with on-screen speech (shot `"voices"`) as a mono fold
    (both channels = (L + R) / 2, cached in `edit/mono/`); music, audio cues, SFX and other shots stay stereo.
    Generated speech carries a slight stereo spread (H3 shot audio: L/R correlation ~0.95, side 15–18 dB under
    mid) that a listener may hear as a phasey voice. Narration is already mono (its own mono track).
  - Media is imported from content-addressed copies in `edit/media/`. Resolve doesn't notice a file rewritten in
    place, and `ReplaceClip` on the same path keeps the stale frames.
- Clips are cached in `<film>/narration/`.
- The voice-clone bake-off on one original line gave these speaker-similarity scores to the narrator's reference:
  Qwen3-TTS 0.98, ChatterBox 0.98, IndexTTS-2 0.97, F5 0.96, VibeVoice-Large 0.96. All transcribed
  correctly. The scores are close, so VibeVoice is the default for its more natural, expressive delivery,
  with IndexTTS-2 as the alternative.

## Editing tools (researched Sep 2026)
Cutting, mixing and rendering happen in **DaVinci Resolve Studio** 21.1.
- `tools/resolve_edit.py` drives it through the official scripting API (`DaVinciResolveScript`, `fusionscript.dll`).
  System Python 3.14 loads it; no extra venv is needed. When Resolve is not in the default location, set
  `RESOLVE_SCRIPT_API` (its `Developer/Scripting` folder) and `RESOLVE_SCRIPT_LIB` (its `fusionscript.dll`).
- What the API can't do: automation curves, Fairlight plugin settings, bus routing, and the master limiter.
  That's why ducking is split pieces plus cross fades, and why peak limiting happens at the AAC encode.
- Resolve has to be running, with Preferences > General > External scripting using = Local.
- For interactive editing by an agent there's also the samuelgursky DaVinci Resolve MCP server (37-tool compound
  server, its own Python venv). Register it in your MCP client config; its installer only looks under
  `C:/Program Files`, so for another install location pass `RESOLVE_SCRIPT_API`/`RESOLVE_SCRIPT_LIB` in the
  server's env. Its `scripts/doctor.py` health check needs the same env vars.
- **OpenTimelineIO**: every cut is also exported as `.otio`, which opens in Premiere or Kdenlive too.
- Replacement branches: when re-cutting already-committed footage, assemble into a new manifest, recipe list and
  timeline name rather than overwriting. Keep the original selection files, timeline and earlier cuts intact,
  run the same review, source-range and immutable-prefix checks on the branch, and verify actual rendered frames
  before adopting replacement footage. A reordered branch needs its own recipe list for chronological checks.
- Retiming on Resolve 21.1+: native `SetSpeed` with nearest-frame sampling keeps source frames unchanged and gives an
  exact integer output duration. Place retimed clips individually before their successors and keep dialogue and
  effects separate at normal speed; read back duration, sampled source endpoints and native speed. Retiming
  requires remapping effect cues to inspected contacts and reviewing the actual export: shorter duration alone does
  not establish coherent choreography. Original blocking can guide a much tighter rhythm (one fight beat went
  from 21.167 s to 6.208 s).
- Masked Klein repairs go through `tools/gpu_queue.py` on the aux card by default; route a repair to the H3 card
  when it plus its references exceeds the aux admission budget, rather than raising the budget. Every result must
  verify unchanged pixels outside its mask.
- Shot-specific mask composites (locking a prop, freezing an aftermath, clearing stray fragments) can repair a
  locked shot without regenerating its action: protect moving foreground parts, remove a fragment across its full
  visibility interval (not just its first frames), refuse existing outputs, record decoded frame counts, copy the
  original audio packets and verify their hashes. Such masks are not a general tracker and do not create or approve
  character motion; inspect playback and native Resolve cut frames, and remember a repaired preview does not update
  the cut's source ranges.
- Resolve rounds a non-frame-aligned PCM source end: preserve the complete dialogue by padding trailing silence to
  an exact frame boundary (e.g. 120159 samples + 1841 silent = 61 frames at 48 kHz) rather than trimming speech or
  weakening range checks; verify the decoded PCM prefix is unchanged.
- Delivery from the native H.264/24-bit PCM MOV master can copy its picture stream into the MP4 and record the
  audio loudness normalization and oversampled limiter. Playback, source-range checks and loudness meters are
  evidence, not subjective listening approval.

Remotion is useful only for titles. Shotstack and Descript Underlord are cloud services, and
VideoAgent/Director don't suit a fixed cut list.

## shots.json
- `stills.<id>`: `prompt`, `width`/`height`, optional `refs` (`"@hero"` = another still, or a file path),
  optional `engine` (`krea2` or `klein`). A still with no refs uses Krea-2 Turbo; a still with refs uses
  Flux.2 Klein 9B reference editing. Build character and set sheets first, then create each shot's keyframe
  from those sheets so the characters stay on-model between shots.
  - `init` + `denoise` (default 0.85) on a Klein still: img2img from that picture. Start keyframes from the greybox
    render of the shot (`"init": "blocking/s04_t0.png"`) and they keep its camera and placement; giving the set
    sheet as a ref instead made Klein copy the sheet's camera for every shot (in a sci-fi action short, reverse
    angles came out as the same forward view). Refs then carry only the characters, as single-pose sheets.
  - `control` (preferred for action): depth-locked keyframes. Krea-2 Turbo with the depth Control LoRA
    (`models/loras/krea2_depth_control_lora.safetensors`, custom node `comfyui-krea2-controlnet`) draws the frame
    from the greybox depth map (`blocking/<id>_t<sec>_depth.png`), then Klein img2img at `refine` 0.5 puts the
    characters on-model from their sheets. Klein img2img from the greybox render alone kept the rough layout but not
    poses, aim or props; H3 attaches muzzle flashes to whatever gun it drew. Z-Image-Turbo's Fun ControlNet
    (`"engine": "zimage"`) follows depth too but drew a thinner, off-model creature.
  - `still_clio_style` (top level) / `clio_style` (per still, `"none"` to disable): route the still prompt through
    the ClioStyle node (`custom_nodes/clio-style-node`, 398 named styles, e.g. `"Akira Style"`). H3 has no such node,
    so paste the same style prose into the film `style`.
- Blocking (`tools/greybox.py`, `film/<f>/blocking.json`, schema in its docstring): box greybox of the set with
  rigged mannequins (armature with IK arms and legs, a head that tracks a `look` point, guns on the hand bones, swords
  that point at a keyed tip) and spider drones on floor/walls/ceiling, keyed per shot with pose presets, `aim_r`/`aim_l`
  world points (the arm goes straight at them, so the gun points there), gore/debris/spark/muzzle-flash effects, and
  camera keys. Renders each shot's key frames (the `init` for its keyframes), a Workbench animatic `<id>_anim.mp4`, a
  top-down plan with the camera frustum and every cast key, and a `.blend` per shot, on the GPU whose name contains
  `--gpu` (default: the `aux_gpu_name` setting; pass part of your aux card's name, so it runs beside the H3 instance; the Blender path
  below is the default install):
  `"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup -P tools/greybox.py -- film/<f>/blocking.json`.
  Keep one camera axis per scene, carry bodies and wreckage into every later shot, and write the screen side of every
  character into the treatment.
  For imported characters, use `kind: "model"` with `model` pointing to a folder containing a Mixamo-rigged FBX
  (relative to `blocking.json`) and `height` in metres. Empty-handed characters can hold separate prop FBXs:
  each prop supplies `model`, `hand`, `length` in metres and a `grip` point in the prop's own coordinates
  (+X muzzle/blade, +Z gun top). The importer closes the fingers around that point. Model `look` tracks anatomical
  forward through an unweighted aim bone, preserving the imported head bind pose despite differing bone rolls.
  Using the textured render as both `init` and a reference is a geometry-faithful paint-over, not a full art-direction
  redraw: it can retain the literal mesh look. For a freer drawn scene, omit `init`, lead with the approved character
  and environment artwork, and reduce the blocking reference to a small, blurred greyscale staging diagram.
  Review the resulting anatomy and props; approximate staging deliberately gives up exact silhouette matching.
  Post-impact guides must contain the accumulated blood, damage and debris. A clean closing guide can erase effects
  generated between keys. Carry a selected generated aftermath into later shots instead of returning to a clean render.
  A background plate can remove standing actors while retaining damage and remains. To widen a framing, widen the
  actual generated aftermath (record the source-frame hash and canvas expansion) rather than chaining another
  increasingly cropped `@prev` shot. For planted characters, use actual end-frame handoffs and first/last-frame
  conditioning. Pasted prop composites (a pasted gun, an exposed limb patch) were rejected in review; drawing the
  prop with model geometry as a reference worked instead.
  Colour drift across a sequence can start before the edit: successive full-frame reference redraws recolour
  unchanged architecture, then H3 propagates the differently coloured endpoints. Trace it to the source, then fix
  it with editable temporal Fusion corrections toward the opening palette while protecting character colours and
  practical lighting. Keep previous timelines/exports, save an editable `.drp` snapshot, credit recorded sound
  effects (e.g. CC BY 4.0 attribution), and treat frame counts, meters and final-mix ASR as evidence, not
  listening approval.
  `burst` effects use rounded airborne droplets that flatten into floor splashes; sparks and solid debris retain
  their existing particle geometry.
- Turnaround sheets: one Krea-2 prompt at 2048×768, panel by panel ("From left to right: 1. FRONT VIEW: … 2. LEFT SIDE
  VIEW: exact profile, its head points to the left edge … 3. BACK VIEW: … hidden behind the body … 4. RIGHT SIDE VIEW:
  … head points to the right edge"), plus a layout paragraph: each view alone in its own quarter with wide gaps, head
  to toe with empty space above and below, nothing crossing into a neighbouring view, relaxed A-pose, weapons hanging
  straight down. A creature needs a distinct rear feature (an exhaust vent), or the back view comes out as a second
  front. `python tools/split_views.py <sheet.png>` cuts a sheet into `<stem>_front/_left/_back/_right.png` at one
  shared scale and ground line (for multi-view → 3D tools) and rejects sheets whose views touch or are cropped.
- `shots[]`: `id`, `duration` (seconds; H3 is trained on 5–15 s), `mode`, `prompt`, `audio`,
  `first_frame` / `last_frame` (a still path, or `"@prev"` for the last frame of the previous shot), `seed`.
  - `mode: "fl"` uses the fl2va model: text→video, image→video, or first+last-frame interpolation.
  - `mode: "ref"` uses the ref2va model: `subjects` (keys of top-level `subjects` → `{desc, ref}`; `ref` is one
    path or a list of several real photos of that person/thing), `voices` (keys of top-level `voices` →
    `{desc, audio}`), and `guides` (`[{t, image, desc}]`) that pin keyframes to timestamps.
    `first_frame`/`last_frame` become guides automatically. `framing` (a still path) is the soft alternative:
    the still becomes one more reference image, described as a framing/layout reference only, so likeness
    comes from the subjects' photos rather than from a generated still. At most 9 reference images per shot.
    Video-reference/control graphs can expose both a `LoadVideo` input preview and a generated video in ComfyUI
    history. The downloader accepts only files marked `type: "output"`; an input preview is never a finished render.
    Video-conditioned caches from before this fix are invalidated on their next render.
- Consistent voices: pick a clean line from any render (`speech_qa.py` writes `renders/<id>.vocals.flac`),
  trim/normalise it into `voices/<name>.wav`, then put the shots where that character speaks in `ref` mode
  with `"voices": ["<name>"]` and write "in the voice of <Audio 1>" in the `audio` line. For voice-over, also say
  the character's mouth stays closed, or H3 lip-syncs the narration. The narrator's reference can come from their
  first approved shot.
- Prompt style: H3 follows timestamped beats (`[0s-3s] …`), camera language, and an explicit audio line.
  Put dialogue in quotes in `audio`; H3 generates the voices and lip-syncs them.
- `styles` (top level) plus `"style": "<name>"` on a shot: replaces the film-wide `style` for that shot.
- `"loras": [["file.safetensors", strength], ...]` on a shot stacks extra LoRAs from `models/loras` after the turbo
  LoRA and before the sigma shift. They are part of the cache key. `"trigger": "wushu_action,"` puts trigger words at the
  very start of the prompt, ahead of the style. We tried the Jojocodex H3 LoRAs `minimax_h3_wushu_action_v5_fl2va`
  (0.5) and `minimax_h3_spatial_physics_clean_3000` (0.3). On r01 and s14b under fl4 they produced glitchy distortion
  and nonsensical combat, such as shots hitting things behind the shooter. Only s19b (the door kicked outward, seen
  from the street) came out well, so it's the only shot that keeps them. The author warns the wushu LoRA can blur at
  4 and 8 steps. The MATLOWAI motion adapter needs ComfyUI-MAINodes (a de-rope pass), which we haven't installed.
- Action that reads: the fix was staging, not LoRAs.
  - The `action` style is the film look minus "glitchy chromatic aberration / scanline flickers / smear frames", plus
    "one continuous camera setup with a clear and consistent sense of space".
  - Pick a keyframe where the shooter's gun already points at the target and nothing sits behind them. Klein often
    draws the shooter aiming sideways into a wall, so sweep seeds. A muzzle flash in the still gets animated as a
    tracer going the wrong way, so show the aim, not the shot.
  - Then describe the hit in terms of the target "straight in front of her gun", from a fixed camera.
- Narration voice drift: VibeVoice sometimes clones the wrong timbre, in roughly 1 take in 10. Score every clip with
  `voice_similarity.py` against the narrator's `voices/<name>.wav`. Anything under ~0.9 gets re-rolled with a per-line `"seed"` in
  `narration` (the cache key includes it). Check the re-rolled take with ASR too: one seed read "End of recording" as
  "it's been end of recording".
- Colour priming: H3 attaches any colour named in the prompt to faces. Writing "emerald only on machines", or even
  "no green", gave the hero and the rescuers green eyes. For shots where nobody should have machine eyes, use the
  `human` style and describe eye colour positively ("dark brown eyes"). Also name the colour of screen glow.
- Negation primes too: "no glasses" added glasses, "no green" added green. Describe what *is* there ("bare eyes").
  The `noeye` style is the film style without the eye-colour sentence, for shots with non-green machine eyes.
- Framing and scale lock: H3 tends to push into close-ups even when told "static, no push-in", and that makes
  props (a hand-held figurine) and foreground characters look huge. Setting `last_frame` to the same still as
  `first_frame` pins the composition, palette and scale for the whole clip. Text alone doesn't hold framing.
- State relative scale explicitly in the still prompt ("his head only reaches his father's chest", "a figurine no
  bigger than her hand"). Don't use a character sheet as a ref for a prop, because the sheet's close-up scale
  leaks into the keyframe. Background children stopped looking like clones of a prop once the prop's still was
  dropped from their keyframe's refs.
- `"mode": "black"`: black picture with a faint room-tone bed and no GPU work. Use it for beats carried by
  narration alone (a character losing their sight).
- `"mode": "motion_graphic"`: an animated infographic drawn frame by frame by `tools/motion_graphic.py` (PIL, no GPU),
  silent, cached on its spec, image and the renderer's code. `"graphic": {"template": ..., ...}` picks a template:
  `blister_card` (collector blister card with feature list, starburst sticker, age badge, warning strip), `stat_card`
  (trading card with filling stat bars; a fill over 1.0 bursts past the end), `bar_chart` (growing bars, counters,
  a sticker callout), `meter` (gauge with marks and a needle that can smash past the end into a verdict stamp), and
  creator-dashboard screens that replace unreliable generated monitor close-ups: `counter` (rolling subscriber count
  with milestone stickers), `line_crash` (views graph that falls to zero), `empty_feed` (a comment feed that never
  loads, with a banner) and `error_stack` (stalled copy, stacking error dialogs, the screen dies).
  Everything stays inside the 2.39:1 letterbox. `"image"` + `"cutout": true` keys a white sheet background away
  (character sheet crops work as-is). Preview without encoding: `python tools/motion_graphic.py <shots.json> --only
  mg01 --preview` → `edit/sheets/mg_mg01.png`. Fields per template are in the module docstring.
- Anime look: the r/StableDiffusion "proper anime in MiniMax H3" tips come down to flat cel-shaded
  reference frames plus visual framing guides. Adding "The target video is 2d colored anime…" to the prompt
  made no measurable difference in our A/B on n05 (same frames, no held frames), because our Klein keyframes are
  already cel-shaded. The same post suggests a latent-upscale pass (ref2va 4-step LoRA, 50% denoise) for
  full HD, which we haven't tried.

## Timing (96 GB card, 20 steps)
- H3 at 864×480, 5 s: about 80 s per shot. The first shot needs about 40 s more to load the model.
- H3 at 1344×768, 9 s: about 12–13 min per shot. Twenty shots take roughly 4 h.
- Stills: Krea-2 takes about 5–30 s, Klein with refs about 10–40 s. Switching between the image models and H3 adds
  model-load time, so run all stills first and then all shots.
- Turbo: `"turbo": true` uses each mode's default preset. You can also name a preset per shot, or set a
  film-level per-mode map such as `"turbo": {"fl": "fl4", "ref": "ref8"}`.
  Presets (all in `models/loras`):
  - `fl8`: lightx2v 8-step v1.0.
  - `fl8_768`: 8-step 768p. It invented glasses in the A/B test.
  - `fl4`: lightx2v 4-step v1.2 768p with video/audio shift 6/3. v1.2 is the audio-fix release.
  - `ref4`: Comfy-Org 4-step v0.1.
  - `ref8`: lightx2v 8-step 768p.
- A/B on s05 (1344×768, 7 s): fl8 took 256 s, fl4 at 4 steps 140 s, fl4 at 6 steps 160 s, fl8_768 210 s.
  fl4 at 4 steps matched fl8 on picture and transcribed identically, so v2 uses it for new shots. Our
  ComfyUI has native AV sampling, which fixes the old 4-step audio distortion.

## MCP
The official **Comfy-Org `comfy-mcp`** server (installed in its own venv, driving your local ComfyUI install through
comfy-cli) can be registered in your MCP client config (e.g. a project `.mcp.json` for Claude Code or omp). It adds
39 tools: `run_workflow`, `job`, `fetch_outputs`,
`search_templates`, `fetch_template`, `set_workflow_slot`, `vary_workflow`, `system_stats`, `free_memory`,
`upload_file`, `download_model`, and others. An agent can use it for interactive work, such as
iterating on a single shot or finding templates and models. The Python tools above handle the
repeatable batch pipeline.

## Known issues
- If ComfyUI has been running a long time with other large models loaded (e.g. Trellis), H3 can fail with
  `MemoryError: VBAR allocation failed`. comfy-aimdo can't reserve GPU virtual address space. `/free` doesn't
  fix it. Restart ComfyUI with `--disable-dynamic-vram`.
- `fl` shots invent a new voice every time. Any shot where a recurring character speaks needs `ref` mode with a
  reference voice (see "Consistent voices"). The first-frame keyframe then only guides frame 0, so camera
  instructions matter more.
- H3 sometimes adds a few invented words before or after a line. `speech_qa.py` catches this as low
  precision. Telling it the character "says exactly one line and nothing else" and adding "No other speech"
  usually fixes it.
