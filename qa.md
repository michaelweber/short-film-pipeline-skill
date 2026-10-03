# QA gates

Apply gates to the media/operation actually used: existing footage does not require generated-still,
H3-cache or rig checks. All deliveries require continuity, pacing, speech/mix and exported playback review.

`PY_TORCH` = a Python with torch/transformers (the `py_torch` setting in `tools/pipeline_settings.py`: env var
`PY_TORCH` or `tools/local_settings.json`).

| Gate | Command | Pass | On fail |
|---|---|---|---|
| Dialogue words | `python tools/speech_qa.py film/<f>/shots.json`, then listen against the source/script | recall ≥ 0.9 (precision ≥ 0.8) is a screen only; approved/source lines remain exact and clear | retain a clean source take or re-roll; do not shorten an approved line to satisfy ASR |
| Narration timing | `python tools/narrate.py film/<f>/shots.json --plan` | no `!!` except over non-speech | move `at`, shorten the line, or move it to a quiet shot |
| Voice identity | `$PY_TORCH tools/voice_similarity.py voices/<v>.wav renders/<id>.vocals.flac narration/*.wav` | ≥ 0.92 narration clips, ≥ 0.85 on-screen stems | re-roll that clip/shot `seed` |
| Re-rolled takes | `speech_qa.transcribe(Path(clip), Path("tmp.flac"))` | exact words | re-roll again |
| Stills | `python tools/sheet.py film/<f>/shots.json --stills --tag stills` | one of each character, right scale, no garbled text | seed sweep (+1..+3) |
| Action shots | `python tools/sheet.py film/<f>/shots.json --ids <ids> --cols 5 --tag action` (≈1 fps tile) | action reads left to right, no reversed physics | restage the still, fixed camera |
| Silent mouths | `python tools/sheet.py film/<f>/shots.json --ids <id> --cols 16 --crop <box around the silent face>` | the non-speaking character's mouth never opens (check the cut, with holds applied) | `tools/mouth_hold.py` → `"hold"` on the shot (see prompting.md) |
| Invented words | `speech_qa.py` heard text; then ASR each `narrate.dialogue_spans()` span of the vocal stem up to the cut end | nothing before or after the scripted line inside the cut | `"out"` at the line's last word + ~0.4 s, and before the next speech span starts: a fixed "+0.8 s" once let 0.6 s of babble into a cut; `"in"` after words invented before it |
| Internal cuts | `ffmpeg -i renders/<id>.mp4 -vf scdet=threshold=18 -an -f null -` inside `[in, out)`, plus frame-0 vs frame-2.2 s difference (64×36 grey, mean-removed; > ~45 flagged a photo opening or dissolve in a full episode) | one continuous shot | pin `first_frame`, sheet-only subject pictures, no room description on close-ups (prompting.md); trim only when the cut falls outside the line |
| Narration words | ASR every narration clip, compare with the line | recall ≥ 0.9 and precision ≥ 0.85 (H3 takes often open with 1–2 s of garbage or the voice reference's own words) | re-roll with a per-line `seed` |
| Score vocals | `music.py` vocal check, then the vocal stem's loudness (`ebur128` on `score_*.vocals.flac`) | the ASR check hallucinates text on instrumentals ("the new york times…"); a stem ≥ 30 LU below the mix is instrumental | ≤ ~5 LU below the mix is a real vocal (or a lead instrument): re-roll the seed |
| Hands | `python tools/sheet.py film/<f>/shots.json --cols 5 --tag hands` | five fingers on every visible hand | re-roll `seed`; restage the keyframe with hands resting or holding a prop |
| Cache | `python tools/h3_render.py film/<f>/shots.json --dry-run` | 0 stale (every line `= … up to date`) | render the stale shots |
| Titles | frame grab at every title's `at + 1 s` in the cut (`ffmpeg -ss <t> -i <f>.mp4 -frames:v 1 …`) | no badge or lower third over a face; the right text | `"align": "left"` on the rank badge |
| Delivery loudness | `ffmpeg -nostats -i film/<f>/<f>.mp4 -af ebur128=peak=true -f null -` | within ±0.5 LU of `loudness_lufs`, true peak ≤ −1 dBTP | rerun `resolve_edit.py` |
| Master clipping | `ffmpeg -i film/<f>/edit/<f>_master.mov -map 0:a:0 -af volumedetect -f null -` (`resolve_edit.py` stops on it) | `max_volume` < −0.5 dB: the 24-bit master clamps overs, and no delivery limiter undoes that distortion | lower the hot track(s) in the timeline; never reach loudness by raising timeline gain (the delivery encode's make-up gain does that) |
| Delivery size | `ffprobe` the mp4 | fits the destination (YouTube: `"crf": 23`, ~60 MB for 4 min at 1344×768) | set film `"crf"`, re-deliver |

Checking frames at scale: use one ffmpeg call per shot (`-vf "fps=4,crop=…,scale=-2:100,tile=Nx1"`), not a
Python loop of per-frame ffmpeg seeks. The loops timed out the eval kernel twice. For long checks (ASR sweeps,
previews of every shot), write a throwaway script and run it with bash.

## Editing and reference-repair gates

- **Fizgig repairs:** compare the approved source, raw candidate and reviewed composite at full size.
  A red-painted reference is not an executable mask. Verify unchanged outside-mask pixels, registration,
  edge blending, anatomy, identity, occlusion and contact/reflection before adopting a repair. See
  [fizgig.md](fizgig.md); a valid still does not approve its generated video.
- **Pacing and causality:** play the actual cut at delivery speed. Remove dead holds without losing
  anticipation, contact, response or recovery. Entrances must show a motivated route; reject teleporting
  cast, false flashes and sounds without corresponding actions. Shorter duration alone is not approval.
- **Retiming:** preserve approved dialogue at normal speed on independent tracks. Step the inspected
  contacts in the retimed export and realign effects there; check cut boundaries and screen direction.
  Record source fps/ranges, speed, exact integer output frames and final timeline cue positions.
- **Structure and extent:** read back native picture/audio items, gaps/overlaps and source ranges.
  Compare expected frame count/runtime to the actual rendered file, not just render-job completion.
  Recheck the final encoded audio/video extent and decoded true peak after delivery processing.

For explicitly requested rigged/stylized animation, also apply
[the source/generated sword gates](art-animation.md#sword-blocking-and-generated-fight-gates).
Do not make rig qualification a prerequisite for ordinary edits or reference-guided stills.


## Sequence, sound and delivery review

- **Continuity:** inspect every keyframe edit, shot boundary and action transition at full size. Match
  camera scale, floor positions, occlusion, limb/prop design and accumulated aftermath to approved frames.
  For fast shots, step adjacent frames around lift, flash, recoil and target hit; a ≈1 fps sheet cannot
  prove timing. Reject texture/style seams, popping patches, duplicate flashes/bangs or delayed reactions.
- **Palette (mandatory):** compare the first and last frames of the entire sequence, each shot's
  endpoints and intermediate edited keyframes against the approved starting frame. Check the same neutral
  structural surfaces under comparable lighting, preserving motivated light and intentional object colours.
  E.g. in a sci-fi corridor sequence, remove progressive green contamination from the walls, not green skin
  or olive clothes.
  Compare corresponding source frames, Resolve viewer frames and exported frames to locate the first
  divergence before changing grades, colour management or encoding; do not assume a codec fault.
  **Before H3:** compare all first/last guides and intermediate edits to the approved palette, including
  materials that should be unchanged. Reject mismatched endpoints before rendering: H3 can propagate
  their colour difference. A "change only the hand" prompt is not a local-edit guarantee; inspect whether
  the actual workflow has a spatial preservation mask or resynthesizes the full frame. An `init` image
  alone does not freeze unedited pixels; reference conditioning and ordinary img2img can both shift colour.
- **Sound selection and mix:** verify source licensing/attribution and waveform onset/tail, then audition
  the source solo, its timeline track solo and the final mixed export at the same playback level. Each
  visible pistol discharge must read unmistakably as a real recorded muzzle report; ricochets/armor hits
  supplement it, never replace or mask it. Keep the cannon heavier and distinct without an extra bang.
  Door slide/motor/latch must sound grounded and follow visible travel/start/stop, not an unrelated pulse
  or whoosh. Adjust competing cue gains, timing and processing in Resolve, then listen again in context.
- **Speech:** compare the clean source dialogue to the mixed cut by ear, including the last word and
  surrounding silence; no masking, clipped words, filler or invented speech. ASR can assist, not approve.
- **Delivery:** inspect the editable Resolve picture cuts and separate sound tracks, then preview actual
  exported frames and play the entire export with sound. Meters, waveform peaks, ASR and successful tool
  calls cannot prove subjective mix quality or visual consistency. Retain older approved versions and
  exact source-frame/cue recipes; describe user preferences, observed failures and unverified diagnoses
  separately. A small representative pass must pass before broader regeneration.
