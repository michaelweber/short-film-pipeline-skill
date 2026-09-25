---
name: short-film-pipeline
description: Turn a script or story into a short AI film with local ComfyUI (Krea-2/Flux.2 Klein stills, MiniMax H3 shots), cloned narration, a generated score, speech QA, and a DaVinci Resolve cut
---

# Short film pipeline

Style-agnostic workflow for `film/<name>/` projects (photoreal, anime, motion graphics, documentary parody).
The scripts live in this skill's `tools/` directory; below, `tools/x.py` means `<skill dir>/tools/x.py`, run from
the project root with the shot list as the first argument. `reference.md` documents every shots.json field.
Companion files: `prompting.md` (prompt rules), `qa.md` (gates), `troubleshooting.md`, `templates/`.

```
treatment.md ─► shots.json ─► keyframes.py ─► stills/*.png ─► sheet.py --stills (review)
                                   │
                    h3_render.py ─► renders/*.mp4 ─► speech_qa.py / voice_similarity.py / sheet.py (review)
                                   │                  └► mouth_hold.py ("hold"), "out" trims (edit-only fixes)
                                   │
                    narrate.py ─► narration/*.wav     music.py ─► music/score_*.flac
                                   │                     │
                                   └──► resolve_edit.py ◄┘ ─► <film>.mp4
```

## 1. Intake
- Pin down: source material, target runtime, visual style, every voice (who speaks, cloned from what).
- Real people's likeness or voice (including a public figure or well-known character for a parody): the user
  decides and is responsible for the rights. Keep parody clearly framed as parody, and check before monetising.
- Real people: collect 2–3 real photos (the subject `ref` list; generated stills are framing only) and a clean
  10–20 s voice clip. The clip goes to H3 as `ref_audio` on every shot they speak in, and carries their voice.
- Copyrighted source: write own-words summaries into the treatment; never paste the text into prompts.
- Never name existing shows, films, studios, franchises or brands in prompts; describe the look ("flat cel
  shading, bold line art", "a 20-metre white-and-red giant robot statue with a horned head crest").
- Existing footage (channel intro, stock): copy into `media/` and use `"mode": "clip"`.
- Voice references from a video: `python tools/voice_ref.py <url> --out voices/<name>.wav --keep "phrase|phrase"`.
  Keep-phrase spans often include the other speaker. Score every span in `<out>_spans/` with
  `voice_similarity.py` against one span you know is pure. Keep only spans ≥ 0.95 and rerun with `--use`.
- Voice references must be dry speech (interview, radio, podcast), never a trailer or narration with a score under
  it: the TTS clones the music bed along with the voice and puts it under every line, and vocal separation
  afterwards only lowers it. Check the reference's spectrogram: the pauses must be plain room tone, with no
  sustained horizontal bands. A clean 10 s clip beats a longer one with music.
- Vet every voice clip: score its first and second halves against each other. Below 0.85 can mean two
  speakers, but an expressive single speaker can score as low as ~0.75. Ask the user before cloning it.
- Product/prop photos: after download, rewrite each prop description to match the photo (current products
  differ from memory). Text that contradicts the reference image wins half the time. Blank any readable text in
  a reference photo (shirt logos, number plates), or the image model copies it.

## 2. Treatment
- Copy `templates/treatment.md` to `film/<name>/treatment.md`: logline, characters, beats, script.
- One idea per shot, 5–10 s (H3 trains on 5–15 s). Dialogue shot length = 1 s lead + words / 2.6 wps, min 5 s.
- H3 often spreads a short line across the whole 5 s shot, or pads it with invented words; plan to trim with `out`.
  If there's a runtime floor, write ~10 % more script than you need.
- Decide what a first-time viewer must understand in the first 20 s and stage it on screen, not in narration.
- A local or interviewee who answers the host: shoot them as an over-the-shoulder reverse angle (the host's back
  soft in the foreground), not as a solo shot in the host's framing, or they seem to replace the host.
- Documentary style: most shots should be generated footage of the lead acting or talking. Keep Ken Burns moves
  (`"mode": "still"` + `"kenburns"`) for standalone photographs and flashbacks, and turn a key photo into a
  talking shot (`first_frame` = the photo) instead of zooming on it. Archival TV clips: `"pillarbox": 1.3333`.

## 3. Style block
- `style`: one paragraph with medium, camera, light, texture. Prepended to every shot prompt.
- `styles`: named variants (`human`, `action`, `card`) selected per shot with `"style": "<name>"`.
- `still_style` (optional): the same look for keyframes.
- Never name equipment in a style block: "large-format camera", "drone" or "gimbal" put cameras and drones in the
  frame. Describe the result ("widescreen anamorphic look, shallow depth of field, sweeping aerial views").

## 4. Stills
- Order: character sheets and set first (Krea-2, no refs), then keyframes (Klein, refs `@sheet`, `@set`, photos).
- Pin every still's `seed` explicitly so edits elsewhere never re-roll it.
- `python tools/keyframes.py film/<name>/shots.json`, then
  `python tools/sheet.py film/<name>/shots.json --stills --tag stills` and look at every sheet.
- Failing still: sweep 3 temp ids with seed +1..+3, adopt the best (copy prompt/seed back), delete temps.
- Never render video from an unreviewed still.
- Look at every still at full size, not just the contact sheet: hands, merged characters, stray hands on props,
  and scale errors hide at thumbnail size (fixes in prompting.md).
- Unusual real-world objects (regional vehicles, folk toys): give Klein 1–2 real photos as refs and say the person
  in the photo is only a reference for the object. Text alone gets you the nearest common object (a kick scooter,
  a bicycle).
- Well-known characters: likeness from 2–4 clean pictures of the character; costume from a separate photo of the
  outfit, with "only his outfit comes from picture N; his face and head are his own". Then build a costumed
  character sheet and use it as a ref for the keyframes (prompting.md).
- If the user hand-edits a still, don't touch its prompt, refs or seed afterwards: that re-renders it over their
  edit. Re-render the shots that use it.

## 5. Render
- Start as a supervised background process, never a blocking shell call with a timeout.
- One GPU job at a time (stills, H3, TTS, ASR, music all share the card). Wait for exit before the next. Queuing
  several jobs at once can leave TTS models holding VRAM outside ComfyUI's manager, and H3 then crawls at minutes
  per step; restart ComfyUI to recover. VibeVoice does this on its own after ~10 lines in one ComfyUI session:
  restart ComfyUI after a narration batch and before the next H3 render.
- Default turbo `{"fl": "fl4", "ref": "ref8"}`. Recurring speakers use `ref` mode with `voices`.
- Budget at 1344×768: `ref8` takes ~1.5–10 min per shot, `fl4` ~1–2 min, so ~30 shots ≈ 1.5–2.5 h.
- After any change of approach (refs, framing, prompt pattern), render ONE representative shot and check it
  (framing, mouths, words) before queueing the list.
- Re-roll one shot: change its `seed` and rerun with `--only <id>` (the effective seed is in the cache key).

## 6. QA gates
Run every gate in `qa.md`. A shot passes only when all its gates pass; after 3 re-rolls list it as a known issue.

## 7. Narration
- `python tools/narrate.py film/<name>/shots.json --plan` first: no narration over on-screen dialogue.
- Engines: `vibevoice` (TTS), `indextts2`, or `h3` (the narrator reads the line on camera in H3 and only the audio
  is kept; matches the on-screen voice best when the narrator is also a character).
- Long takes read more naturally than one-line takes: render several lines per take (H3: ≤ 15 s per take;
  VibeVoice: the whole script), cut each line out with `narrate.split_take()`, pick the best piece per line
  across seeds, and point the line's `"file"` at it.
- Cut generously. Trimming at the quietest point clips soft word endings ("babies" → "baby"); overlaps are cheap
  to fix in the edit, missing syllables are not.
- Score every clip with `voice_similarity.py` and ASR it; re-roll low scorers with a per-line `seed`.
- `"narration_isolate": true` keeps only the voice of every clip (vocal stem); `"narration_tempo"` (film) or a
  line's `"tempo"` speeds clips up with pitch kept; a line's `"voice"` reads it in another cloned voice (a
  character's letter). All three are cached derivatives, so none re-renders a take.

## 8. Score
- `python tools/music.py film/<name>/shots.json` renders an instrumental score with YuE2 and prints its length,
  the hit onsets (big loudness rises) and a vocal check. `--seed N` renders candidates; adopt one with `"seed"`.
- A supplied track works too: `"music": {..., "file": "music/track.mp3"}`.
- Align a hit with the title card: `"in"` = hit time − title-card start (score starts at 0 in the cut).

## 9. Cut
- `python tools/resolve_edit.py film/<name>/shots.json` (Resolve Studio running, External scripting = Local).
- The previous mp4 is archived automatically as `_iterN`; the cut list goes to `edit/*.otio`.
- Edit-only shot fields are applied here and never re-render a shot: `titles`, `zoom` (punch-in), `hold` (keeps a
  silent character's mouth shut; `tools/mouth_hold.py`), `out` (ends the shot early) and `bed`.
- Sound: shots without a spoken line should usually be `"bed": "none"` (music and narration only). Dialogue shots
  play until ~0.8 s after the line, or in full.
- Titles: grab a frame at every title and use `"align": "left"` wherever a rank badge covers a face.
- File size: film-level `"crf"` (default 16 ≈ 4 Mbps). 23 halves the file at SSIM 0.99, which is right for
  YouTube uploads.
- Finish with a whole-film consistency sweep (`sheet.py --cols 5`), ASR every line in the delivered mix, and
  report exactly what's still broken.
