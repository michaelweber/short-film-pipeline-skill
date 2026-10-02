---
name: short-film-pipeline
description: Make and revise short films with reference-guided Fizgig image edits, existing footage or ordinary generated shots, and editable DaVinci Resolve cuts; bespoke art-style animation is optional
---

# Short film pipeline

Editing-first workflow for `film/<name>/` projects in this repo, independent of visual medium.
Use normal cuts, trims, source performances, sound editing and reference-guided repairs before
commissioning animation. Do not turn an ordinary film/edit request into a 3D asset-and-rig project.
Tools live in `tools/`; `reference.md` documents every shots.json field and native edit recipes.

Read [fizgig.md](fizgig.md) for the preferred image-editing/reference pattern, [prompting.md](prompting.md)
for ordinary generated-shot prompts, and [qa.md](qa.md) for applicable gates. [troubleshooting.md](troubleshooting.md)
and `templates/` support those flows. Load [art-animation.md](art-animation.md) only for a requested bespoke
art style that needs animation construction, or an explicit 3D blocking/motion-transfer task; it routes
to `models.md` when rigs are actually needed. An action scene alone does not trigger that reference.

```
treatment + references → shots.json / source selections
                              ├→ existing footage
                              └→ Fizgig/reference stills → ordinary H3 shots, only if needed
                                             ↓
                        reviewed sources + dialogue/foley
                                             ↓
                 versioned Resolve cut → pacing/mix pass → verified export
```

## 1. Intake
- Pin down: source material, target runtime, visual style, every voice (who speaks, cloned from what).
- Real people's likeness or voice: the user decides whether to clone it (e.g. a public figure's voice for a parody).
  Keep such films clearly framed as parody, and check rights before monetising.
- Real people: collect 2–3 real photos (the subject `ref` list; generated stills are framing only) and a clean
  10–20 s voice clip. The clip goes to H3 as `ref_audio` on every shot they speak in, and carries their voice.
- Copyrighted source: write own-words summaries into the treatment; never paste the text into prompts.
- Never name existing shows, films or studios in prompts; describe the look ("flat cel shading, bold line art").
- Existing footage (channel intro, stock): copy into `media/` and use `"mode": "clip"`.
- Voice references from a video: `python tools/voice_ref.py <url> --out voices/<name>.wav --keep "phrase|phrase"`.
  Keep-phrase spans often include the other speaker. Score every span in `<out>_spans/` with
  `voice_similarity.py` against one span you know is pure. Keep only spans ≥ 0.95 and rerun with `--use`.
- Vet every voice clip: score its first and second halves against each other. Below 0.85 can mean two
  speakers. Ask the user before cloning it (one host's clip scored 0.76 and was all him).
- Product/prop photos: after download, rewrite each prop description to match the photo (current products
  differ from memory). Text that contradicts the reference image wins half the time.

## 2. Treatment
- Copy `templates/treatment.md` to `film/<name>/treatment.md`: logline, characters, beats, script.
- One idea per shot; plan the final cut around readable beats, not generator minimum duration.
  For H3, generate 5–10 s shots (training range 5–15 s); dialogue allowance = 1 s lead + words / 2.6 wps,
  min 5 s. Short lines can acquire invented filler; trim after the approved line, not through it.
  Generated duration is not cut duration. If there is a runtime floor, plan additional useful beats.
- Decide what a first-time viewer must understand in the first 20 s and stage it on screen, not in narration.

## 3. References and image editing
- Prefer [Fizgig's editing/reference flow](fizgig.md) for still repairs and reference-guided generation,
  whether photoreal or illustrated. Separate scene/camera, identity, detail geometry and appearance roles.
- Start from approved photos/frames. For a local defect, use a marked scene copy plus the relevant
  unmarked identity/prop reference; keep the untouched original and a separately reviewed composite mask.
- Red paint and “preserve everything else” guide the model; they do not freeze pixels. Verify registration,
  contacts, occlusion, palette and unchanged surroundings in the actual result before adopting it.
- Keep one approved scene/palette anchor and consistent prop/character references across the sequence.
  Do not re-invent each keyframe independently or replace a whole frame to solve a tiny defect.
- For generated shots, `style` describes medium/camera/light/texture and `styles` selects named variants.
  `still_style` controls Klein keyframes; keep it aligned with video `style`. Custom animation construction
  lives in [art-animation.md](art-animation.md), not the default path.

## 4. Stills
- Use existing approved images or the Fizgig pattern first. Only make new sheets/sets when the shot needs
  invented designs; the optional art reference retains the Krea-2 sheets → Klein keyframe route.
- Pin every still's `seed` explicitly so edits elsewhere never re-roll it.
- For the existing Krea-2/Klein route, run `python tools/keyframes.py film/<name>/shots.json`.
  Review selected stills with `python tools/sheet.py film/<name>/shots.json --stills --tag stills`;
  Fizgig candidates come from their saved Comfy workflow, not an implicit `keyframes.py` mode.
- Failing still: repair from the approved source/reference before a bounded seed sweep. The existing
  `keyframes.py` route can sweep 3 temp ids with seed +1..+3; adopt the best prompt/seed and remove temps.
- Never render video from an unreviewed still.
- Look at every still at full size, not just the contact sheet: hands, merged characters and scale errors hide
  at thumbnail size. Klein turned a puppet into a hybrid with the potato-figure prop beside it, and shrank an
  adult in a toddler car into a child (fixes in prompting.md).
- Keyframes with two or more characters (over-the-shoulder reverses, two-shots): stage them with the Fizgig
  H3 still graph, one picture per role (location crop, each character's single-pose sheet crop, prop still), not
  Klein. Klein invented a bearded stranger for the host's back and could not open a nesting doll into halves;
  Fizgig got both right (fizgig.md, "Multi-character keyframes").
- Costume variants (tuxedo, helmet) must be written into the keyframe prompt: Klein copies the outfit of the
  identity photos and ignores the subject `desc`.
- Screens, dashboards, counters, error dialogs and infographics: use `"mode": "motion_graphic"`
  (`tools/motion_graphic.py`, no GPU), not generated monitor close-ups. H3 toured the room or cut to a webcam face
  in most monitor shots; the graphics are exact, legible and cost seconds.

## 5. Render
- Generate only missing/rejected footage; do not re-render a good performance to fix an editable hold.
  Read Comfy `server_info` first, validate the workflow and run long jobs asynchronously:
  `run_workflow(wait=false)` → `job(action="wait")` → `fetch_outputs`. For the repo's H3 batch route, use
  a supervised process: `hub start name=render application=python args=["-u","tools/h3_render.py","film/<name>/shots.json"]`.
- Two ComfyUI instances, one per GPU role. The **H3 card** → `:8188` (default) for H3 video only, one H3 job at a
  time. The **aux card** → `:8189` (default; a `hub` process such as `comfy-aux`, pinned with
  `CUDA_VISIBLE_DEVICES=<aux GPU UUID>`, its own `--output-directory` and `--database-url`) for stills, depth,
  TTS/ASR/CLAP, and Blender Cycles (OptiX). Pin by UUID: CUDA order differs from `nvidia-smi`. GPU UUIDs and the
  torch-capable Python come from `tools/pipeline_settings.py` (`h3_gpu_uuid`, `aux_gpu_uuid`, `py_torch`): set
  them as env vars (`H3_GPU_UUID`, …) or in the untracked `tools/local_settings.json`.
- Aux VRAM budget ≤ 60 % when that card also drives the desktop/remote access: run its ComfyUI with
  `--reserve-vram 16`; `tools/gpu_queue.py` admits aux jobs and Blender leases only under budget and frees models
  after aux jobs. Jobs too big for it go to `--card h3`. H3 itself stays on the H3 card (on a smaller aux card it
  swapped: 2.4-2.8× slower).
- `tools/gpu_queue.py submit WF.api.json --card h3|aux --wait` submits without routing through the lead agent;
  `status` shows both queues, GPUs, leases; every job is logged to `ledger.jsonl` in the lease directory
  (`gpu_lease_dir` setting, default `.gpu_leases/`). Outputs carry
  their node id: ComfyUI also returns echoed input videos, in unstable order.
- Turbo presets `{"fl": "fl4", "ref": "ref8"}` are for plain FL/ref shots and drafts. With a ControlNet control
  video (white contour rim) or a reference video (ignored), render non-turbo. `h3_render.py` warns.
- Never stop a process you didn't launch. The user works on this machine too: a worker killed the user's open
  GUI Blender ("Unsaved") while cleaning up its own strays. Stop only your own hub names or printed PIDs.
- Drafts: `--tier draft` renders `draft_seeds` (default 3) takes at 512p with turbo into `renders/_draft/`.
  Seeds do not transfer across resolutions (noise is drawn per latent shape); prompt, guides, refs and
  timing do. `"audio_lock": true` pins dialogue through the frame-0 guide for draft → final timing.
  Specialized motion-video draft settings belong in [art-animation.md](art-animation.md).
- Budget at 1344×768: `ref8` takes ~1.5–10 min per shot (a 12 s line took 9.7 min), so ~30 shots ≈ 2–2.5 h.
  Anything whose picture is discarded (narration takes) renders small: see §7.
- After any change of approach (refs, framing, prompt pattern, sound or colour), review one small
  representative pass before broader regeneration; include motion, dialogue, mixed audio and palette.
- Re-roll one shot: change its `seed` and rerun with `--only <id>` (the effective seed is in the cache key).
- Ref shots open on, or cut mid-shot to, a subject picture that has a background (a film still, a real photo in a
  room), and dissolve in from it. Give non-host characters sheet crops on white only, and pin `first_frame` to the
  keyframe on every ref shot (it counts as a reference: subjects + framing + guide ≤ 9 pictures).
- Budget with contention: while narration takes or fix passes share the H3 card, a 5 s shot took 5–9 min
  instead of ~3; a 222-shot episode is an overnight job. Run segments as supervised processes, and QA each finished
  segment while the next renders.

## 6. QA gates
Run the applicable gates in `qa.md`; special rig/fight gates are conditional, not default prerequisites.
After 3 unsuccessful re-rolls report the unresolved issue and repair/restage it; never label it approved.

## 7. Narration
- `python tools/narrate.py film/<name>/shots.json --plan` first: no narration over on-screen dialogue.
- **Write every H3 speech prompt in MiniMax's official format** (`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` and
  `_base_en.md` in the MiniMaxAI/MiniMax-H3 HF repo): spoken words only inside `<d>[English] …</d>` ending in `.`,
  `?` or `!`; speaker IDs `(S1)`; `<Audio 1> is the voice-timbre reference for <Subject 1> (S1)` with retention
  `<Audio 1>: reference - … without copying the original signal`; voice-over as `says in an off-screen voiceover:
  <d>…</d> while his lips remain completely closed`. Free-text lines ("says exactly one line: \"…\"") made H3 read
  the prompt itself aloud before the line: the style paragraph ("premium travel documentary cinema…"), the
  host's name, "that is the only speech". In a 36-take A/B (a documentary-parody episode) the old format put the
  line at 3.5–4.6 s after babble in 2/3 of takes; the official format started it at 0–0.4 s in 11 of 12.
  Film `"narration_h3_guide": true` builds narration takes this way; `compose_h3_ref` (`"prompt_format":
  "h3_ref"`) is the shot equivalent — put dialogue in the body with `<d>`, never in `overall_soundscape`.
- **No spare seconds in a speech take.** H3 fills silence with babble or a garbled replay of the voice
  reference (community reports agree); size the take to the words (0.8 s + words / 2.6), not with headroom.
- **Clean voice references.** A noisy or mixed reference is the other half of the problem: the user's cleaned
  host clip (vocal-separated with `voice_ref.py`, halves 0.95) scored 0.95–0.97 on takes vs 0.84–0.91 for the old
  clip. Ask the user for a clean recording early and use it for every input (narration and on-screen lines).
  Check the reference's spectrum too: H3 copies its tone. A denoised clip band-limited to ~3 kHz (99 % rolloff;
  ~5 % of energy above 1 kHz) made every take tinny, and EQ on it barely helped. A full-band, hiss-free recording
  fixed it: same word accuracy, 11–18 % energy above 1 kHz, similarity to the host's original voice 0.87 vs 0.82
  (10-line A/B). The cut/normalise pipeline does not change the spectrum; non-turbo 20-step
  takes were no fuller and garbled more words than `ref8`. With fast takes and cutting, redoing all narration
  for a better reference is ~20 min for 70 lines: worth it.
- **Lists invite inserted babble.** "A lamp with nine wicks, a rock with a song, and a whistle from a war"
  got an invented extra item mid-line on 18 of 18 seeds (two ways of splitting it); one line per item passed on
  the first seed. When a line fails the same way on every seed, rewrite or split it instead of re-rolling.
  Multi-line shots need per-line timing edits (key them per line, e.g. `vo_at_<i>`/`vo_tempo_<i>`).
- H3 narration takes: set film `"narration_h3_short_side": 256`. Only the take's audio is kept, so its picture is
  denoised at 448×256: 11–20 s per take instead of 3–7 min at 1344×768, and small takes jump the H3 queue.
  Lines that already have a full-size take keep it. H3 takes need the H3 card (not the aux card).
- Still expect some babble: `narrate.split_take` cuts each line from word timestamps (`tools/asr_words.py`, local
  Whisper `whisper-small.en` on the aux card, a persistent worker: ~8 s start once, then 0.4–0.7 s per take) and pads only across
  quiet audio. It falls back to the ComfyUI ASR dip search (~5 s per call, ~15 calls per take) when the line is not
  found in the words. Babble *inside* a line can't be cut: ASR every clip and re-roll anything under
  recall/precision 0.9 (up to 8 seeds, keep the best, voice similarity breaks ties).
- **Gate with a strong, verbatim recogniser.** Whisper tidies speech into the sentence it expects: small.en passed
  "no warranting, no warning", and even large-v3's default decode passed "no traveller has, traveller has
  ever slept", "nest -nesting, uh, nesting" and "the train only stops the -the train only stops". The user heard
  them all. `tools/asr_words.py` (large-v3, ~2.5 s per take) also decodes with a disfluent prompt ("Um, uh… I-I
  mean, the, the thing"), which makes Whisper write stutters, repeats, fillers and gibberish out.
  `narrate.check_clip` passes a clip only with: recall ≥ 0.9; **no extra words** in the verbatim decode (a
  precision of 0.9 still let 2-word repeats through) and no cut-off fragment ("expla -"); the first and last
  scripted words heard ("babies" heard as "bait" is a clipped ending and costs only 0.1 recall); no word holding
  > 0.65 s + 0.08 s/letter of *voiced* audio (Whisper folds babble into one word's span: "twice" held 1.44 s); and
  < 0.1 s of voiced audio after a pause outside the words.
- **Cut at real pauses; Whisper only picks which pause.** Whisper's word edges can't place a cut: it ends
  drawn-out words early ("expla|nation"), labels a final "-st"/"-ts" as the next (babble) word ("heading ea|st",
  "giant|s", "month|s"), starts words inside the pause before them, and completes a clipped word from context
  ("then fifty" heard as "fifty thousand"). The user heard every one of these. `narrate.split_take` anchors inside
  the first and last words (the outermost point within 6 dB of the span's peak) and cuts 60 ms into the first
  silence of ≥ 120 ms beyond them (`_pause_cut`; closures inside a word are 40–60 ms). The gate checks the cut on
  the *take* (`.cutv` sidecar `end_db` < −30 dB), not the finished clip, because no transcript reveals a clipped
  ending. Re-cutting from saved takes (`CUT_VERSION`) fixed most clips with no H3 time.
  Align on *every* spoken token: Whisper's word tokens split "50,000" into "50" + ",000" and write "100%", and
  matching only each word's first token dropped lines below the match threshold into the old dip search, which
  cut mid-word ("fifty thousa-", "one hundred per-"); a fallback cut now fails the gate (`.cutv` `aligned`).
  With pause-based cuts, give takes ~1 s of headroom (`0.8 + 1.0 + words / 2.6`): a tight take ended mid-word on
  slow reads, and babble in the spare second sits after a pause and is cut away.
- **Same rules for on-screen dialogue trims.** Dialogue `in`/`out` trims set from ASR timings cut lines short
  ("Wow, that looks so hard" at `out` 4.0, "That was the hardest landing" missing "That") or kept gibberish
  before a line. Check every trim's level on the vocal stem (mid-sound = wrong), recut the scripted line from the
  stem at its pauses, and where babble is glued to the line, cut at the deepest dip between them (a throwaway
  sweep: RMS of the stem in 10 ms frames between the line's ASR edges and the babble, cut at the minimum).
  A reused clip shortened on purpose goes back to its previous pause.
- ASR every re-rolled take: a new seed can change the words.
- **Direct the read.** One "earnest narrator" tone for 70 lines sounds flat. Give each line a `"delivery"` that
  follows the story (e.g. excited/authoritative premise and rise → slow, solemn downfall → quiet turn → rising
  resolve → hype; a hushed wildlife-documentary narrator for a nature segment). H3 follows it: solemn lines came
  out at 1.2–1.6 words/s vs 2.1–2.6 for hype, with the same word accuracy.

## 7b. Music and sound effects
- YuE2 (the score model) is Chinese-trained and drifts to C-drama/pop-ballad cues on generic tags ("majestic",
  verse/chorus, 100 BPM): the user called a wildlife-segment cue "a Chinese soap opera". Name the tradition and the
  instruments, exclude the drift ("western classical symphony orchestra, BBC wildlife documentary, pizzicato,
  bassoon, tuba, … no erhu, no guzheng, no pop drums"), use `[inst]` sections, render 4 seeds and rank them with
  `tools/clap_score.py` against wanted vs unwanted styles (the rejected cue scored 0.88 "sentimental pop ballad";
  the adopted seed 0.96 "western orchestral nature-documentary score"). Run the same check on every cue.
- `music.py`'s vocal check ignores ASR on a stem under -50 LUFS: Granite hallucinates whole sentences on silence.
- Sound effects make a doc parody land. Prompt a render's `audio` for its effect (explosion, engine whine,
  clicking drive) and keep it with `"bed": "full"` (CLAP-check first); add others with shot `"sfx"` entries
  (`tools/sfx.py --audition 3`: audio-only H3 takes of the shot's subjects doing the action, CLAP picks).

## 8. Cut
- `python tools/resolve_edit.py film/<name>/shots.json` (Resolve Studio running, External scripting = Local).
- The previous mp4 is archived automatically as `_iterN`; the cut list goes to `edit/*.otio`.
- Actual picture cuts, sound placement and editable mixing belong in the Resolve timeline; a flattened
  FFmpeg assembly alone is not delivery. Keep dialogue, pistol reports, armor impacts/ricochets, cannon,
  door mechanism and ambience on separately editable tracks when present.
- Work on a versioned timeline, retaining older approved timelines, exports and source media. Record exact
  source frame ranges, frame rate, cue positions, trims, gains and processing so an approved cut is reproducible.
- Make a pacing pass before generating replacements: remove dead setup, static holds and delayed reactions,
  while retaining readable anticipation → action/contact → response → recovery. Enter near the first useful
  motion and cut once the beat lands. Earlier blocking can supply timing/choreography without becoming the
  new video input. Faster is not automatically better; inspect the actual motion and cut.
- Establish arrivals through a visible route with consistent scale, cast count, screen direction and
  accumulated aftermath. An unexplained appearance or false discharge needs a source repair, not just a trim
  or an audio mute. Diagnose source → Resolve viewer → export before changing the wrong layer.
- Where the installed Resolve supports native speed changes, retime picture only for selected non-dialogue
  action. Keep approved dialogue/source performances at normal speed on separate audio tracks. Use verified
  frame sampling for hard illustrated action when interpolation invents limbs/blades; inspect the result.
  Do not add a generic `shots.json` speed field: follow the supported native edit recipe in `reference.md`.
- Place each shortened/retimed picture item before its successor; an append using the old duration can
  overwrite the shorter span. Read back exact source endpoints, speed and integer output-frame duration.
  Distinguish source frames, timeline frames (including nonzero start) and audio sample-rate time bases.
- Remap contact effects to the inspected retimed frames, not only a duration ratio. Keep reports, impacts,
  footsteps, drops and room tone independently editable; retain source dialogue words, rate and clean tails.
- Keep each recorded sound's source URL, license and required attribution with its cue recipe. Inspect
  waveforms and listen source-solo, then in the final mix; peak readings and tool success are not listening QA.
- Edit-only shot fields are applied here and never re-render a shot: `titles` (`rank`, `lower`, `card`
  overlays), `zoom` (punch-in), `hold` (keeps a silent character's mouth shut; `tools/mouth_hold.py`), `out`
  (ends the shot early) and `in` (starts it later, render seconds).
- H3 invents words before lines as often as after them ("leafy you use, inside the mother…"), and speaks short
  lines with long pauses. Find both trims by ASR on the vocal stem: the shortest prefix that still holds the whole
  line → `out` = that + 0.5 s (0.7 s for a sung last note); only when words precede it, the latest start that still
  holds it → `in` = that − 0.25 s. Transcribe the already-separated stem directly (no second separation pass).
- Titles: the rank badge sits top right by default; grab a frame at every title and use `"align": "left"`
  wherever it covers a face (punch-ins move people).
- File size: film-level `"crf"` (default 16 ≈ 4 Mbps). 23 halves the file at SSIM 0.99, which is right for
  YouTube uploads.
- Check native timeline gaps/overlaps and actual render extent against expected integer frames. Recheck
  video/audio timestamps and final sample extent after mastering; bound audio to the cut without truncating
  approved speech. Measure the decoded final delivery's loudness/true peak after AAC, not only the PCM master.
  Encoding can overshoot the limiter ceiling; allow headroom and remeasure instead of trusting settings.
- Finish with a whole-film consistency sweep (`sheet.py --cols 5`), actual exported-frame inspection and
  full exported playback. Apply `qa.md`'s sequence palette and mix gates; report what remains broken or
  unverified rather than treating ASR, meters or a successful export as approval.
