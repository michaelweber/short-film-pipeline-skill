# Custom art style and animation

Load this reference only when the requested art direction requires a bespoke drawn/stylized animation
pipeline, or the user explicitly requests 3D blocking, rigging or motion transfer. Normal editing,
photoreal/reference-guided stills and ordinary H3 shots do not require this pipeline. Start with
[the main skill](SKILL.md) and [Fizgig editing/references](fizgig.md); do not load `models.md` by default.

## Style and generated keyframes

- `style`: one paragraph with medium, camera, light, texture. Prepended to H3 video prompts.
- `styles`: named variants (`human`, `action`, `card`) selected per shot with `"style": "<name>"`.
- `still_style` (optional): keyframe look; `keyframes.py` prefixes Klein prompts with this, not `style`.
  Keep both aligned: changing video `style` alone cannot repair an already recoloured still.
- Keep an approved starting frame as the palette/medium anchor through every keyframe edit and shot.
  Geometry references guide staging, not final surface texture; project-specific art choices belong here,
  not in the default workflow.
- For new invented cast/sets, build approved character/set references before dependent keyframes.
  The existing Krea-2 sheets → Klein keyframes route remains available through `keyframes.py`.
  Prefer Fizgig's explicit reference roles for repair/reference work; neither route guarantees consistency.

## Rig qualification and rendering

For MPFB/Rigify rewraps and one-katana-per-actor blocking, follow
[models.md](models.md#repeatable-build-and-qualification): build → qualify matching grips → measure
swings → inspect baked poses → camera-specific Ref2VA test. Completion markers are not collision approval.
Use earlier blocking as choreography/timing evidence before deciding to regenerate: a native edit may suffice.

- Blender renders on GPU without editing scripts: `blender -b F --python tools/blender_gpu.py --python S.py`
  (OptiX on the aux card by default; `BLENDER_GPU_CARD=h3` selects the H3 card; never falls back to CPU).
  Card UUIDs come from the `aux_gpu_uuid`/`h3_gpu_uuid` settings (env vars or `tools/local_settings.json`,
  read by `tools/pipeline_settings.py`). GPU and CPU pixels differ slightly: re-baseline any pixel-hash
  evidence. Gate/BVH evaluation is CPU-only: shard sweeps with `tools/blender_shard.py` (8 shards: identical
  results, 4.1× faster on 16 frames; use `--no-preload`).
- CPU cap: at most 16 Blender processes machine-wide, `-t 1`, low priority (`blender_shard.py --jobs N
  --low-priority`, the `blender-cpu` lease in `gpu_queue.py`). Uncapped parallel agents reached 53 processes and
  100 % CPU, starving the desktop. Never stop the user's GUI Blender or any process you did not launch.

## Blocking and motion-reference prompting

- **Action sequences: block in Blender, depth-lock every beat.** `tools/greybox.py` blocks each shot with rigged
  mannequins (aim points, sword tips, drops/pickups, gore/debris, muzzle flashes that come out of the barrel) and
  writes a depth map per beat. Each keyframe is a `control` still (Krea-2 depth Control LoRA → Klein refine 0.5), so
  poses, aim and placement match the blocking; the H3 shot pins all of them as `guides`. Plain Klein img2img from
  the greybox render drifted (guards facing the wrong way, flashes from the wrong end of guns).
  Geometry/3D props are staging guides only: translate their surfaces into the approved medium rather
  than pasting reference textures into a mismatched style. Preserve occlusion instead of inventing a hidden limb.
  Using real rigged character models instead of mannequins? Qualify them first (`models.md`): rig defects cost
  one production 3–4× the agent time of the shots themselves.
- **Say the cast count in every keyframe prompt** ("Exactly one person, the brute; nobody else standing"): the
  depth map's mannequins and bodies otherwise turn into extra people.
- **Blender blocking as a Ref2VA reference video works, used this way** (measured on a sci-fi action short):
  a low-res (448×256) clay blocking clip labelled as the motion/camera video,
  separate labelled pictures for the character sheet and the environment, NO keyframe guides, NO turbo. At
  672×384, 20 steps (~80 s) H3 followed the eased dolly (zoom 1.19 vs 1.178), the actor placement (bbox IoU 0.917)
  and the gesture timing, with no clay leak. Turbo ignored the video (IoU 0, invented walk-in); pinned guides make
  it jump between keyframes. That was the real cause of an earlier failed A/B (whose review sheet's middle row
  was the animatic input, not an output — label review rows). Use it for camera/staging shots with 1-2 characters;
  not as a guarantee for plate-contract or grip-gated action. Sword fights need the separate source/generated
  gates below and in `qa.md`. Identity follows the supplied sheet, so give it the current design.
- **No turbo LoRA with a depth ControlNet control video**: the fl8 turbo LoRA plus the Fun ControlNet union drew
  a flickering white rim along the control silhouette. Weaker control, backdrop depth, normalized depth and pixel
  defringe all failed; dropping the LoRA (20 steps, same seed/control) removed it (rim px/frame 158 → 46).
- **Write action prompts in the native H3 format** (`"prompt_format": "h3_ref"`): `[Shot 1]` plus timestamped beats
  ("At 00:01.550 he fires point-blank…"). Every beat names the limb, the screen direction, the contact point and
  the result. In a pinned-guide workflow cite its keyframe ("matching <key2> at 00:01.600"); in a reference-video
  workflow cite `@anim` and its timing instead, without adding guide pins. Camera moves are type + amplitude +
  speed ("trucks right with small amplitude at slow speed… never cuts"), and hits get "a single white impact frame".
- **Side-view fights (Oldboy style):** hide the near wall in the greybox, put the camera outside it, and keep one
  screen direction (the hero moves left→right) for the whole fight.

## Sword blocking to Ref2VA

Prepare the rig, separate props and camera-specific 24 fps video with
[models.md](models.md#measure-a-two-handed-swing-before-rendering). Reference video conditions motion;
it does not certify grip contact, blade continuity or collisions in the generated frames. The earlier
camera/gesture result above does not establish a sword-fight result for new rigs.

Use an experiment-local `shots.json`. At film level set `"prompt_format": "h3_ref"`,
`"width": 672`, `"height": 384`, `"steps": 20` and `"turbo": false`. Define `subjects.fighter_a`,
`subjects.fighter_b` and `subjects.set` with `desc` and `ref` paths to independently approved
character and environment pictures. Paths resolve relative to the shot list. Do not use the clay
render as their appearance, or a full scene as a character identity picture. Environment pictures
are separately labelled set-design references, not shots to cut to.

The shot fields below illustrate a five-second exchange; first author those beats in the blocking,
then replace the timestamps and wording to match its actual source-frame recipe. A 124-frame
blocking clip is 124/24 seconds; H3's `duration: 5` selects its supported 124-frame length.

```json
{
  "id": "exchange_a",
  "mode": "ref",
  "duration": 5,
  "seed": 530101,
  "turbo": false,
  "subjects": ["fighter_a", "fighter_b", "set"],
  "ref_video": {
    "video": "blocking/exchange_a.mp4",
    "desc": "a low-resolution 3D blocking render of this two-person sword exchange",
    "motion": {
      "fighter_a": "screen-left position, facing, both hands on one katana and attack timing",
      "fighter_b": "screen-right position, facing, both hands on one katana and defence timing"
    }
  },
  "prompt": "[Shot 1] One continuous locked wide shot of @set. Exactly two fighters, @fighter_a on screen left and @fighter_b on screen right, each holding exactly one separate katana with both hands. Follow the camera, spacing and motion timing of @anim. At 00:00.000 both settle into guard. At 00:01.000 @fighter_a raises his katana and strikes down toward @fighter_b. At 00:02.000 @fighter_b raises his own katana to intercept; the two blades meet between the fighters, and both keep their hands attached to their own handles. At 00:03.000 the blades separate as both recover, retaining their screen sides. At 00:04.000 both return to guard through the end."
}
```

The explicit `ref_video.motion` map contains only the two actors; leaving the set out causes
`compose_h3_ref` to label it as design/palette from its picture, redrawn from the video's viewpoint.
Do not add `guides`, `first_frame`, `last_frame` or `@keyN` references to this motion-video recipe.
Pinned guides can make H3 jump between keys, and turbo can ignore the motion video entirely.

Preflight with `python tools/h3_render.py <experiment>/shots.json --dry-run`.
Run the reviewed experiment using the supervised render process from `SKILL.md`, pointing at this
shot list, not the production one. Review the generated SaveVideo output, not an echoed input video.
Keep the next exchange's starting spacing, blade ownership, direction and recovery continuous with
the selected out-frame of the first; two unrelated solo demonstrations do not constitute a fight.
Match seed, camera, timing, prompt, appearance pictures, resolution and model settings for old/new-rig
A/B claims. Without that baseline, describe only the observed new-rig result. Apply
[the source/generated sword gates](#sword-blocking-and-generated-fight-gates) to both stages. The recorded
new-rig experiment transferred the broad motion, but is not production-approved; keep each experiment's
evidence and limitations in a `review.json` beside its shot list.

Motion-reference drafts use non-turbo at 672×384 for 16:9 and 20 steps (`turbo: false`), unlike plain
FL/ref turbo drafts. Retain independently approved character/set pictures and inspect the actual output.

## Example: adult cel-animation action short (user-approved direction)

- **Look:** coherent hand-drawn adult cel animation, bold clean ink and restrained flat shading. Keep the
  industrial corridor's approved starting palette and motivated red warning lights. Neutral structural
  surfaces stay neutral; a character's non-human skin colour and olive clothes remain intentional local colours.
- **Anatomy and staging:** keep the sword arm/hand behind the solid foreground door jamb; the gun-carrying
  arm is the existing segmented dark cybernetic arm. Preserve camera scale, guards' floor positions and
  accumulated blood, severed legs, helmet and other aftermath across cuts.
- **Cannon beat:** define lowered/ready initial state and settled post-recoil final state, with a fast
  purposeful lift between them. One cannon flash, recoil and an immediate target hit on the adjacent
  frame; no second bang or delayed reaction. Carry the same drawing and prop design through the motion.
- **Dialogue:** preserve the clear source line exactly as recorded (a two-word exclamation). No filler or invented speech; retain
  the clean source performance rather than replacing it merely because a generated take exists.
- **Rejected approaches:** a rendered cannon overlay and a generated sword-arm patch looked
  stylistically wrong and popped in. These are user-reported failures, not approved fixes or permission
  to expose an occluded limb. Restage/redraw coherently rather than adding another moving patch.
- **Sound direction:** unmistakable real recorded pistol discharges, supporting armor hits/ricochets,
  and a heavier distinct cannon. The current door-opening sound was rejected: prefer a short, grounded
  heavy mechanical slide/motor/latch matching the door's travel, not an unrelated sci-fi pulse/whoosh.
  These are acceptance targets; new sound/colour revisions still need the gates in `qa.md`.

## Sword blocking and generated fight gates

Apply these separately to the **source blocking** and the **generated H3 video**, retaining exact
shot IDs/frame numbers for each failure. Rig qualification does not transfer automatically to a new
camera, retimed action, second actor or generated take. See
[models.md](models.md#measure-a-two-handed-swing-before-rendering) for measured scope and residuals.

| Gate | Inspect | Pass / action on fail |
|---|---|---|
| Source sword ownership/count | Separate objects in the staged blend, actual camera frames and full playback | Two fighters, exactly one separate katana each; two hands may hold the same handle. No sword welded into either body, fused double blade, spare sword or duplicate handle. Repair/reject source before H3. |
| Generated sword ownership/count | Generated SaveVideo output at full size, stepped through strikes and occlusions; compare with source | Still exactly one sword per fighter, continuously identifiable before/after occlusion. Count physical blades/handles, not motion trails or mesh objects. Reject inventions, swaps and unexplained disappearance; a correct source count is not proof of generated count. |
| Blade continuity | Adjacent frames around raise, fastest strike, contact and recovery in both videos | A coherent guard-to-tip blade, stable length/curve/edge and plausible foreshortening. No splits, merging blades, detached tips, rubber bends or blade teleport. Inspect beyond the contact-sheet sample. |
| Hand attachment / arm shape | Both hands and handles, close views plus shot camera, final baked metrics | Hands remain on their own handle, fingers close plausibly, wrists and elbows move continuously. Source grip error ≤5 mm is a reach screen only, not proof of surface contact or no crossed arms. Inspect forearm/hand intersections and wrist twisting explicitly. |
| Camera-specific defects | Actual shot camera throughout, including opponent, floor, armour, blade and face occlusions | Reject visible new weapon/body and hand/hand intersections or floating props. List authored-rest/hidden/small-fold exemptions with frame, camera and evidence under `models.md`; never convert a solo envelope metric into whole-fight clearance. |
| Exchange timing and cuts | Frame-step source and output around attack, defence, contact/near-miss, response and selected shot boundary | Response follows the staged event without delayed/duplicate contact; distance, screen direction, blade ownership and hand/pose continuity survive the cut. Record source-frame mapping, fps and trim points. Solo studies or unrelated swings fail sequence acceptance. |

For a 24 fps exchange, a ≈1 fps action sheet is only navigation: inspect adjacent frames at each
critical event and play the whole sequence. Compare against the **final smoothed/baked** motion,
not only an authored or unsmoothed solver candidate. `SWING_COMPLETE` / `QUALIFICATION_COMPLETE`
are execution markers, not approvals. `crossed_elbow_frames=0` checks right/left elbow ordering,
not arm segment/surface crossings. The completed source swings of the two exercised characters still
report 14/160 clear frames (max penetration 0.147 m, 2.2 m armoured fighter) and 32/160 (0.0862 m,
1.95 m armoured swordsman); neither is collision-free.
Any staged subset needs its own camera-specific review, and H3 must pass independently.

To claim the rewrap improved H3, retain a matched old/new-rig comparison with the same camera,
source timing, prompts, appearance refs, seed, resolution, steps and model settings. Otherwise
label it feasibility evidence and identify confounds. Do not replace production assets/cut with
an experimental take. The recorded new-rig test demonstrated broad motion transfer, not certified
contacts, detailed hands or a causal rig improvement.
