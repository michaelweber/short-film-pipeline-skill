# Working with 3D models and rigs

Rules for rigged character/prop models used for Blender staging, guides and depth control. Everything here
comes from a sci-fi action short: repairing the source poses there took roughly 3–4× the agent time of making
the H3 shots from them (~15 agent-hours of pose repair vs ~4.3 for five complete shots in one session, plus days
more in earlier sessions). Almost all of it traced back to the model, not the shots.

## Qualify the model before any shot uses it

Run this once per model, before staging a single shot. Fixing the rig once is far cheaper than repairing
the same defect in every shot's pose.

1. **Weights.** Report per vertex group: influence count per vertex, weights below ~0.05, and groups that
   deform but aren't bones.
   - Cap influences at 4 per vertex and prune weights under ~0.05.
   - Remove non-bone groups (e.g. a derived `lower` group) from deformation.
   - Smooth the shoulder, hip and elbow regions.
   - Stray tiny weights pull faraway skin into a limb. A 0.024 ForeArm weight on a shoulder-height vertex, and
     hip/spine faces with 0.1–0.5 Arm weight, produced most of the torso-fold crossings on every shot.
2. **Deformation.** Check whether the rig has forearm twist bones and corrective shape keys at the elbow,
   shoulder and wrist.
   - Plain linear blend skinning without them folds the creases.
   - Switching to dual-quaternion (`use_deform_preserve_volume`) made it worse (2,033 events vs baseline),
     because the weights weren't painted for DQ.
   - Missing twist bones force a choice between two forearm-twist solution families. In one shot that became a
     55° one-frame pronation pop (52 px on screen), fixed only by a 30-frame ramp.
3. **Topology defects.** Find loose islands, and any polygon pairs that already intersect in the pristine REST
   pose. Delete the loose ones or list them as authored before any gate runs. A 9-vertex loose island on a
   character's inner upper arm cut into the skin at rest and turned up in every shot's evidence.
4. **Props.** Every held prop must be its own object with its own origin, never welded into the body mesh.
   - One armoured swordsman's katana was part of the body: the handle wrap stayed in BODY, and finger
     patches were cut into the prop.
   - A pass reclassifying 22,300 faces from the pristine FBX was needed before any grip could be evaluated.
5. **Costume bulk.** Pose the character in the story's hardest stances (two-handed guard, deep crouch, reach
   across the body) at rest geometry. Bulky plate armour sat at 1,200–4,400 crossings per two-handed pose even
   after full repair. Budget for shrunk inner shells, or plan one-handed / hidden-arm staging, before
   committing to the shots.
6. **Range-of-motion (ROM) test.** Run ~30 standard poses (reach, crouch, aim, two-handed guard, pickup,
   fallen) through the gate below. Fix the rig until they pass, then start shots.

## Rewrapping a Tripo character onto the MPFB/Rigify donor

The rig-repair scripts (build → qualify → swing) rebuild any Mixamo-named Tripo character on a clean donor
rig (native weights, Rigify controls, face shape keys). One profile per character; everything else is measured.

- **Profile, not constants** (one JSON profile per character): height, face `replace` (fitted donor face,
  DWPose landmarks generated automatically) or `covered` (helmet/mask kept rigid), per-arm `organic` /
  `covered` / `rigid`, rigid `shoulder_armour` / `boots`. Thresholds are fractions of skeleton segments or
  height; limb radii come from the source's inner envelope.
- **Garment weights are region-gated.** Unrestricted nearest-surface transfer gave a long jacket's hem hand and
  forearm weights, so bending an elbow moved the hips by up to 158 mm. The source rig's own groups now label
  each vertex's body region, and a vertex may inherit weights only from donor skin in a compatible region.
- **Clearance is part of the rig.** Each asset carries a hidden closed envelope: its own rest surface without
  the arms, openings capped. The clearance step measures hand/forearm penetration by ray parity and resolves
  it with the smallest shoulder rotation (FK `upper_arm_fk`, IK `hand_ik`). Resolve authored animation frame
  by frame from a clean start and bake to keys; resolve a held pose statically. Do not inflate the donor body
  into an envelope: its offsets folded thousands of triangles and made inside/outside meaningless.
- Qualification bakes clearance by default and records residual penetration as failures in its
  `metrics.json`; a successful exit is not an approval. Run it in measure-only mode to keep the authored
  motion and see how much clipping it has.

### Repeatable build and qualification

Keep a command/asset index and a documented profile schema beside the rig scripts, and record per project
which profiles exist. The two exercised profiles were a 2.2 m armoured fighter (replaced face, organic left
arm, covered right hand, rigid right forearm) and a 1.95 m armoured swordsman (covered face/hands, rigid
forearm armour); both have rigid shoulder armour and boots. They are not proof of arbitrary character
compatibility.

Keep each character's canonical asset and its matching grip evidence as a pair: the asset directory holds
the wrapped `.blend` and `character_manifest.json`; the qualification directory holds `grip_presets.json`,
`metrics.json`, `playback.blend`, `motion.mp4` and `contact_sheet.jpg`. Never substitute older shared
output folders for these per-character pairs. The manifest's `blend_path` is what the swing solver opens.

For an intentional rebuild, write to a separate output directory per retained iteration. To reuse an existing
asset, skip the build and point at its directory, but write qualification to a new directory so previous
qualification is not overwritten. After qualifying, read `status`, `failures` and `warnings` from its
`metrics.json`.

Inspect the build's `visuals/`, manifest and qualification's local detail renders and whole-body
playback, not only the status. Preset replay checks visible-hand rest geometry, weights and joint
signatures; mismatches are a reason to drop the saved grip presets and solve/qualify new grips, never to
force stale presets through. The swing solver does not repeat that signature gate. Keep the exact
asset/manifest/presets together. `QUALIFICATION_COMPLETE` means the command completed, including
runs marked `FAILED_OBSERVED_REGIONS`; even `MEASURED_AWAITING_VISUAL_REVIEW` requires inspection.

Animate `Donor.rig` controls, not `ORG-`, `MCH-` or `DEF-` bones. Use Rigify snapping to change modes:
`upper_arm_parent.L/R` and `thigh_parent.L/R` have `IK_FK=0` for IK, `1` for FK. Keep `Donor.body`,
the hidden `<Name>.clearance_envelope`, `<Name>.rigid_seams` helper, coherent seam modifier and the
donor's native Armature/Armature PV stack. The embedded `Donor.rig_ui.py` supplies the optional UI.
Qualification records pre-existing REST finger intersections separately from new visible pairs and
retains authored rest-contact depth in clearance. Those exemptions are provenance, not repaired
topology. Its local finger endpoint tests and hand/forearm envelope probes are not a whole-body,
all-prop, cloth-physics or actor-to-actor collision certificate.

### One separate katana per actor

Generate **one sword per model**, separate from the character; two hands may hold that one sword.
For a new prop:

1. Make a full-length, single-sword image from the approved design, with plain background and margins.
   Show a three-quarter view with the guard face, round handle and blade thickness legible. In one
   recorded run, an exact side view produced two fused blades/handles on one guard.
2. Use the logged-in **Tripo Studio** at `https://studio.tripo3d.ai/`, in a dedicated browser tab.
   The recorded API wallet had zero credits; Studio credits are separate. Do not redirect this workflow
   to an empty API wallet or assume current balances. Respect the authorised generation spend.
3. The recorded working route was Generate Model → HD Model → single image, H3.1, geometry +
   texture, Generate in Parts off, no multiview and no rigging. Orbit the result **before export**:
   one handle, one guard, one continuous blade; reject a fused pair even if Studio calls it one mesh.
4. Export the inspected model (GLB with embedded textures, or FBX with its texture directory), retain
   the reference, task ID, settings and provenance, then inspect it in Blender. The swing solver's slice
   check rejects substantial split blade/handle sections, but cannot prove a flawless prop.

Record each actor's katana file in the project notes; these are not interchangeable with swords welded into
a character. Keep an FBX's adjacent `.fbm` texture directory, and the reference image, provenance, verify
report and previews beside a GLB. One recorded export had five tiny loose fragments near the pommel/guard
and an arbitrary roughly 129-unit length: one sword is not the same as clean topology or correct scale.

### Measure a two-handed swing before rendering

After qualification has been reviewed, run the swing solver with the manifest, the qualified grip presets,
the actor's katana and `--length-fraction 0.52` in no-render mode, then read `katana`, `summary` and the
frames that are not `clear` from its `metrics.json`.

No-render mode writes `swings.blend` and `metrics.json`, **not** `key_poses.jpg` or a movie. Open the
baked blend and inspect full-body and hand-close views at the recorded `stages`: guard 1, overhead
raise/strike/follow 26/32/38, diagonal 74/82/88, horizontal 124/130/136 and recovery 160. Also inspect
the intervening motion and worst metric frames. Only after this review, the full (rendering) run
produces front/side `swings.mp4`, three-view `key_poses.jpg` and individual stills.

The rendering run **solves again**; it is not a render-only switch. For staging an existing measured swing,
reuse that character's existing `swings.blend` instead of rerunning the solver. Keep subsequent camera
and edit experiments separate. The current solver:

- Measures the long axis, widest guard section, short handle side and handle radius. The cutting edge
  comes from the blade's curve relative to the guard-to-tip chord when sori exceeds 0.004 of blade
  length. Generated thick oval blades gave a misleading wedge direction; the straight-blade fallback
  uses the thin wedge side as weak evidence and requires visual checking. A straight symmetric blade
  is rejected as undetermined. Inspect `edge_method`, `sori_m` and wedge agreement in the report.
- Scales overall length to `0.52 × height`, fits only the handle radially to qualified power grips,
  and reports handle overhang. Right hand sits below the guard, left hand a quarter palm behind it;
  a wide pommel grip exceeded short/armoured arms' reach. A mirrored 45° roll presents each forearm
  below-outside the handle, rather than rolling wrists across the midline.
- Drives both IK hands from sword-parented grip empties; chest-parented poles put elbows out/forward.
  Poses are anchored at the rear hand in front of the measured body surface. Sword offsets trade
  clearance against reach; elbow swivel is bounded to ±60°. Grip reach is prioritised, so short arms
  and bulky torsos can retain substantial penetration even when both hands stay attached.
- Smooths sword offsets and elbow swivel (Gaussian sigma 1.5 frames), corrects reach again and
  re-measures the final path used for baking. Compare each frame's final values with `authored` and
  `unsmoothed`; inspect correction steps for pops rather than trusting smoothing alone.

Recorded completed swing evidence:

| Character | Clear frames / 160 | Max grip error | Max reported penetration |
|---|---:|---:|---:|
| 2.2 m armoured fighter | 14 / 160 | 4.81 mm | 0.1470 m |
| 1.95 m armoured swordsman | 32 / 160 | 4.93 mm | 0.0862 m |

Here `clear` means grip-position error ≤5 mm and measured arm/blade envelope depth within the
solver tolerance; it does not include every collision or visual gate. Both runs have
`crossed_elbow_frames=0`, meaning no reversed right/left elbow ordering along the body-right axis.
**This is not a zero-crossed-arms test**: it does not check arm segments, forearm surfaces, hands
against each other, or the opponent. `SWING_COMPLETE` only marks completed output. Neither run is
collision-free or production-approved. Qualification is limited to the recorded poses, handle sizes,
profiles and measured regions; retain pre-existing defects and residual failures in the report.

### Export the blocking to H3 without migrating production

Use the existing baked swings in a separate staging scene, two actors with exactly one separate
katana each. A pair of solo swing studies is not a fight sequence: author distance, facing, attack,
defence, contact/near-miss, response and recovery across shots. Record the source-frame mapping
(including holds/retiming), actor transforms, camera and cut frames. Recheck the result after retiming
and composition: solo-body measurements do not cover either actor's blade versus the opponent.

Render **each shot's actual camera** as low-resolution 24 fps blocking (448×256 is the established
input size), not the diagnostic front/side montage. For a 124-frame clip, retain all 124 frames
and its exact 124/24-second timing in the recipe; inspect contact sheets and playback before H3.
Use an experiment-local shot list, renders and review outputs in their own directory; never replace
production models, shots, source media, approved renders or the Resolve cut as part of a test.

Follow [the optional Ref2VA recipe](art-animation.md#sword-blocking-to-ref2va): `mode: ref`,
film-level `prompt_format: h3_ref`, `ref_video`, explicit `turbo: false`, independent approved
character/environment pictures, timestamped beats and **no pinned guides**. Start at 672×384,
20 steps and a fixed seed. The established camera/gesture example (a clay blocking clip driving a
1–2 character staging shot) is not evidence that new rigs' sword contacts survive H3. A new fight
experiment is unapproved until its own source and generated-video gates in
[art-animation.md](art-animation.md#sword-blocking-and-generated-fight-gates) pass.

To attribute gains to the rewrap, compare a matched old-rig/new-rig pair with the same camera,
blocking beats and timing, appearance references, prompt, seed, resolution, steps and model settings.
If a matched baseline cannot be made, report a new-rig feasibility study, not an improvement claim.

### Recorded two-actor pipeline experiment

**Rejected by the user:** these stationary swing drills were substantially worse than the existing
tested fight blocking. Do not reuse such drills as the fight direction or as evidence of improved output.
For rig comparisons, preserve the released multi-actor choreography, camera, timing, footwork and sword
paths. Carry the replicated blocking through H3 finishing; Blender-only previews are an intermediate
check, not the vertical-slice deliverable.

The experiment's pattern: a staging script with a preview phase (inspect one contact sheet per exchange
before full blocking) and a render phase, then `tools/h3_render.py <experiment>/shots.json --dry-run`,
then the same command without `--dry-run` in a supervised hub process.

The saved scenes compose both native rigs, their appearance layers, swords, grip empties and poles;
all action-bearing dependencies use the same per-actor source-frame map. Fractional interpolation
raised one actor's measured grip-center error to 6.557 mm. Sampling whole baked source frames
reduced the maxima to 4.921 mm in A and 4.930 mm in B; integer retiming can still repeat/skip frames,
so this is not a general smooth-retargeting solution. Record every source frame in a `recipe.json`.
Workbench clay rendering retains the actual armour geometry, but loses texture-painted segmentation;
record the visible layers in an appearance-inspection file. Render on the aux card under its lease.

Both generated outputs contain 124 frames at 24 fps, 672×384, non-turbo Ref2VA at 20 steps (80 seconds
per generation). A is overhead attack/diagonal reply; B is diagonal attack/horizontal counter.
Source/generated playback and consecutive strike-frame review showed recognizable action timing,
stable screen sides and cel-style translation. These are **feasibility results, not production approval**:
the source remains planted-foot near-miss choreography with inherited clipping; small dark hands
do not establish finger anatomy or an organic-left/mechanical-right arm design. No matched old-rig
baseline was rendered, so no causal rig-improvement claim follows.
Keep a `review.html`, side-by-side comparison videos and a `review.json` with evidence and limitations.

## Grip library, not per-shot grips

- Author hand-to-prop grips once per prop (cannon, katana, two-handed katana), with finger curls that
  actually close on the real handle.
  - A frozen grasp stored per shot forced a 36–65° wrist bend on four shots, and each shot re-derived the
    same fix.
  - The corrected grasp (a 25° roll about the handle, 5 mm slide) cleared all of them.
- Keep the grip the same across a cut: one shot had to adopt the next shot's left grasp so the hand wouldn't
  jump at the cut. Check blade screen angles across cuts (134° vs 142° across one cut was fine because the
  camera changed).
- Record each prop's floor marks slightly above the floor. Authored marks at −2e-7 m or −1.13 mm read as
  sub-floor crossings and needed rigid lifts in every shot.

## Gate what the camera can see

H3 only receives the depth/colour guide from one camera. Certify against that camera from the first pass.

- **Hard gates, always 0:** props/weapons/drones vs actor, hand vs hand, hand vs held prop where visible,
  floor.
- **Arm/body self-crossings are acceptable** if they fall in one of these classes. List every exempted pair
  with its provenance.
  - The same polygon pair already crosses in pristine REST (authored).
  - An authored defect sliding a little: the loose-island rule, the island vs skin within 2 mm of its REST
    penetration.
  - Hidden from the production camera (ray-cast).
  - A visible fold of at most ~4 px span and ~2.5 px depth at the production camera.
  - Larger visible folds are fixed.
- Rebuild `calc_loop_triangles` from the evaluated mesh at every pose. Posing changes renderer diagonals (52
  triangles in one pose); cached or fan triangulation missed a real finger/cannon crossing.
- Measure what a gate actually selects. Any-positive-weight masks pull in faces through 0.02 weights; report
  both the any-weight count and the >0.001 count.
- Before the visibility rule, hours per shot went into clearing crossings that turned out to be hidden or
  under 2 px. Deciding visibility first would have removed most of that even with the flawed rig.

## Staging and art checks the gates don't catch

- **Continuity across cuts on the same camera.** Match every bone, IK target and attached prop at the cut
  frame, then ease into the authored motion over ~8–12 frames. Hand jumps across two consecutive cuts were
  38–78 px on a locked camera. Compare against the cut's actual out frame (`shot_selects.json`), not the take's
  last frame.
- **Readable art, not just zero crossings.**
  - A gate-clean solve hunched the shoulder 36° and buried the face.
  - Bound the clavicles (~15–25°), and add a face/head occlusion check for held-pose frames only.
  - A sword arm crossing the face mid-swing is natural.
- **Teleports in authored motion.** A drone prop jumped 1.3 m from mid-air to its floor wreck in one frame.
  Scan prop world positions for per-frame jumps and keyframe a real fall (blast knockback, tumble, bounce).
- **Hidden props must stay hidden.** Check with a camera ray-cast, not just the frustum. A prop can sit inside
  the frustum yet be fully occluded; a swung katana was once accepted this way.

## Tools

- `tools/blender_shard.py`: shard gate sweeps across single-thread Blender processes (`--no-preload`); it
  takes a slot from the machine-wide `blender-cpu` lease (16).
- `tools/blender_gpu.py`: OptiX renders on the aux card (or `BLENDER_GPU_CARD=h3`) without editing scripts.
- Worked patterns worth keeping per project: an FK arm solve with gate and visibility check, a loose-island
  rule script, and a certify/start-match/rigid-clearance module for grapples.
