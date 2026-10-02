# Fizgig image editing and references

Default reference-guided still/edit pattern for any medium, including ordinary photoreal shots.
This is not the custom art-style/3D-animation pipeline. Use existing photos, approved frames and
prop references before commissioning sheets, rigs or a set. Timeline edits still happen in Resolve.

## Choose the operation

- **Reference generation:** give each picture one role: scene/camera, subject identity, prop geometry,
  or appearance/palette. Describe which role wins if references conflict. Match the approved scene's
  medium; a model render supplies geometry, not a grey backdrop or its surface finish.
- **Local repair:** retain the untouched approved image; paint the defect on a separate reference copy.
  Pair that marked scene with an unmarked identity/detail reference. Red paint communicates location
  to H3; it is not a spatial mask wired into the sampler.
- **Whole-frame change:** explicitly permit restaging/redrawing. Do not describe it as a pixel-preserving
  repair, and review all characters, camera, props, backgrounds and colours again.

## Repeatable graph and prompt

Reuse the user's supplied workflow: copy the exercised Fizgig still graph (the user's saved workflow,
exported to API format) into the current project's `workflows/`; never overwrite the source experiment or
selected assets.
This graph is a baseline, not a new `shots.json` engine: `keyframes.py` does not automatically run it.

1. Call Comfy `server_info` first. Inspect the live nodes/models and validate the copied workflow.
   The baseline needs `FizgigH3StillLatent`, `FizgigH3StillDecode`, H3 Ref2VA, its text encoder and VAEs.
   Missing packs/models are prerequisites; do not silently substitute a different editing method.
2. Upload the approved/marked scene and separate detail/identity pictures. Update the `LoadImage` names
   to returned input filenames. Keep dimensions, crop coordinates and reference order explicit.
3. The exercised graph feeds scaled pictures into `MiniMaxH3ReferenceToVideo`, samples a
   `FizgigH3StillLatent`, decodes via `FizgigH3StillDecode` and writes `SaveImage` candidates.
   It used `er_sde`, `beta`, 18 steps, denoise 1, batch 1, 1344×768 output and 1 MP `nearest-exact`
   references. Its conditioning node has `length: 5`; its final output is a still, not a video.
   Treat these as observed settings, not universal quality requirements. The local fp16 video VAE
   replaced the supplied int8 VAE; log such differences instead of claiming an identical reproduction.
4. Pin the seed, assign a unique output prefix and record the graph, model filenames and source pictures.
   Submit through `tools/gpu_queue.py submit WF.api.json --card h3|aux --wait` with the appropriate
   VRAM budget, or Comfy `run_workflow(wait=false)` → `job(action="wait")` → `fetch_outputs`.
   Read the actual `SaveImage` result; echoed input references are not generated outputs.
5. Inspect one representative candidate at full resolution before sweeping seeds or repairing a sequence.

Use the exercised structured reference prompt, replacing the subject-specific details:

```text
integrated_multimodal_description:
subject_definitions:
<Picture 1> is the approved scene with the repair region painted red.
<Picture 2> defines the exact identity/geometry of the replacement detail.
summary:
[reference generation] Repair the marked detail in the existing scene.
retention_analysis:
Camera, subject scale, unmarked characters, their clothing, props and background fully_preserved.
detailed_description:
Describe the replacement's shape, screen direction, scale, placement and contact/occlusion.
Use Picture 2 for identity/geometry and Picture 1 for the scene's medium and palette.
Restore the underlying surface where the red paint extends beyond the replacement.
Remove the repair paint and retain the unmarked scene.
[Shot 1] static image.
overall_soundscape: N/A
non_diegetic_music: N/A
```

For unmarked reference generation, describe the actual pictures rather than claiming red paint exists.
`fully_preserved` and “repair only” are conditioning instructions, not guarantees: the exercised empty
latent with denoise 1 resynthesizes the whole frame. Ordinary img2img also does not freeze unedited pixels.

## Character/location sheets and multi-character keyframes

- **Sheets:** the same graph at 2016×1152 (set node `28` and node `65` sizes together) draws character sprite
  sheets (portrait, front/side/back, detail circles) and location sheets (wide, 2×2 views, property circles) from
  1–3 input pictures in ~35 s. 3 of 8 character sheets needed a seed +1..+3 (pompom appearing only on side views,
  missing front view); layouts vary (views stacked in one column, touching views), so crop by white-gutter panels
  and valley cuts and record hand-set boxes when it drifts. Use the crops, never the whole sheet, as references.
- **Multi-character keyframes:** one picture per role, in this order: the location crop closest to the camera
  (scene/camera/palette), each character's single-pose crop (portrait for a face; the sheet's back view for a person
  seen from behind, plus his portrait for the sliver of cheek), then prop stills ("prop design only, ignore its grey
  backdrop"). Summary: "static cinematic film still of one new camera setup in the location of <Picture 1>";
  retention: location and each identity `fully_preserved`; the description is the shot's staging with each role
  tied to its picture. ~15–35 s each at 1344×768; 31 of 36 passed first time. Failures: the set drifted outdoors
  when the prompt never named it (write the location's description into the prompt), a helmet dropped, an extra
  person appeared in an animal shot — all fixed by seed +1 or naming the missing element.

## Preserve and approve

- If surrounding pixels must remain exact, register the candidate to the approved source and composite
  only a reviewed repair mask onto that source. Store both the visual marking and actual composite mask.
  Verify unchanged pixels outside the mask, including its feathered boundary. Inspect the seam at full size.
- Registration is a gate, not an excuse to warp anatomy into place. A shifted wrist, truncated fingertip,
  moved arm, bad grip, floating weapon or wrong floor reflection rejects the candidate even when all
  outside-mask pixels are unchanged. Expand a mask only when the expanded repair is intentional and reviewed.
- Review identity, count, pose, contact shadow/reflection, occlusion, palette and adjacent shot designs.
  A canonical prop can still be the wrong replacement if the selected sequence established another design.
  Repair the affected continuity span together; never introduce an isolated design swap.
- Compare approved source, raw generation and composite side by side; retain rejected candidates and the
  selection rationale. Promote only reviewed composites to production stills/guides.
- A corrected guide is not corrected footage. Replace/regenerate affected video, or use a proven tracked
  multi-frame repair; inspect temporal edges, contacts and the actual cut. Do not loop a still patch over motion.

## Evidence and limits

In the recorded trials (logged in an evaluation JSON beside that project's review outputs), Fizgig
produced candidates in roughly 10–15 seconds. Model-reference weapon geometry transferred usefully;
three hand trials were rejected for redesign, shifted registration or whole-arm movement. The weapon
candidate was promising, not adopted into that cut. Masked compositing preserved the surrounding scene,
not the correctness of the generated detail. Those trials did not prove animation or sequence acceptance.
Use this pattern preferentially for edits/references, with the same approval gates for every medium.
