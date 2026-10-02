# Troubleshooting

| Symptom | Cause / diagnostic hypothesis | Fix |
|---|---|---|
| `MemoryError: VBAR allocation failed` | dynamic VRAM (comfy-aimdo) is on and fragmented | restart ComfyUI with `--disable-dynamic-vram` |
| ~150 s per step, then `Fatal Python error: Aborted` | VRAM spilled into shared memory (two GPU jobs at once) | one GPU job at a time; restart ComfyUI via hub (`hub restart name=comfyui`) |
| Narrator timbre flips on one line | a bad TTS take | re-roll that line's `seed` in `narration`, rescore with `voice_similarity.py` |
| Dialogue sounds cut off | narration placed on top of it | `narrate.py --plan`, move the line off the `!!` span |
| `Resolve not reachable` | Resolve closed or scripting off | start DaVinci Resolve Studio, Preferences > General > External scripting using = Local |
| Re-rendered shot still stale in Resolve | Resolve caches media by path | never import from `renders/` directly; `resolve_edit.py` imports content-addressed copies from `edit/media/` |
| Silent puppet's mouth moves while the host talks | H3 lip-syncs every mouth in frame (worst in the first ~2 s, when speech starts at 0 s) | `tools/mouth_hold.py` → `"hold"` on the shot; don't tell the speaker to hold still at the start |
| Background copied from a likeness photo (e.g. a yellow screen behind the host) | ref mode with a `framing` still but no set description in the prompt | describe the set and layout in words in every ref-shot prompt |
| Extra words after a short line ("please stand", "and I will not be misled") | a 5 s minimum shot for a 2–7 word line | "after it, silence…" in `audio`, then `"out"` at the end of the line |
| Resolve crop values look wrong | `CropLeft`/`CropRight` are pixels of the timeline, not fractions | pass fractions in shots.json; `resolve_edit.py` converts and verifies |
| Prop/limb repair looks pasted on or pops in | medium mismatch or independently moving patch; inspect frame-by-frame | use [Fizgig references and a reviewed repair mask](fizgig.md) for stills; verify registration and temporal repair separately; keep staging refs geometric and hidden limbs occluded (stylized examples in [art-animation.md](art-animation.md#example-adult-cel-animation-action-short-user-approved-direction)) |
| Pistol fire is lost beneath ricochets | source selection, cue alignment or masking are candidates, not established by peak readings | compare recorded source solo → Resolve track solo → mixed export; align the muzzle report and reduce/re-time competing impacts until it is unmistakable |
| Door opening sounds wrong | user rejection establishes the mismatch, not a technical cause | audition a short heavy mechanical slide/motor/latch against visible travel; reject unrelated sci-fi pulses/whooshes and verify the mixed export |
| Scene grows progressively green | drift may enter keyframe edits, generated video, Resolve processing or export; codec fault is unverified | compare sequence endpoints and intermediate edits, then matching source/Resolve/export frames; correct the first diverging stage, anchoring neutral structures without removing intentional red lights, green skin or olive clothes |
| Peaks/ASR/export status pass but review fails | automated checks do not establish perceptual quality | perform source-solo versus final-mix listening, exported-frame comparison and full playback; keep the issue open until the actual cut passes |

## Example: three-shot palette drift trace (observed, not a new fix)

In a sci-fi corridor sequence (shots 4–6), the reviewed source/export comparison found essentially matching
colour, placing the observed drift upstream of Resolve/export: cool shot-4 handoff → whole-frame ReferenceLatent
shot-5 endpoint redraws →
already-green old ending → three integrated endpoint corrections becoming yellower → H3 propagating
the mismatched first/last colours. The correction workflows lacked `init` and resynthesized the whole
image despite "change only hand" wording. Treat such edits as full-frame redraws, not pixel-preserving
repairs. Match unchanged materials and every endpoint to the approved palette before H3; set the Klein
`still_style`, not only H3's `style`. This trace does not establish that any subsequent fix has passed.
