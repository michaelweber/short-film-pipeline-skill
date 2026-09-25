# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MemoryError: VBAR allocation failed` | dynamic VRAM (comfy-aimdo) is on and fragmented | restart ComfyUI with `--disable-dynamic-vram` |
| ~150 s per step, or a 2 min shot taking over an hour | VRAM spilled into shared memory (several GPU jobs queued; TTS nodes keep models outside ComfyUI's manager, so `/free` doesn't release them) | one GPU job at a time; clear the queue and restart ComfyUI |
| Narrator timbre flips on one line | a bad TTS take | re-roll that line's `seed` in `narration`, rescore with `voice_similarity.py` |
| Last syllable of a line missing ("babies" → "baby") | the cut sits in the envelope dip inside a soft ending, or the take ended on the word | cut later (`split_take` pads 0.15 s / 0.6 s); if the take ends on the word, render it longer |
| Dialogue sounds cut off | narration placed on top of it, or `out` too tight | `narrate.py --plan`, move the line off the `!!` span; set `out` ~0.8 s after the line |
| Silent B-roll has distracting ambience | the shot's own audio is on A1 | `"bed": "none"` on every shot without a spoken line |
| `Resolve not reachable` | Resolve closed or scripting off | start DaVinci Resolve Studio, Preferences > General > External scripting using = Local; set `RESOLVE_SCRIPT_API` / `RESOLVE_SCRIPT_LIB` if Resolve is not in the default location |
| Re-rendered shot still stale in Resolve | Resolve caches media by path | never import from `renders/` directly; `resolve_edit.py` imports content-addressed copies from `edit/media/` |
| Silent puppet's mouth moves while the host talks | H3 lip-syncs every mouth in frame (worst in the first ~2 s, when speech starts at 0 s) | `tools/mouth_hold.py` → `"hold"` on the shot; don't tell the speaker to hold still at the start |
| A second character appears to replace the host | a solo shot in the host's exact framing | shoot them as an over-the-shoulder reverse angle (prompting.md) |
| Background copied from a likeness photo (e.g. a yellow screen behind the host) | ref mode with a `framing` still but no set description in the prompt | describe the set and layout in words in every ref-shot prompt |
| Extra words after a short line | a 5 s minimum shot for a 2–7 word line | "after it, silence…" in `audio`, then `"out"` at the end of the line |
| Cameras or drones in the stills | equipment named in `style` / `still_style` | describe the look, not the gear |
| True peak above −1 dBTP after delivery | AAC overshoots the limiter on dense mixes | `resolve_edit.py` lowers the limiter ceiling and re-encodes; rerun it |
| Resolve crop values look wrong | `CropLeft`/`CropRight` are pixels of the timeline, not fractions | pass fractions in shots.json; `resolve_edit.py` converts and verifies |
