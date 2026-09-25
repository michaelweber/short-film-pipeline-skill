# QA gates

`PY_TORCH` = a Python with torch, transformers and librosa (ComfyUI's embedded Python works, e.g.
`<ComfyUI_windows_portable>/python_embeded/python.exe`).

| Gate | Command | Pass | On fail |
|---|---|---|---|
| Dialogue words | `python tools/speech_qa.py film/<f>/shots.json` | recall ≥ 0.9 (precision ≥ 0.8) | re-roll `seed` with `--only <id> --force`; after 2, shorten the line |
| Narration timing | `python tools/narrate.py film/<f>/shots.json --plan` | no `!!` except over non-speech | move `at`, shorten the line, or move it to a quiet shot |
| Voice identity | `$PY_TORCH tools/voice_similarity.py voices/<v>.wav renders/<id>.vocals.flac narration/*.wav` | ≥ 0.85 (short lines score lower; compare against the clip's own half-vs-half score) | re-roll that clip/shot `seed` |
| Re-rolled takes | `speech_qa.transcribe(Path(clip), None)` | exact words | re-roll again |
| Clipped endings | last 60 ms of each narration piece vs its median speech level | ≤ −10 dB (near 0 dB = cut mid-word) | cut the piece later; re-render the take longer if the take itself ends on the word |
| Stills | `python tools/sheet.py film/<f>/shots.json --stills --tag stills` | one of each character, right scale, no garbled text, no stray cameras | seed sweep (+1..+3); remove equipment words from the style |
| Action shots | `python tools/sheet.py film/<f>/shots.json --ids <ids> --cols 5 --tag action` (≈1 fps tile) | action reads left to right, no reversed physics | restage the still, fixed camera |
| Silent mouths | `python tools/sheet.py film/<f>/shots.json --ids <id> --cols 16 --crop <box around the silent face>` | the non-speaking character's mouth never opens (check the cut, with holds applied) | `tools/mouth_hold.py` → `"hold"` on the shot (see prompting.md) |
| Invented words | `speech_qa.py` heard text | nothing after the scripted line | trim with `"out"` about 0.8 s after the line (`narrate.split_take` on the vocal stem finds it) |
| Hands | `python tools/sheet.py film/<f>/shots.json --cols 5 --tag hands` | five fingers on every visible hand, no floating hands | re-roll `seed`; restage the keyframe with hands resting, holding a prop, or out of frame |
| Cache | `python tools/h3_render.py film/<f>/shots.json --dry-run` | 0 stale (every line `= … up to date`) | render the stale shots |
| Titles | frame grab at every title's `at + 1 s` in the cut (`ffmpeg -ss <t> -i <f>.mp4 -frames:v 1 …`) | no badge or lower third over a face; the right text | `"align": "left"` on the rank badge |
| Delivered mix | ASR every narration and dialogue window of `<f>.mp4` | every line recall ≥ 0.9 | move narration, raise `duck_db`, or recut the clip |
| Delivery loudness | `ffmpeg -nostats -i film/<f>/<f>.mp4 -af ebur128=peak=true -f null -` | within ±0.5 LU of `loudness_lufs`, true peak ≤ −1 dBTP | rerun `resolve_edit.py` (it iterates loudness and the limiter ceiling) |
| Delivery size | `ffprobe` the mp4 | fits the destination (YouTube: `"crf": 23`, ~60 MB for 4 min at 1344×768) | set film `"crf"`, re-deliver |

Checking frames at scale: use one ffmpeg call per shot (`-vf "fps=4,crop=…,scale=-2:100,tile=Nx1"`), not a
Python loop of per-frame ffmpeg seeks. For long checks (ASR sweeps, previews of every shot), write a throwaway
script and run it as a background process.
