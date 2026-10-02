"""Transcribe each rendered shot and flag dialogue that doesn't match the script (H3 sometimes babbles).

    python tools/speech_qa.py film/<name>/shots.json               # every rendered shot
    python tools/speech_qa.py film/<name>/shots.json --only s05,s11

For each shot: ffmpeg extracts the audio -> the aux ComfyUI (h3_render.AUX_URL, :8189) isolates vocals (Mel-Band
RoFormer) -> Granite ASR transcribes them. The transcript is compared with the scripted lines: quoted lines in the
shot's "audio" field, or <d>[Language] ...</d> blocks in its "prompt" (the native H3 format, "prompt_format":
"h3_ref").
recall = scripted words heard, precision = heard words that were scripted (catches invented babble).
Writes renders/<id>.vocals.flac (an isolated vocal stem, handy as a voice reference) and
renders/speech_qa.json. Shots whose recall or precision falls below --threshold are flagged;
shots with no quoted lines are skipped.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from h3_render import AUX_URL, Graph, http_json, upload_image, view, wait_for

SEPARATOR = "MelBandRoformer_fp32.safetensors"


def expected_lines(shot: dict, film: dict | None = None) -> list[str]:
    """The shot's scripted dialogue, in order."""
    tagged = re.findall(r"<d>\[[^\]]*\]\s*(.*?)</d>", shot.get("prompt") or "")
    return tagged or re.findall(r'"([^"]+)"', shot.get("audio") or (film or {}).get("audio", ""))


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _spell(n: int) -> str:
    """English words for 0 <= n < 10**12 ("fifty six", "ten thousand"), so ASR digits match scripted words."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _spell(n % 100))
    for size, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            return _spell(n // size) + " " + name + ("" if n % size == 0 else " " + _spell(n % size))
    return str(n)


def _number(tok: str) -> str:
    n = int(tok.replace(",", ""))
    if "," not in tok and len(tok) == 4 and 1100 <= n < 2100 and n % 100:  # a year: "twenty twenty one"
        return _spell(n // 100) + " " + ("oh " if n % 100 < 10 else "") + _spell(n % 100)
    return _spell(n) if n < 10 ** 12 else tok


def words(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"(\d[\d,]*)\s*%", r"\1 percent", text)
    text = re.sub(r"(\d)\s+,(\d{3})", r"\1,\2", text)  # word-timestamp tokens rejoin as "10 ,000"
    # whole numbers only: "3d" stays a token of its own
    text = re.sub(r"(?<![a-z\d])(?:\d{1,3}(?:,\d{3})+|\d+)(?![a-z\d])", lambda m: _number(m.group()), text)
    return re.findall(r"[a-z']+|\d+[a-z]+", text)


def _split_compounds(tokens: list[str], other: list[str]) -> list[str]:
    """Splits a token that is two adjacent tokens of `other` run together ("bluejay" vs "blue jay",
    "kitbashes" vs "kit bashes"): ASR and script spell compounds either way."""
    pairs = {a + b: [a, b] for a, b in zip(other, other[1:])}
    return [w for t in tokens for w in pairs.get(t, [t])]


def score(expected: str, heard: str) -> tuple[float, float]:
    """(recall, precision) of heard words against the scripted words, in order."""
    e, h = words(expected), words(heard)
    if not e:
        return 1.0, 1.0
    if not h:
        return 0.0, 0.0
    e, h = _split_compounds(e, h), _split_compounds(h, e)
    matched = sum(b.size for b in difflib.SequenceMatcher(None, e, h, autojunk=False).get_matching_blocks())
    return matched / len(e), matched / len(h)


def extra_words(expected: str, heard: str) -> list[str]:
    """Heard words beyond the script: insertions, and the surplus of a replacement that is longer than the scripted
    words it replaces (a same-length replacement is a mishearing, e.g. "patients" for "patience"). Repeats, stutter
    fragments, fillers and babble all land here; run it on a verbatim decode (asr_words.py "words")."""
    e, h = words(expected), words(heard)
    e, h = _split_compounds(e, h), _split_compounds(h, e)
    extra = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, e, h, autojunk=False).get_opcodes():
        if op == "insert":
            extra += h[j1:j2]
        elif op == "replace" and j2 - j1 > i2 - i1:
            extra += h[j1 + (i2 - i1):j2]
    return extra


def transcribe(video: Path, stem_dest: Path | None) -> str:
    """ASR of the file's speech. With `stem_dest`, vocals are first separated from music/ambience (Mel-Band RoFormer)
    and the vocal stem is saved there; without it, the audio goes to ASR as-is (dry voice takes)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / f"{video.stem}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
                        str(wav)], check=True)
        name = upload_image(wav, base=AUX_URL)
    g = Graph()
    speech = [g.add("LoadAudio", audio=name), 0]
    if stem_dest is not None:
        sep = g.add("MelBandRoFormerModelLoader", model_name=SEPARATOR)
        speech = [g.add("MelBandRoFormerSampler", model=[sep, 0], audio=speech), 0]
        g.add("SaveAudio", audio=speech, filename_prefix=f"h3film/qa/{video.stem}_vocals")
    asr = g.add("GraniteASREngineNode", model_name="granite-speech-4.1-2b", device="auto", max_new_tokens=256,
                asr_use_forced_aligner=False)
    text = g.add("UnifiedASRTranscribeNode", engine=[asr, 0], audio=speech, language="English",
                 chunk_size=0, enable_asr_cache=False)
    g.add("PreviewAny", source=[text, 0])
    resp = http_json("/prompt", {"prompt": g.nodes, "client_id": "speech_qa"}, base=AUX_URL)
    if resp.get("node_errors"):
        raise RuntimeError(json.dumps(resp["node_errors"])[:3000])
    entry = wait_for(resp["prompt_id"], video.stem, base=AUX_URL)
    heard = ""
    for out in entry["outputs"].values():
        if "text" in out:
            heard = " ".join(out["text"]) if isinstance(out["text"], list) else str(out["text"])
        for item in out.get("audio", []):
            stem_dest.write_bytes(view(item, AUX_URL))
    return heard.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--only", help="comma-separated shot ids")
    ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    renders = a.shotlist.parent / "renders"
    report_path = renders / "speech_qa.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    only = set(a.only.split(",")) if a.only else None
    for shot in film["shots"]:
        sid = shot["id"]
        video = renders / f"{sid}.mp4"
        lines = expected_lines(shot, film)
        if (only and sid not in only) or not video.exists() or not lines:
            continue
        heard = transcribe(video, renders / f"{sid}.vocals.flac")
        expected = " ".join(lines)
        recall, precision = score(expected, heard)
        flagged = min(recall, precision) < a.threshold
        report[sid] = {"recall": round(recall, 2), "precision": round(precision, 2), "expected": expected,
                       "heard": heard, "flagged": flagged}
        print(f"\r{'!!' if flagged else 'ok'} {sid} recall {recall:.2f} precision {precision:.2f}\n"
              f"   expected: {expected}\n   heard:    {heard}")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    flagged = sorted(k for k, v in report.items() if v["flagged"])
    print(f"\nflagged ({len(flagged)}): {', '.join(flagged) or 'none'}  -> {report_path}")


if __name__ == "__main__":
    sys.exit(main())
