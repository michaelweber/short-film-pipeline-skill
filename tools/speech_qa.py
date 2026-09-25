"""Transcribe each rendered shot and flag dialogue that doesn't match the script (H3 sometimes babbles).

    python tools/speech_qa.py film/<name>/shots.json               # every rendered shot
    python tools/speech_qa.py film/<name>/shots.json --only s05,s11

For each shot: ffmpeg extracts the audio -> ComfyUI isolates vocals (Mel-Band RoFormer) -> Granite ASR
transcribes them. The transcript is compared with the quoted lines in the shot's "audio" field:
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
import urllib.parse
import urllib.request
from pathlib import Path

from h3_render import COMFY, Graph, http_json, upload_image, wait_for

SEPARATOR = "MelBandRoformer_fp32.safetensors"


def expected_lines(audio: str) -> list[str]:
    return re.findall(r'"([^"]+)"', audio or "")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


def score(expected: str, heard: str) -> tuple[float, float]:
    """(recall, precision) of heard words against the scripted words, in order."""
    e, h = words(expected), words(heard)
    if not e:
        return 1.0, 1.0
    if not h:
        return 0.0, 0.0
    matched = sum(b.size for b in difflib.SequenceMatcher(None, e, h, autojunk=False).get_matching_blocks())
    return matched / len(e), matched / len(h)


def transcribe(video: Path, stem_dest: Path | None) -> str:
    """ASR of the file's speech. With `stem_dest`, vocals are first separated from music/ambience (Mel-Band RoFormer)
    and the vocal stem is saved there; without it, the audio goes to ASR as-is (dry voice takes)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / f"{video.stem}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
                        str(wav)], check=True)
        name = upload_image(wav)
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
    resp = http_json("/prompt", {"prompt": g.nodes, "client_id": "speech_qa"})
    if resp.get("node_errors"):
        raise RuntimeError(json.dumps(resp["node_errors"])[:3000])
    entry = wait_for(resp["prompt_id"], video.stem)
    heard = ""
    for out in entry["outputs"].values():
        if "text" in out:
            heard = " ".join(out["text"]) if isinstance(out["text"], list) else str(out["text"])
        for item in out.get("audio", []):
            q = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
            with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=120) as r:
                stem_dest.write_bytes(r.read())
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
        lines = expected_lines(shot.get("audio") or film.get("audio", ""))
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
