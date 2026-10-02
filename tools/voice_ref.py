"""Build a clean voice-reference wav for one speaker from a video (YouTube URL or local file).

    python tools/voice_ref.py https://www.youtube.com/watch?v=XXXX --out film/x/voices/guest.wav \
        --start 0 --end 266 --keep "thank you for having me|that is a great question"
    python tools/voice_ref.py src.mp4 --out voices/host.wav --use 1,3,4      # pick spans by index

Steps: download audio (yt-dlp) -> trim [start, end] -> isolate vocals (Mel-Band RoFormer, via ComfyUI) ->
split the vocal stem at silences (-35 dB, 0.3 s; spans >= 0.8 s) -> transcribe each span (Granite ASR) ->
print "NN  start-end  score  text" (score = best word recall of any --keep phrase) and write every span to
<out stem>_spans/NN.wav -> concatenate the selected spans (--use, else keep-score >= 0.6, else all), cap at
--max seconds, loudnorm to -20 LUFS, mono 48 kHz -> --out. Separation and ASR run on the aux ComfyUI
(speech_qa.transcribe -> h3_render.AUX_URL, env AUX_COMFY_URL, default :8189).
Needs yt-dlp for URLs: python -m pip install yt-dlp
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from narrate import duration
from speech_qa import score, transcribe

MIN_SPAN = 0.8
KEEP_SCORE = 0.6


def speech_spans(stem: Path) -> list[tuple[float, float]]:
    """Same silencedetect parameters as narrate.dialogue_spans."""
    log = subprocess.run(["ffmpeg", "-i", str(stem), "-af", "silencedetect=n=-35dB:d=0.3", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [0.0] + [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return [(e, s) for e, s in zip(ends, starts + [duration(stem)]) if s - e >= MIN_SPAN]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="http(s) URL or local audio/video file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--keep", help="|-separated phrases the wanted speaker says")
    ap.add_argument("--use", help="comma-separated span indices (overrides --keep selection)")
    ap.add_argument("--max", type=float, default=20.0, help="cap on the assembled reference, seconds")
    a = ap.parse_args()
    keep = [p.strip() for p in a.keep.split("|")] if a.keep else []
    spans_dir = a.out.with_name(a.out.stem + "_spans")
    spans_dir.mkdir(parents=True, exist_ok=True)
    for old in spans_dir.glob("*.wav"):
        old.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if a.source.startswith("http"):
            r = subprocess.run([sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "wav", "-o",
                                str(tmp / "src.%(ext)s"), a.source], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"yt-dlp failed:\n{r.stderr}")
            src = tmp / "src.wav"
        else:
            src = Path(a.source)
        trim = tmp / "trim.wav"
        cut = ["-ss", str(a.start)] + (["-to", str(a.end)] if a.end is not None else [])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *cut, "-vn", "-ac", "2", "-ar", "44100",
                        str(trim)], check=True)
        stem = spans_dir / "vocals.flac"
        transcribe(trim, stem)  # writes the isolated vocal stem

        chosen: list[Path] = []
        use = {int(x) for x in a.use.split(",")} if a.use else None
        for n, (s, t) in enumerate(speech_spans(stem), start=1):
            span = spans_dir / f"{n:02d}.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-to", f"{t:.3f}", "-i", str(stem),
                            "-ac", "1", "-ar", "48000", str(span)], check=True)
            text = transcribe(span, tmp / f"{n:02d}.flac")
            best = max((score(p, text)[0] for p in keep), default=0.0)
            print(f"\r{n:02d}  {s:6.1f}-{t:6.1f}  {best:.2f}  {text}")
            if (use is not None and n in use) or (use is None and (best >= KEEP_SCORE if keep else True)):
                chosen.append(span)
        stem.unlink()
        if not chosen:
            raise SystemExit("no spans matched --keep; rerun with --use")

        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in chosen), encoding="utf-8")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-t", str(a.max), "-af", "loudnorm=I=-20:TP=-2", "-ac", "1", "-ar", "48000", str(a.out)],
                       check=True)
    print(f"-> {a.out} ({duration(a.out):.1f}s from spans {', '.join(p.stem for p in chosen)})")


if __name__ == "__main__":
    sys.exit(main())
