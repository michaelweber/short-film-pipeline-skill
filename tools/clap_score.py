"""Score audio files against text descriptions with CLAP (laion/clap-htsat-unfused, HF cache; needs torch, so run
with ComfyUI's Python). Picks the SFX take that sounds most like its description, or ranks score candidates
against wanted/unwanted styles.

    $PY_TORCH tools/clap_score.py --label "a dog barking" --label "a door slamming" take1.flac take2.flac

Prints one JSON line per file: {"file", "scores": {label: softmax probability over the labels}, "sims": {label:
cosine similarity}}. The audio is scored over its first 10 s window and, for longer files, averaged over 10 s
windows every 20 s (CLAP's input is 10 s at 48 kHz).
"""
import argparse
import json
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    with torch.no_grad():
        t = proc(text=a.label, return_tensors="pt", padding=True).to(dev)
        text = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1)
        for f in a.files:
            y, _ = librosa.load(f, sr=48000, mono=True)
            win = 48000 * 10
            starts = range(0, max(1, len(y) - win + 1), 48000 * 20) if len(y) > win else [0]
            chunks = [y[s:s + win] for s in starts]
            x = proc(audios=chunks, sampling_rate=48000, return_tensors="pt").to(dev)
            emb = torch.nn.functional.normalize(model.get_audio_features(**x), dim=-1).mean(0, keepdim=True)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            sims = (emb @ text.T)[0]
            probs = torch.softmax(sims * model.logit_scale_a.exp(), dim=0)
            print(json.dumps({"file": f, "scores": {lbl: round(float(p), 3) for lbl, p in zip(a.label, probs)},
                              "sims": {lbl: round(float(s), 3) for lbl, s in zip(a.label, sims)}}), flush=True)


if __name__ == "__main__":
    main()
