"""Speaker similarity of vocal stems against a reference voice (WavLM x-vector cosine).

Needs torch + transformers + librosa, so run it with ComfyUI's embedded Python:
    <ComfyUI>/python_embeded/python.exe tools/voice_similarity.py \
        film/<name>/voices/hero.wav film/<name>/renders/s04.vocals.flac ...
Stems come from tools/speech_qa.py (renders/<id>.vocals.flac). Rough guide: same speaker is usually
> 0.90, a clearly different voice < 0.80. Stems with several speakers score lower.
"""
import sys

import librosa
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

MODEL = "microsoft/wavlm-base-plus-sv"


def main() -> None:
    ref, *stems = sys.argv[1:]
    extractor = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).eval()

    def embed(path: str) -> torch.Tensor:
        wav, _ = librosa.load(path, sr=16000, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=35)
        inputs = extractor(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            return torch.nn.functional.normalize(model(**inputs).embeddings, dim=-1)[0]

    r = embed(ref)
    for path in stems:
        print(f"{float(torch.dot(r, embed(path))):.3f}  {path}")


if __name__ == "__main__":
    main()
