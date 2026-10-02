"""Word-timestamped ASR of audio files with local Whisper (transformers; needs torch, so run with ComfyUI's Python).

    $PY_TORCH tools/asr_words.py take.flac [...]

Prints one JSON line per file: {"file", "words": [[word, start, end], ...], "plain_words": [...]}: "words" from a
verbatim-prompted decode (VERBATIM_PROMPT; stutters, repeats, fillers and babble stay in as their own words),
"plain_words" from the default decode. ~2.5 s per narration take for both on the aux card after a ~2 s model load,
and one pass yields every word's time, so narrate.split_take cuts a line without a binary search. Model:
openai/whisper-large-v3 (HF cache); override with env ASR_WORDS_MODEL. small.en smoothed flubs into the script
("the bridge and the bridge is" -> "the bridge is"). Runs on the aux card when CUDA_VISIBLE_DEVICES pins it (narrate.py
does).
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


# Disfluent prompt text: Whisper copies the prompt's style, so it transcribes stutters, false starts, repeats and
# fillers verbatim instead of tidying them into the sentence it expects (measured: "nobody in this town has, nobody
# in this town has ever left", "nest -nesting, uh, nesting" and "asso - assume" all read as clean lines without it).
VERBATIM_PROMPT = ("Um, uh, so, like, hmm... I-I mean, the, the thing. Uh-huh. Mm. Ah, hm, er, mm-hmm, nah, yeah, "
                   "uh, whoa.")


def main() -> None:
    import torch
    from transformers import WhisperProcessor, pipeline

    cuda = torch.cuda.is_available()
    model = os.environ.get("ASR_WORDS_MODEL", "openai/whisper-large-v3")
    asr = pipeline("automatic-speech-recognition", model=model, device=0 if cuda else -1,
                   dtype=torch.float16 if cuda else torch.float32)
    gen = {} if model.endswith(".en") else {"language": "en", "task": "transcribe"}
    prompt = torch.tensor(WhisperProcessor.from_pretrained(model).get_prompt_ids(VERBATIM_PROMPT))
    prompt = prompt.to("cuda") if cuda else prompt

    def decode(path: str, **extra) -> list:
        out = asr(path, return_timestamps="word", chunk_length_s=30, generate_kwargs={**gen, **extra})
        return [[c["text"].strip(), float(c["timestamp"][0]), float(c["timestamp"][1] if c["timestamp"][1] is not None
                                                                    else c["timestamp"][0] + 0.3)]
                for c in out["chunks"]]

    def run(path: str) -> None:
        # "words": verbatim-prompted decode (cutting and the extra-words gate); "plain_words": default decode
        # (recall: the prompted decode occasionally stops early).
        words = decode(path, prompt_ids=prompt)
        print(json.dumps({"file": path, "words": words, "plain_words": decode(path)}), flush=True)

    if sys.argv[1:] == ["--serve"]:  # persistent worker: one path per stdin line, one JSON line back (narrate.py)
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            if line.strip():
                try:
                    run(line.strip())
                except Exception as e:  # noqa: BLE001 - report and keep serving
                    print(json.dumps({"file": line.strip(), "error": str(e)}), flush=True)
        return
    for path in sys.argv[1:]:
        run(path)


if __name__ == "__main__":
    main()
