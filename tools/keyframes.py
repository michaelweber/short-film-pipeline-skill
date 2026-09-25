"""Generate character/set reference sheets and keyframe stills on the local ComfyUI.

    python tools/keyframes.py film/<name>/shots.json               # render stills not yet rendered
    python tools/keyframes.py film/<name>/shots.json --only hero,k03
    python tools/keyframes.py film/<name>/shots.json --force

Reads the "stills" object of the same shot-list JSON used by h3_render.py:
  "stills": {
    "hero": {"prompt": "character sheet ...", "width": 1024, "height": 1024},
    "k03a": {"prompt": "Hero at the window ...", "refs": ["@hero", "@apartment"], "width": 1344, "height": 768}
  }
Engines (per still "engine", default: krea2 without refs, klein with refs):
  krea2 — Krea-2 Turbo text-to-image (8 steps): best look for from-scratch character and set sheets.
  klein — Flux.2 [Klein] 9B distilled with reference latents (4 steps): composes keyframes that keep
          the referenced characters/sets on-model.
refs are image paths (relative to the shot list) or "@<still id>" to reuse another still. Stills are
rendered in dependency order to <shotlist dir>/stills/<id>.png, cached by a digest of
prompt + params + ref contents. "still_style" (top level) is prepended to every still prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

from h3_render import Graph, http_json, upload_image, wait_for, COMFY  # noqa: F401  (shared ComfyUI helpers)
import urllib.parse
import urllib.request

KLEIN = {"unet": "flux-2-klein-9b-fp8.safetensors", "clip": "qwen_3_8b_fp8mixed.safetensors",
         "vae": "flux2-vae.safetensors", "steps": 4}
KREA2 = {"unet": "krea2_turbo_fp8_scaled.safetensors", "clip": "qwen3vl_4b_fp8_scaled.safetensors",
         "vae": "qwen_image_vae.safetensors", "steps": 8}


def build_krea2(prompt: str, w: int, h: int, seed: int, prefix: str) -> dict:
    g = Graph()
    unet = g.add("UNETLoader", unet_name=KREA2["unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=KREA2["clip"], type="krea2", device="default")
    vae = g.add("VAELoader", vae_name=KREA2["vae"])
    pos = g.add("CLIPTextEncode", clip=[clip, 0], text=prompt)
    neg = g.add("ConditioningZeroOut", conditioning=[pos, 0])
    latent = g.add("EmptyLatentImage", width=w, height=h, batch_size=1)
    out = g.add("KSampler", model=[unet, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                seed=seed, steps=KREA2["steps"], cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    image = g.add("VAEDecode", samples=[out, 0], vae=[vae, 0])
    g.add("SaveImage", images=[image, 0], filename_prefix=prefix)
    return g.nodes


def build_klein(prompt: str, w: int, h: int, seed: int, refs: list[str], prefix: str) -> dict:
    g = Graph()
    unet = g.add("UNETLoader", unet_name=KLEIN["unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=KLEIN["clip"], type="flux2", device="default")
    vae = g.add("VAELoader", vae_name=KLEIN["vae"])
    pos = [g.add("CLIPTextEncode", clip=[clip, 0], text=prompt), 0]
    neg = [g.add("ConditioningZeroOut", conditioning=pos), 0]
    for name in refs:
        img = g.add("LoadImage", image=name)
        scaled = g.add("ImageScaleToTotalPixels", image=[img, 0], upscale_method="lanczos", megapixels=1.0,
                       resolution_steps=1)
        lat = [g.add("VAEEncode", pixels=[scaled, 0], vae=[vae, 0]), 0]
        pos = [g.add("ReferenceLatent", conditioning=pos, latent=lat), 0]
        neg = [g.add("ReferenceLatent", conditioning=neg, latent=lat), 0]
    latent = g.add("EmptyFlux2LatentImage", width=w, height=h, batch_size=1)
    sigmas = g.add("Flux2Scheduler", steps=KLEIN["steps"], width=w, height=h)
    guider = g.add("CFGGuider", model=[unet, 0], positive=pos, negative=neg, cfg=1.0)
    noise = g.add("RandomNoise", noise_seed=seed)
    sampler = g.add("KSamplerSelect", sampler_name="euler")
    out = g.add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0],
                sigmas=[sigmas, 0], latent_image=[latent, 0])
    image = g.add("VAEDecode", samples=[out, 0], vae=[vae, 0])
    g.add("SaveImage", images=[image, 0], filename_prefix=prefix)
    return g.nodes


def order(stills: dict) -> list[str]:
    done: list[str] = []

    def visit(sid: str, stack: tuple[str, ...] = ()) -> None:
        if sid in done:
            return
        if sid in stack:
            raise SystemExit(f"ref cycle: {' -> '.join(stack + (sid,))}")
        if sid not in stills:
            raise SystemExit(f"unknown still @{sid}")
        for r in stills[sid].get("refs", []):
            if r.startswith("@"):
                visit(r[1:], stack + (sid,))
        done.append(sid)

    for sid in stills:
        visit(sid)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", type=Path)
    ap.add_argument("--only", help="comma-separated still ids")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    film = json.loads(a.shotlist.read_text(encoding="utf-8"))
    root = a.shotlist.parent
    out_dir = root / "stills"
    out_dir.mkdir(exist_ok=True)
    stills = film.get("stills", {})
    only = set(a.only.split(",")) if a.only else None
    style = film.get("still_style", "")
    for i, sid in enumerate(order(stills)):
        spec = stills[sid]
        dest = out_dir / f"{sid}.png"
        if only and sid not in only:
            continue
        refs = [out_dir / f"{r[1:]}.png" if r.startswith("@") else root / r for r in spec.get("refs", [])]
        for r in refs:
            if not r.exists():
                raise SystemExit(f"{sid}: missing ref {r}")
        prompt = f"{style}\n\n{spec['prompt']}".strip() if style else spec["prompt"]
        w, h = spec.get("width", film.get("width", 1024)), spec.get("height", film.get("height", 1024))
        # Keyed off the still id, so adding stills doesn't re-roll the others.
        seed = spec.get("seed", film.get("seed", 0) + zlib.crc32(sid.encode()) % 100_000)
        engine = spec.get("engine", "klein" if refs else "krea2")
        if engine == "krea2" and refs:
            raise SystemExit(f"{sid}: krea2 takes no refs; use engine klein")
        digest = hashlib.sha1(json.dumps([prompt, w, h, seed, engine, [hashlib.sha1(r.read_bytes()).hexdigest()
                                                                         for r in refs]]).encode()).hexdigest()
        sidecar = dest.with_suffix(".json")
        if not a.force and dest.exists() and sidecar.exists() and json.loads(sidecar.read_text())["digest"] == digest:
            print(f"= {sid}: up to date")
            continue
        prefix = f"h3film/{root.name}/stills/{sid}"
        graph = (build_klein(prompt, w, h, seed, [upload_image(r) for r in refs], prefix) if engine == "klein"
                 else build_krea2(prompt, w, h, seed, prefix))
        t0 = time.time()
        resp = http_json("/prompt", {"prompt": graph, "client_id": "keyframes"})
        if resp.get("node_errors"):
            raise RuntimeError(f"{sid}: {json.dumps(resp['node_errors'])[:3000]}")
        entry = wait_for(resp["prompt_id"], sid)
        item = next(img for o in entry["outputs"].values() for img in o.get("images", []))
        q = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type")})
        with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=120) as r:
            dest.write_bytes(r.read())
        sidecar.write_text(json.dumps({"digest": digest, "seed": seed, "prompt": prompt}, indent=2), encoding="utf-8")
        print(f"\r+ {sid}: {dest} ({time.time() - t0:.0f}s)          ")


if __name__ == "__main__":
    sys.exit(main())
