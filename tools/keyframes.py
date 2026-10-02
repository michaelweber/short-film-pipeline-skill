"""Generate character/set reference sheets and keyframe stills on the aux ComfyUI (h3_render.AUX_URL, :8189).

    python tools/keyframes.py film/<name>/shots.json               # render stills not yet rendered
    python tools/keyframes.py film/<name>/shots.json --only kid,k03
    python tools/keyframes.py film/<name>/shots.json --force

Reads the "stills" object of the same shot-list JSON used by h3_render.py:
  "stills": {
    "kid":  {"prompt": "character sheet ...", "width": 1024, "height": 1024},
    "k03a": {"prompt": "The boy at the window ...", "refs": ["@kid", "@apartment"], "width": 1344, "height": 768}
  }
Engines (per still "engine", default: krea2 without refs, klein with refs):
  krea2 — Krea-2 Turbo text-to-image (8 steps): best look for from-scratch character and set sheets.
  klein — Flux.2 [Klein] 9B distilled with reference latents (4 steps): composes keyframes that keep
          the referenced characters/sets on-model.
refs are image paths (relative to the shot list) or "@<still id>" to reuse another still. Stills are
rendered in dependency order to <shotlist dir>/stills/<id>.png, cached by a digest of
prompt + params + ref contents. "still_style" (top level) is prepended to every still prompt.
"still_clio_style" (top level) routes every still prompt through the ClioStyle node (clio-style-preview
custom node, 398 named styles); a still's own "clio_style" wins, and "none" disables it for that still.
"init" (a path relative to the shot list, or "@<still id>") plus "denoise" (default 0.85) on a klein still starts
it from that picture instead of noise (img2img): camera, layout and placement survive, e.g. from a greybox render
(tools/greybox.py). refs still steer the characters.
"control": {"image": "blocking/p08_t1.6_depth.png", "engine": "krea2"|"zimage", "strength", "refine": 0.5,
"refine_prompt"} makes a depth-locked keyframe: Krea-2 with the depth Control LoRA (or Z-Image with its Fun
ControlNet) draws the frame from a greybox depth map (tools/greybox.py <id>_t<sec>_depth.png, near = white), so
camera, poses, aim and placement match the blocking exactly; with refs, a Klein img2img pass at "refine" denoise
then puts the characters on-model from their sheets. All in one graph.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import sys
import time
import zlib
from pathlib import Path

from h3_render import AUX_URL, Graph, http_json, upload_image, view, wait_for
import urllib.parse

KLEIN = {"unet": "flux-2-klein-9b-fp8.safetensors", "clip": "qwen_3_8b_fp8mixed.safetensors",
         "vae": "flux2-vae.safetensors", "steps": 4}
KREA2 = {"unet": "krea2_turbo_fp8_scaled.safetensors", "clip": "qwen3vl_4b_fp8_scaled.safetensors",
         "vae": "qwen_image_vae.safetensors", "steps": 8}
# ClioStyle's default template, passed explicitly so a node update can't silently change the prompt.
CLIO_TEMPLATE = "Style: {name}: {style}. Subject: {prompt}"


def text_input(g: Graph, prompt: str, clio: str | None) -> str | list:
    """CLIPTextEncode "text": the literal prompt, or the ClioStyle node's styled_prompt output."""
    return [g.add("ClioStyle", prompt=prompt, style=clio, template=CLIO_TEMPLATE), 0] if clio else prompt


KREA2_DEPTH = "krea2_depth_control_lora.safetensors"  # Patil/Krea-2-depth-controlnet, via comfyui-krea2-controlnet
ZIMAGE = {"unet": "z_image_turbo_bf16.safetensors", "clip": "qwen_3_4b.safetensors", "vae": "ae.safetensors",
          "patch": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors", "steps": 8}


def krea2_stage(g: Graph, prompt: str, w: int, h: int, seed: int, clio: str | None = None,
                control: list | None = None, strength: float = 1.0) -> list:
    """Krea-2 Turbo text-to-image; with `control` (an IMAGE link, a depth map with near = white) the depth Control
    LoRA locks the layout. Returns the decoded IMAGE link."""
    unet = [g.add("UNETLoader", unet_name=KREA2["unet"], weight_dtype="default"), 0]
    clip = g.add("CLIPLoader", clip_name=KREA2["clip"], type="krea2", device="default")
    vae = g.add("VAELoader", vae_name=KREA2["vae"])
    pos = g.add("CLIPTextEncode", clip=[clip, 0], text=text_input(g, prompt, clio))
    neg = g.add("ConditioningZeroOut", conditioning=[pos, 0])
    latent = g.add("EmptyLatentImage", width=w, height=h, batch_size=1)
    if control:
        unet = [g.add("Krea2ControlLoRALoader", model=unet, lora_name=KREA2_DEPTH, strength=strength), 0]
        clat = g.add("Krea2ControlImageEncode", control_image=control, vae=[vae, 0], resize="match_latent_size",
                     upscale_method="lanczos", crop="center", channel_mode="grayscale", normalize="per_image_minmax",
                     invert=False, batch_mode="independent_images", latent=[latent, 0])
        unet = [g.add("Krea2ControlApply", model=unet, control_latent=[clat, 0]), 0]
    out = g.add("KSampler", model=unet, positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                seed=seed, steps=KREA2["steps"], cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    return [g.add("VAEDecode", samples=[out, 0], vae=[vae, 0]), 0]


def zimage_stage(g: Graph, prompt: str, w: int, h: int, seed: int, clio: str | None, control: list,
                 strength: float = 0.9) -> list:
    """Z-Image-Turbo with the Fun ControlNet Union on a depth map. Returns the decoded IMAGE link."""
    unet = g.add("UNETLoader", unet_name=ZIMAGE["unet"], weight_dtype="default")
    model = g.add("ModelSamplingAuraFlow", model=[unet, 0], shift=3)
    clip = g.add("CLIPLoader", clip_name=ZIMAGE["clip"], type="lumina2", device="default")
    vae = g.add("VAELoader", vae_name=ZIMAGE["vae"])
    patch = g.add("ModelPatchLoader", name=ZIMAGE["patch"])
    model = g.add("QwenImageDiffsynthControlnet", model=[model, 0], model_patch=[patch, 0], vae=[vae, 0],
                  image=control, strength=strength)
    pos = g.add("CLIPTextEncode", clip=[clip, 0], text=text_input(g, prompt, clio))
    neg = g.add("ConditioningZeroOut", conditioning=[pos, 0])
    latent = g.add("EmptySD3LatentImage", width=w, height=h, batch_size=1)
    out = g.add("KSampler", model=[model, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[latent, 0],
                seed=seed, steps=ZIMAGE["steps"], cfg=1.0, sampler_name="res_multistep", scheduler="simple",
                denoise=1.0)
    return [g.add("VAEDecode", samples=[out, 0], vae=[vae, 0]), 0]


def klein_stage(g: Graph, prompt: str, w: int, h: int, seed: int, refs: list[str], clio: str | None = None,
                init: list | None = None, denoise: float = 1.0) -> list:
    """Flux.2 Klein with reference latents; `init` (an IMAGE link) makes it img2img at `denoise`."""
    unet = g.add("UNETLoader", unet_name=KLEIN["unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=KLEIN["clip"], type="flux2", device="default")
    vae = g.add("VAELoader", vae_name=KLEIN["vae"])
    pos = [g.add("CLIPTextEncode", clip=[clip, 0], text=text_input(g, prompt, clio)), 0]
    neg = [g.add("ConditioningZeroOut", conditioning=pos), 0]
    for name in refs:
        img = g.add("LoadImage", image=name)
        scaled = g.add("ImageScaleToTotalPixels", image=[img, 0], upscale_method="lanczos", megapixels=1.0,
                       resolution_steps=1)
        lat = [g.add("VAEEncode", pixels=[scaled, 0], vae=[vae, 0]), 0]
        pos = [g.add("ReferenceLatent", conditioning=pos, latent=lat), 0]
        neg = [g.add("ReferenceLatent", conditioning=neg, latent=lat), 0]
    if init:
        # img2img: start from the init picture so camera and placement survive; stretch the schedule so the kept
        # tail still has KLEIN["steps"] steps.
        scaled = g.add("ImageScale", image=init, upscale_method="lanczos", width=w, height=h, crop="center")
        latent = g.add("VAEEncode", pixels=[scaled, 0], vae=[vae, 0])
        full = g.add("Flux2Scheduler", steps=math.ceil(KLEIN["steps"] / denoise), width=w, height=h)
        sigmas = [g.add("SplitSigmasDenoise", sigmas=[full, 0], denoise=denoise), 1]
    else:
        latent = g.add("EmptyFlux2LatentImage", width=w, height=h, batch_size=1)
        sigmas = [g.add("Flux2Scheduler", steps=KLEIN["steps"], width=w, height=h), 0]
    guider = g.add("CFGGuider", model=[unet, 0], positive=pos, negative=neg, cfg=1.0)
    noise = g.add("RandomNoise", noise_seed=seed)
    sampler = g.add("KSamplerSelect", sampler_name="euler")
    out = g.add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0],
                sigmas=sigmas, latent_image=[latent, 0])
    return [g.add("VAEDecode", samples=[out, 0], vae=[vae, 0]), 0]


def build_krea2(prompt: str, w: int, h: int, seed: int, prefix: str, clio: str | None = None) -> dict:
    g = Graph()
    g.add("SaveImage", images=krea2_stage(g, prompt, w, h, seed, clio), filename_prefix=prefix)
    return g.nodes


def build_klein(prompt: str, w: int, h: int, seed: int, refs: list[str], prefix: str,
                clio: str | None = None, init: str | None = None, denoise: float = 1.0) -> dict:
    g = Graph()
    link = [g.add("LoadImage", image=init), 0] if init else None
    g.add("SaveImage", images=klein_stage(g, prompt, w, h, seed, refs, clio, link, denoise), filename_prefix=prefix)
    return g.nodes


def build_controlled(prompt: str, w: int, h: int, seed: int, refs: list[str], prefix: str, clio: str | None,
                     control: dict, control_image: str) -> dict:
    """Depth-locked keyframe: a Krea-2 (or Z-Image) pass on the control depth map fixes camera, poses, aim and
    placement; with refs, a Klein img2img pass at control["refine"] denoise then puts the characters on-model."""
    g = Graph()
    depth = [g.add("LoadImage", image=control_image), 0]
    stage = krea2_stage if control.get("engine", "krea2") == "krea2" else zimage_stage
    image = stage(g, prompt, w, h, seed, clio, depth, control.get("strength", 1.0 if stage is krea2_stage else 0.9))
    if refs and control.get("refine", 0.5) > 0:
        image = klein_stage(g, control.get("refine_prompt", prompt), w, h, seed, refs, clio, image,
                            control.get("refine", 0.5))
    g.add("SaveImage", images=image, filename_prefix=prefix)
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
        for r in [*stills[sid].get("refs", []), stills[sid].get("init") or ""]:
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
        init = spec.get("init")
        init = (out_dir / f"{init[1:]}.png" if init.startswith("@") else root / init) if init else None
        if init and not init.exists():
            raise SystemExit(f"{sid}: missing init {init}")
        denoise = spec.get("denoise", 0.85)
        prompt = f"{style}\n\n{spec['prompt']}".strip() if style else spec["prompt"]
        w, h = spec.get("width", film.get("width", 1024)), spec.get("height", film.get("height", 1024))
        # Keyed off the still id, so adding stills doesn't re-roll the others.
        seed = spec.get("seed", film.get("seed", 0) + zlib.crc32(sid.encode()) % 100_000)
        control = spec.get("control")
        control_img = root / control["image"] if control else None
        if control_img and not control_img.exists():
            raise SystemExit(f"{sid}: missing control image {control_img}")
        engine = "controlled" if control else spec.get("engine", "klein" if refs or init else "krea2")
        if engine == "krea2" and (refs or init):
            raise SystemExit(f"{sid}: krea2 takes no refs or init; use engine klein")
        parts = [prompt, w, h, seed, engine, [hashlib.sha1(r.read_bytes()).hexdigest() for r in refs]]
        if init:  # only when set, so existing stills keep their digest
            parts.append(["init", hashlib.sha1(init.read_bytes()).hexdigest(), denoise])
        if control:
            parts.append(["control", control, hashlib.sha1(control_img.read_bytes()).hexdigest()])
        clio = spec.get("clio_style", film.get("still_clio_style"))
        clio = None if clio in (None, "none") else clio
        if clio:
            try:
                prose = http_json(f"/clio_style/prose?style={urllib.parse.quote(clio)}", base=AUX_URL).get("prose", "")
            except RuntimeError:
                prose = ""
            if not prose:
                raise SystemExit(f"{sid}: unknown Clio style {clio!r} (is clio-style-preview installed in ComfyUI?)")
            parts.append([clio, CLIO_TEMPLATE, prose])  # only when set, so unstyled stills keep their digest
        digest = hashlib.sha1(json.dumps(parts).encode()).hexdigest()
        sidecar = dest.with_suffix(".json")
        if not a.force and dest.exists() and sidecar.exists() and json.loads(sidecar.read_text())["digest"] == digest:
            print(f"= {sid}: up to date")
            continue
        prefix = f"h3film/{root.name}/stills/{sid}"
        if engine == "controlled":
            graph = build_controlled(prompt, w, h, seed, [upload_image(r, base=AUX_URL) for r in refs], prefix, clio,
                                     control, upload_image(control_img, base=AUX_URL))
        elif engine == "klein":
            graph = build_klein(prompt, w, h, seed, [upload_image(r, base=AUX_URL) for r in refs], prefix, clio,
                                upload_image(init, base=AUX_URL) if init else None, denoise)
        else:
            graph = build_krea2(prompt, w, h, seed, prefix, clio)
        t0 = time.time()
        resp = http_json("/prompt", {"prompt": graph, "client_id": "keyframes"}, base=AUX_URL)
        if resp.get("node_errors"):
            raise RuntimeError(f"{sid}: {json.dumps(resp['node_errors'])[:3000]}")
        entry = wait_for(resp["prompt_id"], sid, base=AUX_URL)
        item = next(img for o in entry["outputs"].values() for img in o.get("images", []))
        dest.write_bytes(view(item, AUX_URL))
        sidecar.write_text(json.dumps({"digest": digest, "seed": seed, "prompt": f"[{clio}] {prompt}" if clio else prompt},
                                      indent=2), encoding="utf-8")
        print(f"\r+ {sid}: {dest} ({time.time() - t0:.0f}s)          ")


if __name__ == "__main__":
    sys.exit(main())
