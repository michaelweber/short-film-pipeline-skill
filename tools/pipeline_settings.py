"""Machine-specific settings for the pipeline tools.

    from pipeline_settings import setting
    py = setting("py_torch", sys.executable)

Resolution order for a key: the environment variable KEY.upper() (e.g. PY_TORCH), then tools/local_settings.json
(untracked; this machine's values), then the caller's default. Keys used by the tools:
  py_torch                Python with torch + transformers (+ librosa): ASR, voice similarity, CLAP
  aux_gpu_uuid            nvidia-smi UUID of the aux card (stills, TTS/ASR/CLAP, Blender)
  h3_gpu_uuid             nvidia-smi UUID of the H3 video card
  gpu_lease_dir           directory for the cross-process GPU/CPU leases (tools/gpu_queue.py)
  resolve_project_prefix  prefix of every Resolve project name (a film's own "project_prefix" wins)
  resolve_script_lib      path of DaVinci Resolve's fusionscript.dll / .so
  audiocpp_dir            audio.cpp release folder (audiocpp_cli + models/): narrate.py's audio.cpp engines

Example tools/local_settings.json:
  {"py_torch": "C:/ComfyUI_windows_portable/python_embeded/python.exe",
   "aux_gpu_uuid": "GPU-xxxxxxxx-...", "h3_gpu_uuid": "GPU-yyyyyyyy-...", "resolve_project_prefix": "myfilms_"}

Stdlib only, no side effects beyond reading that file once (safe to import inside Blender's Python).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

LOCAL_SETTINGS = Path(__file__).resolve().parent / "local_settings.json"
_cache: dict | None = None


def _local() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(LOCAL_SETTINGS.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _cache = {}
    return _cache


def setting(key: str, default=None):
    """The value of `key`: env var key.upper(), else tools/local_settings.json, else `default`."""
    env = os.environ.get(key.upper())
    if env:
        return env
    return _local().get(key, default)
