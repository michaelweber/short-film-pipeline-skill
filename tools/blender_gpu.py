"""Render Cycles on a chosen NVIDIA card (OptiX) instead of the CPU, without editing the calling script.

Pre-hook usage (existing scripts stay untouched):

    blender -b scene.blend --python tools/blender_gpu.py --python existing_script.py -- args
    set BLENDER_GPU_CARD=h3 & blender -b ... --python tools/blender_gpu.py --python script.py -- args

Cards (matched by device name inside Blender, by UUID for nvidia-smi; never by CUDA index), from the
settings in tools/pipeline_settings.py (env var or tools/local_settings.json):
    aux (default)  name substring aux_gpu_name, nvidia-smi UUID aux_gpu_uuid
    h3             name substring h3_gpu_name,  nvidia-smi UUID h3_gpu_uuid
An unset name matches every device of the backend; an unset UUID skips the memory log/guard.

The hook enables only the selected card's Cycles devices (CPU deselected), sets
scene.cycles.device='GPU' and prints ``BLENDER_GPU devices: [...]``. Persistent render_init /
render_complete / render_cancel handlers switch the scene back to the GPU right before every render,
so a script that later sets scene.cycles.device='CPU' (or opens another .blend) still renders on the
card. Before each render the card's memory is logged from nvidia-smi.

Guards (all exit or raise; nothing ever falls back to the CPU):
    - no matching device: process exits with code 3 ("BLENDER_GPU FATAL ...").
    - h3: refuses (exit 3) when the h3 card has under 8 GB free, so H3 jobs are not starved.
    - aux: the 60 % VRAM budget is enforced by gpu_queue's lease; usage is logged per render.
    - CUDA/OptiX out-of-memory (or any GPU render error) raises RuntimeError("BLENDER_GPU OOM ...")
      from bpy.ops.render.render and makes Blender exit with code 4.
If tools/gpu_queue.py is importable, each render holds ``gpu_lease('blender-<card>')``.

Memory savers applied per render: use_persistent_data off for single-frame renders; the render
texture limit is capped at 8192 px only when the scene holds a larger image.

Environment knobs:
    BLENDER_GPU_CARD     aux (default) or h3
    BLENDER_GPU_MATCH    override the device-name substring
    BLENDER_GPU_BACKEND  OPTIX (default) or CUDA
    BLENDER_GPU_DENOISE  1 (default) = OptiX denoiser on the GPU; 0 = leave denoiser settings as set
    BLENDER_GPU_LEASE    1 (default) = use the gpu_queue lease when available; 0 = never lease

Library usage inside Blender:

    sys.path.insert(0, '<repo>/tools'); import blender_gpu
    blender_gpu.enable_gpu(scene, card='aux')   # -> ['NVIDIA GeForce RTX ...']
    blender_gpu.install_render_hooks()          # optional: re-assert GPU + guards around every render

CPU and GPU Cycles output differ slightly per pixel: re-baseline pixel-hash evidence after switching.
"""
import atexit
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import bpy
from bpy.app.handlers import persistent

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from pipeline_settings import setting

CARDS = {card: {'match': setting(f'{card}_gpu_name', ''), 'uuid': setting(f'{card}_gpu_uuid')} for card in ('aux', 'h3')}
H3_MIN_FREE_MIB = 8 * 1024
AUX_BUDGET_FRACTION = 0.60
TEXTURE_LIMIT_PX = 8192
OOM_MARKERS = ('out of memory', 'out_of_memory', 'cuda_error', 'optix_error', 'cumemalloc', 'illegal address')

_CONFIG = {
    'card': os.environ.get('BLENDER_GPU_CARD', 'aux').lower(),
    'device_name_contains': os.environ.get('BLENDER_GPU_MATCH') or None,
    'backend': os.environ.get('BLENDER_GPU_BACKEND', 'OPTIX').upper(),
    'denoise': os.environ.get('BLENDER_GPU_DENOISE', '1') != '0',
}
_USE_LEASE = os.environ.get('BLENDER_GPU_LEASE', '1') != '0'
_lease_stack = None
_gpu_failed = False


def enable_gpu(scene=None, device_name_contains=None, backend='OPTIX', denoise=True, card='aux'):
    """Select only the card's `backend` devices, set `scene` to GPU, return enabled device names.

    `card` is 'aux' or 'h3' (see CARDS); `device_name_contains` overrides its name
    match. Raises RuntimeError when no device matches; the CPU device is always deselected.
    denoise=True switches the (already enabled or disabled) denoiser to OptiX on the GPU; it does
    not turn denoising on for scenes that render without it.
    """
    if card not in CARDS:
        raise ValueError(f'BLENDER_GPU unknown card {card!r}; expected one of {sorted(CARDS)}')
    match = device_name_contains or CARDS[card]['match']
    scene = scene or bpy.context.scene
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = backend
    prefs.refresh_devices()
    enabled = []
    for device in prefs.devices:
        device.use = device.type == backend and match in device.name
        if device.use:
            enabled.append(device.name)
    if not enabled:
        available = [f'{d.name} ({d.type})' for d in prefs.devices]
        raise RuntimeError(f'BLENDER_GPU no {backend} device matching {match!r} for card {card!r}; available: {available}')
    scene.cycles.device = 'GPU'
    if denoise:
        scene.cycles.denoising_use_gpu = True
        if backend == 'OPTIX':
            scene.cycles.denoiser = 'OPTIX'
    return enabled


def card_memory(card='aux'):
    """Return (used_mib, total_mib) of the card from nvidia-smi, queried by UUID."""
    if not CARDS[card]['uuid']:
        raise RuntimeError(f'no {card}_gpu_uuid setting')
    out = subprocess.run(['nvidia-smi', '-i', CARDS[card]['uuid'], '--query-gpu=memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    used, total = (int(v) for v in out.stdout.strip().split(','))
    return used, total


def _check_memory(card, where):
    """Log card memory; refuse (exit 3) to start an h3 render with under 8 GB free."""
    try:
        used, total = card_memory(card)
    except Exception as error:  # noqa: BLE001 - the log line is informational; the lease still guards
        print(f'BLENDER_GPU memory {where} card={card} unavailable: {error}', flush=True)
        return
    free = total - used
    line = f'BLENDER_GPU memory {where} card={card} used={used} MiB total={total} MiB free={free} MiB ({used / total:.0%} used)'
    if card == 'aux':
        line += f' budget={int(total * AUX_BUDGET_FRACTION)} MiB'
    print(line, flush=True)
    if card == 'h3' and free < H3_MIN_FREE_MIB:
        _fatal(f'h3 card has {free} MiB free (< {H3_MIN_FREE_MIB} MiB); refusing to render so H3 jobs are not starved')


def _fatal(error):
    print(f'BLENDER_GPU FATAL {error}', flush=True)
    sys.stderr.flush()
    _release_lease()
    os._exit(3)  # Blender swallows handler/--python exceptions and would keep rendering on CPU.


def _release_lease():
    global _lease_stack
    if _lease_stack is not None:
        stack, _lease_stack = _lease_stack, None
        stack.close()
        print('BLENDER_GPU lease released', flush=True)


def _acquire_lease(card):
    global _lease_stack
    if not _USE_LEASE or _lease_stack is not None:
        return
    try:
        from gpu_queue import gpu_lease
    except ImportError:
        return
    stack = contextlib.ExitStack()
    stack.enter_context(gpu_lease(f'blender-{card}', label=f'blender_gpu pid {os.getpid()}'))
    _lease_stack = stack
    print(f'BLENDER_GPU lease acquired blender-{card}', flush=True)


def _apply_memory_savers(scene, animation):
    if not animation and scene.render.use_persistent_data:
        scene.render.use_persistent_data = False
        print('BLENDER_GPU persistent_data off (single-frame render)', flush=True)
    largest = max((max(image.size[:]) for image in bpy.data.images if image.size[0]), default=0)
    if largest > TEXTURE_LIMIT_PX and scene.cycles.texture_limit_render == 'OFF':
        scene.cycles.texture_limit_render = str(TEXTURE_LIMIT_PX)
        print(f'BLENDER_GPU texture_limit_render={TEXTURE_LIMIT_PX} (largest image {largest} px)', flush=True)


@persistent
def _blender_gpu_render_init(scene, *_):
    if scene.render.engine != 'CYCLES':
        return
    card = _CONFIG['card']
    try:
        enabled = enable_gpu(scene, **_CONFIG)
    except Exception as error:  # noqa: BLE001 - any failure must stop the render, not degrade to CPU
        _fatal(error)
    _acquire_lease(card)
    _check_memory(card, 'pre-render')
    print(f'BLENDER_GPU render_init scene={scene.name!r} device={scene.cycles.device} devices: {enabled}', flush=True)


@persistent
def _blender_gpu_render_done(*_):
    _release_lease()


class _RenderOpsProxy:
    """Stands in for bpy.ops.render: wraps render() with memory savers and OOM detection."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def render(self, *args, **kwargs):
        global _gpu_failed
        scene = bpy.context.scene
        on_gpu = scene.render.engine == 'CYCLES'
        if on_gpu:
            _apply_memory_savers(scene, kwargs.get('animation', False))
        try:
            result = self._real.render(*args, **kwargs)
        except RuntimeError as error:
            _release_lease()
            if on_gpu and any(marker in str(error).lower() for marker in OOM_MARKERS):
                _gpu_failed = True
                _check_memory(_CONFIG['card'], 'after-failure')
                raise RuntimeError(f'BLENDER_GPU OOM/GPU error on card {_CONFIG["card"]!r}; no CPU fallback: {error}') from error
            raise
        if on_gpu and 'CANCELLED' in result:
            _release_lease()
            _gpu_failed = True
            raise RuntimeError(f'BLENDER_GPU render cancelled on card {_CONFIG["card"]!r} (GPU error or out of memory; '
                               'see the Cycles log above); no CPU fallback')
        return result


def install_render_hooks():
    """Register persistent handlers and the render() wrapper that guard every render."""
    handlers = bpy.app.handlers
    for handler_list, function in ((handlers.render_init, _blender_gpu_render_init),
                                   (handlers.render_complete, _blender_gpu_render_done),
                                   (handlers.render_cancel, _blender_gpu_render_done)):
        for existing in [h for h in handler_list if getattr(h, '__name__', '') == function.__name__]:
            handler_list.remove(existing)
        handler_list.append(function)
    if not isinstance(bpy.ops.render, _RenderOpsProxy):
        bpy.ops.render = _RenderOpsProxy(bpy.ops.render)
    atexit.register(_at_exit)


def _at_exit():
    _release_lease()
    if _gpu_failed:
        print('BLENDER_GPU exiting with code 4 after a GPU render failure', flush=True)
        sys.stdout.flush()
        os._exit(4)


def _main():
    card = _CONFIG['card']
    try:
        enabled = enable_gpu(bpy.context.scene, **_CONFIG)
    except Exception as error:  # noqa: BLE001
        _fatal(error)
    _check_memory(card, 'startup')
    install_render_hooks()
    print(f'BLENDER_GPU devices: {enabled}', flush=True)


if __name__ == '__main__':
    _main()
