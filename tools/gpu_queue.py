"""Direct GPU job runner for the two ComfyUI instances, plus a cross-process Blender GPU lease.

Workers submit their own jobs instead of routing through Main. ComfyUI queues jobs per instance;
this tool only POSTs, waits, fetches and records.

Cards:
  h3  -> H3_URL  (env H3_COMFY_URL,  default http://127.0.0.1:8188, the big card): H3 video; also the escape hatch
         for aux jobs too big for the aux budget (e.g. large Flux stills)
  aux -> AUX_URL (env AUX_COMFY_URL, default http://127.0.0.1:8189, the aux card): stills, depth, TTS/ASR/CLAP

Aux VRAM budget: the aux card also drives the desktop/remote access, so total use stays under AUX_VRAM_BUDGET_FRAC
(default 0.60) of its memory. `submit --card aux` waits (polling nvidia-smi every 5 s, logging to stderr) while
used + --est-gb (default 8) exceeds the budget; blender-aux leases do the same with est 4 GB. After each aux job
(`submit --wait`, `wait`), if the aux card is above AUX_FREE_ABOVE_FRAC (default 0.50) used, POST
{"unload_models":true,"free_memory":true} to the aux /free.

Subcommands:
  submit WORKFLOW.api.json --card h3|aux [--wait] [--out-dir DIR] [--label TEXT] [--priority front] [--est-gb G]
  wait PROMPT_ID --card h3|aux [--out-dir DIR]
  status
  lease blender-aux|blender-h3|NAME [--slots N] [--est-gb G] [--label TEXT] -- COMMAND...
    blender-aux (alias blender): 1 slot, aux budget admission (est 4 GB)
    blender-h3: 2 slots, needs est 8 GB free on the h3 card
    blender-cpu: machine-wide CPU Blender pool, BLENDER_CPU_SLOTS slots (default 16), no VRAM check;
      tools/blender_shard.py takes one per shard. `status` lists holders and the live blender.exe count.

Result JSON (submit --wait / wait): {prompt_id, card, status, outputs:[paths], files:[{path, node_id,
class_type, type, output_node}], wall_s, queue_wait_s, run_s}. `files` names the node that produced each
file: ComfyUI also echoes input media (e.g. a LoadVideo preview, type "input"/"temp") and its output order
is not stable, so select the real render by node_id/class_type, never by index.

Ledger: <lease dir>/ledger.jsonl, one JSON row per event: an "submit" row when a job is queued and an
"end" row (status success|error, submit/start/end timestamps from /history, outputs) when it is fetched.
Rows share prompt_id; the latest row per prompt_id is the job's state.

Lease: `gpu_lease(name, slots)` is a counting semaphore over slot files <lease dir>/<name>/slot-<i>.lock.
A holder keeps an OS byte-lock (msvcrt on Windows, fcntl elsewhere) on one slot file and writes its pid to
slot-<i>.json. The OS drops the lock when the holder dies, so a crashed holder's slot frees itself; the stale
pid file is reaped by the next acquirer or `status`. The `lease` CLI puts its child in a Windows job object
with kill-on-close, so killing the lease process also kills the command it guards.

Lease dir: setting gpu_lease_dir (env GPU_LEASE_DIR or tools/local_settings.json, see tools/pipeline_settings.py),
default <repo>/.gpu_leases. Cards are matched in nvidia-smi by the aux_gpu_uuid / h3_gpu_uuid settings (or, as a
fallback, a substring of the device name from aux_gpu_name / h3_gpu_name); unmatched cards skip VRAM admission.
Stdlib only, no top-level side effects (safe to import inside Blender's Python).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from pipeline_settings import setting

LEASE_DIR = Path(setting("gpu_lease_dir", Path(__file__).resolve().parent.parent / ".gpu_leases"))
CARDS = ("h3", "aux")
_WIN = os.name == "nt"


# ---- instance routing ---------------------------------------------------------------------------
def card_url(card: str) -> str:
    """Resolves lazily: tools/h3_render.py owns H3_URL/AUX_URL; fall back to the same env-var logic."""
    if card not in CARDS:
        raise ValueError(f"unknown card {card!r}; expected one of {CARDS}")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            import h3_render  # noqa: PLC0415
        finally:
            sys.path.pop(0)
        url = getattr(h3_render, "H3_URL" if card == "h3" else "AUX_URL")
    except Exception:
        url = (os.environ.get("H3_COMFY_URL", "http://127.0.0.1:8188") if card == "h3"
               else os.environ.get("AUX_COMFY_URL", "http://127.0.0.1:8189"))
    return url.rstrip("/")


def http_json(base: str, path: str, payload: dict | None = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(base + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"{base}{path}: HTTP {e.code}: {body[:4000]}") from None


# ---- small file-lock helpers -------------------------------------------------------------------
def _try_lock(fh) -> bool:
    try:
        fh.seek(0)
        if _WIN:
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh) -> None:
    try:
        fh.seek(0)
        if _WIN:
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def _file_lock(path: Path, poll: float = 0.05):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    try:
        while not _try_lock(fh):
            time.sleep(poll)
        try:
            yield
        finally:
            _unlock(fh)
    finally:
        fh.close()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _WIN:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5  # access denied => exists
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# ---- ledger ------------------------------------------------------------------------------------
def ledger_path() -> Path:
    return LEASE_DIR / "ledger.jsonl"


def ledger_append(row: dict) -> None:
    row = {"iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _file_lock(LEASE_DIR / "ledger.lock"):
        with open(ledger_path(), "a", encoding="utf-8") as f:
            f.write(line)


def ledger_jobs(tail_bytes: int = 4 << 20) -> dict[str, dict]:
    """Latest merged row per prompt_id (earlier fields kept unless overwritten)."""
    p = ledger_path()
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        size = f.seek(0, 2)
        f.seek(max(0, size - tail_bytes))
        data = f.read().decode("utf-8", "replace").splitlines()
    if size > tail_bytes:
        data = data[1:]
    jobs: dict[str, dict] = {}
    for line in data:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        pid = row.get("prompt_id")
        if pid:
            jobs.setdefault(pid, {}).update(row)
    return jobs


# ---- ComfyUI jobs ------------------------------------------------------------------------------
def load_api_workflow(path: Path) -> dict:
    try:
        wf = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"gpu_queue: cannot read workflow {path}: {e}")
    if isinstance(wf, dict) and "nodes" in wf and "links" in wf:
        raise SystemExit(f"gpu_queue: {path} is a UI-format workflow (has 'nodes'/'links'); "
                         "export it with 'Save (API)' or convert it to API format first")
    if not isinstance(wf, dict) or not wf:
        raise SystemExit(f"gpu_queue: {path} is not an API-format workflow (expected a non-empty "
                         "{node_id: {class_type, inputs}} object)")
    bad = [k for k, v in wf.items()
           if not (isinstance(v, dict) and isinstance(v.get("class_type"), str) and isinstance(v.get("inputs"), dict))]
    if bad:
        raise SystemExit(f"gpu_queue: {path} is not an API-format workflow: node(s) {bad[:10]} lack "
                         "'class_type'/'inputs'")
    return wf


def submit(workflow: dict, card: str, label: str = "", front: bool = False, source: str = "",
           est_gb: float | None = None, admit_timeout: float | None = None) -> dict:
    """POSTs to the card; for aux, first waits for the aux VRAM budget to admit `est_gb` (default
    EST_GB_AUX_JOB; 0 skips)."""
    base = card_url(card)
    admit = 0.0
    est_gb = EST_GB_AUX_JOB if est_gb is None else est_gb
    if card == "aux" and est_gb:
        admit = wait_vram("aux", est_gb, f"submit {label or source}",
                          None if admit_timeout is None else time.time() + admit_timeout)
    payload = {"prompt": workflow, "client_id": f"gpu_queue-{uuid.uuid4().hex[:12]}"}
    if front:
        payload["front"] = True
    t = time.time()
    resp = http_json(base, "/prompt", payload)
    if resp.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected the workflow: {json.dumps(resp['node_errors'])[:4000]}")
    row = {"event": "submit", "prompt_id": resp["prompt_id"], "number": resp.get("number"), "card": card,
           "url": base, "label": label, "status": "queued", "submit_ts": round(t, 3), "front": front,
           "workflow": source, "pid": os.getpid(), "admit_wait_s": round(admit, 3)}
    ledger_append(row)
    return row


def _history_times(entry: dict) -> tuple[float | None, float | None]:
    start = end = None
    for msg in entry.get("status", {}).get("messages", []):
        if not (isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict)):
            continue
        kind, data = msg
        ts = data.get("timestamp")
        if ts is None:
            continue
        if kind == "execution_start":
            start = ts / 1000
        elif kind in ("execution_success", "execution_error", "execution_interrupted"):
            end = ts / 1000
    return start, end


def _error_text(entry: dict) -> str | None:
    for msg in entry.get("status", {}).get("messages", []):
        if isinstance(msg, list) and msg and msg[0] in ("execution_error", "execution_interrupted"):
            d = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
            return f"{msg[0]}: node {d.get('node_id')} {d.get('node_type')}: {d.get('exception_message', '')}".strip()
    return None


def wait_history(base: str, prompt_id: str, timeout: float | None = None, poll: float = 1.0) -> dict:
    t0 = time.time()
    missing = 0
    down_since = None
    while True:
        try:
            hist = http_json(base, f"/history/{prompt_id}")
            entry = hist.get(prompt_id)
            if entry and entry.get("status", {}).get("completed") is not None and (
                    entry["status"].get("completed") or entry["status"].get("status_str") == "error"):
                return entry
            if not entry:
                q = http_json(base, "/queue")
                ids = {item[1] for item in q.get("queue_running", []) + q.get("queue_pending", [])}
                missing = 0 if prompt_id in ids else missing + 1
                if missing >= 5:
                    raise RuntimeError(f"prompt {prompt_id} is neither queued nor in history at {base}")
            down_since = None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:  # instance busy/restarting
            down_since = down_since or time.time()
            if time.time() - down_since > 300:
                raise RuntimeError(f"{base} unreachable for 300 s while waiting on {prompt_id}: {e}") from None
        if timeout is not None and time.time() - t0 > timeout:
            raise TimeoutError(f"prompt {prompt_id} not finished after {timeout}s")
        time.sleep(poll)


def fetch_outputs(base: str, entry: dict, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = entry.get("prompt") or []
    graph = prompt[2] if len(prompt) > 2 and isinstance(prompt[2], dict) else {}
    output_nodes = set(prompt[4]) if len(prompt) > 4 and isinstance(prompt[4], list) else set()
    files, used = [], set()
    for node_id, node_out in entry.get("outputs", {}).items():
        for kind, items in node_out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not (isinstance(item, dict) and item.get("filename")):
                    continue
                name = Path(item["filename"]).name
                if name.lower() in used:
                    name = f"{node_id}_{name}"
                used.add(name.lower())
                q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                            "type": item.get("type", "output")})
                dest = out_dir / name
                with urllib.request.urlopen(f"{base}/view?{q}", timeout=600) as r, open(dest, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                files.append({"path": str(dest).replace("\\", "/"), "node_id": node_id,
                              "class_type": graph.get(node_id, {}).get("class_type"), "kind": kind,
                              "type": item.get("type", "output"), "output_node": node_id in output_nodes,
                              "subfolder": item.get("subfolder", ""), "filename": item["filename"]})
    return files


def finish(prompt_id: str, card: str, out_dir: Path | None, timeout: float | None = None,
           submit_ts: float | None = None, label: str | None = None) -> dict:
    base = card_url(card)
    t_wait = time.time()
    entry = wait_history(base, prompt_id, timeout)
    prior = ledger_jobs().get(prompt_id, {})
    submit_ts = submit_ts or prior.get("submit_ts")
    label = label if label is not None else prior.get("label", "")
    out_dir = Path(out_dir or (LEASE_DIR / "outputs" / prompt_id)).resolve()
    files = fetch_outputs(base, entry, out_dir)
    start, end = _history_times(entry)
    end = end or time.time()
    status = entry.get("status", {}).get("status_str") or ("success" if entry.get("status", {}).get("completed") else "error")
    r3 = (lambda x: None if x is None else round(x, 3))
    result = {
        "prompt_id": prompt_id, "card": card, "status": status, "label": label,
        "outputs": [f["path"] for f in files], "files": files,
        "wall_s": r3(end - (submit_ts or t_wait)),
        "queue_wait_s": r3(start - submit_ts) if (start and submit_ts) else None,
        "run_s": r3(end - start) if start else None,
    }
    err = _error_text(entry)
    if err:
        result["error"] = err
    if card == "aux":
        try:
            result["freed"] = free_aux_if_high()
        except Exception as e:  # noqa: BLE001 - freeing is best effort; the job result stands
            result["freed"] = {"error": str(e)}
    ledger_append({"event": "end", "prompt_id": prompt_id, "card": card, "url": base, "label": label,
                   "status": status, "submit_ts": r3(submit_ts), "start_ts": r3(start), "end_ts": r3(end),
                   "queue_wait_s": result["queue_wait_s"], "run_s": result["run_s"], "wall_s": result["wall_s"],
                   "outputs": files, "freed": result.get("freed"), **({"error": err} if err else {})})
    return result


# ---- VRAM admission ----------------------------------------------------------------------------
GPU_IDS = {card: (setting(f"{card}_gpu_uuid"), setting(f"{card}_gpu_name")) for card in ("aux", "h3")}
VRAM_POLL_S = 5.0
AUX_FREE_ABOVE_FRAC = float(os.environ.get("AUX_FREE_ABOVE_FRAC", "0.50"))  # POST /free after aux jobs above this
EST_GB_AUX_JOB = 8.0         # default headroom for an aux ComfyUI submit

# Lease profiles: card whose VRAM gates admission, default slots, default est_gb.
# aux: wait while used + est > AUX_VRAM_BUDGET_FRAC * total.  h3: wait while free < est.
# blender-cpu is a machine-wide CPU slot pool (no VRAM check); its width comes from env BLENDER_CPU_SLOTS (default
# 16) at construction time. Holders using a smaller explicit `slots` only contend for the first slots.
LEASE_PROFILES = {"blender-aux": ("aux", 1, 4.0), "blender-h3": ("h3", 2, 8.0), "blender-cpu": (None, None, None)}
LEASE_ALIASES = {"blender": "blender-aux"}


def blender_cpu_slots() -> int:
    return int(os.environ.get("BLENDER_CPU_SLOTS", "16"))


def aux_budget_frac() -> float:
    return float(os.environ.get("AUX_VRAM_BUDGET_FRAC", "0.60"))


def _is_card(card: str, uid: str, name: str) -> bool:
    want_uid, want_name = GPU_IDS[card]
    return bool(want_uid and uid == want_uid or want_name and want_name in name)


def card_gpu(card: str, gpus: list[dict] | None = None) -> dict | None:
    for g in gpus if gpus is not None else gpu_stats():
        if _is_card(card, g.get("uuid", ""), g.get("name", "")):
            return g
    return None


def vram_check(card: str, est_gb: float, gpus: list[dict] | None = None) -> dict:
    """{ok, used_mib, total_mib, limit_mib, reason}; ok=True when nvidia-smi is unavailable (never deadlock)."""
    g = card_gpu(card, gpus)
    if not g:
        return {"ok": True, "reason": "gpu not found via nvidia-smi; admission skipped"}
    used, total, est = g["mem_used_mib"], g["mem_total_mib"], est_gb * 1024
    if card == "aux":
        limit = aux_budget_frac() * total
        ok = used + est <= limit
        reason = f"aux used {used} MiB + est {est:.0f} MiB {'<=' if ok else '>'} budget {limit:.0f} MiB " \
                 f"({aux_budget_frac():.2f} x {total})"
    else:
        limit = total - est
        ok = used <= limit
        reason = f"h3 free {total - used} MiB {'>=' if ok else '<'} required {est:.0f} MiB"
    return {"ok": ok, "used_mib": used, "total_mib": total, "limit_mib": round(limit), "reason": reason}


def wait_vram(card: str, est_gb: float, who: str, deadline: float | None = None) -> float:
    """Blocks (polling every VRAM_POLL_S, logging to stderr) until `card` admits `est_gb`; returns seconds waited."""
    t0 = time.time()
    while True:
        c = vram_check(card, est_gb)
        if c["ok"]:
            if time.time() - t0 > 0.5:
                print(f"gpu_queue: {who}: admitted after {time.time() - t0:.0f}s: {c['reason']}", file=sys.stderr,
                      flush=True)
            return time.time() - t0
        if card == "aux" and idle_free_aux(who):
            continue  # re-check right away with the freed memory
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"{who}: VRAM admission timed out: {c['reason']}")
        print(f"gpu_queue: {who}: waiting for VRAM ({time.time() - t0:.0f}s): {c['reason']}", file=sys.stderr,
              flush=True)
        time.sleep(VRAM_POLL_S)


IDLE_RESIDENT_MIB = 6 * 1024  # status: aux idle but above this -> models resident


def _post_free() -> int:
    req = urllib.request.Request(card_url("aux") + "/free",
                                 data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status


IDLE_FREE_MIN_INTERVAL_S = 60.0


def aux_queue_idle() -> bool | None:
    """True when the aux ComfyUI has nothing running or pending; None when it is unreachable."""
    try:
        q = http_json(card_url("aux"), "/queue", timeout=10)
    except Exception:  # noqa: BLE001
        return None
    return not q.get("queue_running") and not q.get("queue_pending")


def idle_free_aux(who: str) -> bool:
    """Admission would wait: if :8189 is idle, unload its resident models (POST /free), at most once per
    IDLE_FREE_MIN_INTERVAL_S machine-wide (timestamp in <lease dir>/aux_idle_free.json). Returns True if it freed."""
    stamp = LEASE_DIR / "aux_idle_free.json"
    with _file_lock(LEASE_DIR / "aux_idle_free.lock"):
        with contextlib.suppress(OSError, ValueError):
            if time.time() - json.loads(stamp.read_text(encoding="utf-8"))["ts"] < IDLE_FREE_MIN_INTERVAL_S:
                return False
        if aux_queue_idle() is not True:
            return False
        before = (card_gpu("aux") or {}).get("mem_used_mib")
        try:
            code = _post_free()
        except Exception as e:  # noqa: BLE001
            print(f"gpu_queue: {who}: aux idle but POST /free failed: {e}", file=sys.stderr, flush=True)
            return False
        stamp.write_text(json.dumps({"ts": time.time(), "by": who, "pid": os.getpid()}), encoding="utf-8")
    time.sleep(VRAM_POLL_S)
    after = (card_gpu("aux") or {}).get("mem_used_mib")
    print(f"gpu_queue: {who}: aux ComfyUI idle with models resident; POST /free -> {code}; aux used "
          f"{before} -> {after} MiB", file=sys.stderr, flush=True)
    ledger_append({"event": "idle_free", "card": "aux", "by": who, "http": code, "used_mib_before": before,
                   "used_mib_after": after})
    return True


def free_aux_if_high(gpus: list[dict] | None = None) -> dict | None:
    """POST /free (unload models) to the aux ComfyUI when the aux card is above AUX_FREE_ABOVE_FRAC."""
    g = card_gpu("aux", gpus)
    if not g or g["mem_used_mib"] <= AUX_FREE_ABOVE_FRAC * g["mem_total_mib"]:
        return None
    before = g["mem_used_mib"]
    code = _post_free()
    time.sleep(2)
    after = (card_gpu("aux") or {}).get("mem_used_mib")
    info = {"url": card_url("aux") + "/free", "http": code, "used_mib_before": before, "used_mib_after": after}
    print(f"gpu_queue: aux at {before} MiB (> {AUX_FREE_ABOVE_FRAC:.0%}); POST /free -> {code}, now {after} MiB",
          file=sys.stderr, flush=True)
    return info


# ---- leases ------------------------------------------------------------------------------------
def _slot_paths(name: str, i: int) -> tuple[Path, Path]:
    d = LEASE_DIR / name
    return d / f"slot-{i}.lock", d / f"slot-{i}.json"


class GpuLease:
    """Counting semaphore `slots` wide shared by every process using the same `name`, plus VRAM admission for
    the profiled names (blender-aux / blender-h3; 'blender' is an alias of 'blender-aux')."""

    def __init__(self, name: str, slots: int | None = None, *, timeout: float | None = None, label: str = "",
                 poll: float = 0.5, est_gb: float | None = None, cancel=None):
        name = LEASE_ALIASES.get(name, name)
        card, dslots, dest = LEASE_PROFILES.get(name, (None, 1, None))
        if name == "blender-cpu":
            dslots = blender_cpu_slots()
        slots = dslots if slots is None else slots
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.name, self.slots, self.timeout, self.label, self.poll = name, slots, timeout, label, poll
        self.card, self.est_gb = card, (dest if est_gb is None else est_gb)
        self.slot: int | None = None
        self._fh = None
        self.cancel = cancel  # optional callable; truthy -> stop waiting with InterruptedError

    def acquire(self) -> int:
        (LEASE_DIR / self.name).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        deadline = None if self.timeout is None else t0 + self.timeout
        announced = False
        while True:
            for i in range(self.slots):
                lock, meta = _slot_paths(self.name, i)
                fh = open(lock, "a+b")
                if _try_lock(fh):
                    self._fh, self.slot = fh, i
                    info = {"pid": os.getpid(), "label": self.label, "slot": i, "card": self.card,
                            "est_gb": self.est_gb}
                    if self.card and self.est_gb:
                        meta.write_text(json.dumps({**info, "state": "admitting"}), encoding="utf-8")
                        try:
                            wait_vram(self.card, self.est_gb, f"lease {self.name}", deadline)
                        except BaseException:
                            self.release()
                            raise
                    meta.write_text(json.dumps({**info, "state": "held", "acquired_ts": round(time.time(), 3),
                                                "waited_s": round(time.time() - t0, 3)}), encoding="utf-8")
                    return i
                fh.close()
            if self.cancel is not None and self.cancel():
                raise InterruptedError(f"{self.name} lease wait cancelled")
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"no {self.name} lease slot free after {self.timeout}s")
            if not announced:
                print(f"gpu_queue: waiting for a {self.name} lease ({self.slots} slots busy)", file=sys.stderr,
                      flush=True)
                announced = True
            time.sleep(self.poll)

    def release(self) -> None:
        if self._fh is None:
            return
        _, meta = _slot_paths(self.name, self.slot)
        with contextlib.suppress(OSError):
            meta.unlink()
        _unlock(self._fh)
        self._fh.close()
        self._fh = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def gpu_lease(name: str, slots: int | None = None, *, timeout: float | None = None, label: str = "",
              poll: float = 0.5, est_gb: float | None = None, cancel=None) -> GpuLease:
    """`with gpu_lease('blender-aux'):` blocks until a slot is free and the card admits est_gb of VRAM.
    Defaults per name: blender-aux (alias blender) slots=1, est 4 GB under the aux budget; blender-h3 slots=2,
    needs est 8 GB free on the h3 card; blender-cpu slots=BLENDER_CPU_SLOTS (default 16), no VRAM check; other
    names slots=1, no VRAM check. TimeoutError past timeout; InterruptedError when cancel() turns truthy."""
    return GpuLease(name, slots, timeout=timeout, label=label, poll=poll, est_gb=est_gb, cancel=cancel)


def lease_state(name: str) -> list[dict]:
    """Holders of `name` leases; reaps pid files whose lock is no longer held (crashed holders)."""
    d = LEASE_DIR / name
    out = []
    for lock in sorted(d.glob("slot-*.lock")) if d.exists() else []:
        meta = lock.with_suffix(".json")
        with open(lock, "a+b") as fh:
            free = _try_lock(fh)
            if free:
                if meta.exists():  # holder died without releasing
                    with contextlib.suppress(OSError):
                        meta.unlink()
                _unlock(fh)
                continue
        info = {}
        with contextlib.suppress(OSError, ValueError):
            info = json.loads(meta.read_text(encoding="utf-8"))
        info.update({"name": name, "slot": int(lock.stem.split("-")[1])})
        if "pid" in info:
            info["alive"] = pid_alive(int(info["pid"]))
            info["held_s"] = round(time.time() - info.get("acquired_ts", time.time()), 1)
        out.append(info)
    return out


def _kill_on_close_job():
    """Windows job object that kills its processes when the last handle (ours) closes."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None, k32
    info = EXTENDED()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    return job, k32


def run_leased(name: str, slots: int | None, cmd: list[str], label: str = "", timeout: float | None = None,
               est_gb: float | None = None) -> int:
    with gpu_lease(name, slots, timeout=timeout, label=label or " ".join(cmd)[:200], est_gb=est_gb) as lease:
        print(f"gpu_queue: {lease.name} slot {lease.slot + 1}/{lease.slots} acquired by pid {os.getpid()}",
              file=sys.stderr, flush=True)
        proc = subprocess.Popen(cmd)
        if _WIN:
            with contextlib.suppress(Exception):
                job, k32 = _kill_on_close_job()
                if job:
                    k32.AssignProcessToJobObject(job, int(proc._handle))  # job handle lives until we exit
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return proc.wait()


# ---- status ------------------------------------------------------------------------------------
def gpu_stats() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return [{"error": str(e)}]
    gpus = []
    for line in out.strip().splitlines():
        idx, name, uid, util, used, total = [s.strip() for s in line.split(",")]
        card = next((c for c in GPU_IDS if _is_card(c, uid, name)), None)
        gpus.append({"card": card, "name": name, "uuid": uid, "smi_index": int(idx), "util_pct": int(util),
                     "mem_used_mib": int(used), "mem_total_mib": int(total)})
    return gpus


def status() -> dict:
    jobs = ledger_jobs()
    cards = {}
    for card in CARDS:
        base = card_url(card)
        try:
            q = http_json(base, "/queue", timeout=10)
        except Exception as e:  # noqa: BLE001
            cards[card] = {"url": base, "error": str(e)}
            continue
        desc = (lambda item: {"prompt_id": item[1], "number": item[0],
                              "label": jobs.get(item[1], {}).get("label"),
                              "submit_ts": jobs.get(item[1], {}).get("submit_ts")})
        running = [desc(i) for i in q.get("queue_running", [])]
        pending = sorted((desc(i) for i in q.get("queue_pending", [])), key=lambda d: d["number"])
        cards[card] = {"url": base, "queue_running": len(running), "queue_pending": len(pending),
                       "running": running, "pending": pending}
    leases = {d.name: lease_state(d.name) for d in sorted(LEASE_DIR.iterdir())
              if d.is_dir() and any(d.glob("slot-*.lock"))} if LEASE_DIR.exists() else {}
    gpus = gpu_stats()
    vram = {}
    g = card_gpu("aux", gpus)
    if g:
        frac = aux_budget_frac()
        vram["aux"] = {"used_mib": g["mem_used_mib"], "total_mib": g["mem_total_mib"],
                       "used_frac": round(g["mem_used_mib"] / g["mem_total_mib"], 3), "budget_frac": frac,
                       "budget_mib": round(frac * g["mem_total_mib"]),
                       "headroom_mib": round(frac * g["mem_total_mib"] - g["mem_used_mib"]),
                       "over_budget": g["mem_used_mib"] > frac * g["mem_total_mib"]}
        aq = cards.get("aux", {})
        if "error" not in aq and not aq.get("queue_running") and not aq.get("queue_pending") \
                and g["mem_used_mib"] > IDLE_RESIDENT_MIB:
            vram["aux"]["note"] = "aux idle, models resident (admission will POST /free when it needs the memory)"
    g = card_gpu("h3", gpus)
    if g:
        vram["h3"] = {"used_mib": g["mem_used_mib"], "total_mib": g["mem_total_mib"],
                      "free_mib": g["mem_total_mib"] - g["mem_used_mib"],
                      "blender_h3_min_free_mib": round(LEASE_PROFILES["blender-h3"][2] * 1024)}
    return {"cards": cards, "vram": vram, "leases": leases, "blender_processes": blender_process_count(),
            "gpus": gpus}


def blender_process_count() -> int | None:
    """Live blender.exe processes machine-wide (tasklist); None when tasklist is unavailable."""
    if not _WIN:
        return None
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq blender.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(1 for line in out.splitlines() if line.lower().startswith('"blender.exe"'))


# ---- CLI ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpu_queue.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="POST an API-format workflow to a card's ComfyUI",
                       description="POST an API-format workflow to the card's ComfyUI queue and record it in the "
                                   "ledger. Prints {prompt_id, card, ...}; with --wait, blocks, downloads every "
                                   "output into --out-dir and prints the result JSON.")
    s.add_argument("workflow", type=Path, help="API-format workflow JSON ({node_id: {class_type, inputs}})")
    s.add_argument("--card", choices=CARDS, required=True,
                   help="h3 = :8188 big card (H3; also the escape hatch for aux jobs too big for the aux "
                        "budget, e.g. large Flux stills); aux = :8189 aux card (VRAM-budget admission)")
    s.add_argument("--est-gb", type=float, default=EST_GB_AUX_JOB,
                   help=f"aux only: VRAM headroom the job needs; waits while aux used + est > "
                        f"AUX_VRAM_BUDGET_FRAC x total (default {EST_GB_AUX_JOB:g}; 0 disables)")
    s.add_argument("--admit-timeout", type=float, help="give up waiting for VRAM admission after S seconds (exit 75)")
    s.add_argument("--wait", action="store_true", help="block until done and fetch outputs")
    s.add_argument("--out-dir", type=Path, help="download dir (default <lease dir>/outputs/<prompt_id>)")
    s.add_argument("--label", default="", help="free text shown by `status` and stored in the ledger")
    s.add_argument("--priority", choices=("normal", "front"), default="normal",
                   help="front = ComfyUI front-of-queue (interactive single checks only)")
    s.add_argument("--timeout", type=float, help="--wait timeout in seconds (default: none)")

    w = sub.add_parser("wait", help="block on a prompt and fetch its outputs",
                       description="Block until PROMPT_ID finishes on the card, download its outputs and print "
                                   "the same JSON as `submit --wait`.")
    w.add_argument("prompt_id")
    w.add_argument("--card", choices=CARDS, required=True)
    w.add_argument("--out-dir", type=Path, help="download dir (default <lease dir>/outputs/<prompt_id>)")
    w.add_argument("--timeout", type=float, help="seconds (default: none)")

    sub.add_parser("status", help="queues, current jobs, leases and GPU load for both cards",
                   description="Both instances' running/pending counts with ledger labels, active leases, aux "
                               "usage against AUX_VRAM_BUDGET_FRAC, h3 free memory, and nvidia-smi "
                               "utilization/memory for both GPUs, as JSON.")

    le = sub.add_parser("lease", help="run a command while holding one slot of a named GPU lease",
                        description="Hold one slot of lease NAME while COMMAND runs; waits while all slots are busy "
                                    "or the card lacks VRAM. Profiles: blender-aux (alias blender): 1 slot, waits "
                                    "while aux used + est(4 GB) > AUX_VRAM_BUDGET_FRAC(0.60) x total; blender-h3: "
                                    "2 slots, needs est(8 GB) free on the h3 card; blender-cpu: BLENDER_CPU_SLOTS "
                                    "slots (default 16), CPU pool, no VRAM check; other names: 1 slot, no VRAM "
                                    "check. Exit code is COMMAND's.",
                        usage="gpu_queue.py lease NAME [--slots N] [--est-gb G] [--label TEXT] [--timeout S] "
                              "-- COMMAND...")
    le.add_argument("name")
    le.add_argument("--slots", type=int, help="semaphore width (default per profile: blender-aux 1, blender-h3 2, "
                                               "blender-cpu $BLENDER_CPU_SLOTS or 16)")
    le.add_argument("--est-gb", type=float, help="VRAM headroom estimate (default per profile: aux 4, h3 8)")
    le.add_argument("--label", default="")
    le.add_argument("--timeout", type=float, help="give up waiting for a slot/VRAM after S seconds (exit 75)")

    argv = list(sys.argv[1:] if argv is None else argv)
    cmd: list[str] = []
    if argv[:1] == ["lease"] and "--" in argv:  # argparse REMAINDER would swallow the lease's own options
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if a.cmd == "submit":
        wf = load_api_workflow(a.workflow)
        try:
            row = submit(wf, a.card, a.label, a.priority == "front", str(a.workflow.resolve()).replace("\\", "/"),
                         est_gb=a.est_gb, admit_timeout=a.admit_timeout)
        except TimeoutError as e:
            print(f"gpu_queue: {e}", file=sys.stderr)
            return 75
        if not a.wait:
            print(json.dumps({k: row[k] for k in ("prompt_id", "number", "card", "label", "status", "admit_wait_s")}))
            return 0
        res = finish(row["prompt_id"], a.card, a.out_dir, a.timeout, row["submit_ts"], a.label)
        res["admit_wait_s"] = row["admit_wait_s"]
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "success" else 1
    if a.cmd == "wait":
        res = finish(a.prompt_id.strip(), a.card, a.out_dir, a.timeout)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "success" else 1
    if a.cmd == "status":
        print(json.dumps(status(), indent=2))
        return 0
    if a.cmd == "lease":
        if not cmd:
            ap.error("lease needs a command after --")
        try:
            return run_leased(a.name, a.slots, cmd, a.label, a.timeout, a.est_gb)
        except TimeoutError as e:
            print(f"gpu_queue: {e}", file=sys.stderr)
            return 75
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"gpu_queue: error: {exc}", file=sys.stderr)
        sys.exit(1)
