"""Shard a per-frame Blender gate/sweep script across N background Blender processes.

Blender gate/collision work (depsgraph evaluation, calc_loop_triangles, BVHTree) is single-threaded:
one process uses about 1.25 cores. Splitting the frame list across N processes run with ``-t 1`` turns
a 14-24 min sweep into a couple of minutes. The gate script itself is not edited; this runner only
changes which frames each process is told to evaluate and where it writes its JSON.

CLI (END is inclusive, like Blender frame ranges; FRAMES may also be a comma list "1,5,9"):

    python tools/blender_shard.py --blend FILE --script GATE.py --frames START:END[:STEP] --shards N
        [--threads 1] [--jobs J] [--split contiguous|strided] [--out-dir DIR]
        [--merge json-concat|none] [--frame-arg-style frames|start-end|shard|env|template]
        [--arg-template "..."] [--shard-out PATTERN] [--no-preload] [--low-priority]
        [--timeout SEC] [--blender EXE] -- <extra args passed to the gate script>

Each shard k runs, with stdout+stderr in DIR/shard_k.log and exit code/wall time reported:

    blender -b [FILE] -t THREADS --python-exit-code 1 --python GATE.py -- <style args> <extra>

``--python-exit-code 1`` makes a Python exception in the gate a non-zero exit (plain Blender exits 0).
The runner exits non-zero if any shard fails or its expected output file is missing.

Concurrency: each shard holds one slot of the machine-wide ``blender-cpu`` lease (tools/gpu_queue.py,
BLENDER_CPU_SLOTS slots, default 16) while its Blender runs, so every sharded sweep on the machine shares
one CPU budget. ``--jobs`` (default min(shards, 4)) is only a per-run upper bound. Shards start at
below-normal priority; ``--low-priority`` uses idle priority.

Frame-argument styles (``--frame-arg-style``):
    frames     (default)  --frames a,b,c --out DIR/shard_k.json
    start-end             --start a --end b --step s --out DIR/shard_k.json  (works with both splits:
                          a strided shard is start=START+k*STEP, step=N*STEP)
    shard                 --shard=k/N   (the gate picks its own frames; pair with --shard-out, and
                          --frames is optional)
    env                   no args; the gate reads the environment variables below
    template              --arg-template is shlex-split; each token is formatted with
                          {blend} {out} {frames_csv} {first} {last} {step} {index} {count} {out_dir};
                          a token that is exactly {frames_list} expands to one argument per frame.
Every style also exports BLENDER_SHARD_INDEX, BLENDER_SHARD_COUNT, BLENDER_SHARD_FRAMES (csv) and
BLENDER_SHARD_OUT to the child.

``--shard-out`` names the file a gate writes when its output path is not ours to choose (it is
formatted with {index} {count} {out_dir}); default DIR/shard_{index}.json.

``--blend`` is loaded by Blender on its command line. Gates that call bpy.ops.wm.open_mainfile
themselves (most of them) should get ``--no-preload`` so the file is not loaded twice; the path is
still available as {blend} in templates.

Merge (``--merge json-concat``, default): each shard's JSON is scanned for per-frame containers,
i.e. a list of dicts carrying a frame key ('frame', 'output_frame', 'blender_frame'), a dict keyed by
frame numbers, a top-level list of such dicts, or a list of frame numbers drawn from the shard's own
frames (e.g. 'failing_frames'). Those are concatenated and sorted by frame into DIR/merged.json.
Other keys are copied from shard 0; keys whose values differ between shards are listed in
merged['_shard_merge']['divergent_keys'] (aggregate them yourself or use the gate's own merger).
If no per-frame container is found the shards are left unmerged and the runner says so.

Gate calling conventions that work as-is:
    gate.py BLEND OUT a,b,c                  (frames as one comma list)
        --no-preload --frame-arg-style template --arg-template "{blend} {out} {frames_csv}"
        (merges: 'frames' rows + 'failing_frames'; smoke-tested 16 frames, 1 vs 8 shards identical)
    classify.py CANDIDATE.blend OUT f1 f2 ...  (frames as separate args)
        --no-preload --frame-arg-style template --arg-template "{blend} {out} {frames_list}"
        (summary keys are per-shard maxima; they show up as divergent_keys)
    solve.py range FIRST LAST OUT            (opens its own blend; warm-starts frame to frame)
        --no-preload --split contiguous --frame-arg-style template --arg-template "range {first} {last} {out}"
        (contiguous STEP=1 only: the solver warm-starts along each shard; output is a top-level row list)
    scripts that take frame lists but print results/render instead of writing JSON: use --merge none.
    dense_gate.py SHOT --shard=k/N           (writes <shot dir>/dense_gate_shard_k_N.json)
        --no-preload --frame-arg-style shard --shard-out "<shot dir>/dense_gate_shard_{index}_{count}.json"
        (then run the gate's own merger for the evidence JSON; gate-specific aggregations show up here as
        divergent)
Scripts that need a frame/shard flag before they can be sharded: ones that take a single state/frame (shard by
launching states instead), hard-coded frame ranges, or state ids instead of frames.

Python API (for orchestration scripts; run with any CPython, not inside Blender):

    import sys; sys.path.insert(0, '<repo>/tools')
    from blender_shard import run_shards
    res = run_shards(blend, gate, frames=range(1, 125), shards=16, no_preload=True,
                     frame_arg_style='template', arg_template='{blend} {out} {frames_csv}',
                     out_dir='blocking/gate_shards')
    res.ok, res.wall_s, [s.code for s in res.shards], res.merged_path, res.merge_note
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

BLENDER = os.environ.get('BLENDER_EXE', 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
FRAME_KEYS = ('frame', 'output_frame', 'blender_frame')
STYLES = ('frames', 'start-end', 'shard', 'env', 'template')


@dataclass
class ShardResult:
    index: int
    frames: list
    cmd: list
    log: str
    out: str
    code: int | None = None
    wall_s: float = 0.0
    output_exists: bool = False
    ok: bool = False
    error: str = ''
    lease_slot: int | None = None
    lease_wait_s: float = 0.0
    started_ts: float | None = None
    ended_ts: float | None = None


@dataclass
class RunResult:
    ok: bool
    wall_s: float
    shards: list = field(default_factory=list)
    merged_path: str | None = None
    merge_note: str = ''
    summary_path: str | None = None


def parse_frames(spec):
    """'START:END[:STEP]' (END inclusive) or 'a,b,c' -> list of ints; also accepts an iterable."""
    if spec is None:
        return []
    if not isinstance(spec, str):
        return [int(f) for f in spec]
    if ':' in spec:
        parts = [int(p) for p in spec.split(':')]
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] == 0):
            raise ValueError(f'bad frame range {spec!r}; expected START:END[:STEP]')
        step = parts[2] if len(parts) == 3 else 1
        return list(range(parts[0], parts[1] + (1 if step > 0 else -1), step))
    return [int(p) for p in spec.split(',') if p.strip()]


def split_frames(frames, shards, split='contiguous'):
    """N non-empty chunks (fewer if there are fewer frames): contiguous blocks or every N-th frame."""
    n = max(1, min(shards, len(frames))) if frames else shards
    if not frames:
        return [[] for _ in range(n)]
    if split == 'strided':
        return [frames[k::n] for k in range(n)]
    base, extra = divmod(len(frames), n)
    chunks, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        chunks.append(frames[i:i + size])
        i += size
    return chunks


def _chunk_step(chunk, default=1):
    steps = {b - a for a, b in zip(chunk, chunk[1:])}
    if len(steps) > 1:
        raise ValueError(f'start-end style needs evenly spaced frames per shard, got {chunk}')
    return steps.pop() if steps else default


def gate_args(style, index, count, chunk, out, blend, out_dir, arg_template=None):
    csv = ','.join(map(str, chunk))
    if style == 'frames':
        return ['--frames', csv, '--out', out]
    if style == 'start-end':
        return ['--start', str(chunk[0]), '--end', str(chunk[-1]), '--step', str(_chunk_step(chunk)), '--out', out]
    if style == 'shard':
        return [f'--shard={index}/{count}']
    if style == 'env':
        return []
    if style == 'template':
        if not arg_template:
            raise ValueError('--frame-arg-style template needs --arg-template')
        values = {'blend': blend or '', 'out': out, 'frames_csv': csv, 'index': index, 'count': count,
                  'first': chunk[0] if chunk else '', 'last': chunk[-1] if chunk else '',
                  'step': (_chunk_step(chunk) if chunk else 1) if '{step}' in arg_template else '', 'out_dir': out_dir}
        args = []
        for token in shlex.split(arg_template, posix=True):
            if token == '{frames_list}':
                args += [str(f) for f in chunk]
            else:
                args.append(token.format(**values))
        return args
    raise ValueError(f'unknown frame-arg-style {style!r}; choose from {STYLES}')


# ---------------------------------------------------------------- merge

def _frame_of(row):
    if isinstance(row, dict):
        for k in FRAME_KEYS:
            v = row.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return v
    return None


def _is_row_list(v):
    return isinstance(v, list) and v and all(_frame_of(r) is not None for r in v)


def _num_key(k):
    try:
        return float(k)
    except (TypeError, ValueError):
        return None


def _is_frame_dict(v):
    return isinstance(v, dict) and v and all(_num_key(k) is not None for k in v)


def merge_json(docs, frames_per_shard=None):
    """Merge shard JSON documents. Returns (merged, note); merged is None when nothing per-frame was found."""
    if not docs:
        return None, 'no shard outputs'
    if all(isinstance(d, list) for d in docs):
        if not all(_is_row_list(d) or d == [] for d in docs) or not any(docs):
            return None, 'top-level lists without per-frame rows; left unmerged'
        rows = sorted((r for d in docs for r in d), key=_frame_of)
        dup = _duplicates([_frame_of(r) for r in rows])
        return rows, f'concatenated {len(rows)} top-level rows' + (f'; DUPLICATE frames {dup}' if dup else '')
    if not all(isinstance(d, dict) for d in docs):
        return None, 'shard outputs are not all JSON objects or lists; left unmerged'

    keys = list(dict.fromkeys(k for d in docs for k in d))
    row_keys = [k for k in keys if any(_is_row_list(d.get(k)) for d in docs)
                and all(_is_row_list(d.get(k)) or d.get(k) in (None, []) for d in docs)]
    dict_keys = [k for k in keys if k not in row_keys and any(_is_frame_dict(d.get(k)) for d in docs)
                 and all(_is_frame_dict(d.get(k)) or d.get(k) in (None, {}) for d in docs)]
    if not row_keys and not dict_keys:
        return None, 'no per-frame list/dict found in shard outputs; left unmerged'

    # Frames each shard actually evaluated, from its row lists (falls back to the frames we sent it).
    shard_frames = []
    for i, d in enumerate(docs):
        seen = {_frame_of(r) for k in row_keys for r in (d.get(k) or [])}
        seen |= {_num_key(x) for k in dict_keys for x in (d.get(k) or {})}
        if not seen and frames_per_shard:
            seen = set(frames_per_shard[i])
        shard_frames.append(seen)

    merged, frame_lists, divergent, notes = {}, [], [], []
    for k in keys:
        values = [d.get(k) for d in docs]
        if k in row_keys:
            rows = sorted((r for v in values for r in (v or [])), key=_frame_of)
            dup = _duplicates([_frame_of(r) for r in rows])
            if dup:
                notes.append(f'{k}: DUPLICATE frames {dup}')
            merged[k] = rows
        elif k in dict_keys:
            out = {}
            for v in values:
                for fk, fv in (v or {}).items():
                    if fk in out and out[fk] != fv:
                        notes.append(f'{k}: conflicting entries for frame {fk}')
                    out[fk] = fv
            merged[k] = dict(sorted(out.items(), key=lambda kv: _num_key(kv[0])))
        elif (all(isinstance(v, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) for v in values)
              and any(values) and all(set(v) <= shard_frames[i] for i, v in enumerate(values))):
            merged[k] = sorted(x for v in values for x in v)
            frame_lists.append(k)
        else:
            merged[k] = values[0]
            if any(v != values[0] for v in values[1:]):
                divergent.append(k)
    merged['_shard_merge'] = {'shards': len(docs), 'per_frame_lists': row_keys, 'per_frame_dicts': dict_keys,
                              'frame_number_lists': frame_lists, 'divergent_keys': divergent,
                              'note': 'divergent_keys hold shard 0 values; aggregate them from the shard files'}
    note = f'merged {row_keys + dict_keys} (+ frame lists {frame_lists})'
    if divergent:
        note += f'; divergent non-frame keys kept from shard 0: {divergent}'
    if notes:
        note += '; ' + '; '.join(notes)
    return merged, note


def _duplicates(values):
    seen, dup = set(), []
    for v in values:
        if v in seen and v not in dup:
            dup.append(v)
        seen.add(v)
    return dup


# ---------------------------------------------------------------- run

def default_jobs(shards):
    return max(1, min(shards, 4))


def _cpu_lease_factory():
    """gpu_queue.gpu_lease when importable (machine-wide 'blender-cpu' pool); None otherwise."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from gpu_queue import gpu_lease
    except Exception as e:  # noqa: BLE001
        print(f'blender_shard: gpu_queue unavailable ({e}); running without the blender-cpu lease', flush=True)
        return None
    return gpu_lease


def run_shards(blend, script, frames=None, shards=8, *, threads=1, out_dir=None, merge='json-concat',
               frame_arg_style='frames', arg_template=None, extra=(), jobs=None, split='contiguous',
               shard_out=None, no_preload=False, low_priority=False, timeout=None, blender=BLENDER,
               cwd=None, env=None, quiet=False):
    """Run the gate over `frames` in `shards` background Blender processes; returns a RunResult.

    frames: 'START:END[:STEP]', 'a,b,c' or an iterable of ints (may be empty for the 'shard' style).
    Everything else mirrors the CLI flags (see the module docstring).
    """
    if frame_arg_style not in STYLES:
        raise ValueError(f'unknown frame-arg-style {frame_arg_style!r}; choose from {STYLES}')
    if merge not in ('json-concat', 'none'):
        raise ValueError("merge must be 'json-concat' or 'none'")
    frames = parse_frames(frames)
    if not frames and frame_arg_style not in ('shard', 'template'):
        raise ValueError(f'--frames is required for frame-arg-style {frame_arg_style!r}')
    script = str(Path(script).resolve())
    blend = str(Path(blend).resolve()) if blend else None
    if blend and not Path(blend).is_file():
        raise FileNotFoundError(blend)
    out_dir = Path(out_dir or Path.cwd() / 'blender_shard_out').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_frames(frames, shards, split)
    count = len(chunks)
    jobs = jobs or default_jobs(count)
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))

    results = []
    for k, chunk in enumerate(chunks):
        out = str(Path(shard_out.format(index=k, count=count, out_dir=out_dir))) if shard_out else str(out_dir / f'shard_{k}.json')
        cmd = [blender, '-b'] + ([blend] if blend and not no_preload else []) + ['-t', str(threads), '--python-exit-code', '1',
                                                                                  '--python', script, '--']
        cmd += gate_args(frame_arg_style, k, count, chunk, out, blend, str(out_dir), arg_template) + list(extra)
        results.append(ShardResult(index=k, frames=chunk, cmd=cmd, log=str(out_dir / f'shard_{k}.log'), out=out))

    lock = threading.Lock()
    procs = {}
    stop = threading.Event()
    flags = 0
    if os.name == 'nt':  # below-normal by default; --low-priority drops to idle
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | (
            subprocess.IDLE_PRIORITY_CLASS if low_priority else subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    lease_factory = _cpu_lease_factory()
    label = f'blender_shard {Path(script).stem} pid {os.getpid()}'

    def run_one(r):
        if stop.is_set():
            r.error = 'cancelled'
            return r
        child_env = dict(os.environ if env is None else env)
        child_env.update(BLENDER_SHARD_INDEX=str(r.index), BLENDER_SHARD_COUNT=str(count),
                         BLENDER_SHARD_FRAMES=','.join(map(str, r.frames)), BLENDER_SHARD_OUT=r.out)
        stale = Path(r.out)
        if stale.exists():
            stale.unlink()  # never report a previous run's output as this shard's result
        lease = None
        if lease_factory is not None:
            t_lease = time.perf_counter()
            lease = lease_factory('blender-cpu', label=f'{label} shard {r.index}/{count}', cancel=stop.is_set)
            try:
                r.lease_slot = lease.acquire()
            except InterruptedError:
                r.error = 'cancelled'
                return r
            r.lease_wait_s = round(time.perf_counter() - t_lease, 2)
        try:
            _run_leased(r, child_env)
        finally:
            if lease is not None:
                lease.release()
        return r

    def _run_leased(r, child_env):
        start = time.perf_counter()
        r.started_ts = round(time.time(), 3)
        with open(r.log, 'w', encoding='utf-8', errors='replace') as log:
            log.write('# ' + subprocess.list2cmdline(r.cmd) + '\n')
            log.flush()
            p = subprocess.Popen(r.cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=child_env,
                                 creationflags=flags, preexec_fn=(lambda: os.nice(19 if low_priority else 10)) if os.name != 'nt' else None)
            with lock:
                procs[r.index] = p
            try:
                r.code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                r.code = p.wait()
                r.error = f'timeout after {timeout}s'
        with lock:
            procs.pop(r.index, None)
        r.wall_s = round(time.perf_counter() - start, 2)
        r.ended_ts = round(time.time(), 3)
        r.output_exists = Path(r.out).is_file()
        needs_output = merge != 'none'
        r.ok = r.code == 0 and not r.error and (r.output_exists or not needs_output)
        if not r.error and r.code != 0:
            r.error = f'exit code {r.code}'
        elif not r.error and needs_output and not r.output_exists:
            r.error = f'missing output {r.out}'
        say(f'shard {r.index}/{count} frames={_fmt_frames(r.frames)} exit={r.code} wall={r.wall_s}s'
            + (f' cpu-slot={r.lease_slot} lease-wait={r.lease_wait_s}s' if r.lease_slot is not None else '')
            + (f' ERROR {r.error}' if r.error else '') + f' log={r.log}')

    say(f'blender_shard: {len(frames)} frames -> {count} shards ({split}), {jobs} concurrent, -t {threads}, style={frame_arg_style}')
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(run_one, results))
    except BaseException:
        stop.set()
        with lock:
            for p in procs.values():
                p.kill()
        raise
    wall = round(time.perf_counter() - t0, 2)

    run = RunResult(ok=all(r.ok for r in results), wall_s=wall, shards=results)
    if merge == 'none':
        run.merge_note = 'merge disabled (--merge none); shard outputs left as-is'
    elif not run.ok:
        run.merge_note = 'not merged: some shards failed'
    else:
        docs, bad = [], []
        for r in results:
            try:
                docs.append(json.loads(Path(r.out).read_text(encoding='utf-8')))
            except (OSError, ValueError) as e:
                bad.append(f'{r.out}: {e}')
        if bad:
            run.merge_note = 'not merged: unreadable shard JSON ' + '; '.join(bad)
        else:
            merged, run.merge_note = merge_json(docs, [r.frames for r in results])
            if merged is not None:
                path = out_dir / 'merged.json'
                path.write_text(json.dumps(merged, indent=1) + '\n', encoding='utf-8')
                run.merged_path = str(path)
    summary = out_dir / 'shard_summary.json'
    summary.write_text(json.dumps(asdict(run) | {'frames': frames, 'jobs': jobs, 'threads': threads, 'split': split,
                                                 'frame_arg_style': frame_arg_style, 'script': script, 'blend': blend,
                                                 'sum_shard_wall_s': round(sum(r.wall_s for r in results), 2)},
                                  indent=1) + '\n', encoding='utf-8')
    run.summary_path = str(summary)
    say(f'blender_shard: {"OK" if run.ok else "FAILED"} wall={wall}s sum-of-shards={round(sum(r.wall_s for r in results), 2)}s; '
        f'{run.merge_note}' + (f' -> {run.merged_path}' if run.merged_path else '') + f'; summary {summary}')
    return run


def _fmt_frames(frames):
    if len(frames) <= 6:
        return ','.join(map(str, frames))
    return f'{frames[0]}..{frames[-1]} ({len(frames)})'


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], epilog='See the module docstring for styles and gate recipes.')
    ap.add_argument('--blend', help='.blend to load (use --no-preload if the gate opens it itself)')
    ap.add_argument('--script', required=True, help='gate script run with --python')
    ap.add_argument('--frames', help="START:END[:STEP] (END inclusive) or a,b,c")
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--threads', type=int, default=1, help='Blender -t per shard (default 1)')
    ap.add_argument('--jobs', type=int, help='upper bound on concurrent shards (default min(shards, 4)); the '
                                              'machine-wide blender-cpu lease (BLENDER_CPU_SLOTS, default 16) is the real limit')
    ap.add_argument('--split', choices=('contiguous', 'strided'), default='contiguous')
    ap.add_argument('--out-dir', default='blender_shard_out')
    ap.add_argument('--merge', choices=('json-concat', 'none'), default='json-concat')
    ap.add_argument('--frame-arg-style', choices=STYLES, default='frames')
    ap.add_argument('--arg-template', help='gate args for --frame-arg-style template')
    ap.add_argument('--shard-out', help='path the gate writes per shard; {index} {count} {out_dir}')
    ap.add_argument('--no-preload', action='store_true', help='do not pass --blend on the Blender command line')
    ap.add_argument('--low-priority', action='store_true', help='idle process priority (default is below-normal)')
    ap.add_argument('--timeout', type=float, help='per-shard timeout in seconds')
    ap.add_argument('--blender', default=BLENDER)
    a = ap.parse_args(argv)
    if a.shards < 1:
        ap.error('--shards must be >= 1')
    try:
        run = run_shards(a.blend, a.script, a.frames, a.shards, threads=a.threads, out_dir=a.out_dir, merge=a.merge,
                         frame_arg_style=a.frame_arg_style, arg_template=a.arg_template, extra=extra, jobs=a.jobs,
                         split=a.split, shard_out=a.shard_out, no_preload=a.no_preload, low_priority=a.low_priority,
                         timeout=a.timeout, blender=a.blender)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))
    return 0 if run.ok else 1


if __name__ == '__main__':
    sys.exit(main())
