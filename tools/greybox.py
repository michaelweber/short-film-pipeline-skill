"""Greybox a film's sets and block every shot in Blender (headless) with rigged, posable mannequins.

    "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup -P tools/greybox.py -- \
        film/<f>/blocking.json [--only s01,s03] [--gpu NAME] [--no-anim]   (--no-anim skips the animatic and depth)

Per shot, in <blocking dir>/blocking/:
  <id>_t<sec>.png   Cycles frames at each "frames" time (img2img inits for keyframes.py)
  <id>_plan.png     top-down plan: ceilings hidden, camera frustum in yellow, a dot per cast key in the cast colour
  <id>_anim.mp4     Workbench animatic of the whole shot at half size
  <id>_depth.mp4    depth control video (near = white) for the H3 Fun ControlNet ("control" in shots.json)
  <id>.blend        the shot, openable in Blender
Rendering uses Cycles on the GPU whose name contains --gpu (default: the aux_gpu_name setting, see
tools/pipeline_settings.py; unset = any GPU), so it can run beside ComfyUI on another card.

blocking.json (metres; x right, y forward along the set, z up; heading 0 = facing +y, 90 = facing +x):
  "set": [{"id", "box": [cx,cy,cz], "size": [sx,sy,sz], "color", "emit", "alpha", "ceiling",
           "only": [shot ids], "move": [[t, [cx,cy,cz]], ...]}]         # "move" keys a part (a sliding door)
  "actors": {"hero": {"kind": "mannequin", "height": 2.2, "width": 1.45, "color", "head_color",
                       "props": [{"kind": "cannon"|"pistol"|"smg"|"rifle"|"sword", "hand": "R"|"L", "color", "size"}]},
             "drone": {"kind": "spider", "size": 0.9, "color"}}
  "shots": [{"id", "duration", "frames": [t...], "hide": [set ids], "shake": deg,
             "camera": [[t, [x,y,z], [tx,ty,tz], lens_mm], ...], "fx": [...], "cast": [...]}]
Cast entry: {"actor", "name" (defaults to actor; used by "name.part" refs), "pose" (default preset),
             "surface" (spiders: floor|wall_left|wall_right|ceiling), "keys": [...],
             "sever": t (a mannequin loses everything above the hips from t), "hide_from": t, "show_from": t,
             "drop": [{"prop": kind, "t", "at": [x,y,z]}], "pickup": [{"prop": kind, "t"}]}
Mannequin key: {"t", "at": [x,y,z] | "name.part", "offset", "heading", "pose", "hand_r"/"hand_l"/"foot_r"/"foot_l":
                local [x,y,z] (1.8 m figure), "aim_r"/"aim_l": world point (arm straight at it, so a gun in that hand
                points there), "look": world point, "sword_tip": world point, "twist", "lean", "tilt" (degrees)}.
Spider key: {"t", "at", "offset", "heading", "surface", "tilt"}. Lists [t, x, y, z, heading] work for both kinds.
Parts for "name.part": hand_r, hand_l, foot_r, foot_l, look, head, chest, pelvis.
fx: {"kind": "flash"|"burst"|"sparks"|"debris", "t", "at" (point or "name.part"), "offset", "color", "count",
     "size", "spread", "dir": [x,y,z]}. flash = one-frame muzzle flash; burst = gore that lands and stays;
     sparks = short-lived yellow; debris = grey pieces that land and stay.
Keys interpolate linearly.
"""
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_settings import setting  # noqa: E402

SURFACE = {"floor": Euler((0, 0, 0)), "wall_left": Euler((0, math.radians(90), 0)),
           "wall_right": Euler((0, math.radians(-90), 0)), "ceiling": Euler((0, math.radians(180), 0))}

# Rest skeleton of a 1.8 m figure facing +y (x is its right): bone -> (head, tail, parent, connected).
# Elbows rest slightly back and knees slightly forward, so the IK chains bend the right way without poles.
REST = {
    "hips": ((0, 0, 0.95), (0, 0, 1.08), None, False),
    "spine": ((0, 0, 1.08), (0, 0, 1.40), "hips", True),
    "neck": ((0, 0, 1.40), (0, 0, 1.50), "spine", True),
    "head": ((0, 0, 1.50), (0, 0, 1.74), "neck", True),
}
for _side, _sx in (("R", 1), ("L", -1)):
    REST.update({
        f"shoulder.{_side}": ((0, 0, 1.36), (0.19 * _sx, 0, 1.37), "spine", False),
        f"upper_arm.{_side}": ((0.19 * _sx, 0, 1.37), (0.21 * _sx, -0.04, 1.08), f"shoulder.{_side}", True),
        f"forearm.{_side}": ((0.21 * _sx, -0.04, 1.08), (0.23 * _sx, 0.04, 0.83), f"upper_arm.{_side}", True),
        f"hand.{_side}": ((0.23 * _sx, 0.04, 0.83), (0.235 * _sx, 0.06, 0.74), f"forearm.{_side}", True),
        f"thigh.{_side}": ((0.1 * _sx, 0, 0.95), (0.1 * _sx, 0.04, 0.52), "hips", False),
        f"shin.{_side}": ((0.1 * _sx, 0.04, 0.52), (0.1 * _sx, 0, 0.09), f"thigh.{_side}", True),
        f"foot.{_side}": ((0.1 * _sx, 0, 0.09), (0.1 * _sx, 0.17, 0.03), f"shin.{_side}", True),
    })
PELVIS_Z = 0.95
SHOULDER = (0.19, 0.0, 1.37 - PELVIS_Z)  # right shoulder, relative to the pelvis
ARM = 0.293 + 0.263                       # upper arm + forearm, the IK reach
# Cross-section (x, y) of each body segment for a figure of width 1.
SEG = {"hips": ("box", 0.34, 0.22), "spine": ("box", 0.42, 0.26), "neck": ("cyl", 0.1, 0.1),
       "shoulder": ("cyl", 0.1, 0.1), "upper_arm": ("cyl", 0.11, 0.11), "forearm": ("cyl", 0.09, 0.09),
       "hand": ("box", 0.09, 0.05), "thigh": ("cyl", 0.16, 0.16), "shin": ("cyl", 0.11, 0.11),
       "foot": ("box", 0.1, 0.08)}
UPPER = ("spine", "neck", "head", "shoulder", "upper_arm", "forearm", "hand")
# Local targets for a 1.8 m figure: pelvis position, feet, hands (x right, y forward, z up, from the root on the floor).
POSES = {
    "stand": {"pelvis": (0, 0, 0.95), "foot_l": (-0.12, 0.02, 0.03), "foot_r": (0.12, 0.02, 0.03),
              "hand_l": (-0.24, 0.04, 0.84), "hand_r": (0.24, 0.04, 0.84)},
    "ready": {"pelvis": (0, 0, 0.86), "foot_l": (-0.2, 0.28, 0.03), "foot_r": (0.22, -0.22, 0.03),
              "hand_l": (-0.2, 0.38, 1.05), "hand_r": (0.22, 0.38, 1.1)},
    "wide": {"pelvis": (0, 0, 0.8), "foot_l": (-0.35, 0.1, 0.03), "foot_r": (0.35, -0.1, 0.03),
             "hand_l": (-0.3, 0.3, 1.0), "hand_r": (0.3, 0.3, 1.0)},
    "run_a": {"pelvis": (0, 0.05, 0.9), "foot_l": (-0.1, 0.5, 0.1), "foot_r": (0.1, -0.4, 0.3),
              "hand_l": (-0.24, -0.25, 1.0), "hand_r": (0.24, 0.35, 1.15)},
    "run_b": {"pelvis": (0, 0.05, 0.9), "foot_l": (-0.1, -0.4, 0.3), "foot_r": (0.1, 0.5, 0.1),
              "hand_l": (-0.24, 0.35, 1.15), "hand_r": (0.24, -0.25, 1.0)},
    "lunge": {"pelvis": (0, 0.15, 0.72), "foot_l": (-0.15, 0.7, 0.03), "foot_r": (0.15, -0.55, 0.05),
              "hand_l": (-0.2, 0.6, 1.1), "hand_r": (0.2, 0.6, 1.1)},
    "crouch": {"pelvis": (0, -0.1, 0.6), "foot_l": (-0.22, 0.12, 0.03), "foot_r": (0.22, 0.12, 0.03),
               "hand_l": (-0.25, 0.35, 0.8), "hand_r": (0.25, 0.35, 0.8)},
    "kneel": {"pelvis": (0, -0.05, 0.55), "foot_l": (-0.15, 0.4, 0.03), "foot_r": (0.15, -0.5, 0.05),
              "hand_l": (-0.2, 0.35, 0.9), "hand_r": (0.2, 0.35, 0.9)},
    "stomp_up": {"pelvis": (0, 0, 0.95), "foot_l": (-0.12, -0.05, 0.03), "foot_r": (0.14, 0.4, 0.55),
                 "hand_l": (-0.35, 0.1, 1.0), "hand_r": (0.35, 0.1, 1.0)},
    "stomp_down": {"pelvis": (0, 0.05, 0.84), "foot_l": (-0.14, -0.15, 0.03), "foot_r": (0.14, 0.45, 0.08),
                   "hand_l": (-0.35, 0.1, 0.95), "hand_r": (0.35, 0.1, 0.95)},
    "overhead": {"pelvis": (0, 0, 0.86), "foot_l": (-0.2, 0.28, 0.03), "foot_r": (0.22, -0.22, 0.03),
                 "hand_l": (-0.08, 0.25, 1.85), "hand_r": (0.08, 0.25, 1.85)},
    "arms_wide": {"pelvis": (0, 0, 0.88), "foot_l": (-0.25, 0.1, 0.03), "foot_r": (0.25, -0.1, 0.03),
                  "hand_l": (-0.6, 0.2, 1.35), "hand_r": (0.6, 0.2, 1.35)},
}
PROPS = {"cannon": (0.1, 0.16, 0.55), "pistol": (0.05, 0.12, 0.22), "smg": (0.06, 0.16, 0.42),
         "rifle": (0.06, 0.12, 0.85)}
PART_TARGETS = ("hand_r", "hand_l", "foot_r", "foot_l", "look")
MX = "mixamorig:"                          # bone prefix of Tripo's humanoid (Mixamo) auto-rig
SIDE = {"r": "Right", "l": "Left"}
FLIP = Matrix.Diagonal((-1, -1, 1))        # Tripo models face -y with their right at -x; greybox figures face +y


def args() -> tuple[Path, set | None, str, bool]:
    argv = sys.argv[sys.argv.index("--") + 1:]
    only, gpu = None, setting("aux_gpu_name", "")
    for i, a in enumerate(argv):
        if a == "--only":
            only = set(argv[i + 1].split(","))
        if a == "--gpu":
            gpu = argv[i + 1]
    return Path(argv[0]), only, gpu, "--no-anim" not in argv


def use_gpu(scene, name: str) -> None:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        prefs.compute_device_type = backend
        prefs.get_devices()
        hits = [d for d in prefs.devices if d.type == backend and name in d.name]
        if hits:
            for d in prefs.devices:
                d.use = d in hits
            scene.cycles.device = "GPU"
            return
    raise SystemExit(f"greybox: no GPU matching {name!r}")


class Shot:
    """Everything built for one shot, with helpers shared by the builders."""

    def __init__(self, spec: dict, shot: dict):
        self.spec, self.shot = spec, shot
        self.fps = spec.get("fps", 24)
        self.mats: dict = {}
        self.cast: dict = {}
        self.plan_only: list = []
        self.models: dict = {}  # actor -> imported model template (one FBX import per actor per shot)
        self.base = Path(spec.get("_dir", "."))

    def frame(self, t: float) -> int:
        return round(t * self.fps) + 1

    def mat(self, color, emit=0.0, alpha=1.0):
        key = (tuple(color), emit, alpha)
        if key not in self.mats:
            m = bpy.data.materials.new(f"m{len(self.mats)}")
            m.use_nodes = True
            m.diffuse_color = (*color, alpha)  # Workbench animatic colour
            bsdf = m.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (*color, 1)
            bsdf.inputs["Roughness"].default_value = 0.8
            if emit:
                bsdf.inputs["Emission Color"].default_value = (*color, 1)
                bsdf.inputs["Emission Strength"].default_value = emit
            if alpha < 1:
                bsdf.inputs["Alpha"].default_value = alpha
            self.mats[key] = m
        return self.mats[key]

    def mesh(self, kind: str, name: str, matrix: Matrix, size, material):
        if kind == "cyl":
            bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.5, depth=1)
        elif kind == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=16, ring_count=8)
        else:
            bpy.ops.mesh.primitive_cube_add(size=1)
        ob = bpy.context.object
        ob.name = name
        ob.matrix_world = matrix @ Matrix.Diagonal((*size, 1))
        ob.data.materials.append(material)
        return ob

    def key_hidden(self, obs, changes):
        """changes: [(t, hidden)], applied to render and viewport visibility."""
        for ob in obs:
            for t, hidden in changes:
                ob.hide_render = ob.hide_viewport = hidden
                ob.keyframe_insert("hide_render", frame=self.frame(t))
                ob.keyframe_insert("hide_viewport", frame=self.frame(t))

    def gun_axis(self, name: str, side: str, t: float) -> tuple[Vector, Vector]:
        """(muzzle point, barrel direction) of the gun in `name`'s hand at time t."""
        inst = self.cast[name]
        gun = next((h for k, h in inst["held"].items() if k != "sword" and h["hand"] == side), None)
        if gun is None:
            raise ValueError(f"{name} holds no gun in hand {side}")
        bpy.context.scene.frame_set(self.frame(t))
        dg = bpy.context.evaluated_depsgraph_get()
        if "bone" in gun:  # model prop: muzzle and breech are in its hand bone's space
            rig = inst["rig"].evaluated_get(dg)
            mw = rig.matrix_world @ rig.pose.bones[gun["bone"]].matrix
            tip, back = mw @ gun["tip"], mw @ gun["back"]
        else:  # mannequin gun: a unit box along its local z
            mw = gun["obs"][0].evaluated_get(dg).matrix_world
            tip, back = mw @ Vector((0, 0, 0.5)), mw @ Vector((0, 0, -0.5))
        return tip, (tip - back).normalized()

    def resolve(self, at, t: float, offset=None) -> Vector:
        """A world point, or "name.part" evaluated at time t."""
        if not isinstance(at, str):
            p = Vector(at)
        else:
            name, part = at.split(".")
            inst = self.cast[name]
            if part in ("muzzle_r", "muzzle_l"):
                return self.gun_axis(name, part[-1].upper(), t)[0] + Vector(offset or (0, 0, 0))
            scene = bpy.context.scene
            scene.frame_set(self.frame(t))
            dg = bpy.context.evaluated_depsgraph_get()
            if inst["kind"] == "spider":
                p = inst["root"].evaluated_get(dg).matrix_world.translation.copy()
            elif part in PART_TARGETS:
                p = inst["targets"][part].evaluated_get(dg).matrix_world.translation.copy()
            elif inst["kind"] == "model":
                rig = inst["rig"].evaluated_get(dg)
                pb = rig.pose.bones
                top = pb.get(MX + "HeadTop_End")
                local = {"head": (pb[MX + "Head"].head + (top.head if top else pb[MX + "Head"].tail)) / 2,
                         "chest": pb[MX + "Spine2"].head, "pelvis": pb[MX + "Hips"].head}[part]
                p = rig.matrix_world @ local
            else:
                rig = inst["rig"].evaluated_get(dg)
                pb = rig.pose.bones
                local = {"head": (pb["head"].head + pb["head"].tail) / 2, "chest": pb["spine"].tail * 0.85
                         + pb["spine"].head * 0.15, "pelvis": pb["hips"].head}[part]
                p = rig.matrix_world @ local
        return p + Vector(offset or (0, 0, 0))


def frame_along(a, b) -> tuple[Matrix, float]:
    """Matrix with local z along a->b (x kept near world x), origin at the midpoint; and the length."""
    a, b = Vector(a), Vector(b)
    z = (b - a).normalized()
    ref = Vector((1, 0, 0)) if abs(z.x) < 0.9 else Vector((0, 1, 0))
    x = (ref - z * ref.dot(z)).normalized()
    m = Matrix((x, z.cross(x), z)).transposed().to_4x4()
    m.translation = (a + b) / 2
    return m, (b - a).length


def attach(ob, rig, bone: str) -> None:
    """Parent to a bone, keeping the world transform (the rig is in rest pose while building)."""
    mw = ob.matrix_world.copy()
    ob.parent, ob.parent_type, ob.parent_bone = rig, "BONE", bone
    bpy.context.view_layer.update()
    ob.matrix_world = mw


def bone_axis_facing(bone, direction: Vector) -> str:
    """Damped Track axis of `bone` (in rest pose) that points closest to `direction` in armature space."""
    m = bone.matrix_local.to_3x3()
    best = max(((m.col[i] * s).dot(direction), s, i) for i in (0, 2) for s in (1, -1))
    return ("TRACK_" if best[1] > 0 else "TRACK_NEGATIVE_") + "XYZ"[best[2]]


# ---- mannequin -----------------------------------------------------------------------------
def build_mannequin(S: Shot, name: str, spec: dict) -> dict:
    w = spec.get("width", 1.0)
    arm = bpy.data.armatures.new(name)
    rig = bpy.data.objects.new(name, arm)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for b, (h, t, parent, conn) in REST.items():
        eb = arm.edit_bones.new(b)
        eb.head, eb.tail, eb.roll = h, t, 0
        if parent:
            eb.parent, eb.use_connect = arm.edit_bones[parent], conn
    bpy.ops.object.mode_set(mode="OBJECT")
    body = S.mat(spec["color"])
    skin = S.mat(spec.get("head_color", spec["color"]))
    dark = S.mat(tuple(v * 0.45 for v in spec.get("head_color", spec["color"])))
    parts: dict[str, list] = {}
    for b, (h, t, _, _) in REST.items():
        base = b.split(".")[0]
        if base == "head":
            c = (Vector(h) + Vector(t)) / 2
            head = S.mesh("sphere", f"{name}.head", Matrix.Translation(c), (0.21, 0.24, 0.26), skin)
            nose = S.mesh("box", f"{name}.nose", Matrix.Translation(c + Vector((0, 0.12, -0.01))),
                          (0.06, 0.07, 0.06), dark)
            obs = [head, nose]
        else:
            kind, sx, sy = SEG[base]
            k = w if base in ("hips", "spine") else w ** 0.5
            m, length = frame_along(h, t)
            obs = [S.mesh(kind, f"{name}.{b}", m, (sx * k, sy * (1 + (k - 1) * 0.6), length), body)]
        for ob in obs:
            attach(ob, rig, b)
        parts[b] = obs
    # IK chains and their targets (children of the rig, so their locations are rig-local).
    targets = {}
    for part in PART_TARGETS:
        e = bpy.data.objects.new(f"{name}.tgt_{part}", None)
        e.empty_display_size = 0.05
        bpy.context.collection.objects.link(e)
        e.parent = rig
        targets[part] = e
    for side in "LR":
        s = side.lower()
        for bone, tgt in ((f"forearm.{side}", f"hand_{s}"), (f"shin.{side}", f"foot_{s}")):
            ik = rig.pose.bones[bone].constraints.new("IK")
            ik.target, ik.chain_count, ik.use_tail = targets[tgt], 2, True
    look = rig.pose.bones["head"].constraints.new("DAMPED_TRACK")
    look.target = targets["look"]
    look.track_axis = bone_axis_facing(rig.data.bones["head"], Vector((0, 1, 0)))
    # Props: guns ride the hand bone (they point along the forearm, so a straight arm aims them); swords are free
    # objects that follow the hand and point at a keyed tip.
    held = {}
    for p in spec.get("props", []):
        side = p.get("hand", "R")
        color = p.get("color", (0.08, 0.08, 0.09))
        if p["kind"] == "sword":
            root = bpy.data.objects.new(f"{name}.sword", None)
            bpy.context.collection.objects.link(root)
            length = p.get("size", 1.0)
            blade = S.mesh("box", f"{name}.blade", Matrix.Translation((0, 0, 0.25 * length + 0.375 * length)),
                           (0.012, 0.045, 0.75 * length), S.mat((0.85, 0.87, 0.9)))
            hilt = S.mesh("box", f"{name}.hilt", Matrix.Translation((0, 0, 0.1 * length)),
                          (0.035, 0.035, 0.25 * length), S.mat(color))
            guard = S.mesh("box", f"{name}.guard", Matrix.Translation((0, 0, 0.24 * length)),
                           (0.09, 0.09, 0.02), S.mat((0.7, 0.55, 0.2)))
            for ob in (blade, hilt, guard):
                ob.parent = root
            c = root.constraints.new("COPY_LOCATION")
            c.target, c.subtarget, c.head_tail = rig, f"hand.{side}", 1.0
            tip = bpy.data.objects.new(f"{name}.sword_tip", None)
            bpy.context.collection.objects.link(tip)
            d = root.constraints.new("DAMPED_TRACK")
            d.target, d.track_axis = tip, "TRACK_Z"
            held["sword"] = {"obs": [blade, hilt, guard], "hand": side, "tip": tip, "size": (0.05, 0.05, length)}
        else:
            thick, tall, length = [v * p.get("scale", 1.0) for v in PROPS[p["kind"]]]
            h, t = Vector(REST[f"hand.{side}"][0]), Vector(REST[f"hand.{side}"][1])
            d = (t - Vector(REST[f"forearm.{side}"][0])).normalized()
            m, _ = frame_along(t - d * 0.06, t + d * (length - 0.06))
            gun = S.mesh("box", f"{name}.{p['kind']}", m, (thick, tall, length), S.mat(color))
            attach(gun, rig, f"hand.{side}")
            held[p["kind"]] = {"obs": [gun], "hand": side, "size": (thick, tall, length)}
    return {"kind": "mannequin", "rig": rig, "targets": targets, "parts": parts, "held": held,
            "s": spec["height"] / 1.8, "color": spec["color"]}


def norm_key(k) -> dict:
    if isinstance(k, dict):
        return k
    t, x, y, z, heading = k
    return {"t": t, "at": [x, y, z], "heading": heading}


def key_mannequin(S: Shot, inst: dict, cast: dict) -> list:
    rig, s = inst["rig"], inst["s"]
    rig.rotation_mode = "XYZ"
    hips, spine = rig.pose.bones["hips"], rig.pose.bones["spine"]
    spine.rotation_mode = "QUATERNION"
    hips_rest = hips.bone.matrix_local.to_3x3()
    spine_rest = spine.bone.matrix_local.to_3x3()
    rig.scale = (s, s, s)
    spots = []
    for raw in cast["keys"]:
        k = norm_key(raw)
        t, f = k["t"], S.frame(k["t"])
        pose = POSES[k.get("pose", cast.get("pose", "stand"))]
        loc = S.resolve(k["at"], t, k.get("offset"))
        # aim/look/sword_tip may be "name.part": evaluate them now, before this key's unkeyed values are set
        # (resolve() jumps frames, which re-applies the animation and would wipe them).
        points = {a: S.resolve(k[a], t) for a in ("aim_r", "aim_l", "look", "sword_tip") if a in k}
        R = Euler((math.radians(k.get("tilt", 0)), 0, math.radians(-k.get("heading", 0)))).to_matrix()
        Rinv = R.inverted()

        def local(name):
            return Rinv @ (points[name] - loc) / s

        pelvis = Vector(pose["pelvis"])
        spots.append(loc.copy())
        for side in "lr":
            hand = Vector(k.get(f"hand_{side}", pose[f"hand_{side}"]))
            if f"aim_{side}" in k:
                sh = pelvis + Vector((SHOULDER[0] * (1 if side == "r" else -1), SHOULDER[1], SHOULDER[2]))
                # 99.9% of the reach: the arm is straight, so the gun on the hand bone points exactly at the target
                hand = sh + (local(f"aim_{side}") - sh).normalized() * ARM * 0.999
            inst["targets"][f"hand_{side}"].location = hand
            inst["targets"][f"foot_{side}"].location = k.get(f"foot_{side}", pose[f"foot_{side}"])
        inst["targets"]["look"].location = (local("look") if "look" in k
                                            else pelvis + Vector((0, 2.0, 0.67)))
        hips.location = hips_rest.inverted() @ (pelvis - Vector((0, 0, PELVIS_Z)))
        q = Euler((math.radians(-k.get("lean", 0)), 0, math.radians(-k.get("twist", 0)))).to_matrix()
        spine.rotation_quaternion = (spine_rest.inverted() @ q @ spine_rest).to_quaternion()
        rig.location = loc
        rig.rotation_euler = (math.radians(k.get("tilt", 0)), 0, math.radians(-k.get("heading", 0)))
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)
        hips.keyframe_insert("location", frame=f)
        spine.keyframe_insert("rotation_quaternion", frame=f)
        for e in inst["targets"].values():
            e.keyframe_insert("location", frame=f)
        sword = inst["held"].get("sword")
        if sword:
            side = sword["hand"].lower()
            hand_world = loc + R @ (Vector(inst["targets"][f"hand_{side}"].location) * s)
            sword["tip"].location = (points["sword_tip"] if "sword_tip" in k
                                     else hand_world + R @ Vector((0, 0.6, 0.8)))
            sword["tip"].keyframe_insert("location", frame=f)
    return spots


def cast_visibility(S: Shot, inst: dict, cast: dict) -> None:
    if inst["kind"] in ("spider", "model"):
        obs = list(inst["obs"])
    else:
        obs = [ob for part in inst["parts"].values() for ob in part]
        obs += [ob for h in inst["held"].values() for ob in h["obs"]]
    if "sever" in cast:
        props = [ob for h in inst["held"].values() for ob in h["obs"]]
        if inst["kind"] == "model":
            body, t = inst["body"], cast["sever"]
            mask = body.modifiers.new("lower", "MASK")  # keeps the "lower" group: hips and legs
            mask.vertex_group = "lower"
            for tt, on in ((0, False), (t, True)):
                mask.show_render = mask.show_viewport = on
                body.keyframe_insert('modifiers["lower"].show_render', frame=S.frame(tt))
                body.keyframe_insert('modifiers["lower"].show_viewport', frame=S.frame(tt))
            upper = props
        else:
            upper = [ob for b, part in inst["parts"].items() if b.split(".")[0] in UPPER for ob in part] + props
        S.key_hidden(upper, [(0, False), (cast["sever"], True)])
    if "hide_from" in cast:
        S.key_hidden(obs, [(0, False), (cast["hide_from"], True)])
    if "show_from" in cast:
        S.key_hidden(obs, [(0, True), (cast["show_from"], False)])
    for d in cast.get("drop", []):
        h = inst["held"][d["prop"]]
        back = next((p["t"] for p in cast.get("pickup", []) if p["prop"] == d["prop"] and p["t"] > d["t"]), None)
        S.key_hidden(h["obs"], [(0, False), (d["t"], True)] + ([(back, False)] if back is not None else []))
        at = Vector(d["at"])
        if inst["kind"] == "model":
            lying = lay_down(h["obs"][0], at, f"{inst['root'].name}.dropped_{d['prop']}",
                             inst["scale"] * h.get("k", 1.0))
        else:
            thick, tall, length = h["size"]
            m, _ = frame_along(at - Vector((length / 2, 0.1, 0)), at + Vector((length / 2, 0.1, 0)))
            lying = S.mesh("box", f"{inst['rig'].name}.dropped_{d['prop']}", m, (max(thick, 0.03), tall, length),
                           S.mat((0.85, 0.87, 0.9) if d["prop"] == "sword" else (0.08, 0.08, 0.09)))
        S.key_hidden([lying], [(0, True), (d["t"], False)] + ([(back, True)] if back is not None else []))


# ---- Tripo models --------------------------------------------------------------------------
def import_fbx(path: Path) -> list:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path))
    return [o for o in bpy.data.objects if o not in before]


def model_template(S: Shot, actor: str, spec: dict) -> dict:
    """Import an actor's model once per shot: {"rig", "body", "props": {kind: prop}, "m": measurements, "used"}.
    A prop with "model" is its own Tripo model (a folder with one FBX, lying along +x, muzzle or blade tip at +x,
    top at +z), placed in the hand's palm at "grip" (a point in the prop's own units) and scaled to "length" metres
    (see hold_prop). A prop without "model" is cut out of a model that was generated holding it (split_prop).
    Either way each prop knows its long axis in hand-bone space, so aim_*/sword_tip can turn the hand to point it.
    With "fix_skin", an embedded prop's optional "rigid_regions": [[min_xyz, max_xyz], ...] defines its
    complete fixed-grip skin in imported mesh rest coordinates. These vertices bind rigidly to its hand,
    rather than inheriting incorrect leg weights or being confused with neighbouring coat geometry.
    "recolor": [{"bones": ["{r}ForeArm", "{r}Hand"], "color": [r, g, b], "mix": 0.85}] repaints the skin weighted to
    those bones ("{r}"/"{l}" = the character's right/left side name; a name ending in "*" is a prefix)."""
    if actor in S.models:
        return S.models[actor]
    new = import_fbx(next((S.base / spec["model"]).glob("*.fbx")))
    rig = next(o for o in new if o.type == "ARMATURE")
    body = next(o for o in new if o.type == "MESH")
    for o in new:
        if o not in (rig, body):
            bpy.data.objects.remove(o)
    for m in body.data.materials:
        m.diffuse_color = (*spec["color"], 1)  # Workbench animatic colour
    bones = rig.data.bones
    vgroups = {g.index: g.name for g in body.vertex_groups}
    verts = body.data.vertices
    tmpl = {"rig": rig, "body": body, "props": {}, "used": 0, "side": SIDE}
    if "mixamorig:Hips" in bones:
        def flip(v):
            return FLIP @ Vector(v)
        # The auto-rig sometimes names the sides from the viewer's point of view: the character's right is -x here.
        if bones[MX + "RightArm"].head_local.x > bones[MX + "LeftArm"].head_local.x:
            tmpl["side"] = {"r": "Left", "l": "Right"}
        side_of = tmpl["side"]
        top = max(v.co.z for v in verts)
        arm_r = bones[MX + "RightArm"]
        tmpl["m"] = m = {"hips": flip(bones[MX + "Hips"].head_local), "H": top, "f": top / 1.8,
                         "head_z": bones[MX + "Head"].head_local.z,
                         "arm": (bones[MX + "RightForeArm"].head_local - arm_r.head_local).length
                         + (bones[MX + "RightHand"].head_local - bones[MX + "RightForeArm"].head_local).length}
        for s, side in side_of.items():
            m[f"shoulder_{s}"] = flip(bones[MX + f"{side}Arm"].head_local)
            m[f"ankle_{s}"] = flip(bones[MX + f"{side}Foot"].head_local)
        props = [(p, p.get("hand", "R").upper()) for p in spec.get("props", [])]
        if spec.get("fix_skin"):  # coat hems hanging beside the hands, skinned to them by the auto-rig
            keep = set()
            for p, letter in props:
                if "model" in p:
                    continue
                side = side_of[letter.lower()]
                regions = p.get("rigid_regions")
                if regions:
                    selected = [v.index for v in verts if any(
                        all(lo[axis] <= v.co[axis] <= hi[axis] for axis in range(3)) for lo, hi in regions)]
                    for group in body.vertex_groups:
                        if group.name in bones and bones[group.name].use_deform:
                            group.remove(selected)
                    body.vertex_groups[MX + side + "Hand"].add(selected, 1.0, "REPLACE")
                else:
                    selected = prop_selection(body, rig, side, vgroups)[0]
                keep.update(selected)
            fix_skin(body, rig, keep, 0.08 * top)
        vgroups = {g.index: g.name for g in body.vertex_groups}
        # "lower" vertex group for "sever": everything whose strongest bone is the hips or a leg
        legs = {MX + "Hips"} | {MX + f"{side}{b}" for side in SIDE.values()
                                 for b in ("UpLeg", "Leg", "Foot", "ToeBase", "Toe_End")}
        lower = [v.index for v in verts if v.groups and vgroups[max(v.groups, key=lambda g: g.weight).group] in legs]
        body.vertex_groups.new(name="lower").add(lower, 1.0, "REPLACE")
        for rc in spec.get("recolor", []):
            recolor(body, vgroups, [MX + b.format(r=side_of["r"], l=side_of["l"]) for b in rc["bones"]],
                    rc["color"], rc.get("mix", 0.85))
        scale = spec["height"] / top  # model units -> metres, as build_model scales the root
        for p, letter in props:
            side = side_of[letter.lower()]
            if "model" in p:
                tmpl["props"][p["kind"]] = hold_prop(S, rig, p, side, letter, scale)
            else:
                tmpl["props"][p["kind"]] = split_prop(body, rig, p["kind"], side, letter, vgroups,
                                                      cut=p.get("split", True))
        # Track anatomical forward, not the nearest auto-rig bone axis (which can pitch the gaze by 25+ degrees).
        # An unweighted parent preserves the head's bind matrix while supplying an exact forward-facing aim axis.
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        head = rig.data.edit_bones[MX + "Head"]
        aim = rig.data.edit_bones.new("head_aim")
        aim.head = head.head
        aim.tail = head.head + Vector((0, -head.length, 0))
        aim.align_roll(Vector((0, 0, 1)))
        aim.parent, aim.use_deform = head.parent, False
        head.use_connect = False
        head.parent = aim
        bpy.ops.object.mode_set(mode="OBJECT")
    S.models[actor] = tmpl
    return tmpl


def recolor(body, vgroups: dict, bones: list, color, mix: float) -> None:
    """Blend `color` over the base colour of the skin weighted to `bones`, by weight x `mix` (a mask attribute)."""
    def hit(name):
        return any(name.startswith(b[:-1]) if b.endswith("*") else name == b for b in bones)
    groups = {i for i, n in vgroups.items() if hit(n)}
    attr = body.data.attributes.get("recolor") or body.data.attributes.new("recolor", "FLOAT", "POINT")
    for v in body.data.vertices:
        w = min(1.0, sum(g.weight for g in v.groups if g.group in groups))
        attr.data[v.index].value = max(attr.data[v.index].value, w)
    for mat in body.data.materials:
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        src = bsdf.inputs["Base Color"].links[0].from_socket if bsdf.inputs["Base Color"].links else None
        att = nt.nodes.new("ShaderNodeAttribute")
        att.attribute_name = "recolor"
        amt = nt.nodes.new("ShaderNodeMath")
        amt.operation = "MULTIPLY"
        amt.inputs[1].default_value = mix
        nt.links.new(att.outputs["Fac"], amt.inputs[0])
        mixn = nt.nodes.new("ShaderNodeMix")
        mixn.data_type = "RGBA"
        nt.links.new(amt.outputs[0], mixn.inputs["Factor"])
        if src:
            nt.links.new(src, mixn.inputs["A"])
        else:
            mixn.inputs["A"].default_value = bsdf.inputs["Base Color"].default_value
        mixn.inputs["B"].default_value = (*color, 1)
        nt.links.new(mixn.outputs["Result"], bsdf.inputs["Base Color"])


def hand_frame(rig, side: str) -> tuple:
    """(palm centre, F, U, N) of `side`'s hand in rest armature space: F wrist -> middle knuckle (a held gun's
    barrel), U pinky knuckle -> index knuckle (up a gun's grip, out along a sword's blade), N out of the palm. In the
    rest A-pose the palms face the thighs, so N points toward the body's centre line."""
    b = rig.data.bones
    pre = MX + side + "Hand"
    wrist = b[pre].head_local
    idx, mid, pinky = (b[pre + n + "1"].head_local for n in ("Index", "Middle", "Pinky"))
    F = (mid - wrist).normalized()
    U = idx - pinky
    U = (U - F * U.dot(F)).normalized()
    N = F.cross(U)
    centre_x = b[MX + "Hips"].head_local.x
    if N.x * (centre_x - wrist.x) < 0:
        N = -N
    width = (idx - pinky).length
    centre = wrist.lerp((idx + pinky) / 2, 0.8) + N * 0.5 * width
    return centre, F, U, N


def hold_prop(S: Shot, rig, p: dict, side: str, letter: str, scale: float) -> dict:
    """Import a prop model and parent it to the hand bone with its "grip" point in the palm: guns point along the
    hand (barrel +x along F, top +z along U), swords come out of the fist on the thumb side (tip +x along U, the edge
    at -y facing F). "length" is its real length in metres (the model is about 1 unit long)."""
    new = import_fbx(next((S.base / p["model"]).glob("*.fbx")))
    ob = next(o for o in new if o.type == "MESH")
    for o in new:
        if o is not ob:
            bpy.data.objects.remove(o)
    ob.name = f"{rig.name}.{p['kind']}"
    xs = [v.co.x for v in ob.data.vertices]
    k = p["length"] / scale / (max(xs) - min(xs))
    centre, F, U, N = hand_frame(rig, side)
    if p["kind"] == "sword":
        X, Y = U, -F
    else:
        X, Y = F, U.cross(F)
    R = Matrix((X, Y, X.cross(Y))).transposed().to_4x4()
    ob.matrix_world = Matrix.Translation(centre) @ R @ Matrix.Scale(k, 4) @ Matrix.Translation(-Vector(p["grip"]))
    bpy.context.view_layer.update()
    attach(ob, rig, MX + side + "Hand")
    # Props are authored along +X; a rear-most grip vertex must not tilt the barrel's aiming axis.
    lo, hi = min(xs), max(xs)
    end = [v.co for v in ob.data.vertices if v.co.x > hi - 0.02]
    tip = sum(end, Vector()) / len(end)
    back = tip - Vector((hi - lo, 0, 0))
    to_bone = rig.data.bones[MX + side + "Hand"].matrix_local.inverted() @ ob.matrix_world
    up = ob.matrix_world.to_3x3().col[2].normalized()  # the prop's +z (a gun's top) in rest armature space
    return {"hand": letter, "bone": MX + side + "Hand", "tip": to_bone @ tip, "back": to_bone @ back,
            "axis_bone": (to_bone @ tip - to_bone @ back).normalized(), "ob": ob, "k": k, "palm": N,
            "up_bone": None if p["kind"] == "sword" else rig.data.bones[MX + side + "Hand"].matrix_local.to_3x3().inverted() @ up,
            "curl": p.get("curl", 75)}


def curl_fingers(rig, hand: str, palm: Vector, degrees: float) -> None:
    """Bend each joint matching a hand or finger bone-name prefix toward the palm normal `palm`
    (rest armature space); the thumb bends less. The axis is (bone direction x palm),
    so the joints curl instead of splaying regardless of the auto-rig's bone rolls."""
    for pb in rig.pose.bones:
        name = pb.name
        if not name.startswith(hand) or name == hand or name[-1] not in "123":
            continue
        seg = int(name[-1])
        amount = degrees * ({1: 0.8, 2: 1.1, 3: 0.8}[seg]) * (0.4 if "Thumb" in name else 1.0)
        rest = pb.bone.matrix_local.to_3x3()
        axis = rest.col[1].cross(palm)
        if axis.length < 1e-6:
            continue
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion(rest.inverted() @ axis.normalized(), math.radians(amount))


def prop_selection(body, rig, side: str, vgroups: dict) -> tuple[list, object]:
    """(vertex indices, hand bone) of what `side`'s hand holds: vertices mostly weighted to the hand, farther from
    the wrist than the fist reaches, in connected pieces at least a quarter the size of the biggest (small strays,
    a strap or a coat seam, stay on the body)."""
    bones = rig.data.bones
    hand = bones[MX + f"{side}Hand"]
    wrist = hand.head_local
    tips = [b.head_local for b in bones if b.name.startswith(MX + f"{side}Hand") and b.name.endswith("4")]
    fist = max((p - wrist).length for p in tips) * 1.15 if tips else 0.06
    hand_groups = {i for i, n in vgroups.items() if n.startswith(MX + f"{side}Hand")}
    cand = {v.index for v in body.data.vertices
            if sum(g.weight for g in v.groups if g.group in hand_groups) >= 0.5 and (v.co - wrist).length > fist}
    adj = {i: [] for i in cand}
    for e in body.data.edges:
        a, b = e.vertices
        if a in cand and b in cand:
            adj[a].append(b)
            adj[b].append(a)
    pieces, seen = [], set()
    for i in cand:
        if i not in seen:
            stack, piece = [i], []
            seen.add(i)
            while stack:
                x = stack.pop()
                piece.append(x)
                stack += [y for y in adj[x] if y not in seen and not seen.add(y)]
            pieces.append(piece)
    biggest = max((len(p) for p in pieces), default=0)
    return [i for p in pieces if len(p) >= biggest * 0.25 for i in p], hand


def fix_skin(body, rig, keep: set, reach: float) -> None:
    """Remove every distant forearm/hand influence, including mixed hand-and-leg weights.
    Preserve valid local weights; grow missing weights from repaired neighbours, with
    spatial transfer for isolated regions. Protected held-prop vertices stay unchanged."""
    from mathutils.geometry import intersect_point_line
    from mathutils.kdtree import KDTree
    bones = rig.data.bones
    names = {g.index: g.name for g in body.vertex_groups}
    verts = body.data.vertices
    bone_groups = {i for i, name in names.items() if name in bones and bones[name].use_deform}

    def dist(p, a, b):
        _, t = intersect_point_line(p, a, b)
        return (p - (a + (b - a) * min(1.0, max(0.0, t)))).length

    arms = {}
    for side in SIDE.values():
        pre = MX + side
        tip = max((b.tail_local for b in bones if b.name.startswith(pre + "Hand")),
                  key=lambda p: (p - bones[pre + "Hand"].head_local).length)
        chain = [bones[pre + n].head_local for n in ("Shoulder", "Arm", "ForeArm", "Hand")] + [tip]
        arms[side] = list(zip(chain, chain[1:]))
    group_sides = {i: side for i, name in names.items() for side in SIDE.values()
                   if name == MX + side + "ForeArm" or name.startswith(MX + side + "Hand")}
    weights = {v.index: {g.group: g.weight for g in v.groups
                        if g.group in bone_groups and g.weight > 0.0} for v in verts}
    bad, grown = set(), {}
    for v in verts:
        if v.index in keep:
            continue
        ws = weights[v.index]
        candidates = {group_sides[g] for g in ws if g in group_sides}
        distant = {side for side in candidates
                   if min(dist(v.co, a, b) for a, b in arms[side]) > reach}
        if not distant:
            continue
        bad.add(v.index)
        valid = {g: w for g, w in ws.items() if group_sides.get(g) not in distant}
        total = sum(valid.values())
        weights[v.index] = {g: w / total for g, w in valid.items()} if total else {}
        if total:
            grown[v.index] = weights[v.index]
    if not bad:
        return
    nbrs = {i: [] for i in bad}
    for e in body.data.edges:
        a, b = e.vertices
        if a in bad:
            nbrs[a].append(b)
        if b in bad:
            nbrs[b].append(a)
    done = {i for i, ws in weights.items()
            if i not in keep and ws and not any(g in group_sides for g in ws)}
    anchors = set(done)
    todo = bad - grown.keys()
    while todo:
        layer = [i for i in todo if any(n in done for n in nbrs[i])]
        if not layer:
            break
        for i in layer:
            acc = {}
            src = [n for n in nbrs[i] if n in done]
            for n in src:
                for g, w in weights[n].items():
                    acc[g] = acc.get(g, 0.0) + w / len(src)
            total = sum(acc.values()) or 1.0
            weights[i] = grown[i] = {g: w / total for g, w in acc.items()}
        done.update(layer)
        todo.difference_update(layer)
    if todo:
        if not anchors:
            raise ValueError(f"{body.name}: no anchored skin weights for {len(todo)} coat vertices")
        tree = KDTree(len(anchors))
        for i in sorted(anchors):
            tree.insert(verts[i].co, i)
        tree.balance()
        for i in todo:
            neighbours = tree.find_n(verts[i].co, min(4, len(anchors)))
            acc = {}
            for _, source, distance in neighbours:
                factor = 1.0 / max(distance * distance, 1e-12)
                for group, weight in weights[source].items():
                    acc[group] = acc.get(group, 0.0) + weight * factor
            total = sum(acc.values())
            grown[i] = {group: weight / total for group, weight in acc.items()}
    for i, ws in grown.items():
        for group in bone_groups:
            body.vertex_groups[group].remove([i])
        for g, w in ws.items():
            body.vertex_groups[g].add([i], w, "REPLACE")
    print(f"greybox: {body.name}: re-skinned {len(grown)} of {len(bad)} distant arm-weighted vertices "
          f"({len(todo)} isolated vertices transferred from anchored skin)")


def split_prop(body, rig, kind: str, side: str, letter: str, vgroups: dict, cut: bool = True) -> dict:
    import bmesh
    import numpy as np
    sel, hand = prop_selection(body, rig, side, vgroups)
    wrist = hand.head_local
    if len(sel) < 20:
        raise SystemExit(f"greybox: no {kind} found in {body.name}'s {side.lower()} hand")
    pts = np.array([tuple(body.data.vertices[i].co) for i in sel])
    centre = pts.mean(axis=0)
    axis = Vector(np.linalg.eigh(np.cov((pts - centre).T))[1][:, -1])
    proj = (pts - centre) @ np.array(axis)
    if abs(proj.max()) < abs(proj.min()):  # point the axis at the end farther from the wrist
        axis, proj = -axis, -proj
    if (Vector(centre) + axis * proj.max() - wrist).length < (Vector(centre) + axis * proj.min() - wrist).length:
        axis, proj = -axis, -proj
    tip, back = Vector(centre) + axis * proj.max(), Vector(centre) + axis * proj.min()
    to_bone = hand.matrix_local.inverted()  # rest armature space -> hand-bone space
    held = {"hand": letter, "bone": hand.name, "tip": to_bone @ tip, "back": to_bone @ back,
            "axis_bone": to_bone.to_3x3() @ axis.normalized(), "ob": None}
    if not cut:  # welded: moves with the hand's skin; only its axis is known
        return held
    # cut: the prop object keeps only `sel`, the body loses it
    prop = body.copy()
    prop.data = body.data.copy()
    prop.name = f"{body.name}.{kind}"
    bpy.context.collection.objects.link(prop)
    keep = set(sel)
    for ob, drop in ((prop, lambda i: i not in keep), (body, lambda i: i in keep)):
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if drop(v.index)], context="VERTS")
        bm.to_mesh(ob.data)
        bm.free()
    prop.modifiers.clear()
    prop.vertex_groups.clear()
    prop.parent = None
    prop.matrix_world = body.matrix_world.copy()
    attach(prop, rig, hand.name)
    held["ob"] = prop
    return held


def lay_down(prop, at: Vector, name: str, scale: float):
    """An unparented copy of a prop (at the model's scale), its long axis along x, resting on the floor at `at`."""
    ob = prop.copy()
    ob.name = name
    ob.parent = None
    bpy.context.collection.objects.link(ob)
    pts = [Vector(v.co) for v in ob.data.vertices]
    c = sum(pts, Vector()) / len(pts)
    far = max(pts, key=lambda p: (p - c).length)
    rot = (far - c).normalized().rotation_difference(Vector((1, 0, 0))).to_matrix().to_4x4()
    ob.matrix_world = Matrix.Translation(at) @ rot @ Matrix.Scale(scale, 4) @ Matrix.Translation(-c)
    bpy.context.view_layer.update()
    lowest = min((ob.matrix_world @ p).z for p in pts)
    ob.matrix_world = Matrix.Translation((0, 0, at.z - lowest)) @ ob.matrix_world
    return ob


def instance_of(S: Shot, tmpl: dict, name: str) -> tuple:
    """(root empty, rig, body, props) for one cast member: the template's own objects the first time, linked copies
    after that (shared mesh and armature data; own pose, constraints and visibility)."""
    rig, body = tmpl["rig"], tmpl["body"]
    props = {k: p["ob"] for k, p in tmpl["props"].items() if p["ob"]}
    if tmpl["used"]:
        rig = rig.copy()  # the copy carries the first instance's constraints, keys and pose: start clean
        rig.animation_data_clear()
        for b in rig.pose.bones:
            for c in list(b.constraints):
                b.constraints.remove(c)
            b.location, b.rotation_quaternion, b.rotation_euler, b.scale = (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (1, 1, 1)
        bpy.context.collection.objects.link(rig)
        body = tmpl["body"].copy()
        body.animation_data_clear()
        for mod in [mod for mod in body.modifiers if mod.type == "MASK"]:  # the first instance's "sever"
            body.modifiers.remove(mod)
        bpy.context.collection.objects.link(body)
        body.parent = rig
        body.hide_render = body.hide_viewport = False
        for mod in body.modifiers:
            if mod.type == "ARMATURE":
                mod.object = rig
        props = {}
        for k, p in tmpl["props"].items():
            if not p["ob"]:
                continue
            ob = p["ob"].copy()
            ob.animation_data_clear()
            ob.hide_render = ob.hide_viewport = False
            bpy.context.collection.objects.link(ob)
            ob.parent = rig
            props[k] = ob
    tmpl["used"] += 1
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    rig.name, body.name = f"{name}.rig", f"{name}.body"
    rig.parent = root
    rig.matrix_parent_inverse = Matrix.Identity(4)
    rig.location, rig.rotation_euler, rig.scale = (0, 0, 0), (0, 0, math.pi), (1, 1, 1)
    for k, ob in props.items():
        ob.name = f"{name}.{k}"
    return root, rig, body, props


def signed_angle(u: Vector, v: Vector, normal: Vector) -> float:
    a = u.angle(v)
    return -a if u.cross(v).angle(normal) < 1 else a


def build_model(S: Shot, name: str, spec: dict, actor: str) -> dict:
    tmpl = model_template(S, actor, spec)
    root, rig, body, props = instance_of(S, tmpl, name)
    m = tmpl["m"]
    bones = rig.data.bones
    targets = {}
    for part in PART_TARGETS + ("pole_hand_r", "pole_hand_l", "pole_foot_r", "pole_foot_l"):
        e = bpy.data.objects.new(f"{name}.tgt_{part}", None)
        e.empty_display_size = 0.05
        bpy.context.collection.objects.link(e)
        e.parent = rig
        targets[part] = e
    for s, side in tmpl["side"].items():
        sx = 1 if s == "r" else -1
        f = m["f"]
        # poles (body frame, then into the model's frame): elbows bend back and down, knees forward
        poles = {"hand": m[f"shoulder_{s}"] + Vector((sx * 0.3, -1.0, -0.6)) * f,
                 "foot": flip_v(bones[MX + f"{side}UpLeg"].head_local) + Vector((sx * 0.1, 1.2, -0.5)) * f}
        for limb, (base, mid) in {"hand": ("Arm", "ForeArm"), "foot": ("UpLeg", "Leg")}.items():
            pole = targets[f"pole_{limb}_{s}"]
            pole.location = FLIP @ poles[limb]
            ik = rig.pose.bones[MX + side + mid].constraints.new("IK")
            ik.target, ik.chain_count, ik.use_tail = targets[f"{limb}_{s}"], 2, True
            ik.pole_target = pole
            b0, b1 = bones[MX + side + base], bones[MX + side + mid]
            pn = (b1.tail_local - b0.head_local).cross(pole.location - b0.head_local)
            # Bone.x_axis is relative to the parent bone; the pole maths needs the armature-space axis
            ik.pole_angle = signed_angle(b0.matrix_local.to_3x3().col[0], pn.cross(b0.tail_local - b0.head_local),
                                         b0.tail_local - b0.head_local)
            calibrate_pole(rig, ik, targets[f"{limb}_{s}"], pole, b0.name, b1.name)
    look = rig.pose.bones["head_aim"].constraints.new("DAMPED_TRACK")
    look.target = targets["look"]
    look.track_axis = "TRACK_Y"
    held = {k: {"obs": [props[k]] if k in props else [], "hand": p["hand"], "bone": p["bone"], "tip": p["tip"],
                "back": p["back"], "axis_bone": p["axis_bone"], "up_bone": p.get("up_bone"), "k": p.get("k", 1.0)}
            for k, p in tmpl["props"].items()}
    for p in tmpl["props"].values():  # close the fist around a separate prop model
        if "palm" in p:
            curl_fingers(rig, p["bone"], p["palm"], p["curl"])
    return {"kind": "model", "root": root, "rig": rig, "body": body, "targets": targets, "held": held,
            "obs": [body, *props.values()], "m": m, "scale": spec["height"] / m["H"],
            "arm_out": spec.get("arm_out", 0.0), "color": spec["color"], "side": tmpl["side"]}


def calibrate_pole(rig, ik, target, pole, base: str, mid: str) -> None:
    """Make the joint (elbow/knee) bend exactly toward the pole: pull the IK target in to 75 % of the chain's reach so
    the limb must bend, measure the angle about the limb's axis from where the joint went to where the pole is, and
    correct the pole angle by it (the textbook pole-angle formula depends on bone rolls, which auto-rigs vary)."""
    b0, b1 = rig.data.bones[base], rig.data.bones[mid]
    root, end = b0.head_local, b1.tail_local
    axis = (end - root).normalized()
    saved = target.location.copy()
    target.location = root + (end - root) * 0.75
    want = (pole.location - root) - axis * (pole.location - root).dot(axis)

    def error():
        bpy.context.view_layer.update()
        joint = rig.pose.bones[mid].head
        bend = (joint - root) - axis * (joint - root).dot(axis)
        return signed_angle(bend, want, axis) if bend.length > 1e-6 else 0.0

    err = error()
    for _ in range(4):
        if abs(err) < math.radians(2):
            break
        ik.pole_angle += err
        new = error()
        if abs(new) > abs(err):  # the correction goes the other way round for this rig
            ik.pole_angle -= 2 * err
            new = error()
        err = new
    target.location = saved
    bpy.context.view_layer.update()


def flip_v(v) -> Vector:
    return FLIP @ Vector(v)


def key_model(S: Shot, inst: dict, cast: dict) -> list:
    """Same keys as key_mannequin. Pose presets are for the 1.8 m mannequin: the pelvis scales with the model's hip
    height, feet move from the model's own stance, hands keep their offset from the shoulder scaled by arm length.
    aim_*/sword_tip straighten the arm at the point and then turn the hand so the held prop's axis points there."""
    root, rig, m, s = inst["root"], inst["rig"], inst["m"], inst["scale"]
    root.rotation_mode = "XYZ"
    root.scale = (s, s, s)
    pb = rig.pose.bones
    hips = pb[MX + "Hips"]
    spines = [pb[MX + n] for n in ("Spine", "Spine1", "Spine2")]
    hands = {x: pb[MX + f"{inst['side'][x]}Hand"] for x in "lr"}
    for b in spines + list(hands.values()):
        b.rotation_mode = "QUATERNION"
    f = m["f"]
    spots = []
    for raw in cast["keys"]:
        k = norm_key(raw)
        t, fr = k["t"], S.frame(k["t"])
        pose = POSES[k.get("pose", cast.get("pose", "stand"))]
        loc = S.resolve(k["at"], t, k.get("offset"))
        points = {a: S.resolve(k[a], t) for a in ("aim_r", "aim_l", "look", "sword_tip") if a in k}
        R = Euler((math.radians(k.get("tilt", 0)), 0, math.radians(-k.get("heading", 0)))).to_matrix()
        Rinv = R.inverted()

        def local(name):  # body frame (facing +y), model units
            return Rinv @ (points[name] - loc) / s

        pel_fig = Vector(pose["pelvis"])
        pelvis = m["hips"] + Vector((pel_fig.x * f, pel_fig.y * f, (pel_fig.z - PELVIS_Z) * m["hips"].z / PELVIS_Z))
        spots.append(loc.copy())
        hand_world = {}
        for x in "lr":
            sx = 1 if x == "r" else -1
            shoulder = pelvis + (m[f"shoulder_{x}"] - m["hips"])
            if f"aim_{x}" in k:
                hand = shoulder + (local(f"aim_{x}") - shoulder).normalized() * m["arm"] * 0.999
            else:
                sh_fig = pel_fig + Vector((SHOULDER[0] * sx, SHOULDER[1], SHOULDER[2]))
                hand = (shoulder + (Vector(k.get(f"hand_{x}", pose[f"hand_{x}"])) - sh_fig) * (m["arm"] / ARM)
                        + Vector((sx * inst["arm_out"] * f, 0, 0)))
            foot = m[f"ankle_{x}"] + (Vector(k.get(f"foot_{x}", pose[f"foot_{x}"]))
                                      - Vector(POSES["stand"][f"foot_{x}"])) * f
            inst["targets"][f"hand_{x}"].location = FLIP @ hand
            hand_world[x] = loc + R @ (hand * s)
            inst["targets"][f"foot_{x}"].location = FLIP @ foot
            hands[x].rotation_quaternion = (1, 0, 0, 0)
        look = local("look") if "look" in k else pelvis + Vector((0, 2.0 * f, m["head_z"] - m["hips"].z))
        inst["targets"]["look"].location = FLIP @ look
        hips.location = hips.bone.matrix_local.to_3x3().inverted() @ (FLIP @ (pelvis - m["hips"]))
        q = FLIP @ Euler((math.radians(-k.get("lean", 0) / 3), 0, math.radians(-k.get("twist", 0) / 3))).to_matrix() @ FLIP
        for b in spines:
            rest = b.bone.matrix_local.to_3x3()
            b.rotation_quaternion = (rest.inverted() @ q @ rest).to_quaternion()
        root.location = loc
        root.rotation_euler = (math.radians(k.get("tilt", 0)), 0, math.radians(-k.get("heading", 0)))
        root.keyframe_insert("location", frame=fr)
        root.keyframe_insert("rotation_euler", frame=fr)
        hips.keyframe_insert("location", frame=fr)
        for b in spines + list(hands.values()):
            b.keyframe_insert("rotation_quaternion", frame=fr)
        for name, e in inst["targets"].items():
            if not name.startswith("pole"):
                e.keyframe_insert("location", frame=fr)
        for x in "lr":
            held = next((h for h in inst["held"].values() if h["hand"].lower() == x), None)
            want = points.get(f"aim_{x}")
            if held is None or not held["obs"]:  # a welded prop can't turn without tearing the hand's skin
                continue
            if want is None and "sword" in inst["held"] and inst["held"]["sword"] is held:
                # default: blade forward and up from the fist, as the mannequin holds it
                want = points.get("sword_tip", hand_world[x] + R @ Vector((0, 0.6, 0.8)))
            if want is not None:
                point_hand(S, inst, x, held, want, fr)
    return spots


def point_hand(S: Shot, inst: dict, x: str, held: dict, want: Vector, fr: int) -> None:
    """Rotate (and key) the hand so the held prop's axis, from the wrist, points at world point `want`; a gun is also
    rolled about its barrel so its top faces up (a gun held level, not canted or upside down)."""
    scene = bpy.context.scene
    scene.frame_set(fr)
    dg = bpy.context.evaluated_depsgraph_get()
    rig = inst["rig"].evaluated_get(dg)
    name, parent = MX + f"{inst['side'][x]}Hand", MX + f"{inst['side'][x]}ForeArm"
    pose_hand, pose_fore = rig.pose.bones[name], rig.pose.bones[parent]
    W = rig.matrix_world
    W3 = W.to_3x3().normalized()
    M = pose_hand.matrix.to_3x3()
    cur = (W3 @ M @ held["axis_bone"]).normalized()
    pivot = W @ pose_hand.matrix.translation
    d = (want - pivot).normalized()
    turn = W3.inverted() @ cur.rotation_difference(d).to_matrix() @ W3
    if held.get("up_bone") is not None:
        up = W3 @ turn @ M @ held["up_bone"]
        up_now, up_want = up - d * up.dot(d), Vector((0, 0, 1)) - d * d.z
        if up_now.length > 1e-6 and up_want.length > 1e-6:
            roll = math.atan2(d.dot(up_now.cross(up_want)), up_now.dot(up_want))
            turn = W3.inverted() @ Quaternion(d, roll).to_matrix() @ W3 @ turn
    new = Matrix.Translation(pose_hand.matrix.translation) @ (turn @ M).to_4x4()
    rest, prest = inst["rig"].data.bones[name].matrix_local, inst["rig"].data.bones[parent].matrix_local
    basis = (prest.inverted() @ rest).inverted() @ pose_fore.matrix.inverted() @ new
    hand = inst["rig"].pose.bones[name]
    hand.rotation_quaternion = basis.to_quaternion()
    hand.keyframe_insert("rotation_quaternion", frame=fr)


def build_spider_model(S: Shot, name: str, spec: dict, actor: str) -> dict:
    tmpl = model_template(S, actor, spec)
    root, rig, body, _ = instance_of(S, tmpl, name)
    top = max(max(v.co.x for v in body.data.vertices) - min(v.co.x for v in body.data.vertices),
              max(v.co.y for v in body.data.vertices) - min(v.co.y for v in body.data.vertices))
    s = spec["size"] / top
    rig.scale = (s, s, s)
    rnd = random.Random(name)  # each drone's legs set a little differently
    for b in rig.pose.bones:
        if b.name.endswith("_Limb_0") or b.name.endswith("_Limb_1"):
            b.rotation_mode = "XYZ"
            b.rotation_euler = Euler([math.radians(rnd.uniform(-8, 8)) for _ in range(3)])
    return {"kind": "spider", "root": root, "obs": [body], "color": spec["color"]}


# ---- spider drone --------------------------------------------------------------------------
def build_spider(S: Shot, name: str, spec: dict) -> dict:
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    s, c = spec["size"], spec["color"]
    dark = S.mat((0.05, 0.05, 0.05))
    obs = [
        S.mesh("box", f"{name}.body", Matrix.Translation((0, 0, 0.35 * s)), (0.5 * s, 0.8 * s, 0.25 * s), S.mat(c)),
        S.mesh("box", f"{name}.eyes", Matrix.Translation((0, 0.42 * s, 0.38 * s)), (0.25 * s, 0.06 * s, 0.1 * s),
               S.mat((1, 0.1, 0.1), 3)),
        S.mesh("box", f"{name}.vent", Matrix.Translation((0, -0.42 * s, 0.38 * s)), (0.18 * s, 0.04 * s, 0.1 * s),
               S.mat((1, 0.5, 0.1), 3)),
    ]
    for side in (-1, 1):
        obs.append(S.mesh("cyl", f"{name}.saw", Matrix.Translation((side * 0.18 * s, 0.55 * s, 0.25 * s))
                          @ Euler((0, math.radians(90), 0)).to_matrix().to_4x4(),
                          (0.25 * s, 0.25 * s, 0.03 * s), S.mat((0.75, 0.75, 0.78))))
        for j in range(4):
            leg = S.mesh("box", f"{name}.leg", Matrix.Translation((side * 0.5 * s, (0.3 - 0.2 * j) * s, 0.2 * s))
                         @ Euler((0, side * math.radians(35), 0)).to_matrix().to_4x4(),
                         (0.5 * s, 0.05 * s, 0.05 * s), dark)
            obs.append(leg)
    for ob in obs:
        ob.parent = root
    return {"kind": "spider", "root": root, "obs": obs, "color": c}


def key_spider(S: Shot, inst: dict, cast: dict) -> list:
    root = inst["root"]
    root.rotation_mode = "QUATERNION"
    spots = []
    for raw in cast["keys"]:
        k = norm_key(raw)
        root.location = S.resolve(k["at"], k["t"], k.get("offset"))
        spots.append(root.location.copy())
        surface = k.get("surface", cast.get("surface", "floor"))
        root.rotation_quaternion = (SURFACE[surface].to_quaternion()
                                    @ Quaternion((0, 0, 1), math.radians(-k.get("heading", 0)))
                                    @ Quaternion((1, 0, 0), math.radians(k.get("tilt", 0))))
        root.keyframe_insert("location", frame=S.frame(k["t"]))
        root.keyframe_insert("rotation_quaternion", frame=S.frame(k["t"]))
    return spots


# ---- effects, camera, set --------------------------------------------------------------------
def build_fx(S: Shot, i: int, e: dict) -> None:
    t = e["t"]
    at = S.resolve(e["at"], t, e.get("offset"))
    f = S.frame(t)
    kind = e["kind"]
    rnd = random.Random(f"{S.shot['id']}:{i}")
    if kind == "flash":
        # A muzzle flash is a cone of fire from the gun's muzzle along its barrel ("at": "name.muzzle_r");
        # anything else gets a round flash.
        size = e.get("size", 0.45)
        if isinstance(e["at"], str) and e["at"].split(".")[1] in ("muzzle_r", "muzzle_l"):
            name, part = e["at"].split(".")
            tip, d = S.gun_axis(name, part[-1].upper(), t)
            bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=0.5, radius2=0.0, depth=1)
            ob = bpy.context.object
            ob.name = f"flash{i}"
            m, length = frame_along(tip + d * size * 1.6, tip)  # wide end away from the gun, apex at the muzzle
            ob.matrix_world = m @ Matrix.Diagonal((size, size, length, 1))
            ob.data.materials.append(S.mat(e.get("color", (1, 0.85, 0.3)), 30))
        else:
            ob = S.mesh("sphere", f"flash{i}", Matrix.Translation(at), [size] * 3,
                        S.mat(e.get("color", (1, 0.85, 0.3)), 30))
        S.key_hidden([ob], [(0, True), (t, False), (t + 2 / S.fps, True)])
        return
    color = e.get("color", {"burst": (0.45, 0.02, 0.02), "sparks": (1, 0.8, 0.2), "debris": (0.7, 0.7, 0.68)}[kind])
    emit = 20 if kind == "sparks" else 0
    life = 0.35 if kind == "sparks" else 0.7
    bias = Vector(e.get("dir", (0, 0, 0)))
    for n in range(e.get("count", 24)):
        size = e.get("size", 0.08) * rnd.uniform(0.5, 1.5)
        dims = (size * 0.55, size * 0.75, size * 1.6) if kind == "burst" else (size, size, size)
        ob = S.mesh("sphere" if kind == "burst" else "box", f"fx{i}_{n}", Matrix.Translation(at), dims, S.mat(color, emit))
        d = (Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.2, 1))) + bias).normalized()
        end = at + d * e.get("spread", 1.5) * rnd.uniform(0.4, 1.0)
        mid = at + (end - at) * 0.6 + Vector((0, 0, 0.3))
        if kind != "sparks":
            end.z = size / 2
        if kind == "burst":
            # Rounded airborne droplets spread into thin floor splashes instead of permanent proxy cubes.
            ob.keyframe_insert("scale", frame=f)
            ob.keyframe_insert("scale", frame=S.frame(t + life / 2))
            ob.scale = (size * 1.6, size * 0.9, size * 0.12)
            ob.keyframe_insert("scale", frame=S.frame(t + life))
            end.z = size * 0.06
        for tt, p in ((t, at), (t + life / 2, mid), (t + life, end)):
            ob.location = p
            ob.keyframe_insert("location", frame=S.frame(tt))
        S.key_hidden([ob], [(0, True), (t, False)] + ([(t + life, True)] if kind == "sparks" else []))


def fcurves(idb) -> list:
    ad = idb.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        try:
            return list(act.fcurves)
        except Exception:
            pass
    return [fc for layer in act.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]


def build_camera(S: Shot):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.scene.collection.objects.link(cam)
    for t, pos, target, lens in S.shot["camera"]:
        cam.location = pos
        cam.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = lens
        f = S.frame(t)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        cam.data.keyframe_insert("lens", frame=f)
    if S.shot.get("shake"):
        for fc in fcurves(cam):
            if fc.data_path == "rotation_euler":
                mod = fc.modifiers.new("NOISE")
                mod.strength, mod.scale = math.radians(S.shot["shake"]), 3.0
    return cam


def set_material(S: Shot, p: dict):
    """The part's material: flat colour, or with "tex" a procedural surface in world space, so a panel seam or a floor
    tile is in the same place in every shot: "panel" (wall panels with dark seams), "tile" (floor/ceiling tiles),
    "leds" (a grid of lit dots on a dark face; needs "emit"). "tex_size": [w, h] metres of one panel/tile/led."""
    kind = p.get("tex")
    if not kind:
        return S.mat(p["color"], p.get("emit", 0), p.get("alpha", 1))
    key = (kind, tuple(p["color"]), p.get("emit", 0), tuple(p.get("tex_size", ())), tuple(p["size"]))
    if key in S.mats:
        return S.mats[key]
    m = bpy.data.materials.new(f"{kind}{len(S.mats)}")
    m.use_nodes = True
    m.diffuse_color = (*p["color"], 1)
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.7
    # the part's two long axes span the surface; on walls the horizontal one goes first
    u, v = sorted(range(3), key=lambda i: -p["size"][i])[:2]
    if 2 in (u, v):
        u, v = (v if u == 2 else u), 2
    pos = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(pos.outputs["Position"], sep.inputs[0])
    nt.links.new(sep.outputs["XYZ"[u]], comb.inputs[0])
    nt.links.new(sep.outputs["XYZ"[v]], comb.inputs[1])
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.0
    w, h = p.get("tex_size", {"panel": (1.2, 0.75), "tile": (0.6, 0.6), "leds": (0.06, 0.04)}[kind])
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Brick Width"].default_value = w
    brick.inputs["Row Height"].default_value = h
    brick.inputs["Mortar Size"].default_value = {"panel": 0.012, "tile": 0.008, "leds": 0.015}[kind]
    c = Vector(p["color"])
    if kind == "leds":
        brick.inputs["Color1"].default_value = brick.inputs["Color2"].default_value = (*c, 1)
        brick.inputs["Mortar"].default_value = (0.01, 0.01, 0.012, 1)
    else:
        brick.inputs["Color1"].default_value = (*c, 1)
        brick.inputs["Color2"].default_value = (*(c * 0.88), 1)
        brick.inputs["Mortar"].default_value = (*(c * 0.3), 1)
    nt.links.new(comb.outputs[0], brick.inputs["Vector"])
    grime = nt.nodes.new("ShaderNodeTexNoise")
    grime.inputs["Scale"].default_value = 2.5
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.25
    nt.links.new(pos.outputs["Position"], grime.inputs["Vector"])
    nt.links.new(brick.outputs["Color"], mix.inputs["A"])
    nt.links.new(grime.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    if p.get("emit"):
        nt.links.new(brick.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = p["emit"]
    S.mats[key] = m
    return m


def dress(S: Shot, p: dict) -> None:
    """Fixed extra geometry on a wall ("dress": ["pipes", "trim"]): two pipes along its top and a dark baseboard, on
    the face towards the room: "inward": +1/-1 along the wall's thin axis (default: towards x = 0 or y = 15, the
    corridor's centre line)."""
    size, box = Vector(p["size"]), Vector(p["box"])
    thin = min(range(2), key=lambda i: size[i])  # 0: wall in the yz plane, 1: in the xz plane
    along = 1 - thin
    inward = p.get("inward", -1 if box[thin] > (15 if thin == 1 else 0) else 1)
    face = box[thin] + inward * size[thin] / 2
    top, bottom = box.z + size.z / 2, box.z - size.z / 2
    dark = S.mat((0.12, 0.13, 0.14))
    pipe = S.mat((0.32, 0.34, 0.3))
    for d in p["dress"]:
        if d == "pipes":
            for i, (r, dz) in enumerate(((0.055, 0.32), (0.035, 0.48))):
                c = box.copy()
                c[thin], c.z = face + inward * (0.08 + i * 0.1), top - dz
                rot = Euler((math.radians(90), 0, 0) if along == 1 else (0, math.radians(90), 0)).to_matrix().to_4x4()
                S.mesh("cyl", f"{p['id']}.pipe{i}", Matrix.Translation(c) @ rot, (r * 2, r * 2, size[along]), pipe)
        elif d == "trim":
            c = box.copy()
            c[thin], c.z = face + inward * 0.015, bottom + 0.08
            s = [0.03, 0.03, 0.16]
            s[along] = size[along]
            S.mesh("box", f"{p['id']}.trim", Matrix.Translation(c), s, dark)


def build_set(S: Shot) -> list:
    ceilings = []
    for p in S.spec["set"]:
        if p["id"] in S.shot.get("hide", []) or ("only" in p and S.shot["id"] not in p["only"]):
            continue
        ob = S.mesh("box", p["id"], Matrix.Translation(p["box"]), p["size"], set_material(S, p))
        for t, pos in p.get("move", []):
            ob.location = pos
            ob.keyframe_insert("location", frame=S.frame(t))
        if p.get("dress"):
            dress(S, p)
        if p.get("ceiling"):
            ceilings.append(ob)
    return ceilings


def frustum(S: Shot, cam, length=5.0):
    scene = bpy.context.scene
    frame = [v * (length / abs(v.z)) for v in cam.data.view_frame(scene=scene)]
    mesh = bpy.data.meshes.new("frustum")
    mesh.from_pydata([Vector((0, 0, 0))] + frame, [], [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)])
    ob = bpy.data.objects.new("frustum", mesh)
    ob.matrix_world = cam.matrix_world.copy()
    ob.data.materials.append(S.mat((1, 0.9, 0.1), 4, 0.5))
    bpy.context.collection.objects.link(ob)


def render_animatic(S: Shot, out: Path) -> None:
    scene = bpy.context.scene
    engine, pct = scene.render.engine, scene.render.resolution_percentage
    scene.render.engine = "BLENDER_WORKBENCH"
    ink, scene.render.use_freestyle = scene.render.use_freestyle, False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.render.resolution_percentage = 50
    scene.render.image_settings.file_format = "PNG"
    with tempfile.TemporaryDirectory() as tmp:
        scene.render.filepath = str(Path(tmp) / "f")
        bpy.ops.render.render(animation=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(S.fps), "-i", str(Path(tmp) / "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / f"{S.shot['id']}_anim.mp4")], check=True)
    scene.render.engine, scene.render.resolution_percentage = engine, pct
    scene.render.use_freestyle = ink
    print(f"+ {out / (S.shot['id'] + '_anim.mp4')}")



def render_depth(S: Shot, out: Path, spots: dict) -> None:
    """<id>_depth.mp4: a depth control video for the H3 Fun ControlNet (near = white, far = black). The range is the
    shot's "depth": [near, far] metres, or by default 1.5 m nearer than the closest cast key and 4 m beyond the
    farthest, as seen from the camera keys, so the figures stand out from the walls behind them. Every surface gets
    one emission material driven by the camera's view depth; muzzle flashes are left out."""
    scene = bpy.context.scene
    if "depth" in S.shot:
        near, far = S.shot["depth"]
    else:
        cams = [Vector(pos) for _, pos, _, _ in S.shot["camera"]]
        dists = [(p - c).length for points, _ in spots.values() for p in points for c in cams]
        near, far = max(0.2, min(dists) - 1.5), max(dists) + 4.0
    m = bpy.data.materials.new("depth")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    cam = nt.nodes.new("ShaderNodeCameraData")
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value, rng.inputs["From Max"].default_value = near, far
    rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 1.0, 0.0
    rng.clamp = True
    emit = nt.nodes.new("ShaderNodeEmission")
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(cam.outputs["View Z Depth"], rng.inputs["Value"])
    nt.links.new(rng.outputs["Result"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], outn.inputs["Surface"])
    hidden = [ob for ob in scene.objects if ob.name.startswith("flash") or ob.name.startswith("frustum")]
    for ob in hidden:
        ob.hide_render = True
        if ob.animation_data:
            ob.animation_data.action = None
    layer = bpy.context.view_layer
    samples, denoise = scene.cycles.samples, scene.cycles.use_denoising
    scene.cycles.samples, scene.cycles.use_denoising = 1, False
    layer.material_override = m
    ink, scene.render.use_freestyle = scene.render.use_freestyle, False  # depth maps carry no ink lines
    bg = scene.world.node_tree.nodes["Background"].inputs[0].default_value[:]
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    scene.render.image_settings.file_format = "PNG"
    with tempfile.TemporaryDirectory() as tmp:
        scene.render.filepath = str(Path(tmp) / "d")
        bpy.ops.render.render(animation=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(S.fps), "-i", str(Path(tmp) / "d%04d.png"),
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out / f"{S.shot['id']}_depth.mp4")],
                       check=True)
        for t in S.shot.get("frames", [0, S.shot["duration"]]):  # control images for keyframes.py
            shutil.copy(Path(tmp) / f"d{S.frame(t):04d}.png", out / f"{S.shot['id']}_t{t:g}_depth.png")
    layer.material_override = None
    scene.render.use_freestyle = ink
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = bg
    scene.cycles.samples, scene.cycles.use_denoising = samples, denoise
    print(f"+ {out / (S.shot['id'] + '_depth.mp4')} (depth {near:.1f}-{far:.1f} m)")


def build_shot(spec: dict, shot: dict, gpu: str) -> tuple[Shot, object, list, dict]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    S = Shot(spec, shot)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    use_gpu(scene, gpu)
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = spec.get("width", 1344), spec.get("height", 768)
    scene.render.fps = S.fps
    scene.frame_start, scene.frame_end = 1, S.frame(shot["duration"])
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.02, 0.03, 1)
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 1.5
    sun.rotation_euler = (math.radians(35), math.radians(15), math.radians(30))
    scene.collection.objects.link(sun)
    for i, l in enumerate(spec.get("lights", [])):  # fixed fill lights: {"at", "energy" (W), "color", "radius"}
        if "only" in l and shot["id"] not in l["only"]:
            continue
        ob = bpy.data.objects.new(f"light{i}", bpy.data.lights.new(f"light{i}", "POINT"))
        ob.data.energy, ob.data.color = l.get("energy", 300), l.get("color", (1, 1, 1))
        ob.data.shadow_soft_size = l.get("radius", 0.5)
        ob.location = l["at"]
        scene.collection.objects.link(ob)
    if spec.get("outline"):  # ink lines (Freestyle) of this many pixels on silhouettes, creases and borders
        scene.render.use_freestyle = True
        scene.render.line_thickness_mode = "ABSOLUTE"
        scene.render.line_thickness = spec["outline"]
        fs = bpy.context.view_layer.freestyle_settings
        ls = fs.linesets[0] if len(fs.linesets) else fs.linesets.new("ink")
        ls.select_by_visibility, ls.select_by_edge_types = True, True
        ls.select_silhouette = ls.select_border = ls.select_crease = True
        ls.linestyle = ls.linestyle or bpy.data.linestyles.new("ink")
        ls.linestyle.color = (0.02, 0.02, 0.03)
        fs.crease_angle = math.radians(150)  # only real corners get ink, not near-flat faceting
    ceilings = build_set(S)
    spots = {}
    # Mannequins first: spiders and fx may be keyed to their parts ("hero.hand_r").
    order = sorted(shot["cast"], key=lambda c: spec["actors"][c["actor"]]["kind"] == "spider")
    for n, c in enumerate(order):
        a = spec["actors"][c["actor"]]
        name = c.get("name", c["actor"])
        if name in S.cast:
            name = f"{name}{n}"
        if a["kind"] == "spider":
            inst = build_spider_model(S, name, a, c["actor"]) if "model" in a else build_spider(S, name, a)
            S.cast[name] = inst
            spots[name] = (key_spider(S, inst, c), a["color"])
        elif a["kind"] == "model":
            inst = build_model(S, name, a, c["actor"])
            S.cast[name] = inst
            spots[name] = (key_model(S, inst, c), a["color"])
        else:
            inst = build_mannequin(S, name, a)
            S.cast[name] = inst
            spots[name] = (key_mannequin(S, inst, c), a["color"])
        cast_visibility(S, inst, c)
    for i, e in enumerate(shot.get("fx", [])):
        build_fx(S, i, e)
    cam = build_camera(S)
    scene.camera = cam
    scene.frame_set(1)
    return S, cam, ceilings, spots


def main():
    path, only, gpu, anim = args()
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["_dir"] = str(path.resolve().parent)  # "model" paths are relative to blocking.json
    out = path.resolve().parent / "blocking"  # absolute: Blender resolves relative render paths oddly
    out.mkdir(exist_ok=True)
    for shot in spec["shots"]:
        if only and shot["id"] not in only:
            continue
        S, cam, ceilings, spots = build_shot(spec, shot, gpu)
        scene = bpy.context.scene
        bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{shot['id']}.blend"))
        for t in shot.get("frames", [0, shot["duration"]]):
            scene.frame_set(S.frame(t))
            scene.render.filepath = str(out / f"{shot['id']}_t{t:g}.png")
            bpy.ops.render.render(write_still=True)
            print(f"+ {scene.render.filepath}")
        if anim:
            render_animatic(S, out)
            render_depth(S, out, spots)
        # plan: top-down orthographic, set axis horizontal (+y to the right), ceilings hidden
        scene.frame_set(1)
        frustum(S, cam)
        for ob in ceilings:
            ob.hide_render = True
        for name, (points, color) in spots.items():
            for p in points:
                S.mesh("cyl", f"spot_{name}", Matrix.Translation((p.x, p.y, 2.9)), (0.18, 0.18, 0.02),
                       S.mat(color, 2))
        plan = spec.get("plan", {})
        top = bpy.data.objects.new("top", bpy.data.cameras.new("top"))
        top.data.type = "ORTHO"
        top.data.ortho_scale = plan.get("scale", 34)
        cx, cy = plan.get("center", [0, 15])
        top.location = (cx, cy, 30)
        top.rotation_euler = (0, 0, math.radians(90))
        scene.collection.objects.link(top)
        scene.camera = top
        scene.render.filepath = str(out / f"{shot['id']}_plan.png")
        bpy.ops.render.render(write_still=True)
        print(f"+ {scene.render.filepath}")


if __name__ == "__main__":
    main()
