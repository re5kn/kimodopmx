# SPDX-License-Identifier: Apache-2.0
"""Retarget Kimodo skeleton motion (SOMA 30/77 or SMPL-X 22) onto a PMX model.

Method (global rotation transfer with rest-pose compensation):

1. Kimodo world (right-handed, Y-up, faces +Z, meters) is mirrored to MMD world
   (left-handed, Y-up, faces -Z) by flipping Z.  Rotations become S R S.
2. Kimodo joint rotations are relative to its neutral T-pose. A PMX model's rest
   pose is whatever its bone positions describe (often an A-pose). For every mapped
   bone we compute a constant correction C that rotates the PMX rest bone direction
   onto the Kimodo rest bone direction, so ``G_pmx(t) = G_kimodo(t) @ C``.
3. Local VMD rotations are obtained by walking the PMX hierarchy (including
   "append"/付与 helper bones) so that every mapped bone reaches its target global
   rotation. Upper-arm and forearm twist is moved to 腕捩 / 手捩 when present.
4. The root translation goes to センター, scaled by the leg-length ratio. Leg IK
   targets (足ＩＫ) are keyed at the FK ankle so the motion also works with IK on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .pmx import BONE_FLAG_APPEND_ROTATE, BONE_FLAG_APPEND_TRANSLATE, PmxModel
from .rotation import (
    flip_points,
    flip_rotmats,
    rotation_between,
    rotmat_power,
    rotmat_to_quat_xyzw,
    slerp_quats,
    swing_twist,
)
from .vmd import VMD_FPS, VmdBoneFrame, VmdMotion, VmdPropertyFrame

# (source joint, target bone, source direction child, target direction child)
# A direction pair of None means "no rest-pose correction" for torso bones, or
# "inherit the parent's correction" for limb ends (see _build).
_SIDES = (("Left", "左", "left"), ("Right", "右", "right"))


def _soma_mapping() -> list[tuple[str, str, Optional[str], Optional[str]]]:
    m: list[tuple[str, str, Optional[str], Optional[str]]] = [
        ("Hips", "下半身", None, None),
        ("Spine2", "上半身", None, None),
        ("Chest", "上半身2", None, None),
        ("Neck1", "首", None, None),
        ("Head", "頭", None, None),
    ]
    for en, jp, _ in _SIDES:
        m += [
            (f"{en}Shoulder", f"{jp}肩", f"{en}Arm", f"{jp}腕"),
            (f"{en}Arm", f"{jp}腕", f"{en}ForeArm", f"{jp}ひじ"),
            (f"{en}ForeArm", f"{jp}ひじ", f"{en}Hand", f"{jp}手首"),
            (f"{en}Hand", f"{jp}手首", f"{en}HandMiddle2", f"{jp}中指１"),
            (f"{en}HandThumb1", f"{jp}親指０", f"{en}HandThumb2", f"{jp}親指１"),
            (f"{en}HandThumb2", f"{jp}親指１", f"{en}HandThumb3", f"{jp}親指２"),
            (f"{en}HandThumb3", f"{jp}親指２", f"{en}HandThumbEnd", f"{jp}親指先"),
        ]
        for fen, fjp in (("Index", "人指"), ("Middle", "中指"), ("Ring", "薬指"), ("Pinky", "小指")):
            m += [
                (f"{en}Hand{fen}1", f"{jp}{fjp}０", f"{en}Hand{fen}2", f"{jp}{fjp}１"),
                (f"{en}Hand{fen}2", f"{jp}{fjp}１", f"{en}Hand{fen}3", f"{jp}{fjp}２"),
                (f"{en}Hand{fen}3", f"{jp}{fjp}２", f"{en}Hand{fen}4", f"{jp}{fjp}３"),
                (f"{en}Hand{fen}4", f"{jp}{fjp}３", f"{en}Hand{fen}End", f"{jp}{fjp}先"),
            ]
        m += [
            (f"{en}Leg", f"{jp}足", f"{en}Shin", f"{jp}ひざ"),
            (f"{en}Shin", f"{jp}ひざ", f"{en}Foot", f"{jp}足首"),
            (f"{en}Foot", f"{jp}足首", (f"{en}ToeEnd", f"{en}ToeBase"), f"{jp}つま先"),
        ]
    return m


def _smplx_mapping() -> list[tuple[str, str, Optional[str], Optional[str]]]:
    m: list[tuple[str, str, Optional[str], Optional[str]]] = [
        ("pelvis", "下半身", None, None),
        ("spine2", "上半身", None, None),
        ("spine3", "上半身2", None, None),
        ("neck", "首", None, None),
        ("head", "頭", None, None),
    ]
    for _, jp, en in _SIDES:
        m += [
            (f"{en}_collar", f"{jp}肩", f"{en}_shoulder", f"{jp}腕"),
            (f"{en}_shoulder", f"{jp}腕", f"{en}_elbow", f"{jp}ひじ"),
            (f"{en}_elbow", f"{jp}ひじ", f"{en}_wrist", f"{jp}手首"),
            (f"{en}_wrist", f"{jp}手首", None, None),
            (f"{en}_hip", f"{jp}足", f"{en}_knee", f"{jp}ひざ"),
            (f"{en}_knee", f"{jp}ひざ", f"{en}_ankle", f"{jp}足首"),
            (f"{en}_ankle", f"{jp}足首", f"{en}_foot", f"{jp}つま先"),
        ]
    return m


# Torso bones never get a rest-pose correction.
_TORSO = {"下半身", "上半身", "上半身2", "首", "頭"}

CENTER = "センター"
LEG_IK = {"左": ("左足ＩＫ", "左つま先ＩＫ", "左足首"), "右": ("右足ＩＫ", "右つま先ＩＫ", "右足首")}
TWIST = {"腕": "腕捩", "ひじ": "手捩"}


def detect_mapping(joint_names: Sequence[str]):
    names = set(joint_names)
    if {"Hips", "LeftArm", "RightLeg"} <= names:
        return _soma_mapping()
    if {"pelvis", "left_shoulder", "right_hip"} <= names:
        return _smplx_mapping()
    raise ValueError("Unsupported skeleton for MMD export (expected SOMA or SMPL-X joints).")


@dataclass
class RetargetResult:
    """Retargeted motion on the PMX model (all in MMD space / units)."""

    keyed_bones: list[str]
    local_quats: np.ndarray  # [T, K, 4] xyzw rotation per keyed bone
    local_offsets: np.ndarray  # [T, K, 3] translation offsets per keyed bone
    movable: set  # names of keyed bones whose offsets are meaningful
    global_rots: np.ndarray  # [T, B, 3, 3] PMX bone global rotations (FK)
    global_pos: np.ndarray  # [T, B, 3] PMX bone global positions (FK)
    fps: float


class MMDRetargeter:
    def __init__(
        self,
        pmx: PmxModel,
        joint_names: Sequence[str],
        joint_parents: Sequence[int],
        neutral_joints: np.ndarray,
        mapping=None,
    ):
        self.pmx = pmx
        self.joint_names = list(joint_names)
        self.joint_parents = np.asarray(joint_parents, dtype=np.int64)
        self.src_rest = flip_points(np.asarray(neutral_joints, dtype=np.float64))
        self.mapping = mapping if mapping is not None else detect_mapping(self.joint_names)
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        pmx = self.pmx
        jidx = {n: i for i, n in enumerate(self.joint_names)}
        self.tgt_rest = pmx.bone_positions
        self.tgt_parents = pmx.bone_parents
        nb = len(pmx.bones)

        # Mapped pairs that exist on both sides.
        entries = []
        for src, tgt, src_c, tgt_c in self.mapping:
            if src not in jidx or pmx.bone_index(tgt) < 0:
                continue
            entries.append((jidx[src], pmx.bone_index(tgt), src_c, tgt_c, tgt))
        if not entries:
            raise ValueError("No standard MMD bones found in the PMX model.")

        # Rest-pose corrections, parents first (mapping is ordered root->leaf).
        corr: dict[int, np.ndarray] = {}
        self.map_src = np.zeros(nb, dtype=np.int64) - 1
        self.corr = np.tile(np.eye(3), (nb, 1, 1))
        for s, t, src_c, tgt_c, tname in entries:
            C = np.eye(3)
            if tname not in _TORSO:
                cands = (src_c,) if isinstance(src_c, str) else (src_c or ())
                sc = next((jidx[c] for c in cands if c in jidx), -1)
                tc = pmx.bone_index(tgt_c) if tgt_c else -1
                if sc >= 0 and tc >= 0:
                    C = rotation_between(self.tgt_rest[tc] - self.tgt_rest[t], self.src_rest[sc] - self.src_rest[s])
                else:
                    # Inherit the correction of the closest mapped PMX ancestor.
                    p = self.tgt_parents[t]
                    while p >= 0 and p not in corr:
                        p = self.tgt_parents[p]
                    if p >= 0:
                        C = corr[p]
            corr[t] = C
            self.map_src[t] = s
            self.corr[t] = C

        # Twist bones (腕捩 / 手捩) that sit between a mapped bone and its mapped child.
        self.twist_of: dict[int, tuple[int, np.ndarray]] = {}
        for _, jp, _ in _SIDES:
            for base, tw in TWIST.items():
                b, w = pmx.bone_index(jp + base), pmx.bone_index(jp + tw)
                if b < 0 or w < 0 or self.map_src[b] < 0 or self.tgt_parents[w] < 0:
                    continue
                # The twist bone must be a descendant of the base bone.
                p = self.tgt_parents[w]
                while p >= 0 and p != b:
                    p = self.tgt_parents[p]
                if p != b:
                    continue
                fa = pmx.bones[w].fixed_axis
                if fa is None or np.linalg.norm(fa) < 1e-6:
                    child = [i for i in range(nb) if self.tgt_parents[i] == w]
                    if not child:
                        continue
                    fa = self.tgt_rest[child[0]] - self.tgt_rest[w]
                self.twist_of[b] = (w, np.asarray(fa, dtype=np.float64))

        # FK evaluation order: parents and append parents first.
        order, state = [], np.zeros(nb, dtype=np.int8)

        def visit(i: int) -> None:
            if state[i] == 2:
                return
            if state[i] == 1:  # cycle; give up on strict ordering for this bone
                return
            state[i] = 1
            p = self.tgt_parents[i]
            if 0 <= p < nb:
                visit(p)
            b = pmx.bones[i]
            if b.flags & (BONE_FLAG_APPEND_ROTATE | BONE_FLAG_APPEND_TRANSLATE) and 0 <= b.append_parent < nb:
                visit(b.append_parent)
            state[i] = 2
            order.append(i)

        for i in range(nb):
            visit(i)
        self.order = order

        # Scale: PMX units per Kimodo meter, from leg length (hip joint -> ankle).
        ratios = []
        for (en, jp, en2) in _SIDES:
            for hip, ank in ((f"{en}Leg", f"{en}Foot"), (f"{en2}_hip", f"{en2}_ankle")):
                if hip in jidx and ank in jidx and pmx.bone_index(jp + "足") >= 0 and pmx.bone_index(jp + "足首") >= 0:
                    ls = np.linalg.norm(self.src_rest[jidx[hip]] - self.src_rest[jidx[ank]])
                    lt = np.linalg.norm(
                        self.tgt_rest[pmx.bone_index(jp + "足")] - self.tgt_rest[pmx.bone_index(jp + "足首")]
                    )
                    if ls > 1e-6:
                        ratios.append(lt / ls)
        self.scale = float(np.mean(ratios)) if ratios else 1.0 / 0.08

        # Ground alignment: PMX ankle height vs. scaled Kimodo ankle height above the floor.
        # Kimodo puts its floor 2 cm below the toe-base joints of the neutral pose.
        self.src_hip_joints = [jidx[n] for n in ("LeftLeg", "RightLeg", "left_hip", "right_hip") if n in jidx]
        self.tgt_hip_bones = [pmx.bone_index(n) for n in ("左足", "右足") if pmx.bone_index(n) >= 0]
        self.y_adjust = 0.0
        src_ank = [jidx[n] for n in ("LeftFoot", "RightFoot", "left_ankle", "right_ankle") if n in jidx]
        tgt_ank = [pmx.bone_index(n) for n in ("左足首", "右足首") if pmx.bone_index(n) >= 0]
        if src_ank and tgt_ank:
            toe = [jidx[n] for n in ("LeftToeBase", "RightToeBase", "left_foot", "right_foot") if n in jidx]
            floor = (self.src_rest[toe, 1].min() if toe else self.src_rest[:, 1].min()) - 0.02
            src_h = np.mean(self.src_rest[src_ank, 1]) - floor
            tgt_h = np.mean(self.tgt_rest[tgt_ank, 1])
            self.y_adjust = float(tgt_h - self.scale * src_h)

        self.center = pmx.bone_index(CENTER)
        self.mapped_bones = [i for i in range(nb) if self.map_src[i] >= 0]

    # --------------------------------------------------------------- retarget
    def _fk(self, local: np.ndarray, offsets: dict[int, np.ndarray]):
        """FK over PMX bones. local: [T, B, 3, 3] (modified for append bones)."""
        pmx = self.pmx
        T, nb = local.shape[:2]
        G = np.zeros((T, nb, 3, 3))
        P = np.zeros((T, nb, 3))
        eff = local.copy()
        for i in self.order:
            b = pmx.bones[i]
            if b.flags & BONE_FLAG_APPEND_ROTATE and 0 <= b.append_parent < nb and b.append_ratio != 0.0:
                eff[:, i] = rotmat_power(eff[:, b.append_parent], b.append_ratio) @ local[:, i]
            p = self.tgt_parents[i]
            rel = self.tgt_rest[i] - (self.tgt_rest[p] if p >= 0 else 0.0)
            off = offsets.get(i)
            if off is not None:
                rel = rel + off
            if p >= 0:
                G[:, i] = G[:, p] @ eff[:, i]
                P[:, i] = P[:, p] + np.einsum("tij,tj->ti", G[:, p], np.broadcast_to(rel, (T, 3)))
            else:
                G[:, i] = eff[:, i]
                P[:, i] = np.broadcast_to(rel, (T, 3))
        return G, P

    def retarget(
        self,
        global_rots: np.ndarray,
        joints_pos: np.ndarray,
        fps: float = VMD_FPS,
        leg_ik: bool = True,
    ) -> RetargetResult:
        """global_rots: [T, J, 3, 3] Kimodo global joint rotations; joints_pos: [T, J, 3] (meters)."""
        pmx = self.pmx
        Gs = flip_rotmats(np.asarray(global_rots, dtype=np.float64))
        Ps = flip_points(np.asarray(joints_pos, dtype=np.float64))
        T = Gs.shape[0]
        nb = len(pmx.bones)

        local = np.tile(np.eye(3), (T, nb, 1, 1))
        Gp_cache = np.zeros((T, nb, 3, 3))
        eff = local.copy()
        # Solve local rotations parent-first so each mapped bone hits its target.
        for i in self.order:
            b = pmx.bones[i]
            p = self.tgt_parents[i]
            Gp = Gp_cache[:, p] if p >= 0 else np.tile(np.eye(3), (T, 1, 1))
            append = None
            if b.flags & BONE_FLAG_APPEND_ROTATE and 0 <= b.append_parent < nb and b.append_ratio != 0.0:
                append = rotmat_power(eff[:, b.append_parent], b.append_ratio)
            if self.map_src[i] >= 0:
                target = Gs[:, self.map_src[i]] @ self.corr[i]
                L = np.swapaxes(Gp, -1, -2) @ target
                if append is not None:
                    L = np.swapaxes(append, -1, -2) @ L
                if i in self.twist_of:
                    w, axis = self.twist_of[i]
                    L, twist = swing_twist(L, axis)
                    local[:, w] = twist
                local[:, i] = L
            eff[:, i] = local[:, i] if append is None else append @ local[:, i]
            Gp_cache[:, i] = Gp @ eff[:, i]

        # Root translation on センター so that the hip joints follow the scaled source hips.
        offsets: dict[int, np.ndarray] = {}
        G0, P0 = self._fk(local, offsets)
        if self.center >= 0 and self.src_hip_joints and self.tgt_hip_bones:
            src_mid = Ps[:, self.src_hip_joints].mean(axis=1) * self.scale
            src_mid[:, 1] += self.y_adjust
            tgt_mid = P0[:, self.tgt_hip_bones].mean(axis=1)
            delta = src_mid - tgt_mid  # world-space translation to apply
            cp = self.tgt_parents[self.center]
            if cp >= 0:  # express in the parent's frame
                delta = np.einsum("tji,tj->ti", G0[:, cp], delta)
            offsets[self.center] = delta
        G, P = self._fk(local, offsets)

        # Leg IK targets at the FK ankle (position) with the ankle orientation.
        if leg_ik:
            for side, (ik_name, toe_ik_name, ankle_name) in LEG_IK.items():
                ik, ank = pmx.bone_index(ik_name), pmx.bone_index(ankle_name)
                if ik < 0 or ank < 0:
                    continue
                p = self.tgt_parents[ik]
                if p >= 0:
                    Rp, tp = G[:, p], P[:, p]
                    rel_rest = self.tgt_rest[ik] - self.tgt_rest[p]
                else:
                    Rp, tp = np.tile(np.eye(3), (T, 1, 1)), np.zeros((T, 3))
                    rel_rest = self.tgt_rest[ik]
                RpT = np.swapaxes(Rp, -1, -2)
                offsets[ik] = np.einsum("tij,tj->ti", RpT, P[:, ank] - tp) - rel_rest
                local[:, ik] = RpT @ G[:, ank]

        keyed: list[int] = []
        if self.center >= 0:
            keyed.append(self.center)
        keyed += self.mapped_bones
        keyed += [w for (w, _) in self.twist_of.values()]
        if leg_ik:
            for ik_name, toe_ik_name, _ in LEG_IK.values():
                for n in (ik_name, toe_ik_name):
                    if pmx.bone_index(n) >= 0:
                        keyed.append(pmx.bone_index(n))
        keyed = list(dict.fromkeys(keyed))
        quats = rotmat_to_quat_xyzw(local[:, keyed])
        offs = np.zeros((T, len(keyed), 3))
        for k, i in enumerate(keyed):
            if i in offsets:
                offs[:, k] = offsets[i]
        movable = {pmx.bones[i].name for i in keyed if i in offsets}
        return RetargetResult(
            keyed_bones=[pmx.bones[i].name for i in keyed],
            local_quats=quats,
            local_offsets=offs,
            movable=movable,
            global_rots=G,
            global_pos=P,
            fps=float(fps),
        )

    # --------------------------------------------------------------- export
    def to_vmd(self, result: RetargetResult, leg_ik: bool = True) -> VmdMotion:
        quats, offs = _resample(result.local_quats, result.local_offsets, result.fps, VMD_FPS)
        motion = VmdMotion(model_name=self.pmx.name)
        T = quats.shape[0]
        for k, name in enumerate(result.keyed_bones):
            use_pos = name in result.movable
            for t in range(T):
                pos = tuple(float(x) for x in offs[t, k]) if use_pos else (0.0, 0.0, 0.0)
                motion.bone_frames.append(VmdBoneFrame(name, t, pos, tuple(float(x) for x in quats[t, k])))
        ik_names = [n for n, _, _ in LEG_IK.values()] + [n for _, n, _ in LEG_IK.values()]
        ik_names = [n for n in ik_names if self.pmx.bone_index(n) >= 0]
        if ik_names:
            motion.property_frames.append(VmdPropertyFrame(0, True, {n: bool(leg_ik) for n in ik_names}))
        return motion

    def export_vmd_bytes(self, global_rots, joints_pos, fps: float, leg_ik: bool = True) -> bytes:
        res = self.retarget(global_rots, joints_pos, fps=fps, leg_ik=leg_ik)
        return self.to_vmd(res, leg_ik=leg_ik).to_bytes()


def _resample(quats: np.ndarray, offs: np.ndarray, src_fps: float, dst_fps: float):
    if abs(src_fps - dst_fps) < 1e-6 or quats.shape[0] < 2:
        return quats, offs
    T = quats.shape[0]
    duration = (T - 1) / src_fps
    n_out = int(round(duration * dst_fps)) + 1
    t_src = np.arange(n_out) * (src_fps / dst_fps)
    i0 = np.clip(np.floor(t_src).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    a = (t_src - i0)[:, None]
    q = slerp_quats(quats[i0], quats[i1], np.broadcast_to(a, quats[i0].shape[:-1]))
    o = offs[i0] * (1 - a[..., None]) + offs[i1] * a[..., None]
    return q, o
