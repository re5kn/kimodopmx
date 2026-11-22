# SPDX-License-Identifier: Apache-2.0
"""Demo glue for MikuMikuDance: PMX model loading, preview and VMD export."""

from __future__ import annotations

import os
import traceback
from typing import Optional

import numpy as np

from kimodo.mmd import MMDRetargeter, PmxModel, load_pmx

from .state import ClientSession


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def get_retargeter(session: ClientSession) -> Optional[MMDRetargeter]:
    """Retargeter for the session's PMX model and current skeleton (cached)."""
    pmx = session.pmx_model
    skeleton = session.skeleton
    if pmx is None or skeleton is None:
        return None
    rt = session.pmx_retargeter
    key = (id(pmx), getattr(skeleton, "name", type(skeleton).__name__))
    if rt is not None and getattr(rt, "_session_key", None) == key:
        return rt
    try:
        rt = MMDRetargeter(
            pmx,
            list(skeleton.bone_order_names),
            _np(skeleton.joint_parents),
            _np(skeleton.neutral_joints),
        )
    except ValueError:
        # e.g. G1 robot skeleton: no MMD mapping.
        session.pmx_retargeter = None
        return None
    rt._session_key = key
    session.pmx_retargeter = rt
    return rt


def attach_pmx_to_motion(session: ClientSession, motion, opacity: float = 1.0, dark_mode: bool = False) -> bool:
    """Create the PMX preview mesh for one CharacterMotion. Returns True on success."""
    rt = get_retargeter(session)
    if rt is None:
        return False
    character = motion.character
    character.attach_pmx(rt, visible=session.pmx_visible, dark_mode=dark_mode, opacity=opacity)
    character.precompute_pmx(motion.joints_pos, motion.joints_rot)
    if motion.cur_frame_idx is not None:
        idx = min(motion.cur_frame_idx, motion.length - 1)
        character.pmx_rig.set_pose(motion.joints_pos[idx], motion.joints_rot[idx], frame_idx=idx)
    return True


def set_session_pmx(
    session: ClientSession,
    pmx: Optional[PmxModel],
    opacity: float = 1.0,
    dark_mode: bool = False,
) -> int:
    """Install (or remove, with pmx=None) a PMX model for every motion of the session."""
    session.pmx_model = pmx
    session.pmx_retargeter = None
    n = 0
    for motion in session.motions.values():
        motion.character.detach_pmx()
        if pmx is None:
            continue
        try:
            if attach_pmx_to_motion(session, motion, opacity=opacity, dark_mode=dark_mode):
                n += 1
        except Exception:
            traceback.print_exc()
    return n


def load_pmx_source(path_or_bytes) -> PmxModel:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return load_pmx(bytes(path_or_bytes))
    path = os.path.expanduser(str(path_or_bytes).strip())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PMX file not found: {path}")
    return load_pmx(path)


def describe_pmx(pmx: PmxModel, rt: Optional[MMDRetargeter]) -> str:
    lines = [
        f"**{pmx.name or '(no name)'}**",
        f"頂点 {len(pmx.vertices):,} / 面 {len(pmx.faces):,} / ボーン {len(pmx.bones)}",
    ]
    if rt is not None:
        names = [pmx.bones[i].name for i in rt.mapped_bones]
        lines.append(f"対応ボーン {len(names)} 本, スケール {rt.scale:.2f} 単位/m")
    else:
        lines.append("このスケルトンはMMDボーンに対応していません")
    return "  \n".join(lines)


def motion_to_vmd_bytes(session: ClientSession, motion, leg_ik: bool = True) -> bytes:
    rt = get_retargeter(session)
    if rt is None:
        if session.pmx_model is None:
            raise RuntimeError("Load a PMX model first (MMD (PMX / VMD) panel).")
        raise RuntimeError("VMD export is only available for SOMA and SMPL-X models.")
    return rt.export_vmd_bytes(
        _np(motion.joints_rot),
        _np(motion.joints_pos),
        fps=float(session.model_fps),
        leg_ik=leg_ik,
    )
