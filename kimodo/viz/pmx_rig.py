# SPDX-License-Identifier: Apache-2.0
"""Show a PMX (MikuMikuDance) model in the viser scene, driven by Kimodo motion.

The PMX pose for each frame comes from :class:`kimodo.mmd.MMDRetargeter`, i.e. the
exact same retargeting that is written to VMD, so the preview matches the export.
The model is drawn as a GPU skinned mesh (textures are not loaded, one color).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import viser
import viser.transforms as tf
from kimodo.mmd import MMDRetargeter, PmxModel
from kimodo.mmd.rotation import FLIP_Z

PMX_COLOR_LIGHT = (245, 200, 210)
PMX_COLOR_DARK = (200, 150, 165)


def _visible_faces(pmx: PmxModel) -> np.ndarray:
    """Faces of materials that are not fully transparent."""
    keep = []
    start = 0
    for m in pmx.materials:
        if m.diffuse[3] > 0.01 and m.face_count > 0:
            keep.append(pmx.faces[start : start + m.face_count])
        start += m.face_count
    if start < len(pmx.faces):
        keep.append(pmx.faces[start:])
    return np.concatenate(keep) if keep else pmx.faces


class PMXRig:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        retargeter: MMDRetargeter,
        dark_mode: bool = False,
        visible: bool = True,
        opacity: Optional[float] = None,
    ):
        self.server = server
        self.name = name
        self.retargeter = retargeter
        pmx = retargeter.pmx
        self.inv_scale = 1.0 / retargeter.scale

        # Only bones that actually deform vertices are sent to the client.
        sidx = pmx.skin_indices
        swt = pmx.skin_weights
        valid = (sidx >= 0) & (sidx < len(pmx.bones)) & (swt > 0)
        used = np.unique(sidx[valid])
        self.used_bones = used
        remap = np.full(len(pmx.bones), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))

        V = len(pmx.vertices)
        dense = np.zeros((V, len(used)), dtype=np.float32)
        rows = np.repeat(np.arange(V)[:, None], 4, axis=1)
        np.add.at(dense, (rows[valid], remap[sidx[valid]]), swt[valid])
        dense /= np.maximum(dense.sum(axis=1, keepdims=True), 1e-8)

        verts = (pmx.vertices.astype(np.float64) @ FLIP_Z) * self.inv_scale
        rest = (pmx.bone_positions[used] @ FLIP_Z) * self.inv_scale
        wxyz = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(used), 1))

        self.mesh = server.scene.add_mesh_skinned(
            f"/{name}/pmx",
            vertices=verts.astype(np.float32),
            faces=_visible_faces(pmx),
            bone_wxyzs=wxyz,
            bone_positions=rest,
            skin_weights=dense,
            color=PMX_COLOR_DARK if dark_mode else PMX_COLOR_LIGHT,
            material="toon3",
            side="double",
            opacity=opacity,
            visible=visible,
        )
        self.cache_rots: Optional[np.ndarray] = None  # [T, B', 3, 3] world rotations
        self.cache_pos: Optional[np.ndarray] = None  # [T, B', 3] world positions

    # ------------------------------------------------------------ retargeting
    def _to_world(self, G: np.ndarray, P: np.ndarray):
        G = G[..., self.used_bones, :, :]
        P = P[..., self.used_bones, :]
        return FLIP_Z @ G @ FLIP_Z, (P @ FLIP_Z) * self.inv_scale

    def _solve(self, joints_pos: np.ndarray, joints_rot: np.ndarray):
        res = self.retargeter.retarget(joints_rot, joints_pos, leg_ik=False)
        return self._to_world(res.global_rots, res.global_pos)

    def precompute(self, joints_pos, joints_rot) -> None:
        """joints_pos: [T, J, 3], joints_rot: [T, J, 3, 3] (Kimodo global)."""
        self.cache_rots, self.cache_pos = self._solve(_np(joints_pos), _np(joints_rot))

    def update_cache(self, joints_pos, joints_rot, frame_idx: int) -> None:
        if self.cache_rots is None or frame_idx >= len(self.cache_rots):
            return
        R, P = self._solve(_np(joints_pos)[None], _np(joints_rot)[None])
        self.cache_rots[frame_idx] = R[0]
        self.cache_pos[frame_idx] = P[0]

    def set_pose(self, joints_pos, joints_rot, frame_idx: Optional[int] = None) -> None:
        if self.cache_rots is not None and frame_idx is not None and frame_idx < len(self.cache_rots):
            R, P = self.cache_rots[frame_idx], self.cache_pos[frame_idx]
        else:
            R, P = self._solve(_np(joints_pos)[None], _np(joints_rot)[None])
            R, P = R[0], P[0]
        wxyzs = tf.SO3.from_matrix(R).wxyz
        for i, bone in enumerate(self.mesh.bones):
            bone.wxyz = wxyzs[i]
            bone.position = P[i]

    # ------------------------------------------------------------ appearance
    def set_visibility(self, visible: bool) -> None:
        self.mesh.visible = visible

    def set_opacity(self, opacity: float) -> None:
        self.mesh.opacity = None if opacity >= 1.0 else opacity

    def set_wireframe(self, wireframe: bool) -> None:
        self.mesh.wireframe = wireframe

    def set_dark_mode(self, dark: bool) -> None:
        self.mesh.color = PMX_COLOR_DARK if dark else PMX_COLOR_LIGHT

    def clear(self) -> None:
        self.server.scene.remove_by_name(self.mesh.name)


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)
