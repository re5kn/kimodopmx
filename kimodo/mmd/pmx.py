# SPDX-License-Identifier: Apache-2.0
"""Minimal PMX (MikuMikuDance model, v2.0/2.1) reader.

Only the parts needed for retargeting and preview are parsed:
header, vertices (position + skin weights), faces, materials (diffuse color +
face count) and bones (name, parent, position, flags). Morphs, rigid bodies
and joints are skipped.

PMX coordinates are left-handed, Y-up, character facing -Z, 1 unit = 8 cm.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np

# Bone flags (PMX 2.0 spec)
BONE_FLAG_TAIL_IS_BONE = 0x0001
BONE_FLAG_ROTATABLE = 0x0002
BONE_FLAG_MOVABLE = 0x0004
BONE_FLAG_VISIBLE = 0x0008
BONE_FLAG_OPERABLE = 0x0010
BONE_FLAG_IK = 0x0020
BONE_FLAG_LOCAL_APPEND = 0x0080
BONE_FLAG_APPEND_ROTATE = 0x0100
BONE_FLAG_APPEND_TRANSLATE = 0x0200
BONE_FLAG_FIXED_AXIS = 0x0400
BONE_FLAG_LOCAL_AXIS = 0x0800
BONE_FLAG_PHYSICS_AFTER_DEFORM = 0x1000
BONE_FLAG_EXTERNAL_PARENT = 0x2000

# 1 MMD unit is commonly treated as 8 cm.
MMD_UNIT_IN_METERS = 0.08


@dataclass
class PmxBone:
    name: str
    name_en: str
    position: np.ndarray  # (3,) in PMX model space
    parent: int
    layer: int
    flags: int
    tail_index: int = -1
    tail_offset: np.ndarray | None = None
    append_parent: int = -1
    append_ratio: float = 0.0
    fixed_axis: np.ndarray | None = None
    local_axis_x: np.ndarray | None = None
    local_axis_z: np.ndarray | None = None
    ik_target: int = -1
    ik_links: list[int] = field(default_factory=list)

    @property
    def is_ik(self) -> bool:
        return bool(self.flags & BONE_FLAG_IK)


@dataclass
class PmxMaterial:
    name: str
    name_en: str
    diffuse: np.ndarray  # (4,) RGBA 0..1
    face_count: int  # number of triangles (spec stores index count)
    texture_index: int = -1


@dataclass
class PmxModel:
    version: float
    name: str
    name_en: str
    comment: str
    comment_en: str
    vertices: np.ndarray  # (V, 3) float32
    normals: np.ndarray  # (V, 3) float32
    uvs: np.ndarray  # (V, 2) float32
    skin_indices: np.ndarray  # (V, 4) int32 (-1 = unused)
    skin_weights: np.ndarray  # (V, 4) float32
    faces: np.ndarray  # (F, 3) int32
    textures: list[str]
    materials: list[PmxMaterial]
    bones: list[PmxBone]

    def bone_index(self, name: str) -> int:
        """Index of the bone with the given Japanese name, or -1."""
        if not hasattr(self, "_bone_lut"):
            self._bone_lut = {}
            for i, b in enumerate(self.bones):
                self._bone_lut.setdefault(b.name, i)
        return self._bone_lut.get(name, -1)

    @property
    def bone_names(self) -> list[str]:
        return [b.name for b in self.bones]

    @property
    def bone_positions(self) -> np.ndarray:
        return np.stack([b.position for b in self.bones]).astype(np.float64)

    @property
    def bone_parents(self) -> np.ndarray:
        return np.array([b.parent for b in self.bones], dtype=np.int64)


class _Reader:
    def __init__(self, f: BinaryIO):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("Unexpected end of PMX file")
        return b

    def unpack(self, fmt: str):
        fmt = "<" + fmt
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def i8(self):
        return self.unpack("b")[0]

    def u8(self):
        return self.unpack("B")[0]

    def i32(self):
        return self.unpack("i")[0]

    def f32(self):
        return self.unpack("f")[0]

    def vec(self, n: int) -> np.ndarray:
        return np.array(self.unpack(f"{n}f"), dtype=np.float32)


def _index_reader(r: _Reader, size: int, unsigned: bool):
    # Vertex indices are unsigned for size 1/2; all other indices are signed.
    if size == 1:
        return (lambda: r.u8()) if unsigned else (lambda: r.i8())
    if size == 2:
        return (lambda: r.unpack("H")[0]) if unsigned else (lambda: r.unpack("h")[0])
    if size == 4:
        return lambda: r.i32()
    raise ValueError(f"Invalid PMX index size: {size}")


def load_pmx(src: Union[str, Path, bytes, BinaryIO]) -> PmxModel:
    """Load a PMX model from a path, raw bytes or a binary file object."""
    if isinstance(src, (str, Path)):
        with open(src, "rb") as f:
            data = f.read()
    elif isinstance(src, (bytes, bytearray, memoryview)):
        data = bytes(src)
    else:
        data = src.read()
    f = io.BytesIO(data)
    r = _Reader(f)

    if r.read(4) != b"PMX ":
        raise ValueError("Not a PMX file (bad magic)")
    version = r.f32()
    n_globals = r.u8()
    g = list(r.read(n_globals))
    encoding = "utf-16-le" if g[0] == 0 else "utf-8"
    n_add_uv = g[1]
    # index sizes: vertex, texture, material, bone, morph, rigid body
    vsz, tsz, bsz = g[2], g[3], g[5]

    def text() -> str:
        n = r.i32()
        return r.read(n).decode(encoding, errors="replace") if n > 0 else ""

    read_tidx = _index_reader(r, tsz, unsigned=False)
    read_bidx = _index_reader(r, bsz, unsigned=False)

    name, name_en, comment, comment_en = text(), text(), text(), text()

    # ---- vertices
    nv = r.i32()
    verts = np.zeros((nv, 3), np.float32)
    normals = np.zeros((nv, 3), np.float32)
    uvs = np.zeros((nv, 2), np.float32)
    sidx = np.full((nv, 4), -1, np.int32)
    swt = np.zeros((nv, 4), np.float32)
    for i in range(nv):
        vals = r.unpack("8f")
        verts[i] = vals[0:3]
        normals[i] = vals[3:6]
        uvs[i] = vals[6:8]
        if n_add_uv:
            r.read(16 * n_add_uv)
        wtype = r.u8()
        if wtype == 0:  # BDEF1
            sidx[i, 0] = read_bidx()
            swt[i, 0] = 1.0
        elif wtype == 1:  # BDEF2
            sidx[i, 0] = read_bidx()
            sidx[i, 1] = read_bidx()
            w = r.f32()
            swt[i, 0], swt[i, 1] = w, 1.0 - w
        elif wtype in (2, 4):  # BDEF4 / QDEF
            for k in range(4):
                sidx[i, k] = read_bidx()
            swt[i] = r.vec(4)
        elif wtype == 3:  # SDEF (approximated as BDEF2)
            sidx[i, 0] = read_bidx()
            sidx[i, 1] = read_bidx()
            w = r.f32()
            swt[i, 0], swt[i, 1] = w, 1.0 - w
            r.read(36)  # C, R0, R1
        else:
            raise ValueError(f"Unknown PMX weight deform type {wtype} at vertex {i}")
        r.f32()  # edge scale

    # ---- faces
    n_idx = r.i32()
    if vsz == 1:
        faces = np.frombuffer(r.read(n_idx), np.uint8)
    elif vsz == 2:
        faces = np.frombuffer(r.read(2 * n_idx), "<u2")
    else:
        faces = np.frombuffer(r.read(4 * n_idx), "<i4")
    faces = faces.astype(np.int32).reshape(-1, 3)

    # ---- textures
    textures = [text() for _ in range(r.i32())]

    # ---- materials
    materials: list[PmxMaterial] = []
    for _ in range(r.i32()):
        mname, mname_en = text(), text()
        diffuse = r.vec(4)
        r.read(4 * (3 + 1 + 3))  # specular, power, ambient
        r.u8()  # draw flags
        r.read(4 * 4 + 4)  # edge color, edge size
        tex = read_tidx()
        read_tidx()  # sphere texture
        r.u8()  # sphere mode
        shared_toon = r.u8()
        if shared_toon == 0:
            read_tidx()
        else:
            r.u8()
        text()  # memo
        count = r.i32()
        materials.append(PmxMaterial(mname, mname_en, diffuse, count // 3, tex))

    # ---- bones
    bones: list[PmxBone] = []
    for _ in range(r.i32()):
        bname, bname_en = text(), text()
        pos = r.vec(3)
        parent = read_bidx()
        layer = r.i32()
        flags = r.unpack("H")[0]
        b = PmxBone(bname, bname_en, pos, parent, layer, flags)
        if flags & BONE_FLAG_TAIL_IS_BONE:
            b.tail_index = read_bidx()
        else:
            b.tail_offset = r.vec(3)
        if flags & (BONE_FLAG_APPEND_ROTATE | BONE_FLAG_APPEND_TRANSLATE):
            b.append_parent = read_bidx()
            b.append_ratio = r.f32()
        if flags & BONE_FLAG_FIXED_AXIS:
            b.fixed_axis = r.vec(3)
        if flags & BONE_FLAG_LOCAL_AXIS:
            b.local_axis_x = r.vec(3)
            b.local_axis_z = r.vec(3)
        if flags & BONE_FLAG_EXTERNAL_PARENT:
            r.i32()
        if flags & BONE_FLAG_IK:
            b.ik_target = read_bidx()
            r.i32()  # loop count
            r.f32()  # limit angle
            for _ in range(r.i32()):
                b.ik_links.append(read_bidx())
                if r.u8():
                    r.read(24)  # angle limits
        bones.append(b)

    return PmxModel(
        version=version,
        name=name,
        name_en=name_en,
        comment=comment,
        comment_en=comment_en,
        vertices=verts,
        normals=normals,
        uvs=uvs,
        skin_indices=sidx,
        skin_weights=swt,
        faces=faces,
        textures=textures,
        materials=materials,
        bones=bones,
    )
