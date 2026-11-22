# SPDX-License-Identifier: Apache-2.0
"""VMD (Vocaloid Motion Data) writer/reader.

Bone keyframes store, per bone and frame:
  * position: offset from the bone's rest position (MMD units, parent-aligned axes)
  * rotation: quaternion (x, y, z, w) relative to the parent, in rest-pose axes
Bone names are Shift-JIS (cp932), max 15 bytes. MMD plays VMD at 30 fps.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import numpy as np

VMD_MAGIC = b"Vocaloid Motion Data 0002"
VMD_FPS = 30.0

# Linear interpolation curve (control points (20,20)-(107,107)) laid out the way
# MikuMikuDance writes it: 4 channels (X, Y, Z, R) x 4 values (x1, y1, x2, y2)
# repeated 4 times, each row shifted one byte to the left (zero padded).
_LINEAR_CURVE = (20, 20, 107, 107)


def _default_interpolation() -> bytes:
    # Build the canonical 64-byte block from per-channel (x1,y1,x2,y2) values.
    x1, y1, x2, y2 = _LINEAR_CURVE
    base = []
    for v in (x1, y1, x2, y2):
        base.extend([v, v, v, v])  # X, Y, Z, R channels share the curve
    # Row k is base shifted left by k bytes, padded with zeros (MMD convention).
    rows = []
    for k in range(4):
        rows.extend(base[k:] + [0] * k)
    return bytes(rows)


DEFAULT_INTERPOLATION = _default_interpolation()


def _encode_name(name: str, size: int) -> bytes:
    raw = name.encode("cp932", errors="replace")
    if len(raw) > size:
        # Avoid cutting a double-byte character in half.
        raw = raw[:size]
        while raw:
            try:
                raw.decode("cp932")
                break
            except UnicodeDecodeError:
                raw = raw[:-1]
    return raw + b"\x00" * (size - len(raw))


def _decode_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("cp932", errors="replace")


@dataclass
class VmdBoneFrame:
    name: str
    frame: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w
    interpolation: bytes = DEFAULT_INTERPOLATION


@dataclass
class VmdMorphFrame:
    name: str
    frame: int
    weight: float


@dataclass
class VmdPropertyFrame:
    """Visibility + IK on/off keyframe ("表示・IK")."""

    frame: int
    visible: bool = True
    ik_states: dict[str, bool] = field(default_factory=dict)


@dataclass
class VmdMotion:
    model_name: str = ""
    bone_frames: list[VmdBoneFrame] = field(default_factory=list)
    morph_frames: list[VmdMorphFrame] = field(default_factory=list)
    property_frames: list[VmdPropertyFrame] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        out = io.BytesIO()
        out.write(VMD_MAGIC + b"\x00" * (30 - len(VMD_MAGIC)))
        out.write(_encode_name(self.model_name, 20))

        out.write(struct.pack("<I", len(self.bone_frames)))
        for kf in self.bone_frames:
            out.write(_encode_name(kf.name, 15))
            out.write(struct.pack("<I3f4f", int(kf.frame), *map(float, kf.position), *map(float, kf.rotation)))
            interp = kf.interpolation
            if len(interp) != 64:
                raise ValueError("VMD interpolation block must be 64 bytes")
            out.write(interp)

        out.write(struct.pack("<I", len(self.morph_frames)))
        for mf in self.morph_frames:
            out.write(_encode_name(mf.name, 15))
            out.write(struct.pack("<If", int(mf.frame), float(mf.weight)))

        out.write(struct.pack("<I", 0))  # camera
        out.write(struct.pack("<I", 0))  # light
        out.write(struct.pack("<I", 0))  # self shadow

        out.write(struct.pack("<I", len(self.property_frames)))
        for pf in self.property_frames:
            out.write(struct.pack("<IB", int(pf.frame), 1 if pf.visible else 0))
            out.write(struct.pack("<I", len(pf.ik_states)))
            for ik_name, enabled in pf.ik_states.items():
                out.write(_encode_name(ik_name, 20))
                out.write(struct.pack("<B", 1 if enabled else 0))
        return out.getvalue()

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            f.write(self.to_bytes())

    def bone_tracks(self) -> dict[str, list[VmdBoneFrame]]:
        tracks: dict[str, list[VmdBoneFrame]] = {}
        for kf in self.bone_frames:
            tracks.setdefault(kf.name, []).append(kf)
        for v in tracks.values():
            v.sort(key=lambda k: k.frame)
        return tracks


def load_vmd(src: Union[str, Path, bytes]) -> VmdMotion:
    """Read bone, morph and property keyframes from a VMD file."""
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else bytes(src)
    if not data.startswith(b"Vocaloid Motion Data"):
        raise ValueError("Not a VMD file")
    off = 30
    motion = VmdMotion(model_name=_decode_name(data[off : off + 20]))
    off += 20

    def u32() -> int:
        nonlocal off
        (v,) = struct.unpack_from("<I", data, off)
        off += 4
        return v

    for _ in range(u32()):
        name = _decode_name(data[off : off + 15])
        frame, px, py, pz, qx, qy, qz, qw = struct.unpack_from("<I3f4f", data, off + 15)
        interp = data[off + 47 : off + 111]
        motion.bone_frames.append(VmdBoneFrame(name, frame, (px, py, pz), (qx, qy, qz, qw), interp))
        off += 111
    if off >= len(data):
        return motion
    for _ in range(u32()):
        name = _decode_name(data[off : off + 15])
        frame, w = struct.unpack_from("<If", data, off + 15)
        motion.morph_frames.append(VmdMorphFrame(name, frame, w))
        off += 23
    if off >= len(data):
        return motion
    count = u32()  # camera
    off += 61 * count
    if off >= len(data):
        return motion
    count = u32()  # light
    off += 28 * count
    if off >= len(data):
        return motion
    count = u32()  # self shadow
    off += 9 * count
    if off >= len(data):
        return motion
    for _ in range(u32()):
        frame, vis = struct.unpack_from("<IB", data, off)
        off += 5
        states = {}
        for _ in range(u32()):
            states[_decode_name(data[off : off + 20])] = bool(data[off + 20])
            off += 21
        motion.property_frames.append(VmdPropertyFrame(frame, bool(vis), states))
    return motion


def bone_frames_from_arrays(
    names: Iterable[str],
    positions: np.ndarray,
    rotations_xyzw: np.ndarray,
    frame_offset: int = 0,
    keep_position: Iterable[str] = (),
) -> list[VmdBoneFrame]:
    """Make keyframes for every frame of every bone.

    positions: [T, B, 3], rotations_xyzw: [T, B, 4]. Positions are only written for
    bones in ``keep_position`` (others get zero offsets, as MMD ignores them anyway
    for non-movable bones).
    """
    names = list(names)
    keep = set(keep_position)
    T = rotations_xyzw.shape[0]
    frames: list[VmdBoneFrame] = []
    for b, name in enumerate(names):
        use_pos = name in keep
        for t in range(T):
            pos = tuple(float(x) for x in positions[t, b]) if use_pos else (0.0, 0.0, 0.0)
            rot = tuple(float(x) for x in rotations_xyzw[t, b])
            frames.append(VmdBoneFrame(name, t + frame_offset, pos, rot))
    return frames
