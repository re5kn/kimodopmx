# SPDX-License-Identifier: Apache-2.0
"""Small batched rotation helpers (numpy only)."""

from __future__ import annotations

import numpy as np

# Reflection between Kimodo world (right-handed, Y-up, character faces +Z)
# and MMD world (left-handed, Y-up, character faces -Z): flip Z.
FLIP_Z = np.diag([1.0, 1.0, -1.0])


def flip_rotmats(R: np.ndarray) -> np.ndarray:
    """Conjugate rotation matrices by the Z reflection: S R S."""
    return FLIP_Z @ R @ FLIP_Z


def flip_points(p: np.ndarray) -> np.ndarray:
    out = np.array(p, dtype=np.float64, copy=True)
    out[..., 2] *= -1.0
    return out


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] -> unit quaternions [..., 4] (x, y, z, w), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    shape = R.shape[:-2]
    m = R.reshape(-1, 3, 3)
    q = np.empty((m.shape[0], 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # Choose the numerically best branch per matrix.
    cond0 = tr > 0
    cond1 = (~cond0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    cond2 = (~cond0) & (~cond1) & (m[:, 1, 1] > m[:, 2, 2])
    cond3 = ~(cond0 | cond1 | cond2)

    s = np.sqrt(np.maximum(tr[cond0] + 1.0, 1e-12)) * 2
    q[cond0, 3] = 0.25 * s
    q[cond0, 0] = (m[cond0, 2, 1] - m[cond0, 1, 2]) / s
    q[cond0, 1] = (m[cond0, 0, 2] - m[cond0, 2, 0]) / s
    q[cond0, 2] = (m[cond0, 1, 0] - m[cond0, 0, 1]) / s

    s = np.sqrt(np.maximum(1.0 + m[cond1, 0, 0] - m[cond1, 1, 1] - m[cond1, 2, 2], 1e-12)) * 2
    q[cond1, 3] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / s
    q[cond1, 0] = 0.25 * s
    q[cond1, 1] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / s
    q[cond1, 2] = (m[cond1, 0, 2] + m[cond1, 2, 0]) / s

    s = np.sqrt(np.maximum(1.0 + m[cond2, 1, 1] - m[cond2, 0, 0] - m[cond2, 2, 2], 1e-12)) * 2
    q[cond2, 3] = (m[cond2, 0, 2] - m[cond2, 2, 0]) / s
    q[cond2, 0] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / s
    q[cond2, 1] = 0.25 * s
    q[cond2, 2] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / s

    s = np.sqrt(np.maximum(1.0 + m[cond3, 2, 2] - m[cond3, 0, 0] - m[cond3, 1, 1], 1e-12)) * 2
    q[cond3, 3] = (m[cond3, 1, 0] - m[cond3, 0, 1]) / s
    q[cond3, 0] = (m[cond3, 0, 2] + m[cond3, 2, 0]) / s
    q[cond3, 1] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / s
    q[cond3, 2] = 0.25 * s

    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[q[:, 3] < 0] *= -1.0
    return q.reshape(*shape, 4)


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_power(R: np.ndarray, ratio: float) -> np.ndarray:
    """R^ratio (scale the rotation angle; negative ratio gives the inverse direction)."""
    if ratio == 1.0:
        return R
    q = rotmat_to_quat_xyzw(R)
    w = np.clip(q[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = np.where(sin_half[..., None] > 1e-9, q[..., :3] / np.maximum(sin_half[..., None], 1e-9), 0.0)
    half = 0.5 * angle * ratio
    out = np.concatenate([axis * np.sin(half)[..., None], np.cos(half)[..., None]], axis=-1)
    return quat_xyzw_to_rotmat(out)


def rotation_between(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Smallest rotation R with R @ a_hat == b_hat."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - eps:
        return np.eye(3)
    if c < -1.0 + eps:
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, seed)
        u /= np.linalg.norm(u)
        return -np.eye(3) + 2.0 * np.outer(u, u)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / float(np.dot(v, v)))


def swing_twist(R: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose R = swing @ twist with twist about ``axis`` (batched over leading dims)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    q = rotmat_to_quat_xyzw(R)
    proj = (q[..., :3] @ axis)[..., None] * axis
    tw = np.concatenate([proj, q[..., 3:4]], axis=-1)
    n = np.linalg.norm(tw, axis=-1, keepdims=True)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    tw = np.where(n > 1e-9, tw / np.maximum(n, 1e-9), identity)
    twist = quat_xyzw_to_rotmat(tw)
    swing = R @ np.swapaxes(twist, -1, -2)
    return swing, twist


def slerp_quats(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched slerp between xyzw quaternions; t broadcast against leading dims."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    t = np.asarray(t, dtype=np.float64)[..., None]
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin_t))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin_t))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)
