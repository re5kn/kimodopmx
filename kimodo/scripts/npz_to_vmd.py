# SPDX-License-Identifier: Apache-2.0
"""Convert a Kimodo motion NPZ to a MikuMikuDance VMD for a given PMX model.

Example:
    kimodo_vmd output.npz --pmx /workspace/models/model.pmx -o output.vmd
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _skeleton_for_joint_count(n: int):
    from kimodo.skeleton import SMPLXSkeleton22, SOMASkeleton30, SOMASkeleton77

    if n == 77:
        return SOMASkeleton77(load=True)
    if n == 30:
        return SOMASkeleton30(load=True)
    if n == 22:
        return SMPLXSkeleton22(load=True)
    raise ValueError(f"Unsupported joint count {n} (expected SOMA 30/77 or SMPL-X 22).")


def motion_to_vmd_bytes(joints_pos, global_rots, pmx, fps: float = 30.0, leg_ik: bool = True):
    """Retarget one Kimodo motion ([T, J, 3] positions, [T, J, 3, 3] global rotations) to ``pmx``.

    Returns ``(vmd_bytes, retargeter)``. The skeleton is picked from the joint count.
    """
    from kimodo.mmd import MMDRetargeter

    joints_pos = np.asarray(joints_pos)
    global_rots = np.asarray(global_rots)
    skeleton = _skeleton_for_joint_count(joints_pos.shape[1])
    rt = MMDRetargeter(
        pmx,
        skeleton.bone_order_names,
        skeleton.joint_parents.cpu().numpy(),
        skeleton.neutral_joints.cpu().numpy(),
    )
    return rt.export_vmd_bytes(global_rots, joints_pos, fps=fps, leg_ik=leg_ik), rt


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Retarget a Kimodo NPZ motion to a PMX model and write VMD.")
    p.add_argument("input", help="Kimodo motion .npz (needs posed_joints and global_rot_mats)")
    p.add_argument("--pmx", required=True, help="Target PMX model")
    p.add_argument("-o", "--output", default=None, help="Output .vmd (default: input name with .vmd)")
    p.add_argument("--fps", type=float, default=30.0, help="Frame rate of the input motion (default 30)")
    p.add_argument(
        "--no-leg-ik",
        action="store_true",
        help="Write pure FK legs and switch leg IK off in the VMD instead of keying 足ＩＫ",
    )
    args = p.parse_args(argv)

    from kimodo.mmd import load_pmx

    data = np.load(args.input)
    if "posed_joints" not in data or "global_rot_mats" not in data:
        raise SystemExit("Input NPZ must contain 'posed_joints' and 'global_rot_mats' (Kimodo NPZ).")
    joints_pos = data["posed_joints"]
    global_rots = data["global_rot_mats"]
    if joints_pos.ndim == 4:  # batched output: take the first sample
        joints_pos, global_rots = joints_pos[0], global_rots[0]

    pmx = load_pmx(args.pmx)
    payload, rt = motion_to_vmd_bytes(joints_pos, global_rots, pmx, fps=args.fps, leg_ik=not args.no_leg_ik)
    out = args.output or os.path.splitext(args.input)[0] + ".vmd"
    with open(out, "wb") as f:
        f.write(payload)
    print(f"Wrote {out} ({joints_pos.shape[0]} frames, model '{pmx.name}', scale {rt.scale:.2f} units/m)")


if __name__ == "__main__":
    main()
