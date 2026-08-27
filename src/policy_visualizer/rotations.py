# Copyright 2026 Ben Yang
# SPDX-License-Identifier: Apache-2.0

"""Rotation representations and the geodesic metric between them.

Everything here is vectorized over leading axes: a function documented as taking
``(..., 4)`` accepts ``(4,)``, ``(H, 4)`` or ``(K, H, 4)`` alike and returns a matching
leading shape.

Two facts drive this module, and both bite hard when visualizing action chunks:

1. A quaternion and its negation are the SAME rotation. If chunk k stores ``q`` and chunk
   k+1 stores ``-q``, a naive per-component difference reports maximal disagreement where
   there is none. Canonicalization (:func:`canonicalize_quat`) has to happen before either
   plotting or diffing.
2. Orientation error is not the L2 norm of a component difference. It is the geodesic angle
   on SO(3) (:func:`geodesic_angle`), which is what "the wrist is 4 degrees off" actually
   means and is the only orientation number worth putting on a plot axis.

Quaternion component order is explicit everywhere via ``order``: ``"wxyz"`` (the default,
used by MuJoCo, Drake, Isaac) or ``"xyzw"`` (used by SciPy, ROS, PyBullet). Getting this
wrong is silent and produces plausible-looking garbage, so it is never inferred.
"""

from __future__ import annotations

import numpy as np

QuatOrder = str  # "wxyz" | "xyzw"

_VALID_ORDERS = ("wxyz", "xyzw")


def _as_wxyz(q: np.ndarray, order: QuatOrder) -> np.ndarray:
    """Reorder ``(..., 4)`` quaternions into internal w-first layout."""
    if order not in _VALID_ORDERS:
        raise ValueError(f"quat order must be one of {_VALID_ORDERS}, got {order!r}")
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion array must have trailing dim 4, got {q.shape}")
    if order == "xyzw":
        q = q[..., [3, 0, 1, 2]]
    return q


def _from_wxyz(q: np.ndarray, order: QuatOrder) -> np.ndarray:
    if order == "xyzw":
        return q[..., [1, 2, 3, 0]]
    return q


def normalize_quat(q: np.ndarray, order: QuatOrder = "wxyz") -> np.ndarray:
    """Unit-normalize ``(..., 4)`` quaternions, leaving component order unchanged.

    Degenerate (zero-norm) quaternions become identity rather than NaN, so a malformed
    frame degrades to "no rotation" instead of poisoning every downstream metric.
    """
    w = _as_wxyz(q, order)
    n = np.linalg.norm(w, axis=-1, keepdims=True)
    bad = n[..., 0] <= 1e-12
    out = np.where(n > 1e-12, w / np.where(n > 1e-12, n, 1.0), 0.0)
    out[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    return _from_wxyz(out, order)


def canonicalize_quat(
    q: np.ndarray,
    order: QuatOrder = "wxyz",
    reference: np.ndarray | None = None,
    axis: int = -2,
) -> np.ndarray:
    """Remove the double-cover sign ambiguity from ``(..., 4)`` quaternions.

    Two passes, in this order:

    1. **Sequential alignment** along ``axis`` (the time axis by default): each sample is
       negated if it lies in the opposite hemisphere from its predecessor. This kills the
       mid-chunk sign flips that otherwise draw as a full-scale square wave.
    2. **Reference alignment**: if ``reference`` is given (broadcastable to ``q``), each
       sample is negated to share a hemisphere with it. Pass the ground-truth orientation
       here so that separate chunks are mutually comparable, not merely self-consistent.

    Without ``reference``, chunks are each internally smooth but may still sit in opposite
    hemispheres from one another, which is exactly the artifact this exists to prevent.
    """
    w = _as_wxyz(q, order).copy()

    if w.ndim >= 2 and w.shape[axis] > 1:
        moved = np.moveaxis(w, axis, 0)
        for i in range(1, moved.shape[0]):
            flip = np.sum(moved[i] * moved[i - 1], axis=-1) < 0.0
            moved[i] = np.where(flip[..., None], -moved[i], moved[i])
        w = np.moveaxis(moved, 0, axis)

    if reference is not None:
        ref = _as_wxyz(reference, order)
        flip = np.sum(w * ref, axis=-1) < 0.0
        w = np.where(flip[..., None], -w, w)
    else:
        # Fall back to the w >= 0 hemisphere so repeated calls are idempotent.
        flip = w[..., 0] < 0.0
        w = np.where(flip[..., None], -w, w)

    return _from_wxyz(w, order)


def quat_multiply(a: np.ndarray, b: np.ndarray, order: QuatOrder = "wxyz") -> np.ndarray:
    """Hamilton product of ``(..., 4)`` quaternions."""
    aw, ax, ay, az = np.moveaxis(_as_wxyz(a, order), -1, 0)
    bw, bx, by, bz = np.moveaxis(_as_wxyz(b, order), -1, 0)
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )
    return _from_wxyz(out, order)


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Axis-angle ``(..., 3)`` (direction = axis, magnitude = angle in rad) to w-first quat.

    Uses the small-angle series near zero, where ``sin(t/2)/t`` is numerically unstable.
    """
    v = np.asarray(v, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"rotvec array must have trailing dim 3, got {v.shape}")
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    small = theta < 1e-8
    # sin(t/2)/t -> 1/2 - t^2/48 as t -> 0
    half = 0.5 * theta
    scale = np.where(small, 0.5 - theta**2 / 48.0, np.sin(half) / np.where(small, 1.0, theta))
    w = np.where(small, 1.0 - theta**2 / 8.0, np.cos(half))
    return np.concatenate([w, v * scale], axis=-1)


def euler_to_quat(e: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Euler angles ``(..., 3)`` in radians to w-first quaternions.

    ``seq`` follows SciPy's convention: lowercase letters are extrinsic (fixed-frame)
    rotations, uppercase are intrinsic (body-frame). ``"xyz"`` is extrinsic roll-pitch-yaw.
    """
    e = np.asarray(e, dtype=float)
    if e.shape[-1] != 3:
        raise ValueError(f"euler array must have trailing dim 3, got {e.shape}")
    if len(seq) != 3 or any(c.lower() not in "xyz" for c in seq):
        raise ValueError(f"euler seq must be 3 letters from xyz/XYZ, got {seq!r}")

    intrinsic = seq.isupper()
    if not intrinsic and not seq.islower():
        raise ValueError(f"euler seq must be all-lower or all-upper, got {seq!r}")

    axis_of = {"x": 0, "y": 1, "z": 2}
    elementary = []
    for i, letter in enumerate(seq.lower()):
        half = 0.5 * e[..., i]
        q = np.zeros(e.shape[:-1] + (4,))
        q[..., 0] = np.cos(half)
        q[..., 1 + axis_of[letter]] = np.sin(half)
        elementary.append(q)

    # Extrinsic xyz == intrinsic ZYX reversed: compose left-to-right for intrinsic,
    # right-to-left for extrinsic.
    ordered = elementary if intrinsic else elementary[::-1]
    out = ordered[0]
    for q in ordered[1:]:
        out = quat_multiply(out, q, order="wxyz")
    return out


def rot6d_to_matrix(a: np.ndarray) -> np.ndarray:
    """6D continuous rotation representation ``(..., 6)`` to ``(..., 3, 3)``.

    Zhou et al. 2019: Gram-Schmidt the first two 3-vectors, cross for the third. This is
    the representation most diffusion/flow policies regress, precisely because it has no
    discontinuities for the network to trip over.
    """
    a = np.asarray(a, dtype=float)
    if a.shape[-1] != 6:
        raise ValueError(f"rot6d array must have trailing dim 6, got {a.shape}")
    a1, a2 = a[..., :3], a[..., 3:]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12, None)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.clip(np.linalg.norm(a2p, axis=-1, keepdims=True), 1e-12, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrices to w-first quaternions.

    Uses Shepperd's branch selection (largest diagonal term) rather than the naive
    trace formula, which loses precision when the trace approaches -1.
    """
    m = np.asarray(m, dtype=float)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"matrix array must have trailing shape (3,3), got {m.shape}")

    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22

    cands = np.stack([trace, m00, m11, m22], axis=-1)
    branch = np.argmax(cands, axis=-1)

    def _b0():
        s = np.sqrt(np.clip(trace + 1.0, 1e-12, None)) * 2.0
        return np.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], -1)

    def _b1():
        s = np.sqrt(np.clip(1.0 + m00 - m11 - m22, 1e-12, None)) * 2.0
        return np.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], -1)

    def _b2():
        s = np.sqrt(np.clip(1.0 + m11 - m00 - m22, 1e-12, None)) * 2.0
        return np.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], -1)

    def _b3():
        s = np.sqrt(np.clip(1.0 + m22 - m00 - m11, 1e-12, None)) * 2.0
        return np.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], -1)

    out = _b0()
    for idx, fn in ((1, _b1), (2, _b2), (3, _b3)):
        out = np.where((branch == idx)[..., None], fn(), out)
    return normalize_quat(out, order="wxyz")


def to_quat(
    x: np.ndarray,
    rotation: str,
    *,
    quat_order: QuatOrder = "wxyz",
    euler_seq: str = "xyz",
) -> np.ndarray:
    """Convert any supported orientation encoding to w-first unit quaternions.

    ``rotation`` is one of ``"quat"``, ``"euler"``, ``"rotvec"``, ``"rot6d"``.
    """
    if rotation == "quat":
        return normalize_quat(_as_wxyz(x, quat_order), order="wxyz")
    if rotation == "euler":
        return euler_to_quat(x, seq=euler_seq)
    if rotation == "rotvec":
        return rotvec_to_quat(x)
    if rotation == "rot6d":
        return matrix_to_quat(rot6d_to_matrix(x))
    raise ValueError(
        f"unknown rotation {rotation!r}; expected quat, euler, rotvec or rot6d"
    )


def geodesic_angle(qa: np.ndarray, qb: np.ndarray, order: QuatOrder = "wxyz") -> np.ndarray:
    """Angle in radians of the relative rotation between two ``(..., 4)`` quaternion sets.

    Returns values in ``[0, pi]``, and is double-cover safe: ``q`` and ``-q`` score exactly 0.

    Computed as ``2 * atan2(|vec(q_rel)|, |w(q_rel)|)`` on the relative quaternion rather
    than the more common ``2 * arccos(<qa, qb>)``. The arccos form is accurate only in the
    middle of its range: near identity its argument approaches 1 where arccos has infinite
    derivative, so it returns garbage on the order of ``sqrt(eps)`` (~1e-8 rad) for
    quaternions that are bitwise identical. Chunk-overlap disagreement is usually *small*,
    which is exactly the regime the arccos form gets wrong, so the atan2 form is the one
    worth having here.
    """
    a = normalize_quat(_as_wxyz(qa, order), order="wxyz")
    b = normalize_quat(_as_wxyz(qb, order), order="wxyz")
    a_conj = a * np.array([1.0, -1.0, -1.0, -1.0])
    rel = quat_multiply(a_conj, b, order="wxyz")
    vec = np.linalg.norm(rel[..., 1:], axis=-1)
    return 2.0 * np.arctan2(vec, np.abs(rel[..., 0]))


def unwrap_angles(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """``np.unwrap`` along ``axis``, tolerant of NaN gaps.

    Euler channels plotted raw show a full 2*pi cliff every time the angle crosses the
    branch cut, which reads as a catastrophic chunk disagreement and is purely cosmetic.
    """
    x = np.asarray(x, dtype=float)
    if x.shape[axis] < 2:
        return x
    if not np.any(np.isnan(x)):
        return np.unwrap(x, axis=axis)
    out = x.copy()
    moved = np.moveaxis(out, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    for row in flat:
        ok = ~np.isnan(row)
        if ok.sum() >= 2:
            row[ok] = np.unwrap(row[ok])
    return np.moveaxis(flat.reshape(moved.shape), -1, axis)
