# Copyright 2026 Ben Yang
# SPDX-License-Identifier: Apache-2.0

"""Synthetic episodes for development and for exercising both space kinds.

Chunk error is shaped the way real chunking policies fail: a step offset at the chunk head
(the policy re-planned and landed somewhere slightly different) plus drift that grows with
horizon index (predictions decay further out). The tail of chunk k therefore disagrees with
the head of chunk k+1, which is the discontinuity a chunk debugger exists to find.

The Cartesian generator can be pushed through gimbal lock on purpose
(``through_gimbal=True``), which is the cheapest way to see why per-axis Euler rows cannot be
trusted near the singularity while the geodesic metric stays well behaved.
"""

from __future__ import annotations

import numpy as np

from .episode import ChunkEpisode
from .spaces import ActionSpace

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _smooth(t: np.ndarray, d: int, seed_phase: float = 0.0) -> np.ndarray:
    phase = 0.7 * d + seed_phase
    return (
        1.00 * np.sin(2 * np.pi * (0.13 + 0.045 * d) * t + phase)
        + 0.45 * np.sin(2 * np.pi * (0.31 + 0.070 * d) * t + 1.7 * phase)
        + 0.18 * np.sin(2 * np.pi * (0.77 + 0.110 * d) * t + 2.9 * phase)
    ) / 1.63


def _chunkify(
    gt: np.ndarray,
    horizon: int,
    stride: int,
    rng: np.random.Generator,
    scale: np.ndarray,
    head_sigma: float = 0.030,
    drift_sigma: float = 0.115,
) -> tuple[list[np.ndarray], np.ndarray]:
    T, D = gt.shape
    starts = np.arange(0, max(1, T - horizon + 1), stride, dtype=int)
    j = np.arange(horizon)
    env = (j / max(1, horizon - 1)) ** 1.7  # ~0 at the head, grows to the tail

    chunks = []
    for s0 in starts:
        idx = s0 + j
        idx = idx[idx < T]
        e = env[: idx.size]
        base = gt[idx]
        head = rng.normal(0.0, head_sigma, size=D) * scale
        drift = e[:, None] * (rng.normal(0.0, drift_sigma, size=D) * scale)[None, :]
        jitter = rng.normal(0.0, 0.004, size=(idx.size, D)) * scale
        chunks.append(base + head[None, :] + drift + jitter)
    return chunks, starts


def joint_episode(
    n_frames: int = 600,
    fps: float = 30.0,
    horizon: int = 50,
    stride: int = 25,
    seed: int = 0,
) -> ChunkEpisode:
    """A 6-DoF arm in joint space: five joints in radians plus a normalized gripper."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    D = len(JOINT_NAMES)

    scale = np.array([1.20, 0.90, 1.10, 0.70, 1.50, 0.45])
    offset = np.array([0.00, -0.35, 0.40, 0.00, 0.00, 0.50])
    gt = np.empty((n_frames, D))
    for d in range(D):
        gt[:, d] = offset[d] + scale[d] * _smooth(t, d)
    gt[:, 5] = 0.5 + 0.42 * np.tanh(2.4 * np.sin(2 * np.pi * 0.11 * t + 0.4))

    chunks, starts = _chunkify(gt, horizon, stride, rng, scale)
    for c in chunks:
        c[:, 5] = np.clip(c[:, 5], 0.0, 1.0)
    gt[:, 5] = np.clip(gt[:, 5], 0.0, 1.0)

    space = ActionSpace.joint(JOINT_NAMES, unit="rad")
    return ChunkEpisode.from_chunks(
        chunks, starts, space=space, fps=fps, ground_truth=gt, n_frames=n_frames
    )


def cartesian_episode(
    n_frames: int = 600,
    fps: float = 30.0,
    horizon: int = 50,
    stride: int = 25,
    seed: int = 0,
    euler_seq: str = "xyz",
    through_gimbal: bool = False,
) -> ChunkEpisode:
    """End-effector pose: ``[x, y, z, roll, pitch, yaw, gripper]`` with Euler rotation.

    Position is metres on a ~0.3 m workspace; angles are radians; the gripper is 0-1. Set
    ``through_gimbal`` to sweep pitch across +/-pi/2, where roll and yaw become degenerate.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps

    gt = np.empty((n_frames, 7))
    gt[:, 0] = 0.45 + 0.12 * _smooth(t, 0)
    gt[:, 1] = 0.00 + 0.18 * _smooth(t, 1)
    gt[:, 2] = 0.28 + 0.09 * _smooth(t, 2)
    gt[:, 3] = 0.9 * _smooth(t, 3)  # roll
    if through_gimbal:
        # Sweep pitch through +/-pi/2 so the singularity is actually visited.
        gt[:, 4] = (np.pi / 2) * 1.15 * np.sin(2 * np.pi * 0.09 * t)
    else:
        gt[:, 4] = 0.6 * _smooth(t, 4)
    gt[:, 5] = 1.4 * _smooth(t, 5)  # yaw
    gt[:, 6] = 0.5 + 0.42 * np.tanh(2.4 * np.sin(2 * np.pi * 0.11 * t + 0.4))

    # Per-channel error scale in each channel's own unit: millimetres of position error
    # are not comparable to radians of angle error.
    scale = np.array([0.02, 0.02, 0.02, 0.25, 0.25, 0.25, 0.45])
    chunks, starts = _chunkify(gt, horizon, stride, rng, scale)

    if through_gimbal:
        # Generic noise does NOT expose gimbal degeneracy: independent roll/pitch/yaw
        # error is a real rotation change and both metrics report it alike. The
        # pathology only appears for error along the degenerate direction, which at
        # pitch = +/-pi/2 (xyz Tait-Bryan) is roll and yaw moving together: that
        # combination is a no-op rotation, so per-component Euler differencing reports a
        # large disagreement where nothing has actually moved.
        #
        # Inject exactly that, scaled by how close each frame is to the singularity, so
        # the demo shows the failure mode it claims to.
        # risk**8 rather than risk: the cancellation is only near-exact within a few
        # degrees of the singularity, so a broad envelope would inject large *genuine*
        # rotation error everywhere and the demo would just look broken instead of
        # showing a coordinate artifact confined to the shaded band.
        risk = np.abs(np.sin(gt[:, 4])) ** 8
        for c, s0 in zip(chunks, starts, strict=True):
            idx = np.arange(c.shape[0]) + int(s0)
            degen = rng.normal(0.0, 1.4) * risk[np.clip(idx, 0, n_frames - 1)]
            c[:, 3] += degen  # roll
            c[:, 5] += degen  # yaw, same sign -> cancels at lock

    for c in chunks:
        c[:, 6] = np.clip(c[:, 6], 0.0, 1.0)
    gt[:, 6] = np.clip(gt[:, 6], 0.0, 1.0)

    space = ActionSpace.cartesian(rotation="euler", euler_seq=euler_seq)
    return ChunkEpisode.from_chunks(
        chunks, starts, space=space, fps=fps, ground_truth=gt, n_frames=n_frames
    )
