"""Disagreement metrics between overlapping action chunks.

The question this module answers: at frame ``t``, the policy has emitted several chunks that
each predict frame ``t``. How much do they contradict each other?

The answer is per-*group*, never per-column, and never averaged across groups. Position
disagreement is a distance in metres; orientation disagreement is an angle in radians; a
gripper disagreement is a unitless fraction. Adding them produces a number with no units and
no meaning. Every function here returns one series per group and leaves them separate.

Aggregation for the overview strip is a deliberate exception: to steer a brush you want a
single "look here" curve. :func:`overview_score` builds one, but by normalizing each group by
a robust scale first, so the number is explicitly a z-score-like heuristic rather than a
physical quantity. It is never reported with a unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import geodesic_angle, to_quat
from .spaces import ActionSpace, ChannelGroup


@dataclass
class GroupDisagreement:
    """Per-frame disagreement for one channel group."""

    key: str
    label: str
    unit: str
    metric: str
    spread: np.ndarray
    """``(T,)`` disagreement across the chunks in flight at each frame. NaN where fewer
    than two chunks cover the frame, since one prediction cannot disagree with itself."""
    n_in_flight: np.ndarray
    """``(T,)`` int count of chunks covering each frame."""

    @property
    def peak(self) -> float:
        return float(np.nanmax(self.spread)) if np.any(~np.isnan(self.spread)) else 0.0

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.spread)) if np.any(~np.isnan(self.spread)) else 0.0


def _group_values(
    space: ActionSpace, group: ChannelGroup, actions: np.ndarray
) -> np.ndarray:
    """Slice a group's columns out of ``(..., D)`` actions."""
    return np.asarray(actions, dtype=float)[..., list(group.channels)]


def pairwise_disagreement(
    space: ActionSpace,
    group: ChannelGroup,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Disagreement between two ``(..., D)`` action sets, using the group's own metric.

    * ``euclidean`` - L2 norm over the group's columns. One number for ``(x, y, z)``,
      in metres, not three separate component errors.
    * ``geodesic`` - angle of the relative rotation, in radians. Converts whatever
      representation the space uses (Euler by default) into quaternions first, which is what
      makes it immune to both branch cuts and gimbal lock.
    * ``abs`` - elementwise absolute difference, then max across the group's columns.
      Correct for joints, where each column is independently meaningful.
    """
    ga, gb = _group_values(space, group, a), _group_values(space, group, b)

    if group.metric == "euclidean":
        return np.linalg.norm(ga - gb, axis=-1)

    if group.metric == "geodesic":
        rot = group.rotation or space.rotation or "euler"
        qa = to_quat(ga, rot, quat_order=space.quat_order, euler_seq=space.euler_seq)
        qb = to_quat(gb, rot, quat_order=space.quat_order, euler_seq=space.euler_seq)
        return geodesic_angle(qa, qb, order="wxyz")

    if group.metric == "abs":
        return np.max(np.abs(ga - gb), axis=-1)

    raise ValueError(f"unknown metric {group.metric!r}")


def in_flight_disagreement(
    space: ActionSpace,
    group: ChannelGroup,
    chunks: list[np.ndarray],
    starts: np.ndarray,
    n_frames: int,
) -> GroupDisagreement:
    """Spread across every chunk predicting each frame, for one group.

    For scalar-valued metrics the spread at a frame is the maximum pairwise distance among
    the chunks covering it. With the usual ``H``/``S`` ratio only two or three chunks are ever
    in flight, so the exact pairwise maximum is cheap and avoids the bias of a min/max
    envelope taken per component.
    """
    per_frame: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
    for values, s0 in zip(chunks, starts, strict=True):
        vals = np.asarray(values, dtype=float)
        for j in range(vals.shape[0]):
            t = int(s0) + j
            if 0 <= t < n_frames:
                per_frame[t].append(vals[j])

    spread = np.full(n_frames, np.nan)
    counts = np.zeros(n_frames, dtype=int)

    for t, preds in enumerate(per_frame):
        counts[t] = len(preds)
        if len(preds) < 2:
            continue
        stack = np.stack(preds)  # (n, D)
        best = 0.0
        for i in range(len(stack)):
            d = pairwise_disagreement(space, group, stack[i][None, :], stack[i + 1 :])
            if d.size:
                best = max(best, float(np.max(d)))
        spread[t] = best

    return GroupDisagreement(
        key=group.key,
        label=group.label,
        unit=group.unit,
        metric=group.metric,
        spread=spread,
        n_in_flight=counts,
    )


def all_disagreements(
    space: ActionSpace,
    chunks: list[np.ndarray],
    starts: np.ndarray,
    n_frames: int,
) -> dict[str, GroupDisagreement]:
    """:func:`in_flight_disagreement` for every group in the space."""
    return {
        g.key: in_flight_disagreement(space, g, chunks, starts, n_frames)
        for g in space.groups
    }


def _robust_scale(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be comparable to a standard deviation.

    Chunk disagreement is spiky by construction (it peaks at every seam), so a plain
    standard deviation is dragged around by the very spikes we want to rank.
    """
    ok = x[~np.isnan(x)]
    if ok.size == 0:
        return 1.0
    med = np.median(ok)
    mad = np.median(np.abs(ok - med))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.max(ok)) if np.max(ok) > 1e-12 else 1.0
    return float(scale)


def overview_score(disagreements: dict[str, GroupDisagreement]) -> np.ndarray:
    """One unitless ``(T,)`` curve for steering the brush.

    Each group is divided by its own robust scale before combining, so a position error in
    metres and an orientation error in radians contribute comparably instead of whichever
    happens to have larger raw numbers. The result ranks frames; it does not measure
    anything, and is deliberately never labelled with a unit in the UI.
    """
    if not disagreements:
        return np.zeros(0)
    series = []
    for d in disagreements.values():
        s = np.nan_to_num(d.spread, nan=0.0)
        series.append(s / _robust_scale(d.spread))
    return np.max(np.stack(series), axis=0)


def seam_disagreement(
    space: ActionSpace,
    chunks: list[np.ndarray],
    starts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Disagreement at each chunk seam: the tail of chunk k vs the head of chunk k+1.

    This is the number that matters for real-time chunking. A large seam value is exactly
    the discontinuity the robot executes as a jerk, and it is invisible in a plot that only
    shows each chunk in isolation.

    Returns one ``(K-1,)`` array per group, NaN where two consecutive chunks do not overlap.
    """
    out: dict[str, np.ndarray] = {}
    for g in space.groups:
        vals = np.full(max(0, len(chunks) - 1), np.nan)
        for k in range(len(chunks) - 1):
            a, b = np.asarray(chunks[k]), np.asarray(chunks[k + 1])
            s_a, s_b = int(starts[k]), int(starts[k + 1])
            lo, hi = s_b, min(s_a + a.shape[0], s_b + b.shape[0])
            if hi <= lo:
                continue
            seg_a = a[lo - s_a : hi - s_a]
            seg_b = b[lo - s_b : hi - s_b]
            d = pairwise_disagreement(space, g, seg_a, seg_b)
            vals[k] = float(np.max(d)) if d.size else np.nan
        out[g.key] = vals
    return out
