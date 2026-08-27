# Copyright 2026 Ben Yang
# SPDX-License-Identifier: Apache-2.0

"""Normalizing whatever you have into something plottable.

Action chunks arrive in inconsistent shapes: a dense ``(K, H, D)`` tensor from a batched
rollout, a ragged list from a live loop that truncated at the episode end, torch tensors on
a GPU, absolute poses or deltas, with or without a ground-truth trace to compare against.
:meth:`ChunkEpisode.from_chunks` accepts all of it and validates hard, because every
downstream number is wrong in a plausible-looking way if the layout was misread.

It also performs the display-time repairs that depend on the action space:

* **Angle branch alignment.** Euler channels are unwrapped along time, and each chunk is then
  shifted by whole turns to sit on the same branch as the ground truth. Unwrapping chunks
  independently is not enough: two chunks can each be internally smooth yet land 2*pi apart,
  which draws as a full-scale disagreement that does not exist.
* **Quaternion canonicalization**, when ``rotation="quat"``, for the same reason in the
  double-cover form.

Repairs affect display only. Metrics run on the raw values and are invariant to both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import rotations as R
from .metrics import GroupDisagreement, all_disagreements, overview_score, seam_disagreement
from .spaces import ActionSpace


def to_numpy(x: Any) -> np.ndarray:
    """Accept numpy arrays, torch tensors, or nested sequences."""
    if x is None:
        return None  # type: ignore[return-value]
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    # torch / jax / anything with the usual escape hatches, without importing them.
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(x, attr):
            break
    else:
        return np.asarray(x, dtype=float)
    y = x
    if hasattr(y, "detach"):
        y = y.detach()
    if hasattr(y, "cpu"):
        y = y.cpu()
    if hasattr(y, "numpy"):
        y = y.numpy()
    return np.asarray(y, dtype=float)


@dataclass
class ChunkEpisode:
    """One episode's worth of overlapping action chunks, ready to plot."""

    space: ActionSpace
    fps: float
    n_frames: int
    chunks: list[np.ndarray]
    """Length-K list of ``(L_k, D)`` raw predicted actions. ``L_k`` may vary."""
    starts: np.ndarray
    """``(K,)`` int step index at which each chunk begins."""
    ground_truth: np.ndarray | None = None
    """``(T, D)`` executed/reference actions, or None."""

    display_chunks: list[np.ndarray] = field(default_factory=list, repr=False)
    display_ground_truth: np.ndarray | None = field(default=None, repr=False)
    disagreements: dict[str, GroupDisagreement] = field(default_factory=dict, repr=False)
    overview: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    seams: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    gimbal_risk: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------ construction --
    @classmethod
    def from_chunks(
        cls,
        chunks: Any,
        starts: Sequence[int] | np.ndarray | None = None,
        *,
        space: ActionSpace | None = None,
        fps: float = 30.0,
        ground_truth: Any = None,
        n_frames: int | None = None,
        stride: int | None = None,
        delta: bool = False,
        names: Sequence[str] | None = None,
    ) -> ChunkEpisode:
        """Build an episode from chunk predictions.

        Parameters
        ----------
        chunks
            Either a dense ``(K, H, D)`` array-like, or a length-K sequence of ``(L_k, D)``
            array-likes with possibly differing ``L_k``. numpy, torch and nested lists all work.
        starts
            ``(K,)`` step index where each chunk begins. If omitted, ``stride`` is used to
            generate ``0, stride, 2*stride, ...``; if both are omitted, chunks are assumed
            contiguous and non-overlapping.
        space
            The action space. If omitted it is inferred from ``names``, or failing that a
            joint space with generic names is assumed.
        ground_truth
            ``(T, D)`` executed actions. Optional, but without it the chunks have nothing to
            be compared against and angle branch alignment falls back to chunk 0.
        delta
            If True, chunks hold per-step increments rather than absolute actions. They are
            cumulatively integrated from the ground truth at each chunk's start so that
            chunks become directly comparable on a shared axis. Requires ``ground_truth``.
        """
        # ---- chunks -> list of (L_k, D) ------------------------------------------
        if isinstance(chunks, (list, tuple)) and not isinstance(chunks, np.ndarray):
            chunk_list = [to_numpy(c) for c in chunks]
        else:
            arr = to_numpy(chunks)
            if arr.ndim != 3:
                raise ValueError(
                    f"dense chunks must be (K, H, D); got shape {arr.shape}. "
                    "For variable-length chunks pass a list of (L_k, D) arrays."
                )
            chunk_list = [arr[k] for k in range(arr.shape[0])]

        if not chunk_list:
            raise ValueError("no chunks given")
        for k, c in enumerate(chunk_list):
            if c.ndim != 2:
                raise ValueError(f"chunk {k} must be 2-D (L, D); got shape {c.shape}")
        dims = {c.shape[1] for c in chunk_list}
        if len(dims) != 1:
            raise ValueError(f"all chunks must share an action dim; got {sorted(dims)}")
        dim = dims.pop()

        # ---- space ---------------------------------------------------------------
        if space is None:
            if names is not None:
                space = ActionSpace.infer(names)
            else:
                space = ActionSpace.joint([f"dim_{i}" for i in range(dim)], gripper=None)
        if space.dim != dim:
            raise ValueError(
                f"action space has dim {space.dim} ({', '.join(space.names)}) but chunks "
                f"have dim {dim}. Pass a matching ActionSpace, or check whether the "
                "gripper column is included."
            )

        # ---- starts --------------------------------------------------------------
        if starts is None:
            if stride is None:
                offs, acc = [], 0
                for c in chunk_list:
                    offs.append(acc)
                    acc += c.shape[0]
                starts_arr = np.asarray(offs, dtype=int)
            else:
                starts_arr = np.arange(len(chunk_list), dtype=int) * int(stride)
        else:
            starts_arr = np.asarray(starts, dtype=int).reshape(-1)
        if starts_arr.size != len(chunk_list):
            raise ValueError(
                f"got {starts_arr.size} starts for {len(chunk_list)} chunks"
            )
        if np.any(starts_arr < 0):
            raise ValueError("chunk starts must be non-negative")
        if np.any(np.diff(starts_arr) < 0):
            order = np.argsort(starts_arr, kind="stable")
            starts_arr = starts_arr[order]
            chunk_list = [chunk_list[i] for i in order]

        # ---- ground truth & frame count ------------------------------------------
        gt = to_numpy(ground_truth) if ground_truth is not None else None
        if gt is not None:
            if gt.ndim != 2 or gt.shape[1] != dim:
                raise ValueError(
                    f"ground_truth must be (T, {dim}); got shape {gt.shape}"
                )
        reach = int(max(s + c.shape[0] for s, c in zip(starts_arr, chunk_list, strict=True)))
        T = int(n_frames or (gt.shape[0] if gt is not None else reach))
        if gt is not None and gt.shape[0] < reach:
            # Chunks predicting past the end of the recorded trace is normal at the tail.
            T = max(T, gt.shape[0])

        # ---- delta integration ---------------------------------------------------
        if delta:
            if gt is None:
                raise ValueError("delta=True needs ground_truth to integrate from")
            integrated = []
            for c, s0 in zip(chunk_list, starts_arr, strict=True):
                base = gt[min(int(s0), gt.shape[0] - 1)]
                integrated.append(base[None, :] + np.cumsum(c, axis=0))
            chunk_list = integrated

        ep = cls(
            space=space,
            fps=float(fps),
            n_frames=T,
            chunks=chunk_list,
            starts=starts_arr,
            ground_truth=gt,
        )
        ep._prepare()
        return ep

    # -------------------------------------------------------------- derived data --
    def _prepare(self) -> None:
        self.display_ground_truth, self.display_chunks = self._repair_for_display()
        self.disagreements = all_disagreements(
            self.space, self.chunks, self.starts, self.n_frames
        )
        self.overview = overview_score(self.disagreements)
        self.seams = seam_disagreement(self.space, self.chunks, self.starts)
        if self.ground_truth is not None:
            self.gimbal_risk = self.space.gimbal_risk(self.ground_truth)

    def _repair_for_display(self) -> tuple[np.ndarray | None, list[np.ndarray]]:
        """Unwrap angles and canonicalize quaternions onto a common branch."""
        space = self.space
        gt = None if self.ground_truth is None else self.ground_truth.copy()
        chunks = [c.copy() for c in self.chunks]

        angular = [i for i, ch in enumerate(space.channels) if ch.angular]
        if angular:
            if gt is not None:
                gt[:, angular] = R.unwrap_angles(gt[:, angular], axis=0)
            # Reference for branch alignment: the unwrapped ground truth if we have it,
            # else the first chunk, so that chunks are at least mutually consistent.
            ref = gt
            if ref is None:
                base = chunks[0].copy()
                base[:, angular] = R.unwrap_angles(base[:, angular], axis=0)
                ref = np.full((self.n_frames, space.dim), np.nan)
                s0 = int(self.starts[0])
                ref[s0 : s0 + base.shape[0]] = base

            for k, c in enumerate(chunks):
                c[:, angular] = R.unwrap_angles(c[:, angular], axis=0)
                s0 = int(self.starts[k])
                lo, hi = s0, min(s0 + c.shape[0], ref.shape[0])
                if hi > lo:
                    tgt = ref[lo:hi][:, angular]
                    cur = c[: hi - lo][:, angular]
                    ok = ~np.isnan(tgt)
                    if np.any(ok):
                        # One whole-turn shift per channel, chosen on the overlap median.
                        diff = np.where(ok, tgt - cur, np.nan)
                        with np.errstate(invalid="ignore"):
                            turns = np.round(np.nanmedian(diff, axis=0) / (2 * np.pi))
                        turns = np.nan_to_num(turns)
                        c[:, angular] += 2 * np.pi * turns[None, :]
                chunks[k] = c

        if space.rotation == "quat":
            idx = list(space.group("orientation").channels)
            gt_q = None
            if gt is not None:
                gt[:, idx] = R.canonicalize_quat(gt[:, idx], order=space.quat_order, axis=0)
                gt_q = gt[:, idx]
            for k, c in enumerate(chunks):
                s0 = int(self.starts[k])
                ref_q = None
                if gt_q is not None:
                    lo, hi = s0, min(s0 + c.shape[0], gt_q.shape[0])
                    if hi > lo:
                        ref_q = np.zeros((c.shape[0], 4))
                        ref_q[: hi - lo] = gt_q[lo:hi]
                        ref_q[hi - lo :] = gt_q[hi - 1]
                c[:, idx] = R.canonicalize_quat(
                    c[:, idx], order=space.quat_order, reference=ref_q, axis=0
                )
                chunks[k] = c

        return gt, chunks

    # ---------------------------------------------------------------- accessors --
    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    def chunk_times(self, k: int) -> np.ndarray:
        s0 = int(self.starts[k])
        return (s0 + np.arange(self.chunks[k].shape[0])) / self.fps

    def n_in_flight(self) -> np.ndarray:
        """``(T,)`` how many chunks predict each frame."""
        counts = np.zeros(self.n_frames, dtype=int)
        for c, s0 in zip(self.chunks, self.starts, strict=True):
            lo = max(0, int(s0))
            hi = min(self.n_frames, int(s0) + c.shape[0])
            if hi > lo:
                counts[lo:hi] += 1
        return counts

    def summary(self) -> str:
        f = self.n_in_flight()
        lines = [
            f"{self.n_chunks} chunks over {self.n_frames} frames "
            f"({self.duration:.1f}s @ {self.fps:g} fps)",
            f"chunks in flight: min {f.min()}, max {f.max()}",
            self.space.describe(),
            "disagreement (across chunks in flight):",
        ]
        for d in self.disagreements.values():
            unit = f" {d.unit}" if d.unit else ""
            lines.append(
                f"  {d.label:<12} mean {d.mean:.4f}{unit}  peak {d.peak:.4f}{unit}"
                f"   [{d.metric}]"
            )
        if self.gimbal_risk is not None:
            worst = float(np.nanmax(self.gimbal_risk))
            note = "  <- per-axis rotation rows unreliable there" if worst > 0.9 else ""
            lines.append(f"gimbal-lock proximity: peak {worst:.3f}{note}")
        return "\n".join(lines)
