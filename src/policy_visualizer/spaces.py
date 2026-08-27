"""Action-space description: what each column of an action vector actually means.

An action vector is just ``(D,)`` floats. Whether column 3 is a wrist angle in radians or
the roll of an end-effector pose changes how it must be scaled, unwrapped, differenced and
aggregated. :class:`ActionSpace` carries that meaning so the rest of the package never has
to guess.

Two kinds are built in:

``joint``
    Homogeneous. Every channel is a joint coordinate in the same unit, plus an optional
    gripper. Channels are independent, so error is per-joint and aggregates by max or mean.

``cartesian``
    Heterogeneous, and this is the case that punishes naive handling. A pose action mixes
    position in metres, orientation in radians, and a unitless gripper. Three consequences:

    * They cannot share a y-scale. 0.4 m of travel and 0.4 rad of roll are unrelated
      quantities that happen to print the same.
    * Per-component differencing then averaging is meaningless, because it adds metres to
      radians. Position error is a Euclidean norm over ``(x, y, z)``; orientation error is a
      geodesic angle on SO(3). See :mod:`policy_visualizer.metrics`.
    * Orientation needs representation-specific repair before it can be drawn at all.

Orientation defaults to **Euler** angles (roll/pitch/yaw), which is what most teleop stacks
and end-effector controllers expose. Two Euler-specific hazards are handled explicitly:

* **Branch cuts.** An angle crossing +/-pi jumps by 2*pi. Drawn raw this is a full-scale
  vertical cliff that reads as catastrophic chunk disagreement and is pure artifact.
  :func:`policy_visualizer.rotations.unwrap_angles` removes it.
* **Gimbal lock.** Near the sequence's singular configuration the first and third angles
  become degenerate: they can swing across their whole range while the underlying rotation
  barely moves. Per-component Euler disagreement is therefore *worst exactly where it is
  least meaningful*. This is why the reported orientation metric converts to quaternions and
  measures the geodesic angle, and why :meth:`ActionSpace.gimbal_risk` exists to flag the
  frames where the per-channel rows should not be trusted.

``quat``, ``rotvec`` and ``rot6d`` remain available via ``rotation=``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Kind = Literal["joint", "cartesian"]
Rotation = Literal["euler", "quat", "rotvec", "rot6d"]
Metric = Literal["abs", "euclidean", "geodesic"]

ROTATION_DIM: dict[str, int] = {"euler": 3, "quat": 4, "rotvec": 3, "rot6d": 6}

DEFAULT_EULER_SEQ = "xyz"

#: Channels whose per-component rows are degenerate near gimbal lock.
_EULER_ROW_NAMES = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class Channel:
    """One scalar column of the action vector."""

    name: str
    group: str
    unit: str = ""
    angular: bool = False
    """True if the channel is an angle needing unwrap before display."""


@dataclass(frozen=True)
class ChannelGroup:
    """A set of columns that must be measured together rather than one at a time."""

    key: str
    label: str
    channels: tuple[int, ...]
    unit: str
    metric: Metric
    rotation: Rotation | None = None

    @property
    def dim(self) -> int:
        return len(self.channels)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


@dataclass(frozen=True)
class ActionSpace:
    """Semantic description of a ``(D,)`` action vector."""

    kind: Kind
    channels: tuple[Channel, ...]
    groups: tuple[ChannelGroup, ...]
    rotation: Rotation | None = None
    euler_seq: str = DEFAULT_EULER_SEQ
    quat_order: str = "wxyz"
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        seen: list[int] = []
        for g in self.groups:
            seen.extend(g.channels)
        if sorted(seen) != list(range(len(self.channels))):
            raise ValueError(
                "groups must partition the channels exactly once each; "
                f"got indices {sorted(seen)} for {len(self.channels)} channels"
            )

    # ------------------------------------------------------------------ basics --
    @property
    def dim(self) -> int:
        return len(self.channels)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    def group_of(self, index: int) -> ChannelGroup:
        for g in self.groups:
            if index in g.channels:
                return g
        raise IndexError(index)

    def group(self, key: str) -> ChannelGroup:
        for g in self.groups:
            if g.key == key:
                return g
        raise KeyError(f"no group {key!r}; have {[g.key for g in self.groups]}")

    def has_group(self, key: str) -> bool:
        return any(g.key == key for g in self.groups)

    # ------------------------------------------------------------ constructors --
    @classmethod
    def joint(
        cls,
        names: Sequence[str],
        *,
        unit: str = "rad",
        gripper: str | int | None = "auto",
    ) -> ActionSpace:
        """Homogeneous joint-space actions.

        ``gripper`` may be a channel name, an index, ``None`` to disable, or ``"auto"``
        (the default) to split off any channel whose name looks like a gripper. A gripper
        is separated because it is usually normalized 0-1 rather than an angle, so folding
        it into the joint error would mix units again.
        """
        names = [str(n) for n in names]
        if not names:
            raise ValueError("joint space needs at least one channel")

        grip_idx: int | None = None
        if gripper == "auto":
            for i, n in enumerate(names):
                if "gripper" in _norm(n) or _norm(n) in ("grip", "hand", "jaw"):
                    grip_idx = i
                    break
        elif isinstance(gripper, int):
            grip_idx = gripper
        elif isinstance(gripper, str):
            grip_idx = names.index(gripper)

        angular = unit in ("rad", "deg")
        channels, joint_idx = [], []
        for i, n in enumerate(names):
            if i == grip_idx:
                channels.append(Channel(n, "gripper", "norm", False))
            else:
                channels.append(Channel(n, "joint", unit, angular))
                joint_idx.append(i)

        groups = [
            ChannelGroup("joint", "joints", tuple(joint_idx), unit, "abs"),
        ]
        if grip_idx is not None:
            groups.append(
                ChannelGroup("gripper", "gripper", (grip_idx,), "norm", "abs")
            )
        return cls(kind="joint", channels=tuple(channels), groups=tuple(groups))

    @classmethod
    def cartesian(
        cls,
        *,
        rotation: Rotation = "euler",
        euler_seq: str = DEFAULT_EULER_SEQ,
        quat_order: str = "wxyz",
        position_unit: str = "m",
        angle_unit: str = "rad",
        gripper: bool = True,
        position_names: Sequence[str] | None = None,
        rotation_names: Sequence[str] | None = None,
        gripper_name: str = "gripper",
    ) -> ActionSpace:
        """End-effector pose actions: position, then orientation, then optional gripper.

        Layout is ``[x, y, z] + rotation + [gripper]``. With the default Euler rotation the
        vector is ``[x, y, z, roll, pitch, yaw, gripper]`` (D=7).
        """
        if rotation not in ROTATION_DIM:
            raise ValueError(
                f"unknown rotation {rotation!r}; expected one of {sorted(ROTATION_DIM)}"
            )
        rdim = ROTATION_DIM[rotation]

        pos_names = list(position_names or ("x", "y", "z"))
        if len(pos_names) != 3:
            raise ValueError(f"position_names must have 3 entries, got {pos_names}")

        if rotation_names is not None:
            rot_names = list(rotation_names)
            if len(rot_names) != rdim:
                raise ValueError(
                    f"rotation_names must have {rdim} entries for {rotation!r}, "
                    f"got {len(rot_names)}"
                )
        elif rotation == "euler":
            rot_names = list(_EULER_ROW_NAMES)
        elif rotation == "quat":
            rot_names = list(quat_order)  # w,x,y,z or x,y,z,w
            rot_names = [f"q{c}" for c in rot_names]
        elif rotation == "rotvec":
            rot_names = ["rx", "ry", "rz"]
        else:
            rot_names = [f"r6d_{i}" for i in range(6)]

        # Only Euler channels are true angles that unwrap sensibly per component.
        # Quaternion and 6D components are not angles; rotvec components are angle-scaled
        # axis parts and do not wrap independently either.
        rot_angular = rotation == "euler"
        rot_unit = angle_unit if rotation in ("euler", "rotvec") else ""

        channels: list[Channel] = []
        for n in pos_names:
            channels.append(Channel(n, "position", position_unit, False))
        for n in rot_names:
            channels.append(Channel(n, "orientation", rot_unit, rot_angular))
        if gripper:
            channels.append(Channel(gripper_name, "gripper", "norm", False))

        groups = [
            ChannelGroup("position", "position", (0, 1, 2), position_unit, "euclidean"),
            ChannelGroup(
                "orientation",
                "orientation",
                tuple(range(3, 3 + rdim)),
                angle_unit,
                "geodesic",
                rotation=rotation,
            ),
        ]
        if gripper:
            groups.append(
                ChannelGroup("gripper", "gripper", (3 + rdim,), "norm", "abs")
            )

        return cls(
            kind="cartesian",
            channels=tuple(channels),
            groups=tuple(groups),
            rotation=rotation,
            euler_seq=euler_seq,
            quat_order=quat_order,
        )

    # -------------------------------------------------------------- inference --
    @classmethod
    def infer(
        cls,
        names: Sequence[str],
        *,
        euler_seq: str = DEFAULT_EULER_SEQ,
        quat_order: str = "wxyz",
        position_unit: str = "m",
        joint_unit: str = "rad",
    ) -> ActionSpace:
        """Guess a space from channel names, falling back to joint space.

        Recognizes the common end-effector spellings (``x/y/z`` plus ``roll/pitch/yaw``,
        ``rx/ry/rz``, or ``qw/qx/qy/qz``) and otherwise treats the vector as joints. LeRobot
        suffixes like ``.pos`` and prefixes like ``action.`` are stripped before matching.

        Inference is a convenience, not a contract. When it matters, construct the space
        explicitly: a wrong guess is silent and changes every number downstream.
        """
        raw = [str(n) for n in names]
        keys = [_norm(re.sub(r"^(action|observation)\.?", "", n).replace(".pos", "")) for n in raw]

        pos_keys = ("x", "y", "z")
        has_pos = len(keys) >= 3 and tuple(keys[:3]) == pos_keys

        if has_pos:
            rest = keys[3:]
            rot: Rotation | None = None
            if len(rest) >= 3 and tuple(rest[:3]) in (("roll", "pitch", "yaw"), ("rl", "pt", "yw")):
                rot = "euler"
            elif len(rest) >= 3 and tuple(rest[:3]) == ("rx", "ry", "rz"):
                rot = "rotvec"
            elif len(rest) >= 4 and set(rest[:4]) == {"qw", "qx", "qy", "qz"}:
                rot = "quat"
                quat_order = "".join(k[1] for k in rest[:4])
            elif len(rest) >= 6 and all(k.startswith("r6d") for k in rest[:6]):
                rot = "rot6d"

            if rot is not None:
                rdim = ROTATION_DIM[rot]
                tail = keys[3 + rdim :]
                has_grip = len(tail) == 1
                if len(tail) <= 1:
                    space = cls.cartesian(
                        rotation=rot,
                        euler_seq=euler_seq,
                        quat_order=quat_order,
                        position_unit=position_unit,
                        gripper=has_grip,
                        position_names=raw[:3],
                        rotation_names=raw[3 : 3 + rdim],
                        gripper_name=raw[-1] if has_grip else "gripper",
                    )
                    return space

        return cls.joint(raw, unit=joint_unit)

    # ------------------------------------------------------------- diagnostics --
    def gimbal_risk(self, actions: np.ndarray) -> np.ndarray | None:
        """Per-frame gimbal-lock proximity in ``[0, 1]``, or ``None`` if not applicable.

        Returns 1.0 at the singularity and falls to 0 away from it. Only meaningful for
        ``rotation="euler"``; every other representation is free of the artifact and
        returns ``None``.

        For a Tait-Bryan sequence (three distinct axes, e.g. ``xyz``) the singularity is the
        middle angle at +/-pi/2. For a proper Euler sequence (first axis repeated, e.g.
        ``zxz``) it is the middle angle at 0 or pi.
        """
        if self.rotation != "euler":
            return None
        a = np.asarray(actions, dtype=float)
        idx = self.group("orientation").channels
        middle = a[..., idx[1]]
        seq = self.euler_seq.lower()
        if seq[0] == seq[2]:  # proper Euler: singular at 0 and pi
            return np.abs(np.cos(middle))
        return np.abs(np.sin(middle))  # Tait-Bryan: singular at +/-pi/2

    def describe(self) -> str:
        lines = [f"ActionSpace(kind={self.kind}, dim={self.dim})"]
        if self.rotation:
            extra = f" seq={self.euler_seq}" if self.rotation == "euler" else ""
            if self.rotation == "quat":
                extra = f" order={self.quat_order}"
            lines.append(f"  rotation: {self.rotation}{extra}")
        for g in self.groups:
            chans = ", ".join(self.channels[i].name for i in g.channels)
            unit = f" [{g.unit}]" if g.unit else ""
            lines.append(f"  {g.label:<12} metric={g.metric:<9}{unit:<8} {chans}")
        return "\n".join(lines)
