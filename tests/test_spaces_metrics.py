# Copyright 2026 Ben Yang
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from policy_visualizer import ActionSpace
from policy_visualizer.metrics import pairwise_disagreement


# ------------------------------------------------------------------- spaces --
def test_joint_space_splits_gripper_automatically():
    s = ActionSpace.joint(["a", "b", "gripper"])
    assert s.kind == "joint"
    assert s.group("joint").channels == (0, 1)
    assert s.group("gripper").channels == (2,)
    assert s.dim == 3


def test_joint_space_without_gripper():
    s = ActionSpace.joint(["a", "b"], gripper=None)
    assert [g.key for g in s.groups] == ["joint"]


def test_cartesian_default_is_euler_7dof():
    s = ActionSpace.cartesian()
    assert s.rotation == "euler"
    assert s.dim == 7
    assert s.names == ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    assert s.group("position").metric == "euclidean"
    assert s.group("orientation").metric == "geodesic"
    assert s.group("gripper").metric == "abs"


def test_cartesian_rotation_dims():
    assert ActionSpace.cartesian(rotation="quat").dim == 8
    assert ActionSpace.cartesian(rotation="rotvec").dim == 7
    assert ActionSpace.cartesian(rotation="rot6d").dim == 10


def test_only_euler_channels_are_marked_angular():
    """Quaternion components are not angles and must not be unwrapped."""
    e = ActionSpace.cartesian(rotation="euler")
    assert [c.angular for c in e.channels] == [False] * 3 + [True] * 3 + [False]
    q = ActionSpace.cartesian(rotation="quat")
    assert not any(c.angular for c in q.channels)


def test_groups_partition_channels_exactly():
    for s in (ActionSpace.cartesian(), ActionSpace.joint(["a", "b", "gripper"])):
        seen = sorted(i for g in s.groups for i in g.channels)
        assert seen == list(range(s.dim))


def test_infer_cartesian_from_names():
    s = ActionSpace.infer(["x", "y", "z", "roll", "pitch", "yaw", "gripper"])
    assert s.kind == "cartesian" and s.rotation == "euler"


def test_infer_strips_lerobot_decorations():
    s = ActionSpace.infer(
        ["action.x", "action.y", "action.z", "action.roll", "action.pitch",
         "action.yaw", "action.gripper"]
    )
    assert s.kind == "cartesian"


def test_infer_falls_back_to_joint():
    s = ActionSpace.infer(["shoulder_pan.pos", "elbow_flex.pos", "gripper.pos"])
    assert s.kind == "joint"
    assert s.group("gripper").channels == (2,)


def test_infer_quat_order_from_names():
    s = ActionSpace.infer(["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
    assert s.rotation == "quat" and s.quat_order == "xyzw"


# ------------------------------------------------------------------ metrics --
def test_position_error_is_euclidean_not_per_component():
    s = ActionSpace.cartesian()
    a = np.zeros((1, 7))
    b = np.zeros((1, 7))
    b[0, :3] = [3.0, 4.0, 0.0]
    d = pairwise_disagreement(s, s.group("position"), a, b)
    assert d[0] == pytest.approx(5.0)  # not 3, not 4, not 7


def test_orientation_error_is_geodesic_in_radians():
    s = ActionSpace.cartesian()
    a = np.zeros((1, 7))
    b = np.zeros((1, 7))
    b[0, 3] = 0.4  # roll
    d = pairwise_disagreement(s, s.group("orientation"), a, b)
    assert d[0] == pytest.approx(0.4, abs=1e-9)


def test_orientation_error_ignores_2pi_wrap():
    """A 2*pi difference in an Euler channel is the same rotation."""
    s = ActionSpace.cartesian()
    a = np.zeros((1, 7))
    b = np.zeros((1, 7))
    b[0, 5] = 2 * np.pi  # yaw wrapped a full turn
    d = pairwise_disagreement(s, s.group("orientation"), a, b)
    assert d[0] == pytest.approx(0.0, abs=1e-7)


def test_orientation_error_survives_gimbal_lock():
    """At pitch = pi/2, roll and yaw trade off exactly and the rotation is unchanged.

    Per-component Euler differencing reports a huge error here; the geodesic metric
    must report ~zero. This is the single strongest reason not to diff Euler columns.
    """
    s = ActionSpace.cartesian(euler_seq="xyz")
    a = np.array([[0, 0, 0, 0.3, np.pi / 2, 0.0, 0]], dtype=float)
    b = np.array([[0, 0, 0, 0.3 + 0.5, np.pi / 2, 0.5, 0]], dtype=float)
    naive = np.max(np.abs(a[0, 3:6] - b[0, 3:6]))
    geo = pairwise_disagreement(s, s.group("orientation"), a, b)[0]
    assert naive == pytest.approx(0.5)
    assert geo < 1e-6, f"geodesic should see no rotation change, got {geo}"


def test_joint_metric_is_max_abs_per_channel():
    s = ActionSpace.joint(["a", "b", "c"], gripper=None)
    a = np.zeros((1, 3))
    b = np.array([[0.1, -0.7, 0.2]])
    d = pairwise_disagreement(s, s.group("joint"), a, b)
    assert d[0] == pytest.approx(0.7)


def test_gimbal_risk_peaks_at_singularity():
    s = ActionSpace.cartesian(euler_seq="xyz")
    a = np.zeros((3, 7))
    a[:, 4] = [0.0, np.pi / 2, -np.pi / 2]
    risk = s.gimbal_risk(a)
    assert risk[0] == pytest.approx(0.0, abs=1e-9)
    assert risk[1] == pytest.approx(1.0, abs=1e-9)
    assert risk[2] == pytest.approx(1.0, abs=1e-9)


def test_gimbal_risk_none_for_quaternions():
    assert ActionSpace.cartesian(rotation="quat").gimbal_risk(np.zeros((3, 8))) is None


def test_proper_euler_sequence_has_different_singularity():
    s = ActionSpace.cartesian(euler_seq="zxz")
    a = np.zeros((2, 7))
    a[:, 4] = [0.0, np.pi / 2]
    risk = s.gimbal_risk(a)
    assert risk[0] == pytest.approx(1.0)  # singular at 0 for proper Euler
    assert risk[1] == pytest.approx(0.0, abs=1e-9)
