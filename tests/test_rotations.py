import numpy as np
import pytest

from policy_visualizer import rotations as R


def test_euler_roundtrip_matches_geodesic_zero():
    rng = np.random.default_rng(0)
    e = rng.uniform(-np.pi, np.pi, size=(50, 3))
    q = R.euler_to_quat(e, seq="xyz")
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)
    assert np.allclose(R.geodesic_angle(q, q), 0.0, atol=1e-9)


def test_euler_single_axis_matches_known_angle():
    for _axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        e = np.zeros(3)
        e[idx] = 0.7
        q = R.euler_to_quat(e, seq="xyz")
        ident = np.array([1.0, 0.0, 0.0, 0.0])
        assert R.geodesic_angle(q, ident) == pytest.approx(0.7, abs=1e-9)


def test_geodesic_is_double_cover_safe():
    """q and -q are the same rotation and must score zero, not maximal."""
    rng = np.random.default_rng(1)
    q = R.normalize_quat(rng.normal(size=(20, 4)))
    assert np.allclose(R.geodesic_angle(q, -q), 0.0, atol=1e-9)


def test_geodesic_symmetry_and_range():
    rng = np.random.default_rng(2)
    a = R.normalize_quat(rng.normal(size=(30, 4)))
    b = R.normalize_quat(rng.normal(size=(30, 4)))
    d1, d2 = R.geodesic_angle(a, b), R.geodesic_angle(b, a)
    assert np.allclose(d1, d2)
    assert np.all(d1 >= 0) and np.all(d1 <= np.pi + 1e-9)


def test_canonicalize_removes_sign_flips_along_time():
    q = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (10, 1))
    q[3:7] *= -1  # inject a sign flip run
    out = R.canonicalize_quat(q, axis=0)
    dots = np.sum(out[1:] * out[:-1], axis=-1)
    assert np.all(dots > 0), "consecutive samples must share a hemisphere"


def test_canonicalize_reference_alignment():
    ref = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
    q = -ref.copy()
    out = R.canonicalize_quat(q, reference=ref, axis=0)
    assert np.all(np.sum(out * ref, axis=-1) >= 0)


def test_quat_order_xyzw_roundtrip():
    # Deliberately asymmetric: [0.5,0.5,0.5,0.5] is invariant under reordering and
    # would make this test pass even if the order were ignored entirely.
    wxyz = R.normalize_quat(np.array([[0.9, 0.3, 0.2, 0.1]]))
    xyzw = wxyz[:, [1, 2, 3, 0]]
    assert R.geodesic_angle(wxyz, xyzw, order="wxyz") > 0.1  # different if misread
    a = R.to_quat(xyzw, "quat", quat_order="xyzw")
    assert np.allclose(R.geodesic_angle(a, wxyz), 0.0, atol=1e-12)


def test_geodesic_is_machine_precise_at_identity():
    """The arccos form returns ~1e-8 here. Small disagreements are the common case,
    so that error floor would sit right on top of the signal we care about."""
    rng = np.random.default_rng(7)
    q = R.normalize_quat(rng.normal(size=(64, 4)))
    assert np.max(R.geodesic_angle(q, q)) < 1e-15


def test_geodesic_accurate_for_tiny_angles():
    for eps in (1e-3, 1e-5, 1e-7):
        q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
        q1 = R.euler_to_quat(np.array([[eps, 0.0, 0.0]]), seq="xyz")
        got = R.geodesic_angle(q0, q1)[0]
        assert got == pytest.approx(eps, rel=1e-6), f"eps={eps} got={got}"


def test_rotvec_small_angle_stable():
    v = np.array([[1e-12, 0.0, 0.0]])
    q = R.rotvec_to_quat(v)
    assert np.all(np.isfinite(q))
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0)


def test_rotvec_matches_euler_single_axis():
    q1 = R.rotvec_to_quat(np.array([0.0, 0.0, 1.1]))
    q2 = R.euler_to_quat(np.array([0.0, 0.0, 1.1]), seq="xyz")
    assert R.geodesic_angle(q1, q2) == pytest.approx(0.0, abs=1e-9)


def test_rot6d_is_orthonormal_and_recovers_rotation():
    rng = np.random.default_rng(3)
    e = rng.uniform(-np.pi, np.pi, size=(10, 3))
    q = R.euler_to_quat(e, seq="xyz")
    m = R.rot6d_to_matrix(np.concatenate([np.eye(3)[None].repeat(10, 0).reshape(10, 9)[:, :6]], -1))
    assert np.allclose(m @ np.swapaxes(m, -1, -2), np.eye(3), atol=1e-9)
    back = R.matrix_to_quat(R.rot6d_to_matrix(np.zeros((1, 6)) + np.array([1, 0, 0, 0, 1, 0])))
    assert R.geodesic_angle(back, np.array([[1.0, 0, 0, 0]])) == pytest.approx(0.0, abs=1e-9)
    assert q.shape == (10, 4)


def test_matrix_to_quat_stable_at_180_degrees():
    """Trace approaches -1 here, where the naive formula loses all precision."""
    m = np.diag([1.0, -1.0, -1.0])  # 180 deg about x
    q = R.matrix_to_quat(m)
    assert np.allclose(np.linalg.norm(q), 1.0)
    assert R.geodesic_angle(q, np.array([1.0, 0, 0, 0])) == pytest.approx(np.pi, abs=1e-6)


def test_unwrap_removes_branch_cut():
    t = np.linspace(0, 6 * np.pi, 200)
    wrapped = np.arctan2(np.sin(t), np.cos(t))
    out = R.unwrap_angles(wrapped[:, None], axis=0)[:, 0]
    assert np.max(np.abs(np.diff(out))) < 0.5
    assert np.max(np.abs(np.diff(wrapped))) > 5.0


def test_unwrap_tolerates_nan():
    x = np.array([0.0, 0.1, np.nan, 0.3])[:, None]
    out = R.unwrap_angles(x, axis=0)
    assert np.isnan(out[2, 0])
    assert np.all(np.isfinite(out[[0, 1, 3], 0]))


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        R.normalize_quat(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        R.rotvec_to_quat(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        R.euler_to_quat(np.zeros((3, 3)), seq="xy")
    with pytest.raises(ValueError):
        R.to_quat(np.zeros((3, 3)), "nope")
