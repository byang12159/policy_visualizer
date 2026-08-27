import numpy as np
import pytest

from policy_visualizer import ActionSpace, ChunkEpisode, render
from policy_visualizer.synthetic import cartesian_episode, joint_episode


def _dense(K=4, H=10, D=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(K, H, D))


# ------------------------------------------------------------ input shapes --
def test_dense_array_input():
    ep = ChunkEpisode.from_chunks(_dense(), stride=5, fps=30)
    assert ep.n_chunks == 4
    assert list(ep.starts) == [0, 5, 10, 15]


def test_ragged_list_input():
    chunks = [np.zeros((10, 3)), np.zeros((7, 3)), np.zeros((4, 3))]
    ep = ChunkEpisode.from_chunks(chunks, [0, 5, 10], fps=30)
    assert ep.n_chunks == 3
    assert [c.shape[0] for c in ep.chunks] == [10, 7, 4]


def test_nested_list_input():
    ep = ChunkEpisode.from_chunks([[[0.0, 1.0], [2.0, 3.0]]], [0], fps=30)
    assert ep.chunks[0].shape == (2, 2)


def test_torch_like_input_is_converted():
    class FakeTensor:
        def __init__(self, a):
            self._a = a

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    ep = ChunkEpisode.from_chunks([FakeTensor(np.zeros((5, 3)))], [0], fps=30)
    assert isinstance(ep.chunks[0], np.ndarray)


def test_unsorted_starts_are_reordered_with_their_chunks():
    a, b = np.full((3, 2), 1.0), np.full((3, 2), 2.0)
    ep = ChunkEpisode.from_chunks([a, b], [10, 0], fps=30)
    assert list(ep.starts) == [0, 10]
    assert ep.chunks[0][0, 0] == 2.0  # b moved to the front with its start


def test_starts_default_to_contiguous():
    ep = ChunkEpisode.from_chunks(_dense(K=3, H=4), fps=30)
    assert list(ep.starts) == [0, 4, 8]


def test_dim_mismatch_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="dim"):
        ChunkEpisode.from_chunks(_dense(D=6), stride=5,
                                 space=ActionSpace.cartesian())  # dim 7


def test_ragged_dims_rejected():
    with pytest.raises(ValueError, match="share an action dim"):
        ChunkEpisode.from_chunks([np.zeros((3, 2)), np.zeros((3, 5))], [0, 1])


def test_1d_chunk_rejected():
    with pytest.raises(ValueError, match="2-D"):
        ChunkEpisode.from_chunks([np.zeros(5)], [0])


def test_wrong_start_count_rejected():
    with pytest.raises(ValueError, match="starts"):
        ChunkEpisode.from_chunks(_dense(K=3), [0, 1])


def test_delta_actions_are_integrated():
    gt = np.zeros((20, 2))
    deltas = [np.ones((5, 2))]
    ep = ChunkEpisode.from_chunks(deltas, [0], fps=30, ground_truth=gt, delta=True)
    assert np.allclose(ep.chunks[0][:, 0], [1, 2, 3, 4, 5])


def test_delta_without_ground_truth_rejected():
    with pytest.raises(ValueError, match="ground_truth"):
        ChunkEpisode.from_chunks([np.ones((5, 2))], [0], delta=True)


def test_names_infer_the_space():
    ep = ChunkEpisode.from_chunks(
        _dense(D=7), stride=5,
        names=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    )
    assert ep.space.kind == "cartesian"


# ------------------------------------------------------------ display repair --
def test_euler_chunks_are_put_on_a_common_branch():
    """Two chunks 2*pi apart describe the same motion and must draw on top of each other."""
    space = ActionSpace.cartesian()
    gt = np.zeros((20, 7))
    a = np.zeros((10, 7))
    b = np.zeros((10, 7))
    b[:, 5] = 2 * np.pi  # yaw, one whole turn off
    ep = ChunkEpisode.from_chunks([a, b], [0, 5], space=space, fps=30, ground_truth=gt)
    gap = abs(ep.display_chunks[0][5, 5] - ep.display_chunks[1][0, 5])
    assert gap < 1e-6, f"branch alignment failed, gap={gap}"


def test_display_repair_does_not_touch_raw_values():
    space = ActionSpace.cartesian()
    b = np.zeros((10, 7))
    b[:, 5] = 2 * np.pi
    ep = ChunkEpisode.from_chunks([np.zeros((10, 7)), b], [0, 5], space=space,
                                  fps=30, ground_truth=np.zeros((20, 7)))
    assert ep.chunks[1][0, 5] == pytest.approx(2 * np.pi)


def test_quaternion_sign_flip_is_canonicalized():
    space = ActionSpace.cartesian(rotation="quat")
    gt = np.zeros((20, 8))
    gt[:, 3] = 1.0
    a = gt[:10].copy()
    b = gt[5:15].copy()
    b[:, 3:7] *= -1
    ep = ChunkEpisode.from_chunks([a, b], [0, 5], space=space, fps=30, ground_truth=gt)
    assert ep.display_chunks[1][0, 3] > 0


# ---------------------------------------------------------------- metrics ----
def test_in_flight_is_nan_where_only_one_chunk_covers():
    ep = ChunkEpisode.from_chunks([np.zeros((5, 2)), np.zeros((5, 2))], [0, 10],
                                  fps=30, n_frames=15)
    d = ep.disagreements["joint"]
    assert np.isnan(d.spread[0])  # single chunk cannot disagree with itself
    assert d.n_in_flight[0] == 1


def test_disagreement_detects_overlap_conflict():
    a = np.zeros((10, 2))
    b = np.full((10, 2), 0.5)
    ep = ChunkEpisode.from_chunks([a, b], [0, 5], fps=30, n_frames=15)
    d = ep.disagreements["joint"]
    assert d.spread[7] == pytest.approx(0.5)


def test_seam_disagreement_measures_tail_vs_next_head():
    a = np.zeros((10, 2))
    b = np.full((10, 2), 0.25)
    ep = ChunkEpisode.from_chunks([a, b], [0, 5], fps=30, n_frames=15)
    assert ep.seams["joint"][0] == pytest.approx(0.25)


def test_overview_score_is_finite_and_nonnegative():
    ep = cartesian_episode(n_frames=200)
    assert ep.overview.shape == (200,)
    assert np.all(np.isfinite(ep.overview)) and np.all(ep.overview >= 0)


def test_n_in_flight_matches_stride_ratio():
    ep = joint_episode(n_frames=300, horizon=50, stride=25)
    mid = ep.n_in_flight()[100:200]
    assert mid.max() == 2  # H/S == 2


# ------------------------------------------------------------------- view ----
def test_render_joint_writes_selfcontained_html(tmp_path):
    ep = joint_episode(n_frames=200)
    out = render(ep, tmp_path / "j.html")
    html = out.read_text()
    assert out.stat().st_size > 200_000
    assert "RangeTool" in html
    assert "cdn.bokeh.org" not in html, "must inline BokehJS, not fetch it"


def test_render_cartesian_labels_units(tmp_path):
    ep = cartesian_episode(n_frames=200)
    html = render(ep, tmp_path / "c.html").read_text()
    assert "[m]" in html and "[rad]" in html


def test_gimbal_demo_flags_the_singularity(tmp_path):
    ep = cartesian_episode(n_frames=300, through_gimbal=True)
    assert ep.gimbal_risk is not None
    assert ep.gimbal_risk.max() > 0.99
    assert "gimbal" in render(ep, tmp_path / "g.html").read_text().lower()


def test_render_without_ground_truth(tmp_path):
    ep = ChunkEpisode.from_chunks(_dense(K=5, H=10, D=3), stride=5, fps=30)
    assert ep.ground_truth is None
    assert render(ep, tmp_path / "n.html").stat().st_size > 100_000


def test_summary_reports_per_group_units():
    ep = cartesian_episode(n_frames=200)
    s = ep.summary()
    assert "position" in s and "orientation" in s
    assert "euclidean" in s and "geodesic" in s
