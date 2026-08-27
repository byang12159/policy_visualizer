# Copyright 2026 Ben Yang
# SPDX-License-Identifier: Apache-2.0

"""Interactive debugging views for action-chunking robot policies.

Overview + brush + live detail, emitted as one self-contained HTML file.

    from policy_visualizer import ActionSpace, ChunkEpisode, render

    space = ActionSpace.cartesian(rotation="euler")      # x y z roll pitch yaw grip
    ep = ChunkEpisode.from_chunks(chunks, starts, space=space, fps=30,
                                  ground_truth=executed)
    render(ep, "chunks.html")

Joint space and Cartesian space are both first class, and they are measured differently:
per-joint absolute error for joints, Euclidean position error plus geodesic orientation
error for poses. See :mod:`policy_visualizer.spaces` for why that distinction is not
cosmetic.
"""

from .episode import ChunkEpisode
from .metrics import (
    GroupDisagreement,
    all_disagreements,
    overview_score,
    pairwise_disagreement,
    seam_disagreement,
)
from .spaces import ActionSpace, Channel, ChannelGroup
from .view import build, render

__all__ = [
    "ActionSpace",
    "Channel",
    "ChannelGroup",
    "ChunkEpisode",
    "GroupDisagreement",
    "all_disagreements",
    "build",
    "overview_score",
    "pairwise_disagreement",
    "render",
    "seam_disagreement",
]

__version__ = "0.1.0"
