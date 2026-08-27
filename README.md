# policy-visualizer

Interactive debugging views for action-chunking robot policies.

An action-chunking policy re-plans every `S` steps and predicts `H` steps ahead, so several chunks are always in flight at once and they disagree where they overlap.
That disagreement is what the robot executes as a jerk, and it is invisible in a plot that draws each chunk in isolation.

This renders the whole episode as one **self-contained HTML file**: an overview of every channel, a two-handled brush on the time axis, and an enlarged detail panel that redraws live as you drag.
No server and no network at view time.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/policy-viz --demo joint      -o chunks.html --open
.venv/bin/policy-viz --demo cartesian  -o pose.html
.venv/bin/policy-viz --demo gimbal     -o gimbal.html
```

## Reading the view

Consecutive chunks alternate blue and orange, so every seam is a colour change and the gap between one chunk's tail and the next chunk's head is directly visible.
Grey is the executed trajectory. The red strip at the top ranks the episode by disagreement so you can steer the brush at the frames where the policy actually contradicts itself.

Drag the box **edges** to resize the window, the **middle** to pan.

## The API

```python
from policy_visualizer import ActionSpace, ChunkEpisode, render

space = ActionSpace.cartesian(rotation="euler")   # x y z roll pitch yaw gripper
ep = ChunkEpisode.from_chunks(
    chunks,                  # (K, H, D) array, or a list of ragged (L_k, D)
    starts,                  # (K,) step index per chunk
    space=space,
    fps=30,
    ground_truth=executed,   # (T, D), optional
)
print(ep.summary())
render(ep, "chunks.html")
```

`from_chunks` takes numpy arrays, torch tensors, or nested lists; dense `(K, H, D)` blocks or ragged lists of varying length; explicit `starts`, a fixed `stride`, or neither.
It validates hard, because a misread layout produces numbers that look plausible and are wrong.
Pass `delta=True` if your chunks hold per-step increments and they will be integrated from the ground truth at each chunk's start.

## Joint space vs Cartesian space

Which one you are in changes how error must be measured, and getting this wrong is silent.

**Joint space** is homogeneous. Every channel is a joint coordinate in the same unit, they are independent, and error is the per-joint absolute difference.

**Cartesian space** is not. A pose action mixes metres, radians and a unitless gripper in one vector, so:

- **Position error is a Euclidean norm** over `(x, y, z)`, in metres. One number, not three component errors.
- **Orientation error is a geodesic angle** on SO(3), in radians. That is what "the wrist is 4 degrees off" means.
- **They are never averaged together.** Adding metres to radians gives a number with no unit and no meaning. Each group is reported separately in its own unit.

The overview strip is the one deliberate exception: it normalizes each group by a robust scale before combining, so it ranks frames rather than measuring anything. It is never labelled with a unit.

## Why Euler needs care

Euler angles are the default because that is what most end-effector controllers expose, but they have two failure modes this handles for you.

**Branch cuts.** An angle crossing ±π jumps by 2π. Drawn raw that is a full-scale vertical cliff reading as catastrophic disagreement, and it is pure artifact.
Channels are unwrapped along time, and each chunk is then shifted by whole turns onto the same branch as the ground truth. Unwrapping chunks independently is not enough: two chunks can each be internally smooth and still land 2π apart.

**Gimbal lock.** Near the sequence's singularity the first and third angles become degenerate: roll and yaw can swing across their whole range while the actual rotation barely moves.
Per-component Euler disagreement is therefore *worst exactly where it is least meaningful*.

The reported orientation number sidesteps both by converting to quaternions and measuring the geodesic angle, which is invariant to wrapping and well-behaved at the singularity.
Frames near gimbal lock are shaded purple in the orientation rows, meaning: read the geodesic figure in the readout, not the individual traces.

`policy-viz --demo gimbal` sweeps pitch through ±π/2 to show it.

`rotation=` also accepts `"quat"` (with `quat_order` `wxyz` or `xyzw`, never inferred), `"rotvec"`, and `"rot6d"`.
Quaternions get double-cover canonicalization for the same reason Euler gets unwrapping: `q` and `-q` are the same rotation, and a naive difference reports maximal disagreement between identical orientations.

## Layout

| Module | Role |
| --- | --- |
| `spaces.py` | What each column means: channels, groups, units, which metric applies |
| `rotations.py` | Euler/quat/rotvec/rot6d conversion and the geodesic metric |
| `metrics.py` | Per-group disagreement, seam disagreement, overview ranking |
| `episode.py` | Input normalization, validation, display repair |
| `view.py` | The Bokeh overview + brush + detail builder |
| `synthetic.py` | Demo episodes for both space kinds |

The brush-to-detail link is object identity, not a callback: `RangeTool` mutates the same `Range1d` that every detail figure holds as its `x_range`, so BokehJS redraws on every mouse move with no Python involved.

```bash
.venv/bin/pytest        # 60 tests
.venv/bin/ruff check .
```
