"""Command line entry point: ``policy-viz``."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from .episode import ChunkEpisode
from .spaces import ActionSpace
from .synthetic import cartesian_episode, joint_episode
from .view import render


def _load_npz(path: pathlib.Path, args) -> ChunkEpisode:
    """Load an episode from a .npz holding ``chunks`` and optionally ``starts``/``gt``.

    Expected keys (only ``chunks`` is required):
      chunks  (K, H, D) or an object array of (L_k, D)
      starts  (K,)   step index per chunk; defaults to ``--stride``
      gt      (T, D)  executed actions
      names   (D,)    channel names, used to infer the space when one is not given
    """
    z = np.load(path, allow_pickle=True)
    if "chunks" not in z:
        raise SystemExit(f"{path}: no 'chunks' array (have {list(z.keys())})")

    chunks = z["chunks"]
    if chunks.dtype == object:
        chunks = [np.asarray(c, dtype=float) for c in chunks]

    starts = z["starts"] if "starts" in z else None
    gt = z["gt"] if "gt" in z else (z["ground_truth"] if "ground_truth" in z else None)
    names = [str(n) for n in z["names"]] if "names" in z else None

    space = None
    if args.space == "joint":
        space = ActionSpace.joint(names or [f"dim_{i}" for i in range(_dim(chunks))],
                                  unit=args.joint_unit)
    elif args.space == "cartesian":
        space = ActionSpace.cartesian(rotation=args.rotation, euler_seq=args.euler_seq,
                                      quat_order=args.quat_order)
    elif names is not None:
        space = ActionSpace.infer(names, euler_seq=args.euler_seq,
                                  quat_order=args.quat_order)

    return ChunkEpisode.from_chunks(
        chunks, starts, space=space, fps=args.fps, ground_truth=gt,
        stride=args.stride, delta=args.delta, names=names,
    )


def _dim(chunks) -> int:
    first = chunks[0] if isinstance(chunks, list) else chunks[0]
    return int(np.asarray(first).shape[-1])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="policy-viz",
        description="Interactive action-chunk debugging view (self-contained HTML).",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--npz", type=pathlib.Path,
                     help="load chunks/starts/gt/names from a .npz file")
    src.add_argument("--demo", choices=("joint", "cartesian", "gimbal"),
                     help="render a synthetic episode instead of real data")

    p.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("chunks.html"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--stride", type=int, default=None,
                   help="chunk stride, used when the input has no explicit starts")
    p.add_argument("--delta", action="store_true",
                   help="chunks hold per-step increments; integrate before plotting")
    p.add_argument("--space", choices=("auto", "joint", "cartesian"), default="auto")
    p.add_argument("--rotation", choices=("euler", "quat", "rotvec", "rot6d"),
                   default="euler")
    p.add_argument("--euler-seq", default="xyz",
                   help="Euler convention; lowercase extrinsic, uppercase intrinsic")
    p.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    p.add_argument("--joint-unit", default="rad")
    p.add_argument("--width", type=int, default=1180)
    p.add_argument("--window", type=float, nargs=2, metavar=("T0", "T1"),
                   help="initial brush window in seconds; defaults to the worst disagreement")
    p.add_argument("--open", action="store_true", help="open the result in a browser")
    args = p.parse_args(argv)

    if args.npz:
        ep = _load_npz(args.npz, args)
    else:
        demo = args.demo or "joint"
        if demo == "joint":
            ep = joint_episode(fps=args.fps)
        else:
            ep = cartesian_episode(fps=args.fps, euler_seq=args.euler_seq,
                                   through_gimbal=(demo == "gimbal"))

    print(ep.summary())
    out = render(ep, args.out, width=args.width,
                 window=tuple(args.window) if args.window else None)
    size = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({size:.2f} MB, self-contained)")

    if args.open:
        import webbrowser
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
