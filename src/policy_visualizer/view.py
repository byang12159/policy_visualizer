# Copyright 2026 Ben Yang
# SPDX-License-Identifier: Apache-2.0

"""The Bokeh overview + brush + live-detail view.

Layout is an overview of the whole episode above a brush, and an enlarged detail panel
below showing exactly the brushed window. The link between them is object identity, not a
callback: :class:`~bokeh.models.RangeTool` mutates the same ``Range1d`` instance that every
detail figure holds as its ``x_range``, so BokehJS redraws the detail panel on every mouse
move with no Python running at view time. The output is one self-contained HTML file.

What the action space changes here:

* Rows are ordered and visually banded by channel group, and each group's rows are labelled
  with that group's unit. Position rows say metres, orientation rows say radians; nothing
  invites you to compare them.
* Each group gets its own y-scaling. Sharing one scale across position and orientation
  would flatten whichever has the smaller numeric range into a straight line.
* The readout reports one disagreement figure per group in that group's own unit, rather
  than a single blended number.
* With Euler rotations, frames near gimbal lock are shaded in the orientation rows. There
  the individual roll/pitch/yaw traces can swing wildly while the actual rotation barely
  moves, so the per-axis rows are not trustworthy and the geodesic figure in the readout is
  the one to read.
"""

from __future__ import annotations

import pathlib

import numpy as np
from bokeh.layouts import column
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Range1d,
    RangeTool,
)
from bokeh.plotting import figure, save
from bokeh.resources import INLINE

from .episode import ChunkEpisode

C_EVEN = "#2563eb"
C_ODD = "#f97316"
C_GT = "#64748b"
C_DIS = "#e11d48"
C_GIMBAL = "#a855f7"

#: Left border reserved for row labels, identical on both panels so the overview and
#: detail frames line up to the pixel.
_BORDER_L = 132
_BORDER_R = 16

_GROUP_TINT = {
    "position": "#f8fafc",
    "orientation": "#fdf9ff",
    "joint": "#f8fafc",
    "gripper": "#fffdf5",
}


def _style(p, label: str, unit: str, small: bool) -> None:
    p.min_border_left = _BORDER_L
    p.min_border_right = _BORDER_R
    p.yaxis.axis_label = f"{label}  [{unit}]" if unit else label
    # Horizontal, not the default rotated label: a 46px overview row is shorter than
    # "shoulder_lift" set vertically, so rotated labels from adjacent rows collide.
    p.yaxis.axis_label_orientation = "horizontal"
    p.yaxis.axis_label_standoff = 8
    p.yaxis.axis_label_text_font_size = "10px" if small else "11px"
    p.yaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_color = "#334155"
    p.xaxis.axis_label_text_font_size = "11px"
    p.xaxis.axis_label_text_font_style = "italic"
    p.grid.grid_line_color = "#e2e8f0"
    p.outline_line_color = "#cbd5e1"


def _chunk_source(ep: ChunkEpisode, d: int) -> ColumnDataSource:
    """All K chunks for channel ``d`` as one ``multi_line`` glyph."""
    xs, ys, colors, labels = [], [], [], []
    for k in range(ep.n_chunks):
        xs.append(ep.chunk_times(k))
        ys.append(ep.display_chunks[k][:, d])
        colors.append(C_EVEN if k % 2 == 0 else C_ODD)
        s0 = int(ep.starts[k])
        labels.append(f"chunk {k} · starts step {s0} ({s0 / ep.fps:.2f}s)")
    return ColumnDataSource(dict(xs=xs, ys=ys, color=colors, label=labels))


def _head_source(ep: ChunkEpisode) -> ColumnDataSource:
    data = {
        "x": [float(ep.chunk_times(k)[0]) for k in range(ep.n_chunks)],
        "color": [C_EVEN if k % 2 == 0 else C_ODD for k in range(ep.n_chunks)],
    }
    for d in range(ep.space.dim):
        data[f"y{d}"] = [float(ep.display_chunks[k][0, d]) for k in range(ep.n_chunks)]
    return ColumnDataSource(data)


def _gimbal_spans(ep: ChunkEpisode, threshold: float = 0.95) -> list[tuple[float, float]]:
    """Contiguous time spans where Euler rows are degenerate."""
    if ep.gimbal_risk is None:
        return []
    bad = ep.gimbal_risk > threshold
    if not bad.any():
        return []
    spans, start = [], None
    for i, flag in enumerate(bad):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start / ep.fps, i / ep.fps))
            start = None
    if start is not None:
        spans.append((start / ep.fps, len(bad) / ep.fps))
    return spans


def build(
    ep: ChunkEpisode,
    *,
    width: int = 1180,
    window: tuple[float, float] | None = None,
    title: str = "Action-chunk debug view",
):
    """Assemble the Bokeh layout. Returns a model; use :func:`render` to write HTML."""
    space = ep.space
    t = ep.times()
    T, D = ep.n_frames, space.dim
    t_max = float(ep.duration)

    if window is None:
        # Open on the worst stretch of disagreement, which is where a debugger would go
        # first anyway, rather than on an arbitrary prefix.
        span = min(5.0, t_max)
        if ep.overview.size:
            centre = float(np.argmax(ep.overview)) / ep.fps
        else:
            centre = span / 2
        lo = float(np.clip(centre - span / 2, 0.0, max(0.0, t_max - span)))
        window = (lo, lo + span)
    w0, w1 = window

    gt_disp = ep.display_ground_truth
    gt_src = None
    if gt_disp is not None:
        gt_src = ColumnDataSource(
            {"t": t[: gt_disp.shape[0]], **{f"a{d}": gt_disp[:, d] for d in range(D)}}
        )

    dis_src = ColumnDataSource(
        {"t": t, "score": ep.overview if ep.overview.size == T else np.zeros(T),
         "zero": np.zeros(T)}
    )
    head_src = _head_source(ep)
    chunk_srcs = {d: _chunk_source(ep, d) for d in range(D)}
    gimbal = _gimbal_spans(ep)

    detail_x = Range1d(start=w0, end=w1, bounds=(0, t_max))
    ov_x = Range1d(start=0, end=t_max, bounds=(0, t_max))

    range_tool = RangeTool(x_range=detail_x, start_gesture="pan")
    ov = range_tool.overlay
    ov.fill_color, ov.fill_alpha = "#0ea5e9", 0.16
    ov.line_color, ov.line_width, ov.line_alpha = "#0369a1", 1.5, 0.9
    ov.use_handles = True
    ov.handles.all.fill_color = "#0369a1"
    ov.handles.all.line_color = "white"
    ov.handles.all.line_width = 2

    ordered: list[tuple[int, object]] = []
    for g in space.groups:
        for d in g.channels:
            ordered.append((d, g))

    # ------------------------------------------------------------- overview ----
    overview_rows = []
    strip = figure(height=48, width=width, x_range=ov_x, tools="",
                   toolbar_location=None, background_fill_color="#f8fafc",
                   x_axis_location=None)
    strip.varea("t", "zero", "score", source=dis_src, fill_color=C_DIS, fill_alpha=0.18)
    strip.line("t", "score", source=dis_src, color=C_DIS, line_width=1, alpha=0.8)
    strip.yaxis.major_label_text_font_size = "0pt"
    strip.yaxis.major_tick_line_color = strip.yaxis.minor_tick_line_color = None
    strip.yaxis.axis_line_color = None
    _style(strip, "disagreement", "", small=True)
    strip.add_tools(range_tool)
    overview_rows.append(strip)

    for i, (d, g) in enumerate(ordered):
        last = i == len(ordered) - 1
        p = figure(height=80 if last else 46, width=width, x_range=ov_x, tools="",
                   toolbar_location=None,
                   background_fill_color=_GROUP_TINT.get(g.key, "#f8fafc"),
                   x_axis_location="below" if last else None)
        if gt_src is not None:
            p.line("t", f"a{d}", source=gt_src, color=C_GT, line_width=2, alpha=0.45)
        p.multi_line("xs", "ys", source=chunk_srcs[d], line_color="color",
                     line_width=1.0, alpha=0.85)
        p.ygrid.grid_line_color = None
        p.yaxis.major_label_text_font_size = "0pt"
        p.yaxis.major_tick_line_color = p.yaxis.minor_tick_line_color = None
        p.yaxis.axis_line_color = None
        _style(p, space.channels[d].name, g.unit, small=True)
        p.add_tools(range_tool)
        overview_rows.append(p)
    overview_rows[-1].xaxis.axis_label = (
        "time (s)   -   OVERVIEW  ·  drag the box EDGES (two handles) to resize, "
        "the middle to pan")

    # --------------------------------------------------------------- detail ----
    detail_rows = []
    for i, (d, g) in enumerate(ordered):
        last = i == len(ordered) - 1
        p = figure(height=150 if last else 122, width=width, x_range=detail_x,
                   tools="xpan,xwheel_zoom,reset,save",
                   toolbar_location="right" if i == 0 else None,
                   background_fill_color="white",
                   x_axis_location="below" if last else None)
        if g.key == "orientation":
            for lo, hi in gimbal:
                p.add_layout(BoxAnnotation(left=lo, right=hi, fill_color=C_GIMBAL,
                                           fill_alpha=0.10, level="underlay"))
        if gt_src is not None:
            p.line("t", f"a{d}", source=gt_src, color=C_GT, line_width=3.5, alpha=0.5)
        r = p.multi_line("xs", "ys", source=chunk_srcs[d], line_color="color",
                         line_width=1.9, alpha=0.95)
        p.scatter("x", f"y{d}", source=head_src, size=5.5, marker="circle",
                  fill_color="color", line_color="white", line_width=1)
        p.add_tools(HoverTool(renderers=[r], tooltips="<b>@label</b>",
                              line_policy="nearest", attachment="above"))
        _style(p, space.channels[d].name, g.unit, small=False)
        detail_rows.append(p)
    detail_rows[-1].xaxis.axis_label = (
        "time (s)   -   DETAIL  ·  exactly the selected window, redrawn live while you drag")

    # -------------------------------------------------------------- readout ----
    # The readout wraps: 3 fixed chips plus one per group. A fixed height clips the
    # second line behind the detail panel as soon as a space has more than one group,
    # which every Cartesian space does.
    _chips = 3 + len(ep.disagreements)
    _lines = max(1, -(-_chips // 4))
    readout = Div(width=width, height=8 + 27 * _lines)
    chip = ("<span style='display:inline-block;padding:2px 9px;margin:0 7px 3px 0;"
            "border-radius:5px;background:#f1f5f9;border:1px solid #e2e8f0;"
            "font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#0f172a'>"
            "<b style='color:#475569;font-weight:600'>{}</b> {}</span>")

    groups_js = {
        k: [None if np.isnan(v) else float(v) for v in d.spread]
        for k, d in ep.disagreements.items()
    }
    labels_js = {k: d.label for k, d in ep.disagreements.items()}
    units_js = {k: d.unit for k, d in ep.disagreements.items()}

    js = """
const t0 = xr.start, t1 = xr.end;
const i0 = Math.max(0, Math.ceil(t0 * fps));
const i1 = Math.min(nframes - 1, Math.floor(t1 * fps));
let nc = 0;
for (let k = 0; k < starts.length; k++) {
  const a = starts[k] / fps, b = (starts[k] + lens[k] - 1) / fps;
  if (b >= t0 && a <= t1) nc++;
}
const nframesInWindow = Math.max(0, i1 - i0 + 1);
let html = chip.replace('{}', 'window')
                .replace('{}', t0.toFixed(2) + ' to ' + t1.toFixed(2) + ' s')
         + chip.replace('{}', 'span')
                .replace('{}', (t1 - t0).toFixed(2) + ' s / ' + nframesInWindow + ' frames')
         + chip.replace('{}', 'chunks').replace('{}', nc + ' in view');
for (const key of order) {
  const arr = groups[key];
  let sum = 0, peak = 0, n = 0;
  for (let i = i0; i <= i1; i++) {
    const v = arr[i];
    if (v === null || v === undefined) continue;
    sum += v; if (v > peak) peak = v; n++;
  }
  const mean = n ? sum / n : 0;
  const u = units[key] ? ' ' + units[key] : '';
  html += chip.replace('{}', labels[key])
              .replace('{}', 'mean ' + mean.toFixed(4) + u + ' · peak ' + peak.toFixed(4) + u);
}
div.text = html;
"""
    cb = CustomJS(
        args=dict(xr=detail_x, div=readout, chip=chip, fps=ep.fps, nframes=T,
                  starts=[int(s) for s in ep.starts],
                  lens=[int(c.shape[0]) for c in ep.chunks],
                  groups=groups_js, labels=labels_js, units=units_js,
                  order=list(groups_js)),
        code=js)
    detail_x.js_on_change("start", cb)
    detail_x.js_on_change("end", cb)
    readout.text = _readout_html(ep, chip, w0, w1)

    # --------------------------------------------------------------- header ----
    sw = ("<span style='display:inline-block;width:16px;height:0;border-top:3px solid "
          "{};vertical-align:middle;margin:0 5px 0 12px'></span>{}")
    rot = ""
    if space.rotation:
        rot = f" · rotation {space.rotation}"
        if space.rotation == "euler":
            rot += f" ({space.euler_seq})"
    gimbal_note = ""
    if gimbal:
        gimbal_note = (
            "<span style='display:inline-block;width:16px;height:9px;background:"
            f"{C_GIMBAL};opacity:.25;vertical-align:middle;margin:0 5px 0 12px'></span>"
            "near gimbal lock - per-axis rows unreliable")
    header = Div(width=width, height=48, text=(
        "<div style='font:600 16px/1.35 ui-sans-serif,system-ui;color:#0f172a'>"
        f"{title} <span style='font-weight:400;color:#64748b'>"
        "· overview + brush + live detail</span></div>"
        "<div style='font:12px/1.6 ui-sans-serif,system-ui;color:#475569'>"
        f"{ep.n_chunks} chunks · {D} channels · {T} frames @ {ep.fps:g} fps · "
        f"{space.kind} space{rot}"
        "<span style='color:#cbd5e1'> &nbsp;|&nbsp; </span>"
        + sw.format(C_EVEN, "even chunk") + sw.format(C_ODD, "odd chunk")
        + (sw.format(C_GT, "ground truth") if gt_src is not None else "")
        + gimbal_note + "</div>"))

    return column(header, *overview_rows,
                  Div(text="<div style='height:6px'></div>", width=width, height=6),
                  readout, *detail_rows, spacing=0)


def _readout_html(ep: ChunkEpisode, chip: str, t0: float, t1: float) -> str:
    i0 = int(np.ceil(t0 * ep.fps))
    i1 = min(ep.n_frames - 1, int(np.floor(t1 * ep.fps)))
    nc = sum(
        1 for k in range(ep.n_chunks)
        if (int(ep.starts[k]) + ep.chunks[k].shape[0] - 1) / ep.fps >= t0
        and int(ep.starts[k]) / ep.fps <= t1
    )
    html = (chip.format("window", f"{t0:.2f} to {t1:.2f} s")
            + chip.format("span", f"{t1 - t0:.2f} s / {max(0, i1 - i0 + 1)} frames")
            + chip.format("chunks", f"{nc} in view"))
    for d in ep.disagreements.values():
        seg = d.spread[i0 : i1 + 1]
        ok = seg[~np.isnan(seg)]
        mean = float(ok.mean()) if ok.size else 0.0
        peak = float(ok.max()) if ok.size else 0.0
        u = f" {d.unit}" if d.unit else ""
        html += chip.format(d.label, f"mean {mean:.4f}{u} · peak {peak:.4f}{u}")
    return html


def render(
    ep: ChunkEpisode,
    path: str | pathlib.Path,
    *,
    width: int = 1180,
    window: tuple[float, float] | None = None,
    title: str = "Action-chunk debug view",
) -> pathlib.Path:
    """Write a self-contained HTML file. No server and no network at view time."""
    out = pathlib.Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    layout = build(ep, width=width, window=window, title=title)
    save(layout, filename=str(out), resources=INLINE, title=title)
    return out
