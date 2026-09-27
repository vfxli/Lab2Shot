"""Tracked points as Nuke nodes (.nk text): a Tracker4 with one track per point, and a plane's four corners as a
CornerPin2D.

Nuke's pixel origin is the bottom-left corner with y up, so every y is flipped (height - y) on output; frame numbers
remain the plate's own.

Tracker4 stores its tracks in a single table knob. The first line gives the table version, the column count and the
number of tracks; the columns are then described in a fixed order (COLUMNS below, exactly as Nuke writes them),
followed by one row per track with a value for every column. A time-varying value is an animation curve and a constant
is a plain number, as for any other knob.
"""

from __future__ import annotations

import numpy as np

from ...errors import Invalid
from ...messages import Msg
from . import script

TRACKER_NODE = "Tracker4"
CORNER_PIN_NODE = "CornerPin2D"
# Tracker4's table columns as (type, name). Types are Nuke's: 1 number, 2 animatable number, 3 name, 4 check box,
# 5 animatable check box.
COLUMNS: tuple[tuple[int, str], ...] = (
    (5, "enable"), (3, "name"), (2, "track_x"), (2, "track_y"), (2, "offset_x"), (2, "offset_y"),
    (4, "T"), (4, "R"), (4, "S"), (2, "error"), (1, "error_min"), (1, "error_max"),
    (1, "pattern_x"), (1, "pattern_y"), (1, "pattern_r"), (1, "pattern_t"),
    (1, "search_x"), (1, "search_y"), (1, "search_r"), (1, "search_t"),
    (2, "key_track"), (2, "key_search_x"), (2, "key_search_y"), (2, "key_search_r"), (2, "key_search_t"),
    (2, "key_track_x"), (2, "key_track_y"), (2, "key_track_r"), (2, "key_track_t"),
    (2, "key_centre_offset_x"), (2, "key_centre_offset_y"),
)
WIDTHS = {"enable": 20, "name": 75, "track_x": 58, "track_y": 58, "offset_x": 63, "offset_y": 63, "T": 27, "R": 27,
          "S": 27, "error": 45}  # display width of each column in Nuke; other columns are hidden (0)
PATTERN = 20  # half-size of a track's pattern box in pixels (close to Nuke's default)
SEARCH = 40  # half-size of the search box
MANY_TRACKS = 500  # above this the script becomes impractical to paste; the node reports it


def _column_lines() -> list[str]:
    return [f"{{ {kind} 1 {WIDTHS.get(name, 0)} {name} {'e' if name == 'enable' else name} 1 }}" for kind, name in COLUMNS]


def _keyed(frames: list[int], values) -> str:
    """One column's per-frame value: a curve, or a plain number when constant."""
    return script.curve(frames, values)


def write_tracker(xy, visible, frames: list[int], height: int, name: str, note: str = "", names=None) -> str:
    """Tracked points as a Tracker4 node: one track per point, keyed on every frame on which it is visible.

    `xy` [P,F,2] in image pixels (origin top-left, y down, the Lab2Shot convention), `visible` [P,F]. On frames where a
    point is hidden it keeps its last visible position and is disabled (enable 0), as a compositor would handle an
    occluded track. `names`: the point names (from the packet); if omitted, points are numbered after the node.
    """
    xy = np.asarray(xy, np.float64)
    seen = np.asarray(visible, bool)
    if not len(xy):
        raise Invalid(Msg("E-NUKE-NOTRACKS"))
    x = xy[..., 0]
    y = height - xy[..., 1]
    node = script.node_name(name, "Lab2Shot_tracker")
    rows = []
    for i in range(len(xy)):
        kept = np.maximum.accumulate(np.where(seen[i], np.arange(len(frames)), 0))  # index of the last visible frame
        px, py = x[i][kept], y[i][kept]
        row = [
            "{curve K " + " ".join(f"x{f} {1 if v else 0}" for f, v in zip(frames, seen[i])) + "}",
            script.quoted(str(names[i]) if names is not None and i < len(names) else name_of(i, node)),
            _keyed(frames, px), _keyed(frames, py),
            "{curve x%d 0}" % frames[0], "{curve x%d 0}" % frames[0],
            "1", "0", "0",  # translation only: a point tracker gives no rotation or scale
            "{curve x%d 0}" % frames[0], "0", "1",
            str(-PATTERN), str(-PATTERN), str(PATTERN), str(PATTERN),
            str(-SEARCH), str(-SEARCH), str(SEARCH), str(SEARCH),
            "{curve K x%d 1}" % frames[0],
            "{curve x%d %d}" % (frames[0], -SEARCH), "{curve x%d %d}" % (frames[0], -SEARCH),
            "{curve x%d %d}" % (frames[0], SEARCH), "{curve x%d %d}" % (frames[0], SEARCH),
            "{curve x%d %d}" % (frames[0], -PATTERN), "{curve x%d %d}" % (frames[0], -PATTERN),
            "{curve x%d %d}" % (frames[0], PATTERN), "{curve x%d %d}" % (frames[0], PATTERN),
            "{curve x%d 0}" % frames[0], "{curve x%d 0}" % frames[0],
        ]
        if len(row) != len(COLUMNS):  # every row must fill every column, since the table is read by position
            raise ValueError(f"a Tracker4 row has {len(row)} of {len(COLUMNS)} columns")
        rows.append(" { " + " ".join(row) + " }")
    table = ["{ { 1 %d %d }" % (len(COLUMNS), len(rows)), "{ " + "".join(f"\n{line}" for line in _column_lines()) + "\n}",
             "{" + "".join(f"\n{r}" for r in rows) + "\n}", "}"]
    knobs = [("tracks", "\n".join(table)), ("name", node),
             ("label", script.label(f"{note or 'tracked points'}, {len(rows)} tracks, frames {frames[0]}-{frames[-1]}"))]
    return script.script([script.block(TRACKER_NODE, knobs)])


def name_of(index: int, base: str) -> str:
    """The default name of an unnamed point: the node name plus a number (track_001)."""
    return f"{base}_{index + 1:03d}"


def corner_pin(xy, visible, query_frames, frames: list[int], height: int, name: str, note: str = "") -> str:
    """Four tracked points (a plane's corners, in order around it) as a Nuke CornerPin2D node: to1-to4 keyed on every
    frame, from1-from4 the corners on their start frame (or, if they start on different frames, the first frame on
    which all four are visible)."""
    if len(xy) != 4:
        raise Invalid(Msg("E-OUTPUT-CORNERPIN", count=len(xy)))
    starts = sorted(set(int(q) for q in query_frames))
    seen_all = [j for j in range(len(frames)) if visible[:, j].all()]
    ref = frames.index(starts[0]) if len(starts) == 1 and starts[0] in frames else (seen_all[0] if seen_all else 0)
    x = np.asarray(xy, np.float64)[..., 0]
    y = height - np.asarray(xy, np.float64)[..., 1]
    knobs = [(f"to{k + 1}", "{%s %s}" % (script.curve(frames, x[k]), script.curve(frames, y[k]))) for k in range(4)]
    knobs += [(f"from{k + 1}", "{%s %s}" % (script.number(x[k, ref]), script.number(y[k, ref]))) for k in range(4)]
    knobs += [("name", script.node_name(name, "Lab2Shot_cornerpin")),
              ("label", script.label(f"{note or 'planar track'}, start frame {frames[ref]}"))]
    return script.script([script.block(CORNER_PIN_NODE, knobs)])
