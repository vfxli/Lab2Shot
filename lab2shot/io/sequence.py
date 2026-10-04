"""Image sequence paths the way CG pipelines write them: the one place that tells frame numbers and sequences apart.

Accepted spellings for the frame number in a written path:
    plate.####.exr     (# count = padding)
    plate.%04d.exr     (%d = no padding)
    plate.$F4.exr      (Houdini; $F = no padding)

Files on disk are read the way Nuke reads a folder: every run of digits in a name may be the frame number, wherever it
sits (plate.1001.exr, 000000_left.png, render_0001_beauty.exr). Names alike except for their digit runs share a
shape (render_#_beauty.exr); within a shape the run that changes from file to file is the frame number. Anything
else in the name tells sequences apart: 000000_left.png and 000000_right.png are two sequences, as are beauty and
depth passes. A folder holding several sequences is never read as one of them picked quietly: the caller lists
them (SeveralSequences) and the user chooses.

Rules:
- hidden and system files (._plate.0001.exr from macOS, .DS_Store, Thumbs.db, desktop.ini) are not pictures;
- a minus sign right after a dot or an underscore (or at the start) belongs to the number: plate.-005.exr is -5,
  shot-0005.exr is 5;
- padding: numbers written with leading zeros are padded to their width (#### = 0001 ... 9999, then 10000 on); a
  minus sign counts in the width, as printf's %04d, Nuke and Houdini write it (-002 0000 0001 is one #### sequence
  running through 0);
  numbers of one width without leading zeros keep that width (1001-1100: ####); numbers of mixed widths without
  leading zeros are unpadded (8, 9, 10: #). 1.png and 0001.png are frame 1 twice: two sequences, never one;
- several digit runs changing (cam01_0001 ... cam02_0100): the run with the most different values is the frame
  number (ties: the last, as Nuke), the others tell sequences apart;
- a numbered file with no sibling is a sequence of one frame at its number (the last digit run, as Nuke): a render
  of one frame keeps its frame number; only a name without digits is a still picture (photo.png);
- digits followed straight by a letter are part of a word, not a frame number: studio_2k.hdr, plate_4K.jpg and
  shot_1080p.png are still pictures, plate_4K.1001.exr is frame 1001 of plate_4K.####.exr. The trade-off, on purpose:
  render_1001b.exr next to render_1002b.exr are two pictures, not a sequence (a frame number is never glued to a
  letter in a pipeline's names; a resolution tag often is).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .. import i18n
from ..errors import Invalid, MessageError
from ..messages import Msg
from . import NotThere

IMAGE_EXTS = {".exr", ".png", ".jpg", ".jpeg", ".hdr"}  # .hdr: Radiance, how HDRI libraries (Poly Haven) ship
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".mxf"}

_TOKEN = re.compile(r"#+|%0?(\d*)d|\$F(\d*)")
# a digit run; a minus sign in front belongs to it only right after a dot, an underscore or the start of the name. Digits
# followed straight by a letter are part of a word, not a frame number (2k, 4K, 1080p, render_1001b)
_RUN = re.compile(r"(?<![^._])-\d+(?![\dA-Za-z])|\d+(?![\dA-Za-z])")
# the extension: a last dot followed by letters and digits, at least one letter (plate.1001 has none)
_EXT = re.compile(r"\.[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*$")
JUNK_NAMES = {"thumbs.db", "desktop.ini", "icon\r"}


def is_junk(name: str) -> bool:
    """A file no picture reader should take: hidden (a leading dot: macOS's ._ copies, .DS_Store) or a system file."""
    return name.startswith(".") or name.lower() in JUNK_NAMES


@dataclass(frozen=True)
class FrameSequence:
    directory: Path
    head: str  # "sh010_plate."
    tail: str  # ".exr"
    padding: int  # digits, 0 = unpadded
    frames: tuple[int, ...]

    def path(self, frame: int) -> Path:
        return self.directory / f"{self.head}{_spell(frame, self.padding)}{self.tail}"

    @property
    def pattern(self) -> str:
        """Display form, e.g. /plates/sh010_plate.####.exr"""
        return str(self.directory / self.name)

    @property
    def name(self) -> str:
        """The pattern's file name: sh010_plate.####.exr"""
        return f"{self.head}{'#' * max(self.padding, 1)}{self.tail}"

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    def missing(self) -> list[int]:
        present = set(self.frames)
        return [f for f in range(self.first, self.last + 1) if f not in present]

    def found(self) -> str:
        """What was found of it, without its name: 1001-1100，100 帧. Only what a listing showed at that moment —
        never part of which sequence it is (nodes/core/input.py SequenceEntry)."""
        frames = format_frame_range(self.frames) if len(self.frames) < 4 else f"{_shown(self.first)}-{_shown(self.last)}"
        return i18n.t("io.sequence.found", frames=frames, count=len(self.frames))

    def describe(self) -> str:
        """How the sequence is listed to the user: sh010_plate.####.exr（1001-1100，100 帧）"""
        return i18n.t("io.sequence.described", name=self.name, found=self.found())

    def __len__(self) -> int:
        return len(self.frames)


class SeveralSequences(MessageError, ValueError):
    """A folder (or a set of picked files) holds more than one sequence: which one is the user's choice."""

    status = 400

    def __init__(self, folder: str, sequences: list[FrameSequence]):
        self.sequences = sequences
        listed = i18n.separator().join(s.describe() for s in sequences[:6]) + (i18n.t("io.more") if len(sequences) > 6 else "")
        super().__init__(Msg("E-SEQUENCE-SEVERAL", folder=folder, count=len(sequences), listed=listed, example=sequences[0].name))


def _spell(frame: int, padding: int) -> str:
    """A frame number as a file name writes it, as printf's %0Nd: -5 with #### is -005 (the sign counts in the width)."""
    return f"{frame:0{padding}d}" if padding else str(frame)


def _padded(run: str) -> bool:
    """A digit run written with leading zeros (0001, -002): its length, sign included, is its sequence's padding."""
    body = run.lstrip("-")
    return len(body) > 1 and body.startswith("0")


def _digits_match(digits: str, padding: int) -> bool:
    if padding <= 1:
        return digits.lstrip("-") == "0" or not digits.lstrip("-").startswith("0")
    # Frames beyond the padding width (e.g. 10000 with ####) are not zero-padded.
    return len(digits) == padding or (len(digits) > padding and not _padded(digits))


# ------------------------------------------------------------------ names -> sequences (the one detector)


@dataclass
class _Name:
    name: str
    pieces: list[str]  # the text between the digit runs (len(runs) + 1), the extension in the last one
    runs: list[str]


def _split(name: str) -> _Name:
    ext = _EXT.search(name)
    stem, suffix = (name[: ext.start()], ext.group()) if ext else (name, "")
    pieces, runs, at = [], [], 0
    for m in _RUN.finditer(stem):
        pieces.append(stem[at : m.start()])
        runs.append(m.group())
        at = m.end()
    pieces.append(stem[at:] + suffix)
    return _Name(name, pieces, runs)


def shape_of(name: str) -> str:
    """Names of one shape differ in their digit runs only: render_0001_beauty.exr and render_0002_beauty.exr are
    render_#_beauty.exr. (lab2shot/client.py _shape keeps a standard-library copy of this rule; the two must stay equal.)"""
    return "#".join(p.replace("#", "##") for p in _split(name).pieces)


def _padding(runs: list[str]) -> list[tuple[int, list[str]]]:
    """Frame numbers of one sequence-to-be, as written with their sign, split by how they are written: [(padding,
    runs)]. Leading zeros fix a width (the sign counted, _padded); numbers of that width or longer without leading
    zeros join it; the rest are unpadded (their one width when all share one of two digits or more)."""
    widths = sorted({len(r) for r in runs if _padded(r)})
    groups: dict[int, list[str]] = defaultdict(list)
    loose = []
    for r in runs:
        if _padded(r):
            groups[len(r)].append(r)
        elif (fit := [w for w in widths if w <= len(r)]):
            groups[fit[-1]].append(r)
        else:
            loose.append(r)
    if loose:
        lengths = {len(r) for r in loose}
        groups[lengths.pop() if len(lengths) == 1 and min(len(r.lstrip("-")) for r in loose) > 1 and not widths else 0] += loose
    return sorted(groups.items())


@dataclass
class Grouping:
    """What a list of file names holds: sequences (longest first) and the files without a number."""

    sequences: list[tuple[str, str, int, dict[int, str]]] = field(default_factory=list)  # head, tail, padding, frame -> name
    singles: list[str] = field(default_factory=list)
    junk: list[str] = field(default_factory=list)


def group_names(names: Iterable[str]) -> Grouping:
    """File names (of one folder) -> the sequences and single files among them, the way Nuke lists a folder."""
    out = Grouping()
    shapes: dict[tuple[str, ...], list[_Name]] = defaultdict(list)
    for name in names:
        if is_junk(name):
            out.junk.append(name)
            continue
        n = _split(name)
        shapes[tuple(n.pieces)].append(n)
    for pieces, members in shapes.items():
        if not members[0].runs:
            out.singles += [m.name for m in members]
            continue
        count = len(members[0].runs)
        values = [{int(m.runs[i]) for m in members} for i in range(count)]
        varying = [i for i in range(count) if len(values[i]) > 1]
        # most different values; ties, or nothing changing (one file; 1.png and 01.png), the last run
        at = max(varying, key=lambda i: (len(values[i]), i)) if varying else count - 1
        split: dict[tuple[str, ...], list[_Name]] = defaultdict(list)
        for m in members:
            split[tuple(r for i, r in enumerate(m.runs) if i != at)].append(m)
        for others, part in split.items():
            head = "".join(p + r for p, r in zip(pieces[:at], others[:at])) + pieces[at]
            tail = pieces[at + 1] + "".join(r + p for r, p in zip(others[at:], pieces[at + 2 :]))
            by_run = defaultdict(list)
            for m in part:
                by_run[m.runs[at]].append(m)
            for padding, runs in _padding(list(by_run)):
                frames: dict[int, str] = {}
                for r in runs:
                    for m in by_run[r]:
                        frames.setdefault(int(m.runs[at]), m.name)
                out.sequences.append((head, tail, padding, dict(sorted(frames.items()))))
    out.sequences.sort(key=lambda s: (-len(s[3]), s[0] + s[1]))
    out.singles.sort()
    return out


def find_sequences(directory: str | Path) -> list[FrameSequence]:
    """Every picture sequence in a folder (EXR / PNG / JPG), longest first."""
    directory = Path(directory)
    names = [e.name for e in directory.iterdir() if e.suffix.lower() in IMAGE_EXTS and e.is_file()]
    return [FrameSequence(directory, h, t, p, tuple(f)) for h, t, p, f in group_names(names).sequences]


def sequence_of(path: str | Path) -> FrameSequence | None:
    """The sequence a file is a frame of (with its siblings on disk), or None for a name without a number."""
    path = Path(path)
    shape = shape_of(path.name)
    siblings = [e.name for e in path.parent.iterdir() if e.is_file() and shape_of(e.name) == shape]
    for h, t, p, frames in group_names(siblings).sequences:
        if path.name in frames.values():
            return FrameSequence(path.parent, h, t, p, tuple(frames))
    return None


# ------------------------------------------------------------------ a path as the user writes it


def _scan(directory: Path, head: str, tail: str, padding: int) -> tuple[int, ...]:
    regex = re.compile(re.escape(head) + r"(-?\d+)" + re.escape(tail) + "$")
    frames = []
    if directory.is_dir():
        for entry in directory.iterdir():
            m = regex.match(entry.name)
            if m and not is_junk(entry.name) and _digits_match(m.group(1), padding):
                frames.append(int(m.group(1)))
    return tuple(sorted(frames))


def _from_pattern(spec: Path) -> FrameSequence:
    name = spec.name
    tokens = list(_TOKEN.finditer(name))
    if len(tokens) != 1:
        raise Invalid(Msg("E-SEQUENCE-NOTOKEN", name=name))
    tok = tokens[0]
    text = tok.group(0)
    if text.startswith("#"):
        padding = len(text)
    else:
        digits = tok.group(1) if text.startswith("%") else tok.group(2)
        padding = int(digits) if digits else 0
    head, tail = name[: tok.start()], name[tok.end() :]
    frames = _scan(spec.parent, head, tail, padding)
    if not frames:
        raise NotThere(Msg("E-SEQUENCE-NOTFOUND", spec=str(spec)))
    return FrameSequence(spec.parent, head, tail, padding, frames)


def is_pattern(name: str) -> bool:
    """Does the file name hold a frame number placeholder (plate.####.exr, plate.%04d.exr, plate.$F4.exr)?"""
    return bool(_TOKEN.search(name))


def find_sequence(spec: str | Path) -> FrameSequence:
    """Resolve a pattern, a folder holding one sequence, or one frame of a sequence.

    A folder with several sequences raises SeveralSequences (listing them); a file without a frame number raises
    ValueError."""
    path = Path(spec).expanduser()
    if is_pattern(path.name):
        return _from_pattern(path)
    if path.is_dir():
        found = find_sequences(path)
        if not found:
            raise NotThere(Msg("E-SEQUENCE-EMPTYFOLDER", path=str(path)))
        if len(found) > 1:
            raise SeveralSequences(path.name, found)
        return found[0]
    if path.is_file():
        seq = sequence_of(path)
        if seq is None:
            raise Invalid(Msg("E-SEQUENCE-NOFRAMENUMBER", file=path.name))
        return seq
    raise NotThere(Msg("E-SEQUENCE-NOINPUT", spec=str(spec)))


def _shown(frame: int) -> str:
    """A frame number in a range shown to the user: a negative one in brackets, so its sign is not read as the dash
    of the range ((-2)-2, not -2-2)."""
    return f"({frame})" if frame < 0 else str(frame)


def format_frame_range(frames: list[int] | tuple[int, ...]) -> str:
    """[1001, 1002, 1003, 1010] -> '1001-1003,1010'; [-2, -1] -> '(-2)-(-1)'"""
    if not frames:
        return ""
    parts, start, prev = [], frames[0], frames[0]
    for f in list(frames[1:]) + [None]:
        if f is not None and f == prev + 1:
            prev = f
            continue
        parts.append(_shown(start) if start == prev else f"{_shown(start)}-{_shown(prev)}")
        if f is not None:
            start = prev = f
    return ",".join(parts)
