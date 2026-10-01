"""One interface over image sequences, single images and video files."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from ..errors import Invalid
from ..messages import Msg
from . import NotThere, images
from .sequence import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    FrameSequence,
    find_sequence,
    sequence_of,
)

DEFAULT_START_FRAME = 1001

_YUV_MATRIX = {5: av.video.reformatter.Colorspace.ITU601, 6: av.video.reformatter.Colorspace.ITU601,
               7: av.video.reformatter.Colorspace.SMPTE240M, 4: av.video.reformatter.Colorspace.FCC,
               9: av.video.reformatter.Colorspace.BT2020, 10: av.video.reformatter.Colorspace.BT2020}

# FFmpeg's decoding threads for a video (frame and slice threads, thread_type AUTO). PyAV's own default is SLICE, one
# thread for an H.264 of one slice per frame: 24.5 ms a frame for a 4K 4:4:4 10-bit one, 4.4 ms with AUTO. Frame threads
# each hold a frame (50 MB for that 4K), so they are bounded: 8 decoded as fast as FFmpeg's own choice (0: every core)
# at 60% of its memory (measured on a 4K 4:4:4 10-bit H.264 with one keyframe).
DECODE_THREADS = max(1, min(8, os.cpu_count() or 1))


def threaded(stream) -> None:
    """Decode `stream` with frame and slice threads (DECODE_THREADS): the pictures are the same, only faster."""
    stream.thread_type = "AUTO"
    stream.thread_count = DECODE_THREADS


@dataclass
class Source:
    """Frames keyed by frame number; pixels in their file encoding (not color converted)."""

    kind: str  # "sequence" | "video" | "still"
    display: str  # human readable input, e.g. /plates/sh010.####.exr
    frames: tuple[int, ...]
    width: int
    height: int
    fps: float | None  # known for video only
    colorspace_hint: str  # a file path for OCIO file rules
    colorspace: str | None = None  # detected from the stream (HDR video); wins over file rules
    _sequence: FrameSequence | None = None
    _video: Path | None = None
    _still: Path | None = None
    _start: int = DEFAULT_START_FRAME
    _rotate_cw: int = 0  # phone videos: clockwise turns needed to stand the picture upright

    def missing(self) -> list[int]:
        """Frames a sequence skips between its first and last (a video or a still has none)."""
        return self._sequence.missing() if self._sequence is not None else []

    def files(self) -> dict[int, Path]:
        """Frame -> image file (a sequence, or a still as one frame); videos have none."""
        if self._sequence is not None:
            return {f: self._sequence.path(f) for f in self.frames}
        if self._still is not None:
            return {self.frames[0]: self._still}
        raise Invalid(Msg("E-VIDEO-NOFRAMEFILES"))

    def iter_frames(self, frames: list[int] | None = None, scale: float = 1.0) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame, float32 HxWx3) in order; `scale` resizes video frames (proxy resolution)."""
        wanted = sorted(set(frames)) if frames is not None else list(self.frames)
        unknown = set(wanted) - set(self.frames)
        if unknown:
            raise Invalid(Msg("E-SOURCE-FRAMES", frames=sorted(unknown)[:10]))
        if self._sequence is not None:
            for f in wanted:
                yield f, images.read_rgb(self._sequence.path(f))
            return
        if self._still is not None:
            yield wanted[0], images.read_rgb(self._still)
            return
        keep = set(wanted)
        with open_video(self._video) as container:
            stream = container.streams.video[0]
            threaded(stream)
            picture = self._picture_of(stream, scale)
            for index, frame in enumerate(container.decode(stream)):
                number = self._start + index
                if number > wanted[-1]:
                    break
                if number in keep:
                    yield number, picture(frame)

    def _picture_of(self, stream, scale: float = 1.0):
        """How a decoded frame of this video's stream becomes float32 HxWx3 (iter_frames, VideoReader): the one way."""
        cc = stream.codec_context
        # more than 8 bits a component (the pixel format's own depth, not its name: nv12 is 8-bit): through rgb48
        deep = bool(cc.pix_fmt) and max(c.bits for c in av.VideoFormat(cc.pix_fmt).components) > 8
        # the YUV matrix the stream says it was made with (AVColorSpace: 5 / 6 BT.601 for SD, 9 / 10 BT.2020, 7 SMPTE
        # 240M, 4 FCC); unspecified or BT.709: BT.709. Decoding an SD BT.601 stream with 709 shifts its colours
        src_space = _YUV_MATRIX.get(int(cc.colorspace or 0), av.video.reformatter.Colorspace.ITU709)
        # full range when the stream says so (AVColorRange 2, JPEG) though its pixel format is not a yuvj one, else
        # limited: stated, not left to the converter's guess from the format's name
        src_range = (av.video.reformatter.ColorRange.JPEG if int(cc.color_range or 0) == 2 or (cc.pix_fmt or "").startswith("yuvj")
                     else av.video.reformatter.ColorRange.MPEG)
        # decode at the stored (unrotated) size, then turn upright
        sw, sh = (self.height, self.width) if self._rotate_cw in (90, 270) else (self.width, self.height)
        w = max(2, int(round(sw * scale / 2)) * 2)
        h = max(2, int(round(sh * scale / 2)) * 2)
        fmt = "rgb48le" if deep else "rgb24"

        def picture(frame) -> np.ndarray:
            out = frame.reformat(width=w, height=h, format=fmt, src_colorspace=src_space, interpolation="AREA",
                                 src_color_range=src_range, dst_color_range=av.video.reformatter.ColorRange.JPEG)
            rgb = out.to_ndarray().astype(np.float32) / (65535.0 if deep else 255.0)
            if self._rotate_cw:
                rgb = np.ascontiguousarray(np.rot90(rgb, k=-self._rotate_cw // 90))
            return rgb

        return picture


class VideoReader:
    """A video source's frames decoded forward by one decoder that stays open between reads (view/frames.py keeps one
    per video packet): a read after the last one goes on from where the decoder is, instead of decoding the file from
    its first frame again. Only one thread uses it at a time.

    Frames are numbered as iter_frames numbers them: the source's start plus the decoding order. Going back (a frame
    before the decoder's place) seeks to the nearest keyframe at or before it, by the timestamp the decoder met that
    frame with; after a seek a frame is numbered by its timestamp when it was met before, else as the one after the
    last numbered. Where that cannot be told (a frame without a timestamp, none numbered yet) the file is decoded from
    its start again, as iter_frames does."""

    def __init__(self, source: Source) -> None:
        if source._video is None:
            raise ValueError("VideoReader reads a video source")
        self.source = source
        self._container = None
        self._decoded = None  # the decoder's frames, in order
        self._seeked = False  # numbered by timestamps (after a seek) rather than by order from the start
        self._last: int | None = None  # the number of the last frame decoded
        self._pts: dict[int, int] = {}  # frame number -> its timestamp, as the decoder met it
        self._number: dict[int, int] = {}  # and back

    def _open(self) -> None:
        self.close()
        self._container = open_video(self.source._video)
        self._stream = self._container.streams.video[0]
        threaded(self._stream)
        self.picture = self.source._picture_of(self._stream)
        self._decoded = self._container.decode(self._stream)
        self._seeked, self._last = False, self.source._start - 1

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = self._decoded = self._last = None

    @property
    def next(self) -> int | None:
        """The number of the frame the decoder gives next without going back; None: unknown (not open, at the end)."""
        return None if self._container is None or self._last is None else self._last + 1

    def frames(self, first: int) -> Iterator[tuple[int, object]]:
        """(number, decoded frame) from `first` on, in order, until the video ends or the caller stops asking; frames
        before `first` are decoded (the way there) but not yielded. `picture(frame)` turns one into float32 HxWx3."""
        if self._container is None:
            self._open()
        elif self._last is None or first <= self._last:
            self._back_to(first)
        while True:
            frame = next(self._decoded, None)
            if frame is None:  # the end: reading on means going back
                self._last = None
                return
            if not self._seeked:
                number = self._last + 1
            elif frame.pts is not None and frame.pts in self._number:
                number = self._number[frame.pts]
            elif self._last is not None:
                number = self._last + 1
            else:  # nothing to number it by: from the start again, numbered by order
                self._open()
                yield from self.frames(first)
                return
            self._last = number
            if frame.pts is not None and number not in self._pts:
                self._pts[number], self._number[frame.pts] = frame.pts, number
            if number >= first:
                yield number, frame

    def _back_to(self, first: int) -> None:
        """Seek to the keyframe at or before frame `first` (its timestamp known), else start again from the start."""
        pts = self._pts.get(first)
        if pts is None or self._container is None:
            self._open()
            return
        self._container.seek(pts, stream=self._stream, backward=True, any_frame=False)
        self._decoded = self._container.decode(self._stream)
        self._seeked, self._last = True, None


# FFmpeg opens a video as what its content says, whatever its name: a playlist or a concat list "named" .mp4 would make
# it read other files of this server (or fetch addresses) into frames anyone can look at. Only the containers of
# VIDEO_EXTS, and only the file itself.
VIDEO_OPTIONS = {"format_whitelist": "mov,matroska,avi,mxf", "protocol_whitelist": "file"}


def open_video(path: Path) -> av.container.InputContainer:
    try:
        # the container's text tags (handler names and the like) may be in any encoding, GBK from a Chinese QuickTime
        # for one; nothing here reads them, so a byte that is not UTF-8 is replaced instead of failing the whole file
        return av.open(str(path), options=VIDEO_OPTIONS, metadata_errors="replace")
    except av.error.FFmpegError as exc:
        raise Invalid(Msg("E-VIDEO-UNREADABLE", file=path.name)) from exc


def open_source(spec: str | Path, start_frame: int = DEFAULT_START_FRAME) -> Source:
    """Open a sequence pattern / folder / frame file, or a video file.

    Video frames are numbered from `start_frame` (pipeline default 1001).
    """
    path = Path(spec).expanduser()
    if path.suffix.lower() in VIDEO_EXTS:
        if not path.is_file():
            raise NotThere(Msg("E-VIDEO-NOTFOUND", path=str(path)))
        with open_video(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            width, height = stream.codec_context.width, stream.codec_context.height
            from .color import video_colorspace

            colorspace = video_colorspace(stream.codec_context.color_trc or 0)
            # FFmpeg reports the display rotation counter-clockwise; turn the frames clockwise by its negative
            first = next(container.decode(video=0), None)
            count = stream.frames
            if not count:  # e.g. WebM stores no frame count: decode to count, from the start again
                container.seek(0)
                count = sum(1 for _ in container.decode(video=0))
            rotate_cw = int(round(-(getattr(first, "rotation", 0) or 0))) % 360 if first is not None else 0
            if rotate_cw in (90, 270):
                width, height = height, width
        return Source(
            kind="video",
            display=str(path),
            frames=tuple(range(start_frame, start_frame + count)),
            width=width,
            height=height,
            fps=fps,
            colorspace_hint=str(path),
            colorspace=colorspace,
            _video=path,
            _start=start_frame,
            _rotate_cw=rotate_cw,
        )

    if path.is_file() and path.suffix.lower() in IMAGE_EXTS and sequence_of(path) is None:
        # a picture without a frame number in its name (sequence.py): one frame at start_frame
        width, height = images.image_size(path)
        return Source(
            kind="still",
            display=str(path),
            frames=(start_frame,),
            width=width,
            height=height,
            fps=None,
            colorspace_hint=str(path),
            _still=path,
            _start=start_frame,
        )

    seq = find_sequence(path)
    suffix = seq.tail[seq.tail.rfind("."):].lower()  # the frame number may sit before more of the name: ####_left.png
    if suffix not in IMAGE_EXTS:
        raise Invalid(Msg("E-IMAGE-FORMAT", suffix=suffix or seq.tail))
    first = seq.path(seq.first)
    width, height = images.image_size(first)
    return Source(
        kind="sequence",
        display=seq.pattern,
        frames=seq.frames,
        width=width,
        height=height,
        fps=None,
        colorspace_hint=str(first),
        _sequence=seq,
    )
