"""One way to say where a picture's pixels sit.

A picture has two windows, as EXR, Nuke and 3DE have them:
  display window  the plate frame — the format everything downstream measures against (`width` x `height`);
  data window     every pixel the file keeps, as (x, y, w, h) from the display window's top-left corner. It may
                  reach past the display window (overscan) or stop short of it (a render's bounding box).

The undistorted picture's canvas is a data window, not a second kind of size: the
plate frame stays the size everything is measured in, the canvas is the pixels around it. An ST-map file says the
same thing the same way, and so does a camera (data/camera.py: its plate size and its overscan). A packet carries
only `width`, `height` and `data_window` for this (`meta()`), never a separate overscan, canvas or window key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

Box = tuple[int, int, int, int]  # x, y, width, height


@dataclass(frozen=True)
class Window:
    width: int  # the display window: the plate frame
    height: int
    data: Box = (0, 0, 0, 0)  # x, y, w, h from the display window's corner; (0,0,0,0) means the display window itself

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        box = tuple(int(v) for v in self.data)
        if len(box) != 4:
            raise ValueError(f"a data window is (x, y, w, h), not {self.data!r}")
        if box[2:] == (0, 0):
            box = (0, 0, self.width, self.height)
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"a data window has pixels in it, not {box!r}")
        object.__setattr__(self, "data", box)

    # ---- the display window
    @property
    def plate(self) -> tuple[int, int]:
        """The plate frame's size (the display window): what every 2D size is measured against."""
        return self.width, self.height

    @property
    def display_box(self) -> Box:
        return 0, 0, self.width, self.height

    # ---- the data window
    @property
    def canvas(self) -> tuple[int, int]:
        """The data window's size: the canvas an undistorted plate sits on."""
        return self.data[2], self.data[3]

    @property
    def offset(self) -> tuple[int, int]:
        """Where the plate frame's top-left corner sits inside the data window (0, 0 without overscan)."""
        return -self.data[0], -self.data[1]

    @property
    def overscan(self) -> tuple[int, int, int, int]:
        """Pixels the data window keeps past the plate frame: left, top, right, bottom (never negative)."""
        x, y, w, h = self.data
        return max(0, -x), max(0, -y), max(0, x + w - self.width), max(0, y + h - self.height)

    @property
    def has_overscan(self) -> bool:
        return any(self.overscan)

    @property
    def same(self) -> bool:
        """Data window = display window: nothing kept past the plate frame and nothing missing."""
        return self.data == self.display_box

    # ---- reading and writing
    @property
    def exr_windows(self) -> tuple[Box, Box]:
        """(display, data) as io/images.windows gives them and lab2shot_worker.files.write_exr takes them."""
        return self.display_box, self.data

    def meta(self) -> dict:
        """What a packet stores: its size and its data window (data/contracts.py META)."""
        return {"width": self.width, "height": self.height, "data_window": list(self.data)}

    # ---- making one
    @classmethod
    def of(cls, meta: Mapping) -> Window:
        """A packet's window from its meta (a packet without a data window: the display window)."""
        return cls(int(meta["width"]), int(meta["height"]), tuple(meta.get("data_window") or (0, 0, 0, 0)))  # type: ignore[arg-type]

    @classmethod
    def of_size(cls, width: int, height: int) -> Window:
        return cls(width, height)

    @classmethod
    def of_files(cls, paths, width: int, height: int) -> Window:
        """The window of a sequence in a `width` x `height` format: the union of its files' data windows (headers
        only), so every frame of the packet is read in one box."""
        from ..io import images

        box = None
        for path in paths:
            _, (x, y, w, h) = images.windows(path)
            box = (x, y, x + w, y + h) if box is None else (min(box[0], x), min(box[1], y), max(box[2], x + w), max(box[3], y + h))
        if box is None:
            return cls(width, height)
        x0, y0, x1, y1 = box
        return cls(width, height, (x0, y0, x1 - x0, y1 - y0))

    @classmethod
    def of_file(cls, path) -> Window:
        """One file's own windows: its display window is the plate frame, its data window the canvas."""
        from ..io import images

        (_, _, w, h), data = images.windows(path)
        return cls(w, h, data)

    @classmethod
    def canvas_of(cls, plate: tuple[int, int], size: tuple[int, int], offset: tuple[int, int]) -> Window:
        """A plate frame of `plate` on a canvas of `size` whose top-left corner sits `offset` before it."""
        return cls(plate[0], plate[1], (-int(offset[0]), -int(offset[1]), int(size[0]), int(size[1])))

    @classmethod
    def centred(cls, plate: tuple[int, int], size: tuple[int, int], centre_px: tuple[float, float]) -> Window:
        """A canvas of `size` around a plate frame placed so the lens centre (`centre_px`, plate pixels) sits in the
        middle of the canvas within half a pixel, the plate frame kept inside it."""
        (w, h), (cw, ch), (cx, cy) = plate, size, centre_px
        left, top = int(round(cw / 2 - cx)), int(round(ch / 2 - cy))
        return cls.canvas_of((w, h), (cw, ch), (min(max(left, 0), cw - w), min(max(top, 0), ch - h)))

    def scaled(self, factor: float) -> Window:
        """The same window on a picture `factor` times as wide and tall (a proxy)."""
        x, y, w, h = self.data
        return Window(round(self.width * factor), round(self.height * factor),
                      (round(x * factor), round(y * factor), round(w * factor), round(h * factor)))
