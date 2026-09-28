"""Frame ranges: the frames a set of packets covers together (`union`, `shot_frames`)."""

from __future__ import annotations

from .packet import Packet


def union(packets: list[Packet]) -> list[int]:
    """Every frame present in any of the packets."""
    frames: set[int] = set()
    for p in packets:
        frames.update(p.meta.get("frames", []))
    return sorted(frames)


def shot_frames(packets: list[Packet]) -> list[int]:
    """The frames covered by packets used together: the union over those that are shots. A still (a single picture for
    the whole shot, such as a garbage matte or an HDRI) applies to every frame and therefore contributes its own frames
    only when all packets are stills."""
    return union([p for p in packets if not p.meta.get("still")] or packets)
