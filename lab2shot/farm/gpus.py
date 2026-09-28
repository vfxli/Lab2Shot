"""The GPUs of this machine, and which of them the administrator lets the farm use.

The authorization is kept by GPU UUID (indices change with the driver's ordering) in the database (meta
gpus.authorized). Until the administrator authorizes them on the admin page (/admin) no GPU takes jobs.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass

from ..database import db
from ..errors import Invalid
from ..messages import Msg

QUERY_TIMEOUT_S = 10.0  # the longest one nvidia-smi call may take (inventory.py stop waits by it)
QUERY = "index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,compute_cap"


@dataclass(frozen=True)
class Gpu:
    index: int  # as nvidia-smi numbers them (PCI bus order)
    name: str
    uuid: str
    memory_mb: int
    used_mb: int  # by every program on the GPU, not only the farm
    utilization: int  # percent
    temperature: int  # °C
    compute_cap: str = ""  # nvidia-smi's compute_cap ("8.9", "12.0"); "" when it could not be read

    @property
    def short_name(self) -> str:
        return self.name.removeprefix("NVIDIA ").removeprefix("GeForce ")

    def describe(self) -> dict:
        return {**asdict(self), "short_name": self.short_name}


def _number(text: str) -> int:
    try:
        return int(float(text))
    except ValueError:  # "[N/A]"
        return 0


def inventory() -> list[Gpu]:
    """The GPUs nvidia-smi sees now (empty without an NVIDIA driver)."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=QUERY_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        idx, name, uuid, total, used, util, temp, cap = [x.strip() for x in line.split(",")]
        gpus.append(Gpu(int(idx), name, uuid, _number(total), _number(used), _number(util), _number(temp),
                        "" if cap in ("", "[N/A]") else cap))
    return gpus


def authorized() -> list[str]:
    """UUIDs of the GPUs that take jobs."""
    return list(db().meta("gpus.authorized", []))


def authorize(uuids: list[str], known: list[Gpu]) -> None:
    """Let exactly these GPUs (of `known`, this machine's) take jobs."""
    names = {g.uuid for g in known}
    unknown = [u for u in uuids if u not in names]
    if unknown:
        raise Invalid(Msg("E-GPU-UNKNOWN", uuids=unknown))
    db().set_meta("gpus.authorized", sorted(set(uuids)))
