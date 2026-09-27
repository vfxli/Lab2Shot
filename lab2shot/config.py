"""Project paths and the machine's settings.

Every setting is declared once, in SCHEMA below: key, kind, default, range, unit, the admin page's short label and
hover help, and, when a change only takes effect after the server restarts, the reason. Values come from the
following sources, later ones taking precedence:

    the schema's defaults
    config/local.toml          this machine's settings (not in git): written by the admin page's 设置 (/admin), also
                               editable by hand; it keeps only the values that differ from the defaults
                               ($LAB2SHOT_CONFIG names another file: tests and second servers)
    the environment            LAB2SHOT_WORK_DIR
    the command line           `lab2shot ui --host / --port / --https`, for that run

A setting that applies at once is read where it is used, every time (settings()["storage.delivery_days"]). One that
needs a restart keeps, in this process, the value it started with; the admin page says it is pending until the
server restarts.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import threading
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .errors import Invalid, MessageError, message_of
from .messages import Msg

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
ADAPTERS_DIR = ROOT / "adapters"
THIRD_PARTY_DIR = ROOT / "third_party"
WORKER_SDK_DIR = ROOT / "worker_sdk"
TEMPLATES_DIR = ROOT / "templates"
MENU_DIR = ROOT / "menu"  # category tree of the node menu and node placements (lab2shot/categories.py)
WEBUI_DIST = ROOT / "webui" / "dist"

# the admin page's groups, in its order
GROUPS = {"people": "人员", "queue": "队列与显存", "memory": "内存", "storage": "存储与清理", "network": "网络",
          "view": "视图", "install": "安装", "env": "环境"}


def _mirrors(value: str) -> Msg | str:
    bad = [u for u in value.split() if not re.match(r"^https://[A-Za-z0-9.-]+(:\d+)?(/\S*)?$", u)]
    return Msg("E-SETTINGS-MIRRORS", items=bad) if bad else ""


# ------------------------------------------------------------------ checks a value must pass when it is changed


def machine_memory_gb() -> float:
    """This machine's memory (MemTotal)."""
    try:
        for line in open("/proc/meminfo", encoding="ascii"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def _below_memory(gb: float) -> Msg | str:
    total = machine_memory_gb()
    return Msg("E-SETTINGS-MEMORY", total=total) if total and gb >= total else ""


def port_problem(port: int) -> Msg | str:
    """Why a server could not listen on `port` ("" it can)."""
    with socket.socket() as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return Msg("E-SETTINGS-PORTBUSY", number=port)
    return ""


def gpu_order(value: str) -> list[int]:
    """The CUDA numbers listed in 显卡优先顺序, in order, each once (separated by commas or spaces)."""
    out: list[int] = []
    for part in value.replace("，", ",").replace(",", " ").split():
        if re.fullmatch(r"[0-9]+", part) and int(part) not in out:
            out.append(int(part))
    return out


def _gpu_order(value: str) -> Msg | str:
    bad = next((p for p in value.replace("，", ",").replace(",", " ").split() if not re.fullmatch(r"[0-9]+", p)), None)
    return Msg("E-SETTINGS-GPUORDER", item=bad) if bad is not None else ""


def _names(value: str) -> Msg | str:
    """Host names or addresses, separated by spaces or commas: each one a certificate can name."""
    import ipaddress
    import re

    for name in value.replace("，", ",").replace(",", " ").split():
        try:
            ipaddress.ip_address(name)
        except ValueError:
            if not re.fullmatch(r"(?=.{1,253}$)([A-Za-z0-9-]{1,63}\.)*[A-Za-z0-9-]{1,63}", name):
                return Msg("E-SETTINGS-HOSTNAME", name=name)
    return ""


def _clock(value: str) -> Msg | str:
    """A time of day, HH:MM (24 hours)."""
    import re

    m = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
    if not m or int(m[1]) > 23 or int(m[2]) > 59:
        return Msg("E-SETTINGS-CLOCK", value=value)
    return ""


def _ocio(value: str) -> Msg | str:
    if not value:
        return ""
    import PyOpenColorIO as OCIO

    try:
        OCIO.Config.CreateFromFile(value)
    except Exception as exc:  # OCIO raises its own exception type
        return Msg("E-SETTINGS-OCIO", detail=str(exc))
    return ""


def _cuda(value: str) -> Msg | str:
    return "" if not value or (Path(value) / "bin" / "nvcc").is_file() else Msg("E-SETTINGS-NONVCC", folder=value)


def _compiler(value: str) -> Msg | str:
    return "" if not value or shutil.which(value) else Msg("E-SETTINGS-NOCOMPILER", compiler=value)


# ------------------------------------------------------------------ defaults this machine works out for itself


def cpu_cores() -> int:
    """This machine's CPU cores (1 when the system will not say)."""
    return os.cpu_count() or 1


def _interactive_pool() -> int:
    """Default number of immediate-cook slots: light tasks use no GPU, so it follows the core count:
    max(2, min(6, cores / 8))."""
    return max(2, min(6, cpu_cores() // 8))


def _interactive_pool_says() -> str:
    return f"这台机器 {cpu_cores()} 核：核数 ÷ 8 限在 2 到 6 之间，算出 {_interactive_pool()} 个"


def _reserved_cores() -> int:
    """Default number of reserved cores: cores / 4, at least 2 (at least 1 on machines with few cores).

    The server and the browser commonly run on the same machine, so a cook must not occupy the whole machine: a single
    COLMAP task saturates all cores for minutes, and the browser on that machine, including other tabs, stops
    responding until the task ends."""
    return max(1, min(cpu_cores() - 1, max(2, cpu_cores() // 4)))


def _reserved_cores_says() -> str:
    return f"这台机器 {cpu_cores()} 核：核数 ÷ 4、至少 2，留出 {_reserved_cores()} 个"


def worker_cpus() -> list[int]:
    """The core ids a cook may use; the reserved cores at the end are never used for computation.

    Computed in one place: worker processes set their affinity from it (`engine/resident.py`), and extensions with
    their own thread parameter (COLMAP's `num_threads`) receive its count. All worker processes share the same set of
    cores rather than being assigned per task, so the reserved cores stay free however many tasks run, leaving room
    for the browser and other programs."""
    have = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(cpu_cores()))
    keep = int(settings()["queue.reserved_cores"])  # type: ignore[arg-type]
    return have[: max(1, len(have) - keep)]


def cpu_budget() -> int:
    """Maximum number of threads a cook may use (the length of `worker_cpus()`)."""
    return len(worker_cpus())


def apply_reservation() -> None:
    """Restrict this process (the server itself) to the cores available for computation: light computations (file
    reads, masks, numeric operations) run in the server process, so without this the reserved cores would not be kept
    free. Called at service start and after the setting changes; failing to set the affinity is not an error."""
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(worker_cpus()))
    except OSError:
        pass


@dataclass(frozen=True)
class Auto:
    """A default this machine works out for itself, so moving to a bigger or smaller machine needs no setting changed.
    `value`: the default now; `says`: one line saying how it was worked out, for the admin page's 「按本机算」."""

    value: Callable[[], object]
    says: Callable[[], str]


# The three view proxy tiers. The owning table is lab2shot/view/proxy.py TIERS; it is copied here because config is
# the lowest layer and must not import upward (lower layers never import higher ones). Both copies must be identical.
_PROXY_TIERS = (512, 1024, 2048)
# The three local proxy tiers (default 1024); read by the browser only
_LOCAL_TIERS = (1024, 2048, 4096)


# ------------------------------------------------------------------ the schema


@dataclass(frozen=True)
class Setting:
    key: str  # <section>.<name>, as in config/local.toml
    group: str  # the admin page's group (GROUPS)
    label: str  # one short line, no brackets
    default: object
    help: str  # the hover text: what it does, what to consider
    kind: str = "number"  # number / int / bool / choice / text / list (of short names)
    min: float | None = None
    max: float | None = None
    unit: str = ""
    options: tuple[tuple[str, str], ...] = ()  # choice: (value, label)
    restart: str = ""  # why a change takes effect only after the server restarts ("": at once)
    admin: bool = True  # False: shown on the admin page, changed in config/local.toml only
    only_if: str = ""  # a switch this one needs on to matter
    empty: str = ""  # text: what leaving it empty means
    check: Callable | None = None  # a changed value's problem (a Msg; "" none), beyond kind and range
    auto: Auto | None = None  # the default is worked out from this machine (cores, cards), not a fixed number

    @property
    def default_now(self) -> object:
        """The default on this machine: what `auto` works out, else the fixed `default`. Everything that compares a
        value to 「the default」 (what the file keeps, what 恢复默认 gives back) goes through this one property."""
        return self.coerce(self.auto.value()) if self.auto else self.default

    def coerce(self, value: object) -> object:
        """The value as this setting keeps it; Invalid (its message for the admin) when it does not fit."""
        name = self.label
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise Invalid(Msg("E-SETTINGS-NOTBOOL", setting=name))
            return value
        if self.kind in ("number", "int"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise Invalid(Msg("E-SETTINGS-NOTNUMBER", setting=name))
            if self.kind == "int":
                if value != int(value):
                    raise Invalid(Msg("E-SETTINGS-NOTINT", setting=name))
                value = int(value)
            else:
                value = float(value)
            if (self.min is not None and value < self.min) or (self.max is not None and value > self.max):
                raise Invalid(Msg("E-SETTINGS-RANGE", setting=name, min=self.min, max=self.max, unit=f" {self.unit}" if self.unit else ""))
            return value
        if self.kind == "list":
            items = value.replace("，", ",").replace(",", " ").split() if isinstance(value, str) else value
            if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
                raise Invalid(Msg("E-SETTINGS-NOTLIST", setting=name))
            items = [x.strip() for x in items if x.strip()]
            if not items:
                raise Invalid(Msg("E-SETTINGS-EMPTYLIST", setting=name))
            if len(set(items)) != len(items):
                raise Invalid(Msg("E-SETTINGS-DUPLICATES", setting=name, items=sorted({x for x in items if items.count(x) > 1})))
            if bad := [x for x in items if len(x) > 12 or not x.replace("-", "").replace("_", "").isalnum()]:
                raise Invalid(Msg("E-SETTINGS-BADITEMS", setting=name, items=bad))
            return items
        if not isinstance(value, str):
            raise Invalid(Msg("E-SETTINGS-NOTTEXT", setting=name))
        value = value.strip()
        if self.kind == "choice" and value not in dict(self.options):
            raise Invalid(Msg("E-SETTINGS-CHOICE", setting=name, options=[label for _, label in self.options]))
        return value

    def says(self, value: object) -> str:
        """A value of this setting in words, as the admin page shows it (a switch as 开 / 关, a choice as its label,
        a number with its unit, nothing as what empty means). Said here, beside the setting, so the page never
        writes a second version of it."""
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        if self.kind == "bool":
            return "开" if value else "关"
        if self.kind == "choice":
            return next((label for v, label in self.options if v == value), str(value))
        if value == "":
            return f"空：{self.empty}"
        return f"{value}{f' {self.unit}' if self.unit else ''}"

    def describe(self) -> dict:
        return {"tip": f"{self.help}\n默认：{self.says(self.default_now)}", "default_text": self.says(self.default_now),
                "key": self.key, "group": self.group, "label": self.label, "help": self.help, "default": self.default_now,
                "auto": self.auto.says() if self.auto else "",
                "kind": self.kind, "min": self.min, "max": self.max, "unit": self.unit,
                "options": [{"value": v, "label": label} for v, label in self.options], "restart": bool(self.restart),
                "why": self.restart, "admin": self.admin, "only_if": self.only_if, "empty": self.empty}


_WHEN_IDLE = "只在队列里没有任务时清理。"

SCHEMA: dict[str, Setting] = {s.key: s for s in (
    # ---------------------------------------------------------------- 人员
    Setting("people.departments", "people", "部门", ["EFX", "CMP", "LAY", "ANI", "LGT"], kind="list", help=(
        "账号所属的部门，用空格或逗号隔开：管理页面「用户」里给每个账号选一个，使用统计按它分部门。加一个新部门就在"
        "后面写上；账号的部门不在这张表里时，统计照旧按原名列出（标「已不在列表」）。马上生效。")),
    # ---------------------------------------------------------------- 队列与显存
    Setting("queue.cpu_jobs", "queue", "CPU 任务数", 2, kind="int", min=1, max=32, unit="个", help=(
        "不用显卡、但要算很久的任务（视频转序列、场景投影成 2D、不用 SiftGPU 的 COLMAP……）同时最多跑几个，"
        "多的按提交顺序排队。机器上同时有训练或渲染、电脑变卡时调小；机器空闲、CPU 核多时调大。马上生效。")),
    Setting("queue.interactive_minutes", "queue", "立即计算限时", 10, kind="int", min=1, max=240, unit="分钟", help=(
        "只用到轻量节点、不排队马上开始的计算（读文件、数值、遮罩调整、用缓存交付……）最多算多久，超过就停下，"
        "算好的节点留在缓存里，再点计算接着算。防止分享出去的地址被人长时间占着服务器；"
        "常有单个轻量节点就要算很久的长镜头时调长。马上生效。")),
    Setting("queue.interactive_pool", "queue", "立即计算名额", 3, kind="int", min=1, max=16, unit="个",
            auto=Auto(_interactive_pool, _interactive_pool_says), help=(
                "只用轻量节点、不排队马上开始的计算（读文件、数值、看已经算好的结果）同时最多跑几个，多的等空位。"
                "轻量节点不占显卡，只吃 CPU，所以默认按这台机器的核数算：核数 ÷ 8，限在 2 到 6 之间。"
                "几个人同时看图、总要等空位时调大；机器核少、或同时在跑训练和渲染、页面变卡时调小。马上生效。")),
    Setting("queue.reserved_cores", "queue", "保留核心数", 8, kind="int", min=0, max=256, unit="个",
            auto=Auto(_reserved_cores, _reserved_cores_says), help=(
                "算任务**永远不许用**的核数：留给浏览器、DCC、别的程序。计算进程只在剩下的核上跑（绑核 + 降优先级），"
                "自己带线程参数的项目（COLMAP）也按剩下的核数传。**服务器和浏览器常常在同一台机器上**："
                "实际情况：一个 COLMAP 任务把 32 个核全吃满 207 秒，整个浏览器连别的标签页一起卡死，"
                "任务结束才恢复。默认按这台机器的核数算：核数 ÷ 4、至少 2。"
                "机器是专用服务器、没人在上面干别的时调到 0（算得最快）；一边算一边要用这台机器时调大。马上生效。")),
    # The factory default must not be lower than the largest frame count the bundled cards declare; otherwise on a
    # fresh install some cards are rejected by `B-JOB-TOOMANYFRAMES` and 「提交」 is disabled (the bundled maximum is
    # `深度与相机 · LingBot-Map` at 300 frames, then `动作生成 · Sketch2Anim 画线` at 235 frames). The frame count is
    # limited by this setting only and must not be limited again elsewhere.
    Setting("queue.max_frames", "queue", "帧数上限", 400, kind="int", min=1, max=10000, unit="帧", help=(
        "一次提交的计算最多算多少帧，超过的服务器直接拒绝并说清楚（网页在提交前就拦下，这里挡绕过网页直接提交的）。"
        "防止一个长镜头把显卡和流量占满。**只看这一次算的帧范围**，和素材本身有多少帧无关：素材 1000 帧、"
        "只算其中 200 帧照样能提交。**看图不受这一条限制**（显示一个节点触发的轻量计算照常），"
        "不然长镜头连看都看不了。人少、镜头长时调大；机器要留给别的事时调小。马上生效。")),
    Setting("queue.waiting_max", "queue", "排队上限", 30, kind="int", min=1, max=200, unit="个", help=(
        "「立即计算」最多有几个在等空位，再多的直接拒绝并说明服务器忙。"
        "防止一个人（或分享出去的地址）把队列塞满，让别人连看图都看不了。"
        "人多、等一会儿也能接受时调大；只有几个人用、宁可马上被拒绝也不要干等时调小。马上生效。")),
    Setting("queue.age_minutes", "queue", "久等优先", 15, kind="int", min=1, max=240, unit="分钟", help=(
        "一个要大显存的任务等了这么久，就先把它能用的那张卡留给它，短任务不再一个接一个地抢在它前面。"
        "调短：大任务早点轮到，短任务多等一会儿；调长：短任务更快，大任务可能等很久。"
        "镜头长短差别大、大任务老是排不上时调短。马上生效。")),
    Setting("queue.account_share", "queue", "单账号占卡", 50, kind="int", min=10, max=100, unit="%", help=(
        "别人也在排队时，一个账号同时最多占几成显卡：按成数算，所以 2 张卡、4 张卡、8 张卡都不用改。"
        "50% 表示 2 张卡里最多占 1 张、4 张里最多占 2 张。只有一张卡时这条不起作用（不能把唯一的一张判给别人）；"
        "没有别人在排队时也不起作用，卡不会空着。100% 表示不限，谁先提交谁先用。马上生效。")),
    Setting("queue.vram_margin_gb", "queue", "显存余量", 1.0, min=0.5, max=8, unit="GB", help=(
        "挑卡时在节点声明的显存之外再留这么多：节点声明的是模型本身的峰值，新开的计算进程还要一份 CUDA 环境"
        "（实测 4090 约 0.4 GB、5090 约 0.7 GB）。任务老是刚开始就显存不够时调大；显存紧、想让大任务挤得进去时调小，"
        "但调到 0.5 以下很容易失败。马上生效。")),
    Setting("queue.gpu_order", "queue", "显卡优先顺序", "", kind="text", check=_gpu_order,
            empty="够用的显卡中显存最小的优先", help=(
        "多张显卡都能计算某个任务时，按此顺序优先选用。填写 CUDA 编号，以逗号分隔，例如 1,0；编号按 PCI 总线顺序，"
        "与「显卡」页及 nvidia-smi 显示的编号一致。未列出的显卡排在其后，按原规则（够用的显卡中显存最小的优先）。"
        "仅在几张显卡都放得下任务时起作用：放不下的显卡不会因此被选用。把显存大的显卡排在前面时，小任务也会优先占用它，"
        "只能在大显卡上计算的任务可能因此多等，最长等到「久等优先」的时间后为它预留。立即生效。")),
    Setting("queue.gpu_jobs", "queue", "显卡任务", True, kind="bool", help=(
        "关：所有显卡都不再接新任务，跟一张都没授权一样；正在算的任务算完为止，排队的等这个开关重新打开后自动接着算。"
        "查看（导入、读取序列、数值、看已经算好的结果）不用显卡，不受影响。显卡要检修、机器要给训练腾出来时关掉。马上生效。")),
    Setting("queue.compute_jobs", "queue", "计算任务", True, kind="bool", help=(
        "关：服务器不再接受新的计算任务——要显卡的、算很久的 CPU 任务、还有交付（哪怕结果已经缓存，交付也算一次计算任务），"
        "都会被拒绝，说明现在只能查看；正在算的算完为止。查看（导入、读取序列、数值、看已经算好的结果）不受影响，永远能用。"
        "服务器要整个暂停接任务时关掉。马上生效。")),
    Setting("resident.keep", "queue", "常驻模型", True, kind="bool", help=(
        "开：任务算完后模型留在它的进程里，下一个用同样模型的任务不用重新加载，大模型省下一分钟以上。"
        "关：每个任务算完就结束进程，显存和内存马上还给别的程序，适合机器上同时有训练或渲染。关掉时已经常驻的模型立刻卸载。")),
    Setting("resident.idle_minutes", "queue", "空闲卸载", 30, kind="int", min=1, max=1440, unit="分钟",
            only_if="resident.keep", help=(
                "常驻的模型这么久没被用到就完全卸载，显存和内存都释放。"
                "长一些：隔一阵再算同一个项目也不用重新加载；短一些：显存和内存早点还给别的程序。")),
    Setting("resident.per_gpu", "queue", "每卡常驻数", 3, kind="int", min=0, max=8, unit="个", only_if="resident.keep", help=(
        "一张显卡上最多留几个空闲的常驻进程，正在计算的那个不算；再多的，最久没用的先卸载。"
        "每个进程在显卡上占约 0.5 GB 的 CUDA 环境，模型移到内存里时也一样。几个项目轮流用时调大，显存紧张时调小。")),
    Setting("resident.to_ram", "queue", "让路移到内存", True, kind="bool", only_if="resident.keep", help=(
        "别的项目要用这张显卡时，常驻的模型怎么让出显存。"
        "开：移到内存（移完机器还剩「保留内存」那么多时），下次用时几秒就移回显卡；"
        "关：直接卸载，下次用时重新加载，内存全留给别的程序。")),
    # ---------------------------------------------------------------- 内存
    Setting("memory.keep_free_gb", "memory", "保留内存", 8.0, min=2, max=512, unit="GB", check=_below_memory, help=(
        "机器至少留这么多内存给别的程序：任务等空出这么多才开始（节点声明要更多的按节点的），"
        "常驻模型也只在移过去之后还剩这么多时才移到内存。机器上同时有训练、渲染时调大，防止内存耗尽整台机器死机。")),
    # ---------------------------------------------------------------- 存储与清理
    Setting("storage.delivery_days", "storage", "结果保留", 3, kind="int", min=1, max=365, unit="天", help=(
        "输出节点写好的文件在服务器上等用户电脑取回，放这么多天后自动删除；过期了从缓存再算一次很快。"
        "用户常隔几天才取结果就调大，硬盘紧张就调小。")),
    Setting("storage.upload_days", "storage", "素材保留", 0, kind="int", min=0, max=3650, unit="天", help=(
        "用户上传的素材这么多天没被用到就自动删除；0 表示不自动删，只在「硬盘」里手动清理。"
        "删了之后，用到它的节点图要重新选文件上传。" + _WHEN_IDLE)),
    Setting("storage.cache_days", "storage", "缓存保留", 0, kind="int", min=0, max=3650, unit="天", help=(
        "算好的节点结果和模型的原始结果这么多天没被用到就自动删除；0 表示不按时间删。删了的节点下次要重新算。" + _WHEN_IDLE)),
    Setting("storage.cache_gb", "storage", "缓存上限", 0, kind="int", min=0, max=100_000, unit="GB", help=(
        "缓存超过这么大时，最久没用的先删，直到低于上限；0 表示不限。留出硬盘给素材和结果，"
        "又不想常常手动清理时设一个。" + _WHEN_IDLE)),
    Setting("storage.db_backups", "storage", "备份份数", 14, kind="int", min=2, max=365, unit="份", help=(
        "数据库（任务记录、使用统计、名字合并……）自动备份留几份：每天一份，另外每次升级数据库和每次合并名字前各一份，"
        "最旧的先删。备份在 work/db/backups/，很小；留多一些，出错时能退回更早的样子。")),
    Setting("storage.upload_gb", "storage", "上传上限", 50, kind="int", min=1, max=10_000, unit="GB", help=(
        "用户上传的一个文件最大多少，大了直接拒绝；另外服务器硬盘剩不到 10 GB 时不再收上传。"
        "分享给外面的人用时防止有人把硬盘传满；要传很大的视频或 EXR 就调大。马上生效。")),
    # The limit shared by every account without its own quota (`users.quota_gb` NULL falls back to this, see
    # server/quota.py), not only new accounts: after a change, existing accounts read the new value on the next read.
    Setting("storage.quota_gb", "storage", "账号配额", 100, kind="int", min=1, max=10_000, unit="GB", help=(
        "没单独设过配额的账号能用多少硬盘：上传的素材、待取回的结果、算好的缓存和存在服务器上的模板加起来。"
        "满了就不收新上传，页面上写明占了多少、上限多少、清哪里能腾多少，用户自己清了缓存接着用。"
        "单个账号的配额在「用户」里按账号改，改了马上生效。")),
    Setting("paths.cache_dir", "storage", "缓存位置", "", kind="text", empty="工作文件夹里的 cache",
            restart="缓存在哪个盘上，服务启动时定下", help=(
                "算好的节点结果放哪：生产里指向数据盘。相对路径从项目文件夹算起；空：工作文件夹里的 cache。"
                "换了以后原来的缓存留在旧位置，要一起搬过去，不搬的下次重新算。")),
    Setting("paths.uploads_dir", "storage", "素材位置", "", kind="text", empty="工作文件夹里的 uploads",
            restart="素材在哪个盘上，服务启动时定下", help=(
                "用户上传的素材放哪：生产里指向数据盘。相对路径从项目文件夹算起；空：工作文件夹里的 uploads。"
                "换了以后原来的素材留在旧位置，要一起搬过去，不搬的用户要重新选文件。")),
    Setting("logs.max_mb", "storage", "日志大小", 10, kind="int", min=1, max=1000, unit="MB", help=(
        "服务日志 work/logs/lab2shot.log 长到这么大就换一个新文件，旧的改名留着。")),
    Setting("logs.files", "storage", "旧日志份数", 5, kind="int", min=1, max=100, unit="份", help=(
        "留几份换下来的旧日志，更早的删掉；日志一共最多占「日志大小」×（份数 + 1）。要查更早的问题就调大。")),
    Setting("logs.debug", "storage", "详细日志", False, kind="bool", help=(
        "开：日志里另外记下 worker 进程空闲时打印的内容等更细的过程，查问题时用；平时关着，日志小，好读。")),
    # ---------------------------------------------------------------- 视图
    # View proxy size: three fixed tiers; any source size is scaled proportionally to the selected tier. The default is
    # the smallest tier, to be raised when the network is fast enough. A solve scales and compresses the proxy at this
    # tier within the same task (lab2shot/view/proxy.py), and the viewer only loads proxies. The tiers are
    # view/proxy.py TIERS, the single source. Changes apply at once: proxies of the old tier remain on disk, and
    # proxies of the new tier are generated and cached on first view. Delivered files are not affected.
    Setting("view.proxy_px", "view", "视图代理尺寸", "512", kind="choice",
            options=tuple((str(px), f"{px} 像素") for px in _PROXY_TIERS), help=(
                "二维那一半：视图里看到的每一帧有多大。解算算完就把结果等比缩到长边这么多像素、"
                "压好存起来，网页只拿这一份。512 最省流量，公网上播放最顺；线路好就调到 1024 或 2048，看得更清楚。"
                "比这一档还小的画面不放大。三维那一半看下面的「点云上限」。"
                "交付写出去的文件永远是原尺寸、原精度、无损，不受它影响；想看无损的就下载交付到 Nuke 或 DCC 里看。"
                "改了马上生效，下次看图就是新的这一档。")),
    # Local proxy: material on the user's own machine is not shown losslessly either. After a sequence is chosen, the
    # browser scales every layer of every frame to this tier in the background, compresses it and stores it in the
    # browser's private file system (on the user's disk); the viewer draws from this proxy and no data passes through
    # the server. Quality is below a lossless Nuke preview and above the server proxy's default tier. The setting
    # applies on the client side but is controlled by the administrator. The two values below are only passed to the
    # browser (via /api/server); the server itself does not act on them.
    Setting("view.local_px", "view", "本机代理尺寸", "1024", kind="choice",
            options=tuple((str(px), f"{px} 像素") for px in _LOCAL_TIERS), help=(
                "用户自己机器上的素材（读取序列选的、交付写回的）在视图里有多大：浏览器后台把每一帧每一层"
                "等比缩到长边这么多像素、压好存在用户自己的硬盘上，视图从这份画，不上传、不经过服务器。"
                "比服务器代理清楚，比 Nuke 里无损看要糊一点。改了马上生效，下次看图重新生成。")),
    Setting("view.local_cache_gb", "view", "本机缓存上限", 10, kind="int", min=1, max=500, unit="GB", help=(
                "浏览器私有文件系统里本机代理最多占用户硬盘多少。满了按最久没看的淘汰。"
                "这是硬盘，和浏览器内存无关；内存里只装用户显示过的那几层。")),
    # Point cloud limit: 3D has no compressed-preview switch; a cloud above the limit is always decimated. There is only
    # this limit, no lossless / compressed tiers. The default is small (5 MB) for the same reason the 2D proxy defaults
    # to the smallest tier: start conservatively and raise it when the network is fast enough. The size counted is what
    # is actually sent to the browser (about 15 bytes per point, about 7 bytes per depth cell); attributes the viewer
    # does not use are not sent and not counted.
    Setting("view.points_max_mb", "view", "点云上限", 5, kind="int", min=1, max=4096, unit="MB", help=(
        "三维那一半：一片点云一帧最多传这么大。超过就每 N 个点取一个，视图里一直写着"
        "「显示了 N / 共 M 点」，一眼看得出删了多少。删的只是看的那一份：留下的点坐标一个位都不改，"
        "计算、缓存和交付出去的文件一个点都不少。不到这个数的点云一个点不少地发。"
        "5 最省流量，公网上拖时间条最顺；线路好就调大，点更密。改了马上生效。"
        "二维那一半看上面的「视图代理尺寸」。")),
    # ---------------------------------------------------------------- 网络
    Setting("server.host", "network", "访问范围", "127.0.0.1", kind="choice",
            options=(("127.0.0.1", "只有本机"), ("0.0.0.0", "局域网")),
            restart="服务只在启动时打开监听的网络接口", help=(
                "只有本机：只有这台机器上的浏览器和程序能连。局域网：局域网里别的电脑的浏览器、DCC 插件都能连；"
                "WSL2 要开镜像网络才分得清各台机器。每个人都要用账号登录（管理页面「用户」里新建）。")),
    Setting("server.port", "network", "端口", 8765, kind="int", min=1024, max=65535, check=port_problem,
            restart="端口只在服务启动时打开", help=(
                "网页和 DCC 插件连接的端口：地址是 http(s)://服务器:端口。改了以后所有人的书签、DCC 插件"
                "和命令行的 --server 都要换成新地址。")),
    Setting("server.https", "network", "HTTPS", False, kind="bool", restart="证书只在服务启动时装上", help=(
        "用这台服务器自己的证书走 HTTPS。局域网里别的电脑的浏览器要把结果直接存进用户的文件夹、把节点图存回原文件，"
        "必须是 HTTPS，用 http 时保存都变成下载。每台用户电脑装一次证书：浏览器打开 /api/tls/ca.pem。"
        "用内网穿透分享时，最好让穿透工具在外面那一头用正式证书加密，这里就不用开。")),
    Setting("server.names", "network", "对外域名", "", kind="text", check=_names, only_if="server.https",
            empty="只用本机的名字和地址", restart="证书只在服务启动时做", help=(
                "别人从外面访问用的域名或 IP，用空格隔开，比如内网穿透给的 abc.example.com。"
                "开了 HTTPS、穿透工具只转发端口不管加密时，证书要写上这个名字，浏览器才认。"
                "空：证书只写这台机器自己的名字和地址。")),
    # ---------------------------------------------------------------- 安装
    Setting("install.retries", "install", "重试次数", 5, kind="int", min=1, max=20, unit="次", help=(
        "安装扩展包时，下载、克隆代码、装包遇到网络问题（断开、超时、服务器忙）时每个来源最多试几次。"
        "每次之间等的时间翻倍（1、2、4、8 秒……），不超过「最长等待」。试完还不行就换下一个来源（镜像）。")),
    Setting("install.backoff_max", "install", "最长等待", 60, kind="int", min=5, max=600, unit="秒", help=(
        "两次重试之间最多等多久。网络时断时续时调大一点，能少失败几次。")),
    Setting("install.mirror_hf", "install", "HF 镜像", "https://hf-mirror.com", kind="text", check=_mirrors,
            empty="只用 Hugging Face 官方", help=(
                "Hugging Face 连不上时按顺序试的镜像网址，用空格隔开。只有带 sha256 的模型文件才会从镜像下载，"
                "下载完和官方的 sha256 对上才用，对不上就删掉换下一个来源。登录令牌从不发给镜像。")),
    Setting("install.mirror_pypi", "install", "PyPI 镜像", "https://pypi.tuna.tsinghua.edu.cn/simple", kind="text",
            check=_mirrors, empty="只用 PyPI 官方", help=(
                "Python 包的官方源连不上时试的镜像（包索引网址），用空格隔开。只对带哈希锁文件（requirements.lock）"
                "的扩展包生效：每个包都要和锁里的官方哈希对上才装；没有锁文件的扩展包只从官方源装。")),
    Setting("install.mirror_github", "install", "GitHub 镜像", "", kind="text", check=_mirrors,
            empty="只用 GitHub 官方", help=(
                "GitHub 连不上时试的代理镜像，用空格隔开，官方网址接在镜像后面访问。代码按锁定的提交号取，"
                "拿到的提交号和锁定的不一样就不用；发布文件要和 sha256 对上才用。默认不设：请只填你信任的镜像。")),
    # ---------------------------------------------------------------- 环境
    Setting("color.config", "env", "OCIO 配置", "", kind="text", check=_ocio, empty="用 $OCIO 或内置 ACES",
            restart="色彩配置在服务启动时读入，网页里的色彩空间列表也跟着它", help=(
                "服务器上 OCIO 配置文件 config.ocio 的路径，或 ocio:// 开头的 OCIO 内置配置。"
                "空：用环境变量 $OCIO，没有就用 OCIO 内置的最新 ACES studio 配置。换配置会改变读图时默认的色彩空间。")),
    Setting("build.cuda_home", "env", "CUDA 路径", "", kind="text", check=_cuda, empty="用 $CUDA_HOME", help=(
        "安装扩展包时编译 CUDA 代码（detectron2 等）用的 CUDA 工具包文件夹，如 /usr/local/cuda-12.8。"
        "空：用 $CUDA_HOME，没有就用 /usr/local/cuda。下次安装扩展包时生效。")),
    Setting("build.cc", "env", "C 编译器", "", kind="text", check=_compiler, empty="系统默认", help=(
        "安装扩展包时用的 C 编译器，如 gcc-14。CUDA 对 gcc 的版本有上限，系统默认的 gcc 太新时在这里指定旧一点的。"
        "空：系统默认。下次安装扩展包时生效。")),
    Setting("build.cxx", "env", "C++ 编译器", "", kind="text", check=_compiler, empty="系统默认", help=(
        "安装扩展包时用的 C++ 编译器，如 g++-14，和 C 编译器同一个版本。空：系统默认。下次安装扩展包时生效。")),
    Setting("paths.work_dir", "env", "工作文件夹", "work", kind="text", admin=False,
            restart="缓存、任务记录、显卡授权都在这个文件夹里，服务启动时定下", help=(
                "缓存、上传的素材、待取回的结果、任务记录、显卡授权和日志都在这里；相对路径从项目文件夹算起。"
                "只能在 config/local.toml 里改：换了以后原来的这些都留在旧文件夹，要一起搬过去。")),
)}


# ------------------------------------------------------------------ the values


def config_file() -> Path:
    return Path(os.environ.get("LAB2SHOT_CONFIG") or CONFIG_DIR / "local.toml")


def _read(file: Path) -> dict[str, object]:
    """The settings a file sets, checked against the schema (an unknown name or a wrong value is an error: a typo
    never goes unnoticed). Only kind and range: whether a path exists is asked when the admin changes it."""
    if not file.exists():
        return {}
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(Msg("E-CONFIG-TOML", file=str(file), detail=str(exc))) from exc
    out: dict[str, object] = {}
    for section, items in data.items():
        if not isinstance(items, dict):
            raise SettingsError(Msg("E-CONFIG-SECTION", file=str(file), section=section))
        for name, value in items.items():
            key = f"{section}.{name}"
            if key not in SCHEMA:
                raise SettingsError(Msg("E-CONFIG-UNKNOWN", file=str(file), section=section, name=name))
            try:
                out[key] = SCHEMA[key].coerce(value)
            except ValueError as exc:
                raise SettingsError(Msg("E-CONFIG-VALUE", file=str(file), section=section, name=name, reason=message_of(exc))) from exc
    return out


def _toml(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, list)):
        return json.dumps(value, ensure_ascii=False)  # a TOML basic string, or an array of them
    return repr(value)


HEADER = """\
# 本机设置（不纳入版本控制）。由管理后台「设置」页与 ./setup.sh 的「设置」写入，也可手工编辑；修改后须重启服务。
# 仅记录与默认值不同的项；各项的含义与默认值见管理后台「设置」页。
# 保存设置时本文件按设置项的顺序整体重写，手工添加的注释不会保留。"""


def _write(file: Path, values: dict[str, object]) -> None:
    from .io.atomic import write_text

    lines = [HEADER]
    section = ""
    for s in SCHEMA.values():
        if s.key not in values:
            continue
        head, name = s.key.split(".", 1)
        if head != section:
            lines += ["", f"[{head}]"]
            section = head
        lines += [f"# {s.label}", f"{name} = {_toml(values[s.key])}"]
    write_text(file, "\n".join(lines) + "\n")


class SettingsError(MessageError, ValueError):
    """config/local.toml can't be read as settings (the server does not start)."""


class Settings:
    """This machine's settings, as this process sees them (see the module doc)."""

    def __init__(self, file: Path) -> None:
        self.file = file
        self._lock = threading.Lock()
        self.saved = _read(file)  # what the file sets
        self._stamp = self._file_stamp()  # the file as read: save() re-reads when it changed under this process
        self.command: dict[str, tuple[object, str]] = {}  # key -> (value, what set it) for this run
        if os.environ.get("LAB2SHOT_WORK_DIR"):
            self.command["paths.work_dir"] = (os.environ["LAB2SHOT_WORK_DIR"], "环境变量 LAB2SHOT_WORK_DIR")
        self.running = {k: self.value(k) for k, s in SCHEMA.items() if s.restart}  # what this process started with

    def _file_stamp(self) -> int | None:
        """The file's modification time (ns), None when it is not there."""
        try:
            return self.file.stat().st_mtime_ns
        except OSError:
            return None

    def _reload_if_changed(self) -> None:
        """Take what another process wrote since this one read the file (the menu `lab2shot setup` while the server
        runs, or the server while the menu is open): save() rewrites the whole file, so writing from a stale copy
        would silently undo the other's setting."""
        stamp = self._file_stamp()
        if stamp != self._stamp:
            self.saved = _read(self.file)
            self._stamp = stamp

    def file_value(self, key: str) -> object:
        """What the file sets, else the default: what the admin page edits."""
        return self.saved.get(key, SCHEMA[key].default_now)

    def value(self, key: str) -> object:
        """What is set now: this run's command line or environment, else the file, else the default."""
        return self.command[key][0] if key in self.command else self.file_value(key)

    def __getitem__(self, key: str) -> object:
        """What this process goes by: what is set now, or for a setting that needs a restart what it started with."""
        return self.running[key] if key in self.running else self.value(key)

    def run_with(self, key: str, value: object, what: str) -> None:
        """`lab2shot ui --port ...`: this run goes by this value (`what`: the option, for the admin page). The
        command line may name any address to listen on, not only the page's choices."""
        s = SCHEMA[key]
        self.command[key] = (value if s.kind == "choice" else s.coerce(value), what)
        if key in self.running:
            self.running[key] = self.command[key][0]

    @property
    def work_dir(self) -> Path:
        return self._folder("paths.work_dir", "work")

    @property
    def cache_dir(self) -> Path:
        """Where computed results are stored (read by data/store.py): the configured path, or cache in the work folder
        when empty."""
        return self._folder("paths.cache_dir", "") or self.work_dir / "cache"

    @property
    def uploads_dir(self) -> Path:
        """Where user uploads are stored (read by transfer/uploads.py): the configured path, or uploads in the work
        folder when empty."""
        return self._folder("paths.uploads_dir", "") or self.work_dir / "uploads"

    def _folder(self, key: str, default: str) -> Path | None:
        """The location a path setting points to (relative paths are resolved from the project folder; empty: None, and
        the caller applies its own default)."""
        text = str(self[key] or default)
        if not text:
            return None
        p = Path(text).expanduser()
        return p if p.is_absolute() else ROOT / p

    def pending(self) -> list[str]:
        """Settings changed since this process started that take effect after a restart."""
        return [k for k, v in self.running.items() if self.value(k) != v]

    def save(self, changes: dict[str, object]) -> list[str]:
        """Check and keep the administrator's changes (all or none); returns the keys that changed. A changed value
        is checked further (does the path exist, is the port free) before anything is written."""
        with self._lock:
            self._reload_if_changed()
            new = dict(self.saved)
            problems = {}
            for key, raw in changes.items():
                s = SCHEMA.get(key)
                if s is None or not s.admin:
                    problems[key] = Msg("E-SETTINGS-NOTHERE", key=key) if s is None else Msg("E-SETTINGS-FILEONLY", setting=s.label, file=self.file.name)
                    continue
                try:
                    value = s.coerce(raw)
                    if s.check and value != self.file_value(key) and (problem := s.check(value)):
                        raise Invalid(Msg("E-SETTINGS-CHECK", setting=s.label, reason=problem))
                except ValueError as exc:
                    problems[key] = message_of(exc)
                    continue
                if value == s.default_now:
                    new.pop(key, None)
                else:
                    new[key] = value
            if problems:
                raise InvalidSettings(problems)
            changed = [k for k in SCHEMA if new.get(k, SCHEMA[k].default_now) != self.saved.get(k, SCHEMA[k].default_now)]
            if changed:
                _write(self.file, new)
                self.saved = new
                self._stamp = self._file_stamp()
            if "queue.reserved_cores" in changed:
                apply_reservation()  # applies at once: this process is restricted with the new value (worker processes apply it when they start)
            return changed

    def _source(self, key: str, s: Setting) -> dict:
        """Where a setting's value comes from, for the admin page's tag: the administrator changed it, or it is the
        default, and for a default this machine works out (Auto), how it was derived. The sentence is composed here,
        not in the page, so no page duplicates it."""
        if key in self.saved:
            return {"source": "已改", "source_tip": f"管理员改过，不再跟着默认走（默认 {s.says(s.default_now)}）"}
        if s.auto:
            return {"source": "按本机算", "source_tip": f"默认随这台机器算出来：{s.auto.says()}。改了就一直按改的来"}
        return {"source": "", "source_tip": ""}

    def describe(self) -> dict:
        """For the admin page: every setting with its value, what this process goes by, what overrides the file."""
        return {
            "file": str(self.file),
            "groups": [{"id": g, "label": label} for g, label in GROUPS.items()],
            # every value that is shown in words is put into words here, beside the setting (Setting.says), so the
            # page never writes a second version of 开 / 关, a choice's label, a unit or what empty means
            "settings": [{**s.describe(), "value": self.file_value(k), "running": self[k],
                          "value_text": s.says(self.file_value(k)), "running_text": s.says(self[k]),
                          "overridden": self.command[k][1] if k in self.command else "",
                          **self._source(k, s)} for k, s in SCHEMA.items()],
            "pending": self.pending(),
        }


class InvalidSettings(MessageError, ValueError):
    """Changes that can't be kept: key -> why (a message each)."""

    def __init__(self, problems: dict[str, Msg]) -> None:
        super().__init__(Msg("E-SETTINGS-INVALID", problems=list(problems.values())))
        self.problems = problems


@cache
def settings() -> Settings:
    return Settings(config_file())
