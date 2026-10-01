"""Project paths and the machine's settings.

Every setting is declared once, in SCHEMA below: key, kind, default, range, unit, the admin page's short label and
hover help, and, when a change only takes effect after the server restarts, the reason. Values come from the
following sources, later ones taking precedence:

    the schema's defaults
    config/local.toml          this machine's settings (not in git): written by the admin page's 设置 pages (/admin), also
                               editable by hand; it keeps only the values that differ from the defaults
                               ($LAB2SHOT_CONFIG names another file: a second server)
    the environment            LAB2SHOT_WORK_DIR
    the command line           `lab2shot ui --host / --port / --https`, for that run

A setting that applies at once is read where it is used, every time (settings()["tasks.keep_days"]). One that
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



@dataclass(frozen=True)
class Page:
    """One settings page: the admin side list's 设置 band has one entry per page, and the setup menu's 设置 one line.
    A page is a named set of groups; each group is one card on the page, under its heading, and every Setting names
    its group. `tip`: what the page holds, for the side list's and the menu's hover text."""

    label: str
    tip: str
    groups: dict[str, str]  # group id -> its heading, in the page's order


# The settings pages, in their order. The admin page, the admin settings API and the setup menu all read this one
# definition: which pages there are, what they are called, and which groups (so which settings) each one holds.
PAGES: dict[str, Page] = {
    "register": Page("注册设置", "自己注册：开不开放、要不要邀请码、注册账号的有效期、配额和可用类别、全站上限；下面是邀请码的新建、停用和谁用它注册了，按邀请码或时间段批量停用",
                     {"register": "注册"}),
    "accounts": Page("账号设置", "环节表和账号默认配额；还有页面顶上的管理员通知和管理员自己的密码",
                     {"accounts": "账号"}),
    "compute": Page("计算与显卡", "接不接计算任务和显卡任务，一个节点最多算多久，一个任务和整台机器同时算几个节点，帧数上限，留给别的程序的核；"
                    "挑卡时的显存余量和顺序，常驻模型留多久、留几个、怎么让路，留给别的程序的内存和服务进程的内存复用",
                    {"compute": "计算调度", "gpu": "显卡", "resident": "常驻模型", "memory": "内存"}),
    "storage": Page("存储与视图", "任务在服务器上留几天、上传上限、数据放哪，数据库备份留几份，日志多大、留几份、要不要详细；"
                    "视图里的画面多大（服务器代理和用户本机代理）、本机缓存多大、点云一帧最多传多少",
                    {"storage": "任务与数据", "database": "数据库", "logs": "日志", "view": "视图"}),
    "network": Page("网络与安装", "访问范围、端口、HTTPS、对外域名、可信代理；装扩展包用的镜像和重试，编译目标架构、CUDA 与编译器，OCIO 配置",
                    {"network": "网络", "install": "下载", "env": "编译与色彩"}),
}

# every group -> the page it is on
PAGE_OF: dict[str, str] = {g: page for page, p in PAGES.items() for g in p.groups}


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


def proxy_networks(value: str) -> list:
    """The addresses and networks listed in 可信代理 (separated by spaces or commas), as ipaddress networks; an item
    that is neither is left out (the setting's check refuses it when it is changed)."""
    import ipaddress

    out = []
    for item in value.replace("，", ",").replace(",", " ").split():
        try:
            out.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            pass
    return out


def _proxies(value: str) -> Msg | str:
    import ipaddress

    for item in value.replace("，", ",").replace(",", " ").split():
        try:
            ipaddress.ip_network(item, strict=False)
        except ValueError:
            return Msg("E-SETTINGS-PROXY", item=item)
    return ""


# Options a setting offers that a higher layer declares (the tags an account may be given are nodes/tags.py's, the
# view proxy tiers view/proxy.py's): that layer provides them here by name when it is imported (provide_choices), and
# a Setting names them (`choices`). This module never imports upward. Reading config/local.toml may happen before the provider is
# imported, so a value read from the file is checked for its form only; a value saved (Settings.save, the admin
# page and the setup menu) is checked against the options in full, and refused when they are not provided.
_CHOICES: dict[str, Callable[[], tuple[tuple[str, str], ...]]] = {}


def provide_choices(name: str, options: Callable[[], tuple[tuple[str, str], ...]]) -> None:
    _CHOICES[name] = options


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


def _archs(value: list[str]) -> Msg | str:
    """Compile targets: each one an architecture CUDA code can be compiled for (lab2shot_shared.gpu_arch)."""
    from lab2shot_shared.gpu_arch import TARGET_LABELS

    bad = [a for a in value if a not in TARGET_LABELS]
    return Msg("E-SETTINGS-ARCH", items=bad, known=list(TARGET_LABELS)) if bad else ""


# ------------------------------------------------------------------ defaults this machine works out for itself


def cpu_cores() -> int:
    """This machine's CPU cores (1 when the system will not say)."""
    return os.cpu_count() or 1


def _cpu_nodes() -> int:
    """Default number of nodes without a GPU the machine runs at once: cores / 8, at least 1. Each such node uses
    several cores itself (a per-frame node runs its frames on up to eight threads, engine/cook.py FRAME_THREADS), so
    this counts nodes, not the cores they use."""
    return max(1, cpu_cores() // 8)


def _cpu_nodes_says() -> str:
    return f"这台机器 {cpu_cores()} 核：核数 ÷ 8，算出 {_cpu_nodes()} 个"


def _reserved_cores() -> int:
    """Default number of reserved cores: cores / 4, at least 2 (at least 1 on machines with few cores).

    The server and the browser commonly run on the same machine, so a cook must not occupy the whole machine: a single
    COLMAP task saturates all cores for minutes, and the browser on that machine, including other tabs, stops
    responding until the task ends."""
    return max(1, min(cpu_cores() - 1, max(2, cpu_cores() // 4)))


def _reserved_cores_says() -> str:
    return f"这台机器 {cpu_cores()} 核：核数 ÷ 4、至少 2，留出 {_reserved_cores()} 个"


@dataclass(frozen=True)
class Auto:
    """A default this machine works out for itself, so moving to a bigger or smaller machine needs no setting changed.
    `value`: the default now; `says`: one line saying how it was worked out, for the admin page's 「按本机算」."""

    value: Callable[[], object]
    says: Callable[[], str]


# The three local proxy tiers (default 1024); read by the browser only
_LOCAL_TIERS = (1024, 2048, 4096)

# what an item of a list setting may hold besides letters and digits (any script): the joiners of a short name
# (sm_89, a 环节 such as 解算（布料/毛发/肌肉）). Never a space or a comma: an item is one short name
LIST_MARKS = str.maketrans("", "", "-_/·()（）")


# ------------------------------------------------------------------ the schema


@dataclass(frozen=True)
class Setting:
    key: str  # <section>.<name>, as in config/local.toml
    group: str  # the card it is on (a group of PAGES)
    label: str  # one short line, no brackets
    default: object
    help: str  # the hover text: what it does, what to consider
    kind: str = "number"  # number / int / bool / choice / text / list (of short names) / multi (any of the options)
    min: float | None = None
    max: float | None = None
    unit: str = ""
    options: tuple[tuple[str, str], ...] = ()  # choice, multi: (value, label)
    choices: str = ""  # the options, by the name another layer provides them under (provide_choices)
    restart: str = ""  # why a change takes effect only after the server restarts ("": at once)
    admin: bool = True  # False: shown on the admin page, changed in config/local.toml only
    only_if: str = ""  # a switch this one needs on to matter
    empty: str = ""  # text: what leaving it empty means
    check: Callable | None = None  # a changed value's problem (a Msg; "" none), beyond kind and range
    auto: Auto | None = None  # the default is worked out from this machine (cores, cards), not a fixed number

    def offered(self) -> tuple[tuple[str, str], ...]:
        """The options of a choice or a multi: its own, or those provided under its `choices` (empty while the
        layer providing them is not imported)."""
        if not self.choices:
            return self.options
        found = _CHOICES.get(self.choices)
        return found() if found else ()

    def provided(self) -> bool:
        """Its options are known here (its own, or provided under `choices`): only then is a value checked in full."""
        return not self.choices or self.choices in _CHOICES

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
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise Invalid(Msg("E-SETTINGS-NOTLIST", setting=name))
            items = [x.strip() for x in value if x.strip()]
            if not items:
                raise Invalid(Msg("E-SETTINGS-EMPTYLIST", setting=name))
            if len(set(items)) != len(items):
                raise Invalid(Msg("E-SETTINGS-DUPLICATES", setting=name, items=sorted({x for x in items if items.count(x) > 1})))
            if bad := [x for x in items if len(x) > 12 or not x.translate(LIST_MARKS).isalnum()]:
                raise Invalid(Msg("E-SETTINGS-BADITEMS", setting=name, items=bad))
            return items
        if self.kind == "multi":
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise Invalid(Msg("E-SETTINGS-NOTLIST", setting=name))
            if not self.provided():  # read from the file before the options' layer: its form only (see _CHOICES)
                return list(dict.fromkeys(value))
            known = dict(self.offered())
            if bad := [x for x in value if x not in known]:
                raise Invalid(Msg("E-SETTINGS-CHOICE", setting=name, options=list(known.values())))
            return [v for v in known if v in value]  # each once, in the options' order
        if not isinstance(value, str):
            raise Invalid(Msg("E-SETTINGS-NOTTEXT", setting=name))
        value = value.strip()
        if self.kind == "choice" and self.provided() and value not in dict(self.offered()):
            raise Invalid(Msg("E-SETTINGS-CHOICE", setting=name, options=[label for _, label in self.offered()]))
        return value

    def says(self, value: object) -> str:
        """A value of this setting in words, as the admin page shows it (a switch as 开 / 关, a choice as its label,
        a number with its unit, nothing as what empty means). Said here, beside the setting, so the page never
        writes a second version of it."""
        if self.kind == "multi":
            return "、".join(label for v, label in self.offered() if v in value) or f"空：{self.empty}"
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        if self.kind == "bool":
            return "开" if value else "关"
        if self.kind == "choice":
            return next((label for v, label in self.offered() if v == value), str(value))
        if value == "":
            return f"空：{self.empty}"
        return f"{value}{f' {self.unit}' if self.unit else ''}"

    def describe(self) -> dict:
        return {"tip": f"{self.help}\n默认：{self.says(self.default_now)}", "default_text": self.says(self.default_now),
                "key": self.key, "group": self.group, "label": self.label, "help": self.help, "default": self.default_now,
                "auto": self.auto.says() if self.auto else "",
                "kind": self.kind, "min": self.min, "max": self.max, "unit": self.unit,
                "options": [{"value": v, "label": label} for v, label in self.offered()], "restart": bool(self.restart),
                "why": self.restart, "admin": self.admin, "only_if": self.only_if, "empty": self.empty}


_WHEN_IDLE = "只在队列里没有任务时清理。"

QUOTA_GB_MAX = 10_000  # the largest disk quota an account may have, GB: the setting's and one set per account (server/quota.py)

SCHEMA: dict[str, Setting] = {s.key: s for s in (
    # ---------------------------------------------------------------- 注册 (lab2shot/registration.py)
    Setting("register.open", "register", "开放注册", False, kind="bool", help=(
        "开了，登录页上就有「注册」：来的人自己填用户名、中文名、环节和密码建账号，建好直接登录，和在「用户」里建的账号"
        "一模一样、一起管。关着时只有管理员能建账号，注册的接口也不收。马上生效。")),
    Setting("register.invite", "register", "邀请码验证", True, kind="bool", only_if="register.open", help=(
        "开着时注册要填一个能用的邀请码（在这一页下面的邀请码列表里建，可以限次数、限到期、随时停用）；关了谁都能直接注册，"
        "只受下面的注册上限和按 IP 的限制。马上生效。")),
    Setting("register.days", "register", "注册账号有效期", 30, kind="int", min=1, max=3650, unit="天", only_if="register.open", help=(
        "自己注册的账号从注册起能用多少天，到期就登录不了；以后可以在「用户」里延期，和别的账号一样。改了只管以后注册的。")),
    Setting("register.quota_gb", "register", "注册账号配额", 20, kind="int", min=1, max=10_000, unit="GB", only_if="register.open", help=(
        "自己注册的账号能用多少硬盘（任务的文件夹、还没有任务用到的上传，加上存在服务器上的模板，缓存不算），注册时写进这个账号自己的配额，"
        "以后在「用户」里按账号改。改了只管以后注册的。")),
    Setting("register.tags", "register", "注册可用模型类别", ["commercial"], kind="multi", choices="account_tags",
            empty="只有基础节点", only_if="register.open", help=(
                "自己注册的账号能用哪些类别的节点和模板，和「用户」里新建账号时的「可用」是同一套；基础节点人人都能用。"
                "改了只管以后注册的。")),
    Setting("register.per_hour", "register", "每小时注册上限", 20, kind="int", min=1, max=100_000, unit="个", only_if="register.open", help=(
        "全站所有人加起来，一小时里最多注册这么多个账号；到了就自动暂停注册，「概览」里写着，日志里记一笔，过了这一小时自己恢复。"
        "对方换多少 IP 都绕不过去。马上生效。")),
    Setting("register.per_day", "register", "每天注册上限", 100, kind="int", min=1, max=1_000_000, unit="个", only_if="register.open", help=(
        "全站所有人加起来，24 小时里最多注册这么多个账号；到了就自动暂停注册，和每小时的上限一样。马上生效。")),
    # ---------------------------------------------------------------- 账号
    Setting("people.departments", "accounts", "环节", [
        "前期策划", "概念设计", "故事板", "预演", "镜头布局", "模型", "贴图", "材质", "绑定", "动画", "动作捕捉", "特效",
        "解算（布料/毛发/肌肉）", "场景环境", "数字绘景", "摄像机跟踪", "抠像", "擦除修补", "合成", "灯光", "渲染", "调色", "剪辑",
        "技术（TD/流程）", "研发", "制片管理", "数据管理", "其他"], kind="list", help=(
        "每个账号属于哪个制作环节，一条一个（名字里不能有空格和逗号）：管理页面「用户」里给每个账号选一个，"
        "自己注册的人在注册页上从这张表里选，使用统计按它分环节。加一个新环节就添加一条；账号的环节不在这张表里时，"
        "统计照旧按原名列出（标「已不在列表」）。马上生效。")),
    # The limit shared by every account without its own quota (`users.quota_gb` NULL falls back to this, see
    # server/quota.py), not only new accounts: after a change, existing accounts read the new value on the next read.
    Setting("storage.quota_gb", "accounts", "账号配额", 100, kind="int", min=1, max=QUOTA_GB_MAX, unit="GB", help=(
        "没单独设过配额的账号能用多少硬盘：它所有任务的文件夹（上传的素材、输出的文件夹和 zip）、还没有任务用到的上传，"
        "加上存在服务器上的模板；缓存不算。满了就不收新上传、不能提交新任务，页面上写明占了多少、上限多少；删掉不要的任务就腾出来。"
        "单个账号的配额在「用户」里按账号改，改了马上生效。")),
    # ---------------------------------------------------------------- 计算调度
    Setting("queue.compute_jobs", "compute", "计算任务", True, kind="bool", help=(
        "关：服务器不再接受新的计算任务，提交都会被拒绝，说明现在只能查看；还没开始的等重新打开，已经开始的算完为止。"
        "看已经算好的结果不受影响，永远能用。"
        "服务器要整个暂停接任务时关掉。马上生效。")),
    Setting("queue.gpu_jobs", "compute", "显卡任务", True, kind="bool", help=(
        "关：所有显卡都不再接新任务，跟一张都没授权一样；正在算的任务算完为止，排队的等这个开关重新打开后自动接着算。"
        "不用显卡的任务和看已经算好的结果不受影响。显卡要检修、机器要给训练腾出来时关掉。马上生效。")),
    Setting("queue.node_minutes", "compute", "单节点超时", 15, kind="int", min=1, max=1440, unit="分钟", help=(
        "一个节点最多算多久：超过就停下，按出错处理（节点标红，用到它的节点跳过，别的分支照常算完，任务显示「部分失败」）。"
        "扩展的节点（跑在计算进程里）直接结束它的进程；核心节点每算完一帧看一次，在下一帧停下。"
        "常有单个节点就要算很久的长镜头时调长。马上生效。")),
    Setting("queue.task_gpus", "compute", "单任务最多占卡数", 2, kind="int", min=1, max=16, unit="张", help=(
        "一个任务同时最多用几张显卡：任务里互不依赖的显卡节点可以同时在几张卡上算，一张卡同时只算一个节点。"
        "显卡只在节点算的时候占着，算完马上给排在最前面、有节点在等显卡的任务。马上生效。")),
    Setting("queue.task_cpus", "compute", "单任务 CPU 节点上限", 2, kind="int", min=1, max=64, unit="个", help=(
        "一个任务同时最多算几个不用显卡的节点（每个节点自己会用好几个核）。防止一个任务接了一堆并行的节点把机器占满。"
        "马上生效。")),
    Setting("queue.cpu_nodes", "compute", "全局 CPU 节点上限", 4, kind="int", min=1, max=256, unit="个",
            auto=Auto(_cpu_nodes, _cpu_nodes_says), help=(
                "整台机器同时最多算几个不用显卡的节点，所有任务加起来。数的是节点，不是核：每个节点自己用好几个核"
                "（逐帧的节点同时算最多 8 帧），所以默认按核数 ÷ 8 算。空出来的名额按排队顺序给第一个有节点在等、"
                "又没超过自己上限的任务；查看器的预览只在没有节点在等时才用空着的名额，从不耽误节点。"
                "机器上同时有训练或渲染、电脑变卡时调小。马上生效。")),
    # The factory default must not be lower than the frame count a bundled card needs; otherwise on a fresh install
    # that card is refused with `B-JOB-TOOMANYFRAMES` and 「提交」 is disabled. The frame count is limited by this
    # setting only (farm/queue.py, at submission) and must not be limited again elsewhere.
    Setting("queue.max_frames", "compute", "帧数上限", 400, kind="int", min=1, max=10000, unit="帧", help=(
        "一次提交的计算最多算多少帧，超过的服务器直接拒绝并说清楚（网页在提交前就拦下，这里挡绕过网页直接提交的）。"
        "防止一个长镜头把显卡和流量占满。只看这一次算的帧范围，和素材本身有多少帧无关：素材 1000 帧、"
        "只算其中 200 帧照样能提交。"
        "人少、镜头长时调大；机器要留给别的事时调小。马上生效。")),
    Setting("queue.reserved_cores", "compute", "保留核心数", 8, kind="int", min=0, max=256, unit="个",
            auto=Auto(_reserved_cores, _reserved_cores_says), help=(
                "算任务永远不许用的核数：留给浏览器、DCC、别的程序。计算进程只在剩下的核上跑（绑核 + 降优先级），"
                "自己带线程参数的项目（COLMAP）也按剩下的核数传。服务器和浏览器常常在同一台机器上："
                "实际情况：一个 COLMAP 任务把 32 个核全吃满 207 秒，整个浏览器连别的标签页一起卡死，"
                "任务结束才恢复。默认按这台机器的核数算：核数 ÷ 4、至少 2。"
                "机器是专用服务器、没人在上面干别的时调到 0（算得最快）；一边算一边要用这台机器时调大。马上生效。")),
    # ---------------------------------------------------------------- 显卡
    Setting("queue.vram_margin_gb", "gpu", "显存余量", 1.0, min=0.5, max=8, unit="GB", help=(
        "挑卡时在节点声明的显存之外再留这么多：节点声明的是模型本身的峰值，新开的计算进程还要一份 CUDA 环境"
        "（实测 4090 约 0.4 GB、5090 约 0.7 GB）。任务老是刚开始就显存不够时调大；显存紧、想让大任务挤得进去时调小，"
        "但调到 0.5 以下很容易失败。马上生效。")),
    Setting("queue.gpu_order", "gpu", "显卡优先顺序", "", kind="text", check=_gpu_order,
            empty="够用的显卡中显存最小的优先", help=(
        "几张空着的显卡都能计算某个节点时，按此顺序优先选用。填写 CUDA 编号，以逗号分隔，例如 1,0；编号按 PCI 总线顺序，"
        "与「显卡」页及 nvidia-smi 显示的编号一致。未列出的显卡排在其后，按原规则（够用的显卡中显存最小的优先）。"
        "仅在几张空着的显卡都放得下节点时起作用：放不下的显卡不会因此被选用，也不改变排队的先后。"
        "把显存大的显卡排在前面时，小节点也会优先占用它，只能在大显卡上计算的节点可能因此多等。立即生效。")),
    # ---------------------------------------------------------------- 常驻模型
    Setting("resident.keep", "resident", "常驻模型", True, kind="bool", help=(
        "开：任务算完后模型留在它的进程里，下一个用同样模型的任务不用重新加载，大模型省下一分钟以上。"
        "关：每个任务算完就结束进程，显存和内存马上还给别的程序，适合机器上同时有训练或渲染。关掉时已经常驻的模型立刻卸载。")),
    Setting("resident.idle_minutes", "resident", "空闲卸载", 30, kind="int", min=1, max=1440, unit="分钟",
            only_if="resident.keep", help=(
                "常驻的模型这么久没被用到就完全卸载，显存和内存都释放。"
                "长一些：隔一阵再算同一个项目也不用重新加载；短一些：显存和内存早点还给别的程序。")),
    Setting("resident.per_gpu", "resident", "每卡常驻数", 3, kind="int", min=0, max=8, unit="个", only_if="resident.keep", help=(
        "一张显卡上最多留几个空闲的常驻进程，正在计算的那个不算；再多的，最久没用的先卸载。"
        "每个进程在显卡上占约 0.5 GB 的 CUDA 环境，模型移到内存里时也一样。几个项目轮流用时调大，显存紧张时调小。")),
    Setting("resident.to_ram", "resident", "让路移到内存", True, kind="bool", only_if="resident.keep", help=(
        "别的项目要用这张显卡时，常驻的模型怎么让出显存。"
        "开：移到内存（移完机器还剩「保留内存」那么多时），下次用时几秒就移回显卡；"
        "关：直接卸载，下次用时重新加载，内存全留给别的程序。")),
    # ---------------------------------------------------------------- 内存
    Setting("memory.keep_free_gb", "memory", "保留内存", 8.0, min=2, max=512, unit="GB", check=_below_memory, help=(
        "机器至少留这么多内存给别的程序：任务等空出这么多才开始（节点声明要更多的按节点的），"
        "常驻模型也只在移过去之后还剩这么多时才移到内存。机器上同时有训练、渲染时调大，防止内存耗尽整台机器死机。")),
    Setting("memory.reuse", "memory", "内存复用", True, kind="bool",
            restart="内存的分配方式在服务启动时定下，运行中改不回系统原来的方式", help=(
                "开：服务进程算完一帧用过的内存先不还给系统，留着给下一帧直接用，逐帧处理整幅画面的节点（图像合成、"
                "遮罩转 Alpha……）快约 20% 到 30%，结果完全一样；代价是服务进程一直多占最多约「复用内存上限」那么多"
                "暂时没在用的内存。关：用完马上还给系统，内存紧张、机器上同时有训练或渲染时关掉。"
                "只管服务进程自己，扩展的计算进程不受影响。")),
    Setting("memory.reuse_gb", "memory", "复用内存上限", 2.0, min=0.5, max=16, unit="GB", only_if="memory.reuse", help=(
        "「内存复用」开着时，服务进程最多留约这么多用过、暂时没在用的内存给下一帧。1080p 的素材 2 GB 就够；"
        "常算 4K 等大画面时调大，内存紧张时调小（调小马上把多留的还给系统）。马上生效。")),
    # ---------------------------------------------------------------- 任务与数据
    # Everything is kept per task (transfer/tasks.py): a task's folder holds its graph, its footage and its outputs,
    # and goes whole 任务保留天数 days after it ended; the cache follows the tasks that reference it (farm/disk.py).
    Setting("tasks.keep_days", "storage", "任务保留天数", 3, kind="int", min=1, max=365, unit="天", help=(
        "每个任务（点「计算」提交的一次计算）从结束时起在服务器上留这么多天，到期整个清理：它的文件夹（节点图、素材、"
        "输出的文件夹和 zip、日志）一起删掉，只被它用到的缓存随后也清掉。选了文件、还没被任何任务用到的素材，"
        "过了这么多天也清掉。用户常隔几天才取结果就调大，硬盘紧张就调小。马上生效。")),
    Setting("tasks.upload_gb", "storage", "单任务上传上限", 1, kind="number", min=0.01, max=10_000, unit="GB", help=(
        "一个任务最多带多少上传的素材（只算上传的素材，不算算出来的结果），超了服务器直接拒绝提交并说清楚；"
        "一个文件本身比这还大时上传就不收。防止一个任务传上来的东西把硬盘撑爆；常要传很大的视频或 EXR 就调大。马上生效。")),
    Setting("paths.data_dir", "storage", "数据位置", "", kind="text", empty="工作文件夹里的 data",
            restart="数据在哪个盘上，服务启动时定下", help=(
                "用户的数据放哪：每个任务的文件夹、上传的素材、算好的缓存都在这里，生产里指向数据盘。"
                "放在一处，任务之间复用素材用的硬链接才都在同一个文件系统上。相对路径从项目文件夹算起；"
                "空：工作文件夹里的 data。换了以后原来的数据留在旧位置，要一起搬过去，不搬的任务和素材就不在了。")),
    Setting("paths.work_dir", "storage", "工作文件夹", "work", kind="text", admin=False,
            restart="缓存、任务记录、显卡授权都在这个文件夹里，服务启动时定下", help=(
                "缓存、上传的素材、待取回的结果、任务记录、显卡授权和日志都在这里；相对路径从项目文件夹算起。"
                "只能在 config/local.toml 里改：换了以后原来的这些都留在旧文件夹，要一起搬过去。")),
    # ---------------------------------------------------------------- 数据库
    Setting("database.backups", "database", "数据库备份份数", 14, kind="int", min=2, max=365, unit="份", help=(
        "数据库（账号、任务记录、使用统计……）自动备份留几份：每天一份（手动备份也算在里面），最旧的先删。"
        "每次升级数据库、更新程序之前的那一份另外留最近 5 份，不会被每天的挤掉。"
        "备份在 work/db/backups/，很小；留多一些，出错时能退回更早的样子。马上生效。")),
    # ---------------------------------------------------------------- 日志
    Setting("logs.max_mb", "logs", "日志大小", 10, kind="int", min=1, max=1000, unit="MB", help=(
        "服务日志 work/logs/lab2shot.log 长到这么大就换一个新文件，旧的改名留着。")),
    Setting("logs.files", "logs", "旧日志份数", 5, kind="int", min=1, max=100, unit="份", help=(
        "留几份换下来的旧日志，更早的删掉；日志一共最多占「日志大小」×（份数 + 1）。要查更早的问题就调大。")),
    Setting("logs.debug", "logs", "详细日志", False, kind="bool", help=(
        "开：日志里另外记下 worker 进程空闲时打印的内容等更细的过程，查问题时用；平时关着，日志小，好读。")),
    # ---------------------------------------------------------------- 视图
    # View proxy size: three fixed tiers; any source size is scaled proportionally to the selected tier. The default is
    # the smallest tier, to be raised when the network is fast enough. A solve scales and compresses the proxy at this
    # tier within the same task (lab2shot/view/proxy.py), and the viewer only loads proxies. The tiers are
    # view/proxy.py TIERS, which provides them as the options. Changes apply at once: proxies of the old tier remain on
    # disk, and proxies of the new tier are generated and cached on first view. What 「输出」 writes is not affected.
    Setting("view.proxy_px", "view", "视图代理尺寸", "512", kind="choice", choices="proxy_tiers", help=(
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
                "三维整段缓存（点云等逐帧数据的块）也算在这份额度里。"
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
    Setting("server.names", "network", "对外域名", "", kind="text", check=_names,
            empty="只用本机的名字和地址", restart="证书只在服务启动时做（认不认这个域名马上生效）", help=(
                "别人访问用的域名或 IP，用空格隔开，比如内网穿透给的 abc.example.com、内网 DNS 给这台机器起的别名。"
                "服务器只回答用这台机器的名字、IP 地址或这里写了的域名打开的请求（别的网站把自己的域名指到这里的「DNS 重绑定」"
                "就进不来）；经「可信代理」转发来的请求由代理按域名分发，不看这一项。"
                "开了 HTTPS、穿透工具只转发端口不管加密时，证书也要写上这个名字，浏览器才认。"
                "空：只认这台机器自己的名字和地址。")),
    Setting("server.trusted_proxies", "network", "可信代理", "", kind="text", check=_proxies,
            empty="不信任何代理：一律按连接的地址算", help=(
                "在这台服务器前面、替用户转发请求的反向代理的地址，用空格隔开，可以写网段，如 127.0.0.1 或 10.0.0.0/8。"
                "只有从这些地址连来的请求，才按它在 X-Forwarded-For / X-Real-IP 里转告的地址算用户的 IP：注册按 IP 和网段的限制、"
                "输错邀请码和密码按来源的计数、登录记录里的 IP 都按它；它转告的 X-Forwarded-Proto（是不是 HTTPS）和 "
                "X-Forwarded-Host（用户打开的域名）也只从这些地址认。别的来源发来的这些头一律不看（谁都能随便写）。\n"
                "两种接法：① 云服务器上的 nginx / Caddy / frp https2http 用正式证书解开 HTTPS，再用 HTTP 转发到这里"
                "（这里的 HTTPS 关掉）：把转发过来的那个地址填在这里（frpc 在本机就填 127.0.0.1），就看得到每个用户的真实 IP。"
                "② frp 只转发 TCP 连接到这里自己的 HTTPS 端口：这里看到的来源全是本机地址，没有头可读，这一项留空；"
                "按 IP 的限制这时自动不算（不然会把所有人当成一个人一起挡住），全站注册上限、邀请码和工作量证明照常；"
                "这时每个请求在这里看都像是从本机来的：本机命令行的机器令牌照样要令牌本身才认，不靠地址，"
                "但没登录的请求（登录页、注册页）只能共用一个请求频率，一个人发得多，别人也会被说「太频繁」。"
                "对外公开注册时请用第 ① 种（HTTPS 在云服务器上解开）：只有它看得到每个人的真实 IP，按 IP 的限制才起作用。"
                "现在是哪一种，看这一组下面的「用户的 IP」。马上生效。")),
    # ---------------------------------------------------------------- 下载
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
    Setting("install.retries", "install", "重试次数", 5, kind="int", min=1, max=20, unit="次", help=(
        "安装扩展包时，下载、克隆代码、装包遇到网络问题（断开、超时、服务器忙）时每个来源最多试几次。"
        "每次之间等的时间翻倍（1、2、4、8 秒……），不超过「最长等待」。试完还不行就换下一个来源（镜像）。")),
    Setting("install.backoff_max", "install", "最长等待", 60, kind="int", min=5, max=600, unit="秒", help=(
        "两次重试之间最多等多久。网络时断时续时调大一点，能少失败几次。")),
    # ---------------------------------------------------------------- 编译与色彩
    Setting("build.archs", "env", "编译目标架构", ["sm_89", "sm_120"], kind="list", check=_archs, help=(
        "安装扩展包时，CUDA 代码（pytorch3d、DPVO、SAM 2 的 _C 等）为哪几代架构编译，一条一个："
        "sm_75 Turing、sm_80 Ampere 数据中心、sm_86 Ampere、sm_89 Ada Lovelace、sm_90 Hopper、"
        "sm_100 Blackwell 数据中心、sm_120 Blackwell。编出来的代码只在这里列出的架构上能跑，和编译这台机器插的是什么卡"
        "无关；扩展包自己声明了只支持哪几代时，只编两边都有的。CUDA 工具包要支持所选的每一代"
        "（sm_120 要 CUDA 12.8 以上；「检查编译工具」会逐个核对）。改了以后新装的扩展包按新的来；"
        "已经装好的不会自动重编，要重装才按新的架构编译。")),
    Setting("build.cuda_home", "env", "CUDA 路径", "", kind="text", check=_cuda, empty="用 $CUDA_HOME", help=(
        "安装扩展包时编译 CUDA 代码（detectron2 等）用的 CUDA 工具包文件夹，如 /usr/local/cuda-12.8。"
        "空：用 $CUDA_HOME，没有就用 /usr/local/cuda。下次安装扩展包时生效。")),
    Setting("build.cc", "env", "C 编译器", "", kind="text", check=_compiler, empty="系统默认", help=(
        "安装扩展包时用的 C 编译器，如 gcc-14。CUDA 对 gcc 的版本有上限，系统默认的 gcc 太新时在这里指定旧一点的。"
        "空：系统默认。下次安装扩展包时生效。")),
    Setting("build.cxx", "env", "C++ 编译器", "", kind="text", check=_compiler, empty="系统默认", help=(
        "安装扩展包时用的 C++ 编译器，如 g++-14，和 C 编译器同一个版本。空：系统默认。下次安装扩展包时生效。")),
    Setting("color.config", "env", "OCIO 配置", "", kind="text", check=_ocio, empty="用 $OCIO 或内置 ACES",
            restart="色彩配置在服务启动时读入，网页里的色彩空间列表也跟着它", help=(
                "服务器上 OCIO 配置文件 config.ocio 的路径，或 ocio:// 开头的 OCIO 内置配置。"
                "空：用环境变量 $OCIO，没有就用 OCIO 内置的最新 ACES studio 配置。换配置会改变读图时默认的色彩空间。")),
)}


# ------------------------------------------------------------------ the values


def config_file() -> Path:
    return Path(os.environ.get("LAB2SHOT_CONFIG") or CONFIG_DIR / "local.toml")


def _raw(file: Path) -> dict:
    if not file.exists():
        return {}
    try:
        return tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(Msg("E-CONFIG-TOML", file=str(file), detail=str(exc))) from exc


def _read(file: Path) -> dict[str, object]:
    """The settings a file sets, checked against the schema (an unknown name or a wrong value is an error: a typo
    never goes unnoticed). Only kind and range: whether a path exists is asked when the admin changes it."""
    data = _raw(file)
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
# 本机设置（不纳入版本控制）。由管理后台「设置」一组的各页与 ./setup.sh 的「设置」写入，也可手工编辑；修改后须重启服务。
# 仅记录与默认值不同的项；各项的含义与默认值见管理后台「设置」一组的各页。
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

    status = 500


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
    def data_dir(self) -> Path:
        """Where every user's data is kept (「数据位置」: caches, uploads, task folders), one place so every hard link
        between them is on one file system: the configured path, or data in the work folder when empty."""
        return self._folder("paths.data_dir", "") or self.work_dir / "data"

    @property
    def folders(self) -> list[Path]:
        """Every folder the settings place this server's things in, one per 「paths.」 setting (each read through its
        property of the same name, so its default applies): where they are is never shown to someone who is not the
        administrator (server/access.py scrub)."""
        return [getattr(self, key.removeprefix("paths.")) for key in SCHEMA if key.startswith("paths.")]

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
        is checked further (does the path exist, is the port free) before anything is written. Saving only keeps the
        values: the server applies what it applies to its own process (process.apply_changed)."""
        with self._lock:
            self._reload_if_changed()
            new = dict(self.saved)
            problems = {}
            for key, raw in changes.items():
                s = SCHEMA.get(key)
                if s is None or not s.admin:
                    problems[key] = Msg("E-SETTINGS-NOTHERE", key=key) if s is None else Msg("E-SETTINGS-FILEONLY", setting=s.label, file=self.file.name)
                    continue
                if not s.provided():  # a bug: whoever saves settings imports the layers that provide their options
                    raise RuntimeError(f"setting {key}: options {s.choices!r} not provided (config.provide_choices)")
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
            "pages": [{"id": page, "label": p.label, "tip": p.tip, "groups": [{"id": g, "label": label} for g, label in p.groups.items()]}
                      for page, p in PAGES.items()],
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

    status = 400

    def __init__(self, problems: dict[str, Msg]) -> None:
        super().__init__(Msg("E-SETTINGS-INVALID", problems=list(problems.values())))
        self.problems = problems


@cache
def settings() -> Settings:
    return Settings(config_file())
