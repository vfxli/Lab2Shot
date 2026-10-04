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

from . import i18n
from .errors import FieldErrors, Invalid, MessageError, message_of
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
    its group. Its words are in the catalogue (lab2shot/i18n/<lang>/settings.toml), read in the language now:
    settings_page.<id>.label, .tip (what the page holds, for the side list's and the menu's hover text) and
    .group.<group> (each group's heading)."""

    id: str
    group_ids: tuple[str, ...]  # its groups, in the page's order

    @property
    def label(self) -> str:
        return i18n.t(f"settings_page.{self.id}.label")

    @property
    def tip(self) -> str:
        return i18n.t(f"settings_page.{self.id}.tip")

    @property
    def groups(self) -> dict[str, str]:
        """group id -> its heading, in the page's order."""
        return {g: i18n.t(f"settings_page.{self.id}.group.{g}") for g in self.group_ids}


# The settings pages, in their order. The admin page, the admin settings API and the setup menu all read this one
# definition: which pages there are, what they are called, and which groups (so which settings) each one holds.
PAGES: dict[str, Page] = {
    "register": Page("register", ("register",)),
    "accounts": Page("accounts", ("accounts", "feedback", "plugins")),
    "compute": Page("compute", ("compute", "gpu", "resident", "memory")),
    "storage": Page("storage", ("storage", "database", "logs", "view")),
    "network": Page("network", ("network", "install", "env")),
}

# every group -> the page it is on
PAGE_OF: dict[str, str] = {g: page for page, p in PAGES.items() for g in p.group_ids}


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


# a comma typed with a Chinese input method reads like ",": written as its code point (no CJK literal in code)
FULLWIDTH_COMMA = chr(0xFF0C)


def gpu_order(value: str) -> list[int]:
    """The CUDA numbers listed in 显卡优先顺序, in order, each once (separated by commas or spaces)."""
    out: list[int] = []
    for part in value.replace(FULLWIDTH_COMMA, ",").replace(",", " ").split():
        if re.fullmatch(r"[0-9]+", part) and int(part) not in out:
            out.append(int(part))
    return out


def _gpu_order(value: str) -> Msg | str:
    bad = next((p for p in value.replace(FULLWIDTH_COMMA, ",").replace(",", " ").split() if not re.fullmatch(r"[0-9]+", p)), None)
    return Msg("E-SETTINGS-GPUORDER", item=bad) if bad is not None else ""


def _names(value: str) -> Msg | str:
    """Host names or addresses, separated by spaces or commas: each one a certificate can name."""
    import ipaddress
    import re

    for name in value.replace(FULLWIDTH_COMMA, ",").replace(",", " ").split():
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
    for item in value.replace(FULLWIDTH_COMMA, ",").replace(",", " ").split():
        try:
            out.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            pass
    return out


def _proxies(value: str) -> Msg | str:
    import ipaddress

    for item in value.replace(FULLWIDTH_COMMA, ",").replace(",", " ").split():
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
    return i18n.t("settings.auto.cpu_nodes", cores=cpu_cores(), nodes=_cpu_nodes())


def _reserved_cores() -> int:
    """Default number of reserved cores: cores / 4, at least 2 (at least 1 on machines with few cores).

    The server and the browser commonly run on the same machine, so a cook must not occupy the whole machine: a single
    COLMAP task saturates all cores for minutes, and the browser on that machine, including other tabs, stops
    responding until the task ends."""
    return max(1, min(cpu_cores() - 1, max(2, cpu_cores() // 4)))


def _reserved_cores_says() -> str:
    return i18n.t("settings.auto.reserved_cores", cores=cpu_cores(), reserved=_reserved_cores())


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
LIST_MARKS = str.maketrans("", "", "-_/·()" + chr(0xFF08) + chr(0xFF09))  # with the full-width brackets


# ------------------------------------------------------------------ the schema


@dataclass(frozen=True)
class Setting:
    """One setting. Its words are in the catalogue (lab2shot/i18n/<lang>/settings.toml), read in the language now:
    setting.<key>.label (one short line, no brackets), .note (shown under it, always visible: what it does, what a
    change leads to),
    .restart (why a change takes effect only after the server restarts), .empty (what leaving it empty means),
    .option.<value> (a choice's label); a unit's words are settings.unit.<unit id>."""

    key: str  # <section>.<name>, as in config/local.toml
    group: str  # the card it is on (a group of PAGES)
    default: object
    kind: str = "number"  # number / int / bool / choice / text / list (of short names) / multi (any of the options)
    min: float | None = None
    max: float | None = None
    unit_id: str = ""  # its unit (settings.unit.<id>); "": none
    option_values: tuple[str, ...] = ()  # choice, multi: its own options' values (their labels: .option.<value>)
    choices: str = ""  # the options, by the name another layer provides them under (provide_choices)
    restarts: bool = False  # a change takes effect only after the server restarts (why: .restart)
    admin: bool = True  # False: shown on the admin page, changed in config/local.toml only
    only_if: str = ""  # a switch this one needs on to matter
    empties: bool = False  # text, multi: may be left empty (what that means: .empty)
    check: Callable | None = None  # a changed value's problem (a Msg; "" none), beyond kind and range
    auto: Auto | None = None  # the default is worked out from this machine (cores, cards), not a fixed number
    # list: its items are ids with words of their own (<item_words>.<id> in the catalogue, e.g. department.<id>): the
    # page shows those words for them; an item an administrator typed has none and shows as written
    item_words: str = ""

    def _word(self, part: str) -> str:
        return i18n.t(f"setting.{self.key}.{part}")

    @property
    def label(self) -> str:
        # kept by its key (i18n.Word, a str in the language now): a message naming the setting reads in its reader's
        return i18n.Word(f"setting.{self.key}.label")

    def item_labels(self) -> dict[str, str]:
        """A list's items that are ids with words of their own (item_words), in the language now: id -> its words."""
        if not self.item_words:
            return {}
        return {k: w for k in i18n.words(i18n.current()) if k.startswith(f"{self.item_words}.")
                for k, w in [(k.removeprefix(f"{self.item_words}."), i18n.t(k))] if "." not in k}

    @property
    def note(self) -> str:
        return self._word("note")

    @property
    def restart(self) -> str:
        """Why a change takes effect only after the server restarts ("": at once)."""
        return self._word("restart") if self.restarts else ""

    @property
    def empty(self) -> str:
        """What leaving it empty means ("": it may not be empty)."""
        return self._word("empty") if self.empties else ""

    @property
    def unit(self) -> str:
        """Its unit in words ("": none)."""
        return i18n.t(f"settings.unit.{self.unit_id}") if self.unit_id else ""

    @property
    def options(self) -> tuple[tuple[str, str], ...]:
        """Its own options: (value, label)."""
        if self.key == "view.local_px":
            return tuple((v, i18n.t("settings.pixels", px=v)) for v in self.option_values)
        return tuple((v, self._word(f"option.{v}")) for v in self.option_values)

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
        """A value of this setting in words, in the language now, as the admin page shows it (a switch as on / off, a choice as its label,
        a number with its unit, nothing as what empty means). Said here, beside the setting, so the page never
        writes a second version of it."""
        if self.kind == "multi":
            return i18n.separator().join(label for v, label in self.offered() if v in value) or i18n.t("settings.empty_is", empty=self.empty)
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        if self.kind == "bool":
            return i18n.t("settings.on") if value else i18n.t("settings.off")
        if self.kind == "choice":
            return next((label for v, label in self.offered() if v == value), str(value))
        if value == "":
            return i18n.t("settings.empty_is", empty=self.empty)
        return f"{value}{f' {self.unit}' if self.unit else ''}"

    def describe(self) -> dict:
        return {"default_text": self.says(self.default_now),
                "key": self.key, "group": self.group, "label": self.label, "note": self.note, "default": self.default_now,
                "auto": self.auto.says() if self.auto else "",
                "kind": self.kind, "min": self.min, "max": self.max, "unit": self.unit,
                "options": [{"value": v, "label": label} for v, label in self.offered()], "restart": bool(self.restart),
                "item_labels": self.item_labels(),
                "why": self.restart, "admin": self.admin, "only_if": self.only_if, "empty": self.empty}


# the factory list of 环节 (people.departments): ids, their words in the catalogue (department.<id>, shown through
# accounts.department_label); an administrator edits the list (one added is user data, shown as written; each account
# keeps the one it was given, the usage statistics group by it)
DEPARTMENTS: list[str] = json.loads((Path(__file__).with_name("departments.json")).read_text(encoding="utf-8"))

QUOTA_GB_MAX = 10_000  # the largest disk quota an account may have, GB: the setting's and one set per account (server/quota.py)

SCHEMA: dict[str, Setting] = {s.key: s for s in (
    # ---------------------------------------------------------------- 注册 (lab2shot/site/registration.py)
    Setting("register.open", "register", False, kind="bool"),
    Setting("register.invite", "register", True, kind="bool", only_if="register.open"),
    Setting("register.days", "register", 30, kind="int", min=1, max=3650, unit_id="days", only_if="register.open"),
    Setting("register.quota_gb", "register", 20, kind="int", min=1, max=10_000, unit_id="GB", only_if="register.open"),
    Setting("register.tags", "register", ["commercial", "noncommercial", "research", "registration"], kind="multi", choices="account_tags", empties=True, only_if="register.open"),
    Setting("register.per_hour", "register", 20, kind="int", min=1, max=100_000, unit_id="count", only_if="register.open"),
    Setting("register.per_day", "register", 100, kind="int", min=1, max=1_000_000, unit_id="count", only_if="register.open"),
    # ---------------------------------------------------------------- 账号
    Setting("people.departments", "accounts", DEPARTMENTS, kind="list", item_words="department"),
    # The limit shared by every account without its own quota (`users.quota_gb` NULL falls back to this, see
    # server/quota.py), not only new accounts: after a change, existing accounts read the new value on the next read.
    Setting("storage.quota_gb", "accounts", 20, kind="int", min=1, max=QUOTA_GB_MAX, unit_id="GB"),
    # ---------------------------------------------------------------- 有效反馈奖励 (lab2shot/site/feedback.py rate)
    Setting("feedback.reward_count", "feedback", 1, kind="int", min=1, max=1000, unit_id="items"),
    Setting("feedback.reward_days", "feedback", 7, kind="int", min=0, max=3650, unit_id="days"),
    # ---------------------------------------------------------------- DCC 插件 (server/plugins.py, server/available.py)
    # off: the top bar's 「DCC 插件」 and its downloads are the administrators' only (plugins still being tested); a
    # plugin already installed still connects, whatever this says
    Setting("plugins.download", "plugins", False, kind="bool"),
    # ---------------------------------------------------------------- 计算调度
    Setting("queue.compute_jobs", "compute", True, kind="bool"),
    Setting("queue.gpu_jobs", "compute", True, kind="bool"),
    Setting("queue.node_minutes", "compute", 15, kind="int", min=1, max=1440, unit_id="minutes"),
    Setting("queue.task_gpus", "compute", 2, kind="int", min=1, max=16, unit_id="cards"),
    Setting("queue.task_cpus", "compute", 2, kind="int", min=1, max=64, unit_id="count"),
    Setting("queue.cpu_nodes", "compute", 4, kind="int", min=1, max=256, unit_id="count", auto=Auto(_cpu_nodes, _cpu_nodes_says)),
    # The factory default must not be lower than the frame count a bundled card needs; otherwise on a fresh install
    # that card is refused with `B-JOB-TOOMANYFRAMES` and 「提交」 is disabled. The frame count is limited by this
    # setting only (farm/queue.py, at submission) and must not be limited again elsewhere.
    Setting("queue.max_frames", "compute", 400, kind="int", min=1, max=10000, unit_id="frames"),
    Setting("queue.reserved_cores", "compute", 8, kind="int", min=0, max=256, unit_id="count", auto=Auto(_reserved_cores, _reserved_cores_says)),
    # ---------------------------------------------------------------- 显卡
    Setting("queue.vram_margin_gb", "gpu", 1.0, min=0.5, max=8, unit_id="GB"),
    Setting("queue.gpu_order", "gpu", "", kind="text", check=_gpu_order, empties=True),
    # ---------------------------------------------------------------- 常驻模型
    Setting("resident.keep", "resident", True, kind="bool"),
    Setting("resident.idle_minutes", "resident", 30, kind="int", min=1, max=1440, unit_id="minutes", only_if="resident.keep"),
    # a process whose models moved to RAM still holds its CUDA context on the card and its copy in RAM (engine/resident.py tidy)
    Setting("resident.ram_idle_minutes", "resident", 5, kind="int", min=1, max=1440, unit_id="minutes", only_if="resident.keep"),
    # twice the factory 显存余量 (queue.vram_margin_gb): room for another program's or another server's job to start
    Setting("resident.release_below_gb", "resident", 2.0, min=0, max=64, unit_id="GB", only_if="resident.keep"),
    Setting("resident.per_gpu", "resident", 3, kind="int", min=0, max=8, unit_id="count", only_if="resident.keep"),
    Setting("resident.to_ram", "resident", True, kind="bool", only_if="resident.keep"),
    # the RAM idle kept processes hold together (VmRSS), a share of the machine's: engine/resident.py bound_ram
    Setting("resident.ram_pct", "resident", 30, kind="int", min=5, max=90, unit_id="percent", only_if="resident.keep"),
    # ---------------------------------------------------------------- 内存
    Setting("memory.keep_free_gb", "memory", 8.0, min=2, max=512, unit_id="GB", check=_below_memory),
    Setting("memory.reuse", "memory", True, kind="bool", restarts=True),
    Setting("memory.reuse_gb", "memory", 2.0, min=0.5, max=16, unit_id="GB", only_if="memory.reuse"),
    # ---------------------------------------------------------------- 任务与数据
    # Everything is kept per task (transfer/tasks.py): a task's folder holds its graph, its footage and its outputs,
    # and goes whole 任务保留天数 days after it ended; the cache follows the tasks that reference it (farm/disk.py).
    Setting("tasks.keep_days", "storage", 3, kind="int", min=1, max=365, unit_id="days"),
    Setting("tasks.upload_gb", "storage", 1, kind="number", min=0.01, max=10_000, unit_id="GB"),
    Setting("storage.pause_free_pct", "storage", 10, kind="int", min=1, max=50, unit_id="percent"),
    Setting("tasks.keep_most", "storage", 1000, kind="int", min=10, max=100_000, unit_id="items"),
    Setting("paths.data_dir", "storage", "", kind="text", empties=True, restarts=True),
    Setting("paths.work_dir", "storage", "work", kind="text", admin=False, restarts=True),
    # ---------------------------------------------------------------- 数据库
    Setting("database.backups", "database", 14, kind="int", min=2, max=365, unit_id="copies"),
    # ---------------------------------------------------------------- 日志
    Setting("logs.max_mb", "logs", 10, kind="int", min=1, max=1000, unit_id="MB"),
    Setting("logs.files", "logs", 5, kind="int", min=1, max=100, unit_id="copies"),
    Setting("logs.debug", "logs", False, kind="bool"),
    # ---------------------------------------------------------------- 视图
    # View proxy size: three fixed tiers; any source size is scaled proportionally to the selected tier. The default is
    # the smallest tier, to be raised when the network is fast enough. A solve scales and compresses the proxy at this
    # tier within the same task (lab2shot/view/proxy.py), and the viewer only loads proxies. The tiers are
    # view/proxy.py TIERS, which provides them as the options. Changes apply at once: proxies of the old tier remain on
    # disk, and proxies of the new tier are generated and cached on first view. What 「输出」 writes is not affected.
    Setting("view.proxy_px", "view", "512", kind="choice", choices="proxy_tiers"),
    # Local proxy: material on the user's own machine is not shown losslessly either. After a sequence is chosen, the
    # browser scales every layer of every frame to this tier in the background, compresses it and stores it in the
    # browser's private file system (on the user's disk); the viewer draws from this proxy and no data passes through
    # the server. Quality is below a lossless Nuke preview and above the server proxy's default tier. The setting
    # applies on the client side but is controlled by the administrator. The two values below are only passed to the
    # browser (via /api/server); the server itself does not act on them.
    Setting("view.local_px", "view", "1024", kind="choice", option_values=tuple(str(px) for px in _LOCAL_TIERS)),
    Setting("view.local_cache_gb", "view", 10, kind="int", min=1, max=500, unit_id="GB"),
    # Point cloud limit: 3D has no compressed-preview switch; a cloud above the limit is always decimated. There is only
    # this limit, no lossless / compressed tiers. The default is small (5 MB) for the same reason the 2D proxy defaults
    # to the smallest tier: start conservatively and raise it when the network is fast enough. The size counted is what
    # is actually sent to the browser (about 15 bytes per point, about 7 bytes per depth cell); attributes the viewer
    # does not use are not sent and not counted.
    Setting("view.points_max_mb", "view", 5, kind="int", min=1, max=4096, unit_id="MB"),
    # Gaussian display limit: 3D 高斯 has no compressed preview either. The viewer draws one frame's splats through a
    # CPU depth sort on every camera move, so the limit is a splat count (not bytes): 300 000 sorts in ~50 ms and stays
    # interactive (a 1.99 M-splat cloud would sort in ~450 ms, about 2 fps). Over the limit every N-th splat is kept,
    # exactly like the point cloud's 「点云上限」; only the viewed copy is thinned — computation, cache and delivered
    # files keep every splat.
    Setting("view.gaussian_max", "view", 300_000, kind="int", min=10_000, max=10_000_000, unit_id="count"),
    # ---------------------------------------------------------------- 网络
    Setting("server.host", "network", "127.0.0.1", kind="choice", option_values=('127.0.0.1', '0.0.0.0'), restarts=True),
    Setting("server.port", "network", 8765, kind="int", min=1024, max=65535, check=port_problem, restarts=True),
    Setting("server.https", "network", False, kind="bool", restarts=True),
    Setting("server.names", "network", "", kind="text", check=_names, empties=True, restarts=True),
    Setting("server.trusted_proxies", "network", "", kind="text", check=_proxies, empties=True),
    # ---------------------------------------------------------------- 下载
    Setting("install.mirror_hf", "install", "https://hf-mirror.com", kind="text", check=_mirrors, empties=True),
    Setting("install.mirror_pypi", "install", "https://pypi.tuna.tsinghua.edu.cn/simple", kind="text", check=_mirrors, empties=True),
    Setting("install.mirror_github", "install", "", kind="text", check=_mirrors, empties=True),
    Setting("install.retries", "install", 5, kind="int", min=1, max=20, unit_id="times"),
    Setting("install.backoff_max", "install", 60, kind="int", min=5, max=600, unit_id="seconds"),
    # ---------------------------------------------------------------- 编译与色彩
    Setting("build.archs", "env", ["sm_89", "sm_120"], kind="list", check=_archs),
    Setting("build.cuda_home", "env", "", kind="text", check=_cuda, empties=True),
    Setting("build.cc", "env", "", kind="text", check=_compiler, empties=True),
    Setting("build.cxx", "env", "", kind="text", check=_compiler, empties=True),
    Setting("color.config", "env", "", kind="text", check=_ocio, empties=True, restarts=True),
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
# This machine's settings (not under version control). Written by the admin page's Settings pages and the Settings of
# ./setup.sh, and may be edited by hand; a change takes effect after the server restarts.
# Only the values that differ from the defaults are kept; what each one means and its default: the admin page's Settings.
# Saving the settings rewrites this whole file in the settings' order: comments added by hand are not kept."""


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
        lines += [f"# {i18n.t(f'setting.{s.key}.label', in_lang=i18n.LOG_LANG)}", f"{name} = {_toml(values[s.key])}"]  # the back end's language
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
        # key -> (value, what set it) for this run; what set it is a mark, said in words by overridden_by: a command-line
        # option as itself ("--port"), an environment variable by its name ("LAB2SHOT_WORK_DIR"), "server" the server
        self.command: dict[str, tuple[object, str]] = {}
        if os.environ.get("LAB2SHOT_WORK_DIR"):
            self.command["paths.work_dir"] = (os.environ["LAB2SHOT_WORK_DIR"], "LAB2SHOT_WORK_DIR")
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

    def overridden_by(self, key: str) -> str:
        """What overrides the file for this run, in words in the language now ("": nothing)."""
        if key not in self.command:
            return ""
        what = self.command[key][1]
        if what.startswith("--"):
            return i18n.t("settings.overridden.option", option=what)
        if what == "server":
            return i18n.t("settings.overridden.server")
        if what.isupper():
            return i18n.t("settings.overridden.env", name=what)
        return what

    def overridden_kind(self, key: str) -> str:
        """What kind of thing overrides the file for this run: "option" (the command line), "env" (the environment),
        "server" (the server itself), "" nothing."""
        if key not in self.command:
            return ""
        what = self.command[key][1]
        return "option" if what.startswith("--") else "server" if what == "server" else "env" if what.isupper() else "option"

    def run_with(self, key: str, value: object, what: str) -> None:
        """`lab2shot ui --port ...`: this run goes by this value (`what`: the option itself, "--port"). The
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
            return {"source": i18n.t("settings.source.changed"),
                    "source_tip": i18n.t("settings.source.changed_tip", default=s.says(s.default_now))}
        if s.auto:
            return {"source": i18n.t("settings.source.auto"), "source_tip": i18n.t("settings.source.auto_tip", how=s.auto.says())}
        return {"source": "", "source_tip": ""}

    def describe(self) -> dict:
        """For the admin page: every setting with its value, what this process goes by, what overrides the file."""
        return {
            "file": str(self.file),
            "pages": [{"id": page, "label": p.label, "tip": p.tip, "groups": [{"id": g, "label": label} for g, label in p.groups.items()]}
                      for page, p in PAGES.items()],
            # every value that is shown in words is put into words here, beside the setting (Setting.says), so the
            # page never writes a second version of on / off, a choice's label, a unit or what empty means
            "settings": [{**s.describe(), "value": self.file_value(k), "running": self[k],
                          "value_text": s.says(self.file_value(k)), "running_text": s.says(self[k]),
                          "overridden": self.overridden_by(k), "overridden_kind": self.overridden_kind(k),
                          **self._source(k, s)} for k, s in SCHEMA.items()],
            "pending": self.pending(),
        }


class InvalidSettings(FieldErrors):
    """Changes that can't be kept: key -> why (a message each); 403 when it is the caller's rights that refuse them."""

    def __init__(self, problems: dict[str, Msg], status: int | None = None) -> None:
        super().__init__(Msg("E-SETTINGS-INVALID", problems=list(problems.values())), problems, status)


@cache
def settings() -> Settings:
    return Settings(config_file())
