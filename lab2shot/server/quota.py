"""单账号磁盘配额：一个账号在服务器上实际占的全部字节有一个上限：配额 = 该账号所有任务的文件夹（上传的素材 + 输出的
文件夹和 zip）+ 其余的上传 + 它自己的计算缓存 + 保存在服务器上的模板。同一份字节（硬链接）只算一次（farm/space.py 按 inode 量）。

上限：`users.quota_gb`（该账号自身的），未设置时使用设置中的 `storage.quota_gb`（「账号配额」，默认 20 GB）。0 表示不限。
占用：分为四部分，每部分说明其内容及何时释放；配额已满时，这些数值即使用者看到的提示（已用、上限、主要占在哪、怎么腾）。

    任务      该账号每个任务的文件夹（transfer/tasks.py：节点图、硬链接进来的素材、输出、日志）。
              任务结束「任务保留天数」后整个删除，删掉任务也立刻腾出来
    缓存      该账号自己的计算缓存（data/store.py data/cache/<账号 id>/）。只被删掉的任务用到的随任务清掉
              （farm/disk.py），还有任务用到的留着
    上传      该账号的上传文件夹（transfer/uploads.py home_of）里还没有链接进任务的：选了文件时申报的文件头、传了
              一半的分段、传完还没有任务用到的素材。没有任务用到的，「任务保留天数」后清理
    模板      该账号保存在服务器上的节点图（回收站中的不计入）

满了：新的提交和上传拒绝（`refuse_if_full`、`room_for`），正在算的照常算完；不自动删用户的东西。释放空间只有一个操作：
删除任务（单条、一组、全部，或「腾出空间」一次删最旧的几条：`plan`、`free`）。它的文件夹整个删除，只被它用到的缓存随之清掉。
占到 80%（WARN_AT）时 `gate` 的 stage 变成 1，满了是 2：顶栏按阶段提醒一次（webui/src/editor/Chrome.tsx）。

量一次占用要把账号的任务文件夹、上传和缓存逐个 stat 一遍（缓存里一个序列就是成千上万个文件；已结束的任务和完整的缓存项
不会再变，量过一次就记住：farm/space.py）。所以账号的占用是一个数（`_latest`）：最近一次量的结果（`_measured`），加上这以后
本进程收进来的上传字节（`wrote`：每个分段只加自己的字节，server/transfer.py 收字节的那一处）。轮询（队列窗口和每个页面的
/api/load，每 1.5–30 秒；`gate`）读这个数，从不在请求里量：结果超过 MEASURED_S 秒、或这个账号从没量过时在后台线程里量一次，
这一次仍用上一次的数（从没量过时是 0，量完下一次轮询就对了）。写入之前的额度检查（`room_for`、`refuse_if_full`）也读这个数，
只在它说放不下时才当场量一次（占用可能已经因删任务变小；从没量过时当场量）。「我的占用」打开时、删掉任务之后（`usage`）当场量，
并更新这份结果：删掉任务后的下一次轮询就看到空间释放。"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from .. import i18n
from ..config import QUOTA_GB_MAX, settings
from ..database import db
from ..errors import TooLarge
from ..messages import Msg
from .words import Word

GB = 1 << 30
WARN_AT = 0.8  # 占到上限的这么多时顶栏提醒一次（stage 1）


def _gb(value) -> float:
    """A quota as kept, in GB. One past what can be set (config QUOTA_GB_MAX; an infinite one, or one so large its bytes
    are no number: written before requests' numbers were bounded, routes.Gigabytes) reads as 0, 不限, which is what it
    meant."""
    gb = float(value)
    return gb if math.isfinite(gb) and gb <= QUOTA_GB_MAX else 0.0


def limit_of(user_id: int) -> int:
    """该账号的上限，单位为字节（0：不限）。"""
    from .. import accounts

    own = accounts.quota_gb(user_id)
    return int(_gb(own if own is not None else settings()["storage.quota_gb"]) * GB)


def set_limit(user_id: int, gb: float | None) -> None:
    """按账号修改配额（None：使用设置中的默认值）。"""
    from .. import accounts

    accounts.set_quota_gb(user_id, gb)


def own_gb(user_id: int) -> float | None:
    from .. import accounts

    own = accounts.quota_gb(user_id)
    return None if own is None else _gb(own)


def stage_of(total: int, limit: int) -> int:
    """0：不到 80%（或不限）；1：到了 80%；2：满了。顶栏按阶段提醒，每个阶段一次。"""
    if not limit:
        return 0
    return 2 if total >= limit else 1 if total >= WARN_AT * limit else 0


# ------------------------------------------------------------------ 占用


def _areas_of(user_id: int, fp) -> list[dict]:
    """四部分占用（farm/space.py 的一次测量 + 模板），每部分包括：大小、内容说明、何时释放。

    仅用于报告数值，不作为清理入口：前台「我的占用」只显示总数和每个已结束任务的占用，这四部分供后台「用户」页
    （webui/src/admin/UserQuota.tsx）和配额已满的提示使用，说明空间用在何处。"""
    from .. import accounts
    from ..site import library

    # their words (label, note) are added where they are shown (_worded): these figures are kept and measured off requests
    return [{"id": "tasks", "bytes": fp.areas["tasks"]}, {"id": "cache", "bytes": fp.areas["cache"]},
            {"id": "uploads", "bytes": fp.areas["uploads"]},
            {"id": "templates", "bytes": library.user_bytes(accounts.get(user_id).username)}]


def _worded(found: list[dict]) -> list[dict]:
    """The four areas with their words in the language now: label (server.quota.area.<id>.label) and note."""
    from ..transfer import tasks

    days = tasks.keep_days()
    return [{**a, "label": i18n.t(f"server.quota.area.{a['id']}.label"), "note": i18n.t(f"server.quota.area.{a['id']}.note", days=days)}
            for a in found]


@dataclass
class Measured:
    at: float
    total: int
    before: int  # what `wrote` had counted when it was measured
    found: list[dict]  # the four areas
    fp: object  # farm/space.py Footprint


MEASURED_S = 10.0  # how old the figures a poll gets may be
_measured: dict[int, Measured] = {}
_written: dict[int, int] = {}  # account -> bytes of uploads this process took in (wrote), ever growing
_measuring: set[int] = set()  # accounts measured again in the background now
_measured_lock = threading.Lock()


def wrote(user_id: int, count: int) -> None:
    """`count` bytes of an upload of the account's have just reached the disk: its figure grows by them (_latest)."""
    with _measured_lock:
        _written[user_id] = _written.get(user_id, 0) + count


def _measure(user_id: int) -> Measured:
    """Measure the account now (farm/space.py), kept as its latest figure."""
    from ..farm import space

    with _measured_lock:
        before = _written.get(user_id, 0)  # what arrives while it is measured may be counted twice: never too little
    fp = space.measure(user_id)
    found = _areas_of(user_id, fp)
    m = Measured(time.time(), sum(a["bytes"] for a in found), before, found, fp)
    with _measured_lock:
        _measured[user_id] = m
    return m


def areas(user_id: int) -> list[dict]:
    """四部分占用，当前时刻的值（当场量）。"""
    return _measure(user_id).found


def _total(user_id: int) -> int:
    return _measure(user_id).total


def _remeasure(user_id: int) -> None:
    try:
        _measure(user_id)
    finally:
        with _measured_lock:
            _measuring.discard(user_id)


def _latest(user_id: int, wait: bool = True) -> int:
    """The account's latest total: its last measured figure and the upload bytes taken in since (wrote). Older than
    MEASURED_S, measured again on a thread of its own while this answer still says the last one. Never measured:
    measured now (`wait`), or for a poll on that thread too, answering 0 meanwhile."""
    with _measured_lock:
        kept = _measured.get(user_id)
        stale = kept is None or time.time() - kept.at > MEASURED_S
        again = stale and user_id not in _measuring and (kept is not None or not wait)
        if again:
            _measuring.add(user_id)
        since = _written.get(user_id, 0) - kept.before if kept is not None else 0
    if kept is None and wait:
        return _total(user_id)
    if again:
        threading.Thread(target=_remeasure, args=(user_id,), name=f"quota-{user_id}", daemon=True).start()
    return kept.total + since if kept is not None else 0


def _trimmed(user_id: int) -> dict | None:
    from ..farm.queue import trimmed

    got = trimmed.get(user_id)
    return {"at": got[0], "count": got[1]} if got else None


def _groups(user_id: int, fp) -> dict[str, int]:
    """每组（transfer/groups.py）已结束的任务一起删掉能腾出多少：组里几条共用的缓存也算在里面（「删除这一组」腾出的）。"""
    members: dict[str, list[str]] = {}
    done = set(fp.finished)
    for r in db().rows("SELECT id, group_key FROM tasks WHERE user_id = ? AND group_key != ''", (user_id,)):
        if r["id"] in done:
            members.setdefault(r["group_key"], []).append(r["id"])
    return {key: fp.frees(ids) for key, ids in members.items()}


def usage(user_id: int, with_traffic: bool = False) -> dict:
    """账号的磁盘占用：四部分 + 上限 + 每个已结束任务各自的占用（删掉它腾出多少：只被它用到的缓存也算在里面）。
    `with_traffic`：同时附带其网络流量（server/traffic.py）。

    默认不附带流量：流量每次响应都会增加，附带后该响应的 ETag 每次都不同，本应返回 304 的轮询会变成
    整包重发。`with_traffic=True` 只用于后台管理员的两条路由（`/api/admin/users/{id}/quota` 的读和写，
    权限为 users.manage_normal）；前台用户的所有路由都不附带流量，流量只在后台可见。"""
    from .. import traffic
    from ..farm import policy

    m = _measure(user_id)
    limit = limit_of(user_id)
    return {"user": user_id, "limit": limit, "own_gb": own_gb(user_id),
            "default_gb": float(settings()["storage.quota_gb"]), "total": m.total,
            "left": max(limit - m.total, 0) if limit else 0, "over": bool(limit and m.total >= limit),
            "stage": stage_of(m.total, limit), "areas": _worded(m.found),
            # 每个已结束任务删掉能腾出多少（队列窗口每一行），和账号最多留几条已结束的任务（多了最旧的自动删除）
            "tasks": m.fp.own(), "groups": _groups(user_id, m.fp), "finished": len(m.fp.finished),
            "keep_most": policy.keep_most(),
            "trimmed": _trimmed(user_id),
            **({"traffic": traffic.of(user_id)} if with_traffic else {})}


def gate(user_id: int) -> dict:
    """判断是否还能写入内容的数值：已占用、上限、是否已满、阶段（0 / 1 到了 80% / 2 满了）。

    供轮询使用（队列每 1.5–30 秒查询一次）：只有这几个值，不含明细（明细的说明文字等不必每次都发），
    占用读最近一次量的结果（`_latest`，从不在请求里量）。明细见 `usage()`，由「我的占用」单独查询一次。"""
    total = _latest(user_id, wait=False)
    limit = limit_of(user_id)
    return {"total": total, "limit": limit, "over": bool(limit and total >= limit), "stage": stage_of(total, limit)}


def _parts(found: list[dict]) -> list[Word] | Word:
    """「缓存 12.0 GB、任务 3.2 GB、……」：占用从大到小，空的不写。"""
    return [Word("server.quota.part", what=Word(f"server.quota.area.{a['id']}.label"), gb=f"{a['bytes'] / GB:.2f}")
            for a in sorted(found, key=lambda a: -a["bytes"]) if a["bytes"]] or Word("server.quota.all_empty")


def full_message(user_id: int, adding: int = 0, m: Measured | None = None) -> Msg:
    """配额已满时的提示：已占用、上限、本次写入的大小、主要占在哪（四部分从大到小）以及释放方法。释放空间的操作只有
    删除任务，因此提示报告可删除的已结束任务数。没有可删除任务时使用另一条提示（必须说明处理方法，不能只报「失败」）。
    `adding` 为 0 时对应「已满，不能再算」的两条提示。"""
    m = m or _measure(user_id)
    jobs = len(m.fp.finished)
    how = {"used_gb": m.total / GB, "limit_gb": limit_of(user_id) / GB, "parts": _parts(m.found)}
    if jobs:  # 有已结束的任务可删：去队列中删除（或一次删最旧的几条）
        how["jobs"] = jobs
        return Msg("E-QUOTA-COOKFULL", **how) if not adding else Msg("E-QUOTA-FULL", **how, adding_gb=adding / GB)
    # 没有可删除的任务：说明占用来自哪一部分，否则提示只能随意指向某处（从未提交过任务时，空间可能全部在
    # 「上传的素材」上，因为选择文件时即已上传；此时让使用者去「我的模板」中删除便是指错了位置）。
    big = max(m.found, key=lambda a: a["bytes"])
    how |= {"area": Word(f"server.quota.area.{big['id']}.label"), "area_gb": big["bytes"] / GB}
    return Msg("E-QUOTA-COOKSTUCK", **how) if not adding else Msg("E-QUOTA-FULLSTUCK", **how, adding_gb=adding / GB)


def room_for(user_id: int, adding: int, but: str = "") -> None:
    """再写入 `adding` 字节是否可行；不可行时附带提示拒绝（TooLarge：网页原样显示）。按账号的占用数（`_latest`）加上
    它已开的分段还没传来的字节（transfer/uploads.py promised；`but`：不算的那个分段，即再次打开的它自己）判断，
    占用数说放不下时才当场量一次再判断。"""
    from ..transfer import uploads

    limit = limit_of(user_id)
    if not limit:
        return
    adding = max(adding, 0) + uploads.promised(user_id, but)
    if _latest(user_id) + adding > limit:
        m = _measure(user_id)
        if m.total + adding > limit:
            raise TooLarge(full_message(user_id, adding, m))


def refuse_if_full(user_id: int) -> None:
    """点击「计算」「提交」之前（所有提交都经 POST /api/jobs：网页、DCC 插件、命令行）：已满时停止，并说明已占用、上限、
    主要占在哪及怎么腾（网页先按同样规则拦截一次，此处拦截绕过网页直接提交的请求）。只检查是否已满：计算结果的大小在提交时
    尚不可知；正在算的任务照常算完。读最近一次的数，它说满了才当场量一次。"""
    limit = limit_of(user_id)
    if not limit or _latest(user_id) < limit:
        return
    m = _measure(user_id)
    if m.total >= limit:
        raise TooLarge(full_message(user_id, 0, m))


# ------------------------------------------------------------------ 使用者自行清理


def drop_for_job(job_id: str, row: dict) -> dict:
    """删除一个任务时随之释放的空间：它的任务文件夹（节点图、素材、输出的文件夹和 zip、日志），整个删除
    （`farm.queue.forget_job` 删除，此处先测量）。它引用的缓存不在此处删除：没有别的任务再引用、也没有任务在用的，
    由随后的清理带走（farm/disk.py）。正在排队或计算的任务不能删除（先取消）。

    `row`：`farm.queue.job_row` 提供的数据（节点图、记录、账号），路由的 owned 已确认归属。
    返回 {job, bytes}；删除之后的占用由路由在 `forget_job` 之后另取（server/farm.py）。"""
    from ..farm.queue import ensure_finished
    from ..io.files import folder_bytes
    from ..transfer import tasks

    ensure_finished(job_id)  # 首先确认：正在运行的任务不会丢失任何内容（farm/queue.py ensure_finished）
    folders = [tasks.task_dir(job_id)] if tasks.is_task(job_id) else []
    return {"job": job_id, "bytes": folder_bytes(folders)}


PLAN_MOST = 1000  # 一次「腾出空间」最多列出（删除）这么多条：和每账号保留的已完成任务同一量级


def plan(user_id: int, want: int) -> dict:
    """「腾出空间」的预览：删掉最旧的哪几条已结束的任务能腾出至少 `want` 字节（全删也不够时就是全部），一共腾出多少
    （farm/space.py Footprint.plan：几条任务共用的缓存也算进去，和删完实际腾出的一致）。什么都不删。"""
    m = _measure(user_id)
    ids, freed = m.fp.plan(max(int(want), 1))
    ids = ids[:PLAN_MOST]
    if len(ids) == PLAN_MOST:
        freed = m.fp.frees(ids)
    own = m.fp.own()
    records = {}
    if ids:
        from ..database import json_of
        from ..farm.queue import spoken_record

        marks = ",".join("?" * len(ids))
        records = {r["id"]: r for r in db().rows(f"SELECT id, submitted, record FROM jobs WHERE user_id = ? AND id IN ({marks})",
                                                  (user_id, *ids))}
    jobs = []
    for jid in ids:
        r = records.get(jid)
        title = (spoken_record(json_of(r["record"]))["title"] if r is not None else "") or i18n.t("server.quota.untitled_graph")
        jobs.append({"id": jid, "title": title, "submitted": r["submitted"] if r is not None else None, "bytes": own.get(jid, 0)})
    limit = limit_of(user_id)
    return {"want": int(want), "bytes": freed, "jobs": jobs, "total": m.total, "limit": limit,
            "enough": freed >= want, "finished": len(m.fp.finished)}


def free(user_id: int, ids: list[str]) -> dict:
    """执行「腾出空间」：删掉预览里的这几条任务（只删本账号已结束的；预览之后开始算的、已经没了的跳过），每条和单独删除一样
    （farm/queue.py forget_job：它的文件夹整个删除），只被它们用到的缓存随即清掉（farm/disk.py collect_named），
    不等下一次清理：腾出的就是预览说的那些。账号还有任务在排队或计算时，这些缓存留到它们算完后的清理（`pending`）。"""
    from ..farm import disk
    from ..farm.queue import _forget, farm
    from ..errors import Invalid, NotFound

    m = _measure(user_id)
    finished = set(m.fp.finished)
    going = [jid for jid in dict.fromkeys(ids) if jid in finished]
    names = m.fp.entries_of(going)
    expected = m.fp.frees(going)
    done, skipped = [], len(ids) - len(going)
    for jid in going:
        try:
            _forget(jid, user_id)
            done.append(jid)
        except (Invalid, NotFound):
            skipped += 1
    if len(done) != len(going):  # what the preview counted of the skipped ones stays
        names = m.fp.entries_of(done)
        expected = m.fp.frees(done)
    queue = farm()
    disk.collect_named(user_id, names, guard=queue.cleaner())
    if not queue.ending:
        queue._spawn(queue._tidy, name="farm-tidy-now")
    after = usage(user_id)
    freed = max(m.total - after["total"], 0)
    return {**after, "ok": True, "jobs": len(done), "skipped": skipped, "bytes": freed, "expected": expected,
            "pending": max(expected - freed, 0)}


# ------------------------------------------------------------------ 路由

from fastapi import Request  # noqa: E402
from pydantic import Field  # noqa: E402

from . import auth, owners  # noqa: E402
from .access import audit  # noqa: E402
from .routes import Access, Body, Gigabytes, Router  # noqa: E402

router = Router(tags=["Disk Usage"])
admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])


@router.get("/api/my/storage", access=Access.user("Your own disk usage and limit"), summary="How much disk you use: what job folders and templates kept on the server take, and the limit")
def my_storage(request: Request) -> dict:
    # 流量不提供给普通用户，在此处拦截而不是在网页中做角色判断：服务器按当前用户计算可用性。流量仍正常统计
    # （server/traffic.py Meter），读取流量的接口只有后台管理员的几条（本文件下方的 user_quota、/api/admin/users 的列表、
    # 后台用户页的「流量」，webui/src/admin/traffic.tsx），均要求 users.manage_normal。
    return usage(auth.me(request).id)


# 此处没有「按部分清理」「按任务清理缓存」一类路由：释放空间只有一个操作，即删除任务（`DELETE /api/jobs/{id}`、
# `DELETE /api/jobs`，以及下面一次删最旧几条的「腾出空间」），它会整个删除该任务的文件夹；只被它引用的缓存随之清掉
# （farm/disk.py）。按部分清理会忽略引用而删除整个部分，第二种释放空间的途径即为一个会丢失数据的入口。
# 回收站不计入用户配额（library.user_bytes）。


class PlanIn(Body):
    gb: Gigabytes  # 想腾出多少


class FreeIn(Body):
    jobs: list[str] = Field(default_factory=list, max_length=PLAN_MOST)  # 预览列出的任务（确认的就是这几条）


@router.post("/api/my/storage/plan", access=Access.user("Preview of Free Up Space"),
             summary="Preview of Free Up Space: which of your oldest finished jobs free at least gb GB when deleted, and how much "
                     "they free in all (cache used only by them included); nothing is deleted")
def storage_plan(req: PlanIn, request: Request) -> dict:
    return plan(auth.me(request).id, int(req.gb * GB))


@router.post("/api/my/storage/free", access=Access.user("Free Up Space: delete the jobs in the preview", owned=owners.own_jobs,
                                                       body_ids=("jobs",)),
             summary="Free Up Space: delete the finished jobs the preview listed (like deleting each one, with cache used only by "
                     "them); the answer says how many were deleted, how much was freed and the usage afterwards; jobs started since "
                     "the preview or already gone are skipped")
def storage_free(req: FreeIn, request: Request) -> dict:
    return free(auth.me(request).id, request.state.owned[:PLAN_MOST])  # the account's own (owners.own_jobs)


class QuotaIn(Body):
    gb: Gigabytes | None = None  # None：使用设置中的默认值


@admin.get("/users/{user_id}/quota", access=Access.admin("users.manage_normal"), summary="What one account uses: disk limit, this account's own limit (none: the default), what each part takes, and "
                                                                                         "network traffic used (today / last 7 days / in all)")
def user_quota(user_id: int, request: Request) -> dict:
    from ..accounts import get
    from .users import _managed

    # 与「用户」页的其他路由一样，只能查看自身有权管理的角色（roles.manages）：仅有 users.manage_normal 不能读取或
    # 限制内置管理员及其他管理员的配额。账号不存在时返回 404。
    _managed(auth.session(request), get(user_id))
    return usage(user_id, with_traffic=True)


@admin.put("/users/{user_id}/quota", access=Access.admin("users.manage_normal"), summary="Change an account's disk quota (empty: the default in settings); effective at once, an account over it cannot "
                                                                                         "upload until it cleans up")
def set_user_quota(user_id: int, req: QuotaIn, request: Request) -> dict:
    from ..accounts import get

    from .users import _managed

    u = get(user_id)
    _managed(auth.session(request), u)
    set_limit(user_id, req.gb)
    audit(Msg("I-AUDIT-QUOTASET", who=auth.actor(request).label, username=u.username,
              gb=req.gb if req.gb is not None else float(settings()["storage.quota_gb"])),
          about=u.id, session=auth.session(request), method="PUT", path=str(request.url.path))
    return usage(user_id, with_traffic=True)
