"""单账号磁盘配额：一个账号所有任务的文件夹、还没有任务用到的上传、保存在服务器上的模板合计有一个上限：配额 = 该账号所有
任务文件夹的大小（上传的素材 + 输出的文件夹和 zip）+ 其余的上传 + 保存在服务器上的模板；缓存不计入。

上限：`users.quota_gb`（该账号自身的），未设置时使用设置中的 `storage.quota_gb`（「账号配额」）。0 表示不限。
占用：分为三部分，每部分说明其内容及何时释放；配额已满时，这些说明即使用者看到的提示（配额已满时不能只报「失败」）。

    任务      该账号每个任务的文件夹（transfer/tasks.py：节点图、硬链接进来的素材、输出、日志）。
              同一份字节（同一素材硬链接进几个任务）只算一次。任务结束「任务保留天数」后整个删除，删掉任务也立刻腾出来
    上传      该账号的上传文件夹（transfer/uploads.py home_of）里还没有链接进任务的：选了文件时申报的文件头、传了
              一半的分段、传完还没有任务用到的素材。不计入的话，只上传不提交就能把服务器的硬盘占满。没有任务用到的，
              「任务保留天数」后清理
    模板      该账号保存在服务器上的节点图（回收站中的不计入）

缓存不计入配额（每个账号各自的缓存随引用它的任务清理，farm/disk.py；它由计算产生，随任务走）。

释放空间只有一个操作：在队列中删除任务（`drop_for_job`）：它的文件夹整个删除，只被它引用的缓存随后的清理带走。
三部分占用只用于报告数值（说明空间用在何处），不作为清理入口。

量一次占用要把账号的上传和进行中任务的文件夹逐个 stat 一遍（传过几万帧序列就是几万次）。所以账号的占用是一个数
（`_latest`）：最近一次量的结果（`_measured`），加上这以后本进程收进来的上传字节（`wrote`：每个分段只加自己的字节，
server/transfer.py 收字节的那一处）。轮询（队列窗口和每个页面的 /api/load，每 1.5–30 秒；`gate`）和写入之前的额度检查
（`room_for`）都读这个数；量的结果超过 MEASURED_S 秒时在后台线程里重量一次，这一次仍用上一次的数；只有这个账号第一次被问时
在请求里量。额度检查只在这个数说放不下时才当场量一次（占用可能已经因删任务变小）。「我的占用」打开时、删掉任务之后
（`usage`）当场量，并更新这份结果：删掉任务后的下一次轮询就看到空间释放。"""

from __future__ import annotations

import math
import threading
import time

from ..config import QUOTA_GB_MAX, settings
from ..database import db
from ..errors import TooLarge
from ..messages import Msg

GB = 1 << 30


def _gb(value) -> float:
    """A quota as kept, in GB. One past what can be set (config QUOTA_GB_MAX; an infinite one, or one so large its bytes
    are no number: written before requests' numbers were bounded, routes.Gigabytes) reads as 0, 不限, which is what it
    meant."""
    gb = float(value)
    return gb if math.isfinite(gb) and gb <= QUOTA_GB_MAX else 0.0


def limit_of(user_id: int) -> int:
    """该账号的上限，单位为字节（0：不限）。"""
    r = db().row("SELECT quota_gb FROM users WHERE id = ?", (user_id,))
    gb = r["quota_gb"] if r is not None and r["quota_gb"] is not None else settings()["storage.quota_gb"]
    return int(_gb(gb) * GB)


def set_limit(user_id: int, gb: float | None) -> None:
    """按账号修改配额（None：使用设置中的默认值）。"""
    with db().write() as c:
        c.execute("UPDATE users SET quota_gb = ? WHERE id = ?", (None if gb is None else float(gb), user_id))


def own_gb(user_id: int) -> float | None:
    r = db().row("SELECT quota_gb FROM users WHERE id = ?", (user_id,))
    return None if r is None or r["quota_gb"] is None else _gb(r["quota_gb"])


# ------------------------------------------------------------------ 占用


def areas(user_id: int) -> list[dict]:
    """三部分占用，每部分包括：大小、内容说明、何时释放。始终为当前时刻的值。

    仅用于报告数值，不作为清理入口：前台「我的占用」面板只显示总数，这三部分供后台「用户」页
    （webui/src/admin/UserQuota.tsx）和配额已满的提示使用，说明空间用在何处。"""
    from .. import accounts, library
    from ..transfer import tasks, uploads

    in_tasks, in_uploads = tasks.account_bytes(user_id, uploads.home_of(user_id))
    return [
        {"id": "tasks", "label": "任务", "bytes": in_tasks,
         "note": f"每个任务的文件夹：节点图、上传的素材（同一份只算一次）、输出的文件夹和 zip、日志；"
                 f"任务结束 {tasks.keep_days()} 天后整个删除，在「队列」里删掉任务马上腾出来"},
        {"id": "uploads", "label": "还没用上的素材", "bytes": in_uploads,
         "note": f"传上来、还没有任务用到的素材（选了文件时传的文件头、传了一半的也算）；用到它的任务提交之后就算在任务里。"
                 f"没有任务用到的 {tasks.keep_days()} 天后自己清掉"},
        {"id": "templates", "label": "我的模板", "bytes": library.user_bytes(accounts.get(user_id).username),
         "note": "存在服务器上的节点图（在「我的模板」里删掉就不算了；删掉的那一份留给管理员恢复，不占你的额度）"},
    ]


MEASURED_S = 10.0  # how old the figures a poll gets may be
_measured: dict[int, tuple[float, int, int]] = {}  # account -> (when, total bytes, what `wrote` had counted by then)
_written: dict[int, int] = {}  # account -> bytes of uploads this process took in (wrote), ever growing
_measuring: set[int] = set()  # accounts measured again in the background now
_measured_lock = threading.Lock()


def wrote(user_id: int, count: int) -> None:
    """`count` bytes of an upload of the account's have just reached the disk: its figure grows by them (_latest)."""
    with _measured_lock:
        _written[user_id] = _written.get(user_id, 0) + count


def _total(user_id: int, found: list[dict] | None = None) -> int:
    """The account's total now (the three areas, measured unless given), kept as its latest figure."""
    with _measured_lock:
        before = _written.get(user_id, 0)  # what arrives while it is measured may be counted twice: never too little
    total = sum(a["bytes"] for a in (areas(user_id) if found is None else found))
    with _measured_lock:
        _measured[user_id] = (time.time(), total, before)
    return total


def _remeasure(user_id: int) -> None:
    try:
        _total(user_id)
    finally:
        with _measured_lock:
            _measuring.discard(user_id)


def _latest(user_id: int) -> int:
    """The account's latest total: its last measured figure and the upload bytes taken in since (wrote). Measured now
    only the first time; older than MEASURED_S, measured again on a thread of its own while this answer still says the
    last one."""
    with _measured_lock:
        kept = _measured.get(user_id)
        again = kept is not None and time.time() - kept[0] > MEASURED_S and user_id not in _measuring
        if again:
            _measuring.add(user_id)
        since = _written.get(user_id, 0) - kept[2] if kept is not None else 0
    if kept is None:
        return _total(user_id)
    if again:
        threading.Thread(target=_remeasure, args=(user_id,), name=f"quota-{user_id}", daemon=True).start()
    return kept[1] + since


def usage(user_id: int, with_traffic: bool = False) -> dict:
    """账号的磁盘占用：三部分 + 上限。`with_traffic`：同时附带其网络流量（server/traffic.py）。

    默认不附带流量：流量每次响应都会增加，附带后该响应的 ETag 每次都不同，本应返回 304 的轮询会变成
    整包重发。`with_traffic=True` 只用于后台管理员的两条路由（`/api/admin/users/{id}/quota` 的读和写，
    权限为 users.manage_normal）；前台用户的所有路由都不附带流量，流量只在后台可见。"""
    from .. import traffic

    found = areas(user_id)
    total = _total(user_id, found)
    limit = limit_of(user_id)
    return {"user": user_id, "limit": limit, "own_gb": own_gb(user_id),
            "default_gb": float(settings()["storage.quota_gb"]), "total": total,
            "left": max(limit - total, 0) if limit else 0, "over": bool(limit and total >= limit),
            "areas": found, **({"traffic": traffic.of(user_id)} if with_traffic else {})}


def gate(user_id: int) -> dict:
    """判断是否还能写入内容的三个数值：已占用、上限、是否已满。

    供轮询使用（队列每 1.5–30 秒查询一次）：只有这三个值，不含三部分明细（明细的说明文字等不必每次都发），
    占用读最近一次量的结果（`_latest`）。明细见 `usage()`，由「我的占用」单独查询一次。"""
    total = _latest(user_id)
    limit = limit_of(user_id)
    return {"total": total, "limit": limit, "over": bool(limit and total >= limit)}


def _finished_jobs(user_id: int) -> int:
    """该账号可删除的已结束任务数（排队和计算中的任务需先取消，不计入）。"""
    from ..farm.queue import finished_of

    return len(finished_of(user_id))


def full_message(user_id: int, adding: int = 0) -> Msg:
    """配额已满时的提示：已占用、上限、本次写入的大小以及释放方法。释放空间的操作只有在队列中删除任务，
    因此提示报告可删除的已结束任务数。没有可删除任务时使用另一条提示（必须说明处理方法，不能只报「失败」）。
    `adding` 为 0 时对应「已满，无法再写入任何内容」的两条提示。"""
    found = usage(user_id)
    jobs = _finished_jobs(user_id)
    how = {"used_gb": found["total"] / GB, "limit_gb": found["limit"] / GB}
    if jobs:  # 有已结束的任务可删：提示只说明去队列中删除，无需推断占用来自哪一部分
        how["jobs"] = jobs
        return Msg("E-QUOTA-COOKFULL", **how) if not adding else Msg("E-QUOTA-FULL", **how, adding_gb=adding / GB)
    # 没有可删除的任务：说明占用来自哪一部分，否则提示只能随意指向某处（从未提交过任务时，空间可能全部在
    # 「上传的素材」上，因为选择文件时即已上传；此时让使用者去「我的模板」中删除便是指错了位置）。
    big = max(found["areas"], key=lambda a: a["bytes"])
    how |= {"area": big["label"], "area_gb": big["bytes"] / GB}
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
    if _latest(user_id) + adding > limit and _total(user_id) + adding > limit:
        raise TooLarge(full_message(user_id, adding))


def refuse_if_full(user_id: int) -> None:
    """点击「计算」「提交」之前：已满时停止，并说明已占用、上限及清理位置（网页先按同样规则拦截一次，
    此处拦截绕过网页直接提交的请求）。只检查是否已满：计算结果的大小在提交时尚不可知。"""
    if usage(user_id)["over"]:
        raise TooLarge(full_message(user_id))


# ------------------------------------------------------------------ 使用者自行清理


def drop_for_job(job_id: str, row: dict) -> dict:
    """删除一个任务时随之释放的空间：它的任务文件夹（节点图、素材、输出的文件夹和 zip、日志），整个删除
    （`farm.queue.forget_job` 删除，此处先测量）。它引用的缓存不在此处删除：没有别的任务再引用、也没有任务在用的，
    由随后的清理带走（farm/disk.py，缓存不计入配额）。正在排队或计算的任务不能删除（先取消）。

    `row`：`farm.queue.job_row` 提供的数据（节点图、记录、账号），路由的 owned 已确认归属。
    返回 {job, bytes}；删除之后的占用由路由在 `forget_job` 之后另取（server/farm.py）。"""
    from ..farm.queue import ensure_finished
    from ..io.files import folder_bytes
    from ..transfer import tasks

    ensure_finished(job_id)  # 首先确认：正在运行的任务不会丢失任何内容（farm/queue.py ensure_finished）
    folders = [tasks.task_dir(job_id)] if tasks.is_task(job_id) else []
    return {"job": job_id, "bytes": folder_bytes(folders)}


# ------------------------------------------------------------------ 路由

from fastapi import Request  # noqa: E402

from . import auth  # noqa: E402
from .access import audit  # noqa: E402
from .routes import Access, Body, Gigabytes, Router  # noqa: E402

router = Router(tags=["磁盘占用"])
admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])


@router.get("/api/my/storage", access=Access.user("自己的磁盘占用和上限"), summary="自己占了多少硬盘：任务文件夹、保存在服务器上的模板各占多少，以及上限")
def my_storage(request: Request) -> dict:
    # 流量不提供给普通用户，在此处拦截而不是在网页中做角色判断：服务器按当前用户计算可用性。流量仍正常统计
    # （server/traffic.py Meter），读取流量的接口只有后台管理员的几条（本文件下方的 user_quota、/api/admin/users 的列表、
    # 后台用户页的「流量」，webui/src/admin/traffic.tsx），均要求 users.manage_normal。
    return usage(auth.me(request).id)


# 此处没有「按部分清理」「按任务清理缓存」一类路由：释放空间只有一个操作，即在队列中删除任务（`DELETE /api/jobs/{id}` 和
# `DELETE /api/jobs`），它会整个删除该任务的文件夹；只被它引用的缓存由随后的清理带走（farm/disk.py）。
# 按部分清理会忽略引用而删除整个部分，第二种释放空间的途径即为一个会丢失数据的入口。回收站不计入用户配额（library.user_bytes）。


class QuotaIn(Body):
    gb: Gigabytes | None = None  # None：使用设置中的默认值


@admin.get("/users/{user_id}/quota", access=Access.admin("users.manage_normal"), summary="一个账号占了多少资源：磁盘上限、这个账号自己的上限（没有就跟默认）、每一块占多少，和用掉的网络流量（今天 / 近 7 天 / 总计）")
def user_quota(user_id: int, request: Request) -> dict:
    from ..accounts import get
    from .users import _managed

    # 与「用户」页的其他路由一样，只能查看自身有权管理的角色（roles.manages）：仅有 users.manage_normal 不能读取或
    # 限制内置管理员及其他管理员的配额。账号不存在时返回 404。
    _managed(auth.session(request), get(user_id))
    return usage(user_id, with_traffic=True)


@admin.put("/users/{user_id}/quota", access=Access.admin("users.manage_normal"), summary="改一个账号的磁盘配额（不填：跟着设置里的默认走）；马上生效，超了的账号不能再上传，要先清理")
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
