"""单账号磁盘配额：一个账号的上传素材、待取回结果、计算缓存以及保存在服务器上的模板合计有一个上限；
默认每个账号 10 GB（设置「账号配额」），可在「用户」中按账号修改。

上限：`users.quota_gb`（该账号自身的），未设置时使用设置中的 `storage.quota_gb`。0 表示不限。
占用：分为四部分，每部分说明其内容及何时释放；配额已满时，这些说明即使用者看到的提示（配额已满时不能只报「失败」）。

    缓存      计算得到的节点结果中只有该账号取回过的部分（`grants`）；其他账号也取回过的计入双方，清理时不删除
    上传素材  该账号上传的文件（相同字节在服务器上只存一份，因此按其名下的部分计算）
    待取回结果 「输出」交付、等待其电脑取回的结果（到期自动删除：设置「结果保留」）
    模板      该账号保存在服务器上的节点图，包括回收站中的

释放空间只有一个操作：在队列中删除任务（`drop_for_job`）。删除一个任务时，其交付包、仅被它使用的缓存和仅被它使用的
上传素材一并删除；仍被其他任务或模板引用的一律保留。四部分占用只用于报告数值（说明空间用在何处），不作为清理入口：
按部分清理会忽略引用而删除整个部分，保留两套释放空间的方式等于保留一个会丢失数据的入口。

清理缓存不得影响正在使用的内容：排队和计算中的任务（不论属于哪个账号）所需的文件夹保留不清（farm/queue.py busy_fingerprints），
清理后该节点图仍可计算，只是不再命中缓存。

队列窗口每 1.5 秒读取一次占用，而计算占用需要访问磁盘。结果始终是当前时刻的值（哪怕相差一秒，「清理后立即看到空间
释放」也会不成立），节省的是计算方式：结果文件夹按内容命名，写完后不再改变，因此每个文件夹的大小按
(名称, 修改时间) 记录一次即可（`_folder_bytes`），之后每次只需 stat。"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import settings
from ..database import db, json_of
from ..errors import TooLarge
from ..messages import Msg

GB = 1 << 30


def limit_of(user_id: int) -> int:
    """该账号的上限，单位为字节（0：不限）。"""
    r = db().row("SELECT quota_gb FROM users WHERE id = ?", (user_id,))
    gb = r["quota_gb"] if r is not None and r["quota_gb"] is not None else settings()["storage.quota_gb"]
    return int(float(gb) * GB)


def set_limit(user_id: int, gb: float | None) -> None:
    """按账号修改配额（None：使用设置中的默认值）。"""
    with db().write() as c:
        c.execute("UPDATE users SET quota_gb = ? WHERE id = ?", (None if gb is None else float(gb), user_id))


def own_gb(user_id: int) -> float | None:
    r = db().row("SELECT quota_gb FROM users WHERE id = ?", (user_id,))
    return None if r is None or r["quota_gb"] is None else float(r["quota_gb"])


# ------------------------------------------------------------------ 占用


def _dir_bytes(path) -> int:
    import os

    total = 0
    for base, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(base, f)).st_size
            except OSError:
                pass
    return total


_SIZES: dict[str, tuple[float, int]] = {}  # 文件夹 -> (其修改时间, 遍历测得的大小)
_SIZES_MAX = 20_000


def _folder_bytes(path: Path) -> int:
    """文件夹的占用大小。结果和交付文件夹都按内容命名、写完后不再改变，因此名称和修改时间相同即内容相同，
    遍历测量一次后只需 stat；占用条每 1.5 秒读取一次也不会造成过多磁盘访问。"""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return 0
    hit = _SIZES.get(str(path))
    if hit is not None and hit[0] == mtime:
        return hit[1]
    size = _dir_bytes(path)
    if len(_SIZES) > _SIZES_MAX:
        _SIZES.clear()
    _SIZES[str(path)] = (mtime, size)
    return size


def _cache_entries(user_id: int) -> list[tuple[Path, int, bool]]:
    """该账号的缓存条目：(文件夹, 大小, 是否仅被该账号使用)。结果由谁取回记录在 `grants` 中
    （server/access.py grant）；其他账号也取回过的部分在删除该账号的任务时保留。

    管理员的这一部分始终为 0，并非遗漏：`grant` 对持有 `data.others` 的账号不记录任何行（该账号本就能读取任何结果，
    无需授权）。因此任何账号都不「拥有」管理员计算的缓存，删除管理员的任务也不会带走它；这部分由后台「磁盘」页按
    「结果保留」天数清理（farm/disk.py）。不应为使这一部分有数值而去掉 `grant` 中针对 data.others 的短路：
    管理员计算的内容最多，否则会最先达到配额上限而无法继续计算。配额用于限制普通账号，而非管理员。"""
    from ..data.packet import cache_root

    mine = {r["fp"] for r in db().rows("SELECT fp FROM grants WHERE user_id = ?", (user_id,))}
    others = {r["fp"] for r in db().rows("SELECT DISTINCT fp FROM grants WHERE user_id <> ?", (user_id,))}
    root = cache_root()
    out = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir() or d.name.startswith("."):
            continue
        fp = _fp(d)
        if fp in mine:
            out.append((d, _folder_bytes(d), fp not in others))
    return out


def _upload_bytes(user_id: int) -> int:
    from ..transfer.uploads import blob_path

    total = 0
    for r in db().rows("SELECT key FROM uploads WHERE user_id = ?", (user_id,)):
        key = str(r["key"])
        if len(key) == 64:  # blob 的 sha256：即字节本身
            try:
                total += blob_path(key).stat().st_size
            except (OSError, ValueError):
                pass
    return total


def _delivery_bytes(user_id: int) -> int:
    from ..transfer import deliveries

    total = 0
    for r in db().rows("SELECT run, node FROM deliveries WHERE user_id = ?", (user_id,)):
        total += _folder_bytes(deliveries.folder(r["run"], r["node"]))
    return total


def areas(user_id: int) -> list[dict]:
    """四部分占用，每部分包括：大小、内容说明、何时释放。始终为当前时刻的值。

    仅用于报告数值，不作为清理入口：前台「我的占用」面板只显示总数，这四部分供后台「用户」页
    （webui/src/admin/UserQuota.tsx）和配额已满的提示使用，说明空间用在何处。"""
    from .. import accounts, library

    cache = _cache_entries(user_id)
    return [
        {"id": "cache", "label": "缓存", "bytes": sum(b for _, b, _ in cache),
         "note": "算好的节点结果：删掉算出它的任务时，没有别的任务再用它的就一起腾出来"},
        {"id": "uploads", "label": "上传的素材", "bytes": _upload_bytes(user_id),
         "note": "自己上传的画面和文件：删掉用到它们的任务时，没有别的任务和模板再用它的就一起腾出来"},
        {"id": "deliveries", "label": "待取回的结果", "bytes": _delivery_bytes(user_id),
         "note": f"「输出」算好、等这台电脑取回的结果：删掉那条任务就马上腾出来；不删的话取回后过 "
                 f"{int(settings()['storage.delivery_days'])} 天自动删除"},
        {"id": "templates", "label": "我的模板", "bytes": library.user_bytes(accounts.get(user_id).username),
         "note": "存在服务器上的节点图（在「我的模板」里删掉就不算了；删掉的那一份留给管理员恢复，不占你的额度）"},
    ]


def usage(user_id: int, with_traffic: bool = False) -> dict:
    """账号的磁盘占用：四部分 + 上限。`with_traffic`：同时附带其网络流量（server/traffic.py）。

    默认不附带流量：流量每次响应都会增加，附带后该响应的 ETag 每次都不同，本应返回 304 的轮询会变成
    整包重发。`with_traffic=True` 只用于后台管理员的两条路由（`/api/admin/users/{id}/quota` 的读和写，
    权限为 users.manage_normal）；前台用户的所有路由都不附带流量，流量只在后台可见。"""
    from .. import traffic

    found = areas(user_id)
    total = sum(a["bytes"] for a in found)
    limit = limit_of(user_id)
    return {"user": user_id, "limit": limit, "own_gb": own_gb(user_id),
            "default_gb": float(settings()["storage.quota_gb"]), "total": total,
            "left": max(limit - total, 0) if limit else 0, "over": bool(limit and total >= limit),
            "areas": found, **({"traffic": traffic.of(user_id)} if with_traffic else {})}


def gate(user_id: int) -> dict:
    """判断是否还能写入内容的三个数值：已占用、上限、是否已满。

    供轮询使用（队列每 1.5–30 秒查询一次）：只有这三个值，不含四部分明细。明细中的「缓存」在每次计算后
    都会变化，响应每次不同就永远无法命中 ETag，本应返回 304 的轮询会变成整包重发。这三个值只在用户自行保存
    内容后才会变化，因此轮询仍可正常退避。明细见 `usage()`，由「我的占用」单独查询一次。"""
    total = sum(a["bytes"] for a in areas(user_id))
    limit = limit_of(user_id)
    return {"total": total, "limit": limit, "over": bool(limit and total >= limit)}


def _finished_jobs(user_id: int) -> int:
    """该账号可删除的已结束任务数（排队和计算中的任务需先取消，不计入）。"""
    from ..farm.queue import farm, history

    live = farm().active_ids()  # 在队列的锁内获取：各通道会同时遍历 `jobs`
    return sum(1 for e in history(limit=1000, user_id=user_id) if str(e.get("id") or "") not in live and e.get("id"))


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


def room_for(user_id: int, adding: int) -> None:
    """再写入 `adding` 字节是否可行；不可行时附带提示拒绝（TooLarge：网页原样显示）。"""
    limit = limit_of(user_id)
    if not limit:
        return
    total = sum(a["bytes"] for a in areas(user_id))
    if total + max(adding, 0) > limit:
        raise TooLarge(full_message(user_id, adding))


def refuse_if_full(user_id: int) -> None:
    """点击「计算」「提交」之前：已满时停止，并说明已占用、上限及清理位置（网页先按同样规则拦截一次，
    此处拦截绕过网页直接提交的请求）。只检查是否已满：计算结果的大小在提交时尚不可知。"""
    if usage(user_id)["over"]:
        raise TooLarge(full_message(user_id))


# ------------------------------------------------------------------ 使用者自行清理


def _busy() -> set[str]:
    """排队和计算中的任务所需的缓存文件夹：清理时一概不动。无法计算（某任务此刻无法规划）时不清理任何内容，
    宁可让使用者再操作一次，也不从正在计算的任务下删除文件。"""
    from ..errors import Conflict
    from ..farm.queue import busy_fingerprints

    found = busy_fingerprints()
    if found is None:
        raise Conflict(Msg("E-QUOTA-BUSY"))
    return found


def _fp(folder: Path) -> str:
    """缓存文件夹名为 `<指纹>`、`<指纹>_display`（数据包指纹）或 `<指纹>_work` / `<指纹>_failed`（节点指纹）：取出其中的指纹部分。"""
    return folder.name.split("_")[0]


def _drop(user_id: int, entries: list[tuple[Path, int, bool]], seen: set[str] | None = None) -> tuple[int, int]:
    """删除这些缓存条目，并一并删除该账号对它们的取回记录（grants）。返回（份数, 字节数）。
    `seen`：计算无人使用的部分（`_busy()`）时队列中的任务号；每删除一份之前再次检查（`Farm.removing`），
    队列中出现新任务即停止并只报告已删除的部分：测量磁盘需要时间，期间提交的任务所需的缓存不能被删除。"""
    removed = freed = 0
    gone: list[str] = []
    from ..data import packet as packets
    from ..farm.queue import farm

    for folder, size, _only in entries:
        with farm().removing(seen) as go:
            if not go:
                break  # 队列中出现新任务：尚未计算其所需的缓存，本次不再删除剩余部分
            packets.remove(folder.name, "quota")  # 删除缓存文件夹只经过这一个入口
        gone.append(_fp(folder))
        removed += 1
        freed += size
    if gone:
        from ..farm.queue import forget_marks

        forget_marks()  # 清除的可能是多个任务共用的缓存：每一行的缓存标记都重新计算
        with db().write() as c:
            c.executemany("DELETE FROM grants WHERE user_id = ? AND fp = ?", [(user_id, fp) for fp in set(gone)])
        from ..engine.evaluations import EVALUATIONS

        EVALUATIONS.bump()
    return removed, freed


def _after_clean(user_id: int, **said) -> dict:
    """清理后返回给网页的内容：一份完整的占用，面板直接用它 `setUsage(got)` 替换当前数据。
    不附带流量：前台的所有路由都不发送流量。"""
    return {**said, **usage(user_id)}


def _only_this_job_s(user_id: int, job_id: str, going: set[str], fps: set[str], sids: set[str]) -> tuple[set[str], set[str]]:
    """从「该任务使用的」两类候选中剔除仍被其他对象使用的部分；剩余部分才可随该任务一并删除。
    参数 `fps`：其结果所在的缓存指纹；`sids`：其引用的上传集合（`upload:<id>/…`）。

    一份素材被多个任务引用时，只有完全无引用才删除。「仍被其他对象使用」包括两类引用：

      模板      该账号保存在服务器上的节点图（`library.user_graphs`，包括回收站中的）：模板的生命周期比任务长，
                删除任务时若删除模板所需的素材，该模板下次打开时会显示「素材已经不在了」。先检查模板，因为无需规划，速度快
      其他任务  该账号名下的其他任务记录（节点图 + 记录）：同一节点图提交两次时共用同一份缓存

    `going`：本次一并删除的任务号（「删除全部」）：这些任务自身也在删除中，不计入「仍被其他对象使用」，否则 N 个
    任务互相阻挡，一个也无法释放。
    两类候选都为空时立即结束：剩余任务一个都不规划（规划一张节点图需构建一次 Evaluation，开销不小）。"""
    from ..farm.queue import farm, job_fingerprints, job_graph
    from ..serving import Account
    from ..transfer.uploads import refs_in

    from .. import accounts, library

    if sids:
        for graph in library.user_graphs(accounts.get(user_id).username):  # 包括回收站中的
            sids -= refs_in(graph)
            if not sids:
                break
    account = Account(user_id)
    for r in farm().db.rows("SELECT id, record FROM jobs WHERE user_id = ?", (user_id,)):
        if not fps and not sids:
            break
        jid = str(r["id"])
        if jid == job_id or jid in going:
            continue
        graph, record = job_graph(jid), json_of(r["record"], {})
        if sids:
            sids -= refs_in(graph) | refs_in(record)
        if fps:
            fps -= job_fingerprints(graph, record, account)
    return fps, sids


def _job_delivery_bytes(user_id: int, job_id: str) -> int:
    """该任务留在服务器上的交付包大小（`forget_job` 紧接着会删除它们：必须先测量再删除）。"""
    from ..transfer import deliveries

    return sum(_folder_bytes(deliveries.folder(r["run"], r["node"]))
               for r in db().rows("SELECT run, node FROM deliveries WHERE user_id = ? AND run = ?", (user_id, job_id)))


def drop_for_job(job_id: str, row: dict, going: set[str] = frozenset()) -> dict:
    """删除一个任务时随之释放的空间。包括三项，每项都先确认是否仍被其他对象使用：

      交付包    该任务自身的交付包（由 `forget_job` 删除，此处只将其大小计入释放的字节数）
      缓存      该任务计算得出、只有该账号取回过、当前没有排队或计算中的任务需要、
                且其他任务和模板都不再使用的部分
      上传素材  该节点图及其记录引用的上传集合（`upload:<id>/…`）中，其他任务和模板都不再引用的部分；
                一份 blob 只有在没有任何账号登记、也没有其他上传集合的清单引用它时才实际删除（uploads.drop_sets）

    无法规划时不删除任何内容（`_busy()` 抛出 E-QUOTA-BUSY，路由返回 409）：此时无法得知正在计算的任务需要哪些内容，
    宁可让使用者再操作一次，也不从正在计算的任务下删除文件。抛出异常时任务也不会被删除
    （`server/farm.py forget` 先调用此函数、后调用 `forget_job`），因此使用者看到的是「未删除，请再试一次」，
    而不是「任务已删除但空间仍被占用」。

    `row`：`farm.queue.job_row` 提供的数据（节点图、记录、账号），路由的 owned 已确认归属。
    `going`：本次一并删除的任务号（「删除全部」），见 `_only_this_job_s`。
    响应中附带一份完整的占用（磁盘四部分 + 上限，不含流量）：面板直接用它替换当前数据（`_after_clean`）。"""
    from ..farm.queue import ensure_finished, farm, job_fingerprints
    from ..serving import Account
    from ..transfer.uploads import drop_sets, refs_in

    ensure_finished(job_id)  # 首先确认：正在运行的任务不会丢失任何内容（farm/queue.py ensure_finished）
    user_id = int(row["user"])
    account = Account(user_id)
    seen = farm().active_ids()  # 先记录此刻队列中的任务，再计算其所需内容：之后新增的任务会被 `_drop` 在每次删除前发现
    busy = _busy()  # 无法规划时在此停止，此时尚未删除任何内容
    fps, sids = _only_this_job_s(user_id, job_id, set(going),
                                 job_fingerprints(row["graph"], row["record"], account) - busy,
                                 refs_in(row["graph"]) | refs_in(row["record"]))
    removed, freed = _drop(user_id, [e for e in _cache_entries(user_id) if e[2] and _fp(e[0]) in fps], seen)
    uploads, upload_bytes = drop_sets(user_id, sids)

    delivered = _job_delivery_bytes(user_id, job_id)
    return _after_clean(user_id, job=job_id, cache=removed, uploads=uploads,
                        bytes=freed + upload_bytes + delivered)


# ------------------------------------------------------------------ 路由

from fastapi import Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import auth, owners  # noqa: E402
from .access import audit  # noqa: E402
from .routes import Access, Router  # noqa: E402

router = Router(tags=["磁盘占用"])
admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])


@router.get("/api/my/storage", access=Access.user("自己的磁盘占用和上限"), summary="自己占了多少资源：硬盘四块（每一块是什么占的）和上限")
def my_storage(request: Request) -> dict:
    # 流量不提供给普通用户，在此处拦截而不是在网页中做角色判断：服务器按当前用户计算可用性。流量仍正常统计
    # （server/traffic.py Meter），读取流量的接口只有后台管理员的几条（本文件下方的 user_quota、/api/admin/users 的列表、
    # resources.py 的「流量」页签），均要求 users.manage_normal。
    return usage(auth.me(request).id)


# 此处没有「按部分清理」「按任务清理缓存」一类路由：释放空间只有一个操作，即在队列中删除任务（`DELETE /api/jobs/{id}` 和
# `DELETE /api/jobs`），它会一并删除交付包、仅被其使用的缓存和仅被其使用的上传素材（`drop_for_job`）。
# 按部分清理会忽略引用而删除整个部分，第二种释放空间的途径即为一个会丢失数据的入口。回收站不计入用户配额（library.bytes_of）。


class QuotaIn(BaseModel):
    gb: float | None = None  # None：使用设置中的默认值


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
    audit(Msg("I-AUDIT-QUOTASET", who=auth.label(request), username=u.username,
              gb=req.gb if req.gb is not None else float(settings()["storage.quota_gb"])),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return usage(user_id, with_traffic=True)
