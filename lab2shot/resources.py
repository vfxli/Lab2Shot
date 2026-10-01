"""与账号关联的每种资源各登记一行：后台「用户」详情页的页签和各资源列表页的「按用户筛选」都读取此表。

每行说明五项内容：中文名、如何按账号分页列出（一个函数，即该资源本身的列出函数）、每条显示哪些列、在后台中
从哪个区打开、需要哪项权限才能查看。页面不为每种资源单独编写代码，今后新增用户资源只需添加一行。

    Resource(id, label, table, needs, section, columns, rows, when, state, for_admins, acts, dim)

    table    其在数据库中的表：规则是数据库中带账号列的表都必须在此登记，未登记的计为 0
    needs    查看所需的权限（lab2shot/roles.py 中的一项），由路由按声明检查
    section  后台中打开它的分区（webui 的后台分区 id），"" 表示尚无独立页面
    columns  每条显示的列：(字段, 中文列名) 或 (字段, 中文列名, 显示方式)，第一列为该条的标识。
             「显示方式」目前只有 "size"（字节数，网页按 platform/format.ts 显示为 215 B、1.2 MB，大小的显示方式
             仍只有该处实现）；未指定时按值本身显示（时间、是 / 否、数字、用户填写的文字）
    rows     按账号分页列出：rows(user_id, offset, limit, q) -> (总数, [行])；即该资源自身的列出函数，其他位置不再重复实现
    when     按时间筛选所依据的列（时间戳列的键），"" 表示该资源不支持时间筛选
    state    按状态筛选所依据的列（取值较少的列），"" 表示该资源不支持状态筛选
    for_admins 仅对具有管理权限的账号显示（管理操作）：被查看账号的角色不具备任何管理能力时，页签也不显示
    acts     每行可执行的操作（Act）：按钮的中文名及调用的路由；所需权限写在该路由上（server/routes.py），
             此处不重复声明，服务器按当前登录用户计算可用的操作，无权限的按钮不显示
    dim      需要灰显的行：行中的一个布尔字段（`__dim`），如「用户已删除」的模板

搜索、状态、时间三种筛选都在此处计算：网页只回传用户的选择，页面中不为任何一种资源编写代码。

`kinds()` 为登记表本身，`page(kind, user_id, ...)` 是所有页面读取的统一结果，`counts(user_id)` 为页签上的数量。"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .errors import NotFound
from .messages import Msg, names, template

Rows = Callable[[int, int, int, str], tuple[int, list[dict]]]

# 行上的两个通用字段，不属于登记表声明的列：该行是否灰显（`__dim`），以及该行可执行的操作（`__acts`，Act 的 id）
RESERVED = "__"


@dataclass(frozen=True)
class Act:
    """行上的一个操作（后台「用户」详情页的表格中绘制为按钮）：中文名、说明、调用的路由。

    权限不在此处声明：路由自身声明所需能力（server/routes.py Access），由 server/available.py 按当前登录用户
    解析（与其他按钮使用同一机制），无法使用的按钮不显示；网页中不编写任何角色判断。
    `path` 中的 `{id}` 由该行第一列（登记表声明的标识列）替换。"""

    id: str
    label: str
    tip: str
    route: str  # 其调用的路由，按 server/routes.py 中登记的写法："POST /api/admin/graphs/{gid}/restore"
    danger: bool = False

    def json(self) -> dict:
        """网页读取的形式：路径中的参数替换为 `{id}`，由网页代入该行的标识。
        该操作的可用性由 server/available.py 按其路由计算（`route` 保留在结果中以供核对）。"""
        method, _, path = self.route.partition(" ")
        return {"id": self.id, "label": self.label, "tip": self.tip, "method": method,
                "path": re.sub(r"\{[^{}]+\}", "{id}", path), "danger": self.danger, "route": self.route}


@dataclass(frozen=True)
class Resource:
    id: str
    label: str
    table: str
    needs: str
    section: str
    columns: tuple[tuple[str, ...], ...]  # (字段, 中文列名) 或 (字段, 中文列名, 显示方式)
    rows: Rows
    when: str = ""  # 「时间」筛选读取的列（时间戳）
    state: str = ""  # 「状态」筛选读取的列（取值较少）
    for_admins: bool = False  # 仅对自身具有管理权限的账号显示
    acts: tuple[Act, ...] = ()  # 对单行可执行的操作（Act；页面按此绘制）
    dim: bool = False  # 行可带有 `__dim`：表示该行已不再使用（由共享表格灰显）


def _match(row: dict, fields: tuple[str, ...], q: str) -> bool:
    return not q or any(q.lower() in str(row.get(f) or "").lower() for f in fields)


def _cut(found: list[dict], offset: int, limit: int) -> tuple[int, list[dict]]:
    return len(found), found[offset:offset + limit]


def _take(rows: list[dict], columns: tuple[tuple[str, ...], ...], offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    """列出函数返回的行经过搜索并截取为一页，每行只保留声明的列。

    「时间」和「状态」筛选在此之后由 page() 按登记表为其声明的列执行。"""
    fields = tuple(c[0] for c in columns)
    kept = [{**{f: r.get(f) for f in fields}, **{k: v for k, v in r.items() if k.startswith(RESERVED)}}
            for r in rows if _match(r, fields, q)]
    return _cut(kept, offset, limit)


# ------------------------------------------------------------------ 各资源自身的列出函数


# 页面上任务或运行状态的显示名称。列表示的是该行的内容，因此登记表给出显示文字而非键；共享表格按原样绘制，
# 不了解任务的细节（webui/src/ui/Queue.tsx 中编辑器自身的队列使用相同的六个词）。
STATES = {"queued": "排队", "running": "计算中", "done": "完成", "failed": "出错", "cancelled": "已取消",
          "interrupted": "中断"}


def _jobs(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    from .farm.queue import history

    rows = [{"id": j["id"], "title": j["title"], "targets": "、".join(j.get("targets") or []),
             "state": STATES.get(j["state"], j["state"]),
             "submitted": j["submitted"], "finished": j.get("finished")} for j in history(HISTORY, user_id=user_id)]
    return _take(rows, KINDS["jobs"].columns, offset, limit, q)


def _uploads(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    from .transfer.uploads import of_account

    return _take(of_account(user_id), KINDS["uploads"].columns, offset, limit, q)


def _feedback(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    from .feedback import listing

    rows = [r for r in listing()["items"] if r.get("user") == user_id]
    return _take(rows, KINDS["feedback"].columns, offset, limit, q)


def _outputs(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    """该账号各任务里「输出」打包好的结果（transfer/outputs.py）：随任务保留。"""
    from .transfer.outputs import of_account

    return _take(of_account(user_id), KINDS["outputs"].columns, offset, limit, q)


def _logins(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    from .accounts import SESSION_KINDS, login_recent

    rows = [{**r, "kind": SESSION_KINDS[r["kind"]]} for r in login_recent(user_id, LOGINS)]
    return _take(rows, KINDS["login_log"].columns, offset, limit, q)


def _sessions(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    from .accounts import SESSION_KINDS, online_now

    rows = [{**r, "kind": SESSION_KINDS[r["kind"]]} for r in online_now(user_id)]
    return _take(rows, KINDS["sessions"].columns, offset, limit, q)


def _admin_actions(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    """某个管理员自身的操作记录，按时间倒序（server/access.py audit()；该表只增不减）。语句按消息目录在此处
    生成：页面从不拼接文字，其中来自用户的内容（用户名、环节）作为文本处理，不作为标记。"""
    from .database import db, json_of
    from .messages import render

    rows = []
    for r in db().rows("SELECT * FROM admin_actions WHERE user_id = ? ORDER BY at DESC LIMIT ?", (user_id, ACTIONS)):
        params = json_of(r["params"], {})
        try:
            # 该行写入后模板可能增加或减少了参数：模板新需要而行中缺少的参数留空，行中多余的参数丢弃，
            # 使记录仍能读成完整的语句
            wanted = names(template(r["code"]))
            what = render(r["code"], {**{k: "" for k in wanted}, **{k: v for k, v in params.items() if k in wanted}})
            # 页面本身即针对该账号，因此语句省略开头的操作者（「谁（角色）」）
            head = f"{params.get('who', '')}（{params.get('role', '')}）"
            what = what[len(head):].lstrip() if head != "（）" and what.startswith(head) else what
        except Exception:  # 目录中已不存在的代码或参数：记录本身不会丢失
            what = r["code"]
        if r["repeats"] > 1:  # the same refusal again and again, counted on this row (server/access.py audit)
            what += render("I-AUDIT-REPEATED", {"count": r["repeats"], "last": time.strftime("%m-%d %H:%M", time.localtime(r["last"] or r["at"]))})
        rows.append({"at": r["at"], "what": what, "code": r["code"],
                     "where": f"{r['method']} {r['path']}".strip(), "status": r["status"]})
    return _take(rows, KINDS["admin_actions"].columns, offset, limit, q)


def _templates(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    """该账号保存在服务器上的节点图（「我的模板」和已录入的），包括用户自行删除的，这些行灰显并标注「用户已删除」。"""
    from . import accounts
    from .library import rows_for_account

    return _take(rows_for_account(accounts.get(user_id).username), KINDS["templates"].columns, offset, limit, q)


def _traffic(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    """该账号每天从服务器取走的数据量（traffic.py）。「用户」栏给出今天 / 近 7 天 / 总计，
    此处按天列出：公网通过 frp 按流量计费，需要能查出哪天用量较大。"""
    from .traffic import per_day

    return _take(per_day(user_id), KINDS["traffic"].columns, offset, limit, q)


def _tasks(user_id: int, offset: int, limit: int, q: str) -> tuple[int, list[dict]]:
    """该账号的任务文件夹（transfer/tasks.py）：任务号、是否已结束、占多大（配额按它算）、何时提交与结束。"""
    from .transfer.tasks import of_account

    return _take(of_account(user_id), KINDS["tasks"].columns, offset, limit, q)


HISTORY = 500  # 登记表回溯查看的单个账号的任务数
LOGINS = 200
ACTIONS = 1000  # 登记表回溯查看的单个账号的管理操作数
MOST = 5000  # 单次筛选列表在截取一页之前最多检查的行数

REGISTRY: tuple[Resource, ...] = (
    Resource("jobs", "任务", "jobs", "queue.manage", "queue",
             (("id", "任务"), ("title", "节点图"), ("targets", "算的节点"), ("state", "状态"), ("submitted", "提交"), ("finished", "结束")), _jobs, when="submitted", state="state"),
    Resource("tasks", "任务文件夹", "tasks", "data.others", "disk",
             (("id", "任务"), ("state", "状态"), ("bytes", "大小", "size"), ("created", "提交"), ("ended", "结束")), _tasks,
             when="created", state="state"),
    Resource("uploads", "上传的素材", "", "data.others", "disk",
             (("key", "素材"), ("kind", "种类"), ("at", "上传")), _uploads, when="at", state="kind"),
    Resource("feedback", "反馈", "feedback", "feedback.reply", "feedback",
             (("id", "反馈"), ("category", "类别"), ("text", "内容"), ("status", "状态"), ("at", "提交")), _feedback, when="at", state="status"),
    Resource("outputs", "输出的结果", "", "data.others", "disk",
             (("name", "结果"), ("task", "任务"), ("label", "节点"), ("count", "文件数"), ("bytes", "大小", "size"), ("finished", "打包")),
             _outputs, when="finished"),
    Resource("login_log", "登录记录", "login_log", "logins.view", "users",
             (("at", "时间"), ("ok", "成功"), ("reason", "原因"), ("kind", "方式"), ("ip", "地址"), ("device", "设备")), _logins, when="at", state="kind"),
    Resource("sessions", "在线的登录", "sessions", "logins.view", "users",
             (("kind", "方式"), ("device", "设备"), ("hostname", "计算机"), ("ip", "地址"), ("seen", "最近活动")), _sessions, when="seen", state="kind"),
    # 我的模板：用户自行保存的节点图；管理员在此可查看其全部模板，
    # 用户删除的模板灰显并标注「用户已删除」，可恢复或永久删除（按钮所需权限写在其调用的路由上）。
    # 页签本身只要求最低的一项权限（templates.restore）：二级管理员需要能进入此处以帮助用户找回误删的模板，
    # 移入回收站和永久删除的按钮按其各自路由的 data.others 计算，结果为灰显。
    Resource("templates", "模板", "", "templates.restore", "users",
             (("id", "模板"), ("name", "名字"), ("state", "状态"), ("bytes", "大小", "size"), ("updated", "最近改动")), _templates,
             when="updated", state="state", dim=True,
             acts=(Act("restore", "恢复", "把这张模板从回收站拿回来：用户又能在「我的模板」里看到它",
                       "POST /api/admin/graphs/{gid}/restore"),
                   Act("bin", "放进回收站", "先放进回收站，不是删掉：用户看不到它了，随时能恢复",
                       "POST /api/admin/graphs/{gid}/bin"),
                   Act("purge", "永久删除", "彻底删掉这张模板：删了就找不回来了",
                       "DELETE /api/admin/graphs/{gid}", danger=True))),
    # 流量：每天一行。「用户」栏和用户页上方的「网络流量」给出今天 / 近 7 天 / 总计，本页签为其明细
    Resource("traffic", "流量", "traffic", "users.manage_normal", "users",
             (("day", "日期"), ("bytes", "流量", "size")), _traffic),
    # 管理操作：仅对具有管理职能的账号显示，「普通用户」没有该页签
    Resource("admin_actions", "管理操作", "admin_actions", "audit.view", "logs",
             (("at", "时间"), ("what", "操作"), ("where", "接口"), ("status", "结果")), _admin_actions,
             when="at", state="status", for_admins=True),
)

KINDS: dict[str, Resource] = {r.id: r for r in REGISTRY}


def manages(user_id: int) -> bool:
    """该账号具有管理职能：其角色至少拥有一项能力（「管理操作」只涉及此类账号；「普通用户」在该表中没有记录）。"""
    from . import roles
    from .accounts import get

    try:
        return bool(roles.capabilities(get(user_id).role))
    except Exception:
        return False


def kinds(capabilities: frozenset[str] | set[str], user_id: int | None = None) -> list[Resource]:
    """具有这些能力的会话可以查看的资源，按登记顺序。`user_id`：被查看的账号；声明了 `for_admins` 的资源
    在该账号不具备任何管理职能时被排除。"""
    return [r for r in REGISTRY if r.needs in capabilities and not (r.for_admins and user_id is not None and not manages(user_id))]


def get(kind: str, capabilities: frozenset[str] | set[str], user_id: int | None = None) -> Resource:
    found = KINDS.get(kind)
    if found is None or found.needs not in capabilities or (found.for_admins and user_id is not None and not manages(user_id)):
        raise NotFound(Msg("E-RESOURCE-NOKIND", kind=kind))
    return found


def page(kind: str, user_id: int, capabilities: frozenset[str] | set[str], offset: int = 0, limit: int = 50,
         q: str = "", since: float = 0.0, state: str = "") -> dict:
    """某账号某资源的一页：列、「时间」和「状态」筛选读取的列、可选的状态以及行。站点的每个页面都读取这一结果，
    因此「用户」详情页与资源自身的列表页显示相同的内容，且都不为某种资源单独编写代码。"""
    r = get(kind, capabilities, user_id)
    limit = max(min(limit, 500), 1)
    offset = max(offset, 0)
    _, found = r.rows(user_id, 0, MOST, q.strip())
    states = _states(found, r.state)
    kept = [row for row in found
            if (not since or _at_least(row.get(r.when), since)) and (not state or str(row.get(r.state)) == state)]
    total, rows = _cut(kept, offset, limit)
    return {"kind": r.id, "label": r.label, "section": r.section,
            "columns": [{"key": c[0], "label": c[1], "says": c[2] if len(c) > 2 else ""} for c in r.columns],
            "when": r.when, "state_column": r.state, "states": states, "dim": r.dim,
            "acts": [a.json() for a in r.acts], "total": total, "offset": offset, "rows": rows}


def _at_least(value: object, since: float) -> bool:
    """时间戳列的值不早于 `since`（没有时间戳的行保留：它没有日期，不视为过期）。"""
    return not isinstance(value, (int, float)) or float(value) >= since


def _states(rows: list[dict], column: str) -> list[str]:
    """「状态」筛选提供的值，按首次出现的顺序（资源未声明该列时为 []）。"""
    if not column:
        return []
    out: list[str] = []
    for row in rows:
        value = row.get(column)
        text = "" if value is None else str(value)
        if text and text not in out:
            out.append(text)
    return out


def counts(user_id: int, capabilities: frozenset[str] | set[str]) -> list[dict]:
    """某账号详情页的页签：本会话可查看的每种资源，及该账号在其中的数量。"""
    return [{"kind": r.id, "label": r.label, "section": r.section, "count": r.rows(user_id, 0, 1, "")[0]}
            for r in kinds(capabilities, user_id)]
