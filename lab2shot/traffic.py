"""每个账号使用的网络流量（后台「用户」页签和编辑器的「队列」窗口读取）。

用途：服务通过 frp 暴露到公网时 frp 按流量计费，因此需要掌握各账号的用量。流量的主要部分不是交付物，而是视图数据
（一段 150 帧 1080p 的深度点云完整加载约 136 MB）。因此每个响应的正文字节都计入：取帧、场景数据分块、
交付物下载、目录和接口的响应，一律按实际发送的字节计算。

统计的是实际发出的字节：计数层位于服务器最外层（server/traffic.py `Meter`，即 server/app.py 最后添加的那一层），
此时压缩已完成，计得的即线上传输的字节。响应头不计入（每条数百字节，与正文不在一个量级）。
本文件只负责计数、存储和读取，不涉及 HTTP，因此可在服务器层以下使用（lab2shot/resources.py 的「流量」页签即读取它；
分层规则：本包不 import server）。

开销很小：每个请求不访问数据库，按 (账号, 日期) 在内存中累加（`_PENDING`），最多每 `FLUSH_S` 秒写盘一次，服务停止时
（进程正常退出、排空重启）再写盘一次。读取时也不写盘：将内存中累积的量加到已写盘的量上，因此页面上的数字始终是
当前值，也不会因为队列窗口每 1.5 秒查询一次而变成每个请求写一次盘。服务被强制终止（断电、kill -9）时会丢失最后
不到一分钟的计数。

可见范围：与磁盘占用规则相同（server/quota.py），管理员在后台「用户」中查看所有账号，普通账号只在编辑器的
「队列」窗口中查看自己的用量。权限由路由自身声明，此处不做角色判断。
"""

from __future__ import annotations

import atexit
import threading
import time

from .database import db

FLUSH_S = 60.0  # 内存中的计数最多累积这么久即写盘（重启时丢失的即为这一段）
SCOPE_USER = "lab2shot_user"  # server/access.py Guard 将本次请求的归属写在 ASGI 的 scope 上，server/traffic.py Meter 读取

_LOCK = threading.Lock()
_PENDING: dict[tuple[int, str], int] = {}  # (账号, 日期) -> 尚未写盘的字节数
_LAST = 0.0


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def count(user_id: int, sent: int) -> bool:
    """该账号又发出了 `sent` 字节。只访问内存；返回 True 表示已累积足够长时间、应写盘一次。写盘（`flush`，一次 sqlite 写入）
    由调用方在线程中执行（server/traffic.py Meter 使用 `asyncio.to_thread`），不在事件循环上执行：在此直接写盘会使
    整个服务器的事件循环阻塞在磁盘写入上。未登录的请求（user_id 0）不计入。"""
    if not user_id or sent <= 0:
        return False
    global _LAST
    key = (user_id, _today())
    with _LOCK:
        _PENDING[key] = _PENDING.get(key, 0) + sent
        due = time.time() - _LAST > FLUSH_S
        if due:
            _LAST = time.time()  # 只有一个调用方得到 True：同一时刻的其他请求不再各自启动线程写盘
    return due


def flush() -> None:
    """将累积的计数写入数据库（读取前、定时、退出时各一次）。"""
    global _LAST
    with _LOCK:
        due = dict(_PENDING)
        _PENDING.clear()
        _LAST = time.time()
    if not due:
        return
    with db().write() as c:
        c.executemany("INSERT INTO traffic (user_id, day, bytes) VALUES (?, ?, ?) "
                      "ON CONFLICT (user_id, day) DO UPDATE SET bytes = bytes + excluded.bytes",
                      [(u, day, n) for (u, day), n in due.items()])


def _flush_at_exit() -> None:
    """进程退出时再写盘一次。此时数据库可能已关闭：流量是尽力而为的计数，写入失败时丢弃这一小段，
    不得在退出路径上抛出异常（真正的错误在别处报告，此处报告也无人可见）。"""
    try:
        flush()
    except Exception:  # noqa: BLE001 - 退出路径：任何原因都不该改变退出结果
        pass


atexit.register(_flush_at_exit)  # 进程正常退出时不丢失（排空重启也经过此处）


def _since(days: int) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(time.time() - (days - 1) * 86400))


def _rows(user_id: int | None) -> list:
    """已写盘的量加上内存中累积的量。读取时不写数据库：队列窗口每 1.5 秒查询一次自身占用，为此写盘等于
    每个请求写一次。加上内存中的部分后，数字仍为当前值。"""
    kept = (db().rows("SELECT user_id, day, bytes FROM traffic") if user_id is None
            else db().rows("SELECT user_id, day, bytes FROM traffic WHERE user_id = ?", (user_id,)))
    with _LOCK:
        waiting = [{"user_id": u, "day": d, "bytes": n} for (u, d), n in _PENDING.items()
                   if user_id is None or u == user_id]
    return [*kept, *waiting]


def _add(into: dict, row) -> None:
    box = into.setdefault(row["user_id"], {"today": 0, "week": 0, "total": 0})
    box["total"] += row["bytes"]
    if row["day"] >= _since(7):
        box["week"] += row["bytes"]
    if row["day"] == _today():
        box["today"] += row["bytes"]


EMPTY = {"today": 0, "week": 0, "total": 0}


def of(user_id: int) -> dict:
    """账号发出的字节数：今天、近 7 天（含今天）、总计。"""
    found: dict[int, dict] = {}
    for row in _rows(user_id):
        _add(found, row)
    return found.get(user_id, dict(EMPTY))


def per_day(user_id: int) -> list[dict]:
    """该账号每天一行，最近的在前（后台用户页的「流量」页签：可查出哪天用量较大）。"""
    found: dict[str, int] = {}
    for row in _rows(user_id):
        found[row["day"]] = found.get(row["day"], 0) + row["bytes"]
    return [{"day": day, "bytes": n} for day, n in sorted(found.items(), reverse=True)]


def of_users() -> dict[int, dict]:
    """每个账号发出的字节数（后台「用户」栏中每个账号一行）。"""
    found: dict[int, dict] = {}
    for row in _rows(None):
        _add(found, row)
    return found
