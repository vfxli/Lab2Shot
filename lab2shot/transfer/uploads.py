"""使用者上传的文件：画面、视频、相机文件。每个账号各自一份，按内容保存：同一账号相同的字节无论上传多少次都只存一份，
不同账号之间从不共用、从不去重（用户隔离）。

    <数据位置>/uploads/<account>/blobs/<sha[:2]>/<sha>      该账号上传的一个文件的字节（sha256），写完即只读
    <数据位置>/uploads/<account>/sets/<id>/                节点所见的一份上传：其文件以各自的名称存放（硬链接到 blob），
                                                         因此所有读取方都按普通路径处理
    <数据位置>/uploads/<account>/sets/<id>.json            {"name", "files": {name: sha}, "origins": [上传者及来源]}
    <数据位置>/uploads/<account>/sets/<id>.declared.json / .layers.json / .head
                                                         字节到达之前已申报的上传（见下文相应部分）
    <数据位置>/uploads/<account>/parts/<id>                正在上传的文件：已收到的字节，包括中断的请求所传的部分
    <数据位置>/uploads/<account>/parts/<id>.json           增长期间为 {"size", "who"（账号）, "time"}
    <数据位置>/uploads/<account>/parts/done/<id>.json      同上，另加 "sha" 和 "done"：其字节已成为 blob（保留 DONE_KEEP_S，
                                                         以便途中丢失的响应可再次查询）
    <数据位置>/uploads/<account>/subsets/<sha[:2]>/<sha>.json
                                                         该账号对一个原文件的通道子集（见「通道级上传」）

文件分段上传：open_part(size)，然后每个请求依次调用 begin(offset) / add(chunk) / end()；
连接中断不造成损失，发送方调用 part_state() 后从该字节继续。完整的分段在 end() 中成为该账号的 blob，这是上传完成时
唯一经过的位置。blob 写完即设为只读：它会被硬链接到上传文件夹和任务文件夹（transfer/tasks.py footage/），
任何名字的内容都不能原地改动，否则所有名字看到的内容都会跟着变。

节点参数以 "upload:<id>/<name>" 引用上传，<name> 为上传中的文件或序列模式（plate.####.exr）。
id 由文件内容和名称计算得出：同一账号再次上传相同的文件得到相同的引用。

归属：上传位于某账号的文件夹（`uploads/<account>/`）即属于该账号，此外不做任何记录（没有与磁盘不一致的表）。
两个账号上传相同的文件得到相同的 id（由内容计算）、两个文件夹、两份字节，各自只链接该账号自己发送的字节：
一方之后发送的内容不会改变另一方读取的内容，一方也无法通过「上传瞬间完成」推断另一方有没有这个文件（`kept` 只看
本账号的 blob）。决定账号可读取内容的位置只有一处：`resolve()`，所有读取方都经过它（节点经过 PlanEnv.upload，见
lab2shot/catalog.py）。它针对当前服务的账号回答（lab2shot/serving.py，一个 Account）：其他账号的上传视为不存在，
与已被清理的上传完全相同：相同的消息、相同的节点级失败，且其文件不会被打开，甚至不用于识别。上层不单独询问归属。
仅知道 sha 或上传 id 也不够（kept、has_blob、part_state 同样需要账号）。

保留：选文件时上传的素材先归属于节点图；任务提交时，其节点图引用的上传被硬链接进任务文件夹（transfer/tasks.py），
并记入任务（task_uploads）。任务还在时它引用的上传一直保留；没有任何任务用到的上传，最后一次使用超过
任务保留天数后清理（farm/disk.py）。blob 只在没有任何名字链接它时（nlink 为 1）才删除，因此任务文件夹里的
一份硬链接仍在时，同一账号再选同一文件无需重新上传。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data.packet import used
from ..errors import Invalid, NotFound
from ..io.atomic import write_text
from ..io.files import folder_bytes, link_or_copy, mtime, stats
from ..io.sequence import IMAGE_EXTS, FrameSequence, find_sequence, is_pattern, sequence_of
from ..messages import Msg
from ..serving import account
from . import BODY_MAX, relative_name

PREFIX = "upload:"
ID_LEN = 20
_ID = re.compile(rf"^[0-9a-f]{{{ID_LEN}}}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def root() -> Path:
    """素材存放位置：设置中「数据位置」（config.py paths.data_dir，生产环境指向数据盘）里的 uploads，每个账号一个文件夹。"""
    from ..config import settings

    return settings().data_dir / "uploads"


ORIGINS_KEPT = 20  # 一份上传的来源（清单中的 "origins"）：无论重复发送多少次，只保留最新的这么多条


def home_of(user_id: int) -> Path:
    """某账号的全部上传：`uploads/<账号 id>/`。这是账号上传位置的唯一命名处。"""
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise NotFound(GONE)
    return root() / str(int(user_id))


def blob_path(sha: str, user_id: int) -> Path:
    """该账号的一份字节。"""
    if not _SHA.match(sha):
        raise Invalid(Msg("E-UPLOAD-BADSHA", sha=sha))
    return home_of(user_id) / "blobs" / sha[:2] / sha


def of_account(user_id: int) -> list[dict]:
    """某账号的上传，按时间倒序（lab2shot/resources.py 的登记表用它列出「上传的素材」）：每项为 {key, kind, at}，
    即一份上传（其在账号 `sets/` 中的文件夹，key 为 upload:<id>/<name>）。"""
    out = []
    for manifest in _manifests(_sets(user_id)):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            out.append({"key": f"{PREFIX}{manifest.stem}/{data.get('name') or ''}", "kind": "上传", "at": manifest.stat().st_mtime})
        except (OSError, ValueError):
            pass
    return sorted(out, key=lambda r: -float(r["at"] or 0))


def _sets(user_id: int) -> Path:
    """某账号的上传：`uploads/<账号id>/sets/`。上传的归属即其所在位置（见模块说明）。"""
    return home_of(user_id) / "sets"


def accounts() -> list[int]:
    """有上传文件夹的每个账号（清理按账号逐个进行）。"""
    base = root()
    return sorted(int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit() and int(d.name) > 0) if base.is_dir() else []


def _accounts() -> list[Path]:
    """各账号的 `sets/` 文件夹。"""
    return [d for d in (_sets(u) for u in accounts()) if d.is_dir()]


def _manifests(home: Path) -> list[Path]:
    """某账号文件夹中各上传的清单（`<id>.json`），不含申报的附属文件。"""
    return sorted(m for m in home.glob("*.json") if not m.name.endswith(DECLARED_SIDE) and _ID.match(m.stem)) if home.is_dir() else []


def _has(home: Path, sid: str) -> bool:
    return (home / sid).is_dir() or (home / f"{sid}.declared.json").is_file()


def _home(sid: str) -> Path | None:
    """针对当前服务的账号（serving()），存放上传 `sid`（已组装，或已申报并等待字节）的账号 `sets/` 文件夹；没有时为 None，
    与已清理的上传给出相同的回答。服务所有账号的管理员（all_accounts）可访问任何账号的上传，优先查找自己的；
    没有指明账号时（serving.ANYONE）什么都找不到。
    这是查找上传所属账号的唯一位置：resolve、declared、describe 以及 head/plane 辅助函数都经过此处。"""
    if not _ID.match(sid):
        return None
    who = account()
    if who.user_id > 0:
        mine = _sets(who.user_id)
        if _has(mine, sid):
            return mine
    if who.all_accounts:
        for d in _accounts():
            if _has(d, sid):
                return d
    return None


def set_folder(sid: str) -> Path | None:
    """上传 `sid` 已组装的文件夹，针对当前服务的账号（与 `resolve` 同一规则：其他账号的视为不存在）；没有时为 None。
    任务提交时由此找到要硬链接进任务文件夹的素材（transfer/tasks.py）。"""
    home = _home(sid)
    folder = home / sid if home is not None else None
    return folder if folder is not None and folder.is_dir() else None


def has_blob(sha: str, user_id: int) -> bool:
    """该账号是否持有这些字节（亲自上传过，仍在服务器上）。"""
    return blob_path(sha, user_id).is_file()


def _readonly(path: Path) -> None:
    """写完的字节从此只读：它们会被硬链接（上传文件夹、任务文件夹），任何名字都不得原地改动。"""
    try:
        os.chmod(path, 0o444)
    except OSError:
        pass


GONE = Msg("E-UPLOAD-GONE")


# ------------------------------------------------------------------ 分段：正在上传的文件

PART_ID = re.compile(r"^[0-9a-f]{32}$")
PART_KEEP_DAYS = 3  # 超过此时长无人追加的分段被丢弃（其文件从头重新上传）
DONE_KEEP_S = 24 * 3600  # 已完成的分段在此时长内仍报告其 sha（途中丢失的响应可再次查询）
PARTS_PER_CLIENT = 64  # 单个账号同时打开的分段数；超出时丢弃最早的


class Moved(Exception):
    """分段不在请求声明的起始位置（期间有其他请求向其追加）：实际位置为 `offset`。"""

    def __init__(self, offset: int) -> None:
        super().__init__(f"这个文件已经传到第 {offset} 字节")
        self.offset = offset


class _Growing:
    """本进程对增长中分段的了解：其累计 sha256，以及当前可以向其追加的请求。"""

    def __init__(self, offset: int, size: int) -> None:
        self.sha = hashlib.sha256() if offset == 0 else None  # 重启前的分段：在结束时回读计算
        self.size = size
        self.writer = ""


_growing: dict[str, _Growing] = {}
_lock = threading.Lock()


def _parts(who: int) -> Path:
    """某账号分段的存放位置。每个账号有自己的文件夹，成为 blob 的分段会将其记录移入旁边的 `done`，
    因此打开文件时只读取该账号仍在上传的分段，最多 PARTS_PER_CLIENT 个，与服务器上的其他内容无关。

    若所有记录位于同一个平铺文件夹中，打开分段就要读取全部记录：开销随服务器上的记录数增长
    （2000 条时约 18 ms，并行上传读取同样的记录时更多），长序列的上传将是 O(N²)，远低于线路的带宽。"""
    return home_of(who) / "parts"


def _done(who: int) -> Path:
    """该账号已完成分段的记录在 DONE_KEEP_S 期间的存放位置（此时其字节已属于 blob）。"""
    return _parts(who) / "done"


def _part_file(pid: str, who: int) -> Path:
    if not PART_ID.match(pid):
        raise NotFound(Msg("E-UPLOAD-NOPART"))
    return _parts(who) / pid


def _record(pid: str, who: int) -> dict | None:
    """分段的记录：上传期间位于其字节旁边，成为 blob 后位于 `done` 中（None：两处都没有）。
    它在请求可能正在读取时从一处移到另一处（_finish 先写入完成的记录再移除另一份），因此两处都要检查。"""
    for path in (_part_file(pid, who).with_suffix(".json"), _done(who) / f"{pid}.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def part_state(pid: str, who: int) -> dict:
    """分段的进度：{id, offset, size}，完整后另含其 sha。只有发送它的账号可以查询：其他账号的分段不在
    其文件夹中，因此视为不存在。"""
    r = _record(pid, who)
    if r is None:
        raise NotFound(Msg("E-UPLOAD-PARTGONE"))
    if r.get("sha"):
        return {"id": pid, "offset": r["size"], "size": r["size"], "sha": r["sha"]}
    try:  # 此文件夹的约定是读取时已不存在即视为不存在：同账号的另一个请求可能恰好将其剪除，
        # 先 exists() 再 stat() 之间可能落空。落空时视为尚未写入任何字节，不得将 FileNotFoundError 抛给请求
        written = _part_file(pid, who).stat().st_size
    except OSError:
        written = 0
    return {"id": pid, "offset": written, "size": r["size"]}


def open_part(size: int, who: int, pid: str = "") -> dict:
    """为 `size` 字节的文件新建一个分段，由账号 `who` 发送（其最早打开的分段超出 PARTS_PER_CLIENT 时被丢弃）。
    `pid`：发送方为其生成的名称（随机，32 位十六进制），在发送第一个字节之前即已确定：在收到响应之前中断的请求
    仍会留下发送方可以再次找到的分段。同名分段已存在时即为同一个（请求被重发）；其他账号的同名分段视为不存在。"""
    if size < 0:
        raise Invalid(Msg("E-UPLOAD-BADSIZE", size=size))
    if pid:
        _part_file(pid, who)  # 校验名称格式
        known = _record(pid, who)
        if known is not None:
            if known["size"] != size:
                raise NotFound(Msg("E-UPLOAD-NOPART"))
            return part_state(pid, who)
    _parts(who).mkdir(parents=True, exist_ok=True)
    mine = _going_up(who)
    for _, old in mine[: max(0, len(mine) - PARTS_PER_CLIENT + 1)]:
        _drop(old, who)
    pid = pid or uuid.uuid4().hex
    _part_file(pid, who).touch()
    write_text(_part_file(pid, who).with_suffix(".json"), json.dumps({"size": size, "who": who, "time": time.time()}))
    with _lock:
        _growing[pid] = _Growing(0, size)
    # 刚建好的分段，其状态即为「0 字节，共 size 字节」：无需再读一次磁盘。同账号的另一个请求可能恰好在这
    # 两步之间把它当作最早的分段剪除（上面的循环剪除的正是其他分段），再读一次就会对自己刚建好的分段
    # 报告「这份上传不在了」，四个请求同时打开时即可能发生。
    return {"id": pid, "offset": 0, "size": size}


def _going_up(who: int) -> list[tuple[float, str]]:
    """该账号仍在上传的分段，最早的在前：(最后写入时间, id)。

    执行期间任何记录都可能消失（同账号的另一个请求完成或丢弃了它，或清理任务剪除了它），因此该文件夹的约定是：
    读取时已不存在的记录不出现在结果中。此处不得假定列出的名称仍是文件。"""
    out: list[tuple[float, str]] = []
    for record in _parts(who).glob("*.json"):
        try:
            if not json.loads(record.read_text(encoding="utf-8")).get("sha"):
                out.append((record.stat().st_mtime, record.stem))
        except (OSError, ValueError):
            continue  # 已消失或写了一半：不是该账号仍在上传的分段
    return sorted(out)


def _drop(pid: str, who: int) -> None:
    with _lock:
        _growing.pop(pid, None)
    _part_file(pid, who).unlink(missing_ok=True)
    _part_file(pid, who).with_suffix(".json").unlink(missing_ok=True)
    (_done(who) / f"{pid}.json").unlink(missing_ok=True)


def begin(pid: str, offset: int, who: int) -> str:
    """一个请求从 `offset` 开始向分段追加（分段在其他位置时抛出 Moved）；返回其令牌。最新的请求拥有该分段：
    来自未被察觉即已中断的连接（隧道可能保持其打开）的请求停止追加。只有发送它的账号可以追加（它只存在于该账号的文件夹中）。"""
    state = part_state(pid, who)
    if state.get("sha") or offset != state["offset"]:
        raise Moved(state["offset"])
    token = uuid.uuid4().hex
    with _lock:
        g = _growing.setdefault(pid, _Growing(offset, state["size"]))
        g.writer = token
    return token


def add(pid: str, token: str, chunk: bytes, who: int) -> int:
    """追加请求带来的字节（仅限其拥有者）；返回分段当前的大小。"""
    f = _part_file(pid, who)
    with _lock:
        g = _growing.get(pid)
        if g is None or g.writer != token:
            raise Moved(f.stat().st_size if f.exists() else 0)
        have = f.stat().st_size
        if have + len(chunk) > g.size:
            raise Invalid(Msg("E-UPLOAD-OVERSIZE", size=g.size))
        try:
            with f.open("ab") as out:
                out.write(chunk)
        except OSError:
            g.sha = None  # 文件可能只含该块的一部分：其 sha 在 _finish 中从保留的字节回读计算，不再增量计算
            raise
        if g.sha is not None:
            g.sha.update(chunk)
        return have + len(chunk)


def end(pid: str, token: str, who: int) -> dict:
    """请求结束（其带来的内容均已保存，连接中断时亦然）：完整的分段成为其 blob。"""
    with _lock:
        g = _growing.get(pid)
        mine = g is not None and g.writer == token
        if mine:
            g.writer = ""
    state = part_state(pid, who)
    if not mine or state.get("sha") or state["offset"] < state["size"]:  # 只有最后的写入者才能完成它
        return state
    return {**state, "sha": _finish(pid, who)}


def _finish(pid: str, who: int) -> str:
    f = _part_file(pid, who)
    with _lock:
        g = _growing.pop(pid, None)
    if g is not None and g.sha is not None:
        sha = g.sha.hexdigest()
    else:  # 跨越重启增长的分段：回读一次其字节
        h = hashlib.sha256()
        with f.open("rb") as src:
            while block := src.read(1 << 24):
                h.update(block)
        sha = h.hexdigest()
    target = blob_path(sha, who)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():  # 该账号之前发送过这些字节：只存一份（从不用其他账号的）
        f.unlink()
    else:
        os.replace(f, target)
        _readonly(target)
    # 记录移出 `open_part` 读取的文件夹，因此已完成的分段不会拖慢下一个文件
    going = f.with_suffix(".json")
    _done(who).mkdir(parents=True, exist_ok=True)
    write_text(_done(who) / f"{pid}.json", json.dumps({**json.loads(going.read_text(encoding="utf-8")), "sha": sha, "done": time.time()}))
    going.unlink(missing_ok=True)
    return sha


def kept(shas: list[str], who: int) -> list[str]:
    """账号之前发送过且服务器仍保存的内容（浏览器在发送之前询问哪些内容已发送过，有的就不再上传）。只看该账号自己的
    字节：其他账号有没有同样的内容不予告知，也无从推断（「上传瞬间完成」只对自己传过的内容发生）。"""
    return [s for s in shas if _SHA.match(s) and has_blob(s, who)]


def prune_parts() -> int:
    """丢弃 PART_KEEP_DAYS 内无人追加的分段，以及超过 DONE_KEEP_S 的已完成分段记录；返回数量。记录可能在此期间
    消失（各账号仍在上传）：已消失的记录即少丢弃一个，不属于错误。"""
    now, gone = time.time(), 0
    for who in accounts():
        folder = _parts(who)
        if not folder.is_dir():
            continue
        for record in list(folder.glob("*.json")) + list((folder / "done").glob("*.json")):
            try:
                r = json.loads(record.read_text(encoding="utf-8"))
                bytes_at = folder / record.stem
                last = max(record.stat().st_mtime, bytes_at.stat().st_mtime if bytes_at.exists() else 0)
            except (OSError, ValueError):
                continue
            if now - (r.get("done", 0) if r.get("sha") else last) > (DONE_KEEP_S if r.get("sha") else PART_KEEP_DAYS * 86400):
                _drop(record.stem, who)
                gone += 1
    return gone


def _name(name: str) -> str:
    return str(relative_name(name))


def set_id(name: str, files: dict[str, str]) -> str:
    """一份上传的 id：只由其中包含的文件、各文件的内容以及节点读取的对象计算得出
    （sha256 的前 20 位十六进制）。再次上传相同的文件得到同一个引用。

    全项目只有此处计算该值：`make_set`（字节到齐、组装文件夹）
    和 `declare_set`（字节尚未上传、先申报）必须得出同一个值，
    否则在「先申报、后传字节」的流程中，参数中的引用会与最终组装的文件夹不一致。

    网页端也会预先计算同样的值（以便参数立即写入引用），但以此处的计算为准：
    申报的响应中带有服务器计算的 `ref`，网页按其写入。 """
    files = {_name(k): v for k, v in files.items()}
    key = json.dumps([_name(name) if name else "", sorted(files.items())], ensure_ascii=False)
    return hashlib.sha256(key.encode()).hexdigest()[:ID_LEN]


def make_set(name: str, files: dict[str, str], origin: dict, user_id: int) -> str:
    """将某账号一起上传的 blob 按名称组装；返回节点使用的引用，此后属于该账号。`name`：节点读取的对象
    （`files` 中的一个文件、一个序列模式，或 "" 表示整份上传作为文件夹）。`origin`：上传者及其在本机上的名称（留作记录）。"""
    files = {_name(k): v for k, v in files.items()}
    if not files:
        raise Invalid(Msg("E-UPLOAD-NOFILES"))
    missing = [k for k, sha in files.items() if not available(sha, user_id)]
    if missing:
        raise Invalid(Msg("E-UPLOAD-NOTSENT", files=missing[:5]))
    sid = set_id(name, files)
    folder = _sets(user_id) / sid
    # 每个文件链接哪份字节：完整 blob，或其当前的通道子集（见「通道级上传」部分）。文件夹属于该账号，
    # 其中每个链接都由这一行按该账号所持有的字节确定：其他账号上传的内容不会进入该账号的文件夹。
    sources = {rel: _blob_for(sha, user_id) for rel, sha in files.items()}
    if not folder.is_dir():
        folder.parent.mkdir(parents=True, exist_ok=True)
        building = folder.with_name(f".{sid}.{uuid.uuid4().hex[:8]}")
        for rel, src in sources.items():
            target = building / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(src, target)
        try:
            os.rename(building, folder)
        except OSError:  # 其他请求同时组装了相同的集合
            shutil.rmtree(building, ignore_errors=True)
    else:
        # 子集扩大（又接入了新通道，并集换成了新 blob）：文件夹中的硬链接原地替换为新的。
        # `os.replace` 是原子操作：读取方要么看到旧的完整文件，要么看到新的完整文件，不会看到一半。
        for rel, src in sources.items():
            target = folder / rel
            try:
                if target.exists() and os.path.samefile(target, src):
                    continue
            except OSError:
                pass
            fresh = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}")
            target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(src, fresh)
            os.replace(fresh, target)
    manifest = folder.with_suffix(".json")
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {"name": name, "files": files, "origins": []}
    data["origins"] = [*data["origins"], {**origin, "time": time.time()}][-ORIGINS_KEPT:]
    # 哪些文件链接的是子集 EXR（`declared_layers_for` 据此得知端口需按申报的原文件生成）
    subsets = {}
    for rel, sha in files.items():
        if not has_blob(sha, user_id) and (rec := subset_of(sha, user_id)) is not None:
            subsets[rel] = {"channels": rec["channels"], "blob": rec["blob"]}
    if subsets:
        data["subsets"] = subsets
    else:
        data.pop("subsets", None)
    write_text(manifest, json.dumps(data, ensure_ascii=False, indent=1))
    return f"{PREFIX}{sid}/{name}"


# ------------------------------------------------------------------ 先申报，后传字节
#
# 目的：选择文件后不应立即上传并显示计算中；只有某个通道接入了计算节点并需要计算时，
# 才按连线用到的通道上传。
#
# 约束：读取节点的输出端口由服务器打开文件后统计得出（nodes/core/input.py made_ports
# → base.py ReadsFile.path → 本文件的 resolve）。服务器没有该文件就无法生成任何端口，
# 使用者也就无法连接通道，不上传与有端口之间形成死结。
#
# 解决方法：EXR 的头位于文件最前面，`data/layers.py describe_file` 只读头即可统计出全部图层
# （例如一个 9 个图层、共 2 347 698 字节的多层 EXR，前 8192 字节即可统计出全部图层）。因此：
#
#   申报（选择文件时）：网页上传「该上传包含哪些文件、各自内容的 sha256」以及第一帧开头的几十 KB。
#   服务器计算该上传的 id（`set_id`，与 `make_set` 使用同一公式），记录清单，
#   并用自身的 `describe_file` 读出图层保存。线路上只传输几十 KB，且描述文件的代码只有一份
#   （不在浏览器中重新实现）。
#
#   上传字节（需要计算时）：需要哪些通道由 `engine/evaluation.py needed_outputs` 决定，
#   网页上传仍缺少的部分，最后照常由 `make_set` 组装文件夹。
#
# 上传仍然不可变且按内容寻址：申报不创建 `sets/<id>/` 文件夹，只写两份附属说明文件。
# 文件夹只在字节到齐时由 `make_set` 一次组装完成，组装后即完整且不再改变，
# 因此缓存指纹（`NodeDef.source_identity`）及 `_opened` / `_layers_of` 等缓存均无需修改。
#
# 无法读取时必须明确报告，不得视为「没有图层」：
# 头部数据不足时返回所需的额外字节数（`need`），由网页补发。若静默返回空图层，
# 端口会在无任何报错的情况下减少。

# 申报时一次最多接收的头部字节数（整个文件也超过该值时不再追加请求）。头部以 JSON 中的 base64 传输（server/transfer.py declare_upload），
# 单个请求体最大为 BODY_MAX：base64 膨胀 4/3，另为清单（文件名、sha）预留 2 MB。若该值超过请求体可容纳的量，`need` 超出
# 的部分网页将永远无法发送（413）。
HEAD_MAX = BODY_MAX * 3 // 4 - (2 << 20)


def _declared_path(home: Path, sid: str) -> Path:
    return home / f"{sid}.declared.json"


# 申报一份上传时写在 `sets/` 旁的说明文件（不是该上传自身的文件夹，也不是其清单
# `<id>.json`）。删除上传时这些文件必须一并删除：`resolve` 按是否申报过给出不同的提示
# （`not_sent_yet`），若遗留孤立的申报记录，已被清理的素材会被报告为「字节还没传上来」，与事实不符。
DECLARED_SIDE = (".declared.json", ".layers.json", ".head")


def declared_shas(user_id: int) -> set[str]:
    """该账号已申报且仍在等待字节的上传的内容（sha256）：其 blob 可能已在磁盘上但尚无任何链接
    （farm/disk.py 的孤儿清理不得将其删除）。"""
    out: set[str] = set()
    for said in _sets(user_id).glob("*.declared.json") if _sets(user_id).is_dir() else []:
        try:
            out |= set((json.loads(said.read_text(encoding="utf-8")).get("files") or {}).values())
        except (OSError, ValueError):
            pass
    return out


def _forget_declared(home: Path, sid: str) -> None:
    """删除一份上传的申报记录（清单、读出的图层、保存的头部数据），是 `remove_set` 的一部分。"""
    for suffix in DECLARED_SIDE:
        (home / f"{sid}{suffix}").unlink(missing_ok=True)


def _layers_path(home: Path, sid: str) -> Path:
    return home / f"{sid}.layers.json"


def declare_set(name: str, files: dict[str, str], origin: dict, user_id: int, sizes: dict[str, int] | None = None) -> str:
    """申报一份上传：其中包含哪些文件、各自的内容（sha256），字节可以尚未上传。

    返回节点参数中使用的引用 `upload:<id>/<名称>`，从此刻起即为最终引用：
    id 只由文件名和内容计算（`set_id`），与字节何时上传无关。
    申报写在该账号自己的文件夹中：申报不等于拥有。其他账号已为相同文件组装过文件夹时，
    本账号申报后也无法读取那一份（`_home` 只查看本账号的文件夹）。

    `sizes`：每个文件在使用者本机上的大小。字节尚未上传时，文件参数行需要显示「8 帧 · 18 MB」，
    而服务器没有任何 blob，无法计算，因此由网页提供（网页持有这些文件）。
    字节到齐后 `describe` 改用 blob 自身的大小，那才是权威值。"""
    files = {_name(k): v for k, v in files.items()}
    if not files:
        raise Invalid(Msg("E-UPLOAD-NOFILES"))
    for sha in files.values():
        if not _SHA.match(sha):
            raise Invalid(Msg("E-UPLOAD-BADSHA", sha=sha))
    name = _name(name) if name else ""
    sid = set_id(name, files)
    at = _declared_path(_sets(user_id), sid)
    at.parent.mkdir(parents=True, exist_ok=True)
    had = json.loads(at.read_text(encoding="utf-8")) if at.is_file() else {"name": name, "files": files, "origins": []}
    had["origins"] = [*had.get("origins", []), {**origin, "time": time.time()}][-ORIGINS_KEPT:]
    if sizes:
        had["sizes"] = {_name(k): int(v) for k, v in sizes.items()}
    write_text(at, json.dumps(had, ensure_ascii=False, indent=1))
    return f"{PREFIX}{sid}/{name}"


def declared(sid: str) -> dict | None:
    """一份已申报上传的清单（`{"name", "files"}`），未申报时为 None。只提供给该账号本身
    （与 `resolve` 规则相同：仅知道 id 不够）。"""
    home = _home(sid)
    if home is None:
        return None
    at = _declared_path(home, sid)
    try:
        return json.loads(at.read_text(encoding="utf-8")) if at.is_file() else None
    except (OSError, ValueError):
        return None


def still_missing(sid: str, user_id: int) -> list[str]:
    """该已申报上传中服务器尚未持有的文件（按名称排序）。
    空列表表示字节已到齐，可以调用 `make_set` 组装文件夹。"""
    info = declared(sid)
    if info is None:
        return []
    return sorted(k for k, sha in info["files"].items() if not available(sha, user_id))


def not_sent_yet(sid: str) -> Msg | None:
    """上传已申报但字节尚未到齐时应给出的提示（`E-UPLOAD-SENDING`）；不属于此情况时为 None。

    需要区分的原因：`E-UPLOAD-GONE`「上传的文件已经不在服务器上了（缓存清理过）：请重新选择文件」
    表示已上传过并被清理。而在「先申报、后传字节」的流程中还有另一种情况：从未上传过，
    素材仍在使用者本机。例如选择 `.usd` 后点击「选择…」在层级中挑选时，若报告「缓存清理过，
    请重新选择文件」则与事实不符，重新选择后仍是同一提示。

    不复用 `B-UPLOAD-NOTSENT`：该条（`messages/web.toml`）是网页端的消息，表示
    「刷新之后浏览器不再允许网页读取那些文件，请重新选择同一文件」，需要使用者重新选择。
    此处无需重新选择（字节正在上传，或点击「计算」时即上传），因此使用另一条提示。

    归属规则与 `resolve` 相同：`declared()` 自行检查该上传是否属于本账号，
    因此其他账号的上传不会到达此处，仍与「已被清理」给出相同的回答（见 `resolve` 的说明）。"""
    return None if declared(sid) is None else Msg("E-UPLOAD-SENDING")


def _head_path(home: Path, sid: str) -> Path:
    return home / f"{sid}.head"


def head_described(sid: str) -> dict | None:
    """该已申报上传从头部读出的图层（`{"layers", "of"}`，另含帧号和画面尺寸），尚未读出时为 None。

    立体文件（左右眼）读取文件中的第一个视角，在 `describe_head` 步骤即已确定，
    与字节上传完成后实际打开文件得到的端口一致（原因见 `nodes/core/input.py _view`）。
    没有选择视角的参数，因此此处不重新计算，直接返回保存的结果。"""
    if declared(sid) is None or (home := _home(sid)) is None:
        return None
    try:
        at = _layers_path(home, sid)
        if not at.is_file():
            return None
        said = json.loads(at.read_text(encoding="utf-8"))
        # 帧号和画面尺寸一并返回：`frame_source` 节点需要说明该段包含哪些帧及其尺寸
        # （`engine/evaluation.py` 的 `info`），字节尚未上传时同样必须能够回答，否则会被
        # 判为 E-COOK-FAILED「上传的文件已经不在服务器上了」，而使用者并无操作错误。
        return {**said, **_declared_frames(sid)}
    except (OSError, ValueError, KeyError):
        return None


def _declared_frames(sid: str) -> dict:
    """该申报包含的帧和画面尺寸（`{"frames": [...], "size": (宽, 高)}`）。

    帧号由申报的文件名清单计算（`io/sequence.py group_names`，与 `/api/uploads/sequences`
    使用同一规则）；画面尺寸从保存的头部数据读取（EXR 的尺寸写在头中）。"""
    said = declared(sid)
    if said is None:
        return {}
    names = list(said["files"])
    name = said.get("name") or (names[0] if names else "")
    frames: list[int] = []
    if is_pattern(name):
        from ..io.sequence import group_names

        for head, tail, padding, got in group_names(names).sequences:
            if FrameSequence(Path(), head, tail, padding, tuple(got)).name == name:
                frames = sorted(got)
                break
    home = _home(sid)
    at = _head_path(home, sid) if home is not None else None
    size = None
    if at is not None and at.is_file():
        try:
            size = _head_size(at, said.get("of") or name)
        except Exception:  # noqa: BLE001 — 读不出来就不说，调用的人自己有回落
            size = None
    return {"frames": frames, "size": size}


@lru_cache(maxsize=64)
def _head_size(head: Path, of: str) -> tuple[int, int]:
    """存下来那段头部里写着的画面尺寸（缓存：那段字节写下来就再也不会变）。"""
    import tempfile

    from ..io.images import image_size

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / _name(of)
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(head.read_bytes())
        w, h = image_size(probe)
        return int(w), int(h)


def describe_head(sid: str, first: str, head: bytes, whole: int) -> dict:
    """用第一个文件开头的几十 KB 读出该上传的图层（使用服务器自身的 `describe_file`），读出后保存。

    返回 `{"of", "layers"}`，或 `{"need": 还需要的字节数}`：无法读取时必须明确报告，
    不得返回空图层（否则端口会在无任何报错的情况下减少）。`whole` 为该文件的完整大小：
    已提供完整文件仍无法读取时，说明文件本身有问题，抛出异常供使用者查看。

    立体 EXR（左右眼）按文件中的第一个视角读取（原因见 `nodes/core/input.py _view`）：
    不指定视角读取时两只眼的通道会混在一起，申报时得到 10 个端口、字节上传完成后变为 6 个，
    端口在使用者面前变化却没有任何提示。

    只有画面才有图层：场景文件（各格式模块「导入 …」节点读取的文件）没有图层，读取其头部必然失败，
    而此路径的读取失败会向上抛出，导致选择 .usd 文件时在申报步骤即返回 HTTP 500，
    使用者无法选择文件（`E-IMAGE-OPEN` 变为 500，该行显示「服务器没收下」）。
    因此此处先检查是否为画面格式（`io/sequence.py IMAGE_EXTS`，全项目唯一的判定位置）：
    不是画面格式时返回空结果，表示没有图层可描述，而非读取失败，节点按其自身声明的端口工作。"""
    import tempfile

    if declared(sid) is None or (home := _home(sid)) is None:
        raise NotFound(GONE)
    if Path(_name(first)).suffix.lower() not in IMAGE_EXTS:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / _name(first)
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(head)
        try:
            from ..data.layers import describe_file
            from ..io.images import views

            found = list(views(probe))
            got = {"of": _name(first), "layers": describe_file(probe, found[0] if found else None)}
        except Exception:  # noqa: BLE001 — 头给少了、或者这一层要读像素（没有 manifest 的 Cryptomatte）
            if len(head) >= whole or len(head) >= HEAD_MAX:
                raise  # 已提供完整文件仍无法读取：属于文件本身的问题，原样抛给使用者
            return {"need": min(whole, HEAD_MAX, max(len(head) * 8, 64 << 10))}
    # 头部字节本身也保留：画面尺寸需从中读取（`_head_size`），而 `frame_source` 节点在字节尚未上传时
    # 就需要报告该段的尺寸（`_declared_frames`，原因见 `head_described` 的说明）。
    _head_path(home, sid).write_bytes(head)
    write_text(_layers_path(home, sid), json.dumps(got, ensure_ascii=False, indent=1))
    return got


# ------------------------------------------------------------------ 通道级上传
#
# 目的：文件有 N 个通道而下游只使用其中几个时，只上传用到的通道。文件级按需上传（上一部分）实现了
# 「选择后不上传、点击计算才上传」，但上传的仍是完整文件：一张 20 个通道的多层渲染 EXR，COLMAP 只接入了 rgba 四个通道，
# 其余 16 个通道仍会传输。本部分将其细化到通道级：
#
#   浏览器（`webui/src/transfer/planes.ts`）在 worker 中解出已连线通道的原始平面
#   （half / float / uint 原样，不经过显示变换、不缩小），gzip 后每帧一份，按上述分段协议上传；
#   服务器（`add_planes`）还原平面，并用唯一的 EXR 写入器（`lab2shot_shared.exr.write_exr`）写成
#   只含这些通道的 EXR：通道名、像素类型、压缩方式与原文件一致，不带任何元数据；按内容寻址存入 blob 库。
#
# 存储位置与记录方式：
#   <数据位置>/uploads/<账号 id>/subsets/<原文件 sha[:2]>/<原文件 sha>.json
#       {"blob": 子集 EXR 的 sha256, "channels": [写入的通道名], "types": {通道名: half|float|uint},
#        "width", "height", "display": [x, y, w, h], "data": [x, y, w, h], "compression", "time"}
#       每个原文件、每个账号只有一份当前子集：之后接入新通道时只上传缺少的通道，服务器将并集重写为新 blob，
#       记录随之替换（旧 blob 没有其他文件夹链接时即成为孤儿，由 `farm/disk.py` 清理时按 nlink 回收）。
#       按账号分别记录（在该账号自己的文件夹里）：一个账号上传的像素从不并入另一个账号的文件夹。
#   sets/<id>.json 中增加一项 "subsets": {文件名: {"channels": [...], "blob": 子集 sha}}，表示该上传的文件夹中
#       哪些文件链接的是子集 EXR 而非原文件。`make_set` 每次都会核对：并集换成新 blob 时，文件夹中的硬链接
#       原地替换为新的（`os.replace`，读取方要么看到旧的完整文件，要么看到新的完整文件）。
#
# 身份不变：清单的 `files` 中记录的仍是原文件的 sha，`set_id` 也按其计算，因此参数中的引用、
# `sha_of()`、`content_id()`、节点指纹（`NodeDef.source_identity` 即该引用）均不改变：
# 子集从 {R,G,B,A} 扩大为 {R,G,B,A,Z} 时读取节点的指纹不变，已计算的 rgba 数据包及其下游均无需重新计算，
# 只增加计算 depth 端口（引擎本就按端口记录结果：`engine/cook.py _to_give`）。重新编码的字节从不进入任何身份。
#
# 端口来源：子集文件只包含若干通道，而节点输出端口需要原文件的全部图层。因此
# `nodes/core/input.py _layers_of` 先查询此处的 `declared_layers_for(path)`：文件夹中该文件为子集时，
# 返回申报时从头部读出的图层（`sets/<id>.layers.json`），而非子集文件自身的图层。
#
# 归属：子集 blob 以其自身的 sha 存进该账号的 blob 库，原文件的 sha 只有亲自上传过完整文件时才在其中：
# `has_blob(原文件 sha, 账号)` 只看该账号自己的 blob 库，因此只上传了一个通道（甚至是伪造的）不会让完整原文件进入
# 该账号的文件夹。`available()` 是本部分判断「该账号持有该文件」的唯一依据：完整 blob 由其亲自上传，或其自身的子集存在。

# 每个像素每条通道的字节数：传输的平面与内存里还原出的平面相同（half → float16，float → float32，uint → uint32，
# 各通道保持自己的类型，写回 EXR 逐位不变）
_TYPE_BYTES = {"half": 2, "float": 4, "uint": 4}
# 一帧在内存里还原成的平面（add_planes 的 planes，已有子集并进来的通道也算）最多
# 这么大，同时最多还原 PLANES_AT_ONCE 帧：一份几 MB 的 gzip 能膨胀成任意大，宽高又是请求自己报的，不设上限一个请求就能
# 让网页服务的进程耗尽内存。8K 的一帧七条 float 通道约 1 GB；第一帧头部就超过的，channel_plan 让它整份上传。
PLANES_MAX = 1 << 30
PLANES_AT_ONCE = 2
_planes_slots = threading.BoundedSemaphore(PLANES_AT_ONCE)
# 子集 EXR 的压缩方式沿用原文件，但仅限无损方式：此路径传输的是原始无损像素，
# 将解出的像素再经过有损压缩（B44、DWA，或对 32 位浮点使用 PXR24）后便不再是原文件的像素。
# 有损方式一律改为 ZIPS（与 Nuke 默认相同，无损）。RLE 虽然无损，但写入器不支持，同样改为 ZIPS。
_LOSSLESS = ("none", "zips", "zip", "piz")


def _subset_path(sha: str, user_id: int) -> Path:
    if not _SHA.match(sha):
        raise Invalid(Msg("E-UPLOAD-BADSHA", sha=sha))
    return home_of(user_id) / "subsets" / sha[:2] / f"{sha}.json"


def subset_of(sha: str, user_id: int) -> dict | None:
    """该原文件在服务器上属于本账号的当前通道子集（上述记录），不存在或其 blob 已被清理时
    为 None。其他账号的子集一律不计：仅知道 sha 不够（与 `kept` 规则相同）。"""
    at = _subset_path(sha, user_id)
    try:
        rec = json.loads(at.read_text(encoding="utf-8")) if at.is_file() else None
    except (OSError, ValueError):
        return None
    if rec is None or not _SHA.match(str(rec.get("blob", ""))) or not blob_path(rec["blob"], user_id).is_file():
        return None
    return rec


def available(sha: str, user_id: int) -> bool:
    """该账号是否持有该文件：完整 blob（亲自上传），或其一份通道子集（亲自上传过其中的通道）。
    `make_set` / `still_missing` 判断字节是否到齐只使用此条件。"""
    if has_blob(sha, user_id):
        return True
    return subset_of(sha, user_id) is not None


def _blob_for(sha: str, user_id: int) -> Path:
    """文件夹中该文件应链接的字节：完整 blob 由本账号上传时链接完整 blob，否则链接其自身当前的子集 EXR。
    （若不考虑上传者、只要完整 blob 存在就链接，只上传了一个通道的账号也会获得完整原文件，见上方「归属」部分。）"""
    whole = blob_path(sha, user_id)
    if whole.is_file():
        return whole
    rec = subset_of(sha, user_id)
    if rec is None:
        raise NotFound(GONE)
    return blob_path(rec["blob"], user_id)


def _head_facts(sid: str) -> dict | None:
    """一次读出申报头部（第一帧）中本路径所需的几项信息：
    `{"pairs": [(文件中的通道名, 节点使用的名称)], "nchannels": 文件中的通道总数, "multipart": 是否为多部分,
      "windows": (画幅 (x, y, w, h), 数据窗口 (x, y, w, h))}`，均按文件自身的位置（`io/images.py _windows`）。
    两个名称只在立体文件（一个文件包含左右眼）上不同：节点按文件中的第一个视角读取，名称去掉视角部分
    （`io/images.py _parts`：forward.right.u → forward.u）；单视角的画面（绝大多数）两者相同。
    未申报、头部尚未读出或头部无法打开时为 None。账号校验在此处进行（`declared`），纯计算部分按
    (id, 头部的修改时间) 缓存，因为头部字节写入后不再改变。"""
    said, home = declared(sid), _home(sid)
    if said is None or home is None or not (head := _head_path(home, sid)).is_file():
        return None
    return _head_facts_at(head, _name(said.get("of") or next(iter(said["files"]))), head.stat().st_mtime)


@lru_cache(maxsize=64)
def _head_facts_at(head: Path, of: str, mtime: float) -> dict | None:
    import tempfile

    import OpenImageIO as oiio

    from ..io.images import _parts, _windows, views

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / of
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(head.read_bytes())
        inp = oiio.ImageInput.open(str(probe))
        if inp is None:
            return None
        try:
            multipart = bool(inp.seek_subimage(1, 0))
            inp.seek_subimage(0, 0)
            raw = list(inp.spec().channelnames)
            nchannels = int(inp.spec().nchannels)
            found = views(probe)
            named = _parts(inp, found[0] if len(found) > 1 else None)[0]
            windows = _windows(inp)
        finally:
            inp.close()
    return {"pairs": [(r, n) for r, n in zip(raw, named) if n is not None], "nchannels": nchannels,
            "multipart": multipart, "windows": windows}


def channel_plan(sid: str, wanted: list[str]) -> dict | None:
    """本次计算该上传需要传输的通道：`{"take": [文件中的名称], "write": [子集中的名称]}`，
    浏览器按 `take` 解码，服务器按 `write` 写入。None 表示上传完整文件：所需的即文件中的全部通道、该上传不是
    已申报的画面，或为多部分（multi-part）EXR（浏览器端解码器按名称取通道，无法区分不同部分中的同名通道）。
    `wanted`：节点使用的通道名（`nodes/core/input.py ReadSequence.upload_channels`）。"""
    if not _ID.match(sid) or not wanted:
        return None
    facts = _head_facts(sid)
    if facts is None or facts["multipart"]:
        return None
    by_name = {n: r for r, n in facts["pairs"]}
    take = [by_name[n] for n in wanted if n in by_name]
    if len(take) != len(wanted):
        return None  # 所需通道不在文件头中：此路径无法处理，整份上传，读取时照常报错
    if len(set(take)) >= facts["nchannels"]:
        return None  # 所需的已是文件中的全部通道
    _, box = facts["windows"]
    if box[2] * box[3] * len(wanted) * max(_TYPE_BYTES.values()) > PLANES_MAX:
        return None  # 一帧还原出来超过 PLANES_MAX（add_planes）：整份上传
    return {"take": take, "write": list(wanted)}


def planes_missing(shas: list[str], wanted: list[str], user_id: int) -> dict[str, list[str]]:
    """这些原文件在服务器上仍缺少的通道（按 `write` 的名称）：完整文件存在或子集中均已具备时为空列表；
    其他账号上传的不计入（与 `kept` 规则相同）。"""
    out: dict[str, list[str]] = {}
    for sha in shas:
        if not _SHA.match(sha):
            continue
        if has_blob(sha, user_id):
            out[sha] = []
            continue
        rec = subset_of(sha, user_id)
        have = set(rec["channels"]) if rec else set()
        out[sha] = [c for c in wanted if c not in have]
    return out


def _read_native(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """子集 EXR 中各通道的像素，各按自己的类型原样读出（half → float16，float → float32，uint → uint32，
    `write_exr` 写回时逐位不变），以及各通道的像素类型名。"""
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise NotFound(GONE)
    try:
        spec = inp.spec()
        kinds = {}
        out = {}
        for i, name in enumerate(spec.channelnames):
            fmt = spec.channelformat(i)
            kind = "uint" if fmt == oiio.TypeDesc(oiio.UINT32) else "half" if fmt == oiio.TypeDesc(oiio.HALF) else "float"
            kinds[name] = kind
            px = inp.read_image(0, 0, i, i + 1, {"half": oiio.HALF, "float": oiio.FLOAT, "uint": oiio.UINT32}[kind])
            out[name] = np.asarray(px).reshape(spec.height, spec.width)
        return out, kinds
    finally:
        inp.close()


def add_planes(sid: str, sha: str, blob: str, channels: list[dict], width: int, height: int, compression: str,
               user_id: int, display: list[int] | None = None, data: list[int] | None = None) -> dict:
    """浏览器上传的一帧中的若干通道（一份经 gzip 压缩的 blob，平面按 `channels` 的顺序首尾相接）：
    与该原文件已有的子集合并，写为新的子集 EXR，并替换记录。

    `sid`：该上传（已申报的 id，头部和文件名由此获取）；`sha`：原文件的 sha256；`blob`：包含平面的 blob；
    `channels`：[{"take": 文件中的名称, "write": 子集中的名称, "type": half|float|uint}]；
    `width`/`height`：平面尺寸；`compression`：原文件的压缩方式（OpenEXR 名称，小写）；
    `display` / `data`：该帧的画幅和数据窗口 [x, y, w, h]（按文件自身的位置），浏览器无法提供时按第一帧的头部
    推断：平面与头部中的数据窗口大小相同则置于该处，与画幅大小相同则为整幅，均不符合时拒绝（E-UPLOAD-PLANESWINDOW）。

    返回 `{"sha": 原文件 sha, "channels": [子集当前包含的通道], "blob": 子集 blob}`。同一帧重复上传是幂等的：
    所需通道在子集中均已存在时，即使平面 blob 不存在也视为成功（浏览器未收到上一次响应时会再次请求）。"""
    import gzip

    from lab2shot_shared.exr import PIXEL_TYPES

    if not _SHA.match(sha) or not _ID.match(sid):
        raise Invalid(Msg("E-UPLOAD-BADSHA", sha=sha))
    said = declared(sid)
    if said is None:
        raise NotFound(GONE)
    file = next((k for k, v in said["files"].items() if v == sha), None)
    if file is None:
        raise Invalid(Msg("E-UPLOAD-NOTSENT", files=[sha[:12]]))
    for c in channels:
        if c.get("type") not in PIXEL_TYPES:
            raise Invalid(Msg("E-UPLOAD-PLANESTYPE", file=file, channel=c.get("write"), type=c.get("type")))
    facts = _head_facts(sid) or {"pairs": [], "windows": None}
    pairs = facts["pairs"]
    known = {n for _, n in pairs}
    for c in channels:
        if c["write"] not in known:
            raise Invalid(Msg("E-UPLOAD-PLANESNAME", file=file, channel=c["write"]))
    wanted = [c["write"] for c in channels]
    had = subset_of(sha, user_id)
    if had is not None and all(n in had["channels"] for n in wanted):
        return {"sha": sha, "channels": had["channels"], "blob": had["blob"]}  # 幂等：所需通道均已存在
    if not _SHA.match(blob) or not has_blob(blob, user_id):
        raise NotFound(Msg("E-UPLOAD-PLANESBLOB", blob=blob[:12]))
    # ---- 窗口：浏览器提供时按其提供，否则按第一帧的头部推断
    w, h = int(width), int(height)
    if len(set(wanted)) != len(wanted) or w <= 0 or h <= 0:
        raise Invalid(Msg("E-UPLOAD-PLANESSIZE", file=file, channels=len(channels), width=w, height=h, expected=0, got=0))
    if display and data and len(display) == 4 and len(data) == 4:
        full, box = tuple(int(v) for v in display), tuple(int(v) for v in data)
    else:
        heads = facts["windows"]
        full, box = heads if heads else ((0, 0, w, h), (0, 0, w, h))
        if (box[2], box[3]) != (w, h):
            if (full[2], full[3]) == (w, h):
                box = (full[0], full[1], w, h)
            else:
                raise Invalid(Msg("E-UPLOAD-PLANESWINDOW", file=file, width=w, height=h, dw=box[2], dh=box[3], fw=full[2], fh=full[3]))
    if (box[2], box[3]) != (w, h):
        raise Invalid(Msg("E-UPLOAD-PLANESSIZE", file=file, channels=len(channels), width=w, height=h, expected=0, got=0))
    # ---- 这一帧还原出来多大：先算清楚，超过 PLANES_MAX 就不解压（一份小 gzip 能膨胀到数十 GB，宽高又是请求自己报的）
    kinds = {c["write"]: c["type"] for c in channels}
    size = w * h * sum(_TYPE_BYTES[k] for k in kinds.values())
    if size > PLANES_MAX:
        raise Invalid(Msg("E-UPLOAD-PLANESTOOBIG", file=file, channels=len(channels), width=w, height=h,
                          gb=size / (1 << 30), most=PLANES_MAX / (1 << 30)))
    with _planes_slots:
        # 已有子集中的通道原样并入（同名通道以新上传的为准），按原文件的通道顺序，放得下 PLANES_MAX 的为止：
        # 放不下的不留在子集里，以后用到时 planes_missing 让浏览器再传一次
        planes: dict[str, np.ndarray] = {}  # 通道名 -> [H, W] 平面，各为自己的类型
        if had is not None:
            got, old_kinds = _read_native(blob_path(had["blob"], user_id))
            for n in [n for _, n in pairs if n in had["channels"] and n in got and n not in kinds]:
                if got[n].shape == (h, w) and size + w * h * _TYPE_BYTES[old_kinds[n]] <= PLANES_MAX:
                    planes[n], kinds[n] = got[n], old_kinds[n]
                    size += w * h * _TYPE_BYTES[old_kinds[n]]
            del got
        order = [n for _, n in pairs if n in kinds]  # 通道顺序与原文件一致
        # ---- 还原平面：逐条通道从 gzip 读出，每条只读它应有的字节数，读完不许还有剩余；各通道保持自己的类型
        expected = sum(_TYPE_BYTES[c["type"]] * w * h for c in channels)
        with gzip.open(blob_path(blob, user_id), "rb") as f:
            at = 0
            for c in channels:
                n = _TYPE_BYTES[c["type"]] * w * h
                chunk = f.read(n)
                at += len(chunk)
                if len(chunk) != n:
                    raise Invalid(Msg("E-UPLOAD-PLANESSIZE", file=file, channels=len(channels), width=w, height=h, expected=expected, got=at))
                dtype = {"half": np.float16, "float": np.float32, "uint": np.uint32}[c["type"]]
                planes[c["write"]] = np.frombuffer(chunk, dtype).reshape(h, w)
                del chunk
            if f.read(1):
                raise Invalid(Msg("E-UPLOAD-PLANESSIZE", file=file, channels=len(channels), width=w, height=h, expected=expected, got=expected + 1))
        return _write_subset(sha, blob, [planes[n] for n in order], order, kinds, full, box, compression, user_id)


def _write_subset(sha: str, blob: str, planes: list[np.ndarray], order: list[str], kinds: dict[str, str], full: tuple,
                  box: tuple, compression: str, user_id: int) -> dict:
    """把还原好的平面（按 `order`，各为自己类型的 [H, W]）写成子集 EXR，按内容寻址存下，替换该原文件的子集记录（add_planes）。"""
    import tempfile

    from lab2shot_shared.exr import EXR_COMPRESSIONS, write_exr

    h, w = planes[0].shape
    comp = str(compression or "zips").lower()
    if comp not in _LOSSLESS and not (comp == "pxr24" and all(kinds[n] != "float" for n in order)):
        comp = "zips"
    if comp not in EXR_COMPRESSIONS:
        comp = "zips"
    home_of(user_id).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home_of(user_id)) as tmp:  # 位于该账号自己的磁盘上：写完后改名放入
        out = Path(tmp) / "subset.exr"
        # windows 按 write_exr 的约定：画幅为 (x, y, w, h)，数据窗口的 x, y 从画幅左上角计算。
        # 不带任何元数据：OpenImageIO 写 EXR 时会自动添加 DateTime（写入时刻），传入空值即不添加；
        # 其余的 compression、lineOrder、pixelAspectRatio、screenWindow* 是 EXR 头中的必需字段，不属于元数据。
        write_exr(out, planes, order, compression=comp, types=[kinds[n] for n in order], header={"DateTime": ""},
                  windows=((full[0], full[1], full[2], full[3]), (box[0] - full[0], box[1] - full[1], w, h)))
        digest = hashlib.sha256()
        with out.open("rb") as f:
            while block := f.read(1 << 24):
                digest.update(block)
        new = digest.hexdigest()
        target = blob_path(new, user_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.replace(out, target)
            _readonly(target)
    rec = {"blob": new, "channels": order, "types": {n: kinds[n] for n in order}, "width": w, "height": h,
           "display": list(full), "data": list(box), "compression": comp, "time": time.time()}
    at_ = _subset_path(sha, user_id)
    at_.parent.mkdir(parents=True, exist_ok=True)
    write_text(at_, json.dumps(rec, ensure_ascii=False, indent=1))
    # 包含平面的 blob 用完即删除（它只属于本帧的本次上传）
    _drop_blob(blob, user_id)
    return {"sha": sha, "channels": order, "blob": new, "bytes": target.stat().st_size}


def _drop_blob(sha: str, user_id: int) -> None:
    """删除该账号的一份 blob（装平面的那份用完即删）；它被别处链接着时（nlink > 1）只是少一个名字，那一份仍在。"""
    try:
        blob_path(sha, user_id).unlink(missing_ok=True)
    except OSError:
        pass


def declared_layers_for(path: Path | str) -> dict | None:
    """文件夹中该文件链接的是子集 EXR 时，返回其原文件的全部图层（申报时从头部读出的结果，`sets/<id>.layers.json`）；
    不是子集（完整原文件，或根本不是上传的文件）时为 None，此时打开文件自行统计即为正确结果。
    `nodes/core/input.py _layers_of` 先查询此处：端口需按原文件生成，不能随子集中剩余的通道数变化。"""
    where = _placed(path)
    if where is None:
        return None
    home, sid, rel = where
    try:
        subsets = _manifest(home, sid).get("subsets") or {}
    except (OSError, ValueError, KeyError):
        return None
    if rel not in subsets:
        return None
    try:
        said = json.loads(_layers_path(home, sid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return said.get("layers") or None


def _placed(path: Path | str) -> tuple[Path, str, str] | None:
    """存储中某路径的位置：(账号文件夹, 上传 id, 其中的文件名)，不是上传集合中的文件时为 None。
    此处只解析目录结构，不做账号校验：调用方从 `resolve` 获得该路径，`resolve` 已针对其账号作答。"""
    p = Path(path)
    try:
        rel = p.relative_to(root())
    except ValueError:
        return None
    # <account>/sets/<id>/<name>
    if len(rel.parts) < 4 or not rel.parts[0].isdigit() or rel.parts[1] != "sets" or not _ID.match(rel.parts[2]):
        return None
    return root() / rel.parts[0] / "sets", rel.parts[2], "/".join(rel.parts[3:])


def is_ref(value: str) -> bool:
    return value.startswith(PREFIX)


def ref_parts(ref: str) -> tuple[str, str] | None:
    """`upload:<id>/<name>` 的 (id, name)；不是合法引用时为 None。"""
    if not isinstance(ref, str) or not is_ref(ref):
        return None
    sid, _, name = ref[len(PREFIX):].partition("/")
    return (sid, name) if _ID.match(sid) else None


def resolve(ref: str) -> Path:
    """节点读取上传的位置：其文件夹，或其中的文件 / 序列模式，针对当前服务的账号（serving()）。
    这是决定归属的唯一位置：不属于本账号的上传视为不存在（GONE），与已清理的上传给出相同的回答，且其内容不会被打开或区分。
    所有读取上传的一方（规划或计算中的节点 PlanEnv.upload、查询参数所指内容的页面、基准登记表）都经过此处，
    因此引擎、农场、状态回复和部分结果在构造上保持一致。"""
    if not is_ref(ref):
        raise Invalid(Msg("E-UPLOAD-SERVERPATH", path=ref))
    sid, _, name = ref[len(PREFIX):].partition("/")
    home = _home(sid)
    folder = home / sid if home is not None else None
    if folder is None or not folder.is_dir():
        raise NotFound(not_sent_yet(sid) or GONE)
    used(folder.with_suffix(".json"))  # 清理时保留正在使用的内容
    return folder / _name(name) if name else folder


def sha_of(path: Path | str) -> str | None:
    """上传集合中某文件的内容（sha256），以其清单为准（None：不是上传中的文件）。"""
    where = _placed(path)
    if where is None:
        return None
    home, sid, rel = where
    try:
        files = _manifest(home, sid)["files"]
    except (OSError, ValueError, KeyError):
        return None
    return files.get(rel)


@lru_cache(maxsize=256)
def _manifest_at(manifest: Path, mtime: float) -> dict:
    return json.loads(manifest.read_text(encoding="utf-8"))


def _manifest(home: Path, sid: str) -> dict:
    manifest = home / f"{sid}.json"
    return _manifest_at(manifest, manifest.stat().st_mtime)


def files_of(sid: str, user_id: int) -> dict[str, str] | None:
    """该账号的上传 `sid` 包含的文件：{名称: sha256}（已组装的清单或申报的清单都算，与字节是否已到齐无关）。
    只看这个账号自己的文件夹（任务分组，transfer/groups.py：分组只在账号内部）；没有时为 None。"""
    if not _ID.match(sid or ""):
        return None
    home = _sets(user_id)
    for manifest in (home / f"{sid}.json", _declared_path(home, sid)):
        try:
            files = _manifest_at(manifest, manifest.stat().st_mtime)["files"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(files, dict):
            return {str(k): str(v) for k, v in files.items()}
    return None


def picked_name(sid: str, user_id: int) -> str:
    """该账号的上传 `sid` 在用户那边叫什么：上传时选的文件夹名（序列、整个文件夹），或文件名（视频、单个文件），
    例如 `sh030_plate`、`dance.mp4`（任务分组的组名，transfer/groups.py）。取自清单里最近一次的来源
    （`origins[].path`：网页给的「文件夹/名称」，本机脚本和 DCC 给的本机路径）；来源没给路径时用节点读取的名称。
    只是一段文字，原样交给调用方清理（groups.clean_name）；该账号没有这份上传时为 ""。"""
    if not _ID.match(sid or ""):
        return ""
    home = _sets(user_id)
    found = []
    for manifest in (home / f"{sid}.json", _declared_path(home, sid)):
        try:
            found.append(_manifest_at(manifest, manifest.stat().st_mtime))
        except FileNotFoundError:
            continue
    if not found:
        return ""
    item = found[0]["name"].rsplit("/", 1)[-1]  # 节点读取的：一个文件、一个序列图案，或 "" 表示整个文件夹
    latest = max((o for data in found for o in data["origins"]), key=lambda o: o["time"])
    parts = [p for p in re.split(r"[\\/]", latest["path"]) if p]
    if not parts:
        return item
    if not item:  # 整个文件夹：来源就是那个文件夹
        return parts[-1]
    if not is_pattern(item) and len(found[0]["files"]) == 1:  # 一个文件：它自己的名字
        return item
    # 一段序列（节点读的是图案，或者其中一帧）：来源是「文件夹/图案」或其中一帧（最后一段和它同一个扩展名），
    # 或者就是那个文件夹
    if Path(parts[-1]).suffix.lower() == Path(item).suffix.lower():
        return parts[-2] if len(parts) > 1 else item
    return parts[-1]


def boundary(path: Path) -> Path:
    """使用者上传的文件除自身之外可引用的内容（USD 文件的子层和引用、其引用的图片）：只能是与其一起上传的文件，
    不得是本服务器上的其他任何内容。存储之外的文件（服务器上脚本或测试自带的文件）只能引用其自身所在的文件夹。"""
    p = path.resolve()
    base = root().resolve()
    if base in p.parents and len(rel := p.relative_to(base).parts) >= 4 and rel[1] == "sets":
        return base / rel[0] / "sets" / rel[2]  # <账号>/sets/<id>：仅限该上传
    return p.parent


def may_draw_on(layer: Path, file: Path) -> bool:
    """场景文件（`layer`）是否可以引用 `file`。这是唯一的判定位置（读取场景层所引用文件的一方都询问此处），
    结果取决于该层本身的来源：文件的可引用范围即其作者的范围：

      - 使用者上传的层只能引用与其一起上传的文件（边界）：文本由其作者提供，
        因此其引用必须位于作者自己的上传之内，不得是其他账号的上传、交付物或他人反馈的截图，
        这些都位于同一工作文件夹中；
      - 本项目计算写出的层可以引用结果存储：该路径由本项目的节点写出，而非他人发送的文件
        （「灯光」将 HDRI 作为自己的数据包写在引用它的场景旁边，因此结果可以合理地引用另一个数据包的文件）。
        账号能读取其中哪些结果与本问题无关：每个账号只读得到自己缓存里的结果（server/access.py readable）。

    其他情况（既非上述两者的层，或位于两者之外的文件）一律不可。"""
    from ..data.packet import cooked_here

    limit, target = boundary(layer), Path(file).resolve()
    if limit in target.parents:
        return True
    return cooked_here(layer) and cooked_here(file)


def _describe_declared(ref: str, said: dict) -> dict:
    """字节尚未上传时（「先申报、后传字节」流程）文件参数行的显示内容。

    与下方 `describe` 回答同一问题、结构相同，只是不访问磁盘上的文件：
    文件名、名称、数量、首帧和末帧全部由申报的清单计算（序列识别仍使用
    `io/sequence.py group_names`，与 `/api/uploads/sequences` 规则相同）；
    大小由网页在申报时提供（服务器此时没有任何 blob，原因见 `declare_set` 的 `sizes` 说明）。

    不得报告「服务器上已经没有这份文件了」：该提示的含义是素材已丢失、需重新选择，
    而此时素材仍在使用者本机，只是尚未上传；使用者看到该提示会重新选择，徒增操作。"""
    files = list(said["files"])
    name = said.get("name") or (files[0] if files else "")
    first = last = None
    stem = Path(name).stem
    if is_pattern(name):
        from ..io.sequence import group_names

        for head, tail, padding, frames in group_names(files).sequences:
            if FrameSequence(Path(), head, tail, padding, tuple(frames)).name == name:
                first, last = min(frames), max(frames)
                stem = head.rstrip("._- ") or stem
                break
    return {"ref": ref, "name": name, "stem": stem or name, "files": len(files),
            "bytes": sum(int(v) for v in (said.get("sizes") or {}).values()), "first": first, "last": last}


def describe(ref: str) -> dict:
    """编辑器在参数上显示的内容：上传的名称、文件数、总大小，序列另含首帧和末帧；`stem` 为序列读取节点
    所读取的画面自身名称（plate.####.exr -> plate，cam.usda -> cam，文件夹 -> 其名称）：交付物以此命名，
    因此页面不自行去除帧号。"""
    sid = ref[len(PREFIX):].split("/", 1)[0]
    home = _home(sid)
    if (home is None or not (home / sid).is_dir()) and (said := declared(sid)) is not None:
        return _describe_declared(ref, said)  # 已申报、字节尚未上传：按申报数据回答（原因见下方函数的说明）
    target = resolve(ref)
    data = json.loads((home / f"{sid}.json").read_text(encoding="utf-8"))
    size = 0
    for rel in data["files"]:  # 文件夹中实际链接的文件：子集只计算子集本身（服务器上实际存在的字节）
        try:
            size += (home / sid / rel).stat().st_size
        except OSError:
            pass
    seq = (find_sequence(target) if is_pattern(target.name)
           else sequence_of(target) if target.is_file() and target.suffix.lower() in IMAGE_EXTS else None)
    name = data["name"] or target.name
    stem = seq.head.rstrip("._- ") if seq else (Path(name).stem if target.is_file() else Path(name).name)
    return {"ref": ref, "name": name, "stem": stem or Path(name).stem, "files": len(data["files"]), "bytes": size,
            "first": seq and seq.first, "last": seq and seq.last}


def refs_in(value: object) -> set[str]:
    """一份 JSON（节点图、任务记录）中引用的上传集合 id：所有 `upload:<id>/...` 中的 id。

    按字符串查找，而不按节点类型查找：哪个参数是文件参数由节点自行决定，而「该节点图用到哪些上传」
    是删除任务时必须准确回答的问题（删除任务会一并删除相关内容，一份素材被多个任务引用时只有完全无引用才删除）。
    字符串扫描对任何节点都成立，新增文件参数时无需修改此处。"""
    out: set[str] = set()
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            if is_ref(v):
                sid = v[len(PREFIX):].partition("/")[0]
                if _ID.match(sid):
                    out.add(sid)
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
    return out


def on_disk() -> list[dict]:
    """服务器上的每份上传，供磁盘页和自动清理使用（farm/disk.py）：每项为
    `{"user_id", "sid", "path", "bytes", "used"}`，`path` 为上传的文件夹（<account>/sets/<id>，无论是否已组装），
    `bytes` 为仅由其持有的字节（只链接到它的文件，或申报的头部），`used` 为最后使用时间（其清单的 mtime：
    data/packet.py `used` 会更新它，任务提交时也更新：transfer/tasks.py）或申报时间。除本模块自身的辅助函数外，
    这是唯一了解目录结构的位置。"""
    out = []
    for user_id in accounts():
        home = _sets(user_id)
        seen = set()
        for manifest in _manifests(home):
            sid = manifest.stem
            seen.add(sid)
            folder = home / sid
            own = sum(st.st_size for _, st in stats(folder, set()) if st.st_nlink <= 2)
            out.append({"user_id": user_id, "sid": sid, "path": folder, "bytes": own, "used": mtime(manifest)})
        for said in sorted(home.glob("*.declared.json")) if home.is_dir() else []:
            sid = said.name[: -len(".declared.json")]
            if sid in seen or not _ID.match(sid):
                continue  # 已组装：即上一行（其附属说明文件在 remove_set 中一并删除）
            head = home / f"{sid}.head"
            out.append({"user_id": user_id, "sid": sid, "path": home / sid,
                        "bytes": head.stat().st_size if head.is_file() else 0, "used": mtime(said)})
    return out


def orphan_blobs(user_id: int) -> list[Path]:
    """该账号不再被任何名字链接（nlink 1：没有上传文件夹、也没有任务文件夹链接它）且不会被链接的 blob：既未被申报
    并等待字节，也不在 DONE_KEEP_S 内完成（刚完成的分段在 `make_set` 链接之前是 blob，发送方可能仍在进行中）。
    供磁盘清理（farm/disk.py）使用，清理时在队列的保护下逐个通过 `drop_blob` 删除。"""
    declared = declared_shas(user_id)
    young = time.time() - DONE_KEEP_S
    base = home_of(user_id) / "blobs"
    out = []
    for blob in base.rglob("*") if base.is_dir() else []:
        try:
            st = blob.stat()
        except OSError:
            continue
        if blob.is_file() and _SHA.match(blob.name) and st.st_nlink == 1 and blob.name not in declared and st.st_mtime < young:
            out.append(blob)
    return out


def drop_blob(blob: Path) -> int:
    """删除一个孤儿 blob（`orphan_blobs`）；返回释放的字节数（它此时已没有别的名字：nlink 为 1）。"""
    try:
        size = blob.stat().st_size if blob.stat().st_nlink == 1 else 0
        blob.unlink()
    except OSError:
        return 0
    return size


def remove_set(user_id: int, sid: str) -> tuple[int, int]:
    """上传离开服务器的唯一途径（自动清理，farm/disk.py；账号的数据清理）：删除其文件夹、清单和申报附属文件，以及
    读取节点从中生成的该账号缓存数据包（若保留这些数据包，已不存在的文件会显示为「已缓存」）。任务文件夹里对这些
    字节的硬链接不受影响（它们是另外的名字）。返回（仅由该文件夹持有的字节数, 删除的缓存数据包数）。
    blob 不在此处删除：由孤儿清理在没有任何名字链接它时释放。"""
    from ..data import packet as packets

    home = _sets(user_id)
    if not _ID.match(sid) or not _has(home, sid):
        return 0, 0
    folder = home / sid
    freed = folder_bytes([folder])
    shutil.rmtree(folder, ignore_errors=True)
    (home / f"{sid}.json").unlink(missing_ok=True)
    _forget_declared(home, sid)
    marker = f"/{int(user_id)}/sets/{sid}/"  # 该账号中该上传的文件夹，无论「数据位置」在何处
    from ..serving import Account, serving

    with serving(Account(user_id)):  # 读取它的缓存数据包只在该账号自己的缓存里（data/store.py）
        dropped = packets.remove_referring(lambda ref: marker in ref, "upload")
    return freed, dropped


def forget_account(user_id: int) -> None:
    """账号被永久删除（lab2shot/accounts.py purge，经 farm/disk.py forget_account）：它的全部上传（字节、分段、上传、子集）
    立即删除，SQLite 之后可能把同一个 id 给新账号，新账号不得看到其中任何内容。任务文件夹里的硬链接此前已随任务删除。"""
    shutil.rmtree(home_of(user_id), ignore_errors=True)
