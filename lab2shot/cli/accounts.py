"""`lab2shot login | logout`（Lab2Shot 服务上的账号）与 `lab2shot admin`（本机上的管理员密码、口令和登录会话，
以及通过本机令牌管理正在运行的服务）。"""

from __future__ import annotations

import time
from typing import Optional

import typer

from ..messages import Msg
from .base import app, console, failed, group, when

admin = group("管理员账号的密码与「我的口令」：仅可在本服务器上通过以下命令修改（服务运行期间亦可执行）。")


def accounts_admin_name() -> str:
    from .. import accounts

    return accounts.admin().username


@app.command()
def login(
    server: str = typer.Option(..., "--server", help="Lab2Shot 服务地址，例如 http://render-box:8765"),
    username: Optional[str] = typer.Argument(None, help="用户名（由管理员在「用户」中创建）；省略时提示输入"),
) -> None:
    """以账号登录一台 Lab2Shot 服务：此后本机的 lab2shot cook --server 和 DCC 插件均使用该账号（令牌保存在 ~/.lab2shot/tokens.json，仅当前用户可读）。"""
    from ..client import Lab2Shot, Lab2ShotError, tokens_file

    name = username or typer.prompt("用户名")
    try:
        user = Lab2Shot(server, app="cli").login(name, typer.prompt("密码", hide_input=True))
    except Lab2ShotError as exc:
        failed(exc)
    console.print(f"[green]已登录[/green]：{user['name']}（{user['username']} · {user['department'] or '未分环节'}），令牌已保存至 {tokens_file()}")


@app.command()
def logout(server: str = typer.Option(..., "--server", help="Lab2Shot 服务地址")) -> None:
    """退出一台 Lab2Shot 服务的登录：令牌作废，并从本机删除。"""
    from ..client import Lab2Shot

    Lab2Shot(server, app="cli").logout()
    console.print("已退出登录")


def _new_secret(what: str) -> str:
    from .. import accounts

    while True:
        text = typer.prompt(f"新{what}", hide_input=True, confirmation_prompt=f"再次输入新{what}")
        if not (problem := accounts.rule_problem(text, what)):
            return text
        console.print(f"[red]{problem}[/red]")


@admin.command("password")
def admin_password() -> None:
    """重设内置管理员账号的密码，用于忘记密码或因输错次数过多被锁定的情况。立即生效：管理员在所有网页上的登录须用新密码重新登录，此前的错误次数清零。"""
    from .. import accounts
    from ..server.access import audit

    accounts.set_password(accounts.ADMIN_ID, _new_secret("管理员密码"), "命令行")
    audit(Msg("I-LOGIN-CLIRESET"))  # an administrator's action from the command line: no session, no request
    console.print(f"[green]管理员密码已修改[/green]：用户名 {accounts.admin().username}，立即生效，此前的密码错误次数已清零。")


@admin.command("passphrase")
def admin_passphrase() -> None:
    """设置「我的口令」：仅可在本服务器上设置，网页上不可查看或修改。忘记管理员密码时，在登录页点击「忘记密码」，凭口令设置新密码。"""
    from .. import accounts
    from ..server.access import audit

    had = accounts.passphrase() is not None
    accounts.set_passphrase(_new_secret("口令"))
    audit(Msg("I-LOGIN-CLIPASSPHRASECHANGED" if had else "I-LOGIN-CLIPASSPHRASESET"))
    console.print(f"[green]口令已{'更新' if had else '设置'}[/green]：忘记管理员密码时，可在登录页点击「忘记密码」，输入口令后设置新密码。"
                  "系统仅保存口令的不可逆指纹，任何人均无法查看原文，请妥善保管。")


@admin.command("status")
def admin_status() -> None:
    """显示管理员密码是否仍为默认值、口令是否已设置、账号数量（谁在线只有运行中的服务知道：看后台「用户」页）。"""
    from .. import accounts

    owner, phrase = accounts.admin(), accounts.passphrase()
    if owner.no_password:
        console.print(f"管理员    {owner.username}，[yellow]尚未设置密码，无法登录：请执行 ./setup.sh 并选择「账号与安全 → 设置管理员密码」[/yellow]")
    else:
        console.print(f"管理员    {owner.username}，密码于 {when(owner.password_set or 0)} 由{owner.password_by}修改")
    console.print(f"我的口令  {'未设置：运行 lab2shot admin passphrase 进行设置' if phrase is None else when(phrase['set']) + ' 设置'}")
    users = [u for u in accounts.listing() if not u["deleted"]]
    # 在线是运行中的服务在内存里记的（accounts.presence）：命令行这个进程看不到，只指向后台页面
    console.print(f"账号      {len(users)} 个（在「用户」中新建），谁在线看后台「用户」页")


@admin.command("logout")
def admin_logout() -> None:
    """使所有网页、DCC 插件和命令行退出登录（所有使用者须重新登录）。怀疑密码泄露时，应先修改密码，再执行此命令。"""
    from .. import accounts
    from ..server.access import audit

    n = accounts.end_all()
    audit(Msg("I-LOGIN-CLIREVOKED", count=n))
    console.print(f"已使 {n} 个登录会话退出。")


def local_client():
    """返回连接本机 Lab2Shot 服务（设置项 server.port）的客户端，以本机命令行身份使用机器令牌，而非密码登录：
    一个账号同一时间只能在一处登录，在此处密码登录会结束管理员的浏览器会话。

    HTTPS 是一项设置（server.https），因此协议取自服务读取的同一设置，不固定为「http://」：开启 HTTPS 时，
    http 请求会被直接断开连接，所有管理命令（查看队列、重启）都将失败。证书由本服务器自有的
    证书颁发机构签发（server/tls.py），默认不受信任，因此向客户端提供该机构的证书文件，而不是跳过验证。

    地址优先取运行中的服务自己记下的地址（server/restart.py recorded_address）：端口与 HTTPS 须重启才生效，
    设置文件里的值未必是服务正在用的。没有运行中的服务时取设置的当前值（Settings.value），不取本进程启动时的值：
    配置菜单在同一次会话里改了端口或 HTTPS、再启动服务后，仍须连得上。"""
    from .. import accounts
    from ..client import Lab2Shot
    from ..config import settings
    from ..server import tls
    from ..server.restart import recorded_address

    s = settings()
    address = recorded_address() or f"{'https' if s.value('server.https') else 'http'}://127.0.0.1:{s.value('server.port')}"
    ca = tls.authority_file() if address.startswith("https:") else None
    return Lab2Shot(address, app="cli", machine=accounts.machine_token(), cafile=str(ca) if ca else None)


@admin.command("restart")
def admin_restart_cmd(
    mode: str = typer.Option("drain", "--mode", help="drain：等待计算中的任务完成，排队中的任务在重启后继续排队；now：立即停止计算中的任务"),
    wait: bool = typer.Option(False, "--wait", help="等待新服务启动（启动编号变化）后再返回，最长等待 5 分钟"),
) -> None:
    """重启本机正在运行的 lab2shot ui。协调重启应使用此命令，不应通过密码登录调用 /api/admin/restart，否则会使管理员浏览器中的登录失效。"""
    from ..client import Lab2ShotError

    lab = local_client()
    try:
        before = lab.server_state()
        lab.admin_restart(mode)
    except Lab2ShotError as exc:
        failed(exc)
    console.print(f"已请求重启（{mode}）：当前为第 {before['boot']} 次启动")
    if not wait:
        return
    console.print("正在等待新服务启动…")
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            now = lab.server_state()
        except Lab2ShotError:
            continue
        if now["boot"] != before["boot"]:
            console.print(f"[green]已重启[/green]：第 {now['boot']} 次启动")
            return
    failed("已等待 5 分钟，重启仍未完成，请检查服务状态")


@admin.command("unlock")
def admin_unlock_cmd() -> None:
    """将输错密码的计数清零：被陌生设备试错而拖慢的账号、被锁住的来源马上可以再试。仅能在本服务器上执行（机器令牌），
    浏览器上的管理员登录无法清零，因为计数挡的正是用密码猜。"""
    from ..client import Lab2ShotError

    try:
        got = local_client().admin_unlock()
    except Lab2ShotError as exc:
        failed(exc)
    console.print(f"[green]已清零[/green]：已清除 {got.get('cleared', 0)} 次输错密码的记录。")


@admin.command("gpus")
def admin_gpus_cmd(as_json: bool = typer.Option(False, "--json", help="以 JSON 格式输出每张显卡的 uuid、model、vram_gb、arch、authorized（供 GPU 测试工具选择显卡）")) -> None:
    """显示本机正在运行的 lab2shot ui 检测到的显卡：型号、显存、架构、是否接受任务（使用本机令牌，无需密码登录）。"""
    import json

    from rich.table import Table

    from ..client import Lab2ShotError

    try:
        cards = local_client().admin_cards()["cards"]
    except Lab2ShotError as exc:
        failed(exc)
    rows = [{"uuid": c["uuid"], "model": c["model"], "vram_gb": c["memory_gb"], "arch": c["arch"], "authorized": c["authorized"]}
            for c in cards]
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return
    table = Table("UUID", "显卡", "显存", "架构", "")
    for r in rows:
        table.add_row(r["uuid"], r["model"], f"{r['vram_gb']:g} GB", r["arch"], "[green]接受任务[/green]" if r["authorized"] else "[dim]不接受任务[/dim]")
    console.print(table)


@admin.command("authorize")
def admin_authorize_cmd(
    uuid: Optional[list[str]] = typer.Argument(None, help="接受任务的显卡 UUID（可通过 lab2shot admin gpus 查看；可指定多个，整体替换原有授权）"),
    none: bool = typer.Option(False, "--none", help="不授权任何显卡（计算中的任务继续完成，此后不再接受任务）"),
) -> None:
    """按 UUID 授权本机正在运行的 lab2shot ui 使用显卡接受任务（使用本机令牌，无需密码登录）。所列即全部授权，未列出的显卡取消授权。"""
    from ..client import Lab2ShotError

    uuids = [] if none else list(uuid or [])
    if not uuids and not none:
        failed("必须指定要授权的显卡 UUID（可通过 lab2shot admin gpus 查看）；如不授权任何显卡，请使用 --none")
    try:
        view = local_client().admin_authorize(uuids)
    except Lab2ShotError as exc:
        failed(exc)
    taking = [g["uuid"] for g in view["gpus"] if g.get("authorized", True)]
    console.print(f"[green]已授权 {len(uuids)} 张显卡[/green]" + (f"：{'、'.join(uuids)}" if uuids else "：所有显卡均不接受任务") +
                  (f"（服务当前接受任务的显卡共 {len(taking)} 张）" if taking else ""))


@admin.command("switches")
def admin_switches_cmd(
    gpu: Optional[bool] = typer.Option(None, "--gpu/--no-gpu", help="显卡任务开关（queue.gpu_jobs）：关闭后显卡任务仅排队，不开始计算"),
    compute: Optional[bool] = typer.Option(None, "--compute/--no-compute", help="计算任务开关（queue.compute_jobs）：关闭后不再接新的计算任务"),
) -> None:
    """显卡任务与计算任务两个开关（使用本机令牌）。不带参数时仅显示当前状态。正在计算的任务不受影响，会继续运行至完成。"""
    from ..client import Lab2ShotError

    try:
        view = local_client().admin_switches(gpu, compute) if (gpu is not None or compute is not None) else local_client().admin_queue()
    except Lab2ShotError as exc:
        failed(exc)
    s = view["switches"]
    console.print(f"显卡任务  {'[green]开启[/green]' if s['gpu'] else '[yellow]关闭[/yellow]'}\n"
                  f"计算任务  {'[green]开启[/green]' if s['compute'] else '[yellow]关闭[/yellow]'}")


@admin.command("queue")
def admin_queue_cmd() -> None:
    """检查队列是否空闲（协调重启前以此确认，不应通过密码登录调用 /api/admin/queue）：空闲时退出码为 0；忙碌时列出进行中的任务，退出码为 3。"""
    from ..client import Lab2ShotError

    try:
        q = local_client().admin_queue()
    except Lab2ShotError as exc:
        failed(exc)
    active = [j for j in q["jobs"] if j["state"] in ("queued", "running")]
    if not active:
        console.print("空闲：队列中没有排队中或计算中的任务")
        return
    for j in active:
        console.print(f"{j['state']:<8} {j.get('title') or j['id']}")
    raise typer.Exit(3)


@admin.command("joblog")
def admin_joblog_cmd(
    code: str = typer.Argument(..., help="报错信息中的八位出错编号（由使用者提供）"),
    lines: int = typer.Option(80, "--lines", "-n", help="输出日志末尾的行数；0 表示全部"),
) -> None:
    """按出错编号查询一次计算的完整记录（节点、参数、模型、显存、完整堆栈）。

    根据报错中的八位编号，在缓存中查找该任务的目录（`<编号>*_job`），打印 job.json 中的
    节点、扩展、时间、帧数、参数以及 worker.log 的末尾；同一编号匹配多个目录时取最新的一次。未找到时退出码为 4。
    """
    import json

    from ..data.store import current

    if not code.isalnum():  # an error number is hexadecimal: nothing but letters and digits goes into the glob pattern
        console.print(f"[yellow]编号格式不对：{code}[/yellow]")
        raise typer.Exit(4)
    store = current()  # every account's own cache (data/store.py): the administrator looks through all of them
    hits = sorted((d for uid in store.cached_accounts() for d in store.cache_of(uid).glob(f"{code}*_job")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        console.print(f"[yellow]未找到编号 {code}[/yellow]：缓存中没有该任务（可能已被清除，或编号有误）")
        raise typer.Exit(4)
    job_dir = hits[0]
    if len(hits) > 1:
        console.print(f"[yellow]编号 {code} 匹配到 {len(hits)} 次任务，使用最新的一次：{job_dir.name}[/yellow]")
    info = job_dir / "job.json"
    if info.is_file():
        try:
            j = json.loads(info.read_text(encoding="utf-8"))
            node = j.get("node")
            console.print(f"节点  {node.get('label') if isinstance(node, dict) else node}\n"
                          f"扩展  {j.get('extension')}\n"
                          f"时间  {j.get('created')}\n"
                          f"帧数  {len(j.get('frames') or [])}\n"
                          f"参数  {json.dumps(j.get('params', {}), ensure_ascii=False)}\n")
        except (ValueError, OSError):
            pass
    log = job_dir / "worker.log"
    if not log.is_file():
        console.print(f"[yellow]该任务没有日志文件[/yellow]（{log}）")
        raise typer.Exit(4)
    text = log.read_text(encoding="utf-8", errors="replace").splitlines()
    shown = text if lines <= 0 else text[-lines:]
    if len(shown) < len(text):
        console.print(f"[dim]（仅显示最后 {len(shown)} 行，共 {len(text)} 行；如需显示全部，请使用 --lines 0）[/dim]")
    console.print("\n".join(shown))
    console.print(f"\n[dim]完整日志：{log}[/dim]")


@admin.command("feedback")
def admin_feedback_cmd(
    clear: bool = typer.Option(False, "--clear", help="删除全部反馈历史（包括截图和诊断文件）"),
    yes: bool = typer.Option(False, "--yes", "-y", help="删除前不再确认"),
) -> None:
    """查看反馈历史，或使用 `--clear` 全部清除。

    后台页面上每条反馈都有删除按钮，但需要密码登录，而一个账号同一时间只能在一处登录，密码登录会使管理员浏览器中的
    登录失效。此命令直接调用 feedback 模块，不涉及登录。

    删除范围为数据库 `feedback` 表中的行，以及 `work/feedback/<编号>/` 中的截图和诊断文件，不涉及其他数据。
    数据库备份中的副本会保留到备份轮换为止。
    """
    from .. import feedback as fb

    rows = fb.listing()["items"]
    if not clear:
        if not rows:
            console.print("反馈历史为空")
            return
        console.print(f"反馈历史 {len(rows)} 条：")
        for r in rows:
            when_ = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["at"]))
            console.print(f"  {r['id']}  {when_}  {r["status_label"]:<6}  {(r.get('text') or '')[:44]}")
        console.print("\n[dim]如需全部删除：uv run lab2shot admin feedback --clear[/dim]")
        return

    # 表为空时也须继续执行：孤儿文件夹即「表中没有、磁盘上仍存在」的文件夹，在此提前返回会遗漏它们
    if not rows:
        console.print("反馈历史已为空")
        _sweep_orphans()
        return
    if not yes and not typer.confirm(f"确认删除 {len(rows)} 条反馈及其截图？此操作不可撤销"):
        console.print("已取消，未删除任何内容")
        raise typer.Exit(1)
    from ..accounts import Actor
    from ..server.access import audit

    for r in rows:
        fb.delete(r["id"], by=Actor(None, "命令行"))
    audit(Msg("I-AUDIT-FEEDBACKCLEARED", count=len(rows)))
    console.print(f"已删除 {len(rows)} 条反馈")
    _sweep_orphans()


@admin.command("jobs")
def admin_jobs_cmd(
    user: str = typer.Option("", "--user", help="仅处理此账号的任务（默认：管理员本人）"),
    clear: bool = typer.Option(False, "--clear", help="删除全部已结束的任务历史（并释放其占用的空间）"),
    yes: bool = typer.Option(False, "--yes", "-y", help="删除前不再确认"),
) -> None:
    """查看队列历史，或使用 `--clear` 清除所有已结束的任务。

    队列面板上有「删除全部」按钮，但需要密码登录，而一个账号同一时间只能在一处登录，密码登录会使管理员浏览器中的
    登录失效。此命令使用本机令牌，通过本机正在运行的服务操作（哪些任务还在排队或计算只有服务知道），不涉及登录；
    服务没有运行时不做任何删除。

    删除路径与页面按钮相同（`server/farm.py forget_finished`：`quota.drop_for_job` + `farm.forget_job`），因此规则一致：
      · 排队中或计算中的任务不删除（取消后才视为「已结束」），结果中说明跳过的条数；
      · 每个任务的文件夹（节点图、素材、输出、日志）整个删除；只被它引用的缓存由随后的清理带走
        （farm/disk.py：没有别的任务引用、也没有任务在用的才删）。
    """
    from ..accounts import listing as accounts_listing
    from ..client import Lab2ShotError

    people = {str(u["username"]): int(u["id"]) for u in accounts_listing()}
    name = user or accounts_admin_name()
    if name not in people:
        console.print(f"账号不存在：{name}（现有账号：{'、'.join(sorted(people))}）")
        raise typer.Exit(1)
    uid = people[name]
    lab = local_client()
    try:
        rows = lab.admin_history(uid)
    except Lab2ShotError as exc:
        failed(exc)
    active = ("queued", "running")
    if not clear:
        if not rows:
            console.print(f"{name} 的队列历史为空")
            return
        console.print(f"{name} 的队列历史 {len(rows)} 条（排队或计算中 {sum(1 for r in rows if r.get('state') in active)} 条）：")
        for r in rows[:30]:
            when_ = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("finished") or r.get("submitted") or 0))
            console.print(f"  {str(r.get('id') or ''):<14}{when_}  {str(r.get('state') or ''):<10}{str(r.get('title') or '')[:40]}")
        if len(rows) > 30:
            console.print(f"  …… 另有 {len(rows) - 30} 条")
        console.print("\n[dim]如需全部删除：uv run lab2shot admin jobs --clear[/dim]")
        return

    ended = sum(1 for r in rows if r.get("id") and r.get("state") not in active)
    if not ended:
        console.print("没有可删除的已结束任务（排队中或计算中的任务须先取消）")
        return
    if not yes and not typer.confirm(f"确认删除 {name} 的 {ended} 条已结束任务并释放其占用的空间？此操作不可撤销"):
        console.print("已取消，未删除任何内容")
        raise typer.Exit(1)
    try:
        done = lab.admin_forget_finished(uid)
    except Lab2ShotError as exc:
        failed(exc)
    skipped = done.get("skipped") or 0
    console.print(f"已删除 {done['jobs']} 条任务，释放 {done['bytes'] / 1e6:.1f} MB" + (f"；跳过 {skipped} 条（排队中或计算中）" if skipped else ""))


def _sweep_orphans() -> None:
    """一并清除磁盘上存在、但数据库中没有对应行的文件夹。

    更换数据库后，`work/feedback/` 中会留下没有对应记录的文件夹。它们不会报错，不易察觉，持续占用空间，
    且包含诊断文件（可能含有当时的日志片段）。因此「清空反馈历史」同时清除数据库的行和磁盘上的文件夹。
    """
    import shutil

    from .. import feedback as fb
    from ..config import settings

    root = settings().work_dir / "feedback"  # fb.folder() 会先查询该反馈是否存在，而此处处理的正是不存在的情况
    if not root.is_dir():
        return
    alive = {r["id"] for r in fb.listing()["items"]}
    gone = [d for d in root.iterdir() if d.is_dir() and d.name not in alive]
    if not gone:
        return
    size = sum(f.stat().st_size for d in gone for f in d.rglob("*") if f.is_file())
    for d in gone:
        shutil.rmtree(d, ignore_errors=True)
    console.print(f"另已清除 {len(gone)} 个没有对应记录的文件夹（{size / (1 << 20):.1f} MB）："
                  "数据库中已无相应记录，但文件夹仍残留在磁盘上")
