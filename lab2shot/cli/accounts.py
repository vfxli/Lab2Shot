"""`lab2shot login | logout` (an account on a Lab2Shot server) and `lab2shot admin` (this machine's administrator
password, passphrase and login sessions, and the running service managed with the machine token)."""

from __future__ import annotations

import time
from typing import Optional

import typer

from .. import i18n
from ..messages import Msg
from .base import app, console, failed, group, when

admin = group(i18n.t("cli.admin.help"))


def accounts_admin_name() -> str:
    from .. import accounts

    return accounts.admin().username


@app.command(help=i18n.t("cli.accounts.login.help"))
def login(
    server: str = typer.Option(..., "--server", help=i18n.t("cli.accounts.option.server_example")),
    username: Optional[str] = typer.Argument(None, help=i18n.t("cli.accounts.login.username")),
) -> None:
    from ..client import Lab2Shot, Lab2ShotError, tokens_file

    name = username or typer.prompt(i18n.t("cli.accounts.login.prompt_username"))
    try:
        user = Lab2Shot(server, app="cli").login(name, typer.prompt(i18n.t("cli.accounts.login.prompt_password"), hide_input=True))
    except Lab2ShotError as exc:
        failed(exc)
    console.print(i18n.t("cli.accounts.login.done", name=user["name"], username=user["username"],
                         department=user["department"] or i18n.t("cli.accounts.login.no_department"), file=tokens_file()))


@app.command(help=i18n.t("cli.accounts.logout.help"))
def logout(server: str = typer.Option(..., "--server", help=i18n.t("cli.accounts.option.server"))) -> None:
    from ..client import Lab2Shot

    Lab2Shot(server, app="cli").logout()
    console.print(i18n.t("cli.accounts.logout.done"))


def _new_secret(what: str) -> str:
    from .. import accounts

    while True:
        text = typer.prompt(i18n.t("cli.admin.secret.new", what=what), hide_input=True,
                            confirmation_prompt=i18n.t("cli.admin.secret.again", what=what))
        if not (problem := accounts.rule_problem(text, what)):
            return text
        console.print(f"[red]{problem}[/red]")


@admin.command("password", help=i18n.t("cli.admin.password.help"))
def admin_password() -> None:
    from .. import accounts
    from ..server.access import audit

    accounts.set_password(accounts.ADMIN_ID, _new_secret(i18n.t("cli.admin.password.what")), accounts.BY_COMMAND_LINE)
    audit(Msg("I-LOGIN-CLIRESET"))  # an administrator's action from the command line: no session, no request
    console.print(i18n.t("cli.admin.password.done", username=accounts.admin().username))


@admin.command("passphrase", help=i18n.t("cli.admin.passphrase.help"))
def admin_passphrase() -> None:
    from .. import accounts
    from ..server.access import audit

    had = accounts.passphrase() is not None
    accounts.set_passphrase(_new_secret(i18n.t("cli.admin.passphrase.what")))
    audit(Msg("I-LOGIN-CLIPASSPHRASECHANGED" if had else "I-LOGIN-CLIPASSPHRASESET"))
    console.print(i18n.t("cli.admin.passphrase.updated" if had else "cli.admin.passphrase.set"))


@admin.command("status", help=i18n.t("cli.admin.status.help"))
def admin_status() -> None:
    from .. import accounts

    owner, phrase = accounts.admin(), accounts.passphrase()
    if owner.no_password:
        console.print(i18n.t("cli.admin.status.no_password", username=owner.username))
    else:
        console.print(i18n.t("cli.admin.status.password", username=owner.username, when=when(owner.password_set or 0),
                             by=accounts.password_by_text(owner.password_by)))
    console.print(i18n.t("cli.admin.status.no_passphrase") if phrase is None else
                  i18n.t("cli.admin.status.passphrase", when=when(phrase["set"])))
    users = [u for u in accounts.listing() if not u["deleted"]]
    # 在线是运行中的服务在内存里记的（accounts.presence）：命令行这个进程看不到，只指向后台页面
    console.print(i18n.t("cli.admin.status.accounts", count=len(users)))


@admin.command("logout", help=i18n.t("cli.admin.logout.help"))
def admin_logout() -> None:
    from .. import accounts
    from ..server.access import audit

    n = accounts.end_all()
    audit(Msg("I-LOGIN-CLIREVOKED", count=n))
    console.print(i18n.t("cli.admin.logout.done", count=n))


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


@admin.command("restart", help=i18n.t("cli.admin.restart.help"))
def admin_restart_cmd(
    mode: str = typer.Option("drain", "--mode", help=i18n.t("cli.admin.restart.mode")),
    wait: bool = typer.Option(False, "--wait", help=i18n.t("cli.admin.restart.wait")),
) -> None:
    from ..client import Lab2ShotError

    lab = local_client()
    try:
        before = lab.server_state()
        lab.admin_restart(mode)
    except Lab2ShotError as exc:
        failed(exc)
    console.print(i18n.t("cli.admin.restart.asked", mode=mode, boot=before["boot"]))
    if not wait:
        return
    console.print(i18n.t("cli.admin.restart.waiting"))
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            now = lab.server_state()
        except Lab2ShotError:
            continue
        if now["boot"] != before["boot"]:
            console.print(i18n.t("cli.admin.restart.done", boot=now["boot"]))
            return
    failed(i18n.t("cli.admin.restart.timeout"))


@admin.command("unlock", help=i18n.t("cli.admin.unlock.help"))
def admin_unlock_cmd() -> None:
    from ..client import Lab2ShotError

    try:
        got = local_client().admin_unlock()
    except Lab2ShotError as exc:
        failed(exc)
    console.print(i18n.t("cli.admin.unlock.done", count=got.get("cleared", 0)))


@admin.command("gpus", help=i18n.t("cli.admin.gpus.help"))
def admin_gpus_cmd(as_json: bool = typer.Option(False, "--json", help=i18n.t("cli.admin.gpus.json"))) -> None:
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
    table = Table("UUID", i18n.t("cli.admin.gpus.card"), i18n.t("cli.admin.gpus.memory"), i18n.t("cli.admin.gpus.arch"), "")
    for r in rows:
        table.add_row(r["uuid"], r["model"], f"{r['vram_gb']:g} GB", r["arch"], i18n.t("cli.admin.gpus.taking") if r["authorized"] else i18n.t("cli.admin.gpus.not_taking"))
    console.print(table)


@admin.command("authorize", help=i18n.t("cli.admin.authorize.help"))
def admin_authorize_cmd(
    uuid: Optional[list[str]] = typer.Argument(None, help=i18n.t("cli.admin.authorize.uuid")),
    none: bool = typer.Option(False, "--none", help=i18n.t("cli.admin.authorize.none")),
) -> None:
    from ..client import Lab2ShotError

    uuids = [] if none else list(uuid or [])
    if not uuids and not none:
        failed(i18n.t("cli.admin.authorize.need_uuid"))
    try:
        view = local_client().admin_authorize(uuids)
    except Lab2ShotError as exc:
        failed(exc)
    taking = [g["uuid"] for g in view["gpus"] if g.get("authorized", True)]
    console.print(i18n.t("cli.admin.authorize.done", count=len(uuids))
                  + (i18n.t("cli.admin.authorize.listed", uuids=i18n.separator().join(uuids)) if uuids
                     else i18n.t("cli.admin.authorize.none_taking"))
                  + (i18n.t("cli.admin.authorize.taking", count=len(taking)) if taking else ""))


@admin.command("switches", help=i18n.t("cli.admin.switches.help"))
def admin_switches_cmd(
    gpu: Optional[bool] = typer.Option(None, "--gpu/--no-gpu", help=i18n.t("cli.admin.switches.gpu")),
    compute: Optional[bool] = typer.Option(None, "--compute/--no-compute", help=i18n.t("cli.admin.switches.compute")),
) -> None:
    from ..client import Lab2ShotError

    try:
        view = local_client().admin_switches(gpu, compute) if (gpu is not None or compute is not None) else local_client().admin_queue()
    except Lab2ShotError as exc:
        failed(exc)
    s = view["switches"]
    on, off = i18n.t("cli.admin.switches.on"), i18n.t("cli.admin.switches.off")
    console.print(i18n.t("cli.admin.switches.gpu_state", state=on if s["gpu"] else off) + "\n"
                  + i18n.t("cli.admin.switches.compute_state", state=on if s["compute"] else off))


@admin.command("queue", help=i18n.t("cli.admin.queue.help"))
def admin_queue_cmd() -> None:
    from ..client import Lab2ShotError

    try:
        q = local_client().admin_queue()
    except Lab2ShotError as exc:
        failed(exc)
    active = [j for j in q["jobs"] if j["state"] in ("queued", "running")]
    if not active:
        console.print(i18n.t("cli.admin.queue.idle"))
        return
    for j in active:
        console.print(f"{j['state']:<8} {j.get('title') or j['id']}")
    raise typer.Exit(3)


@admin.command("joblog", help=i18n.t("cli.admin.joblog.help"))
def admin_joblog_cmd(
    code: str = typer.Argument(..., help=i18n.t("cli.admin.joblog.code")),
    lines: int = typer.Option(80, "--lines", "-n", help=i18n.t("cli.admin.joblog.lines")),
) -> None:
    import json

    from ..data.store import current

    if not code.isalnum():  # an error number is hexadecimal: nothing but letters and digits goes into the glob pattern
        console.print(i18n.t("cli.admin.joblog.bad_code", code=code))
        raise typer.Exit(4)
    store = current()  # every account's own cache (data/store.py): the administrator looks through all of them
    hits = sorted((d for uid in store.cached_accounts() for d in store.cache_of(uid).glob(f"{code}*_job")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        console.print(i18n.t("cli.admin.joblog.not_found", code=code))
        raise typer.Exit(4)
    job_dir = hits[0]
    if len(hits) > 1:
        console.print(i18n.t("cli.admin.joblog.several", code=code, count=len(hits), folder=job_dir.name))
    info = job_dir / "job.json"
    if info.is_file():
        try:
            j = json.loads(info.read_text(encoding="utf-8"))
            node = j.get("node")
            console.print(i18n.t("cli.admin.joblog.record", node=node.get("label") if isinstance(node, dict) else node,
                                 extension=j.get("extension"), created=j.get("created"),
                                 frames=len(j.get("frames") or []),
                                 params=json.dumps(j.get("params", {}), ensure_ascii=False)) + "\n")
        except (ValueError, OSError):
            pass
    log = job_dir / "worker.log"
    if not log.is_file():
        console.print(i18n.t("cli.admin.joblog.no_log", file=log))
        raise typer.Exit(4)
    text = log.read_text(encoding="utf-8", errors="replace").splitlines()
    shown = text if lines <= 0 else text[-lines:]
    if len(shown) < len(text):
        console.print(i18n.t("cli.admin.joblog.tail", shown=len(shown), total=len(text)))
    console.print("\n".join(shown))
    console.print("\n" + i18n.t("cli.admin.joblog.full", file=log))


@admin.command("feedback", help=i18n.t("cli.admin.feedback.help"))
def admin_feedback_cmd(
    clear: bool = typer.Option(False, "--clear", help=i18n.t("cli.admin.feedback.clear")),
    yes: bool = typer.Option(False, "--yes", "-y", help=i18n.t("cli.admin.yes")),
) -> None:
    """The feedback history, or with --clear all of it removed: the rows of the database's `feedback` table and the
    screenshots and diagnostics in `work/feedback/<id>/`, nothing else (copies in database backups stay until rotated).
    The admin page's delete buttons need a password login, which would end the administrator's browser session: this
    calls the feedback module directly."""
    from ..site import feedback as fb

    rows = fb.listing()["items"]
    if not clear:
        if not rows:
            console.print(i18n.t("cli.admin.feedback.empty"))
            return
        console.print(i18n.t("cli.admin.feedback.count", count=len(rows)))
        for r in rows:
            when_ = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["at"]))
            console.print(f"  {r['id']}  {when_}  {r["status_label"]:<6}  {(r.get('text') or '')[:44]}")
        console.print("\n" + i18n.t("cli.admin.feedback.how_to_clear"))
        return

    # 表为空时也须继续执行：孤儿文件夹即「表中没有、磁盘上仍存在」的文件夹，在此提前返回会遗漏它们
    if not rows:
        console.print(i18n.t("cli.admin.feedback.already_empty"))
        _sweep_orphans()
        return
    if not yes and not typer.confirm(i18n.t("cli.admin.feedback.confirm", count=len(rows))):
        console.print(i18n.t("cli.admin.cancelled"))
        raise typer.Exit(1)
    from ..accounts import Actor
    from ..server.access import audit

    for r in rows:
        fb.delete(r["id"], by=Actor(None, i18n.t("cli.admin.by_command_line", in_lang=i18n.LOG_LANG)))  # the log's (English)
    audit(Msg("I-AUDIT-FEEDBACKCLEARED", count=len(rows)))
    console.print(i18n.t("cli.admin.feedback.done", count=len(rows)))
    _sweep_orphans()


@admin.command("jobs", help=i18n.t("cli.admin.jobs.help"))
def admin_jobs_cmd(
    user: str = typer.Option("", "--user", help=i18n.t("cli.admin.jobs.user")),
    clear: bool = typer.Option(False, "--clear", help=i18n.t("cli.admin.jobs.clear")),
    yes: bool = typer.Option(False, "--yes", "-y", help=i18n.t("cli.admin.yes")),
) -> None:
    """The queue history, or with --clear every finished job removed, through the running service with the machine
    token (only it knows what is still queued or running; nothing is removed when it is not running). The same path as
    the queue panel's button (server/farm.py forget_finished: quota.drop_for_job + farm.forget_job): queued or running
    jobs are skipped (and counted), each job's folder goes whole, caches only it referenced go with the next sweep."""
    from ..accounts import listing as accounts_listing
    from ..client import Lab2ShotError

    people = {str(u["username"]): int(u["id"]) for u in accounts_listing()}
    name = user or accounts_admin_name()
    if name not in people:
        console.print(i18n.t("cli.admin.jobs.no_account", name=name, names=i18n.separator().join(sorted(people))))
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
            console.print(i18n.t("cli.admin.jobs.empty", name=name))
            return
        console.print(i18n.t("cli.admin.jobs.count", name=name, count=len(rows),
                             active=sum(1 for r in rows if r.get("state") in active)))
        for r in rows[:30]:
            when_ = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("finished") or r.get("submitted") or 0))
            console.print(f"  {str(r.get('id') or ''):<14}{when_}  {str(r.get('state') or ''):<10}{str(r.get('title') or '')[:40]}")
        if len(rows) > 30:
            console.print("  " + i18n.t("cli.admin.jobs.more", count=len(rows) - 30))
        console.print("\n" + i18n.t("cli.admin.jobs.how_to_clear"))
        return

    ended = sum(1 for r in rows if r.get("id") and r.get("state") not in active)
    if not ended:
        console.print(i18n.t("cli.admin.jobs.nothing"))
        return
    if not yes and not typer.confirm(i18n.t("cli.admin.jobs.confirm", name=name, count=ended)):
        console.print(i18n.t("cli.admin.cancelled"))
        raise typer.Exit(1)
    try:
        done = lab.admin_forget_finished(uid)
    except Lab2ShotError as exc:
        failed(exc)
    skipped = done.get("skipped") or 0
    console.print(i18n.t("cli.admin.jobs.done", count=done["jobs"], mb=done["bytes"] / 1e6)
                  + (i18n.t("cli.admin.jobs.skipped", count=skipped) if skipped else ""))


def _sweep_orphans() -> None:
    """Remove the folders of work/feedback/ that have no row in the database (left behind after a database was
    replaced: silent, taking space, holding diagnostics with log excerpts), so clearing the feedback history clears both."""
    import shutil

    from ..site import feedback as fb
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
    console.print(i18n.t("cli.admin.feedback.orphans", count=len(gone), mb=size / (1 << 20)))
