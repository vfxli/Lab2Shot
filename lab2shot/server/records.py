"""The admin page's 数据库 section: the database's state (file, size, integrity check, backups) and the two actions
on it (back up now, check now). The state itself comes from lab2shot/database/__init__.py's `status()`."""

from __future__ import annotations

from .routes import Access, Router
from .. import logs
from ..messages import Msg
from ..database import db

log = logs.get("admin")


admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # this module's admin routes (app.py includes it)


@admin.get("/db", access=Access.admin("db.backup_restore"), summary="数据库：文件、大小、版本、上次完整性检查、上次备份、留着的备份")
def admin_db() -> dict:
    return db().status()


@admin.post("/db/backup", access=Access.admin("db.backup_restore"), summary="立即备份数据库（在线备份，服务照常运行）")
def admin_db_backup() -> dict:
    made = db().backup("manual")
    logs.say(log, Msg("I-DB-ADMINBACKUP", name=made.name))
    return admin_db()


@admin.post("/db/check", access=Access.admin("db.backup_restore"), summary="现在检查一遍数据库的完整性")
def admin_db_check() -> dict:
    found = db().check()
    logs.say(log, Msg("I-DB-ADMINCHECK", detail=found["detail"]))
    return admin_db()
