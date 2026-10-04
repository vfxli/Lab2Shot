"""The admin page's 数据库 section: the database's state (file, size, integrity check, backups) and the two actions
on it (back up now, check now). The state itself comes from lab2shot/database/__init__.py's `status()`."""

from __future__ import annotations

from .routes import Access, Router
from .. import logs
from ..messages import Msg
from ..database import db

log = logs.get("admin")


admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])  # this module's admin routes (app.py includes it)


@admin.get("/db", access=Access.admin("db.backup_restore"), summary="Database: file, size, version, last integrity check, last backup, backups kept")
def admin_db() -> dict:
    return db().status()


@admin.post("/db/backup", access=Access.admin("db.backup_restore"), summary="Back up the database now (online backup, the service keeps running)")
def admin_db_backup() -> dict:
    made = db().backup("manual")
    logs.say(log, Msg("I-DB-ADMINBACKUP", name=made.name))
    return admin_db()


@admin.post("/db/check", access=Access.admin("db.backup_restore"), summary="Check the database's integrity now")
def admin_db_check() -> dict:
    found = db().check()
    logs.say(log, Msg("I-DB-ADMINCHECK", detail=found["detail"]))
    return admin_db()
