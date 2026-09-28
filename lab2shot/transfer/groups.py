"""Task groups: which of an account's tasks belong together, so the queue and the task history show a few groups
instead of every task.

A task's group is worked out once, when it is submitted, from the graph it was submitted with, and kept on its row
(tasks.group_key, group_name, group_slot); nothing about it changes afterwards. Groups are each account's own: the
key is made with the account's id in it and every query that reads a group names the account, so two accounts with the
same footage have two groups and neither sees the other's.

    with footage     the tasks whose graph reads the same footage: the content (sha256 of every file) of the upload
                     each input node reads. Changed content, an input node added or removed: another group. The frame
                     range, the reading parameters (colour space ...), every other node and parameter and which template
                     it came from do not count: only what the input nodes read. The group's name is the main footage's
                     name as the user picked it (the folder's name of a sequence, the video's or file's name:
                     `sh030_plate`, `dance.mp4`; uploads.picked_name) for the first file parameter of the first input
                     node, in the graph's node order; with several uploads read, how many: 「sh030_plate 等 2 个素材」.
                     Two of an account's groups that still have the same name are told apart only as they are shown,
                     by the time of each one's first task (of_tasks `twin`, `first`); nothing stored changes.
    without footage  one account, one template, one fixed two-hour slot of the server's local clock (0-2, 2-4 ... 22-24:
                     twelve a day). The template is the one the graph was opened from (meta.template); a graph that
                     came from none is known by its name. The group's name is the graph's name; the page adds the slot.

An input node is a node one of whose parameters names an upload (upload:<id>/<name>, transfer/uploads.py refs_in): a
node with its file not picked yet reads nothing and does not count. There is no starting a group by hand: it would bring
complexity for nothing.

A group can be renamed: its user renames their own, an administrator anyone's (`rename`). The name is kept per account
and group key (task_group_names), apart from the tasks: the grouping (the key) never changes, the tasks submitted into
the group later are shown by that name too, and clearing it shows the automatic name again (the tasks' own group_name
is never touched). A renamed group is shown by exactly that name, never with a time added. The name goes with the
group: when its last task is removed (transfer/tasks.py remove), so is the name, and the same footage coming back later
starts with the automatic name.

A group's name comes from what a user typed or picked (a file's name, a graph's name, a rename): it is data only. It is
cleaned here once (clean_name: plain text, at most NAME_MOST characters), kept in the database by bound parameters,
shown as plain text, and never part of a path: nothing on disk is named after a group."""

from __future__ import annotations

import hashlib
import json
import time

from ..text import plain_text

NAME_MOST = 64  # characters
SLOT_HOURS = 2  # a group of tasks without footage: one fixed slot of the clock (a group every 2 hours, 12 a day)
KEY_LEN = 24  # hexadecimal characters of the key: the id the page names a group by


def clean_name(text: object) -> str:
    """A name as a group keeps and shows it: plain text (text.py plain_text), at most NAME_MOST characters."""
    return plain_text(text, NAME_MOST)


def footage_name(name: object, count: int) -> str:
    """The name of a group with footage: the main footage's name, and with several uploads read how many
    (「sh030_plate 等 2 个素材」), cleaned and at most NAME_MOST characters all told (the name is cut, never the count)."""
    if count < 2:
        return clean_name(name)
    more = f" 等 {count} 个素材"
    return (clean_name(name)[:NAME_MOST - len(more)].strip() + more).strip()


def is_key(value: object) -> bool:
    """Whether `value` has the form of a group key (it reaches a query only as a bound parameter all the same)."""
    return isinstance(value, str) and len(value) == KEY_LEN and all(c in "0123456789abcdef" for c in value)


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()[:KEY_LEN]


def _refs(value: object, out: list[str]) -> list[str]:
    """The upload references in `value`, in the order its parameters come (the first one is the main footage)."""
    from . import uploads

    if isinstance(value, str):
        if uploads.is_ref(value):
            out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _refs(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _refs(v, out)
    return out


def slot_of(at: float) -> float:
    """The start of the fixed two-hour slot of the server's local clock that `at` falls in (0:00, 2:00 ... 22:00)."""
    t = time.localtime(at)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour - t.tm_hour % SLOT_HOURS, 0, 0, 0, 0, -1))


def of_graph(data: dict, user_id: int, at: float) -> dict:
    """The group of a task of account `user_id` submitted at `at` with graph `data`: {key, name, slot} (slot: the start
    of its clock slot for a task without footage, None with footage)."""
    from . import uploads

    nodes = [n for n in data.get("nodes") or [] if isinstance(n, dict)]
    inputs = []  # one entry per input node: what it reads
    name = ""
    footage: set[str] = set()  # the uploads read, each once (the count in 「<name> 等 N 个素材」)
    for n in nodes:
        sids = [p for r in _refs(n.get("params"), []) if (p := uploads.ref_parts(r)) is not None]
        if not sids:
            continue
        reads = []
        for sid, rest in sids:
            files = uploads.files_of(sid, user_id)
            # the content, never the name: the same bytes picked again (or from another folder) are the same footage.
            # An upload this account does not have (cleaned; never its own) is known by its id: its node fails anyway
            reads.append(sorted(files.values()) if files is not None else [f"?{sid}"])
            footage.add(sid)
            if not name:
                # what the user picked it as (the folder's name, the video's name), else what the node reads
                name = uploads.picked_name(sid, user_id) or rest.rsplit("/", 1)[-1] or (min(files) if files else "")
        inputs.append(sorted(reads))
    if inputs:
        return {"key": _key("footage", int(user_id), sorted(inputs)), "name": footage_name(name, len(footage)), "slot": None}
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    template = str(meta.get("template") or "")
    title = str(meta.get("name") or "")
    slot = slot_of(at)
    return {"key": _key("slot", int(user_id), template or f"graph:{title}", slot), "name": clean_name(title), "slot": slot}


# ------------------------------------------------------------------ the groups an account has


def of_tasks(ids) -> dict[str, dict]:
    """task id -> its group {key, name, slot, count, first, twin, renamed} (the group's name: the one it was renamed to,
    else its first task's; renamed: whether it was; count: how many tasks its account has in it now, shown or not;
    first: when its first task was submitted; twin: another of the same account's groups with footage is shown by the
    same name and this one was not renamed, so the page tells them apart by `first` (「sh030_plate · 9月28日 14:05」:
    only as it is shown, nothing stored changes)), for the ids given that are tasks."""
    from ..database import db

    ids = [i for i in ids if isinstance(i, str)]
    if not ids:
        return {}
    member: dict[str, tuple[int, str]] = {}
    for at in range(0, len(ids), 500):  # SQLite's bound-parameter limit
        part = ids[at:at + 500]
        for r in db().rows(f"SELECT id, user_id, group_key FROM tasks WHERE id IN ({','.join('?' * len(part))}) AND group_key != ''",
                           tuple(part)):
            member[r["id"]] = (int(r["user_id"]), r["group_key"])
    named: dict[tuple[int, str], dict] = {}
    for user in {u for u, _ in member.values()}:
        # every group of the account (the twins are among all of them, not only the ones shown): its tasks are bounded
        # by 任务保留天数, and the oldest first, so a group's first row is its first task
        given = {r["group_key"]: r["name"] for r in db().rows("SELECT group_key, name FROM task_group_names WHERE user_id = ?", (user,))}
        mine: dict[str, dict] = {}
        for r in db().rows("SELECT group_key, group_name, group_slot, created FROM tasks WHERE user_id = ? AND group_key != '' "
                           "ORDER BY created", (user,)):
            g = mine.get(r["group_key"])
            if g is None:
                mine[r["group_key"]] = {"key": r["group_key"], "name": given.get(r["group_key"], r["group_name"]),
                                        "slot": r["group_slot"], "count": 1, "first": r["created"], "twin": False,
                                        "renamed": r["group_key"] in given}
            else:
                g["count"] += 1
        seen: dict[str, list[dict]] = {}
        for g in mine.values():
            if g["slot"] is None and g["name"]:  # a group without footage is told apart by its slot already
                seen.setdefault(g["name"], []).append(g)
        for same in seen.values():
            if len(same) > 1:
                for g in same:
                    g["twin"] = not g["renamed"]  # a name its user chose is shown as it is
        named.update({(user, k): g for k, g in mine.items()})
    return {i: named[m] for i, m in member.items() if m in named}


def tasks_of(user_id: int, key: str) -> list[str]:
    """The ids of the tasks of account `user_id` in group `key` (none for another account's group, or no such group)."""
    from ..database import db

    if not is_key(key):
        return []
    return [r["id"] for r in db().rows("SELECT id FROM tasks WHERE user_id = ? AND group_key = ? ORDER BY created DESC",
                                       (int(user_id), key))]


def rename(user_id: int, key: str, name: object) -> str:
    """Rename group `key` of account `user_id`: the name is cleaned (clean_name) and kept for the account and
    the key; an empty one shows the automatic name again. Another account's group, or none, is not there (NotFound, the
    same answer as none: the query names the account). Returns the name the group is shown by now."""
    from ..database import db
    from ..errors import NotFound
    from ..messages import Msg

    ids = tasks_of(user_id, key)
    if not ids:
        raise NotFound(Msg("E-TASKGROUP-GONE"))
    clean = clean_name(name)
    with db().write() as c:
        if clean:
            c.execute("INSERT INTO task_group_names (user_id, group_key, name, renamed) VALUES (?, ?, ?, ?) "
                      "ON CONFLICT (user_id, group_key) DO UPDATE SET name = excluded.name, renamed = excluded.renamed",
                      (int(user_id), key, clean, time.time()))
        else:
            c.execute("DELETE FROM task_group_names WHERE user_id = ? AND group_key = ?", (int(user_id), key))
    return of_tasks(ids[:1])[ids[0]]["name"]
