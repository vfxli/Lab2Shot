"""The database's schema: one baseline migration, and whatever comes after it. A database knows its version (PRAGMA
user_version); the server brings an older one up to date at start, after a backup, one migration per transaction.

The baseline makes the whole schema: every compute request is a task with a folder of its own, and caches and uploads
are each account's own. A version this code does not know (above VERSION) is refused as it is (E-DB-NEWER): nothing is
ever guessed about it.

A task's graph is not kept in the database: it is a file in the task's folder (transfer/tasks.py graph.json); the
tasks table says whose it is and when it ended.

A migration already released is never changed in substance: an installation that ran it keeps what it made, and a
change is a new migration at the end."""

from __future__ import annotations

BASELINE_SQL = """
    CREATE TABLE meta (                   -- small state: usage statistics' start, GPUs taking jobs, ...
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL               -- JSON
    );

    -- accounts (lab2shot/accounts.py); a deleted one stays for the statistics
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        hash TEXT NOT NULL,               -- scrypt of the password ('' until the administrator's default is made)
        name TEXT NOT NULL,               -- Chinese name
        department TEXT NOT NULL,
        tags TEXT NOT NULL,               -- JSON: the tags of what it may use (nodes/tags.py)
        expires REAL,                     -- NULL: never (the administrator)
        enabled INTEGER NOT NULL,
        created REAL NOT NULL,
        last_login REAL,
        password_set REAL,
        password_by TEXT NOT NULL,
        deleted REAL,
        role TEXT NOT NULL DEFAULT 'user',  -- admin / deputy / user (lab2shot/roles.py ROLES)
        quota_gb REAL                     -- this account's disk quota; NULL: the setting's default (storage.quota_gb)
    );
    CREATE UNIQUE INDEX users_username ON users (username) WHERE deleted IS NULL;
    -- the built-in administrator
    INSERT INTO users (id, username, hash, name, department, tags, expires, enabled, created, password_set, password_by, role)
        VALUES (1, 'admin', '', '管理员', '', '[]', NULL, 1, unixepoch('subsec'), NULL, '未设', 'admin');
    CREATE TABLE sessions (               -- a browser's cookie or a client's token, as its sha256
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        kind TEXT NOT NULL,               -- web / client / machine
        created REAL NOT NULL,
        expires REAL NOT NULL,
        admin_until REAL NOT NULL,        -- the administrator's rights, from typing the password
        seen REAL NOT NULL,
        ip TEXT NOT NULL,
        agent TEXT NOT NULL,
        device_id TEXT NOT NULL DEFAULT '',   -- the browser's own random id (localStorage), or the client's (its config)
        device TEXT NOT NULL DEFAULT '',      -- a short label: "Chrome · Windows", or a client's app name
        hostname TEXT NOT NULL DEFAULT '',    -- a DCC plugin or the command line may report it; a browser cannot
        replaced_at REAL,                     -- a later login of the same account and kind ended this one
        replaced_by TEXT NOT NULL DEFAULT ''  -- JSON: {at, ip, device, kind} of the login that ended it
    );
    CREATE INDEX sessions_user ON sessions (user_id);
    CREATE INDEX sessions_replaced ON sessions (replaced_at);
    CREATE TABLE login_log (              -- every attempt (lab2shot/accounts.py log_login), for 最近登录
        id INTEGER PRIMARY KEY,
        at REAL NOT NULL,
        username TEXT NOT NULL,           -- as typed (a wrong username never says whether it exists back to the caller)
        user_id INTEGER REFERENCES users (id),  -- resolved, when it did
        ok INTEGER NOT NULL,
        reason TEXT NOT NULL,             -- '' on success; else why not (wrong password, expired, disabled, rate-limited)
        kind TEXT NOT NULL,               -- web / client
        ip TEXT NOT NULL,
        agent TEXT NOT NULL,
        device TEXT NOT NULL,
        device_id TEXT NOT NULL,
        hostname TEXT NOT NULL,
        ended TEXT NOT NULL               -- JSON: [{kind, device}, ...] of the sessions this login ended
    );
    CREATE INDEX login_log_user ON login_log (user_id, at);
    CREATE INDEX login_log_at ON login_log (at);
    CREATE TABLE admin_actions (          -- one administrator's action, written once and never changed
        id INTEGER PRIMARY KEY,
        at REAL NOT NULL,
        user_id INTEGER REFERENCES users (id),   -- who did it (NULL: not logged in — a refused attempt)
        role TEXT NOT NULL,               -- the role they had then (a role changes; this line does not)
        code TEXT NOT NULL,               -- the message code (I-AUDIT-USERCREATED, W-AUDIT-REFUSED ...)
        params TEXT NOT NULL,             -- JSON: the message's parameters, as the catalogue renders the line
        method TEXT NOT NULL,             -- the request it came from ('' none: a command line action)
        path TEXT NOT NULL,
        status TEXT NOT NULL              -- 成功 / 拒绝 (the message's own level says which)
    );
    CREATE INDEX admin_actions_user ON admin_actions (user_id, at);
    CREATE TABLE role_rights (            -- 一级管理员分配给一个角色的能力（现在只有 deputy 会有行）
        role TEXT PRIMARY KEY,            -- lab2shot/roles.py ASSIGNABLE；管理员永远全部、普通用户永远没有，都不在这里
        capabilities TEXT NOT NULL,       -- JSON 名字数组（roles.py CAPABILITIES 里的名字）
        updated REAL NOT NULL,            -- 最近一次分配的时间（权限的缓存跟着它走）
        updated_by TEXT NOT NULL          -- 谁分配的（当时的名字）；完整留底在 admin_actions
    );
    CREATE TABLE traffic (                -- 这个账号哪一天发出去了多少字节（server/traffic.py）
        user_id INTEGER NOT NULL,
        day TEXT NOT NULL,                -- 本地日期 YYYY-MM-DD
        bytes INTEGER NOT NULL,
        PRIMARY KEY (user_id, day)
    );
    CREATE TABLE consents (               -- licences accepted before a manual download is installed (extensions/manual.py accept)
        id INTEGER PRIMARY KEY,
        record TEXT NOT NULL              -- JSON
    );

    -- the farm: every job, how its nodes served it (usage statistics), node timings, the queue kept over a restart
    CREATE TABLE jobs (                   -- one row per job: submitted, then updated when it ends
        id TEXT PRIMARY KEY,
        submitted REAL NOT NULL,
        finished REAL,
        state TEXT NOT NULL,
        record TEXT NOT NULL,             -- JSON: the job's record (farm/queue.py Job.record)
        user_id INTEGER REFERENCES users (id)
    );
    CREATE INDEX jobs_finished ON jobs (finished);
    CREATE INDEX jobs_user ON jobs (user_id, submitted);
    CREATE TABLE job_usage (              -- how a job's nodes served it, per node type (usage statistics)
        job_id TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
        node_type TEXT NOT NULL,
        runs INTEGER NOT NULL,
        reuses INTEGER NOT NULL,
        seconds REAL NOT NULL,
        gpu_seconds REAL NOT NULL,
        frames INTEGER NOT NULL,
        PRIMARY KEY (job_id, node_type)
    );
    CREATE TABLE held (                   -- jobs waiting when the server restarted: queued again by the next one
        job_id TEXT PRIMARY KEY REFERENCES jobs (id) ON DELETE CASCADE,
        position INTEGER NOT NULL,
        targets TEXT NOT NULL,            -- JSON: node ids
        force INTEGER NOT NULL
    );
    CREATE TABLE timings (                -- how long a node computed (farm/timings.py)
        id INTEGER PRIMARY KEY,
        t REAL NOT NULL,
        node TEXT NOT NULL,
        gpu TEXT NOT NULL,
        setting TEXT NOT NULL,
        frames INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        seconds REAL NOT NULL
    );

    -- tasks: a compute request a user submitted, with its folder (transfer/tasks.py); kept 任务保留天数 after it ends
    CREATE TABLE tasks (
        id TEXT PRIMARY KEY,              -- the job's id: the folder's name <数据位置>/tasks/<id>
        user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        created REAL NOT NULL,
        ended REAL,                       -- NULL while it is queued or running
        -- its group (transfer/groups.py), worked out when it is submitted: by the footage its input nodes read, or
        -- without footage by template and two-hour slot of the clock
        group_key TEXT NOT NULL DEFAULT '',   -- the account's group (transfer/groups.py of_graph)
        group_name TEXT NOT NULL DEFAULT '',  -- data only: cleaned, shown as text, never a path
        group_slot REAL                       -- a task without footage: its two-hour slot's start
    );
    CREATE INDEX tasks_user ON tasks (user_id, created);
    CREATE INDEX tasks_ended ON tasks (ended);
    CREATE INDEX tasks_group ON tasks (user_id, group_key, created);
    -- a group renamed by its user or an administrator (transfer/groups.py rename): the name it is shown by, per account
    -- and group key, so the tasks submitted into the group later are shown by it too. The grouping (the key) never
    -- changes; the tasks keep their automatic name (group_name), shown again when the name is cleared. The row goes
    -- with the group's last task (transfer/tasks.py remove)
    CREATE TABLE task_group_names (
        user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        group_key TEXT NOT NULL,          -- tasks.group_key
        name TEXT NOT NULL,               -- data only: cleaned (groups.clean_name), shown as text, never a path
        renamed REAL NOT NULL,
        PRIMARY KEY (user_id, group_key)
    ) WITHOUT ROWID;
    CREATE TABLE task_cache (             -- the cache entries a task computed or reused (in its account's cache root):
        task_id TEXT NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,  -- an entry no live task names is cleaned
        name TEXT NOT NULL,               -- a packet fingerprint, or a worker job's `<key>_job`
        PRIMARY KEY (task_id, name)
    ) WITHOUT ROWID;
    CREATE INDEX task_cache_name ON task_cache (name);
    CREATE TABLE task_uploads (           -- the uploads a task's graph reads (its footage/<id>): kept while it lives
        task_id TEXT NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
        upload TEXT NOT NULL,             -- the upload's id (upload:<id>/...)
        PRIMARY KEY (task_id, upload)
    ) WITHOUT ROWID;
    CREATE INDEX task_uploads_upload ON task_uploads (upload);

    -- users' feedback (screenshots are files in work/feedback/<id>/)
    CREATE TABLE feedback (
        id TEXT PRIMARY KEY,
        at REAL NOT NULL,
        category TEXT NOT NULL,           -- '' / error / usage / idea
        text TEXT NOT NULL,
        status TEXT NOT NULL,             -- new / seen / solved
        note TEXT NOT NULL,               -- the administrator's
        updated REAL,
        updated_by TEXT NOT NULL,
        files TEXT NOT NULL,              -- JSON: [{name, bytes, kind}]
        bytes INTEGER NOT NULL,
        reply TEXT NOT NULL DEFAULT '',   -- the administrator's answer: the user sees it
        replied REAL,
        replied_by TEXT NOT NULL DEFAULT '',
        changed REAL,                     -- when what the user sees of it (status, reply) last changed
        read_at REAL,                     -- when the user last looked at their feedback
        user_id INTEGER REFERENCES users (id)
    );
    CREATE INDEX feedback_at ON feedback (at);
    CREATE INDEX feedback_user ON feedback (user_id);
"""

# 自行注册（lab2shot/registration.py）：邀请码，以及每个自己注册的账号是怎么来的。注册出来的账号本身和管理员建的
# 一样，就在 users 里；这里只多记来路，给邀请码的「谁用它注册了」、全站和按 IP / 网段的注册上限、按邀请码或时间段
# 批量停用用。
REGISTRATION_SQL = """
    CREATE TABLE invites (
        id INTEGER PRIMARY KEY,
        code TEXT NOT NULL,               -- the code as it is handed out (the 邀请码 page shows and copies it)
        code_hash TEXT NOT NULL,          -- sha256 of registration.normal_code(code): what a typed code is looked up by
        hint TEXT NOT NULL,               -- its first characters: all that logs and audit lines ever name it by
        note TEXT NOT NULL,               -- 备注: plain text (text.py plain_text), data only
        uses_max INTEGER,                 -- 可用次数; NULL: no limit
        used INTEGER NOT NULL DEFAULT 0,  -- registrations made with it (claimed in the same transaction as the account)
        expires REAL,                     -- NULL: never
        enabled INTEGER NOT NULL,
        created REAL NOT NULL,
        created_by TEXT NOT NULL          -- who made it (their name then); the full record is in admin_actions
    );
    CREATE UNIQUE INDEX invites_code ON invites (code_hash);
    CREATE TABLE registrations (          -- an account made by registering itself (the account is an ordinary users row)
        user_id INTEGER PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
        at REAL NOT NULL,
        invite_id INTEGER REFERENCES invites (id) ON DELETE SET NULL,  -- NULL: without a code, or the code since deleted
        invite_hint TEXT NOT NULL,        -- the code's hint then ('' without one): still shown once the code is deleted
        ip TEXT NOT NULL,                 -- the client's address (server/auth.py who)
        net TEXT NOT NULL                 -- its /24 (IPv4) or /64 (IPv6), for the per-network limit
    );
    CREATE INDEX registrations_at ON registrations (at);
    CREATE INDEX registrations_ip ON registrations (ip, at);
    CREATE INDEX registrations_net ON registrations (net, at);
    CREATE INDEX registrations_invite ON registrations (invite_id);
"""

# 用户协议与隐私政策（lab2shot/terms）：两份文字的每一版原样留底，每个账号同意了哪一版、什么时候、从哪个地址。
# 一个版本号同时管两份文字；文字一变（程序自带的改了，或者管理员在后台改了）就是新的一版，账号要重新同意才能接着用。
TERMS_SQL = """
    CREATE TABLE terms_versions (         -- one row per version of the two texts, as they were then (never changed)
        version INTEGER PRIMARY KEY,      -- 1, 2, ...: a new one whenever the texts in effect change
        at REAL NOT NULL,                 -- when it came into effect
        by TEXT NOT NULL,                 -- who saved it (their name then); '' for the text that comes with the program
        digest TEXT NOT NULL,             -- sha256 of the two texts: how a change is noticed
        agreement TEXT NOT NULL,          -- 用户协议, as written (placeholders not filled in)
        privacy TEXT NOT NULL             -- 隐私政策, the same
    );
    CREATE TABLE terms_agreed (           -- an account agreed to a version: at registering, or at a login after a change
        user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
        version INTEGER NOT NULL REFERENCES terms_versions (version),
        at REAL NOT NULL,
        ip TEXT NOT NULL,                 -- the client's address then (server/auth.py client_source)
        PRIMARY KEY (user_id, version)
    ) WITHOUT ROWID;
"""

# 后台概览的「今天和最近」（server/settings.py admin_overview_recent）按时间段数任务、流量，按状态数未解决的反馈；
# 这几张表只增不减（任务记录、每个账号每天一行流量），不加索引每次都要整表扫一遍。
RECENT_SQL = """
    CREATE INDEX jobs_submitted ON jobs (submitted, state);
    CREATE INDEX traffic_day ON traffic (day, bytes);
    CREATE INDEX feedback_status ON feedback (status);
"""

# 按模板统计（farm/usage.py templates）：任务是从哪张模板打开的节点图提交的。template 是模板的 id（lab2shot/library.py
# card_id），"" 是自己搭的节点图；template_name 是提交时模板的名字，模板删掉以后按它显示。这一版之前提交的任务两列都是
# NULL：不补，也不计入。按账号、按时间段统计时走已有的 jobs_user、jobs_submitted 两个索引，不另加索引。
TEMPLATES_SQL = """
    ALTER TABLE jobs ADD COLUMN template TEXT;
    ALTER TABLE jobs ADD COLUMN template_name TEXT;
"""

# 「按模板」统计去掉了（节点图不记来自哪张模板，按模板数任务没有意义）：上面那一步加的两列删掉。
# 迁移只往后加，不改已发布的那一步（数据库按版本号一步一步升）。
NO_TEMPLATES_SQL = """
    ALTER TABLE jobs DROP COLUMN template;
    ALTER TABLE jobs DROP COLUMN template_name;
"""

# 节点用时（farm/timings.py）：每种节点只留最新的 KEPT_PER_NODE（200）条（写入时删掉更旧的），估计时只按用到的
# 节点类型查：加 (node, id) 索引，并把已有的表剪到同样的条数。
TIMINGS_SQL = """
    CREATE INDEX timings_node ON timings (node, id);
    DELETE FROM timings WHERE id IN (
        SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY node ORDER BY id DESC) AS k FROM timings) WHERE k > 200);
"""

# 记录里「谁做的」除了名字再记账号 id（accounts.py Actor）：永久删除账号时按 id 找到他做的那几行，只改那几行的名字
# （PURGE named），不再按文字去猜——两个人同名、名字是别的字的一部分时，别人的行不会被改。许可同意（consents）也记
# 账号 id，已有的行从记录里的 who.user 补上。
ACTORS_SQL = """
    ALTER TABLE invites ADD COLUMN created_by_id INTEGER;
    ALTER TABLE terms_versions ADD COLUMN by_id INTEGER;
    ALTER TABLE role_rights ADD COLUMN updated_by_id INTEGER;
    ALTER TABLE feedback ADD COLUMN updated_by_id INTEGER;
    ALTER TABLE feedback ADD COLUMN replied_by_id INTEGER;
    ALTER TABLE consents ADD COLUMN user_id INTEGER;
    UPDATE consents SET user_id = json_extract(record, '$.who.user') WHERE json_valid(record);
"""

# 上一步只加了列：升级上来的旧行 *_by_id 都是空的，永久删除按 id 找不到它们。这里按行里写的名字补上 id，只认能确定
# 是哪一个账号的写法：「名（用户名）」整串、用户名整串，或者全库只有一个账号叫这个中文名；认不出的保持空（宁可留名，
# 不误改别人）。管理操作记录加上重复次数和最后一次的时间：同一个人在同一处被同样拒绝，记在一行上（access.py audit）。
ACTOR_IDS_SQL = """
    UPDATE invites SET created_by_id = (SELECT id FROM (
        SELECT id, 1 AS k FROM users WHERE name || '（' || username || '）' = invites.created_by
        UNION ALL SELECT id, 2 FROM users WHERE username = invites.created_by
        UNION ALL SELECT MIN(id), 3 FROM users WHERE name = invites.created_by HAVING COUNT(*) = 1) ORDER BY k LIMIT 1) WHERE created_by_id IS NULL;
    UPDATE role_rights SET updated_by_id = (SELECT id FROM (
        SELECT id, 1 AS k FROM users WHERE name || '（' || username || '）' = role_rights.updated_by
        UNION ALL SELECT id, 2 FROM users WHERE username = role_rights.updated_by
        UNION ALL SELECT MIN(id), 3 FROM users WHERE name = role_rights.updated_by HAVING COUNT(*) = 1) ORDER BY k LIMIT 1) WHERE updated_by_id IS NULL;
    UPDATE terms_versions SET by_id = (SELECT id FROM (
        SELECT id, 1 AS k FROM users WHERE name || '（' || username || '）' = terms_versions.by
        UNION ALL SELECT id, 2 FROM users WHERE username = terms_versions.by
        UNION ALL SELECT MIN(id), 3 FROM users WHERE name = terms_versions.by HAVING COUNT(*) = 1) ORDER BY k LIMIT 1) WHERE by_id IS NULL;
    UPDATE feedback SET updated_by_id = (SELECT id FROM (
        SELECT id, 1 AS k FROM users WHERE name || '（' || username || '）' = feedback.updated_by
        UNION ALL SELECT id, 2 FROM users WHERE username = feedback.updated_by
        UNION ALL SELECT MIN(id), 3 FROM users WHERE name = feedback.updated_by HAVING COUNT(*) = 1) ORDER BY k LIMIT 1) WHERE updated_by_id IS NULL;
    UPDATE feedback SET replied_by_id = (SELECT id FROM (
        SELECT id, 1 AS k FROM users WHERE name || '（' || username || '）' = feedback.replied_by
        UNION ALL SELECT id, 2 FROM users WHERE username = feedback.replied_by
        UNION ALL SELECT MIN(id), 3 FROM users WHERE name = feedback.replied_by HAVING COUNT(*) = 1) ORDER BY k LIMIT 1) WHERE replied_by_id IS NULL;
    UPDATE meta SET value = json_set(value, '$.by_id', (SELECT id FROM (
        SELECT id, 1 AS k FROM users WHERE name || '（' || username || '）' = json_extract(meta.value, '$.by')
        UNION ALL SELECT id, 2 FROM users WHERE username = json_extract(meta.value, '$.by')
        UNION ALL SELECT MIN(id), 3 FROM users WHERE name = json_extract(meta.value, '$.by') HAVING COUNT(*) = 1) ORDER BY k LIMIT 1))
        WHERE key = 'server.notice' AND json_valid(value) AND json_extract(value, '$.by_id') IS NULL;
    ALTER TABLE admin_actions ADD COLUMN repeats INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE admin_actions ADD COLUMN last REAL;
"""

# 管理操作记录记下「对谁做的」账号 id（access.audit about）：永久删除按 id 找到关于这个人的行，不再按参数里各条消息
# 叫法不同的键（username、owner……）去猜。已有的行按参数里写的用户名补上（用户名整串）。
TARGETS_SQL = """
    ALTER TABLE admin_actions ADD COLUMN target_id INTEGER;
    UPDATE admin_actions SET target_id = (SELECT id FROM users WHERE username = json_extract(admin_actions.params, '$.username')
                                              OR username = json_extract(admin_actions.params, '$.owner'))
        WHERE json_valid(params);
"""

MIGRATIONS: list[tuple[str, str]] = [
    ("基线：账号与登录、任务（每个任务一个文件夹）与任务分组、任务↔缓存、任务↔素材、用时与使用统计、许可协议记录、用户反馈", BASELINE_SQL),
    ("自行注册：邀请码，和每个自己注册的账号的来路", REGISTRATION_SQL),
    ("用户协议与隐私政策：每一版的原文，和每个账号同意了哪一版", TERMS_SQL),
    ("概览统计用的索引：任务按提交时间、流量按日期、反馈按状态", RECENT_SQL),
    ("按模板统计：任务记下是从哪张模板提交的", TEMPLATES_SQL),
    ("去掉按模板统计：任务不再记模板", NO_TEMPLATES_SQL),
    ("节点用时：按节点类型查的索引，每种节点只留最新的 200 条", TIMINGS_SQL),
    ("记录里谁做的也记账号 id；许可同意记账号 id", ACTORS_SQL),
    ("旧行按名字补上账号 id；管理操作记录记重复次数", ACTOR_IDS_SQL),
    ("管理操作记录记下对谁做的账号 id", TARGETS_SQL),
]

FIRST = 1  # the version the baseline makes
VERSION = FIRST + len(MIGRATIONS) - 1


def pending(version: int) -> list[tuple[int, str, str]]:
    """The migrations a database at `version` still needs, in order: (the version each one makes, what it is, its SQL).
    A new database (0) needs every one."""
    start = 0 if version == 0 else version - FIRST + 1
    return [(FIRST + n, what, sql) for n, (what, sql) in enumerate(MIGRATIONS) if n >= start]
