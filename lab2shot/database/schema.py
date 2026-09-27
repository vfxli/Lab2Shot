"""The database's schema, one explicit migration per version. A database knows its version (PRAGMA user_version); the
server brings an older one up to date at start, after a backup, one migration per transaction.

Versions start at FIRST (5): the first migration makes the whole schema of version 5 at once (the records, users'
feedback, user accounts). Versions 1-4 are the history the main branch walked before accounts; installations that ran
them are at 5 and go on from there. A database still at 1-4 is refused and left as it is, never guessed at
(E-DB-TOOOLD). No version is an empty placeholder kept only for its number.

The benchmark tables created by the third migration belong to a feature that has been removed; a later migration
drops them.

A job's graph is not kept in the database: it is a file in the submitting account's folder
(work/users/<username>/jobs/<job id>.json, lab2shot/farm/queue.py), and the jobs table keeps only its path
(graph_file). The migration that made this change drops the former graph column without carrying its contents over.

A migration already released is never changed in substance: an installation that ran it keeps what it made, and a
change is a new migration at the end. Its SQL text may be simplified, as long as a new database ends up exactly where an
installation that walked the real history is (the same tables, columns and indexes)."""

from __future__ import annotations

MIGRATIONS: list[tuple[str, str]] = [
    ("任务记录、用时记录、结果记录、许可协议记录、队列状态；用户反馈；用户账号：登录、各人只看自己的，旧记录归管理员", """
        CREATE TABLE meta (                   -- small state: usage statistics' start, GPUs taking jobs, the import
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL               -- JSON
        );
        CREATE TABLE jobs (                   -- one row per job: submitted, then updated when it ends
            id TEXT PRIMARY KEY,
            submitted REAL NOT NULL,
            finished REAL,
            state TEXT NOT NULL,
            record TEXT NOT NULL,             -- JSON: the job's record (farm/queue.py Job.record)
            graph TEXT                        -- JSON: the graph it was submitted with
        );
        CREATE INDEX jobs_finished ON jobs (finished);
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
        CREATE TABLE deliveries (             -- what 「输出」 delivered and what became of it (the files stay on disk)
            run TEXT NOT NULL,
            node TEXT NOT NULL,
            created REAL NOT NULL,
            client TEXT NOT NULL,
            record TEXT NOT NULL,             -- JSON (transfer/deliveries.py)
            PRIMARY KEY (run, node)
        );
        CREATE TABLE consents (               -- licences users accepted on the help page (extensions/manual.py)
            id INTEGER PRIMARY KEY,
            record TEXT NOT NULL              -- JSON
        );

        -- users' feedback (main's versions 3 and 4 made it, by way of columns version 5 dropped again: built here
        -- straight in the shape version 5 left it)
        CREATE TABLE feedback (
            id TEXT PRIMARY KEY,              -- and screenshots are files in work/feedback/<id>/
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
            read_at REAL                      -- when the user last looked at their feedback
        );
        CREATE INDEX feedback_at ON feedback (at);

        -- user accounts: logging in, each account seeing its own; the records from before accounts are the administrator's
        CREATE TABLE users (                  -- accounts (lab2shot/accounts.py); a deleted one stays for the statistics
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            hash TEXT NOT NULL,               -- scrypt of the password ('' until the administrator's default is made)
            name TEXT NOT NULL,               -- Chinese name
            department TEXT NOT NULL,
            admin INTEGER NOT NULL,
            tags TEXT NOT NULL,               -- JSON: the tags of what it may use (nodes/tags.py)
            expires REAL,                     -- NULL: never (the administrator)
            enabled INTEGER NOT NULL,
            created REAL NOT NULL,
            last_login REAL,
            password_set REAL,
            password_by TEXT NOT NULL,
            deleted REAL
        );
        CREATE UNIQUE INDEX users_username ON users (username) WHERE deleted IS NULL;
        -- the administrator first: the old 管理密码 carries over (none: the default is made when first needed)
        INSERT INTO users (id, username, hash, name, department, admin, tags, expires, enabled, created, password_set, password_by)
            VALUES (1, 'admin', COALESCE((SELECT json_extract(value, '$.hash') FROM meta WHERE key = 'auth.password'), ''),
                    '管理员', '', 1, '[]', NULL, 1, unixepoch('subsec'),
                    (SELECT json_extract(value, '$.set') FROM meta WHERE key = 'auth.password'),
                    COALESCE((SELECT CASE WHEN json_extract(value, '$.default') THEN '默认' ELSE '管理员' END
                              FROM meta WHERE key = 'auth.password'), '未设'));
        CREATE TABLE sessions (               -- a browser's cookie or a client's token, as its sha256
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            kind TEXT NOT NULL,               -- web / token
            created REAL NOT NULL,
            expires REAL NOT NULL,
            admin_until REAL NOT NULL,        -- the administrator's rights, from typing the password
            seen REAL NOT NULL,
            ip TEXT NOT NULL,
            agent TEXT NOT NULL
        );
        CREATE INDEX sessions_user ON sessions (user_id);
        CREATE TABLE uploads (                -- who uploaded what: only they may use it (transfer/uploads.py)
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            key TEXT NOT NULL,                -- a blob's sha256, or an upload's id (upload:<id>/...)
            at REAL NOT NULL,
            PRIMARY KEY (user_id, key)
        ) WITHOUT ROWID;
        CREATE TABLE grants (                 -- the results the server handed a user from a graph of theirs (server/access.py)
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            fp TEXT NOT NULL,
            at REAL NOT NULL,
            PRIMARY KEY (user_id, fp)
        ) WITHOUT ROWID;
        CREATE INDEX grants_at ON grants (at);

        ALTER TABLE jobs ADD COLUMN user_id INTEGER REFERENCES users (id);
        UPDATE jobs SET user_id = 1;
        CREATE INDEX jobs_user ON jobs (user_id, submitted);

        ALTER TABLE deliveries ADD COLUMN user_id INTEGER REFERENCES users (id);
        UPDATE deliveries SET user_id = 1;
        ALTER TABLE deliveries DROP COLUMN client;
        CREATE INDEX deliveries_user ON deliveries (user_id, created);

        ALTER TABLE feedback ADD COLUMN user_id INTEGER REFERENCES users (id);
        UPDATE feedback SET user_id = 1;
        CREATE INDEX feedback_user ON feedback (user_id);

        DELETE FROM meta WHERE key IN ('auth.password', 'auth.access', 'auth.invites', 'auth.sessions');
    """),
    ("单点登录、设备识别、登录日志（一处只能在线一次，浏览器和 DCC 插件各算一处）", """
        ALTER TABLE sessions ADD COLUMN device_id TEXT NOT NULL DEFAULT '';   -- the browser's own random id (localStorage), or the client's (its config)
        ALTER TABLE sessions ADD COLUMN device TEXT NOT NULL DEFAULT '';     -- a short label: "Chrome · Windows", or a client's app name
        ALTER TABLE sessions ADD COLUMN hostname TEXT NOT NULL DEFAULT '';   -- a DCC plugin or the command line may report it; a browser cannot
        ALTER TABLE sessions ADD COLUMN replaced_at REAL;                    -- a later login of the same account and kind ended this one
        ALTER TABLE sessions ADD COLUMN replaced_by TEXT NOT NULL DEFAULT '';-- JSON: {at, ip, device, kind} of the login that ended it
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
    """),
    ("基准测试管理：素材、真值、审核（主管或自动审核）、修改历史、评估运行（农场任务的运行、项、结果）", """
        CREATE TABLE bench_samples (          -- one benchmark sample: a stretch of frames of one shot (lab2shot/bench/registry.py)
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT NOT NULL,             -- dataset / upload
            dataset TEXT NOT NULL,            -- the catalog's dataset id ('' for an upload)
            sequence TEXT NOT NULL,           -- the catalog's sequence id ('' for an upload)
            still TEXT NOT NULL,              -- a still-image set's picture ('' otherwise)
            media TEXT NOT NULL,              -- where the plate is: input/bench/<path>, or work/bench/samples/<id>
            first INTEGER NOT NULL,           -- the frame range, raw frame numbers, both included: never implicit
            last INTEGER NOT NULL,
            fps REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            pixel_aspect REAL NOT NULL,
            licence TEXT NOT NULL,
            commercial INTEGER NOT NULL,
            notes_first TEXT NOT NULL,        -- what whoever added it saw on the first, middle and last frame
            notes_middle TEXT NOT NULL,
            notes_last TEXT NOT NULL,
            notes TEXT NOT NULL,              -- warnings, what it can and cannot test
            origin TEXT NOT NULL,             -- where it came from
            templates TEXT NOT NULL,          -- JSON: the templates it is test material for (kind 'template')
            plate_state TEXT NOT NULL,        -- raw (as shot, lens distortion in it) / undistorted / unknown
            plate_lens TEXT NOT NULL,         -- JSON: how an undistorted plate was undistorted {truth: a distortion truth id, note}
            content TEXT NOT NULL,            -- JSON: what is in the frames (camera motion,
                                              -- people count / full body / moving, face size, alpha source, key interval): the kind rules read it
            plate_kind TEXT NOT NULL,         -- sequence / video / still / skeletal / asset (a 3D file: no pictures, no player)
            frame_flags TEXT NOT NULL,        -- JSON: {near_duplicate: [frame, ...], subject_absent: [[first, last], ...]}, set by QA or by
                                              -- hand; the evaluation skips or expects them (bench/measure.py kept); a change bumps the version
            seed TEXT UNIQUE,                 -- its key in lab2shot/bench/seeds/ (NULL: added by hand)
            version INTEGER NOT NULL,         -- +1 on every change of plate, range or truth: approvals name it
            checks TEXT NOT NULL,             -- JSON: the automatic QA's findings inside the range (lab2shot/datasets/qa.py; supporting only)
            motion TEXT NOT NULL,             -- JSON: motion measured from the truth (supporting only)
            checked REAL,
            added_by TEXT NOT NULL,
            created REAL NOT NULL,
            updated REAL NOT NULL,
            deleted REAL
        );
        CREATE TABLE bench_truths (           -- one piece of ground truth of a sample, with where it came from
            id INTEGER PRIMARY KEY,
            sample_id INTEGER NOT NULL REFERENCES bench_samples (id) ON DELETE CASCADE,
            kind TEXT NOT NULL,               -- camera / lens / filmback / pixel_aspect / distortion / people / hands / ...
            source TEXT NOT NULL,             -- dataset / file / typed
            data TEXT NOT NULL,               -- JSON: the typed values, or what was read
            file TEXT NOT NULL,               -- the stored file (work/bench/truths/<id>/...) or the dataset's gt/ path
            precision TEXT NOT NULL,          -- high / normal / rough / dataset
            made_by TEXT NOT NULL,
            method TEXT NOT NULL,
            made_on TEXT NOT NULL,            -- YYYY-MM-DD ('' unknown)
            tool TEXT NOT NULL,               -- the software and its version (3DEqualizer4 R7.1, Nuke 15.1 ...)
            qc_status TEXT NOT NULL,          -- '' not QC'd / passed / failed: a production track QC'd is the reference, error 0
            qc_by TEXT NOT NULL,
            qc_on TEXT NOT NULL,
            notes TEXT NOT NULL,
            added_by TEXT NOT NULL,
            created REAL NOT NULL,
            updated REAL NOT NULL
        );
        CREATE INDEX bench_truths_sample ON bench_truths (sample_id);
        CREATE TABLE bench_uses (             -- a sample proposed for a benchmark kind, and the owner's 「使用」 tick
            sample_id INTEGER NOT NULL REFERENCES bench_samples (id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            approved INTEGER NOT NULL,
            approved_by TEXT NOT NULL,
            approved_at REAL,
            approved_version INTEGER,         -- the sample's version the tick was given to
            approved_kind TEXT NOT NULL,      -- owner (the 「使用」 tick on the page) / agent (automated, by the administrator's authorization 2026-09-14)
            approved_note TEXT NOT NULL,      -- what the approver looked at and why it fits (required for agent)
            reset_reason TEXT NOT NULL,       -- why a tick was last taken back by a change
            reset_at REAL,
            created REAL NOT NULL,
            PRIMARY KEY (sample_id, kind)
        ) WITHOUT ROWID;
        CREATE TABLE bench_history (          -- every change of a sample, its truth and its ticks
            id INTEGER PRIMARY KEY,
            sample_id INTEGER NOT NULL REFERENCES bench_samples (id) ON DELETE CASCADE,
            at REAL NOT NULL,
            who TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT NOT NULL              -- JSON
        );
        CREATE INDEX bench_history_sample ON bench_history (sample_id, at);
        CREATE TABLE bench_runs (             -- one 「运行」: the items it planned, each a farm job (lab2shot/bench/jobs.py)
            id INTEGER PRIMARY KEY,
            kinds TEXT NOT NULL,              -- JSON: the benchmark kinds asked for ('template': templates and combination graphs)
            scope TEXT NOT NULL,              -- JSON: the samples asked for ([]: every approved one)
            trigger TEXT NOT NULL,            -- manual / stale (「重跑过期的」) / schedule
            requested_by TEXT NOT NULL,       -- the account's label when it was asked
            user_id INTEGER NOT NULL,         -- the account its jobs belong to (queued again after a restart as it)
            created REAL NOT NULL,
            finished REAL,
            state TEXT NOT NULL,              -- running / done / cancelled / failed (failed: some item failed)
            code TEXT NOT NULL,               -- the git revision it was planned on
            note TEXT NOT NULL
        );
        CREATE TABLE bench_items (            -- one thing a run cooks: a case (all its variants) or a graph, on one sample
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES bench_runs (id) ON DELETE CASCADE,
            what TEXT NOT NULL,               -- case / graph
            ref TEXT NOT NULL,                -- the case id, or the graph's id (a template's or a combination graph's)
            kind TEXT NOT NULL,               -- the benchmark kind it counts for ('template' for a graph)
            sample_id INTEGER NOT NULL REFERENCES bench_samples (id),
            sample_version INTEGER NOT NULL,
            stamps TEXT NOT NULL,             -- JSON: the approval stamps it was planned with, by kind
            first INTEGER NOT NULL,
            last INTEGER NOT NULL,
            variants TEXT NOT NULL,           -- JSON: [[variant, condition], ...] of a case ([] for a graph)
            graph_hash TEXT NOT NULL,         -- a graph's file content (sha256); '' for a case
            job_id TEXT NOT NULL,             -- the farm job ('' until queued)
            state TEXT NOT NULL,              -- queued / running / done / failed / cancelled / refused
            message TEXT NOT NULL,            -- JSON: why it failed or was refused ({code, params, text}; {} none)
            checks TEXT NOT NULL,             -- JSON: a graph's minimal expectations [{code, ok, detail}]
            code TEXT NOT NULL,               -- JSON: node versions and extension code identities of what it cooked
            env TEXT NOT NULL,                -- JSON: extension environment fingerprints
            seconds REAL NOT NULL,
            created REAL NOT NULL,
            finished REAL
        );
        CREATE INDEX bench_items_run ON bench_items (run_id);
        CREATE INDEX bench_items_ref ON bench_items (what, ref, sample_id, finished);
        CREATE TABLE bench_results (          -- one variant of one case on one sample
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL REFERENCES bench_items (id) ON DELETE CASCADE,
            run_id INTEGER NOT NULL REFERENCES bench_runs (id) ON DELETE CASCADE,
            case_id TEXT NOT NULL,
            node TEXT NOT NULL,
            port TEXT NOT NULL,
            variant TEXT NOT NULL,            -- what fed the input (a node type, or a truth feeder's name; '-' none)
            feeder TEXT NOT NULL,
            condition TEXT NOT NULL,          -- raw / undistorted_ref / undistorted_ml / solver_own (lab2shot/bench/cases.py)
            sample_id INTEGER NOT NULL REFERENCES bench_samples (id),
            sample_version INTEGER NOT NULL,
            kind TEXT NOT NULL,               -- the benchmark kind of the main output
            stamps TEXT NOT NULL,             -- JSON: the approval stamps it used, by kind
            first INTEGER NOT NULL,
            last INTEGER NOT NULL,
            metrics TEXT NOT NULL,            -- JSON {port: {metric: value}}
            dropped TEXT NOT NULL,            -- JSON: output groups not counted (their kind not approved)
            better TEXT NOT NULL,             -- JSON
            error TEXT NOT NULL,
            warnings TEXT NOT NULL,           -- JSON
            seconds REAL NOT NULL,
            code TEXT NOT NULL,               -- JSON: node versions and extension code identities
            env TEXT NOT NULL,                -- JSON: extension environment fingerprints
            metrics_code TEXT NOT NULL,
            packets TEXT NOT NULL,            -- JSON
            created REAL NOT NULL
        );
        CREATE INDEX bench_results_run ON bench_results (run_id);
        CREATE INDEX bench_results_case ON bench_results (case_id, sample_id, created);
    """),
    ("账号的角色：管理员、二级管理员、普通用户（lab2shot/roles.py）；原来的管理员还是管理员，别的账号是普通用户", """
        ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user';   -- admin / deputy / user (lab2shot/roles.py ROLES)
        UPDATE users SET role = 'admin' WHERE admin = 1;
        ALTER TABLE users DROP COLUMN admin;
    """),
    ("基准素材记下是哪个账号加的；管理操作留底（谁、对谁、做了什么、接口和结果）", """
        ALTER TABLE bench_samples ADD COLUMN user_id INTEGER REFERENCES users (id);  -- NULL: added before this, or by
        -- the command line / a seed (added_by keeps the name whoever added it had then, which never changes afterwards)
        ALTER TABLE bench_samples ADD COLUMN inputs TEXT NOT NULL DEFAULT '{}';  -- JSON: what a template needs besides the
        -- plate, declared by whoever chose this footage (lab2shot/bench/registry.py INPUTS): the frame a reference picture
        -- is aligned to, the frame a fix is painted on, the plane's quad, the HDRI to relight with. Named on the footage so
        -- a card's test never invents them.
        CREATE TABLE admin_actions (          -- one administrator's action, written once and never changed (设计 I.2)
            id INTEGER PRIMARY KEY,
            at REAL NOT NULL,
            user_id INTEGER REFERENCES users (id),   -- who did it (NULL: not logged in — a refused attempt)
            role TEXT NOT NULL,               -- the role they had then (a role changes; this line does not)
            code TEXT NOT NULL,              -- the message code (I-AUDIT-USERCREATED, W-AUDIT-REFUSED ...)
            params TEXT NOT NULL,            -- JSON: the message's parameters, as the catalogue renders the line
            method TEXT NOT NULL,            -- the request it came from ('' none: a command line action)
            path TEXT NOT NULL,
            status TEXT NOT NULL             -- 成功 / 拒绝 (the message's own level says which)
        );
        CREATE INDEX admin_actions_user ON admin_actions (user_id, at);
    """),
    ("「我的模板」和管理员录入的模板（同一套存储，用户自己删的进回收站）；分类树的管理员那一份；每个账号的磁盘配额", """
        CREATE TABLE saved_graphs (           -- 用户存到服务器上的节点图（server/library.py）：换台电脑登录还在
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,  -- 谁存的（录入成模板后仍记着）
            name TEXT NOT NULL,               -- 用户起的名字（用户写的字：一律按纯文本显示）
            intro TEXT NOT NULL,
            suits TEXT NOT NULL,              -- 适合（录入成模板时填）
            not_suits TEXT NOT NULL,
            graph TEXT NOT NULL,              -- JSON：节点图本身（素材不随模板保存，每次自己选）
            bytes INTEGER NOT NULL,           -- 它占多少（磁盘配额按这个算）
            listed INTEGER NOT NULL,          -- 0 只有自己看得到；1 管理员录入，所有人在模板面板里看得到
            deliverable TEXT NOT NULL,        -- 录入时归到哪个二级分类（''：未分类）
            created REAL NOT NULL,
            updated REAL NOT NULL,
            deleted REAL,                     -- 用户删掉的：进回收站，管理员可恢复或永久删除
            deleted_by TEXT NOT NULL          -- user / admin（''：没删）
        );
        CREATE INDEX saved_graphs_user ON saved_graphs (user_id, updated);
        CREATE TABLE template_categories (    -- 分类树管理员加的那一份（内置的随代码走，两者合成一棵：server/categories.py）
            id TEXT PRIMARY KEY,              -- 和内置分类同 id：改它的名字、说明、颜色、次序；别的 id：新分类
            parent TEXT NOT NULL,             -- '' 一级分类；否则它属于哪个一级分类
            label TEXT NOT NULL,
            tip TEXT NOT NULL,
            color TEXT NOT NULL,              -- '' 跟内置的或它的一级分类
            rank REAL NOT NULL,               -- 排序（小的在前）
            created REAL NOT NULL,
            updated REAL NOT NULL
        );
        ALTER TABLE users ADD COLUMN quota_gb REAL;  -- 这个账号的磁盘配额；NULL：用设置里的默认（storage.quota_gb）
    """),
    ("二级管理员的权限由一级管理员分配（lab2shot/roles.py）：不再写死在代码里，没有这一行就还是代码里的默认那一份", """
        CREATE TABLE role_rights (            -- 一级管理员分配给一个角色的能力（现在只有 deputy 会有行）
            role TEXT PRIMARY KEY,            -- lab2shot/roles.py ASSIGNABLE；管理员永远全部、普通用户永远没有，都不在这里
            capabilities TEXT NOT NULL,       -- JSON 名字数组（roles.py CAPABILITIES 里的名字）
            updated REAL NOT NULL,            -- 最近一次分配的时间（权限的缓存跟着它走）
            updated_by TEXT NOT NULL          -- 谁分配的（当时的名字）；完整留底在 admin_actions
        );
    """),
    ("每个账号的网络流量（按天记；server/traffic.py 在内存里累加，一分钟落一次盘）", """
        CREATE TABLE traffic (                -- 这个账号哪一天发出去了多少字节（后台「用户」那一栏、队列窗口的「我的占用」都读它）
            user_id INTEGER NOT NULL,
            day TEXT NOT NULL,                -- 本地日期 YYYY-MM-DD
            bytes INTEGER NOT NULL,
            PRIMARY KEY (user_id, day)
        );
    """),
    ("模板全部改成文件（lab2shot/library.py：templates/、adapters/<包>/templates/、work/users/<用户名>/templates/）：数据库里不再有模板", """
        DROP TABLE IF EXISTS saved_graphs;
        DROP TABLE IF EXISTS template_categories;
    """),
    ("删除已移除的基准测试功能遗留的数据表", """
        DROP TABLE IF EXISTS bench_results;
        DROP TABLE IF EXISTS bench_items;
        DROP TABLE IF EXISTS bench_runs;
        DROP TABLE IF EXISTS bench_history;
        DROP TABLE IF EXISTS bench_uses;
        DROP TABLE IF EXISTS bench_truths;
        DROP TABLE IF EXISTS bench_samples;
    """),
    ("任务提交时的节点图改为文件（work/users/<用户名>/jobs/<任务号>.json，lab2shot/farm/queue.py）：数据库只记文件路径，原有任务的节点图不保留", """
        ALTER TABLE jobs ADD COLUMN graph_file TEXT;  -- 提交时的节点图文件，相对工作目录的路径（NULL：没有节点图）
        ALTER TABLE jobs DROP COLUMN graph;
    """),
]

FIRST = 5  # version produced by the first migration (versions 1-4 predate accounts; see the module docstring)
VERSION = FIRST + len(MIGRATIONS) - 1


def pending(version: int) -> list[tuple[int, str, str]]:
    """The migrations a database at `version` still needs, in order: (the version each one makes, what it is, its SQL).
    A new database (0) needs every one; one before FIRST is not this code's to migrate."""
    if 0 < version < FIRST:
        raise ValueError(f"version {version} is older than the first migration's ({FIRST})")
    start = 0 if version == 0 else version - FIRST + 1
    return [(FIRST + n, what, sql) for n, (what, sql) in enumerate(MIGRATIONS) if n >= start]
